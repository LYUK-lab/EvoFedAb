#  implemented with reference to https://github.com/MediaBrain-SJTU/FedDisco

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from copy import deepcopy
from typing import Dict, List
import torch
from modules import ClientLossPool
from utils.config import resolve_train_cfg
from fl_algorithms.training import (
    aggregate_test_metrics,
    resolve_metric_compute,
    resolve_client_concurrency,
    apply_aggregation,
    clone_model,
    eval_local,
    train_client_thread,
    select_distribution_states,
    state_for_cid,
)


def _parse_server_update(server_cfg: dict) -> dict:
    disco_a = float(server_cfg.get("disco_a", server_cfg.get("a", 0.5)))
    disco_b = float(server_cfg.get("disco_b", server_cfg.get("b", 0.1)))
    if disco_a < 0.0:
        raise ValueError("server_update.disco_a must be >= 0 for feddisco.")
    return {"disco_a": disco_a, "disco_b": disco_b}


def _collect_client_records(train_loaders, cid: int) -> List[dict]:
    loader = train_loaders[cid]
    dataset = getattr(loader, "dataset", None)
    records = getattr(dataset, "records", None)
    if isinstance(records, list):
        return records
    return []


def _label_from_record(rec: dict) -> str | None:
    val = rec.get("meta_id")
    if val is None:
        val = rec.get("antigen_id")
    if val is None:
        return None
    return str(val)


def _compute_disco_weights(
    train_loaders, client_ids: List[int], client_samples: Dict[int, int], opts: dict
) -> tuple[Dict[int, float], Dict[int, float]]:
    disco_a = float(opts.get("disco_a", 0.5))
    disco_b = float(opts.get("disco_b", 0.1))
    hist_by_cid: Dict[int, Dict[str, int]] = {cid: {} for cid in client_ids}
    global_hist: Dict[str, int] = {}
    for cid in client_ids:
        for rec in _collect_client_records(train_loaders, cid):
            label = _label_from_record(rec)
            if label is None:
                continue
            hist = hist_by_cid[cid]
            hist[label] = hist.get(label, 0) + 1
            global_hist[label] = global_hist.get(label, 0) + 1
    if not global_hist:
        base = {
            cid: float(max(int(client_samples.get(cid, 0)), 0)) for cid in client_ids
        }
        total = sum(base.values())
        if total <= 0:
            return (
                {cid: 1.0 / max(len(client_ids), 1) for cid in client_ids},
                {cid: 0.0 for cid in client_ids},
            )
        return (
            {cid: w / total for cid, w in base.items()},
            {cid: 0.0 for cid in client_ids},
        )
    labels = sorted(global_hist.keys())
    global_total = float(sum(global_hist.values()))
    global_dist = {label: float(global_hist[label]) / global_total for label in labels}
    discrepancies_raw: Dict[int, float] = {}
    for cid in client_ids:
        hist = hist_by_cid.get(cid, {})
        n = float(sum(hist.values()))
        if n <= 0.0:
            discrepancies_raw[cid] = 0.0
            continue
        kl = 0.0
        for label in labels:
            local_prob = float(hist.get(label, 0)) / n
            global_prob = float(global_dist[label])
            if local_prob > 0.0 and global_prob > 0.0:
                kl += local_prob * math.log(local_prob / global_prob)
        discrepancies_raw[cid] = kl
    kl_sum = float(sum(discrepancies_raw.values()))
    if kl_sum > 0.0:
        discrepancies: Dict[int, float] = {
            cid: float(discrepancies_raw.get(cid, 0.0)) / kl_sum for cid in client_ids
        }
    else:
        discrepancies = {cid: 0.0 for cid in client_ids}
    base_weights: Dict[int, float] = {
        cid: float(max(int(client_samples.get(cid, 0)), 0)) for cid in client_ids
    }
    total_base = sum(base_weights.values())
    if total_base <= 0.0:
        base_weights = {cid: 1.0 / max(len(client_ids), 1) for cid in client_ids}
    else:
        base_weights = {cid: w / total_base for cid, w in base_weights.items()}
    raw_weights: Dict[int, float] = {}
    for cid in client_ids:
        d_k = float(discrepancies.get(cid, 0.0))
        raw_weights[cid] = float(base_weights.get(cid, 0.0)) - disco_a * d_k + disco_b
    if any((w > 0.0 for w in raw_weights.values())):
        raw_weights = {cid: max(w, 0.0) for cid, w in raw_weights.items()}
    else:
        raw_weights = dict(base_weights)
    total_raw = sum(raw_weights.values())
    if total_raw <= 0.0:
        return (
            {cid: 1.0 / max(len(client_ids), 1) for cid in client_ids},
            discrepancies,
        )
    return ({cid: w / total_raw for cid, w in raw_weights.items()}, discrepancies)


def train_feddisco(
    global_model,
    train_loaders,
    cfg,
    device,
    device_pool,
    aggregator,
    logger=None,
    test_loader=None,
):
    train_cfg = resolve_train_cfg(cfg)
    optim_cfg = train_cfg.get("client_optimizer")
    if optim_cfg is None or "loss" not in train_cfg:
        algo = cfg.get("fl_algorithm", "feddisco")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    server_cfg = train_cfg.get("server_update")
    if not isinstance(server_cfg, dict):
        raise KeyError("train config missing server_update for feddisco.")
    server_opts = _parse_server_update(server_cfg)
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    distribution_enabled = bool(cfg.get("distribution", True))
    if not distribution_enabled:
        raise ValueError("distribution=false is not supported for feddisco.")
    local_states = {}
    metric_mode = resolve_metric_compute(cfg)

    def _log_info(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    client_ids = sorted(getattr(train_loaders, "loaders", train_loaders).keys())
    device_pool = device_pool if device_pool else [device]
    device_by_cid: Dict[int, torch.device] = {
        cid: device_pool[idx % len(device_pool)] for idx, cid in enumerate(client_ids)
    }
    loader_map = getattr(train_loaders, "loaders", train_loaders)
    loader_items = (
        loader_map.values() if hasattr(loader_map, "values") else [loader_map]
    )
    has_worker_loaders = any(
        (getattr(loader, "num_workers", 0) > 0 for loader in loader_items)
    )
    use_thread_pool = resolve_client_concurrency(cfg, has_worker_loaders, client_ids)
    if len(client_ids) > 1 and has_worker_loaders and logger:
        logger.warning(
            "Detected DataLoader num_workers > 0 with multi-client threading. PyTorch can deadlock when spawning workers from threads, so training will run sequentially. Set data.num_workers=0 to enable parallel client training."
        )
    init_samples = {
        cid: int(train_loaders.client_counts.get(cid, 0)) for cid in client_ids
    }
    disco_weights, discrepancies = _compute_disco_weights(
        train_loaders, client_ids, init_samples, server_opts
    )
    if disco_weights:
        top_cid, top_w = max(disco_weights.items(), key=lambda kv: kv[1])
        avg_disc = sum(discrepancies.values()) / max(len(discrepancies), 1)
        _log_info(
            f"[feddisco] init weight by data-size/discrepancy label_key=meta_id disco_a={server_opts['disco_a']} disco_b={server_opts['disco_b']} avg_discrepancy={avg_disc:.6f} top=client{top_cid}:{top_w:.4f}"
        )
    for epoch in range(comm_rounds):
        base_state = deepcopy(global_model.state_dict())
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        client_samples: Dict[int, int] = {}
        train_loss_total: Dict[int, float] = {}
        train_loss_components: Dict[int, Dict[str, float]] = {}
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futures = []
                for pos, cid in enumerate(client_ids):
                    futures.append(
                        ex.submit(
                            train_client_thread,
                            cid,
                            global_model,
                            state_for_cid(train_state_by_cid, cid, base_state),
                            train_loaders[cid],
                            loss_pool,
                            optim_cfg,
                            local_epochs,
                            device_by_cid[cid],
                            clip_norm,
                            pos,
                        )
                    )
                for fut in as_completed(futures):
                    cid, state_dict, sample_total, loss_sum_total, comp_sum_total = (
                        fut.result()
                    )
                    client_states[cid] = state_dict
                    client_samples[cid] = sample_total
                    denom = max(sample_total, 1)
                    train_loss_total[cid] = loss_sum_total / denom
                    train_loss_components[cid] = {
                        k: v / denom for k, v in comp_sum_total.items()
                    }
        else:
            for pos, cid in enumerate(client_ids):
                cid, state_dict, sample_total, loss_sum_total, comp_sum_total = (
                    train_client_thread(
                        cid,
                        global_model,
                        state_for_cid(train_state_by_cid, cid, base_state),
                        train_loaders[cid],
                        loss_pool,
                        optim_cfg,
                        local_epochs,
                        device_by_cid[cid],
                        clip_norm,
                        pos,
                    )
                )
                client_states[cid] = state_dict
                client_samples[cid] = sample_total
                denom = max(sample_total, 1)
                train_loss_total[cid] = loss_sum_total / denom
                train_loss_components[cid] = {
                    k: v / denom for k, v in comp_sum_total.items()
                }
        round_weights = {
            cid: float(disco_weights.get(cid, 0.0)) for cid in client_states
        }
        if sum(round_weights.values()) <= 0.0:
            round_weights = {
                cid: float(client_samples.get(cid, 0)) for cid in client_states
            }
        local_states = client_states
        apply_aggregation(
            aggregator, global_model, client_states, round_weights, _log_info, epoch + 1
        )
        if test_loader is not None:
            test_stats: Dict[int, Dict[str, float]] = {}
            base_state_test = deepcopy(global_model.state_dict())
            test_loader_map = getattr(test_loader, "loaders", test_loader)
            test_items = (
                list(test_loader_map.values())
                if hasattr(test_loader_map, "values")
                else [test_loader_map]
            )
            test_has_workers = any(
                (getattr(loader, "num_workers", 0) > 0 for loader in test_items)
            )
            use_thread_pool_test = use_thread_pool and (not test_has_workers)
            if use_thread_pool and test_has_workers and logger:
                logger.warning(
                    "Detected DataLoader num_workers > 0 during evaluation. Running eval sequentially to avoid thread-worker deadlock."
                )

            def _select_test_loader(cid: int):
                return test_loader[cid]

            if use_thread_pool_test:
                with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                    futures: Dict = {}
                    for pos, cid in enumerate(client_ids):
                        eval_state = base_state_test
                        model_copy = clone_model(
                            global_model, eval_state, device_by_cid[cid]
                        )
                        fut = ex.submit(
                            eval_local,
                            model_copy,
                            _select_test_loader(cid),
                            loss_pool.get(str(cid)),
                            device_by_cid[cid],
                            cid,
                            pos + len(client_ids),
                            "test",
                            True,
                        )
                        futures[fut] = cid
                    for fut in as_completed(futures):
                        cid = futures[fut]
                        stats = fut.result()
                        test_stats[cid] = stats
            else:
                for pos, cid in enumerate(client_ids):
                    eval_state = base_state_test
                    model_copy = clone_model(
                        global_model, eval_state, device_by_cid[cid]
                    )
                    stats = eval_local(
                        model_copy,
                        _select_test_loader(cid),
                        loss_pool.get(str(cid)),
                        device_by_cid[cid],
                        cid,
                        pos + len(client_ids),
                        "test",
                        True,
                    )
                    test_stats[cid] = stats
            if test_stats:
                pred_records: List[Dict[str, object]] = []
                for stats in test_stats.values():
                    pred_records.extend(stats.get("pred_records", []))
                avg_metrics = aggregate_test_metrics(
                    test_stats, pred_records, mode=metric_mode
                )
                if logger:
                    logger.info(
                        f"[round {epoch + 1}] test avg loss={avg_metrics['loss_total']:.4f} mse={avg_metrics['mse']:.4f} spearman={avg_metrics['spearman']:.4f} kendall={avg_metrics['kendall']:.4f} r2={avg_metrics['r2']:.4f}"
                    )
