# implemented with reference to paper Communication-Efficient Learning of Deep Networks from Decentralized Data.

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from copy import deepcopy
from typing import Dict, List
import torch
from modules import ClientLossPool
from utils.config import resolve_train_cfg
from fl_algorithms.training import (
    aggregate_test_metrics,
    aggregate_test_metrics_ranking,
    resolve_metric_compute,
    resolve_client_concurrency,
    apply_aggregation,
    clone_model,
    eval_local,
    train_client_thread,
    select_distribution_states,
    state_for_cid,
)


def train_fedavg(
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
        algo = cfg.get("fl_algorithm", "fedavg")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    miss_client_ratio_raw = train_cfg.get("miss_client_ratio", 0.0)
    try:
        miss_client_ratio = float(miss_client_ratio_raw)
    except Exception as err:
        raise ValueError(
            f"train.fedavg.miss_client_ratio must be a float in [0, 1), got {miss_client_ratio_raw!r}."
        ) from err
    if miss_client_ratio < 0.0 or miss_client_ratio >= 1.0:
        raise ValueError(
            f"train.fedavg.miss_client_ratio must be in [0, 1), got {miss_client_ratio}."
        )
    distribution_enabled = bool(cfg.get("distribution", True))
    if miss_client_ratio > 0.0 and not distribution_enabled:
        raise ValueError("train.fedavg.miss_client_ratio requires distribution=true.")
    local_states = {}
    metric_mode = resolve_metric_compute(cfg)
    ranking_mode = (
        str(cfg.get("dataset", "")).strip().lower() == "abrank"
        and str(cfg.get("task", "regression")).strip().lower() == "ranking"
    )

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
    for epoch in range(comm_rounds):
        base_state = deepcopy(global_model.state_dict())
        participating_clients = list(client_ids)
        if miss_client_ratio > 0.0 and client_ids:
            keep_ratio = 1.0 - miss_client_ratio
            keep_count = max(1, int(round(len(client_ids) * keep_ratio)))
            keep_count = min(keep_count, len(client_ids))
            participating_clients = sorted(random.sample(client_ids, keep_count))
        participating_client_set = set(participating_clients)
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        if miss_client_ratio > 0.0:
            train_state_by_cid = {
                cid: (
                    base_state
                    if cid in participating_client_set
                    else local_states.get(cid, base_state)
                )
                for cid in client_ids
            }
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
        local_states = client_states
        agg_client_states = client_states
        agg_client_samples = client_samples
        if miss_client_ratio > 0.0:
            agg_client_states = {
                cid: client_states[cid] for cid in participating_clients
            }
            agg_client_samples = {
                cid: client_samples[cid] for cid in participating_clients
            }
        apply_aggregation(
            aggregator,
            global_model,
            agg_client_states,
            agg_client_samples,
            _log_info,
            epoch + 1,
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
                if ranking_mode:
                    avg_metrics = aggregate_test_metrics_ranking(
                        test_stats, pred_records, mode=metric_mode
                    )
                else:
                    avg_metrics = aggregate_test_metrics(
                        test_stats, pred_records, mode=metric_mode
                    )
                if logger:
                    if ranking_mode:
                        logger.info(
                            f"[round {epoch + 1}] test avg loss={avg_metrics['loss_total']:.4f} auc={avg_metrics['auc']:.4f} acc={avg_metrics['accuracy']:.4f} precision={avg_metrics['precision']:.4f} recall={avg_metrics['recall']:.4f}"
                        )
                    else:
                        logger.info(
                            f"[round {epoch + 1}] test avg loss={avg_metrics['loss_total']:.4f} mse={avg_metrics['mse']:.4f} spearman={avg_metrics['spearman']:.4f} kendall={avg_metrics['kendall']:.4f} r2={avg_metrics['r2']:.4f}"
                        )
