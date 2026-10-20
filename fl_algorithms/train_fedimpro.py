# implemented with reference to https://github.com/wizard1203/FedImpro

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List
import torch
from tqdm.auto import tqdm
from modules import ClientLossPool, build_optimizer
from utils.config import resolve_train_cfg
from fl_algorithms.training import (
    aggregate_test_metrics,
    resolve_metric_compute,
    resolve_client_concurrency,
    apply_aggregation,
    clone_model,
    eval_local,
    select_distribution_states,
    state_for_cid,
)


def _extract_labels(
    batch: Dict, label_key: str, bins: int, label_min: float, label_max: float
) -> List[str]:
    if label_key != "target":
        raise ValueError(
            "fedimpro only supports target-bin labels now; labels must come from target."
        )
    targets = batch["target"].detach().cpu().tolist()
    if label_max <= label_min:
        raise ValueError(
            "fedimpro.label_max must be greater than label_min when label_key='target'."
        )
    span = label_max - label_min
    labels = []
    for t in targets:
        idx = int((float(t) - label_min) / span * bins)
        idx = max(0, min(bins - 1, idx))
        labels.append(f"bin_{idx}")
    return labels


def _forward_with_repr(model, ab_seqs, ag_seqs):
    if model._pair_encoder:
        ab_vec, ag_vec = model._encode_pair(ab_seqs, ag_seqs)
    else:
        ab_vec = model._encode_sequence(ab_seqs)
        ag_vec = model._encode_sequence(ag_seqs)
    return (model.head(ab_vec, ag_vec), torch.cat([ab_vec, ag_vec], dim=-1))


def _update_proto_sums(
    proto_sums: Dict[str, torch.Tensor],
    proto_counts: Dict[str, int],
    labels: List[str],
    reps: torch.Tensor,
):
    for idx, label in enumerate(labels):
        vec = reps[idx].detach().cpu()
        if label not in proto_sums:
            proto_sums[label] = vec.clone()
            proto_counts[label] = 1
        else:
            proto_sums[label] = proto_sums[label] + vec
            proto_counts[label] = proto_counts[label] + 1


def _train_local_impro(
    model,
    loader,
    loss_fn,
    optim,
    proto_global: Dict[str, torch.Tensor],
    proto_loss_weight: float,
    label_key: str,
    bins: int,
    label_min: float,
    label_max: float,
    device,
    clip_norm: float,
    client_id: int,
    position: int = 0,
    local_epoch: int | None = None,
    local_epochs: int | None = None,
):
    model.train()
    loss_sum = 0.0
    comp_sum: Dict[str, float] = {}
    sample_count = 0
    preds: List[float] = []
    targets_all: List[float] = []
    proto_sums: Dict[str, torch.Tensor] = {}
    proto_counts: Dict[str, int] = {}
    desc = f"train-c{client_id}"
    if local_epoch is not None and local_epochs is not None:
        desc = f"train-c{client_id}-le{local_epoch}/{local_epochs}"
    proto_global_device = {k: v.to(device) for k, v in proto_global.items()}
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        optim.zero_grad(set_to_none=True)
        ab = batch["ab_seq"]
        ag = batch.get("ag_seq")
        target = batch["target"].to(device)
        batch_size = target.size(0)
        outputs, reps = _forward_with_repr(model, ab, ag)
        total_loss, comps = loss_fn(outputs, target)
        labels = _extract_labels(batch, label_key, bins, label_min, label_max)
        if reps is not None and proto_loss_weight > 0:
            proto_list = []
            mask = []
            for label in labels:
                proto = proto_global_device.get(label)
                if proto is None:
                    proto_list.append(torch.zeros(reps.size(-1), device=device))
                    mask.append(False)
                else:
                    proto_list.append(proto)
                    mask.append(True)
            mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
            if torch.any(mask_tensor):
                proto_tensor = torch.stack(proto_list, dim=0)
                rep_loss = torch.mean(
                    (reps[mask_tensor] - proto_tensor[mask_tensor]) ** 2
                )
                total_loss = total_loss + proto_loss_weight * rep_loss
                comps = dict(comps)
                comps["proto"] = rep_loss.detach()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optim.step()
        loss_sum += total_loss.item() * batch_size
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
        sample_count += batch_size
        preds.extend(outputs["mu"].detach().cpu().flatten().tolist())
        targets_all.extend(target.detach().cpu().flatten().tolist())
        if reps is not None:
            _update_proto_sums(proto_sums, proto_counts, labels, reps)
    return (
        loss_sum,
        comp_sum,
        sample_count,
        preds,
        targets_all,
        proto_sums,
        proto_counts,
    )


def _train_client_thread_impro(
    cid: int,
    proto: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    train_loader,
    loss_pool: ClientLossPool,
    optim_cfg: Dict,
    local_epochs: int,
    proto_global: Dict[str, torch.Tensor],
    proto_loss_weight: float,
    label_key: str,
    bins: int,
    label_min: float,
    label_max: float,
    client_device: torch.device,
    clip_norm: float,
    position: int,
):
    model = clone_model(proto, state, client_device)
    optimizer = build_optimizer(model.parameters(), optim_cfg)
    loss_fn = loss_pool.get(str(cid))
    sample_total = 0
    loss_sum_total = 0.0
    comp_sum_total: Dict[str, float] = {}
    proto_sums_total: Dict[str, torch.Tensor] = {}
    proto_counts_total: Dict[str, int] = {}
    for _ in range(local_epochs):
        local_epoch = _ + 1
        loss_sum, comp_sum, samples, _, _, proto_sums, proto_counts = (
            _train_local_impro(
                model,
                train_loader,
                loss_fn,
                optimizer,
                proto_global,
                proto_loss_weight,
                label_key,
                bins,
                label_min,
                label_max,
                client_device,
                clip_norm,
                client_id=cid,
                position=position,
                local_epoch=local_epoch,
                local_epochs=local_epochs,
            )
        )
        sample_total += samples
        loss_sum_total += loss_sum
        for k, v in comp_sum.items():
            comp_sum_total[k] = comp_sum_total.get(k, 0.0) + v
        for label, vec in proto_sums.items():
            if label not in proto_sums_total:
                proto_sums_total[label] = vec
                proto_counts_total[label] = proto_counts[label]
            else:
                proto_sums_total[label] = proto_sums_total[label] + vec
                proto_counts_total[label] = (
                    proto_counts_total[label] + proto_counts[label]
                )
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    proto_means: Dict[str, torch.Tensor] = {}
    proto_counts: Dict[str, int] = {}
    for label, vec in proto_sums_total.items():
        count = max(int(proto_counts_total.get(label, 0)), 1)
        proto_means[label] = vec / count
        proto_counts[label] = count
    return (
        cid,
        state_cpu,
        sample_total,
        loss_sum_total,
        comp_sum_total,
        proto_means,
        proto_counts,
    )


def _aggregate_prototypes(
    global_protos: Dict[str, torch.Tensor],
    client_protos: Dict[int, Dict[str, torch.Tensor]],
    client_counts: Dict[int, Dict[str, int]],
) -> Dict[str, torch.Tensor]:
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for cid, proto_map in client_protos.items():
        count_map = client_counts.get(cid, {})
        for label, vec in proto_map.items():
            count = int(count_map.get(label, 0))
            if count <= 0:
                continue
            if label not in sums:
                sums[label] = vec * count
                counts[label] = count
            else:
                sums[label] = sums[label] + vec * count
                counts[label] = counts[label] + count
    updated = dict(global_protos)
    for label, vec in sums.items():
        denom = max(counts.get(label, 0), 1)
        updated[label] = vec / denom
    return updated


def train_fedimpro(
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
        algo = cfg.get("fl_algorithm", "fedimpro")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    proto_cfg = (
        train_cfg.get("prototype", {})
        if isinstance(train_cfg.get("prototype", {}), dict)
        else {}
    )
    proto_loss_weight = float(proto_cfg.get("loss_weight", 1.0))
    label_key = "target"
    bins = int(proto_cfg.get("label_bins", 10))
    label_min = float(proto_cfg.get("label_min", 0.0))
    label_max = float(proto_cfg.get("label_max", 1.0))
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    distribution_enabled = bool(cfg.get("distribution", True))
    local_states = {}
    if not distribution_enabled:
        raise ValueError("distribution=false is not supported for fedimpro.")
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
    global_protos: Dict[str, torch.Tensor] = {}
    for epoch in range(comm_rounds):
        base_state = deepcopy(global_model.state_dict())
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        client_samples: Dict[int, int] = {}
        train_loss_total: Dict[int, float] = {}
        train_loss_components: Dict[int, Dict[str, float]] = {}
        client_protos: Dict[int, Dict[str, torch.Tensor]] = {}
        client_proto_counts: Dict[int, Dict[str, int]] = {}
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futures = []
                for pos, cid in enumerate(client_ids):
                    futures.append(
                        ex.submit(
                            _train_client_thread_impro,
                            cid,
                            global_model,
                            state_for_cid(train_state_by_cid, cid, base_state),
                            train_loaders[cid],
                            loss_pool,
                            optim_cfg,
                            local_epochs,
                            global_protos,
                            proto_loss_weight,
                            label_key,
                            bins,
                            label_min,
                            label_max,
                            device_by_cid[cid],
                            clip_norm,
                            pos,
                        )
                    )
                for fut in as_completed(futures):
                    (
                        cid,
                        state_dict,
                        sample_total,
                        loss_sum_total,
                        comp_sum_total,
                        proto_means,
                        proto_counts,
                    ) = fut.result()
                    client_states[cid] = state_dict
                    client_samples[cid] = sample_total
                    client_protos[cid] = proto_means
                    client_proto_counts[cid] = proto_counts
                    denom = max(sample_total, 1)
                    train_loss_total[cid] = loss_sum_total / denom
                    train_loss_components[cid] = {
                        k: v / denom for k, v in comp_sum_total.items()
                    }
        else:
            for pos, cid in enumerate(client_ids):
                (
                    cid,
                    state_dict,
                    sample_total,
                    loss_sum_total,
                    comp_sum_total,
                    proto_means,
                    proto_counts,
                ) = _train_client_thread_impro(
                    cid,
                    global_model,
                    state_for_cid(train_state_by_cid, cid, base_state),
                    train_loaders[cid],
                    loss_pool,
                    optim_cfg,
                    local_epochs,
                    global_protos,
                    proto_loss_weight,
                    label_key,
                    bins,
                    label_min,
                    label_max,
                    device_by_cid[cid],
                    clip_norm,
                    pos,
                )
                client_states[cid] = state_dict
                client_samples[cid] = sample_total
                client_protos[cid] = proto_means
                client_proto_counts[cid] = proto_counts
                denom = max(sample_total, 1)
                train_loss_total[cid] = loss_sum_total / denom
                train_loss_components[cid] = {
                    k: v / denom for k, v in comp_sum_total.items()
                }
        local_states = client_states
        apply_aggregation(
            aggregator,
            global_model,
            client_states,
            client_samples,
            _log_info,
            epoch + 1,
        )
        global_protos = _aggregate_prototypes(
            global_protos, client_protos, client_proto_counts
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
