# implemented with reference to https://github.com/ki-ljl/Scaffold-Federated-Learning

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


def _train_local_scaffold(
    model,
    loader,
    loss_fn,
    optim,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    use_correction: bool,
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
    steps = 0
    desc = f"train-c{client_id}"
    if local_epoch is not None and local_epochs is not None:
        desc = f"train-c{client_id}-le{local_epoch}/{local_epochs}"
    c_global_device = {k: v.to(device) for k, v in c_global.items()}
    c_local_device = {k: v.to(device) for k, v in c_local.items()}
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        optim.zero_grad(set_to_none=True)
        ab = batch["ab_seq"]
        ag = batch.get("ag_seq")
        target = batch["target"].to(device)
        batch_size = target.size(0)
        out = model(ab_seqs=ab, ag_seqs=ag)
        total_loss, comps = loss_fn(out, target)
        total_loss.backward()
        if use_correction:
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                g = param.grad
                g = g - c_local_device.get(name, 0.0) + c_global_device.get(name, 0.0)
                param.grad = g
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optim.step()
        steps += 1
        loss_sum += total_loss.item() * batch_size
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
        sample_count += batch_size
        preds.extend(out["mu"].detach().cpu().flatten().tolist())
        targets_all.extend(target.detach().cpu().flatten().tolist())
    return (loss_sum, comp_sum, sample_count, preds, targets_all, steps)


def _train_client_thread_scaffold(
    cid: int,
    proto: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    base_state: Dict[str, torch.Tensor],
    train_loader,
    loss_pool: ClientLossPool,
    optim_cfg: Dict,
    local_epochs: int,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    use_correction: bool,
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
    steps_total = 0
    for _ in range(local_epochs):
        local_epoch = _ + 1
        loss_sum, comp_sum, samples, _, _, steps = _train_local_scaffold(
            model,
            train_loader,
            loss_fn,
            optimizer,
            c_global,
            c_local,
            use_correction,
            client_device,
            clip_norm,
            client_id=cid,
            position=position,
            local_epoch=local_epoch,
            local_epochs=local_epochs,
        )
        sample_total += samples
        loss_sum_total += loss_sum
        steps_total += steps
        for k, v in comp_sum.items():
            comp_sum_total[k] = comp_sum_total.get(k, 0.0) + v
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    lr = float(optim_cfg.get("lr", 0.01))
    steps_total = max(int(steps_total), 1)
    base_state_cpu = {k: v.detach().cpu() for k, v in base_state.items()}
    c_global_cpu = {k: v.detach().cpu() for k, v in c_global.items()}
    c_local_cpu = {k: v.detach().cpu() for k, v in c_local.items()}
    c_local_new: Dict[str, torch.Tensor] = {}
    c_delta: Dict[str, torch.Tensor] = {}
    for name, base_val in base_state_cpu.items():
        local_val = state_cpu.get(name)
        if local_val is None:
            continue
        delta = (base_val - local_val) / (lr * steps_total)
        new_val = (
            c_local_cpu.get(name, torch.zeros_like(base_val))
            - c_global_cpu.get(name, torch.zeros_like(base_val))
            + delta
        )
        c_local_new[name] = new_val
        c_delta[name] = new_val - c_local_cpu.get(name, torch.zeros_like(base_val))
    return (
        cid,
        state_cpu,
        sample_total,
        loss_sum_total,
        comp_sum_total,
        c_delta,
        c_local_new,
    )


def train_scaffold(
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
        algo = cfg.get("fl_algorithm", "scaffold")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    use_correction = bool(optim_cfg.get("scaffold_correction", True))
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    distribution_enabled = bool(cfg.get("distribution", True))
    local_states = {}
    metric_mode = resolve_metric_compute(cfg)
    c_global: Dict[str, torch.Tensor] = {}
    c_local: Dict[int, Dict[str, torch.Tensor]] = {}

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
        if not c_global:
            c_global = {
                k: torch.zeros_like(v.detach().cpu()) for k, v in base_state.items()
            }
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        client_samples: Dict[int, int] = {}
        train_loss_total: Dict[int, float] = {}
        train_loss_components: Dict[int, Dict[str, float]] = {}
        c_delta_sum: Dict[str, torch.Tensor] = {}
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futures = []
                for pos, cid in enumerate(client_ids):
                    futures.append(
                        ex.submit(
                            _train_client_thread_scaffold,
                            cid,
                            global_model,
                            state_for_cid(train_state_by_cid, cid, base_state),
                            state_for_cid(train_state_by_cid, cid, base_state),
                            train_loaders[cid],
                            loss_pool,
                            optim_cfg,
                            local_epochs,
                            c_global,
                            c_local.get(
                                cid,
                                {k: torch.zeros_like(v) for k, v in c_global.items()},
                            ),
                            use_correction,
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
                        c_delta,
                        c_new,
                    ) = fut.result()
                    client_states[cid] = state_dict
                    client_samples[cid] = sample_total
                    denom = max(sample_total, 1)
                    train_loss_total[cid] = loss_sum_total / denom
                    train_loss_components[cid] = {
                        k: v / denom for k, v in comp_sum_total.items()
                    }
                    c_local[cid] = c_new
                    for name, delta in c_delta.items():
                        c_delta_sum[name] = (
                            c_delta_sum.get(name, torch.zeros_like(delta)) + delta
                        )
        else:
            for pos, cid in enumerate(client_ids):
                (
                    cid,
                    state_dict,
                    sample_total,
                    loss_sum_total,
                    comp_sum_total,
                    c_delta,
                    c_new,
                ) = _train_client_thread_scaffold(
                    cid,
                    global_model,
                    state_for_cid(train_state_by_cid, cid, base_state),
                    state_for_cid(train_state_by_cid, cid, base_state),
                    train_loaders[cid],
                    loss_pool,
                    optim_cfg,
                    local_epochs,
                    c_global,
                    c_local.get(
                        cid, {k: torch.zeros_like(v) for k, v in c_global.items()}
                    ),
                    use_correction,
                    device_by_cid[cid],
                    clip_norm,
                    pos,
                )
                client_states[cid] = state_dict
                client_samples[cid] = sample_total
                denom = max(sample_total, 1)
                train_loss_total[cid] = loss_sum_total / denom
                train_loss_components[cid] = {
                    k: v / denom for k, v in comp_sum_total.items()
                }
                c_local[cid] = c_new
                for name, delta in c_delta.items():
                    c_delta_sum[name] = (
                        c_delta_sum.get(name, torch.zeros_like(delta)) + delta
                    )
        if client_ids:
            scale = 1.0 / max(len(client_ids), 1)
            for name, delta in c_delta_sum.items():
                c_global[name] = (
                    c_global.get(name, torch.zeros_like(delta)) + delta * scale
                )
        local_states = client_states
        apply_aggregation(
            aggregator,
            global_model,
            client_states,
            client_samples,
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
                avg_metrics = aggregate_test_metrics(
                    test_stats, pred_records, mode=metric_mode
                )
                if logger:
                    logger.info(
                        f"[round {epoch + 1}] test avg loss={avg_metrics['loss_total']:.4f} mse={avg_metrics['mse']:.4f} spearman={avg_metrics['spearman']:.4f} kendall={avg_metrics['kendall']:.4f} r2={avg_metrics['r2']:.4f}"
                    )
