# implemented with reference to https://github.com/junkangLiu0/FedSWA

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


def _parse_fedmoswa_hparams(
    train_cfg: Dict, optim_cfg: Dict
) -> tuple[float, float, float, float]:
    eta_l = float(train_cfg.get("eta_l", optim_cfg.get("lr", 0.01)))
    rho = float(train_cfg.get("rho", 0.01))
    alpha = float(train_cfg.get("alpha", 1.0))
    gamma = float(train_cfg.get("gamma", 0.5))
    if eta_l <= 0.0:
        raise ValueError("fedmoswa requires eta_l > 0.")
    if rho < 0.0 or rho > 1.0:
        raise ValueError("fedmoswa requires rho in [0, 1].")
    if alpha < 0.0:
        raise ValueError("fedmoswa requires alpha >= 0.")
    if gamma < 0.0 or gamma >= 1.0:
        raise ValueError("fedmoswa requires gamma in [0, 1).")
    return (eta_l, rho, alpha, gamma)


def _linear_anneal_lr(
    eta_l: float, rho: float, step_idx: int, total_steps: int
) -> float:
    if total_steps <= 1:
        return eta_l
    progress = max(0.0, min(1.0, float(step_idx) / float(total_steps - 1)))
    return eta_l * (1.0 - progress + progress * rho)


def _update_swa_state(
    swa_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    swa_count: int,
) -> int:
    new_count = int(swa_count) + 1
    coef = 1.0 / float(new_count)
    for name, value in model_state.items():
        value_cpu = value.detach().cpu()
        if not torch.is_floating_point(value_cpu):
            swa_state[name] = value_cpu.clone()
            continue
        prev = swa_state.get(name)
        if prev is None:
            swa_state[name] = value_cpu.clone()
        else:
            prev.add_((value_cpu - prev) * coef)
    return new_count


def _train_local_moswa(
    model,
    loader,
    loss_fn,
    optim,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    alpha: float,
    eta_l: float,
    rho: float,
    total_steps: int,
    step_offset: int,
    swa_state: Dict[str, torch.Tensor],
    swa_count: int,
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
    lr_sum = 0.0
    lr_steps = 0
    desc = f"train-c{client_id}"
    if local_epoch is not None and local_epochs is not None:
        desc = f"train-c{client_id}-le{local_epoch}/{local_epochs}"
    step_count = int(step_offset)
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        current_lr = _linear_anneal_lr(eta_l, rho, step_count, total_steps)
        for group in optim.param_groups:
            group["lr"] = current_lr
        optim.zero_grad(set_to_none=True)
        ab = batch["ab_seq"]
        ag = batch.get("ag_seq")
        target = batch["target"].to(device)
        batch_size = target.size(0)
        out = model(ab_seqs=ab, ag_seqs=ag)
        total_loss, comps = loss_fn(out, target)
        total_loss.backward()
        if alpha > 0.0:
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                c_g = c_global.get(name)
                c_l = c_local.get(name)
                correction = None
                if c_g is not None:
                    correction = c_g
                if c_l is not None:
                    correction = -c_l if correction is None else correction - c_l
                if correction is not None:
                    param.grad.add_(correction, alpha=alpha)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optim.step()
        step_count += 1
        lr_sum += current_lr
        lr_steps += 1
        swa_count = _update_swa_state(swa_state, model.state_dict(), swa_count)
        loss_sum += total_loss.item() * batch_size
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
        sample_count += batch_size
        preds.extend(out["mu"].detach().cpu().flatten().tolist())
        targets_all.extend(target.detach().cpu().flatten().tolist())
    return (
        loss_sum,
        comp_sum,
        sample_count,
        preds,
        targets_all,
        step_count,
        lr_sum,
        lr_steps,
        swa_count,
    )


def _train_client_thread_moswa(
    cid: int,
    proto: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    train_loader,
    loss_pool: ClientLossPool,
    optim_cfg: Dict,
    local_epochs: int,
    c_global: Dict[str, torch.Tensor],
    c_local: Dict[str, torch.Tensor],
    alpha: float,
    eta_l: float,
    rho: float,
    client_device: torch.device,
    clip_norm: float,
    position: int,
):
    model = clone_model(proto, state, client_device)
    client_optim_cfg = dict(optim_cfg)
    client_optim_cfg["lr"] = eta_l
    optimizer = build_optimizer(model.parameters(), client_optim_cfg)
    loss_fn = loss_pool.get(str(cid))
    c_global_dev = {k: v.to(client_device) for k, v in c_global.items()}
    c_local_dev = {k: v.to(client_device) for k, v in c_local.items()}
    try:
        steps_per_epoch = len(train_loader)
    except Exception:
        steps_per_epoch = 1
    total_steps = max(int(local_epochs) * max(int(steps_per_epoch), 1), 1)
    sample_total = 0
    loss_sum_total = 0.0
    comp_sum_total: Dict[str, float] = {}
    step_count = 0
    lr_sum_total = 0.0
    lr_steps_total = 0
    swa_state: Dict[str, torch.Tensor] = {}
    swa_count = 0
    for _ in range(local_epochs):
        local_epoch = _ + 1
        loss_sum, comp_sum, samples, _, _, step_count, lr_sum, lr_steps, swa_count = (
            _train_local_moswa(
                model,
                train_loader,
                loss_fn,
                optimizer,
                c_global_dev,
                c_local_dev,
                alpha,
                eta_l,
                rho,
                total_steps,
                step_count,
                swa_state,
                swa_count,
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
        lr_sum_total += lr_sum
        lr_steps_total += lr_steps
        for k, v in comp_sum.items():
            comp_sum_total[k] = comp_sum_total.get(k, 0.0) + v
    if swa_count > 0:
        state_cpu = {k: v.detach().cpu() for k, v in swa_state.items()}
    else:
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    start_state_cpu = {k: v.detach().cpu() for k, v in state.items()}
    avg_lr = lr_sum_total / max(lr_steps_total, 1)
    denom = max(avg_lr * max(step_count, 1), 1e-12)
    c_local_new: Dict[str, torch.Tensor] = {}
    c_delta: Dict[str, torch.Tensor] = {}
    for name, c_global_val in c_global.items():
        base_val = start_state_cpu.get(name)
        local_val = state_cpu.get(name)
        if base_val is None or local_val is None:
            continue
        if not torch.is_floating_point(base_val) or not torch.is_floating_point(
            local_val
        ):
            continue
        prev_c = c_local.get(name)
        if prev_c is None:
            prev_c = torch.zeros_like(c_global_val)
        delta_est = (base_val - local_val) / denom
        new_c = prev_c - c_global_val + delta_est
        c_local_new[name] = new_c
        c_delta[name] = new_c - prev_c
    return (
        cid,
        state_cpu,
        sample_total,
        loss_sum_total,
        comp_sum_total,
        c_delta,
        c_local_new,
    )


def _control_norm(control_state: Dict[str, torch.Tensor]) -> float:
    if not control_state:
        return 0.0
    norm_sq = 0.0
    for value in control_state.values():
        v = value.detach().to(dtype=torch.float32)
        norm_sq += float(torch.sum(v * v).item())
    return norm_sq**0.5


def train_fedmoswa(
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
        algo = cfg.get("fl_algorithm", "fedmoswa")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    eta_l, rho, alpha, gamma = _parse_fedmoswa_hparams(train_cfg, optim_cfg)
    optim_cfg = dict(optim_cfg)
    optim_cfg["lr"] = eta_l
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    distribution_enabled = bool(cfg.get("distribution", True))
    if not distribution_enabled:
        raise ValueError("distribution=false is not supported for fedmoswa.")
    local_states = {}
    metric_mode = resolve_metric_compute(cfg)
    c_global: Dict[str, torch.Tensor] = {}
    c_momentum: Dict[str, torch.Tensor] = {}
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
        base_state_cpu = {k: v.detach().cpu() for k, v in base_state.items()}
        if not c_global:
            for name, param in global_model.named_parameters():
                if not param.requires_grad:
                    continue
                ref = base_state_cpu.get(name)
                if ref is None or not torch.is_floating_point(ref):
                    continue
                c_global[name] = torch.zeros_like(ref)
                c_momentum[name] = torch.zeros_like(ref)
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        client_samples: Dict[int, int] = {}
        train_loss_total: Dict[int, float] = {}
        train_loss_components: Dict[int, Dict[str, float]] = {}
        c_delta_by_cid: Dict[int, Dict[str, torch.Tensor]] = {}
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futures = []
                for pos, cid in enumerate(client_ids):
                    local_ci = c_local.get(cid)
                    if local_ci is None:
                        local_ci = {k: torch.zeros_like(v) for k, v in c_global.items()}
                    futures.append(
                        ex.submit(
                            _train_client_thread_moswa,
                            cid,
                            global_model,
                            state_for_cid(train_state_by_cid, cid, base_state),
                            train_loaders[cid],
                            loss_pool,
                            optim_cfg,
                            local_epochs,
                            c_global,
                            local_ci,
                            alpha,
                            eta_l,
                            rho,
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
                    c_delta_by_cid[cid] = c_delta
        else:
            for pos, cid in enumerate(client_ids):
                local_ci = c_local.get(cid)
                if local_ci is None:
                    local_ci = {k: torch.zeros_like(v) for k, v in c_global.items()}
                (
                    cid,
                    state_dict,
                    sample_total,
                    loss_sum_total,
                    comp_sum_total,
                    c_delta,
                    c_new,
                ) = _train_client_thread_moswa(
                    cid,
                    global_model,
                    state_for_cid(train_state_by_cid, cid, base_state),
                    train_loaders[cid],
                    loss_pool,
                    optim_cfg,
                    local_epochs,
                    c_global,
                    local_ci,
                    alpha,
                    eta_l,
                    rho,
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
                c_delta_by_cid[cid] = c_delta
        if c_global and c_delta_by_cid:
            total_weight = float(sum((max(int(v), 0) for v in client_samples.values())))
            if total_weight <= 0.0:
                total_weight = float(max(len(c_delta_by_cid), 1))
            delta_avg = {name: torch.zeros_like(val) for name, val in c_global.items()}
            for cid, c_delta in c_delta_by_cid.items():
                if sum((max(int(v), 0) for v in client_samples.values())) > 0:
                    weight = (
                        float(max(int(client_samples.get(cid, 0)), 0)) / total_weight
                    )
                else:
                    weight = 1.0 / float(max(len(c_delta_by_cid), 1))
                for name, delta in c_delta.items():
                    if name not in delta_avg:
                        continue
                    delta_avg[name].add_(delta, alpha=weight)
            for name in c_global.keys():
                mom = c_momentum.get(name)
                if mom is None:
                    mom = torch.zeros_like(c_global[name])
                mom = mom * gamma + delta_avg[name] * (1.0 - gamma)
                c_momentum[name] = mom
                c_global[name] = c_global[name] + mom
            _log_info(f"[round {epoch + 1}] control_norm={_control_norm(c_global):.6f}")
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
