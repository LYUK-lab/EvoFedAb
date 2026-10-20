# implemented with reference to https://github.com/junkangLiu0/FedAdamW

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List
import torch
from tqdm.auto import tqdm
from modules import ClientLossPool
from utils.config import resolve_train_cfg
from fl_algorithms.training import (
    aggregate_test_metrics,
    resolve_metric_compute,
    resolve_client_concurrency,
    clone_model,
    eval_local,
    apply_aggregation,
    select_distribution_states,
    state_for_cid,
)


def _parse_server_update(server_cfg: dict) -> float:
    if not isinstance(server_cfg, dict):
        raise KeyError("train config missing server_update for fedadamw.")
    allowed = {"alpha"}
    extra = sorted(set(server_cfg) - allowed)
    if extra:
        raise ValueError(f"Unsupported server_update keys for fedadamw: {extra}")
    if "alpha" not in server_cfg:
        raise KeyError("server_update.alpha required for fedadamw.")
    alpha = float(server_cfg["alpha"])
    if alpha == 0.0:
        raise ValueError(
            "server_update.alpha must be non-zero for fedadamw (set to 0.1 to enable global correction)."
        )
    return alpha


def _parse_client_optimizer(
    optim_cfg: dict,
) -> tuple[float, float, float, float, float]:
    if not isinstance(optim_cfg, dict):
        raise KeyError("train config missing client_optimizer for fedadamw.")
    allowed = {"name", "lr", "beta1", "beta2", "eps", "weight_decay"}
    extra = sorted(set(optim_cfg) - allowed)
    if extra:
        raise ValueError(f"Unsupported client_optimizer keys for fedadamw: {extra}")
    name = str(optim_cfg.get("name", "")).lower()
    if name != "adamw":
        raise ValueError("fedadamw requires client_optimizer.name='adamw' on clients.")
    missing = [
        k for k in ("lr", "beta1", "beta2", "eps", "weight_decay") if k not in optim_cfg
    ]
    if missing:
        raise KeyError(
            f"client_optimizer missing required keys for fedadamw: {missing}"
        )
    lr = float(optim_cfg["lr"])
    beta1 = float(optim_cfg["beta1"])
    beta2 = float(optim_cfg["beta2"])
    eps = float(optim_cfg["eps"])
    weight_decay = float(optim_cfg["weight_decay"])
    if lr <= 0.0:
        raise ValueError("client_optimizer.lr must be > 0 for fedadamw.")
    if beta1 < 0.0 or beta1 >= 1.0:
        raise ValueError("client_optimizer.beta1 must be in [0, 1).")
    if beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("client_optimizer.beta2 must be in [0, 1).")
    if eps <= 0.0:
        raise ValueError("client_optimizer.eps must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("client_optimizer.weight_decay must be >= 0.")
    return (lr, beta1, beta2, eps, weight_decay)


def _train_local_adamw(
    model,
    loader,
    loss_fn,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    alpha: float,
    weight_decay: float,
    delta_global: Dict[str, torch.Tensor],
    m_state: Dict[str, torch.Tensor],
    v_state: Dict[str, torch.Tensor],
    device,
    clip_norm: float,
    client_id: int,
    position: int = 0,
    local_epoch: int | None = None,
    local_epochs: int | None = None,
    step_offset: int = 0,
):
    model.train()
    loss_sum = 0.0
    comp_sum: Dict[str, float] = {}
    sample_count = 0
    preds: List[float] = []
    targets_all: List[float] = []
    desc = f"train-c{client_id}"
    if local_epoch is not None and local_epochs is not None:
        desc = f"train-c{client_id}-le{local_epoch}/{local_epochs}"
    delta_global_device = {k: v.to(device) for k, v in delta_global.items()}
    step_count = step_offset
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        model.zero_grad(set_to_none=True)
        ab = batch["ab_seq"]
        ag = batch.get("ag_seq")
        target = batch["target"].to(device)
        batch_size = target.size(0)
        out = model(ab_seqs=ab, ag_seqs=ag)
        total_loss, comps = loss_fn(out, target)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        step_count += 1
        bias_correction1 = 1.0 - beta1**step_count
        bias_correction2 = 1.0 - beta2**step_count
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                grad = param.grad
                m = m_state.get(name)
                v = v_state.get(name)
                if m is None:
                    m = torch.zeros_like(grad)
                if v is None:
                    v = torch.zeros_like(grad)
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad * grad)
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                theta = 1.0 / (torch.sqrt(v_hat) + eps)
                update = m_hat * theta
                if alpha != 0.0:
                    update = update + alpha * delta_global_device.get(name, 0.0)
                if weight_decay != 0.0:
                    update = update - weight_decay * param
                param.add_(-lr * update)
                m_state[name] = m
                v_state[name] = v
        loss_sum += total_loss.item() * batch_size
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
        sample_count += batch_size
        preds.extend(out["mu"].detach().cpu().flatten().tolist())
        targets_all.extend(target.detach().cpu().flatten().tolist())
    return (loss_sum, comp_sum, sample_count, preds, targets_all, step_count)


def _train_client_thread_adamw(
    cid: int,
    proto: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    train_loader,
    loss_pool: ClientLossPool,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    alpha: float,
    weight_decay: float,
    delta_global: Dict[str, torch.Tensor],
    v_bar: Dict[str, torch.Tensor],
    local_epochs: int,
    client_device: torch.device,
    clip_norm: float,
    position: int,
):
    model = clone_model(proto, state, client_device)
    loss_fn = loss_pool.get(str(cid))
    m_state: Dict[str, torch.Tensor] = {}
    v_state = {k: v.to(client_device) for k, v in v_bar.items()}
    sample_total = 0
    loss_sum_total = 0.0
    comp_sum_total: Dict[str, float] = {}
    step_count = 0
    for _ in range(local_epochs):
        local_epoch = _ + 1
        loss_sum, comp_sum, samples, _, _, step_count = _train_local_adamw(
            model,
            train_loader,
            loss_fn,
            lr,
            beta1,
            beta2,
            eps,
            alpha,
            weight_decay,
            delta_global,
            m_state,
            v_state,
            client_device,
            clip_norm,
            client_id=cid,
            position=position,
            local_epoch=local_epoch,
            local_epochs=local_epochs,
            step_offset=step_count,
        )
        sample_total += samples
        loss_sum_total += loss_sum
        for k, v in comp_sum.items():
            comp_sum_total[k] = comp_sum_total.get(k, 0.0) + v
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    v_cpu = {k: v.detach().cpu() for k, v in v_state.items()}
    return (cid, state_cpu, sample_total, loss_sum_total, comp_sum_total, v_cpu)


def train_fedadamw(
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
        algo = cfg.get("fl_algorithm", "fedadamw")
        raise KeyError(
            f"train config missing client_optimizer/loss for algorithm '{algo}'."
        )
    client_lr, beta1, beta2, eps, weight_decay = _parse_client_optimizer(optim_cfg)
    server_cfg = train_cfg.get("server_update")
    alpha = _parse_server_update(server_cfg)
    clip_norm = train_cfg.get("clip_norm", 1.0)
    loss_pool = ClientLossPool(train_cfg["loss"])
    comm_rounds = int(train_cfg["comm_rounds"])
    local_epochs = train_cfg.get("local_epochs", 1)
    distribution_enabled = bool(cfg.get("distribution", True))
    local_states = {}
    if not distribution_enabled:
        raise ValueError("distribution=false is not supported for fedadamw.")
    metric_mode = resolve_metric_compute(cfg)
    v_bar: Dict[str, torch.Tensor] = {}
    delta_global: Dict[str, torch.Tensor] = {}

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
        if not v_bar:
            v_bar = {k: torch.zeros_like(v) for k, v in base_state_cpu.items()}
        if not delta_global:
            delta_global = {k: torch.zeros_like(v) for k, v in base_state_cpu.items()}
        train_state_by_cid = select_distribution_states(
            local_states, distribution_enabled
        )
        client_states: Dict[int, Dict[str, torch.Tensor]] = {}
        client_samples: Dict[int, int] = {}
        client_v_states: Dict[int, Dict[str, torch.Tensor]] = {}
        train_loss_total: Dict[int, float] = {}
        train_loss_components: Dict[int, Dict[str, float]] = {}
        if use_thread_pool:
            with ThreadPoolExecutor(max_workers=len(client_ids)) as ex:
                futures = []
                for pos, cid in enumerate(client_ids):
                    futures.append(
                        ex.submit(
                            _train_client_thread_adamw,
                            cid,
                            global_model,
                            state_for_cid(train_state_by_cid, cid, base_state),
                            train_loaders[cid],
                            loss_pool,
                            client_lr,
                            beta1,
                            beta2,
                            eps,
                            alpha,
                            weight_decay,
                            delta_global,
                            v_bar,
                            local_epochs,
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
                        v_state,
                    ) = fut.result()
                    client_states[cid] = state_dict
                    client_samples[cid] = sample_total
                    client_v_states[cid] = v_state
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
                    v_state,
                ) = _train_client_thread_adamw(
                    cid,
                    global_model,
                    state_for_cid(train_state_by_cid, cid, base_state),
                    train_loaders[cid],
                    loss_pool,
                    client_lr,
                    beta1,
                    beta2,
                    eps,
                    alpha,
                    weight_decay,
                    delta_global,
                    v_bar,
                    local_epochs,
                    device_by_cid[cid],
                    clip_norm,
                    pos,
                )
                client_states[cid] = state_dict
                client_samples[cid] = sample_total
                client_v_states[cid] = v_state
                denom = max(sample_total, 1)
                train_loss_total[cid] = loss_sum_total / denom
                train_loss_components[cid] = {
                    k: v / denom for k, v in comp_sum_total.items()
                }
        weights = dict(client_samples)
        total = sum(weights.values())
        if total <= 0:
            weights = {cid: 1.0 for cid in client_states}
            total = float(len(weights))
        else:
            weights = {cid: w / total for cid, w in weights.items()}
        delta_global = {}
        for name, base_val in base_state_cpu.items():
            agg = None
            for cid, state in client_states.items():
                local_val = state.get(name)
                if local_val is None:
                    continue
                weight = weights.get(cid, 0.0)
                if weight <= 0:
                    continue
                diff = base_val - local_val
                agg = diff * weight if agg is None else agg + diff * weight
            if agg is not None:
                delta_global[name] = agg
            else:
                delta_global[name] = torch.zeros_like(base_val)
        v_bar = {}
        for name in base_state_cpu.keys():
            agg = None
            for cid, v_state in client_v_states.items():
                local_val = v_state.get(name)
                if local_val is None:
                    continue
                weight = weights.get(cid, 0.0)
                if weight <= 0:
                    continue
                agg = local_val * weight if agg is None else agg + local_val * weight
            if agg is not None:
                v_bar[name] = agg
            else:
                v_bar[name] = torch.zeros_like(base_state_cpu[name])
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
