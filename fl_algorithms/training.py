# References
# The FedAvg algorithm is implemented with reference to
# Communication-Efficient Learning of Deep Networks from Decentralized Data.
# https://arxiv.org/abs/1602.05629
# Reference repository: https://github.com/google-research/federated

import math
from copy import deepcopy
from typing import Dict, Iterable, List, Mapping
import torch
from scipy.stats import kendalltau, rankdata, spearmanr
from tqdm.auto import tqdm
from modules import ClientLossPool, build_optimizer


def safe_corr(fn, preds, targets):
    try:
        if len(preds) == 0 or len(targets) == 0:
            return 0.0
        if max(preds) == min(preds) or max(targets) == min(targets):
            return 0.0
        v, _ = fn(preds, targets)
        return float(v) if v == v else 0.0
    except Exception:
        return 0.0


def _is_ranking_batch(batch: Mapping[str, object]) -> bool:
    return (
        "ab_seq_1" in batch
        and "ag_seq_1" in batch
        and ("ab_seq_2" in batch)
        and ("ag_seq_2" in batch)
    )


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _ranking_prob_positive(
    mu_1: torch.Tensor,
    log_var_1: torch.Tensor,
    mu_2: torch.Tensor,
    log_var_2: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    var_1 = torch.exp(log_var_1).clamp_min(1e-08)
    var_2 = torch.exp(log_var_2).clamp_min(1e-08)
    std = torch.sqrt(var_1 + var_2 + 1e-08)
    z = (mu_1 - mu_2 - float(margin)) / std
    probs = _normal_cdf(z)
    return probs.clamp(1e-08, 1.0 - 1e-08)


def _binary_auc(labels_01: List[int], scores: List[float]) -> float:
    if not labels_01:
        return 0.0
    n_pos = sum(labels_01)
    n_neg = len(labels_01) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    sum_pos_ranks = 0.0
    for idx, label in enumerate(labels_01):
        if label == 1:
            sum_pos_ranks += float(ranks[idx])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _ranking_metrics_from_records(
    pred_records: List[Dict[str, object]],
) -> Dict[str, float]:
    if not pred_records:
        return {"auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    targets_pm1 = [
        1 if float(item.get("groundtruth", 0.0)) >= 0 else -1 for item in pred_records
    ]
    labels_01 = [1 if t > 0 else 0 for t in targets_pm1]
    probs = [float(item.get("predicted", 0.5)) for item in pred_records]
    preds_01 = [1 if p >= 0.5 else 0 for p in probs]
    total = len(labels_01)
    correct = sum((1 for y, p in zip(labels_01, preds_01) if y == p))
    tp = sum((1 for y, p in zip(labels_01, preds_01) if y == 1 and p == 1))
    fp = sum((1 for y, p in zip(labels_01, preds_01) if y == 0 and p == 1))
    fn = sum((1 for y, p in zip(labels_01, preds_01) if y == 1 and p == 0))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "auc": _binary_auc(labels_01, probs),
        "accuracy": float(correct) / float(total) if total > 0 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate_test_metrics(
    test_stats: Mapping[int, Mapping[str, object]],
    pred_records: List[Dict[str, object]] | None = None,
    mode: str = "sample-wise",
) -> Dict[str, float]:
    mode = str(mode or "sample-wise").lower()
    if mode not in {"client-wise", "sample-wise", "client-specific-sample-wise"}:
        raise ValueError(
            f"metric_compute must be 'client-wise', 'sample-wise', or 'client-specific-sample-wise', got '{mode}'."
        )
    if mode == "client-specific-sample-wise":
        mode = "sample-wise"
    valid_stats = [stats for stats in test_stats.values() if int(stats.get("n", 0)) > 0]
    total_samples = sum((int(stats.get("n", 0)) for stats in valid_stats))
    if mode == "client-wise":
        avg_metrics: Dict[str, float] = {"n": total_samples}
        for key in ("loss_total", "mse", "spearman", "kendall", "r2"):
            vals = [float(stats.get(key, 0.0)) for stats in valid_stats]
            avg_metrics[key] = sum(vals) / max(len(vals), 1)
        return avg_metrics
    loss_total = 0.0
    if total_samples > 0:
        loss_total = (
            sum(
                (
                    float(stats.get("loss_total", 0.0)) * int(stats.get("n", 0))
                    for stats in valid_stats
                )
            )
            / total_samples
        )
    if pred_records is None:
        pred_records = []
        for stats in valid_stats:
            pred_records.extend(stats.get("pred_records", []))
    preds = [float(item.get("predicted", 0.0)) for item in pred_records]
    targets = [float(item.get("groundtruth", 0.0)) for item in pred_records]
    mse = 0.0
    spearman = 0.0
    kendall = 0.0
    r2 = 0.0
    if preds:
        preds_tensor = torch.tensor(preds)
        targets_tensor = torch.tensor(targets)
        mse = float(torch.nn.MSELoss()(preds_tensor, targets_tensor).item())
        spearman = safe_corr(spearmanr, preds, targets)
        kendall = safe_corr(kendalltau, preds, targets)
        mean_targets = float(targets_tensor.mean().item())
        denom = float(torch.sum((targets_tensor - mean_targets) ** 2).item())
        if denom > 0:
            r2 = (
                1.0
                - float(torch.sum((preds_tensor - targets_tensor) ** 2).item()) / denom
            )
    return {
        "n": total_samples,
        "loss_total": loss_total,
        "mse": mse,
        "spearman": spearman,
        "kendall": kendall,
        "r2": r2,
    }


def aggregate_test_metrics_ranking(
    test_stats: Mapping[int, Mapping[str, object]],
    pred_records: List[Dict[str, object]] | None = None,
    mode: str = "sample-wise",
) -> Dict[str, float]:
    mode = str(mode or "sample-wise").lower()
    if mode not in {"client-wise", "sample-wise", "client-specific-sample-wise"}:
        raise ValueError(
            f"metric_compute must be 'client-wise', 'sample-wise', or 'client-specific-sample-wise', got '{mode}'."
        )
    if mode == "client-specific-sample-wise":
        mode = "sample-wise"
    valid_stats = [stats for stats in test_stats.values() if int(stats.get("n", 0)) > 0]
    total_samples = sum((int(stats.get("n", 0)) for stats in valid_stats))
    if mode == "client-wise":
        avg_metrics: Dict[str, float] = {"n": total_samples}
        for key in ("loss_total", "auc", "accuracy", "precision", "recall"):
            vals = [float(stats.get(key, 0.0)) for stats in valid_stats]
            avg_metrics[key] = sum(vals) / max(len(vals), 1)
        return avg_metrics
    loss_total = 0.0
    if total_samples > 0:
        loss_total = (
            sum(
                (
                    float(stats.get("loss_total", 0.0)) * int(stats.get("n", 0))
                    for stats in valid_stats
                )
            )
            / total_samples
        )
    if pred_records is None:
        pred_records = []
        for stats in valid_stats:
            pred_records.extend(stats.get("pred_records", []))
    rank_metrics = _ranking_metrics_from_records(pred_records)
    return {
        "n": total_samples,
        "loss_total": loss_total,
        "auc": rank_metrics["auc"],
        "accuracy": rank_metrics["accuracy"],
        "precision": rank_metrics["precision"],
        "recall": rank_metrics["recall"],
    }


def resolve_metric_compute(cfg: Mapping[str, object]) -> str:
    log_cfg = cfg.get("log") if isinstance(cfg.get("log"), dict) else {}
    mode = str(log_cfg.get("metric_compute", "sample-wise")).lower()
    if mode not in {"client-wise", "sample-wise"}:
        raise ValueError(
            f"log.metric_compute must be 'client-wise' or 'sample-wise', got '{mode}'."
        )
    return mode


def resolve_client_concurrency(
    cfg: Mapping[str, object], has_worker_loaders: bool, client_ids: Iterable[int]
) -> bool:
    if bool(cfg.get("no_concurrent_clients", False)):
        return False
    return len(list(client_ids)) > 1 and (not has_worker_loaders)


def fedavg_state_dicts(
    client_states: Mapping[int, Mapping[str, torch.Tensor]],
    client_weights: Mapping[int, float] | None = None,
) -> Dict[str, torch.Tensor]:
    if not client_states:
        return {}
    keys = set()
    for state in client_states.values():
        keys.update(state.keys())
    weight_map = {
        cid: float(client_weights.get(cid, 1.0) if client_weights else 1.0)
        for cid in client_states
    }
    weight_map = {cid: w for cid, w in weight_map.items() if w > 0}
    averaged: Dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        total = None
        denom = 0.0
        for cid, state in client_states.items():
            w = weight_map.get(cid)
            if w is None or k not in state:
                continue
            val = state[k]
            total = val * w if total is None else total + val * w
            denom += w
        if denom > 0 and total is not None:
            averaged[k] = total / denom
    return averaged


def normalize_batch_field(field, n: int):
    if isinstance(field, list):
        return field
    if torch.is_tensor(field):
        return field.detach().cpu().tolist()
    return [field] * n


def train_local(
    model,
    loader,
    loss_fn,
    optim,
    device,
    clip_norm: float,
    client_id: int,
    position: int = 0,
    local_epoch: int | None = None,
    local_epochs: int | None = None,
    desc_prefix: str = "train",
):
    model.train()
    loss_sum = 0.0
    comp_sum: Dict[str, float] = {}
    sample_count = 0
    preds: List[float] = []
    targets_all: List[float] = []
    desc = f"{desc_prefix}-c{client_id}"
    if local_epoch is not None and local_epochs is not None:
        desc = f"{desc_prefix}-c{client_id}-le{local_epoch}/{local_epochs}"
    for batch in tqdm(loader, desc=desc, position=position, leave=False):
        optim.zero_grad(set_to_none=True)
        target = batch["target"].to(device)
        batch_size = target.size(0)
        if _is_ranking_batch(batch):
            out_1 = model(batch["ab_seq_1"], batch["ag_seq_1"])
            out_2 = model(batch["ab_seq_2"], batch["ag_seq_2"])
            out = {
                "mu_1": out_1["mu"],
                "log_var_1": out_1["log_var"],
                "mu_2": out_2["mu"],
                "log_var_2": out_2["log_var"],
            }
            total_loss, comps = loss_fn(out, target)
            probs = _ranking_prob_positive(
                out["mu_1"], out["log_var_1"], out["mu_2"], out["log_var_2"]
            )
            preds.extend(probs.detach().cpu().flatten().tolist())
            targets_all.extend(target.detach().cpu().flatten().tolist())
        else:
            ab = batch["ab_seq"]
            ag = batch.get("ag_seq")
            out = model(ab, ag)
            total_loss, comps = loss_fn(out, target)
            preds.extend(out["mu"].detach().cpu().flatten().tolist())
            targets_all.extend(target.detach().cpu().flatten().tolist())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optim.step()
        loss_sum += total_loss.item() * batch_size
        for k, v in comps.items():
            comp_sum[k] = comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
        sample_count += batch_size
    return (loss_sum, comp_sum, sample_count, preds, targets_all)


def eval_local(
    model,
    loader,
    loss_fn,
    device,
    client_id: int,
    position: int = 0,
    desc_prefix: str = "eval",
    collect_records: bool = False,
):
    model.eval()
    mse_loss = torch.nn.MSELoss()
    loss_sum = 0.0
    comp_sum: Dict[str, float] = {}
    sample_count = 0
    preds: List[float] = []
    targets_all: List[float] = []
    pred_records: List[Dict[str, object]] = []
    ranking_mode_detected = False
    with torch.no_grad():
        for batch in tqdm(
            loader, desc=f"{desc_prefix}-c{client_id}", position=position, leave=False
        ):
            target = batch["target"].to(device)
            batch_size = target.size(0)
            ranking_mode = _is_ranking_batch(batch)
            ranking_mode_detected = ranking_mode_detected or ranking_mode
            if ranking_mode:
                out_1 = model(batch["ab_seq_1"], batch["ag_seq_1"])
                out_2 = model(batch["ab_seq_2"], batch["ag_seq_2"])
                out = {
                    "mu_1": out_1["mu"],
                    "log_var_1": out_1["log_var"],
                    "mu_2": out_2["mu"],
                    "log_var_2": out_2["log_var"],
                }
                total_loss, comps = loss_fn(outputs=out, targets=target)
                preds_batch = (
                    _ranking_prob_positive(
                        out["mu_1"], out["log_var_1"], out["mu_2"], out["log_var_2"]
                    )
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )
            else:
                ab = batch["ab_seq"]
                ag = batch.get("ag_seq")
                out = model(ab, ag)
                total_loss, comps = loss_fn(outputs=out, targets=target)
                preds_batch = out["mu"].detach().cpu().flatten().tolist()
            loss_sum += total_loss.item() * batch_size
            for k, v in comps.items():
                comp_sum[k] = (
                    comp_sum.get(k, 0.0) + float(v.detach().item()) * batch_size
                )
            sample_count += batch_size
            targets_batch = target.detach().cpu().flatten().tolist()
            preds.extend(preds_batch)
            targets_all.extend(targets_batch)
            if collect_records:
                client_ids = normalize_batch_field(
                    batch.get("arch_index", client_id), batch_size
                )
                sample_ids = normalize_batch_field(
                    batch.get("sample_id", -1), batch_size
                )
                antigen_ids = normalize_batch_field(
                    batch.get("antigen_id", ""), batch_size
                )
                meta_ids = normalize_batch_field(batch.get("meta_id", ""), batch_size)
                mutant_names = normalize_batch_field(
                    batch.get("mutant_name", ""), batch_size
                )
                for i in range(batch_size):
                    pred_records.append(
                        {
                            "client_id": int(client_ids[i]),
                            "sample_id": sample_ids[i],
                            "mutant_name": mutant_names[i],
                            "antigen_id": antigen_ids[i],
                            "meta_id": meta_ids[i],
                            "groundtruth": targets_batch[i],
                            "predicted": preds_batch[i],
                        }
                    )
    if sample_count == 0:
        return {
            "n": 0,
            "loss_total": 0.0,
            "mse": 0.0,
            "spearman": 0.0,
            "kendall": 0.0,
            "r2": 0.0,
            "loss_components": {},
            "pred_records": pred_records,
        }
    if ranking_mode_detected:
        if pred_records:
            rank_metrics = _ranking_metrics_from_records(pred_records)
        else:
            rank_metrics = _ranking_metrics_from_records(
                [
                    {"groundtruth": float(t), "predicted": float(p)}
                    for t, p in zip(targets_all, preds)
                ]
            )
        result = {
            "n": sample_count,
            "loss_total": loss_sum / sample_count,
            "auc": rank_metrics["auc"],
            "accuracy": rank_metrics["accuracy"],
            "precision": rank_metrics["precision"],
            "recall": rank_metrics["recall"],
            "loss_components": {k: v / sample_count for k, v in comp_sum.items()},
            "pred_records": pred_records,
        }
        if bool(getattr(model, "_serial_client_vram_mode", False)):
            if device.type == "cuda" and torch.cuda.is_available():
                model.to(torch.device("cpu"))
                torch.cuda.empty_cache()
        return result
    preds_tensor = torch.tensor(preds)
    targets_tensor = torch.tensor(targets_all)
    mse = float(mse_loss(preds_tensor, targets_tensor).item())
    spearman = safe_corr(spearmanr, preds, targets_all)
    kendall = safe_corr(kendalltau, preds, targets_all)
    mean_targets = float(targets_tensor.mean().item())
    denom = float(torch.sum((targets_tensor - mean_targets) ** 2).item())
    if denom > 0:
        r2 = 1.0 - float(torch.sum((preds_tensor - targets_tensor) ** 2).item()) / denom
    else:
        r2 = 0.0
    result = {
        "n": sample_count,
        "loss_total": loss_sum / sample_count,
        "mse": mse,
        "spearman": spearman,
        "kendall": kendall,
        "r2": r2,
        "loss_components": {k: v / sample_count for k, v in comp_sum.items()},
        "pred_records": pred_records,
    }
    if bool(getattr(model, "_serial_client_vram_mode", False)):
        if device.type == "cuda" and torch.cuda.is_available():
            model.to(torch.device("cpu"))
            torch.cuda.empty_cache()
    return result


def clone_model(
    proto: torch.nn.Module, state: Dict[str, torch.Tensor], device: torch.device
):
    serial_vram_mode = bool(getattr(proto, "_serial_client_vram_mode", False))
    if serial_vram_mode and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = deepcopy(proto)
    m.load_state_dict(state, strict=True)
    if serial_vram_mode:
        setattr(m, "_serial_client_vram_mode", True)
    return m.to(device)


def train_client_thread(
    cid: int,
    proto: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    train_loader,
    loss_pool: ClientLossPool,
    optim_cfg: Dict,
    local_epochs: int,
    client_device: torch.device,
    clip_norm: float,
    position: int,
    desc_prefix: str = "train",
):
    model = clone_model(proto, state, client_device)
    optimizer = build_optimizer(model.parameters(), optim_cfg)
    loss_fn = loss_pool.get(str(cid))
    sample_total = 0
    loss_sum_total = 0.0
    comp_sum_total: Dict[str, float] = {}
    for _ in range(local_epochs):
        local_epoch = _ + 1
        loss_sum, comp_sum, samples, _, _ = train_local(
            model,
            train_loader,
            loss_fn,
            optimizer,
            client_device,
            clip_norm,
            client_id=cid,
            position=position,
            local_epoch=local_epoch,
            local_epochs=local_epochs,
            desc_prefix=desc_prefix,
        )
        sample_total += samples
        loss_sum_total += loss_sum
        for k, v in comp_sum.items():
            comp_sum_total[k] = comp_sum_total.get(k, 0.0) + v
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return (cid, state_cpu, sample_total, loss_sum_total, comp_sum_total)


class FedAvgAggregator:

    def average(self, client_states, client_weights):
        return fedavg_state_dicts(client_states, client_weights)

    def apply_global(self, model, global_state):
        model.load_state_dict(global_state, strict=True)


def apply_aggregation(
    aggregator, global_model, client_states, client_samples, log_fn, round_id
):
    global_state = aggregator.average(client_states, client_samples)
    aggregator.apply_global(global_model, global_state)
    log_fn(f"[round {round_id}] aggregated {len(client_states)} clients")


def state_for_cid(state_map, cid, fallback_state):
    return fallback_state if state_map is None else state_map.get(cid, fallback_state)


def select_distribution_states(local_states, distribution_enabled):
    return None if distribution_enabled else local_states
