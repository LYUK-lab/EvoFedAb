from typing import Callable, Dict, Union
import math
import torch
from torch.nn import functional as F


def _mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(preds, targets)


def _l1_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(preds, targets)


_BASE_LOSSES: Dict[str, Callable] = {
    "mse": _mse_loss,
    "l1": _l1_loss,
    "gaussian_nll": "gaussian_nll",
    "gaussian_nll_ranking": "gaussian_nll_ranking",
}


def _normalize_loss_cfg(loss_cfg: Union[str, Dict]) -> Dict[str, float]:
    if isinstance(loss_cfg, str):
        raw = str(loss_cfg).strip()
        if ":" in raw:
            name, weight_raw = raw.split(":", 1)
            name = name.strip().lower()
            weight_text = weight_raw.replace(" ", "")
            weight = float(weight_text) if weight_text else 1.0
            return {name: weight}
        return {raw.lower(): 1.0}
    if isinstance(loss_cfg, dict):
        if "name" in loss_cfg:
            name = str(loss_cfg["name"]).lower()
            weight = float(loss_cfg.get("weight", 1.0))
            return {name: weight}
        normalized: Dict[str, float] = {}
        for name, cfg in loss_cfg.items():
            weight = cfg.get("weight", 1.0) if isinstance(cfg, dict) else cfg
            normalized[str(name).lower()] = float(weight)
        if not normalized:
            raise ValueError("Loss config is empty; specify at least one loss.")
        return normalized
    raise ValueError(f"Unsupported loss config type: {type(loss_cfg)}")


class WeightedLoss:

    def __init__(self, components: Dict[str, float]):
        if not components:
            raise ValueError("WeightedLoss requires at least one loss component.")
        self.components: Dict[str, Dict[str, object]] = {}
        for name, weight in components.items():
            if name not in _BASE_LOSSES:
                raise ValueError(f"Unknown loss fn: {name}")
            self.components[name] = {"fn": _BASE_LOSSES[name], "weight": float(weight)}

    def __call__(self, outputs, targets: torch.Tensor):
        device = targets.device
        dtype = targets.dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)
        raw: Dict[str, torch.Tensor] = {}
        for name, comp in self.components.items():
            fn = comp["fn"]
            weight: float = comp["weight"]
            if weight <= 0.0:
                raw[name] = torch.tensor(0.0, device=device, dtype=dtype)
                continue
            if name == "gaussian_nll":
                if not isinstance(outputs, dict):
                    raise ValueError(
                        "gaussian_nll expects outputs with 'mu' and 'log_var'."
                    )
                mu = outputs.get("mu")
                log_var = outputs.get("log_var")
                if mu is None or log_var is None:
                    raise ValueError(
                        "gaussian_nll requires outputs['mu'] and outputs['log_var']."
                    )
                var = torch.exp(log_var)
                value = F.gaussian_nll_loss(mu, targets, var, reduction="mean")
            elif name == "gaussian_nll_ranking":
                if not isinstance(outputs, dict):
                    raise ValueError(
                        "gaussian_nll_ranking expects outputs with 'mu_1', 'log_var_1', 'mu_2', and 'log_var_2'."
                    )
                mu_1 = outputs.get("mu_1")
                log_var_1 = outputs.get("log_var_1")
                mu_2 = outputs.get("mu_2")
                log_var_2 = outputs.get("log_var_2")
                if (
                    mu_1 is None
                    or log_var_1 is None
                    or mu_2 is None
                    or (log_var_2 is None)
                ):
                    raise ValueError(
                        "gaussian_nll_ranking requires outputs['mu_1'], outputs['log_var_1'], outputs['mu_2'], and outputs['log_var_2']."
                    )
                y = torch.where(
                    targets >= 0, torch.ones_like(targets), -torch.ones_like(targets)
                )
                var_1 = torch.exp(log_var_1).clamp_min(1e-08)
                var_2 = torch.exp(log_var_2).clamp_min(1e-08)
                std = torch.sqrt(var_1 + var_2 + 1e-08)
                margin = outputs.get("ranking_margin", 0.0)
                if torch.is_tensor(margin):
                    margin_t = margin.to(device=std.device, dtype=std.dtype)
                else:
                    margin_t = torch.tensor(
                        float(margin), device=std.device, dtype=std.dtype
                    )
                z = (y * (mu_1 - mu_2) - margin_t) / std
                phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
                phi = torch.clamp(phi, min=1e-12, max=1.0)
                value = -torch.log(phi).mean()
            else:
                preds = (
                    outputs
                    if not isinstance(outputs, dict)
                    else outputs.get("pred", outputs.get("mu"))
                )
                if preds is None:
                    raise ValueError(f"Loss {name} expects 'pred' or 'mu' in outputs.")
                value = fn(preds, targets)
            raw[name] = value
            total = total + weight * value
        return (total, raw)


def get_loss_fn(loss_cfg: Union[str, Dict]) -> WeightedLoss:
    components = _normalize_loss_cfg(loss_cfg)
    return WeightedLoss(components)


class ClientLossPool:

    def __init__(self, loss_cfg: Union[str, Dict]):
        self.loss_cfg = loss_cfg
        self._cache: Dict[str, WeightedLoss] = {}

    def get(self, client_id: str) -> WeightedLoss:
        key = str(client_id)
        if key not in self._cache:
            self._cache[key] = get_loss_fn(self.loss_cfg)
        return self._cache[key]
