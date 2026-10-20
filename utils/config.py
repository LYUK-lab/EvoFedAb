from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

SUPPORTED_DATASETS = ("abbibench", "abrank")
DEFAULT_DATASET = "abbibench"
DEFAULT_CONFIG_BY_DATASET = {
    "abbibench": "configs/abbibench.json",
    "abrank": "configs/abrank.json",
}


def normalize_dataset_name(dataset: str | None) -> str:
    name = str(dataset or "").strip().lower()
    if not name:
        return DEFAULT_DATASET
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}."
        )
    return name


def resolve_config_path(
    config_path: str | None = None, dataset: str | None = None
) -> str:
    if config_path:
        return str(config_path)
    name = normalize_dataset_name(dataset)
    return DEFAULT_CONFIG_BY_DATASET[name]


def resolve_cache_dir(data_cfg: Dict | None, dataset: str | None = None) -> Path:
    if isinstance(data_cfg, dict):
        cache_dir = data_cfg.get("cache_dir")
        if cache_dir:
            return Path(str(cache_dir))
        if dataset is None:
            dataset = data_cfg.get("dataset")
    name = normalize_dataset_name(dataset)
    return Path(f"data/{name}")


def apply_dataset_defaults(cfg: Dict, dataset: str | None = None) -> Dict:
    if not isinstance(cfg, dict):
        raise TypeError("config must be a dict.")
    name = normalize_dataset_name(dataset or cfg.get("dataset"))
    cfg["dataset"] = name
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        data_cfg = {}
        cfg["data"] = data_cfg
    data_cfg.setdefault("dataset", name)
    data_cfg.setdefault("cache_dir", str(resolve_cache_dir(data_cfg, name)))
    return cfg


def resolve_meta_key_split_cfg(data_cfg: Dict | None) -> Tuple[str, bool, bool, float]:
    if not isinstance(data_cfg, dict):
        return ("", False, False, 0.5)
    meta_cfg = data_cfg.get("meta-key")
    if not isinstance(meta_cfg, dict):
        return ("", False, False, 0.5)
    shared_cfg = meta_cfg.get("shared_key_equilibrium")
    if not isinstance(shared_cfg, dict):
        shared_cfg = {}
    dirichlet_cfg = meta_cfg.get("dirichlet")
    if not isinstance(dirichlet_cfg, dict):
        dirichlet_cfg = {}
    shared_enabled = bool(shared_cfg.get("enabled", False))
    shared_key = str(shared_cfg.get("key") or "").strip()
    dirichlet_enabled = bool(dirichlet_cfg.get("enabled", False))
    alpha_raw = dirichlet_cfg.get("alpha", 0.5)
    try:
        dirichlet_alpha = float(alpha_raw)
    except Exception as err:
        raise ValueError(
            f"data['meta-key'].dirichlet.alpha must be a float, got {alpha_raw!r}."
        ) from err
    if dirichlet_alpha <= 0:
        raise ValueError("data['meta-key'].dirichlet.alpha must be > 0.")
    if shared_enabled and dirichlet_enabled:
        raise ValueError(
            "data['meta-key']: shared_key_equilibrium.enabled and dirichlet.enabled cannot both be true."
        )
    if shared_enabled and (not shared_key):
        raise ValueError(
            "data['meta-key'].shared_key_equilibrium.key is required when enabled=true."
        )
    return (shared_key, shared_enabled, dirichlet_enabled, dirichlet_alpha)


def resolve_train_cfg(cfg: Dict, fl_algorithm: str | None = None) -> Dict:
    train_cfg = cfg.get("train")
    if not isinstance(train_cfg, dict):
        raise KeyError("config missing train dict.")
    algo = fl_algorithm or cfg.get("fl_algorithm")
    if not algo:
        raise KeyError("fl_algorithm is required (set via --fl-algorithm).")
    if algo not in train_cfg or not isinstance(train_cfg[algo], dict):
        raise KeyError(f"train config missing algorithm section '{algo}'.")
    return train_cfg[algo]
