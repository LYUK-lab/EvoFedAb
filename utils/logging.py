import datetime
import json
import logging
import os
import re
from utils.config import resolve_train_cfg


def setup_logging(base_dir: str = "logs", exp_name: str = "") -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        base_dir, f"{timestamp}_{exp_name}" if exp_name else timestamp
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def sanitize_log_tag(value: str) -> str:
    return re.sub("[^a-zA-Z0-9]+", "_", str(value)).strip("_") or "none"


def build_exp_name(cfg: dict) -> str:
    model_cfg, data_cfg = (cfg["model"], cfg["data"])
    train_cfg = resolve_train_cfg(cfg)
    esm_name = str(model_cfg["esm_path"]).rstrip("/").rsplit("/", 1)[-1]
    values = [
        cfg["fl_algorithm"],
        f"{cfg['dataset']}-{data_cfg['group']}",
        esm_name,
        "feature",
        model_cfg["head_type"],
        f"clients-{data_cfg['num_clients']}",
        f"split-{data_cfg['split_strategy']}",
        f"dist-{cfg.get('distribution', True)}",
        f"local-{train_cfg['local_epochs']}",
        cfg["mode"],
        cfg.get("exp_name", ""),
    ]
    return "_".join((sanitize_log_tag(value) for value in values if str(value)))


def create_logger() -> logging.Logger:
    logger = logging.getLogger("fedaff")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_json(path: str, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)
