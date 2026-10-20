from torch import optim
from typing import Dict, Iterable


def build_optimizer(params: Iterable, cfg: Dict):
    name = cfg.get("name", "sgd").lower()
    lr = cfg.get("lr", 0.01)
    weight_decay = cfg.get("weight_decay", 0.0)
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError("optimizer got an empty parameter list (no trainable params).")
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    raise ValueError("Only SGD (no momentum) is supported.")
