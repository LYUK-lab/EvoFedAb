import argparse
import json
import os
import torch
from dataset import get_federated_dataloaders, prepare_esm_embeddings
from dataset.federated_dataset import embedding_cache_available
from models.factory import build_model
from fl_algorithms.training import FedAvgAggregator
from fl_algorithms.train_fedavg import train_fedavg
from fl_algorithms.train_fedavgM import train_fedavgM
from fl_algorithms.train_fedadamw import train_fedadamw
from fl_algorithms.train_fedprox import train_fedprox
from fl_algorithms.train_scaffold import train_scaffold
from fl_algorithms.train_feddisco import train_feddisco
from fl_algorithms.train_fedimpro import train_fedimpro
from fl_algorithms.train_fedmoswa import train_fedmoswa
from utils import set_seed
from utils.config import (
    SUPPORTED_DATASETS,
    apply_dataset_defaults,
    resolve_config_path,
)
from utils.esm import validate_esm_source
from utils.logging import build_exp_name, create_logger, save_json, setup_logging

TRAINERS = {
    "fedavg": train_fedavg,
    "fedavgm": train_fedavgM,
    "fedadamw": train_fedadamw,
    "fedprox": train_fedprox,
    "scaffold": train_scaffold,
    "feddisco": train_feddisco,
    "fedimpro": train_fedimpro,
    "fedmoswa": train_fedmoswa,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="EvoFedAffinity federated affinity learning"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, choices=SUPPORTED_DATASETS, default="abrank"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "preprocess"], default="train"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-concurrent", type=str, nargs="+", default=[])
    parser.add_argument(
        "--no-concurrent-clients",
        action="store_true",
        help="Run clients sequentially",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fl-algorithm", type=str, default="fedavg")
    parser.add_argument(
        "--feature-space-adaptor",
        type=str,
        default="mlp",
        choices=["mlp", "mint", "antibinder", "duadeep"],
    )
    return parser.parse_args()


def load_config(path: str):
    with open(path) as handle:
        return json.load(handle)


def _canonical_fl_algorithm(name: str) -> str:
    return {"fedavgm": "fedavgM", "fedadamw": "fedadamW"}.get(
        str(name).lower(), str(name).lower()
    )


def main():
    args = parse_args()
    cfg = apply_dataset_defaults(
        load_config(resolve_config_path(args.config, args.dataset)), args.dataset
    )
    cfg["mode"] = args.mode
    cfg["device"] = args.device
    cfg["fl_algorithm"] = _canonical_fl_algorithm(args.fl_algorithm)
    cfg["no_concurrent_clients"] = args.no_concurrent_clients
    cfg["model"]["head_type"] = args.feature_space_adaptor
    algorithm = str(args.fl_algorithm).lower()
    if algorithm not in TRAINERS:
        raise ValueError(f"Unsupported fl_algorithm '{algorithm}'.")
    no_esm = str(cfg["model"]["esm_path"]).strip().lower() == "no-esm"
    if not no_esm:
        validate_esm_source(cfg["model"]["esm_path"])
    logger = create_logger()
    dev_str = cfg["device"]
    if dev_str.startswith("cuda") and (not torch.cuda.is_available()):
        logger.info("Requested device %s but CUDA is unavailable; using CPU.", dev_str)
        dev_str = "cpu"
    device = torch.device(dev_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(cfg.get("seed", 0))
    if args.mode == "preprocess":
        if no_esm:
            logger.info(
                "no-esm uses raw sequences; embedding preprocessing is unnecessary."
            )
            return
        if embedding_cache_available(cfg):
            logger.info(
                "ESM embedding cache exists for group %s.", cfg["data"]["group"]
            )
        else:
            path = prepare_esm_embeddings(cfg, device=device)
            logger.info("Saved embeddings to %s.", path)
        return
    loaders = get_federated_dataloaders(cfg, args.mode)
    model = build_model(cfg["model"], embedding_dim=loaders["embedding_dim"])
    serial = args.mode == "train" and cfg["no_concurrent_clients"]
    if serial:
        model = model.to("cpu")
        model._serial_client_vram_mode = True
    else:
        model = model.to(device)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
    device_pool = [device]
    if not cfg["no_concurrent_clients"]:
        for dev_str_conc in args.device_concurrent:
            try:
                dev_obj = torch.device(dev_str_conc)
                if dev_obj not in device_pool:
                    device_pool.append(dev_obj)
            except (RuntimeError, ValueError):
                continue
    log_dir = setup_logging(exp_name=build_exp_name(cfg))
    save_json(os.path.join(log_dir, "configs.json"), cfg)
    TRAINERS[algorithm](
        model,
        loaders["train"],
        cfg,
        device,
        device_pool,
        FedAvgAggregator(),
        logger=logger,
        test_loader=loaders["test"],
    )
    torch.save(model.state_dict(), os.path.join(log_dir, "final_model.pt"))
    logger.info("Saved model to %s.", os.path.join(log_dir, "final_model.pt"))


if __name__ == "__main__":
    main()
