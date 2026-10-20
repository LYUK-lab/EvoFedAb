from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from utils.config import (
    normalize_dataset_name,
    resolve_cache_dir,
    resolve_meta_key_split_cfg,
)
from utils.data_utils import split_records
from .abrank_ranking import (
    AbRankRankingPairDataset,
    collate_abrank_ranking_batch,
    drop_largest_source_for_ranking,
    is_abrank_ranking_task,
    load_or_build_abrank_ranking_pairs,
    split_abrank_ranking_pairs_iid,
)


def _sanitize_name(name: str) -> str:
    safe = re.sub("[^a-zA-Z0-9]+", "_", str(name))
    safe = safe.strip("_")
    return safe or "default"


def _extract_esm_name(model_cfg: Dict) -> str:
    esm_path = model_cfg["esm_path"]
    esm_name = str(esm_path).rstrip("/").rsplit("/", 1)[-1]
    return _sanitize_name(esm_name or "unknown")


def _embedding_cache_path(data_cfg: Dict, model_cfg: Dict) -> Path:
    dataset_name = data_cfg["dataset_name"]
    group = str(data_cfg["group"])
    if data_cfg.get("dataset") == "abrank" or dataset_name == "AbRank_dataset":
        from .abrank import resolve_group

        group = resolve_group(data_cfg)
    cache_dir = resolve_cache_dir(data_cfg)
    esm_name = _extract_esm_name(model_cfg)
    fname = f"esm_embeddings_{_sanitize_name(dataset_name)}_{_sanitize_name(group)}_{_sanitize_name(esm_name)}.pt"
    return cache_dir / "embeddings" / fname


def _load_records(
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict], Path, str]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    dataset_name = normalize_dataset_name(cfg.get("dataset") or data_cfg.get("dataset"))
    if dataset_name == "abbibench":
        from .abbibench import load_records
    elif dataset_name == "abrank":
        from .abrank import load_records
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    return load_records(cfg)


class EmbeddingCache:

    def __init__(self, payload: Dict, path: Path, expected_meta: Optional[Dict] = None):
        self.path = path
        self.meta = payload.get("meta", {})
        self.ab_embeddings = payload.get("ab_embeddings", [])
        self.ag_embeddings = payload.get("ag_embeddings", [])
        if expected_meta:
            self._validate_meta(expected_meta)

    def _validate_meta(self, expected_meta: Dict):
        ds_name = expected_meta.get("dataset_name")
        group = expected_meta.get("group")
        esm_name = expected_meta.get("esm_name")
        preprocessing_version = expected_meta.get("preprocessing_version")
        if (
            preprocessing_version
            and self.meta.get("preprocessing_version") != preprocessing_version
        ):
            raise ValueError(
                "Embedding cache preprocessing mismatch; regenerate embeddings with preprocess mode."
            )
        if (
            ds_name
            and self.meta.get("dataset_name")
            and (self.meta.get("dataset_name") != ds_name)
        ):
            raise ValueError(
                f"Embedding cache dataset mismatch: expected {ds_name}, got {self.meta.get('dataset_name')}"
            )
        if (
            group
            and self.meta.get("group")
            and (str(self.meta.get("group")).lower() != str(group).lower())
        ):
            raise ValueError(
                f"Embedding cache group mismatch: expected {group}, got {self.meta.get('group')}"
            )
        if (
            esm_name
            and self.meta.get("esm_name")
            and (self.meta.get("esm_name") != esm_name)
        ):
            raise ValueError(
                f"Embedding cache ESM mismatch: expected {esm_name}, got {self.meta.get('esm_name')}"
            )

    def get(self, sample_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            ab = self.ab_embeddings[sample_id]
            ag = self.ag_embeddings[sample_id]
        except Exception as err:
            raise KeyError(
                f"Missing embeddings for sample_id={sample_id} in {self.path}"
            ) from err
        if ab is None or ag is None:
            raise KeyError(
                f"Missing embeddings for sample_id={sample_id} in {self.path}"
            )
        return (ab, ag)


_EMBEDDING_CACHE_REGISTRY: Dict[str, EmbeddingCache] = {}


def _get_embedding_cache(path: Path, expected_meta: Optional[Dict]) -> EmbeddingCache:
    key = str(path.resolve())
    cache = _EMBEDDING_CACHE_REGISTRY.get(key)
    if cache is None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Embedding cache is missing: {path}. Run preprocess mode first."
            )
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        cache = EmbeddingCache(payload, path, expected_meta)
        _EMBEDDING_CACHE_REGISTRY[key] = cache
    elif expected_meta:
        cache._validate_meta(expected_meta)
    return cache


def _build_client_loaders_ranking(
    records: List[Dict],
    num_clients: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    embedding_path: Optional[Path] = None,
) -> ClientDataLoaders:
    per_client_records: Dict[int, List[Dict]] = {}
    for rec in records:
        cid = int(rec["arch_index"])
        per_client_records.setdefault(cid, []).append(rec)
    per_client_records = {cid: recs for cid, recs in per_client_records.items() if recs}
    client_counts: Dict[int, int] = {
        cid: len(recs) for cid, recs in per_client_records.items()
    }
    if not client_counts:
        raise ValueError("No valid ranking client records found.")
    bs_uniform = max(1, int(batch_size))
    batch_sizes = {cid: bs_uniform for cid in client_counts.keys()}
    steps_per_client = {
        cid: (count + bs_uniform - 1) // bs_uniform
        for cid, count in client_counts.items()
    }
    loaders: Dict[int, DataLoader] = {}
    for cid, recs in per_client_records.items():
        if shuffle:
            rng = np.random.default_rng(seed + cid)
            rng.shuffle(recs)
        ds = AbRankRankingPairDataset(
            recs, num_clients=num_clients, embedding_path=embedding_path
        )
        generator = torch.Generator().manual_seed(seed + cid)
        loaders[cid] = DataLoader(
            ds,
            batch_size=batch_sizes[cid],
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_abrank_ranking_batch,
            generator=generator,
        )
    return ClientDataLoaders(loaders, client_counts, batch_sizes, steps_per_client)


def prepare_esm_embeddings(cfg: Dict, device: Optional[torch.device] = None) -> Path:
    _canonical_group(cfg)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    esm_path = model_cfg["esm_path"]
    dataset_name = normalize_dataset_name(cfg.get("dataset") or data_cfg.get("dataset"))
    _, valid_records, metadata, _, group = _load_records(cfg)
    if not valid_records:
        raise RuntimeError("No valid records found for preprocessing embeddings.")
    embedding_path = _embedding_cache_path(data_cfg, model_cfg)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    if device is None:
        dev_str = cfg.get("device") or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if str(dev_str).startswith("cuda") and (not torch.cuda.is_available()):
            print(
                f"Requested device {dev_str} but CUDA not available, falling back to CPU."
            )
            dev_str = "cpu"
        device = torch.device(dev_str)
    from modules import FrozenESMEncoder

    encoder = FrozenESMEncoder(esm_path).to(device)
    encoder.eval()
    meta_lookup = {str(k).lower(): v for k, v in metadata.items()}
    meta_key_lower = {str(k).lower(): k for k in metadata.keys()}
    meta_base_lower = {str(k).lower().split("_")[0]: k for k in metadata.keys()}
    seen = set()
    max_id = max((int(r.get("sample_id", 0)) for r in valid_records))
    ab_embeddings: List[Optional[torch.Tensor]] = [None] * (max_id + 1)
    ag_embeddings: List[Optional[torch.Tensor]] = [None] * (max_id + 1)
    seq_cache: Dict[str, torch.Tensor] = {}

    def _resolve_meta_key(rec: Dict) -> str:
        meta_id = str(rec.get("meta_id") or "").strip().lower()
        if meta_id:
            return meta_id
        if dataset_name == "abbibench":
            ag = str(rec.get("antigen_id") or "").lower()
            base_ag = ag.split("_")[0]
            meta_key = meta_key_lower.get(ag) or meta_base_lower.get(base_ag) or ag
            return str(meta_key).lower()
        return "unknown"

    def _encode_sequence(seq: str) -> torch.Tensor:
        cached = seq_cache.get(seq)
        if cached is not None:
            return cached
        tok_repr, _ = encoder.encode([seq])
        emb = tok_repr.squeeze(0).cpu()
        seq_cache[seq] = emb
        return emb

    with torch.no_grad():
        for rec in tqdm(valid_records, desc="ESM embedding", unit="pair"):
            meta_key = _resolve_meta_key(rec)
            if meta_key and meta_key not in seen:
                meta = meta_lookup.get(meta_key, {})
                name = (
                    meta.get("antigen_id")
                    or meta.get("name")
                    or meta.get("id")
                    or meta_key
                )
                tqdm.write(f"preparing embedding for {name} using {esm_path}")
                seen.add(meta_key)
            sample_id = int(rec.get("sample_id", -1))
            if sample_id < 0:
                continue
            ab_seq = rec.get("antibody_seq", "") or ""
            ag_seq = rec.get("antigen_seq", "") or ""
            ab_embeddings[sample_id] = _encode_sequence(ab_seq)
            ag_embeddings[sample_id] = _encode_sequence(ag_seq)
    payload = {
        "meta": {
            "dataset_name": data_cfg["dataset_name"],
            "group": group,
            "esm_path": esm_path,
            "esm_name": _extract_esm_name(model_cfg),
            "num_records": len(valid_records),
            "embedding_dim": int(encoder.hidden_dim),
        },
        "ab_embeddings": ab_embeddings,
        "ag_embeddings": ag_embeddings,
    }
    if dataset_name == "abrank":
        from .abrank import PREPROCESSING_VERSION

        payload["meta"]["preprocessing_version"] = PREPROCESSING_VERSION
    torch.save(payload, embedding_path)
    if dataset_name == "abrank":
        from .abrank_ranking import _EMBEDDING_CACHE

        key = str(embedding_path.resolve())
        _EMBEDDING_CACHE_REGISTRY.pop(key, None)
        _EMBEDDING_CACHE.pop(key, None)
    return embedding_path


class ClientDataLoaders:

    def __init__(self, loaders, client_counts, batch_sizes, steps_per_client=None):
        self.loaders = loaders
        self.client_counts = client_counts
        self.batch_sizes = batch_sizes
        self.steps_per_client = steps_per_client or {}

    def __getitem__(self, key):
        return self.loaders[key]


class FedAffinityTorchDataset(Dataset):

    def __init__(self, records, num_clients, embedding_path=None, embedding_meta=None):
        self.records = records
        self.num_clients = num_clients
        self._embedding_cache = (
            _get_embedding_cache(embedding_path, embedding_meta)
            if embedding_path is not None
            else None
        )
        self.client_counts = {}
        for record in records:
            cid = int(record["arch_index"])
            self.client_counts[cid] = self.client_counts.get(cid, 0) + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        ab = record.get("antibody_seq", "")
        ag = record.get("antigen_seq", "")
        if self._embedding_cache is not None:
            ab, ag = self._embedding_cache.get(int(record["sample_id"]))
        return {
            "ab_seq": ab,
            "ag_seq": ag,
            "target": torch.tensor(record["binding_score"], dtype=torch.float),
            "arch_index": torch.tensor(record["arch_index"], dtype=torch.long),
            "sample_id": record["sample_id"],
            "antigen_id": record.get("antigen_id", ""),
            "meta_id": record.get("meta_id", ""),
            "mutant_name": record.get("mutant_name", ""),
        }


def _collate_federated_batch(batch):
    result = {
        key: [record[key] for record in batch]
        for key in ["sample_id", "antigen_id", "meta_id", "mutant_name"]
    }
    result["target"] = torch.stack([record["target"] for record in batch])
    result["arch_index"] = torch.stack([record["arch_index"] for record in batch])
    for key in ["ab_seq", "ag_seq"]:
        items = [record[key] for record in batch]
        result[key] = (
            pad_sequence(items, batch_first=True)
            if items and torch.is_tensor(items[0])
            else items
        )
    return result


def _build_client_loaders(
    records,
    num_clients,
    batch_size,
    num_workers,
    shuffle,
    seed,
    embedding_path=None,
    embedding_meta=None,
):
    per_client = {}
    for record in records:
        per_client.setdefault(int(record["arch_index"]), []).append(record)
    if not per_client:
        raise ValueError("No valid client records found.")
    counts = {cid: len(rows) for cid, rows in per_client.items()}
    loaders = {}
    for cid, rows in per_client.items():
        if shuffle:
            np.random.default_rng(seed + cid).shuffle(rows)
        dataset = FedAffinityTorchDataset(
            rows, num_clients, embedding_path, embedding_meta
        )
        loaders[cid] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=_collate_federated_batch,
            generator=torch.Generator().manual_seed(seed + cid),
        )
    sizes = {cid: batch_size for cid in counts}
    steps = {cid: len(loader) for cid, loader in loaders.items()}
    return ClientDataLoaders(loaders, counts, sizes, steps)


def _canonical_group(cfg):
    if cfg["dataset"] == "abrank":
        from .abrank import resolve_group

        group = resolve_group(cfg["data"])
    else:
        group = str(cfg["data"]["group"]).lower()
        if group not in {"pkd", "lenrich"}:
            raise ValueError("AbBibench data.group must be 'pkd' or 'lenrich'.")
    cfg["data"]["group"] = group.lower()
    return group


def embedding_cache_available(cfg):
    _canonical_group(cfg)
    return _embedding_cache_path(cfg["data"], cfg["model"]).is_file()


def get_federated_dataloaders(cfg: Dict, mode: str):
    if mode != "train":
        raise ValueError("Federated loaders are used only in train mode.")
    data_cfg = cfg["data"]
    group = _canonical_group(cfg)
    batch_size, num_clients = (
        int(data_cfg["batch_size"]),
        int(data_cfg["num_clients"]),
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 0))
    if batch_size <= 0 or num_clients <= 0 or num_workers < 0:
        raise ValueError(
            "batch_size and num_clients must be positive; num_workers must be nonnegative."
        )
    if not 0 < float(data_cfg["test_ratio"]) < 1:
        raise ValueError("data.test_ratio must be between 0 and 1.")
    if num_clients > 1:
        num_workers = 0
    task = str(cfg.get("task", "regression")).lower()
    if task not in {"regression", "ranking"}:
        raise ValueError("task must be 'regression' or 'ranking'.")
    ranking = is_abrank_ranking_task(cfg)
    if task == "ranking" and (not ranking):
        raise ValueError("Ranking is supported only for AbRank.")
    strategy = str(data_cfg["split_strategy"]).lower().replace("-", "_")
    if ranking and (
        str(cfg["fl_algorithm"]).lower() != "fedavg" or strategy != "iid"
    ):
        raise ValueError(
            "AbRank ranking requires FedAvg and an IID split."
        )
    if (
        ranking
        and cfg.get("log", {}).get("metric_compute", "sample-wise") != "sample-wise"
    ):
        raise ValueError("AbRank ranking requires sample-wise metrics.")
    _, records, _, cache_dir, group = _load_records(cfg)
    use_embeddings = str(cfg["model"]["esm_path"]).strip().lower() != "no-esm"
    path = None
    dim = None
    expected_meta = None
    if use_embeddings:
        path = _embedding_cache_path(data_cfg, cfg["model"])
        if not embedding_cache_available(cfg):
            raise FileNotFoundError(
                f"Preprocessed embeddings not found at {path}; run preprocess mode first."
            )
        expected_meta = {
            "dataset_name": data_cfg["dataset_name"],
            "group": group,
            "esm_name": _extract_esm_name(cfg["model"]),
        }
        if cfg["dataset"] == "abrank":
            from .abrank import PREPROCESSING_VERSION

            expected_meta["preprocessing_version"] = PREPROCESSING_VERSION
        cache = _get_embedding_cache(path, expected_meta)
        from utils.esm import load_esm_tokenizer

        tokenizer = load_esm_tokenizer(cfg["model"]["esm_path"])
        dim = int(cache.meta["embedding_dim"])
        for record in records:
            ab, ag = cache.get(int(record["sample_id"]))
            if (
                ab.ndim != 2
                or ag.ndim != 2
                or ab.shape[-1] != dim
                or (ag.shape[-1] != dim)
            ):
                raise ValueError("Invalid embedding dimensions; regenerate embeddings.")
        for record in records[:8]:
            ab, ag = cache.get(int(record["sample_id"]))
            for sequence, embedding in [
                (record["antibody_seq"], ab),
                (record["antigen_seq"], ag),
            ]:
                length = len(tokenizer(sequence, add_special_tokens=True)["input_ids"])
                if embedding.shape[0] != length:
                    raise ValueError(
                        "Embedding/sequence length mismatch; regenerate embeddings."
                    )
    if ranking:
        ranking_records, _ = drop_largest_source_for_ranking(records)
        pair_bins = load_or_build_abrank_ranking_pairs(
            ranking_records,
            cache_dir=cache_dir,
            min_label_gap=float(cfg["ranking"]["min_label_gap"]),
            group=group,
        )
        train_records, test_records, _ = split_abrank_ranking_pairs_iid(
            pair_bins,
            num_clients=num_clients,
            test_ratio=float(data_cfg["test_ratio"]),
            seed=seed,
        )
        build = _build_client_loaders_ranking
        kwargs = dict(embedding_path=path)
    else:
        shared_key, shared, dirichlet, alpha = resolve_meta_key_split_cfg(data_cfg)
        train_records, test_records = split_records(
            records,
            num_clients=num_clients,
            test_ratio=float(data_cfg["test_ratio"]),
            seed=seed,
            split_strategy=strategy,
            shared_key=shared_key,
            enable_shared_key_equilibrium=shared,
            dirichlet_enabled=dirichlet,
            dirichlet_alpha=alpha,
        )
        build = _build_client_loaders
        kwargs = dict(embedding_path=path, embedding_meta=expected_meta)
    train = build(
        train_records, num_clients, batch_size, num_workers, True, seed, **kwargs
    )
    test = build(
        test_records, num_clients, batch_size, num_workers, False, seed, **kwargs
    )
    if set(train.loaders) != set(test.loaders):
        raise ValueError(
            "Each client must have train and test samples; reduce num_clients or increase test_ratio."
        )
    return {"train": train, "test": test, "embedding_dim": dim}
