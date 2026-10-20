from __future__ import annotations
import hashlib
import json
import math
from decimal import Decimal
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

_EMBEDDING_CACHE: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}


def _resolve_task(cfg: Dict) -> str:
    task = cfg.get("task", "regression") if isinstance(cfg, dict) else "regression"
    return str(task or "regression").strip().lower()


def is_abrank_ranking_task(cfg: Dict) -> bool:
    dataset = str((cfg or {}).get("dataset", "")).strip().lower()
    return dataset == "abrank" and _resolve_task(cfg) == "ranking"


def _load_embedding_arrays(
    embedding_path: Path,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    key = str(embedding_path.resolve())
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    from .federated_dataset import _get_embedding_cache

    cache = _get_embedding_cache(embedding_path, None)
    payload = {
        "ab_embeddings": cache.ab_embeddings,
        "ag_embeddings": cache.ag_embeddings,
    }
    ab_embeddings = payload.get("ab_embeddings", [])
    ag_embeddings = payload.get("ag_embeddings", [])
    if not isinstance(ab_embeddings, list) or not isinstance(ag_embeddings, list):
        raise ValueError(f"Invalid embedding cache format: {embedding_path}")
    _EMBEDDING_CACHE[key] = (ab_embeddings, ag_embeddings)
    return _EMBEDDING_CACHE[key]


def _get_pair_embedding(
    embedding_path: Path, sample_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    ab_embeddings, ag_embeddings = _load_embedding_arrays(embedding_path)
    try:
        ab = ab_embeddings[sample_id]
        ag = ag_embeddings[sample_id]
    except Exception as err:
        raise KeyError(
            f"Missing pair embedding for sample_id={sample_id} in {embedding_path}"
        ) from err
    if ab is None or ag is None:
        raise KeyError(
            f"Missing pair embedding for sample_id={sample_id} in {embedding_path}"
        )
    return (ab, ag)


def _clone_pair_record(pair: Dict, client_id: int) -> Dict:
    rec = dict(pair)
    rec["arch_index"] = int(client_id)
    rec["binding_score"] = float(rec.get("pair_label", rec.get("binding_score", 0.0)))
    return rec


def _stratified_client_assign(
    rows: List[Dict], num_clients: int, rng: np.random.Generator
) -> Tuple[Dict[int, List[Dict]], int]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    usable = len(shuffled) // num_clients * num_clients
    dropped = len(shuffled) - usable
    shuffled = shuffled[:usable]
    per_client: Dict[int, List[Dict]] = {cid: [] for cid in range(num_clients)}
    if usable == 0:
        return (per_client, dropped)
    chunk = usable // num_clients
    for cid in range(num_clients):
        begin = cid * chunk
        end = begin + chunk
        for pair in shuffled[begin:end]:
            per_client[cid].append(_clone_pair_record(pair, client_id=cid))
    return (per_client, dropped)


def _split_train_test(
    rows: List[Dict], test_ratio: float
) -> Tuple[List[Dict], List[Dict]]:
    n_total = len(rows)
    if n_total <= 1:
        return (list(rows), [])
    n_test = int(round(float(n_total) * float(test_ratio)))
    n_test = max(1, min(n_test, n_total - 1))
    test_rows = list(rows[:n_test])
    train_rows = list(rows[n_test:])
    return (train_rows, test_rows)


def build_abrank_ranking_pairs(
    records: List[Dict], min_label_gap: float = 1.0, *, group: str
) -> Dict[str, object]:
    from .abrank import resolve_group

    group = resolve_group({"group": group})
    if not math.isfinite(min_label_gap) or min_label_gap <= 0:
        raise ValueError(f"min_label_gap must be finite and > 0, got {min_label_gap}.")
    gap = Decimal(str(min_label_gap))
    scores = [Decimal(str(r["binding_score"])) for r in records]
    if any((not score.is_finite() for score in scores)):
        raise ValueError("Ranking requires finite labels.")
    pos_pairs: List[Dict] = []
    neg_pairs: List[Dict] = []
    dropped_gap = 0
    pair_id = 0
    total_n = len(records)
    with tqdm(
        total=total_n, desc="AbRank ranking pair filtering", unit="sample", leave=False
    ) as pbar:
        for i in range(total_n):
            rec_i = records[i]
            score_i = scores[i]
            sid_i = int(rec_i.get("sample_id", i))
            for j in range(i + 1, len(records)):
                rec_j = records[j]
                score_j = scores[j]
                if abs(score_i - score_j) <= gap:
                    dropped_gap += 1
                    continue
                stronger = score_i > score_j
                label = 1.0 if stronger else -1.0
                rec_1, rec_2 = (rec_i, rec_j)
                sid_1, sid_2 = (sid_i, int(rec_j.get("sample_id", j)))
                if pair_id % 2 == 1:
                    rec_1, rec_2 = (rec_2, rec_1)
                    sid_1, sid_2 = (sid_2, sid_1)
                    label = -label
                pair = {
                    "pair_id": pair_id,
                    "sample_id": pair_id,
                    "sample_id_1": sid_1,
                    "sample_id_2": sid_2,
                    "pair_label": float(label),
                    "binding_score": float(label),
                    "antibody_seq_1": rec_1.get("antibody_seq", ""),
                    "antigen_seq_1": rec_1.get("antigen_seq", ""),
                    "antibody_seq_2": rec_2.get("antibody_seq", ""),
                    "antigen_seq_2": rec_2.get("antigen_seq", ""),
                    "antibody_seq": rec_1.get("antibody_seq", ""),
                    "antigen_seq": rec_1.get("antigen_seq", ""),
                    "antigen_id": f"{rec_1.get('antigen_id', '')}|{rec_2.get('antigen_id', '')}",
                    "meta_id": f"{rec_1.get('meta_id', '')}|{rec_2.get('meta_id', '')}",
                    "mutant_name": f"pair_{pair_id}",
                }
                pair_id += 1
                if label > 0:
                    pos_pairs.append(pair)
                else:
                    neg_pairs.append(pair)
            pbar.update(1)
            pbar.set_postfix(
                invalid=dropped_gap, positive=len(pos_pairs), negative=len(neg_pairs)
            )
    return {"positive": pos_pairs, "negative": neg_pairs, "dropped_gap": dropped_gap}


def drop_largest_source_for_ranking(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    if not records:
        raise ValueError("No records available for ranking Source filtering.")
    counts = Counter((rec["meta_id"] for rec in records))
    source = min(counts, key=lambda key: (-counts[key], key))
    retained = [rec for rec in records if rec["meta_id"] != source]
    if not retained:
        raise ValueError("Removing the largest Source leaves no ranking records.")
    return (
        retained,
        {
            "largest_source": source,
            "removed_samples": counts[source],
            "total_after": len(retained),
            "sources_after": len(counts) - 1,
        },
    )


def _records_signature(records: List[Dict]) -> str:
    hasher = hashlib.sha256()
    for rec in records:
        identity = (
            rec["sample_id"],
            rec["binding_score"],
            rec.get("meta_id", ""),
            rec.get("antibody_seq", ""),
            rec.get("antigen_seq", ""),
        )
        hasher.update(json.dumps(identity, separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _pair_cache_path(cache_dir: Path, min_label_gap: float, group: str) -> Path:
    gap_text = str(min_label_gap).replace(".", "_")
    return cache_dir / "pairs" / f"abrank_ranking_{group}_gap_{gap_text}.pt"


def load_or_build_abrank_ranking_pairs(
    records: List[Dict], cache_dir: Path, min_label_gap: float = 1.0, *, group: str
) -> Dict[str, object]:
    from .abrank import PREPROCESSING_VERSION, resolve_group

    group = resolve_group({"group": group})
    if not math.isfinite(min_label_gap) or min_label_gap <= 0:
        raise ValueError("min_label_gap must be finite and > 0.")
    cache_path = _pair_cache_path(cache_dir, min_label_gap=min_label_gap, group=group)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected = {
        "min_label_gap": float(min_label_gap),
        "group": group,
        "preprocessing_version": PREPROCESSING_VERSION,
        "records_signature": _records_signature(records),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if all((meta.get(k) == v for k, v in expected.items())) and isinstance(
            payload.get("pairs"), dict
        ):
            print(f"[dataset] AbRank ranking pair cache hit: {cache_path}")
            return payload["pairs"]
        print(f"[dataset] AbRank ranking pair cache stale, rebuilding: {cache_path}")
    pairs = build_abrank_ranking_pairs(
        records, min_label_gap=min_label_gap, group=group
    )
    torch.save(
        {"meta": {**expected, "num_records": len(records)}, "pairs": pairs}, cache_path
    )
    print(f"[dataset] AbRank ranking pair cache saved: {cache_path}")
    return pairs


def split_abrank_ranking_pairs_iid(
    pair_bins: Dict[str, object], num_clients: int, test_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict], Dict[str, object]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0 for ranking split.")
    rng = np.random.default_rng(seed)
    pos_pairs = list(pair_bins.get("positive", []))
    neg_pairs = list(pair_bins.get("negative", []))
    rng.shuffle(pos_pairs)
    rng.shuffle(neg_pairs)
    train_pos, test_pos = _split_train_test(pos_pairs, test_ratio)
    train_neg, test_neg = _split_train_test(neg_pairs, test_ratio)
    train_pos_by_client, drop_train_pos = _stratified_client_assign(
        train_pos, num_clients, rng
    )
    train_neg_by_client, drop_train_neg = _stratified_client_assign(
        train_neg, num_clients, rng
    )
    test_pos_by_client, drop_test_pos = _stratified_client_assign(
        test_pos, num_clients, rng
    )
    test_neg_by_client, drop_test_neg = _stratified_client_assign(
        test_neg, num_clients, rng
    )
    train_records: List[Dict] = []
    test_records: List[Dict] = []
    stats = {
        "overall": {
            "positive": len(pos_pairs),
            "negative": len(neg_pairs),
            "dropped_gap": int(pair_bins.get("dropped_gap", 0)),
        },
        "train": {},
        "test": {},
        "dropped_remainder": {
            "train_positive": drop_train_pos,
            "train_negative": drop_train_neg,
            "test_positive": drop_test_pos,
            "test_negative": drop_test_neg,
        },
    }
    for cid in range(num_clients):
        train_rows = list(train_pos_by_client[cid]) + list(train_neg_by_client[cid])
        test_rows = list(test_pos_by_client[cid]) + list(test_neg_by_client[cid])
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)
        train_records.extend(train_rows)
        test_records.extend(test_rows)
        stats["train"][cid] = {
            "positive": len(train_pos_by_client[cid]),
            "negative": len(train_neg_by_client[cid]),
            "total": len(train_rows),
        }
        stats["test"][cid] = {
            "positive": len(test_pos_by_client[cid]),
            "negative": len(test_neg_by_client[cid]),
            "total": len(test_rows),
        }
    empty_train_clients = [
        cid for cid, item in stats["train"].items() if int(item.get("total", 0)) == 0
    ]
    if empty_train_clients:
        client_text = ", ".join((str(int(cid) + 1) for cid in empty_train_clients))
        raise ValueError(
            f"No usable ranking train pairs for clients: {client_text}. Please reduce num_clients or min_label_gap."
        )
    if not train_records:
        raise ValueError(
            "No usable ranking train pairs after IID split. Please reduce num_clients or gap threshold."
        )
    return (train_records, test_records, stats)


class AbRankRankingPairDataset(Dataset):

    def __init__(
        self,
        records: List[Dict],
        num_clients: int,
        embedding_path: Optional[Path] = None,
    ):
        self.records = records
        self.num_clients = num_clients
        self.embedding_path = embedding_path
        self.client_counts: Dict[int, int] = {}
        for rec in records:
            cid = int(rec.get("arch_index", -1))
            self.client_counts[cid] = self.client_counts.get(cid, 0) + 1

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        target = torch.tensor(
            float(rec.get("pair_label", rec.get("binding_score", 0.0))),
            dtype=torch.float,
        )
        arch_index = torch.tensor(int(rec.get("arch_index", -1)), dtype=torch.long)
        if self.embedding_path is not None:
            sid1 = int(rec.get("sample_id_1", -1))
            sid2 = int(rec.get("sample_id_2", -1))
            ab_1, ag_1 = _get_pair_embedding(self.embedding_path, sid1)
            ab_2, ag_2 = _get_pair_embedding(self.embedding_path, sid2)
        else:
            ab_1 = rec.get("antibody_seq_1", "")
            ag_1 = rec.get("antigen_seq_1", "")
            ab_2 = rec.get("antibody_seq_2", "")
            ag_2 = rec.get("antigen_seq_2", "")
        return {
            "ab_seq_1": ab_1,
            "ag_seq_1": ag_1,
            "ab_seq_2": ab_2,
            "ag_seq_2": ag_2,
            "target": target,
            "arch_index": arch_index,
            "sample_id": rec.get("sample_id", idx),
            "sample_id_1": rec.get("sample_id_1", -1),
            "sample_id_2": rec.get("sample_id_2", -1),
            "antigen_id": rec.get("antigen_id", ""),
            "meta_id": rec.get("meta_id", ""),
            "mutant_name": rec.get("mutant_name", ""),
            "pair_label": rec.get("pair_label", 0.0),
        }


def collate_abrank_ranking_batch(batch: List[Dict]) -> Dict[str, object]:
    targets = torch.stack([b["target"] for b in batch])
    arch_index = torch.stack([b["arch_index"] for b in batch])
    ab_1_items = [b["ab_seq_1"] for b in batch]
    ag_1_items = [b["ag_seq_1"] for b in batch]
    ab_2_items = [b["ab_seq_2"] for b in batch]
    ag_2_items = [b["ag_seq_2"] for b in batch]
    if ab_1_items and torch.is_tensor(ab_1_items[0]):
        ab_1_items = pad_sequence(ab_1_items, batch_first=True, padding_value=0.0)
    if ag_1_items and torch.is_tensor(ag_1_items[0]):
        ag_1_items = pad_sequence(ag_1_items, batch_first=True, padding_value=0.0)
    if ab_2_items and torch.is_tensor(ab_2_items[0]):
        ab_2_items = pad_sequence(ab_2_items, batch_first=True, padding_value=0.0)
    if ag_2_items and torch.is_tensor(ag_2_items[0]):
        ag_2_items = pad_sequence(ag_2_items, batch_first=True, padding_value=0.0)
    return {
        "ab_seq_1": ab_1_items,
        "ag_seq_1": ag_1_items,
        "ab_seq_2": ab_2_items,
        "ag_seq_2": ag_2_items,
        "target": targets,
        "arch_index": arch_index,
        "sample_id": [b.get("sample_id", -1) for b in batch],
        "sample_id_1": [b.get("sample_id_1", -1) for b in batch],
        "sample_id_2": [b.get("sample_id_2", -1) for b in batch],
        "antigen_id": [b.get("antigen_id", "") for b in batch],
        "meta_id": [b.get("meta_id", "") for b in batch],
        "mutant_name": [b.get("mutant_name", "") for b in batch],
        "pair_label": [b.get("pair_label", 0.0) for b in batch],
    }


__all__ = [
    "AbRankRankingPairDataset",
    "build_abrank_ranking_pairs",
    "collate_abrank_ranking_batch",
    "drop_largest_source_for_ranking",
    "is_abrank_ranking_task",
    "load_or_build_abrank_ranking_pairs",
    "split_abrank_ranking_pairs_iid",
]
