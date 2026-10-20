from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
from utils.config import resolve_cache_dir
from utils.data_utils import (
    build_antibody_wildtype_map,
    build_antigen_sequence_map,
    filter_metadata_by_group,
    filter_records_by_group,
    load_metadata,
)


def _compute_aar(wild_type: str, mutant: str) -> Optional[float]:
    wt = (wild_type or "").replace(" ", "").upper()
    mt = (mutant or "").replace(" ", "").upper()
    if not wt or not mt:
        return None
    max_len = max(len(wt), len(mt))
    if max_len == 0:
        return None
    min_len = min(len(wt), len(mt))
    matches = sum((1 for i in range(min_len) if wt[i] == mt[i]))
    return matches / max_len


def load_records(
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict], Path, str]:
    data_cfg = cfg["data"]
    group = str(data_cfg["group"]).lower()
    if group not in {"lenrich", "pkd"}:
        raise ValueError("data.group must be one of ['lenrich', 'pkd'].")
    cache_dir = resolve_cache_dir(data_cfg, cfg.get("dataset"))
    metadata = load_metadata(str(cache_dir))
    metadata = filter_metadata_by_group(metadata, group)
    meta_key_lower = {str(k).lower(): k for k in metadata.keys()}
    meta_base_lower = {str(k).lower().split("_")[0]: k for k in metadata.keys()}
    meta_lookup = {str(k).lower(): v for k, v in metadata.items()}
    meta_base_lookup = {str(k).lower().split("_")[0]: v for k, v in metadata.items()}

    def _norm_chain_id(chain_id: Optional[str]) -> str:
        if not chain_id:
            return ""
        return str(chain_id).strip().upper()

    def _select_paratope_sequence(row: Dict, meta_entry: Dict) -> str:
        paratope = meta_entry.get("paratope_chain") or meta_entry.get("paratope")
        if isinstance(paratope, (list, tuple)):
            paratope_ids = {_norm_chain_id(p) for p in paratope}
        else:
            paratope_ids = {_norm_chain_id(paratope)} if paratope else set()
        heavy_id = _norm_chain_id(
            meta_entry.get("heavy_chain")
            or meta_entry.get("h_chain")
            or meta_entry.get("heavy")
        )
        light_id = _norm_chain_id(
            meta_entry.get("light_chain")
            or meta_entry.get("l_chain")
            or meta_entry.get("light")
        )
        heavy_seq = str(row.get("heavy_chain_seq") or "")
        light_seq = str(row.get("light_chain_seq") or "")
        if heavy_id and heavy_id in paratope_ids:
            return heavy_seq
        if light_id and light_id in paratope_ids:
            return light_seq
        antibody_seq = (
            row.get("antibody_seq") or row.get("antibody") or row.get("ab_seq") or ""
        )
        if antibody_seq and (not (heavy_seq or light_seq)):
            return str(antibody_seq)
        return ""

    def _resolve_meta_key(antigen_id: str) -> str:
        ag = str(antigen_id).lower()
        base_ag = ag.split("_")[0]
        meta_key = meta_key_lower.get(ag) or meta_base_lower.get(base_ag) or ag
        return str(meta_key)

    def _resolve_meta_entry(antigen_id: str) -> Dict:
        meta_key = _resolve_meta_key(antigen_id).lower()
        base_key = meta_key.split("_")[0]
        return meta_lookup.get(meta_key) or meta_base_lookup.get(base_key) or {}

    pattern = cache_dir / "binding_affinity" / "*.csv"
    csv_paths = sorted(pattern.parent.glob(pattern.name))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found under {pattern}")
    records: List[Dict] = []
    sample_idx = 0
    for csv_path in csv_paths:
        antigen_from_name = csv_path.stem
        suffix = "_benchmarking_data"
        if antigen_from_name.endswith(suffix):
            antigen_from_name = antigen_from_name[: -len(suffix)]
        antigen_from_name = antigen_from_name.lower()
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            ag = (
                row.get("antigen_id")
                or row.get("antigen")
                or row.get("antigen_name")
                or antigen_from_name
            )
            ag = str(ag).lower()
            meta_entry = _resolve_meta_entry(ag)
            antibody_seq = _select_paratope_sequence(row, meta_entry)
            antigen_seq = (
                row.get("antigen_seq")
                or row.get("antigen_sequence")
                or row.get("ag_seq")
                or ""
            )
            records.append(
                {
                    "antibody_seq": antibody_seq,
                    "heavy_chain_seq": row.get("heavy_chain_seq", ""),
                    "light_chain_seq": row.get("light_chain_seq", ""),
                    "binding_score": float(row["binding_score"]),
                    "sample_id": sample_idx,
                    "antigen_id": ag,
                    "antigen_seq": antigen_seq,
                    "meta_id": _resolve_meta_key(ag),
                }
            )
            sample_idx += 1
    records = filter_records_by_group(records, group)
    if not records:
        raise RuntimeError(f"No records found after filtering by group '{group}'.")
    if metadata:
        try:
            antibody_wt_lookup = build_antibody_wildtype_map(
                metadata, cache_dir / "complex_structure"
            )
        except Exception as err:
            print(f"[dataset] Wild-type lookup failed during cleaning: {err}")
            antibody_wt_lookup = {}
    else:
        antibody_wt_lookup = {}

    def _non_paratope_score(rec: Dict) -> Optional[float]:
        key = _resolve_meta_key(rec.get("antigen_id", "")).lower()
        base_key = key.split("_")[0]
        meta_entry = meta_lookup.get(key) or meta_base_lookup.get(base_key) or {}
        paratope_id = _norm_chain_id(
            meta_entry.get("paratope_chain") or meta_entry.get("paratope")
        )
        heavy_id = _norm_chain_id(
            meta_entry.get("heavy_chain")
            or meta_entry.get("h_chain")
            or meta_entry.get("heavy")
        )
        light_id = _norm_chain_id(
            meta_entry.get("light_chain")
            or meta_entry.get("l_chain")
            or meta_entry.get("light")
        )
        wt = antibody_wt_lookup.get(key) or antibody_wt_lookup.get(base_key)
        if not wt:
            return None
        if paratope_id and paratope_id == heavy_id:
            return _compute_aar(wt.get("light_seq", ""), rec.get("light_chain_seq", ""))
        if paratope_id and paratope_id == light_id:
            return _compute_aar(wt.get("heavy_seq", ""), rec.get("heavy_chain_seq", ""))
        return None

    kept_records: Dict[Tuple[str, str], Dict] = {}
    kept_scores: Dict[Tuple[str, str], float] = {}
    for rec in records:
        paratope_seq = rec.get("antibody_seq") or ""
        if not paratope_seq:
            continue
        key = _resolve_meta_key(rec.get("antigen_id", "")).lower()
        dedupe_key = (key, paratope_seq)
        if dedupe_key not in kept_records:
            kept_records[dedupe_key] = rec
            score = _non_paratope_score(rec)
            if score is not None:
                kept_scores[dedupe_key] = score
            continue
        score = _non_paratope_score(rec)
        if score is None:
            continue
        prev_score = kept_scores.get(dedupe_key)
        if prev_score is None or score > prev_score:
            kept_records[dedupe_key] = rec
            kept_scores[dedupe_key] = score
    cleaned_records = list(kept_records.values())
    counts_after_raw: Dict[str, int] = {str(k).lower(): 0 for k in metadata.keys()}
    for rec in cleaned_records:
        key = _resolve_meta_key(rec.get("antigen_id", "")).lower()
        if key in counts_after_raw:
            counts_after_raw[key] = counts_after_raw.get(key, 0) + 1
    min_records = 100
    removed_keys = {k for k, v in counts_after_raw.items() if v < min_records}
    if removed_keys:
        cleaned_records = [
            rec
            for rec in cleaned_records
            if _resolve_meta_key(rec.get("antigen_id", "")).lower() not in removed_keys
        ]
        if cleaned_records:
            print(
                f"[dataset] Removed {len(removed_keys)} metadata entries with <{min_records} records (remaining records={len(cleaned_records)})."
            )
        else:
            print(
                f"[dataset] Removed all records due to <{min_records} per-metadata threshold."
            )
    records = cleaned_records
    antigen_seq_map = (
        build_antigen_sequence_map(
            metadata.keys(), metadata, cache_dir / "complex_structure"
        )
        if metadata
        else {}
    )
    antigen_seq_lookup = {str(k).lower(): v for k, v in antigen_seq_map.items()}
    for rec in records:
        lookup_key = str(rec.get("antigen_id", "")).lower()
        base_key = lookup_key.split("_")[0]
        info = antigen_seq_lookup.get(lookup_key, {}) or antigen_seq_lookup.get(
            base_key, {}
        )
        if not rec.get("antigen_seq"):
            rec["antigen_seq"] = info.get("antigen_seq", "")
        rec["antigen_pdb_id"] = info.get("pdb_id", "")
    valid_records = [
        r for r in records if r.get("antibody_seq") and r.get("antigen_seq")
    ]
    return (records, valid_records, metadata, cache_dir, group)
