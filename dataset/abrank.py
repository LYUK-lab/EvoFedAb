from __future__ import annotations
import csv
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from utils.config import resolve_cache_dir
from .abrank_sequences import normalize_preprocessing_sequences


def _as_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na"}:
        return ""
    return text


def _clean_seq(seq: object) -> str:
    return "".join(_as_str(seq).split()).upper()


def _normalize_key(key: object) -> str:
    return "".join((ch.lower() for ch in str(key) if ch.isalnum()))


def _get_field(row: Dict[str, object], keys: List[str]) -> str:
    for key in keys:
        if key in row:
            val = _as_str(row.get(key))
            if val:
                return val
    normalized = {_normalize_key(k): v for k, v in row.items()}
    for key in keys:
        val = _as_str(normalized.get(_normalize_key(key), ""))
        if val:
            return val
    return ""


def _detect_delimiter(path: Path) -> str:
    with open(path, "r", newline="") as handle:
        head = handle.readline()
    return "\t" if "\t" in head else ","


def _iter_csv_rows(cache_dir: Path):
    paths = sorted(cache_dir.glob("AbRank_dataset*.csv"))
    if not paths:
        paths = sorted(cache_dir.glob("*.csv"))
    if not paths:
        raise RuntimeError(f"No AbRank CSV found under {cache_dir}.")
    for path in paths:
        delimiter = _detect_delimiter(path)
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                yield row


PREPROCESSING_VERSION = "abrank_pkd_pic50_min_class50_v12"
_PRECISION = Decimal("0.0001")
_MIN_CLASS_SIZE = 50
_SKEMPI_LONG_ANTIGEN_NAMES = ("agskempi-2nyy_a_", "agskempi-2nz9_a_")
_SKEMPI_REFERENCE_WINDOW = (901, 1293)
_SABDAB_LONG_ANTIGEN_NAMES = ("ag-2nyy_a", "ag-2nz9_b")
_ABCOV_REFERENCE_NAME = "sars-cov-2"
_ABCOV_REFERENCE_WINDOW = (331, 531)
_ABCOV_EXPECTED_ANTIGENS = 5
_IC50_REFERENCE_MW_G_MOL = Decimal("150000")


def _align_window(ref_seq: str, qry_seq: str, start: int, end: int) -> Tuple[int, int]:
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.end_gap_score = 0.0
    ref_aln, qry_aln = aligner.align(ref_seq, qry_seq)[0][:2]
    ref_pos = qry_pos = 0
    mapped_start = mapped_end = None
    for ref_ch, qry_ch in zip(ref_aln, qry_aln):
        if ref_ch != "-":
            ref_pos += 1
        if qry_ch != "-":
            qry_pos += 1
        if ref_ch != "-" and qry_ch != "-" and (start <= ref_pos <= end):
            if mapped_start is None:
                mapped_start = qry_pos
            mapped_end = qry_pos
    if mapped_start is None:
        raise ValueError(
            f"AbRank alignment mapped no residue for window [{start}, {end}]."
        )
    return (mapped_start, mapped_end)


def _apply_sabdab_window(selected: List[Dict], reference_seq: str | None) -> Dict:
    start, end = _SKEMPI_REFERENCE_WINDOW
    entry = {
        "mode": "aligned",
        "reference": "SKEMPIv2-2nyy-2nz9",
        "reference_start": start,
        "reference_end": end,
        "trimmed_rows": len(selected),
    }
    if not selected:
        return entry
    if reference_seq is None:
        raise ValueError(
            "AbRank SabDab 2nyy/2nz9 window needs the SKEMPIv2 antigen as reference."
        )
    variants = {rec["antigen_seq"] for rec in selected}
    if len(variants) != 1:
        raise ValueError(
            f"AbRank SabDab 2nyy/2nz9 expects one antigen, found {len(variants)}."
        )
    seq = variants.pop()
    lo, hi = _align_window(reference_seq, seq, start, end)
    if hi - lo + 1 != end - start + 1:
        raise ValueError(
            f"AbRank SabDab 2nyy/2nz9 mapped window [{lo}, {hi}] does not match the reference width {end - start + 1}."
        )
    for rec in selected:
        rec["antigen_seq"] = rec["antigen_seq"][lo - 1 : hi]
    entry.update(
        distinct_antigens=1, wt_length=len(seq), start=lo, end=hi, length=hi - lo + 1
    )
    return entry


def _apply_abcov_window(records: List[Dict]) -> Dict:
    selected = [
        rec
        for rec in records
        if str(rec.get("meta_id") or "").strip().lower() == "abcov"
    ]
    start, end = _ABCOV_REFERENCE_WINDOW
    entry = {
        "mode": "aligned",
        "reference": _ABCOV_REFERENCE_NAME,
        "reference_start": start,
        "reference_end": end,
        "trimmed_rows": len(selected),
    }
    if not selected:
        return entry
    by_antigen: Dict[str, str] = {}
    for rec in selected:
        name = str(rec.get("ag_name") or "").strip().lower()
        if by_antigen.setdefault(name, rec["antigen_seq"]) != rec["antigen_seq"]:
            raise ValueError(
                f"AbRank AbCoV antigen '{name}' carries more than one sequence."
            )
    if len(by_antigen) != _ABCOV_EXPECTED_ANTIGENS:
        raise ValueError(
            f"AbRank AbCoV expects {_ABCOV_EXPECTED_ANTIGENS} distinct antigens, found {sorted(by_antigen)}."
        )
    ref_seq = by_antigen.get(_ABCOV_REFERENCE_NAME)
    if ref_seq is None:
        raise ValueError(
            f"AbRank AbCoV reference antigen '{_ABCOV_REFERENCE_NAME}' is missing."
        )
    if len(ref_seq) < end:
        raise ValueError(
            f"AbRank AbCoV window [{start}, {end}] exceeds reference length {len(ref_seq)}."
        )
    windows = {
        name: (
            (start, end)
            if name == _ABCOV_REFERENCE_NAME
            else _align_window(ref_seq, seq, start, end)
        )
        for name, seq in sorted(by_antigen.items())
    }
    for rec in selected:
        lo, hi = windows[str(rec.get("ag_name") or "").strip().lower()]
        rec["antigen_seq"] = rec["antigen_seq"][lo - 1 : hi]
    entry["distinct_antigens"] = len(by_antigen)
    entry["windows"] = {
        name: {"start": lo, "end": hi, "length": hi - lo + 1}
        for name, (lo, hi) in windows.items()
    }
    return entry


def _apply_antigen_windows(records: List[Dict]) -> Dict:
    targets = {"RBD-escape": [], "SKEMPIv2-2nyy-2nz9": [], "SabDab-2nyy-2nz9": []}
    for rec in records:
        source = str(rec.get("meta_id") or "").strip().lower()
        ag_name = str(rec.get("ag_name") or "").strip().lower()
        if source == "rbd-escape":
            targets["RBD-escape"].append(rec)
        elif source == "skempiv2" and ag_name.startswith(_SKEMPI_LONG_ANTIGEN_NAMES):
            targets["SKEMPIv2-2nyy-2nz9"].append(rec)
        elif source == "sabdab" and ag_name.startswith(_SABDAB_LONG_ANTIGEN_NAMES):
            targets["SabDab-2nyy-2nz9"].append(rec)
    stats = {}
    skempi_reference = None
    for family, start, end, expected_length in (
        ("RBD-escape", 331, 531, None),
        ("SKEMPIv2-2nyy-2nz9", *_SKEMPI_REFERENCE_WINDOW, 1295),
    ):
        selected = targets[family]
        entry = {
            "start": start,
            "end": end,
            "length": end - start + 1,
            "trimmed_rows": len(selected),
        }
        if not selected:
            stats[family] = entry
            continue
        lengths = {len(rec["antigen_seq"]) for rec in selected}
        if len(lengths) != 1 or (
            expected_length is not None and lengths != {expected_length}
        ):
            raise ValueError(
                f"AbRank {family} antigens have unexpected lengths {sorted(lengths)}."
            )
        wt_length = lengths.pop()
        if wt_length < end:
            raise ValueError(
                f"AbRank {family} window [{start}, {end}] exceeds antigen length {wt_length}."
            )
        variants = {rec["antigen_seq"] for rec in selected}
        sites = [
            i + 1 for i, column in enumerate(zip(*variants)) if len(set(column)) > 1
        ]
        if sites and (sites[0] < start or sites[-1] > end):
            raise ValueError(
                f"AbRank {family} window [{start}, {end}] does not cover substitution span [{sites[0]}, {sites[-1]}]."
            )
        if family == "SKEMPIv2-2nyy-2nz9":
            skempi_reference = Counter(
                (rec["antigen_seq"] for rec in selected)
            ).most_common(1)[0][0]
        for rec in selected:
            rec["antigen_seq"] = rec["antigen_seq"][start - 1 : end]
        entry.update(
            wt_length=wt_length,
            distinct_antigens=len(variants),
            substitution_sites=len(sites),
            substitution_start=sites[0] if sites else None,
            substitution_end=sites[-1] if sites else None,
        )
        stats[family] = entry
    stats["SabDab-2nyy-2nz9"] = _apply_sabdab_window(
        targets["SabDab-2nyy-2nz9"], skempi_reference
    )
    stats["AbCoV"] = _apply_abcov_window(records)
    return stats


def resolve_group(data_cfg: Dict) -> str:
    value = str(data_cfg.get("group", "") or "").strip().lower()
    groups = {"pkd": "pKd", "pic50": "pIC50"}
    if value not in groups:
        raise ValueError("AbRank requires explicit data.group='pkd' or 'pic50'.")
    return groups[value]


def _metric(raw: str, *, positive: bool = False) -> Decimal:
    if not raw:
        raise ValueError("missing")
    if any((c in raw for c in "<>≤≥")):
        raise ValueError("censored")
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation as err:
        raise ValueError("invalid") from err
    if (
        not value.is_finite()
        or not math.isfinite(float(value))
        or (positive and value <= 0)
    ):
        raise ValueError("invalid")
    return value


def _pic50_value(row: Dict) -> Tuple[Decimal | None, str]:
    try:
        ic50 = _metric(_get_field(row, ["IC50 [ug/mL]"]), positive=True)
    except ValueError as err:
        return (None, f"IC50_{err}")
    with localcontext() as context:
        context.prec = 40
        molar = ic50 * Decimal("0.001") / _IC50_REFERENCE_MW_G_MOL
        pic50 = (-molar.log10()).quantize(_PRECISION, rounding=ROUND_HALF_EVEN)
    return (pic50, "retained")


def _pkd_value(row: Dict) -> Tuple[Decimal | None, str]:
    kd_raw = _get_field(row, ["Affinity_Kd [nM]"])
    log_raw = _get_field(row, ["log_Aff"])
    if not kd_raw and (not log_raw):
        return (None, "missing_Kd_and_logAff")
    candidates = []
    with localcontext() as context:
        context.prec = 40
        if kd_raw:
            try:
                kd = _metric(kd_raw, positive=True)
            except ValueError as err:
                return (None, f"Kd_{err}")
            candidates.append(
                (Decimal(9) - kd.log10()).quantize(_PRECISION, rounding=ROUND_HALF_EVEN)
            )
        if log_raw:
            try:
                log_aff = _metric(log_raw)
            except ValueError as err:
                return (None, f"logAff_{err}")
            if _get_field(row, ["Aff_op"]) not in ("", "="):
                return (None, "logAff_censored")
            if not kd_raw and _get_field(row, ["IC50 [ug/mL]"]):
                return (None, "IC50_only_not_pKd")
            try:
                candidates.append(
                    (Decimal(9) - log_aff).quantize(
                        _PRECISION, rounding=ROUND_HALF_EVEN
                    )
                )
            except InvalidOperation:
                return (None, "logAff_invalid")
    if len(set(candidates)) != 1:
        return (None, "Kd_logAff_inconsistent_4dp")
    return (candidates[0], "retained")


def _label_value(row: Dict, group: str) -> Tuple[Decimal | None, str]:
    if group == "pKd":
        return _pkd_value(row)
    if group == "pIC50":
        return _pic50_value(row)
    raise ValueError("AbRank label group must be pKd or pIC50.")


def _deduplicate(records: List[Dict]) -> Tuple[List[Dict], Dict]:
    buckets = {}
    for rec in records:
        key = (rec["antibody_seq"], rec["antigen_seq"])
        bucket = buckets.setdefault(
            key, {"record": rec, "labels": set(), "sources": set(), "count": 0}
        )
        bucket["labels"].add(Decimal(str(rec["binding_score"])))
        bucket["sources"].add(rec["meta_id"])
        bucket["count"] += 1
    result = []
    conflicting_groups = cross_source_groups = 0
    for bucket in buckets.values():
        rec = bucket["record"]
        values = sorted(bucket["labels"])
        with localcontext() as context:
            context.prec = 40
            mean = sum(values) / Decimal(len(values))
            mean = mean.quantize(_PRECISION, rounding=ROUND_HALF_EVEN)
        rec["binding_score"] = float(mean)
        rec["duplicate_count"] = bucket["count"]
        rec["sources"] = sorted(bucket["sources"])
        conflicting_groups += len(values) > 1
        cross_source_groups += len(bucket["sources"]) > 1
        result.append(rec)
    removed_by_source = Counter()
    for bucket in buckets.values():
        removed_by_source[bucket["record"]["meta_id"]] += bucket["count"] - 1
    return (
        result,
        {
            "removed_rows": len(records) - len(result),
            "removed_rows_by_source": {
                k: v for k, v in sorted(removed_by_source.items()) if v
            },
            "duplicate_input_groups": sum((b["count"] > 1 for b in buckets.values())),
            "different_label_groups": conflicting_groups,
            "cross_source_input_groups": cross_source_groups,
        },
    )


def load_records(
    cfg: Dict,
) -> Tuple[List[Dict], List[Dict], Dict[str, Dict], Path, str]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    if "antigen_window" in data_cfg:
        raise ValueError(
            "AbRank antigen windows are built into preprocessing; remove data.antigen_window."
        )
    group = resolve_group(data_cfg)
    cache_dir = resolve_cache_dir(data_cfg, cfg.get("dataset"))
    records, raw_labels = ([], [])
    for raw_id, row in enumerate(_iter_csv_rows(cache_dir)):
        ab_name = _get_field(row, ["Ab_name"])
        ag_name = _get_field(row, ["Ag_name"])
        source = _get_field(row, ["Source"])
        records.append(
            {
                "sample_id": raw_id,
                "ab_name": ab_name,
                "ag_name": ag_name,
                "heavy_chain_seq": _clean_seq(_get_field(row, ["Ab_heavy_chain_seq"])),
                "light_chain_seq": _clean_seq(_get_field(row, ["Ab_light_chain_seq"])),
                "antigen_seq": _clean_seq(_get_field(row, ["Ag_seq"])),
                "antigen_id": ag_name.lower() or "unknown",
                "mutant_name": ab_name,
                "meta_id": source,
                "label_group": group,
            }
        )
        raw_labels.append(
            {
                k: _get_field(row, [k])
                for k in ("Affinity_Kd [nM]", "log_Aff", "Aff_op", "IC50 [ug/mL]")
            }
        )
    raw_count = len(records)
    records, sequence_stats = normalize_preprocessing_sequences(records)
    sequence_stats["antigen_windows"] = _apply_antigen_windows(records)
    label_removed = Counter()
    selected = []
    for rec in records:
        label, reason = _label_value(raw_labels[rec["sample_id"]], group)
        if label is None:
            label_removed[reason] += 1
        else:
            rec["binding_score"] = label
            selected.append(rec)
    selected, dedup_stats = _deduplicate(selected)
    excluded_sources = Counter()
    remaining = []
    for rec in selected:
        source = rec["meta_id"]
        if source.strip().lower() == "alphaseq":
            excluded_sources["AlphaSeq"] += 1
        elif not source:
            excluded_sources["missing_Source"] += 1
        else:
            remaining.append(rec)
    source_counts = Counter((r["meta_id"] for r in remaining))
    small_classes = dict(
        sorted(
            (
                (source, count)
                for source, count in source_counts.items()
                if count < _MIN_CLASS_SIZE
            )
        )
    )
    small_class_removed = sum(small_classes.values())
    records = [rec for rec in remaining if rec["meta_id"] not in small_classes]
    assert raw_count == sequence_stats["removed_rows"] + sum(
        label_removed.values()
    ) + sum(excluded_sources.values()) + dedup_stats[
        "removed_rows"
    ] + small_class_removed + len(
        records
    )
    print(f"[dataset] AbRank {group.lower()}: loaded {len(records)} records.")
    return (records, records, {}, cache_dir, group)
