import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, PPBuilder


def load_metadata(cache_dir: str) -> Dict[str, Dict]:
    path = Path(cache_dir) / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        meta = json.load(f)
    mapping: Dict[str, Dict] = {}
    if isinstance(meta, list):
        iterable = enumerate(meta)
    elif isinstance(meta, dict):
        iterable = meta.items()
    else:
        return mapping
    for key, entry in iterable:
        if not isinstance(entry, dict):
            continue
        antigen = (
            entry.get("antigen_id")
            or entry.get("antigen")
            or entry.get("antigen_name")
            or entry.get("name")
            or key
        )
        antigen = str(antigen)
        if antigen:
            mapping[antigen] = entry
    return mapping


def _find_structure_file(complex_dir: Path, pdb_id: str) -> Optional[Path]:
    if not pdb_id:
        return None
    candidates = [
        complex_dir / f"{pdb_id}.pdb",
        complex_dir / f"{pdb_id}.cif",
        complex_dir / f"{pdb_id}.mmcif",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    for cand in complex_dir.glob(f"{pdb_id}.*"):
        if cand.is_file():
            return cand
    return None


def _resolve_pdb_path(pdb_path: str, complex_dir: Path) -> Optional[Path]:
    if not pdb_path:
        return None
    raw = Path(pdb_path)
    candidates = [
        raw,
        complex_dir / raw.name,
        complex_dir / raw,
        complex_dir.parent / raw,
        complex_dir.parent / raw.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _extract_chain_sequences(
    structure_path: Path, chains: Optional[List[str]] = None
) -> Dict[str, str]:
    parser = (
        PDBParser(QUIET=True)
        if structure_path.suffix.lower() == ".pdb"
        else MMCIFParser(QUIET=True)
    )
    structure = parser.get_structure(structure_path.stem, structure_path)
    builder = PPBuilder()
    wanted = set(chains) if chains else None
    seqs: Dict[str, str] = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if wanted is not None and chain_id not in wanted:
                continue
            peptides = builder.build_peptides(chain)
            seq = "".join((str(p.get_sequence()) for p in peptides))
            if seq:
                seqs[chain_id] = seq
        break
    return seqs


def _extract_chain_sequences_with_ids(
    structure_path: Path, chains: Optional[List[str]] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    parser = (
        PDBParser(QUIET=True)
        if structure_path.suffix.lower() == ".pdb"
        else MMCIFParser(QUIET=True)
    )
    structure = parser.get_structure(structure_path.stem, structure_path)
    builder = PPBuilder()
    wanted = set(chains) if chains else None
    seqs: Dict[str, str] = {}
    res_ids: Dict[str, List[str]] = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if wanted is not None and chain_id not in wanted:
                continue
            seq_chunks: List[str] = []
            ids: List[str] = []
            for pep in builder.build_peptides(chain):
                seq_chunks.append(str(pep.get_sequence()))
                for res in pep:
                    het_flag, resseq, icode = res.get_id()
                    if het_flag.strip():
                        continue
                    icode = str(icode or "").strip()
                    resid = (
                        f"{chain_id}{resseq}{icode}" if icode else f"{chain_id}{resseq}"
                    )
                    ids.append(resid)
            if seq_chunks:
                seqs[chain_id] = "".join(seq_chunks)
                res_ids[chain_id] = ids
        break
    return (seqs, res_ids)


def build_antibody_wildtype_map(
    metadata: Dict[str, Dict], complex_dir: Path
) -> Dict[str, Dict]:
    complex_dir.mkdir(parents=True, exist_ok=True)
    seq_map: Dict[str, Dict] = {}
    for raw_key, meta in metadata.items():
        k_lower = str(raw_key).lower()
        base_key = k_lower.split("_")[0]
        pdb_id = (
            meta.get("pdb")
            or meta.get("pdb_id")
            or meta.get("structure")
            or meta.get("structure_id")
        )
        pdb_path = meta.get("pdb_path")
        heavy_chain = (
            meta.get("heavy_chain") or meta.get("h_chain") or meta.get("heavy")
        )
        light_chain = (
            meta.get("light_chain") or meta.get("l_chain") or meta.get("light")
        )
        if not (heavy_chain or light_chain):
            continue
        structure_path: Optional[Path] = _resolve_pdb_path(pdb_path, complex_dir)
        if structure_path is None and pdb_id:
            structure_path = _find_structure_file(complex_dir, pdb_id)
        if not structure_path:
            continue
        wanted_chains = [c for c in (heavy_chain, light_chain) if c]
        chain_seqs, chain_res_ids = _extract_chain_sequences_with_ids(
            structure_path, wanted_chains if wanted_chains else None
        )
        if not chain_seqs:
            continue
        info = {
            "pdb_id": pdb_id,
            "pdb_path": str(structure_path),
            "heavy_chain_id": heavy_chain,
            "light_chain_id": light_chain,
            "heavy_seq": chain_seqs.get(heavy_chain, ""),
            "light_seq": chain_seqs.get(light_chain, ""),
            "heavy_res_ids": chain_res_ids.get(heavy_chain, []),
            "light_res_ids": chain_res_ids.get(light_chain, []),
            "meta_key": str(
                meta.get("antigen_id") or meta.get("name") or meta.get("id") or raw_key
            ),
        }
        seq_map[k_lower] = info
        seq_map.setdefault(base_key, info)
    return seq_map


def build_antigen_sequence_map(
    antigen_ids: Iterable[str], metadata: Dict[str, Dict], complex_dir: Path
) -> Dict[str, Dict]:
    complex_dir.mkdir(parents=True, exist_ok=True)
    seq_map: Dict[str, Dict] = {}
    meta_by_key = {str(k).lower(): v for k, v in metadata.items()}
    meta_by_base = {str(k).lower().split("_")[0]: v for k, v in metadata.items()}
    for ag in antigen_ids:
        ag_key = str(ag).lower()
        base_key = ag_key.split("_")[0]
        meta = meta_by_key.get(ag_key) or meta_by_base.get(base_key) or {}
        pdb_id = (
            meta.get("pdb_id")
            or meta.get("pdb")
            or meta.get("structure")
            or meta.get("structure_id")
        )
        pdb_path = meta.get("pdb_path")
        chains = (
            meta.get("antigen_chains")
            or meta.get("chains")
            or meta.get("antigen_chain_ids")
            or []
        )
        if isinstance(chains, str):
            chains = [chains]
        chains = list(chains)
        epitope_chain = meta.get("epitope_chain")
        structure_path: Optional[Path] = _resolve_pdb_path(pdb_path, complex_dir)
        if structure_path is None and pdb_id:
            structure_path = _find_structure_file(complex_dir, pdb_id)
        if not structure_path:
            continue
        chain_seqs = _extract_chain_sequences(
            structure_path, chains if chains else None
        )
        if not chain_seqs:
            continue
        if epitope_chain and epitope_chain in chain_seqs:
            ordered_ids = [epitope_chain]
        else:
            ordered_ids = chains if chains else list(chain_seqs.keys())
        concat = "".join((chain_seqs.get(cid, "") for cid in ordered_ids))
        seq_map[ag] = {
            "pdb_id": pdb_id,
            "meta_key": meta.get("antigen_id")
            or meta.get("name")
            or meta.get("id")
            or ag,
            "antigen_seq": concat,
        }
    return seq_map


def split_records(
    records: List[Dict],
    num_clients: int,
    test_ratio: float = 0.1,
    seed: int = 0,
    split_strategy: str = "iid",
    shared_key: str = "",
    enable_shared_key_equilibrium: bool = False,
    dirichlet_enabled: bool = False,
    dirichlet_alpha: float = 0.5,
    randomize_test_clients: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0 for splitting.")
    if not records:
        return ([], [])
    strategy = str(split_strategy or "iid").lower().replace("-", "_")
    if strategy not in {"iid", "meta_key"}:
        raise ValueError("split_strategy must be one of: iid, meta_key (or meta-key).")
    rng = np.random.default_rng(seed)
    rows = list(records)
    rng.shuffle(rows)
    n_total = len(rows)
    if n_total <= 1:
        n_test = 0
    else:
        n_test = int(round(n_total * float(test_ratio)))
        n_test = max(1, min(n_test, n_total - 1))
    test_rows = rows[:n_test]
    train_rows = rows[n_test:]

    def _round_robin(rows: List[Dict]) -> List[Dict]:
        per_client: Dict[int, List[Dict]] = {cid: [] for cid in range(num_clients)}
        for pos, row in enumerate(rows):
            cid = pos % num_clients
            rec = attach_record(dict(row), client_id=cid, idx=row.get("sample_id", pos))
            per_client[cid].append(rec)
        assigned: List[Dict] = []
        for cid in range(num_clients):
            assigned.extend(per_client[cid])
        return assigned

    if strategy == "meta_key":
        if enable_shared_key_equilibrium and dirichlet_enabled:
            raise ValueError(
                "meta_key split: shared_key_equilibrium and dirichlet cannot both be enabled."
            )
        if dirichlet_enabled and float(dirichlet_alpha) <= 0:
            raise ValueError("meta_key split: dirichlet_alpha must be > 0.")

        def _meta_key_assign_dirichlet(rows: List[Dict]) -> List[Dict]:
            if not rows:
                return []
            grouped: Dict[str, List[Dict]] = {}
            for row in rows:
                key = str(row.get("meta_id") or row.get("antigen_id") or "").lower()
                grouped.setdefault(key or "unknown", []).append(row)
            per_client: Dict[int, List[Dict]] = {cid: [] for cid in range(num_clients)}
            alpha_vec = np.full(num_clients, float(dirichlet_alpha), dtype=float)
            for key in sorted(grouped.keys()):
                chunk = list(grouped[key])
                rng.shuffle(chunk)
                probs = rng.dirichlet(alpha_vec)
                counts = rng.multinomial(len(chunk), probs)
                offset = 0
                for cid in range(num_clients):
                    take = int(counts[cid])
                    if take <= 0:
                        continue
                    per_client[cid].extend(chunk[offset : offset + take])
                    offset += take
            for cid in range(num_clients):
                if per_client[cid]:
                    continue
                donor = max(range(num_clients), key=lambda k: (len(per_client[k]), -k))
                if len(per_client[donor]) <= 1:
                    break
                per_client[cid].append(per_client[donor].pop())
            assigned: List[Dict] = []
            for cid in range(num_clients):
                for pos, row in enumerate(per_client[cid]):
                    assigned.append(
                        attach_record(
                            dict(row), client_id=cid, idx=row.get("sample_id", pos)
                        )
                    )
            return assigned

        def _meta_key_assign(rows: List[Dict]) -> List[Dict]:
            if not rows:
                return []
            grouped: Dict[str, List[Dict]] = {}
            for row in rows:
                key = str(row.get("meta_id") or row.get("antigen_id") or "").lower()
                grouped.setdefault(key or "unknown", []).append(row)
            meta_key_count = len(grouped)
            shared_key_norm = str(shared_key).strip().lower()
            if not shared_key_norm or shared_key_norm in {"none", "null"}:
                raise ValueError(
                    "meta_key 划分需要 shared_key，请设置 data['meta-key'].shared_key_equilibrium.key。"
                )
            assigned: List[Dict] = []
            if shared_key_norm not in grouped:
                available = sorted(grouped.keys())
                raise ValueError(
                    f"shared_key='{shared_key_norm}' 不在 meta_key 列表中。可用 meta_key 样例: {available[:10]}{('...' if len(available) > 10 else '')}"
                )
            shared_rows = grouped.pop(shared_key_norm)
            if enable_shared_key_equilibrium:
                expected_clients = meta_key_count - 1
                if num_clients != expected_clients:
                    raise ValueError(
                        f"meta_key 划分要求 num_clients = meta_key 数量 - 1。当前 meta_key 数量={meta_key_count}, num_clients={num_clients}。shared_key 为 '{shared_key_norm}' (样本数={len(shared_rows)}) 将作为共享 padding。因此 num_clients 应设为 {expected_clients}。"
                    )
            else:
                expected_clients = meta_key_count
                if num_clients != expected_clients:
                    raise ValueError(
                        f"meta_key 划分要求 num_clients = meta_key 数量。当前 meta_key 数量={meta_key_count}, num_clients={num_clients}。shared_key 为 '{shared_key_norm}' 将作为独立 client 分配。因此 num_clients 应设为 {expected_clients}。"
                    )
            target = max((len(rows) for rows in grouped.values())) if grouped else 0
            shared_selected: List[Dict] = []
            for cid, key in enumerate(sorted(grouped.keys())):
                exclusive_rows = grouped[key]
                rng.shuffle(exclusive_rows)
                for pos, row in enumerate(exclusive_rows):
                    assigned.append(
                        attach_record(
                            dict(row), client_id=cid, idx=row.get("sample_id", pos)
                        )
                    )
                pad_needed = max(0, target - len(exclusive_rows))
                if pad_needed:
                    replace = len(shared_rows) < pad_needed
                    picks = rng.choice(
                        len(shared_rows), size=pad_needed, replace=replace
                    )
                    for offset, pick in enumerate(picks):
                        row = shared_rows[int(pick)]
                        if enable_shared_key_equilibrium:
                            assigned.append(
                                attach_record(
                                    dict(row),
                                    client_id=cid,
                                    idx=row.get(
                                        "sample_id", len(exclusive_rows) + offset
                                    ),
                                )
                            )
                        shared_selected.append(dict(row))
            if not enable_shared_key_equilibrium:
                shared_cid = len(grouped)
                for pos, row in enumerate(shared_selected):
                    assigned.append(
                        attach_record(
                            dict(row),
                            client_id=shared_cid,
                            idx=row.get("sample_id", pos),
                        )
                    )
            return assigned

        if randomize_test_clients:
            train_assigned = (
                _meta_key_assign_dirichlet(train_rows)
                if dirichlet_enabled
                else _meta_key_assign(train_rows)
            )
            randomized_test_rows = list(test_rows)
            np.random.default_rng(seed).shuffle(randomized_test_rows)
            return (train_assigned, _round_robin(randomized_test_rows))
        if dirichlet_enabled:
            assigned_all = _meta_key_assign_dirichlet(rows)
            per_client: Dict[int, List[Dict]] = {cid: [] for cid in range(num_clients)}
            for rec in assigned_all:
                cid = int(rec.get("arch_index", -1))
                if cid < 0:
                    continue
                per_client.setdefault(cid, []).append(rec)
            train_assigned: List[Dict] = []
            test_assigned: List[Dict] = []
            ratio = float(test_ratio)
            for cid in range(num_clients):
                client_rows = list(per_client.get(cid, []))
                if not client_rows:
                    continue
                n = len(client_rows)
                if n <= 1:
                    n_test_local = 0
                else:
                    n_test_local = max(1, int(round(n * ratio)))
                    n_test_local = min(n_test_local, n - 1)
                grouped_client: Dict[str, List[Dict]] = {}
                for rec in client_rows:
                    key = str(rec.get("meta_id") or rec.get("antigen_id") or "").lower()
                    grouped_client.setdefault(key or "unknown", []).append(rec)
                for key in grouped_client:
                    rng.shuffle(grouped_client[key])
                keys = sorted(grouped_client.keys())
                exact = {k: len(grouped_client[k]) * ratio for k in keys}
                base = {
                    k: min(len(grouped_client[k]), int(np.floor(exact[k])))
                    for k in keys
                }
                assigned_test = sum(base.values())
                need = max(0, n_test_local - assigned_test)
                if need > 0:
                    candidates = [
                        (exact[k] - base[k], k)
                        for k in keys
                        if base[k] < len(grouped_client[k])
                    ]
                    candidates.sort(reverse=True)
                    idx = 0
                    while need > 0 and candidates:
                        _, key = candidates[idx % len(candidates)]
                        if base[key] < len(grouped_client[key]):
                            base[key] += 1
                            need -= 1
                        idx += 1
                        if idx > 10 * len(candidates) and need > 0:
                            break
                for key in keys:
                    rows_key = grouped_client[key]
                    take_test = min(base[key], len(rows_key))
                    test_assigned.extend(rows_key[:take_test])
                    train_assigned.extend(rows_key[take_test:])
            return (train_assigned, test_assigned)
        return (_meta_key_assign(train_rows), _meta_key_assign(test_rows))
    usable_test = len(test_rows) // num_clients * num_clients
    usable_train = len(train_rows) // num_clients * num_clients
    if usable_train <= 0:
        raise ValueError("Not enough samples to split across clients.")
    test_rows = test_rows[:usable_test]
    train_rows = train_rows[:usable_train]
    test = _round_robin(test_rows)
    train = _round_robin(train_rows)
    return (train, test)


LENRICH_KEYS = {"1mlc", "2fjg", "4d5_her2", "5a12_ang2", "g6_lc"}


def is_lenrich_key(key: str) -> bool:
    base = str(key).lower()
    return base in LENRICH_KEYS or base.split("_")[0] in LENRICH_KEYS


def filter_metadata_by_group(metadata: Dict[str, Dict], group: str) -> Dict[str, Dict]:
    g = str(group).lower()
    return {
        k: v
        for k, v in metadata.items()
        if is_lenrich_key(k)
        and g == "lenrich"
        or (not is_lenrich_key(k) and g == "pkd")
    }


def filter_records_by_group(records: List[Dict], group: str) -> List[Dict]:
    g = str(group).lower()
    filtered: List[Dict] = []
    for rec in records:
        ag = str(rec.get("antigen_id", "")).lower()
        base = ag.split("_")[0]
        in_lenrich = ag in LENRICH_KEYS or base in LENRICH_KEYS
        if g == "lenrich" and in_lenrich or (g == "pkd" and (not in_lenrich)):
            filtered.append(rec)
    return filtered


def attach_record(row: Dict, client_id: int, idx: int) -> Dict:
    antibody_seq = row.get("antibody_seq") or ""
    if not antibody_seq:
        heavy_seq = row.get("heavy_chain_seq", "")
        light_seq = row.get("light_chain_seq", "")
        antibody_seq = (heavy_seq or "") + (light_seq or "")
    antigen_seq = row.get("antigen_seq", "")
    return {
        "binding_score": float(row["binding_score"]),
        "arch_index": client_id,
        "sample_id": row.get("sample_id", idx),
        "antigen_id": row.get("antigen_id", ""),
        "meta_id": row.get("meta_id", ""),
        "antibody_seq": antibody_seq,
        "antigen_seq": antigen_seq,
        "mutant_name": row.get("mutant_name", ""),
        "seq_len": len(antibody_seq),
        "antigen_len": len(antigen_seq),
    }
