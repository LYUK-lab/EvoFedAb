from __future__ import annotations
import re
from collections import Counter, defaultdict
from itertools import permutations

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
SEQUENCE_FIELDS = ("heavy_chain_seq", "light_chain_seq", "antigen_seq")
_DESCRIPTION = re.compile("(.+)_\\(([A-Z])([1-9]\\d*)([A-Z])\\)")
_OUTER = re.compile("(.+)_\\(([^()]*)\\)")
_CHAIN_TOKEN = re.compile(
    "([ACDEFGHIKLMNPQRSTVWY])([A-Za-z0-9])([1-9]\\d*)([ACDEFGHIKLMNPQRSTVWY])"
)
_SINGLE_SUFFIX = re.compile(
    "(.+)_([ACDEFGHIKLMNPQRSTVWY])([1-9]\\d*)([ACDEFGHIKLMNPQRSTVWY])"
)


def _name_key(value: str) -> str:
    return "".join((c.lower() for c in value if c.isalnum()))


def _numbering_offset(wt: str, constraints) -> int:
    constraints = sorted(set(constraints))

    def fits(offset):
        return all(
            (
                0 <= pos - offset - 1 < len(wt) and wt[pos - offset - 1] == before
                for pos, before in constraints
            )
        )

    if fits(0):
        return 0
    if len({pos for pos, _ in constraints}) >= 2:
        pos, before = constraints[0]
        offsets = [
            pos - i - 1 for i, aa in enumerate(wt) if aa == before and fits(pos - i - 1)
        ]
        if len(offsets) == 1:
            return offsets[0]
    raise ValueError("numbering/residues do not uniquely match WT")


def _mutate(wt: str, mutations, offset: int) -> str:
    replacements = {}
    for before, pos, after in mutations:
        index = pos - offset - 1
        if not 0 <= index < len(wt) or wt[index] != before:
            raise ValueError("WT residue/numbering mismatch")
        if index in replacements and replacements[index] != after:
            raise ValueError("conflicting substitutions")
        replacements[index] = after
    result = list(wt)
    for index, after in replacements.items():
        result[index] = after
    return "".join(result)


def normalize_preprocessing_sequences(records: list[dict]) -> tuple[list[dict], dict]:
    references = defaultdict(set)
    paired_references = defaultdict(set)
    pending = defaultdict(list)
    families = defaultdict(list)
    suffixes = []
    rejected = {}
    stats = Counter()

    def reject(rec, reason):
        rejected.setdefault(rec["sample_id"], reason)

    for rec in records:
        for field in SEQUENCE_FIELDS:
            rec[field] = "".join(str(rec.get(field, "") or "").split()).upper()
        rec["antibody_seq"] = rec["heavy_chain_seq"] + rec["light_chain_seq"]
        ab_name, ag_name = (rec.get("ab_name", ""), rec.get("ag_name", ""))
        seqs = tuple((rec[field] for field in SEQUENCE_FIELDS))
        for field, name, seq in zip(SEQUENCE_FIELDS, (ab_name, ab_name, ag_name), seqs):
            if seq and set(seq) <= AA_ALPHABET:
                references[field, _name_key(name)].add(seq)
            elif seq:
                match = _DESCRIPTION.fullmatch(seq)
                if match and {match[2], match[4]} <= AA_ALPHABET:
                    pending[field, _name_key(match[1])].append(
                        (rec, match[2], int(match[3]), match[4])
                    )
                else:
                    reject(rec, "unsupported_sequence_description")
        if all((set(s) <= AA_ALPHABET for s in seqs)):
            if seqs[0] or seqs[1]:
                paired_references["ab", _name_key(ab_name)].add(seqs[:2])
            if seqs[2]:
                paired_references["ag", _name_key(ag_name)].add(seqs[2:])
        ab_match, ag_match = (_OUTER.fullmatch(ab_name), _OUTER.fullmatch(ag_name))
        if ab_match or ag_match:
            if not ab_match or not ag_match or ab_match[2] != ag_match[2]:
                reject(rec, "structured_name_lists_disagree")
                continue
            tokens = [_CHAIN_TOKEN.fullmatch(t) for t in ab_match[2].split(";")]
            if not all(tokens):
                reject(rec, "unsupported_structured_name_mutations")
                continue
            mutations = [(m[1], m[2], int(m[3]), m[4]) for m in tokens]
            families[ab_match[1], ag_match[1]].append((rec, mutations))
        elif seqs[2] and set(seqs[2]) <= AA_ALPHABET:
            match = _SINGLE_SUFFIX.fullmatch(ag_name)
            if match:
                suffixes.append((rec, match[1], match[2], int(match[3]), match[4]))
    for (field, base), entries in pending.items():
        candidates = references[field, base]
        if len(candidates) != 1:
            for rec, *_ in entries:
                reject(rec, "description_WT_reference_not_unique")
            continue
        wt = next(iter(candidates))
        try:
            offset = _numbering_offset(
                wt, [(pos, before) for _, before, pos, _ in entries]
            )
        except ValueError:
            for rec, *_ in entries:
                reject(rec, "description_numbering_unresolved")
            continue
        mutants = {}
        for rec, before, pos, after in entries:
            mutation = (before, pos, after)
            if mutation not in mutants:
                mutants[mutation] = _mutate(wt, [mutation], offset)
            rec[field] = mutants[mutation]
            stats["sequence_descriptions_resolved"] += 1
    for (ab_base, ag_base), entries in families.items():
        ab_refs = paired_references["ab", _name_key(ab_base) + "wt"]
        ag_refs = paired_references["ag", _name_key(ag_base) + "wt"]
        if len(ab_refs) != 1 or len(ag_refs) != 1:
            for rec, _ in entries:
                reject(rec, "name_mutation_WT_reference_not_unique")
            continue
        heavy, light = next(iter(ab_refs))
        antigen = next(iter(ag_refs))[0]
        wt = dict(zip(SEQUENCE_FIELDS, (heavy, light, antigen)))
        ab_chains, ag_chain = (ab_base.rsplit("_", 1)[-1], ag_base.rsplit("_", 1)[-1])
        if (
            len(ab_chains) != 2
            or len(set(ab_chains)) != 2
            or len(ag_chain) != 1
            or set(ab_chains) & set(ag_chain)
        ):
            for rec, _ in entries:
                reject(rec, "ambiguous_chain_identifiers")
            continue
        constraints = defaultdict(set)
        for _, mutations in entries:
            for before, chain, pos, _ in mutations:
                constraints[chain].add((pos, before))
        candidates = {}
        for fields in permutations(SEQUENCE_FIELDS[:2]):
            mapping = dict(zip(ab_chains, fields))
            mapping[ag_chain] = "antigen_seq"
            try:
                offsets = {
                    chain: _numbering_offset(wt[mapping[chain]], cs)
                    for chain, cs in constraints.items()
                }
            except (KeyError, ValueError):
                continue
            key = tuple(
                sorted(
                    ((chain, mapping[chain], offsets[chain]) for chain in constraints)
                )
            )
            candidates[key] = (mapping, offsets)
        if len(candidates) != 1:
            for rec, _ in entries:
                reject(rec, "chain_or_numbering_unresolved")
            continue
        mapping, offsets = next(iter(candidates.values()))
        for rec, mutations in entries:
            per_chain = defaultdict(list)
            for before, chain, pos, after in mutations:
                per_chain[chain].append((before, pos, after))
            try:
                updated = {
                    mapping[c]: _mutate(wt[mapping[c]], ms, offsets[c])
                    for c, ms in per_chain.items()
                }
            except ValueError:
                reject(rec, "name_mutations_conflict")
                continue
            expected = tuple((updated.get(f, wt[f]) for f in SEQUENCE_FIELDS))
            provided = tuple((rec[f] for f in SEQUENCE_FIELDS))
            if any(
                (
                    value not in (original, target)
                    for value, original, target in zip(
                        provided, (heavy, light, antigen), expected
                    )
                )
            ):
                reject(rec, "provided_sequence_neither_WT_nor_expected_MT")
                continue
            stats["name_MT_rows_restored"] += any(
                (rec[f] != value for f, value in updated.items())
            )
            rec.update(updated)
    for rec, base, before, pos, after in suffixes:
        candidates = references["antigen_seq", _name_key(base)]
        if len(candidates) != 1:
            continue
        wt = next(iter(candidates))
        if (
            len(wt) != len(rec["antigen_seq"])
            or not 0 <= pos - 1 < len(wt)
            or wt[pos - 1] != before
        ):
            continue
        expected = _mutate(wt, [(before, pos, after)], 0)
        if rec["antigen_seq"] == wt:
            rec["antigen_seq"] = expected
            stats["name_MT_rows_restored"] += expected != wt
    for rec in records:
        rec["antibody_seq"] = rec["heavy_chain_seq"] + rec["light_chain_seq"]
        if not rec["antibody_seq"] or not rec["antigen_seq"]:
            reject(rec, "missing_sequence")
        elif any(
            (not set(rec[f]) <= AA_ALPHABET for f in ("antibody_seq", "antigen_seq"))
        ):
            reject(rec, "non_AA_sequence")
    valid = [rec for rec in records if rec["sample_id"] not in rejected]
    return (
        valid,
        {
            **dict(stats),
            "removed_rows": len(rejected),
            "removal_reasons": dict(Counter(rejected.values())),
            "rejected_raw_rows": [i + 1 for i in sorted(rejected)],
        },
    )
