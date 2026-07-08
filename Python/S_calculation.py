#!/usr/bin/env python3
"""
Compute MRCA clade sizes for unphased VCF carriers in a tskit tree sequence.

For each VCF row on one chromosome:
  1. Read carrier individuals from the VCF GT field.
  2. Map VCF sample names to tskit individual metadata names.
     If only some carrier samples map, continue with the mapped carriers and
     record the missing carrier counts in the output.
  3. For unphased heterozygotes, treat both diploid nodes as candidate haplotypes.
  4. Phase-correct by choosing one candidate haplotype per carrier group that lies
     in the smallest covering clade.
  5. Minimize the phase-corrected clade size across flanking local trees in:
        (x - window_bp, x - exclude_bp) U (x + exclude_bp, x + window_bp)
     where x is the VCF position plus pos_offset.

The output is a tab-separated summary table.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tskit


SAMPLE_ID_RE = re.compile(r"\b(?:HG|NA|GM)\d+\b")


def open_text(path: Path):
    """Open plain text or gzip-compressed text."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom


def chrom_matches(row_chrom: str, wanted_chrom: Optional[str]) -> bool:
    if wanted_chrom is None:
        return True
    return normalize_chrom(row_chrom) == normalize_chrom(wanted_chrom)


def parse_info(info_string: str) -> Dict[str, str]:
    info = {}
    for item in info_string.split(";"):
        if item == "":
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = "1"
    return info


def get_vcf_samples(vcf_path: Path) -> List[str]:
    with open_text(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 10:
                    raise ValueError("VCF header has no sample columns")
                return fields[9:]
    raise ValueError(f"No #CHROM header line found in {vcf_path}")


def metadata_to_names(metadata) -> set[str]:
    """Extract possible sample names from tskit metadata."""
    names = set()

    if isinstance(metadata, bytes):
        text = metadata.decode("utf-8", errors="ignore")
        try:
            decoded = json.loads(text)
            names.update(metadata_to_names(decoded))
        except Exception:
            pass
        names.update(SAMPLE_ID_RE.findall(text))
        return names

    if isinstance(metadata, str):
        names.add(metadata)
        names.update(SAMPLE_ID_RE.findall(metadata))
        try:
            decoded = json.loads(metadata)
            names.update(metadata_to_names(decoded))
        except Exception:
            pass
        return names

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            key_lower = str(key).lower()
            if key_lower in {"name", "id", "sample", "sample_id", "individual", "individual_id", "vcf_id"}:
                if not isinstance(value, (dict, list, tuple)):
                    names.add(str(value))
            names.update(metadata_to_names(value))
        return names

    if isinstance(metadata, (list, tuple)):
        for value in metadata:
            names.update(metadata_to_names(value))
        return names

    if metadata is not None:
        text = str(metadata)
        names.update(SAMPLE_ID_RE.findall(text))

    return {name for name in names if name and name != "None"}


def sample_nodes_for_individual(ts: tskit.TreeSequence, individual) -> List[int]:
    nodes = []
    sample_set = set(map(int, ts.samples()))
    for node_id in individual.nodes:
        node_id = int(node_id)
        if node_id in sample_set:
            nodes.append(node_id)
    return nodes


def build_sample_to_nodes(ts: tskit.TreeSequence) -> Dict[str, List[int]]:
    """Map metadata sample name to sample node IDs."""
    out: Dict[str, List[int]] = {}
    for individual in ts.individuals():
        nodes = sample_nodes_for_individual(ts, individual)
        if len(nodes) == 0:
            continue
        for name in metadata_to_names(individual.metadata):
            out.setdefault(name, nodes)
    return out


def build_node_to_sample(sample_to_nodes: Dict[str, List[int]]) -> Dict[int, str]:
    node_to_sample = {}
    for sample, nodes in sample_to_nodes.items():
        for node in nodes:
            node_to_sample[int(node)] = sample
    return node_to_sample


def genotype_alt_count(gt: str, alt_allele: str) -> Optional[int]:
    """Return number of requested ALT alleles in a GT field, or None if missing."""
    if gt in {"", ".", "./.", ".|."}:
        return None
    alleles = re.split(r"[/|]", gt)
    if any(a == "." or a == "" for a in alleles):
        return None
    if alt_allele.upper() == "ANY":
        return sum(1 for a in alleles if a not in {"0", "."})
    return sum(1 for a in alleles if a == alt_allele)


def parse_gt_gq(fmt_keys: List[str], sample_field: str) -> Tuple[Optional[str], Optional[float]]:
    values = sample_field.split(":")
    if "GT" not in fmt_keys:
        return None, None
    gt_index = fmt_keys.index("GT")
    if gt_index >= len(values):
        return None, None
    gt = values[gt_index]
    gq = None
    if "GQ" in fmt_keys:
        gq_index = fmt_keys.index("GQ")
        if gq_index < len(values) and values[gq_index] not in {"", "."}:
            try:
                gq = float(values[gq_index])
            except ValueError:
                gq = None
    return gt, gq


def carrier_groups_for_sample(
    sample_id: str,
    nodes: List[int],
    alt_count: int,
    homo_alt_policy: str,
) -> List[Tuple[str, List[int]]]:
    """
    Return groups for the phase-corrected covering-clade problem.

    A group means: the chosen clade must contain at least one node from this group.

    For 0/1 unphased genotypes, use one group with both haplotypes as candidates.
    For 1/1 genotypes, default policy is to require both haplotypes by creating two
    singleton groups. Use homo_alt_policy='one' to mimic one-representative-per-individual.
    """
    clean_nodes = [int(u) for u in nodes]
    if alt_count <= 0 or len(clean_nodes) == 0:
        return []

    if alt_count >= 2 and homo_alt_policy == "both":
        groups = []
        for i, node in enumerate(clean_nodes[:alt_count]):
            groups.append((f"{sample_id}:hap{i}", [node]))
        return groups

    return [(sample_id, clean_nodes)]


def mrca_of_nodes(tree: tskit.Tree, nodes: Sequence[int]) -> int:
    nodes = [int(u) for u in nodes]
    if len(nodes) == 0:
        return tskit.NULL
    mrca = nodes[0]
    for node in nodes[1:]:
        mrca = tree.mrca(mrca, int(node))
        if mrca == tskit.NULL:
            return tskit.NULL
    return int(mrca)


def interval_left_right(tree: tskit.Tree) -> Tuple[float, float]:
    interval = tree.interval
    if hasattr(interval, "left"):
        return float(interval.left), float(interval.right)
    return float(interval[0]), float(interval[1])


def select_smallest_covering_clade_by_groups(
    tree: tskit.Tree,
    groups: Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[int, List[int], int]:
    """
    Find smallest clade containing at least one candidate node from every group.

    Returns:
      best_node, selected_haps, best_size
    """
    groups = [(str(label), [int(u) for u in nodes]) for label, nodes in groups if len(nodes) > 0]
    if len(groups) == 0:
        return tskit.NULL, [], 0

    labels = [label for label, _nodes in groups]
    label_to_bit = {label: i for i, label in enumerate(labels)}
    full_mask = (1 << len(labels)) - 1

    node_bits = defaultdict(int)
    for label, nodes in groups:
        bit = 1 << label_to_bit[label]
        for node in nodes:
            node_bits[int(node)] |= bit

    ts = tree.tree_sequence
    mask = np.zeros(ts.num_nodes, dtype=object)

    for u in tree.nodes(order="postorder"):
        u = int(u)
        if tree.is_sample(u):
            mask[u] = node_bits.get(u, 0)
        else:
            m = 0
            for v in tree.children(u):
                m |= mask[int(v)]
            mask[u] = m

    roots = [int(r) for r in tree.roots if mask[int(r)] == full_mask]
    if len(roots) == 0:
        return tskit.NULL, [], 0

    queue = deque(roots)
    terminal_candidates = []
    while queue:
        u = queue.popleft()
        qualifying_children = [int(v) for v in tree.children(u) if mask[int(v)] == full_mask]
        if qualifying_children:
            queue.extend(qualifying_children)
        else:
            terminal_candidates.append(u)

    best_node = min(terminal_candidates, key=lambda u: (tree.num_samples(u), tree.time(u), u))
    best_size = int(tree.num_samples(best_node))

    selected_by_label: Dict[str, int] = {}
    for h in tree.samples(best_node):
        h = int(h)
        bits = node_bits.get(h, 0)
        if bits == 0:
            continue
        for label in labels:
            if label in selected_by_label:
                continue
            bit = 1 << label_to_bit[label]
            if bits & bit:
                selected_by_label[label] = h
                break
        if len(selected_by_label) == len(labels):
            break

    selected_haps = [selected_by_label[label] for label in labels if label in selected_by_label]
    return int(best_node), selected_haps, best_size


def descendant_vcf_sample_count(tree: tskit.Tree, node: int, node_to_sample: Dict[int, str]) -> int:
    names = set()
    for u in tree.samples(node):
        name = node_to_sample.get(int(u))
        if name is not None:
            names.add(name)
    return len(names)


class TreeIntervalIndex:
    def __init__(self, ts: tskit.TreeSequence):
        lefts = []
        rights = []
        for tree in ts.trees():
            left, right = interval_left_right(tree)
            lefts.append(left)
            rights.append(right)
        self.lefts = np.array(lefts, dtype=np.float64)
        self.rights = np.array(rights, dtype=np.float64)

    def overlapping_indices(self, left: float, right: float) -> np.ndarray:
        if right <= left:
            return np.array([], dtype=np.int64)
        return np.flatnonzero((self.rights > left) & (self.lefts < right)).astype(np.int64)


def flanking_intervals(x: float, sequence_length: float, window_bp: float, exclude_bp: float) -> List[Tuple[float, float, str]]:
    intervals = []
    left_a = max(0.0, x - window_bp)
    left_b = max(0.0, x - exclude_bp)
    right_a = min(float(sequence_length), x + exclude_bp)
    right_b = min(float(sequence_length), x + window_bp)
    if left_a < left_b:
        intervals.append((left_a, left_b, "left"))
    if right_a < right_b:
        intervals.append((right_a, right_b, "right"))
    return intervals


def side_for_tree(left: float, right: float, x: float, exclude_bp: float) -> str:
    if right <= x - exclude_bp:
        return "left"
    if left >= x + exclude_bp:
        return "right"
    return "boundary_spanning"


def process_variant_row(
    ts: tskit.TreeSequence,
    tree_index: TreeIntervalIndex,
    node_to_sample: Dict[int, str],
    sample_to_nodes: Dict[str, List[int]],
    vcf_samples: Sequence[str],
    fields: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    chrom = fields[0]
    pos = int(fields[1])
    query_pos = float(pos + args.pos_offset)
    variant_id = fields[2]
    ref = fields[3]
    alt = fields[4]
    filt = fields[6]
    info = parse_info(fields[7])
    fmt_keys = fields[8].split(":")

    base_row: Dict[str, object] = {
        "chrom": chrom,
        "pos": pos,
        "query_pos": query_pos,
        "id": variant_id,
        "ref_len": len(ref),
        "alt_len": len(alt),
        "ins_len": len(alt) - len(ref),
        "filter": filt,
        "info_AC": info.get("AC", ""),
        "info_AN": info.get("AN", ""),
        "info_AF": info.get("AF", ""),
        "info_MAF": info.get("MAF", ""),
        "ABBA_BEST_TIER": info.get("ABBA_BEST_TIER", ""),
        "ABBA_CLASSES": info.get("ABBA_CLASSES", ""),
        "ABBA_SOURCE_INTERVALS": info.get("ABBA_SOURCE_INTERVALS", ""),
        "ABBA_CROSS_CHR": info.get("ABBA_CROSS_CHR", ""),
        "format": fields[8],
        "num_carrier_samples": 0,
        "alt_alleles_from_gt": 0,
        "num_mapped_carrier_samples": 0,
        "mapped_alt_alleles_from_gt": 0,
        "num_groups_for_phasing": 0,
        "num_candidate_haps": 0,
        "missing_carrier_samples": 0,
        "missing_alt_alleles_from_gt": 0,
        "carrier_mapping_status": "complete",
        "focal_all_candidate_mrca_node": "",
        "focal_all_candidate_mrca_time": "",
        "focal_all_candidate_num_sample_nodes_under": "",
        "focal_all_candidate_num_vcf_samples_under": "",
        "focal_phase_node": "",
        "focal_phase_time": "",
        "focal_phase_num_sample_nodes_under": "",
        "focal_phase_num_vcf_samples_under": "",
        "flank_phase_best_node": "",
        "flank_phase_best_time": "",
        "flank_phase_best_num_sample_nodes_under": "",
        "flank_phase_best_num_vcf_samples_under": "",
        "flank_phase_best_tree_index": "",
        "flank_phase_best_tree_left": "",
        "flank_phase_best_tree_right": "",
        "flank_phase_best_side": "",
        "num_flank_trees_evaluated": 0,
        "status": "ok",
    }

    if args.include_carrier_list:
        base_row["carrier_samples"] = ""
    if args.include_selected_haps:
        base_row["flank_phase_selected_haps"] = ""

    if args.pass_only and filt not in {"PASS", "."}:
        base_row["status"] = "filtered_non_pass"
        return base_row

    if query_pos < 0 or query_pos >= ts.sequence_length:
        base_row["status"] = "position_outside_tree_sequence"
        return base_row

    if "GT" not in fmt_keys:
        base_row["status"] = "no_GT_in_FORMAT"
        return base_row

    groups: List[Tuple[str, List[int]]] = []
    candidate_haps = []
    carrier_samples = []
    mapped_carrier_samples = []
    missing_carriers = 0
    alt_alleles_from_gt = 0
    mapped_alt_alleles_from_gt = 0
    missing_alt_alleles_from_gt = 0

    for sample_id, sample_field in zip(vcf_samples, fields[9:]):
        gt, gq = parse_gt_gq(fmt_keys, sample_field)
        if gt is None:
            continue
        if args.min_gq is not None and (gq is None or gq < args.min_gq):
            continue
        alt_count = genotype_alt_count(gt, args.alt_allele)
        if alt_count is None or alt_count == 0:
            continue
        alt_alleles_from_gt += int(alt_count)
        carrier_samples.append(sample_id)
        nodes = sample_to_nodes.get(sample_id)
        if nodes is None:
            missing_carriers += 1
            missing_alt_alleles_from_gt += int(alt_count)
            continue
        mapped_carrier_samples.append(sample_id)
        mapped_alt_alleles_from_gt += int(alt_count)
        new_groups = carrier_groups_for_sample(
            sample_id=sample_id,
            nodes=nodes,
            alt_count=int(alt_count),
            homo_alt_policy=args.homo_alt_policy,
        )
        groups.extend(new_groups)
        for _label, group_nodes in new_groups:
            candidate_haps.extend(group_nodes)

    unique_candidate_haps = sorted(set(int(u) for u in candidate_haps))

    base_row["num_carrier_samples"] = len(carrier_samples)
    base_row["alt_alleles_from_gt"] = alt_alleles_from_gt
    base_row["num_mapped_carrier_samples"] = len(mapped_carrier_samples)
    base_row["mapped_alt_alleles_from_gt"] = mapped_alt_alleles_from_gt
    base_row["num_groups_for_phasing"] = len(groups)
    base_row["num_candidate_haps"] = len(unique_candidate_haps)
    base_row["missing_carrier_samples"] = missing_carriers
    base_row["missing_alt_alleles_from_gt"] = missing_alt_alleles_from_gt
    if missing_carriers > 0:
        base_row["carrier_mapping_status"] = "partial"
    if args.include_carrier_list:
        base_row["carrier_samples"] = ",".join(carrier_samples)
        base_row["mapped_carrier_sample_ids"] = ",".join(mapped_carrier_samples)
        missing_sample_ids = sorted(set(carrier_samples) - set(mapped_carrier_samples))
        base_row["missing_carrier_sample_ids"] = ",".join(missing_sample_ids)

    if len(carrier_samples) == 0:
        base_row["status"] = "no_carriers"
        return base_row
    if len(mapped_carrier_samples) == 0:
        base_row["status"] = "no_mapped_carriers"
        return base_row
    if args.max_carrier_samples is not None and len(mapped_carrier_samples) > args.max_carrier_samples:
        base_row["status"] = "too_many_carriers"
        return base_row
    if len(groups) == 0 or len(unique_candidate_haps) == 0:
        base_row["status"] = "no_candidate_haps"
        return base_row

    focal_tree = ts.at(query_pos)
    focal_mrca = mrca_of_nodes(focal_tree, unique_candidate_haps)
    if focal_mrca != tskit.NULL:
        base_row["focal_all_candidate_mrca_node"] = int(focal_mrca)
        base_row["focal_all_candidate_mrca_time"] = focal_tree.time(focal_mrca)
        base_row["focal_all_candidate_num_sample_nodes_under"] = int(focal_tree.num_samples(focal_mrca))
        base_row["focal_all_candidate_num_vcf_samples_under"] = descendant_vcf_sample_count(
            focal_tree, focal_mrca, node_to_sample
        )

    focal_phase_node, focal_selected, focal_phase_size = select_smallest_covering_clade_by_groups(focal_tree, groups)
    if focal_phase_node != tskit.NULL:
        base_row["focal_phase_node"] = int(focal_phase_node)
        base_row["focal_phase_time"] = focal_tree.time(focal_phase_node)
        base_row["focal_phase_num_sample_nodes_under"] = int(focal_phase_size)
        base_row["focal_phase_num_vcf_samples_under"] = descendant_vcf_sample_count(
            focal_tree, focal_phase_node, node_to_sample
        )

    intervals = flanking_intervals(query_pos, ts.sequence_length, args.window_bp, args.exclude_bp)
    tree_ids = []
    for left, right, _side in intervals:
        tree_ids.extend(tree_index.overlapping_indices(left, right).tolist())
    tree_ids = sorted(set(int(i) for i in tree_ids))

    best = None
    evaluated = 0
    excluded_overlap_left = query_pos - args.exclude_bp
    excluded_overlap_right = query_pos + args.exclude_bp

    for tree_id in tree_ids:
        tree_left = float(tree_index.lefts[tree_id])
        tree_right = float(tree_index.rights[tree_id])

        if args.strict_exclude_overlapping_trees:
            overlaps_excluded_center = (tree_right > excluded_overlap_left) and (tree_left < excluded_overlap_right)
            if overlaps_excluded_center:
                continue

        tree = ts.at_index(tree_id)
        node, selected_haps, size = select_smallest_covering_clade_by_groups(tree, groups)
        evaluated += 1
        if node == tskit.NULL:
            continue
        record = {
            "size": int(size),
            "node": int(node),
            "time": float(tree.time(node)),
            "tree_id": int(tree_id),
            "tree_left": tree_left,
            "tree_right": tree_right,
            "side": side_for_tree(tree_left, tree_right, query_pos, args.exclude_bp),
            "num_vcf_samples_under": descendant_vcf_sample_count(tree, node, node_to_sample),
            "selected_haps": selected_haps,
            "distance_to_focal": min(abs(tree_left - query_pos), abs(tree_right - query_pos)),
        }
        if best is None:
            best = record
        else:
            old_key = (best["size"], best["time"], best["distance_to_focal"], best["tree_id"])
            new_key = (record["size"], record["time"], record["distance_to_focal"], record["tree_id"])
            if new_key < old_key:
                best = record

    base_row["num_flank_trees_evaluated"] = evaluated
    if best is None:
        base_row["status"] = "no_flanking_phase_clade"
        return base_row

    base_row["flank_phase_best_node"] = best["node"]
    base_row["flank_phase_best_time"] = best["time"]
    base_row["flank_phase_best_num_sample_nodes_under"] = best["size"]
    base_row["flank_phase_best_num_vcf_samples_under"] = best["num_vcf_samples_under"]
    base_row["flank_phase_best_tree_index"] = best["tree_id"]
    base_row["flank_phase_best_tree_left"] = best["tree_left"]
    base_row["flank_phase_best_tree_right"] = best["tree_right"]
    base_row["flank_phase_best_side"] = best["side"]
    if args.include_selected_haps:
        base_row["flank_phase_selected_haps"] = ",".join(map(str, best["selected_haps"]))

    return base_row


def output_columns(include_carrier_list: bool, include_selected_haps: bool) -> List[str]:
    cols = [
        "chrom",
        "pos",
        "query_pos",
        "id",
        "ref_len",
        "alt_len",
        "ins_len",
        "filter",
        "info_AC",
        "info_AN",
        "info_AF",
        "info_MAF",
        "ABBA_BEST_TIER",
        "ABBA_CLASSES",
        "ABBA_SOURCE_INTERVALS",
        "ABBA_CROSS_CHR",
        "format",
        "num_carrier_samples",
        "alt_alleles_from_gt",
        "num_mapped_carrier_samples",
        "mapped_alt_alleles_from_gt",
        "num_groups_for_phasing",
        "num_candidate_haps",
        "missing_carrier_samples",
        "missing_alt_alleles_from_gt",
        "carrier_mapping_status",
        "focal_all_candidate_mrca_node",
        "focal_all_candidate_mrca_time",
        "focal_all_candidate_num_sample_nodes_under",
        "focal_all_candidate_num_vcf_samples_under",
        "focal_phase_node",
        "focal_phase_time",
        "focal_phase_num_sample_nodes_under",
        "focal_phase_num_vcf_samples_under",
        "flank_phase_best_node",
        "flank_phase_best_time",
        "flank_phase_best_num_sample_nodes_under",
        "flank_phase_best_num_vcf_samples_under",
        "flank_phase_best_tree_index",
        "flank_phase_best_tree_left",
        "flank_phase_best_tree_right",
        "flank_phase_best_side",
        "num_flank_trees_evaluated",
        "status",
    ]
    if include_carrier_list:
        cols.append("carrier_samples")
        cols.append("mapped_carrier_sample_ids")
        cols.append("missing_carrier_sample_ids")
    if include_selected_haps:
        cols.append("flank_phase_selected_haps")
    return cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-corrected MRCA clade size for unphased VCF carriers in a tskit tree sequence."
    )
    parser.add_argument("--tree-sequence", required=True, type=Path, help="Input .trees file for one chromosome")
    parser.add_argument("--vcf", required=True, type=Path, help="Genome-wide or per-chromosome VCF, plain or .gz")
    parser.add_argument("--chrom", required=True, help="Chromosome to process, e.g. 1 or chr1")
    parser.add_argument("--out", required=True, type=Path, help="Output TSV")
    parser.add_argument("--window-bp", type=float, default=75000.0, help="Half window around VCF position")
    parser.add_argument("--exclude-bp", type=float, default=200.0, help="Central exclusion half-width around VCF position")
    parser.add_argument("--pos-offset", type=int, default=0, help="Add this to VCF POS before tree lookup")
    parser.add_argument("--alt-allele", default="1", help='ALT allele index to count, or "ANY"')
    parser.add_argument("--min-gq", type=float, default=None, help="Optional minimum GQ for carrier genotypes")
    parser.add_argument(
        "--max-carrier-samples",
        type=int,
        default=None,
        help="Skip variants with more mapped carrier samples than this",
    )
    parser.add_argument(
        "--homo-alt-policy",
        choices=["both", "one"],
        default="both",
        help="For 1/1 genotypes, require both haplotypes or one representative haplotype",
    )
    parser.add_argument("--pass-only", action="store_true", help="Only process FILTER=PASS or FILTER=.")
    parser.add_argument(
        "--strict-exclude-overlapping-trees",
        action="store_true",
        help="Skip any tree interval that overlaps the central excluded region at all",
    )
    parser.add_argument("--include-carrier-list", action="store_true", help="Include carrier sample IDs in output")
    parser.add_argument("--include-selected-haps", action="store_true", help="Include selected haplotype node IDs in output")
    parser.add_argument("--max-variants", type=int, default=None, help="Process at most this many matching VCF rows")
    args = parser.parse_args()

    if args.exclude_bp < 0 or args.window_bp <= 0 or args.exclude_bp >= args.window_bp:
        raise ValueError("Require 0 <= exclude_bp < window_bp")

    print(f"Loading tree sequence: {args.tree_sequence}", file=sys.stderr)
    ts = tskit.load(args.tree_sequence)
    print(
        f"Tree sequence: samples={ts.num_samples} individuals={ts.num_individuals} "
        f"trees={ts.num_trees} sites={ts.num_sites} sequence_length={ts.sequence_length}",
        file=sys.stderr,
    )

    print("Building tree interval index", file=sys.stderr)
    tree_index = TreeIntervalIndex(ts)

    print("Building sample-name to node mapping from individual metadata", file=sys.stderr)
    sample_to_nodes = build_sample_to_nodes(ts)
    node_to_sample = build_node_to_sample(sample_to_nodes)
    print(f"Mapped named individuals: {len(sample_to_nodes)}", file=sys.stderr)

    vcf_samples = get_vcf_samples(args.vcf)
    missing_vcf_samples = [s for s in vcf_samples if s not in sample_to_nodes]
    print(f"VCF samples: {len(vcf_samples)}", file=sys.stderr)
    if missing_vcf_samples:
        print(
            f"WARNING: {len(missing_vcf_samples)} VCF samples are not in tree metadata. "
            f"Examples: {missing_vcf_samples[:10]}",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = output_columns(args.include_carrier_list, args.include_selected_haps)

    processed = 0
    seen_matching_chrom = 0
    with open_text(args.vcf) as fin, open(args.out, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for line in fin:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            if not chrom_matches(fields[0], args.chrom):
                continue
            seen_matching_chrom += 1
            row = process_variant_row(
                ts=ts,
                tree_index=tree_index,
                node_to_sample=node_to_sample,
                sample_to_nodes=sample_to_nodes,
                vcf_samples=vcf_samples,
                fields=fields,
                args=args,
            )
            writer.writerow(row)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} variants", file=sys.stderr)
            if args.max_variants is not None and processed >= args.max_variants:
                break

    print(f"Matching VCF rows seen on chrom {args.chrom}: {seen_matching_chrom}", file=sys.stderr)
    print(f"Rows written: {processed}", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
