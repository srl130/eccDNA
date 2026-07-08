#!/usr/bin/env python3
"""
Simulate phase-corrected MRCA nulls by allele-count/frequency class.

For each observed TSV row on a chromosome, this script:
  1. Reads k from the frequency column, usually info_AC.
  2. Infers the observed heterozygous/homozygous carrier structure from
     num_mapped_carrier_samples and mapped_alt_alleles_from_gt when those
     columns are present, otherwise from num_carrier_samples and
     alt_alleles_from_gt.
  3. Samples random null carriers with the same het/hom structure by default.
  4. Runs the same phase-corrected focal/flanking clade search used by
     S_calculation.py.

The default null is therefore conditioned on the observed variant positions,
frequency class, local tree topology, and observed het/hom carrier structure.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tskit

from vcf_phase_corrected_mrca import (
    TreeIntervalIndex,
    carrier_groups_for_sample,
    chrom_matches,
    descendant_vcf_sample_count,
    flanking_intervals,
    select_smallest_covering_clade_by_groups,
    side_for_tree,
)


EXACT_GROUP_LIMIT = 10
EXACT_COMBO_LIMIT = 65536


def parse_int_set(text: Optional[str]) -> Optional[set[int]]:
    if text is None or text.strip() == "":
        return None
    out = set()
    for item in text.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def parse_status_set(text: Optional[str]) -> Optional[set[str]]:
    if text is None or text.strip() == "":
        return None
    return {item.strip() for item in text.split(",") if item.strip()}


def load_observed_rows(
    path: Path,
    chrom: str,
    frequency_col: str,
    position_col: str,
    status_filter: Optional[set[str]],
    frequencies: Optional[set[int]],
    min_frequency: int,
    max_rows: Optional[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "rt", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        carrier_count_col = (
            "num_mapped_carrier_samples" if "num_mapped_carrier_samples" in fieldnames else "num_carrier_samples"
        )
        alt_allele_count_col = (
            "mapped_alt_alleles_from_gt" if "mapped_alt_alleles_from_gt" in fieldnames else "alt_alleles_from_gt"
        )
        required = {
            frequency_col,
            position_col,
            "chrom",
            "pos",
            carrier_count_col,
            alt_allele_count_col,
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Observed TSV is missing required columns: {missing}")

        for i, row in enumerate(reader, start=1):
            if not chrom_matches(str(row["chrom"]), chrom):
                continue
            if status_filter is not None and row.get("status", "") not in status_filter:
                continue

            raw_k = row.get(frequency_col, "")
            raw_x = row.get(position_col, "")
            if raw_k in {"", None} or raw_x in {"", None}:
                continue

            k = int(float(raw_k))
            num_carriers = int(float(row[carrier_count_col]))
            alt_alleles_from_gt = int(float(row[alt_allele_count_col]))
            num_hom = alt_alleles_from_gt - num_carriers
            num_het = num_carriers - num_hom
            if num_hom < 0 or num_het < 0:
                raise ValueError(
                    f"Invalid het/hom counts at source row {i}: "
                    f"{carrier_count_col}={num_carriers}, "
                    f"{alt_allele_count_col}={alt_alleles_from_gt}"
                )
            if k < min_frequency:
                continue
            if frequencies is not None and k not in frequencies:
                continue

            rows.append(
                {
                    "source_row": i,
                    "source_id": row.get("id", ""),
                    "chrom": row["chrom"],
                    "pos": int(float(row["pos"])),
                    "query_pos": float(raw_x),
                    "frequency_class": k,
                    "observed_num_carrier_samples": num_carriers,
                    "observed_alt_alleles_from_gt": alt_alleles_from_gt,
                    "observed_het_carriers": num_het,
                    "observed_hom_carriers": num_hom,
                    "observed_status": row.get("status", ""),
                    "observed_flank_size": row.get("flank_phase_best_num_sample_nodes_under", ""),
                    "observed_focal_size": row.get("focal_phase_num_sample_nodes_under", ""),
                }
            )

            if max_rows is not None and len(rows) >= max_rows:
                break

    return rows


def build_individual_node_maps(
    ts: tskit.TreeSequence,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, List[int]], Dict[int, str]]:
    sample_nodes = np.array(list(map(int, ts.samples())), dtype=np.int64)
    sample_node_set = set(map(int, sample_nodes))
    node_to_individual: Dict[int, int] = {}
    individual_to_nodes: Dict[int, List[int]] = {}
    node_to_individual_label: Dict[int, str] = {}

    for individual in ts.individuals():
        nodes = [int(u) for u in individual.nodes if int(u) in sample_node_set]
        if not nodes:
            continue
        individual_id = int(individual.id)
        individual_to_nodes[individual_id] = nodes
        for node in nodes:
            node_to_individual[node] = individual_id
            node_to_individual_label[node] = str(individual_id)

    usable_sample_nodes = np.array(
        [int(u) for u in sample_nodes if int(u) in node_to_individual],
        dtype=np.int64,
    )
    return usable_sample_nodes, node_to_individual, individual_to_nodes, node_to_individual_label


def groups_from_sampled_haplotypes(
    sampled_haps: Sequence[int],
    node_to_individual: Dict[int, int],
    individual_to_nodes: Dict[int, List[int]],
    homo_alt_policy: str,
) -> Tuple[List[Tuple[str, List[int]]], int, int, int]:
    haps_by_individual: Dict[int, List[int]] = defaultdict(list)
    for hap in sampled_haps:
        haps_by_individual[node_to_individual[int(hap)]].append(int(hap))

    groups: List[Tuple[str, List[int]]] = []
    candidate_haps = set()
    num_homozygous_carriers = 0

    for individual_id in sorted(haps_by_individual):
        alt_count = len(haps_by_individual[individual_id])
        if alt_count >= 2:
            num_homozygous_carriers += 1
        nodes = individual_to_nodes[individual_id]
        sample_id = f"ind_{individual_id}"
        new_groups = carrier_groups_for_sample(
            sample_id=sample_id,
            nodes=nodes,
            alt_count=alt_count,
            homo_alt_policy=homo_alt_policy,
        )
        groups.extend(new_groups)
        for _label, group_nodes in new_groups:
            candidate_haps.update(map(int, group_nodes))

    return groups, len(haps_by_individual), num_homozygous_carriers, len(candidate_haps)


def groups_from_matched_het_hom_draw(
    rng: np.random.Generator,
    usable_individual_ids: np.ndarray,
    individual_to_nodes: Dict[int, List[int]],
    num_het: int,
    num_hom: int,
    homo_alt_policy: str,
) -> Tuple[np.ndarray, List[Tuple[str, List[int]]], int, int, int]:
    num_carriers = num_het + num_hom
    if num_carriers <= 0:
        return np.array([], dtype=np.int64), [], 0, 0, 0
    if num_carriers > len(usable_individual_ids):
        raise ValueError(
            f"Need {num_carriers} carrier individuals for matched null, "
            f"but only {len(usable_individual_ids)} usable diploid individuals exist"
        )

    chosen_individuals = rng.choice(usable_individual_ids, size=num_carriers, replace=False)
    hom_individuals = chosen_individuals[:num_hom]
    het_individuals = chosen_individuals[num_hom:]

    sampled_haps: List[int] = []
    groups: List[Tuple[str, List[int]]] = []
    candidate_haps = set()

    for individual_id in sorted(map(int, hom_individuals)):
        nodes = list(map(int, individual_to_nodes[individual_id]))
        sampled_haps.extend(nodes[:2])
        new_groups = carrier_groups_for_sample(
            sample_id=f"ind_{individual_id}",
            nodes=nodes,
            alt_count=2,
            homo_alt_policy=homo_alt_policy,
        )
        groups.extend(new_groups)
        for _label, group_nodes in new_groups:
            candidate_haps.update(map(int, group_nodes))

    for individual_id in sorted(map(int, het_individuals)):
        nodes = list(map(int, individual_to_nodes[individual_id]))
        sampled_haps.append(int(rng.choice(np.array(nodes, dtype=np.int64))))
        new_groups = carrier_groups_for_sample(
            sample_id=f"ind_{individual_id}",
            nodes=nodes,
            alt_count=1,
            homo_alt_policy=homo_alt_policy,
        )
        groups.extend(new_groups)
        for _label, group_nodes in new_groups:
            candidate_haps.update(map(int, group_nodes))

    return (
        np.array(sampled_haps, dtype=np.int64),
        groups,
        num_carriers,
        num_hom,
        len(candidate_haps),
    )


def combination_count(groups: Sequence[Tuple[str, Sequence[int]]]) -> int:
    total = 1
    for _label, nodes in groups:
        total *= len(nodes)
        if total > EXACT_COMBO_LIMIT:
            return total
    return total


def iter_group_choices(groups: Sequence[Tuple[str, Sequence[int]]]):
    if not groups:
        yield []
        return

    choices = [list(map(int, nodes)) for _label, nodes in groups]
    indices = [0] * len(choices)
    while True:
        yield [choices[i][indices[i]] for i in range(len(choices))]

        j = len(indices) - 1
        while j >= 0:
            indices[j] += 1
            if indices[j] < len(choices[j]):
                break
            indices[j] = 0
            j -= 1
        if j < 0:
            return


def mrca_of_nodes(tree: tskit.Tree, nodes: Sequence[int]) -> int:
    if not nodes:
        return tskit.NULL
    mrca = int(nodes[0])
    for node in nodes[1:]:
        mrca = tree.mrca(mrca, int(node))
        if mrca == tskit.NULL:
            return tskit.NULL
    return int(mrca)


def select_covering_clade_for_null(
    tree: tskit.Tree,
    groups: Sequence[Tuple[str, Sequence[int]]],
) -> Tuple[int, List[int], int]:
    """
    Use exact phase enumeration for small variants and the tree-wide mask
    algorithm for larger variants.

    With the default matched het/hom null and homo_alt_policy=both,
    len(groups) is the number of ALT haplotypes that must be represented
    after phasing. An all-heterozygote row with k=10 has at most 2^10 phase
    choices, while k=15 has 2^15 choices and is too slow across many trees.
    """
    groups = [(label, list(map(int, nodes))) for label, nodes in groups if nodes]
    if not groups:
        return tskit.NULL, [], 0

    if len(groups) > EXACT_GROUP_LIMIT or combination_count(groups) > EXACT_COMBO_LIMIT:
        return select_smallest_covering_clade_by_groups(tree, groups)

    if len(groups) == 1:
        selected = [int(groups[0][1][0])]
        return selected[0], selected, 1

    best = None
    for selected_haps in iter_group_choices(groups):
        node = mrca_of_nodes(tree, selected_haps)
        if node == tskit.NULL:
            continue
        record = (int(tree.num_samples(node)), float(tree.time(node)), int(node), selected_haps)
        if best is None or record[:3] < best[:3]:
            best = record

    if best is None:
        return tskit.NULL, [], 0
    size, _time, node, selected_haps = best
    return int(node), list(map(int, selected_haps)), int(size)


def evaluate_groups(
    ts: tskit.TreeSequence,
    tree_index: TreeIntervalIndex,
    node_to_individual_label: Dict[int, str],
    groups: Sequence[Tuple[str, Sequence[int]]],
    query_pos: float,
    window_bp: float,
    exclude_bp: float,
    strict_exclude_overlapping_trees: bool,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "focal_phase_node": "",
        "focal_phase_time": "",
        "focal_phase_num_sample_nodes_under": "",
        "focal_phase_num_individuals_under": "",
        "flank_phase_best_node": "",
        "flank_phase_best_time": "",
        "flank_phase_best_num_sample_nodes_under": "",
        "flank_phase_best_num_individuals_under": "",
        "flank_phase_best_tree_index": "",
        "flank_phase_best_tree_left": "",
        "flank_phase_best_tree_right": "",
        "flank_phase_best_side": "",
        "num_flank_trees_evaluated": 0,
        "status": "ok",
    }

    if query_pos < 0 or query_pos >= ts.sequence_length:
        row["status"] = "position_outside_tree_sequence"
        return row

    if len(groups) == 1:
        hap = int(groups[0][1][0])
        focal_tree = ts.at(query_pos)
        row["focal_phase_node"] = hap
        row["focal_phase_time"] = focal_tree.time(hap)
        row["focal_phase_num_sample_nodes_under"] = 1
        row["focal_phase_num_individuals_under"] = 1
        row["flank_phase_best_node"] = hap
        row["flank_phase_best_time"] = focal_tree.time(hap)
        row["flank_phase_best_num_sample_nodes_under"] = 1
        row["flank_phase_best_num_individuals_under"] = 1
        return row

    focal_tree = ts.at(query_pos)
    focal_phase_node, _focal_selected, focal_phase_size = select_covering_clade_for_null(focal_tree, groups)
    if focal_phase_node != tskit.NULL:
        row["focal_phase_node"] = int(focal_phase_node)
        row["focal_phase_time"] = focal_tree.time(focal_phase_node)
        row["focal_phase_num_sample_nodes_under"] = int(focal_phase_size)
        row["focal_phase_num_individuals_under"] = descendant_vcf_sample_count(
            focal_tree, focal_phase_node, node_to_individual_label
        )

    tree_ids = []
    for left, right, _side in flanking_intervals(query_pos, ts.sequence_length, window_bp, exclude_bp):
        tree_ids.extend(tree_index.overlapping_indices(left, right).tolist())
    tree_ids = sorted(set(int(i) for i in tree_ids))

    best = None
    evaluated = 0
    excluded_overlap_left = query_pos - exclude_bp
    excluded_overlap_right = query_pos + exclude_bp

    tree = ts.first()
    have_positioned_tree = False
    for tree_id in tree_ids:
        tree_left = float(tree_index.lefts[tree_id])
        tree_right = float(tree_index.rights[tree_id])

        if strict_exclude_overlapping_trees:
            overlaps_excluded_center = (tree_right > excluded_overlap_left) and (
                tree_left < excluded_overlap_right
            )
            if overlaps_excluded_center:
                continue

        if not have_positioned_tree or tree.index != tree_id:
            tree.seek_index(tree_id)
            have_positioned_tree = True
        node, _selected_haps, size = select_covering_clade_for_null(tree, groups)
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
            "side": side_for_tree(tree_left, tree_right, query_pos, exclude_bp),
            "num_individuals_under": descendant_vcf_sample_count(tree, node, node_to_individual_label),
            "distance_to_focal": min(abs(tree_left - query_pos), abs(tree_right - query_pos)),
        }
        if best is None:
            best = record
        else:
            old_key = (best["size"], best["time"], best["distance_to_focal"], best["tree_id"])
            new_key = (record["size"], record["time"], record["distance_to_focal"], record["tree_id"])
            if new_key < old_key:
                best = record

    row["num_flank_trees_evaluated"] = evaluated
    if best is None:
        row["status"] = "no_flanking_phase_clade"
        return row

    row["flank_phase_best_node"] = best["node"]
    row["flank_phase_best_time"] = best["time"]
    row["flank_phase_best_num_sample_nodes_under"] = best["size"]
    row["flank_phase_best_num_individuals_under"] = best["num_individuals_under"]
    row["flank_phase_best_tree_index"] = best["tree_id"]
    row["flank_phase_best_tree_left"] = best["tree_left"]
    row["flank_phase_best_tree_right"] = best["tree_right"]
    row["flank_phase_best_side"] = best["side"]
    return row


def make_replicate_record(
    source: Dict[str, object],
    replicate: int,
    sampled: np.ndarray,
    groups: Sequence[Tuple[str, Sequence[int]]],
    num_carriers: int,
    num_homozygous: int,
    num_candidate_haps: int,
    include_sampled_haps: bool,
) -> Dict[str, object]:
    row = {
        **source,
        "replicate": replicate,
        "num_sampled_haplotypes": int(len(sampled)),
        "num_carrier_individuals": num_carriers,
        "num_homozygous_carriers": num_homozygous,
        "num_groups_for_phasing": len(groups),
        "num_candidate_haps": num_candidate_haps,
        "focal_phase_node": "",
        "focal_phase_time": "",
        "focal_phase_num_sample_nodes_under": "",
        "focal_phase_num_individuals_under": "",
        "flank_phase_best_node": "",
        "flank_phase_best_time": "",
        "flank_phase_best_num_sample_nodes_under": "",
        "flank_phase_best_num_individuals_under": "",
        "flank_phase_best_tree_index": "",
        "flank_phase_best_tree_left": "",
        "flank_phase_best_tree_right": "",
        "flank_phase_best_side": "",
        "num_flank_trees_evaluated": 0,
        "status": "ok",
    }
    if include_sampled_haps:
        row["sampled_haplotypes"] = ",".join(map(str, sampled.tolist()))
    return row


def evaluate_replicates_for_source(
    ts: tskit.TreeSequence,
    tree_index: TreeIntervalIndex,
    node_to_individual_label: Dict[int, str],
    replicate_items: List[Dict[str, object]],
    query_pos: float,
    window_bp: float,
    exclude_bp: float,
    strict_exclude_overlapping_trees: bool,
) -> List[Dict[str, object]]:
    if query_pos < 0 or query_pos >= ts.sequence_length:
        for item in replicate_items:
            item["row"]["status"] = "position_outside_tree_sequence"
        return [item["row"] for item in replicate_items]

    singleton_items = []
    nonsingleton_items = []
    focal_tree = ts.at(query_pos)

    for item in replicate_items:
        groups = item["groups"]
        row = item["row"]
        if len(groups) == 1:
            hap = int(groups[0][1][0])
            row["focal_phase_node"] = hap
            row["focal_phase_time"] = focal_tree.time(hap)
            row["focal_phase_num_sample_nodes_under"] = 1
            row["focal_phase_num_individuals_under"] = 1
            row["flank_phase_best_node"] = hap
            row["flank_phase_best_time"] = focal_tree.time(hap)
            row["flank_phase_best_num_sample_nodes_under"] = 1
            row["flank_phase_best_num_individuals_under"] = 1
            singleton_items.append(item)
            continue

        focal_phase_node, _focal_selected, focal_phase_size = select_covering_clade_for_null(
            focal_tree, groups
        )
        if focal_phase_node != tskit.NULL:
            row["focal_phase_node"] = int(focal_phase_node)
            row["focal_phase_time"] = focal_tree.time(focal_phase_node)
            row["focal_phase_num_sample_nodes_under"] = int(focal_phase_size)
            row["focal_phase_num_individuals_under"] = descendant_vcf_sample_count(
                focal_tree, focal_phase_node, node_to_individual_label
            )
        nonsingleton_items.append(item)

    if not nonsingleton_items:
        return [item["row"] for item in replicate_items]

    tree_ids = []
    for left, right, _side in flanking_intervals(query_pos, ts.sequence_length, window_bp, exclude_bp):
        tree_ids.extend(tree_index.overlapping_indices(left, right).tolist())
    tree_ids = sorted(set(int(i) for i in tree_ids))

    excluded_overlap_left = query_pos - exclude_bp
    excluded_overlap_right = query_pos + exclude_bp
    eligible_tree_ids = []
    for tree_id in tree_ids:
        tree_left = float(tree_index.lefts[tree_id])
        tree_right = float(tree_index.rights[tree_id])
        if strict_exclude_overlapping_trees:
            overlaps_excluded_center = (tree_right > excluded_overlap_left) and (
                tree_left < excluded_overlap_right
            )
            if overlaps_excluded_center:
                continue
        eligible_tree_ids.append(tree_id)

    best_by_item = [None] * len(nonsingleton_items)
    tree = ts.first()
    have_positioned_tree = False
    for tree_id in eligible_tree_ids:
        tree_left = float(tree_index.lefts[tree_id])
        tree_right = float(tree_index.rights[tree_id])
        if not have_positioned_tree or tree.index != tree_id:
            tree.seek_index(tree_id)
            have_positioned_tree = True

        distance_to_focal = min(abs(tree_left - query_pos), abs(tree_right - query_pos))
        side = side_for_tree(tree_left, tree_right, query_pos, exclude_bp)
        for i, item in enumerate(nonsingleton_items):
            node, _selected_haps, size = select_covering_clade_for_null(tree, item["groups"])
            if node == tskit.NULL:
                continue
            record = {
                "size": int(size),
                "node": int(node),
                "time": float(tree.time(node)),
                "tree_id": int(tree_id),
                "tree_left": tree_left,
                "tree_right": tree_right,
                "side": side,
                "distance_to_focal": distance_to_focal,
            }
            best = best_by_item[i]
            if best is None:
                best_by_item[i] = record
            else:
                old_key = (best["size"], best["time"], best["distance_to_focal"], best["tree_id"])
                new_key = (record["size"], record["time"], record["distance_to_focal"], record["tree_id"])
                if new_key < old_key:
                    best_by_item[i] = record

    tree_for_descendants = ts.first()
    have_descendant_tree = False
    for item, best in zip(nonsingleton_items, best_by_item):
        row = item["row"]
        row["num_flank_trees_evaluated"] = len(eligible_tree_ids)
        if best is None:
            row["status"] = "no_flanking_phase_clade"
            continue
        row["flank_phase_best_node"] = best["node"]
        row["flank_phase_best_time"] = best["time"]
        row["flank_phase_best_num_sample_nodes_under"] = best["size"]
        if not have_descendant_tree or tree_for_descendants.index != best["tree_id"]:
            tree_for_descendants.seek_index(best["tree_id"])
            have_descendant_tree = True
        row["flank_phase_best_num_individuals_under"] = descendant_vcf_sample_count(
            tree_for_descendants, best["node"], node_to_individual_label
        )
        row["flank_phase_best_tree_index"] = best["tree_id"]
        row["flank_phase_best_tree_left"] = best["tree_left"]
        row["flank_phase_best_tree_right"] = best["tree_right"]
        row["flank_phase_best_side"] = best["side"]

    return [item["row"] for item in replicate_items]


def output_columns(include_sampled_haps: bool) -> List[str]:
    cols = [
        "chrom",
        "pos",
        "query_pos",
        "source_row",
        "source_id",
        "observed_status",
        "observed_flank_size",
        "observed_focal_size",
        "frequency_class",
        "observed_num_carrier_samples",
        "observed_alt_alleles_from_gt",
        "observed_het_carriers",
        "observed_hom_carriers",
        "replicate",
        "num_sampled_haplotypes",
        "num_carrier_individuals",
        "num_homozygous_carriers",
        "num_groups_for_phasing",
        "num_candidate_haps",
        "focal_phase_node",
        "focal_phase_time",
        "focal_phase_num_sample_nodes_under",
        "focal_phase_num_individuals_under",
        "flank_phase_best_node",
        "flank_phase_best_time",
        "flank_phase_best_num_sample_nodes_under",
        "flank_phase_best_num_individuals_under",
        "flank_phase_best_tree_index",
        "flank_phase_best_tree_left",
        "flank_phase_best_tree_right",
        "flank_phase_best_side",
        "num_flank_trees_evaluated",
        "status",
    ]
    if include_sampled_haps:
        cols.append("sampled_haplotypes")
    return cols


def finite_float(value: object) -> Optional[float]:
    if value in {"", None}:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), q))


def empirical_pvalue(observed: float, null_values: Sequence[float], tail: str) -> float:
    values = np.array(null_values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan")

    lower = (int(np.sum(values <= observed)) + 1) / (n + 1)
    upper = (int(np.sum(values >= observed)) + 1) / (n + 1)
    if tail == "lower":
        return float(lower)
    if tail == "upper":
        return float(upper)
    if tail == "two-sided":
        return float(min(1.0, 2.0 * min(lower, upper)))
    raise ValueError(f"Unknown p-value tail: {tail}")


def write_summary(summary_values: Dict[int, Dict[str, List[float]]], summary_out: Path) -> None:
    cols = [
        "frequency_class",
        "num_null_replicates",
        "mean_flank_sample_nodes",
        "sd_flank_sample_nodes",
        "min_flank_sample_nodes",
        "q05_flank_sample_nodes",
        "median_flank_sample_nodes",
        "q95_flank_sample_nodes",
        "max_flank_sample_nodes",
        "mean_carrier_individuals",
        "mean_homozygous_carriers",
    ]
    with open(summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for k in sorted(summary_values):
            values = summary_values[k]["flank"]
            carrier_individuals = summary_values[k]["carrier_individuals"]
            homozygous = summary_values[k]["homozygous_carriers"]
            if not values:
                continue
            writer.writerow(
                {
                    "frequency_class": k,
                    "num_null_replicates": len(values),
                    "mean_flank_sample_nodes": float(np.mean(values)),
                    "sd_flank_sample_nodes": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min_flank_sample_nodes": min(values),
                    "q05_flank_sample_nodes": percentile(values, 5),
                    "median_flank_sample_nodes": percentile(values, 50),
                    "q95_flank_sample_nodes": percentile(values, 95),
                    "max_flank_sample_nodes": max(values),
                    "mean_carrier_individuals": float(np.mean(carrier_individuals)),
                    "mean_homozygous_carriers": float(np.mean(homozygous)),
                }
            )


def write_observed_pvalues(
    observed_rows: Sequence[Dict[str, object]],
    summary_values: Dict[int, Dict[str, List[float]]],
    pvalue_values_by_source_row: Dict[int, List[float]],
    pvalues_out: Path,
    pvalue_tail: str,
    carrier_structure: str,
) -> None:
    cols = [
        "chrom",
        "pos",
        "query_pos",
        "source_row",
        "source_id",
        "observed_status",
        "frequency_class",
        "observed_num_carrier_samples",
        "observed_alt_alleles_from_gt",
        "observed_het_carriers",
        "observed_hom_carriers",
        "carrier_structure",
        "observed_flank_size",
        "num_null_replicates",
        "pvalue_tail",
        "empirical_p_value",
        "null_mean_flank_sample_nodes",
        "null_median_flank_sample_nodes",
        "null_q05_flank_sample_nodes",
        "null_q95_flank_sample_nodes",
        "pvalue_status",
    ]

    with open(pvalues_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()

        for observed_row in observed_rows:
            k = int(observed_row["frequency_class"])
            observed = finite_float(observed_row.get("observed_flank_size"))
            source_row = int(observed_row["source_row"])
            null_values = pvalue_values_by_source_row.get(source_row, [])
            if not null_values:
                null_values = summary_values.get(k, {}).get("flank", [])

            out = {
                "chrom": observed_row["chrom"],
                "pos": observed_row["pos"],
                "query_pos": observed_row["query_pos"],
                "source_row": observed_row["source_row"],
                "source_id": observed_row["source_id"],
                "observed_status": observed_row["observed_status"],
                "frequency_class": k,
                "observed_num_carrier_samples": observed_row.get("observed_num_carrier_samples", ""),
                "observed_alt_alleles_from_gt": observed_row.get("observed_alt_alleles_from_gt", ""),
                "observed_het_carriers": observed_row.get("observed_het_carriers", ""),
                "observed_hom_carriers": observed_row.get("observed_hom_carriers", ""),
                "carrier_structure": carrier_structure,
                "observed_flank_size": observed_row.get("observed_flank_size", ""),
                "num_null_replicates": len(null_values),
                "pvalue_tail": pvalue_tail,
                "empirical_p_value": "",
                "null_mean_flank_sample_nodes": "",
                "null_median_flank_sample_nodes": "",
                "null_q05_flank_sample_nodes": "",
                "null_q95_flank_sample_nodes": "",
                "pvalue_status": "ok",
            }

            if observed is None:
                out["pvalue_status"] = "missing_observed_flank_size"
            elif len(null_values) == 0:
                out["pvalue_status"] = "no_null_replicates_for_frequency_class"
            else:
                out["empirical_p_value"] = empirical_pvalue(observed, null_values, pvalue_tail)
                out["null_mean_flank_sample_nodes"] = float(np.mean(null_values))
                out["null_median_flank_sample_nodes"] = percentile(null_values, 50)
                out["null_q05_flank_sample_nodes"] = percentile(null_values, 5)
                out["null_q95_flank_sample_nodes"] = percentile(null_values, 95)

            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate phase-corrected MRCA nulls by VCF allele-count/frequency class."
    )
    parser.add_argument("--tree-sequence", required=True, type=Path, help="Input .trees file for one chromosome")
    parser.add_argument("--observed-tsv", required=True, type=Path, help="Observed MRCA TSV from vcf_phase_corrected_mrca.py")
    parser.add_argument("--chrom", required=True, help="Chromosome to process, e.g. 10 or chr10")
    parser.add_argument("--out", required=True, type=Path, help="Raw null replicate output TSV")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional summary TSV by frequency class")
    parser.add_argument("--pvalues-out", type=Path, default=None, help="Optional observed-row empirical p-value TSV")
    parser.add_argument(
        "--pvalue-tail",
        choices=["lower", "upper", "two-sided"],
        default="lower",
        help="Empirical p-value tail; lower asks whether observed clades are unusually small",
    )
    parser.add_argument("--replicates-per-row", type=int, default=100, help="Null replicates per observed row")
    parser.add_argument("--frequency-col", default="info_AC", help="Observed TSV column containing k")
    parser.add_argument("--position-col", default="query_pos", help="Observed TSV column containing tree-sequence position")
    parser.add_argument(
        "--carrier-structure",
        choices=["matched_het_hom", "allele_count"],
        default="matched_het_hom",
        help="Match observed het/hom carrier counts, or sample k haplotypes only",
    )
    parser.add_argument("--frequencies", default=None, help="Optional comma-separated list of k values to process")
    parser.add_argument("--min-frequency", type=int, default=2, help="Skip observed rows with k below this value")
    parser.add_argument("--status-filter", default=None, help="Optional comma-separated observed statuses to process, e.g. ok")
    parser.add_argument("--max-rows", type=int, default=None, help="Process at most this many observed rows after filters")
    parser.add_argument("--window-bp", type=float, default=75000.0, help="Half window around variant position")
    parser.add_argument("--exclude-bp", type=float, default=200.0, help="Central exclusion half-width around variant position")
    parser.add_argument(
        "--homo-alt-policy",
        choices=["both", "one"],
        default="both",
        help="For sampled homozygous null carriers, require both haplotypes or one representative haplotype",
    )
    parser.add_argument(
        "--strict-exclude-overlapping-trees",
        action="store_true",
        help="Skip any tree interval that overlaps the central excluded region at all",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--include-sampled-haps", action="store_true", help="Include sampled haplotype node IDs")
    args = parser.parse_args()

    if args.exclude_bp < 0 or args.window_bp <= 0 or args.exclude_bp >= args.window_bp:
        raise ValueError("Require 0 <= exclude_bp < window_bp")
    if args.replicates_per_row <= 0:
        raise ValueError("--replicates-per-row must be positive")
    if args.min_frequency < 1:
        raise ValueError("--min-frequency must be at least 1")

    frequencies = parse_int_set(args.frequencies)
    status_filter = parse_status_set(args.status_filter)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(args.out.suffix + ".summary.tsv")
    if args.pvalues_out is None:
        args.pvalues_out = args.out.with_suffix(args.out.suffix + ".pvalues.tsv")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.pvalues_out.parent.mkdir(parents=True, exist_ok=True)

    observed_rows = load_observed_rows(
        path=args.observed_tsv,
        chrom=args.chrom,
        frequency_col=args.frequency_col,
        position_col=args.position_col,
        status_filter=status_filter,
        frequencies=frequencies,
        min_frequency=args.min_frequency,
        max_rows=args.max_rows,
    )
    if not observed_rows:
        print(
            f"No observed TSV rows matched requested filters for {args.chrom}; "
            f"writing header-only outputs.",
            file=sys.stderr,
        )
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=output_columns(args.include_sampled_haps),
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
        write_summary({}, args.summary_out)
        write_observed_pvalues([], {}, {}, args.pvalues_out, args.pvalue_tail, args.carrier_structure)
        print(f"Rows written: 0", file=sys.stderr)
        print(f"Output: {args.out}", file=sys.stderr)
        print(f"Summary: {args.summary_out}", file=sys.stderr)
        print(f"Observed p-values: {args.pvalues_out}", file=sys.stderr)
        return 0
    print(f"Observed rows selected: {len(observed_rows)}", file=sys.stderr)

    print(f"Loading tree sequence: {args.tree_sequence}", file=sys.stderr)
    ts = tskit.load(args.tree_sequence)
    print(
        f"Tree sequence: samples={ts.num_samples} individuals={ts.num_individuals} "
        f"trees={ts.num_trees} sequence_length={ts.sequence_length}",
        file=sys.stderr,
    )

    print("Building tree interval index", file=sys.stderr)
    tree_index = TreeIntervalIndex(ts)

    print("Building sample-node to individual mapping", file=sys.stderr)
    usable_sample_nodes, node_to_individual, individual_to_nodes, node_to_individual_label = build_individual_node_maps(ts)
    print(
        f"Usable sample haplotypes: {len(usable_sample_nodes)}; "
        f"carrier individuals: {len(individual_to_nodes)}",
        file=sys.stderr,
    )

    max_k = max(int(row["frequency_class"]) for row in observed_rows)
    max_alt_alleles = max(int(row["observed_alt_alleles_from_gt"]) for row in observed_rows)
    max_carriers = max(int(row["observed_num_carrier_samples"]) for row in observed_rows)
    usable_diploid_individual_ids = np.array(
        sorted(individual_id for individual_id, nodes in individual_to_nodes.items() if len(nodes) >= 2),
        dtype=np.int64,
    )
    if args.carrier_structure == "allele_count" and max_k > len(usable_sample_nodes):
        raise ValueError(f"Requested k={max_k}, but only {len(usable_sample_nodes)} usable sample haplotypes exist")
    if args.carrier_structure == "matched_het_hom":
        if max_alt_alleles > len(usable_sample_nodes):
            raise ValueError(
                f"Requested {max_alt_alleles} ALT haplotypes, but only "
                f"{len(usable_sample_nodes)} usable sample haplotypes exist"
            )
        if max_carriers > len(usable_diploid_individual_ids):
            raise ValueError(
                f"Requested {max_carriers} carrier individuals, but only "
                f"{len(usable_diploid_individual_ids)} usable diploid individuals exist"
            )

    rng = np.random.default_rng(args.seed)
    cols = output_columns(args.include_sampled_haps)
    summary_values: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"flank": [], "carrier_individuals": [], "homozygous_carriers": []}
    )
    pvalue_values_by_source_row: Dict[int, List[float]] = defaultdict(list)
    written = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        writer.writeheader()

        for source in observed_rows:
            k = int(source["frequency_class"])
            replicate_items: List[Dict[str, object]] = []
            for replicate in range(1, args.replicates_per_row + 1):
                if args.carrier_structure == "matched_het_hom":
                    sampled, groups, num_carriers, num_homozygous, num_candidate_haps = (
                        groups_from_matched_het_hom_draw(
                            rng=rng,
                            usable_individual_ids=usable_diploid_individual_ids,
                            individual_to_nodes=individual_to_nodes,
                            num_het=int(source["observed_het_carriers"]),
                            num_hom=int(source["observed_hom_carriers"]),
                            homo_alt_policy=args.homo_alt_policy,
                        )
                    )
                else:
                    sampled = rng.choice(usable_sample_nodes, size=k, replace=False)
                    groups, num_carriers, num_homozygous, num_candidate_haps = groups_from_sampled_haplotypes(
                        sampled_haps=sampled,
                        node_to_individual=node_to_individual,
                        individual_to_nodes=individual_to_nodes,
                        homo_alt_policy=args.homo_alt_policy,
                    )

                replicate_items.append(
                    {
                        "row": make_replicate_record(
                            source=source,
                            replicate=replicate,
                            sampled=sampled,
                            groups=groups,
                            num_carriers=num_carriers,
                            num_homozygous=num_homozygous,
                            num_candidate_haps=num_candidate_haps,
                            include_sampled_haps=args.include_sampled_haps,
                        ),
                        "groups": groups,
                    }
                )

            rows = evaluate_replicates_for_source(
                ts=ts,
                tree_index=tree_index,
                node_to_individual_label=node_to_individual_label,
                replicate_items=replicate_items,
                query_pos=float(source["query_pos"]),
                window_bp=args.window_bp,
                exclude_bp=args.exclude_bp,
                strict_exclude_overlapping_trees=args.strict_exclude_overlapping_trees,
            )
            for row in rows:
                writer.writerow(row)
                flank_size = finite_float(row.get("flank_phase_best_num_sample_nodes_under"))
                if row.get("status") == "ok" and flank_size is not None:
                    summary_k = int(row["frequency_class"])
                    summary_values[summary_k]["flank"].append(flank_size)
                    summary_values[summary_k]["carrier_individuals"].append(float(row["num_carrier_individuals"]))
                    summary_values[summary_k]["homozygous_carriers"].append(float(row["num_homozygous_carriers"]))
                    pvalue_values_by_source_row[int(row["source_row"])].append(flank_size)
                written += 1

            print(
                f"Finished source_row={source['source_row']} k={k}; total null rows={written}",
                file=sys.stderr,
            )

    write_summary(summary_values, args.summary_out)
    write_observed_pvalues(
        observed_rows,
        summary_values,
        pvalue_values_by_source_row,
        args.pvalues_out,
        args.pvalue_tail,
        args.carrier_structure,
    )
    print(f"Rows written: {written}", file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)
    print(f"Summary: {args.summary_out}", file=sys.stderr)
    print(f"Observed p-values: {args.pvalues_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
