from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def stable_seed(text: str, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stable_row_ids(example_id: str, n_rows: int) -> dict[int, str]:
    ids: dict[int, str] = {}
    used: set[str] = set()
    for row in range(n_rows):
        salt = 0
        while True:
            raw = f"{example_id}\0row={row}\0salt={salt}".encode("utf-8")
            candidate = "u" + hashlib.sha256(raw).hexdigest()[:10]
            if candidate not in used:
                used.add(candidate)
                ids[row] = candidate
                break
            salt += 1
    return ids
BANDS: dict[str, tuple[float, float]] = {
    "q1": (0.0, 0.2),
    "q2": (0.2, 0.4),
    "q3": (0.4, 0.6),
    "q4": (0.6, 0.8),
    "q5": (0.8, 1.0),
}


def normalized_center(start: int, block_size: int, n_rows: int) -> float:
    if n_rows <= 1:
        return 0.5
    return (start + (block_size - 1) / 2) / (n_rows - 1)


def eligible_starts(n_rows: int, block_size: int, band: str) -> list[int]:
    low, high = BANDS[band]
    max_start = n_rows - block_size
    starts = []
    for start in range(max_start + 1):
        center = normalized_center(start, block_size, n_rows)
        inside = low <= center < high if band != "q5" else low <= center <= high
        if inside:
            starts.append(start)
    interior = [start for start in starts if start not in {0, max_start}]
    if interior:
        starts = interior
    if not starts:
        raise RuntimeError(
            f"no valid placement for n_rows={n_rows} block_size={block_size} band={band}"
        )
    return starts


def make_band_orders(
    design: dict[str, Any], num_seeds: int
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    targets = [int(value) for value in design["target_original_rows_offline_only"]]
    target_set = set(targets)
    n_rows = int(design["n_rows"])
    orders: dict[str, list[int]] = {}
    placements: dict[str, dict[str, Any]] = {}
    for seed_index in range(num_seeds):
        non_targets = [row for row in range(n_rows) if row not in target_set]
        random.Random(
            stable_seed(design["example_id"], f"H127-background-s{seed_index}")
        ).shuffle(non_targets)
        for band in BANDS:
            starts = eligible_starts(n_rows, len(targets), band)
            start = random.Random(
                stable_seed(design["example_id"], f"H127-placement-s{seed_index}-{band}")
            ).choice(starts)
            arm = f"s{seed_index}_{band}"
            order = non_targets[:start] + targets + non_targets[start:]
            center = normalized_center(start, len(targets), n_rows)
            orders[arm] = order
            placements[arm] = {
                "seed": f"s{seed_index}",
                "band": band,
                "band_bounds": list(BANDS[band]),
                "start": start,
                "block_size": len(targets),
                "realized_target_center": center,
                "touches_serialized_boundary": start == 0 or start == n_rows - len(targets),
                "eligible_start_count": len(starts),
            }
    return orders, placements
CENTERS = {"c25": 0.25, "c50": 0.50, "c75": 0.75}


SPACINGS = ("compact", "medium", "wide")


NUM_SEEDS = 3


MAX_RADIUS_FRACTION = 0.20


def snapped_compact_start(n_rows: int, target_count: int, center_fraction: float) -> int:
    desired_center = center_fraction * (n_rows - 1)
    start = round(desired_center - (target_count - 1) / 2)
    return max(1, min(n_rows - target_count - 1, start))


def select_evenly(values: list[int], count: int) -> list[int]:
    if count == 1:
        return [values[-1]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    for index in range(1, count):
        if indices[index] <= indices[index - 1]:
            indices[index] = indices[index - 1] + 1
    for index in range(count - 1, -1, -1):
        maximum = len(values) - (count - index)
        if indices[index] > maximum:
            indices[index] = maximum
        if index and indices[index - 1] >= indices[index]:
            indices[index - 1] = indices[index] - 1
    return [values[index] for index in indices]


def symmetric_target_positions(
    n_rows: int, target_count: int, center_fraction: float, spacing: str
) -> list[int]:
    start = snapped_compact_start(n_rows, target_count, center_fraction)
    center2 = 2 * start + target_count - 1
    boundary_max2 = min(center2 - 2, 2 * (n_rows - 2) - center2)
    radius_cap2 = int(2 * MAX_RADIUS_FRACTION * (n_rows - 1))
    max_offset2 = min(boundary_max2, radius_cap2)
    parity = center2 % 2
    allowed = list(range(1 if parity else 2, max_offset2 + 1, 2))
    pair_count = target_count // 2
    if len(allowed) < pair_count:
        raise RuntimeError(
            f"insufficient symmetric slots n={n_rows} k={target_count} center={center_fraction}"
        )
    compact_end = pair_count - 1
    if spacing == "compact":
        end = compact_end
    elif spacing == "medium":
        end = round((compact_end + len(allowed) - 1) / 2)
    elif spacing == "wide":
        end = len(allowed) - 1
    else:
        raise ValueError(spacing)
    offsets = select_evenly(allowed[: end + 1], pair_count)
    positions = [(center2 - offset) // 2 for offset in reversed(offsets)]
    if target_count % 2:
        positions.append(center2 // 2)
    positions.extend((center2 + offset) // 2 for offset in offsets)
    if len(positions) != target_count or len(set(positions)) != target_count:
        raise AssertionError("invalid target position set")
    if min(positions) <= 0 or max(positions) >= n_rows - 1:
        raise AssertionError("target touches serialized boundary")
    if sum(positions) / target_count != center2 / 2:
        raise AssertionError("target centroid drift")
    return positions


def spacing_metrics(positions: list[int], n_rows: int) -> dict[str, float]:
    ordered = sorted(positions)
    adjacent = [right - left for left, right in zip(ordered, ordered[1:])]
    nearest = []
    for index, position in enumerate(ordered):
        distances = []
        if index:
            distances.append(position - ordered[index - 1])
        if index + 1 < len(ordered):
            distances.append(ordered[index + 1] - position)
        nearest.append(min(distances))
    pairwise = [
        right - left
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    ]
    scale = max(1, n_rows - 1)
    return {
        "realized_target_center": (sum(ordered) / len(ordered)) / scale,
        "mean_nearest_neighbor_gap": (sum(nearest) / len(nearest)) / scale,
        "mean_adjacent_gap": (sum(adjacent) / len(adjacent)) / scale,
        "mean_pairwise_gap": (sum(pairwise) / len(pairwise)) / scale,
        "target_span": (ordered[-1] - ordered[0]) / scale,
    }


def make_orders(
    source: dict[str, Any]
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    targets = sorted(set(map(int, source["target_original_rows_offline_only"])))
    target_set = set(targets)
    n_rows = int(source["n_rows"])
    orders: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for seed_index in range(NUM_SEEDS):
        seed = f"s{seed_index}"
        non_targets = [row for row in range(n_rows) if row not in target_set]
        random.Random(stable_seed(source["example_id"], f"H133-background-{seed}")).shuffle(
            non_targets
        )
        target_order = list(targets)
        random.Random(stable_seed(source["example_id"], f"H133-target-order-{seed}")).shuffle(
            target_order
        )
        for center, fraction in CENTERS.items():
            for spacing in SPACINGS:
                positions = symmetric_target_positions(
                    n_rows, len(targets), fraction, spacing
                )
                target_by_position = dict(zip(positions, target_order))
                non_target_iter = iter(non_targets)
                order = [
                    target_by_position[position]
                    if position in target_by_position
                    else next(non_target_iter)
                    for position in range(n_rows)
                ]
                arm = f"{seed}_{center}_{spacing}"
                orders[arm] = order
                metrics = spacing_metrics(positions, n_rows)
                metadata[arm] = {
                    "seed": seed,
                    "center": center,
                    "requested_center": fraction,
                    "spacing": spacing,
                    "target_positions": positions,
                    "target_order_original_rows": target_order,
                    "target_position_by_original_row": {
                        str(row): position
                        for row, position in zip(target_order, positions)
                    },
                    **metrics,
                }
    return orders, metadata
def build_visible(
    example: dict[str, Any], design: dict[str, Any], arm: str
) -> tuple[pd.DataFrame, dict[int, str], list[int], list[str]]:
    source = example["df"].astype(str).reset_index(drop=True)
    row_ids = {int(row): str(value) for row, value in design["row_id_by_original_row"].items()}
    row_order = [int(row) for row in design["row_orders"][arm]]
    columns = [str(column) for column in source.columns]
    visible = source.iloc[row_order][columns].reset_index(drop=True)
    visible.insert(0, "__row_id__", [row_ids[row] for row in row_order])
    return visible, row_ids, row_order, columns


def question_table_prompt(question: str, visible: pd.DataFrame) -> str:
    return (
        f"Downstream question:\n{question}\n\n"
        "Visible table (also available as pandas dataframe `df` in repair mode):\n"
        f"{visible.astype(str).to_csv(index=False, lineterminator=chr(10))}\n"
        "All cells are object dtype. `__row_id__` is runtime metadata, not a source-table field.\n"
    )


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload.get("candidates"), list):
        raise ValueError("candidates must be a list")
    return payload
