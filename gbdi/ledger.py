from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

FORBIDDEN_EVIDENCE = re.compile(
    r"\b(replacement|replace|edit|drop|exclude|remove|repair|normalize|"
    r"recover(?:ed|y|able)?|reconstruct|interpolat(?:e|ion)|"
    r"no[-_ ]?op|overwrite|fill|operation)\b",
    re.IGNORECASE,
)
INDIRECT_LEAKAGE = re.compile(
    r"\b(should\s+be|must\s+be|correct(?:ed)?\s+value|uniquely\s+"
    r"(?:imply|determin)|can\s+be\s+(?:derived|inferred|reconstructed)|"
    r"would\s+be)\b",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def evidence_is_safe(value: str) -> bool:
    return (
        bool(value)
        and not FORBIDDEN_EVIDENCE.search(value)
        and not INDIRECT_LEAKAGE.search(value)
    )
LEDGER_PREAMBLE = """GRADED DISCOVERY LEDGER

Five independent inspections examined semantically equivalent views of this
same table. Every nominated canonical row is retained below, including
single-view nominations. Consensus is an inspection-priority signal, not proof
that a row is wrong and not authorization to change or exclude data.

Before answering:
1. establish the exact query operation and answer scope;
2. inspect every ledger row in the complete canonical table;
3. use the vote pattern and observations only to prioritize verification;
4. ignore nominations not supported by the visible table; and
5. directly compute and submit the final answer.

The ledger contains no Gold labels, prescribed operations, replacements,
recovered table, Repair Skill, or answer."""


def render_ledger(packet: dict[str, Any]) -> str:
    targets = list(packet["targets"])
    if not targets:
        return ""
    lines = [
        LEDGER_PREAMBLE,
        "",
        (
            "View reliability: "
            f'{packet["parseable_views"]}/{packet["requested_views"]} '
            "inspection outputs were parseable."
        ),
    ]
    by_support: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        by_support[int(target["identified_in_views"])].append(target)
    for support in sorted(by_support, reverse=True):
        lines.extend(["", f"Support {support}/{packet['requested_views']}:"])
        for target in sorted(by_support[support], key=lambda item: int(item["row_id"])):
            columns = ", ".join(
                json.dumps(item, ensure_ascii=False) for item in target["columns"]
            )
            lines.append(
                f'- row_id={target["row_id"]}; columns=[{columns}]; '
                f'votes={target["vote_mask"]}; '
                f'valid_views={packet["parseable_views"]}/{packet["requested_views"]}'
            )
            values = "; ".join(
                f"{column}={json.dumps(value, ensure_ascii=False)}"
                for column, value in target["visible_values"].items()
            )
            if values:
                lines.append(f"  Visible candidate cells: {values}")
            for observation in target["visible_observations"]:
                lines.append(f"  Scout observation to verify: {observation}")
    return "\n".join(lines)


def compile_ledgers(
    examples: list[dict[str, Any]], scouts: list[dict[str, Any]], layouts=("random_row5",)
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    example_by_id = {str(item["example_id"]): item for item in examples}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scouts:
        if row.get("track") == "direct" and row.get("layout") in layouts:
            grouped[(str(row["example_id"]), str(row["layout"]))].append(row)

    packets: dict[str, dict[str, dict[str, Any]]] = {}
    removed: list[dict[str, Any]] = []
    for example_id, example in example_by_id.items():
        frame = example["df"].astype(str).reset_index(drop=True)
        packets[example_id] = {}
        for layout in layouts:
            rows = sorted(
                grouped[(example_id, layout)], key=lambda item: int(item.get("pass", 0))
            )
            support: Counter[int] = Counter()
            columns: dict[int, set[str]] = defaultdict(set)
            observations: dict[int, list[str]] = defaultdict(list)
            seen_per_view: list[set[int]] = []
            valid_per_view: list[bool] = []
            for scout in rows:
                parse_ok = bool(scout.get("parse_ok"))
                valid_per_view.append(parse_ok)
                if not parse_ok:
                    seen_per_view.append(set())
                    continue
                payload = scout.get("payload") or {}
                pass_seen: set[int] = set()
                for item in payload.get("candidates", []):
                    try:
                        row_id = int(item["row_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if row_id < 0 or row_id >= len(frame):
                        continue
                    pass_seen.add(row_id)
                    for column in item.get("columns", []):
                        if str(column) in frame.columns:
                            columns[row_id].add(str(column))
                    evidence = normalize_text(item.get("evidence", ""))
                    if evidence and evidence_is_safe(evidence):
                        if evidence not in observations[row_id]:
                            observations[row_id].append(evidence)
                    elif evidence:
                        removed.append(
                            {
                                "example_id": example_id,
                                "layout": layout,
                                "row_id": row_id,
                                "evidence": evidence,
                            }
                        )
                support.update(pass_seen)
                seen_per_view.append(pass_seen)

            targets: list[dict[str, Any]] = []
            for row_id in sorted(support):
                named_columns = sorted(columns[row_id])
                vote_mask = "".join(
                    "?" if not valid else ("1" if row_id in seen else "0")
                    for valid, seen in zip(valid_per_view, seen_per_view)
                )
                targets.append(
                    {
                        "row_id": row_id,
                        "columns": named_columns,
                        "identified_in_views": int(support[row_id]),
                        "requested_views": len(rows),
                        "vote_mask": vote_mask,
                        "visible_values": {
                            column: str(frame.iloc[row_id][column])
                            for column in named_columns
                        },
                        "visible_observations": observations[row_id],
                    }
                )
            packet = {
                "targets": targets,
                "parseable_views": sum(valid_per_view),
                "requested_views": len(rows),
            }
            packets[example_id][layout] = {
                "structured": packet,
                "rendered": render_ledger(packet),
            }
    audit = {
        "removed_evidence_count": len(removed),
        "removed_evidence": removed,
        "packet_examples": len(packets),
    }
    return packets, audit
