from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

ROW_ID_SALT = "GBEA-row-id-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_df(example: dict[str, Any]) -> pd.DataFrame:
    return example["df"].astype(str).reset_index(drop=True)


def canonical_table_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def stable_row_ids(df: pd.DataFrame) -> tuple[dict[int, str], str, int]:
    table_hash = sha256_text(canonical_table_csv(df))
    digests = [
        hashlib.sha256(
            f"{ROW_ID_SALT}\x1f{table_hash}\x1f{index}".encode("utf-8")
        ).hexdigest()
        for index in range(len(df))
    ]
    prefix_length = 12
    while len({value[:prefix_length] for value in digests}) != len(digests):
        prefix_length += 1
        if prefix_length > 64:
            raise RuntimeError("Unable to construct collision-free row identifiers")
    return (
        {index: f"rid_{digest[:prefix_length]}" for index, digest in enumerate(digests)},
        table_hash,
        prefix_length,
    )


def render_discovery_table(
    df: pd.DataFrame, row_ids: dict[int, str], order: list[int]
) -> str:
    view = df.iloc[order].copy()
    view.insert(0, "__row_id__", [row_ids[index] for index in order])
    return view.to_csv(index=False, lineterminator="\n")


def build_discovery_user(question: str, table: str) -> str:
    return (
        "Downstream question:\n"
        f"{question}\n\n"
        "Complete visible table view:\n"
        f"{table}\n"
        "All source-table cells are displayed as strings. `__row_id__` is stable, "
        "nonordinal runtime metadata used only to identify rows across views."
    )
def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def render_ledger(
    df: Any,
    row_id_to_index: dict[str, int],
    outputs: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    by_row: dict[str, dict[str, Any]] = {}
    for output in outputs:
        seen_in_view: dict[str, dict[str, set[str]]] = {}
        for candidate in output.get("parsed", {}).get("candidates", []):
            row_id = str(candidate["row_id"])
            item = seen_in_view.setdefault(
                row_id,
                {"columns": set(), "relevance": set(), "evidence": set()},
            )
            item["columns"].update(map(str, candidate["columns"]))
            item["relevance"].add(str(candidate["query_relevance"]))
            item["evidence"].add(str(candidate["visible_evidence"]))
        for row_id, item in seen_in_view.items():
            aggregate = by_row.setdefault(
                row_id,
                {
                    "columns": set(),
                    "relevance": set(),
                    "evidence": set(),
                    "supporting_views": set(),
                },
            )
            aggregate["columns"].update(item["columns"])
            aggregate["relevance"].update(item["relevance"])
            aggregate["evidence"].update(item["evidence"])
            aggregate["supporting_views"].add(int(output["pass_index"]))

    column_order = {str(column): index for index, column in enumerate(df.columns)}
    candidates: list[dict[str, Any]] = []
    for row_id, item in by_row.items():
        canonical_index = row_id_to_index[row_id]
        columns = sorted(item["columns"], key=lambda value: (column_order[value], value))
        candidates.append(
            {
                "discovery_row_id_offline_only": row_id,
                "canonical_df_index": canonical_index,
                "columns": columns,
                "displayed_cells": {
                    column: str(df.iloc[canonical_index][column]) for column in columns
                },
                "query_relevance": sorted(item["relevance"]),
                "evidence": sorted(item["evidence"]),
                "support": len(item["supporting_views"]),
                "supporting_views_offline_only": sorted(item["supporting_views"]),
            }
        )
    candidates.sort(
        key=lambda item: (-int(item["support"]), str(item["discovery_row_id_offline_only"]))
    )
    parseable = sum(bool(output.get("parsed", {}).get("parseable")) for output in outputs)
    lines = [
        "CANDIDATE-EVIDENCE LEDGER",
        "",
        "Independent inspections nominated the rows below for verification. The notes",
        "are fallible and non-exhaustive. Cross-view support indicates recurrence across",
        "inspections; it is not proof of corruption and does not authorize a data",
        "change. The complete canonical table and original question remain authoritative.",
        "",
        f"Parseable discovery outputs: {parseable}/{PASSES}.",
    ]
    if not candidates:
        lines.extend(["", "No row was nominated by a parseable discovery output."])
        return "\n".join(lines), candidates
    current_support: int | None = None
    for candidate in candidates:
        support = int(candidate["support"])
        if support != current_support:
            lines.extend(["", f"Nominated by {support}/{PASSES} views:"])
            current_support = support
        lines.append(
            f"- df_index={candidate['canonical_df_index']}; "
            f"columns={json.dumps(candidate['columns'], ensure_ascii=False)}"
        )
        lines.append(
            "  displayed cells: "
            + json.dumps(candidate["displayed_cells"], ensure_ascii=False, sort_keys=True)
        )
        lines.append("  query-relevance claims to verify:")
        for relevance in candidate["query_relevance"]:
            lines.append(f"  - {relevance}")
        lines.append("  visible observations to verify:")
        for evidence in candidate["evidence"]:
            lines.append(f"  - {evidence}")
    return "\n".join(lines), candidates
def parse_discovery_v2(
    raw: str, valid_row_ids: set[str], visible_columns: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parseable": False,
        "parser_error": "",
        "candidates": [],
        "top_level_extra_fields": [],
        "candidate_extra_fields": [],
        "invalid_candidate_count": 0,
        "parser_version": "empty_observed_value_valid",
    }
    try:
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, dict):
            raise ValueError("top-level value is not an object")
        if not isinstance(parsed.get("candidates"), list):
            raise ValueError("candidates is not a list")
    except Exception as exc:
        result["parser_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["parseable"] = True
    result["top_level_extra_fields"] = sorted(set(parsed) - {"candidates"})
    allowed = {
        "row_id",
        "column",
        "observed_value",
        "query_relevance",
        "visible_evidence",
    }
    extras: set[str] = set()
    for item in parsed["candidates"]:
        if not isinstance(item, dict):
            result["invalid_candidate_count"] += 1
            continue
        extras.update(set(item) - allowed)
        row_id = item.get("row_id")
        column = item.get("column")
        observed_value = item.get("observed_value")
        relevance = clean_text(item.get("query_relevance", ""))
        evidence = clean_text(item.get("visible_evidence", ""))
        if (
            not isinstance(row_id, str)
            or row_id not in valid_row_ids
            or not isinstance(column, str)
            or column not in visible_columns
            or "observed_value" not in item
            or not isinstance(observed_value, str)
            or not relevance
            or not evidence
        ):
            result["invalid_candidate_count"] += 1
            continue
        result["candidates"].append(
            {
                "row_id": row_id,
                "columns": [column],
                "observed_value": clean_text(observed_value),
                "query_relevance": relevance,
                "visible_evidence": evidence,
            }
        )
    result["candidate_extra_fields"] = sorted(extras)
    return result

PASSES = 5
