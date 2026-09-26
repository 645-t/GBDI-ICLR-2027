from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

import csv
SYSTEM_PROMPT = """SETTING

You are a diagnostic table inspector. Inspect the visible table only for the
downstream question. Do not repair the table and do not answer the question.

Identify only rows for which the visible table provides concrete evidence that
a query-relevant value or relation is invalid or unusable. Do not list a row
merely because it is unusual, irrelevant, extreme, or difficult to interpret.
When evidence suggests a problem but does not identify one row, record it as
unresolved instead of listing every possible row.

Return exactly one JSON object and no markdown:
{
  "candidates": [
    {
      "row_id": "opaque __row_id__",
      "columns": ["visible column name"],
      "evidence": "short visible-table reason"
    }
  ],
  "unresolved": ["short unresolved issue"]
}

An empty candidates list is valid. Hidden clean values, target coordinates,
and reference answers are unavailable.
"""


def visible_csv(design: dict[str, Any], arm: str) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["__row_id__", *design["columns"]])
    row_ids = {int(key): value for key, value in design["row_id_by_original_row"].items()}
    for original_row in design["row_orders"][arm]:
        writer.writerow([row_ids[int(original_row)], *design["dirty_rows_offline_only"][int(original_row)]])
    return output.getvalue()


def user_prompt(design: dict[str, Any], arm: str) -> str:
    return (
        f"Downstream question:\n{design['question']}\n\n"
        f"Visible table:\n{visible_csv(design, arm)}"
        "All fields are shown as serialized table values. `__row_id__` is runtime metadata.\n"
    )


def parse_response(raw: str, valid_rows: set[str], valid_columns: set[str]) -> tuple[dict[str, Any], list[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    payload = json.loads(text[start : end + 1])
    candidates = payload.get("candidates")
    unresolved = payload.get("unresolved", [])
    if not isinstance(candidates, list) or not isinstance(unresolved, list):
        raise ValueError("invalid candidates or unresolved")
    normalized = []
    warnings = []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            warnings.append(f"candidate_{index}_not_object")
            continue
        row_id = str(item.get("row_id") or "")
        columns = item.get("columns")
        if isinstance(columns, str):
            columns = [columns]
        if row_id not in valid_rows or not isinstance(columns, list):
            warnings.append(f"candidate_{index}_invalid_row_or_columns")
            continue
        columns = [str(column) for column in columns if str(column) in valid_columns]
        if not columns:
            warnings.append(f"candidate_{index}_no_valid_columns")
            continue
        normalized.append(
            {"row_id": row_id, "columns": columns, "evidence": str(item.get("evidence") or "")}
        )
    return {"candidates": normalized, "unresolved": [str(value) for value in unresolved]}, warnings
