from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from . import geometry_code

DIRECT_SCOUT_PROMPT = """SETTING

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
      "row_id": "canonical __row_id__",
      "columns": ["visible column name"],
      "evidence": "short visible-table reason"
    }
  ],
  "unresolved": ["short unresolved issue"]
}

An empty candidates list is valid. Gold answers, hidden references, artifact
labels, and external facts are unavailable.
"""


def visible_table(example: dict[str, Any], order: list[int]) -> pd.DataFrame:
    visible = example["df"].iloc[order].copy(deep=True).astype(str).reset_index(drop=True)
    visible.insert(0, "__row_id__", [str(index) for index in order])
    return visible


def scout_user_prompt(question: str, visible: pd.DataFrame) -> str:
    return (
        f"Downstream question:\n{question}\n\n"
        "Visible table:\n"
        f"{visible.to_csv(index=False, lineterminator=chr(10))}\n"
        "All cells are object dtype. `__row_id__` is runtime metadata identifying "
        "the canonical source row; it is not a source-table field.\n"
    )


def parse_direct_payload(raw: str, visible: pd.DataFrame) -> tuple[dict[str, Any], str]:
    try:
        parsed = geometry_code.parse_json_response(raw)
        return geometry_code.validate_payload(parsed, visible), ""
    except Exception as exc:
        return {"candidates": [], "unresolved": []}, f"{type(exc).__name__}: {exc}"
