from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from .geometry import build_visible, parse_json_response
from .parsing import _parse_command

MAX_OBSERVATION_CHARS = 16000


SYSTEM_PROMPT = """SETTING

You are a diagnostic table inspector. Inspect the visible table only for the
downstream question. Do not repair the table and do not answer the question.

Identify only rows for which the visible table provides concrete evidence that
a query-relevant value or relation is invalid or unusable. Do not list a row
merely because it is unusual, irrelevant, extreme, or difficult to interpret.
When evidence suggests a problem but does not identify one row, record it as
unresolved instead of listing every possible row.

You may inspect the complete pandas dataframe `df` with read-only Python. It
contains the exact visible CSV and all cells are object dtype. Use Python when
it helps check complete-column patterns, cross-column relations, masks, or
candidate coverage. `pd`, `np`, `json`, `math`, `re`, and `statistics` are
already available. Variables persist across inspection turns, while `df` is
restored after each command.

On every turn return exactly one YAML command and no prose.

To inspect the table:
command: python
kwargs:
  code: |
    print(df.head())

To finish:
command: submit_localization
kwargs:
  payload: |
    {"candidates":[{"row_id":"opaque __row_id__","columns":["visible column name"],"evidence":"short visible-table reason"}],"unresolved":[]}

An empty candidates list is valid. Candidate row ids and columns must exist in
the visible table. Gold answers, hidden references, artifact labels, and
external facts are unavailable.
"""


PROHIBITED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}


ALLOWED_IMPORTS = {"json", "math", "numpy", "pandas", "re", "statistics"}


PROHIBITED_IO_METHODS = {
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
}


def question_table_prompt(question: str, visible: pd.DataFrame) -> str:
    return (
        f"Downstream question:\n{question}\n\n"
        "Visible table (also available as pandas dataframe `df`):\n"
        f"{visible.astype(str).to_csv(index=False, lineterminator=chr(10))}\n"
        "All cells are object dtype. `__row_id__` is runtime metadata, not a "
        "source-table field.\n"
    )


def validate_readonly_code(code: str) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= ALLOWED_IMPORTS:
                raise ValueError(f"disallowed Python imports: {sorted(roots - ALLOWED_IMPORTS)}")
        if isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise ValueError(f"disallowed Python import: {root}")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError(f"disallowed Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in PROHIBITED_NAMES:
            raise ValueError(f"disallowed Python name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("dunder attribute access is disallowed")
            if node.attr in PROHIBITED_IO_METHODS:
                raise ValueError(f"external I/O method is disallowed: {node.attr}")
    return tree


def run_readonly_python(
    code: str,
    visible: pd.DataFrame,
    question: str,
    namespace: dict[str, Any],
) -> tuple[str, bool]:
    try:
        tree = validate_readonly_code(code)
        namespace["df"] = visible.copy(deep=True)
        namespace["question"] = question
        stdout = io.StringIO()
        result: Any = None
        with redirect_stdout(stdout):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(prefix, "<h169-readonly>", "exec"), namespace)
                result = eval(
                    compile(ast.Expression(tree.body[-1].value), "<h169-readonly>", "eval"),
                    namespace,
                )
            else:
                exec(compile(tree, "<h169-readonly>", "exec"), namespace)
        printed = stdout.getvalue().strip()
        if result is not None:
            printed = f"{printed}\n{result}".strip()
        if len(printed) > MAX_OBSERVATION_CHARS:
            printed = printed[:MAX_OBSERVATION_CHARS] + "\n...[observation truncated]"
        return printed, True
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", False


def validate_payload(
    payload: dict[str, Any], visible: pd.DataFrame
) -> dict[str, Any]:
    candidates = payload.get("candidates")
    unresolved = payload.get("unresolved")
    if not isinstance(candidates, list) or not isinstance(unresolved, list):
        raise ValueError("payload requires list fields candidates and unresolved")
    valid_rows = set(visible["__row_id__"].astype(str))
    valid_columns = {str(column) for column in visible.columns if column != "__row_id__"}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        row_id = str(item.get("row_id", "")).strip()
        columns = item.get("columns")
        evidence = str(item.get("evidence", "")).strip()
        if row_id not in valid_rows:
            raise ValueError(f"candidate {index} has unknown row_id {row_id!r}")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"candidate {index} requires non-empty columns")
        cooked_columns = [str(column) for column in columns]
        unknown = sorted(set(cooked_columns) - valid_columns)
        if unknown:
            raise ValueError(f"candidate {index} has unknown columns {unknown}")
        if not evidence:
            raise ValueError(f"candidate {index} requires evidence")
        key = (row_id, tuple(cooked_columns))
        if key not in seen:
            normalized.append(
                {"row_id": row_id, "columns": cooked_columns, "evidence": evidence}
            )
            seen.add(key)
    return {
        "candidates": normalized,
        "unresolved": [str(item) for item in unresolved if str(item).strip()],
    }
