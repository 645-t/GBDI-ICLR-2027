from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def normalize_column(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()
def mark_table(
    item: dict[str, Any], locations: list[dict[str, Any]]
) -> tuple[pd.DataFrame, int]:
    table = pd.DataFrame(item["rows"], columns=item["headers"], dtype=object)
    marked: set[tuple[int, str]] = set()
    for location in locations:
        row = int(location["row"])
        if row < 0 or row >= len(table):
            raise RuntimeError(f"Invalid marked row in {item['table_id']}: {row}")
        for column in location["columns"]:
            column = str(column)
            if column not in table.columns:
                raise RuntimeError(f"Invalid marked column in {item['table_id']}: {column}")
            key = (row, column)
            if key in marked:
                continue
            value = "" if table.at[row, column] is None else str(table.at[row, column])
            table.at[row, column] = f"<<ERROR:{value}>>"
            marked.add(key)
    return table, len(marked)
MARKED_NOTICE = (
    "The values marked <<ERROR:...>> have been verified as erroneous and relevant "
    "to the question; the text inside each marker is the original erroneous "
    "value, not a correction. Treat these markings as facts when answering: "
    "correct a marked value when the table supports a reliable replacement, "
    "or exclude its row if it cannot be repaired."
)


INTERVENED_NOTICE = (
    "The table below has already been repaired for this question. All reviewed "
    "errors relevant to the answer have been corrected, and rows that could "
    "not be repaired have been removed. Treat the supplied table as final and "
    "answer the question without making further changes."
)


def with_notice(messages: list[dict[str, str]], notice: str) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[1]["content"] = notice + "\n" + result[1]["content"]
    return result


def repaired_table(
    item: dict[str, Any], entries: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, int]]:
    table = pd.DataFrame(item["rows"], columns=item["headers"], dtype=object)
    column_lookup = {
        normalize_column(header): header for header in item["headers"]
    }
    if len(column_lookup) != len(item["headers"]):
        raise AssertionError(f"{item['table_id']}: ambiguous normalized headers")
    drops: set[int] = set()
    edits: dict[tuple[int, str], str] = {}
    for entry in entries:
        row = int(entry["row"])
        if row < 0 or row >= len(table):
            raise AssertionError(f"{item['table_id']}: invalid reviewed row {row}")
        if entry["operation_family"] == "DROP":
            drops.add(row)
        elif entry["operation_family"] == "EDIT":
            column = column_lookup.get(normalize_column(entry["edit_column"]))
            if column is None:
                raise AssertionError(f"{item['table_id']}: invalid edited column {entry['edit_column']}")
            value = str(entry["replacement_value"])
            if not value.strip():
                raise AssertionError(f"{item['table_id']}: empty reviewed replacement")
            key = (row, column)
            if key in edits and edits[key] != value:
                raise AssertionError(f"{item['table_id']}: conflicting replacements at {key}")
            edits[key] = value
        else:
            raise AssertionError(f"{item['table_id']}: unknown reviewed operation")
    if any(row in drops for row, _ in edits):
        raise AssertionError(f"{item['table_id']}: EDIT/DROP overlap")
    for (row, column), value in edits.items():
        table.at[row, column] = value
        if str(table.at[row, column]) != value:
            raise AssertionError(f"{item['table_id']}: failed to apply EDIT")
    repaired = table.drop(index=sorted(drops)).reset_index(drop=True)
    if len(repaired) != len(item["rows"]) - len(drops):
        raise AssertionError(f"{item['table_id']}: failed to apply DROP")
    return repaired, {"edit_cells": len(edits), "drop_rows": len(drops)}
