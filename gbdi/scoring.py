from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def extract_first_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group() if match else None


def official_match_answer(predicted: Any, ground_truth: Any) -> bool:
    """The RADAR answer scorer used for the reported experiments."""

    predicted_str = "" if predicted is None else str(predicted)

    def normalize(val: Any) -> Any:
        if isinstance(val, str):
            return val.strip().lower()
        return val

    def parse_stringified_ground_truth(gt: Any) -> Any:
        if not isinstance(gt, str):
            return gt
        stripped = gt.strip()
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (str, int, float, list)):
                return parsed
        except (ValueError, SyntaxError):
            pass
        numeric = stripped.replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", numeric):
            return float(numeric) if "." in numeric else int(numeric)
        return gt

    def is_float_match(predicted_text: str, gt: Any) -> bool:
        if not isinstance(gt, float):
            return False
        num_str = extract_first_number(predicted_text)
        if num_str is None:
            return False
        predicted_val = float(num_str)
        gt_str = str(gt)
        if "." in gt_str:
            decimal_places = len(gt_str.split(".")[-1])
            decimal_places_num_str = len(num_str.split(".")[-1])
            if decimal_places_num_str <= 3:
                decimal_places = max(decimal_places, decimal_places_num_str)
        else:
            decimal_places = 0
        float_tol = 10**-decimal_places
        return abs(predicted_val - float(gt)) <= float_tol

    def is_int_match(predicted_text: str, gt: Any) -> bool:
        if not isinstance(gt, int):
            return False
        num_str = extract_first_number(predicted_text)
        if num_str is None:
            return False
        try:
            num = float(num_str)
            return num.is_integer() and int(num) == gt
        except ValueError:
            return False

    def is_string_match(predicted_text: str, gt: Any) -> bool:
        return isinstance(gt, str) and normalize(predicted_text) == normalize(gt)

    def is_list_of_strings_match(predicted_text: str, gt_list: list[Any]) -> bool:
        predicted_items = [normalize(p) for p in predicted_text.split(",")]
        gt_items = [normalize(g) for g in gt_list]
        return set(predicted_items) == set(gt_items)

    def is_list_of_numbers_match(predicted_text: str, gt_list: list[Any]) -> bool:
        try:
            predicted_items = [float(p.strip()) for p in predicted_text.split(",")]
        except ValueError:
            return False
        if not all(isinstance(g, (int, float)) for g in gt_list):
            return False
        if len(gt_list) != len(predicted_items):
            return False
        for pred_val, gt_val in zip(sorted(predicted_items), sorted(gt_list)):
            if not is_float_match(str(pred_val), gt_val):
                return False
        return True

    def is_list_match(predicted_text: str, gt_list: list[Any]) -> bool:
        if all(isinstance(g, str) for g in gt_list):
            return is_list_of_strings_match(predicted_text, gt_list)
        if all(isinstance(g, (int, float)) for g in gt_list):
            return is_list_of_numbers_match(predicted_text, gt_list)
        return False

    ground_truth = parse_stringified_ground_truth(ground_truth)
    if isinstance(ground_truth, list) and all(isinstance(sub, list) for sub in ground_truth):
        return any(is_list_match(predicted_str, sublist) for sublist in ground_truth)
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if is_string_match(predicted_str, gt) or is_int_match(predicted_str, gt) or is_float_match(predicted_str, gt):
                return True
        return False
    return (
        is_string_match(predicted_str, ground_truth)
        or is_int_match(predicted_str, ground_truth)
        or is_float_match(predicted_str, ground_truth)
    )
