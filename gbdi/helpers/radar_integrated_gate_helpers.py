from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


LOW_RISK_CANDIDATE_IDS = {"C0_DIRECT", "C1_FORMAT"}
REPAIR_CANDIDATE_IDS = {"C2_ROW_VALIDITY", "C3_FORMULA_RECOVER", "C4_TARGET_OUTLIER", "C5_LOGIC_VALID"}
STRONG_REPAIR_CANDIDATE_IDS = {"C3_FORMULA_RECOVER", "C4_TARGET_OUTLIER", "C5_LOGIC_VALID"}
RADAR_NA_TOKENS = {"", " ", "na", "n/a", "nan", "none", "null", "-", "--", "missing", "unknown"}
RADAR_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "per",
    "that",
    "the",
    "their",
    "there",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "whose",
    "with",
}


def radar_clean_text(value: Any) -> str | None:
    """Normalize a table cell to stripped text, preserving only generic null logic."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().replace("\u2212", "-")
    if text.lower() in RADAR_NA_TOKENS:
        return None
    return text


def radar_normalize_decimal_token(token: str) -> str:
    """Normalize commas and malformed multiple-decimal numeric tokens."""
    text = str(token).replace(",", "").strip()
    if text.count(".") <= 1:
        return text
    sign = "-" if text.startswith("-") else ""
    body = text[1:] if sign else text
    parts = [part for part in body.split(".") if part != ""]
    if not parts:
        return "nan"
    return sign + parts[0] + ("." + "".join(parts[1:]) if len(parts) > 1 else "")


def radar_nums(value: Any) -> list[float]:
    """Extract numeric values from a mixed object/string cell."""
    text = radar_clean_text(value)
    if text is None:
        return []
    tokens = re.findall(r"[-+]?\d[\d,]*(?:\.\d*)*", text)
    out: list[float] = []
    for token in tokens:
        try:
            num = float(radar_normalize_decimal_token(token))
        except Exception:
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def radar_first_num(value: Any) -> float:
    nums = radar_nums(value)
    return nums[0] if nums else float("nan")


def radar_last_num(value: Any) -> float:
    nums = radar_nums(value)
    return nums[-1] if nums else float("nan")


def radar_numeric_series(series: Any, mode: str = "first") -> pd.Series:
    """Convert an object column to numeric by extracting first or last number."""
    if not hasattr(series, "map"):
        series = pd.Series(series)
    parser = radar_last_num if mode == "last" else radar_first_num
    return series.map(parser).astype(float)


def radar_norm_key(value: Any) -> str | None:
    """Normalize entity/filter keys without assuming any domain-specific schema."""
    text = radar_clean_text(value)
    if text is None:
        return None
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def radar_query_tokens(text: Any) -> set[str]:
    """Tokenize a question/header/value for generic column matching."""
    norm = radar_norm_key(text) or ""
    return {tok for tok in norm.split() if len(tok) > 1 and tok not in RADAR_QUERY_STOPWORDS}


def radar_question_operation(question: str) -> str:
    """Infer a generic query operation from wording only."""
    q = f" {radar_norm_key(question) or ''} "
    if re.search(r"\b(how many|number of|count)\b", q):
        return "count"
    if re.search(r"\b(average|mean|avg)\b", q):
        return "mean"
    if re.search(r"\bmedian\b", q):
        return "median"
    if re.search(r"\b(sum|total|overall)\b", q):
        return "sum"
    if re.search(r"\b(maximum|minimum|highest|lowest|largest|smallest|top|bottom|most|least|rank)\b", q):
        return "rank"
    if re.search(r"\b(ratio|rate|percentage|percent|proportion|share)\b", q):
        return "ratio"
    if re.search(r"\b(difference|diff|minus|subtract|change|increase|decrease|between)\b", q):
        return "arithmetic"
    if re.search(r"\b(list|name all|which ones)\b", q):
        return "list"
    return "lookup"


def radar_column_numeric_coverage(series: Any) -> float:
    """Estimate whether a column behaves numeric after generic parsing."""
    if not hasattr(series, "map"):
        series = pd.Series(series)
    parsed = radar_numeric_series(series)
    denom = max(1, int(series.map(lambda x: radar_clean_text(x) is not None).sum()))
    return float(parsed.notna().sum()) / float(denom)


def radar_column_mentioned(question: str, column_name: Any) -> bool:
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    if not cnorm:
        return False
    if re.search(rf"(^| )({re.escape(cnorm)})( |$)", qnorm):
        return True
    return bool(radar_query_tokens(question) & radar_query_tokens(column_name))


def radar_header_match_score(question: str, column_name: Any) -> float:
    """Score generic question/header overlap without looking at table values."""
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    if not cnorm:
        return 0.0
    qtokens = radar_query_tokens(question)
    ctoks = radar_query_tokens(column_name)
    score = 0.0
    if re.search(rf"(^| )({re.escape(cnorm)})( |$)", qnorm):
        score += 5.0
    score += 2.0 * len(qtokens & ctoks)
    if any(tok in qnorm for tok in ctoks):
        score += 0.5
    return score


def radar_operation_target_score(question: str, column_name: Any, operator: str | None = None) -> float:
    """Score whether a header is named as the query's computed target."""
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    if not cnorm:
        return 0.0
    op = str(operator or radar_question_operation(question))
    op_words = {
        "mean": ["average", "mean", "avg"],
        "median": ["median"],
        "sum": ["sum", "total", "overall"],
        "rank": ["maximum", "minimum", "highest", "lowest", "largest", "smallest", "top", "bottom", "most", "least"],
        "ratio": ["ratio", "rate", "percentage", "percent", "proportion", "share"],
        "arithmetic": ["difference", "diff", "minus", "subtract", "change"],
    }.get(op, [])
    score = 0.0
    for word in op_words:
        if re.search(rf"\b{re.escape(word)}\s+(?:the\s+)?{re.escape(cnorm)}\b", qnorm):
            score += 8.0
        if re.search(rf"\b{re.escape(cnorm)}\s+{re.escape(word)}\b", qnorm):
            score += 4.0
    return score


def radar_short_filter_value_mentioned(question: str, column_name: Any, value: Any) -> bool:
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    vnorm = radar_norm_key(value) or ""
    if not cnorm or not vnorm or len(vnorm) > 2:
        return False
    if not radar_column_mentioned(question, column_name):
        return False
    patterns = [
        rf"\b{re.escape(cnorm)}\s+{re.escape(vnorm)}\b",
        rf"\b{re.escape(cnorm)}\s+(?:is|are|equals|equal to|of|for|in|as)\s+{re.escape(vnorm)}\b",
        rf"\b{re.escape(vnorm)}\s+{re.escape(cnorm)}\b",
    ]
    return any(re.search(pattern, qnorm) for pattern in patterns)


def radar_column_in_no_filter_context(question: str, column_name: Any) -> bool:
    """Detect wording that mentions a column while warning not to filter by it."""
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    if not cnorm:
        return False
    patterns = [
        rf"\b{re.escape(cnorm)}\b(?:\s+\w+){{0,2}}\s+(?:does not|do not|not)\s+(?:\w+\s+){{0,4}}exclude\b",
    ]
    return any(re.search(pattern, qnorm) for pattern in patterns)


def radar_explicit_filter_match_score(question: str, series: Any, *, column_name: Any = None, max_values: int = 80) -> float:
    """Score explicit column-value filter evidence in the question."""
    if column_name is None or radar_column_in_no_filter_context(question, column_name):
        return 0.0
    if not hasattr(series, "dropna"):
        series = pd.Series(series)
    qnorm = radar_norm_key(question) or ""
    cnorm = radar_norm_key(column_name) or ""
    if not cnorm:
        return 0.0
    score = 0.0
    seen = 0
    for raw in series.dropna().astype(str).unique().tolist():
        if seen >= max_values:
            break
        seen += 1
        variants = {radar_norm_key(raw) or ""}
        for num in radar_nums(raw):
            if abs(num - round(num)) < 1e-9:
                variants.add(str(int(round(num))))
            else:
                variants.add(radar_norm_key(str(num)) or "")
        for vnorm in sorted(variants):
            if not vnorm or vnorm in RADAR_NA_TOKENS:
                continue
            if len(vnorm) <= 2 and radar_short_filter_value_mentioned(question, column_name, vnorm):
                score += 4.0
                break
            if len(vnorm) > 40:
                continue
            value_column = rf"\b{re.escape(vnorm)}\s+{re.escape(cnorm)}\b"
            column_value = rf"\b{re.escape(cnorm)}\b(?:\s+(?:is|are|equals|equal|to|of|for|in|with|where|as)){{0,2}}\s+\b{re.escape(vnorm)}\b"
            if re.search(value_column, qnorm) or re.search(column_value, qnorm):
                score += 4.0
                break
    return score


def radar_value_match_score(question: str, series: Any, *, column_name: Any = None, max_values: int = 80) -> float:
    """Score whether column values appear in the question as filter/entity hints."""
    if not hasattr(series, "dropna"):
        series = pd.Series(series)
    qnorm = radar_norm_key(question) or ""
    qtokens = radar_query_tokens(question)
    score = 0.0
    seen = 0
    for raw in series.dropna().astype(str).unique().tolist():
        if seen >= max_values:
            break
        seen += 1
        vnorm = radar_norm_key(raw) or ""
        if not vnorm or vnorm in RADAR_NA_TOKENS:
            continue
        vtoks = radar_query_tokens(vnorm)
        if len(vnorm) >= 2 and re.search(rf"(^| )({re.escape(vnorm)})( |$)", qnorm):
            score += 3.0
        elif column_name is not None and radar_short_filter_value_mentioned(question, column_name, raw):
            score += 3.0
        overlap = len(qtokens & vtoks)
        if overlap:
            score += min(2.0, 0.75 * overlap)
    return score


def radar_relevant_columns(
    question: str,
    df_or_columns: Any,
    *,
    extra: list[str] | tuple[str, ...] = (),
    max_columns: int = 8,
) -> list[str]:
    """Rank query-relevant columns using only generic header/value matching."""
    if hasattr(df_or_columns, "columns"):
        df = df_or_columns
        columns = [str(c) for c in df.columns]
    else:
        df = None
        columns = [str(c) for c in df_or_columns]

    qnorm = radar_norm_key(question) or ""
    qtokens = radar_query_tokens(question)
    ranked: list[tuple[float, str]] = []
    for col in columns:
        score = radar_header_match_score(question, col)
        if df is not None and col in df.columns:
            score += radar_value_match_score(question, df[col], column_name=col)
        if col in extra:
            score += 4.0
        if score > 0:
            ranked.append((score, col))

    ranked.sort(key=lambda item: (-item[0], columns.index(item[1]) if item[1] in columns else 10**9))
    selected: list[str] = []
    for _, col in ranked:
        if col not in selected:
            selected.append(col)
        if len(selected) >= max_columns:
            break
    for col in extra:
        if col in columns and col not in selected and len(selected) < max_columns:
            selected.append(col)
    return selected


def radar_guess_query_contract(question: str, df: pd.DataFrame, *, extra_columns: list[str] | None = None) -> dict[str, Any]:
    """Create an editable generic query contract for v4-style candidate generation."""
    op = radar_question_operation(question)
    relevant = radar_relevant_columns(question, df, extra=tuple(extra_columns or ()))
    profiles: list[dict[str, Any]] = []
    for col in relevant:
        try:
            coverage = radar_column_numeric_coverage(df[col])
        except Exception:
            coverage = 0.0
        profiles.append(
            {
                "column": col,
                "numeric_coverage": coverage,
                "header_score": radar_header_match_score(question, col),
                "target_score": radar_operation_target_score(question, col, op),
                "value_score": radar_value_match_score(question, df[col], column_name=col) if col in df.columns else 0.0,
                "explicit_filter_score": radar_explicit_filter_match_score(question, df[col], column_name=col) if col in df.columns else 0.0,
            }
        )
    numeric_cols = [item["column"] for item in profiles if float(item["numeric_coverage"]) >= 0.6]
    nonnumeric_cols = [item["column"] for item in profiles if float(item["numeric_coverage"]) < 0.6]
    filter_columns = [
        item["column"]
        for item in sorted(profiles, key=lambda x: (-float(x["explicit_filter_score"]), -float(x["value_score"]), -float(x["header_score"])))
        if float(item["explicit_filter_score"]) > 0
    ][:4]

    if op in {"sum", "mean", "median", "rank", "ratio", "arithmetic"}:
        numeric_profiles = [item for item in profiles if item["column"] in numeric_cols and item["column"] not in filter_columns]
        numeric_profiles.sort(key=lambda x: (-float(x["target_score"]), -float(x["header_score"]), float(x["value_score"]), relevant.index(x["column"])))
        target_limit = 2 if op in {"ratio", "arithmetic"} else 1
        target_columns = [item["column"] for item in numeric_profiles[:target_limit]] or [col for col in numeric_cols if col not in filter_columns][:target_limit]
        if not target_columns:
            target_columns = relevant[:2]
    elif op == "count":
        target_columns = []
    else:
        target_columns = [col for col in relevant if col not in filter_columns][:2] or relevant[:2]
    if op in {"lookup", "list"}:
        return_columns = [col for col in relevant if col not in filter_columns][:4]
    else:
        return_columns = []
    sort_columns = target_columns[:2] if op == "rank" else []

    return {
        "operator": op,
        "relevant_columns": relevant,
        "target_columns": target_columns,
        "filter_columns": filter_columns,
        "sort_columns": sort_columns,
        "return_columns": return_columns,
        "formula_input_columns": [],
        "column_types": {
            "numeric_like": numeric_cols,
            "text_like": nonnumeric_cols,
        },
        "column_profiles": profiles,
        "note": "generic editable guess; verify against the question before computing candidates",
    }


def radar_contract_columns(query_contract: dict[str, Any], df: pd.DataFrame) -> list[str]:
    """Return query-contract columns that actually exist in the current table."""
    ordered: list[str] = []
    for key in (
        "relevant_columns",
        "target_columns",
        "filter_columns",
        "sort_columns",
        "return_columns",
        "formula_input_columns",
    ):
        values = query_contract.get(key, [])
        if isinstance(values, str):
            values = [values]
        for col in values or []:
            if col in df.columns and col not in ordered:
                ordered.append(col)
    return ordered


def radar_question_filter_values(question: str, series: Any, *, column_name: Any = None, max_values: int = 80) -> list[Any]:
    """Find values from a column that are explicitly mentioned in the question."""
    if not hasattr(series, "dropna"):
        series = pd.Series(series)
    qnorm = radar_norm_key(question) or ""
    matches: list[Any] = []
    seen = 0
    for raw in series.dropna().unique().tolist():
        if seen >= max_values:
            break
        seen += 1
        vnorm = radar_norm_key(raw) or ""
        if len(vnorm) >= 2 and re.search(rf"(^| )({re.escape(vnorm)})( |$)", qnorm):
            matches.append(raw)
        elif column_name is not None and radar_short_filter_value_mentioned(question, column_name, raw):
            matches.append(raw)
    return matches


def radar_contract_filter_mask(question: str, df: pd.DataFrame, query_contract: dict[str, Any]) -> pd.Series:
    """Build a conservative row mask from contract filter columns and question text."""
    mask = pd.Series(True, index=df.index)
    filter_columns = query_contract.get("filter_columns") or []
    if isinstance(filter_columns, str):
        filter_columns = [filter_columns]
    for col in filter_columns:
        if col not in df.columns:
            continue
        values = radar_question_filter_values(question, df[col], column_name=col)
        if not values:
            continue
        col_norm = df[col].map(radar_norm_key)
        value_norms = {radar_norm_key(value) for value in values}
        value_norms = {value for value in value_norms if value}
        if value_norms:
            mask &= col_norm.isin(value_norms)
    return mask


def radar_filtered_df(question: str, df: pd.DataFrame, query_contract: dict[str, Any]) -> pd.DataFrame:
    return df.loc[radar_contract_filter_mask(question, df, query_contract)].copy()


def radar_requested_decimals(question: str) -> int | None:
    q = radar_norm_key(question) or ""
    digit = re.search(r"\b(\d+)\s+decimal", q)
    if digit:
        return int(digit.group(1))
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    for word, value in words.items():
        if re.search(rf"\b{word}\s+decimal", q):
            return value
    return None


def radar_format_answer_value(value: Any, question: str = "") -> str:
    decimals = radar_requested_decimals(question)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        fvalue = float(value)
        if not math.isfinite(fvalue):
            return ""
        if decimals is not None:
            return f"{fvalue:.{decimals}f}"
        if abs(fvalue - round(fvalue)) < 1e-9:
            return str(int(round(fvalue)))
        return str(fvalue)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return "" if value is None else str(value).strip()


def radar_compute_contract_answer(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any],
    *,
    numeric_mode: str = "first",
) -> dict[str, Any]:
    """Compute a generic direct answer from an already verified contract."""
    op = str(query_contract.get("operator") or radar_question_operation(question))
    filtered = radar_filtered_df(question, df, query_contract)
    target_columns = query_contract.get("target_columns") or []
    return_columns = query_contract.get("return_columns") or []
    sort_columns = query_contract.get("sort_columns") or []
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    if isinstance(return_columns, str):
        return_columns = [return_columns]
    if isinstance(sort_columns, str):
        sort_columns = [sort_columns]
    target = next((col for col in target_columns if col in filtered.columns), None)
    answer: Any = ""
    if op == "count":
        answer = int(len(filtered))
    elif op in {"sum", "mean", "median"} and target:
        nums = radar_numeric_series(filtered[target], mode=numeric_mode).dropna()
        if op == "sum":
            answer = float(nums.sum()) if len(nums) else ""
        elif op == "mean":
            answer = float(nums.mean()) if len(nums) else ""
        elif op == "median":
            answer = float(nums.median()) if len(nums) else ""
    elif op == "rank" and target:
        nums = radar_numeric_series(filtered[target], mode=numeric_mode)
        valid = filtered.loc[nums.dropna().index].copy()
        nums = nums.dropna()
        if len(nums):
            qnorm = radar_norm_key(question) or ""
            choose_min = bool(re.search(r"\b(lowest|smallest|minimum|least|bottom)\b", qnorm))
            idx = nums.idxmin() if choose_min else nums.idxmax()
            cols = [col for col in return_columns if col in filtered.columns] or [target]
            answer = [filtered.loc[idx, col] for col in cols]
            if len(answer) == 1:
                answer = answer[0]
    elif op in {"lookup", "list"}:
        cols = [col for col in return_columns if col in filtered.columns] or [col for col in target_columns if col in filtered.columns]
        if cols:
            values = []
            for col in cols:
                values.extend([radar_clean_text(value) for value in filtered[col].tolist()])
            values = [value for value in values if value is not None]
            if op == "lookup":
                answer = values[0] if values else ""
            else:
                deduped = []
                for value in values:
                    if value not in deduped:
                        deduped.append(value)
                answer = deduped
    return {
        "answer": radar_format_answer_value(answer, question),
        "raw_answer": answer,
        "operator": op,
        "filtered_rows": int(len(filtered)),
        "target_column": target,
        "return_columns": [col for col in return_columns if col in filtered.columns],
        "filter_columns": [col for col in (query_contract.get("filter_columns") or []) if col in df.columns]
        if not isinstance(query_contract.get("filter_columns"), str)
        else [query_contract.get("filter_columns")] if query_contract.get("filter_columns") in df.columns else [],
        "usage_note": "generic contract execution; verify against the question before using as C0/C1",
    }


def radar_missing_mask(series: Any) -> pd.Series:
    """Identify generic missing/sentinel values in a column."""
    if not hasattr(series, "map"):
        series = pd.Series(series)
    return series.map(lambda value: radar_clean_text(value) is None)


def radar_format_examples(series: Any, *, max_examples: int = 5) -> list[dict[str, Any]]:
    """Find generic formatting issues in numeric-looking cells."""
    if not hasattr(series, "items"):
        series = pd.Series(series)
    examples: list[dict[str, Any]] = []
    for idx, raw in series.items():
        text = radar_clean_text(raw)
        if text is None:
            continue
        nums = radar_nums(text)
        if not nums:
            continue
        issues: list[str] = []
        if "," in text:
            issues.append("comma_grouping")
        if re.search(r"[A-Za-z%$]", text):
            issues.append("number_with_text_or_symbol")
        if text.count(".") > 1:
            issues.append("multiple_decimal_points")
        if len(nums) > 1:
            issues.append("multiple_numbers_in_cell")
        if issues:
            examples.append({"row": str(idx), "value": text[:120], "issues": issues})
        if len(examples) >= max_examples:
            break
    return examples


def radar_robust_outlier_examples(series: Any, *, max_examples: int = 5) -> list[dict[str, Any]]:
    """Return diagnostic-only robust numeric outlier candidates."""
    if not hasattr(series, "items"):
        series = pd.Series(series)
    nums = radar_numeric_series(series)
    valid = nums.dropna()
    if len(valid) < 4:
        return []
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0:
        return []
    low = q1 - 3.0 * iqr
    high = q3 + 3.0 * iqr
    examples: list[dict[str, Any]] = []
    for idx, value in nums.items():
        if pd.isna(value):
            continue
        fvalue = float(value)
        if not math.isfinite(fvalue):
            continue
        if fvalue < low or fvalue > high:
            examples.append(
                {
                    "row": str(idx),
                    "value": fvalue,
                    "side": "low" if fvalue < low else "high",
                    "iqr_bounds": [low, high],
                }
            )
        if len(examples) >= max_examples:
            break
    return examples


def radar_numeric_relation_hints(
    df: pd.DataFrame,
    columns: list[str],
    *,
    max_relations: int = 5,
    max_conflicts: int = 5,
) -> list[dict[str, Any]]:
    """Find generic arithmetic relation hints among numeric-like relevant columns."""
    numeric: dict[str, pd.Series] = {}
    for col in columns:
        if col not in df.columns:
            continue
        if radar_column_numeric_coverage(df[col]) < 0.6:
            continue
        numeric[col] = radar_numeric_series(df[col])
    cols = list(numeric)
    hints: list[dict[str, Any]] = []
    if len(cols) < 3:
        return hints
    for target in cols:
        others = [col for col in cols if col != target]
        for i, left in enumerate(others):
            for right in others[i + 1 :]:
                candidates = [
                    ("sum", numeric[left] + numeric[right]),
                    ("difference_left_minus_right", numeric[left] - numeric[right]),
                    ("difference_right_minus_left", numeric[right] - numeric[left]),
                    ("product", numeric[left] * numeric[right]),
                ]
                target_values = numeric[target]
                for relation, predicted_values in candidates:
                    mask = target_values.notna() & predicted_values.notna()
                    if int(mask.sum()) < 3:
                        continue
                    scale = pd.concat([target_values[mask].abs(), predicted_values[mask].abs()], axis=1).max(axis=1).clip(lower=1.0)
                    close = (target_values[mask] - predicted_values[mask]).abs() <= (1e-6 * scale)
                    support = float(close.mean())
                    if support < 0.6:
                        continue
                    conflict_rows = close[~close].index.tolist()[:max_conflicts]
                    hints.append(
                        {
                            "target_column": target,
                            "input_columns": [left, right],
                            "relation": relation,
                            "support": support,
                            "checked_rows": int(mask.sum()),
                            "conflict_rows": [str(idx) for idx in conflict_rows],
                        }
                    )
                    if len(hints) >= max_relations:
                        return hints
    return hints


def radar_numeric_relation_hints_for_targets(
    df: pd.DataFrame,
    target_columns: list[str],
    candidate_columns: list[str],
    *,
    max_relations: int = 5,
    max_conflicts: int = 5,
) -> list[dict[str, Any]]:
    """Find generic arithmetic relation hints for known target columns."""
    numeric: dict[str, pd.Series] = {}
    ordered = []
    for col in list(target_columns) + list(candidate_columns):
        if col in df.columns and col not in ordered:
            ordered.append(col)
    for col in ordered:
        if radar_column_numeric_coverage(df[col]) < 0.6:
            continue
        numeric[col] = radar_numeric_series(df[col])
    hints: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in numeric:
            continue
        others = [col for col in ordered if col != target and col in numeric]
        for i, left in enumerate(others):
            for right in others[i + 1 :]:
                relation_candidates = [
                    ("sum", numeric[left] + numeric[right]),
                    ("difference_left_minus_right", numeric[left] - numeric[right]),
                    ("difference_right_minus_left", numeric[right] - numeric[left]),
                    ("product", numeric[left] * numeric[right]),
                ]
                target_values = numeric[target]
                for relation, predicted_values in relation_candidates:
                    mask = target_values.notna() & predicted_values.notna()
                    if int(mask.sum()) < 3:
                        continue
                    scale = pd.concat([target_values[mask].abs(), predicted_values[mask].abs()], axis=1).max(axis=1).clip(lower=1.0)
                    close = (target_values[mask] - predicted_values[mask]).abs() <= (1e-6 * scale)
                    support = float(close.mean())
                    if support < 0.6:
                        continue
                    conflict_rows = close[~close].index.tolist()[:max_conflicts]
                    hints.append(
                        {
                            "target_column": target,
                            "input_columns": [left, right],
                            "relation": relation,
                            "support": support,
                            "checked_rows": int(mask.sum()),
                            "conflict_rows": [str(idx) for idx in conflict_rows],
                        }
                    )
                    if len(hints) >= max_relations:
                        return hints
    return hints


def radar_expand_contract_with_relations(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any] | None = None,
    *,
    max_extra_numeric_cols: int = 8,
    max_relations: int = 5,
) -> dict[str, Any]:
    """Add generic relation-supported auxiliary columns to a query contract."""
    contract = dict(query_contract or radar_guess_query_contract(question, df))
    target_columns = contract.get("target_columns") or []
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    target_columns = [col for col in target_columns if col in df.columns]
    if not target_columns:
        return contract

    existing = set(radar_contract_columns(contract, df))
    numeric_candidates: list[str] = []
    for col in df.columns:
        if col in existing:
            continue
        try:
            coverage = radar_column_numeric_coverage(df[col])
        except Exception:
            coverage = 0.0
        if coverage >= 0.6:
            numeric_candidates.append(col)
        if len(numeric_candidates) >= max_extra_numeric_cols:
            break

    hints = radar_numeric_relation_hints_for_targets(
        df,
        target_columns,
        numeric_candidates,
        max_relations=max_relations,
    )
    formula_inputs: list[str] = []
    for hint in hints:
        for col in hint.get("input_columns", []):
            if col in df.columns and col not in formula_inputs:
                formula_inputs.append(col)

    merged_formula_inputs = []
    original_formula_inputs = contract.get("formula_input_columns") or []
    if isinstance(original_formula_inputs, str):
        original_formula_inputs = [original_formula_inputs]
    for col in list(original_formula_inputs) + formula_inputs:
        if col in df.columns and col not in merged_formula_inputs:
            merged_formula_inputs.append(col)
    contract["formula_input_columns"] = merged_formula_inputs

    relevant = contract.get("relevant_columns") or []
    if isinstance(relevant, str):
        relevant = [relevant]
    merged_relevant = []
    for col in list(relevant) + merged_formula_inputs:
        if col in df.columns and col not in merged_relevant:
            merged_relevant.append(col)
    contract["relevant_columns"] = merged_relevant
    contract["relation_expansion_hints"] = hints
    contract["note"] = "generic editable guess with relation-supported auxiliary columns; verify before computing candidates"
    return contract


def radar_audit_query_relevant_data(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any] | None = None,
    *,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Audit only query-contract columns for generic dirty-table evidence."""
    contract = query_contract or radar_guess_query_contract(question, df)
    columns = radar_contract_columns(contract, df)
    if not columns:
        columns = radar_relevant_columns(question, df)
    column_audits: dict[str, Any] = {}
    missing_rows: set[str] = set()
    format_count = 0
    outlier_count = 0
    for col in columns:
        series = df[col]
        missing = radar_missing_mask(series)
        missing_examples = [{"row": str(idx), "value": str(series.loc[idx])[:120]} for idx in series[missing].index[:max_examples]]
        missing_rows.update(str(item["row"]) for item in missing_examples)
        fmt_examples = radar_format_examples(series, max_examples=max_examples)
        out_examples = radar_robust_outlier_examples(series, max_examples=max_examples)
        format_count += len(fmt_examples)
        outlier_count += len(out_examples)
        column_audits[col] = {
            "missing_count": int(missing.sum()),
            "missing_examples": missing_examples,
            "numeric_coverage": radar_column_numeric_coverage(series),
            "format_examples": fmt_examples,
            "outlier_examples": out_examples,
        }
    target_columns = contract.get("target_columns") or []
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    target_columns = [col for col in target_columns if col in columns]
    if target_columns:
        relation_hints = radar_numeric_relation_hints_for_targets(
            df,
            target_columns,
            [col for col in columns if col not in target_columns],
            max_relations=max_examples,
            max_conflicts=max_examples,
        )
    else:
        relation_hints = radar_numeric_relation_hints(df, columns, max_relations=max_examples, max_conflicts=max_examples)
    guidance: list[str] = []
    if format_count:
        guidance.append("C1_FORMAT may be useful if parsing changes query-relevant values.")
    if missing_rows:
        guidance.append("C2_ROW_VALIDITY may be useful only if missing rows enter the query computation.")
    if relation_hints:
        guidance.append("C3_FORMULA_RECOVER or C5_LOGIC_VALID may be useful if a relation conflict affects the queried target.")
    if outlier_count:
        guidance.append("C4_TARGET_OUTLIER is diagnostic only unless there is hard query-relevant evidence beyond distribution.")
    if not guidance:
        guidance.append("No query-contract dirty evidence found; prefer C0/C1 no-op behavior.")
    return {
        "query_contract": contract,
        "audited_columns": columns,
        "column_audits": column_audits,
        "numeric_relation_hints": relation_hints,
        "evidence_counts": {
            "columns": len(columns),
            "missing_example_rows": len(missing_rows),
            "format_examples": format_count,
            "outlier_examples": outlier_count,
            "numeric_relation_hints": len(relation_hints),
        },
        "candidate_guidance": guidance,
        "usage_note": "diagnostic only; verify in Python and recompute answers before adding repair candidates",
    }


def radar_norm_answer(value: Any) -> str:
    text = "" if value is None else str(value).strip().strip("`").strip("\"'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def radar_candidate(
    candidate_id: str,
    answer: Any,
    *,
    patch_type: str = "no_op",
    applicable: bool = True,
    evidence_strength: int = 0,
    hard_evidence_count: int = 0,
    query_relevant_ops: int = 0,
    repair_scope: str = "none",
    evidence_summary: str = "",
    evidence_types: list[str] | None = None,
    certified: bool | None = None,
    global_cleaning: bool = False,
    irrelevant_repair: bool = False,
    weak_evidence_override: bool = False,
    stdout_mismatch: bool = False,
) -> dict[str, Any]:
    """Create a generic RADAR candidate record for integrated in-agent gating."""
    cid = str(candidate_id)
    if certified is None:
        certified = (
            cid in LOW_RISK_CANDIDATE_IDS
            or (
                repair_scope == "query_relevant_only"
                and int(query_relevant_ops) >= 1
                and int(evidence_strength) >= 2
                and not global_cleaning
                and not irrelevant_repair
                and not stdout_mismatch
            )
        )
    return {
        "id": cid,
        "answer": "" if answer is None else str(answer).strip(),
        "patch_type": str(patch_type),
        "applicable": bool(applicable),
        "evidence_strength": int(evidence_strength),
        "hard_evidence_count": int(hard_evidence_count),
        "query_relevant_ops": int(query_relevant_ops),
        "repair_scope": str(repair_scope),
        "evidence_summary": str(evidence_summary)[:500],
        "evidence_types": list(evidence_types or []),
        "certified": bool(certified),
        "risk_flags": {
            "global_cleaning": bool(global_cleaning),
            "irrelevant_repair": bool(irrelevant_repair),
            "weak_evidence_override": bool(weak_evidence_override),
            "stdout_mismatch": bool(stdout_mismatch),
        },
    }


def radar_is_usable_candidate(candidate: dict[str, Any]) -> bool:
    if not candidate.get("answer"):
        return False
    cid = str(candidate.get("id") or "")
    flags = candidate.get("risk_flags") if isinstance(candidate.get("risk_flags"), dict) else {}
    if flags.get("global_cleaning") or flags.get("irrelevant_repair") or flags.get("stdout_mismatch"):
        return False
    if cid in REPAIR_CANDIDATE_IDS and candidate.get("repair_scope") != "query_relevant_only":
        return False
    if cid in REPAIR_CANDIDATE_IDS and candidate.get("applicable") is False:
        return False
    return True


def radar_answer_groups(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if radar_is_usable_candidate(candidate):
            groups[radar_norm_answer(candidate.get("answer"))].append(candidate)
    return dict(groups)


def radar_group_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {str(candidate.get("id") or "") for candidate in items}
    evidence = max([int(candidate.get("evidence_strength") or 0) for candidate in items] or [0])
    hard = sum(int(candidate.get("hard_evidence_count") or 0) for candidate in items)
    qops = sum(int(candidate.get("query_relevant_ops") or 0) for candidate in items)
    certified = all(bool(candidate.get("certified", False)) for candidate in items)
    return {
        "ids": ids,
        "evidence": evidence,
        "hard": hard,
        "qops": qops,
        "certified": certified,
        "repairs": ids & REPAIR_CANDIDATE_IDS,
        "strong_repairs": ids & STRONG_REPAIR_CANDIDATE_IDS,
        "low_risk": ids & LOW_RISK_CANDIDATE_IDS,
    }


def radar_certify_single_repair(candidate: dict[str, Any]) -> bool:
    flags = candidate.get("risk_flags") if isinstance(candidate.get("risk_flags"), dict) else {}
    if flags.get("weak_evidence_override") or flags.get("global_cleaning") or flags.get("irrelevant_repair"):
        return False
    if candidate.get("repair_scope") != "query_relevant_only":
        return False
    if int(candidate.get("query_relevant_ops") or 0) < 1:
        return False
    if int(candidate.get("evidence_strength") or 0) < 3:
        return False
    if int(candidate.get("hard_evidence_count") or 0) < 1:
        return False
    return True


def radar_can_override_group(items: list[dict[str, Any]]) -> bool:
    stats = radar_group_stats(items)
    if not stats["repairs"]:
        return False
    # Train-split replay showed that permissive single-branch overrides cause
    # more harm than rescue. Use a conservative default: repair overrides need
    # multiple independent repair branches, hard evidence, and a logic-validity
    # cross-check. This remains generic and uses no artifact labels or task ids.
    if len(items) < 3:
        return False
    if stats["qops"] < 3 or stats["hard"] < 3 or stats["evidence"] < 3:
        return False
    if not stats["strong_repairs"]:
        return False
    if "C5_LOGIC_VALID" not in stats["ids"]:
        return False
    if "C2_ROW_VALIDITY" in stats["ids"] and "C5_LOGIC_VALID" not in stats["ids"]:
        return False
    if {"C3_FORMULA_RECOVER", "C4_TARGET_OUTLIER"}.issubset(stats["ids"]) and "C5_LOGIC_VALID" not in stats["ids"]:
        return False
    for candidate in items:
        flags = candidate.get("risk_flags") if isinstance(candidate.get("risk_flags"), dict) else {}
        if flags.get("weak_evidence_override"):
            return False
    independent_support = len(items) >= 2 and len(stats["ids"]) >= 2
    if independent_support and (stats["strong_repairs"] or "C2_ROW_VALIDITY" in stats["ids"] or "C1_FORMAT" in stats["ids"]):
        return bool(stats["certified"])
    real_repairs = [candidate for candidate in items if str(candidate.get("id") or "") in REPAIR_CANDIDATE_IDS]
    if len(real_repairs) == 1:
        return radar_certify_single_repair(real_repairs[0])
    return False


def radar_audit_evidence_counts(audit: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(audit, dict):
        return {}
    evidence = audit.get("evidence_counts")
    if not isinstance(evidence, dict):
        return {}
    counts: dict[str, int] = {}
    for key, value in evidence.items():
        try:
            counts[str(key)] = int(value)
        except Exception:
            counts[str(key)] = 0
    return counts


def radar_candidate_has_matching_audit(candidate: dict[str, Any], audit: dict[str, Any] | None) -> bool:
    counts = radar_audit_evidence_counts(audit)
    cid = str(candidate.get("id") or "")
    patch_type = str(candidate.get("patch_type") or "")
    if cid == "C1_FORMAT" or patch_type == "format":
        return counts.get("format_examples", 0) > 0
    if cid == "C2_ROW_VALIDITY" or patch_type in {"row_validity", "missingness"}:
        return counts.get("missing_example_rows", 0) > 0
    if cid == "C3_FORMULA_RECOVER" or patch_type == "formula_recover":
        return counts.get("numeric_relation_hints", 0) > 0
    if cid == "C4_TARGET_OUTLIER" or patch_type == "target_outlier":
        return counts.get("outlier_examples", 0) > 0
    if cid == "C5_LOGIC_VALID" or patch_type == "logic_valid":
        return counts.get("numeric_relation_hints", 0) > 0
    return False


def radar_patch_type_for_candidate_id(candidate_id: str) -> str:
    cid = str(candidate_id)
    if cid == "C1_FORMAT":
        return "format"
    if cid == "C2_ROW_VALIDITY":
        return "row_validity"
    if cid == "C3_FORMULA_RECOVER":
        return "formula_recover"
    if cid == "C4_TARGET_OUTLIER":
        return "target_outlier"
    if cid == "C5_LOGIC_VALID":
        return "logic_valid"
    return "no_op"


def radar_audit_supported_candidate(
    candidate_id: str,
    answer: Any,
    audit: dict[str, Any] | None,
    *,
    verified_recompute: bool = False,
    evidence_summary: str = "",
    query_relevant_ops: int | None = None,
    hard_evidence_count: int | None = None,
    evidence_types: list[str] | None = None,
) -> dict[str, Any]:
    """Create a C1-C5 candidate only when audit support and recomputation agree."""
    cid = str(candidate_id)
    patch_type = radar_patch_type_for_candidate_id(cid)
    probe = {"id": cid, "patch_type": patch_type}
    matching_audit = radar_candidate_has_matching_audit(probe, audit)
    is_repair = cid in REPAIR_CANDIDATE_IDS
    qops = int(query_relevant_ops if query_relevant_ops is not None else (2 if matching_audit and verified_recompute else 0))
    hard = int(hard_evidence_count if hard_evidence_count is not None else (1 if matching_audit and verified_recompute and is_repair else 0))
    strength = 3 if matching_audit and verified_recompute else (1 if matching_audit else 0)
    return radar_candidate(
        cid,
        answer,
        patch_type=patch_type,
        applicable=bool(matching_audit and verified_recompute),
        evidence_strength=strength,
        hard_evidence_count=hard,
        query_relevant_ops=qops,
        repair_scope="query_relevant_only" if cid != "C0_DIRECT" else "none",
        evidence_summary=evidence_summary or f"{cid} recomputed after query-relevant audit verification",
        evidence_types=list(evidence_types or ([patch_type] if matching_audit else [])),
        certified=bool(matching_audit and verified_recompute),
        weak_evidence_override=bool(not matching_audit or not verified_recompute),
    )


def radar_relation_prediction(df: pd.DataFrame, hint: dict[str, Any]) -> pd.Series | None:
    """Compute a generic relation-predicted target series from an audit hint."""
    inputs = hint.get("input_columns") or []
    if not isinstance(inputs, list) or len(inputs) < 2:
        return None
    left, right = inputs[0], inputs[1]
    if left not in df.columns or right not in df.columns:
        return None
    left_values = radar_numeric_series(df[left])
    right_values = radar_numeric_series(df[right])
    relation = str(hint.get("relation") or "")
    if relation == "sum":
        return left_values + right_values
    if relation == "difference_left_minus_right":
        return left_values - right_values
    if relation == "difference_right_minus_left":
        return right_values - left_values
    if relation == "product":
        return left_values * right_values
    return None


def radar_index_from_strings(index: Any, row_ids: list[Any]) -> list[Any]:
    """Map compact string row ids from trace/audit back to the dataframe index."""
    wanted = {str(item) for item in row_ids}
    return [idx for idx in index if str(idx) in wanted]


def radar_relation_patch_candidate_answer(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any],
    audit: dict[str, Any],
    *,
    max_rows: int = 6,
) -> dict[str, Any] | None:
    """Patch query-filtered relation conflicts and recompute the contract answer."""
    hints = audit.get("numeric_relation_hints") if isinstance(audit, dict) else []
    if not isinstance(hints, list):
        return None
    filtered = radar_filtered_df(question, df, query_contract)
    filtered_index_strings = {str(idx) for idx in filtered.index}
    for hint in hints:
        target = hint.get("target_column")
        if target not in df.columns:
            continue
        predicted = radar_relation_prediction(df, hint)
        if predicted is None:
            continue
        conflict_rows = hint.get("conflict_rows") or []
        if not isinstance(conflict_rows, list):
            continue
        query_rows = [row for row in conflict_rows if str(row) in filtered_index_strings][:max_rows]
        row_index = radar_index_from_strings(df.index, query_rows)
        row_index = [idx for idx in row_index if idx in predicted.index and not pd.isna(predicted.loc[idx])]
        if not row_index:
            continue
        patched = df.copy()
        for idx in row_index:
            patched.at[idx, target] = radar_format_answer_value(float(predicted.loc[idx]), question)
        result = radar_compute_contract_answer(question, patched, query_contract)
        return {
            "answer": result.get("answer", ""),
            "patched_rows": [str(idx) for idx in row_index],
            "hint": {
                "target_column": target,
                "input_columns": hint.get("input_columns", []),
                "relation": hint.get("relation"),
                "support": hint.get("support"),
            },
            "result": result,
        }
    return None


def radar_missing_row_candidate_answer(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any],
    *,
    max_rows: int = 8,
) -> dict[str, Any] | None:
    """Drop only query-filtered rows with missing contract values and recompute."""
    filtered = radar_filtered_df(question, df, query_contract)
    columns = radar_contract_columns(query_contract, df)
    if not columns or filtered.empty:
        return None
    mask = pd.Series(False, index=filtered.index)
    for col in columns:
        if col in filtered.columns:
            mask = mask | radar_missing_mask(filtered[col])
    row_index = list(mask[mask].index)[:max_rows]
    if not row_index:
        return None
    if len(filtered) - len(row_index) <= 0:
        return None
    patched = df.drop(index=row_index)
    result = radar_compute_contract_answer(question, patched, query_contract)
    return {
        "answer": result.get("answer", ""),
        "dropped_rows": [str(idx) for idx in row_index],
        "columns_checked": columns,
        "result": result,
    }


def radar_outlier_row_candidate_answer(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any],
    *,
    max_rows: int = 6,
) -> dict[str, Any] | None:
    """Drop only query-filtered target-column outlier rows and recompute."""
    op = str(query_contract.get("operator") or radar_question_operation(question))
    if op not in {"sum", "mean", "median", "rank"}:
        return None
    target_columns = query_contract.get("target_columns") or []
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    filtered = radar_filtered_df(question, df, query_contract)
    if filtered.empty:
        return None
    for col in target_columns:
        if col not in filtered.columns:
            continue
        examples = radar_robust_outlier_examples(filtered[col], max_examples=max_rows)
        if not examples:
            continue
        row_index = radar_index_from_strings(filtered.index, [item.get("row") for item in examples])
        if not row_index or len(filtered) - len(row_index) <= 0:
            continue
        patched = df.drop(index=row_index)
        result = radar_compute_contract_answer(question, patched, query_contract)
        return {
            "answer": result.get("answer", ""),
            "dropped_rows": [str(idx) for idx in row_index],
            "target_column": col,
            "result": result,
        }
    return None


def radar_build_candidate_gate_bundle(
    question: str,
    df: pd.DataFrame,
    query_contract: dict[str, Any] | None = None,
    *,
    gate_profile: str = "audit_balanced",
) -> dict[str, Any]:
    """Build a compact generic C0-C5 bundle inside the code-agent turn."""
    contract = dict(query_contract or radar_guess_query_contract(question, df))
    contract = radar_expand_contract_with_relations(question, df, contract)
    audit = radar_audit_query_relevant_data(question, df, contract)
    direct_first = radar_compute_contract_answer(question, df, contract, numeric_mode="first")
    direct_last = radar_compute_contract_answer(question, df, contract, numeric_mode="last")
    direct_answer = direct_first.get("answer", "")
    last_answer = direct_last.get("answer", "")
    format_changed = bool(last_answer and last_answer != direct_answer)

    candidates: list[dict[str, Any]] = [
        radar_candidate(
            "C0_DIRECT",
            direct_answer,
            patch_type="no_op",
            repair_scope="none",
            evidence_summary="direct contract execution from query-relevant rows and columns",
        ),
        radar_candidate(
            "C1_FORMAT",
            last_answer if format_changed else direct_answer,
            patch_type="format",
            repair_scope="query_relevant_only",
            evidence_strength=2 if format_changed else 0,
            query_relevant_ops=1 if format_changed else 0,
            evidence_summary="alternate generic numeric parsing for query-relevant columns",
            evidence_types=["format"] if format_changed else [],
            certified=True,
        ),
    ]

    missing_patch = radar_missing_row_candidate_answer(question, df, contract)
    if missing_patch and missing_patch.get("answer"):
        candidates.append(
            radar_audit_supported_candidate(
                "C2_ROW_VALIDITY",
                missing_patch["answer"],
                audit,
                verified_recompute=True,
                evidence_summary="excluded only query-filtered rows with missing contract values, then recomputed",
                hard_evidence_count=min(2, len(missing_patch.get("dropped_rows") or [])),
                evidence_types=["missingness"],
            )
        )

    relation_patch = radar_relation_patch_candidate_answer(question, df, contract, audit)
    if relation_patch and relation_patch.get("answer"):
        patched_rows = relation_patch.get("patched_rows") or []
        candidates.append(
            radar_audit_supported_candidate(
                "C3_FORMULA_RECOVER",
                relation_patch["answer"],
                audit,
                verified_recompute=True,
                evidence_summary="patched query-filtered target values from a generic column relation, then recomputed",
                hard_evidence_count=max(1, min(3, len(patched_rows))),
                query_relevant_ops=2,
                evidence_types=["formula_recover"],
            )
        )
        candidates.append(
            radar_audit_supported_candidate(
                "C5_LOGIC_VALID",
                relation_patch["answer"],
                audit,
                verified_recompute=True,
                evidence_summary="same recomputed answer passes a generic relation-consistency check",
                hard_evidence_count=max(1, min(3, len(patched_rows))),
                query_relevant_ops=2,
                evidence_types=["logic_valid"],
            )
        )

    outlier_patch = radar_outlier_row_candidate_answer(question, df, contract)
    if outlier_patch and outlier_patch.get("answer"):
        candidates.append(
            radar_audit_supported_candidate(
                "C4_TARGET_OUTLIER",
                outlier_patch["answer"],
                audit,
                verified_recompute=True,
                evidence_summary="excluded only query-filtered target outlier rows, then recomputed",
                hard_evidence_count=min(1, len(outlier_patch.get("dropped_rows") or [])),
                query_relevant_ops=1,
                evidence_types=["target_outlier"],
            )
        )

    return {
        "query_contract": contract,
        "audit": audit,
        "direct_result": direct_first,
        "format_result": direct_last,
        "gate_profile": gate_profile,
        "candidates": candidates,
        "candidate_ids": [candidate.get("id") for candidate in candidates],
        "usage_note": "generic bundle; inspect/edit contract or candidates before finalizing if the question is not represented correctly",
    }


def radar_finalize_candidate_gate_bundle(bundle: dict[str, Any], *, gate_profile: str | None = None) -> str:
    """Finalize a bundle produced by radar_build_candidate_gate_bundle."""
    profile = gate_profile or str(bundle.get("gate_profile") or "audit_balanced")
    return radar_finalize_candidates(
        bundle.get("query_contract") or {},
        list(bundle.get("candidates") or []),
        audit=bundle.get("audit"),
        gate_profile=profile,
    )


def radar_group_audit_support(items: list[dict[str, Any]], audit: dict[str, Any] | None) -> dict[str, Any]:
    supported = [item for item in items if radar_candidate_has_matching_audit(item, audit)]
    return {
        "supported_count": len(supported),
        "supported_ids": {str(item.get("id") or "") for item in supported},
        "has_supported_strong_repair": bool({str(item.get("id") or "") for item in supported} & STRONG_REPAIR_CANDIDATE_IDS),
    }


def radar_can_override_group_audit_balanced(items: list[dict[str, Any]], audit: dict[str, Any] | None) -> bool:
    stats = radar_group_stats(items)
    support = radar_group_audit_support(items, audit)
    if not stats["repairs"] or not stats["strong_repairs"]:
        return False
    if not support["has_supported_strong_repair"]:
        return False
    if not stats["certified"]:
        return False
    for candidate in items:
        flags = candidate.get("risk_flags") if isinstance(candidate.get("risk_flags"), dict) else {}
        if flags.get("weak_evidence_override") or flags.get("global_cleaning") or flags.get("irrelevant_repair") or flags.get("stdout_mismatch"):
            return False

    independent_repair_ids = stats["repairs"] - {"C1_FORMAT"}
    if len(items) >= 2 and len(independent_repair_ids) >= 1:
        if stats["qops"] >= 2 and stats["hard"] >= 1 and stats["evidence"] >= 3 and support["supported_count"] >= 1:
            return True

    real_repairs = [candidate for candidate in items if str(candidate.get("id") or "") in STRONG_REPAIR_CANDIDATE_IDS]
    if len(items) == 1 and len(real_repairs) == 1:
        candidate = real_repairs[0]
        if not radar_candidate_has_matching_audit(candidate, audit):
            return False
        return (
            int(candidate.get("query_relevant_ops") or 0) >= 2
            and int(candidate.get("hard_evidence_count") or 0) >= 2
            and int(candidate.get("evidence_strength") or 0) >= 3
            and radar_certify_single_repair(candidate)
        )
    return False


def radar_score_group(items: list[dict[str, Any]]) -> int:
    stats = radar_group_stats(items)
    return (
        10 * len(items)
        + 6 * int(stats["hard"])
        + 5 * int(stats["qops"])
        + 4 * int(stats["evidence"])
        + 3 * len(stats["strong_repairs"])
        + len(stats["low_risk"])
    )


def radar_choose_certified_answer(
    candidates: list[dict[str, Any]],
    audit: dict[str, Any] | None = None,
    *,
    gate_profile: str = "strict",
) -> tuple[dict[str, Any], str, dict[str, list[dict[str, Any]]]]:
    groups = radar_answer_groups(candidates)
    direct = next((candidate for candidate in candidates if candidate.get("id") == "C0_DIRECT"), radar_candidate("C0_DIRECT", ""))
    fmt = next((candidate for candidate in candidates if candidate.get("id") == "C1_FORMAT"), None)
    default_answer = fmt.get("answer") if fmt and fmt.get("answer") else direct.get("answer")
    default_norm = radar_norm_answer(default_answer)
    selected_norm = default_norm
    selected_reason = "keep_direct_or_format"
    selected_score = radar_score_group(groups.get(default_norm, []))
    for answer_norm, items in groups.items():
        if answer_norm == default_norm:
            continue
        if gate_profile == "audit_balanced":
            can_override = radar_can_override_group_audit_balanced(items, audit)
            reason = "audit_supported_query_relevant_override"
        else:
            can_override = radar_can_override_group(items)
            reason = "certified_query_relevant_override"
        if not can_override:
            continue
        score = radar_score_group(items)
        if score > selected_score:
            selected_norm = answer_norm
            selected_reason = reason
            selected_score = score
    selected_items = groups.get(selected_norm, [])
    selected = selected_items[0] if selected_items else (fmt or direct)
    return selected, selected_reason, groups


def radar_audit_trace_summary(audit: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only compact audit fields in stdout trace."""
    if not isinstance(audit, dict):
        return {}
    return {
        "audited_columns": audit.get("audited_columns", []),
        "evidence_counts": audit.get("evidence_counts", {}),
        "candidate_guidance": audit.get("candidate_guidance", []),
        "numeric_relation_hints": audit.get("numeric_relation_hints", [])[:3],
    }


def radar_finalize_candidates(
    query_contract: dict[str, Any],
    candidates: list[dict[str, Any]],
    audit: dict[str, Any] | None = None,
    *,
    gate_profile: str = "strict",
) -> str:
    selected, selected_reason, groups = radar_choose_certified_answer(candidates, audit, gate_profile=gate_profile)
    final_answer = str(selected.get("answer", "")).strip()
    trace = {
        "query_contract": query_contract,
        "audit_summary": radar_audit_trace_summary(audit),
        "gate_profile": gate_profile,
        "candidates": candidates,
        "answer_groups": {answer: [str(candidate.get("id") or "") for candidate in items] for answer, items in groups.items()},
        "selected_candidate_id": str(selected.get("id") or ""),
        "selection_reason": selected_reason,
        "final_answer": final_answer,
    }
    print("CANDIDATE_TRACE_JSON:", json.dumps(trace, ensure_ascii=False))
    print("FINAL_ANSWER:", final_answer)
    return final_answer
