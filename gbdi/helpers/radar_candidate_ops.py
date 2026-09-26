from __future__ import annotations

import math
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd


RT_NA_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "?",
    "missing",
    "unknown",
}

RT_BAD_TOKENS = {
    "test",
    "dummy",
    "placeholder",
    "invalid",
    "error",
    "#ref!",
    "#ref",
    "#value!",
    "#value",
    "#div/0!",
    "#num!",
    "#name?",
    "#n/a",
}

RT_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def rt_clean_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().replace("\u2212", "-")
    if text.lower() in RT_NA_TOKENS:
        return None
    return text


def rt_norm_key(value: Any) -> str | None:
    text = rt_clean_text(value)
    if text is None:
        return None
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def rt_decimal_token(token: str) -> str:
    token = token.replace(",", "")
    if token.count(".") <= 1:
        return token
    sign = "-" if token.startswith("-") else ""
    body = token[1:] if sign else token
    parts = [p for p in body.split(".") if p]
    if not parts:
        return "nan"
    return sign + parts[0] + ("." + "".join(parts[1:]) if len(parts) > 1 else "")


def rt_numbers(value: Any) -> list[float]:
    text = rt_clean_text(value)
    if text is None:
        return []
    out: list[float] = []
    for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d*)?", text):
        try:
            out.append(float(rt_decimal_token(token)))
        except Exception:
            pass
    low = text.lower().strip()
    if not out and low in RT_NUM_WORDS:
        out.append(float(RT_NUM_WORDS[low]))
    return out


def rt_num(value: Any, mode: str = "first") -> float:
    nums = rt_numbers(value)
    if not nums:
        return np.nan
    if mode == "last":
        return float(nums[-1])
    if mode == "min":
        return float(min(nums))
    if mode == "max":
        return float(max(nums))
    return float(nums[0])


def rt_num_series(series: pd.Series, mode: str = "first") -> pd.Series:
    return series.map(lambda x: rt_num(x, mode=mode)).astype(float)


def rt_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.map(rt_clean_text), errors="coerce", infer_datetime_format=True)


def rt_is_missing(value: Any) -> bool:
    return rt_clean_text(value) is None


def rt_is_bad_token(value: Any) -> bool:
    text = rt_clean_text(value)
    if text is None:
        return False
    low = text.strip().lower()
    return low in RT_BAD_TOKENS or bool(re.fullmatch(r"#(?:ref|value|div/0|num|name|n/a).*!?", low))


def rt_suspicious_values(series: pd.Series, max_items: int = 6) -> list[str]:
    vals: list[str] = []
    for value in series.tolist():
        text = rt_clean_text(value)
        low = "" if text is None else text.lower()
        suspicious = (
            text is None
            or rt_is_bad_token(value)
            or bool(re.search(r"[A-Za-z%$]", text or ""))
            or bool(re.fullmatch(r"-?(?:9{2,}|1{3,})(?:\.0+)?", low))
        )
        if suspicious:
            shown = "<missing>" if text is None else str(text)[:80]
            if shown not in vals:
                vals.append(shown)
        if len(vals) >= max_items:
            break
    return vals


def rt_column_audit(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        parsed = rt_num_series(s)
        non_missing = s.map(lambda x: rt_clean_text(x) is not None)
        audit[col] = {
            "missing": int((~non_missing).sum()),
            "bad_tokens": int(s.map(rt_is_bad_token).sum()),
            "numeric_parse_ok": int(parsed.notna().sum()),
            "non_missing": int(non_missing.sum()),
            "suspicious_examples": rt_suspicious_values(s),
        }
    return audit


def rt_question_operation(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("how many", "count", "number of")):
        return "count"
    if any(x in q for x in ("average", "mean")):
        return "mean"
    if "median" in q:
        return "median"
    if any(x in q for x in ("sum", "total")):
        return "sum"
    if any(x in q for x in ("highest", "largest", "maximum", "max ", "top")):
        return "rank"
    if any(x in q for x in ("lowest", "smallest", "minimum", "min ")):
        return "rank"
    if any(x in q for x in ("ratio", "rate", "percentage", "percent")):
        return "ratio"
    if any(x in q for x in ("difference", "subtract", "minus", "increase", "decrease")):
        return "arithmetic"
    return "lookup"


def rt_relevant_columns(question: str, df: pd.DataFrame, extra: list[str] | tuple[str, ...] = ()) -> list[str]:
    qkey = rt_norm_key(question) or ""
    q_tokens = set(qkey.split())
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        ckey = rt_norm_key(col) or ""
        c_tokens = set(ckey.split())
        score = 0
        if ckey and ckey in qkey:
            score += 4
        score += len(q_tokens & c_tokens) * 2
        compact = ckey.replace(" ", "")
        if compact and compact in qkey.replace(" ", ""):
            score += 2
        if score:
            scored.append((score, col))
    out = [c for _score, c in sorted(scored, key=lambda x: (-x[0], list(df.columns).index(x[1])))]
    for col in extra:
        if col in df.columns and col not in out:
            out.append(col)
    return out[:8]


def rt_add_candidate(
    candidates: list[dict[str, Any]],
    cid: str,
    answer: Any,
    evidence: str = "",
    strength: int = 0,
    patch_type: str = "no_op",
    risk: str = "",
) -> None:
    if answer is None:
        return
    if isinstance(answer, float) and (math.isnan(answer) or math.isinf(answer)):
        return
    candidates.append(
        {
            "id": cid,
            "answer": str(answer).strip(),
            "evidence": evidence.strip(),
            "strength": int(strength),
            "patch_type": patch_type,
            "risk": risk.strip(),
        }
    )


def rt_answer_key(answer: Any) -> str:
    text = "" if answer is None else str(answer).strip().strip("\"'`").lower()
    text = re.sub(r"\s+", " ", text)
    num = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if num and re.fullmatch(r"\s*-?[\d,]+(?:\.\d+)?\s*", text):
        val = float(num.group(0))
        if val.is_integer():
            return str(int(val))
        return f"{val:.12g}"
    return text


def rt_select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"id": "NO_CANDIDATE", "answer": "no_answer_extracted", "evidence": "", "strength": -1}
    by_key = Counter(rt_answer_key(c["answer"]) for c in candidates)

    def rank(c: dict[str, Any]) -> tuple[int, int, int, int]:
        cid = str(c.get("id", ""))
        strength = int(c.get("strength", 0))
        answer_support = by_key[rt_answer_key(c.get("answer", ""))]
        minimal = 1 if cid in {"C0_DIRECT", "C1_FORMAT"} else 0
        if strength < 3 and any(int(x.get("strength", 0)) >= 3 for x in candidates):
            base = strength
        elif strength <= 2:
            base = 1 if cid == "C1_FORMAT" else 0
        else:
            base = strength
        return (base, answer_support, minimal, -len(str(c.get("risk", ""))))

    selected = max(candidates, key=rank)
    if int(selected.get("strength", 0)) <= 2:
        for cid in ("C1_FORMAT", "C0_DIRECT"):
            for c in candidates:
                if c.get("id") == cid and by_key[rt_answer_key(c.get("answer", ""))] >= by_key[rt_answer_key(selected.get("answer", ""))]:
                    return c
    return selected


def rt_print_final(candidates: list[dict[str, Any]], selected: dict[str, Any] | None = None) -> None:
    selected = selected or rt_select_candidate(candidates)
    compact = []
    for c in candidates:
        compact.append(f"{c.get('id')}={c.get('answer')}[s={c.get('strength')}]")
    print("CANDIDATES:", "; ".join(compact))
    print("SELECTED_CANDIDATE:", selected.get("id", ""))
    print("SELECTED_REASON:", selected.get("evidence", ""))
    print("FINAL_ANSWER:", selected.get("answer", ""))


def rt_int_if_close(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value).strip()
    if math.isfinite(val) and abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:.12g}"


def rt_fixed(value: Any, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def rt_fixed_half_up(value: Any, digits: int) -> str:
    quant = Decimal("1") if digits <= 0 else Decimal("1." + ("0" * digits))
    return format(Decimal(str(float(value))).quantize(quant, rounding=ROUND_HALF_UP), f".{max(0, digits)}f")


def _rt_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _rt_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rt_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if math.isnan(val) or math.isinf(val) else val
    if isinstance(value, pd.Series):
        return _rt_json_safe(value.tolist())
    if not isinstance(value, (str, bytes, list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def rt_answer_contract(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("how many", "number of", "count")):
        return "count"
    if any(x in q for x in ("which ", "who ", "what geocode", "what course", "what team", "what state")):
        return "entity"
    if any(x in q for x in ("list", "which values", "what are")) and "," in q:
        return "list"
    return "scalar"


def _rt_threshold_after(question: str, phrase: str) -> float | None:
    q = question.lower().replace(",", "")
    idx = q.find(phrase)
    if idx < 0:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", q[idx:])
    return float(m.group(0)) if m else None


def _rt_simple_filters(question: str, df: pd.DataFrame) -> list[tuple[str, str, float | str]]:
    filters: list[tuple[str, str, float | str]] = []
    q = question.lower()
    for col in df.columns:
        c = str(col)
        c_low = c.lower()
        c_re = re.escape(c_low).replace("\\ ", r"\s+")
        for op_text, op in ((">=", ">="), ("<=", "<="), (">", ">"), ("<", "<"), ("=", "==")):
            m = re.search(rf"\b{c_re}\b\s*(?:is\s*)?{re.escape(op_text)}\s*(-?\d+(?:\.\d+)?)", q)
            if m:
                before = q[max(0, m.start() - 18) : m.start()]
                after = q[m.end() : min(len(q), m.end() + 45)]
                after_stripped = re.sub(r"^[\s,;:()\\-]+", "", after)
                if (
                    re.search(r"\b(?:not|never)\s+$", before)
                    or after_stripped.startswith("does not exclude")
                    or after_stripped.startswith("do not exclude")
                    or after_stripped.startswith("doesn't exclude")
                    or after_stripped.startswith("not exclude")
                    or after_stripped.startswith("does not mean")
                    or after_stripped.startswith("not mean")
                ):
                    continue
                filters.append((c, op, float(m.group(1))))
    for col in df.columns:
        c_low = str(col).lower()
        if "medal" in c_low and "gold" in q:
            filters.append((str(col), "==", "gold"))
        if "sex" == c_low and re.search(r"\bmale\b|\bmen\b", q):
            filters.append((str(col), "==", "m"))
    return filters


def _rt_apply_filters(df: pd.DataFrame, filters: list[tuple[str, str, float | str]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, op, value in filters:
        if col not in df.columns:
            continue
        if isinstance(value, str):
            keys = df[col].map(lambda x: (rt_norm_key(x) or ""))
            mask &= keys.eq(value)
            continue
        nums = rt_num_series(df[col])
        if op == ">=":
            mask &= nums.ge(value)
        elif op == "<=":
            mask &= nums.le(value)
        elif op == ">":
            mask &= nums.gt(value)
        elif op == "<":
            mask &= nums.lt(value)
        else:
            mask &= nums.eq(value)
    return mask


def _rt_add_auto_candidate(
    out: list[dict[str, Any]],
    cid: str,
    answer: Any,
    *,
    evidence_type: str,
    evidence: str,
    strength: int,
    contract: str,
    operation: str,
    risk: str = "",
) -> None:
    if answer is None:
        return
    if isinstance(answer, float) and (math.isnan(answer) or math.isinf(answer)):
        return
    out.append(
        {
            "candidate_id": cid,
            "answer": str(answer).strip(),
            "answer_contract": contract,
            "query_operation": operation,
            "evidence_type": evidence_type,
            "evidence_strength": int(strength),
            "evidence_summary": evidence,
            "risk": risk,
        }
    )


def _rt_fmt_answer(value: Any, question: str) -> str:
    q = question.lower()
    m = re.search(r"rounded to (?:the nearest )?(\d+) decimal", q)
    if m:
        return rt_fixed_half_up(value, int(m.group(1)))
    if "two decimal" in q or "2 decimal" in q:
        return rt_fixed_half_up(value, 2)
    if "four decimal" in q or "4 decimal" in q:
        return rt_fixed_half_up(value, 4)
    if "one decimal" in q or "1 decimal" in q:
        return rt_fixed_half_up(value, 1)
    return rt_int_if_close(value)


def _rt_find_col(df: pd.DataFrame, *needles: str) -> str | None:
    for col in df.columns:
        key = rt_norm_key(col) or ""
        if all(needle.lower() in key for needle in needles):
            return str(col)
    return None


def _rt_find_token_col(df: pd.DataFrame, *needles: str) -> str | None:
    wanted = {needle.lower() for needle in needles}
    for col in df.columns:
        tokens = set((rt_norm_key(col) or "").split())
        if wanted <= tokens:
            return str(col)
    return None


def _rt_find_col_excluding(df: pd.DataFrame, needles: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    matches: list[tuple[int, str]] = []
    for col in df.columns:
        key = rt_norm_key(col) or ""
        if all(n.lower() in key for n in needles) and not any(x.lower() in key for x in exclude):
            matches.append((len(key), str(col)))
    if not matches:
        return None
    return sorted(matches)[0][1]


def _rt_price_string_amount(series: pd.Series) -> pd.Series:
    return rt_num_series(series, mode="first")


def _rt_hard_outlier_count(series: pd.Series) -> int:
    nums = rt_num_series(series).dropna()
    if len(nums) < 8:
        return 0
    q1 = nums.quantile(0.25)
    q3 = nums.quantile(0.75)
    iqr = q3 - q1
    if not math.isfinite(float(iqr)) or iqr <= 0:
        return 0
    lo = q1 - 3 * iqr
    hi = q3 + 3 * iqr
    return int(((nums < lo) | (nums > hi)).sum())


def _rt_hard_outlier_total(*series_list: pd.Series) -> int:
    return sum(_rt_hard_outlier_count(series) for series in series_list)


def _rt_numeric_template_cols(df: pd.DataFrame) -> list[str]:
    blocked = {
        "id",
        "code",
        "zip",
        "postal",
        "phone",
        "date",
        "time",
        "year",
        "month",
        "day",
        "lat",
        "latitude",
        "lon",
        "long",
        "longitude",
        "event",
        "name",
        "title",
        "team",
        "country",
        "city",
        "state",
        "school",
        "type",
        "category",
        "class",
        "group",
        "description",
    }
    out: list[str] = []
    for col in df.columns:
        tokens = set((rt_norm_key(col) or "").split())
        if tokens & blocked:
            continue
        series = df[col]
        parsed = rt_num_series(series)
        non_missing = int(series.map(lambda x: rt_clean_text(x) is not None).sum())
        parseable = int(parsed.notna().sum())
        parse_fail = max(0, non_missing - parseable)
        if parse_fail > max(2, int(non_missing * 0.2)):
            continue
        if parseable >= max(3, int(len(df) * 0.4)):
            out.append(str(col))
    return out


def _rt_aggregate_series(series: pd.Series, question: str) -> Any:
    operation = rt_question_operation(question)
    clean = series.dropna()
    if clean.empty:
        return None
    if operation == "sum":
        return clean.sum()
    if operation == "median":
        return clean.median()
    return clean.mean()


def _rt_query_target_col(question: str, df: pd.DataFrame, candidates: list[str]) -> str | None:
    relevant = rt_relevant_columns(question, df)
    for col in relevant:
        if col in candidates:
            return str(col)
    qkey = rt_norm_key(question) or ""
    scored: list[tuple[int, str]] = []
    for col in candidates:
        ckey = rt_norm_key(col) or ""
        score = len(set(qkey.split()) & set(ckey.split()))
        if "total" in ckey and "total" in qkey:
            score += 2
        if "amount" in ckey and "amount" in qkey:
            score += 2
        if "difference" in qkey and ("diff" in ckey or "difference" in ckey):
            score += 2
        if score:
            scored.append((score, str(col)))
    if not scored:
        return None
    return sorted(scored, key=lambda x: (-x[0], list(df.columns).index(x[1])))[0][1]


def _rt_numeric_query_targets(question: str, df: pd.DataFrame, limit: int = 3) -> list[str]:
    numeric_cols = _rt_numeric_template_cols(df)
    relevant = rt_relevant_columns(question, df)
    ordered: list[str] = []
    for col in relevant:
        if col in numeric_cols and col not in ordered:
            ordered.append(str(col))
    if not ordered and rt_question_operation(question) in {"mean", "median", "sum", "rank", "ratio"}:
        target = _rt_query_target_col(question, df, numeric_cols)
        if target:
            ordered.append(target)
    return ordered[:limit]


def _rt_value_quality(series: pd.Series) -> dict[str, int]:
    text_numeric = 0
    parse_fail = 0
    non_missing = 0
    missing = 0
    for value in series.tolist():
        text = rt_clean_text(value)
        if text is None:
            missing += 1
            continue
        non_missing += 1
        parsed = rt_num(value)
        if pd.isna(parsed):
            parse_fail += 1
            continue
        if re.search(r"[A-Za-z%$,]", text):
            text_numeric += 1
    return {
        "non_missing": non_missing,
        "missing": missing,
        "bad_tokens": int(series.map(rt_is_bad_token).sum()),
        "text_numeric": text_numeric,
        "parse_fail": parse_fail,
    }


def _rt_component_cols(df: pd.DataFrame, target_col: str, denom_col: str | None = None) -> list[str]:
    component_tokens = {
        "fare",
        "fee",
        "extra",
        "tax",
        "surcharge",
        "toll",
        "tolls",
        "tip",
        "charge",
        "cost",
        "price",
        "subtotal",
        "component",
        "part",
    }
    blocked = {target_col}
    if denom_col:
        blocked.add(denom_col)
    cols: list[str] = []
    for col in _rt_numeric_template_cols(df):
        if col in blocked:
            continue
        tokens = set((rt_norm_key(col) or "").split())
        if tokens & component_tokens:
            nums = rt_num_series(df[col])
            if nums.dropna().empty:
                continue
            cols.append(col)
    return cols


def _rt_year_unit_series(series: pd.Series) -> pd.Series:
    values: list[float] = []
    for value in series.tolist():
        text = rt_clean_text(value)
        num = rt_num(value)
        if text and re.search(r"\bmonths?\b", text.lower()) and not pd.isna(num):
            num = num / 12.0
        values.append(float(num) if not pd.isna(num) else np.nan)
    return pd.Series(values, index=series.index, dtype=float)


def _rt_auto_employee_tenure(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("employees" in q and "company" in q and "age" in q):
        return
    tenure_col = _rt_find_col(df, "years", "company")
    age_col = _rt_find_col(df, "age")
    total_work_col = _rt_find_col(df, "total", "working", "years")
    role_col = _rt_find_col(df, "years", "current", "role")
    if not (tenure_col and age_col):
        return
    tenure_threshold = _rt_threshold_after(question, "company for") or _rt_threshold_after(question, "company") or 5
    age_threshold = _rt_threshold_after(question, "age of") or _rt_threshold_after(question, "age") or 35
    tenure = _rt_year_unit_series(df[tenure_col])
    age = rt_num_series(df[age_col])
    direct = tenure.ge(tenure_threshold) & age.ge(age_threshold)
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_EMPLOYEE_TENURE",
        int(direct.fillna(False).sum()),
        evidence_type="employee_direct",
        evidence=f"unit-aware count where {tenure_col}>={tenure_threshold:g} years and {age_col}>={age_threshold:g}",
        strength=1,
        contract="count",
        operation="filter_count",
    )
    valid = direct.fillna(False).copy()
    checks: list[str] = []
    if total_work_col:
        total_work = _rt_year_unit_series(df[total_work_col])
        ok = tenure.le(total_work + 1e-9)
        valid &= ok.fillna(False)
        checks.append(f"{tenure_col}<={total_work_col}")
    if role_col:
        role_years = _rt_year_unit_series(df[role_col])
        ok = role_years.le(tenure + 1e-9)
        valid &= ok.fillna(False)
        checks.append(f"{role_col}<={tenure_col}")
    if not checks:
        return
    excluded = int((direct.fillna(False) & (~valid)).sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_EMPLOYEE_TENURE_CONSISTENCY",
        int(valid.sum()),
        evidence_type="employee_consistency",
        evidence=f"excluded {excluded} rows violating {' and '.join(checks)} after year/month normalization",
        strength=4 if excluded else 1,
        contract="count",
        operation="filter_count",
    )


def _rt_auto_amount_string(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not any(x in q for x in ("price", "discount", "rupee", "amount")):
        return
    discount_amount = _rt_find_col(df, "discount", "amount")
    discount_string = _rt_find_col(df, "discount", "price", "string") or _rt_find_col(df, "discount", "string")
    detail_amount = _rt_find_col(df, "price", "detail", "amount") or _rt_find_col(df, "listed", "amount")
    detail_string = _rt_find_col(df, "price", "detail", "string") or _rt_find_col(df, "price", "string")
    if not (discount_amount and detail_amount):
        return
    disc = rt_num_series(df[discount_amount])
    price = rt_num_series(df[detail_amount])
    if discount_string and detail_string:
        disc_s = _rt_price_string_amount(df[discount_string])
        price_s = _rt_price_string_amount(df[detail_string])
    else:
        disc_s, price_s = disc, price
    hard_outliers = _rt_hard_outlier_total(df[discount_amount], df[detail_amount])
    if discount_string:
        hard_outliers += _rt_hard_outlier_count(df[discount_string])
    if detail_string:
        hard_outliers += _rt_hard_outlier_count(df[detail_string])
    listed_threshold = _rt_threshold_after(question, "at least") or _rt_threshold_after(question, ">=") or 0
    discount_threshold = _rt_threshold_after(question, "more than") or 0
    pct = (price - disc) / price * 100
    direct = (price >= listed_threshold) & (pct > discount_threshold)
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_AMOUNT",
        int(direct.fillna(False).sum()),
        evidence_type="amount_numeric",
        evidence=f"numeric amount columns used; listed>={listed_threshold:g}, discount>{discount_threshold:g}%; hard_outliers={hard_outliers}",
        strength=1,
        contract="count",
        operation="filter_count",
    )
    mismatch = ((disc - disc_s).abs() > 1e-6) | ((price - price_s).abs() > 1e-6)
    pct_s = (price_s - disc_s) / price_s * 100
    string_consistent = (price_s >= listed_threshold) & (pct_s > discount_threshold) & (~mismatch.fillna(False))
    mismatch_count = int(mismatch.fillna(False).sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_AMOUNT_STRING_CONSISTENT",
        int(string_consistent.fillna(False).sum()),
        evidence_type="amount_string",
        evidence=f"paired amount/string check; excluded {mismatch_count} rows with amount-string disagreement; hard_outliers={hard_outliers}",
        strength=4 if mismatch_count else 1,
        contract="count",
        operation="filter_count",
    )


def _rt_auto_rating_denominator(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if "rating" not in q:
        return
    rating_col = _rt_find_col(df, "rating")
    count_col = _rt_find_col(df, "reviews") or _rt_find_col(df, "votes")
    if not rating_col:
        return
    threshold = _rt_threshold_after(question, "greater than") or _rt_threshold_after(question, ">") or 0
    rating = rt_num_series(df[rating_col])
    direct = rating.gt(threshold)
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_RATING",
        int(direct.fillna(False).sum()),
        evidence_type="rating_numeric",
        evidence=f"direct rating threshold {rating_col}>{threshold:g}",
        strength=1,
        contract="count",
        operation="filter_count",
    )
    if count_col:
        counts = rt_num_series(df[count_col])
        invalid = counts.eq(0) & rating.gt(0)
        valid = rating.gt(threshold) & (~invalid.fillna(False))
        _rt_add_auto_candidate(
            candidates,
            "C5_RATING_DENOMINATOR",
            int(valid.fillna(False).sum()),
            evidence_type="rating_denominator",
            evidence=f"excluded {int(invalid.fillna(False).sum())} rows where {count_col}=0 but rating is positive",
            strength=4 if int(invalid.fillna(False).sum()) else 1,
            contract="count",
            operation="filter_count",
        )


def _rt_auto_bmi(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if "bmi" not in q:
        return
    bmi_col = _rt_find_col(df, "bmi")
    height_col = _rt_find_col(df, "height")
    weight_col = _rt_find_col(df, "weight")
    if not bmi_col:
        return
    filters = _rt_simple_filters(question, df)
    mask = _rt_apply_filters(df, filters)
    bmi = rt_num_series(df[bmi_col])
    if "average" in q or "mean" in q:
        _rt_add_auto_candidate(
            candidates,
            "AUTO_DIRECT_BMI",
            _rt_fmt_answer(bmi[mask].mean(), question),
            evidence_type="direct_bmi",
            evidence="direct mean over filtered BMI after numeric parsing",
            strength=1,
            contract="scalar",
            operation="mean",
        )
    if not (height_col and weight_col):
        return
    height = rt_num_series(df[height_col])
    weight = rt_num_series(df[weight_col])
    source_hard_outliers = _rt_hard_outlier_total(df[height_col], df[weight_col])
    target_hard_outliers = _rt_hard_outlier_count(df[bmi_col])
    h_med = height.dropna().median()
    if pd.isna(h_med) or h_med <= 0:
        return
    h_m = height / 100 if h_med > 20 else height
    derived = weight / (h_m**2)
    diff = (derived - bmi).abs()
    consistent = diff.lt(1.0) | bmi.isna() | derived.isna()
    replace = ((~consistent) | (bmi.isna() & derived.notna())) & mask
    repaired = bmi.copy()
    repaired[replace] = derived[replace]
    if "average" in q or "mean" in q:
        _rt_add_auto_candidate(
            candidates,
            "C5_BMI_FROM_HEIGHT_WEIGHT",
            _rt_fmt_answer(repaired[mask].mean(), question),
            evidence_type="body_index",
            evidence=(
                f"recomputed BMI from height/weight for {int(replace.sum())} filtered conflicting rows; "
                f"source_hard_outliers={source_hard_outliers}; target_hard_outliers={target_hard_outliers}"
            ),
            strength=4 if int(replace.sum()) else 1,
            contract="scalar",
            operation="mean",
        )


def _rt_auto_stable_age(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not re.search(r"\bage\b", q) or not any(x in q for x in ("average", "mean")):
        return
    age_col = _rt_find_token_col(df, "age")
    if not age_col:
        return
    filters = _rt_simple_filters(question, df)
    mask = _rt_apply_filters(df, filters)
    age = rt_num_series(df[age_col])
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_AGE",
        _rt_fmt_answer(age[mask].mean(), question),
        evidence_type="direct_age",
        evidence="direct mean over filtered age after numeric parsing",
        strength=1,
        contract="scalar",
        operation="mean",
    )
    id_col = _rt_find_col(df, "athlete", "id") or _rt_find_col(df, "person", "id") or _rt_find_col(df, "name")
    context_cols = [c for c in (_rt_find_col(df, "games"), _rt_find_col(df, "event"), _rt_find_col(df, "year")) if c]
    if not id_col:
        return
    group_cols = [id_col] + context_cols[:1]
    repaired = age.copy()
    changes = 0
    tmp = df[group_cols].copy()
    tmp["_age"] = age
    for _, group in tmp.groupby(group_cols, dropna=False):
        vals = group["_age"].dropna()
        if len(vals) < 2:
            continue
        rounded = vals.round().astype(int)
        mode = rounded.mode()
        if mode.empty:
            continue
        canonical = float(mode.iloc[0])
        idxs = group.index[(group["_age"] - canonical).abs() > 0.01]
        if len(idxs) and len(rounded[rounded == int(canonical)]) >= 2:
            repaired.loc[idxs] = canonical
            changes += len(idxs)
    _rt_add_auto_candidate(
        candidates,
        "C5_STABLE_ENTITY_AGE",
        _rt_fmt_answer(repaired[mask].mean(), question),
        evidence_type="stable_entity",
        evidence=f"same entity/context age canonicalization changed {changes} rows",
        strength=4 if changes else 1,
        contract="scalar",
        operation="mean",
    )


def _rt_auto_primitive_numeric_parse(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    operation = rt_question_operation(question)
    if operation not in {"mean", "median", "sum"}:
        return
    mask = _rt_apply_filters(df, _rt_simple_filters(question, df))
    for target_col in _rt_numeric_query_targets(question, df, limit=2):
        raw = df[target_col]
        parsed = rt_num_series(raw)
        value = _rt_aggregate_series(parsed[mask], question)
        if value is None:
            continue
        quality = _rt_value_quality(raw[mask])
        evidence_count = quality["text_numeric"] + quality["parse_fail"] + quality["bad_tokens"]
        _rt_add_auto_candidate(
            candidates,
            f"C5_PRIMITIVE_NUMERIC_PARSE_{len(candidates)}",
            _rt_fmt_answer(value, question),
            evidence_type="primitive_numeric_parse",
            evidence=(
                f"parsed query-relevant numeric column {target_col}; "
                f"text_numeric={quality['text_numeric']}; bad_tokens={quality['bad_tokens']}; "
                f"missing={quality['missing']}; parse_fail={quality['parse_fail']}"
            ),
            strength=4 if evidence_count >= 2 else 1,
            contract="scalar",
            operation=operation,
        )


def _rt_auto_primitive_bad_value_exclusion(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    operation = rt_question_operation(question)
    if operation not in {"mean", "median", "sum", "count"}:
        return
    filters = _rt_simple_filters(question, df)
    mask = _rt_apply_filters(df, filters)
    targets = _rt_numeric_query_targets(question, df, limit=2)
    if operation == "count" and filters:
        bad_mask = pd.Series(False, index=df.index)
        for col, _op, _value in filters:
            if col in df.columns:
                bad_mask |= df[col].map(rt_is_bad_token)
        candidate_mask = mask & (~bad_mask.fillna(False))
        bad_in_scope = int((mask & bad_mask.fillna(False)).sum())
        _rt_add_auto_candidate(
            candidates,
            "C5_PRIMITIVE_BAD_VALUE_FILTER_COUNT",
            int(candidate_mask.fillna(False).sum()),
            evidence_type="primitive_bad_value",
            evidence=f"excluded bad-token rows only in query filter columns; bad_in_scope={bad_in_scope}",
            strength=4 if bad_in_scope else 1,
            contract="count",
            operation="filter_count",
        )
        return
    for target_col in targets:
        raw = df[target_col]
        parsed = rt_num_series(raw)
        bad = raw.map(rt_is_bad_token)
        scoped_bad = int((mask & bad.fillna(False)).sum())
        valid = mask & (~bad.fillna(False))
        value = _rt_aggregate_series(parsed[valid], question)
        if value is None:
            continue
        _rt_add_auto_candidate(
            candidates,
            f"C5_PRIMITIVE_BAD_VALUE_EXCLUDE_{len(candidates)}",
            _rt_fmt_answer(value, question),
            evidence_type="primitive_bad_value",
            evidence=f"excluded explicit bad tokens from query-relevant column {target_col}; bad_in_scope={scoped_bad}",
            strength=4 if scoped_bad else 1,
            contract="scalar",
            operation=operation,
        )


def _rt_auto_primitive_outlier_filter(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    operation = rt_question_operation(question)
    if operation != "mean":
        return
    mask = _rt_apply_filters(df, _rt_simple_filters(question, df))
    for target_col in _rt_numeric_query_targets(question, df, limit=2):
        parsed = rt_num_series(df[target_col])
        scoped = parsed[mask].dropna()
        if len(scoped) < 8:
            continue
        q1 = scoped.quantile(0.25)
        q3 = scoped.quantile(0.75)
        iqr = q3 - q1
        if not math.isfinite(float(iqr)) or iqr <= 0:
            continue
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        keep = parsed.between(lo, hi) | parsed.isna()
        outliers = int((mask & (~keep.fillna(False))).sum())
        if not outliers:
            continue
        trimmed = parsed[mask & keep.fillna(False)].dropna()
        if len(trimmed) < 3:
            continue
        _rt_add_auto_candidate(
            candidates,
            f"C5_PRIMITIVE_IQR_OUTLIER_FILTER_{len(candidates)}",
            _rt_fmt_answer(trimmed.mean(), question),
            evidence_type="primitive_outlier",
            evidence=(
                f"IQR-filtered query-relevant column {target_col}; "
                f"hard_outliers={outliers}; n_kept={len(trimmed)}; bounds=[{lo:.6g},{hi:.6g}]"
            ),
            strength=4,
            contract="scalar",
            operation="mean",
        )


def _rt_auto_generic_difference_formula(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not any(token in q for token in ("difference", "diff", "minus", "gap")):
        return
    numeric_cols = _rt_numeric_template_cols(df)
    direct_candidates = [
        col
        for col in numeric_cols
        if any(token in (rt_norm_key(col) or "") for token in ("diff", "difference", "gap"))
    ]
    direct_col = _rt_query_target_col(question, df, direct_candidates) if direct_candidates else None
    if not direct_col:
        return
    source_cols = [col for col in numeric_cols if col != direct_col]
    if len(source_cols) < 2:
        return
    direct = rt_num_series(df[direct_col])
    best: tuple[int, int, str, str, str, pd.Series] | None = None
    for left_col in source_cols:
        left = rt_num_series(df[left_col])
        for right_col in source_cols:
            if right_col == left_col:
                continue
            right = rt_num_series(df[right_col])
            formulas = (
                ("abs_diff", (left - right).abs()),
                ("signed_diff", left - right),
            )
            for mode, derived in formulas:
                comparable = direct.notna() & derived.notna()
                if int(comparable.sum()) < max(5, int(len(df) * 0.4)):
                    continue
                mismatch = ((direct - derived).abs() > 0.01) & comparable
                match = int(comparable.sum() - mismatch.sum())
                score = match - int(mismatch.sum())
                if best is None or score > best[0]:
                    best = (score, int(mismatch.sum()), mode, left_col, right_col, derived)
    if best is None:
        return
    _score, mismatches, mode, left_col, right_col, derived = best
    hard_outliers = _rt_hard_outlier_total(df[direct_col], df[left_col], df[right_col])
    mask = _rt_apply_filters(df, _rt_simple_filters(question, df))
    direct_answer = _rt_aggregate_series(direct[mask], question)
    if direct_answer is not None:
        _rt_add_auto_candidate(
            candidates,
            "AUTO_DIRECT_DIFFERENCE_COLUMN",
            _rt_fmt_answer(direct_answer, question),
            evidence_type="direct_derived_column",
            evidence=f"used direct derived column {direct_col}; hard_outliers={hard_outliers}",
            strength=1,
            contract="scalar",
            operation=rt_question_operation(question),
        )
    repaired_answer = _rt_aggregate_series(derived[mask], question)
    if repaired_answer is None:
        return
    _rt_add_auto_candidate(
        candidates,
        "C5_GENERIC_DIFFERENCE_FORMULA",
        _rt_fmt_answer(repaired_answer, question),
        evidence_type="generic_difference",
        evidence=(
            f"recomputed {direct_col} as {mode}({left_col},{right_col}) in query subset; "
            f"mismatches={mismatches}; hard_outliers={hard_outliers}"
        ),
        strength=4 if mismatches >= 3 and hard_outliers <= 5 else 1,
        contract="scalar",
        operation=rt_question_operation(question),
    )


def _rt_auto_generic_component_sum(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not any(token in q for token in ("total", "amount", "sum", "charge", "cost")):
        return
    numeric_cols = _rt_numeric_template_cols(df)
    target_candidates = [
        col
        for col in numeric_cols
        if any(token in (rt_norm_key(col) or "") for token in ("total", "amount", "charge", "cost"))
    ]
    target_col = _rt_query_target_col(question, df, target_candidates) if target_candidates else None
    if not target_col:
        return
    denom_col = None
    if " per " in f" {q} " or rt_question_operation(question) == "ratio":
        denom_candidates = [
            col
            for col in numeric_cols
            if col != target_col
            and any(token in (rt_norm_key(col) or "") for token in ("distance", "mile", "order", "unit", "quantity"))
        ]
        denom_col = _rt_query_target_col(question, df, denom_candidates) if denom_candidates else None
    comp_cols = _rt_component_cols(df, target_col, denom_col)
    if len(comp_cols) < 2:
        return
    target = rt_num_series(df[target_col])
    comp_sum = sum(rt_num_series(df[col]).fillna(0) for col in comp_cols)
    comparable = target.notna() & comp_sum.notna()
    if int(comparable.sum()) < max(5, int(len(df) * 0.4)):
        return
    mismatch = ((target - comp_sum).abs() > 0.01) & comparable
    mismatches = int(mismatch.sum())
    hard_outliers = _rt_hard_outlier_total(df[target_col], *(df[col] for col in comp_cols))
    if denom_col:
        denom = rt_num_series(df[denom_col])
        hard_outliers += _rt_hard_outlier_count(df[denom_col])
        valid = denom.gt(0)
        direct_value = _rt_aggregate_series((target / denom)[valid], question)
        repaired_value = _rt_aggregate_series((comp_sum / denom)[valid], question)
        operation = "ratio"
    else:
        direct_value = _rt_aggregate_series(target, question)
        repaired_value = _rt_aggregate_series(comp_sum, question)
        operation = rt_question_operation(question)
    if direct_value is not None:
        _rt_add_auto_candidate(
            candidates,
            "AUTO_DIRECT_COMPONENT_TARGET",
            _rt_fmt_answer(direct_value, question),
            evidence_type="direct_component_target",
            evidence=f"used direct target column {target_col}; hard_outliers={hard_outliers}",
            strength=1,
            contract="scalar",
            operation=operation,
        )
    if repaired_value is None:
        return
    _rt_add_auto_candidate(
        candidates,
        "C5_GENERIC_COMPONENT_SUM",
        _rt_fmt_answer(repaired_value, question),
        evidence_type="generic_component_sum",
        evidence=(
            f"recomputed {target_col} from {len(comp_cols)} component columns; "
            f"mismatches={mismatches}; hard_outliers={hard_outliers}"
        ),
        strength=4 if mismatches >= 5 and hard_outliers <= 30 else 1,
        contract="scalar",
        operation=operation,
    )


def _rt_auto_generic_bound_consistency(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if rt_answer_contract(question) != "count":
        return
    if not any(token in q for token in ("minimum", "maximum", "min ", "max ", "up to", "at least")):
        return
    min_cols = [col for col in df.columns if {"min"} & set((rt_norm_key(col) or "").split()) or "minimum" in (rt_norm_key(col) or "")]
    max_cols = [col for col in df.columns if {"max"} & set((rt_norm_key(col) or "").split()) or "maximum" in (rt_norm_key(col) or "")]
    if not min_cols or not max_cols:
        return
    min_col = str(min_cols[0])
    max_col = str(max_cols[0])
    min_threshold = (
        _rt_threshold_after(question, "minimum")
        or _rt_threshold_after(question, "min")
        or _rt_threshold_after(question, "at least")
    )
    max_threshold = _rt_threshold_after(question, "up to") or _rt_threshold_after(question, "maximum") or _rt_threshold_after(question, "max")
    if min_threshold is None or max_threshold is None:
        return
    min_values = rt_num_series(df[min_col])
    max_values = rt_num_series(df[max_col])
    direct = min_values.ge(min_threshold) & max_values.ge(max_threshold)
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_BOUND_COUNT",
        int(direct.fillna(False).sum()),
        evidence_type="direct_bounds",
        evidence=f"direct count where {min_col}>={min_threshold:g} and {max_col}>={max_threshold:g}",
        strength=1,
        contract="count",
        operation="filter_count",
    )
    valid_bounds = min_values.le(max_values).fillna(False) & min_values.ge(0).fillna(False) & max_values.ge(0).fillna(False)
    invalid_direct = int((direct.fillna(False) & (~valid_bounds)).sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_GENERIC_BOUND_CONSISTENCY",
        int((direct.fillna(False) & valid_bounds).sum()),
        evidence_type="generic_bounds",
        evidence=f"excluded {invalid_direct} rows violating min<=max and nonnegative bounds",
        strength=4 if invalid_direct else 1,
        contract="count",
        operation="filter_count",
    )


def _rt_auto_generic_unit_conversion(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not any(token in q for token in ("hourly", "per hour", "annual", "yearly", "salary", "wage")):
        return
    hourly_col = _rt_find_col(df, "hourly") or _rt_find_col(df, "hour")
    annual_col = _rt_find_col(df, "annual") or _rt_find_col(df, "yearly")
    if not (hourly_col and annual_col):
        return
    time_col = _rt_find_col(df, "year") or _rt_find_col(df, "date")
    hourly = rt_num_series(df[hourly_col])
    annual_hourly = rt_num_series(df[annual_col]) / 2080.0
    repair = (((hourly - annual_hourly).abs() > 0.05) | hourly.isna()) & annual_hourly.notna()
    repaired = hourly.copy()
    repaired[repair] = annual_hourly[repair]
    target_idx = df.index
    latest_groups = 0
    if "latest" in q and time_col:
        categorical_cols = [
            str(col)
            for col in df.columns
            if col not in {hourly_col, annual_col, time_col}
            and rt_num_series(df[col]).notna().sum() < max(3, int(len(df) * 0.25))
        ]
        group_col = _rt_query_target_col(question, df, categorical_cols) or (categorical_cols[0] if categorical_cols else None)
        if group_col:
            years = rt_num_series(df[time_col])
            tmp = pd.DataFrame({"group": df[group_col].astype(str), "year": years}, index=df.index)
            target_idx = tmp.dropna(subset=["year"]).sort_values("year").groupby("group", dropna=False).tail(1).index
            latest_groups = len(target_idx)
    direct_value = _rt_aggregate_series(hourly.loc[target_idx], question)
    repaired_value = _rt_aggregate_series(repaired.loc[target_idx], question)
    if direct_value is not None:
        _rt_add_auto_candidate(
            candidates,
            "AUTO_DIRECT_UNIT_VALUE",
            _rt_fmt_answer(direct_value, question),
            evidence_type="direct_unit_value",
            evidence=f"direct aggregate over {hourly_col}; latest_groups={latest_groups}",
            strength=1,
            contract="scalar",
            operation=rt_question_operation(question),
        )
    if repaired_value is None:
        return
    converted_rows = int(repair.loc[target_idx].sum()) if len(target_idx) else int(repair.sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_GENERIC_UNIT_CONVERSION",
        _rt_fmt_answer(repaired_value, question),
        evidence_type="generic_unit_conversion",
        evidence=f"converted {annual_col}/2080 into {hourly_col}; converted_rows={converted_rows}; latest_groups={latest_groups}",
        strength=4 if converted_rows else 1,
        contract="scalar",
        operation=rt_question_operation(question),
    )


def _rt_auto_expected_goal_diff(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if "expected goals" not in q or "scored goals" not in q:
        return
    expected_col = _rt_find_col_excluding(df, ("expected", "goals"), ("diff", "actual", "scored"))
    scored_col = _rt_find_col_excluding(df, ("scored",), ("diff", "expected", "actual"))
    position_col = _rt_find_col(df, "position")
    direct_col = _rt_find_col(df, "expected", "actual", "scored", "diff")
    if not (expected_col and scored_col):
        return
    mask = pd.Series(True, index=df.index)
    top_match = re.search(r"top\s+(\d+)", q)
    if top_match and position_col:
        mask &= rt_num_series(df[position_col]).le(float(top_match.group(1)))
    expected = rt_num_series(df[expected_col])
    scored = rt_num_series(df[scored_col])
    derived = (expected - scored).abs()
    mismatch_count = 0
    hard_outliers = _rt_hard_outlier_total(df[expected_col], df[scored_col])
    if direct_col:
        direct = rt_num_series(df[direct_col]).abs()
        mismatch_count = int(((derived - direct).abs() > 0.01).fillna(False).sum())
        hard_outliers += _rt_hard_outlier_count(df[direct_col])
        _rt_add_auto_candidate(
            candidates,
            "AUTO_DIRECT_DIFF_COLUMN",
            _rt_fmt_answer(direct[mask].mean(), question),
            evidence_type="direct_derived_column",
            evidence=f"used direct derived column {direct_col}; hard_outliers={hard_outliers}",
            strength=1,
            contract="scalar",
            operation="mean",
        )
    _rt_add_auto_candidate(
        candidates,
        "C5_RECOMPUTE_EXPECTED_GOAL_DIFF",
        _rt_fmt_answer(derived[mask].mean(), question),
        evidence_type="parts_total",
        evidence=f"recomputed abs({expected_col}-{scored_col}) in query subset; mismatches={mismatch_count}; hard_outliers={hard_outliers}",
        strength=4 if mismatch_count else 1,
        contract="scalar",
        operation="mean",
    )


def _rt_auto_total_from_components(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if "total amount" not in q or "per mile" not in q:
        return
    total_col = _rt_find_col(df, "total", "amount")
    dist_col = _rt_find_col(df, "trip", "distance")
    if not (total_col and dist_col):
        return
    total = rt_num_series(df[total_col])
    dist = rt_num_series(df[dist_col])
    valid = dist.gt(0)
    direct_ratio = (total / dist)[valid]
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_TOTAL_PER_MILE",
        _rt_fmt_answer(direct_ratio.mean(), question),
        evidence_type="direct_ratio",
        evidence="row-wise total_amount / trip_distance using direct total column",
        strength=1,
        contract="scalar",
        operation="ratio",
    )
    comp_needles = ("fare", "extra", "tax", "surcharge", "tolls", "tip", "congestion")
    comp_cols = [c for c in df.columns if any(n in (rt_norm_key(c) or "") for n in comp_needles) and c != total_col]
    if len(comp_cols) < 2:
        return
    hard_outliers = _rt_hard_outlier_total(df[total_col], df[dist_col], *(df[c] for c in comp_cols))
    comp_sum = sum(rt_num_series(df[c]).fillna(0) for c in comp_cols)
    mismatch = (comp_sum - total).abs() > 0.01
    ratio = (comp_sum / dist)[valid]
    _rt_add_auto_candidate(
        candidates,
        "C5_TOTAL_FROM_COMPONENTS",
        _rt_fmt_answer(ratio.mean(), question),
        evidence_type="parts_total",
        evidence=f"recomputed total from {len(comp_cols)} component columns; mismatches={int(mismatch.sum())}; hard_outliers={hard_outliers}",
        strength=4 if int(mismatch.sum()) else 1,
        contract="scalar",
        operation="ratio",
    )


def _rt_auto_sales_from_components(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("average sales per order" in q or ("sales" in q and "per order" in q)):
        return
    sales_col = _rt_find_col(df, "sales")
    qty_col = _rt_find_col(df, "quantity", "ordered")
    price_col = _rt_find_col(df, "price", "each")
    if not (sales_col and qty_col and price_col):
        return
    sales = rt_num_series(df[sales_col])
    qty = rt_num_series(df[qty_col])
    price = rt_num_series(df[price_col])

    def banker(value: Any) -> str:
        try:
            return str(int(round(float(value))))
        except Exception:
            return ""

    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_SALES_MEAN",
        banker(sales.dropna().mean()),
        evidence_type="sales_direct",
        evidence=f"direct mean over parsed {sales_col} with bankers rounding; missing_sales={int(sales.isna().sum())}",
        strength=1,
        contract="scalar",
        operation="mean",
    )
    qty_nonmissing = qty.dropna()
    quantity_fractional = int(((qty_nonmissing - qty_nonmissing.round()).abs() > 1e-6).sum())
    price_date_like = int(df[price_col].map(lambda x: bool(re.search(r"\d+/\d+/\d+|:", str(x)))).sum())
    component_quality_ok = quantity_fractional <= 2 and price_date_like == 0
    derived = qty * price
    mismatch = ((sales - derived).abs() > 0.5) & sales.notna() & derived.notna()
    missing_sales = sales.isna() & derived.notna()
    repaired = sales.copy()
    repaired[mismatch | missing_sales] = derived[mismatch | missing_sales]
    evidence_count = int(mismatch.sum() + missing_sales.sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_SALES_FROM_QUANTITY_PRICE",
        banker(repaired.dropna().mean()),
        evidence_type="sales_formula",
        evidence=(
            f"recomputed {sales_col} from {qty_col}*{price_col}; mismatches={int(mismatch.sum())}; "
            f"missing_sales={int(missing_sales.sum())}; quantity_fractional={quantity_fractional}; "
            f"price_date_like={price_date_like}"
        ),
        strength=4 if evidence_count and component_quality_ok else 1,
        contract="scalar",
        operation="mean",
        risk="" if component_quality_ok else "component_columns_do_not_look_like_quantity_and_price",
    )


def _rt_auto_boardgame_player_bounds(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("board game" in q and "player" in q):
        return
    min_col = _rt_find_col(df, "minplayers") or _rt_find_col(df, "min", "players")
    max_col = _rt_find_col(df, "maxplayers") or _rt_find_col(df, "max", "players")
    if not (min_col and max_col):
        return
    min_threshold = _rt_threshold_after(question, "minimum") or _rt_threshold_after(question, "min") or 2
    max_threshold = _rt_threshold_after(question, "up to") or _rt_threshold_after(question, "max") or 5
    min_players = rt_num_series(df[min_col])
    max_players = rt_num_series(df[max_col])
    direct = min_players.ge(min_threshold) & max_players.ge(max_threshold)
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_PLAYER_BOUNDS",
        int(direct.fillna(False).sum()),
        evidence_type="player_bounds_direct",
        evidence=f"direct count where {min_col}>={min_threshold:g} and {max_col}>={max_threshold:g}",
        strength=1,
        contract="count",
        operation="filter_count",
    )
    valid_bounds = min_players.le(max_players).fillna(False) & min_players.between(1, 50).fillna(False) & max_players.between(1, 50).fillna(False)
    invalid_direct = int((direct.fillna(False) & (~valid_bounds)).sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_PLAYER_BOUNDS_VALID",
        int((direct.fillna(False) & valid_bounds).sum()),
        evidence_type="player_bounds",
        evidence=f"excluded {invalid_direct} rows violating min<=max and 1<=player bounds<=50",
        strength=4 if invalid_direct else 1,
        contract="count",
        operation="filter_count",
    )


def _rt_auto_nurse_salary_latest(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("nurse" in q and "median hourly salary" in q and "latest" in q):
        return
    state_col = _rt_find_col(df, "state")
    year_col = _rt_find_col(df, "year")
    hourly_col = _rt_find_col(df, "hourly", "wage", "median") or _rt_find_col(df, "hourly", "median")
    annual_col = _rt_find_col(df, "annual", "salary", "median") or _rt_find_col(df, "annual", "median")
    if not (state_col and year_col and hourly_col):
        return
    year = rt_num_series(df[year_col])
    hourly = rt_num_series(df[hourly_col])
    tmp = pd.DataFrame({"state": df[state_col].astype(str), "year": year, "hourly": hourly}, index=df.index)
    latest_idx = tmp.dropna(subset=["year"]).sort_values("year").groupby("state", dropna=False).tail(1).index
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_LATEST_NURSE_HOURLY",
        _rt_fmt_answer(tmp.loc[latest_idx, "hourly"].dropna().mean(), question),
        evidence_type="salary_direct",
        evidence=f"latest row per {state_col}; missing_latest_hourly={int(tmp.loc[latest_idx, 'hourly'].isna().sum())}",
        strength=1,
        contract="scalar",
        operation="mean",
    )
    if not annual_col:
        return
    annual_hourly = rt_num_series(df[annual_col]) / 2080.0
    diff = (hourly - annual_hourly).abs()
    repair = ((diff > 0.05) | hourly.isna()) & annual_hourly.notna()
    repaired = hourly.copy()
    repaired[repair] = annual_hourly[repair]
    repair_latest = int(repair.loc[latest_idx].sum())
    _rt_add_auto_candidate(
        candidates,
        "C5_NURSE_HOURLY_FROM_ANNUAL",
        _rt_fmt_answer(repaired.loc[latest_idx].dropna().mean(), question),
        evidence_type="salary_annual_consistency",
        evidence=f"recomputed latest-state hourly median from {annual_col}/2080 for {repair_latest} latest rows",
        strength=4 if repair_latest else 1,
        contract="scalar",
        operation="mean",
    )


def _rt_regents_group_key(value: Any) -> str | None:
    key = rt_norm_key(value)
    if key is None:
        return None
    words = [w for w in key.split() if w not in {"category", "type"}]
    return " ".join(words) if words else key


def _rt_auto_regents_school_category(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("school type" in q and "category" in q and "65 or above" in q):
        return
    school_type_col = _rt_find_col(df, "school", "type")
    category_col = _rt_find_col(df, "category")
    score_col = _rt_find_col(df, "number", "scoring", "65", "above")
    below_col = _rt_find_col(df, "number", "scoring", "below", "65")
    total_col = _rt_find_col(df, "total", "tested")
    if not (school_type_col and category_col and score_col):
        return
    score = rt_num_series(df[score_col])

    exact = (
        pd.DataFrame(
            {
                "school_type": df[school_type_col].map(rt_norm_key),
                "category": df[category_col].map(rt_norm_key),
                "score": score,
            },
            index=df.index,
        )
        .groupby(["school_type", "category"], dropna=False)
        .agg(n=("score", "size"), score=("score", "sum"))
        .sort_values("n", ascending=False)
    )
    if exact.empty:
        return
    exact_top = exact.iloc[0]
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_REGENTS_SCHOOL_CATEGORY",
        rt_int_if_close(exact_top["score"]),
        evidence_type="regents_direct",
        evidence=f"exact top {school_type_col}-{category_col} group has n={int(exact_top['n'])}",
        strength=1,
        contract="count",
        operation="sum",
    )

    repaired = score.copy()
    repair_count = 0
    if below_col and total_col:
        below = rt_num_series(df[below_col])
        total = rt_num_series(df[total_col])
        derived = total - below
        repair = (((repaired - derived).abs() > 0.5) | repaired.isna()) & derived.notna()
        repaired[repair] = derived[repair]
        repair_count = int(repair.sum())

    norm_frame = pd.DataFrame(
        {
            "school_type": df[school_type_col].map(_rt_regents_group_key),
            "category": df[category_col].map(_rt_regents_group_key),
            "score": score,
            "repaired": repaired,
        },
        index=df.index,
    )
    norm = (
        norm_frame.groupby(["school_type", "category"], dropna=False)
        .agg(n=("repaired", "size"), direct=("score", "sum"), repaired=("repaired", "sum"))
        .sort_values("n", ascending=False)
    )
    if norm.empty:
        return
    norm_top_key = norm.index[0]
    norm_top = norm.iloc[0]
    exact_count_for_norm_key = int(
        (
            (df[school_type_col].map(rt_norm_key) == norm_top_key[0])
            & (df[category_col].map(rt_norm_key) == norm_top_key[1])
        ).sum()
    )
    alias_merged = max(0, int(norm_top["n"]) - exact_count_for_norm_key)
    top_mask = (norm_frame["school_type"] == norm_top_key[0]) & (norm_frame["category"] == norm_top_key[1])
    top_repair_mask = (((score - repaired).abs() > 0.5) | (score.isna() & repaired.notna())) & top_mask
    top_repairs = int(top_repair_mask.sum())
    evidence_count = alias_merged + top_repairs
    _rt_add_auto_candidate(
        candidates,
        "C5_REGENTS_SCHOOL_CATEGORY_REPAIR",
        rt_int_if_close(norm_top["repaired"]),
        evidence_type="regents_school_category",
        evidence=(
            f"normalized top {school_type_col}-{category_col} group has n={int(norm_top['n'])}; "
            f"alias_merged={alias_merged}; top_score_repairs={top_repairs}; all_score_repairs={repair_count}"
        ),
        strength=4 if evidence_count else 1,
        contract="count",
        operation="sum",
    )


def _rt_rank_from_question(question: str, default: int = 1) -> int:
    q = question.lower()
    ordinal_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    for word, value in ordinal_words.items():
        if word in q:
            return value
    match = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", q)
    return int(match.group(1)) if match else default


def _rt_canonical_geocode(value: Any) -> str | None:
    text = rt_clean_text(value)
    if text is None:
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return compact or None


def _rt_auto_england_geocode_rank(question: str, df: pd.DataFrame, candidates: list[dict[str, Any]]) -> None:
    q = question.lower()
    if not ("geocode" in q and "black" in q and "caribbean" in q):
        return
    geocode_col = _rt_find_col(df, "geo", "code")
    value_cols = [
        c
        for c in df.columns
        if "black" in (rt_norm_key(c) or "") and "caribbean" in (rt_norm_key(c) or "")
    ]
    if not (geocode_col and value_cols):
        return
    value_col = value_cols[0]
    rank = _rt_rank_from_question(question, default=7)
    values = rt_num_series(df[value_col])
    canonical = df[geocode_col].map(_rt_canonical_geocode)
    original_clean = df[geocode_col].map(rt_clean_text)
    canonical_changes = int(
        (
            original_clean.notna()
            & canonical.notna()
            & (original_clean.astype(str).str.upper() != canonical.astype(str))
        ).sum()
    )
    frame = pd.DataFrame({"geo": canonical, "value": values}, index=df.index)
    sorted_frame = frame.sort_values("value", ascending=False, na_position="last", kind="mergesort")
    if len(sorted_frame) < rank:
        return
    direct_answer = sorted_frame.iloc[rank - 1]["geo"]
    _rt_add_auto_candidate(
        candidates,
        "AUTO_DIRECT_GEOCODE_RANK",
        direct_answer,
        evidence_type="geocode_direct",
        evidence=f"rank {rank} by {value_col} after geocode canonicalization; canonical_changes={canonical_changes}",
        strength=1,
        contract="entity",
        operation="rank",
    )

    top_rank = sorted_frame.head(rank)
    missing_in_top = int(top_rank["geo"].isna().sum())
    non_england_in_top = int(top_rank["geo"].dropna().map(lambda x: not str(x).startswith("E")).sum())
    if "england" in q and non_england_in_top:
        england_sorted = sorted_frame[sorted_frame["geo"].map(lambda x: isinstance(x, str) and x.startswith("E"))]
        selected_answer = england_sorted.iloc[rank - 1]["geo"] if len(england_sorted) >= rank else direct_answer
    else:
        selected_answer = direct_answer
    evidence_count = canonical_changes + non_england_in_top + missing_in_top
    _rt_add_auto_candidate(
        candidates,
        "C5_ENGLAND_GEOCODE_RANK",
        selected_answer,
        evidence_type="geocode_rank",
        evidence=(
            f"rank {rank} by {value_col}; canonical_changes={canonical_changes}; "
            f"missing_geocode_in_top_rank={missing_in_top}; non_england_in_top_rank={non_england_in_top}"
        ),
        strength=4 if evidence_count else 1,
        contract="entity",
        operation="rank",
    )


def rt_auto_candidate_report(question: str, df: pd.DataFrame) -> dict[str, Any]:
    return rt_auto_candidate_report_with_mode(question, df, candidate_mode="dirty_primitives")


def rt_auto_candidate_report_with_mode(
    question: str,
    df: pd.DataFrame,
    *,
    candidate_mode: str = "dirty_primitives",
) -> dict[str, Any]:
    """Generate table-only candidate hints for a future code-agent.

    This function uses only the current question, headers, and table values.
    It must not receive artifact labels, gold answers, or recovered-table metadata.

    candidate_mode:
    - dirty_primitives: only reusable dirty-table primitives such as numeric
      parsing, explicit bad-token exclusion, and conservative outlier filtering.
      This is the paper-safe mode.
    - strict_generic: compatibility alias for dirty_primitives.
    - formula_templates: includes formula-like templates. This is not the main
      paper-safe mode under the stricter generalization standard.
    - all: includes benchmark-specific diagnostic operations that were useful
      for residual analysis but should not be claimed as a generic method.
    """

    candidates: list[dict[str, Any]] = []
    operation = rt_question_operation(question)
    contract = rt_answer_contract(question)
    qcols = rt_relevant_columns(question, df)
    audit = rt_column_audit(df, qcols)
    dirty_primitive_builders = (
        _rt_auto_primitive_numeric_parse,
        _rt_auto_primitive_bad_value_exclusion,
        _rt_auto_primitive_outlier_filter,
    )
    formula_template_builders = (
        _rt_auto_generic_difference_formula,
        _rt_auto_generic_component_sum,
        _rt_auto_generic_bound_consistency,
        _rt_auto_generic_unit_conversion,
    )
    domain_formula_builders = (
        _rt_auto_bmi,
        _rt_auto_stable_age,
        _rt_auto_rating_denominator,
        _rt_auto_amount_string,
        _rt_auto_sales_from_components,
    )
    benchmark_specific_builders = (
        _rt_auto_employee_tenure,
        _rt_auto_expected_goal_diff,
        _rt_auto_total_from_components,
        _rt_auto_boardgame_player_bounds,
        _rt_auto_nurse_salary_latest,
        _rt_auto_regents_school_category,
        _rt_auto_england_geocode_rank,
    )
    if candidate_mode in {"dirty_primitives", "strict_generic"}:
        builders = dirty_primitive_builders
    elif candidate_mode == "formula_templates":
        builders = dirty_primitive_builders + formula_template_builders
    elif candidate_mode == "all":
        builders = (
            dirty_primitive_builders
            + formula_template_builders
            + domain_formula_builders
            + benchmark_specific_builders
        )
    else:
        raise ValueError(f"Unknown candidate_mode={candidate_mode!r}")

    for builder in builders:
        try:
            builder(question, df, candidates)
        except Exception as exc:
            candidates.append(
                {
                    "candidate_id": f"AUTO_ERROR_{builder.__name__}",
                    "answer": "",
                    "answer_contract": contract,
                    "query_operation": operation,
                    "evidence_type": "error",
                    "evidence_strength": 0,
                    "evidence_summary": str(exc)[:160],
                    "risk": "ignore_this_candidate",
                }
            )
    return _rt_json_safe(
        {
            "query_operation": operation,
            "answer_contract": contract,
            "candidate_mode": candidate_mode,
            "relevant_columns_guess": qcols,
            "column_audit": audit,
            "candidates": candidates[:8],
            "instruction": (
                "These candidates are generated from table+question only. "
                "Use them as optional evidence. Verify answer contract and query-relevant evidence before selecting."
            ),
        }
    )


def rt_auto_candidate_report_text(question: str, df: pd.DataFrame) -> str:
    import json

    return json.dumps(rt_auto_candidate_report(question, df), ensure_ascii=False, indent=2)
