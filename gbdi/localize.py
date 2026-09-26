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
from .geometry_code import *
from .parsing import _parse_command
from .runtime import chat_with_usage

def source_design(d): return d
def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()
def run_one(
    example: dict[str, Any],
    design: dict[str, Any],
    *,
    axis: str,
    arm: str,
    client: LLMClient,
    max_turns: int,
    transport_retries: int,
) -> dict[str, Any]:
    visible, _, row_order, columns = build_visible(example, source_design(design), arm)
    question = str(example["query"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_table_prompt(question, visible)},
    ]
    assistant_responses: list[str] = []
    events: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    total_latency = 0.0
    total_retries = 0
    payload = {"candidates": [], "unresolved": []}
    parse_ok = False
    completion_status = "max_turns_without_submission"
    error = completion_status
    namespace: dict[str, Any] = {
        "df": visible.copy(deep=True),
        "question": question,
        "pd": pd,
        "np": np,
        "json": json,
        "math": math,
        "re": re,
        "statistics": statistics,
    }

    for turn in range(1, max_turns + 1):
        if turn == max_turns:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is the final turn. Submit one submit_localization command "
                        "now; do not run another Python command."
                    ),
                }
            )
        response, usage, retries, latency = chat_with_usage(
            client, messages, transport_retries=transport_retries
        )
        assistant_responses.append(response)
        messages.append({"role": "assistant", "content": response})
        for key in usage_total:
            usage_total[key] += usage[key]
        total_latency += latency
        total_retries += retries

        try:
            command = _parse_command(response)
        except Exception as exc:
            events.append(
                {
                    "turn": turn,
                    "command": "parse_error",
                    "response_digest": sha256_text(response),
                    "success": False,
                    "error": str(exc),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Command parse error: {exc}. Return one valid YAML command.",
                }
            )
            continue

        if command["command"] == "python":
            code = command["kwargs"].get("code", "")
            if turn == max_turns:
                observation, ok = "Python is unavailable on the reserved final turn.", False
            else:
                observation, ok = run_readonly_python(
                    code, visible, question, namespace
                )
            events.append(
                {
                    "turn": turn,
                    "command": "python",
                    "code": code,
                    "code_digest": sha256_text(code),
                    "stdout": observation,
                    "success": ok,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Read-only Python observation (success={str(ok).lower()}):\n"
                        f"{observation}\n\nThe dataframe remains unchanged. Continue with "
                        "one python or submit_localization command."
                    ),
                }
            )
            continue

        if command["command"] == "submit_localization":
            try:
                parsed = parse_json_response(command["kwargs"].get("payload", ""))
                payload = validate_payload(parsed, visible)
                parse_ok = True
                completion_status = "submitted_localization"
                error = ""
                events.append(
                    {
                        "turn": turn,
                        "command": "submit_localization",
                        "response_digest": sha256_text(response),
                        "success": True,
                    }
                )
                break
            except Exception as exc:
                events.append(
                    {
                        "turn": turn,
                        "command": "submit_localization",
                        "response_digest": sha256_text(response),
                        "success": False,
                        "error": str(exc),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Localization payload failed mechanical validation: {exc}. "
                            "Correct it and submit again."
                        ),
                    }
                )
            continue

        events.append(
            {
                "turn": turn,
                "command": str(command["command"]),
                "response_digest": sha256_text(response),
                "success": False,
                "error": "unsupported_command",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": "Unsupported command. Use only python or submit_localization.",
            }
        )

    return {
        "study": "code_geometry",
        "axis": axis,
        "mode": "code_enabled_localize",
        "arm": arm,
        "model": client.config.model,
        "example_id": example["example_id"],
        "task_id": example["task_id"],
        "artifact_type_offline_only": example["artifact_type"],
        "question": question,
        "parse_ok": parse_ok,
        "payload": payload,
        "completion_status": completion_status,
        "turns": len(assistant_responses),
        "python_calls": sum(event["command"] == "python" for event in events),
        "python_successes": sum(
            event["command"] == "python" and event.get("success") for event in events
        ),
        "events": events,
        "assistant_responses": assistant_responses,
        "messages": messages,
        "usage": usage_total,
        "latency_seconds": total_latency,
        "transport_retries": total_retries,
        "view_metadata_offline_only": {"row_order": row_order, "column_order": columns},
        "error": error,
    }
