from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from .prompts import OFFICIAL_STATE_PROMPT, build_system_prompt, build_task_prompt
from .runtime import chat_with_usage, build_namespace
from .parsing import _parse_command, _parse_structured_answer, _run_python_command
from .scoring import official_match_answer

FINAL_QA_MAX_STEPS = 5
ARM = "gbdi"
def sha256_text(s): return hashlib.sha256(s.encode()).hexdigest()
def answer_relaxed_em(a,b): return official_match_answer(a,b)
def add_usage(total: dict[str, int], value: dict[str, int]) -> None:
    for key in total:
        total[key] += int(value.get(key, 0))


def answer_with_namespace(
    messages: list[dict[str, str]],
    namespace: dict[str, Any],
    example: dict[str, Any],
    client: Any,
    retries: int,
) -> dict[str, Any]:
    assistant_responses: list[str] = []
    python_snippets: list[str] = []
    observations: list[str] = []
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency = 0.0
    transport_retries = 0
    answer = ""

    for turn in range(1, FINAL_QA_MAX_STEPS + 1):
        response, usage, used_retries, elapsed = chat_with_usage(
            client, messages, transport_retries=retries
        )
        assistant_responses.append(response)
        messages.append({"role": "assistant", "content": response})
        add_usage(usage_total, usage)
        latency += elapsed
        transport_retries += used_retries
        try:
            command = _parse_command(response)
        except Exception as exc:
            message = f"Failed to parse command: {exc}. Return one valid official YAML command."
            errors.append(message)
            events.append({"turn": turn, "command": "parse_error", "success": False})
            messages.append({"role": "user", "content": message})
            continue

        if command["command"] == "python":
            code = str(command["kwargs"].get("code", ""))
            python_snippets.append(code)
            observation = _run_python_command(code, namespace)
            observations.append(observation)
            events.append({"turn": turn, "command": "python", "success": True})
            messages.append(
                {
                    "role": "user",
                    "content": OFFICIAL_STATE_PROMPT.format(observation=observation),
                }
            )
            continue

        if command["command"] == "done":
            answer = str(command["kwargs"].get("answer", ""))
            events.append({"turn": turn, "command": "done", "success": True})
            break

        message = f"Invalid command: {command['command']}. Supported commands are python, done."
        errors.append(message)
        events.append({"turn": turn, "command": str(command["command"]), "success": False})
        messages.append({"role": "user", "content": message})

    if not answer:
        answer = "no_answer_extracted"
        errors.append("no_answer_extracted_after_max_steps")

    structured = _parse_structured_answer(answer)
    final_answer = structured.get("answer", answer) if structured else answer
    return {
        "pred_answer": final_answer,
        "answer_raw": answer,
        "responses": assistant_responses,
        "python_code": python_snippets,
        "observations": observations,
        "errors": errors,
        "events": events,
        "usage": usage_total,
        "latency_seconds": latency,
        "transport_retries": transport_retries,
        "messages": messages,
    }


def run_one(
    example: dict[str, Any],
    ledger_text: str,
    skill: str,
    contract: str,
    client: Any,
    retries: int,
) -> dict[str, Any]:
    system = (
        build_system_prompt("no_memory", []).rstrip()
        + "\n\nQUERY-CONDITIONED INTERVENTION SKILL:\n\n"
        + skill.strip()
        + "\n\n"
        + contract.strip()
    )
    user = build_task_prompt(
        str(example["query"]), example["df"].astype(str), table_format="csv"
    )
    if ledger_text.strip():
        user = f"{user.rstrip()}\n\n{ledger_text.rstrip()}\n"
    user += (
        "\nBEGIN RESERVED INTERVENTION TURN\n"
        "Perform the intervention preturn now. Do not answer the question in this turn."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    namespace = build_namespace(example)
    namespace["work_df"] = namespace["df"].copy(deep=True)
    namespace["intervention_log"] = []

    pre_response, pre_usage, pre_retries, pre_latency = chat_with_usage(
        client, messages, transport_retries=retries
    )
    messages.append({"role": "assistant", "content": pre_response})
    pre_code = ""
    pre_observation = ""
    pre_error = ""
    pre_command = "parse_error"
    contract_ok = False
    try:
        command = _parse_command(pre_response)
        pre_command = str(command["command"])
        if pre_command != "python":
            pre_error = "reserved intervention turn returned a non-python command"
        else:
            pre_code = str(command["kwargs"].get("code", ""))
            pre_observation = _run_python_command(pre_code, namespace)
            contract_ok = bool(pre_code.strip())
    except Exception as exc:
        pre_error = f"{type(exc).__name__}: {exc}"

    work_df = namespace.get("work_df")
    work_df_valid = isinstance(work_df, pd.DataFrame)
    intervention_log = namespace.get("intervention_log")
    log_valid = isinstance(intervention_log, list)
    contract_ok = contract_ok and work_df_valid and log_valid
    if not work_df_valid:
        namespace["work_df"] = namespace["df"].copy(deep=True)
    if not log_valid:
        namespace["intervention_log"] = []

    runtime_note = (
        f"Reserved-turn command: {pre_command}. Contract accepted: {contract_ok}. "
        f"work_df rows: {len(namespace['work_df'])}. "
        f"intervention_log entries: {len(namespace['intervention_log'])}."
    )
    observed = pre_observation if pre_observation else "(no Python observation)"
    messages.append(
        {
            "role": "user",
            "content": (
                "INTERVENTION TURN OBSERVATION\n"
                f"{observed}\n\n"
                f"{runtime_note}\n"
                + (f"Contract error: {pre_error}\n" if pre_error else "")
                + "\nBEGIN FINAL QA\n"
                "Continue as the same agent in the same context. Answer the original question "
                "using the persistent `work_df` and the unchanged literal answer program. "
                "The intervention ledger is fallible: retain only operations justified by visible "
                "evidence. You may use ordinary Python inspection, then submit exactly one final "
                "answer with the official `done(answer)` command. Do not export a repaired table."
            ),
        }
    )
    answered = answer_with_namespace(messages, namespace, example, client, retries)
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    add_usage(total_usage, pre_usage)
    add_usage(total_usage, answered["usage"])
    predicted = answered["pred_answer"]
    return {
        "schema_version": "gbdi_prediction_v1",
        "arm": ARM,
        "model": client.config.model,
        "example_id": str(example["example_id"]),
        "task_id_offline_only": example["task_id"],
        "pred_answer": predicted,
        "answer_completed": predicted != "no_answer_extracted",
        "reserved_turn": {
            "response": pre_response,
            "command": pre_command,
            "python_code": pre_code,
            "observation": pre_observation,
            "error": pre_error,
            "contract_ok": contract_ok,
            "work_df_valid": work_df_valid,
            "intervention_log_valid": log_valid,
            "intervention_log": namespace.get("intervention_log", []),
            "usage": pre_usage,
            "latency_seconds": pre_latency,
            "transport_retries": pre_retries,
        },
        "answer_responses": answered["responses"],
        "answer_python_code": answered["python_code"],
        "answer_observations": answered["observations"],
        "answer_errors": answered["errors"],
        "answer_events": answered["events"],
        "usage": total_usage,
        "latency_seconds": pre_latency + answered["latency_seconds"],
        "transport_retries": pre_retries + answered["transport_retries"],
        "llm_messages": answered["messages"],
        "ledger_sha256": sha256_text(ledger_text),
        "skill_sha256": sha256_text(skill),
        "contract_sha256": sha256_text(contract),
    }
