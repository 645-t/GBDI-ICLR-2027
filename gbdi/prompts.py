from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

OFFICIAL_CODE_AGENT_PREAMBLE = """
  SETTING: You are an expert-level data scientist. Your job is to answer a data driven question \
  in rigorous manner given a data table.
  In your analysis:
  * Carefully address
  1) missing data: empty or null entries simulating incomplete information
  2) bad values: clearly erroneous or placeholder entries (e.g., `-1`, `9999`, `TEST`, `#REF!` etc.)
  3) outliers: implausible extreme values that distort analysis (e.g., 220 breathing rate per minute)
  4) inconsistent formatting: variations in representing the same value (e.g., `22 lbs`, `22 pounds`, `weight = 22`)
  5) inconsistent logic: cross-field contradictions violating common-sense logic (e.g., end time before start time)
  * Attempt to safely recover or correct flawed data when reasonable based on the existing data. If data is irrecoverable or suspect, discard the row.
  You will be working within a Python shell and can use the following commands to answer the question.
  
  AVAILABLE COMMANDS:
  {command_docs}
  RESPONSE_FORMAT:
  Each response must include:
  1. A DISCUSSION field â€” where you will methodically break down the reasoning process, illustrating how you arrive at conclusions and decide what to do next.
  2. A command field â€” proprtly formatted YAML within triple backticks and following the structure from COMMANDS.
  Important rules:
  - Always include exactly one DISCUSSION and one command block.
  - Ensure the command block is properly formatted YAML with proper indents and newlines (see the example below).
  For example, given a question asking for the average income. You might respond:
  DISCUSSION
  Let's think step by step. \
  We need to first find the average income of the population. \
  We can do this by summing up the income column and dividing by the number of rows.
  ```yaml
  command: "python"
  kwargs:
    code: |-
      income_avg = df['income'].sum() / len(df)
      income_avg
  ```
"""


OFFICIAL_TASK_PROMPT = """
  Begin!
  Data table (stored in a pandas dataframe named `df`):
  {table}
  All cells in the `df` are `object` data type, regardless of their appearance.
  Question:
  {question}
"""


OFFICIAL_STATE_PROMPT = """
  Observation:
  {observation}
"""


def command_docs() -> str:
    docs = {
        "python": {
            "docstring": (
                "Execute Python code within a persistent Python shell. The shell "
                "maintains state across executions, so variables and imports from "
                "previous runs remain available. When first using this command, the "
                "data table is provided as a global variable named `df`, and "
                "`pandas` has already been imported as `pd`."
            ),
            "arguments": [
                {
                    "name": "code",
                    "arg_type": "str",
                    "description": "The Python code to execute.",
                    "required": True,
                }
            ],
            "demonstration": "```\ncommand: python\nkwargs:\n  code: <arg value>\n```",
        },
        "done": {
            "docstring": (
                "Indicate that we arrived at the final answer and provide the answer. "
                "Use this command only when you have arrived at the final answer."
            ),
            "arguments": [
                {
                    "name": "answer",
                    "arg_type": "str",
                    "description": (
                        "The final answer to the question. Do not apply any formatting, "
                        "bolding, or markup. If the question asks for a list of values, "
                        "then the answer should be a comma-separated list of values "
                        "(e.g., '42, 43, 44')"
                    ),
                    "required": True,
                }
            ],
            "demonstration": "```\ncommand: done\nkwargs:\n  answer: <arg value>\n```",
        },
    }
    return yaml.dump(docs, sort_keys=False) + '\nPlease enclose all command kwargs values in ""'


def build_task_prompt(query: str, df: pd.DataFrame, table_format: str = "csv") -> str:
    if table_format == "csv":
        table = df.to_csv(index=False, lineterminator="\r\n")
    elif table_format == "html":
        table = df.to_html(index=False, escape=True, border=0)
    else:
        raise ValueError(f"Unsupported official table format: {table_format}")
    return OFFICIAL_TASK_PROMPT.format(table=table, question=query)


def build_system_prompt(method="no_memory", memory_items=None):
    if method != "no_memory" or memory_items:
        raise ValueError("Only the paper's memory-free base interface is supported")
    return OFFICIAL_CODE_AGENT_PREAMBLE.format(command_docs=command_docs())
DIRECT_PROMPT_PREAMBLE = """
  You are an expert-level data scientist. Your job is to answer a data analysis question in rigorous manner given a data table.
  In your analysis:
  * Carefully address
    1) missing data: empty or null entries simulating incomplete information
    2) bad values: clearly erroneous or placeholder entries (e.g., `-1`, `9999`, `TEST`, `#REF!` etc.)
    3) outliers: implausible extreme values that distort analysis (e.g., 220 breathing rate per minute)
    4) inconsistent formatting: variations in representing the same value (e.g., `22 lbs`, `22 pounds`, `weight = 22`)
    5) inconsistent logic: cross-field contradictions violating common-sense logic (e.g., end time before start time)
  * Attempt to safely recover or correct flawed data when reasonable based on the existing data. If data is irrecoverable or suspect, discard the row.
  * Do NOT write or execute any code. Focus purely on logical reasoning and analytical judgment.
  You must conclude with your most reasonable answer.

  When you provide the final answer, please use the prefix "The answer is:" \
  without any modification, and provide the answer directly, with no formatting, no bolding, and \
  no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question asks \
  for a list of values, then the answer should be a comma-separated list of values, \
  without any formatting, no bolding, and no markup. For instance: "The answer is: 42, 43, 44" or "The answer is: yes, no".
"""


DIRECT_TASK_PROMPT = """
  Data:
  {table}
  Based on the given table, answer the following question:
  {question}
"""


def build_direct_prompt_messages(question: str, table_csv: str, extra_instruction: str | None = None) -> list[dict[str, str]]:
    system_content = DIRECT_PROMPT_PREAMBLE
    if extra_instruction:
        system_content = f"{DIRECT_PROMPT_PREAMBLE}\n\n{extra_instruction.strip()}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": DIRECT_TASK_PROMPT.format(table=table_csv, question=question)},
    ]


def extract_value_from_answer(text: str) -> str:
    prefix = "The answer is:"
    lower_text = text.lower()
    if prefix.lower() not in lower_text:
        raise ValueError(
            'No "The answer is:" found in the response. Please write your precise'
            ' answer after the prefix "The answer is:". Do not return any more'
            " thinking just the answer. If there is no answer, return The answer"
            " is: no answer found"
        )
    idx = lower_text.rfind(prefix.lower())
    return text[idx + len(prefix) :].strip()
