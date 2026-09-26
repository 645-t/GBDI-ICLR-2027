from __future__ import annotations
import ast, contextlib, hashlib, html, io, json, math, random, re, statistics, time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

def _decode_provider_tool_envelope(text: str) -> dict[str, Any] | None:
    """Decode explicit tool envelopes emitted by compatible endpoints."""

    def decode_explicit_value(raw_value: str) -> str:
        try:
            return str(json.loads(raw_value))
        except (json.JSONDecodeError, TypeError):
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            # Decode only ordinary quoted-string escapes while preserving
            # Python-specific sequences such as regex ``\d``. Some provider
            # envelopes contain literal source newlines; in that case, an
            # embedded ``\n`` belongs to the Python source and must stay an
            # escape rather than becoming a line break inside a string literal.
            has_literal_newlines = "\n" in value or "\r" in value
            value = (
                value.replace(r'\"', '"')
                .replace(r"\'", "'")
                .replace("\\\\", "\\")
            )
            if not has_literal_newlines:
                value = value.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")
            return value

    def result(match: re.Match[str], *, unescape_html: bool = False) -> dict[str, Any]:
        value = match.group("value")
        if unescape_html:
            value = html.unescape(value).strip()
        else:
            value = decode_explicit_value(value)
        return {
            "command": match.group("tool").lower(),
            "kwargs": {match.group("key").lstrip("-").lower(): str(value)},
        }

    def from_structure(value: Any) -> dict[str, Any] | None:
        if isinstance(value, list):
            for item in value:
                decoded = from_structure(item)
                if decoded is not None:
                    return decoded
            return None
        if not isinstance(value, dict):
            return None

        command_value = str(value.get("command", "")).strip()
        kwargs_value = value.get("kwargs")
        if command_value.lower() in {"python", "done"} and isinstance(kwargs_value, dict):
            key = "code" if command_value.lower() == "python" else "answer"
            payload = kwargs_value.get(key, kwargs_value.get(f"--{key}"))
            if payload is not None:
                return {
                    "command": command_value.lower(),
                    "kwargs": {key: str(payload)},
                }

        tool_value = str(value.get("tool", "")).strip().lower()
        args_value = value.get("args")
        if tool_value in {"python", "done"} and isinstance(args_value, dict):
            key = "code" if tool_value == "python" else "answer"
            payload = args_value.get(key, args_value.get(f"--{key}"))
            if payload is not None:
                return {"command": tool_value, "kwargs": {key: str(payload)}}

        embedded = re.match(
            r"(?is)^(?P<tool>python|done)\s*\nkwargs\s*:\s*\n(?P<body>.*)$",
            command_value,
        )
        if embedded:
            try:
                parsed_body = yaml.safe_load("kwargs:\n" + embedded.group("body"))
            except Exception:
                parsed_body = None
            kwargs = parsed_body.get("kwargs") if isinstance(parsed_body, dict) else None
            key = "code" if embedded.group("tool").lower() == "python" else "answer"
            if isinstance(kwargs, dict) and kwargs.get(key) is not None:
                return {
                    "command": embedded.group("tool").lower(),
                    "kwargs": {key: str(kwargs[key])},
                }

        # Some MiniMax responses expose a tool object as
        # {"code": <payload>, "kwargs": {"code": <same payload>}}.
        # Requiring both explicit copies prevents arbitrary JSON data from
        # being mistaken for an executable command.
        if isinstance(kwargs_value, dict) and value.get("code") is not None:
            outer_code = str(value["code"])
            inner_code = kwargs_value.get("code")
            if inner_code is not None and str(inner_code) == outer_code:
                return {"command": "python", "kwargs": {"code": outer_code}}

        for nested in value.values():
            decoded = from_structure(nested)
            if decoded is not None:
                return decoded
        return None

    decoder = json.JSONDecoder()
    for position, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        decoded = from_structure(value)
        if decoded is not None:
            return decoded

    xml_call = re.search(
        r'(?is)<minimax:tool_call>\s*<invoke\s+name=["\'](?P<tool>python|done)["\']>\s*'
        r'<parameter\s+name=["\'](?P<key>code|answer)["\']>'
        r'(?P<value>(?:(?!</code>|<invoke).)*?)</parameter>\s*'
        r'</invoke>\s*</minimax:tool_call>',
        text,
    )
    if xml_call:
        return result(xml_call, unescape_html=True)

    malformed_xml_call = re.search(
        r'(?is)<invoke\s+name=["\'](?P<tool>python|done)["\']>.*?'
        r'<parameter\s+name=["\'](?P<key>code|answer)["\']>(?P<value>.*?)'
        r'</(?:code|answer)>',
        text,
    )
    if malformed_xml_call:
        return result(malformed_xml_call, unescape_html=True)

    command_tag_call = re.search(
        r'(?is)<command>\s*(?P<tool>python|done)\s*'
        r'<(?P<key>code|answer)>(?P<value>.*?)</(?P=key)>\s*</command>',
        text,
    )
    if command_tag_call:
        return result(command_tag_call, unescape_html=True)

    colon_command_tag_calls = re.finditer(
        r'(?is)<command\s*:\s*(?P<tool>python|done)>\s*'
        r'kwargs\s*:\s*(?P<key>code|answer)\s*:\s*'
        r'(?P<value>"(?:\\.|[^"\\])*")\s*</command>',
        text,
    )
    for colon_command_tag_call in colon_command_tag_calls:
        decoded = result(colon_command_tag_call)
        if decoded["command"] == "done":
            return decoded
        try:
            compile(decoded["kwargs"]["code"], "<provider-colon-command>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return decoded

    tool_tag_calls = re.finditer(
        r'(?is)<tool>\s*(?P<tool>python|done)\s*'
        r'(?P<key>code|answer)\s*:\s*(?P<value>".*?")\s*</tool>',
        text,
    )
    for tool_tag_call in tool_tag_calls:
        decoded = result(tool_tag_call)
        if decoded["command"] == "done":
            return decoded
        try:
            compile(decoded["kwargs"]["code"], "<provider-tool-envelope>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return decoded

    arrow_call = re.search(
        r'(?is)\[TOOL_CALL\]\s*\{\s*tool\s*=>\s*"(?P<tool>python|done)"\s*,\s*'
        r'args\s*=>\s*\{\s*--(?P<key>code|answer)\s+(?P<value>"(?:\\.|[^"\\])*")\s*'
        r'\}\s*\}\s*\[/TOOL_CALL\]',
        text,
    )
    if arrow_call:
        return result(arrow_call)

    string_args_call = re.search(
        r'(?is)\[TOOL_CALL\].*?"tool"\s*:\s*"(?P<tool>python|done)"\s*,\s*'
        r'"args"\s*:\s*"\s*--(?P<key>code|answer)\s+'
        r'(?P<value>"(?:\\.|[^"\\])*")\s*"\s*\}.*?\[/TOOL_CALL\]',
        text,
    )
    if string_args_call:
        return result(string_args_call)

    bare_key_args_call = re.search(
        r'(?is)\[TOOL_CALL\].*?"tool"\s*:\s*"(?P<tool>python|done)"\s*,\s*'
        r'"args"\s*:\s*\{\s*--(?P<key>code|answer)(?:\s*:\s*|\s+)'
        r'(?P<value>"(?:\\.|[^"\\])*")\s*\}.*?\[/TOOL_CALL\]',
        text,
    )
    if bare_key_args_call:
        return result(bare_key_args_call)

    start_tool_call = re.search(
        r'(?is)<StartToolCall>\s*\{\s*tool\s*=>\s*["\'](?P<tool>python|done)["\']\s*,\s*'
        r'args\s*=>\s*\{\s*(?:--)?(?P<key>code|answer)\s*=>\s*'
        r'(?P<value>"(?:\\.|[^"\\])*")\s*\}\s*\}\s*</EndToolCall>',
        text,
    )
    if start_tool_call:
        return result(start_tool_call)

    command_kwargs_call = re.search(
        r'(?is)"command"\s*:\s*"(?P<tool>python|done)(?:\\n|\s+)kwargs"\s*:\s*\{\s*'
        r'"(?P<key>code|answer)"\s*:\s*(?P<value>"(?:\\.|[^"\\])*")\s*\}',
        text,
    )
    if command_kwargs_call:
        return result(command_kwargs_call)

    command_code_call = re.search(
        r'(?is)"command"\s*:\s*"(?P<tool>python|done)\\n(?P<key>code|answer)"\s*:\s*'
        r'(?P<value>"(?:\\.|[^"\\])*")',
        text,
    )
    if command_code_call:
        return result(command_code_call)

    compact_kwargs_calls = re.finditer(
        r'(?ims)^\s*command\s*:\s*(?P<tool>python|done)\s*$\s*'
        r'^\s*kwargs\s*:\s*(?P<key>code|answer)\s*:\s*'
        r'(?P<value>"(?:\\.|[^"\\])*")',
        text,
    )
    for compact_kwargs_call in compact_kwargs_calls:
        decoded = result(compact_kwargs_call)
        if decoded["command"] == "done":
            return decoded
        try:
            compile(decoded["kwargs"]["code"], "<provider-compact-kwargs>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return decoded

    plain_command_calls = re.finditer(
        r'(?ims)^\s*command\s*:\s*(?P<tool>python|done)\s*$\s*'
        r'^\s*(?:--)?(?P<key>code|answer)\s*:\s*'
        r'(?P<value>"(?:\\.|[^"\\])*")',
        text,
    )
    for plain_command_call in plain_command_calls:
        decoded = result(plain_command_call)
        if decoded["command"] == "done":
            return decoded
        try:
            compile(decoded["kwargs"]["code"], "<provider-plain-command>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return decoded

    code_only_fence_calls = re.finditer(
        r'(?is)```(?:python|py|yaml|yml)\s*'
        r'(?P<key>code)\s*:\s*(?P<value>"(?:\\.|[^"\\])*")\s*```',
        text,
    )
    for code_only_fence_call in code_only_fence_calls:
        value = decode_explicit_value(code_only_fence_call.group("value"))
        try:
            compile(value, "<provider-code-only-fence>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return {"command": "python", "kwargs": {"code": value}}

    direct_python_fences = re.finditer(
        r'(?is)```(?:python|py)\s*(?P<value>.*?)```',
        text,
    )
    for direct_python_fence in direct_python_fences:
        value = direct_python_fence.group("value").strip()
        try:
            compile(value, "<provider-direct-python-fence>", "exec")
        except (SyntaxError, ValueError, TypeError):
            continue
        return {"command": "python", "kwargs": {"code": value}}

    return None


def _parse_command(llm_resp: str) -> dict[str, Any]:
    fence_pattern = re.compile(
        r"```(?:yaml|yml|json|python|py)?\s*(?P<yaml>.*?)(?=```)",
        re.DOTALL | re.IGNORECASE,
    )

    stripped = llm_resp.strip()
    provider_command = _decode_provider_tool_envelope(stripped)
    if provider_command is not None:
        return provider_command
    candidates = [match.group("yaml").strip() for match in fence_pattern.finditer(stripped)]
    if not candidates:
        command_match = re.search(r"(?ms)^command\s*:", stripped)
        if command_match:
            candidates = [stripped[command_match.start() :]]
        else:
            unclosed = re.search(
                r"```(?:yaml|yml|json|python|py)?\s*(?P<yaml>.*)$",
                stripped,
                re.DOTALL | re.IGNORECASE,
            )
            candidates = [unclosed.group("yaml").strip()] if unclosed else [stripped]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = yaml.safe_load(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed command is not a mapping.")
            command = str(parsed.get("command", "")).strip("\"'")
            if not command:
                raise ValueError("Parsed command name is empty.")
            kwargs = parsed.get("kwargs") or {}
            if not isinstance(kwargs, dict):
                raise ValueError("Parsed kwargs is not a mapping.")
            return {"command": command, "kwargs": {str(k): "" if v is None else str(v) for k, v in kwargs.items()}}
        except Exception as exc:
            last_error = exc
    raise ValueError(f"We failed to parse a tool command: {last_error}")


def _parse_structured_answer(answer: str) -> dict[str, Any] | None:
    text = str(answer or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    candidates = [text]
    obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if obj_match and obj_match.group(1) != text:
        candidates.append(obj_match.group(1))
    for candidate in candidates:
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed
            except Exception:
                pass
    return None


def _run_python_command_with_status(code: str, namespace: dict[str, Any]) -> tuple[str, bool]:
    stdout = io.StringIO()
    try:
        parsed = ast.parse(code, mode="exec")
        result = None
        with contextlib.redirect_stdout(stdout):
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                prefix = ast.Module(body=parsed.body[:-1], type_ignores=[])
                exec(compile(prefix, "<radar-code-agent>", "exec"), namespace)
                expr = ast.Expression(parsed.body[-1].value)
                result = eval(compile(expr, "<radar-code-agent>", "eval"), namespace)
            else:
                exec(compile(parsed, "<radar-code-agent>", "exec"), namespace)
        printed = stdout.getvalue().strip()
        if result is not None:
            if printed:
                return f"{printed}\n{result}".strip(), True
            return str(result), True
        return printed, True
    except Exception as exc:
        return str(exc), False


def _run_python_command(code: str, namespace: dict[str, Any]) -> str:
    observation, _ = _run_python_command_with_status(code, namespace)
    return observation
