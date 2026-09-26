"""Paper QA interfaces. Run generated Python in an isolated environment."""
from pathlib import Path
import json
import math
import re
import statistics
import numpy as np
import pandas as pd
from .parsing import _parse_command, _parse_structured_answer, _run_python_command
from .prompts import (build_system_prompt, build_task_prompt, OFFICIAL_STATE_PROMPT,
                      build_direct_prompt_messages, extract_value_from_answer)


def build_namespace(example, baseline=False):
    namespace = {'df': example['df'].astype(str).copy(deep=True), 'question': example['query'],
                 'pd': pd, 'np': np, 'json': json, 'math': math, 're': re}
    if not baseline:
        namespace['statistics'] = statistics
    helpers = Path(__file__).parent / 'helpers'
    if baseline:
        namespace['radar_ops_path'] = str(helpers / 'radar_candidate_ops.py')
    for name in ('radar_candidate_ops.py', 'radar_integrated_gate_helpers.py'):
        source = helpers / name
        exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
    return namespace


def chat_with_usage(client, messages, transport_retries=4):
    return client.chat(messages, transport_retries)


def direct(example, client, notice=''):
    messages = build_direct_prompt_messages(example['query'], example['df'].astype(str).to_csv(index=False, lineterminator='\n'))
    if notice:
        messages[1]['content'] = notice + '\n' + messages[1]['content']
    answer, valid, responses = '', False, []
    for _ in range(3):
        response, *_ = client.chat(messages)
        responses.append(response)
        messages.append({'role': 'assistant', 'content': response})
        try:
            answer = extract_value_from_answer(response)
            valid = True
            break
        except Exception as exc:
            messages.append({'role': 'user', 'content': str(exc)})
    return {'pred_answer': answer, 'valid_submission': valid, 'responses': responses, 'messages': messages}


def code_agent(example, client, notice=''):
    user = build_task_prompt(example['query'], example['df'].astype(str), table_format='csv')
    if notice:
        user = notice + '\n' + user
    messages = [{'role': 'system', 'content': build_system_prompt()}, {'role': 'user', 'content': user}]
    namespace = build_namespace(example, baseline=True)
    answer, valid, responses = '', False, []
    for _ in range(5):
        response, *_ = client.chat(messages)
        responses.append(response)
        messages.append({'role': 'assistant', 'content': response})
        try:
            command = _parse_command(response)
        except Exception as exc:
            messages.append({'role': 'user', 'content': f'Failed to parse command from response with error: {exc}\nPlease ensure the YAML is properly formatted.'})
            continue
        if command['command'] == 'python':
            obs = _run_python_command(command['kwargs'].get('code', ''), namespace)
            messages.append({'role': 'user', 'content': OFFICIAL_STATE_PROMPT.format(observation=obs)})
        elif command['command'] == 'done':
            answer, valid = command['kwargs'].get('answer', ''), True
            break
        else:
            messages.append({'role': 'user', 'content': f"Invalid command: {command['command']}\nSupported commands are python, done"})
    return {'pred_answer': answer, 'valid_submission': valid, 'responses': responses, 'messages': messages}


def submitted_answer(record):
    """Only an explicit final done command counts for Code Agent / GBDI."""
    responses = record.get('answer_responses', record.get('responses', []))
    if not responses:
        responses = [m['content'] for m in record.get('llm_messages', []) if m.get('role')=='assistant']
    for response in responses[:5]:
        try:
            cmd = _parse_command(response)
        except Exception:
            continue
        if cmd['command'] == 'done':
            answer = str(cmd['kwargs'].get('answer', ''))
            return answer, True
    return '', False
