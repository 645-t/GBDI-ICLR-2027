"""Frozen paper inputs. Gold labels are kept outside model-facing examples."""
from pathlib import Path
import gzip
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw.decode('utf-8-sig'))


def load_data():
    return read_json(ROOT / 'data/radar_t.json.gz')


def examples():
    # Deliberately excludes answers, artifact labels and review annotations.
    return {x['example_id']: {'example_id': x['example_id'], 'task_id': x['task_id'],
            'query': x['question'], 'df': pd.DataFrame(x['rows'], columns=x['headers'], dtype=object)}
            for x in load_data()}


def prompt(name):
    return (ROOT / 'prompts' / (name + '.txt')).read_text(encoding='utf-8')


def designs(axis, external=False):
    return read_json(ROOT / f'data/{"external" if external else "geometry"}_{axis}.json.gz')


def table_state(example, condition):
    from .table_states import mark_table, repaired_table, MARKED_NOTICE, INTERVENED_NOTICE
    if condition == 'raw':
        return example, ''
    annotations = read_json(ROOT / 'data/reviewed_interventions.json')[example['example_id']]
    item = {'table_id': example['example_id'], 'headers': list(example['df'].columns),
            'rows': example['df'].values.tolist()}
    if condition == 'discovery':
        frame, _ = mark_table(item, annotations['evidence'])
        notice = MARKED_NOTICE
    elif condition == 'action':
        frame, _ = repaired_table(item, annotations['actions'])
        notice = INTERVENED_NOTICE
    else:
        raise ValueError(condition)
    return dict(example, df=frame), notice
