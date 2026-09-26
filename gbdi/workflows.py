"""Unified entry points for the final paper experiments."""
import hashlib
import json
from pathlib import Path
from . import data, discovery, geometry, gpt_discovery, ledger, runtime, intervention
from .budget import render_budget_ledger


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def discovery_view(example, order, client, gpt=False):
    if gpt:
        ids, _, _ = gpt_discovery.stable_row_ids(example['df'])
        table = gpt_discovery.render_discovery_table(example['df'], ids, order)
        messages = [{'role': 'system', 'content': data.prompt('discovery_gpt')},
                    {'role': 'user', 'content': gpt_discovery.build_discovery_user(example['query'], table)}]
    else:
        visible = discovery.visible_table(example, order)
        messages = [{'role': 'system', 'content': discovery.DIRECT_SCOUT_PROMPT},
                    {'role': 'user', 'content': discovery.scout_user_prompt(example['query'], visible)}]
    response, *_ = client.chat(messages)
    if gpt:
        parsed = gpt_discovery.parse_discovery_v2(response, set(ids.values()), set(example['df'].columns))
        return {'parsed': parsed}
    payload, error = discovery.parse_direct_payload(response, visible)
    return {'payload': payload, 'parse_ok': not error}


def compile_discovery(example, scouts, view, m, gpt=False):
    if gpt:
        ids, _, _ = gpt_discovery.stable_row_ids(example['df'])
        return gpt_discovery.render_ledger(example['df'], {v:k for k,v in ids.items()}, scouts)[0]
    layout = 'random_row5' if view == 'random' else 'repeated5'
    packets, _ = ledger.compile_ledgers([example], scouts, layouts=(layout,))
    rendered = packets[example['example_id']][layout]['rendered']
    return render_budget_ledger(rendered, m) if m != 5 else rendered


def gbdi(example, spec, client, cache=None):
    scouts = []
    is_gpt = spec['system'] == 'gpt5mini'
    m = spec['m']
    if m:
        orders = data.read_json(data.ROOT / 'data/random_views.json.gz')[example['example_id']]['orders'] if spec['views'] == 'random' else [list(range(len(example['df']))) for _ in range(5)]
        for index, order in enumerate(orders[:m]):
            key = digest({'system':spec['system'], 'settings':getattr(client,'settings',{}),
                          'example_id':example['example_id'], 'order':order, 'pass':index,
                          'question':example['query'],'table':digest(example['df'].astype(str).to_csv(index=False,lineterminator='\n')),
                          'prompt':data.prompt('discovery_gpt') if is_gpt else discovery.DIRECT_SCOUT_PROMPT})
            path = Path(cache) / (key + '.json') if cache else None
            if path and path.exists():
                row = data.read_json(path)
            else:
                row = discovery_view(example, order, client, is_gpt)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
            row = dict(row, example_id=example['example_id'], track='direct', pass_index=index,
                       layout='random_row5' if spec['views']=='random' else 'repeated5', **{'pass':index+1})
            scouts.append(row)
        rendered = compile_discovery(example, scouts, spec['views'], m, is_gpt)
    else:
        rendered = ''
    name = {'skill':'intervention_skill', 'generic':'generic_guidance'}.get(spec['guidance'], 'skill_' + spec['guidance'])
    result = intervention.run_one(example, rendered, data.prompt(name), data.prompt('intervention_preturn'), client,
                                  getattr(client, 'settings', {}).get('transport_retries', 4))
    answer, valid = runtime.submitted_answer(result)
    return {'pred_answer': answer, 'valid_submission': valid, 'scouts': scouts,
            'ledger_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
            'answer_responses':result['answer_responses'], 'reserved_turn':result['reserved_turn']}


def geometry_run(example, design, spec, arm, client):
    if spec['mode'] == 'code':
        from . import localize
        # Preserve the frozen per-instance formatting instruction.
        localize.SYSTEM_PROMPT = data.prompt('geometry_code_system')
        if spec['system'] == 'minimaxm25':
            variant = data.read_json(data.ROOT/'data/geometry_code_prompt_variants.json')['minimaxm25'][example['example_id']]
            if variant == 'yaml_explicit':
                localize.SYSTEM_PROMPT = data.prompt('geometry_code_minimax_yaml')
        result = localize.run_one(dict(example, artifact_type=''), design, axis=spec['axis'], arm=arm,
                                  client=client, max_turns=5, transport_retries=4)
        return {'payload':result['payload'], 'parse_ok':result['parse_ok'], 'responses':result['assistant_responses']}
    visible, _, _, _ = geometry.build_visible(example, design, arm)
    messages = [{'role':'system','content':data.prompt('geometry_direct_system')},
                {'role':'user','content':geometry.question_table_prompt(example['query'],visible)}]
    response,*_ = client.chat(messages)
    try:
        payload, valid = geometry.parse_json_response(response), True
    except Exception:
        payload, valid = {'candidates':[]}, False
    return {'payload':payload, 'parse_ok':valid, 'responses':[response]}


def external_run(design, arm, client):
    from .external import SYSTEM_PROMPT, user_prompt
    messages = [{'role':'system','content':SYSTEM_PROMPT}, {'role':'user','content':user_prompt(design,arm)}]
    response,*_ = client.chat(messages)
    try:
        payload, valid = geometry.parse_json_response(response), True
    except Exception:
        payload, valid = {'candidates':[]}, False
    # Paper endpoint: unmatched row IDs remain false positives.
    return {'payload':payload,'parse_ok':valid,'responses':[response]}


def jobs(spec):
    if spec['family'] in {'geometry','external'}:
        for d in data.designs(spec['axis'],external=spec['family']=='external'):
            for arm in sorted(d['row_orders']):
                yield d['example_id'], arm, d
    else:
        cohort = 'reviewed' if spec['family']=='table-state' else 'benchmark'
        for eid in data.read_json(data.ROOT/'data/cohorts.json')[cohort]:
            yield eid, '', None


def run_job(example, spec, arm, design, client, cache=None):
    family = spec['family']
    if family == 'geometry':
        return geometry_run(example, design, spec, arm, client)
    if family == 'external':
        return external_run(design, arm, client)
    if family == 'gbdi':
        return gbdi(example, spec, client, cache)
    notice = ''
    if family == 'table-state':
        example, notice = data.table_state(example, spec['condition'])
    return (runtime.direct if spec['mode']=='direct' else runtime.code_agent)(example,client,notice)
