"""Offline integrity checks for the distributed reproduction package."""
import hashlib
from collections import Counter
from . import data, geometry
from .evaluation import archived_qa_groups


def verify():
    manifest=data.read_json(data.ROOT/'manifest.json')
    for name,expected in manifest['sha256'].items():
        path=data.ROOT/name
        assert path.is_file(),f'Missing file: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'Changed file: {name}'
    examples=data.load_data();index={x['example_id']:x for x in examples}
    assert len(examples)==len(index)==313
    assert len({x['task_id'] for x in examples})==53
    sources=data.read_json(data.ROOT/'data/task_sources.json')
    assert len(set(sources.values()))==27
    reference=data.read_json(data.ROOT/'data/human_reviewed_reference.json')
    assert len(reference)==154 and sum(bool(r['critical_rows']) for r in reference)==141
    refs={r['example_id']:r for r in reference}
    assert len(refs)==len(reference),'Duplicate human-reviewed reference instance'
    assert len({index[e]['task_id'] for e in refs})==34
    assert len({sources[index[e]['task_id']] for e in refs})==24
    cohorts=data.read_json(data.ROOT/'data/cohorts.json')
    assert len(cohorts['benchmark'])==len(set(cohorts['benchmark']))==313
    assert set(cohorts['benchmark'])==set(index),'Frozen benchmark cohort differs from benchmark instances'
    assert len(cohorts['reviewed'])==len(set(cohorts['reviewed']))==154
    assert set(cohorts['reviewed'])==set(refs),'Frozen reviewed cohort differs from the human-reviewed reference'
    layouts=0
    for axis,n in [('position',60),('spacing',59)]:
        designs=data.designs(axis)
        assert len(designs)==n
        for d in designs:
            assert d['target_original_rows_offline_only']==refs[d['example_id']]['critical_rows']
            orders,_=geometry.make_band_orders(d,3) if axis=='position' else geometry.make_orders(d)
            assert orders==d['row_orders'],'Frozen layout regeneration mismatch'
            for order in orders.values():
                assert sorted(order)==list(range(d['n_rows']))
            layouts+=len(orders)
    orders=data.read_json(data.ROOT/'data/random_views.json.gz')
    for eid,entry in orders.items():
        assert len(entry['orders'])==5
        assert all(sorted(order)==list(range(len(index[eid]['rows']))) for order in entry['orders'])
    for axis,n in [('position',615),('spacing',369)]:
        designs=data.designs(axis,external=True)
        assert len(designs)==41 and sum(len(d['row_orders']) for d in designs)==n
        for d in designs:
            assert len(d['target_original_rows_offline_only'])==3
            assert all(sorted(order)==list(range(d['n_rows'])) for order in d['row_orders'].values())
    recipes=data.read_json(data.ROOT/'configs/experiments.json')
    assert len({r['id'] for r in recipes})==len(recipes),'Duplicate experiment recipe'
    qa=data.read_json(data.ROOT/'results/qa.json.gz')
    archived_qa_groups(qa,index,{r['id']:r for r in recipes},cohorts)
    counts=Counter(r['recipe'] for r in qa)
    counts.update(r['recipe'] for r in data.read_json(data.ROOT/'results/geometry.json.gz'))
    assert {r['id']:r['expected_jobs'] for r in recipes}==dict(counts)
    return {'status':'passed','api_calls':0,'files_checked':len(manifest['sha256']),
            'paper_recipes':len(recipes),'benchmark_instances':313,'reviewed_instances':154,
            'geometry_views_per_system_and_mode':layouts,'frozen_random_views':1565,
            'qa_records_checked':len(qa),'qa_scoring':'strict_final_submission'}
