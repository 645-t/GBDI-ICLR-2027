"""Offline, paired evaluation of frozen predictions; no model calls."""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import csv
import json
import random
import numpy as np
import pandas as pd
from . import data
from .scoring import official_match_answer


def row_metrics(predicted, target):
    pred, truth = set(predicted), set(target)
    tp = len(pred & truth)
    return {'cdr':float(truth <= pred) if truth else None,
            'precision':tp/len(pred) if pred else 0.0,
            'recall':tp/len(truth) if truth else None,
            'f1':2*tp/(len(pred)+len(truth)) if truth else None,
            'rows':len(pred), 'empty_fp':bool(pred) if not truth else None}


def archived_scores(record, gold):
    """Derive both scoring profiles from the one answer used in the paper."""
    matched=bool(official_match_answer(record['answer'],gold))
    strict=bool(record['valid_submission'] and matched)
    fallback=record.get('answer_source')=='python_observation_after_limit'
    recorded=bool((record['valid_submission'] or fallback) and matched)
    if bool(record['correct'])!=recorded:
        raise ValueError('Archived correctness does not match the saved answer and its provenance')
    return strict,recorded,fallback


def cluster_interval(values, clusters, draws=10000, seed=20260905427):
    """Instance-weighted source-table bootstrap used for QA."""
    frame = pd.DataFrame({'a':values,'b':np.ones(len(values)),'cluster':clusters})
    groups = frame.groupby('cluster',sort=True)[['a','b']].sum()
    indices = np.random.default_rng(seed).integers(0,len(groups),size=(draws,len(groups)))
    samples = 100*groups.a.to_numpy()[indices].sum(1)/groups.b.to_numpy()[indices].sum(1)
    return {'estimate_pp':100*float(np.mean(values)), 'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),
            'clusters':len(groups),'instances':len(values),'draws':draws}


def geometry_interval(values, tasks, seed):
    """Exact random generator and percentile convention of geometry analyses."""
    ids = sorted({tasks[eid] for eid in values})
    grouped = [[v for eid,v in values.items() if tasks[eid]==t] for t in ids]
    totals, counts = np.array([sum(v) for v in grouped]),np.array([len(v) for v in grouped])
    rng = random.Random(seed)
    indices=np.array([[rng.randrange(len(ids)) for _ in ids] for _ in range(20000)])
    samples=np.sort(100*totals[indices].sum(1)/counts[indices].sum(1))
    return {'estimate_pp':100*sum(values.values())/len(values),'ci95_pp':[float(samples[500]),float(samples[19500])],
            'base_tasks':len(ids),'instances':len(values),'draws':20000,'seed':seed}


def write_csv(path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def validate_replay(output):
    output=Path(output)
    expected=data.read_json(data.ROOT/'results/expected_metrics.json')
    tables={name:pd.read_csv(output/name) for name in {c['file'] for c in expected['checks']}}
    checked=0
    for check in expected['checks']:
        frame=tables[check['file']]
        for key,value in check['selector'].items():frame=frame[frame[key]==value]
        assert len(frame)==1,(check['file'],check['selector'],len(frame))
        row=frame.iloc[0]
        for key,value in check['values'].items():
            assert np.isclose(row[key],value,rtol=1e-11,atol=1e-9),(check['selector'],key,float(row[key]),value)
            checked+=1
    intervals=data.read_json(output/'intervals.json')['geometry_contrasts']
    for recipe,wanted in expected['geometry_intervals'].items():
        for key,value in wanted.items():
            assert np.allclose(intervals[recipe][key],value,rtol=1e-11,atol=1e-9),(recipe,key)
            checked+=1
    return checked


def score_run(directory):
    directory=Path(directory)
    manifest=data.read_json(directory/'run_manifest.json');spec=manifest['recipe']
    records=[data.read_json(p) for p in directory.glob('*.json') if p.name not in {'run_manifest.json','metrics.json'}]
    expected=manifest['selected_jobs']
    if len(records)!=expected:
        raise ValueError(f'Run incomplete: {len(records)} of {expected} selected jobs are present')
    seen=set();scored=[]
    if spec['family'] in {'geometry','external'}:
        ds={d['example_id']:d for d in data.designs(spec['axis'],external=spec['family']=='external')}
        for r in records:
            key=(r['example_id'],r['arm']);assert key not in seen;seen.add(key)
            d=ds[r['example_id']]
            truth={d['row_id_by_original_row'][str(i)] for i in d['target_original_rows_offline_only']}
            pred={str(c['row_id']) for c in r.get('payload',{}).get('candidates',[]) if isinstance(c,dict) and c.get('row_id') is not None} if r.get('parse_ok') else set()
            scored.append({'example_id':r['example_id'],'arm':r['arm'],**row_metrics(pred,truth)})
        summary={k:100*np.mean([r[k] for r in scored]) for k in ['cdr','precision','recall','f1']}
    else:
        gold={x['example_id']:x['answer'] for x in data.load_data()}
        for r in records:
            assert r['example_id'] not in seen;seen.add(r['example_id'])
            correct=bool(r.get('valid_submission') and official_match_answer(r.get('pred_answer',''),gold[r['example_id']]))
            scored.append({'example_id':r['example_id'],'correct':correct})
        summary={'accuracy_pct':100*np.mean([r['correct'] for r in scored])}
    summary.update(recipe=spec['id'],records=len(records),full_paper_jobs=spec['expected_jobs'],
                   is_full_paper_cohort=len(records)==spec['expected_jobs'],api_calls=0)
    write_csv(directory/'scored.csv',scored)
    (directory/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def replay(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inputs={x['example_id']:x for x in data.load_data()}
    sources=data.read_json(data.ROOT/'data/task_sources.json')
    recipes={r['id']:r for r in data.read_json(data.ROOT/'configs/experiments.json')}
    qa=data.read_json(data.ROOT/'results/qa.json.gz')
    groups=defaultdict(dict)
    for r in qa:
        strict,recorded,fallback=archived_scores(r,inputs[r['example_id']]['answer'])
        r.update(correct=strict,recorded_correct=recorded,fallback_in_record=fallback)
        groups[r['recipe']][r['example_id']]=r
    qa_summary=[];discrepancies=[]
    for recipe,items in sorted(groups.items()):
        assert len(items)==recipes[recipe]['expected_jobs']
        strict=sum(r['correct'] for r in items.values())
        recorded=sum(r.get('recorded_correct',r['correct']) for r in items.values())
        qa_summary.append({'recipe':recipe,'n':len(items),'correct_strict':strict,
                           'accuracy_strict_pct':100*strict/len(items),'correct_recorded':recorded,
                           'accuracy_recorded_pct':100*recorded/len(items)})
        for eid,r in items.items():
            if r.get('recorded_correct',r['correct'])!=r['correct']:
                discrepancies.append({'recipe':recipe,'example_id':eid,'strict_correct':r['correct'],
                                      'recorded_correct':r['recorded_correct'],'fallback_in_record':r.get('fallback_in_record',False)})
    contrasts={}
    def paired(name,left,right):
        a,b=groups[left],groups[right];assert set(a)==set(b)
        ids=sorted(a);cluster=[sources[inputs[e]['task_id']] for e in ids]
        contrasts[name]={}
        for profile in ['strict','recorded']:
            value=lambda r:r.get('recorded_correct',r['correct']) if profile=='recorded' else r['correct']
            contrasts[name][profile]=cluster_interval([int(value(a[e]))-int(value(b[e])) for e in ids],cluster)
    for system in ['qwenplus','glm45air','minimaxm25','gpt5mini','deepseek']:
        main=f'gbdi-{system}-random5-skill'
        for baseline in ['direct','code']:
            paired(f'{system}/gbdi-minus-{baseline}',main,f'baseline-{system}-{baseline}')
        repeated=f'gbdi-{system}-repeated5-skill'
        if repeated in groups:paired(f'{system}/random-minus-repeated',main,repeated)
    for system in ['qwenplus','glm45air','minimaxm25']:
        for view in ['random','repeated']:
            paired(f'{system}/{view}-skill-minus-generic',f'gbdi-{system}-{view}5-skill',f'gbdi-{system}-{view}5-generic')
    for guidance in ['no_recovery_rules','no_format_rules','no_execution_checks']:
        paired('qwenplus/'+guidance+'-minus-full','gbdi-qwenplus-random5-'+guidance,'gbdi-qwenplus-random5-skill')
    for a,b in [(3,1),(5,3)]:
        paired(f'qwenplus/views-{a}-minus-{b}',f'gbdi-qwenplus-random{a}-skill',f'gbdi-qwenplus-random{b}-skill')
    paired('qwenplus/views-1-minus-0','gbdi-qwenplus-random1-skill','gbdi-qwenplus-none0-skill')
    for system in ['qwenplus','minimaxm25','gpt5mini']:
        for mode in ['direct','code']:
            prefix=f'table-state-{system}-{mode}-'
            for left,right in [('discovery','raw'),('action','discovery'),('action','raw')]:
                paired(prefix+left+'-minus-'+right,prefix+left,prefix+right)
    write_csv(output/'qa.csv',qa_summary)
    artifacts=defaultdict(list)
    for r in qa:artifacts[r['recipe'],inputs[r['example_id']]['artifact_type']].append(r)
    write_csv(output/'qa_by_artifact.csv',[{'recipe':k[0],'artifact_type':k[1],'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy_pct':100*np.mean([r['correct'] for r in rs])} for k,rs in sorted(artifacts.items())])
    write_csv(output/'scoring_discrepancies.csv',discrepancies)

    frozen={}
    for external in [False,True]:
        for axis in ['position','spacing']:
            frozen[external,axis]={d['example_id']:d for d in data.designs(axis,external)}
    geometry_rows=[];aggregate=defaultdict(list);effects={};geometry_cis={}
    for r in data.read_json(data.ROOT/'results/geometry.json.gz'):
        spec=recipes[r['recipe']];external=spec['family']=='external';d=frozen[external,spec['axis']][r['example_id']]
        target={d['row_id_by_original_row'][str(i)] for i in d['target_original_rows_offline_only']}
        metric=row_metrics(r['predicted_rows'],target)
        placement=d['placement_metadata_offline_only'][r['arm']]
        level=r['arm'].split('_')[-1]
        row=dict(r,**metric,**placement)
        geometry_rows.append(row);aggregate[r['recipe'],level].append(row)
    gsummary=[]
    for (recipe,level),rs in sorted(aggregate.items()):
        gsummary.append({'recipe':recipe,'level':level,'n_views':len(rs),
                         **{k:100*sum(r[k] for r in rs)/len(rs) for k in ['cdr','precision','recall','f1']},
                         'mean_rows':sum(r['rows'] for r in rs)/len(rs),'invalid':sum(not r['parse_ok'] for r in rs)})
    for recipe,spec in recipes.items():
        if spec['family']!='geometry':continue
        rs=[r for r in geometry_rows if r['recipe']==recipe]
        cells=defaultdict(dict);tasks={}
        for r in rs:
            eid=r['example_id'];tasks[eid]=inputs[eid]['task_id']
            key=(eid,r['seed'],r.get('center',''))
            cells[key][r['arm'].split('_')[-1]]=r['cdr']
        per=defaultdict(list);slope=defaultdict(list)
        for (eid,*_),levels in cells.items():
            if spec['axis']=='position':
                per[eid].append(levels['q5']-levels['q1'])
                slope[eid].append(sum((j-3)*levels[f'q{j}'] for j in range(1,6))/10)
            else:per[eid].append(levels['wide']-levels['compact'])
        values={e:sum(v)/len(v) for e,v in per.items()}
        seed=f"{spec['system']}-{spec['mode']}-{spec['axis']}-complete"+('-endpoint' if spec['axis']=='position' else '')
        geometry_cis[recipe]=geometry_interval(values,tasks,seed)
        effects[recipe]=values
        if slope:
            geometry_cis[recipe+'-ordinal-trend']=geometry_interval({e:sum(v)/len(v) for e,v in slope.items()},tasks,f"{spec['system']}-{spec['mode']}-position-complete-ordinal")
    for system in ['qwenplus','minimaxm25','gpt5mini']:
        a,b=effects[f'geometry-{system}-code-spacing'],effects[f'geometry-{system}-direct-spacing']
        geometry_cis[system+'-interface-contrast']=geometry_interval({e:a[e]-b[e] for e in a},{e:inputs[e]['task_id'] for e in a},f'{system}-complete_attenuation')
    write_csv(output/'geometry.csv',gsummary)
    # Joint layout means are needed for the mean-position x dispersion tables.
    joint=defaultdict(list)
    for r in geometry_rows:
        if 'center' in r:joint[r['recipe'],r['center'],r['arm'].split('_')[-1]].append(r)
    write_csv(output/'geometry_joint.csv',[{'recipe':k[0],'center':k[1],'spacing':k[2],'n':len(rs),**{m:100*np.mean([r[m] for r in rs]) for m in ['cdr','precision','recall','f1']}} for k,rs in sorted(joint.items())])

    refs={r['example_id']:set(r['critical_rows']) for r in data.read_json(data.ROOT/'data/human_reviewed_reference.json')}
    passes=defaultdict(dict)
    for r in data.read_json(data.ROOT/'results/discovery.json.gz'):
        eid=r['example_id']
        if eid not in refs or 'payload' not in r:continue
        pred=set()
        for c in r['payload'].get('candidates',[]):
            try:i=int(str(c['row_id']))
            except (KeyError,ValueError,TypeError):continue
            if 0<=i<len(inputs[eid]['rows']):pred.add(i)
        passes[r['system'],r['view'],eid][r['pass']]=pred
    discovery_summary=[];decomposition=[];unions={}
    for system,view in sorted({k[:2] for k in passes}):
        for m in range(1,6):
            for threshold in (range(1,6) if m==5 else [1]):
                metrics=[]
                for subset in combinations(range(1,6),m):
                    for eid,truth in refs.items():
                        support=Counter(i for j in subset for i in passes[system,view,eid][j])
                        pred={i for i,n in support.items() if n>=threshold}
                        metrics.append(row_metrics(pred,truth))
                        if m==5 and threshold==1:unions[system,view,eid]=pred
                nonempty=[r for r in metrics if r['cdr'] is not None];empty=[r for r in metrics if r['cdr'] is None]
                discovery_summary.append({'system':system,'views':view,'m':m,'support':threshold,'subsets':len(list(combinations(range(1,6),m))),
                    **{k:100*np.mean([r[k] for r in nonempty]) for k in ['cdr','precision','recall','f1']},
                    'mean_rows':np.mean([r['rows'] for r in metrics]),'empty_fp_pct':100*np.mean([r['empty_fp'] for r in empty])})
        for guidance in ['generic','skill']:
            recipe=f'gbdi-{system}-{view}5-{guidance}'
            if recipe in groups:
                counts=Counter((bool(truth <= unions[system,view,eid]),groups[recipe][eid]['correct']) for eid,truth in refs.items() if truth)
                for (complete,correct),n in sorted(counts.items()):
                    decomposition.append({'system':system,'views':view,'guidance':guidance,'complete_discovery':complete,'correct_qa':correct,'n':n})
    write_csv(output/'discovery.csv',discovery_summary)
    write_csv(output/'discovery_qa_decomposition.csv',decomposition)
    prefixes=[]
    for m in [0,1,3,5]:
        metrics=[row_metrics(set().union(*(passes['qwenplus','random',e][j] for j in range(1,m+1))),t) for e,t in refs.items()]
        nonempty=[r for r in metrics if r['cdr'] is not None];empty=[r for r in metrics if r['cdr'] is None]
        prefixes.append({'m':m,**{k:100*np.mean([r[k] for r in nonempty]) for k in ['cdr','precision','recall','f1']},'mean_rows':np.mean([r['rows'] for r in metrics]),'empty_fp_pct':100*np.mean([r['empty_fp'] for r in empty])})
    write_csv(output/'discovery_prefix.csv',prefixes)
    discovery_cis={}
    for system in ['qwenplus','glm45air','minimaxm25']:
        task_groups=defaultdict(list)
        for e in refs:task_groups[inputs[e]['task_id']].append(e)
        taskids=sorted(task_groups)
        rng=random.Random('human154-completion')
        samples=[]
        for _ in range(10000):
            selected=[e for _ in taskids for e in task_groups[rng.choice(taskids)] if refs[e]]
            samples.append(100*sum(int(refs[e]<=unions[system,'random',e])-int(refs[e]<=unions[system,'repeated',e]) for e in selected)/len(selected))
        samples.sort();discovery_cis[system]={'ci95_pp':[samples[250],samples[9750]],'base_tasks':len(taskids),'draws':10000}
    result={'qa_contrasts':contrasts,'geometry_contrasts':geometry_cis,'discovery_contrasts':discovery_cis}
    (output/'intervals.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    numeric_checks=validate_replay(output)
    summary={'api_calls':0,'qa_records':len(qa),'geometry_records':len(geometry_rows),
             'numeric_checks_passed':numeric_checks,'scoring_disagreements':len(discrepancies),'output_files':sorted(p.name for p in output.iterdir() if p.is_file())}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
