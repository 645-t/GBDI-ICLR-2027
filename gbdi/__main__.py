import argparse
import json
from pathlib import Path
from . import data


def main():
    parser = argparse.ArgumentParser(description='Frozen GBDI paper reproduction')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List all reported experimental cells')
    commands.add_parser('verify', help='Offline input, integrity and protocol checks')
    replay = commands.add_parser('replay', help='Recompute paper metrics from archived minimal outputs')
    replay.add_argument('--output', type=Path, default=Path('runs/replay'))
    score = commands.add_parser('score', help='Score a newly collected run offline')
    score.add_argument('run_directory', type=Path)
    run = commands.add_parser('run', help='Plan a cell; live APIs require --execute')
    run.add_argument('recipe')
    run.add_argument('--execute', action='store_true')
    run.add_argument('--limit', type=int)
    run.add_argument('--output', type=Path, default=Path('runs'))
    args = parser.parse_args()
    if args.command == 'verify':
        from .verify import verify
        print(json.dumps(verify(), indent=2))
        return
    if args.command == 'replay':
        from .evaluation import replay
        print(json.dumps(replay(args.output), indent=2))
        return
    if args.command == 'score':
        from .evaluation import score_run
        print(json.dumps(score_run(args.run_directory),indent=2))
        return
    recipes = data.read_json(data.ROOT/'configs/experiments.json')
    if args.command == 'list':
        for spec in recipes:
            print(f"{spec['id']:<53} {spec['expected_jobs']:>5} instances/views")
        return
    selected = [r for r in recipes if r['id']==args.recipe]
    if not selected:
        parser.error('Unknown recipe; use python -m gbdi list')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    spec = selected[0]
    from .workflows import jobs, run_job, digest
    work = list(jobs(spec))
    if len(work) != spec['expected_jobs']:
        raise AssertionError('Recipe coverage differs from frozen design')
    if args.limit:
        work = work[:args.limit]
    print(json.dumps({'recipe':spec,'selected_jobs':len(work),'execute':args.execute},indent=2))
    if not args.execute:
        return
    from .client import Client
    client = Client(spec['system'], spec['family'])
    examples = data.examples()
    output = args.output/spec['id']
    output.mkdir(parents=True,exist_ok=True)
    manifest = {'recipe':spec,'settings':client.settings,'selected_jobs':len(work),'protocol_digest':data.read_json(data.ROOT/'manifest.json')['protocol_digest']}
    frozen = output/'run_manifest.json'
    if frozen.exists() and data.read_json(frozen)!=manifest:
        raise RuntimeError('Existing run settings differ; select a new --output directory')
    frozen.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    failures = 0
    for index,(eid,arm,design) in enumerate(work,1):
        destination = output/(digest([eid,arm])+'.json')
        if destination.exists():
            failures += bool(data.read_json(destination).get('error_type'))
            continue
        try:
            result = run_job(examples.get(eid),spec,arm,design,client,args.output/'discovery_cache')
        except Exception as exc:
            # Never write exception strings containing request headers or local paths.
            result = {'error_type':type(exc).__name__,'valid_submission':False,'parse_ok':False}
            failures += 1
        result.update(example_id=eid,arm=arm,recipe=spec['id'])
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        temporary.replace(destination)
        print(f'{index}/{len(work)} saved; {failures} errors',flush=True)
    if failures:
        raise SystemExit(f'{failures} failed jobs remain recorded and count as incorrect/empty predictions')


if __name__ == '__main__':
    main()
