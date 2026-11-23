"""Fresh V9 prequential study. Update repair memory only between ten-case waves."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

from clean_harness.graph import key
from clean_harness.portfolio import rank
from retroagent_langgraph import cli, reasoning
from retroagent_langgraph.config import Config
from retroagent_langgraph.runtime import Runtime
from retroagent_langgraph.provider_memory import molecule_key
from retrosynthesis_client import atomic_json
from retroagent_langgraph.strategy_memory import GUIDANCE, contract_tasks


def read(p):
    return json.loads(p.read_bytes())


def weak_steps(metric):
    return [s['product'] for s,d in zip(metric['route']['steps'],metric['step_diagnostics'])
            if d['score_with_fg']<=0] if metric else []


def condition(metric):
    n=len(weak_steps(metric))
    return ('stock' if metric and metric['stock_closed'] else 'open')+'|'+('2+' if n>=2 else str(n))


def worker(payload):
    job,stats=payload
    out=job[1]
    original_advance,original_reason=Runtime.advance,reasoning.run

    def advance(runtime,state):
        best=max(runtime.metrics.values(),key=rank,default=None)
        atomic_json(out/'phase_one_incumbent.json',best)
        return original_advance(runtime,state)

    def guided(runtime,tasks,views=3):
        if runtime.phase!=2 or runtime.quality()['strict_closed'] or not stats.get('all',{}).get('wins'):
            return original_reason(runtime,tasks,views)
        anchor=max(runtime.metrics.values(),key=rank,default=None)
        scope=condition(anchor)
        weak=weak_steps(anchor) if anchor and anchor['stock_closed'] else []
        def preference(strategy):
            local=stats.get(scope,{}); total=stats['all']
            return (local.get(strategy,0)+2*(total.get(strategy,0)+1)/(total['wins']+3))/(local.get('attempts',0)+2)
        order=sorted(GUIDANCE,key=lambda k:(-preference(k),k))
        roles=reasoning.ROLES
        reasoning.ROLES=tuple((role[0],GUIDANCE[k]) for role,k in zip(roles,order))
        try:
            return original_reason(runtime,contract_tasks(runtime,tasks,order,anchor,weak),views)
        finally:
            reasoning.ROLES=roles

    Runtime.advance=advance
    reasoning.run=guided
    try:
        return cli.worker(job)
    finally:
        Runtime.advance,reasoning.run=original_advance,original_reason


def observation(case,row):
    first=case/'phase_one_incumbent.json'
    result=read(case/'result.json')
    if not first.exists() or result['first_phase']['quality']['strict_closed']:
        return None
    if result.get('errors') or row.get('llm_failed'):
        return None
    before=read(first)
    strategy=None
    if row['metrics']['strict_closed'] and result['selected']:
        old={s['product']:key(s) for s in (before or {}).get('route',{}).get('steps',[])}
        new={s['product']:key(s) for s in result['selected']['route']['steps']}
        root=result['target']['SMILES']
        weak=weak_steps(before)
        strategy=('change_root' if new.get(root)!=old.get(root) else
                  'replace_weak' if any(old.get(p)!=new.get(p) for p in weak) else 'supply')
    return dict(context=condition(before),strategy=strategy,strict=bool(strategy),
                native_calls=result['costs']['native_calls'],llm_calls=result['costs']['llm_calls'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('targets','config','out','service','cc-service'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--llm-concurrency',type=int,default=18)
    a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    targets=sorted(read(a.targets),key=lambda t:sha256(('prequential-v9-v1:'+t['Structure ID']).encode()).digest())
    assert len(targets)>0 and len(targets)==len({molecule_key(t['SMILES']) for t in targets})
    cfg=Config(**read(a.config))
    manifest=dict(order=targets,config=asdict(cfg),wave_size=10,seed_memory='empty',
                  evaluation='prequential',source=sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')!=manifest:raise ValueError('Study protocol changed')
    else:atomic_json(out/'protocol.json',manifest)
    stats=defaultdict(Counter);all_rows=[];waves=[]
    ctx=get_context('spawn')
    blocked=sorted(t['SMILES'] for t in targets)
    with ctx.Manager() as manager:
        gate=manager.BoundedSemaphore(a.llm_concurrency)
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx,max_tasks_per_child=1) as pool:
            for start in range(0,len(targets),10):
                wave=out/f'wave_{start//10+1:02d}'
                wave.mkdir(exist_ok=True)
                summary=wave/'complete.json'
                if summary.exists():
                    bundle=read(summary)
                else:
                    atomic_json(wave/'memory_before.json',stats)
                    jobs=[]
                    for t in targets[start:start+10]:
                        folder=wave/'cases'/t['Structure ID']
                        if folder.exists():
                            from time import time_ns
                            folder.rename(folder.with_name(f'{folder.name}.interrupted-{time_ns()}'))
                        jobs.append(((t,folder,a.service.resolve(),a.cc_service.resolve(),blocked,cfg,gate),dict(stats)))
                    rows=list(pool.map(worker,jobs))
                    lessons=[o for r in rows if r['status']=='completed' and r['audit']['passed']
                             if (o:=observation(wave/'cases'/r['target'],r))]
                    bundle=dict(rows=rows,observations=lessons)
                    atomic_json(summary,bundle)
                for lesson in bundle['observations']:
                    for group in ('all',lesson['context']):
                        stats[group]['attempts']+=1
                        if lesson['strategy']:
                            stats[group]['wins']+=1
                            stats[group][lesson['strategy']]+=1
                        else:stats[group]['failures']+=1
                atomic_json(wave/'memory_after.json',stats)
                all_rows.extend(bundle['rows'])
                waves.append(dict(wave=start//10+1,metrics=cli.aggregate(bundle['rows'],10),
                                  learning_observations=len(bundle['observations'])))
                state=dict(processed=len(all_rows),waves=waves,metrics=cli.aggregate(all_rows,len(all_rows)))
                atomic_json(out/'progress.json',state)
                print(json.dumps(state['waves'][-1]),flush=True)
    atomic_json(out/'summary.json',state)


if __name__=='__main__':
    main()
