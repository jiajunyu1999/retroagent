"""One public entry for fresh single-target and batch LangGraph inference."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LANGCHAIN_TRACING_V2='false',LANGSMITH_TRACING='false')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,replace
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import re
import shutil
import time
import traceback

from prepare_yield_distributions import canon_side
from retrosynthesis_client import atomic_json
from .config import Config

ROOT=Path(__file__).resolve().parents[1]


def worker(job):
    target,out,service,cc,blocked,cfg,gate=job
    out=Path(out);out.mkdir(parents=True,exist_ok=False);chem=None;runtime=None
    try:
        from clean_harness.yields import YIELD_CONTEXTS,YIELD_REFERENCE
        from reaction_evidence import INDEX
        spec=json.loads((cc/'ready.json').read_text());center=Path(spec['db']['local'])
        from clean_harness.boundary import install
        install(out,service,cc,[YIELD_CONTEXTS,YIELD_REFERENCE,INDEX,center],None)
        from clean_harness.adapters import Backend
        from .runtime import Runtime
        chem=Backend(service,cc,out,target,blocked,cfg.yield_reporting)
        runtime=Runtime(target,out,chem,cfg,center,gate);result=runtime.run()
        from .audit import audit_case
        validation=audit_case(out,set(blocked))
        return dict(target=target['Structure ID'],status='completed',metrics=result['metrics'],costs=result['costs'],
                    phases=result['phases'],llm_failed=result['llm_failed'],audit=validation)
    except Exception as exc:
        error=dict(target=target['Structure ID'],status='failed',error=str(exc),traceback=traceback.format_exc(),
                   costs=runtime.budget.report()if runtime else {})
        atomic_json(out/'error.json',error);return error
    finally:
        if runtime:runtime.evidence.close()
        if chem:chem.close()


def aggregate(results,total):
    valid=[r for r in results if r['status']=='completed']
    metric=lambda k:sum(r['metrics'][k]for r in valid)
    full=metric('complete_yield_routes')
    return dict(cases=total,completed=len(valid),failed=len(results)-len(valid),
        stock_closed=metric('stock_closed'),strict_closed=metric('strict_closed'),
        mean_first_cc=metric('first_cc')/total,mean_all_routes_cc=metric('all_routes_mean_cc')/total,
        diversity=metric('diversity')/total,mean_routes=metric('routes')/total,
        mean_stock_routes=metric('stock_routes')/total,mean_strict_routes=metric('strict_routes')/total,
        complete_yield_routes=full,complete_yield_mean_pct=sum(r['metrics']['complete_yield_routes']*
            (r['metrics']['complete_yield_mean_pct']or 0.)for r in valid)/full if full else None,
        costs={k:sum(r.get('costs',{}).get(k,0)for r in results)for k in('native_calls','llm_calls',
            'evidence_calls','exact_queries','material_queries','support_queries','search_work','llm_input_tokens',
            'llm_output_tokens','elapsed_seconds')},
        audits_passed=sum(r['audit']['passed']for r in valid),llm_failed=sum(r.get('llm_failed',0)for r in valid))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--smiles');group.add_argument('--targets',type=Path)
    parser.add_argument('--id',default='sample');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--service',type=Path,required=True);parser.add_argument('--cc-service',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=10);parser.add_argument('--llm-concurrency',type=int,default=12)
    parser.add_argument('--blocked-targets-csv',type=Path)
    parser.add_argument('--no-yield',action='store_true')
    parser.add_argument('--config',type=Path);parser.add_argument('--workflow',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=Config(**json.loads(args.config.read_text()))if args.config else Config()
    if args.no_yield:cfg=replace(cfg,yield_reporting=False)
    if args.workflow:cfg=replace(cfg,llm_calls=0,phase_llm=0)
    if args.smoke:cfg=replace(cfg,native_calls=12,first_native=6,llm_calls=0,phase_llm=0,
        exact_queries=4,material_queries=4,support_queries=0,evidence_calls=8,search_work=1024,
        first_search=512,solver_calls=1,solver_nodes=128,solver_seconds=1.,wall_seconds=120.,yield_reporting=False)
    targets=json.loads(args.targets.read_text())if args.targets else[{'Structure ID':args.id,'SMILES':args.smiles}]
    targets=[{'Structure ID':str(t['Structure ID']),'SMILES':canon_side(t['SMILES'])}for t in targets]
    if not targets or len({t['SMILES']for t in targets})!=len(targets):parser.error('Nonempty unique molecules required')
    if len({t['Structure ID']for t in targets})!=len(targets)or any(not re.fullmatch(r'[A-Za-z0-9_-]+',t['Structure ID'])for t in targets):
        parser.error('Unique safe IDs required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    blocked={t['SMILES']for t in targets}
    if args.blocked_targets_csv:
        with args.blocked_targets_csv.open() as stream:
            blocked.update(canon_side(t['SMILES']) for t in csv.DictReader(stream))
    source=out/'source_snapshot';source.mkdir();manifest={}
    files = sorted(path for package in ('retroagent_langgraph', 'clean_harness', 'scripts')
                   for path in (ROOT / package).glob('*.py'))
    for path in files:
        dest=source/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    from importlib.metadata import version
    atomic_json(out/'protocol.json',dict(version='langgraph-0.1',targets=targets,config=asdict(cfg),
        source_manifest=manifest,blocked_products=sorted(blocked),model='gpt-6.1-sol',effort='medium',
        langgraph_version=version('langgraph'),smoke=args.smoke,mode='workflow'if args.workflow else'agent',
        initialization='Empty per-target graph and episode memory; no cross-case answer inputs',
        workers=args.workers,llm_concurrency=args.llm_concurrency,
        memory='Current rollout: depth budgets, context-specific failures, positive branches and actual provider outcomes'))
    atomic_json(out/'targets.json',targets)
    context=get_context('spawn');results=[];started=time.time()
    with context.Manager()as manager:
        gate=manager.BoundedSemaphore(args.llm_concurrency)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=context,max_tasks_per_child=1)as pool:
            pending={pool.submit(worker,(t,out/'cases'/t['Structure ID'],args.service.resolve(),args.cc_service.resolve(),
                sorted(blocked),cfg,gate)):t for t in targets}
            for future in as_completed(pending):
                result=future.result();results.append(result);atomic_json(out/'results.json',results)
                atomic_json(out/'progress.json',dict(completed=len(results),total=len(targets),elapsed_seconds=time.time()-started,
                    metrics=aggregate(results,len(results))))
                print(json.dumps(result,ensure_ascii=False),flush=True)
    summary=dict(elapsed_seconds=time.time()-started,metrics=aggregate(results,len(targets)))
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    changed=[name for name,sha in manifest.items()if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    atomic_json(out/'validation.json',dict(passed=not changed and all(
        r['status']=='completed'and r['audit']['passed']for r in results),source_changed=changed,
        cases=len(targets),completed=summary['metrics']['completed'],audits_passed=summary['metrics']['audits_passed']))


if __name__=='__main__':main()
