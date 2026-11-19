"""Runtime support for the frozen V9 controller."""

import os

from concurrent.futures import ProcessPoolExecutor

import hashlib

import json

from multiprocessing import get_context

from multiprocessing.managers import BaseManager

from pathlib import Path

import secrets

import signal

import time

import uuid

from retrosynthesis_client import atomic_json, digest

ROOT=Path(__file__).resolve().parents[1]

SOURCE_DB=Path(os.environ.get('RETROAGENT_ASSETS', ROOT/'assets')).resolve()/'ChemCensor-DB-U2-1.0.0.sqlite'

def prepare_copy(out):
    out.mkdir(parents=True,exist_ok=False)
    before=SOURCE_DB.stat();dest=Path('/dev/shm')/('retroagent-cc-'+uuid.uuid4().hex+'.sqlite')
    h=hashlib.sha256();began=time.monotonic();size=0
    with SOURCE_DB.open('rb')as src,dest.open('xb')as dst:
        while block:=src.read(16*1024*1024):
            h.update(block);dst.write(block);size+=len(block)
    after=SOURCE_DB.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    copied=hashlib.sha256()
    with dest.open('rb')as f:
        while block:=f.read(16*1024*1024):copied.update(block)
    assert size==before.st_size and copied.hexdigest()==h.hexdigest()
    info=dict(source=str(SOURCE_DB),local=str(dest),bytes=size,sha256=h.hexdigest(),
        copy_seconds=time.monotonic()-began,same_source_bytes=True,source_mtime_ns=before.st_mtime_ns)
    atomic_json(out/'database_copy.json',info)
    return info

def initialize(db,folder):
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    torch.set_num_threads(1)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from chemcensor import ChemCensor
    from chemcensor.db.manager import DBManager
    import score_route_steps
    score_route_steps.CENSOR=ChemCensor(manager=DBManager.open_readonly(db))
    atomic_json(Path(folder)/(str(os.getpid())+'.json'),dict(pid=os.getpid(),db=db,ready=True))

def ping():return os.getpid()

class Backend:
    def __init__(self,out,info,workers):
        self.info=info;self.workers=workers;self.requests=0
        self.pool=ProcessPoolExecutor(max_workers=workers,initializer=initialize,
            initargs=(info['local'],str(out/'workers')),mp_context=get_context('spawn'))
        futures=[self.pool.submit(ping)for _ in range(workers)]
        for f in futures:f.result()
        deadline=time.monotonic()+180
        while len(list((out/'workers').glob('*.json')))<workers:
            if time.monotonic()>deadline:raise TimeoutError('CC worker warmup')
            time.sleep(.1)

    def health(self):
        return dict(ready=True,workers=self.workers,db=self.info,same_original_score_function=True,
            cross_request_score_cache=False)

    def batch(self,reactions):
        if not isinstance(reactions,list)or not 1<=len(reactions)<=64:raise ValueError('CC batch must be1..64')
        from score_route_steps import score
        self.requests+=len(reactions)
        return list(self.pool.map(score,reactions))

class Manager(BaseManager):pass

class Client:
    def __init__(self,spec):
        self.spec=json.loads((Path(spec)/'ready.json').read_text())
        Manager.register('scorer')
        self.manager=Manager(address=self.spec['socket'],authkey=Path(self.spec['key_file']).read_bytes())
        self.manager.connect();self.remote=self.manager.scorer()

class ScoringClient:
    """Function client adapter; native/stock/retrieval delegate to original client."""
    def __init__(self,client,cc_service):
        self.client=client;self.cc=Client(cc_service)
        if not self.cc.remote.health()['ready']:raise RuntimeError('RAM CC service unavailable')

    def __getattr__(self,name):return getattr(self.client,name)

    def call(self,name,arguments,call_id=None):
        if name!='score_reactions':return self.client.call(name,arguments,call_id)
        from resident_function_tools_v1 import validate_call
        arguments=validate_call(name,arguments);began=time.perf_counter()
        record=dict(tool=name,arguments=arguments,call_id=call_id or uuid.uuid4().hex,
            session=self.client.session,request_key=digest(dict(name=name,arguments=arguments)),
            execution='resident_ram_CC12',source_db_sha256=self.cc.spec['db']['sha256'])
        try:record.update(status='completed',result=self.cc.remote.batch(arguments['reactions']))
        except Exception as exc:record.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)))
        record['total_seconds']=time.perf_counter()-began
        return record

def serve(out,workers):
    info=prepare_copy(out);backend=Backend(out,info,workers)
    key=out/'private.key'
    with os.fdopen(os.open(key,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'wb')as f:f.write(secrets.token_bytes(32))
    Manager.register('scorer',callable=lambda:backend,exposed=['health','batch'])
    socket='/tmp/retroagent-ram-cc-'+uuid.uuid4().hex[:14]+'.sock'
    server=Manager(address=socket,authkey=key.read_bytes()).get_server()
    def stop(sig,frame):raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop)
    atomic_json(out/'ready.json',dict(pid=os.getpid(),socket=socket,key_file=str(key),db=info,workers=workers))
    print(json.dumps(dict(ready=True,pid=os.getpid(),db=info,workers=workers)),flush=True)
    try:server.serve_forever()
    finally:
        backend.pool.shutdown(wait=True)
        atomic_json(out/'stopped.json',dict(requests=backend.requests))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Serve ChemCensor scores from a verified local database.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    serve(args.out, args.workers)
