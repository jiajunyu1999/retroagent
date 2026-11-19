"""Worker cannot read historical reports, sibling results, or the supervisor's memory DB."""
import json
import os
from pathlib import Path
import sys
import threading


def install(out, service, cc_service, assets, memory_path):
    root = Path(__file__).resolve().parents[1]
    out = Path(out).resolve()
    files = {Path(p).resolve() for p in assets}
    for specdir in (service,cc_service):
        if specdir:
            ready = Path(specdir).resolve()/'ready.json'
            files.update((ready,Path(json.loads(ready.read_text())['key_file']).resolve()))
    forbidden = {Path(str(memory_path)+suffix).resolve() for suffix in ('','-wal','-shm','-journal')} if memory_path else set()
    local = threading.local()
    log = os.open(out/'boundary.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    seen = set()
    def audit(event,args):
        if event not in ('open','sqlite3.connect') or getattr(local,'busy',False):
            return
        if not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        local.busy = True
        try:
            name = os.fsdecode(args[0])
            if name.startswith('file:'):
                name = name[5:].split('?',1)[0]
            p = Path(name).resolve()
            allowed = p not in forbidden
            if p.is_relative_to(out.parent) and not p.is_relative_to(out):
                allowed = False
            if p.is_relative_to(root/'reports'):
                allowed &= p.is_relative_to(out) or p in files
            if p.is_relative_to(root/'reports') or not allowed:
                if (str(p),allowed) not in seen:
                    seen.add((str(p),allowed))
                    os.write(log,(json.dumps(dict(event=event,path=str(p),allowed=bool(allowed)))+'\n').encode())
            if not allowed:
                raise PermissionError('Generation read boundary: '+str(p))
        finally:
            local.busy = False
    sys.addaudithook(audit)
