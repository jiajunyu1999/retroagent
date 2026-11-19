"""Runtime support for the frozen V9 controller."""

import hashlib

import json

from pathlib import Path

import tempfile

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def atomic_json(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');tmp=Path(f.name)
    tmp.replace(path)
