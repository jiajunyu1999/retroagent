"""Runtime support for the frozen V9 controller."""

from pathlib import Path

from resident_function_tools_v1 import Client

from retrosynthesis_client import atomic_json

import threading

class Tools:
    def __init__(self,service,out):
        self.client=Client(service);self.out=Path(out);self.records=[];self.lock=threading.Lock()

    def call(self,name,arguments):
        record=self.client.call(name,arguments)
        file=self.out/'function_results'/(record['call_id']+'.json')
        atomic_json(file,record)
        with self.lock:
            self.records.append({k:record[k] for k in ['call_id','tool','status','total_seconds']})
            self.records[-1]['record']=str(file)
            atomic_json(self.out/'function_calls.json',self.records)
        if record['status']!='completed':raise RuntimeError(record.get('error'))
        return record['result']
