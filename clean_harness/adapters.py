"""I/O boundary only: unchanged resident models, original CC/stock/evaluation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from divergent_resident_adapter_v1 import Evaluator
from resident_logged_tools_v2 import Tools
from resident_ram_cc_v86 import ScoringClient
from evaluate_domain_models_test100 import route_eval
from strict_source_audit_v1 import screen
from retrosynthesis_client import atomic_json
from .planner import make_route


class Backend:
    def __init__(self, service, cc_service, out, target, blocked, yield_reporting):
        self.out, self.target, self.blocked = Path(out),target,set(blocked)
        self.tools = Tools(service,self.out/'tools')
        if not self.tools.client.remote.health()['ready']:
            raise RuntimeError('Native service is unhealthy')
        if cc_service:
            self.tools.client = ScoringClient(self.tools.client,cc_service)
            expected = '458dc238f6a6b0f8af2cf8570e8760f18e6d6bb05855c55c42d33d245c81dc32'
            if self.tools.client.cc.spec['db']['sha256'] != expected:
                raise ValueError('CC reference DB differs from the pinned comparison database')
        self.ev = Evaluator(self.tools,self.out/'evaluation')
        self.yields = None
        if yield_reporting:
            from .yields import YieldEvidence
            self.yields = YieldEvidence(self.ev.scored,self.blocked)
        self.rejections = []

    def facts(self, graph):
        missing = sorted(set(graph.nodes)-graph.facts.keys())
        if missing:
            graph.facts.update(self.ev.stock.query(missing))

    def prepare(self, graph, rows, product, origin, *, patch=False, sources=()):
        prepared = {}
        for row in rows:
            try:
                s = graph.normalize(row,None if patch else product)
                from .graph import key
                prepared[key(s)] = s
            except (ValueError,KeyError,TypeError) as exc:
                self.rejections.append(dict(origin=origin,reason=str(exc),step=row))
        if patch:
            # Follow ALL alternatives for a product, not a single overwritten parent.
            reachable, todo = set(),[product]
            while todo:
                p = todo.pop()
                if p in reachable:
                    continue
                reachable.add(p)
                for s in prepared.values():
                    if s['product'] == p:
                        todo.extend(s['reactants'])
            prepared = {k:s for k,s in prepared.items() if s['product'] in reachable}
        steps = list(prepared.values())
        scores = self.ev.batch_scores(['.'.join(s['reactants'])+'>>'+s['product'] for s in steps])
        added = sum(graph.add(s,d['score_with_fg'],origin,sources) for s,d in zip(steps,scores))
        self.facts(graph)
        return dict(admitted=len(steps),supported=sum(d['score_with_fg']>0 for d in scores),added=added)

    def predict(self, pairs):
        def call(pair):
            p,t = pair
            began = time.monotonic()
            try:
                raw = self.tools.call('predict_reactions',dict(provider=t,products=[p]))
                from prepare_yield_distributions import canon_side
                if len(raw['results']) != 1 or canon_side(raw['results'][0]['product']) != p:
                    raise ValueError('Native response product mismatch')
                return p,t,raw['results'][0]['predictions'][:10],dict(status='completed',seconds=time.monotonic()-began)
            except Exception as exc:
                return p,t,[],dict(status='failed',error=str(exc),seconds=time.monotonic()-began)
        # The provider holds session RNG/buffer state. Preserve the submitted
        # order within a provider; independent providers still run concurrently.
        groups={}
        for pair in pairs:groups.setdefault(pair[1],[]).append(pair)
        def run_group(group):return [call(pair) for pair in group]
        with ThreadPoolExecutor(max_workers=min(5,len(groups))) as pool:
            rows=[row for block in pool.map(run_group,groups.values()) for row in block]
        by_pair={(p,t):(p,t,result,record) for p,t,result,record in rows}
        return [by_pair[pair] for pair in pairs]

    def retrieve(self, graph, p):
        if p in self.blocked:
            raise ValueError('Direct benchmark target retrieval is forbidden')
        rows = self.tools.call('retrieve_reactions',dict(product=p,limit=8))
        admitted = []
        for row in rows:
            if isinstance(row.get('document_id'),str) and row['document_id']:
                admitted.append(row)
            else:
                self.rejections.append(dict(origin='public_CC_exact',reason='Missing document witness'))
        return self.prepare(graph,admitted,p,'public_CC_exact')

    def evaluate(self, graph, steps, tid):
        route = make_route(graph.root,steps,tid)
        return screen(route_eval(self.ev,self.target,route,'clean-1'))

    def close(self):
        atomic_json(self.out/'rejected.json',self.rejections)
        self.ev.close()
        if self.yields:
            self.yields.close()
