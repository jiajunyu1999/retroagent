"""Size-aware AND witnesses plus bounded mean-CC MILP; no model decisions here."""
from collections import defaultdict
from itertools import product
from .graph import key
from .optimizer import mean_cc_plan


def retain(rows, keep):
    bank = {r['signature']:r for r in rows}
    buckets = defaultdict(list)
    for r in bank.values():
        buckets[r['steps_count'],r['height']].append(r)
    representatives = [max(v,key=lambda r:(r['mean'],r['minimum'],r['signature'])) for _,v in sorted(buckets.items())]
    limit = max(keep, min(64,len(representatives)))
    chosen = {r['signature']:r for r in representatives[:limit]}
    for r in sorted(bank.values(),key=lambda r:(-r['mean'],-r['minimum'],r['steps_count'],r['signature'])):
        if len(chosen) >= limit:
            break
        chosen.setdefault(r['signature'],r)
    return list(chosen.values())


class Witnesses:
    """Persistent within-case subgoal memory. Every distinct AND join charged once."""
    def __init__(self, strict):
        self.strict, self.plans, self.seen = strict, {}, set()

    def update(self, graph, budget, quota=512):
        empty = dict(steps={},signature=(),height=0,mean=0.,minimum=5.,steps_count=0)
        for p,f in graph.facts.items():
            if p != graph.root and f['in_stock']:
                self.plans[p] = [empty]
        joined = 0
        for _ in range(6):
            changed = False
            for k,e in sorted(graph.edges.items()):
                s,p,rs = e['step'],e['step']['product'],e['step']['reactants']
                if self.strict and e['cc'] <= 0 or p != graph.root and graph.facts[p]['in_stock']:
                    continue
                if not all(r in self.plans for r in rs):
                    continue
                for children in product(*(list(self.plans[r]) for r in rs)):
                    token = (k,tuple(c['signature'] for c in children))
                    if token in self.seen:
                        continue
                    if joined >= quota or not budget.take('search_work'):
                        return
                    self.seen.add(token)
                    joined += 1
                    merged, bad = {}, False
                    for c in children:
                        if p in c['steps']:
                            bad = True
                            break
                        for q,step in c['steps'].items():
                            if q in merged and key(merged[q]) != key(step):
                                bad = True
                                break
                            merged[q] = step
                        if bad:
                            break
                    if bad:
                        continue
                    merged[p] = s
                    height = 1+max(c['height'] for c in children)
                    if height > 6 or len(merged) > 16:
                        continue
                    signature = tuple(sorted(key(s) for s in merged.values()))
                    values = [graph.edges[k]['cc'] for k in signature]
                    row = dict(steps=merged,signature=signature,height=height,mean=sum(values)/len(values),
                               minimum=min(values),steps_count=len(merged))
                    old = self.plans.get(p,[])
                    new = retain([*old,row],32 if p == graph.root else 12)
                    if {r['signature'] for r in old} != {r['signature'] for r in new}:
                        self.plans[p] = new
                        changed = True
            if not changed:
                break


class Planner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strict, self.stock = Witnesses(True), Witnesses(False)
        self.proofs, self.nodes, self.seconds, self.solved_signatures = [],0,0.,set()

    def search(self, graph, budget, *, solve=False, final=False):
        # Reserve branch nodes for final exact recombination instead of consuming everything in joins.
        reserve = max(0,self.cfg.solver_nodes-self.nodes)
        available = max(0,self.cfg.search_work-budget.used['search_work']-reserve)
        quota = min(1024 if final else 256,available//2)
        self.strict.update(graph,budget,quota)
        self.stock.update(graph,budget,quota)
        rows = [r['steps'] for domain in (self.strict,self.stock) for r in domain.plans.get(graph.root,[])]
        signature = tuple(sorted(graph.edges))
        max_calls = self.cfg.solver_calls-(0 if final else 1)
        max_nodes = self.cfg.solver_nodes-(0 if final else min(256,self.cfg.solver_nodes))
        if solve and signature not in self.solved_signatures and len(self.proofs)<max_calls and self.nodes<max_nodes:
            self.solved_signatures.add(signature)
            remaining = min(max_nodes-self.nodes,self.cfg.search_work-budget.used['search_work'])
            seconds = min(2.,self.cfg.solver_seconds-self.seconds,budget.remaining_seconds)
            if remaining > 0 and seconds > .02:
                best,proof = mean_cc_plan(graph.root,graph.edges,graph.facts,seconds=seconds,node_limit=remaining)
                self.nodes += proof['branch_nodes']
                self.seconds += proof['seconds']
                # Work has already occurred: record even if wall time elapsed during a solve.
                budget.used['search_work'] += proof['branch_nodes']
                self.proofs.append(proof)
                if best:
                    rows.append({e['step']['product']:e['step'] for e in best['edges']})
        return rows

    @property
    def solved(self):
        return {p for p,rows in self.strict.plans.items() if rows}


def make_route(root, steps, tid):
    return dict(target_id=tid,target_smiles=root,steps=[dict(s,step_id='R%02d'%i)
                for i,(_,s) in enumerate(sorted(steps.items()),1)])
