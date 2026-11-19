"""Bounded whole-route sketches: closure first, then whole-route CC."""
from collections import defaultdict
from functools import lru_cache
import heapq
from clean_harness.config import PROVIDERS
from clean_harness.portfolio import closed,strict


@lru_cache(maxsize=20000)
def burden(product):
    from rdkit import Chem
    # Structural effort proxy only. Exact inventory and actual route scores are
    # unchanged; a large unstocked precursor is not automatically an easy gap.
    return ((Chem.MolFromSmiles(product).GetNumHeavyAtoms()+3)/13.)**1.25

def inspect(root, bank, stock, edges, cfg):
    """Visit ALL required reactants. Reject cycles, inconsistent depth and size."""
    opened, active, visited, reachable = {}, set(), {}, set()
    def walk(p, depth):
        if depth > 6:
            raise ValueError("depth_limit")
        if p != root and stock[p]["in_stock"]:
            return
        if p in active:
            raise ValueError("cycle")
        if p not in bank:
            opened[p] = max(depth, opened.get(p, 0))
            return
        if visited.get(p, -1) >= depth:
            return
        visited[p] = depth
        reachable.add(p)
        active.add(p)
        for r in edges[bank[p]]["step"]["reactants"]:
            walk(r, depth + 1)
        active.remove(p)
    walk(root, 0)
    if len(reachable) > 16:
        raise ValueError("step_limit")
    normalized = {p: bank[p] for p in reachable}
    values = [max(0., edges[k]["cc"]) for k in normalized.values()]
    return dict(bank=normalized, opened=opened,
                mean_cc=sum(values) / len(values) if values else 0.,
                unsupported=sum(edges[k]["cc"] <= 0 for k in normalized.values()))


class RouteSearch:
    def __init__(self):
        self.states={};self.seen=set();self.work=0;self.cache_key=None
        self.served=defaultdict(int);self.log=[];self._expanded_cache=None

    @staticmethod
    def phase(engine):
        values=list(engine.metrics.values())
        return 'quality' if any(strict(m) for m in values) else 'strict' if any(closed(m) for m in values) else 'stock'

    def reusable(self,engine,state,p,depth):
        domain=engine.planner.stock if self.phase(engine)=='stock' else engine.planner.strict
        candidates=list(domain.plans.get(p,[]))
        if self.phase(engine)=='stock':candidates.extend(engine.planner.strict.plans.get(p,[]))
        options=[]
        for witness in candidates:
            if witness['height']+depth>6:continue
            conflict=False
            for q,step in witness['steps'].items():
                k=(step['product'],tuple(sorted(step['reactants'])))
                if q in state['bank'] and state['bank'][q]!=k:conflict=True;break
            if conflict or len(state['bank'].keys()|witness['steps'].keys())>16:continue
            options.append(witness)
        return sorted(options,key=lambda w:-w['mean'])[:2]

    def expandable_products(self,graph,positive):
        signature=len(graph.edges)
        if self._expanded_cache is None or self._expanded_cache[0]!=signature:
            self._expanded_cache=(signature,
                {e['step']['product']for e in graph.edges.values()},
                {e['step']['product']for e in graph.edges.values()if e['cc']>0})
        return self._expanded_cache[2 if positive else 1]

    def rank(self,engine,state):
        g=engine.graph;work=0.;actionable=[];stalled=False
        for p,depth in state['opened'].items():
            if depth>=6:stalled=True;continue
            if self.phase(engine)!='quality' and self.reusable(engine,state,p,depth):
                continue
            unused=[t for t in PROVIDERS if (p,t) not in g.attempts]
            reference=p not in engine.backend.blocked and p not in engine.retrieved
            # A fully expanded node with known continuations is not a dead end.
            # Keep unresolved AND siblings live while update traverses those
            # continuations. Only a node with no usable expansion blocks them.
            if not unused and not reference and p not in self.expandable_products(g,self.phase(engine)!='stock'):
                stalled=True
            if unused or reference:actionable.append(p)
            attempts=sum((p,t) in g.attempts for t in PROVIDERS)
            work+=burden(p)*(.75 if reference else 1.+.3*attempts)
        if stalled or not actionable:return None
        phase=self.phase(engine)
        if phase!='stock' and state['unsupported']:return None
        family=state['bank'].get(g.root)
        if phase=='quality':
            best=max((m['mean_cc'] for m in engine.metrics.values() if strict(m)),default=0.)
            n=len(state['bank']);unknown=len(state['opened'])
            # Optimistic ordering only; unknown reaction scores are NEVER used
            # for route evaluation, closure or reported mean CC.
            upper=(state['mean_cc']*n+5.*unknown)/max(1,n+unknown)
            priority=(-(upper-best)/max(1.,work),work,-state['mean_cc'],len(state['bank']))
        else:
            priority=(work,state['unsupported']>0,-state['mean_cc'],len(state['bank']))
        return priority,actionable

    def update(self,engine,quota=24):
        g=engine.graph
        signature=(len(g.edges),len(engine.metrics),len(g.attempts),len(engine.planner.stock.plans),
                   len(engine.planner.strict.plans),engine.cfg.search_work,self.phase(engine))
        if signature==self.cache_key:return []
        self.cache_key=signature
        reserve=max(0,engine.cfg.solver_nodes-engine.planner.nodes)
        available=max(0,engine.cfg.search_work-engine.budget.used['search_work']-reserve)
        cap=min(quota,available);used=0;new_complete=[]
        by=defaultdict(list)
        for k,e in g.edges.items():
            if e['step']['product']==g.root or not g.facts[e['step']['product']]['in_stock']:
                by[e['step']['product']].append(k)
        heap=[];serial=0;queued=set()
        def enqueue(state,k):
            nonlocal serial
            if k in queued:return
            queued.add(k)
            ranked=self.rank(engine,state)
            # Complete/exhausted sketches still need graph expansion, so give
            # them a separate structural priority instead of calling them dead chemistry.
            priority=ranked[0] if ranked else (float('inf'),state['unsupported'], -state['mean_cc'],len(state['bank']))
            phase=self.phase(engine)
            if not ranked and state['opened'] and any(by[p] for p in state['opened']) and (phase=='stock' or not state['unsupported']):
                if phase=='quality':
                    n=len(state['bank']);unknown=len(state['opened']);upper=(state['mean_cc']*n+5.*unknown)/max(1,n+unknown)
                    best=max((m['mean_cc'] for m in engine.metrics.values() if strict(m)),default=0.)
                    priority=(-(upper-best)/max(1,unknown),float(unknown),-state['mean_cc'],n)
                else:priority=(len(state['opened'])+.25,state['unsupported']>0,-state['mean_cc'],len(state['bank']))
            heapq.heappush(heap,(priority,serial,state));serial+=1
        def push(bank):
            nonlocal used
            k=tuple(sorted(bank.values()))
            if k in self.seen or used>=cap:return
            if not engine.budget.take('search_work'):return
            used+=1;self.work+=1;self.seen.add(k)
            try:state=inspect(g.root,bank,g.facts,g.edges,engine.cfg)
            except ValueError:return
            # An unstocked leaf at depth6 cannot be expanded within the contract.
            if any(d>=6 for d in state['opened'].values()):return
            normalized=tuple(sorted(state['bank'].values()))
            if normalized in self.states:return
            self.states[normalized]=state;enqueue(state,normalized)
            if not state['opened']:new_complete.append(state['bank'])
        for k,state in self.states.items():enqueue(state,k)
        for k in sorted(by[g.root],key=lambda k:(-g.edges[k]['cc'],k)):
            push({g.root:k})
        # Once closed, reopen holes in the best routes so quality work can
        # replace an internal weak step and score the resulting WHOLE route.
        winners=sorted((m for m in engine.metrics.values() if closed(m)),key=lambda m:(not strict(m),-m['mean_cc']))[:3]
        for metric in winners:
            bank={s['product']:(s['product'],tuple(sorted(s['reactants']))) for s in metric['route']['steps']}
            for p in bank:
                hole=dict(bank);hole.pop(p)
                if hole:push(hole)
        while heap and used<cap:
            _,_,state=heapq.heappop(heap)
            for p,depth in state['opened'].items():
                for witness in self.reusable(engine,state,p,depth):
                    bank=dict(state['bank'])
                    bank.update({q:(s['product'],tuple(sorted(s['reactants']))) for q,s in witness['steps'].items()})
                    push(bank)
                    if used>=cap:break
                if used>=cap:break
            if used>=cap:break
            expanded=[p for p in state['opened'] if by[p]]
            if not expanded:continue
            # All AND siblings remain in the state; only one is expanded at a time.
            p=min(expanded,key=lambda p:(len(by[p]),state['opened'][p],p))
            for k in sorted(by[p],key=lambda k:(-g.edges[k]['cc'],k)):
                push(dict(state['bank'],**{p:k}))
                if used>=cap:break
        return new_complete

    def candidates(self,engine):
        rows=[]
        for signature,state in self.states.items():
            ranked=self.rank(engine,state)
            if ranked:
                priority,products=ranked
                rows.append((priority,signature,state,products))
        return sorted(rows,key=lambda r:(r[0],r[1]))

    def targets(self,engine):
        g=engine.graph;rows=[];seen=set()
        for priority,signature,state,products in self.candidates(engine):
            family=state['bank'].get(g.root)
            for p in sorted(products,key=lambda p:(sum((p,t) in g.attempts for t in PROVIDERS),state['opened'][p],p)):
                unused=[t for t in PROVIDERS if (p,t) not in g.attempts]
                if p in seen or not unused:continue
                seen.add(p)
                rows.append(dict(product=p,node=g.node(p),tools=unused,root_family=family,
                    priority=priority,on_incumbent=False,weak_cc=None,route_mean_cc=state['mean_cc'],
                    unresolved=len(state['opened']),known_steps=len(state['bank'])))
                if len(rows)>=12:return rows
        return rows

    def compact(self,engine):
        g=engine.graph
        return [dict(phase=self.phase(engine),root_cut=g.edges[state['bank'][g.root]]['cut'],
            open_nodes=[g.node(p) for p in products],known_steps=len(state['bank']),
            known_mean_cc=round(state['mean_cc'],3),unsupported_steps=state['unsupported'],
            ordering_estimate=list(priority)) for priority,_,state,products in self.candidates(engine)[:3]
                if g.root in state['bank']]

    def evidence_targets(self,engine):
        seen=set();rows=[]
        for _,_,state,products in self.candidates(engine):
            for p in products:
                if p in seen or p in engine.backend.blocked or p in engine.retrieved:continue
                seen.add(p);rows.append(p)
                if len(rows)>=4:return rows
        return rows

    def pairs(self,engine):
        rows=self.targets(engine);pairs=[]
        for row in rows:
            if len(pairs)>=4:break
            tools=list(row['tools']);engine.rng.shuffle(tools)
            count=2 if not pairs and row['unresolved']==1 else 1
            for tool in tools[:count]:
                pairs.append((row['product'],tool));self.served[row['root_family']]+=1
                if len(pairs)>=4:break
        if pairs:self.log.append(dict(phase=self.phase(engine),queries=[dict(node=engine.graph.node(p),tool=t) for p,t in pairs]))
        return pairs
