"""LangGraph nodes own the actual search loop, not a wrapper around an old agent."""
from dataclasses import replace
from collections import Counter
import json
from pathlib import Path
import random
from threading import Semaphore
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from clean_harness.config import PROVIDERS
from clean_harness.graph import Graph, key
from clean_harness.planner import Planner
from .route_search import RouteSearch,burden
from clean_harness.portfolio import select, identity, closed, strict, rank, diversity
from retrosynthesis_client import atomic_json
from .config import Budget
from .evidence import EvidenceTools
from .memory import EpisodeMemory
from .routes import prepare, route_id, fragments
from . import reasoning


class State(TypedDict,total=False):
    phase:int
    cycle:int
    next:str
    proposals:int
    done:bool


class Runtime:
    def __init__(self,target,out,chemistry,cfg,center_db,llm_gate=None):
        self.target,self.tid,self.out=target,target['Structure ID'],Path(out)
        self.chem=chemistry;self.full=cfg;self.cfg=replace(cfg,native_calls=cfg.first_native,
            llm_calls=min(cfg.phase_llm,cfg.llm_calls),search_work=cfg.first_search)
        self.budget=Budget(cfg);self.budget.cfg=self.cfg
        self.graph=Graph(target['SMILES']);self.planner=Planner(self.cfg);self.guide=RouteSearch()
        self.memory=EpisodeMemory();self.metrics={};self.evaluated=set()
        self.evidence=EvidenceTools(chemistry,self.graph,self.budget,out,cfg.literature_cutoff,center_db)
        self.backend=chemistry;self.retrieved=self.evidence.exact_seen
        self.rng=random.Random(cfg.seed);self.llm_gate=llm_gate or Semaphore(3)
        self.llm_records=[];self.errors=[];self.pending=[];self.responses=[];self.tasks=[]
        self.pending_exact=[];self.quality_groups=0;self.round=0;self.last_feedback=0
        self.phase=1;self.first_phase=None;self.reason='budget_exhausted';self.stock_target=False
        self.bases={};self.current_state={};self.after_evidence='combine'
        self.reason_was_strict=False
        self.rule_streams={};self.rule_pending={};self.rule_seen=set();self.rule_work=0
        self.rule_exhausted=set()
        self.route_visits=Counter()
        self.salvaged_steps=0

    def quality(self):
        best=max(self.metrics.values(),key=rank,default=None)
        return dict(stock_closed=any(closed(m)for m in self.metrics.values())or self.stock_target,
                    strict_closed=any(strict(m)for m in self.metrics.values())or self.stock_target,
                    first_cc=best['mean_cc']if best else 0.)

    def trace(self,node,state):
        with(self.out/'graph_events.jsonl').open('a')as f:
            f.write(json.dumps(dict(node=node,phase=self.phase,cycle=state.get('cycle',0),
                quality=self.quality(),used=self.budget.used))+'\n')

    def evaluate(self,steps):
        signature=tuple(sorted(key(s)for s in steps.values()))
        if not signature or signature in self.evaluated:return
        self.evaluated.add(signature)
        metric=self.chem.evaluate(self.graph,steps,self.tid)
        if metric['structural_valid']and not metric.get('source_issues'):
            self.metrics[identity(metric)]=metric
            self.bases[route_id(metric['route'])]=metric['route']
            if closed(metric)and not strict(metric):
                self.memory.observe(metric,self.round,'graph',self.chem.ev.scored)

    def initialize(self,state):
        self.chem.facts(self.graph);self.stock_target=self.graph.facts[self.graph.root]['in_stock']
        if not self.stock_target:self.predict([(self.graph.root,t)for t in PROVIDERS])
        self.trace('initialize',state)
        return dict(phase=1,cycle=0,next='combine')

    def predict(self,pairs):
        admitted=[]
        for p,t in pairs:
            if t not in PROVIDERS or p not in self.graph.nodes or (p,t)in self.graph.attempts:continue
            if not self.budget.take('native_calls'):break
            self.graph.attempts[p,t]=dict(status='pending');admitted.append((p,t))
        if not admitted:return
        for p,t,rows,record in self.chem.predict(admitted):
            if record['status']=='completed':
                try:record.update(self.chem.prepare(self.graph,rows,p,'native:'+t))
                except Exception as exc:record.update(status='failed',error=str(exc))
            self.graph.attempts[p,t]=record
            self.memory.observe_tool(p,t,record,self.graph,self.planner)
            if record['status']!='completed':self.errors.append(dict(tool=t,product=p,error=record.get('error')))
            self.evidence.exact(p)
        # Actual root alternatives are connected partial routes, not fake closure.
        for edge in self.graph.edges.values():
            if edge['step']['product']==self.graph.root:self.evaluate({self.graph.root:edge['step']})

    def drain_exact(self,limit=4):
        queue,self.pending_exact=self.pending_exact,[];used=0
        for index,p in enumerate(queue):
            if self.evidence.exact(p):used+=1
            if used>=limit:
                self.pending_exact=queue[index+1:];break

    def combine(self,state):
        if not self.stock_target:
            plans=self.planner.search(self.graph,self.budget,
                solve=self.budget.used['native_calls']%24==0)
            for steps in plans:self.evaluate(steps)
            self.guide.update(self,quota=192)
            # Keep explicit positive partials so a later stock-only hypothesis
            # cannot erase the memory of an almost finished strict branch.
            states=[s for s in self.guide.states.values()if not s['unsupported']]
            states.sort(key=lambda s:(len(s['opened']),sum(len(p)for p in s['opened']),-s['mean_cc']))
            for row in states[:8]:
                self.evaluate({p:self.graph.edges[k]['step']for p,k in row['bank'].items()})
        self.trace('combine',state);return dict(next='remember')

    def remember(self,state):
        # Actual positive contexts help interpret a failed center without
        # treating it as a universal ban. Scoring facts are already cached.
        for edge in self.graph.edges.values():
            if edge['cc']>0:self.memory.family(edge['step'],self.chem.ev.scored)
        for m in self.metrics.values():
            if all(d['score_with_fg']>0 for d in m['step_diagnostics']):
                self.memory.protected[route_id(m['route'])]=m
        atomic_json(self.out/'working_memory.json',self.memory.snapshot())
        atomic_json(self.out/'progress.json',dict(phase=self.phase,round=self.round,
            quality=self.quality(),costs=self.budget.report(),nodes=len(self.graph.nodes),cuts=len(self.graph.edges)))
        self.trace('remember',state);return dict(next='schedule')

    def tool_pairs(self,*,consume=True):
        limit=min(4,self.cfg.native_calls-self.budget.used['native_calls'])
        if limit<=0:return []
        closed_metrics=[m for m in self.metrics.values()if closed(m)]
        if any(strict(m)for m in closed_metrics):closed_metrics=[m for m in closed_metrics if strict(m)]
        frontier=self.graph.frontier(self.planner.solved,closed_metrics)
        # Unrelated observed inventory hits may change node numbering. They
        # must not change which chemical gap wins an otherwise equal priority.
        frontier.sort(key=lambda row:(row['priority'][:-1],row['product']),reverse=True)
        available={r['product']:r for r in frontier};pairs=[]
        # One slot for measured LLM decisions; two for complete positive native
        # branches; one for alternative positive branches. Never pursue a union
        # of unrelated dependencies as if it were a feasible AND route.
        queue=self.pending if consume else list(self.pending)
        while queue and len(pairs)<min(1,limit):
            p,t=queue.pop(0)
            if p in available and t in available[p]['tools']and(p,t)not in pairs:pairs.append((p,t))
        if len(pairs)<min(1,limit):
            if self.quality()['strict_closed']:
                best=max((m for m in self.metrics.values()if strict(m)),key=rank)
                goals=[s['product']for s,d in sorted(zip(best['route']['steps'],best['step_diagnostics']),
                                                      key=lambda pair:pair[1]['score_with_fg'])]
            else:
                contracts=self.memory.contracts(self.metrics.values(),self.graph,self.planner,self.chem.ev.scored)
                goals=[r['task']['repair_anchor']for r in contracts]
            for p in goals:
                if p in available and not any(q==p for q,_ in pairs):
                    pairs.append((p,self.rng.choice(available[p]['tools'])))
                if len(pairs)>=min(1,limit):break
        candidates=self.guide.candidates(self)
        quality_mode=self.quality()['strict_closed']
        if quality_mode:candidates.sort(key=lambda row:(row[0],row[1]))
        else:candidates.sort(key=lambda row:(self.route_visits[row[2]['bank'].get(self.graph.root)]//2,row[0],row[1]))
        families=Counter()
        lanes=((False,limit),)if quality_mode else((True,min(3,limit)),(False,limit))
        for native_only,stop in lanes:
            for _,_,branch,products in candidates:
                if len(pairs)>=stop:break
                family=branch['bank'].get(self.graph.root)
                family_limit=2 if quality_mode else 1
                if family is None or families[family]>=family_limit or branch['unsupported']:continue
                if native_only and any(not any(o.startswith('native:')or o.startswith('reference:')
                    for o in self.graph.edges[k]['origins'])for k in branch['bank'].values()):continue
                goals=sorted(products,key=lambda p:(sum((p,t)in self.graph.attempts for t in PROVIDERS),
                    burden(p),-branch['opened'][p],p))
                for p in goals:
                    if p not in available or any(q==p for q,_ in pairs):continue
                    pairs.append((p,self.rng.choice(available[p]['tools'])));families[family]+=1
                    if consume:self.route_visits[family]+=1
                    if not quality_mode or families[family]>=family_limit or len(pairs)>=stop:break
        reached={self.graph.root}
        for _ in range(6):
            added={r for e in self.graph.edges.values()if e['cc']>0 and
                   any(o.startswith('native:')for o in e['origins'])and e['step']['product']in reached
                   for r in e['step']['reactants']}
            if added<=reached:break
            reached.update(added)
        protected=[r for r in frontier if r['product']in reached]
        ordered=protected+[r for r in frontier if r['product']not in reached]
        for row in ordered:
            if len(pairs)>=limit:break
            if not any(p==row['product']for p,_ in pairs):
                pairs.append((row['product'],self.rng.choice(row['tools'])))
        return pairs

    def schedule(self,state):
        if self.stock_target or self.budget.remaining_seconds<=0:
            self.reason='target_in_stock'if self.stock_target else 'wall_limit';return dict(next='finish')
        pairs=self.tool_pairs(consume=False);native=self.budget.used['native_calls'];used=self.budget.used['llm_calls']
        is_strict=self.quality()['strict_closed']
        due=native>=self.full.warmup_native and (native-self.last_feedback>=self.full.feedback_every
            or not pairs or native>=self.cfg.native_calls)
        if is_strict:
            due=(self.quality_groups<self.full.quality_llm_groups and self.quality()['first_cc']<3.
                 and native-self.last_feedback>=self.full.feedback_every)
        if due and used<self.cfg.llm_calls:
            self.reason_was_strict=is_strict
            self.tasks=self.memory.tasks(self.metrics.values(),self.graph,self.planner,self.chem.ev.scored,self.round+1)
            self.after_evidence='reason';return dict(next='evidence')
        if pairs and native<self.cfg.native_calls:
            self.pending_batch=self.tool_pairs();return dict(next='tools')
        if self.phase==1:return dict(next='advance')
        if not is_strict and used<self.cfg.llm_calls:
            self.reason_was_strict=False
            self.tasks=self.memory.tasks(self.metrics.values(),self.graph,self.planner,self.chem.ev.scored,self.round+1)
            self.after_evidence='reason';return dict(next='evidence')
        return dict(next='finish')

    def tools(self,state):
        self.predict(self.pending_batch);self.trace('tools',state)
        return dict(cycle=state.get('cycle',0)+1,next='combine')

    def material_focus(self):
        from .evidence import fragments as parts
        anchors=[t['repair_anchor']for t in self.tasks if t.get('repair_anchor')]
        contracts=self.memory.contracts(self.metrics.values(),self.graph,self.planner,self.chem.ev.scored)
        gaps=[p for row in contracts for p in row['gaps']]
        active=[r['product']for r in self.guide.targets(self)[:6]]
        roots=parts(self.graph.root)[:6]
        local=[p for anchor in anchors[:2]for p in parts(anchor)[:2]]
        return [p for p in dict.fromkeys(anchors+roots[:2]+gaps+active+roots[2:]+local)
                if not self.graph.facts.get(p,{}).get('in_stock')]

    def evidence_node(self,state):
        if self.after_evidence=='reason':
            for task in self.tasks[:2]:
                p=task.get('repair_anchor')
                if p:self.evidence.exact(p)
            used=0
            for p in self.material_focus():
                before=self.budget.used['material_queries'];self.evidence.material(p)
                used+=self.budget.used['material_queries']-before
                if used>=6 or self.budget.used['material_queries']>=self.full.material_queries:break
            broken=max((m for m in self.metrics.values()if closed(m)and not strict(m)),key=rank,default=None)
            if broken:
                for s,d in zip(broken['route']['steps'],broken['step_diagnostics']):
                    if d['score_with_fg']<=0:self.evidence.analog(s);break
        else:
            # A blocked/cached/stock product consumes no query slot. Do not
            # drop the later, genuinely new intermediates from the proposal.
            self.drain_exact(4)
        self.trace('evidence',state);return dict(next='refresh'if self.after_evidence=='reason'else'repair')

    def refresh(self,state):
        self.combine(state)
        if self.quality()['strict_closed']and not self.reason_was_strict:
            self.trace('refresh',state);return dict(next='remember')
        self.trace('refresh',state);return dict(next='reason')

    def rule_focus(self):
        from .rules import formed_cuts,direct_supply_kind
        from rdkit import Chem
        focus=[]
        passing=max((m for m in self.metrics.values()if strict(m)),key=rank,default=None)
        if passing:
            steps=[s for s,d in sorted(zip(passing['route']['steps'],passing['step_diagnostics']),
                key=lambda pair:pair[1]['score_with_fg'])[:2]]
        else:
            broken=max((m for m in self.metrics.values()if closed(m)),key=rank,default=None)
            steps=[s for s,d in zip(broken['route']['steps'],broken['step_diagnostics'])if d['score_with_fg']<=0][:1]if broken else[]
        for step in steps:
            raw=self.chem.ev.scored('.'.join(step['reactants'])+'>>'+step['product'])
            focus.append((step['product'],formed_cuts(step,raw)))
        if not passing:
            # Stock-near suppliers of any currently positive construction must
            # remain eligible even when absent from the compact LLM summary.
            reached={self.graph.root}
            for _ in range(6):
                added={p for edge in self.graph.edges.values()if edge['cc']>0
                    and edge['step']['product']in reached for p in edge['step']['reactants']}
                if added<=reached:break
                reached.update(added)
            suppliers=[p for p in reached if p!=self.graph.root
                and not self.graph.facts.get(p,{}).get('in_stock')and p not in self.planner.solved
                and direct_supply_kind(p)]
            suppliers.sort(key=lambda p:(direct_supply_kind(p)!='boron',Chem.MolFromSmiles(p).GetNumHeavyAtoms(),p))
            focus.extend((p,[])for p in suppliers)
            contracts=self.memory.contracts(self.metrics.values(),self.graph,self.planner,self.chem.ev.scored)
            gaps=list(dict.fromkeys(p for row in contracts for p in row['gaps']))
            def priority(product):
                mol=Chem.MolFromSmiles(product)
                return (not any(a.GetAtomicNum()==5 for a in mol.GetAtoms()),
                        product in self.planner.solved,mol.GetNumHeavyAtoms(),product)
            focus.extend((p,[])for p in sorted(gaps,key=priority))
        unique={}
        for product,preferred in focus:
            if product not in self.rule_exhausted:unique.setdefault(product,preferred)
        return list(unique.items())[:3]

    def repair(self,state):
        if self.budget.used['native_calls']<24 or self.rule_work>=self.full.rule_work:
            return dict(next='combine')
        from .rules import core_proposals
        active=[]
        for product,preferred in self.rule_focus():
            if product in self.rule_exhausted:continue
            if product not in self.rule_streams:self.rule_streams[product]=iter(core_proposals(product,preferred))
            active.append(product)
        cap=min(24,self.full.rule_work-self.rule_work);spent=0;scanned=Counter();exhausted=False
        while active and spent<cap and not exhausted:
            progressed=False
            for product in list(active):
                if spent>=cap:break
                while scanned[product]<64:
                    scanned[product]+=1
                    try:proposal=self.rule_pending.pop(product)if product in self.rule_pending else next(self.rule_streams[product])
                    except StopIteration:
                        self.rule_exhausted.add(product);active.remove(product);break
                    except (ValueError,RuntimeError)as exc:
                        self.errors.append(dict(tool='rules',error=str(exc)));active.remove(product);break
                    token=tuple((step['product'],tuple(sorted(step['reactants'])))for step in proposal['steps'])
                    if token in self.rule_seen:continue
                    cost=max(1,len(proposal['steps']))
                    if spent+cost>cap:
                        self.rule_pending[product]=proposal;break
                    if not self.budget.take('search_work',cost):
                        self.rule_pending[product]=proposal;exhausted=True;break
                    self.rule_seen.add(token);self.rule_work+=cost;spent+=cost;progressed=True
                    required=proposal.get('requires_stock',[])
                    if required and not all(v['in_stock']for v in self.chem.ev.stock.query(required).values()):break
                    self.chem.prepare(self.graph,proposal['steps'],product,'rule:context',patch=True)
                    break
                if exhausted:break
            if not progressed:break
        self.trace('repair',state);return dict(next='combine')

    def observation(self,tasks):
        chosen=[];seen=set()
        for task in tasks:
            rid=task.get('base_route_id')
            if rid and rid not in seen:
                route=self.bases[rid];m=self.metrics[identity(dict(route=route))]
                chosen.append(reasoning.brief(m));seen.add(rid)
        for m in sorted(self.metrics.values(),key=rank,reverse=True):
            if len(chosen)>=5:break
            rid=route_id(m['route'])
            if rid not in seen:chosen.append(reasoning.brief(m));seen.add(rid)
        wanted={self.graph.root,*[t['repair_anchor']for t in tasks if t.get('repair_anchor')]}
        options=[];certificates=[]
        for p in sorted(wanted):
            edges=sorted((e for e in self.graph.edges.values()if e['step']['product']==p),key=lambda e:-e['cc'])[:4]
            options.append(dict(product=p,unused_providers=[t for t in PROVIDERS if(p,t)not in self.graph.attempts],
                attempts={t:self.graph.attempts[p,t]['status']for t in PROVIDERS if(p,t)in self.graph.attempts},
                actual_options=[dict(step=e['step'],cc=e['cc'],origins=e['origins'],
                    stock={r:self.graph.facts[r]['in_stock']for r in e['step']['reactants']})for e in edges]))
            plans=self.planner.strict.plans.get(p,[])
            if plans and any(r['steps']for r in plans):
                best=max(plans,key=lambda r:r['mean'])
                certificates.append(dict(product=p,in_stock=self.graph.facts[p]['in_stock'],
                    mean_cc=best['mean'],steps=list(best['steps'].values())))
        budget=self.budget.report();budget.pop('elapsed_seconds',None)
        return dict(routes=chosen,working_memory=self.memory.context(self.metrics.values(),self.graph,
            self.planner,self.chem.ev.scored),quality=self.quality(),budget=budget,
            observed_supply_options=options,positive_supply_certificates=certificates,
            materials=self.evidence.context(wanted),
            analog_precedents=self.evidence.analogs[-4:],
            literature_sources=list(self.evidence.literature.sources.values())[-4:],
            route_frontiers=self.guide.compact(self),
            queued_tool_requests=[dict(product=p,provider=t)for p,t in self.pending[:8]],
            attempted_tools=[dict(product=p,provider=t,status=r['status'])for(p,t),r in list(self.graph.attempts.items())[-16:]])

    def reason_node(self,state):
        self.round+=1;self.last_feedback=self.budget.used['native_calls']
        is_strict=self.quality()['strict_closed']
        if is_strict:self.quality_groups+=1
        self.responses=reasoning.run(self,self.tasks,views=3)
        for _,_,record in self.responses:self.llm_records.append(record)
        self.trace('reason',state);return dict(proposals=len(self.responses),next='verify')

    def verify_chemistry(self,value,task,role):
        if not isinstance(value,dict):raise ValueError('Typed candidate object required')
        if not value.get('steps'):return
        if value.get('source_ids'):self.evidence.literature.validate_suggestion(value)
        before=len(self.graph.edges)
        rows,issues=fragments(value,self.graph,self.tid)
        for issue in issues:self.memory.reject(task,issue,self.round)
        # A bad whole-tree graft cannot erase independently measured supplies.
        for step in rows:
            if key(step)not in self.graph.edges:
                self.chem.prepare(self.graph,[step],step['product'],'llm-fragment:'+role)
                self.salvaged_steps+=int(key(step)in self.graph.edges)
            if key(step)in self.graph.edges and self.graph.edges[key(step)]['cc']<=0:
                self.memory.failed_step(step,self.round,role,self.chem.ev.scored)
            self.pending_exact.extend(p for p in(step['product'],*step['reactants'])
                if p in self.graph.nodes and not self.graph.facts[p]['in_stock'])
        route=prepare(value,self.graph,self.tid,self.bases)
        known_route=route_id(route)in self.bases
        self.chem.prepare(self.graph,route['steps'],self.graph.root,'llm:'+role,patch=True)
        metric=self.chem.evaluate(self.graph,{s['product']:s for s in route['steps']},self.tid)
        if not metric['structural_valid']or metric.get('source_issues'):raise ValueError('Independent route/source validation failed')
        self.metrics[identity(metric)]=metric;self.bases[route_id(metric['route'])]=metric['route']
        self.memory.observe(metric,self.round,role,self.chem.ev.scored)
        if known_route and len(self.graph.edges)==before:
            self.memory.reject(task,'No new chemistry: copied a known route; propose a different supplying '
                'transformation or a new target-rooted assembly, rather than repeat the same dependency.',self.round)

    def verify(self,state):
        new_requests=[]
        for role,task,record in self.responses:
            if record['status']!='completed':
                self.errors.append(dict(tool='llm',request=record['request_key'],error=record.get('error')));continue
            value=record['route'];candidates=[value]
            alternatives=value.get('alternatives',[])
            allowance=self.full.supply_alternatives if role=='supply'else 1
            if not isinstance(alternatives,list)or len(alternatives)>allowance:
                self.memory.reject(task,'At most '+str(allowance)+' short alternatives are allowed for this view',self.round)
            else:
                for alternative in alternatives:
                    if isinstance(alternative,dict)and isinstance(alternative.get('steps'),list)and len(alternative['steps'])<=3:
                        candidates.append(alternative)
                    else:self.memory.reject(task,'Alternative needs at most3 explicit steps',self.round)
            for candidate in candidates:
                context=task
                try:
                    from prepare_yield_distributions import canon_side
                    anchor=canon_side(candidate.get('repair_anchor'))if candidate.get('base_route_id')else self.graph.root
                    context=dict(task,repair_anchor=anchor,base_route_id=candidate.get('base_route_id'),
                        proposal_scope='patch'if candidate.get('base_route_id')else'full')
                    self.verify_chemistry(candidate,context,role)
                except (ValueError,KeyError,TypeError)as exc:self.memory.reject(context,exc,self.round)
            # Route validation and control decisions are independent channels.
            queries=value.get('next_searches')or []
            if not isinstance(queries,list):
                self.memory.reject(task,'Typed next_searches list required',self.round);queries=[]
            for query in queries[:4]:
                try:
                    if not isinstance(query,dict):raise ValueError('Typed next_search item required')
                    from prepare_yield_distributions import canon_side
                    p,t=canon_side(query.get('product')),query.get('provider')
                    if p in self.graph.nodes and t in PROVIDERS and(p,t)not in self.graph.attempts:
                        new_requests.append((p,t))
                except (ValueError,KeyError,TypeError)as exc:self.memory.reject(task,exc,self.round)
            try:
                if value.get('literature'):
                    if not isinstance(value['literature'],dict):raise ValueError('Typed literature query required')
                    anchor=task.get('repair_anchor',self.graph.root)
                    if all(self.graph.attempts.get((anchor,t),{}).get('status')=='completed'for t in PROVIDERS):
                        self.evidence.paper(value['literature'])
            except (ValueError,KeyError,TypeError)as exc:self.memory.reject(task,exc,self.round)
        self.pending=list(dict.fromkeys(new_requests+self.pending))
        self.pending_exact=list(dict.fromkeys(self.pending_exact))
        self.responses=[];self.after_evidence='combine';self.trace('verify',state)
        return dict(next='evidence')

    def advance(self,state):
        self.first_phase=dict(quality=self.quality(),costs=self.budget.report())
        atomic_json(self.out/'first_phase.json',self.first_phase)
        self.phase=2;self.cfg=replace(self.full,llm_calls=min(self.full.llm_calls,
            self.budget.used['llm_calls']+self.full.phase_llm))
        self.budget.cfg=self.cfg;self.planner.cfg=self.cfg
        self.trace('advance',state);return dict(phase=2,next='combine')

    def finish(self,state):
        self.cfg=self.full;self.budget.cfg=self.full;self.planner.cfg=self.full
        if not self.stock_target:
            for steps in self.planner.search(self.graph,self.budget,solve=True,final=True):self.evaluate(steps)
        if self.chem.yields:
            for metric in self.metrics.values():metric['yield_proxy']=self.chem.yields.route(metric['route'])
        result=select(self.metrics.values(),self.full.routes);outputs=result['output_routes']
        stock=[m for m in outputs if closed(m)];passing=[m for m in outputs if strict(m)]
        complete=[m for m in stock if m.get('yield_proxy',{}).get('complete')];first=result['selected']
        metrics=dict(stock_closed=bool(stock)or self.stock_target,strict_closed=bool(passing)or self.stock_target,
            first_cc=first['mean_cc']if first else 0.,routes=len(outputs),stock_routes=len(stock),strict_routes=len(passing),
            all_routes_mean_cc=sum(m['mean_cc']for m in outputs)/len(outputs)if outputs else 0.,diversity=diversity(stock),
            complete_yield_routes=len(complete),complete_yield_mean_pct=100*sum(
                m['yield_proxy']['all_steps_product_proxy']for m in complete)/len(complete)if complete else None)
        costs=self.budget.report();costs.update(llm_input_tokens=sum((r.get('usage')or {}).get('input_tokens',0)for r in self.llm_records),
            llm_output_tokens=sum((r.get('usage')or {}).get('output_tokens',0)for r in self.llm_records))
        result.update(target=self.target,metrics=metrics,costs=costs,first_phase=self.first_phase,
            phases=self.phase,errors=self.errors,stopping_reason=self.reason,version='langgraph-0.1',
            llm_failed=sum(r['status']!='completed'for r in self.llm_records),
            solver=dict(nodes=self.planner.nodes,seconds=self.planner.seconds,proofs=self.planner.proofs))
        result['independently_admitted_llm_steps']=self.salvaged_steps
        result['search_accounting']=dict(witness_joins=len(self.planner.strict.seen)+len(self.planner.stock.seen),
            route_states=self.guide.work,solver_nodes=self.planner.nodes,rule_proposals=self.rule_work)
        atomic_json(self.out/'graph.json',dict(nodes=self.graph.nodes,edges=list(self.graph.edges.values()),facts=self.graph.facts,
            attempts=[dict(product=p,tool=t,**r)for(p,t),r in self.graph.attempts.items()]))
        atomic_json(self.out/'candidate_bank.json',list(self.metrics.values()))
        atomic_json(self.out/'working_memory.json',self.memory.snapshot())
        self.evidence.save();self.result=result;atomic_json(self.out/'result.json',result)
        self.trace('finish',state);return dict(done=True,next='end')

    def build(self):
        graph=StateGraph(State)
        nodes=dict(initialize=self.initialize,combine=self.combine,remember=self.remember,
            schedule=self.schedule,tools=self.tools,evidence=self.evidence_node,
            reason=self.reason_node,verify=self.verify,advance=self.advance,finish=self.finish,
            refresh=self.refresh,repair=self.repair)
        for name,node in nodes.items():graph.add_node(name,node)
        graph.add_edge(START,'initialize');graph.add_edge('initialize','combine')
        graph.add_edge('combine','remember');graph.add_edge('remember','schedule')
        graph.add_conditional_edges('schedule',lambda s:s['next'],{n:n for n in('tools','evidence','advance','finish')})
        graph.add_edge('tools','repair');graph.add_conditional_edges('evidence',lambda s:s['next'],{'refresh':'refresh','repair':'repair'})
        graph.add_conditional_edges('refresh',lambda s:s['next'],{'reason':'reason','remember':'remember'})
        graph.add_edge('repair','combine')
        graph.add_edge('reason','verify');graph.add_edge('verify','evidence');graph.add_edge('advance','combine');graph.add_edge('finish',END)
        return graph.compile()

    def run(self):
        self.build().invoke(dict(phase=1,cycle=0,next='initialize'),config=dict(recursion_limit=512))
        return self.result
