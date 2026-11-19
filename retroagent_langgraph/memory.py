"""Working memory built ONLY from this target's measured rollout.

It remembers compatible branches, reaction-center failures and rejection
budgets. No case IDs, historical routes or prewritten chemical recipes.
"""
from collections import Counter, defaultdict
from rdkit import Chem
from clean_harness.portfolio import closed, strict, rank
from .routes import route_id, parent, patch_window


def heavy(product):
    return Chem.MolFromSmiles(product).GetNumHeavyAtoms()


class EpisodeMemory:
    def __init__(self):
        self.failures={};self.rejections=[];self.assignments=Counter()
        self.protected={};self.centers={};self.tool_stats=defaultdict(Counter)
        self.positive_centers=defaultdict(set)
        self.decisions=[]

    def family(self,step,scorer):
        rxn='.'.join(step['reactants'])+'>>'+step['product']
        if rxn not in self.centers:
            raw=scorer(rxn)
            centers=tuple(sorted(c['reaction_center_smiles']for c in raw.get('centers',[])
                                 if c.get('center_type')==1))
            self.centers[rxn]=centers or ('unmapped:'+step['product'],)
            if raw.get('score_with_fg',0)>0:self.positive_centers[self.centers[rxn]].add(rxn)
        return self.centers[rxn]

    def observe(self,metric,round_no,role,scorer):
        if not metric.get('structural_valid')or metric.get('source_issues'):return
        positives=all(d['score_with_fg']>0 for d in metric['step_diagnostics'])
        if positives:
            self.protected[route_id(metric['route'])]=metric
        for step,diagnostic in zip(metric['route']['steps'],metric['step_diagnostics']):
            if diagnostic['score_with_fg']>0:continue
            self.failed_step(step,round_no,role,scorer)

    def failed_step(self,step,round_no,role,scorer):
        rxn='.'.join(step['reactants'])+'>>'+step['product'];raw=scorer(rxn)
        if raw.get('score_with_fg',0)>0:return
        family=self.family(step,scorer)
        failure_type='functional_group_context'if raw.get('score_without_fg',0)>0 else'reaction_center_missing'
        token=(step['product'],family)
        item=self.failures.setdefault(token,dict(anchor=step['product'],center=list(family),
            rounds=set(),reactions=set(),roles=set(),failure_type=failure_type))
        item['rounds'].add(round_no);item['reactions'].add(rxn);item['roles'].add(role)

    def reject(self,task,error,round_no):
        self.rejections.append(dict(anchor=task.get('repair_anchor'),base_route_id=task.get('base_route_id'),
            scope=task.get('proposal_scope','assignment'),round=round_no,error=str(error)[:400],chemical_failure=False))

    def observe_tool(self,product,provider,record,graph,planner):
        state=graph.state_key(product,planner.solved)
        stats=self.tool_stats[state,provider]
        if record['status']!='completed':
            stats['transport_failures']+=1;return
        stats['calls']+=1
        for k in ('admitted','supported','added'):stats[k]+=record.get(k,0)

    def _task(self,metric,anchor,purpose,scorer,*,force_parent=False):
        route=metric['route'];original=anchor
        depth_errors=sum(r['anchor']==anchor and ('depth'in r['error']or 'branch'in r['error'])
                         for r in self.rejections)
        window=patch_window(route,anchor)
        if force_parent or depth_errors or window['remaining_branch_transformations']==0:
            up=parent(route,anchor)
            if up:anchor=up;window=patch_window(route,anchor)
        while depth_errors and window['remaining_branch_transformations']<3:
            up=parent(route,anchor)
            if not up:break
            anchor=up;window=patch_window(route,anchor)
        return dict(base_route_id=route_id(route),repair_anchor=anchor,original_anchor=original,
                    purpose=purpose,**window,source='current_episode',
                    preserve_supported_siblings=True,strict_contract=False)

    def contracts(self,metrics,graph,planner,scorer):
        candidates=[]
        bank={route_id(m['route']):m for m in metrics if m.get('structural_valid')and not m.get('source_issues')}
        bank.update(self.protected)
        for rid,m in bank.items():
            if closed(m)or not m['route']['steps']:continue
            if any(d['score_with_fg']<=0 for d in m['step_diagnostics']):continue
            root_step=next((s for s in m['route']['steps']if s['product']==graph.root),None)
            if root_step is None:continue
            gaps=[p for p in m.get('open_leaves',[])if not graph.facts[p]['in_stock']]
            if not gaps:continue
            anchor=max(gaps,key=lambda p:(p not in planner.solved,heavy(p),p))
            family=self.family(root_step,scorer)
            visits=self.assignments[family,anchor]
            attempts=sum((anchor,t)in graph.attempts for t in ('LocalRetro','RetroChimeraEdit','AiZynthFinder','Graph2Edits','RetroChimeraDeNovo'))
            burden=sum((heavy(p)+3)**1.25 for p in gaps)
            task=self._task(m,anchor,'Complete this actually positive-CC branch, supplying ALL missing inputs. '
                'If its old supply repeatedly failed, replace the parent construction instead.',scorer,
                force_parent=visits>=2 and attempts>=3 and anchor not in planner.solved)
            candidates.append(dict(metric=m,family=family,gaps=gaps,task=task,
                priority=(visits//2,task['remaining_branch_transformations']==0,len(gaps),burden,-m['mean_cc'],rid)))
        candidates.sort(key=lambda x:x['priority'])
        selected=[];seen=set()
        for row in candidates:
            if row['family']in seen:continue
            seen.add(row['family']);selected.append(row)
            if len(selected)==4:break
        return selected

    def tasks(self,metrics,graph,planner,scorer,round_no):
        values=list(metrics)
        best=max((m for m in values if strict(m)),key=rank,default=None)
        chosen=[]
        if best:
            pairs=sorted(zip(best['route']['steps'],best['step_diagnostics']),key=lambda x:x[1]['score_with_fg'])
            chosen=[self._task(best,s['product'],'Replace this low-CC step or subtree while preserving the '
                'closed incumbent. Judge the mean CC of the complete graft, not this step alone.',scorer)
                for s,_ in pairs[:2]]
            global_task=dict(purpose='Develop an independent higher-CC target-rooted assembly. '
                'Use actual observed positive root options and supply certificates; reorder bond formation '
                'when a local patch would retain the same low-CC skeleton. Judge the whole connected route, '
                'including every required supply, and preserve the existing strict incumbent.',
                remaining_branch_transformations=6,remaining_total_steps=16,strict_contract=False,
                source='current_episode')
            while len(chosen)<2:chosen.append(dict(global_task))
            chosen.append(global_task)
            self.decisions.append(dict(round=round_no,tasks=chosen));return chosen
        else:
            for contract in self.contracts(values,graph,planner,scorer)[:2]:
                task=contract['task'];anchor=task['original_anchor']
                self.assignments[contract['family'],anchor]+=1
                chosen.append(task)
        broken=sorted((m for m in values if closed(m)and not strict(m)),
                      key=lambda m:(m['failed_steps'],-m['mean_cc'],m['n_steps']))
        if broken:
            m=broken[0]
            bad=[(s,d)for s,d in zip(m['route']['steps'],m['step_diagnostics'])if d['score_with_fg']<=0]
            def repeats(pair):
                s,d=pair;item=self.failures.get((s['product'],self.family(s,scorer)),{})
                rounds=set(item.get('rounds',()))
                for (_,family),other in self.failures.items():
                    if family==self.family(s,scorer):rounds.update(other['rounds'])
                return len(rounds)
            step,_=max(bad,key=repeats)
            raw=scorer('.'.join(step['reactants'])+'>>'+step['product'])
            purpose=('The reaction center has precedent but this functional-group context fails. '
                'Try a different protection context or assembly order, with every real transformation explicit and verified.'
                if raw.get('score_without_fg',0)>0 else
                'Bypass the unsupported construction. A high-CC precursor '
                'cannot repair a CC-zero parent reaction. Change assembly topology or timing; do not repeat '
                'the same failed center with a renamed protecting group.')
            task=self._task(m,step['product'],purpose,scorer,force_parent=repeats((step,None))>=2)
        else:
            task=dict(purpose='Develop an independent target-rooted assembly, using a different observed '
                'root reaction center from the other views. Preserve existing valid supplies.',
                remaining_branch_transformations=6,remaining_total_steps=16,strict_contract=False)
        while len(chosen)<2:chosen.append(dict(task))
        chosen.append(task)
        self.decisions.append(dict(round=round_no,tasks=chosen))
        return chosen

    def snapshot(self):
        failures=[dict(anchor=v['anchor'],center=v['center'],rounds=sorted(v['rounds']),
                       distinct_failed_reactions=len(v['reactions']),roles=sorted(v['roles']),failure_type=v['failure_type'],
                       failed_reaction_examples=sorted(v['reactions'])[:2])
                  for v in self.failures.values()]
        failures.sort(key=lambda r:(-len(r['rounds']),-r['distinct_failed_reactions'],r['anchor']))
        grouped={}
        for row in failures:
            family=tuple(row['center']);g=grouped.setdefault(family,dict(center=row['center'],rounds=set(),anchors=set()))
            g['rounds'].update(row['rounds']);g['anchors'].add(row['anchor'])
        families=[dict(center=g['center'],rounds=sorted(g['rounds']),anchors=sorted(g['anchors']),
            positive_in_other_contexts=len(self.positive_centers[family]),
            positive_reaction_examples=sorted(self.positive_centers[family])[:2])for family,g in grouped.items()]
        families.sort(key=lambda r:-len(r['rounds']))
        return dict(schema='episode-memory-1',failed_centers=failures,
            failed_center_families=families,
            rejected_proposals=self.rejections,protected_routes=sorted(self.protected),
            assignments=[dict(center=list(k[0]),anchor=k[1],visits=v)for k,v in self.assignments.items()],
            tool_outcomes=[dict(state=k[0],provider=k[1],**v)for k,v in self.tool_stats.items()],
            decisions=self.decisions)

    def context(self,metrics,graph,planner,scorer):
        snapshot=self.snapshot()
        contracts=self.contracts(metrics,graph,planner,scorer)
        return dict(failed_centers=snapshot['failed_centers'][:8],
            failed_center_families=snapshot['failed_center_families'][:6],
            validation_failures=self.rejections[-6:],
            supported_contracts=[dict(route_id=route_id(r['metric']['route']),
                center=list(r['family']),open_inputs=r['gaps'],repair=r['task'])for r in contracts],
            tool_outcomes=sorted(snapshot['tool_outcomes'],key=lambda r:-r.get('calls',0))[:12],
            rule='These are observed failures in their actual contexts, not a universal ban on the reaction family.')
