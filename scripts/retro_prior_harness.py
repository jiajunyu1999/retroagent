"""Runtime support for the frozen V9 controller."""

from prepare_yield_distributions import canon_side

from summarize_route_evaluation import validate_route

def strict_validate(target,route):
    if not isinstance(route,dict) or not isinstance(route.get('steps'),list):
        return {'structural_valid':False,'structural_errors':['invalid_route_schema'],'leaves':[], 'n_steps':0,'max_depth':0}
    steps=route['steps'];errors=[]
    if len(steps)>16:errors.append('total_steps_exceeds_16')
    labels=[]
    for s in steps:
        if not isinstance(s,dict) or not isinstance(s.get('product'),str) or not isinstance(s.get('reactants'),list):
            errors.append('invalid_step_schema');continue
        labels.append(s.get('step_id'))
        if not 1<=len(s['reactants'])<=6 or not all(isinstance(r,str) and 0<len(r)<=2000 for r in s['reactants']):
            errors.append('invalid_reactant_list')
    if len(set(str(x) for x in labels))!=len(labels) or any(not isinstance(x,str) or not x for x in labels):
        errors.append('invalid_or_duplicate_step_ids')
    try:
        if route.get('target_smiles') and canon_side(route['target_smiles'])!=canon_side(target):errors.append('target_smiles_mismatch')
    except (ValueError,TypeError):errors.append('invalid_target_smiles')
    if errors:return {'structural_valid':False,'structural_errors':sorted(set(errors)),'leaves':[], 'n_steps':len(steps),'max_depth':0}
    return validate_route(target,steps)

class Evaluator:
    def evaluate(self,target,route,source):
        v=strict_validate(target,route)
        e={**v,'source':source,'route':route,'full_pass':False,'stock_closed':False,
           'all_steps_supported':False,'min_cc':0,'mean_cc':0,'step_support_fraction':0,
           'leaf_stock_fraction':0,'failed_steps':0,'step_diagnostics':[],'open_leaves':[]}
        if not v['structural_valid']:return e
        stock=self.stock.query(v['leaves']);e['open_leaves']=[s for s in v['leaves'] if not stock[s]['in_stock']]
        e['stock_closed']=bool(v['leaves']) and not e['open_leaves']
        e['leaf_stock_fraction']=sum(stock[s]['in_stock'] for s in v['leaves'])/len(v['leaves']) if v['leaves'] else 0
        for s in route['steps']:
            rxn='.'.join(s['reactants'])+'>>'+s['product'];d=self.scored(rxn)
            badcenters=[c for c in d.get('centers',[]) if not c['center_has_precedent'] or c['missing_functional_groups']]
            e['step_diagnostics'].append({'step_id':s['step_id'],'reaction_smiles':rxn,
                'score_with_fg':d['score_with_fg'],'score_without_fg':d['score_without_fg'],
                'failure_stage':d.get('failure_stage'),'failure_message':d.get('failure_message'),
                'missing_functional_groups':sorted({f for c in badcenters for f in c['missing_functional_groups']}),
                'unsupported_center_types':[c['center_type'] for c in badcenters if not c['center_has_precedent']]})
        values=[d['score_with_fg'] for d in e['step_diagnostics']]
        e['min_cc']=min(max(v,0) for v in values);e['mean_cc']=sum(max(v,0) for v in values)/len(values)
        e['failed_steps']=sum(v<0 for v in values);e['step_support_fraction']=sum(v>0 for v in values)/len(values)
        e['all_steps_supported']=all(v>0 for v in values)
        e['full_pass']=e['structural_valid'] and e['stock_closed'] and e['all_steps_supported']
        return e
