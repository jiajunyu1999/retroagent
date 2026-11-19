"""Independent output, physical-call and data-boundary checks."""
import json
import math
from pathlib import Path
from clean_harness.graph import key
from clean_harness.portfolio import rank
from retro_prior_harness import strict_validate
from element_source_guard_v9 import deficits
from retrosynthesis_client import atomic_json


def audit_case(out,blocked):
    out=Path(out);result=json.loads((out/'result.json').read_text());g=json.loads((out/'graph.json').read_text())
    errors=[];score_facts={};stock_facts={};native=exact=material=0
    for p in(out/'tools/function_results').glob('*.json'):
        r=json.loads(p.read_text())
        if r['tool']=='predict_reactions':
            native+=len(r['arguments']['products'])
            if len(r['arguments']['products'])!=1:errors.append('Non-single-product native call')
        if r['tool']=='retrieve_reactions':
            exact+=1
            if r['arguments']['product']in blocked:errors.append('Forbidden target exact query')
        if r['tool']=='find_materials':material+=1
        if r['status']!='completed':continue
        if r['tool']=='check_stock':stock_facts.update(r['result'])
        if r['tool']=='score_reactions':
            for s in r['result']:score_facts[s['reaction_smiles']]=max(0.,s['score_with_fg'])
    edges={key(e['step']):e for e in g['edges']}
    for m in result['output_routes']:
        route=m['route'];valid=strict_validate(result['target']['SMILES'],route)
        if not valid['structural_valid']:errors.append('Structurally invalid output')
        values=[]
        for s in route['steps']:
            rxn='.'.join(s['reactants'])+'>>'+s['product']
            if key(s)not in edges:errors.append('Step absent from current graph')
            if rxn not in score_facts:errors.append('Step lacks actual scoring call');continue
            values.append(score_facts[rxn])
            if deficits(s['product'],s['reactants']):errors.append('Missing explicit heavy-atom source')
        if values and not math.isclose(m['mean_cc'],sum(values)/len(values),abs_tol=1e-9):errors.append('Incorrect CC mean')
        stock=all(stock_facts.get(p,{}).get('in_stock',False)for p in valid['leaves'])
        if bool(m['stock_closed'])!=stock:errors.append('Inventory closure differs from exact stock calls')
        passing=stock and bool(values)and min(values)>0 and valid['structural_valid']and not m.get('source_issues')
        if bool(m['full_pass_after_source_screen'])!=passing:errors.append('Incorrect strict closure')
    if result['selected']and rank(result['selected'])!=max(rank(m)for m in result['output_routes']):
        errors.append('First route is not best closure/CC tier')
    costs=result['costs'];limits=costs['limits']
    for k in('native_calls','llm_calls','evidence_calls','exact_queries','material_queries','support_queries','search_work'):
        if costs[k]>limits[k]:errors.append('Exceeded '+k)
    if native!=costs['native_calls']:errors.append('Physical native counter mismatch')
    if exact!=costs['exact_queries']or material!=costs['material_queries']:errors.append('Physical evidence counter mismatch')
    if sum(result['search_accounting'].values())!=costs['search_work']:errors.append('Search work ledger mismatch')
    if result['phases']>2:errors.append('More than one retry')
    first=result.get('first_phase')
    if first:
        c=first['costs']
        if c['native_calls']>limits['first_native']or c['llm_calls']>limits['phase_llm']or c['search_work']>limits['first_search']:
            errors.append('First-phase limits exceeded')
        if costs['llm_calls']-c['llm_calls']>limits['phase_llm']:errors.append('Second-phase LLM limits exceeded')
    records=[]
    for p in(out/'llm').glob('*.json'):
        r=json.loads(p.read_text())
        if 'request_key'not in r:continue
        records.append(r)
        if len(r.get('attempts',[]))>1:errors.append('LLM transport retried')
        for a in r.get('attempts',[]):
            if a.get('api_key_file_used')or a.get('forbidden_tool_events'):errors.append('LLM isolation violation')
    if len(records)!=costs['llm_calls']:errors.append('LLM counter mismatch')
    denied=[]
    boundary=out/'boundary.jsonl'
    if boundary.exists():
        denied=[json.loads(s)for s in boundary.read_text().splitlines()if not json.loads(s)['allowed']]
    if denied:errors.append('Denied generation read attempted')
    previous=(False,False,0.)
    for line in(out/'graph_events.jsonl').read_text().splitlines():
        q=json.loads(line)['quality'];current=(q['stock_closed'],q['strict_closed'],q['first_cc'])
        if current<previous:errors.append('Incumbent regressed inside one episode')
        previous=current
    report=dict(passed=not errors,errors=sorted(set(errors)),physical_native=native,
                physical_exact=exact,physical_material=material,llm_failed=result['llm_failed'],denied_reads=denied)
    atomic_json(out/'validation.json',report);return report
