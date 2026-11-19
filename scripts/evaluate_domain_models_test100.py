"""Runtime support for the frozen V9 controller."""

from prepare_yield_distributions import canon_side

from retro_retrieval_harness import prune_stock

from retro_prior_harness import strict_validate

from luna_skeleton_guard import explicit_source_deficits

def normalize(route):
    """Last actual action per product; follow every reachable reactant subtree."""
    target = canon_side(route['target_smiles'])
    bank = {canon_side(s['product']):dict(product=canon_side(s['product']),
        reactants=[canon_side(r) for r in s['reactants']],
        reaction_type=s.get('reaction_type','Native prediction')) for s in route['steps']}
    steps,visited = [],set()
    def walk(product):
        if product in visited or product not in bank:
            return
        visited.add(product)
        s = bank[product]
        steps.append({**s,'step_id':'R%02d'%(len(steps)+1)})
        for r in s['reactants']:
            walk(r)
    walk(target)
    return dict(target_smiles=target,target_id=route.get('target_id'),steps=steps)

def route_eval(evaluator,row,route,source):
    normalized = normalize(route)
    if strict_validate(row['SMILES'],normalized)['structural_valid']:
        normalized = prune_stock(normalized,evaluator)
    e = evaluator.evaluate(row['SMILES'],normalized,source)
    issues = [dict(step_id=s['step_id'],deficits=explicit_source_deficits(s))
              for s in normalized['steps'] if explicit_source_deficits(s)]
    e['source_issues'] = issues
    e['full_pass_after_source_screen'] = e['full_pass'] and not issues
    return e
