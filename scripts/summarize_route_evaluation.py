"""Runtime support for the frozen V9 controller."""

from prepare_yield_distributions import canon_side

def validate_route(target, steps):
    errors = []
    by_product = {}
    for step in steps:
        try:
            product = canon_side(step['product'])
            reactants = [canon_side(s) for s in step['reactants']]
            if not reactants or not product or any(not s for s in reactants):
                raise ValueError('Empty product or reactants')
            if product in by_product:
                errors.append('duplicate_product:' + product)
            by_product[product] = reactants
        except (ValueError, KeyError, TypeError):
            errors.append('invalid_step:' + str(step.get('step_id')))
    target = canon_side(target)
    if steps:
        try:
            if canon_side(steps[0]['product']) != target:
                errors.append('first_step_does_not_produce_target')
        except (ValueError, KeyError):
            errors.append('invalid_first_step')
    leaves = set()
    visited = set()
    max_depth = 0
    def walk(product, ancestors, depth):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        if product in ancestors:
            errors.append('cycle:' + product)
            return
        if product not in by_product:
            leaves.add(product)
            return
        visited.add(product)
        for reactant in by_product[product]:
            walk(reactant, ancestors | {product}, depth + 1)
    walk(target, set(), 0)
    if set(by_product) - visited:
        errors.append('disconnected_steps:' + str(len(set(by_product)-visited)))
    if max_depth > 6:
        errors.append('branch_depth_exceeds_6')
    if not steps:
        errors.append('no_transformations')
    return {'structural_valid': not errors, 'structural_errors': sorted(set(errors)),
            'max_depth': max_depth, 'leaves': sorted(leaves), 'n_steps': len(steps)}
