"""Runtime support for the frozen V9 controller."""

from prepare_yield_distributions import canon_side

def assemble(target, bank, choices=None):
    """Prune unused subtrees by walking actual connectivity, never claimed leaves."""
    choices = choices or {}; steps = []; visited = set()
    def walk(product, depth):
        product = canon_side(product)
        if depth > 6 or product in visited: return
        step = choices.get(product, bank.get(product))
        if step is None: return
        visited.add(product)
        steps.append({'step_id': 'R%02d' % (len(steps) + 1), 'product': product,
                      'reactants': step['reactants'], 'reaction_type': step.get('reaction_type', 'Database-supported transformation')})
        for reactant in step['reactants']: walk(reactant, depth + 1)
    walk(target, 1)
    return {'target_smiles': target, 'steps': steps}

def prune_stock(route, evaluator):
    # Stop legitimately at independently verified stock intermediates.
    bank = {canon_side(s['product']): s for s in route['steps']}
    products = list(bank); stock = evaluator.stock.query(products)
    target = canon_side(route['target_smiles'])
    bank = {p: s for p, s in bank.items() if p == target or not stock[p]['in_stock']}
    result = assemble(target, bank); result['target_id'] = route.get('target_id')
    return result
