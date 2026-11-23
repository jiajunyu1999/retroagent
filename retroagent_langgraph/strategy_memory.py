"""Outcome-driven strategy contracts for batched inference."""
from .routes import patch_window, route_id

GUIDANCE = {
    'change_root': (
        'Try a new target-rooted construction with a different first-step reactant set from the supplied leading routes. '
        'Bind every unsupplied input to either a complete positive supply certificate or a distinct explicit testable '
        'subtree. Use a different bond construction or assembly order. Reusing the blocked dependency under a new '
        'protecting group does not test this strategy.'),
    'replace_weak': (
        'The CC-zero product is the provisional repair anchor. If the zero edge introduces an unworkable dependency, '
        'compare its parent as an alternate anchor. Keep unaffected siblings. Replace the zero step itself and check '
        'all inputs of the replacement. Improving an already-supported descendant alone cannot close this route.'),
    'supply': (
        'All verified steps in this partial branch can stand. Choose an actual missing input with a feasible remaining '
        'depth. Prefer short supply hypotheses with distinct precursors and a complete list of co-inputs.'),
}

def contract_tasks(runtime,tasks,ordered,anchor,weak):
    result=[]
    for strategy,old in zip(ordered,tasks):
        if strategy=='change_root':
            task=dict(purpose=GUIDANCE[strategy],remaining_branch_transformations=6,
                      remaining_total_steps=16,strict_contract=False)
        elif strategy=='replace_weak' and weak:
            p=weak[0]
            route=anchor['route']
            window=patch_window(route,p)
            task=dict(base_route_id=route_id(route),repair_anchor=p,original_anchor=p,
                      purpose=GUIDANCE[strategy],preserve_supported_siblings=True,
                      strict_contract=False,**window)
        elif strategy=='supply':
            contracts=runtime.memory.contracts(runtime.metrics.values(),runtime.graph,
                                                runtime.planner,runtime.chem.ev.scored)
            task=dict(contracts[0]['task']) if contracts else dict(old)
            task['purpose']=GUIDANCE[strategy]
        else:
            task=old
        result.append(task)
    return result
