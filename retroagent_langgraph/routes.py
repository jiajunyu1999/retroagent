"""Connected proposals and depth-aware repair contracts; no stored solutions."""
from clean_harness.graph import key
from clean_harness.planner import make_route
from retro_prior_harness import strict_validate
from retrosynthesis_client import digest


def route_id(route):
    return 'R'+digest(sorted(key(s)for s in route['steps']))[:16]


def depth_map(route):
    bank={s['product']:s for s in route['steps']}
    root=route['target_smiles'];depths={}
    def visit(p,depth,path):
        if p in path:raise ValueError('Route cycle')
        depths[p]=max(depth,depths.get(p,0))
        if p in bank:
            for r in bank[p]['reactants']:visit(r,depth+1,path|{p})
    visit(root,1,set())
    return depths


def parent(route,anchor):
    depths=depth_map(route)
    candidates=[s['product']for s in route['steps']if anchor in s['reactants']]
    return max(candidates,key=lambda p:depths[p]) if candidates else None


def patch_window(route,anchor):
    depths=depth_map(route)
    if anchor not in depths:raise ValueError('Unobserved repair anchor')
    bank={s['product']:s for s in route['steps']}
    fixed=set()
    def walk(p):
        if p==anchor or p in fixed or p not in bank:return
        fixed.add(p)
        for r in bank[p]['reactants']:walk(r)
    walk(route['target_smiles'])
    return dict(anchor_depth=depths[anchor],remaining_branch_transformations=max(0,7-depths[anchor]),
                remaining_total_steps=max(0,16-len(fixed)))


def fragments(value,graph,tid):
    """Keep source-valid local hypotheses, without accepting an invalid tree.

    Only products already observed or connected through the proposed target/
    observed-anchor dependency tree are eligible. Raw connectivity may explain
    relevance, but never establishes source validity, score or closure.
    """
    if not isinstance(value,dict)or value.get('target_id',tid)!=tid:raise ValueError('Wrong proposal target')
    rows=value.get('steps')
    if not isinstance(rows,list)or len(rows)>16:raise ValueError('At most16 typed candidate steps required')
    from prepare_yield_distributions import canon_side
    parsed=[];issues=[]
    for row in rows:
        try:
            if not isinstance(row,dict)or not isinstance(row.get('product'),str):raise ValueError('Typed product required')
            rs=row.get('reactants');rs=rs.split('.')if isinstance(rs,str)else rs
            if not isinstance(rs,list)or not all(isinstance(r,str)for r in rs):raise ValueError('Typed reactants required')
            parsed.append(dict(row,product=canon_side(row['product']),
                reactants=[p for r in rs for p in canon_side(r).split('.')]))
        except (ValueError,KeyError,TypeError)as exc:issues.append(str(exc))
    relevant=set(graph.nodes);changed=True
    while changed:
        changed=False
        for row in parsed:
            if row['product']in relevant:
                for p in row['reactants']:
                    if p not in relevant:relevant.add(p);changed=True
    valid={}
    for row in parsed:
        if row['product']not in relevant:continue
        try:
            step=graph.normalize(row);valid[key(step)]=step
        except (ValueError,KeyError,TypeError)as exc:
            issues.append('At product '+row['product']+', reactants '+'.'.join(row['reactants'])+': '+str(exc))
    return list(valid.values()),issues


def prepare(value,graph,tid,bases):
    if not isinstance(value,dict)or value.get('target_id',tid)!=tid:
        raise ValueError('Wrong proposal target')
    rows=value.get('steps')
    if not isinstance(rows,list)or not 1<=len(rows)<=16:raise ValueError('Need1..16 explicit steps')
    if any(not isinstance(s,dict)or not isinstance(s.get('product'),str)or
           not isinstance(s.get('reactants'),(list,str))or isinstance(s['reactants'],list)and
           not all(isinstance(r,str)for r in s['reactants'])for s in rows):
        raise ValueError('Typed product/reactant steps required')
    steps=[]
    for row in rows:
        try:steps.append(graph.normalize(row))
        except ValueError:
            from prepare_yield_distributions import canon_side
            from element_source_guard_v9 import deficits
            rs=row['reactants'].split('.')if isinstance(row['reactants'],str)else row['reactants']
            rs=[p for r in rs for p in canon_side(r).split('.')]
            missing=deficits(canon_side(row['product']),rs)
            if missing:raise ValueError('Missing explicit heavy-atom sources at '+row['product']+
                ' from '+'.'.join(rs)+': '+str(missing))
            raise
    if len({s['product']for s in steps})!=len(steps):raise ValueError('Conflicting production steps')
    if value.get('base_route_id'):
        base=bases.get(value['base_route_id'])
        if base is None:raise ValueError('Unknown current-episode route')
        from prepare_yield_distributions import canon_side
        anchor=canon_side(value.get('repair_anchor',''))
        if anchor not in depth_map(base)or steps[0]['product']!=anchor:
            raise ValueError('Patch must produce its observed repair anchor')
        bank={s['product']:s for s in base['steps']}
        bank.update({s['product']:s for s in steps})
    else:
        if steps[0]['product']!=graph.root:raise ValueError('Full route must produce exact target first')
        bank={s['product']:s for s in steps}
    connected={}
    def visit(p,depth,path):
        if p in path:raise ValueError('Patch creates a cycle')
        if p not in bank or p!=graph.root and graph.facts.get(p,{}).get('in_stock'):return
        if depth>6:raise ValueError('Patch exceeds remaining branch depth at '+p+'; actual depth='+str(depth))
        connected[p]=bank[p]
        if len(connected)>16:raise ValueError('Patch exceeds remaining total steps')
        for r in bank[p]['reactants']:visit(r,depth+1,path|{p})
    visit(graph.root,1,set())
    route=make_route(graph.root,connected,tid)
    route['steps'].sort(key=lambda s:(s['product']!=graph.root,s['product']))
    for index,step in enumerate(route['steps'],1):step['step_id']='R%02d'%index
    valid=strict_validate(graph.root,route)
    if not valid['structural_valid']:raise ValueError('Invalid connected route: '+str(valid['structural_errors']))
    return route
