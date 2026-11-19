"""Bounded current-graph mean-CC MILP, extracted from validated V84, no agent imports."""
from collections import defaultdict
import time
import warnings
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def mean_cc_plan(root,edges,facts,*,strict_only=True,seconds=1.,node_limit=256,initial_ratio=0.):
    """V57 AND MILP/Dinkelbach, with total wall and branch-node limits added."""
    started=time.monotonic();by=defaultdict(list)
    for e in edges.values():
        s=e['step']
        if(e['cc']<=0 if strict_only else e['cc']<0)or s['product']!=root and facts[s['product']]['in_stock']:continue
        by[s['product']].append(e)
    reachable={root};pending=[root]
    while pending:
        p=pending.pop()
        for e in by[p]:
            for r in e['step']['reactants']:
                if r not in reachable:reachable.add(r);pending.append(r)
    es=[e for p in sorted(reachable)for e in by[p]];ne=len(es)
    if not ne or seconds<=0 or node_limit<=0:return None,dict(branch_nodes=0,iterations=[],certified=False,seconds=0.)
    molecules=sorted(reachable);mid={p:i for i,p in enumerate(molecules)};nm=len(mid);n=ne+2*nm
    rows=[];cols=[];vals=[];lo=[];hi=[]
    def add(coeff,l=-np.inf,h=np.inf):
        row=len(lo)
        for j,v in coeff.items():rows.append(row);cols.append(j);vals.append(v)
        lo.append(l);hi.append(h)
    def z(p):return ne+mid[p]
    def depth(p):return ne+nm+mid[p]
    outgoing=defaultdict(list);incoming=defaultdict(list)
    for i,e in enumerate(es):
        p=e['step']['product'];outgoing[p].append(i)
        for r in sorted(set(e['step']['reactants'])):
            incoming[r].append(i);add({z(r):1,i:-1},l=0)
            add({depth(r):1,depth(p):-1,i:-7},l=-6)
    for p in molecules:
        if p==root or not facts[p]['in_stock']:add({**{i:1 for i in outgoing[p]},z(p):-1},l=0,h=0)
        if p!=root:add({z(p):1,**{i:-1 for i in incoming[p]}},h=0)
    add({i:1 for i in range(ne)},l=1,h=16);add({z(root):1},l=1,h=1);add({depth(root):1},l=0,h=0)
    matrix=coo_matrix((vals,(rows,cols)),shape=(len(lo),n)).tocsc()
    constraint=LinearConstraint(matrix,np.asarray(lo),np.asarray(hi))
    bounds=Bounds(np.zeros(n),np.r_[np.ones(ne+nm),np.full(nm,6.)])
    integrality=np.r_[np.ones(ne+nm),np.zeros(nm)]
    scores=np.array([e['cc']for e in es]);ratio=initial_ratio;best=None;nodes=0;history=[];certified=False
    for _ in range(6):
        remaining=seconds-(time.monotonic()-started)
        if remaining<.02 or nodes>=node_limit:break
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result=milp(np.r_[-(scores-ratio),np.zeros(2*nm)],integrality=integrality,bounds=bounds,constraints=constraint,
                options=dict(time_limit=remaining,node_limit=max(1,node_limit-nodes),mip_rel_gap=1e-8,threads=1))
        nodes+=int(getattr(result,'mip_node_count',0)or 0)
        if result.x is None:history.append(dict(status=int(result.status),message=result.message));break
        active=np.rint(result.x[:ne]);count=int(active.sum())
        if not count:break
        mean=float(scores@active/count);residual=float((scores-ratio)@active)
        if best is None or mean>best['mean_cc']+1e-10:best=dict(mean_cc=mean,edges=[es[i]for i in range(ne)if active[i]>.5])
        history.append(dict(status=int(result.status),mean_cc=mean,residual=residual,steps=count))
        if result.status==0 and abs(residual)<1e-7:certified=True;break
        if abs(mean-ratio)<1e-9:break
        ratio=mean
    return best,dict(branch_nodes=nodes,iterations=history,certified=certified,seconds=time.monotonic()-started,eligible_edges=ne)
