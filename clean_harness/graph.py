"""One episode's observed AND/OR graph. A cut is a reaction, not a whole route."""
from collections import defaultdict, deque
from prepare_yield_distributions import canon_side
from element_source_guard_v9 import deficits
from .config import PROVIDERS


def key(step):
    return step['product'], tuple(sorted(step['reactants']))


class Graph:
    def __init__(self, root):
        self.root = canon_side(root)
        self.nodes = {self.root: 'N0'}
        self.edges, self.cuts, self.facts, self.attempts = {}, {}, {}, {}
        self.focus = []

    def node(self, p):
        if p not in self.nodes:
            self.nodes[p] = 'N'+str(len(self.nodes))
        return self.nodes[p]

    def product(self, node):
        return next((p for p, n in self.nodes.items() if n == node), None)

    def normalize(self, row, product=None):
        p = canon_side(row.get('product', product))
        if product is not None and p != product:
            raise ValueError('Product mismatch')
        rs = row['reactants']
        rs = rs.split('.') if isinstance(rs, str) else rs
        if not isinstance(rs, list) or not all(isinstance(r, str) for r in rs):
            raise ValueError('Reactants must be SMILES strings')
        rs = sorted(r for s in rs for r in canon_side(s).split('.'))
        if not 1 <= len(rs) <= 6 or p in rs or self.root in rs or deficits(p, rs):
            raise ValueError('Invalid reaction, cycle, or missing explicit heavy-element sources')
        return dict(product=p, reactants=rs, reaction_type=row.get('reaction_type', 'Proposal'))

    def add(self, step, score, origin, sources=()):
        k = key(step)
        self.node(step['product'])
        for p in step['reactants']:
            self.node(p)
        if k in self.edges:
            self.edges[k]['origins'] = sorted(set(self.edges[k]['origins']) | {origin})
            return False
        cut = 'C'+str(len(self.edges))
        self.edges[k] = dict(step=step, cc=max(0., float(score)), raw_cc=float(score),
                             origin=origin, origins=[origin], cut=cut, sources=list(sources))
        self.cuts[cut] = self.edges[k]
        return True

    def all_tools_failed(self, p):
        # Errors are NOT chemical evidence. Only completed, validated tool attempts qualify.
        return all(self.attempts.get((p,t), {}).get('status') == 'completed'
                   for t in PROVIDERS) and not any(e['cc'] > 0 for e in self.edges.values()
                                                 if e['step']['product'] == p)

    def reachable_depths(self):
        by = defaultdict(list)
        for e in self.edges.values():
            by[e['step']['product']].extend(e['step']['reactants'])
        depth, queue = {self.root: 0}, deque([self.root])
        while queue:
            p = queue.popleft()
            if depth[p] >= 6:
                continue
            for r in sorted(set(by[p])):
                if r not in depth:
                    depth[r] = depth[p]+1
                    queue.append(r)
        return depth

    def backtrack(self, node, cut, solved):
        p = self.product(node)
        if p is None or p not in self.reachable_depths():
            raise ValueError('Unknown/unreachable backtrack node')
        if cut is not None:
            e = self.cuts.get(cut)
            if not e or e['step']['product'] != p:
                raise ValueError('Backtrack cut must be observed at that node')
            self.focus = [r for r in e['step']['reactants']
                          if not self.facts[r]['in_stock'] and r not in solved]
        else:
            self.focus = [p]
        # Do not delete solved siblings, observed alternatives or any incumbent.
        return list(self.focus)

    def frontier(self, solved, metrics=()):
        depths = self.reachable_depths()
        weak = {}
        for m in sorted(metrics, key=lambda m: -m['mean_cc'])[:3]:
            for s, d in zip(m['route']['steps'], m['step_diagnostics']):
                weak[s['product']] = min(weak.get(s['product'], 5.), d['score_with_fg'])
        rows = []
        for p, depth in depths.items():
            if depth >= 6 or (p != self.root and self.facts.get(p, {}).get('in_stock')):
                continue
            unused = [t for t in PROVIDERS if (p,t) not in self.attempts]
            if not unused:
                continue
            if p in solved and p not in weak:
                continue
            used = sum((p,t) in self.attempts for t in PROVIDERS)
            priority = (p in self.focus, p not in solved, -used, -depth,
                        -weak.get(p, 0.), self.node(p))
            rows.append(dict(product=p, node=self.node(p), tools=unused, priority=priority,
                             on_incumbent=p in weak,weak_cc=weak.get(p)))
        return sorted(rows, key=lambda r: r['priority'], reverse=True)

    def state_key(self, p, solved):
        from rdkit import Chem
        from rdkit.Chem import Fragments
        m = Chem.MolFromSmiles(p)
        groups = [n[3:] for n in ('fr_amide','fr_ester','fr_ether','fr_halogen','fr_NH0','fr_NH1','fr_Ar_N')
                  if getattr(Fragments,n)(m)]
        return '|'.join([('solved' if p in solved else 'open'),
                         'failed' if self.all_tools_failed(p) else 'searching',
                         str(min(5,m.GetNumHeavyAtoms()//10)), str(sum(a.GetIsAromatic() for a in m.GetAtoms())//6),
                         ','.join(groups)])
