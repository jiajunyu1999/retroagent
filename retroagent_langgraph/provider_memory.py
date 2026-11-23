"""Persistent, outcome-based guidance for selecting a single-step provider."""

from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from statistics import mean

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator

from .skill_memory import target_key

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
METRICS = ('positive', 'near', 'closed', 'new', 'seconds', 'route_strict', 'route_stock')
molecule_key = lru_cache(maxsize=32768)(target_key)


def read(path):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=8192)
def features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms()) / max(1, heavy)
    stereo = bool(Chem.FindMolChiralCenters(mol, includeUnassigned=False))
    return ('size:' + ('small' if heavy <= 16 else 'mid' if heavy <= 32 else 'large'),
            f'rings:{min(rings, 3)}', 'aromatic:' + ('high' if aromatic >= .5 else 'some' if aromatic else 'none'),
            f'stereo:{int(stereo)}')


@lru_cache(maxsize=32768)
def bits(smiles):
    return frozenset(GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles)).GetOnBits())


def change(product, reactants):
    return bits(product) ^ frozenset().union(*(bits(s) for s in reactants))


def compare(product, edges, observed, elapsed):
    """Credit distinct fingerprint changes that also have positive support."""
    positive = [e for e in edges if e['cc'] > 0]
    unique = [e for e in positive if change(product, e['reactants']) not in observed]
    return dict(positive=bool(positive), near=any(e['gaps'] <= 1 for e in unique),
                closed=any(e['gaps'] == 0 for e in unique), new=bool(unique), seconds=max(0., elapsed))


def failure(record):
    if not record.get('admitted'):
        return 'no_admitted'
    if not record.get('supported'):
        return 'zero_cc'
    return 'supported'


class ProviderMemory:
    def __init__(self, path, *, writable=False):
        path = Path(path).resolve()
        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path if writable else path.as_uri()+'?mode=ro',
                                  uri=not writable, timeout=30)
        if writable:
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS observations ('
                    'source TEXT, target TEXT, node TEXT, provider TEXT, stage TEXT, prior TEXT, '
                    'failure TEXT, context TEXT, values_json TEXT, '
                    'PRIMARY KEY(source,target,node,provider,stage,prior,failure))')

    def close(self):
        self.db.close()

    def record(self, source, target, product, provider, stage, prior, condition, values):
        if stage not in ('root', 'frontier', 'numeric_frontier', 'route_feedback'):
            raise ValueError('Unknown evidence stage')
        self.db.execute('INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?)',
            (source, molecule_key(target), molecule_key(product), provider, stage, prior, condition,
             json.dumps(features(product)), json.dumps(values, sort_keys=True)))

    def observe_case(self, case, excluded=()):
        """Only successful physical calls in an audited V9 case can update chemistry evidence."""
        if not read(case/'validation.json')['passed']:
            return 0
        r = read(case/'result.json')
        target = r['target']['SMILES']
        if molecule_key(target) in excluded:
            return 0
        graph_file = case/'graph.json'
        graph_bytes = graph_file.read_bytes()
        graph = json.loads(graph_bytes)
        emitted = defaultdict(list)
        for e in graph['edges']:
            step = e['step']
            gaps = sum(not graph['facts'][p]['in_stock'] for p in set(step['reactants']))
            reaction = step['product'], tuple(sorted(step['reactants']))
            for origin in e['origins']:
                if origin.startswith('native:'):
                    emitted[step['product'], origin.removeprefix('native:')].append(
                        dict(reactants=step['reactants'], cc=e['cc'], gaps=gaps,
                             reaction=reaction, first=origin==e['origin']))
        source = 'v9:'+sha256(graph_bytes).hexdigest()
        selected = r.get('selected')
        quality = r['metrics']
        selected_steps = {(s['product'], tuple(sorted(s['reactants'])))
                          for s in (selected or {}).get('route',{}).get('steps',())}
        actual_feedback = case/'provider_feedback.json'
        feedback = {(f['product'], f['tool']):f for f in read(actual_feedback)} if actual_feedback.exists() else {}
        last, seen = {}, defaultdict(set)
        before = self.db.total_changes
        with self.db:
            for call in graph['attempts']:
                if call['status'] != 'completed':
                    continue
                p, m = call['product'], call['tool']
                if molecule_key(p) in excluded:
                    continue
                if p == target:
                    continue  # Paired root observations have a separate evaluation protocol.
                previous = last.get(p)
                values = compare(p, emitted[p,m], seen[p], call['seconds'])
                prior = previous['tool'] if previous else '*'
                condition = failure(previous) if previous else 'first'
                self.record(source, target, p, m, 'frontier', prior, condition, values)
                # For historic cases the graph's first origin is the observable
                # chronology. A live event records the same quantity at dispatch.
                primary = [e for e in emitted[p,m] if e['first']]
                if (p,m) in feedback:
                    immediate = feedback[p,m]
                    recorded = {(s[0],tuple(s[1])) for s in immediate['admitted_new']}
                    primary = [e for e in primary if e['reaction'] in recorded]
                credit = any(e['reaction'] in selected_steps for e in primary)
                local = compare(p, primary, seen[p], call['seconds'])
                self.record(source, target, p, m, 'route_feedback', prior, condition,
                            dict(local,route_strict=credit and quality['strict_closed'],
                                 route_stock=credit and quality['stock_closed']))
                seen[p].update(change(p, e['reactants']) for e in emitted[p,m] if e['cc'] > 0)
                last[p] = call
        return self.db.total_changes - before

    def snapshot(self, excluded=()):
        """Only abstract aggregates cross into the inference process."""
        nodes = defaultdict(lambda: defaultdict(list))
        for target,node,model,stage,prior,condition,context,values in self.db.execute(
                'SELECT target,node,provider,stage,prior,failure,context,values_json FROM observations'):
            if target in excluded or node in excluded:
                continue
            values = json.loads(values)
            tags = json.loads(context)
            # The grouped fallback uses a molecule's average, so twenty reruns
            # of one intermediate do not vote as twenty independent molecules.
            for p,c in {(prior,condition), (prior,'*'), ('*',condition), ('*','*')}:
                for tag in ('*', *tags):
                    nodes[stage,p,c,tag,model][node].append(values)
        output = {}
        for k, samples in nodes.items():
            totals = {}
            for metric in METRICS:
                valid = [[v[metric] for v in traces if v.get(metric) is not None] for traces in samples.values()]
                values = [mean(x) for x in valid if x]
                if values:
                    totals[metric] = round(mean(values), 6)
            output['|'.join(k)] = dict(n=len(samples), **totals)
        return output


class Guide:
    def __init__(self, snapshot):
        self.rows = snapshot

    def rank(self, product, providers, *, previous=(), stage='frontier', limit=4):
        last = next((r for r in reversed(previous) if r['status']=='completed'), None)
        prior, condition = (last['tool'],failure(last)) if last else ('*','first')
        scored = []
        for model in providers:
            def rows(scope, p, c):
                return [r for t in features(product) if (r:=self.rows.get('|'.join((scope,p,c,t,model))))
                        and r['n']>=4]
            options = (rows(stage,prior,condition) or rows(stage,'*',condition)
                       or rows(stage,'*','*') or rows('root',prior,'*') or rows('root','*','*'))
            if not options:
                continue
            summary = {k:mean(r[k] for r in options if k in r) for k in METRICS if any(k in r for r in options)}
            score = (.40*summary.get('closed',0.) + .30*summary.get('near',0.)
                     + .20*summary.get('new',0.) + .10*summary.get('positive',0.))
            # The numeric archive supplies only a small, observational tie-break.
            numeric = self.rows.get('|'.join(('numeric_frontier','*','*','*',model)),{})
            if stage=='frontier' and numeric.get('n',0)>=4:
                score += .02*numeric.get('new',0.)
            scored.append(dict(provider=model, score=round(score,5), support=max(r['n'] for r in options),
                               **{k:round(v,4) for k,v in summary.items()}))
        return sorted(scored,key=lambda x:(-x['score'],x.get('seconds',float('inf')),x['provider']))[:limit]


class RouteFeedbackGuide(Guide):
    """Use matched frontier conditions with shrinkage and abstain on weak differences."""

    @staticmethod
    def utility(row):
        return (.45*row.get('route_strict',0.) + .15*row.get('route_stock',0.)
                + .20*row.get('closed',0.) + .15*row.get('near',0.) + .05*row.get('new',0.))

    def rank(self, product, providers, *, previous=(), stage='frontier', limit=4):
        if stage!='frontier':
            return []
        prior = next((r for r in reversed(previous) if r['status']=='completed'),None)
        if prior is None:
            return []
        condition = failure(prior)
        tags = features(product)
        ranked = []
        for model in providers:
            def row(tool, outcome, tag):
                return self.rows.get('|'.join(('route_feedback',tool,outcome,tag,model)),{})
            fallback = row('*','*','*')
            if fallback.get('n',0)<12:
                continue
            score = self.utility(fallback)
            match = []
            for tool, outcome in ((prior['tool'],condition),('*',condition),('*','*')):
                match = [stat for tag in tags if (stat:=row(tool,outcome,tag)) and stat['n']>=8]
                if match:
                    break
            n = max((stat['n'] for stat in match),default=0)
            if n:
                weight = min(.75,n/(n+24))
                score = (1-weight)*score+weight*mean(self.utility(stat) for stat in match)
            ranked.append(dict(provider=model,score=round(score,5),support=n,
                               global_support=fallback['n'],evidence='first-origin route feedback'))
        ranked.sort(key=lambda r:(-r['score'],r['provider']))
        if len(ranked)<2 or ranked[0]['support']<8 or ranked[0]['score']-ranked[1]['score']<.015:
            return []
        return ranked[:limit]
