"""Learn bounded planning advice from independently evaluated route contrasts."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from clean_harness.portfolio import closed, rank, strict


SKILL = Path(__file__).resolve().parents[1] / 'skills/retro-trajectory-learning/SKILL.md'
PATTERNS = ('replace_unsupported_step', 'avoid_single_supply_gap')
MIN_CASES = 3


def read(path):
    return json.loads(path.read_text())


def route_views(metrics):
    for metric in metrics:
        if not metric['structural_valid'] or metric.get('source_issues'):
            continue
        route = metric['route']
        steps = route['steps']
        diagnostics = metric['step_diagnostics']
        if not steps or len(steps) != len(diagnostics):
            continue
        edges = {s['product']: (tuple(sorted(s['reactants'])), d['score_with_fg'])
                 for s, d in zip(steps, diagnostics)}
        yield metric, route['target_smiles'], edges


def changed_subtree(root, first, second, anchor):
    """Require one changed branch below shared ancestors and fixed siblings."""
    def differences(product):
        a, b = first.get(product), second.get(product)
        if not a or not b or a[0] != b[0]:
            return {product} if a != b else set()
        return set().union(*(differences(child) for child in a[0]))

    return differences(root) == {anchor}


def branches(metrics):
    """Expose measured failure anchors only for actionable, single-gap contrasts."""
    for metric, root, edges in route_views(metrics):
        missing = set(metric.get('open_leaves', ()))
        unsupported = [p for p, (_, cc) in edges.items() if cc <= 0]
        if closed(metric) and len(unsupported) == 1:
            yield 'replace_unsupported_step', unsupported[0], root, edges, None
        if len(missing) == 1 and not unsupported:
            gap = next(iter(missing))
            parents = [p for p, (children, _) in edges.items() if gap in children]
            if len(parents) == 1:
                yield 'avoid_single_supply_gap', parents[0], root, edges, gap


def lessons(metrics):
    """Each pattern contributes at most one outcome per target and run."""
    views = list(route_views(metrics))
    evidence = {}
    for pattern, anchor, root, failed, gap in branches(metrics):
        if evidence.get(pattern) == 1:
            continue
        # Require a different measured edge at that anchor. Extending the failed
        # route or fixing an unrelated sibling supplies no evidence for the skill.
        for metric, other_root, alternative in views:
            if root != other_root or anchor not in alternative:
                continue
            children, cc = alternative[anchor]
            if children == failed[anchor][0] or not changed_subtree(root, failed, alternative, anchor):
                continue
            if strict(metric) and cc > 0 and (gap is None or
                    all(gap not in inputs for inputs, _ in alternative.values())):
                evidence[pattern] = 1
                break
            if ((pattern == 'replace_unsupported_step' and closed(metric) and cc <= 0)
                    or (gap is not None and gap in metric.get('open_leaves', ()))):
                evidence[pattern] = -1
    return evidence


def target_key(smiles):
    from rdkit import Chem

    # Treat stereochemical variants of one connectivity as the same vote.
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('RSI target must parse')
    return hashlib.sha256(Chem.MolToSmiles(molecule, isomericSmiles=False).encode()).hexdigest()


class SkillMemory:
    def __init__(self, path, *, writable=False):
        path = Path(path).resolve()
        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path if writable else path.as_uri() + '?mode=ro',
                                  uri=not writable, timeout=30)
        if writable:
            with self.db:
                self.db.execute('CREATE TABLE IF NOT EXISTS contrasts ('
                                'molecule TEXT, pattern TEXT, outcome INTEGER, '
                                'PRIMARY KEY(molecule, pattern, outcome))')

    def close(self):
        self.db.close()

    def observe(self, case, protected):
        if not read(case / 'validation.json')['passed']:
            return 0
        result = read(case / 'result.json')
        molecule = target_key(result['target']['SMILES'])
        if molecule in protected or result.get('errors') or result['llm_failed']:
            return 0
        votes = lessons(read(case / 'candidate_bank.json'))
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO contrasts VALUES (?, ?, ?)',
                                [(molecule, pattern, value) for pattern, value in votes.items()])
        return len(votes)

    def snapshot(self, excluded):
        directions = {}
        for molecule, pattern, outcome in self.db.execute('SELECT molecule,pattern,outcome FROM contrasts'):
            if molecule not in excluded and pattern in PATTERNS:
                directions.setdefault((molecule, pattern), set()).add(outcome)
        counts = Counter((pattern, next(iter(values)) if len(values) == 1 else 0)
                         for (_, pattern), values in directions.items())
        return [dict(pattern=pattern, support=counts[pattern, 1], counter=counts[pattern, -1],
                     mixed=counts[pattern, 0]) for pattern in PATTERNS]


def advisory(cards, metrics):
    eligible = {card['pattern']: card for card in cards if card['support'] >= MIN_CASES
                and card['support'] > card['counter'] + card['mixed']}
    if not eligible:
        return []
    selected = []
    for pattern, anchor, _, _, gap in branches(sorted(metrics, key=rank, reverse=True)):
        if pattern in eligible:
            selected.append(dict(eligible.pop(pattern), current_anchor=anchor, current_gap=gap))
            if not eligible:
                break
    return selected
