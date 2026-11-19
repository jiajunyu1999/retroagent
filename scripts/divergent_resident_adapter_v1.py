"""Runtime support for the frozen V9 controller."""

from pathlib import Path

import threading

from retro_prior_harness import Evaluator as OriginalEvaluator

from retrosynthesis_client import atomic_json

from score_route_steps import key

class Stock:
    def __init__(self, tools, folder):
        self.tools, self.folder = tools, folder
        self.cache = {}
        self.lock = threading.Lock()

    def query(self, smiles):
        smiles = list(smiles)
        with self.lock:
            missing = sorted(set(smiles) - self.cache.keys())
            for start in range(0, len(missing), 256):
                self.cache.update(self.tools.call('check_stock', dict(smiles=missing[start:start+256])))
            return {s: self.cache[s] for s in smiles}

    def close(self):
        atomic_json(self.folder / 'stock_cache.json', self.cache)

class Evaluator(OriginalEvaluator):
    def __init__(self, tools, folder):
        # No legacy initializer, old cache reads, or per-target worker startup.
        self.tools, self.folder = tools, Path(folder)
        self.cache_dir = self.folder / 'step_scores'
        self.cache_dir.mkdir(parents=True)
        self.stock = Stock(tools, self.folder)
        self.scores = {}
        self.lock = threading.RLock()

    def batch_scores(self, reactions):
        reactions = list(reactions)
        with self.lock:
            missing = list(dict.fromkeys(r for r in reactions if r not in self.scores))
            for start in range(0, len(missing), 64):
                batch = missing[start:start+64]
                values = self.tools.call('score_reactions', dict(reactions=batch))
                assert len(values) == len(batch)
                for reaction, value in zip(batch, values):
                    assert value['reaction_smiles'] == reaction
                    assert value['chemcensor_version'] == '1.2.0'
                    self.scores[reaction] = value
                    atomic_json(self.cache_dir / (key(reaction) + '.json'), value)
            return [self.scores[r] for r in reactions]

    def scored(self, reaction):
        return self.batch_scores([reaction])[0]

    def evaluate(self, target, route, source):
        # Batch only the steps that the original evaluator will actually score.
        from retro_prior_harness import strict_validate
        if strict_validate(target, route)['structural_valid']:
            self.batch_scores(['.'.join(s['reactants'])+'>>'+s['product'] for s in route['steps']])
        return super().evaluate(target, route, source)

    def close(self):
        self.stock.close()
