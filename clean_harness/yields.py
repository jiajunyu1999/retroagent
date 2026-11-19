"""Optional final-route yield reporting; never reads historical answers."""
import json, math, sqlite3, os
from pathlib import Path
from prepare_yield_distributions import canon_side
from soft_center_yield_v74 import SoftCoreLookup, compact_mask, normalize_core, summarize_bins
ROOT=Path(__file__).resolve().parents[1]
YIELD_CONTEXTS = Path(os.environ.get("RETROAGENT_ASSETS", ROOT / "assets")).resolve() / "train_RC1_contexts.sqlite"
YIELD_REFERENCE = Path(os.environ.get("RETROAGENT_ASSETS", ROOT / "assets")).resolve() / "yield_reference.sqlite"
def reaction(step):
    return canon_side('.'.join(step['reactants']))+'>>'+canon_side(step['product'])

class YieldEvidence:
    """No historical per-query estimates, center caches, or route files."""
    def __init__(self, scorer, blocked, contexts=YIELD_CONTEXTS, reference=YIELD_REFERENCE):
        self.scorer, self.blocked = scorer, set(blocked)
        self.lookup = SoftCoreLookup(contexts)
        self.reference = sqlite3.connect("file:" + str(reference) + "?mode=ro&immutable=1", uri=True)
        self.used = {}

    def train_reaction(self, step):
        return self.reference.execute("SELECT 1 FROM yield_groups WHERE reaction_smiles=? AND scope='train_reference'",
                                      (reaction(step),)).fetchone() is not None

    def step(self, step):
        rxn = reaction(step)
        if rxn not in self.used:
            exact = None
            if step["product"] not in self.blocked:
                exact = self.reference.execute("SELECT summary_json FROM yield_groups WHERE reaction_smiles=? AND scope='train_reference'",
                                               (rxn,)).fetchone()
            if exact:
                d = json.loads(exact[0])
                value = summarize_bins(d["bin_counts"], status="exact_train_median", n=d["n"],
                                       reference_reaction_groups=1, source_counts=d["source_counts"])
            else:
                raw = self.scorer(".".join(step["reactants"]) + ">>" + step["product"])
                center = next((c for c in raw.get("centers", []) if c.get("center_type") == 1), None)
                value = self.lookup.estimate(normalize_core(center["reaction_center_smiles"]),
                                             compact_mask(center["fg_signature"])) if center else dict(
                                                 status="no_query_RC1", representative_yield_pct=None,
                                                 reference_reaction_groups=0)
            self.used[rxn] = value
        return self.used[rxn]

    def route(self, r):
        records = [self.step(s) for s in r["steps"]]
        values = [d["representative_yield_pct"] for d in records]
        known = [v for v in values if v is not None]
        complete = bool(values) and len(known) == len(values)
        return dict(complete=complete, coverage=len(known) / len(values) if values else 0.,
                    covered_steps=len(known), step_count=len(values), step_evidence=records,
                    all_steps_product_proxy=math.prod(v / 100 for v in values) if complete else None)

    def close(self):
        self.lookup.close(); self.reference.close()
