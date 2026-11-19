"""The complete route-selection policy. No cohort-specific thresholds or answers."""
from functools import lru_cache
from itertools import combinations


def identity(metric):
    return tuple(sorted((s["product"], tuple(sorted(s["reactants"])))
                        for s in metric["route"]["steps"]))


def closed(m):
    return bool(m.get("structural_valid") and m.get("stock_closed")
                and not m.get("source_issues"))


def strict(m):
    return bool(closed(m) and m.get("full_pass_after_source_screen"))


def rank(m):
    # A high score on an unfinished route never outranks an actual solution.
    return (closed(m), strict(m), m["mean_cc"], -m["n_steps"])


@lru_cache(maxsize=50000)
def features(key):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    target, steps = key
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                   includeChirality=True)
    def bits(s):
        return set(fp.GetFingerprint(Chem.MolFromSmiles(s)).GetOnBits())
    root = next((rs for p, rs in steps if p == target), ())
    reactants = set()
    for r in root:
        reactants.update(bits(r))
    return bits(target) ^ reactants, set(steps)


def distance(a, b):
    # Keep the already published numeric18 evaluation definition unchanged.
    def key(route):
        return route["target_smiles"], tuple(sorted(
            (s["product"], tuple(sorted(s["reactants"]))) for s in route["steps"]))
    x, y = features(key(a)), features(key(b))
    return sum(w * (1 - len(u & v) / max(1, len(u | v)))
               for w, u, v in zip((.7, .3), x, y))


def diversity(metrics):
    pairs = list(combinations(metrics, 2))
    return sum(distance(a["route"], b["route"]) for a, b in pairs) / len(pairs) if pairs else 0.


def select(metrics, keep=20, cc_margin=.10):
    """Lock the best primary; diversify only inside a small CC window.

    Each remaining slot first chooses the highest available closure tier, then
    admits routes within cc_margin of that tier's best CC. Yield completeness,
    average distance to the chosen routes, yield proxy and CC break ties, in
    that order. There is no yield-only rerun and no replacement of the primary.
    """
    if keep < 1 or cc_margin < 0:
        raise ValueError("keep must be positive and cc_margin nonnegative")
    bank = {identity(m): m for m in metrics
            if m.get("structural_valid") and m["route"]["steps"]}
    if not bank:
        return dict(selected=None, output_routes=[])
    primary = max(bank.values(), key=lambda m: (*rank(m), identity(m)))
    chosen = [primary]
    del bank[identity(primary)]
    while bank and len(chosen) < keep:
        tier = max((closed(m), strict(m)) for m in bank.values())
        eligible = [m for m in bank.values() if (closed(m), strict(m)) == tier]
        best_cc = max(m["mean_cc"] for m in eligible)
        eligible = [m for m in eligible if m["mean_cc"] >= best_cc - cc_margin - 1e-10]
        def secondary(m):
            y = m.get("yield_proxy", {})
            peers = [p for p in chosen if (closed(p), strict(p)) == tier]
            novelty = sum(distance(m["route"], p["route"]) for p in peers) / len(peers) if peers else 0.
            return (bool(y.get("complete")), novelty,
                    y.get("all_steps_product_proxy") or 0., y.get("coverage", 0.),
                    m["mean_cc"], -m["n_steps"], identity(m))
        winner = max(eligible, key=secondary)
        chosen.append(winner)
        del bank[identity(winner)]
    return dict(selected=primary, output_routes=chosen)
