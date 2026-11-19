"""Runtime support for the frozen V9 controller."""

import hashlib

import time

def score(smiles):
    import numpy as np
    from chemcensor.basic import Reaction
    from chemcensor.rules.functional_groups import FUNCTIONAL_GROUPS
    from chemcensor.processing.errors import ProcessingError
    from chemcensor.extraction.errors import ExtractionError
    c = CENSOR
    result = {'reaction_smiles': smiles, 'chemcensor_version': '1.2.0',
              'score_with_fg': -1.0, 'score_without_fg': -1.0, 'centers': []}
    reaction = Reaction(reaction_smiles=smiles)
    started = time.monotonic()
    for processor in c._processor.processors:
        try:
            reaction = processor.process(reaction)
        except ProcessingError as e:
            result.update(failure_stage=type(processor).__name__, failure_type=type(e).__name__,
                          failure_message=str(e), elapsed_seconds=time.monotonic()-started)
            return result
    official = c.evaluate_processed(reaction)
    result.update(score_with_fg=official.with_functional_groups,
                  score_without_fg=official.without_functional_groups,
                  mapped_reaction_smiles=reaction.mapped_reaction_smiles,
                  canonical_reaction_smiles=reaction.canonical_smiles,
                  is_sis_reaction=reaction.is_sis_reaction)
    try:
        extracted = c._rc_extractor.extract_rc(reaction)
        names = {r['idx']: r['name'] for r in FUNCTIONAL_GROUPS}
        for level, rc in sorted(extracted.reaction_centers.items()):
            ref = c._reference_fg_for_center_scoring(extracted, rc)
            required = np.flatnonzero(rc.fg_signature).tolist()
            missing = required if ref is None else np.flatnonzero(rc.fg_signature * (1-ref)).tolist()
            precedents = c._manager._conn.execute(
                'SELECT reaction_smiles FROM centers_to_reactions '
                'WHERE reaction_center_smiles=? LIMIT 3', (rc.reaction_center_smiles,)).fetchall()
            result['centers'].append({
                'center_type': int(level), 'reaction_center_smiles': rc.reaction_center_smiles,
                'center_has_precedent': ref is not None,
                'required_functional_groups': [names.get(i, str(i)) for i in required],
                'missing_functional_groups': [names.get(i, str(i)) for i in missing],
                'fg_signature': rc.fg_signature.tolist(),
                'sear_signature': extracted.sear_signature.tolist(),
                'is_sear_reaction': extracted.is_sear_reaction,
                'representative_center_precedents': [x[0] for x in precedents],
                'precedent_note': 'Examples support the center; aggregate FG support need not occur in each example.'})
    except ExtractionError as e:
        result.update(extraction_failure_type=type(e).__name__, extraction_failure_message=str(e))
    result['elapsed_seconds'] = time.monotonic()-started
    return result

def key(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()
