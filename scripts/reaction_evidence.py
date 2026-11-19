"""Runtime support for the frozen V9 controller."""

import functools
import os

import sqlite3

from pathlib import Path

from rdkit import Chem, DataStructs

from rdkit.Chem import rdFingerprintGenerator

from prepare_yield_distributions import canon_side

ROOT = Path(__file__).resolve().parents[1]

ASSETS = Path(os.environ.get('RETROAGENT_ASSETS', ROOT / 'assets')).resolve()
DB = ASSETS / 'ChemCensor-DB-U2-1.0.0.sqlite'

INDEX = ASSETS / 'product_index.sqlite'

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def build_index(path=INDEX):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), uri=True, timeout=120)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)')
    if not conn.execute("SELECT 1 FROM metadata WHERE key='complete'").fetchone():
        conn.execute('ATTACH DATABASE ? AS source', ('file:' + str(DB) + '?mode=ro',))
        conn.execute('DROP TABLE IF EXISTS precedents')
        conn.execute('CREATE TABLE precedents(product TEXT,reaction TEXT,document_id TEXT)')
        conn.execute("INSERT INTO precedents SELECT substr(reaction_smiles,instr(reaction_smiles,'>>')+2),reaction_smiles,min(document_id) FROM source.reactions GROUP BY reaction_smiles")
        conn.execute('CREATE INDEX by_product ON precedents(product)')
        conn.execute("INSERT OR REPLACE INTO metadata VALUES('source',?)", (str(DB),))
        conn.execute("INSERT OR REPLACE INTO metadata VALUES('complete','true')")
        conn.commit()
    count = conn.execute('SELECT count(*) FROM precedents').fetchone()[0]
    conn.close()
    return count

@functools.lru_cache(maxsize=50000)
def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return FP.GetFingerprint(mol) if mol else None

class EvidenceDB:
    """One instance per target worker: SQLite connections remain thread-local."""
    def __init__(self, index=INDEX, db=DB):
        self.products = sqlite3.connect('file:' + str(index) + '?mode=ro', uri=True)
        self.centers = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)

    def close(self):
        self.products.close(); self.centers.close()

    def exact(self, product, limit=12):
        product = canon_side(product)
        rows = self.products.execute('SELECT reaction,document_id FROM precedents WHERE product=? ORDER BY reaction LIMIT ?', (product, limit)).fetchall()
        out = []
        for rxn, doc in rows:
            left, right = rxn.split('>>')
            # Exact canonical isomeric identity is mandatory, including stereo.
            if canon_side(right) != product or canon_side(left) == product:
                continue
            out.append({'reaction_smiles': rxn, 'document_id': doc,
                        'product': product, 'reactants': canon_side(left).split('.'),
                        'evidence_kind': 'exact_product_precedent'})
        return out

    def neighbors(self, diagnostic, limit=4, scan_limit=256):
        """Select real individual precedents, with explicit FG coverage and similarity.

        A deterministic bounded scan avoids expensive popular-center table scans.
        Missing FGs are retained as caveats, not presented as compatibility proof.
        """
        current = diagnostic['reaction_smiles'].split('>>')[-1]
        query_fp = fingerprint(current)
        out = {}
        for center in sorted(diagnostic.get('centers', []), key=lambda c: c['center_type'], reverse=True):
            if not center['center_has_precedent']:
                continue
            rows = self.centers.execute('SELECT reaction_smiles,fg_signature,sear_signature FROM centers_to_reactions WHERE reaction_center_smiles=? ORDER BY reaction_smiles LIMIT ?', (center['reaction_center_smiles'], scan_limit)).fetchall()
            required = [i for i, value in enumerate(center['fg_signature']) if value]
            for rxn, fg, sear in rows:
                if center.get('is_sear_reaction') and sear != bytes(center['sear_signature']):
                    continue
                product = rxn.split('>>')[-1]; fp = fingerprint(product)
                similarity = DataStructs.TanimotoSimilarity(query_fp, fp) if query_fp is not None and fp is not None else 0
                missing = [i for i in required if i >= len(fg) or not fg[i]]
                item = {'reaction_smiles': rxn, 'matching_center_type': center['center_type'],
                        'product_similarity': round(similarity, 4),
                        'individual_precedent_fg_compatible': not missing,
                        'missing_fg_indices': missing,
                        'evidence_kind': 'analogous_center_precedent',
                        'caveat': 'Analogy is not an executable reaction for the target. Aggregate FG support need not co-occur in one precedent; steric compatibility is unverified.'}
                rank = (not missing, center['center_type'], similarity)
                if rxn not in out or rank > out[rxn][0]: out[rxn] = (rank, item)
            if len(out) >= limit * 4:
                break
        return [item for _, item in sorted(out.values(), key=lambda pair: pair[0], reverse=True)[:limit]]
