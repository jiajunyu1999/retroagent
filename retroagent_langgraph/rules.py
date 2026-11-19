"""Pure, target-independent bond/context hypotheses; always independently scored.

Core bond/context functions extracted from integrated_v002_policy_v2.py,
plus generic explicit acid-chloride supplier hypotheses. No agent runtime,
cached answers, file reads or case-specific recipes.
Original core SHA256: 2e6b9bebde0cbd42a3fc25598d21f44403d6eb9b3aa19deabd682de19c0a30fd
"""
import itertools
from functools import lru_cache


@lru_cache(maxsize=20000)
def direct_supply_kind(product):
    """Cheap generic supplier classes, independent of route IDs or examples."""
    from rdkit import Chem
    mol=Chem.MolFromSmiles(product)
    if any(a.GetAtomicNum()==5 and any(n.GetAtomicNum()==6 for n in a.GetNeighbors())for a in mol.GetAtoms()):
        return 'boron'
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[#6](=[O])-[Cl]')):return 'acyl_halide'
    return None

def _canonical_structure(smiles, stereo=True):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid reaction structure")
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)

def _formed_cuts(step, raw):
    """Map actual newly formed product bonds to canonical product atom indices.

    Missing/ambiguous mapping returns no preference, never a failure diagnosis.
    Used for proposal ordering only; no atom mapping is used for acceptance.
    """
    from rdkit import Chem
    try:
        left, right = raw['mapped_reaction_smiles'].split('>>')
        rm, pm = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
        before = {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),
                                b.GetEndAtom().GetAtomMapNum()))) for b in rm.GetBonds()}
        newly = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in pm.GetBonds()
                 if b.GetBeginAtom().GetAtomMapNum() and b.GetEndAtom().GetAtomMapNum()
                 and tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),
                                   b.GetEndAtom().GetAtomMapNum()))) not in before]
        for a in pm.GetAtoms():
            a.SetAtomMapNum(0)
        target = Chem.MolFromSmiles(_canonical_structure(step['product']))
        match = target.GetSubstructMatch(pm, useChirality=True)
        if len(match) != target.GetNumAtoms() or pm.GetNumAtoms() != target.GetNumAtoms():
            return []
        return [tuple(sorted((match[a], match[b]))) for a, b in newly]
    except (KeyError, ValueError, AttributeError, RuntimeError):
        return []

def _capped(fragment, handle):
    """Replace the single cut dummy by a real chemical handle, preserving stereo."""
    from rdkit import Chem
    dummies = [a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        raise ValueError('one_cut_required')
    idx = dummies[0]
    neighbors = list(fragment.GetAtomWithIdx(idx).GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError('one_attachment_required')
    neighbor = neighbors[0].GetIdx()
    mol = Chem.RWMol(fragment)
    if handle:
        addition = Chem.MolFromSmiles(handle)
        offset = mol.GetNumAtoms()
        mol = Chem.RWMol(Chem.CombineMols(mol, addition))
        mol.AddBond(neighbor, offset, Chem.BondType.SINGLE)
    else:
        # H-capping must be explicit for aromatic N (pyrrole/pyrazole), which
        # otherwise becomes invalid aromatic n after detaching its substituent.
        atom = mol.GetAtomWithIdx(neighbor)
        atom.SetNumExplicitHs(atom.GetTotalNumHs()+1)
        atom.SetNoImplicit(True)
    mol.RemoveAtom(idx)
    result = mol.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, isomericSmiles=True)

def _bond_proposals(mol, preferred=()):
    """Both coupling polarities and actual heteroatom sources at eligible bonds.

    Pure proposals: neither graph consistency nor a Suzuki label implies CC>0.
    Ring bonds, stereogenic C-C bonds and target atom/stereo edits are excluded.
    """
    from rdkit import Chem
    product = Chem.MolToSmiles(mol, isomericSmiles=True)
    preferred = {tuple(sorted(pair)) for pair in preferred}
    bonds = sorted(mol.GetBonds(), key=lambda b: (
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) not in preferred,
        b.GetIdx()))
    for bond in bonds:
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        cc = a.GetIsAromatic() and b.GetIsAromatic() and a.GetAtomicNum() == b.GetAtomicNum() == 6
        aryl_hetero = ((a.GetIsAromatic() and a.GetAtomicNum() == 6 and b.GetAtomicNum() in (7, 16))
                      or (b.GetIsAromatic() and b.GetAtomicNum() == 6 and a.GetAtomicNum() in (7, 16)))
        if not (cc or aryl_hetero):
            continue
        cut = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(911, 912)])
        parts = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
        if len(parts) != 2:
            continue
        cut_pair = tuple(sorted((a.GetIdx(), b.GetIdx())))
        if cc:
            # Each leaving-group choice visits BOTH polarities before the next.
            choices = [(x, y) for x in ('Br', 'Cl', 'I')
                       for y in ('B(O)O', 'B1OC(C)(C)C(C)(C)O1')]
            assignments = [(x, y) for pair in choices for x, y in (pair, pair[::-1])]
        else:
            # Locate the carbon side without relying on FragmentOnBonds order.
            neighbor = next(a for a in parts[0].GetAtoms() if a.GetAtomicNum() == 0).GetNeighbors()[0]
            assignments = [(x, None) if neighbor.GetAtomicNum() == 6 else (None, x)
                           for x in ('Br', 'Cl', 'F', 'I')]
        for handles in assignments:
            try:
                rs = sorted(_capped(f, h) for f, h in zip(parts, handles))
            except (ValueError, RuntimeError):
                continue
            if product in rs:
                continue
            yield dict(family='bond_polarity' if cc else 'heteroatom_connection',
                cut=list(cut_pair), steps=[dict(product=product, reactants=rs,
                    reaction_type='Failure-guided explicit precursor reconstruction')])

def _contexts(mol, preferred=()):
    """Move assembly into a protected/ester context; return a real removal edge.

    Every single-site variant retains all original atoms/stereotags. The removal
    edge and the new assembly must each get their own original positive score.
    """
    from rdkit import Chem
    product = Chem.MolToSmiles(mol, isomericSmiles=True)
    sites = []
    acid_oxygen = {match[2] for match in mol.GetSubstructMatches(
        Chem.MolFromSmarts('[C](=[O])[O;H1]'))}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in acid_oxygen:
            sites.append((atom.GetIdx(), 'C', 'ester_context'))
        elif atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0 and atom.GetFormalCharge() == 0:
            # Free amines or aromatic NH; omit ordinary amide nitrogens.
            amide = any(n.GetAtomicNum() == 6 and any(
                x.GetBondType() == Chem.BondType.DOUBLE and
                x.GetOtherAtom(n).GetAtomicNum() == 8 for x in n.GetBonds())
                for n in atom.GetNeighbors())
            if atom.GetIsAromatic() or not amide:
                sites.append((atom.GetIdx(), 'C(=O)OC(C)(C)C', 'NH_Boc_context'))
    endpoints = {i for cut in preferred for i in cut}
    if endpoints:
        sites.sort(key=lambda site: min(len(Chem.GetShortestPath(mol, site[0], i))
                                       if site[0] != i else 1 for i in endpoints))
    for idx, group, family in sites[:4]:
        edit = Chem.RWMol(mol)
        atom = edit.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs()-1))
        atom.SetNoImplicit(True)
        offset = edit.GetNumAtoms()
        edit = Chem.RWMol(Chem.CombineMols(edit, Chem.MolFromSmiles(group)))
        edit.AddBond(idx, offset, Chem.BondType.SINGLE)
        masked = edit.GetMol()
        try:
            Chem.SanitizeMol(masked)
            precursor = Chem.MolToSmiles(masked, isomericSmiles=True)
        except (ValueError, RuntimeError):
            continue
        yield masked, dict(product=product, reactants=[precursor],
                           reaction_type='Context reconstruction: deprotection or ester hydrolysis'), family

def _boron_supply(mol):
    """Supply the actual proposed organoboron branch, not just change its name."""
    from rdkit import Chem
    product = Chem.MolToSmiles(mol, isomericSmiles=True)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if sorted((a.GetAtomicNum(), b.GetAtomicNum())) != [5, 6] or bond.IsInRing():
            continue
        if not (a.GetIsAromatic() or b.GetIsAromatic()):
            continue
        parts = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, [bond.GetIdx()]), asMols=True)
        if len(parts) != 2:
            continue
        aryl = next(f for f in parts if next(x for x in f.GetAtoms() if x.GetAtomicNum() == 0).GetNeighbors()[0].GetAtomicNum() == 6)
        for handle, others in [
            ('B1OC(C)(C)C(C)(C)O1', []),
            ('B(O)O', ['OC(C)(C)C(C)(C)O']),
            ('Br', ['CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C']),
            ('I', ['CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C']),
            ('Br', ['OB(O)B(O)O'])]:
            try:
                rs = [_capped(aryl, handle), *others]
            except (ValueError, RuntimeError):
                continue
            if product not in rs:
                yield dict(family='boron_supply', cut=[], steps=[dict(product=product,
                    reactants=sorted(rs), reaction_type='Explicit organoboron supply hypothesis')])

def _acyl_halide_supply(mol):
    """Explicit acid-chloride activation hypotheses, with real atom sources.

    Require stocked activating reagents; never add atomic chlorine merely to
    balance a missing source. Every proposed transformation is independently
    scored; the template itself establishes neither support nor closure.
    """
    from rdkit import Chem
    product=Chem.MolToSmiles(mol,isomericSmiles=True)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()!=17 or atom.GetDegree()!=1:continue
        carbon=atom.GetNeighbors()[0]
        if carbon.GetAtomicNum()!=6 or carbon.GetIsAromatic():continue
        if not any(b.GetBondType()==Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetAtomicNum()==8
                   for b in carbon.GetBonds()):continue
        edit=Chem.RWMol(mol);oxygen=edit.GetAtomWithIdx(atom.GetIdx())
        oxygen.SetAtomicNum(8);oxygen.SetFormalCharge(0);oxygen.SetNumExplicitHs(1);oxygen.SetNoImplicit(True)
        acid=edit.GetMol();Chem.SanitizeMol(acid)
        precursor=Chem.MolToSmiles(acid,isomericSmiles=True)
        for reagent in ('O=S(Cl)Cl','O=C(Cl)C(=O)Cl'):
            reagent=Chem.MolToSmiles(Chem.MolFromSmiles(reagent),isomericSmiles=True)
            yield dict(family='acyl_halide_supply',cut=[],requires_stock=[reagent],
                steps=[dict(product=product,reactants=sorted([precursor,reagent]),
                    reaction_type='Explicit acid chloride activation hypothesis')])


def _core_proposals(product, preferred=()):
    """Fair stream: unchanged context competes with each new assembly context."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(_canonical_structure(product))
    streams = [iter(_acyl_halide_supply(mol)),iter(_boron_supply(mol)), iter(_bond_proposals(mol, preferred))]
    for masked, removal, family in _contexts(mol, preferred):
        def linked(masked=masked, removal=removal, family=family):
            # Protection must repair boron SUPPLY as well as the final coupling.
            # Otherwise a positive protected core only invents a new dead leaf.
            for proposal in itertools.chain(_acyl_halide_supply(masked),_boron_supply(masked),
                                             _bond_proposals(masked, preferred)):
                yield dict(proposal, family=family, steps=[removal, *proposal['steps']])
        streams.append(iter(linked()))
    while streams:
        remaining = []
        for stream in streams:
            try:
                yield next(stream)
                remaining.append(stream)
            except StopIteration:
                pass
        streams = remaining

core_proposals=_core_proposals
formed_cuts=_formed_cuts
