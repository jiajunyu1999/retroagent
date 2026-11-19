"""Runtime support for the frozen V9 controller."""

from functools import lru_cache

from rdkit import Chem

@lru_cache(maxsize=100000)
def canon_side(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return '.'.join(sorted(Chem.MolToSmiles(mol, isomericSmiles=True).split('.')))
