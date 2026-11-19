"""Runtime support for the frozen V9 controller."""

from collections import Counter

from rdkit import Chem

def elements(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid SMILES in element-source check')
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum()>1)

def deficits(product,reactants):
    if isinstance(reactants,str):reactants=reactants.split('.')
    available=Counter()
    for r in reactants:available.update(elements(r))
    return dict(elements(product)-available)
