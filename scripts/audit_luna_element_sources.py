"""Runtime support for the frozen V9 controller."""

from rdkit import Chem

def missing_elements(step):
    supplied = {a.GetSymbol() for r in step['reactants'] for a in Chem.MolFromSmiles(r).GetAtoms()}
    required = {a.GetSymbol() for a in Chem.MolFromSmiles(step['product']).GetAtoms()}
    return sorted(required-supplied)
