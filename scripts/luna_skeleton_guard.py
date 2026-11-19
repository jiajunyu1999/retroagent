"""Runtime support for the frozen V9 controller."""

from rdkit import Chem

from audit_luna_element_sources import missing_elements

def carbon_count(smiles):
    return sum(a.GetAtomicNum() == 6 for a in Chem.MolFromSmiles(smiles).GetAtoms())

def explicit_source_deficits(step):
    deficits=missing_elements(step)
    required=carbon_count(step['product'])
    supplied=sum(carbon_count(r) for r in step['reactants'])
    if required > supplied:
        deficits.append('Explicit carbon deficit: product=%d, inputs=%d' % (required,supplied))
    return deficits
