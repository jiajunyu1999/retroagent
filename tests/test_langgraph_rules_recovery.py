"""Supplier hypotheses must work before closure and share bounded work fairly."""
from collections import Counter
import tempfile
import unittest
from unittest.mock import Mock,patch

from rdkit import Chem
from test_langgraph_harness import FakeChemistry
from clean_harness.graph import key
from element_source_guard_v9 import deficits
from retroagent_langgraph.config import Config
from retroagent_langgraph.runtime import Runtime
from retroagent_langgraph.rules import _acyl_halide_supply


class SupplierRecoveryTests(unittest.TestCase):
    def runtime(self,folder,target='CCOCC'):
        return Runtime({'Structure ID':'toy','SMILES':target},folder,FakeChemistry(),Config(),'unused')

    def test_positive_open_branch_can_be_repaired_before_any_stock_closure(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder);root=dict(product='CCOCC',reactants=['CCO','CC'])
            r.graph.add(root,2.,'native:LocalRetro');r.chem.facts(r.graph)
            m=r.chem.evaluate(r.graph,{'CCOCC':root},'toy');r.metrics={1:m}
            self.assertFalse(r.quality()['stock_closed']);r.budget.used['native_calls']=24
            def proposals(product,preferred=()):
                if product=='CCO':yield dict(steps=[dict(product='CCO',reactants=['CC','O'])])
            with patch('retroagent_langgraph.rules.core_proposals',proposals):r.repair({})
            r.combine({})
            self.assertTrue(r.quality()['strict_closed'])
            self.assertEqual(r.rule_work,1)
            self.assertEqual(r.budget.used['native_calls'],24)
            self.assertEqual(r.budget.used['llm_calls'],0)

    def test_multiple_supplier_streams_share_the_same_round_quota(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder);r.budget.used['native_calls']=24
            r.rule_focus=Mock(return_value=[('CCO',[]),('CCCO',[])])
            r.chem.prepare=Mock(return_value={})
            def proposals(product,preferred=()):
                for i in range(50):yield dict(steps=[dict(product=product,reactants=['C'*(i+4),'O'])])
            with patch('retroagent_langgraph.rules.core_proposals',proposals):r.repair({})
            observed=Counter(call.args[2]for call in r.chem.prepare.call_args_list)
            self.assertGreater(observed['CCO'],0);self.assertGreater(observed['CCCO'],0)
            self.assertEqual(r.rule_work,24);self.assertEqual(r.budget.used['search_work'],24)

    def test_acyl_activation_preserves_substrate_and_explicit_atom_sources(self):
        target='C[C@H](F)C(=O)Cl';acid=Chem.MolToSmiles(Chem.MolFromSmiles('C[C@H](F)C(=O)O'),isomericSmiles=True)
        rows=list(_acyl_halide_supply(Chem.MolFromSmiles(target)))
        self.assertEqual(len(rows),2)
        for row in rows:
            step=row['steps'][0]
            self.assertIn(acid,step['reactants'])
            self.assertFalse(deficits(step['product'],step['reactants']))
            self.assertTrue(set(row['requires_stock'])<=set(step['reactants']))

    def test_aromatic_chloride_is_not_misclassified_as_acyl_activation(self):
        self.assertEqual(list(_acyl_halide_supply(Chem.MolFromSmiles('Clc1ccccc1'))),[])

    def test_absent_activating_reagent_does_not_create_a_supplier(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder);r.budget.used['native_calls']=24
            r.rule_focus=Mock(return_value=[('CC(=O)Cl',[])])
            r.chem.prepare=Mock(return_value={});r.repair({})
            self.assertEqual(r.chem.prepare.call_count,0)
            self.assertGreater(r.rule_work,0)

    def supplier_focus(self,folder,score):
        r=self.runtime(folder,'c1ccc(-c2ccccc2)cc1')
        boron=Chem.MolToSmiles(Chem.MolFromSmiles('B(O)(O)c1ccccc1'))
        r.graph.add(dict(product=r.graph.root,reactants=[boron,'Brc1ccccc1']),score,'native:LocalRetro')
        r.chem.facts(r.graph);r.memory.contracts=Mock(return_value=[])
        return r,boron

    def test_positive_native_supplier_remains_eligible_outside_compact_contracts(self):
        with tempfile.TemporaryDirectory()as folder:
            r,boron=self.supplier_focus(folder,2.)
            self.assertIn(boron,dict(r.rule_focus()))

    def test_zero_score_parent_does_not_create_positive_supplier_focus(self):
        with tempfile.TemporaryDirectory()as folder:
            r,boron=self.supplier_focus(folder,0.)
            self.assertNotIn(boron,dict(r.rule_focus()))


if __name__=='__main__':unittest.main()
