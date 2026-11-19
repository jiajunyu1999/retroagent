"""Regression tests for expanded-vs-dead branches and shared search budgets."""
from dataclasses import replace
import tempfile
import unittest

from test_langgraph_harness import FakeChemistry
from clean_harness.config import PROVIDERS
from clean_harness.graph import key
from retroagent_langgraph.config import Config
from retroagent_langgraph.runtime import Runtime


class SearchRecoveryTests(unittest.TestCase):
    def runtime(self,folder):
        return Runtime({'Structure ID':'toy','SMILES':'CCOCC'},folder,FakeChemistry(),
                       Config(solver_calls=0,solver_nodes=0),'unused')

    def test_fully_expanded_sibling_with_known_supply_is_not_a_dead_end(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder);root=dict(product='CCOCC',reactants=['CCO','CCC'])
            r.graph.add(root,2.,'native:LocalRetro')
            r.graph.add(dict(product='CCO',reactants=['CC','O']),1.,'native:LocalRetro')
            r.chem.facts(r.graph);r.retrieved.add('CCO')
            r.graph.attempts={('CCO',t):dict(status='completed')for t in PROVIDERS}
            state=dict(bank={'CCOCC':key(root)},opened={'CCO':1,'CCC':1},mean_cc=2.,unsupported=0)
            ranked=r.guide.rank(r,state)
            self.assertIsNotNone(ranked)
            self.assertEqual(ranked[1],['CCC'])

    def test_exhausted_node_without_any_supply_remains_a_dead_end(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder);root=dict(product='CCOCC',reactants=['CCO','CCC'])
            r.graph.add(root,2.,'native:LocalRetro');r.chem.facts(r.graph);r.retrieved.add('CCO')
            r.graph.attempts={('CCO',t):dict(status='completed')for t in PROVIDERS}
            state=dict(bank={'CCOCC':key(root)},opened={'CCO':1,'CCC':1},mean_cc=2.,unsupported=0)
            self.assertIsNone(r.guide.rank(r,state))

    def test_new_reaction_still_enters_search_after_4096_prior_states(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder)
            r.graph.add(dict(product='CCOCC',reactants=['CCO','CC']),2.,'native:LocalRetro')
            r.chem.facts(r.graph)
            r.guide.seen={('previously_rejected',i)for i in range(4096)};r.guide.work=4096
            r.budget.used['search_work']=4096
            r.guide.update(r,quota=8)
            self.assertTrue(r.guide.states)
            self.assertGreater(r.guide.work,4096)
            self.assertLessEqual(r.budget.used['search_work'],r.cfg.search_work)

    def test_phase_two_resumes_same_graph_when_shared_budget_opens(self):
        with tempfile.TemporaryDirectory()as folder:
            r=self.runtime(folder)
            r.graph.add(dict(product='CCOCC',reactants=['CCO','CC']),2.,'native:LocalRetro')
            r.chem.facts(r.graph);r.budget.used['search_work']=r.cfg.search_work
            r.guide.update(r,quota=8);self.assertFalse(r.guide.states)
            r.cfg=r.full;r.budget.cfg=r.full;r.planner.cfg=r.full
            r.guide.update(r,quota=8)
            self.assertTrue(r.guide.states)


if __name__=='__main__':unittest.main()
