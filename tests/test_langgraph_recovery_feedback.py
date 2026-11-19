"""Regressions for preserving valid information from a rejected LLM patch."""
import tempfile
import unittest

from test_langgraph_harness import FakeChemistry
from clean_harness.graph import key
from retroagent_langgraph.config import Config
from retroagent_langgraph.runtime import Runtime


class RejectedPatchRecoveryTests(unittest.TestCase):
    def scenario(self,folder):
        runtime=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},folder,FakeChemistry(),Config(),'unused')
        runtime.graph.add(dict(product='CCOCC',reactants=['CCO','CC']),2.,'native:LocalRetro')
        runtime.chem.facts(runtime.graph)
        # The proposed root lacks O and C sources; its separate observed-subgoal
        # supply and provider decision remain independently valid information.
        value=dict(steps=[dict(product='CCOCC',reactants=['CCC']),
                          dict(product='CCO',reactants=['CC','O'])],
                   next_searches=[dict(product='CCOCC',provider='Graph2Edits')])
        runtime.responses=[('backtrack',dict(repair_anchor='CCOCC'),
                            dict(status='completed',request_key='bad_patch',route=value))]
        return runtime

    def test_rejected_chemistry_keeps_valid_existing_node_tool_decision(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder);runtime.verify({})
            self.assertIn(('CCOCC','Graph2Edits'),runtime.pending)
            self.assertTrue(runtime.memory.rejections)

    def test_rejected_whole_route_can_supply_an_independently_valid_subgoal(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder);runtime.verify({})
            self.assertNotIn(key(dict(product='CCOCC',reactants=['CCC'])),runtime.graph.edges)
            self.assertIn(key(dict(product='CCO',reactants=['CC','O'])),runtime.graph.edges)
            runtime.combine({})
            self.assertTrue(runtime.quality()['strict_closed'])
            self.assertEqual(runtime.quality()['first_cc'],1.5)

    def test_disconnected_unobserved_fragment_is_not_admitted(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder)
            runtime.responses[0][2]['route']['steps'][1]=dict(product='CCN',reactants=['CC','N'])
            runtime.verify({})
            self.assertNotIn(key(dict(product='CCN',reactants=['CC','N'])),runtime.graph.edges)

    def test_valid_short_alternative_survives_invalid_primary(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder);value=runtime.responses[0][2]['route']
            value['steps']=value['steps'][:1]
            value['alternatives']=[dict(steps=[dict(product='CCOCC',reactants=['CCO','CC']),
                                             dict(product='CCO',reactants=['CC','O'])])]
            runtime.verify({});runtime.combine({})
            self.assertTrue(runtime.quality()['strict_closed'])
            self.assertTrue(runtime.memory.rejections)

    def test_wrong_target_alternative_is_not_admitted(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder);value=runtime.responses[0][2]['route']
            value['steps']=value['steps'][:1]
            value['alternatives']=[dict(target_id='another_case',steps=[dict(product='CCO',reactants=['CC','O'])])]
            runtime.verify({})
            self.assertNotIn(key(dict(product='CCO',reactants=['CC','O'])),runtime.graph.edges)

    def test_blocked_queries_do_not_displace_new_supply_query(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.scenario(folder)
            runtime.pending_exact=['CCOCC']*4+['CCO']
            runtime.drain_exact(1)
            self.assertEqual(runtime.budget.used['exact_queries'],1)
            self.assertIn(('retrieve_reactions',dict(product='CCO',limit=4)),runtime.chem.calls)

    def beam_scenario(self,folder,role,count):
        runtime=self.scenario(folder);_,task,record=runtime.responses[0]
        record['route']['steps']=record['route']['steps'][:1]
        bad=dict(steps=[dict(product='CCOCC',reactants=['CCC'])])
        good=dict(steps=[dict(product='CCOCC',reactants=['CCO','CC']),dict(product='CCO',reactants=['CC','O'])])
        record['route']['alternatives']=[dict(bad)for _ in range(count-1)]+[good]
        runtime.responses=[(role,task,record)]
        return runtime

    def test_supply_beam_preserves_valid_fourth_candidate(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.beam_scenario(folder,'supply',4)
            runtime.verify({});runtime.combine({})
            self.assertTrue(runtime.quality()['strict_closed'])
            self.assertEqual(runtime.quality()['first_cc'],1.5)
            self.assertEqual(runtime.budget.used['native_calls'],0)

    def test_supply_beam_over_limit_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.beam_scenario(folder,'supply',5);runtime.verify({})
            self.assertNotIn(key(dict(product='CCO',reactants=['CC','O'])),runtime.graph.edges)

    def test_other_views_keep_smaller_candidate_allowance(self):
        with tempfile.TemporaryDirectory()as folder:
            runtime=self.beam_scenario(folder,'backtrack',2);runtime.verify({})
            self.assertNotIn(key(dict(product='CCO',reactants=['CC','O'])),runtime.graph.edges)

    def test_config_rejects_unbounded_supply_beam(self):
        with self.assertRaises(ValueError):Config(supply_alternatives=5)


if __name__=='__main__':unittest.main()
