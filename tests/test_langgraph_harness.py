import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from clean_harness.graph import Graph,key
from clean_harness.planner import make_route
from retro_prior_harness import strict_validate
from retroagent_langgraph.config import Config,Budget
from retroagent_langgraph.memory import EpisodeMemory
from retroagent_langgraph.routes import patch_window,prepare


class FakeChemistry:
    def __init__(self):
        self.blocked={'CCOCC'};self.yields=None;self.tools=self;self.calls=[]
        self.ev=SimpleNamespace(stock=self,scored=lambda rxn:dict(centers=[dict(center_type=1,
            reaction_center_smiles='toy-center')],score_with_fg=1.))
    def query(self,products):return {p:dict(in_stock=p in ('CC','O'))for p in products}
    def facts(self,g):g.facts.update(self.query(g.nodes))
    def call(self,name,args):self.calls.append((name,args));return []
    def prepare(self,g,rows,p,origin,**kwargs):
        steps=[g.normalize(s,p if not kwargs.get('patch')else None)for s in rows]
        for s in steps:g.add(s,2. if s['product']==g.root else 1.,origin)
        self.facts(g)
        return dict(admitted=len(steps),supported=len(steps),added=len(steps))
    def predict(self,pairs):
        for p,t in pairs:
            self.calls.append(('predict',dict(product=p,provider=t)))
            rows=[dict(product=p,reactants=['CCO','CC']if p=='CCOCC'else ['CC','O'])]if p in('CCOCC','CCO')else[]
            yield p,t,rows,dict(status='completed',seconds=0.)
    def evaluate(self,g,steps,tid):
        from evaluate_domain_models_test100 import normalize
        r=normalize(make_route(g.root,steps,tid));v=strict_validate(g.root,r)
        stock=all(self.query([p])[p]['in_stock']for p in v['leaves'])
        ds=[dict(score_with_fg=g.edges[key(s)]['cc'],reaction_smiles='.'.join(s['reactants'])+'>>'+s['product'])for s in r['steps']]
        return dict(**v,route=r,stock_closed=stock,full_pass=stock and v['structural_valid'],full_pass_after_source_screen=stock and v['structural_valid'],
            source_issues=[],mean_cc=sum(d['score_with_fg']for d in ds)/len(ds),min_cc=min(d['score_with_fg']for d in ds),
            step_diagnostics=ds,open_leaves=[p for p in v['leaves']if not self.query([p])[p]['in_stock']],
            failed_steps=0)


class LangGraphHarnessTests(unittest.TestCase):
    def test_valid_connected_proposal_keeps_target_first(self):
        g=Graph('CCOCC');g.facts={p:dict(in_stock=p in('CC','O'))for p in('CCOCC','CCO','CC','O')}
        r=prepare(dict(steps=[dict(product='CCOCC',reactants=['CCO','CC']),dict(product='CCO',reactants=['CC','O'])]),g,'toy',{})
        self.assertEqual(r['steps'][0]['product'],g.root)
        self.assertTrue(strict_validate(g.root,r)['structural_valid'])

    def test_depth_window_counts_existing_siblings(self):
        route=dict(target_smiles='CCCCCCO',steps=[dict(product='C'*i+'O',reactants=['C'*(i-1)+'O','C'])for i in range(6,0,-1)])
        self.assertEqual(patch_window(route,'O')['remaining_branch_transformations'],0)
        self.assertEqual(patch_window(route,'CCO')['remaining_branch_transformations'],2)

    def test_source_failure_reports_missing_element(self):
        g=Graph('CCBr');g.facts={'CCBr':dict(in_stock=False)}
        with self.assertRaisesRegex(ValueError,'Br'):
            prepare(dict(steps=[dict(product='CCBr',reactants=['CC'])]),g,'toy',{})

    def test_depth_rejections_move_to_parent_with_real_space(self):
        route=dict(target_smiles='CCCCCCO',steps=[dict(product='C'*i+'O',reactants=['C'*(i-1)+'O','C'])for i in range(6,0,-1)])
        memory=EpisodeMemory();memory.reject(dict(repair_anchor='O'),'Patch exceeds remaining branch depth',1)
        task=memory._task(dict(route=route),'O','repair',lambda x:{})
        self.assertNotEqual(task['repair_anchor'],'O')
        self.assertGreaterEqual(task['remaining_branch_transformations'],3)

    def test_phase_budget_counters_do_not_reset(self):
        from dataclasses import replace
        cfg=Config();b=Budget(cfg);b.cfg=replace(cfg,native_calls=48,llm_calls=9)
        self.assertTrue(b.take('native_calls',48));self.assertFalse(b.take('native_calls'))
        b.cfg=cfg;self.assertTrue(b.take('native_calls',48));self.assertFalse(b.take('native_calls'))
        self.assertTrue(b.take('exact_queries',64));self.assertTrue(b.take('material_queries',36))
        self.assertTrue(b.take('literature_calls',2));self.assertTrue(b.take('support_queries',2))
        self.assertFalse(b.take('support_queries'));self.assertEqual(b.used['evidence_calls'],104)

    def test_transport_errors_do_not_become_negative_chemistry(self):
        m=EpisodeMemory();g=Graph('CCO');g.facts={'CCO':dict(in_stock=False)}
        m.observe_tool('CCO','LocalRetro',dict(status='failed'),g,SimpleNamespace(solved=set()))
        row=m.snapshot()['tool_outcomes'][0]
        self.assertEqual(row['transport_failures'],1);self.assertNotIn('calls',row)

    def test_failure_memory_survives_rounds_and_contexts(self):
        m=EpisodeMemory()
        scorer=lambda rxn:dict(centers=[dict(center_type=1,reaction_center_smiles='same-center')],score_with_fg=0.)
        for round_no,p in [(1,'CCO'),(3,'CCCO')]:
            s=dict(product=p,reactants=['C','O'])
            metric=dict(structural_valid=True,source_issues=[],route=dict(steps=[s]),
                step_diagnostics=[dict(score_with_fg=0.,reaction_smiles='C.O>>'+p)])
            m.observe(metric,round_no,'backtrack',scorer)
        families=m.snapshot()['failed_center_families']
        self.assertEqual(families[0]['rounds'],[1,3]);self.assertEqual(len(families[0]['anchors']),2)
        examples=[rxn for row in m.snapshot()['failed_centers']for rxn in row['failed_reaction_examples']]
        self.assertEqual(set(examples),{'C.O>>CCO','C.O>>CCCO'})

    def test_actual_langgraph_runs_complete_and_or_search(self):
        from retroagent_langgraph.runtime import Runtime
        cfg=Config(native_calls=10,first_native=5,llm_calls=0,phase_llm=0,search_work=1024,
            first_search=512,solver_calls=0,solver_nodes=0,solver_seconds=0.,wall_seconds=30.,yield_reporting=False)
        with tempfile.TemporaryDirectory()as tmp:
            chem=FakeChemistry();runtime=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,chem,cfg,'unused')
            self.assertIn('remember',runtime.build().get_graph().nodes)
            result=runtime.run()
            self.assertTrue(result['metrics']['strict_closed'])
            self.assertEqual(result['metrics']['first_cc'],1.5)
            self.assertEqual(result['costs']['llm_calls'],0)
            self.assertLessEqual(result['costs']['native_calls'],10)
            self.assertEqual(result['phases'],2)

    def test_llm_requested_provider_is_not_randomly_replaced(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            r.chem.facts(r.graph);r.pending=[('CCOCC','Graph2Edits')]
            self.assertEqual(r.tool_pairs()[0],('CCOCC','Graph2Edits'))

    def test_phase_boundary_does_not_discard_unspent_llm_requests(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            r.chem.facts(r.graph);r.budget.used['native_calls']=48
            r.pending=[('CCOCC','Graph2Edits')]
            self.assertEqual(r.tool_pairs(),[])
            self.assertEqual(r.pending,[('CCOCC','Graph2Edits')])

    def test_diagnostic_scheduling_does_not_consume_queued_tool_requests(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            r.chem.facts(r.graph);r.budget.used['native_calls']=24
            r.pending=[('CCOCC','Graph2Edits')]
            r.memory.tasks=Mock(return_value=[])
            self.assertEqual(r.schedule({})['next'],'evidence')
            self.assertEqual(r.pending,[('CCOCC','Graph2Edits')])
            self.assertEqual(r.tool_pairs()[0],('CCOCC','Graph2Edits'))

    def test_high_cc_failed_routes_do_not_hide_strict_quality_targets(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            r.chem.facts(r.graph)
            base=dict(structural_valid=True,stock_closed=True,source_issues=[],n_steps=1,
                route=dict(steps=[dict(product='CCOCC',reactants=['CCO','CC'])]),step_diagnostics=[dict(score_with_fg=1.)])
            good=dict(base,full_pass_after_source_screen=True,mean_cc=1.)
            bad=dict(base,full_pass_after_source_screen=False,mean_cc=5.)
            r.metrics={1:good,2:bad};r.graph.frontier=Mock(return_value=[])
            r.tool_pairs()
            self.assertEqual(r.graph.frontier.call_args.args[1],[good])

    def test_connected_native_route_gap_precedes_unrelated_flat_frontier(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            step=dict(product='CCOCC',reactants=['CCO','CC'])
            r.graph.add(step,2.,'native:LocalRetro');r.graph.node('CCC');r.chem.facts(r.graph)
            r.graph.frontier=Mock(return_value=[dict(product=p,tools=['Graph2Edits'],priority=(0,0,'N0'))for p in('CCC','CCO')])
            branch=dict(bank={'CCOCC':key(step)},unsupported=0,opened={'CCO':1})
            r.guide.candidates=Mock(return_value=[((1.,0.,-2.,1),(key(step),),branch,['CCO'])])
            preview=r.tool_pairs(consume=False)
            self.assertEqual(preview[0][0],'CCO');self.assertFalse(r.route_visits)
            self.assertEqual(r.tool_pairs()[0][0],'CCO');self.assertEqual(r.route_visits[key(step)],1)

    def test_unrelated_nodes_do_not_change_equal_priority_gap_order(self):
        from retroagent_langgraph.runtime import Runtime
        from clean_harness.config import PROVIDERS
        chosen=[]
        for padding in(0,8):
            with tempfile.TemporaryDirectory()as tmp:
                r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
                for i in range(padding):r.graph.node('C'*(i+7))
                r.graph.add(dict(product='CCOCC',reactants=['CCO','CCC']),2.,'native:LocalRetro')
                r.chem.facts(r.graph)
                r.graph.attempts={(r.graph.root,t):dict(status='completed')for t in PROVIDERS}
                chosen.append([p for p,_ in r.tool_pairs()])
        self.assertEqual(chosen[0],chosen[1])

    def test_early_inventory_match_remains_visible_for_current_gap(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            queries=['O','N','CCO']+['C'*i for i in range(3,14)]
            r.evidence.materials={p:[dict(smiles='CC',similarity=.7,in_stock=True)]for p in queries}
            context=r.evidence.context({'CCO'},limit=3)
            self.assertEqual(context[0]['query'],'CCO')
            self.assertEqual(context[0]['hits'][0]['smiles'],'CC')

    def test_material_queries_rotate_and_cached_queries_do_not_use_round_slots(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            r.material_focus=Mock(return_value=['C'*i for i in range(1,21)])
            r.after_evidence='reason';r.evidence_node({})
            self.assertEqual(r.budget.used['material_queries'],6)
            r.evidence_node({})
            self.assertEqual(r.budget.used['material_queries'],12)
            self.assertEqual(len(r.evidence.materials),12)

    def test_quality_mode_prefers_cc_potential_over_root_visit_rotation(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            a=dict(product='CCOCC',reactants=['CCO','CC']);b=dict(product='CCOCC',reactants=['CCCO','C'])
            r.graph.add(a,1.,'native:LocalRetro');r.graph.add(b,2.,'native:LocalRetro');r.chem.facts(r.graph)
            r.metrics={1:dict(structural_valid=True,stock_closed=True,source_issues=[],full_pass_after_source_screen=True,
                mean_cc=1.,n_steps=1,route=dict(steps=[a]),step_diagnostics=[dict(score_with_fg=1.)])}
            r.graph.frontier=Mock(return_value=[dict(product=p,tools=['Graph2Edits'],priority=(0,0,'N0'))for p in('CCO','CCCO')])
            first=dict(bank={'CCOCC':key(a)},unsupported=0,opened={'CCO':1})
            better=dict(bank={'CCOCC':key(b)},unsupported=0,opened={'CCCO':1})
            r.guide.candidates=Mock(return_value=[((1.,1.,-1.,1),(key(a),),first,['CCO']),
                ((0.,1.,-2.,1),(key(b),),better,['CCCO'])])
            r.route_visits[key(b)]=100
            self.assertEqual(r.tool_pairs()[0][0],'CCCO')

    def test_closed_route_quality_view_does_not_bind_to_failed_alternative(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            root=dict(product='CCOCC',reactants=['CCO','CC']);supply=dict(product='CCO',reactants=['CC','O'])
            badroot=dict(product='CCOCC',reactants=['CC','CC','O'])
            for step,score in((root,2.),(supply,1.),(badroot,0.)):r.graph.add(step,score,'native:LocalRetro')
            r.chem.facts(r.graph)
            good=r.chem.evaluate(r.graph,{'CCOCC':root,'CCO':supply},'toy')
            bad=r.chem.evaluate(r.graph,{'CCOCC':badroot},'toy')
            bad.update(full_pass_after_source_screen=False,failed_steps=1)
            tasks=r.memory.tasks([good,bad],r.graph,r.planner,r.chem.ev.scored,1)
            self.assertEqual(len(tasks),3)
            self.assertNotIn('base_route_id',tasks[2])
            self.assertNotIn('repair_anchor',tasks[2])
            self.assertEqual(tasks[2]['remaining_branch_transformations'],6)

    def test_quality_batch_can_complete_two_inputs_of_best_route(self):
        from retroagent_langgraph.runtime import Runtime
        with tempfile.TemporaryDirectory()as tmp:
            r=Runtime({'Structure ID':'toy','SMILES':'CCOCC'},tmp,FakeChemistry(),Config(),'unused')
            root=dict(product='CCOCC',reactants=['CCO','CCCO'])
            r.graph.add(root,2.,'native:LocalRetro');r.graph.node('CCN');r.chem.facts(r.graph)
            r.metrics={1:dict(structural_valid=True,stock_closed=True,source_issues=[],full_pass_after_source_screen=True,
                mean_cc=1.,n_steps=1,route=dict(steps=[root]),step_diagnostics=[dict(score_with_fg=1.)])}
            r.graph.frontier=Mock(return_value=[dict(product=p,tools=['Graph2Edits'],priority=(0,0,'N0'))for p in('CCO','CCN','CCCO')])
            branch=dict(bank={'CCOCC':key(root)},unsupported=0,opened={'CCO':1,'CCCO':1})
            r.guide.candidates=Mock(return_value=[((0.,1.,-2.,1),(key(root),),branch,['CCO','CCCO'])])
            r.budget.used['native_calls']=46
            self.assertEqual({p for p,_ in r.tool_pairs()},{'CCO','CCCO'})


if __name__=='__main__':unittest.main()
