from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from retroagent_langgraph.provider_memory import (Guide, ProviderMemory, RouteFeedbackGuide,
                                                 change, compare, features, molecule_key)


class ProviderMemoryTests(TestCase):
    def test_route_guide_uses_measured_route_credit_and_abstains_on_ties(self):
        p='CCO'
        previous=[dict(tool='LocalRetro',status='completed',admitted=1,supported=0)]
        rows={}
        for model, strict, near in (('GLN',.9,.2),('RetroSim',.1,.9)):
            rows[f'route_feedback|*|*|*|{model}']=dict(n=30,route_strict=strict,near=near)
            rows[f'route_feedback|LocalRetro|zero_cc|{features(p)[0]}|{model}']=dict(
                n=20,route_strict=strict,near=near)
        guide=RouteFeedbackGuide(rows)
        self.assertEqual(guide.rank(p,['GLN','RetroSim'],previous=previous)[0]['provider'],'GLN')
        self.assertEqual(guide.rank(p,['GLN'],previous=previous),[])
        self.assertEqual(guide.rank(p,['GLN','RetroSim']),[])
        for key in list(rows):
            rows[key].update(route_strict=.1,near=.9)
        self.assertEqual(guide.rank(p,['GLN','RetroSim'],previous=previous),[])

    def test_positive_duplicates_and_inventory_gaps_are_recorded_separately(self):
        product = 'CCO'
        a = dict(reactants=['CC','O'],cc=1,gaps=0)
        b = dict(reactants=['C','CO'],cc=2,gaps=1)
        zero = dict(reactants=['CCC'],cc=0,gaps=0)
        result = compare(product,[a,b,zero],{change(product,a['reactants'])},2.)
        self.assertEqual(result,dict(positive=True,near=True,closed=False,new=True,seconds=2.))
        self.assertFalse(compare(product,[a,zero],{change(product,a['reactants'])},2.)['new'])

    def test_rank_depends_on_previous_outcome_and_counts_unique_nodes(self):
        with TemporaryDirectory() as d:
            path = Path(d)/'providers.sqlite'
            bank = ProviderMemory(path,writable=True)
            molecules=['CCO','CCCO','CCCCO','CCCCCO','CCCCCCO']
            for index,p in enumerate(molecules):
                with bank.db:
                    for outcome in ('zero_cc','supported'):
                        for provider in ('GLN','RetroSim'):
                            winner=(outcome=='zero_cc')==(provider=='GLN')
                            for draw in range(index+1):
                                bank.record(f'round-{draw}',p,p,provider,'frontier','LocalRetro',outcome,
                                            dict(positive=True,new=winner,near=winner,closed=winner,seconds=.8))
            bank.close()
            bank=ProviderMemory(path)
            snapshot=bank.snapshot({molecule_key('OCC')})
            self.assertNotIn(molecule_key('CCO'),str(snapshot))
            guide=Guide(snapshot)
            prior=dict(tool='LocalRetro',status='completed',admitted=2)
            valid=['GLN','RetroSim']
            bad=guide.rank('CCO',valid,previous=[dict(prior,supported=0)])
            good=guide.rank('CCO',valid,previous=[dict(prior,supported=1)])
            self.assertEqual((bad[0]['provider'],good[0]['provider']),('GLN','RetroSim'))
            self.assertEqual(bad[0]['support'],4)
            self.assertEqual(guide.rank('CCO',['MEGAN']),[])
            bank.close()
