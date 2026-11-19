"""Budgeted real evidence tools. No historical per-target caches are loaded."""
from pathlib import Path
from prepare_yield_distributions import canon_side
from retrosynthesis_client import atomic_json
from clean_harness.literature import Literature, query_terms


def fragments(product):
    from rdkit import Chem
    from rdkit.Chem import BRICS
    values=set()
    for mol in Chem.GetMolFrags(BRICS.BreakBRICSBonds(Chem.MolFromSmiles(product)),asMols=True):
        try:
            mol=Chem.DeleteSubstructs(mol,Chem.MolFromSmiles('[#0]'));Chem.SanitizeMol(mol)
            smi=Chem.MolToSmiles(mol,isomericSmiles=True)
            if '.'not in smi and 4<=mol.GetNumHeavyAtoms()<=24 and smi!=product:values.add(smi)
        except (ValueError,RuntimeError):pass
    return sorted(values,key=lambda p:(-Chem.MolFromSmiles(p).GetNumHeavyAtoms(),p))


class EvidenceTools:
    def __init__(self, chemistry, graph, budget, out, cutoff, center_db):
        self.chem,self.graph,self.budget,self.out=chemistry,graph,budget,Path(out)
        self.center_db=center_db;self.db=None
        self.exact_seen=set();self.materials={};self.analog_seen=set();self.analogs=[]
        self.literature=Literature(cutoff);self.paper_seen=set();self.log=[];self.fingerprints={}

    def context(self,wanted,limit=8):
        """Keep relevant earlier stock landmarks, rather than a recent tail."""
        from rdkit import Chem,DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
        def fingerprint(p):
            if p not in self.fingerprints:self.fingerprints[p]=generator.GetFingerprint(Chem.MolFromSmiles(p))
            return self.fingerprints[p]
        wanted=set(wanted)
        def priority(item):
            p,hits=item
            proximity=max((DataStructs.TanimotoSimilarity(fingerprint(p),fingerprint(q))for q in wanted),default=0.)
            return p in wanted,proximity,max((h['similarity']for h in hits),default=0.),p
        rows=[(p,hits)for p,hits in self.materials.items()if hits or p in wanted]
        statuses={r['query']:r['status']for r in self.log if r['kind']=='material'}
        return [dict(query=p,hits=hits,currently_relevant=p in wanted,status=statuses.get(p,'completed'))
                for p,hits in sorted(rows,key=priority,reverse=True)[:limit]]

    def save(self):
        atomic_json(self.out/'evidence.json',dict(calls=self.log,materials=self.materials,
            analogs=self.analogs,literature=dict(searches=self.literature.log,sources=self.literature.sources)))

    def exact(self,product):
        product=canon_side(product)
        if product in self.chem.blocked or product in self.exact_seen:return False
        if product!=self.graph.root and self.graph.facts.get(product,{}).get('in_stock'):return False
        if not self.budget.take('exact_queries'):return False
        self.exact_seen.add(product)
        record=dict(kind='exact',query=product,status='pending')
        self.log.append(record)
        try:
            rows=self.chem.tools.call('retrieve_reactions',dict(product=product,limit=4))
            valid=[r for r in rows if r.get('document_id')and canon_side(r['product'])==product]
            record.update(self.chem.prepare(self.graph,valid,product,'reference:exact'),status='completed')
        except Exception as exc:
            record.update(status='failed',error=str(exc))
        self.save();return True

    def material(self,product):
        product=canon_side(product)
        if product in self.materials:return self.materials[product]
        if not self.budget.take('material_queries'):return []
        record=dict(kind='material',query=product,status='pending');self.log.append(record)
        try:
            hits=self.chem.tools.call('find_materials',dict(smiles=product,limit=5))
            stock=self.chem.ev.stock.query([h['smiles']for h in hits])
            verified=[]
            for hit in hits:
                p=canon_side(hit['smiles'])
                if not stock[hit['smiles']]['in_stock']:raise ValueError('Material index disagrees with exact stock')
                self.graph.node(p);self.graph.facts[p]=stock[hit['smiles']]
                verified.append(dict(smiles=p,similarity=hit['similarity'],in_stock=True))
            self.materials[product]=verified[:4];record.update(status='completed',hits=len(verified))
        except Exception as exc:
            self.materials[product]=[];record.update(status='failed',error=str(exc))
        self.save();return self.materials[product]

    def analog(self,step):
        rxn='.'.join(step['reactants'])+'>>'+step['product']
        if rxn in self.analog_seen:return []
        # Keep two of the shared support slots for the literature fallback.
        if self.budget.used['support_queries']>=max(0,self.budget.full.support_queries-2):return []
        if not self.budget.take('support_queries'):return []
        self.analog_seen.add(rxn);record=dict(kind='analog',query=rxn,status='pending');self.log.append(record)
        try:
            if self.db is None:
                from reaction_evidence import EvidenceDB
                self.db=EvidenceDB(db=self.center_db)
            raw=self.chem.ev.scored(rxn)
            rows=self.db.neighbors(raw,limit=2,scan_limit=64)
            rows=[r for r in rows if canon_side(r['reaction_smiles'].split('>>')[-1])not in self.chem.blocked]
            self.analogs.extend(rows);record.update(status='completed',hits=len(rows))
        except Exception as exc:
            rows=[];record.update(status='failed',error=str(exc))
        self.save();return rows

    def paper(self,request):
        query=query_terms(request)
        for provider in ('crossref','europepmc'):
            token=(provider,query)
            if token in self.paper_seen:continue
            if not self.budget.take('literature_calls'):break
            self.paper_seen.add(token)
            record=dict(kind='literature',query=query,provider=provider,status='pending');self.log.append(record)
            try:
                rows=self.literature.search(request,provider,timeout=max(.1,min(10.,self.budget.remaining_seconds)))
                record.update(status='completed',hits=len(rows))
            except Exception as exc:record.update(status='failed',error=str(exc))
        self.save()

    def close(self):
        self.save()
        if self.db:self.db.close()
