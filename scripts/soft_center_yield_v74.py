"""Runtime support for the frozen V9 controller."""

from functools import lru_cache

import math

import sqlite3

import numpy as np

from rdkit import Chem

FG_WEIGHT_FLOOR=.25

@lru_cache(maxsize=100000)
def normalize_core(center):
    """Retain atoms/bonds/charges/aromaticity; ignore stereo/map/isotope labels."""
    try:
        parts=center.split('>')
        if len(parts)!=3:raise ValueError('Not a reaction center')
        result=[]
        for side in [parts[0],parts[2]]:
            mol=Chem.MolFromSmiles(side,sanitize=False)
            if mol is None:raise ValueError('Invalid center side')
            mol.UpdatePropertyCache(strict=False)
            for a in mol.GetAtoms():a.SetAtomMapNum(0)
            Chem.RemoveStereochemistry(mol)
            result.append('.'.join(sorted(Chem.MolToSmiles(mol,isomericSmiles=False).split('.'))))
        return '>>'.join(result)
    except (ValueError,RuntimeError):
        return 'raw:'+center

def compact_mask(signature):
    a=np.asarray(signature,dtype=np.uint8) if not isinstance(signature,(bytes,bytearray)) else np.frombuffer(signature,dtype=np.uint8)
    return np.packbits(a!=0,bitorder='little').tobytes()

def similarity(a,b):
    union=(a|b).bit_count()
    return (a&b).bit_count()/union if union else 1.

def weighted_quantile(bins,q):
    values=[float(v) for v in bins];total=math.fsum(values)
    if total<=0:return None
    threshold=q*total;acc=0.
    for i,n in enumerate(values):
        acc+=n
        if acc+1e-12*total>=threshold:
            if q==.5 and abs(acc-threshold)<=1e-12*total:
                following=next((j for j in range(i+1,10) if values[j]>0),None)
                if following is not None:return ((5+10*i)+(5+10*following))/2
            return float(5+10*i)
    return 95.

def summarize_bins(bins,**metadata):
    n=math.fsum(float(v) for v in bins)
    if n<=0:return dict(status='no_train_yield_evidence',representative_yield_pct=None,reference_reaction_groups=0)
    return dict(representative_yield_pct=weighted_quantile(bins,.5),analog_p10_pct=weighted_quantile(bins,.1),
        analog_p90_pct=weighted_quantile(bins,.9),weighted_bin_counts=[float(v) for v in bins],
        weighted_observation_mass=n,**metadata)

class SoftCoreLookup:
    def __init__(self,path):
        self.db=sqlite3.connect('file:'+str(path)+'?mode=ro&immutable=1',uri=True)
        self.db.execute('PRAGMA mmap_size=2147483648')

    @lru_cache(maxsize=64)
    def contexts(self,core):
        rows=self.db.execute('SELECT fg_mask,reaction_groups,b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,uspto_n,reaxys_n FROM contexts WHERE core=?',(core,)).fetchall()
        if not rows:return None
        masks=[int.from_bytes(r[0],'little') for r in rows]
        counts=np.asarray([r[2:12] for r in rows],dtype=np.float64)
        return masks,counts,sum(r[1] for r in rows),sum(r[12] for r in rows),sum(r[13] for r in rows)

    @lru_cache(maxsize=50000)
    def estimate(self,core,query_mask):
        data=self.contexts(core)
        if data is None:return dict(status='no_RC1_train_precedent',representative_yield_pct=None,reference_reaction_groups=0,core=core)
        masks,counts,groups,uspto,reaxys=data
        query=int.from_bytes(query_mask,'little')
        sims=np.asarray([similarity(query,m) for m in masks])
        weights=FG_WEIGHT_FLOOR+(1-FG_WEIGHT_FLOOR)*sims
        bins=weights@counts
        n=counts.sum(axis=1)
        return summarize_bins(bins,status='soft_RC1_train_median',core=core,reference_reaction_groups=groups,
            n=int(n.sum()),source_counts=dict(USPTO=uspto,Reaxys=reaxys),
            fg_similarity_mean=float((sims@n)/n.sum()),fg_similarity_max=float(sims.max()),
            matched_contexts=len(masks),fg_weight_floor=FG_WEIGHT_FLOOR)

    def close(self):self.db.close()
