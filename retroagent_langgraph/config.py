from dataclasses import dataclass, asdict
from threading import RLock
import time


@dataclass(frozen=True)
class Config:
    native_calls: int = 96
    llm_calls: int = 18
    exact_queries: int = 64
    material_queries: int = 36
    support_queries: int = 4  # Reaction analogs and actual literature SHARE this.
    evidence_calls: int = 104
    search_work: int = 12288
    solver_calls: int = 4
    solver_nodes: int = 1024
    solver_seconds: float = 8.
    wall_seconds: float = 900.
    routes: int = 20
    seed: int = 20261116
    first_native: int = 48
    first_search: int = 6144
    phase_llm: int = 9
    warmup_native: int = 24
    feedback_every: int = 12
    quality_llm_groups: int = 6
    supply_alternatives: int = 4  # One primary + up to4 short supply candidates.
    rule_work: int = 768  # Included in search_work; never extra model calls.
    yield_reporting: bool = True
    stop_at_strict: bool = False
    literature_cutoff: str = '2026-10-01'

    def __post_init__(self):
        for key in ('native_calls','llm_calls','exact_queries','material_queries','support_queries',
                    'evidence_calls','search_work','solver_calls','solver_nodes','first_native',
                    'first_search','phase_llm','warmup_native','feedback_every','quality_llm_groups','rule_work','supply_alternatives'):
            if type(getattr(self,key)) is not int or getattr(self,key)<0:
                raise ValueError(key+' must be a nonnegative integer')
        if (self.routes<1 or self.wall_seconds<=0 or self.first_native>self.native_calls
                or self.first_search>self.search_work or self.phase_llm>self.llm_calls):
            raise ValueError('Invalid shared or phase limits')
        if self.supply_alternatives>4:raise ValueError('At most4 short supply alternatives')


class Budget:
    """One counter set survives the phase transition. No implicit retries."""
    def __init__(self, cfg):
        self.cfg=self.full=cfg
        self.started=time.monotonic()
        self.used=dict(native_calls=0,llm_calls=0,evidence_calls=0,search_work=0,
                       exact_queries=0,material_queries=0,support_queries=0,literature_calls=0)
        self.lock=RLock()

    @property
    def remaining_seconds(self):
        return max(0.,self.full.wall_seconds-(time.monotonic()-self.started))

    def take(self,kind,n=1):
        if type(n)is not int or n<0:raise ValueError('Invalid budget debit')
        with self.lock:
            if self.remaining_seconds<=0:return False
            if kind in ('exact_queries','material_queries','support_queries','literature_calls'):
                category='support_queries'if kind=='literature_calls'else kind
                if self.used[category]+n>getattr(self.full,category):return False
                if self.used['evidence_calls']+n>self.full.evidence_calls:return False
                self.used[category]+=n
                if kind=='literature_calls':self.used[kind]+=n
                self.used['evidence_calls']+=n
                return True
            if kind not in self.used or self.used[kind]+n>getattr(self.cfg,kind):return False
            self.used[kind]+=n
            return True

    def report(self):
        return dict(**self.used,elapsed_seconds=time.monotonic()-self.started,
                    limits=asdict(self.full),phase_limits=asdict(self.cfg))
