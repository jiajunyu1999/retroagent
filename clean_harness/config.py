from dataclasses import dataclass, asdict
import time

PROVIDERS = ("LocalRetro", "RetroChimeraEdit", "AiZynthFinder", "Graph2Edits", "RetroChimeraDeNovo")


@dataclass(frozen=True)
class Config:
    native_calls: int = 96
    evidence_calls: int = 32
    llm_calls: int = 6
    search_work: int = 12288
    solver_calls: int = 4
    solver_nodes: int = 1024
    solver_seconds: float = 8.
    wall_seconds: float = 900.
    routes: int = 20
    literature_calls: int = 4  # Included in evidence_calls, NOT an extra budget.
    feedback_every: int = 16
    seed: int = 0
    yield_reporting: bool = True
    literature_cutoff: str = "2026-10-01"
    strict_repair: bool = False
    route_search: bool = False

    def __post_init__(self):
        for k in ("native_calls", "evidence_calls", "llm_calls", "search_work", "solver_calls", "solver_nodes", "literature_calls"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError(k + " must be a nonnegative integer")
        if self.routes < 1 or self.feedback_every < 1 or self.wall_seconds <= 0 or self.solver_seconds < 0:
            raise ValueError("Invalid route, feedback or time limit")


class Budget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.started = time.monotonic()
        self.used = dict(native_calls=0, evidence_calls=0, llm_calls=0, search_work=0, literature_calls=0)

    @property
    def remaining_seconds(self):
        return max(0., self.cfg.wall_seconds - (time.monotonic() - self.started))

    def take(self, kind, n=1):
        if n < 0:
            raise ValueError("Negative budget debit")
        if self.remaining_seconds <= 0 or self.used[kind] + n > getattr(self.cfg, kind):
            return False
        self.used[kind] += n
        return True

    def report(self):
        return dict(**self.used, elapsed_seconds=time.monotonic()-self.started, limits=asdict(self.cfg))
