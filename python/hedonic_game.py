 
import random
from typing import Callable, Dict, List, Optional, Tuple

from wealth_process_V_opt import *
 
 
# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
 
class Miner:
    def __init__(self, h: float, c: float, x: float, idx: int, gamma: float = None):
        self.h = h
        self.c = c
        self.x = x          # initial reserve, used by the ruin-theoretic dividend objective
        self.gamma = gamma  # risk aversion, used by the mean-variance objective
        self.idx = idx
 
 
class PPS_Pool:
    """Also used to represent solo mining: PPS_Pool(f=0.0, delta=1.0, name='solo')."""
    __slots__ = ("f", "delta", "name")
 
    def __init__(self, f: float = 0.0, delta: float = 1.0, name: str = None):
        self.f = f
        self.delta = delta
        self.name = name
 
 
class PPLNS_Pool:
    __slots__ = ("members", "H", "fee_function", "f", "name")
 
    def __init__(self, members: List[Miner], fee_function: Callable[[float], float], name: str):
        self.members = set(members) if members else set()
        self.H = sum(m.h for m in self.members)
        self.fee_function = fee_function
        self.name = name
        self.f = self._price()
 
    def _price(self) -> float:
        # k == 1 (or 0) is the solo boundary case: no coordination, no fee.
        return 0.0 if len(self.members) <= 1 else self.fee_function(self.H)
 
    def add(self, miner: Miner) -> None:
        self.members.add(miner)
        self.H += miner.h
        self.f = self._price()
 
    def remove(self, miner: Miner) -> None:
        self.members.discard(miner)
        self.H -= miner.h
        self.f = self._price()
 
 
class Environment:
    """Blockchain / network primitives. Same role as your blockchain_network,
    trimmed to what the payoff engine needs."""
 
    def __init__(self, Lam: float, b: float, q: float, H: float):
        self.Lam = Lam   # network-wide block arrival rate
        self.b = b       # block reward
        self.q = q       # discount rate (dividend objective)
        self.H = H       # total network hashrate
 


# ---------------------------------------------------------------------------
# Payoff engine: turns a (miner, target) pair into a scalar score, without
# mutating anything. Both objectives share the same (rate, jump) primitives
# of the underlying compound-Poisson payout process.
# ---------------------------------------------------------------------------
 
class PayoffEngine:
    def __init__(self, env: Environment, objective: str, wealth_process_cls=None):
        assert objective in ("dividend", "mean_variance")
        self.env = env
        self.objective = objective
        self.wealth_process_cls = wealth_process_cls  # required only for 'dividend'
 
    def moments_pps(self, miner: Miner, pool: PPS_Pool) -> Tuple[float, float]:
        env = self.env
        rate = env.Lam * miner.h / env.H / pool.delta
        jump = env.b * (1 - pool.f) * pool.delta
        return rate, jump
 
    def moments_pplns(self, miner: Miner, pool: PPLNS_Pool) -> Tuple[float, float]:
        env = self.env
        already = miner in pool.members
        H_pool = pool.H if already else pool.H + miner.h
        k = len(pool.members) if already else len(pool.members) + 1
        fee = 0.0 if k <= 1 else pool.fee_function(H_pool)
        rate = env.Lam * H_pool / env.H
        jump = env.b * (1 - fee) * miner.h / H_pool
        return rate, jump
 
    def score(self, miner: Miner, rate: float, jump: float) -> float:
        if self.objective == "mean_variance":
            mean, var = rate * jump, rate * jump ** 2
            gamma = miner.gamma or 0.0
            return mean - miner.c * miner.h - 0.5 * gamma * var
        # dividend objective: plug in your ruin-theoretic wealth process
        wp = wealth_process(rate, jump, miner.c, 0, 1)
        return wp.V(miner.x, self.env.q)[1]


# ---------------------------------------------------------------------------
# Game orchestrator: owns all mutable state and the two-phase dynamics.
# ---------------------------------------------------------------------------
 
class Game:
    def __init__(self, env: Environment, miners: List[Miner], pps_pools: Dict[str, PPS_Pool],
                 fee_function: Callable[[float], float], objective: str,
                 wealth_process_cls=None):
        assert "solo" in pps_pools, "include PPS_Pool(f=0.0, delta=1.0, name='solo') in pps_pools"
        self.env = env
        self.miners = {m.idx: m for m in miners}
        self.pps_pools = dict(pps_pools)
        self.pplns_pools: Dict[str, PPLNS_Pool] = {}
        self.fee_function = fee_function
        self.engine = PayoffEngine(env, objective, wealth_process_cls)
        self.assignment: Dict[int, Tuple[str, str]] = {m.idx: ("pps", "solo") for m in miners}
        self._next_pool_id = 0
 
    # -- pure evaluation, no mutation --
 
    def payoff_at(self, miner: Miner, target: Tuple[str, str]) -> float:
        kind, name = target
        if kind == "pps":
            rate, jump = self.engine.moments_pps(miner, self.pps_pools[name])
        else:
            rate, jump = self.engine.moments_pplns(miner, self.pplns_pools[name])
        return self.engine.score(miner, rate, jump)
 
    def current_payoff(self, miner: Miner) -> float:
        return self.payoff_at(miner, self.assignment[miner.idx])
 
    def deviation_targets(self, miner: Miner) -> List[Tuple[str, str]]:
        """Unilateral, open-enrollment moves only: any PPS contract, or any
        EXISTING pplns pool with >=2 members. Founding a pool from scratch is
        handled separately -- see propose_founding."""
        targets = [("pps", name) for name in self.pps_pools]
        targets += [("pplns", name) for name, pool in self.pplns_pools.items()
                    if len(pool.members) >= 2]
        return targets
 
    def best_response(self, miner: Miner) -> Tuple[Tuple[str, str], float]:
        options = {t: self.payoff_at(miner, t) for t in self.deviation_targets(miner)}
        target = max(options, key=options.get)
        return target, options[target]
 
    # -- mutation --
 
    def move(self, miner: Miner, target: Tuple[str, str]) -> None:
        old_kind, old_name = self.assignment[miner.idx]
        if old_kind == "pplns":
            pool = self.pplns_pools[old_name]
            pool.remove(miner)
            if not pool.members:
                del self.pplns_pools[old_name]
        self.assignment[miner.idx] = target
        if target[0] == "pplns":
            self.pplns_pools[target[1]].add(miner)
 
    def propose_founding(self, group: List[Miner]) -> bool:
        """Coordinator proposes that `group` (typically 2 currently-solo/PPS
        miners) found a new PPLNS pool together. Accepted iff every member
        strictly gains relative to their current payoff."""
        H = sum(m.h for m in group)
        fee = self.fee_function(H) if len(group) >= 2 else 0.0
        gains = {}
        for m in group:
            rate = self.env.Lam * H / self.env.H
            jump = self.env.b * (1 - fee) * m.h / H
            gains[m.idx] = self.engine.score(m, rate, jump) - self.current_payoff(m)
        if all(g > 0 for g in gains.values()):
            name = f"pplns_{self._next_pool_id}"
            self._next_pool_id += 1
            self.pplns_pools[name] = PPLNS_Pool([], self.fee_function, name)
            for m in group:
                self.move(m, ("pplns", name))
            return True
        return False
 
    # -- dynamics --
 
    def run_formation(self, max_rounds: int = 1000, rng: Optional[random.Random] = None) -> None:
        """Phase 1: repeatedly propose pairwise foundings among currently
        unpooled miners until none improve. Order-dependent by construction --
        pass an rng for reproducibility, or call with different seeds to check
        how much the terminal partition depends on proposal order."""
        rng = rng or random.Random(0)
        for _ in range(max_rounds):
            unpooled = [self.miners[i] for i, a in self.assignment.items() if a == ("pps", "solo")]
            if len(unpooled) < 2:
                break
            rng.shuffle(unpooled)
            improved = False
            for a, b in zip(unpooled[::2], unpooled[1::2]):
                if self.propose_founding([a, b]):
                    improved = True
            if not improved:
                break
            
    def set_initial_pplns_partition(self, groups: List[List[int]], name_prefix: str = "pplns") -> None:
        """Seed an initial PPLNS partition directly, bypassing propose_founding.
        Call this right after constructing the Game, before running any dynamics.
 
        `groups` is a list of miner-idx lists, e.g. [[1, 2], [3, 4, 5]] -- each
        idx must appear in at most one group. Singleton groups are harmless
        (they just get fee 0, same as leaving that miner out). Any miner not
        listed in any group stays on ('pps', 'solo').
 
        Typical uses:
          - skip formation and start best-response dynamics from a partition
            you already have in mind (e.g. an observed real-world partition)
          - seed a candidate equilibrium and call is_nash_stable() immediately,
            with no dynamics run at all, to check it directly
        """
        seen = set()
        for group_idx in groups:
            overlap = seen & set(group_idx)
            assert not overlap, f"miner(s) {overlap} appear in more than one group"
            seen |= set(group_idx)
            members = [self.miners[i] for i in group_idx]
            name = f"{name_prefix}_{self._next_pool_id}"
            self._next_pool_id += 1
            pool = PPLNS_Pool(members, self.fee_function, name)
            self.pplns_pools[name] = pool
            for m in members:
                self.assignment[m.idx] = ("pplns", name)
 
    def run_best_response(self, max_rounds: int = 1000, rng: Optional[random.Random] = None) -> bool:
        """Phase 2 (also reusable for Phase 1's aftermath): individual
        best-response to a fixed point. Returns True if it converged (Nash
        stable), False if max_rounds was hit -- treat that as 'possible cycle,
        inspect before trusting the result', not as a guarantee of instability."""
        rng = rng or random.Random(0)
        order = list(self.miners.values())
        for _ in range(max_rounds):
            rng.shuffle(order)
            changed = False
            for m in order:
                target, val = self.best_response(m)
                if target != self.assignment[m.idx] and val > self.current_payoff(m):
                    self.move(m, target)
                    changed = True
            if not changed:
                return True
        return False
 
    def is_nash_stable(self) -> bool:
        return all(self.best_response(m)[0] == self.assignment[m.idx] for m in self.miners.values())
 
    def introduce_pps(self, pool: PPS_Pool) -> None:
        """The exogenous entrant shock -- terms assumed attractive, not
        optimised here. Call run_best_response() again afterward."""
        self.pps_pools[pool.name] = pool
 
 