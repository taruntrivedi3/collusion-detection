"""n-player Rock Paper Scissors: agents, payoffs, simulation.

Actions are encoded 0 = Rock, 1 = Paper, 2 = Scissors.
Move a beats move b  <=>  (a - b) % 3 == 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R, P, S = 0, 1, 2
MOVE_NAMES = "RPS"


def beats(x):
    """Move that beats x."""
    return (np.asarray(x) + 1) % 3


def loses_to(x):
    """Move that loses to x (what a helper plays to feed x a point)."""
    return (np.asarray(x) + 2) % 3


# ---------------------------------------------------------------- payoffs

def payoff_round_robin(actions):
    """payoff_i = #players i beats - #players who beat i. actions: [R, n] or [n]."""
    a = np.atleast_2d(actions)
    counts = action_counts(a)                       # [R, 3]
    win = np.take_along_axis(counts, (a + 2) % 3, axis=1)   # players my move beats
    lose = np.take_along_axis(counts, (a + 1) % 3, axis=1)  # players who beat me
    return (win - lose).reshape(np.shape(actions))


def payoff_paper3(actions):
    """Bonjour et al. 3-player payoff: all same -> 0; all three moves present -> 1 each;
    exactly two moves present -> winners get 1, losers 0."""
    a = np.atleast_2d(actions)
    assert a.shape[1] == 3
    counts = action_counts(a)
    present = counts > 0
    n_present = present.sum(1)
    out = np.zeros(a.shape, dtype=int)
    out[n_present == 3] = 1
    two = n_present == 2
    # winner move w: present and the move it beats is present
    beaten_present = np.take_along_axis(present, (a + 2) % 3, axis=1)
    out[two] = beaten_present[two].astype(int)
    return out.reshape(np.shape(actions))


def action_counts(actions):
    """[R, n] actions -> [R, 3] counts of (R, P, S)."""
    a = np.atleast_2d(actions)
    return np.stack([(a == m).sum(1) for m in range(3)], axis=1)


# ---------------------------------------------------------------- honest agents

@dataclass
class Uniform:
    reactive = False


@dataclass
class LeaderReactive:
    """With prob p, play the move that beats last round's most common move (ties broken
    uniformly); else uniform."""
    p: float
    reactive = True


@dataclass
class MarginReactive:
    """Like LeaderReactive, but only when the leader leads the runner-up by >= m."""
    p: float
    m: int
    reactive = True


@dataclass
class Exploiter:
    """With prob p, play the move that beats what LeaderReactive agents play next (the move
    that loses to last round's most common move); else uniform. Honest, no coordination."""
    p: float
    reactive = True


def _react(agent, counts, rng):
    """Action of a reactive honest agent given last round's counts [3]."""
    if rng.random() >= agent.p:
        return rng.integers(3)
    top = counts.max()
    if isinstance(agent, Exploiter):
        return loses_to(rng.choice(np.flatnonzero(counts == top)))
    if isinstance(agent, MarginReactive):
        second = np.sort(counts)[-2]
        if top - second < agent.m:
            return rng.integers(3)
    leaders = np.flatnonzero(counts == top)
    return beats(rng.choice(leaders))


# ---------------------------------------------------------------- colluder structures

@dataclass
class Coalition:
    """A planted colluding structure.

    kind:
      pair      members = [P, H]            H plays loses_to(P) w.p. cp
      hub       members = [B, A, D, ...]    each helper independently plays loses_to(B) w.p. cp
      triangle  members = [c0, c1, ...]     shared event w.p. cp -> all play same uniform move
      clique    alias of triangle for k >= 4 members
      xor       members = [A, B, C]         C plays (A + B) % 3 w.p. cp
      steer     members = [c0, c1, ...]     w.p. cp all play the same uniform move M (a clear
                                            majority); next round all play loses_to(M), which
                                            beats the honest players who react to M. Profitable
                                            against MarginReactive players; uses only the
                                            coalition's own private memory, not public history.
    Members not driven by the collusion event play uniform.
    """
    kind: str
    members: list
    cp: float

    @property
    def true_set(self):
        return frozenset(self.members)


def _coalition_actions(c: Coalition, rounds, rng):
    """[rounds, len(members)] actions for a history-free coalition."""
    k = len(c.members)
    a = rng.integers(3, size=(rounds, k))
    if c.kind in ("pair", "hub"):
        for h in range(1, k):
            fire = rng.random(rounds) < c.cp
            a[fire, h] = loses_to(a[fire, 0])
    elif c.kind in ("triangle", "clique"):
        fire = rng.random(rounds) < c.cp
        shared = rng.integers(3, size=rounds)
        a[fire] = shared[fire, None]
    elif c.kind == "steer":
        t = 0
        while t < rounds:
            if rng.random() < c.cp:
                m = rng.integers(3)
                a[t] = m
                if t + 1 < rounds:
                    a[t + 1] = loses_to(m)
                t += 2
            else:
                t += 1
    elif c.kind == "xor":
        assert k == 3
        fire = rng.random(rounds) < c.cp
        a[fire, 2] = (a[fire, 0] + a[fire, 1]) % 3
    else:
        raise ValueError(c.kind)
    return a


# ---------------------------------------------------------------- population / simulate

@dataclass
class Population:
    """n agents. `honest` maps agent index -> honest agent object; every index not in a
    coalition and not in `honest` is Uniform."""
    n: int
    coalitions: list = field(default_factory=list)
    honest: dict = field(default_factory=dict)

    def __post_init__(self):
        used = [m for c in self.coalitions for m in c.members]
        assert len(used) == len(set(used)), "coalitions overlap"
        assert not set(used) & set(self.honest), "agent both honest and colluder"
        assert all(0 <= m < self.n for m in used + list(self.honest))

    @property
    def true_sets(self):
        return [c.true_set for c in self.coalitions]


def random_seats(pop: Population, rng: np.random.Generator) -> Population:
    """Same population with agents moved to uniformly random seats."""
    perm = rng.permutation(pop.n)
    return Population(pop.n, [Coalition(c.kind, [int(perm[m]) for m in c.members], c.cp) for c in pop.coalitions],
                      {int(perm[i]): ag for i, ag in pop.honest.items()})


def simulate(pop: Population, rounds: int, rng: np.random.Generator):
    """Return actions [rounds, n] (int8)."""
    a = rng.integers(3, size=(rounds, pop.n))
    for c in pop.coalitions:
        a[:, c.members] = _coalition_actions(c, rounds, rng)
    reactive = [i for i, ag in pop.honest.items() if ag.reactive]
    if reactive:
        # round 0 has no history -> uniform (already drawn). Colluders and uniform agents are
        # history-free, so only reactive columns need a sequential pass.
        for t in range(1, rounds):
            counts = np.bincount(a[t - 1], minlength=3)
            for i in reactive:
                a[t, i] = _react(pop.honest[i], counts, rng)
    return a.astype(np.int8)
