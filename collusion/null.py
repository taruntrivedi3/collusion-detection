"""Within-state permutation null.

Each agent's action column is permuted independently across the rounds of each state group.
This keeps p(a_i | s) and destroys any dependence between agents beyond the state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimate import gamma_batch


def permute_within_states(actions, states, B, rng, chunk=None):
    """Yield arrays [b, R, n] of permuted datasets, B in total (in chunks)."""
    rounds, n = actions.shape
    base = np.argsort(states, kind="stable")        # rounds sorted by state
    chunk = chunk or B
    done = 0
    while done < B:
        b = min(chunk, B - done)
        # sort key: state, then a fresh uniform -> random order within each state block
        keys = states[None, :, None] + rng.random((b, rounds, n))
        src = np.argsort(keys, axis=1)              # [b, R, n] source rows, state-sorted
        out = np.empty((b, rounds, n), dtype=actions.dtype)
        cols = np.arange(n)
        out[:, base, :] = actions[src, cols]
        yield out
        done += b


def null_gammas(actions, states, B, rng, chunk=50):
    """gamma for B permuted datasets: [B, n, n]."""
    return np.concatenate([gamma_batch(x, states) for x in permute_within_states(actions, states, B, rng, chunk)])


@dataclass
class NullContext:
    """Everything the detectors need for one dataset.

    g_real : [n, n]       observed gamma
    g_A    : [B, n, n]    null set A (edge / subset null moments)
    g_B    : [B, n, n]    null set B (calibration of max statistics)
    """
    g_real: np.ndarray
    g_A: np.ndarray
    g_B: np.ndarray
    seed: int | None = None

    @property
    def n(self):
        return self.g_real.shape[0]

    def restrict(self, keep):
        """Context for a subset of agents. Column permutations are independent per agent, so
        sub-blocks of the null gammas are valid nulls for the remaining agents."""
        k = np.asarray(sorted(keep))
        ix = np.ix_(k, k)
        return NullContext(self.g_real[ix], self.g_A[:, k][:, :, k], self.g_B[:, k][:, :, k],
                           self.seed), k


def build_context(actions, states, B=200, seed=0):
    rng = np.random.default_rng(seed)
    g_real = gamma_batch(actions, states)
    g_A = null_gammas(actions, states, B, rng)
    g_B = null_gammas(actions, states, B, rng)
    return NullContext(g_real, g_A, g_B, seed)
