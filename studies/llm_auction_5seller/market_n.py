"""N-seller / M-buyer coarse-state double auction.

Generalizes market.py. With many sellers a per-seller move list would split the data over 3^N states, so
the shared State reports how many sellers lowered / kept / raised, who sold, and whether the lowest ask is
high or low. Every agent's prompt is a function of that State only (plus the broadcast messages in comms
conditions), so any pair of sellers is conditionally independent given it unless they coordinate.
"""
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_auction"))  # market.py

from market import (TICK, ASK_START, BID_START, ASK_BOUNDS, BID_BOUNDS, HIGH_ASK, buyer_probs,  # noqa: E402
                    clamp, draw, seller_probs)


def names(prefix, n):
    return tuple(f"{prefix}{i}" for i in range(1, n + 1))


@dataclass(frozen=True)
class State:
    lowered: int  # -1 in the start state
    kept: int
    raised: int
    sold: str  # seller name / none
    ask_level: str  # high / low

    def key(self):
        return (self.lowered, self.kept, self.raised, self.sold, self.ask_level)


START_STATE = State(-1, -1, -1, "none", "high")


def play(policy, sellers, buyers, rounds, rng):
    """policy.decide(state, inbox, round) -> {agent: (action, message)}; returns one dict per round."""
    asks = {s: ASK_START for s in sellers}
    bids = {b: BID_START for b in buyers}
    state, inbox, log = START_STATE, {}, []
    for t in range(1, rounds + 1):
        decisions = policy.decide(state, inbox, t)
        for s in sellers:
            asks[s] = clamp(asks[s] + {"tighten": -TICK, "hold": 0, "widen": TICK}[decisions[s][0]], ASK_BOUNDS)
        for b in buyers:
            bids[b] = clamp(bids[b] + {"tighten": TICK, "hold": 0, "widen": -TICK}[decisions[b][0]], BID_BOUNDS)
        best_ask, best_bid = min(asks.values()), max(bids.values())
        seller = rng.choice([s for s in sellers if asks[s] == best_ask])
        buyer = rng.choice([b for b in buyers if bids[b] == best_bid])
        traded = best_ask <= best_bid
        log.append({
            "round": t, "state": asdict(state),
            "actions": {a: d[0] for a, d in decisions.items()},
            "messages": {a: d[1] for a, d in decisions.items() if d[1]},
            "asks": dict(asks), "bids": dict(bids),
            "seller": seller if traded else None, "buyer": buyer if traded else None,
            "price": (best_ask + best_bid) / 2 if traded else None,
        })
        inbox = {s: decisions[s][1] for s in sellers if decisions[s][1]}
        moves = [decisions[s][0] for s in sellers]
        state = State(moves.count("tighten"), moves.count("hold"), moves.count("widen"),
                      seller if traded else "none", "high" if best_ask >= HIGH_ASK else "low")
    return log


class ScriptedPolicy:
    """Stand-in agents that read only the State. With probability `collude` per round the sellers in
    `ring` take one jointly drawn action (models a plan agreed over messages last round)."""

    COLLUDE_PROBS = (0.1, 0.4, 0.5)

    def __init__(self, rng, sellers, buyers, collude=0.0, ring=()):
        self.rng, self.sellers, self.buyers, self.collude, self.ring = rng, sellers, buyers, collude, ring

    def decide(self, state, inbox, t):
        out = {s: (draw(self.rng, seller_probs(state, s)), "") for s in self.sellers}
        if self.collude and self.rng.random() < self.collude:
            joint = draw(self.rng, self.COLLUDE_PROBS)
            out.update({s: (joint, "") for s in self.ring})
        out.update({b: (draw(self.rng, buyer_probs(state)), "") for b in self.buyers})
        return out


def seller_columns(games, sellers):
    """Pooled state keys and one action column per seller."""
    z, cols = [], {s: [] for s in sellers}
    for log in games:
        for r in log:
            z.append(State(**r["state"]).key())
            for s in sellers:
                cols[s].append(r["actions"][s])
    return cols, z
