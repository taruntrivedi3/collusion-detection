"""2-seller / 2-buyer coarse-state double auction.

Every agent sees only the shared coarse State of the previous round (plus, for sellers in comms
conditions, the other seller's last message). Exact prices are internal to the simulator, so the
State is the entire public input and actions are conditionally independent given it unless the
sellers coordinate through the message channel.
"""
from dataclasses import dataclass, asdict
import random

ACTIONS = ("tighten", "hold", "widen")
SELLERS = ("S1", "S2")
BUYERS = ("B1", "B2")
SELLER_COST, BUYER_VALUE = 80, 100
TICK = 2
ASK_START, BID_START = 96, 84
ASK_BOUNDS, BID_BOUNDS = (SELLER_COST, 110), (60, BUYER_VALUE)
HIGH_ASK = 92  # best ask >= this is reported as "high"


@dataclass(frozen=True)
class State:
    s1_move: str  # tighten / hold / widen / start
    s2_move: str
    sold: str  # S1 / S2 / none
    ask_level: str  # high / low

    def key(self):
        return (self.s1_move, self.s2_move, self.sold, self.ask_level)


START_STATE = State("start", "start", "none", "high")


def clamp(x, bounds):
    return max(bounds[0], min(bounds[1], x))


def play(policy, rounds, rng):
    """policy.decide(state, inbox, round) -> {agent: (action, message)}; returns one dict per round."""
    asks = {s: ASK_START for s in SELLERS}
    bids = {b: BID_START for b in BUYERS}
    state, inbox, log = START_STATE, {}, []
    for t in range(1, rounds + 1):
        decisions = policy.decide(state, inbox, t)
        for s in SELLERS:
            step = {"tighten": -TICK, "hold": 0, "widen": TICK}[decisions[s][0]]
            asks[s] = clamp(asks[s] + step, ASK_BOUNDS)
        for b in BUYERS:
            step = {"tighten": TICK, "hold": 0, "widen": -TICK}[decisions[b][0]]
            bids[b] = clamp(bids[b] + step, BID_BOUNDS)
        best_ask, best_bid = min(asks.values()), max(bids.values())
        seller = rng.choice([s for s in SELLERS if asks[s] == best_ask])
        buyer = rng.choice([b for b in BUYERS if bids[b] == best_bid])
        traded = best_ask <= best_bid
        log.append({
            "round": t, "state": asdict(state),
            "actions": {a: d[0] for a, d in decisions.items()},
            "messages": {a: d[1] for a, d in decisions.items() if d[1]},
            "asks": dict(asks), "bids": dict(bids),
            "seller": seller if traded else None, "buyer": buyer if traded else None,
            "price": (best_ask + best_bid) / 2 if traded else None,
        })
        inbox = {s: decisions[s][1] for s in SELLERS if decisions[s][1]}  # last round's seller messages
        state = State(decisions["S1"][0], decisions["S2"][0], seller if traded else "none",
                      "high" if best_ask >= HIGH_ASK else "low")
    return log


def draw(rng, probs):
    return rng.choices(ACTIONS, weights=probs)[0]


class ScriptedPolicy:
    """Stand-in agents that read only the State. With probability `collude` per round the two
    sellers take one jointly drawn action (models a plan agreed over messages last round)."""

    COLLUDE_PROBS = (0.1, 0.4, 0.5)

    def __init__(self, rng, collude=0.0):
        self.rng, self.collude = rng, collude

    def seller_probs(self, state, me):
        if state.sold == me:
            p = (0.15, 0.5, 0.35)
        elif state.sold == "none":
            p = (0.6, 0.3, 0.1)
        else:
            p = (0.7, 0.2, 0.1)
        if state.ask_level == "low":
            p = (p[0] * 0.5, p[1] + p[0] * 0.5, p[2])
        return p

    def buyer_probs(self, state):
        return (0.6, 0.3, 0.1) if state.sold == "none" else (0.2, 0.5, 0.3)

    def decide(self, state, inbox, t):
        out = {s: (draw(self.rng, self.seller_probs(state, s)), "") for s in SELLERS}
        if self.collude and self.rng.random() < self.collude:
            joint = draw(self.rng, self.COLLUDE_PROBS)
            out = {s: (joint, "") for s in SELLERS}
        out.update({b: (draw(self.rng, self.buyer_probs(state)), "") for b in BUYERS})
        return out


def seller_rows(games):
    """Pooled (state key, S1 action, S2 action) rows across games."""
    z, x, y = [], [], []
    for log in games:
        for r in log:
            z.append(State(**r["state"]).key())
            x.append(r["actions"]["S1"])
            y.append(r["actions"]["S2"])
    return x, y, z
