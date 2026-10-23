"""Step 5 populations (n = 10 by default): honest reactive mixes, optional planted coalition."""
from collusion.game import Coalition, LeaderReactive, MarginReactive, Population

MIXES = ("uniform", "leader", "margin", "mixed")
STATE_DEFS = ("none", "top", "order", "full")


def honest_types(mix, idx):
    """Assign honest types to agent indices `idx` (list). Roughly half reactive."""
    k = len(idx)
    if mix == "uniform":
        return {}
    if mix == "leader":
        return {i: LeaderReactive(0.5) for i in idx[: k // 2]}
    if mix == "margin":
        return {i: MarginReactive(0.5, 3) for i in idx[: k // 2]}
    if mix == "mixed":
        h = k // 3
        d = {i: LeaderReactive(0.5) for i in idx[:h]}
        d.update({i: MarginReactive(0.5, 3) for i in idx[h: 2 * h]})
        return d
    raise ValueError(mix)


def population(mix, n=10, planted=None, cp=0.3):
    """planted: None, 'triangle' (agents 0-2) or 'clique4' (agents 0-3). Colluders are
    history-free; honest types fill the remaining indices."""
    k = {None: 0, "triangle": 3, "clique4": 4}[planted]
    coal = [Coalition("clique", list(range(k)), cp)] if k else []
    return Population(n, coal, honest_types(mix, list(range(k, n))))
