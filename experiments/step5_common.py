"""Step 5 populations (n = 10 by default): honest reactive mixes, optional planted coalition."""
from collusion.game import Coalition, Exploiter, LeaderReactive, MarginReactive, Population

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
    """planted: None, 'triangle' (agents 0-2), 'clique4' (agents 0-3), 'steer' (agents 0-3, a
    coalition that profits from reactive players) or 'exploit' (agents 0-3, honest Exploiters
    that react to the same signal independently, at the steer coalition's exploit rate
    cp / (1 + cp)). Honest types fill the remaining indices."""
    k = {None: 0, "triangle": 3, "clique4": 4, "steer": 4, "exploit": 4}[planted]
    honest = honest_types(mix, list(range(k, n)))
    if planted == "exploit":
        return Population(n, [], {**{i: Exploiter(cp / (1 + cp)) for i in range(k)}, **honest})
    kind = "steer" if planted == "steer" else "clique"
    coal = [Coalition(kind, list(range(k)), cp)] if k else []
    return Population(n, coal, honest)


def group(pop):
    """Agents whose payoff we track: the coalition, or the Exploiters."""
    if pop.coalitions:
        return sorted(pop.coalitions[0].members)
    return sorted(i for i, ag in pop.honest.items() if isinstance(ag, Exploiter))
