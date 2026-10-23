# Graph-based collusion detection in multi-agent RPS

Apart Research AI Collusion Sprint (2026-10-23 to 2026-10-25 America/Chicago), RQ 1.4.

    collusion/game.py      agents, coalitions, payoffs, simulate
    collusion/estimate.py  state encoders, gamma = I(a_i; a_j | s) in bits, net influence Gamma
    collusion/null.py      within-state permutation null, NullContext (Set A / Set B)
    collusion/detect.py    bonjour, edge_bh, node_score, subgraph; detect(actions, states, method)
    experiments/           one script per step; outputs land in results/<step>/ as CSV + PNG
    tests/                 unit tests

Setup / run:

    uv venv .venv && uv pip install --python .venv/bin/python numpy scipy pandas matplotlib pytest numba
    .venv/bin/python -m pytest -q tests
    cd experiments && ../.venv/bin/python step1_reproduce.py
