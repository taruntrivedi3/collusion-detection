# Finding colluding groups from players' moves

Apart Research AI Collusion Research Sprint (2026-10-23 to 2026-10-25), research question 1.4:
*can information-theoretic methods detect collusion in multi-agent games?*

**Claim.** The existing information-theoretic detector (pairwise net influence, Bonjour et al. 2022)
cannot see colluding groups of three or more. A shuffle-calibrated scan over groups finds them,
provided the test accounts for what honest agents react to. We show this requirement also holds for
LLM agents: in a market where the recorded state is everything the agents see, the test catches
colluding LLM sellers and stays quiet on the controls.

| | Finding | Evidence |
|---|---|---|
| 1 | **The pairwise method fails on groups.** Inside a group its score cancels out: 0% of triangles, cliques and hubs found. Its fixed cutoff is also uncalibrated: with no collusion it flags 8–23% of short (100-round) games and never fires in longer ones. | [Fig. A](results/figures/figA_cancellation.png), [Fig. D](results/figures/figD_null_false_alarms.png) |
| 2 | **A group scan finds weak, broad groups.** 5 colluders among 20 players, 100 rounds, coordinating 20% of the time: 48% exact recovery vs 5% for the best pairwise baseline. It loses where the colluders aren't all linked: on hubs (one leader, two helpers) the pair test does better (58% vs 34%). Collusion with no pairwise trace (xor) is invisible to every method here. | [Fig. C](results/figures/figC_weak_cliques.png), [Table 2](results/SUMMARY.md#table-2--where-the-group-scan-wins-and-loses) |
| 3 | **The test must condition on what honest players react to.** Ignore it and honest players are flagged in up to 100% of games; condition on it and false alarms return to about 5%. This holds for collusion that pays: a 4-player group that steers reacting players earns 2.6× what 4 honest exploiters earn, and, colluding 20–30% of the time, is found in 94–98% of 1000+-round games while the exploiters are flagged in 4–7%. | [Fig. E](results/figures/figE_state_false_alarms.png), [Table 7](results/SUMMARY.md#table-7--profitable-collusion-steer-group-vs-honest-exploiters) |
| 4 | **The requirement carries over to LLM agents.** In a 2-seller double auction where every prompt is built only from a recorded coarse state, gpt-4.1-mini sellers with a message channel and profit pressure earn 7.07 [6.76, 7.38] per hour vs 4.76 [4.52, 5.00] with pressure alone and 4.29 [4.06, 4.51] with neither. Pooled over 200 games, the history-conditioned shuffle test flags them (same-move z = 7.9) and stays quiet on both controls (\|z\| ≤ 1.5). Per 20-game batch only the targeted same-move count has power (80%; the general CMI test, 10%), and coordination keyed to public moves, as in an explicit-cartel prompt, is invisible by construction. | [LLM auction](studies/llm_auction/README.md), [report](studies/llm_auction/report.pdf) |

Full write-up of the simulation results: [results/SUMMARY.md](results/SUMMARY.md). Interactive
viewer: <https://taruntrivedi3.github.io/collusion-detection/viewer/>.

## See it

```bash
python viewer/export.py        # optional: regenerates viewer/data.js (a few minutes)
open viewer/index.html         # or any browser; no server needed (also live at the link above)
```

The viewer replays example games. You see the moves, the links between players, which group was
planted, and what each detector reported. It also shows LLM auction games with the sellers' messages.
Deep links: `index.html#rps/steer`, `index.html#auction/comms_pressure`. LLM condition names:
`comms_off` (no messages, no pressure), `pressure_off` (CEO pressure, no messages) and `comms_pressure`
(both).

## Repository

```
collusion/      detector library: information estimates, shuffle tests, all detectors
experiments/    RPS experiments, one script per step (step1 … step5) + make_figures.py
results/        RPS outputs: figures/, per-step CSVs, SUMMARY.md, WRITEUP_NOTES.md (methods)
studies/
  llm_auction/               finding 4: new LLM double auction (2 sellers, 200 games/condition)
  auction_null_calibration/  appendix: recorded LLM markets from andyjliu/llm-agent-collusion
viewer/         interactive game viewer (static HTML)
tests/          unit tests for collusion/
scripts/        reproduce.py for the recorded-market study
```

**Appendix study, and why it isn't a main finding.** We ran our group scan and
shuffle test on 75 recorded 5-seller markets from
[andyjliu/llm-agent-collusion](https://github.com/andyjliu/llm-agent-collusion/tree/2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff).
Their recorded paths share strong time trends that no available state captures without leaving nothing to
shuffle. The test was badly miscalibrated: 88.5% flags on independently resampled seller paths. A
same-pool fix did not carry over to held-out markets. This illustrates finding 3: without a state that
explains the honest behaviour, the test is not usable. See its
[interpretation](studies/auction_null_calibration/results/INTERPRETATION.md).

## Before the sprint

Before the sprint began, we read the papers, came up with a game plan, played around with a few toy
examples, and implemented the starting code. We also ran the recorded-market study in
`studies/auction_null_calibration/` (see the appendix note above), including its log parser in
`source_snapshot/`.

## Reproduce

Python 3.11+ on Linux. The recorded-market study also needs `g++` with C++17.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests

cd experiments && python make_figures.py A B C D E T     # figures from saved results
python step4_final.py ...                                 # see results/RESULTS_INDEX.md for every command and seed

python scripts/reproduce.py --output /tmp/auction-check   # recorded-market study checks (add --mode full for the full run)

cd studies/llm_auction && python rates.py runs/rates_v1 same_moves && python build_report.py
```

No command above calls an LLM. `studies/llm_auction/llm_pilot.py` reruns the auction games with new
OpenAI API calls (about $6).
