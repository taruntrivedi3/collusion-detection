# Finding colluding groups from players' moves

Apart Research AI Collusion Research Sprint (2026-10-23 to 2026-10-25), research question 1.4:
*can information-theoretic methods detect collusion in multi-agent games?*

**Claim.** The existing information-theoretic detector (pairwise net influence, Bonjour et al. 2022)
cannot see colluding groups of three or more. A shuffle-calibrated scan over groups finds them, also
among LLM agents, provided the test accounts for the history the agents react to.

| | Finding | Evidence |
|---|---|---|
| 1 | **The pairwise method fails on groups.** Inside a group its score cancels out: 0% of triangles, cliques and hubs found. Its fixed cutoff also gives 0–43% false alarms with no collusion. | [Fig. A](results/figures/figA_cancellation.png), [Fig. D](results/figures/figD_null_false_alarms.png) |
| 2 | **A group scan finds weak, broad groups.** 5 colluders among 20 players, 100 rounds, coordinating 20% of the time: 44% exact recovery vs 6% for the best pairwise baseline. Triple scans catch "xor" collusion that every pairwise method misses. | [Fig. C](results/figures/figC_weak_cliques.png), [Table 8](results/SUMMARY.md#table-8--triple-scans-vs-the-group-scan-step-6-200-shuffles) |
| 3 | **The test must condition on what honest players react to.** Ignore it and honest players are flagged in up to 100% of games; condition on it and false alarms return to about 5%. | [Fig. E](results/figures/figE_state_false_alarms.png), [robustness study](studies/rps_null_calibration/results/INTERPRETATION.md) |
| 4 | **It works on LLM agents.** gpt-4.1-mini sellers with a message channel and profit pressure earn 7.07/hour vs 4.29–4.76 in the controls. The test flags them (same-move z = 7.9) and stays quiet on both controls (\|z\| ≤ 1.5). | [LLM auction](studies/llm_auction/README.md), [report](studies/llm_auction/report.pdf) |

Full write-up of the simulation results: [results/SUMMARY.md](results/SUMMARY.md).

## See it

```bash
python viewer/export.py        # optional: regenerates viewer/data.js (~2 min)
open viewer/index.html         # or any browser; no server needed
```

The viewer replays example games. You see the moves, the links between players, which group was
planted, and what each detector reported. It also shows LLM auction games with the sellers' messages.
Deep links: `index.html#rps/xor`, `index.html#auction/comms_pressure`.

## Repository

```
collusion/      detector library: information estimates, shuffle tests, all detectors
experiments/    RPS experiments, one script per step (step1 … step6) + make_figures.py
results/        RPS outputs: figures/, per-step CSVs, SUMMARY.md, WRITEUP_NOTES.md (methods)
studies/
  llm_auction/               finding 4: new LLM double auction (2 sellers, 200 games/condition)
  rps_null_calibration/      finding 3 robustness: honest adaptive players, 25k+ games
  auction_null_calibration/  appendix: recorded LLM markets from andyjliu/llm-agent-collusion
viewer/         interactive game viewer (static HTML)
tests/          unit tests for collusion/
scripts/        reproduce.py for the two calibration studies
```

**Appendix study, and why it isn't a main finding.** We reused 75 recorded 5-seller markets from
[andyjliu/llm-agent-collusion](https://github.com/andyjliu/llm-agent-collusion/tree/2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff).
Their recorded paths share strong time trends that no available state captures without leaving nothing to
shuffle. The test was badly miscalibrated: 88.5% flags on independently resampled seller paths. A
same-pool fix did not carry over to held-out markets. This illustrates finding 3: without a state that
explains the honest behaviour, the test is not usable. See its
[interpretation](studies/auction_null_calibration/results/INTERPRETATION.md).

## Reproduce

Python 3.11+ on Linux. The calibration studies also need `g++` with C++17.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests

cd experiments && python make_figures.py A B C D E T     # figures from saved results
python step4_final.py ...                                 # see results/RESULTS_INDEX.md for every command and seed

python scripts/reproduce.py rps --output /tmp/rps-check   # calibration study checks (add --mode full for the full run)
python scripts/reproduce.py auction --output /tmp/auction-check

cd studies/llm_auction && python rates.py runs/rates_v1 same_moves && python build_report.py
```

No command above calls an LLM. `studies/llm_auction/llm_pilot.py` reruns the auction games with new
OpenAI API calls (about $6).

`results/MANIFEST.tsv` records the hash of every file at the pre-sprint freeze (2026-10-19 UTC).
