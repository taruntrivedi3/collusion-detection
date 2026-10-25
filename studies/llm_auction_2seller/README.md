# LLM double auction: 2 sellers, 2 buyers, 200 games per condition

Within-state shuffle test for seller collusion in a double auction where every agent sees only a shared,
coarse summary of the last hour. Report: [report.pdf](report.pdf).

**Result.** With a message channel plus CEO profit-margin pressure, gpt-4.1-mini sellers earn 7.07 per hour combined
[95% CI 6.76, 7.38] versus 4.76 [4.52, 5.00] with pressure alone and 4.29 [4.06, 4.51] with neither
(bootstrap over games). Pooled over 200 games per condition, the test flags
comms + pressure (CMI z = 5.0, same-move z = 7.9) and stays silent on both controls (|z| ≤ 1.5). Per 20-game
batch, the same-move statistic detects it in 80% of batches and CMI in 10%.

**Limits.** With two sellers this is a pairwise test; the group scan is not used here. Coordination that is a
function of the public state (for example "copy S1's last move") is invisible by construction; an
explicit-cartel prompt produced exactly that and was missed (pilots below). The market is designed for
detectability: the prompt contains nothing but the recorded state. The same-move statistic was chosen after
this run showed CMI was underpowered, so its 80% is post hoc; the 5-seller study
([llm_auction_5seller](../llm_auction_5seller/README.md)) tests it as fixed in advance.

**Pilots.** Four 20-game pilots ran before this study, including conditions later dropped (`comms_on`,
`comms_instructed`, `comms_cartel`). The shuffle test missed the cartel and instructed sellers in every pilot
and missed comms + pressure in one of two. All results: `python3 pilots.py`, and Appendix D of the
[report](../../report/report.md).

## Setup

- Two LLM sellers (cost $80) and two scripted buyers (value $100); each hour every agent lowers, keeps or raises
  its quote by $2; at most one trade per hour, at the midpoint; 30 hours per game.
- Prompts are built only from the shared state: each seller's last move, who sold, and whether the lowest ask
  is high or low. No memory, no hour number. In comms conditions sellers also see both of last hour's messages.
- Conditions: `comms_off` (no channel, no pressure), `pressure_off` (CEO pressure, no channel; the name means
  "pressure, comms off"), `comms_pressure` (both). Seeds 0–199 each.
- gpt-4.1-mini, temperature 0.7, 36,000 calls (about $6.35).

## Files

| File | Purpose |
| --- | --- |
| `market.py` | Market simulator, coarse state, scripted agents |
| `stats.py` | CMI, same-move statistic, within-state shuffle test |
| `llm_pilot.py` | Runs LLM games and writes `games.jsonl` / `calls.jsonl` |
| `rates.py` | Rejection rates over disjoint 5 / 10 / 20-game batches |
| `build_report.py` | Recomputes every number and builds `report.pdf` and `detection.png` |
| `runs/rates_v1/games.jsonl` | All 600 games: per-hour state, actions, messages, quotes, trades |
| `runs/rates_v1/settings.json` | Arguments used for the run |
| `pilots.py` | Summary of the four pilots: profit, shuffle tests (same-move, CMI), direction test |
| `runs/pilot_v1` … `pilot_v4/` | Pilot games (`games.jsonl`) and arguments. v1 used LLM buyers; v1–v3 used earlier prompt wording that was not saved |

Per-call transcripts (`calls.jsonl`) are not included; every action and message is in `games.jsonl`.

## Reproduce

From this directory (standard library only, except `build_report.py`, which needs `reportlab` and `matplotlib`):

```bash
python3 rates.py runs/rates_v1 cmi          # batch rejection rates, CMI statistic
python3 rates.py runs/rates_v1 same_moves   # batch rejection rates, same-move statistic
python3 build_report.py                     # report.pdf + detection.png from the same data
python3 pilots.py                           # pilot results (report Appendix D)
```

To rerun the games with new LLM calls (results will differ by sampling):

```bash
python3 llm_pilot.py --seeds 200 --rounds 30 --conditions comms_off,pressure_off,comms_pressure --out runs/rerun
```

This calls the OpenAI chat-completions API with the key in `OPENAI_API_KEY`. For another compatible endpoint,
pass `--base-url` (or set `OPENAI_BASE_URL`) and `--api-key-env`.
