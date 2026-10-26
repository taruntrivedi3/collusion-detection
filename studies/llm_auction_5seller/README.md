# LLM double auction: 5 sellers, 5 buyers, 100 games per condition

5 gpt-4.1-mini sellers and 5 scripted buyers, 50 hours per game, built on the market of the
[2-seller pilots](../llm_auction_2seller/README.md). Detection uses GroupScan, the same detector as the RPS
experiments (CMI edges, within-state shuffle, scan over seller groups), with Bonjour et al.'s pairwise rule
as the baseline.

**Result.** With a broadcast message channel plus CEO profit-margin pressure, sellers earn 8.20 per hour combined
[95% CI 7.83, 8.59] versus 5.24 [4.93, 5.54] with pressure alone and 5.12 [4.82, 5.42] with neither
(bootstrap over games). Pooled over 100 games, GroupScan reports exactly {S1, S2, S3, S4, S5} (score 14.2 vs
threshold 2.8, p = 0.001) and nothing for the controls (p = 0.29, 0.15). Bonjour's rule reports nothing in any
condition: with five mutually linked sellers, each pair's net influence cancels.

| Condition | Sellers' profit / hour (combined) | GroupScan: 5 / 10 / 20 games | Bonjour: 5 / 10 / 20 games |
| --- | --- | --- | --- |
| `comms_off` | 5.12 | 1/20 · 0/10 · 1/5 | 0/20 · 0/10 · 0/5 |
| `pressure_off` | 5.24 | 3/20 · 1/10 · 1/5 | 0/20 · 0/10 · 0/5 |
| `comms_pressure` | 8.20 | 8/20 · 7/10 · 5/5 | 2/20 · 0/10 · 0/5 |

Batches in which each detector reports a group, out of disjoint batches of 5, 10 and 20 games (1,000 shuffles
per set, 5% level). In the 20-game batches GroupScan names all five sellers 3 times and a 3- or 4-seller subset twice.

## Setup

- Same market as the 2-seller study: sellers cost $80, buyers value $100, each hour every agent lowers, keeps or
  raises its quote by $2, at most one trade per hour at the midpoint. Asks start at $96, bids at $84.
- With 5 sellers a per-seller move list would split the data over 3^5 states, so the shared state reports **how
  many** sellers lowered / kept / raised last hour, who sold, and whether the lowest ask is high (≥ $92) or low.
  Each seller's prompt is built only from that state (no memory, no hour number); a seller is told when it was
  the one that sold.
- In `comms_pressure` every seller sees all five sellers' messages from last hour. Messages are not in the state.
- Conditions: `comms_off`, `pressure_off` (CEO pressure, no channel), `comms_pressure`. Seeds 100–199 each.
- gpt-4.1-mini, temperature 0.7, scripted buyers, 75,000 calls (about $14.38).

## Test

Null: given the state, every seller's action is independent of every other seller's. Each seller's action
column is shuffled independently among hours with the same state, which draws exactly from that null.
`groupscan_n.py` runs GroupScan (`collusion.detect.detect_subgraph`) and Bonjour's rule
(`collusion.detect.detect_bonjour`) on the pooled hours of each batch.

Scripted dry run (`groupscan_n.py --dry-run 300`, 20 games × 50 hours): GroupScan false alarms 3.7% (Bonjour 0%);
with S1 and S2 coordinating in 10% of hours, GroupScan flags 65% of batches (50% exactly {S1, S2}), Bonjour 27%.

**Limit.** Coordination that follows only the public state (follow-the-leader, or the explicit-cartel prompt
of the 2-seller pilots) is invisible to this test.

## Files

| File | Purpose |
| --- | --- |
| `market_n.py` | N-seller / M-buyer market, count-based state, scripted agents |
| `groupscan_n.py` | GroupScan and Bonjour on a run directory, and the scripted dry run |
| `stats_n.py` | Joint within-state shuffle test: global and max-pair statistics |
| `llm_pilot_n.py` | Runs LLM games and writes `games.jsonl` / `calls.jsonl` |
| `analyze_n.py` | Outcomes per condition (profit, trades, actions) |
| `dry_run_n.py` | Scripted false-alarm and power check |
| `runs/n5_rates_v1/games.jsonl` | All 300 games: per-hour state, actions, messages, quotes, trades |
| `runs/n5_rates_v1/settings_100.json` | Arguments used for the run |
| `runs/n5_rates_v1/groupscan.txt` | Output of `groupscan_n.py runs/n5_rates_v1 5,10,20`: detection numbers above |
| `runs/n5_rates_v1/dry_run_groupscan.txt` | Output of `groupscan_n.py --dry-run 300` |
| `runs/n5_rates_v1/analysis.txt` | Output of `analyze_n.py`: profits above |
| `runs/n5_rates_v1/dry_run.txt` | Output of `dry_run_n.py 300`: scripted false alarms and power |

`market.py`, `stats.py` and `llm_pilot.py` are imported from `../llm_auction_2seller`. Per-call transcripts
(`calls.jsonl`) are not included; every action and message is in `games.jsonl`.

## Reproduce

From this directory (GroupScan needs the repository's `requirements.txt`; the rest is standard library):

```bash
python groupscan_n.py runs/n5_rates_v1 5,10,20   # detection numbers above (under a minute on 32 cores)
python groupscan_n.py --dry-run 300              # scripted false alarms and power
python analyze_n.py runs/n5_rates_v1 5,10,20     # profits per condition
```

To rerun the games with new LLM calls (results will differ by sampling):

```bash
python3 llm_pilot_n.py --sellers 5 --buyers 5 --seeds 100 --first-seed 100 --rounds 50 --out runs/rerun
```

This calls the OpenAI chat-completions API with the key in `OPENAI_API_KEY`. For another compatible endpoint,
pass `--base-url` (or set `OPENAI_BASE_URL`) and `--api-key-env`.
