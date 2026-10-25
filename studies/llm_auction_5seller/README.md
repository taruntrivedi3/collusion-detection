# LLM double auction: 5 sellers, 5 buyers, 100 games per condition

Extends [llm_auction_2seller](../llm_auction_2seller/README.md) (2 sellers / 2 buyers) to 5 gpt-4.1-mini sellers, 5 scripted buyers
and 50 hours per game, with a joint within-state shuffle test over all seller pairs.

**Result.** With a broadcast message channel plus CEO profit-margin pressure, sellers earn 8.20 per hour combined
[95% CI 7.83, 8.59] versus 5.24 [4.93, 5.54] with pressure alone and 5.12 [4.82, 5.42] with neither
(bootstrap over games). The global test flags comms + pressure in 95% of 5-game batches
and every 10- and 20-game batch, and flags neither control in any batch.

| Condition | Sellers' profit / hour (combined) | 5 games (20 batches) | 10 games (10 batches) | 20 games (5 batches) | Pooled, 100 games |
| --- | --- | --- | --- | --- | --- |
| `comms_off` | 5.12 | 0.00 (0.05) | 0.00 (0.00) | 0.00 (0.00) | p = 0.17 |
| `pressure_off` | 5.24 | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.00) | p = 0.71 |
| `comms_pressure` | 8.20 | 0.95 (0.80) | 1.00 (1.00) | 1.00 (1.00) | p = 0.001 |

Rejection rate at α = 0.05 for the global test (max-pair test in parentheses). Pooled p is the global test;
0.001 is the floor at 1,000 shuffles. Pooled comms + pressure: 1,991 more matching seller pairs than the
shuffle expects (out of 50,000 pair-hours), every one of the 10 pairs above z = 5 (top pair z ≈ 11.5).

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
column is shuffled independently among rounds with the same state, which draws exactly from that null.

- **Global test:** total same-move count summed over all 10 seller pairs.
- **Max-pair test:** largest standardized pair count; controls the family-wise error rate over pairs and
  names the pair. In the scripted dry run (only S1 and S2 collude) it names S1–S2 in 99.7% of batches.

Scripted dry run (`dry_run_n.py`, 300 replications of 20 games × 50 hours): false alarms 4.3% (global) and
5.0% (max-pair); planted S1–S2 collusion in 10% of rounds is found by the max-pair test in every replication.

**Limit.** Here every pair is in sync (the whole group agrees to "hold, don't undercut"), so the max-pair test
does not isolate a sub-cartel. As in the 2-seller study, coordination that follows only the public state
(follow-the-leader) is invisible to this test.

## Files

| File | Purpose |
| --- | --- |
| `market_n.py` | N-seller / M-buyer market, count-based state, scripted agents |
| `stats_n.py` | Joint within-state shuffle test: global and max-pair statistics |
| `llm_pilot_n.py` | Runs LLM games and writes `games.jsonl` / `calls.jsonl` |
| `analyze_n.py` | Outcomes, pooled test and rejection rates over disjoint batches |
| `dry_run_n.py` | Scripted false-alarm and power check |
| `runs/n5_rates_v1/games.jsonl` | All 300 games: per-hour state, actions, messages, quotes, trades |
| `runs/n5_rates_v1/settings_100.json` | Arguments used for the run |
| `runs/n5_rates_v1/analysis.txt` | Output of `analyze_n.py`: every number in the table above |
| `runs/n5_rates_v1/dry_run.txt` | Output of `dry_run_n.py 300`: scripted false alarms and power |

`market.py`, `stats.py` and `llm_pilot.py` are imported from `../llm_auction_2seller`. Per-call transcripts
(`calls.jsonl`) are not included; every action and message is in `games.jsonl`.

## Reproduce

From this directory, standard library only:

```bash
python3 analyze_n.py runs/n5_rates_v1 5,10,20   # table above (about 1 minute on 32 cores)
python3 dry_run_n.py 300                        # scripted false-alarm and power check
```

To rerun the games with new LLM calls (results will differ by sampling):

```bash
python3 llm_pilot_n.py --sellers 5 --buyers 5 --seeds 100 --first-seed 100 --rounds 50 --out runs/rerun
```

This calls the OpenAI chat-completions API with the key in `OPENAI_API_KEY`. For another compatible endpoint,
pass `--base-url` (or set `OPENAI_BASE_URL`) and `--api-key-env`.
