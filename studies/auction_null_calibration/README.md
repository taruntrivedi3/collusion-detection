# Recorded LLM double auctions (appendix)

**Question.** Does the shuffle test stay calibrated on 75 recorded LLM markets (five sellers, five
buyers, 30 rounds) from [andyjliu/llm-agent-collusion](https://github.com/andyjliu/llm-agent-collusion/tree/2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff)?
**Answer.** No. The recorded paths have strong shared time trends: agents all quote high early
and lower later. A coarse time state leaves those trends in, and exact rounds leave nothing to shuffle. Independently resampled
seller paths were flagged in 88.5% of groups under within-state shuffling and 37.2% under circular
shifts. Calibrating on the same pool fixed the rate, but the fix did not carry over to held-out
markets (4.2–48.8% flags). This is why the main LLM experiment (`studies/llm_auction`) records a
coarse state that is the agents' entire public input. Details: [findings](results/INTERPRETATION.md),
[full report](results/REPORT.md).

The resampled-path benchmark draws from a small pool of recorded agents. It does not generate
consistent auction histories or certify the agents as honest, and there are only ten no-message base
markets. Flags on the real markets therefore do not establish collusion. No new LLM runs were made.

- [Protocol](PROTOCOL.md), [configuration](config.json), [statistical notes](STATISTICAL_NOTES.md).
- [Benchmark](results/benchmark_false_alarm.csv), [copied-triple results](results/copy_power.csv),
  [score-calibration transfer](results/source_score_transfer.csv),
  [actual-market flags](results/real_condition_summary.csv), [parser check](results/parse_audit.json).

From the repository root:

```bash
python scripts/reproduce.py auction --output /path/to/new/auction-check     # build + validation
python scripts/reproduce.py auction --mode full --workers 8 --output /path/to/new/auction-run
```

Prepared inputs (`results/cases.json`, `pool.npz`, `pool_provenance.json`, `state_audit.json`) are
included, so no download is needed. To fetch and parse the original logs yourself:

```bash
python studies/auction_null_calibration/source_snapshot/fetch_data.py --destination /path/to/new/auction-data
python studies/auction_null_calibration/source_snapshot/ingest.py --source /path/to/new/auction-data/upstream --output /path/to/new/auction-data/results
```
