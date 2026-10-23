# Archived LLM double-auction null calibration

Start with the [findings](results/INTERPRETATION.md),
[full report](results/REPORT.md), or downloadable [HTML report](results/report.html).

This extends the calibration question to 75 archived markets with five sellers,
five buyers and 30 rounds each. It includes two action views, 52 subset/view
candidates, within-state shuffles, circular shifts, independent whole-path
resampling, copied triples, and transfer across held-out source markets.

The independent-path benchmark is conditional on a finite empirical donor pool.
It does not generate dynamically consistent auction histories or certify the
donor agents as honest. There are only ten no-message base markets. The report
therefore distinguishes surrogate flags, synthetic false-alarm rates and actual
market evidence. No new LLM runs were made.

- [Primary protocol](PROTOCOL.md), [configuration](config.json), [statistical notes](STATISTICAL_NOTES.md).
- [Benchmark](results/benchmark_false_alarm.csv), [copied-triple results](results/copy_power.csv),
  [score-calibration transfer](results/source_score_transfer.csv),
  [actual-market flags](results/real_condition_summary.csv).
- [Source verification](source_integrity.json), [parser reconciliation](results/parse_audit.json),
  [original replay audit](results/audit.json).
- Exploratory additions: [matched donor counts](supplement_config.json),
  [raw-score repair](analysis_additions.json), [repair transfer](score_transfer_config.json).

From the repository root:

```bash
python scripts/reproduce.py auction --output /path/to/new/auction-check
python scripts/reproduce.py auction --mode full --workers 8 --audit --output /path/to/new/auction-run
```

Prepared inputs (`results/cases.json`, `pool.npz`, `pool_provenance.json` and
`state_audit.json`) are included, so the main experiment does not need a download.
The full audit fetches the public upstream source and verifies its checksums.
To fetch and parse the original prompts independently:

```bash
python studies/auction_null_calibration/source_snapshot/fetch_data.py --destination /path/to/new/auction-data
python studies/auction_null_calibration/source_snapshot/ingest.py --source /path/to/new/auction-data/upstream --output /path/to/new/auction-data/results
```

Source: [andyjliu/llm-agent-collusion](https://github.com/andyjliu/llm-agent-collusion/tree/2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff),
commit `2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff`. Derived action tables and
prepared inputs retain source-market and agent identifiers. The upstream source
and its original prompts are fetched separately rather than vendored here.

See the [root README](../../README.md) for setup and provenance. The historical
`code/reproduce.sh` remains unchanged and machine-specific. Use the portable
wrapper above. `code/prepare.py` creates a new frozen design from parsed data;
it refuses to replace an existing configuration and is unnecessary for the
included prepared inputs. Intermediate banks/chunks are regenerated in a full run.
