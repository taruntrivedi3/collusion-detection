# LLM double-auction calibration study

Read the [full report](results/report.html), [Markdown report](results/REPORT.md), or [findings](results/INTERPRETATION.md).

This directory contains the pinned public auction repository, all 75 markets, the parsed/audited actual LLM decisions, the independent-path benchmark, copied-triple tests, held-out source-market transfer tests and exploratory calibration repairs. No LLM API calls or new auction rollouts were made. All new files live under `/scratchB/quote_coverage_20261015/auction_null_calibration_20261023`.

- [Original frozen protocol](PROTOCOL.md), [configuration](config.json), [statistical interpretation](STATISTICAL_NOTES.md).
- [No-interaction benchmark figure](results/benchmark.png), [copied-path figure](results/copy_power.png), [transfer figure](results/transfer.png).
- [Source-style benchmark rates](results/benchmark_false_alarm.csv), [actual-market flags](results/real_condition_summary.csv), [copied-path results](results/copy_power.csv), [repaired-score transfer](results/source_score_transfer.csv).
- [Data verification](source_integrity.json), [parse reconciliation](results/parse_audit.json), [final integrity/replay audit](results/audit.json).
- Post-result additions are explicit: [matched donor counts](supplement_config.json), [raw-score calibration](analysis_additions.json), [raw-score transfer](score_transfer_config.json).

Reproduce or resume:

```bash
bash /scratchB/quote_coverage_20261015/auction_null_calibration_20261023/code/reproduce.sh
```

The code uses the existing Andromeda Python environment plus a small C++17 MI engine. It resumes only missing experiment outputs and regenerates analysis. The frozen source dataset is already present; the wrapper does not launch the upstream auction or any LLM API client. Source-style and common-normalization results are labeled separately. Generated groups are conditional empirical-path experiments, not additional independent real auction markets.
