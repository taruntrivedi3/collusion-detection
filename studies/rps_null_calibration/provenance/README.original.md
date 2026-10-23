# Null calibration experiment

Start with [the readable report](results/report.html), [Markdown report](results/REPORT.md), or [the findings](results/INTERPRETATION.md).

The study includes 25 settings; 25,000 independent honest evaluation games, 25,000 separate honest calibration games, 12,000 coordinated games, and 11,000 additional source-style honest games. Independent normalization banks contain a further 50,000 games. No LLM calls were made. All new files are in this directory; the supplied repository was only read.

- [Frozen protocol](PROTOCOL.md), [exact configuration](config.json), [statistical notes](STATISTICAL_NOTES.md).
- [False-alarm figure](results/false_alarm.png), [power figure](results/power.png); PDF versions are alongside them.
- CSV summaries and per-game results: `results/`. Observed trajectories, seeds, scores and reference quantiles: `results/chunks/`. Normalization data: `results/banks/`.
- [Final integrity and replay audit](results/audit.json). Source hashes: `source_manifest.json`; frozen design/engine hashes: `experiment_manifest.json`.

Reproduce or resume with:

```bash
bash /scratchB/quote_coverage_20261015/null_calibration_20261022/code/reproduce.sh
```

The wrapper uses the existing Andromeda Python environment and a small C++17 engine. It does not refreeze the configuration. New simulations are generated only for missing chunks. The main comparison and the source-style sensitivity use different normalizations; consult the report before combining their results.
