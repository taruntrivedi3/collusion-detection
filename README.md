# Collusion detection and null calibration

Research code for Apart Research's AI Collusion Sprint, research question 1.4.
The project studies subset and triple dependence scans, and whether their
reference distributions remain calibrated when agents respond to public history.

## Results

| Study | Question | Report |
| --- | --- | --- |
| Original RPS experiments | Detecting groups with pairwise, total-correlation and synergy scans | [Summary and figures](results/SUMMARY.md), [PDF](results/SUMMARY.pdf) |
| Adaptive RPS null calibration | Compare within-state shuffles, circular shifts and newly simulated honest games | [Report](studies/rps_null_calibration/results/REPORT.md) |
| Archived LLM double auctions | Do the same references work on short, changing seller trajectories? | [Report](studies/auction_null_calibration/results/REPORT.md) |

In the RPS study, sufficient-state shuffling was approximately calibrated in the
tested regimes; omitting relevant history produced severe false alarms. This is
empirical evidence for those policies, not a general validity theorem.

In the auction study, independently sampled complete seller paths still produced
88.5% within-state flags and 37.2% circular-shift flags under the source-style
three-phase scan. Same-pool calibration could repair size, but transfer to held-out
source markets was unstable. Those synthetic paths are not newly simulated honest
auction histories, and flags on actual markets do not establish collusion.
The study used 75 existing markets and made no new LLM calls.

## Setup and checks

Use Python 3.11+ on Linux, with `g++` supporting C++17 for the two calibration
studies. The original Python experiments also use Numba.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_package.py
python -m pytest -q tests -p no:cacheprovider
python scripts/reproduce.py rps --output /path/to/new/rps-check
python scripts/reproduce.py auction --output /path/to/new/auction-check
```

The default study command compiles its engine and runs the original numerical
and statistical invariant checks. It writes into the requested new directory,
keeping the published results unchanged. `/path/to/new/...` is a placeholder for
your own writable output path.

For the complete simulations and reports, use a separate output directory:

```bash
python scripts/reproduce.py rps --mode full --workers 8 --output /path/to/new/rps-run
python scripts/reproduce.py auction --mode full --workers 8 --output /path/to/new/auction-run
```

Full runs are substantially more expensive than the checks: they regenerate
thousands of surrogate banks. The auction supplements retain their original
12-worker setting; `--workers` controls the primary runner. Add `--audit` to a
full run for the archived replay/count audit. The auction audit also downloads
the pinned public source archive (about 25 MB compressed, 304 MB unpacked).
No command above runs the upstream auction or calls an LLM.

The earlier RPS scripts remain under `experiments/`; for example:

```bash
cd experiments
python step1_reproduce.py
```

The original implementation and historical outputs retain older comparison
methods. Both new calibration studies use the first subset scan, without group
growth or repeated peeling. They do not claim strong family-wise error control
for iterative coalition removal.

## Layout and provenance

- `collusion/`, `experiments/`, `tests/`, `results/`: original RPS project.
- `studies/rps_null_calibration/`: adaptive-policy simulation, frozen protocol,
  source snapshots, per-game CSVs, reports and original audit receipt.
- `studies/auction_null_calibration/`: auction parser snapshot, prepared inputs,
  resampling and transfer code, frozen designs, results and audit receipts.
- `scripts/`: portable reproduction and package verification.
- `study_provenance.json`: original SHA-256 for each copied study artifact.
- `package_manifest.json`: SHA-256 inventory of this distribution.

Scientific source, frozen configurations and reported results were copied
unchanged. Study READMEs and the portable wrapper are distribution additions;
the original READMEs are under each study's `provenance/` directory. Historical
manifests retain the original run's paths and platform information. They also
list intermediate banks/chunks that are regenerated rather than bundled here.
Compiled libraries, caches and the upstream auction archive are not included.

The portable audit preserves a copy of the original design manifest and records
the rebuilt library's hash separately before substituting only that library hash
in the output directory's audit manifest. This allows a different compiler while
keeping the scientific source and input checks intact; it is not a claim of
byte-identical binaries across platforms.

The auction data comes from
[andyjliu/llm-agent-collusion](https://github.com/andyjliu/llm-agent-collusion/tree/2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff),
commit `2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff`. See the auction study for
attribution, data limits and the distinction between primary and exploratory
analyses. Dependency versions used for the saved results appear in each
`results/environment.json`.
