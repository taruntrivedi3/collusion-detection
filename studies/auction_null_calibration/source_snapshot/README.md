# Offline Agrawal auction detector prototype

This project analyzes published LLM auction trajectories without invoking any
LLM or importing the upstream simulator. It implements the exhaustive subset
scan, triple total correlation, regularized third-order interaction, a limited
lagged message/action scan, and a Bonjour-style pairwise baseline.

Upstream: https://github.com/andyjliu/llm-agent-collusion

Frozen commit: `2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff`.

The executed dataset, manifests, normalized tables, per-run results, scientific
figures, and report are under `/scratchB/trivedit/agrawal_offline_20261022/`.
The main report is `results/report.html` (also `results/REPORT.md`).

## What is implemented

- `fetch_data.py`: download a pinned archive, hash every file, and verify each
  file against the corresponding Git blob in the pinned upstream tree.
- `ingest.py`: parse every agent's actual prompt and response, reconcile numeric
  quotes against round results, verify trade prices where current quotes exist,
  independently compute seller profit/buyer surplus/welfare, and map evaluator
  IDs using the upstream alphabetical naming convention.
- `information.py`: batched discrete conditional information estimates, IPF,
  all subsets, two independent surrogate banks, and one maximum over all tested
  candidate subsets, encodings, and methods.
- `run_analysis.py`: score all episodes, preserve exact seeds and code/data
  hashes, and compare GPT-only conditions with the ten no-message GPT episodes.
- `validate_synthetic.py` and `test_information.py`: analytical examples,
  end-to-end recovery of planted signals, and independent-play calibration.
- `build_report.py`: report and exportable figures from saved results.

## Run

The executed analysis uses Python 3.11 at
`/home/trivedit/.conda/envs/andromeda_env/bin/python`. Package versions are saved
in `results/environment.json`. A new environment can install `requirements.txt`.
The scratch mount is BlobFuse; installing a virtual environment there failed
on a temporary RECORD file, so the existing scientific environment was used.

From this source directory:

```bash
export TMPDIR=/scratchB/trivedit/agrawal_offline_20261022
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$TMPDIR/matplotlib" MPLBACKEND=Agg
analysis_python=/home/trivedit/.conda/envs/andromeda_env/bin/python

"$analysis_python" fetch_data.py --destination "$TMPDIR"
"$analysis_python" ingest.py --source "$TMPDIR/upstream" --output "$TMPDIR/results"
"$analysis_python" -m pytest -q -s -p no:cacheprovider -c /dev/null test_information.py
"$analysis_python" run_analysis.py --data "$TMPDIR/results" --output "$TMPDIR/results" --permutations 1000 --workers 4
"$analysis_python" validate_synthetic.py --output "$TMPDIR/results" --null-trials 100 --workers 4
"$analysis_python" build_report.py --task-dir "$TMPDIR"
```

`pytest -s` avoids temporary-file capture, which this scratch filesystem cannot
reliably support. There are no LLM inference calls in these scripts. The only
network operations are the public GitHub download and source verification.
The scanner refuses to resume results with different code/data/seed/bank size;
choose a new output directory for a changed analysis.

## Frozen analysis choices

Each market episode is analyzed separately. Seller names remain local to the
episode; unrelated people with the same name in another episode are not pooled.
Round 1 is excluded from scanning so every observation has a previous action
and a previous-round message. This leaves 29 observations per episode.

Two action views:

1. **Direction** relative to the preceding submitted numeric ask: lower, hold,
   raise, explicit withdrawal, quote after no previous numeric submission, or
   JSON-null/no update. The upstream runner skips JSON `null`, whereas the
   literal string `"null"` is its withdrawal command. Those are not conflated.
2. **Level:** `(ask - seller_cost)/(maximum_buyer_value - seller_cost)` binned at
   0.50 and 0.75, plus separate withdrawal/no-update categories. With the saved
   costs/values this means `<= $90`, `($90, $95]`, and `> $95`.

The coarse observed state is the time third (rounds 1–10 / 11–20 / 21–30)
crossed with whether the previous round had any trades. It is not a sufficient
representation of the public history and does not identify causal influence.

The message view uses the **delivered** previous-round message. An explicitly
written dollar amount is mapped to the same bands; no amount and multiple bands
are separate categories. This is an observable price-mention feature, not an
LLM judge, intent classifier, or reliable extraction of a proposed price. Each
sender's lagged message feature and current action views are permuted as one
bundle, preserving their within-agent association.

For five sellers each episode has:

- 26 subset candidates for each of the two action views;
- 10 TC triples and 10 third-order triples per action view;
- 26 subsets of directed cross-agent lagged message/action edges.

One maximum therefore covers **118 candidates** per surrogate scheme. Both
surrogate banks contain 1,000 samples: bank A estimates normalization and bank B
calibrates the maximum. The reported p-value is `(1 + exceedances)/(1000 + 1)`.
Edges/candidates with null standard deviation below `1e-10` are marked
degenerate and assigned zero, not infinity. Every such flag is retained.

Third-order information is `KL(P || Q_pairwise)` in bits. `Q_pairwise` is fitted
by iterative proportional fitting. For the empirical scan, a uniform prior of
**total mass one per state/triple table** stabilizes sparse counts. This changes
the estimator from an unsmoothed plug-in estimate. All surrogate tables receive
the same prior. IPF must match pair marginals within `1e-8` or the run fails.
Analytical unit tests use zero prior and verify independent, Ising, triangle,
and XOR distributions. This is connected third-order interaction; it is not a
general partial-information-decomposition measure of synergy.

## Two explicitly limited surrogate schemes

**Within-state shuffle:** independently reorder each agent's bundles within
each coarse state. This preserves its observed conditional marginal frequencies
but destroys serial dependence. It requires exchangeability within those states
for a randomization-test interpretation; the auction trajectories do not
establish that assumption.

**Circular shift:** independently shift each agent's entire 29-round sequence,
including its feature bundle. This preserves within-agent cyclic ordering but
wraps the ending into the start and changes alignment with the public states.
It is a temporal-sensitivity diagnostic, not a validated market null.

Neither scheme establishes absence of collusion. Their flags are labeled
**surrogate dependence flags** throughout the outputs.

## Comparisons and limits

- The ten `base` episodes are no-message GPT controls, not known noncolluding
  ground truth: tacit coordination remains possible.
- For GPT-only conditions, a run's maximum is also ranked against the ten
  complete base episodes. For base episodes themselves this is leave-one-out.
  Ten controls give a minimum held-out rank of **1/11 = 0.0909**, so this
  reference cannot support a 5% threshold. Other model compositions do not get
  an unmatched reference rank.
- An episode-level maximum adjusts for its 118 candidates. It does not adjust
  exploratory browsing across 75 episodes or choosing between surrogate
  schemes. No study-wide 5% family-wise collusion claim is made.
- A significant subset can contain a dependent pair plus bystanders or combine
  separate groups. Ranked candidates are not identified coalition memberships.
  There is no iterative removal/step-down procedure.
- The published evaluator scores are comparison labels only. They are never
  used to choose bins, fit information scores, or select candidates.
- A fixed agreement with constant or independently randomized actions can have
  zero action MI/TC. Action scans cannot establish its absence.
- The source logs show different within-round public books in most
  Claude-containing runs, consistent with the smaller thread pool and mutable
  queues in the upstream runner. Model-composition comparisons therefore also
  involve observation timing; they do not isolate model effects.
- No new auction rollouts, counterfactual branches, or message interventions
  were run. The observed market outcome contrasts are descriptive.
