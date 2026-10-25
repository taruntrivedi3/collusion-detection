# Results index

Every figure and table, the data behind it, and the exact command and seeds that regenerate it.

## How to regenerate

Setup once, from the project root:

    uv venv .venv && uv pip install --python .venv/bin/python numpy scipy pandas matplotlib pytest numba
    .venv/bin/python -m pytest -q tests          # 14 tests

Run every experiment command from `experiments/`:

    cd experiments && ../.venv/bin/python <script> <args>

Notes that apply to everything:
- **Seeds.** Every run derives its seed deterministically (formulas below). Most CSVs also store
  the seed per row (`seed` column). Worker count does not change results.
- **Resumable runs.** Scripts that write `cells/*.csv` skip cells already on disk. To regenerate a
  cell, delete (or move) its CSV first. Writing into an existing results folder overwrites files.
- **Workers.** Set `WORKERS=<k>` in the environment (default 12, about 0.5 GB each at n = 20).
  All outputs were produced with 64 workers.
- **Seats.** In Steps 4–5, colluders (and honest types) sit at random seats drawn from
  `np.random.default_rng([seed, 1])`; detectors are not told which seats are planted. Steps 1–2
  and Figure A use fixed seats because they are small worked examples.

Seed notation: `SS[a, b, ...]` = `int(np.random.SeedSequence([a, b, ...]).generate_state(1)[0])`.
`STRUCTS` index order (Step 4): pair 0, two_pairs 1, hub 2, triangle 3, clique4 4, clique5 5,
xor 6, none 7.

## Figures (`results/figures/`)

| Figure | Data | Command | Seeds / settings |
|---|---|---|---|
| A — cancellation (`figA_cancellation.png`) | `figA_cancellation.csv` | `make_figures.py A` | simulation and permutation seed 7 (triangle), 8 (hub); 5 players, CP 0.5, 1000 rounds, B = 1000 |
| B — dilution with n (`figB_dilution.png`) | `figB_dilution.csv` | `make_figures.py B` | reads `step4/runs_n{5,10,20}.csv` (seeds there) |
| C — weak broad cliques (`figC_weak_cliques.png`) | `figC_weak_cliques.csv` | `make_figures.py C` | reads `step4_final/cells/` |
| D — null false alarms (`figD_null_false_alarms.png`) | `figD_null_false_alarms.csv` | `make_figures.py D` | reads `step4_final/cells/none_*` |
| E — false alarms by history state (`figE_state_false_alarms.png`) | `figE_state_false_alarms.csv` | `make_figures.py E` | reads `step5/partA_n10/` |
| Table: structure × n with CIs | `table_structure_by_n_ci.csv` | `make_figures.py T` | reads `step4/runs_n*.csv` |

Summary tables in `results/SUMMARY.md` are copied from the files below (Table 1: step4 / figures T;
Tables 2–3: step4_final; Tables 4 and 8: step1; Tables 5–7: step5).

## Step 1 — reproduce Bonjour et al. (`results/step1/`)

| Output | Command | Seeds |
|---|---|---|
| `gamma_trace.csv/.png` (Γ vs rounds) | `step1_reproduce.py` | 100 + k, k = CP index 0..9 (CP 0.1..1.0) |
| `gamma_vs_cp.csv/.png` (Γ vs CP, 200 runs) | `step1_reproduce.py` | 200 + k |
| `cda.csv` (detection accuracy, no state) | `step1_reproduce.py` | 10000 + 1000 × case + rounds (case 0..5 in script order) |
| `cda_state_sensitivity.csv` | `step1_state_sensitivity.py` | 20000 + rounds + int(10 × CP) |
| `cda_low_cp.csv` | `step1_addenda.py` (also writes the next file) | 30000 + int(10 × CP) |
| `null_fpr_bonjour.csv` (3 uniform agents) | `step1_addenda.py` | one stream, seed 31000; permutation seed = iteration index; B = 200 |
| `cda_low_cp_by_rounds.csv` | `step1_addenda.py by_rounds` | 32000 + rounds + int(10 × CP) |

Run time: under 10 minutes total. The 27-state estimator is state kind `profile`.

## Step 2 — 5-agent counterexample (`results/step2/`)

| Output | Command | Seeds |
|---|---|---|
| `matrices.txt` (γ, Γ, z matrices, detector outputs) | `step2_counterexample.py > ../results/step2/stdout.txt` | 7 (triangle), 8 (hub), 9 (pair); B = 200 |
| `recovery_runs.csv`, `recovery_summary.csv` | same | 1000000 × (cell index + 1) + rep; B = 200 |

## Step 4 — planted structures, history-free (`results/step4/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `runs_n{5,10,20}.csv` (one row per run × detector) | `step4_planted.py 5 10 20` | SS[n, STRUCTS index, round(100 × CP), rounds, rep]; same seed for permutations; B = 200; 100 reps (n ≤ 10), 50 (n = 20) |
| `summary.txt`, `summary_structure_by_n.csv`, `cell_recovered_pct.csv`, `cell_false_coalitions.csv`, `null_fpr.csv`, `flagged_cells.csv`, `scan_loses_cells.csv` | `step4_planted.py --summary > ../results/step4/summary.txt` | reads `runs_n*.csv` |

Run time: about 30 minutes (n = 20 dominates).

## Step 4 final — firmed-up cells (`results/step4_final/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `cells/*.csv` (50 cells) | `step4_final.py run 16` | SS[n, STRUCTS index, round(100 × CP), rounds, rep, 1]; B = 1000; 100 reps (planted), 500 (null) |
| `summary.txt`, `recovered_ci.csv`, `recovered_ci_wide.csv`, `null_fpr_ci.csv`, `null_fpr_ci_wide.csv` | `step4_final.py summary > ../results/step4_final/summary.txt` | reads `cells/` |

Cells = every row of `step4/flagged_cells.csv` and `step4/scan_loses_cells.csv`, plus clique4 /
clique5 at n = 20, 100 rounds, all CP, plus null cells. Run time: about 60 minutes.

## Step 5 — history dependence, n = 10 (`results/step5/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `validate_shortcuts.csv` (gamma-fit and closed-form checks) | `step5_validate.py 1 > ../results/step5/validate_stdout.txt` | SS[mix index, state index, rounds, run, 55]; B = 1000 |
| `partA_n10/*.csv` (12 cells, no colluders) | `step5_run.py A 16 10` | SS[n, mix index, 0, rounds, rep, 5]; permutation seed = seed + state index + 1; B = 1000; 200 reps |
| `partB_n10/*.csv` (6 cells, triangle / clique4) | `step5_run.py B 16 10` | SS[n, mix index, planted index (1 triangle, 2 clique4), rounds, rep, 5]; as above |
| `partC_n10/*.csv` (18 cells, steer coalition / independent Exploiters, among `margin` honest agents) | `step5_run.py C 16 10` | SS[n, mix index, planted index (3 steer, 4 exploit), rounds, rep, 5, round(100 × CP)]; CP {0.1, 0.2, 0.3}; as above |
| `summary.txt`, `partA_false_alarm_ci_n10.csv`, `partB_recovery_ci_n10.csv`, `partB_false_coalitions_n10.csv`, `partB_exact_and_clean_n10.csv`, `partC_payoff_n10.csv`, `partC_detection_ci_n10.csv` | `step5_run.py summary > ../results/step5/summary.txt` | reads the two folders |

Mix index: uniform 0, leader 1, margin 2, mixed 3. State index: none 0, top 1, order 2, full 3.
Run time: about 35 minutes.

## Very large group — 18 of 20 players collude (`results/large_group/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `cells/*.csv` (10 cells) | `large_group.py run 64` | SS[20, 18, round(100 × CP), rounds, rep, 18]; seats from `default_rng([seed, 1])`; CP {0.05, 0.1, 0.2, 0.3, 0.5} × rounds {100, 300}; B = 1000; 100 reps |
| `summary.txt`, `summary_recovered.csv`, `summary_detected.csv`, `summary_best_jacc.csv` | `large_group.py summary > ../results/large_group/summary.txt` | reads `cells/` |

## Investigation — scan false alarms at n = 20, 300 rounds (`results/investigate_scan_fpr/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `V0.csv` (original 500 games) | `investigate_scan_fpr.py run` | as Step 4 final (`none`, n = 20, 300 rounds); B = 1000 |
| `V1.csv`, `V2.csv`, `V3.csv` (1000 fresh games each) | same | SS[20, 300, rep, 777]; permutation seed = seed (V1) or seed + 1 (V2, V3); V3 permutes the data with seed + 2; B = 200 |
| `summary.csv`, `analysis.txt` | `investigate_scan_fpr.py summary` | reads V0–V3 |

Run time: about 40 minutes.

## LLM auctions (`studies/llm_auction/`, `studies/llm_auction_5seller/`)

No command here calls an LLM; all games are saved in `runs/*/games.jsonl`.

| Output | Command (from the study directory) | Seeds / settings |
|---|---|---|
| 2 sellers: `report.pdf`, `detection.png` (every number in the 2-seller study) | `build_report.py` | games seeds 0–199 per condition; 2,000 shuffles; profit CIs bootstrap seed 0, 10,000 resamples |
| 2 sellers: batch rejection rates | `rates.py runs/rates_v1 same_moves` (or `cmi`) | disjoint batches of 5 / 10 / 20 games |
| 5 sellers: `runs/n5_rates_v1/analysis.txt` (profits, pooled test, batch rates) | `analyze_n.py runs/n5_rates_v1 5,10,20` | games seeds 100–199 per condition; 1,000 shuffles; test seed = job index |
| 5 sellers: `runs/n5_rates_v1/dry_run.txt` (scripted false alarms and power) | `dry_run_n.py 300` | 300 replications of 20 games × 50 hours; replication seed 1000 × i + 17 |
| Report Figure 4: `report/figures/fig4_llm_detection.png` | `python report/make_llm_figure.py` (from the repo root) | reads the two outputs above |

## Documents

| File | Content |
|---|---|
| `results/SUMMARY.md` | Plain-language summary of all experiments, figures embedded, data tables in the appendix |
| `results/WRITEUP_NOTES.md` | Method notes, derivations, validation checks, decisions |
| `results/RESULTS_INDEX.md` | This file |
