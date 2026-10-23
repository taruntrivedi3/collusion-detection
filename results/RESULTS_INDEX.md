# Results index

Every figure and table, the data behind it, and the exact command and seeds that regenerate it.
Pre-sprint freeze: 2026-10-19 (UTC). `results/MANIFEST.tsv` records every file's size, SHA-256
hash and modification time at the freeze. It is the record of what existed before the sprint;
to check a file later, compare its hash and time with the manifest.

## How to regenerate

Setup once, from the project root:

    uv venv .venv && uv pip install --python .venv/bin/python numpy scipy pandas matplotlib pytest numba
    .venv/bin/python -m pytest -q tests          # 21 tests

Run every experiment command from `experiments/`:

    cd experiments && ../.venv/bin/python <script> <args>

Notes that apply to everything:
- **Seeds.** Every run derives its seed deterministically (formulas below). Most CSVs also store
  the seed per row (`seed` column). Worker count does not change results.
- **Resumable runs.** Scripts that write `cells/*.csv` skip cells already on disk. To regenerate a
  cell, delete (or move) its CSV first. Writing into an existing results folder overwrites files.
- **Run time** is for 16 workers on this container (24 GB memory limit; keep workers ≤ 16).
- **Code versions.** All outputs except those marked † were produced by code that is equivalent to
  the frozen code (later changes were speed-ups checked to give the same values to ≤ 1e-5).
  † marks outputs produced by an earlier method version; regenerating them with frozen code gives
  different numbers.

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

Summary tables in `results/SUMMARY.md` are copied from the files below (Table 1–2: step1;
Table 3: step4 / figures T; Tables 4–5: step4_final; Tables 6–7: step5; Table 8: step6).

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

## Step 2 — 5-agent counterexample (`results/step2/`) †

| Output | Command | Seeds |
|---|---|---|
| `matrices.txt` (γ, Γ, z matrices, detector outputs) | `step2_counterexample.py > ../results/step2/stdout.txt` | 7 (triangle), 8 (hub), 9 (pair); B = 200 |
| `recovery_runs.csv`, `recovery_summary.csv` | same | 1000000 × (cell index + 1) + rep; B = 200 |

† Produced before two method changes: edge-BH used raw permutation p-values (now gamma-fit), and
the scan + grow detector did not exist yet. Regenerating with frozen code changes the edge-BH
column and adds a scan + grow column. Γ, z and all other detectors are unaffected.

## Step 4 — planted structures, history-free (`results/step4/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `runs_n{5,10,20}.csv` (one row per run × detector) | `step4_planted.py 5 10 20` | SS[n, STRUCTS index, round(100 × CP), rounds, rep]; same seed for permutations; B = 200; 100 reps (n ≤ 10), 50 (n = 20) |
| `summary.txt`, `summary_structure_by_n.csv`, `cell_recovered_pct.csv`, `cell_false_coalitions.csv`, `null_fpr.csv`, `flagged_cells.csv`, `scan_loses_cells.csv` | `step4_planted.py --summary > ../results/step4/summary.txt` | reads `runs_n*.csv` |
| `superseded_perm_pvalues/` † | not regenerable by CLI | first pass with raw permutation p-values (edge-BH could not fire at n ≥ 10) |
| `n20_stdout.txt` | log | — |

Run time: about 30 minutes (n = 20 dominates).

## Step 4 final — firmed-up cells (`results/step4_final/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `cells/*.csv` (50 cells) | `step4_final.py run 16` | SS[n, STRUCTS index, round(100 × CP), rounds, rep, 1]; B = 1000; 100 reps (planted), 500 (null) |
| `summary.txt`, `recovered_ci.csv`, `recovered_ci_wide.csv`, `null_fpr_ci.csv`, `null_fpr_ci_wide.csv` | `step4_final.py summary > ../results/step4_final/summary.txt` | reads `cells/` |
| `stdout.txt` | log (empty: run restarted) | — |

Cells = every row of `step4/flagged_cells.csv` and `step4/scan_loses_cells.csv`, plus clique4 /
clique5 at n = 20, 100 rounds, all CP, plus null cells. Run time: about 60 minutes.

## Step 5 — history dependence, n = 10 (`results/step5/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `validate_shortcuts.csv` (gamma-fit and closed-form checks) | `step5_validate.py 1 > ../results/step5/validate_stdout.txt` | SS[mix index, state index, rounds, run, 55]; B = 1000 |
| `partA_n10/*.csv` (12 cells, no colluders) | `step5_run.py A 16 10` | SS[n, mix index, 0, rounds, rep, 5]; permutation seed = seed + state index + 1; B = 1000; 200 reps |
| `partB_n10/*.csv` (6 cells, triangle / clique4) | `step5_run.py B 16 10` | SS[n, mix index, planted index (1 triangle, 2 clique4), rounds, rep, 5]; as above |
| `summary.txt`, `partA_false_alarm_ci_n10.csv`, `partB_recovery_ci_n10.csv`, `partB_false_coalitions_n10.csv`, `partB_exact_and_clean_n10.csv` | `step5_run.py summary > ../results/step5/summary.txt` | reads the two folders |
| `partA_stdout.txt` | log (empty: run restarted) | — |

Mix index: uniform 0, leader 1, margin 2, mixed 3. State index: none 0, top 1, order 2, full 3.
Run time: about 35 minutes.

## Step 6 — triple scans (`results/step6/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `cells/*.csv` (72 cells) | `step6_triples.py 5 10 20 --workers 16` | SS[n, kind index (xor 0, triangle 1, pair 2, none 3), round(100 × CP), rounds, rep, 6]; B = 200; 100 reps (n ≤ 10), 50 (n = 20) |
| `summary.txt`, `summary_recovered_exact.csv`, `summary_contains_pair01_pair.csv`, `summary_detected_none.csv`, `planted_triple_values.csv`, `null_vs_chi2.csv` | `step6_triples.py --summary > ../results/step6/summary.txt` | reads `cells/` |

The B = 1000 null rerun (`step6_triples.py --null-final`) was skipped; no outputs exist.
Run time: about 40 minutes.

## Investigation — scan false alarms at n = 20, 300 rounds (`results/investigate_scan_fpr/`)

| Output | Command | Seeds / settings |
|---|---|---|
| `V0.csv` (original 500 games) | `investigate_scan_fpr.py run` | as Step 4 final (`none`, n = 20, 300 rounds); B = 1000 |
| `V1.csv`, `V2.csv`, `V3.csv` (1000 fresh games each) | same | SS[20, 300, rep, 777]; permutation seed = seed (V1) or seed + 1 (V2, V3); V3 permutes the data with seed + 2; B = 200 |
| `summary.csv`, `analysis.txt` | `investigate_scan_fpr.py summary` | reads V0–V3 |

Run time: about 40 minutes.

## Documents

| File | Content |
|---|---|
| `results/SUMMARY.md` | Plain-language summary of all experiments, figures embedded, data tables in the appendix |
| `results/SUMMARY.pdf` | PDF of SUMMARY.md (9 pages). Rebuild: `experiments/pdf/build_summary_pdf.sh` (pandoc 3 + xelatex, Liberation fonts; uses `experiments/pdf/template.tex` and `fonts.latex`, pandoc's defaults without `lmodern`) |
| `results/WRITEUP_NOTES.md` | Method notes, derivations, validation checks, decisions |
| `results/RESULTS_INDEX.md` | This file |
| `results/MANIFEST.tsv` | Path, bytes, SHA-256 and UTC modification time of every file at the freeze (written by `experiments/make_manifest.py`; rerunning it overwrites the record) |
