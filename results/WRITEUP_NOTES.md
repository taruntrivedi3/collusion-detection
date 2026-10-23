# Write-up notes (kept for the report appendix)

## Naming / attribution
- edge-BH: standard multiple-testing baseline (Benjamini & Hochberg 1995). Not in Bonjour et al.
- node score: our per-agent baseline, inspired by Rose et al. (2026)'s per-agent detectors
  (Peak Suspicion / Deception Split). It is NOT their method. Not in Bonjour et al.
- Bonjour-calibrated: same Gamma statistic and both-directions + exactly-one-pair rule as
  Bonjour et al.; only the threshold is permutation-calibrated.

## Step 1 reproduction (results/step1)
- Units are bits: mean Gamma(A;B) at CP = 1 is 1.58 (paper Fig 3 ~1.47; nats max is ln 3 = 1.10).
- Table 2 CDA is matched within ~5 points only when gamma conditions on the previous round's full
  action profile (27 states): cda_state_sensitivity.csv. The paper never states its state.
- CP = 0.1 "≈8% on average" is matched by the 27-state estimator averaged over 100-1000 rounds
  (6.7%; 17% at 100 rounds falling to 0.7% at 1000). True min Gamma ≈ 0.013 < alpha, so these
  detections are noise pushing Gamma over alpha. cda_low_cp_by_rounds.csv.
- UNEXPLAINED: CP = 0.2 at 1000 rounds: paper 24.3%; we get 49% (27-state) / 58% (no state).
- (i) Fixed-alpha false alarms on 3 uniform agents (% runs flagging any pair, 1000 runs):
  100 rounds 8.5% (none) / 43.4% (27-state); 300: 0.3% / 19.2%; 1000: 0.0% / 0.1%.
  Calibrated variant: 4.8-6.4% everywhere. null_fpr_bonjour.csv.
  Step 4 adds: fixed alpha at 100 rounds, no state: 11% (n=5), 19% (n=10), 28% (n=20).

## Step 2 (results/step2)
- Triangle: Gamma inside coalition within ±0.014 (all < alpha); z 182-212; outsiders |z| < 1.6.
- Hub: Gamma(primary; helper) 0.23-0.26 passes, Gamma(helper; primary) 0.029 fails.
- (iii) The hub's indirect helper-helper edge has z ≈ 30 at CP = 0.5 / 1000 rounds: motivation
  for conditional-MI direct/indirect analysis (e.g., I(a_A; a_D | a_B)) in future work.

## Step 4 method notes (results/step4)
- Per-edge permutation p-values have floor 1/(B+1); BH cannot reject when q / #edges < 1/(B+1)
  (n >= 10 at B = 200). Edge-BH and the grow step use gamma-fit p-values instead (moment-matched
  gamma on the Set-A permutation null; implied chi2 df ≈ 4.1 vs G-test theory df = 4). Null calibration
  at n = 10: P(p < x) / x = 1.0-1.2 for x = 0.05, 0.01; 1.0-1.6 at x = 0.001 (few events).
  First-pass results with raw permutation p-values: results/step4/superseded_perm_pvalues/.
- Subset null moments computed in closed form (T linear in edge z's): identical to sample
  moments to 1e-5.
- Full grid at B = 200 (dev). Final numbers at B = 1000 for all flagged / scan-loses cells, the
  Fig C cells, and null cells (500 reps): results/step4_final/.
- Peeling: per-stage 5% (step-down max-T); report false coalitions per run separately.

## Method notes added after Step 4 review

### Edge-BH p-value floor and the gamma-fit fix
- Problem: a permutation p-value p = (1 + #{null >= obs}) / (B + 1) cannot go below 1/(B + 1).
  BH rejects the smallest p only if p <= q / m (m = number of edges). At B = 200, q = 0.05:
  n = 5 (m = 10) -> bar 0.005 ≈ floor 0.00498, barely possible; n = 10 (m = 45) -> bar 0.0011,
  impossible; n = 20 (m = 190) would need B >= 3800 just to reject at all. First-pass Step 4
  edge-BH was 0% recovery and 0% false alarms at n >= 10, and the grow step never fired.
- Fix: per edge, fit a gamma distribution by moments (mean, variance) to the Set-A permutation
  null of gamma and use its upper tail. Justification: under independence with no state, the
  G-statistic 2N * MI_nats = 2N ln2 * gamma_bits is asymptotically chi2 with (3-1)(3-1) = 4
  degrees of freedom (chi2_df is a gamma with shape df / 2). The fits imply df = 4.06-4.10.
  With states, 2N ln2 * gamma = sum over states of G_s, approximately chi2 with sum of df_s, which
  is still a gamma, but sparse states break the asymptotics. Fitting moments to the
  permutation null (not using theoretical df) adapts to that; validated per state in Step 5.

### Closed-form subset null (scan)
- T(C) = sum_{i<j in C} z_ij = w_C . v, with v = vector of edge z's, w_C = 0/1 pair-membership.
  For the B Set-A permutation datasets v^(1..B): sample mean of T(C) = w_C . mean(v), sample
  variance = w_C' Cov(v) w_C (Cov with ddof = 0). This is an algebraic identity of sample
  moments: it holds for any null distribution and any state definition. It needs only that
  T is linear in v and that v is standardized by the same Set-A per-edge moments in both
  places. Numerical difference vs direct computation: <= 1.2e-5 (float32 accumulation).
- What it does not claim: S(C) is not assumed normal. The detection threshold is the
  empirical 95th percentile of max_C S(C) over Set B, so calibration is exact w.r.t. the
  permutation null up to Monte Carlo error.

### Trade-off: isolated weak pairs in short games
- On two_pairs at 100 rounds, CP 0.3, edge-BH beats the scan: 55 vs 42% (n = 10) and
  43 vs 32% (n = 20) in the B = 200 grid. The scan's misses there are "nothing found" or only
  one pair. Reason: for a lone edge, the scan's bar is the 95th percentile of a max over ~1M
  correlated subsets (S ≈ 4 at n = 20). BH's bar for the strongest edge is p <= 0.05 / 190,
  about z ≈ 3.5, and lower for later edges. The scan pays for searching all sizes; per-edge
  correction is cheaper when the signal is a single edge. This is a real trade-off, not a bug.

### Step 5 pre-check: shortcuts validated under state (results/step5/validate_shortcuts.csv)
Null data, n = 10, mixes {uniform, leader, margin, mixed} x state {none, top, order, full}
x rounds {300, 1000, 3000}, 4 runs each, B = 1000 (192 runs). States used: none 1, top 7,
order 13, full 48-63 (n = 10).
- (a) Gamma-fit p-values. Method: fit on Set A, evaluate on the 1000 Set-B datasets (fresh draws
  from the same within-state permutation null) -> 45 edges x 1000 = 45,000 null p's per run.
  Mean ratio P(p < x) / x per state x rounds cell (16 runs): x = 0.05: 0.99-1.02;
  x = 0.01: 0.96-1.04; x = 0.001: 0.83-1.14. Worst single run: 1.53 at x = 0.001
  (≈45 expected events per run, Poisson sd of ratio ≈ 0.15; max over 192 runs).
  Verdict: within the ±20% criterion at every state; no fallback needed. Slight liberal
  tendency in the far tail with state = none (1.11-1.14 at x = 0.001).
- (b1) Closed form vs Set-A sample moments of T(C): max error 7e-16 (identity, as derived).
- (b2) Out-of-sample: Set-A closed-form sd of T(C) vs empirical sd on Set B, all 1013 subsets.
  Median ratio 0.993-1.002; worst subset |ratio - 1| 0.10-0.20; worst mean shift 0.12-0.20 sd.
  Expected Monte Carlo spread for two independent B = 1000 sets, max over ~1000 subsets, is of
  this size (per-subset sd-ratio SE ≈ 4-5%). Verdict: no state-dependent bias; no fallback.

## Step 4 final numbers (B = 1000; results/step4_final/summary.txt)
- Null false alarms, 500 reps, 95% Wilson CI: calibrated detectors 3-9% everywhere; most CIs
  include 5%. Highest: edge-BH 9% [6, 11] and scan 8% [6, 11] at n = 20, 300 rounds (mildly
  liberal, consistent with the far-tail gamma-fit excess noted above). Node score's earlier
  12% (50 reps) is 6% [4, 8] at 500 reps. Fixed-alpha Bonjour: 8 / 14 / 23% at 100 rounds
  for n = 5 / 10 / 20, 0% at 300+ rounds.
- Scan beats max(node, edge-BH) by >= 20 points (100 reps): clique4 n=20 CP 0.3/100: 91 vs 59;
  clique5 n=20 CP 0.2/100: 44 vs 6; clique5 n=20 CP 0.3/100: 91 vs 61; triangle n=20 CP 0.2/300:
  91 vs 68. Scan loses: hub CP 0.2 (scan+grow closes it); two_pairs CP 0.3/100 rounds
  (edge-BH 58 vs 50 at n=10, 50 vs 36 at n=20).
- Table "structure x n" (table_structure_by_n_ci.csv) pools runs across a structure's CP x rounds
  cells; its Wilson CIs treat pooled runs as exchangeable and so understate between-cell
  heterogeneity. Per-cell CIs: results/step4_final/recovered_ci_wide.csv.

## Step 5 (n = 10, B = 1000, 200 reps; results/step5/summary.txt)
- Part A (no colluders). Uniform honest agents: all calibrated detectors ~5% under every state.
  Leader-reactive: state none -> 100% false alarms (edge-BH, node, scan); top/order/full ~5%
  (top is sufficient for leader_reactive). Margin-reactive: none, top and order are too coarse;
  false alarms grow with rounds (scan: top 12 / 45 / 100%, order 8 / 30 / 100% at
  300 / 1000 / 3000 rounds); full ~5%. Mixed: like margin.
- Fixed-alpha Bonjour reports few false alarms here (0-10%) for a structural reason: several
  mutually correlated honest agents cancel in Gamma, the same cancellation that hides coalitions.
  Calibrated Bonjour inflates (up to 70% with state none).
- Part B (triangle / clique4, CP 0.3, among mixed honest agents). "Recovered" alone hides the
  spurious honest coalitions; report "clean recovery" = planted found and no false coalition.
  Scan clean recovery: none 0-4%; top/order high at 300-1000 rounds, falling at 3000
  (triangle top 42%, order 61%) as power to detect unmodelled honest reactions grows;
  full 57-72% at 300 rounds, 93-98% at 1000+. This is the bias-variance trade-off of the state.

## Step 6: triple scans (no state, B = 200; results/step6/summary.txt)
- Null matches chi2 theory on the G scale G = 2N ln2 x stat (chi2_df has mean df, variance 2 df).
  Implied df (B = 200): TC 20.0-20.4 from the mean, 20.0-20.9 from the variance (theory df = 20
  = 27 - 1 - 3 x 2); third-order component E 8.0-8.3 from the mean, 8.0-8.6 from the variance
  (theory df = 8 = (3 - 1)^3). B = 1000 values: see "Step 6 null rerun" below. False alarms 4-12%
  (100 reps; one cell at 12% [7, 20], TC n = 10, 300 rounds).
- xor: edge scan 0-4% (false-alarm level); TC scan 80-100%; synergy scan 94-100%.
- triangle: TC scan >= edge scan (96-100 vs 86-100%); synergy scan weak at 300 rounds (2-68%),
  70-100% at 1000, 100% at 3000. The shared-signal triangle has real third-order structure:
  median E 0.024-0.071 bits (7-18% of TC), z up to 53.
- pair: TC scan returns a triple containing the pair in 98-100% of runs (expected: TC includes
  pairwise dependence); synergy scan 0-4% (correctly ignores pairwise-only structure).
- Framing correction: the TC scan did NOT lose power to the edge scan on triangles here. The cost
  is in the synergy scan (needs ~1000+ rounds) and in triples-only reporting (cannot return an
  exact pair or a 4+ coalition).
- Limitation: an order-k scan is blind to order-(k+1) parity. D = (A + B + C) mod 3 has every
  triple mutually independent (TC = E = 0). Detecting k-sparse parity needs search over C(n, k)
  groups with 3^k-cell tables; no general catch-all exists with finite data. Future work: Amari's
  full decomposition, partial information decomposition (Williams & Beer 2010), and
  payoff-based tests that target harmful coordination of any order.

## Compute note
- Container memory limit is 24 GB (cgroup memory.max) and CPU quota 50 cores (the environment
  presets OMP_NUM_THREADS = 8, which makes `nproc` report 8). An n = 20 scan worker at B = 1000
  used ~4 GB with cached membership matrices; 44 such workers most likely exceeded the limit and
  restarted the container. Fixed: uncached 32 MB chunks (0.51 GB peak per worker), one resumable
  CSV per cell (experiments/common.py), 16 workers.

## Investigation: subset scan 8% false alarms at n = 20, 300 rounds (results/investigate_scan_fpr/)
- Observed (Step 4 final): 40/500 = 8.0% [5.9, 10.7]; edge-BH 8.6% on the same games.
- (b) Code review of subset_scores: the real statistic and every Set-B statistic use the same
  Set-A edge mean / sd (edge_z), the same subset mean / variance formula (one mu, one inv_sd),
  and the same subset universe (masks). The only difference is float64 vs float32 sums for the
  real vs Set-B values (~1e-6 relative), far too small to matter.
- Seed coupling check: simulation and permutation used the same integer seed. The first permuted
  datasets from that seed show normal null gamma (0.0084-0.0101 vs theory 0.0096) and
  corr(action, key) = -0.019. Not the cause. (Hygiene: future runs should use distinct child
  seeds for simulation and permutation; not changed, to keep existing results reproducible.)
- Level of the rule S > np.quantile(maxB, 0.95) under an exact null: 5.45% at B = 200,
  5.09% at B = 1000 (interpolated quantile; see analysis.txt).
- (a)/(c) Reruns, n = 20, 300 rounds, no collusion:
  V0 original 500 games, B = 1000: scan 8.0% [5.9, 10.7]; p-value deciles chi2 = 27.2 (df 9).
  V1 1000 fresh games, same pipeline (shared seed), B = 200: scan 4.8% [3.6, 6.3]; chi2 6.5.
  V2 same games, independent permutation seed: scan 3.7% [2.7, 5.1]; chi2 5.6.
  V3 "real" input = a within-state permutation of the simulated data: scan 6.9% [5.5, 8.6]; chi2 10.9.
  V1-V3 pooled: scan 154/3000 = 5.13% [4.40, 5.98] vs level 5.45%. Edge-BH pooled 7.73% [6.83, 8.74].
- Conclusions.
  Strongly supported: no bug; the scan is calibrated (fresh data reproduce the level; p-values
  uniform in all three fresh variants; V3, the exact-exchangeability test, is within ~2 SE).
  Strongly supported: simulated null data and the permutation null do not differ in a way that
  matters (i.i.d. uniform agents make them identical by construction; V1 vs V3 within ~2 SE).
  Tentative: the original 8% is chance in that specific set of 500 games (p ≈ 0.001 for its
  p-value non-uniformity; it was the worst of ~45 calibrated detector x null-cell rates, so
  selection makes it less surprising; edge-BH also elevated on the same games). No fix applied.
- Edge-BH excess is real and B-dependent: 7.7% pooled at B = 200, 5-9% at B = 1000. Consistent
  with the liberal far tail of the gamma-fit p-values (moment estimates noisier at small B).
  Left as is, per decision; report it.

## Step 6 null rerun: skipped (decision 2026-10-19)
- The B = 1000 / 500-rep rerun of the Step 6 null cells was cancelled before it started. Step 6
  false-alarm rates stand at B = 200, 100 reps (50 at n = 20): 4-12%, consistent with 5%. The
  triple scans use the same max-over-Set-B calibration whose exactness the investigation above
  confirmed for the subset scan. The code path remains (step6_triples.py --null-final).
