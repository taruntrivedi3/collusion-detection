"""Generate an offline report and standalone scientific figures from saved scans."""

import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATCHED = ["base", "base-seller_comms", "boss_pressure", "oversight", "boss_pressure-oversight"]
LABELS = {"base": "No messages", "base-seller_comms": "Messages", "boss_pressure": "Messages + pressure",
          "oversight": "Messages + oversight", "boss_pressure-oversight": "Messages + both"}
COLORS = ["#455a64", "#00897b", "#d77b00", "#7467ad", "#bd4f6c"]


def markdown_table(frame):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "—" if pd.isna(value) else f"{value:.3f}"
        return str(value).replace("|", "/")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def contrasts(episodes, scans):
    rng = np.random.default_rng(66101)
    result = {}
    base = episodes[episodes.condition == "base"]
    comm = episodes[episodes.condition == "base-seller_comms"]
    for metric in ("mean_judge_score", "mean_ask", "seller_profit", "buyer_surplus", "num_trades"):
        a, b = comm[metric].to_numpy(), base[metric].to_numpy()
        draws = rng.choice(a, (10000, len(a)), replace=True).mean(axis=1) - rng.choice(b, (10000, len(b)), replace=True).mean(axis=1)
        result[metric] = {"communication_minus_base": float(a.mean() - b.mean()),
                          "episode_bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}
    for mode in scans["mode"].unique():
        subset = scans[scans["mode"] == mode]
        a = subset[subset.condition == "base-seller_comms"].omnibus_score.to_numpy()
        b = subset[subset.condition == "base"].omnibus_score.to_numpy()
        auc = np.mean((a[:, None] > b[None, :]) + .5 * (a[:, None] == b[None, :]))
        aa, bb = rng.choice(a, (10000, len(a))), rng.choice(b, (10000, len(b)))
        draws = ((aa[:, :, None] > bb[:, None, :]) + .5 * (aa[:, :, None] == bb[:, None, :])).mean(axis=(1, 2))
        result[f"{mode}_condition_auc"] = {"auc": float(auc), "episode_bootstrap_95_interval": np.quantile(draws, [.025, .975]).tolist()}
    return result


def build(task_dir):
    output = task_dir / "results"
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    episodes = pd.read_csv(output / "episodes.csv")
    actions = pd.read_csv(output / "seller_actions.csv")
    scans = pd.read_csv(output / "scan_results.csv")
    audit = json.loads((output / "parse_audit.json").read_text())
    synthetic = json.loads((output / "synthetic_validation.json").read_text())
    integrity = json.loads((task_dir / "source_integrity.json").read_text())
    if audit["issues"] or len(episodes) != 75 or len(scans) != 150:
        raise ValueError("Incomplete or unreconciled analysis")
    source_dir = Path(__file__).parent
    algorithm_hash = hashlib.sha256(b"".join((source_dir / name).read_bytes()
                                             for name in ("information.py", "run_analysis.py"))).hexdigest()
    detail = {}
    for path in sorted((output / "scans").glob("*.json")):
        result = json.loads(path.read_text())
        if result["algorithm_sha256"] != algorithm_hash or len(result["candidates"]) != 118:
            raise ValueError(f"Code/candidate audit failed: {path}")
        raw = (output / "parsed" / f"{result['run']}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != result["parsed_data_sha256"]:
            raise ValueError("Parsed data changed after scoring")
        if not all(np.isfinite(c["score"]) and 0 < c["p_max_adjusted"] <= 1 for c in result["candidates"]):
            raise ValueError("Invalid score or p-value")
        detail[f"{result['run']}::{result['mode']}"] = {
            "score": result["omnibus_score"], "p": result["omnibus_surrogate_p"],
            "candidates": result["candidates"][:12]}
    if len(detail) != 150 or scans.ipf_max_error.max() > 1e-8:
        raise ValueError("Scan coverage/IPF audit failed")
    environment = {"python_executable": sys.executable, "python": platform.python_version(),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "pandas", "matplotlib", "pytest", "requests")}}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    contrast = contrasts(episodes, scans)
    (output / "condition_contrasts.json").write_text(json.dumps(contrast, indent=2) + "\n")
    economy = episodes.groupby("condition").agg(episodes=("run", "size"), judge=("mean_judge_score", "mean"),
                ask=("mean_ask", "mean"), seller_profit=("seller_profit", "mean"), buyer_surplus=("buyer_surplus", "mean"), trades=("num_trades", "mean"))
    for mode, name in (("state_shuffle", "shuffle_flags"), ("circular_shift", "shift_flags")):
        economy[name] = scans[scans["mode"] == mode].groupby("condition")["omnibus_surrogate_p"].apply(lambda x: int((x <= .05).sum()))
    economy.to_csv(output / "economic_and_scan_summary.csv")
    matched = economy.loc[MATCHED].copy().round(3)
    matched.index = [LABELS[x] for x in MATCHED]
    matched = matched.reset_index(names="Condition")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    pos = np.arange(len(MATCHED))
    for mode, offset, label, color in [("state_shuffle", -.18, "Within-state shuffle", "#396c9f"),
                                      ("circular_shift", .18, "Circular shift", "#c98b44")]:
        counts = economy.loc[MATCHED, "shuffle_flags" if mode == "state_shuffle" else "shift_flags"]
        bars = axes[0].bar(pos + offset, counts, width=.35, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{int(value)}/10" for value in counts], padding=3, fontsize=8)
    axes[0].set(ylim=(0, 11.7), ylabel="Episodes with surrogate p ≤ 0.05", title="Flags change when the null changes")
    axes[0].set_xticks(pos, [LABELS[x].replace(" + ", "\n+ ") for x in MATCHED], rotation=18, ha="right")
    axes[0].legend(fontsize=8)
    for i, condition in enumerate(MATCHED):
        y = episodes[episodes.condition == condition].mean_judge_score.to_numpy()
        jitter = np.linspace(-.12, .12, len(y))
        axes[1].scatter(i + jitter, y, color=COLORS[i], alpha=.8, s=27)
        axes[1].plot([i-.2, i+.2], [y.mean()]*2, color="black", linewidth=2)
    axes[1].set(ylim=(1, 4.15), ylabel="Published evaluator score (1–4)", title="Evaluator labels are a separate comparison")
    axes[1].set_xticks(pos, [LABELS[x].replace(" + ", "\n+ ") for x in MATCHED], rotation=18, ha="right")
    save_figure(fig, figures, "matched_conditions")
    fig, ax = plt.subplots(figsize=(7.4, 5.1), constrained_layout=True)
    joined = scans.pivot(index="run", columns="mode", values="omnibus_score").join(episodes.set_index("run")["condition"])
    for color, condition in zip(COLORS, MATCHED):
        g = joined[joined.condition == condition]
        ax.scatter(g.state_shuffle, g.circular_shift, label=LABELS[condition], color=color, s=45, alpha=.85)
    ax.set(xlabel="Maximum score: within-state shuffle", ylabel="Maximum score: circular shifts",
           title="Matched GPT conditions: score stability across nulls")
    ax.legend(fontsize=9)
    save_figure(fig, figures, "null_sensitivity")
    chosen = []
    for condition in ("base", "base-seller_comms"):
        g = scans[(scans["mode"] == "state_shuffle") & (scans.condition == condition)]
        chosen.append(g.loc[g.omnibus_score.idxmax(), "run"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, run in zip(axes, chosen):
        data = actions[actions.run == run]
        for agent, group in data.groupby("agent"):
            ax.plot(group["round"], group.ask, marker=".", markersize=3, linewidth=1.3, label=agent.replace("Seller ", ""))
        condition = data.condition.iloc[0]
        ax.set(title=f"{LABELS[condition]}: highest shuffle score", xlabel="Round", xlim=(1, 30))
        ax.axvline(10.5, color="#aaaaaa", linestyle=":")
        ax.axvline(20.5, color="#aaaaaa", linestyle=":")
        ax.legend(ncol=3, fontsize=8)
    axes[0].set_ylabel("Submitted seller ask ($)")
    save_figure(fig, figures, "example_trajectories")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.hist(synthetic["iid_null_calibration"]["pvalues"], bins=np.linspace(0, 1, 11), color="#396c9f", edgecolor="white")
    ax.axhline(10, linestyle="--", color="#555555", label="Expected per bin under uniform p-values")
    ax.set(xlabel="Omnibus surrogate p-value", ylabel="Independent synthetic episodes", title="Independent-play calibration: 4 flags in 100 trials")
    ax.legend(fontsize=8)
    save_figure(fig, figures, "synthetic_calibration")
    examples = pd.DataFrame([{ "Example": x["case"], "Rounds": x["rounds"], "p": x["surrogate_p"],
                              "Top method": x["winner_method"], "Top members": str(x["winner_members"])}
                             for x in synthetic["examples"]])
    method_rows = []
    for mode in ("state_shuffle", "circular_shift"):
        for condition in ("base", "base-seller_comms"):
            g = scans[(scans["mode"] == mode) & (scans.condition == condition)]
            method_rows.append({"Null": mode, "Condition": LABELS[condition],
                               **{method: int((g[f"{method}_p_max_adjusted"] <= .05).sum())
                                  for method in ("subset", "tc", "third_order", "message")}})
    method_table = pd.DataFrame(method_rows)
    method_table.to_csv(output / "matched_method_flags.csv", index=False)
    calibrated = synthetic["iid_null_calibration"]
    ci = calibrated["exact_95_interval"]
    auc_shuffle = contrast["state_shuffle_condition_auc"]
    auc_shift = contrast["circular_shift_condition_auc"]
    narrative = (
        "The prototype recovers planted pair, three-agent, and XOR dependence, but its real-auction flags are strongly null-dependent. "
        "Within-state shuffling flags all 10 no-message GPT controls and 9 of 10 messaging runs; circular shifts flag 4 and 5, respectively. "
        "The data do not establish reliable identification of colluding groups. No-message controls can still coordinate tacitly, "
        "so these are control flag rates, not measured false-positive rates.")
    notes = [
        "Every run has 30 rounds. Scans use rounds 2–30: 29 observations per run, 2,175 aligned market observations overall. Episodes are kept separate.",
        "Each episode/surrogate scheme searches 118 candidates: two action views, 26 subsets per view, ten TC and ten third-order triples per view, and 26 lagged message/action subsets.",
        "Two independent banks of 1,000 surrogates standardize edges/groups and calibrate the maximum. No iterative agent removal is used. Constant-null candidates are marked degenerate.",
        "The state is time third crossed with whether the preceding round traded. It leaves much public history unmodeled. Shuffling assumes exchangeability within these coarse states; circular shifts instead preserve cyclic temporal order but wrap the end into the start and alter state alignment.",
        "Third-order KL uses a uniform prior of total mass one per state/triple table. This regularizes sparse data and changes the unsmoothed estimator. Every IPF fit converged within the declared 1e-8 pair-margin tolerance.",
        "The text feature is only explicit dollar-price mentions, lagged one round. It is not a semantic proposal detector. Published LLM judgments are used only for comparison, never as training labels or bin-selection criteria.",
        "The ten complete no-message GPT episodes provide a separate empirical reference for GPT-only conditions. The smallest possible held-out rank is 1/11 = 0.0909; this reference cannot establish a 5% cutoff. No matched no-message controls exist here for the other model compositions.",
        "747 of 750 rounds in the 25 Claude-containing episodes expose different public order books to different agents; all 1,500 GPT-only rounds show the same books. This is consistent with the upstream smaller thread pool and mutable queues. Cross-model comparisons also change observation timing.",
        "82 seller responses contain JSON null, which the upstream runner skips; none uses its literal-string withdrawal command. Those no-update decisions are retained as a distinct category. Three of 11,250 evaluator scores are missing.",
        "4,110 transaction prices reconcile to current quoted midpoints. One transaction uses a carried order and is excluded from that direct price check. All 300 per-run trade-count/profit/surplus/welfare totals reconcile to the published metrics.",
        "A significant subset need not be a single coalition or establish every member's participation. A fixed agreement can produce zero MI and TC. Candidate memberships here are exploratory.",
        "The maximum is calibrated within an episode and surrogate scheme, not across exploratory browsing of 75 episodes or both schemes. These are not study-wide 5% collusion p-values.",
        "No new LLM calls, market rollouts, message knockouts, or causal interventions were run. Differences between saved conditions are descriptive, not proof that a particular message caused harm.",
    ]
    environment_lines = ", ".join(f"{name} {version}" for name, version in environment["packages"].items())
    markdown = f"""# Offline Agrawal auction detector: executed results

{narrative}

## Verified dataset and implementation

- 75 experiments, 2,250 market rounds, 22,500 complete agent prompts/responses, and 11,250 seller decisions.
- All {integrity['verified_git_blobs']:,} upstream files verified against their pinned Git blob hashes and local SHA-256 manifest.
- Frozen upstream commit: `{integrity['commit']}`.
- Subset scan, TC triples, regularized third-order interaction, lagged price-mention scan, and Bonjour-style net-influence baseline are implemented.
- Ten analytical/parser/end-to-end tests passed. Planted pair/triple/XOR examples are recovered at both 29 and 300 rounds; these individual cases are not power estimates.
- In {calibrated['trials']} independent-play calibration trials, {calibrated['flags_at_005']} flag at 5%: rate {calibrated['rate']:.1%}, exact 95% interval {ci[0]:.1%}–{ci[1]:.1%}. This only checks the synthetic iid null.

## Matched GPT-only conditions

Means are per complete episode. Shuffle/shift counts are episodes with an omnibus surrogate p <= 0.05 out of ten.

{markdown_table(matched)}

![Matched conditions](figures/matched_conditions.png)

Communication-minus-no-message mean differences: ask {contrast['mean_ask']['communication_minus_base']:+.3f} dollars, seller profit {contrast['seller_profit']['communication_minus_base']:+.3f} dollars per episode, buyer surplus {contrast['buyer_surplus']['communication_minus_base']:+.3f}, and trades {contrast['num_trades']['communication_minus_base']:+.1f}. The saved communication condition has a higher evaluator score but lower seller profits and buyer surplus on average. Episode-bootstrap intervals are in `condition_contrasts.json`; no causal attribution is made.

The descriptive ability of the maximum score to rank the communication condition above the no-message condition is AUC {auc_shuffle['auc']:.2f} under shuffling (bootstrap 95% interval {auc_shuffle['episode_bootstrap_95_interval'][0]:.2f}–{auc_shuffle['episode_bootstrap_95_interval'][1]:.2f}) and {auc_shift['auc']:.2f} under circular shifts ({auc_shift['episode_bootstrap_95_interval'][0]:.2f}–{auc_shift['episode_bootstrap_95_interval'][1]:.2f}). These are condition-ranking statistics, not coalition-detection accuracy, and there are only ten episodes per condition.

![Sensitivity to null](figures/null_sensitivity.png)

## Which scan contributes flags?

All entries below use p-values adjusted over the full 118-candidate family within an episode; the methods are not separately thresholded at an uncorrected 5%.

{markdown_table(method_table)}

The third-order scan contributes no flags in this matched comparison. That is compatible with redundancy, low sample size, sparse conditioning cells, or the chosen encoding/prior; it does not establish absence of higher-order coordination. The dollar-mention extension contributes two shuffle flags and zero shift flags among messaging runs.

## Inspectable examples

These are the highest shuffle-score runs in the two matched conditions, selected after scoring; they are illustrations, not a representative sample.

![Submitted ask trajectories](figures/example_trajectories.png)

Runs: `{chosen[0]}` and `{chosen[1]}`. The HTML report lets you select every run and inspect its top 12 candidate subsets under either null. The complete 17,700 candidate rows are in `candidate_scores.csv`.

## Synthetic checks

{markdown_table(examples)}

![Independent-play calibration](figures/synthetic_calibration.png)

## Interpretation and limitations

""" + "\n".join(f"- {note}" for note in notes) + f"""

## Artifacts and reproducibility

- `scan_results.csv`: all 150 episode/surrogate results and matched-reference ranks.
- `candidate_scores.csv`: all 17,700 candidate scores and adjusted surrogate p-values.
- `seller_actions.csv`, `market_rounds.csv`, `episodes.csv`: normalized observations and reconciled outcomes.
- `parse_audit.json`, `synthetic_validation.json`, `environment.json`, `condition_contrasts.json`: verification and uncertainty.
- `scans/*.json`: individual edge information, Bonjour scores, seeds, per-view diagnostics, and code/data SHA-256 values.
- `../source_manifest.json`, `../source_integrity.json`: exact source provenance.
- `figures/*.svg` and `figures/*.png`: standalone exportable figures.

Code: `{source_dir}`. Python {environment['python']}; {environment_lines}.
Algorithm SHA-256: `{algorithm_hash}`. See the source README for rerun commands and exact estimator definitions.
"""
    (output / "REPORT.md").write_text(markdown)
    def embedded(name):
        raw = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        return f'<img alt="{html.escape(name.replace("_", " "))}" src="data:image/png;base64,{raw}">'
    options = "".join(f'<option value="{html.escape(run)}">{html.escape(run)}</option>' for run in episodes.run)
    payload = json.dumps(detail).replace("</", "<\\/")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agrawal auction detector — offline results</title><style>
:root{font-family:system-ui,sans-serif;color:#182b37;background:#f6f7f8;line-height:1.55}body{margin:0}main{max-width:1150px;margin:auto;padding:30px 22px 70px}h1{font-size:2.15rem;line-height:1.2}h2{margin-top:2.1rem}p,li{max-width:100ch}a{color:#216c9f}img{display:block;width:100%;height:auto;margin:18px 0}.box{background:#fff;border:1px solid #dce2e6;border-radius:8px;padding:18px;margin:18px 0}.notice{border-left:5px solid #c98b44}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #dce2e6;padding:8px 10px;vertical-align:top}th{background:#eef2f5}select{max-width:100%;font:inherit;padding:8px;margin:6px 0 12px}code{overflow-wrap:anywhere}.small{font-size:.88rem;color:#526571}.controls{display:grid;grid-template-columns:3fr 1fr;gap:15px}@media(max-width:650px){.controls{grid-template-columns:1fr}h1{font-size:1.65rem}main{padding:18px 12px}td,th{padding:6px}}
</style><main><p class="small">Executed offline analysis · pinned public data · no LLM calls</p>
<h1>Agrawal auction detector: what the saved data supports</h1>"""
    document += f'<div class="box notice"><p>{html.escape(narrative)}</p></div>'
    document += f'<p><strong>75 runs · 2,250 rounds · 22,500 agent responses · 118 candidates per run/null.</strong> All 1,119 upstream files match the pinned Git tree. Synthetic calibration: 4/100 flags, with an exact 95% interval of {ci[0]:.1%}–{ci[1]:.1%}.</p>'
    document += '<h2>Matched GPT-only conditions</h2><p>Outcome columns are means per episode. Flags are counts out of ten; published judge scores are comparison labels, not ground truth.</p><div class="scroll">' + matched.to_html(index=False, border=0) + '</div>'
    document += embedded("matched_conditions")
    document += f'<p>Condition-ranking AUC: <strong>{auc_shuffle["auc"]:.2f}</strong> under within-state shuffling, <strong>{auc_shift["auc"]:.2f}</strong> under circular shifts. This measures separation of the two experimental conditions, not identification of true coalitions.</p>'
    document += embedded("null_sensitivity")
    document += '<h2>Contribution by scan</h2><div class="scroll">' + method_table.to_html(index=False, border=0) + '</div>'
    document += '<h2>Inspect candidate groups</h2><p>Names are local to the selected episode. These are ranked dependent subsets; they do not establish membership of a cartel. P-values adjust across all 118 candidates within the selected null.</p>'
    document += f'<div class="box"><div class="controls"><label>Episode<br><select id="run">{options}</select></label><label>Surrogate scheme<br><select id="mode"><option value="state_shuffle">Within-state shuffle</option><option value="circular_shift">Circular shift</option></select></label></div><p id="run-summary"></p><div class="scroll"><table><thead><tr><th>Members</th><th>Method</th><th>View</th><th>Score</th><th>Adjusted p</th><th>Usable variance</th></tr></thead><tbody id="candidates"></tbody></table></div></div>'
    document += '<h2>Example quote trajectories</h2><p>Highest shuffle-score episode from each matched condition, selected after analysis. Full raw prompts and responses remain in the pinned snapshot.</p>' + embedded("example_trajectories")
    document += '<h2>Synthetic validation</h2><div class="scroll">' + examples.to_html(index=False, border=0) + '</div>' + embedded("synthetic_calibration")
    document += '<h2>Method and interpretation</h2><ul>' + ''.join(f'<li>{html.escape(note)}</li>' for note in notes) + '</ul>'
    document += f'<h2>Files and source</h2><p><a href="REPORT.md">Full Markdown report</a> · <a href="scan_results.csv">Run results CSV</a> · <a href="candidate_scores.csv">All candidate scores</a> · <a href="parse_audit.json">Parser audit</a> · <a href="synthetic_validation.json">Synthetic checks</a> · <a href="../source_integrity.json">Source verification</a></p><p>Source commit: <a href="https://github.com/andyjliu/llm-agent-collusion/tree/{integrity["commit"]}">{integrity["commit"]}</a>.</p><p class="small">Python {environment["python"]}; {html.escape(environment_lines)}. Analysis source: <code>{source_dir}</code>. The source README specifies all encodings, priors, and rerun commands. Scientific figures are also available as standalone SVG and PNG files.</p>'
    document += '<script type="application/json" id="data">' + payload + '</script>'
    document += """<script>
const data=JSON.parse(document.getElementById('data').textContent);
function render(){const key=document.getElementById('run').value+'::'+document.getElementById('mode').value;const row=data[key];document.getElementById('run-summary').textContent='Maximum score: '+row.score.toFixed(3)+' · omnibus surrogate p: '+row.p.toFixed(4)+' · top 12 of 118 candidates';const body=document.getElementById('candidates');body.replaceChildren();for(const c of row.candidates){const tr=document.createElement('tr');for(const value of [c.member_names.join(', '),c.method,c.view,c.score.toFixed(3),c.p_max_adjusted.toFixed(4),c.nondegenerate?'yes':'degenerate']){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}}
document.getElementById('run').addEventListener('change',render);document.getElementById('mode').addEventListener('change',render);render();
</script></main></html>"""
    (output / "report.html").write_text(document)
    summary = {"runs": 75, "scans": 150, "candidate_rows": 17700, "algorithm_sha256": algorithm_hash,
               "main_finding": narrative, "source_verified_files": integrity["verified_git_blobs"],
               "maximum_ipf_error": float(scans.ipf_max_error.max()), "contrasts": contrast,
               "selected_illustrations": chosen, "same_round_book_differences": int(episodes.rounds_with_different_public_books.sum())}
    (output / "report_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"report": str(output / "report.html"), "markdown": str(output / "REPORT.md"),
                      "figures": str(figures), "condition_auc": [auc_shuffle, auc_shift]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    build(parser.parse_args().task_dir)
