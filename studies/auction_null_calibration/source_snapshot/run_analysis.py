"""Run offline information scans on complete parsed market episodes."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from information import scan


def assemble(episode):
    sellers = episode["sellers"]
    actions = {(row["round"], row["agent"]): row for row in episode["actions"]}
    rounds = range(2, episode["summary"]["rounds"] + 1)
    views = {view: np.array([[actions[(number, agent)][view] for agent in sellers] for number in rounds])
             for view in ("direction", "level")}
    messages = np.array([[actions[(number-1, agent)]["message_code"] for agent in sellers] for number in rounds])
    # This is an intentionally coarse observed state, not a sufficient market state.
    states = np.array([min((number - 1) // 10, 2) * 2 + int(episode["rounds"][number-2]["trades"] > 0)
                       for number in rounds])
    return views, states, messages


def analyze_one(path, output, mode, permutations, seed):
    raw = path.read_bytes()
    episode = json.loads(raw)
    source_dir = Path(__file__).parent
    algorithm_hash = hashlib.sha256(b"".join((source_dir / name).read_bytes()
                                             for name in ("information.py", "run_analysis.py"))).hexdigest()
    data_hash = hashlib.sha256(raw).hexdigest()
    destination = output / "scans" / f"{path.stem}__{mode}.json"
    effective_seed = seed + int(hashlib.sha256(f"{path.stem}:{mode}".encode()).hexdigest()[:8], 16)
    if destination.exists():
        result = json.loads(destination.read_text())
        if (result["permutations_per_bank"] != permutations or result["seed"] != effective_seed
                or result.get("algorithm_sha256") != algorithm_hash or result.get("parsed_data_sha256") != data_hash):
            raise ValueError("Refusing resume with different code, data, seed, or permutation count")
        return result
    if episode["summary"]["audit_issues"]:
        raise ValueError(f"Unresolved parse audit issues: {path.stem}")
    views, states, messages = assemble(episode)
    result = scan(views, states, permutations=permutations, seed=effective_seed, mode=mode, messages=messages)
    result["run"] = path.stem
    result["sellers"] = episode["sellers"]
    result["condition"] = episode["summary"]["condition"]
    result["algorithm_sha256"] = algorithm_hash
    result["parsed_data_sha256"] = data_hash
    for candidate in result["candidates"]:
        candidate["member_names"] = [episode["sellers"][i] for i in candidate["members"]]
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def aggregate(results, output):
    rows, candidates = [], []
    for result in results:
        best = result["candidates"][0]
        row = {"run": result["run"], "condition": result["condition"], "mode": result["mode"],
               "omnibus_score": result["omnibus_score"], "omnibus_surrogate_p": result["omnibus_surrogate_p"],
               "winner_method": best["method"], "winner_view": best["view"],
               "winner_members": ";".join(best["member_names"]), "winner_size": len(best["members"]),
               "states_used": len(result["states"]), "smallest_state": min(result["states"].values()),
               "nondegenerate_candidates": sum(c["nondegenerate"] for c in result["candidates"]),
               "bonjour_direction_pairs": result["details"]["direction"]["bonjour_pairs_above_005"],
               "bonjour_level_pairs": result["details"]["level"]["bonjour_pairs_above_005"],
               "ipf_max_error": max(result["details"][view]["ipf_max_error"] for view in ("direction", "level")),
               "ipf_max_iterations": max(result["details"][view]["ipf_max_iterations"] for view in ("direction", "level"))}
        for method in ("subset", "tc", "third_order", "message"):
            winner = next(c for c in result["candidates"] if c["method"] == method)
            row[f"{method}_score"] = winner["score"]
            row[f"{method}_p_max_adjusted"] = winner["p_max_adjusted"]
            row[f"{method}_members"] = ";".join(winner["member_names"])
            row[f"{method}_view"] = winner["view"]
        rows.append(row)
        for candidate in result["candidates"]:
            candidates.append({"run": result["run"], "mode": result["mode"], "condition": result["condition"],
                               "method": candidate["method"], "view": candidate["view"],
                               "members": ";".join(candidate["member_names"]), "size": len(candidate["members"]),
                               "score": candidate["score"], "p_max_adjusted": candidate["p_max_adjusted"],
                               "nondegenerate": candidate["nondegenerate"]})
    frame = pd.DataFrame(rows).sort_values(["mode", "run"])
    # Each session contributes one maximum. This is an empirical reference rank,
    # not a claim that no-message sessions are known noncolluding null samples.
    frame["base_reference_rank"] = np.nan
    frame["base_reference_n"] = 0
    matched = {"base", "base-seller_comms", "boss_pressure", "oversight", "boss_pressure-oversight"}
    for mode in frame["mode"].unique():
        reference = frame[(frame["condition"] == "base") & (frame["mode"] == mode)]
        for index, row in frame[(frame["mode"] == mode) & frame["condition"].isin(matched)].iterrows():
            other = reference[reference["run"] != row["run"]]
            frame.at[index, "base_reference_rank"] = (1 + (other["omnibus_score"] >= row["omnibus_score"] - 1e-12).sum()) / (len(other) + 1)
            frame.at[index, "base_reference_n"] = len(other)
    frame.to_csv(output / "scan_results.csv", index=False)
    pd.DataFrame(candidates).to_csv(output / "candidate_scores.csv", index=False)
    summary = frame.groupby(["mode", "condition"]).agg(
        episodes=("run", "size"), median_score=("omnibus_score", "median"),
        surrogate_flags=("omnibus_surrogate_p", lambda x: int((x <= .05).sum())),
        median_surrogate_p=("omnibus_surrogate_p", "median"),
        median_base_reference_rank=("base_reference_rank", "median"))
    summary.to_csv(output / "scan_condition_summary.csv")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modes", nargs="+", default=["state_shuffle", "circular_shift"])
    args = parser.parse_args()
    (args.output / "scans").mkdir(parents=True, exist_ok=True)
    files = sorted((args.data / "parsed").glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_one, path, args.output, mode, args.permutations, args.seed): (path.stem, mode)
                   for path in files for mode in args.modes}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"scored {len(results)}/{len(futures)} {result['run']} {result['mode']}: surrogate p={result['omnibus_surrogate_p']:.4f}", flush=True)
    aggregate(results, args.output)
