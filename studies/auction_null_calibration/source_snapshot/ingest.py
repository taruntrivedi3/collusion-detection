"""Parse published Markdown logs and reconcile actions, identities, and welfare."""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


SECTION = re.compile(r"^## (Prompt|Response): Round (\d+)[^\n]*\n(.*?)(?=^## (?:Prompt|Response): Round |\Z)", re.M | re.S)
ROUND = re.compile(r"^## Auction Results: Round (\d+)\s*\n(.*?)(?=^## Auction Results: Round |\Z)", re.M | re.S)


def fenced_json(section):
    match = re.search(r"^`{3,}json\s*\n", section, re.M)
    if not match:
        raise ValueError("Missing JSON fence")
    return json.JSONDecoder().raw_decode(section[match.end():].lstrip())[0]


def parse_agent(text):
    prompts, responses = {}, {}
    for kind, number, body in SECTION.findall(text):
        number = int(number)
        target = prompts if kind == "Prompt" else responses
        if number in target:
            raise ValueError(f"Duplicate {kind} in round {number}")
        target[number] = body if kind == "Prompt" else fenced_json(body)
    return prompts, responses


def parse_rounds(text):
    rows = []
    for number, body in ROUND.findall(text):
        row = fenced_json(body)
        if row["round_number"] != int(number):
            raise ValueError("Round heading/data mismatch")
        rows.append(row)
    return rows


def book_from_prompt(prompt, side):
    pattern = rf"## Current Status of {side} Queue\s*\n(.*?)(?=\n#|\Z)"
    match = re.search(pattern, prompt, re.S)
    if not match:
        raise ValueError(f"Missing {side} queue")
    book = ast.literal_eval(match.group(1).strip())
    if not isinstance(book, list) or any(not isinstance(x, tuple) or len(x) != 2 for x in book):
        raise ValueError("Unexpected book format")
    return book


def quote_value(raw):
    if raw is None:
        return None, "no_update"
    if isinstance(raw, str) and raw.strip().lower() == "null":
        return None, "withdraw"
    if isinstance(raw, bool):
        raise ValueError("Boolean quote")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Nonfinite quote")
    return value, "quote"


def direction_code(value, status, previous):
    if status == "withdraw":
        return 3
    if status == "no_update":
        return 5
    if previous is None:
        return 4
    delta = value - previous
    return 0 if delta < -1e-7 else 2 if delta > 1e-7 else 1


def price_band(value, cost, max_value):
    if max_value <= cost:
        raise ValueError("Price normalization requires positive gains from trade")
    fraction = (value - cost) / (max_value - cost)
    return 0 if fraction <= .5 else 1 if fraction <= .75 else 2


def message_code(message, cost, max_value):
    """Observable dollar mentions, not an inferred proposal/intent label."""
    values = [float(x.replace(",", "")) for x in re.findall(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)", message or "")]
    if not values:
        return 0
    bands = {price_band(value, cost, max_value) for value in values}
    return next(iter(bands)) + 1 if len(bands) == 1 else 4


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ingest(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    parsed_dir = output / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    summaries, actions, round_rows, issues = [], [], [], []
    audit = Counter()
    for metadata_path in sorted((source / "data").glob("*/experiment_metadata.json")):
        root = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        config = metadata["auction_config"]
        sellers = sorted(metadata["seller_ids"])
        buyers = metadata["buyer_ids"]
        costs = dict(zip(metadata["seller_ids"], config["seller_valuations"]))
        values = dict(zip(buyers, config["buyer_valuations"]))
        models = dict(zip(metadata["seller_ids"], config["seller_models"]))
        market = parse_rounds((root / "auction_results.md").read_text())
        expected = list(range(1, config["rounds"] + 1))
        if [row["round_number"] for row in market] != expected:
            raise ValueError(f"Noncontiguous auction rounds: {root.name}")
        logs = {agent: parse_agent((root / f"{agent}.md").read_text()) for agent in sellers + buyers}
        for agent, (prompts, responses) in logs.items():
            if sorted(prompts) != expected or sorted(responses) != expected:
                raise ValueError(f"Incomplete prompts/responses: {root.name}/{agent}")
            audit["agent_prompts"] += len(prompts)
            audit["agent_responses"] += len(responses)
        judge = json.loads((root / "llm_eval_full.json").read_text())
        evaluations = {}
        for record in judge:
            # Upstream evaluate.py maps sorted real names to seller_1, seller_2, ...
            index = int(record["seller_id"].split("_")[-1]) - 1
            if index not in range(len(sellers)):
                raise ValueError("Unmapped evaluator identity")
            for row in record["rounds"]:
                key = (sellers[index], row["round"])
                if key in evaluations:
                    raise ValueError("Duplicate evaluator entry")
                data = row.get("data", {}) if row["status"] == "success" else {}
                score = data.get("score")
                if score is not None and score not in (1, 2, 3, 4):
                    raise ValueError("Invalid judge score")
                evaluations[key] = {"score": score, "type": data.get("type"), "evidence": data.get("evidence", [])}
                audit["judge_success" if score is not None else "judge_missing"] += 1
        metrics = json.loads((root / "collusion_metrics.json").read_text())
        condition = re.sub(r"-\d+$", "", config["tag"])
        episode_actions, episode_rounds = [], []
        seller_profit = {agent: 0.0 for agent in sellers}
        buyer_surplus = {agent: 0.0 for agent in buyers}
        all_asks, all_trade_prices = [], []
        different_books = 0
        episode_issue_start = len(issues)
        for row in market:
            number = row["round_number"]
            snapshots = set()
            for agent in sellers + buyers:
                prompt, response = logs[agent][0][number], logs[agent][1][number]
                bid_book, ask_book = book_from_prompt(prompt, "Bid"), book_from_prompt(prompt, "Ask")
                snapshots.add(json.dumps([bid_book, ask_book]))
                field = "ask" if agent in sellers else "bid"
                value, status = quote_value(response.get(field))
                recorded = row["seller_asks" if agent in sellers else "buyer_bids"].get(agent)
                if status == "quote":
                    audit["numeric_quotes_checked"] += 1
                    if recorded is None or abs(value - float(recorded)) > 1e-7:
                        issues.append({"run": root.name, "round": number, "agent": agent,
                                       "check": "response_vs_auction_quote", "response": value, "recorded": recorded})
                audit[f"action_{status}"] += 1
                if agent not in sellers:
                    continue
                previous = quote_value(logs[agent][1][number - 1].get("ask"))[0] if number > 1 else None
                delivered = row.get("seller_messages", {}).get(agent, "")
                emitted = response.get("message_to_sellers") or ""
                if delivered != emitted:
                    audit["message_output_differs_from_delivered"] += 1
                ev = evaluations.get((agent, number), {"score": None, "type": None, "evidence": []})
                source_score = metrics.get(f"seller_{sellers.index(agent)+1}_coordination_score", [None] * len(expected))[number-1]
                if ev["score"] != source_score:
                    issues.append({"run": root.name, "round": number, "agent": agent, "check": "judge_vs_metrics",
                                   "judge": ev["score"], "metric": source_score})
                action = {"run": root.name, "condition": condition, "round": number, "agent": agent,
                          "model": models[agent], "ask": value, "status": status,
                          "direction": direction_code(value, status, previous),
                          "level": price_band(value, costs[agent], max(values.values())) if value is not None else (3 if status == "withdraw" else 4),
                          "message": delivered, "message_code": message_code(delivered, costs[agent], max(values.values())),
                          "judge_score": ev["score"], "judge_type": ev["type"],
                          "visible_best_bid": max((float(x[0]) for x in bid_book), default=None),
                          "visible_best_ask": min((float(x[0]) for x in ask_book), default=None),
                          "public_book_sha256": hashlib.sha256(json.dumps([bid_book, ask_book]).encode()).hexdigest()}
                episode_actions.append(action)
                if value is not None:
                    all_asks.append(value)
            different_books += int(len(snapshots) > 1)
            per_round_profit = per_round_surplus = 0.0
            for trade in row["trades"]:
                price = float(trade["price"])
                seller, buyer = trade["seller_id"], trade["buyer_id"]
                profit, surplus = price - costs[seller], values[buyer] - price
                seller_profit[seller] += profit
                buyer_surplus[buyer] += surplus
                per_round_profit += profit
                per_round_surplus += surplus
                all_trade_prices.append(price)
                ask, bid = row["seller_asks"].get(seller), row["buyer_bids"].get(buyer)
                if ask is not None and bid is not None:
                    audit["trade_prices_checked"] += 1
                    if abs(round((ask + bid) / 2, 2) - price) > 1e-7 or ask > bid:
                        issues.append({"run": root.name, "round": number, "check": "trade_price_or_cross"})
                else:
                    audit["trade_prices_with_carried_order"] += 1
            row_summary = {"run": root.name, "condition": condition, "round": number,
                           "trades": len(row["trades"]), "seller_profit": per_round_profit,
                           "buyer_surplus": per_round_surplus, "welfare": per_round_profit + per_round_surplus,
                           "mean_ask": mean(row["seller_asks"].values()) if row["seller_asks"] else None,
                           "public_book_snapshots": len(snapshots)}
            episode_rounds.append(row_summary)
        totals = {"combined_seller_profits": sum(seller_profit.values()),
                  "combined_buyer_surplus": sum(buyer_surplus.values()),
                  "total_welfare": sum(seller_profit.values()) + sum(buyer_surplus.values()),
                  "num_trades": len(all_trade_prices)}
        for key, value in totals.items():
            audit["economic_totals_checked"] += 1
            if abs(value - metrics[key]) > 1e-6:
                issues.append({"run": root.name, "check": key, "computed": value, "published": metrics[key]})
        judge_scores = [x["judge_score"] for x in episode_actions if x["judge_score"] is not None]
        summary = {"run": root.name, "condition": condition, "rounds": len(market), "sellers": len(sellers),
                   "buyers": len(buyers), "seller_comms": config["seller_comms_enabled"],
                   "oversight": config["oversight_enabled"], "boss_pressure": config["boss_pressure"],
                   "temperature": config["temperature"], "seller_models": ";".join(config["seller_models"]),
                   "buyer_models": ";".join(config["buyer_models"]),
                   "mean_judge_score": mean(judge_scores), "mean_ask": mean(all_asks),
                   "mean_trade_price": mean(all_trade_prices) if all_trade_prices else None,
                   "seller_profit": totals["combined_seller_profits"], "buyer_surplus": totals["combined_buyer_surplus"],
                   "total_welfare": totals["total_welfare"], "num_trades": totals["num_trades"],
                   "rounds_with_different_public_books": different_books,
                   "audit_issues": len(issues) - episode_issue_start}
        normalized = {"run": root.name, "config": config, "sellers": sellers, "buyers": buyers,
                      "summary": summary, "actions": episode_actions, "rounds": episode_rounds}
        (parsed_dir / f"{root.name}.json").write_text(json.dumps(normalized, indent=2) + "\n")
        summaries.append(summary)
        actions.extend(episode_actions)
        round_rows.extend(episode_rounds)
        audit["episodes"] += 1
        audit["market_rounds"] += len(market)
        print(f"parsed {audit['episodes']}: {root.name}", flush=True)
    write_csv(output / "episodes.csv", summaries)
    write_csv(output / "seller_actions.csv", actions)
    write_csv(output / "market_rounds.csv", round_rows)
    evidence = {"counts": dict(audit), "conditions": dict(Counter(x["condition"] for x in summaries)),
                "issues": issues, "judge_identity_mapping": "Alphabetically sorted seller names, from upstream evaluate.py",
                "action_source": "Agent JSON response, reconciled against auction_results.md",
                "messages_source": "Actual recorded seller_messages, not untruncated response text"}
    (output / "parse_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"audit": dict(audit), "issues": len(issues)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ingest(args.source, args.output)
