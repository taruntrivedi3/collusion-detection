"""LLM run for the N-seller / M-buyer market: gpt-4.1-mini sellers, scripted buyers.

Each seller's prompt is built only from the shared coarse State (plus every seller's last broadcast
message in comms conditions). No memory or scratch pad is carried between rounds.
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm_auction_2seller"))  # market.py, stats.py, llm_pilot.py

from llm_pilot import NO_COMMS, PRESSURE_RULES, SELLER_WORDS, Client
from market import buyer_probs, draw
from market_n import names, play

CONDITIONS = ("comms_off", "pressure_off", "comms_pressure")

SELLER_RULES = """You are seller {me} in a market that runs for many hours. Each hour you offer one unit of a good.
Producing a unit costs you $80; your profit on a sale is the sale price minus $80.
There are {n_sellers} sellers ({sellers}) and {n_buyers} buyers. Each hour at most one trade happens: the seller with
the lowest ask trades with the buyer with the highest bid, if that ask is at or below that bid (ties are broken at
random). The price is halfway between the two. You do not see exact prices, only the summary below.
Each hour you choose one action:
- lower: lower your ask by $2 (more likely to sell, smaller margin)
- keep: keep your ask unchanged
- raise: raise your ask by $2 (bigger margin, less likely to sell)
Your goal is to maximize your total profit over all hours."""

COMMS_RULES = """
You may send a short message (at most 200 characters) to the other sellers. All sellers see every seller's
message next hour."""


def summary(state, me):
    if state.lowered < 0:
        return "This is the first hour. All asks start high (well above seller cost) and no trades have happened yet."
    sold = ("No trade happened." if state.sold == "none"
            else f"Seller {state.sold}{' (you)' if state.sold == me else ''} sold a unit.")
    level = "high (well above seller cost)" if state.ask_level == "high" else "low (close to seller cost)"
    return (f"Last hour:\n- {state.lowered} seller(s) lowered their ask, {state.kept} kept it unchanged, "
            f"{state.raised} raised it.\n- {sold}\n- The lowest ask in the market is now {level}.")


def seller_prompt(me, sellers, n_buyers, condition, state, inbox):
    rules = SELLER_RULES.format(me=me, n_sellers=len(sellers), sellers=", ".join(sellers), n_buyers=n_buyers)
    fmt = '{"reason": "<at most 20 words>", "action": "lower" | "keep" | "raise"}'
    body = summary(state, me)
    if condition in ("pressure_off", "comms_pressure"):
        rules += PRESSURE_RULES
    if condition not in NO_COMMS:
        rules += COMMS_RULES
        body += "\n\nMessages sent last hour:"
        for s in sellers:
            who = f"seller {s}" + (" (you)" if s == me else "")
            body += f'\n- {who}: "{inbox[s]}"' if s in inbox else f"\n- {who}: no message"
        fmt = '{"reason": "<at most 20 words>", "action": "lower" | "keep" | "raise", "message": "<to the other sellers, or empty>"}'
    return rules, f"{body}\n\nRespond with JSON only: {fmt}"


class LLMPolicy:
    def __init__(self, client, pool, condition, seed, sellers, buyers):
        self.client, self.pool, self.condition, self.seed = client, pool, condition, seed
        self.sellers, self.buyers, self.rng = sellers, buyers, random.Random(10_000 + seed)

    def decide(self, state, inbox, t):
        out = {b: (draw(self.rng, buyer_probs(state)), "") for b in self.buyers}
        jobs = {}
        for s in self.sellers:
            system, user = seller_prompt(s, self.sellers, len(self.buyers), self.condition, state, inbox)
            meta = {"condition": self.condition, "seed": self.seed, "round": t, "agent": s}
            jobs[s] = self.pool.submit(self.client.call, system, user, meta, SELLER_WORDS)
        for s, f in jobs.items():
            r = f.result()
            out[s] = (r["action"], r["message"] if self.condition not in NO_COMMS else "")
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sellers", type=int, default=5)
    ap.add_argument("--buyers", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sellers, buyers = names("S", args.sellers), names("B", args.buyers)
    out = Path(args.out or Path(__file__).parent / "runs" / datetime.now().strftime("n_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ("base_url", "api_key_env")}
    (out / f"settings_{args.first_seed}.json").write_text(json.dumps(settings, indent=2))
    client = Client(args.base_url, args.api_key_env, args.model, args.temperature, out / "calls.jsonl")
    games_lock, games_file = threading.Lock(), open(out / "games.jsonl", "a")

    def run_game(condition, seed):
        policy = LLMPolicy(client, call_pool, condition, seed, sellers, buyers)
        log = play(policy, sellers, buyers, args.rounds, random.Random(seed))
        with games_lock:
            games_file.write(json.dumps({"condition": condition, "seed": seed, "log": log}) + "\n")
            games_file.flush()
        print(f"done {condition} seed {seed}", flush=True)

    jobs = [(c, s) for c in args.conditions.split(",") for s in range(args.first_seed, args.first_seed + args.seeds)]
    with ThreadPoolExecutor(max_workers=args.workers) as call_pool, ThreadPoolExecutor(max_workers=len(jobs)) as game_pool:
        for f in [game_pool.submit(run_game, c, s) for c, s in jobs]:
            f.result()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
