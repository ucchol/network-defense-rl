"""Run the analyst demonstration or the matched containment benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .benchmark import run_benchmark
from .model import (
    HOSTS, INITIAL_BUDGET, MONITOR, PRESENT, SERVICE_FLOWS, ZONES, Controls, Regime, initial_belief,
)
from .policies import recommendation


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("recommend", help="rank actions after a synthetic initial observation")
    demo.add_argument("--alert", choices=("positive", "negative"), default="positive")
    demo.add_argument("--depth", type=int, choices=(1, 2, 3), default=2)

    benchmark = commands.add_parser("benchmark", help="run matched simulations")
    benchmark.add_argument("--suite", choices=("core", "full"), default="full")
    benchmark.add_argument("--episodes", type=int, default=100)
    benchmark.add_argument("--seed", type=int, default=1000)
    benchmark.add_argument("--budget", type=int, default=INITIAL_BUDGET)
    benchmark.add_argument("--output", type=Path, help="optional summary CSV")
    benchmark.add_argument("--episodes-out", type=Path, help="optional paired episode CSV")
    args = parser.parse_args()
    if args.command == "recommend":
        belief = initial_belief(args.alert == "positive")
        controls = Controls()
        report = {
            "scenario": "synthetic three-zone containment",
            "initial_observation": {"host": HOSTS[MONITOR[0]], "alert": args.alert},
            "budget_tokens": controls.budget,
            "planning_ticks": args.depth,
            "risk_window_ticks": args.depth,
            "risk_assumption": "Apply the first action, then wait for the remainder of the risk window.",
            "required_service_flows": [[HOSTS[src], HOSTS[dst]] for src, dst in SERVICE_FLOWS],
            "host_compromise_estimates": [
                {"host": host, "zone": zone, "probability": round(float(probability), 6)}
                for host, zone, probability in zip(HOSTS, ZONES, belief @ PRESENT, strict=True)
            ],
            "ranked_actions": recommendation(belief, Regime(), controls, depth=args.depth),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        if args.episodes < 2 or args.seed < 0 or args.budget < 0:
            parser.error("--episodes must be at least 2 and --seed and --budget must be nonnegative")
        if args.output and args.episodes_out and args.output.resolve() == args.episodes_out.resolve():
            parser.error("summary and episode outputs must have different paths")
        summary, episodes = run_benchmark(args.episodes, args.seed, args.suite, budget=args.budget)
        if args.output:
            write_csv(args.output, summary)
        if args.episodes_out:
            write_csv(args.episodes_out, episodes)
        if not args.output:
            print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
