"""Paired, reproducible simulation with latent state hidden from policies."""

from __future__ import annotations

from itertools import product

import numpy as np

from .model import (
    CRITICAL, INITIAL_BUDGET, LOSS, MONITOR, PRIOR, SERVICE_FLOWS, TICKS, Controls, Regime,
    attack_edges, available_services, frontier, initial_belief, legal_moves,
    observation_branches, positive_probability, sample, tick_cost, transition,
)
from .policies import Planner, value_first


def run_episode(
    regime: Regime, policy, seed: int, *, stress: bool = False, budget: int = INITIAL_BUDGET
) -> dict:
    # Separate precomputed channels keep the draws aligned even if policies inspect
    # in different rounds or one run has no reachable attack target.
    rng = np.random.default_rng(seed)
    initial_draw, initial_alert = rng.random(2)
    spread_draw, alert_draw, scan_draw = rng.random((3, TICKS))
    state = sample(PRIOR, initial_draw)
    foothold = state != 0
    controls = Controls(budget=budget)
    belief = np.zeros_like(PRIOR)
    if regime.noisy:
        belief = initial_belief(initial_alert < positive_probability(MONITOR[0])[state])
    else:
        belief[state] = 1.

    loss = cost = downtime = spent = inspections = 0.
    critical_hit = False
    contained_at = None

    for tick in range(TICKS):
        action = policy(belief.copy(), controls, tick)
        options = dict(legal_moves(regime, controls))
        if action not in options:
            raise ValueError(f"Policy proposed infeasible action: {action}")
        controls = options[action]
        state = sample(transition(regime, controls, stress)[state], spread_draw[tick])
        loss += LOSS[state]
        observed_state = np.zeros_like(PRIOR)
        observed_state[state] = 1.
        cost += tick_cost(observed_state, action, controls)
        downtime += len(SERVICE_FLOWS) - sum(available_services(controls))
        spent += action.cost
        inspections += action.kind == "inspect"
        critical_hit |= bool(CRITICAL[state])

        if (
            foothold and not critical_hit and contained_at is None
            and not frontier(state, attack_edges(regime, controls))
        ):
            contained_at = tick + 1

        if not regime.noisy:
            belief = observed_state
        else:
            alert = alert_draw[tick] < positive_probability(MONITOR[tick + 1])[state]
            scan = None
            if action.kind == "inspect":
                scan = scan_draw[tick] < positive_probability(action.host, inspection=True)[state]
            prediction = belief @ transition(regime, controls)
            belief = next(
                posterior for obs, _, posterior in
                observation_branches(prediction, regime, tick + 1, action) if obs == (alert, scan)
            )

    return {
        "seed": seed,
        "foothold": foothold,
        "critical_hit": critical_hit,
        "compromise_value_fraction": float(loss / TICKS),
        "cost": float(cost),
        "downtime_flow_ticks": int(downtime),
        "tokens_used": int(spent),
        "inspections": int(inspections),
        "contained_at": contained_at,
    }


def suites(name: str):
    if name == "core":
        return [(Regime(), False)]
    factorial = [
        (Regime(graph=graph, noisy=noisy, protect=protect), False)
        for graph, noisy, protect in product((False, True), repeat=3)
    ]
    return factorial + [
        (Regime(), True),
        (Regime(alternate=True), False),
        (Regime(alternate=True), True),
    ]


def paired_ci(values: np.ndarray, *, seed: int = 0):
    """Paired percentile bootstrap interval across independent scenario seeds."""
    rng = np.random.default_rng(seed)
    # Bound peak memory for large runs while keeping the sample order fixed.
    means = []
    for _ in range(20):
        samples = rng.integers(len(values), size=(100, len(values)))
        means.extend(values[samples].mean(axis=1))
    lo, hi = np.quantile(means, [.025, .975])
    return float(lo), float(hi)


def run_benchmark(episodes: int, seed: int, suite: str, *, budget: int = INITIAL_BUDGET):
    if episodes < 2 or seed < 0 or budget < 0:
        raise ValueError("Benchmark requires at least two episodes and nonnegative seed and budget")
    if suite not in {"core", "full"}:
        raise ValueError(f"Unknown benchmark suite: {suite}")
    summary, traces = [], []
    for regime, stress in suites(suite):
        greedy = Planner(regime, 1)
        planner = Planner(regime, 2)
        open_loop = Planner(regime, 2, adaptive=False)

        def value_policy(belief, controls, tick):
            return value_first(belief, regime, controls, tick)

        policies = {
            "value_first": value_policy,
            "greedy": greedy.act,
            "open_loop": open_loop.act,
            "planner": planner.act,
        }
        results = {
            name: [
                run_episode(regime, policy, seed + i, stress=stress, budget=budget)
                for i in range(episodes)
            ]
            for name, policy in policies.items()
        }
        for name, rows in results.items():
            for row in rows:
                traces.append({
                    "regime": regime.name, "stress": stress, "policy": name,
                    "budget_tokens": budget, **row,
                })
            costs = np.array([r["cost"] for r in rows])
            differences = np.array([r["cost"] for r in results["greedy"]]) - costs
            lo, hi = paired_ci(differences)
            incidents = [r for r in rows if r["foothold"]]
            contained = [r["contained_at"] for r in incidents if r["contained_at"] is not None]
            tail = max(1, int(np.ceil(.05 * episodes)))
            compromise = sorted(r["compromise_value_fraction"] for r in rows)
            summary.append({
                "regime": regime.name,
                "stress": stress,
                "policy": name,
                "episodes": episodes,
                "budget_tokens": budget,
                "start_seed": seed,
                "mean_cost": float(costs.mean()),
                "mean_cost_saved_vs_greedy": float(differences.mean()),
                "paired_cost_saved_ci_low": lo,
                "paired_cost_saved_ci_high": hi,
                "critical_compromise_rate": float(np.mean([r["critical_hit"] for r in rows])),
                "critical_compromise_rate_given_foothold": (
                    float(np.mean([r["critical_hit"] for r in incidents])) if incidents else None
                ),
                "mean_compromise_value_fraction": float(np.mean(compromise)),
                "worst_5pct_compromise_value_fraction": float(np.mean(compromise[-tail:])),
                "mean_downtime_flow_ticks": float(np.mean([r["downtime_flow_ticks"] for r in rows])),
                "mean_tokens_used": float(np.mean([r["tokens_used"] for r in rows])),
                "mean_inspections": float(np.mean([r["inspections"] for r in rows])),
                "episodes_with_foothold": len(incidents),
                "contained_footholds": len(contained),
                "mean_ticks_to_contain_when_contained": float(np.mean(contained)) if contained else None,
            })
    return summary, traces
