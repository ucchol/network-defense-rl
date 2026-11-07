"""Policies and counterfactual explanations for the containment exercise."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .model import (
    CRITICAL, HOSTS, PRESENT, TICKS, VALUES, WAIT, Action, Controls, Regime,
    attack_edges, available_services, legal_moves, observation_branches, tick_cost, transition,
)


def value_first(belief, regime: Regime, controls: Controls, tick: int) -> Action:
    """One-node value heuristic adapted to the feasible containment actions."""
    probability = belief @ PRESENT
    best_action, best_score = WAIT, 0.
    for action, _ in legal_moves(regime, controls):
        if action.kind == "isolate":
            score = probability[action.host] * VALUES[action.host] / action.cost
        elif action.kind == "restrict":
            score = probability[action.host] * (1 - probability[action.destination])
            score *= VALUES[action.destination] / action.cost
        else:
            continue
        if score > best_score:
            best_action, best_score = action, score
    return best_action


class Planner:
    """One-step greedy or exact finite-horizon planning over noisy observations."""

    def __init__(self, regime: Regime, depth: int, *, adaptive: bool = True):
        if not 1 <= depth <= 3:
            raise ValueError("Planning depth must be between one and three")
        self.regime = regime
        self.depth = depth
        self.adaptive = adaptive
        self._best = lru_cache(maxsize=16384)(self._minimum)

    def _minimum(self, belief, controls: Controls, tick: int, depth: int) -> float:
        return min(cost for _, _, cost in self._costs(np.asarray(belief), controls, tick, depth))

    def _costs(self, belief, controls: Controls, tick: int, depth: int):
        predictions = {}
        for action, after in legal_moves(self.regime, controls):
            key = (after.isolated, after.blocked)
            if key not in predictions:
                predictions[key] = belief @ transition(
                    self.regime, Controls(0, after.isolated, after.blocked)
                )
            predicted = predictions[key]
            cost = tick_cost(predicted, action, after)
            if depth > 1:
                if self.adaptive:
                    for _, mass, posterior in observation_branches(
                        predicted, self.regime, tick + 1, action
                    ):
                        cost += mass * self._best(tuple(posterior), after, tick + 1, depth - 1)
                else:
                    cost += self._best(tuple(predicted), after, tick + 1, depth - 1)
            yield action, after, cost

    def rank(self, belief, controls: Controls, tick: int):
        depth = min(self.depth, TICKS - tick)
        if depth < 1:
            raise ValueError("Episode is already complete")
        return sorted(self._costs(belief, controls, tick, depth), key=lambda move: move[2])

    def act(self, belief, controls: Controls, tick: int) -> Action:
        return self.rank(belief, controls, tick)[0][0]


def route_count(regime: Regime, controls: Controls, start: int) -> int:
    """Count simple modeled paths from the named lead host to critical assets."""
    edges = attack_edges(regime, controls)
    count = 0
    pending = [(start, (start,))]
    while pending:
        node, path = pending.pop()
        if len(path) > 1 and node in (3, 4):
            count += 1
        pending.extend((dst, (*path, dst)) for src, dst in edges if src == node and dst not in path)
    return count


def recommendation(belief, regime: Regime, controls: Controls, tick: int = 0, depth: int = 2):
    ranked = Planner(regime, depth).rank(belief, controls, tick)
    wait_cost = next(cost for action, _, cost in ranked if action == WAIT)
    risk_ticks = min(depth, TICKS - tick)

    def risk_without_followup(after):
        predictions = belief @ np.linalg.matrix_power(transition(regime, after), risk_ticks)
        return float(predictions @ CRITICAL)

    wait_risk = risk_without_followup(controls)
    start = int(np.argmax(belief @ PRESENT))
    baseline_paths = route_count(regime, controls, start)
    rows = []
    for action, after, cost in ranked:
        risk = risk_without_followup(after)
        paths = route_count(regime, after, start)
        services = available_services(after)
        available = sum(services)
        if action.kind == "inspect":
            explanation = f"Gather a higher-quality observation of {HOSTS[action.host]}."
        elif action.kind in {"isolate", "restrict"}:
            removed = baseline_paths - paths
            unit = "path" if removed == 1 else "paths"
            explanation = f"Removes {removed} modeled {unit} from {HOSTS[start]} to critical hosts."
        else:
            explanation = "Preserve the response budget and observe the next alert."
        explanation += f" Keeps {available}/{len(services)} required service flows reachable."
        rows.append({
            "action": action.label,
            "tokens": action.cost,
            "tokens_remaining": after.budget,
            "expected_horizon_cost": round(cost, 6),
            "cost_reduction_vs_wait": round(wait_cost - cost, 6),
            "critical_risk_without_followup": round(risk, 6),
            "critical_risk_reduction_vs_wait_without_followup": round(wait_risk - risk, 6),
            "modeled_paths_before": baseline_paths,
            "modeled_paths_after": paths,
            "service_flows_available": available,
            "explanation": explanation,
        })
    return rows
