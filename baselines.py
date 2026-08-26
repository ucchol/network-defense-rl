"""
Baseline defender policies, including a Quantal Stackelberg Equilibrium
solver used as the game theoretic reference point.

Policy interface
----------------
Every policy exposes ``act(env) -> int`` and reads state directly off the
environment. Reading the environment rather than the flattened observation
keeps the baselines readable, and none of them see anything the learned
agents cannot recover from the observation vector.

The QSE baseline
----------------
The one shot game underneath each step is a standard Stackelberg security
game. The defender commits to a coverage vector c over n targets subject to
a budget, and a boundedly rational attacker quantal responds

    q_i(c) = softmax( lambda * v_i * (1 - c_i) )

The defender then maximises expected utility

    U_d(c) = - sum_i q_i(c) * v_i * (1 - c_i)

subject to sum_i c_i <= B and 0 <= c_i <= 1. We solve this with SLSQP from
several random restarts, since the objective is not concave in general.

Solving the one shot game gives a target coverage vector. To act in the
sequential environment we reinforce whichever node currently sits furthest
below its equilibrium target, which drives the realised coverage toward the
equilibrium allocation and keeps it there against decay.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Simple baselines
# ----------------------------------------------------------------------


class RandomPolicy:
    """Uniformly random action over active nodes."""

    name = "Random"

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env) -> int:
        active = np.where(env.active_mask == 1.0)[0]
        return int(self._rng.choice(active))


class GreedyValuePolicy:
    """Always act on the highest value node that needs attention.

    Compromised nodes are remediated first, ordered by value. Otherwise the
    highest value node below full coverage is reinforced. This is the
    obvious heuristic a practitioner would reach for, so beating it is the
    minimum bar for the learned agents.
    """

    name = "Greedy value"

    def reset(self, env):
        pass

    def act(self, env) -> int:
        active = np.where(env.active_mask == 1.0)[0]
        comp = active[env.compromised[active] == 1.0]
        if len(comp) > 0:
            return int(comp[np.argmax(env.value[comp])])

        room = active[env.defense_level[active] < env.max_defense_level]
        if len(room) == 0:
            return int(active[np.argmax(env.value[active])])
        return int(room[np.argmax(env.value[room])])


class ProportionalPolicy:
    """Spread effort in proportion to node value.

    Samples a node with probability proportional to value times remaining
    coverage gap. A reasonable stochastic allocation heuristic, and a
    sanity check that the QSE solver is doing something beyond value
    weighting.
    """

    name = "Value proportional"

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env) -> int:
        active = np.where(env.active_mask == 1.0)[0]
        gap = 1.0 - env.defense_level[active] / env.max_defense_level
        weight = env.value[active] * np.maximum(gap, 0.0)
        weight = weight + 1e-8 * env.compromised[active] * env.value[active]
        if weight.sum() <= 0:
            return int(self._rng.choice(active))
        return int(self._rng.choice(active, p=weight / weight.sum()))


# ----------------------------------------------------------------------
# Quantal Stackelberg Equilibrium
# ----------------------------------------------------------------------


def solve_qse(
    values: np.ndarray,
    budget: float,
    rationality: float = 4.0,
    n_restarts: int = 6,
    seed: int | None = 0,
) -> np.ndarray:
    """Solve for the defender's optimal coverage against a quantal attacker.

    Parameters
    ----------
    values
        Per node asset values, shape (n,).
    budget
        Total coverage the defender can sustain, in units where 1.0 is one
        fully covered node.
    rationality
        Attacker lambda. Zero is uniform random, large is best response.
    n_restarts
        Random restarts for the non convex program.

    Returns
    -------
    Coverage vector in [0, 1]^n summing to at most ``budget``.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    budget = float(np.clip(budget, 0.0, n))

    def attacker_probs(c):
        logits = rationality * values * (1.0 - c)
        logits = logits - logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def neg_defender_utility(c):
        q = attacker_probs(c)
        # Defender loses expected value on whatever the attacker hits
        return float(np.sum(q * values * (1.0 - c)))

    constraints = [{"type": "ineq", "fun": lambda c: budget - np.sum(c)}]
    bounds = [(0.0, 1.0)] * n

    rng = np.random.default_rng(seed)
    best_c, best_obj = None, np.inf

    # Start points, uniform spread plus value weighted plus random draws
    starts = [np.full(n, min(1.0, budget / n))]
    vw = values / values.sum() * budget
    starts.append(np.clip(vw, 0.0, 1.0))
    for _ in range(max(0, n_restarts - 2)):
        r = rng.random(n)
        starts.append(np.clip(r / r.sum() * budget, 0.0, 1.0))

    for c0 in starts:
        res = minimize(
            neg_defender_utility,
            c0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if res.success and res.fun < best_obj:
            best_obj, best_c = res.fun, res.x

    if best_c is None:
        best_c = np.clip(vw, 0.0, 1.0)
    return np.clip(best_c, 0.0, 1.0)


class QSEPolicy:
    """Track the Quantal Stackelberg Equilibrium coverage allocation.

    The equilibrium is solved once per episode from the node values and the
    sustainable budget, then followed in the sequential game by always
    correcting the largest shortfall against the target. Compromised nodes
    are remediated when the value lost to them exceeds what another
    reinforcement would buy.
    """

    name = "QSE"

    def __init__(self, rationality: float | None = None, seed: int | None = 0):
        self.rationality = rationality
        self.seed = seed
        self._target = None
        self._n = None

    def reset(self, env):
        self._target = None
        self._n = None

    def _ensure_target(self, env):
        n = env.n_active_nodes
        if self._target is not None and self._n == n:
            return
        values = env.value[:n]
        budget = env.sustainable_defense_mass() / env.max_defense_level
        lam = self.rationality if self.rationality is not None else env.attacker_rationality
        self._target = solve_qse(values, budget, rationality=lam, seed=self.seed)
        self._n = n

    def act(self, env) -> int:
        self._ensure_target(env)
        n = env.n_active_nodes

        coverage = env.defense_level[:n] / env.max_defense_level
        shortfall = self._target - coverage

        # Best available coverage correction and its marginal value. One
        # reinforcement raises coverage by 1 / max_defense_level, which cuts
        # the attacker's success probability on that node by
        # aggressiveness / max_defense_level, saving that much expected
        # damage per step going forward.
        j = int(np.argmax(shortfall))
        delta_cov = 1.0 / env.max_defense_level
        gain_reinforce = (
            env.damage_rate
            * env.value[j]
            * env.attacker_aggressiveness
            * delta_cov
            * max(float(shortfall[j]), 0.0)
        )

        # Remediation stops the full ongoing bleed on a compromised node.
        comp = np.where(env.compromised[:n] == 1.0)[0]
        if len(comp) > 0:
            i = int(comp[np.argmax(env.value[comp])])
            gain_remediate = env.damage_rate * env.value[i]
            if gain_remediate >= gain_reinforce:
                return i

        if shortfall[j] > 0:
            return j
        # Equilibrium coverage already met, top up the most valuable node
        # that still has headroom against decay.
        room = np.where(env.defense_level[:n] < env.max_defense_level)[0]
        if len(room) == 0:
            return int(np.argmax(env.value[:n]))
        return int(room[np.argmax(env.value[room])])


def get_baselines(seed: int | None = 0):
    """Return the standard baseline suite used in the evaluation."""
    return [
        RandomPolicy(seed=seed),
        ProportionalPolicy(seed=seed),
        GreedyValuePolicy(),
        QSEPolicy(seed=seed),
    ]
