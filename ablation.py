"""
Ablation, what does the equilibrium solve actually contribute?

QSE outperforms the greedy heuristic by a wide margin in the main study. Two
things could be responsible. Either the equilibrium allocation itself, or the
policy structure wrapped around it, which tracks a coverage target and
remediates on a marginal utility threshold rather than acting greedily.

This script separates them. TopKPolicy keeps the tracking and remediation
logic identical and replaces only the coverage target, filling nodes in
descending value order until the budget runs out. Any remaining gap between
TopKPolicy and QSEPolicy is what solving the game is worth.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

from network_defense_env import NetworkDefenseEnv
from baselines import QSEPolicy, GreedyValuePolicy
from evaluate import run_episode

N_EPISODES = 300
NODE_RANGE = (8, 20)


class TopKPolicy(QSEPolicy):
    """QSE policy structure with a naive value ordered coverage target."""

    name = "Top-k fill (ablation)"

    def _ensure_target(self, env):
        n = env.n_active_nodes
        if self._target is not None and self._n == n:
            return
        values = env.value[:n]
        budget = env.sustainable_defense_mass() / env.max_defense_level
        target = np.zeros(n)
        remaining = budget
        for i in np.argsort(-values):
            take = min(1.0, remaining)
            target[i] = take
            remaining -= take
            if remaining <= 0:
                break
        self._target = target
        self._n = n


def main(outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(777)
    node_counts = rng.integers(NODE_RANGE[0], NODE_RANGE[1] + 1, N_EPISODES)

    qse = QSEPolicy(rationality=4.0, seed=0)
    qse.name = "QSE"

    rows = []
    for policy in [GreedyValuePolicy(), TopKPolicy(seed=0), qse]:
        env = NetworkDefenseEnv(attacker_rationality=4.0)
        for i in range(N_EPISODES):
            rows.append(
                run_episode(policy, env, seed=70_000 + i, n_nodes=int(node_counts[i]))
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "ablation_episodes.csv"), index=False)

    summary = df.groupby("policy", sort=False).agg(
        mean_return=("return", "mean"),
        sem_return=("return", "sem"),
        value_lost_frac=("value_lost_frac", "mean"),
    )
    summary.to_csv(os.path.join(outdir, "ablation.csv"))
    print(summary.round(4).to_string())

    a = df[df.policy == "QSE"].sort_values("seed")["return"].values
    b = df[df.policy == TopKPolicy.name].sort_values("seed")["return"].values
    t, p = stats.ttest_rel(a, b)
    print(
        f"\nQSE vs top-k fill, paired over {len(a)} shared seeds: "
        f"mean difference {(a - b).mean():+.4f}, p = {p:.4g}"
    )
    return df, summary


if __name__ == "__main__":
    main()
