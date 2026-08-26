"""
Attacker misspecification sweep.

The question
------------
QSE needs the attacker's rationality parameter lambda as an input. A
reinforcement learning defender never sees lambda at all, it only sees
reward. So there should be a crossover. When lambda is known or nearly
known, solving the equilibrium exactly should win. Once the assumed lambda
is far enough from the truth, a policy learned across a range of attackers
should win instead.

The arms
--------
QSE (assumed 4)   Solves every episode assuming lambda = 4, whatever the
                  true attacker is. This is the misspecified equilibrium.
QSE (oracle)      Solves with the true lambda. Not deployable, since it
                  requires knowing the attacker exactly. Included as the
                  ceiling that misspecification is measured against.
PPO (trained 4)   Trained against lambda = 4 only. The misspecified
                  learner, matched to QSE (assumed 4).
PPO (lambda rand) Trained with lambda resampled log uniformly over
                  [0.5, 16] every episode. Never told lambda at test time,
                  so it plays one policy against all attackers.
Greedy value      Model free heuristic reference that uses no attacker
                  model whatsoever.

All arms use the 8 to 20 node range and the domain randomised PPO
checkpoints, so the node size generalisation effect documented in the
README does not confound the lambda story.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from network_defense_env import NetworkDefenseEnv
from baselines import GreedyValuePolicy, QSEPolicy
from evaluate import run_episode, SB3PolicyWrapper

LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
ASSUMED_LAMBDA = 4.0
N_EPISODES = 150
NODE_RANGE = (8, 20)


class OracleQSEPolicy(QSEPolicy):
    """QSE that is handed the true attacker rationality each episode."""

    name = "QSE (oracle)"

    def __init__(self, seed=0):
        super().__init__(rationality=None, seed=seed)


def build_arms(modeldir="models"):
    qse_fixed = QSEPolicy(rationality=ASSUMED_LAMBDA, seed=0)
    qse_fixed.name = f"QSE (assumed {ASSUMED_LAMBDA:g})"

    arms = [
        GreedyValuePolicy(),
        qse_fixed,
        OracleQSEPolicy(seed=0),
        SB3PolicyWrapper(
            PPO.load(os.path.join(modeldir, "ppo_dr_defender")), "PPO (trained 4)"
        ),
        SB3PolicyWrapper(
            PPO.load(os.path.join(modeldir, "ppo_lam_defender")), "PPO (lambda rand)"
        ),
    ]
    return arms


def main(outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    node_rng = np.random.default_rng(777)
    node_counts = node_rng.integers(NODE_RANGE[0], NODE_RANGE[1] + 1, N_EPISODES)

    for lam in LAMBDAS:
        for policy in build_arms():
            env = NetworkDefenseEnv(attacker_rationality=lam)
            for i in range(N_EPISODES):
                r = run_episode(
                    policy, env, seed=50_000 + i, n_nodes=int(node_counts[i])
                )
                r["true_lambda"] = lam
                rows.append(r)
        print(f"  lambda {lam:g} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "misspecification_episodes.csv"), index=False)

    summary = (
        df.groupby(["true_lambda", "policy"], sort=False)
        .agg(
            mean_return=("return", "mean"),
            sem_return=("return", "sem"),
            value_lost_frac=("value_lost_frac", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(outdir, "misspecification.csv"), index=False)

    piv = summary.pivot(index="true_lambda", columns="policy", values="mean_return")
    print()
    print("Mean return by true attacker rationality")
    print(piv.round(3).to_string())
    return df, summary


if __name__ == "__main__":
    main()
