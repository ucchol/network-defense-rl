"""
Run the full policy comparison and write result CSVs.

Two evaluation regimes
----------------------
In distribution, 8 to 14 nodes, the range the agents trained on.
Held out, 15 to 20 nodes, network sizes never seen during training.

Every policy sees identical episode seeds and identical node counts, so
differences in return come from the policies rather than from luck in the
draw.
"""

from __future__ import annotations

import os
import pandas as pd
from stable_baselines3 import PPO, DQN

from baselines import get_baselines
from evaluate import evaluate_policies, summarise, SB3PolicyWrapper

TRAIN_RANGE = (8, 14)
HOLDOUT_RANGE = (15, 20)
N_EPISODES = 300


def load_policies(modeldir: str = "models"):
    """Baselines plus every trained agent.

    PPO and DQN are trained on 8 to 14 nodes only. PPO-DR is the same PPO
    configuration trained with domain randomisation across the full 8 to 20
    range, which is the fix for the generalisation failure documented in
    the README.
    """
    policies = get_baselines(seed=0)
    policies.append(SB3PolicyWrapper(PPO.load(os.path.join(modeldir, "ppo_defender")), "PPO"))
    policies.append(SB3PolicyWrapper(DQN.load(os.path.join(modeldir, "dqn_defender")), "DQN"))
    dr_path = os.path.join(modeldir, "ppo_dr_defender.zip")
    if os.path.exists(dr_path):
        policies.append(SB3PolicyWrapper(PPO.load(dr_path[:-4]), "PPO-DR"))
    return policies


def main(outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)
    frames = []

    for label, rng in [("in_distribution", TRAIN_RANGE), ("holdout", HOLDOUT_RANGE)]:
        policies = load_policies()
        df = evaluate_policies(
            policies,
            n_episodes=N_EPISODES,
            seed_offset=10_000,
            node_range=rng,
        )
        df["regime"] = label
        frames.append(df)

        print(f"\n=== {label}  nodes {rng[0]} to {rng[1]} ===")
        print(summarise(df).round(4).to_string())

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(outdir, "episode_results.csv"), index=False)

    summary = (
        all_df.groupby(["regime", "policy"], sort=False)
        .agg(
            mean_return=("return", "mean"),
            sem_return=("return", "sem"),
            value_lost_frac=("value_lost_frac", "mean"),
            mean_compromised=("mean_compromised", "mean"),
            remediations=("remediations", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    return all_df, summary


if __name__ == "__main__":
    main()
