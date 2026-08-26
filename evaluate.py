"""
Evaluation harness.

Scores any defender policy over a set of episodes and reports mean return
alongside the metric that actually matters operationally, the fraction of
total network value lost to compromise over the episode.

Both baseline policies and Stable Baselines3 models go through the same
path, so the comparison is apples to apples. Episode seeds are shared
across policies, meaning every policy faces exactly the same sequence of
networks and the same attacker randomness draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from network_defense_env import NetworkDefenseEnv


class SB3PolicyWrapper:
    """Adapt a Stable Baselines3 model to the baseline policy interface."""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self._obs = None

    def reset(self, env):
        self._obs = env._get_obs()

    def act(self, env) -> int:
        obs = env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def run_episode(policy, env: NetworkDefenseEnv, seed: int, n_nodes: int | None = None):
    """Run one episode and return summary statistics."""
    options = {"n_nodes": n_nodes} if n_nodes is not None else None
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env)

    total_reward = 0.0
    compromised_steps = 0
    n_remediations = 0
    value_lost_trace = []

    done = False
    while not done:
        action = policy.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        compromised_steps += info["n_compromised"]
        n_remediations += int(info["remediated"])
        value_lost_trace.append(info["compromised_value"] / max(info["total_value"], 1e-9))
        done = terminated or truncated

    return {
        "policy": policy.name,
        "seed": seed,
        "n_nodes": info["n_active_nodes"],
        "return": total_reward,
        "value_lost_frac": float(np.mean(value_lost_trace)),
        "mean_compromised": compromised_steps / env.episode_length,
        "remediations": n_remediations,
        "cumulative_damage": info["cumulative_damage"],
    }


def evaluate_policies(
    policies,
    n_episodes: int = 200,
    seed_offset: int = 10_000,
    env_kwargs: dict | None = None,
    node_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Evaluate every policy on an identical set of episodes.

    ``node_range`` overrides the environment's node sampling, which is how
    the generalisation test evaluates on network sizes never seen during
    training.
    """
    env_kwargs = dict(env_kwargs or {})
    rows = []

    node_rng = np.random.default_rng(seed_offset)
    if node_range is not None:
        node_counts = node_rng.integers(node_range[0], node_range[1] + 1, n_episodes)
    else:
        node_counts = [None] * n_episodes

    for policy in policies:
        env = NetworkDefenseEnv(**env_kwargs)
        for i in range(n_episodes):
            rows.append(
                run_episode(policy, env, seed=seed_offset + i, n_nodes=node_counts[i])
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per episode results into a comparison table."""
    g = df.groupby("policy", sort=False)
    out = pd.DataFrame(
        {
            "mean_return": g["return"].mean(),
            "std_return": g["return"].std(),
            "sem_return": g["return"].sem(),
            "value_lost_frac": g["value_lost_frac"].mean(),
            "mean_compromised": g["mean_compromised"].mean(),
            "remediations": g["remediations"].mean(),
        }
    )
    return out.sort_values("mean_return", ascending=False)
