"""
Train PPO and DQN defenders on NetworkDefenseEnv.

Agents train only on small networks (8 to 14 nodes). Evaluation later
includes larger networks (15 to 20 nodes) that were never seen during
training, which tests whether the padded observation plus active mask
design actually produces size generalisation rather than memorisation of
one network scale.
"""

from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from network_defense_env import NetworkDefenseEnv

TRAIN_NODE_RANGE = (8, 14)
HOLDOUT_NODE_RANGE = (15, 20)


class RewardLogger(BaseCallback):
    """Record episode returns during training for the learning curves."""

    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.returns = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.timesteps.append(self.num_timesteps)
                self.returns.append(ep["r"])
        return True


def make_train_env(n_envs: int = 8, seed: int = 0, node_range=TRAIN_NODE_RANGE, env_kwargs=None):
    """Vectorised training env.

    ``node_range`` controls which network sizes are sampled during
    training. The padded observation width stays at 20 regardless, so a
    policy trained on any range can be loaded and run on any other.
    """

    def _factory(rank: int):
        def _init():
            env = NetworkDefenseEnv(
                min_nodes=node_range[0],
                max_nodes=20,
                seed=seed + rank,
                **(env_kwargs or {}),
            )
            return Monitor(_RangeWrapper(env, node_range))
        return _init

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


class _RangeWrapper(gym.Wrapper):
    """Restrict episode node counts to a range without changing obs width.

    The padded observation stays 20 nodes wide regardless of the range, so
    a policy trained here can be evaluated on larger networks unchanged.
    """

    def __init__(self, env: NetworkDefenseEnv, node_range):
        super().__init__(env)
        self.node_range = node_range
        self._rng = np.random.default_rng()

    def reset(self, **kwargs):
        n = int(self._rng.integers(self.node_range[0], self.node_range[1] + 1))
        options = dict(kwargs.pop("options", None) or {})
        options["n_nodes"] = n
        return self.env.reset(options=options, **kwargs)


def train_ppo(
    total_timesteps: int = 700_000,
    seed: int = 0,
    outdir: str = "models",
    node_range=TRAIN_NODE_RANGE,
    tag: str = "ppo",
    env_kwargs=None,
):
    """Train the PPO defender and save its model and learning curve."""
    os.makedirs(outdir, exist_ok=True)
    env = make_train_env(n_envs=8, seed=seed, node_range=node_range, env_kwargs=env_kwargs)
    log = RewardLogger()
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=256,
        batch_size=512,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=total_timesteps, callback=log)
    model.save(os.path.join(outdir, f"{tag}_defender"))
    df = pd.DataFrame(
        {"algo": tag.upper(), "timestep": log.timesteps, "return": log.returns}
    )
    df.to_csv(os.path.join(outdir, f"curve_{tag}.csv"), index=False)
    return model, df


def train_dqn(total_timesteps: int = 150_000, seed: int = 0, outdir: str = "models"):
    """Train the DQN defender and save its model and learning curve.

    DQN runs on a single environment with a reduced update frequency. Per
    step it is far more expensive than PPO because every environment step
    can trigger a gradient update, so the step budget is smaller and
    train_freq is raised to keep wall clock time reasonable.
    """
    os.makedirs(outdir, exist_ok=True)
    env = make_train_env(n_envs=1, seed=seed)
    log = RewardLogger()
    model = DQN(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        train_freq=8,
        gradient_steps=2,
        target_update_interval=1_000,
        exploration_fraction=0.3,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=total_timesteps, callback=log)
    model.save(os.path.join(outdir, "dqn_defender"))
    df = pd.DataFrame(
        {"algo": "DQN", "timestep": log.timesteps, "return": log.returns}
    )
    df.to_csv(os.path.join(outdir, "curve_dqn.csv"), index=False)
    return model, df


def train_agents(total_timesteps: int = 400_000, seed: int = 0, outdir: str = "models"):
    """Train both agents sequentially and write the combined curve file."""
    _, ppo_df = train_ppo(total_timesteps, seed, outdir)
    _, dqn_df = train_dqn(min(total_timesteps, 150_000), seed, outdir)
    curve_df = pd.concat([ppo_df, dqn_df], ignore_index=True)
    curve_df.to_csv(os.path.join(outdir, "training_curves.csv"), index=False)
    return curve_df


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which == "ppo":
        train_ppo()
    elif which == "dqn":
        train_dqn()
    else:
        train_agents()
