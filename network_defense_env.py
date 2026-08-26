"""
NetworkDefenseEnv, a Gymnasium environment for defender vs attacker
resource allocation on a network with a variable node count per episode.

Key design decisions
--------------------
Variable node count. The number of nodes is resampled every episode
between min_nodes and max_nodes. Observations and actions are padded to a
fixed max_nodes so the policy network keeps a constant input and output
shape, and an active mask marks which slots are real. This is the standard
padding plus masking pattern for variable sized RL problems, and it lets us
train on one range of network sizes and evaluate on unseen sizes.

Binding budget. Defense decays every step, so the defender cannot simply
reinforce everything and coast. Sustainable total defense mass is roughly
1 / defense_decay, which is set below 3 * max_nodes on purpose. Allocation
therefore stays a live decision at every node count in the training range.
Decay also has a natural reading in the deception literature, where a
defensive measure loses value once an attacker fingerprints it.

Prevention vs remediation. Reinforcing a clean node raises its defense
level. Reinforcing a compromised node cleans it instead. The defender
therefore trades off hardening healthy assets against recovering lost ones,
which is the decision that makes the policy interesting to inspect.

Dense damage signal. Compromised nodes bleed value every step rather than
incurring a single penalty at the moment of compromise. This gives a dense
reward and makes remediation urgent.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NetworkDefenseEnv(gym.Env):
    """Sequential network defense game with a boundedly rational attacker."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_nodes: int = 20,
        min_nodes: int = 8,
        max_defense_level: float = 3.0,
        episode_length: int = 50,
        attacker_aggressiveness: float = 0.6,
        attacker_rationality: float = 4.0,
        rationality_range: tuple | None = None,
        attacker_model: str = "qr",
        suqr_weights: tuple = (1.0, -1.5),
        defense_decay: float = 0.08,
        damage_rate: float = 0.15,
        reinforce_cost: float = 0.01,
        seed: int | None = None,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.max_defense_level = max_defense_level
        self.episode_length = episode_length
        self.attacker_aggressiveness = attacker_aggressiveness
        self.attacker_rationality = attacker_rationality
        # When set, the attacker's rationality is resampled every episode
        # from this range. Used to train defenders that cannot rely on a
        # single known attacker model, which is the misspecification
        # experiment in results/misspecification.csv.
        self.rationality_range = rationality_range
        # "qr"   multiplicative utility v * (1 - coverage), the model the
        #        QSE solver assumes.
        # "suqr" subjective utility quantal response, an additive form
        #        w_v * v + w_c * coverage in the spirit of Nguyen et al.
        #        2013. Structurally different from what QSE assumes, so it
        #        misspecifies the solver's model rather than just its
        #        lambda parameter.
        self.attacker_model = attacker_model
        self.suqr_weights = suqr_weights
        self._nominal_rationality = attacker_rationality
        self.defense_decay = defense_decay
        self.damage_rate = damage_rate
        self.reinforce_cost = reinforce_cost

        # Per node features, [compromised, value, normalised defense, active]
        self.n_features = 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_nodes * self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_nodes)

        self._rng = np.random.default_rng(seed)
        self.n_active_nodes = None
        self.compromised = None
        self.value = None
        self.defense_level = None
        self.active_mask = None
        self.t = 0
        self._cumulative_damage = 0.0

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.max_nodes, self.n_features), dtype=np.float32)
        obs[:, 0] = self.compromised
        obs[:, 1] = self.value
        obs[:, 2] = self.defense_level / self.max_defense_level
        obs[:, 3] = self.active_mask
        return obs.flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        forced_n = None
        if options is not None:
            forced_n = options.get("n_nodes")

        if forced_n is not None:
            self.n_active_nodes = int(np.clip(forced_n, 1, self.max_nodes))
        else:
            self.n_active_nodes = int(
                self._rng.integers(self.min_nodes, self.max_nodes + 1)
            )

        if self.rationality_range is not None:
            lo, hi = self.rationality_range
            self.attacker_rationality = float(
                np.exp(self._rng.uniform(np.log(max(lo, 1e-3)), np.log(hi)))
            )

        self.compromised = np.zeros(self.max_nodes, dtype=np.float32)
        self.value = np.zeros(self.max_nodes, dtype=np.float32)
        self.defense_level = np.zeros(self.max_nodes, dtype=np.float32)
        self.active_mask = np.zeros(self.max_nodes, dtype=np.float32)

        self.active_mask[: self.n_active_nodes] = 1.0
        self.value[: self.n_active_nodes] = self._rng.uniform(
            0.1, 1.0, self.n_active_nodes
        )

        self.t = 0
        self._cumulative_damage = 0.0
        return self._get_obs(), {
            "n_active_nodes": self.n_active_nodes,
            "attacker_rationality": self.attacker_rationality,
        }

    def step(self, action):
        self.t += 1
        reward = 0.0
        action = int(action)

        # ---- Defender move -------------------------------------------
        remediated = False
        acted = False
        if action < self.max_nodes and self.active_mask[action] == 1.0:
            acted = True
            if self.compromised[action] == 1.0:
                self.compromised[action] = 0.0
                remediated = True
            else:
                self.defense_level[action] = min(
                    self.defense_level[action] + 1.0, self.max_defense_level
                )
            reward -= self.reinforce_cost

        # ---- Attacker move -------------------------------------------
        target, success = self._attacker_move()

        # ---- Ongoing damage from every compromised node --------------
        active = self.active_mask == 1.0
        damage = self.damage_rate * float(
            (self.value[active] * self.compromised[active]).sum()
        )
        reward -= damage
        self._cumulative_damage += damage

        # ---- Defense decay -------------------------------------------
        self.defense_level[active] = np.maximum(
            self.defense_level[active] - self.defense_decay, 0.0
        )

        truncated = self.t >= self.episode_length
        info = {
            "n_active_nodes": self.n_active_nodes,
            "n_compromised": int(self.compromised[active].sum()),
            "compromised_value": float(
                (self.value[active] * self.compromised[active]).sum()
            ),
            "total_value": float(self.value[active].sum()),
            "attacker_target": target,
            "attacker_success": success,
            "defender_acted": acted,
            "remediated": remediated,
            "cumulative_damage": self._cumulative_damage,
            "mean_defense": float(self.defense_level[active].mean()),
        }
        return self._get_obs(), float(reward), False, truncated, info

    # ------------------------------------------------------------------
    # Attacker model
    # ------------------------------------------------------------------

    def _attacker_move(self):
        """Boundedly rational attacker.

        The attacker scores each clean, active node by the value it would
        gain discounted by current coverage, then quantal responds with a
        softmax over those scores. attacker_rationality is the lambda
        parameter, where 0 is uniform random and large values approach a
        pure best response. This mirrors the quantal response model used in
        the Stackelberg security game literature and keeps the scripted
        attacker consistent with the QSE baseline in baselines.py.
        """
        active_idx = np.where(self.active_mask == 1.0)[0]
        clean = active_idx[self.compromised[active_idx] == 0.0]
        if len(clean) == 0:
            return None, False

        coverage = self.defense_level[clean] / self.max_defense_level
        if self.attacker_model == "suqr":
            w_v, w_c = self.suqr_weights
            utility = w_v * self.value[clean] + w_c * coverage
        else:
            utility = self.value[clean] * (1.0 - coverage)

        logits = self.attacker_rationality * utility
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()

        target = int(self._rng.choice(clean, p=probs))

        success_prob = self.attacker_aggressiveness * (
            1.0 - self.defense_level[target] / self.max_defense_level
        )
        success = bool(self._rng.random() < success_prob)
        if success:
            self.compromised[target] = 1.0
        return target, success

    # ------------------------------------------------------------------

    def sustainable_defense_mass(self) -> float:
        """Total defense level the defender can hold at steady state.

        One reinforcement arrives per step and every active node decays by
        defense_decay per step, so the defender can sustain about
        1 / defense_decay total levels. Reported so the notebook can show
        the allocation budget is genuinely binding.
        """
        return 1.0 / self.defense_decay

    def render(self):
        active = self.active_mask == 1.0
        print(
            f"t={self.t} nodes={self.n_active_nodes} "
            f"compromised={int(self.compromised[active].sum())} "
            f"mean_defense={self.defense_level[active].mean():.2f}"
        )
