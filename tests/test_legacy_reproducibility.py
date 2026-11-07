import unittest

import numpy as np

from network_defense_env import NetworkDefenseEnv
from train import _RangeWrapper


class LegacyEnvironmentTests(unittest.TestCase):
    def test_invalid_action_is_rejected_without_advancing(self):
        env = NetworkDefenseEnv(max_nodes=20)
        env.reset(seed=1, options={"n_nodes": 20})
        for action in (-1, 20, 1.5):
            with self.assertRaisesRegex(ValueError, "declared space"):
                env.step(action)
        self.assertEqual(env.t, 0)
        self.assertFalse(env.defense_level.any())

    def test_padded_slots_stay_compatible_with_existing_checkpoints(self):
        env = NetworkDefenseEnv(max_nodes=20, attacker_aggressiveness=0)
        env.reset(seed=1, options={"n_nodes": 8})
        np.testing.assert_array_equal(env.action_masks(), np.arange(20) < 8)
        _, reward, _, _, info = env.step(19)
        self.assertEqual(reward, 0)
        self.assertFalse(info["defender_acted"])

    def test_wrapper_seeding_reproduces_network_sizes_and_observations(self):
        wrappers = [_RangeWrapper(NetworkDefenseEnv(), (8, 14)) for _ in range(2)]
        for i in range(12):
            a, info_a = wrappers[0].reset(seed=32 if i == 0 else None)
            b, info_b = wrappers[1].reset(seed=32 if i == 0 else None)
            self.assertEqual(info_a, info_b)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(8 <= info_a["n_active_nodes"] <= 14)


if __name__ == "__main__":
    unittest.main()
