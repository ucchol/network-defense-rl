import unittest

import numpy as np

from containment.benchmark import run_benchmark, run_episode
from containment.model import (
    HOSTS, PRESENT, STATES, WAIT, Action, Controls, Regime, attack_edges,
    available_services, frontier, initial_belief, legal_moves, observation_branches,
    tick_cost, transition,
)
from containment.policies import Planner, recommendation


class ContainmentTests(unittest.TestCase):
    def test_graph_spread_is_local_and_cannot_jump_a_blocked_link(self):
        protected = Regime()
        state = 1 << HOSTS.index("workstation")
        self.assertEqual(frontier(state, attack_edges(protected, Controls())), (2,))
        after = dict(legal_moves(protected, Controls()))[Action("restrict", 0, 2)]
        np.testing.assert_array_equal(transition(protected, after)[state] > 0, STATES == state)
        self.assertIn(4, frontier(state, attack_edges(Regime(graph=False), Controls())))

    def test_action_filter_enforces_service_dependencies_and_budget(self):
        controls = Controls(budget=1)
        feasible = dict(legal_moves(Regime(), controls))
        self.assertNotIn(Action("isolate", 0), feasible)
        self.assertNotIn(Action("restrict", 2, 3), feasible)
        after = feasible[Action("restrict", 0, 2)]
        self.assertEqual(after.budget, 0)
        self.assertEqual(set(dict(legal_moves(Regime(), after))), {WAIT})
        relaxed = dict(legal_moves(Regime(protect=False), controls))
        self.assertEqual(available_services(relaxed[Action("restrict", 2, 3)]), (False, True))
        self.assertNotIn(Action("inspect", 0), dict(legal_moves(Regime(noisy=False), controls)))

    def test_posterior_matches_sensor_likelihood_and_conserves_mass(self):
        belief = initial_belief(True)
        self.assertAlmostEqual((belief @ PRESENT)[0], .5 * .76 / (.5 * .76 + .5 * .13))
        for stress in (False, True):
            matrix = transition(Regime(), Controls(), stress)
            np.testing.assert_allclose(matrix.sum(axis=1), 1.)
        predicted = belief @ transition(Regime(), Controls())
        branches = list(observation_branches(predicted, Regime(), 1, Action("inspect", 0)))
        self.assertEqual(len(branches), 4)
        self.assertAlmostEqual(sum(mass for _, mass, _ in branches), 1.)
        np.testing.assert_allclose(sum(mass * posterior for _, mass, posterior in branches), predicted)

    def test_planner_uses_contingent_followup_and_reports_model_limits(self):
        regime, controls = Regime(), Controls()
        belief = initial_belief(True)
        planner = Planner(regime, 2)
        moves = planner.rank(belief, controls, 0)
        action, after, expected = moves[0]
        self.assertEqual(action, Action("restrict", 0, 2))
        prediction = belief @ transition(regime, after)
        future = 0.
        for _, mass, posterior in observation_branches(prediction, regime, 1, action):
            future += mass * min(
                tick_cost(posterior @ transition(regime, next_controls), next_action, next_controls)
                for next_action, next_controls in legal_moves(regime, after)
            )
        self.assertAlmostEqual(expected, tick_cost(prediction, action, after) + future)
        report = recommendation(belief, regime, controls)
        self.assertEqual(report[0]["modeled_paths_before"] - report[0]["modeled_paths_after"], 3)
        self.assertEqual(report[0]["service_flows_available"], 2)
        self.assertGreater(report[0]["critical_risk_reduction_vs_wait_without_followup"], 0)

    def test_rollouts_are_reproducible_and_reject_infeasible_actions(self):
        regime = Regime()
        planner = Planner(regime, 2)
        row = run_episode(regime, planner.act, 8)
        self.assertEqual(row, run_episode(regime, planner.act, 8))
        self.assertEqual(row["downtime_flow_ticks"], 0)
        self.assertLessEqual(row["tokens_used"], 4)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            run_episode(regime, lambda *_: Action("isolate", 4), 8)

    def test_zero_budget_makes_all_policies_wait(self):
        summary, traces = run_benchmark(6, 10, "core", budget=0)
        self.assertEqual(len({r["mean_cost"] for r in summary}), 1)
        self.assertTrue(all(r["tokens_used"] == 0 for r in traces))

    def test_summary_uses_matched_seed_pairs(self):
        summary, traces = run_benchmark(5, 20, "core")
        self.assertEqual({r["policy"] for r in summary},
                         {"value_first", "greedy", "open_loop", "planner"})
        per_policy = {
            p: {r["seed"]: r["cost"] for r in traces if r["policy"] == p}
            for p in ("greedy", "planner")
        }
        self.assertEqual(per_policy["greedy"].keys(), per_policy["planner"].keys())
        saved = np.mean([per_policy["greedy"][i] - per_policy["planner"][i] for i in range(20, 25)])
        row = next(r for r in summary if r["policy"] == "planner")
        self.assertAlmostEqual(row["mean_cost_saved_vs_greedy"], saved)


if __name__ == "__main__":
    unittest.main()
