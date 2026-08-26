# Trained checkpoints

| File | Trained on | Used in |
|---|---|---|
| `ppo_defender.zip` | 8 to 14 nodes, lambda 4 | Study 1, the agent that fails to generalise |
| `dqn_defender.zip` | 8 to 14 nodes, lambda 4 | Study 1 |
| `ppo_dr_defender.zip` | 8 to 20 nodes, lambda 4 | Study 1 fix, and the QR arm of study 2 |
| `ppo_lam_defender.zip` | 8 to 20 nodes, lambda log uniform [0.5, 16] | Study 2, never told lambda |
| `ppo_suqr_defender.zip` | 8 to 20 nodes, SUQR attacker | Study 2, structural misspecification |

`curve_*.csv` holds the episode returns logged during each run.

Checkpoints are committed so the evaluation scripts run without retraining.
Regenerate them with the commands in the root README.
