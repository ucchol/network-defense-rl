"""
Generate the figures used in the README and the notebook.

Every figure is written as a standalone PNG at 150 dpi so it can be
embedded directly in GitHub markdown.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "figures"
PALETTE = {
    "QSE": "#1b6ca8",
    "PPO": "#e07b39",
    "PPO-DR": "#2e8b57",
    "DQN": "#9467bd",
    "Greedy value": "#7f7f7f",
    "Value proportional": "#bcbd22",
    "Random": "#c0392b",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_learning_curves(modeldir="models", outdir=FIGDIR):
    """Smoothed training returns for each learned agent."""
    os.makedirs(outdir, exist_ok=True)
    files = {
        "PPO": "curve_ppo.csv",
        "PPO-DR": "curve_ppo_dr.csv",
        "DQN": "curve_dqn.csv",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, fname in files.items():
        path = os.path.join(modeldir, fname)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path).sort_values("timestep")
        smooth = d["return"].rolling(150, min_periods=20).mean()
        ax.plot(d["timestep"], smooth, label=label, color=PALETTE[label], linewidth=1.8)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (rolling mean)")
    ax.set_title("Training curves")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "learning_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_regime_comparison(summary_csv="results/summary.csv", outdir=FIGDIR):
    """Grouped bars comparing every policy across both evaluation regimes."""
    os.makedirs(outdir, exist_ok=True)
    s = pd.read_csv(summary_csv)
    order = (
        s[s.regime == "in_distribution"]
        .sort_values("mean_return", ascending=False)["policy"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.38
    x = np.arange(len(order))

    for k, (regime, hatch, alpha) in enumerate(
        [("in_distribution", None, 1.0), ("holdout", "//", 0.75)]
    ):
        sub = s[s.regime == regime].set_index("policy").reindex(order)
        ax.bar(
            x + (k - 0.5) * width,
            sub["mean_return"],
            width,
            yerr=sub["sem_return"],
            capsize=3,
            color=[PALETTE.get(p, "#888") for p in order],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white",
            label="8 to 14 nodes (trained)" if k == 0 else "15 to 20 nodes (held out)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("Mean episode return")
    ax.set_title("Policy performance, in distribution vs held out network sizes")
    ax.legend(frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "regime_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_generalisation_gap(summary_csv="results/summary.csv", outdir=FIGDIR):
    """Highlight how much each policy degrades on unseen network sizes."""
    os.makedirs(outdir, exist_ok=True)
    s = pd.read_csv(summary_csv)
    piv = s.pivot(index="policy", columns="regime", values="mean_return")
    piv["gap"] = piv["in_distribution"] - piv["holdout"]
    piv = piv.sort_values("gap")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = [PALETTE.get(p, "#888") for p in piv.index]
    ax.barh(piv.index, piv["gap"], color=colors, edgecolor="white")
    ax.set_xlabel("Return lost when moving to unseen network sizes")
    ax.set_title("Generalisation gap, smaller is better")
    for i, (p, v) in enumerate(zip(piv.index, piv["gap"])):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(outdir, "generalisation_gap.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_action_usage(csv="results/action_usage.csv", outdir=FIGDIR):
    """The diagnostic that explains the generalisation failure."""
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(d))
    ax.bar(x, d["share_unseen"] * 100, color=[PALETTE[a] for a in d["agent"]],
           edgecolor="white", width=0.55)
    ax.axhline(30, color="black", linestyle="--", linewidth=1.2)
    ax.text(len(d) - 0.5, 31.5, "expected if generalising", ha="right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["agent"])
    ax.set_ylabel("Share of actions on node slots 14 to 19 (%)")
    ax.set_title("Actions placed on node slots never active during training")
    for i, v in enumerate(d["share_unseen"] * 100):
        ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "action_usage.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_qse_allocation(outdir=FIGDIR):
    """Show the structure of the equilibrium coverage the QSE solver finds."""
    os.makedirs(outdir, exist_ok=True)
    from network_defense_env import NetworkDefenseEnv
    from baselines import solve_qse

    env = NetworkDefenseEnv(seed=0)
    _, info = env.reset(seed=0, options={"n_nodes": 14})
    n = info["n_active_nodes"]
    values = env.value[:n]
    budget = env.sustainable_defense_mass() / env.max_defense_level
    cov = solve_qse(values, budget, rationality=env.attacker_rationality)

    order = np.argsort(-values)
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    x = np.arange(n)
    ax.bar(x - 0.2, values[order], 0.4, label="Node value", color="#b0b0b0",
           edgecolor="white")
    ax.bar(x + 0.2, cov[order], 0.4, label="QSE coverage", color=PALETTE["QSE"],
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n{i}" for i in order], fontsize=8)
    ax.set_xlabel("Nodes, sorted by value")
    ax.set_title(
        f"Equilibrium coverage concentrates on high value nodes "
        f"(budget {budget:.2f} across {n} nodes)"
    )
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "qse_allocation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_all():
    outs = [
        fig_learning_curves(),
        fig_regime_comparison(),
        fig_generalisation_gap(),
        fig_action_usage(),
        fig_qse_allocation(),
        fig_misspecification(),
        fig_ablation(),
    ]
    for o in outs:
        print("wrote", o)
    return outs


if __name__ == "__main__":
    make_all()


def fig_misspecification(csv="results/misspecification.csv", outdir=FIGDIR):
    """Return against true attacker rationality, with QSE assuming lambda 4."""
    os.makedirs(outdir, exist_ok=True)
    s = pd.read_csv(csv)
    colors = {
        "QSE (assumed 4)": "#1b6ca8",
        "QSE (oracle)": "#5fa8d3",
        "PPO (trained 4)": "#e07b39",
        "PPO (lambda rand)": "#2e8b57",
        "Greedy value": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for pol, g in s.groupby("policy"):
        g = g.sort_values("true_lambda")
        ax.errorbar(g["true_lambda"], g["mean_return"], yerr=g["sem_return"],
                    marker="o", ms=4, capsize=3, linewidth=1.7,
                    label=pol, color=colors.get(pol, "#888"))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("True attacker rationality (log scale)")
    ax.set_ylabel("Mean episode return")
    ax.set_title("No crossover under attacker misspecification")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    out = os.path.join(outdir, "misspecification.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ablation(csv="results/ablation_episodes.csv", outdir=FIGDIR):
    """What the equilibrium solve actually contributes over naive fill."""
    os.makedirs(outdir, exist_ok=True)
    d = pd.read_csv(csv)
    g = d.groupby("policy", sort=False)["return"].agg(["mean", "sem"])
    g = g.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(g.index, g["mean"], xerr=g["sem"], capsize=4,
            color=["#7f7f7f", "#c8a15a", "#1b6ca8"], edgecolor="white")
    for i, (m, e) in enumerate(zip(g["mean"], g["sem"])):
        ax.text(m - 0.35, i, f"{m:.3f}", va="center", fontsize=9, color="white")
    ax.set_xlabel("Mean episode return")
    ax.set_title("Ablation, the equilibrium solve adds little over naive fill")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(outdir, "ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
