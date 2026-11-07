"""Plot the two-step planner's paired cost differences from a benchmark CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_results(source: Path, output: Path):
    with source.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["policy"] == "planner"]
    if not rows:
        raise ValueError("No planner rows in the benchmark CSV")
    labels = []
    for row in rows:
        label = row["regime"].replace("-full-", "-full state-").replace("-", " · ")
        if row["stress"] == "True":
            label += " · faster spread"
        labels.append(label[0].upper() + label[1:])
    contexts = {(row["episodes"], row["budget_tokens"]) for row in rows}
    if len(contexts) != 1:
        raise ValueError("Use a benchmark CSV with one episode count and budget for this plot")
    episodes, budget = next(iter(contexts))

    plt.rcParams.update({"svg.hashsalt": "ndrl-containment", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.6, max(4.4, len(rows) * .37 + 2.1)), facecolor="#fafbfd")
    fig.subplots_adjust(left=.42, right=.94, top=.80, bottom=.18)
    ax.set_facecolor("#fafbfd")
    ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
    for i, row in enumerate(rows):
        mean, lo, hi = (
            float(row[key]) for key in (
                "mean_cost_saved_vs_greedy", "paired_cost_saved_ci_low", "paired_cost_saved_ci_high"
            )
        )
        color = "#1e6b59" if lo > 0 else "#526581"
        ax.hlines(i, lo, hi, color=color, linewidth=2)
        ax.scatter(mean, i, color=color, s=32, zorder=3)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0")
    ax.tick_params(axis="both", length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Mean episode cost saved versus one-step greedy", labelpad=12)
    fig.text(.05, .93, "When does two-step planning help?", fontsize=19, weight="bold", color="#182b43")
    fig.text(.05, .875, f"Synthetic containment benchmark · {episodes} paired seeds · budget {budget} tokens",
             fontsize=11, color="#425675")
    fig.text(.05, .06, "Points show the mean paired difference. Lines show 95% bootstrap intervals.",
             fontsize=10, color="#425675")
    svg = output.suffix.lower() == ".svg"
    fig.savefig(output, metadata={"Date": None} if svg else None)
    plt.close(fig)
    if svg:
        lines = output.read_text(encoding="utf-8").splitlines()
        output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_results(args.source, args.output)


if __name__ == "__main__":
    main()
