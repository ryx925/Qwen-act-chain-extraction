"""Export actual logged rewards; no interpolation or synthesized measurements."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from trip_chain.training.grpo_rewards import REWARD_NAMES


def reward_rows(run_dir: Path) -> list[dict]:
    source = run_dir / "progress.jsonl"
    if source.exists():
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A running writer may not have finished its final line.
    else:
        state = json.loads((run_dir / "trainer_state.json").read_text(encoding="utf-8"))
        rows = state["log_history"]
    by_step = {int(row["step"]): row for row in rows if "reward" in row}
    return [by_step[step] for step in sorted(by_step)]


def rolling(values: list[float]) -> list[float]:
    return [sum(values[max(0, i - 2):i + 1]) / len(values[max(0, i - 2):i + 1])
            for i in range(len(values))]


def plot_comparison(baseline: Path, experiment: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    a, b = reward_rows(baseline), reward_rows(experiment)
    if not a or not b:
        raise ValueError("Both runs need at least one actual reward log")
    manifests = [json.loads((p / "run_manifest.json").read_text(encoding="utf-8"))
                 for p in (baseline, experiment)]
    same_data = (manifests[0]["run_contract"]["data_sha256"]
                 == manifests[1]["run_contract"]["data_sha256"])
    smoke = manifests[1]["smoke_test"]
    output = experiment / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["step", "epoch", "reward", "reward_std", "kl", "loss", "grad_norm",
              "learning_rate", "frac_reward_zero_std", "completions/clipped_ratio"]
    fields += [f"rewards/reward_{name}/mean" for name in REWARD_NAMES]
    for name, rows in (("baseline_a", a), ("experiment_b", b)):
        with (output / f"{name}_rewards.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 180})
    status = "SMOKE / SUBSET: functionality only" if smoke else "Training rewards; all passages"
    if not same_data and not smoke:
        status = "Different training data: inspect comparison_summary.json"
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.85, bottom=0.10, hspace=0.48)
    for rows, label, color in ((a, "A: 2 candidates", "#718096"),
                               (b, "B: 6 candidates", "#C66A24")):
        steps, values = [r["step"] for r in rows], [r["reward"] for r in rows]
        axes[0].plot(steps, values, marker="o", markersize=3, lw=0.9, alpha=0.45,
                     color=color, label=label + " (raw)")
        axes[0].plot(steps, rolling(values), marker="o", markersize=3, lw=1.7,
                     color=color, label=label + " (3-step mean)")
        axes[1].plot(steps, [r["frac_reward_zero_std"] for r in rows], marker="o",
                     markersize=3, color=color, label=label)
    axes[0].set(ylabel="Mean total reward", xlabel="Optimizer step")
    all_rewards = [r["reward"] for r in a + b]
    low, high = min(all_rewards), max(all_rewards)
    span = max(0.3, high - low)
    axes[0].set_ylim(low - 0.10 * span, high + 0.35 * span)
    axes[1].set(ylabel="Zero-reward-variance group fraction", xlabel="Optimizer step",
                ylim=(-0.05, 1.05))
    axes[0].axhline(0, lw=0.7, color="#A0AEC0", linestyle="--")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("GRPO training reward: historical A vs experiment B", fontsize=13, y=0.97)
    fig.text(0.12, 0.91, status, fontsize=10, color="#9B3F25" if smoke else "#4A5568")
    fig.text(0.12, 0.025, "Six task weights = 1. KL beta = 0.04 is separate from plotted rewards.",
             fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"reward_comparison.{suffix}", facecolor="white")
    plt.close(fig)
    # B's own raw reward curve, suitable for short or full training.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    steps, values = [r["step"] for r in b], [r["reward"] for r in b]
    ax.plot(steps, values, "o-", color="#A4AAB0", lw=1, label="Raw step mean")
    ax.plot(steps, rolling(values), "o-", color="#2468A2", lw=1.8, label="Trailing 3-step mean")
    ax.set(xlabel="Optimizer step", ylabel="Mean total reward",
           title="B: six candidates, serial generation" + (" (SMOKE)" if smoke else ""))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(steps) == 1:
        ax.set_xlim(steps[0] - 0.5, steps[0] + 0.5)
        ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"reward_curve.{suffix}", facecolor="white")
    plt.close(fig)
    summary = {"baseline": str(baseline), "experiment": str(experiment),
               "same_training_data": same_data, "b_is_smoke": smoke,
               "interpretation": "smoke only; no trend inference" if smoke else
               "training reward comparison only; no held-out evaluation",
               "a_steps": len(a), "b_steps": len(b),
               "a_mean_reward": sum(r["reward"] for r in a) / len(a),
               "b_mean_reward": sum(r["reward"] for r in b) / len(b),
               "b_first_reward": b[0]["reward"], "b_last_reward": b[-1]["reward"],
               "b_first3_mean": sum(r["reward"] for r in b[:3]) / len(b[:3]),
               "b_last3_mean": sum(r["reward"] for r in b[-3:]) / len(b[-3:])}
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
