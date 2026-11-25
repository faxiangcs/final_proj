"""Plot the GRPO evidence chain from saved logs, validation, and test results."""

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "grpo_refinement_20261119"
WORK = ROOT / "experiments" / EXPERIMENT
OUTPUT = ROOT / "output" / EXPERIMENT
STAGES = ("reference", "dense", "loc_only")
LABELS = ("Original", "Shaped + Dx", "Shaped only")
COLORS = ("#576474", "#397aa3", "#136756")


def tied_reward_fractions(stage, seeds, train_steps):
    fractions = []
    for seed in seeds:
        steps = {}
        run = OUTPUT / f"{stage}_seed{seed}"
        logs = sorted(run.glob("v*/logging.jsonl"), key=lambda log: log.stat().st_mtime)
        for log in logs:
            for line in log.read_text().splitlines():
                row = json.loads(line)
                if "frac_reward_zero_std" in row:
                    step = int(row["global_step/max_steps"].split("/")[0])
                    steps[step] = row["frac_reward_zero_std"]
        if set(steps) != set(range(1, train_steps + 1)):
            raise ValueError(f"{run.name} is missing training logs")
        fractions.append(mean(steps.values()))
    return fractions


def tied_reward_fraction(stage, seeds, train_steps):
    return mean(tied_reward_fractions(stage, seeds, train_steps))


def evidence_chain(plan):
    """Separate the analytical mechanism from measured signal and validation."""
    selections = json.loads((WORK / "selection.json").read_text())
    fig, (reward, signal, validation) = plt.subplots(
        1, 3, figsize=(7.16, 2.15), gridspec_kw={"width_ratios": [1, 1.12, 1.16]}
    )
    fig.subplots_adjust(left=.065, right=.99, top=.83, bottom=.225, wspace=.82)
    for ax in (reward, signal, validation):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e1e5e8", linewidth=.5)

    # One prediction / one reference: hard reward is a threshold indicator,
    # and the quality term equals IoU. Split the line to show the discontinuity.
    below = [i / 1000 for i in range(500)]
    above = [i / 1000 for i in range(500, 1001)]
    reward.plot([0, .5, .5, 1], [0, 0, 1, 1], color=COLORS[0],
                linestyle="--", linewidth=1.1, label="Hard")
    reward.plot(below, [.5 * x for x in below], color=COLORS[2],
                linewidth=1.4, label="Shaped")
    reward.plot(above, [.5 + .5 * x for x in above], color=COLORS[2], linewidth=1.4)
    reward.plot([.5, .5], [.25, .75], color=COLORS[2], linewidth=.9, linestyle=":")
    reward.scatter([.10, .20, .30, .45], [.05, .10, .15, .225],
                   color=COLORS[2], s=10, zorder=3)
    reward.set(xlim=(-.025, 1.025), ylim=(-.04, 1.07), xlabel="Box IoU",
               ylabel="Localization reward", xticks=[0, .5, 1], yticks=[0, .5, 1])
    reward.set_title("(a) Reward mechanism", loc="left", pad=9)
    reward.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.5)

    signal.grid(False)
    signal.grid(axis="x", color="#e1e5e8", linewidth=.5)
    signal.spines["left"].set_visible(False)
    signal.set_yticks(range(3), LABELS)
    signal.set(xlim=(0, 18), ylim=(2.55, -.55), xlabel="Groups with tied rewards",
               xticks=[0, 5, 10, 15], xticklabels=["0%", "5%", "10%", "15%"])
    signal.tick_params(axis="y", length=0, labelsize=7)
    signal.set_title("(b) Training signal", loc="left", pad=9)

    for index, (stage, color, label) in enumerate(zip(STAGES, COLORS, LABELS)):
        rates = [100 * v for v in tied_reward_fractions(
            stage, plan["seeds"], plan["train_steps"])]
        signal.barh(index, mean(rates), height=.5, color=color, alpha=.45)
        signal.scatter(rates, [index - .12, index, index + .12],
                       s=13, color=color, edgecolors="white", linewidths=.35, zorder=3)
        signal.text(mean(rates) + .65, index, f"{mean(rates):.2f}%",
                    va="center", fontsize=7.2, color=color)

        runs = [row for row in selections if row["name"].startswith(stage + "_")]
        if len(runs) != len(plan["seeds"]):
            raise ValueError(f"Missing validation runs for {stage}")
        expected_steps = [0, *plan["checkpoints"]]
        for row in runs:
            if [c["step"] for c in row["candidates"]] != expected_steps:
                raise ValueError(f"Missing validation checkpoints for {row['name']}")
        values = [[c["score"][0] for c in row["candidates"]] for row in runs]
        columns = list(zip(*values))
        validation.plot(expected_steps, [mean(c) for c in columns],
                        color=color, linewidth=1.1, marker="o", markersize=2.8, label=label)
        validation.fill_between(expected_steps, [min(c) for c in columns],
                                [max(c) for c in columns], color=color, alpha=.10, linewidth=0)
    validation.set_title("(c) Validation behavior", loc="left", pad=9)
    validation.set(xlabel="GRPO step", ylabel="Localization F1@0.5",
                   xticks=[0, 100, 175, 266], ylim=(.36, .525), yticks=[.375, .425, .475, .525])
    validation.legend(loc="lower right", frameon=False, fontsize=6.4,
                      handlelength=1.4, labelspacing=.25)
    fig.savefig(ROOT / "paper" / "figures" / "fig_evidence_chain.pdf")
    plt.close(fig)


def main():
    plan = json.loads((WORK / "plan.json").read_text())
    aggregate = json.loads((WORK / "aggregate_results.json").read_text())
    plt.rcParams.update({
        "pdf.fonttype": 42, "font.family": "serif",
        "font.serif": ["Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8,
        "axes.linewidth": 0.6, "xtick.labelsize": 7.3, "ytick.labelsize": 7.5,
    })
    evidence_chain(plan)
    fig, (signal, score) = plt.subplots(2, 1, figsize=(3.45, 2.75))
    fig.subplots_adjust(left=.285, right=.97, top=.92, bottom=.155, hspace=.85)
    for ax in (signal, score):
        ax.set_yticks([0, 1, 2], LABELS)
        ax.set_ylim(2.55, -.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=2.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#dee3e7", linewidth=.55)

    signal.set_title("(a) Training groups with tied rewards", loc="left", pad=5)
    signal.set_xlim(0, 18)
    signal.set_xticks([0, 5, 10, 15], ["0%", "5%", "10%", "15%"])
    score.set_title("(b) Test localization F1 change", loc="left", pad=5)
    score.set_xlim(-.09, .105)
    score.set_xticks([-.08, -.04, 0, .04, .08], ["−.08", "−.04", "0", "+.04", "+.08"])
    score.axvline(0, color="#566474", linewidth=.7, linestyle="--")
    score.set_xlabel("Difference versus SFT-201")

    for index, (stage, color) in enumerate(zip(STAGES, COLORS)):
        tied = 100 * tied_reward_fraction(stage, plan["seeds"], plan["train_steps"])
        signal.barh(index, tied, height=.58, color=color)
        signal.text(tied + .26, index, f"{tied:.2f}%", va="center", color=color)
        summary = aggregate[f"{stage}_selected"]
        delta = summary["mean_delta_raw"]
        low, high = summary["paired_seed_case_ci95_raw"]
        score.errorbar(delta, index, xerr=[[delta - low], [high - delta]],
                       fmt="o", color=color, markersize=4, capsize=2.2, linewidth=1.1)
        print(f"{stage}: tied {tied:.2f}%, Δ {delta:+.4f}, CI [{low:+.4f}, {high:+.4f}]")

    fig.savefig(ROOT / "paper" / "figures" / "fig_refinement.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
