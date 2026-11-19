import json, glob, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.4,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "lines.markersize": 3.2, "legend.frameon": False,
})
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
C_BLUE, C_ORANGE, C_GREEN, C_GRAY = "#1f5fa8", "#d2691e", "#2e8b57", "#777777"
steps = [0, 50, 100, 175, 266]
weak = [0.1456, 0.1742, 0.2162, 0.2087, 0.1907]
strong = [0.4009, 0.3108, 0.3198, 0.3453, 0.3904]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.42, 3.15),
                               gridspec_kw={"height_ratios": [1.2, 0.9], "hspace": 0.72})
ax1.plot(steps, weak, color=C_BLUE, marker="o")
ax1.plot(steps, strong, color=C_ORANGE, marker="s")
ax1.text(8, 0.121, "SFT-50", color=C_BLUE, fontsize=6.0)
ax1.text(8, 0.411, "SFT-201", color=C_ORANGE, fontsize=6.0)
ax1.annotate("GRPO, weak init", xy=(100, 0.2162), xytext=(118, 0.250),
             color=C_BLUE, fontsize=6.0,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=C_BLUE))
ax1.annotate("GRPO, strong init", xy=(175, 0.3453), xytext=(118, 0.378),
             color=C_ORANGE, fontsize=6.0,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=C_ORANGE))
ax1.set_xlim(-8, 278)
ax1.set_ylim(0.10, 0.45)
ax1.set_xlabel("GRPO optimizer step")
ax1.set_ylabel("Test F1@IoU 0.5")
ax1.set_title("(a) Localization on held-out test cases", pad=3)

labels = ["Qwen\nweak init", "Qwen\nstrong init", "LLaVA\nweak init"]
runs = ["qwen_grpo_v1", "qwen_grpo_ck201", "llava_grpo_v1"]
vals = []
for run in runs:
    log = next((ROOT / "output" / run).glob("*/logging.jsonl"))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    vals.append(statistics.mean(row["frac_reward_zero_std"] for row in rows if "frac_reward_zero_std" in row))
colors = [C_BLUE, C_ORANGE, C_GREEN]
bars = ax2.bar(range(3), vals, color=colors, width=0.62, edgecolor="black", linewidth=0.35)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2)
ax2.set_xticks(range(3))
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 0.63)
ax2.set_ylabel("Zero-variance fraction")
ax2.set_title("(b) Groups with identical rewards (mean)", pad=3)
ax2.tick_params(axis="x", length=0)

fig.savefig(f"{ROOT}/paper/figures/fig_results.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{ROOT}/paper/figures/fig_preview.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
print("saved")
