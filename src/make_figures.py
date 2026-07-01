"""
Builds the figures from the raw results.

Writes high-DPI PNGs to figures/. Style is on purpose: one accent color
(muted teal), warm neutral text, no chartjunk, no gridlines we don't need,
no 3D anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

ACCENT = "#20808D"
ACCENT_DARK = "#1B474D"
ACCENT_LIGHT = "#BCE2E7"
WARNING = "#A84B2F"
TEXT = "#28251D"
MUTED = "#7A7974"
SURFACE = "#F7F6F2"

SCENARIO_ORDER = ["outside", "insider", "accidental"]
SCENARIO_LABELS = {
    "outside": "Outside attacker",
    "insider": "Insider probing",
    "accidental": "Accidental self-injection",
}
CONFIG_ORDER = ["off", "input_only", "input_output", "full_stack", "provenance_block"]
CONFIG_LABELS = {
    "off": "No defense",
    "input_only": "Input filter\nonly",
    "input_output": "Input +\noutput",
    "full_stack": "Full stack\n(log-only)",
    "provenance_block": "Full stack +\nquarantine",
}


def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "figure.facecolor": "#F7F6F2",
        "axes.facecolor": "#F7F6F2",
        "savefig.facecolor": "#F7F6F2",
        "savefig.dpi": 200,
    })


def figure_heatmap(summary: Dict, out_path: Path) -> None:
    matrix = np.zeros((len(SCENARIO_ORDER), len(CONFIG_ORDER)))
    for i, scenario in enumerate(SCENARIO_ORDER):
        for j, config in enumerate(CONFIG_ORDER):
            matrix[i, j] = summary[config]["per_scenario"][scenario]["success_rate"]

    fig, ax = plt.subplots(figsize=(9, 3.4))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("teal", [SURFACE, ACCENT_LIGHT,
                                                       ACCENT, ACCENT_DARK])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=9)
    ax.set_yticks(range(len(SCENARIO_ORDER)))
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER], fontsize=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.55 else TEXT
            ax.text(j, i, f"{int(val * 100)}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Attack success rate by scenario and defense configuration",
                 loc="left", pad=12, color=TEXT)
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_grouped_bars(summary: Dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(CONFIG_ORDER))
    width = 0.26

    colors = {"outside": ACCENT, "insider": WARNING, "accidental": ACCENT_DARK}

    for i, scenario in enumerate(SCENARIO_ORDER):
        heights = [summary[c]["per_scenario"][scenario]["success_rate"] * 100
                   for c in CONFIG_ORDER]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, heights, width,
                      color=colors[scenario],
                      edgecolor="none",
                      label=SCENARIO_LABELS[scenario])
        for bar, h in zip(bars, heights):
            if h > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{int(h)}%", ha="center", fontsize=8, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER], fontsize=9)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(0, 108)
    ax.axhline(0, color=TEXT, linewidth=0.6)
    ax.set_title("Where each defense earns its keep",
                 loc="left", pad=12, color=TEXT)
    ax.set_ylabel("Attack success rate", color=TEXT)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_leaked_categories(combined: Dict, out_path: Path) -> None:
    from collections import Counter

    # split into with/without output filter. all three output-filter configs
    # zero out the verbatim categories, so grouping them keeps the chart honest
    groups = {
        "Without output filter": ["off", "input_only"],
        "With output filter": ["input_output", "full_stack", "provenance_block"],
    }
    # average per config within a group so groups of different sizes are
    # comparable (2 configs vs 3)
    counts_by_group: Dict[str, Dict[str, float]] = {}
    for group_name, configs in groups.items():
        per_config_counters: List[Counter] = []
        for config in configs:
            counter: Counter = Counter()
            for outcome in combined[config]["outcomes"]:
                trial_categories = set()
                for response in outcome["responses"]:
                    for cat in response["leaked_categories"]:
                        trial_categories.add(cat)
                for cat in trial_categories:
                    counter[cat] += 1
            per_config_counters.append(counter)
        all_keys = {k for c in per_config_counters for k in c.keys()}
        counts_by_group[group_name] = {
            k: sum(c.get(k, 0) for c in per_config_counters) / len(per_config_counters)
            for k in all_keys
        }

    # not alphabetical on purpose, verbatim categories first, inferred last
    category_order = [
        "client_records",
        "approval_threshold",
        "policy_verbatim",
        "system_prompt_signature",
        "approval_threshold_inferred",
    ]
    all_categories = [c for c in category_order
                      if any(c in counts for counts in counts_by_group.values())]
    if not all_categories:
        return
    friendly_labels = {
        "client_records": "Client records",
        "approval_threshold": "Threshold\n(verbatim)",
        "approval_threshold_inferred": "Threshold\n(inferred)",
        "policy_verbatim": "Policy wording",
        "system_prompt_signature": "System prompt",
    }

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(all_categories))
    width = 0.36
    palette = [WARNING, ACCENT]  # rust = without filter, teal = with

    for i, (group_name, counts) in enumerate(counts_by_group.items()):
        heights = [counts.get(cat, 0.0) for cat in all_categories]
        bars = ax.bar(x + (i - 0.5) * width, heights, width,
                      color=palette[i], edgecolor="none", label=group_name)
        for bar, height in zip(bars, heights):
            if height > 0:
                label = f"{height:.1f}" if height % 1 else f"{int(height)}"
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                        label, ha="center", va="bottom",
                        fontsize=9, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([friendly_labels.get(c, c) for c in all_categories],
                       fontsize=9)
    ax.set_ylabel("Trials that leaked (avg per configuration)", color=TEXT)
    ax.set_title("Verbatim categories fall to zero. Inferred threshold does not.",
                 loc="left", pad=12, color=TEXT)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(top=max(max(c.values()) for c in counts_by_group.values()) * 1.35)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel=None, color=SURFACE, edge=TEXT,
            text_color=TEXT):
        from matplotlib.patches import FancyBboxPatch
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.9, edgecolor=edge, facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)
        if sublabel:
            sub_color = "#BCE2E7" if text_color == "white" else MUTED
            ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                    fontsize=8, color=sub_color)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.9))

    box(0.2, 4.4, 1.6, 1.0, "Advisor", "user turn", color="white")
    box(2.2, 4.4, 2.0, 1.0, "Input filter", "adversarial phrasing",
        color=ACCENT_LIGHT)
    box(4.5, 4.4, 2.0, 1.0, "Retriever", "TF-IDF cosine", color="white")
    box(6.8, 4.4, 2.0, 1.0, "Provenance filter", "trusted vs. imported",
        color=ACCENT_LIGHT)

    box(4.5, 2.4, 2.0, 1.2, "Model backend", "SimulatedBackend\nor OpenAI",
        color=ACCENT, text_color="white")
    box(6.8, 2.4, 2.0, 1.2, "Output filter", "known-secret patterns",
        color=ACCENT_LIGHT)
    box(8.5, 0.6, 1.3, 1.2, "Advisor", "response", color="white")

    arrow(1.8, 4.9, 2.2, 4.9)
    arrow(4.2, 4.9, 4.5, 4.9)
    arrow(6.5, 4.9, 6.8, 4.9)
    arrow(5.5, 4.4, 5.5, 3.6)
    arrow(6.5, 3.0, 6.8, 3.0)
    arrow(7.8, 2.4, 8.5, 1.4)

    box(0.2, 0.6, 3.8, 1.2, "Sandbox corpus",
        "client notes  policies  calendar  +  untrusted", color=SURFACE)
    arrow(2.0, 1.8, 5.0, 2.4)

    ax.text(0.2, 5.7, "Assistant under test", fontsize=11,
            fontweight="bold", color=TEXT)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    _set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "summary.json").open() as f:
        summary = json.load(f)
    with (RESULTS_DIR / "all_configurations.json").open() as f:
        combined = json.load(f)

    figure_heatmap(summary, FIGURES_DIR / "fig1_heatmap.png")
    figure_grouped_bars(summary, FIGURES_DIR / "fig2_grouped_bars.png")
    figure_leaked_categories(combined, FIGURES_DIR / "fig3_leaked_categories.png")
    figure_architecture(FIGURES_DIR / "fig0_architecture.png")

    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
