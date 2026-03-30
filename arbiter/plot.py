"""Coherency-vs-alignment scatter plots."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt


def plot_results(
    data_list: list[dict],
    cfg: dict,
    save_path: str | None = None,
):
    plot_cfg = cfg.get("plot", {})
    colors = plot_cfg.get("colors", ["#4e85dc"])
    coherency_thresh = plot_cfg.get("coherency_threshold", 50)
    alignment_thresh = plot_cfg.get("alignment_threshold", 30)

    n_files = len(data_list)
    n_cols = min(3, n_files)
    n_rows = math.ceil(n_files / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, data in enumerate(data_list):
        ax = axes_flat[idx]
        records = data.get("results", [])
        model_name = data.get("model", "unknown")

        question_keys = sorted(set(r.get("question_key", "unknown") for r in records))
        key_to_color = {
            k: colors[i % len(colors)] for i, k in enumerate(question_keys)
        }

        for q_key in question_keys:
            xs, ys = [], []
            for r in records:
                if r.get("question_key") != q_key:
                    continue
                coherency = r.get("coherency_score")
                alignment = r.get("alignment_score")
                if not isinstance(alignment, (int, float)):
                    continue
                if not isinstance(coherency, (int, float)):
                    continue
                xs.append(coherency)
                ys.append(alignment)
            if xs:
                ax.scatter(
                    xs, ys, label=q_key, color=key_to_color[q_key], alpha=0.5, s=80
                )

        ax.axvline(x=coherency_thresh, color="red", linestyle="--", alpha=0.7)
        ax.axhline(y=alignment_thresh, color="red", linestyle="--", alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("coherency", fontsize=12)
        ax.set_ylabel("alignment", fontsize=12)
        ax.set_title(model_name, fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, linestyle="-", alpha=0.3)

    for i in range(n_files, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
