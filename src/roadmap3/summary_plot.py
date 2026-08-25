"""Summary chart of every control-following measurement taken across this
roadmap's experiments (D3 through the whole-structure attempt) -- all the
same metric, corr(requested/size_gradient_target, generated diameter),
so they're directly comparable on one axis. Not a new experiment; just
collects numbers already reported in ROADMAP3.md into one view.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from src.roadmap3.train import OUT_DIR

RESULTS = [
    ("free-joint\n(patch, real)", 0.458, "in-distribution"),
    ("pos-cond retrain\n(patch, real)", 0.364, "in-distribution"),
    ("RePaint-pinned\n(patch, real)", 0.167, "pinning mechanism"),
    ("pos-cond\n(synthetic boundary)", 0.083, "generalization"),
    ("RePaint-pinned\n(synthetic boundary)", 0.049, "pinning mechanism"),
    ("pos-cond\n(full structure, n=6748)", 0.024, "generalization"),
]

COLORS = {
    "in-distribution": "#2a9d8f",
    "pinning mechanism": "#e76f51",
    "generalization": "#e9c46a",
}


def plot():
    labels = [r[0] for r in RESULTS]
    values = [r[1] for r in RESULTS]
    kinds = [r[2] for r in RESULTS]
    colors = [COLORS[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)

    ax.set_xlabel("corr(requested size_gradient_target, generated diameter)")
    ax.set_xlim(0, 0.55)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_title("Control-following across every experiment in ROADMAP3\n"
                 "(same metric throughout, so directly comparable)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), loc="lower right", title="what's being tested")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "roadmap3_summary.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot()
