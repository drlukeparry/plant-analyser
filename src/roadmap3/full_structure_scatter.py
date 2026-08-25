"""Whole-stem visualization for full_structure.py's generated set. The
power-diagram renderer (geometry.py) is O(N^2) pure Python -- fine at
patch scale, far too slow across all 6,748 of DO_0000's cells (that's why
full_structure.py only rendered a small crop with it). A scatter plot
(position from the real, held-fixed (x, y); marker size + color from the
generated equivalent_diameter_norm) is O(N) and shows the entire structure,
at the cost of not showing actual cell-wall geometry -- a legitimate
coarser view, not a substitute for the crop's real tessellation.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.roadmap3.full_structure import load_do0000_full
from src.roadmap3.sample_poscond import FEATURE_NAMES, POS_IDX, generate, load_model
from src.roadmap3.train import OUT_DIR


def run(num_inference_steps: int = 50, seed: int = 0):
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    x_raw, ctrl_raw = load_do0000_full()
    n = x_raw.size(0)
    print(f"DO_0000 full structure: {n} real cells")

    pos_raw = x_raw[:, POS_IDX]
    x_gen = generate(model, pos_raw, ctrl_raw, mean, std, ctrl_mean, ctrl_std, dev,
                      num_train_timesteps, num_inference_steps, seed=seed)

    r = pos_raw[:, 0].numpy()
    theta = pos_raw[:, 1].numpy()
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    diam_idx = FEATURE_NAMES.index("equivalent_diameter_norm")
    real_diam = x_raw[:, diam_idx].numpy()
    gen_diam = x_gen[:, diam_idx]
    # marker area proportional to diameter^2 (~ cell area), scaled for visibility
    real_size = 4000 * (real_diam / real_diam.max()) ** 2
    gen_size = 4000 * (np.clip(gen_diam, 0, None) / real_diam.max()) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, xy_, diam, size, title in [
        (axes[0], xy, real_diam, real_size, f"real DO_0000 (n={n})"),
        (axes[1], xy, gen_diam, gen_size, f"generated (whole-structure DDIM, n={n})"),
    ]:
        sc = ax.scatter(xy_[:, 0], xy_[:, 1], s=size, c=diam, cmap="viridis",
                         vmin=0, vmax=real_diam.max(), edgecolors="none", alpha=0.85)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="equivalent_diameter_norm", shrink=0.7)

    fig.suptitle(f"DO_0000 whole-structure generation, all {n} real cell positions\n"
                 "(scatter, not tessellation -- marker size/color = generated diameter)")
    out_path = OUT_DIR / "do0000_full_structure_scatter.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    run()
