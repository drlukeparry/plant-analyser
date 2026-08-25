"""Test the ORIGINAL free-joint model (train.py/sample.py, before the
position-as-conditioning fix) at DO_0000's full scale (6,748 tokens) --
unlike the position-conditioned model used everywhere else in the
whole-structure experiments, this one actually generates its own positions
from noise, conditioned only on DO_0000's real per-node control field. This
is the genuine "does DDIM invent a plausible stem shape on its own" test --
never attempted before this pass, since the free-joint model was previously
only ever run at <=300-token patch scale.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.roadmap3.full_structure import load_do0000_full
from src.roadmap3.sample import generate, load_model
from src.roadmap3.sample_poscond import POS_IDX
from src.roadmap3.train import OUT_DIR


def to_xy(pos: np.ndarray) -> np.ndarray:
    r, theta = pos[:, 0], pos[:, 1]
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def run(seed: int = 0):
    x_raw, ctrl_raw = load_do0000_full()
    n = x_raw.size(0)
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()

    x_gen = generate(model, ctrl_raw, mean, std, ctrl_mean, ctrl_std, dev, num_train_timesteps, seed=seed)

    real_pos = x_raw[:, POS_IDX].numpy()
    gen_pos = x_gen[:, POS_IDX]
    real_xy, gen_xy = to_xy(real_pos), to_xy(gen_pos)

    print(f"real radial_distance_norm range: {real_pos[:, 0].min():.3f} - {real_pos[:, 0].max():.3f}")
    print(f"generated radial_distance_norm range: {gen_pos[:, 0].min():.3f} - {gen_pos[:, 0].max():.3f} "
          f"(collapsed if much narrower than real -- see docstring)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].scatter(real_xy[:, 0], real_xy[:, 1], s=2, color="black")
    axes[0].set_title(f"real DO_0000 positions (n={n})")
    axes[1].scatter(gen_xy[:, 0], gen_xy[:, 1], s=2, color="black")
    axes[1].set_title(f"free-joint DDIM generated positions (n={n})")
    for ax in axes:
        ax.set_aspect("equal")
    fig.suptitle("Real vs. DDIM-generated cell positions (free-joint model, whole DO_0000 scale, "
                 "34x beyond training length)")
    fig.tight_layout()
    out_path = OUT_DIR / "do0000_positions_vs_generated.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    run()
