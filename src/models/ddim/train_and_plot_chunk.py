"""Runs one resumable training chunk (train_boundary.py's train()) and
immediately produces two progress artifacts from the updated checkpoint:
a loss-curve plot (full history so far) and the same real-vs-generated sdf
grid used for the first 4000-step check -- same patch indices and
generation seeds every time, so consecutive chunks are a controlled
before/after comparison of the model weights alone, nothing else changing.

Deliberately one process per chunk (train + plot together, then exit) --
matches the memory-reset rationale train_boundary.py's train() docstring
already gives for chunked training.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.ddim.model import FieldUNet
from src.models.ddim.train_boundary import N_TARGET_CH, train
from src.models.gvae.train import device

DATA_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"

# fixed once (from the initial 4000-step check) so every chunk's grid uses
# the exact same real patches -- only the model weights differ between plots
PICK_IDX = [1682, 334, 754, 116, 20, 1856]


@torch.no_grad()
def generate(model, cond, boundary_norm, tgt_mean, tgt_std, dev, seed):
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(50)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, N_TARGET_CH, 256, 256, generator=gen).to(dev)
    cond, boundary_norm = cond.to(dev), boundary_norm.to(dev)
    for t in scheduler.timesteps:
        model_in = torch.cat([x, boundary_norm], dim=1)
        pred_noise = model(model_in, t.expand(1).to(dev), cond)
        x = scheduler.step(pred_noise, t, x).prev_sample
    return (x.cpu()[0] * tgt_std[0] + tgt_mean[0]).permute(1, 2, 0).numpy()


def plot_loss(step: int):
    h = np.load(DATA_DIR / "ddim_train_history_boundary.npy")
    steps, loss = h[:, 0], h[:, 1]
    k = min(100, max(1, len(loss) // 20))
    sm = np.convolve(loss, np.ones(k) / k, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, logscale in zip(axes, [False, True]):
        ax.plot(steps[k - 1:], sm, color="tab:blue", lw=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("MSE loss (smoothed)")
        if logscale:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
        ax.grid(alpha=0.3)
    fig.suptitle(f"boundary-conditioned DDIM loss, step {step}", fontsize=12)
    fig.tight_layout()
    out_path = DATA_DIR / f"progress_step{step:05d}_loss.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_grid(step: int):
    dev = device()
    scaler = np.load(DATA_DIR / "field_scaler_boundary.npz")
    tgt_mean, tgt_std = torch.tensor(scaler["target_mean"]), torch.tensor(scaler["target_std"])
    bnd_mean, bnd_std = torch.tensor(scaler["boundary_mean"]), torch.tensor(scaler["boundary_std"])
    config = np.load(DATA_DIR / "ddim_config_boundary.npz")
    model = FieldUNet(in_channels=int(config["in_channels"]), out_channels=int(config["out_channels"]),
                       cond_dim=int(config["cond_dim"]), base_ch=int(config["base_ch"])).to(dev)
    ckpt = torch.load(DATA_DIR / "ddim_ckpt_boundary.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fields = np.load(DATA_DIR / "patch_fields_boundary.npy", mmap_mode="r")
    conds = np.load(DATA_DIR / "patch_conds_boundary.npy")

    fig, axes = plt.subplots(len(PICK_IDX), 3, figsize=(12, 4 * len(PICK_IDX)))
    for row, idx in enumerate(PICK_IDX):
        patch = np.asarray(fields[idx]).astype(np.float32)
        real_sdf = patch[..., 0]
        boundary_raw = patch[..., N_TARGET_CH]
        b_mean = float(boundary_raw.mean())
        cond = torch.tensor(conds[idx:idx + 1], dtype=torch.float32)
        boundary_norm = (torch.tensor(boundary_raw, dtype=torch.float32)[None, None] - bnd_mean) / bnd_std

        gen_field = generate(model, cond, boundary_norm, tgt_mean, tgt_std, dev, seed=row)
        gen_sdf = gen_field[..., 0]

        v = max(abs(boundary_raw.min()), abs(boundary_raw.max()), 1e-9)
        axes[row, 0].imshow(boundary_raw, cmap="RdBu", vmin=-v, vmax=v)
        axes[row, 0].set_title(f"boundary_sdf input\nmean={b_mean:.3f}", fontsize=9)
        axes[row, 1].imshow(real_sdf, cmap="viridis", vmin=real_sdf.min(), vmax=real_sdf.max())
        axes[row, 1].set_title("real sdf (ground truth)", fontsize=9)
        axes[row, 2].imshow(gen_sdf, cmap="viridis", vmin=real_sdf.min(), vmax=real_sdf.max())
        axes[row, 2].set_title(f"generated sdf (step {step})", fontsize=9)
        for c in range(3):
            axes[row, c].axis("off")

    fig.suptitle(f"real vs generated sdf, step {step} (same patches/seeds every chunk)", fontsize=12)
    fig.tight_layout()
    out_path = DATA_DIR / f"progress_step{step:05d}_fields.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(chunk_steps: int = 1000):
    model = train(steps=chunk_steps, batch_size=16)
    del model  # plot_grid reloads from the saved checkpoint, fresh

    ckpt = torch.load(DATA_DIR / "ddim_ckpt_boundary.pt", map_location="cpu", weights_only=False)
    step = ckpt["step"]
    loss_path = plot_loss(step)
    grid_path = plot_grid(step)
    print(f"PROGRESS_ARTIFACTS step={step} loss={loss_path} grid={grid_path}")


if __name__ == "__main__":
    chunk_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(chunk_steps)
