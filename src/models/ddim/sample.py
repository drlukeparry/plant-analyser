"""Phase 5 sampling: generate a field patch conditioned on a Phase 3
composition latent, run it through Phase 4's inverse (tessellate_from_field)
to recover instance labels, and compare generated-vs-real node feature
distributions -- the same comparison Phase 4's round-trip validation used,
now applied to a genuinely generated (not just field-encoded) sample.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.features import extract_cell_features
from src.graph.tessellate import tessellate_from_field
from src.models.ddim.model import FieldUNet
from src.models.ddim.train import DATA_DIR, load_patches
from src.models.gvae.train import device

OUT_DIR = DATA_DIR
# area/elongation/orientation are computed purely from recovered cell
# geometry, meaningful regardless of patch cropping. radial_distance is
# deliberately excluded here -- Phase 4 found it's not a stable per-patch
# quantity (small-crop "distance from center" is dominated by which patch
# happened to be sampled, see PLAN.md), so it's not a fair generated-vs-real
# comparison at this patch scale.
COMPARE_FEATURES = ["area", "elongation", "orientation"]


def load_model():
    dev = device()
    fields, conds = load_patches()
    scaler = np.load(OUT_DIR / "field_scaler.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])

    model = FieldUNet(in_channels=fields.shape[-1], cond_dim=conds.shape[-1]).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / "ddim_unet.pt", map_location=dev))
    model.eval()
    return model, mean, std, dev


@torch.no_grad()
def generate(model, cond: torch.Tensor, mean, std, dev, num_inference_steps: int = 50,
             num_train_timesteps: int = 1000, patch_size: int = 128, in_channels: int = 4,
             seed: int = 0):
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, in_channels, patch_size, patch_size, generator=gen).to(dev)
    cond = cond.to(dev)

    for t in scheduler.timesteps:
        t_batch = t.expand(1).to(dev)
        pred_noise = model(x, t_batch, cond)
        x = scheduler.step(pred_noise, t, x).prev_sample

    field = (x.cpu() * std + mean)[0].permute(1, 2, 0).numpy()  # (H, W, C)
    return field


def evaluate_sample(field: np.ndarray, real_node_df, out_path: Path, tag: str):
    recovered = tessellate_from_field(field)
    recovered_df = extract_cell_features(recovered)
    print(f"[{tag}] generated field -> {len(recovered_df)} recovered cells "
          f"(real patch had {len(real_node_df)})")

    if len(recovered_df) == 0:
        print(f"[{tag}] 0 cells recovered -- generated field likely too noisy/uniform this early in training")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(field[..., 0], cmap="viridis")
        ax[0].set_title(f"{tag}: generated sdf channel")
        ax[1].imshow(np.zeros_like(field[..., 0]), cmap="gray")
        ax[1].set_title("0 cells recovered")
        fig.savefig(out_path, dpi=130)
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(field[..., 0], cmap="viridis")
    axes[0, 0].set_title(f"{tag}: generated sdf field")
    axes[0, 1].imshow(recovered, cmap="nipy_spectral")
    axes[0, 1].set_title(f"tessellated ({len(recovered_df)} cells)")
    axes[0, 2].axis("off")

    for ax, feat in zip(axes[1], COMPARE_FEATURES[:3]):
        a = real_node_df[feat].values if feat in real_node_df else None
        b = recovered_df[feat].values if feat in recovered_df else None
        if a is None or b is None:
            ax.axis("off")
            continue
        lo, hi = np.percentile(np.concatenate([a, b]), [1, 99])
        bins = np.linspace(lo, hi, 25)
        ax.hist(a, bins=bins, alpha=0.5, label="real patches", density=True)
        ax.hist(b, bins=bins, alpha=0.5, label="generated", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[{tag}] saved {out_path}")


def main(n_samples: int = 4, seed: int = 0):
    model, mean, std, dev = load_model()
    fields, conds = load_patches()

    # Real feature distributions to compare generated samples against --
    # recompute node features (with the norm columns) across all real patches.
    real_dfs = []
    for f in fields[: min(len(fields), 200)]:
        labels_stub = tessellate_from_field(f)  # re-tessellate real fields too, for an apples-to-apples comparison
        df = extract_cell_features(labels_stub)
        if len(df) > 0:
            real_dfs.append(df)
    import pandas as pd
    real_node_df = pd.concat(real_dfs, ignore_index=True) if real_dfs else pd.DataFrame()

    rng = np.random.default_rng(seed)
    for i in range(n_samples):
        cond_idx = rng.integers(0, len(conds))
        cond = torch.tensor(conds[cond_idx:cond_idx + 1], dtype=torch.float32)
        field = generate(model, cond, mean, std, dev, seed=seed + i)
        evaluate_sample(field, real_node_df, OUT_DIR / f"generated_sample_{i}.png", f"sample{i}")


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(n_samples)
