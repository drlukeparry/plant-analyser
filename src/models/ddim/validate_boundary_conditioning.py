"""The actual test this experiment exists to answer: does giving the model a
spatial boundary_sdf crop change what it generates, in the direction real
data actually varies (smaller/sparser cells near the tissue edge)? Compares
three correlations (boundary_sdf value vs. generated cell size) computed the
same way: real DO patches (the target signature to match), the baseline
model (no positional input at all -- expected ~0 correlation, it has no way
to know), and the boundary-conditioned model (expected closer to real if the
conditioning channel is actually being used).

Reuses real (boundary_sdf, GVAE-latent) pairs from the training patch set as
generation inputs rather than a held-out split -- this experiment is a
first-pass "does the mechanism work at all" check, not a generalization
claim; a held-out split would be the natural next step if this looks
promising.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.graph.features import extract_cell_features
from src.graph.tessellate import tessellate_from_field
from src.models.ddim.model import FieldUNet
from src.models.ddim.train_boundary import N_TARGET_CH
from src.models.gvae.train import device

DATA_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"
N_SAMPLES = 40


def load_baseline(dev):
    scaler = np.load(DATA_DIR / "field_scaler_baseline_do256.npz")
    mean, std = torch.tensor(scaler["mean"]), torch.tensor(scaler["std"])
    config = np.load(DATA_DIR / "ddim_config_baseline_do256.npz")
    model = FieldUNet(in_channels=int(config["in_channels"]), cond_dim=int(config["cond_dim"]),
                       base_ch=int(config["base_ch"])).to(dev)
    model.load_state_dict(torch.load(DATA_DIR / "ddim_unet_baseline_do256.pt", map_location=dev))
    model.eval()
    return model, mean, std


def load_boundary(dev):
    scaler = np.load(DATA_DIR / "field_scaler_boundary.npz")
    target_mean, target_std = torch.tensor(scaler["target_mean"]), torch.tensor(scaler["target_std"])
    boundary_mean, boundary_std = torch.tensor(scaler["boundary_mean"]), torch.tensor(scaler["boundary_std"])
    config = np.load(DATA_DIR / "ddim_config_boundary.npz")
    model = FieldUNet(in_channels=int(config["in_channels"]), out_channels=int(config["out_channels"]),
                       cond_dim=int(config["cond_dim"]), base_ch=int(config["base_ch"])).to(dev)
    model.load_state_dict(torch.load(DATA_DIR / "ddim_unet_boundary.pt", map_location=dev))
    model.eval()
    return model, target_mean, target_std, boundary_mean, boundary_std


@torch.no_grad()
def generate_baseline(model, cond, mean, std, dev, patch_size=256, num_inference_steps=50,
                       num_train_timesteps=1000, seed=0):
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, N_TARGET_CH, patch_size, patch_size, generator=gen).to(dev)
    cond = cond.to(dev)
    for t in scheduler.timesteps:
        pred_noise = model(x, t.expand(1).to(dev), cond)
        x = scheduler.step(pred_noise, t, x).prev_sample
    return (x.cpu() * std + mean)[0].permute(1, 2, 0).numpy()


@torch.no_grad()
def generate_boundary(model, cond, boundary_crop_norm, dev, patch_size=256, num_inference_steps=50,
                       num_train_timesteps=1000, seed=0):
    """boundary_crop_norm: (1,1,H,W) tensor, already normalized the same way
    training data was -- held fixed (never noised) across every denoising
    step, exactly like train_boundary.py's training-time conditioning."""
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, N_TARGET_CH, patch_size, patch_size, generator=gen).to(dev)
    cond = cond.to(dev)
    boundary_crop_norm = boundary_crop_norm.to(dev)
    for t in scheduler.timesteps:
        model_in = torch.cat([x, boundary_crop_norm], dim=1)
        pred_noise = model(model_in, t.expand(1).to(dev), cond)
        x = scheduler.step(pred_noise, t, x).prev_sample
    return x.cpu()[0]  # still normalized target-space -- caller denormalizes with target_mean/std


def field_summary(field_3ch: np.ndarray):
    """One (mean_size, density, orientation_coherence) triple per
    generated/real field. density is recovered cell count per unit area, the
    same "how packed is this patch" signal real data actually shows a
    radial/boundary trend in. orientation_coherence is the circular mean
    resultant length R of per-cell `orientation` -- orientation is 180deg
    periodic (skimage regionprops convention, range [-pi/2, pi/2]), so a
    plain average would be wrong (ROADMAP2 flags this same issue for a flow
    field); using the double-angle trick (R computed from cos/sin(2*theta))
    is the standard fix. R in [0,1]: ~0 means cell orientations in this
    patch are uncorrelated/random, ~1 means they're all aligned -- this is
    "how aligned is this patch," not a directional value, so it's the
    natural analogue of size/density for testing whether boundary_sdf
    changes *how much* alignment the model generates, not which direction."""
    labels = tessellate_from_field(field_3ch)
    df = extract_cell_features(labels)
    if len(df) == 0:
        return None, None, None
    h, w = field_3ch.shape[:2]
    mean_size = df["equivalent_diameter"].mean()
    density = len(df) / (h * w)
    theta2 = 2 * df["orientation"].values
    resultant = np.hypot(np.cos(theta2).mean(), np.sin(theta2).mean())
    return mean_size, density, resultant


def main(n_samples: int = N_SAMPLES, seed: int = 0, mismatch_cond: bool = False):
    """mismatch_cond=True is the confound-control rerun: `cond` (the GVAE
    composition latent) is drawn from a *different*, randomly chosen patch
    than the one supplying boundary_raw, for both generated variants. Since
    the GVAE latent already strongly encodes radial position (decoder eval:
    radial_distance_norm corr=0.99), using the matched real `cond` let both
    models' outputs correlate with boundary_sdf even though only one of them
    ever saw it -- the first run's baseline correlation (0.735) proved this.
    Decoupling cond removes that shared channel: whatever correlation
    survives here can only come from the boundary_sdf input itself."""
    dev = device()
    base_model, base_mean, base_std = load_baseline(dev)
    bnd_model, tgt_mean, tgt_std, bnd_mean, bnd_std = load_boundary(dev)

    fields = np.load(DATA_DIR / "patch_fields_boundary.npy", mmap_mode="r")  # (N,256,256,4)
    conds = np.load(DATA_DIR / "patch_conds_boundary.npy")
    n_total = fields.shape[0]

    # boundary_sdf per patch summarized as its mean value -- pick indices
    # spanning that range (not a random sample) so the correlation check
    # actually has spread to detect, rather than relying on luck.
    print("computing per-patch mean boundary_sdf to pick a spanning sample...")
    bsdf_means = np.array([float(np.asarray(fields[i, ..., N_TARGET_CH]).mean()) for i in range(n_total)])
    order = np.argsort(bsdf_means)
    idx = order[np.linspace(0, n_total - 1, n_samples).astype(int)]

    rng = np.random.default_rng(seed)
    if mismatch_cond:
        # a derangement-ish shuffle: cond_idx[i] != idx[i] for every i, so no
        # sample accidentally keeps its own matched cond
        cond_source = rng.permutation(idx)
        clashes = cond_source == idx
        while clashes.any():
            cond_source[clashes] = rng.permutation(cond_source[clashes])
            clashes = cond_source == idx
    else:
        cond_source = idx

    real_boundary, real_size, real_density, real_orient = [], [], [], []
    base_boundary, base_size, base_density, base_orient = [], [], [], []
    bnd_boundary, bnd_size, bnd_density, bnd_orient = [], [], [], []

    # save full field arrays at a few spanning indices (near-boundary, mid,
    # interior) for a visual preview -- not needed for the correlation
    # numbers, just so the result is inspectable, not only a table.
    preview_slots = set(np.linspace(0, n_samples - 1, min(4, n_samples)).astype(int))
    preview = []

    for i, patch_idx in enumerate(idx):
        patch = np.asarray(fields[patch_idx]).astype(np.float32)  # (256,256,4)
        target_real = patch[..., :N_TARGET_CH]
        boundary_raw = patch[..., N_TARGET_CH]
        b_mean_raw = float(boundary_raw.mean())
        cond_idx = cond_source[i]
        cond = torch.tensor(conds[cond_idx:cond_idx + 1], dtype=torch.float32)

        # --- real data signature ---
        sz, dens, orient = field_summary(target_real)
        if sz is not None:
            real_boundary.append(b_mean_raw)
            real_size.append(sz)
            real_density.append(dens)
            real_orient.append(orient)

        # --- baseline (no positional input) ---
        field_base = generate_baseline(base_model, cond, base_mean, base_std, dev, seed=seed + i)
        sz, dens, orient = field_summary(field_base)
        if sz is not None:
            base_boundary.append(b_mean_raw)
            base_size.append(sz)
            base_density.append(dens)
            base_orient.append(orient)

        # --- boundary-conditioned, using this patch's real boundary_sdf crop ---
        boundary_t = torch.tensor(boundary_raw, dtype=torch.float32)[None, None]
        boundary_norm = (boundary_t - bnd_mean) / bnd_std
        x_bnd = generate_boundary(bnd_model, cond, boundary_norm, dev, seed=seed + i)
        field_bnd = (x_bnd * tgt_std[0] + tgt_mean[0]).permute(1, 2, 0).numpy()
        sz, dens, orient = field_summary(field_bnd)
        if sz is not None:
            bnd_boundary.append(b_mean_raw)
            bnd_size.append(sz)
            bnd_density.append(dens)
            bnd_orient.append(orient)

        if i in preview_slots:
            preview.append(dict(b_mean_raw=b_mean_raw, target_real=target_real,
                                 field_base=field_base, field_bnd=field_bnd, boundary_raw=boundary_raw))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_samples} samples done")

    def report(name, boundary, size, density, orient):
        if len(boundary) < 3:
            print(f"{name}: too few valid samples ({len(boundary)}) to correlate")
            return
        boundary, size, density, orient = np.array(boundary), np.array(size), np.array(density), np.array(orient)
        r_size = np.corrcoef(boundary, size)[0, 1]
        r_density = np.corrcoef(boundary, density)[0, 1]
        r_orient = np.corrcoef(boundary, orient)[0, 1]
        print(f"{name}: n={len(boundary)}, corr(boundary_sdf, size)={r_size:.3f}, "
              f"corr(boundary_sdf, density)={r_density:.3f}, "
              f"corr(boundary_sdf, orientation_coherence)={r_orient:.3f}")
        return r_size, r_density, r_orient

    tag = "mismatch" if mismatch_cond else "matched"
    print(f"\n=== correlation between boundary_sdf and generated/real cell size/density/orientation ({tag} cond) ===")
    report("real DO patches (target signature)", real_boundary, real_size, real_density, real_orient)
    report("baseline (no positional input)", base_boundary, base_size, base_density, base_orient)
    report("boundary-conditioned", bnd_boundary, bnd_size, bnd_density, bnd_orient)

    out_path = DATA_DIR / f"validate_boundary_conditioning_results_{tag}.npz"
    np.savez(out_path,
             real_boundary=real_boundary, real_size=real_size, real_density=real_density, real_orient=real_orient,
             base_boundary=base_boundary, base_size=base_size, base_density=base_density, base_orient=base_orient,
             bnd_boundary=bnd_boundary, bnd_size=bnd_size, bnd_density=bnd_density, bnd_orient=bnd_orient)
    print(f"\nsaved raw results to {out_path}")

    save_preview(preview, tag)


def save_preview(preview, tag):
    import matplotlib.pyplot as plt
    n = len(preview)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["boundary_sdf input (conditioning)", "real (sdf channel)",
                  "baseline generated (sdf)", "boundary-conditioned generated (sdf)"]
    for row, p in enumerate(preview):
        ax = axes[row, 0]
        # per-row scale, not a fixed global one -- a patch straddling the
        # tissue edge can have a real local range of a few hundredths (e.g.
        # a thin lobe), which a fixed 0..0.9 scale (appropriate for a deep-
        # interior patch) washes out to a flat color even though the sign
        # and shape are correct (verified directly: sign(boundary_sdf) vs.
        # tissue_mask matched 100% for the patch that first exposed this).
        b = p["boundary_raw"]
        vmax = max(abs(b.min()), abs(b.max()), 1e-6)
        im = ax.imshow(b, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{col_titles[0]}\nmean={p['b_mean_raw']:.3f} (range {b.min():.3f}..{b.max():.3f})" if row == 0
                     else f"mean={p['b_mean_raw']:.3f} (range {b.min():.3f}..{b.max():.3f})", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for col, (key, title) in enumerate(zip(["target_real", "field_base", "field_bnd"], col_titles[1:]), start=1):
            ax = axes[row, col]
            ax.imshow(p[key][..., 0], cmap="viridis", vmin=-0.03, vmax=0.03)
            ax.set_title(f"{title}\nboundary_sdf mean={p['b_mean_raw']:.3f}" if row == 0 else
                         f"boundary_sdf mean={p['b_mean_raw']:.3f}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Preview: rows span near-boundary -> interior (by real boundary_sdf mean)", fontsize=11)
    fig.tight_layout()
    out_path = DATA_DIR / f"validate_boundary_conditioning_preview_{tag}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved preview to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    mismatch = len(sys.argv) > 2 and sys.argv[2] == "mismatch"
    main(n, mismatch_cond=mismatch)
