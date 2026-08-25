"""D3: DDIM reverse sampling + the sanity-check ROADMAP3 calls for before
testing any novel/synthetic conditioning -- can the model regenerate
plausible cell sets when conditioned on a REAL image's own extracted
control fields? Two checks, matching the bar already established elsewhere
in this repo rather than inventing a new one:

1. Pooled feature-distribution comparison (generated vs. real), same
   effect-size style Phase 6 used (PLAN.md: 0.5-std bar).
2. Control-following correlation (does generated equivalent_diameter_norm
   actually track the requested size_gradient_target?), the check R2
   introduced specifically because FieldUNet was never asked to follow an
   explicit control signal and a plausible-looking distribution alone
   doesn't prove the model is using the conditioning.

Sampling does not pad to n_max -- the Transformer denoiser has no fixed
sequence-length requirement (see denoiser.py), so each real subgraph's own
node count is used directly, with an all-active mask.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_graph import NODE_FEATURE_COLS_PX, _PX_TO_NORM_COL
from src.models.gvae.train import device
from src.roadmap3.dataset import load_graphs_with_ctrl, sample_token_subgraph
from src.roadmap3.denoiser import SetDenoiser
from src.roadmap3.train import OUT_DIR

FEATURE_NAMES = [_PX_TO_NORM_COL.get(c, c) for c in NODE_FEATURE_COLS_PX]
COMPARE_FEATURES = ["area_norm", "elongation", "orientation"]
SIZE_FEATURE = "equivalent_diameter_norm"


def load_model():
    dev = device()
    config = np.load(OUT_DIR / "denoiser_config.npz")
    model = SetDenoiser(
        attr_dim=int(config["attr_dim"]), ctrl_dim=int(config["ctrl_dim"]),
        hidden_dim=int(config["hidden_dim"]), n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
    ).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / "set_denoiser.pt", map_location=dev))
    model.eval()
    scaler = np.load(OUT_DIR / "feature_scaler.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    num_train_timesteps = int(config["num_train_timesteps"])
    return model, mean, std, num_train_timesteps, dev


@torch.no_grad()
def generate(model, ctrl: torch.Tensor, mean, std, dev, num_train_timesteps: int,
             num_inference_steps: int = 50, seed: int = 0) -> torch.Tensor:
    """ctrl: [N, ctrl_dim] real per-cell control-field values for one
    subgraph. Returns generated (de-standardized) attribute vectors [N, attr_dim]."""
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    n, ctrl_dim = ctrl.shape
    attr_dim = mean.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, n, attr_dim, generator=gen).to(dev)
    ctrl_b = ctrl.unsqueeze(0).to(dev)
    active = torch.ones(1, n, device=dev)

    for t in scheduler.timesteps:
        t_b = t.expand(1).to(dev)
        pred_noise = model(x, t_b, ctrl_b, active)
        x = scheduler.step(pred_noise, t, x).prev_sample

    x = x.squeeze(0).cpu() * std.squeeze(0) + mean.squeeze(0)
    return x


def pooled_comparison(n_subgraphs: int = 60, seed: int = 1):
    model, mean, std, num_train_timesteps, dev = load_model()
    graphs, ctrls = load_graphs_with_ctrl("norm")
    for g in graphs:
        g.x = (g.x - mean) / std  # match training-time standardization for real-side comparison too

    torch.manual_seed(seed)
    real_all, gen_all = [], []
    size_target_all, size_gen_all = [], []
    for i in range(n_subgraphs):
        gi = torch.randint(0, len(graphs), (1,)).item()
        x, c = sample_token_subgraph(graphs[gi], ctrls[gi], num_hops=3, max_nodes=64)
        if x.size(0) < 8:
            continue
        x_real = (x * std + mean).numpy()  # de-standardize real side too, for a fair comparison
        x_gen = generate(model, c, mean, std, dev, num_train_timesteps, seed=seed * 1000 + i).numpy()
        real_all.append(x_real)
        gen_all.append(x_gen)
        size_target_all.append(c[:, 0].numpy())  # size_gradient_target is CTRL_COLS[0]
        size_gen_all.append(x_gen[:, FEATURE_NAMES.index(SIZE_FEATURE)])

    real = np.concatenate(real_all, axis=0)
    gen = np.concatenate(gen_all, axis=0)
    size_target = np.concatenate(size_target_all)
    size_gen = np.concatenate(size_gen_all)

    print(f"\npooled over {len(real_all)} subgraphs, {real.shape[0]} real / {gen.shape[0]} generated cells")
    print(f"{'feature':<24} {'real mean':>10} {'gen mean':>10} {'real std':>10} {'effect size':>12}")
    effect_sizes = {}
    for name in COMPARE_FEATURES:
        idx = FEATURE_NAMES.index(name)
        r_mean, r_std = real[:, idx].mean(), real[:, idx].std()
        g_mean = gen[:, idx].mean()
        effect = abs(g_mean - r_mean) / max(r_std, 1e-8)
        effect_sizes[name] = effect
        flag = "OK" if effect < 0.5 else "OVER 0.5-std bar"
        print(f"{name:<24} {r_mean:>10.3f} {g_mean:>10.3f} {r_std:>10.3f} {effect:>10.3f}  {flag}")

    control_corr = float(np.corrcoef(size_target, size_gen)[0, 1])
    print(f"\ncontrol-following: corr(requested size_gradient_target, generated {SIZE_FEATURE}) = {control_corr:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(COMPARE_FEATURES) + 1, figsize=(5 * (len(COMPARE_FEATURES) + 1), 4))
    for ax, name in zip(axes, COMPARE_FEATURES):
        idx = FEATURE_NAMES.index(name)
        ax.hist(real[:, idx], bins=40, alpha=0.5, density=True, label="real")
        ax.hist(gen[:, idx], bins=40, alpha=0.5, density=True, label="generated")
        ax.set_title(f"{name}\neffect size={effect_sizes[name]:.3f}")
        ax.legend()
    axes[-1].scatter(size_target, size_gen, s=4, alpha=0.3)
    axes[-1].set_xlabel("requested size_gradient_target")
    axes[-1].set_ylabel(f"generated {SIZE_FEATURE}")
    axes[-1].set_title(f"control-following, corr={control_corr:.3f}")
    fig.tight_layout()
    out_path = OUT_DIR / "d3_pooled_comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved {out_path}")
    return effect_sizes, control_corr


if __name__ == "__main__":
    pooled_comparison()
