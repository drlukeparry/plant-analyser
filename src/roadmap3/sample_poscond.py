"""Sampling for the position-conditioned model (train_poscond.py). No
RePaint/pinning trick needed -- position is a plain conditioning input the
model was actually trained on, so it's just concatenated into ctrl like any
other conditioning value and a standard DDIM reverse loop runs over the
remaining 11 attribute dims only.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.gvae.train import device
from src.roadmap3.dataset import load_graphs_with_ctrl, sample_token_subgraph
from src.roadmap3.denoiser import SetDenoiser
from src.roadmap3.sample import COMPARE_FEATURES, SIZE_FEATURE
from src.roadmap3.train_poscond import ATTR_IDX, FEATURE_NAMES, OUT_DIR, POS_IDX


def load_model(suffix: str = ""):
    dev = device()
    config = np.load(OUT_DIR / f"denoiser_config_poscond{suffix}.npz")
    model = SetDenoiser(
        attr_dim=int(config["attr_dim"]), ctrl_dim=int(config["ctrl_dim"]),
        hidden_dim=int(config["hidden_dim"]), n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
    ).to(dev)
    model.load_state_dict(torch.load(OUT_DIR / f"set_denoiser_poscond{suffix}.pt", map_location=dev))
    model.eval()
    scaler = np.load(OUT_DIR / f"feature_scaler_poscond{suffix}.npz")
    mean = torch.tensor(scaler["mean"])
    std = torch.tensor(scaler["std"])
    ctrl_mean = torch.tensor(scaler["ctrl_mean"])
    ctrl_std = torch.tensor(scaler["ctrl_std"])
    num_train_timesteps = int(config["num_train_timesteps"])
    return model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev


@torch.no_grad()
def generate(model, pos_raw: torch.Tensor, ctrl_raw: torch.Tensor, mean, std, ctrl_mean, ctrl_std,
             dev, num_train_timesteps: int, num_inference_steps: int = 50, seed: int = 0) -> np.ndarray:
    """pos_raw: [N, 2] raw (radial_distance_norm, angular_position).
    ctrl_raw: [N, 4] raw control-field values at those positions. Returns
    the full de-standardized [N, 13] attribute vector (FEATURE_NAMES
    order), with the position columns exactly equal to `pos_raw` (never
    generated, always the requested input) and the other 11 columns the
    model's actual diffused output."""
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    pos_mean, pos_std = mean[0, POS_IDX], std[0, POS_IDX]
    attr_mean, attr_std = mean[0, ATTR_IDX], std[0, ATTR_IDX]
    pos_std_t = (pos_raw - pos_mean) / pos_std
    ctrl_std_t = (ctrl_raw - ctrl_mean) / ctrl_std
    ctrl_full = torch.cat([ctrl_std_t, pos_std_t], dim=-1).unsqueeze(0).to(dev)

    n = pos_raw.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, n, len(ATTR_IDX), generator=gen).to(dev)
    active = torch.ones(1, n, device=dev)

    for t in scheduler.timesteps:
        t_b = t.expand(1).to(dev)
        pred_noise = model(x, t_b, ctrl_full, active)
        x = scheduler.step(pred_noise, t, x).prev_sample

    attr_gen = x.squeeze(0).cpu() * attr_std + attr_mean

    full = torch.zeros(n, len(FEATURE_NAMES))
    full[:, POS_IDX] = pos_raw
    full[:, ATTR_IDX] = attr_gen
    return full.numpy()


def pooled_comparison(n_subgraphs: int = 60, seed: int = 1):
    """Mirrors sample.py's pooled_comparison exactly (same subgraph draw
    style, same effect-size formula, same features) so the two models'
    numbers are directly comparable, not artifacts of different sample
    sizes -- a single small ad hoc draw showed a real_std small enough to
    make the effect-size ratio wildly unstable, this pools over enough
    subgraphs to avoid that."""
    model, mean, std, ctrl_mean, ctrl_std, num_train_timesteps, dev = load_model()
    graphs, ctrls = load_graphs_with_ctrl("norm")
    for g in graphs:
        g.x = (g.x - mean) / std

    torch.manual_seed(seed)
    real_all, gen_all = [], []
    size_target_all, size_gen_all = [], []
    for i in range(n_subgraphs):
        gi = torch.randint(0, len(graphs), (1,)).item()
        x, c = sample_token_subgraph(graphs[gi], ctrls[gi], num_hops=3, max_nodes=64)
        if x.size(0) < 8:
            continue
        x_real = (x * std + mean).numpy()
        pos_raw = torch.tensor(x_real[:, POS_IDX], dtype=torch.float32)
        x_gen = generate(model, pos_raw, c, mean, std, ctrl_mean, ctrl_std, dev,
                          num_train_timesteps, seed=seed * 1000 + i)
        real_all.append(x_real)
        gen_all.append(x_gen)
        size_target_all.append(c[:, 0].numpy())
        size_gen_all.append(x_gen[:, FEATURE_NAMES.index(SIZE_FEATURE)])

    real = np.concatenate(real_all, axis=0)
    gen = np.concatenate(gen_all, axis=0)
    size_target = np.concatenate(size_target_all)
    size_gen = np.concatenate(size_gen_all)

    print(f"\npooled over {len(real_all)} subgraphs, {real.shape[0]} real / {gen.shape[0]} generated cells")
    print(f"{'feature':<24} {'real mean':>12} {'gen mean':>12} {'real std':>12} {'effect size':>12}")
    for name in COMPARE_FEATURES:
        idx = FEATURE_NAMES.index(name)
        r_mean, r_std = real[:, idx].mean(), real[:, idx].std()
        g_mean = gen[:, idx].mean()
        effect = abs(g_mean - r_mean) / max(r_std, 1e-8)
        print(f"{name:<24} {r_mean:>12.4g} {g_mean:>12.4g} {r_std:>12.4g} {effect:>12.3f}")

    control_corr = float(np.corrcoef(size_target, size_gen)[0, 1])
    print(f"\ncontrol-following: corr(size_gradient_target, generated {SIZE_FEATURE}) = {control_corr:.3f}")
    return control_corr


if __name__ == "__main__":
    pooled_comparison()
