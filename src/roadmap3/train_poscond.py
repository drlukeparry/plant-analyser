"""Position-as-conditioning variant of the set-Transformer DDIM, built to
fix D4a's finding: pinning a jointly-trained model's positions post-hoc
(inpaint.py) broke control-following even on real, in-distribution
positions (0.458 -> 0.167). The fix here is architectural/training-time,
not a better sampling trick -- position (radial_distance_norm,
angular_position) is moved OUT of the diffusion target entirely and INTO
the per-token conditioning vector, concatenated with the existing 4-dim
control field (size_gradient_target, alignment flow) the same way ctrl
already is. The model is only ever asked to diffuse the remaining 11
attribute dims, conditioned on a position it's told, never one it also has
to jointly produce and later reconcile with an externally pinned value.

Reuses SetDenoiser unchanged (denoiser.py) -- it was already agnostic to
what "ctrl" contains, so widening ctrl_dim from 4 to 6 and narrowing
attr_dim from 13 to 11 needs no architecture change, only a different way
of slicing the same FixedSetSampler batches (dataset.py, also unchanged).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.graph.build_graph import NODE_FEATURE_COLS_PX, _PX_TO_NORM_COL
from src.models.gvae.train import device
from src.roadmap3.dataset import FixedSetSampler, load_graphs_with_ctrl
from src.roadmap3.denoiser import SetDenoiser
from src.roadmap3.train import masked_mse

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "roadmap3"
FEATURE_NAMES = [_PX_TO_NORM_COL.get(c, c) for c in NODE_FEATURE_COLS_PX]
POS_IDX = [FEATURE_NAMES.index("radial_distance_norm"), FEATURE_NAMES.index("angular_position")]
ATTR_IDX = [i for i in range(len(FEATURE_NAMES)) if i not in POS_IDX]


def train(steps: int = 3000, batch_size: int = 16, n_max: int = 64, lr: float = 2e-4,
          num_train_timesteps: int = 1000, hidden_dim: int = 128, n_layers: int = 4,
          n_heads: int = 4, tag: str = "norm", seed: int = 0,
          num_hops: int = 3, max_nodes: int = 300, out_suffix: str = ""):
    torch.manual_seed(seed)
    dev = device()
    print("loading graphs + extracting control fields ...")
    graphs, ctrls = load_graphs_with_ctrl(tag)
    all_x = torch.cat([g.x for g in graphs], dim=0)
    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    for g in graphs:
        g.x = (g.x - mean) / std

    all_ctrl = torch.cat(ctrls, dim=0)
    ctrl_mean = all_ctrl.mean(dim=0, keepdim=True)
    ctrl_std = all_ctrl.std(dim=0, keepdim=True).clamp(min=1e-6)
    ctrls = [(c - ctrl_mean) / ctrl_std for c in ctrls]
    print(f"loaded {len(graphs)} graphs, {sum(g.num_nodes for g in graphs)} total nodes")

    attr_dim = len(ATTR_IDX)
    ctrl_dim = ctrls[0].size(1) + len(POS_IDX)  # original 4-dim ctrl + 2-dim position
    sampler = FixedSetSampler(graphs, ctrls, n_max=n_max, num_hops=num_hops, max_nodes=max_nodes)

    model = SetDenoiser(attr_dim=attr_dim, ctrl_dim=ctrl_dim, hidden_dim=hidden_dim,
                         n_layers=n_layers, n_heads=n_heads).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model (position-conditioned): attr_dim={attr_dim}, ctrl_dim={ctrl_dim}, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    history = []
    for step in range(steps):
        x_full, ctrl, active = sampler.sample_batch(batch_size)
        x_full, ctrl, active = x_full.to(dev), ctrl.to(dev), active.to(dev)
        x0 = x_full[:, :, ATTR_IDX]
        pos = x_full[:, :, POS_IDX]  # already standardized, same scale x_full is in
        ctrl_full = torch.cat([ctrl, pos], dim=-1)

        noise = torch.randn_like(x0)
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=dev).long()
        x_noisy = scheduler.add_noise(x0, noise, t)

        pred_noise = model(x_noisy, t, ctrl_full, active)
        loss = masked_mse(pred_noise, noise, active)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append((step, loss.item()))
        if step % 200 == 0 or step == steps - 1:
            print(f"step {step:4d} loss={loss.item():.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / f"set_denoiser_poscond{out_suffix}.pt")
    np.save(OUT_DIR / f"train_history_poscond{out_suffix}.npy", np.array(history))
    np.savez(
        OUT_DIR / f"denoiser_config_poscond{out_suffix}.npz",
        attr_dim=attr_dim, ctrl_dim=ctrl_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, n_heads=n_heads, num_train_timesteps=num_train_timesteps,
    )
    np.savez(OUT_DIR / f"feature_scaler_poscond{out_suffix}.npz", mean=mean.numpy(), std=std.numpy(),
              ctrl_mean=ctrl_mean.numpy(), ctrl_std=ctrl_std.numpy())
    print(f"saved model + config + scaler to {OUT_DIR}")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train(steps=steps)
