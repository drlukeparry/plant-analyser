"""D2/D3 training loop: standard eps-prediction DDIM training (same
diffusers.DDIMScheduler already validated in Phase 5/6), masked to active
(non-padded) tokens only. Per PLAN.md's established pattern and ROADMAP3's
own "start small" note: small model, short run, first prove the loop works
end-to-end (loss goes down, DDIM sampling is non-degenerate -- see
sample.py) before any scale-up decision.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.gvae.train import device
from src.roadmap3.dataset import FixedSetSampler, load_graphs_with_ctrl
from src.roadmap3.denoiser import SetDenoiser

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "roadmap3"


def masked_mse(pred: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    mask = active.unsqueeze(-1)
    n = mask.sum().clamp(min=1)
    return ((pred - target) ** 2 * mask).sum() / (n * pred.size(-1))


def train(steps: int = 3000, batch_size: int = 16, n_max: int = 64, lr: float = 2e-4,
          num_train_timesteps: int = 1000, hidden_dim: int = 128, n_layers: int = 4,
          n_heads: int = 4, tag: str = "norm", seed: int = 0):
    torch.manual_seed(seed)
    dev = device()
    print("loading graphs + extracting control fields ...")
    graphs, ctrls = load_graphs_with_ctrl(tag)
    all_x = torch.cat([g.x for g in graphs], dim=0)
    mean = all_x.mean(dim=0, keepdim=True)
    std = all_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    for g in graphs:
        g.x = (g.x - mean) / std
    print(f"loaded {len(graphs)} graphs, {sum(g.num_nodes for g in graphs)} total nodes")

    attr_dim = graphs[0].x.size(1)
    ctrl_dim = ctrls[0].size(1)
    sampler = FixedSetSampler(graphs, ctrls, n_max=n_max)

    model = SetDenoiser(attr_dim=attr_dim, ctrl_dim=ctrl_dim, hidden_dim=hidden_dim,
                         n_layers=n_layers, n_heads=n_heads).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: hidden_dim={hidden_dim}, n_layers={n_layers}, {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    history = []
    for step in range(steps):
        x0, ctrl, active = sampler.sample_batch(batch_size)
        x0, ctrl, active = x0.to(dev), ctrl.to(dev), active.to(dev)

        noise = torch.randn_like(x0)
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=dev).long()
        x_noisy = scheduler.add_noise(x0, noise, t)

        pred_noise = model(x_noisy, t, ctrl, active)
        loss = masked_mse(pred_noise, noise, active)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append((step, loss.item()))
        if step % 200 == 0 or step == steps - 1:
            print(f"step {step:4d} loss={loss.item():.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "set_denoiser.pt")
    np.save(OUT_DIR / "train_history.npy", np.array(history))
    np.savez(
        OUT_DIR / "denoiser_config.npz",
        attr_dim=attr_dim, ctrl_dim=ctrl_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, n_heads=n_heads, num_train_timesteps=num_train_timesteps,
    )
    np.savez(OUT_DIR / "feature_scaler.npz", mean=mean.numpy(), std=std.numpy())
    print(f"saved model + config + scaler to {OUT_DIR}")
    return model


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    train(steps=steps)
