"""Phase 5 training loop: conditional DDIM on field patches. Per PLAN.md:
start with a small model, short training run to prove the loop works
end-to-end before scaling. Standardizes each field channel (per-channel
z-score over the whole patch dataset) before training, same rationale as
Phase 3's feature standardization -- the four channels (sdf, orientation,
size, radial_distance) are on very different native scales.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.ddim.model import FieldUNet
from src.models.gvae.train import device

DATA_DIR = Path(__file__).resolve().parents[3] / "outputs" / "phase5"
OUT_DIR = DATA_DIR


def load_patches():
    fields = np.load(DATA_DIR / "patch_fields.npy")  # (N, H, W, C)
    conds = np.load(DATA_DIR / "patch_conds.npy")    # (N, latent_dim)
    return fields, conds


def train(steps: int = 3000, batch_size: int = 16, lr: float = 2e-4, num_train_timesteps: int = 1000):
    dev = device()
    fields, conds = load_patches()
    print(f"loaded {fields.shape[0]} patches, field shape {fields.shape[1:]}, cond dim {conds.shape[1]}")

    fields_t = torch.tensor(fields, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    mean = fields_t.mean(dim=(0, 2, 3), keepdim=True)
    std = fields_t.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    fields_t = (fields_t - mean) / std
    conds_t = torch.tensor(conds, dtype=torch.float32)

    np.savez(OUT_DIR / "field_scaler.npz", mean=mean.numpy(), std=std.numpy())

    n = fields_t.size(0)
    model = FieldUNet(in_channels=fields_t.size(1), cond_dim=conds_t.size(1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    history = []
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        x0 = fields_t[idx].to(dev)
        cond = conds_t[idx].to(dev)

        noise = torch.randn_like(x0)
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=dev).long()
        x_noisy = scheduler.add_noise(x0, noise, t)

        pred_noise = model(x_noisy, t, cond)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append((step, loss.item()))
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:4d} loss={loss.item():.4f}")

    torch.save(model.state_dict(), OUT_DIR / "ddim_unet.pt")
    np.save(OUT_DIR / "ddim_train_history.npy", np.array(history))
    print(f"saved model + history to {OUT_DIR}")
    return model


if __name__ == "__main__":
    train()
