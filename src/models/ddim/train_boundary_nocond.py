"""GVAE-conditioning ablation: identical to train_boundary.py, except `cond`
is always a constant zero vector instead of the GVAE composition latent --
isolates whether the (confirmed-defective, see gvae_reproduction_test.png:
the GVAE decoder collapses per-cell orientation/size to the population mean
even given real topology) GVAE FiLM signal was contributing to, or just
irrelevant to, the DDIM's noisy sdf output. boundary_sdf channel-concat
conditioning is unchanged -- this is the one variable being isolated.

Model architecture is untouched (FieldUNet still takes a cond_dim=32 input)
rather than restructuring it to drop FiLM entirely -- passing zeros makes
cond_mlp(zeros) a fixed learned bias, carrying no per-patch information,
which is what "no GVAE conditioning" means for this test without a riskier
architecture change.
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
CKPT_PATH = OUT_DIR / "ddim_ckpt_boundary_nocond.pt"
N_TARGET_CH = 3
COND_DIM = 32  # matches train_boundary.py's GVAE latent dim, kept for architecture parity


STATS_SAMPLE = 256


def load_patches():
    fields = np.load(DATA_DIR / "patch_fields_boundary.npy", mmap_mode="r")  # (N, H, W, 4)
    return fields


def train(steps: int = 1000, batch_size: int = 16, lr: float = 2e-4, num_train_timesteps: int = 1000,
          base_ch: int = 48, resume: bool = True):
    """See train_boundary.py's train() docstring for the chunked-resume
    rationale -- identical here. `cond` is a fixed zero vector every step,
    for every sample; no GVAE checkpoint or patch_conds file is loaded."""
    dev = device()
    fields = load_patches()
    n = fields.shape[0]
    print(f"loaded {n} patches (mmap), field shape {fields.shape[1:]}, cond=zeros (GVAE dropped)")

    rng = np.random.default_rng(0)
    stats_idx = np.sort(rng.choice(n, size=min(STATS_SAMPLE, n), replace=False))
    stats_sample = torch.tensor(np.asarray(fields[stats_idx]), dtype=torch.float32).permute(0, 3, 1, 2)
    target_sample = stats_sample[:, :N_TARGET_CH]
    boundary_sample = stats_sample[:, N_TARGET_CH:]

    target_mean = target_sample.mean(dim=(0, 2, 3), keepdim=True)
    target_std = target_sample.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    boundary_mean = boundary_sample.mean(dim=(0, 2, 3), keepdim=True)
    boundary_std = boundary_sample.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    del stats_sample, target_sample, boundary_sample

    in_channels = fields.shape[-1]

    np.savez(OUT_DIR / "field_scaler_boundary_nocond.npz",
              target_mean=target_mean.numpy(), target_std=target_std.numpy(),
              boundary_mean=boundary_mean.numpy(), boundary_std=boundary_std.numpy())

    model = FieldUNet(in_channels=in_channels, out_channels=N_TARGET_CH,
                       cond_dim=COND_DIM, base_ch=base_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_step, history = 0, []
    if resume and CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        history = list(ckpt["history"])
        print(f"resumed from {CKPT_PATH.name} at step {start_step} ({len(history)} history entries)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: base_ch={base_ch}, {n_params:,} params, in={in_channels}ch out={N_TARGET_CH}ch, "
          f"running steps {start_step}..{start_step + steps - 1}")
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    zero_cond = torch.zeros(batch_size, COND_DIM, device=dev)

    for i in range(steps):
        step = start_step + i
        idx = np.sort(np.random.randint(0, n, size=batch_size))
        batch = torch.tensor(np.asarray(fields[idx]), dtype=torch.float32).permute(0, 3, 1, 2)
        x0 = ((batch[:, :N_TARGET_CH] - target_mean) / target_std).to(dev)
        boundary = ((batch[:, N_TARGET_CH:] - boundary_mean) / boundary_std).to(dev)

        noise = torch.randn_like(x0)
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=dev).long()
        x_noisy = scheduler.add_noise(x0, noise, t)
        model_in = torch.cat([x_noisy, boundary], dim=1)

        pred_noise = model(model_in, t, zero_cond)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append((step, loss.item()))
        if step % 100 == 0 or i == steps - 1:
            print(f"step {step:5d} loss={loss.item():.4f}")

    final_step = start_step + steps
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": final_step, "history": history}, CKPT_PATH)
    torch.save(model.state_dict(), OUT_DIR / "ddim_unet_boundary_nocond.pt")
    np.save(OUT_DIR / "ddim_train_history_boundary_nocond.npy", np.array(history))
    np.savez(OUT_DIR / "ddim_config_boundary_nocond.npz", base_ch=base_ch,
              in_channels=in_channels, out_channels=N_TARGET_CH, cond_dim=COND_DIM)
    print(f"saved checkpoint at step {final_step} (+ model + history) to {OUT_DIR}")
    return model


if __name__ == "__main__":
    train()
