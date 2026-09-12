"""SDF-crop conditioning experiment: same DDIM training loop as train.py,
but the last channel (boundary_sdf crop, from build_patches_boundary.py) is
treated as clean spatial conditioning concatenated onto the noisy input at
every step, never noised and never predicted -- only the first N_TARGET_CH
channels (sdf/orientation/size) are the actual denoising target. This is the
one variable this experiment isolates: does the model pick up on a real
positional signal, versus train.py's baseline which has no such input at
all.
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
CKPT_PATH = OUT_DIR / "ddim_ckpt_boundary.pt"
N_TARGET_CH = 3  # was 4 -- radial_distance dropped (see field.py); boundary_sdf
# replaces it as conditioning rather than something the model reproduces


STATS_SAMPLE = 256  # patches used to estimate normalization stats -- see train_baseline_do256.py


def load_patches():
    """mmap, not a full read -- see train_baseline_do256.py's load_patches
    for why (2.7GB array here; the training loop only needs one batch at a
    time, not the whole thing resident)."""
    fields = np.load(DATA_DIR / "patch_fields_boundary.npy", mmap_mode="r")  # (N, H, W, 4)
    conds = np.load(DATA_DIR / "patch_conds_boundary.npy")    # small (N, latent_dim), fine in RAM
    return fields, conds


def train(steps: int = 1000, batch_size: int = 16, lr: float = 2e-4, num_train_timesteps: int = 1000,
          base_ch: int = 48, resume: bool = True):
    """`steps` is how many NEW steps to run THIS call, not a target total --
    call repeatedly (e.g. steps=1000 each time) to train in resumable
    chunks. Each call is a fresh, short-lived process that exits cleanly,
    which avoids the MPS memory accumulation observed across several
    long-lived processes in a row this session, and gives natural
    checkpoints to verify against (loss, memory) rather than committing to
    one multi-hour blocking run. Pass resume=False to discard any existing
    checkpoint and start over."""
    dev = device()
    fields, conds = load_patches()
    n = fields.shape[0]
    print(f"loaded {n} patches (mmap), field shape {fields.shape[1:]}, cond dim {conds.shape[1]}")

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

    conds_t = torch.tensor(conds, dtype=torch.float32)
    in_channels = fields.shape[-1]  # N_TARGET_CH target + 1 conditioning channel (boundary_sdf)

    np.savez(OUT_DIR / "field_scaler_boundary.npz",
              target_mean=target_mean.numpy(), target_std=target_std.numpy(),
              boundary_mean=boundary_mean.numpy(), boundary_std=boundary_std.numpy())

    model = FieldUNet(in_channels=in_channels, out_channels=N_TARGET_CH,
                       cond_dim=conds_t.size(1), base_ch=base_ch).to(dev)
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

    for i in range(steps):
        step = start_step + i
        idx = np.sort(np.random.randint(0, n, size=batch_size))  # sorted: friendlier mmap access pattern
        batch = torch.tensor(np.asarray(fields[idx]), dtype=torch.float32).permute(0, 3, 1, 2)
        x0 = ((batch[:, :N_TARGET_CH] - target_mean) / target_std).to(dev)
        boundary = ((batch[:, N_TARGET_CH:] - boundary_mean) / boundary_std).to(dev)
        cond = conds_t[idx].to(dev)

        noise = torch.randn_like(x0)
        t = torch.randint(0, num_train_timesteps, (batch_size,), device=dev).long()
        x_noisy = scheduler.add_noise(x0, noise, t)
        model_in = torch.cat([x_noisy, boundary], dim=1)

        pred_noise = model(model_in, t, cond)
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
    torch.save(model.state_dict(), OUT_DIR / "ddim_unet_boundary.pt")
    np.save(OUT_DIR / "ddim_train_history_boundary.npy", np.array(history))
    np.savez(OUT_DIR / "ddim_config_boundary.npz", base_ch=base_ch,
              in_channels=in_channels, out_channels=N_TARGET_CH, cond_dim=conds_t.size(1))
    print(f"saved checkpoint at step {final_step} (+ model + history) to {OUT_DIR}")
    return model


if __name__ == "__main__":
    train()
