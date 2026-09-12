"""Control arm of the SDF-crop conditioning experiment: identical training
loop to train.py, pointed at the DO-only 256px patch set (no boundary_sdf
channel) built by build_patches_baseline_do256.py, with its own output
names so it doesn't collide with train_boundary.py's checkpoint or with the
original Phase 5/6 run's outputs. Kept as a separate script (not a
parameterized train.py) for the same reproducibility reason as the build
scripts.
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
CKPT_PATH = OUT_DIR / "ddim_ckpt_baseline_do256.pt"


STATS_SAMPLE = 256  # patches used to estimate the normalization mean/std -- a full-array
# pass isn't needed for a z-score estimate, and avoids materializing all N patches at once


def load_patches():
    """mmap, not a full read -- with N patches at 256x256 this is a multi-GB
    array (2GB for this baseline's 2,048 patches), and loading it wholesale
    into RAM (plus the duplicate copies permute/normalize would make) is
    unnecessary peak memory the training loop never actually needs at once:
    each step only touches one batch's worth of patches."""
    fields = np.load(DATA_DIR / "patch_fields_baseline_do256.npy", mmap_mode="r")  # (N, H, W, 3)
    conds = np.load(DATA_DIR / "patch_conds_baseline_do256.npy")  # small (N, 32), fine in RAM
    return fields, conds


def train(steps: int = 1000, batch_size: int = 16, lr: float = 2e-4, num_train_timesteps: int = 1000,
          base_ch: int = 48, resume: bool = True):
    """Batch 16 measured directly on this machine, not carried over from the
    128px Phase 6 config: batch 32 at 256px pushes MPS forward+backward into
    ~2.5GB of extra swap per step (vs. ~0 up to batch 20).

    `steps` is how many NEW steps to run THIS call, not a target total --
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
    mean = stats_sample.mean(dim=(0, 2, 3), keepdim=True)
    std = stats_sample.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    del stats_sample

    conds_t = torch.tensor(conds, dtype=torch.float32)
    np.savez(OUT_DIR / "field_scaler_baseline_do256.npz", mean=mean.numpy(), std=std.numpy())

    in_channels = fields.shape[-1]  # was hardcoded 4 -- stale from before radial_distance was
    # dropped from the target field (see field.py); now reads the real channel count from data
    model = FieldUNet(in_channels=in_channels, cond_dim=conds_t.size(1), base_ch=base_ch).to(dev)
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
    print(f"model: base_ch={base_ch}, {n_params:,} params, running steps {start_step}..{start_step + steps - 1}")
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    for i in range(steps):
        step = start_step + i
        idx = np.sort(np.random.randint(0, n, size=batch_size))  # sorted: friendlier mmap access pattern
        batch = torch.tensor(np.asarray(fields[idx]), dtype=torch.float32).permute(0, 3, 1, 2)
        x0 = ((batch - mean) / std).to(dev)
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
        if step % 100 == 0 or i == steps - 1:
            print(f"step {step:5d} loss={loss.item():.4f}")

    final_step = start_step + steps
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": final_step, "history": history}, CKPT_PATH)
    torch.save(model.state_dict(), OUT_DIR / "ddim_unet_baseline_do256.pt")
    np.save(OUT_DIR / "ddim_train_history_baseline_do256.npy", np.array(history))
    np.savez(OUT_DIR / "ddim_config_baseline_do256.npz", base_ch=base_ch,
              in_channels=in_channels, cond_dim=conds_t.size(1))
    print(f"saved checkpoint at step {final_step} (+ model + history) to {OUT_DIR}")
    return model


if __name__ == "__main__":
    train()
