"""D4a: DDIM sampling with cell positions held fixed (seeded by
synthetic.py's blue-noise process) while the other 11 attribute dims are
freely diffused, conditioned on control-field values sampled at each fixed
position. Positions aren't simply pasted in once -- per RePaint (Lugmayr et
al.), the known region is re-noised to the CURRENT step's noise level and
overwritten into x before every model call, so it stays consistent with
however noisy the free dims currently are; a single one-shot overwrite
would let position and attributes drift to mismatched noise levels across
steps.
"""
import torch
from diffusers import DDIMScheduler


@torch.no_grad()
def generate_fixed_positions(model, known_pos_std: torch.Tensor, ctrl_std: torch.Tensor,
                              pos_idx: list[int], mean: torch.Tensor, std: torch.Tensor,
                              dev, num_train_timesteps: int, num_inference_steps: int = 50,
                              seed: int = 0) -> torch.Tensor:
    """known_pos_std: [N, 2] ALREADY-standardized (radial_distance_norm,
    angular_position) target values. ctrl_std: [N, ctrl_dim] already-
    standardized control-field values sampled at those same positions.
    Returns de-standardized generated attribute vectors [N, attr_dim], with
    the position dims exactly equal to the requested (de-standardized)
    known positions."""
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    n, attr_dim = known_pos_std.shape[0], mean.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, n, attr_dim, generator=gen).to(dev)
    known_pos_b = known_pos_std.unsqueeze(0).to(dev)
    ctrl_b = ctrl_std.unsqueeze(0).to(dev)
    active = torch.ones(1, n, device=dev)
    pos_idx_t = torch.tensor(pos_idx, device=dev)

    for t in scheduler.timesteps:
        t_b = t.expand(1).to(dev)
        noise_pos = torch.randn(known_pos_b.shape, generator=gen).to(dev)
        known_noised = scheduler.add_noise(known_pos_b, noise_pos, t_b)
        x[:, :, pos_idx_t] = known_noised

        pred_noise = model(x, t_b, ctrl_b, active)
        x = scheduler.step(pred_noise, t, x).prev_sample

    x[:, :, pos_idx_t] = known_pos_b  # exact at the end, not just noise-annealed toward it
    x = x.squeeze(0).cpu() * std.squeeze(0) + mean.squeeze(0)
    return x
