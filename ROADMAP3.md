# ROADMAP3 — Set/graph diffusion: DDIM + Transformer denoiser for cell infill

## Relationship to ROADMAP2

ROADMAP2's R2 built an **autoregressive/masked-infill GNN** (`infill_model.py`, 3× GATConv) that predicts one node's attributes at a time from real, already-known topology. It works, modestly, but never touches diffusion and depends on topology being given rather than generated.

This roadmap tests a different mechanism for the same target (see ROADMAP2's "Final target specification" — boundary + size gradient + center/direction field in, full cell tessellation out): **run DDIM directly on the set of cells**, denoising all of them together via a Transformer, instead of growing them one at a time via a GNN or rasterizing them as pixels (Phase 5/6's FieldUNet). Not a replacement for R2 — a third point of comparison alongside R2 (GNN growth) and Phase 6 (pixel DDIM).

Key framing points from the request this roadmap is built against:
- **The denoiser does not produce an image, a patch, or anything upscaling-shaped.** Its output is a list of cells (attribute vectors). Rendering that list into actual polygon geometry is Phase R3's job (Voronoi/Laguerre from centroids + sizes) — a separate, deterministic downstream step, same separation ROADMAP2 already committed to.
- **Denoiser architecture is a Transformer with self-attention, not a UNet.** A cell tessellation is an *unordered set* of cells — attention over a token set is the natural inductive bias (permutation-equivariant, no arbitrary node ordering or fixed adjacency to bake in), unlike a GNN (needs edges up front) or a patch UNet (needs a pixel grid).
- **Conditioning is spatial and per-cell, ControlNet-style**: boundary shape, size gradient, direction/alignment field are all sampled *at each cell's position* and fed in per-token, not as one global vector — the graph-space analogue of ControlNet feeding a spatial conditioning map into a UNet's residual blocks.

## Why diffusion over autoregressive growth here

- **Global consistency in one shot.** R2's GNN grows cells sequentially from local neighborhood context only — nothing enforces a long-range consistent layout except however far growth has propagated. A set-level DDIM denoises the whole token set jointly at every step; each cell's final attributes are informed by attention over the *entire* set (and the full conditioning field) from the start, not just already-placed neighbors. This is the same "no seams by construction" argument ROADMAP2 makes for graph generation in general, pushed one level further: no growth-order artifacts either.
- **Matches the existing pipeline's mechanism.** Phase 5/6 already validated the DDIM/`diffusers.DDIMScheduler` training-and-sampling loop on this exact codebase and dataset. Reusing the scheduler, just swapping the denoiser's input/output tensor shape from "pixel grid" to "token set," is a small, well-precedented change — lower-risk than inventing a new generative mechanism from scratch.
- **Cardinality is still awkward either way.** Diffusion doesn't remove R2's hardest open problem (how many cells, when to stop) — see Open Questions below. This roadmap treats that as a shared unsolved problem, not a reason to prefer one mechanism over the other; the comparison itself is the point.

## Representation

- Each cell = one token: **position** `(radial_distance_norm, angular_position)` or raw `(x, y)` in tissue-normalized coordinates (Phase 3's existing normalization) concatenated with the **13 existing node attributes** (`area_norm`, `elongation`, `orientation`, `solidity`, etc. — same feature set R1/R2 already use). No image channels anywhere.
- **Adjacency is not part of the generative target.** Dropped entirely, same call ROADMAP2's R3 already made for the GNN path: edges are recovered deterministically from generated positions via Voronoi/Delaunay once geometry is rendered. This sidesteps R2's still-deferred "attachment/topology prediction" problem rather than solving it — worth stating plainly, not implicitly assuming diffusion solves it for free.
- **Fixed-cardinality token set (N_max) with a padding mask**, standard practice for point-set/set diffusion. Real subgraphs (reusing Phase 3's k-hop sampler) supply between a few and N_max real cells; pad with a learned pad token and an explicit active/inactive flag per token, loss computed only over active tokens. Variable cell-count generation (arbitrary region, arbitrary target density) is handled at the *sampling* stage, not by training a variable-length model — see D4.

## Denoiser architecture (D2)

DiT-style (Diffusion Transformer) design, adapted for a set instead of a pixel grid:

- **Backbone:** standard Transformer encoder, full self-attention across all N_max tokens (no causal mask — this is a set, not a sequence).
- **Per-token input:** noisy attribute vector (13-dim) + the token's `(x, y)` position (positions are conditioning, not denoised — see D4 for why) + sinusoidal embedding of `(x, y)` + diffusion-timestep embedding, combined via AdaLN-Zero (DiT's conditioning mechanism) rather than concatenation, so timestep conditioning modulates every block cleanly.
- **ControlNet-style spatial conditioning, per token:**
  - `size_gradient_target` sampled at `(x, y)` from the input size field (reuses `control_fields.py`'s extraction logic, applied to a synthetic or transplanted-real field at inference time).
  - `alignment_flow_angle` / `alignment_flow_coherence` sampled at `(x, y)` (same source).
  - signed distance from `(x, y)` to the input boundary polygon — tells the model "near the edge, shrink/orient tangentially" vs. "interior."
  - These are concatenated into a per-token conditioning vector and injected the same way the timestep embedding is (AdaLN-Zero), zero-initialized on the conditioning pathway specifically (ControlNet's own trick) so the base denoiser can't be destabilized early in training before the conditioning signal is useful.
- **Scheduler:** reuse `diffusers.DDIMScheduler` unchanged — only the noise-prediction network's input/output shape changes, from Phase 5/6's `[B, 4, H, W]` field to `[B, N_max, 13]` (+ mask).
- **Start small**, per this repo's established pattern (Phase 5 started at ~1M params before Phase 6 scaled to 2.1M) — a first prototype should prove the loop end-to-end on a small model/short run before any scale-up.

## Proposed phases

### D1 — Fixed-cardinality tokenized dataset
**Status: implemented.** `src/roadmap3/dataset.py` — `FixedSetSampler` draws padded `[B, n_max, attr_dim]`/`[B, n_max, ctrl_dim]` batches + active mask from Phase 3's k-hop sampler + `control_fields.py`, reused unchanged from ROADMAP2 R2. `n_max` is a training/batching convenience only, not an architectural limit (see D2).

- Build a sampler on top of Phase 3's k-hop subgraph sampler that emits fixed-size `[N_max, 13]` attribute tensors + `[N_max, 2]` positions + active-mask, from real images across `DO`/`EH`/`VM`.
- Reuse `control_fields.py` as-is to attach per-node `size_gradient_target` / `alignment_flow_angle` / `alignment_flow_coherence` to every real cell (already validated in ROADMAP2 R2 — no rework needed).
- Pick N_max empirically from the real subgraph size distribution (same sampler ROADMAP2 already uses, num_hops=3, up to 300 nodes) — start smaller (e.g. 64–128) for the first prototype, matching the "start small" pattern, not the eventual full-structure target.

### D2 — Transformer denoiser + training loop
**Status: implemented, smoke-tested only (not a real training run).** `src/roadmap3/denoiser.py` — DiT-style `SetDenoiser`: per-token input is a token embedding (attribute projection + per-token control-field projection, added) plus AdaLN-Zero blocks modulated by the (per-batch-item, not per-token) diffusion-timestep embedding, zero-gated at init per the DiT stabilization trick. Full self-attention with a key-padding mask over inactive tokens; no positional/sequence-length constraint, so sampling (D3) doesn't need to pad to `n_max`. `src/roadmap3/train.py` reuses `diffusers.DDIMScheduler` unchanged, masked eps-prediction MSE over active tokens only. A 300-step / 168K-param smoke test (`hidden_dim=64, n_layers=2`) confirms the loop runs end-to-end: loss 0.98 → 0.20. This is proof-of-loop only, per the plan's own "start small, short run, prove the loop before scaling" bar — not evidence of generation quality.
- Implement the DiT-style architecture above (`src/roadmap3/denoiser.py`, `src/roadmap3/train.py`), reusing `diffusers.DDIMScheduler`.
- Loss: standard eps-prediction MSE, masked to active tokens only. Positions are conditioning inputs, not part of the loss target (see D4).
- First run: small model, short schedule, prove loss goes down and DDIM sampling produces non-degenerate output (same exit bar Phase 5 used) before any scale-up decision.

### D3 — Unconditional / self-conditioned sanity check
**Status: harness implemented, not yet run at a real training budget.** `src/roadmap3/sample.py` — DDIM reverse sampling conditioned on a real subgraph's own extracted control fields, pooled feature-distribution comparison (`area_norm`/`elongation`/`orientation`, same 0.5-std effect-size bar Phase 6 used) plus the control-following correlation check R2 introduced (`corr(size_gradient_target, generated equivalent_diameter_norm)`). Ran once against the 300-step smoke-test checkpoint from D2: distributional effect sizes were within bar (0.48/0.05/0.06), but control-following correlation was ≈0 (-0.005) — expected and uninformative at this training budget, not a negative result about the architecture. **Not yet done: an actual training run** (Phase 5/6-scale: thousands of steps, larger `hidden_dim`/`n_layers`) to get a real reading on whether control-following correlation rises the way R2's ablation showed for the GNN baseline.
- Before testing arbitrary novel conditioning, first confirm the model can regenerate plausible *real-distribution* cell sets when conditioned on a real image's own extracted control fields (i.e. reproduce roughly what R2/Phase 6 already validate against) — same "does this even work" gate every prior phase used before testing generalization.
- Validate with ROADMAP2's existing bar: pooled feature-distribution comparison (area/elongation/orientation) against real cells, effect-size threshold consistent with Phase 6's 0.5-std bar, plus the control-following check R2 introduced (does generated size track the requested gradient — correlation, not just distributional match).

### D4 — Position handling: the actual open design decision
Diffusion needs *some* token position to sample the spatial conditioning fields at, but position is also plausibly something the model should generate, not receive as fixed input. Two variants to build and compare, cheapest first:

- **D4a (build first): positions fixed, only attributes diffused.** Seed token positions via a blue-noise/Poisson-disk point process over the boundary polygon, local intensity set from the size-gradient field (denser where target cell size is smaller) — a deterministic, non-learned step, same spirit as R3's "geometry is a deterministic rendering step, not something the model learns implicitly." The Transformer then only denoises the 13 attributes at each fixed, pre-placed position. Simpler, avoids re-sampling spatial conditioning mid-trajectory, and directly reuses this repo's existing "decouple placement from content" philosophy (ROADMAP2 already decouples geometry-rendering from graph-generation this same way).
- **D4b (only if D4a's ceiling is reached): positions are part of the diffused vector too.** Requires re-sampling the control fields at the *current noisy position estimate* at every DDIM step (literal ControlNet-style re-conditioning per denoising step, since the conditioning signal is itself position-dependent) — meaningfully more complex, and only worth building if D4a demonstrably can't produce good layouts because it's too dependent on the blue-noise seed's quality.

### D5 — Whole-region infill + comparison against R2 and Phase 6
- Generate on a synthetic boundary polygon with a hand-specified size gradient and center(s)/direction field not seen in training (same generalization test ROADMAP2's R4 specifies).
- Three-way comparison, same metrics throughout this project: pooled feature-distribution match, control-field-following correlation, generation cost (wall-clock/memory vs. structure size — the Transformer's attention is O(N²) in token count, worth measuring explicitly since it's a real potential scaling ceiling at large N unlike R2's local GNN steps or Phase 6's fixed-tile cost), and presence/absence of layout artifacts (seams, clumping, density mismatch vs. the target gradient).
- Feed the winning representation into ROADMAP2's R3 geometry renderer (Voronoi/Laguerre from centroids + sizes) unchanged — both R2 and this roadmap are meant to be interchangeable upstream sources for the same rendering step.

## Open questions / risks, stated up front

- **Cardinality is unsolved, not sidestepped.** Fixed N_max training + a separately-seeded point process (D4a) chooses *where* cells go and implicitly *how many*, but that choice is made by the non-learned blue-noise step, not learned generation — worth being honest that this roadmap doesn't solve "learn how many cells to generate," it delegates the question to a deterministic point process, same as it delegates geometry to a deterministic renderer.
- **O(N²) attention cost at scale.** Full self-attention over thousands of tokens (a real whole-structure cross-section) may be the actual bottleneck that determines whether this beats Phase 6's tiled pixel-DDIM at scale — needs measuring (D5), not assumed favorable just because it avoids per-tile seams.
- **R1's decoder finding is a warning, not directly applicable, but relevant in spirit:** R1 found the existing GVAE decoder reconstructs position well but collapses shape/size features toward the mean. This roadmap doesn't reuse that decoder, but the same failure mode (a model that's good at the spatially-smooth part of the target and bad at the intrinsically-noisy per-cell part) is worth watching for here too, especially for `eccentricity`/`solidity`-type features that R1 and R2 both struggled with.
- **No held-out image split**, same caveat as R1/R2 — all evaluation here is in-sample until a real train/test split by image is set up.

## Dependencies

No new dependencies expected — `diffusers` (`DDIMScheduler`) and `torch` are already in `pyproject.toml`. A plain `nn.TransformerEncoder` (or a small hand-rolled pre-norm attention stack, matching the "start small" precedent of Phase 5's hand-rolled UNet rather than pulling in a full DiT library) covers D2.
