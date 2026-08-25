# ROADMAP2 — Structural (graph-based) cell generation, as an alternative to pixel-space diffusion

## Motivation

PLAN.md's Phase 5/6 generative approach (FieldUNet, a conditional DDIM) generates *pixels* — a 128×128, 4-channel field (sdf, orientation, size, radial_distance) that gets tessellated back into cells. This works at the patch level (validated: area/elongation/orientation effect sizes 0.04–0.22 std against real cells, per PLAN.md Phase 6 status), but the Phase 5 cross-section extension exposed the approach's ceiling:

- **No global consistency.** Patches are generated independently. The only thing tying a full cross-section together is a *real* reference image supplying the silhouette and per-tile conditioning latents — the model itself has no notion of one tile relative to another. Visible seams at every 128px tile boundary are a direct, structural consequence of this, not a tuning problem.
- **Doesn't scale to whole-structure generation.** A full stem cross-section at native resolution is thousands of tiles. Independently sampling each one (50 DDIM inference steps apiece) is expensive and only gets more so as resolution/area grows — pixel-space diffusion cost scales with area, not with cell count or structural complexity.
- **The GVAE decoder is unused** (confirmed in this session — see PLAN.md Phase 3). The GVAE only ever functions as an encoder in the current pipeline; FieldUNet is the actual generative model, operating in a completely different (pixel) output space from what the GVAE was trained to reconstruct (graph: node features + edges).

**The core idea for this roadmap:** generate structure *algorithmically/directly in graph space* — cell positions, connectivity, and per-cell attributes — rather than generating pixels and inverting them into cells via tessellation. Geometry (actual cell polygons) would then be a deterministic rendering step (e.g. Voronoi/Delaunay-style tessellation from generated centroids + attributes), not something the generative model has to learn implicitly through pixel statistics.

This is an **exploration roadmap**, not a committed plan — it should be validated incrementally against the existing pixel-based approach rather than assumed superior.

---

## Final target specification

The end goal is **infill generation**: given an arbitrary 2D boundary shape and a set of control signals, generate a full cell tessellation (graph + geometry) that fills the shape consistently with those controls — not reconstruction of a specific real specimen.

**Inputs:**
- **Boundary shape**: an arbitrary 2D region (polygon or mask) — not limited to the roughly-circular stem cross-sections in the current dataset. Analogous to the `tissue_mask` silhouette Phase 5's cross-section extension already used, but here it's a free input rather than lifted from a real reference image.
- **Control fields**, sampled at any point inside the boundary:
  - **Size gradient**: target cell size as a function of position (e.g. small near the boundary, larger toward the interior, or an arbitrary user-specified gradient) — generalizes the `radial_distance` signal the GVAE already picked up unsupervised (PLAN.md Phase 3), but here it's an explicit input rather than an inferred latent correlation.
  - **Center point(s)**: one or more origin points that the size/orientation gradient is measured relative to — generalizes the single-center radial assumption baked into the current pipeline's `radial_distance` feature (distance from one tissue centroid) to multiple or arbitrarily-placed centers.
  - Room for additional control channels later (e.g. orientation field, local density target) if the first two prove out — not committing to a fixed control schema yet.
- **Output**: a cell graph (positions, per-cell attributes, adjacency) covering the full interior of the boundary, consistent with the control fields, plus rendered geometry (R3).

**This reframes R2/R4 below**: the growth model's conditioning signal is not a GVAE latent extracted from a real reference patch (as in the current FieldUNet pipeline), but the *local value of the control field* at each candidate cell's position, plus the boundary itself (both for shape-filling and for stopping/edge behavior near the boundary). The GVAE/composition-latent conditioning approach is still relevant as a way to control *style* (what "kind" of tissue composition to generate) independent of the explicit size/center controls, but it's no longer the only or primary conditioning signal — worth deciding in R2 whether style-latent and explicit control fields are combined, or whether explicit controls alone are sufficient for this target and the composition latent is dropped.

---

## Why this direction is plausible

1. **The data is already graph-shaped.** Phase 2 already extracts exactly this representation (node features + spatial adjacency edges) for every image. Phase 3's GVAE already encodes it. The pixel round-trip (field.py → tessellate.py) exists only because Phase 5 chose to generate in pixel space — it isn't an intrinsic requirement of the data.
2. **Local cell adjacency is naturally graph-structured** — cells tile a 2D surface and touch a small number of neighbors (typically 4–8). A generative model that outputs "next cell's position + attributes + which existing cells it connects to" is a much smaller, more structured decision than "what should every pixel be."
3. **Scalability**: generating N cells autoregressively (or via a fixed-size local graph diffusion step) scales with N, the actual structural complexity, not with the number of pixels needed to represent them at some resolution. A large cross-section with the same average cell size as a small patch costs proportionally more cells, not a quadratic blow-up in pixel-grid inference cost.
4. **No seams by construction.** If generation is autoregressive/growth-based (place a new cell adjacent to already-generated ones, conditioned on local neighborhood), there's no tile boundary to seam — the whole structure is one connected graph from the start, unlike stitched-together independent DDIM patches.

---

## Why this is genuinely harder (be honest about the tradeoffs)

- **Geometry isn't free.** Pixel-space generation gets a plausible cell *shape* for free from the SDF field. Graph-space generation only gives centroid + attributes (area, elongation, orientation) + adjacency — actual cell wall geometry has to be reconstructed procedurally (e.g. weighted/Laguerre Voronoi from centroids + target areas), which is a real algorithmic step with its own fidelity ceiling. This needs validating against real cell wall shapes, not assumed to look right.
- **Variable-size graph generation is a harder ML problem than fixed-size image generation.** Diffusion on images has a fixed tensor shape; diffusion/autoregression on graphs of varying size (how many cells? when to stop growing?) needs an explicit termination/growth model. This is a known harder subfield (graph generative modeling), not a solved drop-in swap.
- **The GVAE decoder — the natural starting point for this direction — has never been validated.** PLAN.md flags this explicitly ("Not yet done: quantify GVAE decoder/round-trip reconstruction on held-out validation graphs"). Before betting this roadmap on it, its reconstruction quality needs to actually be measured, not assumed adequate because the encoder side works.
- **Loses the free consistency-checking pixel space gave for free.** Phase 4's field/tessellation round-trip was a strong, cheap validation tool (real image → field → recovered cells → compare to original). A graph-generation approach needs an equivalent — likely: does the generated graph's node-feature distribution + degree distribution match real graphs (the "graph-statistic comparison" already flagged as not-yet-done in Phase 5/6).

---

## Extracting real control fields (alignment flow, size gradient)

The control fields in the target spec above don't have to be hand-specified (a synthetic radial gradient, a manually placed center) — they can be **extracted from real images as smooth spatial fields**, then reused either as (a) ground-truth training targets for R2, or (b) a library of real, transplantable design inputs applied to a *novel* boundary shape (e.g. "generate a new arbitrary shape using the alignment flow pattern measured from a real `EH` specimen").

- **Size gradient field**: already partially available per-cell (`equivalent_diameter`/`equivalent_diameter_norm` vs. `radial_distance`, per Phase 3's features). Extend this from a single scalar-vs-radius trend to a full smooth 2D field — interpolate/smooth each real image's per-cell size values across space (e.g. Gaussian-kernel smoothing over cell centroids, or fit a smooth surface) rather than only capturing the single-center radial trend the current `radial_distance` feature assumes. This generalizes cleanly to the multi-center case in the target spec (real tissue may have local size variation not explained by distance from one centroid).
- **Cellular alignment flow field**: cell `orientation` is already extracted per-cell (Phase 2/`features.py`) but only used as a per-node scalar/pixel-painted channel (`field.py::rasterize_field`'s orientation channel) — never treated as a coherent spatial *flow*. Extract a smoothed orientation/alignment vector field (local circular-mean of neighboring cells' orientation, respecting orientation's 180°-periodic nature — plain averaging of angles is wrong here and needs the usual double-angle/circular-mean treatment) across each real image. This is the natural graph-space analogue of a fiber/flow field, and is exactly the kind of structural signal a pixel-only SDF/orientation-channel field representation was never positioned to expose as a controllable, standalone input.
- **Validation before use as a control signal**: visualize both extracted fields (quiver plot for alignment flow, heatmap for size gradient) across a handful of real DO/EH/VM images and sanity-check that they look like coherent, real spatial trends (not per-cell noise) before trusting them as training targets — a noisy or degenerate extracted field would silently corrupt R2's training data.
- **Relationship to R2's training-data requirement**: this generalizes the "derive a per-cell local size gradient label from each real image's own actual radial/size trend" bullet already in R2 below — the extracted fields *are* that per-cell label, computed properly as smooth spatial fields rather than a single scalar-vs-radius fit, and the alignment-flow field adds a second, currently-unused real signal as a control/training target alongside size.

---

## Proposed exploration phases

### R1 — Validate the unused half: GVAE decoder reconstruction quality
Before designing anything new, measure what's already there.
- Run the trained GVAE decoder on held-out real graphs (not used in training).
- Quantify: node feature reconstruction error (per feature, not just aggregate loss), edge/adjacency reconstruction accuracy (precision/recall on predicted edges vs. real adjacency).
- Decide, based on real numbers: is the existing decoder architecture good enough to build on, or does graph-space generation need a different decoder design entirely?
- **This is cheap** (no new training, just evaluation) and directly resolves the biggest unknown before committing further effort.

### R2 — Minimal viable graph generator: local growth model
Start smaller than full autoregressive whole-structure generation.
- Model: given a partial graph (some cells already placed) + the local control-field value(s) at the candidate position (target size, distance/direction to the nearest center — see "Final target specification" above) + optionally a GVAE-style composition latent for style, predict the next cell's attributes + which existing cell(s) it attaches to.
- **Training data needs a control-field target attached to every real cell**, not just the raw feature — e.g. derive a per-cell "local size gradient" label from each real image's own actual radial/size trend (already computable from existing node features: `equivalent_diameter` vs. `radial_distance`), so the model learns to *follow* a size-vs-position signal rather than only reproduce one fixed real distribution. This is what makes the trained model usable on a novel, arbitrary boundary+gradient at inference time instead of only ever regenerating something close to the training images' own layout.
- Train on real graphs by masking/removing cells and predicting them back — same spirit as Phase 3/5's train-from-real-data approach, adapted to a growth/infill task instead of full-graph VAE reconstruction.
- Validate the same way Phase 5 validated FieldUNet: pool many generated cells, compare feature distributions (area/elongation/orientation) and graph statistics (degree distribution) against real, using effect-size thresholds consistent with Phase 6's 0.5-std bar. Additionally validate the control-following behavior specifically: does generated cell size actually track the requested gradient (e.g. correlation between requested target size and generated size at each position), not just "does the overall distribution look real" — this is a new check the current pipeline's validation never needed, since FieldUNet was never asked to follow an explicit control signal.

### R3 — Geometry rendering from generated graph
Once R2 produces plausible graphs, two tiers of geometry rendering, cheapest first:
- **Fast preview tier: plain/weighted Voronoi from generated centroids + sizes.** Standard Voronoi (or area-weighted/Laguerre, so cell size targets actually influence the resulting polygon size, not just centroid spacing) is near-instant to compute and gives an immediate visual/qualitative check of a generated graph — is the layout plausible, do sizes/gradients look right, are there obvious topology problems — without waiting on a slower or more careful geometry step. Use this as the default preview during R2 training/iteration and for the R4 control-generalization tests, where many candidate structures need a quick look before any are worth rendering carefully.
- **Higher-fidelity tier (only if the fast preview shows the approach is worth pursuing further): area-constrained relaxation** (e.g. Lloyd's-algorithm-style adjustment, or fitting wall curvature) to better match real cell wall shapes, since plain Voronoi cells are always straight-edged/convex and real cell walls are not.
- Compare rendered cell wall geometry (both tiers, but especially the higher-fidelity one) against real cell shapes on the same features Phase 4 used for its round-trip validation (so results are apples-to-apples with the existing pixel-pipeline benchmark).

### R4 — Whole-structure infill generation and scale comparison
- Grow cells to fill an **arbitrary boundary shape** (not necessarily one lifted from a real reference image — a synthetic polygon/mask is a legitimate, arguably better, test since it directly probes generalization beyond the training images' own shapes) using the local model (R2) + renderer (R3), driven by explicit control fields (size gradient, center point(s)) rather than a real reference's inferred composition.
- Test control-field generalization deliberately: gradients/centers not seen in training (e.g. a center placed off-axis, or a gradient steeper than anything in the real dataset) — this is the actual point of the "arbitrary shape + explicit controls" target, and needs testing as such rather than only re-running on shapes/gradients close to the training distribution.
- Direct comparison against the existing FieldUNet cross-section approach on: (a) visual/statistical quality, (b) generation cost at scale (time/memory vs. structure size), (c) presence/absence of seam artifacts, (d) — new, since FieldUNet has no equivalent — controllability: can the pixel pipeline express an arbitrary size gradient/center at all, or does this target inherently require the graph-space approach regardless of the other comparisons?
- Only after this comparison should a decision be made about replacing vs. complementing the pixel-based pipeline — this roadmap does not assume the outcome.

---

## Relationship to existing pipeline (PLAN.md)

This is additive exploration, not a replacement of Phases 0–6.6:
- Phases 0–4 (segmentation → graph extraction → GVAE encoder → field/tessellation round-trip) are all reused as-is — R1–R4 build on the same graph representation and the same trained GVAE encoder.
- Phase 5/6's FieldUNet remains the baseline this roadmap is measured against, not something to discard preemptively.
- If R1 (decoder validation) turns up that the existing decoder is inadequate, that's a real, useful finding either way — it resolves a question PLAN.md has left open since Phase 3.

## Open questions to resolve before committing significant time
- Is procedural geometry rendering (R3) going to be visually/statistically convincing, or does cell wall shape carry information that only a learned pixel-level model captures? (Genuinely unknown — worth a quick spike before investing in R2's full training.)
- What's the actual generation-cost crossover point where graph-space generation is cheaper than tiled pixel diffusion? Needs measuring, not assuming.
- Does a growth/infill model need the full GVAE composition latent as conditioning, or does local neighborhood context alone suffice (i.e., is the global "what kind of tissue is this" signal even necessary at the per-cell generation step, or only for large-scale layout)?
