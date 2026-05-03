# DCBA Improvement Plan — Loss + Theta Encoder

## Context

Test-time cross-modal prediction path:
`graph → GINEncoder.encode → z_G → ConfigAutoEncoder.decode → theta_hat`

`z_G` must land in the same embedding space as `z_θ` for the config decoder to work. The contrastive
loss is responsible for enforcing this alignment.

Training curves show `l_reg ≈ 0.002` vs `l_supcon ≈ 3.0` — the GNN receives almost no gradient from
the regression head and is trained almost entirely by the contrastive signal.

---

## Task A — Symmetric bidirectional loss (CLIP-style)

**Problem:** Current `MultiPositiveSupConLoss` only uses `z_G` as anchors. The loss is asymmetric —
it pushes `z_G` toward `z_θ` but not the reverse. For the test-time decoding path to work, `z_G`
must be interchangeable with `z_θ` in the decoder's input space, which requires alignment from both
sides.

**Implementation:** Run the loss in both directions and sum:

- `z_G` anchors vs `z_θ` pool (existing direction)
- `z_θ` anchors vs `z_G` pool (new direction)

Same positive/negative structure applies in both directions.

---

## Task B — Deeper theta encoder

**Problem:** Single hidden-layer MLP (`9 → 64 → 32`) may produce a poorly-shaped `z_θ` latent space
even if reconstruction MSE is low. A richer encoder may produce better geometry for the contrastive
loss to pull `z_G` toward. Use funnel-shaped hidden dims for the theta encoder (gradual compression
toward the bottleneck).

**Two variants to evaluate (separate runs):**

- **B1 — Deeper MLP:** add hidden layers in a funnel shape, e.g. `9 → 128 → 64 → 32`
- **B2 — Lightweight transformer:** small transformer over the 9-dim config vector

**Note:** GINEncoder shape (`1 → 64 → 128 → 32`, expand-compress) stays unchanged.
