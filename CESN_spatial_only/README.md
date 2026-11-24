# CESN Spatial-Only

Alternative formulation of the Pain Affordance experiment.

## Scientific idea
The CESN does not receive SAFE/PAIN as context.
It receives only:

[static PCA16, grasp_x, grasp_y, causal time index]

Total context dimension: 19.

During inversion, only grasp (x,y) is optimized.
The discrete affordance is then derived from geometry:

- relative grasp X >= 0 -> handle -> SAFE
- relative grasp X < 0 -> blade -> PAIN

## Reused from the original CESN package
- `CESN.core.cesn.ContextualESNTorch`
- native256 latent200 dataset
- same 70/11 paired split
- PCA16
- reservoir 1500
- time index = frame_index / 100

## Separation
All new results are stored only under:
`CESN_spatial_only/outputs/`

This keeps the original `CESN/outputs/` untouched.
