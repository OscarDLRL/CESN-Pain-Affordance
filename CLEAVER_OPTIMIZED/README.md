# CLEAVER_OPTIMIZED

V6 dual-encoder autoencoder (z_static 128 + z_dynamic 128) + CESN C161 (no explicit time).
`CLEAVER/` is the previous attempt (latent 200, time in the CESN context): frozen, nothing here imports it.
Old checkpoints and experiments: `~/CLEAVER_BACKUPS/pre_clean_20261126_181409/`.

## Layout
```
config.py                 single source of truth (paths, split, REFERENCE_FRAME=1, dims, C161)
data.py                   dataset access; the reference frame is applied only here / in extract_latents
autoencoder/model.py      V6DualEncoder (loads old V6Compact128 checkpoints)
autoencoder/losses.py     V6 objective
autoencoder/train.py      two-stage training (static, then dynamic with static frozen)
autoencoder/extract_latents.py
cesn/model.py             CESN C161, latent loading, normalisation, context inversion
cesn/train.py             fit + oracle + inversion
analysis/reconstruction.py  static / settle / moving-region errors + panels
experiments/ae_benchmark.py AE candidates comparison
outputs/                  (git-ignored) autoencoder/<run>/{autoencoder.pt, latents.h5, cesn_c161/, analysis/}
```

## Reference frame
The cleaver drops ~14.7 mm between frame 0 and 1 (physics settling) and is static
afterwards until grasped. Frame 0 as reference puts that drop into every delta
(knife edges "move" with the arm). Everything uses frame 1; sequences are frames 1..222.

## Pipeline (run from the repository root)
```
python -m CLEAVER_OPTIMIZED.autoencoder.train --run v6_ref1
python -m CLEAVER_OPTIMIZED.autoencoder.extract_latents --run v6_ref1
python -m CLEAVER_OPTIMIZED.cesn.train --ae-run v6_ref1                 # validation
python -m CLEAVER_OPTIMIZED.analysis.reconstruction --checkpoint outputs/autoencoder/v6_ref1/autoencoder.pt
```
Smoke test: `--run smoke --smoke` on train.
