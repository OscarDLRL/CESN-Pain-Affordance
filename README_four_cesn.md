# Four CESN experiments (Lightweight128)

Install from project root: `unzip ~/Downloads/CLEAVER_four_CESN.zip -d .`

Run in `maniskill-esn`:

```
mkdir -p CLEAVER_OPTIMIZED/logs
set -o pipefail
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u -m CLEAVER_OPTIMIZED.run_four_cesn 2>&1 | tee -a CLEAVER_OPTIMIZED/logs/four_cesn.log
```

Same command resumes completed models and completed prefix batches. Do not run two jobs in the same output folder concurrently. A different configuration requires a distinct `--run-name`. `--max-iter 80 --run-name four_cesn_80iter` is a stricter follow-up. `--matmul-precision highest` disables reduced internal matmul precision (new run name required).

Shared experimental settings: 1500 reservoir units, spectral radius .95, leak .25, ridge .5, seed 42, 5% connectivity, zero reservoir state, no time input. Static PCA16 is trained on training only, scores standardized on training only. XY normalized with training bounds. Dynamic latent centered per component and scaled by a shared training scalar. Four readouts trained on 48 pairs; validation 6 pairs, test 9 pairs. Common recurrent weights and corresponding context columns shared across variants. Feedback models receive previous normalized latent during fit and prefix inversion; after the observed prefix, prediction is autonomous. The first feedback input is zero in normalized space. True label/XY used for supervised fit and separately labeled oracle metrics, not inversion. Nine restart positions include center and eight seeded random starts; both labels searched over identical training bounds. L-BFGS has a joint line search across independent candidate variables; per-candidate minima retained across evaluations. Default max 40 iterations per batch, 80 max evaluations (subject to optimizer behavior). Timings are synchronized batch inversion elapsed and amortized per trajectory, not single-query latency. Forward RMSE uses original latent units. Future RMSE missing at full prefix; nonfinite future predictions explicitly counted. Precontact uses first PAIN edge impulse in each pair, n observed frames means indices 0..n-1; proxy only, not actual pain. Test is for final comparison; use validation for any selection/tuning. New implementation, not an exact reproduction of older CESN parameters or optimizer. No physical rollout, dense-grid/nearest baseline, or reservoir sweep included in this package.

Outputs under `CLEAVER_OPTIMIZED/outputs/four_cesn_lightweight128/`: model checkpoints, per-prefix JSON progress, per-trajectory and summary CSVs (validation/test), comparison.csv, oracle_forward.csv, normalization.json, manifest.json, PNG comparisons. Original AE untouched. Manifest checks latent hash, script, split, settings. If GPU OOM, start another named run with `--trajectory-batch 1` (changes joint line search and timing). Tested with synthetic dataset on CPU (32 reservoir units, 2 restarts, 2 iterations, 1 and 3 observed frames), including resume. Actual dataset and full GPU configuration not executed here.
