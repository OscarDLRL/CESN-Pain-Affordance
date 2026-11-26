# CLEAVER — Final CESN experiments (Lightweight128)

This package adds two files under `CLEAVER_OPTIMIZED/` and does not modify the autoencoder, its checkpoint, the latent HDF5, or previous experiment outputs.

## Experiments

- **E20**: Raw128 absolute prediction (reference baseline).
- **E22**: Raw128 incremental prediction (reference baseline).
- **E24**: Incremental prediction with per-latent increment clipping. The clipping multiplier is selected using validation autonomous RMSE only.
- **E25**: Persistent static memory. The initial static latent is injected into the reservoir initial state once; it is not continuously supplied as an input or direct readout feature.
- **E26**: Hybrid absolute/incremental readouts sharing one reservoir and feedback state. The absolute contribution `alpha` is selected using validation autonomous RMSE only.

All experiments use the existing Lightweight128 latents, Raw128 static representation, no explicit time input, same reservoir seed, same grouped validation protocol, same optimizer settings and same observation prefixes. Test is evaluated once after validation-based selection (unless `--skip-test` is used).

## Install

From the repository root, extract the package so the two Python files land in `CLEAVER_OPTIMIZED/`:

```bash
unzip -o ~/Downloads/CLEAVER_final_experiments.zip -d .
```

## Run

```bash
cd ~/Projects/CESN_Context_Inversion_Pain_Affordance
conda activate maniskill-esn

mkdir -p CLEAVER_OPTIMIZED/logs
set -o pipefail
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLBACKEND=Agg

python -u -m CLEAVER_OPTIMIZED.run_final_experiments \
    --reservoir 1500 \
    --ridge 0.5 \
    --radius 0.95 \
    --leak 0.25 \
    --restarts 9 \
    --max-iter 40 \
    --trajectory-batch 3 \
    --validation grouped \
    2>&1 | tee CLEAVER_OPTIMIZED/logs/final_cesn_round3.log
```

Re-running the exact same command resumes completed inversion batches from JSON cache files. If you change arguments or source code, use a new `--run-name`.

## Outputs

Default output folder:

`CLEAVER_OPTIMIZED/outputs/final_cesn_round3_lightweight128/`

Important files:

- `comparison.csv`: inversion metrics by experiment, split, observed prefix and pre-contact subset.
- `oracle_forward.csv`: teacher-forced and autonomous latent RMSE and stability diagnostics.
- `hyperparameter_selection.csv`: validation sweeps for E24 clipping and E26 alpha.
- `selected_hyperparameters.json`: selected settings.
- `E*/validation/per_trajectory.csv` and `E*/test/per_trajectory.csv`: per-trajectory predictions.
- `paper/figures/`: six publication-oriented figures per split.

To regenerate figures without rerunning experiments:

```bash
python -m CLEAVER_OPTIMIZED.plot_final_figures \
    --output CLEAVER_OPTIMIZED/outputs/final_cesn_round3_lightweight128
```

## Interpretation safeguards

- E24 clipping is a stability ablation, not proof that clipping is physically optimal.
- E25 tests a specific persistent-memory mechanism: one-time initial-state injection. It is distinct from E20, which continuously drives the reservoir with static context.
- E26 alpha is selected on validation, not test.
- The dataset's PAIN label denotes simulated blade-edge contact, not physiological pain.


## CSV compatibility fix

This package fixes heterogeneous hyperparameter rows in
`hyperparameter_selection.csv`. E24 contributes `clip_multiplier`, while
E26 contributes `alpha_absolute`; the CSV writer now creates the union of
columns and leaves non-applicable cells blank.

## Resuming after the CSV fix

The corrected runner permits reuse of existing inversion cache files when
only the runner's code hash changes but the dataset hash, split, protocol,
and command-line arguments are identical. It records the previous code hash
in the manifest. If the data, split, or arguments differ, use a new run name.
