# CESN Context Inversion

Reproducible experiments on spatial-context inversion for a **frozen Contextual Echo State Network (CESN)**, using ManiSkill demonstrations of a Panda robot performing **Pick** and **Push**. This repository produces experimental measurements, CSV tables and PNG figures; scientific interpretation is written separately.

The forward CESN predicts nine robot joint positions from task, cube position and normalized trajectory time:

```text
[task, cube_x, cube_y, t]  ->  q(t) ∈ R^9
```

During inversion, the model weights remain fixed. Given the observed prefix `q[0:K]` and trajectory length `T`, the selected method evaluates discrete Pick/Push hypotheses, optimizes cube XY with batched **L-BFGS** (up to 60 iterations, four initial XY points), and selects the lowest observed-prefix reconstruction loss. The 9 predicted positions include seven arm joints (radians) and two gripper positions (meters).

## Dataset and spatial protocol

Training and validation cube positions lie within the **20 × 20 cm inner support**:

```text
X ∈ [-0.15, 0.05] m
Y ∈ [-0.10, 0.10] m
```

Test interpolation uses unseen positions inside this support; test extrapolation uses positions outside it but within the feasible workspace. The current dataset contains **140 training, 20 validation, 40 test interpolation and 80 test extrapolation trajectories**. These are **trajectory counts**, not unique context locations. `Data_makers/build_cesn_spatial_context_dataset.py` defines the generator and its actual default bounds; its CLI arguments can override them. The spatial split visualization is `00_spatial_context_split.png`.

The main dataset is `datasets/cesn_spatial_context_v1/multimodal_spatial_context_v1.h5`. Its source summaries are under `datasets/cesn_spatial_context_v1/`.

## Environments

Run commands from the repository root. The project uses **two separate Conda environments**:

```bash
# Data generation / ManiSkill
conda create -n maniskill-data python=3.10 -y
conda activate maniskill-data
python -m pip install -r requirements-maniskill-data.txt

# Separate environment for frozen-CESN evaluation / optimization
conda create -n maniskill-esn python=3.10 -y
conda activate maniskill-esn
python -m pip install -r requirements-maniskill-esn.txt
```

Python 3.10 is a documented setup choice; for strict reproduction, match the Python and dependency versions used in the original experiments. GPU experiments require a compatible PyTorch/CUDA installation.

## Reproducing the experiments

### A. Create or obtain the dataset

```bash
conda activate maniskill-data
python Data_makers/build_cesn_spatial_context_dataset.py --help
python Data_makers/build_cesn_spatial_context_dataset.py
```

For an **exact** recreation of the published dataset, use the original generator arguments/seed and dataset artifact. Regenerating from defaults alone does not guarantee identical trajectories.

### B. Train the CESN (or restore the original frozen model)

Model training is a **separate step**; `run_all_experiments.sh final` does not train the network. Switch to the model environment after generating the dataset:

```bash
conda activate maniskill-esn
python3 -m CESN.core.trainingCESN --help
```

The training script accepts `--neurons` and `--runs` (verified using `--help`). To train **one CESN with 900 reservoir neurons**, run from the repository root:

```bash
conda activate maniskill-esn
python3 -m CESN.core.trainingCESN --neurons 900 --runs 1
```

Check the training configuration in `CESN/core/parametersCESN.py` and the training script before starting. These flags control reservoir size and number of runs, **not necessarily the random seed or all hyperparameters**. The pipeline expects the trained model at `CESN/outputs/models/N_900/run_01/model.npz`; verify that path after training. Training may overwrite existing model files depending on the script's save logic, so preserve the original frozen model before retraining. For exact numerical reproduction, record the seed, data split, regularization, leak rate, dependency versions, and model checksum.

The final evaluation requires:

```text
datasets/cesn_spatial_context_v1/multimodal_spatial_context_v1.h5
CESN/outputs/models/N_900/run_01/model.npz
```

For exact reproduction of the reported numerical results, **restore the original model and dataset artifacts**; training another model can produce different weights and metrics even if the source code is identical.

### C. Run the frozen-model evaluation

```bash
conda activate maniskill-esn
bash run_all_experiments.sh status  # Check required inputs and saved outputs
bash run_all_experiments.sh quick   # Small GPU smoke test; separate *_smoke folders
bash run_all_experiments.sh final   # Missing full experiments + figures/tables
bash run_all_experiments.sh report  # Rebuild figures/tables from saved data; no optimization
```

The default `bash run_all_experiments.sh` mode is **`status`**, not `final`. The full runner reuses an experiment when its expected detail CSV already exists; this is a **file-existence check**, not a check that all expected cases are present. Audit the saved row counts before treating an interrupted run as complete. To choose a batch size and compute device:

```bash
CESN_BATCH_CASES=64 CESN_DEVICE=cuda bash run_all_experiments.sh final
```

Changing the batch size **may change inferred contexts**: L-BFGS uses a shared curvature history across candidates in a batch. Record the batch size used to obtain any reported numbers.

Optional Cartesian TCP plots (ManiSkill forward kinematics; may require the ManiSkill environment/dependencies):

```bash
CESN_TASK_SPACE=1 bash run_all_experiments.sh final
# Or run only the two qualitative cases (this re-runs their inversions):
python3 -m CESN.analysis.final.movement_completion --device cuda --task-space \
  --out-dir CESN/outputs/final_lbfgs/movement_completion_with_fk
```

### D. Test any exact number of observed frames (custom K)

`task_emergence.py` evaluates an **exact frame count K**, not a fraction of the trajectory. It applies the selected Method A (discrete Pick/Push + L-BFGS 60), recovering the unknown task and cube XY from `q[:K]`. `--full` selects all TEST cases; without it, the default is an eight-case smoke test. The requested K is clipped to trajectory length when necessary.

Single K (example: **30 observed frames**, all TEST trajectories):

```bash
python3 -m CESN.analysis.final.task_emergence \
  --full --k-values 30 --batch-cases 64 --device cuda \
  --out-dir CESN/outputs/final_lbfgs/task_emergence_K30
```

Multiple exact frame counts in one run:

```bash
python3 -m CESN.analysis.final.task_emergence \
  --full --k-values 1 5 10 20 30 40 --batch-cases 64 --device cuda \
  --out-dir CESN/outputs/final_lbfgs/task_emergence_custom_K
```

For a fast check on only eight TEST trajectories, **omit** `--full` and specify a separate output directory, e.g. `--out-dir CESN/outputs/final_lbfgs/task_emergence_K30_smoke`. The script writes detail, summary, batches, and metadata files. Avoid using the directory of a completed full experiment unless you intend to replace it.

To inspect a *single named trajectory* at an exact K, the loss-landscape diagnostic supports `--trajectory` and `--K`:

```bash
python3 -m CESN.analysis.final.loss_landscape \
  --trajectory pick_interpolation_0080 --K 30 --device cuda --verify \
  --out-dir CESN/outputs/final_lbfgs/custom_K30_landscape
```

This produces a single-trajectory **loss landscape and optimization trace**, not the full TEST-set metric table.

### E. Recreate the **Adam vs. L-BFGS** tables/plots from measured data

```bash
python3 -m CESN.analysis.comparisons.report_optimizer_benchmark
```

This reads the **existing original benchmark** files:

```text
CESN/outputs/reports/optimizer_benchmark_full/detail.csv
CESN/outputs/reports/optimizer_benchmark_full/batches.csv
```

It checks paired observations and computes actual measured runtime **once per batch**, accuracy, XY error, future RMSE and gradient-evaluation counts. It does **not** rerun Adam or L-BFGS. To reproduce benchmark timings from scratch, rerun the corresponding benchmark source under `CESN/analysis/comparisons/` with the original settings and hardware, then regenerate these tables. The comparative benchmark is exploratory and is not necessarily identical to the final batch-64 evaluation protocol.

## Experiments and code

| Purpose | Source |
| --- | --- |
| Dataset generation | `Data_makers/build_cesn_spatial_context_dataset.py` |
| CESN implementation/training | `CESN/core/` |
| Differentiable rollout | `CESN/rollout/` |
| Canonical inversion backend | `CESN/inversion/` |
| Final known-task, unknown-task, exact-K task emergence, noise, Hessian, movement completion, baselines, loss landscape | `CESN/analysis/final/` |
| Adam/L-BFGS and Method A/Method B benchmarks | `CESN/analysis/comparisons/` |
| Original Adam experiment sources | `CESN/analysis/legacy_adam/` |
| NumPy/PyTorch parity and dataset checks | `CESN/analysis/check_torch_equivalence_final.py`, `CESN/analysis/check_spatial_context_dataset.py` |

`task_emergence.py` uses an **exact observed frame count K**, not an observed fraction. The loss-landscape Figure 11 traces optimizer **closure evaluations**, including line-search trials, and is a single-case illustration rather than a full-batch optimizer trace. Hessian eigenvalues characterize local loss curvature, not calibrated confidence. For noise experiments, joint positions have mixed physical units.

## Outputs and reproducibility

```text
CESN/outputs/models/                             Frozen model
CESN/outputs/testing/                            Historical forward evaluation
CESN/outputs/reports/                            Historical benchmarks (input CSVs)
CESN/outputs/final_lbfgs/<experiment>/            Final experiment detail CSV and metadata
CESN/outputs/final_lbfgs/final_report/tables/     Consolidated numeric CSV tables
CESN/outputs/final_lbfgs/final_report/figures/    Figures (PNG)
CESN/outputs/final_lbfgs/final_report/report_manifest.json
CESN/outputs/reports/                          Generated benchmark CSVs (not tracked by Git)
```

To check the input artifacts used in your run:

```bash
sha256sum \
  datasets/cesn_spatial_context_v1/multimodal_spatial_context_v1.h5 \
  CESN/outputs/models/N_900/run_01/model.npz

git rev-parse HEAD
python3 -m CESN.analysis.final.check_inversion_parity --device cuda
```

**Data availability:** Source code and selected benchmark CSVs may be versioned; ignored model/dataset/full benchmark files must be preserved or provided separately for **exact** reproduction. The selected method and optimizer settings were evaluated on the TEST split; results are exploratory rather than an independent post-selection holdout assessment.