# CESN - Context Inversion for Pain Affordance

Research on visual context inversion using Contextual Echo State
Networks (CESN) and robotic demonstrations generated with ManiSkill.

## Implemented Methods

### 1. Explicit Context CESN

Located in `CESN/`.

The model includes explicit SAFE/PAIN task information and spatial
grasp coordinates in its context.

### 2. Spatial-Only CESN

Located in `CESN_spatial_only/`.

The model does not receive explicit SAFE/PAIN information.
Grasp coordinates are recovered through context inversion, and
affordance classification is derived from the inferred geometry.

## Shared Configuration

| Parameter | Value |
|---|---|
| Robot | Franka Panda |
| Simulator | ManiSkill |
| RGB resolution | 256 x 256 |
| Dynamic latent dimension | 200 |
| Static PCA dimension | 16 |
| Reservoir size | 1500 |
| Ridge regularization | 0.5 |
| Spectral radius | 0.95 |
| Leak rate | 0.25 |
| Input scale | 0.50 |
| Bias scale | 0.10 |
| CESN seed | 42 |
| Scene seed | 123 |
| Training pairs | 70 |
| Test pairs | 11 |
| Validation pairs | 0 |
| Inversion optimizer | L-BFGS |

## Repository Structure

- `CESN/core/`: Autoencoder and explicit CESN implementation.
- `CESN/analysis/`: Context inversion, baselines and diagnostics.
- `CESN/outputs/`: Experimental results and figures.
- `CESN_spatial_only/`: Spatial-only implementation and results.
- `Data_makers/`: ManiSkill dataset generation.

## Environments

- `maniskill-data`: Dataset generation.
- `maniskill-esn`: Model training, inversion and analysis.

Dependencies are provided in the corresponding requirements files.

## Experimental Evidence

The repository includes selected CSV and JSON results, paper figures,
baseline comparisons, context stability diagnostics and ridge ablations.

Hyperparameter selection used test-set metrics. These results should
therefore not be interpreted as an independent post-selection evaluation.

## Data Availability

Large datasets, trained checkpoints, generated latents and videos
are excluded from Git and must be preserved separately for exact
numerical reproduction.

