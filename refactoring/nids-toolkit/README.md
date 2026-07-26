# nids-toolkit

Shared library for the **NIDS continual test-time adaptation (CTTA)** experiments,
including the **AGSA** (Activation-Gated Spline Adaptation) technique. It packages
everything the `refactoring/` notebooks used to copy-paste — config, data pipeline,
models, losses, the CTTA loop, ablation, visualisation, and the two experiment
orchestrators — so a notebook is now just a config block plus one call.

## Install

```bash
pip install -q git+https://github.com/braim/nids-toolkit.git
```

This pulls the git-only `efficient-kan` dependency automatically.

> **PyPI note.** `pip install nids-toolkit` from PyPI is not yet supported because
> `efficient-kan` is a direct git dependency (not on PyPI), which PyPI rejects in a
> published package. To enable a clean `pip install nids-toolkit` later, vendor or
> publish the KAN layer, then remove the direct reference from `pyproject.toml`.

For local development:

```bash
pip install -e refactoring/nids-toolkit          # editable, with deps
# or, if the git dep is already present / offline:
pip install -e refactoring/nids-toolkit --no-deps
```

## Quickstart — the episodic grid

```python
from nids_toolkit import ExperimentConfig, run_grid, DEFAULT_DATASETS

cfg = ExperimentConfig(               # defaults reproduce the TTAExp notebook
    scaler_type="quantile",
    loss_type="weighted_ce",
    few_shot_ratio=1e-4,
    min_pool_size=100,
    sample_n=1_000_000,
)

results_df, ablation_df = run_grid(
    cfg,
    architectures=["kan", "cnn", "tab", "flowtransformer"],
    datasets=DEFAULT_DATASETS,
)
# also written to grid_results.csv / ablation_results.csv
```

`TTAExp50` is the same call with two overrides: `few_shot_ratio=5e-5, min_pool_size=50`.

## Sequential (continual) protocol

```python
from nids_toolkit import ExperimentConfig, run_sequential

seq_df = run_sequential(ExperimentConfig())   # AGSA variants across domain sequences
# written to sequential_results.csv
```

## Weight-space merging (SSR experiment)

The reusable merge primitives live in `nids_toolkit.merging`:

```python
from nids_toolkit import merging

sd_src   = merging.snap_state(source_model)
merged   = merging.agsa_routed_merge(sd_src, sd_a, sd_b, mask_a, mask_b)
soup     = merging.uniform_soup(sd_src, [sd_a, sd_b])
theta_a  = merging.interp_states(sd_src, sd_a, alpha=0.5)
```

## Configuration

Every hyperparameter is a field on `ExperimentConfig` (see `config.py`); defaults
match the `TTAExp` notebook. Because config is a plain dataclass with no torch
dependency, it can be imported and validated on its own:

```python
from nids_toolkit.config import ExperimentConfig
ExperimentConfig(loss_type="focal", tta_lr=5e-3)
```

## Tests

```bash
pytest refactoring/nids-toolkit/tests
```

The smoke test runs each architecture through pretrain → CTTA → evaluate on
**synthetic CPU data** (no Kaggle download, no GPU). It validates wiring, not
research numbers — reproduce those on a GPU with the real NF-* datasets.

## Package layout

```
src/nids_toolkit/
  config.py       ExperimentConfig (all hyperparameters; torch-free)
  utils.py        seeding, device, logging
  scaling.py      FeatureScaler (minmax / robust / quantile)
  losses.py       class weights, focal loss, criteria, robust stream entropy
  adaptation.py   StochasticRestore, SplineActivationGate (AGSA), KAN_Retention
  data.py         dataset load, feature engineering/alignment, loaders, replay pool
  models.py       KAN / TabTransformer / CNN / FlowTransformer + param selectors
  training.py     pretrain, evaluate, run_ctta, few_shot_baseline, phases
  ablation.py     TypeBAblation (Table 3 / Table 4)
  viz.py          logit separation, KAN spline viz, importance/Sankey, trajectory
  experiment.py   run_grid (episodic) + run_sequential (continual)
  merging.py      state-dict snapshot/diff/interp + merge strategies
```
