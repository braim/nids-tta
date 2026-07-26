"""nids-toolkit — shared library for NIDS continual test-time adaptation (CTTA).

Typical usage from a notebook::

    from nids_toolkit import ExperimentConfig, run_grid, DEFAULT_DATASETS

    cfg = ExperimentConfig(scaler_type="quantile", few_shot_ratio=1e-4)
    results_df, ablation_df = run_grid(cfg)

``ExperimentConfig`` is imported eagerly and has **no third-party dependencies**,
so ``import nids_toolkit`` (or ``import nids_toolkit.config``) works without torch
installed. Every other public symbol is loaded lazily on first access (PEP 562),
which also keeps import time low and avoids importing the whole scientific stack
until it is actually used.
"""

from __future__ import annotations

import importlib

from .config import ExperimentConfig  # torch-free, always available

__version__ = "0.1.0"

# public name -> submodule that defines it (lazily imported on first access)
_LAZY = {
    # utils
    "seed_everything": "utils",
    "seeded_generator": "utils",
    "get_device": "utils",
    "log_step": "utils",
    # scaling / losses
    "FeatureScaler": "scaling",
    "FocalLoss": "losses",
    "compute_class_weights": "losses",
    "make_criterion": "losses",
    "labels_of": "losses",
    "pool_prior": "losses",
    "stream_entropy_loss": "losses",
    # adaptation
    "StochasticRestore": "adaptation",
    "SplineActivationGate": "adaptation",
    "KAN_Retention": "adaptation",
    # data
    "engineer_features": "data",
    "align_features": "data",
    "load_dataset": "data",
    "make_source_loaders": "data",
    "make_target_loaders": "data",
    "reset_loader_generator": "data",
    "build_replay_pool": "data",
    # models
    "KanAEClassifier": "models",
    "CnnAEClassifier": "models",
    "TabTransformerAEClassifier": "models",
    "FlowTransformerAEClassifier": "models",
    "build_model": "models",
    "get_trainable_params": "models",
    "get_spline_only_params": "models",
    # training
    "pretrain_source": "training",
    "evaluate": "training",
    "run_ctta": "training",
    "few_shot_baseline": "training",
    "run_phase1_pretraining": "training",
    "run_phase2_zeroshot": "training",
    # ablation
    "TypeBAblation": "ablation",
    # viz
    "plot_logit_separation": "viz",
    "visualize_kan_layer": "viz",
    "plot_importance_sankey": "viz",
    "plot_adaptation_trajectory": "viz",
    "run_all_visualizations": "viz",
    # experiment orchestration
    "run_grid": "experiment",
    "run_sequential": "experiment",
    "DEFAULT_DATASETS": "experiment",
    "DEFAULT_ARCHITECTURES": "experiment",
    "DEFAULT_SEQUENCES": "experiment",
    "DEFAULT_SEQ_ARCHS": "experiment",
    "DEFAULT_SEQ_VARIANTS": "experiment",
}


def __getattr__(name: str):
    """PEP 562 lazy attribute loading for the heavy (torch-backed) API."""
    if name == "merging":
        return importlib.import_module(".merging", __name__)
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(_LAZY) + ["ExperimentConfig", "merging", "__version__"])


__all__ = ["ExperimentConfig", "merging", "__version__", *_LAZY.keys()]
