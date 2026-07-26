"""Experiment configuration.

Every hyperparameter that used to live as an UPPER_CASE module-level constant
at the top of the notebooks is now a field on :class:`ExperimentConfig`. Field
defaults reproduce the ``TTAExp`` notebook exactly, so ``ExperimentConfig()`` is
the canonical run and any variant (e.g. ``TTAExp50``) is expressed as a handful
of keyword overrides.

This module intentionally has **no third-party imports** (no torch/numpy), so it
can be imported and validated on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

LossType = Literal["ce", "weighted_ce", "focal"]
ScalerType = Literal["minmax", "robust", "quantile"]
ReconSource = Literal["stream", "pool", "both", "none"]


@dataclass
class ExperimentConfig:
    """All tunable hyperparameters for a CTTA experiment run.

    Grouped the same way the notebook config cell was. Defaults == ``TTAExp``.
    """

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Method switch ──────────────────────────────────────────────────────
    #: Headline mode. True = layer-selective (spline-only) CTTA; False = full
    #: trainable subset. Governs which run is treated as the reported result.
    spline_true: bool = True

    # ── Data ───────────────────────────────────────────────────────────────
    #: Rows sampled from each dataset (``None`` = use all).
    sample_n: int = 1_000_000
    batch_size: int = 256
    #: Larger TTA batches give more stable gradient estimates.
    tta_batch_size: int = 512

    # ── Model ──────────────────────────────────────────────────────────────
    latent_dim: int = 32

    # ── Pre-training ───────────────────────────────────────────────────────
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    weight_decay: float = 1e-4
    #: Reconstruction weight during pretraining (regularises the encoder for transfer).
    recon_w: float = 0.5

    # ── CTTA / few-shot ────────────────────────────────────────────────────
    #: Fraction of the target used as the labelled pool.
    few_shot_ratio: float = 1e-4
    #: Give the few-shot baseline the SAME optimizer-step budget as CTTA so that
    #: stream access is the only difference between the two methods.
    few_shot_match_steps: bool = False
    #: For KAN, rerun the headline CTTA with the spline gate OFF (same RNG/weights)
    #: and record source-set F1 after adaptation — validates AGSA.
    agsa_ablation: bool = True
    #: Also run few-shot + CTTA from a RANDOM init (no source pretraining),
    #: training all params — quantifies the value of source pretraining.
    scratch_baseline: bool = True
    #: Hard floor so the labelled pool never collapses.
    min_pool_size: int = 100
    #: Supervised CE weight during CTTA.
    few_shot_w: float = 1.0
    #: CTTA learning rate (higher is fine — only a small param subset updates).
    tta_lr: float = 1e-2
    tta_steps: int = 1

    # ── CTTA unsupervised objectives ───────────────────────────────────────
    #: Robust confidence-filtered entropy term on the stream.
    use_entropy: bool = True
    entropy_w: float = 1.0
    #: EATA-style reliability threshold: only samples with entropy <
    #: ent_conf_frac * ln(2) contribute.
    ent_conf_frac: float = 0.4
    #: Weight on KL(batch-mean prediction || pool prior) — anti-collapse anchor.
    marginal_w: float = 1.0
    #: Reconstruction weight during CTTA.
    recon_w_tta: float = 0.5
    #: Where CTTA reconstruction is applied: 'stream' | 'pool' | 'both' | 'none'.
    recon_source: ReconSource = "stream"
    #: CoTTA-style stochastic restore probability per weight per step (0 disables).
    restore_prob: float = 0.01

    # ── AGSA: Activation-Gated Spline Adaptation (KAN-only novelty) ─────────
    use_spline_gate: bool = True
    #: EMA horizon for basis-activation mass.
    spline_gate_ema: float = 0.99
    #: A coefficient is 'active' if its EMA mass exceeds this fraction of the
    #: per-input-dim max mass.
    spline_gate_thresh: float = 0.05
    #: EWC retention penalty weight toward source weights (0 disables; try 100.0).
    ewc_lambda: float = 0.0

    # ── Loss (class imbalance) ─────────────────────────────────────────────
    #: 'ce' | 'weighted_ce' | 'focal'.
    loss_type: LossType = "weighted_ce"
    focal_gamma: float = 2.0
    #: Cap on inverse-frequency class weights (stabilises CTTA's high LR).
    max_class_weight: float = 20.0

    # ── Scaling (transfer-friendly) ────────────────────────────────────────
    #: 'minmax' | 'robust' | 'quantile'.
    scaler_type: ScalerType = "quantile"
    #: Only used when scaler_type == 'robust' — ±robust_clip IQR units map to ±1.
    robust_clip: float = 4.0

    # ── Sequential (continual) CTTA ────────────────────────────────────────
    #: Fraction of each replay-pool epoch drawn from the CURRENT domain.
    replay_current_frac: float = 0.5

    def __post_init__(self) -> None:
        if self.loss_type not in ("ce", "weighted_ce", "focal"):
            raise ValueError(
                f"Unknown loss_type={self.loss_type!r}. "
                "Choose 'ce', 'weighted_ce', or 'focal'."
            )
        if self.scaler_type not in ("minmax", "robust", "quantile"):
            raise ValueError(
                f"Unknown scaler_type={self.scaler_type!r}. "
                "Choose 'minmax', 'robust', or 'quantile'."
            )
        if self.recon_source not in ("stream", "pool", "both", "none"):
            raise ValueError(
                f"Unknown recon_source={self.recon_source!r}. "
                "Choose 'stream', 'pool', 'both', or 'none'."
            )

    def to_dict(self) -> dict:
        """Flat dict of all fields (handy for logging / CSV metadata)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
