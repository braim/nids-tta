"""Pluggable, transfer-friendly feature scaling."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
)

from .config import ExperimentConfig


class FeatureScaler:
    """Pluggable scaling behind the same fit/transform interface the pipeline
    already uses, so it drops in wherever a ``MinMaxScaler`` was passed around.
    Every mode guarantees float32 output in ``[-1, 1]``, which the KAN
    ``grid_range=[-1, 1]`` requires.

    * ``minmax``   — MinMax to [-1, 1], then clip (original behaviour).
    * ``robust``   — (x - median) / IQR per feature; ``±cfg.robust_clip`` scaled
      units map to ±1 and everything beyond clips. Outliers no longer define the
      range, so the bulk of the data keeps its resolution.
    * ``quantile`` — rank-based map to uniform [0, 1] rescaled to [-1, 1]. Fully
      immune to heavy tails; out-of-range target values saturate at the boundary
      instead of compressing the whole feature. The transfer-friendliest option.

    The scaling ``kind`` and RNG/clip settings are read from ``cfg``; pass
    ``kind`` to override ``cfg.scaler_type`` for a one-off scaler.
    """

    def __init__(self, cfg: ExperimentConfig, kind: str | None = None):
        kind = cfg.scaler_type if kind is None else kind
        if kind not in ("minmax", "robust", "quantile"):
            raise ValueError(
                f"Unknown scaler_type={kind!r}. Choose 'minmax', 'robust', or 'quantile'"
            )
        self.cfg = cfg
        self.kind = kind
        self._sk = None

    def fit(self, X):
        if self.kind == "minmax":
            self._sk = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        elif self.kind == "robust":
            self._sk = RobustScaler(quantile_range=(5.0, 95.0)).fit(X)
        else:  # quantile
            self._sk = QuantileTransformer(
                output_distribution="uniform",
                n_quantiles=min(1000, len(X)),  # n_quantiles must be <= n_samples
                subsample=1_000_000,
                random_state=self.cfg.seed,
            ).fit(X)
        return self

    def transform(self, X):
        if self._sk is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        Z = self._sk.transform(X)
        if self.kind == "robust":
            Z = Z / self.cfg.robust_clip  # ±robust_clip IQR-units -> ±1
        elif self.kind == "quantile":
            Z = Z * 2.0 - 1.0  # [0,1] -> [-1,1]
        return np.clip(Z, -1, 1).astype(np.float32)
