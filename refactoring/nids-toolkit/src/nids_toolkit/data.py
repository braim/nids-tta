"""Dataset loading, feature engineering / alignment, and DataLoader builders."""

from __future__ import annotations

import os

import kagglehub
import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .scaling import FeatureScaler
from .utils import seeded_generator


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive flow-level features and drop identifier/label columns."""
    if "FLOW_END_MILLISECONDS" in df.columns and "FLOW_START_MILLISECONDS" in df.columns:
        df = df.with_columns(
            # clip at 0 — corrupted flows with end < start produced negative
            # durations, and log1p(x < -1) = NaN downstream.
            (pl.col("FLOW_END_MILLISECONDS") - pl.col("FLOW_START_MILLISECONDS"))
            .clip(lower_bound=0)
            .alias("FLOW_DURATION")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("FLOW_DURATION"))
    if "IN_BYTES" in df.columns and "IN_PKTS" in df.columns:
        df = df.with_columns(
            (pl.col("IN_BYTES") / (pl.col("IN_PKTS") + 1e-5)).alias("BYTES_PER_PKT")
        )
    log_cols = [
        "IN_BYTES",
        "IN_PKTS",
        "FLOW_DURATION",
        "BYTES_PER_PKT",
        "SRC_TO_DST_IAT_MAX",
        "DST_TO_SRC_IAT_MAX",
    ]
    existing = [c for c in log_cols if c in df.columns]
    if existing:
        df = df.with_columns([pl.col(c).clip(lower_bound=0).log1p() for c in existing])
    drop_cols = [
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "Label",
        "Attack",
        "label",
        "attack",
        "Date",
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])
    # keep only numeric columns — a stray string/categorical column in a dataset
    # variant would otherwise crash the run or produce garbage.
    numeric = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    dropped = [c for c in df.columns if c not in numeric]
    if dropped:
        print(f"   [engineer_features] Dropping non-numeric columns: {dropped}")
    return df.select(numeric)


def align_features(df: pl.DataFrame, reference_features: list) -> pl.DataFrame:
    """Reindex the target to the source feature schema so ``scaler.transform`` and
    the model always see the same features in the same order. Shared columns are
    reordered, missing ones zero-filled (with a warning), extras dropped.
    """
    missing = [c for c in reference_features if c not in df.columns]
    extra = [c for c in df.columns if c not in reference_features]
    if missing:
        print(
            f"   [align_features] WARNING: target missing {len(missing)} source "
            f"features (zero-filled): {missing}"
        )
        df = df.with_columns([pl.lit(0.0, dtype=pl.Float32).alias(c) for c in missing])
    if extra:
        print(f"   [align_features] Dropping {len(extra)} target-only features: {extra}")
    return df.select(reference_features)


def load_dataset(dataset_name: str, cfg: ExperimentConfig, sample_n=None, align_to=None):
    """Download a dataset and return ``(X, y, feature_names)``.

    ``sample_n`` defaults to ``cfg.sample_n`` (pass an explicit value to override;
    set ``cfg.sample_n = None`` to use every row). All CSVs under the dataset are
    concatenated (``vertical_relaxed`` handles dtype drift between files).
    """
    if sample_n is None:
        sample_n = cfg.sample_n
    print(f"[Data] Loading {dataset_name} ...")
    path = kagglehub.dataset_download(dataset_name)
    csv_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    frames = [pl.scan_csv(f) for f in csv_files]
    df = pl.concat(frames, how="vertical_relaxed").collect(engine="streaming")
    if sample_n and sample_n < df.height:
        df = df.sample(n=sample_n, seed=cfg.seed)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None:
        print(
            "   WARNING: no 'label' column found — y set to all zeros; "
            "F1 on this dataset is meaningless."
        )
        y = np.zeros(df.height, dtype=np.int64)
    else:
        y = df[label_col].to_numpy().astype(np.int64)
    df = engineer_features(df)
    if align_to is not None:
        df = align_features(df, align_to)
    feature_names = df.columns
    X = df.to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"   -> Shape: {X.shape} | Attack rate: {np.mean(y):.2%}")
    return X, y, feature_names


def make_source_loaders(X, y, cfg: ExperimentConfig):
    """Stratified 80/20 split. Scaler fitted on the training split only.
    Returns ``(loader_train, loader_test, scaler)``.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )
    scaler = FeatureScaler(cfg).fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    loader_tr = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=seeded_generator(cfg.seed),
    )
    loader_te = DataLoader(test_ds, batch_size=cfg.tta_batch_size, shuffle=False)
    return loader_tr, loader_te, scaler


def make_target_loaders(X, y, cfg: ExperimentConfig, external_scaler=None):
    """Split the target into a small labelled pool and an unlabelled-at-adaptation
    stream (labels kept for evaluation only). Returns ``(pool_loader, stream_loader)``.

    If ``external_scaler`` is given it is used (deployment-realistic — no target
    statistics assumed); otherwise a scaler is fitted on the pool only, which is
    the only data an analyst would have labelled/inspected. The pool represents a
    brief initial analyst review period at deployment.
    """
    pool_n = max(int(round(len(y) * cfg.few_shot_ratio)), cfg.min_pool_size)
    X_pool, X_stream, y_pool, y_stream = train_test_split(
        X, y, train_size=pool_n, random_state=cfg.seed, stratify=y
    )
    if y_pool.sum() < 2:
        print(
            f"   WARNING: pool contains only {int(y_pool.sum())} attack "
            f"sample(s); supervised anchoring will be very weak."
        )

    scaler = (
        external_scaler if external_scaler is not None else FeatureScaler(cfg).fit(X_pool)
    )

    X_pool = scaler.transform(X_pool)
    X_stream = scaler.transform(X_stream)

    pool_ds = TensorDataset(torch.from_numpy(X_pool), torch.from_numpy(y_pool))
    stream_ds = TensorDataset(torch.from_numpy(X_stream), torch.from_numpy(y_stream))

    pool_loader = DataLoader(
        pool_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=seeded_generator(cfg.seed),
    )
    stream_loader = DataLoader(stream_ds, batch_size=cfg.tta_batch_size, shuffle=False)

    print(
        f"   -> Pool: {len(y_pool)} samples "
        f"(attack rate: {np.mean(y_pool):.2%}) | "
        f"Stream: {len(y_stream)} "
        f"(attack rate: {np.mean(y_stream):.2%})"
    )
    return pool_loader, stream_loader


def reset_loader_generator(loader, seed: int = 42) -> None:
    """Re-seed a shuffling DataLoader's private generator.

    Shuffling DataLoaders carry their own ``torch.Generator``, which
    ``torch.get_rng_state()/set_rng_state()`` do NOT cover. Re-seeding it at the
    start of each comparable run makes every run see the identical shuffle
    sequence by construction, independent of what ran before it.
    """
    g = getattr(loader, "generator", None)
    if g is not None:
        g.manual_seed(seed)


#: Alias kept for parity with the original notebook (identical behaviour).
reseed_loader = reset_loader_generator


def build_replay_pool(current_pool, past_pools, cfg: ExperimentConfig, current_frac=None):
    """Replay pool = current domain's pool + all past domains' pools, with the
    current domain oversampled (by deterministic duplication) so it makes up
    ~``current_frac`` of the mix. Uniform union (``current_frac=None``) dilutes the
    current domain to 1/k at stage k; weighting restores plasticity while keeping
    the full retention signal.
    """
    xs_p = torch.cat([l.dataset.tensors[0] for l in past_pools])
    ys_p = torch.cat([l.dataset.tensors[1] for l in past_pools])
    x_c, y_c = current_pool.dataset.tensors
    if current_frac is not None and len(past_pools) > 0:
        # duplicate current pool so n_current*k / (n_current*k + n_past) ~ frac
        k = max(
            1,
            int(
                round(
                    current_frac
                    / max(1e-9, 1 - current_frac)
                    * len(ys_p)
                    / max(1, len(y_c))
                )
            ),
        )
        x_c = x_c.repeat(k, *([1] * (x_c.dim() - 1)))
        y_c = y_c.repeat(k)
    xs = torch.cat([x_c, xs_p])
    ys = torch.cat([y_c, ys_p])
    ds = TensorDataset(xs, ys)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, generator=seeded_generator(cfg.seed)
    )
