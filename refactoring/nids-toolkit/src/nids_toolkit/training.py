"""Pretraining, evaluation, the CTTA loop, and the few-shot baseline."""

from __future__ import annotations

import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from .adaptation import SplineActivationGate, StochasticRestore
from .config import ExperimentConfig
from .data import reset_loader_generator
from .losses import labels_of, make_criterion, pool_prior, stream_entropy_loss
from .models import (
    KanAEClassifier,
    build_model,
    get_spline_only_params,
    get_trainable_params,
)
from .utils import log_step


def pretrain_source(model, loader, epochs, device, cfg: ExperimentConfig):
    """Joint supervised + reconstruction pre-training.
    Loss = criterion(logits, y) + cfg.recon_w * MSE(recon, x).
    """
    optimizer = optim.Adam(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)
    ce_crit = make_criterion(labels_of(loader), device, cfg)
    mse_crit = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total, n_batches = 0.0, 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, recon, _ = model(x)
            loss = ce_crit(logits, y) + cfg.recon_w * mse_crit(recon, x)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # a single NaN batch would otherwise make the epoch loss NaN forever.
                total_loss += loss.item()
                n_batches += 1
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(
            f"[Pretrain] Epoch {epoch + 1}/{epochs} | "
            f"Loss: {total_loss / max(n_batches, 1):.4f} | "
            f"Acc: {correct / total:.4f}"
        )


def evaluate(model, loader, device, desc="Eval"):
    """Evaluate using the classifier head (argmax of logits). Returns F1."""
    was_training = model.training  # restore mode instead of leaving eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _, _ = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    if was_training:
        model.train()
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    print(f"[{desc}] F1: {f1:.4f} | Acc: {acc:.4f}")
    return f1


def run_ctta(
    model,
    stream_loader,
    pool_loader,
    device,
    cfg: ExperimentConfig,
    spline_only=False,
    ewc=None,
    ewc_lambda=0.0,
    spline_gate=None,
    train_all=False,
    restore_prob=None,
    prior=None,
    external_gates=None,
):
    """Few-Shot Layer-Selective CTTA with the predict-then-adapt protocol.

    Each incoming stream batch is first predicted with the current model (adapted
    on all *previous* batches only), and only then used for adaptation — the
    standard online TTA protocol (TENT et al.). Per stream batch:
      0. Predict with the current model — these predictions are what's scored.
      1. Supervised CE on a pool batch — anchors the boundary to the target.
      2. (cfg.use_entropy) robust entropy + KL(marginal || prior) anti-collapse.
      3. Reconstruction (cfg.recon_source) — TTT-style self-supervision.
      4. (ewc_lambda>0) EWC retention penalty toward source weights.
      5. Stochastic restore (cfg.restore_prob) — CoTTA-style anti-drift.

    Returns ``(preds, labels, trajectory)`` on the stream.
    """
    if train_all:
        train_params = list(model.parameters())
        print("[CTTA] TRAIN-ALL mode: all parameters trainable (scratch baseline).")
    elif spline_only:
        train_params = get_spline_only_params(model)
        print("[CTTA] SPLINE-ONLY mode: updating only last encoder layer spline weights.")
    else:
        train_params = get_trainable_params(model)

    # Adam raises an opaque error on an empty parameter list.
    if len(train_params) == 0:
        raise RuntimeError(
            "run_ctta: no trainable parameters were selected — "
            "check get_spline_only_params for this architecture."
        )

    trainable_ids = {id(p) for p in train_params}
    for p in model.parameters():
        p.requires_grad = id(p) in trainable_ids

    print(
        f"[CTTA] Updating {len(train_params)} param tensors "
        f"({sum(p.numel() for p in train_params)} params). "
        f"All other weights frozen."
    )

    optimizer = optim.Adam(train_params, lr=cfg.tta_lr)
    ce_crit = make_criterion(labels_of(pool_loader), device, cfg)
    if cfg.use_entropy:
        prior = pool_prior(pool_loader, device) if prior is None else prior.to(device)
    else:
        prior = None
    restore = StochasticRestore(
        train_params, cfg.restore_prob if restore_prob is None else restore_prob
    )
    use_gate = cfg.use_spline_gate if spline_gate is None else spline_gate
    manage_gates = external_gates is None
    gates = []
    if not manage_gates:
        gates = list(external_gates)
        for g in gates:
            g.capture = True  # record activations only during this stage
        if gates:
            print(
                f"[CTTA] AGSA (external/cumulative) active on {len(gates)} "
                f"KAN layer(s); gate state persists across stages."
            )
    elif use_gate and isinstance(model, KanAEClassifier):
        for layer in model.encoder.layers:
            if any(id(p) in trainable_ids for p in layer.parameters()):
                try:
                    gates.append(SplineActivationGate(layer, cfg))
                except ValueError:
                    pass
        if gates:
            print(
                f"[CTTA] AGSA active on {len(gates)} KAN layer(s): spline "
                f"updates confined to target-activated basis regions."
            )
    model.train()

    # reset the pool loader's private shuffle generator so every CTTA run sees
    # the identical pool batch sequence regardless of prior runs.
    reset_loader_generator(pool_loader, cfg.seed)
    pool_iter = iter(pool_loader)

    def next_pool():
        nonlocal pool_iter
        try:
            return next(pool_iter)
        except StopIteration:
            pool_iter = iter(pool_loader)
            return next(pool_iter)

    all_preds, all_labels = [], []
    trajectory = []
    batch_count = 0
    TRAJ_INTERVAL = 20

    try:  # hooks must be removed even on exception — otherwise gates leak.
        for x_stream, y_stream in stream_loader:
            x_stream = x_stream.to(device)

            # ── 0. Predict FIRST with the current model ──────────────────────
            with torch.no_grad():
                model.eval()
                logits_f, _, _ = model(x_stream)
                preds = logits_f.argmax(1).cpu().numpy()
                model.train()

            all_preds.extend(preds)
            all_labels.extend(y_stream.numpy())

            # ── Adapt on this batch (affects future batches only) ────────────
            for _ in range(cfg.tta_steps):
                optimizer.zero_grad()

                # ── 1. Supervised CE on pool batch ───────────────────────────
                x_pool, y_pool = next_pool()
                x_pool = x_pool.to(device)
                y_pool = y_pool.to(device)
                logits_pool, recon_pool, _ = model(x_pool)
                loss_ce = ce_crit(logits_pool, y_pool)

                loss = cfg.few_shot_w * loss_ce

                # ── 2+3a. Stream forward: entropy and/or stream reconstruction ─
                if cfg.use_entropy or cfg.recon_source in ("stream", "both"):
                    logits_s, recon_s, _ = model(x_stream)
                    if cfg.use_entropy:
                        loss = loss + stream_entropy_loss(logits_s, prior, cfg)
                    if cfg.recon_source in ("stream", "both"):
                        loss = loss + cfg.recon_w_tta * F.mse_loss(recon_s, x_stream)

                # ── 3b. Reconstruction on benign pool samples (legacy option) ─
                if cfg.recon_source in ("pool", "both"):
                    benign_mask = y_pool == 0
                    if benign_mask.any():
                        loss_recon = F.mse_loss(recon_pool[benign_mask], x_pool[benign_mask])
                        loss = loss + cfg.recon_w_tta * loss_recon

                # ── 4. EWC retention penalty (optional) ──────────────────────
                if ewc is not None and ewc_lambda > 0:
                    loss = loss + ewc_lambda * ewc.penalty()

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
                    optimizer.step()
                    restore()  # stochastic restore after each update

            batch_count += 1
            if batch_count % TRAJ_INTERVAL == 0:
                trajectory.append(
                    (batch_count, f1_score(all_labels, all_preds, zero_division=0))
                )

        # the final (partial) segment was otherwise never recorded.
        if batch_count % TRAJ_INTERVAL != 0:
            trajectory.append((batch_count, f1_score(all_labels, all_preds, zero_division=0)))

        if manage_gates:
            for i, g in enumerate(gates):
                print(
                    f"[CTTA] AGSA layer {i}: {g.coverage():.1%} of spline "
                    f"coefficients active (rest frozen at source)."
                )
    finally:
        if manage_gates:
            for g in gates:  # always detach hooks, success or failure
                g.remove()
        else:
            for g in gates:  # external gates persist — just stop capturing
                g.capture = False

    print("[CTTA] Stream complete.")
    return np.array(all_preds), np.array(all_labels), trajectory


def few_shot_baseline(
    model_state,
    pool_loader,
    stream_loader,
    input_dim,
    arch,
    device,
    cfg: ExperimentConfig,
    spline_only,
    epochs=20,
    train_all=False,
):
    """Fair few-shot baseline = "CTTA minus the stream".

    Identical to ``run_ctta`` in starting point, trainable subset, supervised pool
    objective, and optimizer (including ``cfg.tta_lr``). Differs ONLY in never
    seeing the stream.
    """
    # RNG snapshot must be taken BEFORE build_model — weight init consumes the
    # global RNG, so snapshotting after it fails to isolate this baseline.
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    fs_model = build_model(arch, input_dim, cfg).to(device)
    fs_model.load_state_dict(model_state)

    if train_all:
        train_params = list(fs_model.parameters())
    else:
        train_params = (
            get_spline_only_params(fs_model) if spline_only else get_trainable_params(fs_model)
        )
    if len(train_params) == 0:
        raise RuntimeError("few_shot_baseline: empty trainable parameter set.")
    trainable_ids = {id(p) for p in train_params}
    for p in fs_model.parameters():
        p.requires_grad = id(p) in trainable_ids

    mode = "all-params" if train_all else ("spline-only" if spline_only else "full")
    print(
        f"[Few-shot/{mode}] updating {len(train_params)} tensors "
        f"({sum(p.numel() for p in train_params)} params)."
    )

    try:
        optimizer = optim.Adam(train_params, lr=cfg.tta_lr)  # truly matched to CTTA
        ce_crit = make_criterion(labels_of(pool_loader), device, cfg)

        # identical pool shuffle sequence regardless of what ran before.
        reset_loader_generator(pool_loader, cfg.seed)
        fs_model.train()
        for epoch in range(epochs):
            for x, y in pool_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, recon, _ = fs_model(x)

                loss_ce = ce_crit(logits, y)
                loss = cfg.few_shot_w * loss_ce

                benign_mask = y == 0
                if benign_mask.any():
                    # recon here stays pool-based by definition — this baseline's
                    # defining constraint is zero stream access.
                    loss_recon = F.mse_loss(recon[benign_mask], x[benign_mask])
                    loss = loss + cfg.recon_w_tta * loss_recon

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
                    optimizer.step()

        f1 = evaluate(fs_model, stream_loader, device, desc=f"Few-shot ({mode}, {arch})")
    finally:
        torch.set_rng_state(rng_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    del fs_model
    gc.collect()
    return f1


def run_phase1_pretraining(arch, source_name, input_dim, loader, device, cfg: ExperimentConfig):
    """Build a model and pretrain it on the source (``cfg.pretrain_epochs``)."""
    log_step(2, f"SOURCE PRE-TRAINING ({source_name}) | {arch.upper()}")

    model = build_model(arch, input_dim, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = get_trainable_params(model)
    print(
        f"[Model] {arch} | input_dim={input_dim} | latent_dim={cfg.latent_dim} | "
        f"total params={n_params:,} | CTTA-trainable params="
        f"{sum(p.numel() for p in trainable_params):,}\n"
    )

    pretrain_source(model, loader, epochs=cfg.pretrain_epochs, device=device, cfg=cfg)
    return model


def run_phase2_zeroshot(model, stream_loader, target_name, device):
    """Zero-shot baseline: evaluate the source model directly on the target stream."""
    log_step(3, f"\n--- ZERO-SHOT BASELINE ({target_name}) ---")
    return evaluate(model, stream_loader, device, desc=f"Zero-shot {target_name}")
