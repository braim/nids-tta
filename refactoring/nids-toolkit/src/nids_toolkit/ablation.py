"""Type-B ablation: trainable-subset (Table 3) and loss-term (Table 4) sweeps.

Every ablation row runs the SAME method as the headline ``run_ctta`` (predict-
then-adapt, robust entropy, AGSA gates, stochastic restore) so the numbers stay
directly comparable to it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

from .adaptation import SplineActivationGate, StochasticRestore
from .config import ExperimentConfig
from .data import reset_loader_generator
from .losses import labels_of, make_criterion, pool_prior, stream_entropy_loss
from .models import (
    CnnAEClassifier,
    KanAEClassifier,
    TabTransformerAEClassifier,
    get_spline_only_params,
    get_trainable_params,
)


class TypeBAblation:
    def run(self, model, target_name, stream_loader, pool_loader, cfg: ExperimentConfig):
        device = next(model.parameters()).device
        model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # save/restore RNG state (mirrors few_shot_baseline) so CTTA afterwards
        # sees the same RNG for KAN as for CNN/Tab (where the ablation never runs).
        rng_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        prev_flags = {id(p): p.requires_grad for p in model.parameters()}

        def _norm_params(m):
            out = []
            for mod in m.modules():
                if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)):
                    out.extend(mod.parameters())
            return out

        def _classifier_params(m):
            return list(m.classifier.parameters())

        def _last_layer_full_params(m):
            params = []
            if isinstance(m, KanAEClassifier):
                params.extend(m.encoder.layers[-1].parameters())
            elif isinstance(m, CnnAEClassifier):
                last = None
                for mod in m.encoder.modules():
                    if isinstance(mod, (nn.Conv1d, nn.Linear)):
                        last = mod
                if last is not None:
                    params.extend(last.parameters())
            elif isinstance(m, TabTransformerAEClassifier):
                params.extend(m.encoder_out.parameters())
            return params

        def run_ctta_ablate(
            m, s_loader, p_loader, dev, trainable="spline_only", use_ce=True, use_ent=True, use_recon=True
        ):
            if trainable == "spline_only":
                train_params = get_spline_only_params(m)
            elif trainable == "splines_norm":
                train_params = get_spline_only_params(m) + _norm_params(m)
            elif trainable == "splines_cls":
                train_params = get_spline_only_params(m) + _classifier_params(m)
            elif trainable == "splines_norm_cls":
                train_params = get_spline_only_params(m) + _norm_params(m) + _classifier_params(m)
            elif trainable == "last_layer_full":
                train_params = _last_layer_full_params(m)
            elif trainable == "full":
                train_params = get_trainable_params(m)
            else:
                raise ValueError(f"unknown trainable={trainable!r}")

            # dedupe — some subsets collect the same tensor twice; Adam raises on that.
            seen, deduped = set(), []
            for p in train_params:
                if id(p) not in seen:
                    seen.add(id(p))
                    deduped.append(p)
            train_params = deduped

            trainable_ids = {id(p) for p in train_params}
            for p in m.parameters():
                p.requires_grad = id(p) in trainable_ids

            optimizer = optim.Adam(train_params, lr=cfg.tta_lr)
            ce_crit = make_criterion(labels_of(p_loader), device, cfg)
            restore = StochasticRestore(train_params, cfg.restore_prob)
            # AGSA gates attached here too — the ablation must run the SAME method.
            trainable_ids_a = {id(p) for p in train_params}
            gates = []
            if cfg.use_spline_gate and isinstance(m, KanAEClassifier):
                for layer in m.encoder.layers:
                    if any(id(p) in trainable_ids_a for p in layer.parameters()):
                        try:
                            gates.append(SplineActivationGate(layer, cfg))
                        except ValueError:
                            pass
            m.train()

            # identical pool shuffle sequence for every ablation config.
            reset_loader_generator(p_loader, cfg.seed)
            pool_iter = iter(p_loader)

            def next_pool():
                nonlocal pool_iter
                try:
                    return next(pool_iter)
                except StopIteration:
                    pool_iter = iter(p_loader)
                    return next(pool_iter)

            all_preds, all_labels = [], []
            prior = pool_prior(p_loader, dev)

            try:  # gates must detach even on failure.
                for x_stream, y_stream in s_loader:
                    x_stream = x_stream.to(dev)

                    # predict-then-adapt — same protocol as run_ctta.
                    with torch.no_grad():
                        m.eval()
                        logits_f, _, _ = m(x_stream)
                        preds = logits_f.argmax(1).cpu().numpy()
                        m.train()
                    all_preds.extend(preds)
                    all_labels.extend(y_stream.numpy())

                    for _ in range(cfg.tta_steps):
                        optimizer.zero_grad()
                        x_pool, y_pool = next_pool()
                        x_pool = x_pool.to(dev)
                        y_pool = y_pool.to(dev)
                        logits_pool, recon_pool, _ = m(x_pool)
                        loss_ce = ce_crit(logits_pool, y_pool)

                        loss = torch.tensor(0.0, device=dev)
                        if use_ce:
                            loss = loss + cfg.few_shot_w * loss_ce

                        need_stream = use_ent or (
                            use_recon and cfg.recon_source in ("stream", "both")
                        )
                        if need_stream:
                            logits_s, recon_s, _ = m(x_stream)
                        if use_ent:
                            loss = loss + stream_entropy_loss(logits_s, prior, cfg)
                        if use_recon:
                            if cfg.recon_source in ("stream", "both"):
                                loss = loss + cfg.recon_w_tta * F.mse_loss(recon_s, x_stream)
                            if cfg.recon_source in ("pool", "both"):
                                benign_mask = y_pool == 0
                                if benign_mask.any():
                                    loss = loss + cfg.recon_w_tta * F.mse_loss(
                                        recon_pool[benign_mask], x_pool[benign_mask]
                                    )

                        if loss.requires_grad and not (torch.isnan(loss) or torch.isinf(loss)):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
                            optimizer.step()
                            restore()
            finally:
                for g in gates:
                    g.remove()

            return f1_score(all_labels, all_preds, zero_division=0)

        def run_experiments_for_target(m, t_name, s_loader, p_loader):
            print("\n" + "=" * 80)
            print(f"ABLATION B TARGET: {t_name}")
            print("=" * 60)
            print("Table 3 — Trainable subset ablation")
            print("=" * 60)
            table3_order = [
                ("spline_only", "Spline-only"),
                ("splines_norm", "Splines + norm"),
                ("splines_cls", "Splines + classifier"),
                ("splines_norm_cls", "Splines + norm + classifier"),
                ("last_layer_full", "Last KAN layer (incl. base path)"),
                ("full", "Norm + classifier + last layer"),
            ]
            t3 = {}
            for key, label in table3_order:
                m.load_state_dict(model_state)
                t3[key] = run_ctta_ablate(m, s_loader, p_loader, device, trainable=key)
                print(f"  {label:35s}  F1 = {t3[key]:.4f}")

            print("\n" + "=" * 60)
            print("Table 4 — Loss-term ablation (spline-only)")
            print("=" * 60)
            loss_configs = [
                ("CE + ent + recon", True, True, True),
                ("CE + ent only", True, True, False),
                ("CE + recon only", True, False, True),
                ("CE only", True, False, False),
                ("ent only (TENT-like)", False, True, False),
            ]
            t4 = {}
            for name, ce, ent, recon in loss_configs:
                m.load_state_dict(model_state)
                t4[name] = run_ctta_ablate(
                    m, s_loader, p_loader, device, trainable="spline_only",
                    use_ce=ce, use_ent=ent, use_recon=recon,
                )
                print(f"  {name:25s}  F1 = {t4[name]:.4f}")

            print(f"\n--- Summary for {t_name} (rounded to 2 dp) ---")
            print("Table 3:")
            for key, label in table3_order:
                print(f"  {label:35s} {t3[key]:.2f}")
            print("Table 4:")
            for name, *_ in loss_configs:
                print(f"  {name:25s} {t4[name]:.2f}")
            print("=" * 80)

            rows = [
                {"Target": t_name, "Table": "T3-trainable", "Config": label, "F1": t3[key]}
                for key, label in table3_order
            ]
            rows += [
                {"Target": t_name, "Table": "T4-loss", "Config": name, "F1": t4[name]}
                for name, *_ in loss_configs
            ]
            return rows

        try:
            rows = run_experiments_for_target(model, target_name, stream_loader, pool_loader)
        finally:
            model.load_state_dict(model_state)  # restore weights
            for p in model.parameters():  # restore flags
                p.requires_grad = prev_flags.get(id(p), True)
            torch.set_rng_state(rng_state)  # restore RNG
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return rows
