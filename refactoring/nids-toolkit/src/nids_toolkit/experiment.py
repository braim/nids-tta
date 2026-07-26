"""High-level orchestration: the episodic experiment grid (``run_grid``) and the
sequential/continual protocol (``run_sequential``).

These are the two former notebook "main execution" cells. A notebook now builds
an :class:`ExperimentConfig` and calls one of these — nothing else.
"""

from __future__ import annotations

import gc
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .ablation import TypeBAblation
from .adaptation import KAN_Retention, SplineActivationGate
from .config import ExperimentConfig
from .data import (
    build_replay_pool,
    load_dataset,
    make_source_loaders,
    make_target_loaders,
)
from .losses import pool_prior
from .models import (
    KanAEClassifier,
    build_model,
    get_spline_only_params,
    get_trainable_params,
)
from .training import (
    evaluate,
    few_shot_baseline,
    run_ctta,
    run_phase1_pretraining,
    run_phase2_zeroshot,
)
from .utils import get_device, log_step, seed_everything, seeded_generator
from .viz import plot_adaptation_trajectory, run_all_visualizations

#: Default Kaggle dataset handles (name -> kagglehub path).
DEFAULT_DATASETS = {
    "CICIDS2018": "seyhed/nf-cicids2018-v3",
    "ToN-IoT": "seyhed/nf-ton-iot-v3",
    "UNSW-NB15": "seyhed/nf-unsw-nb15-v3",
}

#: Default architectures for the episodic grid.
DEFAULT_ARCHITECTURES = ["kan", "cnn", "tab", "flowtransformer"]

#: Defaults for the sequential (continual) protocol.
DEFAULT_SEQ_ARCHS = ["kan"]
DEFAULT_SEQUENCES = [
    ("CICIDS2018", "ToN-IoT", "UNSW-NB15"),  # source, then targets in order
    ("UNSW-NB15", "ToN-IoT", "CICIDS2018"),
]
DEFAULT_SEQ_VARIANTS = [
    {"name": "AGSA per-stage", "replay": False, "gate": "stage"},
    {"name": "no AGSA", "replay": False, "gate": "off"},
    {"name": "replay, no AGSA", "replay": True, "gate": "off"},
    {"name": "replay + AGSA", "replay": True, "gate": "stage"},
    {"name": "replay + cumulative AGSA", "replay": True, "gate": "cumulative"},
]


def _show(df: pd.DataFrame) -> None:
    """Display in Jupyter when available, else print."""
    try:
        display(df)  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        print(df.to_string())


def run_grid(cfg: ExperimentConfig, architectures=None, datasets=None, out_dir="."):
    """Episodic TTA grid over (architecture x source x target).

    The model is reset between targets. Returns ``(results_df, ablation_df)`` and
    writes ``grid_results.csv`` / ``ablation_results.csv`` under ``out_dir``.
    """
    architectures = architectures if architectures is not None else DEFAULT_ARCHITECTURES
    datasets = datasets if datasets is not None else DEFAULT_DATASETS
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    results = []
    ablation_results = []

    for arch in architectures:
        for src_name, src_path in datasets.items():
            # re-seed at the top of every (arch, source) config so each run is
            # independent of everything that executed before it.
            seed_everything(cfg.seed)

            print("=" * 80)
            print(f"EXPERIMENT | Arch: {arch.upper()} | Source: {src_name}")
            print("-" * 80)

            log_step(1, "load data set")
            # 1. Load source
            X_src, y_src, feature_names = load_dataset(src_path, cfg)
            input_dim = X_src.shape[1]
            loader_src_train, loader_src_test, source_scaler = make_source_loaders(X_src, y_src, cfg)
            del X_src, y_src
            gc.collect()

            # 2. Pre-train model
            model = run_phase1_pretraining(arch, src_name, input_dim, loader_src_train, device, cfg)

            # Save post-pretrain state
            model_state = {k: v.clone() for k, v in model.state_dict().items()}

            # Fisher information depends only on (arch, source) — hoisted out of
            # the target loop.
            kan_ret = None
            if arch == "kan":
                model.load_state_dict(model_state)
                params_to_protect = (
                    get_spline_only_params(model) if cfg.spline_true else get_trainable_params(model)
                )
                kan_ret = KAN_Retention(model, loader_src_test, device, params_to_protect, cfg)

            # 3. Determine targets
            targets = {k: v for k, v in datasets.items() if k != src_name}
            trajectories = {}

            for tgt_name, tgt_path in targets.items():
                log_step(
                    3,
                    f"\n---  Executing | Arch: {arch.upper()} | "
                    f"Source: {src_name} -> Target: {tgt_name} ---",
                )

                # align the target's feature schema to the source's.
                X_tgt, y_tgt, _ = load_dataset(tgt_path, cfg, align_to=list(feature_names))

                # Using source scaler for consistency (deployment-realistic)
                pool_tgt_src_sc, stream_tgt_src_sc = make_target_loaders(
                    X_tgt, y_tgt, cfg, external_scaler=source_scaler
                )
                del X_tgt, y_tgt
                gc.collect()

                # Zero-shot baseline
                log_step(4, "zero shot")
                model.load_state_dict(model_state)
                zero_shot_f1 = run_phase2_zeroshot(model, stream_tgt_src_sc, tgt_name, device)

                # source performance before any adaptation
                src_f1_pre = evaluate(
                    model, loader_src_test, device, desc=f"Source {src_name} (pre-CTTA)"
                )

                # step-matched few-shot — same optimizer-step budget as CTTA
                if cfg.few_shot_match_steps:
                    fs_epochs = max(
                        1,
                        math.ceil(
                            len(stream_tgt_src_sc) * cfg.tta_steps
                            / max(len(pool_tgt_src_sc), 1)
                        ),
                    )
                else:
                    fs_epochs = 20
                print(
                    f"[Few-shot] step-matched epochs = {fs_epochs} "
                    f"({fs_epochs * len(pool_tgt_src_sc)} steps vs CTTA "
                    f"{len(stream_tgt_src_sc) * cfg.tta_steps})"
                )

                # Few-shot baselines
                log_step(5, "few shot")
                few_shot_spline_f1 = few_shot_baseline(
                    model_state, pool_tgt_src_sc, stream_tgt_src_sc, input_dim, arch, device,
                    cfg, spline_only=True, epochs=fs_epochs,
                )  # param-matched control
                few_shot_full_f1 = few_shot_baseline(
                    model_state, pool_tgt_src_sc, stream_tgt_src_sc, input_dim, arch, device,
                    cfg, spline_only=False, epochs=fs_epochs,
                )  # full-capacity reference

                # CTTA
                log_step(6, "CTTA")
                model.load_state_dict(model_state)

                if arch == "kan":
                    print("===Type B Ablation===")
                    abl_rows = TypeBAblation().run(
                        model, tgt_name, stream_tgt_src_sc, pool_tgt_src_sc, cfg
                    )
                    for r in abl_rows:
                        r.update({"Architecture": arch, "Source": src_name})
                    ablation_results.extend(abl_rows)
                    model.load_state_dict(model_state)  # explicit reset

                # run BOTH CTTA modes so the final table reports spline-only and
                # full side by side, from identical model AND RNG state.
                ctta = {}
                headline = "spline" if cfg.spline_true else "full"
                mode_order = (
                    [("full", False), ("spline", True)]
                    if cfg.spline_true
                    else [("spline", True), ("full", False)]
                )
                rng_ckpt = torch.get_rng_state()
                cuda_ckpt = (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                )

                for mode_name, sp in mode_order:
                    torch.set_rng_state(rng_ckpt)
                    if cuda_ckpt is not None:
                        torch.cuda.set_rng_state_all(cuda_ckpt)
                    model.load_state_dict(model_state)
                    preds_m, labels_m, traj_m = run_ctta(
                        model, stream_tgt_src_sc, pool_tgt_src_sc, device, cfg,
                        spline_only=sp, ewc=kan_ret, ewc_lambda=cfg.ewc_lambda,
                    )
                    # post-adaptation ('final') evaluation over the full stream.
                    final_f1 = evaluate(
                        model, stream_tgt_src_sc, device,
                        desc=f"CTTA/{mode_name} FINAL {tgt_name}",
                    )
                    # source retention — how much source performance survived.
                    src_after = evaluate(
                        model, loader_src_test, device,
                        desc=f"Source {src_name} after CTTA/{mode_name}",
                    )
                    ctta[mode_name] = {
                        "f1": f1_score(labels_m, preds_m, zero_division=0),
                        "acc": accuracy_score(labels_m, preds_m),
                        "final_f1": final_f1,
                        "src_after": src_after,
                        "traj": traj_m,
                    }
                    print(
                        f"[CTTA/{mode_name}] {tgt_name} - "
                        f'stream F1: {ctta[mode_name]["f1"]:.4f} | '
                        f"final F1: {final_f1:.4f} | "
                        f'Acc: {ctta[mode_name]["acc"]:.4f}'
                    )

                # AGSA validation: rerun the headline mode with the gate OFF.
                agsa_off = {}
                if cfg.agsa_ablation and arch == "kan":
                    torch.set_rng_state(rng_ckpt)
                    if cuda_ckpt is not None:
                        torch.cuda.set_rng_state_all(cuda_ckpt)
                    model.load_state_dict(model_state)
                    preds_n, labels_n, _ = run_ctta(
                        model, stream_tgt_src_sc, pool_tgt_src_sc, device, cfg,
                        spline_only=cfg.spline_true, ewc=kan_ret, ewc_lambda=cfg.ewc_lambda,
                        spline_gate=False,
                    )
                    agsa_off = {
                        "f1": f1_score(labels_n, preds_n, zero_division=0),
                        "final_f1": evaluate(
                            model, stream_tgt_src_sc, device,
                            desc=f"CTTA no-AGSA FINAL {tgt_name}",
                        ),
                        "src_after": evaluate(
                            model, loader_src_test, device,
                            desc=f"Source {src_name} after no-AGSA",
                        ),
                    }
                    print(
                        f"[AGSA ablation] gate ON  -> tgt(final) "
                        f'{ctta[headline]["final_f1"]:.4f} | src-after '
                        f'{ctta[headline]["src_after"]:.4f}'
                    )
                    print(
                        f"[AGSA ablation] gate OFF -> tgt(final) "
                        f'{agsa_off["final_f1"]:.4f} | src-after '
                        f'{agsa_off["src_after"]:.4f} | src-pre {src_f1_pre:.4f}'
                    )
                    # restore the headline-mode adapted state for visualizations
                    torch.set_rng_state(rng_ckpt)
                    if cuda_ckpt is not None:
                        torch.cuda.set_rng_state_all(cuda_ckpt)
                    model.load_state_dict(model_state)
                    run_ctta(
                        model, stream_tgt_src_sc, pool_tgt_src_sc, device, cfg,
                        spline_only=cfg.spline_true, ewc=kan_ret, ewc_lambda=cfg.ewc_lambda,
                    )

                trajectories[tgt_name] = ctta[headline]["traj"]

                retention_val = None
                if arch == "kan":
                    with torch.no_grad():  # diagnostic — no graph needed
                        retention_val = kan_ret.penalty().item()
                    print(f"[KAN Retention] Penalty ({headline}): {retention_val:.4f}")
                    run_all_visualizations(
                        model_state, model, stream_tgt_src_sc, pool_tgt_src_sc,
                        src_name, tgt_name, device, feature_names,
                    )

                # scratch baseline: same objectives + step budget from a RANDOM init.
                scratch = {}
                if cfg.scratch_baseline:
                    # pretrained + train-all — isolates the value of the pretrained init.
                    torch.set_rng_state(rng_ckpt)
                    if cuda_ckpt is not None:
                        torch.cuda.set_rng_state_all(cuda_ckpt)
                    model.load_state_dict(model_state)
                    preds_pa, labels_pa, _ = run_ctta(
                        model, stream_tgt_src_sc, pool_tgt_src_sc, device, cfg,
                        spline_only=False, train_all=True, spline_gate=False, restore_prob=0.0,
                    )
                    scratch["pre_all_final"] = evaluate(
                        model, stream_tgt_src_sc, device,
                        desc=f"Pretrained TRAIN-ALL CTTA FINAL {tgt_name}",
                    )

                    torch.set_rng_state(rng_ckpt)
                    if cuda_ckpt is not None:
                        torch.cuda.set_rng_state_all(cuda_ckpt)
                    rand_model = build_model(arch, input_dim, cfg).to(device)
                    rand_state = {k: v.clone() for k, v in rand_model.state_dict().items()}
                    # scratch few-shot: pool-only, all params, matched steps
                    scratch["fs"] = few_shot_baseline(
                        rand_state, pool_tgt_src_sc, stream_tgt_src_sc, input_dim, arch, device,
                        cfg, spline_only=False, epochs=fs_epochs, train_all=True,
                    )
                    # scratch CTTA: same stream objectives, all params
                    preds_r, labels_r, _ = run_ctta(
                        rand_model, stream_tgt_src_sc, pool_tgt_src_sc, device, cfg,
                        spline_only=False, train_all=True, spline_gate=False, restore_prob=0.0,
                    )
                    scratch["stream"] = f1_score(labels_r, preds_r, zero_division=0)
                    scratch["final"] = evaluate(
                        rand_model, stream_tgt_src_sc, device, desc=f"Scratch CTTA FINAL {tgt_name}"
                    )
                    print(
                        f'[Scratch] few-shot {scratch["fs"]:.4f} | '
                        f'CTTA stream {scratch["stream"]:.4f} | '
                        f'CTTA final {scratch["final"]:.4f}'
                    )
                    del rand_model, rand_state
                    gc.collect()

                results.append(
                    {
                        "Architecture": arch,
                        "Source": src_name,
                        "Target": tgt_name,
                        "Zero-Shot F1": zero_shot_f1,
                        "Few-Shot F1 (spline)": few_shot_spline_f1,
                        "Few-Shot F1 (full)": few_shot_full_f1,
                        "CTTA F1 stream (spline)": ctta["spline"]["f1"],
                        "CTTA F1 stream (full)": ctta["full"]["f1"],
                        "CTTA F1 final (spline)": ctta["spline"]["final_f1"],
                        "CTTA F1 final (full)": ctta["full"]["final_f1"],
                        "CTTA Acc (spline)": ctta["spline"]["acc"],
                        "CTTA Acc (full)": ctta["full"]["acc"],
                        "Scratch Few-Shot F1": scratch.get("fs"),
                        "Scratch CTTA F1 stream": scratch.get("stream"),
                        "Scratch CTTA F1 final": scratch.get("final"),
                        "Pretrained TrainAll CTTA F1 final": scratch.get("pre_all_final"),
                        "Src F1 pre": src_f1_pre,
                        "Src F1 after (spline)": ctta["spline"]["src_after"],
                        "Src F1 after (full)": ctta["full"]["src_after"],
                        "CTTA F1 final (no AGSA)": agsa_off.get("final_f1"),
                        "Src F1 after (no AGSA)": agsa_off.get("src_after"),
                        "Retention Penalty": retention_val,
                    }
                )

            if arch == "kan" and src_name == "CICIDS2018":
                plot_adaptation_trajectory(trajectories, src_name)

            # free the model + loaders between configs (GPU memory hygiene)
            del model, model_state, loader_src_train, loader_src_test, kan_ret
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Overall results
    results_df = pd.DataFrame(results)
    _show(results_df)
    results_df.to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)

    ablation_df = None
    if ablation_results:
        ablation_df = pd.DataFrame(ablation_results)[
            ["Architecture", "Source", "Target", "Table", "Config", "F1"]
        ]
        _show(ablation_df)
        ablation_df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)

    return results_df, ablation_df


def run_sequential(
    cfg: ExperimentConfig,
    archs=None,
    sequences=None,
    variants=None,
    datasets=None,
    out_dir=".",
):
    """Sequential / continual CTTA: pretrain on a source, then adapt through a
    SEQUENCE of target streams with NO reset. After pretraining and each stage,
    the model is evaluated on EVERY domain, yielding the continual-learning matrix
    from which average-F1 and per-domain forgetting can be read off.

    Returns ``seq_df`` and writes ``sequential_results.csv`` under ``out_dir``.
    """
    archs = archs if archs is not None else DEFAULT_SEQ_ARCHS
    sequences = sequences if sequences is not None else DEFAULT_SEQUENCES
    variants = variants if variants is not None else DEFAULT_SEQ_VARIANTS
    datasets = datasets if datasets is not None else DEFAULT_DATASETS
    device = get_device()
    os.makedirs(out_dir, exist_ok=True)

    seq_results = []
    for seq_arch in archs:
        for seq in sequences:
            seq_src = seq[0]
            seed_everything(cfg.seed)
            print("=" * 80)
            print(f"SEQUENTIAL | Arch: {seq_arch.upper()} | " + " -> ".join(seq))
            print("=" * 80)

            # Source pretraining
            X_s, y_s, seq_feat = load_dataset(datasets[seq_src], cfg)
            seq_input_dim = X_s.shape[1]
            seq_l_tr, seq_l_te, seq_scaler = make_source_loaders(X_s, y_s, cfg)
            # a small stratified SOURCE pool for replay variants.
            src_pool_n = max(int(round(len(y_s) * cfg.few_shot_ratio)), cfg.min_pool_size)
            Xp_s, _, yp_s, _ = train_test_split(
                X_s, y_s, train_size=src_pool_n, random_state=cfg.seed, stratify=y_s
            )
            Xp_s = seq_scaler.transform(Xp_s)
            seq_src_pool = DataLoader(
                TensorDataset(torch.from_numpy(Xp_s), torch.from_numpy(yp_s)),
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=seeded_generator(cfg.seed),
            )
            del X_s, y_s, Xp_s, yp_s
            gc.collect()
            seq_model = run_phase1_pretraining(
                seq_arch, seq_src, seq_input_dim, seq_l_tr, device, cfg
            )
            seq_state = {k: v.clone() for k, v in seq_model.state_dict().items()}

            # Loaders for every target in the sequence (fresh pool per domain).
            seq_pools, seq_streams = {}, {}
            for name in seq[1:]:
                X_t, y_t, _ = load_dataset(datasets[name], cfg, align_to=list(seq_feat))
                p, s = make_target_loaders(X_t, y_t, cfg, external_scaler=seq_scaler)
                seq_pools[name], seq_streams[name] = p, s
                del X_t, y_t
                gc.collect()

            seq_rng = torch.get_rng_state()
            seq_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            for variant in variants:
                var_name = variant["name"]
                torch.set_rng_state(seq_rng)
                if seq_cuda is not None:
                    torch.cuda.set_rng_state_all(seq_cuda)
                seq_model.load_state_dict(seq_state)

                # cumulative AGSA: persistent gates, built once per variant.
                cum_gates = []
                if variant["gate"] == "cumulative" and isinstance(seq_model, KanAEClassifier):
                    tp = (
                        get_spline_only_params(seq_model)
                        if cfg.spline_true
                        else get_trainable_params(seq_model)
                    )
                    tp_ids = {id(p) for p in tp}
                    for layer in seq_model.encoder.layers:
                        if any(id(p) in tp_ids for p in layer.parameters()):
                            try:
                                g = SplineActivationGate(layer, cfg)
                                g.capture = False  # no mass from stage-0 evals
                                cum_gates.append(g)
                            except ValueError:
                                pass

                def eval_all_domains(stage_label, extra=None):
                    row = {
                        "Architecture": seq_arch,
                        "Sequence": " -> ".join(seq),
                        "Source": seq_src,
                        "Target1": seq[1],
                        "Target2": seq[2],
                        "Variant": var_name,
                        "Stage": stage_label,
                    }
                    row["F1 source"] = evaluate(
                        seq_model, seq_l_te, device,
                        desc=f"[{var_name}|{stage_label}] {seq_src} (source)",
                    )
                    for pos, name in enumerate(seq[1:], start=1):
                        row[f"F1 target{pos}"] = evaluate(
                            seq_model, seq_streams[name], device,
                            desc=f"[{var_name}|{stage_label}] {name}",
                        )
                    if extra:
                        row.update(extra)
                    seq_results.append(row)

                eval_all_domains(f"0: pretrain({seq_src})")

                # NO reset between stages — this is the continual protocol.
                past_pools = [seq_src_pool] if variant["replay"] else []
                try:
                    for i, name in enumerate(seq[1:], start=1):
                        print(f"\n--- [{var_name}] Stage {i}: CTTA on {name} (no reset) ---")
                        current_pool = seq_pools[name]
                        if variant["replay"]:
                            stage_pool = build_replay_pool(
                                current_pool, past_pools, cfg, current_frac=cfg.replay_current_frac
                            )
                            print(
                                f"[Replay] pool = current({name}) at "
                                f"~{cfg.replay_current_frac:.0%} + "
                                f"{len(past_pools)} past domain(s) "
                                f"({len(stage_pool.dataset)} samples)"
                            )
                        else:
                            stage_pool = current_pool
                        # entropy anchor always uses the CURRENT domain's rate
                        stage_prior = pool_prior(seq_pools[name], device)

                        run_ctta(
                            seq_model, seq_streams[name], stage_pool, device, cfg,
                            spline_only=cfg.spline_true,
                            spline_gate=(False if variant["gate"] == "off" else None),
                            prior=stage_prior,
                            external_gates=(cum_gates or None)
                            if variant["gate"] == "cumulative"
                            else None,
                        )

                        extra = None
                        if variant["gate"] == "cumulative" and cum_gates:
                            fracs = []
                            for gi, g in enumerate(cum_gates):
                                cov = g.coverage()
                                pf = g.freeze_current()
                                fracs.append(pf)
                                print(
                                    f"[Cumulative AGSA] layer {gi}: "
                                    f"{cov:.1%} active this stage; "
                                    f"{pf:.1%} now protected (owned by "
                                    f"domains seen so far)."
                                )
                            extra = {"Protected frac": float(np.mean(fracs))}
                        past_pools.append(current_pool)  # becomes 'past' now
                        eval_all_domains(f"{i}: after CTTA {name}", extra=extra)
                finally:
                    for g in cum_gates:
                        g.remove()

            del seq_model, seq_state, seq_pools, seq_streams, seq_l_tr, seq_l_te, seq_src_pool
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seq_df = pd.DataFrame(seq_results)
    _show(seq_df)
    seq_df.to_csv(os.path.join(out_dir, "sequential_results.csv"), index=False)
    return seq_df
