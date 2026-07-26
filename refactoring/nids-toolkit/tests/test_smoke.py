"""CPU synthetic-data smoke test.

No Kaggle download and no GPU required: builds each architecture on random
tensors and drives it through pretrain -> zero-shot -> few-shot -> CTTA ->
evaluate, plus the AGSA path and a two-model merge. This does NOT check research
numbers — it checks that the config-threaded wiring holds together end to end,
which is the main risk of the refactor.

Run with:  pytest refactoring/nids-toolkit/tests
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("efficient_kan")

from torch.utils.data import DataLoader, TensorDataset

from nids_toolkit import (
    ExperimentConfig,
    SplineActivationGate,
    build_model,
    evaluate,
    few_shot_baseline,
    make_criterion,
    merging,
    pretrain_source,
    run_ctta,
)
from nids_toolkit.data import reset_loader_generator
from nids_toolkit.models import KanAEClassifier

ARCHS = ["kan", "cnn", "tab", "flowtransformer"]
INPUT_DIM = 12
DEVICE = "cpu"


def _tiny_cfg() -> ExperimentConfig:
    # small, fast settings that still exercise every code path
    return ExperimentConfig(
        seed=0,
        pretrain_epochs=1,
        latent_dim=8,
        batch_size=32,
        tta_batch_size=64,
        min_pool_size=32,
        few_shot_ratio=0.2,
    )


def _synthetic(n=400, seed=0):
    """Two Gaussian blobs -> a learnable binary task, class-imbalanced ~25% attack."""
    rng = np.random.default_rng(seed)
    n_attack = n // 4
    n_benign = n - n_attack
    X0 = rng.normal(-0.3, 0.3, size=(n_benign, INPUT_DIM)).astype(np.float32)
    X1 = rng.normal(+0.3, 0.3, size=(n_attack, INPUT_DIM)).astype(np.float32)
    X = np.clip(np.concatenate([X0, X1]), -1, 1).astype(np.float32)
    y = np.concatenate([np.zeros(n_benign), np.ones(n_attack)]).astype(np.int64)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def _loader(X, y, batch_size, shuffle, cfg):
    from nids_toolkit.utils import seeded_generator

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = seeded_generator(cfg.seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@pytest.mark.parametrize("arch", ARCHS)
def test_arch_end_to_end(arch):
    cfg = _tiny_cfg()
    X, y = _synthetic(seed=1)
    Xt, yt = _synthetic(seed=2)  # "target" domain

    src_train = _loader(X, y, cfg.batch_size, True, cfg)
    src_test = _loader(X, y, cfg.tta_batch_size, False, cfg)
    pool = _loader(Xt[:64], yt[:64], cfg.batch_size, True, cfg)
    stream = _loader(Xt, yt, cfg.tta_batch_size, False, cfg)

    model = build_model(arch, INPUT_DIM, cfg).to(DEVICE)

    # forward returns the (logits, recon, z) triple
    logits, recon, z = model(torch.from_numpy(X[:8]))
    assert logits.shape == (8, 2)
    assert recon.shape == (8, INPUT_DIM)

    pretrain_source(model, src_train, epochs=cfg.pretrain_epochs, device=DEVICE, cfg=cfg)

    state = {k: v.clone() for k, v in model.state_dict().items()}

    # zero-shot
    evaluate(model, stream, DEVICE, desc=f"zero-shot {arch}")

    # few-shot baseline (spline-only + full)
    fs = few_shot_baseline(state, pool, stream, INPUT_DIM, arch, DEVICE, cfg, spline_only=True, epochs=1)
    assert 0.0 <= fs <= 1.0

    # CTTA — spline-only (headline)
    model.load_state_dict(state)
    preds, labels, traj = run_ctta(model, stream, pool, DEVICE, cfg, spline_only=True)
    assert len(preds) == len(labels) == len(yt)
    assert isinstance(traj, list) and len(traj) >= 1

    # CTTA — full trainable subset
    model.load_state_dict(state)
    run_ctta(model, stream, pool, DEVICE, cfg, spline_only=False)


def test_agsa_gate_and_ablation_flag():
    """KAN-specific: AGSA on/off both run, and the gate reports coverage."""
    cfg = _tiny_cfg()
    Xt, yt = _synthetic(seed=3)
    pool = _loader(Xt[:64], yt[:64], cfg.batch_size, True, cfg)
    stream = _loader(Xt, yt, cfg.tta_batch_size, False, cfg)

    model = build_model("kan", INPUT_DIM, cfg).to(DEVICE)
    assert isinstance(model, KanAEClassifier)
    src = _loader(*_synthetic(seed=4), batch_size=cfg.batch_size, shuffle=True, cfg=cfg)
    pretrain_source(model, src, epochs=1, device=DEVICE, cfg=cfg)
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # gate ON
    model.load_state_dict(state)
    run_ctta(model, stream, pool, DEVICE, cfg, spline_only=True, spline_gate=True)
    # gate OFF (AGSA ablation path)
    model.load_state_dict(state)
    run_ctta(model, stream, pool, DEVICE, cfg, spline_only=True, spline_gate=False)

    # a standalone gate reports a coverage fraction after forwards
    gate = SplineActivationGate(model.encoder.layers[-1], cfg)
    with torch.no_grad():
        model(torch.from_numpy(Xt[:32]))
    cov = gate.coverage()
    gate.remove()
    assert (cov != cov) or (0.0 <= cov <= 1.0)  # nan (warm-up) or a fraction


def test_merging_primitives():
    cfg = _tiny_cfg()
    model = build_model("kan", INPUT_DIM, cfg)
    sd_src = merging.snap_state(model)
    # perturb to simulate two adapted copies
    sd_a = {k: v.clone() for k, v in sd_src.items()}
    sd_b = {k: v.clone() for k, v in sd_src.items()}
    for sd in (sd_a, sd_b):
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k] = sd[k] + 0.01 * torch.randn_like(sd[k])

    assert merging.diff_keys(sd_src, sd_a)  # something changed
    mid = merging.interp_states(sd_src, sd_a, 0.5)
    assert set(mid.keys()) == set(sd_src.keys())

    soup = merging.uniform_soup(sd_src, [sd_a, sd_b])
    arith = merging.task_arithmetic(sd_src, [sd_a, sd_b])
    assert set(soup.keys()) == set(sd_src.keys()) == set(arith.keys())

    # both merges load cleanly back into a model of the same shape
    model.load_state_dict(soup)
    model.load_state_dict(arith)


def test_criterion_variants():
    cfg = ExperimentConfig(loss_type="focal")
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    crit = make_criterion(y, DEVICE, cfg)
    logits = torch.randn(4, 2)
    loss = crit(logits, torch.from_numpy(y))
    assert torch.isfinite(loss)


def test_reset_loader_generator_is_deterministic():
    cfg = _tiny_cfg()
    X, y = _synthetic(seed=5)
    loader = _loader(X, y, cfg.batch_size, True, cfg)
    reset_loader_generator(loader, cfg.seed)
    first = next(iter(loader))[0].clone()
    reset_loader_generator(loader, cfg.seed)
    second = next(iter(loader))[0].clone()
    assert torch.equal(first, second)
