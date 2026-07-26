"""Weight-space model-merging primitives used by the SSR merging experiment.

These are the reusable building blocks (state-dict snapshot/diff/interpolation
and three merge strategies, including the AGSA-routed merge). The higher-level
experiment orchestration — fingerprint recording, window-SSR routing, the joint
baseline — lives in the SSR notebook, which composes these primitives.
"""

from __future__ import annotations

import torch

from .config import ExperimentConfig


def snap_state(model) -> dict:
    """CPU snapshot of a model's state dict (detached clones)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def diff_keys(sd_a: dict, sd_b: dict) -> list:
    """State-dict keys whose tensors differ between two snapshots (same shape)."""
    return [
        k
        for k in sd_a
        if sd_a[k].shape == sd_b[k].shape and not torch.equal(sd_a[k], sd_b[k])
    ]


def interp_states(sd_src: dict, sd_tgt: dict, alpha: float) -> dict:
    """``theta(alpha) = theta_src + alpha * (theta_tgt - theta_src)`` elementwise.

    A zero-data retention dial: ``alpha=0`` is the source model, ``alpha=1`` the
    adapted model. Only floating-point tensors are interpolated.
    """
    out = {k: v.clone() for k, v in sd_src.items()}
    for k in sd_src:
        if sd_src[k].dtype.is_floating_point:
            out[k] = sd_src[k] + alpha * (sd_tgt[k] - sd_src[k])
    return out


def active_mask(mass, cfg: ExperimentConfig | None = None, thresh: float | None = None):
    """Region-activation mask from an AGSA activation-mass tensor, using the same
    rule as :class:`~nids_toolkit.adaptation.SplineActivationGate`: a coefficient
    is active if its mass exceeds ``thresh`` of the per-input-dim maximum.
    """
    if thresh is None:
        thresh = cfg.spline_gate_thresh if cfg is not None else 0.05
    return mass >= thresh * mass.max(dim=-1, keepdim=True).values


def uniform_soup(sd_src: dict, adapted_sds) -> dict:
    """Model soup: ``theta_src + mean_i(Delta_i)`` over the changed tensors."""
    sd = {k: v.clone() for k, v in sd_src.items()}
    changed = set()
    for a in adapted_sds:
        changed |= set(diff_keys(sd_src, a))
    n = len(adapted_sds)
    for k in changed:
        sd[k] = sd_src[k] + sum((a[k] - sd_src[k]) for a in adapted_sds) / n
    return sd


def task_arithmetic(sd_src: dict, adapted_sds) -> dict:
    """Task arithmetic: ``theta_src + sum_i(Delta_i)`` (full-strength sum)."""
    sd = {k: v.clone() for k, v in sd_src.items()}
    changed = set()
    for a in adapted_sds:
        changed |= set(diff_keys(sd_src, a))
    for k in changed:
        sd[k] = sd_src[k] + sum((a[k] - sd_src[k]) for a in adapted_sds)
    return sd


def agsa_routed_merge(sd_src: dict, sd_a: dict, sd_b: dict, mask_a, mask_b) -> dict:
    """Locality-aware two-domain merge (KAN-only).

    Per spline coefficient, the domain that ACTIVATED a basis region owns it;
    contested regions (activated by both) are averaged; regions neither domain
    visited stay at the source value. Non-spline tensors fall back to the uniform
    soup of the two adapted models.

    ``mask_a`` / ``mask_b`` are ``(in_dim, n_coeff)`` boolean activation masks
    (see :func:`active_mask`); ``spline_weight`` tensors are ``(out, in, coeff)``.
    """
    sd = uniform_soup(sd_src, [sd_a, sd_b])  # default for non-spline tensors
    for k in diff_keys(sd_src, sd_a):
        if not k.endswith("spline_weight"):
            continue
        if sd_src[k].shape[1:] != mask_a.shape:  # (out, in, coeff) vs (in, coeff)
            print(f"[merge] shape mismatch on {k}, leaving uniform")
            continue
        only_a, only_b, both = (mask_a & ~mask_b), (mask_b & ~mask_a), (mask_a & mask_b)
        w = sd_src[k].clone()
        w[:, only_a] = sd_a[k][:, only_a]
        w[:, only_b] = sd_b[k][:, only_b]
        w[:, both] = 0.5 * sd_a[k][:, both] + 0.5 * sd_b[k][:, both]
        sd[k] = w  # dormant regions stay at source
    return sd
