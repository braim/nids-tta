"""Class-imbalance criteria and the robust CTTA stream objectives."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


def compute_class_weights(y, cfg: ExperimentConfig) -> torch.Tensor:
    """Inverse-frequency ('balanced') weights ``w_c = N / (n_classes * N_c)``,
    capped at ``cfg.max_class_weight``. Accepts a numpy array or torch tensor.
    Falls back to uniform weights if a class is absent (possible in a tiny pool)
    so the criterion never divides by zero.
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    if (counts == 0).any():
        print(
            "   [class weights] WARNING: a class is missing from the labelled "
            "data — falling back to uniform weights."
        )
        return torch.ones(2, dtype=torch.float32)
    w = len(y) / (2.0 * counts)
    w = np.minimum(w, cfg.max_class_weight)
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) for class-imbalanced classification.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``. ``alpha`` is a per-class
    weight tensor (here: the same inverse-frequency weights as 'weighted_ce');
    ``gamma`` down-weights easy examples. Reduces to alpha-weighted CE at gamma=0.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        # registered as a buffer so .to(device) moves it with the module
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(2))

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        alpha_t = self.alpha.to(logits.device)[targets]
        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()


def make_criterion(y, device, cfg: ExperimentConfig) -> nn.Module:
    """Build the classification criterion per ``cfg.loss_type``, with class
    statistics taken from the labelled data actually used in that phase (source
    train split for pretraining; labelled pool for CTTA and the few-shot
    baseline). Centralised so every training path stays consistent.
    """
    if cfg.loss_type == "ce":
        return nn.CrossEntropyLoss()
    weights = compute_class_weights(y, cfg).to(device)
    if cfg.loss_type == "weighted_ce":
        print(
            f"   [criterion] weighted CE | weights="
            f"[benign {weights[0]:.2f}, attack {weights[1]:.2f}]"
        )
        return nn.CrossEntropyLoss(weight=weights)
    if cfg.loss_type == "focal":
        print(
            f"   [criterion] focal (gamma={cfg.focal_gamma}) | alpha="
            f"[benign {weights[0]:.2f}, attack {weights[1]:.2f}]"
        )
        return FocalLoss(alpha=weights, gamma=cfg.focal_gamma).to(device)
    raise ValueError(
        f"Unknown loss_type={cfg.loss_type!r}. Choose 'ce', 'weighted_ce', or 'focal'"
    )


def labels_of(loader) -> torch.Tensor:
    """Pull the label tensor back out of a TensorDataset-backed loader so
    criteria can be built from exactly the data a phase trains on."""
    return loader.dataset.tensors[1]


def pool_prior(loader, device) -> torch.Tensor:
    """Class prior estimated from the labelled pool. Because the pool is a
    stratified sample of the target, its attack rate is an unbiased estimate of
    the stream's — the one piece of distributional knowledge CTTA may use.
    Clamped away from 0 so the KL term is always finite.
    """
    y = labels_of(loader).cpu().numpy()
    r = float(np.mean(y))
    prior = torch.tensor([1.0 - r, r], dtype=torch.float32, device=device)
    return prior.clamp_min(1e-3)


def stream_entropy_loss(logits_s, prior, cfg: ExperimentConfig):
    """Robust replacement for plain entropy minimisation, which degrades CTTA.

    Remedies applied together (see Niu et al. 2022 / EATA, SHOT):
      * reliability filter: only samples with entropy below
        ``E0 = cfg.ent_conf_frac * ln(2)`` contribute (ln(2) = max binary entropy);
      * confidence weighting: contributions scaled by ``exp(E0 - e)``, so the most
        confident samples dominate;
      * prior anchor: ``KL(batch-mean prediction || pool prior)`` penalises drift of
        the predicted class ratio away from the pool's attack rate.

    Returns a scalar loss (0 contribution from the entropy term if no sample
    passes the filter — the KL anchor is always active).
    """
    E0 = cfg.ent_conf_frac * float(np.log(2.0))
    log_probs = F.log_softmax(logits_s, dim=1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=1)

    mask = ent < E0
    if mask.any():
        w = torch.exp(E0 - ent[mask]).detach()  # EATA weights — no grad through w
        loss_ent = (w * ent[mask]).sum() / w.sum()
    else:
        loss_ent = logits_s.new_zeros(())

    p_bar = probs.mean(dim=0).clamp_min(1e-8)
    loss_kl = (p_bar * (p_bar / prior).log()).sum()

    return cfg.entropy_w * loss_ent + cfg.marginal_w * loss_kl
