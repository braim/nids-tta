"""Test-time-adaptation building blocks: stochastic restore, the AGSA spline
gate (the paper's novelty), and KAN-specific EWC retention."""

from __future__ import annotations

import torch

from .config import ExperimentConfig
from .losses import labels_of, make_criterion


class StochasticRestore:
    """CoTTA-style stochastic restore (Wang et al., 2022). After each optimizer
    step, every element of every trainable tensor is reset to its source
    (post-pretrain) value with probability ``prob``.

    Over a long stream (~2000 updates) small per-step errors compound — the
    dominant failure of unsupervised objectives (the entropy-only 'F1 = 0.01'
    collapse). Stochastic restore makes total drift mean-reverting toward the
    source weights without freezing anything, and unlike a hard periodic reset
    it never discards adaptation wholesale.
    """

    def __init__(self, params, prob: float):
        self.prob = float(prob)
        self.params = list(params)
        self.source = [p.data.clone() for p in self.params]

    @torch.no_grad()
    def __call__(self):
        if self.prob <= 0:
            return
        for p, s in zip(self.params, self.source):
            mask = torch.rand_like(p) < self.prob
            p.data[mask] = s[mask]


class SplineActivationGate:
    """Activation-Gated Spline Adaptation (AGSA) — KAN-specific TTA novelty.

    Mechanism:
      * a forward pre-hook on a KANLinear layer captures its input on every
        forward pass — including the label-free predict pass, so activation
        statistics come for free, at zero extra compute;
      * the layer's own ``b_splines()`` gives per-(input-dim, coefficient) basis
        activation; an EMA accumulates this 'visitation mass' over the stream;
      * a gradient hook on ``spline_weight`` zero-masks coefficients whose mass is
        below ``thresh`` of the per-dim maximum.

    Active coefficients pass their gradient through unchanged; dormant ones
    receive only basis-tail noise (which Adam would renormalise into full-size
    updates), so cutting that path keeps the function outside the target's
    support at its source shape — retention by architecture, not by penalty.

    ``ema``/``thresh`` default from ``cfg`` when a config is supplied.
    """

    def __init__(
        self,
        layer,
        cfg: ExperimentConfig | None = None,
        ema: float | None = None,
        thresh: float | None = None,
    ):
        if not (hasattr(layer, "b_splines") and hasattr(layer, "spline_weight")):
            raise ValueError(
                "SplineActivationGate needs an efficient-kan KANLinear-style "
                "layer (b_splines + spline_weight)."
            )
        if ema is None:
            ema = cfg.spline_gate_ema if cfg is not None else 0.99
        if thresh is None:
            thresh = cfg.spline_gate_thresh if cfg is not None else 0.05
        self.layer = layer
        self.ema = float(ema)
        self.thresh = float(thresh)
        self.mass = None  # (in_features, n_coeff) EMA of |basis|
        # cumulative AGSA: coefficients owned by previously-adapted domains.
        self.protected = None
        # capture switch: external (cumulative) gates stay attached across a
        # whole domain sequence; capture is enabled only while the current
        # adaptation stage runs so mass reflects the current domain.
        self.capture = True
        self._h_in = layer.register_forward_pre_hook(self._capture)
        self._h_grad = layer.spline_weight.register_hook(self._gate)

    def _capture(self, module, inputs):
        if not self.capture:
            return
        x = inputs[0]
        if x.dim() != 2:  # defensive: expect (batch, in)
            return
        with torch.no_grad():
            b = module.b_splines(x.detach()).abs().mean(dim=0)  # (in, n_coeff)
            self.mass = (
                b
                if self.mass is None
                else self.ema * self.mass + (1.0 - self.ema) * b
            )

    def mask(self):
        if self.mass is None:
            return None
        ref = self.mass.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
        return (self.mass > self.thresh * ref).to(self.layer.spline_weight.dtype)

    def _gate(self, grad):
        m = self.mask()
        allow = m  # activation-localised updates
        if self.protected is not None:
            # cumulative AGSA: zero gradient on coefficients owned by earlier
            # domains, regardless of whether the current domain activates them.
            keep = (~self.protected).to(grad.dtype)
            allow = keep if allow is None else allow * keep
        if allow is None:
            return grad  # warm-up: ungated
        # spline_weight: (out, in, n_coeff); allow: (in, n_coeff) — broadcast
        return grad * allow.unsqueeze(0)

    def freeze_current(self) -> float:
        """Call at the END of a domain's adaptation stage. Folds the current
        domain's activation mask into the protected set (union) and resets the
        mass EMA so the next stage measures the next domain from scratch.
        Returns the protected fraction.

        NOTE: exact freezing requires a FRESH optimizer for the next stage —
        ``run_ctta`` constructs a new optimizer on every call, so the sequential
        protocol satisfies this by construction.
        """
        m = self.mask()
        if m is not None:
            mb = m.bool()
            self.protected = mb if self.protected is None else (self.protected | mb)
        self.mass = None
        return 0.0 if self.protected is None else float(self.protected.float().mean())

    def coverage(self) -> float:
        m = self.mask()
        return float("nan") if m is None else float(m.mean())

    def remove(self):
        self._h_in.remove()
        self._h_grad.remove()


class KAN_Retention:
    """KAN-specific Elastic Weight Consolidation (EWC).

    Computes a Fisher-information matrix over source data once; ``penalty()``
    returns ``sum(Fisher * (theta - theta_source)^2)``, which ``run_ctta`` adds to
    the loss when ``cfg.ewc_lambda > 0``.
    """

    def __init__(self, model, dataloader, device, params_to_protect, cfg: ExperimentConfig):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.cfg = cfg
        self.params_to_protect = params_to_protect

        self.optpar_dict = {
            id(p): p.data.clone().detach() for p in self.params_to_protect
        }

        print("[KAN Retention] Computing Fisher Information on Source Data...")
        self.fisher_dict = self._compute_fisher()
        print("[KAN Retention] Initialization Complete.")

    def _compute_fisher(self):
        fisher_dict = {
            id(p): torch.zeros_like(p.data) for p in self.params_to_protect
        }
        # remember/restore requires_grad flags so we don't leak state.
        prev_flags = {id(p): p.requires_grad for p in self.params_to_protect}
        for p in self.params_to_protect:
            p.requires_grad_(True)

        was_training = self.model.training
        self.model.eval()
        # compute Fisher under the same imbalance-aware criterion used in training.
        ce_crit = make_criterion(labels_of(self.dataloader), self.device, self.cfg)

        num_samples = 0
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits, _, _ = self.model(x)
            loss = ce_crit(logits, y)
            loss.backward()
            for p in self.params_to_protect:
                if p.grad is not None:
                    fisher_dict[id(p)] += (p.grad.data ** 2) * x.size(0)
            num_samples += x.size(0)

        for p in self.params_to_protect:
            fisher_dict[id(p)] /= max(num_samples, 1)
            p.requires_grad_(prev_flags[id(p)])
        self.model.zero_grad(set_to_none=True)  # don't leave stale grads
        if was_training:
            self.model.train()
        return fisher_dict

    def penalty(self):
        """EWC penalty: ``sum(Fisher * (theta - theta_source)^2)``."""
        loss = torch.tensor(0.0, device=self.device)
        for p in self.params_to_protect:
            fisher = self.fisher_dict[id(p)]
            optpar = self.optpar_dict[id(p)]
            loss = loss + (fisher * (p - optpar) ** 2).sum()
        return loss
