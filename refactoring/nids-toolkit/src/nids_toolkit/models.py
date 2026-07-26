"""Model architectures (KAN / TabTransformer / CNN / FlowTransformer) and the
layer-selective parameter selectors used by CTTA.

Every classifier is an autoencoder-classifier whose ``forward`` returns the
triple ``(logits, recon, z)``, so they all run through the identical pretraining
/ CTTA / evaluation pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_kan import KAN

from .config import ExperimentConfig


class KanAEClassifier(nn.Module):
    """Shared KAN encoder -> classifier head + decoder head.

    Encoder: input_dim -> 64 -> latent_dim (KAN); Classifier: latent_dim -> 2;
    Decoder: latent_dim -> 64 -> input_dim (KAN). ``forward`` -> (logits, recon, z).
    """

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = KAN([input_dim, 64, latent_dim], grid_range=[-1, 1])
        self.ln = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder = KAN([latent_dim, 64, input_dim], grid_range=[-1, 1])

    def forward(self, x):
        z = self.ln(self.encoder(x))
        return self.classifier(z), self.decoder(z), z


class TabTransformerAEClassifier(nn.Module):
    """Shared TabTransformer encoder -> classifier head + decoder head.
    ``forward`` -> (logits, recon, z)."""

    def __init__(self, input_dim: int, latent_dim: int = 32, n_heads: int = 4, d_token: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        self.feature_proj = nn.Linear(1, d_token, bias=False)
        self.feature_bias = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.ln1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 4),
            nn.GELU(),
            nn.Linear(d_token * 4, d_token),
        )
        self.encoder_out = nn.Linear(d_token, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_token),
            nn.LayerNorm(d_token),
            nn.GELU(),
            nn.Linear(d_token, input_dim),
        )

    def forward(self, x):
        tokens = self.feature_proj(x.unsqueeze(-1)) + self.feature_bias.unsqueeze(0)
        tokens = self.ln1(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = tokens + attn_out
        tokens = self.ln2(tokens)
        tokens = tokens + self.ffn(tokens)
        pooled = tokens.mean(dim=1)
        z = self.ln(self.encoder_out(pooled))
        return self.classifier(z), self.decoder(z), z


class CnnAEClassifier(nn.Module):
    """Shared CNN encoder -> classifier head + decoder head.

    NOTE: Conv1d with kernel_size=1 over a length-1 sequence is mathematically
    identical to a Linear layer — this 'CNN' baseline is really an MLP with
    GroupNorm. Kept as-is for continuity with prior runs.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv1d(64, latent_dim, kernel_size=1),
        )
        self.ln = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=1),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=1),
        )

    def forward(self, x):
        z = self.ln(self.encoder(x.unsqueeze(-1)).squeeze(-1))
        recon = self.decoder(z.unsqueeze(-1)).squeeze(-1)
        return self.classifier(z), recon, z


class FlowTransformerEncoderBlock(nn.Module):
    """Transformer encoder block ported verbatim from FlowTransformer
    (liamdm/FlowTransformer). Reproduces the framework's exact wiring, which
    differs from a textbook post-norm block: both feed-forward layers carry a
    ReLU, and the second residual adds the *attention output* (pre-first-residual),
    i.e. ``LN(attention_output + FFN(x))``.
    """

    def __init__(self, d_model: int, inner_dim: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout_rate, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.attention_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.feed_forward_0 = nn.Linear(d_model, inner_dim)
        self.feed_forward_1 = nn.Linear(inner_dim, d_model)
        self.feed_forward_dropout = nn.Dropout(dropout_rate)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        attn, _ = self.attention(x, x, x, need_weights=False)
        attention_output = self.attention_dropout(attn) if self.dropout_rate > 0 else attn
        x = x + attention_output
        x = self.attention_layer_norm(x)
        x = F.relu(self.feed_forward_0(x))
        x = F.relu(self.feed_forward_1(x))
        x = self.feed_forward_dropout(x) if self.dropout_rate > 0 else x
        feed_forward_output = x
        return self.feed_forward_layer_norm(attention_output + feed_forward_output)


class FlowTransformerAEClassifier(nn.Module):
    """FlowTransformer baseline (Manocchio et al., 2024; liamdm/FlowTransformer),
    adapted to this study's per-flow tabular harness and wrapped in the shared
    encoder -> classifier + decoder AE layout. ``forward`` -> (logits, recon, z).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        d_model: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        inner_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_proj = nn.Linear(1, d_model, bias=False)
        self.feature_bias = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                FlowTransformerEncoderBlock(d_model, inner_dim, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        # CLS-token projection into the shared latent space — the model's "last
        # encoder layer" for layer-selective CTTA (cf. TabTransformer's encoder_out).
        self.encoder_out = nn.Linear(d_model, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, x):
        tokens = self.feature_proj(x.unsqueeze(-1)) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([tokens, cls], dim=1)  # CLS token appended last
        for blk in self.blocks:
            tokens = blk(tokens)
        pooled = tokens[:, -1, :]  # CLS-token classification head
        z = self.ln(self.encoder_out(pooled))
        return self.classifier(z), self.decoder(z), z


def build_model(arch: str, input_dim: int, cfg: ExperimentConfig) -> nn.Module:
    """Instantiate an architecture with ``cfg.latent_dim``."""
    latent_dim = cfg.latent_dim
    if arch == "kan":
        return KanAEClassifier(input_dim, latent_dim)
    elif arch == "cnn":
        return CnnAEClassifier(input_dim, latent_dim)
    elif arch == "tab":
        return TabTransformerAEClassifier(input_dim, latent_dim)
    elif arch in ("flowtransformer", "ft", "flowtr"):
        return FlowTransformerAEClassifier(input_dim, latent_dim)
    else:
        raise ValueError(
            f"Unknown arch={arch!r}. Choose 'kan', 'cnn', 'tab', or 'flowtransformer'"
        )


def get_trainable_params(model: nn.Module):
    """Parameters for layer-selective CTTA updates.

    Updated: all LayerNorm/GroupNorm params (outside decoder), classifier head,
    last encoder layer. Frozen: early encoder layers, decoder.
    """
    params = []
    seen = set()

    def add(p):
        if id(p) not in seen:
            seen.add(id(p))
            params.append(p)

    for name, module in model.named_modules():
        if "decoder" in name:
            continue
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            for p in module.parameters():
                add(p)

    for p in model.classifier.parameters():
        add(p)

    if isinstance(model, KanAEClassifier):
        if hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
            for p in model.encoder.layers[-1].parameters():
                add(p)
    elif isinstance(model, CnnAEClassifier):
        last_layer = None
        for m in model.encoder.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                last_layer = m
        if last_layer is not None:
            for p in last_layer.parameters():
                add(p)
    elif isinstance(model, TabTransformerAEClassifier):
        for p in model.encoder_out.parameters():
            add(p)
    elif isinstance(model, FlowTransformerAEClassifier):
        for p in model.encoder_out.parameters():
            add(p)

    return params


def get_spline_only_params(model):
    """Minimal-parameter trainable set for CTTA, per architecture.

    KAN: only spline_weight (+ spline_scaler) of the last encoder layer.
    CNN/Tab/FlowTransformer: last encoder projection ('last-layer-only' semantics).
    """
    params = []

    if isinstance(model, KanAEClassifier):
        if not hasattr(model.encoder, "layers") or len(model.encoder.layers) == 0:
            return []
        last = model.encoder.layers[-1]
        if hasattr(last, "spline_weight") and isinstance(last.spline_weight, torch.nn.Parameter):
            params.append(last.spline_weight)
        if hasattr(last, "spline_scaler") and isinstance(last.spline_scaler, torch.nn.Parameter):
            params.append(last.spline_scaler)

    elif isinstance(model, CnnAEClassifier):
        last_layer = None
        for m in model.encoder.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                last_layer = m
        if last_layer is not None:
            params.extend(last_layer.parameters())

    elif isinstance(model, TabTransformerAEClassifier):
        params.extend(model.encoder_out.parameters())

    elif isinstance(model, FlowTransformerAEClassifier):
        params.extend(model.encoder_out.parameters())

    return params
