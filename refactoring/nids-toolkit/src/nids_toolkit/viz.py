"""Visualisation helpers: logit separation, KAN spline activations, feature
importance / Sankey flow, and adaptation trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch


def plot_logit_separation(model, loader, device, title, ax):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _, _ = model(x.to(device))
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    margin = logits[:, 1] - logits[:, 0]  # attack logit minus benign logit

    ax.hist(margin[y == 0], bins=80, alpha=0.6, color="#2196F3", label="Benign", density=True)
    ax.hist(margin[y == 1], bins=80, alpha=0.6, color="#F44336", label="Attack", density=True)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Attack logit − Benign logit")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def visualize_kan_layer(model, layer_idx, feature_names, ax_row, row_title, device, top_k=4):
    """Plot the KAN spline response curves for the top-``top_k`` input dims of an
    encoder layer. ``device`` is explicit so the function is self-contained.
    """
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        return
    layer = encoder.layers[layer_idx]

    with torch.no_grad():
        if hasattr(layer, "scaled_spline_weight"):
            w = layer.scaled_spline_weight.cpu().numpy()
        elif hasattr(layer, "spline_weight"):
            w = layer.spline_weight.cpu().numpy()
        else:
            return
        importance = np.sum(np.abs(w), axis=(0, 2))
        top_indices = np.argsort(importance)[-top_k:][::-1]

    for i, (ax, feat_idx) in enumerate(zip(ax_row, top_indices)):
        x_in = torch.zeros(200, w.shape[1], device=device)
        x_in[:, feat_idx] = torch.linspace(-1, 1, 200)
        with torch.no_grad():
            out = layer(x_in).cpu().numpy()

        out_imp = np.sum(np.abs(w[:, feat_idx, :]), axis=-1)
        top_out_idx = np.argsort(out_imp)[-3:][::-1]

        for out_idx in top_out_idx:
            ax.plot(np.linspace(-1, 1, 200), out[:, out_idx], alpha=0.7, label=f"→ h{out_idx}")
        name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        ax.set_title(f"{name}\n(imp: {importance[feat_idx]:.2f})", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel(row_title, fontsize=9, fontweight="bold")
            ax.legend(fontsize=6)


def get_raw_feature_importance(model, layer_idx):
    """Per-(output, input) importance of a KAN encoder layer, or ``None`` when no
    weights are available (returns None consistently so callers can guard)."""
    layer = model.encoder.layers[layer_idx]
    with torch.no_grad():
        w_spline = None
        if hasattr(layer, "scaled_spline_weight"):
            w_spline = layer.scaled_spline_weight.cpu().numpy()
        elif hasattr(layer, "spline_weight"):
            w_spline = layer.spline_weight.cpu().numpy()

        spline_imp = np.sum(np.abs(w_spline), axis=2) if w_spline is not None else None

        base_imp = None
        if hasattr(layer, "base_weight") and layer.base_weight is not None:
            base_imp = np.abs(layer.base_weight.cpu().numpy())

    if spline_imp is None and base_imp is None:
        return None
    if spline_imp is None:
        return base_imp
    if base_imp is None:
        return spline_imp
    return spline_imp + base_imp


def trace_importance_to_features(model, feature_names):
    """Chain first- and last-layer importance to score raw input features."""
    W1 = get_raw_feature_importance(model, 0)  # (Hidden, Input)
    W2 = get_raw_feature_importance(model, -1)  # (Latent, Hidden)
    if W1 is None or W2 is None:
        return np.zeros(len(feature_names))
    chained = W2 @ W1
    return np.sum(chained, axis=0)


def plot_importance_sankey(model, feature_names, title):
    W1 = get_raw_feature_importance(model, 0)  # (hidden, input)
    W2 = get_raw_feature_importance(model, -1)  # (latent, hidden)
    if W1 is None or W2 is None:
        print("[Sankey] No importance weights available for this model — skipped.")
        return

    input_imp = W1.sum(axis=0)
    top_inputs = np.argsort(input_imp)[-8:][::-1]

    hidden_imp = W1.sum(axis=1) + W2.sum(axis=0)
    top_hidden = np.argsort(hidden_imp)[-8:][::-1]

    latent_imp = W2.sum(axis=1)
    top_latents = np.argsort(latent_imp)[-6:][::-1]

    n_i, n_h, n_l = len(top_inputs), len(top_hidden), len(top_latents)

    input_labels = [
        feature_names[i] if i < len(feature_names) else f"Feat {i}" for i in top_inputs
    ]
    hidden_labels = [f"Hidden {i}" for i in top_hidden]
    latent_labels = [f"Latent {i}" for i in top_latents]
    all_labels = input_labels + hidden_labels + latent_labels

    x_pos = [0.01] * n_i + [0.5] * n_h + [0.99] * n_l

    def spread(n):
        if n == 1:
            return [0.5]
        return [0.05 + 0.9 * i / (n - 1) for i in range(n)]

    y_pos = spread(n_i) + spread(n_h) + spread(n_l)

    node_colors = (
        ["rgba(33, 150, 243, 0.8)"] * n_i
        + ["rgba(156, 39, 176, 0.8)"] * n_h
        + ["rgba(244, 67, 54, 0.8)"] * n_l
    )

    sources, targets, values, link_colors = [], [], [], []

    for hi, h_idx in enumerate(top_hidden):
        for ii, i_idx in enumerate(top_inputs):
            w = W1[h_idx, i_idx]
            if w > 0.1:
                sources.append(ii)
                targets.append(n_i + hi)
                values.append(float(w))
                link_colors.append("rgba(33, 150, 243, 0.12)")

    for li, l_idx in enumerate(top_latents):
        for hi, h_idx in enumerate(top_hidden):
            w = W2[l_idx, h_idx]
            if w > 0.05:
                sources.append(n_i + hi)
                targets.append(n_i + n_h + li)
                values.append(float(w))
                link_colors.append("rgba(244, 67, 54, 0.12)")

    if not sources:  # plotly errors on an empty Sankey
        print("[Sankey] No links above threshold — skipped.")
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=20,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color=node_colors,
                    x=x_pos,
                    y=y_pos,
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
            )
        ]
    )

    fig.update_layout(
        title_text=title,
        font_size=11,
        width=1000,
        height=550,
        annotations=[
            dict(x=0.01, y=1.08, text="<b>Input Features</b>", showarrow=False,
                 font=dict(size=12, color="#2196F3"), xref="paper", yref="paper"),
            dict(x=0.5, y=1.08, text="<b>Hidden Layer</b>", showarrow=False,
                 font=dict(size=12, color="#9C27B0"), xref="paper", yref="paper"),
            dict(x=0.99, y=1.08, text="<b>Latent Layer</b>", showarrow=False,
                 font=dict(size=12, color="#F44336"), xref="paper", yref="paper"),
        ],
    )
    fig.show()


def plot_adaptation_trajectory(trajectories, src_name):
    plt.figure(figsize=(10, 5))
    for tgt_name, traj in trajectories.items():
        if traj:
            batches, f1_scores = zip(*traj)
            plt.plot(batches, f1_scores, marker="o", label=f"Target: {tgt_name}")
    plt.title(f"CTTA Adaptation Trajectory (Source: {src_name})", fontsize=13)
    plt.xlabel("Batches Processed")
    plt.ylabel("Streaming F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_all_visualizations(
    model_state, model, stream_loader, pool_loader, src_name, tgt_name, device, feature_names
):
    print(f"\n--- Running KAN Visualizations: {src_name} -> {tgt_name} ---")
    after_state = {k: v.clone() for k, v in model.state_dict().items()}

    try:
        # 1. Logit separation (one selected target)
        if tgt_name == "UNSW-NB15":
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.suptitle(f"KAN — Logit Separation ({src_name} -> {tgt_name})", fontsize=13)
            model.load_state_dict(model_state)
            plot_logit_separation(model, stream_loader, device, "Before CTTA", axes[0])
            model.load_state_dict(after_state)
            plot_logit_separation(model, stream_loader, device, "After CTTA", axes[1])
            plt.tight_layout()
            plt.show()

        # 2. KAN spline activations (one selected target)
        if tgt_name == "CICIDS2018":
            fig, axes = plt.subplots(4, 4, figsize=(16, 14))
            fig.suptitle(
                f"KAN Spline Activations — Before vs After CTTA ({src_name} -> {tgt_name})",
                fontsize=13,
            )
            # the last encoder layer's INPUTS are the 64 hidden units, not latents.
            hidden_names = [f"Hidden {i}" for i in range(64)]
            model.load_state_dict(model_state)
            visualize_kan_layer(model, 0, feature_names, axes[0], "First Layer\n(Before)", device, top_k=4)
            visualize_kan_layer(model, -1, hidden_names, axes[1], "Last Layer\n(Before)", device, top_k=4)
            model.load_state_dict(after_state)
            visualize_kan_layer(model, 0, feature_names, axes[2], "First Layer\n(After)", device, top_k=4)
            visualize_kan_layer(model, -1, hidden_names, axes[3], "Last Layer\n(After)", device, top_k=4)
            plt.tight_layout()
            plt.show()

        # 3. Sankey diagram
        if tgt_name == "ToN-IoT" and src_name == "CICIDS2018":
            model.load_state_dict(model_state)
            plot_importance_sankey(model, feature_names, f"KAN Encoder Flow (Before: {src_name})")
            model.load_state_dict(after_state)
            plot_importance_sankey(model, feature_names, f"KAN Encoder Flow (After: {tgt_name})")
    except Exception as e:
        print(f"Visualization skipped or failed: {e}")
    finally:
        model.load_state_dict(after_state)
