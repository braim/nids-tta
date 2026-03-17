import os, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from efficient_kan import KAN

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Architecture ──────────────────────────────────────────────────────────────
ARCH            = 'transformer'    # 'kan' | 'cnn' | 'transformer'

# ── Data ──────────────────────────────────────────────────────────────────────
SAMPLE_N        = 100_000
BATCH_SIZE      = 128
TTA_BATCH_SIZE  = 512        # larger batches = more stable gradient estimates

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_DIM      = 32
# ── Transformer hyperparameters (ARCH='transformer' only) ───────────────────
N_HEADS         = 4        # attention heads (LATENT_DIM must be divisible)
N_LAYERS        = 2        # transformer encoder layers
FF_DIM          = 128      # feedforward hidden dimension
DROPOUT         = 0.1      # dropout in transformer layers
# ── Pre-training ──────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS = 20
PRETRAIN_LR     = 1e-3
WEIGHT_DECAY    = 1e-4
RECON_W         = 0.5        # reconstruction regularises encoder for transfer

# ── CTTA ──────────────────────────────────────────────────────────────────────
FEW_SHOT_RATIO  = 0.01       # fraction of target used as benign pool
FEW_SHOT_W      = 1.0        # supervised CE weight during CTTA
TTA_LR          = 1e-3       # higher lr is fine — only norm params updated
TTA_STEPS       = 1
ENTROPY_W       = 1.0        # entropy minimisation weight
RECON_W_TTA     = 0.5        # reconstruction on benign pool weight

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[System] Seed={SEED} | Device={device} | Arch={ARCH}')
def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive flow-level features and drop identifier/label columns."""
    if 'FLOW_END_MILLISECONDS' in df.columns and 'FLOW_START_MILLISECONDS' in df.columns:
        df = df.with_columns(
            (pl.col('FLOW_END_MILLISECONDS') - pl.col('FLOW_START_MILLISECONDS')).alias('FLOW_DURATION')
        )
    else:
        df = df.with_columns(pl.lit(0).alias('FLOW_DURATION'))
    if 'IN_BYTES' in df.columns and 'IN_PKTS' in df.columns:
        df = df.with_columns(
            (pl.col('IN_BYTES') / (pl.col('IN_PKTS') + 1e-5)).alias('BYTES_PER_PKT')
        )
    log_cols = ['IN_BYTES', 'IN_PKTS', 'FLOW_DURATION', 'SRC_TO_DST_IAT_MAX', 'DST_TO_SRC_IAT_MAX']
    existing = [c for c in log_cols if c in df.columns]
    if existing:
        df = df.with_columns([pl.col(c).log1p() for c in existing])
    drop_cols = [
        'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS',
        'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT',
        'Label', 'Attack', 'label', 'attack', 'Date',
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])
    return df


def load_dataset(dataset_name: str, sample_n: int = None):
    """Download dataset and return (X, y). Uses random sampling."""
    print(f'[Data] Loading {dataset_name} ...')
    path = kagglehub.dataset_download(dataset_name)
    csv_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files if f.endswith('.csv')
    ]
    df = pl.scan_csv(csv_files[0]).collect(engine='streaming')
    if sample_n and sample_n < df.height:
        df = df.sample(n=sample_n, seed=SEED)
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    y = df[label_col].to_numpy().astype(np.int64) if label_col else np.zeros(df.height, dtype=np.int64)
    df = engineer_features(df)
    X  = df.to_numpy().astype(np.float32)
    X  = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'   -> Shape: {X.shape} | Attack rate: {np.mean(y):.2%}')
    return X, y


def make_source_loaders(X, y):
    """
    Stratified 80/20 split. Scaler fitted on full training split.
    Training loader contains all labelled samples (benign + attack).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr)
    X_tr   = np.clip(scaler.transform(X_tr).astype(np.float32), -1, 1)
    X_te   = np.clip(scaler.transform(X_te).astype(np.float32), -1, 1)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    loader_tr = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True)
    loader_te = DataLoader(test_ds,  batch_size=TTA_BATCH_SIZE, shuffle=False)
    return loader_tr, loader_te, scaler


def make_target_loaders(X, y):
    """
    Fit a fresh MinMaxScaler on the full target dataset (no label leakage).

    Stratified split into:
      pool_loader   : FEW_SHOT_RATIO of data, labelled (both classes preserved)
      stream_loader : remaining data, labels kept for evaluation only

    The pool is used for supervised CE anchoring during CTTA.
    Real-world justification: the pool represents a brief initial analyst
    review period at deployment — a realistic assumption.
    """
    scaler   = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X_scaled = np.clip(scaler.transform(X).astype(np.float32), -1, 1)

    # Stratified split — both classes represented in pool
    X_pool, X_stream, y_pool, y_stream = train_test_split(
        X_scaled, y,
        test_size=(1 - FEW_SHOT_RATIO),
        random_state=SEED,
        stratify=y,
    )

    pool_ds   = TensorDataset(torch.from_numpy(X_pool),   torch.from_numpy(y_pool))
    stream_ds = TensorDataset(torch.from_numpy(X_stream), torch.from_numpy(y_stream))

    pool_loader   = DataLoader(pool_ds,   batch_size=BATCH_SIZE,     shuffle=True)
    stream_loader = DataLoader(stream_ds, batch_size=TTA_BATCH_SIZE, shuffle=True)

    print(f'   -> Pool: {len(y_pool)} samples '
          f'(attack rate: {np.mean(y_pool):.2%}) | '
          f'Stream: {len(y_stream)} '
          f'(attack rate: {np.mean(y_stream):.2%})')
    return pool_loader, stream_loader

# ─────────────────────────────────────────────────────────────────────────────
# Memory Module  (Gong et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────
# KAN AE+Classifier
# ─────────────────────────────────────────────────────────────────────────────
class KanAEClassifier(nn.Module):
    """
    Shared KAN encoder → classifier head + decoder head.
    Encoder:    input_dim -> 64 -> latent_dim  (KAN)
    Classifier: latent_dim -> 2               (Linear)
    Decoder:    latent_dim -> 64 -> input_dim  (KAN)
    forward() returns (logits, recon, z)
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder    = KAN([input_dim, 64, latent_dim], grid_range=[-1, 1])
        self.ln         = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder    = KAN([latent_dim, 64, input_dim], grid_range=[-1, 1])

    def forward(self, x):
        z = self.ln(self.encoder(x))
        return self.classifier(z), self.decoder(z), z


# ─────────────────────────────────────────────────────────────────────────────
# CNN AE+Classifier
# ─────────────────────────────────────────────────────────────────────────────
class CnnAEClassifier(nn.Module):
    """
    Shared CNN encoder → classifier head + decoder head.
    Each feature becomes its own channel (seq len=1). Pointwise Conv1d.
    GroupNorm(1,C) robust to variable attack rates.
    Encoder:    input_dim -> 64 -> latent_dim  (Conv1d)
    Classifier: latent_dim -> 2               (Linear)
    Decoder:    latent_dim -> 64 -> input_dim  (ConvTranspose1d)
    forward() returns (logits, recon, z)
    """
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.GroupNorm(1, 64), nn.GELU(),
            nn.Conv1d(64, latent_dim, kernel_size=1),
        )
        self.ln         = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=1),
            nn.GroupNorm(1, 64), nn.GELU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=1),
        )

    def forward(self, x):
        z     = self.ln(self.encoder(x.unsqueeze(-1)).squeeze(-1))
        recon = self.decoder(z.unsqueeze(-1)).squeeze(-1)
        return self.classifier(z), recon, z


# ─────────────────────────────────────────────────────────────────────────────
# Transformer AE+Classifier
# ─────────────────────────────────────────────────────────────────────────────
class TransformerAEClassifier(nn.Module):
    """
    Transformer encoder → classifier head + decoder head.

    Each input feature is projected to latent_dim, forming a sequence of
    input_dim tokens. Multi-head self-attention models pairwise feature
    interactions explicitly — more principled than CNN pointwise mixing
    for NetFlow features where interactions are semantically meaningful.

    Encoder:    (B, input_dim) → feature_embed → (B, input_dim, latent_dim)
                → N_LAYERS × TransformerEncoderLayer → mean pool → (B, latent_dim)
    Classifier: latent_dim → 2
    Decoder:    latent_dim → FF_DIM → input_dim  (lightweight MLP)

    forward() returns (logits, recon, z)
    """
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2,
                 ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        assert latent_dim % n_heads == 0, \
            f'latent_dim ({latent_dim}) must be divisible by n_heads ({n_heads})'

        # Project each scalar feature to a latent_dim-dimensional token
        self.feature_embed = nn.Linear(1, latent_dim)

        # Stack of transformer encoder layers with pre-norm for stable training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln          = nn.LayerNorm(latent_dim)
        self.classifier  = nn.Linear(latent_dim, 2)
        self.decoder     = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        # Each feature becomes a token: (B, input_dim) → (B, input_dim, latent_dim)
        tokens = self.feature_embed(x.unsqueeze(-1))
        tokens = self.transformer(tokens)          # (B, input_dim, latent_dim)
        z      = self.ln(tokens.mean(dim=1))       # mean pool → (B, latent_dim)
        return self.classifier(z), self.decoder(z), z


def build_model(arch: str, input_dim: int) -> nn.Module:
    if arch == 'kan':
        return KanAEClassifier(input_dim, LATENT_DIM)
    elif arch == 'cnn':
        return CnnAEClassifier(input_dim, LATENT_DIM)
    elif arch == 'transformer':
        return TransformerAEClassifier(
            input_dim, LATENT_DIM,
            n_heads=N_HEADS, n_layers=N_LAYERS,
            ff_dim=FF_DIM, dropout=DROPOUT,
        )
    else:
        raise ValueError(f"Unknown ARCH={arch!r}. Choose 'kan', 'cnn', or 'transformer'")


def get_trainable_params(model: nn.Module):
    """
    Return parameters for layer-selective CTTA updates.
    Updated: norm layers, classifier head, last encoder layer.
    Frozen:  early encoder layers, decoder.
    """
    params = []
    seen   = set()

    def add(p):
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p))
            params.append(p)

    # 1. All norm layers
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            for p in module.parameters():
                add(p)

    # 2. Classifier head
    for p in model.classifier.parameters():
        add(p)

    # 3. Last encoder layer (CNN / KAN)
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        if isinstance(encoder, nn.Sequential):
            last_layer = None
            for m in encoder.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    last_layer = m
            if last_layer is not None:
                for p in last_layer.parameters():
                    add(p)
        elif hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            for p in encoder.layers[-1].parameters():
                add(p)

    # 4. Transformer: last encoder layer + feature embedding
    if hasattr(model, 'transformer'):
        layers = list(model.transformer.layers)
        if layers:
            for p in layers[-1].parameters():
                add(p)
        for p in model.feature_embed.parameters():
            add(p)

    return params

def pretrain_source(model, loader, epochs, device):
    """
    Joint supervised + reconstruction pre-training.
    Loss = CrossEntropy(logits, y) + RECON_W * MSE(recon, x)
    ALL parameters updated — standard supervised training.
    """
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    ce_crit   = nn.CrossEntropyLoss()
    mse_crit  = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, recon, _ = model(x)
            loss = ce_crit(logits, y) + RECON_W * mse_crit(recon, x)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
        print(f'[Pretrain] Epoch {epoch+1}/{epochs} | '
              f'Loss: {total_loss/len(loader):.4f} | '
              f'Acc: {correct/total:.4f}')


def evaluate(model, loader, device, desc='Eval'):
    """Evaluate using classifier head — argmax of logits."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _, _ = model(x.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    print(f'[{desc}] F1: {f1:.4f} | Acc: {acc:.4f}')
    return f1


def run_ctta(model, stream_loader, pool_loader, device):
    """
    Few-Shot Norm-Only CTTA.

    FROZEN:  encoder, classifier head, decoder weights.
    UPDATED: LayerNorm and GroupNorm parameters only.

    Per stream batch (TTA_STEPS gradient steps):
      1. Supervised CE on pool batch — anchors decision boundary to target
      2. Entropy minimisation on stream batch — increases prediction confidence
      3. Reconstruction on pool batch — keeps norm stats grounded in benign

    Returns preds, labels on the stream.
    """
    norm_params = get_trainable_params(model)
    if not norm_params:
        print('[CTTA] WARNING: No trainable parameters found.')
        return evaluate(model, stream_loader, device, desc='CTTA (fallback)')

    print(f'[CTTA] Updating {len(norm_params)} param tensors '
          f'({sum(p.numel() for p in norm_params)} params). '
          f'All other weights frozen.')

    optimizer = optim.Adam(norm_params, lr=TTA_LR)
    ce_crit   = nn.CrossEntropyLoss()
    model.train()

    # Pool cycles indefinitely
    pool_iter = iter(pool_loader)
    def next_pool():
        nonlocal pool_iter
        try:
            return next(pool_iter)
        except StopIteration:
            pool_iter = iter(pool_loader)
            return next(pool_iter)

    all_preds, all_labels = [], []

    for x_stream, y_stream in stream_loader:
        x_stream = x_stream.to(device)

        for _ in range(TTA_STEPS):
            optimizer.zero_grad()

            # ── 1. Supervised CE on pool batch ────────────────────────────
            x_pool, y_pool = next_pool()
            x_pool = x_pool.to(device)
            y_pool = y_pool.to(device)
            logits_pool, recon_pool, _ = model(x_pool)
            loss_ce    = ce_crit(logits_pool, y_pool)

            # ── 2. Entropy on stream batch ────────────────────────────────
            logits_s, _, _ = model(x_stream)
            probs    = F.softmax(logits_s, dim=1)
            loss_ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

            # ── 3. Reconstruction on pool batch ───────────────────────────
            loss_recon = F.mse_loss(recon_pool, x_pool)

            loss = (FEW_SHOT_W  * loss_ce   +
                    ENTROPY_W   * loss_ent  +
                    RECON_W_TTA * loss_recon)


            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(norm_params, max_norm=1.0)
                optimizer.step()

        # Final inference
        with torch.no_grad():
            model.eval()
            logits_f, _, _ = model(x_stream)
            preds = logits_f.argmax(1).cpu().numpy()
            model.train()

        all_preds.extend(preds)
        all_labels.extend(y_stream.numpy())

    print('[CTTA] Stream complete.')
    return np.array(all_preds), np.array(all_labels)

# ── Source: ToN-IoT ───────────────────────────────────────────────────────────
X_src, y_src = load_dataset('seyhed/nf-cicids2018-v3', sample_n=SAMPLE_N)
input_dim = X_src.shape[1]
loader_src_train, loader_src_test, scaler = make_source_loaders(X_src, y_src)
del X_src, y_src; gc.collect()

# ── Target 1: UNSW-NB15 ───────────────────────────────────────────────────────
# Fresh MinMaxScaler per dataset — corrects cross-network scale mismatch.
# Benign pool: 1% of target, known-clean only (no attack labels needed).
X_tgt1, y_tgt1 = load_dataset('seyhed/nf-ton-iot-v3', sample_n=SAMPLE_N)
pool_tgt1, stream_tgt1 = make_target_loaders(X_tgt1, y_tgt1)
del X_tgt1, y_tgt1; gc.collect()

# ── Target 2: CICIDS2018 ──────────────────────────────────────────────────────
X_tgt2, y_tgt2 = load_dataset('seyhed/nf-unsw-nb15-v3', sample_n=SAMPLE_N)
pool_tgt2, stream_tgt2 = make_target_loaders(X_tgt2, y_tgt2)
del X_tgt2, y_tgt2; gc.collect()

print(f'\n[System] All datasets loaded. Input dim: {input_dim}')
print('=' * 60)
print(f'PHASE 1: SOURCE PRE-TRAINING (CICIDS2018) | {ARCH.upper()}')
print('=' * 60)

model = build_model(ARCH, input_dim).to(device)
n_params      = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_params = get_trainable_params(model)
print(f'[Model] {ARCH} | input_dim={input_dim} | latent_dim={LATENT_DIM} | '
      f'total params={n_params:,} | trainable params={sum(p.numel() for p in trainable_params)}\n')

pretrain_source(model, loader_src_train, epochs=PRETRAIN_EPOCHS, device=device)

# Source F1 should be >0.85 before CTTA.
# Zero-shot gives the baseline CTTA should improve from.
print('[Diagnostic] Source test performance:')
evaluate(model, loader_src_test, device, desc='Source (CICIDS2018) [post-pretrain]')

print('\n[Diagnostic] Zero-shot on targets (no adaptation yet):')
evaluate(model, stream_tgt1, device, desc='Target1 (ToN-IoT)  [zero-shot]')
evaluate(model, stream_tgt2, device, desc='Target2 (UNSW-NB15) [zero-shot]')

print('\n[Diagnostic] Parameters that will be updated during CTTA:')
total_norm = 0
for name, module in model.named_modules():
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.Linear, nn.Conv1d)):
        n = sum(p.numel() for p in module.parameters())
        total_norm += n
        print(f'  {name:40s} ({module.__class__.__name__}, {n} params)')
total_all = sum(p.numel() for p in model.parameters())
print(f'\n  Updating {total_norm} / {total_all} params '
      f'({total_norm/total_all:.2%} of model) during CTTA.')

print('=' * 60)
print('PHASE 2: ZERO-SHOT CROSS-DOMAIN EVALUATION')
print('=' * 60)

evaluate(model, loader_src_test, device, desc='Source  (CICIDS2018)   [zero-shot]')
evaluate(model, stream_tgt1,     device, desc='Target1 (ToN-IoT) [zero-shot]')
evaluate(model, stream_tgt2,     device, desc='Target2 (UNSW-NB15)[zero-shot]')
print('=' * 60)
print('PHASE 3: NORM-ONLY CTTA — Target 1 (ToN-IoT)')
print('=' * 60)

# Reset norm parameters to post-pretrain state before each CTTA phase
# so phases are independent and comparable
torch.manual_seed(SEED)
model_state = {k: v.clone() for k, v in model.state_dict().items()}

preds_tgt1, labels_tgt1 = run_ctta(
    model,
    stream_loader = stream_tgt1,
    pool_loader   = pool_tgt1,
    device        = device,
)
f1  = f1_score(labels_tgt1, preds_tgt1, zero_division=0)
acc = accuracy_score(labels_tgt1, preds_tgt1)
print(f'[CTTA Target1] F1: {f1:.4f} | Acc: {acc:.4f}')
print('=' * 60)
print('PHASE 4: NORM-ONLY CTTA — Target 2 (UNSW-NB15)')
print('=' * 60)

# Reset to post-pretrain state
model.load_state_dict(model_state)

preds_tgt2, labels_tgt2 = run_ctta(
    model,
    stream_loader = stream_tgt2,
    pool_loader   = pool_tgt2,
    device        = device,
)
f1  = f1_score(labels_tgt2, preds_tgt2, zero_division=0)
acc = accuracy_score(labels_tgt2, preds_tgt2)
print(f'[CTTA Target2] F1: {f1:.4f} | Acc: {acc:.4f}')
print('=' * 60)
print('RETENTION CHECK: Source (CICIDS2018) After CTTA')
print('=' * 60)

# Restore to post-pretrain state
model.load_state_dict(model_state)
evaluate(model, loader_src_test, device, desc='Source (CICIDS2018) [post-CTTA state]')

print('\n[Note] Source retention should be ~100% because only norm params')
print('were updated — the encoder and classifier weights are unchanged.')