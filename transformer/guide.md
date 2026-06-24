# TTC Transformer — Architecture & Parameter Guide

## What Is This System?

A **two-stage regression model** that predicts Time-to-Completion (TTC) in seconds from a sequence of robot camera frames.

- **Input:** raw RGB video frames from the robot's head camera
- **Output:** estimated seconds remaining until the task is complete
- **Model type:** regression (continuous output), not classification

Two stages:
1. **DINO-v2** — frozen feature extractor, converts each frame to a 768-d vector
2. **CausalTTCTransformer** — trained sequence model, reasons over time and predicts TTC

---

## Stage 1 — DINO-v2 (Frozen Feature Extractor)

### What it is
A **Vision Transformer (ViT-B/14)** trained by Meta on 142M images using self-supervised learning.
- "ViT-B" = base size (86M parameters)
- "14" = each image is divided into 14×14 pixel patches

### Why frozen
~44 episodes (~30K frames) is far too little data to fine-tune 86M parameters. Freezing it means the model borrows Meta's learned visual understanding without needing data to teach it. The sanity check confirmed this was justified — Ridge regression on frozen DINO features alone achieved R² of 0.6–0.82 on single frames.

### The [CLS] token
ViT processes an image as a sequence of patches plus one special "class" token. After all attention layers, the [CLS] token aggregates the whole image into a single vector. This is the only output we use. DINO's own classification head is ignored.

### Feature cache
DINO-v2 was run **once** over all frames and the outputs saved to `output/ttc_sanity/dino_features.npz`. Every training run reads from this file. DINO-v2 is not loaded during training.

| Parameter | Value | Meaning |
|---|---|---|
| `d_input` | 768 | [CLS] token dimension — fixed by DINO-v2, cannot change |

---

## Stage 2 — CausalTTCTransformer (Trained)

The only model whose weights are learned. Saved to `best_model.pt` alongside the config used to create it.

### Architecture (in order)

```
Input: (B, K, 768)   — batch of K-frame windows of DINO features

Linear(768 → 256)    — input projection: compress DINO dim to transformer dim
+ Positional embedding (K, 256)   — learned sense of frame order in window

4 × TransformerEncoderLayer
    MultiheadAttention (4 heads, causal mask)
    FeedForward (256 → 512 → 256)
    LayerNorm (pre-norm)
    Dropout (0.1)

LayerNorm

Linear(256 → 1)      — output head: predict log1p(TTC) for last frame

Output: (B,)  →  expm1()  →  predicted TTC in seconds
```

---

## Parameter Reference

### `window_size` (K = 32)
**What:** Number of consecutive frames the model sees at once.
At 50 fps, K=32 = 0.64 seconds of temporal context.

**Effect:** Larger K → more context, better boundary accuracy, longer warmup, more VRAM.
The first K−1 frames of every episode have no prediction (window not yet full).

| K | Context | Warmup |
|---|---|---|
| 32 | 0.64 s | 31 frames |
| 64 | 1.28 s | 63 frames |
| 128 | 2.56 s | 127 frames |

---

### `d_model` = 256
**What:** Internal dimension of the transformer. Every token is a 256-d vector inside the model.

**Effect:** Larger → more expressive, more parameters, higher overfitting risk on small datasets.
Chosen smaller than DINO's 768-d to keep the model proportionate to the dataset size (~300K total parameters vs 86M in DINO).

---

### `n_layers` = 4
**What:** Number of stacked transformer encoder layers.

**Effect:** More layers = deeper reasoning, more parameters. Each layer builds increasingly abstract temporal representations. 4 is conservative for 44 episodes.

---

### `n_heads` = 4
**What:** Number of parallel attention heads per layer.

**Effect:** Each head can independently attend to different aspects of the sequence (e.g. one tracks motion speed, another tracks object proximity). Must divide evenly into d_model: 256 ÷ 4 = 64 dimensions per head.

---

### `d_ff` = 512
**What:** Hidden dimension of the feedforward network inside each transformer layer. Expands d_model → d_ff → d_model.

**Effect:** More capacity for non-linear feature transformation per position. Typically 2× d_model (512 = 2×256).

---

### `dropout` = 0.1
**What:** Randomly zeros 10% of activations during training.

**Effect:** Regularisation — prevents the model from relying on any single neuron, reducing overfitting. Set to 0.0 automatically at inference and on the robot.

---

### Causal mask
**What:** A triangular mask that prevents each frame from attending to future frames. Frame i can only see frames 0 to i.

**Why it matters:** Makes the model safe for live robot inference. Without it the transformer would be bidirectional — it would need future frames that don't exist yet in real-time use. The causal constraint is architecturally enforced, not learned.

---

### Positional embedding (learned)
**What:** A lookup table of K × d_model values, one 256-d vector per position 0 to K−1. Added to input features before the transformer.

**Why it matters:** Transformers have no inherent sense of order — attention treats inputs as a set. Without positional embeddings the model cannot distinguish "most recent frame" from "31 frames ago." These are learned during training, not fixed.

---

### Pre-norm (`norm_first=True`)
**What:** LayerNorm applied before each attention/feedforward block rather than after.

**Why it matters:** More stable training on small datasets. Prevents gradient explosion during early epochs when weights are randomly initialised.

---

### `predict_all` (dense mode, default: false)
**What:** When false, the model predicts TTC only for the last frame in the window (scalar output). When true, it predicts TTC for all K frames simultaneously (K outputs).

**Effect:** Dense mode provides 32× more gradient signal per training batch. Forces the model to correctly predict high-TTC values (episode start) and near-zero values (episode end) across all window positions, not only when those frames happen to be last. Directly addresses the boundary divergence problem.

---

## Training Parameters

| Parameter | Value | Role |
|---|---|---|
| `lr` | 3e-4 | Learning rate — step size for each weight update |
| `warmup_epochs` | 10 | LR ramps from ~0 to `lr` linearly, then cosine-decays. Prevents early instability |
| `weight_decay` | 1e-4 | L2 regularisation — penalises large weights, reduces overfitting |
| `grad_clip` | 1.0 | Clips gradient norm to 1.0 — prevents catastrophic updates from gradient spikes |
| `batch_size` | 64 | Windows per gradient update. Each window is (K, 768) |
| `epochs` | 150 | Total training passes over the dataset |
| `seed` | 42 | Fixed random seed for reproducibility |

### Loss function: MSE on `log1p(TTC)`
- **MSE (Mean Squared Error):** squares the prediction error, penalising large mistakes more than small ones
- **log1p transform:** compresses the TTC distribution so the model pays equal attention across the full range (0–15s), not just the majority of frames near TTC=0. Undone with `expm1()` to get seconds.
- **R² in log space:** measures how well the model explains variance in the transformed values. R²=1 is perfect, R²=0 = no better than predicting the mean, negative = worse than mean.
- **MAE in seconds:** mean absolute error in raw seconds — the practically interpretable metric.

---

## Online Temporal Filter (live-robot compatible)

After the model outputs a prediction, an exponential moving average (EMA) filter smooths frame-to-frame jumps:

```
smoothed[t] = α × pred[t] + (1 − α) × smoothed[t−1]
```

- **α = 1.0** → raw predictions, no smoothing
- **α = 0.3** → each output is 30% current prediction + 70% previous smoothed value (default)
- **Causal:** only uses past values — safe for live inference
- **Zero latency:** one multiply per frame

---

## Experiment Configs

| Config | K | Dense | Output dir |
|---|---|---|---|
| `configs/config.yaml` | 32 | No | `output/transformer_k32_sparse/` |
| `configs/config_k64.yaml` | 64 | No | `output/transformer_k64_sparse/` |
| `configs/config_k128.yaml` | 128 | No | `output/transformer_k128_sparse/` |
| `configs/config_dense.yaml` | 32 | Yes | `output/transformer_k32_dense/` |

### Training commands (from repo root, hf-env active)
```bash
python -m transformer.train --config transformer/configs/config.yaml
python -m transformer.train --config transformer/configs/config_dense.yaml
python -m transformer.train --config transformer/configs/config_k64.yaml
python -m transformer.train --config transformer/configs/config_k128.yaml
```

---

## Key Files

| File | Role |
|---|---|
| `transformer/model.py` | CausalTTCTransformer definition |
| `transformer/dataset.py` | TTCWindowDataset — sliding window over cached features |
| `transformer/train.py` | Training loop, evaluation, plots |
| `transformer/config*.yaml` | Experiment configurations |
| `transformer/visualize.ipynb` | Frame gallery, TTC curves, interactive scrubber |
| `transformer/compare.ipynb` | Side-by-side experiment comparison |
| `transformer/guide.md` | This file |
| `ttc_sanity.ipynb` | Sanity check — frozen DINO + Ridge regression baseline |
| `output/ttc_sanity/dino_features.npz` | Cached DINO features (N, 768) for all episodes |
| `output/transformer_*/best_model.pt` | Saved checkpoint: weights + config + val R² |
| `output/transformer_*/history.npz` | Training curves data for compare notebook |
