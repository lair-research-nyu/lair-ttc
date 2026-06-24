#!/usr/bin/env python
"""
train.py — Causal transformer for TTC prediction.

Usage (from repo root, with hf-env active):
    python -m transformer.train
    python -m transformer.train --config transformer/config.yaml
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from .dataset import make_splits
from .model import CausalTTCTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_lr(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Return (loss, R², MAE_s, y_true_s, y_pred_s)."""
    model.eval()
    losses, y_log_true, y_log_pred = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            losses.append(criterion(pred, y).item())
            # Dense mode: pred/y are (B, K) — slice last position for metrics
            p = pred[:, -1] if pred.dim() > 1 else pred
            t = y[:,   -1] if y.dim()   > 1 else y
            y_log_true.append(t.cpu().numpy())
            y_log_pred.append(p.cpu().numpy())

    y_log_true = np.concatenate(y_log_true)
    y_log_pred = np.concatenate(y_log_pred)
    y_true_s   = np.expm1(y_log_true).clip(0)
    y_pred_s   = np.expm1(y_log_pred).clip(0)

    loss = float(np.mean(losses))
    r2   = float(r2_score(y_log_true, y_log_pred))
    mae  = float(mean_absolute_error(y_true_s, y_pred_s))
    return loss, r2, mae, y_true_s, y_pred_s


# ──────────────────────────────────────────────────────────────────────────────
# Auto visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ema_filter(pred_s: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out, last = pred_s.copy(), None
    for i in range(len(out)):
        if np.isnan(out[i]):
            continue
        out[i] = alpha * out[i] + (1 - alpha) * last if last is not None else out[i]
        last = out[i]
    return out


def _infer_episode(
    model: nn.Module,
    ep_feats: np.ndarray,
    device: torch.device,
    K: int,
    batch: int = 256,
) -> np.ndarray:
    """Rolling-window inference for one episode. Returns pred_log (N,) with NaN warmup."""
    n = len(ep_feats)
    pred_log = np.full(n, np.nan, dtype=np.float32)
    if n < K:
        return pred_log
    windows = np.stack([ep_feats[i - K + 1 : i + 1] for i in range(K - 1, n)], axis=0)
    preds = []
    for i in range(0, len(windows), batch):
        x = torch.from_numpy(windows[i : i + batch]).to(device)
        out = model(x)
        if out.dim() > 1:
            out = out[:, -1]
        preds.append(out.detach().cpu().numpy())
    pred_log[K - 1:] = np.concatenate(preds)
    return pred_log


def plot_episode_curves_auto(
    model: nn.Module,
    test_ds,
    test_ep_keys: list[str],
    device: torch.device,
    K: int,
    out_dir: Path,
    smooth_alpha: float = 0.3,
) -> Path:
    """Generate per-episode TTC countdown curves and save to out_dir."""
    model.eval()
    n_eps  = len(test_ds.ep_data)
    n_cols = min(3, n_eps)
    n_rows = (n_eps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5), squeeze=False)

    for idx, ((ep_feats, ep_ttc), ep_key) in enumerate(zip(test_ds.ep_data, test_ep_keys)):
        ax = axes[idx // n_cols][idx % n_cols]
        n  = len(ep_feats)
        t  = np.arange(n) / 30.0
        pred_log = _infer_episode(model, ep_feats, device, K)
        pred_s   = np.expm1(pred_log).clip(0)
        smooth_s = _ema_filter(pred_s, smooth_alpha)

        ax.plot(t, ep_ttc,  color="steelblue", lw=1.4, label="actual")
        ax.plot(t, pred_s,  color="tomato",    lw=0.7, ls="--", alpha=0.35)
        ax.plot(t, smooth_s, color="tomato",   lw=1.4, label=f"pred (α={smooth_alpha})")
        ax.axvspan(0, (K - 1) / 30.0, color="gray", alpha=0.12)

        valid = ~np.isnan(pred_s)
        if valid.any():
            r2  = r2_score(np.log1p(ep_ttc[valid]), np.log1p(pred_s[valid]))
            mae = mean_absolute_error(ep_ttc[valid], pred_s[valid])
            ax.set_title(f"{ep_key}\nR²={r2:.3f}  MAE={mae:.2f}s", fontsize=8)
        else:
            ax.set_title(ep_key, fontsize=8)

        ax.set_xlabel("Time in episode (s)", fontsize=8)
        ax.set_ylabel("TTC (s)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_eps, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.suptitle(f"TTC Countdown Curves — Test Episodes (K={K})", fontsize=11, y=1.01)
    plt.tight_layout()
    path = out_dir / "episode_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frame_gallery_auto(
    model: nn.Module,
    test_ds,
    test_ep_keys: list[str],
    hdf5_path: Path,
    device: torch.device,
    K: int,
    out_dir: Path,
    camera: str = "head_left",
    n_frames: int = 6,
) -> Path:
    """Generate a frame gallery with GT/predicted TTC overlaid and save to out_dir."""
    model.eval()
    n_rows = len(test_ds.ep_data)
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.8, n_rows * 2.4), squeeze=False)

    with h5py.File(hdf5_path, "r") as f:
        for row, ((ep_feats, ep_ttc), ep_key) in enumerate(zip(test_ds.ep_data, test_ep_keys)):
            n        = len(ep_feats)
            pred_log = _infer_episode(model, ep_feats, device, K)
            pred_s   = np.expm1(pred_log).clip(0)

            if ep_key not in f or camera not in f[ep_key].get("cameras", {}):
                for ax in axes[row]:
                    ax.axis("off"); ax.set_title(f"{ep_key}\ncamera not found", fontsize=7)
                continue

            frames  = f[ep_key]["cameras"][camera][:]
            indices = np.linspace(K - 1, n - 1, n_frames, dtype=int)

            for col, fi in enumerate(indices):
                ax   = axes[row][col]
                gt   = float(ep_ttc[fi])
                pred = float(pred_s[fi]) if not np.isnan(pred_s[fi]) else None
                err  = (pred - gt) if pred is not None else None

                def _ec(e):
                    if e is None: return "gray"
                    return "#2ecc71" if abs(e) < 2 else "#e67e22" if abs(e) < 5 else "#e74c3c"

                ax.imshow(frames[fi]); ax.axis("off")
                txt = f"GT:   {gt:.1f}s"
                if pred is not None:
                    txt += f"\nPred: {pred:.1f}s\nErr:  {err:+.1f}s"
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
                        color="white", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_color(_ec(err)); spine.set_linewidth(3)
                if col == 0:
                    ax.set_ylabel(ep_key, fontsize=7, rotation=0, labelpad=80, va="center")
                ax.set_title(f"{100*fi/max(n-1,1):.0f}%", fontsize=7, pad=2)

    fig.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="|err| < 2s"),
        mpatches.Patch(color="#e67e22", label="2–5s"),
        mpatches.Patch(color="#e74c3c", label="≥ 5s"),
    ], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle(f"Frame Gallery — K={K}  camera={camera}", fontsize=11, y=1.01)
    plt.tight_layout()
    path = out_dir / "frame_gallery.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _test_ep_keys_from_cfg(cfg: dict, feat_cache: Path, hdf5_path: Path) -> list[str]:
    """Recompute test episode keys from config (mirrors make_splits logic)."""
    data = np.load(feat_cache)
    n_ep = int(data["episode_ids"].max()) + 1
    with h5py.File(hdf5_path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
    excluded: set[int] = set()
    for name in cfg.get("excluded_episodes", []):
        if name in ep_keys:
            excluded.add(ep_keys.index(name))
    n_test   = max(1, round(n_ep * cfg["test_fraction"]))
    test_eps = sorted(set(range(n_ep - n_test, n_ep)) - excluded)
    return [ep_keys[i] for i in test_eps]


# ──────────────────────────────────────────────────────────────────────────────
# No-context evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _infer_episode_nocontext(
    model: nn.Module,
    ep_feats: np.ndarray,
    device: torch.device,
    K: int,
    batch: int = 256,
) -> np.ndarray:
    """Zero-padded single-frame inference for one episode. Returns pred_log (N,) — no warmup gap."""
    n = len(ep_feats)
    d = ep_feats.shape[-1]
    padded = np.zeros((n, K, d), dtype=np.float32)
    padded[:, -1, :] = ep_feats
    preds = []
    for i in range(0, n, batch):
        x = torch.from_numpy(padded[i : i + batch]).to(device)
        out = model(x)
        if out.dim() > 1:
            out = out[:, -1]
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds)


def plot_episode_curves_nocontext_auto(
    model: nn.Module,
    test_ds,
    test_ep_keys: list[str],
    device: torch.device,
    K: int,
    out_dir: Path,
    smooth_alpha: float = 0.3,
) -> Path:
    """Episode curves using zero-padded single-frame inference."""
    model.eval()
    n_eps  = len(test_ds.ep_data)
    n_cols = min(3, n_eps)
    n_rows = (n_eps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5), squeeze=False)

    for idx, ((ep_feats, ep_ttc), ep_key) in enumerate(zip(test_ds.ep_data, test_ep_keys)):
        ax = axes[idx // n_cols][idx % n_cols]
        n  = len(ep_feats)
        t  = np.arange(n) / 30.0
        pred_log = _infer_episode_nocontext(model, ep_feats, device, K)
        pred_s   = np.expm1(pred_log).clip(0)
        smooth_s = _ema_filter(pred_s, smooth_alpha)

        ax.plot(t, ep_ttc,   color="steelblue", lw=1.4, label="actual")
        ax.plot(t, pred_s,   color="tomato",    lw=0.7, ls="--", alpha=0.35)
        ax.plot(t, smooth_s, color="tomato",    lw=1.4, label=f"pred (α={smooth_alpha})")

        r2  = r2_score(np.log1p(ep_ttc), pred_log)
        mae = mean_absolute_error(ep_ttc, pred_s)
        ax.set_title(f"{ep_key}\nR²={r2:.3f}  MAE={mae:.2f}s", fontsize=8)
        ax.set_xlabel("Time in episode (s)", fontsize=8)
        ax.set_ylabel("TTC (s)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_eps, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.suptitle(f"TTC Countdown Curves — No-Context / Single-Frame (K={K})", fontsize=11, y=1.01)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "episode_curves_nocontext.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frame_gallery_nocontext_auto(
    model: nn.Module,
    test_ds,
    test_ep_keys: list[str],
    hdf5_path: Path,
    device: torch.device,
    K: int,
    out_dir: Path,
    camera: str = "head_left",
    n_frames: int = 6,
) -> Path:
    """Frame gallery using zero-padded single-frame inference."""
    model.eval()
    n_rows = len(test_ds.ep_data)
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.8, n_rows * 2.4), squeeze=False)

    with h5py.File(hdf5_path, "r") as f:
        for row, ((ep_feats, ep_ttc), ep_key) in enumerate(zip(test_ds.ep_data, test_ep_keys)):
            n        = len(ep_feats)
            pred_log = _infer_episode_nocontext(model, ep_feats, device, K)
            pred_s   = np.expm1(pred_log).clip(0)

            if ep_key not in f or camera not in f[ep_key].get("cameras", {}):
                for ax in axes[row]:
                    ax.axis("off"); ax.set_title(f"{ep_key}\ncamera not found", fontsize=7)
                continue

            frames  = f[ep_key]["cameras"][camera][:]
            indices = np.linspace(0, n - 1, n_frames, dtype=int)

            for col, fi in enumerate(indices):
                ax   = axes[row][col]
                gt   = float(ep_ttc[fi])
                pred = float(pred_s[fi])
                err  = pred - gt

                def _ec(e):
                    return "#2ecc71" if abs(e) < 2 else "#e67e22" if abs(e) < 5 else "#e74c3c"

                ax.imshow(frames[fi]); ax.axis("off")
                txt = f"GT:   {gt:.1f}s\nPred: {pred:.1f}s\nErr:  {err:+.1f}s"
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.5,
                        color="white", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0))
                for spine in ax.spines.values():
                    spine.set_visible(True); spine.set_color(_ec(err)); spine.set_linewidth(3)
                if col == 0:
                    ax.set_ylabel(ep_key, fontsize=7, rotation=0, labelpad=80, va="center")
                ax.set_title(f"{100*fi/max(n-1,1):.0f}%", fontsize=7, pad=2)

    fig.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="|err| < 2s"),
        mpatches.Patch(color="#e67e22", label="2–5s"),
        mpatches.Patch(color="#e74c3c", label="≥ 5s"),
    ], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle(f"Frame Gallery — No-Context / Single-Frame  K={K}  camera={camera}", fontsize=11, y=1.01)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frame_gallery_nocontext.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_nocontext(
    model: nn.Module,
    test_ds,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate using zero-padded single-frame windows — no temporal context.

    For every frame in the test set, builds a (K, 768) window where positions
    0..K-2 are zeros and only the last position holds the real DINO feature.
    Every frame gets a prediction (no warmup gap).

    Returns (R², MAE_s, y_true_s, y_pred_s).
    """
    model.eval()
    y_log_true, y_log_pred = [], []
    K = test_ds.K
    d = test_ds.ep_data[0][0].shape[-1]
    BATCH = 256

    with torch.no_grad():
        for ep_feats, ep_ttc in test_ds.ep_data:
            n = len(ep_feats)
            padded = np.zeros((n, K, d), dtype=np.float32)
            padded[:, -1, :] = ep_feats   # only last slot = current frame

            preds = []
            for i in range(0, n, BATCH):
                x = torch.from_numpy(padded[i : i + BATCH]).to(device)
                out = model(x)
                if out.dim() > 1:
                    out = out[:, -1]
                preds.append(out.detach().cpu().numpy())

            y_log_true.append(np.log1p(ep_ttc))
            y_log_pred.append(np.concatenate(preds))

    y_log_true = np.concatenate(y_log_true)
    y_log_pred = np.concatenate(y_log_pred)
    y_true_s   = np.expm1(y_log_true).clip(0)
    y_pred_s   = np.expm1(y_log_pred).clip(0)

    r2  = float(r2_score(y_log_true, y_log_pred))
    mae = float(mean_absolute_error(y_true_s, y_pred_s))
    return r2, mae, y_true_s, y_pred_s


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log space)")
    ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["val_r2"])
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("R² (log space)")
    ax.set_title("Validation R²"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history["epoch"], history["val_mae"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (s)")
    ax.set_title("Validation MAE"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    tag: str = "test",
) -> Path:
    r2  = r2_score(np.log1p(y_true), np.log1p(y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, s=2, alpha=0.3, rasterized=True)
    lim = max(float(y_true.max()), float(y_pred.max())) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="perfect")
    ax.set_xlabel("Actual TTC (s)"); ax.set_ylabel("Predicted TTC (s)")
    ax.set_title(f"Predicted vs Actual ({tag})\nR²={r2:.3f}  MAE={mae:.2f} s")
    ax.legend(fontsize=8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=60, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Error (s)"); ax.set_ylabel("Count")
    ax.set_title(
        f"Error distribution  ({tag})\n"
        f"mean={errors.mean():.2f} s   std={errors.std():.2f} s"
    )

    plt.suptitle("Causal TTC Transformer", fontsize=12, y=1.02)
    plt.tight_layout()
    path = out_dir / f"eval_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> tuple[float, float]:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe for HPC/headless runs

    set_seed(cfg["seed"])

    out_dir  = Path(cfg["output_dir"])
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}")

    # ── Optional W&B ──────────────────────────────────────────────────────────
    wandb = None
    if cfg.get("use_wandb"):
        try:
            import wandb as _wandb
            _wandb.init(
                project=cfg["wandb_project"],
                entity=cfg.get("wandb_entity") or None,
                config=cfg,
                name=f"K{cfg['window_size']}_L{cfg['n_layers']}_d{cfg['d_model']}",
            )
            wandb = _wandb
            print("W&B    : enabled")
        except ImportError:
            print("W&B    : wandb not installed — skipping")

    # ── Data ──────────────────────────────────────────────────────────────────
    feat_cache = Path(cfg["feat_cache"])
    hdf5_path  = Path(cfg["hdf5_path"])

    train_ds, val_ds, test_ds = make_splits(feat_cache, hdf5_path, cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"] * 2,
        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CausalTTCTransformer(
        d_input     = cfg["d_input"],
        d_model     = cfg["d_model"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        K           = cfg["window_size"],
        predict_all = cfg.get("predict_all", False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {n_params:,} parameters")

    if wandb:
        wandb.watch(model, log="gradients", log_freq=50)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r2 = -float("inf")
    best_ckpt   = out_dir / "best_model.pt"
    log_every   = cfg.get("log_every", 5)
    history     = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_r2", "val_mae")}

    t0 = time.time()
    print(f"\nTraining for {cfg['epochs']} epochs …\n")

    for epoch in range(cfg["epochs"]):
        # LR schedule: linear warmup → cosine decay
        scale = warmup_cosine_lr(epoch, cfg["warmup_epochs"], cfg["epochs"])
        for g in optimizer.param_groups:
            g["lr"] = cfg["lr"] * scale

        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, val_r2, val_mae, _, _ = evaluate(model, val_loader, device, criterion)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["val_mae"].append(val_mae)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            if cfg.get("save_best"):
                torch.save(
                    {"epoch": epoch + 1, "model": model.state_dict(),
                     "val_r2": val_r2, "cfg": cfg},
                    best_ckpt,
                )

        if wandb:
            wandb.log(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "val_r2": val_r2, "val_mae": val_mae,
                 "lr": get_lr(optimizer)},
                step=epoch + 1,
            )

        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"  ep {epoch+1:3d}/{cfg['epochs']}"
                f"  train={train_loss:.4f}"
                f"  val={val_loss:.4f}"
                f"  R²={val_r2:.3f}"
                f"  MAE={val_mae:.2f}s"
                f"  lr={get_lr(optimizer):.1e}"
                f"  ({elapsed:.0f}s elapsed)"
            )

    print(f"\nBest val R² : {best_val_r2:.3f}")

    # ── Save training curves + history ───────────────────────────────────────
    curve_path = plot_training_curves(history, plot_dir)
    print(f"Saved  : {curve_path}")
    np.savez(out_dir / "history.npz", **{k: np.array(v) for k, v in history.items()})
    print(f"Saved  : {out_dir / 'history.npz'}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint (epoch {ckpt['epoch']},  val R²={ckpt['val_r2']:.3f})")

    test_loss, test_r2, test_mae, y_true, y_pred = evaluate(
        model, test_loader, device, criterion,
    )
    baseline_mae = float(np.abs(y_true - y_true.mean()).mean())
    improvement  = (1 - test_mae / max(baseline_mae, 1e-6)) * 100

    print(f"\n── Test results {'─'*40}")
    print(f"  R²  (log space) : {test_r2:.3f}")
    print(f"  MAE             : {test_mae:.2f} s")
    print(f"  Baseline MAE    : {baseline_mae:.2f} s  (predict-mean)")
    print(f"  Improvement     : {improvement:.1f}% over baseline")

    eval_path = plot_eval(y_true, y_pred, plot_dir, tag="test")
    print(f"  Saved           : {eval_path}")

    # ── Episode curves + frame gallery (auto-generated) ──────────────────────
    test_ep_keys = _test_ep_keys_from_cfg(cfg, feat_cache, hdf5_path)

    # ── Optional no-context (per-frame) evaluation ───────────────────────────
    if cfg.get("eval_nocontext"):
        nc_r2, nc_mae, nc_true_s, nc_pred_s = evaluate_nocontext(model, test_ds, device)
        print(f"\n── No-context (per-frame) results {'─'*34}")
        print(f"  R²  (log space) : {nc_r2:.3f}")
        print(f"  MAE             : {nc_mae:.2f} s")
        if best_ckpt.exists():
            saved = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            saved["test_r2_nc"]  = nc_r2
            saved["test_mae_nc"] = nc_mae
            torch.save(saved, best_ckpt)
            print(f"  Saved to        : {best_ckpt}")

        nc_eval_path = plot_eval(nc_true_s, nc_pred_s, plot_dir, tag="test_nocontext")
        print(f"  Saved           : {nc_eval_path}")

        nc_curves_path = plot_episode_curves_nocontext_auto(
            model, test_ds, test_ep_keys, device, cfg["window_size"], plot_dir
        )
        print(f"  Saved           : {nc_curves_path}")

        nc_gallery_path = plot_frame_gallery_nocontext_auto(
            model, test_ds, test_ep_keys, hdf5_path, device,
            cfg["window_size"], plot_dir, camera=cfg.get("viz_camera", "head_left"),
        )
        print(f"  Saved           : {nc_gallery_path}")
    curves_path  = plot_episode_curves_auto(model, test_ds, test_ep_keys, device, cfg["window_size"], plot_dir)
    print(f"Saved  : {curves_path}")
    gallery_path_auto = plot_frame_gallery_auto(
        model, test_ds, test_ep_keys, hdf5_path, device,
        cfg["window_size"], plot_dir, camera=cfg.get("viz_camera", "head_left"),
    )
    print(f"Saved  : {gallery_path_auto}")

    if wandb:
        wandb.log({"test_r2": test_r2, "test_mae": test_mae, "test_loss": test_loss})
        wandb.log({"eval/scatter": wandb.Image(str(eval_path)),
                   "eval/curves":  wandb.Image(str(curve_path))})
        wandb.finish()

    return test_r2, test_mae


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train causal TTC transformer")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent / "configs" / "config.yaml",
        help="YAML config file (default: transformer/configs/config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Config : {args.config.resolve()}")
    train(cfg)


if __name__ == "__main__":
    main()
