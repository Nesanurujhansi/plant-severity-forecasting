"""Comprehensive IEEE Research Paper Evaluation Suite.

Evaluates the complete Plant Disease Severity & 7-Day Forecasting pipeline on:
  1. Real-world Coffee Leaf Rust benchmark (N=100 images with Golden Standard
     binary lesion masks and expert severity ratings, plus published baselines:
     ImageJ, Pliman, SAM3, DeepLabV3) using strict 5-fold out-of-sample
     cross-validation with frozen DINOv2 + SeveritySegmentationHeads +
     Mondrian Conformal Calibration.
  2. Real-time 7-Day Weather-Driven Forecaster across cached 9 km NWP weather
     profiles (Open-Meteo ECMWF/GFS).
  3. Safety & OOD Stress Test results.

Computes and exports:
  - Segmentation: Dice, IoU, Pixel Accuracy, Precision, Recall, Specificity, Pixel ROC-AUC
  - Regression: MAE, RMSE, R^2, Pearson r, Spearman rho, Lin's CCC, Bland-Altman
  - Conformal Uncertainty: 90% Empirical Coverage, Mean Interval Width
  - Classification & ROC: Ordinal Accuracy, QWK, Macro-F1, Binary ROC-AUC & PR-AUC
  - Publication Figures (PNG) & Structured Tables (JSON / CSV)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
)

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.extract_features import FeatureExtractorEngine, preprocess_image
from ml.severity.heads import build_severity_heads
from ml.severity.losses import DiceLoss, FocalTverskyLoss
from ml.severity.predict import extract_vegetative_pathology_masks
from ml.severity.conformal import MondrianConformalCalibrator
from ml.severity.evaluate import (
    compute_lins_ccc,
    compute_bland_altman_stats,
    compute_quadratic_weighted_kappa,
)
from ml.severity.severity import map_severity_to_ordinal_class, DEFAULT_BIN_LABELS
from ml.forecast.predict import Forecaster


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Compute binary segmentation metrics for a single image mask pair."""
    p = (pred_mask > 0).astype(np.uint8).ravel()
    g = (gt_mask > 0).astype(np.uint8).ravel()

    tp = int(np.sum((p == 1) & (g == 1)))
    tn = int(np.sum((p == 0) & (g == 0)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_regression_series(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute full regression & agreement metrics in percentage points (0-100)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2 = float(r2_score(yt, yp))
    p_r, _ = pearsonr(yt, yp)
    s_rho, _ = spearmanr(yt, yp)
    ccc = compute_lins_ccc(yt / 100.0, yp / 100.0)
    ba = compute_bland_altman_stats(yt / 100.0, yp / 100.0)

    return {
        "mae_pct": round(mae, 4),
        "rmse_pct": round(rmse, 4),
        "r2": round(r2, 4),
        "pearson_r": round(float(p_r), 4),
        "spearman_rho": round(float(s_rho), 4),
        "lins_ccc": round(float(ccc), 4),
        "bland_altman_bias_pct": round(float(ba["mean_bias_pct"]), 4),
        "bland_altman_loa_lower_pct": round(float(ba["lower_loa_pct"]), 4),
        "bland_altman_loa_upper_pct": round(float(ba["upper_loa_pct"]), 4),
    }


def evaluate_binary_roc(y_true_pct: np.ndarray, y_pred_pct: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Evaluate binary severity threshold detection, ROC curve, and PR-AUC."""
    y_bin = (y_true_pct >= threshold).astype(int)
    pred_bin = (y_pred_pct >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_bin, y_pred_pct)
    roc_auc = float(auc(fpr, tpr))
    pr_auc = float(average_precision_score(y_bin, y_pred_pct))

    return {
        "threshold_pct": threshold,
        "positive_count": int(np.sum(y_bin)),
        "negative_count": int(len(y_bin) - np.sum(y_bin)),
        "accuracy": round(float(accuracy_score(y_bin, pred_bin)), 4),
        "precision": round(float(precision_score(y_bin, pred_bin, zero_division=0)), 4),
        "recall": round(float(recall_score(y_bin, pred_bin, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_bin, pred_bin, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fpr": [round(float(x), 4) for x in fpr],
        "tpr": [round(float(x), 4) for x in tpr],
    }


def extract_or_load_dinov2_cache(df: pd.DataFrame, img_dir: Path, mask_dir: Path) -> Dict[str, Any]:
    cache_path = PROJECT_ROOT / "data" / "processed" / "coffee_rust_dinov2_cache.pt"
    if cache_path.exists():
        print(f"Loading cached DINOv2 features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print("Extracting frozen DINOv2-small features for 100 Coffee Leaf Rust images...")
    extractor = FeatureExtractorEngine(
        backbone_id="facebook/dinov2-small",
        revision="main",
        patch_size=14,
        is_stub=False,
        device="cpu",
    )

    samples = []
    for _, row in df.iterrows():
        fname = row["image"]
        pil_img = Image.open(img_dir / fname).convert("RGB")
        img_sq = pil_img.resize((448, 448), Image.BILINEAR)
        tensor_in, _ = preprocess_image(img_sq, target_resolution=448, patch_size=14)
        with torch.no_grad():
            _, _, grid_t = extractor.extract_features(tensor_in, use_amp=False)

        gt_mask = np.array(Image.open(mask_dir / fname).convert("L")) > 127
        gt_128 = np.array(Image.fromarray(gt_mask.astype(np.uint8) * 255).resize((128, 128), Image.NEAREST)) > 127

        leaf_full, _, _, _ = extract_vegetative_pathology_masks(pil_img)
        leaf_128 = np.array(Image.fromarray((leaf_full > 0).astype(np.uint8) * 255).resize((128, 128), Image.NEAREST)) > 127

        samples.append({
            "image": fname,
            "gs_pct": float(row["GS"]),
            "grid_tokens": grid_t.squeeze(0).cpu().to(torch.float32),
            "gt_lesion_128": torch.from_numpy(gt_128.astype(np.float32)),
            "gt_leaf_128": torch.from_numpy(leaf_128.astype(np.float32)),
            "gt_mask_full": gt_mask,
            "leaf_mask_full": (leaf_full > 0),
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples}, cache_path)
    return {"samples": samples}


class DINOv2RGBLabFusionHead(nn.Module):
    """Fuses frozen DINOv2 semantic decoder logits with 6-channel RGB+CIELAB boundary cues."""

    def __init__(self, embed_dim: int = 384, reduced_dim: int = 48):
        super().__init__()
        self.dino_head = build_severity_heads({"input_dim": embed_dim, "reduced_dim": reduced_dim})
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(1 + 16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, grid_tokens: torch.Tensor, rgblab_128: torch.Tensor, target_size=None) -> torch.Tensor:
        out = self.dino_head(grid_tokens, target_size=(128, 128))
        dino_logit = out["lesion_logits"]
        rgb_feat = self.rgb_stem(rgblab_128)
        fused_logit = dino_logit + self.fusion(torch.cat([dino_logit, rgb_feat], dim=1))
        if target_size is not None and target_size != (128, 128):
            fused_logit = F.interpolate(fused_logit, size=target_size, mode="bilinear", align_corners=False)
        return fused_logit


def extract_stage2_features(prob_128: np.ndarray, leaf_128: np.ndarray, lab_128: np.ndarray, tau: float) -> np.ndarray:
    """Extract Stage-2 multi-modal lesion & CIELAB chrominance summary descriptors."""
    lm = leaf_128 > 0.5
    n_leaf = max(1, int(lm.sum()))
    p_in = prob_128[lm]
    a_in = lab_128[:, :, 1][lm]

    sev_hard = float((p_in > tau).sum() / n_leaf) * 100.0
    sev_soft = float(p_in.mean()) * 100.0
    sev_sq = float((p_in ** 2).mean()) * 100.0
    p75 = float(np.percentile(p_in, 75)) * 100.0
    p90 = float(np.percentile(p_in, 90)) * 100.0
    a_high = float(np.mean(a_in > 0.52)) * 100.0
    return np.array([sev_hard, sev_soft, sev_sq, p75, p90, a_high], dtype=np.float64)


def train_and_eval_fold(
    train_samples: List[Dict],
    cal_samples: List[Dict],
    test_samples: List[Dict],
    epochs: int = 25,
    lr: float = 4e-3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    torch.manual_seed(seed)
    model = DINOv2RGBLabFusionHead(embed_dim=384, reduced_dim=48)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dice_loss_fn = DiceLoss()

    x_train = torch.stack([s["grid_tokens"] for s in train_samples])
    rl_train = torch.stack([s["rgblab_128"] for s in train_samples])
    y_les_train = torch.stack([s["gt_lesion_128"] for s in train_samples])
    y_leaf_train = torch.stack([s["gt_leaf_128"] for s in train_samples])

    n_tr = len(train_samples)
    batch_size = 32

    for _ in range(epochs):
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx]
            rb = rl_train[idx]
            yb_les = y_les_train[idx]
            yb_leaf = y_leaf_train[idx]

            if torch.rand(1).item() > 0.5:
                xb = torch.flip(xb, dims=[2])
                rb = torch.flip(rb, dims=[3])
                yb_les = torch.flip(yb_les, dims=[2])
                yb_leaf = torch.flip(yb_leaf, dims=[2])
            if torch.rand(1).item() > 0.5:
                xb = torch.flip(xb, dims=[1])
                rb = torch.flip(rb, dims=[2])
                yb_les = torch.flip(yb_les, dims=[1])
                yb_leaf = torch.flip(yb_leaf, dims=[1])

            les_logits = model(xb, rb).squeeze(1)

            prob = torch.sigmoid(les_logits) * yb_leaf
            pred_sev = prob.sum(dim=(1, 2)) / torch.clamp(yb_leaf.sum(dim=(1, 2)), min=1.0)
            gt_sev = (yb_les * yb_leaf).sum(dim=(1, 2)) / torch.clamp(yb_leaf.sum(dim=(1, 2)), min=1.0)
            loss_sev = F.smooth_l1_loss(pred_sev, gt_sev, beta=0.01)

            loss = 0.5 * F.binary_cross_entropy_with_logits(les_logits, yb_les) + 0.5 * dice_loss_fn(les_logits, yb_les) + 3.5 * loss_sev

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()

    with torch.no_grad():
        tr_probs_128 = torch.sigmoid(model(x_train, rl_train).squeeze(1)).numpy()
        cal_probs_full = [
            torch.sigmoid(model(s["grid_tokens"].unsqueeze(0), s["rgblab_128"].unsqueeze(0), target_size=s["gt_mask_full"].shape).squeeze()).numpy()
            for s in cal_samples
        ]
        cal_probs_128 = [
            torch.sigmoid(model(s["grid_tokens"].unsqueeze(0), s["rgblab_128"].unsqueeze(0)).squeeze()).numpy()
            for s in cal_samples
        ]

    # Tune optimal pixel threshold tau* on calibration split (leak-free)
    best_tau, best_dice = 0.45, -1.0
    for tau in np.linspace(0.35, 0.65, 13):
        dices = []
        for pr, s in zip(cal_probs_full, cal_samples):
            pm = (pr > tau) & s["leaf_mask_full"]
            gm = s["gt_mask_full"]
            inter = np.sum(pm & gm)
            dices.append((2.0 * inter + 1e-6) / (np.sum(pm) + np.sum(gm) + 1e-6))
        if np.mean(dices) > best_dice:
            best_dice = float(np.mean(dices))
            best_tau = float(tau)

    # Fit Stage-2 Ridge Severity Refiner strictly on training split
    X_tr_s2 = np.stack([extract_stage2_features(pr, s["gt_leaf_128"].numpy(), s["lab_128"], best_tau) for pr, s in zip(tr_probs_128, train_samples)])
    y_tr_s2 = np.array([s["gs_pct"] for s in train_samples])
    s2_reg = Ridge(alpha=5.0).fit(X_tr_s2, y_tr_s2)

    # Calibrate Mondrian Conformal on calibration split using Stage-2 predictions
    X_cal_s2 = np.stack([extract_stage2_features(pr, s["gt_leaf_128"].numpy(), s["lab_128"], best_tau) for pr, s in zip(cal_probs_128, cal_samples)])
    cal_pred_frac = np.clip(s2_reg.predict(X_cal_s2) / 100.0, 0.0, 1.0)
    cal_true_frac = np.array([s["gs_pct"] / 100.0 for s in cal_samples])

    calibrator = MondrianConformalCalibrator(alpha=0.10)
    calibrator.fit(cal_true_frac, cal_pred_frac)

    # Evaluate strictly out-of-sample on held-out test split
    fold_preds = []
    with torch.no_grad():
        for s in test_samples:
            t0 = time.time()
            lg_128 = model(s["grid_tokens"].unsqueeze(0), s["rgblab_128"].unsqueeze(0))
            prob_128 = torch.sigmoid(lg_128.squeeze()).numpy()
            lg_full = F.interpolate(lg_128, size=s["gt_mask_full"].shape, mode="bilinear", align_corners=False)
            prob_full = torch.sigmoid(lg_full.squeeze()).numpy()

            leaf_m = s["leaf_mask_full"]
            pred_les = (prob_full > best_tau) & leaf_m

            x_test_s2 = extract_stage2_features(prob_128, s["gt_leaf_128"].numpy(), s["lab_128"], best_tau).reshape(1, -1)
            sev_frac = float(np.clip(s2_reg.predict(x_test_s2)[0] / 100.0, 0.0, 1.0))
            low, high, width = calibrator.predict_interval(sev_frac)
            dt = time.time() - t0

            fold_preds.append({
                "image": s["image"],
                "gs_pct": s["gs_pct"],
                "pred_pct": round(sev_frac * 100.0, 2),
                "low_pct": round(low * 100.0, 2),
                "high_pct": round(high * 100.0, 2),
                "width_pct": round(width * 100.0, 2),
                "pred_mask": pred_les.astype(np.uint8),
                "pred_prob": (prob_full * leaf_m).astype(np.float32),
                "gt_mask": s["gt_mask_full"].astype(np.uint8),
                "head_latency_ms": dt * 1000.0,
            })

    return fold_preds


def run_coffee_rust_benchmark() -> Dict[str, Any]:
    """Run strict 5-fold out-of-sample DINOv2 + SegHeads evaluation on 100 Coffee Rust images."""
    base_dir = PROJECT_ROOT / "data" / "raw" / "coffee_rust"
    csv_path = base_dir / "03_analysis" / "severity_final.csv"
    img_dir = base_dir / "02_binary_images" / "base_image"
    mask_dir = base_dir / "02_binary_images" / "golden_standard"

    df = pd.read_csv(csv_path)
    cache = extract_or_load_dinov2_cache(df, img_dir, mask_dir)
    samples = cache["samples"]
    for s in samples:
        if "rgblab_128" not in s:
            pil = Image.open(img_dir / s["image"]).convert("RGB").resize((128, 128), Image.BILINEAR)
            rgb = np.array(pil, dtype=np.uint8)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            rgb_f = rgb.astype(np.float32) / 255.0
            s["rgblab_128"] = torch.from_numpy(np.concatenate([rgb_f, lab], axis=-1)).permute(2, 0, 1)
            s["lab_128"] = lab

    # Stratified 5-fold split by GS severity
    sorted_indices = np.argsort([s["gs_pct"] for s in samples])
    folds = [sorted_indices[k::5] for k in range(5)]

    all_preds_map = {}
    for k in range(5):
        test_idx = set(folds[k].tolist())
        rem_idx = [i for i in range(len(samples)) if i not in test_idx]
        cal_idx = rem_idx[::5]
        cal_set = set(cal_idx)
        train_idx = [i for i in rem_idx if i not in cal_set]

        train_s = [samples[i] for i in train_idx]
        cal_s = [samples[i] for i in cal_idx]
        test_s = [samples[i] for i in folds[k]]

        print(f"Training & evaluating Fold {k+1}/5 (Train={len(train_s)}, Cal={len(cal_s)}, Test={len(test_s)})...")
        f_preds = train_and_eval_fold(train_s, cal_s, test_s, epochs=25, seed=42 + k)
        for p in f_preds:
            all_preds_map[p["image"]] = p

    records = []
    seg_metrics_list = []
    total_tp = total_tn = total_fp = total_fn = 0
    latencies_ms = []

    # Subsample pixels for fast global pixel-level ROC-AUC (1000 pixels per image -> 100,000 pixels)
    rng = np.random.RandomState(42)
    pix_gt_sample = []
    pix_prob_sample = []

    for _, row in df.iterrows():
        fname = row["image"]
        p = all_preds_map[fname]
        gt_bin = p["gt_mask"]
        pred_bin = p["pred_mask"]

        pm = compute_pixel_metrics(pred_bin, gt_bin)
        seg_metrics_list.append(pm)
        total_tp += pm["tp"]
        total_tn += pm["tn"]
        total_fp += pm["fp"]
        total_fn += pm["fn"]
        latencies_ms.append(p["head_latency_ms"])

        flat_gt = gt_bin.ravel()
        flat_pr = p["pred_prob"].ravel()
        idx_sub = rng.choice(len(flat_gt), size=min(2000, len(flat_gt)), replace=False)
        pix_gt_sample.append(flat_gt[idx_sub])
        pix_prob_sample.append(flat_pr[idx_sub])

        gs_pct = float(row["GS"])
        covered = (gs_pct >= p["low_pct"]) and (gs_pct <= p["high_pct"])

        records.append({
            "image": fname,
            "gs_pct": gs_pct,
            "pred_pct": p["pred_pct"],
            "low_pct": p["low_pct"],
            "high_pct": p["high_pct"],
            "width_pct": p["width_pct"],
            "covered": covered,
            "dice": pm["dice"],
            "iou": pm["iou"],
            "pixel_accuracy": pm["pixel_accuracy"],
            "ImageJ": float(row["ImageJ"]),
            "Pliman": float(row["Pliman"]),
            "SAM3": float(row["SAM3"]),
            "DeeplabV3": float(row["DeeplabV3"]),
        })

    res_df = pd.DataFrame(records)

    # Pixel-level ROC curve
    pix_gt_all = np.concatenate(pix_gt_sample)
    pix_prob_all = np.concatenate(pix_prob_sample)
    pix_fpr, pix_tpr, _ = roc_curve(pix_gt_all, pix_prob_all)
    pix_roc_auc = float(auc(pix_fpr, pix_tpr))

    # Downsample curve points for plotting
    step = max(1, len(pix_fpr) // 250)
    pix_roc = {
        "roc_auc": round(pix_roc_auc, 4),
        "fpr": [round(float(x), 4) for x in pix_fpr[::step]] + [1.0],
        "tpr": [round(float(x), 4) for x in pix_tpr[::step]] + [1.0],
    }

    eps = 1e-7
    global_dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    global_iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    global_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)

    seg_summary = {
        "n_images": len(seg_metrics_list),
        "macro_dice": round(float(np.mean([m["dice"] for m in seg_metrics_list])), 4),
        "macro_dice_std": round(float(np.std([m["dice"] for m in seg_metrics_list])), 4),
        "macro_iou": round(float(np.mean([m["iou"] for m in seg_metrics_list])), 4),
        "macro_pixel_accuracy": round(float(np.mean([m["pixel_accuracy"] for m in seg_metrics_list])), 4),
        "macro_precision": round(float(np.mean([m["precision"] for m in seg_metrics_list])), 4),
        "macro_recall": round(float(np.mean([m["recall"] for m in seg_metrics_list])), 4),
        "macro_specificity": round(float(np.mean([m["specificity"] for m in seg_metrics_list])), 4),
        "micro_global_dice": round(float(global_dice), 4),
        "micro_global_iou": round(float(global_iou), 4),
        "micro_global_pixel_accuracy": round(float(global_acc), 4),
        "pixel_roc_auc": pix_roc["roc_auc"],
    }

    y_true = res_df["gs_pct"].values
    reg_ours = evaluate_regression_series(y_true, res_df["pred_pct"].values)
    baselines = {}
    for col in ["ImageJ", "Pliman", "SAM3", "DeeplabV3"]:
        baselines[col] = evaluate_regression_series(y_true, res_df[col].values)

    conformal_summary = {
        "target_coverage_pct": 90.0,
        "empirical_coverage_pct": round(float(res_df["covered"].mean() * 100.0), 2),
        "mean_interval_width_pct": round(float(res_df["width_pct"].mean()), 4),
    }

    y_true_ord = np.array([map_severity_to_ordinal_class(v / 100.0)[0] for v in y_true])
    y_pred_ord = np.array([map_severity_to_ordinal_class(v / 100.0)[0] for v in res_df["pred_pct"].values])
    ord_acc = float(accuracy_score(y_true_ord, y_pred_ord))
    ord_macro_f1 = float(f1_score(y_true_ord, y_pred_ord, average="macro", zero_division=0))
    qwk = compute_quadratic_weighted_kappa(y_true_ord, y_pred_ord)
    present_labels = sorted(list(set(y_true_ord) | set(y_pred_ord)))
    cm = confusion_matrix(y_true_ord, y_pred_ord, labels=present_labels)

    ordinal_summary = {
        "ordinal_accuracy": round(ord_acc, 4),
        "ordinal_accuracy_pct": round(ord_acc * 100.0, 2),
        "quadratic_weighted_kappa": round(qwk, 4),
        "macro_f1": round(ord_macro_f1, 4),
        "present_class_indices": [int(x) for x in present_labels],
        "present_class_labels": [DEFAULT_BIN_LABELS[x] for x in present_labels],
        "confusion_matrix": cm.tolist(),
    }

    roc_5 = evaluate_binary_roc(y_true, res_df["pred_pct"].values, threshold=5.0)
    roc_10 = evaluate_binary_roc(y_true, res_df["pred_pct"].values, threshold=10.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    latency_summary = {
        "device": "cpu",
        "decoder_head_mean_latency_ms": round(float(np.mean(lat_arr)), 3),
        "decoder_head_std_latency_ms": round(float(np.std(lat_arr)), 3),
    }

    return {
        "records_df": res_df,
        "segmentation": seg_summary,
        "pixel_roc": pix_roc,
        "regression_ours": reg_ours,
        "baselines": baselines,
        "conformal": conformal_summary,
        "ordinal": ordinal_summary,
        "roc_threshold_5pct": roc_5,
        "roc_threshold_10pct": roc_10,
        "latency": latency_summary,
    }


def run_forecast_weather_evaluation() -> Dict[str, Any]:
    """Evaluate real-time Forecaster across cached 9 km Open-Meteo NWP weather profiles."""
    weather_dir = PROJECT_ROOT / "data" / "interim" / "weather_cache"
    weather_files = sorted(weather_dir.glob("nwp_forecast_*.json"))
    forecaster = Forecaster(models_dir=PROJECT_ROOT / "models" / "forecast")

    initial_severities = [0.02, 0.05, 0.12, 0.25]
    results_3d, results_7d, latencies_ms = [], [], []
    monotonic_violations = 0
    bound_violations = 0
    total_runs = 0

    for wf in weather_files:
        payload = json.loads(wf.read_text(encoding="utf-8"))
        hourly = payload.get("hourly", {})
        if not hourly or "time" not in hourly:
            continue
        hdf = pd.DataFrame({
            "date": pd.to_datetime(hourly["time"]).date,
            "t_mean": hourly.get("temperature_2m", 20.0),
            "rh_mean": hourly.get("relative_humidity_2m", 75.0),
            "precip": hourly.get("precipitation", 0.0),
        })
        w_df = hdf.groupby("date", as_index=False).agg({
            "t_mean": "mean",
            "rh_mean": "mean",
            "precip": "sum",
        })
        w_df["date"] = w_df["date"].astype(str)

        for s0 in initial_severities:
            p3 = forecaster.predict(current_severity=s0, dt_days=3.0, weather_df=w_df)
            p7 = forecaster.predict(current_severity=s0, dt_days=7.0, weather_df=w_df)

            total_runs += 2
            latencies_ms.extend([p3.execution_time_sec * 1000.0, p7.execution_time_sec * 1000.0])

            if p7.median_severity_percentage + 1e-6 < p3.median_severity_percentage:
                monotonic_violations += 1
            for p in (p3, p7):
                if not (0.0 <= p.interval_lower_percentage <= p.median_severity_percentage <= p.interval_upper_percentage <= 100.0):
                    bound_violations += 1

            results_3d.append({"s0_pct": s0 * 100.0, "med_pct": p3.median_severity_percentage, "width_pct": p3.interval_width_percentage, "delta_pct": p3.median_severity_percentage - s0 * 100.0})
            results_7d.append({"s0_pct": s0 * 100.0, "med_pct": p7.median_severity_percentage, "width_pct": p7.interval_width_percentage, "delta_pct": p7.median_severity_percentage - s0 * 100.0})

    df3 = pd.DataFrame(results_3d)
    df7 = pd.DataFrame(results_7d)

    return {
        "n_real_nwp_locations": len(weather_files),
        "total_forecast_evaluations": total_runs,
        "physical_monotonicity_pass_rate_pct": round(100.0 * (1.0 - monotonic_violations / max(1, len(results_3d))), 2),
        "interval_bound_compliance_pct": round(100.0 * (1.0 - bound_violations / max(1, total_runs)), 2),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "horizon_3_day": {
            "mean_predicted_delta_pct": round(float(df3["delta_pct"].mean()), 3),
            "mean_cqr_interval_width_pct": round(float(df3["width_pct"].mean()), 3),
        },
        "horizon_7_day": {
            "mean_predicted_delta_pct": round(float(df7["delta_pct"].mean()), 3),
            "mean_cqr_interval_width_pct": round(float(df7["width_pct"].mean()), 3),
        },
    }


def generate_paper_figures(bench: Dict[str, Any], fig_dir: Path) -> List[str]:
    """Generate publication-quality PNG figures for the IEEE research paper."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    res_df = bench["records_df"]

    # 1. ROC Curves Figure (Pixel-level + Image-level 5% and 10%)
    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=300)
    pr = bench["pixel_roc"]
    r5 = bench["roc_threshold_5pct"]
    r10 = bench["roc_threshold_10pct"]
    ax.plot(pr["fpr"], pr["tpr"], color="#2ca02c", lw=2.3, label=f"Pixel-Level Lesion Seg. (AUC = {pr['roc_auc']:.4f})")
    ax.plot(r5["fpr"], r5["tpr"], color="#1f77b4", lw=2.1, label=f"Severity >= 5% Detection (AUC = {r5['roc_auc']:.4f})")
    ax.plot(r10["fpr"], r10["tpr"], color="#d62728", lw=2.1, label=f"Severity >= 10% Detection (AUC = {r10['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Baseline (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves: Pixel Segmentation & Severity Thresholds (5-Fold CV)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax.grid(alpha=0.3)
    roc_path = fig_dir / "paper_roc_curves.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    saved_paths.append(str(roc_path))

    # 2. Parity Plot + Bland-Altman Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    yt = res_df["gs_pct"].values
    yp = res_df["pred_pct"].values

    axes[0].scatter(yt, yp, alpha=0.8, color="#2ca02c", edgecolor="k", linewidth=0.5, s=44)
    max_val = max(yt.max(), yp.max()) * 1.05
    axes[0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Identity (y = x)")
    reg = bench["regression_ours"]
    axes[0].set_xlabel("Expert Golden Standard Severity (%)", fontsize=11)
    axes[0].set_ylabel("Predicted Severity (%)", fontsize=11)
    axes[0].set_title(f"5-Fold Parity Plot (MAE={reg['mae_pct']:.2f}%, CCC={reg['lins_ccc']:.3f}, R2={reg['r2']:.3f})", fontsize=10.5, fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    means = (yt + yp) / 2.0
    diffs = yp - yt
    bias = reg["bland_altman_bias_pct"]
    loa_l = reg["bland_altman_loa_lower_pct"]
    loa_u = reg["bland_altman_loa_upper_pct"]
    axes[1].scatter(means, diffs, alpha=0.8, color="#1f77b4", edgecolor="k", linewidth=0.5, s=44)
    axes[1].axhline(bias, color="black", lw=1.5, label=f"Mean Bias ({bias:+.2f}%)")
    axes[1].axhline(loa_u, color="red", linestyle="--", lw=1.3, label=f"+1.96 SD ({loa_u:+.2f}%)")
    axes[1].axhline(loa_l, color="red", linestyle="--", lw=1.3, label=f"-1.96 SD ({loa_l:+.2f}%)")
    axes[1].set_xlabel("Mean of Predicted & Golden Standard (%)", fontsize=11)
    axes[1].set_ylabel("Difference (Predicted - Golden Standard, %)", fontsize=11)
    axes[1].set_title("Bland-Altman Agreement Plot (95% LoA)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    ba_path = fig_dir / "paper_severity_parity_bland_altman.png"
    fig.tight_layout()
    fig.savefig(ba_path, dpi=300)
    plt.close(fig)
    saved_paths.append(str(ba_path))

    # 3. Baseline Comparison Bar Chart
    models = ["Proposed\n(DINOv2+Head)", "DeepLabV3", "Pliman", "SAM3", "ImageJ"]
    maes = [
        bench["regression_ours"]["mae_pct"],
        bench["baselines"]["DeeplabV3"]["mae_pct"],
        bench["baselines"]["Pliman"]["mae_pct"],
        bench["baselines"]["SAM3"]["mae_pct"],
        bench["baselines"]["ImageJ"]["mae_pct"],
    ]
    cccs = [
        bench["regression_ours"]["lins_ccc"],
        bench["baselines"]["DeeplabV3"]["lins_ccc"],
        bench["baselines"]["Pliman"]["lins_ccc"],
        bench["baselines"]["SAM3"]["lins_ccc"],
        bench["baselines"]["ImageJ"]["lins_ccc"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    colors = ["#2ca02c", "#1f77b4", "#7f7f7f", "#7f7f7f", "#7f7f7f"]
    axes[0].bar(models, maes, color=colors, edgecolor="black", linewidth=0.7)
    axes[0].set_ylabel("MAE (% points, Lower is Better)", fontsize=10.5)
    axes[0].set_title("Severity Estimation MAE vs. Baselines", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=9.5, fontweight="bold" if i == 0 else "normal")

    axes[1].bar(models, cccs, color=colors, edgecolor="black", linewidth=0.7)
    axes[1].set_ylabel("Lin's CCC (Higher is Better)", fontsize=10.5)
    axes[1].set_title("Concordance Correlation (CCC) vs. Baselines", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(cccs):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold" if i == 0 else "normal")

    comp_path = fig_dir / "paper_baseline_comparison.png"
    fig.tight_layout()
    fig.savefig(comp_path, dpi=300)
    plt.close(fig)
    saved_paths.append(str(comp_path))

    # 4. Ordinal Confusion Matrix
    ord_data = bench["ordinal"]
    cm = np.array(ord_data["confusion_matrix"])
    labels = ord_data["present_class_labels"]

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Predicted Severity Grade", fontsize=10.5)
    ax.set_ylabel("Ground Truth Severity Grade", fontsize=10.5)
    ax.set_title(f"Ordinal Grading Confusion Matrix (Acc={ord_data['ordinal_accuracy_pct']}%, QWK={ord_data['quadratic_weighted_kappa']:.3f})", fontsize=10.5, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = fig_dir / "paper_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    saved_paths.append(str(cm_path))

    return saved_paths


def main() -> int:
    tables_dir = PROJECT_ROOT / "reports" / "tables"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bench = run_coffee_rust_benchmark()
    forecast_eval = run_forecast_weather_evaluation()

    safety_path = tables_dir / "safety_stress_test_results.json"
    safety_eval = json.loads(safety_path.read_text(encoding="utf-8")) if safety_path.exists() else {}

    figure_paths = generate_paper_figures(bench, figures_dir)

    full_metrics = {
        "benchmark_dataset": "Coffee Leaf Rust Golden Standard (N=100 images, strict 5-fold out-of-sample CV)",
        "segmentation": bench["segmentation"],
        "severity_regression_proposed": bench["regression_ours"],
        "severity_regression_baselines": bench["baselines"],
        "conformal_uncertainty": bench["conformal"],
        "ordinal_classification": bench["ordinal"],
        "binary_detection_roc": {
            "threshold_5pct": {k: v for k, v in bench["roc_threshold_5pct"].items() if k not in ("fpr", "tpr")},
            "threshold_10pct": {k: v for k, v in bench["roc_threshold_10pct"].items() if k not in ("fpr", "tpr")},
        },
        "inference_latency": bench["latency"],
        "forecast_7day_real_nwp_eval": forecast_eval,
        "safety_stress_test": {
            "total_stress_cases_tested": safety_eval.get("total_stress_cases_tested"),
            "detected_degraded_or_ood_cases": safety_eval.get("detected_degraded_or_ood_cases"),
            "safety_detection_rate_pct": safety_eval.get("safety_detection_rate_pct"),
        },
        "generated_figures": figure_paths,
    }

    json_out = tables_dir / "paper_evaluation_metrics.json"
    json_out.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")

    comp_rows = [{"Method": "Proposed (DINOv2+SegHeads)", **bench["regression_ours"]}]
    for bname, bmetrics in bench["baselines"].items():
        comp_rows.append({"Method": bname, **bmetrics})
    pd.DataFrame(comp_rows).to_csv(tables_dir / "paper_baseline_comparison.csv", index=False)

    summary_rows = [
        {"Category": "Segmentation", "Metric": "Macro Dice (DSC)", "Value": bench["segmentation"]["macro_dice"]},
        {"Category": "Segmentation", "Metric": "Micro Global Dice", "Value": bench["segmentation"]["micro_global_dice"]},
        {"Category": "Segmentation", "Metric": "Macro IoU (Jaccard)", "Value": bench["segmentation"]["macro_iou"]},
        {"Category": "Segmentation", "Metric": "Pixel Accuracy", "Value": bench["segmentation"]["macro_pixel_accuracy"]},
        {"Category": "Segmentation", "Metric": "Precision", "Value": bench["segmentation"]["macro_precision"]},
        {"Category": "Segmentation", "Metric": "Recall (Sensitivity)", "Value": bench["segmentation"]["macro_recall"]},
        {"Category": "Segmentation", "Metric": "Specificity", "Value": bench["segmentation"]["macro_specificity"]},
        {"Category": "Segmentation", "Metric": "Pixel-Level ROC-AUC", "Value": bench["segmentation"]["pixel_roc_auc"]},
        {"Category": "Regression", "Metric": "MAE (% points)", "Value": bench["regression_ours"]["mae_pct"]},
        {"Category": "Regression", "Metric": "RMSE (% points)", "Value": bench["regression_ours"]["rmse_pct"]},
        {"Category": "Regression", "Metric": "R^2 Score", "Value": bench["regression_ours"]["r2"]},
        {"Category": "Regression", "Metric": "Lin's CCC", "Value": bench["regression_ours"]["lins_ccc"]},
        {"Category": "Regression", "Metric": "Pearson r", "Value": bench["regression_ours"]["pearson_r"]},
        {"Category": "Regression", "Metric": "Spearman rho", "Value": bench["regression_ours"]["spearman_rho"]},
        {"Category": "Regression", "Metric": "Bland-Altman Bias (%)", "Value": bench["regression_ours"]["bland_altman_bias_pct"]},
        {"Category": "Conformal UQ", "Metric": "Empirical 90% Coverage (%)", "Value": bench["conformal"]["empirical_coverage_pct"]},
        {"Category": "Conformal UQ", "Metric": "Mean Interval Width (%)", "Value": bench["conformal"]["mean_interval_width_pct"]},
        {"Category": "Classification", "Metric": "Ordinal Accuracy (%)", "Value": bench["ordinal"]["ordinal_accuracy_pct"]},
        {"Category": "Classification", "Metric": "Quadratic Weighted Kappa (QWK)", "Value": bench["ordinal"]["quadratic_weighted_kappa"]},
        {"Category": "Classification", "Metric": "Macro-F1 Score", "Value": bench["ordinal"]["macro_f1"]},
        {"Category": "ROC / AUC", "Metric": "ROC-AUC (>= 5% Severity)", "Value": bench["roc_threshold_5pct"]["roc_auc"]},
        {"Category": "ROC / AUC", "Metric": "PR-AUC (>= 5% Severity)", "Value": bench["roc_threshold_5pct"]["pr_auc"]},
        {"Category": "ROC / AUC", "Metric": "Accuracy (>= 5% Severity)", "Value": bench["roc_threshold_5pct"]["accuracy"]},
        {"Category": "ROC / AUC", "Metric": "F1-Score (>= 5% Severity)", "Value": bench["roc_threshold_5pct"]["f1_score"]},
        {"Category": "ROC / AUC", "Metric": "ROC-AUC (>= 10% Severity)", "Value": bench["roc_threshold_10pct"]["roc_auc"]},
        {"Category": "ROC / AUC", "Metric": "PR-AUC (>= 10% Severity)", "Value": bench["roc_threshold_10pct"]["pr_auc"]},
        {"Category": "ROC / AUC", "Metric": "Accuracy (>= 10% Severity)", "Value": bench["roc_threshold_10pct"]["accuracy"]},
        {"Category": "ROC / AUC", "Metric": "F1-Score (>= 10% Severity)", "Value": bench["roc_threshold_10pct"]["f1_score"]},
        {"Category": "Efficiency", "Metric": "Decoder Head Latency (ms/img)", "Value": bench["latency"]["decoder_head_mean_latency_ms"]},
        {"Category": "Forecasting", "Metric": "Real NWP Locations Tested", "Value": forecast_eval["n_real_nwp_locations"]},
        {"Category": "Forecasting", "Metric": "Interval Bound Compliance (%)", "Value": forecast_eval["interval_bound_compliance_pct"]},
        {"Category": "Forecasting", "Metric": "Forecast Latency (ms)", "Value": forecast_eval["mean_latency_ms"]},
        {"Category": "Safety Gate", "Metric": "OOD/Degraded Detection Rate (%)", "Value": safety_eval.get("safety_detection_rate_pct")},
    ]
    pd.DataFrame(summary_rows).to_csv(tables_dir / "paper_metrics_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("IEEE RESEARCH PAPER EVALUATION COMPLETED (5-FOLD DINOv2 + SegHeads)")
    print("=" * 72)
    for r in summary_rows:
        print(f"[{r['Category']:<14}] {r['Metric']:<34}: {r['Value']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
