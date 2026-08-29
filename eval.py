"""Quantitative evaluation script for Multi-View Normal Consistency and Geometric Fidelity.

This script evaluates generated multi-view normal maps against ground-truth references.
Metrics computed include:
  - Mean Angular Error (MAE in degrees)
  - Median Angular Error (MedAE in degrees)
  - Angular Root Mean Squared Error (RMSE)
  - Percentage Accuracy within thresholds (<5.0°, <11.25°, <22.5°, <30.0°)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Measure (SSIM)
  - Per-view consistency across all camera azimuths (45°, 135°, 225°, 315°)
  - Per-digit class breakdown (0-9)
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import sys
import math
import json
import glob
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure local repository root is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Multi-View Normal Consistency & Geometric Accuracy")

    parser.add_argument("--exp_dir", type=str, required=True, help="Path or name of the experiment directory (e.g. 'outputs/normal_4_rectified_flow_...').")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root outputs directory if exp_dir is given as a relative folder name.")
    parser.add_argument("--generated_dir", type=str, default=None, help="Explicit path to generated directory (default: {exp_dir}/generated).")
    parser.add_argument("--save_json", type=str, default=None, help="Path to save evaluation metrics JSON (default: {exp_dir}/metrics.json).")
    parser.add_argument("--bg_threshold", type=float, default=0.02, help="Threshold to distinguish foreground normal pixels from background.")

    return parser.parse_args()


def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB for images in [0, 1]."""
    mse = np.mean((img_pred - img_gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def compute_ssim_simple(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Computes basic Structural Similarity (SSIM) index for float images in [0, 1]."""
    # img_pred, img_gt: [H, W, 3] in [0, 1]
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu1 = np.mean(img_pred)
    mu2 = np.mean(img_gt)
    sigma1_sq = np.var(img_pred)
    sigma2_sq = np.var(img_gt)
    sigma12 = np.mean((img_pred - mu1) * (img_gt - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.clip(ssim, -1.0, 1.0))


def evaluate_normal_pair(
    gen_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    bg_threshold: float = 0.02,
) -> Dict[str, float]:
    """Computes angular metrics, PSNR, and SSIM between predicted and ground-truth normal maps.

    Args:
        gen_rgb: [H, W, 3] float32 in [0, 1]
        gt_rgb:  [H, W, 3] float32 in [0, 1]
    """
    # 1. Image Quality Metrics
    psnr = compute_psnr(gen_rgb, gt_rgb)
    ssim = compute_ssim_simple(gen_rgb, gt_rgb)

    # 2. Decode RGB color [0, 1] to normal vector [-1, 1]
    gen_norm = gen_rgb * 2.0 - 1.0
    gt_norm = gt_rgb * 2.0 - 1.0

    # Normalize vectors
    gen_len = np.linalg.norm(gen_norm, axis=-1, keepdims=True) + 1e-8
    gt_len = np.linalg.norm(gt_norm, axis=-1, keepdims=True) + 1e-8

    gen_unit = gen_norm / gen_len
    gt_unit = gt_norm / gt_len

    # Identify foreground mask (where ground truth is not zero/background)
    # Background in rendered normal map is either (0, 0, 0) or [0.5, 0.5, 0.5]
    is_zero_bg = (np.abs(gt_rgb) < 1e-3).all(axis=-1)
    is_flat_bg = (np.abs(gt_rgb - 0.5) < bg_threshold).all(axis=-1)
    fg_mask = ~(is_zero_bg | is_flat_bg)

    if not np.any(fg_mask):
        fg_mask = np.ones(gen_rgb.shape[:2], dtype=bool)

    # Compute dot product on foreground
    dot = np.sum(gen_unit[fg_mask] * gt_unit[fg_mask], axis=-1)
    dot = np.clip(dot, -1.0, 1.0)

    # Angular error in degrees
    angles_rad = np.arccos(dot)
    angles_deg = np.degrees(angles_rad)

    mae = float(np.mean(angles_deg))
    medae = float(np.median(angles_deg))
    rmse = float(np.sqrt(np.mean(angles_deg**2)))

    acc_5 = float(np.mean(angles_deg < 5.0) * 100.0)
    acc_11 = float(np.mean(angles_deg < 11.25) * 100.0)
    acc_22 = float(np.mean(angles_deg < 22.5) * 100.0)
    acc_30 = float(np.mean(angles_deg < 30.0) * 100.0)

    # Cosine Similarity metric in [0, 100]%
    cosine_sim_pct = float(np.mean((dot + 1.0) * 0.5) * 100.0)

    return {
        "mae_deg": mae,
        "medae_deg": medae,
        "rmse_deg": rmse,
        "acc_5deg": acc_5,
        "acc_11deg": acc_11,
        "acc_22deg": acc_22,
        "acc_30deg": acc_30,
        "cosine_sim_pct": cosine_sim_pct,
        "psnr_db": psnr,
        "ssim": ssim,
    }


def main():
    args = parse_args()

    # Resolve Experiment Directory
    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        candidate = os.path.join(args.output_dir, exp_dir)
        if os.path.isdir(candidate):
            exp_dir = candidate
        else:
            raise FileNotFoundError(f"Experiment directory not found: '{args.exp_dir}' or '{candidate}'")

    gen_dir = args.generated_dir or os.path.join(exp_dir, "generated")
    views_dir = os.path.join(gen_dir, "views")

    if not os.path.isdir(views_dir):
        raise FileNotFoundError(f"Generated views directory 'views/' not found in: {gen_dir}")

    # Discover all sample subdirectories
    sample_dirs = sorted(glob.glob(os.path.join(views_dir, "class_*", "*")))
    if not sample_dirs:
        raise FileNotFoundError(f"No sample view directories found in: {views_dir}")

    print("================================================================================")
    print(" Multi-View Normal Consistency & Geometric Accuracy Evaluation")
    print("================================================================================")
    print(f"Experiment:       {exp_dir}")
    print(f"Views Directory:  {views_dir}")
    print(f"Total Samples:    {len(sample_dirs)}")
    print("================================================================================")

    # Metrics storage
    all_sample_metrics: List[Dict[str, Any]] = []
    view_metrics: Dict[int, List[Dict[str, float]]] = {}
    class_metrics: Dict[int, List[Dict[str, float]]] = {}

    for s_path in sample_dirs:
        # Determine class and sample name
        parts = s_path.rstrip("/\\").split(os.sep)
        sample_name = parts[-1]
        class_folder = parts[-2]
        class_idx = int(class_folder.replace("class_", "")) if "class_" in class_folder else int(sample_name.split("_")[0])

        if class_idx not in class_metrics:
            class_metrics[class_idx] = []

        # Find all generated view images
        gen_view_paths = sorted(glob.glob(os.path.join(s_path, "gen_view_*.png")))
        if not gen_view_paths:
            # Check for normal specific naming
            gen_view_paths = sorted(glob.glob(os.path.join(s_path, "gen_normal_view_*.png")))

        sample_view_results = []
        for g_path in gen_view_paths:
            # Extract view index (e.g. gen_view_02.png -> 2)
            fname = os.path.basename(g_path)
            v_idx = int(fname.replace("gen_view_", "").replace("gen_normal_view_", "").replace(".png", ""))
            gt_filename = fname.replace("gen_", "gt_")
            gt_path = os.path.join(s_path, gt_filename)

            if not os.path.isfile(gt_path):
                continue

            # Load images as float32 in [0, 1]
            gen_img = np.array(Image.open(g_path).convert("RGB"), dtype=np.float32) / 255.0
            gt_img = np.array(Image.open(gt_path).convert("RGB"), dtype=np.float32) / 255.0

            res = evaluate_normal_pair(gen_img, gt_img, bg_threshold=args.bg_threshold)
            res["view_idx"] = v_idx
            res["sample_name"] = sample_name
            res["class_idx"] = class_idx

            if v_idx not in view_metrics:
                view_metrics[v_idx] = []
            view_metrics[v_idx].append(res)
            class_metrics[class_idx].append(res)
            sample_view_results.append(res)

        if sample_view_results:
            all_sample_metrics.extend(sample_view_results)

    if not all_sample_metrics:
        raise RuntimeError("No matching generated and ground-truth view pairs found for evaluation.")

    # Aggregate Overall Statistics
    def compute_stats(metric_list: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = ["mae_deg", "medae_deg", "rmse_deg", "acc_5deg", "acc_11deg", "acc_22deg", "acc_30deg", "cosine_sim_pct", "psnr_db", "ssim"]
        out = {}
        for k in keys:
            vals = [m[k] for m in metric_list if k in m]
            out[k] = float(np.mean(vals)) if vals else 0.0
        return out

    global_stats = compute_stats(all_sample_metrics)

    # Per-View Statistics
    per_view_stats = {}
    if len(view_metrics) == 12:
        view_names = {
            0: "Eq 45° (El 0°)",
            1: "Eq 135° (El 0°)",
            2: "Eq 225° (El 0°)",
            3: "Eq 315° (El 0°)",
            4: "Top 45° (El +45°)",
            5: "Top 135° (El +45°)",
            6: "Top 225° (El +45°)",
            7: "Top 315° (El +45°)",
            8: "Bot 45° (El -45°)",
            9: "Bot 135° (El -45°)",
            10: "Bot 225° (El -45°)",
            11: "Bot 315° (El -45°)",
        }
    else:
        view_names = {
            0: "45° (Front-Right)",
            1: "135° (Back-Right)",
            2: "225° (Back-Left)",
            3: "315° (Front-Left)",
        }
    for v_idx in sorted(view_metrics.keys()):
        per_view_stats[v_idx] = {
            "name": view_names.get(v_idx, f"View {v_idx}"),
            "stats": compute_stats(view_metrics[v_idx]),
            "num_evaluated": len(view_metrics[v_idx]),
        }

    # Per-Class Statistics
    per_class_stats = {}
    for c_idx in sorted(class_metrics.keys()):
        per_class_stats[c_idx] = {
            "stats": compute_stats(class_metrics[c_idx]),
            "num_evaluated": len(class_metrics[c_idx]),
        }

    # Print Summary Tables
    print("\n--- [1] Overall Multi-View Normal Consistency Metrics ---")
    print(f"  • Mean Angular Error (MAE):     {global_stats['mae_deg']:.2f}°")
    print(f"  • Median Angular Error (MedAE): {global_stats['medae_deg']:.2f}°")
    print(f"  • Angular RMSE:                 {global_stats['rmse_deg']:.2f}°")
    print(f"  • Accuracy (<5.0°):             {global_stats['acc_5deg']:.1f}%")
    print(f"  • Accuracy (<11.25°):           {global_stats['acc_11deg']:.1f}%")
    print(f"  • Accuracy (<22.5°):            {global_stats['acc_22deg']:.1f}%")
    print(f"  • Accuracy (<30.0°):            {global_stats['acc_30deg']:.1f}%")
    print(f"  • Cosine Similarity:            {global_stats['cosine_sim_pct']:.2f}%")
    print(f"  • Normal PSNR:                  {global_stats['psnr_db']:.2f} dB")
    print(f"  • Structural Similarity (SSIM): {global_stats['ssim']:.4f}")

    print("\n--- [2] Per-View Consistency Breakdown ---")
    print(f"{'View Index':<10} | {'Azimuth Angle':<20} | {'MAE (deg)':<10} | {'Acc <11.25°':<12} | {'Acc <22.5°':<12} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 95)
    for v_idx, v_data in per_view_stats.items():
        st = v_data["stats"]
        v_name = v_data["name"]
        print(f"View {v_idx:<5} | {v_name:<20} | {st['mae_deg']:<10.2f}° | {st['acc_11deg']:<11.1f}% | {st['acc_22deg']:<11.1f}% | {st['psnr_db']:<10.2f} | {st['ssim']:<8.4f}")

    print("\n--- [3] Per-Class Geometric Accuracy Breakdown (Digits 0–9) ---")
    print(f"{'Digit Class':<12} | {'MAE (deg)':<10} | {'MedAE':<10} | {'Acc <11.25°':<12} | {'Acc <22.5°':<12} | {'PSNR (dB)':<10}")
    print("-" * 75)
    for c_idx in sorted(per_class_stats.keys()):
        st = per_class_stats[c_idx]["stats"]
        print(f"Class {c_idx:<7} | {st['mae_deg']:<10.2f}° | {st['medae_deg']:<10.2f}° | {st['acc_11deg']:<11.1f}% | {st['acc_22deg']:<11.1f}% | {st['psnr_db']:<10.2f}")

    # Save to JSON
    json_path = args.save_json or os.path.join(exp_dir, "metrics.json")
    output_data = {
        "exp_dir": exp_dir,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_views_evaluated": len(all_sample_metrics),
        "global_metrics": global_stats,
        "per_view_metrics": per_view_stats,
        "per_class_metrics": per_class_stats,
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print("\n================================================================================")
    print(f" Evaluation Completed! Saved structured metrics to:")
    print(f"   {json_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
