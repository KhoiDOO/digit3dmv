"""Benchmark script for evaluating ODE Integration Steps vs. Geometric Quality and Latency.

This script sweeps through different Euler ODE step counts (e.g. 2, 5, 10, 20, 25, 50, 100),
generating multi-view normal maps and measuring the exact Mean Angular Error (MAE),
PSNR, SSIM, angular accuracy thresholds, and per-sample latency.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import sys
import math
import time
import json
import argparse
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure local repository root is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from conquer3d.data import Digit3DMV
from models import MVPixelDiT
from rectified_flow_pytorch import RectifiedFlow, MeanFlow
from eval import evaluate_normal_pair


class MeanFlowModelWrapper(nn.Module):
    """Adapts MVPixelDiT interface (x, t, img_cond) for MeanFlow forward expectations."""

    def __init__(self, inner_model: nn.Module):
        super().__init__()
        self.inner_model = inner_model

    def forward(self, x: torch.Tensor, time: torch.Tensor, delta_time: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner_model(x, t=time, img_cond=cond)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for ODE step evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ODE Sampling Steps vs Multi-View Normal Consistency")

    parser.add_argument("--exp_dir", type=str, required=True, help="Path or name of the experiment directory (e.g. 'outputs/normal_4_rectified_flow_...').")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root outputs directory if exp_dir is a relative folder name.")
    parser.add_argument("--checkpoint", type=str, default="best", help="Checkpoint to load: 'best', 'latest', or specific .pt file name.")
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 20, 25, 50, 100], help="List of Euler ODE step counts to evaluate.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of validation samples per digit class (default: 1 -> 10 samples for classes 0-9).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation generation.")
    parser.add_argument("--flip_vertical", action="store_true", default=True, help="Flip input condition to match model native coordinate space if trained prior to upright re-rendering.")
    parser.add_argument("--save_json", type=str, default=None, help="Path to save benchmark JSON (default: {exp_dir}/ode_steps_benchmark.json).")
    parser.add_argument("--root", type=str, default=None, help="Root directory containing Digit3DMV dataset (default: auto-detects 'data/' or '~/.conquer3d/').")
    parser.add_argument("--use_zip", action="store_true", default=None, help="Force reading dataset from zip archive.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device.")

    return parser.parse_args()


def extract_batch_tensors(batch: Dict[str, Any], modality: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts condition front image and target multi-view ground truth tensors normalized to [-1, 1]."""
    if modality == "normal":
        img_cond = batch["front"]
        x_clean = batch["360"]
    elif modality == "depth":
        img_cond = batch["front"]
        x_clean = batch["360"]
    elif modality == "both":
        img_cond = torch.cat([batch["normal_front"], batch["depth_front"]], dim=1)
        x_clean = torch.cat([batch["normal_360"], batch["depth_360"]], dim=2)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Scale from [0, 1] to [-1, 1]
    img_cond_norm = img_cond * 2.0 - 1.0
    x_clean_norm = x_clean * 2.0 - 1.0
    return img_cond_norm, x_clean_norm


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Resolve Experiment Directory
    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        candidate = os.path.join(args.output_dir, exp_dir)
        if os.path.isdir(candidate):
            exp_dir = candidate
        else:
            raise FileNotFoundError(f"Experiment directory not found: '{args.exp_dir}' or '{candidate}'")

    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file 'config.json' not found in: {exp_dir}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Resolve Checkpoint Path
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if args.checkpoint.lower() == "best":
        ckpt_path = os.path.join(ckpt_dir, "model_best.pt")
    elif args.checkpoint.lower() == "latest":
        ckpt_path = os.path.join(ckpt_dir, "model_latest.pt")
    else:
        if os.path.isfile(args.checkpoint):
            ckpt_path = args.checkpoint
        else:
            candidate = os.path.join(ckpt_dir, args.checkpoint)
            if os.path.isfile(candidate):
                ckpt_path = candidate
            else:
                raise FileNotFoundError(f"Checkpoint not found at: '{args.checkpoint}' or '{candidate}'")

    # Resolve dataset root
    data_root = args.root
    if data_root is None:
        local_data = os.path.join(REPO_ROOT, "data")
        if os.path.isdir(local_data) and (os.path.isdir(os.path.join(local_data, "mv_64")) or os.path.isfile(os.path.join(local_data, "mv_64.zip"))):
            data_root = local_data
        else:
            data_root = os.path.expanduser("~/.conquer3d/")

    print("================================================================================")
    print(" ODE Integration Steps vs Multi-View Geometric Consistency Benchmark")
    print("================================================================================")
    print(f"Experiment:       {exp_dir}")
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Dataset Root:     {data_root}")
    print(f"Step Counts:      {args.steps}")
    print(f"Samples / Class:  {args.num_samples} (across 10 classes)")
    print(f"Coordinate Flip:  {args.flip_vertical}")
    print(f"Device:           {device}")
    print("================================================================================")

    # Determine channel count and parameters
    modality = cfg.get("modality", "normal")
    in_channels = 3 if modality == "normal" else (1 if modality == "depth" else 4)
    num_views = cfg.get("num_views", 4)
    resolution = cfg.get("resolution", 64)

    # Instantiate Model
    model = MVPixelDiT(
        in_channels=in_channels,
        out_channels=in_channels,
        hidden_size=cfg.get("hidden_size", 256),
        pixel_hidden_size=cfg.get("pixel_hidden_size", 64),
        num_heads=cfg.get("num_heads", 8),
        patch_depth=cfg.get("patch_depth", 6),
        pixel_depth=cfg.get("pixel_depth", 2),
        patch_size=cfg.get("patch_size", 4),
        num_views=num_views,
        use_pixel_abs_pos=not cfg.get("no_pixel_abs_pos", False),
        cond_drop_prob=cfg.get("cond_drop_prob", 0.15),
        attn_func=cfg.get("attn_func", "torch"),
    ).to(device)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded EMA shadow model weights.")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights.")
    model.eval()

    # Build Flow Matching Engine
    mode = cfg.get("mode", "rectified_flow")
    data_shape = (num_views, in_channels, resolution, resolution)

    if mode == "rectified_flow":
        flow_model = RectifiedFlow(
            model=model,
            time_cond_kwarg="t",
            data_shape=data_shape,
        )
    elif mode == "mean_flow":
        wrapped_model = MeanFlowModelWrapper(model)
        flow_model = MeanFlow(
            model=wrapped_model,
            data_shape=data_shape,
            accept_cond=True,
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    flow_model.eval()

    # Load Validation Dataset for Evaluation Inputs
    val_dataset = Digit3DMV(
        root=data_root,
        train=False,
        resolution=resolution,
        modality=modality,
        all_360=(num_views == 12),
        return_front_variation=False,
        return_c2w=False,
        use_zip=args.use_zip,
    )

    # Group validation indices by digit class (0-9)
    class_indices: Dict[int, List[int]] = {c: [] for c in range(10)}
    for idx, sample_name in enumerate(val_dataset.samples):
        lbl = int(sample_name.split("_")[0])
        class_indices[lbl].append(idx)

    selected_indices = []
    for c in range(10):
        available = class_indices[c]
        take = min(len(available), args.num_samples)
        selected_indices.extend(available[:take])

    eval_subset = Subset(val_dataset, selected_indices)
    eval_loader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Prepared {len(selected_indices)} evaluation test samples across all 10 digit classes.\n")

    # Step Benchmark Loop
    step_results = []

    print(f"{'ODE Steps':<10} | {'MAE (deg)':<10} | {'MedAE (deg)':<12} | {'Acc <11.25°':<12} | {'Acc <22.5°':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency (ms)':<12}")
    print("-" * 98)

    for num_steps in args.steps:
        all_step_metrics = []
        start_t = time.time()
        total_eval_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                img_cond, x_gt = extract_batch_tensors(batch, modality=modality)
                
                # Coordinate alignment: If upright dataset is used with a model trained prior to upright re-render,
                # flip condition into model space, then flip outputs back to upright space.
                if args.flip_vertical:
                    img_cond_model = torch.flip(img_cond, dims=[-2]).to(device, non_blocking=True)
                else:
                    img_cond_model = img_cond.to(device, non_blocking=True)

                B = img_cond.shape[0]
                total_eval_samples += B

                # Generate via Flow Matching ODE sampler
                if mode == "rectified_flow":
                    gen_views = flow_model.sample(
                        batch_size=B,
                        steps=num_steps,
                        img_cond=img_cond_model,
                    )
                else:
                    gen_views = flow_model.sample(
                        cond=img_cond_model,
                        steps=num_steps,
                    )

                if args.flip_vertical:
                    gen_views = torch.flip(gen_views, dims=[-2])

                # Unnormalize to [0, 1]
                gen_01 = torch.clamp((gen_views + 1.0) * 0.5, 0.0, 1.0).cpu().numpy()
                gt_01 = torch.clamp((x_gt + 1.0) * 0.5, 0.0, 1.0).cpu().numpy()

                for b in range(B):
                    for v in range(num_views):
                        gen_img = np.transpose(gen_01[b, v], (1, 2, 0))  # [H, W, C]
                        gt_img = np.transpose(gt_01[b, v], (1, 2, 0))   # [H, W, C]

                        res = evaluate_normal_pair(gen_img, gt_img)
                        all_step_metrics.append(res)

        elapsed_ms = (time.time() - start_t) * 1000.0
        ms_per_sample = elapsed_ms / max(total_eval_samples, 1)

        # Aggregate metrics for this step count
        keys = ["mae_deg", "medae_deg", "rmse_deg", "acc_5deg", "acc_11deg", "acc_22deg", "acc_30deg", "cosine_sim_pct", "psnr_db", "ssim"]
        avg_stats = {k: float(np.mean([m[k] for m in all_step_metrics])) for k in keys}
        avg_stats["steps"] = num_steps
        avg_stats["ms_per_sample"] = float(ms_per_sample)

        step_results.append(avg_stats)

        print(
            f"{num_steps:<10} | "
            f"{avg_stats['mae_deg']:<10.2f}° | "
            f"{avg_stats['medae_deg']:<12.2f}° | "
            f"{avg_stats['acc_11deg']:<11.1f}% | "
            f"{avg_stats['acc_22deg']:<11.1f}% | "
            f"{avg_stats['psnr_db']:<10.2f} | "
            f"{avg_stats['ssim']:<8.4f} | "
            f"{ms_per_sample:<10.1f} ms"
        )

    # Save to JSON
    json_path = args.save_json or os.path.join(exp_dir, "ode_steps_benchmark.json")
    output_payload = {
        "exp_dir": exp_dir,
        "benchmark_timestamp": datetime.now().isoformat(),
        "total_samples": len(selected_indices),
        "step_results": step_results,
    }

    with open(json_path, "w") as f:
        json.dump(output_payload, f, indent=4)

    print("\n================================================================================")
    print(f" ODE Steps Benchmark Completed! Saved results to:")
    print(f"   {json_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
