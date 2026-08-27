"""Inference generation script for Multi-View PixelDiT using Flow Matching.

This script loads the trained model and configuration from an experiment output folder,
samples condition images from the validation set (randomly or across all digit classes),
and performs batched Euler ODE generation using the official rectified_flow_pytorch API.
Generated samples and comparison grids are saved into the `generated/` directory inside
the experiment folder.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

# Ensure local repository root is in path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from conquer3d.data import Digit3DMV
from models import MVPixelDiT
from rectified_flow_pytorch import RectifiedFlow, MeanFlow


class MeanFlowModelWrapper(nn.Module):
    """Adapts MVPixelDiT interface (x, t, img_cond) for MeanFlow forward expectations."""

    def __init__(self, inner_model: nn.Module):
        super().__init__()
        self.inner_model = inner_model

    def forward(self, x: torch.Tensor, time: torch.Tensor, delta_time: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner_model(x, t=time, img_cond=cond)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for multi-view sample generation."""
    parser = argparse.ArgumentParser(description="Generate Multi-View 3D Samples with MVPixelDiT")

    parser.add_argument("--exp_dir", type=str, required=True, help="Path or name of the experiment directory (e.g. 'outputs/normal_4_rectified_flow_...').")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root outputs directory if exp_dir is given as a relative subfolder name.")
    parser.add_argument("--checkpoint", type=str, default="best", help="Checkpoint to load: 'best', 'latest', or specific .pt file name.")
    parser.add_argument("--full_class", action="store_true", default=False, help="If True, generate samples for all 10 digit classes (0-9).")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate per class (if full_class=True) or total random samples.")
    parser.add_argument("--sample_steps", type=int, default=25, help="Number of ODE integration steps for sampling.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for parallel multi-view generation.")
    parser.add_argument("--save_individual", action="store_true", default=True, help="Save individual view PNG images in addition to comparison grids.")
    parser.add_argument("--root", type=str, default="~/.conquer3d/", help="Root cache directory containing Digit3DMV dataset archives.")
    parser.add_argument("--use_zip", action="store_true", default=None, help="Force reading dataset from zip archive instead of uncompressed folder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device ('cuda' or 'cpu').")

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

    # Scale from [0, 1] to [-1, 1] for model conditioning
    img_cond_norm = img_cond * 2.0 - 1.0
    x_clean_norm = x_clean * 2.0 - 1.0
    return img_cond_norm, x_clean_norm


def build_comparison_row(
    cond_img: torch.Tensor,
    gen_views: torch.Tensor,
    gt_views: torch.Tensor,
    modality: str,
) -> List[torch.Tensor]:
    """Builds a single comparison row [Front_Cond | Gen_0 .. Gen_{V-1} | GT_0 .. GT_{V-1}]."""
    V = gen_views.shape[0]
    rows = []

    if modality in ("normal", "depth"):
        c = cond_img
        if c.shape[0] == 1:
            c = c.repeat(3, 1, 1)
        row = [c]
        for v in range(V):
            g = gen_views[v]
            if g.shape[0] == 1:
                g = g.repeat(3, 1, 1)
            row.append(g)
        for v in range(V):
            t = gt_views[v]
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            row.append(t)
        rows.append(torch.stack(row, dim=0))

    elif modality == "both":
        # Normal Maps row (RGB channels 0..2)
        c_norm = cond_img[:3]
        row_norm = [c_norm]
        for v in range(V):
            row_norm.append(gen_views[v, :3])
        for v in range(V):
            row_norm.append(gt_views[v, :3])
        rows.append(torch.stack(row_norm, dim=0))

        # Depth Maps row (channel 3 repeated to 3 channels)
        c_depth = cond_img[3:4].repeat(3, 1, 1)
        row_depth = [c_depth]
        for v in range(V):
            row_depth.append(gen_views[v, 3:4].repeat(3, 1, 1))
        for v in range(V):
            row_depth.append(gt_views[v, 3:4].repeat(3, 1, 1))
        rows.append(torch.stack(row_depth, dim=0))

    return rows


def save_single_sample_views(
    sample_dir: str,
    cond_01: torch.Tensor,
    gen_01: torch.Tensor,
    gt_01: torch.Tensor,
    modality: str,
):
    """Saves individual PNG files for condition, generated views, and ground truth views."""
    os.makedirs(sample_dir, exist_ok=True)
    V = gen_01.shape[0]

    if modality == "normal":
        save_image(cond_01, os.path.join(sample_dir, "front_condition.png"))
        for v in range(V):
            save_image(gen_01[v], os.path.join(sample_dir, f"gen_view_{v:02d}.png"))
            save_image(gt_01[v], os.path.join(sample_dir, f"gt_view_{v:02d}.png"))

    elif modality == "depth":
        save_image(cond_01, os.path.join(sample_dir, "front_condition.png"))
        for v in range(V):
            save_image(gen_01[v], os.path.join(sample_dir, f"gen_view_{v:02d}.png"))
            save_image(gt_01[v], os.path.join(sample_dir, f"gt_view_{v:02d}.png"))

    elif modality == "both":
        save_image(cond_01[:3], os.path.join(sample_dir, "front_condition_normal.png"))
        save_image(cond_01[3:4], os.path.join(sample_dir, "front_condition_depth.png"))
        for v in range(V):
            save_image(gen_01[v, :3], os.path.join(sample_dir, f"gen_normal_view_{v:02d}.png"))
            save_image(gen_01[v, 3:4], os.path.join(sample_dir, f"gen_depth_view_{v:02d}.png"))
            save_image(gt_01[v, :3], os.path.join(sample_dir, f"gt_normal_view_{v:02d}.png"))
            save_image(gt_01[v, 3:4], os.path.join(sample_dir, f"gt_depth_view_{v:02d}.png"))


def main():
    args = parse_args()

    # Set random seeds
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

    # Target save folder inside experiment directory
    gen_dir = os.path.join(exp_dir, "generated")
    grids_dir = os.path.join(gen_dir, "grids")
    views_dir = os.path.join(gen_dir, "views")
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    print("================================================================")
    print(f" Multi-View PixelDiT Inference Generation")
    print("================================================================")
    print(f"Exp Directory:    {exp_dir}")
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Mode:             {cfg.get('mode', 'rectified_flow')}")
    print(f"Modality:         {cfg.get('modality', 'normal')}")
    print(f"Num Views:        {cfg.get('num_views', 4)}")
    print(f"Resolution:       {cfg.get('resolution', 64)}x{cfg.get('resolution', 64)}")
    print(f"Full Class Mode:  {args.full_class}")
    print(f"Samples/Class:    {args.num_samples if args.full_class else 'N/A'}")
    print(f"Total Samples:    {10 * args.num_samples if args.full_class else args.num_samples}")
    print(f"ODE Steps:        {args.sample_steps}")
    print(f"Device:           {device}")
    print(f"Output Directory: {gen_dir}")
    print("================================================================")

    # Determine channel count
    modality = cfg.get("modality", "normal")
    if modality == "normal":
        in_channels = 3
    elif modality == "depth":
        in_channels = 1
    elif modality == "both":
        in_channels = 4
    else:
        raise ValueError(f"Unknown modality: {modality}")

    num_views = cfg.get("num_views", 4)
    resolution = cfg.get("resolution", 64)

    # Instantiate Model with trained hyperparameters
    model = MVPixelDiT(
        in_channels=in_channels,
        out_channels=in_channels,
        hidden_size=cfg.get("hidden_size", 512),
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
    print(f"\nLoading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
        print("Loaded EMA shadow model weights.")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights.")
    model.eval()

    # Build Flow Matching Inference Engine
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

    # Load Validation Dataset for Condition Inputs
    val_dataset = Digit3DMV(
        root=args.root,
        train=False,
        resolution=resolution,
        modality=modality,
        all_360=(num_views == 12),
        return_front_variation=False,
        return_c2w=False,
        use_zip=args.use_zip,
    )

    # Select Evaluation Sample Indices
    if args.full_class:
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
        print(f"Selected {len(selected_indices)} samples across all 10 digit classes ({args.num_samples} per class).")
    else:
        num_take = min(len(val_dataset), args.num_samples)
        selected_indices = list(range(num_take))
        print(f"Selected {num_take} random samples from validation set.")

    eval_subset = Subset(val_dataset, selected_indices)
    eval_loader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Generation Loop
    all_grid_rows: List[torch.Tensor] = []
    global_sample_counter = 0

    print("\nStarting Batched Flow Generation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            img_cond, x_gt = extract_batch_tensors(batch, modality=modality)
            img_cond = img_cond.to(device, non_blocking=True)
            x_gt = x_gt.to(device, non_blocking=True)
            B = img_cond.shape[0]

            # Generate via official Flow Matching ODE sampler
            if mode == "rectified_flow":
                gen_views = flow_model.sample(
                    batch_size=B,
                    steps=args.sample_steps,
                    img_cond=img_cond,
                )
            else:
                gen_views = flow_model.sample(
                    cond=img_cond,
                    steps=args.sample_steps,
                )

            # Unnormalize tensors to [0, 1]
            cond_01 = torch.clamp((img_cond + 1.0) * 0.5, 0.0, 1.0).cpu()
            gen_01 = torch.clamp((gen_views + 1.0) * 0.5, 0.0, 1.0).cpu()
            gt_01 = torch.clamp((x_gt + 1.0) * 0.5, 0.0, 1.0).cpu()

            labels = batch["label"]
            sample_names = batch["sample_name"]

            for i in range(B):
                lbl = int(labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i])
                s_name = sample_names[i]
                global_sample_counter += 1

                # 1. Build comparison grid row
                rows = build_comparison_row(cond_01[i], gen_01[i], gt_01[i], modality)
                all_grid_rows.extend(rows)

                # Save individual sample comparison grid
                sample_grid_tensor = torch.cat(rows, dim=0)
                cols = 1 + 2 * num_views
                sample_grid = make_grid(sample_grid_tensor, nrow=cols, padding=2, normalize=False)
                sample_grid_path = os.path.join(grids_dir, f"sample_{global_sample_counter:04d}_class_{lbl}_{s_name}.png")
                save_image(sample_grid, sample_grid_path)

                # 2. Save individual view image files
                if args.save_individual:
                    sample_views_folder = os.path.join(views_dir, f"class_{lbl}", s_name)
                    save_single_sample_views(
                        sample_views_folder,
                        cond_01=cond_01[i],
                        gen_01=gen_01[i],
                        gt_01=gt_01[i],
                        modality=modality,
                    )

            print(f"Generated batch [{batch_idx+1}/{len(eval_loader)}] ({global_sample_counter}/{len(selected_indices)} samples completed)")

    # Save comprehensive master grid
    if all_grid_rows:
        master_tensor = torch.cat(all_grid_rows, dim=0)
        cols = 1 + 2 * num_views
        master_grid = make_grid(master_tensor, nrow=cols, padding=2, normalize=False)
        master_grid_path = os.path.join(gen_dir, "all_samples_grid.png")
        save_image(master_grid, master_grid_path)
        print(f"\nSaved master comparison grid ({len(selected_indices)} samples) to:")
        print(f"  {master_grid_path}")

    print("================================================================")
    print(f" Generation Completed! Total Samples: {global_sample_counter}")
    print(f" Grids saved to:        {grids_dir}")
    print(f" Individual views in:  {views_dir}")
    print("================================================================")


if __name__ == "__main__":
    main()
