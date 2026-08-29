"""Main training script for Multi-View PixelDiT using Rectified Flow and Mean Flow matching.

This script manages end-to-end dataset loading, model initialization, Rectified Flow / Mean Flow
training using the official rectified_flow_pytorch API, Exponential Moving Average (EMA), multi-view
ODE validation sampling, config hashing, resume training, and per-iteration wall-clock timing / history tracking.
"""

from typing import Optional, Dict, Any, Tuple
import os
import sys
import math
import time
import copy
import json
import hashlib
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# Ensure local repository root is in path for local models import
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


class EMA:
    """Exponential Moving Average (EMA) shadow model tracker."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Multi-View PixelDiT Flow Matching Training Pipeline")

    # Flow & Modality Modes
    parser.add_argument("--mode", type=str, choices=["rectified_flow", "mean_flow"], default="rectified_flow", help="Flow matching training formulation ('rectified_flow' or 'mean_flow')")
    parser.add_argument("--modality", type=str, choices=["normal", "depth", "both"], default="normal", help="Dataset target modality ('normal', 'depth', or 'both')")
    parser.add_argument("--num_views", type=int, choices=[4, 12], default=4, help="Number of 360 multi-views to generate (4 or 12)")
    parser.add_argument("--resolution", type=int, choices=[64, 128], default=64, help="Image resolution for training (64 or 128)")
    parser.add_argument("--return_front_variation", action="store_true", default=False, help="Enable front view variation training augmentation")

    # Experiment & Output Paths
    parser.add_argument("--exp_name", type=str, default="", help="Experiment base name (default: [modality]_[num_views]_[mode][_front_variation])")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root directory for saving experiment logs and checkpoints")
    parser.add_argument("--resume", type=str, default="", nargs="?", const="latest", help="Resume training: path to checkpoint, or 'latest'/'best'/'auto' from save_dir")
    parser.add_argument("--root", type=str, default="~/.conquer3d/", help="Root cache directory containing Digit3DMV dataset archives")
    parser.add_argument("--use_zip", action="store_true", default=None, help="Force reading dataset from zip archive instead of uncompressed folder")

    # Optimization & Training Loop
    parser.add_argument("--max_iters", type=int, default=50000, help="Total training iterations/steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate for CosineAnnealingLR")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Linear learning rate warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA shadow weight decay coefficient")

    # Evaluation, Sampling & Logging
    parser.add_argument("--val_every", type=int, default=1000, help="Iterations interval between validation evaluations and image generation")
    parser.add_argument("--save_every", type=int, default=5000, help="Iterations interval between checkpoint saves")
    parser.add_argument("--log_every", type=int, default=50, help="Iterations interval between stdout logging")
    parser.add_argument("--sample_steps", type=int, default=25, help="ODE integration steps for validation sampling")
    parser.add_argument("--num_val_samples", type=int, default=4, help="Number of samples to visualize in validation grid")

    # Model Hyperparameters
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden feature dimension for patch transformer")
    parser.add_argument("--pixel_hidden_size", type=int, default=64, help="Pixel-level token feature dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--patch_depth", type=int, default=6, help="Number of patch AugmentedDiTBlocks")
    parser.add_argument("--pixel_depth", type=int, default=2, help="Number of pixel MVPiTBlocks")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch spatial extent (P)")
    parser.add_argument("--attn_func", type=str, choices=["torch"], default="torch", help="Attention implementation backend ('torch')")
    parser.add_argument("--cond_drop_prob", type=float, default=0.15, help="Classifier-free guidance dropout probability for image condition")
    parser.add_argument("--no_pixel_abs_pos", action="store_true", default=False, help="Disable 3D absolute sincos position embedding on pixel tokens")

    # Hardware & Concurrency
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocess worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device (cuda or cpu)")

    return parser.parse_args()


def extract_batch_tensors(batch: Dict[str, Any], modality: str, is_train: bool = True, return_front_variation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts condition front image and target multi-view tensors normalized to [-1, 1].

    If return_front_variation is True and is_train is True, randomly samples one of the 9 front view variations as condition.
    """
    if modality == "normal":
        x_clean = batch["360"]  # [B, V, 3, H, W]
        if return_front_variation and is_train and "front_variation" in batch:
            all_front = torch.cat([batch["front"].unsqueeze(1), batch["front_variation"]], dim=1)
            B = all_front.shape[0]
            rand_idx = torch.randint(0, 9, (B,), device=all_front.device)
            img_cond = all_front[torch.arange(B), rand_idx]  # [B, 3, H, W]
        else:
            img_cond = batch["front"]  # [B, 3, H, W]

    elif modality == "depth":
        x_clean = batch["360"]  # [B, V, 1, H, W]
        if return_front_variation and is_train and "front_variation" in batch:
            all_front = torch.cat([batch["front"].unsqueeze(1), batch["front_variation"]], dim=1)
            B = all_front.shape[0]
            rand_idx = torch.randint(0, 9, (B,), device=all_front.device)
            img_cond = all_front[torch.arange(B), rand_idx]
        else:
            img_cond = batch["front"]

    elif modality == "both":
        x_clean = torch.cat([batch["normal_360"], batch["depth_360"]], dim=2)  # [B, V, 4, H, W]
        if return_front_variation and is_train and "normal_front_variation" in batch:
            all_norm = torch.cat([batch["normal_front"].unsqueeze(1), batch["normal_front_variation"]], dim=1)
            all_depth = torch.cat([batch["depth_front"].unsqueeze(1), batch["depth_front_variation"]], dim=1)
            all_both = torch.cat([all_norm, all_depth], dim=2)  # [B, 9, 4, H, W]
            B = all_both.shape[0]
            rand_idx = torch.randint(0, 9, (B,), device=all_both.device)
            img_cond = all_both[torch.arange(B), rand_idx]
        else:
            img_cond = torch.cat([batch["normal_front"], batch["depth_front"]], dim=1)  # [B, 4, H, W]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Scale from [0, 1] to [-1, 1] for flow matching
    img_cond_norm = img_cond * 2.0 - 1.0
    x_clean_norm = x_clean * 2.0 - 1.0
    return img_cond_norm, x_clean_norm


def save_validation_grid(
    save_path: str,
    img_cond: torch.Tensor,
    gen_views: torch.Tensor,
    gt_views: torch.Tensor,
    modality: str,
    max_samples: int = 4,
):
    """Visualizes and saves side-by-side comparison grids of generated vs ground truth multi-views."""
    B, V, C, H, W = gen_views.shape
    num_show = min(B, max_samples)

    # Convert tensors back to [0, 1]
    cond_01 = torch.clamp((img_cond[:num_show] + 1.0) * 0.5, 0.0, 1.0)
    gen_01 = torch.clamp((gen_views[:num_show] + 1.0) * 0.5, 0.0, 1.0)
    gt_01 = torch.clamp((gt_views[:num_show] + 1.0) * 0.5, 0.0, 1.0)

    rows = []
    for i in range(num_show):
        if modality in ("normal", "depth"):
            c_img = cond_01[i]
            if c_img.shape[0] == 1:
                c_img = c_img.repeat(3, 1, 1)

            row_imgs = [c_img]
            for v in range(V):
                g = gen_01[i, v]
                if g.shape[0] == 1:
                    g = g.repeat(3, 1, 1)
                row_imgs.append(g)
            for v in range(V):
                t = gt_01[i, v]
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                row_imgs.append(t)

            row_tensor = torch.stack(row_imgs, dim=0)
            rows.append(row_tensor)
        elif modality == "both":
            # Normal Maps
            norm_cond = cond_01[i, :3]
            row_norm = [norm_cond]
            for v in range(V):
                row_norm.append(gen_01[i, v, :3])
            for v in range(V):
                row_norm.append(gt_01[i, v, :3])
            rows.append(torch.stack(row_norm, dim=0))

            # Depth Maps
            depth_cond = cond_01[i, 3:4].repeat(3, 1, 1)
            row_depth = [depth_cond]
            for v in range(V):
                row_depth.append(gen_01[i, v, 3:4].repeat(3, 1, 1))
            for v in range(V):
                row_depth.append(gt_01[i, v, 3:4].repeat(3, 1, 1))
            rows.append(torch.stack(row_depth, dim=0))

    all_grid_imgs = torch.cat(rows, dim=0)
    cols = 1 + 2 * V
    grid = make_grid(all_grid_imgs, nrow=cols, padding=2, normalize=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(grid, save_path)


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # Determine base experiment name
    exp_name = args.exp_name
    if not exp_name:
        exp_name = f"{args.modality}_{args.num_views}_{args.mode}"
        if args.return_front_variation:
            exp_name = f"{exp_name}_front_variation"

    # Compute deterministic config hash (8 chars) excluding runtime/progression keys
    config_dict = vars(args).copy()
    excluded_keys = {
        "resume", "output_dir", "num_workers", "device", "exp_name",
        "max_iters", "val_every", "save_every", "log_every",
        "sample_steps", "num_val_samples", "root", "use_zip"
    }
    hash_payload = {k: v for k, v in sorted(config_dict.items()) if k not in excluded_keys}
    config_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]

    # Save directory with config hash postfix
    save_dir_name = f"{exp_name}_{config_hash}"
    save_dir = os.path.join(args.output_dir, save_dir_name)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Save experiment config
    config_dict["config_hash"] = config_hash
    config_dict["save_dir"] = save_dir
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print("================================================================")
    print(f" Multi-View PixelDiT Training: {save_dir_name}")
    print("================================================================")
    print(f"Mode:                   {args.mode}")
    print(f"Modality:               {args.modality}")
    print(f"Num Views:              {args.num_views}")
    print(f"Resolution:             {args.resolution}x{args.resolution}")
    print(f"Front Variation Aug:    {args.return_front_variation}")
    print(f"Attention Function:     {args.attn_func}")
    print(f"Max Iterations:         {args.max_iters}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Config Hash:            {config_hash}")
    print(f"Device:                 {device}")
    print(f"Save Directory:         {save_dir}")
    print("================================================================")

    # Determine channel count
    if args.modality == "normal":
        in_channels = 3
    elif args.modality == "depth":
        in_channels = 1
    elif args.modality == "both":
        in_channels = 4
    else:
        raise ValueError(f"Unknown modality: {args.modality}")

    # Build Training and Validation Datasets
    all_360 = (args.num_views == 121)
    train_dataset = Digit3DMV(
        root=args.root,
        train=True,
        resolution=args.resolution,
        modality=args.modality,
        all_360=all_360,
        return_front_variation=args.return_front_variation,
        return_c2w=False,
        use_zip=args.use_zip,
        download=True
    )
    val_dataset = Digit3DMV(
        root=args.root,
        train=False,
        resolution=args.resolution,
        modality=args.modality,
        all_360=all_360,
        return_front_variation=False,
        return_c2w=False,
        use_zip=args.use_zip,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # Instantiate MVPixelDiT model
    model = MVPixelDiT(
        in_channels=in_channels,
        out_channels=in_channels,
        hidden_size=args.hidden_size,
        pixel_hidden_size=args.pixel_hidden_size,
        num_heads=args.num_heads,
        patch_depth=args.patch_depth,
        pixel_depth=args.pixel_depth,
        patch_size=args.patch_size,
        num_views=args.num_views,
        use_pixel_abs_pos=not args.no_pixel_abs_pos,
        cond_drop_prob=args.cond_drop_prob,
        attn_func=args.attn_func,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Build Flow Matching Engine using rectified_flow_pytorch API
    data_shape = (args.num_views, in_channels, args.resolution, args.resolution)
    if args.mode == "rectified_flow":
        flow_model = RectifiedFlow(
            model=model,
            time_cond_kwarg="t",
            data_shape=data_shape,
        )
    elif args.mode == "mean_flow":
        wrapped_model = MeanFlowModelWrapper(model)
        flow_model = MeanFlow(
            model=wrapped_model,
            data_shape=data_shape,
            accept_cond=True,
        ).to(device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Optimizer & PyTorch CosineAnnealingLR Scheduler with optional Linear warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_steps > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.max_iters - args.warmup_steps), eta_min=args.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_iters, eta_min=args.min_lr
        )

    # Initialize EMA Tracker
    ema = EMA(model, decay=args.ema_decay)

    # Training State & History
    history = {
        "train_loss": [],
        "val_loss": [],
        "iter_time_sec": [],
        "samples_per_sec": [],
        "lr": [],
        "steps": [],
    }
    history_path = os.path.join(save_dir, "history.json")
    start_step = 0
    best_val_loss = float("inf")

    # Handle Resume Training
    if args.resume:
        resume_target = args.resume.strip()
        if os.path.isfile(resume_target):
            resume_path = resume_target
        elif resume_target.lower() in ("latest", "best", "auto"):
            target_name = "model_latest.pt" if resume_target.lower() in ("latest", "auto") else "model_best.pt"
            resume_path = os.path.join(ckpt_dir, target_name)
        else:
            candidate = os.path.join(ckpt_dir, resume_target)
            if os.path.isfile(candidate):
                resume_path = candidate
            else:
                resume_path = resume_target

        if os.path.isfile(resume_path):
            print(f"\n---> Loading checkpoint for resume: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "ema_state_dict" in checkpoint:
                ema.shadow.load_state_dict(checkpoint["ema_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_step = checkpoint.get("step", 0)
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            print(f"Resumed successfully at step {start_step} | Best val loss: {best_val_loss:.5f}")

            # Restore existing history if present
            if os.path.isfile(history_path):
                try:
                    with open(history_path, "r") as f:
                        history = json.load(f)
                    print(f"Restored history log with {len(history['steps'])} recorded steps.")
                except Exception as e:
                    print(f"Warning: Failed to load previous history: {e}")
        else:
            if resume_target.lower() == "auto":
                print(f"Auto-resume: No existing checkpoint at {resume_path}, starting fresh.")
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {resume_path}")

    train_iter = iter(train_loader)
    start_time = time.time()

    if start_step >= args.max_iters:
        print(f"Target max_iters ({args.max_iters}) already reached at resumed step {start_step}. Exiting.")
        return

    print(f"\nStarting Training Loop from Step {start_step + 1} to {args.max_iters}...")
    for step in range(start_step + 1, args.max_iters + 1):
        step_start_time = time.time()
        model.train()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img_cond, x_clean = extract_batch_tensors(
            batch,
            modality=args.modality,
            is_train=True,
            return_front_variation=args.return_front_variation,
        )
        img_cond = img_cond.to(device, non_blocking=True)
        x_clean = x_clean.to(device, non_blocking=True)

        optimizer.zero_grad()

        if args.mode == "rectified_flow":
            loss = flow_model(x_clean, img_cond=img_cond)
        elif args.mode == "mean_flow":
            loss = flow_model(x_clean, cond=img_cond)

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        ema.update(model)

        step_duration = time.time() - step_start_time
        samples_per_sec = args.batch_size / max(step_duration, 1e-6)

        # Record metrics
        current_lr = scheduler.get_last_lr()[0]
        history["steps"].append(step)
        history["train_loss"].append(float(loss.item()))
        history["iter_time_sec"].append(float(step_duration))
        history["samples_per_sec"].append(float(samples_per_sec))
        history["lr"].append(float(current_lr))

        # Console logging
        if step % args.log_every == 0 or step == 1 or step == start_step + 1:
            elapsed = time.time() - start_time
            eta_sec = (args.max_iters - step) * step_duration
            print(
                f"[{step:06d}/{args.max_iters:06d}] "
                f"Loss: {loss.item():.5f} | "
                f"Time/iter: {step_duration*1000.0:.1f}ms ({samples_per_sec:.1f} samples/s) | "
                f"LR: {current_lr:.2e} | "
                f"ETA: {eta_sec/60.0:.1f}m"
            )

        # Validation & ODE Sampling
        if step % args.val_every == 0 or step == args.max_iters:
            print(f"\n---> Running Validation & ODE Sampling at Step {step}...")
            val_loss_list = []
            val_batch_sample = None

            # Build temporary evaluation flow model with EMA shadow weights
            if args.mode == "rectified_flow":
                eval_flow = RectifiedFlow(
                    model=ema.shadow,
                    time_cond_kwarg="t",
                    data_shape=data_shape,
                )
            else:
                eval_wrapped = MeanFlowModelWrapper(ema.shadow)
                eval_flow = MeanFlow(
                    model=eval_wrapped,
                    data_shape=data_shape,
                    accept_cond=True,
                ).to(device)

            eval_flow.eval()
            with torch.no_grad():
                for v_idx, v_batch in enumerate(val_loader):
                    v_cond, v_clean = extract_batch_tensors(
                        v_batch,
                        modality=args.modality,
                        is_train=False,
                        return_front_variation=False,
                    )
                    v_cond = v_cond.to(device, non_blocking=True)
                    v_clean = v_clean.to(device, non_blocking=True)

                    if args.mode == "rectified_flow":
                        v_loss = eval_flow(v_clean, img_cond=v_cond)
                    else:
                        v_loss = eval_flow(v_clean, cond=v_cond)
                    val_loss_list.append(v_loss.item())

                    if val_batch_sample is None:
                        val_batch_sample = (v_cond, v_clean)
                    if v_idx >= 50:  # Validate over 50 mini-batches
                        break

            avg_val_loss = sum(val_loss_list) / max(len(val_loss_list), 1)
            history["val_loss"].append({"step": step, "loss": float(avg_val_loss)})
            print(f"Validation Loss: {avg_val_loss:.5f}")

            # Generate and save comparison image grid
            v_cond, v_clean = val_batch_sample
            num_val = min(len(v_cond), args.num_val_samples)
            if args.mode == "rectified_flow":
                gen_views = eval_flow.sample(
                    batch_size=num_val,
                    steps=args.sample_steps,
                    img_cond=v_cond[:num_val],
                )
            else:
                gen_views = eval_flow.sample(
                    cond=v_cond[:num_val],
                    steps=args.sample_steps,
                )

            sample_img_path = os.path.join(sample_dir, f"sample_step_{step:06d}.png")
            save_validation_grid(
                sample_img_path,
                img_cond=v_cond[:num_val],
                gen_views=gen_views,
                gt_views=v_clean[:num_val],
                modality=args.modality,
                max_samples=num_val,
            )
            print(f"Saved visual sample grid to {sample_img_path}")

            # Checkpoint best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(ckpt_dir, "model_best.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.shadow.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": config_dict,
                }, best_ckpt_path)
                print(f"Saved new BEST checkpoint to {best_ckpt_path}")

        # Regular Checkpointing
        if step % args.save_every == 0 or step == args.max_iters:
            latest_ckpt_path = os.path.join(ckpt_dir, "model_latest.pt")
            step_ckpt_path = os.path.join(ckpt_dir, f"model_step_{step:06d}.pt")
            ckpt_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.shadow.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "config": config_dict,
            }
            torch.save(ckpt_data, latest_ckpt_path)
            torch.save(ckpt_data, step_ckpt_path)

            # Persist history JSON
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)
            print(f"Saved checkpoint and training history at step {step}")

    # Final history dump
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    total_training_time = time.time() - start_time
    print("\n================================================================")
    print(f" Training Completed in {total_training_time/60.0:.2f} minutes!")
    print(f" Checkpoints & History saved to: {save_dir}")
    print("================================================================")


if __name__ == "__main__":
    main()
