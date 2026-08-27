"""High-quality GIF generation script for Multi-View PixelDiT training evolution.

This script merges validation sample grids from `samples/` into a smooth animated GIF.
When `--show_loss` is enabled, each frame renders a synchronized, dark-themed
dynamic loss progression curve (Train EMA Loss & Val Loss) alongside real-time
stat badges (Current Step, Losses, LR) and column annotations with crystal-clear typography.
When `--show_loss` is False, it outputs pure sample figures without any loss values.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import sys
import re
import json
import glob
import argparse
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Loads a high-quality system TrueType font with fallback."""
    font_paths = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for GIF synthesis."""
    parser = argparse.ArgumentParser(description="Generate Training Progress GIF with Dynamic Loss Visualization")

    parser.add_argument("--exp_dir", type=str, required=True, help="Path or name of the experiment directory (e.g. 'outputs/normal_4_rectified_flow_...').")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Root outputs directory if exp_dir is a relative folder name.")
    parser.add_argument("--output_name", type=str, default=None, help="Output GIF filename (default: 'training_evolution_with_loss.gif' if show_loss else 'training_evolution.gif').")
    parser.add_argument("--fps", type=float, default=8.0, help="Frames per second for the animation.")
    parser.add_argument("--duration", type=int, default=None, help="Frame duration in milliseconds (overrides --fps if specified).")
    parser.add_argument("--show_loss", action="store_true", default=False, help="Render an upper dynamic training & validation loss progression figure.")
    parser.add_argument("--flip_vertical", action="store_true", default=False, help="Flip sample images vertically (for legacy inverted coordinate runs).")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to include (subsamples evenly if more exist).")
    parser.add_argument("--width", type=int, default=880, help="Target width for the composite animation frames.")
    parser.add_argument("--loop", type=int, default=0, help="GIF loop count (0 = infinite).")

    return parser.parse_args()


def extract_step_from_filename(filename: str) -> int:
    """Extracts integer step number from sample filename (e.g. sample_step_010000.png -> 10000)."""
    match = re.search(r"step_(\d+)", os.path.basename(filename))
    return int(match.group(1)) if match else 0


def smooth_curve(values: List[float], weight: float = 0.9) -> List[float]:
    """Computes Exponential Moving Average (EMA) smoothing for training loss curve."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for val in values:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return smoothed


def render_loss_plot(
    history: Dict[str, Any],
    current_step: int,
    total_steps: int,
    y_min: float,
    y_max: float,
    plot_width_px: int = 880,
    plot_height_px: int = 300,
) -> np.ndarray:
    """Renders a modern dark-themed loss curve figure with clear high-contrast typography."""
    dpi = 100
    figsize = (plot_width_px / dpi, plot_height_px / dpi)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Styling Palette
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    all_steps = history.get("steps", [])
    all_train_loss = history.get("train_loss", [])
    val_loss_entries = history.get("val_loss", [])

    # Filter data up to current_step
    train_mask = [s <= current_step for s in all_steps]
    sub_train_steps = [s for s, m in zip(all_steps, train_mask) if m]
    sub_train_loss = [l for l, m in zip(all_train_loss, train_mask) if m]

    sub_val = [v for v in val_loss_entries if v.get("step", 0) <= current_step]
    sub_val_steps = [v["step"] for v in sub_val]
    sub_val_loss = [v["loss"] for v in sub_val]

    # Plot Training Loss
    if sub_train_steps and sub_train_loss:
        # Subsample for fast smooth plotting if dense
        step_stride = max(1, len(sub_train_steps) // 600)
        p_steps = sub_train_steps[::step_stride]
        p_loss = sub_train_loss[::step_stride]
        p_smooth = smooth_curve(p_loss, weight=0.92)

        # Raw faint scatter / line
        ax.plot(p_steps, p_loss, color="#00D2FF", alpha=0.18, linewidth=0.8, label="Train Loss (Raw)")
        # Smooth EMA Line
        ax.plot(p_steps, p_smooth, color="#00F0FF", linewidth=2.4, label="Train Loss (EMA)")
        ax.fill_between(p_steps, p_smooth, y_min, color="#00F0FF", alpha=0.08)

    # Plot Validation Loss
    if sub_val_steps and sub_val_loss:
        ax.plot(
            sub_val_steps,
            sub_val_loss,
            color="#FF3366",
            linewidth=2.8,
            marker="o",
            markersize=6.5,
            markerfacecolor="#FF5588",
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.2,
            label="Validation Loss (EMA)",
            zorder=4,
        )

        # Pulsating glowing ring on latest validation point
        latest_val_step = sub_val_steps[-1]
        latest_val_l = sub_val_loss[-1]
        ax.scatter([latest_val_step], [latest_val_l], s=160, facecolor="none", edgecolor="#FF3366", linewidth=2.8, alpha=0.9, zorder=5)

    # Fixed Axis Scaling to eliminate animation jitter
    ax.set_xlim(0, max(total_steps, 100))
    ax.set_ylim(y_min, y_max)

    # Grid & Spines
    ax.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
    for spine in ax.spines.values():
        spine.set_color("#30363D")
        spine.set_linewidth(1.2)

    # Large Crisp Ticks & Labels
    ax.tick_params(colors="#E6EDF3", labelsize=10.5, direction="out", length=4, width=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Iteration Step", color="#F0F6FC", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel("Flow Loss", color="#F0F6FC", fontsize=11.5, fontweight="bold", labelpad=6)

    # High-contrast Legend
    ax.legend(
        loc="upper right",
        framealpha=0.6,
        facecolor="#0D1117",
        edgecolor="#30363D",
        fontsize=10.0,
        labelcolor="#FFFFFF",
    )

    plt.tight_layout(pad=1.0)
    fig.canvas.draw()

    # Extract RGB array from figure buffer
    img_rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return img_rgba[:, :, :3]


def create_composite_frame_with_loss(
    sample_img: Image.Image,
    plot_arr: np.ndarray,
    current_step: int,
    total_steps: int,
    cur_train_loss: Optional[float],
    cur_val_loss: Optional[float],
    cur_lr: Optional[float],
    exp_name: str,
    target_width: int = 880,
    num_views: int = 4,
) -> Image.Image:
    """Composites the header banner, stats cards, loss plot, and sample grid into a single frame."""
    # Scale sample grid to target width
    s_w, s_h = sample_img.size
    scaled_s_h = int(s_h * (target_width / s_w))
    sample_resized = sample_img.resize((target_width, scaled_s_h), Image.Resampling.LANCZOS)

    header_height = 86
    labels_height = 28
    plot_height = plot_arr.shape[0]
    total_height = header_height + plot_height + labels_height + scaled_s_h + 16

    # Create dark canvas
    canvas = Image.new("RGB", (target_width, total_height), color=(13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    f_title = get_font(17, bold=True)
    f_badge = get_font(12, bold=True)
    f_col = get_font(12, bold=True)

    # Top Header Background
    draw.rectangle([(0, 0), (target_width, header_height)], fill=(22, 27, 34))
    draw.line([(0, header_height), (target_width, header_height)], fill=(48, 54, 61), width=1)

    # Header Title
    title_text = f"Multi-View PixelDiT — {exp_name}"
    draw.text((16, 12), title_text, fill=(255, 255, 255), font=f_title)

    # Stat Badges in Header
    progress_pct = (current_step / max(total_steps, 1)) * 100.0
    step_str = f"STEP: {current_step:,} / {total_steps:,} ({progress_pct:.1f}%)"
    train_str = f"TRAIN LOSS: {cur_train_loss:.4f}" if cur_train_loss is not None else "TRAIN LOSS: --"
    val_str = f"VAL LOSS: {cur_val_loss:.4f}" if cur_val_loss is not None else "VAL LOSS: --"
    lr_str = f"LR: {cur_lr:.1e}" if cur_lr is not None else "LR: --"

    badge_y = 44
    badges = [
        (step_str, (0, 240, 255), (0, 45, 65)),
        (train_str, (120, 230, 255), (20, 40, 55)),
        (val_str, (255, 110, 150), (70, 20, 40)),
        (lr_str, (210, 215, 230), (40, 42, 52)),
    ]

    bx = 16
    for b_text, text_col, bg_col in badges:
        # Measure text width
        bbox = draw.textbbox((0, 0), b_text, font=f_badge)
        tw = bbox[2] - bbox[0]
        bw = tw + 20
        draw.rounded_rectangle([(bx, badge_y), (bx + bw, badge_y + 28)], radius=5, fill=bg_col, outline=(60, 70, 90), width=1)
        draw.text((bx + 10, badge_y + 6), b_text, fill=text_col, font=f_badge)
        bx += bw + 12

    # Paste Loss Plot
    current_y = header_height + 4
    plot_img = Image.fromarray(plot_arr)
    canvas.paste(plot_img, (0, current_y))
    current_y += plot_height + 4

    # Section Labels over Sample Grid
    draw.rectangle([(0, current_y), (target_width, current_y + labels_height)], fill=(22, 27, 34))
    draw.line([(0, current_y), (target_width, current_y)], fill=(48, 54, 61), width=1)
    draw.line([(0, current_y + labels_height), (target_width, current_y + labels_height)], fill=(48, 54, 61), width=1)

    col_span = 1 + 2 * num_views
    col_w = target_width / col_span

    # [Front Input] label
    draw.text((10, current_y + 6), "Front Condition", fill=(0, 229, 255), font=f_col)

    # [Generated Multi-Views] label
    gen_start_x = int(col_w * 1) + 10
    draw.text((gen_start_x, current_y + 6), f"Generated 360° Multi-Views ({num_views} Views)", fill=(255, 215, 0), font=f_col)

    # [Ground Truth 360°] label
    gt_start_x = int(col_w * (1 + num_views)) + 10
    draw.text((gt_start_x, current_y + 6), f"Ground Truth 360° ({num_views} Views)", fill=(80, 220, 130), font=f_col)

    current_y += labels_height + 4

    # Paste Sample Image
    canvas.paste(sample_resized, (0, current_y))

    return canvas


def create_clean_sample_frame(
    sample_img: Image.Image,
    target_width: int = 880,
    num_views: int = 4,
) -> Image.Image:
    """Creates a clean sample figure frame with column headers and NO loss values or numbers."""
    s_w, s_h = sample_img.size
    scaled_s_h = int(s_h * (target_width / s_w))
    sample_resized = sample_img.resize((target_width, scaled_s_h), Image.Resampling.LANCZOS)

    labels_height = 28
    total_height = labels_height + scaled_s_h + 8

    canvas = Image.new("RGB", (target_width, total_height), color=(13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    f_col = get_font(12, bold=True)

    # Section Labels over Sample Grid
    draw.rectangle([(0, 0), (target_width, labels_height)], fill=(22, 27, 34))
    draw.line([(0, labels_height), (target_width, labels_height)], fill=(48, 54, 61), width=1)

    col_span = 1 + 2 * num_views
    col_w = target_width / col_span

    # [Front Input] label
    draw.text((10, 6), "Front Condition", fill=(0, 229, 255), font=f_col)

    # [Generated Multi-Views] label
    gen_start_x = int(col_w * 1) + 10
    draw.text((gen_start_x, 6), f"Generated 360° Multi-Views ({num_views} Views)", fill=(255, 215, 0), font=f_col)

    # [Ground Truth 360°] label
    gt_start_x = int(col_w * (1 + num_views)) + 10
    draw.text((gt_start_x, 6), f"Ground Truth 360° ({num_views} Views)", fill=(80, 220, 130), font=f_col)

    # Paste Sample Image
    canvas.paste(sample_resized, (0, labels_height + 4))

    return canvas


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

    samples_dir = os.path.join(exp_dir, "samples")
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory 'samples/' not found in: {exp_dir}")

    # Gather and sort all sample png images by step
    sample_files = sorted(glob.glob(os.path.join(samples_dir, "sample_step_*.png")))
    if not sample_files:
        raise FileNotFoundError(f"No 'sample_step_*.png' images found in: {samples_dir}")

    # Output GIF filename: distinct filenames for show_loss=True vs show_loss=False
    if args.output_name:
        gif_filename = args.output_name
    else:
        gif_filename = "training_evolution_with_loss.gif" if args.show_loss else "training_evolution.gif"

    output_gif_path = os.path.join(exp_dir, gif_filename)

    print("================================================================")
    print(f" Multi-View PixelDiT Training Progress GIF Generator")
    print("================================================================")
    print(f"Exp Directory:    {exp_dir}")
    print(f"Samples Found:    {len(sample_files)} sample grids")
    print(f"Show Loss Curve:  {args.show_loss}")
    print(f"Flip Vertical:    {args.flip_vertical}")
    print(f"FPS:              {args.fps}")
    print(f"Output Width:     {args.width}px")
    print(f"Output Target:    {output_gif_path}")
    print("================================================================")

    # Read config.json
    config_path = os.path.join(exp_dir, "config.json")
    cfg = {}
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)

    exp_name = os.path.basename(exp_dir.rstrip("/\\"))
    total_max_steps = cfg.get("max_iters", 50000)
    num_views = cfg.get("num_views", 4)

    # Read history.json
    history = {}
    history_path = os.path.join(exp_dir, "history.json")
    if os.path.isfile(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    # Precompute global loss bounds for consistent axis scaling across frames
    y_min, y_max = 0.0, 10.0
    if history:
        train_losses = history.get("train_loss", [])
        val_losses = [v["loss"] for v in history.get("val_loss", [])]
        all_l = train_losses + val_losses
        if all_l:
            # Filter out extreme outliers / NaNs for scale calculation
            valid_l = [x for x in all_l if not np.isnan(x) and not np.isinf(x)]
            if valid_l:
                y_min = max(0.0, float(np.percentile(valid_l, 1)) * 0.85)
                y_max = float(np.percentile(valid_l, 99)) * 1.15

    # Subsample frames if max_frames specified
    if args.max_frames and len(sample_files) > args.max_frames:
        indices = np.linspace(0, len(sample_files) - 1, args.max_frames, dtype=int)
        sample_files = [sample_files[i] for i in indices]
        print(f"Subsampled to {len(sample_files)} frames (--max_frames {args.max_frames})")

    # Frame Synthesis Loop
    frames: List[Image.Image] = []
    plot_height = 280 if args.show_loss else 0

    print("\nSynthesizing animation frames...")
    for idx, s_path in enumerate(sample_files):
        step = extract_step_from_filename(s_path)
        with Image.open(s_path) as s_img:
            s_img = s_img.convert("RGB")
            if args.flip_vertical:
                s_img = s_img.transpose(Image.FLIP_TOP_BOTTOM)

            if args.show_loss:
                # Find matching metrics at this step
                cur_train_loss = None
                cur_val_loss = None
                cur_lr = None

                if history:
                    all_steps = history.get("steps", [])
                    if all_steps:
                        t_idx = min(range(len(all_steps)), key=lambda i: abs(all_steps[i] - step))
                        if t_idx < len(history.get("train_loss", [])):
                            cur_train_loss = history["train_loss"][t_idx]
                        if t_idx < len(history.get("lr", [])):
                            cur_lr = history["lr"][t_idx]

                    for v in history.get("val_loss", []):
                        if v.get("step") == step:
                            cur_val_loss = v.get("loss")
                            break

                plot_arr = render_loss_plot(
                    history=history,
                    current_step=step,
                    total_steps=total_max_steps,
                    y_min=y_min,
                    y_max=y_max,
                    plot_width_px=args.width,
                    plot_height_px=plot_height,
                )

                composite = create_composite_frame_with_loss(
                    sample_img=s_img,
                    plot_arr=plot_arr,
                    current_step=step,
                    total_steps=total_max_steps,
                    cur_train_loss=cur_train_loss,
                    cur_val_loss=cur_val_loss,
                    cur_lr=cur_lr,
                    exp_name=exp_name,
                    target_width=args.width,
                    num_views=num_views,
                )
            else:
                # Pure sample figure with NO loss values
                composite = create_clean_sample_frame(
                    sample_img=s_img,
                    target_width=args.width,
                    num_views=num_views,
                )

            frames.append(composite)

        if (idx + 1) % 10 == 0 or idx == len(sample_files) - 1:
            print(f"  Rendered [{idx+1}/{len(sample_files)}] frames (Step {step:,})")

    # Frame duration in milliseconds
    duration_ms = int(args.duration) if args.duration else int(1000.0 / max(args.fps, 0.1))

    print(f"\nWriting GIF to {output_gif_path} (Duration: {duration_ms}ms/frame, {len(frames)} frames)...")

    # Quantize each frame to 256 colors for clean, compact animated GIF
    quantized_frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for f in frames]
    quantized_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=args.loop,
        optimize=True,
    )

    gif_size_mb = os.path.getsize(output_gif_path) / (1024 * 1024)
    print("================================================================")
    print(f" GIF Successfully Generated!")
    print(f" Location:  {output_gif_path}")
    print(f" File Size: {gif_size_mb:.2f} MB ({len(frames)} frames)")
    print("================================================================")


if __name__ == "__main__":
    main()
