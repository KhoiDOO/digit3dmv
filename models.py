"""MVPixelDiT: Multi-View Diffusion Transformer with Dual-Level Pixel and Patch Space.

This module implements a multi-view generation transformer architecture adapted from
PixelDiT, operating across both patch space (semantic global self-attention with 3D-RoPE)
and pixel space (high-frequency local detail refinement with PiTBlocks) for simultaneous
multi-view 3D synthesis.
"""

import math
from typing import Optional, Tuple, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


def apply_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies Adaptive LayerNorm (AdaLN) modulation to input tokens."""
    if shift.ndim == 2 and x.ndim == 3:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def get_3d_sincos_pos_embed(embed_dim: int, num_view: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    """Generates 3D sinusoidal position embeddings for (view, height, width) grid."""
    grid_v = np.arange(num_view, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_v, grid_h, grid_w, indexing='ij')  # (3, num_view, grid_size, grid_size)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, num_view, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Wan2.1 3D Spatio-Temporal dimension partitioning for 3D sinusoidal embeddings."""
    assert embed_dim % 2 == 0
    dim_v = embed_dim - 4 * (embed_dim // 6)
    dim_h = 2 * (embed_dim // 6)
    dim_w = 2 * (embed_dim // 6)
    assert dim_v + dim_h + dim_w == embed_dim
    assert dim_v % 2 == 0 and dim_h % 2 == 0 and dim_w % 2 == 0

    emb_v = get_1d_sincos_pos_embed_from_grid(dim_v, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid[2])
    
    emb = np.concatenate([emb_v, emb_h, emb_w], axis=1)  # (M, embed_dim)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generates 1D sinusoidal position embeddings from 1D position coordinates."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def precompute_freqs_cis_3d(dim: int, num_views: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0) -> torch.Tensor:
    """Precomputes 3D Spatio-Temporal Rotary Position Embedding (3D-RoPE) complex frequencies.

    Args:
        dim: Attention head dimension (e.g. hidden_size // num_heads).
        num_views: Number of camera views (V).
        height: Spatial height in patches (Hs).
        width: Spatial width in patches (Ws).
        theta: Base frequency period. Defaults to 10000.0.
        scale: Coordinate scaling factor. Defaults to 16.0.

    Returns:
        torch.Tensor: Complex polar tensor of shape [V * H * W, dim // 2].
    """
    assert dim % 2 == 0
    d_v = dim - 4 * (dim // 6)
    d_h = 2 * (dim // 6)
    d_w = 2 * (dim // 6)
    assert d_v + d_h + d_w == dim

    v_pos = torch.arange(num_views, dtype=torch.float32)
    freqs_v = 1.0 / (theta ** (torch.arange(0, d_v, 2, dtype=torch.float32) / d_v))
    v_freqs = torch.outer(v_pos, freqs_v)
    v_cis = torch.polar(torch.ones_like(v_freqs), v_freqs)

    h_pos = torch.linspace(0, scale, height, dtype=torch.float32)
    freqs_h = 1.0 / (theta ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
    h_freqs = torch.outer(h_pos, freqs_h)
    h_cis = torch.polar(torch.ones_like(h_freqs), h_freqs)

    w_pos = torch.linspace(0, scale, width, dtype=torch.float32)
    freqs_w = 1.0 / (theta ** (torch.arange(0, d_w, 2, dtype=torch.float32) / d_w))
    w_freqs = torch.outer(w_pos, freqs_w)
    w_cis = torch.polar(torch.ones_like(w_freqs), w_freqs)

    v_exp = v_cis.view(num_views, 1, 1, -1).expand(num_views, height, width, -1)
    h_exp = h_cis.view(1, height, 1, -1).expand(num_views, height, width, -1)
    w_exp = w_cis.view(1, 1, width, -1).expand(num_views, height, width, -1)

    freqs_cis = torch.cat([v_exp, h_exp, w_exp], dim=-1)
    freqs_cis = freqs_cis.reshape(num_views * height * width, dim // 2)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies complex rotary position embeddings to query and key tensors."""
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis[None, None, :, :]  # [1, 1, N, D/2]
    elif freqs_cis.ndim == 3:
        freqs_cis = freqs_cis[:, None, :, :]     # [B, 1, N, D/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class TimestepConditioner(nn.Module):
    """Embeds scalar diffusion timesteps into continuous hidden vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-Head Self-Attention supporting 3D Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rotary_emb: bool = False,
        attn_func: str = "torch",
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rotary_emb = use_rotary_emb

        if attn_func == "torch":
            self.attn_func = scaled_dot_product_attention
        else:
            raise ValueError(f"Unsupported attention function: {attn_func}")

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rotary_emb and pos is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_output = self.attn_func(q, k, v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class MLP(nn.Module):
    """Standard 2-Layer GELU MLP."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class FinalLayer(nn.Module):
    """Final projection layer projecting pixel tokens back to output image channels."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class ResidualConvBlock(nn.Module):
    """Residual 2D Convolutional Block with GroupNorm and GELU."""

    def __init__(self, in_channels: int, num_groups: int = 8, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        groups = min(in_channels, num_groups)
        while in_channels % groups != 0:
            groups -= 1
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.bn1 = nn.GroupNorm(groups, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.bn2 = nn.GroupNorm(groups, in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act(x)


class ImageEncoder(nn.Module):
    """Encodes condition images (e.g. canonical front normal/depth) with CFG dropout."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256, depth: int = 3, cond_drop_prob: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.cond_drop_prob = cond_drop_prob

        layers = []
        curr_channels = in_channels
        for i in range(depth):
            out_channels = embed_dim // (2 ** (depth - i - 1))
            downsample = nn.Conv2d(curr_channels, out_channels, kernel_size=3, stride=2, padding=1)
            layers.append(downsample)
            block = ResidualConvBlock(out_channels)
            layers.append(block)
            curr_channels = out_channels

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dim, embed_dim)

        self.null_embed = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.null_embed, std=0.02)

    def forward(self, img: Optional[torch.Tensor], batch_size: Optional[int] = None) -> torch.Tensor:
        if img is not None:
            feat = self.net(img)
            emb = self.head(self.pool(feat).flatten(1))  # [B, embed_dim]
        else:
            assert batch_size is not None, "Must provide batch_size when img is None"
            emb = self.null_embed.expand(batch_size, -1)  # [B, embed_dim]

        if self.training and self.cond_drop_prob > 0.0:
            mask = torch.rand(len(emb), device=emb.device) < self.cond_drop_prob
            if mask.any():
                null_expanded = self.null_embed.expand(len(emb), -1)
                emb = torch.where(mask[:, None], null_expanded, emb)

        return emb


class PatchTokenEmbedder(nn.Module):
    """Linear projection embedder for unfolded patch tokens."""

    def __init__(self, in_chans: int = 12, embed_dim: int = 256, norm_layer=None, bias: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class AugmentedDiTBlock(nn.Module):
    """Patch-level Diffusion Transformer block with AdaLN modulation and 3D-RoPE."""

    def __init__(self, hidden_size: int, groups: int = 8, mlp_ratio: float = 4.0, adaLN_modulation=None, use_rotary_emb: bool = True, attn_func: str = "torch"):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False, use_rotary_emb=use_rotary_emb, attn_func=attn_func)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if gate_msa.ndim == 2 and x.ndim == 3:
            gate_msa = gate_msa.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos=pos, mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MVPixelTokenEmbedder(nn.Module):
    """Embeds multi-view image pixels into patch-grouped pixel tokens with 3D positional embedding."""

    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_image(self, num_views: int, height: int, width: int, device, dtype):
        key = ("image", num_views, height, width)
        if key in self._pos_cache:
            pe = self._pos_cache[key]
            return pe.to(device=device, dtype=dtype)
        pos_np = get_3d_sincos_pos_embed(self.hidden_size_output, num_views, height)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None) -> torch.Tensor:
        """
        Args:
            inputs: Multi-view image tensor [B, V, C, H, W]
        Returns:
            torch.Tensor: Grouped pixel tokens [(B * V * Hs * Ws), P^2, hidden_size_output]
        """
        if inputs.dim() != 5:
            raise ValueError("MVPixelTokenEmbedder expects inputs of shape [B, V, C, H, W]")
        assert img_height is not None and img_width is not None and patch_size is not None
        B, V, C, H, W = inputs.shape
        assert H == img_height and W == img_width
        assert (H % patch_size == 0) and (W % patch_size == 0)
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size

        x = inputs.permute(0, 1, 3, 4, 2).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(V, H, W, inputs.device, inputs.dtype)
            pos_full = pos_full.view(V, H, W, self.hidden_size_output)
            x = x + pos_full.unsqueeze(0)
        x = x.view(B, V, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B * V * Hs * Ws, P2, self.hidden_size_output)
        return x


class MVPiTBlock(nn.Module):
    """Pixel-level Transformer (PiT) Block for Multi-View Generation with 3D-RoPE."""

    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        attn_func: str = "torch",
        rope_fn=None,
        adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        assert (
            self.attn_dim % self.num_heads == 0
        ), "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = Attention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False, use_rotary_emb=True, attn_func=attn_func)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaln_post_modulation = bool(adaln_post_modulation)
        n_mod = 4 if self.adaln_post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, n_mod * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_3d

    def _fetch_pos(self, num_views: int, height: int, width: int, device):
        key = (num_views, height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, num_views, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        num_views: int,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        BVL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"MVPiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        VL = num_views * L
        B = BVL // VL

        n_mod = 4 if self.adaln_post_modulation else 6
        cond_params = self.adaLN_modulation(s_cond).view(BVL, P2, n_mod * self.pixel_dim)
        if self.adaln_post_modulation:
            scale1, shift1, scale2, shift2 = torch.chunk(cond_params, 4, dim=-1)
            x_norm = self.norm1(x)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
            x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)

        x_flat = x_norm.view(BVL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, VL, self.attn_dim)
        pos_comp = self._fetch_pos(num_views, Hs, Ws, x.device)
        attn_out = self.attn(x_comp, pos=pos_comp, mask=mask)
        attn_flat = self.expand_from_attn(attn_out.view(B * VL, self.attn_dim))
        attn_exp = attn_flat.view(BVL, P2, self.pixel_dim)

        if self.adaln_post_modulation:
            x = x + attn_exp * (1 + scale1) + shift1
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out * (1 + scale2) + shift2
        else:
            x = x + gate_msa * attn_exp
            mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp * mlp_out
        return x


class MVPixelDiT(nn.Module):
    """Multi-View PixelDiT: Dual-Level Pixel and Patch Space Diffusion Transformer."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 256,
        pixel_hidden_size: int = 64,
        num_heads: int = 8,
        patch_depth: int = 6,
        pixel_depth: int = 2,
        patch_size: int = 2,
        num_views: int = 4,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
        cond_drop_prob: float = 0.15,
        attn_func: str = "torch",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_heads = int(num_heads)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.num_views = int(num_views)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.attn_func = attn_func

        # Multi-view pixel and patch embedders
        self.pixel_embedder = MVPixelTokenEmbedder(self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(self.in_channels * (self.patch_size ** 2), self.hidden_size, bias=True)

        # Conditioning embedders
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.img_embedder = ImageEncoder(in_channels=self.in_channels, embed_dim=self.hidden_size, depth=3, cond_drop_prob=cond_drop_prob)

        # Transformer blocks
        self.patch_blocks = nn.ModuleList([
            AugmentedDiTBlock(self.hidden_size, self.num_heads, use_rotary_emb=True, attn_func=self.attn_func)
            for _ in range(self.patch_depth)
        ])
        self.pixel_blocks = nn.ModuleList([
            MVPiTBlock(
                self.pixel_hidden_size,
                self.hidden_size,
                patch_size=self.patch_size,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                attn_func=self.attn_func,
                adaln_post_modulation=self.pit_adaln_post_modulation,
            )
            for _ in range(self.pixel_depth)
        ])

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)
        self.precompute_pos = dict()
        self.initialize_weights()

    def fetch_pos(self, num_views: int, height: int, width: int, device):
        key = (num_views, height, width)
        if key in self.precompute_pos:
            return self.precompute_pos[key].to(device)
        pos = precompute_freqs_cis_3d(self.hidden_size // self.num_heads, num_views, height, width).to(device)
        self.precompute_pos[key] = pos
        return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        img_cond: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Multi-view noisy input [B, V, C, H, W]
            t: Diffusion timesteps [B]
            img_cond: Conditioning front image [B, C, H, W] or None
            s: Optional precomputed patch state [B, V*L, D]
            mask: Optional attention mask
        Returns:
            torch.Tensor: Denoised multi-view output [B, V, C_out, H, W]
        """
        B, V, C, H, W = x.shape
        P = self.patch_size
        Hs, Ws = H // P, W // P
        L = Hs * Ws
        VL = V * L
        P2 = P * P

        # 1. Conditioning
        t_emb = self.t_embedder(t.view(-1)).view(B, 1, self.hidden_size)
        img_emb = self.img_embedder(img_cond, batch_size=B).view(B, 1, self.hidden_size)
        c = torch.nn.functional.silu(t_emb + img_emb)

        # 2. Level 1: Patch Space
        pos = self.fetch_pos(V, Hs, Ws, x.device)
        if s is None:
            x_bv = x.view(B * V, C, H, W)
            x_patches_bv = torch.nn.functional.unfold(x_bv, kernel_size=P, stride=P).transpose(1, 2).contiguous()
            x_patches = x_patches_bv.view(B, VL, C * P2)

            s = self.s_embedder(x_patches)
            for block in self.patch_blocks:
                s = block(s, c, pos, mask=mask)
            s = torch.nn.functional.silu(t_emb + s)

        # 3. Level 2: Pixel Space
        s_cond = s.view(B * VL, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=P)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, V, H, W, P, mask=mask)

        # 4. Output Reconstruction
        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels

        out_bv = x_pixels.view(B * V, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        out_bv = out_bv.view(B * V, C_out * P2, L)
        x_img_bv = torch.nn.functional.fold(out_bv, (H, W), kernel_size=P, stride=P)
        x_img = x_img_bv.view(B, V, C_out, H, W)
        return x_img
