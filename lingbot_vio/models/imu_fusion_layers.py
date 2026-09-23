"""
Novel IMU-Visual Fusion Layers for LingBot-VIO.

This module contains three fusion modules designed for the LingBot-VIO paper:

1. IMUCrossAttentionLayer
   Cross-attention where visual tokens (Q) attend to IMU features (KV).
   Injects motion information into visual tokens at the aggregated token level.

2. IMUPriorAdaLN
   Adaptive Layer Normalization conditioned on IMU pose prior.
   Produces shift/scale/gate to modulate visual tokens.

3. IMUPoseCorrectionHead
   Lightweight head that predicts a pose correction delta given fused
   visual+IMU features and IMU pose prior.  Final refinement step.

Design principles:
- All modules handle both 4D [B, S, P, C] and 3D [B, N, C] tensor shapes.
- Zero initialization for gates / modulation ensures stable training start.
- float32 is used for numerical stability in the correction head.
- visual_dim = 2 * 1024 = 2048  (GCTStream aggregator outputs 2*embed_dim tokens)
- imu_dim   = 256               (AirIMU encoder output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. IMUCrossAttentionLayer
# ---------------------------------------------------------------------------

class IMUCrossAttentionLayer(nn.Module):
    """Cross-attention where visual tokens (Q) attend to IMU features (KV).

    This injects motion information into visual tokens at the aggregated
    token level.  A gated residual connection (gate initialised to 0)
    ensures that the module starts as an identity and gradually learns to
    incorporate IMU signals.

    Args:
        visual_dim: Dimension of visual token channels (default 2048,
            because GCTStream aggregator outputs 2*embed_dim tokens).
        imu_dim: Dimension of IMU feature channels (default 256,
            AirIMU encoder output).
        num_heads: Number of attention heads.
        dropout: Dropout rate applied after attention and output projection.
        qkv_bias: Whether to add bias to Q / K / V linear projections.
    """

    def __init__(
        self,
        visual_dim: int = 2048,
        imu_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        assert visual_dim % num_heads == 0, (
            f"visual_dim ({visual_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.visual_dim = visual_dim
        self.imu_dim = imu_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads

        # Project IMU features to visual dimension
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )

        # Q / K / V projections
        self.q_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(visual_dim, visual_dim)

        # Normalisation layers (pre-norm style)
        self.norm_visual = nn.LayerNorm(visual_dim)
        self.norm_imu = nn.LayerNorm(visual_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gated residual -- initialised to 0 so the layer starts as identity
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform for projections; zero bias for output."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        imu_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual_tokens: Visual tokens, shape ``[B, S, P, C]`` or ``[B, N, C]``.
                B: batch, S: num_frames, P: num_patches, C: visual_dim.
            imu_features: IMU features, shape ``[B, T, D]``.
                T: temporal length, D: imu_dim (256).
            mask: Optional attention mask for ``F.scaled_dot_product_attention``.

        Returns:
            Fused visual tokens with the same shape as *visual_tokens*.
        """
        # ---- Handle both 4D and 3D input shapes ----
        input_shape = visual_tokens.shape
        if len(input_shape) == 4:
            B, S, P, C = input_shape
            visual_tokens = visual_tokens.reshape(B, S * P, C)
        else:
            B, N, C = input_shape
            S, P = None, None

        # ---- Project IMU features to visual dim ----
        imu_proj = self.imu_proj(imu_features)  # [B, T, C]

        # ---- Pre-norm ----
        visual_normed = self.norm_visual(visual_tokens)  # [B, N, C]
        imu_normed = self.norm_imu(imu_proj)             # [B, T, C]

        # ---- Compute Q, K, V ----
        Q = self.q_proj(visual_normed)  # [B, N, C]
        K = self.k_proj(imu_normed)     # [B, T, C]
        V = self.v_proj(imu_normed)     # [B, T, C]

        # ---- Reshape for multi-head attention ----
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # ---- Scaled dot-product attention (efficient PyTorch impl) ----
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )  # [B, H, N, D]

        # ---- Reshape and project ----
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, C)  # [B, N, C]
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # ---- Gated residual connection ----
        gate = torch.sigmoid(self.gate)
        fused_tokens = visual_tokens + gate * attn_output

        # ---- Restore original shape if needed ----
        if S is not None and P is not None:
            fused_tokens = fused_tokens.view(B, S, P, C)

        return fused_tokens


# ---------------------------------------------------------------------------
# 2. IMUPriorAdaLN
# ---------------------------------------------------------------------------

class IMUPriorAdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on IMU pose prior.

    Takes an IMU relative pose encoding and produces (shift, scale, gate)
    to modulate visual tokens.  All modulation weights are initialised to
    zero so the module starts as an identity transform, ensuring stable
    training from the outset.

    The modulation formula is::

        output = gate * (norm(x) * (1 + scale) + shift) + x

    Args:
        dim: Feature dimension of visual tokens (default 2048).
        condition_dim: Dimension of the IMU pose encoding.  Defaults to
            *dim* if not specified.
    """

    def __init__(
        self,
        dim: int = 2048,
        condition_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        condition_dim = condition_dim or dim

        # LayerNorm without learnable affine params (we supply our own)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Condition -> (shift, scale, gate) via SiLU + Linear
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3 * dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init modulation so the module starts as identity."""
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        imu_pose_encoding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive modulation to visual tokens.

        Args:
            x: Visual tokens, shape ``[B, S, P, C]`` or ``[B, N, C]``.
            imu_pose_encoding: IMU pose encoding, shape ``[B, C]`` or
                ``[B, S, C]``.

        Returns:
            Modulated visual tokens with the same shape as *x*.
        """
        # Broadcast condition to match x dimensions
        # Insert dims just before the last (channel) dim so broadcasting works
        while imu_pose_encoding.dim() < x.dim():
            imu_pose_encoding = imu_pose_encoding.unsqueeze(-2)

        # Get modulation parameters: shift, scale, gate
        modulation_params = self.modulation(imu_pose_encoding)  # [..., 3*C]
        shift, scale, gate = modulation_params.chunk(3, dim=-1)  # each [..., C]

        # Apply: gate * (norm(x) * (1 + scale) + shift) + x
        x_normed = self.norm(x)
        x_modulated = gate * (x_normed * (1 + scale) + shift) + x

        return x_modulated


# ---------------------------------------------------------------------------
# 3. IMUPoseCorrectionHead
# ---------------------------------------------------------------------------

class IMUPoseCorrectionHead(nn.Module):
    """Lightweight head that predicts a pose correction delta.

    Given fused visual+IMU features and an IMU pose prior, this head
    predicts a 9-dimensional correction delta in the ``absT_quaR_FoV``
    parameterisation (3 translation + 4 quaternion + 2 FoV).  The output
    is weighted by a per-frame confidence value so that low-confidence
    IMU priors result in smaller corrections.

    Args:
        visual_dim: Dimension of visual token channels (default 2048).
        imu_pose_dim: Dimension of the IMU pose encoding (default 2048).
        hidden_dim: Hidden layer size for the correction MLP.
        pose_dim: Output pose dimension (9 for absT_quaR_FoV).
    """

    def __init__(
        self,
        visual_dim: int = 2048,
        imu_pose_dim: int = 2048,
        hidden_dim: int = 512,
        pose_dim: int = 9,
    ) -> None:
        super().__init__()

        self.visual_dim = visual_dim
        self.imu_pose_dim = imu_pose_dim
        self.pose_dim = pose_dim

        # MLP: concat(camera_token, imu_pose_encoding) -> pose correction delta
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + imu_pose_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Zero-init the final layer so corrections start at zero."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        fused_tokens: torch.Tensor,
        imu_pose_encoding: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict confidence-weighted pose correction delta.

        Args:
            fused_tokens: Fused visual+IMU tokens, shape ``[B, S, P, C]``
                or ``[B, N, C]``.
            imu_pose_encoding: IMU pose encoding, shape ``[B, S, C]`` or
                ``[B, C]``.
            confidence: Per-frame confidence, shape ``[B, S, 1]`` or
                ``[B, 1]``.

        Returns:
            A tuple of:
            - corrected_delta: Confidence-weighted pose correction delta,
              shape ``[B, S, 9]``.
            - raw_delta: Unweighted raw correction delta,
              shape ``[B, S, 9]``.
        """
        # Use float32 for numerical stability
        orig_dtype = fused_tokens.dtype
        fused_tokens = fused_tokens.float()
        imu_pose_encoding = imu_pose_encoding.float()
        confidence = confidence.float()

        # ---- Extract camera tokens (first token per frame) ----
        if fused_tokens.dim() == 4:
            # [B, S, P, C] -> camera token is the first patch token per frame
            camera_tokens = fused_tokens[:, :, 0, :]  # [B, S, C]
        else:
            # [B, N, C] -> treat first token as camera token
            camera_tokens = fused_tokens[:, 0:1, :]   # [B, 1, C]

        # ---- Broadcast imu_pose_encoding to match camera_tokens ----
        if imu_pose_encoding.dim() == 2:
            # [B, C] -> [B, S, C]
            S_cam = camera_tokens.shape[1]
            imu_pose_encoding = imu_pose_encoding.unsqueeze(1).expand(-1, S_cam, -1)
        elif imu_pose_encoding.dim() == 3 and imu_pose_encoding.shape[1] != camera_tokens.shape[1]:
            # Mismatch in sequence dim (e.g. 3D fused_tokens with multi-frame pose encoding)
            # Take only the first frame's encoding
            imu_pose_encoding = imu_pose_encoding[:, :camera_tokens.shape[1], :]

        # ---- Broadcast confidence to match camera_tokens ----
        S_cam = camera_tokens.shape[1]
        if confidence.dim() == 1:
            # [B] -> [B, S, 1]
            confidence = confidence.unsqueeze(1).unsqueeze(2).expand(-1, S_cam, -1)
        elif confidence.dim() == 2:
            if confidence.shape[1] == 1:
                # [B, 1] -> [B, S, 1]
                confidence = confidence.unsqueeze(2).expand(-1, S_cam, -1)
            else:
                # [B, S] or [B, S_conf] -> [B, S, 1]
                confidence = confidence.unsqueeze(2)
                if confidence.shape[1] != S_cam:
                    confidence = confidence[:, :S_cam, :]
        elif confidence.dim() == 3 and confidence.shape[1] != S_cam:
            confidence = confidence[:, :S_cam, :]

        # ---- Concat and predict ----
        mlp_input = torch.cat([camera_tokens, imu_pose_encoding], dim=-1)  # [B, S, C + C_pose]
        raw_delta = self.mlp(mlp_input)  # [B, S, 9]

        # ---- Confidence-weighted output ----
        corrected_delta = raw_delta * confidence  # [B, S, 9]

        # Cast back to original dtype
        corrected_delta = corrected_delta.to(orig_dtype)
        raw_delta = raw_delta.to(orig_dtype)

        return corrected_delta, raw_delta
