"""
Fusion modules for combining AirIMU features with LingBot-Map visual features.

This module implements:
- IMUCrossAttention: Cross-attention from IMU features to visual tokens
- PosePriorEncoder: Encodes IMU-derived relative poses for conditioning
- AdaLNModulation: Adaptive LayerNorm for injecting pose priors
- FusedCameraHead: Camera head with IMU pose prior injection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class IMUCrossAttention(nn.Module):
    """
    Cross-attention module that fuses IMU features with visual tokens.
    
    The visual tokens attend to IMU features, allowing the network to
    leverage motion information when processing visual data.
    
    Architecture:
    - Project IMU features to visual dimension
    - Multi-head cross-attention (visual queries, IMU keys/values)
    - Residual connection with layer norm
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        imu_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.imu_dim = imu_dim
        self.num_heads = num_heads
        self.head_dim = visual_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert visual_dim % num_heads == 0, "visual_dim must be divisible by num_heads"
        
        # Project IMU features to visual dimension
        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, visual_dim),
            nn.LayerNorm(visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )
        
        # Query projection for visual tokens
        self.q_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        
        # Key and Value projections for IMU features
        self.k_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(visual_dim, visual_dim, bias=qkv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(visual_dim, visual_dim)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable gating for residual connection
        self.gate = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
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
        """
        Forward pass for IMU-to-visual cross-attention.
        
        Args:
            visual_tokens: Visual tokens [B, S, P, C] or [B, N, C]
                B: batch, S: num_frames, P: num_patches, C: channels
            imu_features: IMU features [B, T, D]
                T: temporal length, D: imu_dim (256)
            mask: Optional attention mask
        
        Returns:
            fused_tokens: Visual tokens fused with IMU information
        """
        # Handle both [B, S, P, C] and [B, N, C] input formats
        input_shape = visual_tokens.shape
        if len(input_shape) == 4:
            B, S, P, C = input_shape
            visual_tokens = visual_tokens.view(B, S * P, C)
        else:
            B, N, C = input_shape
            S, P = None, None
        
        # Project IMU features to visual dimension
        imu_proj = self.imu_proj(imu_features)  # [B, T, C]
        
        # Normalize inputs
        visual_normed = self.norm1(visual_tokens)
        imu_normed = self.norm2(imu_proj)
        
        # Compute Q, K, V
        Q = self.q_proj(visual_normed)  # [B, N, C]
        K = self.k_proj(imu_normed)  # [B, T, C]
        V = self.v_proj(imu_normed)  # [B, T, C]
        
        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, C)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Gated residual connection
        gate = torch.sigmoid(self.gate)
        fused_tokens = visual_tokens + gate * attn_output
        
        # Restore original shape if needed
        if S is not None and P is not None:
            fused_tokens = fused_tokens.view(B, S, P, C)
        
        return fused_tokens


class PosePriorEncoder(nn.Module):
    """
    Encodes IMU-derived relative pose information for conditioning.
    
    Takes the relative pose (position, rotation, velocity) and covariance
    from AirIMU integration and encodes it into a feature vector that can
    be used to condition the visual transformer.
    
    The encoder also incorporates uncertainty information from the IMU
    covariance to weight the prior appropriately.
    """
    
    def __init__(
        self,
        output_dim: int = 1024,
        use_covariance: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.use_covariance = use_covariance
        
        # Compute input dimensions
        # Position: 3, Velocity: 3, Rotation (quaternion): 4
        pose_dim = 10  # 3 + 3 + 4
        cov_dim = 6 if use_covariance else 0  # pos_cov (3) + rot_cov (3)
        input_dim = pose_dim + cov_dim
        
        # MLP encoder
        hidden_dim = output_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Separate encoders for different pose components (for interpretability)
        self.pos_encoder = nn.Linear(3, output_dim // 4)
        self.vel_encoder = nn.Linear(3, output_dim // 4)
        self.rot_encoder = nn.Linear(4, output_dim // 4)
        
        if use_covariance:
            self.cov_encoder = nn.Linear(6, output_dim // 4)
        else:
            self.cov_encoder = None
        
        # Fusion layer
        fusion_input_dim = output_dim // 4 * (4 if use_covariance else 3)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        
        # Uncertainty-based gating
        if use_covariance:
            self.uncertainty_gate = nn.Sequential(
                nn.Linear(6, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.uncertainty_gate = None
    
    def forward(
        self,
        rel_pos: torch.Tensor,
        rel_rot: torch.Tensor,
        rel_vel: Optional[torch.Tensor] = None,
        pos_cov: Optional[torch.Tensor] = None,
        rot_cov: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode relative pose into a conditioning vector.
        
        Args:
            rel_pos: Relative position [B, 3]
            rel_rot: Relative rotation quaternion [B, 4] (w, x, y, z)
            rel_vel: Relative velocity [B, 3] (optional)
            pos_cov: Position covariance diagonal [B, 3] (optional)
            rot_cov: Rotation covariance diagonal [B, 3] (optional)
        
        Returns:
            pose_encoding: Encoded pose [B, output_dim]
            confidence: Confidence weight based on uncertainty [B, 1] (if covariance provided)
        """
        B = rel_pos.shape[0]
        device = rel_pos.device
        
        # Default velocity to zeros if not provided
        if rel_vel is None:
            rel_vel = torch.zeros(B, 3, device=device, dtype=rel_pos.dtype)
        
        # Encode individual components
        pos_feat = self.pos_encoder(rel_pos)  # [B, D/4]
        vel_feat = self.vel_encoder(rel_vel)  # [B, D/4]
        rot_feat = self.rot_encoder(rel_rot)  # [B, D/4]
        
        if self.use_covariance and pos_cov is not None and rot_cov is not None:
            # Concatenate covariances
            cov = torch.cat([pos_cov, rot_cov], dim=-1)  # [B, 6]
            cov_feat = self.cov_encoder(cov)  # [B, D/4]
            
            # Compute uncertainty-based confidence
            confidence = self.uncertainty_gate(cov)  # [B, 1]
            
            # Fuse all features
            features = torch.cat([pos_feat, vel_feat, rot_feat, cov_feat], dim=-1)
        else:
            confidence = None
            features = torch.cat([pos_feat, vel_feat, rot_feat], dim=-1)
        
        # Final fusion
        pose_encoding = self.fusion(features)  # [B, output_dim]
        
        return pose_encoding, confidence


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization modulation module.
    
    Takes a conditioning vector (e.g., from PosePriorEncoder) and produces
    scale and shift parameters to modulate normalized features. This allows
    the IMU pose prior to influence the visual features adaptively.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        
        condition_dim = condition_dim or dim
        
        # Layer norm without learnable parameters
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Modulation projection: condition -> (shift, scale, gate)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3 * dim, bias=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize modulation weights to produce identity at start."""
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply adaptive modulation to input features.
        
        Args:
            x: Input features [B, ..., C]
            condition: Conditioning vector [B, C] or [B, 1, C]
        
        Returns:
            Modulated features [B, ..., C]
        """
        # Ensure condition has right shape for broadcasting with x
        while condition.dim() < x.dim():
            condition = condition.unsqueeze(1)
        
        # Get modulation parameters
        modulation_params = self.modulation(condition)
        shift, scale, gate = modulation_params.chunk(3, dim=-1)
        
        # Apply adaptive layer norm
        x_normed = self.norm(x)
        x_modulated = modulate(x_normed, shift, scale)
        
        # Gated residual
        output = gate * x_modulated + x
        
        return output


class IMUTemporalEncoder(nn.Module):
    """
    Temporal encoder that summarizes variable-length IMU sequences into
    fixed-size features suitable for cross-attention with visual tokens.
    
    Uses a small transformer to process IMU features and produce a summary
    token that captures the motion information between two visual frames.
    """
    
    def __init__(
        self,
        imu_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.imu_dim = imu_dim
        self.output_dim = output_dim
        
        # Project input if dimensions don't match
        if imu_dim != output_dim:
            self.input_proj = nn.Linear(imu_dim, output_dim)
        else:
            self.input_proj = nn.Identity()
        
        # Learnable summary token
        self.summary_token = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 512, output_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output normalization
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        imu_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode IMU sequence into summary features.
        
        Args:
            imu_features: IMU features [B, T, D]
            mask: Optional padding mask [B, T]
        
        Returns:
            summary: Summary features [B, output_dim]
        """
        B, T, D = imu_features.shape
        
        # Project input
        x = self.input_proj(imu_features)  # [B, T, C]
        
        # Add positional encoding
        if T <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :T, :]
        else:
            # Interpolate positional encoding if sequence is longer
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            x = x + pos_embed
        
        # Prepend summary token
        summary_token = self.summary_token.expand(B, -1, -1)  # [B, 1, C]
        x = torch.cat([summary_token, x], dim=1)  # [B, T+1, C]
        
        # Update mask for summary token
        if mask is not None:
            summary_mask = torch.zeros(B, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([summary_mask, mask], dim=1)
        
        # Transform
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Extract and normalize summary
        summary = self.norm(x[:, 0, :])  # [B, C]
        
        return summary


class FusedCameraHead(nn.Module):
    """
    Camera head with IMU pose prior injection via AdaLN.
    
    This extends the LingBot-Map CameraHead to accept IMU pose priors
    and use them to condition the iterative refinement process.
    """
    
    def __init__(
        self,
        dim_in: int = 1024,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR_FoV",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        trans_act: str = "linear",
        quat_act: str = "linear",
        fl_act: str = "relu",
        num_iterations: int = 4,
        # IMU fusion parameters
        imu_condition_dim: int = 1024,
        use_imu_prior: bool = True,
    ):
        super().__init__()
        
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9  # 3 (trans) + 4 (quat) + 2 (fov)
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")
        
        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.num_iterations = num_iterations
        self.use_imu_prior = use_imu_prior
        
        # Build trunk blocks (simplified version without all LingBot dependencies)
        self.trunk = nn.ModuleList([
            self._build_block(dim_in, num_heads, mlp_ratio, init_values)
            for _ in range(trunk_depth)
        ])
        
        # Normalizations
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        
        # Learnable empty camera pose token
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        
        # Original pose modulation (from LingBot-Map)
        self.poseLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_in, 3 * dim_in, bias=True),
        )
        
        # Adaptive layer norm
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        
        # Pose prediction branch
        self.pose_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.GELU(),
            nn.Linear(dim_in // 2, self.target_dim),
        )
        
        # IMU prior injection via AdaLN
        if use_imu_prior:
            self.imu_adaln = AdaLNModulation(
                dim=dim_in,
                condition_dim=imu_condition_dim,
            )
            
            # Additional gate for IMU influence
            self.imu_gate = nn.Sequential(
                nn.Linear(imu_condition_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
    
    def _build_block(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        init_values: float,
    ) -> nn.Module:
        """Build a transformer block."""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(dim, num_heads, batch_first=True),
            nn.LayerNorm(dim),
            nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Linear(dim * mlp_ratio, dim),
            ),
        )
    
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        imu_pose_prior: Optional[torch.Tensor] = None,
        imu_confidence: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Forward pass with optional IMU pose prior conditioning.
        
        Args:
            aggregated_tokens_list: List of token tensors from aggregator
            imu_pose_prior: Encoded IMU pose prior [B, S, C] or [B, C]
            imu_confidence: Confidence from IMU uncertainty [B, S, 1] or [B, 1]
            num_iterations: Number of refinement iterations
        
        Returns:
            List of predicted camera encodings from each iteration
        """
        if num_iterations is None:
            num_iterations = self.num_iterations
        
        # Get camera tokens from last aggregator output
        tokens = aggregated_tokens_list[-1]  # [B, S, P, C] or similar
        
        # Extract camera tokens (first token per frame)
        if tokens.dim() == 4:
            pose_tokens = tokens[:, :, 0, :]  # [B, S, C]
        else:
            pose_tokens = tokens[:, 0:1, :]  # [B, 1, C]
        
        pose_tokens = self.token_norm(pose_tokens)
        
        # Iterative refinement
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []
        
        for _ in range(num_iterations):
            # Initialize or update module input
            if pred_pose_enc is None:
                module_input = self.embed_pose(
                    self.empty_pose_tokens.expand(B, S, -1)
                )
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)
            
            # Generate modulation parameters
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            
            # Apply adaptive layer norm and modulation
            pose_tokens_modulated = gate_msa * modulate(
                self.adaln_norm(pose_tokens), shift_msa, scale_msa
            )
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens
            
            # Apply IMU prior conditioning if available
            if self.use_imu_prior and imu_pose_prior is not None:
                # Ensure proper shape
                if imu_pose_prior.dim() == 2:
                    imu_prior = imu_pose_prior.unsqueeze(1).expand(-1, S, -1)
                else:
                    imu_prior = imu_pose_prior
                
                # Apply IMU AdaLN modulation
                pose_tokens_modulated = self.imu_adaln(pose_tokens_modulated, imu_prior)
                
                # Apply confidence-based gating if available
                if imu_confidence is not None:
                    imu_influence = self.imu_gate(imu_prior)  # [B, S, 1]
                    if imu_confidence.dim() == 2:
                        imu_confidence = imu_confidence.unsqueeze(1)
                    combined_gate = imu_influence * imu_confidence
                    pose_tokens_modulated = (
                        combined_gate * pose_tokens_modulated +
                        (1 - combined_gate) * pose_tokens
                    )
            
            # Apply trunk blocks
            for block in self.trunk:
                norm1, attn, norm2, mlp = block
                # Self-attention
                x_normed = norm1(pose_tokens_modulated)
                attn_out, _ = attn(x_normed, x_normed, x_normed)
                pose_tokens_modulated = pose_tokens_modulated + attn_out
                # MLP
                pose_tokens_modulated = pose_tokens_modulated + mlp(norm2(pose_tokens_modulated))
            
            # Predict pose delta
            pred_pose_enc_delta = self.pose_branch(
                self.trunk_norm(pose_tokens_modulated)
            )
            
            # Accumulate predictions
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            
            # Apply activations
            activated_pose = self._activate_pose(pred_pose_enc)
            pred_pose_enc_list.append(activated_pose)
        
        return pred_pose_enc_list
    
    def _activate_pose(self, pose_enc: torch.Tensor) -> torch.Tensor:
        """
        Apply activation functions to pose encoding.
        
        Args:
            pose_enc: Raw pose encoding [B, S, 9]
                - [:, :, 0:3]: translation
                - [:, :, 3:7]: quaternion
                - [:, :, 7:9]: field of view
        
        Returns:
            Activated pose encoding
        """
        trans = pose_enc[..., :3]
        quat = pose_enc[..., 3:7]
        fov = pose_enc[..., 7:9]
        
        # Apply activations based on configuration
        if self.trans_act == "linear":
            trans_act = trans
        elif self.trans_act == "tanh":
            trans_act = torch.tanh(trans)
        else:
            trans_act = trans
        
        if self.quat_act == "linear":
            quat_act = quat
        elif self.quat_act == "normalize":
            quat_act = F.normalize(quat, p=2, dim=-1)
        else:
            quat_act = quat
        
        if self.fl_act == "relu":
            fov_act = F.relu(fov)
        elif self.fl_act == "softplus":
            fov_act = F.softplus(fov)
        else:
            fov_act = fov
        
        return torch.cat([trans_act, quat_act, fov_act], dim=-1)


class FusionBlock(nn.Module):
    """
    Complete fusion block that combines IMU features with visual tokens.
    
    This block:
    1. Encodes IMU features temporally
    2. Applies cross-attention from visual to IMU
    3. Encodes the IMU pose prior
    4. Produces modulated visual features
    """
    
    def __init__(
        self,
        visual_dim: int = 1024,
        imu_feature_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temporal_encoding: bool = True,
        use_cross_attention: bool = True,
        use_pose_prior: bool = True,
    ):
        super().__init__()
        
        self.use_temporal_encoding = use_temporal_encoding
        self.use_cross_attention = use_cross_attention
        self.use_pose_prior = use_pose_prior
        
        # Temporal encoder for IMU features
        if use_temporal_encoding:
            self.temporal_encoder = IMUTemporalEncoder(
                imu_dim=imu_feature_dim,
                output_dim=imu_feature_dim,
                num_heads=4,
                num_layers=2,
                dropout=dropout,
            )
        
        # Cross-attention for feature fusion
        if use_cross_attention:
            self.cross_attention = IMUCrossAttention(
                visual_dim=visual_dim,
                imu_dim=imu_feature_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # Pose prior encoder
        if use_pose_prior:
            self.pose_encoder = PosePriorEncoder(
                output_dim=visual_dim,
                use_covariance=True,
                dropout=dropout,
            )
            
            # AdaLN for pose conditioning
            self.adaln = AdaLNModulation(
                dim=visual_dim,
                condition_dim=visual_dim,
            )
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        imu_features: torch.Tensor,
        imu_pose: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through the fusion block.
        
        Args:
            visual_tokens: Visual tokens [B, S, P, C] or [B, N, C]
            imu_features: Raw IMU features [B, T, D]
            imu_pose: Dictionary with rel_pos, rel_rot, rel_vel, pos_cov, rot_cov
        
        Returns:
            fused_tokens: Fused visual tokens
            pose_encoding: Encoded pose prior (if use_pose_prior)
            confidence: Uncertainty-based confidence (if use_pose_prior)
        """
        # Temporal encoding of IMU features
        if self.use_temporal_encoding:
            imu_summary = self.temporal_encoder(imu_features)  # [B, D]
            imu_features_processed = imu_summary.unsqueeze(1)  # [B, 1, D]
        else:
            imu_features_processed = imu_features
        
        # Cross-attention fusion
        if self.use_cross_attention:
            visual_tokens = self.cross_attention(visual_tokens, imu_features_processed)
        
        # Pose prior encoding and conditioning
        pose_encoding = None
        confidence = None
        
        if self.use_pose_prior and imu_pose is not None:
            pose_encoding, confidence = self.pose_encoder(
                rel_pos=imu_pose.get('rel_pos'),
                rel_rot=imu_pose.get('rel_rot'),
                rel_vel=imu_pose.get('rel_vel'),
                pos_cov=imu_pose.get('pos_cov'),
                rot_cov=imu_pose.get('rot_cov'),
            )
            
            # Apply AdaLN conditioning
            visual_tokens = self.adaln(visual_tokens, pose_encoding)
        
        return visual_tokens, pose_encoding, confidence
