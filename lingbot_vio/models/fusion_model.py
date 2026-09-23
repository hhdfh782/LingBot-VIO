"""
IMU-LingBot Fusion Model.

This module implements the main fusion model that combines:
- AirIMU encoder for IMU feature extraction and pose estimation
- LingBot-Map visual encoder for image feature extraction
- Fusion modules for combining IMU and visual features

The project ships with bundled copies of both AirIMU and LingBot-Map
under ``third_party/`` so it runs as a fully standalone repository.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field

from .imu_encoder import AirIMUEncoder, AirIMUConfig, create_airimu_encoder
from .fusion_modules import (
    IMUCrossAttention,
    PosePriorEncoder,
    AdaLNModulation,
    FusedCameraHead,
    FusionBlock,
    IMUTemporalEncoder,
)


@dataclass
class IMULingBotConfig:
    """Configuration for the IMU-LingBot fusion model."""
    
    # Image settings
    img_size: int = 518
    patch_size: int = 14
    
    # Visual encoder settings
    visual_encoder_type: str = "simple"  # "simple" for standalone test or "gct_stream" for real LingBot-Map
    visual_dim: int = 1024
    visual_num_heads: int = 16
    visual_checkpoint: Optional[str] = None
    freeze_visual: bool = True
    
    # IMU encoder settings
    imu_dim: int = 256
    imu_checkpoint: Optional[str] = None
    freeze_imu: bool = True
    imu_propcov: bool = True
    
    # Fusion settings
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    use_cross_attention: bool = True
    use_pose_prior: bool = True
    use_temporal_encoding: bool = True
    
    # Camera head settings
    camera_trunk_depth: int = 4
    camera_num_iterations: int = 4
    
    # Output settings
    enable_depth: bool = True
    enable_points: bool = False


class IMULingBotFusion(nn.Module):
    """
    Fusion model combining AirIMU and LingBot-Map for visual-inertial odometry.
    
    Architecture:
    1. AirIMU branch: Process IMU data to get features and relative poses
    2. LingBot-Map branch: Process images to get visual features
    3. Fusion: Combine IMU features with visual features via cross-attention
    4. Heads: Predict camera poses (and optionally depth/points)
    
    The model supports:
    - Feature-level fusion: IMU features attend to visual tokens
    - Pose prior fusion: IMU relative poses condition camera head
    - Uncertainty-aware fusion: IMU covariance weights the prior
    """
    
    def __init__(self, config: Optional[IMULingBotConfig] = None):
        super().__init__()
        
        self.config = config or IMULingBotConfig()
        
        # Build IMU encoder
        self.imu_encoder = self._build_imu_encoder()
        
        # Build visual encoder (placeholder - actual LingBot-Map integration)
        self.visual_encoder = self._build_visual_encoder()
        
        # Build fusion modules
        self.fusion_block = self._build_fusion_block()
        
        # Build prediction heads
        self.camera_head = self._build_camera_head()
        
        if self.config.enable_depth:
            self.depth_head = self._build_depth_head()
        else:
            self.depth_head = None
        
        # IMU-to-camera extrinsic (learnable or fixed)
        # Default: identity transformation
        self.register_buffer(
            'T_cam_imu',
            torch.eye(4, dtype=torch.float32)
        )
    
    def _build_imu_encoder(self) -> AirIMUEncoder:
        """Build and configure the AirIMU encoder."""
        imu_config = AirIMUConfig(
            propcov=self.config.imu_propcov,
        )
        
        encoder = create_airimu_encoder(
            checkpoint_path=self.config.imu_checkpoint,
            freeze=self.config.freeze_imu,
            config=imu_config,
        )
        
        return encoder
    
    def _build_visual_encoder(self) -> nn.Module:
        """
        Build the visual encoder.
        
        When ``visual_encoder_type == "gct_stream"`` the real LingBot-Map
        GCTStream model (bundled under ``third_party/lingbot_map``) is used.
        Otherwise a lightweight placeholder ViT is created for testing.
        """
        if self.config.visual_encoder_type == "gct_stream":
            return GCTStreamVisualEncoder(
                checkpoint_path=self.config.visual_checkpoint,
                embed_dim=self.config.visual_dim,
                freeze=self.config.freeze_visual,
            )

        return SimpleVisualEncoder(
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.visual_dim,
            num_heads=self.config.visual_num_heads,
            checkpoint_path=self.config.visual_checkpoint,
            freeze=self.config.freeze_visual,
        )
    
    def _build_fusion_block(self) -> FusionBlock:
        """Build the fusion block."""
        return FusionBlock(
            visual_dim=self.config.visual_dim,
            imu_feature_dim=self.config.imu_dim,
            num_heads=self.config.fusion_num_heads,
            dropout=self.config.fusion_dropout,
            use_temporal_encoding=self.config.use_temporal_encoding,
            use_cross_attention=self.config.use_cross_attention,
            use_pose_prior=self.config.use_pose_prior,
        )
    
    def _build_camera_head(self) -> FusedCameraHead:
        """Build the camera head with IMU conditioning."""
        return FusedCameraHead(
            dim_in=self.config.visual_dim,
            trunk_depth=self.config.camera_trunk_depth,
            num_heads=self.config.visual_num_heads,
            num_iterations=self.config.camera_num_iterations,
            imu_condition_dim=self.config.visual_dim,
            use_imu_prior=self.config.use_pose_prior,
        )
    
    def _build_depth_head(self) -> nn.Module:
        """Build a simple depth prediction head."""
        return SimpleDepthHead(
            dim_in=self.config.visual_dim,
            patch_size=self.config.patch_size,
        )
    
    def set_imu_camera_extrinsic(self, T_cam_imu: torch.Tensor):
        """
        Set the IMU-to-camera extrinsic transformation.
        
        Args:
            T_cam_imu: 4x4 transformation matrix from IMU to camera frame
        """
        self.T_cam_imu.copy_(T_cam_imu)
    
    def freeze_pretrained(self):
        """Freeze all pretrained components."""
        if self.config.freeze_imu:
            self.imu_encoder.freeze_encoder()
        
        if self.config.freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all components for end-to-end training."""
        self.imu_encoder.unfreeze_encoder()
        for param in self.visual_encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (fusion modules only by default)."""
        params = []
        
        # Fusion block is always trainable
        params.extend(self.fusion_block.parameters())
        
        # Camera head fusion components are trainable
        params.extend(self.camera_head.parameters())
        
        # Depth head if enabled
        if self.depth_head is not None:
            params.extend(self.depth_head.parameters())
        
        return params
    
    def forward(
        self,
        images: torch.Tensor,
        imu_acc: torch.Tensor,
        imu_gyro: torch.Tensor,
        imu_dt: torch.Tensor,
        init_state: Optional[Dict[str, torch.Tensor]] = None,
        return_imu_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the fusion model.
        
        Args:
            images: RGB images [B, S, 3, H, W] where S is number of frames
            imu_acc: Accelerometer readings [B, T, 3]
            imu_gyro: Gyroscope readings [B, T, 3]
            imu_dt: Time deltas [B, T]
            init_state: Optional initial state for IMU integration
            return_imu_outputs: Whether to return intermediate IMU outputs
        
        Returns:
            Dictionary containing:
            - pose_enc: Predicted camera poses [B, S, 9]
            - depth: Predicted depth (if enabled) [B, S, H, W]
            - imu_features: IMU features (if return_imu_outputs)
            - imu_pose: IMU relative pose (if return_imu_outputs)
        """
        B, S, C, H, W = images.shape
        
        # 1. Process IMU data through AirIMU encoder
        imu_outputs = self.imu_encoder(
            acc=imu_acc,
            gyro=imu_gyro,
            dt=imu_dt,
            init_state=init_state,
            return_integrated=True,
        )
        
        imu_features = imu_outputs['features']  # [B, T', 256]
        
        # Get relative pose from IMU integration
        imu_pose = self.imu_encoder.get_relative_pose(imu_outputs)
        
        # 2. Process images through visual encoder
        visual_outputs = self.visual_encoder(images)
        visual_tokens = visual_outputs['tokens']  # [B, S, P, C]
        
        # 3. Fuse IMU features with visual tokens
        fused_tokens, pose_encoding, confidence = self.fusion_block(
            visual_tokens=visual_tokens,
            imu_features=imu_features,
            imu_pose=imu_pose,
        )
        
        # 4. Predict camera poses with IMU prior conditioning
        pose_enc_list = self.camera_head(
            aggregated_tokens_list=[fused_tokens],
            imu_pose_prior=pose_encoding,
            imu_confidence=confidence,
        )
        
        results = {
            'pose_enc': pose_enc_list[-1],  # [B, S, 9]
            'pose_enc_list': pose_enc_list,
        }
        
        # 5. Predict depth if enabled
        if self.depth_head is not None:
            depth_input = fused_tokens
            depth, depth_conf = self.depth_head(depth_input, images)
            results['depth'] = depth
            results['depth_conf'] = depth_conf
        
        # Include IMU outputs if requested
        if return_imu_outputs:
            results['imu_features'] = imu_features
            results['imu_pose'] = imu_pose
            results['correction_acc'] = imu_outputs['correction_acc']
            results['correction_gyro'] = imu_outputs['correction_gyro']
            if 'cov' in imu_outputs:
                results['imu_cov'] = imu_outputs['cov']
        
        return results
    
    def inference(
        self,
        images: torch.Tensor,
        imu_acc: torch.Tensor,
        imu_gyro: torch.Tensor,
        imu_dt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference-only forward pass (no gradient computation).
        
        Args:
            images: RGB images [B, S, 3, H, W]
            imu_acc: Accelerometer readings [B, T, 3]
            imu_gyro: Gyroscope readings [B, T, 3]
            imu_dt: Time deltas [B, T]
        
        Returns:
            Predictions dictionary
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                images=images,
                imu_acc=imu_acc,
                imu_gyro=imu_gyro,
                imu_dt=imu_dt,
                return_imu_outputs=False,
            )


class SimpleVisualEncoder(nn.Module):
    """
    Simplified visual encoder for standalone testing.
    
    In production, this should be replaced with the actual LingBot-Map
    aggregator. This encoder provides a compatible interface for testing
    the fusion architecture.
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 16,
        depth: int = 4,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # Camera token (special token for pose prediction)
        self.camera_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Freeze if requested
        if freeze:
            self._freeze()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights."""
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded visual encoder from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load visual encoder checkpoint: {e}")
    
    def _freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, S, 3, H, W] batch of image sequences
        
        Returns:
            Dictionary with 'tokens': [B, S, P+1, C]
        """
        B, S, C, H, W = images.shape
        
        # Reshape for processing
        images_flat = images.view(B * S, C, H, W)
        
        # Patch embedding
        x = self.patch_embed(images_flat)  # [B*S, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B*S, P, C]
        
        # Add camera token
        camera_tokens = self.camera_token.expand(B * S, -1, -1)
        x = torch.cat([camera_tokens, x], dim=1)  # [B*S, P+1, C]
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back
        tokens = x.view(B, S, -1, self.embed_dim)  # [B, S, P+1, C]
        
        return {'tokens': tokens}


class SimpleDepthHead(nn.Module):
    """
    Simplified depth prediction head for testing.
    
    In production, this should be replaced with the DPT head from LingBot-Map.
    """
    
    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.decoder = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # depth + confidence
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict depth from tokens.
        
        Args:
            tokens: [B, S, P, C] visual tokens
            images: [B, S, 3, H, W] input images (for size reference)
        
        Returns:
            depth: [B, S, H, W]
            confidence: [B, S, H, W]
        """
        B, S, P, C = tokens.shape
        H, W = images.shape[-2:]
        
        # Skip camera token (first token)
        patch_tokens = tokens[:, :, 1:, :]  # [B, S, P-1, C]
        
        # Decode
        output = self.decoder(patch_tokens)  # [B, S, P-1, 2]
        
        # Reshape to image
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        
        output = output.view(B, S, h_patches, w_patches, 2)
        output = output.permute(0, 1, 4, 2, 3)  # [B, S, 2, h, w]
        
        # Upsample to original size
        output = output.view(B * S, 2, h_patches, w_patches)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        output = output.view(B, S, 2, H, W)
        
        depth = torch.exp(output[:, :, 0])  # Positive depth
        confidence = torch.sigmoid(output[:, :, 1])
        
        return depth, confidence


class GCTStreamVisualEncoder(nn.Module):
    """
    Wrapper around the real LingBot-Map GCTStream model shipped under
    ``third_party/lingbot_map``.  It loads a pretrained checkpoint, freezes
    parameters if requested, and exposes the same ``forward(images) ->
    {'tokens': ...}`` interface as ``SimpleVisualEncoder``.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        embed_dim: int = 1024,
        freeze: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Import the bundled GCTStream
        from third_party.lingbot_map.models.gct_stream import GCTStream

        # Build with default hyper-parameters; the checkpoint will override
        # any mismatched buffers.
        self.gct = GCTStream(embed_dim=embed_dim)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if freeze:
            self._freeze()

    # ------------------------------------------------------------------
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.gct.load_state_dict(state, strict=False)
        if missing:
            print(f"[GCTStreamVisualEncoder] missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[GCTStreamVisualEncoder] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"[GCTStreamVisualEncoder] loaded checkpoint from {path}")

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, S, 3, H, W]
        Returns:
            dict with 'tokens': [B, S, P, C]  (camera + patch tokens per frame)
        """
        preds = self.gct(images)

        # The aggregator returns a list of intermediate token tensors.
        # We take the last one which has shape [B, S, P, 2*C].
        # Project back to C dims by splitting frame/global halves and adding.
        aggregated = preds.get("aggregated_tokens_list")
        if aggregated is not None and len(aggregated) > 0:
            last = aggregated[-1]  # [B, S, P, 2*C]
            # Split into frame / global and keep only the first half (frame)
            tokens = last[..., : self.embed_dim]
        else:
            # Fallback: build a dummy token tensor
            B, S = images.shape[:2]
            tokens = torch.zeros(B, S, 1, self.embed_dim, device=images.device)

        return {"tokens": tokens}


def create_imu_lingbot_model(
    imu_checkpoint: Optional[str] = None,
    visual_checkpoint: Optional[str] = None,
    freeze_pretrained: bool = True,
    config: Optional[IMULingBotConfig] = None,
) -> IMULingBotFusion:
    """
    Factory function to create an IMU-LingBot fusion model.
    
    Args:
        imu_checkpoint: Path to pretrained AirIMU weights
        visual_checkpoint: Path to pretrained LingBot-Map weights
        freeze_pretrained: Whether to freeze pretrained components
        config: Optional configuration
    
    Returns:
        Configured IMULingBotFusion model
    """
    if config is None:
        config = IMULingBotConfig()
    
    config.imu_checkpoint = imu_checkpoint
    config.visual_checkpoint = visual_checkpoint
    config.freeze_imu = freeze_pretrained
    config.freeze_visual = freeze_pretrained
    
    model = IMULingBotFusion(config)
    
    return model
