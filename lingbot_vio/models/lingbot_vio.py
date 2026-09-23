"""
LingBot-VIO: Visual-Inertial Odometry model with IMU fusion.

This module implements the main LingBot-VIO model that subclasses GCTStream
and adds IMU fusion between the visual aggregation and camera prediction stages.

Architecture:
    LingBotVIO(GCTStream)
      - aggregator (inherited)       - visual feature extraction with KV cache
      - camera_head (inherited)      - CameraCausalHead for pose prediction
      - depth_head (inherited)       - DPT depth prediction
      - imu_encoder (AirIMUEncoder)  - IMU feature extraction + integration
      - imu_cross_attn               - IMU→visual cross-attention
      - imu_prior_adaln              - IMU pose prior conditioning
      - imu_pose_encoder             - Encode IMU relative pose for conditioning
      - pose_correction_head         - Predict pose correction delta

Key design: Override ``forward()`` to inject IMU fusion between
``_aggregate_features()`` and ``_predict_camera()``.
"""

import logging
import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from third_party.lingbot_map.models.gct_stream import GCTStream
from .imu_encoder import AirIMUEncoder, AirIMUConfig, create_airimu_encoder
from .imu_fusion_layers import IMUCrossAttentionLayer, IMUPriorAdaLN, IMUPoseCorrectionHead
from .fusion_modules import PosePriorEncoder

logger = logging.getLogger(__name__)

# Reuse the KV debug helpers from gct_stream
_KV_DEBUG = os.environ.get("LINGBOT_DEBUG_KV", "")


def _parse_kv_debug_interval(val: str) -> int:
    if not val:
        return 0
    try:
        n = int(val)
    except ValueError:
        return 1
    return max(0, n)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LingBotVIOConfig:
    """Configuration for the LingBot-VIO model."""

    # IMU encoder
    imu_checkpoint: Optional[str] = None
    freeze_imu: bool = True
    imu_dim: int = 256
    imu_propcov: bool = True

    # Visual encoder (inherited from GCTStream)
    visual_dim: int = 1024          # embed_dim; aggregator outputs 2*visual_dim
    freeze_visual: bool = True

    # Fusion switches
    use_imu_cross_attention: bool = True
    use_imu_prior_adaln: bool = True
    use_pose_correction: bool = True

    # Fusion dimensions (derived from visual_dim)
    # aggregator outputs tokens with dim 2*visual_dim = 2048
    # These are set automatically in __post_init__
    token_dim: int = 0              # 0 → auto (2 * visual_dim)
    pose_enc_dim: int = 0           # 0 → auto (visual_dim)

    # Cross-attention hyper-parameters
    cross_attn_num_heads: int = 8
    cross_attn_dropout: float = 0.1

    # AdaLN hyper-parameters
    adaln_condition_dim: int = 0    # 0 → auto (visual_dim)

    # Pose correction head
    pose_correction_hidden_dim: int = 512
    pose_correction_dropout: float = 0.1

    # Pose prior encoder
    pose_prior_use_covariance: bool = True
    pose_prior_dropout: float = 0.1

    def __post_init__(self):
        if self.token_dim == 0:
            self.token_dim = 2 * self.visual_dim
        if self.pose_enc_dim == 0:
            self.pose_enc_dim = self.visual_dim
        if self.adaln_condition_dim == 0:
            self.adaln_condition_dim = self.visual_dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LingBotVIO(GCTStream):
    """
    LingBot-VIO: Streaming visual-inertial odometry with IMU fusion.

    Subclasses GCTStream and injects IMU fusion between the visual
    aggregation stage and the camera/depth prediction stages.

    The forward pass follows four stages:
        1. Visual aggregation (inherited from GCTStream)
        2. IMU fusion (cross-attention + AdaLN conditioning)
        3. Camera & depth prediction (inherited, but with fused tokens)
        4. IMU pose correction (additive delta to visual pose prediction)
    """

    def __init__(
        self,
        vio_config: Optional[LingBotVIOConfig] = None,
        # GCTStream kwargs — forwarded to super().__init__
        **kwargs,
    ):
        self.vio_config = vio_config or LingBotVIOConfig()

        # Store VIO-specific params before super().__init__
        self._freeze_visual = self.vio_config.freeze_visual
        self._freeze_imu = self.vio_config.freeze_imu

        super().__init__(embed_dim=self.vio_config.visual_dim, **kwargs)

        # ----- IMU encoder -----
        imu_cfg = AirIMUConfig(propcov=self.vio_config.imu_propcov)
        self.imu_encoder = create_airimu_encoder(
            checkpoint_path=self.vio_config.imu_checkpoint,
            freeze=self.vio_config.freeze_imu,
            config=imu_cfg,
        )

        # ----- Pose prior encoder -----
        self.imu_pose_encoder = PosePriorEncoder(
            output_dim=self.vio_config.pose_enc_dim,
            use_covariance=self.vio_config.pose_prior_use_covariance,
            dropout=self.vio_config.pose_prior_dropout,
        )

        # ----- IMU cross-attention -----
        if self.vio_config.use_imu_cross_attention:
            self.imu_cross_attn = IMUCrossAttentionLayer(
                visual_dim=self.vio_config.token_dim,
                imu_dim=self.vio_config.imu_dim,
                num_heads=self.vio_config.cross_attn_num_heads,
                dropout=self.vio_config.cross_attn_dropout,
            )
        else:
            self.imu_cross_attn = None

        # ----- IMU prior AdaLN -----
        if self.vio_config.use_imu_prior_adaln:
            self.imu_prior_adaln = IMUPriorAdaLN(
                dim=self.vio_config.token_dim,
                condition_dim=self.vio_config.adaln_condition_dim,
            )
        else:
            self.imu_prior_adaln = None

        # ----- Pose correction head -----
        if self.vio_config.use_pose_correction:
            self.pose_correction_head = IMUPoseCorrectionHead(
                visual_dim=self.vio_config.token_dim,
                imu_pose_dim=self.vio_config.pose_enc_dim,
                hidden_dim=self.vio_config.pose_correction_hidden_dim,
                pose_dim=9,
            )
        else:
            self.pose_correction_head = None

        # Apply freezing after all sub-modules are built
        if self._freeze_visual:
            self.freeze_visual_backbone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        imu_data: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional IMU fusion.

        Args:
            images: Input images [B, S, 3, H, W]
            imu_data: Optional dict with keys:
                - acc:  [B, T, 3] accelerometer readings
                - gyro: [B, T, 3] gyroscope readings
                - dt:   [B, T] time deltas
                - init_state: Optional dict with 'pos', 'vel', 'rot'
            **kwargs: Forwarded to GCTStream forward (num_frame_for_scale,
                      sliding_window_size, causal_inference, etc.)

        Returns:
            Dictionary containing predictions:
                - pose_enc: [B, S, 9] camera pose encoding (IMU-corrected if available)
                - visual_pose_enc: [B, S, 9] visual-only pose (before IMU correction)
                - depth: [B, S, H, W, 1]
                - depth_conf: [B, S, H, W]
                - world_points: [B, S, H, W, 3]
                - world_points_conf: [B, S, H, W]
                - imu_correction: [B, S, 9] (if IMU data provided)
        """
        # 1. Visual aggregation (inherited)
        aggregated_tokens_list, patch_start_idx = self._aggregate_features(
            images, **kwargs
        )

        # 2. IMU fusion (NEW)
        imu_outputs = None
        pose_encoding = None
        confidence = None

        if imu_data is not None:
            imu_outputs = self.imu_encoder(
                acc=imu_data['acc'],
                gyro=imu_data['gyro'],
                dt=imu_data.get('dt', None),
                init_state=imu_data.get('init_state', None),
                return_integrated=True,
            )

            imu_features = imu_outputs['features']  # [B, T', 256]

            # Get relative pose from IMU integration
            # Handle case where PyPose is not available (no 'pos' key)
            if 'pos' in imu_outputs:
                imu_pose = self.imu_encoder.get_relative_pose(imu_outputs)
                pose_encoding, confidence = self.imu_pose_encoder(
                    rel_pos=imu_pose['rel_pos'],
                    rel_rot=imu_pose['rel_rot'],
                    rel_vel=imu_pose.get('rel_vel', None),
                    pos_cov=imu_pose.get('pos_cov', None),
                    rot_cov=imu_pose.get('rot_cov', None),
                )
            else:
                # PyPose not available: use features-only path
                # Create a zero pose encoding as fallback
                B_feat = imu_features.shape[0]
                device = imu_features.device
                dtype = imu_features.dtype
                pose_encoding = torch.zeros(
                    B_feat, self.vio_config.pose_enc_dim,
                    device=device, dtype=dtype,
                )
                confidence = None

            # Apply cross-attention and AdaLN on each level of aggregated tokens
            fused_tokens_list = []
            for tokens in aggregated_tokens_list:
                # tokens: [B, S, P, 2*C] or [B, N, 2*C]
                orig_shape = tokens.shape
                if tokens.dim() == 4:
                    B, S, P, C = tokens.shape
                    tokens = tokens.view(B, S * P, C)
                else:
                    B, N, C = tokens.shape
                    S, P = None, None

                # Cross-attention: visual tokens attend to IMU features
                if self.imu_cross_attn is not None:
                    tokens = self.imu_cross_attn(tokens, imu_features)

                # AdaLN: condition on IMU pose prior
                if self.imu_prior_adaln is not None:
                    tokens = self.imu_prior_adaln(tokens, pose_encoding)

                # Restore original shape
                if S is not None and P is not None:
                    tokens = tokens.view(B, S, P, C)

                fused_tokens_list.append(tokens)

            aggregated_tokens_list = fused_tokens_list

        # 3. Camera & depth prediction (inherited, but with fused tokens)
        predictions = {}

        predictions.update(self._predict_camera(
            aggregated_tokens_list,
            **{k: v for k, v in kwargs.items()
               if k in ('mask', 'causal_inference', 'num_frame_for_scale',
                        'sliding_window_size', 'num_frame_per_block',
                        'gather_outputs')},
        ))

        predictions.update(self._predict_depth(
            aggregated_tokens_list, images, patch_start_idx,
        ))

        predictions.update(self._predict_points(
            aggregated_tokens_list, images, patch_start_idx,
        ))

        predictions.update(self._predict_local_points(
            aggregated_tokens_list, images, patch_start_idx,
        ))

        # 4. IMU pose correction (NEW)
        if imu_outputs is not None and self.pose_correction_head is not None:
            visual_pose_enc = predictions['pose_enc']

            # Default confidence to 1.0 if not available (no covariance)
            if confidence is None:
                B = visual_pose_enc.shape[0]
                S = visual_pose_enc.shape[1]
                confidence = torch.ones(B, S, 1, device=visual_pose_enc.device, dtype=visual_pose_enc.dtype)

            correction, raw_correction = self.pose_correction_head(
                aggregated_tokens_list[-1],
                pose_encoding,
                confidence,
            )

            # Ensure correction matches visual_pose_enc shape
            if correction.shape != visual_pose_enc.shape:
                # correction might be [B, 1, 9] while visual_pose_enc is [B, S, 9]
                if correction.shape[1] == 1 and visual_pose_enc.shape[1] > 1:
                    correction = correction.expand_as(visual_pose_enc)
                elif correction.shape[1] < visual_pose_enc.shape[1]:
                    correction = correction.expand_as(visual_pose_enc)

            predictions['visual_pose_enc'] = visual_pose_enc
            predictions['pose_enc'] = visual_pose_enc + correction
            predictions['imu_correction'] = correction

        if not self.training:
            predictions["images"] = images

        return predictions

    # ------------------------------------------------------------------
    # Streaming inference with IMU
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_streaming_vio(
        self,
        images: torch.Tensor,
        imu_data_per_frame: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
        output_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Streaming VIO inference: process scale frames first, then frame-by-frame
        with IMU data.

        This extends ``inference_streaming()`` from GCTStream to accept per-frame
        IMU data.  When IMU data is provided for a frame, the IMU fusion path is
        activated; otherwise the model falls back to pure visual mode for that
        frame.

        Args:
            images: Input images [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
            imu_data_per_frame: Optional dict mapping frame index to IMU data
                dict with keys 'acc', 'gyro', 'dt', (optional) 'init_state'.
                If None, runs in pure visual mode.
            num_scale_frames: Number of initial frames for scale estimation.
            keyframe_interval: Every N-th frame (after scale frames) is a keyframe.
            output_device: Device for output tensors (e.g. CPU for long sequences).

        Returns:
            Dictionary containing predictions for all frames:
                - pose_enc: [B, S, 9]
                - visual_pose_enc: [B, S, 9] (if IMU used)
                - imu_correction: [B, S, 9] (if IMU used)
                - depth: [B, S, H, W, 1]
                - depth_conf: [B, S, H, W]
                - world_points: [B, S, H, W, 3]
                - world_points_conf: [B, S, H, W]
        """
        # Normalise input shape
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape

        scale_frames = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale
        scale_frames = min(scale_frames, S)

        def _to_out(t: torch.Tensor) -> torch.Tensor:
            if output_device is not None:
                return t.to(output_device)
            return t

        # Clean KV caches before starting
        self.clean_kv_cache()

        _model_device = next(self.parameters()).device

        # Helper: get IMU data for a range of frames
        def _get_imu_data(start_idx: int, end_idx: int) -> Optional[Dict[str, torch.Tensor]]:
            if imu_data_per_frame is None:
                return None
            # Collect IMU data for frames [start_idx, end_idx)
            collected = {}
            for i in range(start_idx, end_idx):
                if i in imu_data_per_frame:
                    for k, v in imu_data_per_frame[i].items():
                        if k not in collected:
                            collected[k] = []
                        collected[k].append(v)
            if not collected:
                return None
            # Stack along the temporal dimension
            result = {}
            for k, vs in collected.items():
                if k == 'init_state':
                    # Use the first init_state
                    result[k] = vs[0]
                else:
                    result[k] = torch.cat(vs, dim=1) if all(v.dim() == 3 for v in vs) else vs[0]
            return result

        # Phase 1: Process scale frames together
        logger.info(f'Processing {scale_frames} scale frames (VIO)...')
        scale_images = images[:, :scale_frames].to(_model_device, non_blocking=True)
        scale_imu = _get_imu_data(0, scale_frames)

        torch.compiler.cudagraph_mark_step_begin()
        scale_output = self.forward(
            scale_images,
            imu_data=scale_imu,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )

        # Initialise output lists
        all_pose_enc = [_to_out(scale_output["pose_enc"])]
        all_visual_pose_enc = []
        all_imu_correction = []
        if "visual_pose_enc" in scale_output:
            all_visual_pose_enc.append(_to_out(scale_output["visual_pose_enc"]))
        if "imu_correction" in scale_output:
            all_imu_correction.append(_to_out(scale_output["imu_correction"]))
        all_depth = [_to_out(scale_output["depth"])] if "depth" in scale_output else []
        all_depth_conf = [_to_out(scale_output["depth_conf"])] if "depth_conf" in scale_output else []
        all_world_points = [_to_out(scale_output["world_points"])] if "world_points" in scale_output else []
        all_world_points_conf = [_to_out(scale_output["world_points_conf"])] if "world_points_conf" in scale_output else []
        del scale_output

        # Phase 2: Process remaining frames one-by-one
        pbar = tqdm(
            range(scale_frames, S),
            desc='Streaming VIO inference',
            initial=scale_frames,
            total=S,
        )
        dbg_every = _parse_kv_debug_interval(_KV_DEBUG)

        for i in pbar:
            frame_image = images[:, i:i+1].to(_model_device, non_blocking=True)
            frame_imu = _get_imu_data(i, i + 1)

            is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

            if not is_keyframe:
                self._set_skip_append(True)

            torch.compiler.cudagraph_mark_step_begin()
            frame_output = self.forward(
                frame_image,
                imu_data=frame_imu,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )

            if not is_keyframe:
                self._set_skip_append(False)

            all_pose_enc.append(_to_out(frame_output["pose_enc"]))
            if "visual_pose_enc" in frame_output:
                all_visual_pose_enc.append(_to_out(frame_output["visual_pose_enc"]))
            if "imu_correction" in frame_output:
                all_imu_correction.append(_to_out(frame_output["imu_correction"]))
            if "depth" in frame_output:
                all_depth.append(_to_out(frame_output["depth"]))
            if "depth_conf" in frame_output:
                all_depth_conf.append(_to_out(frame_output["depth_conf"]))
            if "world_points" in frame_output:
                all_world_points.append(_to_out(frame_output["world_points"]))
            if "world_points_conf" in frame_output:
                all_world_points_conf.append(_to_out(frame_output["world_points_conf"]))
            del frame_output

        # Free GPU memory before concatenation
        if output_device is not None:
            images_out = _to_out(images)
            del images
            self.clean_kv_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            images_out = images

        # Concatenate all predictions
        predictions = {
            "pose_enc": torch.cat(all_pose_enc, dim=1),
        }
        del all_pose_enc

        if all_visual_pose_enc:
            predictions["visual_pose_enc"] = torch.cat(all_visual_pose_enc, dim=1)
        del all_visual_pose_enc

        if all_imu_correction:
            predictions["imu_correction"] = torch.cat(all_imu_correction, dim=1)
        del all_imu_correction

        if all_depth:
            predictions["depth"] = torch.cat(all_depth, dim=1)
        del all_depth

        if all_depth_conf:
            predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)
        del all_depth_conf

        if all_world_points:
            predictions["world_points"] = torch.cat(all_world_points, dim=1)
        del all_world_points

        if all_world_points_conf:
            predictions["world_points_conf"] = torch.cat(all_world_points_conf, dim=1)
        del all_world_points_conf

        predictions["images"] = images_out

        if self.pred_normalization:
            predictions = self._normalize_predictions(predictions)

        return predictions

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def clean_kv_cache(self):
        """Clean KV caches in aggregator, camera head, and IMU encoder state."""
        super().clean_kv_cache()
        # Reset IMU encoder GRU hidden states if present
        if hasattr(self.imu_encoder, 'gru1'):
            # GRU hidden states are not persistent, but we can reset
            # any cached integration state
            pass

    # ------------------------------------------------------------------
    # Freezing utilities for two-stage training
    # ------------------------------------------------------------------

    def freeze_visual_backbone(self):
        """Freeze the visual backbone (aggregator + patch embedding)."""
        # Freeze aggregator
        for param in self.aggregator.parameters():
            param.requires_grad = False
        logger.info("Visual backbone (aggregator) frozen")

    def unfreeze_camera_head(self):
        """Unfreeze the camera head for second-stage fine-tuning."""
        if self.camera_head is not None:
            for param in self.camera_head.parameters():
                param.requires_grad = True
            logger.info("Camera head unfrozen")

    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end training."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    # ------------------------------------------------------------------
    # Trainable parameter groups
    # ------------------------------------------------------------------

    def get_trainable_params(self, stage: int = 1) -> List[Dict[str, Any]]:
        """
        Return parameter groups for the given training stage.

        Stage 1 (fusion only):
            Only the IMU fusion modules are trainable:
            - imu_cross_attn
            - imu_prior_adaln
            - imu_pose_encoder
            - pose_correction_head

        Stage 2 (camera head + fusion):
            Camera head is additionally unfrozen.

        Stage 3 (full fine-tuning):
            All parameters are trainable.

        Args:
            stage: Training stage (1, 2, or 3)

        Returns:
            List of parameter group dicts suitable for an optimiser.
        """
        fusion_params = []

        if self.imu_cross_attn is not None:
            fusion_params.extend(self.imu_cross_attn.parameters())
        if self.imu_prior_adaln is not None:
            fusion_params.extend(self.imu_prior_adaln.parameters())
        fusion_params.extend(self.imu_pose_encoder.parameters())
        if self.pose_correction_head is not None:
            fusion_params.extend(self.pose_correction_head.parameters())

        if stage == 1:
            return [{"params": fusion_params, "lr_scale": 1.0}]

        elif stage == 2:
            camera_params = list(self.camera_head.parameters()) if self.camera_head else []
            return [
                {"params": fusion_params, "lr_scale": 1.0},
                {"params": camera_params, "lr_scale": 0.1},
            ]

        elif stage == 3:
            all_params = list(self.parameters())
            return [{"params": all_params, "lr_scale": 1.0}]

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def load_vio_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load a LingBot-VIO checkpoint.

        Handles both full VIO checkpoints and GCTStream-only checkpoints
        (in which case the fusion modules are randomly initialised).

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        logger.info(f"Loaded VIO checkpoint from {checkpoint_path}")

    def load_visual_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load only the visual (GCTStream) portion from a checkpoint.

        This is useful for initialising the visual backbone from a
        pretrained GCTStream model before training the fusion modules.

        Args:
            checkpoint_path: Path to GCTStream checkpoint
            strict: Whether to strictly enforce key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter to only visual keys (aggregator, camera_head, depth_head, etc.)
        visual_keys = {
            k for k in self.state_dict().keys()
            if k.startswith(('aggregator.', 'camera_head.', 'depth_head.',
                             'point_head.', 'local_point_head.'))
        }
        visual_state_dict = {k: v for k, v in state_dict.items() if k in visual_keys}

        missing, unexpected = self.load_state_dict(visual_state_dict, strict=False)
        loaded = len(visual_state_dict)
        logger.info(
            f"Loaded {loaded} visual params from {checkpoint_path} "
            f"(missing {len(missing)}, unexpected {len(unexpected)})"
        )
