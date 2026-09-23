"""
AirIMU Encoder Wrapper for IMU-LingBot Fusion.

This module wraps the AirIMU CodeNet encoder to extract IMU features and
relative pose estimates for injection into the visual Transformer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class AirIMUConfig:
    """Configuration for AirIMU encoder."""
    gyro_std: float = np.pi / 180  # rad/s
    acc_std: float = 0.1  # m/s^2
    propcov: bool = True  # Propagate covariance
    gtrot: bool = False  # Use ground truth rotation
    sampling: Optional[int] = None  # Sampling interval for integration
    network: str = "codenet"
    interval: int = 9  # CNN downsampling interval


class CNNEncoder(nn.Module):
    """1D CNN encoder for IMU data."""
    
    def __init__(
        self,
        c_list: list = [6, 32, 64],
        k_list: list = [7, 7],
        s_list: list = [3, 3],
        p_list: Optional[list] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if p_list is None:
            p_list = [k // 2 for k in k_list]
        
        layers = []
        for i in range(len(c_list) - 1):
            layers.extend([
                nn.Conv1d(c_list[i], c_list[i + 1], k_list[i],
                         stride=s_list[i], padding=p_list[i]),
                nn.BatchNorm1d(c_list[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] IMU features
        Returns:
            [B, T', C'] encoded features
        """
        return self.net(x.transpose(-1, -2)).transpose(-1, -2)


class AirIMUEncoder(nn.Module):
    """
    Standalone AirIMU encoder that extracts IMU features and relative poses.
    
    This encoder wraps the CNN + GRU architecture from AirIMU's CodeNet,
    providing:
    - Encoded IMU features (256-dim) for cross-attention fusion
    - IMU corrections (acc + gyro biases)
    - Covariance estimates for uncertainty-aware fusion
    - Integrated relative poses via PyPose IMUPreintegrator
    
    The encoder can be loaded with pretrained weights and frozen for
    training only the fusion modules.
    """
    
    def __init__(
        self,
        config: Optional[AirIMUConfig] = None,
        hidden_dim: int = 256,
        freeze: bool = True
    ):
        super().__init__()
        
        self.config = config or AirIMUConfig()
        self.hidden_dim = hidden_dim
        self.freeze = freeze
        
        # CNN downsampling interval (from CodeNet)
        self.interval = self.config.interval
        self.inter_head = int(np.floor(self.interval / 2.0))
        self.inter_tail = self.interval - self.inter_head
        
        # Register std buffers
        self.register_buffer('gyro_std', torch.tensor(self.config.gyro_std))
        self.register_buffer('acc_std', torch.tensor(self.config.acc_std))
        
        # CNN encoder: [B, T, 6] -> [B, T/9, 64]
        self.cnn = CNNEncoder(
            c_list=[6, 32, 64],
            k_list=[7, 7],
            s_list=[3, 3]
        )
        
        # GRU layers: 64 -> 128 -> 256
        self.gru1 = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Correction decoders
        self.acc_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.gyro_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
        # Covariance decoders
        self.acc_cov_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        self.gyro_cov_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        
        # IMU preintegrator (optional, for getting relative poses)
        self._integrator = None
    
    @property
    def integrator(self):
        """Lazy load integrator to avoid import issues."""
        if self._integrator is None:
            try:
                import pypose as pp
                self._integrator = pp.module.IMUPreintegrator(
                    prop_cov=self.config.propcov,
                    reset=True
                )
            except ImportError:
                print("Warning: PyPose not available. Integration disabled.")
        return self._integrator
    
    def encode(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Encode raw IMU data to features.
        
        Args:
            imu_data: [B, T, 6] concatenated [acc, gyro]
        
        Returns:
            features: [B, T', 256] encoded features
        """
        x = self.cnn(imu_data)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def decode_corrections(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode corrections from features.
        
        Args:
            features: [B, T', 256] encoded features
        
        Returns:
            acc_correction: [B, T', 3]
            gyro_correction: [B, T', 3]
        """
        acc_corr = self.acc_decoder(features) * self.acc_std
        gyro_corr = self.gyro_decoder(features) * self.gyro_std
        return acc_corr, gyro_corr
    
    def decode_covariance(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode covariance estimates from features.
        
        Args:
            features: [B, T', 256] encoded features
        
        Returns:
            acc_cov: [B, T', 3] diagonal covariance
            gyro_cov: [B, T', 3] diagonal covariance
        """
        acc_cov = torch.exp(self.acc_cov_decoder(features) - 5.0)
        gyro_cov = torch.exp(self.gyro_cov_decoder(features) - 5.0)
        return acc_cov, gyro_cov
    
    def _update_corrections(
        self,
        to_update: torch.Tensor,
        feat: torch.Tensor,
        frame_len: int
    ) -> torch.Tensor:
        """
        Spread corrections from downsampled features to full resolution.
        
        This replicates the _update method from CodeNet which distributes
        each correction value to the surrounding IMU samples within the
        interval window.
        """
        def _clip(x, l):
            return max(0, min(x, l))
        
        feat_range = int(np.ceil((frame_len - self.inter_head) / self.interval)) + 1
        
        for i in range(feat_range):
            s_p = _clip(i * self.interval - self.inter_head, frame_len)
            e_p = _clip(i * self.interval + self.inter_tail, frame_len)
            idx = _clip(i, feat.shape[1] - 1)
            to_update[:, s_p:e_p, :] += feat[:, idx:idx + 1, :]
        
        return to_update
    
    def forward(
        self,
        acc: torch.Tensor,
        gyro: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        init_state: Optional[Dict[str, torch.Tensor]] = None,
        return_integrated: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the IMU encoder.
        
        Args:
            acc: [B, T, 3] accelerometer readings
            gyro: [B, T, 3] gyroscope readings
            dt: [B, T] time deltas (required for integration)
            init_state: Optional initial state dict with 'pos', 'vel', 'rot'
            return_integrated: Whether to return integrated poses
        
        Returns:
            Dictionary containing:
            - features: [B, T', 256] encoded IMU features
            - correction_acc: [B, T-interval, 3] accelerometer corrections
            - correction_gyro: [B, T-interval, 3] gyroscope corrections
            - corrected_acc: [B, T-interval, 3] corrected accelerometer
            - corrected_gyro: [B, T-interval, 3] corrected gyroscope
            - acc_cov: [B, T-interval, 3] accelerometer covariance (if propcov)
            - gyro_cov: [B, T-interval, 3] gyroscope covariance (if propcov)
            - pos: [B, T-interval, 3] integrated position (if return_integrated)
            - vel: [B, T-interval, 3] integrated velocity (if return_integrated)
            - rot: [B, T-interval, 4] integrated rotation (if return_integrated)
            - cov: [B, T-interval, 9, 9] state covariance (if return_integrated and propcov)
        """
        B, T, _ = acc.shape
        frame_len = T - self.interval
        
        # Concatenate and encode
        imu_data = torch.cat([acc, gyro], dim=-1)  # [B, T, 6]
        features = self.encode(imu_data)  # [B, T', 256]
        
        # Skip first feature (corresponds to padding region)
        features_valid = features[:, 1:, :]
        
        # Decode corrections at downsampled resolution
        acc_corr_ds, gyro_corr_ds = self.decode_corrections(features_valid)
        
        # Upsample corrections to full resolution
        zero_acc = torch.zeros(B, frame_len, 3, device=acc.device, dtype=acc.dtype)
        zero_gyro = torch.zeros(B, frame_len, 3, device=gyro.device, dtype=gyro.dtype)
        
        correction_acc = self._update_corrections(zero_acc.clone(), acc_corr_ds, frame_len)
        correction_gyro = self._update_corrections(zero_gyro.clone(), gyro_corr_ds, frame_len)
        
        # Apply corrections
        corrected_acc = acc[:, self.interval:, :] + correction_acc
        corrected_gyro = gyro[:, self.interval:, :] + correction_gyro
        
        result = {
            'features': features,
            'correction_acc': correction_acc,
            'correction_gyro': correction_gyro,
            'corrected_acc': corrected_acc,
            'corrected_gyro': corrected_gyro,
        }
        
        # Decode covariance if enabled
        if self.config.propcov:
            acc_cov_ds, gyro_cov_ds = self.decode_covariance(features_valid)
            
            acc_cov = self._update_corrections(
                torch.zeros_like(correction_acc),
                acc_cov_ds,
                frame_len
            )
            gyro_cov = self._update_corrections(
                torch.zeros_like(correction_gyro),
                gyro_cov_ds,
                frame_len
            )
            
            result['acc_cov'] = acc_cov
            result['gyro_cov'] = gyro_cov
        
        # Integrate if requested and integrator available
        if return_integrated and dt is not None and self.integrator is not None:
            import pypose as pp
            dt_valid = dt[:, self.interval:]
            
            # Ensure integrator buffers are on correct device and dtype (double for integration)
            self._integrator = self._integrator.to(device=acc.device, dtype=torch.float64)
            
            if init_state is None:
                init_state = {
                    'pos': torch.zeros(B, 1, 3, device=acc.device, dtype=acc.dtype),
                    'vel': torch.zeros(B, 1, 3, device=acc.device, dtype=acc.dtype),
                    'rot': pp.identity_SO3(B, 1, dtype=acc.dtype, device=acc.device),
                }
            
            # Convert to double for integration (PyPose requirement)
            integrate_state = self.integrator(
                init_state={
                    'pos': init_state['pos'].double(),
                    'vel': init_state['vel'].double(),
                    'rot': init_state['rot'].double() if isinstance(init_state['rot'], pp.LieTensor) 
                           else pp.identity_SO3(B, 1, dtype=torch.float64, device=acc.device),
                },
                dt=dt_valid.unsqueeze(-1).double(),
                gyro=corrected_gyro.double(),
                acc=corrected_acc.double(),
                rot=None,
                acc_cov=result.get('acc_cov', torch.zeros_like(corrected_acc)).double() if self.config.propcov else None,
                gyro_cov=result.get('gyro_cov', torch.zeros_like(corrected_gyro)).double() if self.config.propcov else None,
            )
            
            result['pos'] = integrate_state['pos'].to(acc.dtype)
            result['vel'] = integrate_state['vel'].to(acc.dtype)
            result['rot'] = integrate_state['rot'].to(acc.dtype)
            
            if self.config.propcov and 'cov' in integrate_state:
                result['cov'] = integrate_state['cov'].to(acc.dtype)
        
        return result
    
    def get_relative_pose(
        self,
        result: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract relative pose (last frame relative to first) from integration result.
        
        Args:
            result: Output from forward()
        
        Returns:
            Dictionary with:
            - rel_pos: [B, 3] relative position
            - rel_rot: [B, 4] relative rotation (quaternion)
            - rel_vel: [B, 3] relative velocity
            - pos_cov: [B, 3] position uncertainty (diagonal)
            - rot_cov: [B, 3] rotation uncertainty (diagonal)
        """
        rel_pose = {
            'rel_pos': result['pos'][:, -1, :],  # [B, 3]
            'rel_vel': result['vel'][:, -1, :],  # [B, 3]
        }
        
        if 'rot' in result:
            rel_pose['rel_rot'] = result['rot'][:, -1, :]  # [B, 4]
        
        if 'cov' in result:
            raw_cov = result['cov']
            if raw_cov.ndim == 4:
                cov = raw_cov[:, -1, :, :]  # [B, T, 9, 9] -> [B, 9, 9]
            else:
                cov = raw_cov  # already [B, 9, 9]
            rel_pose['pos_cov'] = torch.diagonal(cov[:, :3, :3], dim1=-2, dim2=-1)  # [B, 3]
            rel_pose['rot_cov'] = torch.diagonal(cov[:, 3:6, 3:6], dim1=-2, dim2=-1)  # [B, 3]
        
        return rel_pose
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """
        Load pretrained AirIMU weights.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce that the keys match
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Map AirIMU CodeNet keys to our keys
        key_mapping = {
            'cnn.': 'cnn.',
            'gru1.': 'gru1.',
            'gru2.': 'gru2.',
            'accdecoder.': 'acc_decoder.',
            'gyrodecoder.': 'gyro_decoder.',
            'acccov_decoder.': 'acc_cov_decoder.',
            'gyrocov_decoder.': 'gyro_cov_decoder.',
        }
        
        mapped_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for old, new in key_mapping.items():
                if k.startswith(old):
                    new_key = k.replace(old, new, 1)
                    break
            mapped_state_dict[new_key] = v
        
        # Filter to only include keys that exist in this model
        model_keys = set(self.state_dict().keys())
        filtered_state_dict = {
            k: v for k, v in mapped_state_dict.items()
            if k in model_keys
        }
        
        missing = model_keys - set(filtered_state_dict.keys())
        if missing and strict:
            print(f"Warning: Missing keys: {missing}")
        
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} parameters from {checkpoint_path}")
    
    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        print("AirIMU encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        print("AirIMU encoder unfrozen")


def create_airimu_encoder(
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    config: Optional[AirIMUConfig] = None
) -> AirIMUEncoder:
    """
    Factory function to create an AirIMU encoder.
    
    Args:
        checkpoint_path: Optional path to pretrained weights
        freeze: Whether to freeze the encoder
        config: Optional configuration
    
    Returns:
        Configured AirIMUEncoder instance
    """
    encoder = AirIMUEncoder(config=config, freeze=freeze)
    
    if checkpoint_path is not None:
        encoder.load_pretrained(checkpoint_path)
    
    if freeze:
        encoder.freeze_encoder()
    
    return encoder


def create_codenet_from_third_party(
    conf_dict: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
):
    """
    Create an original AirIMU CodeNet using the bundled third_party copy.

    This provides full compatibility with the original training pipeline and
    checkpoint format. The returned module can be used directly for inference
    or as the IMU branch in the fusion model.

    Args:
        conf_dict: Configuration dictionary compatible with CodeNet.
                   Required keys: ``propcov``, ``gtrot``, ``network``.
                   Optional keys: ``gyro_std``, ``acc_std``, ``sampling``,
                   ``gravity``, ``posonly``.
        checkpoint_path: Path to a ``best_model.ckpt`` file saved by AirIMU
                         ``train.py``.
        freeze: Freeze parameters after loading.

    Returns:
        An ``nn.Module`` (``CodeNet`` or ``CodeNetKITTI``) ready for use.
    """
    from types import SimpleNamespace
    from third_party.airimu.model.code import CodeNet, CodeNetKITTI

    class _DictNamespace(SimpleNamespace):
        """Minimal namespace that supports ``in`` checks and ``keys()``."""
        def __contains__(self, item):
            return hasattr(self, item)
        def keys(self):
            return vars(self).keys()
        def __getitem__(self, item):
            return getattr(self, item)

    conf = _DictNamespace(**conf_dict)

    network_cls = CodeNetKITTI if conf_dict.get("network") == "codenetkitti" else CodeNet
    model = network_cls(conf)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"Loaded original CodeNet from {checkpoint_path}")

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    return model
