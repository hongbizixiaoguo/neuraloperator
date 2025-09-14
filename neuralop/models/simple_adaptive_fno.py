"""
简化版自适应FNO实现
避免复杂的架构修改，直接在标准FNO基础上添加自适应频率选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional

from .fno import FNO, FNO2d
from ..layers.spectral_convolution import SpectralConv

Number = Union[float, int]


class AdaptiveFrequencySelector(nn.Module):
    """
    自适应频率选择器，用于动态选择重要的频率分量
    """
    def __init__(self, 
                 n_modes: Tuple[int, ...],
                 hidden_channels: int,
                 n_experts: int = 4,
                 temperature: float = 1.0):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_experts = n_experts
        self.temperature = temperature
        self.n_dim = len(n_modes)
        
        # 频率带边界参数（可学习）
        self.band_boundaries = nn.Parameter(torch.rand(n_experts - 1) * 0.5 + 0.25)
        
        # 门控网络：基于空间域特征决定频率权重
        self.gating_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self.n_dim == 2 else nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, n_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x_spatial, x_fft):
        """
        x_spatial: 空间域信号 [batch, channels, spatial_dims...]
        x_fft: 频域信号 [batch, channels, freq_dims...]
        返回: 加权后的频域信号
        """
        batch_size, channels = x_fft.shape[:2]
        freq_shape = x_fft.shape[2:]
        
        # 计算门控权重
        gating_scores = self.gating_network(x_spatial)  # [batch, n_experts]
        
        # 获取排序后的频率带边界
        boundaries = torch.sigmoid(self.band_boundaries)
        boundaries, _ = torch.sort(boundaries)
        
        # 添加起始和结束边界
        boundaries = torch.cat([
            torch.zeros(1, device=boundaries.device),
            boundaries,
            torch.ones(1, device=boundaries.device)
        ])
        
        # 创建频率掩码并应用权重
        weighted_fft = torch.zeros_like(x_fft)
        total_freq_size = freq_shape[-1]
        
        for i in range(self.n_experts):
            start_idx = int(boundaries[i] * total_freq_size)
            end_idx = int(boundaries[i + 1] * total_freq_size)
            
            if end_idx > start_idx:
                # 创建掩码
                mask = torch.zeros_like(x_fft)
                if self.n_dim == 1:
                    mask[..., start_idx:end_idx] = 1.0
                elif self.n_dim == 2:
                    mask[..., start_idx:end_idx] = 1.0
                else:
                    mask[..., start_idx:end_idx] = 1.0
                
                # 应用专家权重
                expert_weight = gating_scores[:, i:i+1]  # [batch, 1]
                for _ in range(len(freq_shape)):
                    expert_weight = expert_weight.unsqueeze(-1)
                expert_weight = expert_weight.unsqueeze(1)  # 添加channel维度
                
                weighted_fft = weighted_fft + mask * x_fft * expert_weight
        
        return weighted_fft, gating_scores, boundaries


class AdaptiveSpectralConvWrapper(nn.Module):
    """
    自适应频谱卷积包装器，在标准SpectralConv基础上添加自适应频率选择
    """
    def __init__(self, spectral_conv: SpectralConv, n_experts: int = 4, temperature: float = 1.0):
        super().__init__()
        self.spectral_conv = spectral_conv
        self.n_experts = n_experts
        self.temperature = temperature
        
        # 创建自适应频率选择器
        self.adaptive_selector = AdaptiveFrequencySelector(
            n_modes=spectral_conv.n_modes,
            hidden_channels=spectral_conv.in_channels,
            n_experts=n_experts,
            temperature=temperature
        )
        
        # 创建通道适配器（如果需要）
        if spectral_conv.in_channels != spectral_conv.out_channels:
            if len(spectral_conv.n_modes) == 2:  # 2D case
                self.channel_adapter = nn.Conv2d(spectral_conv.in_channels, 
                                                spectral_conv.out_channels, 
                                                kernel_size=1, bias=False)
            else:  # 1D case
                self.channel_adapter = nn.Conv1d(spectral_conv.in_channels, 
                                                spectral_conv.out_channels, 
                                                kernel_size=1, bias=False)
        else:
            self.channel_adapter = None
        
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """
        自适应频谱卷积前向传播
        """
        # 保存原始空间域输入
        x_spatial = x.clone()
        
        # 执行FFT
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        if not self.spectral_conv.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.spectral_conv.order, 0))
        
        if self.spectral_conv.complex_data:
            x_fft = torch.fft.fftn(x, norm=self.spectral_conv.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x_fft = torch.fft.rfftn(x, norm=self.spectral_conv.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]
        
        if self.spectral_conv.order > 1:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)
        
        # 应用自适应频率选择
        x_fft_adaptive, gating_scores, boundaries = self.adaptive_selector(x_spatial, x_fft)
        
        # 直接调用原始SpectralConv的forward方法，但使用自适应选择后的输入
        # 为了简化，我们先将自适应选择的结果转换回空间域，然后调用原始方法
        
        # 临时IFFT回空间域
        if self.spectral_conv.order > 1:
            x_fft_temp = torch.fft.fftshift(x_fft_adaptive, dim=dims_to_fft_shift)
        else:
            x_fft_temp = x_fft_adaptive
            
        if self.spectral_conv.complex_data:
            x_temp = torch.fft.ifftn(x_fft_temp, s=mode_sizes, dim=fft_dims, norm=self.spectral_conv.fft_norm)
        else:
            x_temp = torch.fft.irfftn(x_fft_temp, s=mode_sizes, dim=fft_dims, norm=self.spectral_conv.fft_norm)
        
        # 确保是实数（如果原始数据是实数）
        if not self.spectral_conv.complex_data:
            x_temp = x_temp.real
        
        # 调用原始SpectralConv的forward方法
        result = self.spectral_conv(x_temp, output_shape)
        
        return result
    
    def transform(self, x, output_shape=None):
        """
        变换方法，委托给原始的SpectralConv
        """
        return self.spectral_conv.transform(x, output_shape)


class SimpleAdaptiveFNO(FNO):
    """
    简化版自适应FNO，在标准FNO基础上添加自适应频率选择
    """
    def __init__(self,
                 n_modes: Tuple[int, ...],
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 n_layers: int = 4,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 **kwargs):
        """
        Parameters
        ----------
        n_modes : Tuple[int, ...]
            每个维度保留的最大模式数
        n_experts : int
            频率分解的专家数量（频率带数量）
        temperature : float
            门控网络的温度参数，控制选择的锐度
        """
        # 初始化标准FNO
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **kwargs
        )
        
        self.n_experts = n_experts
        self.temperature = temperature
        
        # 用自适应包装器替换FNO blocks中的SpectralConv
        self._wrap_spectral_convs()
        
    def _wrap_spectral_convs(self):
        """
        用自适应包装器包装FNO blocks中的SpectralConv层
        """
        for i in range(len(self.fno_blocks.convs)):
            original_conv = self.fno_blocks.convs[i]
            wrapped_conv = AdaptiveSpectralConvWrapper(
                spectral_conv=original_conv,
                n_experts=self.n_experts,
                temperature=self.temperature
            )
            self.fno_blocks.convs[i] = wrapped_conv


class SimpleAdaptiveFNO2d(SimpleAdaptiveFNO):
    """
    2D简化版自适应FNO
    """
    def __init__(self,
                 n_modes_height: int,
                 n_modes_width: int,
                 hidden_channels: int,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 n_layers: int = 4,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_experts=n_experts,
            temperature=temperature,
            **kwargs
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width


# 为了向后兼容，创建别名
AdaptiveFNO2d = SimpleAdaptiveFNO2d
