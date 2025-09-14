"""
最终版本的自适应FNO实现
采用最简单的方法：在FNO的forward过程中添加自适应频率选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional

from .fno import FNO, FNO2d

Number = Union[float, int]


class FrequencyGatingModule(nn.Module):
    """
    频率门控模块，用于自适应选择重要的频率分量
    """
    def __init__(self, 
                 n_modes: Tuple[int, ...],
                 hidden_channels: int,
                 n_experts: int = 4,
                 temperature: float = 1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_experts = n_experts
        self.temperature = temperature
        self.n_dim = len(n_modes)

        
        
        # 频率带边界参数（可学习）
        self.band_boundaries = nn.Parameter(torch.rand(n_experts - 1) * 0.6 + 0.2)
        
        # 门控网络：基于空间域特征决定频率权重
        if self.n_dim == 1:
            self.gating_network = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, n_experts),
                nn.Softmax(dim=-1)
            )
        else:
            self.gating_network = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, n_experts),
                nn.Softmax(dim=-1)
            )
        
    def forward(self, x):
        """
        x: 空间域信号 [batch, channels, height, width]
        返回: 频率权重 [batch, n_experts]
        """
        # 计算门控权重
        gating_scores = self.gating_network(x)  # [batch, n_experts]
        return gating_scores
    
    def apply_frequency_gating(self, x_fft, gating_scores):
        """
        将频率门控应用到FFT结果上
        x_fft: 频域信号 [batch, channels, freq_dims...]
        gating_scores: 门控权重 [batch, n_experts]
        """
        if self.n_dim == 1:
            batch_size, channels, freq_w = x_fft.shape
            freq_h = 1
        else:
            batch_size, channels, freq_h, freq_w = x_fft.shape
        
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
        
        # 简化：只在最后一个维度（频率维度）上应用门控
        total_freq_size = freq_w
        
        for i in range(self.n_experts):
            start_idx = int(boundaries[i] * total_freq_size)
            end_idx = int(boundaries[i + 1] * total_freq_size)
            
            if end_idx > start_idx:
                # 创建掩码
                mask = torch.zeros_like(x_fft)
                mask[..., start_idx:end_idx] = 1.0
                
                # 应用专家权重
                expert_weight = gating_scores[:, i:i+1]  # [batch, 1]
                expert_weight = expert_weight.view(batch_size, 1, 1, 1)  # [batch, 1, 1, 1]
                
                weighted_fft = weighted_fft + mask * x_fft * expert_weight
        
        return weighted_fft


class AdaptiveFNO(FNO):
    """
    自适应FNO，在标准FNO基础上添加自适应频率选择
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
        
        # 创建频率门控模块
        self.frequency_gating = FrequencyGatingModule(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_experts=n_experts,
            temperature=temperature
        )
        
    def forward(self, x, output_shape=None, **kwargs):
        """
        自适应FNO的前向传播
        """
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # 位置编码
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # Lifting
        x = self.lifting(x)
        
        # 计算频率门控权重（基于lifting后的特征）
        gating_scores = self.frequency_gating(x)

        # Domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # FNO blocks with adaptive frequency selection
        for layer_idx in range(self.n_layers):
            # 在每个FNO block之前应用自适应频率选择
            x = self._adaptive_fno_block(x, layer_idx, gating_scores, output_shape[layer_idx])

        # Remove padding
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # Projection
        x = self.projection(x)

        return x
    
    def _adaptive_fno_block(self, x, layer_idx, gating_scores, output_shape):
        """
        自适应FNO块，在频域应用门控
        """
        # 获取原始FNO block的组件
        fno_block = self.fno_blocks
        
        # 执行FNO block的前半部分（到频域变换）
        x_skip_fno = fno_block.fno_skips[layer_idx](x)
        x_skip_fno = fno_block.convs[layer_idx].transform(x_skip_fno, output_shape=output_shape)

        if fno_block.use_channel_mlp:  
            x_skip_channel_mlp = fno_block.channel_mlp_skips[layer_idx](x)
            x_skip_channel_mlp = fno_block.convs[layer_idx].transform(x_skip_channel_mlp, output_shape=output_shape)

        # 执行频谱卷积，但在频域应用自适应门控
        x_fno = self._adaptive_spectral_conv(x, fno_block.convs[layer_idx], gating_scores, output_shape)
        
        # 应用激活和归一化
        if fno_block.norm is not None:
            x_fno = fno_block.norm[2*layer_idx](x_fno)
        
        x_fno = fno_block.non_linearity(x_fno)
        x = x_fno + x_skip_fno

        if fno_block.use_channel_mlp:
            if fno_block.norm is not None:
                x = fno_block.norm[2*layer_idx + 1](x)
            x = fno_block.channel_mlp[layer_idx](x) + x_skip_channel_mlp

        return x
    
    def _adaptive_spectral_conv(self, x, spectral_conv, gating_scores, output_shape):
        """
        自适应频谱卷积：在频域应用门控后再进行卷积
        """
        batchsize, channels, *mode_sizes = x.shape
        
        fft_size = list(mode_sizes)
        if not spectral_conv.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-spectral_conv.order, 0))
        
        # FFT变换
        if spectral_conv.complex_data:
            x_fft = torch.fft.fftn(x, norm=spectral_conv.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x_fft = torch.fft.rfftn(x, norm=spectral_conv.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]
        
        if spectral_conv.order > 1:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)
        
        # 应用自适应频率门控
        x_fft_gated = self.frequency_gating.apply_frequency_gating(x_fft, gating_scores)
        
        # 将门控后的频域信号转换回空间域
        if spectral_conv.order > 1:
            x_fft_gated = torch.fft.fftshift(x_fft_gated, dim=dims_to_fft_shift)
        
        if spectral_conv.complex_data:
            x_gated = torch.fft.ifftn(x_fft_gated, s=mode_sizes, dim=fft_dims, norm=spectral_conv.fft_norm)
        else:
            x_gated = torch.fft.irfftn(x_fft_gated, s=mode_sizes, dim=fft_dims, norm=spectral_conv.fft_norm)
        
        # 确保是实数
        if not spectral_conv.complex_data:
            x_gated = x_gated.real
        
        # 对门控后的信号应用标准频谱卷积
        result = spectral_conv(x_gated, output_shape)
        
        return result


class AdaptiveFNO2d(AdaptiveFNO):
    """
    2D自适应FNO
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
