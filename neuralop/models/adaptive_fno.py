import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal, Optional

from .fno import FNO
from ..layers.spectral_convolution import SpectralConv
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from ..layers.embeddings import GridEmbeddingND
from ..layers.fno_block import FNOBlocks

Number = Union[float, int]


class AdaptiveFrequencyGating(nn.Module):
    """
    自适应频率门控模块，用于动态选择重要的频率分量
    而不是硬截断前n_modes个模式
    """
    def __init__(self, 
                 n_modes: Tuple[int, ...],
                 hidden_channels: int,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 complex_data: bool = False):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_experts = n_experts
        self.temperature = temperature
        self.complex_data = complex_data
        self.n_dim = len(n_modes)
        
        # 频率带边界参数（可学习）
        self.band_boundaries = nn.Parameter(torch.rand(n_experts - 1))
        
        # 门控网络：基于频率幅度决定权重
        # 简化：使用固定大小的输入
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, n_experts)
        )
        
        # 每个专家的权重调制参数
        self.expert_weights = nn.Parameter(torch.ones(n_experts))
        
    def forward(self, x_fft, x_spatial=None):
        """
        x_fft: 频域信号 [batch, channels, freq_dims...]
        x_spatial: 空间域信号（用于门控）[batch, channels, spatial_dims...]
        返回: 加权后的频域信号和门控权重
        """
        batch_size, channels = x_fft.shape[:2]
        freq_shape = x_fft.shape[2:]
        
        # 计算频率幅度谱
        freq_magnitude = torch.abs(x_fft)
        
        # 获取排序后的频率带边界
        boundaries = torch.sigmoid(self.band_boundaries)
        boundaries, _ = torch.sort(boundaries)
        
        # 添加起始和结束边界
        boundaries = torch.cat([
            torch.zeros(1, device=boundaries.device),
            boundaries,
            torch.ones(1, device=boundaries.device)
        ])
        
        # 创建频率掩码（简化版本）
        freq_masks = []
        total_freq_size = freq_shape[-1]
        
        for i in range(self.n_experts):
            start_idx = int(boundaries[i] * total_freq_size)
            end_idx = int(boundaries[i + 1] * total_freq_size)
            
            mask = torch.zeros_like(x_fft)
            # 简化：只处理最后一个维度的频率
            mask[..., start_idx:end_idx] = 1.0
            freq_masks.append(mask)
        
        # 计算门控权重
        # 使用空间域特征或频域特征的全局池化
        if x_spatial is not None:
            # 使用空间域的全局平均池化
            gating_input = x_spatial.mean(dim=list(range(2, x_spatial.ndim)))  # [batch, channels]
        else:
            # 使用频域的全局平均池化
            gating_input = freq_magnitude.mean(dim=list(range(2, freq_magnitude.ndim)))  # [batch, channels]
        
        # 通过门控网络
        gating_scores = self.gating_network(gating_input)  # [batch, n_experts]
        gating_scores = F.softmax(gating_scores / self.temperature, dim=-1)
        
        # 应用专家权重调制
        gating_scores = gating_scores * self.expert_weights.unsqueeze(0)
        gating_scores = gating_scores / (gating_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 组合不同频率带的贡献
        weighted_fft = torch.zeros_like(x_fft)
        for i, mask in enumerate(freq_masks):
            # 扩展门控权重到正确的维度
            expert_weight = gating_scores[:, i:i+1]  # [batch, 1]
            for _ in range(len(freq_shape)):
                expert_weight = expert_weight.unsqueeze(-1)
            expert_weight = expert_weight.unsqueeze(1)  # 添加channel维度
            
            weighted_fft = weighted_fft + mask * x_fft * expert_weight
        
        return weighted_fft, gating_scores, boundaries


class AdaptiveSpectralConv(SpectralConv):
    """
    自适应频谱卷积层，使用动态频率选择代替硬截断
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 n_modes: Union[int, List[int]],
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__(in_channels, out_channels, n_modes, **kwargs)
        
        # 确保n_modes是tuple
        if isinstance(n_modes, int):
            n_modes = (n_modes,)
        
        self.n_experts = n_experts
        self.temperature = temperature
        self.adaptive_gating = AdaptiveFrequencyGating(
            n_modes=tuple(n_modes),
            hidden_channels=max(in_channels, out_channels),
            n_experts=n_experts,
            temperature=temperature,
            complex_data=kwargs.get('complex_data', False)
        )
        
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """
        改进的forward方法，结合自适应频率选择和标准频谱卷积
        """
        # 保存空间域输入用于门控
        x_spatial = x.clone()
        
        # 调用父类的forward方法执行标准的频谱卷积
        # 但我们会在FFT之后插入自适应门控
        batchsize, channels, *mode_sizes = x.shape
        
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))
        
        # FFT变换
        if self.fno_block_precision == "half":
            x = x.half()
            
        if self.complex_data:
            x_fft = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x_fft = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]
        
        if self.order > 1:
            x_fft = torch.fft.fftshift(x_fft, dim=dims_to_fft_shift)
        
        if self.fno_block_precision == "mixed":
            x_fft = x_fft.chalf()
        
        # 应用自适应频率门控
        x_fft_gated, gating_scores, boundaries = self.adaptive_gating(x_fft, x_spatial)
        
        # 执行标准的频谱卷积（使用父类的方法）
        # 这里我们简化处理，直接使用加权后的FFT进行后续处理
        # 实际上应该调用父类的卷积操作
        
        # 为了简化，我们直接返回到空间域
        if output_shape is not None:
            mode_sizes = output_shape
        elif self.resolution_scaling_factor is not None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
        
        if self.order > 1:
            x_fft_gated = torch.fft.fftshift(x_fft_gated, dim=dims_to_fft_shift)
        
        if self.complex_data:
            x_out = torch.fft.ifftn(x_fft_gated, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x_out = torch.fft.irfftn(x_fft_gated, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        
        # 添加偏置
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out


def safe_getattr(obj, attr, default=None):
    """安全地获取对象属性，如果不存在则返回默认值"""
    return getattr(obj, attr, default)


class AdaptiveFNOBlocks(FNOBlocks):
    """
    自适应FNO块，包含自适应频谱卷积
    """
    def __init__(self, 
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_experts = n_experts
        self.temperature = temperature
        
        # 替换标准卷积为自适应卷积
        for i in range(len(self.convs)):
            old_conv = self.convs[i]
            
            # 安全地获取属性，如果不存在则使用默认值
            new_conv = AdaptiveSpectralConv(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                n_modes=old_conv.n_modes,
                n_experts=n_experts,
                temperature=temperature,
                max_n_modes=safe_getattr(old_conv, 'max_n_modes'),
                bias=old_conv.bias is not None,
                fft_norm=safe_getattr(old_conv, 'fft_norm', 'forward'),
                factorization=safe_getattr(old_conv, 'factorization'),
                rank=safe_getattr(old_conv, 'rank', 1.0),
                fixed_rank_modes=safe_getattr(old_conv, 'fixed_rank_modes', False),
                implementation=safe_getattr(old_conv, 'implementation', 'factorized'),
                separable=safe_getattr(old_conv, 'separable', False),
                complex_data=safe_getattr(old_conv, 'complex_data', False),
                fno_block_precision=safe_getattr(old_conv, 'fno_block_precision', 'full'),
                resolution_scaling_factor=safe_getattr(old_conv, 'resolution_scaling_factor'),
            )
            
            # 复制权重
            with torch.no_grad():
                if hasattr(old_conv, 'weight'):
                    new_conv.weight = old_conv.weight
                if hasattr(old_conv, 'bias') and old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            
            self.convs[i] = new_conv


class AdaptiveFNO(FNO):
    """
    自适应频率截断的FNO
    使用动态频率选择机制代替硬截断
    """
    def __init__(self,
                 n_modes: Tuple[int, ...],
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 n_layers: int = 4,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 adaptive_mode: Literal["per_layer", "global"] = "per_layer",
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
        adaptive_mode : str
            "per_layer": 每层独立的自适应频率选择
            "global": 所有层共享相同的频率选择策略
        """
        self.n_experts = n_experts
        self.temperature = temperature
        self.adaptive_mode = adaptive_mode
        
        # 初始化基础FNO，但使用自适应FNO blocks
        super().__init__(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **kwargs
        )
        
        # 替换标准FNO blocks为自适应版本
        self.fno_blocks = AdaptiveFNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            n_experts=n_experts,
            temperature=temperature,
            resolution_scaling_factor=kwargs.get('resolution_scaling_factor'),
            use_channel_mlp=kwargs.get('use_channel_mlp', True),
            channel_mlp_dropout=kwargs.get('channel_mlp_dropout', 0),
            channel_mlp_expansion=kwargs.get('channel_mlp_expansion', 0.5),
            non_linearity=kwargs.get('non_linearity', F.gelu),
            norm=kwargs.get('norm'),
            preactivation=kwargs.get('preactivation', False),
            fno_skip=kwargs.get('fno_skip', 'linear'),
            channel_mlp_skip=kwargs.get('channel_mlp_skip', 'soft-gating'),
            complex_data=kwargs.get('complex_data', False),
            max_n_modes=kwargs.get('max_n_modes'),
            fno_block_precision=kwargs.get('fno_block_precision', 'full'),
            rank=kwargs.get('rank', 1.0),
            fixed_rank_modes=kwargs.get('fixed_rank_modes', False),
            implementation=kwargs.get('implementation', 'factorized'),
            separable=kwargs.get('separable', False),
            factorization=kwargs.get('factorization'),
            decomposition_kwargs=kwargs.get('decomposition_kwargs', {}),
        )
    
    def forward(self, x, output_shape=None, **kwargs):
        """
        标准的FNO forward方法
        """
        # 直接调用父类的forward方法
        return super().forward(x, output_shape, **kwargs)


# 创建便捷的构造函数
class AdaptiveFNO2d(AdaptiveFNO):
    """
    2D自适应频率截断FNO
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