"""
Darcy Flow对比实验配置
FNO vs FnoMoE 基准测试配置
"""

from typing import Any, List, Optional
from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class DarcyComparisonOptConfig(OptimizationConfig):
    """Darcy对比实验优化配置"""
    n_epochs: int = 100
    learning_rate: float = 5e-3
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 60
    gamma: float = 0.5


class DarcyComparisonDataConfig(ConfigBase):
    """Darcy对比实验数据配置"""
    folder: str = "~/data/darcy/"
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    n_tests: List[int] = [100, 50]
    test_resolutions: List[int] = [16, 32]
    test_batch_sizes: List[int] = [16, 16]
    encode_input: bool = False
    encode_output: bool = False
    download: bool = True


class FNOModelConfig(ModelConfig):
    """FNO模型配置"""
    model_arch: str = "fno"
    n_modes_height: int = 12
    n_modes_width: int = 12
    hidden_channels: int = 32
    n_layers: int = 4
    projection_channels: int = 64
    in_channels: int = 1
    out_channels: int = 1


class FnoMoEModelConfig(ModelConfig):
    """FnoMoE模型配置"""
    model_arch: str = "adaptive_fno"
    n_modes_height: int = 12
    n_modes_width: int = 12
    hidden_channels: int = 32
    n_layers: int = 4
    projection_channels: int = 64
    in_channels: int = 1
    out_channels: int = 1
    n_experts: int = 4
    temperature: float = 1.0


class DarcyComparisonConfig(ConfigBase):
    """Darcy对比实验主配置"""
    experiment_name: str = "darcy_fno_vs_fnomoe"
    verbose: bool = True
    
    # 数据配置
    data: DarcyComparisonDataConfig = DarcyComparisonDataConfig()
    
    # 优化配置
    opt: DarcyComparisonOptConfig = DarcyComparisonOptConfig()
    
    # 模型配置
    fno_model: FNOModelConfig = FNOModelConfig()
    fnomoe_model: FnoMoEModelConfig = FnoMoEModelConfig()
    
    # 其他配置
    distributed: DistributedConfig = DistributedConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
    
    # 实验特定配置
    save_models: bool = True
    save_predictions: bool = True
    create_visualizations: bool = True
    
    # 评估配置
    eval_resolutions: List[int] = [16, 32]
    eval_metrics: List[str] = ["h1", "l2"]


