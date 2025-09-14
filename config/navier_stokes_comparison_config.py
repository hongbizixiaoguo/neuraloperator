from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Medium2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

class NavierStokesOptConfig(OptimizationConfig):
    n_epochs: int = 100
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

class NavierStokesDatasetConfig(ConfigBase):
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 800
    train_resolution: int = 128
    n_tests: List[int] = [200]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = True
    encode_output: bool = True

class FNOModelConfig(ModelConfig):
    """标准FNO模型配置"""
    model_arch: str = "fno"
    data_channels: int = 1
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 32
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    in_channels: int = 1
    out_channels: int = 1

class AdaptiveFNOModelConfig(ModelConfig):
    """自适应FNO (FnoMoE)模型配置"""
    model_arch: str = "adaptive_fno"
    data_channels: int = 1
    n_modes: List[int] = [16, 16]
    hidden_channels: int = 32
    n_layers: int = 4
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    in_channels: int = 1
    out_channels: int = 1
    n_experts: int = 4
    temperature: float = 1.0

class NavierStokesComparisonConfig(ConfigBase):
    """Navier-Stokes对比实验主配置"""
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
    
    # 模型配置
    fno_model: FNOModelConfig = FNOModelConfig()
    adaptive_fno_model: AdaptiveFNOModelConfig = AdaptiveFNOModelConfig()

# 默认配置
class Default(NavierStokesComparisonConfig):
    pass
