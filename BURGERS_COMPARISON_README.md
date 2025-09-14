# Burgers方程FNO vs FNO-MoE对比实验

## 概述
这个实验对比了原始FNO和自适应FNO-MoE在求解Burgers方程上的性能，生成类似论文中的多分辨率对比结果图。

## 快速运行

### 方法1: 直接运行Python脚本
```bash
# 简化版对比实验（推荐）
python run_burgers_comparison.py

# 或者直接运行
python examples/burgers_resolution_simple.py
```

### 方法2: 完整分辨率实验
```bash
# 如果有足够的计算资源和时间
python examples/burgers_resolution_comparison.py
```

## 实验设计

### 简化版实验 (推荐)
- 使用现有的Burgers数据集
- 测试不同的模型配置（模式数和隐藏通道数）
- 模拟不同复杂度的效果
- 训练时间较短，适合快速验证

**配置设置**:
```python
configs = [
    {'name': '8_modes', 'n_modes': 8, 'hidden_channels': 16},
    {'name': '12_modes', 'n_modes': 12, 'hidden_channels': 24}, 
    {'name': '16_modes', 'n_modes': 16, 'hidden_channels': 32},
    {'name': '20_modes', 'n_modes': 20, 'hidden_channels': 40},
    {'name': '24_modes', 'n_modes': 24, 'hidden_channels': 48},
]
```

### 完整版实验
- 生成不同分辨率的Burgers方程数据
- 测试真实的多分辨率性能
- 需要更长的训练时间和更多计算资源

**分辨率设置**:
```python
target_resolutions = [64, 128, 256, 512, 1024]
```

## 实验参数

### 模型配置
- **FNO**: 标准的傅里叶神经算子
- **FNO-MoE**: 自适应频率选择的专家混合FNO
  - 专家数量: 4
  - 温度参数: 1.0
  - 层数: 4

### 训练配置
- 训练轮次: 50 (简化版) / 100 (完整版)
- 学习率: 1e-3
- 权重衰减: 1e-4
- 批大小: 16
- 优化器: AdamW
- 调度器: ReduceLROnPlateau

### Burgers方程设置
- 数学形式: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
- 粘性系数: ν = 0.01
- 空间域: [0, 2π] (周期边界)
- 时间步长: 0.01

## 结果文件

每次实验会生成以下文件：

### 📁 文件结构
```
experiments/burgers_simple_comparison/  # 简化版
├── experiment_summary.txt              # 结果总结
├── figures/
│   └── burgers_model_comparison.png    # 对比图
└── models/
    ├── fno_8_modes.pth                 # 各配置的最佳模型
    ├── fno_moe_8_modes.pth
    └── ...

experiments/burgers_resolution_comparison/  # 完整版
├── experiment_summary.txt
├── figures/
│   └── burgers_resolution_comparison.png
└── models/
    ├── fno_res64.pth                   # 各分辨率的模型
    ├── fno_moe_res64.pth
    └── ...
```

### 📊 结果解读
1. **对比图**: 显示FNO vs FNO-MoE在不同配置/分辨率下的相对误差
2. **相对误差**: L2范数相对误差 = ||pred - true||₂ / ||true||₂
3. **性能改进**: FNO-MoE相对于FNO的改进百分比

## 预期结果

根据论文和理论分析，预期结果：

1. **FNO-MoE优于FNO**: 自适应频率选择应该带来更好的性能
2. **复杂配置下改进更明显**: 在更高模式数/分辨率下，MoE的优势更突出
3. **相对误差降低**: 典型改进范围在5-20%之间

## 生成论文风格的图表

实验会自动生成类似论文中的对比图：
- Y轴：相对误差（对数刻度）
- X轴：模型配置或分辨率
- 两条线：FNO（红色圆点）vs FNO-MoE（蓝色方块）
- 网格和图例

## 故障排除

### 常见问题
1. **数据加载失败**
   ```bash
   # 检查数据文件是否存在
   ls /root/autodl-tmp/neuraloperator/neuralop/data/datasets/data/
   ```

2. **内存不足**
   - 减少批大小: `batch_size=8`
   - 减少隐藏通道数
   - 使用更少的模式数

3. **训练时间过长**
   - 减少训练轮次: `n_epochs=20`
   - 使用更少的配置进行测试

### 调试模式
在脚本中添加调试信息：
```python
print(f"输入形状: {batch_inputs.shape}")
print(f"输出形状: {batch_outputs.shape}")
print(f"预测形状: {predictions.shape}")
```

## 进一步实验

### 参数调优
- 调整专家数量: `n_experts=[2, 4, 8]`
- 调整温度参数: `temperature=[0.5, 1.0, 2.0]`
- 测试不同的学习率和训练轮次

### 其他PDE
- 修改脚本适配其他方程（Heat、Darcy等）
- 测试2D问题的多分辨率性能
- 比较不同损失函数的效果

### 可视化改进
- 添加误差条显示标准差
- 生成训练曲线对比
- 可视化频率选择模式

## 运行示例

```bash
# 进入项目目录
cd /root/autodl-tmp/neuraloperator

# 运行简化版实验（推荐开始）
python run_burgers_comparison.py

# 查看结果
ls experiments/burgers_simple_comparison/figures/
```

预期输出：
```
🚀 Burgers方程简化版多分辨率对比实验
==================================================
🔧 Device: cuda
📊 加载Burgers数据...
✅ 数据加载成功

📐 测试配置: 8_modes (modes=8, channels=16)
  🚀 训练 FNO...
    Epoch 10: Loss = 0.001234
    ...
    ✅ FNO: 相对误差 = 0.045678, 训练时间 = 12.3s
  🚀 训练 FNO-MoE...
    ...
    ✅ FNO-MoE: 相对误差 = 0.041234, 训练时间 = 15.6s

...

============================================================
BURGERS方程模型对比结果
============================================================
8_modes     : FNO = 0.045678, FNO-MoE = 0.041234, 改进 = +9.73%
12_modes    : FNO = 0.039876, FNO-MoE = 0.035432, 改进 = +11.14%
...

平均改进: +10.45%

🎉 FNO-MoE平均改进 10.45%!
```
