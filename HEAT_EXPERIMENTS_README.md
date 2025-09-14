# Heat Equation实验指南

## 概述
这个实验比较了原始FNO和自适应FNO在求解热方程上的性能。

## 快速运行

### 方法1: 使用交互式脚本
```bash
./run_heat_experiments.sh
```
然后选择：
- 1) 100 epochs (快速验证)
- 2) 200 epochs (中等训练) 
- 3) 300 epochs (充分训练)
- 4) 自定义epochs
- 5) 运行所有实验

### 方法2: 直接运行Python脚本
```bash
# 100 epochs (默认)
python examples/heat_equation_100epochs.py

# 200 epochs
python run_heat_200.py

# 300 epochs  
python run_heat_300.py

# 自定义epochs
python examples/heat_equation_100epochs.py --epochs 250
```

### 方法3: 命令行参数
```bash
python examples/heat_equation_100epochs.py --epochs 200 --lr 1e-3 --batch-size 16 --n-experts 4
```

## 参数说明
- `--epochs`: 训练轮次 (默认: 100)
- `--lr`: 学习率 (默认: 1e-3)
- `--batch-size`: 批大小 (默认: 16)
- `--n-experts`: 自适应FNO的专家数量 (默认: 4)

## 实验配置
- **PDE类型**: 热方程 ∂u/∂t = α∇²u
- **扩散系数**: 0.01
- **时间步长**: 0.1
- **样本数量**: 150 (120训练 + 30测试)
- **网格大小**: 64×64
- **FNO模式**: 16×16
- **隐藏通道**: 32
- **FNO层数**: 4

## 结果文件
每次实验会在 `experiments/heat_equation_{epochs}epochs/` 目录下生成：

### 📁 文件结构
```
experiments/heat_equation_{epochs}epochs/
├── experiment_config.txt          # 实验配置
├── experiment_summary.txt         # 结果总结
├── figures/
│   ├── training_curves_{epochs}epochs.png    # 训练曲线
│   └── predictions_{epochs}epochs.png        # 预测对比
└── models/
    ├── original_fno_best.pth      # 原始FNO最佳模型
    ├── original_fno_final.pth     # 原始FNO最终模型
    ├── adaptive_fno_best.pth      # 自适应FNO最佳模型
    └── adaptive_fno_final.pth     # 自适应FNO最终模型
```

### 📊 结果解读
1. **训练曲线**: 显示训练和测试损失随epochs的变化
2. **预测对比**: 可视化原始FNO vs 自适应FNO的预测效果
3. **性能改进**: 自适应FNO相对于原始FNO的改进百分比

## 建议的实验流程
1. **第一步**: 运行100 epochs快速验证代码和初步效果
2. **第二步**: 运行200 epochs查看中等训练效果
3. **第三步**: 运行300 epochs获得充分训练的结果
4. **对比分析**: 比较不同epochs下的性能提升

## 预期结果
- 自适应FNO应该在较少的epochs下达到更好的性能
- 随着epochs增加，两个模型的性能差距可能会更明显
- 自适应FNO的频率选择机制应该带来更好的收敛性

## 故障排除
如果遇到问题：
1. 检查CUDA是否可用: `python -c "import torch; print(torch.cuda.is_available())"`
2. 检查内存使用: 如果GPU内存不足，可以减小batch_size
3. 检查路径: 确保在正确的目录下运行脚本

## 进一步实验
- 尝试不同的专家数量 (2, 4, 8)
- 调整学习率和其他超参数
- 增加样本数量或网格分辨率
- 测试其他PDE类型 (Burgers方程、Darcy流等)
