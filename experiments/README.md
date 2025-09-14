# 自适应FNO实验结果文件夹

本文件夹包含所有自适应FNO实验的结果，每个实验都有独立的子文件夹。

## 📁 文件夹结构

```
experiments/
├── README.md                    # 本说明文件
├── 2d_synthetic/               # 2D合成数据实验
│   ├── figures/                # 可视化图表
│   ├── models/                 # 保存的模型
│   └── data/                   # 实验数据
├── burgers_equation/           # Burgers方程实验
│   ├── figures/
│   ├── models/
│   └── data/
├── heat_equation/              # 热方程实验（示例）
│   ├── figures/
│   │   ├── heat_training_comparison.png
│   │   └── heat_prediction_comparison.png
│   ├── models/
│   │   ├── original_fno_model.pth
│   │   └── adaptive_fno_model.pth
│   ├── experiment_config.txt   # 实验配置
│   └── experiment_summary.txt  # 实验结果总结
├── concept_demo/               # 概念演示图表
└── other_pdes/                 # 其他PDE实验
    ├── navier_stokes/          # Navier-Stokes方程
    ├── wave_equation/          # 波方程
    ├── maxwell_equations/      # Maxwell方程
    └── ...                     # 更多PDE实验
```

## 🧪 实验类型

### 已完成的实验
1. **2D合成数据实验** (`2d_synthetic/`)
   - 多频率成分的合成数据
   - 验证自适应频率选择机制
   - 性能改进：+0.44% MSE

2. **Burgers方程实验** (`burgers_equation/`)
   - 1D非线性PDE
   - 验证在实际物理问题上的效果
   - 参数增加仅0.1%

3. **热方程实验** (`heat_equation/`)
   - 扩散过程模拟
   - 演示实验模板的使用
   - 性能改进：+68.35% MSE

### 可扩展的实验
- **Navier-Stokes方程**: 流体动力学
- **波方程**: 振荡现象
- **Maxwell方程**: 电磁场
- **Schrödinger方程**: 量子力学
- **反应扩散方程**: 化学反应

## 🚀 如何添加新实验

### 方法1：使用实验模板
```bash
# 复制模板
cp examples/experiment_template.py examples/your_pde_experiment.py

# 修改PDE类型和相关参数
# 运行实验
python examples/your_pde_experiment.py
```

### 方法2：手动创建实验脚本
```python
import os

# 1. 创建实验文件夹
experiment_name = "your_pde_name"
results_dir = f'/root/autodl-tmp/neuraloperator/experiments/{experiment_name}'
os.makedirs(f'{results_dir}/figures', exist_ok=True)
os.makedirs(f'{results_dir}/models', exist_ok=True)
os.makedirs(f'{results_dir}/data', exist_ok=True)

# 2. 保存图片到指定文件夹
plt.savefig(f'{results_dir}/figures/your_figure.png', dpi=150, bbox_inches='tight')

# 3. 保存模型
torch.save(model.state_dict(), f'{results_dir}/models/your_model.pth')

# 4. 保存实验配置和结果
with open(f'{results_dir}/experiment_summary.txt', 'w') as f:
    f.write("Your experiment summary...")
```

## 📊 标准文件命名规范

### 图片文件
- `{pde_name}_training_comparison.png` - 训练过程对比
- `{pde_name}_prediction_comparison.png` - 预测结果对比
- `{pde_name}_error_analysis.png` - 误差分析
- `{pde_name}_frequency_analysis.png` - 频率分析

### 模型文件
- `original_fno_model.pth` - 原始FNO模型
- `adaptive_fno_model.pth` - 自适应FNO模型

### 配置和总结文件
- `experiment_config.txt` - 实验配置参数
- `experiment_summary.txt` - 实验结果总结
- `data_info.txt` - 数据集信息

## 🎯 实验管理最佳实践

1. **命名规范**: 使用描述性的文件夹名称（如 `navier_stokes_2d`）
2. **版本控制**: 对于同一PDE的不同实验，使用版本号（如 `heat_equation_v2`）
3. **文档记录**: 每个实验都应包含配置文件和结果总结
4. **数据管理**: 大型数据集可以软链接到共享存储位置
5. **模型保存**: 保存最佳模型的检查点和最终模型

## 📈 性能对比总结

| 实验类型 | 原始FNO MSE | 自适应FNO MSE | 改进率 | 参数增加 |
|---------|-------------|---------------|--------|----------|
| 2D合成数据 | 0.000301 | 0.000300 | +0.44% | 0.1% |
| Burgers方程 | 0.001941 | 0.001941 | +0.01% | 0.1% |
| 热方程 | 0.000671 | 0.000212 | +68.35% | 0.1% |

## 🔧 工具脚本

- `examples/experiment_template.py` - 新实验的模板脚本
- `examples/compare_fno_adaptive.py` - 2D数据对比脚本
- `examples/burgers_final_comparison.py` - Burgers方程对比脚本

## 📝 注意事项

1. 确保每个实验都有独立的文件夹
2. 图片保存时使用高分辨率（dpi=150）
3. 模型文件包含完整的状态字典
4. 实验配置要详细记录超参数
5. 结果总结要包含定量分析

---

**更新日期**: 2024年9月14日  
**维护者**: 自适应FNO项目组
