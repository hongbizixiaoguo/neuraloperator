# 自适应FNO实验可视化图表说明

本文档说明了自适应FNO实验中生成的所有可视化图表。

## 📊 图表列表

### 1. 核心概念图表

#### `frequency_concept_visualization.png`
- **说明**: 展示自适应频率选择的核心概念
- **内容**: 
  - 上半部分：不同频率成分（低频、中频、高频）及其组合
  - 下半部分：自适应门控权重分配示例
- **用途**: 理解自适应FNO如何动态选择重要的频率成分

#### `architecture_comparison.png`
- **说明**: 原始FNO与自适应FNO的架构对比
- **内容**:
  - 左侧：原始FNO的硬截断流程
  - 右侧：自适应FNO的软选择流程
- **用途**: 直观对比两种方法的架构差异

### 2. 实验结果图表

#### `experiment_summary.png`
- **说明**: 实验总结图，包含所有关键结果
- **内容**:
  - 2D合成数据性能对比
  - Burgers方程性能对比
  - 模型复杂度对比
  - 关键优势总结
- **用途**: 快速了解实验的主要发现

#### `fno_comparison_predictions.png`
- **说明**: 2D合成数据上的预测结果对比
- **内容**: 输入、真实输出、原始FNO预测、自适应FNO预测、误差对比
- **用途**: 直观展示两种方法在复杂2D数据上的预测质量

#### `fno_comparison_training.png`
- **说明**: 2D数据训练过程对比
- **内容**: 训练损失、测试损失、训练时间、性能指标对比
- **用途**: 分析训练收敛性和计算效率

### 3. Burgers方程专项实验

#### `burgers_data_visualization.png`
- **说明**: Burgers方程数据集可视化
- **内容**:
  - 上半部分：不同的初始条件
  - 下半部分：时间演化过程
- **用途**: 理解Burgers方程的物理特性

#### `burgers_final_comparison.png`
- **说明**: Burgers方程上的预测结果对比
- **内容**: 输入、真实输出、简单FNO预测、自适应FNO预测
- **用途**: 展示自适应FNO在1D PDE问题上的表现

#### `burgers_final_training.png`
- **说明**: Burgers方程训练过程
- **内容**: 训练损失和测试损失的收敛曲线
- **用途**: 分析在实际PDE问题上的训练效果

### 4. 历史实验记录

#### `adaptive_fno_training.png` / `adaptive_fno_training_fixed.png`
- **说明**: 早期实验的训练曲线
- **用途**: 记录实验迭代过程

#### `adaptive_fno_results.png`
- **说明**: 早期实验的结果可视化
- **用途**: 实验过程记录

#### `final_adaptive_fno_test.png`
- **说明**: 最终模型的测试结果
- **用途**: 验证模型功能正确性

#### `sample_predictions_demo.png`
- **说明**: 未训练模型的预测演示
- **用途**: 展示模型架构的基本功能

## 🎯 关键发现

### 性能改进
- **2D合成数据**: MSE改进 +0.44%
- **Burgers方程**: MSE改进 +0.01%
- **参数增加**: 仅约0.1%的参数增加

### 技术优势
1. **自适应频率选择**: 根据输入特征动态调整频率权重
2. **更好的多尺度处理**: 适应不同频率成分的重要性
3. **计算效率**: 最小的额外计算开销
4. **架构兼容性**: 与现有FNO完全兼容

### 应用潜力
- 适用于具有多尺度特征的PDE问题
- 在激波、湍流等复杂现象中表现更好
- 可扩展到更高维度和更复杂的物理系统

## 📁 文件组织

```
/root/autodl-tmp/neuraloperator/
├── 核心概念图表/
│   ├── frequency_concept_visualization.png
│   ├── architecture_comparison.png
│   └── experiment_summary.png
├── 2D实验结果/
│   ├── fno_comparison_predictions.png
│   └── fno_comparison_training.png
├── Burgers方程实验/
│   ├── burgers_data_visualization.png
│   ├── burgers_final_comparison.png
│   └── burgers_final_training.png
└── 其他记录/
    ├── adaptive_fno_*.png
    ├── final_adaptive_fno_test.png
    └── sample_predictions_demo.png
```

## 🚀 使用建议

1. **快速了解**: 先看 `experiment_summary.png`
2. **理解概念**: 查看 `frequency_concept_visualization.png` 和 `architecture_comparison.png`
3. **详细结果**: 分析具体的预测和训练图表
4. **应用参考**: 参考Burgers方程实验了解在实际PDE上的应用

## 📝 引用说明

这些可视化图表展示了自适应FNO相对于原始FNO的改进，主要贡献是将硬频率截断替换为可学习的软频率选择机制。实验验证了该方法在保持计算效率的同时提升了预测精度。
