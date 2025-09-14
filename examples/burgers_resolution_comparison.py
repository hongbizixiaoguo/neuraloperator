"""
Burgers方程多分辨率对比实验
对比FNO和AdaptiveFNO(FNO-MoE)在不同分辨率下的相对误差
生成类似论文中的结果图
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop.models.fno import FNO1d
from neuralop.models.adaptive_fno import AdaptiveFNO1d
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.utils import get_project_root

def generate_burgers_data(n_samples=1000, spatial_resolution=256, temporal_steps=100, viscosity=0.01):
    """
    生成Burgers方程数据
    """
    # 空间网格
    x = np.linspace(0, 2*np.pi, spatial_resolution, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.01
    
    inputs = []
    outputs = []
    
    print(f"生成 {n_samples} 个样本，空间分辨率: {spatial_resolution}")
    
    for i in range(n_samples):
        # 随机初始条件
        k1, k2 = np.random.randint(1, 6, 2)
        phase1, phase2 = np.random.uniform(0, 2*np.pi, 2)
        amplitude1, amplitude2 = np.random.uniform(0.5, 2.0, 2)
        
        # 初始条件：两个正弦波的叠加
        u0 = amplitude1 * np.sin(k1 * x + phase1) + amplitude2 * np.sin(k2 * x + phase2)
        
        # 使用有限差分求解Burgers方程 (简化版本)
        u = u0.copy()
        
        # 时间演化
        for t in range(temporal_steps):
            # 计算导数
            dudx = np.gradient(u, dx, edge_order=2)
            d2udx2 = np.gradient(dudx, dx, edge_order=2)
            
            # Burgers方程: du/dt + u * du/dx = nu * d2u/dx2
            dudt = -u * dudx + viscosity * d2udx2
            u = u + dt * dudt
            
            # 周期边界条件
            u[0] = u[-1]
        
        inputs.append(u0)
        outputs.append(u)
        
        if (i + 1) % 200 == 0:
            print(f"  生成进度: {i+1}/{n_samples}")
    
    return np.array(inputs), np.array(outputs)

def create_multi_resolution_data(base_inputs, base_outputs, target_resolutions):
    """
    从基础数据创建多分辨率版本
    """
    datasets = {}
    
    for resolution in target_resolutions:
        if resolution == base_inputs.shape[-1]:
            # 原始分辨率
            datasets[resolution] = (base_inputs, base_outputs)
        else:
            # 重采样到目标分辨率
            inputs_resampled = np.array([
                np.interp(np.linspace(0, 1, resolution), 
                         np.linspace(0, 1, base_inputs.shape[-1]), 
                         inp) 
                for inp in base_inputs
            ])
            
            outputs_resampled = np.array([
                np.interp(np.linspace(0, 1, resolution), 
                         np.linspace(0, 1, base_outputs.shape[-1]), 
                         out) 
                for out in base_outputs
            ])
            
            datasets[resolution] = (inputs_resampled, outputs_resampled)
    
    return datasets

def train_and_evaluate_model(model, train_loader, test_loader, n_epochs=100, device='cuda'):
    """
    训练和评估模型
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = F.mse_loss(predictions, batch_outputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.6f}")
    
    # 评估
    model.eval()
    total_relative_error = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch_inputs, batch_outputs in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            predictions = model(batch_inputs)
            
            # 计算相对误差
            relative_error = torch.norm(predictions - batch_outputs, p=2, dim=-1) / \
                           (torch.norm(batch_outputs, p=2, dim=-1) + 1e-8)
            
            total_relative_error += relative_error.sum().item()
            n_samples += batch_inputs.size(0)
    
    avg_relative_error = total_relative_error / n_samples
    return avg_relative_error

def main():
    """运行多分辨率对比实验"""
    print("🚀 Burgers方程多分辨率对比实验")
    print("="*50)
    
    # 创建实验文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/burgers_resolution_comparison'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # 实验参数
    base_resolution = 1024  # 生成高分辨率数据
    target_resolutions = [64, 128, 256, 512, 1024]  # 测试分辨率
    n_train_samples = 800
    n_test_samples = 200
    n_epochs = 100
    batch_size = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 生成基础数据
    print(f"\n📊 生成Burgers方程数据...")
    base_inputs, base_outputs = generate_burgers_data(
        n_samples=n_train_samples + n_test_samples,
        spatial_resolution=base_resolution,
        temporal_steps=100,
        viscosity=0.01
    )
    
    # 分割训练和测试数据
    train_inputs = base_inputs[:n_train_samples]
    train_outputs = base_outputs[:n_train_samples]
    test_inputs = base_inputs[n_train_samples:]
    test_outputs = base_outputs[n_train_samples:]
    
    # 创建多分辨率数据集
    print(f"\n🔄 创建多分辨率数据集...")
    train_datasets = create_multi_resolution_data(train_inputs, train_outputs, target_resolutions)
    test_datasets = create_multi_resolution_data(test_inputs, test_outputs, target_resolutions)
    
    # 存储结果
    results = {
        'FNO': [],
        'FNO-MoE': []
    }
    
    # 对每个分辨率进行实验
    for resolution in target_resolutions:
        print(f"\n📐 测试分辨率: {resolution}")
        
        # 准备数据
        train_inputs_res = torch.FloatTensor(train_datasets[resolution][0]).unsqueeze(1)  # [N, 1, L]
        train_outputs_res = torch.FloatTensor(train_datasets[resolution][1]).unsqueeze(1)  # [N, 1, L]
        test_inputs_res = torch.FloatTensor(test_datasets[resolution][0]).unsqueeze(1)
        test_outputs_res = torch.FloatTensor(test_datasets[resolution][1]).unsqueeze(1)
        
        train_dataset = TensorDataset(train_inputs_res, train_outputs_res)
        test_dataset = TensorDataset(test_inputs_res, test_outputs_res)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 计算合适的模式数
        n_modes = min(16, resolution // 4)
        
        # 创建模型
        models = {
            'FNO': FNO1d(
                n_modes=n_modes,
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                n_layers=4
            ),
            'FNO-MoE': AdaptiveFNO1d(
                n_modes=n_modes,
                in_channels=1,
                out_channels=1,
                hidden_channels=32,
                n_layers=4,
                n_experts=4,
                temperature=1.0
            )
        }
        
        # 训练和评估每个模型
        for model_name, model in models.items():
            print(f"  🚀 训练 {model_name}...")
            start_time = time.time()
            
            relative_error = train_and_evaluate_model(
                model, train_loader, test_loader, n_epochs, device
            )
            
            training_time = time.time() - start_time
            results[model_name].append(relative_error)
            
            print(f"    ✅ {model_name}: 相对误差 = {relative_error:.6f}, 训练时间 = {training_time:.1f}s")
            
            # 保存模型
            torch.save(model.state_dict(), 
                      f'{results_dir}/models/{model_name.lower().replace("-", "_")}_res{resolution}.pth')
    
    # 生成结果图
    print(f"\n📈 生成结果图...")
    
    plt.figure(figsize=(10, 8))
    
    # 绘制结果
    colors = {'FNO': 'darkred', 'FNO-MoE': 'blue'}
    markers = {'FNO': 'o', 'FNO-MoE': 's'}
    
    for model_name in ['FNO', 'FNO-MoE']:
        plt.plot(target_resolutions, results[model_name], 
                color=colors[model_name], marker=markers[model_name], 
                linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel('Resolution', fontsize=14)
    plt.ylabel('Relative error', fontsize=14)
    plt.title('Burger\'s Equation: FNO vs FNO-MoE', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(target_resolutions)
    
    # 美化图表
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/burgers_resolution_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印结果总结
    print(f"\n{'='*60}")
    print(f"BURGERS方程多分辨率对比结果")
    print(f"{'='*60}")
    
    summary_lines = ["BURGERS方程多分辨率对比结果", "="*60, ""]
    
    for i, resolution in enumerate(target_resolutions):
        fno_error = results['FNO'][i]
        moe_error = results['FNO-MoE'][i]
        improvement = ((fno_error - moe_error) / fno_error) * 100
        
        print(f"分辨率 {resolution:4d}: FNO = {fno_error:.6f}, FNO-MoE = {moe_error:.6f}, 改进 = {improvement:+.2f}%")
        summary_lines.append(f"分辨率 {resolution:4d}: FNO = {fno_error:.6f}, FNO-MoE = {moe_error:.6f}, 改进 = {improvement:+.2f}%")
    
    # 计算平均改进
    avg_improvement = np.mean([
        ((results['FNO'][i] - results['FNO-MoE'][i]) / results['FNO'][i]) * 100 
        for i in range(len(target_resolutions))
    ])
    
    print(f"\n平均改进: {avg_improvement:+.2f}%")
    summary_lines.extend(["", f"平均改进: {avg_improvement:+.2f}%"])
    
    # 保存总结
    with open(f'{results_dir}/experiment_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ 结果保存到: {results_dir}")
    print(f"📊 对比图: figures/burgers_resolution_comparison.png")
    print(f"🤖 模型文件: models/")
    print(f"📝 实验总结: experiment_summary.txt")
    
    if avg_improvement > 0:
        print(f"\n🎉 FNO-MoE平均改进 {avg_improvement:.2f}%!")
    else:
        print(f"\n📝 需要进一步调优参数")

if __name__ == "__main__":
    main()
