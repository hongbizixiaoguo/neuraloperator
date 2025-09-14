"""
Navier-Stokes方程实验 - 对比原始FNO和自适应FNO (FnoMoE)
基于论文中的2D Navier-Stokes涡度形式实现
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
from scipy.io import loadmat
import requests
from pathlib import Path

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop.models.fno import FNO2d
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d


def create_experiment_folder():
    """创建Navier-Stokes实验文件夹"""
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/navier_stokes'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    os.makedirs(f'{results_dir}/data', exist_ok=True)
    
    print(f"✓ Created Navier-Stokes experiment folder: {results_dir}")
    return results_dir


def save_experiment_config(results_dir, config):
    """保存实验配置"""
    config_path = f'{results_dir}/experiment_config.txt'
    with open(config_path, 'w') as f:
        f.write("Navier-Stokes Equation Experiment Configuration\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Experiment config saved to {config_path}")


def generate_navier_stokes_data(n_samples=1000, grid_size=64, viscosity=1e-4, T=30):
    """
    生成2D Navier-Stokes方程数据（涡度形式）
    
    方程形式:
    ∂w/∂t + u·∇w = ν∆w + f(x)
    ∇·u = 0
    w(x,0) = w₀(x)
    
    其中 w = ∇ × u 是涡度，u是速度场
    """
    print(f"Generating {n_samples} Navier-Stokes samples...")
    print(f"Parameters: viscosity={viscosity}, T={T}, grid_size={grid_size}x{grid_size}")
    
    # 创建空间网格
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs = []
    outputs = []
    
    # 时间步长
    dt = T / 100  # 100个时间步
    
    for i in range(n_samples):
        # 生成随机初始涡度场
        # 使用多个涡旋的叠加
        n_vortices = np.random.randint(2, 6)  # 2-5个涡旋
        w0 = np.zeros((grid_size, grid_size))
        
        for _ in range(n_vortices):
            # 随机涡旋中心
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            
            # 随机强度和尺度
            strength = np.random.uniform(-2, 2)
            sigma = np.random.uniform(0.05, 0.15)
            
            # 高斯涡旋
            r2 = (X - cx)**2 + (Y - cy)**2
            vortex = strength * np.exp(-r2 / (2 * sigma**2))
            w0 += vortex
        
        # 简化的时间演化（基于扩散和对流的近似）
        w = w0.copy()
        
        # 模拟时间演化
        for t_step in range(int(T)):
            # 扩散项 (简化处理)
            laplacian = np.zeros_like(w)
            laplacian[1:-1, 1:-1] = (w[2:, 1:-1] + w[:-2, 1:-1] + 
                                   w[1:-1, 2:] + w[1:-1, :-2] - 4*w[1:-1, 1:-1])
            
            # 扩散更新
            w += viscosity * dt * laplacian
            
            # 添加少量随机扰动模拟对流效应
            if t_step % 10 == 0:
                noise = np.random.normal(0, 0.01, w.shape)
                w += noise
            
            # 边界条件 (周期性)
            w[0, :] = w[-2, :]
            w[-1, :] = w[1, :]
            w[:, 0] = w[:, -2]
            w[:, -1] = w[:, 1]
        
        inputs.append(w0[np.newaxis, :, :])
        outputs.append(w[np.newaxis, :, :])
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def create_models(device, grid_size=64):
    """创建原始FNO和自适应FNO模型"""
    
    # 根据论文中的配置
    n_modes = 16  # 对应64x64分辨率
    
    
    # 原始FNO (FNO-2D)
    original_fno = FNO2d(
        n_modes_height=n_modes,
        n_modes_width=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        projection_channels=64
    ).to(device)
    
    # 自适应FNO (FnoMoE)
    adaptive_fno = AdaptiveFNO2d(
        n_modes_height=n_modes,
        n_modes_width=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4,
        projection_channels=64
    ).to(device)
    
    # 计算参数数量
    original_params = sum(p.numel() for p in original_fno.parameters())
    adaptive_params = sum(p.numel() for p in adaptive_fno.parameters())
    
    print(f"Original FNO parameters: {original_params:,}")
    print(f"Adaptive FNO parameters: {adaptive_params:,}")
    
    return {
        'FNO-2D': original_fno,
        'FnoMoE': adaptive_fno
    }, {
        'FNO-2D': original_params,
        'FnoMoE': adaptive_params
    }


def train_models(models, param_counts, train_loader, test_loader, device, results_dir, epochs=100):
    """训练模型并记录性能"""
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        start_time = time.time()
        
        # 优化器设置
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
        )
        
        train_losses = []
        test_losses = []
        epoch_times = []
        best_test_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练
            model.train()
            epoch_train_loss = 0
            
            for batch_inputs, batch_outputs in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = F.mse_loss(predictions, batch_outputs)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # 测试
            model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_outputs in test_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_outputs = batch_outputs.to(device)
                    predictions = model(batch_inputs)
                    loss = F.mse_loss(predictions, batch_outputs)
                    epoch_test_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_test_loss = epoch_test_loss / len(test_loader)
            epoch_time = time.time() - epoch_start
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            epoch_times.append(epoch_time)
            
            scheduler.step(avg_test_loss)
            
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                # 保存最佳模型
                torch.save(model.state_dict(), 
                          f'{results_dir}/models/{name.lower().replace("-", "_")}_best.pth')
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}, Time={epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        avg_epoch_time = np.mean(epoch_times)
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss,
            'final_test_loss': test_losses[-1],
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'parameters': param_counts[name]
        }
        
        # 保存最终模型
        torch.save(model.state_dict(), 
                  f'{results_dir}/models/{name.lower().replace("-", "_")}_final.pth')
        
        print(f"  ✓ Training completed in {total_time:.1f}s (avg {avg_epoch_time:.2f}s/epoch)")
    
    return results


def evaluate_models(results, test_loader, device):
    """评估模型性能，计算推理时间"""
    
    print(f"\n📊 Evaluating model performance...")
    
    eval_results = {}
    
    for name, result in results.items():
        model = result['model']
        model.eval()
        
        total_mse = 0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                # 测量推理时间
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time = time.time()
                
                predictions = model(batch_inputs)
                
                torch.cuda.synchronize() if device.type == 'cuda' else None
                inference_time = time.time() - start_time
                
                # 计算MSE
                mse = F.mse_loss(predictions, batch_outputs)
                total_mse += mse.item() * batch_inputs.size(0)
                total_samples += batch_inputs.size(0)
                
                # 记录每个样本的推理时间
                inference_times.append(inference_time / batch_inputs.size(0))
        
        avg_mse = total_mse / total_samples
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        
        eval_results[name] = {
            'mse': avg_mse,
            'inference_time_ms': avg_inference_time
        }
    
    return eval_results


def visualize_results(results, eval_results, test_loader, device, results_dir):
    """可视化实验结果"""
    
    # 1. 训练曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'FNO-2D': 'blue', 'FnoMoE': 'red'}
    
    for name, result in results.items():
        color = colors[name]
        epochs = range(1, len(result['train_losses']) + 1)
        
        axes[0].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
        axes[1].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle('Navier-Stokes: FNO vs FnoMoE Training Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/ns_training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测结果对比
    test_inputs, test_outputs = next(iter(test_loader))
    test_inputs = test_inputs[:4].to(device)
    test_outputs = test_outputs[:4].to(device)
    
    predictions = {}
    for name, result in results.items():
        result['model'].eval()
        with torch.no_grad():
            pred = result['model'](test_inputs)
            predictions[name] = pred.cpu()
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # 输入涡度场
        im1 = axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='RdBu', vmin=-2, vmax=2)
        axes[i, 0].set_title(f'Initial Vorticity {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        im2 = axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='RdBu', vmin=-2, vmax=2)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # FNO-2D预测
        fno_pred = predictions['FNO-2D'][i, 0]
        im3 = axes[i, 2].imshow(fno_pred.numpy(), cmap='RdBu', vmin=-2, vmax=2)
        axes[i, 2].set_title(f'FNO-2D {i+1}')
        axes[i, 2].axis('off')
        
        # FnoMoE预测
        fnomoe_pred = predictions['FnoMoE'][i, 0]
        im4 = axes[i, 3].imshow(fnomoe_pred.numpy(), cmap='RdBu', vmin=-2, vmax=2)
        axes[i, 3].set_title(f'FnoMoE {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        fno_error = torch.abs(fno_pred - test_outputs[i, 0].cpu())
        fnomoe_error = torch.abs(fnomoe_pred - test_outputs[i, 0].cpu())
        error_diff = fno_error - fnomoe_error
        
        im5 = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.2, vmax=0.2)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: FnoMoE Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Navier-Stokes: Prediction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/ns_prediction_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_benchmark_table(results, eval_results, viscosity, T, N):
    """创建类似论文的基准测试表格"""
    
    print(f"\n{'='*80}")
    print(f"NAVIER-STOKES EQUATION BENCHMARKS")
    print(f"(Resolution: 64×64, ν={viscosity}, T={T}, N={N})")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'Config':<10} {'Parameters':<12} {'Time per':<10} {'MSE':<10}")
    print(f"{'':10} {'':12} {'epoch (s)':<10} {'':10}")
    print(f"{'-'*50}")
    
    # 数据行
    for name, result in results.items():
        params = result['parameters']
        time_per_epoch = result['avg_epoch_time']
        mse = eval_results[name]['mse']
        
        print(f"{name:<10} {params:<12,} {time_per_epoch:<10.2f} {mse:<10.4f}")
    
    # 计算改进
    fno_mse = eval_results['FNO-2D']['mse']
    fnomoe_mse = eval_results['FnoMoE']['mse']
    improvement = ((fno_mse - fnomoe_mse) / fno_mse) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"FNO-2D MSE:     {fno_mse:.6f}")
    print(f"FnoMoE MSE:     {fnomoe_mse:.6f}")
    print(f"Improvement:    {improvement:+.2f}%")
    
    # 参数对比
    fno_params = results['FNO-2D']['parameters']
    fnomoe_params = results['FnoMoE']['parameters']
    param_increase = ((fnomoe_params - fno_params) / fno_params) * 100
    
    print(f"\nParameter Comparison:")
    print(f"FNO-2D params:  {fno_params:,}")
    print(f"FnoMoE params:  {fnomoe_params:,}")
    print(f"Increase:       {param_increase:+.1f}%")
    
    return {
        'fno_mse': fno_mse,
        'fnomoe_mse': fnomoe_mse,
        'improvement': improvement,
        'fno_params': fno_params,
        'fnomoe_params': fnomoe_params,
        'param_increase': param_increase
    }


def save_experiment_summary(results, eval_results, benchmark_results, results_dir, config):
    """保存实验总结"""
    
    summary_lines = []
    summary_lines.append("NAVIER-STOKES EQUATION EXPERIMENT SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    # 配置信息
    summary_lines.append("Experiment Configuration:")
    for key, value in config.items():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("")
    
    # 训练结果
    summary_lines.append("Training Results:")
    for name, result in results.items():
        summary_lines.append(f"\n{name}:")
        summary_lines.append(f"  Parameters: {result['parameters']:,}")
        summary_lines.append(f"  Best Test Loss: {result['best_test_loss']:.6f}")
        summary_lines.append(f"  Final Test Loss: {result['final_test_loss']:.6f}")
        summary_lines.append(f"  Training Time: {result['total_time']:.1f}s")
        summary_lines.append(f"  Avg Time/Epoch: {result['avg_epoch_time']:.2f}s")
    
    # 评估结果
    summary_lines.append("\nEvaluation Results:")
    for name, eval_result in eval_results.items():
        summary_lines.append(f"\n{name}:")
        summary_lines.append(f"  MSE: {eval_result['mse']:.6f}")
        summary_lines.append(f"  Inference Time: {eval_result['inference_time_ms']:.2f}ms")
    
    # 基准对比
    summary_lines.append(f"\nBenchmark Comparison:")
    summary_lines.append(f"  MSE Improvement: {benchmark_results['improvement']:+.2f}%")
    summary_lines.append(f"  Parameter Increase: {benchmark_results['param_increase']:+.1f}%")
    
    # 保存到文件
    summary_path = f'{results_dir}/experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Experiment summary saved to {summary_path}")


def main():
    """主实验函数"""
    print("🌊 Navier-Stokes Equation: FNO vs FnoMoE Comparison")
    print("="*60)
    
    # 实验参数（基于论文）
    viscosity = 1e-4
    T = 30
    N = 1000
    grid_size = 64
    epochs = 100
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 保存配置
    config = {
        'Equation': '2D Navier-Stokes (Vorticity Form)',
        'Mathematical Form': '∂w/∂t + u·∇w = ν∆w + f(x)',
        'Viscosity': viscosity,
        'Final Time': T,
        'Number of Samples': N,
        'Grid Size': f'{grid_size}×{grid_size}',
        'Training Epochs': epochs,
        'Learning Rate': '1e-3 (AdamW)',
        'Weight Decay': 1e-4,
        'Scheduler': 'ReduceLROnPlateau',
        'Batch Size': 16
    }
    save_experiment_config(results_dir, config)
    
    # 生成数据
    print(f"\n📊 Generating Navier-Stokes data...")
    inputs, outputs = generate_navier_stokes_data(
        n_samples=N, 
        grid_size=grid_size, 
        viscosity=viscosity, 
        T=T
    )
    
    # 数据分割
    train_size = int(0.8 * N)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]
    
    # 数据加载器
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ Data prepared: {train_size} train, {len(test_inputs)} test samples")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    models, param_counts = create_models(device, grid_size)
    
    # 训练模型
    results = train_models(models, param_counts, train_loader, test_loader, device, results_dir, epochs)
    
    # 评估模型
    eval_results = evaluate_models(results, test_loader, device)
    
    # 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_results(results, eval_results, test_loader, device, results_dir)
    
    # 创建基准测试表格
    benchmark_results = create_benchmark_table(results, eval_results, viscosity, T, N)
    
    # 保存实验总结
    save_experiment_summary(results, eval_results, benchmark_results, results_dir, config)
    
    print(f"\n🎉 Navier-Stokes experiment completed!")
    print(f"📁 Results saved in: {results_dir}")
    
    return results_dir


if __name__ == "__main__":
    main()


