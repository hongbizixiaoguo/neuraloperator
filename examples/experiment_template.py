"""
实验模板脚本 - 展示如何为新的PDE实验创建单独的结果文件夹

使用方法：
1. 复制此模板
2. 修改实验名称和PDE相关代码
3. 运行实验，结果会自动保存到对应文件夹

示例：对于新的PDE（如Navier-Stokes, Heat Equation等）
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

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop.models.fno import FNO2d
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d


def create_experiment_folder(experiment_name):
    """
    为实验创建专门的文件夹
    
    Args:
        experiment_name: 实验名称，如 'navier_stokes', 'heat_equation', 'wave_equation' 等
    
    Returns:
        results_dir: 结果保存目录
    """
    results_dir = f'/root/autodl-tmp/neuraloperator/experiments/{experiment_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建子文件夹
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    os.makedirs(f'{results_dir}/data', exist_ok=True)
    
    print(f"✓ Created experiment folder: {results_dir}")
    return results_dir


def save_experiment_config(results_dir, config):
    """保存实验配置"""
    config_path = f'{results_dir}/experiment_config.txt'
    with open(config_path, 'w') as f:
        f.write("Experiment Configuration\n")
        f.write("=" * 30 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Experiment config saved to {config_path}")


def generate_sample_pde_data(n_samples=100, grid_size=64, pde_type="heat"):
    """
    生成示例PDE数据
    
    Args:
        n_samples: 样本数量
        grid_size: 网格大小
        pde_type: PDE类型 ("heat", "wave", "advection")
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        if pde_type == "heat":
            # 热方程：扩散过程
            k1, k2 = np.random.randint(1, 8, 2)
            phase = np.random.uniform(0, 2*np.pi)
            
            # 初始条件
            u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
            
            # 时间演化（扩散）
            diffusion = 0.01
            t = 0.1
            decay = np.exp(-diffusion * (k1**2 + k2**2) * t)
            u1 = decay * u0
            
        elif pde_type == "wave":
            # 波方程：振荡过程
            k1, k2 = np.random.randint(1, 10, 2)
            phase = np.random.uniform(0, 2*np.pi)
            
            # 初始条件
            u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
            
            # 时间演化（波动）
            c = 1.0  # 波速
            t = 0.1
            u1 = np.cos(c * np.sqrt(k1**2 + k2**2) * t) * u0
            
        else:  # advection
            # 平流方程：传输过程
            k1, k2 = np.random.randint(1, 6, 2)
            phase = np.random.uniform(0, 2*np.pi)
            
            # 初始条件
            u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
            
            # 时间演化（平流）
            vx, vy = 0.5, 0.3  # 速度场
            t = 0.1
            X_shifted = X - vx * t
            Y_shifted = Y - vy * t
            u1 = np.sin(2*np.pi*k1*X_shifted + phase) * np.cos(2*np.pi*k2*Y_shifted)
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def train_models(train_loader, test_loader, device, results_dir):
    """训练原始FNO和自适应FNO"""
    
    # 创建模型
    original_fno = FNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4
    ).to(device)
    
    adaptive_fno = AdaptiveFNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4
    ).to(device)
    
    models = {
        'Original FNO': original_fno,
        'Adaptive FNO': adaptive_fno
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        test_losses = []
        
        # 训练循环
        for epoch in range(20):  # 简化的训练
            model.train()
            epoch_train_loss = 0
            
            for batch_inputs, batch_outputs in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = F.mse_loss(predictions, batch_outputs)
                loss.backward()
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
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            scheduler.step(avg_test_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        # 保存模型
        model_path = f'{results_dir}/models/{name.lower().replace(" ", "_")}_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
    
    return results


def visualize_results(results, test_loader, device, results_dir, pde_type):
    """可视化实验结果"""
    
    # 1. 训练曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        axes[0].plot(result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
        axes[1].plot(result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle(f'{pde_type.title()} Equation: Training Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存训练曲线
    training_fig_path = f'{results_dir}/figures/{pde_type}_training_comparison.png'
    plt.savefig(training_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {training_fig_path}")
    plt.show()
    
    # 2. 预测结果对比
    models = {name: result['model'] for name, result in results.items()}
    
    # 获取测试样本
    test_inputs, test_outputs = next(iter(test_loader))
    test_inputs = test_inputs[:3].to(device)  # 只取前3个样本
    test_outputs = test_outputs[:3].to(device)
    
    predictions = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(test_inputs)
            predictions[name] = pred.cpu()
    
    # 可视化预测结果
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i in range(3):
        # 输入
        axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # 原始FNO预测
        original_pred = predictions['Original FNO'][i, 0]
        axes[i, 2].imshow(original_pred.numpy(), cmap='viridis')
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].axis('off')
        
        # 自适应FNO预测
        adaptive_pred = predictions['Adaptive FNO'][i, 0]
        axes[i, 3].imshow(adaptive_pred.numpy(), cmap='viridis')
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        original_error = torch.abs(original_pred - test_outputs[i, 0].cpu())
        adaptive_error = torch.abs(adaptive_pred - test_outputs[i, 0].cpu())
        error_diff = original_error - adaptive_error
        
        im = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: Adaptive Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle(f'{pde_type.title()} Equation: Prediction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存预测对比
    prediction_fig_path = f'{results_dir}/figures/{pde_type}_prediction_comparison.png'
    plt.savefig(prediction_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction comparison saved to {prediction_fig_path}")
    plt.show()


def print_summary(results, results_dir, pde_type):
    """打印和保存实验总结"""
    print(f"\n{'='*60}")
    print(f"{pde_type.upper()} EQUATION EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    summary_lines = []
    summary_lines.append(f"{pde_type.upper()} EQUATION EXPERIMENT SUMMARY")
    summary_lines.append("="*60)
    
    for name, result in results.items():
        train_loss = result['final_train_loss']
        test_loss = result['final_test_loss']
        param_count = sum(p.numel() for p in result['model'].parameters())
        
        print(f"\n{name}:")
        print(f"  Final Train Loss: {train_loss:.6f}")
        print(f"  Final Test Loss:  {test_loss:.6f}")
        print(f"  Parameters:       {param_count:,}")
        
        summary_lines.append(f"\n{name}:")
        summary_lines.append(f"  Final Train Loss: {train_loss:.6f}")
        summary_lines.append(f"  Final Test Loss:  {test_loss:.6f}")
        summary_lines.append(f"  Parameters:       {param_count:,}")
    
    # 计算改进
    original_test_loss = results['Original FNO']['final_test_loss']
    adaptive_test_loss = results['Adaptive FNO']['final_test_loss']
    improvement = ((original_test_loss - adaptive_test_loss) / original_test_loss) * 100
    
    print(f"\nPerformance Improvement:")
    print(f"  Test Loss Improvement: {improvement:+.2f}%")
    
    summary_lines.append(f"\nPerformance Improvement:")
    summary_lines.append(f"  Test Loss Improvement: {improvement:+.2f}%")
    
    # 保存总结
    summary_path = f'{results_dir}/experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Experiment summary saved to {summary_path}")
    print(f"✓ All results saved in: {results_dir}")


def run_pde_experiment(pde_type="heat", n_samples=200, grid_size=64):
    """
    运行PDE实验的主函数
    
    Args:
        pde_type: PDE类型 ("heat", "wave", "advection")
        n_samples: 样本数量
        grid_size: 网格大小
    """
    print(f"🧪 Starting {pde_type.upper()} Equation Experiment")
    print("="*50)
    
    # 1. 创建实验文件夹
    experiment_name = f"{pde_type}_equation"
    results_dir = create_experiment_folder(experiment_name)
    
    # 2. 保存实验配置
    config = {
        'PDE Type': pde_type,
        'Number of Samples': n_samples,
        'Grid Size': f'{grid_size}x{grid_size}',
        'Training Epochs': 20,
        'Learning Rate': 1e-3,
        'Weight Decay': 1e-4,
        'FNO Modes': '16x16',
        'Hidden Channels': 32,
        'FNO Layers': 4,
        'Adaptive Experts': 4
    }
    save_experiment_config(results_dir, config)
    
    # 3. 生成数据
    print(f"\n📊 Generating {pde_type} equation data...")
    inputs, outputs = generate_sample_pde_data(n_samples, grid_size, pde_type)
    
    # 分割数据
    train_size = int(0.8 * n_samples)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ Data generated: {train_size} train, {len(test_inputs)} test samples")
    
    # 4. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    results = train_models(train_loader, test_loader, device, results_dir)
    
    # 5. 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_results(results, test_loader, device, results_dir, pde_type)
    
    # 6. 打印总结
    print_summary(results, results_dir, pde_type)
    
    return results_dir


if __name__ == "__main__":
    # 示例：运行热方程实验
    print("🎯 PDE Experiment Template")
    print("This script demonstrates how to organize experiments with separate folders")
    print("\nAvailable PDE types: heat, wave, advection")
    
    # 运行示例实验
    pde_type = "heat"  # 可以改为 "wave" 或 "advection"
    results_dir = run_pde_experiment(pde_type=pde_type, n_samples=100, grid_size=64)
    
    print(f"\n🎉 Experiment completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Check the 'figures' subfolder for visualizations")
    print(f"🤖 Check the 'models' subfolder for saved models")
    print(f"📝 Check experiment_summary.txt for detailed results")
