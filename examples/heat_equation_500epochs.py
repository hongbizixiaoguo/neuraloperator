"""
Heat Equation实验 - 500 epochs训练
基于experiment_template.py，专门针对Heat Equation进行长时间训练
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
    """为实验创建专门的文件夹"""
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
        f.write("Heat Equation Experiment Configuration (500 Epochs)\n")
        f.write("=" * 50 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Experiment config saved to {config_path}")


def generate_heat_equation_data(n_samples=200, grid_size=64):
    """
    生成Heat Equation数据
    
    Heat Equation: ∂u/∂t = α∇²u
    解析解: u(x,y,t) = exp(-α(k₁² + k₂²)t) × sin(2πk₁x + φ)cos(2πk₂y)
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs = []
    outputs = []
    
    print(f"Generating {n_samples} Heat Equation samples...")
    
    for i in range(n_samples):
        # 随机波数和相位
        k1, k2 = np.random.randint(1, 8, 2)  # 波数范围1-7
        phase = np.random.uniform(0, 2*np.pi)
        
        # 初始条件 u(x,y,0)
        u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
        
        # 时间演化参数
        diffusion = 0.01  # 扩散系数α
        t = 0.1          # 时间步长
        
        # 解析解 u(x,y,t)
        decay = np.exp(-diffusion * (k1**2 + k2**2) * t)
        u1 = decay * u0
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def train_models_500epochs(train_loader, test_loader, device, results_dir):
    """训练原始FNO和自适应FNO - 500 epochs"""
    
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
        print(f"\n🚀 Training {name} for 500 epochs...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        train_losses = []
        test_losses = []
        best_test_loss = float('inf')
        patience_counter = 0
        
        # 训练循环 - 500 epochs
        for epoch in range(500):
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
            
            # 早停检查
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_path = f'{results_dir}/models/{name.lower().replace(" ", "_")}_best_model.pth'
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:3d}/500: Train={avg_train_loss:.8f}, Test={avg_test_loss:.8f}, Best={best_test_loss:.8f}")
            
            # 早停（可选）
            if patience_counter >= 100:  # 100 epochs没有改进就停止
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1],
            'best_test_loss': best_test_loss,
            'total_epochs': len(train_losses)
        }
        
        # 保存最终模型
        final_model_path = f'{results_dir}/models/{name.lower().replace(" ", "_")}_final_model.pth'
        torch.save(model.state_dict(), final_model_path)
        print(f"✓ Models saved: best and final versions")
    
    return results


def visualize_500epoch_results(results, test_loader, device, results_dir):
    """可视化500 epoch训练结果"""
    
    # 1. 详细训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失（线性尺度）
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        epochs = range(1, len(result['train_losses']) + 1)
        axes[0, 0].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].set_title('Training Loss (Linear Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练损失（对数尺度）
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        epochs = range(1, len(result['train_losses']) + 1)
        axes[0, 1].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Train Loss')
    axes[0, 1].set_title('Training Loss (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 测试损失（线性尺度）
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        epochs = range(1, len(result['test_losses']) + 1)
        axes[1, 0].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Test Loss')
    axes[1, 0].set_title('Test Loss (Linear Scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 测试损失（对数尺度）
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        epochs = range(1, len(result['test_losses']) + 1)
        axes[1, 1].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Test Loss')
    axes[1, 1].set_title('Test Loss (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle('Heat Equation: 500 Epochs Training Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存训练曲线
    training_fig_path = f'{results_dir}/figures/heat_500epochs_training_comparison.png'
    plt.savefig(training_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {training_fig_path}")
    plt.show()
    
    # 2. 预测结果对比
    models = {name: result['model'] for name, result in results.items()}
    
    # 获取测试样本
    test_inputs, test_outputs = next(iter(test_loader))
    test_inputs = test_inputs[:4].to(device)  # 取前4个样本
    test_outputs = test_outputs[:4].to(device)
    
    predictions = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(test_inputs)
            predictions[name] = pred.cpu()
    
    # 可视化预测结果
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # 输入
        im1 = axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # 真实输出
        im2 = axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 原始FNO预测
        original_pred = predictions['Original FNO'][i, 0]
        im3 = axes[i, 2].imshow(original_pred.numpy(), cmap='viridis')
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # 自适应FNO预测
        adaptive_pred = predictions['Adaptive FNO'][i, 0]
        im4 = axes[i, 3].imshow(adaptive_pred.numpy(), cmap='viridis')
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # 误差对比
        original_error = torch.abs(original_pred - test_outputs[i, 0].cpu())
        adaptive_error = torch.abs(adaptive_pred - test_outputs[i, 0].cpu())
        error_diff = original_error - adaptive_error
        
        im5 = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: Adaptive Better)')
        axes[i, 4].axis('off')
        plt.colorbar(im5, ax=axes[i, 4], fraction=0.046, pad=0.04)
    
    plt.suptitle('Heat Equation: 500 Epochs Prediction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存预测对比
    prediction_fig_path = f'{results_dir}/figures/heat_500epochs_prediction_comparison.png'
    plt.savefig(prediction_fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Prediction comparison saved to {prediction_fig_path}")
    plt.show()


def print_500epoch_summary(results, results_dir):
    """打印和保存500 epoch实验总结"""
    print(f"\n{'='*70}")
    print(f"HEAT EQUATION 500 EPOCHS EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    summary_lines = []
    summary_lines.append(f"HEAT EQUATION 500 EPOCHS EXPERIMENT SUMMARY")
    summary_lines.append("="*70)
    
    for name, result in results.items():
        final_train_loss = result['final_train_loss']
        final_test_loss = result['final_test_loss']
        best_test_loss = result['best_test_loss']
        total_epochs = result['total_epochs']
        param_count = sum(p.numel() for p in result['model'].parameters())
        
        print(f"\n{name}:")
        print(f"  Total Epochs:      {total_epochs}")
        print(f"  Final Train Loss:  {final_train_loss:.8f}")
        print(f"  Final Test Loss:   {final_test_loss:.8f}")
        print(f"  Best Test Loss:    {best_test_loss:.8f}")
        print(f"  Parameters:        {param_count:,}")
        
        summary_lines.append(f"\n{name}:")
        summary_lines.append(f"  Total Epochs:      {total_epochs}")
        summary_lines.append(f"  Final Train Loss:  {final_train_loss:.8f}")
        summary_lines.append(f"  Final Test Loss:   {final_test_loss:.8f}")
        summary_lines.append(f"  Best Test Loss:    {best_test_loss:.8f}")
        summary_lines.append(f"  Parameters:        {param_count:,}")
    
    # 计算改进
    original_final = results['Original FNO']['final_test_loss']
    adaptive_final = results['Adaptive FNO']['final_test_loss']
    final_improvement = ((original_final - adaptive_final) / original_final) * 100
    
    original_best = results['Original FNO']['best_test_loss']
    adaptive_best = results['Adaptive FNO']['best_test_loss']
    best_improvement = ((original_best - adaptive_best) / original_best) * 100
    
    print(f"\nPerformance Improvement:")
    print(f"  Final Test Loss Improvement: {final_improvement:+.2f}%")
    print(f"  Best Test Loss Improvement:  {best_improvement:+.2f}%")
    
    summary_lines.append(f"\nPerformance Improvement:")
    summary_lines.append(f"  Final Test Loss Improvement: {final_improvement:+.2f}%")
    summary_lines.append(f"  Best Test Loss Improvement:  {best_improvement:+.2f}%")
    
    # 保存总结
    summary_path = f'{results_dir}/experiment_summary_500epochs.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Experiment summary saved to {summary_path}")
    print(f"✓ All results saved in: {results_dir}")


def main():
    """运行Heat Equation 500 epochs实验"""
    print(f"🔥 Starting Heat Equation 500 Epochs Experiment")
    print("="*60)
    
    # 1. 创建实验文件夹
    experiment_name = "heat_equation_500epochs"
    results_dir = create_experiment_folder(experiment_name)
    
    # 2. 保存实验配置
    config = {
        'PDE Type': 'Heat Equation',
        'Mathematical Form': '∂u/∂t = α∇²u',
        'Diffusion Coefficient': 0.01,
        'Time Step': 0.1,
        'Number of Samples': 200,
        'Grid Size': '64x64',
        'Training Epochs': 500,
        'Learning Rate': 1e-3,
        'Weight Decay': 1e-4,
        'Scheduler': 'ReduceLROnPlateau(patience=20, factor=0.5)',
        'Early Stopping': 'patience=100',
        'FNO Modes': '16x16',
        'Hidden Channels': 32,
        'FNO Layers': 4,
        'Adaptive Experts': 4,
        'Batch Size': 16
    }
    save_experiment_config(results_dir, config)
    
    # 3. 生成数据
    print(f"\n📊 Generating Heat Equation data...")
    inputs, outputs = generate_heat_equation_data(n_samples=200, grid_size=64)
    
    # 分割数据
    train_size = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ Data prepared: {train_size} train, {len(test_inputs)} test samples")
    
    # 4. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    results = train_models_500epochs(train_loader, test_loader, device, results_dir)
    
    # 5. 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_500epoch_results(results, test_loader, device, results_dir)
    
    # 6. 打印总结
    print_500epoch_summary(results, results_dir)
    
    return results_dir


if __name__ == "__main__":
    print("🔥 Heat Equation 500 Epochs Experiment")
    print("This experiment trains both Original FNO and Adaptive FNO for 500 epochs")
    print("to demonstrate the long-term convergence behavior and performance gains.")
    
    results_dir = main()
    
    print(f"\n🎉 500 Epochs Experiment completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Check the 'figures' subfolder for detailed visualizations")
    print(f"🤖 Check the 'models' subfolder for best and final models")
    print(f"📝 Check experiment_summary_500epochs.txt for detailed results")
