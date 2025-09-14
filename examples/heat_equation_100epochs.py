"""
Heat Equation实验 - 100 epochs版本
在500 epochs之前先验证代码和查看初步结果
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

def main():
    """运行Heat Equation 100 epochs实验"""
    print(f"🔥 Heat Equation 100 Epochs Experiment")
    print("="*50)
    
    # 创建实验文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/heat_equation_100epochs'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # 保存配置
    config_text = """Heat Equation 100 Epochs Configuration
=======================================
PDE Type: Heat Equation
Mathematical Form: ∂u/∂t = α∇²u
Diffusion Coefficient: 0.01
Time Step: 0.1
Number of Samples: 150
Grid Size: 64x64
Training Epochs: 100
Learning Rate: 1e-3
Weight Decay: 1e-4
FNO Modes: 16x16
Hidden Channels: 32
FNO Layers: 4
Adaptive Experts: 4
Batch Size: 16"""
    
    with open(f'{results_dir}/experiment_config.txt', 'w') as f:
        f.write(config_text)
    
    # 生成数据
    print(f"\n📊 Generating Heat Equation data...")
    n_samples = 150
    grid_size = 64
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs, outputs = [], []
    
    for i in range(n_samples):
        k1, k2 = np.random.randint(1, 8, 2)
        phase = np.random.uniform(0, 2*np.pi)
        
        # 初始条件
        u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
        
        # 时间演化
        diffusion = 0.01
        t = 0.1
        decay = np.exp(-diffusion * (k1**2 + k2**2) * t)
        u1 = decay * u0
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    
    inputs = torch.FloatTensor(np.array(inputs))
    outputs = torch.FloatTensor(np.array(outputs))
    
    # 分割数据
    train_size = int(0.8 * n_samples)
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]
    
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"✓ Data: {train_size} train, {len(test_inputs)} test samples")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    original_fno = FNO2d(
        n_modes_height=16, n_modes_width=16,
        in_channels=1, out_channels=1,
        hidden_channels=32, n_layers=4
    ).to(device)
    
    adaptive_fno = AdaptiveFNO2d(
        n_modes_height=16, n_modes_width=16,
        in_channels=1, out_channels=1,
        hidden_channels=32, n_layers=4,
        n_experts=4
    ).to(device)
    
    models = {'Original FNO': original_fno, 'Adaptive FNO': adaptive_fno}
    results = {}
    
    # 训练模型
    for name, model in models.items():
        print(f"\n🚀 Training {name} (100 epochs)...")
        start_time = time.time()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        train_losses, test_losses = [], []
        best_test_loss = float('inf')
        
        for epoch in range(100):
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
            
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                # 保存最佳模型
                torch.save(model.state_dict(), f'{results_dir}/models/{name.lower().replace(" ", "_")}_best.pth')
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: Train={avg_train_loss:.8f}, Test={avg_test_loss:.8f}, Best={best_test_loss:.8f}")
        
        training_time = time.time() - start_time
        
        results[name] = {
            'model': model,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': test_losses[-1],
            'best_test_loss': best_test_loss,
            'training_time': training_time
        }
        
        # 保存最终模型
        torch.save(model.state_dict(), f'{results_dir}/models/{name.lower().replace(" ", "_")}_final.pth')
        print(f"  ✓ Training completed in {training_time:.1f}s")
    
    # 可视化结果
    print(f"\n📈 Generating visualizations...")
    
    # 1. 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
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
    
    plt.suptitle('Heat Equation: 100 Epochs Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/training_curves_100epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测对比
    test_inputs_sample, test_outputs_sample = next(iter(test_loader))
    test_inputs_sample = test_inputs_sample[:3].to(device)
    test_outputs_sample = test_outputs_sample[:3].to(device)
    
    predictions = {}
    for name, result in results.items():
        result['model'].eval()
        with torch.no_grad():
            pred = result['model'](test_inputs_sample)
            predictions[name] = pred.cpu()
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i in range(3):
        # 输入
        axes[i, 0].imshow(test_inputs_sample[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        axes[i, 1].imshow(test_outputs_sample[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # 原始FNO
        original_pred = predictions['Original FNO'][i, 0]
        axes[i, 2].imshow(original_pred.numpy(), cmap='viridis')
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].axis('off')
        
        # 自适应FNO
        adaptive_pred = predictions['Adaptive FNO'][i, 0]
        axes[i, 3].imshow(adaptive_pred.numpy(), cmap='viridis')
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        original_error = torch.abs(original_pred - test_outputs_sample[i, 0].cpu())
        adaptive_error = torch.abs(adaptive_pred - test_outputs_sample[i, 0].cpu())
        error_diff = original_error - adaptive_error
        
        im = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: Adaptive Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Heat Equation: 100 Epochs Prediction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/predictions_100epochs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"HEAT EQUATION 100 EPOCHS SUMMARY")
    print(f"{'='*60}")
    
    summary_lines = ["HEAT EQUATION 100 EPOCHS SUMMARY", "="*60]
    
    for name, result in results.items():
        final_loss = result['final_test_loss']
        best_loss = result['best_test_loss']
        train_time = result['training_time']
        param_count = sum(p.numel() for p in result['model'].parameters())
        
        print(f"\n{name}:")
        print(f"  Final Test Loss:  {final_loss:.8f}")
        print(f"  Best Test Loss:   {best_loss:.8f}")
        print(f"  Training Time:    {train_time:.1f}s")
        print(f"  Parameters:       {param_count:,}")
        
        summary_lines.extend([
            f"\n{name}:",
            f"  Final Test Loss:  {final_loss:.8f}",
            f"  Best Test Loss:   {best_loss:.8f}",
            f"  Training Time:    {train_time:.1f}s",
            f"  Parameters:       {param_count:,}"
        ])
    
    # 计算改进
    original_final = results['Original FNO']['final_test_loss']
    adaptive_final = results['Adaptive FNO']['final_test_loss']
    final_improvement = ((original_final - adaptive_final) / original_final) * 100
    
    original_best = results['Original FNO']['best_test_loss']
    adaptive_best = results['Adaptive FNO']['best_test_loss']
    best_improvement = ((original_best - adaptive_best) / original_best) * 100
    
    print(f"\nPerformance Improvement:")
    print(f"  Final Test Loss: {final_improvement:+.2f}%")
    print(f"  Best Test Loss:  {best_improvement:+.2f}%")
    
    summary_lines.extend([
        f"\nPerformance Improvement:",
        f"  Final Test Loss: {final_improvement:+.2f}%",
        f"  Best Test Loss:  {best_improvement:+.2f}%"
    ])
    
    # 保存总结
    with open(f'{results_dir}/experiment_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Results saved to: {results_dir}")
    print(f"📊 Training curves: figures/training_curves_100epochs.png")
    print(f"🎯 Predictions: figures/predictions_100epochs.png")
    print(f"🤖 Models: models/ (best and final versions)")
    
    if best_improvement > 0:
        print(f"\n🎉 Adaptive FNO shows {best_improvement:.2f}% improvement!")
        print(f"💡 Ready to run 500 epochs for even better results")
    else:
        print(f"\n📝 Consider adjusting hyperparameters for 500 epochs run")

if __name__ == "__main__":
    main()


