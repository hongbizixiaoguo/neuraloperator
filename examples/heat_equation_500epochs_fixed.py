"""
Heat Equation实验 - 500 epochs训练 (修复版)
修复了scheduler参数问题，添加了更好的错误处理
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

try:
    from neuralop.models.fno import FNO2d
    from neuralop.models.final_adaptive_fno import AdaptiveFNO2d
    print("✓ Successfully imported FNO models")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


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
        f.write("Heat Equation Experiment Configuration (500 Epochs - Fixed)\n")
        f.write("=" * 60 + "\n")
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
    """训练原始FNO和自适应FNO - 500 epochs (修复版)"""
    
    try:
        # 创建模型
        print("Creating models...")
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
        
        print("✓ Models created successfully")
        
        models = {
            'Original FNO': original_fno,
            'Adaptive FNO': adaptive_fno
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🚀 Training {name} for 500 epochs...")
            start_time = time.time()
            
            # 优化器和调度器 (修复版)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                patience=20, 
                factor=0.5,
                min_lr=1e-6
            )
            
            train_losses = []
            test_losses = []
            best_test_loss = float('inf')
            patience_counter = 0
            
            # 训练循环 - 500 epochs
            for epoch in range(500):
                try:
                    # 训练阶段
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
                    
                    # 测试阶段
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
                    
                    # 学习率调度
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(avg_test_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    
                    if new_lr < old_lr:
                        print(f"    Learning rate reduced to {new_lr:.2e}")
                    
                    # 保存最佳模型
                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
                        patience_counter = 0
                        best_model_path = f'{results_dir}/models/{name.lower().replace(" ", "_")}_best_model.pth'
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                    
                    # 打印进度
                    if (epoch + 1) % 25 == 0:
                        print(f"  Epoch {epoch+1:3d}/500: Train={avg_train_loss:.8f}, Test={avg_test_loss:.8f}, Best={best_test_loss:.8f}, LR={new_lr:.2e}")
                    
                    # 早停检查
                    if patience_counter >= 100:
                        print(f"  Early stopping at epoch {epoch+1} (no improvement for 100 epochs)")
                        break
                        
                except Exception as e:
                    print(f"  ❌ Error in epoch {epoch+1}: {e}")
                    break
            
            training_time = time.time() - start_time
            
            results[name] = {
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_test_loss': test_losses[-1] if test_losses else float('inf'),
                'best_test_loss': best_test_loss,
                'total_epochs': len(train_losses),
                'training_time': training_time
            }
            
            # 保存最终模型
            final_model_path = f'{results_dir}/models/{name.lower().replace(" ", "_")}_final_model.pth'
            torch.save(model.state_dict(), final_model_path)
            print(f"  ✓ Training completed in {training_time:.1f}s ({len(train_losses)} epochs)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in training: {e}")
        return {}


def visualize_results(results, test_loader, device, results_dir):
    """可视化500 epoch训练结果"""
    
    if not results:
        print("❌ No results to visualize")
        return
    
    try:
        # 1. 训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失（线性和对数）
        for name, result in results.items():
            color = 'blue' if 'Original' in name else 'red'
            epochs = range(1, len(result['train_losses']) + 1)
            
            axes[0, 0].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
            axes[0, 1].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
            axes[1, 0].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
            axes[1, 1].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
        
        # 设置图表
        axes[0, 0].set_title('Training Loss (Linear)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training Loss (Log)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        axes[1, 0].set_title('Test Loss (Linear)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Test Loss (Log)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        
        plt.suptitle('Heat Equation: 500 Epochs Training (Fixed)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        training_fig_path = f'{results_dir}/figures/heat_500epochs_training_fixed.png'
        plt.savefig(training_fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to {training_fig_path}")
        plt.show()
        
        # 2. 预测结果对比
        models = {name: result['model'] for name, result in results.items()}
        
        # 获取测试样本
        test_inputs, test_outputs = next(iter(test_loader))
        test_inputs = test_inputs[:3].to(device)
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
            if 'Original FNO' in predictions:
                original_pred = predictions['Original FNO'][i, 0]
                axes[i, 2].imshow(original_pred.numpy(), cmap='viridis')
                axes[i, 2].set_title(f'Original FNO {i+1}')
                axes[i, 2].axis('off')
            
            # 自适应FNO预测
            if 'Adaptive FNO' in predictions:
                adaptive_pred = predictions['Adaptive FNO'][i, 0]
                axes[i, 3].imshow(adaptive_pred.numpy(), cmap='viridis')
                axes[i, 3].set_title(f'Adaptive FNO {i+1}')
                axes[i, 3].axis('off')
            
            # 误差对比
            if 'Original FNO' in predictions and 'Adaptive FNO' in predictions:
                original_error = torch.abs(predictions['Original FNO'][i, 0] - test_outputs[i, 0].cpu())
                adaptive_error = torch.abs(predictions['Adaptive FNO'][i, 0] - test_outputs[i, 0].cpu())
                error_diff = original_error - adaptive_error
                
                im = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
                axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: Adaptive Better)')
                axes[i, 4].axis('off')
        
        plt.suptitle('Heat Equation: 500 Epochs Prediction (Fixed)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        prediction_fig_path = f'{results_dir}/figures/heat_500epochs_prediction_fixed.png'
        plt.savefig(prediction_fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Prediction comparison saved to {prediction_fig_path}")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")


def print_summary(results, results_dir):
    """打印和保存实验总结"""
    
    if not results:
        print("❌ No results to summarize")
        return
    
    print(f"\n{'='*70}")
    print(f"HEAT EQUATION 500 EPOCHS EXPERIMENT SUMMARY (FIXED)")
    print(f"{'='*70}")
    
    summary_lines = []
    summary_lines.append(f"HEAT EQUATION 500 EPOCHS EXPERIMENT SUMMARY (FIXED)")
    summary_lines.append("="*70)
    
    for name, result in results.items():
        final_train_loss = result.get('final_train_loss', 'N/A')
        final_test_loss = result.get('final_test_loss', 'N/A')
        best_test_loss = result.get('best_test_loss', 'N/A')
        total_epochs = result.get('total_epochs', 0)
        training_time = result.get('training_time', 0)
        
        try:
            param_count = sum(p.numel() for p in result['model'].parameters())
        except:
            param_count = 'N/A'
        
        print(f"\n{name}:")
        print(f"  Total Epochs:      {total_epochs}")
        print(f"  Final Train Loss:  {final_train_loss}")
        print(f"  Final Test Loss:   {final_test_loss}")
        print(f"  Best Test Loss:    {best_test_loss}")
        print(f"  Training Time:     {training_time:.1f}s")
        print(f"  Parameters:        {param_count}")
        
        summary_lines.extend([
            f"\n{name}:",
            f"  Total Epochs:      {total_epochs}",
            f"  Final Train Loss:  {final_train_loss}",
            f"  Final Test Loss:   {final_test_loss}",
            f"  Best Test Loss:    {best_test_loss}",
            f"  Training Time:     {training_time:.1f}s",
            f"  Parameters:        {param_count}"
        ])
    
    # 计算改进
    if 'Original FNO' in results and 'Adaptive FNO' in results:
        try:
            original_final = results['Original FNO']['final_test_loss']
            adaptive_final = results['Adaptive FNO']['final_test_loss']
            
            if isinstance(original_final, (int, float)) and isinstance(adaptive_final, (int, float)):
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
        except Exception as e:
            print(f"Could not calculate improvement: {e}")
    
    # 保存总结
    summary_path = f'{results_dir}/experiment_summary_500epochs_fixed.txt'
    try:
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"\n✓ Experiment summary saved to {summary_path}")
    except Exception as e:
        print(f"❌ Could not save summary: {e}")
    
    print(f"✓ All results saved in: {results_dir}")


def main():
    """运行Heat Equation 500 epochs实验 (修复版)"""
    print(f"🔥 Heat Equation 500 Epochs Experiment (Fixed Version)")
    print("="*60)
    
    try:
        # 1. 创建实验文件夹
        experiment_name = "heat_equation_500epochs_fixed"
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
            'Learning Rate': '1e-3 (with ReduceLROnPlateau)',
            'Weight Decay': 1e-4,
            'Scheduler': 'ReduceLROnPlateau(patience=20, factor=0.5)',
            'Early Stopping': 'patience=100',
            'Gradient Clipping': 'max_norm=1.0',
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
        
        if results:
            # 5. 可视化结果
            print(f"\n📈 Generating visualizations...")
            visualize_results(results, test_loader, device, results_dir)
            
            # 6. 打印总结
            print_summary(results, results_dir)
        else:
            print("❌ Training failed, no results to process")
        
        return results_dir
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return None


if __name__ == "__main__":
    print("🔥 Heat Equation 500 Epochs Experiment (Fixed)")
    print("This is the corrected version with proper error handling")
    
    results_dir = main()
    
    if results_dir:
        print(f"\n🎉 500 Epochs Experiment completed!")
        print(f"📁 Results saved in: {results_dir}")
        print(f"📊 Check the 'figures' subfolder for visualizations")
        print(f"🤖 Check the 'models' subfolder for saved models")
        print(f"📝 Check experiment_summary_500epochs_fixed.txt for results")
    else:
        print(f"\n❌ Experiment failed. Check error messages above.")


