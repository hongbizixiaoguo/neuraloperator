"""
对比原始FNO和自适应FNO的性能
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# 导入模型
import sys
sys.path.append('/root/autodl-tmp/neuraloperator')
from neuralop.models.fno import FNO2d
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d

def generate_complex_data(n_samples=1000, grid_size=64):
    """
    生成更复杂的2D PDE数据，包含多种频率成分
    """
    inputs = []
    outputs = []
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    for _ in range(n_samples):
        # 生成多个频率成分的组合
        k1, k2 = np.random.randint(1, 8, 2)  # 低频
        k3, k4 = np.random.randint(8, 15, 2)  # 中频
        k5, k6 = np.random.randint(15, 25, 2)  # 高频
        
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        phase3 = np.random.uniform(0, 2*np.pi)
        
        # 输入：多频率成分的组合
        u0 = (0.6 * np.sin(2*np.pi*k1*X + phase1) * np.cos(2*np.pi*k2*Y) +
              0.3 * np.sin(2*np.pi*k3*X + phase2) * np.sin(2*np.pi*k4*Y) +
              0.1 * np.cos(2*np.pi*k5*X + phase3) * np.cos(2*np.pi*k6*Y))
        
        # 输出：模拟热扩散方程的解（不同频率衰减不同）
        diffusion = 0.01
        t = 0.1
        decay1 = np.exp(-diffusion * (k1**2 + k2**2) * t)
        decay2 = np.exp(-diffusion * (k3**2 + k4**2) * t)
        decay3 = np.exp(-diffusion * (k5**2 + k6**2) * t)
        
        u1 = (0.6 * decay1 * np.sin(2*np.pi*k1*X + phase1) * np.cos(2*np.pi*k2*Y) +
              0.3 * decay2 * np.sin(2*np.pi*k3*X + phase2) * np.sin(2*np.pi*k4*Y) +
              0.1 * decay3 * np.cos(2*np.pi*k5*X + phase3) * np.cos(2*np.pi*k6*Y))
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def create_models(device):
    """
    创建原始FNO和自适应FNO模型
    """
    # 原始FNO
    original_fno = FNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        non_linearity=F.gelu,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',
        channel_mlp_skip='soft-gating'
    ).to(device)
    
    # 自适应FNO
    adaptive_fno = AdaptiveFNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4,
        temperature=1.0,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        non_linearity=F.gelu,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',
        channel_mlp_skip='soft-gating'
    ).to(device)
    
    return original_fno, adaptive_fno


def train_model(model, train_loader, test_loader, device, model_name, n_epochs=50):
    """
    训练模型并记录性能指标
    """
    print(f"\nTraining {model_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    training_times = []
    
    model.train()
    
    for epoch in range(n_epochs):
        # 训练阶段
        start_time = time.time()
        train_loss = 0.0
        
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                  f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_times': training_times,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'avg_epoch_time': np.mean(training_times)
    }


def evaluate_models(original_fno, adaptive_fno, test_loader, device):
    """
    评估模型性能
    """
    print("\nEvaluating models...")
    
    results = {}
    
    for model, name in [(original_fno, 'Original FNO'), (adaptive_fno, 'Adaptive FNO')]:
        model.eval()
        
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                # 测量推理时间
                start_time = time.time()
                predictions = model(batch_inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / batch_inputs.size(0))  # 每个样本的时间
                
                # 计算误差
                mse = F.mse_loss(predictions, batch_outputs)
                mae = F.l1_loss(predictions, batch_outputs)
                
                total_mse += mse.item() * batch_inputs.size(0)
                total_mae += mae.item() * batch_inputs.size(0)
                total_samples += batch_inputs.size(0)
        
        results[name] = {
            'mse': total_mse / total_samples,
            'mae': total_mae / total_samples,
            'avg_inference_time': np.mean(inference_times) * 1000,  # ms
            'std_inference_time': np.std(inference_times) * 1000   # ms
        }
    
    return results


def visualize_predictions(original_fno, adaptive_fno, test_inputs, test_outputs, device):
    """
    可视化预测结果
    """
    print("\nGenerating visualizations...")
    
    original_fno.eval()
    adaptive_fno.eval()
    
    # 选择几个测试样本
    n_samples = 3
    test_sample_inputs = test_inputs[:n_samples].to(device)
    test_sample_outputs = test_outputs[:n_samples].to(device)
    
    with torch.no_grad():
        original_preds = original_fno(test_sample_inputs)
        adaptive_preds = adaptive_fno(test_sample_inputs)
    
    # 创建可视化
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    
    for i in range(n_samples):
        # 输入
        axes[i, 0].imshow(test_sample_inputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        axes[i, 1].imshow(test_sample_outputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # 原始FNO预测
        axes[i, 2].imshow(original_preds[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].axis('off')
        
        # 自适应FNO预测
        axes[i, 3].imshow(adaptive_preds[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        original_error = torch.abs(original_preds[i, 0] - test_sample_outputs[i, 0])
        adaptive_error = torch.abs(adaptive_preds[i, 0] - test_sample_outputs[i, 0])
        
        # 显示误差差异（红色表示自适应FNO误差更大，蓝色表示原始FNO误差更大）
        error_diff = original_error - adaptive_error
        axes[i, 4].imshow(error_diff.cpu().numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: Adaptive Better)')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    # 创建实验结果文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/2d_synthetic'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(f'{results_dir}/fno_comparison_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Predictions comparison saved to '{results_dir}/fno_comparison_predictions.png'")
    plt.show()


def plot_training_comparison(original_results, adaptive_results):
    """
    绘制训练过程对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(original_results['train_losses']) + 1)
    
    # 训练损失对比
    axes[0, 0].plot(epochs, original_results['train_losses'], 'b-', label='Original FNO', linewidth=2)
    axes[0, 0].plot(epochs, adaptive_results['train_losses'], 'r-', label='Adaptive FNO', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 测试损失对比
    axes[0, 1].plot(epochs, original_results['test_losses'], 'b-', label='Original FNO', linewidth=2)
    axes[0, 1].plot(epochs, adaptive_results['test_losses'], 'r-', label='Adaptive FNO', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Test Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 训练时间对比
    axes[1, 0].plot(epochs, original_results['training_times'], 'b-', label='Original FNO', linewidth=2)
    axes[1, 0].plot(epochs, adaptive_results['training_times'], 'r-', label='Adaptive FNO', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Time (s)')
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 性能指标对比
    metrics = ['Final Train Loss', 'Final Test Loss', 'Avg Epoch Time (s)']
    original_values = [original_results['final_train_loss'], 
                      original_results['final_test_loss'],
                      original_results['avg_epoch_time']]
    adaptive_values = [adaptive_results['final_train_loss'], 
                      adaptive_results['final_test_loss'],
                      adaptive_results['avg_epoch_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, original_values, width, label='Original FNO', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, adaptive_values, width, label='Adaptive FNO', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Final Performance Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    # 保存到实验结果文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/2d_synthetic'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(f'{results_dir}/fno_comparison_training.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training comparison saved to '{results_dir}/fno_comparison_training.png'")
    plt.show()


def print_comparison_summary(original_results, adaptive_results, eval_results):
    """
    打印对比总结
    """
    print("\n" + "="*60)
    print("FNO COMPARISON SUMMARY")
    print("="*60)
    
    print("\n📊 TRAINING PERFORMANCE:")
    print(f"{'Metric':<25} {'Original FNO':<15} {'Adaptive FNO':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # 训练损失
    orig_train = original_results['final_train_loss']
    adapt_train = adaptive_results['final_train_loss']
    train_improve = (orig_train - adapt_train) / orig_train * 100
    print(f"{'Final Train Loss':<25} {orig_train:<15.6f} {adapt_train:<15.6f} {train_improve:>+13.2f}%")
    
    # 测试损失
    orig_test = original_results['final_test_loss']
    adapt_test = adaptive_results['final_test_loss']
    test_improve = (orig_test - adapt_test) / orig_test * 100
    print(f"{'Final Test Loss':<25} {orig_test:<15.6f} {adapt_test:<15.6f} {test_improve:>+13.2f}%")
    
    # 训练时间
    orig_time = original_results['avg_epoch_time']
    adapt_time = adaptive_results['avg_epoch_time']
    time_change = (adapt_time - orig_time) / orig_time * 100
    print(f"{'Avg Epoch Time (s)':<25} {orig_time:<15.2f} {adapt_time:<15.2f} {time_change:>+13.2f}%")
    
    print("\n🎯 EVALUATION PERFORMANCE:")
    print(f"{'Metric':<25} {'Original FNO':<15} {'Adaptive FNO':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # MSE
    orig_mse = eval_results['Original FNO']['mse']
    adapt_mse = eval_results['Adaptive FNO']['mse']
    mse_improve = (orig_mse - adapt_mse) / orig_mse * 100
    print(f"{'MSE':<25} {orig_mse:<15.6f} {adapt_mse:<15.6f} {mse_improve:>+13.2f}%")
    
    # MAE
    orig_mae = eval_results['Original FNO']['mae']
    adapt_mae = eval_results['Adaptive FNO']['mae']
    mae_improve = (orig_mae - adapt_mae) / orig_mae * 100
    print(f"{'MAE':<25} {orig_mae:<15.6f} {adapt_mae:<15.6f} {mae_improve:>+13.2f}%")
    
    # 推理时间
    orig_inf = eval_results['Original FNO']['avg_inference_time']
    adapt_inf = eval_results['Adaptive FNO']['avg_inference_time']
    inf_change = (adapt_inf - orig_inf) / orig_inf * 100
    print(f"{'Inference Time (ms)':<25} {orig_inf:<15.2f} {adapt_inf:<15.2f} {inf_change:>+13.2f}%")
    
    print("\n🏆 OVERALL ASSESSMENT:")
    if test_improve > 0 and mse_improve > 0:
        print("✅ Adaptive FNO shows BETTER accuracy than Original FNO")
    elif test_improve < -5 or mse_improve < -5:
        print("❌ Adaptive FNO shows WORSE accuracy than Original FNO")
    else:
        print("➖ Adaptive FNO shows SIMILAR accuracy to Original FNO")
    
    if abs(inf_change) < 10:
        print("✅ Adaptive FNO has SIMILAR computational cost")
    elif inf_change > 10:
        print("⚠️  Adaptive FNO has HIGHER computational cost")
    else:
        print("✅ Adaptive FNO has LOWER computational cost")
    
    print("="*60)


def main():
    """
    主对比函数
    """
    print("🚀 Starting FNO vs Adaptive FNO Comparison")
    print("="*50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成数据
    print("\n📊 Generating complex synthetic data...")
    train_inputs, train_outputs = generate_complex_data(n_samples=800, grid_size=64)
    test_inputs, test_outputs = generate_complex_data(n_samples=200, grid_size=64)
    
    # 创建数据加载器
    batch_size = 16
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("\n🏗️  Creating models...")
    original_fno, adaptive_fno = create_models(device)
    
    # 训练模型
    n_epochs = 50
    original_results = train_model(original_fno, train_loader, test_loader, device, "Original FNO", n_epochs)
    adaptive_results = train_model(adaptive_fno, train_loader, test_loader, device, "Adaptive FNO", n_epochs)
    
    # 评估模型
    eval_results = evaluate_models(original_fno, adaptive_fno, test_loader, device)
    
    # 可视化预测结果
    visualize_predictions(original_fno, adaptive_fno, test_inputs, test_outputs, device)
    
    # 绘制训练过程对比
    plot_training_comparison(original_results, adaptive_results)
    
    # 打印对比总结
    print_comparison_summary(original_results, adaptive_results, eval_results)
    
    # 保存模型
    torch.save(original_fno.state_dict(), '/root/autodl-tmp/neuraloperator/original_fno_comparison.pth')
    torch.save(adaptive_fno.state_dict(), '/root/autodl-tmp/neuraloperator/adaptive_fno_comparison.pth')
    
    print("\n💾 Models saved successfully!")
    print("🎉 Comparison completed!")


if __name__ == "__main__":
    main()
