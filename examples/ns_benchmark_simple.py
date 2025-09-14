"""
Navier-Stokes基准测试 - 简化版
复现论文Table 1的结果格式，对比FNO-2D和FnoMoE
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


def create_experiment_folder():
    """创建实验文件夹"""
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/navier_stokes_benchmark'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    print(f"✓ Created experiment folder: {results_dir}")
    return results_dir


def generate_ns_vorticity_data(n_samples=1000, grid_size=64, viscosity=1e-4, T=30):
    """
    生成Navier-Stokes涡度数据
    使用简化的物理模型来快速生成训练数据
    """
    print(f"Generating {n_samples} NS samples (ν={viscosity}, T={T})...")
    
    # 创建网格
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs = []
    outputs = []
    
    for i in range(n_samples):
        # 生成初始涡度场 - 使用随机傅里叶模式
        w0 = np.zeros((grid_size, grid_size))
        
        # 添加多个随机模式
        for k in range(1, 8):  # 低频模式
            for l in range(1, 8):
                if k**2 + l**2 <= 25:  # 限制频率范围
                    amp = np.random.normal(0, 1.0 / (k**2 + l**2 + 1))
                    phase = np.random.uniform(0, 2*np.pi)
                    w0 += amp * np.sin(k*X + l*Y + phase)
        
        # 简化的时间演化：主要是扩散
        # 在频域中应用扩散算子
        w0_fft = np.fft.fft2(w0)
        
        # 频率网格
        kx = np.fft.fftfreq(grid_size, 2*np.pi/grid_size)
        ky = np.fft.fftfreq(grid_size, 2*np.pi/grid_size)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        
        # 扩散衰减
        decay = np.exp(-viscosity * K2 * T)
        wT_fft = w0_fft * decay
        wT = np.real(np.fft.ifft2(wT_fft))
        
        inputs.append(w0[np.newaxis, :, :])
        outputs.append(wT[np.newaxis, :, :])
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def create_benchmark_models(device):
    """创建基准测试模型"""
    
    # FNO-2D (基于论文配置)
    fno_2d = FNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4
    ).to(device)
    
    # FnoMoE (自适应FNO)
    fnomoe = AdaptiveFNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4
    ).to(device)
    
    models = {
        'FNO-2D': fno_2d,
        'FnoMoE': fnomoe
    }
    
    # 计算参数数量
    param_counts = {}
    for name, model in models.items():
        param_counts[name] = sum(p.numel() for p in model.parameters())
        print(f"{name} parameters: {param_counts[name]:,}")
    
    return models, param_counts


def train_and_benchmark(models, param_counts, train_loader, test_loader, device, results_dir, epochs=50):
    """训练模型并进行基准测试"""
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # 训练设置
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        test_losses = []
        epoch_times = []
        best_test_loss = float('inf')
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
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
            epoch_time = time.time() - epoch_start_time
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            epoch_times.append(epoch_time)
            
            scheduler.step(avg_test_loss)
            
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), f'{results_dir}/models/{name.lower().replace("-", "_")}_best.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}, Time={epoch_time:.2f}s")
        
        total_training_time = time.time() - total_start_time
        avg_epoch_time = np.mean(epoch_times)
        
        results[name] = {
            'model': model,
            'parameters': param_counts[name],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss,
            'final_test_loss': test_losses[-1],
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time
        }
        
        print(f"  ✓ Training completed: {total_training_time:.1f}s total, {avg_epoch_time:.2f}s/epoch")
    
    return results


def evaluate_inference_speed(results, test_loader, device):
    """评估推理速度"""
    
    print(f"\n⚡ Evaluating inference speed...")
    
    eval_results = {}
    
    for name, result in results.items():
        model = result['model']
        model.eval()
        
        inference_times = []
        total_mse = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_inputs, batch_outputs in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)
                
                # 预热
                _ = model(batch_inputs)
                
                # 测量推理时间
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                predictions = model(batch_inputs)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                inference_time = time.time() - start_time
                
                # 计算MSE
                mse = F.mse_loss(predictions, batch_outputs)
                total_mse += mse.item() * batch_inputs.size(0)
                total_samples += batch_inputs.size(0)
                
                # 记录每个样本的推理时间
                inference_times.append(inference_time / batch_inputs.size(0))
        
        avg_mse = total_mse / total_samples
        avg_inference_time = np.mean(inference_times)
        
        eval_results[name] = {
            'mse': avg_mse,
            'inference_time_per_sample': avg_inference_time
        }
    
    return eval_results


def create_benchmark_table(results, eval_results, viscosity, T, N):
    """创建类似论文的基准测试表格"""
    
    print(f"\n{'='*80}")
    print(f"NAVIER-STOKES BENCHMARKS (ν={viscosity}, T={T}, N={N})")
    print(f"Resolution: 64×64 for both training and testing")
    print(f"{'='*80}")
    
    # 表格标题
    print(f"{'Config':<10} {'Parameters':<12} {'Time per':<12} {'MSE':<12}")
    print(f"{'':10} {'':12} {'epoch (s)':<12} {'':12}")
    print(f"{'-'*50}")
    
    # 数据行
    for name, result in results.items():
        params = result['parameters']
        time_per_epoch = result['avg_epoch_time']
        mse = eval_results[name]['mse']
        
        print(f"{name:<10} {params:<12,} {time_per_epoch:<12.2f} {mse:<12.4f}")
    
    # 性能对比
    fno_mse = eval_results['FNO-2D']['mse']
    fnomoe_mse = eval_results['FnoMoE']['mse']
    improvement = ((fno_mse - fnomoe_mse) / fno_mse) * 100
    
    print(f"\n📊 Performance Comparison:")
    print(f"FNO-2D achieves MSE: {fno_mse:.6f}")
    print(f"FnoMoE achieves MSE: {fnomoe_mse:.6f}")
    
    if improvement > 0:
        print(f"FnoMoE shows {improvement:.2f}% improvement over FNO-2D")
    else:
        print(f"FNO-2D shows {-improvement:.2f}% better performance than FnoMoE")
    
    # 参数对比
    fno_params = results['FNO-2D']['parameters']
    fnomoe_params = results['FnoMoE']['parameters']
    param_overhead = ((fnomoe_params - fno_params) / fno_params) * 100
    
    print(f"\nParameter Overhead:")
    print(f"FnoMoE adds {param_overhead:.1f}% more parameters than FNO-2D")
    
    return {
        'improvement': improvement,
        'param_overhead': param_overhead,
        'fno_mse': fno_mse,
        'fnomoe_mse': fnomoe_mse
    }


def visualize_training_and_predictions(results, eval_results, test_loader, device, results_dir):
    """可视化训练过程和预测结果"""
    
    # 1. 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'FNO-2D': 'blue', 'FnoMoE': 'red'}
    
    for name, result in results.items():
        color = colors[name]
        epochs = range(1, len(result['train_losses']) + 1)
        
        axes[0].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
        axes[1].plot(epochs, result['test_losses'], color=color, label=f'{name} Test', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
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
    
    plt.suptitle('Navier-Stokes: FNO-2D vs FnoMoE Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/ns_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测结果对比
    test_inputs, test_outputs = next(iter(test_loader))
    test_inputs = test_inputs[:3].to(device)
    test_outputs = test_outputs[:3].to(device)
    
    predictions = {}
    for name, result in results.items():
        result['model'].eval()
        with torch.no_grad():
            pred = result['model'](test_inputs)
            predictions[name] = pred.cpu()
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i in range(3):
        # 初始涡度
        axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='RdBu')
        axes[i, 0].set_title(f'Initial Vorticity {i+1}')
        axes[i, 0].axis('off')
        
        # 真实演化
        axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='RdBu')
        axes[i, 1].set_title(f'True Evolution {i+1}')
        axes[i, 1].axis('off')
        
        # FNO-2D预测
        fno_pred = predictions['FNO-2D'][i, 0]
        axes[i, 2].imshow(fno_pred.numpy(), cmap='RdBu')
        axes[i, 2].set_title(f'FNO-2D Pred {i+1}')
        axes[i, 2].axis('off')
        
        # FnoMoE预测
        fnomoe_pred = predictions['FnoMoE'][i, 0]
        axes[i, 3].imshow(fnomoe_pred.numpy(), cmap='RdBu')
        axes[i, 3].set_title(f'FnoMoE Pred {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        fno_error = torch.abs(fno_pred - test_outputs[i, 0].cpu())
        fnomoe_error = torch.abs(fnomoe_pred - test_outputs[i, 0].cpu())
        error_diff = fno_error - fnomoe_error
        
        im = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: FnoMoE Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Navier-Stokes: Vorticity Evolution Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/ns_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_benchmark_results(results, eval_results, benchmark_summary, results_dir, config):
    """保存基准测试结果"""
    
    summary_lines = []
    summary_lines.append("NAVIER-STOKES BENCHMARK RESULTS")
    summary_lines.append("="*50)
    summary_lines.append("")
    
    # 配置
    summary_lines.append("Configuration:")
    for key, value in config.items():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("")
    
    # 基准测试表格
    summary_lines.append("Benchmark Table:")
    summary_lines.append(f"{'Config':<10} {'Parameters':<12} {'Time/Epoch':<12} {'MSE':<12}")
    summary_lines.append("-" * 50)
    
    for name, result in results.items():
        params = result['parameters']
        time_per_epoch = result['avg_epoch_time']
        mse = eval_results[name]['mse']
        summary_lines.append(f"{name:<10} {params:<12,} {time_per_epoch:<12.2f} {mse:<12.6f}")
    
    summary_lines.append("")
    summary_lines.append("Performance Summary:")
    summary_lines.append(f"  MSE Improvement: {benchmark_summary['improvement']:+.2f}%")
    summary_lines.append(f"  Parameter Overhead: {benchmark_summary['param_overhead']:+.1f}%")
    
    # 保存到文件
    summary_path = f'{results_dir}/benchmark_results.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Benchmark results saved to {summary_path}")


def main():
    """主函数"""
    print("🌊 Navier-Stokes Benchmark: FNO-2D vs FnoMoE")
    print("="*50)
    
    # 实验配置（基于论文）
    viscosity = 1e-4
    T = 30
    N = 1000
    grid_size = 64
    epochs = 50
    
    config = {
        'Viscosity': viscosity,
        'Final Time': T,
        'Samples': N,
        'Resolution': f'{grid_size}×{grid_size}',
        'Epochs': epochs,
        'Batch Size': 16
    }
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 生成数据
    print(f"\n📊 Generating data...")
    inputs, outputs = generate_ns_vorticity_data(N, grid_size, viscosity, T)
    
    # 数据分割
    train_size = int(0.8 * N)
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
    
    models, param_counts = create_benchmark_models(device)
    
    # 训练和基准测试
    results = train_and_benchmark(models, param_counts, train_loader, test_loader, device, results_dir, epochs)
    
    # 评估推理速度
    eval_results = evaluate_inference_speed(results, test_loader, device)
    
    # 创建基准测试表格
    benchmark_summary = create_benchmark_table(results, eval_results, viscosity, T, N)
    
    # 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_training_and_predictions(results, eval_results, test_loader, device, results_dir)
    
    # 保存结果
    save_benchmark_results(results, eval_results, benchmark_summary, results_dir, config)
    
    print(f"\n🎉 Navier-Stokes benchmark completed!")
    print(f"📁 Results saved in: {results_dir}")
    
    return results_dir


if __name__ == "__main__":
    main()


