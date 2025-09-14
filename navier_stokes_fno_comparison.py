#!/usr/bin/env python3
"""
简化版Navier-Stokes实验：FNO vs AdaptiveFNO (FnoMoE)对比
使用现有的128分辨率Navier-Stokes数据集
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop import H1Loss, LpLoss
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.utils import count_model_params
from neuralop.models.fno import FNO
from neuralop.models.final_adaptive_fno import AdaptiveFNO


def create_experiment_folder():
    """创建实验文件夹"""
    results_dir = Path('/root/autodl-tmp/neuraloperator/experiments/ns_fno_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'figures').mkdir(exist_ok=True)
    (results_dir / 'models').mkdir(exist_ok=True)
    
    print(f"✓ Created experiment folder: {results_dir}")
    return results_dir


def create_models(device):
    """创建FNO和AdaptiveFNO模型"""
    
    # 标准FNO模型 (基于原始配置)
    fno_model = FNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
    ).to(device)
    
    # 自适应FNO模型 (FnoMoE)
    adaptive_fno_model = AdaptiveFNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        n_experts=4,
        temperature=1.0,
    ).to(device)
    
    # 计算参数数量
    fno_params = count_model_params(fno_model)
    adaptive_params = count_model_params(adaptive_fno_model)
    
    print(f"FNO parameters: {fno_params:,}")
    print(f"AdaptiveFNO parameters: {adaptive_params:,}")
    print(f"Parameter increase: {((adaptive_params - fno_params) / fno_params * 100):+.1f}%")
    
    return {
        'FNO': fno_model,
        'AdaptiveFNO': adaptive_fno_model
    }, {
        'FNO': fno_params,
        'AdaptiveFNO': adaptive_params
    }


def train_model(model, model_name, train_loader, test_loaders, device, n_epochs=50):
    """训练单个模型"""
    
    print(f"\n🚀 Training {model_name}...")
    
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    
    # 损失函数
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss_fn = l2loss  # 使用L2损失训练
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0
        
        for batch_data in train_loader:
            # Navier-Stokes数据集返回字典格式 {'x': inputs, 'y': outputs}
            batch_inputs = batch_data['x'].to(device)
            batch_outputs = batch_data['y'].to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = train_loss_fn(predictions, batch_outputs)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        epoch_test_loss = 0
        num_test_batches = 0
        
        with torch.no_grad():
            for resolution, test_loader in test_loaders.items():
                for batch_data in test_loader:
                    batch_inputs = batch_data['x'].to(device)
                    batch_outputs = batch_data['y'].to(device)
                    predictions = model(batch_inputs)
                    loss = train_loss_fn(predictions, batch_outputs)
                    epoch_test_loss += loss.item()
                    num_test_batches += 1
        
        avg_test_loss = epoch_test_loss / num_test_batches
        test_losses.append(avg_test_loss)
        
        scheduler.step(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"✓ {model_name} training completed in {training_time:.1f}s")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
    }


def evaluate_models(models, param_counts, test_loaders, device):
    """评估模型性能"""
    
    print(f"\n📊 Evaluating models...")
    
    eval_results = {}
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    
    for model_name, model in models.items():
        model.eval()
        
        total_l2_error = 0
        total_h1_error = 0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for resolution, test_loader in test_loaders.items():
                for batch_data in test_loader:
                    batch_inputs = batch_data['x'].to(device)
                    batch_outputs = batch_data['y'].to(device)
                    
                    # 测量推理时间
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.time()
                    
                    predictions = model(batch_inputs)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    
                    # 计算误差
                    l2_error = l2loss(predictions, batch_outputs).item()
                    h1_error = h1loss(predictions, batch_outputs).item()
                    
                    batch_size = batch_inputs.size(0)
                    total_l2_error += l2_error * batch_size
                    total_h1_error += h1_error * batch_size
                    total_samples += batch_size
                    
                    inference_times.append(inference_time / batch_size)
        
        avg_l2_error = total_l2_error / total_samples
        avg_h1_error = total_h1_error / total_samples
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        
        eval_results[model_name] = {
            'l2_error': avg_l2_error,
            'h1_error': avg_h1_error,
            'inference_time_ms': avg_inference_time,
            'parameters': param_counts[model_name]
        }
    
    return eval_results


def visualize_results(training_results, eval_results, test_loaders, device, results_dir):
    """可视化实验结果"""
    
    print(f"\n📈 Generating visualizations...")
    
    # 1. 训练曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {'FNO': 'blue', 'AdaptiveFNO': 'red'}
    
    for model_name, result in training_results.items():
        color = colors[model_name]
        epochs = range(1, len(result['train_losses']) + 1)
        
        axes[0].plot(epochs, result['train_losses'], color=color, 
                    label=f'{model_name} Train', alpha=0.8, linewidth=2)
        axes[1].plot(epochs, result['test_losses'], color=color, 
                    label=f'{model_name} Test', alpha=0.8, linewidth=2)
    
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
    
    plt.suptitle('Navier-Stokes: FNO vs AdaptiveFNO (FnoMoE) Training Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'training_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测结果对比
    models = {name: result['model'] for name, result in training_results.items()}
    test_batch = next(iter(list(test_loaders.values())[0]))
    test_inputs = test_batch['x'][:4].to(device)
    test_outputs = test_batch['y'][:4].to(device)
    
    predictions = {}
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(test_inputs)
            predictions[model_name] = pred.cpu()
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # 输入
        im1 = axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='RdBu_r')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        im2 = axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='RdBu_r')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # FNO预测
        fno_pred = predictions['FNO'][i, 0]
        im3 = axes[i, 2].imshow(fno_pred.numpy(), cmap='RdBu_r')
        axes[i, 2].set_title(f'FNO {i+1}')
        axes[i, 2].axis('off')
        
        # AdaptiveFNO预测
        adaptive_pred = predictions['AdaptiveFNO'][i, 0]
        im4 = axes[i, 3].imshow(adaptive_pred.numpy(), cmap='RdBu_r')
        axes[i, 3].set_title(f'AdaptiveFNO {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        fno_error = torch.abs(fno_pred - test_outputs[i, 0].cpu())
        adaptive_error = torch.abs(adaptive_pred - test_outputs[i, 0].cpu())
        error_diff = fno_error - adaptive_error
        
        vmax = max(error_diff.abs().max().item(), 0.01)
        im5 = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: AdaptiveFNO Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Navier-Stokes: Prediction Comparison (128×128)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'prediction_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def create_results_table(training_results, eval_results):
    """创建结果对比表格"""
    
    print(f"\n{'='*80}")
    print(f"NAVIER-STOKES EQUATION BENCHMARKS (128×128 Resolution)")
    print(f"{'='*80}")
    
    # 表头
    print(f"{'Model':<15} {'Parameters':<12} {'L2 Error':<12} {'H1 Error':<12} {'Time(ms)':<10}")
    print(f"{'-'*70}")
    
    # 数据行
    for model_name, eval_result in eval_results.items():
        params = eval_result['parameters']
        l2_error = eval_result['l2_error']
        h1_error = eval_result['h1_error']
        inference_time = eval_result['inference_time_ms']
        
        print(f"{model_name:<15} {params:<12,} {l2_error:<12.6f} {h1_error:<12.6f} {inference_time:<10.2f}")
    
    # 计算改进
    fno_l2 = eval_results['FNO']['l2_error']
    adaptive_l2 = eval_results['AdaptiveFNO']['l2_error']
    l2_improvement = ((fno_l2 - adaptive_l2) / fno_l2) * 100
    
    fno_h1 = eval_results['FNO']['h1_error']
    adaptive_h1 = eval_results['AdaptiveFNO']['h1_error']
    h1_improvement = ((fno_h1 - adaptive_h1) / fno_h1) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"L2 Error Improvement: {l2_improvement:+.2f}%")
    print(f"H1 Error Improvement: {h1_improvement:+.2f}%")
    
    # 参数对比
    fno_params = eval_results['FNO']['parameters']
    adaptive_params = eval_results['AdaptiveFNO']['parameters']
    param_increase = ((adaptive_params - fno_params) / fno_params) * 100
    
    print(f"Parameter Increase: {param_increase:+.1f}%")
    
    # 自适应频率是否有用的判断
    if l2_improvement > 0 and h1_improvement > 0:
        print(f"\n✅ 结论: 自适应频率选择有效!")
        print(f"   AdaptiveFNO在L2和H1误差上都优于标准FNO")
    elif l2_improvement > 0 or h1_improvement > 0:
        print(f"\n🔶 结论: 自适应频率选择部分有效")
        print(f"   AdaptiveFNO在某些指标上优于标准FNO")
    else:
        print(f"\n❌ 结论: 自适应频率选择在此设置下无显著改进")
        print(f"   可能需要调整超参数或模型架构")
    
    return {
        'l2_improvement': l2_improvement,
        'h1_improvement': h1_improvement,
        'param_increase': param_increase
    }


def save_results_summary(training_results, eval_results, summary_stats, results_dir):
    """保存实验结果总结"""
    
    summary_lines = []
    summary_lines.append("NAVIER-STOKES FNO vs AdaptiveFNO (FnoMoE) COMPARISON")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    # 实验配置
    summary_lines.append("Experiment Configuration:")
    summary_lines.append("  Dataset: Navier-Stokes (128×128)")
    summary_lines.append("  Models: FNO vs AdaptiveFNO (4 experts)")
    summary_lines.append("  Training: 50 epochs, AdamW optimizer")
    summary_lines.append("  Loss: L2 (LpLoss)")
    summary_lines.append("")
    
    # 结果
    summary_lines.append("Results:")
    for model_name, eval_result in eval_results.items():
        summary_lines.append(f"  {model_name}:")
        summary_lines.append(f"    Parameters: {eval_result['parameters']:,}")
        summary_lines.append(f"    L2 Error: {eval_result['l2_error']:.6f}")
        summary_lines.append(f"    H1 Error: {eval_result['h1_error']:.6f}")
        summary_lines.append(f"    Inference Time: {eval_result['inference_time_ms']:.2f}ms")
        summary_lines.append("")
    
    # 性能改进
    summary_lines.append("Performance Comparison:")
    summary_lines.append(f"  L2 Error Improvement: {summary_stats['l2_improvement']:+.2f}%")
    summary_lines.append(f"  H1 Error Improvement: {summary_stats['h1_improvement']:+.2f}%")
    summary_lines.append(f"  Parameter Increase: {summary_stats['param_increase']:+.1f}%")
    summary_lines.append("")
    
    # 结论
    l2_imp = summary_stats['l2_improvement']
    h1_imp = summary_stats['h1_improvement']
    
    summary_lines.append("Conclusion:")
    if l2_imp > 0 and h1_imp > 0:
        summary_lines.append("  ✅ 自适应频率选择有效! AdaptiveFNO在两个误差指标上都优于FNO")
    elif l2_imp > 0 or h1_imp > 0:
        summary_lines.append("  🔶 自适应频率选择部分有效，在某些指标上有改进")
    else:
        summary_lines.append("  ❌ 自适应频率选择在当前设置下无显著改进")
    
    # 保存到文件
    summary_path = results_dir / 'experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Results summary saved to {summary_path}")


def main():
    """主实验函数"""
    print("🌊 Navier-Stokes: FNO vs AdaptiveFNO (FnoMoE) Comparison")
    print("="*60)
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # 加载数据
    print(f"\n📊 Loading Navier-Stokes data...")
    data_root = Path("~/data/navier_stokes/").expanduser()
    
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_root,
        train_resolution=128,
        n_train=8000,  # 使用8000个训练样本
        batch_size=8,
        test_resolutions=[128],
        n_tests=[2000],  # 2000个测试样本
        test_batch_sizes=[8],
        encode_input=True,
        encode_output=True,
    )
    
    print(f"✓ Data loaded successfully")
    
    # 创建模型
    print(f"\n🔧 Creating models...")
    models, param_counts = create_models(device)
    
    # 训练模型
    print(f"\n🏋️ Training models...")
    training_results = {}
    for model_name, model in models.items():
        result = train_model(
            model, model_name, train_loader, test_loaders, device, n_epochs=50
        )
        training_results[model_name] = result
        
        # 保存模型
        model_path = results_dir / 'models' / f'{model_name.lower()}_final.pth'
        torch.save(model.state_dict(), model_path)
    
    # 评估模型
    eval_results = evaluate_models(models, param_counts, test_loaders, device)
    
    # 可视化结果
    visualize_results(training_results, eval_results, test_loaders, device, results_dir)
    
    # 创建结果表格
    summary_stats = create_results_table(training_results, eval_results)
    
    # 保存结果总结
    save_results_summary(training_results, eval_results, summary_stats, results_dir)
    
    print(f"\n🎉 Experiment completed!")
    print(f"📁 Results saved in: {results_dir}")


if __name__ == "__main__":
    main()
