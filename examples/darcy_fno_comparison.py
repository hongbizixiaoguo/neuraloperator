"""
Darcy Flow方程: FNO vs FnoMoE 对比实验
使用原始的Darcy Flow数据集进行基准测试
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop import H1Loss, LpLoss, get_model
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models.fno import FNO2d
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d
from neuralop.utils import count_model_params


def create_experiment_folder():
    """创建Darcy实验文件夹"""
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/darcy_flow_comparison'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    print(f"✓ Created Darcy Flow experiment folder: {results_dir}")
    return results_dir


def load_darcy_data(data_root="~/data/darcy/", n_train=1000, batch_size=8):
    """加载Darcy Flow数据集"""
    
    print(f"📊 Loading Darcy Flow dataset...")
    print(f"  Data root: {data_root}")
    print(f"  Training samples: {n_train}")
    print(f"  Batch size: {batch_size}")
    
    try:
        # 使用neuralop的标准数据加载器
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            data_root=Path(data_root).expanduser(),
            n_train=n_train,
            batch_size=batch_size,
            test_resolutions=[16, 32],  # 测试不同分辨率
            n_tests=[100, 50],
            test_batch_sizes=[16, 16],
            encode_input=False,
            encode_output=False,
        )
        
        print(f"✓ Successfully loaded Darcy Flow dataset")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test resolutions: {list(test_loaders.keys())}")
        
        return train_loader, test_loaders, data_processor
        
    except Exception as e:
        print(f"❌ Failed to load Darcy dataset: {e}")
        print("💡 Make sure the Darcy dataset is downloaded")
        return None, None, None


def create_darcy_models(device):
    """创建Darcy Flow的FNO和FnoMoE模型"""
    
    print(f"🏗️  Creating models for Darcy Flow...")
    
    # 基于darcy_config.py的配置
    # FNO-2D (原始FNO)
    fno_model = FNO2d(
        n_modes_height=12,
        n_modes_width=12,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        projection_channels=64
    ).to(device)
    
    # FnoMoE (自适应FNO)
    fnomoe_model = AdaptiveFNO2d(
        n_modes_height=12,
        n_modes_width=12,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4,
        projection_channels=64
    ).to(device)
    
    models = {
        'FNO': fno_model,
        'FnoMoE': fnomoe_model
    }
    
    # 计算参数数量
    param_counts = {}
    for name, model in models.items():
        param_counts[name] = count_model_params(model)
        print(f"  {name} parameters: {param_counts[name]:,}")
    
    return models, param_counts


def train_darcy_models(models, param_counts, train_loader, test_loaders, data_processor, device, results_dir, epochs=100):
    """训练Darcy Flow模型"""
    
    results = {}
    
    # 损失函数 (基于darcy_config.py)
    h1_loss = H1Loss(d=2)
    l2_loss = LpLoss(d=2, p=2)
    
    for name, model in models.items():
        print(f"\n🚀 Training {name} on Darcy Flow...")
        
        # 优化器设置 (基于darcy_config.py)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-3,  # darcy_config中的学习率
            weight_decay=1e-4
        )
        
        # 学习率调度器 (基于darcy_config.py)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=60, 
            gamma=0.5
        )
        
        train_losses = []
        test_losses = {}
        for res in test_loaders.keys():
            test_losses[res] = []
        
        epoch_times = []
        best_test_loss = float('inf')
        
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            
            for batch in train_loader:
                if data_processor is not None:
                    batch = data_processor.preprocess(batch, batched=True)
                
                x, y = batch['x'].to(device), batch['y'].to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                
                # 使用H1损失 (darcy_config中的默认设置)
                loss = h1_loss(pred, y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # 测试阶段
            model.eval()
            epoch_test_losses = {}
            
            with torch.no_grad():
                for res, test_loader in test_loaders.items():
                    res_test_loss = 0
                    n_test_batches = 0
                    
                    for batch in test_loader:
                        if data_processor is not None:
                            batch = data_processor.preprocess(batch, batched=True)
                        
                        x, y = batch['x'].to(device), batch['y'].to(device)
                        pred = model(x)
                        
                        # 使用H1损失评估
                        loss = h1_loss(pred, y)
                        res_test_loss += loss.item()
                        n_test_batches += 1
                    
                    avg_res_test_loss = res_test_loss / n_test_batches if n_test_batches > 0 else float('inf')
                    epoch_test_losses[res] = avg_res_test_loss
                    test_losses[res].append(avg_res_test_loss)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 学习率调度
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # 保存最佳模型 (基于分辨率16的测试损失)
            main_test_loss = epoch_test_losses.get(16, float('inf'))
            if main_test_loss < best_test_loss:
                best_test_loss = main_test_loss
                torch.save(model.state_dict(), f'{results_dir}/models/{name.lower()}_best.pth')
            
            # 打印进度
            if (epoch + 1) % 20 == 0:
                test_loss_str = ", ".join([f"Test-{res}={loss:.6f}" for res, loss in epoch_test_losses.items()])
                print(f"  Epoch {epoch+1:3d}: Train={avg_train_loss:.6f}, {test_loss_str}, Time={epoch_time:.2f}s")
        
        total_training_time = time.time() - total_start_time
        avg_epoch_time = np.mean(epoch_times)
        
        results[name] = {
            'model': model,
            'parameters': param_counts[name],
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss,
            'final_test_loss': test_losses[16][-1] if test_losses[16] else float('inf'),
            'total_training_time': total_training_time,
            'avg_epoch_time': avg_epoch_time
        }
        
        # 保存最终模型
        torch.save(model.state_dict(), f'{results_dir}/models/{name.lower()}_final.pth')
        
        print(f"  ✓ Training completed: {total_training_time:.1f}s total, {avg_epoch_time:.2f}s/epoch")
    
    return results


def evaluate_darcy_models(results, test_loaders, data_processor, device):
    """评估Darcy模型性能"""
    
    print(f"\n📊 Evaluating model performance...")
    
    eval_results = {}
    h1_loss = H1Loss(d=2)
    l2_loss = LpLoss(d=2, p=2)
    
    for name, result in results.items():
        model = result['model']
        model.eval()
        
        eval_results[name] = {}
        
        with torch.no_grad():
            for res, test_loader in test_loaders.items():
                total_h1_loss = 0
                total_l2_loss = 0
                total_samples = 0
                inference_times = []
                
                for batch in test_loader:
                    if data_processor is not None:
                        batch = data_processor.preprocess(batch, batched=True)
                    
                    x, y = batch['x'].to(device), batch['y'].to(device)
                    
                    # 测量推理时间
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    pred = model(x)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    inference_time = time.time() - start_time
                    
                    # 计算损失
                    h1_val = h1_loss(pred, y).item()
                    l2_val = l2_loss(pred, y).item()
                    
                    batch_size = x.size(0)
                    total_h1_loss += h1_val * batch_size
                    total_l2_loss += l2_val * batch_size
                    total_samples += batch_size
                    
                    inference_times.append(inference_time / batch_size)
                
                eval_results[name][res] = {
                    'h1_loss': total_h1_loss / total_samples,
                    'l2_loss': total_l2_loss / total_samples,
                    'inference_time_per_sample': np.mean(inference_times)
                }
    
    return eval_results


def create_darcy_benchmark_table(results, eval_results):
    """创建Darcy Flow基准测试表格"""
    
    print(f"\n{'='*80}")
    print(f"DARCY FLOW BENCHMARKS")
    print(f"{'='*80}")
    
    # 主要结果表格 (分辨率16)
    print(f"\nMain Results (Resolution 16×16):")
    print(f"{'Config':<10} {'Parameters':<12} {'Time/Epoch':<12} {'H1 Loss':<12} {'L2 Loss':<12}")
    print(f"{'-'*70}")
    
    for name, result in results.items():
        params = result['parameters']
        time_per_epoch = result['avg_epoch_time']
        h1_loss = eval_results[name][16]['h1_loss']
        l2_loss = eval_results[name][16]['l2_loss']
        
        print(f"{name:<10} {params:<12,} {time_per_epoch:<12.2f} {h1_loss:<12.6f} {l2_loss:<12.6f}")
    
    # 多分辨率结果
    print(f"\nMulti-Resolution Results:")
    for res in sorted(eval_results['FNO'].keys()):
        print(f"\nResolution {res}×{res}:")
        print(f"{'Config':<10} {'H1 Loss':<12} {'L2 Loss':<12} {'Inference (ms)':<15}")
        print(f"{'-'*50}")
        
        for name in results.keys():
            h1_loss = eval_results[name][res]['h1_loss']
            l2_loss = eval_results[name][res]['l2_loss']
            inf_time = eval_results[name][res]['inference_time_per_sample'] * 1000
            
            print(f"{name:<10} {h1_loss:<12.6f} {l2_loss:<12.6f} {inf_time:<15.2f}")
    
    # 性能对比
    fno_h1 = eval_results['FNO'][16]['h1_loss']
    fnomoe_h1 = eval_results['FnoMoE'][16]['h1_loss']
    h1_improvement = ((fno_h1 - fnomoe_h1) / fno_h1) * 100
    
    fno_l2 = eval_results['FNO'][16]['l2_loss']
    fnomoe_l2 = eval_results['FnoMoE'][16]['l2_loss']
    l2_improvement = ((fno_l2 - fnomoe_l2) / fno_l2) * 100
    
    print(f"\n📊 Performance Summary (Resolution 16×16):")
    print(f"H1 Loss Improvement: {h1_improvement:+.2f}%")
    print(f"L2 Loss Improvement: {l2_improvement:+.2f}%")
    
    # 参数对比
    fno_params = results['FNO']['parameters']
    fnomoe_params = results['FnoMoE']['parameters']
    param_overhead = ((fnomoe_params - fno_params) / fno_params) * 100
    
    print(f"Parameter Overhead: {param_overhead:+.1f}%")
    
    return {
        'h1_improvement': h1_improvement,
        'l2_improvement': l2_improvement,
        'param_overhead': param_overhead,
        'fno_h1': fno_h1,
        'fnomoe_h1': fnomoe_h1,
        'fno_l2': fno_l2,
        'fnomoe_l2': fnomoe_l2
    }


def visualize_darcy_results(results, eval_results, test_loaders, data_processor, device, results_dir):
    """可视化Darcy Flow结果"""
    
    # 1. 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'FNO': 'blue', 'FnoMoE': 'red'}
    
    # 训练损失
    for name, result in results.items():
        color = colors[name]
        epochs = range(1, len(result['train_losses']) + 1)
        axes[0].plot(epochs, result['train_losses'], color=color, label=f'{name} Train', alpha=0.7)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss (H1)')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 测试损失 (分辨率16)
    for name, result in results.items():
        color = colors[name]
        epochs = range(1, len(result['test_losses'][16]) + 1)
        axes[1].plot(epochs, result['test_losses'][16], color=color, label=f'{name} Test-16', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss (H1)')
    axes[1].set_title('Test Loss Comparison (16×16)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # 多分辨率性能对比
    resolutions = sorted(eval_results['FNO'].keys())
    fno_h1_losses = [eval_results['FNO'][res]['h1_loss'] for res in resolutions]
    fnomoe_h1_losses = [eval_results['FnoMoE'][res]['h1_loss'] for res in resolutions]
    
    x = np.arange(len(resolutions))
    width = 0.35
    
    axes[2].bar(x - width/2, fno_h1_losses, width, label='FNO', color='blue', alpha=0.7)
    axes[2].bar(x + width/2, fnomoe_h1_losses, width, label='FnoMoE', color='red', alpha=0.7)
    
    axes[2].set_xlabel('Resolution')
    axes[2].set_ylabel('H1 Loss')
    axes[2].set_title('Multi-Resolution Performance')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'{res}×{res}' for res in resolutions])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.suptitle('Darcy Flow: FNO vs FnoMoE Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/darcy_training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测结果可视化
    # 获取一些测试样本进行可视化
    test_loader_16 = test_loaders[16]
    sample_batch = next(iter(test_loader_16))
    
    if data_processor is not None:
        sample_batch = data_processor.preprocess(sample_batch, batched=True)
    
    x_sample = sample_batch['x'][:4].to(device)  # 取前4个样本
    y_sample = sample_batch['y'][:4].to(device)
    
    predictions = {}
    for name, result in results.items():
        result['model'].eval()
        with torch.no_grad():
            pred = result['model'](x_sample)
            predictions[name] = pred.cpu()
    
    # 可视化预测结果
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # 输入 (渗透率场)
        axes[i, 0].imshow(x_sample[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Permeability {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出 (压力场)
        axes[i, 1].imshow(y_sample[i, 0].cpu().numpy(), cmap='RdBu')
        axes[i, 1].set_title(f'True Pressure {i+1}')
        axes[i, 1].axis('off')
        
        # FNO预测
        fno_pred = predictions['FNO'][i, 0]
        axes[i, 2].imshow(fno_pred.numpy(), cmap='RdBu')
        axes[i, 2].set_title(f'FNO Pred {i+1}')
        axes[i, 2].axis('off')
        
        # FnoMoE预测
        fnomoe_pred = predictions['FnoMoE'][i, 0]
        axes[i, 3].imshow(fnomoe_pred.numpy(), cmap='RdBu')
        axes[i, 3].set_title(f'FnoMoE Pred {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        fno_error = torch.abs(fno_pred - y_sample[i, 0].cpu())
        fnomoe_error = torch.abs(fnomoe_pred - y_sample[i, 0].cpu())
        error_diff = fno_error - fnomoe_error
        
        im = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: FnoMoE Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Darcy Flow: Pressure Field Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/darcy_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_darcy_results(results, eval_results, benchmark_summary, results_dir):
    """保存Darcy Flow实验结果"""
    
    summary_lines = []
    summary_lines.append("DARCY FLOW EXPERIMENT RESULTS")
    summary_lines.append("="*50)
    summary_lines.append("")
    
    # 配置信息
    summary_lines.append("Configuration:")
    summary_lines.append("  Dataset: Darcy Flow (Small)")
    summary_lines.append("  Training Samples: 1000")
    summary_lines.append("  Test Resolutions: [16, 32]")
    summary_lines.append("  Loss Function: H1 Loss")
    summary_lines.append("  Optimizer: AdamW (lr=5e-3)")
    summary_lines.append("  Scheduler: StepLR (step=60, gamma=0.5)")
    summary_lines.append("")
    
    # 主要结果
    summary_lines.append("Main Results (Resolution 16×16):")
    summary_lines.append(f"{'Config':<10} {'Parameters':<12} {'H1 Loss':<12} {'L2 Loss':<12}")
    summary_lines.append("-" * 50)
    
    for name, result in results.items():
        params = result['parameters']
        h1_loss = eval_results[name][16]['h1_loss']
        l2_loss = eval_results[name][16]['l2_loss']
        summary_lines.append(f"{name:<10} {params:<12,} {h1_loss:<12.6f} {l2_loss:<12.6f}")
    
    summary_lines.append("")
    summary_lines.append("Performance Summary:")
    summary_lines.append(f"  H1 Loss Improvement: {benchmark_summary['h1_improvement']:+.2f}%")
    summary_lines.append(f"  L2 Loss Improvement: {benchmark_summary['l2_improvement']:+.2f}%")
    summary_lines.append(f"  Parameter Overhead: {benchmark_summary['param_overhead']:+.1f}%")
    
    # 多分辨率结果
    summary_lines.append("")
    summary_lines.append("Multi-Resolution Results:")
    for res in sorted(eval_results['FNO'].keys()):
        summary_lines.append(f"\nResolution {res}×{res}:")
        for name in results.keys():
            h1_loss = eval_results[name][res]['h1_loss']
            l2_loss = eval_results[name][res]['l2_loss']
            summary_lines.append(f"  {name}: H1={h1_loss:.6f}, L2={l2_loss:.6f}")
    
    # 保存到文件
    summary_path = f'{results_dir}/darcy_experiment_results.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Darcy experiment results saved to {summary_path}")


def main():
    """主函数"""
    print("🏔️  Darcy Flow: FNO vs FnoMoE Comparison")
    print("="*50)
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 加载Darcy Flow数据
    train_loader, test_loaders, data_processor = load_darcy_data(
        data_root="~/data/darcy/",
        n_train=1000,
        batch_size=8
    )
    
    if train_loader is None:
        print("❌ Failed to load Darcy dataset. Exiting...")
        return None
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    models, param_counts = create_darcy_models(device)
    
    # 训练模型
    results = train_darcy_models(
        models, param_counts, train_loader, test_loaders, 
        data_processor, device, results_dir, epochs=100
    )
    
    # 评估模型
    eval_results = evaluate_darcy_models(results, test_loaders, data_processor, device)
    
    # 创建基准测试表格
    benchmark_summary = create_darcy_benchmark_table(results, eval_results)
    
    # 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_darcy_results(results, eval_results, test_loaders, data_processor, device, results_dir)
    
    # 保存结果
    save_darcy_results(results, eval_results, benchmark_summary, results_dir)
    
    print(f"\n🎉 Darcy Flow experiment completed!")
    print(f"📁 Results saved in: {results_dir}")
    
    return results_dir


if __name__ == "__main__":
    main()


