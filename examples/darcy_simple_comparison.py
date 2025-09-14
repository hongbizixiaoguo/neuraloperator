"""
Darcy Flow简化对比实验
使用原始Darcy数据集对比FNO和FnoMoE
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

try:
    from neuralop import H1Loss, LpLoss
    from neuralop.data.datasets import load_darcy_flow_small
    from neuralop.models.fno import FNO2d
    from neuralop.models.final_adaptive_fno import AdaptiveFNO2d
    from neuralop.utils import count_model_params
    print("✓ Successfully imported neuralop modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_experiment_folder():
    """创建实验文件夹"""
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/darcy_flow_simple'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    print(f"✓ Created experiment folder: {results_dir}")
    return results_dir


def load_darcy_dataset():
    """加载Darcy Flow数据集"""
    
    print(f"📊 Loading Darcy Flow dataset...")
    
    try:
        # 尝试加载Darcy数据集
        data_root = Path("~/data/darcy/").expanduser()
        
        train_loader, test_loaders, data_processor = load_darcy_flow_small(
            data_root=data_root,
            n_train=1000,
            batch_size=8,
            test_resolutions=[16, 32],
            n_tests=[100, 50],
            test_batch_sizes=[16, 16],
            encode_input=False,
            encode_output=False,
        )
        
        print(f"✓ Darcy dataset loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test resolutions: {list(test_loaders.keys())}")
        
        return train_loader, test_loaders, data_processor
        
    except Exception as e:
        print(f"❌ Failed to load Darcy dataset: {e}")
        print("💡 Generating synthetic Darcy-like data instead...")
        return generate_synthetic_darcy_data()


def generate_synthetic_darcy_data():
    """生成合成的Darcy-like数据"""
    
    print("🔧 Generating synthetic Darcy-like data...")
    
    n_train = 800
    n_test = 200
    resolution = 16
    
    # 生成合成的渗透率场和对应的压力场
    def generate_sample(res):
        # 生成随机渗透率场 (对数正态分布)
        x = np.linspace(0, 1, res)
        y = np.linspace(0, 1, res)
        X, Y = np.meshgrid(x, y)
        
        # 使用随机傅里叶模式生成渗透率场
        permeability = np.zeros((res, res))
        for k in range(1, 6):
            for l in range(1, 6):
                amp = np.random.normal(0, 1.0 / (k + l))
                phase = np.random.uniform(0, 2*np.pi)
                permeability += amp * np.sin(2*np.pi*k*X + phase) * np.cos(2*np.pi*l*Y)
        
        # 归一化到合理范围
        permeability = np.exp(permeability)
        permeability = (permeability - permeability.min()) / (permeability.max() - permeability.min())
        
        # 简化的压力场计算 (基于拉普拉斯方程的近似解)
        # 这里使用简化的物理关系
        pressure = np.zeros_like(permeability)
        
        # 简单的扩散近似
        for i in range(1, res-1):
            for j in range(1, res-1):
                # 基于周围渗透率的加权平均
                neighbors = permeability[i-1:i+2, j-1:j+2]
                pressure[i, j] = np.mean(neighbors) + 0.1 * np.random.normal()
        
        # 边界条件
        pressure[0, :] = 1.0  # 上边界高压
        pressure[-1, :] = 0.0  # 下边界低压
        pressure[:, 0] = pressure[:, 1]  # 左右边界
        pressure[:, -1] = pressure[:, -2]
        
        return permeability[np.newaxis, :, :], pressure[np.newaxis, :, :]
    
    # 生成训练数据
    train_inputs, train_outputs = [], []
    for _ in range(n_train):
        perm, pres = generate_sample(resolution)
        train_inputs.append(perm)
        train_outputs.append(pres)
    
    # 生成测试数据
    test_inputs, test_outputs = [], []
    for _ in range(n_test):
        perm, pres = generate_sample(resolution)
        test_inputs.append(perm)
        test_outputs.append(pres)
    
    # 转换为PyTorch张量
    train_inputs = torch.FloatTensor(np.array(train_inputs))
    train_outputs = torch.FloatTensor(np.array(train_outputs))
    test_inputs = torch.FloatTensor(np.array(test_inputs))
    test_outputs = torch.FloatTensor(np.array(test_outputs))
    
    # 创建数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 模拟neuralop的数据格式
    class SimpleDataProcessor:
        def preprocess(self, batch, batched=True):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                return {'x': batch[0], 'y': batch[1]}
            return batch
    
    test_loaders = {16: test_loader}
    data_processor = SimpleDataProcessor()
    
    print(f"✓ Generated synthetic data: {n_train} train, {n_test} test samples")
    
    return train_loader, test_loaders, data_processor


def create_models(device):
    """创建FNO和FnoMoE模型"""
    
    print(f"🏗️  Creating models...")
    
    # FNO模型
    fno_model = FNO2d(
        n_modes_height=12,
        n_modes_width=12,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4
    ).to(device)
    
    # FnoMoE模型
    fnomoe_model = AdaptiveFNO2d(
        n_modes_height=12,
        n_modes_width=12,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4
    ).to(device)
    
    models = {
        'FNO': fno_model,
        'FnoMoE': fnomoe_model
    }
    
    # 计算参数数量
    param_counts = {}
    for name, model in models.items():
        param_counts[name] = sum(p.numel() for p in model.parameters())
        print(f"  {name} parameters: {param_counts[name]:,}")
    
    return models, param_counts


def train_models(models, param_counts, train_loader, test_loaders, data_processor, device, results_dir, epochs=50):
    """训练模型"""
    
    results = {}
    
    # 损失函数
    try:
        h1_loss = H1Loss(d=2)
        l2_loss = LpLoss(d=2, p=2)
    except:
        # 如果H1Loss不可用，使用MSE作为替代
        h1_loss = torch.nn.MSELoss()
        l2_loss = torch.nn.MSELoss()
        print("⚠️  Using MSE loss as fallback")
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-3,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=20, 
            gamma=0.5
        )
        
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
            
            for batch in train_loader:
                # 处理数据格式
                if data_processor is not None:
                    batch = data_processor.preprocess(batch, batched=True)
                
                if isinstance(batch, dict):
                    x, y = batch['x'].to(device), batch['y'].to(device)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                
                # 计算损失
                if hasattr(h1_loss, '__call__'):
                    loss = h1_loss(pred, y)
                else:
                    loss = F.mse_loss(pred, y)
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # 测试阶段
            model.eval()
            epoch_test_loss = 0
            n_test_batches = 0
            
            with torch.no_grad():
                for test_loader in test_loaders.values():
                    for batch in test_loader:
                        if data_processor is not None:
                            batch = data_processor.preprocess(batch, batched=True)
                        
                        if isinstance(batch, dict):
                            x, y = batch['x'].to(device), batch['y'].to(device)
                        else:
                            x, y = batch[0].to(device), batch[1].to(device)
                        
                        pred = model(x)
                        
                        if hasattr(h1_loss, '__call__'):
                            loss = h1_loss(pred, y)
                        else:
                            loss = F.mse_loss(pred, y)
                        
                        epoch_test_loss += loss.item()
                        n_test_batches += 1
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_test_loss = epoch_test_loss / n_test_batches if n_test_batches > 0 else float('inf')
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            # 学习率调度
            scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # 保存最佳模型
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), f'{results_dir}/models/{name.lower()}_best.pth')
            
            # 打印进度
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
        
        # 保存最终模型
        torch.save(model.state_dict(), f'{results_dir}/models/{name.lower()}_final.pth')
        
        print(f"  ✓ Training completed: {total_training_time:.1f}s total, {avg_epoch_time:.2f}s/epoch")
    
    return results


def create_benchmark_table(results):
    """创建基准测试表格"""
    
    print(f"\n{'='*70}")
    print(f"DARCY FLOW BENCHMARK RESULTS")
    print(f"{'='*70}")
    
    print(f"{'Config':<10} {'Parameters':<12} {'Time/Epoch':<12} {'Best Loss':<12} {'Final Loss':<12}")
    print(f"{'-'*70}")
    
    for name, result in results.items():
        params = result['parameters']
        time_per_epoch = result['avg_epoch_time']
        best_loss = result['best_test_loss']
        final_loss = result['final_test_loss']
        
        print(f"{name:<10} {params:<12,} {time_per_epoch:<12.2f} {best_loss:<12.6f} {final_loss:<12.6f}")
    
    # 性能对比
    fno_best = results['FNO']['best_test_loss']
    fnomoe_best = results['FnoMoE']['best_test_loss']
    improvement = ((fno_best - fnomoe_best) / fno_best) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"FNO Best Loss:      {fno_best:.6f}")
    print(f"FnoMoE Best Loss:   {fnomoe_best:.6f}")
    print(f"Improvement:        {improvement:+.2f}%")
    
    # 参数对比
    fno_params = results['FNO']['parameters']
    fnomoe_params = results['FnoMoE']['parameters']
    param_overhead = ((fnomoe_params - fno_params) / fno_params) * 100
    
    print(f"Parameter Overhead: {param_overhead:+.1f}%")
    
    return {
        'improvement': improvement,
        'param_overhead': param_overhead,
        'fno_best': fno_best,
        'fnomoe_best': fnomoe_best
    }


def visualize_results(results, test_loaders, data_processor, device, results_dir):
    """可视化结果"""
    
    # 1. 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'FNO': 'blue', 'FnoMoE': 'red'}
    
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
    
    plt.suptitle('Darcy Flow: FNO vs FnoMoE Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/darcy_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 预测结果可视化
    test_loader = list(test_loaders.values())[0]
    sample_batch = next(iter(test_loader))
    
    if data_processor is not None:
        sample_batch = data_processor.preprocess(sample_batch, batched=True)
    
    if isinstance(sample_batch, dict):
        x_sample = sample_batch['x'][:3].to(device)
        y_sample = sample_batch['y'][:3].to(device)
    else:
        x_sample = sample_batch[0][:3].to(device)
        y_sample = sample_batch[1][:3].to(device)
    
    predictions = {}
    for name, result in results.items():
        result['model'].eval()
        with torch.no_grad():
            pred = result['model'](x_sample)
            predictions[name] = pred.cpu()
    
    # 可视化
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    for i in range(3):
        # 输入 (渗透率)
        axes[i, 0].imshow(x_sample[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f'Permeability {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出 (压力)
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


def save_results(results, benchmark_summary, results_dir):
    """保存实验结果"""
    
    summary_lines = []
    summary_lines.append("DARCY FLOW SIMPLE COMPARISON RESULTS")
    summary_lines.append("="*50)
    summary_lines.append("")
    
    summary_lines.append("Benchmark Results:")
    summary_lines.append(f"{'Config':<10} {'Parameters':<12} {'Best Loss':<12}")
    summary_lines.append("-" * 40)
    
    for name, result in results.items():
        params = result['parameters']
        best_loss = result['best_test_loss']
        summary_lines.append(f"{name:<10} {params:<12,} {best_loss:<12.6f}")
    
    summary_lines.append("")
    summary_lines.append("Performance Summary:")
    summary_lines.append(f"  Improvement: {benchmark_summary['improvement']:+.2f}%")
    summary_lines.append(f"  Parameter Overhead: {benchmark_summary['param_overhead']:+.1f}%")
    
    # 保存到文件
    summary_path = f'{results_dir}/darcy_simple_results.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Results saved to {summary_path}")


def main():
    """主函数"""
    print("🏔️  Darcy Flow Simple Comparison: FNO vs FnoMoE")
    print("="*60)
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 加载数据
    train_loader, test_loaders, data_processor = load_darcy_dataset()
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    models, param_counts = create_models(device)
    
    # 训练模型
    results = train_models(
        models, param_counts, train_loader, test_loaders, 
        data_processor, device, results_dir, epochs=50
    )
    
    # 创建基准测试表格
    benchmark_summary = create_benchmark_table(results)
    
    # 可视化结果
    print(f"\n📈 Generating visualizations...")
    visualize_results(results, test_loaders, data_processor, device, results_dir)
    
    # 保存结果
    save_results(results, benchmark_summary, results_dir)
    
    print(f"\n🎉 Darcy Flow comparison completed!")
    print(f"📁 Results saved in: {results_dir}")
    
    return results_dir


if __name__ == "__main__":
    main()


