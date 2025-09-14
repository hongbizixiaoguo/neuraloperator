"""
在Burgers方程上对比原始FNO和自适应FNO的性能
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from collections import defaultdict

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

# 导入neuraloperator组件
from neuralop import H1Loss, LpLoss
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.utils import get_project_root
from neuralop.models.fno import FNO1d
from neuralop.models.final_adaptive_fno import AdaptiveFNO

# 自定义1D自适应FNO
class AdaptiveFNO1d(AdaptiveFNO):
    """
    1D自适应FNO，专门用于Burgers方程
    """
    def __init__(self,
                 n_modes_height: int,
                 hidden_channels: int,
                 in_channels: int = 2,  # Burgers方程通常有2个输入通道
                 out_channels: int = 1,
                 n_layers: int = 4,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__(
            n_modes=(n_modes_height,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_experts=n_experts,
            temperature=temperature,
            **kwargs
        )
        self.n_modes_height = n_modes_height


def load_burgers_data(data_path, n_train=800, n_test=200, batch_size=16):
    """
    加载Burgers数据集
    """
    print(f"Loading Burgers dataset from {data_path}")
    
    try:
        train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
            data_path=data_path,
            n_train=n_train, 
            batch_size=batch_size,
            n_test=n_test, 
            test_batch_size=batch_size,
            temporal_subsample=1,
            spatial_subsample=1,
        )
        
        print(f"✓ Successfully loaded Burgers dataset")
        print(f"  Train batches: {len(train_loader)}")
        
        # 检查test_loaders是否为空
        if test_loaders and len(test_loaders) > 0:
            test_loader = test_loaders[0]
            print(f"  Test batches: {len(test_loader)}")
        else:
            print("  Warning: No test loaders found")
            test_loader = None
        
        # 检查数据形状
        for batch in train_loader:
            x, y = batch['x'], batch['y']
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {y.shape}")
            break
            
        return train_loader, test_loader, data_processor
        
    except Exception as e:
        print(f"✗ Failed to load Burgers dataset: {e}")
        return None, None, None


def create_burgers_models(device, input_shape):
    """
    创建用于Burgers方程的原始FNO和自适应FNO模型
    """
    print("Creating models for Burgers equation...")
    
    # 从输入形状推断参数
    batch_size, in_channels, temporal_len, spatial_len = input_shape
    
    # 原始FNO1d
    original_fno = FNO1d(
        n_modes_height=16,
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        non_linearity=F.gelu,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',
        channel_mlp_skip='soft-gating'
    ).to(device)
    
    # 自适应FNO1d
    adaptive_fno = AdaptiveFNO1d(
        n_modes_height=16,
        in_channels=in_channels,
        out_channels=1,
        hidden_channels=64,
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
    
    print(f"✓ Original FNO1d parameters: {sum(p.numel() for p in original_fno.parameters()):,}")
    print(f"✓ Adaptive FNO1d parameters: {sum(p.numel() for p in adaptive_fno.parameters()):,}")
    
    return original_fno, adaptive_fno


def train_burgers_model(model, train_loader, test_loader, device, model_name, n_epochs=100):
    """
    训练Burgers方程模型
    """
    print(f"\nTraining {model_name} on Burgers equation...")
    
    # 使用适合Burgers方程的损失函数
    l2_loss = LpLoss(d=1, p=2)
    h1_loss = H1Loss(d=1)
    
    def combined_loss(pred, target):
        return l2_loss(pred, target) + 0.1 * h1_loss(pred, target)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, mode='min'
    )
    
    train_losses = []
    test_losses = []
    training_times = []
    
    model.train()
    
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0.0
        
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            
            # 确保预测和目标形状匹配
            if pred.shape != y.shape:
                # 如果时间维度不匹配，取最后一个时间步
                if len(y.shape) == 4:  # [batch, channels, time, space]
                    y = y[:, :, -1:, :]  # 取最后一个时间步
                pred = pred.reshape(y.shape)
            
            loss = combined_loss(pred, y)
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
            for batch in test_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                pred = model(x)
                
                # 确保预测和目标形状匹配
                if pred.shape != y.shape:
                    if len(y.shape) == 4:
                        y = y[:, :, -1:, :]
                    pred = pred.reshape(y.shape)
                
                loss = combined_loss(pred, y)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        model.train()
        
        if (epoch + 1) % 20 == 0:
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


def evaluate_burgers_models(original_fno, adaptive_fno, test_loader, device):
    """
    评估Burgers方程模型性能
    """
    print("\nEvaluating models on Burgers equation...")
    
    l2_loss = LpLoss(d=1, p=2)
    h1_loss = H1Loss(d=1)
    
    results = {}
    
    for model, name in [(original_fno, 'Original FNO'), (adaptive_fno, 'Adaptive FNO')]:
        model.eval()
        
        total_l2 = 0.0
        total_h1 = 0.0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                
                # 测量推理时间
                start_time = time.time()
                pred = model(x)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / x.size(0))
                
                # 确保预测和目标形状匹配
                if pred.shape != y.shape:
                    if len(y.shape) == 4:
                        y = y[:, :, -1:, :]
                    pred = pred.reshape(y.shape)
                
                # 计算误差
                l2_error = l2_loss(pred, y)
                h1_error = h1_loss(pred, y)
                
                total_l2 += l2_error.item() * x.size(0)
                total_h1 += h1_error.item() * x.size(0)
                total_samples += x.size(0)
        
        results[name] = {
            'l2_error': total_l2 / total_samples,
            'h1_error': total_h1 / total_samples,
            'avg_inference_time': np.mean(inference_times) * 1000,  # ms
            'std_inference_time': np.std(inference_times) * 1000   # ms
        }
    
    return results


def visualize_burgers_predictions(original_fno, adaptive_fno, test_loader, device):
    """
    可视化Burgers方程的预测结果
    """
    print("\nGenerating Burgers equation visualizations...")
    
    original_fno.eval()
    adaptive_fno.eval()
    
    # 获取一个测试批次
    for batch in test_loader:
        x, y = batch['x'].to(device), batch['y'].to(device)
        break
    
    with torch.no_grad():
        original_pred = original_fno(x)
        adaptive_pred = adaptive_fno(x)
    
    # 确保预测和目标形状匹配
    if original_pred.shape != y.shape:
        if len(y.shape) == 4:
            y = y[:, :, -1:, :]
        original_pred = original_pred.reshape(y.shape)
        adaptive_pred = adaptive_pred.reshape(y.shape)
    
    # 选择几个样本进行可视化
    n_samples = min(3, x.size(0))
    
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 输入（初始条件）
        if len(x.shape) == 4:  # [batch, channels, time, space]
            input_data = x[i, 0, 0, :].cpu().numpy()  # 第一个通道，第一个时间步
        else:
            input_data = x[i, 0, :].cpu().numpy()
        
        axes[i, 0].plot(input_data)
        axes[i, 0].set_title(f'Initial Condition {i+1}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 真实解
        if len(y.shape) == 4:
            target_data = y[i, 0, -1, :].cpu().numpy()  # 最后一个时间步
        else:
            target_data = y[i, 0, :].cpu().numpy()
        
        axes[i, 1].plot(target_data, 'g-', label='True', linewidth=2)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 原始FNO预测
        if len(original_pred.shape) == 4:
            orig_pred_data = original_pred[i, 0, -1, :].cpu().numpy()
        else:
            orig_pred_data = original_pred[i, 0, :].cpu().numpy()
        
        axes[i, 2].plot(target_data, 'g-', label='True', linewidth=2, alpha=0.7)
        axes[i, 2].plot(orig_pred_data, 'b--', label='Original FNO', linewidth=2)
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        # 自适应FNO预测
        if len(adaptive_pred.shape) == 4:
            adapt_pred_data = adaptive_pred[i, 0, -1, :].cpu().numpy()
        else:
            adapt_pred_data = adaptive_pred[i, 0, :].cpu().numpy()
        
        axes[i, 3].plot(target_data, 'g-', label='True', linewidth=2, alpha=0.7)
        axes[i, 3].plot(adapt_pred_data, 'r--', label='Adaptive FNO', linewidth=2)
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].legend()
        axes[i, 3].grid(True, alpha=0.3)
        
        # 误差对比
        orig_error = np.abs(orig_pred_data - target_data)
        adapt_error = np.abs(adapt_pred_data - target_data)
        
        axes[i, 4].plot(orig_error, 'b-', label='Original FNO Error', linewidth=2)
        axes[i, 4].plot(adapt_error, 'r-', label='Adaptive FNO Error', linewidth=2)
        axes[i, 4].set_title(f'Absolute Error {i+1}')
        axes[i, 4].legend()
        axes[i, 4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/neuraloperator/burgers_comparison_predictions.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_burgers_training_comparison(original_results, adaptive_results):
    """
    绘制Burgers方程训练过程对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(original_results['train_losses']) + 1)
    
    # 训练损失对比
    axes[0, 0].plot(epochs, original_results['train_losses'], 'b-', 
                   label='Original FNO', linewidth=2)
    axes[0, 0].plot(epochs, adaptive_results['train_losses'], 'r-', 
                   label='Adaptive FNO', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Burgers Equation - Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 测试损失对比
    axes[0, 1].plot(epochs, original_results['test_losses'], 'b-', 
                   label='Original FNO', linewidth=2)
    axes[0, 1].plot(epochs, adaptive_results['test_losses'], 'r-', 
                   label='Adaptive FNO', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Burgers Equation - Test Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 训练时间对比
    axes[1, 0].plot(epochs, original_results['training_times'], 'b-', 
                   label='Original FNO', linewidth=2)
    axes[1, 0].plot(epochs, adaptive_results['training_times'], 'r-', 
                   label='Adaptive FNO', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Training Time (s)')
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 损失收敛对比（最后50个epoch）
    if len(epochs) > 50:
        recent_epochs = epochs[-50:]
        axes[1, 1].plot(recent_epochs, original_results['test_losses'][-50:], 'b-', 
                       label='Original FNO', linewidth=2)
        axes[1, 1].plot(recent_epochs, adaptive_results['test_losses'][-50:], 'r-', 
                       label='Adaptive FNO', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Test Loss')
        axes[1, 1].set_title('Convergence (Last 50 Epochs)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/neuraloperator/burgers_comparison_training.png', 
                dpi=150, bbox_inches='tight')
    plt.show()


def print_burgers_comparison_summary(original_results, adaptive_results, eval_results):
    """
    打印Burgers方程对比总结
    """
    print("\n" + "="*70)
    print("BURGERS EQUATION FNO COMPARISON SUMMARY")
    print("="*70)
    
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
    
    # L2 误差
    orig_l2 = eval_results['Original FNO']['l2_error']
    adapt_l2 = eval_results['Adaptive FNO']['l2_error']
    l2_improve = (orig_l2 - adapt_l2) / orig_l2 * 100
    print(f"{'L2 Error':<25} {orig_l2:<15.6f} {adapt_l2:<15.6f} {l2_improve:>+13.2f}%")
    
    # H1 误差
    orig_h1 = eval_results['Original FNO']['h1_error']
    adapt_h1 = eval_results['Adaptive FNO']['h1_error']
    h1_improve = (orig_h1 - adapt_h1) / orig_h1 * 100
    print(f"{'H1 Error':<25} {orig_h1:<15.6f} {adapt_h1:<15.6f} {h1_improve:>+13.2f}%")
    
    # 推理时间
    orig_inf = eval_results['Original FNO']['avg_inference_time']
    adapt_inf = eval_results['Adaptive FNO']['avg_inference_time']
    inf_change = (adapt_inf - orig_inf) / orig_inf * 100
    print(f"{'Inference Time (ms)':<25} {orig_inf:<15.2f} {adapt_inf:<15.2f} {inf_change:>+13.2f}%")
    
    print("\n🏆 BURGERS EQUATION ASSESSMENT:")
    if test_improve > 0 and l2_improve > 0:
        print("✅ Adaptive FNO shows BETTER accuracy on Burgers equation")
    elif test_improve < -5 or l2_improve < -5:
        print("❌ Adaptive FNO shows WORSE accuracy on Burgers equation")
    else:
        print("➖ Adaptive FNO shows SIMILAR accuracy to Original FNO")
    
    if abs(inf_change) < 15:
        print("✅ Adaptive FNO has ACCEPTABLE computational overhead")
    elif inf_change > 15:
        print("⚠️  Adaptive FNO has SIGNIFICANT computational overhead")
    else:
        print("✅ Adaptive FNO has LOWER computational cost")
    
    print("\n📈 FREQUENCY ADAPTATION BENEFITS:")
    print("   Burgers equation involves shock formation and wave propagation,")
    print("   which benefits from adaptive frequency selection for:")
    print("   • Better shock capturing (high-frequency components)")
    print("   • Efficient smooth region modeling (low-frequency components)")
    print("   • Dynamic frequency importance weighting")
    
    print("="*70)


def main():
    """
    主函数：在Burgers方程上对比FNO和自适应FNO
    """
    print("🚀 Starting Burgers Equation FNO vs Adaptive FNO Comparison")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载Burgers数据集
    data_path = get_project_root() / 'neuralop/data/datasets/data'
    train_loader, test_loader, data_processor = load_burgers_data(
        data_path=data_path,
        n_train=800,
        n_test=200,
        batch_size=16
    )
    
    if train_loader is None:
        print("❌ Failed to load Burgers dataset. Exiting.")
        return
    
    # 获取输入形状
    for batch in train_loader:
        input_shape = batch['x'].shape
        break
    
    # 创建模型
    print(f"\n🏗️  Creating models for input shape: {input_shape}")
    original_fno, adaptive_fno = create_burgers_models(device, input_shape)
    
    # 训练模型
    n_epochs = 100
    print(f"\n🎯 Training models for {n_epochs} epochs...")
    
    original_results = train_burgers_model(
        original_fno, train_loader, test_loader, device, "Original FNO", n_epochs
    )
    
    adaptive_results = train_burgers_model(
        adaptive_fno, train_loader, test_loader, device, "Adaptive FNO", n_epochs
    )
    
    # 评估模型
    eval_results = evaluate_burgers_models(original_fno, adaptive_fno, test_loader, device)
    
    # 可视化预测结果
    visualize_burgers_predictions(original_fno, adaptive_fno, test_loader, device)
    
    # 绘制训练过程对比
    plot_burgers_training_comparison(original_results, adaptive_results)
    
    # 打印对比总结
    print_burgers_comparison_summary(original_results, adaptive_results, eval_results)
    
    # 保存模型
    torch.save(original_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_original_fno.pth')
    torch.save(adaptive_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_adaptive_fno.pth')
    
    print("\n💾 Models saved successfully!")
    print("🎉 Burgers equation comparison completed!")


if __name__ == "__main__":
    main()
