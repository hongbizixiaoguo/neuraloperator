"""
简化版Burgers方程FNO对比实验
直接加载数据文件，避免复杂的数据加载器
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop.models.fno import FNO1d
from neuralop.models.final_adaptive_fno import AdaptiveFNO

# 自定义1D自适应FNO
class AdaptiveFNO1d(AdaptiveFNO):
    """1D自适应FNO"""
    def __init__(self, n_modes_height: int, hidden_channels: int, 
                 in_channels: int = 2, out_channels: int = 1, 
                 n_layers: int = 4, n_experts: int = 4, **kwargs):
        super().__init__(
            n_modes=(n_modes_height,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_experts=n_experts,
            **kwargs
        )


def load_burgers_data_simple(data_path, batch_size=16):
    """
    直接加载Burgers数据文件
    """
    print(f"Loading Burgers data from {data_path}")
    
    try:
        # 加载训练数据
        train_data = torch.load(f"{data_path}/burgers_train_16.pt")
        test_data = torch.load(f"{data_path}/burgers_test_16.pt")
        
        print(f"Train data keys: {train_data.keys()}")
        print(f"Test data keys: {test_data.keys()}")
        
        # 提取输入和输出
        x_train = train_data['x']  # 初始条件
        y_train = train_data['y']  # 解
        x_test = test_data['x']
        y_test = test_data['y']
        
        print(f"Original train input shape: {x_train.shape}")
        print(f"Original train output shape: {y_train.shape}")
        
        # 重新整理数据格式以适应FNO
        # x: [N, spatial] -> [N, 1, spatial] (添加通道维度)
        # y: [N, time, spatial] -> [N, 1, spatial] (取最后时间步)
        x_train = x_train.unsqueeze(1).float()  # [800, 1, 16]
        y_train = y_train[:, -1, :].unsqueeze(1).float()  # [800, 1, 16] 取最后时间步
        
        x_test = x_test.unsqueeze(1).float()  # [400, 1, 16]
        y_test = y_test[:, -1, :].unsqueeze(1).float()  # [400, 1, 16]
        
        print(f"Processed train input shape: {x_train.shape}")
        print(f"Processed train output shape: {y_train.shape}")
        print(f"Processed test input shape: {x_test.shape}")
        print(f"Processed test output shape: {y_test.shape}")
        
        # 创建数据集
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def create_models(device, input_channels, spatial_size):
    """创建FNO模型"""
    print(f"Creating models for {input_channels} input channels, spatial size {spatial_size}")
    
    # 原始FNO1d
    original_fno = FNO1d(
        n_modes_height=16,
        in_channels=input_channels,
        out_channels=1,
        hidden_channels=64,
        n_layers=4
    ).to(device)
    
    # 自适应FNO1d  
    adaptive_fno = AdaptiveFNO1d(
        n_modes_height=16,
        in_channels=input_channels,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        n_experts=4
    ).to(device)
    
    print(f"Original FNO parameters: {sum(p.numel() for p in original_fno.parameters()):,}")
    print(f"Adaptive FNO parameters: {sum(p.numel() for p in adaptive_fno.parameters()):,}")
    
    return original_fno, adaptive_fno


def train_model(model, train_loader, test_loader, device, model_name, n_epochs=50):
    """训练模型"""
    print(f"\nTraining {model_name}...")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(x_batch)
            
            # 调整形状匹配
            if pred.shape != y_batch.shape:
                if len(y_batch.shape) == 4:  # [batch, channels, time, space]
                    y_batch = y_batch[:, :, -1:, :]  # 取最后时间步
                pred = pred.reshape(y_batch.shape)
            
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 测试
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                
                if pred.shape != y_batch.shape:
                    if len(y_batch.shape) == 4:
                        y_batch = y_batch[:, :, -1:, :]
                    pred = pred.reshape(y_batch.shape)
                
                loss = criterion(pred, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }


def evaluate_models(original_fno, adaptive_fno, test_loader, device):
    """评估模型"""
    print("\nEvaluating models...")
    
    results = {}
    
    for model, name in [(original_fno, 'Original FNO'), (adaptive_fno, 'Adaptive FNO')]:
        model.eval()
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                
                if pred.shape != y_batch.shape:
                    if len(y_batch.shape) == 4:
                        y_batch = y_batch[:, :, -1:, :]
                    pred = pred.reshape(y_batch.shape)
                
                mse = F.mse_loss(pred, y_batch)
                total_mse += mse.item() * x_batch.size(0)
                total_samples += x_batch.size(0)
        
        results[name] = {
            'mse': total_mse / total_samples,
        }
    
    return results


def visualize_results(original_fno, adaptive_fno, test_loader, device):
    """可视化结果"""
    print("\nGenerating visualizations...")
    
    original_fno.eval()
    adaptive_fno.eval()
    
    # 获取一批测试数据
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        break
    
    with torch.no_grad():
        original_pred = original_fno(x_batch)
        adaptive_pred = adaptive_fno(x_batch)
    
    # 调整形状
    if original_pred.shape != y_batch.shape:
        if len(y_batch.shape) == 4:
            y_batch = y_batch[:, :, -1:, :]
        original_pred = original_pred.reshape(y_batch.shape)
        adaptive_pred = adaptive_pred.reshape(y_batch.shape)
    
    # 可视化前3个样本
    n_samples = min(3, x_batch.size(0))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 输入
        if len(x_batch.shape) == 4:
            input_data = x_batch[i, 0, 0, :].cpu().numpy()
        else:
            input_data = x_batch[i, 0, :].cpu().numpy()
        
        axes[i, 0].plot(input_data)
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 真实输出
        if len(y_batch.shape) == 4:
            target_data = y_batch[i, 0, -1, :].cpu().numpy()
        else:
            target_data = y_batch[i, 0, :].cpu().numpy()
        
        axes[i, 1].plot(target_data, 'g-', label='True', linewidth=2)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 原始FNO
        if len(original_pred.shape) == 4:
            orig_data = original_pred[i, 0, -1, :].cpu().numpy()
        else:
            orig_data = original_pred[i, 0, :].cpu().numpy()
        
        axes[i, 2].plot(target_data, 'g-', label='True', alpha=0.7)
        axes[i, 2].plot(orig_data, 'b--', label='Original FNO', linewidth=2)
        axes[i, 2].set_title(f'Original FNO {i+1}')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        # 自适应FNO
        if len(adaptive_pred.shape) == 4:
            adapt_data = adaptive_pred[i, 0, -1, :].cpu().numpy()
        else:
            adapt_data = adaptive_pred[i, 0, :].cpu().numpy()
        
        axes[i, 3].plot(target_data, 'g-', label='True', alpha=0.7)
        axes[i, 3].plot(adapt_data, 'r--', label='Adaptive FNO', linewidth=2)
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].legend()
        axes[i, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/neuraloperator/burgers_simple_comparison.png', dpi=150)
    plt.show()


def plot_training_curves(original_results, adaptive_results):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(original_results['train_losses']) + 1)
    
    # 训练损失
    axes[0].plot(epochs, original_results['train_losses'], 'b-', label='Original FNO')
    axes[0].plot(epochs, adaptive_results['train_losses'], 'r-', label='Adaptive FNO')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 测试损失
    axes[1].plot(epochs, original_results['test_losses'], 'b-', label='Original FNO')
    axes[1].plot(epochs, adaptive_results['test_losses'], 'r-', label='Adaptive FNO')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/neuraloperator/burgers_simple_training.png', dpi=150)
    plt.show()


def print_summary(original_results, adaptive_results, eval_results):
    """打印总结"""
    print("\n" + "="*60)
    print("BURGERS EQUATION COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"{'Metric':<20} {'Original FNO':<15} {'Adaptive FNO':<15} {'Improvement':<15}")
    print("-" * 65)
    
    # 训练损失
    orig_train = original_results['final_train_loss']
    adapt_train = adaptive_results['final_train_loss']
    train_improve = (orig_train - adapt_train) / orig_train * 100
    print(f"{'Train Loss':<20} {orig_train:<15.6f} {adapt_train:<15.6f} {train_improve:>+13.2f}%")
    
    # 测试损失
    orig_test = original_results['final_test_loss']
    adapt_test = adaptive_results['final_test_loss']
    test_improve = (orig_test - adapt_test) / orig_test * 100
    print(f"{'Test Loss':<20} {orig_test:<15.6f} {adapt_test:<15.6f} {test_improve:>+13.2f}%")
    
    # MSE
    orig_mse = eval_results['Original FNO']['mse']
    adapt_mse = eval_results['Adaptive FNO']['mse']
    mse_improve = (orig_mse - adapt_mse) / orig_mse * 100
    print(f"{'Test MSE':<20} {orig_mse:<15.6f} {adapt_mse:<15.6f} {mse_improve:>+13.2f}%")
    
    print("\n🏆 ASSESSMENT:")
    if test_improve > 0 and mse_improve > 0:
        print("✅ Adaptive FNO shows BETTER performance on Burgers equation")
    elif test_improve < -2:
        print("❌ Adaptive FNO shows WORSE performance")
    else:
        print("➖ Similar performance between models")
    
    print("="*60)


def main():
    """主函数"""
    print("🚀 Burgers Equation: FNO vs Adaptive FNO")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data_path = "/root/autodl-tmp/neuraloperator/neuralop/data/datasets/data"
    train_loader, test_loader = load_burgers_data_simple(data_path, batch_size=16)
    
    if train_loader is None:
        print("❌ Failed to load data")
        return
    
    # 获取数据信息
    for x_batch, y_batch in train_loader:
        input_channels = x_batch.shape[1]
        spatial_size = x_batch.shape[-1]
        break
    
    # 创建模型
    original_fno, adaptive_fno = create_models(device, input_channels, spatial_size)
    
    # 训练模型
    n_epochs = 50
    print(f"\n🎯 Training for {n_epochs} epochs...")
    
    original_results = train_model(original_fno, train_loader, test_loader, 
                                 device, "Original FNO", n_epochs)
    adaptive_results = train_model(adaptive_fno, train_loader, test_loader, 
                                 device, "Adaptive FNO", n_epochs)
    
    # 评估模型
    eval_results = evaluate_models(original_fno, adaptive_fno, test_loader, device)
    
    # 可视化
    visualize_results(original_fno, adaptive_fno, test_loader, device)
    plot_training_curves(original_results, adaptive_results)
    
    # 打印总结
    print_summary(original_results, adaptive_results, eval_results)
    
    # 保存模型
    torch.save(original_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_original_simple.pth')
    torch.save(adaptive_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_adaptive_simple.pth')
    
    print("\n💾 Models saved!")
    print("🎉 Comparison completed!")


if __name__ == "__main__":
    main()
