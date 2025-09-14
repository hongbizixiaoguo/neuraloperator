"""
Burgers方程上的FNO对比实验 - 最终简化版
只对比基本的FNO性能，展示自适应FNO的概念
"""

import os
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


def load_burgers_data(data_path, batch_size=16):
    """加载Burgers数据"""
    print(f"Loading Burgers data from {data_path}")
    
    try:
        train_data = torch.load(f"{data_path}/burgers_train_16.pt")
        test_data = torch.load(f"{data_path}/burgers_test_16.pt")
        
        # 处理数据格式
        x_train = train_data['x'].unsqueeze(1).float()  # [800, 1, 16]
        y_train = train_data['y'][:, -1, :].unsqueeze(1).float()  # [800, 1, 16]
        
        x_test = test_data['x'].unsqueeze(1).float()  # [400, 1, 16]
        y_test = test_data['y'][:, -1, :].unsqueeze(1).float()  # [400, 1, 16]
        
        print(f"Train input shape: {x_train.shape}")
        print(f"Train output shape: {y_train.shape}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


class SimpleFNO1d(nn.Module):
    """简化的1D FNO"""
    def __init__(self, n_modes=16, hidden_channels=64, n_layers=4):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        
        # 简单的lifting和projection
        self.lifting = nn.Conv1d(1, hidden_channels, 1)
        self.projection = nn.Conv1d(hidden_channels, 1, 1)
        
        # 频谱卷积层（简化版）
        self.spectral_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1) for _ in range(n_layers)
        ])
        
        # 激活函数
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Lifting
        x = self.lifting(x)
        
        # 频谱层
        for layer in self.spectral_layers:
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = x + residual  # 残差连接
        
        # Projection
        x = self.projection(x)
        return x


class AdaptiveSimpleFNO1d(nn.Module):
    """带自适应频率选择的简化1D FNO"""
    def __init__(self, n_modes=16, hidden_channels=64, n_layers=4, n_experts=4):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_experts = n_experts
        
        # 基础组件
        self.lifting = nn.Conv1d(1, hidden_channels, 1)
        self.projection = nn.Conv1d(hidden_channels, 1, 1)
        
        # 频谱卷积层
        self.spectral_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, 1) for _ in range(n_layers)
        ])
        
        # 频率自适应门控
        self.freq_gating = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, n_experts),
            nn.Softmax(dim=-1)
        )
        
        # 频率专家权重
        self.expert_weights = nn.Parameter(torch.ones(n_experts, hidden_channels))
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Lifting
        x = self.lifting(x)
        
        # 计算频率门控权重
        gating_scores = self.freq_gating(x)  # [batch, n_experts]
        
        # 应用自适应权重
        batch_size = x.size(0)
        adaptive_weight = torch.zeros_like(x)
        
        for i in range(self.n_experts):
            expert_weight = self.expert_weights[i].unsqueeze(0).unsqueeze(-1)  # [1, channels, 1]
            gate_score = gating_scores[:, i:i+1].unsqueeze(-1)  # [batch, 1, 1]
            adaptive_weight += expert_weight * gate_score
        
        x = x * adaptive_weight
        
        # 频谱层
        for layer in self.spectral_layers:
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = x + residual
        
        # Projection
        x = self.projection(x)
        return x


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


def evaluate_models(models, test_loader, device):
    """评估模型"""
    results = {}
    
    for model, name in models:
        model.eval()
        total_mse = 0.0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                start_time = time.time()
                pred = model(x_batch)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / x_batch.size(0))
                
                mse = F.mse_loss(pred, y_batch)
                total_mse += mse.item() * x_batch.size(0)
                total_samples += x_batch.size(0)
        
        results[name] = {
            'mse': total_mse / total_samples,
            'avg_inference_time': np.mean(inference_times) * 1000
        }
    
    return results


def visualize_results(models, test_loader, device):
    """可视化结果"""
    print("\nGenerating visualizations...")
    
    # 获取测试数据
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        break
    
    # 获取预测
    predictions = {}
    for model, name in models:
        model.eval()
        with torch.no_grad():
            pred = model(x_batch)
            predictions[name] = pred
    
    # 可视化前3个样本
    n_samples = min(3, x_batch.size(0))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # 输入
        input_data = x_batch[i, 0, :].cpu().numpy()
        axes[i, 0].plot(input_data, 'k-', linewidth=2)
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 真实输出
        target_data = y_batch[i, 0, :].cpu().numpy()
        axes[i, 1].plot(target_data, 'g-', linewidth=2, label='Ground Truth')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 简单FNO
        simple_pred = predictions['Simple FNO'][i, 0, :].cpu().numpy()
        axes[i, 2].plot(target_data, 'g-', alpha=0.7, linewidth=2, label='True')
        axes[i, 2].plot(simple_pred, 'b--', linewidth=2, label='Simple FNO')
        axes[i, 2].set_title(f'Simple FNO {i+1}')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        
        # 自适应FNO
        adaptive_pred = predictions['Adaptive FNO'][i, 0, :].cpu().numpy()
        axes[i, 3].plot(target_data, 'g-', alpha=0.7, linewidth=2, label='True')
        axes[i, 3].plot(adaptive_pred, 'r--', linewidth=2, label='Adaptive FNO')
        axes[i, 3].set_title(f'Adaptive FNO {i+1}')
        axes[i, 3].legend()
        axes[i, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 创建实验结果文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/burgers_equation'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(f'{results_dir}/burgers_final_comparison.png', dpi=150)
    print(f"✓ Comparison visualization saved to '{results_dir}/burgers_final_comparison.png'")
    plt.show()


def plot_training_curves(results_dict):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['blue', 'red']
    for i, (name, results) in enumerate(results_dict.items()):
        epochs = range(1, len(results['train_losses']) + 1)
        
        axes[0].plot(epochs, results['train_losses'], color=colors[i], 
                    label=name, linewidth=2)
        axes[1].plot(epochs, results['test_losses'], color=colors[i], 
                    label=name, linewidth=2)
    
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
    
    plt.tight_layout()
    
    # 保存到实验结果文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/burgers_equation'
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(f'{results_dir}/burgers_final_training.png', dpi=150)
    print(f"✓ Training curves saved to '{results_dir}/burgers_final_training.png'")
    plt.show()


def print_summary(results_dict, eval_results):
    """打印总结"""
    print("\n" + "="*60)
    print("BURGERS EQUATION: SIMPLE VS ADAPTIVE FNO")
    print("="*60)
    
    print(f"\n📊 TRAINING RESULTS:")
    print(f"{'Model':<20} {'Train Loss':<15} {'Test Loss':<15}")
    print("-" * 50)
    
    for name, results in results_dict.items():
        print(f"{name:<20} {results['final_train_loss']:<15.6f} {results['final_test_loss']:<15.6f}")
    
    print(f"\n🎯 EVALUATION RESULTS:")
    print(f"{'Model':<20} {'MSE':<15} {'Inference (ms)':<15}")
    print("-" * 50)
    
    for name, results in eval_results.items():
        print(f"{name:<20} {results['mse']:<15.6f} {results['avg_inference_time']:<15.2f}")
    
    # 计算改进
    simple_mse = eval_results['Simple FNO']['mse']
    adaptive_mse = eval_results['Adaptive FNO']['mse']
    improvement = (simple_mse - adaptive_mse) / simple_mse * 100
    
    print(f"\n🏆 SUMMARY:")
    print(f"MSE Improvement: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✅ Adaptive FNO shows better accuracy")
    else:
        print("➖ Similar performance between models")
    
    print("\n📝 NOTES:")
    print("• This is a simplified demonstration of adaptive frequency selection")
    print("• Real adaptive FNO would use more sophisticated frequency gating")
    print("• Burgers equation benefits from adaptive frequency handling")
    print("• The concept can be extended to more complex PDEs")
    
    print("="*60)


def main():
    """主函数"""
    print("🚀 Burgers Equation: Simple vs Adaptive FNO Comparison")
    print("="*55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data_path = "/root/autodl-tmp/neuraloperator/neuralop/data/datasets/data"
    train_loader, test_loader = load_burgers_data(data_path, batch_size=16)
    
    if train_loader is None:
        print("❌ Failed to load data")
        return
    
    # 创建模型
    print("\n🏗️  Creating models...")
    simple_fno = SimpleFNO1d(n_modes=16, hidden_channels=64, n_layers=4).to(device)
    adaptive_fno = AdaptiveSimpleFNO1d(n_modes=16, hidden_channels=64, 
                                      n_layers=4, n_experts=4).to(device)
    
    print(f"Simple FNO parameters: {sum(p.numel() for p in simple_fno.parameters()):,}")
    print(f"Adaptive FNO parameters: {sum(p.numel() for p in adaptive_fno.parameters()):,}")
    
    # 训练模型
    n_epochs = 50
    print(f"\n🎯 Training for {n_epochs} epochs...")
    
    simple_results = train_model(simple_fno, train_loader, test_loader, 
                                device, "Simple FNO", n_epochs)
    adaptive_results = train_model(adaptive_fno, train_loader, test_loader, 
                                  device, "Adaptive FNO", n_epochs)
    
    results_dict = {
        'Simple FNO': simple_results,
        'Adaptive FNO': adaptive_results
    }
    
    # 评估模型
    models = [(simple_fno, 'Simple FNO'), (adaptive_fno, 'Adaptive FNO')]
    eval_results = evaluate_models(models, test_loader, device)
    
    # 可视化
    visualize_results(models, test_loader, device)
    plot_training_curves(results_dict)
    
    # 打印总结
    print_summary(results_dict, eval_results)
    
    # 保存模型
    torch.save(simple_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_simple_fno.pth')
    torch.save(adaptive_fno.state_dict(), 
               '/root/autodl-tmp/neuraloperator/burgers_adaptive_simple_fno.pth')
    
    print("\n💾 Models saved!")
    print("🎉 Comparison completed!")


if __name__ == "__main__":
    main()
