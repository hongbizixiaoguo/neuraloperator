"""
Heat Equation实验测试版本 - 快速验证代码
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 添加neuraloperator到路径
sys.path.append('/root/autodl-tmp/neuraloperator')

from neuralop.models.fno import FNO2d
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d

def main():
    """快速测试Heat Equation实验设置"""
    print("🔥 Heat Equation Quick Test (50 epochs)")
    print("="*50)
    
    # 创建实验文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/heat_equation_test'
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # 生成少量测试数据
    print("📊 Generating test data...")
    n_samples = 50
    grid_size = 32  # 更小的网格加速测试
    
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs, outputs = [], []
    
    for _ in range(n_samples):
        k1, k2 = np.random.randint(1, 6, 2)
        phase = np.random.uniform(0, 2*np.pi)
        
        u0 = np.sin(2*np.pi*k1*X + phase) * np.cos(2*np.pi*k2*Y)
        
        diffusion = 0.01
        t = 0.1
        decay = np.exp(-diffusion * (k1**2 + k2**2) * t)
        u1 = decay * u0
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
    
    inputs = torch.FloatTensor(np.array(inputs))
    outputs = torch.FloatTensor(np.array(outputs))
    
    # 分割数据
    train_size = 40
    train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
    train_outputs, test_outputs = outputs[:train_size], outputs[train_size:]
    
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"✓ Data: {train_size} train, {len(test_inputs)} test samples")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    original_fno = FNO2d(
        n_modes_height=8, n_modes_width=8,
        in_channels=1, out_channels=1,
        hidden_channels=16, n_layers=2
    ).to(device)
    
    adaptive_fno = AdaptiveFNO2d(
        n_modes_height=8, n_modes_width=8,
        in_channels=1, out_channels=1,
        hidden_channels=16, n_layers=2,
        n_experts=4
    ).to(device)
    
    models = {'Original FNO': original_fno, 'Adaptive FNO': adaptive_fno}
    results = {}
    
    # 快速训练测试
    for name, model in models.items():
        print(f"\n🚀 Testing {name} (50 epochs)...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_losses, test_losses = [], []
        
        for epoch in range(50):
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
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}")
        
        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_test_loss': test_losses[-1]
        }
    
    # 简单可视化
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        plt.plot(result['train_losses'], color=color, label=f'{name} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        color = 'blue' if 'Original' in name else 'red'
        plt.plot(result['test_losses'], color=color, label=f'{name} Test')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/quick_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印结果
    print(f"\n📊 Quick Test Results:")
    for name, result in results.items():
        print(f"{name}: Final Test Loss = {result['final_test_loss']:.6f}")
    
    original_loss = results['Original FNO']['final_test_loss']
    adaptive_loss = results['Adaptive FNO']['final_test_loss']
    improvement = ((original_loss - adaptive_loss) / original_loss) * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    print(f"\n✅ Quick test completed! Code is ready for 500 epochs.")
    print(f"📁 Test results saved in: {results_dir}")

if __name__ == "__main__":
    main()


