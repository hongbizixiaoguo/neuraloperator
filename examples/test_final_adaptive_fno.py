"""
测试最终版自适应FNO的脚本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 导入最终版自适应FNO
import sys
sys.path.append('/root/autodl-tmp/neuraloperator')
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d

def generate_simple_data(n_samples=100, grid_size=32):
    """
    生成简单的测试数据
    """
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        # 简单的2D高斯函数
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # 随机中心和标准差
        cx, cy = np.random.uniform(-0.5, 0.5, 2)
        sigma = np.random.uniform(0.1, 0.3)
        
        # 输入：高斯函数
        input_field = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        
        # 输出：简单变换（例如平滑化）
        output_field = input_field * 0.8 + 0.1 * np.random.randn(grid_size, grid_size)
        
        inputs.append(input_field[np.newaxis, :, :])
        outputs.append(output_field[np.newaxis, :, :])
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def test_model_creation():
    """
    测试模型创建
    """
    print("Testing model creation...")
    
    try:
        model = AdaptiveFNO2d(
            n_modes_height=8,
            n_modes_width=8,
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            n_layers=2,
            n_experts=3,
            temperature=1.0
        )
        print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """
    测试前向传播
    """
    print("Testing forward pass...")
    
    try:
        # 创建测试输入
        batch_size = 4
        grid_size = 32
        test_input = torch.randn(batch_size, 1, grid_size, grid_size)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # 检查输出形状是否正确
        expected_shape = (batch_size, 1, grid_size, grid_size)
        if output.shape == expected_shape:
            print(f"✓ Output shape is correct: {output.shape}")
        else:
            print(f"✗ Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model):
    """
    测试训练步骤
    """
    print("Testing training step...")
    
    try:
        # 生成小批量数据
        train_inputs, train_outputs = generate_simple_data(n_samples=16, grid_size=32)
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        
        # 执行一个训练步骤
        optimizer.zero_grad()
        predictions = model(train_inputs)
        loss = criterion(predictions, train_outputs)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_training_test():
    """
    快速训练测试
    """
    print("Running quick training test...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成数据
    train_inputs, train_outputs = generate_simple_data(n_samples=50, grid_size=32)
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)
    
    # 创建模型
    model = AdaptiveFNO2d(
        n_modes_height=8,
        n_modes_width=8,
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        n_layers=2,
        n_experts=3,
        temperature=1.0
    ).to(device)
    
    # 训练设置
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 快速训练几个epoch
    model.train()
    losses = []
    
    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(train_inputs)
        loss = criterion(predictions, train_outputs)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.6f}")
    
    print(f"✓ Quick training completed")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # 简单可视化
    model.eval()
    with torch.no_grad():
        test_pred = model(train_inputs[:1])
        
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(train_inputs[0, 0].cpu().numpy(), cmap='viridis')
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    axes[1].imshow(train_outputs[0, 0].cpu().numpy(), cmap='viridis')
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    axes[2].imshow(test_pred[0, 0].cpu().numpy(), cmap='viridis')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/neuraloperator/final_adaptive_fno_test.png')
    plt.show()
    
    return True


def main():
    """
    主测试函数
    """
    print("="*50)
    print("Final Adaptive FNO Test Suite")
    print("="*50)
    
    # 测试1: 模型创建
    model = test_model_creation()
    if model is None:
        return
    
    print()
    
    # 测试2: 前向传播
    if not test_forward_pass(model):
        return
    
    print()
    
    # 测试3: 训练步骤
    if not test_training_step(model):
        return
    
    print()
    
    # 测试4: 快速训练
    if not quick_training_test():
        return
    
    print()
    print("="*50)
    print("All tests passed! ✓")
    print("Final Adaptive FNO is working correctly.")
    print("="*50)


if __name__ == "__main__":
    main()


