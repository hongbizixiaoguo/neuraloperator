"""
训练自适应FNO的示例脚本（修复版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这个导入
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 导入最终版自适应FNO
import sys
sys.path.append('/root/autodl-tmp/neuraloperator')
from neuralop.models.final_adaptive_fno import AdaptiveFNO2d

def generate_synthetic_data(n_samples=1000, grid_size=64):
    """
    生成合成的2D PDE数据用于演示
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        # 生成随机的初始条件（多个频率成分的组合）
        k1, k2 = np.random.randint(1, 10, 2)
        k3, k4 = np.random.randint(1, 10, 2)
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        
        # 输入：初始条件
        u0 = (np.sin(2*np.pi*k1*X + phase1) * np.cos(2*np.pi*k2*Y) +
              0.5 * np.sin(2*np.pi*k3*X) * np.sin(2*np.pi*k4*Y + phase2))
        
        # 输出：演化后的场（简化的热方程解）
        diffusion = 0.01
        t = 0.1
        decay1 = np.exp(-diffusion * (k1**2 + k2**2) * t)
        decay2 = np.exp(-diffusion * (k3**2 + k4**2) * t)
        
        u1 = (decay1 * np.sin(2*np.pi*k1*X + phase1) * np.cos(2*np.pi*k2*Y) +
              0.5 * decay2 * np.sin(2*np.pi*k3*X) * np.sin(2*np.pi*k4*Y + phase2))
        
        inputs.append(u0[np.newaxis, :, :])
        outputs.append(u1[np.newaxis, :, :])
    
    return (torch.FloatTensor(np.array(inputs)), 
            torch.FloatTensor(np.array(outputs)))


def visualize_results(model, test_input, test_output):
    """
    可视化模型预测结果
    """
    model.eval()
    with torch.no_grad():
        prediction = model(test_input)
        
        # 可视化输入、真实输出、预测输出
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(test_input[0, 0].cpu().numpy(), cmap='viridis')
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        axes[1].imshow(test_output[0, 0].cpu().numpy(), cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(prediction[0, 0].cpu().numpy(), cmap='viridis')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # 误差图
        error = torch.abs(prediction - test_output)
        axes[3].imshow(error[0, 0].cpu().numpy(), cmap='hot')
        axes[3].set_title('Absolute Error')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig('/root/autodl-tmp/neuraloperator/adaptive_fno_results.png')
        plt.show()
        
        # 计算误差统计
        mse = torch.mean((prediction - test_output) ** 2).item()
        mae = torch.mean(torch.abs(prediction - test_output)).item()
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")


def train_adaptive_fno():
    """
    训练自适应FNO的主函数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 16
    learning_rate = 1e-3
    n_epochs = 50  # 减少epoch数以加快测试
    grid_size = 64
    
    # 生成数据
    print("Generating synthetic data...")
    train_inputs, train_outputs = generate_synthetic_data(n_samples=800, grid_size=grid_size)
    test_inputs, test_outputs = generate_synthetic_data(n_samples=200, grid_size=grid_size)
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs, train_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("Creating Final Adaptive FNO model...")
    model = AdaptiveFNO2d(
        n_modes_height=16,
        n_modes_width=16,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_layers=4,
        n_experts=4,  # 4个频率带
        temperature=1.0,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        non_linearity=F.gelu,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
        fno_skip='linear',
        channel_mlp_skip='soft-gating'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 训练循环
    print("Starting training...")
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
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
        
        # 学习率调度
        scheduler.step(test_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Adaptive FNO Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/root/autodl-tmp/neuraloperator/adaptive_fno_training_fixed.png')
    plt.show()
    
    # 可视化结果
    print("\nVisualizing results...")
    test_sample_input = test_inputs[:1].to(device)
    test_sample_output = test_outputs[:1].to(device)
    visualize_results(model, test_sample_input, test_sample_output)
    
    # 保存模型
    torch.save(model.state_dict(), '/root/autodl-tmp/neuraloperator/adaptive_fno_model_fixed.pth')
    print("\nModel saved to adaptive_fno_model_fixed.pth")
    
    return model, train_losses, test_losses


if __name__ == "__main__":
    # 训练自适应FNO
    model, train_losses, test_losses = train_adaptive_fno()
    
    print("\nTraining completed successfully!")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {test_losses[-1]:.6f}")