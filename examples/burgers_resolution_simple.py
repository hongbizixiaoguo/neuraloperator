"""
简化版Burgers方程多分辨率对比实验
使用现有的Burgers数据集，对比FNO和AdaptiveFNO在不同设置下的性能
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

from neuralop.models.fno import FNO1d
from neuralop.models.adaptive_fno import AdaptiveFNO1d
from neuralop.data.datasets import load_mini_burgers_1dtime
from neuralop.utils import get_project_root

def train_and_evaluate_model(model, train_loader, test_loader, n_epochs=50, device='cuda'):
    """
    训练和评估模型
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_data in train_loader:
            # Burgers数据格式处理
            if isinstance(batch_data, dict):
                batch_inputs = batch_data['x'].to(device).float()  # 确保是float类型
                batch_outputs = batch_data['y'].to(device).float()
            else:
                batch_inputs = batch_data[0].to(device).float()  # 确保是float类型
                batch_outputs = batch_data[1].to(device).float()
            
            # 确保输入输出维度正确
            if len(batch_inputs.shape) == 4:  # [B, C, T, S] -> [B, C, S]
                batch_inputs = batch_inputs[:, :, 0, :]  # 取第一个时间步
            if len(batch_outputs.shape) == 4:  # [B, C, T, S] -> [B, C, S]  
                batch_outputs = batch_outputs[:, :, -1, :]  # 取最后一个时间步
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = F.mse_loss(predictions, batch_outputs)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Loss = {avg_epoch_loss:.6f}")
    
    # 评估
    model.eval()
    total_relative_error = 0
    n_samples = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Burgers数据格式处理
            if isinstance(batch_data, dict):
                batch_inputs = batch_data['x'].to(device).float()  # 确保是float类型
                batch_outputs = batch_data['y'].to(device).float()
            else:
                batch_inputs = batch_data[0].to(device).float()  # 确保是float类型
                batch_outputs = batch_data[1].to(device).float()
            
            # 确保输入输出维度正确
            if len(batch_inputs.shape) == 4:  # [B, C, T, S] -> [B, C, S]
                batch_inputs = batch_inputs[:, :, 0, :]
            if len(batch_outputs.shape) == 4:  # [B, C, T, S] -> [B, C, S]
                batch_outputs = batch_outputs[:, :, -1, :]
            
            predictions = model(batch_inputs)
            
            # 计算相对L2误差
            batch_size = batch_inputs.size(0)
            for i in range(batch_size):
                pred = predictions[i].flatten()
                true = batch_outputs[i].flatten()
                relative_error = torch.norm(pred - true, p=2) / (torch.norm(true, p=2) + 1e-8)
                total_relative_error += relative_error.item()
                n_samples += 1
    
    avg_relative_error = total_relative_error / n_samples
    return avg_relative_error

def main():
    """运行简化版多分辨率对比实验"""
    print("🚀 Burgers方程简化版多分辨率对比实验")
    print("="*50)
    
    # 创建实验文件夹
    results_dir = '/root/autodl-tmp/neuraloperator/experiments/burgers_simple_comparison'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/figures', exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # 实验参数
    n_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 不同的模型配置 (模拟不同"分辨率")
    configs = [
        {'name': '8_modes', 'n_modes': 8, 'hidden_channels': 16},
        {'name': '12_modes', 'n_modes': 12, 'hidden_channels': 24}, 
        {'name': '16_modes', 'n_modes': 16, 'hidden_channels': 32},
        {'name': '20_modes', 'n_modes': 20, 'hidden_channels': 40},
        {'name': '24_modes', 'n_modes': 24, 'hidden_channels': 48},
    ]
    
    # 加载数据
    print(f"\n📊 加载Burgers数据...")
    data_path = get_project_root() / 'neuralop/data/datasets/data'
    
    try:
        train_loader, test_loaders, data_processor = load_mini_burgers_1dtime(
            data_path=data_path,
            n_train=800,
            n_test=200, 
            batch_size=16,
            test_batch_size=16,
            temporal_subsample=1,
            spatial_subsample=1
        )
        test_loader = test_loaders[16]  # 使用16分辨率的测试数据
        print("✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("使用模拟数据...")
        # 这里可以添加模拟数据生成代码
        return
    
    # 存储结果
    results = {
        'FNO': [],
        'FNO-MoE': []
    }
    config_names = []
    
    # 对每个配置进行实验
    for config in configs:
        config_name = config['name']
        n_modes = config['n_modes']
        hidden_channels = config['hidden_channels']
        
        print(f"\n📐 测试配置: {config_name} (modes={n_modes}, channels={hidden_channels})")
        config_names.append(config_name)
        
        # 创建模型
        models = {
            'FNO': FNO1d(
                n_modes_height=n_modes,
                in_channels=1,
                out_channels=1,
                hidden_channels=hidden_channels,
                n_layers=4
            ),
            'FNO-MoE': AdaptiveFNO1d(
                n_modes=n_modes,
                in_channels=1,
                out_channels=1,
                hidden_channels=hidden_channels,
                n_layers=4,
                n_experts=4,
                temperature=1.0
            )
        }
        
        # 训练和评估每个模型
        for model_name, model in models.items():
            print(f"  🚀 训练 {model_name}...")
            start_time = time.time()
            
            relative_error = train_and_evaluate_model(
                model, train_loader, test_loader, n_epochs, device
            )
            
            training_time = time.time() - start_time
            results[model_name].append(relative_error)
            
            print(f"    ✅ {model_name}: 相对误差 = {relative_error:.6f}, 训练时间 = {training_time:.1f}s")
            
            # 保存模型
            torch.save(model.state_dict(), 
                      f'{results_dir}/models/{model_name.lower().replace("-", "_")}_{config_name}.pth')
    
    # 生成结果图
    print(f"\n📈 生成结果图...")
    
    plt.figure(figsize=(12, 8))
    
    # 绘制结果
    x_positions = range(len(config_names))
    colors = {'FNO': 'darkred', 'FNO-MoE': 'blue'}
    markers = {'FNO': 'o', 'FNO-MoE': 's'}
    
    for model_name in ['FNO', 'FNO-MoE']:
        plt.plot(x_positions, results[model_name], 
                color=colors[model_name], marker=markers[model_name], 
                linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel('Model Configuration', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('Burger\'s Equation: FNO vs FNO-MoE Performance Comparison', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置x轴标签
    plt.xticks(x_positions, [name.replace('_', '\n') for name in config_names], rotation=45)
    
    # 美化图表
    plt.tight_layout()
    plt.savefig(f'{results_dir}/figures/burgers_model_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印结果总结
    print(f"\n{'='*60}")
    print(f"BURGERS方程模型对比结果")
    print(f"{'='*60}")
    
    summary_lines = ["BURGERS方程模型对比结果", "="*60, ""]
    
    for i, config_name in enumerate(config_names):
        fno_error = results['FNO'][i]
        moe_error = results['FNO-MoE'][i]
        improvement = ((fno_error - moe_error) / fno_error) * 100
        
        print(f"{config_name:12s}: FNO = {fno_error:.6f}, FNO-MoE = {moe_error:.6f}, 改进 = {improvement:+.2f}%")
        summary_lines.append(f"{config_name:12s}: FNO = {fno_error:.6f}, FNO-MoE = {moe_error:.6f}, 改进 = {improvement:+.2f}%")
    
    # 计算平均改进
    improvements = [
        ((results['FNO'][i] - results['FNO-MoE'][i]) / results['FNO'][i]) * 100 
        for i in range(len(config_names))
    ]
    avg_improvement = np.mean(improvements)
    
    print(f"\n平均改进: {avg_improvement:+.2f}%")
    summary_lines.extend(["", f"平均改进: {avg_improvement:+.2f}%"])
    
    # 保存总结
    with open(f'{results_dir}/experiment_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ 结果保存到: {results_dir}")
    print(f"📊 对比图: figures/burgers_model_comparison.png")
    print(f"🤖 模型文件: models/")
    print(f"📝 实验总结: experiment_summary.txt")
    
    if avg_improvement > 0:
        print(f"\n🎉 FNO-MoE平均改进 {avg_improvement:.2f}%!")
    else:
        print(f"\n📝 需要进一步调优参数")

if __name__ == "__main__":
    main()
