#!/usr/bin/env python3
"""
Navier-Stokes方程实验：FNO vs AdaptiveFNO (FnoMoE)对比
使用128分辨率的Navier-Stokes数据集
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb
import numpy as np
import matplotlib.pyplot as plt
import time

# 添加neuraloperator到路径
sys.path.insert(0, '../')

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.training import setup, AdamW
from neuralop.models.fno import FNO
from neuralop.models.final_adaptive_fno import AdaptiveFNO

# 配置导入
from zencfg import make_config_from_cli
from config.navier_stokes_comparison_config import Default


def create_experiment_folder():
    """创建实验文件夹"""
    results_dir = Path('/root/autodl-tmp/neuraloperator/experiments/navier_stokes_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'figures').mkdir(exist_ok=True)
    (results_dir / 'models').mkdir(exist_ok=True)
    (results_dir / 'logs').mkdir(exist_ok=True)
    
    print(f"✓ Created experiment folder: {results_dir}")
    return results_dir


def create_models(config, device):
    """创建FNO和AdaptiveFNO模型"""
    
    # 标准FNO模型
    fno_model = FNO(
        n_modes=tuple(config.fno_model.n_modes),
        in_channels=config.fno_model.in_channels,
        out_channels=config.fno_model.out_channels,
        hidden_channels=config.fno_model.hidden_channels,
        n_layers=config.fno_model.n_layers,
        lifting_channel_ratio=config.fno_model.lifting_channel_ratio,
        projection_channel_ratio=config.fno_model.projection_channel_ratio,
    ).to(device)
    
    # 自适应FNO模型 (FnoMoE)
    adaptive_fno_model = AdaptiveFNO(
        n_modes=tuple(config.adaptive_fno_model.n_modes),
        in_channels=config.adaptive_fno_model.in_channels,
        out_channels=config.adaptive_fno_model.out_channels,
        hidden_channels=config.adaptive_fno_model.hidden_channels,
        n_layers=config.adaptive_fno_model.n_layers,
        lifting_channel_ratio=config.adaptive_fno_model.lifting_channel_ratio,
        projection_channel_ratio=config.adaptive_fno_model.projection_channel_ratio,
        n_experts=config.adaptive_fno_model.n_experts,
        temperature=config.adaptive_fno_model.temperature,
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


def train_single_model(model, model_name, train_loader, test_loaders, data_processor, 
                      config, device, results_dir, is_logger):
    """训练单个模型"""
    
    print(f"\n🚀 Training {model_name}...")
    
    # 优化器设置
    optimizer = AdamW(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )
    
    # 学习率调度器
    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.scheduler_factor,
            patience=config.opt.scheduler_patience,
            mode="min",
            min_lr=config.opt.min_lr,
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")
    
    # 损失函数
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    else:
        raise ValueError(f'Got training_loss={config.opt.training_loss}')
    
    eval_losses = {"h1": h1loss, "l2": l2loss}
    
    # 创建trainer
    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        data_processor=data_processor,
        device=device,
        mixed_precision=config.opt.mixed_precision,
        eval_interval=config.opt.eval_interval,
        log_output=False,  # 关闭wandb输出
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
        wandb_log=False  # 关闭wandb日志
    )
    
    # 训练模型
    start_time = time.time()
    trainer.train(
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )
    training_time = time.time() - start_time
    
    # 保存模型
    model_path = results_dir / 'models' / f'{model_name.lower()}_final.pth'
    torch.save(model.state_dict(), model_path)
    
    # 获取训练历史
    train_losses = trainer.train_err
    test_losses = trainer.test_err
    
    print(f"✓ {model_name} training completed in {training_time:.1f}s")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_test_loss': test_losses[-1] if test_losses else 0,
    }


def evaluate_models(models, param_counts, test_loaders, device):
    """评估模型性能"""
    
    print(f"\n📊 Evaluating models...")
    
    eval_results = {}
    
    for model_name, model in models.items():
        model.eval()
        
        total_l2_error = 0
        total_h1_error = 0
        total_samples = 0
        inference_times = []
        
        l2loss = LpLoss(d=2, p=2)
        h1loss = H1Loss(d=2)
        
        with torch.no_grad():
            for resolution, test_loader in test_loaders.items():
                for batch_inputs, batch_outputs in test_loader:
                    batch_inputs = batch_inputs.to(device)
                    batch_outputs = batch_outputs.to(device)
                    
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
        if result['train_losses'] and result['test_losses']:
            color = colors[model_name]
            epochs = range(1, len(result['train_losses']) + 1)
            
            axes[0].plot(epochs, result['train_losses'], color=color, 
                        label=f'{model_name} Train', alpha=0.7, linewidth=2)
            axes[1].plot(epochs, result['test_losses'], color=color, 
                        label=f'{model_name} Test', alpha=0.7, linewidth=2)
    
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
    plt.close()
    
    # 2. 预测结果对比
    models = {name: result['model'] for name, result in training_results.items()}
    test_inputs, test_outputs = next(iter(list(test_loaders.values())[0]))
    test_inputs = test_inputs[:4].to(device)
    test_outputs = test_outputs[:4].to(device)
    
    predictions = {}
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(test_inputs)
            predictions[model_name] = pred.cpu()
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for i in range(4):
        # 输入
        im1 = axes[i, 0].imshow(test_inputs[i, 0].cpu().numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        # 真实输出
        im2 = axes[i, 1].imshow(test_outputs[i, 0].cpu().numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # FNO预测
        fno_pred = predictions['FNO'][i, 0]
        im3 = axes[i, 2].imshow(fno_pred.numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 2].set_title(f'FNO {i+1}')
        axes[i, 2].axis('off')
        
        # AdaptiveFNO预测
        adaptive_pred = predictions['AdaptiveFNO'][i, 0]
        im4 = axes[i, 3].imshow(adaptive_pred.numpy(), cmap='RdBu_r', vmin=-2, vmax=2)
        axes[i, 3].set_title(f'AdaptiveFNO {i+1}')
        axes[i, 3].axis('off')
        
        # 误差对比
        fno_error = torch.abs(fno_pred - test_outputs[i, 0].cpu())
        adaptive_error = torch.abs(adaptive_pred - test_outputs[i, 0].cpu())
        error_diff = fno_error - adaptive_error
        
        im5 = axes[i, 4].imshow(error_diff.numpy(), cmap='RdBu', vmin=-0.2, vmax=0.2)
        axes[i, 4].set_title(f'Error Diff {i+1}\n(Blue: AdaptiveFNO Better)')
        axes[i, 4].axis('off')
    
    plt.suptitle('Navier-Stokes: Prediction Comparison (128×128)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'figures' / 'prediction_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


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
    
    return {
        'l2_improvement': l2_improvement,
        'h1_improvement': h1_improvement,
        'param_increase': param_increase
    }


def save_results_summary(training_results, eval_results, summary_stats, config, results_dir):
    """保存实验结果总结"""
    
    summary_lines = []
    summary_lines.append("NAVIER-STOKES FNO vs AdaptiveFNO (FnoMoE) COMPARISON")
    summary_lines.append("="*60)
    summary_lines.append("")
    
    # 实验配置
    summary_lines.append("Experiment Configuration:")
    summary_lines.append(f"  Dataset: Navier-Stokes (128×128)")
    summary_lines.append(f"  Training samples: {config.data.n_train}")
    summary_lines.append(f"  Test samples: {config.data.n_tests[0]}")
    summary_lines.append(f"  Epochs: {config.opt.n_epochs}")
    summary_lines.append(f"  Learning rate: {config.opt.learning_rate}")
    summary_lines.append(f"  Batch size: {config.data.batch_size}")
    summary_lines.append("")
    
    # 模型配置
    summary_lines.append("Model Configurations:")
    summary_lines.append("  FNO:")
    summary_lines.append(f"    Modes: {config.fno_model.n_modes}")
    summary_lines.append(f"    Hidden channels: {config.fno_model.hidden_channels}")
    summary_lines.append(f"    Layers: {config.fno_model.n_layers}")
    summary_lines.append("  AdaptiveFNO:")
    summary_lines.append(f"    Modes: {config.adaptive_fno_model.n_modes}")
    summary_lines.append(f"    Hidden channels: {config.adaptive_fno_model.hidden_channels}")
    summary_lines.append(f"    Layers: {config.adaptive_fno_model.n_layers}")
    summary_lines.append(f"    Experts: {config.adaptive_fno_model.n_experts}")
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
    
    # 保存到文件
    summary_path = results_dir / 'experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n✓ Results summary saved to {summary_path}")


def main():
    """主实验函数"""
    print("🌊 Navier-Stokes: FNO vs AdaptiveFNO (FnoMoE) Comparison")
    print("="*60)
    
    # 读取配置
    config = make_config_from_cli(Default)
    config = config.to_dict()
    
    # 设置分布式环境
    device, is_logger = setup(config)
    
    # 创建实验文件夹
    results_dir = create_experiment_folder()
    
    # 加载数据
    print(f"\n📊 Loading Navier-Stokes data...")
    data_dir = Path(config.data.folder).expanduser()
    
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=config.data.train_resolution,
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        test_resolutions=config.data.test_resolutions,
        n_tests=config.data.n_tests,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
    )
    
    data_processor = data_processor.to(device)
    
    # 处理分布式数据加载
    if config.distributed.use_distributed:
        train_db = train_loader.dataset
        train_sampler = DistributedSampler(train_db, rank=get_local_rank())
        train_loader = DataLoader(dataset=train_db,
                                  batch_size=config.data.batch_size,
                                  sampler=train_sampler)
        for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
            test_db = loader.dataset
            test_sampler = DistributedSampler(test_db, rank=get_local_rank())
            test_loaders[res] = DataLoader(dataset=test_db,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          sampler=test_sampler)
    
    print(f"✓ Data loaded: {config.data.n_train} train, {config.data.n_tests[0]} test samples")
    
    # 创建模型
    print(f"\n🔧 Creating models...")
    models, param_counts = create_models(config, device)
    
    # 训练模型
    training_results = {}
    for model_name, model in models.items():
        result = train_single_model(
            model, model_name, train_loader, test_loaders, data_processor,
            config, device, results_dir, is_logger
        )
        training_results[model_name] = result
    
    # 评估模型
    eval_results = evaluate_models(models, param_counts, test_loaders, device)
    
    # 可视化结果
    if is_logger:
        visualize_results(training_results, eval_results, test_loaders, device, results_dir)
    
    # 创建结果表格
    summary_stats = create_results_table(training_results, eval_results)
    
    # 保存结果总结
    if is_logger:
        save_results_summary(training_results, eval_results, summary_stats, config, results_dir)
    
    print(f"\n🎉 Experiment completed!")
    print(f"📁 Results saved in: {results_dir}")
    
    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
