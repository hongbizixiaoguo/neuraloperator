#!/bin/bash

# Heat Equation实验运行脚本
# 支持不同的epochs数量

echo "🔥 Heat Equation Experiments with Adaptive FNO"
echo "================================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python not found!"
    exit 1
fi

# 进入正确的目录
cd /root/autodl-tmp/neuraloperator

# 函数：运行实验
run_experiment() {
    local epochs=$1
    echo ""
    echo "🚀 Running experiment with $epochs epochs..."
    echo "⏰ Start time: $(date)"
    
    python examples/heat_equation_100epochs.py --epochs $epochs
    
    if [ $? -eq 0 ]; then
        echo "✅ $epochs epochs experiment completed successfully!"
        echo "📁 Results saved to: experiments/heat_equation_${epochs}epochs/"
    else
        echo "❌ $epochs epochs experiment failed!"
        return 1
    fi
    
    echo "⏰ End time: $(date)"
    echo ""
}

# 主菜单
echo "请选择要运行的实验："
echo "1) 100 epochs (快速验证)"
echo "2) 200 epochs (中等训练)"
echo "3) 300 epochs (充分训练)"
echo "4) 自定义epochs数量"
echo "5) 运行所有实验 (100, 200, 300)"
echo "6) 退出"

read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        run_experiment 100
        ;;
    2)
        run_experiment 200
        ;;
    3)
        run_experiment 300
        ;;
    4)
        read -p "请输入epochs数量: " custom_epochs
        if [[ $custom_epochs =~ ^[0-9]+$ ]] && [ $custom_epochs -gt 0 ]; then
            run_experiment $custom_epochs
        else
            echo "❌ 无效的epochs数量！"
            exit 1
        fi
        ;;
    5)
        echo "🔄 运行所有实验..."
        run_experiment 100 && run_experiment 200 && run_experiment 300
        echo "🎉 所有实验完成！"
        ;;
    6)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选择！"
        exit 1
        ;;
esac

echo "✨ 实验完成！查看结果："
echo "📊 训练曲线: experiments/heat_equation_*epochs/figures/"
echo "🎯 预测对比: experiments/heat_equation_*epochs/figures/"
echo "🤖 模型文件: experiments/heat_equation_*epochs/models/"
echo "📝 实验总结: experiments/heat_equation_*epochs/experiment_summary.txt"
