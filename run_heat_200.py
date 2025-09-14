#!/usr/bin/env python3
"""
快速运行Heat Equation 200 epochs实验
"""

import sys
import os

# 添加路径
sys.path.append('/root/autodl-tmp/neuraloperator')

# 导入主函数
from examples.heat_equation_100epochs import main

if __name__ == "__main__":
    print("🔥 Running Heat Equation 200 Epochs Experiment")
    main(n_epochs=200)
