#!/usr/bin/env python3
"""
运行Burgers方程FNO vs FNO-MoE对比实验
"""

import sys
import os

# 添加路径
sys.path.append('/root/autodl-tmp/neuraloperator')

# 导入主函数
from examples.burgers_resolution_simple import main

if __name__ == "__main__":
    print("🚀 Running Burgers Equation FNO vs FNO-MoE Comparison")
    main()
