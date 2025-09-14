"""
展示实验文件夹结构的脚本
"""

import os
from pathlib import Path

def show_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """递归显示目录树结构"""
    if current_depth > max_depth:
        return
    
    path = Path(path)
    if not path.exists():
        return
    
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            show_directory_tree(item, prefix + extension, max_depth, current_depth + 1)


def get_file_info(filepath):
    """获取文件信息"""
    try:
        size = os.path.getsize(filepath)
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size/(1024*1024):.1f}MB"
        return size_str
    except:
        return "N/A"


def main():
    """主函数"""
    print("🗂️  自适应FNO实验文件夹结构")
    print("=" * 60)
    
    experiments_dir = "/root/autodl-tmp/neuraloperator/experiments"
    
    if not os.path.exists(experiments_dir):
        print("❌ 实验文件夹不存在")
        return
    
    print(f"\n📁 根目录: {experiments_dir}")
    print("-" * 60)
    show_directory_tree(experiments_dir)
    
    # 统计信息
    print(f"\n📊 统计信息")
    print("-" * 60)
    
    total_files = 0
    total_size = 0
    file_types = {}
    
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            filepath = os.path.join(root, file)
            file_size = os.path.getsize(filepath)
            total_files += 1
            total_size += file_size
            
            ext = os.path.splitext(file)[1].lower()
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            file_types[ext]['count'] += 1
            file_types[ext]['size'] += file_size
    
    print(f"总文件数: {total_files}")
    print(f"总大小: {total_size/(1024*1024):.2f} MB")
    
    print(f"\n📋 文件类型分布:")
    for ext, info in sorted(file_types.items()):
        if ext == '':
            ext = '(无扩展名)'
        size_mb = info['size'] / (1024 * 1024)
        print(f"  {ext:<10}: {info['count']:>3} 个文件, {size_mb:>6.2f} MB")
    
    # 实验详情
    print(f"\n🧪 实验详情")
    print("-" * 60)
    
    experiment_folders = [d for d in os.listdir(experiments_dir) 
                         if os.path.isdir(os.path.join(experiments_dir, d)) and not d.startswith('.')]
    
    for exp_folder in sorted(experiment_folders):
        exp_path = os.path.join(experiments_dir, exp_folder)
        print(f"\n📂 {exp_folder}:")
        
        # 检查子文件夹
        subfolders = ['figures', 'models', 'data']
        for subfolder in subfolders:
            subfolder_path = os.path.join(exp_path, subfolder)
            if os.path.exists(subfolder_path):
                files = os.listdir(subfolder_path)
                if files:
                    print(f"  📁 {subfolder}/: {len(files)} 个文件")
                    for file in sorted(files)[:3]:  # 只显示前3个文件
                        file_path = os.path.join(subfolder_path, file)
                        size = get_file_info(file_path)
                        print(f"    • {file} ({size})")
                    if len(files) > 3:
                        print(f"    ... 还有 {len(files) - 3} 个文件")
                else:
                    print(f"  📁 {subfolder}/: 空文件夹")
        
        # 检查根目录下的文件
        root_files = [f for f in os.listdir(exp_path) 
                     if os.path.isfile(os.path.join(exp_path, f))]
        if root_files:
            print(f"  📄 根目录文件:")
            for file in sorted(root_files):
                file_path = os.path.join(exp_path, file)
                size = get_file_info(file_path)
                print(f"    • {file} ({size})")
    
    # 使用建议
    print(f"\n💡 使用建议")
    print("-" * 60)
    print("1. 每个PDE实验都有独立的文件夹")
    print("2. 图片保存在 figures/ 子文件夹中")
    print("3. 模型保存在 models/ 子文件夹中")
    print("4. 实验配置和总结保存在根目录")
    print("5. 使用 experiment_template.py 创建新实验")
    
    print(f"\n🚀 快速开始新实验:")
    print("cp examples/experiment_template.py examples/your_pde_experiment.py")
    print("# 修改PDE类型和参数")
    print("python examples/your_pde_experiment.py")


if __name__ == "__main__":
    main()


