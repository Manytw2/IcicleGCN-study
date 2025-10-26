#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAT文件转换工具
将 .mat 文件转换为 IcicleGCN 等机器学习算法需要的 .txt 格式

功能特点:
- 支持批量转换指定目录下的所有 .mat 文件
- 自动提取特征矩阵 (X) 和标签向量 (Y)
- 生成标准格式的特征文件和标签文件
- 支持自定义输出目录
- 提供详细的转换统计和结果报告

使用场景:
- 机器学习数据集预处理
- 算法鲁棒性测试数据准备
- 不同数据格式之间的转换
- 批量数据处理任务
"""

import os
import scipy.io
import numpy as np
import argparse
from pathlib import Path

def convert_mat_to_txt(mat_file_path, output_dir="data"):
    """
    将单个 .mat 文件转换为标准 .txt 格式
    
    自动提取 .mat 文件中的特征矩阵 (X) 和标签向量 (Y)，
    并保存为机器学习算法常用的文本格式。
    
    Args:
        mat_file_path (str): .mat 文件路径
        output_dir (str): 输出目录，默认为 "data"
        
    Returns:
        tuple: (成功标志, 数据形状, 类别数量)
    """
    try:
        # 加载 .mat 文件
        print(f"正在处理: {mat_file_path}")
        data = scipy.io.loadmat(mat_file_path)
        
        # 提取数据
        X = data['X']  # 特征矩阵 (samples, features)
        Y = data['Y'].flatten()  # 标签向量 (samples,)
        
        # 获取文件名（不含扩展名）
        file_name = Path(mat_file_path).stem
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征数据
        features_file = os.path.join(output_dir, f"{file_name}.txt")
        np.savetxt(features_file, X, fmt='%.6f', delimiter=' ')
        print(f"  [OK] 特征数据已保存: {features_file} (形状: {X.shape})")
        
        # 保存标签数据
        labels_file = os.path.join(output_dir, f"{file_name}_label.txt")
        np.savetxt(labels_file, Y, fmt='%d', delimiter=' ')
        print(f"  [OK] 标签数据已保存: {labels_file} (形状: {Y.shape})")
        
        # 打印数据集信息
        unique_labels = np.unique(Y)
        print(f"  [OK] 数据集信息: {X.shape[0]} 样本, {X.shape[1]} 特征, {len(unique_labels)} 个类别")
        print(f"  [OK] 类别标签: {unique_labels}")
        
        return True, X.shape, len(unique_labels)
        
    except Exception as e:
        print(f"  [ERROR] 转换失败: {e}")
        return False, None, None

def main():
    """
    主函数：批量转换指定目录下的所有 .mat 文件
    
    支持命令行参数指定输入和输出目录
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MAT文件批量转换工具')
    parser.add_argument('--input_dir', '-i', 
                       default='datasets/noisy_datasets',
                       help='输入目录路径 (默认: datasets/noisy_datasets)')
    parser.add_argument('--output_dir', '-o',
                       default='data', 
                       help='输出目录路径 (默认: data)')
    parser.add_argument('--recursive', '-r',
                       action='store_true',
                       help='递归搜索子目录中的 .mat 文件')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAT文件批量转换工具")
    print("=" * 60)
    
    # 使用命令行参数或默认值
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 获取所有 .mat 文件
    mat_files = []
    if args.recursive:
        # 递归搜索
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
    else:
        # 仅搜索当前目录
        for file in os.listdir(input_dir):
            if file.endswith('.mat'):
                mat_files.append(os.path.join(input_dir, file))
    
    if not mat_files:
        print(f"错误: 在 {input_dir} 中没有找到 .mat 文件")
        return
    
    print(f"找到 {len(mat_files)} 个 .mat 文件:")
    for file in mat_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    print()
    
    # 转换统计
    success_count = 0
    failed_count = 0
    conversion_summary = []
    
    # 批量转换
    for i, mat_file in enumerate(mat_files, 1):
        print(f"[{i}/{len(mat_files)}] 转换 {os.path.basename(mat_file)}")
        success, shape, num_classes = convert_mat_to_txt(mat_file, output_dir)
        
        if success:
            success_count += 1
            conversion_summary.append({
                'file': os.path.basename(mat_file),
                'shape': shape,
                'classes': num_classes,
                'status': '成功'
            })
        else:
            failed_count += 1
            conversion_summary.append({
                'file': os.path.basename(mat_file),
                'shape': None,
                'classes': None,
                'status': '失败'
            })
        
        print()
    
    # 输出转换总结
    print("=" * 60)
    print("转换完成总结")
    print("=" * 60)
    print(f"总文件数: {len(mat_files)}")
    print(f"成功转换: {success_count}")
    print(f"转换失败: {failed_count}")
    print()
    
    # 详细结果表格
    print("详细结果:")
    print("-" * 80)
    print(f"{'文件名':<30} {'状态':<8} {'样本数':<8} {'特征数':<8} {'类别数':<8}")
    print("-" * 80)
    
    for item in conversion_summary:
        if item['status'] == '成功':
            print(f"{item['file']:<30} {item['status']:<8} {item['shape'][0]:<8} {item['shape'][1]:<8} {item['classes']:<8}")
        else:
            print(f"{item['file']:<30} {item['status']:<8} {'-':<8} {'-':<8} {'-':<8}")
    
    print("-" * 80)
    
    # 检查输出文件
    print("\n检查输出文件:")
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    print(f"在 {output_dir} 目录中生成了 {len(output_files)} 个 .txt 文件:")
    for file in sorted(output_files):
        print(f"  - {file}")
    
    print("\n转换完成！现在可以使用这些数据集运行机器学习算法了。")
    print("\n使用示例:")
    print("# IcicleGCN 聚类算法:")
    print("python icicleGCN.py --name control_uni_gamma_01 --n_clusters 6 --n_z 6 --n_input 60")
    print("\n# 其他机器学习算法:")
    print("# 直接加载 .txt 文件进行训练和测试")
    print("\n# 工具使用说明:")
    print("python convert_mat_to_txt.py --help  # 查看帮助")
    print("python convert_mat_to_txt.py -i /path/to/mat/files -o /path/to/output  # 自定义路径")
    print("python convert_mat_to_txt.py -r  # 递归搜索子目录")

if __name__ == "__main__":
    main()
