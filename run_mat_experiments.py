"""
自动化运行 .mat 文件的完整流程脚本
支持训练、评估和结果分析
"""
import os
import subprocess
import yaml
import time
import argparse
from datetime import datetime

# 可用的数据集列表
AVAILABLE_DATASETS = [
    "control_uni_original",
    "control_uni_gamma_01", "control_uni_gamma_05", "control_uni_gamma_10", "control_uni_gamma_20",
    "control_uni_rayleigh_01", "control_uni_rayleigh_05", "control_uni_rayleigh_10", "control_uni_rayleigh_20",
    "control_uni_gaussian_01", "control_uni_gaussian_05", "control_uni_gaussian_10", "control_uni_gaussian_20",
    "control_uni_uniform_01", "control_uni_uniform_05", "control_uni_uniform_10", "control_uni_uniform_20",
]

# 训练参数
TRAINING_PARAMS = {
    "batch_size": 32,
    "epochs": 200,
    "learning_rate": 0.001,
    "start_epoch": 0,
    "reload": False
}

def create_config_file(dataset_name):
    """创建专用的配置文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = f"experiments_output/configs/config_{dataset_name}_{timestamp}.yaml"
    
    config = {
        "seed": 42,
        "workers": 4,
        "dataset_dir": "./datasets",
        "batch_size": TRAINING_PARAMS["batch_size"],
        "image_size": 224,
        "start_epoch": TRAINING_PARAMS["start_epoch"],
        "epochs": TRAINING_PARAMS["epochs"],
        "dataset": dataset_name,
        "resnet": "ResNet34",
        "feature_dim": 128,
        "model_path": f"./save/{dataset_name}",
        "reload": TRAINING_PARAMS["reload"],
        "learning_rate": TRAINING_PARAMS["learning_rate"],
        "weight_decay": 0.0001,
        "instance_temperature": 0.5,
        "cluster_temperature": 1.0
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[INFO] Created config file: {config_path}")
    print(f"[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Model path: {config['model_path']}")
    
    return config_path

def check_dataset_exists(dataset_name):
    """检查数据集文件是否存在"""
    if dataset_name == "control_uni_original":
        mat_file = f"datasets/CMHDC_noisy_datasets/{dataset_name}.mat"
    else:
        mat_file = f"datasets/CMHDC_noisy_datasets/{dataset_name}.mat"
    
    if os.path.exists(mat_file):
        print(f"[SUCCESS] Dataset file found: {mat_file}")
        return True
    else:
        print(f"[ERROR] Dataset file not found: {mat_file}")
        return False

def log_error_to_file(dataset_name, error_type, error_message, stage="unknown"):
    """保存错误信息到日志文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验输出目录
    experiments_dir = "experiments_output"
    error_dir = f"{experiments_dir}/error_logs"
    os.makedirs(error_dir, exist_ok=True)
    
    # 保存错误日志
    error_filename = f"{error_dir}/error_{dataset_name}_{timestamp}.txt"
    with open(error_filename, 'w', encoding='utf-8') as f:
        f.write(f"错误日志报告\n")
        f.write("="*50 + "\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"错误阶段: {stage}\n")
        f.write(f"错误类型: {error_type}\n")
        f.write(f"错误信息: {error_message}\n")
    
    print(f"[ERROR_LOG] 错误日志已保存到: {error_filename}")
    return error_filename

def run_training(dataset_name):
    """运行第一阶段训练"""
    print("\n" + "="*80)
    print(f"[STAGE 1] 第一阶段训练: {dataset_name}")
    print("="*80)
    print("[INFO] 训练内容:")
    print("  - 对比学习预训练 (Contrastive Learning)")
    print("  - 特征向量数据增强 (Gaussian noise + Dropout)")
    print("  - FeatureVectorEncoder 网络训练")
    print("  - 聚类损失 + 重构损失")
    print("="*80)
    
    # 创建专用配置文件
    config_path = create_config_file(dataset_name)
    
    # 检查数据集
    if not check_dataset_exists(dataset_name):
        return False
    
    start_time = time.time()
    
    try:
        # 运行第一阶段训练，传递配置文件路径
        print(f"[INFO] 执行命令: python train.py --config {config_path}")
        print(f"[INFO] 配置文件: {config_path}")
        result = subprocess.run(['python', 'train.py', '--config', config_path], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            training_time = time.time() - start_time
            print(f"[SUCCESS] 第一阶段训练完成: {dataset_name}")
            print(f"[INFO] 训练时间: {training_time:.1f} 秒")
            print(f"[INFO] 模型保存路径: ./save/{dataset_name}/")
            
            # 显示训练输出中的关键信息
            output_lines = result.stdout.split('\n')
            print(f"[INFO] 训练过程关键信息:")
            for line in output_lines:
                if any(keyword in line for keyword in ['Detected', 'Using', 'Initialized', 'Epoch']):
                    print(f"  {line.strip()}")
            
            return True, config_path
        else:
            print(f"[ERROR] Training failed: {dataset_name}")
            print(f"[ERROR] Error output: {result.stderr}")
            # 保存错误日志
            log_error_to_file(dataset_name, "Training Failed", result.stderr, "第一阶段训练")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Training timeout: {dataset_name}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Training Timeout", "训练超时 (3600秒)", "第一阶段训练")
        return False, None
    except Exception as e:
        print(f"[ERROR] Training exception: {dataset_name} - {e}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Training Exception", str(e), "第一阶段训练")
        return False, None

def run_evaluation(dataset_name, config_path):
    """运行第一阶段评估"""
    print("\n" + "="*80)
    print(f"[STAGE 1] 第一阶段评估: {dataset_name}")
    print("="*80)
    print("[INFO] 评估内容:")
    print("  - 加载训练好的 FeatureVectorEncoder 模型")
    print("  - 提取特征向量")
    print("  - 执行聚类算法")
    print("  - 计算 ACC, NMI, ARI, F1 指标")
    print("="*80)
    
    # 更新配置文件用于评估
    print(f"[INFO] 更新配置文件用于评估...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['start_epoch'] = TRAINING_PARAMS["epochs"]  # 使用训练好的模型
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"[INFO] 配置文件已更新: start_epoch = {TRAINING_PARAMS['epochs']}")
    
    try:
        # 运行第一阶段评估，传递配置文件路径
        print(f"[INFO] 执行命令: python cluster.py --config {config_path}")
        result = subprocess.run(['python', 'cluster.py', '--config', config_path], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"[SUCCESS] 第一阶段评估完成: {dataset_name}")
            
            # 提取评估结果
            output_lines = result.stdout.split('\n')
            metrics = {}
            for line in output_lines:
                if 'ACC:' in line:
                    try:
                        metrics['ACC'] = float(line.split('ACC:')[1].strip())
                    except:
                        pass
                elif 'NMI:' in line:
                    try:
                        metrics['NMI'] = float(line.split('NMI:')[1].strip())
                    except:
                        pass
                elif 'ARI:' in line:
                    try:
                        metrics['ARI'] = float(line.split('ARI:')[1].strip())
                    except:
                        pass
                elif 'F1:' in line:
                    try:
                        metrics['F1'] = float(line.split('F1:')[1].strip())
                    except:
                        pass
            
            # 显示结果
            if metrics:
                print(f"[RESULTS] 聚类性能指标:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"[WARNING] 未能提取到性能指标")
            
            return metrics
        else:
            print(f"[ERROR] Evaluation failed: {dataset_name}")
            print(f"[ERROR] Error output: {result.stderr}")
            # 保存错误日志
            log_error_to_file(dataset_name, "Evaluation Failed", result.stderr, "第一阶段评估")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Evaluation timeout: {dataset_name}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Evaluation Timeout", "评估超时 (300秒)", "第一阶段评估")
        return None
    except Exception as e:
        print(f"[ERROR] Evaluation exception: {dataset_name} - {e}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Evaluation Exception", str(e), "第一阶段评估")
        return None

def save_results_to_file(dataset_name, metrics, experiment_type="single"):
    """保存实验结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验输出目录
    experiments_dir = "experiments_output"
    results_dir = f"{experiments_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存单个实验结果
    if experiment_type == "single":
        filename = f"{results_dir}/{dataset_name}_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"实验结果报告\n")
            f.write("="*50 + "\n")
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验类型: 单个数据集实验\n")
            f.write(f"实验流程: 第一阶段训练 -> 第一阶段评估\n\n")
            
            f.write("训练信息:\n")
            f.write(f"  - 网络架构: FeatureVectorEncoder\n")
            f.write(f"  - 训练轮数: {TRAINING_PARAMS['epochs']}\n")
            f.write(f"  - 批次大小: {TRAINING_PARAMS['batch_size']}\n")
            f.write(f"  - 学习率: {TRAINING_PARAMS['learning_rate']}\n\n")
            
            f.write("评估结果:\n")
            if metrics:
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write("  未能提取到性能指标\n")
        
        print(f"[SAVE] 结果已保存到: {filename}")
        return filename
    
    return None

def run_single_experiment_with_prompt(dataset_name):
    """运行单个数据集实验，失败时询问用户"""
    print(f"[EXPERIMENT] 运行单个实验: {dataset_name}")
    print(f"[INFO] 实验流程: 第一阶段训练 -> 第一阶段评估")
    
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        print(f"[INFO] Available datasets: {', '.join(AVAILABLE_DATASETS)}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Invalid Dataset", f"未知数据集: {dataset_name}", "实验初始化")
        
        # 询问用户是否重试
        user_choice = ask_user_continue(f"数据集 {dataset_name} 不存在")
        if user_choice == 'stop':
            return False
        elif user_choice == 'skip':
            return False
        elif user_choice == 'continue':
            # 让用户重新输入数据集名称
            new_dataset = input("[INPUT] 请输入正确的数据集名称: ").strip()
            if new_dataset in AVAILABLE_DATASETS:
                return run_single_experiment_with_prompt(new_dataset)
            else:
                print(f"[ERROR] 数据集 {new_dataset} 仍然不存在")
                return False
    
    # 运行第一阶段训练
    training_success, config_path = run_training(dataset_name)
    
    if training_success:
        # 运行第一阶段评估
        metrics = run_evaluation(dataset_name, config_path)
        if metrics:
            print("\n" + "="*80)
            print(f"[COMPLETE] 完整实验完成: {dataset_name}")
            print("="*80)
            print("[SUCCESS] 第一阶段训练: 完成")
            print("[SUCCESS] 第一阶段评估: 完成")
            print(f"[RESULTS] 最终结果: ACC={metrics.get('ACC', 0):.4f}, NMI={metrics.get('NMI', 0):.4f}")
            print("="*80)
            
            # 保存结果到文件
            save_results_to_file(dataset_name, metrics, "single")
            
            return True
        else:
            print(f"\n[ERROR] 第一阶段评估失败: {dataset_name}")
            print(f"[TERMINATE] 实验终止: 评估阶段失败")
            
            # 询问用户是否重试
            user_choice = ask_user_continue(f"数据集 {dataset_name} 评估失败")
            if user_choice == 'stop':
                return False
            elif user_choice == 'skip':
                return False
            elif user_choice == 'continue':
                # 重试评估
                print(f"[RETRY] 重试评估 {dataset_name}")
                metrics = run_evaluation(dataset_name, config_path)
                if metrics:
                    save_results_to_file(dataset_name, metrics, "single")
                    return True
                else:
                    return False
            
            return False
    else:
        print(f"\n[ERROR] 第一阶段训练失败: {dataset_name}")
        print(f"[TERMINATE] 实验终止: 训练阶段失败")
        
        # 询问用户是否重试
        user_choice = ask_user_continue(f"数据集 {dataset_name} 训练失败")
        if user_choice == 'stop':
            return False
        elif user_choice == 'skip':
            return False
        elif user_choice == 'continue':
            # 重试训练
            print(f"[RETRY] 重试训练 {dataset_name}")
            training_success, config_path = run_training(dataset_name)
            if training_success:
                metrics = run_evaluation(dataset_name, config_path)
                if metrics:
                    save_results_to_file(dataset_name, metrics, "single")
                    return True
            return False
        
        return False

def run_single_experiment(dataset_name):
    """运行单个数据集的完整实验（原始版本，不询问用户）"""
    print(f"[EXPERIMENT] 运行单个实验: {dataset_name}")
    print(f"[INFO] 实验流程: 第一阶段训练 -> 第一阶段评估")
    
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        print(f"[INFO] Available datasets: {', '.join(AVAILABLE_DATASETS)}")
        # 保存错误日志
        log_error_to_file(dataset_name, "Invalid Dataset", f"未知数据集: {dataset_name}", "实验初始化")
        return False
    
    # 运行第一阶段训练
    training_success, config_path = run_training(dataset_name)
    
    if training_success:
        # 运行第一阶段评估
        metrics = run_evaluation(dataset_name, config_path)
        if metrics:
            print("\n" + "="*80)
            print(f"[COMPLETE] 完整实验完成: {dataset_name}")
            print("="*80)
            print("[SUCCESS] 第一阶段训练: 完成")
            print("[SUCCESS] 第一阶段评估: 完成")
            print(f"[RESULTS] 最终结果: ACC={metrics.get('ACC', 0):.4f}, NMI={metrics.get('NMI', 0):.4f}")
            print("="*80)
            
            # 保存结果到文件
            save_results_to_file(dataset_name, metrics, "single")
            
            return True
        else:
            print(f"\n[ERROR] 第一阶段评估失败: {dataset_name}")
            print(f"[TERMINATE] 实验终止: 评估阶段失败")
            return False
    else:
        print(f"\n[ERROR] 第一阶段训练失败: {dataset_name}")
        print(f"[TERMINATE] 实验终止: 训练阶段失败")
        return False

def ask_user_continue(error_message):
    """询问用户是否继续"""
    print(f"\n[ERROR] {error_message}")
    print("[PROMPT] 是否继续运行下一个数据集？")
    print("  [y] 是 - 继续运行")
    print("  [n] 否 - 停止所有实验")
    print("  [s] 跳过 - 跳过当前数据集，继续下一个")
    
    while True:
        try:
            choice = input("[INPUT] 请输入选择 (y/n/s): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                return 'continue'
            elif choice in ['n', 'no', '否']:
                return 'stop'
            elif choice in ['s', 'skip', '跳过']:
                return 'skip'
            else:
                print("[WARNING] 无效输入，请输入 y/n/s")
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，停止实验")
            return 'stop'
        except EOFError:
            print("\n[INFO] 输入结束，停止实验")
            return 'stop'

def run_batch_experiments(datasets_list):
    """运行批量实验"""
    print("="*80)
    print("          [BATCH] 批量噪声鲁棒性实验")
    print("="*80)
    
    results = {}
    total_start_time = time.time()
    
    for i, dataset in enumerate(datasets_list, 1):
        print(f"\n[PROGRESS] 进度: {i}/{len(datasets_list)} - {dataset}")
        
        # 运行单个实验
        success = run_single_experiment(dataset)
        results[dataset] = success
        
        # 如果实验失败，询问用户是否继续
        if not success:
            error_msg = f"数据集 {dataset} 实验失败"
            user_choice = ask_user_continue(error_msg)
            
            if user_choice == 'stop':
                print(f"[STOP] 用户选择停止，终止所有实验")
                break
            elif user_choice == 'skip':
                print(f"[SKIP] 用户选择跳过 {dataset}，继续下一个数据集")
                continue
            elif user_choice == 'continue':
                print(f"[CONTINUE] 用户选择继续，运行下一个数据集")
        
        # 小延迟
        time.sleep(2)
    
    # 打印总结
    total_time = time.time() - total_start_time
    successful = sum(1 for success in results.values() if success)
    
    print("\n" + "="*80)
    print("                   [SUMMARY] 实验总结")
    print("="*80)
    print(f"[STATS] 总数据集数: {len(datasets_list)}")
    print(f"[STATS] 成功完成: {successful}/{len(datasets_list)}")
    print(f"[STATS] 总耗时: {total_time/60:.1f} 分钟")
    
    print(f"\n[DETAILS] 详细结果:")
    for dataset, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {dataset}: {status}")
    
    # 保存批量实验结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_dir = "experiments_output"
    results_dir = f"{experiments_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    batch_filename = f"{results_dir}/batch_experiment_{timestamp}.txt"
    with open(batch_filename, 'w', encoding='utf-8') as f:
        f.write(f"批量实验结果报告\n")
        f.write("="*50 + "\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"实验类型: 批量数据集实验\n")
        f.write(f"数据集数量: {len(datasets_list)}\n")
        f.write(f"成功完成: {successful}/{len(datasets_list)}\n")
        f.write(f"总耗时: {total_time/60:.1f} 分钟\n\n")
        
        f.write("详细结果:\n")
        for dataset, success in results.items():
            status = "成功" if success else "失败"
            f.write(f"  {dataset}: {status}\n")
    
    print(f"[SAVE] 批量结果已保存到: {batch_filename}")
    
    return results

def run_noise_comparison():
    """运行噪声对比实验"""
    print("[COMPARISON] 运行噪声对比实验")
    
    # 定义对比实验的数据集
    comparison_datasets = [
        "control_uni_original",
        "control_uni_gamma_01", "control_uni_gamma_05", "control_uni_gamma_10", "control_uni_gamma_20",
        "control_uni_rayleigh_01", "control_uni_rayleigh_05", "control_uni_rayleigh_10", "control_uni_rayleigh_20",
    ]
    
    return run_batch_experiments(comparison_datasets)

def main():
    parser = argparse.ArgumentParser(description='自动化运行 .mat 文件实验')
    parser.add_argument('--dataset', type=str, help='运行单个数据集实验')
    parser.add_argument('--batch', nargs='+', help='运行批量实验，指定数据集列表')
    parser.add_argument('--comparison', action='store_true', help='运行噪声对比实验')
    parser.add_argument('--list', action='store_true', help='列出所有可用数据集')
    parser.add_argument('--interactive', action='store_true', help='启用交互模式，失败时询问用户')
    
    args = parser.parse_args()
    
    if args.list:
        print("[LIST] 可用数据集:")
        for dataset in AVAILABLE_DATASETS:
            print(f"  - {dataset}")
        return
    
    if args.dataset:
        if args.interactive:
            run_single_experiment_with_prompt(args.dataset)
        else:
            run_single_experiment(args.dataset)
    elif args.batch:
        run_batch_experiments(args.batch)
    elif args.comparison:
        run_noise_comparison()
    else:
        print("[HELP] 请选择运行模式:")
        print("  python run_mat_experiments.py --dataset control_uni_gamma_01          # 运行单个实验")
        print("  python run_mat_experiments.py --dataset control_uni_gamma_01 --interactive  # 运行单个实验（交互模式）")
        print("  python run_mat_experiments.py --batch control_uni_gamma_01 control_uni_rayleigh_01  # 运行批量实验")
        print("  python run_mat_experiments.py --comparison                          # 运行噪声对比实验")
        print("  python run_mat_experiments.py --list                                 # 列出所有数据集")

if __name__ == "__main__":
    print("="*80)
    print("          [SCRIPT] .mat 文件自动化实验脚本")
    print("="*80)
    print(f"[TIME] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DIR] 工作目录: {os.getcwd()}")
    print(f"[CONFIG] 训练参数: {TRAINING_PARAMS}")
    
    main()
