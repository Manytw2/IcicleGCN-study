"""
自动化运行 .mat 文件的完整流程脚本
支持训练、评估和结果分析
"""
import os
import subprocess
import yaml
import time
import argparse
import pandas as pd
from datetime import datetime

# 可用的数据集列表
AVAILABLE_DATASETS = [
    "test_small",  # 测试数据集
    "control_uni_original",
    "control_uni_gamma_01", "control_uni_gamma_05", "control_uni_gamma_10", "control_uni_gamma_20",
    "control_uni_rayleigh_01", "control_uni_rayleigh_05", "control_uni_rayleigh_10", "control_uni_rayleigh_20",
    "control_uni_gaussian_01", "control_uni_gaussian_05", "control_uni_gaussian_10", "control_uni_gaussian_20",
    "control_uni_uniform_01", "control_uni_uniform_05", "control_uni_uniform_10", "control_uni_uniform_20",
]

# 训练参数
TRAINING_PARAMS = {
    "batch_size": 32,  # 正常batch size
    "epochs": 10,      # 减少训练轮数用于测试
    "learning_rate": 0.001,
    "start_epoch": 0,
    "reload": False
}

# 测试数据集的特殊参数
TEST_PARAMS = {
    "batch_size": 20,   # 测试集只有100个样本，batch size 不能太大
    "epochs": 5,        # 测试集训练轮数更少
}

def get_base_config_path(dataset_name):
    """获取基础配置文件路径"""
    if dataset_name.startswith("control_uni") or dataset_name == "test_small":
        # 使用control_uni的配置文件
        return "config/config_control_uni.yaml"
    else:
        # 使用默认配置文件
        return "config/config.yaml"

def create_run_specific_config(dataset_name, run_id=None):
    """创建运行特定的配置文件（仅在需要保存特定运行配置时使用）"""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 简化配置文件存储：只在需要时创建一个运行配置
    config_dir = "experiments_output/run_configs"
    config_filename = f"{dataset_name}_{run_id}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    
    # 读取基础配置
    base_config_path = get_base_config_path(dataset_name)
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新运行特定的参数
    config.update({
        "dataset": dataset_name,
        "model_path": f"./save/{dataset_name}",
        "batch_size": TRAINING_PARAMS["batch_size"],
        "epochs": TRAINING_PARAMS["epochs"],
        "learning_rate": TRAINING_PARAMS["learning_rate"],
        "start_epoch": TRAINING_PARAMS["start_epoch"],
        "reload": TRAINING_PARAMS["reload"],
        "run_id": run_id,
        "run_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # 仅在需要时创建目录和文件
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path, run_id

def check_dataset_exists(dataset_name):
    """检查数据集文件是否存在"""
    if dataset_name == "test_small":
        mat_file = f"datasets/test/{dataset_name}.mat"
    elif dataset_name == "control_uni_original":
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

def run_training(dataset_name, save_run_config=False):
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
    
    # 检查数据集
    if not check_dataset_exists(dataset_name):
        return False, None
    
    # 获取基础配置文件
    base_config_path = get_base_config_path(dataset_name)
    
    # 可选：保存运行特定的配置（用于记录和复现）
    run_id = None
    if save_run_config:
        config_path, run_id = create_run_specific_config(dataset_name)
        print(f"[INFO] 保存运行配置: {config_path}")
    
    start_time = time.time()
    
    # 根据数据集选择参数
    if dataset_name == "test_small":
        params = TEST_PARAMS.copy()  # 创建副本，避免修改原字典
        params['learning_rate'] = TRAINING_PARAMS['learning_rate']
        params['start_epoch'] = TRAINING_PARAMS['start_epoch']
        params['reload'] = TRAINING_PARAMS['reload']
    else:
        params = TRAINING_PARAMS
    
    try:
        # 使用基础配置文件 + 命令行参数覆盖
        cmd = [
            'python', 'train.py',
            '--config', base_config_path,
            '--dataset', dataset_name,
            '--model_path', f'./save/{dataset_name}',
            '--batch_size', str(params.get('batch_size', TRAINING_PARAMS['batch_size'])),
            '--epochs', str(params.get('epochs', TRAINING_PARAMS['epochs'])),
            '--learning_rate', str(params.get('learning_rate', TRAINING_PARAMS['learning_rate'])),
            '--start_epoch', str(params.get('start_epoch', TRAINING_PARAMS['start_epoch']))
        ]
        
        # 只在需要 reload 时添加参数
        if params.get('reload', TRAINING_PARAMS['reload']):
            cmd.extend(['--reload', 'True'])
        
        print(f"[INFO] 执行命令: {' '.join(cmd)}")
        print(f"[INFO] 基础配置文件: {base_config_path}")
        print(f"[INFO] 参数覆盖: dataset={dataset_name}, model_path=./save/{dataset_name}")
        print(f"[INFO] 开始训练，将显示实时进度...")
        print("="*80)
        
        # 启动训练进程，实时显示输出
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True, 
                                 bufsize=1, 
                                 universal_newlines=True)
        
        # 实时显示训练进度
        current_epoch = 0
        total_epochs = params.get('epochs', TRAINING_PARAMS["epochs"])
        last_progress_time = time.time()
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_line = output.strip()
                
                # 检查是否包含epoch信息
                epoch_found = False
                if 'Epoch' in output_line and ('/' in output_line or '[' in output_line):
                    try:
                        # 支持多种epoch格式
                        if 'Epoch [' in output_line:
                            # 格式: Epoch [1/200]
                            epoch_part = output_line.split('Epoch [')[1].split(']')[0]
                            if '/' in epoch_part:
                                current_epoch = int(epoch_part.split('/')[0])
                                epoch_found = True
                        elif 'Epoch' in output_line and '/' in output_line:
                            # 格式: Epoch 1/200 或 Epoch: 1/200
                            parts = output_line.split()
                            for part in parts:
                                if '/' in part and part.replace('/', '').replace(':', '').isdigit():
                                    current_epoch = int(part.split('/')[0])
                                    epoch_found = True
                                    break
                        
                        # 显示进度条（每0.5秒更新一次，避免刷屏）
                        if epoch_found and current_epoch > 0:
                            current_time = time.time()
                            if current_time - last_progress_time >= 0.5:
                                progress = current_epoch / total_epochs
                                bar_length = 30
                                filled_length = int(bar_length * progress)
                                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                                print(f"\r[PROGRESS] {current_epoch}/{total_epochs} [{bar}] {progress:.1%}", end='', flush=True)
                                last_progress_time = current_time
                    except:
                        pass
                
                # 显示其他重要信息
                if not epoch_found and any(keyword in output_line.lower() for keyword in 
                    ['loss', 'acc', 'error', 'warning', 'success', 'loading', 'saving', 'checkpoint']):
                    print(f"\n[OUTPUT] {output_line}")
        
        # 等待进程完成
        return_code = process.wait()
        
        # 显示最终进度
        if current_epoch > 0:
            progress = current_epoch / total_epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r[PROGRESS] {current_epoch}/{total_epochs} [{bar}] {progress:.1%}")
        
        print(f"\n[INFO] 训练进程结束，返回码: {return_code}")
        
        if return_code == 0:
            result = subprocess.CompletedProcess(cmd, 0, "", "")
        else:
            result = subprocess.CompletedProcess(cmd, return_code, "", "")
        
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
            
            return True, run_id
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

def run_evaluation(dataset_name, run_id=None):
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
    
    # 获取基础配置文件
    base_config_path = get_base_config_path(dataset_name)
    
    # 根据数据集选择参数
    if dataset_name == "test_small":
        final_epoch = TEST_PARAMS['epochs']
    else:
        final_epoch = TRAINING_PARAMS['epochs']
    
    try:
        # 使用基础配置文件 + 命令行参数覆盖
        cmd = [
            'python', 'cluster.py',
            '--config', base_config_path,
            '--dataset', dataset_name,
            '--model_path', f'./save/{dataset_name}',
            '--start_epoch', str(final_epoch)  # 评估时使用最终的epoch
        ]
        
        print(f"[INFO] 执行命令: {' '.join(cmd)}")
        print(f"[INFO] 基础配置文件: {base_config_path}")
        print(f"[INFO] 参数覆盖: dataset={dataset_name}, start_epoch={final_epoch}")
        print(f"[INFO] 开始评估...")
        
        # 启动评估进程，实时显示输出
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 text=True, 
                                 bufsize=1, 
                                 universal_newlines=True)
        
        # 实时显示评估进度
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # 显示重要信息
                if any(keyword in output.lower() for keyword in ['loading', 'extracting', 'computing', 'acc', 'nmi', 'ari', 'f1', 'error', 'success']):
                    print(f"[OUTPUT] {output.strip()}")
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            result = subprocess.CompletedProcess(cmd, 0, '\n'.join(output_lines), "")
        else:
            result = subprocess.CompletedProcess(cmd, return_code, '\n'.join(output_lines), "")
        
        if result.returncode == 0:
            print(f"[SUCCESS] 第一阶段评估完成: {dataset_name}")
            
            # 提取评估结果
            output_lines = result.stdout.split('\n')
            metrics = {}
            for line in output_lines:
                # 查找包含所有指标的行
                if 'NMI =' in line and 'ARI =' in line and 'F =' in line and 'ACC =' in line:
                    try:
                        # 解析格式: NMI = 0.5393 ARI = 0.4444 F = 0.5402 ACC = 0.6217
                        parts = line.split()
                        for i in range(0, len(parts)-1, 3):
                            if parts[i] == 'NMI' and parts[i+1] == '=':
                                metrics['NMI'] = float(parts[i+2])
                            elif parts[i] == 'ARI' and parts[i+1] == '=':
                                metrics['ARI'] = float(parts[i+2])
                            elif parts[i] == 'F' and parts[i+1] == '=':
                                metrics['F1'] = float(parts[i+2])
                            elif parts[i] == 'ACC' and parts[i+1] == '=':
                                metrics['ACC'] = float(parts[i+2])
                    except Exception as e:
                        print(f"[WARNING] 解析指标时出错: {e}")
                        print(f"[DEBUG] 原始行: {line}")
                    break  # 找到结果行后就停止
            
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
        # 保存为Excel格式
        excel_filename = f"{results_dir}/{dataset_name}_{timestamp}.xlsx"
        
        # 创建DataFrame
        data = {
            'Dataset': [dataset_name],
            'Experiment_Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Experiment_Type': ['单个数据集实验'],
            'Workflow': ['第一阶段训练 -> 第一阶段评估'],
            'Network_Architecture': ['FeatureVectorEncoder'],
            'Training_Epochs': [TRAINING_PARAMS['epochs']],
            'Batch_Size': [TRAINING_PARAMS['batch_size']],
            'Learning_Rate': [TRAINING_PARAMS['learning_rate']],
            'ACC': [metrics.get('ACC', 0) if metrics else 0],
            'NMI': [metrics.get('NMI', 0) if metrics else 0],
            'ARI': [metrics.get('ARI', 0) if metrics else 0],
            'F1': [metrics.get('F1', 0) if metrics else 0]
        }
        
        df = pd.DataFrame(data)
        
        # 保存为Excel文件
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # 获取工作表以调整列宽
            worksheet = writer.sheets['Results']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"[SAVE] 结果已保存到: {excel_filename}")
        return excel_filename
    
    return None

def run_single_experiment_with_metrics(dataset_name, save_run_config=False):
    """运行单个数据集实验并返回详细指标"""
    print(f"[EXPERIMENT] 运行单个实验: {dataset_name}")
    print(f"[INFO] 实验流程: 第一阶段训练 -> 第一阶段评估")
    
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"[ERROR] Unknown dataset: {dataset_name}")
        print(f"[INFO] Available datasets: {', '.join(AVAILABLE_DATASETS)}")
        log_error_to_file(dataset_name, "Invalid Dataset", f"未知数据集: {dataset_name}", "实验初始化")
        return False, None
    
    # 运行第一阶段训练
    training_success, run_id = run_training(dataset_name, save_run_config)
    
    if training_success:
        # 运行第一阶段评估
        metrics = run_evaluation(dataset_name, run_id)
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
            
            return True, metrics
        else:
            print(f"\n[ERROR] 第一阶段评估失败: {dataset_name}")
            print(f"[TERMINATE] 实验终止: 评估阶段失败")
            return False, None
    else:
        print(f"\n[ERROR] 第一阶段训练失败: {dataset_name}")
        print(f"[TERMINATE] 实验终止: 训练阶段失败")
        return False, None

def save_batch_results_to_excel(datasets_list, detailed_results, total_time, successful):
    """保存批量实验结果到Excel文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_dir = "experiments_output"
    results_dir = f"{experiments_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建Excel文件名
    excel_filename = f"{results_dir}/batch_experiment_{timestamp}.xlsx"
    
    # 准备数据
    batch_data = []
    for dataset in datasets_list:
        if dataset in detailed_results and detailed_results[dataset]['success']:
            # 成功的数据集，使用实际指标
            metrics = detailed_results[dataset]['metrics']
            batch_data.append({
                'Dataset': dataset,
                'Status': 'Success',
                'ACC': f"{metrics.get('ACC', 0):.4f}" if metrics else 'N/A',
                'NMI': f"{metrics.get('NMI', 0):.4f}" if metrics else 'N/A',
                'ARI': f"{metrics.get('ARI', 0):.4f}" if metrics else 'N/A',
                'F1': f"{metrics.get('F1', 0):.4f}" if metrics else 'N/A',
                'Experiment_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Workflow': '第一阶段训练 -> 第一阶段评估',
                'Network_Architecture': 'FeatureVectorEncoder',
                'Training_Epochs': TRAINING_PARAMS['epochs'],
                'Batch_Size': TRAINING_PARAMS['batch_size'],
                'Learning_Rate': TRAINING_PARAMS['learning_rate']
            })
        else:
            # 失败的数据集
            batch_data.append({
                'Dataset': dataset,
                'Status': 'Failed',
                'ACC': 'N/A',
                'NMI': 'N/A',
                'ARI': 'N/A',
                'F1': 'N/A',
                'Experiment_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Workflow': '第一阶段训练 -> 第一阶段评估',
                'Network_Architecture': 'FeatureVectorEncoder',
                'Training_Epochs': TRAINING_PARAMS['epochs'],
                'Batch_Size': TRAINING_PARAMS['batch_size'],
                'Learning_Rate': TRAINING_PARAMS['learning_rate']
            })
    
    # 创建DataFrame
    df = pd.DataFrame(batch_data)
    
    # 创建汇总信息
    summary_data = {
        'Metric': ['Total_Datasets', 'Successful_Datasets', 'Failed_Datasets', 'Success_Rate', 'Total_Time_Minutes'],
        'Value': [
            len(datasets_list),
            successful,
            len(datasets_list) - successful,
            f"{successful/len(datasets_list)*100:.1f}%" if len(datasets_list) > 0 else "0%",
            f"{total_time/60:.1f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # 保存为Excel文件
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 保存详细结果
        df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # 保存汇总信息
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 调整列宽
        for sheet_name in ['Detailed_Results', 'Summary']:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"[SAVE] 批量结果已保存到: {excel_filename}")
    return excel_filename

def run_single_experiment_with_prompt(dataset_name, save_run_config=False):
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
                return run_single_experiment_with_prompt(new_dataset, save_run_config)
            else:
                print(f"[ERROR] 数据集 {new_dataset} 仍然不存在")
                return False
    
    # 运行第一阶段训练
    training_success, run_id = run_training(dataset_name, save_run_config)
    
    if training_success:
        # 运行第一阶段评估
        metrics = run_evaluation(dataset_name, run_id)
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
                metrics = run_evaluation(dataset_name, run_id)
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
            training_success, run_id = run_training(dataset_name, save_run_config)
            if training_success:
                metrics = run_evaluation(dataset_name, run_id)
                if metrics:
                    save_results_to_file(dataset_name, metrics, "single")
                    return True
            return False
        
        return False

def run_single_experiment(dataset_name, save_run_config=False):
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
    training_success, run_id = run_training(dataset_name, save_run_config)
    
    if training_success:
        # 运行第一阶段评估
        metrics = run_evaluation(dataset_name, run_id)
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
    detailed_results = {}
    total_start_time = time.time()
    
    for i, dataset in enumerate(datasets_list, 1):
        print(f"\n[PROGRESS] 进度: {i}/{len(datasets_list)} - {dataset}")
        
        # 运行单个实验并收集详细指标
        success, metrics = run_single_experiment_with_metrics(dataset)
        results[dataset] = success
        detailed_results[dataset] = {
            'success': success,
            'metrics': metrics
        }
        
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
    
    # 使用Excel格式保存批量实验结果
    excel_filename = save_batch_results_to_excel(datasets_list, detailed_results, total_time, successful)
    
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
    parser.add_argument('--save-config', action='store_true', help='保存运行配置用于记录和复现')
    
    args = parser.parse_args()
    
    if args.list:
        print("[LIST] 可用数据集:")
        for dataset in AVAILABLE_DATASETS:
            print(f"  - {dataset}")
        return
    
    if args.dataset:
        if args.interactive:
            run_single_experiment_with_prompt(args.dataset, args.save_config)
        else:
            run_single_experiment(args.dataset, args.save_config)
    elif args.batch:
        run_batch_experiments(args.batch)
    elif args.comparison:
        run_noise_comparison()
    else:
        print("[HELP] 请选择运行模式:")
        print("  python run_mat_experiments.py --dataset control_uni_gamma_01          # 运行单个实验")
        print("  python run_mat_experiments.py --dataset control_uni_gamma_01 --interactive  # 运行单个实验（交互模式）")
        print("  python run_mat_experiments.py --dataset control_uni_gamma_01 --save-config  # 运行实验并保存配置")
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
