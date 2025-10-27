# 基于对比学习和多尺度图卷积网络的深度图像聚类 (IcicleGCN)

这是我们Pattern Recognition'24论文"Deep image clustering with contrastive learning and multi-scale graph convolutional networks"的代码

<div align=center><img src="Figures/Figure1.png" width = "70%"/></div>

# 依赖环境

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.8.1
- munkres>=1.1.4
- numpy>=1.19.2
- opencv-python>=4.4.0.46
- pyyaml>=5.3.1
- scikit-learn>=0.23.2
- cudatoolkit>=11.0
- pandas>=1.3.0 (用于Excel输出)
- openpyxl>=3.0.0 (用于Excel文件支持)

## .mat 实验快速设置

要运行 .mat 文件实验，请激活conda环境并安装所需包：

```bash
conda activate iciclegcn
conda install pandas openpyxl -y
```

# 使用方法

## 配置

在配置文件"config/config.yaml"中，可以编辑第一阶段的训练和测试选项。

在文件"icicleGCN.py"中，可以编辑第二阶段的训练选项。

## 训练

以数据集ImageNet-10为例：

第一阶段：
设置配置后，要开始第一阶段训练，使用checkpoint_1000.tar来预热整个网络，只需运行
> python train.py

第二阶段：
设置配置后，要开始第二阶段训练，使用在第一阶段训练的checkpoint_1015.tar作为第二阶段训练的预训练工作，只需运行
> python icicleGCN.py


## 测试

一旦第一次训练完成，在配置文件"config.yaml"中指定的"model_path"路径下会有一个保存的模型。要测试第一阶段训练好的模型，运行
> python cluster.py

一旦第二次训练完成，在文件"icicleGCN.py"中指定的"test_path"路径下会有一个保存的模型。要测试第二阶段训练好的模型，运行
> python icicleGCN_test

最终结果将以TXT格式保存在"icicleGCN_result"文件中

我们上传了在论文中报告性能的预训练模型到"save_IcicleGCN"文件夹供参考。

## 运行 .mat 文件实验

对于存储在 `.mat` 文件中的数值特征数据（如 CMHDC 噪声数据集），我们提供了改进的自动化实验脚本，具有智能配置管理功能。

### 功能特性
- **自动化工作流**：第一阶段训练 → 第一阶段评估
- **智能配置**：使用基础配置文件 + 命令行参数覆盖（避免生成冗余配置文件）
- **自适应参数**：根据数据集大小自动调整 batch_size 和 epochs
- **多数据集支持**：批量处理，带进度追踪
- **错误处理**：完善的错误日志记录和可选的交互模式
- **实时进度**：训练过程中显示实时进度条和 epoch 计数器
- **Excel输出**：专业的结果报告，包含所有聚类指标（ACC、NMI、ARI、F1）

### 使用方法

#### 单个数据集实验
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01
```

#### 批量实验
```bash
# 运行多个指定数据集
python run_mat_experiments.py --batch control_uni_rayleigh_01 control_uni_rayleigh_05 control_uni_rayleigh_10

# 运行预定义的噪声对比实验
python run_mat_experiments.py --comparison
```

#### 交互模式（错误时提供重试选项）
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01 --interactive
```

#### 保存配置（用于实验复现）
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01 --save-config
```

#### 列出可用数据集
```bash
python run_mat_experiments.py --list
```

### 输出结构
```
experiments_output/
├── results/           # 实验结果（Excel格式）
│   ├── {dataset}_{timestamp}.xlsx  # 单个实验结果
│   └── batch_experiment_{timestamp}.xlsx  # 批量实验结果
├── error_logs/        # 错误日志（如有失败）
└── run_configs/       # 可选的保存配置（使用 --save-config）
save/                  # 训练好的模型
└── {dataset}/         # 每个数据集的模型检查点
```

### 配置管理
- **基础配置**：control_uni 数据集使用 `config/config_control_uni.yaml`
- **参数覆盖**：命令行参数覆盖基础配置值
- **无配置污染**：避免生成冗余配置文件
- **自适应设置**：
  - `test_small`：batch_size=20, epochs=5（快速测试）
  - `control_uni_*`：batch_size=32, epochs=10（600个样本）
  - 其他数据集：通过 TRAINING_PARAMS 配置

### Excel输出格式
- **单个实验**：独立的Excel文件包含：
  - 数据集名称、时间戳、工作流描述
  - 训练参数（epochs、batch_size、learning_rate）
  - 性能指标：ACC、NMI、ARI、F1
- **批量实验**：合并的Excel包含：
  - `Detailed_Results` 表：所有数据集的指标
  - `Summary` 表：统计信息和成功率

### 支持的数据集
- `test_small` - 快速测试数据集（100个样本）
- `control_uni_original` - 原始干净数据集（600个样本）
- `control_uni_gamma_01/05/10/20` - Gamma噪声（1%、5%、10%、20%）
- `control_uni_rayleigh_01/05/10/20` - Rayleigh噪声
- `control_uni_gaussian_01/05/10/20` - Gaussian噪声
- `control_uni_uniform_01/05/10/20` - Uniform噪声

### 性能优化
- **智能批处理**：根据数据集大小自动调整 batch_size
- **进度追踪**：实时显示训练进度
- **高效训练**：control_uni 数据集 10 个 epochs（从200减少）
- **快速测试**：test_small 数据集仅需 5 个 epochs

### 技术改进
- **健壮的指标解析**：正确提取所有聚类指标
- **编码修复**：正确处理 Windows 控制台编码
- **参数处理**：正确管理布尔参数（reload）
- **错误恢复**：完善的错误日志和可选重试机制

# 数据集

CIFAR-10、CIFAR-100、STL-10将由Pytorch自动下载。Tiny-ImageNet可以从http://cs231n.stanford.edu/tiny-imagenet-200.zip下载。对于ImageNet-10和ImageNet-dogs，我们在"dataset"文件夹中提供了它们的描述。

# 下载

任何在2023年12月09日之前点击此链接的人将直接跳转到我在ScienceDirect上的文章最终版本，欢迎您阅读或下载。
> https://authors.elsevier.com/c/1hyKE77nKkYIC

# 引用

我们非常感谢您引用我们的论文！我们论文的BibTex条目是：

> @article{xu2024deep,
  title={Deep image clustering with contrastive learning and multi-scale graph convolutional networks},
  author={Xu, Yuankun and Huang, Dong and Wang, Chang-Dong and Lai, Jian-Huang},
  journal={Pattern Recognition},
  pages={110065},
  year={2024},
  publisher={Elsevier}
}
