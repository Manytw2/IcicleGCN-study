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

对于存储在 `.mat` 文件中的数值特征数据（如 CMHDC 噪声数据集），我们提供了自动化实验脚本，支持：

### 功能特性
- **自动化工作流**：第一阶段训练 → 第一阶段评估
- **多数据集支持**：批量处理多个数据集
- **错误处理**：自动错误日志记录和失败时的用户交互
- **有序输出**：所有结果保存到 `experiments_output/` 目录
- **Excel结果**：结果以Excel格式保存，具有清晰的行列标题
- **批量对比**：多个数据集结果在一张Excel表格中，便于对比分析

### 使用方法

#### 单个数据集实验
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01
```

#### 批量实验
```bash
python run_mat_experiments.py --batch control_uni_rayleigh_01 control_uni_rayleigh_05 control_uni_rayleigh_10 control_uni_rayleigh_20
```

#### 交互模式（错误时询问用户）
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01 --interactive
```

#### 列出可用数据集
```bash
python run_mat_experiments.py --list
```

### 输出结构
```
experiments_output/
├── configs/           # 实验专用配置文件
│   ├── control_uni/   # Control_uni 数据集配置
│   │   ├── train/     # 训练配置
│   │   └── eval/      # 评估配置
│   └── other/         # 其他数据集配置
├── results/           # 实验结果 (Excel格式)
│   ├── {dataset}_{timestamp}.xlsx  # 单个实验结果
│   └── batch_experiment_{timestamp}.xlsx  # 批量实验结果
├── error_logs/        # 错误日志（如有失败）
├── models/            # 训练好的模型文件
└── README.md          # 目录结构说明文档
```

### Excel输出格式
- **单个实验**：独立的Excel文件，包含详细指标
- **批量实验**：合并的Excel文件，包含两个工作表：
  - `Detailed_Results`：所有数据集的ACC、NMI、ARI、F1指标
  - `Summary`：实验统计信息和成功率
- **清晰标题**：数据集、状态、ACC、NMI、ARI、F1、实验时间、工作流等

### 支持的数据集
- `control_uni_original` - 原始数据集
- `control_uni_gamma_01/05/10/20` - Gamma 噪声变体
- `control_uni_rayleigh_01/05/10/20` - Rayleigh 噪声变体
- `control_uni_gaussian_01/05/10/20` - Gaussian 噪声变体
- `control_uni_uniform_01/05/10/20` - Uniform 噪声变体

### 技术细节
- **数据格式**：`.mat` 文件，包含 'X'/'fea'/'data' 特征和 'Y'/'gnd'/'labels' 标签
- **网络架构**：`FeatureVectorEncoder` 用于数值特征数据
- **训练方式**：对比学习配合高斯噪声和dropout增强
- **评估指标**：聚类性能指标 (ACC, NMI, ARI, F1)
- **依赖包**：需要 `pandas` 和 `openpyxl` 用于Excel输出
- **配置管理**：为每个实验动态生成配置文件

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
