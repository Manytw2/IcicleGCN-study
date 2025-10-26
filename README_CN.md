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
