# Deep image clustering with contrastive learning and multi-scale graph convolutional networks (IcicleGCN)

This is the code for our Pattern Recognition'24 paper "Deep image clustering with contrastive learning and multi-scale graph convolutional networks"

<div align=center><img src="Figures/Figure1.png" width = "70%"/></div>

# Dependency

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.8.1
- munkres>=1.1.4
- numpy>=1.19.2
- opencv-python>=4.4.0.46
- pyyaml>=5.3.1
- scikit-learn>=0.23.2
- cudatoolkit>=11.0
- pandas>=1.3.0 (for Excel output)
- openpyxl>=3.0.0 (for Excel file support)

## Quick Setup for .mat Experiments

To run .mat file experiments, activate the conda environment and install required packages:

```bash
conda activate iciclegcn
conda install pandas openpyxl -y
```

# Usage

## Configuration

There is a configuration file "config/config.yaml", where one can edit both the training and test options of the first stage.

In the file "icicleGCN.py", where one can edit training options of the second stage.

## Training

Take dataset ImageNet-10 as an example:

First stage:
After setting the configuration, to start first stage training, use the checkpoint_1000.tar to warm up the whole network, simply run
> python train.py

Second stage:
After setting the configuration, to start second stage training, use the checkpoint_1015.tar which was training in the first stage as the pretrain work for second stage train, simply run
> python icicleGCN.py


## Test

Once the first training is completed, there will be a saved model in the "model_path" specified in the configuration file "config.yaml". To test the trained model of first stage, run
> python cluster.py

Once the second training is completed, there will be a saved model in the "test_path" specified in the file "icicleGCN.py". To test the trained model of second stage, run
> python icicleGCN_test

and the final results will be save in file "icicleGCN_result" in TXT format

We uploaded the pretrained model which achieves the performance reported in the paper to the "save_IcicleGCN" folder for reference.

## Running .mat File Experiments

For numerical feature data stored in `.mat` files (such as CMHDC noisy datasets), we provide an automated experiment script with improved configuration management.

### Features
- **Automated workflow**: First stage training → First stage evaluation
- **Smart configuration**: Uses base config files with command-line parameter overrides (no redundant config generation)
- **Adaptive parameters**: Automatically adjusts batch_size and epochs based on dataset size
- **Multiple dataset support**: Batch processing with progress tracking
- **Error handling**: Comprehensive error logging and optional interactive mode
- **Real-time progress**: Live training progress with epoch counter
- **Excel output**: Professional results with all clustering metrics (ACC, NMI, ARI, F1)

### Usage

#### Single Dataset Experiment
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01
```

#### Batch Experiments
```bash
# Run multiple specific datasets
python run_mat_experiments.py --batch control_uni_rayleigh_01 control_uni_rayleigh_05 control_uni_rayleigh_10

# Run predefined noise comparison experiment
python run_mat_experiments.py --comparison
```

#### Interactive Mode (with retry prompts on errors)
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01 --interactive
```

#### Save Configuration (for reproducibility)
```bash
python run_mat_experiments.py --dataset control_uni_rayleigh_01 --save-config
```

#### List Available Datasets
```bash
python run_mat_experiments.py --list
```

### Output Structure
```
experiments_output/
├── results/           # Experiment results (Excel format)
│   ├── {dataset}_{timestamp}.xlsx  # Single experiment results
│   └── batch_experiment_{timestamp}.xlsx  # Batch experiment results
├── error_logs/        # Error logs (if any failures occur)
└── run_configs/       # Optional saved configurations (with --save-config)
save/                  # Trained models
└── {dataset}/         # Model checkpoints for each dataset
```

### Configuration Management
- **Base configs**: Uses `config/config_control_uni.yaml` for control_uni datasets
- **Parameter override**: Command-line arguments override base config values
- **No config pollution**: Avoids generating redundant config files
- **Adaptive settings**:
  - `test_small`: batch_size=20, epochs=5 (for quick testing)
  - `control_uni_*`: batch_size=32, epochs=10 (600 samples)
  - Other datasets: Configurable via TRAINING_PARAMS

### Excel Output Format
- **Single experiments**: Individual Excel files with:
  - Dataset name, timestamp, workflow description
  - Training parameters (epochs, batch_size, learning_rate)
  - Performance metrics: ACC, NMI, ARI, F1
- **Batch experiments**: Combined Excel with:
  - `Detailed_Results` sheet: All datasets with metrics
  - `Summary` sheet: Statistics and success rates

### Supported Datasets
- `test_small` - Quick test dataset (100 samples)
- `control_uni_original` - Original clean dataset (600 samples)
- `control_uni_gamma_01/05/10/20` - Gamma noise (1%, 5%, 10%, 20%)
- `control_uni_rayleigh_01/05/10/20` - Rayleigh noise
- `control_uni_gaussian_01/05/10/20` - Gaussian noise
- `control_uni_uniform_01/05/10/20` - Uniform noise

### Performance Optimization
- **Smart batching**: Automatically adjusts batch_size for dataset size
- **Progress tracking**: Real-time epoch progress display
- **Efficient training**: 10 epochs for control_uni datasets (reduced from 200)
- **Fast testing**: 5 epochs for test_small dataset

### Technical Improvements
- **Robust metric parsing**: Correctly extracts all clustering metrics
- **Encoding fix**: Handles Windows console encoding properly
- **Parameter handling**: Properly manages boolean parameters (reload)
- **Error recovery**: Comprehensive error logging and optional retry

# Dataset

CIFAR-10, CIFAR-100, STL-10 will be automatically downloaded by Pytorch. Tiny-ImageNet can be downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip. For ImageNet-10 and ImageNet-dogs, we provided their description in the "dataset" folder.

# Download
Anyone clicking on this link before December 09, 2023 will be taken directly to the final version of my article on ScienceDirect, which you are welcome to read or download.
> https://authors.elsevier.com/c/1hyKE77nKkYIC

# Citation
We are truly grateful for citing our paper! The BibTex entry of our paper is:

> @article{xu2024deep,
  title={Deep image clustering with contrastive learning and multi-scale graph convolutional networks},
  author={Xu, Yuankun and Huang, Dong and Wang, Chang-Dong and Lai, Jian-Huang},
  journal={Pattern Recognition},
  pages={110065},
  year={2024},
  publisher={Elsevier}
}
