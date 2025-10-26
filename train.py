import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from util import yaml_config_hook, save_model
from torch.utils import data
import cluster
import torch.nn.functional as F
from utils import MatDataset

def train():
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j, h_i, bar_i,  = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        re_loss = F.mse_loss(bar_i,h_i)

        loss = loss_instance + loss_cluster + re_loss
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t   loss_instance: {loss_instance.item()}\t  loss_cluster: {loss_cluster.item()}\t re_loss: {re_loss.item()}\t ")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "CIFAR-100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/ImageNet-10/train',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    elif args.dataset == "ImageNet-dogs":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/ImageNet-dogs/train',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 15
    elif args.dataset == "tiny-ImageNet":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/tiny-imagenet-200/train',
            transform=transform.Transforms(s=0.5, size=args.image_size),
        )
        class_num = 200
    elif args.dataset == "Flowers17":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/Flowers17/jpg',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 17
    elif args.dataset.startswith("control_uni"):
        # Handle control_uni datasets (both original and noisy versions)
        if args.dataset == "control_uni":
            mat_file_path = 'datasets/control_uni.mat'
        else:
            # For noisy datasets, look in the noisy_datasets folder
            mat_file_path = f'datasets/CMHDC_noisy_datasets/{args.dataset}.mat'
        
        print(f"Loading {args.dataset} dataset from {mat_file_path}")
        dataset = MatDataset(
            mat_file_path=mat_file_path,
            transform=None,
            augmentation_noise=0.1,
            augmentation_dropout=0.1
        )
        
        # Get class number from the dataset
        class_num = len(np.unique(dataset.labels))
        print(f"Detected {class_num} classes in {args.dataset}")
        
        # Check if it's feature vector data
        if len(dataset.data.shape) == 2:
            print(f"Detected feature vector data with shape {dataset.data.shape}")
            input_dim = dataset.data.shape[1]
            print(f"Using FeatureVectorEncoder for feature vector data (input_dim={input_dim})")
        else:
            print(f"Detected image data with shape {dataset.data.shape}")
            input_dim = None
    else:
        raise NotImplementedError


    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    
    # initialize model
    if args.dataset.startswith("control_uni") and len(dataset.data.shape) == 2:
        # Use FeatureVectorEncoder for feature vector data
        res = network.FeatureVectorEncoder(input_dim=input_dim, hidden_dim=256)
        print(f"Initialized FeatureVectorEncoder with input_dim={input_dim}")
    else:
        # Use ResNet for image data
        res = resnet.get_resnet(args.resnet)
        print(f"Initialized ResNet: {args.resnet}")
    
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.reload:

        model_fp = os.path.join(args.model_path,"checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model_new_dict = model.state_dict()
        state_dict = {k:v for k,v in checkpoint['net'].items() if k in model_new_dict.keys()}
        model_new_dict.update(state_dict)
        model.load_state_dict(model_new_dict)
        args.start_epoch = checkpoint['epoch'] + 1


    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 1 == 0:
            save_model(args, model, optimizer, epoch)

        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
