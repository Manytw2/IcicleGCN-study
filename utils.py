import numpy as np
import scipy.sparse as sp
import scipy.io as sio
# import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k)
    else:
        path = 'graph/{}_graph.txt'.format(dataset)

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


class MatDataset(Dataset):
    """
    Dataset class for loading .mat files
    Supports both image data and feature vectors
    """
    def __init__(self, mat_file_path, transform=None, augmentation_noise=0.1, augmentation_dropout=0.1):
        """
        Args:
            mat_file_path: Path to the .mat file
            transform: Optional transform to be applied on images (only for image data)
            augmentation_noise: Std of Gaussian noise for feature augmentation (default: 0.1)
            augmentation_dropout: Probability of feature dropout for augmentation (default: 0.1)
        """
        # Load .mat file
        mat_data = sio.loadmat(mat_file_path)
        
        # Extract data - common keys are 'X', 'fea', 'data'
        if 'X' in mat_data:
            self.data = mat_data['X']
        elif 'fea' in mat_data:
            self.data = mat_data['fea']
        elif 'data' in mat_data:
            self.data = mat_data['data']
        else:
            raise ValueError("Could not find data in .mat file. Expected keys: 'X', 'fea', or 'data'")
        
        # Extract labels - common keys are 'Y', 'gnd', 'labels'
        if 'Y' in mat_data:
            self.labels = mat_data['Y'].flatten()
        elif 'gnd' in mat_data:
            self.labels = mat_data['gnd'].flatten()
        elif 'labels' in mat_data:
            self.labels = mat_data['labels'].flatten()
        else:
            # If no labels, create dummy labels
            self.labels = np.zeros(len(self.data), dtype=int)
        
        # Ensure labels are integers starting from 0
        unique_labels = np.unique(self.labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.labels = np.array([label_map[label] for label in self.labels])
        
        self.transform = transform
        self.augmentation_noise = augmentation_noise
        self.augmentation_dropout = augmentation_dropout
        
        # Normalize feature vectors if it's not image data
        if len(self.data.shape) == 2:  # Feature vector data
            self.data = self._normalize_features(self.data)
    
    def _normalize_features(self, features):
        """Normalize features to zero mean and unit variance"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std
    
    def _augment_features(self, features):
        """Apply augmentation to feature vectors"""
        augmented = features.copy()
        
        # Add Gaussian noise
        if self.augmentation_noise > 0:
            noise = np.random.normal(0, self.augmentation_noise, features.shape)
            augmented = augmented + noise
        
        # Random dropout
        if self.augmentation_dropout > 0:
            dropout_mask = np.random.random(features.shape) > self.augmentation_dropout
            augmented = augmented * dropout_mask
        
        return augmented
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        label = self.labels[idx]
        
        if len(self.data.shape) == 2:  # Feature vector data
            # Create two augmented versions for contrastive learning
            x_i = torch.from_numpy(self._augment_features(data_item)).float()
            x_j = torch.from_numpy(self._augment_features(data_item)).float()
            return (x_i, x_j), label
        else:  # Image data
            # Convert to PIL Image if needed
            if not isinstance(data_item, Image.Image):
                if data_item.dtype != np.uint8:
                    data_item = (data_item * 255).astype(np.uint8)
                data_item = Image.fromarray(data_item)
            
            # Apply transforms
            if self.transform:
                x_i = self.transform(data_item)
                x_j = self.transform(data_item)
            else:
                x_i = torch.from_numpy(np.array(data_item)).float()
                x_j = torch.from_numpy(np.array(data_item)).float()
            
            return (x_i, x_j), label


