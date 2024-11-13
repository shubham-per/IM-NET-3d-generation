import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class ShapeDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.h5')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            vertices = f['vertices'][:]  # Shape should be (2048, 3)
            vertices = vertices.flatten()  # Flatten to (6144,)
        return torch.tensor(vertices, dtype=torch.float32)

# Example usage:
data_dir = 'D:/Code/Python/IM-NET/datasets/train'
train_dataset = ShapeDataset(data_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
