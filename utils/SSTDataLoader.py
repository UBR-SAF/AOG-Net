import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

class PTDataset(Dataset):
    def __init__(self, data_dir, label_dir, indices):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx = self.indices[idx]
        data = torch.load(os.path.join(self.data_dir, f"{file_idx}.pt"))
        label = torch.load(os.path.join(self.label_dir, f"{file_idx}.pt"))
        return data, label

def create_dataloaders(data_dir, label_dir, batch_size=64, seed=42):
    all_indices = [int(f.replace('.pt', '')) 
                   for f in os.listdir(data_dir) 
                   if f.endswith('.pt')]
    all_indices.sort()
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1/3, random_state=seed)
    
    train_dataset = PTDataset(data_dir, label_dir, train_idx)
    val_dataset = PTDataset(data_dir, label_dir, val_idx)
    test_dataset = PTDataset(data_dir, label_dir, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader