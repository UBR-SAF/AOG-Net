import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils.SSTDataLoader import create_dataloaders
from model.model import AOG, NUM, PREDICT_DAY, DATASET, SEA, DATA_DIR, LABEL_DIR

from utils.trainer import Trainer

def save_loss_plot(train_losses, val_losses, time):
    filename='v5_loss_{}.png'.format(time)
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Loss Curve saved as: {filename}")
    plt.close()

def save_losses_pandas(train_loss_list, val_loss_list, filename):
    max_len = max(len(train_loss_list), len(val_loss_list))
    data = {
        'Epoch': range(1, max_len + 1),
        'Train_Loss': train_loss_list + [None] * (max_len - len(train_loss_list)),
        'Val_Loss': val_loss_list + [None] * (max_len - len(val_loss_list))
    }
    
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False, float_format='%.6f')
    print(f"Loss history saved as: {filename}")
    print(f"Data shape: {df.shape}")

def sequential_scheduler(optimizer):
    annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-4
    )
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,     
        eta_min=1e-4 
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[annealing_scheduler, cos_scheduler],
        milestones=[130]
    )
    return scheduler


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AOG(num_nodes=NUM, dim=NUM, num_layers=2, predict_day=PREDICT_DAY)
    model = model.to(device)

    train_loader, val_loader, test_loader, indices = create_dataloaders(DATA_DIR, LABEL_DIR, batch_size=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    # seq_scheduler = sequential_scheduler(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=3,
        eta_min=1e-5
    )
    
    loss_MAE = nn.L1Loss()

    epochs = 100

    trainer = Trainer(model=model, device=device, optimizer=optimizer, shceduler=scheduler, loss_fc=loss_MAE, epochs=epochs, train_loader=train_loader, val_loader=val_loader)
    train_loss_list, val_loss_list = trainer.train()

    now = datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S")

    save_loss_plot(train_loss_list, val_loss_list, time)
    save_losses_pandas(train_loss_list, val_loss_list, 'v5_losses_{}.csv'.format(time))

