import torch
import torch.nn as nn
import json
from pathlib import Path
import numpy as np
import pandas as pd

class ModelCheckpoint:
    def __init__(self, ):
        self.metadata_path = f"./checkpoints/loss/metadata.json"
        self.best_loss = float('inf')
        
    def __call__(self, loss, epoch, model):
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        best_loss = metadata.get('best_loss')

        if loss < best_loss:
            new_best_loss = loss
            
            torch.save(model, f"./checkpoints/model/"+ str(new_best_loss) +"_model.pth")
            print("best model saved as best_model.pth")

            metadata = {
                'best_loss': new_best_loss,
                'best_epoch': epoch,
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"best loss: epoch={epoch}, loss={new_best_loss:.6f}")

class Trainer(object):
    def __init__(self, model, device, optimizer, shceduler, loss_fc, epochs, train_loader, val_loader):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.shceduler = shceduler
        self.loss_fc = loss_fc
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint = ModelCheckpoint()
        self.n =None

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for batch_idx, (data, label) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data, self.n)

            output = output.to(self.device)
            label = label.to(self.device)

            loss = self.loss_fc(output, label)
            loss.backward()

            self.optimizer.step()
            epoch_loss += loss.item()
        self.shceduler.step()
        return epoch_loss / len(self.train_loader)

    def train(self):
        train_loss_list = []
        val_loss_list = []
        for epoch in range(1, self.epochs+1):
            train_epoch_loss = self.train_epoch(epoch)
            val_epoch_loss = self.validate()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_Loss: {:.6f}\tval_loss: {:.6f}'.format(
                epoch, epoch, self.epochs, 100. * epoch / self.epochs, train_epoch_loss, val_epoch_loss)
                )
            self.checkpoint(val_epoch_loss, epoch, self.model)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
        return train_loss_list, val_loss_list

    def validate(self):
        self.model.eval()
        val_loss = 0    
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.val_loader):
                data = data.to(self.device)
                output = self.model(data, self.n)
                output = output.to(self.device)
                label = label.to(self.device)
                loss = self.loss_fc(output, label)
                val_loss += loss.item()
        
        return val_loss/len(self.val_loader)
    
    def test(self):
        t = 1
        self.model.eval()
        test_loss = 0    
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.val_loader):
                data = data.to(self.device)
                output = self.model(data, self.n)
                output = output.to(self.device)
                label = label.to(self.device)
                loss = self.loss_fc(output, label)
                test_loss += loss.item()

                output = output.float().round(decimals=2)
                label = label.float().round(decimals=2)
                t += 1
        return output, label, test_loss/len(self.val_loader)