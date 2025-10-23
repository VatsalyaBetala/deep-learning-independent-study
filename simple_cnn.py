
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# x -> conv -> ReLU -> maxpool -> flatten -> linear -> sigmoid 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(14 * 14 * 1, 1, bias=True) 

    def forward(self, x):
        z1 = self.conv(x)             # conv
        a1 = F.relu(z1)               # ReLU
        p1 = self.pool(a1)            # MaxPool
        flat = p1.view(p1.size(0), -1) # Flatten 
        z2 = self.fc(flat)            # Linear
        return z2

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device).unsqueeze(1) 
        opt.zero_grad()
        logits = model(x)                      
        loss = loss_fn(logits, y)             
        loss.backward()                      
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        total_correct += (preds.squeeze(1) == y.long().squeeze(1)).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total