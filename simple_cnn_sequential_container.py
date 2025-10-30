
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# x -> conv -> ReLU -> maxpool -> flatten -> linear -> sigmoid 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 28x28 ->
            nn.Flatten(),            # -> 196
            nn.Linear(14*14, 1)      # -> 1 logit
        )

    def forward(self, x):           # Linear
        return self.model(x)

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