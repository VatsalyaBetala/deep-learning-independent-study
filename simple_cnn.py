
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt

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

class ShapesDataset(Dataset):
    def __init__(self, n_samples=1000, image_size=28, transform=None):
        self.n_samples = n_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = Image.new("L", (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(img)
        shape = random.choice(["circle", "square"])
        size = random.randint(8, 14)
        x0 = random.randint(5, self.image_size - size - 5)
        y0 = random.randint(5, self.image_size - size - 5)
        x1, y1 = x0 + size, y0 + size

        if shape == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=255)
            label = 0
        else:
            draw.rectangle([x0, y0, x1, y1], fill=255)
            label = 1

        if self.transform:
            img = self.transform(img)
        return img, label

# 3. Simulate Data + Training

transform = transforms.Compose([transforms.ToTensor()])
train_data = ShapesDataset(n_samples=800, transform=transform)
test_data = ShapesDataset(n_samples=200, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

print("Trainable parameters:", count_params(model))


for epoch in range(1, 6):
    train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)

    # Evaluate
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_loss += loss.item() * x.size(0)
            total_correct += (preds.squeeze(1) == y.long().squeeze(1)).sum().item()
            total += x.size(0)
    val_loss, val_acc = total_loss / total, total_correct / total

    print(f"Epoch {epoch}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%")

    
    
dataset = ShapesDataset(n_samples=8, transform=transforms.ToTensor())

fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for i, ax in enumerate(axes):
    img, label = dataset[i]
    ax.imshow(img.squeeze(0), cmap="gray")
    ax.set_title("Circle" if label == 0 else "Square", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()
