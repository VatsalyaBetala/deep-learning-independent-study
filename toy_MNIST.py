import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.ToTensor()

train_full = datasets.MNIST("./data", train=True,  download=True, transform=tfm)
test_full  = datasets.MNIST("./data", train=False, download=True, transform=tfm)

train_full.targets = (train_full.targets >= 5).long()
test_full.targets  = (test_full.targets  >= 5).long()

train_set = Subset(train_full, torch.arange(10_000))
test_set  = Subset(test_full,  torch.arange(2_000))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)


model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),      # 28x28 -> 14x14
    nn.Flatten(),            # -> 196
    nn.Linear(14*14, 1)      # -> 1 logit
).to(device)

# Loss & Optim 
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
epochs = 5

# Train / Eval 
for ep in range(1, epochs + 1):
    # train
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().unsqueeze(1).to(device)
        
        opt.zero_grad()
        z = model(x)
        loss = loss_fn(z, y)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * x.size(0)
        tr_correct += (torch.sigmoid(z).round().long() == y.long()).sum().item()
        tr_total += x.size(0)

    # eval
    model.eval()
    te_loss = te_correct = te_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            z = model(x)
            loss = loss_fn(z, y)
            te_loss += loss.item() * x.size(0)
            te_correct += (torch.sigmoid(z).round().long() == y.long()).sum().item()
            te_total += x.size(0)

    print(f"Epoch {ep:02d} | "
          f"train loss {tr_loss/tr_total:.4f} acc {tr_correct/tr_total:.3f} | "
          f"test loss {te_loss/te_total:.4f} acc {te_correct/te_total:.3f}")
