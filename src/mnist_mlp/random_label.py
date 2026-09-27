# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2, ToTensor


# %%
"""Loading Data"""

# Downloading Training data from open datasets
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(training_data.targets), generator=g)
training_data.targets = training_data.targets[perm]

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# %%
batch_size = 64

# Creating data loader
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of Y: {y.shape} {y.dtype}")
    break

# %%
"""Creating Models"""
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


# Define Models
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# %%
"""Optimizing the Model Parameters"""
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# %%
def train(dataloader, loss_func, optimizer, model):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)
        loss = loss_func(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")


# %%
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batch
    correct /= size
    print(
        f"Test error: \n Accuracy: {100 * correct:>0.1f}%, Avg Loss: {test_loss:>8f}\n"
    )


# %%
epoch = 5
for i in range(epoch):
    print(f"epoch: {i + 1}\n--------------")
    train(train_dataloader, loss_function, optimizer, model)
    test(test_dataloader, model, loss_function)
print("Done!")

# %%
"""Save Model"""
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch model state to model.pth")

# %%
"""Load Model"""
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
# %%
