import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from neumeta.models.lenet import MnistNet, MnistResNet
from smooth.permute import PermutationManager, compute_tv_loss_for_network

from neumeta.toy_cls import compute_tv_loss_for_network


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, target in tqdm(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)

        # Forward pass
        predict = model(x) #+ compute_tv_loss_for_network(model, lambda_tv=1e-2)

        # Compute loss
        loss = criterion(predict, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    preds = []
    gt = []
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.to(device), target.to(device)
            predict = model(x)
            pred = torch.argmax(predict, dim=-1)
            preds.append(pred)
            gt.append(target)
            loss = criterion(predict, target)
            val_loss += loss.item()
    return val_loss / len(val_loader), accuracy_score(torch.cat(gt).cpu().numpy(), torch.cat(preds).cpu().numpy())

    
def main():
    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 10
    hidden_dim= 32

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MnistNet(hidden_dim=hidden_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        if (epoch + 1) % 1 == 0:
            val_loss, acc = validate(model, val_loader, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    torch.save(model.state_dict(), f"toy/mnist_{model.__class__.__name__}_dim{hidden_dim}.pth")
    print("Training finished.")
    
def main_resnet():
    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 10
    hidden_dim= 16
    num_blocks = [3,3]

    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MnistResNet(num_blocks=num_blocks, hidden_dim=hidden_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        if (epoch + 1) % 1 == 0:
            val_loss, acc = validate(model, val_loader, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    torch.save(model.state_dict(), f"toy/mnist_{model.__class__.__name__}_dim{hidden_dim}.pth")
    print("Training finished.")

def permute():
    hidden_dim= 16
    num_blocks = [3,3]
    batch_size = 128
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = MnistResNet(num_blocks=num_blocks, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(f"toy/mnist_{model.__class__.__name__}_dim{hidden_dim}.pth"))
    model.eval()
    print("TV original model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    permute_func = PermutationManager(model, input_tensor)
    permute_dict = permute_func.compute_permute_dict()
    model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('linear.weight', 'out_channels'), ('linear.bias', 'out_channels')])
    print("TV permutated model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    torch.save(model.state_dict(), f"toy/mnist_{model.__class__.__name__}_dim{hidden_dim}_permute.pth")

def test():
    hidden_dim= 32
    batch_size = 128
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MnistNet(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(f"toy/mnist_{model.__class__.__name__}_dim{hidden_dim}.pth"))
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Test on MNIST, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    
    
    

if __name__ == "__main__":
    # main()
    # main_resnet()
    permute()
