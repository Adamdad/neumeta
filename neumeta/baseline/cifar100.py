import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from neumeta.models import create_model_cifar100
import argparse


from neumeta.toy_cls import compute_tv_loss_for_network

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, target in tqdm(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)

        # Forward pass
        predict = model(x) + compute_tv_loss_for_network(model, lambda_tv=1e-2)

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
    
def test():
    parser = argparse.ArgumentParser(description="Training script for various ResNet models on CIFAR-10/CIFAR-100.")

    # Add the model name argument
    parser.add_argument('--model_name', type=str, required=True, choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56'],
                        help='Model to use for training. Options: ResNet20, ResNet32, ResNet44, ResNet56')
    
    args = parser.parse_args()
    model_name = args.model_name
    
    load_path = None
   
    batch_size = 128
    hidden_dim = 64
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
   
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_model_cifar100(model_name, hidden_dim=hidden_dim, path=load_path, smooth=True).to(device)
    model.eval()  # Set to evaluation mode

    criterion = torch.nn.CrossEntropyLoss()

    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    torch.save(model.state_dict(), f"toy/checkpoint/cifar100_{model_name}_smoothed.pth")

def test_smoothed():
    parser = argparse.ArgumentParser(description="Training script for various ResNet models on CIFAR-10/CIFAR-100.")

    # Add the model name argument
    parser.add_argument('--model_name', type=str, required=True, choices=['ResNet20', 'ResNet32', 'ResNet44', 'ResNet56'],
                        help='Model to use for training. Options: ResNet20, ResNet32, ResNet44, ResNet56')
    
    args = parser.parse_args()
    model_name = args.model_name
    
    load_path = f"toy/checkpoint/cifar100_{model_name}_smoothed.pth"
   
    batch_size = 128
    hidden_dim = 64
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409],
                            std=[0.2673, 0.2564, 0.2761])
    ])
   
    val_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform_test)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_model_cifar100(model_name, hidden_dim=hidden_dim, path=load_path, smooth=False).to(device)
    model.eval()  # Set to evaluation mode

    criterion = torch.nn.CrossEntropyLoss()

    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    
if __name__ == "__main__":
    test()
