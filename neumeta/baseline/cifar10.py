import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from neumeta.models import create_model_cifar10
from neumeta.models.utils import replace_bn_with_conv
from smooth.permute import PermutationManager, compute_tv_loss_for_network, PermutationManager_Random, PermutationManager_SingleTSP, PermutationManager_SingleTSP_Greedy

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

def validate(model, val_loader, criterion, device="cuda"):
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
    
def test_fuse():
    model_name = 'ResNet20'
    load_path = 'toy/cifar10_ResNet_dim8.pth'
   
    batch_size = 128
    hidden_dim = 8
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
   
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    resnet18= create_model_cifar10(hidden_dim=hidden_dim).to(device)
    resnet18.load_state_dict(torch.load('toy/cifar10_ResNet_dim8.pth'))
    resnet18.eval()  # Set to evaluation mode

    # Fuse Convolution and BatchNorm layers for inference
    replace_bn_with_conv(resnet18)
    print(resnet18)
    criterion = torch.nn.CrossEntropyLoss()

    val_loss, acc = validate(resnet18, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

def test_fuse_smooth():
    model_name = 'ResNet20'
    # load_path = 'toy/cifar10_ResNet_dim8.pth'
   
    batch_size = 128
    hidden_dim = 64
   
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])
   
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    methods = {
        'Random': PermutationManager_Random,
        'Layer-Opt2': PermutationManager_SingleTSP,
        'Layer-Greedy': PermutationManager_SingleTSP_Greedy,
        'Ours': PermutationManager
    }
    for m in methods:
        print(f"Permute using {m}")
        # model = create_model_cifar10(model_name, hidden_dim, path=None, smooth=True).to(device)
        model = create_model_cifar10(model_name, hidden_dim, path=None, smooth=False).to(device)
        print("Smooth the parameters of the model")
        print("TV original model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
        input_tensor = torch.randn(1, 3, 32, 32).to(device)
        permute_func = methods[m](model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        print("TV original model: ", compute_tv_loss_for_network(model, lambda_tv=1.0).item())
        # model.load_state_dict(torch.load('toy/cifar10_ResNet_dim8.pth'))
        model.eval()  # Set to evaluation mode

        criterion = torch.nn.CrossEntropyLoss()

        val_loss, acc = validate(model, val_loader, criterion)
        print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
        torch.save(model.state_dict(), f"toy/checkpoint/cifar10_{model_name}_smooth_{m}.pth")
            
    
def test():
    model_name = 'ResNet20'
    load_path = 'resnet20-12fca82f.th'
   
    batch_size = 128
    hidden_dim = 64
    #   ('mean', [0.4914, 0.4822, 0.4465]),
    #   ('std', [0.2023, 0.1994, 0.201]),
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.201])
    ])
   
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = create_model_cifar10(model_name, hidden_dim=hidden_dim, path=load_path).to(device)
    model.eval()  # Set to evaluation mode

    criterion = torch.nn.CrossEntropyLoss()

    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    # main()
    # test_fuse()
    test_fuse_smooth()
    # test()
