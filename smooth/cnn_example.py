import torch
from neumeta.models import create_model_cifar10, create_model_cifar100
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from smooth.permute import PermutationManager, compute_tv_loss_for_network, PermutationManager_SingleTSP, PermutationManager_SingleTSP_Greedy, PermutationManager_Random
from torchvision.models import resnet18, resnet50

def validate(model, val_loader, criterion, device="cuda"):
    """
    Validates the model on the given validation dataset.
    
    Args:
        model: The neural network model to be evaluated.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function used for evaluation.
        device: Device to run the model on. Default is "cuda".
    
    Returns:
        Average validation loss and accuracy score.
    """
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

def cifar10_example():
    """
    Loads a pre-trained ResNet20 model, evaluates its performance on the CIFAR10 dataset,
    applies a permutation to the model's layers, and evaluates its performance again.
    """
    # Model and device setup
    model_name = 'ResNet20'
    load_path = 'resnet20-12fca82f.th'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    hidden_dim = 64

    # Data preprocessing for CIFAR10 validation dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
    ])
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained model and evaluate
    model = create_model_cifar10(model_name, hidden_dim=hidden_dim, path=load_path).to(device)
    model.eval()  # Set to evaluation mode

    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Training:", total_tv.detach().cpu().item())

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

    # Apply permutations to the model's layers and evaluate
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    permute_func = PermutationManager(model, input_tensor)
    permute_dict = permute_func.compute_permute_dict()
    model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Permute:", total_tv.detach().cpu().item())
    val_loss, acc = validate(model, val_loader, criterion)
    print(f"Epoch Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

def test_permute():
    """
    Test the effects of permutation on the model.
    """
    # Model setup
    model_name = 'ResNet20'
    load_path = 'resnet20-12fca82f.th'
    device = 'cpu'
    hidden_dim = 64

    model = create_model_cifar10(model_name, hidden_dim=hidden_dim, path=load_path).to(device)
    
    model.eval()  # Set to evaluation mode

    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Training:", total_tv)

    # Apply permutations to the model's layers and check the total variation
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    permute_func = PermutationManager(model, input_tensor)
    permute_dict = permute_func.compute_permute_dict()
    model = permute_func.apply_permutations(permute_dict, ignored_keys=[])
    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Permute:", total_tv)

def benchmark():
    """
    Test the effects of permutation on the model.
    """
    # Model setup
    
    results = {}
    model_name = 'ResNet56'
    load_path = 'resnet20-12fca82f.th'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 64

    # model = resnet18(pretrained=True).to(device)
    model = resnet50(pretrained=True).to(device)
    # model = create_model_cifar10(model_name, hidden_dim=hidden_dim, path=load_path, smooth=False).to(device)
    # model = create_model_cifar100(model_name, hidden_dim=hidden_dim, path=load_path, smooth=False).to(device)
    
    model.eval()  # Set to evaluation mode

    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Training:", total_tv)
    results["original"] = total_tv.detach().cpu().item()
    
    permute_class_dict = {
        'SingleTSP': PermutationManager_SingleTSP,
        'SingleTSP Greedy': PermutationManager_SingleTSP_Greedy,
        'Ours': PermutationManager,
        'Random': PermutationManager_Random
    }

    # Apply permutations to the model's layers and check the total variation
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    for k in permute_class_dict:
        permute_func = permute_class_dict[k](model, input_tensor, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        model_permute = permute_func()
        total_tv = compute_tv_loss_for_network(model_permute, lambda_tv=1.0)
        print(f"Total Total Variation After Permute {k}:", total_tv)
        results[k] = total_tv.detach().cpu().item()
    return results
    
def run_experiments(rounds=10, filename="experiment_results.csv"):
    import pandas as pd
    """
    Run the benchmark for a specified number of rounds and save results to a file.
    """
    all_results = []
    for round_num in range(rounds):
        results = benchmark()
        all_results.append(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    
if __name__ == '__main__':
    # cifar10_example()
    # benchmark()
    run_experiments(rounds=5)
    # test_permute()
