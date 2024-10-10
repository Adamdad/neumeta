import torch
import torch.nn.functional as F
from neumeta.models.resnet_cifar import cifar10_resnet20
from neumeta.utils import get_cifar10 
import torch
from tqdm import tqdm
from neumeta.similarity.cka import cka_linear_torch
from neumeta.models.utils import fuse_module

def load_model(hidden_dim, device, ckp_path=None, fuse=False):
    # Load and return the first model
    if ckp_path is None:
        model = cifar10_resnet20(hidden_dim=hidden_dim, pretrained=True).to(device)
    else:
        model = cifar10_resnet20(hidden_dim=hidden_dim, pretrained=False).to(device)
        if fuse:
            fuse_module(model)
        state_dict = torch.load(ckp_path)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    
    return model

def calculate_kl_divergence(model1_outputs, model2_outputs):
    kl_divs = F.kl_div(F.log_softmax(model1_outputs, dim=1), F.softmax(model2_outputs, dim=1), reduction='none')
    kl_divs = kl_divs.sum(dim=1)
    return kl_divs.mean()  # Returning mean KL divergence for the batch

class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.register_hook()

    def hook_fn(self, module, input, output):
        self.features = output

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.handle = module.register_forward_hook(self.hook_fn)
                break

    def remove_hook(self):
        self.handle.remove()

    def get_features(self):
        return self.features

def main():
    """
    Perform model similarity analysis by comparing the KL divergence and CKA (Centered Kernel Alignment) between pairs of models.
    The analysis is performed on the CIFAR10 dataset using different models with varying hidden dimensions.
    The average KL divergence and CKA values are calculated and stored in matrices for further analysis or saving.

    Returns:
        None
    """
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fuse = False
    exp_name = 'individual'
    models = [
        (None, 64),
        ('toy/experiments/cifar10_56_20231116-134800/checkpoint_last.pth.tar', 56),
        ('toy/experiments/cifar10_48_20231115-201708/checkpoint_last.pth.tar', 48),
        ('toy/experiments/cifar10_40_20231116-134800/checkpoint_last.pth.tar', 40),
        ('toy/experiments/cifar10_32_20231115-201557/checkpoint_last.pth.tar', 32),
        ('toy/experiments/cifar10_24_20231116-134800/checkpoint_last.pth.tar', 24),
        ('toy/experiments/cifar10_16_20231115-201702/checkpoint_last.pth.tar', 16),
        # Add more models as needed
    ]
        
    num_models = len(models)
    kl_matrix = torch.zeros(num_models, num_models)
    cka_matrix = torch.zeros(num_models, num_models)
    
    # Iterating through all unique pairs of models
    for i, (ckp_model1, dim1) in enumerate(models):
        for j, (ckp_model2, dim2) in enumerate(models):
            if i != j:  # Skip identical pairs
                print("Comparing models with hidden dimensions {} and {}".format(dim1, dim2))
                print("Model 1: {}".format(ckp_model1))
                print("Model 2: {}".format(ckp_model2))
                
                model1 = load_model(hidden_dim=dim1, device=device, ckp_path=ckp_model1, fuse=fuse)
                model2 = load_model(hidden_dim=dim2, device=device, ckp_path=ckp_model2, fuse=fuse)

                # Load CIFAR10 dataset
                train_loader, val_loader = get_cifar10(batch_size=128)

                # Set models to evaluation mode
                model1.eval()
                model2.eval()
                layer_name = 'layer3.2'
                # Register hooks to capture the features from layer 3.2
                extractor1 = FeatureExtractor(model1, layer_name)
                extractor2 = FeatureExtractor(model2, layer_name)
                
                
                total_kl_divergence = 0.0
                total_cka = 0.0
                num_batches = 0

                with torch.no_grad():
                    for data, _ in tqdm(val_loader):
                        # Compute the outputs of both models
                        data = data.to(device)
                        output1 = model1(data)
                        output2 = model2(data)

                        # Calculate KL divergence
                        kl_div = calculate_kl_divergence(output1, output2)
                        total_kl_divergence += kl_div.item()

                        # Compute KL divergence and CKA on layer outputs
                        if extractor1 and extractor2:
                            # Calculate CKA
                            cka_value = cka_linear_torch(extractor1.get_features(), extractor2.get_features())  
                            total_cka += cka_value

                        num_batches += 1

                average_kl_divergence = total_kl_divergence / num_batches
                average_cka = total_cka / num_batches
                
                # Store the average results in the matrices
                kl_matrix[i][j] = total_kl_divergence / num_batches
                cka_matrix[i][j] = total_cka / num_batches
                print("Average KL Divergence: {}".format(average_kl_divergence))
                print("Average CKA: {}".format(average_cka))
                
    # You can now use kl_matrix and cka_matrix for further analysis or save them
    # For example, to save the matrices:
    torch.save(kl_matrix, f'kl_matrix_{exp_name}.pth')
    torch.save(cka_matrix, f'cka_matrix_{exp_name}.pth')

if __name__ == "__main__":
    main()
