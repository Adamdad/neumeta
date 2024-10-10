import csv
import os
from typing import Callable, List, Optional, Union

import PIL
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from neumeta.models.utils import fuse_module
from neumeta.vae.model import CelebAVAE  # Adjust the import based on your project structure
from smooth.permute import compute_tv_loss_for_network
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

class cfg(object):
    KL_WEIGHT = 0.0004
    learning_rate = 2e-3
    rep_weight = 0.01
    dim = 64
    batch_size = 128
    num_epochs = 100
    eval_interval = 5
    
    @classmethod
    def print_variables(cls):
        for attr, value in cls.__dict__.items():
            if not attr.startswith('__'):
                print(f"{attr} = {value}")
    
def get_device():
    """
    Selects CUDA or CPU device depending on availability.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CelebADataset(datasets.VisionDataset):
    base_folder = "celeba"

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type= [],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        
    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> datasets.celeba.CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return datasets.celeba.CSV(headers, indices, torch.tensor(data_int))
    
    def __getitem__(self, index: int):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        target = 0

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self) -> int:
        return len(self.filename)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)

    
def create_data_loaders(batch_size, strong_aug=False):
    """
    Creates train and test data loaders for the MNIST dataset.
    """
    if strong_aug:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_dataset = CelebADataset(
        root='data/',
        split='train',
        transform=train_transform,
        download=False)

    test_dataset = CelebADataset(
        root='data/',
        split='valid',
        transform=val_transform,
        download=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Adjust this according to your system's capabilities
        pin_memory=True  # Set to True if using a CUDA-enabled GPU
    )


    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False)

    return train_loader, test_loader

def calculate_sample_wise_vae_loss(recon_x, x, mu, log_var):
    """
    Calculate the sample-wise loss for a Variational Autoencoder (VAE).
    
    Args:
    - recon_x: Reconstructed images/output of the decoder.
    - x: Original images/input images.
    - mu: Mean from the latent space (output of the encoder).
    - log_var: Log variance from the latent space (output of the encoder).
    
    Returns:
    - Tensor of losses for each sample in the batch.
    """
    # Reshape inputs to match dimensions
    # x = x.view(recon_x.size(0), -1)
    # recon_x = recon_x.view(recon_x.size(0), -1)
    assert x.shape == recon_x.shape
    
    # Reconstruction loss (Binary Cross Entropy) for each sample
    recon_loss = F.mse_loss(recon_x, x, reduction='none')
    recon_loss = recon_loss.sum(dim=(1, 2, 3))
    
    # KL divergence for each sample
    kl_divergence = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl_divergence = kl_divergence.sum(dim=1)
    
    # Total loss for each sample
    sample_wise_loss = recon_loss + kl_divergence
    
    return sample_wise_loss


def initialize_model(device, x_dim=3, h_dim1=64, h_dim2=100, z_dim=32, h_var=None):
    """
    Initializes the VAE model and moves it to the specified device.
    """
    model = CelebAVAE(init_channels=h_dim1,
                      image_channels=x_dim,
                      latent_dim=z_dim,
                      h_var=h_var).to(device)
    return model

def configure_optimization(model, learning_rate=1e-3):
    """
    Configures the optimizer for the model.
    """
    return optim.Adam(model.parameters(), lr=learning_rate)

def calculate_loss(recon_x, x, mu, log_var):
    """
    Calculate the loss for a Variational Autoencoder (VAE).
    
    Args:
    - recon_x: Reconstructed images/output of the decoder.
    - x: Original images/input images.
    - mu: Mean from the latent space (output of the encoder).
    - log_var: Log variance from the latent space (output of the encoder).
    
    Returns:
    - Total loss, comprised of the reconstruction loss and KL divergence.
    """
    # Reshape inputs to match dimensions
    # x = x.view(recon_x.size(0), -1)
    # recon_x = recon_x.view(recon_x.size(0), -1)
    assert x.shape == recon_x.shape
    
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')


    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + kld_loss * cfg.KL_WEIGHT
    
    return total_loss/len(x)

def train_one_epoch(model, device, train_loader, optimizer, epoch, scheduler, log_interval=100):
    model.train()
    total_loss = []
    # Process each batch of training data.
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        task_loss = calculate_loss(recon_batch, data, mu, log_var)
        if cfg.rep_weight > 0:
            reg =  compute_tv_loss_for_network(model, lambda_tv=1)
        else:
            reg =  torch.tensor(0)#compute_tv_loss_for_network(model, lambda_tv=1)
        loss = task_loss + cfg.rep_weight * reg
        loss.backward()
        total_loss.append(loss.item())
        # grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        scheduler.step()

        # Print the training status.
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, Reg: {reg.item():.6f}, Task: {task_loss.item():.6f}, Lr: {scheduler.get_last_lr()[0]:.6f}')

    # Compute the average loss.
    average_loss = sum(total_loss) / len(total_loss)
    print(f'====> Epoch: {epoch} Average training loss: {average_loss:.4f}')
    return average_loss

def compute_mse(vae, data_loader, device):
    total_mse = 0
    total_samples = 0

    # Set the VAE to evaluation mode
    vae.eval()

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)  # if you're using a GPU
            # inputs_flat = inputs.view(inputs.size(0), -1)  # Flatten the input
            reconstructed, _, _ = vae(inputs)
            
            
            # Compute MSE for this batch
            mse_loss = F.mse_loss(reconstructed, inputs, reduction='sum')
            
            total_mse += mse_loss.item()  # Sum up the MSE
            total_samples += inputs.shape[0]

    # Compute the average MSE over all samples
    average_mse = total_mse / total_samples
    return average_mse




def test_model(model, device, test_loader, epoch):
    model.eval()
    test_loss = []

    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            data = data.to(device)
            recon, mu, log_var = model(data)
            test_loss.append(calculate_loss(recon, data, mu, log_var).item())

    # Compute the average loss for the epoch.
    average_test_loss = sum(test_loss) / len(test_loss)
    print(f'====> Epoch: {epoch} Average test loss: {average_test_loss:.4f}\n')

    return average_test_loss

def compute_mmd(x, y, kernel='rbf'):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y, using a specified kernel.

    Args:
    x (Tensor): A batch of representations (latent codes) for one set of data.
    y (Tensor): A batch of representations (latent codes) for another set of data.
    kernel (str): The kernel to use for computation ('rbf' for Gaussian kernel).

    Returns:
    mmd (Tensor): The computed MMD between the two sets of data.
    """

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    # Compute the pairwise distances
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    # Choose the bandwidth of the kernel (median distance heuristic)
    # This part can be replaced by other bandwidth selection methods
    median_distance = torch.median(torch.sqrt(dxx[dxx > 0]).detach())
    bandwidth = median_distance ** 2 / torch.log(torch.tensor(x.shape[0]).float())

    # Compute the kernel
    if kernel == 'rbf':
        Kxx = torch.exp(-dxx / bandwidth)
        Kyy = torch.exp(-dyy / bandwidth)
        Kxy = torch.exp(-dxy / bandwidth)
    else:
        raise NotImplementedError("Only the RBF kernel is implemented in this example.")

    # Calculate the MMD
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def calculate_average_neg_log_likelihood(model, data_loader, device="cuda"):
    """
    Calculate the average negative log likelihood across all batches in a DataLoader
    using a pretrained VAE.

    Args:
    - model (VAE): A pretrained VAE model.
    - data_loader (DataLoader): DataLoader containing the dataset to evaluate.

    Returns:
    - average_nll (float): The average negative log likelihood across all batches.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Initialize variables to store total NLL and total number of samples
    total_nll = 0.0
    total_samples = 0
    
    # Loop over all batches
    with torch.no_grad():
        for x, _ in data_loader:
            # Move data to the same device as the model
            x = x.to(device)
            
            # Forward pass through the model
            recon_x, mu, log_var = model(x)

            # Reconstruction loss (using Binary Cross Entropy)
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')

            # KL divergence loss
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss is the sum of reconstruction loss and KL divergence
            total_loss = recon_loss + kld_loss
            
            # Update total NLL and the number of samples
            total_nll += total_loss.item()
            total_samples += x.size(0)
    
    # Calculate the average NLL
    average_nll = total_nll / total_samples
    
    return average_nll

def compute_average_mmd(vae, data_loader, device):
    """
    Calculate the average Maximum Mean Discrepancy (MMD) over all batches in the provided data loader.
    This metric evaluates how closely the distribution of the VAE's encoded latent vectors matches a target distribution (typically a standard normal distribution).

    Parameters:
    vae (VAE): The Variational Autoencoder model set for evaluation.
    data_loader (DataLoader): Provides batches of real data.
    device (torch.device): The device on which the computation will be performed, typically either "cpu" or "cuda".

    Returns:
    float: The average MMD value across all batches of data.
    """

    # Set VAE to evaluation mode. This is necessary as the model behaves differently during training.
    vae.eval()

    total_mmd = 0.0  # Accumulator for the MMD values
    num_examples = 0  # Accumulator for the number of examples processed``

    # We do not need gradients for evaluation, so we use no_grad to save memory.
    with torch.no_grad():
        for _, (real_data, _) in enumerate(data_loader):
            # Move the real data to the appropriate device (CPU/GPU)
            real_data = real_data.to(device)
            # real_data_flat = real_data.view(real_data.size(0), -1)

            # Encode the current batch of real data to the latent space.
            mu, log_var = vae.encode(real_data)
            
            latent_real = vae.sampling(mu, log_var)

            # Simulate the prior distribution (often standard normal) by generating fake latent variables.
            latent_fake = torch.randn_like(latent_real).to(device)  # Ensure same device as latent_real

            # Calculate the MMD metric between the real and fake data sets.
            mmd_value = compute_mmd(latent_real, latent_fake)

            # Accumulate the total MMD value and increment the batch count
            total_mmd += mmd_value.item() * real_data.size(0)
            num_examples += real_data.size(0)

    # Compute the average MMD value
    average_mmd = total_mmd / num_examples

    return average_mmd

def save_celeba_reconstructions(model, save_folder, epoch=None):
    # Set the random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    with torch.no_grad():
        z = torch.randn(64, 32).cuda()
        sample = model.decode(z).cuda()
        if epoch is not None:
            save_image(sample.view(64, 3, 64, 64), f'{save_folder}/sample_' + str(epoch) + '.png')
        else:
            save_image(sample.view(64, 3, 64, 64), f'{save_folder}/sample_' + '.png')
    
def main():
    """
    Main function to set up components and execute training.
    """
    # Configuration parameters
    # dim = 64
    # batch_size = 128
    # num_epochs = 50
    cfg.print_variables()

    # Setup device and model
    device = get_device()
    model = initialize_model(device, h_dim1=cfg.dim, z_dim=32)

    # Prepare data loaders
    train_loader, test_loader = create_data_loaders(cfg.batch_size)

    # Set up the optimizer
    optimizer = configure_optimization(model, cfg.learning_rate)
    T_max = len(train_loader) * cfg.num_epochs  # Total number of iterations
    # Set up the learning rate scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)

    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(current_dir, f'vae_samples_celeba_dim{cfg.dim}_reg{cfg.rep_weight}_lr{cfg.learning_rate}_kl{cfg.KL_WEIGHT}')
    # Create a new directory within the current directory
    os.makedirs(save_folder, exist_ok=True)
    # mse = compute_mse(model, test_loader, device=device)
    # print("Average MSE:", mse)
    # average_mmd_score = compute_average_mmd(model, test_loader, device)
    # print(f"Average MMD score: {average_mmd_score}")
    
    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"Epoch {epoch}/{cfg.num_epochs}\n----------------------------")
        train_avg_loss = train_one_epoch(model, device, train_loader, optimizer, epoch, scheduler)
        save_celeba_reconstructions(model, save_folder, epoch)
        if epoch % cfg.eval_interval == 0:
            test_avg_loss = test_model(model, device, test_loader, epoch)
            # Assuming 'vae' is your trained VAE model
            mse = compute_mse(model, test_loader, device=device)
            print("Average MSE:", mse)
            average_mmd_score = compute_average_mmd(model, test_loader, device)
            print(f"Average MMD score: {average_mmd_score}")
            nll = calculate_average_neg_log_likelihood(model, test_loader)
            print("Average NLL:", nll)
            torch.save({
                "state_dict": model.state_dict(),
                "avg_mmd_score": average_mmd_score,
                "mse": mse,
            }
        , f"{save_folder}/celeba_{model.__class__.__name__}.pth")
            
    print("Training finished.")
    # Print configuration parameters
    cfg.print_variables()
    

def test():
    dim = 64
    batch_size = 128

    # Setup device and model
    device = get_device()
    model = initialize_model(device, h_dim1=dim, z_dim=32)
    model.load_state_dict(torch.load("toy/vae/vae_samples_celeba_dim64/celeba_CelebAVAE.pth")["state_dict"])

    # Prepare data loaders
    train_loader, test_loader = create_data_loaders(batch_size)

    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(current_dir, f'vae_samples_celeba_dim{dim}')
    # Create a new directory within the current directory
    os.makedirs(save_folder, exist_ok=True)

    test_avg_loss = test_model(model, device, test_loader, 0)
    # Assuming 'vae' is your trained VAE model
    mse = compute_mse(model, test_loader, device=device)
    print("Average MSE:", mse)
    average_mmd_score = compute_average_mmd(model, test_loader, device)
    print(f"Average MMD score: {average_mmd_score}")
    nll = calculate_average_neg_log_likelihood(model, test_loader)
    print("Average NLL:", nll)
    

def permute_save():
    dim = 64
    batch_size = 100

    # Setup device and model
    device = get_device()
    model = initialize_model(device, h_dim1=dim, z_dim=32)
    model.load_state_dict(torch.load("toy/vae/vae_samples_celeba_dim64_reg0.01_lr0.002_kl0.0004/celeba_CelebAVAE.pth")["state_dict"])
    model.to(device)
    model.eval()
    fuse_module(model)
    print(model)
    
    tv_loss = compute_tv_loss_for_network(model, lambda_tv=1.0).item()
    print("Total Total Variation Before Permute:", tv_loss)
    # print([k for k, w in model.named_parameters()])
    # # exit()
    # sample_input = torch.randn(1, 3, 64, 64).to(device)
    # permuter = PermutationManager(model, sample_input)
    # permute_dict = permuter.compute_permute_dict()
    # model = permuter.apply_permutations(permute_dict, ignored_keys=[('encoder.0.weight', 'in_channels'), ('decoder.9.weight', 'out_channels'), ('decoder.9.bias', 'out_channels')])
    # # Get the current directory of the script
    # tv_loss = compute_tv_loss_for_network(model, lambda_tv=1.0).item()
    # print("Total Total Variation After Permute:", tv_loss)
    # Prepare data loaders
    _, test_loader = create_data_loaders(batch_size)
    
    # test_avg_loss = test_model(model, device, test_loader, 0)
    # Assuming 'vae' is your trained VAE model
    mse = compute_mse(model, test_loader, device=device)
    print("Average MSE:", mse)
    average_mmd_score = compute_average_mmd(model, test_loader, device)
    print(f"Average MMD score: {average_mmd_score}")
    # nll = calculate_average_neg_log_likelihood(model, test_loader)
    # print("Average NLL:", nll)
    
    torch.save({
        "state_dict": model.state_dict(),
        "avg_mmd_score": average_mmd_score,
        "mse": mse,
    }
        , f"toy/vae/vae_samples_celeba_dim64_reg0.01_lr0.002_kl0.0004/celeba_CelebAVAE_fused.pth")
    
    

if __name__ == "__main__":
    # main()
    permute_save()
    # test()
