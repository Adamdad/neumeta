import torch
import numpy as np
import random
from prettytable import PrettyTable
from omegaconf import OmegaConf
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeRF model with CIFAR-10")

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Ratio used for training purposes')
    parser.add_argument('--resume_from', type=str,
                        help='Checkpoint file path to resume training from')
    parser.add_argument('--load_from', type=str,
                        help='Checkpoint file path to load')
    parser.add_argument('--test_result_path', type=str,
                        help='Path to save the test result')
    parser.add_argument('--test', action='store_true',
                        default=False, help='Test the model')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Load the base configuration
    if config.get('base_config', None):
        print("Loading base config from " + config.base_config)
        base_config = OmegaConf.load(config.base_config)
        config = OmegaConf.merge(base_config, config)

    # Convert args to a dictionary
    # We filter out None values and the 'config' argument
    cli_args = {k: v for k, v in vars(args).items()}

    # Merge command-line arguments into the configuration
    config = OmegaConf.merge(config, cli_args)
    if len(config.dimensions.range) == 2:
        interval = config.dimensions.get('interval', 1)
        config.dimensions.range = list(
            range(config.dimensions.range[0], config.dimensions.range[1] + 1, interval))
    return config


def print_omegaconf(cfg):
    """
    Print an OmegaConf configuration in a table format.

    :param cfg: OmegaConf configuration object.
    """
    # Flatten the OmegaConf configuration to a dictionary
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    # Create a table with PrettyTable
    table = PrettyTable()

    # Define the column names
    table.field_names = ["Key", "Value"]

    # Recursively go through the items and add rows
    def add_items(items, parent_key=""):
        for k, v in items.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is another dict, recursively add its items
                add_items(v, parent_key=current_key)
            else:
                # If it's a leaf node, add it to the table
                table.add_row([current_key, v])

    # Start adding items from the top-level configuration
    add_items(flat_config)

    # Print the table
    print(table)


def set_seed(seed_value=42):
    """Set the seed for generating random numbers for PyTorch and other libraries to ensure reproducibility.

    Args:
        seed_value (int, optional): The seed value. Defaults to 42 (a commonly used value in randomized algorithms requiring a seed).
    """
    print("Setting seed..." + str(seed_value) + " for reproducibility")
    # Set the seed for generating random numbers in Python's random library.
    random.seed(seed_value)

    # Set the seed for generating random numbers in NumPy, which can also affect randomness in cases where PyTorch relies on NumPy.
    np.random.seed(seed_value)

    # Set the seed for generating random numbers in PyTorch. This affects the randomness of various PyTorch functions and classes.
    torch.manual_seed(seed_value)

    # If you are using CUDA, and want to generate random numbers on the GPU, you need to set the seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        # For multi-GPU, if you are using more than one GPU.
        torch.cuda.manual_seed_all(seed_value)

        # Additionally, for even more deterministic behavior, you might need to set the following environment, though it may slow down the performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.set_shadow(model)

    def set_shadow(self, model):
        # Initialize the shadow weights with the model's weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def apply(self):
        # Backup the current model weights and set the model's weights to the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        # Restore the original model weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def update(self):
        # Update the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * \
                    self.shadow[name] + (1.0 - self.decay) * param.data


def save_checkpoint(filepath, model, optimizer, ema, epoch, best_acc):
    """
    Saves the current state including a model, optimizer, and EMA shadow weights.

    Args:
    filepath (str): The file path where the checkpoint will be saved.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    epoch (int): The current epoch.
    best_acc (float): The best accuracy observed during training.
    """
    # Save the model, optimizer, EMA shadow weights, and other elements
    if ema is not None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,  # specifically saving shadow weights
            'best_acc': best_acc,
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, ema, device='cuda'):
    """
    Loads the state from a checkpoint into the model, optimizer, and EMA object.

    Args:
    filepath (str): The file path to load the checkpoint from.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema is not None:
        ema.shadow = {k: checkpoint['ema_shadow'][k].to(
            device) for k in checkpoint['ema_shadow']}
    # ema.shadow = {k:checkpoint['ema_shadow'][k].to(device) for k in checkpoint['ema_shadow'] }  # specifically loading shadow weights

    return checkpoint  # Contains other information like epoch, best_acc
