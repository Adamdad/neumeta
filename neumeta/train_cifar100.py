import argparse
import copy
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose
from neumeta.models import create_model_cifar100 as create_model
from neumeta.utils import (AverageMeter, EMA, create_key_masks, get_cifar100,
                           get_hypernet, get_optimizer, load_checkpoint,
                           parse_args, print_omegaconf, sample_coordinates,
                           sample_subset, sample_weights, save_checkpoint,
                           set_seed, shuffle_coordiates_all, validate,
                           validate_merge, validate_single)
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
def find_max_dim(model_cls):
    checkpoint = model_cls.learnable_parameter
    
    max_value = len(checkpoint)
    # Iterate over the new model's weights
    for i, (k, tensor) in enumerate(checkpoint.items()):
        
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
            coords = [tensor.shape[0], tensor.shape[1]]
            max_value = max(max_value, max(coords))
                    
        elif len(tensor.shape) == 2:

            coords = [tensor.shape[0], tensor.shape[1]]
            max_value = max(max_value, max(coords))
                    
        # Handle 1D tensors (e.g., biases)
        elif len(tensor.shape) == 1:
          
            max_value = max(max_value, tensor.shape[0])
    
    return max_value
    
def initialize_wandb(config):
    import time
    """
    Initializes Weights and Biases (wandb) with the given configuration.
    
    Args:
        configuration (dict): Configuration parameters for the run.
    """
    # Name the run using current time and configuration name
    run_name = f"{time.strftime('%Y%m%d%H%M%S')}-{config.experiment.name}"
    
    wandb.init(project="ninr", name=run_name, config=dict(config), group='cifar100')

    

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None):
    model.train()
    total_loss = 0.0

    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        hidden_dim = random.choice(args.dimensions.range)
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor,
                                                                                         keys_list,
                                                                                         indices_list,
                                                                                         size_list,
                                                                                         key_mask,
                                                                                         ratio=args.ratio)
        if args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        # Compute loss
        cls_loss = criterion(predict, target)  # * 0.01
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2)
                                for w in reconstructed_weights])

        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = torch.mean(torch.stack([F.mse_loss(
                w, w_gt) for w, w_gt in zip(reconstructed_weights, gt_selected_weights)]))
        else:
            reconstruct_loss = torch.tensor(0.0)

        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
            reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        for updated_weight in model_cls.parameters():
            updated_weight.grad = None

        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [
                                w.grad for k, w in model_cls.named_parameters() if k in selected_keys])

        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), args.training.clip_grad)

        optimizer.step()
        if ema:
            ema.update()  # Update the EMA after each training step
        total_loss += loss.item()

        losses.update(loss.item())
        cls_losses.update(cls_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())

        if batch_idx % args.experiment.log_interval == 0:
            wandb.log({
                "Loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            print(
                f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    return losses.avg, dim_dict, gt_model_dict


def init_model_dict(args):
    """
    Initializes a dictionary of models for each dimension in the given range, along with ground truth models for the starting dimension.

    Args:
        args: An object containing the arguments for initializing the models.

    Returns:
        dim_dict: A dictionary containing the models for each dimension, along with their corresponding coordinates, keys, indices, size, and ground truth models.
        gt_model_dict: A dictionary containing the ground truth models for the starting dimension.
    """
    dim_dict = {}
    gt_model_dict = {}
    for dim in args.dimensions.range:
        # for dp in depth_range:
        model_cls = create_model(args.model.type, hidden_dim=dim, path=args.model.pretrained_path, smooth=args.model.smooth).to(device)
        # fuse_module(model_cls)
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, hidden_dim=dim, path=args.model.pretrained_path, smooth=args.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def main_nerf():
    # Hyperparameters
    args = parse_args()

    print_omegaconf(args)
    set_seed(args.experiment.seed)
    train_loader, val_loader = get_cifar100(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None),
                                           )
    
    # checkpoint = torch.load(path, map_location='cpu')
    model = create_model(args.model.type, 
                         hidden_dim=args.dimensions.start, 
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth).to(device)
    print("Maximum DIM: ",find_max_dim(model))

    val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args)
    print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(args, number_param)
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
    
    start_epoch = 0
    best_acc = 0.0
    
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info = load_checkpoint(args.resume_from, hyper_model, optimizer, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    if args.test == False:
        initialize_wandb(args)
        dim_dict, gt_model_dict = init_model_dict(args)
        
        for epoch in range(start_epoch, args.experiment.num_epochs):
            dim_dict = shuffle_coordiates_all(dim_dict)
            train_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
            scheduler.step()

            print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % args.experiment.eval_interval == 0:
                if ema:
                    ema.apply()
                    val_loss, acc = validate(hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                    ema.restore()  # Restore the original weights
                else:
                    val_loss, acc = validate(hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                wandb.log({
                    "Validation Loss": val_loss,
                    "Validation Accuracy": acc
                })
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
                
                # Save the checkpoint
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint(f"{args.training.save_model_path}/cifar100_nerf_best.pth",hyper_model,optimizer,ema,epoch,best_acc)
                    print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
        wandb.finish()
    else:
        for hidden_dim in range(16, 65):
            model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth).to(device)

            for valid_fn in [validate, validate_merge]:
                print(f"Testing using fn {valid_fn.__name__}")

                # Apply Exponential Moving Average (EMA) if enabled
                if ema:
                    print("Applying EMA")
                    ema.apply()
                    val_loss, acc = valid_fn(hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                    ema.restore()  # Restore the original weights after applying EMA
                else:
                    val_loss, acc = valid_fn(hyper_model, val_loader, val_criterion, model_cls=model, args=args)

                print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
                
                # Define the directory and filename structure
                filename = f"cifar100_{valid_fn.__name__}_results_{args.experiment.name}.txt"
                filepath = os.path.join(args.training.save_model_path, filename)


                # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
                with open(filepath, "a") as file:
                    file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")

    print("Training finished.")

    
if __name__ == "__main__":
    # main_fit_nerf()
    main_nerf()
# 