import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import random

from neumeta.models import create_mnist_model as create_model
from neumeta.utils import load_checkpoint, EMA, AverageMeter, save_checkpoint
import os
import torch.nn.functional as F
import wandb
from neumeta.utils import (sample_coordinates, sample_weights, sample_subset,
                       print_omegaconf, shuffle_coordiates_all, set_seed, 
                       parse_args, get_hypernet, validate, get_dataset, validate_merge,
                       get_optimizer, validate_single, linear_decay, weighted_regression_loss)
import copy
from smooth.permute import PermutationManager, compute_tv_loss_for_network


print("Training Nerf On Mnist MLP")

device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_wandb(config):
    import time
    """
    Initializes Weights and Biases (wandb) with the given configuration.
    
    Args:
        configuration (dict): Configuration parameters for the run.
    """
    # Name the run using current time and configuration name
    run_name = f"{time.strftime('%Y%m%d%H%M%S')}-{config.experiment.name}"

    wandb.init(project="ninr", name=run_name,
               config=dict(config), group='MNIST')


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
        model_cls, reconstructed_weights = sample_weights(model, 
                                                          model_cls,
                                                          coords_tensor, 
                                                          keys_list, 
                                                          indices_list, 
                                                          size_list, 
                                                          key_mask, 
                                                          selected_keys,
                                                          device=device, 
                                                          NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        # Compute loss
        cls_loss = criterion(predict, target)  # * 0.01
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2)for w in reconstructed_weights])

        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(reconstructed_weights=reconstructed_weights,
                                                        gt_selected_weights=gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

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
                "Recon Weight": args.hyper_model.loss_weight.recon_weight,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            print(
                f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}, Recon Weight = {args.hyper_model.loss_weight.recon_weight:.3f}")
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
    if args.model.type == "LeNet":
        for dim in args.dimensions.range:
            # for dp in depth_range:
            model_cls = create_model(args.model.type, hidden_dim=dim, path=args.model.pretrained_path).to(device)
            model_cls.eval()
            # fuse_module(model_cls)
            coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
            dim_dict[f"{dim}"] = (model_cls, coords_tensor,
                                keys_list, indices_list, size_list, None)
            if dim == args.dimensions.start:
                print(f"Loading model for dim {dim} from {args.model.pretrained_path}")
                model_trained = create_model(args.model.type, 
                                            hidden_dim=dim, 
                                            path=args.model.pretrained_path).to(device)
                model_trained.load_state_dict(torch.load(args.model.pretrained_path))
                model_trained.eval()
                
                print("TV original model: ", compute_tv_loss_for_network(model_trained, lambda_tv=1.0).item())
                input_tensor = torch.randn(1, 1, 28, 28).to(device)
                permute_func = PermutationManager(model_trained, input_tensor)
                permute_dict = permute_func.compute_permute_dict()
                model_trained = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv_1.weight', 'in_channels'), ('linear.weight', 'out_channels'), ('linear.bias', 'out_channels')])
                print("TV permutated model: ", compute_tv_loss_for_network(model_trained, lambda_tv=1.0).item())
                gt_model_dict[f"{dim}"] = copy.deepcopy(model_trained)
    
    elif args.model.type == "ResNet_width":
            for dim in args.dimensions.range:
            # for dp in depth_range:
                model_cls = create_model(args.model.type, 
                                        hidden_dim=dim, 
                                        path=args.model.pretrained_path,
                                        depths=[3, 3]).to(device)
                model_cls.eval()
                # fuse_module(model_cls)
                coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
                dim_dict[f"{dim}"] = (model_cls, coords_tensor,
                                    keys_list, indices_list, size_list, None)
                if dim == args.dimensions.start:
                    print(f"Loading model for dim {dim} from {args.model.pretrained_path}")
                    model_trained = create_model(args.model.type, 
                                        hidden_dim=dim, 
                                        path=args.model.pretrained_path,
                                        depths=[3, 3]).to(device)
                    
                    model_trained.load_state_dict(torch.load(args.model.pretrained_path))
                    
                    print("TV original model: ", compute_tv_loss_for_network(model_trained, lambda_tv=1.0).item())
                    input_tensor = torch.randn(1, 1, 28, 28).to(device)
                    permute_func = PermutationManager(model_trained, input_tensor)
                    permute_dict = permute_func.compute_permute_dict()
                    model_trained = permute_func.apply_permutations(permute_dict, [('conv1.weight', 'in_channels'), ('linear.weight', 'out_channels'), ('linear.bias', 'out_channels')])
                    print("TV permutated model: ", compute_tv_loss_for_network(model_trained, lambda_tv=1.0).item())
                    model_trained.eval()

                    gt_model_dict[f"{dim}"] = copy.deepcopy(model_trained)
                    
    elif args.model.type == "ResNet":
        for dim in args.dimensions.range:
            # for dp in depth_range:
            model_cls = create_model(args.model.type, 
                                    hidden_dim=16, 
                                    path=args.model.pretrained_path,
                                    depths=[dim, dim]).to(device)
            model_cls.eval()
            # fuse_module(model_cls)
            coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
            dim_dict[f"{dim}"] = (model_cls, coords_tensor,
                                keys_list, indices_list, size_list, None)
            if dim == args.dimensions.start:
                print(f"Loading model for dim {dim} from {args.model.pretrained_path}")
                model_trained = create_model(args.model.type, 
                                    hidden_dim=16, 
                                    path=args.model.pretrained_path,
                                    depths=[dim, dim]).to(device)
                
                model_trained.load_state_dict(torch.load(args.model.pretrained_path))
                model_trained.eval()

                gt_model_dict[f"{dim}"] = copy.deepcopy(model_trained)
    else:
        AssertionError("Model type not supported")
    return dim_dict, gt_model_dict


def main_nerf():
    # Hyperparameters
    args = parse_args()

    print_omegaconf(args)
    set_seed(args.experiment.seed)
    train_loader, val_loader = get_dataset(
        args.training.dataset,
        args.training.batch_size,
        strong_transform=args.training.get('strong_aug', None)
    )

    # checkpoint = torch.load(path, map_location='cpu')
    if args.model.type == "LeNet":
        model = create_model(args.model.type,
                            hidden_dim=args.dimensions.start,
                            path=args.model.pretrained_path).to(device)
    elif args.model.type == "ResNet":
        model = create_model(args.model.type,
                            hidden_dim=16,
                            depths=[args.dimensions.start, args.dimensions.start],
                            path=args.model.pretrained_path).to(device)
    elif args.model.type == "ResNet_width":
        model = create_model(args.model.type,
                        hidden_dim=args.dimensions.start,
                        depths=[3,3],
                        path=args.model.pretrained_path).to(device)
    else:
        AssertionError("Model type not supported")
    model.load_state_dict(torch.load(args.model.pretrained_path))
    model.eval()
    # print("Maximum DIM: ",find_max_dim(model))

    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(args, number_param, device=device)
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)

    start_epoch = 0
    best_acc = 0.0
    initial_recon_weight = args.hyper_model.loss_weight.recon_weight
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info = load_checkpoint(
            args.resume_from, hyper_model, optimizer, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        # Note: If there are more elements to retrieve, do so here.
        
    if args.load_from:
        print(f"Load from checkpoint: {args.load_from}")
        checkpoint_info = load_checkpoint(args.load_from, hyper_model, None, ema)
        # Note: If there are more elements to retrieve, do so here.

    if args.test == False:
        initialize_wandb(args)
        dim_dict, gt_model_dict = init_model_dict(args)
        for k in gt_model_dict:
            tmp_model = gt_model_dict[k].eval()
            val_loss, acc = validate_single(tmp_model, val_loader, val_criterion, args=args)
            print(f"Permutated model: Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}")

        for epoch in range(start_epoch, args.experiment.num_epochs):
            args.hyper_model.loss_weight.recon_weight = linear_decay(epoch, 50, initial_recon_weight)
            
            dim_dict = shuffle_coordiates_all(dim_dict)
            train_loss, dim_dict, gt_model_dict = train_one_epoch(
                hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
            scheduler.step()

            print(
                f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % args.experiment.eval_interval == 0:
                if ema:
                    ema.apply()
                    val_loss, acc = validate(
                        hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                    ema.restore()  # Restore the original weights
                else:
                    val_loss, acc = validate(
                        hyper_model, val_loader, val_criterion, model_cls=model, args=args)
                wandb.log({
                    "Validation Loss": val_loss,
                    "Validation Accuracy": acc
                })
                print(
                    f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

                # Save the checkpoint
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint(
                        f"{args.training.save_model_path}/mnist_nerf_best.pth",
                        hyper_model,
                        optimizer,
                        ema,
                        epoch,
                        best_acc
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
        wandb.finish()
    else:
        for hidden_dim in range(4, 64):  # iterate over range of hidden dimensions
            model = create_model(args.model.type, hidden_dim=hidden_dim,
                                 path=args.model.pretrained_path).to(device)

            for valid_fn in [validate, validate_merge]:
                print(f"Testing using fn {valid_fn.__name__}")

                # Apply Exponential Moving Average (EMA) if enabled
                if ema:
                    ema.apply()
                    val_loss, acc = valid_fn(
                        hyper_model, val_loader, val_criterion, model_cls=model, args=args, device=device)
                    ema.restore()  # Restore the original weights after applying EMA
                else:
                    val_loss, acc = valid_fn(
                        hyper_model, val_loader, val_criterion, model_cls=model, args=args, device=device)

                print(
                    f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

                # Define the directory and filename structure
                filename = f"mnist_{valid_fn.__name__}_results_{args.experiment.name}.txt"
                filepath = os.path.join(
                    args.training.save_model_path, filename)

                # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
                with open(filepath, "a") as file:
                    file.write(
                        f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")

    print("Training finished.")


if __name__ == "__main__":
    # main_fit_nerf()
    main_nerf()
#
