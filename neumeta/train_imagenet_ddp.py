import torch
import torch.nn as nn
import random
import os
import time
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import OmegaConf
from neumeta.models import create_model_imagenet
from neumeta.utils import (load_checkpoint, EMA, AverageMeter, save_checkpoint,
                       sample_coordinates, sample_weights, sample_subset, 
                       print_omegaconf, get_imagenet, 
                       create_key_masks, set_seed, parse_args, get_hypernet, 
                       get_optimizer, validate, shuffle_coordiates_all)

import wandb

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def cleanup_ddp():
    torch.distributed.destroy_process_group()
    
def initialize_wandb(config, rank):
    if rank == 0:
        # Name the run using current time and configuration name
        run_name = f"{time.strftime('%Y%m%d%H%M%S')}-{config.experiment.name}"
        wandb.init(project="ninr", name=run_name, config=dict(config), group='imagenet')


    

def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None, rank=0):
    model.train()
    total_loss = 0.0

    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(rank), target.to(rank)
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
                                                          device=rank, NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        # Compute loss
        cls_loss = criterion(predict, target)  # * 0.01
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])

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

        if batch_idx % args.experiment.log_interval == 0 and rank == 0:
            
            wandb.log({
                "Loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            print(
                f"Epoch {epoch_idx}, Iteration [{batch_idx}/{len(train_loader)}]: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    return losses.avg, dim_dict, gt_model_dict


def init_model_dict(args, rank):
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
        model_cls = create_model_imagenet(args.model.type, hidden_dim=dim, path=args.model.pretrained_path, smooth=args.model.smooth).to(rank)
        # fuse_module(model_cls)
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model_imagenet(args.model.type, hidden_dim=dim, path=args.model.pretrained_path, smooth=args.model.smooth).to(rank)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def main_nerf(rank, world_size):
    # Setup DDP
    setup_ddp(rank, world_size)

    # Hyperparameters
    args = parse_args()

    if rank == 0:
        print_omegaconf(args)
    set_seed(args.experiment.seed)
    train_loader, val_loader = get_imagenet(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None),
                                           data_dir=args.data.data_dir, 
                                           ddp=True)
    
    # checkpoint = torch.load(path, map_location='cpu')
    model = create_model_imagenet(args.model.type, 
                         hidden_dim=args.dimensions.start, 
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth).to(rank)

    # val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args)
    # print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(args, number_param)
    hyper_model.to(rank)
    hyper_model = DDP(hyper_model, device_ids=[rank])
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
    
    start_epoch = 0
    best_acc = 0.0
    
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        checkpoint_info = load_checkpoint(args.resume_from, hyper_model, optimizer, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
            print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    if args.test == False:
        # Initialize wandb only on rank 0
        initialize_wandb(args, rank)
        dim_dict, gt_model_dict = init_model_dict(args, rank)
        dim_dict = shuffle_coordiates_all(dim_dict)
        for epoch in range(start_epoch, args.experiment.num_epochs):
            train_loader.batch_sampler.sampler.set_epoch(epoch)  # Important for shuffling in DDP
            train_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args, rank=rank)
            scheduler.step()
            if rank == 0:
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            if (epoch + 1) % args.experiment.eval_interval == 0:
                if rank == 0:
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
        if rank == 0:
            wandb.finish()

    cleanup_ddp()


    
if __name__ == "__main__":
    # Use torch.multiprocessing to spawn multiple processes
    world_size = torch.cuda.device_count()  # Set world_size to your number of GPU nodes
    torch.multiprocessing.spawn(main_nerf, args=(world_size,), nprocs=world_size, join=True)