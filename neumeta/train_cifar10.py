# Import necessary libraries
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
# Import functions from neumeta module
from neumeta.models import create_model_cifar10 as create_model
from neumeta.utils import (AverageMeter, EMA, load_checkpoint, print_omegaconf, 
                       sample_coordinates, sample_merge_model, 
                       sample_subset, sample_weights, save_checkpoint, 
                       set_seed, shuffle_coordiates_all, validate, validate_ensemble, sample_single_model,validate_merge,
                       validate_single, get_cifar10, 
                       get_hypernet, get_optimizer, 
                       parse_args, 
                       weighted_regression_loss)

# Print message
print("Training INR On CIFAR10")

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to find the maximum dimension of the model
def find_max_dim(model_cls):
    # Get the learnable parameters of the model
    checkpoint = model_cls.learnable_parameter
    # Set the maximum value to the length of the checkpoint
    max_value = len(checkpoint)
    # Iterate over the new model's weights
    for i, (k, tensor) in enumerate(checkpoint.items()):
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
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
    
    wandb.init(project="ninr", name=run_name, config=dict(config), group='cifar10')

# Function to train the model for one epoch
def train_one_epoch(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None):
    # Set the model to training mode
    model.train()
    total_loss = 0.0

    # Initialize AverageMeter objects to track the losses
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    # Iterate over the training data
    for batch_idx, (x, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Move the data to the device
        x, target = x.to(device), target.to(device)
        # Choose a random hidden dimension
        hidden_dim = random.choice(args.dimensions.range)
        # Get the model class, coordinates, keys, indices, size, and key mask for the chosen dimension
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        # Sample a subset of the coordinates, keys, indices, size, and selected keys
        coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor,
                                                                                         keys_list,
                                                                                         indices_list,
                                                                                         size_list,
                                                                                         key_mask,
                                                                                         ratio=args.ratio)
        # Add noise to the coordinates if specified
        if args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        # Sample the weights for the model
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)

        # Forward pass
        predict = model_cls(x)
        # Compute classification loss
        cls_loss = criterion(predict, target) 
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])

        # Compute reconstruction loss if ground truth model is available
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(
                reconstructed_weights, gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        # Compute the total loss
        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
            reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        # Zero the gradients of the updated weights
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None

        # Compute the gradients of the reconstructed weights
        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [
                                w.grad for k, w in model_cls.named_parameters() if k in selected_keys])

        # Clip the gradients if specified
        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), args.training.clip_grad)

        # Update the weights
        optimizer.step()
        # Update the EMA if specified
        if ema:
            ema.update()  # Update the EMA after each training step
        total_loss += loss.item()

        # Update the AverageMeter objects
        losses.update(loss.item())
        cls_losses.update(cls_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())

        # Log the losses and learning rate to wandb
        if batch_idx % args.experiment.log_interval == 0:
            wandb.log({
                "Loss": losses.avg,
                "Cls Loss": cls_losses.avg,
                "Reg Loss": reg_losses.avg,
                "Reconstruct Loss": reconstruct_losses.avg,
                "Learning rate": optimizer.param_groups[0]['lr']
            }, step=batch_idx + epoch_idx * len(train_loader))
            # Print the losses and learning rate
            print(
                f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    return losses.avg, dim_dict, gt_model_dict


# Function to initialize the model dictionary
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
        # Create a model for the given dimension
        model_cls = create_model(args.model.type, 
                                 hidden_dim=dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth).to(device)
        # Sample the coordinates, keys, indices, and size for the model
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        # Add the model, coordinates, keys, indices, size, and key mask to the dictionary
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        # If the dimension is the starting dimension, add the ground truth model to the dictionary
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, 
                                         hidden_dim=dim, 
                                         path=args.model.pretrained_path, 
                                         smooth=args.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

# Main function to train the model
def main():
    # Parse the arguments
    args = parse_args()

    # Print the arguments
    print_omegaconf(args)

    # Set the random seed
    set_seed(args.experiment.seed)

    # Get the training and validation data loaders
    train_loader, val_loader = get_cifar10(args.training.batch_size, 
                                           strong_transform=args.training.get('strong_aug', None),
                                           )
    
    # Create the model for the starting dimension
    model = create_model(args.model.type, 
                         hidden_dim=args.dimensions.start, 
                         path=args.model.pretrained_path, 
                         smooth=args.model.smooth).to(device)

    # Print the maximum dimension of the model
    print("Maximum DIM: ",find_max_dim(model))

    # Validate the model for the starting dimension
    val_loss, acc = validate_single(model, val_loader, nn.CrossEntropyLoss(), args=args)
    print(f"Initial Permutated model Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")

    # Get the learnable parameters of the model
    checkpoint = model.learnable_parameter
    # Get the number of parameters
    number_param = len(checkpoint)
    # Print the keys of the parameters and the number of parameters
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")

    # Get the hypermodel
    hyper_model = get_hypernet(args, number_param)
    # Initialize the EMA
    ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    # Get the criterion, validation criterion, optimizer, and scheduler
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
    
    # Initialize the starting epoch and best accuracy
    start_epoch = 0
    best_acc = 0.0
    
    # Create the directory to save the model
    os.makedirs(args.training.save_model_path, exist_ok=True)

    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info = load_checkpoint(args.resume_from, hyper_model, optimizer, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    # If not testing, initialize wandb, the model dictionary, and the ground truth model dictionary
    if args.test == False:
        initialize_wandb(args)
        dim_dict, gt_model_dict = init_model_dict(args)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        # Iterate over the epochs
        for epoch in range(start_epoch, args.experiment.num_epochs):
            
            # Train the model for one epoch
            train_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, ema=ema, args=args)
            # Step the scheduler
            scheduler.step()

            # Print the training loss and learning rate
            print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # If it's time to evaluate the model
            if (epoch + 1) % args.experiment.eval_interval == 0:
                # If EMA is specified, apply it
                if ema:
                    ema.apply()
                    
                # Sample the merged model
                model = sample_merge_model(hyper_model, model, args)
                # Validate the merged model
                val_loss, acc = validate_single(model, val_loader, val_criterion, args=args)
                
                # If EMA is specified, restore the original weights
                if ema:
                    ema.restore()  # Restore the original weights
               
                # Log the validation loss and accuracy to wandb
                wandb.log({
                    "Validation Loss": val_loss,
                    "Validation Accuracy": acc
                })
                # Print the validation loss and accuracy
                print(f"Epoch [{epoch+1}/{args.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
                
                # Save the checkpoint if the accuracy is better than the previous best
                if acc > best_acc:
                    best_acc = acc
                    save_checkpoint(f"{args.training.save_model_path}/cifar10_nerf_best.pth",hyper_model,optimizer,ema,epoch,best_acc)
                    print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
        wandb.finish()
    # If testing, iterate over the hidden dimensions and test the model
    else:
        for hidden_dim in range(16, 65):
            # Create a model for the given hidden dimension
            model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth).to(device)

            # If EMA is specified, apply it
            if ema:
                print("Applying EMA")
                ema.apply()
                
            # Sample the merged model
            accumulated_model = sample_merge_model(hyper_model, model, args, K=100)

            # Validate the merged model
            val_loss, acc = validate_single(accumulated_model, val_loader, val_criterion, args=args)
            
            # If EMA is specified, restore the original weights after applying EMA
            if ema:
                ema.restore()  # Restore the original weights after applying EMA
            
            # Save the model
            save_name = os.path.join(args.training.save_model_path, f"cifar10_{accumulated_model.__class__.__name__}_dim{hidden_dim}_single.pth")
            torch.save(accumulated_model.state_dict(),save_name)

            # Print the results
            print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
            
            # Define the directory and filename structure
            filename = f"cifar10_results_{args.experiment.name}.txt"
            filepath = os.path.join(args.training.save_model_path, filename)

            # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
            with open(filepath, "a") as file:
                file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")

    # Print message
    print("Training finished.")

    
if __name__ == "__main__":
    main()