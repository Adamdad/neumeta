import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from neumeta.vae.model import VAE # Adjust the import based on your project structure
from neumeta.vae.train_mnist_baseline import get_device, create_data_loaders, initialize_model, configure_optimization, calculate_loss, test_model, compute_average_mmd, compute_mse, test_model, calculate_average_neg_log_likelihood, save_mnist_images
from torchvision.utils import save_image
from neumeta.models.utils import load_checkpoint as load_checkpoint_model
from neumeta.utils import get_hypernet, get_optimizer, EMA, sample_coordinates, parse_args, sample_subset, sample_weights, print_omegaconf, shuffle_coordiates_all, AverageMeter, set_seed, create_key_masks, average_models, save_checkpoint, load_checkpoint, weighted_regression_loss
import copy
import random
import os
import pandas as pd
# import wandb

torch.autograd.set_detect_anomaly(True)

def train_one_epoch(hyper_model, device, train_loader, optimizer, scheduler, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, args=None):
    hyper_model.train()
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()
    
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = x.to(device), target.to(device)
        hidden_dim = random.choice(args.dimensions.range)
        model, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        if args.ratio < 1.0:
            coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, 
                                                                                            keys_list, 
                                                                                            indices_list, 
                                                                                            size_list, 
                                                                                            key_mask, 
                                                                                            ratio=args.ratio)
        else:
            selected_keys = list(key_mask.keys())
            
            
        if args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        model, reconstructed_weights = sample_weights(hyper_model, model, 
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,  
                                                          device=device, NORM=args.dimensions.norm)
        recon_batch, mu, log_var = model(x)

        # Calculate the mean squared error loss
        task_loss = calculate_loss(recon_batch, x, mu, log_var) # torch.mean((model_output - true_image) ** 2) + 
        # reg_loss = compute_tv_loss_for_network(model_cls, lambda_tv=1)
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])
        
        
        reconstruct_loss = 0.0
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(reconstructed_weights,gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        loss =  args.hyper_model.loss_weight.ce_weight * task_loss + args.hyper_model.loss_weight.reg_weight * reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        for updated_weight in model.parameters():
            updated_weight.grad = None


        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [w.grad for k, w in model.named_parameters() if k in selected_keys])
        
        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(hyper_model.parameters(), args.training.clip_grad)
        
        optimizer.step()
        scheduler.step()
        if ema:
            ema.update()  # Update the EMA after each training step
        
        losses.update(loss.item())
        cls_losses.update(task_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())
        # kd_losses.update(kd_loss.item())
        
        if batch_idx % args.experiment.log_interval == 0:
            # print(f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
            print(f"Iteration {batch_idx}:  Loss = ({losses.val:.4f})({losses.avg:.4f}),  Task Loss = ({cls_losses.val:.4f})({cls_losses.avg:.4f}), Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = ({reconstruct_losses.val:.4f})({reconstruct_losses.avg:.4f}), Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    return losses.avg, dim_dict, gt_model_dict
        
        
def init_model_dict(args, device):
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
        model = initialize_model(device, h_dim1=64, h_dim2=32, h_var=dim).to(device)
        state_dict = torch.load(args.model.pretrained_path, map_location='cpu')['state_dict']
        load_checkpoint_model(model, state_dict)
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
        dim_dict[f"{dim}"] = (model, coords_tensor, keys_list, indices_list, size_list, None)
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = initialize_model(device, h_dim1=64, h_dim2=32, h_var=dim).to(device)
            # state_dict = torch.load(f'./seperate_{dim}.pth', map_location='cpu')
            state_dict = torch.load(args.model.pretrained_path, map_location='cpu')['state_dict']
            # load_checkpoint(model_trained, state_dict)
            model_trained.load_state_dict(state_dict)
            model_trained.eval()
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict


def main():
    args = parse_args()
    print_omegaconf(args)
    set_seed(args.experiment.seed)

    # Dataset and Dataloader setup
    
    # Setup device and model
    device = get_device()
    model = initialize_model(device, h_dim1=64, h_dim2=32)
    state_dict = torch.load(args.model.pretrained_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare data loaders
    train_loader, test_loader = create_data_loaders(args.training.batch_size)

    print(model.state_dict().keys())
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(args, number_param, device=device)
    if args.hyper_model.ema_decay > 0:
        ema = EMA(hyper_model, decay=args.hyper_model.ema_decay)
    else:
        ema = None
    criterion, val_criterion, optimizer, scheduler = get_optimizer(args, hyper_model)
    
    
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(current_dir, args.experiment.name)
    # Create a new directory within the current directory
    os.makedirs(save_folder, exist_ok=True)
    # If specified, load the checkpoint
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint_info = load_checkpoint(args.resume_from, hyper_model, optimizer, ema)
        # Note: If there are more elements to retrieve, do so here.
    if args.load_from:
        print(f"Load from checkpoint: {args.load_from}")
        checkpoint_info = load_checkpoint(args.load_from, hyper_model, None, ema)
        # Note: If there are more elements to retrieve, do so here.
        
    if args.test == False:
        
        dim_dict, gt_model_dict = init_model_dict(args, device)
        dim_dict = shuffle_coordiates_all(dim_dict)
        
        # Train the model
        for epoch in range(1, args.experiment.num_epochs + 1):
            print(f"Epoch {epoch}/{args.experiment.num_epochs}\n----------------------------")
            train_avg_loss, dim_dict, gt_model_dict = train_one_epoch(hyper_model, device, train_loader, optimizer, scheduler, criterion, dim_dict, gt_model_dict, epoch, ema=ema, args=args)
            
            if ema:
                ema.apply()
                coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
                key_mask = create_key_masks(keys_list=keys_list)
                with torch.no_grad():
                    model, _ = sample_weights(hyper_model, model, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=args.dimensions.norm)
                model.eval()
                ema.restore()
            else:
                coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
                key_mask = create_key_masks(keys_list=keys_list)
                with torch.no_grad():
                    model, _ = sample_weights(hyper_model, model, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=args.dimensions.norm)
                model.eval()
                
            test_avg_loss = test_model(model, device, test_loader, epoch)
            with torch.no_grad():
                z = torch.randn(64, 2).cuda()
                sample = model.decoder(z).cuda()
                
                save_image(sample.view(64, 1, 28, 28), f'{save_folder}/sample_epoch{epoch}' + '.png')
            torch.save({"state_dict": hyper_model.state_dict()}, f"{save_folder}/mnist_{model.__class__.__name__}.pth")

                
            
        mse = compute_mse(model, test_loader, device=device)
        print("Average MSE:", mse)
        average_mmd_score = compute_average_mmd(model, test_loader, device)
        print(f"Average MMD score: {average_mmd_score}")
        
        
        torch.save({
            "state_dict": hyper_model.state_dict(),
            "avg_mmd_score": average_mmd_score,
            "mse": mse,
        }
            , f"{save_folder}/mnist_{model.__class__.__name__}.pth")
        with torch.no_grad():
            z = torch.randn(64, 2).cuda()
            sample = model.decoder(z).cuda()
            
            save_image(sample.view(64, 1, 28, 28), f'{save_folder}/sample_' + '.png')
    else:
        df = pd.DataFrame(columns=['h_dim', 'mse', 'mmd', 'nll'])
        for h_dim in range(32, 64):
            print(f"Testing for h_dim = {h_dim}")
            model = initialize_model(device, h_dim1=64, h_dim2=32, h_var=h_dim).to(device)
            state_dict = torch.load(args.model.pretrained_path, map_location='cpu')['state_dict']
            load_checkpoint_model(model, state_dict)
            model.eval()
            if ema:
                ema.apply()
                coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
                key_mask = create_key_masks(keys_list=keys_list)
                with torch.no_grad():
                    model, _ = sample_weights(hyper_model, model, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=args.dimensions.norm)
                model.eval()
                ema.restore()
            else:
                coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
                key_mask = create_key_masks(keys_list=keys_list)
                with torch.no_grad():
                    model, _ = sample_weights(hyper_model, model, coords_tensor, keys_list, indices_list, size_list, key_mask, list(key_mask.keys()), device=device, NORM=args.dimensions.norm)
                model.eval()
                
            mse = compute_mse(model, test_loader, device=device)
            print("Average MSE:", mse)
            average_mmd_score = compute_average_mmd(model, test_loader, device)
            print(f"Average MMD score: {average_mmd_score}")
            nll = calculate_average_neg_log_likelihood(model, test_loader)
            print(f"Average Negative Log Likelihood: {nll}")
            save_mnist_images(model, save_folder=save_folder, name=f"sample_winr_{h_dim}")
            # write the results to a csv file
            
            # Append new results and save
            # Append new results
            new_row = pd.DataFrame({'h_dim': [h_dim], 'mse': [mse], 'mmd': [average_mmd_score], 'nll': [nll]})
            df = pd.concat([df, new_row], ignore_index=True)
        print(df)
        print(f"Saving results to {args.test_result_path}")
        df.to_csv(args.test_result_path, index=False)
            
        

if __name__ == "__main__":
    main()
