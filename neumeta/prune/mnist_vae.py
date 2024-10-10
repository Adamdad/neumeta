import torch
import torch_pruning as tp
from neumeta.vae.train_mnist_baseline import compute_mse, calculate_average_neg_log_likelihood, compute_average_mmd, initialize_model, calculate_loss, save_mnist_images, calculate_sample_wise_vae_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def setup_evaluation_data(batch_size=128):
    """
    Creates train and test data loaders for the MNIST dataset.
    """
    train_dataset = datasets.MNIST(
        root='data/', 
        train=True, 
        transform=transforms.ToTensor(), 
        download=True)

    test_dataset = datasets.MNIST(
        root='data/', 
        train=False, 
        transform=transforms.ToTensor(), 
        download=False)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False)

    return test_loader, train_loader

def evaluate_model(model, data_loader, device):
    """
    Evaluate the performance of a model on the provided data.
    """
    # Preparation for model evaluation
    model.eval()  # Set the model to evaluation mode

    # Perform validation and print results
    mse = compute_mse(model, data_loader, device=device)
    # print("Average MSE:", mse)
    average_mmd_score = compute_average_mmd(model, data_loader, device)
    # print(f"Average MMD score: {average_mmd_score}")
    nll = calculate_average_neg_log_likelihood(model, data_loader)
    # print("Average NLL:", nll)
    # print(f"Validation Loss: {val_loss:.4f}, Accuracy: {acc*100:.2f}%")
    return mse, average_mmd_score, nll


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_batchs = 10
    # Prepare the validation data
    val_loader, train_loader = setup_evaluation_data()

    # Pruning setup
    example_inputs = torch.randn(1, 1, 28, 28).to(device)


    # Importance criteria
    imp_dict = {
        # 'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),
        # 'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),
        # 'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='mean'),
        # 'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),
        # 'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),
        'Random': tp.importance.RandomImportance(),
        'Hessian': tp.importance.HessianImportance(group_reduction='first'),
        'Taylor': tp.importance.TaylorImportance(group_reduction='first'),     
        'L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
        'L2': tp.importance.MagnitudeImportance(p=2, group_reduction="first"),   
    }
    # model_name = 'ResNet20'
    # target_layer = 'layer3.2.conv1'
    # model = initialize_model(device, h_dim1=64, h_dim2=32).to(device)
    # for layer_name, layer_module in model.named_modules():
    #     print(layer_name)
    # exit()
    
    
    model_name = 'VAE'
    target_layer = ['fc5']
    # Specific layer to be pruned with varying degrees
    iterative_steps = 5
    for imp_name, imp in imp_dict.items():
        # Open a text file to record the results
        with open(f"pruning_results_{model_name}_{imp_name}_mnist_vae.txt", "w") as results_file:
            # Write the header of the file
            results_file.write("Pruning Ratio,Resulting Channels,MSE,MDD,NLL\n")


            # Experiment with different pruning ratios for the specific layer
            for pruning_ratio in [i * 0.05 for i in range(20)]:  # Adjust the range/sequence as needed
                # Reset the model before each pruning experiment
                model = initialize_model(device, h_dim1=64, h_dim2=32).to(device)
                model.load_state_dict(torch.load("toy/vae/vae_samples_mnist_dim64/mnist_VAE_reg10.pth")['state_dict'])

                # Define the pruning configuration
                pruning_config = {
                    'ignored_layers': [],  # Layers to exclude from pruning
                    'pruning_ratio_dict': {},  # Specific pruning ratios per layer
                }
                for layer_name, layer_module in model.named_modules():
                    if layer_name in target_layer:
                        pruning_config['pruning_ratio_dict'][layer_module] = pruning_ratio  # Set specific pruning ratio
                    else:
                        pruning_config['pruning_ratio_dict'][layer_module] = 0  # No pruning for other layers
                    if layer_name.startswith('fc6'):
                        pruning_config['ignored_layers'].append(layer_module)  # Exclude the final classifier

                # Initialize the pruner
                pruner = tp.pruner.MetaPruner(
                    model=model,
                    example_inputs=example_inputs,
                    importance=imp,
                    iterative_steps=iterative_steps,
                    **pruning_config
                )

                for i in range(iterative_steps):
                    print(f"Pruning step {i+1}/{iterative_steps} with {imp_name} importance and {pruning_ratio * 100} pruning ratio:"  )
                    if isinstance(imp, tp.importance.HessianImportance):
                        # loss = F.cross_entropy(model(images), targets)
                        for k, (imgs, lbls) in enumerate(train_loader):
                            if k>=N_batchs: break
                            imgs = imgs.cuda()
                            recon_batch, mu, log_var = model(imgs)
                            loss = calculate_sample_wise_vae_loss(recon_batch, imgs, mu, log_var)
                            # compute loss for each sample
                            imp.zero_grad() # clear accumulated gradients
                            for l in loss:
                                model.zero_grad() # clear gradients
                                l.backward(retain_graph=True) # simgle-sample gradient
                                imp.accumulate_grad(model) # accumulate g^2
                    elif isinstance(imp, tp.importance.TaylorImportance):
                        # loss = F.cross_entropy(model(images), targets)
                        for k, (imgs, lbls) in enumerate(train_loader):
                            if k>=N_batchs: break
                            imgs = imgs.cuda()
                            recon_batch, mu, log_var = model(imgs)
                            loss = calculate_loss(recon_batch, imgs, mu, log_var)
                            loss.backward()
                    
                    # Execute the pruning
                    pruner.step()

                # Evaluate and display the model performance after pruning
                print(f"\nEvaluating model with pruned at {pruning_ratio * 100}%:")
                model.zero_grad()  # Clear any cached gradients
                # print(model)
                print(model)
                
                
                mse, average_mmd_score, nll = evaluate_model(model, val_loader, device)
                
                # Calculate the number of resulting channels
                resulting_channels = int(64 * (1 - pruning_ratio))  # Assuming 64 is the original number of channels

                # Record the results in the text file
                results_str = f"{pruning_ratio:.2f},{resulting_channels},{mse:.4f},{average_mmd_score:.4f},{nll:.4f}\n"
                results_file.write(results_str)

                # Optionally, print the results to the console
                print(f"Method: {imp_name}, Pruning ratio: {pruning_ratio:.2f}, Resulting Channels: {resulting_channels}, "
                      f"MSE: {mse:.4f}, MMD: {average_mmd_score:.4f}, NLL: {nll:.4f}")
                save_mnist_images(model, save_folder='toy/prune/', name=f'mnist_vae_pruned_{imp_name}_{pruning_ratio:.2f}')
                # exit()

# Entry point of the script
if __name__ == "__main__":
    main()
