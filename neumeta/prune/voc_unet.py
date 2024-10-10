import torch
import torch_pruning as tp
from neumeta.utils import validate_single as validate
from torch.utils.data import DataLoader
from neumeta.segmentation.dataset import VOCSegmentation
from neumeta.segmentation.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, PILToTensor, Resize, RandomResize, ToDtype
import segmentation_models_pytorch as smp
from neumeta.segmentation.train_voc_baseline import evaluate_model
import pandas as pd

def setup_evaluation_data():
    """
    Prepare the ImageNet dataset for model evaluation.
    """

    # Define the transforms based on the provided configuration
    train_transform = Compose([
        RandomResize(480, 640),
        RandomCrop(512),
        RandomHorizontalFlip(flip_prob=0.5),
        PILToTensor(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize([480, 640]),
        PILToTensor(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define dataset paths and other settings directly
    train_dataset = VOCSegmentation(
        root="/local_home/yangxingyi/dataset/VOCdevkit",
        year="2012_aug",
        image_set="train",
        transforms=train_transform,
        download=False
    )
    valid_dataset = VOCSegmentation(
        root="/local_home/yangxingyi/dataset/VOCdevkit",
        year="2012",
        image_set="val",
        transforms=val_transform
    )

    # Define DataLoader settings directly
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=12
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    return valid_loader, train_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_batchs = 10
    # Prepare the validation data
    val_loader, train_loader = setup_evaluation_data()

    # Pruning setup
    example_inputs = torch.randn(1, 3, 256, 256).to(device)


    # Importance criteria
    imp_dict = {
        # 'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),
        # 'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),
        # 'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='mean'),
        # 'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),
        # 'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),
        # 'Random': tp.importance.RandomImportance(),
        # 'Hessian': tp.importance.HessianImportance(group_reduction='first'),
        # 'L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
        'Taylor': tp.importance.TaylorImportance(group_reduction='first'),     
        
        # 'L2': tp.importance.MagnitudeImportance(p=2, group_reduction="first"),   
    }
    model_name = 'UnetResNet18'
    target_layer = 'encoder.layer3.1.conv1'
    
    iterative_steps = 5
    
    prune_list = [0, 0.25, 0.5, 0.75]
    for imp_name, imp in imp_dict.items():
        # Open a text file to record the results
        results = []


        # Experiment with different pruning ratios for the specific layer
        for pruning_ratio in prune_list:  # Adjust the range/sequence as needed
            # Reset the model before each pruning experiment
            # model = resnet18(pretrained=True).to(device)
            model = smp.Unet(
                encoder_name= "resnet18",
                encoder_weights="imagenet",
                in_channels=3,
                classes=21,
            ).to(device)
            state_dict = torch.load('toy/checkpoint/unet_r18_voc_checkpoint.pth.tar')['state_dict']
            model.load_state_dict(state_dict)

            # Define the pruning configuration
            pruning_config = {
                'ignored_layers': [],  # Layers to exclude from pruning
                'pruning_ratio_dict': {},  # Specific pruning ratios per layer
            }
            for layer_name, layer_module in model.named_modules():
                if layer_name.startswith(target_layer):
                    pruning_config['pruning_ratio_dict'][layer_module] = pruning_ratio  # Set specific pruning ratio
                else:
                    pruning_config['pruning_ratio_dict'][layer_module] = 0  # No pruning for other layers
                if layer_name.startswith('fc') and layer_name.startswith('decoder'):
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
                    for k, batch in enumerate(train_loader):
                        if k>=N_batchs: break
                        images, masks = batch
                        images, masks = images.to(device), masks.to(device)
    
                        output = model(images) 
                        # compute loss for each sample
                        loss = torch.nn.functional.cross_entropy(output, masks, reduction='none', ignore_index=255)
                        loss = loss.mean(dim=[1,2,3])
                        imp.zero_grad() # clear accumulated gradients
                        for l in loss:
                            model.zero_grad() # clear gradients
                            l.backward(retain_graph=True) # simgle-sample gradient
                            imp.accumulate_grad(model) # accumulate g^2
                elif isinstance(imp, tp.importance.TaylorImportance):
                    # loss = F.cross_entropy(model(images), targets)
                    for k, batch in enumerate(train_loader):
                        if k>=N_batchs: break
                        images, masks = batch
                        images, masks = images.to(device), masks.to(device)
    
                        output = model(images) 
                        loss = torch.nn.functional.cross_entropy(output, masks, ignore_index=255)
                        loss.backward()
                
                # Execute the pruning
                pruner.step()

            # Evaluate and display the model performance after pruning
            print(f"\nEvaluating model with {target_layer} pruned at {pruning_ratio * 100}%:")
            model.zero_grad()  # Clear any cached gradients
            print(model)
            
            aggregated_scores = evaluate_model(model, val_loader, device)
            
            # Calculate the number of resulting channels
            resulting_channels = int(256 * (1 - pruning_ratio))  # Assuming 64 is the original number of channels

            result_dict = {
                "Pruning Ratio": resulting_channels / 256.0,
                "Resulting Channels": resulting_channels
            }
            result_dict.update(aggregated_scores)
            results.append(result_dict)
            print(result_dict)
            
        df = pd.DataFrame(results)
        df.to_csv(f"pruning_results_{model_name}_{imp_name}_voc.csv", index=False)
    
            # Record the results in the text file
            # results_str = f"{pruning_ratio:.2f},{resulting_channels},{val_loss:.4f},{acc * 100:.2f}\n"
            # results_file.write(results_str)

                # Optionally, print the results to the console
                


# Entry point of the script
if __name__ == "__main__":
    main()
