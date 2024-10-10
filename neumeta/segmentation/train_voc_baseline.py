import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from omegaconf import OmegaConf
from tqdm import tqdm

import segmentation_models_pytorch as smp

from neumeta.models.resnet_imagenet import BasicBlock_Resize
from neumeta.segmentation.dataset.voc import VOCSegmentation
from neumeta.segmentation.losses import ComboLoss
from neumeta.segmentation.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    PILToTensor,
    RandomCrop,
    RandomGaussianBlur,
    RandomHorizontalFlip,
    RandomResize,
    Resize,
    ToDtype,
)
from neumeta.segmentation.utils import (
    create_pascal_voc_colormap,
    save_checkpoint,
    visualize,
    visualize_model_wandb,
)

# Define evaluation function


def evaluate_model(model, dataloader, device, num_classes=21):
    model.eval()
    scores = {
        "accuracy": [],
        "iou": [],
        "f1": []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            # convert mask to one hot vector
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)

            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs, masks, mode='multiclass', ignore_index=255, num_classes=num_classes)
            iou_score = smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")

            scores["accuracy"].append(accuracy.cpu().item())
            scores["iou"].append(iou_score.cpu().item())
            scores["f1"].append(f1_score.cpu().item())

    aggregated_scores = {metric: sum(values)/len(values)
                         for metric, values in scores.items()}
    return aggregated_scores

# Define visualization function

def visualize_single_model(model, valid_loader, model_name, k=5, DEVICE='cuda'):
    # Define the PASCAL VOC colormap
    VOC_COLORMAP, _ = create_pascal_voc_colormap()
    data_vis = valid_loader.dataset
    model.eval()
    with torch.no_grad():
        for i in range(k):
            n = np.random.choice(len(data_vis))
            image_vis = cv2.imread(data_vis.images[n])
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            image, gt_mask = data_vis[n]

            gt_mask = gt_mask.cpu().numpy()
            x_tensor = image.to(DEVICE).unsqueeze(0)
            pr_mask = model(x_tensor)
            pr_mask = torch.argmax(
                pr_mask.squeeze(), dim=0).detach().cpu().numpy()

            visualize(
                color_map=VOC_COLORMAP,
                name=str(i)+model_name,
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask
            )
    

def visualize_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, num_class = create_dataset_voc()
    # model = create_model(num_classes=num_class).to(DEVICE)
    model = smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # use `imagenet` pre-trained weights for encoder initialization
        encoder_weights="imagenet",
        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        in_channels=3,
        # model output channels (number of classes in your dataset)
        classes=num_class,
    ).to(DEVICE)

    model.load_state_dict(torch.load('best_model.pth'))

    # Define the PASCAL VOC colormap
    VOC_COLORMAP, _ = create_pascal_voc_colormap()
    data_vis = valid_loader.dataset
    model.eval()
    with torch.no_grad():
        for i in range(5):
            n = np.random.choice(len(data_vis))
            image_vis = cv2.imread(data_vis.images[n])
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            image, gt_mask = data_vis[n]

            gt_mask = gt_mask.cpu().numpy()
            x_tensor = image.to(DEVICE).unsqueeze(0)
            pr_mask = model(x_tensor)
            pr_mask = torch.argmax(
                pr_mask.squeeze(), dim=0).detach().cpu().numpy()

            visualize(
                color_map=VOC_COLORMAP,
                name=str(i),
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask
            )


def create_dataset_voc(cfg, strong_aug=False):
    # Use the transformations from the configuration
    if strong_aug:
        print("Using strong augmentation")
        train_transform = Compose([
        RandomResize(cfg.transforms.resize[0], cfg.transforms.resize[1]),
        RandomCrop(cfg.transforms.random_crop),
        RandomHorizontalFlip(flip_prob=cfg.transforms.flip_prob),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        RandomGaussianBlur(radius_range=(1, 3)),
        PILToTensor(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
        ])
        
    else:
        train_transform = Compose([
            RandomResize(cfg.transforms.resize[0], cfg.transforms.resize[1]),
            RandomCrop(cfg.transforms.random_crop),
            RandomHorizontalFlip(flip_prob=cfg.transforms.flip_prob),
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
        ])

    val_transform = Compose([
        Resize(cfg.transforms.random_crop),
        PILToTensor(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
    ])

    # Use dataset paths and other settings from the configuration
    train_dataset = VOCSegmentation(
        root=cfg.dataset.root,
        year=cfg.dataset.year_train,
        image_set=cfg.dataset.image_set_train,
        transforms=train_transform,
        download=False
    )
    valid_dataset = VOCSegmentation(
        root=cfg.dataset.root,
        year=cfg.dataset.year_val,
        transforms=val_transform,
        image_set=cfg.dataset.image_set_val
    )

    # Use DataLoader settings from the configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size_train,
        shuffle=True,
        num_workers=cfg.dataset.num_workers_train
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.dataset.batch_size_val,
        shuffle=False,
        num_workers=cfg.dataset.num_workers_val
    )

    return train_loader, valid_loader, cfg.model.num_classes


def train_and_evaluate(cfg):
    wandb.init(project=cfg.wandb.project, config=dict(cfg), name=cfg.wandb.run_name)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, num_class = create_dataset_voc(cfg, strong_aug=cfg.training.get('strong_aug', False))
    if cfg.model.name == "unet":
        model = smp.Unet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.num_classes,
        ).to(DEVICE)
    elif cfg.model.name == "pspnet":
        model = smp.PSPNet(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.num_classes,
        ).to(DEVICE)
    else:
        raise ValueError("Model name not supported")
    
    if cfg.model.get("set_width", None) is not None:
        planes = int(cfg.model.set_width * 256)
        for name, child in model.encoder.named_children():
            print(name)
            # Change the last block of layer3
            if name == 'layer3':
                print("Replace last block of layer4 with new block with hidden dim {}".format(planes))
                layers = list(child.children())[:-1]
                layers.append(BasicBlock_Resize(256, planes, 1))
                # Update layer3 with the new layers
                model._modules[name] = nn.Sequential(*layers)
    else:
        print("No width multiplier specified. Using default model")
                
    # model.load_state_dict(torch.load('best_model.pth'))
    if cfg.training.get("checkpoint_path", None) is not None:
        checkpoint = torch.load(cfg.training.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded checkpoint '{}' (epoch {})".format(
            cfg.training.checkpoint_path, checkpoint['epoch']))

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.max_iterations, eta_min=1e-6
    )
    # criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion = ComboLoss(ignore_index=255)

    # Log model, optimizer, and scheduler to wandb
    wandb.watch(model, criterion, log='all', log_freq=cfg.training.log_every)

    iteration = 0
    while iteration < cfg.training.max_iterations:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if iteration >= cfg.training.max_iterations:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # grad clip
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip, norm_type=2)

            optimizer.step()
            scheduler.step()
            iteration += 1

            if iteration % cfg.training.log_every == 0:
                # Log training loss to wandb
                wandb.log({"train_loss": loss.item(),
                          "learning_rate": scheduler.get_last_lr()[0]})
                print('Iteration {}/{}. Loss: {:.3f}, LR: {:.6f}'.format(iteration,
                      cfg.training.max_iterations, loss.item(), scheduler.get_last_lr()[0]))

            if iteration % cfg.training.eval_every == 0:
                # save checkpoint
                aggregated_scores = evaluate_model(model, valid_loader, DEVICE)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=iteration,
                    scores=aggregated_scores,
                    checkpoint_dir=cfg.training.checkpoint_dir
                )
                wandb.log(aggregated_scores)
                visualize_model_wandb(model, valid_loader, DEVICE)

                print("Validation scores at iteration {}: {}".format(
                    iteration, aggregated_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and evaluating segmentation model.')
    parser.add_argument('--config', type=str,
                        help='Path to the config file.', required=True)
    args = parser.parse_args()
    
    # Load the configuration
    config_path = args.config
    config = OmegaConf.load(config_path)
    config.wandb.run_name = config_path.split('/')[-1].split('.')[0]
    # Start the training and evaluation process
    train_and_evaluate(config)
