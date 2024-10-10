

import os
import random
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import wandb

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin

from neumeta.models.resnet_imagenet import BasicBlock_Resize
from neumeta.models.utils import load_checkpoint as load_checkpoint_model, fuse_module
from neumeta.segmentation.losses import ComboLoss
from neumeta.segmentation.train_voc_baseline import evaluate_model, create_dataset_voc, visualize_single_model
from neumeta.segmentation.utils import visualize_model_wandb
from neumeta.utils import (
    EMA,
    get_hypernet,
    get_optimizer,
    load_checkpoint,
    parse_args,
    sample_coordinates,
    sample_merge_model,
    sample_single_model,
    sample_subset,
    sample_weights,
    save_checkpoint,
    set_seed,
    shuffle_coordiates_all,
    weighted_regression_loss,
)

class MyResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, encoder, out_channels, depth=5, **kwcfg):
        super().__init__(**kwcfg)
        self._depth = encoder._depth
        self._out_channels = encoder._out_channels
        self._in_channels = 3
        self.set_changeable(out_channels, stride=1)
        # self.keys = ['layer3.1.conv1.weight', 'layer3.1.conv1.bias', 'layer3.1.conv2.weight', 'layer3.1.conv2.bias']
        
        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwcfg):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwcfg)
    
    def set_changeable(self, planes, stride):
        for name, child in self.named_children():
        # Change the last block of layer3
            if name == 'layer3':
                print("Replace last block of layer4 with new block with hidden dim {}".format(planes))
                layers = list(child.children())[:-1]
                # layers.append(BasicBlock_Resize(64, planes, stride))
                layers.append(BasicBlock_Resize(256, planes, stride))
                # Update layer3 with the new layers
                self._modules[name] = nn.Sequential(*layers)
    
class Unet(smp.Unet):
    
    @property
    def learnable_parameter(self):
        return {k: v for k, v in self.state_dict().items() if k in self.keys}
    
    @property
    def keys(self):
        return [k for k, w in self.named_parameters() if k.startswith('encoder.layer3.1')]
    
# Define visualization function
def get_unet_model(cfg, hidden_dim=256):
    model = Unet(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.num_classes,
    )


    model.encoder = MyResNetEncoder(model.encoder, hidden_dim, block=BasicBlock, layers=[2, 2, 2, 2])
    return model

def init_model_dict(cfg, device):
    """
    Initializes a dictionary of models for each dimension in the given range, along with ground truth models for the starting dimension.

    cfg:
        cfg: An object containing the arguments for initializing the models.

    Returns:
        dim_dict: A dictionary containing the models for each dimension, along with their corresponding coordinates, keys, indices, size, and ground truth models.
        gt_model_dict: A dictionary containing the ground truth models for the starting dimension.
    """
    dim_dict = {}
    gt_model_dict = {}
    for dim in cfg.dimensions.range:
        model = get_unet_model(cfg, hidden_dim=dim).to(device)
        state_dict = torch.load(cfg.model.pretrained_path, map_location='cpu')['state_dict']
        load_checkpoint_model(model, state_dict)
        fuse_module(model)
        # print(model.keys)
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
        dim_dict[f"{dim}"] = (model, coords_tensor, keys_list, indices_list, size_list, None)
        if dim == cfg.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = get_unet_model(cfg, hidden_dim=dim).to(device)
            state_dict = torch.load(cfg.model.pretrained_path, map_location='cpu')['state_dict']
            model_trained.load_state_dict(state_dict)
            fuse_module(model_trained)
            # print(model.keys)
            model_trained.eval()
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def train_one_step(hyper_model, device, x, target, optimizer, scheduler, criterion, dim_dict, gt_model_dict, epoch_idx, ema=None, cfg=None):
        optimizer.zero_grad()
        hidden_dim = random.choice(cfg.dimensions.range)
        model, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        if cfg.ratio < 1.0:
            coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor, 
                                                                                            keys_list, 
                                                                                            indices_list, 
                                                                                            size_list, 
                                                                                            key_mask, 
                                                                                            ratio=cfg.ratio)
        else:
            selected_keys = list(key_mask.keys())
            
            
        if cfg.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * cfg.training.coordinate_noise
        model, reconstructed_weights = sample_weights(hyper_model, model, 
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,  
                                                          device=device, NORM=cfg.dimensions.norm)
        output = model(x)

        # Calculate the mean squared error loss
        task_loss = criterion(output, target) 
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])
        
        
        reconstruct_loss = 0.0
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(reconstructed_weights,gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        loss =  cfg.hyper_model.loss_weight.ce_weight * task_loss + cfg.hyper_model.loss_weight.reg_weight * reg_loss + cfg.hyper_model.loss_weight.recon_weight * reconstruct_loss

        for updated_weight in model.parameters():
            updated_weight.grad = None

        loss.backward(retain_graph=True)
        torch.autograd.backward(reconstructed_weights, [w.grad for k, w in model.named_parameters() if k in selected_keys])
        
        if cfg.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(hyper_model.parameters(), cfg.training.clip_grad)
        
        optimizer.step()
        scheduler.step()
        if ema:
            ema.update()  # Update the EMA after each training step
        
        return task_loss, reg_loss, reconstruct_loss

def train_and_evaluate(cfg):
    wandb.init(project=cfg.wandb.project, config=dict(cfg), name=cfg.wandb.run_name)

    set_seed(cfg.experiment.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, num_class = create_dataset_voc(cfg, strong_aug=cfg.training.get('strong_aug', False))

    model = get_unet_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(cfg.model.pretrained_path, map_location='cpu')['state_dict'])
    fuse_module(model)
        
    # print(model.state_dict().keys())
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(cfg, number_param, device=DEVICE)
    if cfg.load_from:
        print(f"Load from checkpoint: {cfg.load_from}")
        checkpoint_info = load_checkpoint(cfg.load_from, hyper_model, None, None)
        
    if cfg.hyper_model.ema_decay > 0:
        ema = EMA(hyper_model, decay=cfg.hyper_model.ema_decay)
    else:
        ema = None
    criterion, _, optimizer, scheduler = get_optimizer(cfg, hyper_model)
        
    
    # criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion = ComboLoss(ignore_index=255)

    dim_dict, gt_model_dict = init_model_dict(cfg, DEVICE)
    dim_dict = shuffle_coordiates_all(dim_dict)
    
    if cfg.resume_from:
        print(f"Resuming from checkpoint: {cfg.resume_from}")
        checkpoint_info = load_checkpoint(cfg.resume_from, hyper_model, optimizer, ema)
        # Note: If there are more elements to retrieve, do so here.
        
    # Log model, optimizer, and scheduler to wandb
    wandb.watch(hyper_model, criterion, log='all', log_freq=cfg.training.log_every)

    iteration = 0
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    while iteration < cfg.training.max_iterations:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if iteration >= cfg.training.max_iterations:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)

            task_loss, reg_loss, reconstruct_loss = train_one_step(hyper_model, DEVICE, data, target, optimizer, scheduler, criterion, dim_dict, gt_model_dict, iteration, ema, cfg)
            iteration += 1

            if iteration % cfg.training.log_every == 0:
                # Log training loss to wandb
                wandb.log({"train_loss": task_loss.item(),
                            "train_reg_loss": reg_loss.item(),
                            "train_recon_loss": reconstruct_loss.item(),
                          "learning_rate": scheduler.get_last_lr()[0]})
                print('Iteration {}/{}. Loss: {:.6f}, Reg Loss: {:.6f}, Recon Loss: {:.6f}, LR: {:.6f}'.format(iteration,
                      cfg.training.max_iterations, task_loss.item(), reg_loss.item(), reconstruct_loss.item(),
                        scheduler.get_last_lr()[0]))

            if iteration % cfg.training.eval_every == 0:
                # save checkpoint
                if ema:
                    ema.apply()
                    
                if cfg.training.coordinate_noise > 0.0:
                    model = sample_merge_model(hyper_model=hyper_model, model=model, args=cfg, device=DEVICE)
                else:
                    model = sample_single_model(hyper_model=hyper_model, model=model, device=DEVICE)
                    
                if ema:
                    ema.restore()

                aggregated_scores = evaluate_model(model, valid_loader, DEVICE)

                # save_checkpoint(
                #     model=hyper_model,
                #     optimizer=optimizer,
                #     epoch=iteration,
                #     scores=aggregated_scores,
                #     checkpoint_dir=cfg.training.checkpoint_dir
                # )
                save_checkpoint(
                        f"{cfg.training.checkpoint_dir}/checkpoint.pth.tar",
                        hyper_model,
                        optimizer,
                        ema,
                        iteration,
                        aggregated_scores
                    )
                
                wandb.log(aggregated_scores)
                visualize_model_wandb(model, valid_loader, DEVICE)

                print("Validation scores at iteration {}: {}".format(
                    iteration, aggregated_scores))

def test(cfg):
    import pandas as pd

    set_seed(cfg.experiment.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, num_class = create_dataset_voc(cfg, strong_aug=cfg.training.get('strong_aug', False))

    
    model = get_unet_model(cfg).to(DEVICE)
    model.load_state_dict(torch.load(cfg.model.pretrained_path, map_location='cpu')['state_dict'])
    fuse_module(model)
    visualize_single_model(model, valid_loader, model_name=f'Original',k=64, DEVICE=DEVICE)
    # exit()
        
    # print(model.state_dict().keys())
    checkpoint = model.learnable_parameter
    # print(checkpoint)
    number_param = len(checkpoint)
    print(f"Parameters keys: {model.keys}")
    print(f"Number of parameters to be learned: {number_param}")
    hyper_model = get_hypernet(cfg, number_param, device=DEVICE)
    
        
    if cfg.hyper_model.ema_decay > 0:
        ema = EMA(hyper_model, decay=cfg.hyper_model.ema_decay)
    else:
        ema = None
        
    if cfg.load_from:
        print(f"Load from checkpoint: {cfg.load_from}")
        checkpoint_info = load_checkpoint(cfg.load_from, hyper_model, None, ema)
    
    results = []
    for dim in range(64, 257, 64):
        
        set_seed(cfg.experiment.seed)
        model = get_unet_model(cfg).to(DEVICE)
        state_dict = torch.load(cfg.model.pretrained_path, map_location='cpu')['state_dict']
        load_checkpoint_model(model, state_dict)
        fuse_module(model)
        print("Evaluate model with hidden dim {}".format(dim))
        
        if ema:
            ema.apply()
            
        if cfg.training.coordinate_noise > 0.0:
            model = sample_merge_model(hyper_model=hyper_model, model=model, args=cfg, device=DEVICE, K=100)
        else:
            model = sample_single_model(hyper_model=hyper_model, model=model, device=DEVICE)
            
        if ema:
            ema.restore()
            
        visualize_single_model(model, valid_loader, model_name=f'channel={dim}',k=64, DEVICE=DEVICE)
        
        
    #     aggregated_scores = evaluate_model(model, valid_loader, DEVICE)
    #     print("Validation scores at iteration {}: {}".format(
    #         dim, aggregated_scores))
    #     result_dict = {
    #         "Pruning Ratio": dim / 256.0,
    #         "Resulting Channels": dim
    #     }
    #     result_dict.update(aggregated_scores)
    #     results.append(result_dict)
    
    # df = pd.DataFrame(results)
    # df.to_csv(f"neumeta_results_unet_voc.csv", index=False)
        

if __name__ == '__main__':
    config = parse_args()
    # create cfg.wandb.run_name from config name
    config.wandb.run_name = config.config.split('/')[-1].split('.')[0]
    # Start the training and evaluation process
    if config.test:
        test(config)
    else:
        train_and_evaluate(config)
