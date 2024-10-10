from typing import Any
import torch
import torch_pruning as tp
import fast_tsp
import numpy as np  
import torch.nn as nn


def total_variation_loss_1d(tensor):
    """
    Compute the Total Variation loss for a 1D tensor.
    
    Args:
    - tensor (torch.Tensor): a 1D tensor.
    
    Returns:
    - loss (torch.Tensor): scalar tensor containing the loss.
    """
    return torch.sum(torch.abs(tensor[:-1] - tensor[1:]))

def total_variation_loss_2d(tensor):
    """
    Compute the Total Variation loss for a 2D tensor (image).
    
    Args:
    - tensor (torch.Tensor): a 2D tensor.
    
    Returns:
    - loss (torch.Tensor): scalar tensor containing the loss.
    """
    # Calculate the loss in the x-direction
    loss_x = torch.sum(torch.abs(tensor[:, :-1] - tensor[:, 1:]))
    # Calculate the loss in the y-direction
    loss_y = torch.sum(torch.abs(tensor[:-1, :] - tensor[1:, :]))
    return loss_x + loss_y


def compute_tv_loss_for_network(model, lambda_tv=1e-4):
    """
    Compute the Total Variation loss for the parameters of a neural network.
    
    Args:
    - model (nn.Module): PyTorch model.
    - lambda_tv (float): Weight for the TV loss.
    
    Returns:
    - total_tv_loss (torch.Tensor): scalar tensor containing the total TV loss for the network.
    """
    total_tv_loss = 0.0
    num = 0
    for param in model.parameters():
        if len(param.shape) == 1:  # 1D tensor
            total_tv_loss += total_variation_loss_1d(param)
        elif len(param.shape) == 2 or len(param.shape) == 4:  # 2D tensor
            total_tv_loss += total_variation_loss_2d(param)
        num += 1
        # Note: We're ignoring tensors with more than 2 dimensions (e.g., conv layers in CNNs)
    return lambda_tv * total_tv_loss / num

def precompute_distances(cities_matrix):
    distances = torch.cdist(cities_matrix, cities_matrix, p=1)  # p=1 indicates L1 norm, which is equivalent to Manhattan distance.
    return distances

def precompute_distances(cities_matrix):
    distances = torch.cdist(cities_matrix, cities_matrix, p=1)  # p=1 indicates L1 norm, which is equivalent to Manhattan distance.
    return distances

def shp(cities_matrix):
    K = cities_matrix.shape[1]
    # Add a dummy city to the matrix
    dummy_city = torch.rand(1, K).to(cities_matrix.device)
    cities_matrix_dummpy = torch.cat((dummy_city, cities_matrix), dim=0)
    
    distances = precompute_distances(cities_matrix_dummpy)
    distances[0,:] = 0
    distances[:,0] = 0

    optimized_tour = fast_tsp.find_tour(distances.detach().cpu().numpy())
    # Remove the dummy node from the optimized tour
    dummy_node_index = optimized_tour.index(0)
    optimized_tour = optimized_tour[:dummy_node_index] + optimized_tour[dummy_node_index + 1:]
    
    return np.array(optimized_tour) - 1

def tsp(cities_matrix):
    distances = precompute_distances(cities_matrix)
    optimized_tour = fast_tsp.find_tour(distances.detach().cpu().numpy())
    
    return optimized_tour

def tsp_greedy_nearest_neighbor(cities_matrix):
    distances = precompute_distances(cities_matrix)
    optimized_tour = fast_tsp.greedy_nearest_neighbor(distances.detach().cpu().numpy())
    
    return optimized_tour
    

class PermutationManager:
    OUT_CHANNELS = "out_channels"
    IN_CHANNELS = "in_channels"

    def __init__(self, model, example_tensor, ignored_keys=[]):
        self.model = model
        self.example_tensor = example_tensor
        self.ignored_keys=ignored_keys
        
    def __call__(self, ignored_keys=None) -> Any:
        if ignored_keys is None:
            ignored_keys = self.ignored_keys
        permute_dict = self.compute_permute_dict()
        self.apply_permutations(permute_dict)
        return self.model

    def _check_size(self, params):
        if len(params) == 0:
            return False
        dim = params[0].size(0)
        for param in params:
            if param.size(0) != dim:
                return False
        return True
        
    def compute_permute_dict(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example_tensor)
        permute_dict = {}

        for group in DG.get_all_groups():
            params, metadata = self._process_group(group)

            if not self._check_size(params):
                continue
                
            full_city = torch.cat(params, dim=1)
            order = shp(full_city)
            
            for key, mode in metadata:
                permute_dict.setdefault(key, {})[mode] = order

        return permute_dict

    def _process_group(self, group):
        params, metadata = [], []
        
        for dep, _ in group:
            layer, pruning_fn = dep.layer, dep.pruning_fn.__name__

            for name, param in layer.named_parameters():
                param_key = f"{dep.target._name}.{name}"
                if isinstance(layer, nn.Linear):
                    mode = self._get_pruning_mode(pruning_fn, param)
                    if mode:
                        params.append(self._adjust_param(param, mode))
                        metadata.append((param_key, mode))
                elif isinstance(layer, nn.Conv2d):
                    mode = self._get_pruning_mode(pruning_fn, param)
                    if mode:
                        params.append(self._adjust_param(param, mode))
                        metadata.append((param_key, mode))
                # elif isinstance(layer, nn.ConvTranspose2d):
                #     if param.dim() != 1:
                #         mode = self._get_pruning_mode_transpose(pruning_fn, param)
                #     else:
                #         mode = self._get_pruning_mode(pruning_fn, param)
                        
                #     if mode:
                #         params.append(self._adjust_param(param, mode))
                #         metadata.append((param_key, mode))
                elif isinstance(layer, nn.BatchNorm2d):
                    mode = self._get_pruning_mode(pruning_fn, param)
                    if mode:
                        params.append(self._adjust_param(param, mode))
                        metadata.append((param_key, mode))
                        

        return params, metadata
    
    def _get_pruning_mode_transpose(self, pruning_fn, param):
        if pruning_fn == "prune_out_channels":
            return self.IN_CHANNELS
        elif pruning_fn == "prune_in_channels":
            return self.OUT_CHANNELS
        else:
            return None

    def _get_pruning_mode(self, pruning_fn, param):
        if pruning_fn == "prune_out_channels":
            return self.OUT_CHANNELS
        elif pruning_fn == "prune_in_channels" and param.dim() != 1:
            return self.IN_CHANNELS
        else:
            return None

    def _adjust_param(self, param, mode):
        # Bias Term Output
        if mode == self.OUT_CHANNELS and param.dim() == 1:
            return param.unsqueeze(-1)
        # Linear Weight Term Input
        elif mode == self.IN_CHANNELS and param.dim() == 2:
            return param.permute(1, 0)
        # Linear Weight Term Output
        elif mode == self.OUT_CHANNELS and param.dim() == 2:
            return param
        # Conv2D Weight Term Output
        elif mode == self.OUT_CHANNELS and param.dim() == 4:
            return param.view(param.shape[0], -1)
        # Conv2D Weight Term Input
        elif mode == self.IN_CHANNELS and param.dim() == 4:
            # print(param.shape)
            param = param.permute(1, 0, 2, 3)
            # torch.view(param, param.shape[0], -1)
            param = param.reshape(param.shape[0], -1)
            # print(param.shape)
            return param
        else:
            raise AssertionError(f"Unknown mode {mode} for param {param.shape}")

    def apply_permutations(self, permute_dict, ignored_keys=[]):
        for name, param in self.model.named_parameters():
            if name in permute_dict:
                for mode, order in permute_dict[name].items():
                    if (name, mode) not in ignored_keys:
                        # print(f"Permuting {name} with mode {mode} and order {order}")
                        self._permute_param(param, mode, order)
        return self.model

    def _permute_param(self, param, mode, order):
        # Bias Term Output
        if mode == self.OUT_CHANNELS and param.dim() == 1:
            param.data.copy_(param.data[order])
        # Linear Weight Term Input
        elif mode == self.IN_CHANNELS and param.dim() == 2:
            param.data.copy_(param.data[:, order])
        # Linear Weight Term Output
        elif mode == self.OUT_CHANNELS and param.dim() == 2:
            param.data.copy_(param.data[order])
        # Conv2D Weight Term Input
        elif mode == self.IN_CHANNELS and param.dim() == 4:
            param.data.copy_(param.data[:, order, :, :])
        # Conv2D Weight Term Input
        elif mode == self.OUT_CHANNELS and param.dim() == 4:
            param.data.copy_(param.data[order, :, :, :])
        else:
            raise AssertionError(f"Unknown mode {mode} for param {param}")

class PermutationManager_SingleTSP(PermutationManager):
    def compute_permute_dict(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example_tensor)
        permute_dict = {}

        for group in DG.get_all_groups():
            params, metadata = self._process_group(group)
            full_city = params[0]
            order = tsp(full_city)
            
            for key, mode in metadata:
                permute_dict.setdefault(key, {})[mode] = order

        return permute_dict
    
class PermutationManager_SingleTSP_Greedy(PermutationManager):
    def compute_permute_dict(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example_tensor)
        permute_dict = {}

        for group in DG.get_all_groups():
            params, metadata = self._process_group(group)
            full_city = params[0]
            order = tsp_greedy_nearest_neighbor(full_city)
            
            for key, mode in metadata:
                permute_dict.setdefault(key, {})[mode] = order

        return permute_dict
    
class PermutationManager_Random(PermutationManager):
    def compute_permute_dict(self):
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=self.example_tensor)
        permute_dict = {}

        for group in DG.get_all_groups():
            params, metadata = self._process_group(group)
            # random order
            order = np.random.permutation(params[0].shape[0])
            
            for key, mode in metadata:
                permute_dict.setdefault(key, {})[mode] = order

        return permute_dict