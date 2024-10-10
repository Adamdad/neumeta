import torch
import torch.nn as nn

import torch_pruning as tp
import fast_tsp
import numpy as np  
import torch.optim as optim
from smooth.permute import PermutationManager

def precompute_distances(cities_matrix):
    # Precompute the distance matrix based on the city coordinates.
    N, K = cities_matrix.shape
    distances = torch.zeros(N, N)
    for i in range(N):
        for j in range(i+1, N):
            distance = torch.abs(cities_matrix[i] - cities_matrix[j]).sum()
            distances[i][j] = distances[j][i] = distance
    return distances

def shp(cities_matrix):
    K = cities_matrix.shape[1]
    # Add a dummy city to the matrix
    dummy_city = torch.rand(1, K)
    cities_matrix_dummpy = torch.cat((dummy_city, cities_matrix), dim=0)
    
    distances = precompute_distances(cities_matrix_dummpy)
    distances[0,:] = 0
    distances[:,0] = 0

    optimized_tour = fast_tsp.find_tour(distances.detach().cpu().numpy())
    # Remove the dummy node from the optimized tour
    dummy_node_index = optimized_tour.index(0)
    optimized_tour = optimized_tour[:dummy_node_index] + optimized_tour[dummy_node_index + 1:]
    
    return np.array(optimized_tour) - 1

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
    for param in model.parameters():
        if len(param.shape) == 1:  # 1D tensor
            total_tv_loss += total_variation_loss_1d(param)
        elif len(param.shape) == 2 or len(param.shape) == 4:  # 2D tensor
            total_tv_loss += total_variation_loss_2d(param)
        # Note: We're ignoring tensors with more than 2 dimensions (e.g., conv layers in CNNs)
    return lambda_tv * total_tv_loss

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_hidden=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        for i in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ResMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_hidden=1):
        super(ResMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.layers = []
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))
        for i in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = layer(x) + x
            
        return self.layers[-1](x)
    
def iris_example():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    iris = datasets.load_iris()
    # datasets.load_wine()
    X = iris.data
    y = iris.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)
    
    # Parameters
    input_dim = 4
    output_dim = 3
    hidden_dim = 64
    n_hidden = 1
    learning_rate = 0.01
    epochs = 100

    # Model, Loss and Optimizer
    model = ResMLP(input_dim, output_dim, hidden_dim, n_hidden)
    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation Before Training:", total_tv)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    all_keys = [k for k, w in model.named_parameters()]
    # print(all_keys)
    

    # Training loop
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Training:", total_tv)
    test(model, X_test, y_test)
    
    input_tensor = torch.randn(1, input_dim)
    permute_func = PermutationManager(model, input_tensor)
    permute_dict = permute_func.compute_permute_dict()
    model = permute_func.apply_permutations(permute_dict, ignored_keys=[('layers.0.weight', 'in_channels'),('layers.4.weight', 'out_channels'), ('layers.4.bias', 'out_channels')])
    total_tv = compute_tv_loss_for_network(model, lambda_tv=1.0)
    print("Total Total Variation After Permute:", total_tv)
    
    test(model, X_test, y_test)
    
    
    

def test(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

    print(f'Accuracy on test data: {100 * correct / total}%')

def compute_permute_dict(model, example_tensor):
    # 1. Build the DepGraph with an example input.
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_tensor)
    permute_dict = {}
    for g in DG.get_all_groups():
        
        full_city = []
        temp_dict = {}
        for i, (dep, idxs) in enumerate(g):
            layer = dep.layer
            pruning_fn = dep.pruning_fn.__name__
            
            if isinstance(layer, nn.Linear):
                for k, param in layer.named_parameters():
                    param_key = dep.target._name + "." + k
                    if pruning_fn == "prune_out_channels":
                        if param.dim() == 2:
                            full_city.append(param)
                        elif param.dim() == 1:
                            full_city.append(param.unsqueeze(-1))
                        temp_dict[param_key] = "out_channels"
                    elif pruning_fn == "prune_in_channels" and param.dim() == 2:
                        full_city.append(param.permute(1,0))
                        temp_dict[param_key] = "in_channels"
            
        full_city = torch.cat(full_city, dim=1)
        order = shp(full_city)
        for k in temp_dict:
            mode = temp_dict[k]
            if k in permute_dict:
                permute_dict[k][mode] = order
            else:
                permute_dict[k] = {mode:order}
    return permute_dict

def permute_with_orders(model, permute_dict, ignored_keys=[('layers.0.weight', 'in_channels'),('layers.4.weight', 'out_channels'), ('layers.4.bias', 'out_channels')]):
    for k, param in model.named_parameters():
        if k in permute_dict:
            for mode, order in permute_dict[k].items():
                if (k, mode) in ignored_keys:
                    continue
                elif mode == "out_channels":
                    param.data.copy_(param.data[order])
                elif mode == "in_channels":
                    param.data.copy_(param.data[:,order])
    return model

if __name__ == '__main__':
    iris_example()
    
    
    
        
        
    