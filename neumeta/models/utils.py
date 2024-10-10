import torch
import torch.nn as nn
# test()

def fuse_conv_bn(conv, bn):
    """
    Fuse convolution and batch normalization layers
    """
    # Extract conv layer parameters
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)

    # Extract bn layer parameters
    bn_rm = bn.running_mean
    bn_rv = bn.running_var
    bn_eps = bn.eps
    bn_w = bn.weight
    bn_b = bn.bias

    # Calculate fused parameters
    inv_var = torch.rsqrt(bn_rv + bn_eps)
    bn_w_div_var = bn_w * inv_var
    bn_bias_sub_rm_w_div_var = bn_b - bn_rm * bn_w_div_var

    fused_conv_weight = conv_w * bn_w_div_var.view(-1, 1, 1, 1)
    fused_conv_bias = conv_b * bn_w_div_var + bn_bias_sub_rm_w_div_var

    # Create and return the fused layer
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True
    )
    fused_conv.weight = nn.Parameter(fused_conv_weight)
    fused_conv.bias = nn.Parameter(fused_conv_bias)

    return fused_conv

def fuse_conv_transpose_bn(conv_transpose, bn):
    """
    Fuse ConvTranspose2d and BatchNorm2d layers.
    """
    # Extract conv transpose layer parameters
    conv_transpose_w = conv_transpose.weight
    conv_transpose_b = conv_transpose.bias if conv_transpose.bias is not None else torch.zeros_like(bn.running_mean)

    # Extract bn layer parameters
    bn_rm = bn.running_mean
    bn_rv = bn.running_var
    bn_eps = bn.eps
    bn_w = bn.weight
    bn_b = bn.bias

    # Calculate fused parameters
    inv_var = torch.rsqrt(bn_rv + bn_eps)
    bn_w_div_var = bn_w * inv_var
    bn_bias_sub_rm_w_div_var = bn_b - bn_rm * bn_w_div_var

    # print(conv_transpose_w.shape, bn_w_div_var.shape)
    fused_conv_transpose_weight = conv_transpose_w * bn_w_div_var.view(1, -1, 1, 1)
    fused_conv_transpose_bias = conv_transpose_b * bn_w_div_var + bn_bias_sub_rm_w_div_var

    # Create and return the fused layer
    fused_conv_transpose = nn.ConvTranspose2d(
        conv_transpose.in_channels,
        conv_transpose.out_channels,
        conv_transpose.kernel_size,
        conv_transpose.stride,
        conv_transpose.padding,
        conv_transpose.output_padding,
        groups=conv_transpose.groups,
        dilation=conv_transpose.dilation,
        bias=True
    )
    fused_conv_transpose.weight = nn.Parameter(fused_conv_transpose_weight)
    fused_conv_transpose.bias = nn.Parameter(fused_conv_transpose_bias)

    return fused_conv_transpose


def fuse_module(module):
    """
    Recursively fuse all batch normalization layers in the module with their preceding convolutional layers.
    """
    children = list(module.named_children())
    prev_name, prev_module = None, None

    for name, child in children:
        # print(name)
        if isinstance(child, nn.BatchNorm2d) and isinstance(prev_module, nn.Conv2d):
            # Fuse the conv and bn layers, replace the conv layer with the fused layer
            fused_conv = fuse_conv_bn(prev_module, child)
            module._modules[prev_name] = fused_conv

            # Remove the batch normalization layer
            module._modules[name] = nn.Identity()
        elif isinstance(child, nn.BatchNorm2d) and isinstance(prev_module, nn.ConvTranspose2d):
            # Fuse the conv and bn layers, replace the conv layer with the fused layer
            fused_conv = fuse_conv_transpose_bn(prev_module, child)
            module._modules[prev_name] = fused_conv

            # Remove the batch normalization layer
            module._modules[name] = nn.Identity()
        else:
            # Recursively apply to all submodules
            fuse_module(child)

        prev_name, prev_module = name, child

def bn_to_conv(bn):
    """
    Convert a batch normalization layer into an equivalent conv2d layer.
    """
    # Batch Norm layer parameters
    bn_rm = bn.running_mean
    bn_rv = bn.running_var
    bn_eps = bn.eps
    bn_w = bn.weight
    bn_b = bn.bias

    # Calculate equivalent weights and bias for the conv layer
    inv_var = torch.rsqrt(bn_rv + bn_eps)
    bn_w_div_var = bn_w * inv_var
    bn_bias_sub_rm_w_div_var = bn_b - bn_rm * bn_w_div_var

    # The new conv layer will have a single 1x1 kernel for each output of the batch norm
    num_features = bn.num_features

    # Create identity-like weight with proper scaling from batchnorm
    conv_weight = torch.zeros((num_features, num_features, 1, 1), dtype=bn_w.dtype, device=bn_w.device)
    for i in range(num_features):
        conv_weight[i, i, 0, 0] = bn_w_div_var[i]

    conv_bias = bn_bias_sub_rm_w_div_var

    # Create and return the equivalent conv2d layer. This time, we do not use groups.
    converted_conv = nn.Conv2d(num_features, num_features, kernel_size=1, bias=True)
    converted_conv.weight = nn.Parameter(conv_weight)
    converted_conv.bias = nn.Parameter(conv_bias)

    return converted_conv

def replace_bn_with_conv(module):
    """
    Recursively traverse the module and replace all batch normalization layers with their equivalent conv2d layers.
    """
    children = list(module.named_children())
    prev_name, prev_module = None, None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            # Convert the bn layer to a conv layer and replace it
            converted_conv = bn_to_conv(child)
            module._modules[name] = converted_conv
        else:
            # Recursively apply to all submodules
            replace_bn_with_conv(child)

        prev_name, prev_module = name, child
        
def load_checkpoint(model, checkpoint_path, prefix='module.'):
    """
    Load model weights from a checkpoint file. This function handles checkpoints that may contain keys
    that are either redundant, prefixed, or absent in the model's state_dict.

    :param model: Model instance for which the weights will be loaded.
    :param checkpoint_path: Path to the checkpoint file.
    :param prefix: Optional string to handle prefixed state_dict, common in models trained using DataParallel.
    :return: Model with state dict loaded from the checkpoint.
    """
    # Load the checkpoint.
    if isinstance(checkpoint_path, str):
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = checkpoint_path

    # If the state_dict is wrapped inside a dictionary under the 'state_dict' key, unpack it.
    state_dict = checkpoint.get('state_dict', checkpoint)

    # If the state_dict keys contain a prefix, remove it.
    if list(state_dict.keys())[0].startswith(prefix):
        state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    # Retrieve the state_dict of the model.
    model_state_dict = model.state_dict()

    # Prepare a new state_dict to load into the model, ensuring that only keys that are present in the model
    # and have the same shape are included.
    updated_state_dict = {
        key: value for key, value in state_dict.items()
        if key in model_state_dict and value.shape == model_state_dict[key].shape
    }
    
    # Update the original model state_dict.
    model_state_dict.update(updated_state_dict)
    # for key, value in model_state_dict.items():
    #     print(f"Updated {key}: {value.shape}")

    # Load the updated state_dict into the model.
    model.load_state_dict(model_state_dict)

    return model