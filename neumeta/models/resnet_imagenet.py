from typing import Callable, List, Optional, Type, Union
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
try:
    from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv3x3, model_urls, Any
except:
    from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, conv3x3, Any
    
from .utils import load_checkpoint

class BasicBlock_Resize(BasicBlock):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)
    
class ResNet_ImageNet(ResNet):
    def __init__(self, 
                 block, 
                 hidden_dim, 
                 layers: List[int], 
                num_classes: int = 1000,
                zero_init_residual: bool = False,
                groups: int = 1,
                width_per_group: int = 64,
                replace_stride_with_dilation: Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None
                 ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.set_changeable(hidden_dim, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                    
    def set_changeable(self, planes, stride):
        for name, child in self.named_children():
        # Change the last block of layer3
            if name == 'layer3':
                print("Replace last block of {} with new block with hidden dim {}".format(name, planes))
                layers = list(child.children())[:-1]
                # layers.append(BasicBlock_Resize(64, planes, stride))
                layers.append(BasicBlock_Resize(256, planes, stride))
                # Update layer3 with the new layers
                self._modules[name] = nn.Sequential(*layers)
    
    @property
    def learnable_parameter(self):
        # self.keys = [k for k, w in self.named_parameters() if k.startswith('layer3.2.conv') ] #  ork.startswith('layer3.2')
        self.keys = [k for k, w in self.named_parameters() if k.startswith('layer3.1') ] # or k.startswith('layer3.1') or k.startswith('layer3.0')
        return {k: v for k, v in self.state_dict().items() if k in self.keys}
    

def _resnet(
    arch: str,
    hidden_dim: int, 
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet_ImageNet(block, hidden_dim, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        load_checkpoint(model, state_dict)
        
    return model


def resnet18_imagenet(hidden_dim=512, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18',hidden_dim, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
    
if __name__ == "__main__":
    model = resnet18_imagenet()
    for name, child in model.named_modules():
        print(name)
    