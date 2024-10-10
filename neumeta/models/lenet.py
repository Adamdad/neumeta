import torch.nn as nn
from torch.nn import functional as F
import torch

class MnistNet(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            1, hidden_dim, 3, padding=1, bias=True, padding_mode="replicate"
        )
        self.conv_2 = nn.Conv2d(
            hidden_dim, hidden_dim*2, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.conv_3 = nn.Conv2d(
            hidden_dim*2, hidden_dim*4, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.f_1 = nn.ReLU()
        self.f_2 = nn.ReLU()
        self.f_3 = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(hidden_dim*4, 10)

    def forward(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        x = self.f_2(self.conv_2(x))
        x = self.pool(x)
        x = self.f_3(self.conv_3(x))
        x = self.pool(x)
        x = self.linear(x[:, :, 0, 0])

        return x
    
    @property
    def learnable_parameter(self):
        self.keys = [k for k, w in self.named_parameters()]
        return self.state_dict()
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class MnistResNet(nn.Module):
    def __init__(self, num_blocks, hidden_dim=8, block=BasicBlock, num_classes=10):
        super().__init__()
        self.in_channels = hidden_dim
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, hidden_dim, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, hidden_dim * 2, num_blocks[1], stride=2)
        self.linear = nn.Linear(hidden_dim * 2 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    @property
    def learnable_parameter(self):
        self.keys = [k for k, w in self.named_parameters()]
        return {k:w for k, w in self.state_dict().items() if k in self.keys}
    
    
def test_MnistResNet():
    # Create a model with 2 blocks and hidden dimension of 16
    model = MnistResNet(num_blocks=[2, 2], hidden_dim=8)

    # Create a random input tensor of size (batch_size, channels, height, width)
    input_tensor = torch.randn((1, 1, 28, 28))

    # Test the forward pass of the model
    output = model(input_tensor)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"
    print("Success!")
    
if __name__ == '__main__':

    test_MnistResNet()