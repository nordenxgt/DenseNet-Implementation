import torch
from torch import nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(), 
            nn.BatchNorm2d(out_channels)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, 1),
            nn.ReLU(), 
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3x3(self.conv1x1(x))

class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.conv1x1(x))       