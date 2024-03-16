from typing import List, Union

import torch
from torch import nn

class DenseLayer(nn.Module):
    def __init__(self, channels: int, k: int, bottleneck_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        out_channels = bottleneck_size*k
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, k, 3, 1, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor): x = [x]
        x = torch.cat(x, 1)
        return self.dense_layer(x)

class DenseBlock(nn.Module):
    def __init__(
        self, 
        num_times: int, 
        channels: int, 
        k: int,
        bottleneck_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(channels=channels+(i*k), k=k, bottleneck_size=bottleneck_size, dropout=dropout) for i in range(num_times)
        ])
            
    def forward(self, x: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        x = [x]
        for layer in self.layers:
            out = layer(x)
            x.append(out)
        x = torch.cat(x, 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition_layer(x)

class DenseNet(nn.Module):
    def __init__(
        self, 
        num_layers: int = 121, 
        num_classes: int = 1000, 
        channels: int = 64, 
        k: int = 32, 
        bottleneck_size: int = 4, 
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_config = {
            121: (6, 12, 24, 16),
            169: (6, 12, 32, 32),
            201: (6, 12, 48, 32),
            264: (6, 12, 64, 48)
        }
        self.block = self.block_config[num_layers]
        
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        for i, num_times in enumerate(self.block):
            self.features.append(DenseBlock(num_times, channels, k, bottleneck_size, dropout))
            channels += k*num_times
            if i != len(self.block) - 1:
                self.features.append(TransitionLayer(channels, channels // 2))
                channels = channels // 2

        self.features.append(nn.BatchNorm2d(channels))
        self.features.append(nn.ReLU(True))

        self.classification_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

        self._init_layers()

    def _init_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classification_layer(x)
        return x
        