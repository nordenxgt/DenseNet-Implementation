import torch
from torch import nn

class DenseBlock(nn.Module):
    def __init__(self, channels: int, k: int, bottleneck_size: int, dropout: float = None) -> None:
        super().__init__()
        growth_channels =  k*bottleneck_size
        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, growth_channels, 1)
        )
        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(growth_channels),
            nn.ReLU(True),
            nn.Conv2d(growth_channels, k, 1, 2, 1)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bnorm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.avgpool(x)
        return x

class DenseNet(nn.Module):
    def __init__(
        self, 
        num_layers: int = 121, 
        num_classes: int = 1000, 
        channels: int = 112, 
        k: int = 32, 
        bottleneck_size: int = 4, 
        dropout: float = None
    ) -> None:
        super().__init__()
        self.block_config = {
            121: [6, 12, 24, 16],
            169: [6, 12, 32, 32],
            201: [6, 12, 48, 32],
            264: [6, 12, 64, 48]
        }
        self.block = self.block_config[num_layers]
        
        self.layers = []
        self.layers.append(nn.Conv2d(3, channels, 7, 2, 1))
        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(3, 2))

        for i, num_times in enumerate(self.block):
            for _ in range(num_times):
                self.layers.append(
                    DenseBlock(
                        channels=channels+(i*k),
                        k=k,
                        bottleneck_size=bottleneck_size,
                        dropout=dropout
                    )
                )
            channels += k*num_times  
            if i != len(self.block) - 1:
                self.layers.append(TransitionLayer(channels, channels // 2))
                channels = channels // 2

        self.layers.append(nn.BatchNorm2d(channels))
        self.layers.append(nn.ReLU(True))

        self.features = nn.Sequential(*self.layers)
        self.classification_layer = nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Linear(1, num_classes)
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
        