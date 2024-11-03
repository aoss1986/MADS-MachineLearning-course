import torch
import torch.nn as nn
import gin
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

@gin.configurable
class MyCNN(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int, dropout_rate: float, maxpoolwindow: int) -> None:
        super(MyCNN, self).__init__()

        self.layer1 = self._make_layer(1, 64, kernel_size, maxpoolwindow, dropout_rate)
        self.layer2 = self._make_layer(64, 128, kernel_size, maxpoolwindow, dropout_rate)
        self.layer3 = self._make_layer(128, 256, kernel_size, maxpoolwindow, dropout_rate)
        self.layer4 = self._make_layer(256, 512, kernel_size, maxpoolwindow, dropout_rate)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1),
        )

    def _make_layer(self, in_channels, out_channels, kernel_size, maxpoolwindow, dropout_rate):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        layers = [
            ResidualBlock(in_channels, out_channels, stride=1, downsample=downsample),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
            nn.Dropout(dropout_rate),
        ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dense(x)
        return x