import torch
import torch.nn as nn
import gin

@gin.configurable
class MyCNN(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int, filter1: int, filter2: int, dropout_rate: float, maxpoolwindow: int) -> None:
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.dense(x)
        return x