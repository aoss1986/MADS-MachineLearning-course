import torch
import torch.nn as nn
import gin

@gin.configurable
class MyCNN(nn.Module):
    def __init__(self, num_classes: int, unit1: int, unit2: int, kernel_size: int, dropout_rate: float, maxpoolwindow: int) -> None:
        super(MyCNN, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, unit1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(unit1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
            nn.Conv2d(unit1, unit2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(unit2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, unit1),
            nn.ReLU(),
            nn.Linear(unit1, unit2),
            nn.ReLU(),
            nn.Linear(unit2, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.dense(x)
        return x