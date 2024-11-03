import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes: int, units1: int, units2: int, kernel_size: int, dropout_rate: float, maxpoolwindow: int) -> None:
        super(MyCNN, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, units1, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(units1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
            nn.Conv2d(units1, units2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(units2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpoolwindow),
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = self.dense(x)
        return x