import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def seq_model(n_classes):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=16 * 4 * 4, out_features=120),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=120, out_features=84),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=84, out_features=n_classes)
    )