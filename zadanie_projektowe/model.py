import torch.nn as nn
import torch

class AlbumClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AlbumClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        example_input = torch.zeros(1, 3, 224, 224) 
        with torch.no_grad():
            conv_output = self.conv_layers(example_input)
            self.flattened_size = conv_output.numel()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 39)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
