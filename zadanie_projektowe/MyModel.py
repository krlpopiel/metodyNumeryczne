import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Przykład modelu z warstwami konwolucyjnymi
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Warstwa konwolucyjna
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Kolejna warstwa konwolucyjna
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),  # Dopasuj rozmiar do danych wejściowych
            nn.ReLU(),
            nn.Linear(128, 10)  # Liczba klas wyjściowych
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Przekształcenie tensoru do wektora
        x = self.fc_layers(x)
        return x
