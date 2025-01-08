import torch.nn as nn
import torch

class AlbumClassifier(nn.Module):
    def __init__(self):
        super(AlbumClassifier, self).__init__()

        # Warstwy konwolucyjne
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Pierwsza warstwa konwolucyjna
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling, zmniejsza rozmiar o połowę
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Druga warstwa konwolucyjna
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling, zmniejsza rozmiar o połowę
        )

        # Sprawdzanie rozmiaru wyjścia po warstwach konwolucyjnych
        example_input = torch.zeros(1, 3, 900, 600)  # Zakładając, że wejście to 900x600
        with torch.no_grad():
            conv_output = self.conv_layers(example_input)
            self.flattened_size = conv_output.numel()  # Obliczenie rozmiaru po konwolucjach

        # Warstwy w pełni połączone
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 128),  # Dostosowanie wymiaru wejściowego
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 klas wyjściowych (widok i orientacja)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Przejście przez warstwy konwolucyjne
        x = x.view(x.size(0), -1)  # Spłaszczenie przed wejściem do warstw liniowych
        x = self.fc_layers(x)  # Przejście przez warstwy liniowe
        return x
