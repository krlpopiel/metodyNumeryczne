import torch.nn as nn  # Import modułu do definiowania sieci neuronowych
import torch

class AlbumClassifier(nn.Module):  # Definicja klasy modelu sieci neuronowej
    def __init__(self):  # Konstruktor klasy
        super(AlbumClassifier, self).__init__()  # Wywołanie konstruktora klasy bazowej
        self.conv_layers = nn.Sequential(  # Warstwy konwolucyjne modelu
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Konwolucja: 3 kanały wejściowe (RGB), 16 filtrów
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Maksymalne próbkowanie: zmniejszenie wymiarów o połowę
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Kolejna warstwa konwolucyjna
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.MaxPool2d(kernel_size=2, stride=2)  # Maksymalne próbkowanie
        )

        # Rozmiar wyjściowy obliczany dynamicznie
        example_input = torch.zeros(1, 3, 900, 600)  # Przykładowy tensor o rozmiarze wejściowym
        with torch.no_grad():
            conv_output = self.conv_layers(example_input)
            self.flattened_size = conv_output.numel()  # Obliczenie rozmiaru po konwolucjach
        
        

        self.fc_layers = nn.Sequential(  # Warstwy w pełni połączone (fully connected)
            nn.Linear(self.flattened_size, 128),  # Warstwa w pełni połączona, zakładając wejście 600x900
            nn.ReLU(),  # Funkcja aktywacji ReLU
            nn.Linear(128, 8)  # Wyjście: 8 klas (widok x orientacja)
        )

    def forward(self, x):  # Funkcja definiująca przepływ danych w modelu
        x = self.conv_layers(x)  # Przejście przez warstwy konwolucyjne
        x = x.view(x.size(0), -1)  # Spłaszczenie danych przed podaniem do warstw w pełni połączonych
        x = self.fc_layers(x)  # Przejście przez warstwy w pełni połączone
        return x  # Zwrócenie wyników
