import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
from torchvision.models import alexnet, AlexNet_Weights #zaimportowanie gotowego modelu EfficientNet_B0 z biblioteki PyTorch oraz jego domyślne (wytrenowane) wagi

#zainicjalizowanie modelu, ma on klasyfikować obrazy dla 39 etykiet (album_id) i wskazuje, ze ma on korzystać z wstępnie wytrenowanych parametrów
def get_alexnet(num_classes=39, pretrained=True):
    if pretrained: #chcąc korzystać z przetrenowanego modelu, następuje pobranie jego domyślnych wag
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)
    else:
        model = alexnet(weights=None) #dla wartości pretrained=False model tworzony jest bez domyślnych wag, z losowymi wartosciami

    #dostosowanie ostatniej warstwy (w pełni połączonej) w modelu do ilość swoich danych
    #in_features wskazuje oryginalną liczbę klasyfikowanych etykiet - 1000
    #num_classes wskazuje jaką ilosć etykiet ma klasyfikować model - 39, zgodnie ze zbiorem danych
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    #pobranie odpowiednich transfromacji dla modelu, jeśli model ma korzystać z przetrenowanych parametrów
    model.transform = weights.transforms() if pretrained else None

    return model #zwrócenie utworzonego modelu
