import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights #zaimportowanie gotowego modelu EfficientNet_B0 z biblioteki PyTorch oraz jego domyślne (wytrenowane) wagi

#zainicjalizowanie modelu, ma on klasyfikować obrazy dla 39 etykiet (album_id) i wskazuje, ze ma on korzystać z wstępnie wytrenowanych parametrów
def get_efficientnet_b0(num_classes=39, pretrained=True): #chcąc korzystać z przetrenowanego modelu, następuje pobranie jego domyślnych wag
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights) #utworzenie modelu z odpowiednimi wagami
    else:
        model = efficientnet_b0(weights=None) #dla wartości pretrained=False model tworzony jest bez domyślnych wag, z losowymi wartosciami

    #dostosowanie ostatniej warstwy (w pełni połączonej) w modelu do ilość swoich danych
    #in_features wskazuje oryginalną liczbę klasyfikowanych etykiet - 1000
    #num_classes wskazuje jaką ilosć etykiet ma klasyfikować model - 39, zgodnie ze zbiorem danych
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    #pobranie odpowiednich transfromacji dla modelu, jeśli model ma korzystać z przetrenowanych parametrów
    model.transform = weights.transforms() if pretrained else None

    return model #zwrócenie utworzonego modelu
