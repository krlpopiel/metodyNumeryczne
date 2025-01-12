import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
from torchvision.models import resnet50, ResNet50_Weights #zaimportowanie gotowego modelu Resnet50 z biblioteki PyTorch oraz jego domyślne (wytrenowane) wagi
import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego

#zainicjalizowanie modelu, ma on klasyfikować obrazy dla 39 etykiet (album_id) i wskazuje, ze ma on korzystać z wstępnie wytrenowanych parametrów
#wskazanie pliku ze stanem wag modelu
def get_resnet50(num_classes=39, pretrained=True,  weights_path=None):
    if pretrained: #chcąc korzystać z przetrenowanego modelu, następuje pobranie jego domyślnych wag
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights) #utworzenie modelu z odpowiednimi wagami
    else:
        model = resnet50(weights=None) #dla wartości pretrained=False model tworzony jest bez domyślnych wag, z losowymi wartosciami
    
    #dostosowanie ostatniej warstwy (w pełni połączonej) w modelu do ilość swoich danych
    #in_features wskazuje oryginalną liczbę klasyfikowanych etykiet - 1000 (źródło - Roboflow, dokładny link w sprawozdaniu)
    #num_classes wskazuje jaką ilosć etykiet ma klasyfikować model - 39, zgodnie ze zbiorem danych
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if weights_path: #jeśli podano ścieżke do pliku .pth następuje jego załadowanie (załadowanie zapisanych wcześniej wag modelu)
        model.load_state_dict(torch.load(weights_path))
    
    return model #zwrócenie uwtorzonego modelu
