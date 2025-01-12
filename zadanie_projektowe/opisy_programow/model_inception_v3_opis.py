import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
from torchvision.models import inception_v3, Inception_V3_Weights #zaimportowanie gotowego modelu Inception_V3 z biblioteki PyTorch oraz jego domyślne (wytrenowane) wagi
from torchvision import transforms #zaimportowanie modułu z pakietu Torchvision, umożliwiającego dokonanie przekształceń na obrazach 

#zainicjalizowanie modelu, ma on klasyfikować obrazy dla 39 etykiet (album_id) i wskazuje, ze ma on korzystać z wstępnie wytrenowanych parametrów
def get_inception_v3(num_classes=39, pretrained=True):
    if pretrained: #chcąc korzystać z przetrenowanego modelu, następuje pobranie jego domyślnych wag
        #parametr aux_logits wprowadza dodatkowa warstwe obliczeniową, wspomaga proces uczenia modelu podczas trenowania
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
    else:
        model = inception_v3(weights=None, aux_logits=True) #dla wartości pretrained=False model tworzony jest bez domyślnych wag, z losowymi wartosciami
        #wprowadzana jest warstwa dodatkowa opisana powyżej

    #dostosowanie ostatniej warstwy (w pełni połączonej) w modelu do ilość swoich danych
    #in_features wskazuje oryginalną liczbę klasyfikowanych etykiet - 1000
    #num_classes wskazuje jaką ilosć etykiet ma klasyfikować model - 39, zgodnie ze zbiorem danych
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    #model ten wymaga odpowiedniej transformacji obrazów na wyjściu, obraz ma mieć wymiar 224x224,
    #zostać przekształcony na tensor (standardowe działanie)
    #obraz zostaje znormalizowany, model został wytrenowany na zbiorze ImageNet, przez co nowe dane muszą zostać do niego dopasowane, aby model działał prawidłowo
    #mean - średnia wartość do trzech kanałów RGB, std - odchylenie standardowe dla tych kanałów
    model.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model #zwrócenie modelu
