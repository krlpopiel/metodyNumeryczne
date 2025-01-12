import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch

def get_resnet50(num_classes=39, pretrained=True, weights_path=None):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Wczytaj wagi z pliku, jeśli podano ścieżkę
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    
    return model
