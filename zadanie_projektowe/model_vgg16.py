import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torch

def get_vgg16(num_classes=39, pretrained=True, weights_path=None):
    if pretrained:
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
    else:
        model = vgg16(weights=None)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # Wczytaj wagi z pliku, jeśli podano ścieżkę
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    
    return model
