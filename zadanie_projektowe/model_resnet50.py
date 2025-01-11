import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes=39, pretrained=True):
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
        
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
