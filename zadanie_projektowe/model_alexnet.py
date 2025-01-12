import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
import torch

def get_alexnet(num_classes=39, pretrained=True, weights_path=None):
    if pretrained:
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)
    else:
        model = alexnet(weights=None)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model