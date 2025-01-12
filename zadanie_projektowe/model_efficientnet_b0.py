import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch

def get_efficientnet_b0(num_classes=39, pretrained=True, weights_path=None):
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    if weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model