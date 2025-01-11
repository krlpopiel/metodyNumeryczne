import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

def get_vgg16(num_classes=39, pretrained=True):
    if pretrained:
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
    else:
        model = vgg16(weights=None)

    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
