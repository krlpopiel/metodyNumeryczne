import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms

def get_inception_v3(num_classes=39, pretrained=True):
    if pretrained:
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
    else:
        model = inception_v3(weights=None, aux_logits=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model
