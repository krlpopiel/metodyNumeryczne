import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,  vgg16, VGG16_Weights, inception_v3, Inception_V3_Weights, alexnet, AlexNet_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms

class ModelLoader:
    def __init__(self, num_classes=38):
        self.num_classes = num_classes

    def get_resnet50(self, pretrained=True):
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
        else:
            model = resnet50(weights=None)
            
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
    
    def get_vgg16(self, pretrained=True):
        if pretrained:
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights)
        else:
            model = vgg16(weights=None)

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)
        return model

    def get_inception_v3(self, pretrained=True):
        if pretrained:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=True)
        else:
            model = inception_v3(weights=None, aux_logits=True)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        model.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return model

    def get_alexnet(self, pretrained=True):
        if pretrained:
            weights = AlexNet_Weights.DEFAULT
            model = alexnet(weights=weights)
        else:
            model = alexnet(weights=None)

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        model.transform = weights.transforms() if pretrained else None

        return model

    def get_efficientnet_b0(self, pretrained=True):
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            model = efficientnet_b0(weights=weights)
        else:
            model = efficientnet_b0(weights=None)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        model.transform = weights.transforms() if pretrained else None

        return model