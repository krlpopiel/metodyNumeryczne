from model_resnet50 import get_resnet50
from model_vgg16 import get_vgg16
from model_alexnet import get_alexnet
from model_efficientnet_b0 import get_efficientnet_b0

class ModelLoader:
    def __init__(self, default_num_classes=39):
        self.default_num_classes = default_num_classes

    def load_resnet50(self, num_classes=None, pretrained=True, weights_path=None):
        return get_resnet50(
            num_classes=num_classes or self.default_num_classes,
            pretrained=pretrained,
            weights_path=weights_path
        )

    def load_vgg16(self, num_classes=None, pretrained=True, weights_path=None):
        return get_vgg16(
            num_classes=num_classes or self.default_num_classes,
            pretrained=pretrained,
            weights_path=weights_path
        )

    def load_alexnet(self, num_classes=None, pretrained=True, weights_path=None):
        return get_alexnet(
            num_classes=num_classes or self.default_num_classes,
            pretrained=pretrained,
            weights_path=weights_path
        )

    def load_efficientnet_b0(self, num_classes=None, pretrained=True, weights_path=None):
        return get_efficientnet_b0(
            num_classes=num_classes or self.default_num_classes,
            pretrained=pretrained,
            weights_path=weights_path
        )
