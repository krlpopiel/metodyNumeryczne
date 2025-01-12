#import odpowiednich funkcji inicjalizujących modele z wcześniej utworzonych plików
from model_resnet50 import get_resnet50
from model_vgg16 import get_vgg16
from model_alexnet import get_alexnet
from model_efficientnet_b0 import get_efficientnet_b0

class ModelLoader: #klasa załadowująca odpowiedni gotowy przetrenowany model z biblioteki PyTorch
    def __init__(self, num_classes=39): #konstruktor klasy, num_classes wskazuje na liczbę klas do których ma sie dostosować model
        self.num_classes = num_classes

    #przekazanie odpowiednich parametrów do funkcji wywołujących modele
    def load_resnet50(self, pretrained=True):
        return get_resnet50(self.num_classes, pretrained)

    def load_vgg16(self, pretrained=True):
        return get_vgg16(self.num_classes, pretrained)

    def load_alexnet(self, pretrained=True):
        return get_alexnet(self.num_classes, pretrained)

    def load_efficientnet_b0(self, pretrained=True):
        return get_efficientnet_b0(self.num_classes, pretrained)
