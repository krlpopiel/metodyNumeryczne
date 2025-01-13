#import odpowiednich funkcji inicjalizujących modele z wcześniej utworzonych plików
from model_resnet50 import get_resnet50
from model_vgg16 import get_vgg16
from model_alexnet import get_alexnet
from model_efficientnet_b0 import get_efficientnet_b0

class ModelLoader: #klasa załadowująca odpowiedni gotowy przetrenowany model z biblioteki PyTorch
    def __init__(self, default_num_classes=39): #konstruktor klasy, default_num_classes wskazuje na liczbę klas do których ma sie dostosować model
         self.default_num_classes = default_num_classes

    #przekazanie odpowiednich parametrów do funkcji wywołujących modele, liczbe klas, wskazanie czy 
    #mają mieć byc wstępnie wytrenowane, wskazanie ścieżki do pliku z zapisanym stanem dla danego modelu
    def load_resnet50(self, num_classes=None, pretrained=True, weights_path=None):
         return get_resnet50(
             #liczba klas jest podanym argumenetem, lub jeśli taki nie został podany, przypisywana jest domyślna liczba klas
            num_classes=num_classes or self.default_num_classes,
            #przypisanie podanej wartości, czy model ma byc wstępnie wytrenowany
            pretrained=pretrained,
            #przypisanie podanej ścieżki do pliku .pth
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
