import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego

#klasa tworząca własny model klasyfikacji obrazów (!patrz slownik.txt), opiera się na wykorzystaniu warstwy konwolucyjnej (!patrz slownik.txt)
#dziedziczy ona po klasie bazowej dla wszystkich jej modułów, jest to obowiązkowy krok w celu tworzenia własnego modelu
class AlbumClassifier(nn.Module):
    def __init__(self, num_classes): #metoda umożliwiająca utworzenie obiektu danej klasy, przypisanie mu wartości 
        super(AlbumClassifier, self).__init__() #wywołanie konstruktora klasy bazowej (Module), jest to obowiązkowe aby korzystać z funkcji klasy bazowej, są one zastosowane poniżej
        
        self.conv_layers = nn.Sequential( #utworzenie dwóch warstw konwolucyjnych modelu (!patrz slownik.txt) i połaczenie ich w jeden blok za pomocą klasy Sequential
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), #utworzenie pierwszej warstwy konwolucyjnej, jej parametry to odpowiednio:
            #3 - liczba kanałów koloru w obrazie na wejściu (RGB), 16 - liczba kanałów, która będzie wygenerowana przez warstwe (warstwa ma za zadanie znaleść 16 różnych cech na obrazie, np. tekstury),
            #im więcej kanałów wyjściowych tym większa złożoność obliczeniowa
            #kernel_size - rozmiar filtru (!patrz slownik.txt), oznacza że na obszarze o wymiarach 3x3 piksele model będzie wyszukiwał informacje o obrazie, rozmiar ten jest optymalny, pozwala na lepszą wydajność modelu
            #stride - wskazuje z jakim krokiem ma sie przesuwać filtr, czyli co jeden piksel, #padding - pomaga zachować wejściowy rozmiar obrazu, wartość 1 zapewnia, że na wyjściu obraz będzie miał wymiary 224x224
            
            nn.ReLU(), #pierwsza warstwa aktywacji (!patrz slownik.txt), wprowadza nieliniowość i pozwala na trening bardziej złożonych zależności
            nn.MaxPool2d(kernel_size=2, stride=2), #pierwsza warstwa łaczenia - redukuje rozmiar obrazu i wydobywa z niego cechy o najwyższej wadze (!patrz slownik.txt)
            #zmniejsza liczbę parametrów do wytrenowania, tym samym upraszczając model
            #parametry to: kernel_size - przeszukiwany będzie obszar 2x2 piksele, stride - z krokiem co 2 piksele, tym samym sprawia to że rozmiar obrazu zmniejsza się o połowe
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #druga warstwa konwolucyjna
            nn.ReLU(), #druga warstwa aktywacji
            nn.MaxPool2d(kernel_size=2, stride=2) #druga warstwa łaczenia
            
            #drukrotne zastosowanie tych warstw umożliwia wydobycie bardziej złożonych cech z obrazów
        )

        
        example_input = torch.zeros(1, 3, 224, 224)  #utworzenie przykladowego tensora wypełnionego 0,
        #parametry: w partii trenowania jest 1 obraz, 3 kanały koloru (RGB), wymiary obrazum zgodne z tymi dla transforms w dataset.py
        with torch.no_grad(): #gradienty (!patrz slownik.txt) nie będą obliczane, narazie testowane są wyjścia uzyskane dzięki powyższym warstwą, bez trenowania modelu
            #oszczedza to m.in pamięć
            
            conv_output = self.conv_layers(example_input) #przepuszczenie obrazu przez warstwy konwolucyjne, zwrócenie wyniku do zmiennej conv_output
            self.flattened_size = conv_output.numel() #zwrócenie liczby elementów w tensorze po przejściu obrazu, tak zwany spłaszczony obraz

        self.fc_layers = nn.Sequential( #utworzenie sekwencji warstw w pełni połączonych (!patrz slownik.txt)
            nn.Linear(self.flattened_size, 128), #połączenie neuronów w postaci jednowymiarowego tensora z neuronami z poprzedniej warstwy, 128 to liczba neuronów w tej warstwie
            #taka liczba pozwala na większą złożoność obliczeniową modelu
            
            nn.ReLU(), #warstwa aktywacji
            nn.Linear(128, 39)  #druga warstwa w pełni połaczona, przyjmuje 128 neuronów z wyjścia z pierwszej warstwy Linear,
            #przekształcenie wyjść wektora wejściowego na wektor z liczbę wyjść równą liczbie klas, czyli id albumów płyt
        )

    def forward(self, x): #metoda określająca w jaki sposób dane wejściowe są przekształcane na wyjście, odwołuje się do obiektu, x do dane na wejściu
        x = self.conv_layers(x) #dane przechodzą przez zdefiniowane wcześniej warstwy konwolucyjne
        x = x.view(x.size(0), -1) #dane wejściowe zostają spłaszczone do wektora, przy użyciu size(0) zwracana jest liczba przetwarzanych obrazów w jednym kroku obliczeniowym
        # -1 wskazuje, aby PyTorch automatycznie dopasował wymiar danych do spłaszczenia
        x = self.fc_layers(x) #tak przygotowane dane przechodzą przez warstwy w pełni połaczone
        return x #zwrócenie przetworzonych danych, jako wynik klasyfikacji
