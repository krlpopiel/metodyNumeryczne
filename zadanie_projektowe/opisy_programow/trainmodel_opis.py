import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
import torch.optim as optim #zaimportowanie modułu zawierącego algortymi optymalizacji
import torch.nn as nn #zaimportowanie modułu biblioteki PyTorch zawierającej potrzebne funkcje, wykorzystywane przy pracy z sieciami neuronowymi
from torch.utils.data import DataLoader #zaimportowanie klasy DataLoader umożliwiającej wczytanie przygotowanych danych
from torchvision import datasets, transforms #zaimportowanie dwóch modułów z TorchVision - jeden zawiera gotowe zestawy danych, drugi umożliwia dokonanie przekształceń na obrazach 
from model_loader import ModelLoader #zaimportowanie klasy zawierącej inicjalizacje gotowych modeli PyTorch
from dataset import AlbumDataset  #zaimportowanie klasy AlbumDataset zawierającej niestandardowy zbiór danych - albumy muzyczne

#podanie odpowiednich parametrów do trenowania, batch-size: wielkość partii, model w jednym kroku obliczeniowym będzie przyjmował 32 obrazy
#num_epoch: liczba przejść przez caly zbiór danych, learning_rate - współczynik uczenia, wpływa na szybkość uczenia sie modelu poprzez oddziaływanie na gradient (zmiany wartości danych po wejściu)
#num_clasess - liczba klas, które ma brać pod uwage model
batch_size = 32
num_epochs = 50
learning_rate = 0.001
num_classes = 39

#lista modeli które będą użyte do klasyfikacji obrazów
model_choice = 'efficientnet_b0'  # 'resnet50' 'vgg16', 'alexnet', 'efficientnet_b0'

#wskazanie jak ma zostać zmieniony obraz na wejściu
transform = transforms.Compose([ #połączenie obydwu transformacji w jedno
    transforms.Resize((224, 224)), #przeskalowanie obrazu do rozmiaru 224x224, wartość ta jest kompatybilna z używanymi modelami
    transforms.ToTensor(), #przekształcenie obrazu na tensor (!patrz slownik.txt), czyli format danych obliczeniowych PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #obraz zostaje znormalizowany, aby modele działały poprawnie,
        #model ten został wytrenowany na zbiorze ImageNet, przez co nowe dane muszą zostać do niego dopasowane, aby model działał prawidłowo
        #mean - średnia wartość do trzech kanałów RGB, std - odchylenie standardowe dla tych kanałów
])

#pobranie zbioru danych do trenowania, wskazanie ścieżki pod którą znajduje się plik z etykietami obrazów
#kalatog ze zdjęciami, obrazy będą miały zdefiniowane wcześniej transformacje
train_dataset = AlbumDataset(plik_json='zadanie_projektowe/annotations.json', 
                             zdjecia_kat='zadanie_projektowe/images', 
                             transform=transform)

#załadowanie zdefiniowanego wcześniej zbioru danych, paramatery to: train_dataset - zbiór danych
#batch_size - rozmiar partii danych w jednym kroku obliczeniowym (na raz ładowane będą 32 zdjęcia)
#shuffle - wskazanie, czy dane powinny zostać losowo pomieszane przed każdą epochą (!patrz slownik.txt), 
#wskazana jest wartość True, aby model nie uczył się na podstawie zależności występowania obrazów w tej samej kolejności
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#załadowanie odpowiedniego modelu, obsługującego podaną wcześniej ilość klas
model_loader = ModelLoader(num_classes=num_classes)

#zgodnie z podaną nazwą wybieranego modelu, taki model jest inicjalizowany i trenowany, przy zastosowaniu klasy ModelLoader
#zawierającej odpowiednie funkcje wywołujące, wybrany model posiada już oryginalne przetrenowane parametry, teraz będzie przyjmował
#niestandardowe parametry ze zbioru z albumami muzycznymi
if model_choice == 'resnet50':
    model = model_loader.load_resnet50(pretrained=True)
elif model_choice == 'vgg16':
    model = model_loader.load_vgg16(pretrained=True)
elif model_choice == 'alexnet':
    model = model_loader.load_alexnet(pretrained=True)
elif model_choice == 'efficientnet_b0':
    model = model_loader.load_efficientnet_b0(pretrained=True)
else:
    raise ValueError("Invalid model choice") #zwrócenie odpowiedniego komunikatu, gdy podano nie odpowiedni model

#ustalenie gdzie ma być uruchamiany model - na GPU(cuda) czy CPU
#jeśli karta graficzna obsługuje CUDA (przyśpiesz obliczenia niezbędne do wyświetlania obrazów), obraz uruchomiony zostanie przez GPU, w inny przypadku przez CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) #uruchomienie modelu z odpowiednią ilością klas i przeniesienie go na wybrane powyżej urządzenie

#uwtorzenie optymalizatora Adam z modułu optim - algorytm dostosowujący szybkość uczenia do każdego parametru, 
#parametry: model.parameters() - pobiera wagi modelu, które będą aktualizowane przez optymalizator
#lr - współczynik uczenia, wpływa na szybkość uczenia sie modelu poprzez oddziaływanie na gradient (zmiany wartości danych po wejściu)
#wartość 0.001 jest optymalna w tym kontekście, gdyż zapewnia stabilność działania i zmniejszenie funkcji kosztu (!patrz słownik.txt)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#uruchomienie klasy CrossEntropyLoss z modułu nn, która obliczy różnicę między przewidywaniami modelu a rzeczywistymi etykietami
criterion = nn.CrossEntropyLoss()

#trenowanie modelu i przejście całkego zbioru przez podaną wcześniej liczbę epok
for epoch in range(num_epochs):
    model.train() #ustawienie modelu w tryb trenowania (!patrz slownik.txt), czyli tryb uczenia się modelu na podstawie podanych danych
#w przeciwieństwe do trybu awaluacji, warstwy dropout jest włączona, czyli losowo ustawia wartości wejścia na 0, aby zapobiec przeuczeniu (!patrz slownik.txt)
#warstwa batch_normalization utrzymuje stałe wartości średniej i wariancji dla danych podczas treningu

    running_loss = 0.0 #licznik śledzący sumę błędów dopasować w aktualnej epoche
    correct = 0 #licznik poprawnych przewidywać dla danego modelu
    total = 0 #licznik wszystkich obrazów branych pod uwage w danej epoche

    #iteracyjne przejście przez obrazy i ich etykiety w załadowanym zbiorze danych
    for inputs, labels in train_loader:
        #przeniesienie obrazów i ich etykiet na wybrane urządzenie
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() #zerowanie gradientu (zmian) dla optymalizatora aby wykluczyć kumulacje wag
        outputs = model(inputs) #przekazanie obrazów przez model i wygenerowanie przewidywanych wyników
        loss = criterion(outputs, labels) #obliczenie wspomnianej wcześniej funkcji kosztów
        loss.backward() #sprawdzenie wielkości błędów popęłnionych przez model i dostosowanie działania, aby w kolejnej epoche te błędy były mniejsze
        #dzięki temu wyniki przewidywane będą bardziej zgodne z tymi rzeczywistymi 
        
        optimizer.step() #zaktualizowanie parametrów optymalizatora dla modelu

        running_loss += loss.item() #sumowanie otrzymanych w jednej epochce blędów, w formacie zwykłej liczby - funkcja item()

        #wybór klas o najwyższym prawodopobieństwie i porównanie przewidywanej klasy z tą rzeczywistą
        _, predicted = torch.max(outputs, 1) #pobranie przewidywanego wyniku z największym prawopodobieństwem wystąpienia - funkcja max() z parametrem outputs
        #1 oznacza, że przeszukanie odbywa sie dla jednego zestawu w partii, czyli 32 zdjęć
        total += labels.size(0) #zsumowanie wszystkich obrazów branych pod uwage w danej epoce
        correct += (predicted == labels).sum().item() #porównanie przewidywanej klasy z tą rzeczywistą
        #jeśli wartości są takie same, poprawne trafienia są sumowane - funkcja sum() i zwracane jako liczba za pomoca funkcji item()

    epoch_loss = running_loss / len(train_loader) #wyliczenie średniego blędu epochi
    epoch_acc = 100 * correct / total #obliczenie dokładności klasyfikacji modeli w aktualnej epoche, w formacie procentowym
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%') #wyświetlenie wyników

#zapisanie modelu do pliku sledzącego postępy w trenowaniu, dzięki niemu zachować można wyuczone już parametry i wagi
model_save_path = f"model_{model_choice}.pth" #zależnie od nazwy aktualnie trenowanego modelu, taką nazwe ma plik
torch.save(model.state_dict(), model_save_path)
print(f'Model zapisany do {model_save_path}') #podanie informacji do jakiego pliku zostały zapisane wyniki trenowania
