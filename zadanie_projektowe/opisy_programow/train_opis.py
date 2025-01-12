import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
from torch.utils.data import DataLoader #zaimportowanie klasy DataLoader umożliwiającej wczytanie przygotowanych danych
from torch.optim import Adam #zaimportowanie optymalizatora Adam, algorytm dostosowujący szybkość uczenia do każdego parametru
from torch.nn import CrossEntropyLoss  #zaimportowanie klasy CrossEntropyLoss, która oblicza strate pomiędzy danymi wejściowymi a wyjściowymi, stosowany do klasyfikacji obrazów
from dataset import AlbumDataset #zaimportowanie klasy AlbumDataset zawierającej niestandardowy zbiór danych - albumy muzyczne
from model import AlbumClassifier #zaimportowanie klasy modelu z pliky model.py
from torchvision import transforms #zaimportowanie modułu z pakietu Torchvision, umożliwiającego dokonanie przekształceń na obrazach 

 #wskazanie jak ma zostać zmieniony obraz na wejściu
transform = transforms.Compose([ #połączenie obydwu transformacji w jedno
    transforms.Resize((224, 224)), #przeskalowanie obrazu do rozmiaru 224x224, wartość ta jest kompatybilna z używanym modelem
    transforms.ToTensor(), #przekształcenie obrazu na tensor (!patrz slownik.txt), czyli format danych obliczeniowych PyTorch
])

#pobranie zbioru danych, podanie odpowiednich wartości parametrów (ścieżka do pliku z etykietami i folderem ze zdjęciami), 
#obrazy będą miały zdefiniowane wcześniej transformacje
dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',
    zdjecia_kat='zadanie_projektowe/images',
    transform=transform
)

#iteracyjne przejście przez cały zbiór danych, pobranie id albumu dla każdego zdjęcia (czyli dla 38 płyt)
for album in dataset.etykiety:
    album_id = album['album_id']  #pobranie klucza zawierającego wartość id albumu ze słownika album i przypisanie go do zmiennej album_id
    for zdj in album['zdjecia']: #iteracyjne przejście przez wszystkie zdjęcia dla danej płyty (każda płyta ma 8 zdjęć)
        zdj['album_id'] = album_id  #przypisanie znalezionego id do odpowiedniego zdjęcia


#załadowanie zdefiniowanego wcześniej zbioru danych, paramatery to: dataset - zbiór danych
#batch_size - rozmiar partii danych w jednym kroku obliczeniowym (na raz ładowane będzie 8 zdjęć)
#shuffle - wskazanie, czy dane powinny zostać losowo pomieszane przed każdą epochą (!patrz slownik.txt), 
#wskazana jest wartość True, aby model nie uczył się na podstawie zależności występowania obrazów w tej samej kolejności
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#wyznaczenie liczby klas - id albumów, iteracyjne pobranie przypisanego album_id do zdjęcia, set() umożliwia przypisanie wielu wartości do jednej zmiennej
num_classes = len(set([zdj['album_id'] for album in dataset.etykiety for zdj in album['zdjecia']]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #ustalenie gdzie ma być uruchamiany model - na GPU(cuda) czy CPU
#jeśli karta graficzna obsługuje CUDA (przyśpiesz obliczenia niezbędne do wyświetlania obrazów), obraz uruchomiony zostanie przez GPU, w inny przypadku przez CPU

model = AlbumClassifier(num_classes=num_classes).to(device) #uruchomienie modelu z odpowiednią ilością klas i przeniesienie go na wybrane powyżej urządzenie

#uruchomienie wspomnianej wcześniej klasy, która obliczy różnicę między przewidywaniami modelu a rzeczywistymi etykietami
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001) #uwtorzenie wspomnianego wcześniej optymalizatora, 
#parametry: model.parameters() - pobiera wagi modelu, które będą aktualizowane przez optymalizator
#lr - współczynik uczenia, wpływa na szybkość uczenia sie modelu poprzez oddziaływanie na gradient (zmiany wartości danych po wejściu)
#wartość 0.001 jest optymalna w tym kontekście, gdyż zapewnia stabilność działania i zmniejszenie funkcji kosztu (!patrz słownik.txt)

#trenowanie modelu i podanie liczby epok (!patrz slownik.txt)
num_epochs = 50 #algorytm przejdzie przez cały zbiór danych 50 razy
for epoch in range(num_epochs): #iteracyjny proces trenowania, zbiór przejdzie przez każdą epoche
    model.train() #ustawienie modelu w tryb trenowania (!patrz slownik.txt), czyli tryb uczenia się modelu na podstawie podanych danych
#w przeciwieństwe do trybu awaluacji, warstwy dropout jest włączona, czyli losowo ustawia wartości wejścia na 0, aby zapobiec przeuczeniu (!patrz slownik.txt)
#warstwa batch_normalization utrzymuje stałe wartości średniej i wariancji dla danych podczas treningu
    running_loss = 0.0 #licznik śledzący sumę błędów dopasować w aktualnej epoche

    #iteracyjne przejście przez obrazy i ich etykiety w załadowanym zbiorze danych
    for images, labels in dataloader:  
        #przeniesienie obrazów i ich etykiet na wybrane urządzenie
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images) #przekazanie obrazów przez model i wygenerowanie przewidywanych wyników
        loss = criterion(outputs, labels) #obliczenie wspomnianej wcześniej funkcji kosztów

        #przygotowanie danych do kolejnej epochi
        optimizer.zero_grad() #zerowanie gradientu (zmian) dla optymalizatora aby wykluczyć kumulacje wag
        loss.backward() #sprawdzenie wielkości błędów popęłnionych przez model i dostosowanie działania, aby w kolejnej epoche te błędy były mniejsze
        #dzięki temu wyniki przewidywane będą bardziej zgodne z tymi rzeczywistymi 
        optimizer.step() #zaktualizowanie parametrów optymalizatora dla modelu

        running_loss += loss.item() #sumowanie otrzymanych w jednej epochce blędów

    #podanie aktualnej epochi i jej średniego błędu
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

#zapisanie modelu do pliku sledzącego postępy w trenowaniu, dzięki niemu zachować można wyuczone już parametry i wagi
torch.save(model.state_dict(), 'album_classifier.pth')
