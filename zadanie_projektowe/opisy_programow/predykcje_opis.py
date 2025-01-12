import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
from torchvision import transforms #zaimportowanie modułu z pakietu Torchvision, umożliwiającego dokonanie przekształceń na obrazach 
from PIL import Image #zaimportowanie modułu Image z bilbioteki Pillow, umożliwiająca prace z obrazem
import os #moduł umożliwiający prace z systemem operacyjnym, m.in zarządzanie plikami i katalogami
import json #biblioteka umożliwiająca zarządzanie plikami w formacie JSON
from model import AlbumClassifier  #zaimportowanie klasy modelu z pliky model.py
import matplotlib.pyplot as plt #zaimportowanie modułu pyplot z biblioteki matplotlib, umożliwiającego m.in wizualizacje obrazów

num_classes = 38  #wskazanie ilości klas (etykiet) do klasyfikacji obrazów przez model
model = AlbumClassifier(num_classes=num_classes)  #przypisanie modelu do zmiennej model
model.load_state_dict(torch.load('album_classifier.pth'))  #załadowanie pliku ze stanem modelu, load_state_dict - załadowuje przetrenowane wcześniej wagi modelu do pliku album_clasiffier
model.eval()  #ustawienie modelu w tryb ewaluacji (!patrz slownik.txt), czyli tryb oceny wydajności modelu
#aby trenowanie było stabilne, ewaluacja deaktywuje niektóre neurony, np dla warstwy dropout która losowo ustawia wartości wejścia na 0
#ewaluacja umożliwia działanie warstwy batch_normalization, która przyśpiesza trening, dzięki normalizacji danych wejściowych m.in przez skalowanie ich

#funkcja pobierające dane z pliku z etykietami obrazów (tytuł i wykonawce), jako parametry przyjmuje podane id_albumu i ścieżkę do pliku json
def get_album_info(album_id, json_file='zadanie_projektowe/annotations.json'):
    try: #jeśli plik został znaleziony, następuje otwarcie go w trybie do odczytu i zapisanie danych do zmiennej data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for album in data: #iteracyjne przejście przez wszystkie etykiety w pliku
            if album['album_id'] == album_id: #pobranie i zwórcenie tytułu oraz wykonwacy przypisanego do albumu o wskazanym id
                return album['tytul'], album['wykonawca']
    except FileNotFoundError: #jeśli plik nie został znaleziony drukowany jest odpowiedni komuniat
        print(f"Plik {json_file} nie został znaleziony.")
    return None, None  #jeśli wprowadzono id, które nie istnieje, funkcja zwróci None dla tytułu i wykonawcy

#funkcja przetwarzająca obraz o podanej ścieżke (image_path), klasyfikująca go przy użyciu podanego modelu (model)
#wybiera i zwraca ona top 5 najbardziej prawdopodobnych klas etykiet
def test_model_on_image(image_path, model):
    #wskazanie jak ma zostać zmieniony obraz na wejściu
    transform = transforms.Compose([ #połączenie obydwu transformacji w jedno
        transforms.Resize((224, 224)),  #przeskalowanie obrazu do rozmiaru 224x224, wartość ta jest kompatybilna z używanym modelem
        transforms.ToTensor(),  #przekształcenie obrazu na tensor (!patrz slownik.txt), czyli format danych obliczeniowych PyTorch
    ])

    if not os.path.exists(image_path): #sprawdzenie, czy podana na wejściu ścieżka istnieje, wykorzystanie funkcji exist z modułu os
        print(f"Plik {image_path} nie istnieje.") #jeśli ścieżka jest nieprawidłowa, drukowany jest odpowiedni komunikat
        return None #jeśli ścieżka jest nieprawidłowo, funkcja zwraca None

    image = Image.open(image_path).convert("RGB") #otwarcie obrazu pod wskazana ścieżką, wykorzystanie modułu Image, przekształcenie obrazu na format RGB
        #jest to standardowy format wykorzystywany w przetwarzaniu obrazów i pracy z bibliotekami, zapewnia on że wszystkie zdjęcia mają 3 kanały kolorów (czerwony,zielony,niebieski), bez kanału przezroczystosci
    image = transform(image).unsqueeze(0)  #obraz zostaje przekształcony na tensor, dodany jest do niego nowy wymiar
    # tzw. wymiar wsadowy (batch dimension), model działa na partiach, funkcja unsqueeze(0) dodaje dodatkowy wymiar, aby obraz był potraktowany jako partia

    with torch.no_grad(): #wyłączenie gradientów (!patrz slownik.txt) w celu oszczedzenia pamieci i szybszego działania, jest to zalecane podczas testowania modelu
        output = model(image)  #zapisanie do zmiennej output wyniku działania modelu
        probabilities = torch.softmax(output, dim=1)  #zamiana wyników modelu na wartość procentową (podanie prawdopodobieństwa, która klasa, czyli jaki album, została rozpoznany na zdjęciu)
        #jest to kluczowa operacja, podczas której wartości liczbowe zamieniane są na wartosci w przedziale [0,1] i sumowane do 1
        #parametr to klasyfikowany przez model obraz, oraz dim=1 wskazuje, że wszystkie przekształcone wartości mają sumować sie do 1
        
        top5_classes = torch.topk(probabilities, 5).indices.squeeze().tolist()  #zwrócenie 5 klasy, dla których prawdopodobieństwo wystąpienia na zdjęciu jest największe, 
        #tym samym zwracane są album_id dla obrazu o danym prawdopodobieństwu za pomocą atrybutu indices
        #funkcja squeeze zamienia tensor obrazu na wektor o jedynym wymiarze, funkcja toList zamienia ten wektor na liste Pythona

    return top5_classes #zwrócenie wyników

#funkcja przetwarza wszystkie obrazy we wskazanym folderze, przeprowadza predykcje i zwraca jej wyniki 
#czyli indeks albumu, tytuł i wykonawce (!patrz slownik.txt)
def process_images_in_folder(folder_path, model, json_file): #parametry - folder, gdzie znajdują sie zdjecia, wskazanie używanego modelu, plik z etykietami
    if not os.path.exists(folder_path): #sprawdzenie, czy podana na wejściu ścieżka istnieje, wykorzystanie funkcji exist z modułu os
        print(f"Folder {folder_path} nie istnieje.") #jeśli ścieżka jest nieprawidłowa, drukowany jest odpowiedni komunikat
        return #zakończenie działania funkcji

    #za pomocą os.listdir zwracana jest lista nazw obrazów znajdujących sie w podanym folderze, sprawdzany jest również format zdjęć
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files: #sprawdzenie czy w folderze istnieją jakiekolwiek zdjęcią (czy lista images_files nie jest pusta)
        print(f"Brak zdjęć w folderze {folder_path}.") #jeśli nie ma obrazów, drukowany jest odpowiedni komunikat
        return #zakończenie działania funkcji

    #iteracyjne przejście przez wszystkie obrazy w folderze
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file) #utworzenie pełnej ścieżki do obrazu w podanym folderze
        print(f"\nPrzetwarzanie obrazu: {image_file}") #podanie informacji, który obraz jest aktualnie przetwarzany

        #w celu usprawnienia predykcji pobierane są wartości 5 najbardziej prawdopodobnych klas z opisanej wcześniej funkcji (album_id)
        results = test_model_on_image(image_path, model)

        img = Image.open(image_path) #otwarcie odpowiedniego obrazu
        plt.imshow(img) #wyświetlenie przetwarzanego obrazu
        plt.title(f"Obraz: {image_file}") #wydruk nazwy obrazu
        plt.show() #wyświetlenie obrazu i tytułu

        if results is not None: #iteracyjne wyświetlenie wyników predykcji, następuje jeśli udało się funkcja zwróciła odpowiednie wyniki 
            for rank, album_id in enumerate(results, start=1):  #zwrócone albumy są numerowe od 1 do 5
                album_title, album_artist = get_album_info(album_id, json_file) #wykorzystanie wcześniej opisanej funkcji do pobrania tytułu i wykonawcy dla albumu o wskazanym id
                if album_title and album_artist: #jeśli pobrane informacje istnieją, drukowany jest odpowiedni komunikat z wartościami
                    print(f"{rank}. id_albumu: {album_id}, Tytuł: '{album_title}', Wykonawca: {album_artist}")
                else: #jeśli nie znaleziono informacji na temat klasyfikowanego obrazu
                    print(f"{rank}. id_albumu: {album_id} (informacje o albumie nieznane)")
        else: #jeśli któryś z obrazów nie został przetworzony, drukowany jest odpowiedni komunikat
            print(f"Nie udało się przetworzyć obrazu {image_file}")

#podanie odpowiedniej ścieżki do folderu
folder_path = 'zadanie_projektowe/zdjecia_do_predykcji'

#podanie odpowiedniej ścieżki do pliku z etykietami
json_file = 'zadanie_projektowe/annotations.json'

#wykonanie predykcji - wywołanie powyższej funkcji
process_images_in_folder(folder_path, model, json_file)
