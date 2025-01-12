import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
from torchvision import transforms #zaimportowanie modułu z pakietu Torchvision, umożliwiającego dokonanie przekształceń na obrazach 
from PIL import Image #zaimportowanie modułu Image z bilbioteki Pillow, umożliwiająca prace z obrazem
import os #moduł umożliwiający prace z systemem operacyjnym, m.in zarządzanie plikami i katalogami
import json #biblioteka umożliwiająca zarządzanie plikami w formacie JSON
import matplotlib.pyplot as plt #zaimportowanie modułu pyplot z biblioteki matplotlib, umożliwiającego m.in wizualizacje obrazów
from model_loader import ModelLoader #zaimportowanie klasy zawierącej inicjalizacje gotowych modeli PyTorch

#funkcja pboierająca obraz i przekształcająca go na tensor, odpowiednio go transformująca
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  #otwarcie obrazu pod wskazana ścieżką, wykorzystanie modułu Image, przekształcenie obrazu na format RGB
        #jest to standardowy format wykorzystywany w przetwarzaniu obrazów i pracy z bibliotekami, zapewnia on że wszystkie zdjęcia mają 3 kanały kolorów (czerwony,zielony,niebieski), bez kanału przezroczystosci
        
    #wskazanie jak ma zostać zmieniony obraz na wejściu
    transform = transforms.Compose([ #połączenie obydwu transofrmacji w jedno
        transforms.Resize((224, 224)),  #przeskalowanie obrazu do rozmiaru 224x224, wartość ta jest kompatybilna z używanymi modelami
        transforms.ToTensor(), #przekształcenie obrazu na tensor (!patrz slownik.txt), czyli format danych obliczeniowych PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  #obraz zostaje znormalizowany, aby model Inception_v3 działal poprawnie,
        #model ten został wytrenowany na zbiorze ImageNet, przez co nowe dane muszą zostać do niego dopasowane, aby model działał prawidłowo
        #mean - średnia wartość do trzech kanałów RGB, std - odchylenie standardowe dla tych kanałów
    ])
    image = transform(image).unsqueeze(0)  #obraz zostaje przekształcony na tensor, dodany jest do niego nowy wymiar
    # tzw. wymiar wsadowy (batch dimension), model działa na partiach, funkcja unsqueeze(0) dodaje dodatkowy wymiar, aby obraz był potraktowany jako partia
    return image #zwrócenie odpowiednio załadowanego obrazu

#funkcja przetwarzająca dany obraz, klasyfikująca go przy użyciu podanego modelu (model)
#wybiera i zwraca ona top 5 najbardziej prawdopodobnych klas etykiet
def predict_top5(model, image):
    model.eval() #ustawienie modelu w tryb ewaluacji (!patrz slownik.txt), czyli tryb oceny wydajności modelu
#aby trenowanie było stabilne, ewaluacja deaktywuje niektóre neurony, np dla warstwy dropout która losowo ustawia wartości wejścia na 0
#ewaluacja umożliwia działanie warstwy batch_normalization, która przyśpiesza trening, dzięki normalizacji danych wejściowych m.in przez skalowanie ich

    with torch.no_grad():  #wyłączenie gradientów (!patrz slownik.txt) w celu oszczedzenia pamieci i szybszego działania, jest to zalecane podczas testowania modelu
        outputs = model(image)  #zapisanie do zmiennej output wyniku działania modelu
        probabilities = torch.softmax(outputs, dim=1)  #zamiana wyników modelu na wartość procentową (podanie prawdopodobieństwa, która klasa, czyli jaki album, została rozpoznany na zdjęciu)
        #jest to kluczowa operacja, podczas której wartości liczbowe zamieniane są na wartosci w przedziale [0,1] i sumowane do 1
        #parametr to klasyfikowany przez model obraz, oraz dim=1 wskazuje, że wszystkie przekształcone wartości mają sumować sie do 1
        
        top5_classes = torch.topk(probabilities, 5).indices.squeeze().tolist()  # #zwrócenie 5 klasy, dla których prawdopodobieństwo wystąpienia na zdjęciu jest największe, 
        #tym samym zwracane są album_id dla obrazu o danym prawdopodobieństwu za pomocą atrybutu indices
        #funkcja squeeze zamienia tensor obrazu na wektor o jedynym wymiarze, funkcja toList zamienia ten wektor na liste Pythona

    return top5_classes #zwrócenie wyników

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

#funkcja przetwarza wszystkie obrazy we wskazanym folderze, przeprowadza predykcje i zwraca jej wyniki 
#czyli indeks albumu, tytuł i wykonawce (!patrz slownik.txt)
def process_images_in_folder(folder_path, models, json_file): #parametry - folder, gdzie znajdują sie zdjecia, wskazanie używanego modelu, plik z etykietami
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
        
        #załadowanie obrazu dla podanej ścieżki
        image = load_image(image_path)

        #iteracyjne wyświetlenie wyników predykcji dla każdego modelu, pobieranie kolejnej wartości z listy models
        for model_name, model in models.items():
            top5_predictions = predict_top5(model, image) #w celu usprawnienia predykcji pobierane są wartości 5 najbardziej prawdopodobnych klas z opisanej wcześniej funkcji (album_id)
            #dla odpowiedniego modelu
            
            print(f"\nPredykcja dla modelu {model_name}:") #podanie informacji, który obraz jest aktualnie przetwarzany
            for rank, album_id in enumerate(top5_predictions, start=1):  #zwrócone albumy są numerowe od 1 do 5
                title, artist = get_album_info(album_id, json_file) #wykorzystanie wcześniej opisanej funkcji do pobrania tytułu i wykonawcy dla albumu o wskazanym id
                if title and artist:  #jeśli pobrane informacje istnieją, drukowany jest odpowiedni komunikat z wartościami
                    print(f"{rank}. id_albumu: {album_id}, Tytuł: '{title}', Wykonawca: {artist}")
                else: #jeśli nie znaleziono informacji na temat klasyfikowanego obrazu
                    print(f"{rank}. id_albumu: {album_id} (informacje o albumie nieznane)")

        img = Image.open(image_path) #otwarcie odpowiedniego obrazu
        plt.imshow(img) #wyświetlenie przetwarzanego obrazu
        plt.title(f"Obraz: {image_file}") #wydruk nazwy obrazu
        plt.show() #wyświetlenie obrazu i tytułu

#inicjalizacja obiektu klasy zawierającej odpowiednie modele, wykorzystane do klasyfikacji i predykcji
model_loader = ModelLoader(num_classes=39)

#lista zawierająca nazwe modeli i odpowiednie wywołanie funkcji
models = {
    "ResNet50": model_loader.load_resnet50(pretrained=True),
    "VGG16": model_loader.load_vgg16(pretrained=True),
    "AlexNet": model_loader.load_alexnet(pretrained=True),
    "EfficientNet B0": model_loader.load_efficientnet_b0(pretrained=True),
}

#podanie odpowiedniej ścieżki do folderu
folder_path = 'zadanie_projektowe/zdjecia_do_predykcji'

#podanie odpowiedniej ścieżki do pliku z etykietami
json_file = 'zadanie_projektowe/annotations.json'

#wykonanie predykcji - wywołanie powyższej funkcji
process_images_in_folder(folder_path, models, json_file)

