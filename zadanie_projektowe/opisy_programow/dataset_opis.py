import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
from torch.utils.data import Dataset #zaimportowanie klasy umożliwiającej definicje niestandarowych zbiorów danych
from torchvision import transforms #zaimportowanie modułu z pakietu Torchvision, umożliwiającego dokonanie przekształceń na obrazach 
from PIL import Image #zaimportowanie modułu Image z bilbioteki Pillow, umożliwiająca prace z obrazem
import json #biblioteka umożliwiająca zarządzanie plikami w formacie JSON
import os #moduł umożliwiający prace z systemem operacyjnym, m.in zarządzanie plikami i katalogami
import random #biblioteka umożliwiająca losowanie wartości
import matplotlib.patches as patches #zaimportowanie modułu patches z biblioteki matplotlib umożliwiającego rysowanie kształtów (obramówek) na obrazach
import matplotlib.pyplot as plt #zaimportowanie modułu pyplot z biblioteki matplotlib, umożliwiającego m.in wizualizacje obrazów

#definicja klasy dla niestandarowego datasetu, oparta jest ona na klasie bazowej Dataset, posłuży do wczytania obrazów i związanych z nimi informacji (etykiety)
class AlbumDataset(Dataset):
    #metoda umożliwiająca utworzenie obiektu danej klasy, przypisanie mu wartości 
    def __init__(self, plik_json, zdjecia_kat, transform=None):
        #parametry - odwołanie do aktualnego obiektu klasy, plik_json - nazwa pliku zawierajace dane z etykietami albumów, zdjecia_kat - ścieżka do katalogu ze zdjeciami,
        #transform - transformacje zdjęć, domyślny jest None (brak początkowych przekształceń)
        
        with open(plik_json, 'r') as f: #otwarcie pliku JSON w trybie do odczytu
            self.etykiety = json.load(f) #przypisanie wczytanych danych do atrybutu etykiety, za pomocą funkcji json.load() zawartość pliku przekształcana jest na obiekt Python (słownik)
        
        self.zdjecia_kat = zdjecia_kat #przypisanie ścieżki do katalogu zdjęć do atrybutu zdjecia_kat
        self.transform = transform #nadanie wartości (informacji jak ma zostać przekształcone zdjęcie) do atrybutu transform
        self.image_paths = [] #utworzenie pustej listy w obiekcie, przechowującej ścieżki do zdjęć w danym katalogu
        self.bboxes = [] #utworzenie pustej listy w obiekcie, przechowującej współrzedne obramówek okładek albumow na zdjęciach
        self.labels = [] #utworzenie pustej listy w obiekcie, przechowującej etykiety obrazów

        #iteracyjne zebranie informacji o albumie - id, nazwe pliku, bbox, ścieżkę
        for album in self.etykiety: #iteracja przez każdy element wcześniej utworzonej listy z etykietami z pliku JSON
            album_id = album['album_id'] #pobranie aktualnego id albumu z etykiety z pliku, przypisanie wartości do album_id
            for zdjecie in album['zdjecia']: #iteracja przez każde zdjęcie w pliku
                image_name = zdjecie['nazwaPliku'] + '.jpg'  #pobranie nazwy aktualnego zdjęcia z etykiety z pliku, dodając odpowiednie dla oryginalnych zdjeć rozszezrzenie jpg
                image_path = os.path.join(self.zdjecia_kat, image_name) #utworzenie pełnej ścieżki do aktualnego zdjęcia przy użyciu modułu os, zapewnia to poprawne działanie datasetu, bez ręcznego wpisywania całej scieżki
                bbox = zdjecie['bbox'] #pobranie aktualnych wspólrzednych obramówki obrazu z etykiety z pliku, przypisanie wartości do bbox
                #dodanie pobranych elementów do listy - scieżki do images_path, bbox do bboxes, id albumów do labels
                self.image_paths.append(image_path)
                self.bboxes.append(bbox)
                self.labels.append(album_id)

    #metoda zwracająca ilość zdjęć w datasecie
    def __len__(self): 
        return len(self.image_paths)

    #metoda pobierająca obraz i przypisująca mu odpowiednią etykietę na podstawie podanego indeksu (wskazuje na konkretne zdjecie)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB") #otwarcie obrazu o wskazanym id, wykorzystanie modułu Image, przekształcenie obrazu na format RGB
        #jest to standardowy format wykorzystywany w przetwarzaniu obrazów i pracy z bibliotekami, zapewnia on że wszystkie zdjęcia mają 3 kanały kolorów (czerwony,zielony,niebieski), bez kanału przezroczystosci
        
        bbox = self.bboxes[idx] #pobranie współrzędnych bbox dla wskazywanego zdjęcia
        label = self.labels[idx] #pobranie id_albumu dla wskazywanego zdjęcia
        
        left, upper, right, lower = bbox #podzielenie współrzędnych na 4 zmienne, odpowiednio z etykietą maja wartość [xmin, ymin, xmax, ymax]
        
         #ustawienie odpowiednich rozmiarów prostokąta - sprawdzenie czy szerokość i wysokość nie są <=0
        if right <= left:
            right = left + 1  #przesunięcie xmax o 1 piksel od xmin
        if lower <= upper:
            lower = upper + 1  #przesuniecie ymax o 1 piksel od ymin

        cropped_image = image.crop((left, upper, right, lower)) #przycięcie obrazu zgodnie z wyznaczonymi wyżej wartościami bbox

        #zastosowanie transformacji do przyciętego obrazu, jeśli została podana na wejściu, jeśli nie pozostaje domyślna None
        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label  #zwrócenie obrazu i id albumu

#wskazanie jak ma zostać zmieniony obraz na wejściu
transform = transforms.Compose([ #Compose łączy obydwie transformacje w jedną
    transforms.Resize((224, 224)),  #przeskalowanie obrazu do rozmiaru 224x224, taka wartość z racji kompatybilności z wieloma modelami z Pytorch
    transforms.ToTensor(),  #przekształcenie obrazu na tensor (!patrz slownik.txt), czyli format danych obliczeniowych PyTorch
])

#podanie wstępnych ścieżek do plików, na ich podstawie moduł os znajdzie dokładne ścieżki
plik_json = 'zadanie_projektowe/annotations.json'
zdjecia_kat = 'zadanie_projektowe/images'

#załadowanie zbioru danych, podanie odpowiednich wartości parametrów, obrazy będą miały zdefiniowane wcześniej transformacje
dataset = AlbumDataset(plik_json=plik_json, zdjecia_kat=zdjecia_kat, transform=transform)
"""
# TEST
dataset_length = len(dataset) #liczba elementów w zbiorze
random_index = random.randint(0, dataset_length - 1) #wylosowanie indeksu obrazu, losowy obraz zostanie wyświetlony
zdjecie, label = dataset[random_index] #pobranie obrazu etykiety dla wylosowanego indeksu

#utworzenie prostokąta na osi, wyświetlenie jednego obrazu
fig, ax = plt.subplots(1)
ax.imshow(zdjecie.permute(1, 2, 0))  #przekonwertowanie formatów tensora na wartości, które są kompatybilne z matplotlib, następuje zmiana z (H,W,C) NA (C,W,H)
# gdzie H to wysokość obrazu, W - szerokość obrazu, C - liczba kanałów formatu kolorów, liczby 1, 2, 0 oznaczają po prostu kolejność występowania tych paramterów

#wyświetlenie obrazu i jego id
plt.title(f"Album ID: {label}")
plt.show()
"""