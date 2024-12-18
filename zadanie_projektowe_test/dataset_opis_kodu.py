import os #biblioteka umożliwiająca m.in zarządzanie plikami i katalogami
import torch #biblioteka importująca framework PyTorch, wykorzystywana do uczenia maszynowego
import json #biblioteka umożliwiająca zarządzanie plikami w formacie JSON
import numpy as np #standardowa bilbioteka do operacji numerycznych
from PIL import Image #zaimportowanie modułu Image z bilbioteki Pillow, umożliwiająca prace z obrazem
from torch.utils.data import DataLoader #zaimportowanie klasy umożliwiajacej ładowanie danych do modelu, wykorzystanych z odpowiedniego zbioru
from torch.utils.data import Dataset #zaimportowanie klasy umożliwiającej definicje niestandarowych zbiorów danych
from torchvision import transforms #zaimportowanie modułu umożliwiającego prze
import random #biblioteka umożliwiająca losowanie wartości
import matplotlib.pyplot as plt #biblioteka umożliwiająca m.in wizualizacje obrazów

class AlbumDataset(Dataset): #definicja klasy dla niestandarowego datasetu, oparta jest ona na klasie bazowej Dataset, posłuży do wczytania obrazów i związanych z nimi informacji (etykiety)
    def __init__ (self,plik_json,zdjecia_kat,transform=None): #metoda umożliwiająca utworzenie obiektu danej klasy, przypisanie mu wartości 
#parametry - odwołanie do aktualnego obiektu klasy, plik_json - nazwa pliku zawierajace dane z etykietami albumów, zdjecia_kat - ścieżka do katalogu ze zdjeciami, transform - transformacje zdjęć, domyślny jest None (brak początkowych przekształceń)
        with open(plik_json, 'r') as f: #otwarcie pliku JSON w trybie do odczytu
            self.etykiety = json.load(f) #przypisanie wczytanych danych do atrybutu etykiety, za pomocą funkcji json.load() zawartość pliku przekształcana jest na obiekt Python (słownik)
            
        self.zdjecia_kat = zdjecia_kat #przypisanie ścieżki do katalogu zdjęć do atrybutu zdjecia_kat
        self.transform = transforms.Compose([ #przypisanie do atrubutu self.transform transformacji Compose (pozwala ona na łączenie wielu transformacji w jedną)
            transforms.ToTensor() #dane wejściowe zostają przekonwertowane na tensory (uogólnione wektory), piksele skalowane są do wartości z zakresu [0,1]
        ])
        self.zdjecia_info = [] #pusta lista, która będzie przechowywała informacje (etykiety) zdjęć

#poniższe pętle iterują bo zbiorze danych z JSON (przegląd listy z atrybutu etykiety), odpowiednim kluczą słownika przypisywane są wartości z listy album bądź zdjęcie (pary klucz-wartość)
        for album in self.etykiety:
            album_id = album['album_id'] #para klucz wartość, gdzie album_id to klucz, album['album_id'] to wartość
            tytul = album['tytul']
            wykonawca = album['wykonawca']
            for zdjecie in album['zdjecia']:
                nazwaPliku = zdjecie['nazwaPliku']
                widok = zdjecie['widok']
                orientacja = zdjecie['orientacja']
                bbox = zdjecie['bbox']
                opis = zdjecie['opis']
                self.zdjecia_info.append({ #dodanie uwtorzonych elementów słownika do listy zdjęcia_info
                    'album_id': album_id, #pary klucz-wartość
                    'tytul': tytul,
                    'wykonawca': wykonawca,
                    'nazwaPliku': nazwaPliku,
                    'widok': widok,
                    'orientacja': orientacja,
                    'bbox': bbox,
                    'opis': opis
                })
       
    def __len__(self): #metoda zwracająca ilość zdjęć w datasecie
        return len(self.zdjecia_info)

    def __getitem__(self, index): #metoda pobierająca obraz i przypisująca mu odpowiednią etykietę na podstawie podanego indeksu
        zdj_info = self.zdjecia_info[index] #zmienna przechowująca dane obrazu (przypisane są do niej wartości ze słownika, dla danego obrazu - index)
        nazwaPliku = zdj_info['nazwaPliku'] #pobranie nazwyPliku dla obrazu ze słownika zdj_info, używając odpowiedniego klucza
 
        if not nazwaPliku.endswith('.jpg'): #zabezpieczenie sprawdzajace czy plik ze zdjęciem ma rozszerzenie jpg, jeśli nie, dodaje je na końcu 
            nazwaPliku += '.jpg'
        zdj_sciezka= os.path.join(self.zdjecia_kat, nazwaPliku) #uwtorzenie całej śnieżki do odpowiedniego pliku w folderze ze zdjęciami

        zdjecie = np.array(Image.open(zdj_sciezka).convert("RGB")) #otwarcie zdjęcie za pomocą funkcji z bilbioteki Pillow, każde zdjęcie zostaje przekonwertowane
        # do przestrzeni RGB, w celu ujednolicenia danych i prostrzego procesu przetwarzania / obraz zostaje przekształcony na tablicę - odpowiedni format to użycia z Pytorch
        if self.transform: #sprawdzenie czy istnieje funkcja przekształcająca dany obraz, jeśli tak następuje transformacja
            zdjecie = self.transform(zdjecie)

        etykiety = { #utworzenie slownika etykieta, który będzie zwracany w metodzie
            'album_id': torch.tensor(zdj_info['album_id'], dtype=torch.long), #następuje konwersja wartości, aby mogły one zostać użyte w modelu Pytorch
            'tytul': zdj_info['tytul'],
            'wykonawca': zdj_info['wykonawca'],
            'nazwaPliku': zdj_info['nazwaPliku'],
            'widok': zdj_info['widok'],
            'orientacja': zdj_info['orientacja'],
            'bbox': torch.tensor(zdj_info['bbox'], dtype=torch.float), #współrzędne bramki konwertowane są do float (transformacje mogą wpływać na wartość stąd typ zmiennoprzecinkowy)
            'opis': zdj_info['opis']
        }
        
        return zdjecie,etykiety #zwrócenie obrazu i odpowiednich dla niego etykiet

#załadowanie zbioru danych, podanie odpowiednich wartości parametrów, obrazy zostaną przekonwertowane do tensorów
dataset = AlbumDataset(plik_json='annotations.json', zdjecia_kat='images_450x300', transform=transforms.ToTensor())
#utworzenie obiektu ładującego dane w batchach (partiach), parametry to - utworzony dataset, liczba próbek w jednej partii (w jednej partii będą 4 zdjęcia)
#shuffle sprawi, że dane w zbiorze treningowym będą losowo tasowane z każdym przejściem zbioru danych - aby model nie uzależnił się od określonej, uporządkowanej kolejności obrazów 
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
dataset_length=len(dataset) #obliczenie ile obrazów znajduje się w zbiorze
print('Liczba trenowanych zdjęć:',dataset_length)

random_index = random.randint(0, dataset_length - 1) #losowanie numeru indeksu obrazu z dostępnego zbioru danych
zdjecie, etykiety = dataset[random_index] #do wylosowanego indeksu dobierany jest odpowiadający mu obraz i etykiety
album_id = etykiety['album_id'] #pobranie odpowiednich wartości ze słownika z etykietami
tytul = etykiety['tytul']
wykonawca = etykiety['wykonawca']
widok = etykiety['widok']
orientacja = etykiety['orientacja']
print(f"Wylosowany obraz:\nId albumu: {album_id}\nTytuł: {tytul}\nWykonawca: {wykonawca}\nWidok: {widok}\nOrientacja: {orientacja}")
plt.imshow(zdjecie.permute(1, 2, 0)) #aby wyświetlić obraz należy przekonwertować format z tensora -> (C,H,W) na format (H,W,C) oczekiwany przez matplot
#C - liczba kanałów, wartość przestrzenie kolorów, H - wysokość w pikselach, W - szerokość w pikselach
plt.show() #wyświetlenie obrazu
