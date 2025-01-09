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

        #zebranie wszystkich id albumów
        album_ids = set() #swtorzenie pustego zbioru, który będzie przechowywał unikalne identyfikatory albumów
        for album in self.etykiety: #iteracja przez wszystkie elementy w self.etykiety, które są załadowane z pliku JSON
            album_ids.add(album['album_id']) #pobranie identyfikatora albumu i dodanie go do utworzonego zbioru
            for zdjecie in album['zdjecia']: #ateracja przez listę zdjęć w danym albumie 
                self.zdjecia_info.append({ #swtorzenie słownik z informacjami o zdjęciu i dodanie go do listy, zawiera on id albumu i nazwe pliku
                    'album_id': album['album_id'],
                    'nazwaPliku': zdjecie['nazwaPliku']
                })
        
        #mapowanie id albumów na indeksy klas, id są sortowane rosnąco (dzięki funkcji enumerate), każde id zyskuje unikalny indeks
        #album_id: idx for idx tworzy słownik -> klucz album_id, wartość indeks
        #mapowanie umożliwi klasyfikacje przy użyciu modeli, inkdeksy klas są wymagane do ich poprawnej pracy
        self.album_id_to_index = {album_id: idx for idx, album_id in enumerate(sorted(album_ids))}
       
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

         #stworzenie tensora dla PyTorch, który reprezentuje etykiete klasy zdjęcia / pobranie id_zdjęcia z listy informacji o zdjęciu
         #i znalezenie indeksu klasu odpowiedniego do id albumu / określenie typu formatu etykiety jako całkowity
        etykieta = torch.tensor(self.album_id_to_index[zdj_info['album_id']], dtype=torch.long)
        
        return zdjecie,etykieta #zwrócenie obrazu i odpowiednich dla niego etykiet

#ścieżki względne do pliku JSON i katalogu z obrazami
plik_json = 'zadanie_projektowe/annotations.json'  # Plik JSON w tym samym katalogu
zdjecia_kat = 'zadanie_projektowe/images'  # Katalog z obrazami w tym samym katalogu

#załadowanie zbioru danych, podanie odpowiednich wartości parametrów, obrazy zostaną przekonwertowane do tensorów
dataset = AlbumDataset(plik_json='zadanie_projektowe/annotations.json', zdjecia_kat='zadanie_projektowe/images', transform=transforms.ToTensor())
#utworzenie obiektu ładującego dane w batchach (partiach), parametry to - utworzony dataset, liczba próbek w jednej partii (w jednej partii będą 4 zdjęcia)
#shuffle sprawi, że dane w zbiorze treningowym będą losowo tasowane z każdym przejściem zbioru danych - aby model nie uzależnił się od określonej, uporządkowanej kolejności obrazów 
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
dataset_length=len(dataset) #obliczenie ile obrazów znajduje się w zbiorze
print('Liczba trenowanych zdjęć:',dataset_length)

#wyrażenie listowe, tworzy liste zawierającą wszystkie id_albumów dla każdego zdjęcia / pobranie wartość album_id ze słownika zdj
liczba_klas = len(set([zdj['album_id'] for zdj in dataset.zdjecia_info]))

print('Liczba unikalnych do rozpoznania albumow: ', liczba_klas)
