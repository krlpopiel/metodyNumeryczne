import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class AlbumDataset(Dataset):
    def __init__(self, plik_json, zdjecia_kat, transform=None):
        with open(plik_json, 'r') as f:
            self.etykiety = json.load(f)
        
        self.zdjecia_kat = zdjecia_kat
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.zdjecia_info = []

        # Zbierz wszystkie id_albumu
        album_ids = set()
        for album in self.etykiety:
            album_ids.add(album['album_id'])
            for zdjecie in album['zdjecia']:
                self.zdjecia_info.append({
                    'album_id': album['album_id'],
                    'nazwaPliku': zdjecie['nazwaPliku']
                })
        
        # Mapowanie id_albumu na indeksy klas
        self.album_id_to_index = {album_id: idx for idx, album_id in enumerate(sorted(album_ids))}

    def __len__(self):
        return len(self.zdjecia_info)

    def __getitem__(self, index):
        zdj_info = self.zdjecia_info[index]
        nazwaPliku = zdj_info['nazwaPliku']
        
        if not nazwaPliku.endswith('.jpg'):
            nazwaPliku += '.jpg'
        
        zdj_sciezka = os.path.join(self.zdjecia_kat, nazwaPliku)
        zdjecie = np.array(Image.open(zdj_sciezka).convert("RGB"))
        if self.transform:
            zdjecie = self.transform(zdjecie)

        # Mapowanie id_albumu na indeks klasy
        etykieta = torch.tensor(self.album_id_to_index[zdj_info['album_id']], dtype=torch.long)
        return zdjecie, etykieta


# Ścieżki względne do pliku JSON i katalogu z obrazami
plik_json = 'zadanie_projektowe/annotations.json'  # Plik JSON w tym samym katalogu
zdjecia_kat = 'zadanie_projektowe/images'  # Katalog z obrazami w tym samym katalogu

dataset = AlbumDataset(plik_json=plik_json, zdjecia_kat=zdjecia_kat, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

dataset_length = len(dataset)
print('Liczba trenowanych zdjęć:', dataset_length)

liczba_klas = len(set([zdj['album_id'] for zdj in dataset.zdjecia_info]))

print('Liczba unikalnych do rozpoznania albumow: ', liczba_klas)