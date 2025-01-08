import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class AlbumDataset(Dataset):
    def __init__ (self,plik_json,zdjecia_kat,transform=None):
        with open(plik_json, 'r') as f:
            self.etykiety = json.load(f)
            
        self.zdjecia_kat = zdjecia_kat
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.zdjecia_info = []


        for album in self.etykiety:
            album_id = album['album_id']
            tytul = album['tytul']
            wykonawca = album['wykonawca']
            for zdjecie in album['zdjecia']:
                nazwaPliku = zdjecie['nazwaPliku']
                widok = zdjecie['widok']
                orientacja = zdjecie['orientacja']
                bbox = zdjecie['bbox']
                opis = zdjecie['opis']
                self.zdjecia_info.append({
                    'album_id': album_id,
                    'tytul': tytul,
                    'wykonawca': wykonawca,
                    'nazwaPliku': nazwaPliku,
                    'widok': widok,
                    'orientacja': orientacja,
                    'bbox': bbox,
                    'opis': opis
                })
       
    def __len__(self):
        return len(self.zdjecia_info)

    def __getitem__(self, index):
        zdj_info = self.zdjecia_info[index]
        nazwaPliku = zdj_info['nazwaPliku']
        
        if not nazwaPliku.endswith('.jpg'):
            nazwaPliku += '.jpg' 
        zdj_sciezka= os.path.join(self.zdjecia_kat, nazwaPliku)
        
        zdjecie = np.array(Image.open(zdj_sciezka).convert("RGB"))
        if self.transform:
            zdjecie = self.transform(zdjecie)

        etykiety = {
            'album_id': torch.tensor(zdj_info['album_id'], dtype=torch.long),
            'tytul': zdj_info['tytul'],
            'wykonawca': zdj_info['wykonawca'],
            'nazwaPliku': zdj_info['nazwaPliku'],
            'widok': zdj_info['widok'],
            'orientacja': zdj_info['orientacja'],
            'bbox': torch.tensor(zdj_info['bbox'], dtype=torch.float),
            'opis': zdj_info['opis']
        }
        
        return zdjecie,etykiety
#uwazac na sciezke!!!
dataset = AlbumDataset(plik_json='zadanie_projektowe/annotations.json', zdjecia_kat='zadanie_projektowe/images', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
dataset_length=len(dataset)
print('Liczba trenowanych zdjęć:',dataset_length)

random_index = random.randint(0, dataset_length - 1)
zdjecie, etykiety = dataset[random_index]
album_id = etykiety['album_id']
tytul = etykiety['tytul']
wykonawca = etykiety['wykonawca']
widok = etykiety['widok']
orientacja = etykiety['orientacja']
bbox = etykiety['bbox'].numpy()  # Bounding box jako numpy array
print(f"Wylosowany obraz:\nId albumu: {album_id}\nTytuł: {tytul}\nWykonawca: {wykonawca}\nWidok: {widok}\nOrientacja: {orientacja}")
# Rysowanie obrazu z bounding boxem
fig, ax = plt.subplots(1)
ax.imshow(zdjecie.permute(1, 2, 0))  # Konwersja z Tensor na obraz

# Dodanie bounding boxa
rect = patches.Rectangle(
    (bbox[0], bbox[1]),  # Lewy górny róg (x, y)
    bbox[2] - bbox[0],   # Szerokość
    bbox[3] - bbox[1],   # Wysokość
    linewidth=2, edgecolor='r', facecolor='none'
)
ax.add_patch(rect)

plt.title(f"Tytuł: {tytul}, Wykonawca: {wykonawca}")
plt.show()