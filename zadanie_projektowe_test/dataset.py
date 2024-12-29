import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

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

dataset = AlbumDataset(plik_json='C:/Users/user/OneDrive/Desktop/kodowanie/MN/MN_projekt/metodyNumeryczne/zadanie_projektowe_test/annotations.json', zdjecia_kat='C:/Users/user/OneDrive/Desktop/kodowanie/MN/MN_projekt/metodyNumeryczne/zadanie_projektowe_test/images_900x600', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
dataset_length=len(dataset)
print('Liczba trenowanych zdjęć:',dataset_length)
