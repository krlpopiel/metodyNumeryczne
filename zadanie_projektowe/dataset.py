import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class AlbumDataset(Dataset):
    def __init__(self, plik_json, zdjecia_kat, transform=None):
        with open(plik_json, 'r') as f:
            self.etykiety = json.load(f)
        
        self.zdjecia_kat = zdjecia_kat
        self.transform = transform
        self.image_paths = []
        self.bboxes = []
        self.labels = []

        for album in self.etykiety:
            album_id = album['album_id']
            for zdjecie in album['zdjecia']:
                image_name = zdjecie['nazwaPliku'] + '.jpg' 
                image_path = os.path.join(self.zdjecia_kat, image_name)
                bbox = zdjecie['bbox']
                self.image_paths.append(image_path)
                self.bboxes.append(bbox)
                self.labels.append(album_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        bbox = self.bboxes[idx]
        label = self.labels[idx]
        
        left, upper, right, lower = bbox
        if right <= left:
            right = left + 1 
        if lower <= upper:
            lower = upper + 1

        cropped_image = image.crop((left, upper, right, lower))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

plik_json = 'zadanie_projektowe/annotations.json'
zdjecia_kat = 'zadanie_projektowe/images'

dataset = AlbumDataset(plik_json=plik_json, zdjecia_kat=zdjecia_kat, transform=transform)
"""
# TEST
dataset_length = len(dataset)
random_index = random.randint(0, dataset_length - 1)
zdjecie, label = dataset[random_index]  

# Rysowanie obrazu (bez bbox)
fig, ax = plt.subplots(1)
ax.imshow(zdjecie.permute(1, 2, 0))  # Konwersja z Tensor na obraz

plt.title(f"Album ID: {label}")
plt.show()
"""