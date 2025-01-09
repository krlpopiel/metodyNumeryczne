import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import AlbumClassifier  # Zaimportuj klasę modelu

# Załaduj model
model = AlbumClassifier()  # Użyj tej samej klasy modelu, co w czasie trenowania
model.load_state_dict(torch.load('album_classifier.pth'))  # Załaduj stan modelu
model.eval()  # Przełącz model na tryb testowy

def test_model_on_image(image_path, model):
    # Załaduj obrazek i wykonaj odpowiednie transformacje
    transform = transforms.Compose([
        transforms.Resize((900, 600)),  # Dopasuj rozmiar do modelu
        transforms.ToTensor(),  # Przekształć obraz na tensor
    ])

    image = Image.open(image_path).convert("RGB")  # Załaduj obrazek
    image = transform(image).unsqueeze(0)  # Dodaj wymiar wsadowy (batch dimension)

    # Sprawdź kształt obrazu
    print(f"Shape of image: {image.shape}")

    # Przewidywanie
    with torch.no_grad():
        output = model(image)  # Przekaż obraz przez model
        _, predicted = torch.max(output, 1)  # Uzyskaj predykcję

    return predicted.item()  # Zwróć przewidywaną klasę

# Ścieżka do zdjęcia, które chcesz przetestować
image_path = 'zadanie_projektowe/zdjecia_do_predykcji/test.jpg'

# Testowanie modelu na obrazie
predicted_class = test_model_on_image(image_path, model)
print(f'Predykcja dla zdjęcia {image_path}: id_albumu {predicted_class}')