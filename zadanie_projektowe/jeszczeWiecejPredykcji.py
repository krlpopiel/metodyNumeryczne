import torch
from torchvision import transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from model_loader import ModelLoader

# Funkcja do wczytania obrazu i przekształcenia go na tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Zmiana rozmiaru na 224x224 (dla większości modeli)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizacja
    ])
    image = transform(image).unsqueeze(0)  # Dodanie wymiaru batch
    return image

# Funkcja do predykcji top 5 klas
def predict_top5(model, image):
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    with torch.no_grad():  # Brak obliczania gradientów
        outputs = model(image)  # Predykcja
        probabilities = torch.softmax(outputs, dim=1)  # Zamiana na prawdopodobieństwa
        top5_classes = torch.topk(probabilities, 5).indices.squeeze().tolist()  # Wybierz 5 najwyższych klas
    return top5_classes

# Funkcja do odczytywania informacji o albumie z pliku JSON
def get_album_info(album_id, json_file='zadanie_projektowe/annotations.json'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for album in data:
            if album['album_id'] == album_id:
                return album['tytul'], album['wykonawca']
    except FileNotFoundError:
        print(f"Plik {json_file} nie został znaleziony.")
    return None, None  # Jeśli album o takim ID nie zostanie znaleziony

# Funkcja do przetwarzania folderu ze zdjęciami
def process_images_in_folder(folder_path, models, json_file):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} nie istnieje.")
        return

    # Pobierz listę plików w folderze
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Brak zdjęć w folderze {folder_path}.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nPrzetwarzanie obrazu: {image_file}")
        
        # Załaduj obraz
        image = load_image(image_path)

        # Predykcja dla każdego modelu
        for model_name, model in models.items():
            top5_predictions = predict_top5(model, image)
            print(f"\nPredykcja dla modelu {model_name}:")
            for rank, album_id in enumerate(top5_predictions, start=1):  # Numerowanie od 1
                title, artist = get_album_info(album_id, json_file)
                if title and artist:
                    print(f"{rank}. id_albumu: {album_id}, Tytuł: '{title}', Wykonawca: {artist}")
                else:
                    print(f"{rank}. id_albumu: {album_id} (informacje o albumie nieznane)")

        # Wyświetlenie obrazu
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Obraz: {image_file}")
        plt.show()

# Inicjalizacja modelu loadera
model_loader = ModelLoader(num_classes=39)

# Załaduj modele
models = {
    "ResNet50": model_loader.load_resnet50(pretrained=True),
    "VGG16": model_loader.load_vgg16(pretrained=True),
    "AlexNet": model_loader.load_alexnet(pretrained=True),
    "EfficientNet B0": model_loader.load_efficientnet_b0(pretrained=True),
}

# Ścieżka do folderu ze zdjęciami
folder_path = 'zadanie_projektowe/zdjecia_do_predykcji'

# Ścieżka do pliku JSON z informacjami o albumach
json_file = 'zadanie_projektowe/annotations.json'

# Przetwarzanie wszystkich zdjęć w folderze
process_images_in_folder(folder_path, models, json_file)
