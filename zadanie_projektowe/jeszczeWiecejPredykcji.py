import torch
from torchvision import transforms
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from model_loader import ModelLoader

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    image = transform(image).unsqueeze(0) 
    return image

def predict_top5(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        top5_classes = torch.topk(probabilities, 5).indices.squeeze().tolist() 
    return top5_classes

def get_album_info(album_id, json_file='zadanie_projektowe/annotations.json'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for album in data:
            if album['album_id'] == album_id:
                return album['tytul'], album['wykonawca']
    except FileNotFoundError:
        print(f"Plik {json_file} nie został znaleziony.")
    return None, None 

def process_images_in_folder(folder_path, models, json_file):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} nie istnieje.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"Brak zdjęć w folderze {folder_path}.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"\nPrzetwarzanie obrazu: {image_file}")
        
        image = load_image(image_path)

        for model_name, model in models.items():
            top5_predictions = predict_top5(model, image)
            print(f"\nPredykcja dla modelu {model_name}:")
            for rank, album_id in enumerate(top5_predictions, start=1):
                title, artist = get_album_info(album_id, json_file)
                if title and artist:
                    print(f"{rank}. id_albumu: {album_id}, Tytuł: '{title}', Wykonawca: {artist}")
                else:
                    print(f"{rank}. id_albumu: {album_id} (informacje o albumie nieznane)")

       
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Obraz: {image_file}")
        plt.show()

# Dodajemy funkcję do ładowania wag z plików .pth
def load_model_with_weights(model_class, num_classes, weights_path):
    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path))  # Wczytanie wag z pliku .pth
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    return model

# Przykładowe ścieżki do plików wag dla modeli
weights_paths = {
    "ResNet50": "model_resnet50.pth",
    "VGG16": "model_vgg16.pth",
    "AlexNet": "model_alexnet.pth",
    "EfficientNet B0": "model_efficientnet_b0.pth",
}

# Inicjalizacja modelu loadera
model_loader = ModelLoader(default_num_classes=39)

models = {
    "ResNet50": model_loader.load_resnet50(weights_path="model_resnet50.pth"),
    "VGG16": model_loader.load_vgg16(weights_path="model_vgg16.pth"),
    "AlexNet": model_loader.load_alexnet(weights_path="model_alexnet.pth"),
    "EfficientNet B0": model_loader.load_efficientnet_b0(weights_path="model_efficientnet_b0.pth"),
}


folder_path = 'zadanie_projektowe/zdjecia_do_predykcji'

json_file = 'zadanie_projektowe/annotations.json'

process_images_in_folder(folder_path, models, json_file)
