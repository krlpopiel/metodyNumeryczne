import torch
from torchvision import transforms
from PIL import Image
import os
import json
from model import AlbumClassifier  # Zaimportuj klasę modelu
import matplotlib.pyplot as plt

# Załaduj model z odpowiednią liczbą klas
num_classes = 38  # Liczba klas (id_albumu)
model = AlbumClassifier(num_classes=num_classes)  # Użyj tej samej klasy modelu, co w czasie trenowania
model.load_state_dict(torch.load('album_classifier.pth'))  # Załaduj stan modelu
model.eval()  # Przełącz model na tryb testowy

# Funkcja do wczytania informacji o albumie z pliku JSON
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

# Funkcja do przetwarzania obrazu i zwracania 5 najbardziej prawdopodobnych klas
def test_model_on_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Dopasuj rozmiar do modelu
        transforms.ToTensor(),  # Przekształć obraz na tensor
    ])

    if not os.path.exists(image_path):
        print(f"Plik {image_path} nie istnieje.")
        return None

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Dodaj wymiar wsadowy

    with torch.no_grad():
        output = model(image)  # Wynik modelu
        probabilities = torch.softmax(output, dim=1)  # Zamiana na prawdopodobieństwa
        top5_classes = torch.topk(probabilities, 5).indices.squeeze().tolist()  # Wybierz 5 najwyższych klas

    return top5_classes

# Funkcja do przetwarzania folderu ze zdjęciami
def process_images_in_folder(folder_path, model, json_file):
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

        # Uzyskaj 5 najbardziej prawdopodobnych klas
        results = test_model_on_image(image_path, model)

        # Wyświetlenie obrazu
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Obraz: {image_file}")
        plt.show()

        if results is not None:
            for rank, album_id in enumerate(results, start=1):  # Numerowanie od 1
                album_title, album_artist = get_album_info(album_id, json_file)
                if album_title and album_artist:
                    print(f"{rank}. id_albumu: {album_id}, Tytuł: '{album_title}', Wykonawca: {album_artist}")
                else:
                    print(f"{rank}. id_albumu: {album_id} (informacje o albumie nieznane)")
        else:
            print(f"Nie udało się przetworzyć obrazu {image_file}")

# Ścieżka do folderu ze zdjęciami
folder_path = 'zadanie_projektowe/zdjecia_do_predykcji'

# Ścieżka do pliku JSON z informacjami o albumach
json_file = 'zadanie_projektowe/annotations.json'

# Przetwarzanie wszystkich zdjęć w folderze
process_images_in_folder(folder_path, model, json_file)
