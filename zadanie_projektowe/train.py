import torch  # Framework do głębokiego uczenia
from torch.utils.data import DataLoader  # Narzędzie do ładowania danych w partiach
from torch.optim import Adam  # Optymalizator Adam
from torch.nn import CrossEntropyLoss  # Funkcja kosztu do klasyfikacji
from dataset import AlbumDataset  # Import niestandardowego zbioru danych
from model import AlbumClassifier  # Import modelu
from torchvision import transforms

# Przygotowanie danych
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ustawienie rozmiaru obrazu
    transforms.ToTensor()
])
# Przygotowanie danych
dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',  # Ścieżka do pliku JSON z etykietami
    zdjecia_kat='zadanie_projektowe/images',  # Katalog ze zdjęciami
    transform=None  # Brak dodatkowych transformacji (można dodać później)
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Ładowanie danych w partiach po 4, z losowaniem

# Inicjalizacja modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Wykorzystanie GPU, jeśli dostępne
model = AlbumClassifier().to(device)  # Przeniesienie modelu na GPU/CPU

# Funkcja kosztu i optymalizator
criterion = CrossEntropyLoss()  # Funkcja kosztu dla klasyfikacji wieloklasowej
optimizer = Adam(model.parameters(), lr=0.001)  # Optymalizator Adam z małym współczynnikiem uczenia

# Trenowanie
num_epochs = 10  # Liczba epok treningu
for epoch in range(num_epochs):  # Iteracja przez wszystkie epoki
    model.train()  # Przełączenie modelu w tryb treningowy
    running_loss = 0.0  # Inicjalizacja straty

    for images, labels in dataloader:  # Iteracja przez partie danych
        images = images.to(device)  # Przeniesienie obrazów na GPU/CPU
        targets = torch.tensor([0 if widok == 'przod' else 1 for widok in labels['widok']]).to(device)  # Przygotowanie etykiet

        # Forward pass
        outputs = model(images)  # Obliczenie wyników modelu
        loss = criterion(outputs, targets)  # Obliczenie straty

        # Backward pass
        optimizer.zero_grad()  # Wyzerowanie gradientów
        loss.backward()  # Obliczenie gradientów
        optimizer.step()  # Aktualizacja wag modelu

        running_loss += loss.item()  # Dodanie straty do sumy

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")  # Wyświetlenie straty po każdej epoce

# Zapisz model
torch.save(model.state_dict(), 'album_classifier.pth')  # Zapisanie wag modelu do pliku
