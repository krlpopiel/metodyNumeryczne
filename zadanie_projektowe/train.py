import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss  # Używamy CrossEntropyLoss dla klasyfikacji
from dataset import AlbumDataset
from model import AlbumClassifier
from torchvision import transforms

# Przygotowanie danych
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Dopasowanie rozmiaru
    transforms.ToTensor(),
])

# Tworzymy dataset
dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',
    zdjecia_kat='zadanie_projektowe/images',
    transform=transform
)

# Modyfikacja etykiet, aby były w zakresie od 1 do 38
for album in dataset.etykiety:
    album_id = album['album_id']  # Pobieramy album_id z albumu
    for zdj in album['zdjecia']:
        zdj['album_id'] = album_id  # Ustawiamy album_id w każdym zdjęciu


# Tworzenie DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Inicjalizacja modelu
num_classes = len(set([zdj['album_id'] for album in dataset.etykiety for zdj in album['zdjecia']]))  # Liczba unikalnych albumów
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlbumClassifier(num_classes=num_classes).to(device)

# Funkcja kosztu i optymalizator
criterion = CrossEntropyLoss()  # CrossEntropyLoss dla klasyfikacji
optimizer = Adam(model.parameters(), lr=0.001)

# Trenowanie
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:  
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Zapisz model
torch.save(model.state_dict(), 'album_classifier.pth')
