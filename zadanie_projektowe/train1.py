import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from dataset import AlbumDataset
from models import ModelLoader  # Import klasy ModelLoader
from torchvision import transforms

# Przygotowanie danych
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ustawienie rozmiaru obrazu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja zgodna z ResNet50
])

dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',
    zdjecia_kat='zadanie_projektowe/images',
    transform=transform  # Ustawienie transformacji
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Inicjalizacja modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ładowanie modelu ResNet50 z ModelLoader
model_loader = ModelLoader(num_classes=38)  # Liczba klas w Twoim zadaniu
model = model_loader.get_resnet50(pretrained=True)  # Wybór modelu ResNet50 z wagami pretrenowanymi
model = model.to(device)  # Przeniesienie modelu na GPU/CPU

# Funkcja kosztu i optymalizator
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Trenowanie
num_epochs = 10
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
torch.save(model.state_dict(), 'resnet50_album_classifier.pth')
