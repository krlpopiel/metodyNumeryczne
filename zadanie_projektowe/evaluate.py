import torch
from dataset import AlbumDataset
from model import AlbumClassifier
from torch.utils.data import DataLoader

# Przygotowanie danych testowych
dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',
    zdjecia_kat='zadanie_projektowe/images',
    transform=None
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Wczytaj model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlbumClassifier().to(device)
model.load_state_dict(torch.load('album_classifier.pth')) #uwaga na sciezke
model.eval()

# Ewaluacja
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        targets = torch.tensor([0 if l == 'przod' else 1 for l in labels['widok']]).to(device)


        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
