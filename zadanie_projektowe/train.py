import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss 
from dataset import AlbumDataset
from model import AlbumClassifier
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AlbumDataset(
    plik_json='zadanie_projektowe/annotations.json',
    zdjecia_kat='zadanie_projektowe/images',
    transform=transform
)

for album in dataset.etykiety:
    album_id = album['album_id']
    for zdj in album['zdjecia']:
        zdj['album_id'] = album_id  


dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

num_classes = len(set([zdj['album_id'] for album in dataset.etykiety for zdj in album['zdjecia']]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlbumClassifier(num_classes=num_classes).to(device)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:  
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

torch.save(model.state_dict(), 'album_classifier.pth')
