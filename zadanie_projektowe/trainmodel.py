import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_loader import ModelLoader
from dataset import AlbumDataset

batch_size = 32
num_epochs = 50
learning_rate = 0.001
num_classes = 39

model_choice = 'efficientnet_b0'  # 'resnet50' 'vgg16', 'alexnet', 'efficientnet_b0'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = AlbumDataset(plik_json='zadanie_projektowe/annotations.json', 
                             zdjecia_kat='zadanie_projektowe/images', 
                             transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_loader = ModelLoader(num_classes=num_classes)

if model_choice == 'resnet50':
    model = model_loader.load_resnet50(pretrained=True)
elif model_choice == 'vgg16':
    model = model_loader.load_vgg16(pretrained=True)
elif model_choice == 'alexnet':
    model = model_loader.load_alexnet(pretrained=True)
elif model_choice == 'efficientnet_b0':
    model = model_loader.load_efficientnet_b0(pretrained=True)
else:
    raise ValueError("Invalid model choice")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

model_save_path = f"model_{model_choice}.pth"
torch.save(model.state_dict(), model_save_path)

print(f'Model zapisany do {model_save_path}')