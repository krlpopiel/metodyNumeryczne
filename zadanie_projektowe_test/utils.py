import matplotlib.pyplot as plt  # Biblioteka do wizualizacji danych
from torchvision.utils import make_grid  # Narzędzie do tworzenia siatek obrazów

def show_images(images, labels, predictions=None):  # Funkcja do wyświetlania obrazów
    plt.figure(figsize=(12, 8))  # Ustawienie rozmiaru figury
    grid = make_grid(images, nrow=4)  # Tworzenie siatki obrazów
    plt.imshow(grid.permute(1, 2, 0))  # Wyświetlenie obrazów w poprawnym układzie kanałów
    if predictions is not None:  # Jeśli dostępne są predykcje
        plt.title(f'Labels: {labels}, Predictions: {predictions}')  # Wyświetlenie etykiet i predykcji
    else:
        plt.title(f'Labels: {labels}')  # Wyświetlenie tylko etykiet
    plt.show()  # Wyświetlenie figury
