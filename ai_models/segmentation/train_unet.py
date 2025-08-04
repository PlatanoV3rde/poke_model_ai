import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import UNetPokemon
from utils.dataset import PokemonDataset
from utils.transforms import get_transforms

def train():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    epochs = 50
    
    # Datos
    train_transforms, val_transforms = get_transforms()
    train_dataset = PokemonDataset(
        img_dir='data/images/train',
        mask_dir='data/masks/train',  # Asume que tienes máscaras binarias
        transform=train_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo y optimizador
    model = UNetPokemon().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Para múltiples clases
    
    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Guardar modelo
    os.makedirs('ai_models/segmentation/weights', exist_ok=True)
    torch.save(model.state_dict(), 'ai_models/segmentation/weights/unet_pokemon.pth')

if __name__ == '__main__':
    train()