import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.png', '_mask.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Máscara en escala de grises
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convertir máscara a valores 0-1
        mask = (mask > 0.5).float()
        return image, mask