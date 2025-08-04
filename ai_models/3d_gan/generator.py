import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

class PokemonGenerator(nn.Module):
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256*16*16, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points*3),
            nn.Tanh()
        )

    def forward(self, img):
        latent = self.encoder(img)
        points = self.decoder(latent)
        return points.view(-1, self.num_points, 3)

def generate_from_mask(mask_path, model_path, output_path):
    """Genera OBJ a partir de máscara usando GAN entrenada"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PokemonGenerator().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Preprocesar máscara (simulado)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) / 255.0
    mask_tensor = mask_tensor.to(device)
    
    # Generar puntos
    with torch.no_grad():
        points = model(mask_tensor).squeeze().cpu().numpy()
    
    # Guardar como OBJ
    with open(output_path, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")