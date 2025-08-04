import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils  # Instala con: pip install pointnet2-ops

class PokemonGenerator(nn.Module):
    def __init__(self, latent_dim=256, num_points=2048):
        super().__init__()
        self.num_points = num_points
        
        # Encoder de imágenes 2D (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64x64 -> 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256*16*16, latent_dim)
        )
        
        # Decoder de nube de puntos (PointNet++)
        self.point_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3),  # x, y, z para cada punto
            nn.Tanh()  # Normalizado entre [-1, 1]
        )
    
    def forward(self, img):
        # img: Tensor [batch, 3, 128, 128] (imagen RGB normalizada)
        latent = self.image_encoder(img)  # [batch, latent_dim]
        points = self.point_decoder(latent)  # [batch, num_points*3]
        return points.view(-1, self.num_points, 3)  # [batch, num_points, 3]

class PokemonDiscriminator(nn.Module):
    def __init__(self, num_points=2048):
        super().__init__()
        
        # Procesamiento de nubes de puntos (adaptado para Pokémon)
        self.pointnet = nn.Sequential(
            nn.Conv1d(3, 64, 1),  # Input: [batch, 3, num_points]
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1)  # [batch, 256, 1]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # Clasificador real/falso
        )
    
    def forward(self, points):
        # points: [batch, num_points, 3]
        points = points.permute(0, 2, 1)  # [batch, 3, num_points]
        features = self.pointnet(points).squeeze(-1)  # [batch, 256]
        return self.fc(features)  # [batch, 1]