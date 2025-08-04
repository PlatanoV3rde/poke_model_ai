import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model_arch import PokemonGenerator, PokemonDiscriminator
from smd_loader import load_smd_as_points  # ¡Implementar esto!

class Pokemon3DDataset(Dataset):
    def __init__(self, smd_dir, img_dir, num_points=2048):
        self.smd_files = [f for f in os.listdir(smd_dir) if f.endswith('.smd')]
        self.smd_dir = smd_dir
        self.img_dir = img_dir
        self.num_points = num_points
    
    def __len__(self):
        return len(self.smd_files)
    
    def __getitem__(self, idx):
        # Carga .SMD y muestrea puntos
        smd_path = os.path.join(self.smd_dir, self.smd_files[idx])
        points = load_smd_as_points(smd_path, self.num_points)  # [num_points, 3]
        
        # Carga imagen 2D correspondiente (mismo nombre)
        img_name = os.path.splitext(self.smd_files[idx])[0] + '.png'
        img_path = os.path.join(self.img_dir, img_name)
        img = load_and_preprocess_img(img_path)  # Implementar
        
        return img, points

def train_gan():
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    num_points = 2048
    batch_size = 16
    epochs = 1000
    
    # Modelos
    generator = PokemonGenerator(latent_dim, num_points).to(device)
    discriminator = PokemonDiscriminator(num_points).to(device)
    
    # Optimizadores
    g_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Datos
    dataset = Pokemon3DDataset(
        smd_dir="data/smd_references",
        img_dir="data/images/train",
        num_points=num_points
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss
    criterion = nn.BCELoss()
    
    # Entrenamiento
    for epoch in range(epochs):
        for i, (imgs, real_points) in enumerate(dataloader):
            imgs = imgs.to(device)
            real_points = real_points.to(device)
            batch_size = imgs.size(0)
            
            # Etiquetas
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Entrena Discriminador
            d_optim.zero_grad()
            
            # Real
            d_real = discriminator(real_points)
            d_loss_real = criterion(d_real, real_labels)
            
            # Fake
            fake_points = generator(imgs)
            d_fake = discriminator(fake_points.detach())
            d_loss_fake = criterion(d_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optim.step()
            
            # Entrena Generador
            g_optim.zero_grad()
            g_loss = criterion(discriminator(fake_points), real_labels)
            g_loss.backward()
            g_optim.step()
            
            # Logs
            if i % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} "
                      f"d_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}")
        
        # Guarda checkpoints
        if epoch % 100 == 0:
            torch.save(generator.state_dict(), f"outputs/generator_epoch_{epoch}.pth")

if __name__ == "__main__":
    train_gan()