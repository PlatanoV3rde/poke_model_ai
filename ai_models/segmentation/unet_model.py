import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

class UNetPokemon(nn.Module):
    def __init__(self, num_classes=1):  # Segmentación binaria
        super().__init__()
        backbone = vgg16_bn(pretrained=True).features
        self.enc1 = nn.Sequential(*backbone[:6])
        self.enc2 = nn.Sequential(*backbone[6:13])
        self.enc3 = nn.Sequential(*backbone[13:23])
        self.enc4 = nn.Sequential(*backbone[23:33])
        self.enc5 = nn.Sequential(*backbone[33:43])
        
        # Decoder con skip connections
        self.dec1 = self._make_decoder(512, 256)
        self.dec2 = self._make_decoder(256, 128)
        self.dec3 = self._make_decoder(128, 64)
        self.dec4 = self._make_decoder(64, 32)
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_decoder(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        d1 = self.dec1(e5) + e4
        d2 = self.dec2(d1) + e3
        d3 = self.dec3(d2) + e2
        d4 = self.dec4(d3)
        
        return torch.sigmoid(self.final(d4))