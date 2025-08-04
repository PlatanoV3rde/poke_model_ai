import cv2
import numpy as np
import torch
from .unet_model import UNetPokemon

class PokemonKeypointDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNetPokemon().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def detect(self, image_path):
        # Preprocesamiento
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Predicción
        with torch.no_grad():
            mask = self.model(tensor).squeeze().cpu().numpy()
        
        # Post-procesamiento
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Keypoints simplificados (centroide y esquinas del bounding box)
        keypoints = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            keypoints.extend([
                (x + w//2, y + h//2),  # Centro
                (x, y), (x + w, y), (x, y + h), (x + w, y + h)  # Esquinas
            ])
        
        return {
            'mask': mask_binary,
            'keypoints': keypoints,
            'bounding_box': (x, y, w, h) if contours else None
        }