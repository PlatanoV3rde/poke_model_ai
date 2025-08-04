import cv2
import numpy as np
import torch
from unet_model import UNetPokemon

class PokemonKeypointDetector:
    def __init__(self, model_path='ai_models/segmentation/weights/unet_pokemon.pth'):
        self.model = UNetPokemon()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def detect(self, image_path):
        # Preprocesamiento
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_transforms()[1]  # Usar transforms de validación
        input_tensor = transform(image).unsqueeze(0)
        
        # Segmentación
        with torch.no_grad():
            mask = self.model(input_tensor).squeeze().numpy()
        
        # Encontrar contornos (para bounding box)
        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detección de keypoints (simplificado)
        keypoints = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                keypoints.append((cx, cy))  # Centroide como keypoint principal
        
        return {
            'mask': mask,
            'keypoints': keypoints,
            'contours': contours
        }

# Ejemplo de uso:
# detector = PokemonKeypointDetector()
# result = detector.detect('data/images/test/pikachu.png')