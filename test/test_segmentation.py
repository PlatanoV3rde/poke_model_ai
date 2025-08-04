import unittest
import cv2
from ai_models.segmentation.detect_keypoints import PokemonKeypointDetector

class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PokemonKeypointDetector("ai_models/segmentation/weights/unet_pokemon.pth")
        cls.test_img = "data/images/test/pikachu.png"

    def test_keypoints_detection(self):
        result = self.detector.detect(self.test_img)
        self.assertIsNotNone(result['mask'])
        self.assertGreater(len(result['keypoints']), 0)
        
if __name__ == '__main__':
    unittest.main()