import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("poke_model_ai")

class PokeModelAI:
    def __init__(self, config_path: str = "configs/paths.yaml"):
        """Carga configuraciones e inicializa módulos."""
        self.config = self._load_config(config_path)
        self._verify_paths()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carga archivos YAML de configuración."""
        config_dir = Path(__file__).parent / "configs"
        with open(config_dir / "paths.yaml") as f:
            paths = yaml.safe_load(f)
        with open(config_dir / "blender_settings.yaml") as f:
            blender = yaml.safe_load(f)
        
        # Resuelve rutas relativas
        base_dir = Path(__file__).parent
        for key, value in paths["paths"].items():
            if isinstance(value, str) and value.startswith("./"):
                paths["paths"][key] = str(base_dir / value[2:])
        
        return {"paths": paths["paths"], "blender": blender}

    def _verify_paths(self):
        """Verifica que las rutas críticas existan."""
        required_paths = [
            self.config["paths"]["smd_references"],
            self.config["paths"]["images"]["train"],
            self.config["paths"]["blender_scripts"]
        ]
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ruta crítica no encontrada: {path}")

    def generate_from_image(self, image_path: str, pokemon_type: str = "biped"):
        """
        Pipeline completo desde imagen 2D hasta .SMD:
        1. Segmentación de la imagen
        2. Generación 3D con GAN
        3. Rigging y exportación a .SMD
        """
        try:
            logger.info(f"Iniciando generación para {image_path}")
            
            # 1. Segmentación
            mask_path = self._segment_image(image_path)
            
            # 2. Generación 3D
            temp_obj = self._generate_3d_model(mask_path)
            
            # 3. Procesamiento en Blender
            output_smd = self._process_in_blender(
                input_obj=temp_obj,
                pokemon_type=pokemon_type,
                output_name=Path(image_path).stem
            )
            
            logger.info(f"Modelo generado en {output_smd}")
            return output_smd
            
        except Exception as e:
            logger.error(f"Error en el pipeline: {str(e)}")
            raise

    def _segment_image(self, image_path: str) -> str:
        """Usa el módulo de segmentación para aislar al Pokémon."""
        from ai_models.segmentation.detect_keypoints import PokemonKeypointDetector
        detector = PokemonKeypointDetector(self.config["paths"]["unet_weights"])
        result = detector.detect(image_path)
        
        # Guarda máscara para uso posterior
        output_dir = self.config["paths"]["outputs"] / "masks"
        os.makedirs(output_dir, exist_ok=True)
        mask_path = output_dir / f"{Path(image_path).stem}_mask.png"
        cv2.imwrite(str(mask_path), result["mask"])
        
        return mask_path

    def _generate_3d_model(self, mask_path: str) -> str:
        """Genera modelo 3D usando la GAN entrenada."""
        from ai_models.3d_gan.generator import generate_from_mask
        output_path = self.config["paths"]["generated_3d"] / f"{Path(mask_path).stem}.obj"
        generate_from_mask(
            mask_path=mask_path,
            model_path=self.config["paths"]["gan_weights"],
            output_path=output_path
        )
        return output_path

    def _process_in_blender(self, input_obj: str, pokemon_type: str, output_name: str) -> str:
        """Importa, riggea y exporta el modelo usando scripts de Blender."""
        import subprocess
        
        blender_script = self.config["paths"]["blender_scripts"] / "pipeline.py"
        output_smd = self.config["paths"]["converted_smd"] / f"{output_name}.smd"
        
        cmd = [
            self.config["paths"]["blender_install"]["windows"],  # Ajustar según OS
            "--background",
            "--python", str(blender_script),
            "--",
            "--input", str(input_obj),
            "--output", str(output_smd),
            "--type", pokemon_type
        ]
        
        subprocess.run(cmd, check=True)
        return output_smd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Ruta a la imagen PNG del Pokémon")
    parser.add_argument("--type", choices=["biped", "quadruped"], default="biped")
    args = parser.parse_args()

    try:
        pipeline = PokeModelAI()
        result = pipeline.generate_from_image(args.image_path, args.type)
        print(f"¡Modelo generado con éxito! Ruta: {result}")
    except Exception as e:
        logger.critical(f"Fallo en el pipeline: {str(e)}")
        exit(1)