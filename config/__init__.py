import os
import yaml
from pathlib import Path

def load_config():
    base_dir = Path(__file__).parent
    paths = yaml.safe_load((base_dir / "paths.yaml").read_text())
    blender = yaml.safe_load((base_dir / "blender_settings.yaml").read_text())
    
    # Resuelve variables de entorno y rutas
    def resolve_paths(d):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("${"):
                keys = v[2:-1].split('.')
                d[k] = paths
                for key in keys:
                    d[k] = d[k][key]
            elif isinstance(v, dict):
                resolve_paths(v)
    
    resolve_paths(paths)
    return {"paths": paths, "blender": blender}

config = load_config()