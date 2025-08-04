import numpy as np
import struct

def read_smd(smd_path):
    """
    Lee un archivo .SMD y extrae vértices.
    Formato esperado: SMD de Source Engine (versión binaria o ASCII).
    Retorna:
        - vertices: np.array de shape [N, 3] (coordenadas x, y, z)
        - triangles: np.array de shape [M, 3] (índices de vértices)
    """
    vertices = []
    triangles = []
    
    with open(smd_path, 'rb') as f:  # Modo binario para compatibilidad
        # Implementación simplificada (adaptar según tus .SMD)
        # Ejemplo para SMD ASCII (necesitarás parsing más robusto):
        for line in f:
            if line.startswith(b'v '):  # Línea de vértice
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            elif line.startswith(b'tri '):  # Línea de triángulo
                parts = line.strip().split()
                v0, v1, v2 = int(parts[1]), int(parts[2]), int(parts[3])
                triangles.append([v0, v1, v2])
    
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)

def load_smd_as_points(smd_path, num_points=2048):
    """Convierte .SMD a nube de puntos muestreada."""
    vertices, _ = read_smd(smd_path)
    
    if len(vertices) == 0:
        raise ValueError(f"No se encontraron vértices en {smd_path}")
    
    # Muestreo aleatorio o interpolación
    if len(vertices) >= num_points:
        indices = np.random.choice(len(vertices), num_points, replace=False)
        points = vertices[indices]
    else:
        # Repetir puntos si el modelo es muy bajo-poly
        repeat = (num_points // len(vertices)) + 1
        points = np.tile(vertices, (repeat, 1))[:num_points]
    
    return points.astype(np.float32)