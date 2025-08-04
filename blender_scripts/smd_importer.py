import bpy
import os
from mathutils import Vector

def clean_scene():
    """Elimina todos los objetos existentes en la escena."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_smd(smd_path):
    """Carga un archivo .SMD usando SourceIO (debes instalarlo manualmente)."""
    try:
        bpy.ops.import_scene.smd(filepath=smd_path)
        print(f"Modelo {os.path.basename(smd_path)} cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar {smd_path}: {str(e)}")

def center_origin_to_geometry(obj):
    """Centra el origen del objeto en su geometría."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

def import_pokemon_smd(smd_path, scale=0.01):
    """
    Importa un .SMD y aplica ajustes para Pokémon:
    - Escala estándar (porque Source Engine usa unidades grandes)
    - Rota el modelo para eje Z arriba.
    """
    clean_scene()
    load_smd(smd_path)
    
    pokemon_obj = bpy.context.selected_objects[0]
    pokemon_obj.scale = Vector((scale, scale, scale))
    pokemon_obj.rotation_euler = (1.5708, 0, 0)  # 90 grados en X
    
    center_origin_to_geometry(pokemon_obj)
    return pokemon_obj

# Ejemplo de uso (descomenta para probar):
# import_pokemon_smd("C:/ruta/a/tu/data/smd_references/pikachu.smd")