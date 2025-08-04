import bpy
import os

def export_smd(output_path, apply_modifiers=True):
    """
    Exporta el modelo activo a .SMD con ajustes para Source Engine.
    Args:
        output_path: Ruta completa del archivo de salida.
        apply_modifiers: Aplica modificadores (ej: subdivision surface).
    """
    if not output_path.endswith('.smd'):
        output_path += '.smd'
    
    obj = bpy.context.active_object
    if not obj:
        raise Exception("¡No hay objeto seleccionado!")
    
    # Configuración recomendada para Pokémon
    bpy.ops.export_scene.smd(
        filepath=output_path,
        use_selection=True,
        apply_modifiers=apply_modifiers,
        global_scale=100.0,  # Compensa la escala de importación
        bone_roll_mode='SOURCE',  # Compatibilidad con Source Engine
        use_animations=False      # No exportar animaciones por defecto
    )
    print(f"Modelo exportado a {output_path}")

# Ejemplo de uso:
# export_smd("C:/ruta/a/tu/outputs/converted_smd/charizard.smd")