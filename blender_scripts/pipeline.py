import bpy
import argparse

def process_pokemon(input_path: str, output_path: str, pokemon_type: str):
    # 1. Limpiar escena y importar OBJ
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.import_scene.obj(filepath=input_path)
    
    # 2. Auto-rigging
    obj = bpy.context.active_object
    if pokemon_type == "biped":
        bpy.ops.pokemon.rig_biped()  # Implementar en auto_rigging.py
    else:
        bpy.ops.pokemon.rig_quadruped()
    
    # 3. Exportar SMD
    bpy.ops.export_scene.smd(
        filepath=output_path,
        use_selection=True,
        global_scale=100.0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", choices=["biped", "quadruped"], default="biped")
    args = parser.parse_args()
    
    process_pokemon(args.input, args.output, args.type)