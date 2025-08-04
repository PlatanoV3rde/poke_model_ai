import bpy

def create_pokemon_rig(obj, rig_type='biped'):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Crear armadura
    armature = bpy.data.armatures.new(f"{obj.name}_Armature")
    rig = bpy.data.objects.new(f"{obj.name}_Rig", armature)
    bpy.context.scene.collection.objects.link(rig)
    
    # Rigging básico
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones = []
    root = rig.data.edit_bones.new("root")
    root.head = (0, 0, 0)
    root.tail = (0, 0, 0.5)
    bones.append(root)
    
    if rig_type == 'biped':
        spine = rig.data.edit_bones.new("spine")
        spine.head = root.head
        spine.tail = (0, 0, 1.5)
        spine.parent = root
        bones.append(spine)
        
        # Añadir más huesos...
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig