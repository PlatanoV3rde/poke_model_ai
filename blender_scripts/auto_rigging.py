import bpy
from mathutils import Vector

def create_bone(name, head, tail, parent=None):
    """Crea un hueso y lo retorna."""
    bone = bpy.context.active_object.data.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    if parent:
        bone.parent = parent
    return bone

def auto_rig_pokemon(obj, rig_type='biped'):
    """
    Crea un esqueleto básico según el tipo de Pokémon:
    - 'biped': Para Pokémon bípedos (Lucario, Pikachu).
    - 'quad': Para cuadrúpedos (Arcanine, Raikou).
    """
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    armature = bpy.data.armatures.new(f"{obj.name}_Armature")
    rig = bpy.data.objects.new(f"{obj.name}_Rig", armature)
    bpy.context.scene.collection.objects.link(rig)
    
    # Forzamos a Blender a entrar en modo edición de armadura
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Hueso raíz (obligatorio para Source Engine)
    root_bone = create_bone("root", Vector((0, 0, 0)), Vector((0, 0, 1)))
    
    if rig_type == 'biped':
        # Esqueleto para bípedos
        spine = create_bone("spine", Vector((0, 0, 1.2)), Vector((0, 0, 2)), root_bone)
        create_bone("head", Vector((0, 0, 2)), Vector((0, 0, 2.5)), spine)
        # Brazos
        create_bone("arm_L", Vector((0, 0, 1.8)), Vector((1, 0, 1.8)), spine)
        create_bone("arm_R", Vector((0, 0, 1.8)), Vector((-1, 0, 1.8)), spine)
    elif rig_type == 'quad':
        # Esqueleto para cuadrúpedos
        spine = create_bone("spine", Vector((0, 0, 0.8)), Vector((0, 0, 1.5)), root_bone)
        create_bone("head", Vector((0, 0, 1.5)), Vector((0, 1, 1.5)), spine)
        # Patas
        create_bone("leg_FL", Vector((0.5, 0.5, 0.5)), Vector((1, 1, 0)), spine)
        create_bone("leg_FR", Vector((-0.5, 0.5, 0.5)), Vector((-1, 1, 0)), spine)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig

# Ejemplo de uso:
# obj = bpy.context.active_object
# auto_rig_pokemon(obj, rig_type='quad')