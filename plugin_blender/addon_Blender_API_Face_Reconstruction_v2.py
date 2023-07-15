bl_info = {
    "name": "Reconstruct 3D-Face",
    "author": "Hưng Nguyễn",
    "version": (2, 0),
    "blender": (2, 93, 0),
    "location": "Viewport > Right panel",
    "description": "Reconstruct 3d face from an 2d-face image",
    "category": "3D Face Reconstruction"
}

import bpy
from mathutils import Vector
import requests
from requests.exceptions import Timeout
from bpy.props import ( BoolProperty, EnumProperty, FloatProperty, PointerProperty )
from bpy.types import ( PropertyGroup )


global dis_obj, location
dis_obj = 0.4
location = Vector((1.0,0.0,0.0))

def delete_all_objects():
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')

    # Delete all selected objects
    bpy.ops.object.delete()

def send_request(url, data):
    try:
        response = requests.post(url, json=data, timeout=90)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def import_obj(obj_path,uv1_path,uv2_path):
    
    if bpy.context.window_manager.tool_3dFace.render_fine_texture:
        # Import the mesh with fine UV image
        bpy.ops.import_scene.obj(filepath=obj_path)

        # Get the imported object with default UV image
        object0 = bpy.context.selected_objects[0]
        
        object0.location = location
    
    if bpy.context.window_manager.tool_3dFace.render_coarse_texture:
        # Import the mesh with a original UV image
        bpy.ops.import_scene.obj(filepath=obj_path)

        # Get the imported object with the first UV image
        object1 = bpy.context.selected_objects[0]
        # define locations of object1: obj1.x = obj.x + dis_obj
        object1.location = (location.x + dis_obj, location.y, location.z)
        #object1.location = (location.x , location.y, location.z + dis_obj)
        
        # add uv to object1
        change_uv_of_obj(object1,uv1_path)

    if bpy.context.window_manager.tool_3dFace.render_original_texture:
        # Import the second mesh with another different UV image
        bpy.ops.import_scene.obj(filepath=obj_path)

        # Get the imported object with the second UV image
        object2 = bpy.context.selected_objects[0]
        # define locations of object2: obj2.x = obj.x - dis_obj
        object2.location = (location.x - dis_obj, location.y, location.z)
        #object2.location = (location.x , location.y, location.z - dis_obj)
        
        # add uv to object2
        change_uv_of_obj(object2,uv2_path)

def change_uv_of_obj(obj, image_path):
    # Get the first material assigned to the object
    material = obj.data.materials[0]

    # Check if the material uses nodes
    if material.use_nodes:
        # Find the Principled BSDF node
        principled_bsdf_node = None
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled_bsdf_node = node
                break

        if principled_bsdf_node is not None:
            # Load the UV image # Replace with the actual path to your UV image
            uv_image = bpy.data.images.load(image_path)

            # Create a new image texture node
            texture_node = material.node_tree.nodes.new('ShaderNodeTexImage')
            texture_node.image = uv_image

            # Remove existing links to the Principled BSDF node
            for link in material.node_tree.links:
                if link.to_node == principled_bsdf_node:
                    material.node_tree.links.remove(link)

            # Connect the texture node to the base color input of the Principled BSDF node
            material.node_tree.links.new(texture_node.outputs['Color'], principled_bsdf_node.inputs['Base Color'])
    
class Reconstruct(bpy.types.Operator):
    bl_idname = "object.reconstruct"
    bl_label = "3d Face Reconstruct"
    bl_description = ("Reconstruct 3D face with the chosen face-image")
    bl_options = {'PRESET'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        print('connecting')
        # Define the API endpoint URL
        api_url = 'http://127.0.0.1:5000/api/Reconstruct3dFace'
        
        # Define the input data
        data = {'image_path': self.filepath,
                'exp': context.window_manager.tool_3dFace.expression,
                'use_user_exp':  context.window_manager.tool_3dFace.use_user_expression, }
        
        self.report({'INFO'}, "processing")
        
        # Send the request to the API
        result = send_request(api_url, data)
        obj_path = result['obj_path']
        uv_3dmm_path = result['uv_3dmm_path']
        uv_org_path = result['uv_org_path']

        # Import the .obj file
        import_obj(obj_path,uv_3dmm_path,uv_org_path)
        
        print('successful!!')
        
        # fit objects fit on screen
        bpy.ops.view3d.view_all()
        
        # set 3d shading mode to "rendered"
        #bpy.context.space_data.shading.type = 'RENDERED'
        bpy.context.space_data.shading.type = 'MATERIAL'
        
        self.report({'INFO'}, "render successfully")
        return {'FINISHED'}
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class Import_expression(bpy.types.Operator):
    bl_idname = "object.import_expression"
    bl_label = "Provide expression"
    bl_description = ("choose a face-image that you want to provide expression")
    bl_options = {'PRESET'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        print('connecting')
        # Define the API endpoint URL
        api_url = 'http://127.0.0.1:5000/api/GetUserExpression'
        
        # Define the input data
        data = {'exp_path': self.filepath, }
        
        # Send the request to the API
        result = send_request(api_url, data)
        
        print(result)
        self.report({'INFO'}, result['result'])
        return {'FINISHED'}
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class FitObjectsToScreenOperator(bpy.types.Operator):
    bl_idname = "view3d.fit_objects_to_screen"
    bl_label = "Fit Objects to Screen"
    bl_description = ("Fit Objects to Screen")
    
    def execute(self, context):
        bpy.ops.view3d.view_all()
        return {'FINISHED'}

# Define the custom panel class
class _3dFacePanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_3dFacePanel"
    bl_label = "3d face reconstruction"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '3D Face Reconstruction'

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        # choose expression
        col.label(text="Choose expression:")
        col.prop(context.window_manager.tool_3dFace, "expression", text="")
        col.separator()
        
        col.prop(context.window_manager.tool_3dFace, "use_user_expression")
        # import expression
        col.operator("object.import_expression", text="Provide Expression")
        col.separator()
        col.separator()
        
        col.label(text="Render face with:")
        col.prop(context.window_manager.tool_3dFace, "render_fine_texture")
        col.prop(context.window_manager.tool_3dFace, "render_original_texture")
        col.prop(context.window_manager.tool_3dFace, "render_coarse_texture")
        col.separator()
        
        
        # Reconstruct 3D-Face
        col.operator("object.reconstruct", text="Reconstruct 3D-Face")
        col.separator()
        # fit all object to screen
        col.operator("view3d.fit_objects_to_screen", text="Fit Objects to Screen")
        
# Property group for UI sliders
class _3D_Properties(PropertyGroup):

    expression: EnumProperty(
        name = "Expression",
        description = "Create object with chosen expression",
        items = [ ("origin", "origin", ""), ("neutral", "neutral", ""), ("happy", "happy", ""),
                  ("sad", "sad", ""),
                  ("angry", "angry", ""),  ("surprise", "surprise", ""), ]
    )

    use_user_expression: BoolProperty(
        name = "User provided expression",
        description = "use/dont use expression in an image provided by user. If user dont provide yet, the default expression is neutral",
        default = False
    )
    
    render_fine_texture: BoolProperty(
        name = "Fine texture",
        description = "render object with fine texture",
        default = True
    )
    
    render_original_texture: BoolProperty(
        name = "Original texture",
        description = "render object with original texture",
        default = True
    )
    
    render_coarse_texture: BoolProperty(
        name = "Coarse texture",
        description = "render object with coarse texture",
        default = True
    )

classes = [
    _3D_Properties,
    Reconstruct,
    FitObjectsToScreenOperator,
    Import_expression,
    _3dFacePanel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Store properties under WindowManager (not Scene) so that they are not saved in .blend files and always show default values after loading
    bpy.types.WindowManager.tool_3dFace = PointerProperty(type=_3D_Properties)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.tool_3dFace


# Run the registration function
if __name__ == "__main__":
    register()