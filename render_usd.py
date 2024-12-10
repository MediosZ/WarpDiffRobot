import os
import argparse
import bpy
import blendertoolbox as bt
import pathlib

parser = argparse.ArgumentParser(description="Render a USD file with Blender")
parser.add_argument("--usd-path", required=True, help="Path to the USD file", nargs='*')
args = parser.parse_args()

imgRes_x, imgRes_y = 1920, 1080
numSamples = 16
exposure = 1.5
use_GPU = True
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)
cwd = pathlib.Path(__file__).parent.absolute()

for path_to_usd in args.usd_path:
    bpy.ops.wm.usd_import(filepath=(cwd / path_to_usd).as_posix())

bt.invisibleGround(location = (0,0,-.5), shadowBrightness=0.9)

color_dict = {
    "blue": [152, 199, 255, 255],
    "green": [165, 221, 144, 255],
    "red": [255, 154, 156, 255],
    "orange": [243, 163, 124, 255],
    "brown": [216, 176, 107, 255],
}

RGBA = [x / 255.0 for x in color_dict['blue']]
meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)

for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        if 'mesh' in obj.name:
            mesh_obj = obj
            bt.setMat_plastic(mesh_obj, meshColor)
        elif obj.name == 'Plane':
            plane_obj = obj
            plane_obj.hide_render = True
        # hide the outer sphere
        elif obj.name == "shape_8":
            sphere_obj = obj
            sphere_obj.hide_render = True 
    elif obj.type == 'LIGHT':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.delete()


# ----- Camera attributes -----
camLocation = (1, -15, 2)
rotation_euler = (90, 0, 0)
cam = bt.setCamera_from_UI(camLocation, rotation_euler, focalLength = 45)

# ----- Light attributes
lightAngle = (6, -30, -155)
strength = 2
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
bt.setLight_ambient(color=(0.1,0.1,0.1,1))
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

outputFolder = cwd/ 'output' / 'usd_anim'
outputFolder.mkdir(parents=True, exist_ok=True)
outputPath = outputFolder / 'frame'
bpy.ops.wm.save_mainfile(filepath=(outputFolder/ 'usd_anim.blend').as_posix())

duration = bpy.context.scene.frame_end
bpy.context.scene.render.image_settings.file_format = 'PNG'
bt.renderAnimation(outputPath.as_posix(), cam, duration)

os.system("ffmpeg -r 48 -i 'output/usd_anim/frame%*.png' -c:v libx264 -r 48 -pix_fmt yuv420p output/usd_anim.mp4")
