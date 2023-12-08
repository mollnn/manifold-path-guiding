import argparse
import glob
import os
import sys

import numpy as np
import mitsuba

# For some images, render in specific mode (e.g. polarized)
mode_override = {
    "bsdf_polarizer_aligned":   "scalar_spectral_polarized",
    "bsdf_polarizer_absorbing": "scalar_spectral_polarized",
    "bsdf_polarizer_middle":    "scalar_spectral_polarized",
    "integrator_stokes_cbox":   "scalar_mono_polarized",
    "variants_cbox_rgb":        "scalar_rgb",
    "variants_cbox_spectral":   "scalar_spectral",
}


def load_scene(filename, *args, **kwargs):
    """Prepares the file resolver and loads a Mitsuba scene from the given path."""
    from mitsuba.core.xml import load_file
    from mitsuba.core import Thread

    fr = Thread.thread().file_resolver()
    here = os.path.dirname(__file__)
    fr.append(here)
    fr.append(os.path.join(here, filename))
    fr.append(os.path.dirname(filename))

    scene = load_file(filename, *args, **kwargs)
    assert scene is not None
    return scene


def render(scene, write_to):
    from mitsuba.core import Bitmap, Struct

    success = scene.integrator().render(scene, scene.sensors()[0])
    assert success

    film = scene.sensors()[0].film()
    bitmap = film.bitmap(raw=False)

    if not bitmap.pixel_format() == Bitmap.PixelFormat.MultiChannel:
        bitmap.convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)
    elif bitmap.channel_count() == 16:
        # Stokes output, rather specialized for 'integrator_stokes_cbox' scene atm.
        data_np = np.array(bitmap, copy=False).astype(np.float)
        s0 = data_np[:, :, 4]
        z = np.zeros(s0.shape)
        s1 = np.dstack([np.maximum(0, -data_np[:, :, 7]),  np.maximum(0, data_np[:, :, 7]),  z])
        s2 = np.dstack([np.maximum(0, -data_np[:, :, 10]), np.maximum(0, data_np[:, :, 10]), z])
        s3 = np.dstack([np.maximum(0, -data_np[:, :, 13]), np.maximum(0, data_np[:, :, 13]), z])
        Bitmap(s0).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to)
        Bitmap(s1).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s1.jpg'))
        Bitmap(s3).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s3.jpg'))
        Bitmap(s2).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True).write(write_to.replace('.jpg', '_s2.jpg'))
    else:
        for name, b in bitmap.split():
            if name == '<root>':
                continue

            # normalize depth map
            if name == 'depth.y':
                data_np = np.array(b, copy=False)
                min_val = np.min(data_np)
                max_val = np.max(data_np)
                data_np = (data_np - min_val) / (max_val - min_val)
                b = Bitmap(data_np, Bitmap.PixelFormat.Y)

            pixel_format = b.pixel_format()
            if not pixel_format == Bitmap.PixelFormat.XYZ:
                pixel_format = Bitmap.PixelFormat.RGB

            f_name = write_to if name == 'image' else write_to.replace('.jpg', '_%s.jpg' % name)
            b.convert(pixel_format, Struct.Type.UInt8, True).write(f_name)


def main(args):
    parser = argparse.ArgumentParser(prog='RenderDocImages')
    parser.add_argument('--force', action='store_true',
                        help='Force rerendering of all documentation images')
    parser.add_argument('--spp', default=1, type=int,
                        help='Samples per pixel')
    args = parser.parse_args()

    spp = args.spp
    force = args.force
    images_folder = os.path.join(os.path.dirname(__file__), '../images/render')
    os.makedirs(images_folder, exist_ok=True)
    scenes = glob.glob(os.path.join(os.path.dirname(__file__), '*.xml'))
    for scene_path in scenes:
        scene_name = os.path.split(scene_path)[-1][:-4]
        img_path = os.path.join(images_folder, scene_name + ".jpg")
        if not os.path.isfile(img_path) or force:
            print(scene_path)
            if scene_name in mode_override.keys():
                mitsuba.set_variant(mode_override[scene_name])
            else:
                mitsuba.set_variant("scalar_spectral")

            scene_path = os.path.abspath(scene_path)
            scene = load_scene(scene_path, spp=spp)
            render(scene, img_path)


if __name__ == "__main__":
    main(sys.argv[1:])
