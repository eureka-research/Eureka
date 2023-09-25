import os
from os.path import join

from jinja2 import Environment, select_autoescape, FileSystemLoader


def generate_assets(scales, min_volume, max_volume, generated_assets_dir, base_mesh):
    template_dir = join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/asset_templates")
    print(f'Assets template dir: {template_dir}')

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(),
    )
    template = env.get_template("cube_multicolor.urdf.template")

    cube_size_m = 0.05
    idx = 0
    for x_scale in scales:
        for y_scale in scales:
            for z_scale in scales:
                volume = x_scale * y_scale * z_scale / (100 * 100 * 100)
                if volume > max_volume:
                    continue
                if volume < min_volume:
                    continue

                curr_scales = [x_scale, y_scale, z_scale]
                curr_scales.sort()
                if curr_scales[0] * 3 <= curr_scales[1]:
                    # skip thin "plates"
                    continue

                asset = template.render(base_mesh=base_mesh,
                                        x_scale=cube_size_m * (x_scale / 100),
                                        y_scale=cube_size_m * (y_scale / 100),
                                        z_scale=cube_size_m * (z_scale / 100))
                fname = f"{idx:03d}_cube_{x_scale}_{y_scale}_{z_scale}.urdf"
                idx += 1
                with open(join(generated_assets_dir, fname), "w") as fobj:
                    fobj.write(asset)


def generate_small_cuboids(assets_dir, base_mesh):
    scales = [100, 50, 66, 75, 125, 150, 175, 200, 250, 300]
    min_volume = 0.75
    max_volume = 1.5
    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh)


def generate_big_cuboids(assets_dir, base_mesh):
    scales = [100, 125, 150, 200, 250, 300, 350]
    min_volume = 2.5
    max_volume = 15.0

    generate_assets(scales, min_volume, max_volume, assets_dir, base_mesh)
