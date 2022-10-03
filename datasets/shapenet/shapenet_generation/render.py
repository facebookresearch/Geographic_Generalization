"""
Render png using blender from .obj 

based on https://github.com/panmari/stanford-shapenet-renderer

1. Activate blender: module load [blender]
2. Run:
~/blender-3.0.0-linux-x64/blender --python-use-system-env --background -noaudio --python render.py -- [args]
"""
import sys
import argparse

import bpy
import os
import inspect
import pathlib
from typing import Tuple, Optional
import numpy as np
import math
import itertools
from pathlib import Path

from mathutils import Matrix, Vector


USER = os.getenv("USER")


def add_packages_to_path():
    # TODO: this is ugly and does not work with Blender 2.81 (cluster version), we need to solve this problem to use matplotlib (and generally, the other librairies)
    # sys.path.insert(0,"/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/")

    # insert current library
    current_file = Path(inspect.getfile(inspect.currentframe()))
    package_dir = current_file.parent.parent.parent.absolute()
    sys.path.append(str(package_dir))
    # insert site packages
    if USER == "dianeb":
        sys.path.insert(
            0,
            "/private/home/dianeb/.conda/envs/videovariation/lib/python3.9/site-packages/",
        )
    elif USER == "marksibrahim":
        sys.path.insert(
            0,
            "/private/home/marksibrahim/.conda/envs/video-variation/lib/python3.9/site-packages/",
        )

    else:
        raise ValueError(f"USER {USER} environment not recognized")


add_packages_to_path()
import matplotlib
from datasets.shapenet_generation.attributes import Views
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv


def construct_obj_mat(obj):
    """
    Decompose world_matrix's components, and from them assemble 4x4 matrices
    """
    orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
    orig_loc_mat = Matrix.Translation(orig_loc)
    orig_rot_mat = orig_rot.to_matrix().to_4x4()
    orig_scale_mat = (
        Matrix.Scale(orig_scale[0], 4, (1, 0, 0))
        @ Matrix.Scale(orig_scale[1], 4, (0, 1, 0))
        @ Matrix.Scale(orig_scale[2], 4, (0, 0, 1))
    )
    return orig_loc_mat, orig_rot_mat, orig_scale_mat


class ShapeNetRenderer:
    """Renders obj files as pngs with specified views.

    Note all views specified in degrees.

    Args:
        obj_path: path containing model.obj file
        out_dir: directory to save images
    """

    def __init__(
        self,
        obj_path: str,
        out_dir: str,
        resolution: int = 600,
        scale: float = 1.0,
        use_gpu: bool = False,
    ):
        self.views = Views()

        self.obj_path = obj_path
        self.out_dir = out_dir
        self.resolution = resolution
        self.scale = scale

        self.context = bpy.context
        self.cam = None
        self.model = None

        # steps to run
        self.setup_renderer()
        self.insert_object()
        self.add_light()
        self.setup_camera()

        if use_gpu:
            self.setup_gpu()

    def setup_gpu(self):
        # Set the device_type
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"
        for scene in bpy.data.scenes:
            scene.cycles.device = "GPU"

    def setup_renderer(self):
        """Sets up renderer and scene"""
        render = bpy.context.scene.render

        render.engine = "CYCLES"
        render.image_settings.color_mode = "RGB"  # ('RGB', 'RGBA', ...)
        render.image_settings.color_depth = "8"  # 8 or 16
        render.image_settings.file_format = "PNG"  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
        render.resolution_x = self.resolution
        render.resolution_y = self.resolution
        render.resolution_percentage = 100
        render.film_transparent = True

        scene = bpy.context.scene
        scene.cycles.samples = 1024
        scene.use_nodes = True
        scene.view_layers["ViewLayer"].use_pass_normal = True
        scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
        scene.view_layers["ViewLayer"].use_pass_object_index = True

        self._delete_default_cube()
        self.object_rotation_order = "XYZ"

    def _delete_default_cube(self):
        bpy.data.objects["Cube"]
        self.context.active_object.select_set(True)
        bpy.ops.object.delete()

    def insert_object(self):
        """insert ShapeNet in scene"""
        bpy.ops.import_scene.obj(filepath=self.obj_path)

        # rescale object
        bpy.ops.transform.resize(value=(self.scale, self.scale, self.scale))
        bpy.ops.object.transform_apply(scale=True)

    def add_light(self):
        # make light just directional, disable shadows.
        light = bpy.data.lights["Light"]
        light.type = "SUN"
        light.use_shadow = False
        # disable specular shading:
        light.specular_factor = 1.0
        light.energy = 10.0

        # add another light source so stuff facing away from light is not completely dark
        bpy.ops.object.light_add(type="SUN")
        light2 = bpy.data.lights["Sun"]
        light2.use_shadow = False
        light2.specular_factor = 1.0
        light2.energy = 10.0
        bpy.data.objects["Sun"].rotation_euler = bpy.data.objects[
            "Light"
        ].rotation_euler
        bpy.data.objects["Sun"].rotation_euler[0] += 180
        bpy.data.objects["Sun"].rotation_euler[
            1
        ] += 180  # TODO: we should better think about the lightning, what do we want exactly?
        bpy.data.objects["Sun"].rotation_euler[2] += 180

    def setup_camera(self):
        self.cam = bpy.data.objects["Camera"]
        self.cam.location = (0, 2, 0)

        looking_direction = self.cam.location - Vector((0, 0, 0))
        rot_quat = looking_direction.to_track_quat("Z", "Y")
        self.cam.rotation_euler = rot_quat.to_euler()

        self.model = bpy.data.objects["model_normalized"]
        self.context.view_layer.objects.active = self.model
        self.reset_view()

    def reset_view(self):
        """Resets view to original"""
        self.model.rotation_euler = self.views.canonical

    @property
    def view(self) -> Optional[Tuple[int]]:
        """Returns (angle, angle, angle) denoting object's current view.
        Note orientation is specified in degrees
        """
        radians = self.model.rotation_euler
        angles = tuple([int(math.degrees(r)) for r in radians])
        return angles

    @view.setter
    def view(self, angles: Tuple[int]):
        """Sets view based on given angles. Angles are specified in degrees."""

        # convert angles to radians
        radians = tuple([math.radians(float(a)) for a in angles])
        try:
            self.model.rotation_euler.order = self.object_rotation_order
            # define the rotation
            rot_mat = Matrix(np.identity(4))
            for i in range(3):
                ax = self.object_rotation_order[i]
                angle = radians["XYZ".index(ax)]
                rot_mat_ax = Matrix.Rotation(angle, 4, ax)
                rot_mat = rot_mat_ax @ rot_mat

            orig_loc_mat, orig_rot_mat, orig_scale_mat = construct_obj_mat(self.model)
            self.model.matrix_world = (
                orig_loc_mat @ rot_mat @ orig_rot_mat @ orig_scale_mat
            )
        except SyntaxError:
            raise SyntaxError("cannot access object in scene")

    def parse_model_name(self) -> str:
        """Assumes model name is in parent directory of models.

        Example: [model_name]/models/model.obj
        """
        path = pathlib.Path(self.obj_path)
        model_name = path.parent.parent.name
        return model_name

    def render(self, view_name, had_transfo=False):
        """Saves png for the current view."""
        view_name = "_".join([str(i) for i in view_name])
        model_name = self.parse_model_name()
        order = self.model.rotation_euler.order
        file_path = os.path.join(
            self.out_dir, model_name, f"{model_name}_{had_transfo}_{view_name}_{order}"
        )

        scene = bpy.context.scene
        scene.render.filepath = file_path

        bpy.ops.render.render(write_still=True)
        return file_path

    def per_axis_render_views(
        self,
        view_start=0.0,
        view_end=360.0,
        num_views=4,
        synset_id=" ",
        obj_id=" ",
        order="XYZ",
        had_transfo=False,
        csv_writer=None,
        init_transfo=(0, 0, 0),
    ):
        """Saves pngs for the range of views, including both endpoints,
            ranging over all views on a per-axis basis

        Args:
            view_start: float denoting starting orientation in degrees
            view_end: float denoting ending orientation in degrees
            num_views: number of views to render
        """

        # TODO Here, I play with the order to obtain all possible combinations
        # How to pass the order as an argument to view setter? Having it as an attribute is ugly
        # Should order affect all the objects (light, camera, etc)
        self.object_rotation_order = order
        init_matrix = self.model.matrix_world.copy()
        views = Views(
            view_start=view_start,
            view_end=view_end,
            num_views=num_views,
            order="XYZ",
            canonical=init_transfo,
        )
        self.views = views

        for view in tqdm(views.generate(), desc="view ", leave=False):
            # Reset obj_matrix to initial position
            self.model.matrix_world = init_matrix.copy()
            self.view = view
            path = self.render(view, had_transfo)
            if csv_writer is not None:
                row = [synset_id, obj_id, path + ".png"] + [i for i in view]
                csv_writer.writerow(row)

    def gather_all_views(
        self,
        view_start: float = 0.0,
        view_end: float = 360.0,
        num_views=4,
        model_name=None,
        exclude: int = 2,
        extension: str = "png",
    ):
        """Gather pngs for the range of views, including both endpoints,
            for two axess all views possible, in a single plot.
            This does not work for more than 2 axes (2D plot).

        Args:
            view_start: float denoting starting orientation in degrees
            view_end: float denoting ending orientation in degrees
            num_views: number of views to render
            model_name: name of the model to retrieve plots
            exclude: excluded axis (from the 2D plot)
        """

        n = num_views - 1
        _, axes = plt.subplots(n, n, figsize=(30, 30))

        if model_name is None:
            model_name = self.parse_model_name()

        list_views = np.linspace(view_start, view_end, num_views)[:-1]
        a = [list_views, list_views]
        a.insert(exclude, [0.0])
        for k, view in enumerate(list(itertools.product(*a))):
            x = k % n
            y = k // n
            self.view = view
            view_name = "_".join([str(v) for v in self.view])
            file_path = os.path.join(
                self.out_dir, f"{model_name}_{view_name}.{extension}"
            )
            img = mpimg.imread(file_path)
            ax1 = axes[x, y]
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.set_title(" ".join(view_name.split("_")), fontsize=16)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect("equal")
            ax1.imshow(img)

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig("plot.png")


def main(
    path_info,
    out_dir,
    num_views=4,
    order="XYZ",
    init_transfo=(0, 0, 0),
    csv_writer=None,
):
    """Saves pngs for the range of views, including both endpoints.

    Args:
        view_start: tuple denoting starting orientation in degrees
        view_end: tuple denoting ending orientation in degrees
        num_views: number of views to render
        order: order of the rotation in 3d
        init_transfo: initial transformation that changes canonical pose
        csv_writer: csv writer to fov file
    """
    path = os.path.join(*[i for i in path_info])
    renderer = ShapeNetRenderer(path, out_dir)
    # Potential initial transformation applied to all views
    # Can be used to change the canonical orientation of the object
    if init_transfo != (0, 0, 0):
        renderer.object_rotation_order = order
        renderer.view = init_transfo
        had_transfo = True
    else:
        had_transfo = False
    _, synset_id, obj_id, _ = path_info
    renderer.per_axis_render_views(
        0,
        360,
        num_views,
        synset_id,
        obj_id,
        order=order,
        had_transfo=had_transfo,
        csv_writer=csv_writer,
        init_transfo=init_transfo,
    )


if __name__ == "__main__":
    default_out_dir = f"/checkpoint/{USER}/datasets/tmp"

    parser = argparse.ArgumentParser(
        description="Renders given obj file by rotation a camera around it."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets01/ShapeNetCore.v2/080320",
    )
    parser.add_argument(
        "--synset_id",
        type=str,
        default="02691156",
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        default="4561def0c651631122309ea5a3ab0f04",
    )
    parser.add_argument(
        "--obj_path",
        type=str,
        default="models/model_normalized.obj",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help="directory where pngs will be saved",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="number of views to render",
    )
    parser.add_argument(
        "--init_transfo",
        type=int,
        nargs="+",
        default=[0, 0, 0],
        help="initial_transfo that modifies canonical orientation",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="XYZ",
        help="order of rotation",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=f"{default_out_dir}/fov.csv",
        help="order of rotation",
    )

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    args.init_transfo = tuple([int(i) for i in args.init_transfo])

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.csv_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        path_info = (args.data_dir, args.synset_id, args.obj_id, args.obj_path)
        main(
            path_info,
            args.out_dir,
            num_views=args.num_views,
            order=args.order,
            init_transfo=args.init_transfo,
            csv_writer=writer,
        )
