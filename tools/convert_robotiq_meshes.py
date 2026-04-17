"""Convert Robotiq 2F-85 DAE meshes to OBJ/STL for MuJoCo.

MuJoCo accepts OBJ/STL/MSH; DAE is not supported. The original xacro applies
scale=0.001 (mm→m) to every DAE mesh, so we bake that scaling into the output.
The base-link collision is an already-scaled STL (no scale in xacro) and is
copied verbatim.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import trimesh

SRC = Path("/tmp/robotiq_2finger_grippers/robotiq_2f_85_gripper_visualization/meshes")
DST = Path("/tmp/iiwa7-mujoco/iiwa7_mjcf/meshes_2f85")
SCALE = 0.001  # xacro scale="0.001 0.001 0.001"


def load_scaled(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    mesh.apply_scale(SCALE)
    return mesh


def convert_dae_to_obj(src: Path, dst: Path) -> None:
    mesh = load_scaled(src)
    mesh.export(dst)
    print(f"  {src.name} -> {dst.name}  verts={len(mesh.vertices)} bounds={mesh.extents}")


def convert_dae_to_stl(src: Path, dst: Path) -> None:
    mesh = load_scaled(src)
    mesh.export(dst)
    print(f"  {src.name} -> {dst.name}  verts={len(mesh.vertices)} bounds={mesh.extents}")


def main() -> None:
    vis_out = DST / "visual"
    col_out = DST / "collision"
    vis_out.mkdir(parents=True, exist_ok=True)
    col_out.mkdir(parents=True, exist_ok=True)

    visual_files = [
        "robotiq_arg2f_85_base_link",
        "robotiq_arg2f_85_outer_knuckle",
        "robotiq_arg2f_85_outer_finger",
        "robotiq_arg2f_85_inner_knuckle",
        "robotiq_arg2f_85_inner_finger",
        "robotiq_arg2f_85_pad",
    ]
    print("[visual]")
    for name in visual_files:
        convert_dae_to_obj(SRC / "visual" / f"{name}.dae", vis_out / f"{name}.obj")

    print("[collision]")
    # base_link: STL already in meters (xacro has no explicit scale).
    base_stl = SRC / "collision" / "robotiq_arg2f_base_link.stl"
    base_mesh = trimesh.load(base_stl, force="mesh")
    print(f"  base_link.stl raw bounds={base_mesh.extents}")
    # If bounds are in mm (>1m), scale. Otherwise copy verbatim.
    if np.max(base_mesh.extents) > 1.0:
        base_mesh.apply_scale(SCALE)
        base_mesh.export(col_out / "robotiq_arg2f_85_base_link.stl")
        print(f"  -> rescaled, new bounds={base_mesh.extents}")
    else:
        shutil.copy(base_stl, col_out / "robotiq_arg2f_85_base_link.stl")
        print(f"  -> copied verbatim")

    for name in [
        "robotiq_arg2f_85_outer_knuckle",
        "robotiq_arg2f_85_outer_finger",
        "robotiq_arg2f_85_inner_knuckle",
        "robotiq_arg2f_85_inner_finger",
    ]:
        convert_dae_to_stl(SRC / "collision" / f"{name}.dae", col_out / f"{name}.stl")

    print("\nDone. Outputs:")
    for p in sorted(DST.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(DST)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
