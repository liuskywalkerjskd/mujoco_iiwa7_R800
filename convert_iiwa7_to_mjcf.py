#!/usr/bin/env python3
"""Convert iiwa7 xacro/URDF to MuJoCo MJCF.

Pipeline:
  1. Expand iiwa7.urdf.xacro -> plain URDF, resolving $(find iiwa_description)
     against the local filesystem (no ROS install required).
  2. Strip ROS-specific <gazebo> / <transmission> elements.
  3. Rewrite package://iiwa_description/meshes/... mesh paths to mujoco-compatible
     relative paths, and inject a <mujoco> block with <compiler meshdir=...>.
  4. Let mujoco's compiler ingest the cleaned URDF and emit MJCF.

Run:
  conda activate <env with mujoco + xacro installed>
  python3 convert_iiwa7_to_mjcf.py

Outputs (relative to this script):
  ../mjcf/iiwa7/iiwa7.urdf   — cleaned URDF (intermediate artifact)
  ../mjcf/iiwa7/iiwa7.xml    — final MJCF
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import mujoco  # type: ignore
import xacro  # type: ignore
import xacro.substitution_args as subs  # type: ignore

HERE = Path(__file__).resolve().parent
IIWA_DESC = HERE.parent  # .../iiwa_description
XACRO_IN = IIWA_DESC / "urdf" / "iiwa7.urdf.xacro"
OUT_DIR = HERE / "iiwa7"
OUT_URDF = OUT_DIR / "iiwa7.urdf"
OUT_MJCF = OUT_DIR / "iiwa7.xml"
MESH_DIR = IIWA_DESC / "meshes" / "iiwa7"


def monkey_patch_xacro_find(pkg_map: dict[str, str]) -> None:
    def _find_override(resolved: str, a: str, args: list[str], context):
        if len(args) != 1:
            raise subs.SubstitutionException(
                f"$(find pkg) takes exactly one arg, got {args!r}")
        pkg = args[0]
        if pkg not in pkg_map:
            raise subs.SubstitutionException(
                f"package {pkg!r} not in pkg_map")
        return resolved.replace(f"$({a})", pkg_map[pkg])

    original_resolve_args = subs._resolve_args

    def patched_resolve_args(arg_str, context, commands):
        commands = dict(commands)
        commands["find"] = _find_override
        return original_resolve_args(arg_str, context, commands)

    subs._resolve_args = patched_resolve_args


def expand_xacro_to_urdf(xacro_path: Path, out_path: Path) -> str:
    monkey_patch_xacro_find({"iiwa_description": str(IIWA_DESC)})
    doc = xacro.process_file(str(xacro_path))
    xml_text = doc.toprettyxml(indent="  ")
    out_path.write_text(xml_text)
    return xml_text


_GAZEBO_RE = re.compile(r"<gazebo[^>]*>.*?</gazebo>", re.DOTALL)
_TRANSMISSION_RE = re.compile(r"<transmission[^>]*>.*?</transmission>", re.DOTALL)


def strip_ros_specific(urdf_text: str) -> str:
    urdf_text = _GAZEBO_RE.sub("", urdf_text)
    urdf_text = _TRANSMISSION_RE.sub("", urdf_text)
    return urdf_text


def rewrite_mesh_paths_and_inject_mujoco_block(urdf_text: str, mjcf_dir: Path) -> str:
    rel_meshdir = os.path.relpath(MESH_DIR, mjcf_dir)
    urdf_text = urdf_text.replace(
        "package://iiwa_description/meshes/iiwa7/",
        "",
    )
    mujoco_block = (
        f'  <mujoco>\n'
        f'    <compiler meshdir="{rel_meshdir}" '
        f'balanceinertia="true" discardvisual="false" strippath="false"/>\n'
        f'  </mujoco>\n'
    )
    urdf_text = urdf_text.replace(
        '<robot name="iiwa7">',
        '<robot name="iiwa7">\n' + mujoco_block,
        1,
    )
    return urdf_text


def collapse_whitespace(urdf_text: str) -> str:
    lines = [line for line in urdf_text.splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


def compile_mjcf(urdf_path: Path, mjcf_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(mjcf_path), model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(
        f"[OK] compiled MJCF:\n"
        f"     nbody={model.nbody}  njnt={model.njnt}  ngeom={model.ngeom}  "
        f"nmesh={model.nmesh}  nq={model.nq}  nu={model.nu}"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] xacro -> URDF: {XACRO_IN.name}")
    raw = expand_xacro_to_urdf(XACRO_IN, OUT_DIR / "_raw_debug.urdf")

    print("[2/4] strip <gazebo> / <transmission>")
    clean = strip_ros_specific(raw)

    print(f"[3/4] rewrite mesh paths + inject <mujoco> compiler block")
    clean = rewrite_mesh_paths_and_inject_mujoco_block(clean, OUT_DIR)
    clean = collapse_whitespace(clean)
    OUT_URDF.write_text(clean)
    print(f"     wrote {OUT_URDF}")

    print(f"[4/4] mujoco compile -> {OUT_MJCF.name}")
    compile_mjcf(OUT_URDF, OUT_MJCF)
    print(f"     wrote {OUT_MJCF}")

    debug_file = OUT_DIR / "_raw_debug.urdf"
    if debug_file.exists():
        debug_file.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
