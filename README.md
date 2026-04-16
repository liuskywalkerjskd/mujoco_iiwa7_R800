# iiwa7-mujoco

MuJoCo (MJCF) model for the KUKA LBR iiwa 7 R800, converted from the
ROS `iiwa_stack` URDF and tuned to `mujoco_menagerie/kuka_iiwa_14`
engineering conventions. Ships with a suite of motion-control demos
(gravity / inverse-dynamics feedforward + task-space PD) and a
unified end-effector controller class.

## Installation

```bash
git clone <this repo>
cd iiwa7-mujoco
pip install mujoco scipy imageio imageio-ffmpeg
```

## Quick start

Interactive viewer (machine with a display):

```bash
python3 -m mujoco.viewer --mjcf=examples/scenes/iiwa7_scene.xml
```

Headless render of the flagship demo (display-less server, EGL):

```bash
MUJOCO_GL=egl python3 examples/demo_current_state_ff.py
# -> media/videos/demo_square_current_state_ff.mp4
```

Minimal Python API:

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("iiwa7_mjcf/iiwa7_tuned.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
d.ctrl[:] = np.array([0, 0.5, 0, -1.2, 0, 0.8, 0])
for _ in range(2000):
    mujoco.mj_step(m, d)
```

## Repository layout

```
iiwa7-mujoco/
├── iiwa7_mjcf/                 plug-and-play MJCF model
│   ├── iiwa7.xml                 base MJCF (auto-converted from URDF)
│   ├── iiwa7_tuned.xml           menagerie-style tuned model
│   ├── iiwa7.urdf                cleaned URDF (intermediate)
│   └── meshes/                   collision STLs + split visual OBJs
├── iiwa7_controller/           unified EE / joint controller
├── examples/
│   ├── demo_*.py                 10 demos (see below)
│   └── scenes/                   scene wrappers
├── tools/
│   └── convert_iiwa7_to_mjcf.py  URDF -> MJCF pipeline
├── media/                      thumbnails + recorded mp4s
└── TUNING_REPORT.md            parameter diff + tracking benchmarks
```

## Home-pose convention

The `home` keyframe matches **real-robot habits**, not the URDF default
vertical pose. On the physical iiwa we treat **J4 = +90°, J6 = -90°**
(other joints at 0°) as home — arm bent forward, tool flange pointing
roughly downward.

To make the sim open in that pose out-of-the-box, `iiwa7_mjcf/iiwa7.xml`
sets `ref=1.5708` on J4 and `ref=-1.5708` on J6. Per MuJoCo semantics
the physical joint angle is `qpos - ref`, so:

| qpos (J4, J6) | physical angle | visual pose          |
|---------------|----------------|----------------------|
| `(0, 0)`      | `(-90°, +90°)` | real-robot home      |
| `(+90°, -90°)`| `(0°, 0°)`     | URDF fully-vertical  |

The keyframes are therefore all zeros:

```xml
<key name="home" qpos="0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
```

**Trade-off.** `data.qpos[3]` and `data.qpos[5]` differ from the real
robot's J4 / J6 encoder readings by ±1.5708 rad. Add / subtract `ref`
at the sim↔real boundary.

**Revert to vanilla URDF semantics** (`qpos` == encoder readings):

1. Delete `ref="1.5708"` and `ref="-1.5708"` from J4 / J6 in
   `iiwa7_mjcf/iiwa7.xml`.
2. Restore the keyframe in `iiwa7_mjcf/iiwa7_tuned.xml`,
   `examples/scenes/iiwa7_scene.xml`, and
   `examples/scenes/iiwa7_square_scene.xml`:

   ```xml
   <key name="home" qpos="0 0 0 1.5708 0 -1.5708 0" ctrl="0 0 0 1.5708 0 -1.5708 0"/>
   ```

After these edits, `qpos` is 1:1 with real-robot encoders; the cost is
that opening the viewer without a keyframe reset shows the vertical
URDF pose.

## Tuning highlights

`iiwa7_tuned.xml` vs the auto-converted `iiwa7.xml`:

| Parameter           | Base `iiwa7.xml`      | Tuned `iiwa7_tuned.xml`                      | Source            |
|---------------------|-----------------------|----------------------------------------------|-------------------|
| Actuator type       | `<position>` pure P   | `<general>` gaintype=fixed, biastype=affine  | menagerie         |
| kp / kd             | 400/200/100, no kd    | kp=2000 uniform, kd=200 via `biasprm`        | menagerie         |
| forcerange          | none                  | J1/2: ±176, J3-5: ±110, J6/7: ±40 N·m        | iiwa7 datasheet   |
| armature            | none                  | 0.1                                          | motor rotor       |
| contact exclude     | none                  | 7 pairs                                      | menagerie         |
| attachment site     | none                  | on link7 at (0, 0, 0.05)                     | tool mount        |
| Joint damping       | 0.5 (from URDF)       | 0.5 (kept)                                   | passive stability |
| Link masses/inertia | iiwa7 URDF values     | unchanged                                    | iiwa7 ≠ iiwa14    |

Full diff in [`TUNING_REPORT.md`](TUNING_REPORT.md).

## Control stack

The best controller (`demo_current_state_ff.py`) combines
inverse-dynamics feedforward evaluated at the current state with a
task-space PD folded into commanded acceleration:

1. Precompute IK joint targets at 30 Hz, fit per-joint cubic splines
   for a C² reference `(q_d, q̇_d, q̈_d)`.
2. At every sim step (500 Hz), evaluate the spline.
3. Form a commanded acceleration
   `q̈_cmd = q̈_d + Kp·(q_d − q) + Kd·(q̇_d − q̇)` (`Kp=400, Kd=40`).
4. Call `mj_inverse` at the **current** state `(q, q̇, q̈_cmd)` to obtain
   τ_ff; write it to `data.qfrc_applied`.
5. Actuator PD with `ctrl = q_d` closes the residual error.

## Tracking benchmarks

30 × 30 cm horizontal square, 2 loops, actuator-in-loop simulation:

| Controller                                       | mean   | p95    | max    | vs legacy |
|--------------------------------------------------|--------|--------|--------|-----------|
| legacy (position, kp=400/200/100)                | 94.81  | 156.49 | 169.55 | 1.0×      |
| tuned (general, kp=2000 kd=200)                  | 22.28  | 30.37  | 31.90  | 4.3×      |
| tuned + gravity FF                               | 10.75  | 15.56  | 22.18  | 8.8×      |
| tuned + full ID FF (reference state)             | 8.72   | 12.55  | 17.91  | 10.9×     |
| **tuned + current-state ID FF + task-space PD**  | **5.48** | **9.63**  | **12.28** | **17.3×** |

All values in millimetres. See [`TUNING_REPORT.md`](TUNING_REPORT.md)
for the raw logs.

## Demo

Flagship: horizontal 30 × 30 cm square traced twice under the
current-state ID FF + task-space PD controller (mean **5.48 mm**,
max **12.28 mm**). Rendered headlessly via `MUJOCO_GL=egl`, H.264 MP4,
720×540 @ 30 fps.

[![current-state ID FF demo](media/thumbnails/demo_square_current_state_ff.jpg)](media/videos/demo_square_current_state_ff.mp4)

The remaining demo scripts (`demo_motions_v2.py`,
`demo_ee_orientation_cycle.py`, etc.) cover additional trajectories
— figure-8, spiral, obstacle arc, stacking, 6-DOF orientation-locked
square — and write their own MP4s to `media/videos/`.

## Credits

- URDF source: [IFL-CAMP/iiwa_stack](https://github.com/IFL-CAMP/iiwa_stack)
- Tuning reference: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (`kuka_iiwa_14`)
