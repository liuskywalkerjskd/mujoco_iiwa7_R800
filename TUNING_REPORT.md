# iiwa7 MJCF Tuning Report

Cross-referenced against [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) iiwa14.xml. Same reference path (30 cm square at Z=0.55 m, 2 loops) driven through actuators in closed-loop sim on both models.

## Actuator-in-loop EE tracking error

| model                            | mean (mm) | p95 (mm) | max (mm) |
|---                                |---        |---       |---       |
| legacy (position, kp=400/200/100) |   94.81 | 156.49 | 169.55 |
| **tuned (general, kp=2000 kd=200)**| **22.28** | **30.37** | **31.90** |

improvement: mean 4.3x, max 5.3x

## Parameter diff adopted from menagerie

| parameter | legacy iiwa7.xml | tuned iiwa7_tuned.xml | menagerie iiwa14 | rationale |
|---|---|---|---|---|
| actuator type | `<position>` | `<general>` gaintype=fixed biastype=affine | `<general>` same | full PD with kd term |
| gainprm / kp | 400/200/100 (tiered) | 2000 uniform | 2000 uniform | stiffer gravity rejection |
| biasprm kd | implicit 0 | -200 | -200 | velocity damping via actuator |
| forcerange | none | J1/2:±176, J3-5:±110, J6/7:±40 Nm | J1-5:±various, J6/7:±40 | iiwa7 datasheet torque limits |
| joint damping | 0.5 (URDF) | 0.5 (kept) | 0 | let passive sim still settle |
| armature | none | 0.1 | none | reflect motor rotor inertia, stabilise at high kp |
| contact exclude | none | 7 pairs | 7 pairs | avoid false self-contact |
| attachment_site | none | on link7 at (0,0,0.05) | same | standard tool mount point |
| default classes | none | iiwa/joint1..7 classes | same | dedupe joint/actuator config |
| inertial values | URDF iiwa7 (masses 3.45/3.48/4.06...) | URDF iiwa7 (unchanged) | menagerie iiwa14 (5.76/6.35/3.5...) | iiwa7 ≠ iiwa14 — do not copy |

### Update (full ID FF added)

| **tuned + full ID FF (mj_inverse)** | **8.72** | **12.55** | **17.91** |
| **tuned + current-state ID FF + task-PD** | **5.48** | **9.63** | **12.28** |

