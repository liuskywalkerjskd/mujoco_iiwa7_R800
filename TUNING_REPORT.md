# iiwa7 MJCF Tuning Report

Cross-referenced against [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) iiwa14.xml. Same reference path (30 cm square at Z=0.55 m, 2 loops) driven through actuators in closed-loop sim on both models.

## Actuator-in-loop EE tracking error

| model                            | mean (mm) | p95 (mm) | max (mm) |
|---                                |---        |---       |---       |
| legacy (position, kp=400/200/100) |   94.81 | 156.49 | 169.55 |
| **tuned (general, kp=2000 kd=200)**| **22.28** | **30.37** | **31.90** |
| **tuned + gravity FF (PD + qfrc_applied)** | **10.75** | **15.56** | **22.18** |
| **tuned + full ID FF (mj_inverse)** | **8.72** | **12.55** | **17.91** |
| **tuned + current-state ID FF + task-PD** | **5.48** | **9.63** | **12.28** |

improvement vs legacy:
- tuned PD only:                               mean 4.3x, max 5.3x
- tuned + gravity FF:                           mean 8.8x, max 7.6x
- tuned + full ID FF (reference-state):         mean 10.9x, max 9.5x
- **tuned + current-state ID FF + task-PD:      mean 17.3x, max 13.8x**

incremental improvement:
- gravity FF vs tuned PD:              mean 2.1x, max 1.4x   (static gravity compensated)
- full ID FF vs gravity FF:            mean 1.2x, max 1.2x   (Coriolis + inertial added, reference state)
- **current-state vs reference-state ID FF: mean 1.6x, max 1.5x** (FF now evaluated at where the arm actually is)

Why full ID FF didn't drop to <5 mm:
1. **Reference-state FF, not current-state FF.** `mj_inverse` is evaluated at
   `(q_d, q̇_d, q̈_d)` — the planned state, not `data.qpos`. When the arm
   lags (few mm), the FF is slightly wrong at the actual configuration;
   PD must close the residual.
2. **Discrete spline at segment transitions.** Cubic-spline q̈_d is
   continuous but has rapid variation at the corners of the square path
   (smoothstep edge transitions), causing brief FF spikes the PD must track.
3. **Integrator artefacts at kp=2000, dt=2 ms.** Even with `implicitfast`,
   a stiff PD + high-frequency q̈_d content creates bounded oscillation
   <1 rad/s which integrates into a few mm of position error.

Further reduction below 5 mm requires one of:
- current-state FF with PD-in-the-loop formulation (re-solve ID with
  observed `data.qpos`/`data.qvel` instead of reference)
- increase sim timestep quality (`euler` with dt=0.5 ms or `implicit`)
- slow the reference (T_EDGE 2.5 s -> 5 s halves centripetal q̈_d)

FF peak torque per joint (N·m):
- gravity FF only:   J1=0.0  J2=59.8  J3=3.5  J4=32.3  J5=2.1  J6=3.1  J7=0.0
- **full ID FF:      J1=6.6  J2=74.6  J3=5.7  J4=35.8  J5=2.4  J6=2.8  J7=0.0**

The extra inertial/Coriolis load shows up on the proximal joints as expected
(J1 rotates the whole arm, J2 bears the rotated arm's angular momentum).

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
