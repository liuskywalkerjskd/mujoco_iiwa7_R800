"""End-effector pose controller for iiwa7 in MuJoCo.

Plug-and-play companion to `iiwa7_mjcf/` (pure MJCF). Use this package
when you want a high-level "send EE pose, get correct control signals"
API without wiring IK + feedforward yourself.

Usage:
    from iiwa7_controller import IiwaEEController
"""
from .controller import IiwaEEController

__all__ = ["IiwaEEController"]
