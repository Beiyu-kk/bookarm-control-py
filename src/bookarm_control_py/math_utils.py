"""Small math helpers shared by BookArm scripts and examples."""

from __future__ import annotations

import numpy as np


def rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw radians to a rotation matrix.

    The convention is ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    roll, pitch, yaw = np.asarray(rpy_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rotation_z @ rotation_y @ rotation_x


__all__ = ["rpy_to_matrix"]
