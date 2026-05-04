"""相机点和机械臂坐标系之间的几何变换工具。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from bookarm_control_py.math_utils import rpy_to_matrix


@dataclass(frozen=True)
class RigidTransform:
    """三维刚体变换，表示 target 点 = R @ source 点 + t。"""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=float).reshape(3, 3))
        object.__setattr__(self, "translation", np.asarray(self.translation, dtype=float).reshape(3))

    @classmethod
    def from_xyz_rpy_deg(
        cls,
        xyz_m: list[float] | tuple[float, float, float] | np.ndarray,
        rpy_deg: list[float] | tuple[float, float, float] | np.ndarray,
    ) -> "RigidTransform":
        """用平移 xyz 米和欧拉角 rpy 度创建刚体变换。"""

        return cls(
            rotation=rpy_to_matrix(np.deg2rad(np.asarray(rpy_deg, dtype=float))),
            translation=np.asarray(xyz_m, dtype=float),
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "RigidTransform":
        """从 4x4 齐次变换矩阵创建刚体变换。"""

        transform = np.asarray(matrix, dtype=float).reshape(4, 4)
        return cls(rotation=transform[:3, :3], translation=transform[:3, 3])

    @classmethod
    def from_json(cls, path: str | Path) -> "RigidTransform":
        """从 JSON 标定文件加载刚体变换。

        支持两种格式：

        1. ``{"matrix": [[...], [...], [...], [...]]}``
        2. ``{"xyz_m": [x, y, z], "rpy_deg": [roll, pitch, yaw]}``
        """

        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if "matrix" in payload:
            return cls.from_matrix(np.asarray(payload["matrix"], dtype=float))
        if "xyz_m" in payload and "rpy_deg" in payload:
            return cls.from_xyz_rpy_deg(payload["xyz_m"], payload["rpy_deg"])
        raise ValueError(
            "标定文件必须包含 matrix，或同时包含 xyz_m 和 rpy_deg。"
        )

    @property
    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return transform

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """把一个 3D 点从 source 坐标系变换到 target 坐标系。"""

        return self.rotation @ np.asarray(point, dtype=float).reshape(3) + self.translation
