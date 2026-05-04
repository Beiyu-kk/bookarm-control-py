"""可复用的 Intel RealSense D435 RGB-D 相机工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import pyrealsense2 as rs

ColorOrder = Literal["rgb", "bgr"]
DepthUnit = Literal["m", "raw"]


@dataclass(frozen=True)
class CameraIntrinsics:
    """图像流的针孔相机内参。"""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class PointCloud:
    """由深度图生成的点云。

    点云使用 RealSense 相机坐标系：
    x 轴向右，y 轴向下，z 轴从相机向前。
    """

    points_xyz_m: np.ndarray
    colors_rgb: Optional[np.ndarray]
    pixels_uv: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ms: float
    frame_number: int

    @property
    def size(self) -> int:
        return int(self.points_xyz_m.shape[0])

    def save_npz(self, path: str | Path) -> Path:
        """把点、颜色、像素坐标和元数据保存为压缩 NPZ 文件。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object] = {
            "points_xyz_m": self.points_xyz_m,
            "pixels_uv": self.pixels_uv,
            "timestamp_ms": np.array(self.timestamp_ms, dtype=np.float64),
            "frame_number": np.array(self.frame_number, dtype=np.int64),
            "intrinsics": np.array(
                [
                    self.intrinsics.width,
                    self.intrinsics.height,
                    self.intrinsics.fx,
                    self.intrinsics.fy,
                    self.intrinsics.ppx,
                    self.intrinsics.ppy,
                ],
                dtype=np.float64,
            ),
        }
        if self.colors_rgb is not None:
            payload["colors_rgb"] = self.colors_rgb

        np.savez_compressed(output_path, **payload)
        return output_path

    def save_ply(self, path: str | Path) -> Path:
        """把点云保存为二进制小端 PLY 文件。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        has_color = self.colors_rgb is not None
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {self.size}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_color:
            header_lines.extend(
                [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            )
        header_lines.append("end_header")
        header = "\n".join(header_lines) + "\n"

        if has_color:
            data = np.empty(
                self.size,
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            data["red"] = self.colors_rgb[:, 0]
            data["green"] = self.colors_rgb[:, 1]
            data["blue"] = self.colors_rgb[:, 2]
        else:
            data = np.empty(
                self.size,
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")],
            )

        data["x"] = self.points_xyz_m[:, 0]
        data["y"] = self.points_xyz_m[:, 1]
        data["z"] = self.points_xyz_m[:, 2]

        with output_path.open("wb") as file:
            file.write(header.encode("ascii"))
            data.tofile(file)

        return output_path

    def to_open3d(
        self,
        flip_y_for_viewer: bool = False,
        flip_z_for_viewer: bool = False,
    ):
        """转换为 Open3D 点云对象。

        Open3D 默认显示习惯接近 x 向右、y 向上、z 向屏幕外。RealSense
        相机坐标系是 x 向右、y 向下、z 向前。显示时可翻转 y/z 轴来让点云
        画面和 RGB 图像方向一致；原始点云数据和保存文件仍保持相机坐标系。
        """

        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "缺少 open3d，无法进行 3D 点云可视化。请先运行：pip install open3d"
            ) from exc

        points = self.points_xyz_m.astype(np.float64, copy=True)
        if flip_y_for_viewer:
            points[:, 1] *= -1.0
        if flip_z_for_viewer:
            points[:, 2] *= -1.0

        open3d_cloud = o3d.geometry.PointCloud()
        open3d_cloud.points = o3d.utility.Vector3dVector(points)

        if self.colors_rgb is not None:
            colors = self.colors_rgb.astype(np.float64, copy=False) / 255.0
            open3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        return open3d_cloud


@dataclass(frozen=True)
class RGBDFrame:
    """RealSense 相机采集到的一帧同步 RGB-D 数据。"""

    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_m: np.ndarray
    depth_scale: float
    timestamp_ms: float
    frame_number: int
    intrinsics: CameraIntrinsics
    depth_aligned_to_color: bool = True

    @property
    def color_rgb(self) -> np.ndarray:
        """返回 RGB 通道顺序的彩色图。"""

        return self.color_bgr[:, :, ::-1].copy()

    def color(self, order: ColorOrder = "rgb") -> np.ndarray:
        """按 RGB 或 BGR 通道顺序返回彩色图。"""

        if order == "rgb":
            return self.color_rgb
        if order == "bgr":
            return self.color_bgr.copy()
        raise ValueError(f"不支持的彩色图通道顺序：{order}")

    def depth(self, unit: DepthUnit = "m") -> np.ndarray:
        """返回米制深度图或传感器原始单位深度图。"""

        if unit == "m":
            return self.depth_m.copy()
        if unit == "raw":
            return self.depth_raw.copy()
        raise ValueError(f"不支持的深度单位：{unit}")

    def valid_depth_mask(self, max_depth_m: Optional[float] = None) -> np.ndarray:
        """返回有效正深度像素的掩码。"""

        mask = np.isfinite(self.depth_m) & (self.depth_m > 0.0)
        if max_depth_m is not None:
            mask &= self.depth_m <= max_depth_m
        return mask

    def point_at_pixel(
        self,
        u: int,
        v: int,
        *,
        search_radius: int = 5,
        max_depth_m: Optional[float] = None,
    ) -> np.ndarray:
        """返回指定像素对应的相机坐标系 3D 点，单位为米。

        如果点击像素没有有效深度，会在 ``search_radius`` 半径内寻找最近的
        有效深度像素。返回坐标遵循 RealSense 相机坐标系：x 向右，y 向下，
        z 从相机向前。
        """

        height, width = self.depth_m.shape
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(f"像素坐标超出图像范围：u={u}, v={v}, size={width}x{height}")

        valid_mask = self.valid_depth_mask(max_depth_m=max_depth_m)
        if valid_mask[v, u]:
            depth = float(self.depth_m[v, u])
            return self._deproject_pixel(u, v, depth)

        if search_radius < 0:
            raise ValueError("search_radius 必须大于等于 0。")
        if search_radius == 0:
            raise ValueError(f"点击像素没有有效深度：u={u}, v={v}")

        u_min = max(0, u - search_radius)
        u_max = min(width, u + search_radius + 1)
        v_min = max(0, v - search_radius)
        v_max = min(height, v + search_radius + 1)
        local_mask = valid_mask[v_min:v_max, u_min:u_max]
        if not np.any(local_mask):
            raise ValueError(
                f"点击像素附近没有有效深度：u={u}, v={v}, search_radius={search_radius}"
            )

        local_v, local_u = np.nonzero(local_mask)
        candidate_u = local_u + u_min
        candidate_v = local_v + v_min
        distances = (candidate_u - u) ** 2 + (candidate_v - v) ** 2
        best_index = int(np.argmin(distances))
        best_u = int(candidate_u[best_index])
        best_v = int(candidate_v[best_index])
        depth = float(self.depth_m[best_v, best_u])
        return self._deproject_pixel(best_u, best_v, depth)

    def _deproject_pixel(self, u: int, v: int, depth_m: float) -> np.ndarray:
        x = (float(u) - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
        y = (float(v) - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
        return np.array([x, y, depth_m], dtype=float)

    def depth_colormap_bgr(
        self,
        max_depth_m: float = 2.0,
        invalid_color_bgr: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """把深度图转换为 OpenCV 可直接显示的 BGR 伪彩色图。"""

        if max_depth_m <= 0:
            raise ValueError("max_depth_m 必须为正数。")

        import cv2

        depth_normalized = np.clip(self.depth_m / max_depth_m, 0.0, 1.0)
        depth_u8 = (depth_normalized * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        depth_color[~self.valid_depth_mask()] = invalid_color_bgr
        return depth_color

    def to_point_cloud(
        self,
        max_depth_m: Optional[float] = None,
        stride: int = 1,
        include_color: bool = True,
    ) -> PointCloud:
        """把深度图反投影为点云。

        如果采集时已经把深度对齐到彩色图，点云颜色会从匹配的 RGB 像素采样。
        增大 ``stride`` 可以生成更轻量的点云，便于快速预览。
        """

        if stride < 1:
            raise ValueError("stride 必须大于等于 1。")

        depth = self.depth_m[::stride, ::stride]
        height, width = depth.shape
        v_grid, u_grid = np.indices((height, width), dtype=np.float32)
        u_grid *= stride
        v_grid *= stride

        valid = np.isfinite(depth) & (depth > 0.0)
        if max_depth_m is not None:
            valid &= depth <= max_depth_m

        z = depth[valid].astype(np.float32, copy=False)
        u = u_grid[valid]
        v = v_grid[valid]
        x = (u - self.intrinsics.ppx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.ppy) * z / self.intrinsics.fy
        points_xyz_m = np.column_stack((x, y, z)).astype(np.float32, copy=False)
        pixels_uv = np.column_stack((u, v)).astype(np.int32, copy=False)

        colors_rgb: Optional[np.ndarray] = None
        if (
            include_color
            and self.depth_aligned_to_color
            and self.color_bgr.shape[:2] == self.depth_m.shape
        ):
            color_rgb = self.color_rgb[::stride, ::stride]
            colors_rgb = color_rgb[valid].reshape(-1, 3).astype(np.uint8, copy=True)

        return PointCloud(
            points_xyz_m=points_xyz_m,
            colors_rgb=colors_rgb,
            pixels_uv=pixels_uv,
            intrinsics=self.intrinsics,
            timestamp_ms=self.timestamp_ms,
            frame_number=self.frame_number,
        )

    def save_images(
        self,
        directory: str | Path,
        prefix: str = "d435",
        depth_vis_max_m: float = 2.0,
    ) -> dict[str, Path]:
        """保存彩色图、原始深度图和深度伪彩色图。"""

        import cv2

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "color": output_dir / f"{prefix}_color.png",
            "depth_raw": output_dir / f"{prefix}_depth_raw.png",
            "depth_vis": output_dir / f"{prefix}_depth_vis.png",
        }
        cv2.imwrite(str(paths["color"]), self.color_bgr)
        cv2.imwrite(str(paths["depth_raw"]), self.depth_raw)
        cv2.imwrite(
            str(paths["depth_vis"]),
            self.depth_colormap_bgr(max_depth_m=depth_vis_max_m),
        )
        return paths


class RealSenseD435:
    """控制 Intel RealSense D435，并读取 RGB、深度图和点云。

    默认会把深度图对齐到彩色图。这样 ``RGBDFrame`` 更适合做图像处理和点云上色，
    因为每个深度像素都和 RGB 图像共享同一个像素坐标。
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        align_depth_to_color: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.align_depth_to_color = align_depth_to_color

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if serial:
            self._config.enable_device(serial)

        self._config.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps,
        )
        self._config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )

        self._align = rs.align(rs.stream.color) if align_depth_to_color else None
        self._profile: Optional[rs.pipeline_profile] = None
        self._depth_scale: Optional[float] = None

    @property
    def is_started(self) -> bool:
        return self._profile is not None

    @property
    def depth_scale(self) -> float:
        if self._depth_scale is None:
            raise RuntimeError("相机尚未启动，无法读取 depth scale。")
        return self._depth_scale

    def start(self) -> None:
        """启动彩色流和深度流。"""

        if self._profile is not None:
            return

        self._profile = self._pipeline.start(self._config)
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

    def stop(self) -> None:
        """停止 RealSense 数据管线。"""

        if self._profile is None:
            return

        self._pipeline.stop()
        self._profile = None
        self._depth_scale = None

    def warmup(self, frame_count: int = 30, timeout_ms: int = 5000) -> None:
        """丢弃启动后的前几帧，等待自动曝光稳定。"""

        if frame_count <= 0:
            return
        self._ensure_started()
        for _ in range(frame_count):
            self._pipeline.wait_for_frames(timeout_ms)

    def get_frame(self, timeout_ms: int = 5000) -> RGBDFrame:
        """读取一帧同步 RGB-D 数据。"""

        self._ensure_started()
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("未能同时获取 RGB 和深度帧，请检查 D435 连接状态。")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        depth_scale = self.depth_scale
        depth_m = depth_raw.astype(np.float32) * depth_scale
        intrinsics = self._get_frame_intrinsics(depth_frame, color_frame)

        return RGBDFrame(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_m=depth_m,
            depth_scale=depth_scale,
            timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
            intrinsics=intrinsics,
            depth_aligned_to_color=self.align_depth_to_color,
        )

    def iter_frames(self, timeout_ms: int = 5000) -> Iterator[RGBDFrame]:
        """持续生成 RGB-D 帧，直到调用方停止迭代。"""

        self._ensure_started()
        while True:
            yield self.get_frame(timeout_ms=timeout_ms)

    def get_rgbd_frame(self, timeout_ms: int = 5000) -> RGBDFrame:
        """``get_frame`` 的别名，方便调用处显式表达 RGB-D 语义。"""

        return self.get_frame(timeout_ms=timeout_ms)

    def get_color_image(
        self,
        timeout_ms: int = 5000,
        order: ColorOrder = "rgb",
    ) -> np.ndarray:
        """采集一帧并只返回彩色图。"""

        return self.get_frame(timeout_ms=timeout_ms).color(order=order)

    def get_depth_image(
        self,
        timeout_ms: int = 5000,
        unit: DepthUnit = "m",
    ) -> np.ndarray:
        """采集一帧并只返回深度图。"""

        return self.get_frame(timeout_ms=timeout_ms).depth(unit=unit)

    def get_point_cloud(
        self,
        timeout_ms: int = 5000,
        max_depth_m: Optional[float] = None,
        stride: int = 1,
        include_color: bool = True,
    ) -> PointCloud:
        """采集一帧 RGB-D 数据并返回对应点云。"""

        frame = self.get_frame(timeout_ms=timeout_ms)
        return frame.to_point_cloud(
            max_depth_m=max_depth_m,
            stride=stride,
            include_color=include_color,
        )

    def _ensure_started(self) -> None:
        if self._profile is None:
            self.start()

    def _get_frame_intrinsics(self, depth_frame, color_frame) -> CameraIntrinsics:
        if self.align_depth_to_color:
            video_profile = color_frame.profile.as_video_stream_profile()
        else:
            video_profile = depth_frame.profile.as_video_stream_profile()

        intrinsics = video_profile.intrinsics
        return CameraIntrinsics(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
        )

    def __enter__(self) -> "RealSenseD435":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
