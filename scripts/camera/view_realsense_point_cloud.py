"""RealSense D435 点云实时可视化脚本。

请在项目根目录运行：

    conda run -n bookarm-beiyu python scripts/camera/view_realsense_point_cloud.py

默认会把点云显示坐标转换到和 RealSense RGB 图像一致：画面向右就是 RGB
图像向右，画面向上就是 RGB 图像向上，深度沿画面往里延伸。窗口中按 s
保存当前点云，按 r 回到相机视角，按 q 退出。关闭窗口也会结束程序。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from bookarm_control_py.camera import PointCloud, RealSenseD435


DEFAULT_SAVE_DIR = Path("recordings/realsense")


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "缺少 open3d，无法打开 3D 点云窗口。请先运行：pip install open3d"
        ) from exc
    return o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时显示 RealSense D435 彩色点云。")
    parser.add_argument("--serial", default=None, help="指定 D435 序列号；只有一台相机时可不填。")
    parser.add_argument("--width", type=int, default=640, help="彩色和深度流宽度。")
    parser.add_argument("--height", type=int, default=480, help="彩色和深度流高度。")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率。")
    parser.add_argument("--warmup", type=int, default=30, help="丢弃前 N 帧等待自动曝光稳定。")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="等待帧的超时时间。")
    parser.add_argument("--max-depth", type=float, default=1.5, help="点云保留的最大深度，单位米。")
    parser.add_argument("--stride", type=int, default=2, help="点云采样步长；1 表示保留所有有效深度点。")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="点云保存目录。")
    parser.add_argument("--prefix", default="d435_cloud", help="保存文件名前缀。")
    parser.add_argument("--no-color", action="store_true", help="不使用 RGB 给点云上色。")
    parser.add_argument("--raw-camera-coordinates", action="store_true", help="显示时完全使用 RealSense 原始相机坐标系，不做 RGB 方向对齐。")
    parser.add_argument("--mirror-x", action="store_true", help="显示时额外左右镜像；用于排查特殊视角或显示器方向问题。")
    parser.add_argument("--axis-size", type=float, default=0.1, help="坐标轴显示尺寸；设为 0 可隐藏。")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D 窗口中的点大小。")
    parser.add_argument("--free-view", action="store_true", help="启动时使用 Open3D 默认自由视角，不强制相机正视角。")
    parser.add_argument("--view-depth", type=float, default=None, help="相机正视角看向的深度位置，默认使用 max-depth 的一半。")
    parser.add_argument("--view-zoom", type=float, default=0.55, help="相机正视角的缩放比例。")
    parser.add_argument("--save-ply", action="store_true", help="按 s 时额外保存 PLY 点云。")
    return parser.parse_args()


def copy_cloud_to_geometry(source, target) -> None:
    target.points = source.points
    target.colors = source.colors


def save_point_cloud(point_cloud: PointCloud, args: argparse.Namespace) -> None:
    prefix = f"{args.prefix}_{point_cloud.frame_number}"
    npz_path = point_cloud.save_npz(args.save_dir / f"{prefix}.npz")
    print(f"已保存 NPZ 点云：{npz_path}")

    if args.save_ply:
        ply_path = point_cloud.save_ply(args.save_dir / f"{prefix}.ply")
        print(f"已保存 PLY 点云：{ply_path}")


def set_camera_forward_view(visualizer, args: argparse.Namespace) -> None:
    """把 Open3D 视角设置为和 RealSense RGB 图像一致的方向。"""

    lookat_depth = args.view_depth
    if lookat_depth is None:
        lookat_depth = args.max_depth * 0.5
    lookat_z = float(lookat_depth)
    front_z = 1.0
    up_y = -1.0
    if not args.raw_camera_coordinates:
        lookat_z *= -1.0
        front_z = -1.0
        up_y = 1.0

    view_control = visualizer.get_view_control()
    view_control.set_lookat([0.0, 0.0, lookat_z])

    # RGB 方向对齐模式中，点云显示坐标为 x 向右、y 向上、z 向屏幕外，
    # 因此真实深度 +Z 会显示为 -Z，视线沿 -Z 看进去。
    view_control.set_front([0.0, 0.0, front_z])
    view_control.set_up([0.0, up_y, 0.0])
    view_control.set_zoom(args.view_zoom)


def make_display_cloud(point_cloud: PointCloud, args: argparse.Namespace):
    display_cloud = point_cloud.to_open3d(
        flip_y_for_viewer=not args.raw_camera_coordinates,
        flip_z_for_viewer=not args.raw_camera_coordinates,
    )
    if args.mirror_x:
        points = np.asarray(display_cloud.points)
        points[:, 0] *= -1.0
        display_cloud.points = type(display_cloud.points)(points)
    return display_cloud


def main() -> None:
    args = parse_args()
    o3d = import_open3d()

    state = {
        "running": True,
        "save_requested": False,
        "reset_view_requested": False,
    }

    def request_save(_visualizer) -> bool:
        state["save_requested"] = True
        return False

    def request_quit(_visualizer) -> bool:
        state["running"] = False
        return False

    def request_reset_view(_visualizer) -> bool:
        state["reset_view_requested"] = True
        return False

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(
        window_name="RealSense D435 点云",
        width=1280,
        height=720,
    )
    visualizer.register_key_callback(ord("S"), request_save)
    visualizer.register_key_callback(ord("Q"), request_quit)
    visualizer.register_key_callback(ord("R"), request_reset_view)

    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = np.asarray([0.02, 0.02, 0.02])

    display_cloud = o3d.geometry.PointCloud()
    has_display_cloud = False

    if args.axis_size > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size)
        visualizer.add_geometry(axis)

    print("点云可视化已启动：默认和 RGB 画面方向一致；按 r 重置视角，按 s 保存，按 q 退出。")

    previous_time = time.perf_counter()
    frame_counter = 0

    try:
        with RealSenseD435(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.serial,
            align_depth_to_color=True,
        ) as camera:
            camera.warmup(frame_count=args.warmup, timeout_ms=args.timeout_ms)

            for frame in camera.iter_frames(timeout_ms=args.timeout_ms):
                if not state["running"]:
                    break

                point_cloud = frame.to_point_cloud(
                    max_depth_m=args.max_depth,
                    stride=args.stride,
                    include_color=not args.no_color,
                )

                open3d_cloud = make_display_cloud(point_cloud, args)
                copy_cloud_to_geometry(open3d_cloud, display_cloud)
                if not has_display_cloud:
                    visualizer.add_geometry(display_cloud)
                    if args.free_view:
                        visualizer.reset_view_point(True)
                    else:
                        set_camera_forward_view(visualizer, args)
                    has_display_cloud = True
                else:
                    visualizer.update_geometry(display_cloud)

                if state["reset_view_requested"]:
                    set_camera_forward_view(visualizer, args)
                    state["reset_view_requested"] = False

                if state["save_requested"]:
                    save_point_cloud(point_cloud, args)
                    state["save_requested"] = False

                frame_counter += 1
                now = time.perf_counter()
                if now - previous_time >= 1.0:
                    fps = frame_counter / (now - previous_time)
                    print(f"点数：{point_cloud.size}，显示帧率：{fps:.1f} fps")
                    frame_counter = 0
                    previous_time = now

                if not visualizer.poll_events():
                    break
                visualizer.update_renderer()
    finally:
        visualizer.destroy_window()


if __name__ == "__main__":
    main()
