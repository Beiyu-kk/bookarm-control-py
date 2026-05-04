"""RealSense D435 可复用封装的实时显示和冒烟测试脚本。

请在项目根目录运行：

    conda run -n bookarm-beiyu python scripts/camera/check_realsense_d435.py

默认会打开 RGB + 深度图实时窗口。按 q 或 Esc 退出，
按 s 保存当前帧的 RGB 图、深度图和点云文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from bookarm_control_py.camera import RealSenseD435, RGBDFrame


DEFAULT_SAVE_DIR = Path("recordings/realsense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 RealSense D435 的 RGB、深度图和点云接口。")
    parser.add_argument("--serial", default=None, help="指定 D435 序列号；只有一台相机时可不填。")
    parser.add_argument("--width", type=int, default=640, help="彩色和深度流宽度。")
    parser.add_argument("--height", type=int, default=480, help="彩色和深度流高度。")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率。")
    parser.add_argument("--warmup", type=int, default=30, help="丢弃前 N 帧等待自动曝光稳定。")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="等待帧的超时时间。")
    parser.add_argument("--max-depth", type=float, default=2.0, help="深度可视化和点云过滤的最大距离，单位米。")
    parser.add_argument("--stride", type=int, default=2, help="点云采样步长；1 表示保留所有有效深度点。")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="测试输出目录。")
    parser.add_argument("--prefix", default="d435", help="保存文件名前缀。")
    parser.add_argument("--no-align", action="store_true", help="不把深度图对齐到彩色图。")
    parser.add_argument("--once", action="store_true", help="只采集一帧并退出，使用旧的一帧测试流程。")
    parser.add_argument("--no-save", action="store_true", help="在 --once 模式下只打印结果，不保存文件。")
    parser.add_argument("--save-ply", action="store_true", help="额外保存可用 CloudCompare/MeshLab 打开的 PLY 点云。")
    parser.add_argument("--preview", action="store_true", help="在 --once 模式下用 OpenCV 窗口预览单帧。")
    parser.add_argument("--skip-one-shot-check", action="store_true", help="跳过 get_color/depth/point_cloud 便捷接口检查。")
    return parser.parse_args()


def print_frame_summary(frame: RGBDFrame, max_depth_m: float) -> None:
    rgb = frame.color("rgb")
    depth_m = frame.depth("m")
    depth_raw = frame.depth("raw")
    valid_mask = frame.valid_depth_mask(max_depth_m=max_depth_m)
    valid_depth = depth_m[valid_mask]

    print("已采集 RealSense D435 帧")
    print(f"  帧号：{frame.frame_number}")
    print(f"  时间戳 ms：{frame.timestamp_ms:.3f}")
    print(f"  深度是否对齐到彩色图：{frame.depth_aligned_to_color}")
    print(f"  RGB 图：shape={rgb.shape}, dtype={rgb.dtype}")
    print(f"  原始深度图：shape={depth_raw.shape}, dtype={depth_raw.dtype}")
    print(f"  米制深度图：shape={depth_m.shape}, dtype={depth_m.dtype}, scale={frame.depth_scale:.8f}")
    print(
        "  相机内参："
        f"{frame.intrinsics.width}x{frame.intrinsics.height}, "
        f"fx={frame.intrinsics.fx:.3f}, fy={frame.intrinsics.fy:.3f}, "
        f"ppx={frame.intrinsics.ppx:.3f}, ppy={frame.intrinsics.ppy:.3f}"
    )

    if valid_depth.size:
        print(
            "  有效深度："
            f"count={valid_depth.size}, "
            f"min={float(np.min(valid_depth)):.4f} m, "
            f"mean={float(np.mean(valid_depth)):.4f} m, "
            f"max={float(np.max(valid_depth)):.4f} m"
        )
    else:
        print("  有效深度：count=0")


def preview_frame(frame: RGBDFrame, max_depth_m: float) -> None:
    import cv2

    cv2.imshow("D435 color", frame.color_bgr)
    cv2.imshow("D435 depth", frame.depth_colormap_bgr(max_depth_m=max_depth_m))
    print("按任意键关闭预览窗口。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_live_view(frame: RGBDFrame, max_depth_m: float, fps: float) -> np.ndarray:
    import cv2

    color = frame.color_bgr.copy()
    depth_vis = frame.depth_colormap_bgr(max_depth_m=max_depth_m)
    if depth_vis.shape[:2] != color.shape[:2]:
        depth_vis = cv2.resize(
            depth_vis,
            (color.shape[1], color.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    height, width = frame.depth_m.shape
    center_depth = float(frame.depth_m[height // 2, width // 2])
    center_text = (
        f"center depth: {center_depth:.3f} m"
        if np.isfinite(center_depth) and center_depth > 0.0
        else "center depth: invalid"
    )

    for image, title in ((color, "RGB"), (depth_vis, f"Depth 0-{max_depth_m:.1f}m")):
        cv2.putText(
            image,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        color,
        f"fps: {fps:.1f}  frame: {frame.frame_number}",
        (12, color.shape[0] - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        color,
        "q/Esc: quit   s: save",
        (12, color.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_vis,
        center_text,
        (12, depth_vis.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.hstack((color, depth_vis))


def save_frame_outputs(
    frame: RGBDFrame,
    args: argparse.Namespace,
    point_cloud_prefix: str | None = None,
) -> None:
    prefix = point_cloud_prefix or f"{args.prefix}_{frame.frame_number}"
    image_paths = frame.save_images(
        directory=args.save_dir,
        prefix=prefix,
        depth_vis_max_m=args.max_depth,
    )
    point_cloud = frame.to_point_cloud(
        max_depth_m=args.max_depth,
        stride=args.stride,
        include_color=True,
    )
    point_cloud_path = point_cloud.save_npz(args.save_dir / f"{prefix}_point_cloud.npz")

    print("已保存文件：")
    for name, path in image_paths.items():
        print(f"  {name}: {path}")
    print(f"  point_cloud_npz: {point_cloud_path}")

    if args.save_ply:
        ply_path = point_cloud.save_ply(args.save_dir / f"{prefix}_point_cloud.ply")
        print(f"  point_cloud_ply: {ply_path}")


def run_live(camera: RealSenseD435, args: argparse.Namespace) -> None:
    import cv2

    print("实时显示已启动：按 q 或 Esc 退出，按 s 保存当前帧。")
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        for frame in camera.iter_frames(timeout_ms=args.timeout_ms):
            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps <= 0 else 0.9 * fps + 0.1 * instant_fps

            cv2.imshow(
                "RealSense D435 RGB + Depth",
                draw_live_view(frame, max_depth_m=args.max_depth, fps=fps),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_frame_outputs(frame, args)
    finally:
        cv2.destroyAllWindows()


def run_once(camera: RealSenseD435, args: argparse.Namespace) -> None:
    frame = camera.get_rgbd_frame(timeout_ms=args.timeout_ms)
    print_frame_summary(frame, max_depth_m=args.max_depth)

    point_cloud = frame.to_point_cloud(
        max_depth_m=args.max_depth,
        stride=args.stride,
        include_color=True,
    )
    has_color = point_cloud.colors_rgb is not None
    print(
        "  点云："
        f"points={point_cloud.size}, "
        f"xyz dtype={point_cloud.points_xyz_m.dtype}, "
        f"是否带颜色={has_color}, "
        f"stride={args.stride}"
    )

    if not args.skip_one_shot_check:
        one_rgb = camera.get_color_image(timeout_ms=args.timeout_ms, order="rgb")
        one_depth = camera.get_depth_image(timeout_ms=args.timeout_ms, unit="m")
        one_cloud = camera.get_point_cloud(
            timeout_ms=args.timeout_ms,
            max_depth_m=args.max_depth,
            stride=args.stride,
            include_color=True,
        )
        print(
            "  单帧便捷接口检查："
            f"rgb={one_rgb.shape}/{one_rgb.dtype}, "
            f"depth={one_depth.shape}/{one_depth.dtype}, "
            f"cloud_points={one_cloud.size}"
        )

    if args.preview:
        preview_frame(frame, max_depth_m=args.max_depth)

    if not args.no_save:
        save_frame_outputs(frame, args)


def main() -> None:
    args = parse_args()

    with RealSenseD435(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        align_depth_to_color=not args.no_align,
    ) as camera:
        camera.warmup(frame_count=args.warmup, timeout_ms=args.timeout_ms)

        if args.once:
            run_once(camera, args)
        else:
            run_live(camera, args)


if __name__ == "__main__":
    main()
