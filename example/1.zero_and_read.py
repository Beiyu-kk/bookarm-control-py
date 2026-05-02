"""将真实机械臂移动到零位，并通过 ESP32 反馈进行验证。

请在项目根目录运行：

    conda run -n bookarm-beiyu python example/1.zero_and_read.py --port COM8

该脚本用于真实硬件。脚本会先从 ESP32 读取当前关节角度反馈，然后发送零位
关节运动命令；等待机械臂稳定后再次读取反馈，检查机械臂是否接近期望零位，
并用最终反馈关节角通过 BookArm 正运动学计算末端位姿。

该脚本只调用 BookArm 的高层操作。BookArm 会继续把硬件通信委托给 actuator
层和 protocol 层。

如果 ESP32 返回 joints 或 q 这类无法从字段名判断单位的数据，并且单位是角度制，
请使用 --input-unit deg。
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from bookarm_control_py import ArmFeedback, BookArm


DEFAULT_ZERO_TOLERANCE_DEG = 3.0
# 固件零位运动命令单位：speed 为 deg/s，acceleration 为 deg/s^2。
DEFAULT_ZERO_SPEED = 25.0
DEFAULT_ZERO_ACCELERATION = 5.0
DEFAULT_SETTLE_TIME = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取真实 ESP32 关节反馈，并检查 BookArm 零位构型。"
    )
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    parser.add_argument(
        "--input-unit",
        choices=["rad", "deg"],
        default="rad",
        help="ESP32 反馈关节角使用的单位。",
    )
    parser.add_argument(
        "--expected-zero",
        default=None,
        help=(
            "期望零位关节角，用逗号分隔。"
            "单位由 --input-unit 指定，默认全部为 0。"
        ),
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE_DEG,
        help="每个关节允许的零位误差，单位度。",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=DEFAULT_SETTLE_TIME,
        help="发送零位运动命令后等待机械臂稳定的时间，单位秒。",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_ZERO_SPEED,
        help="零位运动命令的速度，单位 deg/s。",
    )
    parser.add_argument(
        "--acc",
        type=float,
        default=DEFAULT_ZERO_ACCELERATION,
        help="零位运动命令的加速度，单位 deg/s^2。",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="在解析结果前打印 ESP32 原始反馈 JSON。",
    )
    return parser.parse_args()


def parse_expected_zero(value: str | None, joint_count: int, input_unit: str) -> np.ndarray:
    if value is None:
        return np.zeros(joint_count)

    angles = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    if angles.shape != (joint_count,):
        raise ValueError(f"--expected-zero 必须包含 {joint_count} 个数值。")

    if input_unit == "deg":
        angles = np.deg2rad(angles)
    return angles


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def print_feedback(
    title: str,
    feedback: ArmFeedback,
    *,
    show_raw: bool,
) -> None:
    print(f"\n{title}:")
    if show_raw:
        print("  原始反馈:")
        print(json.dumps(feedback.raw, ensure_ascii=True, indent=2))
    print(f"  rad: {format_array(feedback.q_rad)}")
    print(f"  deg: {format_array(np.rad2deg(feedback.q_rad))}")
    print(f"  torque: {format_array(feedback.torque)}")


def main() -> None:
    args = parse_args()
    robot = BookArm()
    robot.connect_serial_arm(port=args.port)
    expected_zero_rad = parse_expected_zero(args.expected_zero, robot.nq, args.input_unit)
    tolerance_rad = math.radians(args.tolerance_deg)

    print("BookArm 零位运动与反馈检查")
    print(f"URDF: {robot.urdf_path}")
    print(f"末端执行器 link: {robot.end_effector_link}")
    print(f"关节名称: {robot.joint_names}")

    try:
        initial_feedback = robot.read_arm_feedback(
            input_unit=args.input_unit,
        )
        print_feedback(
            "零位运动前的当前关节反馈",
            initial_feedback,
            show_raw=args.show_raw,
        )

        print("\n向 ESP32 发送零位运动命令。")
        robot.move_zero_pose(
            speed=args.speed,
            acceleration=args.acc,
        )
        print(f"等待 {args.settle_time:.3f} 秒，让机械臂完成运动并稳定。")
        if args.settle_time > 0:
            time.sleep(args.settle_time)

        final_feedback = robot.read_arm_feedback(
            input_unit=args.input_unit,
        )
    finally:
        robot.close()

    print_feedback(
        "零位运动后的最终关节反馈",
        final_feedback,
        show_raw=args.show_raw,
    )

    zero_error_rad = robot.zero_pose_error(
        final_feedback,
        expected_zero_rad=expected_zero_rad,
    )
    zero_error_deg = np.rad2deg(zero_error_rad)
    zero_ok = np.all(np.abs(zero_error_rad) <= tolerance_rad)

    pose = robot.fkine_dict(final_feedback.q_rad)

    print("\n零位检查:")
    print(f"  期望零位 rad: {format_array(expected_zero_rad)}")
    print(f"  误差 deg: {format_array(zero_error_deg)}")
    print(f"  容许误差 deg: {args.tolerance_deg:.3f}")
    print(f"  是否到达零位: {bool(zero_ok)}")

    print("\n根据 ESP32 关节反馈计算得到的末端位姿:")
    print(f"  位置 xyz meter: {format_array(pose['position'])}")
    print(f"  旋转矩阵:\n{format_array(pose['rotation'])}")
    print(f"  齐次变换矩阵:\n{format_array(pose['transform'])}")

    if not zero_ok:
        raise SystemExit("零位检查失败。")

    print("\n零位检查通过。")


if __name__ == "__main__":
    main()
