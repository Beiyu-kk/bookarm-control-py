"""Send a high-level BookArm joint-angle command to real hardware.

Run from the project root:

    conda run -n bookarm-beiyu python example/2.check_arm_move.py --port COM3

This script tests the full high-level control path:

    BookArm.move_joints_deg -> ArmActuator -> ESP32 JSON serial transport

Edit DEFAULT_TARGET_DEG below, or pass --target-deg, to test another pose.
BookArm.move_joints_deg checks URDF joint limits before sending the command.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from bookarm_control_py import BookArm
from bookarm_control_py.actuator import ArmActuator
from bookarm_control_py.protocol.esp32 import DEFAULT_USB_BAUDRATE


# A known legal joint target in degrees, ordered as:
# joint1_base, joint2_shoulder, joint3_elbow, joint4_wrist, joint_pole
DEFAULT_TARGET_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a high-level BookArm joint command to real hardware."
    )
    parser.add_argument("--port", default="COM8", help="Serial port, for example COM3.")
    parser.add_argument("--baud", type=int, default=DEFAULT_USB_BAUDRATE, help="Serial baudrate.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Serial timeout in seconds.")
    parser.add_argument(
        "--open-delay",
        type=float,
        default=3.0,
        help="Delay after opening the serial port, in seconds.",
    )
    parser.add_argument(
        "--target-deg",
        default=None,
        help=(
            "Target joint angles in degrees, comma-separated. "
            "Default uses DEFAULT_TARGET_DEG in this script."
        ),
    )
    parser.add_argument(
        "--wait-response",
        action="store_true",
        help="Wait for one JSON response after sending the move command.",
    )
    parser.add_argument(
        "--read-after",
        action="store_true",
        help="Read ESP32 feedback after sending the move command.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="Seconds to wait before --read-after feedback.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print the command path; do not open serial or send.",
    )
    return parser.parse_args()


def parse_joint_degrees(value: str | None, joint_count: int) -> np.ndarray:
    if value is None:
        return DEFAULT_TARGET_DEG.copy()

    angles = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    if angles.shape != (joint_count,):
        raise ValueError(f"--target-deg must contain {joint_count} values.")
    return angles


def extract_joint_angles(feedback: dict[str, Any], joint_count: int) -> np.ndarray:
    """Extract joint angle list from common ESP32 feedback shapes."""

    candidates: list[Any] = [
        feedback.get("joints_rad"),
        feedback.get("angles_rad"),
        feedback.get("joints"),
        feedback.get("q"),
    ]

    data = feedback.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("joints_rad"),
                data.get("angles_rad"),
                data.get("joints"),
                data.get("q"),
            ]
        )
    elif isinstance(data, list):
        candidates.append(data)

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            angles = np.asarray(candidate, dtype=float)
        except (TypeError, ValueError):
            continue
        if angles.shape == (joint_count,):
            return angles

    raise ValueError(
        "Cannot find joint angle feedback. Expected a list field such as "
        "joints_rad, joints, q, data.joints_rad, data.joints, or data.q. "
        f"Raw feedback: {feedback}"
    )


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def print_target_command(robot: BookArm, target_deg: np.ndarray) -> None:
    print("BookArm high-level joint move check")
    print(f"URDF: {robot.urdf_path}")
    print(f"joint names: {robot.joint_names}")
    print(f"target q deg: {format_array(target_deg)}")
    print(f"target q rad: {format_array(np.deg2rad(target_deg))}")


def main() -> None:
    args = parse_args()
    robot = BookArm()
    target_deg = parse_joint_degrees(args.target_deg, robot.nq)
    print_target_command(robot, target_deg)

    if args.dry_run:
        command = ArmActuator(joint_count=robot.nq).move_joints_deg(target_deg)
        print("\nDry run, no serial command was sent.")
        print(f"ESP32 JSON that would be sent: {command}")
        return

    arm_actuator = ArmActuator(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        open_delay=args.open_delay,
    ).open()
    robot.connect_actuators(arm_actuator=arm_actuator)

    try:
        response = robot.move_joints_deg(
            target_deg,
            wait_response=args.wait_response,
            response_timeout=args.timeout,
        )
        print("\nMove command sent through BookArm.move_joints_deg.")
        print(f"move command result: {response}")

        if args.read_after:
            time.sleep(args.settle_time)
            feedback = robot.read_feedback(response_timeout=args.timeout)
            measured_q = extract_joint_angles(feedback, robot.nq)
            joint_error = measured_q - np.deg2rad(target_deg)

            print("\nRaw ESP32 feedback:")
            print(json.dumps(feedback, ensure_ascii=True, indent=2))
            print("\nMeasured joint angles from ESP32:")
            print(f"  rad: {format_array(measured_q)}")
            print(f"  deg: {format_array(np.rad2deg(measured_q))}")
            print(f"  error deg: {format_array(np.rad2deg(joint_error))}")
    finally:
        robot.disconnect_actuators()


if __name__ == "__main__":
    main()
