"""Check robot zero pose from real ESP32 feedback.

Run from the project root:

    conda run -n bookarm-beiyu python example/1.zero_and_read.py --port COM3

This script is for real hardware. It reads joint angles from ESP32 feedback,
checks whether the arm is near the expected zero pose, and computes the end
effector pose with BookArm forward kinematics.

Expected ESP32 protocol:
1. Host sends one newline-delimited JSON command, for example {"id": 105}.
2. ESP32 returns one newline-delimited JSON object.
3. The returned object contains joint angles in one of these common fields:
   joints_rad, joints, q, angles_rad, data.joints_rad, data.joints, data.q.

Use --input-unit deg if ESP32 returns degrees instead of radians.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from bookarm_control_py import BookArm
from bookarm_control_py.actuator import ArmActuator
from bookarm_control_py.protocol.esp32 import DEFAULT_USB_BAUDRATE


DEFAULT_ZERO_TOLERANCE_DEG = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read real ESP32 joint feedback and check BookArm zero pose."
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
        "--input-unit",
        choices=["rad", "deg"],
        default="rad",
        help="Unit used by ESP32 feedback joint angles.",
    )
    parser.add_argument(
        "--expected-zero",
        default=None,
        help=(
            "Expected zero joint angles, comma-separated. "
            "Uses --input-unit. Default is all zeros."
        ),
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE_DEG,
        help="Allowed zero error in degrees for each joint.",
    )
    return parser.parse_args()


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


def parse_expected_zero(value: str | None, joint_count: int, input_unit: str) -> np.ndarray:
    if value is None:
        return np.zeros(joint_count)

    angles = np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
    if angles.shape != (joint_count,):
        raise ValueError(f"--expected-zero must contain {joint_count} values.")

    if input_unit == "deg":
        angles = np.deg2rad(angles)
    return angles


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def main() -> None:
    args = parse_args()
    arm_actuator = ArmActuator(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        open_delay=args.open_delay,
    ).open()
    robot = BookArm(arm_actuator=arm_actuator)
    expected_zero_rad = parse_expected_zero(args.expected_zero, robot.nq, args.input_unit)
    tolerance_rad = math.radians(args.tolerance_deg)

    print("BookArm zero and feedback check")
    print(f"URDF: {robot.urdf_path}")
    print(f"end effector link: {robot.end_effector_link}")
    print(f"joint names: {robot.joint_names}")
    print(f"feedback command: {arm_actuator.feedback()}")

    try:
        feedback = robot.read_feedback(response_timeout=args.timeout)
    finally:
        robot.disconnect_actuators()

    print("\nRaw ESP32 feedback:")
    print(json.dumps(feedback, ensure_ascii=True, indent=2))

    measured_q = extract_joint_angles(feedback, robot.nq)
    if args.input_unit == "deg":
        measured_q = np.deg2rad(measured_q)

    measured_q = robot.check_joint_angles(measured_q)
    zero_error_rad = measured_q - expected_zero_rad
    zero_error_deg = np.rad2deg(zero_error_rad)
    zero_ok = np.all(np.abs(zero_error_rad) <= tolerance_rad)

    pose = robot.forward_kinematics_dict(measured_q)

    print("\nMeasured joint angles from ESP32:")
    print(f"  rad: {format_array(measured_q)}")
    print(f"  deg: {format_array(np.rad2deg(measured_q))}")

    print("\nZero pose check:")
    print(f"  expected zero rad: {format_array(expected_zero_rad)}")
    print(f"  error deg: {format_array(zero_error_deg)}")
    print(f"  tolerance deg: {args.tolerance_deg:.3f}")
    print(f"  zero ok: {bool(zero_ok)}")

    print("\nEnd effector pose solved from ESP32 joint feedback:")
    print(f"  position xyz meter: {format_array(pose['position'])}")
    print(f"  rotation matrix:\n{format_array(pose['rotation'])}")
    print(f"  transform matrix:\n{format_array(pose['transform'])}")

    if not zero_ok:
        raise SystemExit("Zero pose check failed.")

    print("\nZero pose check passed.")


if __name__ == "__main__":
    main()
