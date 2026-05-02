"""Check the arm serial connection through the high-level BookArm API.

Run from the project root:

    conda run -n bookarm-beiyu python example/0.check_arm_connection.py --port COM8

The script connects to the ESP32, requests arm feedback, prints the raw
response, and prints parsed joint angles/torques.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bookarm_control_py import BookArm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ESP32 arm feedback through BookArm."
    )
    parser.add_argument("--port", default="COM8", help="Serial port, for example COM8.")
    parser.add_argument(
        "--skip-torque-test",
        action="store_true",
        help="Skip the five-joint torque lock/release command test.",
    )
    parser.add_argument(
        "--torque-test-delay",
        type=float,
        default=5,
        help="Delay in seconds between torque lock and release commands.",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def countdown(seconds: float, label: str) -> None:
    if seconds <= 0:
        return

    whole_seconds = int(seconds)
    for remaining in range(whole_seconds, 0, -1):
        print(f"{label}: {remaining}s remaining")
        time.sleep(1)

    fractional = seconds - whole_seconds
    if fractional > 0:
        time.sleep(fractional)


def main() -> None:
    args = parse_args()

    robot = BookArm()
    robot.connect_serial_arm(port=args.port)

    try:
        torque_results: list[tuple[str, dict[str, object] | None]] = []
        if not args.skip_torque_test:
            print("About to release five-joint torque at test start.")
            torque_results.append(("disable torque at start", robot.disable_torque()))
            countdown(args.torque_test_delay, "Torque released")
            torque_results.append(("enable torque", robot.enable_torque()))

        feedback = robot.read_arm_feedback()
    finally:
        robot.close()

    print("BookArm ESP32 connection check")
    print(f"serial port: {args.port}")
    print(f"joint count: {robot.nq}")
    print(f"joint names: {robot.joint_names}")

    if args.skip_torque_test:
        print("\nTorque test skipped.")
    else:
        print("\nFive-joint torque lock/release test:")
        for label, command in torque_results:
            print(f"  {label}: {json.dumps(command, ensure_ascii=True)}")

    print("\nRaw feedback:")
    print(json.dumps(feedback.raw, ensure_ascii=True, indent=2))

    print("\nParsed feedback:")
    print(f"  rad: {format_array(feedback.q_rad)}")
    print(f"  deg: {format_array(np.rad2deg(feedback.q_rad))}")
    print(f"  torque: {format_array(feedback.torque)}")

    print("\nConnection check passed.")


if __name__ == "__main__":
    main()
