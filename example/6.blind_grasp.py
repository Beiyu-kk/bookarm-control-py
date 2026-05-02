"""盲抓测试：移动到目标位置、闭合夹爪、带物体返回起始构型。

请在项目根目录运行：

    conda run -n bookarm-beiyu python example/6.blind_grasp.py --port COM8
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bookarm_control_py import BookArm


START_Q_DEG = np.array([0.0, -70.0, 75.0, 0.0, 0.0], dtype=float)
GRASP_POSITION = np.array([0.3, 0.0, 0.1], dtype=float)
ARM_SETTLE_TIME = 5.0
GRIPPER_SETTLE_TIME = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 BookArm 盲抓测试。")
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def wait_arm() -> None:
    time.sleep(ARM_SETTLE_TIME)


def wait_gripper() -> None:
    time.sleep(GRIPPER_SETTLE_TIME)


def main() -> None:
    args = parse_args()

    robot = BookArm()
    start_q = np.deg2rad(START_Q_DEG)
    start_q = robot.check_joint_angles(start_q, context="起始关节构型")

    ik_result = robot.ikine(
        target_position=GRASP_POSITION,
        q0=start_q,
        max_iterations=500,
        tolerance=1e-4,
    )
    if not ik_result.success:
        raise RuntimeError(f"抓取位置逆解失败，误差: {ik_result.error_norm:.6f}")

    grasp_q = robot.check_joint_angles(ik_result.q, context="抓取关节构型")

    print("BookArm 盲抓测试")
    print(f"串口号: {args.port}")
    print(f"起始构型 deg: {format_array(START_Q_DEG)}")
    print(f"抓取位置 xyz: {format_array(GRASP_POSITION)}")
    print(f"抓取构型 deg: {format_array(np.rad2deg(grasp_q))}")
    print(f"逆解误差: {ik_result.error_norm:.8f}")

    robot.connect_serial(port=args.port)
    try:
        print("\n打开夹爪，准备进入起始构型")
        print(robot.open_gripper())
        wait_gripper()

        print("\n机械臂运动到起始构型")
        print(robot.move_joints_rad(start_q))
        wait_arm()

        print("\n机械臂运动到抓取位置")
        print(robot.move_joints_rad(grasp_q))
        wait_arm()

        grasp_feedback = robot.read_arm_feedback()
        grasp_pose = robot.fkine_dict(grasp_feedback.q_rad)
        grasp_position_error = grasp_pose["position"] - GRASP_POSITION
        print("到达抓取位置后的反馈:")
        print(f"  当前关节角 deg: {format_array(np.rad2deg(grasp_feedback.q_rad))}")
        print(f"  当前末端位置 xyz: {format_array(grasp_pose['position'])}")
        print(f"  位置误差 xyz: {format_array(grasp_position_error)}")
        print(f"  位置误差范数: {np.linalg.norm(grasp_position_error):.8f}")

        print("\n闭合夹爪，执行抓取")
        print(robot.close_gripper())
        wait_gripper()

        print("\n保持夹爪闭合，机械臂返回起始构型")
        print(robot.move_joints_rad(start_q))
        wait_arm()

        print("\n打开夹爪，释放物体")
        print(robot.open_gripper())
        wait_gripper()

    finally:
        robot.close()


if __name__ == "__main__":
    main()
