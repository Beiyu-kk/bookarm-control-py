"""示例 4：连接真实机械臂测试 BookArm.ikine_best_effort。

请在项目根目录运行：

    python example/4.check_ikine_best_effort.py --port COM8 

脚本会通过 connect_serial_arm 连接机械臂，先移动到 START_Q_DEG 定义的起始构型，
等待到位并读取反馈后，再以该反馈关节角作为 q0 做 best-effort IK。
随后脚本会发送目标关节运动命令，并读取真实反馈做 FK 复核。
目标位姿在 main() 中直接给定：位置用 xyz，姿态用欧拉角 rpy，
欧拉角到旋转矩阵的转换调用 bookarm_control_py.math_utils.rpy_to_matrix。

- ikine_best_effort 总会返回一个搜索过程中最接近目标的关节构型。
- 如果 success=False，说明误差没有达到 tolerance，目标位姿可能太难或不可达。
- 本脚本默认会移动到 best-effort 返回的最近构型，并打印真实反馈误差。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bookarm_control_py import BookArm, BestEffortIKResult, rpy_to_matrix


START_Q_DEG = np.array([0.0, -60.0, 70.0, 0.0, -45.0], dtype=float)
DEFAULT_SPEED = 25.0
DEFAULT_ACCELERATION = 5.0
DEFAULT_ARM_WAIT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="连接真实机械臂，测试 BookArm.ikine_best_effort。"
    )
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    parser.add_argument("--max-iterations", type=int, default=500, help="IK 最大迭代次数。")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="IK 收敛容差。")
    parser.add_argument("--damping", type=float, default=1e-6, help="阻尼最小二乘阻尼系数。")
    parser.add_argument("--step-size", type=float, default=0.4, help="IK 每步积分步长。")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="机械臂运动速度。")
    parser.add_argument(
        "--acceleration",
        type=float,
        default=DEFAULT_ACCELERATION,
        help="机械臂运动加速度。",
    )
    parser.add_argument(
        "--arm-wait",
        type=float,
        default=DEFAULT_ARM_WAIT,
        help="发送运动命令后等待机械臂到位的秒数。",
    )
    return parser.parse_args()


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=6, suppress_small=True)


def rotation_error_angle(target_rotation: np.ndarray, current_rotation: np.ndarray) -> float:
    relative_rotation = target_rotation.T @ current_rotation
    cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def solve_best_effort(
    robot: BookArm,
    *,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q0: np.ndarray,
    args: argparse.Namespace,
) -> BestEffortIKResult:
    return robot.ikine_best_effort(
        target_position=target_position,
        target_rotation=target_rotation,
        q0=q0,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        damping=args.damping,
        step_size=args.step_size,
        print_error=False,
    )


def print_pose_error(
    *,
    title: str,
    pose: dict[str, np.ndarray],
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> None:
    position_error = pose["position"] - target_position
    rotation_error_rad = rotation_error_angle(target_rotation, pose["rotation"])

    print(f"\n{title}:")
    print(f"  当前位置 xyz m: {format_array(pose['position'])}")
    print(f"  目标位置 xyz m: {format_array(target_position)}")
    print(f"  位置误差 xyz m: {format_array(position_error)}")
    print(f"  位置误差范数: {np.linalg.norm(position_error):.8f} m")
    print(f"  姿态误差: {np.rad2deg(rotation_error_rad):.8f} deg")


def main() -> None:
    args = parse_args()
    robot = BookArm()
    start_q = robot.check_joint_angles(
        np.deg2rad(START_Q_DEG),
        context="起始关节构型",
    )

    # 目标位姿直接在这里修改。
    # position 单位为米；rpy 单位为度，顺序为 roll, pitch, yaw。
    target_position = np.array([0.38, 0.0, 0.24], dtype=float)
    target_rpy_deg = np.array([0.0, -5.0, 0.0], dtype=float)
    target_rotation = rpy_to_matrix(np.deg2rad(target_rpy_deg))

    print("BookArm ikine_best_effort 真实机械臂测试")
    print(f"串口号: {args.port}")
    print(f"关节名称: {robot.joint_names}")
    print(f"起始构型 q deg: {format_array(START_Q_DEG)}")
    print(f"目标位置 xyz m: {format_array(target_position)}")
    print(f"目标欧拉角 rpy deg: {format_array(target_rpy_deg)}")
    print(f"目标旋转矩阵:\n{format_array(target_rotation)}")

    robot.connect_serial_arm(port=args.port)
    try:
        initial_feedback = robot.read_arm_feedback()

        print("\n机械臂初始反馈:")
        print(f"  当前 q deg: {format_array(np.rad2deg(initial_feedback.q_rad))}")
        print(f"  当前力矩: {format_array(initial_feedback.torque)}")

        print("\n先移动到起始构型。")
        print(robot.move_joints_rad(start_q, speed=args.speed, acceleration=args.acceleration))
        if args.arm_wait > 0:
            time.sleep(args.arm_wait)

        start_feedback = robot.read_arm_feedback()
        q0 = robot.check_joint_angles(start_feedback.q_rad, context="起始构型反馈")

        print("\n到达起始构型后的反馈:")
        print(f"  当前 q deg: {format_array(np.rad2deg(start_feedback.q_rad))}")
        print(f"  目标起始 q deg: {format_array(START_Q_DEG)}")
        print(f"  起始关节误差 deg: {format_array(np.rad2deg(start_feedback.q_rad - start_q))}")
        print(f"  当前力矩: {format_array(start_feedback.torque)}")
        print(f"  IK 初始 q deg: {format_array(np.rad2deg(q0))}")

        result = solve_best_effort(
            robot,
            target_position=target_position,
            target_rotation=target_rotation,
            q0=q0,
            args=args,
        )

        solved_pose = robot.fkine_dict(result.q)
        print("\nbest-effort IK 结果:")
        print(f"  是否达到容差: {result.success}")
        print(f"  迭代次数: {result.iterations}")
        print(f"  q rad: {format_array(result.q)}")
        print(f"  q deg: {format_array(np.rad2deg(result.q))}")
        print(f"  总误差: {result.error_norm:.8f}")
        print(f"  位置误差: {result.position_error_norm:.8f} m")
        print(f"  姿态误差: {np.rad2deg(result.rotation_error_rad):.8f} deg")
        print_pose_error(
            title="逆解 q 的正运动学复核",
            pose=solved_pose,
            target_position=target_position,
            target_rotation=target_rotation,
        )

        command = robot.move_joints_rad(
            result.q,
            speed=args.speed,
            acceleration=args.acceleration,
        )
        print("\n运动命令已发送:")
        print(command)

        if args.arm_wait > 0:
            time.sleep(args.arm_wait)

        final_feedback = robot.read_arm_feedback()
    finally:
        robot.close()

    final_pose = robot.fkine_dict(final_feedback.q_rad)
    joint_error_deg = np.rad2deg(final_feedback.q_rad - result.q)

    print("\n运动后的真实机械臂反馈:")
    print(f"  当前 q deg: {format_array(np.rad2deg(final_feedback.q_rad))}")
    print(f"  目标 q deg: {format_array(np.rad2deg(result.q))}")
    print(f"  关节误差 deg: {format_array(joint_error_deg)}")
    print(f"  力矩: {format_array(final_feedback.torque)}")
    print_pose_error(
        title="真实机械臂当前位姿与目标位姿误差",
        pose=final_pose,
        target_position=target_position,
        target_rotation=target_rotation,
    )


if __name__ == "__main__":
    main()
