"""Usage demo and smoke test for the current BookArm architecture.

Run from the project root:

    conda run -n bookarm-beiyu python scripts/bookarm_usage_demo.py

This script checks:
1. High-level BookArm URDF loading.
2. Forward kinematics and inverse kinematics.
3. IK results stay inside URDF joint limits.
4. High-level Gripper only creates intent commands.
5. actuator modules convert arm/gripper commands to ESP32 JSON dictionaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bookarm_control_py import BookArm, GripperCommand
from bookarm_control_py.actuator import ArmActuator, GripperActuator


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def open(self) -> "FakeTransport":
        return self

    def close(self) -> None:
        pass

    def send(self, command: dict[str, Any]) -> None:
        self.sent.append(dict(command))

    def request(self, command: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        self.sent.append(dict(command))
        return {"ok": True, "command": dict(command)}


def format_array(values: np.ndarray) -> str:
    return np.array2string(values, precision=5, suppress_small=True)


def print_joint_limits(robot: BookArm) -> None:
    lower = robot.model.lowerPositionLimit
    upper = robot.model.upperPositionLimit

    print("Joint limits:")
    for index, name in enumerate(robot.joint_names):
        lower_rad = lower[index]
        upper_rad = upper[index]
        lower_deg = np.rad2deg(lower_rad)
        upper_deg = np.rad2deg(upper_rad)
        print(
            f"  {name}: "
            f"{lower_rad:.5f} to {upper_rad:.5f} rad "
            f"({lower_deg:.2f} to {upper_deg:.2f} deg)"
        )


def test_model_loading() -> BookArm:
    robot = BookArm()

    print("BookArm loaded.")
    print(f"URDF: {robot.urdf_path}")
    print(f"nq={robot.nq}, nv={robot.nv}")
    print(f"actuated joints: {robot.joint_names}")
    print_joint_limits(robot)

    assert robot.nq == 5
    assert robot.nv == 5
    assert len(robot.joint_names) == robot.nq
    assert robot.end_effector_link == "link5"

    return robot


def test_kinematics(robot: BookArm) -> np.ndarray:
    q_sample = robot.check_joint_angles([0.2, -0.3, 0.4, 0.1, 0.0])

    fk = robot.forward_kinematics_dict(q_sample)
    print("\nForward kinematics:")
    print(f"  q: {format_array(q_sample)}")
    print(f"  position: {format_array(fk['position'])}")
    print(f"  rotation:\n{format_array(fk['rotation'])}")
    print(f"  transform:\n{format_array(fk['transform'])}")

    assert fk["position"].shape == (3,)
    assert fk["rotation"].shape == (3, 3)
    assert fk["transform"].shape == (4, 4)

    ik_result = robot.inverse_kinematics(
        target_position=fk["position"],
        q0=np.zeros(robot.nq),
        max_iterations=200,
        tolerance=1e-4,
    )
    solved_fk = robot.forward_kinematics_dict(ik_result.q)
    position_error = np.linalg.norm(fk["position"] - solved_fk["position"])
    within_lower = np.all(ik_result.q >= robot.model.lowerPositionLimit - 1e-9)
    within_upper = np.all(ik_result.q <= robot.model.upperPositionLimit + 1e-9)

    print("\nInverse kinematics:")
    print(f"  success: {ik_result.success}")
    print(f"  iterations: {ik_result.iterations}")
    print(f"  solver error norm: {ik_result.error_norm:.8f}")
    print(f"  position error: {position_error:.8f}")
    print(f"  solved q: {format_array(ik_result.q)}")
    print(f"  inside lower limits: {within_lower}")
    print(f"  inside upper limits: {within_upper}")

    assert ik_result.success
    assert position_error < 1e-3
    assert within_lower
    assert within_upper

    return ik_result.q


def test_gripper_intent(robot: BookArm) -> GripperCommand:
    open_command = robot.gripper.open()
    close_command = robot.gripper.close()
    angle_command = robot.gripper.set_angle(30)
    torque_command = robot.gripper.set_torque(0.5)

    print("\nHigh-level gripper intent:")
    print(f"  open: {open_command}")
    print(f"  close: {close_command}")
    print(f"  angle: {angle_command}")
    print(f"  torque: {torque_command}")

    assert open_command.action == "open"
    assert close_command.action == "close"
    assert angle_command == GripperCommand(action="angle", value=30.0)
    assert torque_command == GripperCommand(action="torque", value=0.5)

    return angle_command


def test_actuator_json(
    robot: BookArm,
    joint_angles_rad: np.ndarray,
    gripper_command: GripperCommand,
) -> None:
    arm_actuator = ArmActuator(joint_count=robot.nq)
    gripper_actuator = GripperActuator()

    arm_rad_json = arm_actuator.move_joints_rad(joint_angles_rad)
    arm_deg_json = arm_actuator.move_joints_deg(np.rad2deg(joint_angles_rad))
    single_joint_json = arm_actuator.move_single_joint_rad(1, joint_angles_rad[0])
    gripper_json = gripper_actuator.to_json(gripper_command)
    gripper_open_json = gripper_actuator.open()

    print("\nESP32 JSON conversion:")
    print(f"  arm joints rad json: {arm_rad_json}")
    print(f"  arm joints deg json: {arm_deg_json}")
    print(f"  single joint rad json: {single_joint_json}")
    print(f"  gripper angle json: {gripper_json}")
    print(f"  gripper open json: {gripper_open_json}")

    assert_json(arm_rad_json, {"id": 102})
    assert_json(arm_deg_json, {"id": 122})
    assert_json(single_joint_json, {"id": 101, "joint": 1})
    assert_json(gripper_json, {"id": 132, "angle": 30.0})
    assert_json(gripper_open_json, {"id": 130})
    assert len(arm_rad_json["joints"]) == robot.nq
    assert len(arm_deg_json["joints"]) == robot.nq


def test_bookarm_builtin_actuator(robot: BookArm, joint_angles_rad: np.ndarray) -> None:
    fake_transport = FakeTransport()
    robot.connect_actuators(
        arm_actuator=ArmActuator(
            transport=fake_transport,  # type: ignore[arg-type]
            joint_count=robot.nq,
        ),
        gripper_actuator=GripperActuator(
            transport=fake_transport,  # type: ignore[arg-type]
        ),
    )

    robot.move_joints_rad(joint_angles_rad)
    gripper_response = robot.set_gripper_angle(30, wait_response=True)

    print("\nBookArm built-in actuator path:")
    print(f"  sent arm command: {fake_transport.sent[0]}")
    print(f"  gripper response: {gripper_response}")

    assert fake_transport.sent[0]["id"] == 102
    assert len(fake_transport.sent[0]["joints"]) == robot.nq
    assert gripper_response == {"ok": True, "command": {"id": 132, "angle": 30.0}}


def test_invalid_inputs(robot: BookArm) -> None:
    try:
        robot.forward_kinematics([0.0, 0.0])
    except ValueError:
        print("\nInvalid input check:")
        print("  BookArm rejected wrong q length.")
    else:
        raise AssertionError("BookArm should reject wrong q length.")

    try:
        ArmActuator(joint_count=robot.nq).move_joints_rad([0.0, 0.0])
    except ValueError:
        print("  ArmActuator rejected wrong joint list length.")
    else:
        raise AssertionError("ArmActuator should reject wrong joint list length.")

    try:
        robot.forward_kinematics([0.0, 0.0, 0.0, 99.0, 0.0])
    except ValueError:
        print("  BookArm rejected out-of-range joint angles.")
    else:
        raise AssertionError("BookArm should reject out-of-range joint angles.")

    try:
        robot.inverse_kinematics(
            target_position=[0.2, 0.0, 0.3],
            q0=[0.0, 0.0, 0.0, 99.0, 0.0],
        )
    except ValueError:
        print("  BookArm rejected out-of-range IK initial configuration.")
    else:
        raise AssertionError("BookArm should reject out-of-range IK q0.")


def assert_json(actual: dict[str, Any], expected_items: dict[str, Any]) -> None:
    for key, expected_value in expected_items.items():
        assert actual[key] == expected_value


def main() -> None:
    robot = test_model_loading()
    solved_q = test_kinematics(robot)
    gripper_command = test_gripper_intent(robot)
    test_actuator_json(robot, solved_q, gripper_command)
    test_bookarm_builtin_actuator(robot, solved_q)
    test_invalid_inputs(robot)

    print("\nAll current BookArm architecture checks passed.")


if __name__ == "__main__":
    main()
