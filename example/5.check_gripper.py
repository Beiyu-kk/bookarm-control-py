"""测试夹爪反馈、开合、指定角度和力矩控制。

请在项目根目录运行：

    conda run -n bookarm-beiyu python example/5.check_gripper.py --port COM8
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Literal

from bookarm_control_py import BookArm


TEST_ANGLE_DEG = 30.0
TORQUE_ON = 1.0
TORQUE_OFF = 0.0
GRIPPER_ACTION_DELAY = 3.0
GRIPPER_CLOSED_POS_MAX = 1000.0
GRIPPER_OPEN_POS_MIN = 2000.0

GripperState = Literal["open", "closed", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 BookArm 夹爪功能。")
    parser.add_argument("--port", default="COM8", help="串口号，例如 COM8。")
    parser.add_argument(
        "--gripper-action-delay",
        type=float,
        default=GRIPPER_ACTION_DELAY,
        help="Seconds to wait after open, close, and angle commands.",
    )
    parser.add_argument(
        "--closed-pos-max",
        type=float,
        default=GRIPPER_CLOSED_POS_MAX,
        help="Raw feedback pos at or below this value is treated as closed.",
    )
    parser.add_argument(
        "--open-pos-min",
        type=float,
        default=GRIPPER_OPEN_POS_MIN,
        help="Raw feedback pos at or above this value is treated as open.",
    )
    return parser.parse_args()


def print_json(title: str, value: dict[str, Any]) -> None:
    print(f"\n{title}:")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def nested_get(feedback: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in feedback:
            return feedback[name]

    data = feedback.get("data")
    if isinstance(data, dict):
        for name in names:
            if name in data:
                return data[name]

    return None


def infer_gripper_state(
    feedback: dict[str, Any],
    *,
    closed_pos_max: float = GRIPPER_CLOSED_POS_MAX,
    open_pos_min: float = GRIPPER_OPEN_POS_MIN,
) -> GripperState:
    is_open = nested_get(feedback, "is_open", "open", "opened")
    if isinstance(is_open, bool):
        return "open" if is_open else "closed"

    is_closed = nested_get(feedback, "is_closed", "closed")
    if isinstance(is_closed, bool):
        return "closed" if is_closed else "open"

    state = nested_get(feedback, "state", "gripper_state")
    if isinstance(state, str):
        normalized = state.lower()
        if "open" in normalized:
            return "open"
        if "close" in normalized or "closed" in normalized:
            return "closed"

    raw_pos = nested_get(feedback, "pos", "position", "raw_pos")
    try:
        pos_value = float(raw_pos)
    except (TypeError, ValueError):
        pos_value = None

    if pos_value is not None:
        if pos_value <= closed_pos_max:
            return "closed"
        if pos_value >= open_pos_min:
            return "open"
        return "unknown"

    angle = nested_get(feedback, "angle", "angle_deg", "gripper_angle")
    try:
        angle_value = float(angle)
    except (TypeError, ValueError):
        return "unknown"

    if angle_value <= 5.0:
        return "closed"
    if angle_value >= 20.0:
        return "open"
    return "unknown"


def run_action(
    title: str,
    command: dict[str, Any] | None,
    delay: float | None = None,
) -> None:
    if delay is None:
        delay = GRIPPER_ACTION_DELAY
    print(f"\n{title}:")
    print(command)
    if delay > 0:
        print(f"Waiting {delay:.1f}s for the gripper action to finish...")
        time.sleep(delay)


def main() -> None:
    global GRIPPER_ACTION_DELAY
    args = parse_args()
    GRIPPER_ACTION_DELAY = args.gripper_action_delay

    robot = BookArm()
    robot.connect_serial_gripper(port=args.port)

    try:
        print("BookArm 夹爪功能测试")
        print(f"串口号: {args.port}")

        initial_feedback = robot.read_gripper_feedback()
        print_json("初始夹爪反馈", initial_feedback)

        state = infer_gripper_state(
            initial_feedback,
            closed_pos_max=args.closed_pos_max,
            open_pos_min=args.open_pos_min,
        )
        print(f"\n根据反馈推断的夹爪状态: {state}")

        if state == "closed":
            run_action("当前夹爪闭合，先执行打开", robot.open_gripper())
        elif state == "open":
            run_action("当前夹爪打开，先执行闭合", robot.close_gripper())
        else:
            run_action("无法判断当前开合状态，先执行打开", robot.open_gripper())

        feedback_after_toggle = robot.read_gripper_feedback()
        print_json("第一次开合动作后的反馈", feedback_after_toggle)

        run_action("测试夹爪打开", robot.open_gripper())
        print_json("打开后的反馈", robot.read_gripper_feedback())

        run_action("测试夹爪闭合", robot.close_gripper())
        print_json("闭合后的反馈", robot.read_gripper_feedback())

        run_action(
            f"测试夹爪指定角度 {TEST_ANGLE_DEG:.1f} deg",
            robot.set_gripper_angle(TEST_ANGLE_DEG),
        )
        print_json("指定角度后的反馈", robot.read_gripper_feedback())

        run_action("测试夹爪力矩开启", robot.set_gripper_torque(TORQUE_ON))
        print_json("力矩开启后的反馈", robot.read_gripper_feedback())

        run_action("测试夹爪力矩关闭", robot.set_gripper_torque(TORQUE_OFF))
        print_json("力矩关闭后的反馈", robot.read_gripper_feedback())

        run_action("测试夹爪持续夹紧 T=138", robot.hold_gripper_closed())
        print_json("持续夹紧后的反馈", robot.read_gripper_feedback())

    finally:
        robot.close()


if __name__ == "__main__":
    main()
