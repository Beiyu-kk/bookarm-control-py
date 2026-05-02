"""ESP32 firmware command and device id definitions."""

from __future__ import annotations

from enum import IntEnum


class CommandId(IntEnum):
    SINGLE_JOINT_RAD = 101
    CMD_JOINTS_ARRAY_FEEDBACK = 121
    JOINTS_TORQUE_LOCK = 122
    JOINTS_NAMED_RAD = 123
    EXT_GRIPPER_OPEN = 130
    EXT_GRIPPER_CLOSE = 131
    EXT_GRIPPER_DEG = 132
    EXT_GRIPPER_FEEDBACK = 133
    EXT_GRIPPER_FEEDBACK_RESPONSE = 1331
    EXT_GRIPPER_TORQUE = 134
    EXT_GRIPPER_HOLD_CLOSE = 138


class Joint(IntEnum):
    BASE = 1
    SHOULDER = 2
    ELBOW = 3
    WRIST = 4
    GRIPPER = 5
    ROD = 6


class ServoId(IntEnum):
    GRIPPER = 1
    BASE = 11
    SHOULDER_DRIVING = 12
    SHOULDER_DRIVEN = 13
    ELBOW = 14
    WRIST = 15
    ROD = 16


__all__ = ["CommandId", "Joint", "ServoId"]
