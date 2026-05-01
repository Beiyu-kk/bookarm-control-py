"""Arm actuator interface.

This module converts arm joint commands to ESP32 JSON dictionaries and, when a
transport is attached, sends them to the firmware.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from bookarm_control_py.protocol.esp32 import DEFAULT_USB_BAUDRATE, JsonSerialTransport
from bookarm_control_py.protocol.id_config import CommandId, Joint


class ArmActuator:
    """Convert arm joint commands to ESP32 JSON and optionally send them."""

    def __init__(
        self,
        joint_count: int = 5,
        *,
        port: str | None = None,
        transport: JsonSerialTransport | None = None,
        baudrate: int = DEFAULT_USB_BAUDRATE,
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        open_delay: float = 3.0,
    ) -> None:
        if transport is None and port is not None:
            transport = JsonSerialTransport(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=write_timeout,
                open_delay=open_delay,
            )

        self.joint_count = joint_count
        self.transport = transport

    def open(self) -> "ArmActuator":
        if self.transport is not None:
            self.transport.open()
        return self

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def move_joints_rad(
        self,
        joint_angles_rad: Iterable[float],
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_move_joints_rad(joint_angles_rad)
        return self._send_or_return(command, wait_response, response_timeout)

    def move_joints_deg(
        self,
        joint_angles_deg: Iterable[float],
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_move_joints_deg(joint_angles_deg)
        return self._send_or_return(command, wait_response, response_timeout)

    def move_single_joint_rad(
        self,
        joint: Joint | int,
        angle_rad: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_move_single_joint_rad(joint, angle_rad)
        return self._send_or_return(command, wait_response, response_timeout)

    def move_single_joint_deg(
        self,
        joint: Joint | int,
        angle_deg: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_move_single_joint_deg(joint, angle_deg)
        return self._send_or_return(command, wait_response, response_timeout)

    def read_feedback(
        self,
        *,
        response_timeout: float | None = None,
        expected_t: int | None = None,
    ) -> dict[str, Any]:
        if self.transport is None:
            raise RuntimeError("ArmActuator has no transport attached.")
        return self.transport.request(
            self.feedback(),
            response_timeout=response_timeout,
            expected_t=expected_t,
        )

    def feedback(self) -> dict[str, Any]:
        return {"id": int(CommandId.FEEDBACK)}

    def build_move_joints_rad(self, joint_angles_rad: Iterable[float]) -> dict[str, Any]:
        return {
            "id": int(CommandId.JOINTS_RAD),
            "joints": self._as_joint_list(joint_angles_rad),
        }

    def build_move_joints_deg(self, joint_angles_deg: Iterable[float]) -> dict[str, Any]:
        return {
            "id": int(CommandId.JOINTS_DEG),
            "joints": self._as_joint_list(joint_angles_deg),
        }

    def build_move_single_joint_rad(
        self,
        joint: Joint | int,
        angle_rad: float,
    ) -> dict[str, Any]:
        return {
            "id": int(CommandId.SINGLE_JOINT_RAD),
            "joint": int(joint),
            "angle": float(angle_rad),
        }

    def build_move_single_joint_deg(
        self,
        joint: Joint | int,
        angle_deg: float,
    ) -> dict[str, Any]:
        return {
            "id": int(CommandId.SINGLE_JOINT_DEG),
            "joint": int(joint),
            "angle": float(angle_deg),
        }

    def _send_or_return(
        self,
        command: Mapping[str, Any],
        wait_response: bool,
        response_timeout: float | None,
    ) -> dict[str, Any] | None:
        if self.transport is None:
            return dict(command)
        if wait_response:
            return self.transport.request(command, response_timeout=response_timeout)
        self.transport.send(command)
        return dict(command)

    def _as_joint_list(self, joint_angles: Iterable[float]) -> list[float]:
        angles = np.asarray(list(joint_angles), dtype=float)
        if angles.shape != (self.joint_count,):
            raise ValueError(f"Expected {self.joint_count} joint values, got {angles.shape[0]}")
        return angles.tolist()
