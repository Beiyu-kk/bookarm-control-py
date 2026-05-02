"""Arm actuator interface.

This module converts arm joint commands to ESP32 JSON dictionaries and, when a
transport is attached, sends them to the firmware.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from bookarm_control_py.protocol.esp32 import JsonSerialTransport
from bookarm_control_py.protocol.id_config import CommandId, Joint


class ArmActuator:
    """Convert arm joint commands to ESP32 JSON and optionally send them."""

    def __init__(
        self,
        joint_count: int = 5,
        *,
        port: str | None = None,
        transport: JsonSerialTransport | None = None,
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        open_delay: float = 3.0,
    ) -> None:
        if transport is None and port is not None:
            transport = JsonSerialTransport(
                port=port,
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
        speed: float,
        acceleration: float,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_move_joints_rad(
            joint_angles_rad,
            speed=speed,
            acceleration=acceleration,
        )
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

    def enable_torque(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_torque_lock(enabled=True)
        return self._send_or_return(command, wait_response, response_timeout)

    def disable_torque(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        command = self.build_torque_lock(enabled=False)
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
            response_filter=self._contains_joint_feedback,
            response_description=(
                "joint array feedback with joints_rad and joints_torque lists"
            ),
        )

    def feedback(self) -> dict[str, Any]:
        return {"T": int(CommandId.CMD_JOINTS_ARRAY_FEEDBACK)}

    def build_torque_lock(self, *, enabled: bool) -> dict[str, Any]:
        return {
            "T": int(CommandId.JOINTS_TORQUE_LOCK),
            "cmd": 1 if enabled else 0,
        }

    def build_move_joints_rad(
        self,
        joint_angles_rad: Iterable[float],
        *,
        speed: float,
        acceleration: float,
    ) -> dict[str, Any]:
        base, shoulder, elbow, hand, r = self._as_joint_list(joint_angles_rad)
        return {
            "T": int(CommandId.JOINTS_NAMED_RAD),
            "base": float(base),
            "shoulder": float(shoulder),
            "elbow": float(elbow),
            "hand": float(hand),
            "r": float(r),
            "spd": float(speed),
            "acc": float(acceleration),
        }

    def build_move_joints_deg(self, joint_angles_deg: Iterable[float]) -> dict[str, Any]:
        raise NotImplementedError(
            "The current ESP32 protocol does not define a five-joint degree command. "
            "Use move_joints_rad/build_move_joints_rad with command T=123."
        )

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
        raise NotImplementedError(
            "The current ESP32 protocol does not define a single-joint degree command."
        )

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

    def _contains_joint_feedback(self, response: dict[str, Any]) -> bool:
        return (
            self._extract_joint_feedback_candidate(response) is not None
            and self._extract_joint_torque_candidate(response) is not None
        )

    def _extract_joint_feedback_candidate(self, response: dict[str, Any]) -> np.ndarray | None:
        candidates: list[Any] = [
            response.get("joints_rad"),
            response.get("angles_rad"),
            response.get("joints_deg"),
            response.get("angles_deg"),
            response.get("joints"),
            response.get("q"),
        ]

        data = response.get("data")
        if isinstance(data, dict):
            candidates.extend(
                [
                    data.get("joints_rad"),
                    data.get("angles_rad"),
                    data.get("joints_deg"),
                    data.get("angles_deg"),
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
            if angles.shape == (self.joint_count,):
                return angles

        return None

    def _extract_joint_torque_candidate(self, response: dict[str, Any]) -> np.ndarray | None:
        candidates: list[Any] = [
            response.get("joints_torque"),
            response.get("torques"),
            response.get("tau"),
        ]

        data = response.get("data")
        if isinstance(data, dict):
            candidates.extend(
                [
                    data.get("joints_torque"),
                    data.get("torques"),
                    data.get("tau"),
                ]
            )

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                torques = np.asarray(candidate, dtype=float)
            except (TypeError, ValueError):
                continue
            if torques.shape == (self.joint_count,):
                return torques

        return None
