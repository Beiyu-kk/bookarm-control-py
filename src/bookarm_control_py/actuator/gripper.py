"""Gripper actuator interface.

This module converts gripper commands to ESP32 JSON dictionaries and, when a
transport is attached, sends them to the firmware.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bookarm_control_py.protocol.esp32 import JsonSerialTransport
from bookarm_control_py.protocol.id_config import CommandId


class GripperActuator:
    """Convert gripper commands to ESP32 JSON and optionally send them."""

    def __init__(
        self,
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
        self.transport = transport

    def open(self) -> "GripperActuator":
        if self.transport is not None:
            self.transport.open()
        return self

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def open_gripper(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_open(speed=speed, acceleration=acceleration, torque=torque),
            wait_response,
            response_timeout,
        )

    def close_gripper(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_close(speed=speed, acceleration=acceleration, torque=torque),
            wait_response,
            response_timeout,
        )

    def set_angle(
        self,
        angle_deg: float,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_set_angle(
                angle_deg,
                speed=speed,
                acceleration=acceleration,
                torque=torque,
            ),
            wait_response,
            response_timeout,
        )

    def set_torque(
        self,
        torque: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_set_torque(torque),
            wait_response,
            response_timeout,
        )

    def hold_close(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        hold: float = -200.0,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_hold_close(
                speed=speed,
                acceleration=acceleration,
                torque=torque,
                hold=hold,
            ),
            wait_response,
            response_timeout,
        )

    def read_feedback(
        self,
        *,
        response_timeout: float | None = None,
    ) -> dict[str, Any]:
        if self.transport is None:
            raise RuntimeError("GripperActuator has no transport attached.")
        return self.transport.request(
            self.feedback(),
            response_timeout=response_timeout,
            expected_t=int(CommandId.EXT_GRIPPER_FEEDBACK_RESPONSE),
            response_filter=self._contains_gripper_feedback,
            response_description="external gripper feedback T=1331 target=gripper",
        )

    def feedback(self) -> dict[str, Any]:
        return self.build_feedback()

    def build_open(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
    ) -> dict[str, Any]:
        return {
            "T": int(CommandId.EXT_GRIPPER_OPEN),
            "spd": float(speed),
            "acc": float(acceleration),
            "torque": float(torque),
        }

    def build_close(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
    ) -> dict[str, Any]:
        return {
            "T": int(CommandId.EXT_GRIPPER_CLOSE),
            "spd": float(speed),
            "acc": float(acceleration),
            "torque": float(torque),
        }

    def build_set_angle(
        self,
        angle_deg: float,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
    ) -> dict[str, Any]:
        return {
            "T": int(CommandId.EXT_GRIPPER_DEG),
            "angle": float(angle_deg),
            "spd": float(speed),
            "acc": float(acceleration),
            "torque": float(torque),
        }

    def build_feedback(self) -> dict[str, Any]:
        return {"T": int(CommandId.EXT_GRIPPER_FEEDBACK)}

    def build_set_torque(self, torque: float) -> dict[str, Any]:
        return {
            "T": int(CommandId.EXT_GRIPPER_TORQUE),
            "cmd": 1 if bool(torque) else 0,
        }

    def build_hold_close(
        self,
        *,
        speed: float = 100.0,
        acceleration: float = 10.0,
        torque: float = 1000.0,
        hold: float = -200.0,
    ) -> dict[str, Any]:
        return {
            "T": int(CommandId.EXT_GRIPPER_HOLD_CLOSE),
            "spd": float(speed),
            "acc": float(acceleration),
            "torque": float(torque),
            "hold": float(hold),
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

    @staticmethod
    def _contains_gripper_feedback(response: dict[str, Any]) -> bool:
        return response.get("target") == "gripper" or "pos" in response
