"""Gripper actuator interface.

This module converts high-level gripper intent to ESP32 JSON dictionaries and,
when a transport is attached, sends them to the firmware.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bookarm_control_py.bookarm import GripperCommand
from bookarm_control_py.protocol.esp32 import DEFAULT_USB_BAUDRATE, JsonSerialTransport
from bookarm_control_py.protocol.id_config import CommandId


class GripperActuator:
    """Convert gripper commands to ESP32 JSON and optionally send them."""

    _ACTION_TO_COMMAND_ID = {
        "open": CommandId.EXT_GRIPPER_OPEN,
        "close": CommandId.EXT_GRIPPER_CLOSE,
        "angle": CommandId.EXT_GRIPPER_DEG,
        "feedback": CommandId.EXT_GRIPPER_FEEDBACK,
        "torque": CommandId.EXT_GRIPPER_TORQUE,
    }

    def __init__(
        self,
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
        self.transport = transport

    def open_transport(self) -> "GripperActuator":
        if self.transport is not None:
            self.transport.open()
        return self

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()

    def send_command(
        self,
        command: GripperCommand,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        message = self.to_json(command)
        return self._send_or_return(message, wait_response, response_timeout)

    def open(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(self.build_open(), wait_response, response_timeout)

    def close_gripper(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(self.build_close(), wait_response, response_timeout)

    def set_angle(
        self,
        angle_deg: float,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(
            self.build_set_angle(angle_deg),
            wait_response,
            response_timeout,
        )

    def feedback(
        self,
        *,
        wait_response: bool = False,
        response_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        return self._send_or_return(self.build_feedback(), wait_response, response_timeout)

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

    def to_json(self, command: GripperCommand) -> dict[str, Any]:
        command_id = self._ACTION_TO_COMMAND_ID[command.action]
        message: dict[str, Any] = {"id": int(command_id)}

        if command.action == "angle":
            message["angle"] = self._require_value(command)
        elif command.action == "torque":
            message["torque"] = self._require_value(command)

        return message

    def build_open(self) -> dict[str, Any]:
        return self.to_json(GripperCommand(action="open"))

    def build_close(self) -> dict[str, Any]:
        return self.to_json(GripperCommand(action="close"))

    def build_set_angle(self, angle_deg: float) -> dict[str, Any]:
        return self.to_json(GripperCommand(action="angle", value=float(angle_deg)))

    def build_feedback(self) -> dict[str, Any]:
        return self.to_json(GripperCommand(action="feedback"))

    def build_set_torque(self, torque: float) -> dict[str, Any]:
        return self.to_json(GripperCommand(action="torque", value=float(torque)))

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
    def _require_value(command: GripperCommand) -> float:
        if command.value is None:
            raise ValueError(f"{command.action} command requires value")
        return float(command.value)
