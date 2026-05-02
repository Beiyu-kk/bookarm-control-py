"""ESP32 newline-delimited JSON transport.

This module only handles sending and receiving JSON dictionaries. Conversion
from high-level robot commands to ESP32 JSON lives in ``bookarm_control_py.actuator``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import time
from typing import Any, Callable

from bookarm_control_py.protocol.id_config import CommandId

DEFAULT_USB_BAUDRATE = 921600


class TransportError(RuntimeError):
    """Raised when ESP32 transport cannot complete a request."""


@dataclass(slots=True)
class JsonSerialTransport:
    """Line-delimited JSON serial transport for ESP32 firmware."""

    port: str
    baudrate: int = DEFAULT_USB_BAUDRATE
    timeout: float = 1.0
    write_timeout: float = 1.0
    open_delay: float = 3.0
    dtr: bool = False
    rts: bool = False
    _serial: Any | None = field(default=None, init=False, repr=False)

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def open(self) -> "JsonSerialTransport":
        if self.is_open:
            return self

        try:
            import serial
        except ImportError as exc:
            raise TransportError(
                "pyserial is required for ESP32 serial transport. "
                "Install it with: pip install pyserial"
            ) from exc

        transport = serial.Serial()
        transport.port = self.port
        transport.baudrate = self.baudrate
        transport.timeout = self.timeout
        transport.write_timeout = self.write_timeout
        transport.dtr = self.dtr
        transport.rts = self.rts
        transport.open()
        transport.dtr = self.dtr
        transport.rts = self.rts

        self._serial = transport
        time.sleep(self.open_delay)
        self.flush()
        return self

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
        self._serial = None

    def __enter__(self) -> "JsonSerialTransport":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def flush(self) -> None:
        if not self.is_open:
            return
        assert self._serial is not None
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def send(self, command: Mapping[str, Any]) -> None:
        if not self.is_open:
            self.open()
        assert self._serial is not None
        payload = json.dumps(self._normalize_command(command), separators=(",", ":")).encode("utf-8") + b"\n"
        self._serial.write(payload)
        self._serial.flush()

    def read_json(self, timeout: float | None = None) -> dict[str, Any]:
        if not self.is_open:
            self.open()
        assert self._serial is not None

        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        last_line = ""
        while time.monotonic() < deadline:
            line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            last_line = line
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        raise TransportError(f"no JSON response received; last line: {last_line!r}")

    def request(
        self,
        command: Mapping[str, Any],
        *,
        response_timeout: float | None = None,
        expected_t: int | None = None,
        response_filter: Callable[[dict[str, Any]], bool] | None = None,
        response_description: str | None = None,
    ) -> dict[str, Any]:
        self.send(command)
        deadline = time.monotonic() + (self.timeout if response_timeout is None else response_timeout)
        last_response: dict[str, Any] | None = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._request_timeout_error(
                    expected_t,
                    response_filter,
                    response_description,
                    last_response,
                )

            try:
                response = self.read_json(timeout=remaining)
            except TransportError as exc:
                if last_response is None:
                    raise
                raise self._request_timeout_error(
                    expected_t,
                    response_filter,
                    response_description,
                    last_response,
                ) from exc

            if expected_t is not None and response.get("T") != expected_t:
                last_response = response
                continue
            if response_filter is not None and not response_filter(response):
                last_response = response
                continue
            return response

    @staticmethod
    def _normalize_command(command: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(command)
        if normalized.get("T") == int(CommandId.JOINTS_NAMED_RAD):
            for field_name in ("base", "shoulder", "elbow", "hand", "r"):
                if field_name in normalized:
                    normalized[field_name] = JsonSerialTransport._normalize_numeric(
                        normalized[field_name],
                        field_name,
                    )
            for field_name in ("spd", "acc"):
                if field_name in normalized:
                    normalized[field_name] = JsonSerialTransport._floor_int(
                        normalized[field_name],
                        field_name,
                    )
        if normalized.get("T") in {
            int(CommandId.EXT_GRIPPER_OPEN),
            int(CommandId.EXT_GRIPPER_CLOSE),
            int(CommandId.EXT_GRIPPER_DEG),
            int(CommandId.EXT_GRIPPER_HOLD_CLOSE),
        }:
            if "angle" in normalized:
                normalized["angle"] = JsonSerialTransport._normalize_numeric(
                    normalized["angle"],
                    "angle",
                )
            for field_name in ("spd", "acc", "torque", "hold"):
                if field_name in normalized:
                    normalized[field_name] = JsonSerialTransport._floor_int(
                        normalized[field_name],
                        field_name,
                    )
        return normalized

    @staticmethod
    def _normalize_numeric(value: Any, field_name: str) -> int | float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TransportError(f"{field_name} must be numeric, got {value!r}") from exc
        if not math.isfinite(numeric):
            raise TransportError(f"{field_name} must be finite, got {value!r}")
        if numeric.is_integer():
            return int(numeric)
        return numeric

    @staticmethod
    def _floor_int(value: Any, field_name: str) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TransportError(f"{field_name} must be numeric, got {value!r}") from exc
        if not math.isfinite(numeric):
            raise TransportError(f"{field_name} must be finite, got {value!r}")
        return math.floor(numeric)

    @staticmethod
    def _request_timeout_error(
        expected_t: int | None,
        response_filter: Callable[[dict[str, Any]], bool] | None,
        response_description: str | None,
        last_response: dict[str, Any] | None,
    ) -> TransportError:
        expected = []
        if expected_t is not None:
            expected.append(f"T={expected_t}")
        if response_filter is not None:
            expected.append(response_description or "a response accepted by response_filter")

        if expected:
            return TransportError(
                f"no matching JSON response received; expected {', '.join(expected)}; "
                f"last response: {last_response}"
            )
        return TransportError(f"no JSON response received; last response: {last_response}")


__all__ = [
    "DEFAULT_USB_BAUDRATE",
    "JsonSerialTransport",
    "TransportError",
]
