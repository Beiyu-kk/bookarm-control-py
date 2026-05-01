"""ESP32 newline-delimited JSON transport.

This module only handles sending and receiving JSON dictionaries. Conversion
from high-level robot commands to ESP32 JSON lives in ``bookarm_control_py.actuator``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import time
from typing import Any

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
        payload = json.dumps(dict(command), separators=(",", ":")).encode("utf-8") + b"\n"
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
    ) -> dict[str, Any]:
        self.send(command)
        deadline = time.monotonic() + (self.timeout if response_timeout is None else response_timeout)
        last_response: dict[str, Any] | None = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if expected_t is None:
                    raise TransportError("no JSON response received")
                raise TransportError(f"expected T={expected_t}, got {last_response}")

            response = self.read_json(timeout=remaining)
            if expected_t is None or response.get("T") == expected_t:
                return response
            last_response = response


__all__ = [
    "DEFAULT_USB_BAUDRATE",
    "JsonSerialTransport",
    "TransportError",
]
