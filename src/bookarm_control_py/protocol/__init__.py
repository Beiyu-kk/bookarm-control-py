"""Protocol helpers for ESP32 firmware."""

from bookarm_control_py.protocol.id_config import CommandId, Joint, ServoId

__all__ = [
    "CommandId",
    "Joint",
    "JsonSerialTransport",
    "ServoId",
    "TransportError",
]


def __getattr__(name: str):
    if name in {"JsonSerialTransport", "TransportError"}:
        from bookarm_control_py.protocol.esp32 import JsonSerialTransport, TransportError

        values = {
            "JsonSerialTransport": JsonSerialTransport,
            "TransportError": TransportError,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
