import os
import socket

import pytest

# Keras 3 needs a backend; the test env uses the CPU torch wheel.
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.pop("ESISC_SILENT", None)


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked(f"network access attempted during tests: {args!r}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test runs with outbound networking patched to fail loudly."""
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket.socket, "sendto", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
