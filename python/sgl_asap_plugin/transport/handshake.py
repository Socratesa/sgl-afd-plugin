"""Rendezvous between an A rank and an F rank: one JSON request/response over TCP.

The handshake validates that both sides agree on the geometry and protocol before any data
moves (a mismatch fails here, not at the first layer), and exchanges the CUDA IPC handles.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time

from sgl_asap_plugin.transport.protocol import HandshakeError, ProtocolError

MAX_MSG = 1 << 20


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("<I", len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict:
    (n,) = struct.unpack("<I", _recv_exact(sock, 4))
    if n > MAX_MSG:
        raise ProtocolError(f"handshake message of {n} bytes exceeds {MAX_MSG}")
    return json.loads(_recv_exact(sock, n))


class HandshakeServer:
    """Accepts A ranks one at a time; ``on_hello(hello) -> reply`` raises to reject."""

    def __init__(self, host: str, port: int, on_hello):
        self._on_hello = on_hello
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(64)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]  # the real port when 0 was requested
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="asap-handshake", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                conn.settimeout(10.0)
                try:
                    reply = self._on_hello(recv_msg(conn))
                except Exception as exc:  # reject with the reason; never kill the server thread
                    reply = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
                try:
                    send_msg(conn, reply)
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._srv.close()


def request(host: str, port: int, hello: dict, *, connect_timeout: float = 60.0, io_timeout: float = 30.0) -> dict:
    """Send ``hello`` to an F rank (retrying until it listens) and return its accept message."""
    deadline = time.monotonic() + connect_timeout
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            break
        except OSError as exc:
            if time.monotonic() > deadline:
                raise HandshakeError(f"cannot reach F at {host}:{port}: {exc}") from exc
            time.sleep(0.1)
    with sock:
        sock.settimeout(io_timeout)
        try:
            send_msg(sock, hello)
            reply = recv_msg(sock)
        except (OSError, ConnectionError) as exc:
            raise HandshakeError(f"handshake with F at {host}:{port} failed: {exc}") from exc
    if not reply.get("ok"):
        raise HandshakeError(f"F at {host}:{port} rejected the connection: {reply.get('reason')}")
    return reply
