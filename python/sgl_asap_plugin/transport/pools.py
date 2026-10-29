"""Small resource pools for the A side."""

from __future__ import annotations

from typing import List

from sgl_asap_plugin.transport import streams as st


class EventPool:
    def __init__(self):
        self._free: List[int] = []

    def acquire(self) -> int:
        return self._free.pop() if self._free else st.event_create()

    def release(self, event: int) -> None:
        self._free.append(event)

    def close(self) -> None:
        for event in self._free:
            st.event_destroy(event)
        self._free.clear()
