# SPDX-License-Identifier: AGPL-3.0-only
import time
from threading import Condition, Event

from audio_narration.domain import Narration


class NarrationScheduler:
    """Bounded pending speech, latest wins within priority; one audio consumer."""

    def __init__(self, max_age=30.0, capacity=3, clock=time.monotonic):
        self.max_age, self.capacity, self.clock = max_age, capacity, clock
        self.condition = Condition()
        self.pending: list[Narration] = []
        self.active: Narration | None = None
        self.cancel = Event()
        self.closed = False

    def submit(self, item: Narration) -> bool:
        with self.condition:
            if self.closed or self.clock() - item.timestamp > self.max_age:
                return False
            if self.active and self.active.key == item.key and not item.summary:
                return False
            if self.active and item.priority > self.active.priority:
                self.cancel.set()
            # Replace outdated pending descriptions of equal/lower priority.
            self.pending = [
                p for p in self.pending if p.priority > item.priority and p.key != item.key
            ]
            self.pending.append(item)
            self.pending.sort(key=lambda p: (p.priority, p.timestamp), reverse=True)
            self.pending = self.pending[: self.capacity]
            self.condition.notify_all()
            return True

    def take(self, timeout: float = 0.2) -> tuple[Narration, Event] | None:
        with self.condition:
            if not self.pending and not self.closed:
                self.condition.wait(timeout)
            self.pending = [p for p in self.pending if self.clock() - p.timestamp <= self.max_age]
            if self.closed or not self.pending:
                return None
            self.active = self.pending.pop(0)
            self.cancel = Event()
            return self.active, self.cancel

    def finished(self) -> None:
        with self.condition:
            self.active = None

    def flush(self) -> None:
        with self.condition:
            self.pending.clear()
            self.cancel.set()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.cancel.set()
            self.condition.notify_all()
