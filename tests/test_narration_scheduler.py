# SPDX-License-Identifier: AGPL-3.0-only
from audio_narration.domain import Narration
from audio_narration.narration.scheduler import NarrationScheduler


def test_priority_interrupts_without_overlap():
    scheduler = NarrationScheduler(clock=lambda: 10)
    scheduler.submit(Narration("Summary", 10, 0, "summary"))
    item, token = scheduler.take()
    scheduler.submit(Narration("New action", 10, 2, "action"))
    assert token.is_set()
    scheduler.finished()
    item, new_token = scheduler.take()
    assert item.text == "New action" and not new_token.is_set()


def test_stale_and_latest_wins():
    now = [0]
    scheduler = NarrationScheduler(max_age=5, clock=lambda: now[0])
    scheduler.submit(Narration("old", 0, 1, "old"))
    scheduler.submit(Narration("new", 1, 1, "new"))
    assert scheduler.take()[0].text == "new"
    scheduler.finished()
    now[0] = 10
    assert not scheduler.submit(Narration("stale", 0))
    assert scheduler.take(0) is None


def test_flush_cancels_active_and_clears_queue():
    scheduler = NarrationScheduler(clock=lambda: 0)
    scheduler.submit(Narration("one", 0))
    _, token = scheduler.take()
    scheduler.submit(Narration("two", 0))
    scheduler.flush()
    assert token.is_set()
    assert scheduler.take(0) is None
    scheduler.close()
    assert not scheduler.submit(Narration("three", 0))


def test_bounded_priority_queue():
    scheduler = NarrationScheduler(capacity=2, clock=lambda: 0)
    for priority in range(10, 0, -1):
        scheduler.submit(Narration(str(priority), 0, priority, str(priority)))
    assert len(scheduler.pending) == 2
    assert scheduler.take()[0].priority == 10
