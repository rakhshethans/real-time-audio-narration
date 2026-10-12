# SPDX-License-Identifier: AGPL-3.0-only
from audio_narration.domain import Observation
from audio_narration.memory.temporal_memory import TemporalMemory


def observation(t, actions=(), changes=(), uncertain=False):
    return Observation(
        t, "A person is sitting at a table.", ("A table is visible.",), actions, changes, uncertain
    )


def test_static_scene_does_not_repeat_but_summary_due():
    memory = TemporalMemory(summary_interval=60)
    assert memory.update(observation(0))
    assert memory.update(observation(5)) is None
    assert memory.update(observation(59)) is None
    assert memory.update(observation(60)).summary


def test_new_action_does_not_repeat_table():
    memory = TemporalMemory()
    memory.update(observation(0))
    item = memory.update(observation(5, ("The person picks up a cup.",)))
    assert "cup" in item.text and "table" not in item.text
    assert item.priority == 2
    assert memory.update(observation(10, ("The person picks up a cup.",))) is None


def test_duplicate_changes_are_suppressed():
    memory = TemporalMemory()
    memory.update(observation(0))
    assert memory.update(observation(5, changes=("The person stands up.",)))
    assert memory.update(observation(10, changes=("The person stands up.",))) is None


def test_uncertainty_and_bounded_memory():
    memory = TemporalMemory(capacity=3)
    assert memory.update(observation(0, uncertain=True)).text.startswith("It appears")
    for t in range(1, 100):
        memory.update(observation(t))
    assert len(memory.observations) == 3
    assert len(memory.spoken_candidates) <= 40


def test_alternate_scene_fact_replaces_previous_in_context():
    memory = TemporalMemory()
    memory.update(observation(0))
    latest = Observation(5, "The desk is empty.", ("The desk is empty.",))
    memory.update(latest)
    assert memory.latest is latest
