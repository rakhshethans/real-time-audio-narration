# SPDX-License-Identifier: AGPL-3.0-only
import re
from collections import deque
from difflib import SequenceMatcher

from audio_narration.domain import Narration, Observation


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def similar(left: str, right: str) -> bool:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio() >= 0.84


class TemporalMemory:
    """Bounded structured history; facts/actions checked independently of narration prose."""

    def __init__(self, capacity=20, summary_interval=60.0):
        self.observations: deque[Observation] = deque(maxlen=capacity)
        self.spoken_candidates: deque[tuple[float, str]] = deque(maxlen=40)
        self.last_summary = float("-inf")
        self.summary_interval = summary_interval

    @property
    def latest(self):
        return self.observations[-1] if self.observations else None

    def update(self, observation: Observation) -> Narration | None:
        previous = self.latest
        self.observations.append(observation)
        now = observation.timestamp
        if previous is None:
            text, priority, summary = observation.description, 1, True
        else:
            new_actions = [
                action
                for action in observation.actions
                if not any(similar(action, old) for old in previous.actions)
            ]
            new_facts = [
                fact
                for fact in observation.facts
                if not any(similar(fact, old) for old in previous.facts)
            ]
            changes = [
                change
                for change in observation.changes
                if not any(
                    now - when < 45 and similar(change, said)
                    for when, said in self.spoken_candidates
                )
            ]
            if changes or new_actions:
                text = " ".join((changes or new_actions)[:2])
                priority, summary = 2, False
            elif new_facts:
                text, priority, summary = observation.description, 1, False
            elif now - self.last_summary >= self.summary_interval:
                text, priority, summary = observation.description, 0, True
            else:
                return None
        if not summary and any(
            now - when < 45 and similar(text, said) for when, said in self.spoken_candidates
        ):
            return None
        if observation.uncertain and not re.search(r"appears|may|might|seems", text, re.I):
            text = "It appears that " + text[0].lower() + text[1:]
        if summary:
            self.last_summary = now
        self.spoken_candidates.append((now, text))
        return Narration(text, now, priority, normalize(text), summary)
