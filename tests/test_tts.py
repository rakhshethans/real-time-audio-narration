# SPDX-License-Identifier: AGPL-3.0-only
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np

from audio_narration.config import Settings
from audio_narration.narration.tts import KokoroSpeech


def test_cancelled_synthesis_never_plays(monkeypatch):
    cancel = Event()
    outputs = []
    speaker = KokoroSpeech.__new__(KokoroSpeech)
    speaker.settings, speaker.voice = Settings(), "test"

    def synthesize(*args, **kwargs):
        cancel.set()
        yield "text", "phonemes", np.zeros(24000)

    speaker.pipeline = synthesize
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=lambda **kwargs: outputs.append(kwargs)),
    )
    speaker.speak("A test.", cancel, lambda t: None, lambda t: None)
    assert outputs == []


def test_playback_checks_cancel_and_aborts(monkeypatch):
    cancel, writes, first = Event(), [], []
    speaker = KokoroSpeech.__new__(KokoroSpeech)
    speaker.settings, speaker.voice = Settings(), "test"
    speaker.pipeline = lambda *a, **kw: iter([("text", "phonemes", np.ones(5000))])

    class Stream:
        aborted = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, samples):
            writes.append(samples)
            cancel.set()

        def abort(self):
            self.aborted = True

    stream = Stream()
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kwargs: stream)
    )
    speaker.speak("A test.", cancel, first.append, lambda t: None)
    assert len(writes) == 1 and stream.aborted and len(first) == 1
    assert np.allclose(writes[0], 0.8)


def test_expired_description_does_not_synthesize_or_play(monkeypatch):
    calls = []
    speaker = KokoroSpeech.__new__(KokoroSpeech)
    speaker.settings, speaker.voice = Settings(), "test"
    speaker.pipeline = lambda *a, **kw: iter(())
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=lambda **kwargs: calls.append(kwargs)),
    )
    speaker.speak("Expired scene.", Event(), lambda t: calls.append(t), lambda t: None, deadline=-1)
    assert calls == []


def test_started_sentence_finishes_after_first_audio_deadline(monkeypatch):
    from audio_narration.narration import tts

    clock, writes = [0.0], []
    monkeypatch.setattr(tts.time, "monotonic", lambda: clock[0])
    speaker = KokoroSpeech.__new__(KokoroSpeech)
    speaker.settings, speaker.voice = Settings(), "test"
    speaker.pipeline = lambda *a, **kw: iter([("text", "phonemes", np.ones(3000))])

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, samples):
            writes.append(samples)
            clock[0] += 0.1 if len(writes) == 1 else 2.0

        def abort(self):
            raise AssertionError("A started sentence should finish unless cancelled")

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(OutputStream=lambda **kwargs: Stream())
    )
    speaker.speak("A short sentence.", Event(), lambda t: None, lambda t: None, deadline=1)
    assert len(writes) == 3
