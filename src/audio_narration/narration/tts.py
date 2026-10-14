# SPDX-License-Identifier: AGPL-3.0-only
import time

import numpy as np


class KokoroSpeech:
    def __init__(self, settings):
        from kokoro import KModel, KPipeline

        from audio_narration.config import KOKORO, cache_dir, select_device

        directory = cache_dir() / "kokoro"
        required = [
            directory / "config.json",
            directory / "kokoro-v1_0.pth",
            directory / "voices" / f"{settings.voice}.pt",
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError("Kokoro weights/voice missing. Run audio-narration-setup.")
        self.device = "cuda" if select_device(settings.device) == "cuda" else "cpu"
        model = KModel(repo_id=KOKORO, config=str(required[0]), model=str(required[1]))
        model.to(self.device).eval()
        self.pipeline = KPipeline(lang_code="a", repo_id=KOKORO, model=model, device=self.device)
        self.voice = str(required[2])
        self.settings = settings

    def warm_up(self) -> None:
        # Synthesize and discard one short chunk. Warm-up never produces narration.
        next(
            iter(
                self.pipeline(
                    "A person appears to be using a laptop.",
                    voice=self.voice,
                    speed=self.settings.speech_rate,
                )
            ),
            None,
        )

    def speak(self, text, cancel, first_audio, synthesis_metric, deadline=float("inf")):
        import sounddevice as sd

        start = time.monotonic()
        first = True
        chunks = self.pipeline(
            text, voice=self.voice, speed=self.settings.speech_rate, split_pattern=r"(?<=[.!?])\s+"
        )
        iterator = iter(chunks)
        while not cancel.is_set() and (not first or time.monotonic() <= deadline):
            synth_start = time.monotonic()
            try:
                _, _, audio = next(iterator)
            except StopIteration:
                break
            synthesis_metric(time.monotonic() - synth_start)
            if cancel.is_set() or (first and time.monotonic() > deadline):
                return
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            samples = np.asarray(audio, dtype=np.float32) * self.settings.volume
            with sd.OutputStream(
                samplerate=24000, channels=1, dtype="float32", blocksize=1024, latency="low"
            ) as stream:
                for offset in range(0, len(samples), 1024):
                    if cancel.is_set() or (first and time.monotonic() > deadline):
                        stream.abort()
                        return
                    stream.write(samples[offset : offset + 1024].reshape(-1, 1))
                    if first:
                        first_audio(time.monotonic() - start)
                        first = False


class SystemSpeech:
    """Explicit optional local OS voice; never a silent substitute for Kokoro."""

    def __init__(self, settings):
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", int(180 * settings.speech_rate))
        self.engine.setProperty("volume", settings.volume)

    def speak(self, text, cancel, first_audio, synthesis_metric, deadline=float("inf")):
        self.engine.say(text)
        self.engine.startLoop(False)
        try:
            while self.engine.isBusy() and not cancel.is_set() and time.monotonic() <= deadline:
                self.engine.iterate()
                cancel.wait(0.01)
            if cancel.is_set() or time.monotonic() > deadline:
                self.engine.stop()
        finally:
            self.engine.endLoop()
        # OS backend does not expose reliable audio-onset or synthesis timings.


def speech_factory(settings):
    return SystemSpeech(settings) if settings.tts_backend == "system" else KokoroSpeech(settings)
