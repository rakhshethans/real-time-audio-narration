# Validation record

Checks were performed on Windows x64 with Python 3.12.14, PyTorch 2.8.0 CPU and approximately
31.3 GiB system RAM. CUDA was unavailable. Results below are observations from this build,
not promises for other hardware. Read the unverified section before judging project readiness.

## Automated and installation checks

* All pinned direct package versions were checked against the official PyPI registry.
* The complete application, development and optional system-voice dependencies installed in an
  isolated virtual environment. `pip check` reported no broken requirements.
* Real imports succeeded for OpenCV, PySide6, YOLO, Qwen3-VL, Kokoro and their inference classes.
* The official Qwen and Kokoro revisions downloaded successfully, as did YOLO26 nano and the
  spaCy English model. A repeated setup with `HF_HUB_OFFLINE=1` reused local files successfully.
* A Windows privilege error in Hugging Face symlink caching was discovered and fixed by using
  plain local directories. TLS verification was retained; a validation-only system certificate
  bundle was needed for this machine's network. It is not committed or required by the app.
* **35 pytest tests passed** after the final implementation, covering parsing, tracking signals,
  state changes, bounded memory/queues, stale speech, cancellation, failures and shutdown.
  These tests use mocked camera/models/audio where appropriate. They do not measure recognition
  accuracy. Ruff static checks passed. The [GitHub CI run for source commit 941584b](https://github.com/rakhshethans/real-time-audio-narration/actions/runs/38317193328)
  also passed all four core-test jobs: Windows/Linux with Python 3.11/3.12. These jobs do not
  install or exercise the real models, webcam, native UI or audio devices.
* The actual Qt window was rendered offscreen and visually inspected. The committed screenshot
  is the real idle UI, not a mockup. The offscreen renderer needed an explicitly loaded system
  font; no Windows font file is redistributed. Screen-reader usability was not tested.

## Real hardware and model smoke checks

| Check | Measured result | Scope |
| --- | --- | --- |
| Webcam 0 | 130 captured frames; reported running average about 14.6 FPS | Capture-only run; startup included in elapsed window; no footage saved; stopped cleanly |
| YOLO26 nano + ByteTrack | First inference 578 ms, then 47 ms and 62 ms | Three identical public bus-image inputs; detected a bus and people with IDs 1–5 |
| Qwen3-VL CPU load | 7.5 seconds | Official local weights; float32, 8 Torch threads in the test harness |
| Qwen scene interpretation | 179.1 seconds | Two identical 336-pixel public still frames, original prompt; valid structured response |
| Kokoro model initialization | 19.2 seconds | First local pipeline/English initialization |
| Kokoro first sentence synthesis | 1.046 seconds | One short test utterance; excludes model initialization |
| Kokoro first audio write | 1.156 seconds | Sent to the default real output device; physical speaker onset/audibility not independently confirmed |

YOLO's later rerun also assigned IDs successfully (first inference 1.703 s, then 125 ms and
109 ms); changing memory pressure makes these timings unsuitable as steady-state benchmarks.
No action-labelled video comparison was available, so no measured baseline-versus-hybrid
benchmark or event-recognition score is claimed. The gate test's 12-versus-4 invocation counts
are explicitly a scheduling simulation on an unchanged scene, not inference measurements.

## Findings that affected implementation

The first Qwen response claimed walking from identical frames and included uncertain scene
details. The integration worked, but the response did **not** establish truthful activity
understanding. This led to stricter prompts, a coarse static-clip motion-evidence check, removal
of common unsupported motion sentences/actions from unchanged clips, forced uncertainty for
those clips, and rejection when filtering leaves no usable description. This is a heuristic
guard, not a guarantee against hallucination. Camera motion may pass the evidence gate; subtle
actions may fail it. Some unsupported details can still survive.

CPU interpretation was far too slow for frequent narration at the original resolution.
The final CPU profile uses at most two frames with maximum side 224 pixels and a more compact
structured prompt. Automatic expiry is 180 seconds on CPU versus 30 seconds on accelerated
hardware; an explicit `--max-age` overrides it. The larger CPU allowance trades freshness for
the possibility of hearing a result and must not be mistaken for real-time scene tracking.
For strict freshness, use `--max-age 30`; slow responses will then be discarded. A preflight
check gives an actionable error when less than 10 GiB RAM is available for float32 CPU loading.

A rerun after prompt/resolution refinement failed during model loading with Windows error 1455
(paging file too small). An aggregate resource check showed approximately 97% RAM usage.
The user requested finishing with verification limits documented rather than freeing memory.
The final static-clip filter is covered by regression tests, but **the final prompt and reduced
CPU profile were not rerun successfully against Qwen**. Earlier model inference and model API
compatibility were verified; this distinction is intentional.

## Not verified

* Continuous live webcam → Qwen → scheduler → Kokoro end-to-end speech. Capture, real model
  adapters, speech output, and orchestration tests succeeded separately.
* Bottle pickup/drinking, sit/stand, entry/exit and other activities with annotated moving clips.
* Final reduced CPU inference latency and accuracy, CUDA/MPS acceleration, 4-bit quantization,
  macOS/Linux native UI and audio, and the optional pyttsx3 backend.
* A sustained ten-minute run, production screen-reader accessibility, GPU utilization, action
  precision/recall, or ground-truth event-to-speech latency.

The [manual protocol](MANUAL_TESTING.md) and real-video benchmark command in the README provide
the next verification steps. SmolVLM2 and ONNX adapters are deliberately deferred. The code
is implemented, but field readiness and trustworthy activity understanding require evaluation.
