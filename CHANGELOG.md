# Changelog

## 0.2.0

- Add double-click Windows install/run launchers and automatic NVIDIA CUDA setup/verification.
- Default to SmolVLM2 fast captions with structured linguistic observations; retain Qwen selection.
- Accelerate Kokoro on CUDA, warm speech before requests, and serialize model initialization.
- Use one-second spacing, two-second refresh and five-second stale-result expiry/cancellation.
- Persist default model caches across downloaded versions and check installation disk space.
- Record a real GPU replay check at 2.109 seconds to first audio write and 41 passing core tests.
- Preserve limitations: moving-action accuracy and worst-case latency are not certified.
