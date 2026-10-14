# Third-party notices

The repository's source is AGPL-3.0-only. Independently downloaded software, weights and voices
retain their own licences. This list is an engineering record, not a substitute for their
complete licence texts or legal advice.

| Component | Licence / authoritative source | Notes |
| --- | --- | --- |
| Ultralytics code and YOLO26 weights | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) | Public corresponding source; no Enterprise licence needed for this AGPL project. |
| ByteTrack original work | [MIT](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE) | The application uses the implementation inside Ultralytics, under its package terms. Credit Yifu Zhang and ByteTrack contributors. |
| SmolVLM2-256M-Video-Instruct weights | [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | Hugging Face Smol Models Research; pinned official revision. |
| Qwen3-VL-2B-Instruct weights | [Apache-2.0 model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Qwen team. Pin the official revision and preserve upstream licence/notice on redistribution. |
| Kokoro code | [Apache-2.0](https://github.com/hexgrad/kokoro/blob/main/LICENSE) | hexgrad and contributors. |
| Kokoro-82M v1 weights and `af_heart` pack | [Apache-2.0 model repository](https://huggingface.co/hexgrad/Kokoro-82M), [voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) | Voice file is fetched from the same licensed model repository. No voice files shipped here. |
| Kokoro training-data acknowledgements | [Model card attribution section](https://huggingface.co/hexgrad/Kokoro-82M#creative-commons-attribution) | Preserve the card's linked CC BY data acknowledgements when redistributing the model. Training-data terms and model-weight licence are distinct. |
| Transformers / Accelerate / Hugging Face Hub | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) | See each package's own licence and notices. |
| PyTorch / torchvision | [BSD-style](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Includes third-party notices for bundled native libraries. |
| OpenCV | [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE) | Python wheels may bundle additional libraries; retain their notices. |
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Retain copyright and disclaimer. |
| PySide6 / Qt | [LGPLv3/GPLv3/commercial options](https://doc.qt.io/qtforpython-6/licenses.html) | Uses unmodified dynamically loaded wheels. Keep relinking/replacement possible; binary redistribution requires Qt notices/source compliance. No proprietary Qt licence used. |
| sounddevice | [MIT](https://github.com/spatialaudio/python-sounddevice/blob/master/LICENSE) | PortAudio has its own permissive notice. |
| Misaki | [Apache-2.0](https://github.com/hexgrad/misaki) | English grapheme-to-phoneme processing. |
| eSpeak NG / loader fallback | [GPL-3.0-or-later](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING) | Dynamically loaded phoneme fallback; GPLv3 compatible with this AGPLv3 source project. Preserve upstream/source obligations if redistributed. |
| spaCy / English small model | [MIT](https://github.com/explosion/spacy-models) | Prepared during setup for English text processing. |
| psutil | [BSD-3-Clause](https://github.com/giampaolo/psutil/blob/master/LICENSE) | Local resource metrics. |
| pytest | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | Development only. |
| pyttsx3 (optional) | [MPL-2.0](https://github.com/nateshmbhat/pyttsx3) | OS voices are not distributed and retain their OS terms. |
| bitsandbytes (optional) | [MIT](https://github.com/bitsandbytes-foundation/bitsandbytes) | Optional CUDA path, not bundled. |

`docs/dependency-licenses.csv` records installed-package metadata for the validated Windows
environment. Missing metadata and upstream bundled libraries require further review for binary
redistribution. The Windows lock is a dependency snapshot, not a claim that every platform was
tested. This repository distributes application source and documentation only.
