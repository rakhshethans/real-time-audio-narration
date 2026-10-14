# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request

from audio_narration.config import (
    KOKORO,
    KOKORO_REVISION,
    QWEN,
    QWEN_REVISION,
    SMOL,
    SMOL_REVISION,
    cache_dir,
)


def main():
    parser = argparse.ArgumentParser(description="Download official local model weights once")
    parser.add_argument(
        "--only", choices=["all", "yolo", "vlm", "smol", "tts", "fast"], default="fast"
    )
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    directory = cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.only in {"all", "vlm"}:
        print("Downloading Qwen3-VL (several GB); existing cached files are reused.")
        snapshot_download(
            QWEN,
            revision=QWEN_REVISION,
            local_dir=str(directory / "qwen"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
        manifest[QWEN] = QWEN_REVISION
    if args.only in {"all", "vlm", "smol", "fast"}:
        print("Downloading the lightweight SmolVLM2 fast scene model")
        snapshot_download(
            SMOL,
            revision=SMOL_REVISION,
            local_dir=str(directory / "smol"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
        )
        manifest[SMOL] = SMOL_REVISION
    if args.only in {"all", "yolo", "fast"}:
        path = directory / "yolo26n.pt"
        if not path.exists():
            url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
            temporary = path.with_suffix(".download")
            print("Downloading official YOLO26 nano weights")
            try:
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        manifest["yolo26n.pt"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.only in {"all", "tts", "fast"}:
        for filename in ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"):
            hf_hub_download(
                KOKORO,
                filename,
                revision=KOKORO_REVISION,
                local_dir=str(directory / "kokoro"),
            )
        # Misaki's English pipeline otherwise tries to install this at first narration.
        import spacy

        if not spacy.util.is_package("en_core_web_sm"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "https://github.com/explosion/spacy-models/releases/download/"
                    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
                ],
                check=True,
            )
        manifest[KOKORO] = KOKORO_REVISION
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Models cached in {directory}. Runtime inference uses local files.")


if __name__ == "__main__":
    main()
