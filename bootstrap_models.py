"""Download public, pinned local models and register the GGUF with Ollama.

No API key required. Only public model files are downloaded; no audio is sent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
QWEN_REPO = "Qwen/Qwen3-8B-GGUF"
QWEN_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
QWEN_FILE = "Qwen3-8B-Q4_K_M.gguf"
WHISPER_REPO = "dropbox-dash/faster-whisper-large-v3-turbo"
WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
OLLAMA_NAME = "local-korean:latest"
TRANSLATOR_NAME = "qwen3:14b"
TRANSLATOR_DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
BIG_FILES = {
    QWEN_FILE: (5027783488, "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"),
    "model.bin": (1617884929, "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da"),
}


def parallel_download(repo: str, revision: str, name: str, folder: Path) -> Path:
    """Bounded range requests avoid long-connection interruptions on this PC."""
    size, checksum = BIG_FILES[name]
    target = folder / name
    folder.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == checksum:
                print(f"Verified existing {name}", flush=True)
                return target
        raise RuntimeError(f"Existing {name} has wrong checksum; kept for inspection")
    temporary = folder / (name + ".download")
    state_path = folder / (name + ".ranges.json")
    chunk_size = 8 * 1024 * 1024
    count = (size + chunk_size - 1) // chunk_size
    if temporary.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        ranges = state.get("done")
        if (state.get("sha256") != checksum or temporary.stat().st_size != size
                or not isinstance(ranges, list)
                or any(type(n) is not int or not 0 <= n < count for n in ranges)):
            raise RuntimeError(f"Invalid resume state for {name}; files retained for inspection")
        done = set(ranges)
    else:
        candidates = list(folder.rglob("*" + checksum + "*.incomplete"))
        seed = max(candidates, key=lambda p: p.stat().st_size, default=None)
        if seed:
            shutil.copyfile(seed, temporary)
            done = set(range(seed.stat().st_size // chunk_size))
        else:
            temporary.touch()
            done = set()
        with temporary.open("r+b") as f:
            f.truncate(size)
    lock = threading.Lock()
    last_progress = [0.0]
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"

    def fetch(index: int) -> int:
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        for attempt in range(6):
            try:
                # Distinct resolver URLs avoid stale CDN signed-range redirects.
                request = urllib.request.Request(url + f"?download=true&chunk={index}&attempt={attempt}", headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=35) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Unexpected range response")
                    payload = response.read()
                if len(payload) != end - start + 1:
                    raise RuntimeError("Incomplete range response")
                with lock:
                    with temporary.open("r+b") as output:
                        output.seek(start)
                        output.write(payload)
                    done.add(index)
                    state_path.write_text(json.dumps({"done": sorted(done), "sha256": checksum}), encoding="utf-8")
                    now = time.monotonic()
                    if now - last_progress[0] > 15 or len(done) == count:
                        print(f"{name}: {len(done)}/{count} chunks ({len(done)/count:.0%})", flush=True)
                        last_progress[0] = now
                return index
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    print(f"Parallel download {name}, resuming {len(done)}/{count} chunks", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, n) for n in range(count) if n not in done]
        for future in as_completed(futures):
            future.result()
    with temporary.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != checksum:
        raise RuntimeError(f"Checksum mismatch for {name}; partial data retained")
    temporary.replace(target)
    print(f"SHA256 verified: {name}", flush=True)
    return target


def download(repo: str, revision: str, name: str, folder: Path) -> Path:
    if name in BIG_FILES:
        return parallel_download(repo, revision, name, folder)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from huggingface_hub import hf_hub_download
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo}/{name}", flush=True)
    result = hf_hub_download(repo, name, revision=revision, local_dir=folder, token=False)
    path = Path(result)
    print(f"Ready {path.name}: {path.stat().st_size:,} bytes", flush=True)
    return path


def whisper() -> Path:
    folder = MODELS / "whisper-turbo"
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json", "model.bin"):
        download(WHISPER_REPO, WHISPER_REVISION, name, folder)
    return folder


def qwen8() -> Path:
    path = download(QWEN_REPO, QWEN_REVISION, QWEN_FILE, MODELS / "qwen")
    modelfile = MODELS / "qwen" / "Modelfile"
    modelfile.write_text(
        f'FROM "{path.as_posix()}"\n'
        'PARAMETER num_ctx 4096\n'
        'PARAMETER temperature 0.1\n'
        'PARAMETER num_predict 256\n', encoding="utf-8")
    creation = subprocess.run(
        ["ollama", "create", OLLAMA_NAME, "-f", str(modelfile)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if creation.returncode:
        raise RuntimeError("Ollama model import failed: " + creation.stdout.decode("utf-8", errors="replace")[-1200:])
    print(f"Registered Ollama model: {OLLAMA_NAME}", flush=True)
    return path


def qwen() -> str:
    """Fetch the tested 14B upgrade; retain the previous 8B model unchanged."""
    from engine import local_request
    tags = local_request('/api/tags', timeout=5)
    installed = next((m for m in tags.get('models', [])
                      if m.get('name') == TRANSLATOR_NAME), None)
    if installed is None:
        subprocess.run(['ollama', 'pull', TRANSLATOR_NAME], check=True)
        tags = local_request('/api/tags', timeout=5)
        installed = next((m for m in tags.get('models', [])
                          if m.get('name') == TRANSLATOR_NAME), None)
    if not installed or installed.get('digest') != TRANSLATOR_DIGEST:
        raise RuntimeError('Translation model digest differs from the tested model. '
                           'Files were retained; review model-lock.json before updating.')
    print(f'Verified Ollama model: {TRANSLATOR_NAME}', flush=True)
    return TRANSLATOR_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["whisper", "qwen", "qwen8", "all"], default="all")
    args = parser.parse_args()
    jobs = {"whisper": whisper, "qwen": qwen}
    selected = jobs if args.only == "all" else {args.only: qwen8 if args.only == 'qwen8' else jobs[args.only]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {name: pool.submit(fn) for name, fn in selected.items()}
        while any(not future.done() for future in pending.values()):
            time.sleep(15)
            print("Model setup active: " + ", ".join(name for name,f in pending.items() if not f.done()), flush=True)
        result = {name: str(future.result()) for name, future in pending.items()}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
