"""macOS system sound via the free BlackHole virtual device; never opens a microphone.

macOS has no WASAPI-style speaker loopback. The user makes a Multi-Output
Device (speakers + BlackHole 2ch) in Audio MIDI Setup and selects it as the
sound output, so playback stays audible while BlackHole receives a copy. This
module records only input devices whose name contains "BlackHole". Same
interface as capture.py (Windows); audio is never stored.
"""
from __future__ import annotations

import hashlib
import queue
import threading
from typing import Callable

import numpy as np

SETUP_HELP = ('BlackHole 2ch 입력 장치를 찾지 못했습니다. BlackHole 2ch를 설치하고 '
              '"오디오 MIDI 설정"에서 스피커와 BlackHole 2ch를 묶은 다중 출력 장치를 만든 뒤, '
              '시스템 설정 → 사운드 → 출력에서 그 장치를 선택하고 새로고침을 눌러 주세요.')


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError('sounddevice 패키지가 없습니다. 앱 폴더에서 setup-mac.sh를 다시 실행해 주세요.') from exc
    return sounddevice


def _source_id(name: str) -> str:
    return "coreaudio:" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:24]


def _devices(sd) -> list[dict]:
    """BlackHole inputs only. Real microphones are never offered."""
    result = []
    for index, device in enumerate(sd.query_devices()):
        device = dict(device)
        if "blackhole" in str(device.get("name", "")).lower() and int(device.get("max_input_channels", 0)) > 0:
            device["index"] = index
            result.append(device)
    return result


def list_sources() -> list[dict]:
    devices = _devices(_sounddevice())
    counts: dict[str, int] = {}
    for d in devices:
        counts[_source_id(d["name"])] = counts.get(_source_id(d["name"]), 0) + 1
    # Identically named devices could swap after reconnect; do not offer them.
    result = [{
        "id": _source_id(d["name"]),
        "label": str(d["name"]) + " · 다중 출력 장치로 보낸 모든 소리",
        "scope": "system",
        "sample_rate": int(d["default_samplerate"]),
        "is_default": i == 0,
    } for i, d in enumerate(devices) if counts[_source_id(d["name"])] == 1]
    if not result:
        raise RuntimeError(SETUP_HELP)
    return result


class AudioCapture:
    """Deliver 20ms float32 mono blocks on a Python worker thread (as capture.py)."""

    def __init__(self, source_id: str,
                 on_audio: Callable[[np.ndarray, int], None],
                 on_error: Callable[[str], None]):
        self.source_id = source_id
        self.on_audio = on_audio
        self.on_error = on_error
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._setup_error: Exception | None = None
        self._packets: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._callback_error: str | None = None
        self._sd = None
        self.sample_rate = 0
        self.channels = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("이미 소리를 수집하고 있습니다.")
        self._stop.clear()
        self._ready.clear()
        self._setup_error = None
        self._callback_error = None
        self._packets = queue.Queue(maxsize=100)
        self._thread = threading.Thread(target=self._run, name="blackhole-capture", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("BlackHole 장치를 여는 시간이 초과됐습니다. 장치를 다시 선택해 주세요.")
        if self._setup_error:
            self.stop()
            raise RuntimeError(str(self._setup_error)) from self._setup_error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
            if thread.is_alive():
                raise RuntimeError("오디오 수집 종료가 지연되고 있습니다.")
        if not thread or not thread.is_alive():
            self._thread = None

    def _callback(self, indata, frames, time_info, status):
        if self._stop.is_set():
            raise self._sd.CallbackStop
        if status:
            self._callback_error = "오디오 입력이 끊겼습니다. 중지 후 다시 시작해 주세요."
            raise self._sd.CallbackAbort
        try:
            self._packets.put_nowait(np.array(indata, dtype=np.float32, copy=True))
        except queue.Full:
            self._callback_error = "오디오 처리가 밀려 수집을 중지했습니다. 다시 시작해 주세요."
            raise self._sd.CallbackAbort

    def _run(self) -> None:
        stream = None
        opened = False
        try:
            self._sd = _sounddevice()
            matches = [d for d in _devices(self._sd) if _source_id(d["name"]) == self.source_id]
            if len(matches) != 1:
                raise RuntimeError("선택한 BlackHole 장치를 찾을 수 없습니다. 목록에서 다시 선택해 주세요.")
            device = matches[0]
            self.sample_rate = int(device["default_samplerate"])
            self.channels = max(1, min(2, int(device["max_input_channels"])))
            stream = self._sd.InputStream(
                device=device["index"], channels=self.channels, samplerate=self.sample_rate,
                dtype="float32", blocksize=max(1, round(self.sample_rate * 0.020)),
                callback=self._callback,
            )
            stream.start()
            opened = True
            self._ready.set()
            while not self._stop.is_set():
                if self._callback_error:
                    raise RuntimeError(self._callback_error)
                try:
                    packet = self._packets.get(timeout=0.1)
                except queue.Empty:
                    if not stream.active:
                        raise RuntimeError("BlackHole 장치가 중지되거나 연결이 끊겼습니다. 다시 선택해 주세요.")
                    continue
                if self._stop.is_set():
                    break
                if packet.size % self.channels:
                    raise RuntimeError("오디오 프레임 형식이 맞지 않습니다.")
                mono = packet.reshape(-1, self.channels).mean(axis=1, dtype=np.float32)
                mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
                np.clip(mono, -1.0, 1.0, out=mono)
                self.on_audio(mono, self.sample_rate)
        except Exception as exc:
            if not opened:
                self._setup_error = exc
            elif not self._stop.is_set():
                try:
                    self.on_error(str(exc))
                except Exception:
                    pass
        finally:
            self._stop.set()
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self._ready.set()
