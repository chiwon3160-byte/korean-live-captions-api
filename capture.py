"""Windows speaker loopback only; never opens a microphone or stores audio.

Each source represents the complete mix playing through ONE output device,
not an individual browser tab/process. Source IDs follow device names across
PortAudio index changes; a missing/ambiguous device fails instead of swapping.
"""
from __future__ import annotations

import hashlib
import queue
import threading
from typing import Callable

import numpy as np
import pyaudiowpatch as pa


def _source_id(device: dict) -> str:
    name = str(device["name"])
    return "wasapi:" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]


def _devices(audio: pa.PyAudio) -> list[dict]:
    return [dict(d) for d in audio.get_loopback_device_info_generator()
            if d.get("isLoopbackDevice") and int(d.get("maxInputChannels", 0)) > 0]


def list_sources() -> list[dict]:
    """Return speaker loopbacks, default output first; no mic fallbacks."""
    audio = pa.PyAudio()
    try:
        try:
            default_index = int(audio.get_default_wasapi_loopback()["index"])
        except (OSError, LookupError):
            default_index = -1
        devices = _devices(audio)
        counts: dict[str, int] = {}
        for d in devices:
            counts[_source_id(d)] = counts.get(_source_id(d), 0) + 1
        result = []
        for d in devices:
            sid = _source_id(d)
            # Selecting between identically named devices by volatile index can
            # silently capture the wrong source after reconnect; do not offer it.
            if counts[sid] != 1:
                continue
            default = int(d["index"]) == default_index
            name = str(d["name"]).removesuffix(" [Loopback]")
            result.append({
                "id": sid,
                "label": ("기본 출력 · " if default else "") + name + " · 이 장치의 모든 소리",
                "scope": "system",
                "sample_rate": int(d["defaultSampleRate"]),
                "is_default": default,
            })
        return sorted(result, key=lambda d: not d["is_default"])
    finally:
        audio.terminate()


class AudioCapture:
    """Deliver 20ms float32 mono blocks on a Python worker thread.

    on_audio should enqueue work and return quickly. Any queue overflow,
    stream failure, or consumer exception stops capture and calls on_error.
    start raises an exception on setup failure (it does not also call on_error).
    """

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
        self._packets: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self._callback_error: str | None = None
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
        self._thread = threading.Thread(target=self._run, name="speaker-loopback", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("출력 장치를 여는 시간이 초과됐습니다. 장치를 다시 선택해 주세요.")
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

    def _callback(self, in_data, frame_count, time_info, status):
        if self._stop.is_set():
            return None, pa.paComplete
        if status:
            self._callback_error = "오디오 입력이 끊겼습니다. 중지 후 다시 시작해 주세요."
            return None, pa.paAbort
        try:
            self._packets.put_nowait(in_data)
        except queue.Full:
            self._callback_error = "오디오 처리가 밀려 수집을 중지했습니다. 다시 시작해 주세요."
            return None, pa.paAbort
        return None, pa.paContinue

    def _run(self) -> None:
        audio = None
        stream = None
        opened = False
        try:
            audio = pa.PyAudio()
            matches = [d for d in _devices(audio) if _source_id(d) == self.source_id]
            if len(matches) != 1:
                raise RuntimeError("선택한 출력 장치를 찾을 수 없습니다. 목록에서 다시 선택해 주세요.")
            device = matches[0]
            self.sample_rate = int(device["defaultSampleRate"])
            self.channels = int(device["maxInputChannels"])
            # paFloat32 preserves original dynamic range before mono downmix.
            stream = audio.open(
                format=pa.paFloat32, channels=self.channels, rate=self.sample_rate,
                input=True, input_device_index=int(device["index"]),
                frames_per_buffer=max(1, round(self.sample_rate * 0.020)),
                stream_callback=self._callback, start=False,
            )
            stream.start_stream()
            opened = True
            self._ready.set()
            while not self._stop.is_set():
                if self._callback_error:
                    raise RuntimeError(self._callback_error)
                try:
                    packet = self._packets.get(timeout=0.1)
                except queue.Empty:
                    if not stream.is_active():
                        raise RuntimeError("출력 장치가 중지되거나 연결이 끊겼습니다. 다시 선택해 주세요.")
                    continue
                if self._stop.is_set():
                    break
                samples = np.frombuffer(packet, dtype=np.float32)
                if samples.size % self.channels:
                    raise RuntimeError("오디오 프레임 형식이 맞지 않습니다.")
                mono = samples.reshape(-1, self.channels).mean(axis=1, dtype=np.float32)
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
                    if stream.is_active():
                        stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            if audio is not None:
                audio.terminate()
            self._ready.set()
