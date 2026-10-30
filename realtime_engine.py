"""Opt-in live translation transport; no audio, transcript, or credentials logged.

Protocol checked 2026-10-30:
https://developers.openai.com/api/docs/guides/realtime-translation
https://developers.openai.com/api/reference/resources/realtime/translation-server-events

The dedicated translation service continuously returns audio AND text. We discard
output audio locally; this is not a cheaper text-only service. No optional source
transcription is requested. The service exposes no per-event billing usage, so
the callback reports SENT AUDIO DURATION ESTIMATES, never billed token usage.

start(on_caption=dict callback, on_error=str callback, on_ready=no-arg callback,
      on_usage=optional provider/model/usage/success callback) is nonblocking.
feed(float32 mono, sample_rate) queues audio; stop() is nonblocking and suppresses
late captions. wait_closed() is only for non-GUI cleanup/tests. reset() creates a
new explicit session; network failures never retry. All callbacks run on worker
threads and must be marshalled to a GUI thread by the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import base64
import itertools
import json
import math
import queue
import ssl
import threading
import time

import numpy as np
from scipy.signal import resample_poly

MODEL = 'gpt-realtime-translate'
PROVIDER = 'openai-realtime'
ENDPOINT = 'wss://api.openai.com/v1/realtime/translations?model=' + MODEL
SAMPLE_RATE = 24000
FRAME_SAMPLES = 4800  # Service consumes 200 ms engine frames.
MAX_PENDING_BLOCKS = 100  # About two seconds of normal 20 ms capture packets.
_IDS = itertools.count(1)


@dataclass
class _Stream:
    on_caption: object
    on_error: object
    on_ready: object
    on_usage: object
    generation: int = field(default_factory=lambda: next(_IDS))
    stop: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    packets: queue.Queue = field(default_factory=lambda: queue.Queue(MAX_PENDING_BLOCKS))
    thread: object = None
    ready: bool = False
    failed: bool = False
    rate: int | None = None
    pending: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    caption: str = ''
    caption_id: int = field(default_factory=lambda: next(_IDS))
    seen_events: set = field(default_factory=set)
    event_order: deque = field(default_factory=deque)


class RealtimeEngine:
    def __init__(self, status=lambda _: None, *, provider=PROVIDER, model=MODEL,
                 api_key, connection_factory=None):
        if provider != PROVIDER or model != MODEL:
            raise ValueError('지원하지 않는 실시간 번역 모델입니다.')
        if (not isinstance(api_key, str) or not api_key.strip() or
                len(api_key) > 4096 or any(c.isspace() for c in api_key)):
            raise ValueError('API 키를 입력해 주세요.')
        self.provider, self.model = provider, model
        self.reasoning_effort = ''
        self.status = status
        self.usage_callback = None
        self._api_key = api_key
        self._connection_factory = connection_factory
        self._stream = None
        self._streams = []
        self._lock = threading.RLock()

    @property
    def ready(self):
        stream = self._stream
        return bool(stream and stream.ready and not stream.stop.is_set())

    def start(self, *, on_caption, on_error, on_ready, on_usage=None):
        with self._lock:
            self.stop()
            stream = _Stream(on_caption, on_error, on_ready,
                             on_usage if on_usage is not None else self.usage_callback)
            self._stream = stream
            self._streams = [old for old in self._streams if not old.closed.is_set()]
            self._streams.append(stream)
            stream.thread = threading.Thread(target=self._run, args=(stream,),
                                             name='realtime-translation', daemon=True)
            stream.thread.start()

    def stop(self):
        """Stop accepting/sending audio now; drain remote close in background."""
        with self._lock:
            if self._stream:
                self._stream.stop.set()

    def wait_closed(self, timeout=3.5):
        deadline = time.monotonic() + max(0, timeout)
        for stream in list(self._streams):
            if stream.thread is threading.current_thread():
                continue
            stream.closed.wait(max(0, deadline - time.monotonic()))
        return all(stream.closed.is_set() for stream in self._streams)

    def reset(self):
        with self._lock:
            stream = self._stream
            if stream and not stream.stop.is_set():
                self.start(on_caption=stream.on_caption, on_error=stream.on_error,
                           on_ready=stream.on_ready, on_usage=stream.on_usage)

    def feed(self, audio, rate):
        stream = self._stream
        if not stream or not stream.ready or stream.stop.is_set():
            return
        try:
            rate = int(rate)
            samples = np.asarray(audio, dtype=np.float32)
            if rate < 8000 or rate > 192000 or samples.ndim != 1 or samples.size > rate:
                raise ValueError('invalid capture format')
            if not samples.size:
                return
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
            stream.packets.put_nowait((samples.copy(), rate))
        except queue.Full:
            self._fail(stream, '실시간 오디오 전송이 밀려 중지했습니다. 다시 시작해 주세요.')
        except Exception:
            self._fail(stream, '실시간 오디오 형식을 처리하지 못했습니다.')

    def _active(self, stream):
        return self._stream is stream and not stream.stop.is_set()

    def _safe_callback(self, callback, *args):
        if callback:
            try:
                callback(*args)
            except Exception:
                pass

    def _fail(self, stream, message):
        with self._lock:
            notify = self._active(stream) and not stream.failed
            stream.failed = True
            stream.stop.set()
        if notify:
            self._safe_callback(stream.on_error, message)

    def _connect(self):
        if self._connection_factory:
            return self._connection_factory()
        import websocket  # Optional; ordinary Luna/local startup does not import it.
        websocket.enableTrace(False)
        socket = websocket.WebSocket(enable_multithread=True,
                                     sslopt={'cert_reqs': ssl.CERT_REQUIRED, 'check_hostname': True})
        try:
            socket.connect(ENDPOINT, header=['Authorization: Bearer ' + self._api_key],
                           timeout=8, redirect_limit=0, http_no_proxy=['*'])
            socket.settimeout(0.05)
            return socket
        except Exception:
            socket.close()
            raise

    @staticmethod
    def _send(socket, event):
        socket.send(json.dumps(event, ensure_ascii=False))

    @staticmethod
    def _configuration():
        return {'type': 'session.update', 'session': {'audio': {
            'input': {'transcription': None, 'noise_reduction': None},
            'output': {'language': 'ko'}}}}

    def _send_audio(self, stream, socket, samples, rate):
        if not self._active(stream):
            return
        divisor = math.gcd(rate, SAMPLE_RATE)
        pcm = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor) if rate != SAMPLE_RATE else samples
        encoded = (np.clip(pcm, -1, 1) * 32767).astype('<i2').tobytes()
        if not encoded:
            return
        self._send(socket, {'type': 'session.input_audio_buffer.append',
                            'audio': base64.b64encode(encoded).decode('ascii')})
        # A completed send is a duration estimate, not server-confirmed usage.
        self._safe_callback(stream.on_usage, self.provider, self.model,
                            {'audio_seconds': len(encoded) / (2 * SAMPLE_RATE),
                             'duration_estimate': True}, True)

    def _consume_audio(self, stream, socket):
        sent = False
        for _ in range(MAX_PENDING_BLOCKS):
            if not self._active(stream):
                break
            try:
                samples, rate = stream.packets.get_nowait()
            except queue.Empty:
                break
            if stream.rate not in (None, rate):
                self._fail(stream, '소리 장치의 샘플레이트가 바뀌었습니다. 다시 시작해 주세요.')
                break
            stream.rate = rate
            stream.pending = np.concatenate((stream.pending, samples))
            if stream.pending.size > round(rate * .6):
                self._fail(stream, '실시간 오디오 전송 지연이 커져 중지했습니다. 다시 시작해 주세요.')
                break
        rate = stream.rate or SAMPLE_RATE
        frame = round(rate * .2)
        if stream.pending.size >= frame and self._active(stream):
            block, stream.pending = stream.pending[:frame], stream.pending[frame:]
            self._send_audio(stream, socket, block, rate)
            sent = True
        return sent

    def _caption_delta(self, stream, delta, elapsed_ms):
        if not self._active(stream) or not isinstance(delta, str) or not delta:
            return
        if len(delta) > 4096:
            self._fail(stream, '실시간 번역 응답이 너무 커서 중지했습니다.')
            return
        # Append exactly: elapsed_ms is not unique and spaces belong to deltas.
        stream.caption += delta
        # UI-only display segments, not claimed to be server utterance boundaries.
        while stream.caption:
            end = next((i + 1 for i, char in enumerate(stream.caption)
                        if char in '!?。！？\n' or (char == '.' and
                           (i == 0 or not stream.caption[i - 1].isdigit()))), None)
            if end is None and len(stream.caption) > 140:
                split = stream.caption.rfind(' ', 0, 140)
                end = split + 1 if split > 40 else 140
            complete = end is not None
            text = stream.caption[:end] if complete else stream.caption
            if text.strip():
                self._safe_callback(stream.on_caption, {
                    'korean': text.strip(), 'segment_id': stream.caption_id,
                    'provisional': not complete, 'elapsed_ms': elapsed_ms})
            if not complete:
                break
            stream.caption = stream.caption[end:]
            stream.caption_id = next(_IDS)

    def _handle_event(self, stream, event):
        if not isinstance(event, dict):
            raise ValueError('invalid event')
        name = event.get('type')
        if name == 'session.closed':
            return 'closed'
        if name == 'error':
            self._fail(stream, '실시간 번역 API 오류입니다. 키·잔액·모델 접근 권한을 확인해 주세요. 자동 재시도하지 않습니다.')
            return 'error'
        if not self._active(stream):
            return None
        event_id = event.get('event_id')
        if isinstance(event_id, str) and len(event_id) <= 128:
            if event_id in stream.seen_events:
                return None
            stream.seen_events.add(event_id)
            stream.event_order.append(event_id)
            if len(stream.event_order) > 512:
                stream.seen_events.discard(stream.event_order.popleft())
        if name == 'session.updated':
            language = event.get('session', {}).get('audio', {}).get('output', {}).get('language')
            if language != 'ko':
                self._fail(stream, '한국어 실시간 번역 설정을 확인하지 못했습니다.')
                return 'error'
            if not stream.ready:
                stream.ready = True
                self._safe_callback(stream.on_ready)
        elif name == 'session.output_transcript.delta':
            elapsed = event.get('elapsed_ms')
            self._caption_delta(stream, event.get('delta'), elapsed if type(elapsed) is int else None)
        # Output audio is deliberately discarded without decoding or playback.
        return None

    def _run(self, stream):
        socket = None
        close_sent = False
        close_deadline = None
        try:
            socket = self._connect()
            if not self._active(stream):
                return
            self._send(socket, self._configuration())
            ready_deadline = time.monotonic() + 10
            last_audio = time.monotonic()
            while True:
                now = time.monotonic()
                if stream.stop.is_set():
                    if not close_sent:
                        self._send(socket, {'type': 'session.close'})
                        close_sent, close_deadline = True, now + 3
                    if now >= close_deadline:
                        break
                elif stream.ready and now - last_audio >= .2:
                    if now - last_audio > .8:
                        self._fail(stream, '실시간 전송이 지연되어 중지했습니다. 다시 시작해 주세요.')
                        continue
                    if self._consume_audio(stream, socket):
                        last_audio += .2
                    # WASAPI may deliver no packets when all playback is idle.
                    # Preserve real-world pauses instead of splicing speech together.
                    elif self._active(stream):
                        rate = stream.rate or SAMPLE_RATE
                        frame = round(rate * .2)
                        block = np.pad(stream.pending, (0, max(0, frame - stream.pending.size)))
                        stream.pending = np.empty(0, dtype=np.float32)
                        self._send_audio(stream, socket, block, rate)
                        last_audio += .2
                elif not stream.ready and now > ready_deadline:
                    self._fail(stream, '실시간 번역 연결 준비 시간이 초과됐습니다. 자동 재시도하지 않습니다.')
                try:
                    raw = socket.recv()
                except Exception as exc:
                    # Both websocket-client and injected mocks use timeout types.
                    if isinstance(exc, TimeoutError) or type(exc).__name__ == 'WebSocketTimeoutException':
                        continue
                    raise
                if raw in ('', None):
                    if not stream.stop.is_set():
                        self._fail(stream, '실시간 번역 연결이 종료됐습니다. 다시 시작해 주세요.')
                    break
                if not isinstance(raw, (str, bytes)) or len(raw) > 524288:
                    raise ValueError('oversized event')
                was_ready = stream.ready
                action = self._handle_event(stream, json.loads(raw))
                if stream.ready and not was_ready:
                    last_audio = time.monotonic()
                if action == 'closed':
                    if not stream.stop.is_set():
                        self._fail(stream, '실시간 번역 세션이 종료됐습니다. 다시 시작해 주세요.')
                    break
        except ImportError:
            self._fail(stream, '실시간 연결 패키지가 없습니다. requirements-realtime.txt를 설치해 주세요.')
        except Exception:
            self._fail(stream, '실시간 번역 연결에 실패했습니다. 네트워크·키·잔액을 확인해 주세요. 자동 재시도하지 않습니다.')
        finally:
            stream.ready = False
            stream.stop.set()
            stream.pending = np.empty(0, dtype=np.float32)
            stream.caption = ''
            while not stream.packets.empty():
                try:
                    stream.packets.get_nowait()
                except queue.Empty:
                    break
            if socket is not None:
                try:
                    socket.close(timeout=.1)
                except TypeError:  # Small fake sockets used by offline tests.
                    socket.close()
                except Exception:
                    pass
            stream.closed.set()
