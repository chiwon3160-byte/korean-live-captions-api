from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal, QLockFile, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from capture import AudioCapture, list_sources
from engine import LocalEngine, TranslationValidationError
from api_engine import RemoteTranslationError
from pipeline import Segmenter
from ui import CaptionWindow
from comment_service import CommentBroker, CommentServer
from usage_meter import UsageMeter

ROOT = Path(__file__).resolve().parent


class Events(QObject):
    status = Signal(int, str)
    result = Signal(int, dict)
    error = Signal(int, str)
    capture_failed = Signal(int, str)
    level = Signal(int, float)
    ready = Signal(int)
    usage = Signal(str)


class Controller:
    def __init__(self, window: CaptionWindow, start_comment_server=True, engine_factory=None):
        self.window = window
        self.events = Events()
        self.usage_meter = UsageMeter()
        self.pending_api_setup = False
        self.session = 0
        self.running = False
        self.language = 'auto'
        self.capture = None
        self.capture_token = 0
        self.jobs = queue.Queue(maxsize=2)
        self.shutdown_event = threading.Event()
        self.last_level_time = 0
        self.last_caption = ''
        self.engine = (engine_factory or LocalEngine)()
        self._engine_lock = threading.RLock()
        self._engine_factory = engine_factory
        self.translation_provider = 'local'
        self.translation_description = '무료 로컬 번역'
        self.remote_comments = False
        self._remote_comment_times = deque()
        self.comments = CommentBroker()
        self.comment_server = None
        self.comment_service_status = '댓글 번역 서비스가 준비되지 않았습니다.'
        self.segmenter = Segmenter(self.enqueue, self.level)
        self.events.status.connect(self.on_status)
        self.events.result.connect(self.on_result)
        self.events.error.connect(self.on_error)
        self.events.capture_failed.connect(self.on_capture_failed)
        self.events.level.connect(self.on_level)
        self.events.ready.connect(self.on_ready)
        self.events.usage.connect(self.on_usage, Qt.ConnectionType.QueuedConnection)
        self.on_usage(self.usage_meter.summary())
        window.start_requested.connect(self.start)
        window.stop_requested.connect(self.stop)
        window.reset_requested.connect(self.reset)
        window.closing.connect(self.close)
        if hasattr(window, 'translation_settings_requested'):
            window.translation_settings_requested.connect(self.apply_translation_settings)
        if hasattr(window, 'refresh_requested'):
            window.refresh_requested.connect(self.refresh_sources)
        if hasattr(window, 'comment_setup_requested'):
            window.comment_setup_requested.connect(self.show_comment_setup)
        if start_comment_server:
            self.comment_server = CommentServer(self.comments, ROOT / '_local' / 'comment-token.txt')
            try:
                self.comment_server.start()
                self.comment_service_status = '댓글 번역 연결 준비 완료 · 처리 방식은 현재 번역 설정을 따릅니다.'
            except Exception:
                self.comment_service_status = '댓글 서비스 연결에 실패했습니다. 다른 자막 앱을 닫고 다시 실행해 주세요.'
        self.refresh_sources()
        self.worker = threading.Thread(target=self.run_worker, daemon=True, name='caption-model')
        self.worker.start()

    def show_comment_setup(self):
        if hasattr(self.window, 'show_comment_setup'):
            token = self.comment_server.token if self.comment_server and self.comment_server.server else ''
            self.window.show_comment_setup(token, str(ROOT / 'browser-extension'),
                                           self.comment_service_status)

    def apply_translation_settings(self, settings):
        if self.running or self.capture is not None or not self._engine_lock.acquire(blocking=False):
            self.window.set_error('번역 처리가 진행 중입니다. 중지 후 잠시 뒤 설정을 적용해 주세요.')
            return False
        try:
            if self._engine_factory is not None:
                self.window.set_error('비교 후보판에서는 번역 방식을 바꿀 수 없습니다.')
                return False
            from api_engine import PROVIDERS, RemoteEngine, resolve_api_key
            provider = settings.get('provider', 'local')
            if provider == 'local':
                candidate = LocalEngine()
                description = '무료 로컬 번역 · PC 안에서 처리'
            else:
                if provider not in PROVIDERS:
                    raise ValueError('provider')
                key = resolve_api_key(provider, settings.get('api_key', ''))
                model = settings.get('model', '').strip()
                if not key or not model:
                    self.window.set_error('선택한 API의 모델과 키를 입력해 주세요. 키는 저장하지 않습니다.')
                    return False
                effort = settings.get('reasoning_effort') or ('low' if model == 'gpt-6.1-sol' else 'none')
                candidate = RemoteEngine(provider=provider, model=model, api_key=key, reasoning_effort=effort)
                candidate.usage_callback = self.record_api_usage
                description = PROVIDERS[provider]['label'] + ' · ' + model
                if provider == 'openai':
                    description += ' · 추론 ' + effort
                description += ' · 유료 API'
            # Never move pending comments submitted under a previous mode to a new provider.
            with self.comments.lock:
                while True:
                    pending = self.comments.take()
                    if pending is None:
                        break
                    self.comments.finish(pending['id'], error='번역 설정이 바뀌었습니다. 댓글 번역을 다시 요청해 주세요.')
            self.session += 1
            self.clear_queue()
            self.engine = candidate
            self.pending_api_setup = False
            self.translation_provider = provider
            self.translation_description = description
            self.remote_comments = provider != 'local' and bool(settings.get('comments', False))
            self.window.clear_captions()
            if hasattr(self.window, 'set_translation_mode'):
                self.window.set_translation_mode(provider, description)
            self.window.set_status('번역 설정 적용 완료 · ' + description)
            self.write_runtime_status()
            self.on_usage(self.usage_meter.summary())
            return True
        except Exception:
            self.window.set_error('번역 설정을 적용하지 못했습니다. 제공사·모델·키 설정을 확인해 주세요.')
            return False
        finally:
            self._engine_lock.release()

    def record_api_usage(self, provider, model, usage, success):
        # Engine reports both audio and comments, even when stopped/reset while
        # awaiting a paid response. No session filter: submitted calls still cost.
        self.usage_meter.record(provider, model, usage, success)
        self.events.usage.emit(self.usage_meter.summary())

    def on_usage(self, text):
        if hasattr(self.window, 'set_usage_summary'):
            self.window.set_usage_summary(text)

    def process_comment(self, job):
        # Called only by the single model worker. Comments never touch speech context.
        if self.translation_provider != 'local' and not self.remote_comments:
            self.comments.finish(job['id'], error='유료 API 댓글 전송이 꺼져 있습니다. 앱의 번역 설정에서 별도로 켜 주세요.')
            return
        if self.translation_provider != 'local':
            now = time.monotonic()
            while self._remote_comment_times and now - self._remote_comment_times[0] >= 60:
                self._remote_comment_times.popleft()
            if len(self._remote_comment_times) >= 10:
                self.comments.finish(job['id'], error='유료 댓글 번역은 분당 최대 10건입니다. 잠시 뒤 요청해 주세요.')
                return
            self._remote_comment_times.append(now)
        try:
            # Client wait is bounded; this is not a guarantee of instant GPU cancellation.
            timeout = 2.5 if self.running else 8.0
            korean = self.engine.translate_for_comment(job['text'], timeout=timeout,
                                                       language=job['language'])
            self.comments.finish(job['id'], korean=korean)
        except RemoteTranslationError as exc:
            self.comments.finish(job['id'], error=str(exc))
        except Exception:
            # Do not return model errors containing private prompts or stop audio.
            self.comments.finish(job['id'], error='댓글 번역 시간이 초과됐거나 결과를 확인하지 못했습니다. 음성 자막을 우선하며, 잠시 뒤 다시 시도해 주세요.')

    def refresh_sources(self):
        try:
            self.window.set_sources(list_sources())
        except Exception as exc:
            self.window.set_error(f'출력 장치를 찾지 못했습니다: {exc}')

    def write_runtime_status(self):
        path = getattr(self, 'runtime_status_path', None)
        if path is None:
            return
        # Operational allowlist only: never serialize keys, text, or errors.
        state = {'pid': os.getpid(), 'provider': self.translation_provider,
                 'model': getattr(self.engine, 'model', ''),
                 'reasoning': getattr(self.engine, 'reasoning_effort', ''),
                 'running': self.running, 'capturing': self.capture is not None,
                 'source': getattr(self, 'source', ''), 'updated_at': time.time()}
        try:
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf8')
            temp.replace(path)
        except OSError:
            pass

    def start(self, source: str, language: str):
        if self.pending_api_setup:
            self.window.set_error('먼저 소리·표시 설정에서 API 키를 입력하고 번역 설정 적용을 눌러 주세요.')
            return
        if self.running:
            return
        if self.capture is not None:
            self.window.set_error('이전 소리 수집이 아직 종료되지 않았습니다. 중지를 다시 누르거나 앱을 종료해 주세요.')
            return
        self.session += 1
        self.running = True
        self.language = language
        self.source = source
        self.segmenter.reset()
        self.last_caption = ''
        self.window.set_running(True)
        self.window.clear_captions()
        self.window.set_status('음성 인식 준비 · ' + self.translation_description)
        self.clear_queue()
        self.jobs.put(('load', self.session, None, language, time.monotonic()))
        self.write_runtime_status()

    def on_ready(self, session: int):
        if session != self.session or not self.running:
            return
        try:
            self.capture_token += 1
            token = self.capture_token
            self.capture = AudioCapture(self.source, self.segmenter.feed,
                                        lambda text: self.capture_error(token, text))
            self.capture.start()
            self.window.set_status('듣는 중 · 영상의 소리를 재생해 주세요')
            self.write_runtime_status()
        except Exception as exc:
            self.on_error(session, f'소리 수집을 시작하지 못했습니다: {exc}')

    def clear_queue(self):
        while True:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break

    def capture_error(self, token, message):
        self.events.capture_failed.emit(token, message)

    def on_capture_failed(self, token, message):
        # Validate the capture lifetime on the GUI thread, after queued delivery.
        if self.running and token == self.capture_token:
            self.stop()
            self.window.set_error(message)

    def stop(self):
        self.running = False
        self.session += 1
        self.capture_token += 1
        self.clear_queue()
        stop_error = None
        if self.capture:
            try:
                self.capture.stop()
            except Exception as exc:
                stop_error = str(exc)
            else:
                self.capture = None
        self.segmenter.reset()
        self.window.set_running(False)
        self.window.set_level(0)
        self.window.set_status('중지됨 · 소리를 수집하지 않습니다')
        if stop_error:
            self.window.set_running(True)  # Keep Stop available; Start stays blocked.
            self.window.set_error(stop_error)
        self.write_runtime_status()

    def reset(self):
        # Serialize with the audio callback: old samples cannot be enqueued with a new ID.
        with self.segmenter.lock:
            self.session += 1
            self.clear_queue()
            self.segmenter.reset()
        self.last_caption = ''
        self.window.clear_captions()
        if self.running:
            if self.capture is None:
                self.jobs.put(('load', self.session, None, self.language, time.monotonic()))
            self.window.set_status('새 영상 · 이전 대기 자막을 비웠습니다')
        else:
            self.window.set_status('자막을 비웠습니다 · 시작을 누르세요')

    def level(self, value: float):
        now = time.monotonic()
        if now - self.last_level_time >= 0.10:
            self.last_level_time = now
            self.events.level.emit(self.session, value)

    def enqueue(self, audio, ended, overlap=0.0):
        if not self.running:
            return
        job = ('audio', self.session, (audio, overlap), self.language, ended)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.clear_queue()
            job = ('audio_gap', self.session, (audio, overlap), self.language, ended)
            self.events.status.emit(self.session, '처리가 늦어져 대기 구간을 건너뛰고 현재 소리부터 처리합니다')
            self.jobs.put_nowait(job)

    def run_worker(self):
        context_session = None
        context_history = []
        next_segment_id = 0
        while not self.shutdown_event.is_set():
            try:
                kind, session, audio, language, ended = self.jobs.get(timeout=0.05)
            except queue.Empty:
                with self._engine_lock:
                    comment = self.comments.take()
                    if comment is not None:
                        self.process_comment(comment)
                continue
            with self._engine_lock:
                if session != self.session or not self.running:
                    continue
                if context_session != session:
                    context_session = session
                    context_history = []
                    next_segment_id = 0
                    if hasattr(self.engine, 'reset_context'):
                        self.engine.reset_context()
                try:
                    if kind == 'load':
                        self.engine.status = lambda text, sid=session: self.events.status.emit(sid, text)
                        self.engine.load()
                        self.events.ready.emit(session)
                        continue
                    if time.monotonic() - ended > 10:
                        context_history = []
                        if hasattr(self.engine, 'reset_context'):
                            self.engine.reset_context()
                        self.events.status.emit(session, '오래 대기한 소리를 건너뛰었습니다 · 필요하면 다시 재생해 주세요')
                        continue
                    if kind == 'audio_gap':
                        context_history = []
                        if hasattr(self.engine, 'reset_context'):
                            self.engine.reset_context()
                    self.events.status.emit(session, '음성 인식·한국어 번역 중…')
                    audio_samples, overlap = audio
                    result = self.engine.process(audio_samples, language, overlap,
                                                 context=context_history[-2:],
                                                 segment_id=next_segment_id)
                    result['end_delay'] = time.monotonic() - ended
                    if session != self.session or not self.running:
                        continue
                    if result['original']:
                        for revision in result.get('revisions', []):
                            for previous in context_history:
                                if previous['id'] == revision['id']:
                                    previous['korean'] = revision['korean']
                        unit_id = result.get('subtitle_unit_id', next_segment_id)
                        context_history = [item for item in context_history if item['id'] != unit_id]
                        context_history.append({'id': unit_id,
                                                'original': result['original'],
                                                'korean': result['korean'],
                                                'language': result['language']})
                        context_history = context_history[-2:]
                        next_segment_id += 1
                    self.events.result.emit(session, result)
                except TranslationValidationError as exc:
                    context_history = []
                    if hasattr(self.engine, 'reset_context'):
                        self.engine.reset_context()
                    message = (str(exc) + ' · 이번 구간은 건너뛰었습니다'
                               if isinstance(exc, RemoteTranslationError)
                               else '이번 구간 번역을 확인하지 못해 건너뛰었습니다 · 다음 말을 계속 듣습니다')
                    self.events.status.emit(session, message)
                except Exception as exc:
                    self.events.error.emit(session, str(exc))

    def on_status(self, session, text):
        if session == self.session and self.running:
            self.window.set_status(text)

    def on_result(self, session, result):
        if session != self.session or not self.running:
            return
        if hasattr(self.window, 'revise_caption'):
            for revision in result.get('revisions', []):
                self.window.revise_caption(revision['id'], revision['korean'])
        if result['original'] and (result.get('segment_id') is not None or result['original'] != self.last_caption):
            self.last_caption = result['original']
            args = (result['original'], result['korean'], result['language'],
                    result['end_delay'], result['uncertain'])
            if 'subtitle_unit_id' in result and hasattr(self.window, 'upsert_caption'):
                self.window.upsert_caption(*args, segment_id=result['subtitle_unit_id'],
                                           provisional=not result.get('unit_complete', True))
            elif hasattr(self.window, 'revise_caption'):
                self.window.show_caption(*args, segment_id=result.get('segment_id'), provisional=self.translation_provider == 'local')
            else:
                self.window.show_caption(*args)
        self.window.set_status('듣는 중 · ' + self.translation_description)

    def on_error(self, session, message):
        if session != self.session or not self.running:
            return
        self.stop()
        self.window.set_error(message)

    def on_level(self, session, level):
        if session == self.session and self.running:
            self.window.set_level(level)

    def close(self):
        self.stop()
        self.shutdown_event.set()
        if self.comment_server:
            self.comment_server.stop()
        else:
            self.comments.close()


def parse_launch_options(argv):
    parser = argparse.ArgumentParser(description='한국어 실시간 자막')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--terra', action='store_true', help='Terra 6.1 유료 API로 시작 (로컬 키 필요)')
    mode.add_argument('--api', action='store_true', help='API 버전 · Luna 6 기본 선택')
    parser.add_argument('--reasoning', choices=('none', 'low'), default='none')
    parser.add_argument('--autostart', action='store_true', help='현재 선택한 소리 장치로 즉시 시작')
    options, _ = parser.parse_known_args(argv)
    if options.reasoning != 'none' and not (options.terra or options.api):
        parser.error('--reasoning은 --terra 또는 --api와 함께 사용해 주세요.')
    return options


def apply_launch_options(window, controller, options):
    if not (options.terra or options.api):
        return
    model = 'gpt-6.1-terra' if options.terra else 'gpt-6-luna'
    window.provider_combo.setCurrentIndex(window.provider_combo.findData('openai'))
    window.model_field.setText(model)
    window.reasoning_combo.setCurrentIndex(window.reasoning_combo.findData(options.reasoning))
    if options.api:
        from api_engine import resolve_api_key
        if not resolve_api_key('openai'):
            controller.pending_api_setup = True
            window.translation_note.setText('API 버전 · Luna 6 · API 키 등록 대기')
            window.set_status('API 키를 입력하고 번역 설정 적용을 눌러 주세요.')
            window.settings_button.setChecked(True)
            return
    if controller.apply_translation_settings({'provider': 'openai', 'model': model,
                                             'reasoning_effort': options.reasoning, 'comments': False}):
        if options.autostart:
            window._request_start()


def main(engine_factory=None):
    options = parse_launch_options(sys.argv[1:])
    app = QApplication(sys.argv)
    app.setApplicationName('한국어 영상 자막')
    app.setOrganizationName('LocalCaptions')
    (ROOT / '_local').mkdir(exist_ok=True)
    lock = QLockFile(str(ROOT / '_local' / 'app.lock'))
    if not lock.tryLock(100):
        QMessageBox.information(None, '한국어 자막', '이미 실행 중입니다. 작업 표시줄의 한국어 자막창을 확인해 주세요.')
        return 0
    window = CaptionWindow()
    controller = Controller(window, engine_factory=engine_factory)
    controller.runtime_status_path = ROOT / '_local' / 'runtime-status.json'
    controller.write_runtime_status()
    if engine_factory is not None:
        window.setWindowTitle('한국어 실시간 자막 · 비교 후보판')
    window.show()
    if (options.terra or options.api) and engine_factory is None:
        QTimer.singleShot(0, lambda: apply_launch_options(window, controller, options))
    from hotkeys import GlobalHotkeys
    hotkeys = GlobalHotkeys(app, controller.reset, controller.stop)
    if hotkeys.register():
        window.reset_button.setToolTip('어느 창에서든 Ctrl+Shift+R: 새 영상 / Ctrl+Shift+S: 중지')
    else:
        window.reset_button.setToolTip('전역 단축키가 사용 중입니다. 자막창에서 Ctrl+R 또는 새 영상 버튼을 사용하세요.')
    app.aboutToQuit.connect(hotkeys.unregister)
    app.aboutToQuit.connect(controller.close)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
