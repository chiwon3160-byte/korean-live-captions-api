"""Authenticated loopback-only comment jobs; inference stays in the app worker.

No comment text, translations, request headers, or tokens are logged to disk.
Only the pairing token is persisted, with the current user's file permissions.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import csv
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time


MAX_TEXT = 800
MAX_BODY = 8192
EXTENSION_ORIGIN = re.compile(r'chrome-extension://[a-p]{32}\Z')
JOB_PATH = re.compile(r'/comments/([a-f0-9]{32})\Z')


class ServiceError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _korean_only(text: str) -> bool:
    def hangul(ch):
        return '\uac00' <= ch <= '\ud7a3' or '\u1100' <= ch <= '\u11ff' or '\u3130' <= ch <= '\u318f'
    return any(hangul(ch) for ch in text) and all(not ch.isalpha() or hangul(ch) for ch in text)


class CommentBroker:
    def __init__(self, queue_limit=5, cache_limit=128, ttl=600, clock=time.monotonic):
        self.queue_limit = queue_limit
        self.cache_limit = cache_limit
        self.ttl = ttl
        self.clock = clock
        self.lock = threading.RLock()
        self.pending = deque()
        self.records = OrderedDict()
        self.closed = False

    def _prune(self):
        now = self.clock()
        for jid, item in list(self.records.items()):
            if now - item['updated'] >= self.ttl:
                del self.records[jid]
        self.pending = deque(jid for jid in self.pending
                             if jid in self.records and self.records[jid]['status'] == 'queued')

    def submit(self, text, language='auto') -> str:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            raise ServiceError(400, '댓글은 1~800자여야 합니다.')
        if language not in ('auto', 'en', 'es'):
            raise ServiceError(400, '지원하지 않는 입력 언어입니다.')
        text = text.strip()
        with self.lock:
            self._prune()
            if self.closed:
                raise ServiceError(503, '자막 앱이 종료되고 있습니다.')
            already_korean = _korean_only(text)
            if not already_korean and len(self.pending) >= self.queue_limit:
                raise ServiceError(429, '댓글 번역 대기열이 찼습니다. 잠시 뒤 다시 시도해 주세요.')
            while len(self.records) >= self.cache_limit:
                removable = next((jid for jid, item in self.records.items()
                                  if item['status'] in ('done', 'error')), None)
                if removable is None:
                    raise ServiceError(429, '댓글 번역 작업이 많습니다. 잠시 뒤 다시 시도해 주세요.')
                del self.records[removable]
            jid = secrets.token_hex(16)
            self.records[jid] = {'id': jid, 'status': 'done' if already_korean else 'queued',
                                 'language': language, 'updated': self.clock()}
            if already_korean:
                self.records[jid]['korean'] = text
            else:
                self.records[jid]['text'] = text
                self.pending.append(jid)
            return jid

    def take(self):
        with self.lock:
            self._prune()
            if self.closed or not self.pending:
                return None
            jid = self.pending.popleft()
            item = self.records[jid]
            item['status'] = 'running'
            item['updated'] = self.clock()
            return {'id': jid, 'text': item['text'], 'language': item['language']}

    def finish(self, jid: str, korean=None, error=None):
        with self.lock:
            item = self.records.get(jid)
            if self.closed or not item or item['status'] != 'running':
                return
            item.pop('text', None)
            item['updated'] = self.clock()
            if error is not None:
                item['status'] = 'error'
                item['error'] = str(error)[:300]
            else:
                item['status'] = 'done'
                item['korean'] = str(korean)

    def snapshot(self, jid):
        with self.lock:
            self._prune()
            item = self.records.get(jid)
            if item is None:
                raise ServiceError(404, '댓글 작업이 없거나 만료됐습니다.')
            return {key: item[key] for key in ('id', 'status', 'korean', 'error') if key in item}

    def close(self):
        with self.lock:
            self.closed = True
            self.pending.clear()
            for item in self.records.values():
                item.pop('text', None)
                if item['status'] in ('queued', 'running'):
                    item.update(status='error', error='자막 앱이 종료됐습니다.', updated=self.clock())


def _user_only(path: Path):
    if os.name == 'nt':
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        info = subprocess.check_output(['whoami', '/user', '/fo', 'csv', '/nh'],
                                       creationflags=flags).decode(errors='replace').strip()
        row = next(csv.reader(info.splitlines()))
        sid = row[-1]
        if not re.fullmatch(r'S-\d+(?:-\d+)+', sid):
            raise RuntimeError('연결 코드 파일의 사용자 권한을 확인하지 못했습니다.')
        subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r', f'*{sid}:(F)'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=flags)
    else:
        path.chmod(0o600)


def load_pairing_token(path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError('연결 코드 파일은 바로가기일 수 없습니다.')
    if path.exists():
        _user_only(path)
        token = path.read_text(encoding='ascii').strip()
        if not token:
            # Recover an empty file left by an interrupted first setup, without deleting it.
            token = secrets.token_urlsafe(32)
            path.write_text(token, encoding='ascii')
        if not re.fullmatch(r'[A-Za-z0-9_-]{43,128}', token):
            raise RuntimeError('연결 코드 파일 형식이 올바르지 않습니다.')
        return token
    token = secrets.token_urlsafe(32)
    # Set permissions on the empty file before writing any secret bytes.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        _user_only(path)
        with os.fdopen(fd, 'w', encoding='ascii') as stream:
            fd = None
            stream.write(token)
    finally:
        if fd is not None:
            os.close(fd)
    return token


class CommentHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    server_version = 'LocalCaptions'
    sys_version = ''

    def log_message(self, *args):
        pass

    def _origin_ok(self):
        origin = self.headers.get('Origin')
        return origin is None or EXTENSION_ORIGIN.fullmatch(origin) is not None

    def _respond(self, code, payload=None, preflight=False):
        body = b'' if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        origin = self.headers.get('Origin')
        if origin and EXTENSION_ORIGIN.fullmatch(origin):
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            if preflight:
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Caption-Token')
                self.send_header('Access-Control-Max-Age', '600')
                if self.headers.get('Access-Control-Request-Private-Network') == 'true':
                    self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _authenticated(self):
        if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}' or not self._origin_ok():
            self._respond(403, {'error': '허용되지 않은 요청입니다.'})
            return False
        token = self.headers.get('X-Caption-Token', '')
        if not hmac.compare_digest(token.encode('utf-8'), self.server.pairing_token.encode('ascii')):
            self._respond(401, {'error': '자막 앱의 연결 코드를 확인해 주세요.'})
            return False
        return True

    def do_OPTIONS(self):
        # Browser preflight carries header names, never the token's value.
        # It exposes no data and creates no work. Actual GET/POST still authenticate.
        origin = self.headers.get('Origin', '')
        if (self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}'
                or not EXTENSION_ORIGIN.fullmatch(origin)
                or self.headers.get('Access-Control-Request-Method') not in ('GET', 'POST')):
            self._respond(403, {'error': '허용되지 않은 사전 요청입니다.'})
            return
        self._respond(204, preflight=True)

    def do_POST(self):
        if not self._authenticated():
            return
        if self.path != '/comments':
            self._respond(404, {'error': '요청 경로를 찾지 못했습니다.'})
            return
        if self.headers.get('Transfer-Encoding'):
            self._respond(400, {'error': '지원하지 않는 본문 형식입니다.'})
            return
        if self.headers.get_content_type() != 'application/json':
            self._respond(415, {'error': 'JSON 본문이 필요합니다.'})
            return
        try:
            length = int(self.headers.get('Content-Length', ''))
            if length < 1 or length > MAX_BODY:
                raise ServiceError(413, '요청 본문이 너무 크거나 비어 있습니다.')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ServiceError(400, '요청 본문이 불완전합니다.')
            body = json.loads(raw.decode('utf-8'))
            if not isinstance(body, dict) or set(body) - {'text', 'language'}:
                raise ServiceError(400, '댓글과 언어만 보낼 수 있습니다.')
            jid = self.server.broker.submit(body.get('text'), body.get('language', 'auto'))
        except ServiceError as exc:
            self._respond(exc.code, {'error': str(exc)})
            return
        except (ValueError, UnicodeError, TimeoutError):
            self._respond(400, {'error': '올바른 JSON 댓글을 보내 주세요.'})
            return
        self._respond(202, {'id': jid})

    def do_GET(self):
        if not self._authenticated():
            return
        if self.path == '/health':
            self._respond(200, {'status': 'ready' if not self.server.broker.closed else 'closing',
                                'service': 'local-captions', 'version': 1})
            return
        match = JOB_PATH.fullmatch(self.path)
        if not match:
            self._respond(404, {'error': '요청 경로를 찾지 못했습니다.'})
            return
        try:
            value = self.server.broker.snapshot(match[1])
        except ServiceError as exc:
            self._respond(exc.code, {'error': str(exc)})
            return
        self._respond(200, value)


class _BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5)
        return request, address

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # No default traceback/request logging: even local comments remain private.
        pass


class CommentServer:
    def __init__(self, broker: CommentBroker, token_path: Path, port=17864):
        self.broker = broker
        self.token_path = Path(token_path)
        self.port = port
        self.token = ''
        self.server = None
        self.thread = None

    def start(self):
        if self.server is not None:
            return
        self.token = load_pairing_token(self.token_path)
        server = _BoundedServer(('127.0.0.1', self.port), CommentHandler)
        server.broker = self.broker
        server.pairing_token = self.token
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever,
                                       kwargs={'poll_interval': .1}, daemon=True,
                                       name='local-comment-http')
        try:
            self.thread.start()
        except Exception:
            server.server_close()
            self.server = None
            self.thread = None
            raise

    def stop(self):
        self.broker.close()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        self.thread = None
