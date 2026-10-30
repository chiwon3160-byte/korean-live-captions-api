'use strict';
importScripts('logic.js');
const API = 'http://127.0.0.1:17864';
const L = CaptionCommentLogic;
let busy = false;
let status = '대기 중 · 댓글 번역은 기본 꺼짐';
const storageReady = chrome.storage.local.setAccessLevel({accessLevel: 'TRUSTED_CONTEXTS'});
async function settings() { await storageReady; return chrome.storage.local.get({enabled: false, token: ''}); }
async function request(path, token, body) {
  const response = await fetch(API + path, {
    method: body ? 'POST' : 'GET',
    headers: {'Content-Type': 'application/json', 'X-Caption-Token': token},
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'omit', cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(7000)
  });
  if (!response.ok) {
    if ([401,403].includes(response.status)) throw new Error('연결 코드가 맞지 않습니다. 앱의 코드를 다시 넣어 주세요.');
    if ([429,503].includes(response.status)) throw new Error('음성 자막을 처리하고 있습니다. 잠시 후 다시 시도합니다.');
    throw new Error('로컬 번역 응답 오류 (' + response.status + ')');
  }
  return response.json();
}
async function translate(text) {
  const cfg = await settings();
  if (!cfg.enabled) return {ok: false, paused: true};
  if (!cfg.token) return {ok: false, error: '확장 버튼에서 연결 코드를 입력해 주세요.'};
  if (!L.eligible(text)) return {ok: false, error: '댓글 길이 또는 언어를 확인해 주세요.'};
  if (busy) return {ok: false, busy: true};
  busy = true;
  try {
    status = '내 PC에서 댓글 번역 중';
    const job = await request('/comments', cfg.token, {text: L.translationText(text), language: 'auto'});
    if (typeof job.id !== 'string' || !/^[\w-]{1,100}$/.test(job.id)) throw new Error('번역 요청 번호가 올바르지 않습니다.');
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      const current = await settings();
      if (!current.enabled || current.token !== cfg.token) return {ok: false, paused: true};
      const result = await request('/comments/' + encodeURIComponent(job.id), cfg.token);
      if (result.status === 'done') {
        if (typeof result.korean !== 'string' || !result.korean.trim() || result.korean.length > 2400) throw new Error('번역 결과를 확인하지 못했습니다.');
        status = '연결됨 · 보이는 댓글을 번역합니다';
        return {ok: true, korean: result.korean};
      }
      if (result.status === 'error') throw new Error('이번 댓글을 번역하지 못했습니다. 잠시 후 다시 시도합니다.');
      if (!['queued', 'running'].includes(result.status)) throw new Error('번역 상태를 확인하지 못했습니다.');
      await new Promise(resolve => setTimeout(resolve, 850));
    }
    throw new Error('음성 자막 우선 처리로 댓글 번역이 지연됩니다. 잠시 후 다시 시도합니다.');
  } catch (error) {
    status = ['TypeError','TimeoutError'].includes(error.name)
      ? '댓글 서버에 연결하지 못했습니다. 자막앱이 켜져 있는지 확인해 주세요.' : error.message;
    return {ok: false, error: status};
  } finally { busy = false; }
}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const popup = sender.url === chrome.runtime.getURL('popup.html');
  if (!popup && (!sender.tab || !L.allowedSender(sender.url))) return false;
  const run = async () => {
    if (message?.type === 'state') {
      const cfg = await settings();
      return {enabled: cfg.enabled, paired: !!cfg.token, busy, status};
    }
    if (message?.type === 'configure' && popup) {
      const token = String(message.token || '').trim();
      if (token && !/^[A-Za-z0-9._~-]{16,256}$/.test(token)) return {ok: false, error: '연결 코드 형식을 확인해 주세요.'};
      const previous = await settings();
      const actualToken = token || previous.token;
      if (message.enabled && !actualToken) return {ok: false, error: '연결 코드를 먼저 입력해 주세요.'};
      await chrome.storage.local.set({token: actualToken, enabled: !!message.enabled});
      status = message.enabled ? '켜짐 · 틱톡에서 댓글을 열어 주세요' : '일시정지 · 새 댓글을 보내지 않습니다';
      const tabs = await chrome.tabs.query({url: 'https://www.tiktok.com/*'});
      await Promise.all(tabs.map(tab => chrome.tabs.sendMessage(tab.id, {type: 'enabled', enabled: !!message.enabled}).catch(() => {})));
      return {ok: true};
    }
    if (message?.type === 'translate' && !popup) return translate(message.text);
    return {ok: false};
  };
  run().then(sendResponse).catch(() => sendResponse({ok: false, error: '확장 프로그램 연결을 다시 확인해 주세요.'}));
  return true;
});
