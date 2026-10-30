'use strict';
const state = document.getElementById('status'), token = document.getElementById('token');
const buttons = [...document.querySelectorAll('button')];
async function refresh() {
  try {
    const s = await chrome.runtime.sendMessage({type: 'state'});
    state.textContent = (s.enabled ? '켜짐 · ' : '꺼짐 · ') + s.status;
    token.placeholder = s.paired ? '연결 코드 저장됨 · 바꿀 때만 입력' : '앱에서 복사한 코드 붙여 넣기';
  } catch { state.textContent = '확장 프로그램을 다시 열어 주세요.'; }
}
async function configure(enabled) {
  buttons.forEach(b => b.disabled = true);
  try {
    const result = await chrome.runtime.sendMessage({type: 'configure', enabled, token: token.value});
    if (!result?.ok) state.textContent = result?.error || '설정을 저장하지 못했습니다.';
    else { token.value = ''; await refresh(); }
  } catch { state.textContent = '설정을 저장하지 못했습니다. 다시 열어 주세요.'; }
  finally { buttons.forEach(b => b.disabled = false); }
}
document.getElementById('start').addEventListener('click', () => configure(true));
document.getElementById('pause').addEventListener('click', () => configure(false));
void refresh();
setInterval(refresh, 2000);
