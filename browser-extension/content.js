(() => {
  'use strict';
  const L = CaptionCommentLogic, CLASS = 'local-caption-comment-ko';
  // Explicit comment-text hooks only. Never transmit full cards with author/counts.
  const SELECTORS = [
    '[data-e2e="comment-level-1"]', '[data-e2e="comment-level-2"]',
    '[data-e2e="comment-text"]',
    '[class*="DivCommentItemWrapper"] [class*="PCommentText"]',
    '[class*="DivCommentListContainer"] [class*="PCommentText"]'
  ].join(',');
  const cache = new L.BoundedCache(150), records = new WeakMap();
  let enabled = false, busy = false, generation = 0, timer = null, lastUrl = location.href;
  function visible(node) {
    const r = node.getBoundingClientRect();
    return !!node.getClientRects().length && r.width > 0 && r.height > 0 &&
      r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth &&
      getComputedStyle(node).visibility !== 'hidden';
  }
  function source(node) {
    if (node.closest('[contenteditable="true"],textarea,form') || node.classList.contains(CLASS)) return '';
    return L.normalize(node.textContent);
  }
  function reset() {
    generation++;
    document.querySelectorAll('.' + CLASS).forEach(node => node.remove());
    cache.clear();
  }
  function render(node, original, korean) {
    const record = records.get(node);
    if (record?.label?.isConnected) record.label.remove();
    const label = document.createElement('div');
    label.className = CLASS;
    label.setAttribute('lang', 'ko');
    label.setAttribute('aria-label', '한국어 번역');
    label.textContent = korean;
    label._captionSource = node;
    node.insertAdjacentElement('afterend', label);
    records.set(node, {original, label, retryAfter: 0});
  }
  function schedule() {
    if (!enabled || timer) return;
    timer = setTimeout(() => { timer = null; void scan(); }, 450);
  }
  async function scan() {
    if (!enabled || document.hidden) return;
    if (lastUrl !== location.href) { lastUrl = location.href; reset(); }
    const candidates = [...document.querySelectorAll(SELECTORS)];
    document.querySelectorAll('.' + CLASS).forEach(label => {
      if (!label._captionSource?.isConnected) label.remove();
    });
    // Virtualized comments can change while another request is pending.
    for (const node of candidates) {
      const record = records.get(node);
      if (record && record.original !== source(node)) {
        record.label?.remove(); records.delete(node);
      }
    }
    if (busy) return;
    for (const node of candidates.filter(visible).slice(0, 50)) {
      const original = source(node);
      if (!L.eligible(original)) continue;
      const record = records.get(node);
      if (record?.original === original && (record.label?.isConnected || record.retryAfter > Date.now())) continue;
      const key = L.translationText(original), cached = cache.get(key);
      if (cached) { render(node, original, cached); continue; }
      busy = true;
      const stamp = generation;
      try {
        const result = await chrome.runtime.sendMessage({type: 'translate', text: key});
        if (!enabled || stamp !== generation || lastUrl !== location.href || !node.isConnected || source(node) !== original) return;
        if (result?.ok) {
          cache.set(key, result.korean);
          if (visible(node)) render(node, original, result.korean);
        } else {
          records.set(node, {original, retryAfter: Date.now() + (result?.busy ? 3000 : 20000)});
        }
      } catch {
        records.set(node, {original, retryAfter: Date.now() + 30000});
      } finally { busy = false; schedule(); }
      return;
    }
  }
  function setEnabled(value) {
    enabled = !!value;
    if (!enabled) reset(); else schedule();
  }
  chrome.runtime.onMessage.addListener(message => {
    if (message?.type === 'enabled') setEnabled(message.enabled);
  });
  chrome.runtime.sendMessage({type: 'state'}).then(s => setEnabled(s?.enabled)).catch(() => {});
  new MutationObserver(schedule).observe(document.body, {childList: true, subtree: true, characterData: true});
  addEventListener('scroll', schedule, {passive: true, capture: true});
  addEventListener('resize', schedule, {passive: true});
  document.addEventListener('visibilitychange', schedule);
  setInterval(schedule, 1800);
})();
