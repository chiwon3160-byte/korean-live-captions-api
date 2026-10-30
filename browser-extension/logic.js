(function (root) {
  'use strict';
  const MAX_TEXT = 800;
  function normalize(text) { return String(text || '').normalize('NFC').replace(/\s+/gu, ' ').trim(); }
  function translationText(text) {
    // Exclude @handles. Personal names written in prose cannot be anonymized reliably.
    return normalize(text).replace(/@[\p{L}\p{N}_.]+/gu, '상대방');
  }
  function eligible(text) {
    const normalized = translationText(text);
    return normalized.length > 1 && normalized.length <= MAX_TEXT && /[A-Za-zÀ-ÿ]/u.test(normalized);
  }
  function allowedSender(url) {
    try { const u = new URL(url); return u.protocol === 'https:' && u.hostname === 'www.tiktok.com'; }
    catch { return false; }
  }
  class BoundedCache {
    constructor(limit = 150) { this.limit = limit; this.values = new Map(); }
    get(key) { return this.values.get(key); }
    set(key, value) {
      this.values.delete(key); this.values.set(key, value);
      while (this.values.size > this.limit) this.values.delete(this.values.keys().next().value);
    }
    clear() { this.values.clear(); }
  }
  const api = {MAX_TEXT, normalize, translationText, eligible, allowedSender, BoundedCache};
  root.CaptionCommentLogic = api;
  if (typeof module !== 'undefined') module.exports = api;
})(globalThis);
