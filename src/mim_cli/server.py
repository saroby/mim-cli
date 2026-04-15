"""mim serve — 로컬 미디어 브라우저 (외부 의존성 없음)."""

from __future__ import annotations

import email as _email
import json
import mimetypes
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from mim_cli.config import get_db_path, get_media_dir
from mim_cli.embeddings import EmbeddingStore
from mim_cli.providers import FetchedMedia
from mim_cli.providers.registry import (
    GEN_PROVIDERS, FETCH_PROVIDERS,
    suffix_from_mime as _suffix_from_mime,
    media_type_from_mime as _type_from_mime,
    media_type_from_suffix as _type_from_suffix,
)
from mim_cli.saver import MetaOverride, save_media
from mim_cli.store import MediaStore


# ── Fetch 결과 캐시 ─────────────────────────────────────────────────────────

class _FetchCache:
    """검색 결과 bytes를 서버 메모리에 임시 보관. 저장 전 미리보기용."""

    _MAX_ENTRIES = 200
    _MAX_BYTES = 512 * 1024 * 1024  # 512 MB

    def __init__(self) -> None:
        self._data: dict[str, FetchedMedia] = {}
        self._bytes: int = 0
        self._lock = threading.Lock()

    def _evict(self) -> None:
        while self._data and (
            len(self._data) > self._MAX_ENTRIES or self._bytes > self._MAX_BYTES
        ):
            key = next(iter(self._data))
            fm = self._data.pop(key)
            self._bytes -= len(fm.data)

    def put(self, items: list[FetchedMedia]) -> list[dict]:
        out = []
        with self._lock:
            for fm in items:
                cid = str(uuid.uuid4())
                self._data[cid] = fm
                self._bytes += len(fm.data)
                out.append({
                    "cache_id": cid,
                    "source_url": fm.source_url,
                    "source_id": fm.source_id,
                    "mime_type": fm.mime_type,
                    "width": fm.width,
                    "height": fm.height,
                    "attribution": fm.attribution,
                    "license": fm.license,
                    "license_url": fm.license_url,
                    "title": fm.metadata.get("title") or fm.metadata.get("name") or "",
                })
            self._evict()
        return out

    def get(self, cid: str) -> Optional[FetchedMedia]:
        with self._lock:
            return self._data.get(cid)

    def remove(self, cid: str) -> None:
        with self._lock:
            fm = self._data.pop(cid, None)
            if fm:
                self._bytes -= len(fm.data)


# ── HTML SPA ────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mim</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0f0f11;--sur:#1a1a1f;--bor:#2a2a33;--acc:#7c6aff;--tex:#e8e8f0;--mut:#888899;--rad:10px;--danger:#ff5f5f}
body{background:var(--bg);color:var(--tex);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;min-height:100vh}

/* ── 헤더 ── */
header{position:sticky;top:0;z-index:100;background:var(--bg);border-bottom:1px solid var(--bor);padding:10px 18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
header h1{font-size:17px;font-weight:700;letter-spacing:-.5px;flex-shrink:0}
header h1 span{color:var(--acc)}
nav{display:flex;gap:4px}
.nav-btn{background:none;border:1px solid transparent;border-radius:7px;color:var(--mut);cursor:pointer;font-size:13px;padding:5px 12px;transition:all .15s}
.nav-btn.on,.nav-btn:hover{border-color:var(--bor);color:var(--tex);background:var(--sur)}
.nav-btn.on{border-color:var(--acc);color:var(--acc)}
#lib-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex:1}
#search{flex:1;min-width:160px;max-width:300px;background:var(--sur);border:1px solid var(--bor);border-radius:7px;color:var(--tex);font-size:13px;padding:6px 11px;outline:none}
#search:focus{border-color:var(--acc)}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{cursor:pointer;border:1px solid var(--bor);border-radius:20px;padding:3px 11px;font-size:12px;color:var(--mut);background:transparent;transition:all .15s;user-select:none}
.chip.on,.chip:hover{border-color:var(--acc);color:var(--tex);background:#7c6aff22}
#count{font-size:12px;color:var(--mut);white-space:nowrap}

/* ── 탭 패널 ── */
.tab{padding:16px 18px}.tab[hidden]{display:none!important}

/* ── 그리드 ── */
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.card{background:var(--sur);border:1px solid var(--bor);border-radius:var(--rad);overflow:hidden;cursor:pointer;transition:transform .15s,border-color .15s}
.card:hover{transform:translateY(-2px);border-color:var(--acc)}
.thumb{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#111}
.card-info{padding:7px 10px}
.card-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card-meta{font-size:11px;color:var(--mut);margin-top:2px}
.badge{display:inline-block;font-size:10px;font-weight:600;border-radius:4px;padding:1px 5px;margin-right:3px}
.badge-image{background:#1a3a5c;color:#7ab8f5}.badge-gif{background:#3a1a4a;color:#c47af5}.badge-video{background:#1a3a2a;color:#7af5a0}
.badge-upload{background:#2a2a1a;color:#f5d07a}.badge-generated{background:#1a2a3a;color:#7ac0f5}.badge-fetched{background:#1a3a3a;color:#7af5e0}

/* ── 빈/로딩 ── */
.empty-msg{text-align:center;padding:60px 20px;color:var(--mut)}
.empty-msg h2{font-size:22px;margin-bottom:6px}

/* ── 폼 패널 ── */
.panel{background:var(--sur);border:1px solid var(--bor);border-radius:var(--rad);padding:16px 18px;margin-bottom:14px}
.form-row{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:11px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.form-group input,.form-group select,.form-group textarea{background:var(--bg);border:1px solid var(--bor);border-radius:7px;color:var(--tex);font-size:13px;padding:7px 11px;outline:none}
.form-group input:focus,.form-group select:focus,.form-group textarea:focus{border-color:var(--acc)}
.form-group textarea{resize:vertical;min-height:72px;font-family:inherit}
.form-group select option{background:var(--sur)}
.fg-grow{flex:1;min-width:160px}

/* ── 버튼 ── */
.btn{cursor:pointer;border-radius:7px;font-size:13px;font-weight:600;padding:7px 16px;border:none;transition:all .15s;display:inline-flex;align-items:center;gap:6px}
.btn-primary{background:var(--acc);color:#fff}.btn-primary:hover{background:#6a5aef}
.btn-ghost{background:var(--sur);color:var(--tex);border:1px solid var(--bor)}.btn-ghost:hover{border-color:var(--acc)}
.btn-danger{background:#ff5f5f22;color:var(--danger);border:1px solid #ff5f5f44}.btn-danger:hover{background:#ff5f5f33}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* ── Fetch 결과 ── */
#fetch-action-bar{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.fetch-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px}
.fc{position:relative;border-radius:var(--rad);overflow:hidden;border:2px solid var(--bor);cursor:pointer;transition:border-color .15s;aspect-ratio:1;background:#111}
.fc:hover{border-color:var(--acc)}
.fc.sel{border-color:var(--acc);box-shadow:0 0 0 2px #7c6aff55}
.fc img,.fc video{width:100%;height:100%;object-fit:cover;display:block}
.fc-check{position:absolute;top:6px;right:6px;width:20px;height:20px;border-radius:50%;background:#0009;border:2px solid #fff5;display:flex;align-items:center;justify-content:center;font-size:11px;transition:all .15s}
.fc.sel .fc-check{background:var(--acc);border-color:var(--acc)}
.fc-label{position:absolute;bottom:0;left:0;right:0;padding:4px 6px;background:#000a;font-size:10px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* ── Generate 결과 ── */
#gen-result{margin-top:14px;display:none}
#gen-result img{border-radius:var(--rad);max-width:360px;border:1px solid var(--bor);display:block;margin-bottom:10px}

/* ── Upload ── */
#dropzone{border:2px dashed var(--bor);border-radius:var(--rad);padding:36px 20px;text-align:center;cursor:pointer;transition:border-color .2s;margin-bottom:14px;color:var(--mut)}
#dropzone.drag{border-color:var(--acc);background:#7c6aff11}
#dropzone input{display:none}
#dropzone p{margin-top:8px;font-size:13px}
#upload-preview{max-width:200px;border-radius:var(--rad);margin:10px 0;display:none}
#upload-result{margin-top:14px;display:none}

/* ── 모달 ── */
.overlay{display:none;position:fixed;inset:0;z-index:200;background:#000a;align-items:center;justify-content:center}
.overlay.open{display:flex}
.modal{background:var(--sur);border:1px solid var(--bor);border-radius:14px;max-width:860px;width:95vw;max-height:90vh;overflow-y:auto;display:flex;flex-direction:column}
.modal-hdr{padding:13px 17px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--bor);position:sticky;top:0;background:var(--sur);z-index:1}
.modal-hdr h2{font-size:15px;font-weight:700}
.modal-close{cursor:pointer;font-size:20px;color:var(--mut);background:none;border:none;line-height:1;padding:4px}
.modal-close:hover{color:var(--tex)}
.modal-body{display:grid;grid-template-columns:1fr 1fr;gap:0}
@media(max-width:580px){.modal-body{grid-template-columns:1fr}}
.modal-preview{background:#000;display:flex;align-items:center;justify-content:center;min-height:260px;border-right:1px solid var(--bor)}
.modal-preview img,.modal-preview video{max-width:100%;max-height:55vh;object-fit:contain;display:block}
.modal-meta{padding:15px 17px;overflow-y:auto}
.modal-actions{display:flex;gap:8px;margin-bottom:14px}
.meta-row{display:flex;flex-direction:column;gap:2px;margin-bottom:10px}
.meta-lbl{font-size:11px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.meta-val{font-size:13px;word-break:break-all}
.tag-list{display:flex;flex-wrap:wrap;gap:4px;margin-top:3px}
.tag{background:#252530;border:1px solid var(--bor);border-radius:4px;padding:2px 7px;font-size:11px}

/* ── 편집 모달 ── */
#edit-overlay .modal{max-width:480px}
#edit-form .form-group{margin-bottom:12px}

/* ── 토스트 ── */
#toast-wrap{position:fixed;bottom:22px;right:22px;z-index:999;display:flex;flex-direction:column;gap:8px}
.toast{padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;animation:fadeup .3s ease;box-shadow:0 4px 20px #0006}
.toast-ok{background:#1a3a2a;color:#7af5a0;border:1px solid #3a5a4a}
.toast-err{background:#3a1a1a;color:#f57a7a;border:1px solid #5a3a3a}
.toast-info{background:#1a2a3a;color:#7ab8f5;border:1px solid #2a4a6a}
@keyframes fadeup{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.spin{display:inline-block;width:14px;height:14px;border:2px solid #fff4;border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<header>
  <h1>m<span>i</span>m</h1>
  <nav>
    <button class="nav-btn on" onclick="showTab('library')">라이브러리</button>
    <button class="nav-btn" onclick="showTab('fetch')">가져오기</button>
    <button class="nav-btn" onclick="showTab('generate')">생성</button>
    <button class="nav-btn" onclick="showTab('upload')">업로드</button>
  </nav>
  <div id="lib-controls">
    <input id="search" type="search" placeholder="검색…" autocomplete="off">
    <div class="chips" id="type-chips">
      <span class="chip on" data-val="">전체</span>
      <span class="chip" data-val="image">이미지</span>
      <span class="chip" data-val="gif">GIF</span>
      <span class="chip" data-val="video">비디오</span>
    </div>
    <div class="chips" id="src-chips">
      <span class="chip on" data-val="">모든 출처</span>
      <span class="chip" data-val="upload">업로드</span>
      <span class="chip" data-val="generated">생성</span>
      <span class="chip" data-val="fetched">가져오기</span>
    </div>
    <span id="count"></span>
  </div>
</header>

<!-- ── 라이브러리 탭 ── -->
<div id="tab-library" class="tab">
  <div id="lib-loading" class="empty-msg"><div class="spin" style="width:24px;height:24px;margin:0 auto"></div></div>
  <div id="lib-empty" class="empty-msg" style="display:none"><h2>🌵</h2><p>항목이 없습니다.</p></div>
  <div id="grid"></div>
</div>

<!-- ── 가져오기 탭 ── -->
<div id="tab-fetch" class="tab" hidden>
  <div class="panel">
    <div class="form-row">
      <div class="form-group">
        <label>프로바이더</label>
        <select id="fp-provider"></select>
      </div>
      <div class="form-group fg-grow">
        <label>검색어</label>
        <input id="fp-query" type="text" placeholder="예: 웃긴 고양이">
      </div>
      <div class="form-group">
        <label>개수</label>
        <input id="fp-limit" type="number" value="6" min="1" max="50" style="width:70px">
      </div>
      <div class="form-group">
        <label>타입</label>
        <select id="fp-type">
          <option value="">전체</option>
          <option value="image">이미지</option>
          <option value="gif">GIF</option>
          <option value="video">비디오</option>
        </select>
      </div>
      <button class="btn btn-primary" id="fp-search-btn" onclick="doFetchSearch()">검색</button>
    </div>
  </div>

  <div id="fetch-action-bar" style="display:none">
    <button class="btn btn-ghost" onclick="fetchSelectAll()">모두 선택</button>
    <button class="btn btn-ghost" onclick="fetchSelectNone()">선택 해제</button>
    <span id="fetch-sel-count" style="color:var(--mut);font-size:13px"></span>
    <button class="btn btn-primary" id="fp-save-btn" onclick="doFetchSave()">선택 저장</button>
  </div>

  <div id="fetch-loading" class="empty-msg" style="display:none"><div class="spin" style="width:24px;height:24px;margin:0 auto"></div></div>
  <div id="fetch-empty" class="empty-msg" style="display:none"><h2>🔍</h2><p>결과가 없습니다.</p></div>
  <div id="fetch-grid" class="fetch-grid"></div>
</div>

<!-- ── 생성 탭 ── -->
<div id="tab-generate" class="tab" hidden>
  <div class="panel">
    <div class="form-group" style="margin-bottom:12px">
      <label>프롬프트</label>
      <textarea id="gp-prompt" placeholder="생성할 이미지를 설명하세요…"></textarea>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>프로바이더</label>
        <select id="gp-provider"></select>
      </div>
      <div class="form-group fg-grow">
        <label>모델 (선택)</label>
        <input id="gp-model" type="text" placeholder="기본값 사용">
      </div>
      <div class="form-group">
        <label>비율</label>
        <select id="gp-ratio">
          <option value="">기본</option>
          <option value="1:1">1:1</option>
          <option value="16:9">16:9</option>
          <option value="9:16">9:16</option>
          <option value="4:3">4:3</option>
          <option value="3:4">3:4</option>
        </select>
      </div>
      <button class="btn btn-primary" id="gp-btn" onclick="doGenerate()">생성</button>
    </div>
  </div>
  <div id="gen-result">
    <img id="gen-img" src="" alt="">
    <div id="gen-meta"></div>
  </div>
</div>

<!-- ── 업로드 탭 ── -->
<div id="tab-upload" class="tab" hidden>
  <div class="panel">
    <div id="dropzone" onclick="document.getElementById('up-file').click()"
         ondragover="event.preventDefault();this.classList.add('drag')"
         ondragleave="this.classList.remove('drag')"
         ondrop="handleDrop(event)">
      <input id="up-file" type="file" accept="image/*,video/*" onchange="handleFileSelect(event)">
      <div style="font-size:28px">📁</div>
      <p>클릭하거나 파일을 여기에 드래그</p>
      <p id="drop-filename" style="color:var(--acc);margin-top:4px"></p>
    </div>
    <img id="upload-preview" src="" alt="">
    <div class="form-row" style="margin-bottom:12px">
      <div class="form-group fg-grow">
        <label>이름 (선택 — 빈칸이면 AI 분석)</label>
        <input id="up-name" type="text" placeholder="자동 생성">
      </div>
      <div class="form-group fg-grow">
        <label>태그 (쉼표 구분, 선택)</label>
        <input id="up-tags" type="text" placeholder="자동 생성">
      </div>
    </div>
    <div class="form-row" style="margin-bottom:14px">
      <div class="form-group fg-grow">
        <label>감정 태그 (선택)</label>
        <input id="up-emotions" type="text" placeholder="자동 생성">
      </div>
      <div class="form-group fg-grow">
        <label>맥락 태그 (선택)</label>
        <input id="up-context" type="text" placeholder="자동 생성">
      </div>
    </div>
    <label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer;margin-bottom:14px">
      <input id="up-skip-meta" type="checkbox">
      AI 메타데이터 분석 스킵
    </label>
    <button class="btn btn-primary" id="up-btn" onclick="doUpload()">업로드</button>
  </div>
  <div id="upload-result"></div>
</div>

<!-- ── 상세 모달 ── -->
<div id="detail-overlay" class="overlay" onclick="if(event.target===this)closeDetail()">
  <div class="modal">
    <div class="modal-hdr">
      <h2 id="detail-title"></h2>
      <button class="modal-close" onclick="closeDetail()">✕</button>
    </div>
    <div class="modal-body">
      <div class="modal-preview" id="detail-preview"></div>
      <div class="modal-meta" id="detail-meta"></div>
    </div>
  </div>
</div>

<!-- ── 편집 모달 ── -->
<div id="edit-overlay" class="overlay" onclick="if(event.target===this)closeEdit()">
  <div class="modal">
    <div class="modal-hdr">
      <h2>메타데이터 편집</h2>
      <button class="modal-close" onclick="closeEdit()">✕</button>
    </div>
    <div style="padding:18px">
      <form id="edit-form" onsubmit="submitEdit(event)">
        <input type="hidden" id="edit-id">
        <div class="form-group" style="margin-bottom:12px">
          <label>이름</label>
          <input id="edit-name" type="text" required>
        </div>
        <div class="form-group" style="margin-bottom:12px">
          <label>설명</label>
          <textarea id="edit-desc"></textarea>
        </div>
        <div class="form-group" style="margin-bottom:12px">
          <label>태그 (쉼표 구분)</label>
          <input id="edit-tags" type="text">
        </div>
        <div class="form-group" style="margin-bottom:12px">
          <label>감정 태그</label>
          <input id="edit-emotions" type="text">
        </div>
        <div class="form-group" style="margin-bottom:16px">
          <label>맥락 태그</label>
          <input id="edit-context" type="text">
        </div>
        <div style="display:flex;gap:8px">
          <button type="submit" class="btn btn-primary">저장</button>
          <button type="button" class="btn btn-ghost" onclick="closeEdit()">취소</button>
        </div>
      </form>
    </div>
  </div>
</div>

<div id="toast-wrap"></div>

<script>
// ── 상태 ────────────────────────────────────────────────────────────
let allItems = [];
let filterType = '', filterSrc = '', searchQ = '';
let currentItem = null;
let fetchResults = [];  // [{cache_id, source_url, ...}]
let fetchSelected = new Set();
let fetchProvider = '', fetchQuery = '';
let uploadFile = null;

// ── 탭 ──────────────────────────────────────────────────────────────
function showTab(name) {
  ['library','fetch','generate','upload'].forEach(t => {
    const el = document.getElementById('tab-' + t);
    if (el) el.hidden = (t !== name);
  });
  document.querySelectorAll('.nav-btn').forEach((b, i) => {
    b.classList.toggle('on', ['library','fetch','generate','upload'][i] === name);
  });
  document.getElementById('lib-controls').style.display = name === 'library' ? '' : 'none';
}

// ── API 헬퍼 ────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

// ── 토스트 ──────────────────────────────────────────────────────────
function toast(msg, type='ok') {
  const w = document.getElementById('toast-wrap');
  const d = document.createElement('div');
  d.className = `toast toast-${type}`;
  d.textContent = msg;
  w.appendChild(d);
  setTimeout(() => d.remove(), 3200);
}

// ── 라이브러리 ──────────────────────────────────────────────────────
async function loadLibrary() {
  try {
    allItems = await api('GET', '/api/items');
    document.getElementById('lib-loading').style.display = 'none';
    renderGrid();
  } catch(e) { toast('라이브러리 로드 실패: ' + e.message, 'err'); }
}

function renderGrid() {
  const q = searchQ.trim().toLowerCase();
  const filtered = allItems.filter(item => {
    if (filterType && item.media_type !== filterType) return false;
    if (filterSrc  && item.source      !== filterSrc)  return false;
    if (q) {
      const hay = [item.name, item.description, ...(item.tags||[]), ...(item.emotions||[])].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  document.getElementById('count').textContent = filtered.length + '개';
  const grid = document.getElementById('grid');
  const empty = document.getElementById('lib-empty');
  grid.innerHTML = '';
  if (!filtered.length) { empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  filtered.forEach(item => grid.appendChild(makeCard(item)));
}

function makeCard(item) {
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = makeThumb(item, 'thumb') +
    `<div class="card-info">
      <div class="card-name" title="${esc(item.name)}">${esc(item.name)}</div>
      <div class="card-meta">
        <span class="badge badge-${item.media_type}">${item.media_type}</span>
        <span class="badge badge-${item.source}">${item.source}</span>
      </div>
    </div>`;
  div.onclick = () => openDetail(item);
  return div;
}

function makeThumb(item, cls) {
  const src = `/media/${encodeURIComponent(item.id)}`;
  if (item.media_type === 'video')
    return `<video class="${cls}" src="${src}" muted preload="metadata" loop></video>`;
  return `<img class="${cls}" src="${src}" loading="lazy" alt="${esc(item.name)}">`;
}

// hover 재생
document.getElementById('grid').addEventListener('mouseover', e => {
  const v = e.target.closest('.card')?.querySelector('video');
  if (v) v.play().catch(()=>{});
});
document.getElementById('grid').addEventListener('mouseout', e => {
  const v = e.target.closest('.card')?.querySelector('video');
  if (v) { v.pause(); v.currentTime = 0; }
});

// 필터 칩
function setupChips(id, setter) {
  document.getElementById(id).addEventListener('click', e => {
    const chip = e.target.closest('.chip'); if (!chip) return;
    document.querySelectorAll('#' + id + ' .chip').forEach(c => c.classList.remove('on'));
    chip.classList.add('on'); setter(chip.dataset.val); renderGrid();
  });
}
setupChips('type-chips', v => filterType = v);
setupChips('src-chips',  v => filterSrc  = v);

let debT;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(debT);
  debT = setTimeout(() => { searchQ = e.target.value; renderGrid(); }, 200);
});

// ── 상세 모달 ──────────────────────────────────────────────────────
function openDetail(item) {
  currentItem = item;
  const src = `/media/${encodeURIComponent(item.id)}`;
  const prev = document.getElementById('detail-preview');
  prev.innerHTML = item.media_type === 'video'
    ? `<video src="${src}" controls autoplay loop style="max-width:100%;max-height:55vh"></video>`
    : `<img src="${src}" alt="${esc(item.name)}" style="max-width:100%;max-height:55vh;object-fit:contain">`;

  document.getElementById('detail-title').textContent = item.name;

  const meta = document.getElementById('detail-meta');
  meta.innerHTML = `
    <div class="modal-actions">
      <button class="btn btn-ghost" onclick="openEdit(currentItem)">편집</button>
      <button class="btn btn-danger" onclick="confirmDelete(currentItem)">삭제</button>
    </div>` +
    rows([
      ['ID', `<span style="font-family:monospace;font-size:11px">${item.id}</span>`],
      ['타입', badge(item.media_type, 'media_type')],
      ['출처', badge(item.source, 'source') + (item.source_provider ? ' ' + esc(item.source_provider) : '')],
      item.description && ['설명', esc(item.description)],
      item.tags?.length && ['태그', tagList(item.tags)],
      item.emotions?.length && ['감정', tagList(item.emotions)],
      item.context?.length && ['맥락', tagList(item.context)],
      item.prompt && ['프롬프트', esc(item.prompt)],
      item.model  && ['모델', esc(item.model)],
      (item.width && item.height) && ['크기', `${item.width} × ${item.height}`],
      item.source_url && ['원본 URL', `<a href="${esc(item.source_url)}" target="_blank" style="color:var(--acc)">${esc(item.source_url).slice(0,60)}…</a>`],
      item.attribution && ['저작자', esc(item.attribution)],
      item.license && ['라이선스', item.license_url
        ? `<a href="${esc(item.license_url)}" target="_blank" style="color:var(--acc)">${esc(item.license)}</a>`
        : esc(item.license)],
      ['생성일', item.created_at.replace('T',' ').slice(0,19)],
    ]);

  document.getElementById('detail-overlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeDetail() {
  document.getElementById('detail-overlay').classList.remove('open');
  document.getElementById('detail-preview').innerHTML = '';
  document.body.style.overflow = '';
}

// ── 편집 ──────────────────────────────────────────────────────────
function openEdit(item) {
  document.getElementById('edit-id').value = item.id;
  document.getElementById('edit-name').value = item.name;
  document.getElementById('edit-desc').value = item.description || '';
  document.getElementById('edit-tags').value = (item.tags||[]).join(', ');
  document.getElementById('edit-emotions').value = (item.emotions||[]).join(', ');
  document.getElementById('edit-context').value = (item.context||[]).join(', ');
  document.getElementById('edit-overlay').classList.add('open');
}

function closeEdit() {
  document.getElementById('edit-overlay').classList.remove('open');
}

async function submitEdit(e) {
  e.preventDefault();
  const id = document.getElementById('edit-id').value;
  const body = {
    name: document.getElementById('edit-name').value,
    description: document.getElementById('edit-desc').value,
    tags: document.getElementById('edit-tags').value.split(',').map(s=>s.trim()).filter(Boolean),
    emotions: document.getElementById('edit-emotions').value.split(',').map(s=>s.trim()).filter(Boolean),
    context: document.getElementById('edit-context').value.split(',').map(s=>s.trim()).filter(Boolean),
  };
  try {
    const updated = await api('PATCH', `/api/items/${id}`, body);
    const idx = allItems.findIndex(i => i.id === id);
    if (idx >= 0) allItems[idx] = updated;
    if (currentItem?.id === id) currentItem = updated;
    renderGrid();
    closeEdit();
    toast('저장됨');
  } catch(e) { toast('편집 실패: ' + e.message, 'err'); }
}

// ── 삭제 ──────────────────────────────────────────────────────────
async function confirmDelete(item) {
  if (!confirm(`'${esc(item.name)}'을(를) 삭제하시겠습니까?`)) return;
  try {
    await api('DELETE', `/api/items/${item.id}`);
    allItems = allItems.filter(i => i.id !== item.id);
    renderGrid();
    closeDetail();
    toast('삭제됨');
  } catch(e) { toast('삭제 실패: ' + e.message, 'err'); }
}

// ── 가져오기 ──────────────────────────────────────────────────────
async function doFetchSearch() {
  const provider = document.getElementById('fp-provider').value;
  const query    = document.getElementById('fp-query').value.trim();
  const limit    = parseInt(document.getElementById('fp-limit').value) || 6;
  const mtype    = document.getElementById('fp-type').value;
  if (!query) { toast('검색어를 입력하세요', 'info'); return; }

  fetchProvider = provider; fetchQuery = query;
  fetchSelected.clear();

  const btn = document.getElementById('fp-search-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span>';
  document.getElementById('fetch-loading').style.display = 'block';
  document.getElementById('fetch-grid').innerHTML = '';
  document.getElementById('fetch-action-bar').style.display = 'none';
  document.getElementById('fetch-empty').style.display = 'none';

  try {
    const body = { provider, query, limit };
    if (mtype) body.media_type = mtype;
    fetchResults = await api('POST', '/api/fetch/search', body);
    renderFetchResults();
  } catch(e) {
    toast('검색 실패: ' + e.message, 'err');
  } finally {
    btn.disabled = false; btn.textContent = '검색';
    document.getElementById('fetch-loading').style.display = 'none';
  }
}

function renderFetchResults() {
  const grid = document.getElementById('fetch-grid');
  grid.innerHTML = '';
  if (!fetchResults.length) {
    document.getElementById('fetch-empty').style.display = 'block'; return;
  }
  document.getElementById('fetch-action-bar').style.display = 'flex';
  updateFetchSelCount();
  fetchResults.forEach(item => {
    const div = document.createElement('div');
    div.className = 'fc';
    div.dataset.cid = item.cache_id;
    const previewSrc = `/api/fetch/preview/${encodeURIComponent(item.cache_id)}`;
    const isVid = item.mime_type?.startsWith('video/');
    div.innerHTML =
      (isVid ? `<video src="${previewSrc}" muted preload="metadata" loop></video>`
              : `<img src="${previewSrc}" loading="lazy" alt="">`) +
      `<div class="fc-check">✓</div>` +
      (item.attribution ? `<div class="fc-label">${esc(item.attribution)}</div>` : '');
    div.onclick = () => toggleFetchItem(item.cache_id, div);
    if (isVid) {
      div.addEventListener('mouseenter', () => div.querySelector('video')?.play().catch(()=>{}));
      div.addEventListener('mouseleave', () => { const v = div.querySelector('video'); if(v){v.pause();v.currentTime=0;} });
    }
    grid.appendChild(div);
  });
}

function toggleFetchItem(cid, el) {
  if (fetchSelected.has(cid)) { fetchSelected.delete(cid); el.classList.remove('sel'); }
  else { fetchSelected.add(cid); el.classList.add('sel'); }
  updateFetchSelCount();
}

function fetchSelectAll()  { fetchResults.forEach(r => { fetchSelected.add(r.cache_id); document.querySelector(`[data-cid="${r.cache_id}"]`)?.classList.add('sel'); }); updateFetchSelCount(); }
function fetchSelectNone() { fetchSelected.clear(); document.querySelectorAll('.fc').forEach(el => el.classList.remove('sel')); updateFetchSelCount(); }

function updateFetchSelCount() {
  const n = fetchSelected.size;
  document.getElementById('fetch-sel-count').textContent = n ? `${n}개 선택됨` : '';
  document.getElementById('fp-save-btn').disabled = n === 0;
}

async function doFetchSave() {
  if (!fetchSelected.size) return;
  const items = fetchResults.filter(r => fetchSelected.has(r.cache_id));
  const btn = document.getElementById('fp-save-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> 저장 중';
  try {
    const result = await api('POST', '/api/fetch/save', {
      provider: fetchProvider, query: fetchQuery, items
    });
    const all = Array.isArray(result) ? result : [result];
    const ok = all.filter(r => r.status === 'saved');
    const skipped = all.filter(r => r.status === 'skipped');
    const errors = all.filter(r => r.status === 'error');
    ok.forEach(item => { if (!allItems.find(i => i.id === item.id)) allItems.unshift(item); });
    const parts = [];
    if (ok.length) parts.push(`${ok.length}개 저장됨`);
    if (skipped.length) parts.push(`${skipped.length}개 중복 스킵`);
    if (errors.length) parts.push(`${errors.length}개 실패`);
    toast(parts.join(', '), errors.length ? 'err' : 'ok');
    fetchSelectNone();
  } catch(e) { toast('저장 실패: ' + e.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = '선택 저장'; updateFetchSelCount(); }
}

// ── 생성 ──────────────────────────────────────────────────────────
async function doGenerate() {
  const prompt = document.getElementById('gp-prompt').value.trim();
  if (!prompt) { toast('프롬프트를 입력하세요', 'info'); return; }
  const provider = document.getElementById('gp-provider').value;
  const model    = document.getElementById('gp-model').value.trim() || null;
  const ratio    = document.getElementById('gp-ratio').value || null;

  const btn = document.getElementById('gp-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> 생성 중';
  document.getElementById('gen-result').style.display = 'none';

  try {
    const item = await api('POST', '/api/generate', { prompt, provider, model, aspect_ratio: ratio });
    allItems.unshift(item);
    renderGrid();
    const imgEl = document.getElementById('gen-img');
    imgEl.src = `/media/${encodeURIComponent(item.id)}`;
    document.getElementById('gen-meta').innerHTML = rows([
      ['이름', esc(item.name)],
      item.tags?.length && ['태그', tagList(item.tags)],
      ['ID', `<span style="font-family:monospace;font-size:11px">${item.id}</span>`],
    ]);
    document.getElementById('gen-result').style.display = 'block';
    toast('생성 완료');
  } catch(e) { toast('생성 실패: ' + e.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = '생성'; }
}

// ── 업로드 ──────────────────────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropzone').classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) setUploadFile(f);
}

function handleFileSelect(e) {
  const f = e.target.files[0];
  if (f) setUploadFile(f);
}

function setUploadFile(f) {
  uploadFile = f;
  document.getElementById('drop-filename').textContent = f.name;
  const prev = document.getElementById('upload-preview');
  if (f.type.startsWith('image/')) {
    prev.src = URL.createObjectURL(f);
    prev.style.display = 'block';
  } else { prev.style.display = 'none'; }
}

async function doUpload() {
  if (!uploadFile) { toast('파일을 선택하세요', 'info'); return; }
  const btn = document.getElementById('up-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spin"></span> 업로드 중';

  const fd = new FormData();
  fd.append('file', uploadFile, uploadFile.name);
  const name = document.getElementById('up-name').value.trim();
  const tags = document.getElementById('up-tags').value.trim();
  const emotions = document.getElementById('up-emotions').value.trim();
  const ctx = document.getElementById('up-context').value.trim();
  const skip = document.getElementById('up-skip-meta').checked;
  if (name) fd.append('name', name);
  if (tags) fd.append('tags', tags);
  if (emotions) fd.append('emotions', emotions);
  if (ctx) fd.append('context', ctx);
  if (skip) fd.append('skip_metadata', '1');

  try {
    const item = await api('POST', '/api/add', fd);
    if (!item.skipped) { allItems.unshift(item); renderGrid(); }
    document.getElementById('upload-result').innerHTML =
      `<div class="panel">
        <div style="color:${item.skipped?'var(--mut)':'#7af5a0'};font-weight:600;margin-bottom:8px">
          ${item.skipped ? '⚠ 중복 — 기존 아이템 반환' : '✓ 업로드 완료'}
        </div>` +
        rows([['이름', esc(item.name)], ['ID', `<span style="font-family:monospace;font-size:11px">${item.id}</span>`]]) +
      `</div>`;
    document.getElementById('upload-result').style.display = 'block';
    toast(item.skipped ? '이미 존재하는 파일' : '업로드 완료', item.skipped ? 'info' : 'ok');
  } catch(e) { toast('업로드 실패: ' + e.message, 'err'); }
  finally { btn.disabled = false; btn.textContent = '업로드'; }
}

// ── 유틸 ──────────────────────────────────────────────────────────
function esc(s) {
  return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function rows(pairs) {
  return pairs.filter(Boolean).map(([l,v]) =>
    `<div class="meta-row"><div class="meta-lbl">${l}</div><div class="meta-val">${v}</div></div>`
  ).join('');
}
function tagList(tags) {
  return `<div class="tag-list">${tags.map(t=>`<span class="tag">${esc(t)}</span>`).join('')}</div>`;
}
function badge(val, kind) {
  return `<span class="badge badge-${val}">${esc(val)}</span>`;
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeDetail(); closeEdit(); }
});

// ── 초기화 ──────────────────────────────────────────────────────────
async function init() {
  showTab('library');
  loadLibrary();
  try {
    const info = await api('GET', '/api/info');
    // 생성 프로바이더 셀렉트 채우기
    const gpSel = document.getElementById('gp-provider');
    info.providers.filter(p => p.kind === 'generate').forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.name; opt.textContent = p.name + (p.authenticated ? '' : ' (미인증)');
      opt.disabled = !p.authenticated;
      gpSel.appendChild(opt);
    });
    // 가져오기 프로바이더 셀렉트
    const fpSel = document.getElementById('fp-provider');
    info.providers.filter(p => p.kind === 'fetch').forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.name; opt.textContent = p.name + (p.authenticated ? '' : ' (미인증)');
      opt.disabled = !p.authenticated;
      fpSel.appendChild(opt);
    });
    // 인증된 첫 번째 항목 기본 선택
    const firstAuth = [...fpSel.options].find(o => !o.disabled);
    if (firstAuth) firstAuth.selected = true;
  } catch(e) { console.warn('info load failed', e); }
}

init();
</script>
</body>
</html>"""


# ── Handler ─────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    store: MediaStore
    emb_store: EmbeddingStore
    media_dir: Path
    fetch_cache: _FetchCache

    def log_message(self, fmt, *args): pass  # 터미널 로그 억제

    # ── 응답 헬퍼 ──

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self._send(code, "application/json", body)

    def _err(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _read_multipart(self) -> tuple[Optional[bytes], Optional[str], dict]:
        """multipart/form-data 파싱. (file_bytes, filename, fields) 반환."""
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        raw = f"Content-Type: {ctype}\r\n\r\n".encode() + body
        msg = _email.message_from_bytes(raw)
        file_bytes, filename, fields = None, None, {}
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            cd = part.get("Content-Disposition", "")
            if "filename" in cd:
                filename = part.get_filename()
                file_bytes = part.get_payload(decode=True)
            else:
                name = part.get_param("name", header="Content-Disposition")
                if name:
                    payload = part.get_payload(decode=True)
                    if payload:
                        fields[name] = payload.decode("utf-8", errors="replace").strip()
        return file_bytes, filename, fields

    # ── 라우팅 ──

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        path, qs = p.path, urllib.parse.parse_qs(p.query)

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())
        elif path == "/api/items":
            self._api_items(qs)
        elif path == "/api/info":
            self._api_info()
        elif path.startswith("/media/"):
            self._serve_media(urllib.parse.unquote(path[7:]))
        elif path.startswith("/api/fetch/preview/"):
            self._serve_fetch_preview(urllib.parse.unquote(path[19:]))
        else:
            self._err(404, "not found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/fetch/search":
            self._api_fetch_search()
        elif path == "/api/fetch/save":
            self._api_fetch_save()
        elif path == "/api/generate":
            self._api_generate()
        elif path == "/api/add":
            self._api_add()
        else:
            self._err(404, "not found")

    def do_PATCH(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/items/"):
            self._api_item_patch(urllib.parse.unquote(path[11:]))
        else:
            self._err(404, "not found")

    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/items/"):
            self._api_item_delete(urllib.parse.unquote(path[11:]))
        else:
            self._err(404, "not found")

    # ── GET 엔드포인트 ──

    def _api_items(self, qs: dict):
        items = self.store.list_all(
            media_type=qs.get("type", [None])[0],
            source=qs.get("source", [None])[0],
            source_provider=qs.get("provider", [None])[0],
        )
        self._json([i.to_dict() for i in items])

    def _api_info(self):
        self._json({
            "db_path": str(self.store.db_path),
            "media_dir": str(self.media_dir),
            "total_items": self.store.count(),
            "providers": self._provider_info,
        })

    def _serve_media(self, item_id: str):
        item = self.store.get(item_id)
        if not item:
            return self._err(404, "item not found")
        path = Path(item.path)
        if not path.exists():
            return self._err(404, "file not found")
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def _serve_fetch_preview(self, cache_id: str):
        fm = self.fetch_cache.get(cache_id)
        if not fm:
            return self._err(404, "cache miss")
        self._send(200, fm.mime_type, fm.data)

    # ── POST 엔드포인트 ──

    def _api_fetch_search(self):
        body = self._read_json()
        provider_name = body.get("provider", "")
        query = body.get("query", "").strip()
        limit = int(body.get("limit", 6))
        media_type = body.get("media_type") or None

        cls = FETCH_PROVIDERS.get(provider_name)
        if not cls:
            return self._err(400, f"unknown provider: {provider_name}")
        try:
            fp = cls(timeout=60.0)
            results = fp.search(query=query, limit=limit, media_type=media_type)
            self._json(self.fetch_cache.put(results))
        except Exception as e:
            self._err(500, str(e))

    def _api_fetch_save(self):
        body = self._read_json()
        provider_name = body.get("provider", "")
        query = body.get("query", "")
        items_meta = body.get("items", [])

        saved = []
        for meta in items_meta:
            cid = meta.get("cache_id", "")
            fm = self.fetch_cache.get(cid)
            if not fm:
                continue
            try:
                item, is_new = save_media(
                    store=self.store, emb_store=self.emb_store,
                    data=fm.data,
                    suffix=_suffix_from_mime(fm.mime_type),
                    media_type=_type_from_mime(fm.mime_type),
                    source="fetched",
                    source_provider=provider_name,
                    source_url=fm.source_url,
                    source_id=fm.source_id,
                    prompt=query,
                    attribution=fm.attribution,
                    license=fm.license,
                    license_url=fm.license_url,
                    width=fm.width, height=fm.height,
                    skip_metadata=False,
                )
                d = item.to_dict()
                d["status"] = "skipped" if not is_new else "saved"
                saved.append(d)
                self.fetch_cache.remove(cid)
            except Exception as e:
                saved.append({"status": "error", "error": str(e), "cache_id": cid})
        self._json(saved)

    def _api_generate(self):
        body = self._read_json()
        prompt = body.get("prompt", "").strip()
        provider_name = body.get("provider", "gemini")
        model = body.get("model") or None
        aspect_ratio = body.get("aspect_ratio") or None

        cls = GEN_PROVIDERS.get(provider_name)
        if not cls:
            return self._err(400, f"unknown provider: {provider_name}")
        try:
            kwargs: dict = {"timeout": 120.0}
            if model:
                kwargs["model"] = model
            gen = cls(**kwargs)
            if not gen.check_auth():
                return self._err(401, f"{provider_name} API 키가 설정되지 않았습니다.")

            tmp_dir = self.media_dir / "generated"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            target = tmp_dir / f"{uuid.uuid4()}.png"

            result = gen.generate(prompt=prompt, output_path=target, aspect_ratio=aspect_ratio)
            item, _ = save_media(
                store=self.store, emb_store=self.emb_store,
                data=Path(result.path),
                suffix=Path(result.path).suffix,
                media_type="image",
                source="generated",
                source_provider=result.provider,
                model=result.model,
                prompt=result.prompt,
            )
            self._json(item.to_dict())
        except Exception as e:
            self._err(500, str(e))

    def _api_add(self):
        file_bytes, filename, fields = self._read_multipart()
        if not file_bytes:
            return self._err(400, "파일이 없습니다")

        suffix = Path(filename).suffix if filename else ".bin"
        tags_raw = fields.get("tags", "")
        emo_raw  = fields.get("emotions", "")
        ctx_raw  = fields.get("context", "")

        def csv(s): return [x.strip() for x in s.split(",") if x.strip()] or None

        override = MetaOverride(
            name=fields.get("name") or None,
            description=fields.get("description") or None,
            tags=csv(tags_raw),
            emotions=csv(emo_raw),
            context=csv(ctx_raw),
        )
        try:
            item, is_new = save_media(
                store=self.store, emb_store=self.emb_store,
                data=file_bytes, suffix=suffix,
                media_type=_type_from_suffix(suffix),
                source="upload",
                meta_override=override,
                skip_metadata=bool(fields.get("skip_metadata")),
            )
            d = item.to_dict()
            d["skipped"] = not is_new
            self._json(d)
        except Exception as e:
            self._err(500, str(e))

    # ── PATCH / DELETE ──

    def _api_item_patch(self, item_id: str):
        item = self.store.get(item_id)
        if not item:
            return self._err(404, "not found")
        body = self._read_json()
        if "name" in body:        item.name = body["name"]
        if "description" in body: item.description = body["description"]
        if "tags" in body:        item.tags = body["tags"]
        if "emotions" in body:    item.emotions = body["emotions"]
        if "context" in body:     item.context = body["context"]
        item.touch()
        try:
            self.store.update(item)
            self.emb_store.upsert(item)
            self._json(item.to_dict())
        except Exception as e:
            self._err(500, str(e))

    def _api_item_delete(self, item_id: str):
        item = self.store.get(item_id)
        if not item:
            return self._err(404, "not found")
        try:
            self.store.delete(item_id)
            self.emb_store.delete(item_id)
            path = Path(item.path)
            if path.exists():
                path.unlink()
            self._json({"deleted": True, "id": item_id})
        except Exception as e:
            self._err(500, str(e))


# ── 공개 API ────────────────────────────────────────────────────────────────

def _safe_auth(cls) -> bool:
    try:
        return cls(timeout=10.0).check_auth()
    except Exception:
        return False


def make_server(
    port: int,
    db_path: Optional[Path] = None,
    media_dir: Optional[Path] = None,
) -> HTTPServer:
    store = MediaStore(db_path=db_path or get_db_path())
    media = media_dir or get_media_dir()
    emb = EmbeddingStore(chroma_dir=media.parent / "chroma")
    cache = _FetchCache()

    provider_info = []
    for name, cls in GEN_PROVIDERS.items():
        provider_info.append({"name": name, "kind": "generate", "authenticated": _safe_auth(cls)})
    for name, cls in FETCH_PROVIDERS.items():
        provider_info.append({"name": name, "kind": "fetch", "authenticated": _safe_auth(cls)})

    class Handler(_Handler):
        pass

    Handler.store = store
    Handler.emb_store = emb
    Handler.media_dir = media
    Handler.fetch_cache = cache
    Handler._provider_info = provider_info

    srv = HTTPServer(("127.0.0.1", port), Handler)
    srv.allow_reuse_address = True
    return srv
