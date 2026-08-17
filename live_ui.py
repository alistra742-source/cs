"""
live_ui.py — the LIVE CONTROL overlay injected into the dashboard.

``LIVE_INJECTION`` is appended just before ``</body>`` by app.handle_root(),
so it does not touch the giant DASHBOARD_HTML string. It adds a full-screen
live-control surface (real screenshot stream + cursor + real keyboard +
fullscreen + smart address bar) that drives the SAME Camoufox page the bot
uses via the /browser/* endpoints.
"""

LIVE_INJECTION = r"""
<style>
#liveOverlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:200;padding:10px}
#liveOverlay.on{display:flex}
.lc-shell{flex:1;display:flex;flex-direction:column;gap:10px;max-width:1680px;margin:0 auto;min-height:0}
.lc-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.lc-title{font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:2px;color:#e7e7ea;font-size:13px;white-space:nowrap}
.lc-title b{color:#34d399}
.lc-bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex:1}
.lc-addr{flex:1;min-width:220px;display:flex;align-items:center;gap:8px;background:#1a1a1e;border:1px solid #34343a;border-radius:12px;padding:0 10px}
.lc-addr input{flex:1;min-width:0;background:none;border:none;outline:none;color:#e7e7ea;font-family:'JetBrains Mono',monospace;font-size:12.5px;padding:11px 0}
.lc-addr input::placeholder{color:#5c5c64}
.lc-go{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;padding:7px 12px;border-radius:8px;border:1px solid #34343a;background:#e7e7ea;color:#0a0a0b;cursor:pointer}
.lc-go:disabled{opacity:.4;cursor:not-allowed}
.lc-ico{display:inline-flex;align-items:center;justify-content:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;padding:10px 13px;border-radius:10px;border:1px solid #34343a;background:#1a1a1e;color:#8a8a92;cursor:pointer;transition:all .15s;white-space:nowrap}
.lc-ico:hover{color:#e7e7ea;border-color:#5c5c64}
.lc-ico.on{background:#e7e7ea;color:#0a0a0b;border-color:#e7e7ea}
.lc-ico:disabled{opacity:.4;cursor:not-allowed}
.lc-x{color:#f87171}
.lc-launch{background:#0f2e24;color:#6ee7b7;border-color:#1d4a3a}
.lc-close{background:#2a1212;color:#fca5a5;border-color:#5a2323}
.lc-meta{display:flex;align-items:center;gap:10px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#8a8a92;flex-wrap:wrap}
.lc-meta .grow{flex:1;min-width:0;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lc-frame{position:relative;flex:1;min-height:0;background:#050506;border:1px solid #34343a;border-radius:14px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.lc-frame.cursor-on{cursor:crosshair}
.lc-frame img{max-width:100%;max-height:100%;user-select:none;-webkit-user-drag:none;display:none}
.lc-ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#5c5c64;font-family:'JetBrains Mono',monospace;font-size:12px;text-align:center;padding:16px}
.lc-cursor{position:absolute;left:0;top:0;z-index:6;pointer-events:none;display:none;transform:translate(-4px,-4px);filter:drop-shadow(0 1px 2px rgba(0,0,0,.9))}
.lc-kbhint{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);z-index:7;display:none;align-items:center;gap:8px;background:rgba(10,10,11,.92);border:1px solid #1d4a3a;color:#6ee7b7;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1px;padding:8px 15px;border-radius:99px;pointer-events:none;white-space:nowrap}
.lc-frame.kb-on .lc-kbhint{display:flex}
.lc-foot{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.lc-tip{flex:1;min-width:220px;color:#5c5c64;font-size:11px;line-height:1.5;font-family:'JetBrains Mono',monospace}
@media(max-width:640px){.lc-title{display:none}.lc-addr{order:99;width:100%;flex-basis:100%}}
</style>
<div id="liveOverlay">
  <div class="lc-shell">
    <div class="lc-head">
      <span class="lc-title">● <b>LIVE CONTROL</b></span>
      <div class="lc-bar">
        <button class="lc-ico" id="lcBack" onclick="lcBack()" title="Back">&#9664;</button>
        <button class="lc-ico" id="lcFwd" onclick="lcFwd()" title="Forward">&#9654;</button>
        <button class="lc-ico" id="lcReload" onclick="lcReload()" title="Reload">&#8635;</button>
        <div class="lc-addr">
          <input id="lcAddr" placeholder="Search or type a URL — youtube, discord.com, anything" onkeydown="lcAddrKey(event)">
          <button class="lc-go" id="lcGo" onclick="lcGo()">GO</button>
        </div>
        <button class="lc-ico" id="lcCur" onclick="lcToggleCursor()">CURSOR</button>
        <button class="lc-ico" id="lcKb" onclick="lcToggleKeyboard()">KEYBOARD</button>
        <button class="lc-ico" onclick="lcFullscreen()" title="Fullscreen">&#9974;</button>
        <button class="lc-ico lc-x" onclick="lcClose()" title="Close">&#10005;</button>
      </div>
    </div>
    <div class="lc-meta">
      <span class="dot" id="lcDot"></span>
      <span id="lcState">connecting&hellip;</span>
      <span class="grow" id="lcTitle"></span>
    </div>
    <div class="lc-frame" id="lcFrame">
      <div class="lc-ph" id="lcPh">connecting to browser&hellip;</div>
      <img id="lcImg" draggable="false" alt="live browser">
      <div class="lc-cursor" id="lcCursor"><svg width="26" height="26" viewBox="0 0 24 24"><path fill="#fff" stroke="#0a0a0b" stroke-width="1.4" d="M5.5 3.2l13.6 8.5-6.3 1.6-2.6 6.2z"/></svg></div>
      <div class="lc-kbhint">&#9000; KEYBOARD ACTIVE — your physical keys are sent to the browser</div>
    </div>
    <div class="lc-foot">
      <button class="lc-ico lc-launch" onclick="lcLaunch(true)">LAUNCH / RECONNECT</button>
      <button class="lc-ico lc-close" onclick="lcCloseBrowser()">CLOSE BROWSER</button>
      <span class="lc-tip">This is the bot's own Camoufox — watch it work or take over. CURSOR = click the page. KEYBOARD = just type on your real keyboard (no on-screen keys). The address bar already knows "youtube" means youtube.com.</span>
    </div>
  </div>
</div>
<script>
(function(){
  window.openLive = function(){ var o=document.getElementById('liveOverlay'); if(o.classList.contains('on')){ lcClose(); } else { o.classList.add('on'); LC.start(); lcAutoLaunch(); } };
  window.closeLive = function(){ lcClose(); };
  window.openView = function(){ window.openLive(); };
  var nav = document.querySelector('nav');
  if(nav){
    var b = document.createElement('button');
    b.textContent = 'LIVE';
    b.style.borderColor = '#1d4a3a';
    b.style.color = '#6ee7b7';
    b.onclick = function(){ window.openLive(); };
    nav.insertBefore(b, nav.firstChild);
  }
})();

var LC = {worker:'B1', cursor:false, keyboard:false, timer:null, busy:false, launching:false, last:null};

function lcSmartUrl(raw){
  var t = String(raw==null?'':raw).trim();
  if(!t) return '';
  if(/^https?:\/\//i.test(t)) return t;
  if(/^[\w-]+(\.[\w-]+)+([\/?#].*)?$/i.test(t)) return 'https://' + t;
  if(/^[\w-]+$/.test(t)) return 'https://www.' + t + '.com';
  return 'https://www.google.com/search?q=' + encodeURIComponent(t);
}

function lcRender(st){
  st = st || {};
  // Keep the last good frame when a state arrives without a screenshot — a
  // transient empty capture must never blank the feed to 'waiting for frame'.
  var prev = LC.last || {};
  if(!st.screenshot && prev.screenshot && st.connected !== false){
    st.screenshot = prev.screenshot;
  }
  LC.last = st;
  var launching = LC.launching || !!st.launching || st.status === 'starting';
  var dot = document.getElementById('lcDot');
  if(dot) dot.className = 'dot' + (st.connected ? ' on' : '');
  var stEl = document.getElementById('lcState');
  if(stEl){
    if(!st.connected){
      if(st.error) stEl.textContent = 'error: ' + st.error;
      else if(launching) stEl.textContent = 'launching browser…';
      else stEl.textContent = 'browser not started — press LAUNCH';
    } else {
      stEl.textContent = 'LIVE · ' + (st.worker_id || LC.worker) + ' · ' + (st.viewport_width || '?') + '×' + (st.viewport_height || '?') + (st.error ? (' · ' + st.error) : '');
    }
  }
  var tEl = document.getElementById('lcTitle');
  if(tEl) tEl.textContent = (st.title || st.url) || '';
  var addr = document.getElementById('lcAddr');
  if(st.connected && st.url && document.activeElement !== addr) addr.value = st.url;
  var img = document.getElementById('lcImg'), ph = document.getElementById('lcPh');
  if(st.screenshot){
    var src = 'data:image/png;base64,' + st.screenshot;
    if(img.getAttribute('src') !== src) img.setAttribute('src', src);
    img.style.display = 'block';
    if(ph) ph.style.display = 'none';
  } else {
    if(img) img.style.display = 'none';
    if(ph){
      ph.style.display = 'flex';
      if(st.connected && st.error) ph.textContent = '⚠ ' + st.error;
      else if(launching) ph.textContent = 'launching browser…';
      else if(st.connected) ph.textContent = 'waiting for frame…';
      else ph.textContent = 'browser not started — press LAUNCH';
    }
  }
  var dis = !st.connected || LC.busy;
  var ids = ['lcBack','lcFwd','lcReload','lcGo'];
  for(var i=0;i<ids.length;i++){ var el=document.getElementById(ids[i]); if(el) el.disabled = dis; }
}

function lcBusy(b){ LC.busy = b; lcRender(LC.last || {}); }

function lcState(){
  var o = document.getElementById('liveOverlay');
  if(!o.classList.contains('on')) return;
  fetch('/browser/state?worker=' + LC.worker + '&t=' + Date.now())
    .then(function(r){ return r.json(); })
    .then(function(st){ lcRender(st); })
    .catch(function(){});
  LC.timer = setTimeout(lcState, 1400);
}
function lcStart(){ if(LC.timer) clearTimeout(LC.timer); LC.timer = null; lcState(); }
function lcStop(){ if(LC.timer){ clearTimeout(LC.timer); LC.timer = null; } }
function lcClose(){
  var o = document.getElementById('liveOverlay');
  o.classList.remove('on');
  lcStop();
  LC.cursor = false; LC.keyboard = false;
  var c = document.getElementById('lcCur'); if(c) c.classList.remove('on');
  var k = document.getElementById('lcKb'); if(k) k.classList.remove('on');
  var f = document.getElementById('lcFrame'); if(f){ f.classList.remove('cursor-on'); f.classList.remove('kb-on'); }
  if(document.fullscreenElement || document.webkitFullscreenElement){
    if(document.exitFullscreen) document.exitFullscreen().catch(function(){});
    else if(document.webkitExitFullscreen) document.webkitExitFullscreen();
  }
}

function lcAction(a, quiet){
  return fetch('/browser/action?worker=' + LC.worker, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(a)
  }).then(function(r){ return r.json(); }).then(function(st){
    if(!quiet) lcRender(st);
    return st;
  }).catch(function(){});
}

function lcGo(){
  var u = lcSmartUrl(document.getElementById('lcAddr').value);
  if(!u) return toast('type a URL or search');
  document.getElementById('lcAddr').value = u;
  lcBusy(true);
  fetch('/browser/navigate?worker=' + LC.worker, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url:u})})
    .then(function(r){ return r.json(); })
    .then(function(st){ lcRender(st); })
    .catch(function(){ toast('navigate failed'); })
    .finally(function(){ lcBusy(false); });
}
function lcAddrKey(e){ if(e.key === 'Enter'){ e.preventDefault(); lcGo(); } }
function lcBack(){ lcAction({action:'back'}); }
function lcFwd(){ lcAction({action:'forward'}); }
function lcReload(){ lcAction({action:'reload'}); }

function lcToggleCursor(){
  LC.cursor = !LC.cursor;
  document.getElementById('lcCur').classList.toggle('on', LC.cursor);
  document.getElementById('lcFrame').classList.toggle('cursor-on', LC.cursor);
  if(!LC.cursor) document.getElementById('lcCursor').style.display = 'none';
  toast(LC.cursor ? 'CURSOR ON — click the page to click the browser' : 'Cursor off');
}
function lcToggleKeyboard(){
  LC.keyboard = !LC.keyboard;
  document.getElementById('lcKb').classList.toggle('on', LC.keyboard);
  document.getElementById('lcFrame').classList.toggle('kb-on', LC.keyboard);
  toast(LC.keyboard ? 'KEYBOARD ON — type on your real keyboard' : 'Keyboard off');
}

function lcPoint(e){
  var img = document.getElementById('lcImg');
  var rect = img.getBoundingClientRect();
  if(rect.width <= 0 || rect.height <= 0) return {x:0, y:0};
  // Map screen click -> CSS-pixel viewport coords. The CDP mouse event
  // expects CSS pixels, but the screenshot can be device-scaled (dpr 1.25
  // makes a 2400x1350 PNG of a 1920x1080 viewport). Always scale against the
  // reported CSS viewport, never the image's natural device-pixel size.
  var last = LC.last || {};
  var iw = last.viewport_width || img.naturalWidth || 1920;
  var ih = last.viewport_height || img.naturalHeight || 1080;
  var px = (e.clientX - rect.left) / rect.width * iw;
  var py = (e.clientY - rect.top) / rect.height * ih;
  return {x: Math.max(0, Math.min(iw, Math.round(px))), y: Math.max(0, Math.min(ih, Math.round(py)))};
}

function lcFullscreen(){
  var f = document.getElementById('liveOverlay');
  if(document.fullscreenElement || document.webkitFullscreenElement){
    if(document.exitFullscreen) document.exitFullscreen().catch(function(){});
    else if(document.webkitExitFullscreen) document.webkitExitFullscreen();
    return;
  }
  if(f.requestFullscreen) f.requestFullscreen().catch(function(){});
  else if(f.webkitRequestFullscreen) f.webkitRequestFullscreen();
}

function lcLaunch(force){
  if(LC.launching) return;
  LC.launching = true;
  lcRender(LC.last || {});
  toast('Launching browser…');
  var u = lcSmartUrl(document.getElementById('lcAddr').value);
  if(!u) u = 'https://discord.com/register';
  fetch('/browser/start?worker=' + LC.worker, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({url:u, force:!!force})})
    .then(function(r){ return r.json(); })
    .then(function(st){
      lcRender(st);
      toast(st.connected ? 'browser live' : ('launch failed: ' + (st.error || 'unknown')));
    })
    .catch(function(){ toast('launch failed'); })
    .finally(function(){ LC.launching = false; lcRender(LC.last || {}); });
}
function lcAutoLaunch(){
  var last = LC.last || {};
  if(last.connected || last.launching) return;
  lcLaunch(false);
}
function lcCloseBrowser(){
  fetch('/browser/close?worker=' + LC.worker, {method:'POST'})
    .then(function(){ lcRender({connected:false}); toast('browser closed'); })
    .catch(function(){ toast('close failed'); });
}

(function(){
  var frame = document.getElementById('lcFrame');
  var overlay = document.getElementById('liveOverlay');
  overlay.addEventListener('click', function(e){ if(e.target === overlay) lcClose(); });
  frame.addEventListener('click', function(e){
    if(!LC.cursor) return;
    var p = lcPoint(e);
    lcAction({action:'click', x:p.x, y:p.y});
  });
  frame.addEventListener('mousemove', function(e){
    if(!LC.cursor){ document.getElementById('lcCursor').style.display = 'none'; return; }
    var fr = frame.getBoundingClientRect();
    var cur = document.getElementById('lcCursor');
    cur.style.display = 'block';
    cur.style.left = (e.clientX - fr.left) + 'px';
    cur.style.top = (e.clientY - fr.top) + 'px';
  });
  frame.addEventListener('wheel', function(e){
    if(!LC.keyboard && !LC.cursor) return;
    e.preventDefault();
    lcAction({action:'scroll', delta_y: (e.deltaY > 0 ? 160 : -160)}, true);
  }, {passive:false});
  document.addEventListener('keydown', function(e){
    if(e.key === 'Escape'){
      var o = document.getElementById('liveOverlay');
      if(o && o.classList.contains('on')){
        var ae = document.activeElement;
        if(ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA')){ ae.blur(); return; }
        e.preventDefault();
        lcClose();
      }
      return;
    }
    if(!LC.keyboard) return;
    var o = document.getElementById('liveOverlay');
    if(!o.classList.contains('on')) return;
    var el = document.activeElement;
    if(el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')) return;
    if(e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key;
    if(!k) return;
    var special = ['Enter','Backspace','Tab','Delete','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End','PageUp','PageDown'];
    if(k.length > 1 && special.indexOf(k) === -1) return;
    e.preventDefault();
    lcAction({action:'key', key:k}, true);
  });
})();
</script>
"""
