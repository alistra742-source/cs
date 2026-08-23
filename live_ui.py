"""Lightweight live-camera modal injected into the dashboard."""

LIVE_INJECTION = r'''
<style>
#liveOverlay{display:none;position:fixed;inset:0;z-index:200;padding:18px;background:rgba(0,0,0,.72);backdrop-filter:blur(4px);align-items:center;justify-content:center}
#liveOverlay.on{display:flex}
.live-modal{width:min(1100px,100%);max-height:min(820px,calc(100vh - 36px));display:flex;flex-direction:column;gap:12px;padding:16px;background:#131316;border:1px solid #34343a;border-radius:16px;box-shadow:0 26px 80px rgba(0,0,0,.62)}
.live-head{display:flex;align-items:center;gap:10px;min-width:0}
.live-title{font-family:'JetBrains Mono','Courier New',monospace;font-size:13px;font-weight:700;letter-spacing:1.5px;color:#e7e7ea}
.live-title b{color:#34d399}
.live-state{margin-left:auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#8a8a92;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px}
.live-close{padding:7px 11px;border-radius:9px;border:1px solid #5a2323;background:#2a1212;color:#fca5a5;cursor:pointer;font-weight:700}
.live-frame{position:relative;min-height:240px;max-height:calc(100vh - 220px);overflow:auto;display:flex;align-items:flex-start;justify-content:center;background:#050506;border:1px solid #34343a;border-radius:12px}
.live-frame img#liveImage{display:none;width:100%;height:auto;object-fit:contain;user-select:none;-webkit-user-drag:none;touch-action:none}
.live-placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#8a8a92;font-family:'JetBrains Mono','Courier New',monospace;font-size:12px;line-height:1.6}
.live-hud{position:absolute;left:10px;top:10px;z-index:3;padding:4px 8px;border-radius:7px;background:rgba(6,6,8,.78);border:1px solid #2a2a32;color:#34d399;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;letter-spacing:.2px;pointer-events:none}
.live-last{position:absolute;right:10px;top:10px;z-index:3;max-width:62%;padding:4px 8px;border-radius:7px;background:rgba(6,6,8,.78);border:1px solid #2a2a32;color:#e7e7ea;font-family:'JetBrains Mono','Courier New',monospace;font-size:10px;pointer-events:none;text-align:right}
.live-trail{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2}
.live-mark{position:absolute;z-index:4;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #34d399;border-radius:50%;box-shadow:0 0 0 2px rgba(52,211,153,.25);pointer-events:none;display:none}
.live-mark.on{display:block}
.live-challenge-wrap{display:none;gap:10px;align-items:stretch}
.live-challenge-wrap.on{display:flex}
.live-challenge-shot{flex:0 0 auto;max-width:220px;max-height:160px;border-radius:10px;border:1px solid #33374b;background:#050506;object-fit:contain}
.live-pointer-log{flex:1;min-width:0;max-height:160px;overflow:auto;padding:8px 10px;background:#08080a;border:1px solid #26262b;border-radius:10px;color:#a1a1aa;font-family:'JetBrains Mono','Courier New',monospace;font-size:11px;line-height:1.55;white-space:pre-wrap}
.live-foot{color:#5c5c64;font-family:'JetBrains Mono','Courier New',monospace;font-size:10px;letter-spacing:.3px}
@media(max-width:640px){#liveOverlay{padding:10px}.live-modal{padding:12px;border-radius:13px}.live-state{display:none}.live-frame{min-height:220px}.live-challenge-wrap.on{flex-direction:column}.live-challenge-shot{max-width:100%}}
</style>
<div id="liveOverlay" role="dialog" aria-modal="true" aria-labelledby="liveTitle">
  <div class="live-modal">
    <div class="live-head">
      <div id="liveTitle" class="live-title"><b>LIVE</b> CAMERA</div>
      <div id="liveState" class="live-state">Waiting for browser</div>
      <button class="live-close" type="button" onclick="closeLive()" aria-label="Close live camera">CLOSE</button>
    </div>
    <div class="live-frame" id="liveFrame">
      <div id="livePlaceholder" class="live-placeholder">Start the browser runner to view its camera.</div>
      <img id="liveImage" alt="Latest real Chrome camera frame" draggable="false">
      <svg id="liveTrail" class="live-trail" aria-hidden="true"></svg>
      <div id="liveMark" class="live-mark" aria-hidden="true"></div>
      <div id="liveHud" class="live-hud">—</div>
      <div id="liveLast" class="live-last"></div>
    </div>
    <div id="liveChallengeWrap" class="live-challenge-wrap">
      <img id="liveChallengeImg" class="live-challenge-shot" alt="Latest hCaptcha challenge screenshot">
      <div id="livePointerLog" class="live-pointer-log"></div>
    </div>
    <div class="live-foot">Real Chrome camera — refreshes every 3 seconds. Click or drag the frame to control Chrome. Click and drag coordinates are logged.</div>
  </div>
</div>
<script>
var LC={worker:'B1',timer:null,interactive:false,dsf:1,drag:null,lastSrc:'',lastChallenge:''};

function lcSetStatus(message){
  var el=document.getElementById('liveState');
  if(el)el.textContent=message;
}
function lcShowPlaceholder(message){
  var img=document.getElementById('liveImage');
  var ph=document.getElementById('livePlaceholder');
  if(img)img.style.display='none';
  if(ph){ph.textContent=message;ph.style.display='flex';}
}
function lcSetImage(src){
  var img=document.getElementById('liveImage');
  var ph=document.getElementById('livePlaceholder');
  if(!img||!src)return;
  if(src===LC.lastSrc && img.style.display==='block')return;
  img.onload=function(){
    img.style.display='block';
    if(ph)ph.style.display='none';
  };
  img.onerror=function(){lcShowPlaceholder('Waiting for the first camera frame.');};
  var next=src.indexOf('data:image/')===0?src:'data:image/png;base64,'+src;
  LC.lastSrc=src;
  img.src=next;
}
function lcSetHud(text){
  var el=document.getElementById('liveHud');
  if(el)el.textContent=text||'—';
}
function lcSetLast(text){
  var el=document.getElementById('liveLast');
  if(el)el.textContent=text||'';
}
function lcFmt(n){return Math.round(Number(n)||0);}
function lcDescribe(p){
  if(!p||!p.kind)return '';
  if(p.kind==='click')return 'clicked '+lcFmt(p.x)+', '+lcFmt(p.y);
  if(p.kind==='drag')return 'drag '+lcFmt(p.x1)+','+lcFmt(p.y1)+' → '+lcFmt(p.x2)+','+lcFmt(p.y2);
  if(p.kind==='mousedown'||p.kind==='mouseup'||p.kind==='mousemove')
    return p.kind+' '+lcFmt(p.x)+', '+lcFmt(p.y);
  return p.kind;
}
function lcRenderPointerLog(items){
  var box=document.getElementById('livePointerLog');
  var wrap=document.getElementById('liveChallengeWrap');
  var shot=document.getElementById('liveChallengeImg');
  var hasShot=!!(shot&&shot.getAttribute('src'));
  var rows=(items||[]).slice(-12).map(function(p){
    return (p.t?p.t+' ':'')+lcDescribe(p);
  }).filter(Boolean);
  if(box)box.textContent=rows.length?rows.join('\n'):'Clicks and drags will appear here with page coordinates.';
  if(wrap)wrap.classList.toggle('on', hasShot || rows.length>0);
}
function lcSetChallenge(src){
  var img=document.getElementById('liveChallengeImg');
  var wrap=document.getElementById('liveChallengeWrap');
  if(!img||!src)return;
  if(src===LC.lastChallenge) return;
  LC.lastChallenge=src;
  img.src=src.indexOf('data:image/')===0?src:'data:image/png;base64,'+src;
  if(wrap)wrap.classList.add('on');
}
function lcPageXY(event){
  var img=document.getElementById('liveImage');
  if(!img||!img.naturalWidth||!img.naturalHeight)return null;
  var rect=img.getBoundingClientRect();
  if(!rect.width||!rect.height)return null;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  return {
    x:(event.clientX-rect.left)*(img.naturalWidth/rect.width)/dsf,
    y:(event.clientY-rect.top)*(img.naturalHeight/rect.height)/dsf
  };
}
function lcMarkAt(x,y){
  var img=document.getElementById('liveImage');
  var mark=document.getElementById('liveMark');
  if(!img||!mark||!img.naturalWidth)return;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  mark.style.left=(img.offsetLeft+(Number(x)*dsf)*(img.clientWidth/img.naturalWidth))+'px';
  mark.style.top=(img.offsetTop+(Number(y)*dsf)*(img.clientHeight/img.naturalHeight))+'px';
  mark.classList.add('on');
}
function lcClearTrail(){
  var svg=document.getElementById('liveTrail');
  if(svg){while(svg.firstChild)svg.removeChild(svg.firstChild);}
}
function lcDrawTrail(x1,y1,x2,y2){
  var img=document.getElementById('liveImage');
  var svg=document.getElementById('liveTrail');
  if(!img||!svg||!img.naturalWidth)return;
  var dsf=Number(LC.dsf)||1;
  if(dsf<=0)dsf=1;
  var sx=img.offsetLeft+(x1*dsf)*(img.clientWidth/img.naturalWidth);
  var sy=img.offsetTop+(y1*dsf)*(img.clientHeight/img.naturalHeight);
  var ex=img.offsetLeft+(x2*dsf)*(img.clientWidth/img.naturalWidth);
  var ey=img.offsetTop+(y2*dsf)*(img.clientHeight/img.naturalHeight);
  svg.setAttribute('viewBox','0 0 '+img.offsetWidth+' '+img.offsetHeight);
  svg.setAttribute('width',img.offsetWidth);
  svg.setAttribute('height',img.offsetHeight);
  lcClearTrail();
  var line=document.createElementNS('http://www.w3.org/2000/svg','line');
  line.setAttribute('x1',sx);line.setAttribute('y1',sy);
  line.setAttribute('x2',ex);line.setAttribute('y2',ey);
  line.setAttribute('stroke','#34d399');
  line.setAttribute('stroke-width','2');
  line.setAttribute('stroke-linecap','round');
  svg.appendChild(line);
}
function lcApplyState(st){
  if(!st)return;
  if(st.device_scale_factor)LC.dsf=Number(st.device_scale_factor)||1;
  if(st.screenshot)lcSetImage(st.screenshot);
  if(st.challenge_screenshot)lcSetChallenge(st.challenge_screenshot);
  if(st.last_pointer){
    lcSetLast(lcDescribe(st.last_pointer));
    if(st.last_pointer.kind==='click')lcMarkAt(st.last_pointer.x,st.last_pointer.y);
    if(st.last_pointer.kind==='drag')lcMarkAt(st.last_pointer.x2,st.last_pointer.y2);
  }
  if(st.pointer_log)lcRenderPointerLog(st.pointer_log);
}
function lcLoadFrame(){
  fetch('/browser/state?worker='+encodeURIComponent(LC.worker)+'&t='+Date.now())
    .then(function(r){if(!r.ok)throw new Error('browser state unavailable');return r.json();})
    .then(function(st){
      if(st&&st.screenshot)lcApplyState(st);
      else if(!st||!st.connected)lcShowPlaceholder('Waiting for the first camera frame.');
      else lcApplyState(st);
    })
    .catch(function(){lcShowPlaceholder('Waiting for the first camera frame.');});
}
function lcSend(body){
  return fetch('/browser/action?worker='+encodeURIComponent(LC.worker),{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  }).then(function(r){return r.json();}).then(function(st){
    lcApplyState(st);
    return st;
  });
}
function lcOnPointerDown(event){
  if(!LC.interactive)return;
  if(event.button!==undefined && event.button!==0)return;
  var p=lcPageXY(event);
  if(!p)return;
  event.preventDefault();
  LC.drag={x1:p.x,y1:p.y,x2:p.x,y2:p.y,moved:false};
  try{event.currentTarget.setPointerCapture(event.pointerId);}catch(e){}
  lcSetHud(lcFmt(p.x)+', '+lcFmt(p.y));
  lcMarkAt(p.x,p.y);
}
function lcOnPointerMove(event){
  var p=lcPageXY(event);
  if(p)lcSetHud(lcFmt(p.x)+', '+lcFmt(p.y));
  if(!LC.drag||!p)return;
  event.preventDefault();
  LC.drag.x2=p.x;LC.drag.y2=p.y;
  var dx=p.x-LC.drag.x1,dy=p.y-LC.drag.y1;
  if((dx*dx+dy*dy)>36)LC.drag.moved=true;
  if(LC.drag.moved){
    lcDrawTrail(LC.drag.x1,LC.drag.y1,p.x,p.y);
    lcSetLast('dragging '+lcFmt(LC.drag.x1)+','+lcFmt(LC.drag.y1)+' → '+lcFmt(p.x)+','+lcFmt(p.y));
  }
}
function lcOnPointerUp(event){
  if(!LC.drag)return;
  event.preventDefault();
  var p=lcPageXY(event)||{x:LC.drag.x2,y:LC.drag.y2};
  var start=LC.drag;
  LC.drag=null;
  lcClearTrail();
  var send;
  if(start.moved){
    lcSetLast('drag '+lcFmt(start.x1)+','+lcFmt(start.y1)+' → '+lcFmt(p.x)+','+lcFmt(p.y));
    lcMarkAt(p.x,p.y);
    send=lcSend({action:'drag',x1:start.x1,y1:start.y1,x2:p.x,y2:p.y});
  }else{
    lcSetLast('clicked '+lcFmt(p.x)+', '+lcFmt(p.y));
    lcMarkAt(p.x,p.y);
    send=lcSend({action:'click',x:p.x,y:p.y});
  }
  send.catch(function(){lcSetStatus('Manual pointer action failed');});
}
function lcRefresh(){
  var overlay=document.getElementById('liveOverlay');
  if(!overlay||!overlay.classList.contains('on'))return;
  fetch('/status?t='+Date.now())
    .then(function(r){if(!r.ok)throw new Error('status unavailable');return r.json();})
    .then(function(data){
      var workers=data.workers||[];
      var worker=workers.find(function(w){return w.id===LC.worker;})||{};
      var state=worker.status||'idle';
      LC.interactive=(state==='demo'||state==='running'||state==='done');
      var liveImg=document.getElementById('liveImage');
      if(liveImg)liveImg.style.cursor=LC.interactive?'crosshair':'default';
      lcSetStatus(state==='running'||state==='starting'?'Live · '+state:
        (state==='demo'?'Demo · click or drag the frame':'Camera · '+state));
      if(state==='idle'||state==='stopped'){
        lcShowPlaceholder('Start the browser runner to view its camera.');
      }else{
        lcLoadFrame();
      }
    })
    .catch(function(){lcSetStatus('Camera status unavailable');});
  LC.timer=setTimeout(lcRefresh,3000);
}
function openLive(){
  var overlay=document.getElementById('liveOverlay');
  if(!overlay)return;
  overlay.classList.add('on');
  if(LC.timer)clearTimeout(LC.timer);
  lcRefresh();
}
function closeLive(){
  var overlay=document.getElementById('liveOverlay');
  if(overlay)overlay.classList.remove('on');
  if(LC.timer){clearTimeout(LC.timer);LC.timer=null;}
  LC.drag=null;
  lcClearTrail();
}
window.openLive=openLive;
window.closeLive=closeLive;
(function(){
  var overlay=document.getElementById('liveOverlay');
  if(overlay)overlay.addEventListener('click',function(e){if(e.target===overlay)closeLive();});
  var img=document.getElementById('liveImage');
  if(img){
    img.addEventListener('pointerdown',lcOnPointerDown);
    img.addEventListener('pointermove',lcOnPointerMove);
    img.addEventListener('pointerup',lcOnPointerUp);
    img.addEventListener('pointercancel',lcOnPointerUp);
    img.addEventListener('dragstart',function(e){e.preventDefault();});
  }
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLive();});
})();
</script>
'''
