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
.live-frame{position:relative;min-height:240px;max-height:calc(100vh - 165px);overflow:auto;display:flex;align-items:flex-start;justify-content:center;background:#050506;border:1px solid #34343a;border-radius:12px}
.live-frame img{display:none;width:100%;height:auto;object-fit:contain}
.live-placeholder{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#8a8a92;font-family:'JetBrains Mono','Courier New',monospace;font-size:12px;line-height:1.6}
.live-foot{color:#5c5c64;font-family:'JetBrains Mono','Courier New',monospace;font-size:10px;letter-spacing:.3px}
@media(max-width:640px){#liveOverlay{padding:10px}.live-modal{padding:12px;border-radius:13px}.live-state{display:none}.live-frame{min-height:220px}}
</style>
<div id="liveOverlay" role="dialog" aria-modal="true" aria-labelledby="liveTitle">
  <div class="live-modal">
    <div class="live-head">
      <div id="liveTitle" class="live-title"><b>LIVE</b> CAMERA</div>
      <div id="liveState" class="live-state">Waiting for generator</div>
      <button class="live-close" type="button" onclick="closeLive()" aria-label="Close live camera">CLOSE</button>
    </div>
    <div class="live-frame">
      <div id="livePlaceholder" class="live-placeholder">Start the generator to view its camera.</div>
      <img id="liveImage" alt="Latest generator camera frame">
    </div>
    <div class="live-foot">Full-page camera — refreshes every 3 seconds. Scroll the frame to see the whole page.</div>
  </div>
</div>
<script>
var LC={worker:'B1',timer:null};

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
function lcLoadFrame(){
  var img=document.getElementById('liveImage');
  var ph=document.getElementById('livePlaceholder');
  if(!img)return;
  img.onload=function(){
    img.style.display='block';
    if(ph)ph.style.display='none';
  };
  img.onerror=function(){
    lcShowPlaceholder('Waiting for the first camera frame.');
  };
  img.src='/latest?worker='+encodeURIComponent(LC.worker)+'&t='+Date.now();
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
      lcSetStatus(state==='running'||state==='starting'?'Live · '+state:'Camera · '+state);
      if(state==='idle'||state==='stopped'){
        lcShowPlaceholder('Start the generator to view its camera.');
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
}
window.openLive=openLive;
window.closeLive=closeLive;
(function(){
  var overlay=document.getElementById('liveOverlay');
  if(overlay)overlay.addEventListener('click',function(e){if(e.target===overlay)closeLive();});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLive();});
})();
</script>
'''
