const video = document.getElementById('video');
const camwrap = document.getElementById('camwrap');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const screen = document.getElementById('screen');
const sctx = screen.getContext('2d');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnCalib = document.getElementById('btnCalib');
const btnCapture = document.getElementById('btnCapture');
const btnFit = document.getElementById('btnFit');
const btnReset = document.getElementById('btnReset');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const calibEl = document.getElementById('calib');
const warnEl = document.getElementById('warn');

const calibOverlay = document.getElementById('calibOverlay');
const calibCanvas = document.getElementById('calibCanvas');
const cctx = calibCanvas.getContext('2d');
const calibProgressEl = document.getElementById('calibProgress');
const btnOverlayCapture = document.getElementById('btnOverlayCapture');
const btnOverlayExit = document.getElementById('btnOverlayExit');

let running = false;
let calibMode = false;
let calibIdx = 0;
let smoothScreen = null;
let smoothDir = null;
const smoothAlpha = 0.35;
let lastLoopTs = null;

const edge = 0.02;
const calibPoints = [
  [edge, edge], [0.50, edge], [1.0-edge, edge],
  [edge, 0.50], [0.50, 0.50], [1.0-edge, 0.50],
  [edge, 1.0-edge], [0.50, 1.0-edge], [1.0-edge, 1.0-edge],
];
const captureBurst = 3;

const off = document.createElement('canvas');
const offCtx = off.getContext('2d');

function drawArrow(ctx2d, x0, y0, x1, y1) {
  ctx2d.beginPath();
  ctx2d.moveTo(x0, y0);
  ctx2d.lineTo(x1, y1);
  ctx2d.stroke();

  const angle = Math.atan2(y1 - y0, x1 - x0);
  const head = 12;
  ctx2d.beginPath();
  ctx2d.moveTo(x1, y1);
  ctx2d.lineTo(x1 - head * Math.cos(angle - Math.PI / 6), y1 - head * Math.sin(angle - Math.PI / 6));
  ctx2d.lineTo(x1 - head * Math.cos(angle + Math.PI / 6), y1 - head * Math.sin(angle + Math.PI / 6));
  ctx2d.closePath();
  ctx2d.stroke();
}

function setCalibOverlayActive(active) {
  if (!calibOverlay) return;
  if (active) calibOverlay.classList.add('active');
  else calibOverlay.classList.remove('active');
}

function isCalibOverlayActive() {
  return calibOverlay && calibOverlay.classList.contains('active');
}

function resizeCalibOverlay() {
  if (!calibCanvas) return;
  calibCanvas.width = window.innerWidth;
  calibCanvas.height = window.innerHeight;
}

function drawCalibOverlay() {
  if (!isCalibOverlayActive() || !calibMode) return;
  const w = calibCanvas.width;
  const h = calibCanvas.height;
  cctx.clearRect(0, 0, w, h);

  cctx.strokeStyle = 'rgba(255,255,255,0.03)';
  cctx.lineWidth = 1;
  const step = 80;
  for (let x = 0; x <= w; x += step) { cctx.beginPath(); cctx.moveTo(x, 0); cctx.lineTo(x, h); cctx.stroke(); }
  for (let y = 0; y <= h; y += step) { cctx.beginPath(); cctx.moveTo(0, y); cctx.lineTo(w, y); cctx.stroke(); }

  const [tx, ty] = calibPoints[calibIdx];
  const x = tx * w;
  const y = ty * h;

  cctx.strokeStyle = '#ff8000';
  cctx.lineWidth = 8;
  cctx.beginPath(); cctx.arc(x, y, 34, 0, Math.PI*2); cctx.stroke();
  cctx.lineWidth = 4;
  cctx.beginPath(); cctx.arc(x, y, 10, 0, Math.PI*2); cctx.stroke();

  cctx.strokeStyle = 'rgba(255,128,0,0.35)';
  cctx.lineWidth = 2;
  cctx.beginPath(); cctx.moveTo(x - 60, y); cctx.lineTo(x + 60, y); cctx.stroke();
  cctx.beginPath(); cctx.moveTo(x, y - 60); cctx.lineTo(x, y + 60); cctx.stroke();

  if (calibProgressEl) calibProgressEl.textContent = `Point ${calibIdx+1}/${calibPoints.length}`;
}

function drawCameraOverlay(resp) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (resp && resp.ok && resp.face) {
    const b = resp.face.bbox;
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.x0, b.y0, (b.x1-b.x0), (b.y1-b.y0));

    const le = resp.eyes.left_bbox;
    const re = resp.eyes.right_bbox;
    ctx.strokeStyle = '#ffe400';
    ctx.strokeRect(le.x0, le.y0, (le.x1-le.x0), (le.y1-le.y0));
    ctx.strokeRect(re.x0, re.y0, (re.x1-re.x0), (re.y1-re.y0));

    if (resp.pupil) {
      const pl = resp.pupil.left;
      const pr = resp.pupil.right;
      ctx.strokeStyle = '#ff3c3c';
      ctx.lineWidth = 3;
      if (pl) {
        const x = le.x0 + pl.x * (le.x1 - le.x0);
        const y = le.y0 + pl.y * (le.y1 - le.y0);
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.stroke();
      }
      if (pr) {
        const x = re.x0 + pr.x * (re.x1 - re.x0);
        const y = re.y0 + pr.y * (re.y1 - re.y0);
        ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.stroke();
      }
    }

    if (resp.gaze_dir) {
      const dx = resp.gaze_dir.x;
      const dy = resp.gaze_dir.y;
      const conf = resp.gaze_dir.conf || 0.0;

      const cx = (b.x0 + b.x1) * 0.5;
      const cy = (b.y0 + b.y1) * 0.5;
      const scale = Math.max(30, Math.min((b.x1 - b.x0), (b.y1 - b.y0)) * 0.35);
      const ex = cx + dx * scale;
      const ey = cy - dy * scale;

      ctx.save();
      ctx.globalAlpha = Math.max(0.25, Math.min(1.0, conf));
      ctx.strokeStyle = '#ff3c3c';
      ctx.lineWidth = 4;
      drawArrow(ctx, cx, cy, ex, ey);
      ctx.restore();
    }
  }
}

function drawScreen(resp) {
  const w = screen.width;
  const h = screen.height;
  sctx.clearRect(0, 0, w, h);
  sctx.fillStyle = '#0c0f16';
  sctx.fillRect(0, 0, w, h);
  sctx.strokeStyle = '#555';
  sctx.lineWidth = 2;
  sctx.strokeRect(1, 1, w-2, h-2);

  if (calibMode) {
    const [tx, ty] = calibPoints[calibIdx];
    const x = tx * w;
    const y = ty * h;
    sctx.strokeStyle = '#ff8000';
    sctx.lineWidth = 4;
    sctx.beginPath(); sctx.arc(x, y, 8, 0, Math.PI*2); sctx.stroke();
  }

  const hasScreen = !!(resp && resp.ok && resp.screen);
  const hasDir = !!(resp && resp.ok && resp.gaze_dir);

  if (hasScreen) {
    const x = resp.screen.x_norm;
    const y = resp.screen.y_norm;
    if (!smoothScreen) smoothScreen = {x, y};
    smoothScreen.x = smoothAlpha * x + (1 - smoothAlpha) * smoothScreen.x;
    smoothScreen.y = smoothAlpha * y + (1 - smoothAlpha) * smoothScreen.y;

    const px = smoothScreen.x * w;
    const py = smoothScreen.y * h;
    sctx.strokeStyle = '#ff3c3c';
    sctx.lineWidth = 4;
    sctx.beginPath(); sctx.arc(px, py, 8, 0, Math.PI*2); sctx.stroke();
    smoothDir = null;
  } else {
    smoothScreen = null;

    if (hasDir) {
      const dx = resp.gaze_dir.x;
      const dy = resp.gaze_dir.y;
      if (!smoothDir) smoothDir = {x: dx, y: dy};
      smoothDir.x = smoothAlpha * dx + (1 - smoothAlpha) * smoothDir.x;
      smoothDir.y = smoothAlpha * dy + (1 - smoothAlpha) * smoothDir.y;

      const cx = w * 0.5;
      const cy = h * 0.5;
      const scale = Math.min(w, h) * 0.35;
      const ex = cx + smoothDir.x * scale;
      const ey = cy - smoothDir.y * scale;

      sctx.strokeStyle = '#ff3c3c';
      sctx.lineWidth = 4;
      sctx.beginPath(); sctx.arc(cx, cy, 5, 0, Math.PI*2); sctx.stroke();
      sctx.beginPath(); sctx.moveTo(cx, cy); sctx.lineTo(ex, ey); sctx.stroke();
    } else {
      smoothDir = null;
      sctx.fillStyle = '#b6b6b9';
      sctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      sctx.fillText('No calibration (Calibrate) / No face', 10, 20);
    }
  }
}

async function sendPredict() {
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  const blob = await new Promise(resolve => off.toBlob(resolve, 'image/jpeg', 0.8));

  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg');
  const r = await fetch('/api/predict', { method: 'POST', body: fd });
  return await r.json();
}

async function captureCalibPoint() {
  const [tx, ty] = calibPoints[calibIdx];
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  const blob = await new Promise(resolve => off.toBlob(resolve, 'image/jpeg', 0.8));
  const fd = new FormData();
  fd.append('file', blob, 'frame.jpg');
  fd.append('target_x', String(tx));
  fd.append('target_y', String(ty));
  const r = await fetch('/api/calibration/capture', { method: 'POST', body: fd });
  return await r.json();
}

async function doCapture() {
  let okCount = 0;
  for (let i = 0; i < captureBurst; i++) {
    const resp = await captureCalibPoint();
    if (!resp.ok) console.warn('capture failed', resp);
    else okCount += 1;
    await new Promise(r => setTimeout(r, 120));
  }
  if (okCount === 0) {
    alert('Capture failed (no valid frames). Try again with better lighting / eyes open.');
    return;
  }

  calibIdx += 1;
  if (calibIdx >= calibPoints.length) {
    calibMode = false;
    btnCapture.disabled = true;
    btnFit.disabled = false;
    setCalibOverlayActive(false);
    try { if (document.fullscreenElement) await document.exitFullscreen(); } catch (e) {}
  } else {
    drawCalibOverlay();
  }
}

async function loop() {
  while (running) {
    const now = performance.now();
    if (lastLoopTs !== null && fpsEl) {
      const fps = 1000.0 / Math.max(1.0, (now - lastLoopTs));
      fpsEl.textContent = fps.toFixed(1);
    }
    lastLoopTs = now;

    try {
      const resp = await sendPredict();
      drawCameraOverlay(resp);
      drawScreen(resp);
      drawCalibOverlay();
      statusEl.textContent = JSON.stringify(resp, null, 2);
      if (calibEl) calibEl.textContent = resp.calibrated ? 'yes' : 'no';
      if (warnEl) warnEl.textContent = (resp.warnings && resp.warnings.length) ? resp.warnings.join(', ') : '—';
    } catch (e) {
      statusEl.textContent = 'ERROR: ' + e;
      if (calibEl) calibEl.textContent = '—';
      if (warnEl) warnEl.textContent = '—';
    }
    await new Promise(r => setTimeout(r, 120));
  }
}

btnStart.onclick = async () => {
  running = true;
  btnStart.disabled = true;
  btnStop.disabled = false;
  loop();
};

btnStop.onclick = () => {
  running = false;
  btnStart.disabled = false;
  btnStop.disabled = true;
};

btnReset.onclick = async () => {
  await fetch('/api/calibration/reset', { method: 'POST' });
  smoothScreen = null;
  smoothDir = null;
};

btnCalib.onclick = async () => {
  calibMode = true;
  calibIdx = 0;
  btnCapture.disabled = false;
  btnFit.disabled = true;

  try { await document.documentElement.requestFullscreen(); } catch (e) {}
  resizeCalibOverlay();
  setCalibOverlayActive(true);
  drawCalibOverlay();
};

btnCapture.onclick = async () => { await doCapture(); };
btnOverlayCapture.onclick = async () => { await doCapture(); };
btnOverlayExit.onclick = async () => {
  calibMode = false;
  btnCapture.disabled = true;
  setCalibOverlayActive(false);
  try { if (document.fullscreenElement) await document.exitFullscreen(); } catch (e) {}
};

btnFit.onclick = async () => {
  const r = await fetch('/api/calibration/fit', { method: 'POST' });
  const resp = await r.json();
  if (!resp.ok) alert('Fit failed: ' + JSON.stringify(resp));
  btnFit.disabled = true;
};

window.addEventListener('resize', () => {
  if (isCalibOverlayActive()) {
    resizeCalibOverlay();
    drawCalibOverlay();
  }
});

document.addEventListener('keydown', (e) => {
  if (!isCalibOverlayActive()) return;
  if (e.code === 'Space') {
    e.preventDefault();
    doCapture();
  } else if (e.code === 'Escape') {
    e.preventDefault();
    calibMode = false;
    btnCapture.disabled = true;
    setCalibOverlayActive(false);
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  }
});

(async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await video.play();
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const capW = 640;
  const capH = Math.max(240, Math.round(capW * (vh / vw)));

  video.width = capW;
  video.height = capH;
  overlay.width = capW;
  overlay.height = capH;
  off.width = capW;
  off.height = capH;

  camwrap.style.width = capW + 'px';
  camwrap.style.height = capH + 'px';
  statusEl.textContent = 'ready';
})();

