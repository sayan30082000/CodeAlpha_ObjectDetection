// Browser version of detect_track.py: video input -> YOLO11 -> SORT -> drawing.
const $ = (id) => document.getElementById(id);
const view = $("view");
const g = view.getContext("2d");
const video = $("video");
const INFO_BAR = 34;

let detector = null;
let names = [];
let tracker = null;
let trails = new Map();
let running = null;      // "webcam" | "file" | "demo" | null
let stream = null;
let demo = null;         // { img, canvas, frame }
let fps = 0;

// ---------------------------------------------------------------- setup
async function init() {
  const meta = await fetch("classes.json").then((r) => r.json());
  names = meta.names;
  detector = await YoloDetector.load("yolo11n.onnx", meta.imgsz, setStatus);
  for (const id of ["btn-webcam", "btn-demo", "file"]) $(id).disabled = false;
  $("file-label").setAttribute("aria-disabled", "false");
  setStatus(`Model ready (running on ${detector.backend}). Start the webcam, open a video file, or try the demo scene.`);
  drawIdle("Choose a video source above");
}

function setStatus(text) {
  $("status").textContent = text;
}

function colorForId(id) {
  // golden-ratio hue spacing: every track ID gets a distinct, stable colour
  const hue = ((id * 0.618033988749895) % 1) * 360;
  return `hsl(${hue}, 85%, 55%)`;
}

function selectedClasses() {
  const value = $("classes").value;
  if (!value) return null;
  return new Set(value.split(",").map((n) => names.indexOf(n.trim())).filter((i) => i >= 0));
}

// ---------------------------------------------------------------- sources
function resetTracking() {
  SortTracker.resetIds();
  tracker = new SortTracker.Sort({ maxAge: 15, minHits: 3, iouThreshold: 0.3 });
  trails = new Map();
  fps = 0;
}

async function startWebcam() {
  stop();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
  } catch (err) {
    setStatus(`Could not open the webcam: ${err.message}. You can try the demo scene or a video file instead.`);
    return;
  }
  video.srcObject = stream;
  video.loop = false;
  await video.play();
  begin("webcam");
}

async function startFile(file) {
  stop();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  await video.play();
  begin("file");
}

async function startDemo() {
  stop();
  if (!demo) {
    const img = new Image();
    img.src = "demo.jpg";
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    demo = { img, canvas, frame: 0 };
  }
  demo.frame = 0;
  begin("demo");
}

window.addEventListener("pagehide", stop);

// The demo scene: a slow "camera pan" across a street photo (same idea as make_sample_video.py).
function demoFrame() {
  const { img, canvas } = demo;
  const frames = 240;
  const t = (Math.sin((demo.frame / frames) * Math.PI * 2 - Math.PI / 2) + 1) / 2; // 0 -> 1 -> 0
  const cropW = img.width * (0.72 + 0.23 * t);
  const cropH = Math.min(cropW * 0.75, img.height);
  const x = (img.width - cropW) * t;
  const y = (img.height - cropH) * (0.35 + 0.3 * Math.sin(Math.PI * t));
  canvas.getContext("2d").drawImage(img, x, y, cropW, cropH, 0, 0, canvas.width, canvas.height);
  demo.frame += 1;
  return canvas;
}

function begin(kind) {
  resetTracking();
  running = kind;
  $("btn-stop").disabled = false;
  setStatus(kind === "webcam" ? "Webcam running — move around!" : kind === "demo" ? "Demo scene running." : "Playing video file.");
  loop();
}

function stop() {
  running = null;
  $("btn-stop").disabled = true;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.pause();
}

// ---------------------------------------------------------------- main loop
async function loop() {
  if (!running) return;
  let source, w, h;
  if (running === "demo") {
    source = demoFrame();
    w = source.width;
    h = source.height;
  } else {
    source = video;
    w = video.videoWidth;
    h = video.videoHeight;
  }

  if (w && h) {
    const start = performance.now();
    const detections = await detector.detect(source, w, h, {
      conf: Number($("conf").value),
      classes: selectedClasses(),
    });
    const tracks = tracker.update(detections);
    const current = 1000 / Math.max(performance.now() - start, 1);
    fps = fps ? 0.9 * fps + 0.1 * current : current;
    draw(source, w, h, tracks);
  }
  if (running) requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- drawing
function draw(source, w, h, tracks) {
  if (view.width !== w || view.height !== h + INFO_BAR) {
    view.width = w;
    view.height = h + INFO_BAR;
  }
  g.drawImage(source, 0, INFO_BAR, w, h);
  g.font = "600 15px 'Segoe UI', system-ui, sans-serif";
  g.textBaseline = "middle";

  const showTrails = $("trails").checked;
  const seen = new Set();
  for (const t of tracks) {
    seen.add(t.id);
    const [x1, y1, x2, y2] = t.box;
    const color = colorForId(t.id);
    const top = INFO_BAR + y1;

    g.lineWidth = 3;
    g.strokeStyle = color;
    g.strokeRect(x1, top, x2 - x1, y2 - y1);

    const label = `${names[t.cls]} #${t.id}  ${t.score.toFixed(2)}`;
    const tw = g.measureText(label).width + 12;
    const lx = Math.min(Math.max(x1, 0), w - tw);
    const ly = Math.max(top - 24, INFO_BAR);
    g.fillStyle = color;
    g.fillRect(lx, ly, tw, 24);
    g.fillStyle = "#000";
    g.fillText(label, lx + 6, ly + 12);

    // motion trail: last 30 centre points of this track
    const trail = trails.get(t.id) || [];
    trail.push([(x1 + x2) / 2, top + (y2 - y1) / 2]);
    if (trail.length > 30) trail.shift();
    trails.set(t.id, trail);
    if (showTrails) {
      for (let i = 1; i < trail.length; i++) {
        g.lineWidth = Math.max(1, (4 * i) / trail.length);
        g.beginPath();
        g.moveTo(...trail[i - 1]);
        g.lineTo(...trail[i]);
        g.stroke();
      }
    }
  }
  for (const id of trails.keys()) if (!seen.has(id) && !tracker.trackers.some((t) => t.id === id)) trails.delete(id);

  // info bar: FPS, number of tracked objects, count per class
  const counts = {};
  for (const t of tracks) counts[names[t.cls]] = (counts[names[t.cls]] || 0) + 1;
  const perClass = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([n, c]) => `${n}: ${c}`).join(", ");
  g.fillStyle = "#181818";
  g.fillRect(0, 0, w, INFO_BAR);
  g.fillStyle = "#fff";
  g.font = "15px 'Segoe UI', system-ui, sans-serif";
  g.fillText(`FPS ${fps.toFixed(1)}   |   Tracking ${tracks.length} objects${perClass ? `   |   ${perClass}` : ""}`, 10, INFO_BAR / 2);
}

function drawIdle(text) {
  g.fillStyle = "#0b0e14";
  g.fillRect(0, 0, view.width, view.height);
  g.fillStyle = "#9aa4b6";
  g.font = "18px 'Segoe UI', system-ui, sans-serif";
  g.textAlign = "center";
  g.fillText(text, view.width / 2, view.height / 2);
  g.textAlign = "start";
}

// ---------------------------------------------------------------- wiring
$("btn-webcam").addEventListener("click", startWebcam);
$("btn-demo").addEventListener("click", startDemo);
$("btn-stop").addEventListener("click", () => {
  stop();
  setStatus("Stopped.");
});
$("file").addEventListener("change", (e) => e.target.files[0] && startFile(e.target.files[0]));
$("conf").addEventListener("input", () => ($("conf-value").textContent = Number($("conf").value).toFixed(2)));
$("classes").addEventListener("change", resetTracking);

drawIdle("Loading the model…");
init().catch((err) => setStatus(`Could not load the model: ${err.message}`));
