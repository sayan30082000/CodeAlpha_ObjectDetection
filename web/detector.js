/*
 * YOLO11 object detector in the browser with ONNX Runtime Web.
 * Same steps as Ultralytics: letterbox resize -> model -> confidence filter -> per-class NMS.
 * The model file (yolo11n.onnx) is exported from the same weights by export_web.py.
 */
(function (global) {
  "use strict";

  const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

  function iou(a, b) {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = w * h;
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
  }

  class YoloDetector {
    // Uses the GPU (WebGPU) when the browser supports it, otherwise WebAssembly on the CPU.
    static async load(modelUrl, size, onStatus = () => {}) {
      const ort = global.ort;
      ort.env.wasm.wasmPaths = ORT_CDN;
      ort.env.wasm.numThreads = global.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

      let session = null, backend = "WebAssembly (CPU)";
      if (navigator.gpu) {
        try {
          session = await ort.InferenceSession.create(modelUrl, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
          backend = "WebGPU (GPU)";
        } catch (err) {
          console.warn("WebGPU not usable, falling back to WebAssembly:", err);
        }
      }
      session ||= await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });

      const detector = new YoloDetector(session, size, backend);
      onStatus(`Warming up ${backend}…`);
      await detector.detect(detector.canvas, size, size); // first run compiles GPU shaders / JIT
      return detector;
    }

    constructor(session, size, backend) {
      this.session = session;
      this.size = size;
      this.backend = backend;
      this.canvas = document.createElement("canvas");
      this.canvas.width = this.canvas.height = size;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
      this.input = new Float32Array(3 * size * size);
    }

    // source: video / canvas / image; returns [[x1, y1, x2, y2, score, classId], ...] in source pixels
    async detect(source, width, height, { conf = 0.4, iouThreshold = 0.7, classes = null } = {}) {
      const S = this.size;
      const scale = Math.min(S / width, S / height);
      const nw = Math.round(width * scale), nh = Math.round(height * scale);
      const padX = Math.round((S - nw) / 2 - 0.1), padY = Math.round((S - nh) / 2 - 0.1);

      // letterbox: keep the aspect ratio, pad with grey (114) like Ultralytics
      this.ctx.fillStyle = "rgb(114,114,114)";
      this.ctx.fillRect(0, 0, S, S);
      this.ctx.drawImage(source, 0, 0, width, height, padX, padY, nw, nh);
      const pixels = this.ctx.getImageData(0, 0, S, S).data;
      const area = S * S;
      for (let i = 0; i < area; i++) {
        this.input[i] = pixels[i * 4] / 255;              // R plane
        this.input[area + i] = pixels[i * 4 + 1] / 255;   // G plane
        this.input[2 * area + i] = pixels[i * 4 + 2] / 255; // B plane
      }

      const ort = global.ort;
      const feeds = { [this.session.inputNames[0]]: new ort.Tensor("float32", this.input, [1, 3, S, S]) };
      const output = (await this.session.run(feeds))[this.session.outputNames[0]];
      const [, channels, n] = output.dims; // [1, 4 + 80, anchors]
      const d = output.data;

      // output layout: rows = cx, cy, w, h, then one score per class; columns = anchors
      const candidates = [];
      for (let i = 0; i < n; i++) {
        let best = 0, cls = -1;
        for (let c = 4; c < channels; c++) {
          const s = d[c * n + i];
          if (s > best) { best = s; cls = c - 4; }
        }
        if (best < conf || (classes && !classes.has(cls))) continue;
        const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
        candidates.push([
          Math.max(0, (cx - w / 2 - padX) / scale),
          Math.max(0, (cy - h / 2 - padY) / scale),
          Math.min(width, (cx + w / 2 - padX) / scale),
          Math.min(height, (cy + h / 2 - padY) / scale),
          best,
          cls,
        ]);
      }

      // non-maximum suppression, per class
      candidates.sort((a, b) => b[4] - a[4]);
      const kept = [];
      for (const c of candidates) {
        if (kept.every((k) => k[5] !== c[5] || iou(k, c) <= iouThreshold)) kept.push(c);
        if (kept.length >= 300) break;
      }
      return kept;
    }
  }

  global.YoloDetector = YoloDetector;
})(window);
