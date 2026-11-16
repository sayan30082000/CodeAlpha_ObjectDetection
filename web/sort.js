/*
 * SORT tracker in JavaScript — a line-by-line port of sort_tracker.py.
 * Kalman filter (constant velocity) per object + Hungarian matching on IoU.
 */
(function (global) {
  "use strict";

  // ---------------------------------------------------------------- tiny matrix helpers
  const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const diag = (values) => values.map((v, i) => values.map((_, j) => (i === j ? v : 0)));
  const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
  const matMul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
  const matVec = (A, v) => A.map((row) => row.reduce((s, a, k) => s + a * v[k], 0));
  const add = (A, B) => A.map((row, i) => row.map((v, j) => v + B[i][j]));
  const sub = (A, B) => A.map((row, i) => row.map((v, j) => v - B[i][j]));

  function inverse(M) {
    // Gauss-Jordan elimination (M is small: 4x4)
    const n = M.length;
    const A = M.map((row, i) => [...row, ...eye(n)[i]]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
      [A[col], A[pivot]] = [A[pivot], A[col]];
      const p = A[col][col];
      for (let j = 0; j < 2 * n; j++) A[col][j] /= p;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = A[r][col];
        for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[col][j];
      }
    }
    return A.map((row) => row.slice(n));
  }

  // ---------------------------------------------------------------- boxes
  function iou(a, b) {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = w * h;
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter + 1e-9);
  }

  const boxToZ = (b) => {
    const w = b[2] - b[0], h = b[3] - b[1];
    return [b[0] + w / 2, b[1] + h / 2, w * h, w / Math.max(h, 1e-6)];
  };

  const xToBox = (x) => {
    const area = Math.max(x[2], 1e-6), ratio = Math.max(x[3], 1e-6);
    const w = Math.sqrt(area * ratio), h = area / w;
    return [x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2];
  };

  // ---------------------------------------------------------------- Hungarian algorithm
  // Minimum-cost assignment for an n x m cost matrix (same result as scipy's linear_sum_assignment).
  function hungarian(cost) {
    const n = cost.length, m = cost[0].length, INF = 1e18;
    const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0);
    const p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Array(m + 1).fill(INF);
      const used = new Array(m + 1).fill(false);
      do {
        used[j0] = true;
        const i0 = p[j0];
        let delta = INF, j1 = 0;
        for (let j = 1; j <= m; j++) {
          if (used[j]) continue;
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= m; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const pairs = [];
    for (let j = 1; j <= m; j++) if (p[j]) pairs.push([p[j] - 1, j - 1]);
    return pairs;
  }

  function linearSumAssignment(cost) {
    if (cost.length <= cost[0].length) return hungarian(cost);
    return hungarian(transpose(cost)).map(([r, c]) => [c, r]);
  }

  // ---------------------------------------------------------------- Kalman tracker
  const F = eye(7);
  F[0][4] = F[1][5] = F[2][6] = 1;
  const FT = transpose(F);
  const H = eye(7).slice(0, 4);
  const HT = transpose(H);
  const R = diag([1, 1, 10, 10]);
  const Q = diag([1, 1, 1, 1, 0.01, 0.01, 0.0001]);
  const I7 = eye(7);

  let nextId = 1;

  class KalmanBoxTracker {
    constructor(box, score, cls) {
      this.x = [...boxToZ(box), 0, 0, 0];
      this.P = diag([10, 10, 10, 10, 1e4, 1e4, 1e4]);
      this.id = nextId++;
      this.score = score;
      this.cls = cls;
      this.hits = 1;
      this.hitStreak = 1;
      this.age = 0;
      this.timeSinceUpdate = 0;
    }

    predict() {
      if (this.x[6] + this.x[2] <= 0) this.x[6] = 0;
      this.x = matVec(F, this.x);
      this.P = add(matMul(matMul(F, this.P), FT), Q);
      this.age += 1;
      if (this.timeSinceUpdate > 0) this.hitStreak = 0;
      this.timeSinceUpdate += 1;
      return xToBox(this.x);
    }

    update(box, score, cls) {
      const z = boxToZ(box);
      const y = z.map((v, i) => v - matVec(H, this.x)[i]);
      const S = add(matMul(matMul(H, this.P), HT), R);
      const K = matMul(matMul(this.P, HT), inverse(S));
      const Ky = matVec(K, y);
      this.x = this.x.map((v, i) => v + Ky[i]);
      this.P = matMul(sub(I7, matMul(K, H)), this.P);
      this.score = score;
      this.cls = cls;
      this.timeSinceUpdate = 0;
      this.hits += 1;
      this.hitStreak += 1;
    }

    get box() { return xToBox(this.x); }
  }

  class Sort {
    constructor({ maxAge = 15, minHits = 3, iouThreshold = 0.3 } = {}) {
      this.maxAge = maxAge;
      this.minHits = minHits;
      this.iouThreshold = iouThreshold;
      this.trackers = [];
      this.frameCount = 0;
    }

    // detections: [[x1, y1, x2, y2, score, cls], ...] -> confirmed tracks [{id, box, score, cls}]
    update(detections) {
      this.frameCount += 1;

      // 1. predict every track; drop broken ones
      const predicted = [];
      this.trackers = this.trackers.filter((t) => {
        const box = t.predict();
        if (box.every(Number.isFinite)) { predicted.push(box); return true; }
        return false;
      });

      // 2. Hungarian matching on IoU
      const matched = new Set();
      if (predicted.length && detections.length) {
        const ious = detections.map((d) => predicted.map((p) => iou(d, p)));
        for (const [d, t] of linearSumAssignment(ious.map((row) => row.map((v) => -v)))) {
          if (ious[d][t] >= this.iouThreshold) {
            this.trackers[t].update(detections[d].slice(0, 4), detections[d][4], detections[d][5]);
            matched.add(d);
          }
        }
      }

      // 3. new tracks for unmatched detections
      detections.forEach((d, i) => {
        if (!matched.has(i)) this.trackers.push(new KalmanBoxTracker(d.slice(0, 4), d[4], d[5]));
      });

      // 4. report confirmed tracks, delete lost ones
      const results = [];
      for (const t of this.trackers) {
        const confirmed = t.hitStreak >= this.minHits || this.frameCount <= this.minHits;
        if (t.timeSinceUpdate === 0 && confirmed) results.push({ id: t.id, box: t.box, score: t.score, cls: t.cls });
      }
      this.trackers = this.trackers.filter((t) => t.timeSinceUpdate <= this.maxAge);
      return results;
    }
  }

  const api = { Sort, iou, linearSumAssignment, resetIds: () => (nextId = 1) };
  global.SortTracker = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
