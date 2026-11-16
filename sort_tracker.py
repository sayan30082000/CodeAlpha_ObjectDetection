"""SORT — Simple Online and Realtime Tracking (Bewley et al., 2016), implemented with NumPy.

Each tracked object gets a Kalman filter with a constant-velocity motion model.
On every frame:
  1. every existing track predicts where its box will be now,
  2. new detections are matched to those predictions with the Hungarian algorithm
     on an IoU (intersection-over-union) cost matrix,
  3. matched tracks are corrected with their detection, unmatched detections start
     new tracks, and tracks that have not been seen for `max_age` frames are deleted.

The result is a stable ID for every object across frames.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    a = boxes_a[:, None, :4]
    b = boxes_b[None, :, :4]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


def box_to_z(box) -> np.ndarray:
    """[x1, y1, x2, y2] -> measurement [cx, cy, area, aspect_ratio]."""
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)], dtype=float)


def x_to_box(x) -> np.ndarray:
    """Kalman state [cx, cy, area, ratio, ...] -> [x1, y1, x2, y2]."""
    area, ratio = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(area * ratio)
    h = area / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTracker:
    """One tracked object. State: [cx, cy, area, ratio, v_cx, v_cy, v_area]."""

    _next_id = 1

    # Constant-velocity model (the same matrices as the original SORT paper).
    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)
    R = np.diag([1.0, 1.0, 10.0, 10.0])                          # measurement noise
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])          # process noise

    def __init__(self, box, score: float, cls: int):
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])  # high uncertainty on velocity
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.score, self.cls = score, cls
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:  # area must not become negative
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, score: float, cls: int) -> None:
        z = box_to_z(box)
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.score, self.cls = score, cls
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    @property
    def box(self) -> np.ndarray:
        return x_to_box(self.x)


class Sort:
    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        max_age:       frames a track survives without a matching detection.
        min_hits:      detections needed before a new track is reported (filters flicker).
        iou_threshold: minimum IoU for a detection to be matched to a track.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> list[dict]:
        """
        detections: array of shape (N, 6) -> [x1, y1, x2, y2, score, class_id].
        Returns the confirmed tracks for this frame as dicts with id, box, score, cls.
        """
        self.frame_count += 1
        detections = np.asarray(detections, dtype=float).reshape(-1, 6)

        # 1. predict the new location of every existing track
        predicted = np.array([t.predict() for t in self.trackers]).reshape(-1, 4)
        valid = np.all(np.isfinite(predicted), axis=1)
        self.trackers = [t for t, ok in zip(self.trackers, valid) if ok]
        predicted = predicted[valid]

        # 2. associate detections to tracks (Hungarian algorithm on IoU)
        matches, unmatched_dets = self._associate(detections, predicted)

        # 3. update matched tracks, create tracks for new objects
        for d, t in matches:
            self.trackers[t].update(detections[d, :4], detections[d, 4], int(detections[d, 5]))
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d, :4], detections[d, 4], int(detections[d, 5])))

        # 4. report confirmed tracks and delete the ones that are gone
        results = []
        for trk in self.trackers:
            confirmed = trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            if trk.time_since_update == 0 and confirmed:
                results.append({"id": trk.id, "box": trk.box, "score": trk.score, "cls": trk.cls})
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return results

    def _associate(self, detections: np.ndarray, predicted: np.ndarray):
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        iou = iou_matrix(detections, predicted)
        det_idx, trk_idx = linear_sum_assignment(-iou)

        matches, matched_dets = [], set()
        for d, t in zip(det_idx, trk_idx):
            if iou[d, t] >= self.iou_threshold:
                matches.append((d, t))
                matched_dets.add(d)
        unmatched = [d for d in range(len(detections)) if d not in matched_dets]
        return matches, unmatched
