"""Real-time Object Detection and Tracking — CodeAlpha AI Internship, Task 4.

Pipeline for every video frame:
    OpenCV frame  ->  YOLO (pre-trained on COCO, 80 classes)  ->  bounding boxes
                  ->  SORT tracker (Kalman filter + Hungarian matching)  ->  stable IDs
                  ->  draw boxes, labels, track IDs and motion trails  ->  display / save

Examples
    python detect_track.py                          # webcam 0
    python detect_track.py --source video.mp4       # a video file
    python detect_track.py --classes person car     # only track some classes
    python detect_track.py --source video.mp4 --save output.mp4 --no-show
Keys:  q / Esc = quit,  p = pause
"""
import argparse
import colorsys
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from sort_tracker import Sort


def parse_args():
    p = argparse.ArgumentParser(description="YOLO + SORT object detection and tracking")
    p.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (downloaded automatically on first run)")
    p.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (smaller = faster, e.g. 480)")
    p.add_argument("--classes", nargs="*", help="only detect these class names, e.g. person car dog")
    p.add_argument("--max-age", type=int, default=15, help="frames to keep a lost track alive")
    p.add_argument("--min-hits", type=int, default=3, help="detections needed to confirm a track")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching detections to tracks")
    p.add_argument("--save", help="write the annotated video to this .mp4 file")
    p.add_argument("--no-show", action="store_true", help="do not open a window (useful with --save)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    return p.parse_args()


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """A distinct, stable BGR colour for each track ID (golden-ratio hue spacing)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")
    return cap


def resolve_class_ids(model: YOLO, names: list[str] | None) -> list[int] | None:
    if not names:
        return None
    lookup = {v.lower(): k for k, v in model.names.items()}
    unknown = [n for n in names if n.lower() not in lookup]
    if unknown:
        sys.exit(f"Unknown class name(s): {unknown}. Available: {sorted(lookup)}")
    return [lookup[n.lower()] for n in names]


def detect(model: YOLO, frame, conf: float, imgsz: int, class_ids) -> np.ndarray:
    """Run YOLO on one frame -> array (N, 6): x1, y1, x2, y2, score, class_id."""
    result = model.predict(frame, conf=conf, imgsz=imgsz, classes=class_ids, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6))
    return np.column_stack([
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy(),
    ])


INFO_BAR_HEIGHT = 34


def draw_tracks(frame, tracks, names, trails):
    frame_w = frame.shape[1]
    for trk in tracks:
        x1, y1, x2, y2 = trk["box"].astype(int)
        color = color_for_id(trk["id"])
        label = f"{names[trk['cls']]} #{trk['id']}  {trk['score']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(y1, th + 8)
        lx = min(max(x1, 0), frame_w - tw - 8)  # keep the label inside the frame
        cv2.rectangle(frame, (lx, top - th - 8), (lx + tw + 8, top), color, -1)
        cv2.putText(frame, label, (lx + 4, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # motion trail: the last positions of the box centre
        trails[trk["id"]].append(((x1 + x2) // 2, (y1 + y2) // 2))
        pts = list(trails[trk["id"]])
        for i in range(1, len(pts)):
            thickness = max(1, int(3 * i / len(pts)))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


def add_info_bar(frame, tracks, names, fps):
    """Return the frame with a status bar on top: FPS, number of tracked objects, count per class."""
    counts = defaultdict(int)
    for trk in tracks:
        counts[names[trk["cls"]]] += 1
    per_class = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    text = f"FPS {fps:4.1f}   |   Tracking {len(tracks)} objects" + (f"   |   {per_class}" if per_class else "")

    bar = np.full((INFO_BAR_HEIGHT, frame.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


def main():
    args = parse_args()
    model = YOLO(args.model)
    names = model.names
    class_ids = resolve_class_ids(model, args.classes)
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)
    trails = defaultdict(lambda: deque(maxlen=30))

    cap = open_source(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps,
                                 (width, height + INFO_BAR_HEIGHT))

    window = "Object Detection & Tracking (q to quit)"
    fps, frames, unique_ids = 0.0, 0, set()
    paused = False
    print(f"Source: {args.source}  ({width}x{height} @ {src_fps:.0f} fps)  model: {args.model}")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                start = time.perf_counter()

                detections = detect(model, frame, args.conf, args.imgsz, class_ids)
                tracks = tracker.update(detections)
                unique_ids.update(t["id"] for t in tracks)

                draw_tracks(frame, tracks, names, trails)
                current = 1.0 / max(time.perf_counter() - start, 1e-6)
                fps = current if frames == 0 else 0.9 * fps + 0.1 * current  # smoothed
                display = add_info_bar(frame, tracks, names, fps)

                if writer:
                    writer.write(display)
                frames += 1
                if args.max_frames and frames >= args.max_frames:
                    break

            if not args.no_show:
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {frames} frames, average speed ~{fps:.1f} FPS, {len(unique_ids)} unique objects tracked.")
    if args.save:
        print(f"Saved annotated video to {args.save}")


if __name__ == "__main__":
    main()
