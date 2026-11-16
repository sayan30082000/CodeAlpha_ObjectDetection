"""Create a short test video without a webcam.

It slides a "camera" window across the sample street photo that ships with the
`ultralytics` package, so the people and the bus move across the frame and the
tracker has something to follow.

    python make_sample_video.py            # -> sample_video.mp4
    python detect_track.py --source sample_video.mp4
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="sample_video.mp4")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    args = p.parse_args()

    package_dir = Path(importlib.util.find_spec("ultralytics").submodule_search_locations[0])
    image_path = package_dir / "assets" / "bus.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Sample image not found: {image_path}")

    out_w, out_h = args.size
    img_h, img_w = image.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (out_w, out_h))

    for i in range(args.frames):
        t = i / max(args.frames - 1, 1)
        # zoom out slowly while panning left -> right and a little down
        crop_w = int(img_w * (0.75 + 0.2 * t))
        crop_h = int(crop_w * out_h / out_w)
        crop_h = min(crop_h, img_h)
        x = int((img_w - crop_w) * t)
        y = int((img_h - crop_h) * (0.35 + 0.3 * np.sin(np.pi * t)))
        frame = cv2.resize(image[y:y + crop_h, x:x + crop_w], (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    print(f"Wrote {args.frames} frames to {args.out}")


if __name__ == "__main__":
    main()
