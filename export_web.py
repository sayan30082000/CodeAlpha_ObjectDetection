"""Build the browser version of the detector in web/ (the live demo on Netlify).

Netlify can't run Python, so the same YOLO11n weights are exported to ONNX and run
in the browser with ONNX Runtime Web (web/detector.js). The SORT tracker is ported
line by line to JavaScript (web/sort.js).

    python export_web.py

Writes web/yolo11n.onnx, web/classes.json and web/demo.jpg.
"""
import importlib.util
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
WEB = ROOT / "web"
IMG_SIZE = 416  # smaller than 640 so it runs smoothly on a laptop CPU in the browser


def main():
    WEB.mkdir(exist_ok=True)
    model = YOLO("yolo11n.pt")
    onnx_path = Path(model.export(format="onnx", imgsz=IMG_SIZE, opset=17, simplify=True, dynamic=False))
    shutil.move(str(onnx_path), WEB / "yolo11n.onnx")

    names = [model.names[i] for i in range(len(model.names))]
    (WEB / "classes.json").write_text(json.dumps({"imgsz": IMG_SIZE, "names": names}), encoding="utf-8")

    # the street photo bundled with ultralytics, used for the "Demo scene" (a slow camera pan)
    package_dir = Path(importlib.util.find_spec("ultralytics").submodule_search_locations[0])
    shutil.copy(package_dir / "assets" / "bus.jpg", WEB / "demo.jpg")

    size_mb = (WEB / "yolo11n.onnx").stat().st_size / 1e6
    print(f"web/yolo11n.onnx ({size_mb:.1f} MB, {IMG_SIZE}x{IMG_SIZE}), classes.json ({len(names)} classes), demo.jpg")


if __name__ == "__main__":
    main()
