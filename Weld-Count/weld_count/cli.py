from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from .evaluate import calibrate, metrics, save_policy
from .io import DecisionPolicy, load_annotations


def _image_path(annotation_file: str, image_name: str, images_dir: str | None) -> Path:
    image = Path(image_name)
    if image.is_absolute() and image.exists():
        return image
    roots = [Path(images_dir)] if images_dir else []
    roots.append(Path(annotation_file).parent)
    for root in roots:
        candidate = root / image
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found: {image_name}; searched {roots}")


def baseline(args):
    out = []
    rows = load_annotations(args.annotations, args.total)
    for row in rows:
        image = _image_path(row["annotation_file"], row["image"], args.images_dir)
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(blur, args.threshold, 255, cv2.THRESH_BINARY)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        detections = [{"score": 1.0, "bbox": [int(x), int(y), int(w), int(h)]}
                      for x, y, w, h, area in stats[1:] if area >= args.min_area]
        out.append({"image": row["image"], "detections": detections})
    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(prog="weld-count")
    sub = p.add_subparsers(required=True)
    b = sub.add_parser("baseline", help="one JSON annotation per image, or legacy aggregate file")
    b.add_argument("--annotations", required=True, help="annotation JSON directory or JSON/JSONL file")
    b.add_argument("--images-dir", help="image root when images are not beside JSON files")
    b.add_argument("--total", type=int, choices=(16, 25), help="default total if JSON omits it")
    b.add_argument("--out", required=True); b.add_argument("--threshold", type=int, default=150); b.add_argument("--min-area", type=int, default=8); b.set_defaults(func=baseline)
    c = sub.add_parser("calibrate"); c.add_argument("--annotations", required=True); c.add_argument("--predictions", required=True); c.add_argument("--out", required=True); c.add_argument("--max-overkill", type=float, default=.06); c.add_argument("--total", type=int, choices=(16, 25)); c.set_defaults(func=lambda a: save_policy(calibrate(a.annotations, a.predictions, a.max_overkill, a.total), a.out))
    e = sub.add_parser("evaluate"); e.add_argument("--annotations", required=True); e.add_argument("--predictions", required=True); e.add_argument("--policy", required=True); e.add_argument("--total", type=int, choices=(16, 25)); e.set_defaults(func=lambda a: print(json.dumps(metrics(a.annotations, a.predictions, DecisionPolicy.from_dict(json.loads(Path(a.policy).read_text())), a.total), ensure_ascii=False, indent=2)))
    args = p.parse_args(); args.func(args)


if __name__ == "__main__": main()
