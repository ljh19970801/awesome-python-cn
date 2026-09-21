from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from .evaluate import calibrate, metrics, save_policy
from .io import DecisionPolicy, load_annotations


def baseline(args):
    out = []
    for row in load_annotations(args.annotations):
        image = str(Path(args.annotations).parent / row["image"])
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image)
        # Generic baseline only: production should replace this with a trained detector.
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(blur, args.threshold, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        detections = [{"score": 1.0, "bbox": [int(x), int(y), int(w), int(h)]}
                      for x, y, w, h, area in stats[1:] if area >= args.min_area]
        out.append({"image": row["image"], "detections": detections})
    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(prog="weld-count")
    sub = p.add_subparsers(required=True)
    b = sub.add_parser("baseline"); b.add_argument("--annotations", required=True); b.add_argument("--out", required=True); b.add_argument("--threshold", type=int, default=150); b.add_argument("--min-area", type=int, default=8); b.set_defaults(func=baseline)
    c = sub.add_parser("calibrate"); c.add_argument("--annotations", required=True); c.add_argument("--predictions", required=True); c.add_argument("--out", required=True); c.add_argument("--max-overkill", type=float, default=.06); c.set_defaults(func=lambda a: save_policy(calibrate(a.annotations, a.predictions, a.max_overkill), a.out))
    e = sub.add_parser("evaluate"); e.add_argument("--annotations", required=True); e.add_argument("--predictions", required=True); e.add_argument("--policy", required=True); e.set_defaults(func=lambda a: print(json.dumps(metrics(a.annotations, a.predictions, DecisionPolicy.from_dict(json.loads(Path(a.policy).read_text())),), ensure_ascii=False, indent=2)))
    args = p.parse_args(); args.func(args)


if __name__ == "__main__": main()
