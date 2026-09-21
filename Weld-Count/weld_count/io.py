from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def _records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if isinstance(data, dict):
        data = data.get("images", data.get("items", [data]))
    if not isinstance(data, list):
        raise ValueError("annotation/prediction file must be a JSON array or JSONL")
    return data


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for row in _records(path):
        image = row.get("image", row.get("file_name", row.get("filename")))
        if not image:
            raise ValueError("each annotation must contain image/file_name")
        objects = row.get("annotations", row.get("objects", row.get("shapes", row.get("points", []))))
        result.append({"image": image, "total": int(row.get("total", row.get("expected", 0))),
                       "label": str(row.get("label", row.get("status", ""))).upper(),
                       "objects": objects or []})
    return result


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(r.get("image", r.get("file_name"))): r for r in _records(path)}


def object_center(obj: dict[str, Any]) -> tuple[float, float] | None:
    if "point" in obj and len(obj["point"]) >= 2:
        return float(obj["point"][0]), float(obj["point"][1])
    if "points" in obj and obj["points"] and len(obj["points"][0]) >= 2:
        return float(obj["points"][0][0]), float(obj["points"][0][1])
    b = obj.get("bbox", obj.get("box"))
    if b and len(b) == 4:
        x, y, a, c = map(float, b)
        if obj.get("bbox_mode") in ("xywh", 1) or (a <= x or c <= y):
            return x + a / 2, y + c / 2
        return (x + a) / 2, (y + c) / 2
    return None
