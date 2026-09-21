from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_IMAGE_KEYS = ("image", "imagePath", "file_name", "filename", "fileName")
_LABEL_KEYS = ("label", "status", "result", "quality", "判定", "结果")
_TOTAL_KEYS = ("total", "expected", "expected_count", "weld_count", "焊点总数")
_OBJECT_KEYS = ("annotations", "objects", "shapes", "points", "instances")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"annotation must be a JSON object: {path}")
    return value


def _records(path: str | Path) -> list[dict[str, Any]]:
    """Read the legacy aggregate JSON/JSONL prediction format."""
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    data = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if isinstance(data, dict):
        data = data.get("images", data.get("items", [data]))
    if not isinstance(data, list):
        raise ValueError("file must be a JSON array or JSONL")
    return data


def _first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _normalise_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return {"合格": "OK", "良品": "OK", "PASS": "OK", "不合格": "NG", "不良": "NG", "FAIL": "NG"}.get(text, text)


def _normalise_record(row: dict[str, Any], source: str | Path, default_total: int | None = None) -> dict[str, Any]:
    image = _first(row, _IMAGE_KEYS)
    objects = _first(row, _OBJECT_KEYS, [])
    if not isinstance(objects, list):
        raise ValueError(f"annotation objects must be a list: {source}")
    total_value = _first(row, _TOTAL_KEYS, default_total)
    if total_value is None:
        raise ValueError(f"missing total/expected in annotation: {source}; pass --total")
    return {
        "image": str(image) if image else "",
        "total": int(total_value),
        "label": _normalise_label(_first(row, _LABEL_KEYS, "")),
        "objects": objects,
        "annotation_file": str(source),
    }


def _infer_image(annotation_file: Path) -> str:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidate = annotation_file.with_suffix(suffix)
        if candidate.exists():
            return candidate.name
    return annotation_file.stem + ".jpg"


def load_annotation_file(path: str | Path, default_total: int | None = None) -> dict[str, Any]:
    path = Path(path)
    row = _read_json(path)
    if not _first(row, _IMAGE_KEYS):
        row["image"] = _infer_image(path)
    return _normalise_record(row, path, default_total)


def load_annotations(path: str | Path, default_total: int | None = None) -> list[dict[str, Any]]:
    """Load either one aggregate file or a directory containing one JSON per image."""
    path = Path(path)
    if path.is_dir():
        return [load_annotation_file(p, default_total) for p in sorted(path.glob("*.json"))]
    return [_normalise_record(row, path, default_total) for row in _records(path)]


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        result = {}
        for file in sorted(path.glob("*.json")):
            row = _read_json(file)
            image = _first(row, _IMAGE_KEYS, _infer_image(file))
            result[str(image)] = row
        return result
    return {str(r.get("image", r.get("file_name"))): r for r in _records(path)}


def object_center(obj: dict[str, Any]) -> tuple[float, float] | None:
    if "point" in obj and len(obj["point"]) >= 2:
        return float(obj["point"][0]), float(obj["point"][1])
    points = obj.get("points")
    if points and len(points[0]) >= 2:
        return float(points[0][0]), float(points[0][1])
    b = obj.get("bbox", obj.get("box"))
    if b and len(b) == 4:
        x, y, a, c = map(float, b)
        if obj.get("bbox_mode") in ("xywh", 1):
            return x + a / 2, y + c / 2
        return (x + a) / 2, (y + c) / 2
    return None
