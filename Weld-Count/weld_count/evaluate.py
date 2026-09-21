from __future__ import annotations

import json
from pathlib import Path
from .io import DecisionPolicy, load_annotations, load_predictions


def _rows(ann_path: str, pred_path: str, policy: DecisionPolicy, default_total: int | None = None):
    preds = load_predictions(pred_path)
    for a in load_annotations(ann_path, default_total):
        p = preds.get(str(a["image"]), {})
        yield a, policy.decide(p.get("detections", []), a["total"])


def metrics(ann_path: str, pred_path: str, policy: DecisionPolicy, default_total: int | None = None) -> dict:
    ok_ok = ok_ng = ng_ng = ng_ok = 0
    errors = []
    for a, out in _rows(ann_path, pred_path, policy, default_total):
        truth, pred = a["label"], out["decision"]
        if truth == "OK" and pred == "OK": ok_ok += 1
        elif truth == "NG" and pred == "NG": ng_ng += 1
        elif truth == "OK": ok_ng += 1
        elif truth == "NG": ng_ok += 1
        if a["objects"]:
            errors.append(abs(out["count"] - len(a["objects"])))
    ok = ok_ok + ok_ng
    ng = ng_ng + ng_ok
    return {"samples": ok_ok + ok_ng + ng_ng + ng_ok,
            "ok_to_ng_overkill": ok_ng / ok if ok else 0.0,
            "ng_to_ok_leak": ng_ok, "ng_recall": ng_ng / ng if ng else 1.0,
            "confusion": {"OK_OK": ok_ok, "OK_NG": ok_ng, "NG_NG": ng_ng, "NG_OK": ng_ok},
            "mean_count_error_vs_annotations": sum(errors) / len(errors) if errors else 0.0}


def calibrate(ann_path: str, pred_path: str, max_overkill: float = .06, default_total: int | None = None) -> DecisionPolicy:
    preds = load_predictions(pred_path)
    scores = sorted({float(d.get("score", 1.0)) for p in preds.values() for d in p.get("detections", [])})
    chosen = None
    for threshold in sorted(set([0.0, .5, 1.0] + scores)):
        candidate = DecisionPolicy(min_score=threshold)
        report = metrics(ann_path, pred_path, candidate, default_total)
        if report["ng_to_ok_leak"] == 0 and report["ok_to_ng_overkill"] <= max_overkill:
            chosen = candidate
            break
    if chosen is None:
        raise RuntimeError("validation cannot satisfy zero leak and <=6% overkill; improve detector or labels")
    return chosen


def save_policy(policy: DecisionPolicy, path: str) -> None:
    Path(path).write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
