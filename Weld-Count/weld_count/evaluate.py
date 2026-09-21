from __future__ import annotations

import json
from pathlib import Path
from .io import DecisionPolicy, load_annotations, load_predictions


def _rows(ann_path: str, pred_path: str, policy: DecisionPolicy):
    preds = load_predictions(pred_path)
    for a in load_annotations(ann_path):
        p = preds.get(str(a["image"]), {})
        yield a, policy.decide(p.get("detections", []), a["total"])


def metrics(ann_path: str, pred_path: str, policy: DecisionPolicy) -> dict:
    tp = tn = fp = fn = 0
    errors = []
    for a, out in _rows(ann_path, pred_path, policy):
        truth = a["label"]
        pred = out["decision"]
        if truth == "OK" and pred == "OK": tp += 1
        elif truth == "NG" and pred == "NG": tn += 1
        elif truth == "OK": fp += 1
        elif truth == "NG": fn += 1
        if a["total"] and out["count"] != len(a["objects"]):
            errors.append(abs(out["count"] - len(a["objects"])))
    ok = tp + fp
    ng = tn + fn
    return {"samples": tp + tn + fp + fn, "ok_to_ng_overkill": fp / ok if ok else 0.0,
            "ng_to_ok_leak": fn, "ng_recall": tn / ng if ng else 1.0,
            "confusion": {"OK_OK": tp, "OK_NG": fp, "NG_NG": tn, "NG_OK": fn},
            "mean_count_error_vs_annotations": sum(errors) / len(errors) if errors else 0.0}


def calibrate(ann_path: str, pred_path: str, max_overkill: float = .06) -> DecisionPolicy:
    """Choose the lowest score threshold that has zero NG->OK on validation.

    Lower thresholds generally increase count and reduce OK->NG overkill. We still
    verify the resulting policy and fail loudly if the business target is missed.
    """
    preds = load_predictions(pred_path)
    scores = sorted({float(d.get("score", 1.0)) for p in preds.values() for d in p.get("detections", [])})
    candidates = sorted(set([0.0, .5, 1.0] + scores))
    chosen = None
    for threshold in candidates:
        candidate = DecisionPolicy(min_score=threshold)
        m = metrics(ann_path, pred_path, candidate)
        if m["ng_to_ok_leak"] == 0 and m["ok_to_ng_overkill"] <= max_overkill:
            chosen = candidate
            break
    if chosen is None:
        # Return the safest policy so the report explains the shortfall.
        chosen = DecisionPolicy(min_score=0.0)
    report = metrics(ann_path, pred_path, chosen)
    if report["ng_to_ok_leak"] != 0 or report["ok_to_ng_overkill"] > max_overkill:
        raise RuntimeError("validation cannot satisfy zero leak and <=6% overkill; improve data/model")
    return chosen


def save_policy(policy: DecisionPolicy, path: str) -> None:
    Path(path).write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
