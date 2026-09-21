import unittest
import tempfile
from pathlib import Path
import json
from weld_count.io import load_annotations
from weld_count.policy import DecisionPolicy


class PolicyTest(unittest.TestCase):
    def test_sixteen_strict_majority(self):
        p = DecisionPolicy()
        self.assertEqual(p.decide_count(8, 16), "NG")
        self.assertEqual(p.decide_count(9, 16), "OK")

    def test_twenty_five(self):
        p = DecisionPolicy()
        self.assertEqual(p.decide_count(12, 25), "NG")
        self.assertEqual(p.decide_count(13, 25), "OK")

    def test_score_filter(self):
        p = DecisionPolicy(min_score=.8)
        self.assertEqual(p.decide([{"score": .7}, {"score": .9}], 16)["count"], 1)

    def test_one_json_per_image_and_labelme_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps({"imagePath": "sample.jpg", "total": 16,
                                        "status": "合格", "shapes": [{"label": "weld", "points": [[1, 2]]}]}), encoding="utf-8")
            row = load_annotations(tmp)[0]
            self.assertEqual(row["image"], "sample.jpg")
            self.assertEqual(row["label"], "OK")
            self.assertEqual(len(row["objects"]), 1)


if __name__ == "__main__": unittest.main()
