import unittest
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


if __name__ == "__main__": unittest.main()
