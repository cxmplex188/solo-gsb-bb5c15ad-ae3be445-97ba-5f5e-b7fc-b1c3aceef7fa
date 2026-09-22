"""Tests for the scoring math: calibration, totals, ranking and reversals."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scoring import MAD_K, analyze, competition_rank, robust_params


def recs(data):
    # data: list of (contestant, judge, item, score)
    return [
        {"id": i + 1, "contestant": c, "judge": j, "item": it, "score": s}
        for i, (c, j, it, s) in enumerate(data)
    ]


class TestRobustParams(unittest.TestCase):
    def test_median_and_mad(self):
        med, raw, scaled = robust_params([1, 2, 3, 4, 100])
        self.assertEqual(med, 3)
        # deviations: 2,1,0,1,97 -> median 1
        self.assertEqual(raw, 1)
        self.assertAlmostEqual(scaled, MAD_K)

    def test_zero_spread(self):
        med, raw, scaled = robust_params([5, 5, 5])
        self.assertEqual((med, raw, scaled), (5, 0, 0))


class TestRank(unittest.TestCase):
    def test_competition_ranking_ties(self):
        ranks = competition_rank([("a", 10.0), ("b", 10.0), ("c", 9.0), ("d", 8.0)])
        self.assertEqual(ranks["a"], 1)
        self.assertEqual(ranks["b"], 1)
        self.assertEqual(ranks["c"], 3)
        self.assertEqual(ranks["d"], 4)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        # Two judges with offset/scale differences, two items, two contestants.
        # Judge A tight low range; judge B wide high range.
        self.data = [
            ("C1", "JA", "I1", 8.0),
            ("C2", "JA", "I1", 7.0),
            ("C1", "JA", "I2", 9.0),
            ("C2", "JA", "I2", 6.0),
            ("C1", "JB", "I1", 12.0),
            ("C2", "JB", "I1", 10.0),
            ("C1", "JB", "I2", 13.0),
            ("C2", "JB", "I2", 9.0),
        ]

    def test_zscore_centered(self):
        a = analyze(recs(self.data), {}, "judge")
        by_judge = {g["judge"]: g for g in a["groups"]}
        # JB: values 12,10,13,9 -> median 11, deviations 1,1,2,2 -> raw MAD 1.5
        self.assertAlmostEqual(by_judge["JB"]["median"], 11)
        self.assertAlmostEqual(by_judge["JB"]["raw_mad"], 1.5)
        # Every judge's mean robust z across their own scores should be ~0 here
        # because symmetric deviations around the median.
        for block in a["contestants"]:
            for ib in block["items"]:
                for s in ib["scores"]:
                    if s["judge"] == "JB" and s["raw"] == 11:
                        self.assertAlmostEqual(s["z"], 0.0)

    def test_totals_and_ranks(self):
        a = analyze(recs(self.data), {}, "judge")
        blocks = {b["contestant"]: b for b in a["contestants"]}
        # C1 wins every item under both judges -> rank 1 raw and calibrated
        self.assertEqual(blocks["C1"]["raw_rank"], 1)
        self.assertEqual(blocks["C1"]["cal_rank"], 1)
        self.assertEqual(blocks["C1"]["eff_rank"], 1)
        self.assertEqual(blocks["C2"]["raw_rank"], 2)

    def test_override_raw_changes_effective_only(self):
        records = recs(self.data)
        first_id = records[0]["id"]
        a = analyze(records, {first_id: (True, "test reason")}, "judge")
        for block in a["contestants"]:
            for ib in block["items"]:
                for s in ib["scores"]:
                    if s["id"] == first_id:
                        self.assertTrue(s["use_raw"])
                        self.assertEqual(s["effective"], s["raw"])
                        self.assertNotEqual(s["effective"], s["calibrated"])
                        self.assertEqual(s["reason"], "test reason")
                    else:
                        self.assertFalse(s["use_raw"])
                        self.assertEqual(s["effective"], s["calibrated"])
        # calibrated totals/ranks must be unaffected by decisions
        blocks = {b["contestant"]: b for b in a["contestants"]}
        self.assertEqual(blocks["C1"]["cal_rank"], 1)
        self.assertEqual(a["counts"]["decisions"], 1)

    def test_judge_item_scope(self):
        a = analyze(recs(self.data), {}, "judge_item")
        self.assertEqual(a["scope"], "judge_item")
        # 2 judges x 2 items = 4 groups
        self.assertEqual(len(a["groups"]), 4)
        self.assertIn("item", a["groups"][0])

    def test_reversal_detected(self):
        # Two tight judges rank C1 first; one wide-spread judge ranks C2 far
        # above C1. Raw average crowns C2; robust calibration downweights the
        # wide judge and crowns C1.
        data = [
            ("C1", "JA", "I1", 8.30), ("C2", "JA", "I1", 8.00),
            ("C3", "JA", "I1", 7.90), ("C4", "JA", "I1", 7.80),
            ("C1", "JB", "I1", 8.30), ("C2", "JB", "I1", 8.00),
            ("C3", "JB", "I1", 7.85), ("C4", "JB", "I1", 7.75),
            ("C1", "JC", "I1", 5.0), ("C2", "JC", "I1", 9.5),
            ("C3", "JC", "I1", 7.2), ("C4", "JC", "I1", 6.8),
        ]
        a = analyze(recs(data), {}, "judge")
        self.assertGreaterEqual(len(a["reversals"]), 1)
        top_reversal = a["reversals"][0]
        self.assertEqual(top_reversal["winner_before"], "C2")
        self.assertEqual(top_reversal["winner_after"], "C1")
        blocks = {b["contestant"]: b for b in a["contestants"]}
        self.assertEqual(blocks["C1"]["cal_rank"], 1)
        self.assertEqual(blocks["C2"]["raw_rank"], 1)

    def test_empty(self):
        a = analyze([], {}, "judge")
        self.assertEqual(a["counts"]["scores"], 0)
        self.assertEqual(a["reversals"], [])
        self.assertEqual(a["contestants"], [])

    def test_bad_scope(self):
        with self.assertRaises(ValueError):
            analyze(recs(self.data), {}, "bogus")


if __name__ == "__main__":
    unittest.main()
