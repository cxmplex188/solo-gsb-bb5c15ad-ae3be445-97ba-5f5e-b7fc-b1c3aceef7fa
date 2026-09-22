"""Tests for the scoring math: calibration, totals, ranking and reversals."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.scoring import MAD_K, analyze, competition_rank, robust_params, stability_experiment


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


class TestStabilityExperiment(unittest.TestCase):
    # 3 contestants x 3 judges x 2 items = 18 scores.
    DATA = [
        ("C1", "JA", "I1", 8.3), ("C2", "JA", "I1", 8.0), ("C3", "JA", "I1", 7.9),
        ("C1", "JB", "I1", 8.2), ("C2", "JB", "I1", 8.1), ("C3", "JB", "I1", 7.8),
        ("C1", "JC", "I1", 5.0), ("C2", "JC", "I1", 9.4), ("C3", "JC", "I1", 7.0),
        ("C1", "JA", "I2", 7.3), ("C2", "JA", "I2", 7.0), ("C3", "JA", "I2", 6.9),
        ("C1", "JB", "I2", 7.2), ("C2", "JB", "I2", 7.1), ("C3", "JB", "I2", 6.8),
        ("C1", "JC", "I2", 5.2), ("C2", "JC", "I2", 9.1), ("C3", "JC", "I2", 6.5),
    ]

    def _rows(self, result):
        return {r["contestant"]: r for r in result["rows"]}

    def test_exclude_judge_metadata_and_counts(self):
        records = recs(self.DATA)
        res = stability_experiment(records, {}, "judge", "judge", "JC")
        self.assertEqual(res["exclude_type"], "judge")
        self.assertEqual(res["exclude_target"], "JC")
        # JC scored every contestant in both items -> 6 scores removed.
        self.assertEqual(res["removed_scores"], 6)
        self.assertEqual(res["experiment_counts"]["scores"], 12)
        self.assertEqual(res["experiment_counts"]["judges"], 2)
        self.assertEqual(res["experiment_counts"]["items"], 2)

    def test_exclude_item(self):
        records = recs(self.DATA)
        res = stability_experiment(records, {}, "judge", "item", "I2")
        self.assertEqual(res["removed_scores"], 9)
        self.assertEqual(res["experiment_counts"]["items"], 1)
        rows = self._rows(res)
        # Every contestant keeps exactly one item, so each total changes.
        for name in ("C1", "C2", "C3"):
            r = rows[name]
            self.assertFalse(r["missing"])
            self.assertEqual(r["affected_scores"], 3)
            self.assertEqual(r["exp_item_count"], 1)
            self.assertIsNotNone(r["eff_rank_exp"])

    def test_rank_delta_direction(self):
        records = recs(self.DATA)
        res = stability_experiment(records, {}, "judge", "judge", "JC")
        rows = self._rows(res)
        # Removing the outlier judge JC (who favors C2) must not help C2.
        r = rows["C2"]
        self.assertEqual(r["raw_rank_delta"], r["raw_rank"] - r["raw_rank_exp"])
        self.assertGreaterEqual(r["eff_rank_exp"], r["eff_rank"])
        self.assertLessEqual(r["eff_rank_delta"], 0)

    def test_total_deltas_match(self):
        records = recs(self.DATA)
        res = stability_experiment(records, {}, "judge", "judge", "JC")
        for r in res["rows"]:
            self.assertAlmostEqual(
                r["raw_total_exp"] - r["raw_total"], r["raw_total_delta"]
            )
            self.assertAlmostEqual(
                r["cal_total_exp"] - r["cal_total"], r["cal_total_delta"]
            )

    def test_exclude_last_item_marks_contestants_missing(self):
        # Single item: removing it leaves no scores at all.
        data = [
            ("C1", "JA", "I1", 8.0), ("C2", "JA", "I1", 7.0),
            ("C1", "JB", "I1", 9.0), ("C2", "JB", "I1", 6.0),
        ]
        res = stability_experiment(recs(data), {}, "judge", "item", "I1")
        rows = self._rows(res)
        for name in ("C1", "C2"):
            self.assertTrue(rows[name]["missing"])
            self.assertIsNone(rows[name]["eff_rank_exp"])
            self.assertEqual(rows[name]["affected_scores"], 2)

    def test_unknown_target_and_type(self):
        records = recs(self.DATA)
        with self.assertRaises(ValueError):
            stability_experiment(records, {}, "judge", "judge", "NOPE")
        with self.assertRaises(ValueError):
            stability_experiment(records, {}, "judge", "contestant", "C1")

    def test_decisions_still_applied_in_experiment(self):
        records = recs(self.DATA)
        first_id = records[0]["id"]
        res = stability_experiment(
            records, {first_id: (True, "x")}, "judge", "judge", "JC"
        )
        # The kept decision count reflects only surviving scores:
        # id 1 belongs to JA, so it survives.
        self.assertEqual(res["experiment_counts"]["decisions"], 1)
        # Excluding the score's own judge drops the decision from the scenario.
        res2 = stability_experiment(
            records, {first_id: (True, "x")}, "judge", "judge", "JA"
        )
        self.assertEqual(res2["experiment_counts"]["decisions"], 0)


if __name__ == "__main__":
    unittest.main()
