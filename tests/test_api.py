"""End-to-end API tests with a temporary data directory."""
import os
import sys
import tempfile
import unittest

_tmp = tempfile.mkdtemp(prefix="calib_test_")
os.environ["CALIB_DATA_DIR"] = _tmp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import database as db  # noqa: E402
from app.main import app  # noqa: E402

db.init_db()
client = TestClient(app)
SID = "test-session"

CSV_BODY = (
    "选手,评委,项目,分数\n"
    "C1,JA,P1,8.30\nC2,JA,P1,8.00\nC3,JA,P1,7.90\nC4,JA,P1,7.80\n"
    "C1,JB,P1,8.30\nC2,JB,P1,8.00\nC3,JB,P1,7.85\nC4,JB,P1,7.75\n"
    "C1,JC,P1,5.0\nC2,JC,P1,9.5\nC3,JC,P1,7.2\nC4,JC,P1,6.8\n"
)


class TestAPI(unittest.TestCase):
    def test_01_health_and_empty(self):
        self.assertEqual(client.get("/api/health").json()["status"], "ok")
        r = client.get(f"/api/sessions/{SID}/analysis")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["counts"]["scores"], 0)

    def test_02_upload_errors(self):
        r = client.post(
            f"/api/sessions/{SID}/upload?scope=judge",
            files={"file": ("bad.csv", "选手,评委,分数\nA,J,9".encode(), "text/csv")},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("项目", r.json()["detail"])

    def test_03_upload_ok(self):
        r = client.post(
            f"/api/sessions/{SID}/upload?scope=judge",
            files={"file": ("s.csv", CSV_BODY.encode("utf-8-sig"), "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["counts"]["scores"], 12)
        self.assertGreaterEqual(len(data["reversals"]), 1)

    def test_04_persists_across_requests(self):
        r = client.get(f"/api/sessions/{SID}/analysis")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["counts"]["scores"], 12)

    def test_05_decision_and_rerank(self):
        # Override C1's first raw score (id 1) and verify effective totals change.
        before = client.get(f"/api/sessions/{SID}/analysis").json()
        r = client.patch(
            f"/api/sessions/{SID}/scores/1/decision",
            json={"use_raw": True, "reason": "复核确认"},
        )
        self.assertEqual(r.status_code, 200)
        after = r.json()
        self.assertEqual(after["counts"]["decisions"], 1)
        self.assertNotEqual(
            after["contestants"][0]["eff_total"], before["contestants"][0]["eff_total"]
        )

    def test_06_scope_switch(self):
        r = client.post(f"/api/sessions/{SID}/scope", json={"scope": "judge_item"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["scope"], "judge_item")
        self.assertEqual(len(r.json()["groups"]), 3)
        # switch back
        client.post(f"/api/sessions/{SID}/scope", json={"scope": "judge"})

    def test_07_reset(self):
        r = client.post(f"/api/sessions/{SID}/decisions/reset")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["counts"]["decisions"], 0)

    def test_08_export(self):
        r = client.get(f"/api/sessions/{SID}/export")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/csv", r.headers["content-type"])
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertIn("审计".encode(), r.content)

    def test_09_sample(self):
        r = client.post(f"/api/sessions/{SID}/sample?scope=judge")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(r.json()["counts"]["scores"], 0)

    def test_09a_stability_exclude_judge(self):
        # Sample: 6 contestants x 4 judges x 3 items = 72 scores.
        r = client.get(
            f"/api/sessions/{SID}/stability",
            params={"exclude_type": "judge", "target": "评委乙"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["experiment"], "ranking_stability")
        self.assertEqual(data["exclude_type"], "judge")
        self.assertEqual(data["exclude_target"], "评委乙")
        self.assertEqual(data["removed_scores"], 18)  # 6 contestants x 3 items
        self.assertEqual(data["experiment_counts"]["judges"], 3)
        self.assertEqual(len(data["rows"]), 6)
        row = data["rows"][0]
        for key in ("raw_total", "raw_total_exp", "raw_rank_delta",
                    "eff_rank", "eff_rank_exp", "affected_scores", "missing"):
            self.assertIn(key, row)
        self.assertFalse(any(rw["missing"] for rw in data["rows"]))
        # Read-only: baseline session data untouched.
        a = client.get(f"/api/sessions/{SID}/analysis").json()
        self.assertEqual(a["counts"]["scores"], 72)
        self.assertEqual(a["counts"]["judges"], 4)

    def test_09b_stability_exclude_item(self):
        r = client.get(
            f"/api/sessions/{SID}/stability",
            params={"exclude_type": "item", "target": "技术展示"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["removed_scores"], 24)  # 6 contestants x 4 judges
        self.assertEqual(data["experiment_counts"]["items"], 2)
        for rw in data["rows"]:
            self.assertEqual(rw["affected_scores"], 4)
            self.assertEqual(rw["exp_item_count"], 2)

    def test_09c_stability_scope_param_and_errors(self):
        # Explicit scope override is honored.
        r = client.get(
            f"/api/sessions/{SID}/stability",
            params={"exclude_type": "judge", "target": "评委甲", "scope": "judge_item"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["scope"], "judge_item")
        # Stored session scope must not have changed.
        self.assertEqual(
            client.get(f"/api/sessions/{SID}/analysis").json()["scope"], "judge"
        )
        # Bad exclude type.
        self.assertEqual(
            client.get(
                f"/api/sessions/{SID}/stability",
                params={"exclude_type": "bogus", "target": "X"},
            ).status_code,
            400,
        )
        # Unknown target.
        self.assertEqual(
            client.get(
                f"/api/sessions/{SID}/stability",
                params={"exclude_type": "judge", "target": "不存在的评委"},
            ).status_code,
            404,
        )
        # Unknown session.
        self.assertEqual(
            client.get(
                "/api/sessions/no-such-session/stability",
                params={"exclude_type": "judge", "target": "X"},
            ).status_code,
            404,
        )

    def test_10_clear(self):
        r = client.delete(f"/api/sessions/{SID}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["cleared"])
        r2 = client.get(f"/api/sessions/{SID}/analysis")
        self.assertEqual(r2.json()["counts"]["scores"], 0)
        # Export after clearing must fail
        self.assertEqual(client.get(f"/api/sessions/{SID}/export").status_code, 404)


if __name__ == "__main__":
    unittest.main()
