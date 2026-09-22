"""Robust score calibration with median and MAD (median absolute deviation).

Each score is turned into a robust z-score within its calibration group
(either per judge, or per judge x item):

    z = (score - group_median) / (group_MAD * 1.4826)

and then mapped back onto the global score scale:

    calibrated = global_median + z * global_MAD_scaled

1.4826 makes MAD a consistent estimator of the standard deviation for
normally distributed data. Groups without spread (MAD == 0) get z = 0,
i.e. every score maps to the global median.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

MAD_K = 1.4826

SCOPE_JUDGE = "judge"
SCOPE_JUDGE_ITEM = "judge_item"
SCOPES = (SCOPE_JUDGE, SCOPE_JUDGE_ITEM)
SCOPE_LABELS = {
    SCOPE_JUDGE: "按评委（跨项目汇总）",
    SCOPE_JUDGE_ITEM: "按评委×项目",
}

TIE_EPS = 1e-9


def robust_params(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (median, raw MAD, MAD scaled by 1.4826)."""
    med = median(values)
    raw_mad = median(abs(v - med) for v in values)
    return med, raw_mad, raw_mad * MAD_K


def competition_rank(pairs: Iterable[tuple[Any, float]]) -> dict[Any, int]:
    """Rank (key, total) pairs highest-first with competition ranking (1224)."""
    ordered = sorted(pairs, key=lambda p: (-p[1], str(p[0])))
    ranks: dict[Any, int] = {}
    for i, (key, total) in enumerate(ordered):
        if i > 0 and abs(total - ordered[i - 1][1]) <= TIE_EPS:
            ranks[key] = ranks[ordered[i - 1][0]]
        else:
            ranks[key] = i + 1
    return ranks


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def analyze(
    records: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, tuple[bool, str]],
    scope: str = SCOPE_JUDGE,
) -> dict[str, Any]:
    """Run the full calibration pipeline.

    records: dicts with id, contestant, judge, item, score.
    decisions: score_id -> (use_raw, reason). Missing decisions default to
    the calibrated score.
    """
    if scope not in SCOPES:
        raise ValueError(f"未知校准范围: {scope}")

    def group_key(rec: Mapping[str, Any]) -> tuple:
        return (rec["judge"],) if scope == SCOPE_JUDGE else (rec["judge"], rec["item"])

    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[group_key(rec)].append(rec)

    group_params: dict[tuple, tuple[float, float, float]] = {
        key: robust_params([r["score"] for r in rs]) for key, rs in grouped.items()
    }

    if records:
        global_med, global_raw_mad, global_mad = robust_params(
            [r["score"] for r in records]
        )
    else:
        global_med = global_raw_mad = global_mad = 0.0

    # Per-record calibrated values.
    enriched: dict[int, dict[str, Any]] = {}
    for rec in records:
        med, raw_mad, mad = group_params[group_key(rec)]
        z = (rec["score"] - med) / mad if mad > 0 else 0.0
        calibrated = global_med + z * global_mad
        use_raw, reason = decisions.get(rec["id"], (False, ""))
        effective = rec["score"] if use_raw else calibrated
        enriched[rec["id"]] = {
            "id": rec["id"],
            "judge": rec["judge"],
            "raw": rec["score"],
            "group_median": med,
            "group_raw_mad": raw_mad,
            "group_mad": mad,
            "z": z,
            "calibrated": calibrated,
            "use_raw": bool(use_raw),
            "reason": reason or "",
            "effective": effective,
        }

    # Preserve first-appearance order for judges / items / contestants.
    judges: list[str] = list(dict.fromkeys(r["judge"] for r in records))
    items: list[str] = list(dict.fromkeys(r["item"] for r in records))
    contestants: list[str] = list(dict.fromkeys(r["contestant"] for r in records))

    by_ci: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        by_ci[(rec["contestant"], rec["item"])].append(rec)

    raw_totals: dict[str, float] = {}
    cal_totals: dict[str, float] = {}
    eff_totals: dict[str, float] = {}
    raw_override_count: dict[str, int] = defaultdict(int)

    contestant_blocks: list[dict[str, Any]] = []
    for contestant in contestants:
        item_blocks = []
        r_total = c_total = e_total = 0.0
        item_count = 0
        for item in items:
            rows = by_ci.get((contestant, item))
            if not rows:
                continue
            item_count += 1
            score_rows = [enriched[r["id"]] for r in rows]
            score_rows.sort(key=lambda s: judges.index(s["judge"]) if s["judge"] in judges else 0)
            raw_avg = _avg([s["raw"] for s in score_rows])
            cal_avg = _avg([s["calibrated"] for s in score_rows])
            eff_avg = _avg([s["effective"] for s in score_rows])
            r_total += raw_avg
            c_total += cal_avg
            e_total += eff_avg
            for s in score_rows:
                if s["use_raw"]:
                    raw_override_count[contestant] += 1
            item_blocks.append(
                {
                    "item": item,
                    "raw_avg": raw_avg,
                    "cal_avg": cal_avg,
                    "eff_avg": eff_avg,
                    "scores": score_rows,
                }
            )
        raw_totals[contestant] = r_total
        cal_totals[contestant] = c_total
        eff_totals[contestant] = e_total
        contestant_blocks.append(
            {
                "contestant": contestant,
                "item_count": item_count,
                "raw_total": r_total,
                "cal_total": c_total,
                "eff_total": e_total,
                "raw_overrides": raw_override_count[contestant],
                "items": item_blocks,
            }
        )

    raw_rank = competition_rank(raw_totals.items())
    cal_rank = competition_rank(cal_totals.items())
    eff_rank = competition_rank(eff_totals.items())

    for block in contestant_blocks:
        name = block["contestant"]
        block["raw_rank"] = raw_rank[name]
        block["cal_rank"] = cal_rank[name]
        block["eff_rank"] = eff_rank[name]
        block["rank_delta"] = raw_rank[name] - cal_rank[name]  # >0: moved up

    # Pairwise rank reversals between raw ranking and calibrated ranking.
    reversals: list[dict[str, Any]] = []
    for i, a in enumerate(contestants):
        for b in contestants[i + 1:]:
            ra, rb = raw_totals[a], raw_totals[b]
            ca, cb = cal_totals[a], cal_totals[b]
            if ra > rb + TIE_EPS and cb > ca + TIE_EPS:
                reversals.append(
                    {
                        "winner_before": a,
                        "winner_after": b,
                        "raw_rank_before": raw_rank[a],
                        "raw_rank_loser": raw_rank[b],
                        "cal_rank_winner": cal_rank[b],
                        "cal_rank_loser": cal_rank[a],
                        "raw_gap": ra - rb,
                        "cal_gap": cb - ca,
                    }
                )
            elif rb > ra + TIE_EPS and ca > cb + TIE_EPS:
                reversals.append(
                    {
                        "winner_before": b,
                        "winner_after": a,
                        "raw_rank_before": raw_rank[b],
                        "raw_rank_loser": raw_rank[a],
                        "cal_rank_winner": cal_rank[a],
                        "cal_rank_loser": cal_rank[b],
                        "raw_gap": rb - ra,
                        "cal_gap": ca - cb,
                    }
                )
    reversals.sort(key=lambda x: -(x["raw_gap"] + x["cal_gap"]))

    contestant_blocks.sort(key=lambda blk: (blk["eff_rank"], blk["contestant"]))

    groups = []
    for key, rs in sorted(grouped.items(), key=lambda kv: kv[0]):
        med, raw_mad, mad = group_params[key]
        entry = {
            "judge": key[0],
            "n": len(rs),
            "median": med,
            "raw_mad": raw_mad,
            "mad": mad,
        }
        if scope == SCOPE_JUDGE_ITEM:
            entry["item"] = key[1]
        groups.append(entry)

    return {
        "scope": scope,
        "scope_label": SCOPE_LABELS[scope],
        "reference": {
            "global_median": global_med,
            "global_raw_mad": global_raw_mad,
            "global_mad": global_mad,
            "mad_k": MAD_K,
        },
        "judges": judges,
        "items": items,
        "groups": groups,
        "contestants": contestant_blocks,
        "reversals": reversals,
        "counts": {
            "scores": len(records),
            "contestants": len(contestants),
            "judges": len(judges),
            "items": len(items),
            "decisions": sum(1 for v in decisions.values() if v[0]),
        },
    }
