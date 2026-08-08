"""POC evaluation harness for workout_analyzer.

Sweeps stored Garmin activities, runs the per-workout analysis on a spread of
GPS (multi-metric) and HR-only sessions, and prints the results for a human to
judge whether the insights are worth anything. Not wired to the agent/UI.

Usage:
    python3 poc_workout_analysis.py                 # default spread
    python3 poc_workout_analysis.py --activity ID   # one activity, verbose
    python3 poc_workout_analysis.py --gps 5 --hr 3  # N of each kind
    python3 poc_workout_analysis.py --k 2.0         # band width override
"""

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, ".")
logging.getLogger("database").setLevel(logging.WARNING)

from database import db
import workout_analyzer as wa

USER_ID = 1
GPS_TYPES = ("running", "treadmill_running", "road_biking", "cycling",
             "walking", "open_water_swimming", "kayaking_v2")


def _activities_with_samples(limit_per_kind, k_gps, k_hr):
    """Return (gps_list, hr_only_list) of activity rows that have samples."""
    rows = db.get_garmin_activities(USER_ID, "2000-01-01", "2100-01-01")
    rows = [r for r in rows if db.has_activity_samples(str(r["activity_id"]))]
    # newest first
    rows.sort(key=lambda r: str(r.get("start_time", "")), reverse=True)
    gps, hr_only = [], []
    for r in rows:
        is_gps = r.get("activity_type") in GPS_TYPES
        if is_gps and len(gps) < k_gps:
            gps.append(r)
        elif not is_gps and len(hr_only) < k_hr:
            hr_only.append(r)
        if len(gps) >= k_gps and len(hr_only) >= k_hr:
            break
    return gps, hr_only


def _meta(row):
    return {
        "type": row.get("activity_type"),
        "start_time": row.get("start_time"),
        "duration_min": row.get("duration_minutes"),
        "avg_hr": row.get("avg_heart_rate"),
        "max_hr": row.get("max_heart_rate"),
        "distance_km": row.get("distance_km"),
    }


def _print_report(row, k, verbose=False):
    aid = str(row["activity_id"])
    meta = _meta(row)
    res = wa.analyze_workout(aid, user_id=USER_ID, k=k, activity_meta=meta)

    hdr = (f"{meta['type']:<20} {str(meta['start_time'])[:16]:<17} "
           f"{meta['duration_min'] or 0:>4}min  avg_hr={meta['avg_hr']}  "
           f"activity_id={aid}")
    print("\n" + "=" * 100)
    print(hdr)
    print("-" * 100)

    if not res["metrics_analyzed"]:
        print("  (no analyzable metrics — skipped: "
              + ", ".join(f"{m}:{why}" for m, why in res["metrics_skipped"].items()) + ")")
        return res

    tag = "HR-ONLY" if res["degraded"] else "MULTI-METRIC"
    print(f"  [{tag}] metrics: {', '.join(res['metrics_analyzed'])}"
          f"   excursions={len(res['excursions'])}  narratives={len(res['narratives'])}")
    if verbose:
        for m, s in res["summaries"].items():
            print(f"    {m:<14} mean={s['mean']:.1f} sd={s['std']:.1f} "
                  f"range=[{s['min']:.0f},{s['max']:.0f}] cov={s['coverage']:.0%} n={s['valid_count']}")

    # group narratives by method so each analyzer's contribution is visible
    methods = ["sd_excursion", "change_point", "decoupling", "hr_recovery", "rolling_band"]
    labels = {"sd_excursion": "SD excursions", "change_point": "Change-points (phases)",
              "decoupling": "Aerobic decoupling", "hr_recovery": "HR recovery",
              "rolling_band": "Rolling-baseline"}
    by_method = {}
    for n in res.get("narratives", []):
        by_method.setdefault(n.get("method", "sd_excursion"), []).append(n)

    if not res["narratives"]:
        print("  (no findings — steady workout)")
    for meth in methods:
        items = by_method.get(meth, [])
        if not items:
            continue
        print(f"  -- {labels[meth]} --")
        for n in sorted(items, key=lambda x: x["minute"]):
            print(f"     • [{n['minute']:g}m {n['confidence']}/{n['quality']}] {n['text']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", help="analyze one activity_id (verbose)")
    ap.add_argument("--gps", type=int, default=6, help="number of GPS activities")
    ap.add_argument("--hr", type=int, default=4, help="number of HR-only activities")
    ap.add_argument("--k", type=float, default=1.5, help="SD band width")
    args = ap.parse_args()

    db.connect()

    if args.activity:
        row = next((r for r in db.get_garmin_activities(USER_ID, "2000-01-01", "2100-01-01")
                    if str(r["activity_id"]) == args.activity), None)
        if not row:
            print(f"activity {args.activity} not found")
            return
        _print_report(row, args.k, verbose=True)
        return

    gps, hr_only = _activities_with_samples(None, args.gps, args.hr)
    print(f"POC sweep: {len(gps)} GPS + {len(hr_only)} HR-only activities, k={args.k}")

    total = with_insight = degraded = 0
    for row in gps + hr_only:
        res = _print_report(row, args.k, verbose=(row in gps))
        total += 1
        if res["narratives"]:
            with_insight += 1
        if res["degraded"] and res["metrics_analyzed"]:
            degraded += 1

    print("\n" + "=" * 100)
    print(f"SUMMARY: {total} workouts | {with_insight} produced >=1 insight "
          f"| {degraded} HR-only (degraded) | {total - with_insight} silent")


if __name__ == "__main__":
    main()
