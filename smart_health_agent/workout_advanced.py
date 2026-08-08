"""Tier-1 advanced workout analyzers (POC).

Beyond the mean +/- SD excursion engine in ``workout_analyzer.py``, these add
methods better suited to non-stationary workout time series and to the
"what happened / what to work on" goal:

  * change-point segmentation  -> phase structure ("warmup / tempo / fade")
  * aerobic decoupling         -> HR:output drift, an endurance-limiter signal
  * HR recovery                -> how fast HR falls after efforts (fitness)
  * rolling-baseline excursions -> local band, robust to slow drift

Each analyzer returns structured data plus a list of narrative dicts tagged
with ``method`` so the harness can show what each surfaces on the same run.
Pure compute; reuses ``load_workout_df`` / config from ``workout_analyzer``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from workout_analyzer import METRIC_CONFIG, _LABEL, _fmt


# --------------------------------------------------------------------------
# Change-point segmentation (numpy binary segmentation, L2 mean-shift cost)
# --------------------------------------------------------------------------
def detect_change_points(values: np.ndarray, min_size: int = 90,
                         max_cps: int = 8, min_shift_sd: float = 0.75) -> List[int]:
    """Return sorted change-point indices where the series mean shifts.

    Binary segmentation: recursively split the segment whose split most reduces
    within-segment squared error, provided the mean shift exceeds
    ``min_shift_sd`` * global std and both sides are >= ``min_size`` samples.
    Lightweight stand-in for ruptures/PELT (no extra dependency for the POC).
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    gstd = np.nanstd(v)
    if n < 2 * min_size or gstd == 0 or not np.isfinite(gstd):
        return []

    def best_split(lo: int, hi: int):
        seg = v[lo:hi]
        m = len(seg)
        if m < 2 * min_size:
            return None
        # cumulative sums for O(1) segment means
        csum = np.cumsum(seg)
        total = csum[-1]
        best_gain, best_t = 0.0, None
        for t in range(min_size, m - min_size):
            left_mean = csum[t - 1] / t
            right_mean = (total - csum[t - 1]) / (m - t)
            shift = abs(left_mean - right_mean)
            # weight the shift by how balanced the split is
            gain = shift * min(t, m - t)
            if gain > best_gain and shift >= min_shift_sd * gstd:
                best_gain, best_t = gain, lo + t
        return best_t

    cps: List[int] = []
    segments = [(0, n)]
    while len(cps) < max_cps:
        candidate = None
        for (lo, hi) in segments:
            t = best_split(lo, hi)
            if t is not None:
                candidate = (lo, hi, t)
                break
        if candidate is None:
            break
        lo, hi, t = candidate
        cps.append(t)
        segments.remove((lo, hi))
        segments.extend([(lo, t), (t, hi)])
        segments.sort()
    return sorted(cps)


def segment_workout(df: pd.DataFrame, covered: List[str],
                    primary: str = "heart_rate") -> Dict[str, Any]:
    """Segment on the primary metric; summarise each phase and the biggest
    transitions."""
    if primary not in df.columns:
        primary = covered[0] if covered else None
    if primary is None:
        return {"segments": [], "narratives": []}

    series = df[primary].interpolate(limit_direction="both")
    idx = df.index.to_numpy()
    cps = detect_change_points(series.to_numpy())
    bounds = [0] + cps + [len(df)]

    segments = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sl = df.iloc[a:b]
        start_s, end_s = int(idx[a]), int(idx[b - 1])
        seg = {
            "start_min": round(start_s / 60.0, 1),
            "end_min": round(end_s / 60.0, 1),
            "duration_s": end_s - start_s + 1,
            "means": {m: round(float(sl[m].dropna().mean()), 1)
                      for m in covered if m in sl and sl[m].notna().any()},
        }
        segments.append(seg)

    # narrate the largest primary-metric step between adjacent phases
    narr = []
    label = _LABEL.get(primary, primary)
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        pv, cv = prev["means"].get(primary), cur["means"].get(primary)
        if pv is None or cv is None:
            continue
        step = cv - pv
        # report only meaningful phase steps
        meaningful = (abs(step) >= 8) if primary == "heart_rate" \
            else (abs(step) >= 0.15 * (abs(pv) or 1))
        if meaningful:
            direction = "up" if step > 0 else "down"
            narr.append({
                "minute": cur["start_min"],
                "method": "change_point",
                "metric": primary,
                "confidence": "medium",
                "quality": "neutral",
                "text": (f"Phase change at minute {cur['start_min']:g}: {label} stepped "
                         f"{direction} from {_fmt(primary, pv)} to {_fmt(primary, cv)} "
                         f"(phase lasted ~{round(cur['duration_s']/60,1):g} min)."),
            })
    return {"segments": segments, "change_points_min": [round(int(idx[c]) / 60.0, 1) for c in cps],
            "narratives": narr}


# --------------------------------------------------------------------------
# Aerobic decoupling (HR:output efficiency drift, first half vs second half)
# --------------------------------------------------------------------------
def aerobic_decoupling(df: pd.DataFrame, covered: List[str]) -> Dict[str, Any]:
    """First-half vs second-half efficiency (output/HR) drift for a steady
    effort. output = power if present, else speed. Only meaningful when the
    effort is reasonably steady (not intervals) — flagged via `steady`.
    """
    if "heart_rate" not in covered:
        return {"result": None, "narratives": []}
    output = "power_w" if "power_w" in covered else ("speed_mps" if "speed_mps" in covered else None)
    if output is None:
        return {"result": None, "narratives": []}

    d = df[["heart_rate", output]].dropna()
    if len(d) < 600:  # need ~10 min of paired data
        return {"result": None, "narratives": []}

    half = len(d) // 2
    first, second = d.iloc[:half], d.iloc[half:]

    def ef(seg):
        hr = seg["heart_rate"].mean()
        return (seg[output].mean() / hr) if hr else None

    ef1, ef2 = ef(first), ef(second)
    if not ef1 or not ef2:
        return {"result": None, "narratives": []}
    decoupling = (ef1 - ef2) / ef1 * 100.0

    # steadiness: coefficient of variation of the output over the whole effort
    cv = d[output].std(ddof=0) / d[output].mean() if d[output].mean() else 1.0
    steady = cv < 0.20  # intervals blow past this

    result = {
        "output_metric": output,
        "decoupling_pct": round(float(decoupling), 1),
        "ef_first": round(float(ef1), 4),
        "ef_second": round(float(ef2), 4),
        "steady": bool(steady),
        "cv": round(float(cv), 3),
    }

    narr = []
    if steady:
        if decoupling >= 5:
            verdict = ("high — your heart rate climbed relative to pace as the effort "
                       "went on, a sign your aerobic base is the limiter (or heat/fatigue).")
            quality = "bad"
        elif decoupling <= -3:
            verdict = "negative — you warmed into it and got more efficient; well-paced."
            quality = "good"
        else:
            verdict = "low — efficiency held steady, a well-supported aerobic effort."
            quality = "good"
        omit = _LABEL.get(output, output)
        narr.append({
            "minute": round(df.index[len(df)//2] / 60.0, 1),
            "method": "decoupling",
            "metric": "heart_rate",
            "confidence": "high",
            "quality": quality,
            "text": (f"Aerobic decoupling {result['decoupling_pct']:g}% ({omit}:HR "
                     f"first vs second half): {verdict}"),
        })
    return {"result": result, "narratives": narr}


# --------------------------------------------------------------------------
# HR recovery (bpm drop in the 60s after HR peaks) — works HR-only
# --------------------------------------------------------------------------
def hr_recovery(df: pd.DataFrame, drop_window_s: int = 60,
                min_prominence: float = 8.0) -> Dict[str, Any]:
    """Detect HR peaks and measure the bpm decline over the following window.
    Reports the steepest and mean recovery. A larger 60s drop => fitter."""
    if "heart_rate" not in df.columns:
        return {"result": None, "narratives": []}
    hr = df["heart_rate"].interpolate(limit_direction="both")
    if hr.notna().sum() < 300:
        return {"result": None, "narratives": []}

    smooth = hr.rolling(5, center=True, min_periods=1).median()
    vals = smooth.to_numpy()
    idx = df.index.to_numpy()
    peaks, _ = find_peaks(vals, prominence=min_prominence, distance=120)

    recoveries = []
    pos = {int(idx[i]): i for i in range(len(idx))}
    for p in peaks:
        t0 = int(idx[p])
        t1 = t0 + drop_window_s
        if t1 not in pos:
            # nearest available within a couple seconds
            cand = [s for s in (t1, t1 - 1, t1 + 1, t1 - 2, t1 + 2) if s in pos]
            if not cand:
                continue
            t1 = cand[0]
        drop = float(vals[p] - vals[pos[t1]])
        if drop > 0:
            recoveries.append({"minute": round(t0 / 60.0, 1),
                               "peak_hr": round(float(vals[p]), 0),
                               "drop_60s": round(drop, 0)})
    if not recoveries:
        return {"result": None, "narratives": []}

    best = max(recoveries, key=lambda r: r["drop_60s"])
    mean_drop = round(float(np.mean([r["drop_60s"] for r in recoveries])), 0)
    result = {"peaks": len(recoveries), "best": best, "mean_drop_60s": mean_drop}

    narr = [{
        "minute": best["minute"],
        "method": "hr_recovery",
        "metric": "heart_rate",
        "confidence": "medium",
        "quality": "neutral",
        "text": (f"After the effort at minute {best['minute']:g} (peak {best['peak_hr']:g} bpm), "
                 f"your heart rate fell {best['drop_60s']:g} bpm in the next 60s"
                 + (f"; across {result['peaks']} efforts it averaged {mean_drop:g} bpm/60s. "
                    "Faster drop = better cardiovascular recovery." if result['peaks'] > 1
                    else ". Faster drop = better cardiovascular recovery.")),
    }]
    return {"result": result, "narratives": narr}


# --------------------------------------------------------------------------
# Rolling-baseline excursions (local band; robust to slow drift)
# --------------------------------------------------------------------------
def rolling_band_excursions(df: pd.DataFrame, covered: List[str],
                            window_s: int = 120, k: float = 2.5,
                            min_duration_s: int = 30) -> Dict[str, Any]:
    """Excursions outside a LOCAL mean +/- k*std (rolling window). Flags sudden
    departures from recent behaviour that a global SD band misses (and ignores
    slow drift the global band over-flags)."""
    findings, narr = [], []
    for m in covered:
        s = df[m].interpolate(limit_direction="both")
        roll_mean = s.rolling(window_s, center=True, min_periods=window_s // 2).mean()
        roll_std = s.rolling(window_s, center=True, min_periods=window_s // 2).std(ddof=0)
        upper, lower = roll_mean + k * roll_std, roll_mean - k * roll_std
        mask = (s > upper) | (s < lower)
        mask = mask.fillna(False)
        if not mask.any():
            continue
        run_id = (mask != mask.shift()).cumsum()
        for _, grp in mask[mask].groupby(run_id[mask]):
            a, b = int(grp.index.min()), int(grp.index.max())
            if b - a + 1 < min_duration_s:
                continue
            seg = s.loc[a:b]
            local = roll_mean.loc[a:b].mean()
            peak = float(seg.max() if seg.mean() > local else seg.min())
            findings.append({"metric": m, "start_min": round(a / 60.0, 1),
                             "duration_s": b - a + 1, "peak_value": round(peak, 2),
                             "local_baseline": round(float(local), 2)})
    findings.sort(key=lambda f: abs(f["peak_value"] - f["local_baseline"]), reverse=True)
    for f in findings[:5]:
        label = _LABEL.get(f["metric"], f["metric"])
        direction = "above" if f["peak_value"] > f["local_baseline"] else "below"
        narr.append({
            "minute": f["start_min"],
            "method": "rolling_band",
            "metric": f["metric"],
            "confidence": "medium",
            "quality": "neutral",
            "text": (f"At minute {f['start_min']:g}, {label} jumped {direction} its recent "
                     f"local baseline ({_fmt(f['metric'], f['peak_value'])} vs "
                     f"~{_fmt(f['metric'], f['local_baseline'])}) — a sudden local shift."),
        })
    return {"findings": findings, "narratives": narr}


# --------------------------------------------------------------------------
# Aggregate entry point
# --------------------------------------------------------------------------
def advanced_analyses(df: pd.DataFrame, covered: List[str]) -> Dict[str, Any]:
    """Run all Tier-1 analyzers on a loaded workout DataFrame."""
    seg = segment_workout(df, covered)
    dec = aerobic_decoupling(df, covered)
    hrr = hr_recovery(df)
    roll = rolling_band_excursions(df, covered)
    narratives = (seg["narratives"] + dec["narratives"]
                  + hrr["narratives"] + roll["narratives"])
    return {
        "segments": seg["segments"],
        "change_points_min": seg.get("change_points_min", []),
        "decoupling": dec["result"],
        "hr_recovery": hrr["result"],
        "rolling_excursions": roll["findings"],
        "narratives": narratives,
    }
