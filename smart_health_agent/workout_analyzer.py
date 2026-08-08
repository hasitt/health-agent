"""Per-workout statistical analysis (POC).

For a single Garmin activity, treat each stored metric (heart rate, speed,
cadence, power, elevation, temperature) as a 1 Hz time series and:

  1. summarise it (mean, standard deviation, range, coverage),
  2. find contiguous *excursions* outside a mean +/- k*std band,
  3. describe what happened to the same metric and to other metrics in a
     window before/during/after each excursion,
  4. correlate metric pairs within the excursion window, and
  5. turn the findings into plain-language "data points".

Reference band is statistical (the workout's own mean/std). Physiological
thresholds (lactate threshold HR, FTP, HR zones) are not stored yet, so the
``ReferenceBand`` seam is left pluggable for a future ``PhysiologicalBand``.

Pure compute, no LangChain deps — importable and testable against the DB
directly. Mirrors the style of ``trend_analyzer.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from database import db

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "workout_analyzer/1.0.0"

# Per-metric semantics.
#   direction: is a HIGH value good, bad, or context-dependent for this metric?
#     'good'    -> an above-band excursion is favourable (e.g. speed, power)
#     'bad'     -> an above-band excursion is unfavourable
#     'context' -> depends on activity intent; phrase neutrally, let a human
#                  / the chat LLM judge
#   min_coverage: fraction of non-null samples required before we analyse it.
#   glitch_floor: values below this in the first ~30s are treated as sensor
#     warmup dropout, not real excursions (a real run had HR=56 at second 1).
METRIC_CONFIG: Dict[str, Dict[str, Any]] = {
    "heart_rate":    {"unit": "bpm",  "direction": "context", "min_coverage": 0.80, "glitch_floor": 60},
    "speed_mps":     {"unit": "m/s",  "direction": "good",    "min_coverage": 0.50, "glitch_floor": None},
    "cadence":       {"unit": "spm",  "direction": "good",    "min_coverage": 0.50, "glitch_floor": None},
    "power_w":       {"unit": "W",    "direction": "good",    "min_coverage": 0.50, "glitch_floor": None},
    "elevation_m":   {"unit": "m",    "direction": "context", "min_coverage": 0.50, "glitch_floor": None},
    # distance_m intentionally excluded: 0% populated (parser bug, deferred).
    # temperature_c intentionally excluded: it's ambient/device sensor reading
    # (equilibrates over the first minutes), not a training metric — it produced
    # only warmup false positives in the POC sweep.
}
ANALYZABLE_METRICS = list(METRIC_CONFIG)

# Physiologically impossible HR values -> treat as sensor error before stats.
_HR_HARD_MIN, _HR_HARD_MAX = 30, 220

_MIN_SAMPLES = 120          # skip activities shorter than ~2 min
_WARMUP_GUARD_S = 60        # window over which glitch_floor suppression applies
_TRIM_START_S = 60          # drop the opening warmup ramp before stats

# Pairs that are physically coupled by definition (moving faster => more power,
# etc.). Correlating them yields r~1 with no insight, so exclude from the
# cross-metric narrative. Order-independent.
_TRIVIAL_PAIRS = {frozenset({"speed_mps", "power_w"})}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_workout_df(activity_id: str) -> pd.DataFrame:
    """Load one activity's samples into a DataFrame indexed by elapsed_seconds.

    Numeric-coerces the metric columns; does NOT interpolate gaps. Returns an
    empty DataFrame if the activity has no samples.
    """
    rows = db.get_activity_samples(str(activity_id))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "elapsed_seconds" not in df.columns:
        return pd.DataFrame()

    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    df = df.dropna(subset=["elapsed_seconds"]).astype({"elapsed_seconds": int})
    df = df.set_index("elapsed_seconds").sort_index()

    for m in ANALYZABLE_METRICS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Clamp physiologically impossible HR to NaN so it doesn't poison stats.
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        df.loc[(hr < _HR_HARD_MIN) | (hr > _HR_HARD_MAX), "heart_rate"] = np.nan

    return df


# --------------------------------------------------------------------------
# Per-metric summary stats
# --------------------------------------------------------------------------
def summarize_metric(series: pd.Series, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Summary stats for one metric, or None if too sparse / no variance."""
    total = len(series)
    valid = series.dropna()
    if total == 0:
        return None
    coverage = len(valid) / total
    if coverage < cfg["min_coverage"] or len(valid) < 2:
        return None

    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return None  # constant series -> no excursions possible

    return {
        "mean": float(valid.mean()),
        "std": std,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "valid_count": int(len(valid)),
        "coverage": round(coverage, 3),
    }


# --------------------------------------------------------------------------
# Reference band (pluggable seam)
# --------------------------------------------------------------------------
class ReferenceBand:
    """Given a metric + its summary, return (lower, upper) band edges."""
    kind = "abstract"

    def bounds(self, metric: str, summary: Dict[str, Any],
               series: pd.Series) -> Tuple[float, float]:
        raise NotImplementedError


class StatisticalBand(ReferenceBand):
    """mean +/- k*std from the workout itself. The POC default."""
    kind = "statistical"

    def __init__(self, k: float = 1.0):
        self.k = k

    def bounds(self, metric, summary, series):
        m, s = summary["mean"], summary["std"]
        return m - self.k * s, m + self.k * s


# Reserved for later: PhysiologicalBand(lactate threshold HR / FTP / HR zones),
# falling back to StatisticalBand for metrics with no known threshold. Not built.


# --------------------------------------------------------------------------
# Excursion detection
# --------------------------------------------------------------------------
def detect_excursions(series: pd.Series, lower: float, upper: float,
                      cfg: Dict[str, Any], summary: Dict[str, Any],
                      min_duration_s: int = 45,
                      smooth_window_s: int = 5) -> List[Dict[str, Any]]:
    """Contiguous runs where a smoothed series sits outside [lower, upper].

    Median smoothing kills single-sample spikes; a warmup/glitch guard drops
    early below-floor dropouts. Returns one dict per excursion run that lasts
    at least ``min_duration_s``.
    """
    s = series.copy()
    # Median rolling smooth: robust to single-sample spikes, keeps transitions.
    smoothed = s.rolling(smooth_window_s, center=True, min_periods=1).median()

    above = smoothed > upper
    below = smoothed < lower

    # Suppress sensor warmup dropout: below-band values under glitch_floor in
    # the opening seconds are not real excursions.
    floor = cfg.get("glitch_floor")
    if floor is not None:
        early = s.index.to_series() < _WARMUP_GUARD_S
        below = below & ~((s < floor) & early.values)

    mean, std = summary["mean"], summary["std"]
    excursions: List[Dict[str, Any]] = []

    for direction, mask in (("above", above), ("below", below)):
        if not mask.any():
            continue
        # Group contiguous True runs.
        run_id = (mask != mask.shift()).cumsum()
        for _, grp in mask[mask].groupby(run_id[mask]):
            idx = grp.index
            start_s, end_s = int(idx.min()), int(idx.max())
            duration = end_s - start_s + 1
            if duration < min_duration_s:
                continue
            raw = s.loc[start_s:end_s].dropna()
            if raw.empty:
                continue
            peak = float(raw.max() if direction == "above" else raw.min())
            excursions.append({
                "metric": series.name,
                "direction": direction,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration,
                "start_min": round(start_s / 60.0, 1),
                "end_min": round(end_s / 60.0, 1),
                "peak_value": round(peak, 2),
                "mean_value": round(float(raw.mean()), 2),
                "severity_sd": round(abs(peak - mean) / std, 2),
                "quality": _quality(cfg["direction"], direction),
            })

    excursions.sort(key=lambda e: e["start_s"])
    return excursions


def _quality(metric_direction: str, exc_direction: str) -> str:
    """Map (is-high-good?) x (above/below) -> good|bad|neutral."""
    if metric_direction == "context":
        return "neutral"
    high_is_good = metric_direction == "good"
    is_high = exc_direction == "above"
    return "good" if (high_is_good == is_high) else "bad"


# --------------------------------------------------------------------------
# Windowed context around an excursion
# --------------------------------------------------------------------------
def excursion_context(df: pd.DataFrame, exc: Dict[str, Any], metric: str,
                      covered_metrics: List[str],
                      window_s: int = 600) -> Dict[str, Any]:
    """Before/during/after means of the excursion metric plus in-window means
    of the other covered metrics."""
    start_s, end_s = exc["start_s"], exc["end_s"]
    lo = max(df.index.min(), start_s - window_s)
    hi = min(df.index.max(), end_s + window_s)

    def _mean(col, a, b):
        if col not in df.columns:
            return None
        seg = df.loc[a:b, col].dropna()
        return round(float(seg.mean()), 2) if len(seg) else None

    before = _mean(metric, lo, start_s - 1)
    during = _mean(metric, start_s, end_s)
    after = _mean(metric, end_s + 1, hi)
    delta = round(after - before, 2) if (before is not None and after is not None) else None

    others = {}
    for other in covered_metrics:
        if other == metric:
            continue
        others[other] = {
            "during": _mean(other, start_s, end_s),
            "after": _mean(other, end_s + 1, hi),
            "before": _mean(other, lo, start_s - 1),
        }

    return {
        "window_s": window_s,
        "self": {"before": before, "during": during, "after": after, "after_minus_before": delta},
        "others": others,
    }


# --------------------------------------------------------------------------
# Cross-metric correlation within a window
# --------------------------------------------------------------------------
def correlate_in_window(df: pd.DataFrame, metric: str, other: str,
                        start_s: int, end_s: int,
                        pad_s: int = 120, min_pairs: int = 10) -> Optional[Dict[str, Any]]:
    """Pearson r between two metrics over [start-pad, end+pad].

    Returns None (never raises) if either column is missing, too short, or has
    zero variance — the guard that keeps the agent loop stable (see 43e1d8f).
    """
    if metric not in df.columns or other not in df.columns:
        return None
    if frozenset({metric, other}) in _TRIVIAL_PAIRS:
        return None  # physically coupled; r~1 carries no insight
    lo = max(df.index.min(), start_s - pad_s)
    hi = min(df.index.max(), end_s + pad_s)
    seg = df.loc[lo:hi, [metric, other]].dropna()
    if len(seg) < min_pairs:
        return None
    a, b = seg[metric], seg[other]
    if a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return None
    try:
        r, _ = stats.pearsonr(a, b)
    except Exception:
        return None
    if not np.isfinite(r):
        return None
    return {"metric_a": metric, "metric_b": other, "r": round(float(r), 2), "n": int(len(seg))}


# --------------------------------------------------------------------------
# Narrative generation
# --------------------------------------------------------------------------
_LABEL = {
    "heart_rate": "heart rate", "speed_mps": "speed", "cadence": "cadence",
    "power_w": "power", "elevation_m": "elevation", "temperature_c": "temperature",
}


def _fmt(metric: str, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    unit = METRIC_CONFIG[metric]["unit"]
    return f"{value:g} {unit}"


def build_narratives(summaries: Dict[str, Dict[str, Any]],
                     excursions: List[Dict[str, Any]],
                     window_s: int, max_points: int = 8) -> List[Dict[str, Any]]:
    """Turn excursions (each carrying its context + correlations) into plain
    data points. Wording stays correlational, not causal. Keeps only the
    ``max_points`` most severe excursions so output stays readable."""
    ranked = sorted(excursions, key=lambda e: e["severity_sd"], reverse=True)[:max_points]
    points: List[Dict[str, Any]] = []
    for exc in ranked:
        metric = exc["metric"]
        label = _LABEL.get(metric, metric)
        summ = summaries[metric]
        ctx = exc["context"]
        corrs = exc.get("correlations", [])

        verb = "rose to" if exc["direction"] == "above" else "dropped to"
        head = (f"Your {label} {verb} {_fmt(metric, exc['peak_value'])} "
                f"({exc['severity_sd']} SD {'above' if exc['direction']=='above' else 'below'} "
                f"your workout average of {_fmt(metric, round(summ['mean'],1))}) "
                f"at minute {exc['start_min']:g} and held for ~{round(exc['duration_s']/60,1):g} min.")

        # Strongest supporting correlation (|r| >= 0.5) drives a cross-metric clause.
        strong = max((c for c in corrs if abs(c["r"]) >= 0.5),
                     key=lambda c: abs(c["r"]), default=None)
        confidence = "medium"
        tail = ""
        if strong:
            other = strong["metric_b"] if strong["metric_a"] == metric else strong["metric_a"]
            o_ctx = ctx["others"].get(other, {})
            ob, oa = o_ctx.get("before"), o_ctx.get("after")
            move = ""
            if ob is not None and oa is not None:
                trend = "higher" if oa > ob else "lower"
                move = (f" Over the following {int(window_s/60)} minutes your "
                        f"{_LABEL.get(other, other)} averaged {_fmt(other, oa)} vs "
                        f"{_fmt(other, ob)} before — {trend}")
            sign = "inverse" if strong["r"] < 0 else "aligned"
            tail = f"{move} ({sign}, r={strong['r']:g})." if move else \
                   f" This coincided with {_LABEL.get(other, other)} ({sign}, r={strong['r']:g})."
            confidence = "high"
        else:
            d = ctx["self"]["after_minus_before"]
            if d is not None:
                trend = "higher" if d > 0 else "lower"
                tail = (f" {label.capitalize()} then averaged {abs(d):g} "
                        f"{METRIC_CONFIG[metric]['unit']} {trend} over the next "
                        f"{int(window_s/60)} minutes than before.")

        points.append({
            "minute": exc["start_min"],
            "metric": metric,
            "quality": exc["quality"],
            "confidence": confidence,
            "text": (head + tail).strip(),
        })
    points.sort(key=lambda p: p["minute"])
    return points


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def analyze_workout(activity_id: str, user_id: int = 1,
                    band: Optional[ReferenceBand] = None, k: float = 1.5,
                    window_s: int = 600,
                    activity_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full per-workout analysis. Recomputes each call (no cache in the POC)."""
    band = band or StatisticalBand(k=k)
    df = load_workout_df(activity_id)

    result: Dict[str, Any] = {
        "activity_id": str(activity_id),
        "extractor_version": EXTRACTOR_VERSION,
        "band": {"type": band.kind, "k": getattr(band, "k", None)},
        "activity_meta": activity_meta or {},
        "metrics_analyzed": [],
        "metrics_skipped": {},
        "summaries": {},
        "excursions": [],
        "correlations": [],
        "narratives": [],
        "degraded": True,
    }

    if df.empty or len(df) < _MIN_SAMPLES:
        result["metrics_skipped"]["_all"] = f"insufficient samples ({len(df)})"
        return result

    # Drop the opening warmup ramp (HR rising from rest, GPS acquiring) so it
    # doesn't skew each metric's mean/std and generate minute-0 false positives.
    # Only if enough of the workout remains afterwards.
    if len(df.loc[df.index >= _TRIM_START_S]) >= _MIN_SAMPLES:
        df = df.loc[df.index >= _TRIM_START_S]

    # 1. summarise each metric, keep those with enough coverage/variance
    summaries: Dict[str, Dict[str, Any]] = {}
    for m in ANALYZABLE_METRICS:
        cfg = METRIC_CONFIG[m]
        if m not in df.columns:
            result["metrics_skipped"][m] = "not present"
            continue
        summ = summarize_metric(df[m], cfg)
        if summ is None:
            result["metrics_skipped"][m] = "low coverage or zero variance"
            continue
        summaries[m] = summ

    covered = list(summaries)
    result["metrics_analyzed"] = covered
    result["summaries"] = summaries
    if not covered:
        return result

    # 2-4. per metric: excursions -> context -> correlations
    all_excursions: List[Dict[str, Any]] = []
    all_correlations: List[Dict[str, Any]] = []
    for m in covered:
        cfg = METRIC_CONFIG[m]
        lower, upper = band.bounds(m, summaries[m], df[m])
        excs = detect_excursions(df[m], lower, upper, cfg, summaries[m])
        for exc in excs:
            exc["context"] = excursion_context(df, exc, m, covered, window_s)
            corrs = []
            for other in covered:
                if other == m:
                    continue
                c = correlate_in_window(df, m, other, exc["start_s"], exc["end_s"])
                if c:
                    corrs.append(c)
            exc["correlations"] = corrs
            all_correlations.extend(corrs)
        all_excursions.extend(excs)

    all_excursions.sort(key=lambda e: e["start_s"])
    result["excursions"] = all_excursions
    result["correlations"] = all_correlations
    result["degraded"] = len(covered) < 2

    # 5. narratives
    result["narratives"] = build_narratives(summaries, all_excursions, window_s)
    return result
