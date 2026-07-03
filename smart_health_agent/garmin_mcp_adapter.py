"""
In-process adapter from the legacy ``sync_garmin_data`` interface to the
``garmin-mcp-server`` backend.

Drop-in replacement: change

    from garmin_utils import sync_garmin_data, initialize_garmin_client

to

    from garmin_mcp_adapter import sync_garmin_data

in ``smart_health_ollama.py``. The call at line ~294 keeps its signature.

What this module does:
  * Resolves Garmin credentials and the ``.garmin_tokens`` path through
    ``GarminMCPConfig`` (same env vars as the legacy code).
  * Lazily creates one shared ``GarminAuthenticator`` + ``GarminDataFetcher``.
  * For each date in the requested window, calls four MCP endpoints
    (daily summary, sleep, heart rate, stress with full detail) and writes
    the results to the existing SQLite tables via ``database.db.upsert_*``.
  * Honors incremental sync via ``db.get_sync_status`` / ``db.update_sync_status``.

This is in-process: no MCP/JSON-RPC plumbing is exercised. The MCP server's
caching, rate limiting, circuit breaker, and Pydantic validation come along
for free because we instantiate the same classes ``server.py`` uses.
"""

from __future__ import annotations

import asyncio
import os
import sys
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# garmin-mcp-server is not pip-installed; point at its src/ layout.
_MCP_SRC = Path(__file__).parent / "garmin-mcp-server" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from garmin_mcp.config import GarminMCPConfig  # noqa: E402
from garmin_mcp.auth import GarminAuthenticator  # noqa: E402
from garmin_mcp.data_fetcher import GarminDataFetcher  # noqa: E402
from garmin_mcp.exceptions import GarminMCPError  # noqa: E402

from database import db  # noqa: E402

logger = logging.getLogger(__name__)
load_dotenv()

# Legacy .env names the credential GARMIN_USERNAME; GarminMCPConfig expects
# GARMIN_EMAIL. Bridge it so both layouts work.
if "GARMIN_EMAIL" not in os.environ and "GARMIN_USERNAME" in os.environ:
    os.environ["GARMIN_EMAIL"] = os.environ["GARMIN_USERNAME"]

# Lazy singletons. The MCP cache lives on the fetcher, so reusing it across
# sync runs is a feature, not a bug.
_config: Optional[GarminMCPConfig] = None
_authenticator: Optional[GarminAuthenticator] = None
_fetcher: Optional[GarminDataFetcher] = None


async def _ensure_fetcher() -> GarminDataFetcher:
    global _config, _authenticator, _fetcher
    if _fetcher is not None:
        return _fetcher

    _config = GarminMCPConfig.from_env()
    _config.validate_required_settings()

    _authenticator = GarminAuthenticator(_config)
    await _authenticator.initialize()  # token-first, credential-fallback

    _fetcher = GarminDataFetcher(_authenticator, _config)
    return _fetcher


def _get_date_range_for_sync(days_to_sync: int) -> tuple[date, date]:
    """Inclusive [start, end] window: today and the previous N-1 days."""
    today = datetime.now().date()
    return today - timedelta(days=days_to_sync - 1), today


def _is_success(payload: dict) -> bool:
    """MCP endpoints return ``{"status": "success", ...}`` on success and
    an ``ErrorResponse`` shape otherwise. Treat anything without an explicit
    error_code as success-ish for backwards compatibility."""
    return payload.get("status") == "success" or "error_code" not in payload


async def sync_garmin_data(user_id: int, days_to_sync: int = 30,
                           force_refresh: bool = False) -> bool:
    """Sync the last ``days_to_sync`` days of Garmin data for ``user_id``.

    Same contract as ``garmin_utils.sync_garmin_data``: returns True on
    overall success (per-day failures are logged and skipped), False if the
    sync could not start (auth/config failure).
    """
    try:
        fetcher = await _ensure_fetcher()
    except GarminMCPError as e:
        logger.error("Failed to initialize Garmin MCP fetcher: %s", e)
        return False
    except Exception as e:
        logger.error("Unexpected error initializing Garmin MCP fetcher: %s", e)
        return False

    today = datetime.now().date()
    start_date, end_date = _get_date_range_for_sync(days_to_sync)
    logger.info("MCP sync window: %s -> %s (user_id=%d)", start_date, end_date, user_id)

    if not force_refresh:
        last_sync = db.get_sync_status(user_id, "garmin")
        if last_sync:
            try:
                last_date = (last_sync.date() if isinstance(last_sync, datetime)
                             else datetime.strptime(last_sync, "%Y-%m-%d").date())
                start_date = max(start_date, last_date + timedelta(days=1))
                logger.info("Incremental sync, resuming at %s", start_date)
            except (ValueError, AttributeError):
                logger.warning("Bad sync_status %r, forcing full window", last_sync)

    if start_date > end_date:
        logger.info("Already up to date through %s", end_date)
        return True

    synced_days = 0
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            await _sync_one_day(user_id, date_str)
            synced_days += 1
        except Exception as e:
            logger.error("Day sync failed for %s: %s", date_str, e)
            # Continue with the next day — matches legacy behavior.
        current += timedelta(days=1)

    db.update_sync_status(user_id, "garmin", today.strftime("%Y-%m-%d"))
    logger.info("Synced %d/%d days via MCP", synced_days, (end_date - start_date).days + 1)
    return True


async def _sync_one_day(user_id: int, date_str: str) -> None:
    """Fetch one day's metrics from the MCP fetcher and persist them.

    Four endpoints per day; the fetcher's cache deduplicates the underlying
    Garmin API calls that ``get_sleep_data`` and ``get_heart_rate_data``
    share.
    """
    fetcher = _fetcher
    assert fetcher is not None  # _ensure_fetcher ran in the caller

    # --- daily summary: steps, distance, active_calories ---
    daily = await fetcher.get_daily_summary(date_str)
    steps = 0
    distance_km = 0.0
    active_calories = 0
    if _is_success(daily):
        summary = daily.get("summary", {})
        steps = summary.get("steps", 0) or 0
        distance_km = summary.get("distance_km", 0) or 0.0
        active_calories = summary.get("active_calories", 0) or 0

    # --- sleep: duration + score ---
    sleep_payload = await fetcher.get_sleep_data(date_str)
    sleep_duration_hours = 0.0
    sleep_score = 0
    if _is_success(sleep_payload):
        sleep_block = sleep_payload.get("sleep_data", {})
        sleep_duration_hours = sleep_block.get("sleep_duration_hours", 0) or 0.0
        sleep_score = sleep_block.get("sleep_score") or 0

    # --- resting HR (extracted from sleep payload in the MCP fetcher) ---
    hr_payload = await fetcher.get_heart_rate_data(date_str)
    avg_rhr = 0
    if _is_success(hr_payload):
        hr_block = hr_payload.get("heart_rate_data", {})
        avg_rhr = hr_block.get("resting_hr") or 0

    # --- stress: aggregates + full-fidelity detail array ---
    stress_payload = await fetcher.get_stress_data(
        date_str, include_details=True, max_detail_points=2000,
    )
    avg_stress = max_stress = min_stress = 0
    stress_details = []
    if _is_success(stress_payload):
        block = stress_payload.get("stress_data", {})
        avg_stress = block.get("avg_stress", 0) or 0
        max_stress = block.get("max_stress", 0) or 0
        min_stress = block.get("min_stress", 0) or 0
        stress_details = stress_payload.get("stress_details", [])

    db.upsert_garmin_daily_summary(
        user_id, date_str, steps, avg_rhr,
        avg_stress=avg_stress, max_stress=max_stress, min_stress=min_stress,
        active_calories=active_calories, distance_km=distance_km,
    )
    db.upsert_garmin_sleep(user_id, date_str, sleep_duration_hours, sleep_score)

    stored_stress_points = 0
    for point in stress_details:
        try:
            ts = datetime.fromisoformat(point["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            db.upsert_garmin_stress_detail(user_id, ts, point["stress_level"])
            stored_stress_points += 1
        except (KeyError, ValueError) as e:
            logger.debug("Skipping stress point %r: %s", point, e)

    activities_persisted, samples_persisted = await _persist_activities_for_date(
        user_id, date_str,
    )

    logger.info(
        "Persisted %s: steps=%d rhr=%d sleep=%.2fh score=%d "
        "stress(avg/max/min)=%d/%d/%d detail_points=%d activities=%d sample_rows=%d",
        date_str, steps, avg_rhr, sleep_duration_hours, sleep_score,
        avg_stress, max_stress, min_stress, stored_stress_points,
        activities_persisted, samples_persisted,
    )


# How long to pause between sample fetches in a backfill run. Garmin doesn't
# publish an official rate limit for the website endpoints, so we err on the
# conservative side. The MCP's internal RateLimiter handles per-minute caps;
# this extra sleep spreads big backfills over a longer wall-clock window so
# we look less like a script and more like a slow user.
_BACKFILL_SAMPLE_SLEEP_SECONDS = 4.0


_ACTIVITY_COLUMNS_TO_FIELDS = {
    # garmin_activities column -> sample-summary dict key from MCP get_activities
    "activity_id": "activity_id",
    "activity_type": "activity_type",
    "start_time": "start_time",
    "duration_minutes": "duration_minutes",
    "distance_km": "distance_km",
    "calories_burned": "calories",
    "avg_heart_rate": "avg_hr",
    "max_heart_rate": "max_hr",
}


def _activity_summary_to_row(summary: dict) -> dict:
    """Translate a single ActivitySummary dict into a garmin_activities row."""
    row = {}
    for col, key in _ACTIVITY_COLUMNS_TO_FIELDS.items():
        val = summary.get(key)
        if val is None:
            continue
        # ActivitySummary serializes start_time as datetime via Pydantic;
        # the DB column is DATETIME and accepts ISO strings.
        if col == "start_time" and isinstance(val, datetime):
            val = val.isoformat()
        row[col] = val
    return row


async def _persist_activities_for_date(user_id: int, date_str: str) -> tuple[int, int]:
    """Persist activity summaries + samples for one calendar day.

    Returns (activities_count, sample_rows_written).
    """
    fetcher = _fetcher
    assert fetcher is not None

    payload = await fetcher.get_activities(date_str)
    if not _is_success(payload):
        return 0, 0

    activities = (payload.get("activities_data") or {}).get("activities", [])
    if not activities:
        return 0, 0

    samples_total = 0
    for summary in activities:
        activity_id = summary.get("activity_id")
        if not activity_id:
            continue

        row = _activity_summary_to_row(summary)
        try:
            db.upsert_garmin_activity(user_id, row)
        except Exception as e:
            logger.error("Failed to upsert activity %s: %s", activity_id, e)
            continue

        if db.has_activity_samples(activity_id):
            logger.debug("Activity %s already has samples, skipping fetch", activity_id)
            continue

        samples_total += await _fetch_and_store_samples(user_id, activity_id)

    return len(activities), samples_total


async def _fetch_and_store_samples(user_id: int, activity_id: str) -> int:
    """Pull intra-activity samples and persist them. Returns rows written."""
    fetcher = _fetcher
    assert fetcher is not None

    try:
        result = await fetcher.get_activity_samples(activity_id)
    except Exception as e:
        logger.warning("Sample fetch failed for activity %s: %s", activity_id, e)
        return 0
    if not _is_success(result):
        logger.info("No samples available for activity %s", activity_id)
        return 0

    samples = result.get("samples", [])
    if not samples:
        return 0
    try:
        return db.insert_activity_samples(user_id, str(activity_id), samples)
    except Exception as e:
        logger.error("Failed to store samples for %s: %s", activity_id, e)
        return 0


async def backfill_activities(user_id: int, lookback_days: int = 365,
                              max_activities: Optional[int] = None,
                              sleep_seconds: float = _BACKFILL_SAMPLE_SLEEP_SECONDS,
                              ) -> dict:
    """One-shot historical activity pull. Slow on purpose.

    Iterates activities from the most recent backwards over ``lookback_days``
    and persists their samples. Pauses ``sleep_seconds`` between sample
    fetches so a multi-hundred-activity backfill spreads over many minutes
    rather than minutes. Idempotent: activities that already have samples
    are skipped.

    Designed to run as a separate task (manual trigger, not part of the
    daily sync). Safe to interrupt and resume.
    """
    fetcher = await _ensure_fetcher()
    end = datetime.now().date()
    start = end - timedelta(days=lookback_days)

    # The MCP fetcher's get_activities is per-day, so iterate.
    days = (end - start).days + 1
    activities_seen = 0
    activities_already = 0
    samples_written = 0
    failures = 0

    cur = end
    while cur >= start:
        date_str = cur.strftime("%Y-%m-%d")
        payload = await fetcher.get_activities(date_str)
        if _is_success(payload):
            for summary in (payload.get("activities_data") or {}).get("activities", []):
                activities_seen += 1
                if max_activities and activities_seen > max_activities:
                    cur = start - timedelta(days=1)  # break outer
                    break
                activity_id = summary.get("activity_id")
                if not activity_id:
                    continue
                try:
                    db.upsert_garmin_activity(user_id, _activity_summary_to_row(summary))
                except Exception as e:
                    logger.error("Backfill upsert failed for %s: %s", activity_id, e)
                    failures += 1
                    continue
                if db.has_activity_samples(activity_id):
                    activities_already += 1
                    continue
                rows = await _fetch_and_store_samples(user_id, activity_id)
                samples_written += rows
                if rows == 0:
                    failures += 1
                await asyncio.sleep(sleep_seconds)
        cur -= timedelta(days=1)

    report = {
        "window_days": days,
        "activities_seen": activities_seen,
        "activities_already_had_samples": activities_already,
        "sample_rows_written": samples_written,
        "failures": failures,
    }
    logger.info("Backfill complete: %s", report)
    return report


async def initialize_garmin_client():
    """Compatibility shim. The legacy import surface includes this name; the
    MCP path makes it a no-op because auth is handled inside ``sync_garmin_data``.
    Returns the underlying ``garminconnect.Garmin`` for any callers that still
    want it."""
    fetcher = await _ensure_fetcher()
    return fetcher.authenticator.get_client()
