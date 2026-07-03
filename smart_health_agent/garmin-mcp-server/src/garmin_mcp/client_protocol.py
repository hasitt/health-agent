"""
Swap seam for the Garmin data source.

Today the MCP fetcher uses ``garminconnect.Garmin`` (an unofficial,
website-scraping client). Garmin's terms prohibit redistribution without
official API access, so when official API access is granted we need to
swap the data source without rewriting the rest of the stack.

This module enumerates the exact methods ``GarminDataFetcher`` calls on
its client. Any client object that implements these methods satisfies
the fetcher. Today: ``garminconnect.Garmin`` (duck-typed). Tomorrow: a
thin wrapper around the official Garmin SDK that exposes the same names.

Keep this list in sync as the fetcher grows. Each new method on the
fetcher that touches the client should be reflected here, otherwise the
contract drifts silently.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class GarminClientProtocol(Protocol):
    """Minimum surface ``GarminDataFetcher`` needs from a Garmin client."""

    def login(self) -> Any: ...
    @property
    def garth(self) -> Any: ...

    # Daily/aggregate endpoints
    def get_daily_steps(self, start: Any, end: Any) -> Iterable[dict]: ...
    def get_activities_by_date(self, start: str, end: str) -> Iterable[dict]: ...
    def get_user_summary(self) -> dict: ...
    def get_sleep_data(self, date: str) -> Optional[dict]: ...
    def get_all_day_stress(self, date: Any) -> Optional[dict]: ...
    def get_heart_rates(self, date: str) -> Optional[dict]: ...
    def get_body_battery(self, date: str) -> Optional[dict]: ...

    # Activity listing + details (used by the post-workout chat path)
    def get_activities(self, start: int = 0, limit: int = 20) -> Iterable[dict]: ...
    def get_activity(self, activity_id: Any) -> dict: ...
    def get_activity_details(
        self, activity_id: Any, maxchart: int = 2000, maxpoly: int = 4000,
    ) -> dict: ...
    def get_activity_hr_in_timezones(self, activity_id: Any) -> Optional[list]: ...
    def get_activity_splits(self, activity_id: Any) -> Optional[dict]: ...
