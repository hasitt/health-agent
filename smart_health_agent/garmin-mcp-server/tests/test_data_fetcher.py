"""Tests for GarminDataFetcher body battery retrieval."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from garmin_mcp.data_fetcher import GarminDataFetcher


def make_fetcher(client):
    client.get_body_battery.__name__ = "get_body_battery"
    authenticator = MagicMock()
    authenticator.get_client.return_value = client
    authenticator.ensure_authenticated = AsyncMock()
    config = MagicMock()
    config.garmin_api_rate_limit = 100
    config.cache_ttl = 300
    return GarminDataFetcher(authenticator, config)


DEVICE_DAY = {
    "date": "2026-07-03",
    "charged": 65,
    "drained": 72,
    "bodyBatteryValuesArray": [
        [1783022400000, 29],
        [1783047060000, 85],
        [1783100880000, 18],
    ],
}


@pytest.mark.asyncio
async def test_body_battery_uses_device_data():
    client = MagicMock()
    client.get_body_battery.return_value = [DEVICE_DAY]
    fetcher = make_fetcher(client)

    result = await fetcher.get_body_battery("2026-07-03")

    assert result["status"] == "success"
    bb = result["body_battery"]
    assert bb["data_source"] == "device"
    assert bb["energy_score"] == 18  # last measured level
    assert bb["start_level"] == 29
    assert bb["max_level"] == 85
    assert bb["charged"] == 65
    assert bb["drained"] == 72
    assert bb["net_change"] == -7
    assert bb["energy_level"] == "low"


@pytest.mark.asyncio
async def test_body_battery_falls_back_to_estimate():
    client = MagicMock()
    client.get_body_battery.return_value = []  # device reports nothing
    fetcher = make_fetcher(client)
    fetcher.get_sleep_data = AsyncMock(return_value={
        "status": "success",
        "sleep_data": {"sleep_duration_hours": 8.0, "sleep_score": 80},
    })
    fetcher.get_heart_rate_data = AsyncMock(return_value={
        "status": "success",
        "heart_rate_data": {"resting_hr": 48},
    })
    fetcher.get_stress_data = AsyncMock(return_value={
        "status": "success",
        "stress_data": {"avg_stress": 20},
    })

    result = await fetcher.get_body_battery("2026-07-03")

    assert result["status"] == "success"
    bb = result["body_battery"]
    assert bb["data_source"] == "estimated"
    assert bb["energy_score"] is not None
    assert bb["energy_level"] == "high"


@pytest.mark.asyncio
async def test_body_battery_device_error_falls_back():
    client = MagicMock()
    client.get_body_battery.side_effect = RuntimeError("endpoint unavailable")
    fetcher = make_fetcher(client)
    fetcher.get_sleep_data = AsyncMock(return_value={
        "status": "success",
        "sleep_data": {"sleep_duration_hours": 5.0, "sleep_score": 40},
    })
    fetcher.get_heart_rate_data = AsyncMock(return_value={"status": "error"})
    fetcher.get_stress_data = AsyncMock(return_value={"status": "error"})

    result = await fetcher.get_body_battery("2026-07-03")

    assert result["status"] == "success"
    assert result["body_battery"]["data_source"] == "estimated"
