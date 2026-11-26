"""Unit tests for Open-Meteo geocoding and forecast integration."""

from datetime import date, timedelta
import httpx

from app.tools.weather import describe_wmo_code, fetch_destination_weather


def test_weather_success_within_forecast_window():
    today = date(2026, 11, 26)
    start = today + timedelta(days=1)
    end = today + timedelta(days=2)

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding-api.open-meteo.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "Lisbon",
                            "country": "Portugal",
                            "latitude": 38.7167,
                            "longitude": -9.1333,
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": [start.isoformat(), end.isoformat()],
                    "weather_code": [1, 61],
                    "temperature_2m_max": [18.5, 16.0],
                    "temperature_2m_min": [11.0, 10.5],
                    "precipitation_probability_max": [10, 70],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = fetch_destination_weather("Lisbon, Portugal", start, end, today=today, client=client)

    assert report.available is True
    assert report.destination_name == "Lisbon, Portugal"
    assert report.latitude == 38.7167
    assert len(report.daily_forecasts) == 2
    assert report.daily_forecasts[0].condition == "Mainly clear"
    assert report.daily_forecasts[1].condition == "Slight rain"
    assert report.daily_forecasts[1].precipitation_probability == 70


def test_weather_unavailable_when_dates_too_far_ahead():
    today = date(2026, 11, 26)
    start = today + timedelta(days=30)
    end = start + timedelta(days=3)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "name": "Tokyo",
                        "country": "Japan",
                        "latitude": 35.6895,
                        "longitude": 139.6917,
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = fetch_destination_weather("Tokyo", start, end, today=today, client=client)

    assert report.available is False
    assert "not available" in report.message.lower()
    assert report.daily_forecasts == []
    assert report.latitude == 35.6895


def test_weather_unknown_destination():
    today = date(2026, 11, 26)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = fetch_destination_weather(
        "AtlantisUnknownCity",
        today + timedelta(days=1),
        today + timedelta(days=2),
        today=today,
        client=client,
    )

    assert report.available is False
    assert "not available" in report.message.lower()
    assert describe_wmo_code(95) == "Thunderstorm"

