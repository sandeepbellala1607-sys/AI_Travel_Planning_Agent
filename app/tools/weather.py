"""Open-Meteo geocoding and daily forecast integration (free, no API key required)."""

from datetime import date, timedelta
import httpx

from app.schemas import DailyWeather, WeatherReport

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 15  # Open-Meteo provides today + 15 days (16-day forecast)

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def describe_wmo_code(code: int) -> str:
    """Translate a WMO weather code into a human-readable condition."""
    return WMO_WEATHER_CODES.get(code, f"Weather code {code}")


def fetch_destination_weather(
    destination: str,
    start_date: date,
    end_date: date,
    today: date | None = None,
    client: httpx.Client | None = None,
) -> WeatherReport:
    """Geocode destination via Open-Meteo and retrieve daily forecast if dates are in range."""
    reference_today = today or date.today()
    max_forecast_date = reference_today + timedelta(days=FORECAST_HORIZON_DAYS)

    owns_client = client is None
    http = client or httpx.Client(timeout=15.0)
    try:
        # Step 1: Geocode the destination
        search_name = destination.split(",")[0].strip() or destination.strip()
        geo_resp = http.get(
            GEOCODING_URL,
            params={"name": search_name, "count": 1, "language": "en", "format": "json"},
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        results = geo_data.get("results") or []
        if not results:
            return WeatherReport(
                available=False,
                destination_name=destination,
                message=f"Weather forecast not available: could not find coordinates for '{destination}'.",
                daily_forecasts=[],
            )

        place = results[0]
        lat = float(place["latitude"])
        lon = float(place["longitude"])
        resolved_name = ", ".join(
            part for part in [place.get("name"), place.get("country")] if part
        ) or destination

        # Step 2: Check forecast horizon before requesting dates out of range
        if start_date > max_forecast_date:
            return WeatherReport(
                available=False,
                destination_name=resolved_name,
                latitude=lat,
                longitude=lon,
                message=(
                    f"Weather forecast not available for {start_date.isoformat()} to "
                    f"{end_date.isoformat()}: travel dates are beyond Open-Meteo's "
                    f"16-day forecast window (through {max_forecast_date.isoformat()})."
                ),
                daily_forecasts=[],
            )

        effective_start = max(start_date, reference_today)
        effective_end = min(end_date, max_forecast_date)

        forecast_resp = http.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
            },
        )
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
        daily = forecast_data.get("daily") or {}

        dates = daily.get("time") or []
        codes = daily.get("weather_code") or []
        t_max = daily.get("temperature_2m_max") or []
        t_min = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_probability_max") or []

        forecasts: list[DailyWeather] = []
        for idx, day_str in enumerate(dates):
            code = int(codes[idx]) if idx < len(codes) and codes[idx] is not None else 0
            precip_val = (
                int(precip[idx]) if idx < len(precip) and precip[idx] is not None else None
            )
            forecasts.append(
                DailyWeather(
                    date=date.fromisoformat(day_str),
                    weather_code=code,
                    condition=describe_wmo_code(code),
                    temp_min_c=float(t_min[idx]) if idx < len(t_min) else 0.0,
                    temp_max_c=float(t_max[idx]) if idx < len(t_max) else 0.0,
                    precipitation_probability=precip_val,
                )
            )

        if not forecasts:
            return WeatherReport(
                available=False,
                destination_name=resolved_name,
                latitude=lat,
                longitude=lon,
                message="Weather forecast not available from Open-Meteo for the requested dates.",
                daily_forecasts=[],
            )

        note = f"Open-Meteo daily forecast loaded for {resolved_name}."
        if end_date > max_forecast_date:
            note += (
                f" Partial forecast only: dates after {max_forecast_date.isoformat()} "
                "are beyond the 16-day forecast horizon."
            )

        return WeatherReport(
            available=True,
            destination_name=resolved_name,
            latitude=lat,
            longitude=lon,
            message=note,
            daily_forecasts=forecasts,
        )
    except httpx.HTTPError as exc:
        return WeatherReport(
            available=False,
            destination_name=destination,
            message=f"Weather forecast not available due to Open-Meteo API error: {exc}",
            daily_forecasts=[],
        )
    finally:
        if owns_client:
            http.close()

