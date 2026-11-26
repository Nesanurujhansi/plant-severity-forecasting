# Weather Strategy & Forecasting Presets Guide

---

## 1. Why Can't NASA POWER Provide Future Weather Forecasts?

A common misconception among beginner developers is assuming that a weather API like NASA POWER can predict weather 7 days into the future.

**NASA POWER is a historical reanalysis service, NOT a weather forecasting service.**
- **Historical Observations**: NASA POWER aggregates observations from satellite instruments, radar, and ground stations.
- **Data Delay**: Data is compiled after observations occur, resulting in a 2-day to 3-day publication lag.
- **Why No Forecasts?**: Predicting future weather requires supercomputer numerical weather prediction (NWP) models (such as GFS or ECMWF) that run physics simulations multiple times per day. NASA POWER does not run forward physics simulations.

To forecast plant disease progression over the next **+3 days** and **+7 days**, our application requires future temperature and humidity inputs. We solve this using **Weather Scenario Presets** and an optional third-party forecast interface.

---

## 2. Weather Scenario Presets Defined

When a user requests a 3-day or 7-day disease forecast, **crop-prognosis** allows selecting one of five standardized weather scenario presets:

| Scenario Preset | Name | Description | Worked Example (+3 Days) |
| :--- | :--- | :--- | :--- |
| **(a)** | **As Recent** | Repeats the daily temperature and humidity observed over the last 7 days. | Day 1: 18.5°C, 82% RH<br>Day 2: 19.0°C, 85% RH<br>Day 3: 17.8°C, 80% RH |
| **(b)** | **Climatology** | Uses 10-year historical average for the same calendar days from NASA POWER. | Day 1: 17.2°C, 78% RH<br>Day 2: 17.5°C, 79% RH<br>Day 3: 17.1°C, 77% RH |
| **(c)** | **Warm & Humid** | High disease risk scenario: +3.0°C above average, 90% RH constant. | Day 1: 21.0°C, 90% RH<br>Day 2: 21.5°C, 90% RH<br>Day 3: 21.2°C, 90% RH |
| **(d)** | **Cool & Dry** | Low disease risk scenario: -3.0°C below average, 55% RH constant. | Day 1: 14.0°C, 55% RH<br>Day 2: 14.5°C, 55% RH<br>Day 3: 14.2°C, 55% RH |
| **(e)** | **User-Entered** | Custom temperature and humidity values entered directly by the user. | Day 1: 20.0°C, 88% RH<br>Day 2: 22.0°C, 92% RH<br>Day 3: 19.5°C, 85% RH |

---

## 3. Optional Third-Party Forecast Interface Protocol (UNVERIFIED)

For live production deployments, an external weather forecasting API (such as Open-Meteo or NOAA NWS) can be integrated using Python's `typing.Protocol`.

> [!WARNING]
> Status: **UNVERIFIED**. Third-party forecast APIs require verifying terms of service, rate limits, and API keys before enabling in production. This interface is disabled by default.

### Python Protocol Interface Stub

```python
from typing import Protocol
import pandas as pd

class WeatherForecastClient(Protocol):
    """Protocol interface for external numerical weather forecast services."""

    def fetch_forecast(self, latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
        """Fetch future daily forecast parameters.
        
        Must return a pandas DataFrame with columns:
        ['date', 't_mean', 't_max', 't_min', 'rh_mean', 'precip_mm']
        """
        ...

class StubForecastClient:
    """Stub forecast provider returning climatology fallback values."""

    def fetch_forecast(self, latitude: float, longitude: float, days: int = 7) -> pd.DataFrame:
        # Stub implementation using climatology preset
        dates = pd.date_range(start=pd.Timestamp.now(), periods=days, freq="D")
        return pd.DataFrame({
            "date": dates,
            "t_mean": [18.0] * days,
            "t_max": [22.0] * days,
            "t_min": [14.0] * days,
            "rh_mean": [80.0] * days,
            "precip_mm": [0.0] * days,
        })
```
