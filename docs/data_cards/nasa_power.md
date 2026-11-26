# Data Card: NASA POWER Daily Agroclimatology API

## Identification & Service Endpoint
- **Service Name**: NASA Prediction Of Worldwide Energy Resources (POWER) Daily Point API
- **Community**: `AG` (Agroclimatology)
- **API Endpoint**: `https://power.larc.nasa.gov/api/temporal/daily/point`
- **Documentation**: [NASA POWER API Docs](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
- **Verification Status**: VERIFIED (Date: 2026-09-20)

## Variables Requested
- `T2M`: Temperature at 2 Meters (°C)
- `T2M_MAX`: Maximum Temperature at 2 Meters (°C)
- `T2M_MIN`: Minimum Temperature at 2 Meters (°C)
- `RH2M`: Relative Humidity at 2 Meters (%)
- `PRECTOTCORR`: Corrected Total Precipitation (mm/day)

## Spatial & Temporal Resolution
- **Spatial Resolution**: ~0.5° latitude x 0.625° longitude (~50 km grid cell size).
- **Temporal Coverage**: Historical and near-real-time daily observations (typically 2–3 days lag).
- **Important Note**: NASA POWER provides **historical observations only, NOT future forecasts**.

## Rate Limits & Caching Rules
- **Caching Policy**: All API responses MUST be cached locally in `data/interim/weather_cache/` keyed by rounded grid cell coordinates, requested parameters, and date range.
- **Blocking Protection**: Repeated API calls for identical coordinates are blocked by NASA servers; caching is strictly mandatory.

## Project Role
- Provides historical meteorological inputs for Module 2 progression modeling and weather parity validation against ETH on-site weather stations.

## License & Citation
- **License**: Public Domain (NASA Open Data Policy).
- **Citation**:
  > NASA POWER (2026). Prediction of Worldwide Energy Resources Daily Agroclimatology API. NASA Langley Research Center. https://power.larc.nasa.gov/
