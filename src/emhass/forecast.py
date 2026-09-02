import asyncio
import bz2
import copy

try:
    import fcntl as _fcntl

    def _flock_acquire(f):
        _fcntl.flock(f, _fcntl.LOCK_EX)

    def _flock_release(f):
        _fcntl.flock(f, _fcntl.LOCK_UN)
except ImportError:  # Windows

    def _flock_acquire(f):
        pass  # type: ignore[misc]

    def _flock_release(f):
        pass  # type: ignore[misc]


import logging
import os
import pickle
import pickle as cPickle
import re
import tempfile
from datetime import datetime, timedelta
from urllib.parse import quote

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
from pvlib.irradiance import disc
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import Array, FixedMount, PVSystem
from pvlib.solarposition import get_solarposition
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from emhass.machine_learning_forecaster import MLForecaster
from emhass.machine_learning_regressor import MLRegressor
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import add_date_features, get_days_list, set_df_index_freq

header_accept = "application/json"
error_msg_list_not_long_enough = "Passed data from passed list is not long enough"
error_msg_method_not_valid = "Passed method is not valid"

# Per-request timeout (seconds) for the Open-Meteo HTTP fetch. Without an
# explicit timeout aiohttp's default is long, so a slow/hanging Open-Meteo
# response could stall the whole EMHASS optimisation cycle.
open_meteo_request_timeout = 12
# Retry policy for the Open-Meteo fetch. Retries are ONLY attempted on a cold
# start (no usable cache exists yet); when a cache is present a single attempt
# is made and any failure falls back to the cache immediately (no added delay).
open_meteo_max_attempts = 3
open_meteo_backoff_seconds = (1, 2, 4)

# Minimum historical days required in a (month, day-of-week, period-of-day)
# bucket before _get_historical_period_spread trusts its quantile ratios -
# mirrors pv_shading_kalman's MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE
# pattern of not trusting a statistic computed from too few samples.
MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD = 5

# Day-to-day load variability isn't uniform across a day - night load
# (standby/fridge, barely changes day to day) is far more stable than
# evening load (cooking, activities, guests). Reconciling P10/P90 at one
# ratio for the whole day (see _reconcile_load_percentile's own docstring
# for why day-level, not per-timestep) would apply the evening's wide
# swings to the night too. These four periods let each get its own
# learned ratio while each period is still wide enough to keep the
# reconciliation's own "don't let every timestep hit its own worst case
# at once" property meaningful within it.
LOAD_QUANTILE_SPREAD_PERIODS = ("night", "morning", "afternoon", "evening")


def get_load_quantile_spread_period(hour: int) -> str:
    """Map an hour-of-day (0-23) to its LOAD_QUANTILE_SPREAD_PERIODS label.

    Boundaries: night 00-06, morning 06-12, afternoon 12-18, evening 18-24.
    """
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _load_quantile_spread_period_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorized get_load_quantile_spread_period, for labeling a whole index at once."""
    return np.select(
        [index.hour < 6, index.hour < 12, index.hour < 18],
        ["night", "morning", "afternoon"],
        default="evening",
    )


# "Pseudo-count" for shrinking a cascade level's own ratio toward the
# next-broader level, weighted by n / (n + this) - empirical-Bayes-style
# shrinkage. A bucket with only MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD (5) days
# gets exactly 50% weight on its own value (the rest pulled from the
# broader level); more days pull it closer to fully trusting its own
# value. Without this, a 5-day bucket could be dominated outright by a
# single unusual day (guests, an outage) - a real, confirmed issue found
# via live data (issue reported 2026-09-02: one 5-day bucket's own P90
# ratio came out over 5x its median from a single outlier day). Set equal
# to MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD so hitting exactly the persist
# threshold lands at 50/50 rather than either extreme.
LOAD_QUANTILE_SHRINKAGE_PSEUDOCOUNT = MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD


def _shrink_ratio_toward(
    bucket: dict | None, base_p10: float, base_p90: float
) -> tuple[float, float]:
    """Blend one cascade level's own (p10_ratio, p90_ratio, n) toward the
    next-broader level's already-blended (base_p10, base_p90) - see
    LOAD_QUANTILE_SHRINKAGE_PSEUDOCOUNT. bucket=None (this level has no
    data for the (date, period) in question yet) is a no-op, returning
    the broader base unchanged.
    """
    if not bucket:
        return base_p10, base_p90
    weight = bucket["n"] / (bucket["n"] + LOAD_QUANTILE_SHRINKAGE_PSEUDOCOUNT)
    return (
        weight * bucket["p10_ratio"] + (1 - weight) * base_p10,
        weight * bucket["p90_ratio"] + (1 - weight) * base_p90,
    )


# Candidate models for the ensemble-derived PV P10 estimate (see
# Forecast._get_pv_p10_weather_from_ensemble) - each confirmed live to
# return real, non-null per-member irradiance for the current/forecast
# window: ECMWF (51 members), GFS (31), ICON (20-40). Not user-
# configurable in v1 - a hardcoded, small, known-good set.
PV_ENSEMBLE_CANDIDATE_MODELS = ("ecmwf_ifs025", "gfs_seamless", "icon_seamless")

# Open-Meteo Ensemble API variable -> _calculate_pvlib_power's own expected
# column names (same values, different target names than
# OPEN_METEO_HISTORICAL_WEATHER_VARS on the Forecast class - that map
# targets the thermal-refit column convention (outdoor_temp/wind_speed),
# this one targets the PV-power column convention (temp_air/wind_speed)
# _get_weather_open_meteo already uses).
_PV_ENSEMBLE_WEATHER_VARS = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
}


def _parse_pv_ensemble_member_arrays(hourly: dict) -> dict[str, np.ndarray] | None:
    """Parse one candidate model's raw Open-Meteo ensemble `hourly` payload
    into {var: array of shape (n_timesteps, n_members)} for every
    _PV_ENSEMBLE_WEATHER_VARS entry (ghi/dni/dhi/temp_air/wind_speed).

    Member columns follow Open-Meteo's own naming convention,
    `<variable>_member01`, `<variable>_member02`, ... (member count varies
    20-64 by model - never hardcoded). Shared by both the pooled P10
    selection (_get_pv_p10_weather_from_ensemble) and the forward-
    accumulating per-model scoring (command_line.py's
    _update_pv_ensemble_model_scores), which needs the same per-model
    member arrays to evaluate that model's own P10/P50/P90 in isolation.

    :param hourly: The `"hourly"` dict from one model's raw ensemble API response.
    :type hourly: dict
    :return: The parsed arrays, or None if any variable has no member
        columns at all (an unusable/malformed response for this purpose).
    :rtype: dict[str, np.ndarray] | None
    """
    model_arrays: dict[str, np.ndarray] = {}
    for om_var, target in _PV_ENSEMBLE_WEATHER_VARS.items():
        member_cols = sorted(
            (k for k in hourly if re.fullmatch(rf"{re.escape(om_var)}_member\d+", k)),
            key=lambda k: int(re.search(r"\d+$", k).group()),
        )
        if not member_cols:
            return None
        model_arrays[target] = np.array([hourly[c] for c in member_cols], dtype=float).T
    return model_arrays


def _select_percentile_member_weather(
    members_by_var: dict[str, np.ndarray],
    member_weights: np.ndarray,
    percentile: float,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Weighted-percentile member selection across a pooled ensemble.

    GHI/DNI/DHI are all driven by the same underlying per-member cloud
    scenario, so taking each variable's percentile independently across
    members could combine e.g. a clearer member's GHI with a cloudier
    member's DNI at the same timestep - physically inconsistent. Ranking
    members and selecting one is therefore done ONCE, by a whole-horizon
    aggregate (mean GHI across the entire forecast window - "how sunny is
    this scenario overall"), not independently at each timestep: every
    real ensemble member is already one internally-consistent simulated
    realization of the atmosphere, including its own temporal cloud-cover
    evolution, and selecting a *different* member at each timestep would
    stitch together unrelated moments from different scenarios - the same
    "assembling a trajectory from independent per-step quantiles" problem
    this feature exists to avoid in the first place, just one level up
    (which scenario, instead of which raw value). The single selected
    member's entire weather vector is returned for every timestep.

    Weighted by member_weights (one weight per pooled member - see
    Forecast._get_pv_p10_weather_from_ensemble for how the pool and
    weights are built): sorts members by their whole-horizon mean GHI,
    then walks the *weight-cumulative* fraction (not the plain
    member-count fraction) to find the member whose cumulative weight
    share first reaches `percentile`, so a model with e.g. 2x another's
    current score effectively counts twice without literally duplicating
    members. Uniform weights (the cold-start default, before any accuracy
    score exists) reduce this to a plain unweighted rank.

    :param members_by_var: variable name -> array of shape
        (n_timesteps, n_members_pooled), one entry per weather variable
        (must include "ghi").
    :type members_by_var: dict[str, np.ndarray]
    :param member_weights: shape (n_members_pooled,) - one weight per
        pooled member, constant across timesteps.
    :type member_weights: np.ndarray
    :param percentile: Target percentile in [0, 100].
    :type percentile: float
    :param index: The DatetimeIndex for the returned DataFrame.
    :type index: pd.DatetimeIndex
    :return: A DataFrame indexed by `index` with one column per key in
        `members_by_var`, all drawn from the single selected member.
    :rtype: pd.DataFrame
    """
    ghi = members_by_var["ghi"]
    mean_ghi_per_member = ghi.mean(axis=0)  # (n_members,) - whole-horizon aggregate
    order = np.argsort(mean_ghi_per_member)
    sorted_weights = member_weights[order]
    cum_weight_frac = np.cumsum(sorted_weights) / sorted_weights.sum()
    rank_pos = np.argmax(cum_weight_frac >= percentile / 100.0)
    member_idx = order[rank_pos]
    return pd.DataFrame(
        {var: arr[:, member_idx] for var, arr in members_by_var.items()}, index=index
    )


def _reindex_ensemble_weather_to(
    ensemble_weather: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align an ensemble-derived weather trajectory onto a live forecast's
    own index.

    ensemble_weather comes from the Open-Meteo Ensemble API's native hourly
    resolution and spans the full local calendar day (including any hours
    already in the past - see _get_pv_p10_weather_from_ensemble's own
    forecast_days handling). target_index (typically df_weather.index) is
    usually a finer sub-hourly step and only ever starts from "now"
    onward, never the past. Without this alignment, directly joining
    ensemble-derived power against a live forecast's power leaves every
    already-past timestep as NaN (present in the ensemble series, absent
    from the live one) - a real, confirmed bug in the pv-forecast-test
    P10/P50/P90 preview.

    Interpolates onto the union of both indexes first (so the original
    hourly points stay real anchors) then reindexes down to target_index -
    the same discipline get_weather_covariates already uses for the same
    kind of coarse-to-fine alignment.
    """
    combined_index = ensemble_weather.index.union(target_index)
    return (
        ensemble_weather.reindex(combined_index)
        .interpolate(method="linear", limit_direction="both")
        .reindex(target_index)
        .ffill()
        .bfill()
    )


class Forecast:
    r"""
    Generate weather, load and costs forecasts needed as inputs to the optimization.

    In EMHASS we have basically 4 forecasts to deal with:

    - PV power production forecast (internally based on the weather forecast and the
      characteristics of your PV plant). This is given in Watts.

    - Load power forecast: how much power your house will demand on the next 24h. This
      is given in Watts.

    - PV production selling price forecast: at what price are you selling your excess
      PV production on the next 24h. This is given in EUR/kWh.

    - Load cost forecast: the price of the energy from the grid on the next 24h. This
      is given in EUR/kWh.

    There are methods that are generalized to the 4 forecast needed. For all there
    forecasts it is possible to pass the data either as a passed list of values or by
    reading from a CSV file. With these methods it is then possible to use data from
    external forecast providers.

    Then there are the methods that are specific to each type of forecast and that
    proposed forecast treated and generated internally by this EMHASS forecast class.
    For the weather forecast a first method (`open-meteo`) uses a open-meteos API
    proposing detailed forecasts based on Lat/Lon locations.
    This method seems stable but as with any scrape method it will fail if any changes
    are made to the webpage API. Another method (`solcast`) is using the SolCast PV
    production forecast service. A final method (`solar.forecast`) is using another
    external service: Solar.Forecast, for which just the nominal PV peak installed
    power should be provided. Search the forecast section on the documentation for examples
    on how to implement these different methods.

    The `get_power_from_weather` method is proposed here to convert from irradiance
    data to electrical power. The PVLib module is used to model the PV plant.

    The specific methods for the load forecast are a first method (`naive`) that uses
    a naive approach, also called persistance. It simply assumes that the forecast for
    a future period will be equal to the observed values in a past period. The past
    period is controlled using parameter `delta_forecast`. A second method (`mlforecaster`)
    uses an internal custom forecasting model using machine learning. There is a section
    in the documentation explaining how to use this method.

    .. note:: This custom machine learning model is introduced from v0.4.0. EMHASS \
        proposed this new `mlforecaster` class with `fit`, `predict` and `tune` methods. \
        Only the `predict` method is used here to generate new forecasts, but it is \
        necessary to previously fit a forecaster model and it is a good idea to \
        optimize the model hyperparameters using the `tune` method. See the dedicated \
        section in the documentation for more help.

    For the PV production selling price and Load cost forecasts the privileged method
    is a direct read from a user provided list of values. The list should be passed
    as a runtime parameter during the `curl` to the EMHASS API.

    I reading from a CSV file, it should contain no header and the timestamped data
    should have the following format:
    2021-04-29 00:00:00+00:00,287.07
    2021-04-29 00:30:00+00:00,274.27
    2021-04-29 01:00:00+00:00,243.38
    ...

    The data columns in these files will correspond to the data in the units expected
    for each forecasting method.

    """

    # Weather covariate columns that ``get_weather_covariates`` can supply to the mlforecaster
    # (via the ``mlforecaster_weather_features`` option). The keys are the Open-Meteo
    # ``minutely_15`` variable names; the values are the friendlier column names exposed to the
    # model. ``heating_degree``/``cooling_degree`` are derived from the temperature, see below.
    OPEN_METEO_COVARIATE_VARS = {
        "temperature_2m": "temp_air",
        "relative_humidity_2m": "relative_humidity",
        "cloud_cover": "cloud_cover",
        "wind_speed_10m": "wind_speed",
        "shortwave_radiation": "ghi",
        "direct_radiation": "direct_radiation",
        "diffuse_radiation": "diffuse_radiation",
        "precipitation": "precipitation",
    }
    # Covariates derived locally from the retrieved temperature (no extra API field needed).
    DERIVED_COVARIATES = ("heating_degree", "cooling_degree")
    # Comfort set-point (deg C) for the heating/cooling-degree transform. A lightweight,
    # forecastable thermal-demand signal that lets the model lift the forecast on cold/hot days.
    WEATHER_COVARIATE_COMFORT_TEMP_C = 18.0
    # Names accepted in ``mlforecaster_weather_features`` (friendly weather names + derived).
    SUPPORTED_WEATHER_COVARIATES = tuple(OPEN_METEO_COVARIATE_VARS.values()) + DERIVED_COVARIATES

    def __init__(
        self,
        retrieve_hass_conf: dict,
        optim_conf: dict,
        plant_conf: dict,
        params: str,
        emhass_conf: dict,
        logger: logging.Logger,
        opt_time_delta: int | None = 24,
        get_data_from_file: bool | None = False,
    ) -> None:
        """
        Define constructor for the forecast class.

        :param retrieve_hass_conf: Dictionary containing the needed configuration
            data from the configuration file, specific to retrieve data from HASS
        :type retrieve_hass_conf: dict
        :param optim_conf: Dictionary containing the needed configuration
            data from the configuration file, specific for the optimization task
        :type optim_conf: dict
        :param plant_conf: Dictionary containing the needed configuration
            data from the configuration file, specific for the modeling of the PV plant
        :type plant_conf: dict
        :param params: Configuration parameters passed from data/options.json
        :type params: str
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging object
        :param opt_time_delta: The time delta in hours used to generate forecasts,
            a value of 24 will generate 24 hours of forecast data, defaults to 24
        :type opt_time_delta: int, optional
        :param get_data_from_file: Select if data should be retrieved from a
            previously saved pickle useful for testing or directly from connection to
            hass database
        :type get_data_from_file: bool, optional

        """
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.freq = self.retrieve_hass_conf["optimization_time_step"]
        self.time_zone = self.retrieve_hass_conf["time_zone"]
        self.method_ts_round = self.retrieve_hass_conf["method_ts_round"]
        self.time_delta = pd.to_timedelta(opt_time_delta, "hours")
        self.var_pv = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        self.var_pv_forecast = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        self.var_load = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        self.var_load_new = self.var_load + "_positive"
        self.lat = self.retrieve_hass_conf["Latitude"]
        self.lon = self.retrieve_hass_conf["Longitude"]
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.get_data_from_file = get_data_from_file
        # Set by get_weather_forecast (open-meteo + open_meteo_pv_ensemble_enabled
        # only) for get_power_from_weather to read - see both methods' own
        # docstrings. None whenever the ensemble path never ran or failed.
        self._pv_p10_weather = None
        # The pooled ensemble members behind _pv_p10_weather (see
        # _get_pv_p10_weather_from_ensemble), kept around so
        # get_pv_ensemble_quantile_forecast can select other percentiles
        # (e.g. P90) from the same pool at zero extra network cost - one
        # fetch this cycle serves both the live bias-blend and the
        # pv-forecast-test P10/P50/P90 preview. None under the same
        # conditions as _pv_p10_weather.
        self._pv_ensemble_pool = None
        self.var_load_cost = "unit_load_cost"
        self.var_prod_price = "unit_prod_price"
        if (params is None) or (params == "null"):
            self.params = {}
        elif type(params) is dict:
            self.params = params
        else:
            self.params = orjson.loads(params)

        if self.method_ts_round == "nearest":
            self.start_forecast = pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0)
        elif self.method_ts_round == "first":
            self.start_forecast = (
                pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0).floor(freq=self.freq)
            )
        elif self.method_ts_round == "last":
            self.start_forecast = (
                pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0).ceil(freq=self.freq)
            )
        else:
            self.logger.error("Wrong method_ts_round passed parameter")
        # check if weather_forecast_cache, if so get 2x the amount of forecast
        _delta_days = self.optim_conf["delta_forecast_daily"].days
        if self.optim_conf["delta_forecast_daily"] != pd.Timedelta(days=_delta_days):
            self.logger.warning(
                "delta_forecast_daily has sub-day components which are ignored; "
                "only the day component (%d) is used for the forecast horizon.",
                _delta_days,
            )
        if self.params["passed_data"].get("weather_forecast_cache", False):
            self.end_forecast = (self.start_forecast + pd.DateOffset(days=_delta_days * 2)).replace(
                microsecond=0
            )
        else:
            self.end_forecast = (self.start_forecast + pd.DateOffset(days=_delta_days)).replace(
                microsecond=0
            )
        self.forecast_dates = (
            pd.date_range(
                start=self.start_forecast,
                end=self.end_forecast - self.freq,
                freq=self.freq,
                tz=self.time_zone,
            )
            .tz_convert("utc")
            .round(self.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.time_zone)
        )
        if (
            params is not None
            and "prediction_horizon" in list(self.params["passed_data"].keys())
            and self.params["passed_data"]["prediction_horizon"] is not None
        ):
            self.forecast_dates = self.forecast_dates[
                0 : self.params["passed_data"]["prediction_horizon"]
            ]
        self.forecast_dates_tz = (
            self.forecast_dates.tz_localize(self.time_zone)
            if self.forecast_dates.tz is None
            else self.forecast_dates.tz_convert(self.time_zone)
        )

    async def get_cached_open_meteo_forecast_json(
        self, max_age: int | None = 30, forecast_days: int = 3
    ) -> dict:
        r"""
        Get weather forecast json from Open-Meteo and cache it for re-use.
        The response json is cached in the local file system and returned
        on subsequent calls until it is older than max_age, at which point
        attempts will be made to replace it with a new version.
        The cached version will not be overwritten until a new version has
        been successfully fetched from Open-Meteo.
        In the event of connectivity issues, the cached version will continue
        to be returned until such time as a new version can be successfully
        fetched from Open-Meteo.
        If you want to force reload, pass max_age value of zero.

        :param max_age: The maximum age of the cached json file, in minutes,
            before it is discarded and a new version fetched from Open-Meteo.
            Defaults to 30 minutes.
        :type max_age: int, optional
        :param forecast_days: The number of days of forecast data required from Open-Meteo.
            One additional day is always fetched from Open-Meteo so there is an extra data in the cache.
            Defaults to 2 days (3 days fetched) to match the prior default.
        :type forecast_days: int, optional
        :return: The json containing the Open-Meteo forecast data
        :rtype: dict

        """

        # Ensure at least 3 weather forecast days (and 1 more than requested)
        if forecast_days is None:
            self.logger.debug("Open-Meteo forecast_days is missing so defaulting to 3 days")
            forecast_days = 3
        elif forecast_days < 3:
            self.logger.debug(
                "Open-Meteo forecast_days is low (%s) so defaulting to 3 days",
                forecast_days,
            )
            forecast_days = 3
        else:
            forecast_days = forecast_days + 1

        # The addition of -b.json file name suffix is because the time format
        # has changed, and it avoids any attempt to use the old format file.
        json_path = os.path.abspath(
            self.emhass_conf["data_path"] / "cached-open-meteo-forecast-b.json"
        )
        # The cached JSON file is always loaded, if it exists, as it is also a fallback
        # in case the REST API call to Open-Meteo fails - the cached JSON will continue to
        # be used until it can successfully fetch a new version from Open-Meteo.
        data = None
        use_cache = False
        if os.path.exists(json_path):
            delta = datetime.now() - datetime.fromtimestamp(os.path.getmtime(json_path))
            json_age = int(delta / timedelta(seconds=60))
            use_cache = json_age < max_age
            self.logger.info("Loading existing cached Open-Meteo JSON file: %s", json_path)
            async with aiofiles.open(json_path) as json_file:
                content = await json_file.read()
                data = orjson.loads(content)
            if use_cache:
                self.logger.info(
                    "The cached Open-Meteo JSON file is recent (age=%.0fm, max_age=%sm)",
                    json_age,
                    max_age,
                )
            else:
                self.logger.info(
                    "The cached Open-Meteo JSON file is old (age=%.0fm, max_age=%sm)",
                    json_age,
                    max_age,
                )

        if not use_cache:
            self.logger.info("Fetching a new weather forecast from Open-Meteo")
            headers = {"User-Agent": "EMHASS", "Accept": header_accept}
            # Open-Meteo has returned non-existent time over DST transitions,
            # so we now return unix timestamps and convert to date/times locally
            # instead.
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                + "latitude="
                + str(round(self.lat, 2))
                + "&longitude="
                + str(round(self.lon, 2))
                + "&minutely_15="
                + "temperature_2m,"
                + "relative_humidity_2m,"
                + "rain,"
                + "cloud_cover,"
                + "wind_speed_10m,"
                + "shortwave_radiation_instant,"
                + "diffuse_radiation_instant,"
                + "direct_normal_irradiance_instant"
                + "&forecast_days="
                + str(forecast_days)
                + "&timezone="
                + quote(str(self.time_zone), safe="")
                + "&timeformat=unixtime"
            )
            # Retry only on a cold start (no usable cache to fall back on). When a
            # cache already exists we keep a single attempt and fall back to it
            # immediately on failure, so the steady-state path adds no delay.
            has_cache = data is not None
            max_attempts = 1 if has_cache else open_meteo_max_attempts
            # A bounded per-request timeout so a slow/hanging Open-Meteo response
            # cannot stall the EMHASS optimisation cycle.
            timeout = aiohttp.ClientTimeout(total=open_meteo_request_timeout)
            for attempt in range(1, max_attempts + 1):
                try:
                    self.logger.debug("Fetching data from Open-Meteo using URL: %s", url)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(url, headers=headers) as response:
                            self.logger.debug("Returned HTTP status code: %s", response.status)
                            response.raise_for_status()
                            """import bz2 # Uncomment to save a serialized data for tests
                            import _pickle as cPickle
                            with bz2.BZ2File("data/test_response_openmeteo_get_method.pbz2", "w") as f:
                                cPickle.dump(response, f)"""
                            data = await response.json()
                            self.logger.info(
                                "Saving response in Open-Meteo JSON cache file: %s",
                                json_path,
                            )
                            async with aiofiles.open(json_path, "w") as json_file:
                                content = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
                                await json_file.write(content)
                    # Successful fetch; stop retrying.
                    break
                except (TimeoutError, aiohttp.ClientError):
                    self.logger.error(
                        "Failed to fetch weather forecast from Open-Meteo (attempt %s/%s)",
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    if attempt < max_attempts:
                        # Cold-start retry path only: back off, then try again.
                        backoff = open_meteo_backoff_seconds[
                            min(attempt - 1, len(open_meteo_backoff_seconds) - 1)
                        ]
                        self.logger.info(
                            "Retrying Open-Meteo fetch in %ss (no usable cache yet)",
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                    elif has_cache:
                        self.logger.warning(
                            "Returning old cached data until next Open-Meteo attempt"
                        )

        return data

    async def _get_weather_open_meteo(
        self, w_forecast_cache_path: str, use_legacy_pvlib: bool
    ) -> pd.DataFrame:
        """Helper to retrieve weather data from Open-Meteo or cache."""
        if not os.path.isfile(w_forecast_cache_path):
            data_raw = await self.get_cached_open_meteo_forecast_json(
                self.optim_conf["open_meteo_cache_max_age"],
                self.optim_conf["delta_forecast_daily"].days,
            )
            data_15min = pd.DataFrame.from_dict(data_raw["minutely_15"])
            # Date/times in the Open-Meteo JSON are unix timestamps
            data_15min["time"] = pd.to_datetime(data_15min["time"], unit="s", utc=True)
            data_15min["time"] = data_15min["time"].dt.tz_convert(self.time_zone)
            data_15min.set_index("time", inplace=True)
            data_15min = data_15min.rename(
                columns={
                    "temperature_2m": "temp_air",
                    "relative_humidity_2m": "relative_humidity",
                    "rain": "precipitable_water",
                    "cloud_cover": "cloud_cover",
                    "wind_speed_10m": "wind_speed",
                    "shortwave_radiation_instant": "ghi",
                    "diffuse_radiation_instant": "dhi",
                    "direct_normal_irradiance_instant": "dni",
                }
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                data_15min.to_csv(
                    self.emhass_conf["data_path"] / "debug-weather-forecast-open-meteo.csv"
                )
            data = data_15min.reindex(self.forecast_dates)
            data.interpolate(
                method="linear",
                axis=0,
                limit=None,
                limit_direction="both",
                inplace=True,
            )
            data = set_df_index_freq(data)
            index_utc = data.index.tz_convert("utc")
            index_tz = index_utc.round(
                freq=data.index.freq, ambiguous="infer", nonexistent="shift_forward"
            ).tz_convert(self.time_zone)
            data.index = index_tz
            data = set_df_index_freq(data)
            # Convert mm to cm and clip minimum to 0.1 cm
            data["precipitable_water"] = (data["precipitable_water"] / 10).clip(lower=0.1)
            if use_legacy_pvlib:
                data = data.drop(columns=["ghi", "dhi", "dni"])
                ghi_est = self.cloud_cover_to_irradiance(data["cloud_cover"])
                data["ghi"] = ghi_est["ghi"]
                data["dni"] = ghi_est["dni"]
                data["dhi"] = ghi_est["dhi"]
            if self.params["passed_data"].get("weather_forecast_cache", False):
                data = await self.set_cached_forecast_data(w_forecast_cache_path, data)
        else:
            data = await self.get_cached_forecast_data(w_forecast_cache_path)
            if data is None:
                # Stale Open-Meteo cache was deleted — fetch fresh data from API.
                self.logger.info(
                    "Stale Open-Meteo cache removed; fetching fresh forecast from API."
                )
                return await self._get_weather_open_meteo(w_forecast_cache_path, use_legacy_pvlib)
        return data

    def _solcast_rate_limit_ok(self, max_calls: int = 8) -> bool:
        """Check and increment a daily Solcast API call counter.

        Uses a file in temporary directory keyed by date. Returns True if under
        the daily limit, False otherwise.
        """
        today = pd.Timestamp.now(tz=self.time_zone).strftime("%Y-%m-%d")
        temp_dir = tempfile.gettempdir()
        counter_path = os.path.join(temp_dir, f"emhass_solcast_calls_{today}.count")

        try:
            # We use a+ mode to read and write without truncating on open.
            with open(counter_path, "a+") as f:
                # Acquire exclusive lock
                _flock_acquire(f)
                f.seek(0)

                content = f.read().strip()
                count = int(content) if content else 0

                if count >= max_calls:
                    _flock_release(f)
                    return False

                # Write incremented count back
                f.seek(0)
                f.truncate()
                f.write(str(count + 1))
                f.flush()
                # Explicit close/unlock occurs via context manager exit, but we unlock explicitly
                _flock_release(f)
            return True
        except (OSError, ValueError) as e:
            self.logger.error(f"Failed to check or increment Solcast rate limit: {e}")
            return False

    async def _get_cached_forecast_or_none(self, w_forecast_cache_path: str) -> pd.DataFrame | None:
        """Return a usable cached forecast, or None when there is no cache file
        or it was discarded as stale/schema-incompatible (issue #932).

        Lets the rate-limited fetchers (solcast, solar.forecast) share one
        cache-recovery path: a non-None result is served directly, None means
        fall through to a fresh API fetch.
        """
        if not os.path.isfile(w_forecast_cache_path):
            return None
        return await self.get_cached_forecast_data(w_forecast_cache_path)

    def _parse_pv_quantile_bias(self) -> float:
        """Return the validated weather_forecast_pv_quantile_bias as a float in [0, 1].

        Coerce-then-validate so a quoted/templated value like "0.5" still works,
        while a bad type (bool, None, list) or NaN falls back to 0.0 with a visible
        warning rather than silently changing or disabling the forecast. Out-of-range
        numerics are clamped to [0, 1]. The default 0.0 keeps the central P50 forecast.
        """
        raw_bias = self.optim_conf.get("weather_forecast_pv_quantile_bias", 0.0)
        try:
            if isinstance(raw_bias, bool):
                raise TypeError
            bias = float(raw_bias)
            if np.isnan(bias):
                raise ValueError
        except (TypeError, ValueError):
            self.logger.warning(
                "weather_forecast_pv_quantile_bias=%r is not a valid number; using 0.0 (P50).",
                raw_bias,
            )
            bias = 0.0
        if bias < 0.0 or bias > 1.0:
            self.logger.warning(
                "weather_forecast_pv_quantile_bias=%s is outside [0, 1]; clamping to that range.",
                bias,
            )
            bias = max(0.0, min(1.0, bias))
        return bias

    async def _get_weather_solcast(self, w_forecast_cache_path: str) -> pd.DataFrame:
        """Helper to retrieve weather data from Solcast or cache."""
        # The explicit `weather-forecast-cache` action sets weather_forecast_cache
        # and is meant to refresh the cache from the Solcast API. Reading the cache
        # first here would return a stale-but-present cache early (for Solcast,
        # get_cached_forecast_data serves reindexed/zero-filled stale data instead
        # of returning None), so the refresh action would never fetch and the cache
        # would freeze permanently once it aged past its coverage window. Bypass the
        # cache read for that action so it always fetches fresh; normal cache_only
        # MPC runs still read the cache. The fetch stays quota-guarded by
        # _solcast_rate_limit_ok.
        force_refresh = self.params["passed_data"].get("weather_forecast_cache", False)
        cached_data = (
            None
            if force_refresh
            else await self._get_cached_forecast_or_none(w_forecast_cache_path)
        )
        if cached_data is not None:
            return cached_data
        # Incompatible/stale cache was discarded (issue #932); fall through
        # to fetch fresh data from the Solcast API (quota-guarded below).
        if self.params["passed_data"].get("weather_forecast_cache_only", False):
            self.logger.warning("Solcast cache file missing or deleted due to being out of date.")
            self.logger.warning(
                "Bypassing 'weather_forecast_cache_only' flag to fetch and cache a fresh forecast."
            )
            # Do NOT return False. We'll let execution continue below to fetch normally.
        if not self._solcast_rate_limit_ok():
            self.logger.warning(
                "Solcast daily API call limit reached (safety cap). "
                "Skipping live API call to preserve quota."
            )
            return False
        if "solcast_api_key" not in self.retrieve_hass_conf:
            self.logger.error("The solcast_api_key parameter was not defined")
            return False
        if "solcast_rooftop_id" not in self.retrieve_hass_conf:
            self.logger.error("The solcast_rooftop_id parameter was not defined")
            return False
        headers = {
            "User-Agent": "EMHASS",
            "Authorization": "Bearer " + self.retrieve_hass_conf["solcast_api_key"],
            "Accept": header_accept,
        }
        days_solcast = int(len(self.forecast_dates) * self.freq.seconds / 3600)
        roof_ids = re.split(r"[,\s]+", self.retrieve_hass_conf["solcast_rooftop_id"].strip())
        total_data = pd.DataFrame()

        # Conservative-bias blend factor, read once before the roof loop.
        # Default 0.0 = pure P50 (no-op). See _parse_pv_quantile_bias for the
        # coerce/validate/clamp policy.
        bias = self._parse_pv_quantile_bias()

        async with aiohttp.ClientSession() as session:
            for roof_id in roof_ids:
                url = f"https://api.solcast.com.au/rooftop_sites/{roof_id}/forecasts?hours={days_solcast}"
                async with session.get(url, headers=headers) as response:
                    if int(response.status) == 200:
                        data = await response.json()
                    elif int(response.status) in [402, 429]:
                        self.logger.error(
                            "Solcast error: May have exceeded your subscription limit."
                        )
                        return False
                    elif int(response.status) >= 400 or (202 <= int(response.status) <= 299):
                        self.logger.error(
                            "Solcast error: Issue with request, check API key and rooftop ID."
                        )
                        return False
                    if len(data["forecasts"]) == 0:
                        self.logger.error("No data retrieved from Solcast service.")
                        return False
                    # Build a timestamped DataFrame from Solcast period_end timestamps
                    solcast_timestamps = [
                        pd.Timestamp(elm["period_end"]) for elm in data["forecasts"]
                    ]
                    # Blend P50 with P10 according to weather_forecast_pv_quantile_bias.
                    # bias=0 (default) => pure P50 (no-op, identical to previous behaviour).
                    # bias=1 => pure P10 (conservative / low estimate).
                    # If pv_estimate10 is absent for an element, fall back to pv_estimate.
                    data_list = []
                    for elm in data["forecasts"]:
                        p50 = elm["pv_estimate"]
                        if bias > 0.0:
                            p10 = elm.get("pv_estimate10")
                            if p10 is not None:
                                est = bias * p10 + (1.0 - bias) * p50
                            else:
                                est = p50
                        else:
                            est = p50
                        data_list.append(est * 1000)
                    data_tmp = pd.DataFrame(
                        {"yhat": data_list},
                        index=pd.DatetimeIndex(solcast_timestamps, name="ts"),
                    )
                    if data_tmp.index.tz is None:
                        data_tmp.index = data_tmp.index.tz_localize("UTC")
                    data_tmp.index = data_tmp.index.tz_convert(self.forecast_dates.tz)
                    # Reindex to target forecast dates and interpolate
                    # (handles Solcast 30-min data -> any optimization_time_step)
                    combined_index = data_tmp.index.union(self.forecast_dates).sort_values()
                    data_tmp = data_tmp.reindex(combined_index)
                    data_tmp.interpolate(method="time", inplace=True)
                    data_tmp = data_tmp.reindex(self.forecast_dates)
                    # Zero-fill edges beyond Solcast data range
                    data_tmp = data_tmp.fillna(0.0)
                    if len(total_data) == 0:
                        total_data = data_tmp.copy()
                    else:
                        total_data = total_data + data_tmp

        data = total_data
        if self.params["passed_data"].get("weather_forecast_cache", False) or self.params[
            "passed_data"
        ].get("weather_forecast_cache_only", False):
            data = await self.set_cached_forecast_data(w_forecast_cache_path, data)
        return data

    async def _get_weather_solar_forecast(self, w_forecast_cache_path: str) -> pd.DataFrame:
        """Helper to retrieve weather data from solar.forecast or cache."""
        cached_data = await self._get_cached_forecast_or_none(w_forecast_cache_path)
        if cached_data is not None:
            return cached_data
        # Incompatible/stale cache was discarded (issue #932); fall through
        # to fetch fresh data from the forecast.solar API.
        # Validation and Default Setup
        if "solar_forecast_kwp" not in self.retrieve_hass_conf:
            self.logger.warning(
                "The solar_forecast_kwp parameter was not defined, using dummy values for testing"
            )
            self.retrieve_hass_conf["solar_forecast_kwp"] = 5
        if self.retrieve_hass_conf["solar_forecast_kwp"] == 0:
            self.logger.warning(
                "The solar_forecast_kwp parameter is set to zero, setting to default 5"
            )
            self.retrieve_hass_conf["solar_forecast_kwp"] = 5
        if self.optim_conf["delta_forecast_daily"].days > 1:
            self.logger.warning(
                "The free public tier for solar.forecast only provides one day forecasts"
            )
        headers = {"Accept": header_accept}
        data = pd.DataFrame()

        async with aiohttp.ClientSession() as session:
            for i in range(len(self.plant_conf["pv_module_model"])):
                url = (
                    "https://api.forecast.solar/estimate/"
                    + str(round(self.lat, 2))
                    + "/"
                    + str(round(self.lon, 2))
                    + "/"
                    + str(self.plant_conf["surface_tilt"][i])
                    + "/"
                    + str(self.plant_conf["surface_azimuth"][i] - 180)
                    + "/"
                    + str(self.retrieve_hass_conf["solar_forecast_kwp"])
                )
                async with session.get(url, headers=headers) as response:
                    data_raw = await response.json()
                    data_dict = {
                        "ts": list(data_raw["result"]["watts"].keys()),
                        "yhat": list(data_raw["result"]["watts"].values()),
                    }
                    data_tmp = pd.DataFrame.from_dict(data_dict)
                    data_tmp.set_index("ts", inplace=True)
                    data_tmp.index = pd.to_datetime(data_tmp.index)
                    data_tmp = data_tmp.tz_localize(
                        self.forecast_dates.tz,
                        ambiguous="infer",
                        nonexistent="shift_forward",
                    )
                    data_tmp = data_tmp.reindex(index=self.forecast_dates)
                    # Gap filling
                    mask_up = data_tmp.copy(deep=True).ffill().isnull()
                    mask_down = data_tmp.copy(deep=True).bfill().isnull()
                    data_tmp.loc[mask_up["yhat"], :] = 0.0
                    data_tmp.loc[mask_down["yhat"], :] = 0.0
                    data_tmp.interpolate(inplace=True, limit=1)
                    data_tmp = data_tmp.fillna(0.0)
                    if len(data) == 0:
                        data = copy.deepcopy(data_tmp)
                    else:
                        data = data + data_tmp

        if self.params["passed_data"].get("weather_forecast_cache", False):
            data = await self.set_cached_forecast_data(w_forecast_cache_path, data)
        return data

    def _get_weather_csv(self, csv_path: str) -> pd.DataFrame:
        """Helper to retrieve weather data from CSV."""
        data = pd.read_csv(csv_path, header=None, names=["ts", "yhat"])
        if len(data) < len(self.forecast_dates):
            self.logger.error("Passed data from CSV is not long enough")
        else:
            data = data.loc[data.index[0 : len(self.forecast_dates)], :]
            data.index = self.forecast_dates
            data.drop("ts", axis=1, inplace=True)
            data = data.copy().loc[self.forecast_dates]
        return data

    def _get_weather_list(self) -> pd.DataFrame:
        """Helper to retrieve weather data from a passed list."""
        data_list = self.params["passed_data"]["pv_power_forecast"]
        forecast_dates = self.forecast_dates_tz
        if data_list is None or (
            len(data_list) < len(forecast_dates)
            and self.params["passed_data"]["prediction_horizon"] is None
        ):
            self.logger.error(error_msg_list_not_long_enough)
            return None
        data_list = data_list[: len(forecast_dates)]
        data_dict = {"ts": forecast_dates, "yhat": data_list}
        data = pd.DataFrame.from_dict(data_dict)
        data.set_index("ts", inplace=True)
        return data

    def _list_method_needs_weather(self) -> bool:
        """Whether a configured deferrable load consumes weather-derived data.

        The ``list`` weather path only carries the passed PV power (``yhat``). A
        thermal load's physics demand additionally needs ``ghi`` (solar gains) and
        ``temp_air`` (the outdoor-temperature fallback), so when one is configured
        we still fetch open-meteo to supply those columns (issue #997). Pure
        PV/battery setups never trigger the extra fetch, keeping the list path a
        no-op for them.
        """
        return any(
            isinstance(cfg, dict)
            and (
                isinstance(cfg.get("thermal_config"), dict)
                or isinstance(cfg.get("thermal_battery"), dict)
            )
            for cfg in self.optim_conf.get("def_load_config", []) or []
        )

    async def _augment_list_with_open_meteo(
        self, data: pd.DataFrame, w_forecast_cache_path: str, use_legacy_pvlib: bool
    ) -> pd.DataFrame:
        """Graft open-meteo weather columns onto a list-method PV frame (issue #997).

        The passed PV power (``yhat``) is preserved untouched; every weather-derived
        column (``ghi``, ``temp_air`` and the rest) is copied over, reindexed onto the
        list frame's index. Fail-soft: if the open-meteo fetch is unavailable the plain
        list frame is returned unchanged, so an offline/air-gapped setup keeps
        working exactly as before (just without solar gains).
        """
        # Only swallow the external failure modes the open-meteo path can raise
        # (network, file/cache IO, malformed or incomplete response data). A
        # programming error should still surface rather than be downgraded to a
        # warning. This keeps the augmentation fail-soft for an offline setup
        # without hiding real bugs.
        try:
            weather = await self._get_weather_open_meteo(w_forecast_cache_path, use_legacy_pvlib)
        except (aiohttp.ClientError, OSError, ValueError, KeyError):
            self.logger.warning(
                "Could not fetch open-meteo weather to accompany the passed "
                "pv_power_forecast; thermal solar gains and the outdoor-temperature "
                "fallback are unavailable for this run (issue #997).",
                exc_info=True,
            )
            return data
        if weather is None:
            self.logger.warning(
                "open-meteo weather fetch returned no data while accompanying the "
                "passed pv_power_forecast; thermal solar gains are unavailable this run."
            )
            return data
        for col in weather.columns:
            if col == "yhat":
                continue
            # Both frames are built on the same forecast horizon; "nearest"
            # only absorbs sub-second index rounding between the list index
            # (forecast_dates_tz) and the open-meteo index.
            data[col] = weather[col].reindex(data.index, method="nearest")
        return data

    async def get_weather_forecast(
        self,
        method: str | None = "open-meteo",
        csv_path: str | None = "data_weather_forecast.csv",
        use_legacy_pvlib: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Get and generate weather forecast data.

        :param method: The desired method, options are 'open-meteo', 'csv', 'list', 'solcast' and \
            'solar.forecast'. Defaults to 'open-meteo'.
        :type method: str, optional
        :return: The DataFrame containing the forecasted data
        :rtype: pd.DataFrame
        """
        csv_path = self.emhass_conf["data_path"] / csv_path
        w_forecast_cache_path = os.path.abspath(
            self.emhass_conf["data_path"] / "weather_forecast_data.pkl"
        )
        self.logger.info("Retrieving weather forecast data using method = " + method)
        if method == "scrapper":
            self.logger.warning(
                "The scrapper method has been deprecated and the keyword is accepted just for backward compatibility, please change the PV forecast method to open-meteo"
            )
        self.weather_forecast_method = method
        # The P50/P10 quantile-bias blend is available from Solcast (which
        # returns pv_estimate10 directly) and, when open_meteo_pv_ensemble_enabled
        # is on, from open-meteo's own ensemble-derived P10 (see
        # _get_pv_p10_weather_from_ensemble below). If the knob is set for any
        # other method/combination, warn and ignore it so the dependency is
        # explicit rather than a silent no-op. (Short-circuits before parsing
        # for solcast, so this never double-logs with the parse inside
        # _get_weather_solcast.)
        pv_ensemble_enabled = method == "open-meteo" and self.optim_conf.get(
            "open_meteo_pv_ensemble_enabled", False
        )
        if method not in ("solcast",) and not pv_ensemble_enabled and self._parse_pv_quantile_bias() > 0.0:
            self.logger.warning(
                "weather_forecast_pv_quantile_bias is set but only applies to the "
                "'solcast' weather_forecast_method, or 'open-meteo' with "
                "open_meteo_pv_ensemble_enabled also on; ignoring it for "
                "weather_forecast_method=%r.",
                method,
            )
        if method in ["open-meteo", "scrapper"]:
            data = await self._get_weather_open_meteo(w_forecast_cache_path, use_legacy_pvlib)
            if pv_ensemble_enabled:
                self._pv_p10_weather = await self._get_pv_p10_weather_from_ensemble(
                    self.optim_conf["delta_forecast_daily"].days
                )
        elif method == "solcast":
            data = await self._get_weather_solcast(w_forecast_cache_path)
        elif method == "solar.forecast":
            data = await self._get_weather_solar_forecast(w_forecast_cache_path)
        elif method == "csv":
            data = self._get_weather_csv(csv_path)
        elif method == "list":
            data = self._get_weather_list()
            # When PV is supplied as a runtime list, weather_forecast_method is
            # forced to "list" and only the PV power survives. If a thermal load
            # needs weather-derived GHI/temperature, still fetch open-meteo for
            # those columns so solar gains are not silently dropped (issue #997).
            if data is not None and self._list_method_needs_weather():
                data = await self._augment_list_with_open_meteo(
                    data, w_forecast_cache_path, use_legacy_pvlib
                )
        else:
            self.logger.error("Method %r is not valid", method)
            data = None
        self.logger.debug("get_weather_forecast returning:\n%s", data)
        return data

    async def _fetch_open_meteo_covariates_json(self, past_days: int, forecast_days: int) -> dict:
        """Fetch (and cache) an Open-Meteo ``minutely_15`` response spanning past + future days.

        This is kept separate from :meth:`get_cached_open_meteo_forecast_json` (which serves the PV
        path and is future-only) so the weather-covariate feature is fully self-contained and the
        existing weather/PV behaviour is untouched. The cache is reused until it is older than the
        configured ``open_meteo_cache_max_age`` and, as with the PV cache, a stale cache is still
        returned as a fallback if a fresh fetch fails.
        """
        cache_path = os.path.abspath(
            self.emhass_conf["data_path"] / "cached-open-meteo-covariates.json"
        )
        max_age = self.optim_conf.get("open_meteo_cache_max_age", 30)
        data = None
        use_cache = False
        if os.path.exists(cache_path):
            delta = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            json_age = int(delta / timedelta(seconds=60))
            use_cache = json_age < max_age
            try:
                async with aiofiles.open(cache_path) as json_file:
                    data = orjson.loads(await json_file.read())
            except (orjson.JSONDecodeError, OSError):
                # A corrupted/truncated cache must not block the fallback fetch: treat it as a miss.
                self.logger.warning("Open-Meteo covariate cache is unreadable; refetching")
                data = None
                use_cache = False
        if not use_cache:
            self.logger.info("Fetching Open-Meteo weather covariates (past_days=%s)", past_days)
            headers = {"User-Agent": "EMHASS", "Accept": header_accept}
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                + "latitude="
                + str(round(self.lat, 2))
                + "&longitude="
                + str(round(self.lon, 2))
                + "&minutely_15="
                + ",".join(self.OPEN_METEO_COVARIATE_VARS.keys())
                + "&past_days="
                + str(int(past_days))
                + "&forecast_days="
                + str(int(forecast_days))
                + "&timezone="
                + quote(str(self.time_zone), safe="")
                + "&timeformat=unixtime"
            )
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                        async with aiofiles.open(cache_path, "w") as json_file:
                            await json_file.write(
                                orjson.dumps(data, option=orjson.OPT_INDENT_2).decode()
                            )
            except aiohttp.ClientError:
                self.logger.error(
                    "Failed to fetch weather covariates from Open-Meteo", exc_info=True
                )
                if data is not None:
                    self.logger.warning("Returning old cached covariate data as a fallback")
        return data

    async def get_weather_covariates(
        self, index: pd.DatetimeIndex, weather_features: list[str]
    ) -> pd.DataFrame:
        """Build the configured weather covariate columns aligned onto an arbitrary time index.

        The covariates are sourced from Open-Meteo over a window that spans the requested ``index``
        (both past, for the training history, and future, for the forecast horizon) and reindexed
        onto ``index``. ``heating_degree``/``cooling_degree`` are derived from the temperature using
        :attr:`WEATHER_COVARIATE_COMFORT_TEMP_C`. The returned frame carries exactly the columns in
        ``weather_features`` (in order) and is indexed by ``index``.

        :param index: The target time index to align the covariates onto (tz-aware).
        :type index: pd.DatetimeIndex
        :param weather_features: The covariate column names to return. Must be a subset of \
            :attr:`SUPPORTED_WEATHER_COVARIATES`.
        :type weather_features: list[str]
        :return: A DataFrame indexed by ``index`` with one column per requested covariate.
        :rtype: pd.DataFrame
        """
        unsupported = [c for c in weather_features if c not in self.SUPPORTED_WEATHER_COVARIATES]
        if unsupported:
            raise ValueError(
                f"Unsupported mlforecaster_weather_features {unsupported}. Supported values are: "
                f"{list(self.SUPPORTED_WEATHER_COVARIATES)}"
            )
        # Size the fetch window from the requested index, clamped to Open-Meteo's 92-day past limit.
        now = pd.Timestamp.now(tz=self.time_zone)
        span_past_days = max(0, (now.normalize() - index.min()).days + 1)
        past_days = int(min(92, span_past_days))
        forecast_days = max(1, (index.max().normalize() - now.normalize()).days + 1)
        data_raw = await self._fetch_open_meteo_covariates_json(past_days, forecast_days)
        if not data_raw or "minutely_15" not in data_raw:
            raise ValueError("Open-Meteo returned no minutely_15 weather covariate data")
        weather = pd.DataFrame.from_dict(data_raw["minutely_15"])
        weather["time"] = pd.to_datetime(weather["time"], unit="s", utc=True).dt.tz_convert(
            self.time_zone
        )
        weather = weather.set_index("time").rename(columns=self.OPEN_METEO_COVARIATE_VARS)
        # Drop any duplicate timestamps (e.g. DST edges) and sort, so the reindex below is safe.
        weather = weather[~weather.index.duplicated(keep="first")].sort_index()
        # Derived thermal-demand covariates (computed even if temp itself was not requested).
        if "temp_air" in weather.columns:
            comfort = self.WEATHER_COVARIATE_COMFORT_TEMP_C
            weather["heating_degree"] = np.maximum(0.0, comfort - weather["temp_air"])
            weather["cooling_degree"] = np.maximum(0.0, weather["temp_air"] - comfort)
        # Align onto the requested index, filling residual gaps the same way the date features are
        # always fully populated, then return only the requested columns in order. The combined
        # index lets the interpolation use the surrounding weather rows when the target instants
        # do not coincide exactly with the 15-min Open-Meteo grid.
        combined_index = weather.index.union(index)
        aligned = weather.reindex(combined_index)
        aligned = aligned.interpolate(method="linear", axis=0, limit_direction="both")
        aligned = aligned.reindex(index).ffill().bfill()
        return aligned[list(weather_features)]

    # Historical Weather API variable -> the internal column names
    # command_line.py's _REFIT_SENSOR_COLUMN_MAP/_HYBRID_HP_SENSOR_COLUMN_MAP/
    # _SELF_LEARNING_PHYSICS_SENSOR_COLUMN_MAP already use - same semantic
    # mapping _get_weather_open_meteo already uses for ghi/dni/dhi
    # (shortwave_radiation/direct_normal_irradiance/diffuse_radiation).
    OPEN_METEO_HISTORICAL_WEATHER_VARS = {
        "temperature_2m": "outdoor_temp",
        "wind_speed_10m": "wind_speed",
        "wind_direction_10m": "wind_bearing",
        "shortwave_radiation": "ghi",
        "direct_normal_irradiance": "dni",
        "diffuse_radiation": "dhi",
    }

    async def get_historical_weather_from_open_meteo(
        self, days_list: pd.date_range, columns: list[str]
    ) -> pd.DataFrame:
        """Fetch historical outdoor-weather columns from Open-Meteo's Historical
        Weather API (archive-api.open-meteo.com) for days_list's span.

        Used as a fallback/override for the thermal-model refits' own
        HA-sensor-sourced outdoor_temp/wind_speed/wind_bearing/ghi/dni/dhi (see
        command_line.py's _fill_missing_weather_from_open_meteo). Unlike
        get_cached_open_meteo_forecast_json/_fetch_open_meteo_covariates_json
        (both future-looking, called on every optimization run), this is only
        ever called from a weekly-ish refit, so deliberately has no cache file -
        the extra machinery would be pure overhead at that call volume, and
        Open-Meteo's free-tier limits (10k/day, 5k/hour) are not a concern at
        this cadence.

        Returned at Open-Meteo's native hourly resolution (deduplicated,
        sorted) rather than resampled here - the caller reindexes onto its own
        target index (see _fill_missing_weather_from_open_meteo's own
        reindex(..., method="nearest", tolerance=...) step).

        :param days_list: The days to fetch (utils.get_days_list's own output -
            a daily pd.date_range). Only the first/last day's calendar date is
            used (Open-Meteo's start_date/end_date are whole-day bounds).
        :type days_list: pd.date_range
        :param columns: The internal column names to fetch - a subset of
            OPEN_METEO_HISTORICAL_WEATHER_VARS's values.
        :type columns: list[str]
        :return: A DataFrame indexed by tz-aware timestamp (self.time_zone)
            with exactly the requested columns, at hourly resolution.
        :rtype: pd.DataFrame
        """
        om_vars = [
            k for k, v in self.OPEN_METEO_HISTORICAL_WEATHER_VARS.items() if v in columns
        ]
        headers = {"User-Agent": "EMHASS", "Accept": header_accept}
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            + "latitude="
            + str(round(self.lat, 2))
            + "&longitude="
            + str(round(self.lon, 2))
            + "&start_date="
            + days_list[0].strftime("%Y-%m-%d")
            + "&end_date="
            + days_list[-1].strftime("%Y-%m-%d")
            + "&hourly="
            + ",".join(om_vars)
            + "&timeformat=unixtime"
        )
        timeout = aiohttp.ClientTimeout(total=open_meteo_request_timeout)
        data = None
        last_exc = None
        for attempt in range(1, open_meteo_max_attempts + 1):
            try:
                self.logger.debug("Fetching historical weather from Open-Meteo: %s", url)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()
                break
            except (TimeoutError, aiohttp.ClientError) as exc:
                last_exc = exc
                self.logger.error(
                    "Failed to fetch historical weather from Open-Meteo (attempt %s/%s)",
                    attempt,
                    open_meteo_max_attempts,
                    exc_info=True,
                )
                if attempt < open_meteo_max_attempts:
                    backoff = open_meteo_backoff_seconds[
                        min(attempt - 1, len(open_meteo_backoff_seconds) - 1)
                    ]
                    await asyncio.sleep(backoff)
        if data is None:
            raise ValueError("Open-Meteo historical weather fetch failed") from last_exc
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            raise ValueError("Open-Meteo returned no hourly historical weather data")
        weather = pd.DataFrame.from_dict(hourly)
        weather["time"] = pd.to_datetime(weather["time"], unit="s", utc=True).dt.tz_convert(
            self.time_zone
        )
        weather = weather.set_index("time").rename(columns=self.OPEN_METEO_HISTORICAL_WEATHER_VARS)
        weather = weather[~weather.index.duplicated(keep="first")].sort_index()
        return weather[[c for c in columns if c in weather.columns]]

    async def _fetch_pv_ensemble_model_json(self, model: str, forecast_days: int) -> dict | None:
        """Fetch one PV_ENSEMBLE_CANDIDATE_MODELS entry's raw ensemble JSON,
        with the same retry/backoff policy as this class's other Open-Meteo
        fetches. Returns None (never raises) on final failure - the caller
        (_get_pv_p10_weather_from_ensemble) just drops this model from the
        pool rather than aborting the whole P10 estimate over one model.
        """
        url = (
            "https://ensemble-api.open-meteo.com/v1/ensemble?"
            + "latitude="
            + str(round(self.lat, 2))
            + "&longitude="
            + str(round(self.lon, 2))
            + "&hourly="
            + ",".join(_PV_ENSEMBLE_WEATHER_VARS.keys())
            + "&models="
            + model
            + "&forecast_days="
            + str(int(forecast_days))
            + "&timezone="
            + quote(str(self.time_zone), safe="")
            + "&timeformat=unixtime"
        )
        headers = {"User-Agent": "EMHASS", "Accept": header_accept}
        timeout = aiohttp.ClientTimeout(total=open_meteo_request_timeout)
        for attempt in range(1, open_meteo_max_attempts + 1):
            try:
                self.logger.debug("Fetching PV ensemble data from Open-Meteo (%s): %s", model, url)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
            except (TimeoutError, aiohttp.ClientError):
                self.logger.error(
                    "Failed to fetch PV ensemble data for model %s (attempt %s/%s)",
                    model,
                    attempt,
                    open_meteo_max_attempts,
                    exc_info=True,
                )
                if attempt < open_meteo_max_attempts:
                    backoff = open_meteo_backoff_seconds[
                        min(attempt - 1, len(open_meteo_backoff_seconds) - 1)
                    ]
                    await asyncio.sleep(backoff)
        return None

    async def _get_pv_p10_weather_from_ensemble(self, forecast_days: int) -> pd.DataFrame | None:
        """Fetch every PV_ENSEMBLE_CANDIDATE_MODELS entry's ensemble members,
        pool them, and return a single weighted-P10 weather trajectory (see
        _select_percentile_member_weather) in the ghi/dni/dhi/temp_air/
        wind_speed shape _calculate_pvlib_power expects.

        Per-model weights come from plant_conf["pv_ensemble_model_weights"]
        (loaded by command_line.py's set_input_data_dict before Forecast is
        constructed, from the forward-accumulating accuracy tracker - see
        _update_pv_ensemble_model_scores); a model missing from that dict
        (never resolved a prediction yet) defaults to weight 1.0, same as
        every other model at cold start - equal weighting throughout is
        numerically a plain unweighted percentile.

        Returns None when every model's fetch failed, or none returned
        usable member data - fails soft, the caller (get_weather_forecast)
        leaves self._pv_p10_weather at its None default and
        get_power_from_weather's blend becomes a no-op.

        :param forecast_days: Forecast horizon length in days. Bumped up the
            same way _get_weather_open_meteo already bumps its own
            forecast_days (at least 3, one more than requested otherwise) -
            the actual forecast window is a rolling ~24h-ahead span from
            "now", not aligned to calendar days, so a run later in the day
            needs real ensemble data reaching into tomorrow too. Without
            this, a low forecast_days (e.g. delta_forecast_daily=1) left
            every hour past local midnight with no real ensemble data,
            silently forward-filled to the last (nighttime, near-zero)
            value by get_power_from_weather's reindex/interpolate - a real,
            confirmed bug (issue reported 2026-09-01: an afternoon
            pv-forecast-test run correctly forecast the rest of today but
            went flat 0W for the whole of tomorrow, sunny hours included).
        :type forecast_days: int
        :return: A DataFrame indexed by tz-aware timestamp with columns
            ghi/dni/dhi/temp_air/wind_speed, or None.
        :rtype: pd.DataFrame | None
        """
        if forecast_days is None or forecast_days < 3:
            forecast_days = 3
        else:
            forecast_days = forecast_days + 1
        model_weights = self.plant_conf.get("pv_ensemble_model_weights") or {}
        pooled_by_var: dict[str, list[np.ndarray]] = {v: [] for v in _PV_ENSEMBLE_WEATHER_VARS.values()}
        pooled_weights: list[np.ndarray] = []
        pooled_model_labels: list[np.ndarray] = []
        index: pd.DatetimeIndex | None = None

        for model in PV_ENSEMBLE_CANDIDATE_MODELS:
            data = await self._fetch_pv_ensemble_model_json(model, forecast_days)
            if data is None:
                continue
            hourly = data.get("hourly")
            if not hourly or "time" not in hourly:
                self.logger.warning("PV ensemble P10: no hourly data for model %s, skipping it", model)
                continue
            model_index = pd.to_datetime(hourly["time"], unit="s", utc=True).tz_convert(self.time_zone)
            if index is None:
                index = model_index
            elif not index.equals(model_index):
                self.logger.warning(
                    "PV ensemble P10: model %s returned a different time index than an "
                    "earlier model, skipping it",
                    model,
                )
                continue

            model_arrays = _parse_pv_ensemble_member_arrays(hourly)
            if model_arrays is None:
                self.logger.warning(
                    "PV ensemble P10: model %s returned no usable member columns, skipping it", model
                )
                continue
            n_members = next(iter(model_arrays.values())).shape[1]

            for var, arr in model_arrays.items():
                pooled_by_var[var].append(arr)
            weight = float(model_weights.get(model, 1.0))
            pooled_weights.append(np.full(n_members, weight))
            pooled_model_labels.append(np.full(n_members, model, dtype=object))

        if index is None or not pooled_by_var["ghi"]:
            self.logger.warning("PV ensemble P10: no usable data from any candidate model")
            return None

        members_by_var = {var: np.concatenate(arrs, axis=1) for var, arrs in pooled_by_var.items() if arrs}
        member_weights = np.concatenate(pooled_weights)
        member_labels = np.concatenate(pooled_model_labels)
        self._pv_ensemble_pool = (members_by_var, member_weights, member_labels, index)
        result = _select_percentile_member_weather(members_by_var, member_weights, 10.0, index)

        # Diagnostics only (mirrors _select_percentile_member_weather's own
        # ranking so it can attribute the selected member back to a model
        # name, without changing that function's shared return contract -
        # it's also called per-model, with a single model's own members,
        # from command_line._update_pv_ensemble_model_scores). Lets a user
        # confirm whether an unexpectedly low P10 forecast reflects a real
        # ensemble member predicting an overcast day, or looks like a
        # data/parsing anomaly worth investigating further.
        mean_ghi_per_member = members_by_var["ghi"].mean(axis=0)
        order = np.argsort(mean_ghi_per_member)
        cum_weight_frac = np.cumsum(member_weights[order]) / member_weights[order].sum()
        rank_pos = np.argmax(cum_weight_frac >= 0.10)
        selected_idx = order[rank_pos]
        self.logger.info(
            "PV ensemble P10: selected member from model=%s (whole-day-mean-GHI pool rank "
            "%d of %d members) - that member's own day mean ghi=%.1f dni=%.1f dhi=%.1f W/m2 "
            "(pool of %d members: mean ghi=%.1f, min=%.1f, max=%.1f W/m2)",
            member_labels[selected_idx],
            int(rank_pos) + 1,
            len(order),
            result["ghi"].mean(),
            result["dni"].mean(),
            result["dhi"].mean(),
            len(order),
            mean_ghi_per_member.mean(),
            mean_ghi_per_member.min(),
            mean_ghi_per_member.max(),
        )
        return result

    def get_pv_ensemble_quantile_forecast(self, df_weather: pd.DataFrame) -> dict[str, pd.Series] | None:
        """PV power forecast at P10/P50/P90, for the pv-forecast-test preview.

        P50 is the plain nominal (unbiased) forecast, computed from
        df_weather - the same weather get_power_from_weather itself starts
        from. P10/P90 are drawn from the ensemble pool _get_pv_p10_weather_
        from_ensemble already fetched this cycle (see that method and
        _pv_ensemble_pool) - reusing it here costs no extra network calls,
        just two more (cheap, local) percentile selections plus PV power
        simulations. All three go through the same horizon-mask handling
        _calculate_pvlib_power's default apply_horizon_mask=True gives
        get_power_from_weather, so the three are directly comparable.

        The ensemble weather is reindexed onto df_weather's own index
        first (_reindex_ensemble_weather_to - the same alignment
        get_power_from_weather's own P10 bias-blend already applies): the
        ensemble's native-hourly data spans the whole local calendar day
        including hours already past, while df_weather only ever starts
        from "now" onward - joining them unaligned left p50 as NaN for
        every already-past hour of today (a real, confirmed bug).

        Each leg is clipped at 0 - the same
        `p_pv_forecast[p_pv_forecast < 0] = 0` get_power_from_weather
        itself applies at night, when a real inverter's own small standby/
        monitoring self-consumption otherwise shows as a tiny negative AC
        value (a real, confirmed bug: this preview leaked that raw
        negative value through unclipped, while every other forecast path
        in this codebase already clips it).

        :param df_weather: The nominal weather forecast, as returned by
            get_weather_forecast (same one get_power_from_weather uses for
            its own P50 leg).
        :type df_weather: pd.DataFrame
        :return: {"p10": ..., "p50": ..., "p90": ...} PV power in Watts, or
            None if no ensemble pool is available this cycle (open_meteo_pv_
            ensemble_enabled is off, get_weather_forecast wasn't called with
            method="open-meteo", or every candidate model's fetch failed).
        :rtype: dict[str, pd.Series] | None
        """
        if self._pv_ensemble_pool is None:
            return None
        members_by_var, member_weights, _, index = self._pv_ensemble_pool
        p10_weather = _select_percentile_member_weather(members_by_var, member_weights, 10.0, index)
        p90_weather = _select_percentile_member_weather(members_by_var, member_weights, 90.0, index)
        p10_weather = _reindex_ensemble_weather_to(p10_weather, df_weather.index)
        p90_weather = _reindex_ensemble_weather_to(p90_weather, df_weather.index)
        return {
            "p10": self._calculate_pvlib_power(p10_weather).clip(lower=0),
            "p50": self._calculate_pvlib_power(df_weather).clip(lower=0),
            "p90": self._calculate_pvlib_power(p90_weather).clip(lower=0),
        }

    def cloud_cover_to_irradiance(
        self, cloud_cover: pd.Series, offset: int | None = 35
    ) -> pd.DataFrame:
        """
        Estimates irradiance from cloud cover in the following steps.

        1. Determine clear sky GHI using Ineichen model and
           climatological turbidity.

        2. Estimate cloudy sky GHI using a function of cloud_cover

        3. Estimate cloudy sky DNI using the DISC model.

        4. Calculate DHI from DNI and GHI.

        (This function was copied and modified from PVLib)

        :param cloud_cover: Cloud cover in %.
        :type cloud_cover: pd.Series
        :param offset: Determines the minimum GHI., defaults to 35
        :type offset: Optional[int], optional
        :return: Estimated GHI, DNI, and DHI.
        :rtype: pd.DataFrame
        """
        location = Location(latitude=self.lat, longitude=self.lon)
        solpos = location.get_solarposition(cloud_cover.index)
        cs = location.get_clearsky(cloud_cover.index, model="ineichen", solar_position=solpos)
        # Using only the linear method
        offset = offset / 100.0
        cloud_cover_unit = copy.deepcopy(cloud_cover) / 100.0
        ghi = (offset + (1 - offset) * (1 - cloud_cover_unit)) * cs["ghi"]
        # Using disc model
        dni = disc(ghi, solpos["zenith"], cloud_cover.index)["dni"]
        dhi = ghi - dni * np.cos(np.radians(solpos["zenith"]))
        irrads = pd.DataFrame({"ghi": ghi, "dni": dni, "dhi": dhi}).fillna(0)
        return irrads

    @staticmethod
    def get_mix_forecast(
        df_now: pd.DataFrame,
        df_forecast: pd.DataFrame,
        alpha: float,
        beta: float,
        col: str,
        ignore_pv_feedback: bool = False,
    ) -> pd.DataFrame:
        """A simple correction method for forecasted data using the current real values of a variable.

        :param df_now: The DataFrame containing the current/real values
        :type df_now: pd.DataFrame
        :param df_forecast: The DataFrame containing the forecast data
        :type df_forecast: pd.DataFrame
        :param alpha: A weight for the forecast data side
        :type alpha: float
        :param beta: A weight for the current/real values sied
        :type beta: float
        :param col: The column variable name
        :type col: str
        :param ignore_pv_feedback: If True, bypass mixing and return original forecast (used during curtailment)
        :type ignore_pv_feedback: bool
        :return: The output DataFrame with the corrected values
        :rtype: pd.DataFrame
        """
        # If ignoring PV feedback (e.g., during curtailment), return original forecast
        if ignore_pv_feedback:
            return df_forecast

        # The mix correction blends the latest real sensor value into the first
        # forecast step. When the forecast was supplied as a runtime list rather
        # than read from the sensor, df_now holds no column for that sensor, so
        # there is no live value to blend. Skip the correction and return the
        # forecast unchanged instead of raising a KeyError (issue #764).
        if df_now is None or col not in df_now.columns or df_now.empty:
            return df_forecast

        first_fcst = alpha * df_forecast.iloc[0] + beta * df_now[col].iloc[-1]
        df_forecast.iloc[0] = int(round(first_fcst))
        return df_forecast

    def _get_model_power(self, params, device_type):
        """
        Helper to extract power rating based on device type and available parameters.
        """
        if device_type == "module":
            if "STC" in params:
                return params["STC"]
            if "I_mp_ref" in params and "V_mp_ref" in params:
                return params["I_mp_ref"] * params["V_mp_ref"]
        elif device_type == "inverter":
            if "Paco" in params:
                return params["Paco"]
            if "Pdco" in params:
                return params["Pdco"]
        return None

    def _find_closest_model(self, target_power, database, device_type):
        """
        Find the model in the database that has a power rating closest to the target_power.
        """
        closest_model = None
        min_diff = float("inf")
        # Handle DataFrame (columns are models) or Dict (keys are models)
        iterator = database.items() if hasattr(database, "items") else database.iteritems()
        for _, params in iterator:
            power = self._get_model_power(params, device_type)
            if power is not None:
                diff = abs(power - target_power)
                if diff < min_diff:
                    min_diff = diff
                    closest_model = params
        if closest_model is not None:
            # Safely get name if it exists (DataFrame Series usually have a .name attribute)
            model_name = getattr(closest_model, "name", "unknown")
            self.logger.info(f"Closest {device_type} model to {target_power}W found: {model_name}")
        else:
            self.logger.warning(f"No suitable {device_type} model found close to {target_power}W")
        return closest_model

    def _get_model(self, model_spec, database, device_type):
        """
        Retrieve a model from the database by name or by power rating.
        """
        # If it's a string, try to find it by name
        if isinstance(model_spec, str):
            if model_spec in database:
                return database[model_spec]
            # If not found by name, check if it is a number string (e.g., "300")
            try:
                target_power = float(model_spec)
                return self._find_closest_model(target_power, database, device_type)
            except ValueError:
                # Not a number, fallback to original behavior (will likely raise KeyError later)
                self.logger.warning(f"{device_type} model '{model_spec}' not found in database.")
                return database[model_spec]
        # If it's a number (int or float), find closest by power
        elif isinstance(model_spec, int | float):
            return self._find_closest_model(model_spec, database, device_type)
        else:
            self.logger.error(f"Invalid type for {device_type} model: {type(model_spec)}")
            return None

    def _apply_pv_horizon_mask(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned horizon/shading profile to weather, in three
        independent layers (see pv_shading_kalman.py / refit_pv_horizon_model
        in command_line.py for how each is fitted):

        1. Diffuse-light (sky-dome) attenuation - DHI is scaled by a
           constant-per-season factor on EVERY row, not just rows below
           the sun's own current horizon: the sky dome (and however much
           of it a real obstruction blocks) is there all the time,
           independent of where the sun currently is. That factor is
           preferably an EMPIRICAL one, measured by regression against
           real production (estimate_empirical_diffuse_transmission_factor,
           see its own docstring) - falling back, per season, to the
           purely theoretical compute_diffuse_transmission_factor when
           there isn't yet enough evidence for that season.
        2. The hard-object ("solid obstruction") horizon - DNI is scaled
           by the learned transmittance for timesteps whose solar position
           falls at/below this season-specific elevation threshold
           (interpolate_horizon_profile) - a real geometric edge (a
           chimney, a roofline), not wherever partial shading merely
           starts (see classify_hard_object_instants).
        3. A separate, additional partial-transmittance filter
           (interpolate_partial_transmittance) - further scales DNI above
           the hard-object horizon where genuinely partial evidence
           exists (a tree canopy letting a varying fraction of light
           through depending on exactly where in its canopy the sun
           sits), gated against the sun's own real yearly elevation
           envelope so it never applies past a physically impossible
           (azimuth, elevation) combination.

        GHI is left untouched throughout - matching the pre-existing
        precedent that DNI masking never kept GHI internally consistent
        with DNI/DHI either, not a new gap introduced here.

        A no-op when plant_conf["pv_horizon_profile"] is missing/empty -
        the default, and the case before a first refit has ever run.
        Layers 1 and 3 additionally no-op (independently) if their own
        persisted data is missing, e.g. a profile persisted before this
        feature existed, or a system with only a hard object and no
        measured partial shading yet.

        The learned horizon is a continuous function of solar azimuth
        (interpolate_horizon_profile), not a lookup into fixed bins - two
        nearby azimuths get two close, but generally different, elevation/
        transmittance values rather than sharing one value up to a
        boundary and jumping at it.
        """
        horizon_profile = self.plant_conf.get("pv_horizon_profile")
        if not horizon_profile or "dni" not in df_weather.columns:
            return df_weather
        from emhass.pv_shading_kalman import (
            compute_diffuse_transmission_factor,
            interpolate_horizon_profile,
            interpolate_partial_transmittance,
            season_labels_for_index,
        )

        df_weather = df_weather.copy()
        angles = Forecast.compute_solar_angles(df_weather, self.lat, self.lon)
        seasons = season_labels_for_index(df_weather.index)

        if "dhi" in df_weather.columns:
            empirical_diffuse_factors = self.plant_conf.get("pv_horizon_diffuse_transmission_factor") or {}
            diffuse_factor_by_season = {
                s: empirical_diffuse_factors.get(s, compute_diffuse_transmission_factor(horizon_profile, s))
                for s in seasons.unique()
            }
            df_weather["dhi"] = df_weather["dhi"] * seasons.map(diffuse_factor_by_season)

        horizon_elevation, transmittance = interpolate_horizon_profile(
            horizon_profile, angles["solar_azimuth"], seasons
        )
        below_horizon = angles["solar_elevation"] <= horizon_elevation
        df_weather.loc[below_horizon, "dni"] = (
            df_weather.loc[below_horizon, "dni"] * transmittance.loc[below_horizon]
        )

        partial_surface = self.plant_conf.get("pv_horizon_partial_transmittance")
        if partial_surface:
            envelope = self.plant_conf.get("pv_horizon_sun_path_envelope") or {}
            sun_min_curve = {float(k): v for k, v in envelope.get("min", {}).items()}
            sun_max_curve = {float(k): v for k, v in envelope.get("max", {}).items()}
            above_horizon = ~below_horizon
            partial_transmittance = interpolate_partial_transmittance(
                partial_surface,
                angles["solar_azimuth"].loc[above_horizon],
                angles["solar_elevation"].loc[above_horizon],
                seasons.loc[above_horizon],
                sun_min_curve,
                sun_max_curve,
            )
            df_weather.loc[above_horizon, "dni"] = (
                df_weather.loc[above_horizon, "dni"] * partial_transmittance
            )
        return df_weather

    def _load_cec_databases(self) -> tuple[dict, dict]:
        """Load the CEC module/inverter databases used by PVLib simulations."""
        cec_modules_path = self.emhass_conf["root_path"] / "data" / "cec_modules.pbz2"
        cec_inverters_path = self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2"
        with bz2.BZ2File(cec_modules_path, "rb") as f:
            cec_modules = cPickle.load(f)
        with bz2.BZ2File(cec_inverters_path, "rb") as f:
            cec_inverters = cPickle.load(f)
        return cec_modules, cec_inverters

    def _run_pvlib_config(
        self,
        df_weather: pd.DataFrame,
        mod_spec,
        inv_spec,
        tilt,
        azimuth,
        mod_per_str,
        str_per_inv,
        cec_modules: dict,
        cec_inverters: dict,
    ) -> pd.Series:
        """Run a single PVLib simulation for one module/inverter/orientation
        configuration (unmasked - callers apply _apply_pv_horizon_mask
        themselves, since the mask to use may be system-wide or per-panel
        depending on the caller)."""
        location = Location(latitude=self.lat, longitude=self.lon)
        temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
        module = self._get_model(mod_spec, cec_modules, "module")
        inverter = self._get_model(inv_spec, cec_inverters, "inverter")
        system = PVSystem(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temp_params,
            modules_per_string=mod_per_str,
            strings_per_inverter=str_per_inv,
        )
        mc = ModelChain(system, location, aoi_model="physical")
        mc.run_model(df_weather)
        return mc.results.ac

    def _calculate_pvlib_power_for_index(
        self,
        df_weather: pd.DataFrame,
        index: int,
        apply_horizon_mask: bool = True,
        cec_databases: tuple[dict, dict] | None = None,
    ) -> pd.Series:
        """Run a single PVLib simulation for just one entry (index) of the
        plant_conf orientation lists - e.g. one physical panel, when
        sensor_power_photovoltaics_per_panel is index-matched one-to-one
        with pv_module_model/surface_tilt/etc (see refit_pv_horizon_model's
        per-panel diagnostics). cec_databases lets a caller looping over
        many indices (e.g. one call per panel) load the CEC files once
        instead of once per call.
        """
        if apply_horizon_mask:
            df_weather = self._apply_pv_horizon_mask(df_weather)
        cec_modules, cec_inverters = cec_databases or self._load_cec_databases()
        return self._run_pvlib_config(
            df_weather,
            self.plant_conf["pv_module_model"][index],
            self.plant_conf["pv_inverter_model"][index],
            self.plant_conf["surface_tilt"][index],
            self.plant_conf["surface_azimuth"][index],
            self.plant_conf["modules_per_string"][index],
            self.plant_conf["strings_per_inverter"][index],
            cec_modules,
            cec_inverters,
        )

    def _run_pvlib_group(
        self,
        df_weather: pd.DataFrame,
        member_indices: list[int],
        cec_modules: dict,
        cec_inverters: dict,
    ) -> pd.Series:
        """Run a single PVLib ModelChain for a group of plant_conf list
        entries that share one physical inverter (see pv_inverter_group) -
        e.g. a central inverter with multiple independently-tracked MPPT
        strings, or DC-optimizer strings landing on separate MPPT inputs.
        Each member becomes its own pvlib Array (independently tracked DC
        side, own module/orientation/string sizing) inside one shared
        PVSystem/inverter, so the inverter's AC clipping ceiling is applied
        exactly once to the combined DC output - not once per member, which
        would overstate the group's real combined capacity.
        """
        location = Location(latitude=self.lat, longitude=self.lon)
        temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["close_mount_glass_glass"]
        arrays = [
            Array(
                mount=FixedMount(
                    surface_tilt=self.plant_conf["surface_tilt"][i],
                    surface_azimuth=self.plant_conf["surface_azimuth"][i],
                ),
                module_parameters=self._get_model(
                    self.plant_conf["pv_module_model"][i], cec_modules, "module"
                ),
                temperature_model_parameters=temp_params,
                modules_per_string=self.plant_conf["modules_per_string"][i],
                strings=self.plant_conf["strings_per_inverter"][i],
            )
            for i in member_indices
        ]
        # All members of a group share one physical inverter - one CEC model
        # can't represent two different real inverters, so a mismatch here
        # is almost certainly a config mistake. Warn and use the first
        # member's inverter rather than crashing.
        inverter_specs = {self.plant_conf["pv_inverter_model"][i] for i in member_indices}
        if len(inverter_specs) > 1:
            self.logger.warning(
                "pv_inverter_group members %s specify different pv_inverter_model values "
                "(%s) - using the first member's inverter for the whole group.",
                member_indices,
                sorted(inverter_specs),
            )
        inverter = self._get_model(
            self.plant_conf["pv_inverter_model"][member_indices[0]], cec_inverters, "inverter"
        )
        system = PVSystem(arrays=arrays, inverter_parameters=inverter)
        mc = ModelChain(system, location, aoi_model="physical")
        mc.run_model(df_weather)
        return mc.results.ac

    def _calculate_pvlib_power(
        self, df_weather: pd.DataFrame, apply_horizon_mask: bool = True
    ) -> pd.Series:
        """
        Helper to simulate PV power generation using PVLib when no direct forecast is available.

        :param apply_horizon_mask: Apply the learned horizon profile (see
            _apply_pv_horizon_mask), if any. refit_pv_horizon_model
            (command_line.py) passes False here - it needs the genuinely
            unobstructed clear-sky simulation to compare against actual
            production, and must never mask against a profile it may
            itself be in the middle of updating.
        :type apply_horizon_mask: bool, optional
        """
        if apply_horizon_mask:
            df_weather = self._apply_pv_horizon_mask(df_weather)
        cec_modules, cec_inverters = self._load_cec_databases()

        # Handle list (mixed orientation) vs single configuration
        if isinstance(self.plant_conf["pv_module_model"], list):
            n = len(self.plant_conf["pv_module_model"])
            # pv_inverter_group: 0 = ungrouped (its own independent
            # inverter, today's default behavior); entries sharing the same
            # non-zero id are combined onto one shared inverter (see
            # _run_pvlib_group). Missing/wrong length -> fully ungrouped.
            raw_groups = self.plant_conf.get("pv_inverter_group")
            if not raw_groups or len(raw_groups) != n:
                if raw_groups:
                    self.logger.warning(
                        "pv_inverter_group has %d entries but the PV plant config has %d - "
                        "ignoring it (every entry treated as its own independent inverter).",
                        len(raw_groups),
                        n,
                    )
                raw_groups = [0] * n
            groups: dict[int, list[int]] = {}
            next_ungrouped_id = -1
            for i, g in enumerate(raw_groups):
                if g == 0:
                    groups[next_ungrouped_id] = [i]
                    next_ungrouped_id -= 1
                else:
                    groups.setdefault(g, []).append(i)

            p_pv_forecast = pd.Series(0, index=df_weather.index)
            for member_indices in groups.values():
                result = self._run_pvlib_group(df_weather, member_indices, cec_modules, cec_inverters)
                p_pv_forecast = p_pv_forecast + result
        else:
            p_pv_forecast = self._run_pvlib_config(
                df_weather,
                self.plant_conf["pv_module_model"],
                self.plant_conf["pv_inverter_model"],
                self.plant_conf["surface_tilt"],
                self.plant_conf["surface_azimuth"],
                self.plant_conf["modules_per_string"],
                self.plant_conf["strings_per_inverter"],
                cec_modules,
                cec_inverters,
            )
        return p_pv_forecast

    def get_power_from_weather(
        self,
        df_weather: pd.DataFrame,
        set_mix_forecast: bool | None = False,
        df_now: pd.DataFrame | None = pd.DataFrame(),
    ) -> pd.Series:
        r"""
        Convert weather forecast data into electrical power.

        :param df_weather: The DataFrame containing the weather forecasted data. \
            This DF should be generated by the 'get_weather_forecast' method or at \
            least contain the same columns names filled with proper data.
        :type df_weather: pd.DataFrame
        :param set_mix_forecast: Use a mixed forecast strategy to integrate now/current values.
        :type set_mix_forecast: Bool, optional
        :param df_now: The DataFrame containing the now/current data.
        :type df_now: pd.DataFrame
        :return: The DataFrame containing the electrical power in Watts
        :rtype: pd.DataFrame
        """
        # If using csv method we consider that yhat is the PV power in W
        if (
            "solar_forecast_kwp" in self.retrieve_hass_conf.keys()
            and self.retrieve_hass_conf["solar_forecast_kwp"] == 0
        ):
            p_pv_forecast = pd.Series(0, index=df_weather.index)
        elif self.weather_forecast_method in [
            "solcast",
            "solar.forecast",
            "csv",
            "list",
        ]:
            p_pv_forecast = df_weather["yhat"]
            p_pv_forecast.name = None
        else:
            # We will transform the weather data into electrical power
            p_pv_forecast = self._calculate_pvlib_power(df_weather)
            bias = self._parse_pv_quantile_bias()
            if self._pv_p10_weather is not None and bias > 0.0:
                p10_weather = _reindex_ensemble_weather_to(self._pv_p10_weather, df_weather.index)
                p10_power = self._calculate_pvlib_power(p10_weather)
                p_pv_forecast = bias * p10_power + (1.0 - bias) * p_pv_forecast
        if set_mix_forecast:
            ignore_pv_feedback = self.params["passed_data"].get(
                "ignore_pv_feedback_during_curtailment", False
            )
            p_pv_forecast = Forecast.get_mix_forecast(
                df_now,
                p_pv_forecast,
                self.params["passed_data"]["alpha"],
                self.params["passed_data"]["beta"],
                self.var_pv,
                ignore_pv_feedback,
            )
        p_pv_forecast[p_pv_forecast < 0] = 0  # replace any negative PV values with zero
        self.logger.debug("get_power_from_weather returning:\n%s", p_pv_forecast)
        return p_pv_forecast

    @staticmethod
    def compute_solar_angles(df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
        """
        Compute solar angles (elevation, azimuth) based on timestamps and location.

        :param df: DataFrame with a DateTime index.
        :param latitude: Latitude of the PV system.
        :param longitude: Longitude of the PV system.
        :return: DataFrame with added solar elevation and azimuth.
        """
        df = df.copy()
        solpos = get_solarposition(df.index, latitude, longitude)
        df["solar_elevation"] = solpos["elevation"]
        df["solar_azimuth"] = solpos["azimuth"]
        return df

    @staticmethod
    def add_cyclic_hour_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the time of day as a continuous sin/cos pair.

        A raw integer hour feature is piecewise constant: with sub-hourly
        optimization time steps a (linear) regression model then produces a
        discontinuity at every hour boundary, which shows up as a sawtooth in
        the adjusted PV forecast. The cyclic encoding is computed from the
        fractional hour (hour + minute/60) so it evolves smoothly within the
        hour and stays continuous across midnight.

        :param df: DataFrame with a DateTime index.
        :type df: pd.DataFrame
        :return: DataFrame with added hour_sin and hour_cos columns.
        :rtype: pd.DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex to compute cyclic hour features.")
        df = df.copy()
        fractional_hour = df.index.hour + df.index.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)
        return df

    def adjust_pv_forecast_data_prep(
        self, data: pd.DataFrame, curtailment_series: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Prepare data for adjusting the photovoltaic (PV) forecast.

        This method aligns the actual PV production data with the forecasted data,
        adds additional features for analysis, and separates the predictors (X)
        from the target variable (y).

        :param data: A DataFrame containing the actual PV production data and the
            forecasted PV production data.
        :type data: pd.DataFrame
        :param curtailment_series: Optional history of the PV curtailment entity.
            Timesteps where curtailment was active (> 0), plus a one-timestep
            margin on either side, are excluded from the training set: during
            curtailment the measured production is deliberately below the
            achievable PV power, so those samples would teach the model a
            downward bias.
        :type curtailment_series: pd.Series, optional
        :return: DataFrame with data for adjusted PV model train.
        """
        # Extract target and predictor
        self.logger.debug("adjust_pv_forecast_data_prep using data:\n%s", data)
        if self.logger.isEnabledFor(logging.DEBUG):
            data.to_csv(
                self.emhass_conf["data_path"] / "debug-adjust-pv-forecast-data-prep-input-data.csv"
            )
        P_PV = data[self.var_pv]  # Actual PV production
        p_pv_forecast = data[self.var_pv_forecast]  # Forecasted PV production
        # Define time ranges
        last_day = data.index.max().normalize()  # Last available day
        three_months_ago = last_day - pd.DateOffset(
            days=self.retrieve_hass_conf["historic_days_to_retrieve"]
        )
        # Train/Test: Last historic_days_to_retrieve days (excluding the last day)
        train_test_mask = (data.index >= three_months_ago) & (data.index < last_day)
        self.p_pv_train_test = P_PV[train_test_mask]
        self.p_pv_forecast_train_test = p_pv_forecast[train_test_mask]
        # Validation: Last day only
        validation_mask = data.index >= last_day
        self.p_pv_validation = P_PV[validation_mask]
        self.p_pv_forecast_validation = p_pv_forecast[validation_mask]
        # Ensure data is aligned
        self.data_adjust_pv = pd.concat(
            [P_PV.rename("actual"), p_pv_forecast.rename("forecast")], axis=1
        ).dropna()

        # Exclude curtailed timesteps (issue #1026): measured production during
        # curtailment is deliberately below the achievable PV power. A one-step
        # margin on either side absorbs execution lag between plan and inverter.
        if curtailment_series is not None:
            curtailed = curtailment_series.reindex(self.data_adjust_pv.index).fillna(0.0) > 0.0
            curtailed = (
                curtailed
                | curtailed.shift(1, fill_value=False)
                | curtailed.shift(-1, fill_value=False)
            )
            n_curtailed = int(curtailed.sum())
            if n_curtailed > 0:
                self.logger.info(
                    f"Excluding {n_curtailed} curtailed timesteps (incl. one-step margin) "
                    f"from {len(self.data_adjust_pv)} PV adjustment training samples."
                )
                self.data_adjust_pv = self.data_adjust_pv[~curtailed]

        # Add more features. The raw integer "hour" date feature is deliberately
        # excluded: it is piecewise constant, so at sub-hourly resolution a linear
        # regression model turns it into a jump at every hour boundary (sawtooth).
        # Time of day is encoded by the cyclic hour features and the solar angles.
        self.data_adjust_pv = add_date_features(
            self.data_adjust_pv,
            date_features=["year", "month", "day_of_week", "day_of_year", "day"],
        )
        self.data_adjust_pv = Forecast.add_cyclic_hour_features(self.data_adjust_pv)

        self.data_adjust_pv = Forecast.compute_solar_angles(self.data_adjust_pv, self.lat, self.lon)
        # Features (X) and target (y)
        self.x_adjust_pv = self.data_adjust_pv.drop(columns=["actual"])  # Predictors
        self.y_adjust_pv = self.data_adjust_pv["actual"]  # Target: actual PV production
        self.logger.debug("adjust_pv_forecast_data_prep output data:\n%s", self.data_adjust_pv)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.data_adjust_pv.to_csv(
                self.emhass_conf["data_path"] / "debug-adjust-pv-forecast-data-prep-output-data.csv"
            )

    def _build_day_level_cv_splits(
        self, index: pd.DatetimeIndex, n_splits: int
    ) -> TimeSeriesSplit | list[tuple[np.ndarray, np.ndarray]]:
        """Day-level blocked time-series CV splits for adjust_pv_forecast_fit.

        A plain TimeSeriesSplit on the raw row sequence can put e.g. 14:00 in
        train and 14:30 in test on the very same day - hours within a day are
        highly autocorrelated (sun position, weather persistence), so that
        leaks information and gives an optimistic CV score. Splitting on
        unique calendar days first, then mapping each day-level fold back to
        its own rows, keeps every row from a given day entirely on one side
        of the split - the standard walk-forward-by-day practice for
        day-ahead solar forecasting.

        :param index: The (possibly sub-daily) DatetimeIndex of the training rows.
        :param n_splits: Requested number of CV folds - reduced automatically
            (with a logged warning) when there aren't enough distinct days to
            support it; falls back to a plain row-level TimeSeriesSplit when
            the data spans fewer than 2 distinct days (day-level blocking is
            meaningless with only one day).
        :return: Either a TimeSeriesSplit (fallback) or a list of
            (train_row_indices, test_row_indices) tuples - both are valid
            `cv` arguments for scikit-learn's GridSearchCV.
        """
        unique_days = pd.DatetimeIndex(sorted(index.normalize().unique()))
        if len(unique_days) < 2:
            self.logger.warning(
                "PV adjustment training data spans only %d distinct day(s) - falling back "
                "to row-level TimeSeriesSplit (day-level blocking needs at least 2 days).",
                len(unique_days),
            )
            return TimeSeriesSplit(n_splits=n_splits)

        n_day_splits = min(n_splits, len(unique_days) - 1)
        if n_day_splits < n_splits:
            self.logger.warning(
                "Only %d distinct days available for PV adjustment training - reducing "
                "day-level CV splits from %d to %d.",
                len(unique_days),
                n_splits,
                n_day_splits,
            )
        day_index = index.normalize()
        cv_splits = []
        for train_days_idx, test_days_idx in TimeSeriesSplit(n_splits=n_day_splits).split(unique_days):
            train_mask = day_index.isin(unique_days[train_days_idx])
            test_mask = day_index.isin(unique_days[test_days_idx])
            cv_splits.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask)))
        return cv_splits

    async def adjust_pv_forecast_fit(
        self,
        n_splits: int = 5,
        regression_model: str = "LassoRegression",
        debug: bool | None = False,
    ) -> pd.DataFrame:
        """
        Fit a regression model to adjust the photovoltaic (PV) forecast.

        This method uses historical actual and forecasted PV production data, along with
        additional solar and date features, to train a regression model. The model is
        optimized using a grid search with time-series cross-validation.

        :param n_splits: The number of splits for time-series cross-validation, defaults to 5.
        :type n_splits: int, optional
        :param regression_model: The type of regression model to use. See REGRESSION_METHODS \
            in machine_learning_regressor.py for the authoritative list of supported models. \
            Currently: 'LinearRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', \
            'KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR', 'RandomForestRegressor', \
            'ExtraTreesRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', \
            'MLPRegressor'. Defaults to "LassoRegression".
        :type regression_model: str, optional
        :param debug: If True, the model is not saved to disk, useful for debugging, defaults to False.
        :type debug: bool, optional
        :return: A DataFrame containing the adjusted PV forecast.
        :rtype: pd.DataFrame
        """
        # Get regression model and hyperparameter grid
        mlr = MLRegressor(
            self.data_adjust_pv,
            "adjusted_pv_forecast",
            regression_model,
            list(self.x_adjust_pv.columns),
            list(self.y_adjust_pv.name),
            None,
            self.logger,
        )
        pipeline, param_grid = mlr._get_model_and_params()
        cv = self._build_day_level_cv_splits(self.x_adjust_pv.index, n_splits)
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="neg_mean_squared_error", verbose=0
        )
        # Train model
        await asyncio.to_thread(grid_search.fit, self.x_adjust_pv, self.y_adjust_pv)
        self.model_adjust_pv = grid_search.best_estimator_
        # Calculate training metrics
        y_pred_train = self.model_adjust_pv.predict(self.x_adjust_pv)
        self.rmse = np.sqrt(mean_squared_error(self.y_adjust_pv, y_pred_train))
        self.r2 = r2_score(self.y_adjust_pv, y_pred_train)
        # Log the metrics
        self.logger.info(f"PV adjust Training metrics: RMSE = {self.rmse}, R2 = {self.r2}")
        # Save model
        if not debug:
            filename = "adjust_pv_regressor.pkl"
            filename_path = self.emhass_conf["data_path"] / filename
            async with aiofiles.open(filename_path, "wb") as outp:
                await outp.write(pickle.dumps(self.model_adjust_pv, pickle.HIGHEST_PROTOCOL))

    def adjust_pv_forecast_predict(self, forecasted_pv: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Predict the adjusted photovoltaic (PV) forecast.

        This method uses the trained regression model to predict the adjusted PV forecast
        based on either the validation data stored in `self` or a new forecasted PV data
        passed as input. It applies additional features such as date and solar angles to
        the forecasted PV production data before making predictions. The solar elevation
        is used to avoid negative values and to fix values at the beginning and end of the day.

        :param forecasted_pv: Optional. A DataFrame containing the forecasted PV production data.
                            It must have a DateTime index and a column named "forecast".
                            If not provided, the method will use `self.p_pv_forecast_validation`.
        :type forecasted_pv: pd.DataFrame, optional
        :return: A DataFrame containing the adjusted PV forecast with additional features.
        :rtype: pd.DataFrame
        """
        # Use the provided forecasted PV data or fall back to the validation data in `self`
        if forecasted_pv is not None:
            # Ensure the input DataFrame has the required structure
            if "forecast" not in forecasted_pv.columns:
                raise ValueError("The input DataFrame must contain a 'forecast' column.")
            forecast_data = forecasted_pv.copy()
        else:
            # Use the validation data stored in `self`
            forecast_data = self.p_pv_forecast_validation.rename("forecast").to_frame()
        # Prepare the forecasted PV data (same feature set as the fit side:
        # calendar features without the raw hour, plus the cyclic hour encoding)
        forecast_data = add_date_features(
            forecast_data,
            date_features=["year", "month", "day_of_week", "day_of_year", "day"],
        )
        forecast_data = Forecast.add_cyclic_hour_features(forecast_data)
        forecast_data = Forecast.compute_solar_angles(forecast_data, self.lat, self.lon)
        # Predict the adjusted forecast
        forecast_data["adjusted_forecast"] = self.model_adjust_pv.predict(forecast_data)

        # Apply solar elevation weighting only for specific cases
        def apply_weighting(row):
            if row["solar_elevation"] <= 0:  # Nighttime or negative solar elevation
                return 0
            elif (
                row["solar_elevation"] < self.optim_conf["adjusted_pv_solar_elevation_threshold"]
            ):  # Early morning or late evening
                return max(
                    row["adjusted_forecast"]
                    * (
                        row["solar_elevation"]
                        / self.optim_conf["adjusted_pv_solar_elevation_threshold"]
                    ),
                    0,
                )
            else:  # Daytime with sufficient solar elevation
                return row["adjusted_forecast"]

        forecast_data["adjusted_forecast"] = forecast_data.apply(apply_weighting, axis=1)
        # Clamp to non-negative: PV power is physically >= 0, but the daytime branch
        # above returns the raw regression output (e.g. Lasso is unconstrained and can
        # extrapolate below zero on cloudy days after sunny training history). See #521.
        forecast_data["adjusted_forecast"] = forecast_data["adjusted_forecast"].clip(lower=0)
        # If using validation data, calculate validation metrics
        if forecasted_pv is None:
            y_true = self.p_pv_validation.values
            y_pred = forecast_data["adjusted_forecast"].values
            self.validation_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            self.validation_r2 = r2_score(y_true, y_pred)
            # Log the validation metrics
            self.logger.info(
                f"PV adjust Validation metrics: RMSE = {self.validation_rmse}, R2 = {self.validation_r2}"
            )
        self.logger.debug("adjust_pv_forecast_predict forecast data:\n%s", forecast_data)
        if self.logger.isEnabledFor(logging.DEBUG):
            forecast_data.to_csv(
                self.emhass_conf["data_path"] / "debug-adjust-pv-forecast-predict-forecast-data.csv"
            )
        # Return the DataFrame with the adjusted forecast
        return forecast_data

    def get_forecast_days_csv(self, timedelta_days: int | None = 1) -> pd.date_range:
        r"""
        Get the date range vector of forecast dates that will be used when loading a CSV file.

        :return: The forecast dates vector
        :rtype: pd.date_range

        """
        start_forecast_csv = pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0)
        if self.method_ts_round == "nearest":
            start_forecast_csv = pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0)
        elif self.method_ts_round == "first":
            start_forecast_csv = (
                pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0).floor(freq=self.freq)
            )
        elif self.method_ts_round == "last":
            start_forecast_csv = (
                pd.Timestamp.now(tz=self.time_zone).replace(microsecond=0).ceil(freq=self.freq)
            )
        else:
            self.logger.error("Wrong method_ts_round passed parameter")
        end_forecast_csv = (
            start_forecast_csv + pd.DateOffset(days=self.optim_conf["delta_forecast_daily"].days)
        ).replace(microsecond=0)
        forecast_dates_csv = (
            pd.date_range(
                start=start_forecast_csv,
                end=end_forecast_csv + timedelta(days=timedelta_days) - self.freq,
                freq=self.freq,
                tz=self.time_zone,
            )
            .tz_convert("utc")
            .round(self.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.time_zone)
        )
        if (
            self.params is not None
            and "prediction_horizon" in list(self.params["passed_data"].keys())
            and self.params["passed_data"]["prediction_horizon"] is not None
        ):
            forecast_dates_csv = forecast_dates_csv[
                0 : self.params["passed_data"]["prediction_horizon"]
            ]
        return forecast_dates_csv

    def _load_forecast_data(
        self,
        csv_path: str,
        data_list: list | None,
        forecast_dates_csv: pd.date_range,
    ) -> pd.DataFrame:
        """
        Helper to load and format forecast data from a CSV file or a list.
        """
        if csv_path is None:
            data_dict = {"ts": forecast_dates_csv, "yhat": data_list}
            df_csv = pd.DataFrame.from_dict(data_dict)
            df_csv.index = forecast_dates_csv
            df_csv = df_csv.drop(["ts"], axis=1)
            df_csv = set_df_index_freq(df_csv)
        else:
            if not os.path.exists(csv_path):
                csv_path = self.emhass_conf["data_path"] / csv_path
            df_csv = pd.read_csv(csv_path, header=None, names=["ts", "yhat"])
            # Check if first column is a valid datetime
            first_col = df_csv.iloc[:, 0]
            if pd.to_datetime(first_col, errors="coerce").notna().all():
                df_csv["ts"] = pd.to_datetime(df_csv["ts"], utc=True)
                df_csv.set_index("ts", inplace=True)
                df_csv.index = df_csv.index.tz_convert(self.time_zone)
            else:
                df_csv.index = forecast_dates_csv
                df_csv = df_csv.drop(["ts"], axis=1)
            df_csv = set_df_index_freq(df_csv)
        return df_csv

    def _extract_daily_forecast(
        self,
        day: int,
        df_timing: pd.DataFrame,
        df_csv: pd.DataFrame,
        csv_path: str,
        list_and_perfect: bool,
    ) -> pd.DataFrame:
        """
        Helper to extract a specific day's forecast data based on timing configuration.
        """
        # Find the start and end indices for the specific day in the timing DataFrame
        day_mask = df_timing.index.day == day
        day_indices = [i for i, x in enumerate(day_mask) if x]
        first_elm_index = day_indices[0]
        last_elm_index = day_indices[-1]
        # Define the target forecast index based on the timing DataFrame
        fcst_index = pd.date_range(
            start=df_timing.index[first_elm_index],
            end=df_timing.index[last_elm_index],
            freq=df_timing.index.freq,
        )
        first_hour = f"{df_timing.index[first_elm_index].hour:02d}:{df_timing.index[first_elm_index].minute:02d}"
        last_hour = f"{df_timing.index[last_elm_index].hour:02d}:{df_timing.index[last_elm_index].minute:02d}"
        # Extract data
        if csv_path is None:
            if list_and_perfect:
                values_array = df_csv.between_time(first_hour, last_hour).values
                # Adjust index length if necessary
                fcst_index = fcst_index[0 : len(values_array)]
                return pd.DataFrame(values_array, index=fcst_index)
            else:
                return pd.DataFrame(
                    df_csv.loc[fcst_index, :].between_time(first_hour, last_hour).values,
                    index=fcst_index,
                )
        else:
            # For CSV path, filter by date string first
            df_csv_filtered_date = df_csv.loc[
                df_csv.index.strftime("%Y-%m-%d") == fcst_index[0].date().strftime("%Y-%m-%d")
            ]
            return pd.DataFrame(
                df_csv_filtered_date.between_time(first_hour, last_hour).values,
                index=fcst_index,
            )

    def get_forecast_out_from_csv_or_list(
        self,
        df_final: pd.DataFrame,
        forecast_dates_csv: pd.date_range,
        csv_path: str,
        data_list: list | None = None,
        list_and_perfect: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Get the forecast data as a DataFrame from a CSV file.

        The data contained in the CSV file should be a 24h forecast with the same frequency as
        the main 'optimization_time_step' parameter in the configuration file. The timestamp will not be used and
        a new DateTimeIndex is generated to fit the timestamp index of the input data in 'df_final'.

        :param df_final: The DataFrame containing the input data.
        :type df_final: pd.DataFrame
        :param forecast_dates_csv: The forecast dates vector
        :type forecast_dates_csv: pd.date_range
        :param csv_path: The path to the CSV file
        :type csv_path: str
        :return: The data from the CSV file
        :rtype: pd.DataFrame

        """
        # Load the source data (df_csv)
        df_csv = self._load_forecast_data(csv_path, data_list, forecast_dates_csv)
        # Configure timing source (df_timing) and iteration list
        if csv_path is None or list_and_perfect:
            df_final = set_df_index_freq(df_final)
            df_timing = copy.deepcopy(df_final)
            days_list = df_final.index.day.unique().tolist()
        else:
            df_timing = copy.deepcopy(df_csv)
            days_list = df_csv.index.day.unique().tolist()
        # Iterate over days and collect forecast parts
        forecast_parts = []
        for day in days_list:
            daily_df = self._extract_daily_forecast(
                day, df_timing, df_csv, csv_path, list_and_perfect
            )
            forecast_parts.append(daily_df)
        if forecast_parts:
            forecast_out = pd.concat(forecast_parts, axis=0)
        else:
            forecast_out = pd.DataFrame()
        if not forecast_out.empty and forecast_out.index.dtype != df_final.index.dtype:
            forecast_out.index = forecast_out.index.astype(df_final.index.dtype)
        # Merge with final DataFrame to align indices
        merged = pd.merge_asof(
            df_final.sort_index(),
            forecast_out.sort_index(),
            left_index=True,
            right_index=True,
            direction="nearest",
        )
        # Keep only forecast_out columns
        forecast_out = merged[forecast_out.columns]
        return forecast_out

    @staticmethod
    def resample_data(data, freq, current_freq):
        r"""
        Resample a DataFrame with a custom frequency.

        :param data: Original time series data with a DateTimeIndex.
        :type data: pd.DataFrame
        :param freq: Desired frequency for resampling (e.g., pd.Timedelta("10min")).
        :type freq: pd.Timedelta
        :return: Resampled data at the specified frequency.
        :rtype: pd.DataFrame
        """
        if freq > current_freq:
            # Downsampling
            # Use 'mean' to aggregate or choose other options ('sum', 'max', etc.)
            resampled_data = data.resample(freq).mean()
        elif freq < current_freq:
            # Upsampling
            # Use 'asfreq' to create empty slots, then interpolate
            resampled_data = data.resample(freq).asfreq()
            resampled_data = resampled_data.interpolate(method="time")
        else:
            # No resampling needed
            resampled_data = data.copy()
        return resampled_data

    @staticmethod
    def get_typical_load_forecast(data, forecast_date):
        r"""
        Forecast the load profile for the next day based on historic data.

        :param data: A DataFrame with a DateTimeIndex containing the historic load data.
                    Must include a 'load' column.
        :type data: pd.DataFrame
        :param forecast_date: The date for which the forecast will be generated.
        :type forecast_date: pd.Timestamp
        :return: A Series with the forecasted load profile for the next day and a list of days used
                to calculate the forecast.
        :rtype: tuple (pd.Series, list)
        """
        # Ensure the 'load' column exists
        if "load" not in data.columns:
            raise ValueError("Data must have a 'load' column.")
        # Filter historic data for the same month and day of the week
        month = forecast_date.month
        day_of_week = forecast_date.dayofweek
        historic_data = data[(data.index.month == month) & (data.index.dayofweek == day_of_week)]
        used_days = np.unique(historic_data.index.date)
        # Align all historic data to the forecast day
        aligned_data = []
        for day in used_days:
            daily_data = data[data.index.date == pd.Timestamp(day).date()]
            aligned_daily_data = daily_data.copy()
            aligned_daily_data.index = aligned_daily_data.index.map(
                lambda x: x.replace(
                    year=forecast_date.year,
                    month=forecast_date.month,
                    day=forecast_date.day,
                )
            )
            aligned_data.append(aligned_daily_data)
        # Combine all aligned historic data into a single DataFrame
        combined_data = pd.concat(aligned_data)
        # Compute the mean load for each timestamp
        forecast = combined_data.groupby(combined_data.index).mean()
        return forecast, used_days

    async def _prepare_hass_load_data(
        self, days_min_load_forecast: int, method: str
    ) -> pd.DataFrame | bool:
        """Helper to retrieve and prepare load data from Home Assistant."""
        self.logger.info(f"Retrieving data from hass for load forecast using method = {method}")
        var_list = [self.var_load]
        var_replace_zero = None
        var_interp = [self.var_load]
        # Pass the configured time zone so the retrieved index stays tz-aware: with None,
        # the websocket statistics path runs the index through tz_convert(None), which
        # strips the tz entirely and later crashes the weather covariate horizon build.
        time_zone_load_forecast = self.time_zone
        rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.freq,
            time_zone_load_forecast,
            self.params,
            self.emhass_conf,
            self.logger,
        )
        if self.get_data_from_file:
            filename_path = self.emhass_conf["data_path"] / "test_df_final.pkl"
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
                self.var_load = var_list[0]
                self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = self.var_load
                var_interp = [var_list[0]]
                self.var_list = [var_list[0]]
                rh.var_list = self.var_list
                self.var_load_new = self.var_load + "_positive"
        else:
            days_list = get_days_list(days_min_load_forecast)
            if not await rh.get_data(days_list, var_list):
                return False
        if not rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=var_replace_zero,
            var_interp=var_interp,
        ):
            return False
        # Handle Stale CSV Headers / Default Name Mismatch
        df = rh.df_final.copy()
        # Check if the expected new variable name exists
        if self.var_load_new not in df.columns:
            self.logger.warning(
                f"Target variable {self.var_load_new} not found in prepared data columns: {df.columns}"
            )
            # Check for default name with positive suffix (Most common case)
            default_name_pos = "sensor.power_load_no_var_loads_positive"
            if default_name_pos in df.columns:
                self.logger.warning(
                    f"Found default '{default_name_pos}'. Renaming to '{self.var_load_new}' to fix mismatch."
                )
                df = df.rename(columns={default_name_pos: self.var_load_new})
            # Check for default name without suffix
            elif "sensor.power_load_no_var_loads" in df.columns:
                self.logger.warning(
                    f"Found default 'sensor.power_load_no_var_loads'. Renaming to '{self.var_load_new}'."
                )
                df = df.rename(columns={"sensor.power_load_no_var_loads": self.var_load_new})
            # Fallback: If dataframe has only 1 column, assume it is the load
            elif len(df.columns) == 1:
                found_col = df.columns[0]
                self.logger.warning(
                    f"Data has single column '{found_col}'. Assuming it is the load and renaming."
                )
                df = df.rename(columns={found_col: self.var_load_new})
        return df[[self.var_load_new]]

    async def _load_long_train_data(self) -> pd.DataFrame:
        """Load and prepare long_train_data.pkl - the 1-year historical load
        reference used by both the 'typical' method's own day-of-week/month
        profile and _get_historical_daily_load_spread's quantile bucketing.

        Extracted from _get_load_forecast_typical so both share the exact
        same reference dataset and pre-processing (stale-header rename,
        tz handling, resample to self.freq) rather than two divergent copies.
        """
        model_type = "long_train_data"
        data_path = self.emhass_conf["data_path"] / str(model_type + ".pkl")
        if not data_path.is_file():
            # Not every data_path actually has this file: the Docker image's
            # first-boot init script only ever seeds the default /data, so a
            # custom data_path (e.g. set via the HA add-on's own
            # Configuration page - a real case that crashed with an
            # unhandled FileNotFoundError) never gets it copied in. Fall
            # back to the copy shipped with the package itself, same
            # pattern _load_cec_databases already uses for the CEC module/
            # inverter databases - works regardless of data_path.
            data_path = self.emhass_conf["root_path"] / "data" / str(model_type + ".pkl")
        async with aiofiles.open(data_path, "rb") as fid:
            content = await fid.read()
            data, _, _, _ = pickle.loads(content)
        # Handle Stale Headers in PKL file
        if self.var_load not in data.columns:
            self.logger.warning(f"Variable {self.var_load} not found in {model_type}.pkl")
            # Check for the old default name
            default_load = "sensor.power_load_no_var_loads"
            if default_load in data.columns:
                self.logger.warning(
                    f"Found legacy column '{default_load}' in pickle. Renaming to '{self.var_load}'"
                )
                data = data.rename(columns={default_load: self.var_load})
        # Ensure the data index is timezone-aware
        data.index = (
            data.index.tz_localize(
                self.forecast_dates.tz,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
            if data.index.tz is None
            else data.index.tz_convert(self.forecast_dates.tz)
        )
        data = data[[self.var_load]]
        current_freq = pd.Timedelta("30min")
        if self.freq != current_freq:
            data = Forecast.resample_data(data, self.freq, current_freq)
        return data

    async def _get_load_forecast_typical(self) -> pd.DataFrame:
        """Helper to generate typical load forecast."""
        data = await self._load_long_train_data()
        dates_list = np.unique(self.forecast_dates.date).tolist()
        forecast = pd.DataFrame()
        for date in dates_list:
            forecast_date = pd.Timestamp(date)
            data.columns = ["load"]
            forecast_tmp, used_days = Forecast.get_typical_load_forecast(data, forecast_date)
            self.logger.debug(f"Using {len(used_days)} days of data to generate the forecast.")
            if len(forecast) == 0:
                forecast = forecast_tmp
            else:
                forecast = pd.concat([forecast, forecast_tmp], axis=0)
        forecast_out = forecast.loc[forecast.index.intersection(self.forecast_dates)]
        forecast_out.index = self.forecast_dates
        forecast_out.index.name = "ts"
        max_power = self.plant_conf["maximum_power_from_grid"]
        # Normalize list/array inputs to a Series aligned with forecast index
        if isinstance(max_power, list | tuple | np.ndarray):
            # Validate length to prevent confusing errors later
            if len(max_power) != len(forecast_out):
                raise ValueError(
                    f"The length of 'maximum_power_from_grid' ({len(max_power)}) "
                    f"does not match the forecast horizon length ({len(forecast_out)})."
                )
            # Create Series for explicit temporal alignment
            scaling_factor = pd.Series(max_power, index=forecast_out.index)
        else:
            # Assume scalar (int/float)
            scaling_factor = max_power
        # Apply scaling
        # The '9000' divisor appears to be a normalization constant specific to the 'typical' model data
        forecast_out["load"] = forecast_out["load"] * scaling_factor / 9000
        return forecast_out.rename(columns={"load": "yhat"})

    async def _compute_generic_period_spread(
        self, forecast_date: pd.Timestamp, period: str
    ) -> tuple[float, float]:
        """The generic bundled reference dataset's (month, weekday) ->
        (weekday, any month) -> no-op cascade, period-scoped - the base
        anchor _get_historical_period_spread's own shrinkage cascade
        blends the user's own history toward. Identical computation to
        before load-quantile-spread-refit existed, just restricted to one
        period's own rows first.
        """
        data = await self._load_long_train_data()
        data.columns = ["load"]
        data = data[_load_quantile_spread_period_labels(data.index) == period]
        # groupby(date), not resample("D"): resample would silently insert a
        # fake 0.0-sum row for any calendar day absent from the data (a real
        # sensor-history gap), corrupting the bucket with fabricated
        # zero-consumption days. groupby only ever aggregates days that
        # actually have at least one recorded row.
        daily_totals = data["load"].groupby(data.index.date).sum()
        daily_totals.index = pd.DatetimeIndex(daily_totals.index)
        daily_totals = daily_totals[daily_totals.index.dayofweek == forecast_date.dayofweek]
        month_bucket = daily_totals[daily_totals.index.month == forecast_date.month]
        bucket = (
            month_bucket
            if len(month_bucket) >= MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD
            else daily_totals
        )
        if len(bucket) < MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD:
            self.logger.debug(
                "load-quantile-bias: only %d historical day(s) for %s (%s), skipping spread (no-op).",
                len(bucket),
                forecast_date.date(),
                period,
            )
            return 1.0, 1.0
        median = bucket.median()
        if median == 0:
            self.logger.debug(
                "load-quantile-bias: zero median in historical bucket for %s (%s), skipping spread (no-op).",
                forecast_date.date(),
                period,
            )
            return 1.0, 1.0
        return float(bucket.quantile(0.1) / median), float(bucket.quantile(0.9) / median)

    async def _get_day_type_period_ratio(
        self, forecast_date: pd.Timestamp, period: str, base_p10: float, base_p90: float
    ) -> tuple[float, float]:
        """The day-type axis alone (weekend-or-weekday, then the specific
        day-of-week), independent of season/month - see
        _get_historical_period_spread's own docstring for why this and
        _get_time_of_year_period_ratio are combined as two independent
        axes rather than one combined cascade.

        Both levels pool every month/season together (the user's own
        weekend_period_buckets/weekday_period_buckets are themselves
        already computed that way - see refit_load_quantile_spread_model),
        so this reflects ONLY how much day-of-week matters, uncontaminated
        by which month it happens to be. Each level shrinks toward the
        previous (_shrink_ratio_toward), starting from (base_p10, base_p90)
        as this axis's own anchor.
        """
        weekday = forecast_date.dayofweek
        is_weekend = "1" if weekday >= 5 else "0"
        own = self.plant_conf.get("load_quantile_spread") or {}

        p10, p90 = base_p10, base_p90
        bucket = (own.get("weekend_period_buckets") or {}).get(f"{is_weekend}_{period}")
        p10, p90 = _shrink_ratio_toward(bucket, p10, p90)
        bucket = (own.get("weekday_period_buckets") or {}).get(f"{weekday}_{period}")
        p10, p90 = _shrink_ratio_toward(bucket, p10, p90)
        return p10, p90

    async def _get_time_of_year_period_ratio(
        self, forecast_date: pd.Timestamp, period: str, base_p10: float, base_p90: float
    ) -> tuple[float, float]:
        """The time-of-year axis alone (season, then the exact month),
        independent of day-of-week - see _get_historical_period_spread's
        own docstring for why this and _get_day_type_period_ratio are
        combined as two independent axes rather than one combined cascade.

        Both levels pool every day-of-week together (the user's own
        season_period_buckets/month_period_buckets are themselves already
        computed that way - see refit_load_quantile_spread_model), so this
        reflects ONLY how much time-of-year matters, uncontaminated by
        which day of the week it happens to be. Each level shrinks toward
        the previous (_shrink_ratio_toward), starting from (base_p10,
        base_p90) as this axis's own anchor.
        """
        from emhass.pv_shading_kalman import season_labels_for_index

        season = season_labels_for_index(pd.DatetimeIndex([forecast_date])).iloc[0]
        own = self.plant_conf.get("load_quantile_spread") or {}

        p10, p90 = base_p10, base_p90
        bucket = (own.get("season_period_buckets") or {}).get(f"{season}_{period}")
        p10, p90 = _shrink_ratio_toward(bucket, p10, p90)
        bucket = (own.get("month_period_buckets") or {}).get(f"{forecast_date.month}_{period}")
        p10, p90 = _shrink_ratio_toward(bucket, p10, p90)
        return p10, p90

    async def _get_historical_period_spread(
        self, forecast_date: pd.Timestamp, period: str
    ) -> tuple[float, float]:
        """Historical spread of one (day, period-of-day)'s total load
        around that bucket's own median, expressed as multiplicative
        ratios (quantile / median) - NOT an absolute offset. A ratio is
        scale-invariant, so it stays valid regardless of units - the same
        reason top-down temporal reconciliation in the forecasting
        literature works with proportions rather than absolute offsets.

        Day-of-week and time-of-year are two independent axes of
        variability (weekday vs. weekend is generally a smaller effect
        than season, and there's no reason a household's "Monday effect"
        and "July effect" should be forced through one single most-to-
        least-specific chain, as an earlier version of this function did -
        that gave day-of-week an arbitrary priority over season, and let a
        sparse, unlucky 5-day bucket dominate outright). Instead:

        1. Compute the generic long_train_data.pkl base ratio for this
           (weekday, period) - see _compute_generic_period_spread. This is
           the shared anchor both axes below are measured against.
        2. Compute the day-type axis's own ratio, shrunk from the base
           through (weekend-or-weekday) then (this exact weekday) - see
           _get_day_type_period_ratio.
        3. Compute the time-of-year axis's own ratio, shrunk from the same
           base through (season) then (this exact month) - see
           _get_time_of_year_period_ratio.
        4. Combine the two as independent multiplicative deviations from
           the shared base: final = day_type_ratio * time_of_year_ratio /
           base - each axis contributes its own deviation from the base,
           and the base itself isn't double-counted. If neither axis has
           any of the user's own history yet, both reduce to exactly the
           base ratio and this correctly returns just the base unchanged.

        Each level within each axis is a no-op in its own shrink
        (_shrink_ratio_toward) whenever the user's own history doesn't
        cover it yet (load-quantile-spread-refit never run, or that
        specific bucket doesn't have MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD
        days yet) - identical graceful degradation to before this function
        existed.

        :param forecast_date: The calendar day to compute the spread for.
        :type forecast_date: pd.Timestamp
        :param period: One of LOAD_QUANTILE_SPREAD_PERIODS.
        :type period: str
        :return: (p10_ratio, p90_ratio)
        :rtype: tuple[float, float]
        """
        base_p10, base_p90 = await self._compute_generic_period_spread(forecast_date, period)
        day_p10, day_p90 = await self._get_day_type_period_ratio(
            forecast_date, period, base_p10, base_p90
        )
        time_p10, time_p90 = await self._get_time_of_year_period_ratio(
            forecast_date, period, base_p10, base_p90
        )
        p10 = (day_p10 * time_p10) / base_p10 if base_p10 != 0 else base_p10
        p90 = (day_p90 * time_p90) / base_p90 if base_p90 != 0 else base_p90
        return p10, p90

    def _parse_load_quantile_bias(self) -> float:
        """Return the validated load_forecast_quantile_bias as a float in [0, 1].

        Same coerce-then-validate shape as _parse_pv_quantile_bias. Unlike PV
        (where the conservative tail is P10, less generation), a conservative
        *load* estimate is P90, more consumption - the tail that actually
        threatens grid-import/battery-sizing constraints. The default 0.0
        keeps the central P50 forecast - a full no-op, byte-identical to
        today's behaviour.
        """
        raw_bias = self.optim_conf.get("load_forecast_quantile_bias", 0.0)
        try:
            if isinstance(raw_bias, bool):
                raise TypeError
            bias = float(raw_bias)
            if np.isnan(bias):
                raise ValueError
        except (TypeError, ValueError):
            self.logger.warning(
                "load_forecast_quantile_bias=%r is not a valid number; using 0.0 (P50).",
                raw_bias,
            )
            bias = 0.0
        if bias < 0.0 or bias > 1.0:
            self.logger.warning(
                "load_forecast_quantile_bias=%s is outside [0, 1]; clamping to that range.",
                bias,
            )
            bias = max(0.0, min(1.0, bias))
        return bias

    async def _reconcile_load_percentile(self, p_load_forecast: pd.Series, percentile: float) -> pd.Series:
        """Top-down temporal reconciliation: build a per-timestep P10/P90
        load series whose (day, period-of-day) sums exactly match a
        historically-informed P10/P90 total for that period, instead of
        independently inflating/deflating each timestep (which would
        implicitly assume every timestep hits its own worst/best case at
        once - the "marginal vs. joint quantiles" problem).

        Reconciled per (day, period) rather than per whole day: night load
        (standby/fridge) barely varies day to day, evening load (cooking,
        activities, guests) varies a lot - one ratio for the whole day
        would apply the evening's wide swings to the night too (see
        LOAD_QUANTILE_SPREAD_PERIODS). Within each (day, period): normalize
        that period's own point-forecast values to a shape summing to 1
        (falls back to a uniform 1/n shape if the period's total is 0),
        scale that shape by (period_total * ratio) from
        _get_historical_period_spread (its p10_ratio or p90_ratio, per
        percentile) - so the reconciled period's own sum is exactly
        period_total * ratio, by construction.

        :param p_load_forecast: The point (P50) load forecast series.
        :type p_load_forecast: pd.Series
        :param percentile: 10.0 or 90.0 - which ratio from
            _get_historical_period_spread to apply.
        :type percentile: float
        :return: The per-timestep P10/P90 series, aligned to
            p_load_forecast's index.
        :rtype: pd.Series
        """
        result = p_load_forecast.copy().astype(float)
        period_labels = _load_quantile_spread_period_labels(p_load_forecast.index)
        for date in np.unique(p_load_forecast.index.date):
            day_mask = p_load_forecast.index.date == date
            for period in LOAD_QUANTILE_SPREAD_PERIODS:
                period_mask = day_mask & (period_labels == period)
                if not period_mask.any():
                    continue
                period_slice = p_load_forecast[period_mask]
                period_total = period_slice.sum()
                if period_total == 0:
                    shape = pd.Series(1.0 / len(period_slice), index=period_slice.index)
                else:
                    shape = period_slice / period_total
                p10_ratio, p90_ratio = await self._get_historical_period_spread(
                    pd.Timestamp(date), period
                )
                ratio = p10_ratio if percentile == 10.0 else p90_ratio
                result[period_mask] = shape.values * (period_total * ratio)
        return result

    async def get_load_quantile_forecast(self, p_load_forecast_p50: pd.Series) -> dict[str, pd.Series]:
        """Load power forecast at P10/P50/P90, for the load-forecast-test preview.

        P50 is simply the point forecast passed in, unchanged. P10/P90 are
        _reconcile_load_percentile applied to that same P50 series - so all
        three share its exact index, no alignment step needed (unlike the
        PV ensemble quantile preview, where P10/P90 come from a different-
        resolution data source - see get_pv_ensemble_quantile_forecast).

        :param p_load_forecast_p50: The point (P50) load forecast, e.g.
            from get_load_forecast with load_forecast_quantile_bias at 0.
        :type p_load_forecast_p50: pd.Series
        :return: {"p10": ..., "p50": ..., "p90": ...} load power in Watts.
        :rtype: dict[str, pd.Series]
        """
        p10 = await self._reconcile_load_percentile(p_load_forecast_p50, 10.0)
        p90 = await self._reconcile_load_percentile(p_load_forecast_p50, 90.0)
        return {"p10": p10, "p50": p_load_forecast_p50, "p90": p90}

    def _get_load_forecast_naive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper for naive forecast."""
        forecast_horizon = len(self.forecast_dates)
        historical_values = df.iloc[-forecast_horizon:]
        return pd.DataFrame(historical_values.values, index=self.forecast_dates, columns=["yhat"])

    async def _build_weather_future(
        self, data_last_window: pd.DataFrame, mlf
    ) -> pd.DataFrame | None:
        """Build the future-weather DataFrame required by a weather-trained MLForecaster.

        Returns a DataFrame aligned to the model's forecast horizon (``mlf.lags_opt`` when tuned,
        ``mlf.num_lags`` otherwise) containing the weather covariate columns the model was trained
        with, or ``None`` when the model was trained without weather features or when
        ``data_last_window`` is None.

        Factoring out this block avoids identical horizon-construction code in both
        ``_get_load_forecast_ml`` and ``command_line.forecast_model_predict``.

        :param data_last_window: The last observed window; its tail index + ``freq`` anchors the \
            future date range.
        :type data_last_window: pd.DataFrame
        :param mlf: A fitted (and optionally tuned) ``MLForecaster`` instance.
        :return: Future weather DataFrame, or None.
        :rtype: pd.DataFrame | None
        """
        if data_last_window is None:
            return None
        weather_features = list(getattr(mlf, "weather_features", []) or [])
        if not weather_features:
            return None
        # Resolve the index frequency — DatetimeIndex.freq can be None when the index was
        # constructed without an explicit freq (e.g. after a reindex or iloc slice).
        window_freq = data_last_window.index.freq
        if window_freq is None:
            window_freq = pd.tseries.frequencies.to_offset(pd.infer_freq(data_last_window.index))
        if window_freq is None:
            raise ValueError(
                "_build_weather_future: could not infer a regular frequency from "
                "data_last_window.index — ensure the index has a uniform step size."
            )
        steps = mlf.lags_opt if getattr(mlf, "is_tuned", False) else mlf.num_lags
        future_index = pd.date_range(
            start=data_last_window.index[-1] + window_freq,
            periods=steps,
            freq=window_freq,
        )
        # get_weather_covariates subtracts a tz-aware "now" from this index, so it must be
        # tz-aware; date_range inherits tz-naivety from data_last_window's index.
        future_index = (
            future_index.tz_localize(
                self.time_zone,
                ambiguous="infer",
                nonexistent="shift_forward",
            )
            if future_index.tz is None
            else future_index.tz_convert(self.time_zone)
        )
        return await self.get_weather_covariates(future_index, weather_features)

    async def _get_load_forecast_ml(
        self, df: pd.DataFrame, use_last_window: bool, mlf, debug: bool
    ) -> pd.DataFrame | bool:
        """Helper for ML forecast."""
        model_type = self.params["passed_data"]["model_type"]
        filename = model_type + "_mlf.pkl"
        filename_path = self.emhass_conf["data_path"] / filename
        if not debug:
            if filename_path.is_file():
                async with aiofiles.open(filename_path, "rb") as inp:
                    content = await inp.read()
                    mlf = pickle.loads(content)
            else:
                self.logger.error(
                    "The ML forecaster file was not found, please run a model fit method before this predict method"
                )
                return False
        data_last_window = None
        if use_last_window:
            data_last_window = copy.deepcopy(df)
            data_last_window = data_last_window.rename(columns={self.var_load_new: self.var_load})
        # When the model was trained with weather covariates, supply the future weather over the
        # forecast horizon so the recursive predict has the exog columns it expects.
        weather_future = await self._build_weather_future(data_last_window, mlf)
        forecast_out = await mlf.predict(data_last_window, weather_future=weather_future)
        self.logger.debug(
            "Number of ML predict forcast data generated (lags_opt): "
            + str(len(forecast_out.index))
        )
        self.logger.debug(
            "Number of forcast dates obtained (prediction_horizon): "
            + str(len(self.forecast_dates))
        )
        if len(self.forecast_dates) < len(forecast_out.index):
            forecast_out = forecast_out.iloc[0 : len(self.forecast_dates)]
        elif len(self.forecast_dates) > len(forecast_out.index):
            self.logger.error(
                "Unable to obtain: "
                + str(len(self.forecast_dates))
                + " lags_opt values from sensor: power load no var loads, check optimization_time_step/freq and historic_days_to_retrieve/days_to_retrieve parameters"
            )
            return False
        data_dict = {
            "ts": self.forecast_dates,
            "yhat": forecast_out.values.tolist(),
        }
        data = pd.DataFrame.from_dict(data_dict)
        data.set_index("ts", inplace=True)
        return data.copy().loc[self.forecast_dates]

    def _get_load_forecast_csv(self, csv_path: str) -> pd.DataFrame:
        """Helper to retrieve load data from CSV."""
        df_csv = pd.read_csv(csv_path, header=None, names=["ts", "yhat"])
        if len(df_csv) < len(self.forecast_dates):
            self.logger.error("Passed data from CSV is not long enough")
            return None
        df_csv = df_csv.loc[df_csv.index[0 : len(self.forecast_dates)], :]
        df_csv.index = self.forecast_dates
        df_csv = df_csv.drop(["ts"], axis=1)
        return df_csv.copy().loc[self.forecast_dates]

    def _get_load_forecast_list(self) -> pd.DataFrame:
        """Helper to retrieve load data from a passed list."""
        data_list = self.params["passed_data"]["load_power_forecast"]
        if data_list is None or (
            len(data_list) < len(self.forecast_dates)
            and self.params["passed_data"]["prediction_horizon"] is None
        ):
            self.logger.error(error_msg_list_not_long_enough)
            return False
        data_list = data_list[0 : len(self.forecast_dates)]
        data_dict = {"ts": self.forecast_dates, "yhat": data_list}
        data = pd.DataFrame.from_dict(data_dict)
        data.set_index("ts", inplace=True)
        return data.copy().loc[self.forecast_dates]

    async def get_load_forecast(
        self,
        days_min_load_forecast: int | None = 3,
        method: str | None = "typical",
        csv_path: str | None = "data_load_forecast.csv",
        set_mix_forecast: bool | None = False,
        df_now: pd.DataFrame | None = pd.DataFrame(),
        use_last_window: bool | None = True,
        mlf: MLForecaster | None = None,
        debug: bool | None = False,
    ) -> pd.Series:
        """
        Get and generate the load forecast data.

        :param days_min_load_forecast: The number of last days to retrieve that \
            will be used to generate a naive forecast, defaults to 3
        :type days_min_load_forecast: int, optional
        :param method: The method to be used to generate load forecast, the options \
            are 'typical' for a typical household load consumption curve, \
            are 'naive' for a persistence model, 'mlforecaster' for using a custom \
            previously fitted machine learning model, 'csv' to read the forecast from \
            a CSV file and 'list' to use data directly passed at runtime as a list of \
            values. Defaults to 'typical'.
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "/data/data_load_forecast.csv"
        :type csv_path: str, optional
        :param set_mix_forecast: Use a mixed forecast strategy to integrate now/current values.
        :type set_mix_forecast: Bool, optional
        :param df_now: The DataFrame containing the now/current data.
        :type df_now: pd.DataFrame, optional
        :param use_last_window: True if the 'last_window' option should be used for the \
            custom machine learning forecast model. The 'last_window=True' means that the data \
            that will be used to generate the new forecast will be freshly retrieved from \
            Home Assistant. This data is needed because the forecast model is an auto-regressive \
            model with lags. If 'False' then the data using during the model train is used.
        :type use_last_window: Bool, optional
        :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
            and unit testing. In production the actual model will be read from a saved pickle file.
        :type mlf: mlforecaster, optional
        :param debug: The DataFrame containing the now/current data.
        :type debug: Bool, optional
        :return: The DataFrame containing the electrical load power in Watts
        :rtype: pd.DataFrame

        """
        csv_path = self.emhass_conf["data_path"] / csv_path
        # Retrieve Data from Home Assistant if needed
        df = None
        if method in ["naive", "mlforecaster"]:
            df = await self._prepare_hass_load_data(days_min_load_forecast, method)
            if df is False:
                return False
        # Generate Forecast based on Method
        if method == "typical":
            forecast_out = await self._get_load_forecast_typical()
        elif method == "naive":
            forecast_out = self._get_load_forecast_naive(df)
        elif method == "mlforecaster":
            forecast_out = await self._get_load_forecast_ml(df, use_last_window, mlf, debug)
            if forecast_out is False:
                return False
        elif method == "csv":
            forecast_out = self._get_load_forecast_csv(csv_path)
            if forecast_out is None:
                return False
        elif method == "list":
            forecast_out = self._get_load_forecast_list()
            if forecast_out is False:
                return False
        else:
            self.logger.error(error_msg_method_not_valid)
            return False
        # Post-processing (Mix Forecast)
        p_load_forecast = copy.deepcopy(forecast_out["yhat"])
        bias = self._parse_load_quantile_bias()
        if bias > 0.0:
            p90 = await self._reconcile_load_percentile(p_load_forecast, 90.0)
            p_load_forecast = bias * p90 + (1.0 - bias) * p_load_forecast
        if set_mix_forecast:
            # Load forecasts don't need curtailment protection - always use feedback
            p_load_forecast = Forecast.get_mix_forecast(
                df_now,
                p_load_forecast,
                self.params["passed_data"]["alpha"],
                self.params["passed_data"]["beta"],
                self.var_load_new,
                False,  # Never ignore feedback for load forecasts
            )
        self.logger.debug("get_load_forecast returning:\n%s", p_load_forecast)
        return p_load_forecast

    def get_load_cost_forecast(
        self,
        df_final: pd.DataFrame,
        method: str | None = "hp_hc_periods",
        csv_path: str | None = "data_load_cost_forecast.csv",
        list_and_perfect: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Get the unit cost for the load consumption based on multiple tariff \
        periods. This is the cost of the energy from the utility in a vector \
        sampled at the fixed freq value.

        :param df_final: The DataFrame containing the input data.
        :type df_final: pd.DataFrame
        :param method: The method to be used to generate load cost forecast, \
            the options are 'hp_hc_periods' for peak and non-peak hours contracts\
            and 'csv' to load a CSV file, defaults to 'hp_hc_periods'
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "data_load_cost_forecast.csv"
        :type csv_path: str, optional
        :return: The input DataFrame with one additionnal column appended containing
            the load cost for each time observation.
        :rtype: pd.DataFrame

        """
        if df_final.index.dtype != "datetime64[ns, " + str(self.time_zone) + "]":
            df_final.index = df_final.index.astype("datetime64[ns, " + str(self.time_zone) + "]")
        csv_path = self.emhass_conf["data_path"] / csv_path
        if method == "hp_hc_periods":
            df_final[self.var_load_cost] = self.optim_conf["load_offpeak_hours_cost"]
            list_df_hp = []
            for _key, period_hp in self.optim_conf["load_peak_hour_periods"].items():
                list_df_hp.append(
                    df_final[self.var_load_cost].between_time(
                        period_hp[0]["start"], period_hp[1]["end"]
                    )
                )
            for df_hp in list_df_hp:
                df_final.loc[df_hp.index, self.var_load_cost] = self.optim_conf[
                    "load_peak_hours_cost"
                ]
        elif method == "csv":
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv_or_list(
                df_final, forecast_dates_csv, csv_path
            )
            # Ensure correct length
            if not list_and_perfect:
                forecast_out = forecast_out[0 : len(self.forecast_dates)]
                df_final = df_final[0 : len(self.forecast_dates)].copy()
            # Convert to Series if needed and align index
            if not isinstance(forecast_out, pd.Series):
                forecast_out = pd.Series(np.ravel(forecast_out), index=df_final.index)
            df_final.loc[:, self.var_load_cost] = forecast_out
        elif method == "list":  # reading a list of values
            # Loading data from passed list
            data_list = self.params["passed_data"]["load_cost_forecast"]
            # Check if the passed data has the correct length
            if (
                len(data_list) < len(self.forecast_dates)
                and self.params["passed_data"]["prediction_horizon"] is None
            ):
                self.logger.error(error_msg_list_not_long_enough)
                return False
            else:
                # Ensure correct length
                data_list = data_list[0 : len(self.forecast_dates)]
                if not list_and_perfect:
                    df_final = df_final.iloc[0 : len(self.forecast_dates)]
                # Define the correct dates
                forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
                forecast_out = self.get_forecast_out_from_csv_or_list(
                    df_final,
                    forecast_dates_csv,
                    None,
                    data_list=data_list,
                    list_and_perfect=list_and_perfect,
                )
                df_final = df_final.copy()
                df_final[self.var_load_cost] = forecast_out
        else:
            self.logger.error(error_msg_method_not_valid)
            return False
        self.logger.debug("get_load_cost_forecast returning:\n%s", df_final)
        return df_final

    def get_prod_price_forecast(
        self,
        df_final: pd.DataFrame,
        method: str | None = "constant",
        csv_path: str | None = "data_prod_price_forecast.csv",
        list_and_perfect: bool | None = False,
    ) -> pd.DataFrame:
        r"""
        Get the unit power production price for the energy injected to the grid.\
        This is the price of the energy injected to the utility in a vector \
        sampled at the fixed freq value.

        :param df_input_data: The DataFrame containing all the input data retrieved
            from hass
        :type df_input_data: pd.DataFrame
        :param method: The method to be used to generate the production price forecast, \
            the options are 'constant' for a fixed constant value and 'csv'\
            to load a CSV file, defaults to 'constant'
        :type method: str, optional
        :param csv_path: The path to the CSV file used when method = 'csv', \
            defaults to "/data/data_load_cost_forecast.csv"
        :type csv_path: str, optional
        :return: The input DataFrame with one additionnal column appended containing
            the power production price for each time observation.
        :rtype: pd.DataFrame

        """
        if df_final.index.dtype != "datetime64[ns, " + str(self.time_zone) + "]":
            df_final.index = df_final.index.astype("datetime64[ns, " + str(self.time_zone) + "]")
        csv_path = self.emhass_conf["data_path"] / csv_path
        if method == "constant":
            df_final[self.var_prod_price] = self.optim_conf["photovoltaic_production_sell_price"]
        elif method == "csv":
            forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
            forecast_out = self.get_forecast_out_from_csv_or_list(
                df_final, forecast_dates_csv, csv_path
            )
            # Ensure correct length
            if not list_and_perfect:
                forecast_out = forecast_out[0 : len(self.forecast_dates)]
                df_final = df_final[0 : len(self.forecast_dates)].copy()
            # Convert to Series if needed and align index
            if not isinstance(forecast_out, pd.Series):
                forecast_out = pd.Series(np.ravel(forecast_out), index=df_final.index)
            df_final.loc[:, self.var_prod_price] = forecast_out
        elif method == "list":  # reading a list of values
            # Loading data from passed list
            data_list = self.params["passed_data"]["prod_price_forecast"]
            # Check if the passed data has the correct length
            if (
                len(data_list) < len(self.forecast_dates)
                and self.params["passed_data"]["prediction_horizon"] is None
            ):
                self.logger.error(error_msg_list_not_long_enough)
                return False
            else:
                # Ensure correct length
                data_list = data_list[0 : len(self.forecast_dates)]
                if not list_and_perfect:
                    df_final = df_final.iloc[0 : len(self.forecast_dates)]
                # Define the correct dates
                forecast_dates_csv = self.get_forecast_days_csv(timedelta_days=0)
                forecast_out = self.get_forecast_out_from_csv_or_list(
                    df_final,
                    forecast_dates_csv,
                    None,
                    data_list=data_list,
                    list_and_perfect=list_and_perfect,
                )
                df_final = df_final.copy()
                df_final[self.var_prod_price] = forecast_out
        else:
            self.logger.error(error_msg_method_not_valid)
            return False
        self.logger.debug("get_prod_price_forecast returning:\n%s", df_final)
        return df_final

    async def get_cached_forecast_data(self, w_forecast_cache_path) -> pd.DataFrame | None:
        r"""
        Get cached weather forecast data from file.

        :param w_forecast_cache_path: the path to file.
        :type method: Any
        :return: The DataFrame containing the forecasted data, or ``None`` when
            the cache is corrupt/missing or was intentionally deleted (e.g. stale
            Open-Meteo cache).  Callers must handle the ``None`` case.
        :rtype: pd.DataFrame | None

        """
        # Read then close the file BEFORE any os.remove below: on Windows a file
        # cannot be unlinked while a handle is still open (PermissionError WinError 32).
        async with aiofiles.open(w_forecast_cache_path, "rb") as file:
            content = await file.read()
        data = pickle.loads(content)
        if not isinstance(data, pd.DataFrame) or data.empty:
            self.logger.error("Cache file is corrupt or empty.")
            self.logger.error(
                "Try running action `weather-forecast-cache` to pull new data from forecast API."
            )
            try:
                os.remove(w_forecast_cache_path)
            except FileNotFoundError:
                pass
            return None
        # Filter cached forecast data to match current forecast_dates start-end range
        if self.forecast_dates[0] in data.index and self.forecast_dates[-1] in data.index:
            data = data.loc[self.forecast_dates[0] : self.forecast_dates[-1]]
            self.logger.info("Retrieved forecast data from the previously saved cache.")
        else:
            if self.weather_forecast_method in ("open-meteo", "list"):
                # Open-Meteo has no rate limits: delete the stale cache so
                # the next call fetches fresh data directly from the API. The
                # "list" method only reaches this cache via the open-meteo
                # weather augmentation (issue #997), which is open-meteo-sourced
                # too, so it gets the same refetch-fresh treatment rather than
                # being served stale, zero-filled irradiance.
                self.logger.warning(
                    "Cache does not fully cover the requested timeframe. "
                    "Removing stale Open-Meteo cache; fresh data will be fetched from the API."
                )
                try:
                    os.remove(w_forecast_cache_path)
                except FileNotFoundError:
                    pass
                return None
            else:
                # Solcast and other rate-limited APIs: serve best-effort
                # stale data to avoid burning daily API quota.
                self.logger.warning(
                    "Cache does not fully cover the requested timeframe. "
                    "Serving best-effort stale data (reindexed + zero-filled). "
                    "Cache preserved to protect API rate limits."
                )
                combined_index = data.index.union(self.forecast_dates).sort_values()
                data = data.reindex(combined_index)
                data.interpolate(method="time", inplace=True)
                data = data.reindex(self.forecast_dates)

                irradiance_cols = [c for c in ["ghi", "dni", "dhi"] if c in data.columns]
                other_cols = [c for c in data.columns if c not in irradiance_cols]

                if other_cols:
                    data[other_cols] = data[other_cols].ffill().bfill()
                if irradiance_cols:
                    data[irradiance_cols] = data[irradiance_cols].fillna(0.0)

                data = data.fillna(0.0)

        # The weather cache file is shared across every weather_forecast_method,
        # so a cache written by a different method can lack the column the active
        # method needs. Serving it would crash later in get_power_from_weather
        # with an opaque KeyError: 'yhat' (issue #932). Treat a schema-incompatible
        # cache as a recoverable miss: drop it and return None so the rate-limited
        # fetchers fall through to a fresh fetch (the open-meteo path already does
        # this for its own stale cache). 'yhat' (PV power) is required by both
        # solcast and solar.forecast regardless of solar_forecast_kwp, so the
        # check does not depend on that key.
        if (
            self.weather_forecast_method in ("solcast", "solar.forecast")
            and "yhat" not in data.columns
        ):
            self.logger.warning(
                "Cached forecast is missing the 'yhat' column required by "
                "weather_forecast_method='%s' (cache likely written by a "
                "different method). Discarding the incompatible cache and "
                "fetching fresh data.",
                self.weather_forecast_method,
            )
            try:
                os.remove(w_forecast_cache_path)
            except FileNotFoundError:
                pass
            return None
        return data

    async def set_cached_forecast_data(self, w_forecast_cache_path, data) -> pd.DataFrame:
        r"""
        Set generated weather forecast data to file.
        Trim data to match the original requested forecast dates

        :param w_forecast_cache_path: the path to file.
        :type method: Any
        :param: The DataFrame containing the forecasted data
        :type: pd.DataFrame
        :return: The DataFrame containing the forecasted data
        :rtype: pd.DataFrame

        """
        async with aiofiles.open(w_forecast_cache_path, "wb") as file:
            content = pickle.dumps(data)
            await file.write(content)
            if not os.path.isfile(w_forecast_cache_path):
                self.logger.warning("forecast data could not be saved to file.")
            else:
                self.logger.info("Saved the forecast results to cache, for later reference.")

        # Trim cached data to match requested dates
        end_forecast = (
            self.start_forecast + pd.DateOffset(days=self.optim_conf["delta_forecast_daily"].days)
        ).replace(microsecond=0)
        forecast_dates = (
            pd.date_range(
                start=self.start_forecast,
                end=end_forecast - self.freq,
                freq=self.freq,
                tz=self.time_zone,
            )
            .tz_convert("utc")
            .round(self.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.time_zone)
        )
        data = data.loc[forecast_dates[0] : forecast_dates[-1]]
        return data
