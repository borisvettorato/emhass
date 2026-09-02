from __future__ import annotations

import ast
import copy
import csv
import logging
import os
import pathlib
import re
import shutil
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

try:
    from datetime import UTC
except ImportError:
    # Python 3.10 compatibility
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - this *is* the fallback for when datetime.UTC doesn't exist

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import pytz
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from emhass.persistence import load_json_blob

if TYPE_CHECKING:
    from emhass.machine_learning_forecaster import MLForecaster

pd.options.plotting.backend = "plotly"

# Unit conversion constants
W_TO_KW = 1000  # Watts to kilowatts conversion factor


def get_root(file: str, num_parent: int = 3) -> str:
    """
    Get the root absolute path of the working directory.

    :param file: The passed file path with __file__
    :return: The root path
    :param num_parent: The number of parents levels up to desired root folder
    :type num_parent: int, optional
    :rtype: str

    """
    if num_parent == 3:
        root = pathlib.Path(file).resolve().parent.parent.parent
    elif num_parent == 2:
        root = pathlib.Path(file).resolve().parent.parent
    elif num_parent == 1:
        root = pathlib.Path(file).resolve().parent
    else:
        raise ValueError("num_parent value not valid, must be between 1 and 3")
    return root


def get_logger(
    fun_name: str,
    emhass_conf: dict[str, pathlib.Path],
    save_to_file: bool = True,
    logging_level: str = "DEBUG",
) -> tuple[logging.Logger, logging.StreamHandler]:
    """
    Create a simple logger object.

    :param fun_name: The Python function object name where the logger will be used
    :type fun_name: str
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param save_to_file: Write log to a file, defaults to True
    :type save_to_file: bool, optional
    :return: The logger object and the handler
    :rtype: object

    """
    # create logger object
    logger = logging.getLogger(fun_name)
    logger.propagate = True
    logger.fileSetting = save_to_file
    if save_to_file:
        if os.path.isdir(emhass_conf["data_path"]):
            ch = logging.FileHandler(emhass_conf["data_path"] / "logger_emhass.log")
        else:
            raise Exception("Unable to access data_path: " + emhass_conf["data_path"])
    else:
        ch = logging.StreamHandler()
    if logging_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        logger.setLevel(logging.ERROR)
        ch.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, ch


def _get_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime. Separated out for easier mocking in tests."""
    return datetime.now(UTC)


def get_forecast_dates(
    freq: int,
    delta_forecast: int,
    time_zone: datetime.tzinfo,
    timedelta_days: int | None = 0,
) -> pd.core.indexes.datetimes.DatetimeIndex:
    """
    Get the date_range list of the needed future dates using the delta_forecast parameter.

    :param freq: Optimization time step.
    :type freq: int
    :param delta_forecast: Number of days to forecast in the future to be used for the optimization.
    :type delta_forecast: int
    :param timedelta_days: Number of truncated days needed for each optimization iteration, defaults to 0
    :type timedelta_days: Optional[int], optional
    :return: A list of future forecast dates.
    :rtype: pd.core.indexes.datetimes.DatetimeIndex

    """
    freq = pd.to_timedelta(freq, "minutes")
    start_time = _get_now()

    # start_time is the timezone-aware UTC instant from _get_now(); tz_convert expresses it in
    # the configured timezone regardless of the host clock. It raises on a naive value, so a
    # refactor that reintroduced a naive "now" here would fail loudly rather than silently shift.
    start_forecast = (
        pd.Timestamp(start_time).tz_convert(time_zone).replace(microsecond=0).floor(freq=freq)
    )
    end_forecast = start_forecast + pd.tseries.offsets.DateOffset(days=delta_forecast)
    final_end_date = end_forecast + pd.tseries.offsets.DateOffset(days=timedelta_days) - freq

    forecast_dates = pd.date_range(
        start=start_forecast,
        end=final_end_date,
        freq=freq,
        tz=time_zone,
    )

    return [ts.isoformat() for ts in forecast_dates]


def normalize_heat_cool_mode(
    value: str,
    *,
    field_name: str = "mode",
    context: str | None = None,
) -> str:
    """Normalize heat/cool mode values and raise on invalid input."""
    mode_norm = str(value).strip().lower()
    if mode_norm not in {"heat", "cool"}:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}invalid {field_name} '{value}'. Expected 'heat' or 'cool'.")
    return mode_norm


def calculate_cop_heatpump(
    supply_temperature: float | np.ndarray | pd.Series,
    carnot_efficiency: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    mode: str = "heat",
) -> np.ndarray:
    r"""
    Calculate heat pump Coefficient of Performance (COP) for each timestep in the prediction horizon.

        The COP is calculated using a Carnot-based formula. Two modes are supported:

        - ``mode='heat'`` (default):

            .. math::
                    COP_{heat}(h) = \eta_{carnot} \times \frac{T_{supply\_K}}{T_{supply\_K} - T_{outdoor\_K}(h)}

        - ``mode='cool'``:

            .. math::
                    COP_{cool}(h) = \eta_{carnot} \times \frac{T_{supply\_K}}{T_{outdoor\_K}(h) - T_{supply\_K}}

    Where temperatures are converted to Kelvin (K = °C + 273.15).

    This formula models real heat pump behavior where COP decreases as the temperature lift
    (difference between supply and outdoor temperature) increases. The carnot_efficiency factor
    represents the real-world efficiency as a fraction of the ideal Carnot cycle efficiency.

    :param supply_temperature: The heat pump supply temperature in degrees Celsius. \
        Can be a scalar (constant supply T) or an array/Series matching the length of \
        ``outdoor_temperature_forecast`` for weather-compensated supply T (heating curve). \
        Typical scalar values: 30-40°C for underfloor heating, 50-70°C for radiator systems.
    :type supply_temperature: float or np.ndarray or pd.Series
    :param carnot_efficiency: Real-world efficiency factor as fraction of ideal Carnot cycle. \
        Typical range: 0.35-0.50 (35-50%). Default in thermal battery config: 0.4 (40%). \
        Higher values represent more efficient heat pumps.
    :type carnot_efficiency: float
    :param outdoor_temperature_forecast: Array of outdoor temperature forecasts in degrees Celsius, \
        one value per timestep in the prediction horizon.
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :param mode: Operating mode, either ``"heat"`` or ``"cool"``. In cooling mode,
        the Carnot lift uses ``T_outdoor - T_supply`` so warm weather no longer
        collapses to COP=1.0 as in heating-only validation.
    :type mode: str
    :return: Array of COP values for each timestep, same length as outdoor_temperature_forecast. \
        Typical COP range: 2-6 for normal operating conditions.
    :rtype: np.ndarray

    Example:
        >>> supply_temp = 35.0  # °C, underfloor heating
        >>> carnot_eff = 0.4  # 40% of ideal Carnot efficiency
        >>> outdoor_temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        >>> cops = calculate_cop_heatpump(supply_temp, carnot_eff, outdoor_temps)
        >>> cops
        array([3.521..., 4.108..., 4.926..., 6.163..., 8.217...])
        >>> # At 5°C outdoor: COP = 0.4 × 308.15K / 30K = 4.11

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Supply temp can be scalar (constant) or array (heating curve). Broadcast either way.
    if isinstance(supply_temperature, pd.Series):
        supply_temps = supply_temperature.values
    elif np.ndim(supply_temperature) > 0:
        supply_temps = np.asarray(supply_temperature, dtype=float)
    else:
        supply_temps = float(supply_temperature)

    # Convert temperatures from Celsius to Kelvin for Carnot formula
    supply_temperature_kelvin = supply_temps + 273.15
    outdoor_temperature_kelvin = outdoor_temps + 273.15

    mode_norm = normalize_heat_cool_mode(mode, field_name="mode", context="COP calculation")

    # Calculate temperature lift depending on mode.
    if mode_norm == "heat":
        # Heating: source is outdoor, sink is supply side.
        temperature_diff = supply_temperature_kelvin - outdoor_temperature_kelvin
        invalid_relation = "outdoor temperature >= supply temperature"
    else:
        # Cooling: source is indoor/chilled supply side, sink is outdoor.
        temperature_diff = outdoor_temperature_kelvin - supply_temperature_kelvin
        invalid_relation = "outdoor temperature <= supply temperature"

    # Check for non-physical scenarios where Carnot lift is non-positive.
    if np.any(temperature_diff <= 0):
        logger = logging.getLogger(__name__)
        num_invalid = int(np.sum(temperature_diff <= 0))
        invalid_indices = np.nonzero(temperature_diff <= 0)[0]
        if np.ndim(supply_temps) > 0:
            supply_range = (
                f"supply range [{np.min(supply_temps):.1f}, {np.max(supply_temps):.1f}]°C"
            )
        else:
            supply_range = f"supply {float(supply_temps):.1f}°C"
        logger.warning(
            f"COP calculation: {num_invalid} timestep(s) have {invalid_relation}. "
            f"This is non-physical for {mode_norm} mode. "
            f"Indices: {invalid_indices.tolist()[:5]}{'...' if len(invalid_indices) > 5 else ''}. "
            f"{supply_range}. Setting COP to 1.0 for these periods."
        )

    # Vectorized Carnot-based COP calculation.
    # For non-physical cases (lift <= 0), use a neutral COP of 1.0.

    # Avoid division by zero: use a mask to only calculate for valid cases
    cop_values = np.ones_like(outdoor_temperature_kelvin)  # Default to 1.0 everywhere
    valid_mask = temperature_diff > 0
    if np.any(valid_mask):
        # supply_temperature_kelvin may be scalar or array - index when array.
        if np.ndim(supply_temperature_kelvin) > 0:
            supply_valid = supply_temperature_kelvin[valid_mask]
        else:
            supply_valid = supply_temperature_kelvin
        cop_values[valid_mask] = carnot_efficiency * supply_valid / temperature_diff[valid_mask]

    # Apply realistic bounds: minimum 1.0, maximum 8.0
    # - Lower bound: 1.0 means direct electric heating (no efficiency gain)
    # - Upper bound: 8.0 is an optimistic but reasonable maximum for modern heat pumps
    #   (prevents numerical instability from very small temperature differences)
    cop_values = np.clip(cop_values, 1.0, 8.0)

    return cop_values


def apply_heating_curve(
    heating_curve: dict,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute per-slot supply temperature from a weather-compensated heating curve.

    A heating curve specifies how the heat source modulates its supply temperature in
    response to outdoor temperature, the way every modern boiler / heat pump controller
    does. The linear form supported here:

        T_supply(t) = clip(offset - slope * T_outdoor(t), min_supply, max_supply)

    :param heating_curve: dict with required ``slope`` and ``offset`` (both °C), plus
        optional ``min_supply`` (default 25°C) and ``max_supply`` (default 70°C).
    :param outdoor_temperature_forecast: Per-slot outdoor temperature in °C.
    :return: Per-slot supply temperature array in °C.
    """
    slope = float(heating_curve["slope"])
    offset = float(heating_curve["offset"])
    min_supply = float(heating_curve.get("min_supply", 25.0))
    max_supply = float(heating_curve.get("max_supply", 70.0))
    if min_supply >= max_supply:
        raise ValueError(
            f"heating_curve: min_supply ({min_supply}) must be < max_supply ({max_supply})"
        )
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor = outdoor_temperature_forecast.values
    else:
        outdoor = np.asarray(outdoor_temperature_forecast, dtype=float)
    supply = offset - slope * outdoor
    return np.clip(supply, min_supply, max_supply)


def resolve_min_temperatures(
    config: dict,
    outdoor_temperature_forecast: np.ndarray | pd.Series | list | None,
    length: int,
) -> list[float]:
    """Compute the effective per-slot lower temperature bound for a storage tank.

    Combines two sources, taking the element-wise max so the more conservative
    floor always wins:

    1. Static ``min_temperatures`` (or ``min_temperature``) list - an absolute
       weather-independent floor for safety / comfort. Always respected.
    2. ``min_temperature_curve`` dict - a weather-compensated floor matching
       the radiator emission law. Same shape as ``heating_curve``:
       ``T = clip(offset - slope * T_outdoor, min_supply, max_supply)``.

    When only one is set, that one is used. When neither is set, returns an
    empty list (caller decides how to react).

    :param config: Tank or thermal_battery dict potentially carrying
        ``min_temperatures`` and/or ``min_temperature_curve``.
    :param outdoor_temperature_forecast: Per-slot outdoor temperature. May be
        ``None`` when only the static floor is configured.
    :param length: Optimization horizon length.
    :return: List of per-slot minimum temperatures (length = ``length``).
    """
    # Accept either the plural list form or a single scalar/list under the
    # singular key. Normalize a scalar to a one-element list so the slicing
    # / padding below works uniformly.
    static = config.get("min_temperatures")
    if static is None:
        static = config.get("min_temperature")
    if static is None:
        static = []
    elif isinstance(static, int | float):
        static = [float(static)]
    curve = config.get("min_temperature_curve")

    if not static and not curve:
        return []

    if curve is not None:
        if outdoor_temperature_forecast is None:
            raise ValueError("min_temperature_curve requires outdoor_temperature_forecast")
        curve_temps = apply_heating_curve(curve, outdoor_temperature_forecast)[:length]
    else:
        curve_temps = None

    if static:
        static_arr = np.asarray(list(static)[:length], dtype=float)
        if len(static_arr) < length:
            pad_value = static_arr[-1] if len(static_arr) else 20.0
            static_arr = np.concatenate([static_arr, np.full(length - len(static_arr), pad_value)])
    else:
        static_arr = None

    if curve_temps is not None and static_arr is not None:
        effective = np.maximum(static_arr, curve_temps)
    elif curve_temps is not None:
        effective = curve_temps
    else:
        effective = static_arr

    return [float(v) for v in effective]


def resolve_thermal_battery_cop(
    hc: dict,
    outdoor_temperature_forecast: Sequence[float] | np.ndarray | pd.Series | None,
    length: int | None = None,
) -> np.ndarray:
    """
    Resolve the per-timestep energy-conversion factor for a thermal_battery heat source.

    The thermal_battery model treats the conversion factor uniformly as a COP-like
    multiplier on input power: `Q_thermal = COP * P_in / 1000 * dt`. Two source types
    are supported:

        - Heat pump (default): COP is computed via the Carnot formula and varies with
            the outdoor temperature. Requires `supply_temperature` in `hc`; uses
            `carnot_efficiency` (default 0.4). If `sense` is set to `"cool"`, cooling
            Carnot lift is used (`T_outdoor - T_supply`).
    - Constant-efficiency source (gas boiler, oil burner, district heating, etc.):
      `efficiency` in `hc` is used as a flat conversion factor for every timestep.
      `supply_temperature` is not required and outdoor temperature is ignored. The
      constant value passes through Carnot bounds intentionally (typical 0.85-0.95
      for combustion sources).

    :param hc: The thermal_battery sub-config dict from def_load_config.
    :param outdoor_temperature_forecast: Outdoor temperature forecast in degrees
        Celsius. Required in heat-pump mode. In constant-efficiency mode it is
        ignored and may be ``None`` provided ``length`` is given explicitly.
    :param length: Number of timesteps in the returned array. Mandatory when
        ``outdoor_temperature_forecast`` is ``None``; otherwise truncates or
        passes through the forecast length when set, or returns the full forecast
        length when unset.
    :return: Numpy array of conversion factors, one per timestep.
    """
    if "efficiency" in hc and hc["efficiency"] is not None:
        efficiency = float(hc["efficiency"])
        if efficiency <= 0:
            raise ValueError(f"thermal_battery 'efficiency' must be positive, got {efficiency}")
        if length is None:
            if outdoor_temperature_forecast is None:
                raise ValueError(
                    "resolve_thermal_battery_cop in constant-efficiency mode "
                    "requires 'length' when outdoor_temperature_forecast is None"
                )
            length = len(outdoor_temperature_forecast)
        return np.full(length, efficiency)

    if outdoor_temperature_forecast is None:
        raise ValueError(
            "resolve_thermal_battery_cop in heat-pump mode requires outdoor_temperature_forecast"
        )
    # Heating-curve mode: per-slot supply T from outdoor T. Falls back to constant
    # `supply_temperature` when no curve is configured.
    heating_curve = hc.get("heating_curve")
    if heating_curve:
        supply_temperature = apply_heating_curve(heating_curve, outdoor_temperature_forecast)
    else:
        supply_temperature = hc.get("supply_temperature")
        if supply_temperature is None:
            raise ValueError(
                "thermal_battery requires either 'efficiency' (constant-efficiency mode), "
                "'supply_temperature' (constant heat-pump mode), or 'heating_curve' "
                "(weather-compensated heat-pump mode)"
            )
    cops = calculate_cop_heatpump(
        supply_temperature=supply_temperature,
        carnot_efficiency=hc.get("carnot_efficiency", 0.4),
        outdoor_temperature_forecast=outdoor_temperature_forecast,
        mode=normalize_heat_cool_mode(
            hc.get("sense") or "heat",
            field_name="sense",
            context="thermal_battery",
        ),
    )
    return cops if length is None else cops[:length]


def calculate_surface_solar_gain(
    hc: dict,
    ghi_forecast: np.ndarray | None,
    optimization_time_step_minutes: float,
    length: int | None = None,
) -> np.ndarray | None:
    """
    Compute per-timestep solar energy absorbed by an exposed thermal mass surface.

    Intended for thermal_battery configs that model a thermal store exposed
    directly to sunlight (a pool, an outdoor tank, a solar-thermal collector
    routed into a buffer). The gain is independent of the heater and acts
    as a negative term on `heating_demand` (i.e. the optimizer needs less
    pumped heat to maintain temperature when there is solar gain).

    Reuses the existing GHI forecast that EMHASS already fetches for PV. No
    second weather API call is required.

    :param hc: The thermal_battery sub-config dict. Reads two keys:
        - `solar_absorption_area`: effective horizontal absorption surface (m²).
        - `solar_absorption_factor`: fraction of GHI absorbed by the thermal
          mass (typical pool with no cover: 0.7-0.9; covered pool: 0.2-0.4).
          Defaults to 0.7 if absent.
    :param ghi_forecast: Global horizontal irradiance forecast in W/m² per
        timestep. Pass None or zero-length array to skip gain.
    :param optimization_time_step_minutes: Timestep duration in minutes.
    :param length: Truncate / pad the returned array to this length.
    :return: Solar gain in kWh per timestep, or None if not applicable.
    """
    absorption_area = hc.get("solar_absorption_area")
    if absorption_area is None or float(absorption_area) <= 0:
        return None
    if ghi_forecast is None:
        return None

    ghi_arr = np.asarray(ghi_forecast, dtype=float)
    if length is not None:
        if len(ghi_arr) < length:
            ghi_arr = np.concatenate((ghi_arr, np.zeros(length - len(ghi_arr))))
        else:
            ghi_arr = ghi_arr[:length]

    absorption_factor = float(hc.get("solar_absorption_factor", 0.7))
    if absorption_factor < 0:
        raise ValueError(
            f"thermal_battery solar_absorption_factor must be >= 0, got {absorption_factor}"
        )
    dt_hours = optimization_time_step_minutes / 60.0
    # W/m² * m² * factor / 1000 (kW per W) * hours = kWh
    return ghi_arr * float(absorption_area) * absorption_factor / 1000.0 * dt_hours


def compile_heat_topology(topology: dict) -> dict:
    """Compile a heat-topology graph descriptor into flat optim_conf fields.

    The graph model lets users declare a small directed graph of
    `sources`, `storage`, `consumers`, `flows`, and `actuator_groups`. This
    function translates that high-level descriptor into the primitives the
    optimizer already understands:

    - Each `flow (source, storage)` becomes one deferrable load with a
      `thermal_source` block in def_load_config.
    - Each `storage` becomes one entry in `shared_thermal_tanks` with
      `load_ids` pointing at the deferrable loads feeding it.
    - Each `consumer` is folded into its target storage (its profile or
      building model populates the storage's draw_off_demand / u_value etc.).
    - Each `actuator_group` becomes one entry in `deferrable_load_groups`.
    - Each source's `cost_track` reference resolves to an entry in
      `cost_forecast_per_deferrable_load`.

    Returns a dict of optim_conf fields to merge into the live optim_conf:
        number_of_deferrable_loads, nominal_power_of_deferrable_loads,
        minimum_power_of_deferrable_loads, treat_deferrable_load_as_semi_cont,
        operating_hours_of_each_deferrable_load, def_load_config,
        shared_thermal_tanks, deferrable_load_groups,
        cost_forecast_per_deferrable_load.

    Validation: missing source/storage references, duplicated ids, and
    consumers targeting unknown storage all raise ValueError with the
    offending field path.
    """
    if not isinstance(topology, dict) or not topology:
        return {}
    sources = topology.get("sources", []) or []
    storage = topology.get("storage", []) or []
    consumers = topology.get("consumers", []) or []
    flows = topology.get("flows", []) or []
    groups = topology.get("actuator_groups", []) or []
    cost_tracks = topology.get("cost_tracks", {}) or {}

    src_by_id = {s["id"]: s for s in sources}
    src_index_by_id = {s["id"]: i for i, s in enumerate(sources)}
    sto_by_id = {s["id"]: s for s in storage}
    if len(src_by_id) != len(sources):
        raise ValueError("heat_topology.sources contains duplicate ids")
    if len(sto_by_id) != len(storage):
        raise ValueError("heat_topology.storage contains duplicate ids")

    # Validate flows reference real source/storage ids
    for i, f in enumerate(flows):
        if f.get("from") not in src_by_id:
            raise ValueError(
                f"heat_topology.flows[{i}].from='{f.get('from')}' does not match any source.id"
            )
        if f.get("to") not in sto_by_id:
            raise ValueError(
                f"heat_topology.flows[{i}].to='{f.get('to')}' does not match any storage.id"
            )

    # Validate consumers target real storage
    for i, c in enumerate(consumers):
        if c.get("target") not in sto_by_id:
            raise ValueError(
                f"heat_topology.consumers[{i}].target='{c.get('target')}' does not match any storage.id"
            )

    # Build deferrable loads from flows
    num_loads = len(flows)
    nominal_power = []
    min_power = []
    treat_semi_cont = []
    operating_hours = []
    def_load_config = []
    cost_per_load: list = []
    is_electric_load: list[bool] = []
    flow_to_load_idx: dict[tuple[str, str], int] = {}

    for i, f in enumerate(flows):
        src = src_by_id[f["from"]]
        nominal_power.append(float(src.get("nominal_power", 0)))
        min_power.append(float(src.get("min_power", 0)))
        treat_semi_cont.append(bool(src.get("treat_as_semi_cont", True)))
        operating_hours.append(int(src.get("operating_hours", 4)))
        # Source-side fields - shape expected by resolve_thermal_battery_cop
        source_block: dict = {}
        src_type = src.get("type", "").lower()
        # Type -> default electric bus membership. Explicit `electric` flag wins.
        type_is_electric = {
            "heatpump": True,
            "heat_pump": True,
            "electric": True,
            "gas": False,
            "oil": False,
            "district": False,
            "constant_efficiency": True,  # ambiguous - default electric unless overridden
        }
        if src_type in {"heatpump", "heat_pump"}:
            # Heating curve takes precedence; constant supply_temperature is the fallback.
            if "heating_curve" in src and src["heating_curve"]:
                hc_block = dict(src["heating_curve"])
                # Validate required fields up-front so users get clear errors
                for required in ("slope", "offset"):
                    if required not in hc_block:
                        raise ValueError(
                            f"heat_topology.sources[{src['id']}].heating_curve missing "
                            f"required field '{required}'"
                        )
                source_block["heating_curve"] = {
                    "slope": float(hc_block["slope"]),
                    "offset": float(hc_block["offset"]),
                    "min_supply": float(hc_block.get("min_supply", 25.0)),
                    "max_supply": float(hc_block.get("max_supply", 70.0)),
                }
            elif "supply_temperature" in src:
                source_block["supply_temperature"] = float(src["supply_temperature"])
            else:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}] (type=heatpump) requires "
                    "either 'supply_temperature' or 'heating_curve'"
                )
            source_block["carnot_efficiency"] = float(src.get("carnot_efficiency", 0.4))
        elif src_type in {"gas", "oil", "district", "constant_efficiency", "electric"}:
            source_block["efficiency"] = float(src["efficiency"])
        else:
            raise ValueError(
                f"heat_topology.sources[{src_index_by_id[src['id']]}] "
                f"(id='{src['id']}'): type='{src.get('type', '')}' is not "
                "recognised. Allowed types: heatpump, heat_pump, gas, oil, "
                "district, electric, constant_efficiency."
            )
        # Propagate the target storage's comfort_sense onto the source block so the
        # COP resolver computes the correct (heating vs cooling) Carnot lift. Without
        # this, resolve_thermal_battery_cop defaults to "heat" and clamps the cooling
        # COP to 1.0 on a warm day (a heat pump can then never cool the zone).
        target_storage = sto_by_id.get(f["to"], {})
        if "comfort_sense" in target_storage:
            source_block["sense"] = str(target_storage["comfort_sense"]).lower()
        is_electric_load.append(bool(src.get("electric", type_is_electric.get(src_type, True))))
        def_load_config.append({"thermal_source": source_block})
        # Cost track resolution
        cost_track_id = src.get("cost_track")
        if cost_track_id is None:
            cost_per_load.append(None)
        else:
            if cost_track_id not in cost_tracks:
                raise ValueError(
                    f"heat_topology.sources[{src['id']}].cost_track='{cost_track_id}' "
                    "not found in cost_tracks"
                )
            cost_per_load.append(list(cost_tracks[cost_track_id]))
        flow_to_load_idx[(f["from"], f["to"])] = i

    # Aggregate consumer demand onto storage
    storage_demand: dict[str, dict] = {}
    for c in consumers:
        target = c["target"]
        if target not in storage_demand:
            storage_demand[target] = {"profile": None, "building": None, "pool": None}
        ctype = (c.get("type") or "").lower()
        if ctype == "profile":
            prof = list(c["profile"])
            existing = storage_demand[target]["profile"]
            if existing is None:
                storage_demand[target]["profile"] = prof
            else:
                # Pad both lists to the common max length so a later profile
                # that is longer than `existing` is not silently truncated by
                # zip(). Demand after the shorter horizon stays at 0.
                max_len = max(len(existing), len(prof))
                existing_padded = existing + [0.0] * (max_len - len(existing))
                prof_padded = prof + [0.0] * (max_len - len(prof))
                storage_demand[target]["profile"] = [
                    a + b for a, b in zip(existing_padded, prof_padded)
                ]
        elif ctype == "building_demand":
            if storage_demand[target]["building"] is not None:
                raise ValueError(
                    f"heat_topology.consumers: storage '{target}' already has a "
                    "building_demand consumer; only one is permitted per storage"
                )
            storage_demand[target]["building"] = {
                k: c[k]
                for k in (
                    "u_value",
                    "envelope_area",
                    "ventilation_rate",
                    "heated_volume",
                    "indoor_target_temperature",
                    "window_area",
                    "shgc",
                    "internal_gains_factor",
                    "specific_heating_demand",
                    "area",
                    "base_temperature",
                    "annual_reference_hdd",
                )
                if k in c
            }
        elif ctype == "pool_comfort":
            storage_demand[target]["pool"] = {
                k: c[k] for k in ("solar_absorption_area", "solar_absorption_factor") if k in c
            }
        else:
            raise ValueError(
                f"heat_topology.consumers[{c.get('id')}].type='{ctype}' must be one of "
                "profile, building_demand, pool_comfort"
            )

    # Build shared_thermal_tanks from storage + aggregated demand + flows
    shared_tanks = []
    for s in storage:
        sid = s["id"]
        load_ids = [flow_to_load_idx[(f["from"], f["to"])] for f in flows if f["to"] == sid]
        tank: dict = {
            "id": sid,
            "load_ids": load_ids,
            "volume": float(s["volume"]),
            "density": float(s.get("density", 1000)),
            "heat_capacity": float(s.get("heat_capacity", 4.186)),
            "start_temperature": float(s.get("start_temperature", 20.0)),
            "thermal_loss": float(s.get("thermal_loss", 0.045)),
            "min_temperatures": list(s.get("min_temperature", []))
            or list(s.get("min_temperatures", [])),
            "max_temperatures": list(s.get("max_temperature", []))
            or list(s.get("max_temperatures", [])),
        }
        # Weather-compensated minimum temperature: when the radiator needs a higher
        # supply T to keep up with building heat loss on a cold day, the buffer min
        # should track. Same linear law as the source's heating_curve.
        if "min_temperature_curve" in s and s["min_temperature_curve"]:
            mc = dict(s["min_temperature_curve"])
            for required in ("slope", "offset"):
                if required not in mc:
                    raise ValueError(
                        f"heat_topology.storage[{sid}].min_temperature_curve "
                        f"missing required field '{required}'"
                    )
            tank["min_temperature_curve"] = {
                "slope": float(mc["slope"]),
                "offset": float(mc["offset"]),
                "min_supply": float(mc.get("min_supply", 25.0)),
                "max_supply": float(mc.get("max_supply", 70.0)),
            }
        # Soft comfort constraints: penalize deviations from a target band rather than
        # hard min/max. Accept either scalar `desired_temperature` (broadcast to horizon
        # at solve time) or per-slot `desired_temperatures`. The optimizer's existing
        # thermal_battery soft-constraint code reads these fields directly.
        desired = s.get("desired_temperatures")
        if desired is None and "desired_temperature" in s:
            desired = s["desired_temperature"]
        if desired is not None:
            tank["desired_temperatures"] = (
                list(desired) if isinstance(desired, list | tuple) else float(desired)
            )
        if "overshoot_temperature" in s:
            tank["overshoot_temperature"] = float(s["overshoot_temperature"])
        if "penalty_factor" in s:
            tank["penalty_factor"] = float(s["penalty_factor"])
        if "comfort_sense" in s:
            sense = str(s["comfort_sense"]).lower()
            if sense not in ("heat", "cool"):
                raise ValueError(
                    f"heat_topology.storage[{sid}].comfort_sense='{sense}' must be 'heat' or 'cool'"
                )
            tank["sense"] = sense
        demand = storage_demand.get(sid, {})
        if demand.get("profile") is not None:
            tank["draw_off_demand"] = demand["profile"]
        if demand.get("building"):
            tank.update(demand["building"])
        if demand.get("pool"):
            tank.update(demand["pool"])
        shared_tanks.append(tank)

    # Build deferrable_load_groups from actuator_groups
    def_groups = []
    for gi, g in enumerate(groups):
        names = []
        for flow_pair in g.get("flows", []):
            key = (flow_pair[0], flow_pair[1])
            if key not in flow_to_load_idx:
                raise ValueError(
                    f"heat_topology.actuator_groups[{gi}].flows references unknown flow {flow_pair}"
                )
            names.append(f"deferrable{flow_to_load_idx[key]}")
        def_group = {
            "names": names,
            "mutual_exclusion": bool(g.get("mutual_exclusion", False)),
        }
        if "max_combined_power" in g:
            def_group["max_power"] = float(g["max_combined_power"])
        def_groups.append(def_group)

    return {
        "number_of_deferrable_loads": num_loads,
        "nominal_power_of_deferrable_loads": nominal_power,
        "minimum_power_of_deferrable_loads": min_power,
        "treat_deferrable_load_as_semi_cont": treat_semi_cont,
        "operating_hours_of_each_deferrable_load": operating_hours,
        "set_deferrable_load_single_constant": [False] * num_loads,
        "set_deferrable_startup_penalty": [0.0] * num_loads,
        "deferrable_load_max_cost": [0.0] * num_loads,
        "set_deferrable_max_startups": [0] * num_loads,
        "start_timesteps_of_each_deferrable_load": [0] * num_loads,
        "end_timesteps_of_each_deferrable_load": [0] * num_loads,
        "def_load_config": def_load_config,
        "shared_thermal_tanks": shared_tanks,
        "deferrable_load_groups": def_groups,
        "cost_forecast_per_deferrable_load": cost_per_load,
        "is_electric_load": is_electric_load,
    }


def calculate_thermal_loss_signed(
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    indoor_temperature: float,
    base_loss: float,
) -> np.ndarray:
    r"""
    Calculate signed thermal loss factor based on indoor/outdoor temperature difference.

    **SIGN CONVENTION:**
    - **Positive** (+loss): outdoor < indoor → heat loss, building cools, heating required
    - **Negative** (-loss): outdoor ≥ indoor → heat gain, building warms passively

    Formula: loss * (1 - 2 * Hot(h)), where Hot(h) = 1 if outdoor ≥ indoor, else 0.
    Based on Langer & Volling (2020) Equation B.13.

    :param outdoor_temperature_forecast: Outdoor temperature forecast (°C)
    :type outdoor_temperature_forecast: np.ndarray or pd.Series
    :param indoor_temperature: Indoor/target temperature threshold (°C)
    :type indoor_temperature: float
    :param base_loss: Base thermal loss coefficient in kW
    :type base_loss: float
    :return: Signed loss array (positive = heat loss, negative = heat gain)
    :rtype: np.ndarray

    """
    # Convert to numpy array if pandas Series
    if isinstance(outdoor_temperature_forecast, pd.Series):
        outdoor_temps = outdoor_temperature_forecast.values
    else:
        outdoor_temps = np.asarray(outdoor_temperature_forecast)

    # Create binary hot indicator: 1 if outdoor temp >= indoor temp, 0 otherwise
    hot_indicator = (outdoor_temps >= indoor_temperature).astype(float)

    return base_loss * (1.0 - 2.0 * hot_indicator)


def calculate_heating_demand(
    specific_heating_demand: float,
    floor_area: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    base_temperature: float = 18.0,
    annual_reference_hdd: float = 3000.0,
    optimization_time_step: int | None = None,
) -> np.ndarray:
    """
    Calculate heating demand per timestep based on heating degree days method.

    Uses heating degree days (HDD) to calculate heating demand based on outdoor temperature
    forecast, specific heating demand, and floor area. The specific heating demand should be
    calibrated to the annual reference HDD value.

    :param specific_heating_demand: Specific heating demand in kWh/m²/year (calibrated to annual_reference_hdd)
    :type specific_heating_demand: float
    :param floor_area: Floor area in m²
    :type floor_area: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param base_temperature: Base temperature for HDD calculation in °C, defaults to 18.0 (European standard)
    :type base_temperature: float, optional
    :param annual_reference_hdd: Annual reference HDD value for normalization, defaults to 3000.0 (Central Europe)
    :type annual_reference_hdd: float, optional
    :param optimization_time_step: Optimization time step in minutes. If None, automatically infers from
        pandas Series DatetimeIndex frequency. Falls back to 30 minutes if not inferrable.
    :type optimization_time_step: int | None, optional
    :return: Array of heating demand values (kWh) per timestep
    :rtype: np.ndarray

    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Calculate heating degree days per timestep
    # HDD = max(base_temperature - outdoor_temperature, 0)
    hdd_per_timestep = np.maximum(base_temperature - outdoor_temps, 0.0)

    # Determine timestep duration in hours
    if optimization_time_step is None:
        # Try to infer from pandas Series DatetimeIndex
        if isinstance(outdoor_temperature_forecast, pd.Series) and isinstance(
            outdoor_temperature_forecast.index, pd.DatetimeIndex
        ):
            if len(outdoor_temperature_forecast.index) > 1:
                freq_minutes = (
                    outdoor_temperature_forecast.index[1] - outdoor_temperature_forecast.index[0]
                ).total_seconds() / 60.0
                hours_per_timestep = freq_minutes / 60.0
            else:
                # Single datapoint, fallback to default 30 min
                hours_per_timestep = 0.5
        else:
            # Cannot infer, use default 30 minutes
            hours_per_timestep = 0.5
    else:
        # Convert minutes to hours
        hours_per_timestep = optimization_time_step / 60.0

    # Scale HDD to timestep duration (standard HDD is per 24 hours)
    hdd_per_timestep_scaled = hdd_per_timestep * (hours_per_timestep / 24.0)

    return specific_heating_demand * floor_area * (hdd_per_timestep_scaled / annual_reference_hdd)


def calculate_heating_demand_physics(
    u_value: float,
    envelope_area: float,
    ventilation_rate: float | np.ndarray,
    heated_volume: float,
    indoor_target_temperature: float,
    outdoor_temperature_forecast: np.ndarray | pd.Series,
    optimization_time_step: int,
    solar_irradiance_forecast: np.ndarray | pd.Series | None = None,
    window_area: float | None = None,
    shgc: float = 0.6,
    internal_gains_forecast: np.ndarray | pd.Series | None = None,
    internal_gains_factor: float = 0.0,
    sense: str = "heat",
) -> np.ndarray:
    """
    Calculate heating or cooling demand per timestep based on building physics heat loss model.

    More accurate than HDD method as it directly calculates transmission and ventilation
    losses based on building thermal properties. Optionally accounts for solar gains
    through windows to reduce heating demand.

    :param u_value: Overall thermal transmittance (U-value) in W/(m²·K). Typical values:
        - 0.2-0.3: Well-insulated modern building
        - 0.4-0.6: Average insulation
        - 0.8-1.2: Poor insulation / old building
    :type u_value: float
    :param envelope_area: Total building envelope area (walls + roof + floor + windows) in m²
    :type envelope_area: float
    :param ventilation_rate: Air changes per hour (ACH). Typical values:
        - 0.3-0.5: Well-sealed modern building with controlled ventilation
        - 0.5-1.0: Average building
        - 1.0-2.0: Leaky old building
        Either a single scalar (applied to every timestep) or a per-timestep
        numpy array the same length as outdoor_temperature_forecast (e.g. to
        model a temporary spike from an open window/door at specific
        timesteps) - the ventilation-loss term is plain elementwise
        multiplication against the per-timestep temperature delta, so both
        shapes work identically with no other change to this function.
    :type ventilation_rate: float | np.ndarray
    :param heated_volume: Total heated volume in m³
    :type heated_volume: float
    :param indoor_target_temperature: Target indoor temperature in °C
    :type indoor_target_temperature: float
    :param outdoor_temperature_forecast: Outdoor temperature forecast in °C for each timestep
    :type outdoor_temperature_forecast: np.ndarray | pd.Series
    :param optimization_time_step: Optimization time step in minutes
    :type optimization_time_step: int
    :param solar_irradiance_forecast: Global Horizontal Irradiance (GHI) in W/m² for each timestep.
        If provided along with window_area, solar gains will be subtracted from heating demand.
    :type solar_irradiance_forecast: np.ndarray | pd.Series | None, optional
    :param window_area: Total window area in m². If provided along with solar_irradiance_forecast,
        solar gains will reduce heating demand. Typical values: 15-25% of floor area.
    :type window_area: float | None, optional
    :param shgc: Solar Heat Gain Coefficient (dimensionless, 0-1). Fraction of solar radiation
        that becomes heat inside the building. Typical values:
        - 0.5-0.6: Modern low-e double-glazed windows
        - 0.6-0.7: Standard double-glazed windows
        - 0.7-0.8: Single-glazed windows
        Default: 0.6
    :type shgc: float, optional
    :param internal_gains_forecast: Electrical load power forecast in W for each timestep.
        If provided along with internal_gains_factor > 0, internal gains from electrical
        appliances will be subtracted from heating demand.
    :type internal_gains_forecast: np.ndarray | pd.Series | None, optional
    :param internal_gains_factor: Factor (0-1) representing what fraction of electrical load
        becomes useful internal heat gains. Typical values:
        - 0.0: No internal gains considered (default, backwards compatible)
        - 0.5-0.7: Conservative estimate (some heat lost to ventilation/drains)
        - 0.8-0.9: Most electrical energy becomes heat (well-insulated building)
        - 1.0: All electrical energy becomes internal heat (theoretical maximum)
        Default: 0.0
    :type internal_gains_factor: float, optional
    :param sense: Thermal mode, ``"heat"`` (default) or ``"cool"``. In heating mode the
        result is the positive heat the building loses (and must be replaced) when it is
        colder outside than the target, with solar and internal gains subtracted. In
        cooling mode the result is the heat the building gains when it is hotter outside
        than the target, with solar and internal gains added, returned as a SIGNED
        negative value (heat gain) to match the sign convention of
        :func:`calculate_thermal_loss_signed`. Default ``"heat"`` is byte-identical to the
        previous behaviour.
    :type sense: str, optional
    :return: Array of demand values (kWh) per timestep. Positive for heating (heat to
        supply), zero or negative for cooling (heat gain to remove).
    :rtype: np.ndarray

    Example:
        >>> outdoor_temps = np.array([5, 8, 12, 15])
        >>> ghi = np.array([0, 100, 400, 600])  # W/m²
        >>> demand = calculate_heating_demand_physics(
        ...     u_value=0.3,
        ...     envelope_area=400,
        ...     ventilation_rate=0.5,
        ...     heated_volume=250,
        ...     indoor_target_temperature=20,
        ...     outdoor_temperature_forecast=outdoor_temps,
        ...     optimization_time_step=30,
        ...     solar_irradiance_forecast=ghi,
        ...     window_area=50,
        ...     shgc=0.6
        ... )
    """

    # Convert outdoor temperature forecast to numpy array if pandas Series
    outdoor_temps = (
        outdoor_temperature_forecast.values
        if isinstance(outdoor_temperature_forecast, pd.Series)
        else np.asarray(outdoor_temperature_forecast)
    )

    # Normalize the thermal mode. Heating models the heat the building loses when
    # it is colder outside than the target; cooling models the heat it gains when it
    # is hotter outside. gain_sign folds the direction of solar/internal gains: in
    # heating they REDUCE the load (free heat), in cooling they ADD to it.
    sense = normalize_heat_cool_mode(sense, field_name="sense", context="thermal demand")
    gain_sign = -1.0 if sense == "cool" else 1.0

    # Temperature difference driving the envelope load (magnitude, per timestep).
    # Heating: load when outdoor < indoor. Cooling: load when outdoor > indoor.
    if sense == "cool":
        temp_diff = outdoor_temps - indoor_target_temperature
    else:
        temp_diff = indoor_target_temperature - outdoor_temps
    temp_diff = np.maximum(temp_diff, 0.0)

    # Transmission losses: Q_trans = U * A * ΔT (W to kW)
    transmission_loss_kw = u_value * envelope_area * temp_diff / 1000.0

    # Ventilation losses: Q_vent = V * ρ * c * n * ΔT / 3600
    # ρ = air density (kg/m³), c = specific heat capacity (kJ/(kg·K)), n = ACH
    air_density = 1.2  # kg/m³ at 20°C
    air_heat_capacity = 1.005  # kJ/(kg·K)
    ventilation_loss_kw = (
        ventilation_rate * heated_volume * air_density * air_heat_capacity * temp_diff / 3600.0
    )

    # Total heat loss in kW
    total_loss_kw = transmission_loss_kw + ventilation_loss_kw

    # Calculate solar gains if irradiance and window area are provided
    if solar_irradiance_forecast is not None and window_area is not None:
        # Convert solar irradiance to numpy array if pandas Series
        solar_irradiance = (
            solar_irradiance_forecast.values
            if isinstance(solar_irradiance_forecast, pd.Series)
            else np.asarray(solar_irradiance_forecast)
        )

        # Solar gains: Q_solar = window_area * SHGC * GHI (W to kW)
        # GHI is in W/m², so multiply by window_area (m²) gives W, then divide by 1000 for kW
        solar_gains_kw = window_area * shgc * solar_irradiance / 1000.0

        # Heating: subtract solar gains from heat loss (but never go negative).
        # Cooling: gain_sign flips this so solar gains ADD to the cooling load, which
        # is left unfloored because gains only ever deepen it.
        total_loss_kw = total_loss_kw - gain_sign * solar_gains_kw
        if sense != "cool":
            total_loss_kw = np.maximum(total_loss_kw, 0.0)

    # Validate internal_gains_factor is in expected range [0, 1]
    if internal_gains_factor < 0 or internal_gains_factor > 1:
        raise ValueError(
            f"internal_gains_factor must be between 0 and 1, got {internal_gains_factor}"
        )

    # Calculate internal gains from electrical load if provided and applicable
    if internal_gains_forecast is not None and internal_gains_factor > 0:
        # Convert internal gains forecast to numpy array and normalize to 1D
        # to align with other forecast inputs and avoid broadcast surprises
        internal_gains = (
            internal_gains_forecast.values
            if isinstance(internal_gains_forecast, pd.Series)
            else internal_gains_forecast
        )
        internal_gains = np.asarray(internal_gains).reshape(-1)

        # Validate that internal gains forecast length matches outdoor temperature forecast
        if len(internal_gains) != len(outdoor_temps):
            raise ValueError(
                f"internal_gains_forecast length ({len(internal_gains)}) must match "
                f"outdoor_temperature_forecast length ({len(outdoor_temps)})"
            )

        # Warn if values seem like they might be in kW instead of W
        # Typical household load is 100-10000W; values below 10 suggest kW was passed
        max_load = np.max(internal_gains)
        if max_load > 0 and max_load < 10:
            import warnings

            warnings.warn(
                f"internal_gains_forecast max value ({max_load:.2f}) is very low. "
                "Expected values in W (e.g., 500-5000), but received values that "
                "look like kW. Please ensure you're passing Watts, not kilowatts.",
                UserWarning,
                stacklevel=2,
            )

        # Internal gains: Q_internal = load_power * factor
        # load_power is in W, convert to kW; factor is dimensionless (0-1)
        internal_gains_kw = internal_gains * internal_gains_factor / W_TO_KW

        # Heating: subtract internal gains from heat loss (never go negative).
        # Cooling: gain_sign flips this so internal gains ADD to the cooling load.
        total_loss_kw = total_loss_kw - gain_sign * internal_gains_kw
        if sense != "cool":
            total_loss_kw = np.maximum(total_loss_kw, 0.0)

    # Convert to kWh for the timestep. Cooling demand is returned as a signed heat
    # gain (<= 0) so the shared thermal state equation (which subtracts the demand
    # term) raises indoor temperature when it is hot, matching calculate_thermal_loss_signed.
    hours_per_timestep = optimization_time_step / 60.0
    demand = total_loss_kw * hours_per_timestep
    if sense == "cool":
        demand = -demand
    return demand


def calculate_shaded_window_irradiance(
    dni: np.ndarray | pd.Series,
    dhi: np.ndarray | pd.Series,
    blind_position: float,
    blind_type: str,
    solar_elevation_deg: np.ndarray | pd.Series | None = None,
    awning_elevation_low_deg: float = 20.0,
    awning_elevation_high_deg: float = 45.0,
) -> np.ndarray:
    """
    Decompose solar irradiance into direct (DNI) and diffuse (DHI) components
    and apply a room's sun-shading device to the direct component only - the
    diffuse component (arriving from the whole sky dome) is never attenuated
    by either shading type modelled here.

    Two shading types, matching how they physically behave:
      - "screen": sits flush on the window glass. Blocks a fraction of the
        direct component equal to its own position (0=open, 1=fully closed),
        regardless of the sun's angle.
      - "awning" (e.g. a "knikarmscherm"): projects outward above the
        window. Only blocks direct sun once the sun is high enough in the
        sky - negligible effect at low solar elevation (early morning, late
        evening, winter), full effect above ``awning_elevation_high_deg``,
        linearly ramping in between. Requires ``solar_elevation_deg``; when
        not supplied, degrades to zero shading effect (the safe/conservative
        default direction used throughout this codebase when an optional
        signal isn't wired) rather than raising.
      - "none" (or unset): no shading applied at all.

    The result is meant to replace a raw GHI reading as the
    ``solar_irradiance_forecast`` input to :func:`calculate_heating_demand_physics`
    - that function's own signature/contract is unchanged by this addition.

    :param dni: Direct Normal Irradiance forecast, W/m² per timestep.
    :param dhi: Diffuse Horizontal Irradiance forecast, W/m² per timestep.
        Must be the same length as ``dni``.
    :param blind_position: Current shading position, 0 (fully open, no
        shading) to 1 (fully closed, maximum shading).
    :param blind_type: One of "none", "screen", "awning".
    :param solar_elevation_deg: Solar elevation angle forecast, degrees.
        Only used when ``blind_type == "awning"``. Must be the same length
        as ``dni`` when supplied.
    :param awning_elevation_low_deg: Elevation below which an awning has no
        shading effect at all.
    :param awning_elevation_high_deg: Elevation above which an awning is at
        full shading effectiveness.
    :return: Effective irradiance array (W/m²), same length as ``dni``.
    """
    dni_arr = np.asarray(dni, dtype=float)
    dhi_arr = np.asarray(dhi, dtype=float)
    if dni_arr.shape != dhi_arr.shape:
        raise ValueError(
            f"calculate_shaded_window_irradiance: dni (length {dni_arr.shape}) and "
            f"dhi (length {dhi_arr.shape}) must be the same length"
        )
    if not (0.0 <= blind_position <= 1.0):
        raise ValueError(
            f"calculate_shaded_window_irradiance: blind_position must be in [0, 1], "
            f"got {blind_position}"
        )
    blind_type_norm = str(blind_type or "none").strip().lower()

    if blind_type_norm == "none":
        direct_block_fraction = 0.0
    elif blind_type_norm == "screen":
        direct_block_fraction = blind_position
    elif blind_type_norm == "awning":
        if solar_elevation_deg is None:
            direct_block_fraction = 0.0
        else:
            elev_arr = np.asarray(solar_elevation_deg, dtype=float)
            if elev_arr.shape != dni_arr.shape:
                raise ValueError(
                    f"calculate_shaded_window_irradiance: solar_elevation_deg (length "
                    f"{elev_arr.shape}) must match dni/dhi (length {dni_arr.shape})"
                )
            if awning_elevation_high_deg <= awning_elevation_low_deg:
                raise ValueError(
                    "calculate_shaded_window_irradiance: awning_elevation_high_deg must "
                    "be > awning_elevation_low_deg"
                )
            elevation_factor = np.clip(
                (elev_arr - awning_elevation_low_deg)
                / (awning_elevation_high_deg - awning_elevation_low_deg),
                0.0,
                1.0,
            )
            direct_block_fraction = blind_position * elevation_factor
    else:
        raise ValueError(
            f"calculate_shaded_window_irradiance: unrecognized blind_type '{blind_type}' "
            "(expected 'none', 'screen', or 'awning')"
        )

    return dni_arr * (1.0 - direct_block_fraction) + dhi_arr


def simulate_physics_room_temperature_trajectory(
    initial_temp: float,
    duty: np.ndarray,
    outdoor_temp: np.ndarray,
    nominal_power_w: float,
    dt_hours: float,
    density: float = 2400.0,
    heat_capacity: float = 0.88,
    volume: float = 15.0,
    base_loss: float = 0.045,
    supply_temperature: float = 35.0,
    carnot_efficiency: float = 0.4,
    heating_demand_kwh: np.ndarray | None = None,
    sense: str = "heat",
) -> np.ndarray:
    """Open-loop, recursive simulation of the SAME physics/RC recurrence
    optimization.py::_add_thermal_battery_constraints uses for real
    dispatch (no-thermal-inertia form, the only one
    utils.py::_append_room_thermal_loads's own auto-generated rooms ever
    use - they never set thermal_inertia_time_constant):

        T[t+1] = T[t] + conversion * (sense_coeff * COP[t]*P[t]/1000*dt_hours
                                       - heating_demand_kwh[t] - thermal_loss[t])

    Given a KNOWN duty trajectory (not solved for - this simulates forward
    from history, it never touches the optimizer), used by
    command_line.py::refit_arx_model to score "what would
    the physics/simple model have predicted for this room" over the same
    holdout window the self-learning fit is scored on, so a room's fitted
    dispatch coefficients are only ever deployed where they're genuinely
    more accurate than the physics fallback - not just individually
    "good enough" against an arbitrary absolute threshold.

    ``duty`` and ``outdoor_temp`` must be the SAME arrays (same length,
    same alignment) used to score the self-learning model, for a fair
    apples-to-apples comparison. ``heating_demand_kwh`` defaults to zero
    for every timestep, matching heatpump_model_family="simple" (the
    default) - _append_room_thermal_loads never gives its auto-generated
    rooms an ongoing-loss demand model unless "physics" family is
    explicitly configured, in which case the caller should precompute it
    via calculate_heating_demand_physics and pass it in here.

    :return: Simulated temperature trajectory, same length as duty/outdoor_temp.
    """
    n = len(duty)
    if density <= 0 or heat_capacity <= 0 or volume <= 0:
        raise ValueError(
            f"simulate_physics_room_temperature_trajectory requires positive density "
            f"({density}), heat_capacity ({heat_capacity}), and volume ({volume})"
        )
    conversion = 3600.0 / (density * heat_capacity * volume)
    sense = normalize_heat_cool_mode(
        sense, field_name="sense", context="simulate_physics_room_temperature_trajectory"
    )
    sense_coeff = 1.0 if sense == "heat" else -1.0

    cop_hc = {"supply_temperature": supply_temperature, "carnot_efficiency": carnot_efficiency, "sense": sense}
    cops = resolve_thermal_battery_cop(cop_hc, outdoor_temp, length=n)
    thermal_losses = calculate_thermal_loss_signed(
        outdoor_temperature_forecast=outdoor_temp, indoor_temperature=initial_temp, base_loss=base_loss
    )
    demand = (
        np.asarray(heating_demand_kwh, dtype=float) if heating_demand_kwh is not None else np.zeros(n)
    )
    power_w = np.asarray(duty, dtype=float) * nominal_power_w

    temp = np.zeros(n)
    temp[0] = initial_temp
    for t in range(n - 1):
        heat_in_kwh = cops[t] * power_w[t] / 1000.0 * dt_hours
        temp[t + 1] = temp[t] + conversion * (sense_coeff * heat_in_kwh - demand[t] - thermal_losses[t])
    return temp


def update_params_with_ha_config(
    params: str,
    ha_config: dict,
) -> dict:
    """
    Update the params with the Home Assistant configuration.

    Parameters
    ----------
    params : str
        The serialized params.
    ha_config : dict
        The Home Assistant configuration.

    Returns
    -------
    dict
        The updated params.
    """
    # Load serialized params
    params = orjson.loads(params)
    # Update params
    currency_to_symbol = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "YEN": "¥",
        "JPY": "¥",
        "AUD": "A$",
        "CAD": "C$",
        "CHF": "CHF",  # Swiss Franc has no special symbol
        "CNY": "¥",
        "INR": "₹",
        "CZK": "Kč",
        "BGN": "лв",
        "DKK": "kr",
        "HUF": "Ft",
        "PLN": "zł",
        "RON": "Leu",
        "SEK": "kr",
        "TRY": "Lira",
        "VEF": "Bolivar",
        "VND": "Dong",
        "THB": "Baht",
        "SGD": "S$",
        "IDR": "Roepia",
        "ZAR": "Rand",
        # Add more as needed
    }
    if "currency" in ha_config.keys():
        ha_config["currency"] = currency_to_symbol.get(ha_config["currency"], "Unknown")
    else:
        ha_config["currency"] = "€"

    updated_passed_dict = {
        "custom_cost_fun_id": {
            "unit_of_measurement": ha_config["currency"],
        },
        "custom_unit_load_cost_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
        "custom_unit_prod_price_id": {
            "unit_of_measurement": f"{ha_config['currency']}/kWh",
        },
    }
    for key, value in updated_passed_dict.items():
        params["passed_data"][key]["unit_of_measurement"] = value["unit_of_measurement"]
    # Serialize the final params
    params = orjson.dumps(params, default=str).decode("utf-8")
    return params


async def treat_runtimeparams(
    runtimeparams: str,
    params: dict[str, dict],
    retrieve_hass_conf: dict[str, str],
    optim_conf: dict[str, str],
    plant_conf: dict[str, str],
    set_type: str,
    logger: logging.Logger,
    emhass_conf: dict[str, pathlib.Path],
) -> tuple[str, dict[str, dict]]:
    """
    Treat the passed optimization runtime parameters.

    :param runtimeparams: Json string containing the runtime parameters dict.
    :type runtimeparams: str
    :param params: Built configuration parameters
    :type params: str
    :param retrieve_hass_conf: Config dictionary for data retrieving parameters.
    :type retrieve_hass_conf: dict
    :param optim_conf: Config dictionary for optimization parameters.
    :type optim_conf: dict
    :param plant_conf: Config dictionary for technical plant parameters.
    :type plant_conf: dict
    :param set_type: The type of action to be performed.
    :type set_type: str
    :param logger: The logger object.
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :return: Returning the params and optimization parameter container.
    :rtype: Tuple[str, dict]

    """
    # Check if passed params is a dict
    if (params is not None) and (params != "null"):
        if type(params) is str:
            params = orjson.loads(params)
    else:
        params = {}

    # Merge current config categories to params
    params["retrieve_hass_conf"].update(retrieve_hass_conf)
    params["optim_conf"].update(optim_conf)
    params["plant_conf"].update(plant_conf)

    # Check defaults on HA retrieved config
    default_currency_unit = "€"
    default_temperature_unit = "°C"

    # Some default data needed
    custom_deferrable_forecast_id = []
    custom_deferrable_state_id = []
    custom_predicted_temperature_id = []
    custom_heating_demand_id = []
    configured_load_names = list(params["optim_conf"].get("load_names", []))
    for k in range(params["optim_conf"]["number_of_deferrable_loads"]):
        raw_name = ""
        if k < len(configured_load_names):
            raw_name = str(configured_load_names[k]).strip()
        if not raw_name:
            raw_name = f"appliance_{k + 1}"
        slug_name = re.sub(r"[^a-z0-9_]+", "_", raw_name.lower()).strip("_")
        if not slug_name:
            slug_name = f"appliance_{k + 1}"

        custom_deferrable_forecast_id.append(
            {
                "entity_id": f"sensor.p_{slug_name}",
                "device_class": "power",
                "unit_of_measurement": "W",
                "friendly_name": raw_name.replace("_", " ").title(),
            }
        )
        custom_deferrable_state_id.append(
            {
                "entity_id": f"sensor.p_deferrable{k}_state",
                "device_class": "enum",
                "unit_of_measurement": "",
                "friendly_name": f"Deferrable Load {k} Command",
            }
        )
        custom_predicted_temperature_id.append(
            {
                "entity_id": f"sensor.temp_predicted{k}",
                "device_class": "temperature",
                "unit_of_measurement": default_temperature_unit,
                "friendly_name": f"Predicted temperature {k}",
            }
        )
        custom_heating_demand_id.append(
            {
                "entity_id": f"sensor.heating_demand{k}",
                "device_class": "energy",
                "unit_of_measurement": "kWh",
                "friendly_name": f"Heating demand {k}",
            }
        )
    default_passed_dict = {
        "custom_pv_forecast_id": {
            "entity_id": "sensor.p_pv_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Forecast",
        },
        "custom_load_forecast_id": {
            "entity_id": "sensor.p_load_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Load Power Forecast",
        },
        "custom_pv_curtailment_id": {
            "entity_id": "sensor.p_pv_curtailment",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Power Curtailment",
        },
        "custom_hybrid_inverter_id": {
            "entity_id": "sensor.p_hybrid_inverter",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "PV Hybrid Inverter",
        },
        "custom_batt_forecast_id": {
            "entity_id": "sensor.p_batt_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Battery Power Forecast",
        },
        "custom_batt_soc_forecast_id": {
            "entity_id": "sensor.soc_batt_forecast",
            "device_class": "battery",
            "unit_of_measurement": "%",
            "friendly_name": "Battery SOC Forecast",
        },
        "custom_grid_forecast_id": {
            "entity_id": "sensor.p_grid_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Grid Power Forecast",
        },
        "custom_cost_fun_id": {
            "entity_id": "sensor.total_cost_fun_value",
            "device_class": "monetary",
            "unit_of_measurement": default_currency_unit,
            "friendly_name": "Total cost function value",
        },
        "custom_optim_status_id": {
            "entity_id": "sensor.optim_status",
            "device_class": "",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        },
        "custom_unit_load_cost_id": {
            "entity_id": "sensor.unit_load_cost",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Load Cost",
        },
        "custom_unit_prod_price_id": {
            "entity_id": "sensor.unit_prod_price",
            "device_class": "monetary",
            "unit_of_measurement": f"{default_currency_unit}/kWh",
            "friendly_name": "Unit Prod Price",
        },
        "custom_deferrable_forecast_id": custom_deferrable_forecast_id,
        "custom_deferrable_state_id": custom_deferrable_state_id,
        "custom_predicted_temperature_id": custom_predicted_temperature_id,
        "custom_heating_demand_id": custom_heating_demand_id,
        "publish_prefix": "",
    }
    if "passed_data" in params.keys():
        for key, value in default_passed_dict.items():
            params["passed_data"][key] = value
    else:
        params["passed_data"] = default_passed_dict

    # Capture defaults for power limits before association loop
    power_limit_defaults = {
        "maximum_power_from_grid": params["plant_conf"].get("maximum_power_from_grid"),
        "maximum_power_to_grid": params["plant_conf"].get("maximum_power_to_grid"),
    }

    # If any runtime parameters where passed in action call
    if runtimeparams is not None:
        if type(runtimeparams) is str:
            runtimeparams = orjson.loads(runtimeparams)

        # Remember which per-battery array params were configured with genuinely
        # distinct per-battery entries: the association loop below overwrites the
        # configured value with the raw runtime one, and the re-normalisation
        # step further down warns when a runtime scalar masks such a list
        # (its broadcast silently flattens the per-battery values, #1032).
        batt_distinct_config_lists: dict[str, list] = {}
        for batt_conf_key, batt_param_names in (
            ("plant_conf", BATT_ARRAY_PARAMS_PLANT_CONF),
            ("optim_conf", BATT_ARRAY_PARAMS_OPTIM_CONF),
        ):
            for batt_param_name in batt_param_names:
                configured = params[batt_conf_key].get(batt_param_name)
                if isinstance(configured, list) and any(
                    element != configured[0] for element in configured[1:]
                ):
                    batt_distinct_config_lists[batt_param_name] = list(configured)

        # Loop though parameters stored in association file, Check to see if any stored in runtime
        # If true, set runtime parameter to params
        if emhass_conf["associations_path"].exists():
            async with aiofiles.open(emhass_conf["associations_path"]) as data:
                content = await data.read()
                associations = list(csv.reader(content.splitlines(), delimiter=","))
                # Association file key reference
                # association[0] = config categories
                # association[1] = legacy parameter name
                # association[2] = parameter (config.json/config_defaults.json)
                # association[3] = parameter list name if exists (not used, from legacy options.json)
                for association in associations:
                    # Check parameter name exists in runtime
                    if runtimeparams.get(association[2], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[2]]
                    # Check Legacy parameter name runtime
                    elif runtimeparams.get(association[1], None) is not None:
                        params[association[0]][association[2]] = runtimeparams[association[1]]
        else:
            logger.warning(
                "Cant find associations file (associations.csv) in: "
                + str(emhass_conf["associations_path"])
            )

        # Special handling for power limit parameters - they can be vectors (Tier 1a)
        def _parse_power_limit(key: str) -> None:
            """Helper to parse list/scalar power limits safely."""
            if key in runtimeparams:
                value = runtimeparams[key]
                try:
                    # If it's a string representation of a list, parse it
                    if isinstance(value, str):
                        parsed = ast.literal_eval(value)
                        params["plant_conf"][key] = parsed
                    # If already a list/array, use it directly
                    # Ruff preferred bitwise OR '|' for union types
                    elif isinstance(value, list | tuple):
                        params["plant_conf"][key] = list(value)
                    # If scalar, use as-is
                    else:
                        params["plant_conf"][key] = value
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse {key}: {e}. Using default.")
                    if power_limit_defaults.get(key) is not None:
                        params["plant_conf"][key] = power_limit_defaults[key]

        # Apply the helper
        _parse_power_limit("maximum_power_from_grid")
        _parse_power_limit("maximum_power_to_grid")

        # Re-normalise per-battery array params after the generic association
        # loop above may have overwritten them with a raw runtime value (#610).
        # N=1 is a no-op (see check_batt_params); this only does real work once
        # number_of_batteries > 1.
        num_batteries = validate_num_batteries(params["plant_conf"])
        for batt_param_name, batt_default in BATT_ARRAY_PARAMS_PLANT_CONF.items():
            _warn_if_runtime_scalar_masks_batt_list(
                num_batteries,
                params["plant_conf"],
                batt_param_name,
                batt_distinct_config_lists,
                logger,
            )
            check_batt_params(
                num_batteries, params["plant_conf"], batt_default, batt_param_name, logger
            )
        for batt_param_name, batt_default in BATT_ARRAY_PARAMS_OPTIM_CONF.items():
            _warn_if_runtime_scalar_masks_batt_list(
                num_batteries,
                params["optim_conf"],
                batt_param_name,
                batt_distinct_config_lists,
                logger,
            )
            check_batt_params(
                num_batteries, params["optim_conf"], batt_default, batt_param_name, logger
            )
        for batt_param_name in BATT_WEIGHT_PARAMS:
            check_batt_weight_params(num_batteries, params["optim_conf"], batt_param_name, logger)

        # Re-normalise the per-phase array params the same way, after any
        # runtime override of number_of_phases/number_of_batteries.
        num_phases = validate_num_phases(params["plant_conf"], logger)
        check_phase_array_params(num_batteries, params["plant_conf"], "", "battery_phase", logger)
        check_phase_array_params(
            num_phases, params["plant_conf"], 4000, "maximum_power_from_grid_per_phase", logger
        )
        check_phase_array_params(
            num_phases, params["plant_conf"], 4000, "maximum_power_to_grid_per_phase", logger
        )

        # Generate forecast_dates
        # Force update optimization_time_step if present in runtimeparams
        if "optimization_time_step" in runtimeparams:
            optimization_time_step = int(runtimeparams["optimization_time_step"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        elif "freq" in runtimeparams:
            optimization_time_step = int(runtimeparams["freq"])
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                optimization_time_step, "minutes"
            )
        else:
            optimization_time_step = int(
                params["retrieve_hass_conf"]["optimization_time_step"].seconds / 60.0
            )

        if (
            runtimeparams.get("delta_forecast_daily", None) is not None
            or runtimeparams.get("delta_forecast", None) is not None
        ):
            # Use old param name delta_forecast (if provided) for backwards compatibility
            delta_forecast = runtimeparams.get("delta_forecast", None)
            # Prefer new param name delta_forecast_daily
            delta_forecast = runtimeparams.get("delta_forecast_daily", delta_forecast)
            # Ensure delta_forecast is numeric and at least 1 day
            if delta_forecast is None:
                logger.warning("delta_forecast_daily is missing so defaulting to 1 day")
                delta_forecast = 1
            else:
                try:
                    delta_forecast = int(delta_forecast)
                except ValueError:
                    logger.warning(
                        "Invalid delta_forecast_daily value (%s) so defaulting to 1 day",
                        delta_forecast,
                    )
                    delta_forecast = 1
            if delta_forecast <= 0:
                logger.warning(
                    "delta_forecast_daily is too low (%s) so defaulting to 1 day",
                    delta_forecast,
                )
                delta_forecast = 1
            params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(days=delta_forecast)
        else:
            delta_forecast = int(params["optim_conf"]["delta_forecast_daily"].days)
        if runtimeparams.get("time_zone", None) is not None:
            time_zone = pytz.timezone(params["retrieve_hass_conf"]["time_zone"])
            params["retrieve_hass_conf"]["time_zone"] = time_zone
        else:
            time_zone = params["retrieve_hass_conf"]["time_zone"]

        forecast_dates = get_forecast_dates(optimization_time_step, delta_forecast, time_zone)

        # Add runtime exclusive (not in config) parameters to params
        # regressor-model-fit
        if set_type == "regressor-model-fit":
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target
            if "timestamp" not in runtimeparams:
                params["passed_data"]["timestamp"] = None
            else:
                timestamp = runtimeparams["timestamp"]
                params["passed_data"]["timestamp"] = timestamp
            if "date_features" not in runtimeparams:
                params["passed_data"]["date_features"] = []
            else:
                date_features = runtimeparams["date_features"]
                params["passed_data"]["date_features"] = date_features

        # regressor-model-predict
        if set_type == "regressor-model-predict":
            if "new_values" in runtimeparams:
                new_values = runtimeparams["new_values"]
                params["passed_data"]["new_values"] = new_values
            if "csv_file" in runtimeparams:
                csv_file = runtimeparams["csv_file"]
                params["passed_data"]["csv_file"] = csv_file
            if "features" in runtimeparams:
                features = runtimeparams["features"]
                params["passed_data"]["features"] = features
            if "target" in runtimeparams:
                target = runtimeparams["target"]
                params["passed_data"]["target"] = target

        # export-influxdb-to-csv
        if set_type == "export-influxdb-to-csv":
            # Use dictionary comprehension to simplify parameter assignment
            export_keys = {
                k: runtimeparams[k]
                for k in (
                    "sensor_list",
                    "csv_filename",
                    "start_time",
                    "end_time",
                    "resample_freq",
                    "timestamp_col_name",
                    "decimal_places",
                    "handle_nan",
                )
                if k in runtimeparams
            }
            params["passed_data"].update(export_keys)

        # thermal-two-stage-plan
        if set_type == "thermal-two-stage-plan":
            thermal_keys = (
                "thermal_data_csv_path",
                "thermal_model_dir",
                "thermal_timestamp_col",
                "thermal_target_col",
                "thermal_outdoor_col",
                "thermal_price_col",
                "thermal_gas_price_forecast_method",
                "thermal_gas_price",
                "thermal_gas_price_col",
                "thermal_target_room_temp_min",
                "thermal_target_room_temp_max",
                "thermal_price_weight",
                "thermal_comfort_weight",
                "thermal_energy_weight",
                "thermal_horizon",
                "thermal_top_k",
                "thermal_coarse_models",
                "thermal_fine_models",
                "thermal_latitude",
                "thermal_longitude",
            )
            for key in thermal_keys:
                if key in runtimeparams:
                    params["passed_data"][key] = runtimeparams[key]

        # MPC control case
        if set_type == "naive-mpc-optim":
            if (
                "prediction_horizon" not in runtimeparams.keys()
                or runtimeparams["prediction_horizon"] is None
            ):
                prediction_horizon = 10  # 10 time steps by default
            else:
                prediction_horizon = runtimeparams["prediction_horizon"]
            params["passed_data"]["prediction_horizon"] = prediction_horizon
            # Auto-extend the forecast window to cover the requested MPC horizon.
            # forecast_dates was sized from delta_forecast_daily (default 1 day) above,
            # before prediction_horizon was known; a longer horizon would otherwise be
            # silently truncated by the [0:prediction_horizon] slice below, and the
            # weather/Solcast fetch (which scales its request to the window length)
            # would only pull 1 day. So grow the window to fit and rebuild it once.
            steps_per_day = int(24 * 60 / optimization_time_step)
            required_delta_forecast = max(
                delta_forecast,
                -(-int(prediction_horizon) // steps_per_day),  # ceil division
            )
            if required_delta_forecast > delta_forecast:
                logger.info(
                    "naive-mpc prediction_horizon=%s exceeds the %s-day forecast window; "
                    "extending delta_forecast_daily to %s day(s) to cover the full horizon "
                    "and its weather forecast.",
                    prediction_horizon,
                    delta_forecast,
                    required_delta_forecast,
                )
                delta_forecast = required_delta_forecast
                params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(days=delta_forecast)
                forecast_dates = get_forecast_dates(
                    optimization_time_step, delta_forecast, time_zone
                )
            num_batteries = validate_num_batteries(params["plant_conf"])
            if num_batteries == 1:
                # Unchanged from before #610: soc_init/soc_final stay plain
                # floats, not [x] lists, so downstream consumers written for
                # the single-battery model (and existing tests pinning this
                # exact shape) are untouched.
                if "soc_init" not in runtimeparams.keys():
                    soc_init = params["plant_conf"]["battery_target_state_of_charge"]
                else:
                    soc_init = runtimeparams["soc_init"]
                    if isinstance(soc_init, list):
                        # A naive-mpc runtime list is handled symmetrically
                        # with the num_batteries>1 branch below
                        # (_resolve_soc_runtime_list) and with the dayahead
                        # passthrough (_passthrough_soc_runtime): a length-1
                        # list unwraps to its scalar, any other length is the
                        # same clear ValueError the N>1 path raises, not an
                        # opaque TypeError from comparing a list to a float.
                        if len(soc_init) != 1:
                            raise ValueError(
                                f"soc_init has {len(soc_init)} entries but "
                                f"number_of_batteries=1; provide a scalar (broadcast "
                                f"to every battery) or a list of exactly 1 entry"
                            )
                        soc_init = soc_init[0]
                if soc_init < params["plant_conf"]["battery_minimum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_init={soc_init} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, keeping real initial SOC for optimization recovery"
                    )
                if soc_init > params["plant_conf"]["battery_maximum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_init={soc_init} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, keeping real initial SOC for optimization recovery"
                    )
                params["passed_data"]["soc_init"] = soc_init
                if "soc_final" not in runtimeparams.keys():
                    soc_final = params["plant_conf"]["battery_target_state_of_charge"]
                else:
                    soc_final = runtimeparams["soc_final"]
                    if isinstance(soc_final, list):
                        # Same symmetric length-1-unwrap / clear-ValueError
                        # handling as soc_init above.
                        if len(soc_final) != 1:
                            raise ValueError(
                                f"soc_final has {len(soc_final)} entries but "
                                f"number_of_batteries=1; provide a scalar (broadcast "
                                f"to every battery) or a list of exactly 1 entry"
                            )
                        soc_final = soc_final[0]
                if soc_final < params["plant_conf"]["battery_minimum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_final={soc_final} is lower than soc_min={params['plant_conf']['battery_minimum_state_of_charge']}, setting soc_final=soc_min"
                    )
                    soc_final = params["plant_conf"]["battery_minimum_state_of_charge"]
                if soc_final > params["plant_conf"]["battery_maximum_state_of_charge"]:
                    logger.warning(
                        f"Passed soc_final={soc_final} is greater than soc_max={params['plant_conf']['battery_maximum_state_of_charge']}, setting soc_final=soc_max"
                    )
                    soc_final = params["plant_conf"]["battery_maximum_state_of_charge"]
                params["passed_data"]["soc_final"] = soc_final
            else:
                # #610: battery_target_state_of_charge/min/max are now
                # per-battery lists (check_batt_params already normalised them
                # above). soc_init/soc_final resolve PER BATTERY k against
                # target[k]; a runtime scalar broadcasts, a runtime list must
                # be exactly num_batteries long (hard error otherwise).
                target_list = params["plant_conf"]["battery_target_state_of_charge"]
                min_list = params["plant_conf"]["battery_minimum_state_of_charge"]
                max_list = params["plant_conf"]["battery_maximum_state_of_charge"]

                def _resolve_soc_runtime_list(key: str, fallback_list: list) -> list:
                    if key not in runtimeparams.keys():
                        return list(fallback_list)
                    value = runtimeparams[key]
                    if isinstance(value, list):
                        if len(value) != num_batteries:
                            raise ValueError(
                                f"{key} has {len(value)} entries but "
                                f"number_of_batteries={num_batteries}; provide a scalar "
                                f"(broadcast to every battery) or a list of exactly "
                                f"{num_batteries} entries"
                            )
                        return list(value)
                    return [value] * num_batteries

                soc_init_list = _resolve_soc_runtime_list("soc_init", target_list)
                for k in range(num_batteries):
                    if soc_init_list[k] < min_list[k]:
                        logger.warning(
                            f"Passed soc_init[{k}]={soc_init_list[k]} is lower than "
                            f"soc_min[{k}]={min_list[k]}, keeping real initial SOC for "
                            "optimization recovery"
                        )
                    if soc_init_list[k] > max_list[k]:
                        logger.warning(
                            f"Passed soc_init[{k}]={soc_init_list[k]} is greater than "
                            f"soc_max[{k}]={max_list[k]}, keeping real initial SOC for "
                            "optimization recovery"
                        )
                params["passed_data"]["soc_init"] = soc_init_list

                soc_final_list = _resolve_soc_runtime_list("soc_final", target_list)
                for k in range(num_batteries):
                    if soc_final_list[k] < min_list[k]:
                        logger.warning(
                            f"Passed soc_final[{k}]={soc_final_list[k]} is lower than "
                            f"soc_min[{k}]={min_list[k]}, setting soc_final[{k}]=soc_min[{k}]"
                        )
                        soc_final_list[k] = min_list[k]
                    if soc_final_list[k] > max_list[k]:
                        logger.warning(
                            f"Passed soc_final[{k}]={soc_final_list[k]} is greater than "
                            f"soc_max[{k}]={max_list[k]}, setting soc_final[{k}]=soc_max[{k}]"
                        )
                        soc_final_list[k] = max_list[k]
                params["passed_data"]["soc_final"] = soc_final_list
            # Optional intermediate SOC target (issue #553). Both are runtime-only
            # and default to None (no intermediate target); clamping/validation of
            # the value and the timestep is handled in Optimization.perform_optimization.
            params["passed_data"]["soc_target"] = runtimeparams.get("soc_target", None)
            params["passed_data"]["soc_target_timestep"] = runtimeparams.get(
                "soc_target_timestep", None
            )
            # Peak grid import already incurred this billing period (issue #623,
            # Phase 2). Runtime-only, in Watts; defaults to None (no floor).
            # Coercion/validation happens in Optimization.perform_optimization.
            params["passed_data"]["current_period_peak"] = runtimeparams.get(
                "current_period_peak", None
            )
            if "operating_timesteps_of_each_deferrable_load" in runtimeparams.keys():
                params["passed_data"]["operating_timesteps_of_each_deferrable_load"] = (
                    runtimeparams["operating_timesteps_of_each_deferrable_load"]
                )
                params["optim_conf"]["operating_timesteps_of_each_deferrable_load"] = runtimeparams[
                    "operating_timesteps_of_each_deferrable_load"
                ]
            if "operating_hours_of_each_deferrable_load" in params["optim_conf"].keys():
                params["passed_data"]["operating_hours_of_each_deferrable_load"] = params[
                    "optim_conf"
                ]["operating_hours_of_each_deferrable_load"]
            params["passed_data"]["start_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("start_timesteps_of_each_deferrable_load", None)
            params["passed_data"]["end_timesteps_of_each_deferrable_load"] = params[
                "optim_conf"
            ].get("end_timesteps_of_each_deferrable_load", None)

            forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        else:
            params["passed_data"]["prediction_horizon"] = None

            def _passthrough_soc_runtime(key: str):
                # The dayahead/perfect branch (this else) never had the
                # naive-mpc branch's target-fallback/clamp logic - it is (and
                # stays) a bare passthrough. The only pre-#610 behaviour was
                # float(scalar); a runtime list must not crash here. Length
                # validation and the scalar-broadcast/target-fallback for a
                # multi-battery plant both happen once, downstream, in
                # Optimization._normalize_soc_arg / perform_optimization - this
                # layer must not duplicate that logic, only stop crashing on it.
                if key not in runtimeparams:
                    return None
                value = runtimeparams[key]
                return list(value) if isinstance(value, list) else float(value)

            params["passed_data"]["soc_init"] = _passthrough_soc_runtime("soc_init")
            params["passed_data"]["soc_final"] = _passthrough_soc_runtime("soc_final")
            params["passed_data"]["soc_target"] = (
                float(runtimeparams["soc_target"]) if "soc_target" in runtimeparams else None
            )
            params["passed_data"]["soc_target_timestep"] = (
                int(runtimeparams["soc_target_timestep"])
                if "soc_target_timestep" in runtimeparams
                else None
            )
            params["passed_data"]["current_period_peak"] = None

        # Parsing the thermal model parameters
        # Load the default config
        if "def_load_config" in runtimeparams:
            params["optim_conf"]["def_load_config"] = runtimeparams["def_load_config"]
            params["optim_conf"]["number_of_deferrable_loads"] = len(
                runtimeparams["def_load_config"]
            )
        if "def_load_config" in params["optim_conf"]:
            for k in range(len(params["optim_conf"]["def_load_config"])):
                if "thermal_config" in params["optim_conf"]["def_load_config"][k]:
                    if (
                        "heater_desired_temperatures" in runtimeparams
                        and len(runtimeparams["heater_desired_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "desired_temperatures"
                        ] = runtimeparams["heater_desired_temperatures"][k]
                    if (
                        "heater_start_temperatures" in runtimeparams
                        and len(runtimeparams["heater_start_temperatures"]) > k
                    ):
                        params["optim_conf"]["def_load_config"][k]["thermal_config"][
                            "start_temperature"
                        ] = runtimeparams["heater_start_temperatures"][k]

        # Treat passed forecast data lists
        list_forecast_key = [
            "pv_power_forecast",
            "load_power_forecast",
            "load_cost_forecast",
            "prod_price_forecast",
            "outdoor_temperature_forecast",
        ]
        forecast_methods = [
            "weather_forecast_method",
            "load_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
            "outdoor_temperature_forecast_method",
        ]

        # Loop forecasts, check if value is a list and greater than or equal to forecast_dates
        for method, forecast_key in enumerate(list_forecast_key):
            if forecast_key in runtimeparams.keys():
                forecast_input = runtimeparams[forecast_key]
                if isinstance(forecast_input, dict):
                    forecast_data_df = pd.DataFrame.from_dict(
                        forecast_input, orient="index"
                    ).reset_index()
                    forecast_data_df.columns = ["time", "value"]
                    forecast_data_df["time"] = pd.to_datetime(
                        forecast_data_df["time"], format="ISO8601", utc=True
                    ).dt.tz_convert(time_zone)

                    # Aggregate any sub-step points to the optimization time step.
                    # Resample in the local time_zone so the buckets line up with the
                    # forecast grid, which get_forecast_dates floors in local time
                    # (this matters for sub-hour UTC offsets such as +05:30).
                    forecast_data_df = forecast_data_df.resample(
                        pd.to_timedelta(optimization_time_step, "minutes"),
                        on="time",
                    ).aggregate({"value": "mean"})
                    # Now move to UTC so the union/reindex below align by instant
                    # across DST edges without mixing two differently-localized
                    # indexes. forecast_dates is a list of ISO strings; parse it to
                    # the same UTC index. tz_convert only relabels, the instants are
                    # unchanged, so the local-time aggregation above is preserved.
                    forecast_data_df.index = forecast_data_df.index.tz_convert("UTC")
                    target_dates = pd.to_datetime(forecast_dates, utc=True)
                    # Align with forecast_dates using hold-last (step) semantics: each
                    # value holds until the next provided point. Union the provided
                    # index with the horizon first so points defined before
                    # forecast_dates[0] still anchor the forward-fill; reindexing
                    # straight onto forecast_dates with method="nearest" dropped that
                    # anchor and let the trailing bfill fill the leading slots with the
                    # NEXT value instead (issue #1003).
                    combined_index = forecast_data_df.index.union(target_dates)
                    forecast_data_df = forecast_data_df.reindex(combined_index)
                    # ffill applies the hold-last; bfill then covers any slots before
                    # the first provided point (a dict that starts after the window
                    # start) by extending that first value back over them.
                    forecast_data_df["value"] = forecast_data_df["value"].ffill().bfill()
                    forecast_data_df = forecast_data_df.reindex(target_dates)
                    forecast_input = forecast_data_df["value"].tolist()
                if isinstance(forecast_input, list) and len(forecast_input) >= len(forecast_dates):
                    params["passed_data"][forecast_key] = forecast_input
                    params["optim_conf"][forecast_methods[method]] = "list"
                else:
                    logger.error(
                        f"ERROR: The passed data is either the wrong type or the length is not correct, length should be {str(len(forecast_dates))}"
                    )
                    logger.error(
                        f"Passed type is {str(type(runtimeparams[forecast_key]))} and length is {str(len(runtimeparams[forecast_key]))}"
                    )
                # Check if string contains list, if so extract
                if isinstance(forecast_input, str) and isinstance(
                    ast.literal_eval(forecast_input), list
                ):
                    forecast_input = ast.literal_eval(forecast_input)
                    runtimeparams[forecast_key] = forecast_input
                list_non_digits = [
                    x for x in forecast_input if not (isinstance(x, int) or isinstance(x, float))
                ]
                if len(list_non_digits) > 0:
                    logger.warning(
                        f"There are non numeric values on the passed data for {forecast_key}, check for missing values (nans, null, etc)"
                    )
                    for x in list_non_digits:
                        logger.warning(
                            f"This value in {forecast_key} was detected as non digits: {str(x)}"
                        )
            else:
                params["passed_data"][forecast_key] = None

        # Explicitly handle historic_days_to_retrieve from runtimeparams BEFORE validation
        if "historic_days_to_retrieve" in runtimeparams:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = int(
                runtimeparams["historic_days_to_retrieve"]
            )

        # Treat passed data for forecast model fit/predict/tune at runtime
        if (
            params["passed_data"].get("historic_days_to_retrieve", None) is not None
            and params["passed_data"]["historic_days_to_retrieve"] < 9
        ):
            logger.warning(
                "warning `days_to_retrieve` is set to a value less than 9, this could cause an error with the fit"
            )
            logger.warning("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
            params["passed_data"]["historic_days_to_retrieve"] = 9
        else:
            if params["retrieve_hass_conf"].get("historic_days_to_retrieve", 0) < 9:
                logger.debug("setting`passed_data:days_to_retrieve` to 9 for fit/predict/tune")
                params["passed_data"]["historic_days_to_retrieve"] = 9
            else:
                params["passed_data"]["historic_days_to_retrieve"] = params["retrieve_hass_conf"][
                    "historic_days_to_retrieve"
                ]

        # UPDATED ML PARAMETER HANDLING
        # Define Helper Functions
        def _cast_bool(value):
            """Helper to cast string inputs to boolean safely."""
            # ast.literal_eval('None') returns None without raising — explicit guard needed.
            if value is None:
                return False
            try:
                return ast.literal_eval(str(value).capitalize())
            except (ValueError, SyntaxError):
                return False

        def _get_ml_param(name, params, runtimeparams, default=None, cast=None):
            """
            Prioritize Runtime Params -> Config Params (optim_conf) -> Default.
            """
            if name in runtimeparams:
                value = runtimeparams[name]
            else:
                value = params["optim_conf"].get(name, default)

            if cast is not None and value is not None:
                try:
                    value = cast(value)
                except Exception:
                    pass
            return value

        # Compute dynamic defaults
        # Default for var_model falls back to the configured load sensor
        default_var_model = params["retrieve_hass_conf"].get(
            "sensor_power_load_no_var_loads", "sensor.power_load_no_var_loads"
        )

        # Define Configuration Table
        # Format: (parameter_name, default_value, cast_function)
        ml_param_defs = [
            ("model_type", "long_train_data", None),
            ("var_model", default_var_model, None),
            ("sklearn_model", "KNeighborsRegressor", None),
            ("regression_model", "AdaBoostRegressor", None),
            ("num_lags", 48, None),
            ("split_date_delta", "48h", None),
            ("n_trials", 10, int),
            ("perform_backtest", False, _cast_bool),
            ("mlforecaster_weather_features", [], None),
        ]

        # Apply Configuration
        for name, default, caster in ml_param_defs:
            params["passed_data"][name] = _get_ml_param(
                name=name,
                params=params,
                runtimeparams=runtimeparams,
                default=default,
                cast=caster,
            )

        # Other non-dynamic options
        if "model_predict_publish" not in runtimeparams.keys():
            model_predict_publish = False
        else:
            model_predict_publish = ast.literal_eval(
                str(runtimeparams["model_predict_publish"]).capitalize()
            )
        params["passed_data"]["model_predict_publish"] = model_predict_publish
        if "model_predict_entity_id" not in runtimeparams.keys():
            model_predict_entity_id = "sensor.p_load_forecast_custom_model"
        else:
            model_predict_entity_id = runtimeparams["model_predict_entity_id"]
        params["passed_data"]["model_predict_entity_id"] = model_predict_entity_id
        if "model_predict_device_class" not in runtimeparams.keys():
            model_predict_device_class = "power"
        else:
            model_predict_device_class = runtimeparams["model_predict_device_class"]
        params["passed_data"]["model_predict_device_class"] = model_predict_device_class
        if "model_predict_unit_of_measurement" not in runtimeparams.keys():
            model_predict_unit_of_measurement = "W"
        else:
            model_predict_unit_of_measurement = runtimeparams["model_predict_unit_of_measurement"]
        params["passed_data"]["model_predict_unit_of_measurement"] = (
            model_predict_unit_of_measurement
        )
        if "model_predict_friendly_name" not in runtimeparams.keys():
            model_predict_friendly_name = "Load Power Forecast custom ML model"
        else:
            model_predict_friendly_name = runtimeparams["model_predict_friendly_name"]
        params["passed_data"]["model_predict_friendly_name"] = model_predict_friendly_name
        if "mlr_predict_entity_id" not in runtimeparams.keys():
            mlr_predict_entity_id = "sensor.mlr_predict"
        else:
            mlr_predict_entity_id = runtimeparams["mlr_predict_entity_id"]
        params["passed_data"]["mlr_predict_entity_id"] = mlr_predict_entity_id
        if "mlr_predict_device_class" not in runtimeparams.keys():
            mlr_predict_device_class = "power"
        else:
            mlr_predict_device_class = runtimeparams["mlr_predict_device_class"]
        params["passed_data"]["mlr_predict_device_class"] = mlr_predict_device_class
        if "mlr_predict_unit_of_measurement" not in runtimeparams.keys():
            mlr_predict_unit_of_measurement = None
        else:
            mlr_predict_unit_of_measurement = runtimeparams["mlr_predict_unit_of_measurement"]
        params["passed_data"]["mlr_predict_unit_of_measurement"] = mlr_predict_unit_of_measurement
        if "mlr_predict_friendly_name" not in runtimeparams.keys():
            mlr_predict_friendly_name = "mlr predictor"
        else:
            mlr_predict_friendly_name = runtimeparams["mlr_predict_friendly_name"]
        params["passed_data"]["mlr_predict_friendly_name"] = mlr_predict_friendly_name

        # Treat passed data for other parameters
        if "alpha" not in runtimeparams.keys():
            alpha = 0.5
        else:
            alpha = runtimeparams["alpha"]
        params["passed_data"]["alpha"] = alpha
        if "beta" not in runtimeparams.keys():
            beta = 0.5
        else:
            beta = runtimeparams["beta"]
        params["passed_data"]["beta"] = beta

        # Param to save forecast cache (i.e. Solcast)
        if "weather_forecast_cache" not in runtimeparams.keys():
            weather_forecast_cache = False
        else:
            weather_forecast_cache = runtimeparams["weather_forecast_cache"]
        params["passed_data"]["weather_forecast_cache"] = weather_forecast_cache

        # Param to make sure optimization only uses cached data. (else produce error)
        if "weather_forecast_cache_only" not in runtimeparams.keys():
            weather_forecast_cache_only = False
        else:
            weather_forecast_cache_only = runtimeparams["weather_forecast_cache_only"]
        params["passed_data"]["weather_forecast_cache_only"] = weather_forecast_cache_only

        # Param to bypass PV-feedback mixing during curtailment events (#818)
        if "ignore_pv_feedback_during_curtailment" not in runtimeparams.keys():
            ignore_pv_feedback_during_curtailment = False
        else:
            ignore_pv_feedback_during_curtailment = bool(
                runtimeparams["ignore_pv_feedback_during_curtailment"]
            )
        params["passed_data"]["ignore_pv_feedback_during_curtailment"] = (
            ignore_pv_feedback_during_curtailment
        )

        # A condition to manually save entity data under data_path/entities after optimization
        if "entity_save" not in runtimeparams.keys():
            entity_save = ""
        else:
            entity_save = runtimeparams["entity_save"]
        params["passed_data"]["entity_save"] = entity_save

        # A condition to put a prefix on all published data, or check for saved data under prefix name
        if "publish_prefix" not in runtimeparams.keys():
            publish_prefix = ""
        else:
            publish_prefix = runtimeparams["publish_prefix"]
        params["passed_data"]["publish_prefix"] = publish_prefix

        # Treat optimization (optim_conf) configuration parameters passed at runtime
        if "def_current_state" in runtimeparams.keys():
            dcs = runtimeparams["def_current_state"]
            # If passed as a string (e.g. '[false, false]'), parse it to a list
            if isinstance(dcs, str):
                try:
                    dcs = orjson.loads(dcs)
                except Exception:
                    logger.warning(f"Could not parse def_current_state string: {dcs}")
            # Use _cast_bool (not bool()) so string 'False' → False, not True.
            # bool('False') = True because non-empty strings are truthy in Python.
            if isinstance(dcs, list):
                params["optim_conf"]["def_current_state"] = [_cast_bool(s) for s in dcs]
            else:
                n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
                params["optim_conf"]["def_current_state"] = [_cast_bool(dcs)] * n_loads

        # def_current_on_timesteps: per-load elapsed ON timesteps for min-on remainder
        # (issue #952). Mirrors def_current_state: absent key -> no initial force (NOT
        # assumed zero). Validates that each entry is a non-negative integer.
        if "def_current_on_timesteps" in runtimeparams.keys():
            dcot = runtimeparams["def_current_on_timesteps"]
            # String -> parse JSON list
            if isinstance(dcot, str):
                try:
                    dcot = orjson.loads(dcot)
                except Exception:
                    logger.warning(f"Could not parse def_current_on_timesteps string: {dcot}")
            n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
            if isinstance(dcot, list):
                params["optim_conf"]["def_current_on_timesteps"] = [int(v) for v in dcot]
            else:
                params["optim_conf"]["def_current_on_timesteps"] = [int(dcot)] * n_loads

        # def_current_off_timesteps: per-load elapsed OFF timesteps for min-off remainder
        # (#952 follow-on). Mirrors def_current_on_timesteps: absent key -> no initial
        # force (NOT assumed zero). Validates that each entry is a non-negative integer.
        if "def_current_off_timesteps" in runtimeparams:
            dcoft = runtimeparams["def_current_off_timesteps"]
            # String -> parse JSON list
            if isinstance(dcoft, str):
                try:
                    dcoft = orjson.loads(dcoft)
                except Exception:
                    logger.warning(f"Could not parse def_current_off_timesteps string: {dcoft}")
            n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
            if isinstance(dcoft, list):
                params["optim_conf"]["def_current_off_timesteps"] = [int(v) for v in dcoft]
            else:
                params["optim_conf"]["def_current_off_timesteps"] = [int(dcoft)] * n_loads

        # def_current_power: per-load current power in watts (issue #605).
        # Absent key -> no pin, no force-on (NOT assumed zero). Mirrors def_current_on_timesteps.
        if "def_current_power" in runtimeparams:
            dcp = runtimeparams["def_current_power"]
            # String -> parse JSON list
            if isinstance(dcp, str):
                try:
                    dcp = orjson.loads(dcp)
                except Exception:
                    logger.warning(f"Could not parse def_current_power string: {dcp}")
            n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
            if isinstance(dcp, list):
                params["optim_conf"]["def_current_power"] = [float(v) for v in dcp]
            else:
                params["optim_conf"]["def_current_power"] = [float(dcp)] * n_loads

        # def_current_operating_timesteps: per-load completed operating timesteps today (issue #983).
        # Absent key -> no decrement (NOT assumed zero). Mirrors def_current_on_timesteps.
        if "def_current_operating_timesteps" in runtimeparams:
            dcots = runtimeparams["def_current_operating_timesteps"]
            # String -> parse JSON list
            if isinstance(dcots, str):
                try:
                    dcots = orjson.loads(dcots)
                except Exception:
                    logger.warning(
                        f"Could not parse def_current_operating_timesteps string: {dcots}"
                    )
            n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
            if isinstance(dcots, list):
                params["optim_conf"]["def_current_operating_timesteps"] = [int(v) for v in dcots]
            else:
                params["optim_conf"]["def_current_operating_timesteps"] = [int(dcots)] * n_loads

        # set_deferrable_load_single_constant arrives via the generic associations.csv
        # path as-is (may be a list of strings from runtimeparams JSON).  Apply the
        # same _cast_bool coercion so 'False' → False, not True (#873, mirrors #876).
        if "set_deferrable_load_single_constant" in runtimeparams.keys():
            sdlsc = runtimeparams["set_deferrable_load_single_constant"]
            if isinstance(sdlsc, list):
                params["optim_conf"]["set_deferrable_load_single_constant"] = [
                    _cast_bool(s) for s in sdlsc
                ]
            else:
                n_loads = len(params["optim_conf"]["nominal_power_of_deferrable_loads"])
                params["optim_conf"]["set_deferrable_load_single_constant"] = [
                    _cast_bool(sdlsc)
                ] * n_loads

        # Treat retrieve data from Home Assistant (retrieve_hass_conf) configuration parameters passed at runtime
        # Secrets passed at runtime
        if "solcast_api_key" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_api_key"] = runtimeparams["solcast_api_key"]
        if "solcast_rooftop_id" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = runtimeparams["solcast_rooftop_id"]
        if "solar_forecast_kwp" in runtimeparams.keys():
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = runtimeparams["solar_forecast_kwp"]
        # Treat custom entities id's and friendly names for variables
        # Runtime-only day windows for the forecast-calibration report. These are
        # report knobs (not config settings), so they live in passed_data next to
        # the custom_* ids and never touch retrieve_hass_conf / optim_conf. Each is
        # optional; the calibration action falls back to its default when unset.
        for calibration_key in (
            "calibration_days_to_retrieve",
            "calibration_test_days",
            "calibration_val_days",
        ):
            if calibration_key in runtimeparams.keys():
                try:
                    params["passed_data"][calibration_key] = int(runtimeparams[calibration_key])
                except (TypeError, ValueError):
                    logger.warning(
                        f"Ignoring non-integer runtime value for {calibration_key}: "
                        f"{runtimeparams[calibration_key]}"
                    )
        if "custom_pv_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_forecast_id"] = runtimeparams["custom_pv_forecast_id"]
        if "custom_load_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_load_forecast_id"] = runtimeparams[
                "custom_load_forecast_id"
            ]
        if "custom_pv_curtailment_id" in runtimeparams.keys():
            params["passed_data"]["custom_pv_curtailment_id"] = runtimeparams[
                "custom_pv_curtailment_id"
            ]
        if "custom_hybrid_inverter_id" in runtimeparams.keys():
            params["passed_data"]["custom_hybrid_inverter_id"] = runtimeparams[
                "custom_hybrid_inverter_id"
            ]
        if "custom_batt_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_forecast_id"] = runtimeparams[
                "custom_batt_forecast_id"
            ]
        if "custom_batt_soc_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_batt_soc_forecast_id"] = runtimeparams[
                "custom_batt_soc_forecast_id"
            ]
        if "custom_grid_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_grid_forecast_id"] = runtimeparams[
                "custom_grid_forecast_id"
            ]
        if "custom_cost_fun_id" in runtimeparams.keys():
            params["passed_data"]["custom_cost_fun_id"] = runtimeparams["custom_cost_fun_id"]
        if "custom_optim_status_id" in runtimeparams.keys():
            params["passed_data"]["custom_optim_status_id"] = runtimeparams[
                "custom_optim_status_id"
            ]
        if "custom_unit_load_cost_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_load_cost_id"] = runtimeparams[
                "custom_unit_load_cost_id"
            ]
        if "custom_unit_prod_price_id" in runtimeparams.keys():
            params["passed_data"]["custom_unit_prod_price_id"] = runtimeparams[
                "custom_unit_prod_price_id"
            ]
        if "custom_deferrable_forecast_id" in runtimeparams.keys():
            params["passed_data"]["custom_deferrable_forecast_id"] = runtimeparams[
                "custom_deferrable_forecast_id"
            ]
        if "custom_predicted_temperature_id" in runtimeparams.keys():
            params["passed_data"]["custom_predicted_temperature_id"] = runtimeparams[
                "custom_predicted_temperature_id"
            ]
        if "custom_heating_demand_id" in runtimeparams.keys():
            params["passed_data"]["custom_heating_demand_id"] = runtimeparams[
                "custom_heating_demand_id"
            ]

    # split config categories from params
    retrieve_hass_conf = params["retrieve_hass_conf"]
    optim_conf = params["optim_conf"]
    plant_conf = params["plant_conf"]

    # If heat_topology is present (static config or runtime override), compile
    # it down to flat optim_conf primitives. Runtime override wins over static
    # config because runtimeparams have already been merged above. Only takes
    # effect in "graph_topology" config mode - in "room_list" mode (the
    # default), heating is configured per-room instead (see
    # _append_room_thermal_loads in build_params), and the two mechanisms are
    # deliberately mutually exclusive so a user can't silently end up with
    # both sets of loads stacked in one MILP.
    heat_topology = optim_conf.get("heat_topology")
    heatpump_config_mode = optim_conf.get("heatpump_config_mode", "room_list")
    if isinstance(heat_topology, dict) and heat_topology and heatpump_config_mode == "graph_topology":
        try:
            compiled = compile_heat_topology(heat_topology)
        except ValueError as e:
            logger.error("heat_topology compile failed: %s", e)
            raise
        # Merge compiled fields into optim_conf, allowing user-set fields to win
        # for things the compiler always populates (e.g. operating_hours).
        for key, val in compiled.items():
            if key not in optim_conf or optim_conf[key] in (None, [], {}):
                optim_conf[key] = val
            else:
                # For the structural fields we ALWAYS want compiled values
                # (otherwise the compiled def_load_config doesn't match
                # number_of_deferrable_loads, etc.)
                if key in {
                    "number_of_deferrable_loads",
                    "def_load_config",
                    "shared_thermal_tanks",
                    "deferrable_load_groups",
                    "nominal_power_of_deferrable_loads",
                    "minimum_power_of_deferrable_loads",
                    "treat_deferrable_load_as_semi_cont",
                    "cost_forecast_per_deferrable_load",
                    # All per-load arrays must match number_of_deferrable_loads,
                    # which the compiler sets - so override any defaults.
                    "set_deferrable_load_single_constant",
                    "set_deferrable_startup_penalty",
                    "deferrable_load_max_cost",
                    "set_deferrable_max_startups",
                    "operating_hours_of_each_deferrable_load",
                    "start_timesteps_of_each_deferrable_load",
                    "end_timesteps_of_each_deferrable_load",
                    "is_electric_load",
                }:
                    optim_conf[key] = val
        params["optim_conf"] = optim_conf
        logger.info(
            "heat_topology compiled: %d sources, %d storage, %d flows, %d groups",
            len(heat_topology.get("sources", [])),
            len(heat_topology.get("storage", [])),
            len(heat_topology.get("flows", [])),
            len(heat_topology.get("actuator_groups", [])),
        )
    elif isinstance(heat_topology, dict) and heat_topology:
        logger.warning(
            "heat_topology is set but heatpump_config_mode is '%s', not "
            "'graph_topology' - ignoring heat_topology. Set heatpump_config_mode "
            "to 'graph_topology' in the Heat Pump section to use it.",
            heatpump_config_mode,
        )
    elif heat_topology:
        logger.warning(
            "heat_topology is set but is %s, not a dict (value: %r). "
            "Treating as 'no topology'. Use JSON null (not the string \"null\") "
            "or omit the key to disable.",
            type(heat_topology).__name__,
            heat_topology,
        )

    # Re-normalise per-load deferrable array params against the FINAL
    # number_of_deferrable_loads (#1040). This has to run last: the
    # association loop above, the def_load_config handling, and the
    # heat_topology compile step can each change number_of_deferrable_loads
    # or one of these arrays, and heat_topology runs even when runtimeparams
    # is None. Mirrors the per-battery re-normalisation above (#610), but
    # pads short arrays rather than truncating or erroring, matching
    # check_def_loads's existing pad-to-fit semantics.
    raw_num_def_loads = optim_conf.get("number_of_deferrable_loads", None)
    if raw_num_def_loads is not None:
        try:
            final_num_def_loads = int(raw_num_def_loads)
        except (TypeError, ValueError):
            logger.warning(
                "number_of_deferrable_loads is not a valid integer "
                f"({raw_num_def_loads!r}); skipping deferrable load array "
                "re-normalisation"
            )
            final_num_def_loads = None
        if final_num_def_loads is not None:
            # Write the cast int back: downstream readers of
            # optim_conf["number_of_deferrable_loads"] need an int, not
            # whatever raw value (e.g. a numeric string) the association
            # loop copied in.
            optim_conf["number_of_deferrable_loads"] = final_num_def_loads
            runtime_source = runtimeparams or {}
            for def_array_name, def_array_default in DEF_LOAD_ARRAY_PARAMS.items():
                legacy_name = DEF_LOAD_ARRAY_LEGACY_NAMES.get(def_array_name)
                runtime_value = runtime_source.get(def_array_name)
                if runtime_value is None and legacy_name is not None:
                    runtime_value = runtime_source.get(legacy_name)
                # A JSON null is treated as "not provided", matching the
                # association loop's own null-skipping - a padded config
                # array should never be reported as a runtime override.
                was_provided = runtime_value is not None

                if was_provided and not isinstance(runtime_value, list):
                    # A runtime scalar means every load, not "pad with the
                    # default" - mirrors check_batt_params's silent
                    # broadcast, and overrides any earlier partial handling
                    # (e.g. set_deferrable_load_single_constant's own scalar
                    # broadcast above, which doesn't know the final count).
                    optim_conf[def_array_name] = [runtime_value] * final_num_def_loads
                    continue

                current_value = optim_conf.get(def_array_name)
                if isinstance(current_value, list):
                    # Copy before padding: check_def_loads mutates in place,
                    # and this list may be the same object as
                    # runtimeparams[name] or a passed_data alias set earlier
                    # in this function - only optim_conf's own value should
                    # change.
                    before_value = list(current_value)
                    optim_conf[def_array_name] = list(current_value)
                else:
                    before_value = current_value
                optim_conf[def_array_name] = check_def_loads(
                    final_num_def_loads,
                    optim_conf,
                    def_array_default,
                    def_array_name,
                    logger,
                )
                _warn_if_runtime_def_array_too_short(
                    was_provided,
                    def_array_name,
                    before_value,
                    optim_conf[def_array_name],
                    final_num_def_loads,
                    logger,
                )
            params["optim_conf"] = optim_conf

    # Serialize the final params
    params = orjson.dumps(params, default=str).decode()
    return params, retrieve_hass_conf, optim_conf, plant_conf


def get_yaml_parse(params: str | dict, logger: logging.Logger) -> tuple[dict, dict, dict]:
    """
    Perform parsing of the params into the configuration catagories

    :param params: Built configuration parameters
    :type params: str or dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: A tuple with the dictionaries containing the parsed data
    :rtype: tuple(dict)

    """
    if params:
        if type(params) is str:
            input_conf = orjson.loads(params)
        else:
            input_conf = params
    else:
        input_conf = {}
        logger.error("No params have been detected for get_yaml_parse")
        return False, False, False

    optim_conf = input_conf.get("optim_conf", {})
    retrieve_hass_conf = input_conf.get("retrieve_hass_conf", {})
    plant_conf = input_conf.get("plant_conf", {})

    # Format time parameters
    if optim_conf.get("delta_forecast_daily", None) is not None:
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=optim_conf["delta_forecast_daily"])
    if retrieve_hass_conf.get("optimization_time_step", None) is not None:
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(
            retrieve_hass_conf["optimization_time_step"], "minutes"
        )
    if retrieve_hass_conf.get("time_zone", None) is not None:
        retrieve_hass_conf["time_zone"] = pytz.timezone(retrieve_hass_conf["time_zone"])

    return retrieve_hass_conf, optim_conf, plant_conf


def get_injection_dict(df: pd.DataFrame, plot_size: int | None = 1366) -> dict:
    """
    Build a dictionary with graphs and tables for the webui.

    :param df: The optimization result DataFrame
    :type df: pd.DataFrame
    :param plot_size: Size of the plot figure in pixels, defaults to 1366
    :type plot_size: Optional[int], optional
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict

    """
    cols_p = [i for i in df.columns.to_list() if "P_" in i]
    # Let's round the data in the DF
    if "optim_status" in df.columns:
        optim_status = df["optim_status"].iloc[0]
    else:
        optim_status = "Status not available"
    df.drop("optim_status", axis=1, inplace=True)
    cols_else = [i for i in df.columns.to_list() if "P_" not in i]
    df = df.apply(pd.to_numeric)
    df[cols_p] = df[cols_p].astype(int)
    df[cols_else] = df[cols_else].round(3)
    # Create plots
    # Figure 0: Systems Powers
    n_colors = len(cols_p)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_0 = px.line(
        df[cols_p],
        title="Systems powers schedule after optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_0.update_layout(xaxis_title="Timestamp", yaxis_title="System powers (W)")
    image_path_0 = fig_0.to_html(full_html=False, default_width="75%")
    # Figure 1: Battery SOC (Optional)
    image_path_1 = None
    if "SOC_opt" in df.columns.to_list():
        fig_1 = px.line(
            df["SOC_opt"],
            title="Battery state of charge schedule after optimization results",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig_1.update_layout(xaxis_title="Timestamp", yaxis_title="Battery SOC (%)")
        image_path_1 = fig_1.to_html(full_html=False, default_width="75%")
    # Figure Thermal: Temperatures (Optional)
    # Detect columns for predicted, target, min, or max temperatures
    cols_temp = [
        i
        for i in df.columns.to_list()
        if "predicted_temp_heater" in i
        or "target_temp_heater" in i
        or "min_temp_heater" in i
        or "max_temp_heater" in i
    ]
    image_path_temp = None
    if len(cols_temp) > 0:
        n_colors = len(cols_temp)
        colors = px.colors.sample_colorscale(
            "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
        )
        fig_temp = px.line(
            df[cols_temp],
            title="Thermal loads temperature schedule",
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig_temp.update_layout(xaxis_title="Timestamp", yaxis_title="Temperature (&deg;C)")
        image_path_temp = fig_temp.to_html(full_html=False, default_width="75%")
    # Figure 2: Costs
    cols_cost = [i for i in df.columns.to_list() if "cost_" in i or "unit_" in i]
    n_colors = len(cols_cost)
    colors = px.colors.sample_colorscale(
        "jet", [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)]
    )
    fig_2 = px.line(
        df[cols_cost],
        title="Systems costs obtained from optimization results",
        template="presentation",
        line_shape="hv",
        color_discrete_sequence=colors,
        render_mode="svg",
    )
    fig_2.update_layout(xaxis_title="Timestamp", yaxis_title="System costs (currency)")
    image_path_2 = fig_2.to_html(full_html=False, default_width="75%")
    # Tables
    table1 = df.reset_index().to_html(classes="mystyle", index=False)
    cost_cols = [i for i in df.columns if "cost_" in i]
    table2 = df[cost_cols].reset_index().sum(numeric_only=True)
    table2["optim_status"] = optim_status
    table2 = (
        table2.to_frame(name="Value")
        .reset_index(names="Variable")
        .to_html(classes="mystyle", index=False)
    )
    # Construct Injection Dict
    injection_dict = {}
    injection_dict["title"] = "<h2>EMHASS optimization results</h2>"
    injection_dict["subsubtitle0"] = "<h4>Plotting latest optimization results</h4>"
    # Add Powers
    injection_dict["figure_0"] = image_path_0
    # Add Thermal
    if image_path_temp is not None:
        injection_dict["figure_thermal"] = image_path_temp
    # Add SOC
    if image_path_1 is not None:
        injection_dict["figure_1"] = image_path_1
    # Add Costs
    injection_dict["figure_2"] = image_path_2
    injection_dict["subsubtitle1"] = "<h4>Last run optimization results table</h4>"
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = "<h4>Summary table for latest optimization results</h4>"
    injection_dict["table2"] = table2
    return injection_dict


def get_injection_dict_thermal_two_stage(df: pd.DataFrame) -> dict:
    """Build a graph-focused dictionary for thermal two-stage plan web UI."""
    if df is None or df.empty:
        return {}

    plot_df = df.copy()
    selected_model = ""
    if "selected_model" in plot_df.columns:
        selected_model = str(plot_df["selected_model"].iloc[0])

    def _build_line_figure(cols: list[str], title: str, y_axis: str) -> str | None:
        if not cols:
            return None
        series = plot_df[cols].apply(pd.to_numeric, errors="coerce")
        if series.dropna(how="all").empty:
            return None
        n_colors = len(cols)
        colors = px.colors.sample_colorscale(
            "jet",
            [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)],
        )
        fig = px.line(
            series,
            title=title,
            template="presentation",
            line_shape="hv",
            color_discrete_sequence=colors,
            render_mode="svg",
        )
        fig.update_layout(xaxis_title="Timestamp", yaxis_title=y_axis)
        return fig.to_html(full_html=False, default_width="75%")

    temp_cols = [
        col
        for col in [
            "predicted_temp_heater0",
            "actual_room_temp",
            "outdoor_temp",
            "baseline_curve",
            "setpoint_min",
            "setpoint_optimal",
            "setpoint_price_aware",
            "setpoint_neutral",
            "setpoint_max",
        ]
        if col in plot_df.columns
    ]
    energy_cols = [
        col
        for col in [
            "predicted_electric_power",
            "actual_electric_power",
            "predicted_gas_consumption",
            "actual_gas_consumption",
            "cv_estimated_electricity_kwh",
            "cv_estimated_gas_kwh",
        ]
        if col in plot_df.columns
    ]
    cost_cols = [
        col
        for col in [
            "electricity_price",
            "gas_price",
            "cv_estimated_electricity_cost",
            "cv_estimated_gas_cost",
            "cv_estimated_total_cost",
        ]
        if col in plot_df.columns
    ]

    fig_temp = _build_line_figure(temp_cols, "Thermal comfort and setpoint schedule", "Temperature")
    fig_energy = _build_line_figure(
        energy_cols,
        "Thermal energy and consumption schedule",
        "Power / energy",
    )
    fig_cost = _build_line_figure(cost_cols, "Thermal pricing and estimated costs", "Price / cost")

    injection_dict = {
        "title": "<h2>Thermal two-stage planning</h2>",
        "subsubtitle0": f"<h4>Selected best model: {selected_model}</h4>",
    }
    if fig_temp is not None:
        injection_dict["figure_temp"] = fig_temp
    if fig_energy is not None:
        injection_dict["figure_energy"] = fig_energy
    if fig_cost is not None:
        injection_dict["figure_cost"] = fig_cost
    return injection_dict


def compute_forecast_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Compute goodness-of-fit metrics for a forecast against realised values.

    Only rows where both the actual and the predicted value are present are
    used; any leading warm-up NaNs are excluded. The guards mirror the degenerate
    cases the metrics can hit: r2_score requires at least 2 samples, MAE/RMSE
    require at least 1, and MAPE excludes zero actuals to avoid division by zero.

    :param actual: The realised values.
    :type actual: pd.Series | np.ndarray
    :param predicted: The forecasted values (same index/length as ``actual``).
    :type predicted: pd.Series | np.ndarray
    :param logger: Optional logger for the degenerate-case warning.
    :type logger: logging.Logger, optional
    :return: A dict with keys mae, rmse, r2, mape, n_samples.
    :rtype: dict
    """
    actual_values = pd.Series(np.asarray(actual, dtype=float))
    predicted_values = pd.Series(np.asarray(predicted, dtype=float))
    valid_mask = predicted_values.notna() & actual_values.notna()
    n_valid_samples = int(valid_mask.sum())
    actual_valid = actual_values[valid_mask]
    predicted_valid = predicted_values[valid_mask]

    # Guard against degenerate cases (all-NaN predictions or single sample).
    # r2_score requires at least 2 samples; MAE/RMSE require at least 1.
    if n_valid_samples == 0:
        if logger is not None:
            logger.warning("Forecast metrics: no valid predictions - metrics set to NaN")
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
            "n_samples": 0,
        }
    mae = float(mean_absolute_error(actual_valid, predicted_valid))
    rmse = float(np.sqrt(mean_squared_error(actual_valid, predicted_valid)))
    # r2_score is undefined for a single sample (variance == 0)
    r2 = float(r2_score(actual_valid, predicted_valid)) if n_valid_samples > 1 else float("nan")
    # MAPE: exclude zero actuals to avoid division by zero
    nonzero_mask = actual_valid != 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(
                np.abs(
                    (actual_valid[nonzero_mask] - predicted_valid[nonzero_mask])
                    / actual_valid[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "n_samples": n_valid_samples,
    }


def get_injection_dict_forecast_model_fit(df_fit_pred: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF fit case.

    :param df_fit_pred: The fit result DataFrame
    :type df_fit_pred: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_fit_pred.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model fit</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Plotting train/test forecast model results for "
        + mlf.model_type
        + "<br>"
        + "Forecasting variable "
        + mlf.var_model
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


def get_room_temp_test_plot_html(df_plot: pd.DataFrame, room_name: str) -> str:
    """
    Render one room's train/test/predicted temperature DataFrame as an
    embeddable Plotly HTML div.

    df_plot's columns must be exactly "train"/"test"/"pred" (see
    command_line.py::refit_arx_model's honest-test-report
    block, which builds this DataFrame from the same never-touched-for-
    decisions held-out test split already used to report room_temp_test_mae_c) -
    the same df.plot()-based pattern get_injection_dict_forecast_model_fit
    already uses for the load forecaster's own train/test/pred chart, so a
    room's temperature fit reads the same way on the dashboard as the load
    model's fit does.

    :param df_plot: DataFrame indexed by timestamp with columns "train"
        (actual temperature during the train+val period), "test" (actual
        temperature during the held-out test period), "pred" (predicted
        temperature during that same test period) - cells outside a
        column's own segment are NaN, which df.plot() renders as a gap.
    :type df_plot: pd.DataFrame
    :param room_name: Room name, used in the y-axis title.
    :type room_name: str
    :return: An HTML string embedding the Plotly figure.
    :rtype: str
    """
    fig = df_plot.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=f"{room_name} temperature (°C)")
    fig.update_xaxes(title_text="Time")
    return fig.to_html(full_html=False, default_width="75%")


def get_forecast_trend_plot_html(df_plot: pd.DataFrame, y_axis_title: str) -> str:
    """Render an arbitrary forecast DataFrame (1+ columns, e.g. a single
    forecasted series, or a forecast alongside a flat reference line like
    comfort_min_temp) as an embeddable Plotly HTML div.

    Same df.plot()-based pattern as get_room_temp_test_plot_html, but
    generalized beyond the fixed train/test/pred column triple - for the
    predict-side forecast charts (compute_rc_model_forecast/
    compute_hybrid_heatpump_forecast/compute_arx_model_forecast
    in command_line.py) there is no train/test split, just the forecasted
    trajectory itself (optionally alongside a flat reference line).

    :param df_plot: DataFrame indexed by timestamp, one or more columns.
    :type df_plot: pd.DataFrame
    :param y_axis_title: Y-axis label, e.g. "Electric power (W)".
    :type y_axis_title: str
    :return: An HTML string embedding the Plotly figure.
    :rtype: str
    """
    fig = df_plot.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=y_axis_title)
    fig.update_xaxes(title_text="Time")
    return fig.to_html(full_html=False, default_width="75%")


# Fixed across every render call so a panel's color always means the same
# thing whether viewed alone or next to its neighbours, and across repeated
# refits - a data-dependent max would make a mildly-shaded roof look as
# "red" as a badly-shaded one just because nothing worse happened to be in
# that particular batch.
_HORIZON_POLAR_TRANSMITTANCE_CMAX = 0.3

# Neutral "open sky" fill for the unobstructed portion of each azimuth's
# wedge - every wedge always spans the full radius (a complete tile, not a
# bar that stops short and leaves bare background around it), so the part
# above the learned horizon needs its own flat color distinct from both the
# transmittance colorscale and the geometrically-blind grey.
_HORIZON_POLAR_CLEAR_SKY_COLOR = "#eaf4fb"

# Elevation resolution for the combined chart's genuine 2D fill (see
# _horizon_polar_2d_fill_traces) - a rendering resolution only, not a
# fitting bin (the underlying query, interpolate_partial_transmittance, is
# a continuous kernel-weighted average - see pv_shading_kalman.py).
_HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG = 5

# The 2D fill's one shared colorscale (see _horizon_polar_2d_fill_traces) -
# both the hard-object horizon's own transmittance (below it) and the
# fitted partial-transmittance surface (above it) are the same physical
# quantity, how much light still gets through, so both are colored from
# this single 0-1 scale rather than two separate ones; green=clear,
# red=blocked reads naturally across that whole range, including the
# partial layer's cold-start default of 1.0 (no extra attenuation).
_HORIZON_POLAR_PARTIAL_COLORSCALE = "RdYlGn"

# Matches the page the chart is embedded into (a plain white EMHASS
# webui page, not an Artifact - no dark-theme awareness needed) - used to
# cut the self-shaded region out to background rather than merely
# outlining it.
_HORIZON_POLAR_PAGE_BACKGROUND_COLOR = "#ffffff"

# render_horizon_polar_grid renders each chart (combined + every panel) as
# its own independent Plotly figure laid out via a responsive CSS grid,
# rather than one Plotly make_subplots figure holding all of them - a
# real, multi-panel install (e.g. 26+ microinverter-monitored panels) was
# observed to silently fail to fully render past the first couple of rows
# once that single figure grew tall enough (no console error - Plotly's
# own internal layout/rendering apparently has a practical complexity
# ceiling for many polar subplots coordinated in one figure). Independent
# figures have no such shared ceiling: each is equally cheap to lay out
# and render no matter how many total panels exist on the page.
_HORIZON_POLAR_CHART_HEIGHT_PX = 420
_HORIZON_POLAR_CHART_MIN_WIDTH_PX = 380
# A bare `auto-fit, minmax(MIN_WIDTH, 1fr)` grid packs in as many columns
# as the container's own width allows - on a wide monitor that's plenty
# of room for 7+, too many small charts to read comfortably at once. The
# grid CSS below combines this cap with _HORIZON_POLAR_CHART_MIN_WIDTH_PX
# via max() so a column is never narrower than that width-based share NOR
# narrower than the pixel floor - capping at this many on a wide screen,
# while still shrinking to fewer (not overflowing) on a narrow one.
_HORIZON_POLAR_CHART_MAX_COLUMNS = 5
_HORIZON_POLAR_CHART_GRID_GAP_PX = 16


def _find_safe_cut_azimuth(curve: dict, az_list) -> float:
    """A boundary curve drawn as a polyline around the full 360-degree
    circle needs one azimuth to treat as its start/end "seam". Picking one
    inside the curve's own longest run of undefined (None) values (if any)
    keeps the drawn line from jumping straight across a real physical gap
    - e.g. a self-shading curve that's undefined (never self-shaded)
    across the panel's own facing side - as if it were data. A curve
    that's defined everywhere has no wrong place to start, so any azimuth
    works.

    :param curve: {azimuth: value_or_None}, e.g. from
        compute_self_shading_curve/compute_sun_path_envelope.
    :type curve: dict
    :param az_list: The azimuths to consider, in render order.
    :return: One azimuth from az_list, safe to cut at.
    :rtype: float
    """
    az_list = list(az_list)
    present = [curve.get(az) is not None for az in az_list]
    n = len(present)
    if all(present):
        return az_list[0]
    doubled = present * 2
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    for i in range(2 * n):
        if not doubled[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
        if cur_len >= n:
            break
    mid = (best_start + best_len // 2) % n
    return az_list[mid]


def _horizon_polar_boundary_trace(curve: dict, render_azimuths, name: str, color: str) -> go.Scatterpolar:
    """One boundary curve (self-shading, or one edge of the sun-path
    envelope) as a Scatterpolar line, cut at its own safe seam azimuth
    (see _find_safe_cut_azimuth) so undefined stretches never get bridged
    by a spurious straight line. showlegend is always off - each chart is
    its own independent Plotly figure now (see render_horizon_polar_grid's
    own docstring), which has no notion of a legend shared across several
    figures, so render_horizon_polar_grid builds one plain HTML legend
    instead, covering whichever overlay(s) actually appear in the grid."""
    cut = _find_safe_cut_azimuth(curve, render_azimuths)
    order = sorted(render_azimuths, key=lambda a: (a - cut) % 360)
    theta = [az for az in order if curve.get(az) is not None]
    r = [90.0 - curve[az] for az in theta]
    return go.Scatterpolar(
        r=r,
        theta=theta,
        mode="lines",
        line=dict(color=color, width=2),
        name=name,
        showlegend=False,
        hoverinfo="skip",
    )


def _horizon_polar_self_shading_traces(curve: dict, render_azimuths) -> list[go.Scatterpolar]:
    """The self-shaded region cut to page background (not merely
    outlined) plus its boundary line - two traces, in the order they must
    be added so the cutout sits underneath the fill it's masking. Both
    always have showlegend off - see _horizon_polar_boundary_trace's own
    docstring for why."""
    cut = _find_safe_cut_azimuth(curve, render_azimuths)
    order = sorted(render_azimuths, key=lambda a: (a - cut) % 360)
    shaded_az = [az for az in order if curve.get(az) is not None]
    if not shaded_az:
        return []
    inner_r = [90.0 - curve[az] for az in shaded_az]
    mask_theta = list(shaded_az) + list(reversed(shaded_az))
    mask_r = inner_r + [90.0] * len(shaded_az)
    mask_trace = go.Scatterpolar(
        r=mask_r,
        theta=mask_theta,
        mode="lines",
        fill="toself",
        fillcolor=_HORIZON_POLAR_PAGE_BACKGROUND_COLOR,
        line=dict(color=_HORIZON_POLAR_PAGE_BACKGROUND_COLOR, width=0),
        showlegend=False,
        hoverinfo="skip",
    )
    boundary_trace = go.Scatterpolar(
        r=inner_r,
        theta=shaded_az,
        mode="lines",
        line=dict(color="#1c1a17", width=2),
        name="Self-shading boundary (tilt)",
        showlegend=False,
        hoverinfo="skip",
    )
    return [mask_trace, boundary_trace]


def _horizon_polar_gradient_bar_html(colorscale_name: str, label: str, cmax_pct: float) -> str:
    """One shared, plain-CSS gradient bar (label + 0% .. cmax_pct% swatch)
    standing in for a Plotly colorbar - built from the SAME colorscale
    stops Plotly itself uses (plotly.colors.get_colorscale), so the
    gradient is an exact match, not an approximation. Each chart is its
    own independent Plotly figure now (see render_horizon_polar_grid's
    own docstring), so a per-figure colorbar can't be shared across the
    grid - this renders once, in the grid's own shared header, instead."""
    stops = pc.get_colorscale(colorscale_name)
    gradient = ", ".join(f"{color} {frac * 100:.4f}%" for frac, color in stops)
    return (
        '<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px;">'
        f"<span>{label}</span><span>0%</span>"
        f'<span style="display:inline-block;width:100px;height:10px;border-radius:2px;'
        f'background:linear-gradient(to right, {gradient});"></span>'
        f"<span>{cmax_pct:.0%}</span></span>"
    )


def _horizon_polar_2d_fill_traces(
    render_azimuths,
    hard_object_elevation: dict,
    hard_object_transmittance: dict,
    partial_transmittance: dict,
    sun_min_curve: dict,
    sun_max_curve: dict,
    season: str,
) -> list:
    """Genuine 2D (azimuth x elevation) fill for the combined chart,
    replacing the usual two-stacked-band wedge when both a fitted
    partial-transmittance surface and the sun-path envelope are available
    (see pv_shading_kalman.aggregate_partial_transmittance_surface/
    compute_sun_path_envelope) - reuses the mechanism validated in this
    feature's own prototyping (single_panel_2d_demo.py). Each render cell
    is classified, in order:
      1. Below the learned hard-object horizon -> that azimuth's
         hard-object transmittance - this is the existing 1D model's own
         boundary, unchanged; the caller draws it as an explicit line (see
         _horizon_polar_boundary_trace) rather than leaving it implied by
         a change in color scale.
      2. Above it, but outside the sun's own real yearly elevation range
         at that azimuth -> grey/unknown, same treatment as an entirely
         geometrically-blind azimuth column.
      3. Above it and physically reachable -> the fitted partial-
         transmittance surface's own value there.

    Both zones share ONE colorscale (_HORIZON_POLAR_PARTIAL_COLORSCALE,
    0-1) - hard-object transmittance and partial transmittance are the
    same physical quantity (how much light still gets through), just
    fitted by two different layers, so one scale reads as one continuous
    measurement rather than two unrelated ones; the horizon line is what
    tells the two fitted layers apart, not the color.

    Colors are pre-baked to hex via plotly.colors.sample_colorscale rather
    than left to Plotly's own per-trace colorscale, since a single ring
    still mixes the grey "unknown" cells with colorscale-mapped ones,
    which one colorscale-mapped marker.color can't express together. No
    colorbar is attached here - each chart is its own independent Plotly
    figure now (see render_horizon_polar_grid's own docstring), so a
    per-figure colorbar can't be shared across the grid either; the
    caller builds one shared plain-HTML gradient bar instead, covering
    whichever colorscale(s) actually appear anywhere in the grid.

    :param render_azimuths: Fine azimuth grid (AZIMUTH_RENDER_SPACING_DEG
        apart), the same grid the ordinary two-band fill and the curve
        overlays already render at.
    :param hard_object_elevation: {azimuth: clamped hard-object horizon
        elevation}, already computed by the caller via
        interpolate_horizon_profile.
    :param hard_object_transmittance: {azimuth: hard-object
        transmittance}, same source.
    :param partial_transmittance: The persisted 2D surface (see
        aggregate_partial_transmittance_surface).
    :param sun_min_curve: {azimuth: lowest elevation the sun is ever
        observed at, or None} (see compute_sun_path_envelope).
    :param sun_max_curve: Same shape, highest elevation.
    :param season: Which season to query the partial surface for.
    :return: A list of go.Barpolar, one per elevation ring.
    :rtype: list
    """
    from emhass.pv_shading_kalman import AZIMUTH_RENDER_SPACING_DEG, interpolate_partial_transmittance

    az_list = list(render_azimuths)
    el_grid = np.arange(0, 90, _HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG)
    traces = []
    for el_start in el_grid:
        el_center = float(el_start) + _HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG / 2
        base = 90.0 - (float(el_start) + _HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG)

        below_horizon = {az: el_center < hard_object_elevation[az] for az in az_list}
        above_az = [az for az in az_list if not below_horizon[az]]
        physically_possible = {}
        for az in above_az:
            lo, hi = sun_min_curve.get(az), sun_max_curve.get(az)
            physically_possible[az] = lo is not None and lo <= el_center <= hi
        query_az = [az for az in above_az if physically_possible[az]]
        partial_values = {}
        if query_az:
            queried = interpolate_partial_transmittance(
                partial_transmittance,
                pd.Series(query_az, dtype=float),
                pd.Series([el_center] * len(query_az)),
                pd.Series([season] * len(query_az)),
                sun_min_curve,
                sun_max_curve,
            )
            partial_values = dict(zip(query_az, queried, strict=True))

        colors, hover = [], []
        el_label = f"{el_start}-{el_start + _HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG}"
        for az in az_list:
            if below_horizon[az]:
                value = hard_object_transmittance[az]
                frac = min(max(value, 0.0), 1.0)
                colors.append(pc.sample_colorscale(_HORIZON_POLAR_PARTIAL_COLORSCALE, [frac])[0])
                hover.append(
                    f"az={az}&deg;, elevation~{el_label}&deg;<br>"
                    f"below the learned horizon, transmittance={value:.0%}"
                )
            elif not physically_possible[az]:
                colors.append("lightgrey")
                hover.append(
                    f"az={az}&deg;, elevation~{el_label}&deg;<br>"
                    "no direct sun ever reaches this panel from here"
                )
            else:
                value = partial_values[az]
                frac = min(max(value, 0.0), 1.0)
                colors.append(pc.sample_colorscale(_HORIZON_POLAR_PARTIAL_COLORSCALE, [frac])[0])
                hover.append(
                    f"az={az}&deg;, elevation~{el_label}&deg;<br>"
                    f"above the learned horizon, partial transmittance={value:.0%}"
                )

        traces.append(
            go.Barpolar(
                r=[_HORIZON_POLAR_ELEVATION_RENDER_SPACING_DEG] * len(az_list),
                theta=az_list,
                base=[base] * len(az_list),
                width=[AZIMUTH_RENDER_SPACING_DEG] * len(az_list),
                marker=dict(color=colors),
                showlegend=False,
                hovertext=hover,
                hoverinfo="text",
            )
        )

    return traces


def render_horizon_polar_grid(
    profile: dict,
    profile_per_panel: dict,
    season: str,
    blind_azimuths_per_panel: dict[str, set[int]] | None = None,
    blind_azimuths_combined: set[int] | None = None,
    partial_transmittance: dict | None = None,
    partial_transmittance_per_panel: dict[str, dict] | None = None,
    sun_path_envelope: tuple[dict, dict] | None = None,
    self_shading_curve_combined: dict | None = None,
    self_shading_curve_per_panel: dict[str, dict] | None = None,
    diffuse_transmission_factor: dict[str, float] | None = None,
) -> str:
    """Render one season's learned horizon as a responsive CSS grid of
    small, INDEPENDENT Plotly polar bar charts: the combined/aggregate
    profile first, then one per panel - replaces the old flat
    per-(panel, azimuth, season) HTML table, which for a real multi-panel
    install is hundreds of rows and unreadable.

    Each chart is its own separate go.Figure/fig.to_html() call, laid out
    via a plain CSS grid (auto-wrapping at _HORIZON_POLAR_CHART_MIN_WIDTH_PX
    per chart, capped at _HORIZON_POLAR_CHART_MAX_COLUMNS per row even on a
    wide screen) rather than one shared Plotly make_subplots figure holding
    every chart at once. A real many-panel install (26+ microinverter-
    monitored panels) was observed to silently stop rendering past the
    first couple of rows once that single shared figure grew tall enough -
    no console error, Plotly's own internal layout/rendering apparently
    has a practical complexity ceiling for coordinating that many polar
    subplots in one figure. Independent figures have no such shared
    ceiling: each is equally cheap to lay out and render no matter how
    many total panels exist on the page, and a rendering problem with one
    panel's chart can no longer take any of the others down with it.
    Since Plotly.js itself only needs loading once, only the FIRST chart
    embeds the library (include_plotlyjs=True, fig.to_html()'s own
    default) - every other chart skips the redundant embed
    (include_plotlyjs=False), relying on the first inline <script> having
    already run (browsers execute synchronous inline scripts in document
    order) to make the global Plotly object available by the time it's
    needed.

    The learned profile is a continuous function of azimuth
    (pv_shading_kalman.interpolate_horizon_profile), not a fixed set of
    bins, so this renders it as ~180 thin wedges sampled every
    AZIMUTH_RENDER_SPACING_DEG degrees - fine enough to read as a smooth
    gradient rather than a visible grid - instead of one wedge per
    persisted anchor. Each thin wedge is a complete tile spanning the full
    radius - center to rim, never a bar that stops short and leaves bare
    background around it - split into two stacked go.Barpolar segments:
    the outer band, from the rim inward to the interpolated horizon
    elevation, colored by transmittance (how much light still gets through
    there); the inner band, from that boundary to the center, the neutral
    open-sky fill (nothing obstructs a panel's view once above its own
    learned horizon). Center=zenith (90deg, always clear), rim=horizon
    (0deg) - the standard sun-path/horizon-diagram convention, so an
    obstruction reads as encroaching inward from the horizon exactly as it
    would look through a fisheye lens pointed straight up from the panel.
    A season with no data ANYWHERE yet (e.g. only "summer" exists until
    enough of the year has passed) renders as an empty chart rather than a
    misleading all-clear default; once at least one anchor has this
    season's data, every other anchor's cold-start default (elevation=0)
    is a legitimate part of the continuous fit, not a gap to hide.

    A direction the sun can *never* test for a given panel (self-shaded by
    its own tilt, or the sun's own path at this latitude never reaching
    there - see pv_shading_kalman.compute_geometrically_blind_azimuths,
    recomputed here at this function's own fine render resolution) stays
    at its cold-start default (elevation=0) forever, which would otherwise
    look identical to a confirmed-clear reading. Those wedges get a third,
    independent full-radius grey wedge instead - "unknown", not "clear".

    When a chart has its own partial-transmittance surface (partial_transmittance
    for the combined chart, partial_transmittance_per_panel[panel] for a
    panel chart) AND sun_path_envelope is given, that chart's fill switches
    from the two-band wedge above to a genuine 2D (azimuth x elevation)
    fill (see _horizon_polar_2d_fill_traces) showing the fitted partial-
    shading surface above the hard-object horizon too, not just below it.
    The hard-object horizon and the partial-transmittance surface are the
    same physical quantity (how much light still gets through) fitted by
    two different layers, so both are colored from ONE shared scale/
    colorbar rather than two - what tells the two fitted layers apart is
    an explicit "Learned horizon (hard object)" boundary line drawn
    directly on top of the fill, not a change in color. A chart with no
    partial-transmittance surface of its own (e.g. a panel not present in
    partial_transmittance_per_panel) keeps the two-band rendering
    unchanged - each chart's 2D fill, when it has one, is always fit from
    that chart's OWN data (see aggregate_partial_transmittance_surface),
    never borrowed from another chart's.

    self_shading_curve_combined/self_shading_curve_per_panel (see
    compute_self_shading_curve), when given for a chart, draw that panel's
    precise tilt-based self-shading boundary and cut the self-shaded
    region to page background rather than merely outlining it - a finer,
    panel-specific alternative to the coarser blind_azimuths_* wedge.
    sun_path_envelope (see compute_sun_path_envelope), when given, draws
    the sun's own real yearly elevation envelope (lowest/highest reach at
    each azimuth) on EVERY chart - it depends only on site latitude/
    longitude, not panel orientation, so the same curve applies to the
    combined chart and every panel alike, independent of whether it's also
    being used to gate the 2D fill above. diffuse_transmission_factor,
    not being spatial, is shown as text appended to the combined chart's
    own title when it has an entry for the season being rendered.

    Every chart's angular axis is set to compass convention (0=North=up,
    clockwise) so it reads like a sun-path/horizon diagram, and every
    chart using the same rendering path (the ordinary two-band fill, or
    the combined chart's own 2D fill) uses the same fixed color scale so
    panels are directly comparable within that path - each chart now
    carries its own colorbar/legend rather than sharing one across the
    whole grid, since each is its own independent figure. Same
    fig.to_html()-embed pattern as
    get_room_temp_test_plot_html/get_forecast_trend_plot_html above - no
    new dependency, Plotly is already used throughout this module.

    :param profile: The combined/aggregate profile - {azimuth_str:
        {season: {"elevation": float, "transmittance": float}}}, the same
        shape refit_pv_horizon_model already produces.
    :type profile: dict
    :param profile_per_panel: {panel_sensor_id: <profile, same shape as
        above>}, or empty/falsy when no per-panel sensors are configured.
    :type profile_per_panel: dict
    :param season: Which season to render (e.g. "summer").
    :type season: str
    :param blind_azimuths_per_panel: {panel_sensor_id: {azimuth anchors
        geometrically blind for that panel}}, or None/omitted.
    :type blind_azimuths_per_panel: dict[str, set[int]] | None
    :param blind_azimuths_combined: Geometrically blind azimuth anchors for
        the combined/aggregate chart - only well-defined for a single
        shared PV orientation group; None for a multi-orientation-group
        plant (no single set applies to the combined total) or omitted.
    :type blind_azimuths_combined: set[int] | None
    :param partial_transmittance: The persisted 2D partial-transmittance
        surface for the combined total (see
        aggregate_partial_transmittance_surface), or None/omitted to keep
        the two-band fill on the combined chart.
    :type partial_transmittance: dict | None
    :param partial_transmittance_per_panel: {panel_sensor_id: <surface,
        same shape as partial_transmittance>}, or None/omitted to keep the
        two-band fill on every panel chart. A panel absent from this dict
        (even when the dict itself is given) also keeps the two-band fill.
    :type partial_transmittance_per_panel: dict[str, dict] | None
    :param sun_path_envelope: (sun_min_curve, sun_max_curve) from
        compute_sun_path_envelope, or None/omitted to skip both the
        envelope overlay and the 2D fill's physical gate.
    :type sun_path_envelope: tuple[dict, dict] | None
    :param self_shading_curve_combined: The combined chart's precise
        self-shading boundary (see compute_self_shading_curve), or
        None/omitted to skip that overlay there.
    :type self_shading_curve_combined: dict | None
    :param self_shading_curve_per_panel: {panel_sensor_id: <curve, same
        shape as above>}, or None/omitted to skip that overlay on every
        panel chart.
    :type self_shading_curve_per_panel: dict[str, dict] | None
    :param diffuse_transmission_factor: {season: factor} - the season's
        empirically- or theoretically-estimated diffuse-light attenuation
        (see forecast.py::_apply_pv_horizon_mask), or None/omitted to
        leave the combined chart's title unchanged.
    :type diffuse_transmission_factor: dict[str, float] | None
    :return: An HTML string embedding one independent Plotly figure per
        chart, wrapped in a responsive CSS grid.
    :rtype: str
    """
    from emhass.pv_shading_kalman import AZIMUTH_RENDER_SPACING_DEG, interpolate_horizon_profile

    blind_azimuths_per_panel = blind_azimuths_per_panel or {}
    self_shading_curve_per_panel = self_shading_curve_per_panel or {}
    partial_transmittance_per_panel = partial_transmittance_per_panel or {}
    diffuse_transmission_factor = diffuse_transmission_factor or {}
    panels_sorted = sorted(profile_per_panel or {})
    entries = [
        (
            "Combined (all panels)",
            profile,
            blind_azimuths_combined,
            self_shading_curve_combined,
            partial_transmittance,
            True,
        )
    ] + [
        (
            panel,
            profile_per_panel[panel],
            blind_azimuths_per_panel.get(panel),
            self_shading_curve_per_panel.get(panel),
            partial_transmittance_per_panel.get(panel),
            False,
        )
        for panel in panels_sorted
    ]

    render_azimuths = np.arange(0, 360, AZIMUTH_RENDER_SPACING_DEG)
    sun_min_curve, sun_max_curve = sun_path_envelope if sun_path_envelope else (None, None)
    # Plotly has no notion of a legend shared across several independent
    # figures, and every chart now IS its own independent figure (see this
    # function's own docstring) - so each chart's own Plotly legend is
    # switched off below, and one plain HTML legend is built here instead,
    # covering whichever overlay(s) actually appear anywhere in this grid.
    any_2d_fill_used = False
    any_ordinary_fill_used = False
    any_self_shading_used = False
    any_sun_path_used = False

    chart_divs = []
    for i, (label, one_profile, blind_azimuths, self_shading_curve, own_partial_transmittance, is_combined) in enumerate(
        entries
    ):
        season_present = any(season in seasons for seasons in one_profile.values())
        if not season_present:
            azimuths, elevations, transmittances = [], [], []
        else:
            query_azimuth = pd.Series(render_azimuths, dtype=float)
            query_season = pd.Series([season] * len(render_azimuths))
            elevation_series, transmittance_series = interpolate_horizon_profile(
                one_profile, query_azimuth, query_season
            )
            azimuths = list(render_azimuths)
            elevations = [max(e, 0.0) for e in elevation_series]
            transmittances = list(transmittance_series)

        use_2d_fill = (
            season_present
            and own_partial_transmittance is not None
            and sun_min_curve is not None
            and sun_max_curve is not None
        )

        fig = go.Figure()

        if use_2d_fill:
            any_2d_fill_used = True
            hard_object_elevation = dict(zip(azimuths, elevations, strict=True))
            for trace in _horizon_polar_2d_fill_traces(
                render_azimuths,
                hard_object_elevation,
                dict(zip(azimuths, transmittances, strict=True)),
                own_partial_transmittance,
                sun_min_curve,
                sun_max_curve,
                season,
            ):
                fig.add_trace(trace)
            # The hard-object horizon is now one flat color-scale like
            # everything else in the 2D fill, so it needs its own explicit
            # line to stay visible as a boundary.
            fig.add_trace(
                _horizon_polar_boundary_trace(
                    hard_object_elevation,
                    render_azimuths,
                    "Learned horizon (hard object)",
                    "#000000",
                )
            )
        else:
            if season_present:
                any_ordinary_fill_used = True
            # Center=zenith, rim=horizon: the open-sky band runs from the
            # center out to (90 - elevation), then the transmittance-
            # colored band stacks on top of that (base=90-elevation) for
            # the remaining `elevation` degrees out to the rim - so its
            # own bar length stays numerically equal to elevation, and the
            # hovertemplate below needs no change to keep showing the true
            # elevation value.
            clear_band = [90.0 - e for e in elevations]

            fig.add_trace(
                go.Barpolar(
                    r=clear_band,
                    theta=azimuths,
                    width=[AZIMUTH_RENDER_SPACING_DEG] * len(azimuths),
                    marker=dict(color=_HORIZON_POLAR_CLEAR_SKY_COLOR),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Barpolar(
                    r=elevations,
                    theta=azimuths,
                    base=clear_band,
                    width=[AZIMUTH_RENDER_SPACING_DEG] * len(azimuths),
                    # No colorbar here - each chart is its own independent
                    # figure now, so a shared plain-HTML gradient bar is
                    # built once for the whole grid instead (below).
                    marker=dict(
                        color=transmittances,
                        colorscale="YlOrRd",
                        cmin=0,
                        cmax=_HORIZON_POLAR_TRANSMITTANCE_CMAX,
                        showscale=False,
                    ),
                    hovertemplate=(
                        "az=%{theta}&deg;<br>elevation=%{r:.1f}&deg;"
                        "<br>transmittance=%{marker.color:.0%}<extra></extra>"
                    ),
                )
            )
        if blind_azimuths:
            blind_sorted = sorted(blind_azimuths)
            fig.add_trace(
                go.Barpolar(
                    r=[90] * len(blind_sorted),
                    theta=blind_sorted,
                    width=[AZIMUTH_RENDER_SPACING_DEG] * len(blind_sorted),
                    marker=dict(color="lightgrey"),
                    showlegend=False,
                    hovertemplate=(
                        "az=%{theta}&deg;<br>no direct sun ever reaches this panel "
                        "from here<extra></extra>"
                    ),
                )
            )
        if self_shading_curve:
            any_self_shading_used = True
            for trace in _horizon_polar_self_shading_traces(self_shading_curve, render_azimuths):
                fig.add_trace(trace)
        if sun_path_envelope:
            any_sun_path_used = True
            fig.add_trace(
                _horizon_polar_boundary_trace(
                    sun_max_curve, render_azimuths, "Sun's highest reach", "#2a6fb0"
                )
            )
            fig.add_trace(
                _horizon_polar_boundary_trace(
                    sun_min_curve, render_azimuths, "Sun's lowest reach", "#2a6fb0"
                )
            )

        title = label
        if is_combined and season in diffuse_transmission_factor:
            title = f"{label} - diffuse: {diffuse_transmission_factor[season]:.0%}"

        fig.update_layout(
            polar=dict(
                angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False),
                radialaxis=dict(range=[0, 90], showticklabels=False),
            ),
            title=dict(text=title, x=0.5, xanchor="center"),
            template="presentation",
            height=_HORIZON_POLAR_CHART_HEIGHT_PX,
            # No per-chart legend - see the shared HTML legend built below
            # instead, since Plotly can't share one legend across several
            # independent figures.
            showlegend=False,
            margin=dict(t=60, b=30, l=30, r=90),
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0), default_width="100%")
        chart_divs.append(f'<div class="horizon-polar-chart">{chart_html}</div>')

    legend_entries = []
    if any_2d_fill_used:
        legend_entries.append(("#000000", "Learned horizon (hard object)"))
    if any_self_shading_used:
        legend_entries.append(("#1c1a17", "Self-shading boundary (tilt)"))
    if any_sun_path_used:
        legend_entries.append(("#2a6fb0", "Sun's highest/lowest reach"))
    swatches = "".join(
        '<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px;">'
        f'<span style="display:inline-block;width:20px;height:3px;background:{color};"></span>'
        f"{label}</span>"
        for color, label in legend_entries
    )
    gradient_bars = ""
    if any_ordinary_fill_used:
        gradient_bars += _horizon_polar_gradient_bar_html(
            "YlOrRd", "Transmittance", _HORIZON_POLAR_TRANSMITTANCE_CMAX
        )
    if any_2d_fill_used:
        gradient_bars += _horizon_polar_gradient_bar_html(
            _HORIZON_POLAR_PARTIAL_COLORSCALE, "Transmittance", 1.0
        )
    shared_header = ""
    if swatches or gradient_bars:
        shared_header = (
            '<div style="display:flex;flex-wrap:wrap;align-items:center;'
            'margin-bottom:12px;font-size:13px;">' + swatches + gradient_bars + "</div>"
        )

    # max(pixel floor, width-based share) - see _HORIZON_POLAR_CHART_MAX_COLUMNS's
    # own comment for why: never narrower than that floor (keeps every
    # chart legible), and never so wide that fewer than that many columns
    # would fit a wide screen (keeps the grid from turning into 2 giant
    # charts side by side just because the window is wide).
    n_gaps = _HORIZON_POLAR_CHART_MAX_COLUMNS - 1
    column_min_width = (
        f"max({_HORIZON_POLAR_CHART_MIN_WIDTH_PX}px, "
        f"calc((100% - {n_gaps} * {_HORIZON_POLAR_CHART_GRID_GAP_PX}px) / "
        f"{_HORIZON_POLAR_CHART_MAX_COLUMNS}))"
    )
    return (
        shared_header
        + '<div style="display:grid;grid-template-columns:'
        f"repeat(auto-fit, minmax({column_min_width}, 1fr));"
        f'gap:{_HORIZON_POLAR_CHART_GRID_GAP_PX}px;align-items:start;">'
        + "".join(chart_divs)
        + "</div>"
    )


def get_injection_dict_forecast_calibration(result: dict) -> dict:
    """
    Build the webui graph + metrics table for the forecast-calibration action.

    :param result: The dict returned by ``compute_forecast_calibration``
        (keys: ``table``, ``plot``, ``val_window``, ``caveats``).
    :type result: dict
    :return: A dictionary containing the graph and table in html format
    :rtype: dict
    """
    plot_df = result["plot"]
    fig = plot_df.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text="Load")
    fig.update_xaxes(title_text="Time")
    figure_0 = fig.to_html(full_html=False, default_width="75%")
    table1 = result["table"].to_html(classes="mystyle", index=False)
    val_start, val_end = result["val_window"]
    injection_dict = {}
    injection_dict["title"] = "<h2>Load forecast calibration</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Actual vs forecast methods over the validation window "
        + f"({val_start} to {val_end})</h4>"
    )
    injection_dict["figure_0"] = figure_0
    injection_dict["subsubtitle1"] = (
        "<h4>Accuracy metrics by method and split (train / test / val)</h4>"
    )
    injection_dict["table1"] = table1
    injection_dict["subsubtitle2"] = "<h5>" + result["caveats"] + "</h5>"
    return injection_dict


def get_injection_dict_forecast_model_tune(df_pred_optim: pd.DataFrame, mlf: MLForecaster) -> dict:
    """
    Build a dictionary with graphs and tables for the webui for special MLF tune case.

    :param df_pred_optim: The tune result DataFrame
    :type df_pred_optim: pd.DataFrame
    :param mlf: The MLForecaster object
    :type mlf: MLForecaster
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    fig = df_pred_optim.plot()
    fig.layout.template = "presentation"
    fig.update_yaxes(title_text=mlf.model_type)
    fig.update_xaxes(title_text="Time")
    image_path_0 = fig.to_html(full_html=False, default_width="75%")
    # The dict of plots
    injection_dict = {}
    injection_dict["title"] = "<h2>Custom machine learning forecast model tune</h2>"
    injection_dict["subsubtitle0"] = (
        "<h4>Performed a tuning routine using bayesian optimization for "
        + mlf.model_type
        + "<br>"
        + "Forecasting variable "
        + mlf.var_model
        + "</h4>"
    )
    injection_dict["figure_0"] = image_path_0
    return injection_dict


def get_injection_dict_thermal_models(results: dict, title: str) -> dict:
    """Build the webui injection_dict for the thermal-models-refit/-tune/
    -forecast consolidated actions: one <h4> + full key/value table per
    model that actually ran (or a "no result" note for a model that was
    enabled but declined - see e.g. refit_enabled_thermal_models), so none
    of the three consolidated actions lose detail compared to calling the
    individual per-model actions separately. An arx_model
    result carrying room_temp_test_plot_df (both refit and tune set this -
    tune's is fit on the winning grid-search hyperparameters, not config's
    static forgetting_factor/ridge) gets its per-room honest-test-vs-
    predicted temperature chart rendered via get_room_temp_test_plot_html,
    exactly as arx-model-refit's own standalone handler does.
    A predict-side result carrying indoor_temp_forecast_df (rc_model)/
    electric_forecast_series/gas_forecast_series (hybrid_heatpump_model,
    arx_model)/room_temp_forecast_df (dict of room name ->
    Series, arx_model only) gets its forecasted
    trajectory rendered via get_forecast_trend_plot_html.

    :param results: {model_key: result_dict_or_None}, as returned by
        refit_enabled_thermal_models/tune_enabled_thermal_models/
        compute_enabled_thermal_forecasts
    :type results: dict
    :param title: The page title for this action, e.g. "<h2>Thermal models refit</h2>"
    :type title: str
    :return: A dictionary containing the graphs and tables in html format
    :rtype: dict
    """
    model_titles = {
        "rc_model": "RC model",
        "hybrid_heatpump_model": "Hybrid heat pump model",
        "arx_model": "ARX model",
    }
    # Keys carrying chart data (rendered separately below) - never dumped
    # into the generic key/value table.
    chart_keys = {
        "room_temp_test_plot_df",
        "indoor_temp_forecast_df",
        "electric_forecast_series",
        "gas_forecast_series",
        "room_temp_forecast_df",
    }
    injection_dict = {"title": title, "subsubtitle0": f"<h4>Models: {', '.join(results.keys())}</h4>"}
    slot = 1
    for model_key, r in results.items():
        heading = model_titles.get(model_key, model_key)
        if r is None:
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading}: no result (see log)</h4>"
            slot += 1
            continue
        table = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>"
            for key, value in r.items()
            if key not in chart_keys
        ) + "</tbody></table>"
        injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading}</h4>"
        injection_dict[f"table{slot}"] = table
        slot += 1
        for room_name, df_plot in r.get("room_temp_test_plot_df", {}).items():
            injection_dict[f"subsubtitle{slot}"] = (
                f"<h4>{heading} - {room_name}: honest held-out test</h4>"
            )
            injection_dict[f"figure_{slot}"] = get_room_temp_test_plot_html(df_plot, room_name)
            slot += 1
        if "indoor_temp_forecast_df" in r:
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading}: forecast</h4>"
            injection_dict[f"figure_{slot}"] = get_forecast_trend_plot_html(
                r["indoor_temp_forecast_df"], "Indoor temperature (°C)"
            )
            slot += 1
        if "electric_forecast_series" in r:
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading}: electric power forecast</h4>"
            injection_dict[f"figure_{slot}"] = get_forecast_trend_plot_html(
                r["electric_forecast_series"].to_frame("forecast"), "Electric power (W)"
            )
            slot += 1
        if "gas_forecast_series" in r:
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading}: gas consumption forecast</h4>"
            injection_dict[f"figure_{slot}"] = get_forecast_trend_plot_html(
                r["gas_forecast_series"].to_frame("forecast"), "Gas consumption (m³)"
            )
            slot += 1
        for room_name, series in r.get("room_temp_forecast_df", {}).items():
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{heading} - {room_name}: forecast</h4>"
            injection_dict[f"figure_{slot}"] = get_forecast_trend_plot_html(
                series.to_frame("forecast"), f"{room_name} temperature (°C)"
            )
            slot += 1
    return injection_dict


def get_injection_dict_thermal_models_fit(
    results: dict, title: str, temperature_winner: str | None = None
) -> dict:
    """Build the webui injection_dict for the thermal-models-refit/-tune
    consolidated actions specifically (NOT -forecast - see
    get_injection_dict_thermal_models above for that, unchanged): ONE
    combined table across every model that ran instead of one table per
    model, and ONE combined chart per physical quantity (temperature/
    electric/gas) overlaying every model+room that has held-out test data
    for it, instead of one chart per model. A forecast result has no
    "actual" to compare against and no held-out test split, so this
    dedicated function only makes sense for refit/tune results - it is
    NOT a drop-in replacement for the function above.

    Relies on each model's refit/tune result already carrying a
    "*_test_plot_df" dict (room/quantity name -> DataFrame with columns
    exactly "train"/"test"/"pred") for whichever quantities it actually
    scored on a held-out split - see refit_rc_model/refit_hybrid_heatpump_model/
    refit_arx_model (and their tune siblings) in command_line.py. A
    quantity/model combination that isn't present (model not enabled,
    fit rejected before an honest test report, or that model doesn't
    predict that quantity at all) simply contributes no columns - no
    empty chart.

    :param results: {model_key: result_dict_or_None}, as returned by
        refit_enabled_thermal_models/tune_enabled_thermal_models.
    :type results: dict
    :param title: The page title for this action, e.g. "<h2>Thermal models refit</h2>"
    :type title: str
    :param temperature_winner: "rc_model"/"arx_model"/None - which of the
        two temperature-predicting families currently has the better
        held-out accuracy in THIS result set, from
        command_line._compare_temperature_model_accuracy(results.get("rc_model"),
        results.get("arx_model")) - computed by the caller (command_line.py
        imports from utils.py, not the other way around, so this function
        takes the already-decided answer rather than importing the
        comparison itself). None when fewer than both ran - no winner is
        shown, rather than declaring one against nothing.
    :type temperature_winner: str | None
    :return: A dictionary containing the graph(s) and table in html format
    :rtype: dict
    """
    model_titles = {
        "rc_model": "RC model",
        "hybrid_heatpump_model": "Hybrid heat pump model",
        "arx_model": "ARX model",
    }
    chart_keys = {"room_temp_test_plot_df", "electric_test_plot_df", "gas_test_plot_df"}

    def _label(model_key: str) -> str:
        heading = model_titles.get(model_key, model_key)
        return f"{heading} (current winner)" if model_key == temperature_winner else heading

    injection_dict = {"title": title}
    slot = 0
    if temperature_winner is not None:
        injection_dict[f"subsubtitle{slot}"] = (
            f"<h4>Temperature model comparison: {model_titles.get(temperature_winner, temperature_winner)} "
            "currently has the better held-out accuracy</h4>"
        )
        slot += 1

    # One combined table - "Model | Metric | Value" long format, since the
    # three models' own metric sets barely overlap (RC has val_mae_c/
    # relabel_source, hybrid has electric_mae_w/electric_only, ARX has a
    # per-room room_temp_mae_c dict) - a wide table would mean mostly-empty
    # cells; this sidesteps that entirely, and is a direct generalization
    # of each model's own former single-model key/value table.
    rows = []
    for model_key, r in results.items():
        label = _label(model_key)
        if r is None:
            rows.append(f"<tr><td>{label}</td><td>-</td><td>no result (see log)</td></tr>")
            continue
        for key, value in r.items():
            if key in chart_keys:
                continue
            rows.append(f"<tr><td>{label}</td><td>{key}</td><td>{value}</td></tr>")
    injection_dict[f"subsubtitle{slot}"] = "<h4>Fit summary - all models</h4>"
    injection_dict[f"table{slot}"] = (
        "<table class='mystyle'><thead><tr><th>Model</th><th>Metric</th><th>Value</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    slot += 1

    # One combined chart per quantity - outer-join every present model's
    # own train/test/pred columns (each column prefixed with its model[/
    # room] label so they don't collide), reusing get_forecast_trend_plot_html
    # directly (already generalized beyond a fixed 3-column shape, and
    # otherwise identical to get_room_temp_test_plot_html - both are just
    # df_plot.plot() + the same layout tweaks).
    def _combined_chart(chart_key: str) -> pd.DataFrame | None:
        combined: pd.DataFrame | None = None
        for model_key, r in results.items():
            if not r:
                continue
            for name, df_plot in r.get(chart_key, {}).items():
                prefix = _label(model_key) if name == "house" else f"{_label(model_key)} - {name}"
                renamed = df_plot.rename(columns={c: f"{prefix} {c}" for c in df_plot.columns})
                combined = renamed if combined is None else combined.join(renamed, how="outer")
        return combined

    for chart_key, chart_title, y_axis in (
        ("room_temp_test_plot_df", "Temperature - honest held-out test", "Temperature (°C)"),
        ("electric_test_plot_df", "Electric power - honest held-out test", "Electric power (W)"),
        ("gas_test_plot_df", "Gas consumption - honest held-out test", "Gas consumption (m³)"),
    ):
        combined = _combined_chart(chart_key)
        if combined is not None:
            injection_dict[f"subsubtitle{slot}"] = f"<h4>{chart_title}</h4>"
            injection_dict[f"figure_{slot}"] = get_forecast_trend_plot_html(combined, y_axis)
            slot += 1

    return injection_dict


async def build_config(
    emhass_conf: dict,
    logger: logging.Logger,
    defaults_path: str,
    config_path: str | None = None,
    legacy_config_path: str | None = None,
) -> dict:
    """
    Retrieve parameters from configuration files.
    priority order (low - high) = defaults_path, config_path legacy_config_path

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param defaults_path: path to config file for parameter defaults (config_defaults.json)
    :type defaults_path: str
    :param config_path: path to the main configuration file (config.json)
    :type config_path: str
    :param legacy_config_path: path to legacy config file (config_emhass.yaml)
    :type legacy_config_path: str
    :return: The built config dictionary
    :rtype: dict
    """

    # Read default parameters (default root_path/data/config_defaults.json)
    if defaults_path and pathlib.Path(defaults_path).is_file():
        async with aiofiles.open(defaults_path) as data:
            content = await data.read()
            config = orjson.loads(content)
    else:
        logger.error("config_defaults.json. does not exist ")
        return False

    # Read user config parameters if provided (default /share/config.json)
    if config_path and pathlib.Path(config_path).is_file():
        async with aiofiles.open(config_path) as data:
            content = await data.read()
            # Set override default parameters (config_defaults) with user given parameters (config.json)
            logger.info("Obtaining parameters from config.json:")
            config.update(orjson.loads(content))
    else:
        logger.info(
            "config.json does not exist, or has not been passed. config parameters may default to config_defaults.json"
        )
        logger.info("you may like to generate the config.json file on the configuration page")

    # Check to see if legacy config_emhass.yaml was provided (default /app/config_emhass.yaml)
    # Convert legacy parameter definitions/format to match config.json
    if legacy_config_path and pathlib.Path(legacy_config_path).is_file():
        async with aiofiles.open(legacy_config_path) as data:
            content = await data.read()
            legacy_config = yaml.safe_load(content)
            legacy_config_parameters = await build_legacy_config_params(
                emhass_conf, legacy_config, logger
            )
            if type(legacy_config_parameters) is not bool:
                logger.info(
                    "Obtaining parameters from config_emhass.yaml: (will overwrite config parameters)"
                )
                config.update(legacy_config_parameters)

    return config


async def build_legacy_config_params(
    emhass_conf: dict[str, pathlib.Path],
    legacy_config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, str]:
    """
    Build a config dictionary with legacy config_emhass.yaml file.
    Uses the associations file to convert parameter naming conventions (to config.json/config_defaults.json).
    Extracts the parameter values and formats to match config.json.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param legacy_config: The legacy config dictionary
    :type legacy_config: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict
    """

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)

    # Check each config catagories exists, else create blank dict for categories (avoid errors)
    legacy_config["retrieve_hass_conf"] = legacy_config.get("retrieve_hass_conf", {})
    legacy_config["optim_conf"] = legacy_config.get("optim_conf", {})
    legacy_config["plant_conf"] = legacy_config.get("plant_conf", {})
    config = {}

    # Use associations list to map legacy parameter name with config.json parameter name
    if emhass_conf["associations_path"].exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Cant find associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Loop through all parameters in association file
    # Append config with existing legacy config parameters (converting alternative parameter naming conventions with associations list)
    for association in associations:
        # if legacy config catagories exists and if legacy parameter exists in config catagories
        if (
            legacy_config.get(association[0]) is not None
            and legacy_config[association[0]].get(association[1], None) is not None
        ):
            config[association[2]] = legacy_config[association[0]][association[1]]

            # If config now has load_peak_hour_periods, extract from list of dict
            if association[2] == "load_peak_hour_periods" and type(config[association[2]]) is list:
                config[association[2]] = {key: d[key] for d in config[association[2]] for key in d}

    return config


def get_keys_to_mask() -> list[str]:
    """
    Return a list of sensitive configuration keys that should be masked in logs
    or treated specially in the UI (e.g., secrets).
    """
    return [
        "influxdb_username",
        "influxdb_password",
        "solcast_api_key",
        "solcast_rooftop_id",
        "long_lived_token",
        "time_zone",
        "Latitude",
        "Longitude",
        "Altitude",
        "hass_url",  # Ensure this is included if you want it masked everywhere
        "solar_forecast_kwp",  # Ensure this is included if you want it masked everywhere
    ]


def param_to_config(param: dict[str, dict], logger: logging.Logger) -> dict[str, str]:
    """
    A function that extracts the parameters from param back to the config.json format.
    Extracts parameters from config catagories.
    Attempts to exclude secrets hosed in retrieve_hass_conf.

    :param params: Built configuration parameters
    :type param: dict[str, dict]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built config dictionary
    :rtype: dict[str, str]
    """
    logger.debug("Converting param to config")

    return_config = {}

    config_categories = ["retrieve_hass_conf", "optim_conf", "plant_conf"]
    secret_params = get_keys_to_mask()

    # Loop through config catagories that contain config params, and extract
    for config in config_categories:
        for parameter in param[config]:
            # If parameter is not a secret, append to return_config
            if parameter not in secret_params:
                return_config[str(parameter)] = param[config][parameter]

    return return_config


async def build_secrets(
    emhass_conf: dict[str, pathlib.Path],
    logger: logging.Logger,
    argument: dict[str, str] | None = None,
    options_path: str | None = None,
    secrets_path: str | None = None,
    no_response: bool = False,
) -> tuple[dict[str, pathlib.Path], dict[str, str | float]]:
    """
    Retrieve and build parameters from secrets locations (ENV, ARG, Secrets file (secrets_emhass.yaml/options.json) and/or Home Assistant (via API))
    priority order (lwo to high) = Defaults (written in function), ENV, Options json file, Home Assistant API,  Secrets yaml file, Arguments

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :param argument: dictionary of secrets arguments passed (url,key)
    :type argument: dict
    :param options_path: path to the options file (options.json) (usually provided by EMHASS-Add-on)
    :type options_path: str
    :param secrets_path: path to secrets file (secrets_emhass.yaml)
    :type secrets_path: str
    :param no_response: bypass get request to Home Assistant (json response errors)
    :type no_response: bool
    :return: Updated emhass_conf, the built secrets dictionary
    :rtype: Tuple[dict, dict]:
    """
    # Set defaults to be overwritten
    if argument is None:
        argument = {}
    params_secrets = {
        "hass_url": "https://myhass.duckdns.org/",
        "long_lived_token": "thatverylongtokenhere",
        "time_zone": "Europe/Paris",
        "Latitude": 45.83,
        "Longitude": 6.86,
        "Altitude": 4807.8,
        "solcast_api_key": "yoursecretsolcastapikey",
        "solcast_rooftop_id": "yourrooftopid",
        "solar_forecast_kwp": 5,
        "influxdb_username": "yourinfluxdbusername",
        "influxdb_password": "yourinfluxdbpassword",
    }

    # Obtain Secrets from ENV?
    params_secrets["hass_url"] = os.getenv("EMHASS_URL", params_secrets["hass_url"])
    params_secrets["long_lived_token"] = os.getenv(
        "SUPERVISOR_TOKEN", params_secrets["long_lived_token"]
    )
    params_secrets["time_zone"] = os.getenv("TIME_ZONE", params_secrets["time_zone"])
    params_secrets["Latitude"] = float(os.getenv("LAT", params_secrets["Latitude"]))
    params_secrets["Longitude"] = float(os.getenv("LON", params_secrets["Longitude"]))
    params_secrets["Altitude"] = float(os.getenv("ALT", params_secrets["Altitude"]))

    # Obtain secrets from options.json (Generated from EMHASS-Add-on, Home Assistant addon Configuration page) or Home Assistant API (from local Supervisor API)?
    # Use local supervisor API to obtain secrets from Home Assistant if hass_url in options.json is empty and SUPERVISOR_TOKEN ENV exists (provided by Home Assistant when running the container as addon)
    options = {}
    if options_path and pathlib.Path(options_path).is_file():
        async with aiofiles.open(options_path) as data:
            content = await data.read()
            options = orjson.loads(content)

            # Obtain secrets from Home Assistant?
            url_from_options = options.get("hass_url", "empty")
            key_from_options = options.get("long_lived_token", "empty")

            # If data path specified by options.json, overwrite emhass_conf['data_path']
            data_path_value = options.get("data_path", None)
            if (
                data_path_value is not None
                and data_path_value != ""
                and data_path_value != "default"
            ):
                # Try to create directory if it doesn't exist. if successful set data_path in emhass_conf
                try:
                    data_path = pathlib.Path(data_path_value)
                    # Use parents=True to create nested directories
                    data_path.mkdir(parents=True, exist_ok=True)
                    emhass_conf["data_path"] = data_path
                    logger.info(f"Using custom data_path: {data_path}")
                except Exception as e:
                    logger.warning(
                        f"Cannot create data_path directory '{data_path_value}' provided via options. Keeping default. Error: {e}"
                    )

            # If config path specified by options.json, overwrite emhass_conf['config_path']
            config_path_value = options.get("config_path", None)
            if (
                config_path_value is not None
                and config_path_value != ""
                and config_path_value != "default"
            ):
                try:
                    config_path = pathlib.Path(config_path_value)
                    # Validate that the config file or its parent directory path is valid
                    if config_path.exists():
                        # File exists - use it
                        emhass_conf["config_path"] = config_path
                        logger.info(f"Using custom config_path from addon settings: {config_path}")
                    elif config_path.parent.exists():
                        # Parent directory exists but file doesn't - set path anyway (file may be created later)
                        emhass_conf["config_path"] = config_path
                        logger.warning(
                            f"Config file does not exist yet: {config_path} (will use defaults until created)"
                        )
                    else:
                        # Neither file nor parent directory exists - this is likely an error
                        logger.error(
                            f"Invalid config_path '{config_path_value}': parent directory does not exist. Keeping default config_path."
                        )
                except Exception as e:
                    logger.warning(
                        f"Cannot set config_path '{config_path_value}' provided via options. Keeping default. Error: {e}"
                    )
            else:
                # No config path provided via options.json, check default and legacy paths
                # This will move the config file to the addon_config_path if it is found in the legacy location.
                logger.info(
                    "No config_path provided via options.json, checking default (/config/config.json) and legacy path (/share/config.json)."
                )
                default_config_path = pathlib.Path("/config/config.json")
                default_config_dir_exists = default_config_path.parent.exists()
                legacy_config_path = pathlib.Path("/share/config.json")

                if default_config_path.is_file():
                    logger.info("Found config.json in /config, using this path for config_path.")
                    emhass_conf["config_path"] = default_config_path
                elif legacy_config_path.is_file():
                    # found legacy config path, move the file to the default addon-mode config path and use it for config_path
                    if default_config_dir_exists:
                        try:
                            shutil.move(str(legacy_config_path), str(default_config_path))
                            logger.info(
                                f"Moved legacy config from {legacy_config_path} to {default_config_path} and using it for config_path."
                            )
                            emhass_conf["config_path"] = default_config_path
                        except Exception as e:
                            logger.warning(
                                f"Failed to move legacy config from {legacy_config_path} to {default_config_path}: {e}"
                            )
                            emhass_conf["config_path"] = legacy_config_path
                    else:
                        logger.warning(
                            f"Directory {default_config_path.parent} does not exist, keeping legacy config_path: {legacy_config_path}."
                        )
                        emhass_conf["config_path"] = legacy_config_path
                elif default_config_dir_exists:
                    logger.info(
                        "No legacy config.json found in /share, using addon-mode default /config/config.json for config_path."
                    )
                    emhass_conf["config_path"] = default_config_path
                else:
                    logger.warning(
                        f"Directory {default_config_path.parent} does not exist, keeping current legacy default"
                    )

            # Check to use Home Assistant local API
            if not no_response and os.getenv("SUPERVISOR_TOKEN", None) is not None:
                params_secrets["long_lived_token"] = os.getenv("SUPERVISOR_TOKEN", None)
                # Use hass_url from options.json if available, otherwise use supervisor API for addon
                if url_from_options != "empty" and url_from_options != "":
                    params_secrets["hass_url"] = url_from_options
                else:
                    # For addons, use supervisor API for both REST and WebSocket access
                    params_secrets["hass_url"] = "http://supervisor/core/api"
                headers = {
                    "Authorization": "Bearer " + params_secrets["long_lived_token"],
                    "content-type": "application/json",
                }
                # Obtain secrets from Home Assistant via API
                logger.debug("Obtaining secrets from Home Assistant Supervisor API")
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        params_secrets["hass_url"] + "/config", headers=headers
                    ) as response:
                        if response.status < 400:
                            config_hass = await response.json()
                            params_secrets.update(
                                {
                                    "hass_url": params_secrets["hass_url"],
                                    "long_lived_token": params_secrets["long_lived_token"],
                                    "time_zone": config_hass["time_zone"],
                                    "Latitude": config_hass["latitude"],
                                    "Longitude": config_hass["longitude"],
                                    "Altitude": config_hass["elevation"],
                                    # If defined in HA config, use them, otherwise keep defaults
                                    "solcast_api_key": config_hass.get(
                                        "solcast_api_key", params_secrets["solcast_api_key"]
                                    ),
                                    "solcast_rooftop_id": config_hass.get(
                                        "solcast_rooftop_id", params_secrets["solcast_rooftop_id"]
                                    ),
                                    "solar_forecast_kwp": config_hass.get(
                                        "solar_forecast_kwp", params_secrets["solar_forecast_kwp"]
                                    ),
                                    "influxdb_username": config_hass.get(
                                        "influxdb_username", params_secrets.get("influxdb_username")
                                    ),
                                    "influxdb_password": config_hass.get(
                                        "influxdb_password", params_secrets.get("influxdb_password")
                                    ),
                                }
                            )
                        else:
                            # Obtain the url and key secrets if any from options.json (default /app/options.json)
                            logger.warning(
                                "Error obtaining secrets from Home Assistant Supervisor API"
                            )
                            logger.debug("Obtaining url and key secrets from options.json")
                            if url_from_options != "empty" and url_from_options != "":
                                params_secrets["hass_url"] = url_from_options
                            if key_from_options != "empty" and key_from_options != "":
                                params_secrets["long_lived_token"] = key_from_options
                            if (
                                options.get("time_zone", "empty") != "empty"
                                and options["time_zone"] != ""
                            ):
                                params_secrets["time_zone"] = options["time_zone"]
                            if options.get("Latitude", None) is not None and bool(
                                options["Latitude"]
                            ):
                                params_secrets["Latitude"] = options["Latitude"]
                            if options.get("Longitude", None) is not None and bool(
                                options["Longitude"]
                            ):
                                params_secrets["Longitude"] = options["Longitude"]
                            if options.get("Altitude", None) is not None and bool(
                                options["Altitude"]
                            ):
                                params_secrets["Altitude"] = options["Altitude"]

            # Obtain the forecast secrets (if any) from options.json (default /app/options.json)
            # This logic runs regardless of whether HA API call above succeeded or failed,
            # so we removed the duplicate logic from the 'else' block above.
            forecast_secrets = [
                "solcast_api_key",
                "solcast_rooftop_id",
                "solar_forecast_kwp",
            ]
            if any(x in forecast_secrets for x in list(options.keys())):
                logger.debug("Obtaining forecast secrets from options.json")
                if (
                    options.get("solcast_api_key", "empty") != "empty"
                    and options["solcast_api_key"] != ""
                ):
                    params_secrets["solcast_api_key"] = options["solcast_api_key"]
                if (
                    options.get("solcast_rooftop_id", "empty") != "empty"
                    and options["solcast_rooftop_id"] != ""
                ):
                    params_secrets["solcast_rooftop_id"] = options["solcast_rooftop_id"]
                if options.get("solar_forecast_kwp", None) and bool(options["solar_forecast_kwp"]):
                    params_secrets["solar_forecast_kwp"] = options["solar_forecast_kwp"]

            # Obtain InfluxDB secrets from options.json
            influx_secrets = ["influxdb_username", "influxdb_password"]
            if any(x in influx_secrets for x in list(options.keys())):
                logger.debug("Obtaining InfluxDB secrets from options.json")
                if (
                    options.get("influxdb_username", "empty") != "empty"
                    and options["influxdb_username"] != ""
                ):
                    params_secrets["influxdb_username"] = options["influxdb_username"]
                if (
                    options.get("influxdb_password", "empty") != "empty"
                    and options["influxdb_password"] != ""
                ):
                    params_secrets["influxdb_password"] = options["influxdb_password"]

    # Obtain secrets from secrets_emhass.yaml? (default /app/secrets_emhass.yaml)
    if secrets_path and pathlib.Path(secrets_path).is_file():
        logger.debug("Obtaining secrets from secrets file")
        async with aiofiles.open(pathlib.Path(secrets_path)) as file:
            content = await file.read()
            params_secrets.update(yaml.safe_load(content))

    # Receive key and url from ARG/arguments?
    if argument.get("url") is not None:
        params_secrets["hass_url"] = argument["url"]
        logger.debug("Obtaining url from passed argument")
    if argument.get("key") is not None:
        params_secrets["long_lived_token"] = argument["key"]
        logger.debug("Obtaining long_lived_token from passed argument")

    return emhass_conf, params_secrets


async def build_params(
    emhass_conf: dict[str, pathlib.Path],
    params_secrets: dict[str, str | float],
    config: dict[str, str],
    logger: logging.Logger,
) -> dict[str, dict]:
    """
    Build the main params dictionary from the config and secrets
    Appends configuration catagories used by emhass to the parameters. (with use of the associations file as a reference)

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict[str, pathlib.Path]
    :param params_secrets: The dictionary containing the built secret variables
    :type params_secrets: dict[str, str | float]
    :param config: The dictionary of built config parameters
    :type config: dict[str, str]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: The built param dictionary
    :rtype: dict[str, dict]
    """
    if not isinstance(params_secrets, dict):
        params_secrets = {}

    params = {}
    # Start with blank config catagories
    params["retrieve_hass_conf"] = {}
    params["params_secrets"] = {}
    params["optim_conf"] = {}
    params["plant_conf"] = {}

    # Obtain associations to categorize parameters to their corresponding config catagories
    if emhass_conf.get(
        "associations_path", get_root(__file__, num_parent=2) / "data/associations.csv"
    ).exists():
        async with aiofiles.open(emhass_conf["associations_path"]) as data:
            content = await data.read()
            associations = list(csv.reader(content.splitlines(), delimiter=","))
    else:
        logger.error(
            "Unable to obtain the associations file (associations.csv) in: "
            + str(emhass_conf["associations_path"])
        )
        return False

    # Association file key reference
    # association[0] = config catagories
    # association[1] = legacy parameter name
    # association[2] = parameter (config.json/config_defaults.json)
    # association[3] = parameter list name if exists (not used, from legacy options.json)
    # Use association list to append parameters from config into params (with corresponding config catagories)
    for association in associations:
        # If parameter has list_ name and parameter in config is presented with its list name
        # (ie, config parameter is in legacy options.json format)
        if len(association) == 4 and config.get(association[3]) is not None:
            # Extract lists of dictionaries
            if config[association[3]] and type(config[association[3]][0]) is dict:
                params[association[0]][association[2]] = [
                    i[association[2]] for i in config[association[3]]
                ]
            else:
                params[association[0]][association[2]] = config[association[3]]
        # Else, directly set value of config parameter to param
        elif config.get(association[2]) is not None:
            params[association[0]][association[2]] = config[association[2]]

    # def_load_config isn't a normal user-facing config field - no
    # associations.csv row, no param_definitions.json UI widget - it's an
    # internal, derived structure that _append_ev_deferrable_loads/
    # _append_room_thermal_loads/_append_boiler_thermal_battery_loads
    # below build from OTHER config (room/EV/boiler names etc.) and tag
    # with a _source marker so repeated build_params calls can tell their
    # own previously-appended entries apart from manually-declared ones
    # (see _strip_auto_appended_loads). Without this explicit passthrough
    # those _source markers never survive a config.json round trip (the
    # generic association loop above only copies fields it recognizes),
    # so _strip_auto_appended_loads would always see an empty list and
    # every save-page round trip would silently double the EV/room/boiler
    # entries - a real, confirmed bug this passthrough fixes together with
    # the strip calls in the three functions below.
    if config.get("def_load_config") is not None:
        params["optim_conf"]["def_load_config"] = config["def_load_config"]
    _prune_orphaned_deferrable_load_slots(params["optim_conf"], logger)

    # Check if we need to create `list_hp_periods` from config (ie. legacy options.json format)
    if (
        params.get("optim_conf") is not None
        and config.get("list_peak_hours_periods_start_hours") is not None
        and config.get("list_peak_hours_periods_end_hours") is not None
    ):
        start_hours_list = [
            i["peak_hours_periods_start_hours"]
            for i in config["list_peak_hours_periods_start_hours"]
        ]
        end_hours_list = [
            i["peak_hours_periods_end_hours"] for i in config["list_peak_hours_periods_end_hours"]
        ]
        num_peak_hours = len(start_hours_list)
        list_hp_periods_list = {
            "period_hp_" + str(i + 1): [
                {"start": start_hours_list[i]},
                {"end": end_hours_list[i]},
            ]
            for i in range(num_peak_hours)
        }
        params["optim_conf"]["load_peak_hour_periods"] = list_hp_periods_list
    else:
        # Else, check param already contains load_peak_hour_periods from config
        if params["optim_conf"].get("load_peak_hour_periods", None) is None:
            logger.warning("Unable to detect or create load_peak_hour_periods parameter")

    # Format load_peak_hour_periods list to dict if necessary
    if params["optim_conf"].get("load_peak_hour_periods", None) is not None and isinstance(
        params["optim_conf"]["load_peak_hour_periods"], list
    ):
        params["optim_conf"]["load_peak_hour_periods"] = {
            key: d[key] for d in params["optim_conf"]["load_peak_hour_periods"] for key in d
        }

    # Call function to check parameter lists that require the same length as deferrable loads
    # If not, set defaults it fill in gaps
    if params["optim_conf"].get("number_of_deferrable_loads", None) is not None:
        num_def_loads = params["optim_conf"]["number_of_deferrable_loads"]
        if params["optim_conf"].get("treat_deferrable_load_as_semi_cont", None) is None:
            load_types = params["optim_conf"].get("load_type", [])
            if isinstance(load_types, list) and len(load_types) > 0:
                # Derive semi-continuous behavior from load_type when the explicit field is absent.
                params["optim_conf"]["treat_deferrable_load_as_semi_cont"] = [
                    not (
                        load_type == "fixed_power_splittable"
                        or load_type == "variable_power_variable_time"
                    )
                    for load_type in load_types
                ]
            else:
                params["optim_conf"]["treat_deferrable_load_as_semi_cont"] = [True] * num_def_loads
        # Looped over DEF_LOAD_ARRAY_PARAMS (name -> default) instead of 9
        # repeated calls (#1040) - same order, same defaults, same call
        # signature per entry, so behaviour is unchanged.
        for def_array_name, def_array_default in DEF_LOAD_ARRAY_PARAMS.items():
            params["optim_conf"][def_array_name] = check_def_loads(
                num_def_loads,
                params["optim_conf"],
                def_array_default,
                def_array_name,
                logger,
            )
        # Validate deferrable_load_groups
        groups = params["optim_conf"].get("deferrable_load_groups", [])
        if groups:
            seen_indices = set()
            for gi, group in enumerate(groups):
                # Validate names
                names = group.get("names", [])
                if not names:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'names' must contain at least 1 deferrable load reference"
                    )
                indices = []
                group_indices = set()
                for name in names:
                    try:
                        idx = int(name.replace("deferrable", ""))
                    except ValueError as err:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: could not parse index from name '{name}'"
                        ) from err
                    if idx < 0 or idx >= num_def_loads:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' references index {idx}, "
                            f"but only {num_def_loads} deferrable loads are configured"
                        )
                    if idx in group_indices:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' is duplicated within the group"
                        )
                    if idx in seen_indices:
                        raise ValueError(
                            f"deferrable_load_groups[{gi}]: '{name}' is already in another group. "
                            f"A deferrable load cannot belong to multiple groups"
                        )
                    indices.append(idx)
                    group_indices.add(idx)
                seen_indices.update(indices)

                # Validate max_power (optional when mutual_exclusion is true)
                max_power = group.get("max_power")
                mutual_exclusion = group.get("mutual_exclusion", False)
                if max_power is not None and max_power <= 0:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'max_power' must be a positive number"
                    )
                if not isinstance(mutual_exclusion, bool):
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'mutual_exclusion' must be a boolean"
                    )
                if max_power is None and not mutual_exclusion:
                    raise ValueError(
                        f"deferrable_load_groups[{gi}]: 'max_power' is required when 'mutual_exclusion' is false"
                    )
        _normalize_deferrable_load_categories(params, logger)
        await _append_boiler_thermal_battery_loads(params, logger, emhass_conf)
        await _append_room_thermal_loads(params, logger, emhass_conf)
        await _append_ev_deferrable_loads(params, logger)
        await _resolve_manual_committed_loads(params, logger)
    else:
        logger.warning("unable to obtain parameter: number_of_deferrable_loads")

    _append_rc_model_forecast_targets(params, logger)
    _append_hybrid_heatpump_forecast_targets(params, logger)
    _append_arx_model_forecast_targets(params, logger)

    # Normalise per-battery array params against number_of_batteries (#610).
    # Missing key defaults to 1 (single-battery, the only shape supported
    # before this parameter existed); N=1 is a true no-op (see
    # check_batt_params docstring).
    num_batteries = validate_num_batteries(params["plant_conf"])
    for batt_param_name, batt_default in BATT_ARRAY_PARAMS_PLANT_CONF.items():
        check_batt_params(
            num_batteries, params["plant_conf"], batt_default, batt_param_name, logger
        )
    for batt_param_name, batt_default in BATT_ARRAY_PARAMS_OPTIM_CONF.items():
        check_batt_params(
            num_batteries, params["optim_conf"], batt_default, batt_param_name, logger
        )
    for batt_param_name in BATT_WEIGHT_PARAMS:
        check_batt_weight_params(num_batteries, params["optim_conf"], batt_param_name, logger)

    # Normalise the per-phase array params (number_of_phases, #phase-balance)
    # the same way, right after the per-battery pass above.
    num_phases = validate_num_phases(params["plant_conf"], logger)
    check_phase_array_params(num_batteries, params["plant_conf"], "", "battery_phase", logger)
    check_phase_array_params(
        num_phases, params["plant_conf"], 4000, "maximum_power_from_grid_per_phase", logger
    )
    check_phase_array_params(
        num_phases, params["plant_conf"], 4000, "maximum_power_to_grid_per_phase", logger
    )

    # historic_days_to_retrieve should be no less then 2
    if params["retrieve_hass_conf"].get("historic_days_to_retrieve", None) is not None:
        if params["retrieve_hass_conf"]["historic_days_to_retrieve"] < 2:
            params["retrieve_hass_conf"]["historic_days_to_retrieve"] = 2
            logger.warning(
                "days_to_retrieve should not be lower then 2, setting days_to_retrieve to 2. Make sure your sensors also have at least 2 days of history"
            )
    else:
        logger.warning("unable to obtain parameter: historic_days_to_retrieve")

    # Configure secrets, set params to correct config categorie
    # retrieve_hass_conf
    params["retrieve_hass_conf"]["hass_url"] = params_secrets.get("hass_url")
    params["retrieve_hass_conf"]["long_lived_token"] = params_secrets.get("long_lived_token")
    params["retrieve_hass_conf"]["time_zone"] = params_secrets.get("time_zone")
    params["retrieve_hass_conf"]["Latitude"] = params_secrets.get("Latitude")
    params["retrieve_hass_conf"]["Longitude"] = params_secrets.get("Longitude")
    params["retrieve_hass_conf"]["Altitude"] = params_secrets.get("Altitude")
    if params_secrets.get("influxdb_username") is not None:
        params["retrieve_hass_conf"]["influxdb_username"] = params_secrets.get("influxdb_username")
        params["params_secrets"]["influxdb_username"] = params_secrets.get("influxdb_username")
    if params_secrets.get("influxdb_password") is not None:
        params["retrieve_hass_conf"]["influxdb_password"] = params_secrets.get("influxdb_password")
        params["params_secrets"]["influxdb_password"] = params_secrets.get("influxdb_password")
    # Update optional param secrets
    if params["optim_conf"].get("weather_forecast_method", None) is not None:
        if params["optim_conf"]["weather_forecast_method"] == "solcast":
            params["retrieve_hass_conf"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["params_secrets"]["solcast_api_key"] = params_secrets.get(
                "solcast_api_key", "123456"
            )
            params["retrieve_hass_conf"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
            params["params_secrets"]["solcast_rooftop_id"] = params_secrets.get(
                "solcast_rooftop_id", "123456"
            )
        elif params["optim_conf"]["weather_forecast_method"] == "solar.forecast":
            params["retrieve_hass_conf"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
            params["params_secrets"]["solar_forecast_kwp"] = params_secrets.get(
                "solar_forecast_kwp", 5
            )
    else:
        logger.warning("Unable to detect weather_forecast_method parameter")
    #  Check if secrets parameters still defaults values
    secret_params = [
        "https://myhass.duckdns.org/",
        "thatverylongtokenhere",
        45.83,
        6.86,
        4807.8,
    ]
    if any(x in secret_params for x in params["retrieve_hass_conf"].values()):
        logger.warning("Some secret parameters values are still matching their defaults")

    # Set empty placeholders for the runtime-override keys in params
    # passed_data (to be later populated with runtime parameters via
    # treat_runtimeparams). This merges rather than replaces passed_data,
    # so it doesn't discard entries already computed earlier in this
    # function - e.g. the default custom_deferrable_forecast_id /
    # custom_predicted_temperature_id / custom_heating_demand_id built
    # above, or the room_load_indices / heatpump_dispatch_load_index /
    # ev_load_indices / custom_*_target_id bookkeeping added by
    # _append_room_thermal_loads / _append_ev_deferrable_loads.
    params.setdefault("passed_data", {})
    params["passed_data"].update(
        {
            "pv_power_forecast": None,
            "load_power_forecast": None,
            "load_cost_forecast": None,
            "prod_price_forecast": None,
            "prediction_horizon": None,
            "soc_init": None,
            "soc_final": None,
            "soc_target": None,
            "soc_target_timestep": None,
            "current_period_peak": None,
            "operating_hours_of_each_deferrable_load": None,
            "start_timesteps_of_each_deferrable_load": None,
            "end_timesteps_of_each_deferrable_load": None,
            "alpha": None,
            "beta": None,
        }
    )

    return params


# Per-deferrable-load ARRAY parameters normalised against
# number_of_deferrable_loads via check_def_loads (#929, #1040). build_params
# loops over this table to pad each one from config; treat_runtimeparams's
# re-normalisation pass reuses it to re-apply the same padding after a
# runtime override may have replaced one of these arrays with a raw,
# potentially short, value (the per-battery arrays already get an equivalent
# pass via check_batt_params, #610).
DEF_LOAD_ARRAY_PARAMS: dict[str, bool | int | float | str] = {
    "start_timesteps_of_each_deferrable_load": 0,
    "end_timesteps_of_each_deferrable_load": 0,
    "set_deferrable_load_single_constant": False,
    "treat_deferrable_load_as_semi_cont": True,
    "set_deferrable_startup_penalty": 0.0,
    "deferrable_load_max_cost": 0.0,
    "set_deferrable_max_startups": 0,
    "operating_hours_of_each_deferrable_load": 0,
    "nominal_power_of_deferrable_loads": 0,
    "is_manual_load": False,
    "manual_load_deadline_hour": "",
    "load_washdata_enabled": False,
    "load_washdata_device": "",
    "load_phase": "",
}
# Legacy (pre-#342) names for the same 9 arrays, from
# src/emhass/data/associations.csv column 2. The association loop accepts
# either the modern name (column 3) or this legacy one, so the
# runtime-provided check in treat_runtimeparams has to match both.
DEF_LOAD_ARRAY_LEGACY_NAMES: dict[str, str] = {
    "start_timesteps_of_each_deferrable_load": "def_start_timestep",
    "end_timesteps_of_each_deferrable_load": "def_end_timestep",
    "set_deferrable_load_single_constant": "set_def_constant",
    "treat_deferrable_load_as_semi_cont": "treat_def_as_semi_cont",
    "set_deferrable_startup_penalty": "def_start_penalty",
    "deferrable_load_max_cost": "deferrable_load_max_cost",
    "set_deferrable_max_startups": "set_deferrable_max_startups",
    "operating_hours_of_each_deferrable_load": "def_total_hours",
    "nominal_power_of_deferrable_loads": "P_deferrable_nom",
}

# The exact 12 per-load arrays every one of _append_ev_deferrable_loads/
# _append_room_thermal_loads/_append_boiler_thermal_battery_loads's own
# base_default_lists dict pads and appends to (see each function) - kept
# as one shared tuple so _strip_auto_appended_loads below filters exactly
# the arrays those three functions actually mutate, not the broader/
# unrelated DEF_LOAD_ARRAY_PARAMS set above (used elsewhere for runtime-
# params normalization).
_AUTO_LOAD_ARRAY_KEYS = (
    "nominal_power_of_deferrable_loads",
    "minimum_power_of_deferrable_loads",
    "operating_hours_of_each_deferrable_load",
    "start_timesteps_of_each_deferrable_load",
    "end_timesteps_of_each_deferrable_load",
    "set_deferrable_startup_penalty",
    "set_deferrable_load_single_constant",
    "treat_deferrable_load_as_semi_cont",
    "load_type",
    "load_dispatch_mode",
    "required_energy_kwh_of_each_deferrable_load",
    "load_phase",
)

# Every remaining per-load array the "Deferrable Loads" config-page section
# drives (configuration_script.js's own setupIndexedSectionTabs targetParamIds
# list for that section) that _AUTO_LOAD_ARRAY_KEYS above doesn't already
# cover - kept as one explicit tuple, same rationale, so
# _prune_orphaned_deferrable_load_slots below can shrink every one of them
# in lockstep with def_load_config when it drops a slot.
_MANUAL_LOAD_ARRAY_KEYS = (
    "load_names",
    "load_washdata_enabled",
    "load_washdata_device",
    "is_manual_load",
    "manual_load_ready_sensor",
    "manual_load_confirm_power_sensor",
    "manual_load_program_select_sensor",
    "manual_load_deadline_hour",
    "load_programs",
    "set_deferrable_max_startups",
    "def_minimum_on_time",
    "def_minimum_off_time",
    "deferrable_load_max_cost",
    "cost_forecast_per_deferrable_load",
)


def _def_load_config_source_marker(entry: object) -> str | None:
    """Shared by _strip_auto_appended_loads and
    _prune_orphaned_deferrable_load_slots: a def_load_config entry's
    _source marker lives either at the top level (EV/boiler) or nested
    under thermal_battery (room/heat-pump-dispatch).
    """
    if not isinstance(entry, dict):
        return None
    if "_source" in entry:
        return entry.get("_source")
    battery = entry.get("thermal_battery")
    return battery.get("_source") if isinstance(battery, dict) else None


def _prune_orphaned_deferrable_load_slots(optim_conf: dict, logger: logging.Logger) -> None:
    """Drop def_load_config slots that are leftover corruption from a
    pre-marker-system save-cycle bug: no name, no recognized _source tag,
    and no configured value anywhere - not one of the three auto-appended
    kinds (those are owned by _strip_auto_appended_loads instead) and not
    one of the user's own named manual loads.

    check_def_loads only ever pads these per-load arrays UP to
    number_of_deferrable_loads, never truncates down, and a normal
    browser Save never sends def_load_config at all (no UI widget for it -
    see build_params's own passthrough above), so once an untagged orphan
    like this exists nothing else in the pipeline ever removes it again;
    it just sits there forever, inflating number_of_deferrable_loads and
    showing up as an empty, unnamed tab. Called once in build_params right
    after the def_load_config passthrough, before the three append
    functions run, so they always compute their own append position from
    an already-clean base.

    Deliberately conservative: only drops a slot that is empty *everywhere*
    it could show real content. Anything with a name, a marker, or any
    configured value is left untouched.

    :param optim_conf: The optimization config dict to clean in place.
    :type optim_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    """
    def_load_cfg = optim_conf.get("def_load_config")
    if not isinstance(def_load_cfg, list) or not def_load_cfg:
        return

    load_names = optim_conf.get("load_names", [])
    named_count = len(load_names) if isinstance(load_names, list) else 0
    recognized_markers = {"room_auto", "heatpump_dispatch_auto", "ev_auto", "boiler_auto"}
    zero_signal_keys = (
        "nominal_power_of_deferrable_loads",
        "is_manual_load",
        "load_washdata_enabled",
        "required_energy_kwh_of_each_deferrable_load",
        "operating_hours_of_each_deferrable_load",
    )

    def _is_empty_orphan(i: int) -> bool:
        if i < named_count:
            return False
        if _def_load_config_source_marker(def_load_cfg[i]) in recognized_markers:
            return False
        if def_load_cfg[i]:
            return False
        for key in zero_signal_keys:
            values = optim_conf.get(key)
            if isinstance(values, list) and i < len(values) and values[i]:
                return False
        return True

    keep = [not _is_empty_orphan(i) for i in range(len(def_load_cfg))]
    if all(keep):
        return

    logger.warning(
        f"Pruning {keep.count(False)} orphaned/empty deferrable-load slot(s) "
        "with no name, no source tag and no configured value - a leftover "
        "from a previous save, safe to drop."
    )
    optim_conf["def_load_config"] = [e for e, k in zip(def_load_cfg, keep) if k]
    for key in _AUTO_LOAD_ARRAY_KEYS + _MANUAL_LOAD_ARRAY_KEYS:
        values = optim_conf.get(key)
        if isinstance(values, list) and len(values) == len(keep):
            optim_conf[key] = [v for v, k in zip(values, keep) if k]
    optim_conf["number_of_deferrable_loads"] = sum(keep)


def _strip_auto_appended_loads(optim_conf: dict, source_markers: set[str]) -> None:
    """Idempotency guard for _append_ev_deferrable_loads/
    _append_room_thermal_loads/_append_boiler_thermal_battery_loads: on a
    fresh config.json these markers aren't present yet (no-op) - but
    build_params can run more than once against an ALREADY-derived config
    (the config page's GET /get-config -> edit -> POST /set-config round
    trip, see web_server.py's parameter_get/parameter_set), and without
    this guard every such round trip silently re-appends another copy of
    each EV charger/room/boiler on top of the last, growing config.json's
    deferrable-load arrays without bound. Called at the very top of each
    of the three functions above, before their own set_use_*/number_of_*
    gate check, so toggling a feature OFF also cleans up its previously-
    appended entries rather than leaving them orphaned forever.

    :param optim_conf: The optimization config dict to clean in place.
    :type optim_conf: dict
    :param source_markers: def_load_config _source values to strip (e.g.
        {"ev_auto"}, or {"room_auto", "heatpump_dispatch_auto"} together
        since _append_room_thermal_loads derives both).
    :type source_markers: set[str]
    """
    def_load_cfg = optim_conf.get("def_load_config", []) or []
    if not def_load_cfg:
        return

    keep = [
        _def_load_config_source_marker(entry) not in source_markers for entry in def_load_cfg
    ]
    if all(keep):
        return

    optim_conf["def_load_config"] = [e for e, k in zip(def_load_cfg, keep) if k]
    for key in _AUTO_LOAD_ARRAY_KEYS:
        values = optim_conf.get(key)
        if isinstance(values, list) and len(values) == len(keep):
            optim_conf[key] = [v for v, k in zip(values, keep) if k]
    optim_conf["number_of_deferrable_loads"] = sum(keep)


def check_def_loads(
    num_def_loads: int,
    parameter: list[dict],
    default: str | float,
    parameter_name: str,
    logger: logging.Logger,
) -> list[dict]:
    """
    Check parameter lists with deferrable loads number, if they do not match, enlarge to fit.

    A missing key or ``None`` value is filled with ``default`` for every load. ``parameter``
    is updated in place (and the same list returned), matching how every call site reassigns
    ``params["optim_conf"][name] = check_def_loads(...)``.

    :param num_def_loads: Total number deferrable loads
    :type num_def_loads: int
    :param parameter: parameter config dict containing paramater
    :type parameter: list[dict]
    :param default: default value for parameter to pad missing
    :type default: str | int | float
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: parameter list
    :rtype: list[dict]
    """
    current = parameter.get(parameter_name, None)
    # Missing key or explicit JSON null: apply the default for every load. This is
    # expected when a per-load array is absent from the user config (e.g. a parameter
    # added in a later release), so do it silently rather than raising KeyError.
    if current is None:
        parameter[parameter_name] = [default] * num_def_loads
        return parameter[parameter_name]
    if isinstance(current, list) and num_def_loads > len(current):
        # Enlarging a short list to match number_of_deferrable_loads is this function's
        # documented job, not an error: the shipped defaults are sized for the default
        # load count, so any user that raises the count without restating every per-load
        # array lands here. Log at debug to avoid spurious startup warnings (#929).
        logger.debug(
            parameter_name
            + " has fewer entries than number_of_deferrable_loads, padding with default ("
            + str(default)
            + ")"
        )
        for _x in range(len(current), num_def_loads):
            current.append(default)
    result = parameter[parameter_name]
    # Replace any None elements with the default (can occur when set-config
    # receives a partial config, e.g. [null, 0] instead of [0, 0]).
    if isinstance(result, list):
        result = [v if v is not None else default for v in result]
        parameter[parameter_name] = result
    return result


def _warn_if_runtime_def_array_too_short(
    was_provided: bool,
    def_array_name: str,
    before_value: object,
    after_value: object,
    final_num_def_loads: int,
    logger: logging.Logger,
) -> None:
    """
    Warn when the re-normalisation pass actually changed a runtime-provided
    per-load deferrable array (#1040).

    check_def_loads pads a short array (and heals a None element) silently at
    debug level (#929) - correct for a config-sourced array, e.g. a config
    predating a load-count bump. A runtime-provided array is different: a
    stale/short one is the foot-gun #1040 reports (an MPC caller resending an
    old, shorter array after the load count changed elsewhere), so it gets a
    visible warning instead. The values are unchanged either way - this only
    makes a runtime-sourced change visible.

    Fires only when the caller has established the value was genuinely
    provided at runtime (`was_provided`) AND check_def_loads actually changed
    it. The comparison is by value, not identity, since check_def_loads
    always returns a freshly built list even when nothing changed - so an
    identity check would report "changed" even when heat_topology's own
    compiled value replaced the runtime one and nothing was really padded.

    Describes the outcome rather than the input shape: a length increase says
    how many entries were padded; a same-length change (a None element
    healed) says that instead of overclaiming padding.

    :param was_provided: whether the modern or legacy name carried a
        non-null value in runtimeparams
    :type was_provided: bool
    :param def_array_name: the modern per-load array parameter name
    :type def_array_name: str
    :param before_value: optim_conf[def_array_name] immediately before this
        call's check_def_loads (an independent copy, never mutated by it)
    :type before_value: object
    :param after_value: optim_conf[def_array_name] immediately after
    :type after_value: object
    :param final_num_def_loads: number_of_deferrable_loads after every reset
        inside treat_runtimeparams has been applied
    :type final_num_def_loads: int
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if not was_provided or before_value == after_value:
        return
    if (
        isinstance(before_value, list)
        and isinstance(after_value, list)
        and len(before_value) < len(after_value)
    ):
        logger.warning(
            f"{def_array_name}: padded from {len(before_value)} to "
            f"{len(after_value)} entries to match "
            f"number_of_deferrable_loads={final_num_def_loads}. Pass a "
            "full-length array to avoid this warning."
        )
    else:
        logger.warning(
            f"{def_array_name}: a None/null entry was replaced with the "
            f"parameter default (number_of_deferrable_loads="
            f"{final_num_def_loads}). Pass a fully populated array to avoid "
            "this warning."
        )


# Per-battery ARRAY parameters (#610): scalar accepted and broadcast to every
# battery, exact-length-N list accepted as-is. Unlike check_def_loads (deferrable
# loads), a wrong-length list is a hard error, not padded - silently enlarging a
# physical-plant array risks masking a genuine per-battery mis-configuration.
# weight_battery_charge/discharge are intentionally excluded: they nest instead
# (see check_batt_weight_params) because a flat list there is already a time
# series today, not a per-battery array.
BATT_ARRAY_PARAMS_PLANT_CONF: dict[str, bool | int | float] = {
    "battery_discharge_power_max": 1000,
    "battery_charge_power_max": 1000,
    "battery_discharge_efficiency": 0.95,
    "battery_charge_efficiency": 0.95,
    "battery_nominal_energy_capacity": 5000,
    "battery_minimum_state_of_charge": 0.3,
    "battery_maximum_state_of_charge": 0.9,
    "battery_target_state_of_charge": 0.6,
    "battery_stress_cost": 0.0,
}
BATT_ARRAY_PARAMS_OPTIM_CONF: dict[str, bool | int | float] = {
    "battery_soc_deficit_threshold": 0.4,
    "battery_soc_deficit_cost": 0.0,
    "battery_soc_surplus_threshold": 0.9,
    "battery_soc_surplus_cost": 0.0,
}
BATT_WEIGHT_PARAMS: tuple[str, ...] = ("weight_battery_charge", "weight_battery_discharge")


def _coerce_batt_element(
    value: bool | float | str | None, default: bool | float, parameter_name: str
) -> bool | int | float:
    """Coerce a single per-battery array element (#610).

    None or a stringly-typed "null" resolve to the per-slot default; a numeric
    string coerces to float; anything else passes through unchanged. A genuinely
    non-numeric string raises a clear ValueError rather than silently becoming 0.
    """
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip().lower() == "null":
            return default
        try:
            return float(value)
        except ValueError as err:
            raise ValueError(f"{parameter_name}: non-numeric string element {value!r}") from err
    return value


def validate_num_batteries(plant_conf: dict) -> int:
    """
    Read and validate plant_conf["number_of_batteries"] (#610).

    Accepts a positive integer (an integral float or numeric string is coerced);
    zero, negative, fractional, boolean or non-numeric values raise ValueError
    here, at config time, rather than surfacing as an empty per-battery list
    deep inside the optimization build.

    :param plant_conf: the plant configuration dict
    :type plant_conf: dict
    :return: the validated battery count (missing key -> 1)
    :rtype: int
    """
    raw = plant_conf.get("number_of_batteries", 1)
    invalid = ValueError(f"number_of_batteries must be a positive integer, got {raw!r}")
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        raise invalid
    try:
        value = float(raw)
    except ValueError:
        raise invalid from None
    if not value.is_integer() or value < 1:
        raise invalid
    return int(value)


def validate_num_phases(plant_conf: dict, logger: logging.Logger) -> int:
    """
    Read and validate plant_conf["number_of_phases"].

    Unlike validate_num_batteries, an out-of-range value is clamped with a
    warning rather than raised: number_of_phases only gates an additive
    safety constraint (see optimization.py::_add_phase_balance_constraints),
    so a bad value should degrade to "feature off"/"feature clamped", never
    abort the whole optimization run.

    :param plant_conf: the plant configuration dict
    :type plant_conf: dict
    :param logger: The logger object
    :type logger: logging.Logger
    :return: the validated phase count, clamped to [1, 3] (missing/invalid key -> 1)
    :rtype: int
    """
    raw = plant_conf.get("number_of_phases", 1)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "number_of_phases=%r is not a number - defaulting to 1 (single-phase, "
            "phase balancing disabled).",
            raw,
        )
        return 1
    if value < 1:
        logger.warning(
            "number_of_phases=%d is out of range - defaulting to 1 (single-phase, "
            "phase balancing disabled).",
            value,
        )
        return 1
    if value > 3:
        logger.warning(
            "number_of_phases=%d is out of range - clamping to 3 (a household "
            "connection has at most 3 phases).",
            value,
        )
        return 3
    return value


def check_batt_params(
    num_batteries: int,
    parameter: dict,
    default: bool | float,
    parameter_name: str,
    logger: logging.Logger,
) -> bool | int | float | list:
    """
    Normalise a per-battery plant_conf/optim_conf array parameter (#610).

    N == 1 (the default/off state) is a true no-op: a scalar stays a scalar,
    byte-identical to master, since optimization.py's per-battery model
    reduces to reading that same scalar at index 0. For N > 1: a missing key
    or scalar value broadcasts to ``[value] * num_batteries``; a list must be
    exactly ``num_batteries`` long (wrong length raises ValueError naming the
    parameter and the expected length - no silent padding, unlike deferrable
    loads). List elements are coerced per _coerce_batt_element (None/"null" ->
    default, numeric string -> float).

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: parameter config dict containing parameter_name
    :type parameter: dict
    :param default: default value used to fill missing/None slots
    :type default: bool | int | float
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: the normalised value: a scalar at N=1, a length-num_batteries list at N>1
    :rtype: bool | int | float | list
    """
    current = parameter.get(parameter_name, None)
    if current is None:
        if num_batteries == 1:
            return current
        parameter[parameter_name] = [default] * num_batteries
        return parameter[parameter_name]
    if isinstance(current, list):
        if len(current) != num_batteries:
            raise ValueError(
                f"{parameter_name} has {len(current)} entries but "
                f"number_of_batteries={num_batteries}; provide a scalar (broadcast "
                f"to every battery) or a list of exactly {num_batteries} entries"
            )
        normalized = [_coerce_batt_element(v, default, parameter_name) for v in current]
        parameter[parameter_name] = normalized[0] if num_batteries == 1 else normalized
        return parameter[parameter_name]
    # Scalar (or stringly-typed scalar, e.g. a "null"/"0.0" string from HA).
    coerced = _coerce_batt_element(current, default, parameter_name)
    if num_batteries == 1:
        parameter[parameter_name] = coerced
        return coerced
    parameter[parameter_name] = [coerced] * num_batteries
    return parameter[parameter_name]


def check_phase_array_params(
    count: int,
    parameter: dict,
    default: bool | int | float | str,
    parameter_name: str,
    logger: logging.Logger,
) -> bool | int | float | str | list:
    """
    Normalise a per-phase array parameter (number_of_phases-indexed, e.g.
    maximum_power_from_grid_per_phase) or a per-battery phase tag
    (number_of_batteries-indexed, battery_phase) against `count`.

    Same broadcast-scalar-or-exact-length-list-else-hard-error shape as
    check_batt_params, deliberately WITHOUT check_batt_params's numeric
    coercion (_coerce_batt_element unconditionally attempts float() on any
    non-"null" string, which would raise on a genuine phase label like
    "L1") - this helper also covers string-typed fields, so it passes every
    element through unchanged instead.

    count == 1 is a true no-op: a scalar stays a scalar. For count > 1: a
    missing key or scalar value broadcasts to ``[value] * count``; a list
    must be exactly ``count`` long (wrong length raises ValueError naming
    the parameter and the expected length - no silent padding).

    :param count: number_of_phases or number_of_batteries
    :type count: int
    :param parameter: parameter config dict containing parameter_name
    :type parameter: dict
    :param default: default value used when the key is missing
    :type default: bool | int | float | str
    :param parameter_name: name of parameter
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :return: the normalised value: a scalar at count=1, a length-count list at count>1
    :rtype: bool | int | float | str | list
    """
    current = parameter.get(parameter_name, None)
    if current is None:
        if count == 1:
            return current
        parameter[parameter_name] = [default] * count
        return parameter[parameter_name]
    if isinstance(current, list):
        if len(current) != count:
            raise ValueError(
                f"{parameter_name} has {len(current)} entries but the expected "
                f"count is {count}; provide a scalar (broadcast to every entry) "
                f"or a list of exactly {count} entries"
            )
        parameter[parameter_name] = current[0] if count == 1 else current
        return parameter[parameter_name]
    # Scalar.
    if count == 1:
        parameter[parameter_name] = current
        return current
    parameter[parameter_name] = [current] * count
    return parameter[parameter_name]


def compute_phase_power_shares(
    df_history: pd.DataFrame,
    phase_sensor_entity_ids: list[str],
    logger: logging.Logger,
) -> list[float] | None:
    """
    Compute each phase's historical share of a total (load or PV) signal,
    for the ratio-based per-phase forecast split used by
    _add_phase_balance_constraints (see
    command_line.py::prepare_forecast_and_weather_data).

    share[p] = mean(phase_p column) / sum_p(mean(phase_p column)) over
    whatever history df_history covers (normally historic_days_to_retrieve,
    the same window every other "historic" computation in this codebase
    already uses). This is a single static ratio, not an hour-of-day-
    varying one - a deliberate simplification appropriate for a safety-
    margin feature (avoid overloading a fuse), not a billing-accuracy one.

    :param df_history: retrieved historical data, expected to contain one
        column per configured phase sensor entity id
    :type df_history: pd.DataFrame
    :param phase_sensor_entity_ids: configured per-phase sensor entity ids,
        in phase order (L1, L2, ...) - unconfigured/blank entries are
        allowed and simply contribute a 0 share
    :type phase_sensor_entity_ids: list[str]
    :param logger: The logger object
    :type logger: logging.Logger
    :return: shares (same length as phase_sensor_entity_ids, summing to
        ~1.0), or None if no usable sensor data is available at all
    :rtype: list[float] | None
    """
    if not phase_sensor_entity_ids or df_history is None or df_history.empty:
        return None
    means: list[float] = []
    for entity_id in phase_sensor_entity_ids:
        entity_id = str(entity_id or "").strip()
        if not entity_id or entity_id not in df_history.columns:
            means.append(0.0)
            continue
        series = pd.to_numeric(df_history[entity_id], errors="coerce").dropna()
        means.append(float(series.mean()) if len(series) else 0.0)
    total = sum(means)
    if total <= 1e-9:
        logger.warning(
            "compute_phase_power_shares: configured per-phase sensors (%s) returned "
            "no usable (nonzero) historical data - cannot compute a phase split from "
            "them.",
            phase_sensor_entity_ids,
        )
        return None
    return [m / total for m in means]


def _warn_if_runtime_scalar_masks_batt_list(
    num_batteries: int,
    parameter: dict,
    parameter_name: str,
    distinct_config_lists: dict[str, list],
    logger: logging.Logger,
) -> None:
    """
    Warn when a runtime scalar masks a configured per-battery list (#1032).

    Runtime values override configured ones by design, and a scalar broadcasts
    to every battery. When the configured value was a list with genuinely
    distinct per-battery entries, that broadcast silently flattens them - a
    documented but easy-to-miss interaction (a real two-battery deployment
    lost its per-unit power limits to an aggregate runtime scalar this way).
    The resulting values are unchanged; this only makes the override visible.

    Uniform configured lists stay silent on purpose: build_params normalises a
    configured scalar to ``[value] * N`` before this runs, so a warning keyed
    on "configured value is a list" alone would fire on every runtime scalar
    a broadcast-config user sends, every MPC cycle.

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: the plant_conf/optim_conf dict AFTER the runtime
        association loop has applied any override
    :type parameter: dict
    :param parameter_name: name of the per-battery array parameter
    :type parameter_name: str
    :param distinct_config_lists: parameter name -> configured list, captured
        BEFORE the association loop, only for lists with distinct entries
    :type distinct_config_lists: dict[str, list]
    :param logger: The logger object
    :type logger: logging.Logger
    """
    if num_batteries <= 1 or parameter_name not in distinct_config_lists:
        return
    current = parameter.get(parameter_name)
    if current is None or isinstance(current, list):
        return
    logger.warning(
        f"{parameter_name}: runtime scalar {current} overrides the configured "
        f"per-battery list {distinct_config_lists[parameter_name]} and will "
        f"broadcast to all {num_batteries} batteries; pass a list of "
        f"{num_batteries} entries to keep per-battery values"
    )


def _coerce_batt_weight_value(
    value: bool | float | str | list | None,
    default: bool | float,
    parameter_name: str,
) -> bool | int | float | list:
    """Recursively coerce a weight_battery_* value/element (#610).

    Applies the same leaf-level rule as _coerce_batt_element (None/"null" ->
    default, numeric string -> float, non-numeric string -> clear ValueError)
    at ANY nesting depth: a bare scalar, a flat time series, or a nested
    per-battery list-of-series. Only leaf elements are coerced - the outer
    SHAPE is always preserved (a list stays a list of the same length/nesting,
    a scalar stays a scalar), so this is safe to run unconditionally,
    including at num_batteries == 1 where check_batt_weight_params must
    otherwise stay a byte-identical no-op.
    """
    if isinstance(value, list):
        return [_coerce_batt_weight_value(v, default, parameter_name) for v in value]
    return _coerce_batt_element(value, default, parameter_name)


def check_batt_weight_params(
    num_batteries: int,
    parameter: dict,
    parameter_name: str,
    logger: logging.Logger,
    default: bool | float = 0.0,
) -> None:
    """
    Normalise weight_battery_charge/weight_battery_discharge into the nested
    per-battery form for N > 1 (#610). Mutates parameter[parameter_name] in
    place; returns nothing (mirrors the value being read back via ``parameter``).

    A stringly-typed guard runs FIRST, at every num_batteries (including 1):
    every leaf element is coerced via _coerce_batt_weight_value (None/"null"
    -> default, numeric string -> float, non-numeric string -> clear
    ValueError naming the parameter). Before #610 these two params had zero
    validation of any kind at N=1 (the default and only value every
    non-adopting user has), unlike every other per-battery array param. Only
    leaf VALUES are coerced; the shape is untouched, so the N=1 no-op still
    holds for any already-well-typed value.

    Disambiguation, in priority order:
    1. num_batteries == 1: no-op (beyond the stringly-typed guard above),
       exactly today's behaviour (scalar or series, no nesting) -
       optimization.py's existing np.array(...) broadcast handles both shapes
       unchanged.
    2. Scalar value: broadcast, every battery gets the same scalar -> [v] * N.
    3. List of length num_batteries whose elements are themselves scalars or
       lists: already per-battery (entry k -> battery k); passed through as-is.
    4. Any other list (a flat time series not of length N): shared by every
       battery, today's semantics preserved -> [v] * N (each battery gets an
       identical copy of the series).
    5. Ambiguous corner: a flat NUMERIC list whose length happens to equal
       num_batteries is caught by rule 3 (per-battery), not rule 4 (shared
       series). Documented in docs/config.md: nest explicitly ([series] * N,
       or distinct per-battery entries) if a shared series of that exact
       length was intended.

    :param num_batteries: plant_conf["number_of_batteries"]
    :type num_batteries: int
    :param parameter: the optim_conf dict containing parameter_name
    :type parameter: dict
    :param parameter_name: "weight_battery_charge" or "weight_battery_discharge"
    :type parameter_name: str
    :param logger: The logger object
    :type logger: logging.Logger
    :param default: fallback value for a None/"null" leaf (config_defaults.json
        value for both weight params is 0.0)
    :type default: bool | int | float
    """
    current = parameter.get(parameter_name, None)
    if current is None:
        return
    current = _coerce_batt_weight_value(current, default, parameter_name)
    parameter[parameter_name] = current
    if num_batteries == 1:
        return
    if not isinstance(current, list):
        parameter[parameter_name] = [current] * num_batteries
        return
    if len(current) == num_batteries and all(
        isinstance(v, list) or (isinstance(v, int | float) and not isinstance(v, bool))
        for v in current
    ):
        # Already per-battery nested (rule 3, including the ambiguous rule-5
        # corner), pass through unchanged.
        return
    # Flat list not of length num_batteries -> shared time series (rule 4). Each
    # battery gets its OWN copy: [current] * N would alias one list object
    # across every battery, so a later in-place mutation of one battery's
    # series would silently rewrite them all.
    parameter[parameter_name] = [list(current) for _ in range(num_batteries)]


def _parse_program_power_sequence(raw_programs: object) -> list[float] | None:
    """Parse a deferrable program payload into a numeric power sequence."""

    def _to_numeric_list(values: list[object]) -> list[float] | None:
        out: list[float] = []
        for value in values:
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val) and val >= 0:
                out.append(val)
        return out or None

    if raw_programs is None:
        return None

    if isinstance(raw_programs, (int, float)):
        val = float(raw_programs)
        if np.isfinite(val) and val >= 0:
            return [val]
        return None

    if isinstance(raw_programs, str):
        text = raw_programs.strip()
        if not text:
            return None
        try:
            parsed = orjson.loads(text)
            return _parse_program_power_sequence(parsed)
        except orjson.JSONDecodeError:
            return _to_numeric_list([item.strip() for item in text.split(",") if item.strip()])

    if isinstance(raw_programs, dict):
        if raw_programs.get("power_pattern") is not None:
            return _parse_program_power_sequence(raw_programs.get("power_pattern"))
        for key in ["programs", "load_programs", "sequence"]:
            if raw_programs.get(key) is not None:
                return _parse_program_power_sequence(raw_programs.get(key))
        return None

    if isinstance(raw_programs, list):
        if len(raw_programs) == 0:
            return None
        if isinstance(raw_programs[0], dict):
            for program in raw_programs:
                sequence = _parse_program_power_sequence(program)
                if sequence:
                    return sequence
            return None
        return _to_numeric_list(raw_programs)

    return None


def _normalize_deferrable_load_categories(params: dict, logger: logging.Logger) -> None:
    """Normalize mixed deferrable load categories into optimizer-ready fields."""
    optim_conf = params.get("optim_conf", {})
    num_def_loads = optim_conf.get("number_of_deferrable_loads")
    if not isinstance(num_def_loads, int) or num_def_loads <= 0:
        return

    optim_conf["load_type"] = check_def_loads(
        num_def_loads,
        optim_conf,
        "fixed_power_non_splittable",
        "load_type",
        logger,
    )
    optim_conf["load_programs"] = check_def_loads(
        num_def_loads,
        optim_conf,
        "[]",
        "load_programs",
        logger,
    )
    optim_conf["load_dispatch_mode"] = check_def_loads(
        num_def_loads,
        optim_conf,
        "hours",
        "load_dispatch_mode",
        logger,
    )
    optim_conf["required_energy_kwh_of_each_deferrable_load"] = check_def_loads(
        num_def_loads,
        optim_conf,
        0.0,
        "required_energy_kwh_of_each_deferrable_load",
        logger,
    )

    valid_modes = {"hours", "program", "energy_kwh"}

    for k in range(num_def_loads):
        load_type = optim_conf["load_type"][k]
        dispatch_mode = str(optim_conf["load_dispatch_mode"][k]).strip().lower()
        if dispatch_mode not in valid_modes:
            dispatch_mode = "program" if load_type == "program_based" else "hours"
            optim_conf["load_dispatch_mode"][k] = dispatch_mode

        if load_type == "program_based":
            sequence = _parse_program_power_sequence(optim_conf["load_programs"][k])
            if sequence:
                optim_conf["nominal_power_of_deferrable_loads"][k] = sequence
                optim_conf["operating_hours_of_each_deferrable_load"][k] = len(sequence)
                optim_conf["treat_deferrable_load_as_semi_cont"][k] = True
                # Program-based loads are sequence-constrained, so dispatch mode must be "program".
                optim_conf["load_dispatch_mode"][k] = "program"
            else:
                logger.warning(
                    "load_programs[%d] is empty/invalid for program_based load, "
                    "falling back to nominal_power_of_deferrable_loads",
                    k,
                )
                if optim_conf["load_dispatch_mode"][k] == "program":
                    optim_conf["load_dispatch_mode"][k] = "hours"
        elif load_type in ["fixed_power_splittable", "variable_power_variable_time"]:
            optim_conf["treat_deferrable_load_as_semi_cont"][k] = False
        else:
            optim_conf["treat_deferrable_load_as_semi_cont"][k] = True

        try:
            optim_conf["required_energy_kwh_of_each_deferrable_load"][k] = float(
                optim_conf["required_energy_kwh_of_each_deferrable_load"][k]
            )
        except (TypeError, ValueError):
            optim_conf["required_energy_kwh_of_each_deferrable_load"][k] = 0.0


def _parse_profile_to_float_list(raw: object) -> list[float]:
    """Parse a profile payload into a float list (supports list/json/csv)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = orjson.loads(text)
            return _parse_profile_to_float_list(parsed)
        except orjson.JSONDecodeError:
            out = []
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    continue
            return out
    return []


def _resample_power_profile(
    power_profile: list[float],
    source_interval_min: float,
    target_step_min: float,
) -> list[float]:
    """Resample a stepped power profile (e.g. a WashData learned cycle) from
    its native time resolution to a different one. Pure function, no I/O.

    Each power_profile[i] is treated as the constant average power over the
    half-open interval [i*source_interval_min, (i+1)*source_interval_min)
    (matching WashData's own power_profile_interval_min semantics: "each
    element is an N-minute average"). The step function is forward-filled to
    a fine common resolution and then mean-aggregated into
    target_step_min-wide bins - one code path correctly covers downsampling
    (e.g. 15min->30min), upsampling (e.g. 15min->5min) and equal resolution.

    :param power_profile: Watt values, one per source_interval_min block.
    :type power_profile: list[float]
    :param source_interval_min: Minutes each power_profile element spans.
    :type source_interval_min: float
    :param target_step_min: Minutes each returned element should span
        (normally retrieve_hass_conf["optimization_time_step"] in minutes).
    :type target_step_min: float
    :return: Resampled Watt values. Empty/degenerate input is returned
        unchanged rather than raising.
    :rtype: list[float]
    """
    if not power_profile:
        return []
    values = [float(v) for v in power_profile]
    if source_interval_min <= 0 or target_step_min <= 0 or len(values) == 1:
        return values
    if abs(source_interval_min - target_step_min) < 1e-9:
        return values

    start = pd.Timestamp("2000-01-01")
    source_freq = pd.Timedelta(minutes=source_interval_min)
    idx = pd.date_range(start=start, periods=len(values), freq=source_freq)
    series = pd.Series(values, index=idx)

    fine_freq_min = min(1.0, source_interval_min, target_step_min)
    fine_freq = pd.Timedelta(minutes=fine_freq_min)
    end = idx[-1] + source_freq  # exclusive end of the last source block
    fine_index = pd.date_range(start=start, end=end - fine_freq, freq=fine_freq)
    fine_series = series.reindex(fine_index, method="ffill")

    resampled = fine_series.resample(pd.Timedelta(minutes=target_step_min)).mean().dropna()
    return [float(v) for v in resampled.to_numpy()]


def _is_legionella_due(
    last_run_iso: str | None,
    interval_days: int,
) -> bool:
    """Return true if a legionella cycle is due based on last execution timestamp."""
    if interval_days <= 0:
        return False
    if not last_run_iso:
        return True
    try:
        last_run = pd.to_datetime(last_run_iso, utc=True)
        if pd.isna(last_run):
            return True
    except Exception:
        return True
    now = pd.Timestamp.now(tz=UTC)
    return (now - last_run) >= pd.Timedelta(days=interval_days)


async def _append_boiler_thermal_battery_loads(
    params: dict, logger: logging.Logger, emhass_conf: dict
) -> None:
    """Map boiler configuration into def_load_config thermal_battery entries."""
    optim_conf = params.get("optim_conf", {})
    _strip_auto_appended_loads(optim_conf, {"boiler_auto"})
    if not optim_conf.get("set_use_boiler", False):
        return

    num_boilers = int(optim_conf.get("number_of_boilers", 1) or 1)
    if num_boilers <= 0:
        return

    # Ensure required lists exist for extension
    base_default_lists = {
        "nominal_power_of_deferrable_loads": 0.0,
        "minimum_power_of_deferrable_loads": 0.0,
        "operating_hours_of_each_deferrable_load": 0,
        "start_timesteps_of_each_deferrable_load": 0,
        "end_timesteps_of_each_deferrable_load": 0,
        "set_deferrable_startup_penalty": 0.0,
        "set_deferrable_load_single_constant": False,
        "treat_deferrable_load_as_semi_cont": False,
        "load_type": "fixed_power_non_splittable",
        "load_dispatch_mode": "hours",
        "required_energy_kwh_of_each_deferrable_load": 0.0,
        "load_phase": "",
    }
    num_def_loads = int(optim_conf.get("number_of_deferrable_loads", 0) or 0)
    for key, default in base_default_lists.items():
        optim_conf[key] = check_def_loads(num_def_loads, optim_conf, default, key, logger)

    # Boiler vectorized parameters
    boiler_types = check_def_loads(num_boilers, optim_conf, "resistive", "boiler_type", logger)
    boiler_names = check_def_loads(num_boilers, optim_conf, "boiler_1", "boiler_names", logger)
    boiler_phase = check_def_loads(num_boilers, optim_conf, "", "boiler_phase", logger)
    boiler_power = check_def_loads(
        num_boilers, optim_conf, 2000.0, "boiler_nominal_power", logger
    )
    boiler_volume_l = check_def_loads(num_boilers, optim_conf, 180.0, "boiler_volume_l", logger)
    boiler_supply_temp = check_def_loads(
        num_boilers, optim_conf, 55.0, "boiler_supply_temperature", logger
    )
    boiler_start_temp = check_def_loads(
        num_boilers, optim_conf, 50.0, "boiler_start_temperature", logger
    )
    boiler_target_temp = check_def_loads(
        num_boilers, optim_conf, 52.0, "boiler_target_temperature", logger
    )
    boiler_min_temp = check_def_loads(num_boilers, optim_conf, 45.0, "boiler_min_temperature", logger)
    boiler_max_temp = check_def_loads(num_boilers, optim_conf, 60.0, "boiler_max_temperature", logger)
    boiler_dhw_profile = check_def_loads(
        num_boilers, optim_conf, "", "boiler_dhw_draw_kwh_forecast", logger
    )
    boiler_loss = check_def_loads(num_boilers, optim_conf, 0.02, "boiler_loss_factor", logger)

    # Legionella controls
    legio_interval = check_def_loads(
        num_boilers, optim_conf, 7, "boiler_legionella_interval_days", logger
    )
    legio_target = check_def_loads(
        num_boilers, optim_conf, 60.0, "boiler_legionella_target_temp", logger
    )
    legio_hold_h = check_def_loads(
        num_boilers, optim_conf, 0.5, "boiler_legionella_hold_hours", logger
    )
    legio_last = check_def_loads(
        num_boilers, optim_conf, "", "boiler_legionella_last_run_iso", logger
    )
    # Overlay backend-persisted completion timestamps (written after a solved
    # plan satisfies the legionella hold, see _maybe_record_legionella_completion
    # in command_line.py) on top of the config-supplied default, without ever
    # rewriting config.json. A persisted value only overrides when non-empty.
    runtime_state = await load_json_blob(
        emhass_conf, "boiler_runtime_state.json", logger, default={}
    )
    persisted_last_run = runtime_state.get("boiler_legionella_last_run_iso", [])
    for i in range(min(num_boilers, len(persisted_last_run))):
        if persisted_last_run[i]:
            legio_last[i] = persisted_last_run[i]
    optim_conf["boiler_legionella_last_run_iso"] = legio_last
    legio_force_res = check_def_loads(
        num_boilers, optim_conf, True, "boiler_legionella_force_resistive", logger
    )

    # HP sharing and phase-3 options
    coupled_hp_idx = check_def_loads(
        num_boilers, optim_conf, -1, "boiler_coupled_heatpump_load_index", logger
    )
    shared_hp_power = check_def_loads(
        num_boilers, optim_conf, 0.0, "boiler_hp_shared_max_power", logger
    )
    uncertainty_margin = check_def_loads(
        num_boilers, optim_conf, 0.0, "boiler_uncertainty_margin_kwh", logger
    )

    # Forecast horizon for static list defaults
    step_td = params.get("retrieve_hass_conf", {}).get("optimization_time_step", pd.to_timedelta(30, "min"))
    if isinstance(step_td, (int, float)):
        step_td = pd.to_timedelta(step_td, "minutes")
    delta = optim_conf.get("delta_forecast_daily", pd.to_timedelta(1, "days"))
    if isinstance(delta, (int, float)):
        delta = pd.to_timedelta(delta, "days")
    try:
        horizon_steps = max(1, int(delta / step_td))
    except Exception:
        horizon_steps = 48

    def_load_cfg = optim_conf.get("def_load_config", []) or []
    # Pad to the pre-existing load count so appended boiler entries land at the
    # correct index; otherwise they'd shift onto whatever load index happens to
    # equal len(def_load_cfg), silently mismatching physical loads to configs.
    while len(def_load_cfg) < num_def_loads:
        def_load_cfg.append({})

    for i in range(num_boilers):
        btype = str(boiler_types[i]).strip().lower()
        if btype not in {"resistive", "hpboiler", "hp_tank_zone"}:
            btype = "resistive"

        due_legio = _is_legionella_due(str(legio_last[i]).strip(), int(legio_interval[i] or 0))
        force_resistive = bool(legio_force_res[i])
        effective_type = "resistive" if (due_legio and force_resistive) else btype

        min_temp = float(boiler_min_temp[i])
        max_temp = float(boiler_max_temp[i])
        target_temp = float(boiler_target_temp[i])

        if due_legio:
            target_temp = max(target_temp, float(legio_target[i]))
            max_temp = max(max_temp, float(legio_target[i]))

        # Phase-3 uncertainty hedge: shift minimum target upward by equivalent thermal margin.
        margin_kwh = float(uncertainty_margin[i] or 0.0)
        if margin_kwh > 0 and boiler_volume_l[i]:
            temp_margin = margin_kwh * 3600.0 / (4.186 * float(boiler_volume_l[i]))
            min_temp = min(max_temp, min_temp + max(0.0, temp_margin))

        draw_profile = _parse_profile_to_float_list(boiler_dhw_profile[i])
        if len(draw_profile) < horizon_steps:
            draw_profile = draw_profile + [0.0] * (horizon_steps - len(draw_profile))
        else:
            draw_profile = draw_profile[:horizon_steps]

        # resolve_thermal_battery_cop only takes the flat constant-efficiency
        # branch when "efficiency" is present in hc - "carnot_efficiency"
        # always selects the heat-pump Carnot-lift formula instead, which for
        # a resistive element (no real heat-pump lift) computed a COP well
        # above 1.0 at typical supply/outdoor temperatures. Resistive heating
        # is physically COP=1.0 flat, so it needs "efficiency", not
        # "carnot_efficiency".
        cop_key = "carnot_efficiency" if effective_type in {"hpboiler", "hp_tank_zone"} else "efficiency"
        cop_eff = 0.42 if effective_type in {"hpboiler", "hp_tank_zone"} else 1.0

        thermal_cfg = {
            "name": str(boiler_names[i]),
            "boiler_type": effective_type,
            "supply_temperature": float(boiler_supply_temp[i]),
            "volume": max(0.05, float(boiler_volume_l[i]) / 1000.0),
            "start_temperature": float(boiler_start_temp[i]),
            "min_temperatures": [min_temp] * horizon_steps,
            "max_temperatures": [max_temp] * horizon_steps,
            "indoor_target_temperature": target_temp,
            "custom_heating_demand_profile": draw_profile,
            "base_loss": float(boiler_loss[i]),
            cop_key: cop_eff,
            "legionella_due": due_legio,
            "legionella_target_temperature": float(legio_target[i]),
            "legionella_hold_hours": float(legio_hold_h[i]),
            "coupled_heatpump_load_index": int(coupled_hp_idx[i]),
            "hp_shared_max_power": float(shared_hp_power[i]),
            "_source": "boiler_auto",
        }

        def_load_cfg.append({"thermal_battery": thermal_cfg})

        # custom_predicted_temperature_id/custom_heating_demand_id are built
        # early in build_params for the *original* deferrable loads only
        # (before this function runs), so boiler-appended loads need their
        # own entries added here or _publish_thermal_loads silently skips
        # them (list-length bound check never matches their higher index).
        passed_data = params.setdefault("passed_data", {})
        passed_data.setdefault("custom_predicted_temperature_id", []).append(
            {
                "entity_id": f"sensor.temp_predicted{num_def_loads}",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "friendly_name": f"Predicted temperature {num_def_loads}",
            }
        )
        passed_data.setdefault("custom_heating_demand_id", []).append(
            {
                "entity_id": f"sensor.heating_demand{num_def_loads}",
                "device_class": "energy",
                "unit_of_measurement": "kWh",
                "friendly_name": f"Heating demand {num_def_loads}",
            }
        )

        # Append matching generic deferrable vectors
        nominal_w = max(0.0, float(boiler_power[i]))
        optim_conf["nominal_power_of_deferrable_loads"].append(nominal_w)
        optim_conf["minimum_power_of_deferrable_loads"].append(0.0)
        optim_conf["operating_hours_of_each_deferrable_load"].append(0)
        optim_conf["start_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["end_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["set_deferrable_startup_penalty"].append(0.0)
        optim_conf["set_deferrable_load_single_constant"].append(False)
        optim_conf["treat_deferrable_load_as_semi_cont"].append(False)
        optim_conf["load_type"].append("fixed_power_non_splittable")
        optim_conf["load_dispatch_mode"].append("hours")
        optim_conf["required_energy_kwh_of_each_deferrable_load"].append(0.0)
        optim_conf["load_phase"].append(str(boiler_phase[i]).strip())
        num_def_loads += 1

        # If legionella should be forced resistive, lock COP to 1 in this cycle.
        if due_legio and force_resistive and btype != "resistive":
            logger.info(
                "Boiler %s: legionella due -> forcing resistive cycle for this optimization run",
                boiler_names[i],
            )

    optim_conf["def_load_config"] = def_load_cfg
    optim_conf["number_of_deferrable_loads"] = num_def_loads


_SCHEDULE_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def flatten_room_schedule(
    week_schedule: dict,
    room_name: str,
    start_time: pd.Timestamp,
    step_td: pd.Timedelta,
    horizon_steps: int,
    default_min: float = 18.0,
    default_max: float = 24.0,
) -> tuple[list[float], list[float]]:
    """Flatten a thermal_comfort.html weekly schedule into flat per-timestep
    min/max temperature arrays aligned to the optimization horizon.

    week_schedule shape: {dayName: {roomName: [{slot, temp_min, temp_max}, ...48]}}
    (dense 48 half-hour slots per day - this is a direct lookup, not
    interpolation; every timestep computes its own day/slot from the
    absolute timestamp, so midnight and week-boundary crossings need no
    special-casing). Missing/invalid entries fall back to (default_min,
    default_max) rather than raising, so a partially-filled or absent
    schedule never breaks optimization.

    :param week_schedule: The stored weekSchedule dict (see room_thermal_schedule.json)
    :param room_name: Room name to look up within each day's schedule
    :param start_time: Timezone-aware timestamp for horizon step 0
    :param step_td: Optimization timestep duration
    :param horizon_steps: Number of timesteps to produce
    :param default_min: Fallback minimum temperature when no schedule entry applies
    :param default_max: Fallback maximum temperature when no schedule entry applies
    :return: (min_temperatures, max_temperatures), each of length horizon_steps
    """
    min_temps: list[float] = []
    max_temps: list[float] = []
    for i in range(horizon_steps):
        t = start_time + i * step_td
        day_name = _SCHEDULE_DAYS[t.dayofweek]
        slot = t.hour * 2 + (1 if t.minute >= 30 else 0)

        entry = None
        day_schedule = week_schedule.get(day_name) if isinstance(week_schedule, dict) else None
        if isinstance(day_schedule, dict):
            room_schedule = day_schedule.get(room_name)
            if isinstance(room_schedule, list) and 0 <= slot < len(room_schedule):
                entry = room_schedule[slot]

        if isinstance(entry, dict) and "temp_min" in entry and "temp_max" in entry:
            try:
                min_temps.append(float(entry["temp_min"]))
                max_temps.append(float(entry["temp_max"]))
                continue
            except (TypeError, ValueError):
                pass
        min_temps.append(default_min)
        max_temps.append(default_max)
    return min_temps, max_temps


def _load_heatpump_units(optim_conf: dict, logger: logging.Logger) -> list[dict]:
    """Load the heatpump_unit_* arrays (Heat Pump Units section) into a
    list of per-unit dicts, one per physical heat pump - control_mode,
    weather-curve slope/intercept and supply-temperature bounds are
    properties of the COMPRESSOR itself, shared by every room it heats,
    not something that varies per room (see heatpump_room_unit's own
    docstring). Always returns at least 1 unit (config_defaults.json's own
    heatpump_number_of_units default), matching today's single-heat-pump
    behavior for a config that has never touched this section.
    """
    num_units = max(1, int(optim_conf.get("heatpump_number_of_units", 1) or 1))
    # Migration: a config saved before this section existed has no
    # heatpump_unit_nominal_power at all, but may still carry the old
    # global heatpump_nominal_power value in its raw config.json (routed
    # here read-only via associations.csv, never rendered in the UI) -
    # seed unit 0's own default from it rather than silently resetting an
    # already-configured value back to a generic 3000W. Only applies while
    # heatpump_unit_nominal_power is entirely absent (a genuinely
    # unmigrated config) - once the user has saved the new section even
    # once, its own value always wins.
    legacy_nominal_power = optim_conf.get("heatpump_nominal_power")
    default_nominal_power = (
        float(legacy_nominal_power)
        if "heatpump_unit_nominal_power" not in optim_conf and legacy_nominal_power is not None
        else 3000.0
    )
    names = check_def_loads(num_units, optim_conf, "", "heatpump_unit_name", logger)
    nominal_power = check_def_loads(
        num_units, optim_conf, default_nominal_power, "heatpump_unit_nominal_power", logger
    )
    control_mode = check_def_loads(
        num_units, optim_conf, "fixed", "heatpump_unit_control_mode", logger
    )
    curve_slope = check_def_loads(num_units, optim_conf, -1.0, "heatpump_unit_curve_slope", logger)
    curve_intercept = check_def_loads(
        num_units, optim_conf, 40.0, "heatpump_unit_curve_intercept", logger
    )
    supply_temp_min = check_def_loads(
        num_units, optim_conf, 20.0, "heatpump_unit_supply_temp_min", logger
    )
    supply_temp_max = check_def_loads(
        num_units, optim_conf, 70.0, "heatpump_unit_supply_temp_max", logger
    )
    units = []
    for i in range(num_units):
        unit_name = str(names[i]).strip() if i < len(names) and names[i] else ""
        units.append(
            {
                "name": unit_name or f"heatpump_{i + 1}",
                "nominal_power": float(nominal_power[i]),
                "control_mode": str(control_mode[i]).strip().lower(),
                "curve_slope": float(curve_slope[i]),
                "curve_intercept": float(curve_intercept[i]),
                "supply_temp_min": float(supply_temp_min[i]),
                "supply_temp_max": float(supply_temp_max[i]),
            }
        )
    return units


def _resolve_room_heatpump_unit(
    units: list[dict], room_unit_index: int, room_name: str, logger: logging.Logger
) -> dict:
    """Resolve a room's own heatpump_room_unit index into one of
    _load_heatpump_units's unit dicts. Falls back to unit 0 (with a
    warning) for a stale/out-of-range index (e.g. a unit was removed after
    a room was already pointed at it) rather than crashing the whole
    config build over one room's bad reference."""
    if 0 <= room_unit_index < len(units):
        return units[room_unit_index]
    logger.warning(
        "Room %s: heatpump_room_unit=%d is out of range (only %d heat pump "
        "unit(s) configured) - falling back to unit 0.",
        room_name,
        room_unit_index,
        len(units),
    )
    return units[0]


async def _append_room_thermal_loads(params: dict, logger: logging.Logger, emhass_conf: dict) -> None:
    """Map configured rooms and the whole-house heat pump dispatch unit into
    def_load_config thermal_battery entries, following the same pattern as
    _append_boiler_thermal_battery_loads. Only appends a room/dispatch load
    for entities the user has actually configured (heatpump_room_names /
    heatpump_dispatch_control_entity) - a room with no real controllable
    hardware yet simply isn't appended.

    Records params["passed_data"]["room_load_indices"] (room name -> deferrable
    load index) and params["passed_data"]["heatpump_dispatch_load_index"] so
    later stages (schedule flattening, sensor readback, publishing) know which
    deferrable load index corresponds to which real-world control point.
    """
    optim_conf = params.get("optim_conf", {})
    _strip_auto_appended_loads(optim_conf, {"room_auto", "heatpump_dispatch_auto"})
    if not optim_conf.get("set_use_heatpump", False):
        return
    if optim_conf.get("heatpump_config_mode", "room_list") == "graph_topology":
        # Heating is configured as a heat_topology graph instead - the room
        # list is deliberately inert in this mode (see treat_runtimeparams's
        # heat_topology gating), so there is nothing to append here.
        return

    num_def_loads = int(optim_conf.get("number_of_deferrable_loads", 0) or 0)

    # Ensure required generic deferrable-load vectors exist and are padded to
    # the current load count before appending, same defensive pattern as
    # _append_boiler_thermal_battery_loads (guards against this function being
    # called without the boiler pass having already normalized these lists).
    base_default_lists = {
        "nominal_power_of_deferrable_loads": 0.0,
        "minimum_power_of_deferrable_loads": 0.0,
        "operating_hours_of_each_deferrable_load": 0,
        "start_timesteps_of_each_deferrable_load": 0,
        "end_timesteps_of_each_deferrable_load": 0,
        "set_deferrable_startup_penalty": 0.0,
        "set_deferrable_load_single_constant": False,
        "treat_deferrable_load_as_semi_cont": False,
        "load_type": "fixed_power_non_splittable",
        "load_dispatch_mode": "hours",
        "required_energy_kwh_of_each_deferrable_load": 0.0,
        "load_phase": "",
    }
    for key, default in base_default_lists.items():
        optim_conf[key] = check_def_loads(num_def_loads, optim_conf, default, key, logger)

    # Forecast horizon for static list defaults (same computation as boiler).
    step_td = params.get("retrieve_hass_conf", {}).get(
        "optimization_time_step", pd.to_timedelta(30, "min")
    )
    if isinstance(step_td, (int, float)):
        step_td = pd.to_timedelta(step_td, "minutes")
    delta = optim_conf.get("delta_forecast_daily", pd.to_timedelta(1, "days"))
    if isinstance(delta, (int, float)):
        delta = pd.to_timedelta(delta, "days")
    try:
        horizon_steps = max(1, int(delta / step_td))
    except Exception:
        horizon_steps = 48

    def_load_cfg = optim_conf.get("def_load_config", []) or []
    while len(def_load_cfg) < num_def_loads:
        def_load_cfg.append({})

    def _append_generic_vectors(nominal_power: float, semi_cont: bool, phase: str = "") -> None:
        optim_conf["nominal_power_of_deferrable_loads"].append(max(0.0, nominal_power))
        optim_conf["minimum_power_of_deferrable_loads"].append(0.0)
        optim_conf["operating_hours_of_each_deferrable_load"].append(0)
        optim_conf["start_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["end_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["set_deferrable_startup_penalty"].append(0.0)
        optim_conf["set_deferrable_load_single_constant"].append(False)
        optim_conf["treat_deferrable_load_as_semi_cont"].append(semi_cont)
        optim_conf["load_type"].append("fixed_power_non_splittable")
        optim_conf["load_dispatch_mode"].append("hours")
        optim_conf["required_energy_kwh_of_each_deferrable_load"].append(0.0)
        optim_conf["load_phase"].append(str(phase or "").strip())

    room_load_indices: dict[str, int] = {}

    # --- Per-room loads (only rooms with a real name configured) ---
    num_rooms = int(optim_conf.get("heatpump_number_of_rooms", 0) or 0)
    if num_rooms > 0:
        # These four are live HA sensor entity ids, so - like
        # heatpump_room_temp_sensors - they live in retrieve_hass_conf, not
        # optim_conf. Padding them here (self-healing on every build_params
        # call, not just a static config_defaults.json default) matters
        # because buildParamElement() on the config UI renders zero <input>
        # elements for a genuinely empty array: any of these left shorter
        # than num_rooms - not just missing entirely - silently hides that
        # room's field from the page instead of just leaving it blank.
        retrieve_hass_conf = params.get("retrieve_hass_conf", {})
        check_def_loads(num_rooms, retrieve_hass_conf, "", "heatpump_room_temp_sensors", logger)
        check_def_loads(num_rooms, retrieve_hass_conf, "", "heatpump_room_valve_sensors", logger)
        check_def_loads(num_rooms, retrieve_hass_conf, "", "heatpump_room_blind_sensors", logger)
        check_def_loads(num_rooms, retrieve_hass_conf, "", "heatpump_room_window_sensors", logger)
        check_def_loads(num_rooms, retrieve_hass_conf, "", "heatpump_room_door_sensors", logger)

        room_names = check_def_loads(num_rooms, optim_conf, "", "heatpump_room_names", logger)
        # Every room appended below - and the whole-house dispatch load
        # further down, if heatpump_dispatch_control_entity is configured -
        # unconditionally joins the SAME shared heatpump_group_member group
        # (there is no sub-grouping today). weather_curve mode's exact-MILP
        # supply-temperature dispatch (see optimization.py) only applies
        # cleanly to a single-member group - peeking at both counts here,
        # before any room's thermal_cfg is built, lets each room's own
        # supply_temp_is_decision_variable marker be set correctly the
        # first time, rather than needing a second patch-up pass once the
        # dispatch block (which runs after this loop) is also known.
        _valid_room_count = sum(1 for n in room_names if str(n).strip())
        _dispatch_entity_configured = bool(
            str(optim_conf.get("heatpump_dispatch_control_entity", "") or "").strip()
        )
        heat_source_group_member_count = _valid_room_count + (
            1 if _dispatch_entity_configured else 0
        )
        room_phase = check_def_loads(num_rooms, optim_conf, "", "heatpump_room_phase", logger)
        check_def_loads(num_rooms, optim_conf, "modulating", "heatpump_room_valve_mode", logger)
        room_min = check_def_loads(num_rooms, optim_conf, 18.0, "heatpump_room_min_temperature", logger)
        room_max = check_def_loads(num_rooms, optim_conf, 24.0, "heatpump_room_max_temperature", logger)
        room_target = check_def_loads(
            num_rooms, optim_conf, 21.0, "heatpump_room_target_temperature", logger
        )
        room_power = check_def_loads(num_rooms, optim_conf, 1500.0, "heatpump_room_nominal_power", logger)
        room_supply_temp = check_def_loads(
            num_rooms, optim_conf, 35.0, "heatpump_room_supply_temperature", logger
        )
        # Which physical heat pump unit serves each room - control_mode/
        # weather-curve/supply-temp-bounds are properties of the UNIT (one
        # compressor, one set of physical limits), not of any one room it
        # happens to heat, so they're resolved from _load_heatpump_units
        # below rather than read as a per-room array here.
        room_unit = check_def_loads(num_rooms, optim_conf, 0, "heatpump_room_unit", logger)
        heatpump_units = _load_heatpump_units(optim_conf, logger)
        # plant_conf["heatpump_nominal_power"] used to be the single,
        # hand-configured global field every aggregate-duty/dispatch-gating
        # call site (optimization.py::_build_aggregate_heatpump_duty_expr
        # and friends, command_line.py's own duty derivations,
        # compute_aggregate_heatpump_duty) reads as "the whole house's heat
        # pump capacity" - now that nominal power is configured per unit
        # (Heat Pump Units), those call sites keep working unchanged by
        # reading the SAME key, populated here as the sum across every
        # configured unit. The one place that genuinely needs a SINGLE
        # unit's own value rather than the total - the shared-power-group
        # cap - reads each room's own resolved
        # thermal_cfg["heatpump_unit_nominal_power"] instead (see below),
        # not this aggregate.
        params.setdefault("plant_conf", {})["heatpump_nominal_power"] = sum(
            u["nominal_power"] for u in heatpump_units
        )
        room_volume = check_def_loads(num_rooms, optim_conf, 15.0, "heatpump_room_volume", logger)
        room_shared_group = check_def_loads(num_rooms, optim_conf, 0, "heatpump_room_shared_group", logger)
        room_coupled_neighbors_raw = check_def_loads(
            num_rooms, optim_conf, "", "heatpump_room_coupled_neighbors", logger
        )
        room_coupling_conductance_raw = check_def_loads(
            num_rooms, optim_conf, "", "heatpump_room_coupling_conductance", logger
        )
        room_self_learning = check_def_loads(
            num_rooms, optim_conf, False, "heatpump_room_self_learning_only", logger
        )
        # RC-physics dispatch (opt-in per room, mutually exclusive with
        # heatpump_room_self_learning_only for the same room - see the
        # installation loop below, self-learning takes priority when both
        # are set on one room, matching how a room only ever gets ONE
        # dispatch mechanism).
        room_rc_model_only = check_def_loads(
            num_rooms, optim_conf, False, "heatpump_room_rc_model_only", logger
        )

        # heatpump_model_family: "physics" swaps the flat thermal-loss-only
        # model (custom_heating_demand_profile forced to zero) for a real
        # per-room envelope/RC demand model, reusing the same
        # calculate_heating_demand_physics/resolve_thermal_battery_cop
        # machinery heat_topology's building_demand consumers already use
        # (optimization.py). Any other value (including "machine_learning"/
        # "deep_learning", not yet wired to live dispatch) keeps today's
        # existing thermal-loss-only behavior unchanged.
        family = str(optim_conf.get("heatpump_model_family", "simple") or "simple").lower()
        use_physics = family == "physics"
        if use_physics:
            room_u_value = check_def_loads(num_rooms, optim_conf, 0.5, "heatpump_room_u_value", logger)
            room_envelope_area = check_def_loads(
                num_rooms, optim_conf, 40.0, "heatpump_room_envelope_area", logger
            )
            room_ventilation_rate = check_def_loads(
                num_rooms, optim_conf, 0.5, "heatpump_room_ventilation_rate", logger
            )
            room_window_area = check_def_loads(
                num_rooms, optim_conf, 0.0, "heatpump_room_window_area", logger
            )
            room_shgc = check_def_loads(num_rooms, optim_conf, 0.6, "heatpump_room_shgc", logger)
            room_internal_gains_factor = check_def_loads(
                num_rooms, optim_conf, 0.0, "heatpump_room_internal_gains_factor", logger
            )
            room_thermal_inertia = check_def_loads(
                num_rooms, optim_conf, 2.0, "heatpump_room_thermal_inertia_time_constant", logger
            )
            room_carnot_efficiency = check_def_loads(
                num_rooms, optim_conf, 0.4, "heatpump_room_carnot_efficiency", logger
            )
            room_blind_type = check_def_loads(
                num_rooms, optim_conf, "none", "heatpump_room_blind_type", logger
            )

        # Optional per-room weekly comfort-schedule overlay. Safe no-op (falls
        # back to the static min/max above) if no schedule has been saved yet
        # via the /room-schedule endpoint / thermal_comfort.html.
        schedule_blob = await load_json_blob(
            emhass_conf, "room_thermal_schedule.json", logger, default={}
        )
        week_schedule = schedule_blob.get("weekSchedule") if isinstance(schedule_blob, dict) else None
        try:
            schedule_start_time = pd.Timestamp.now(
                tz=params.get("retrieve_hass_conf", {}).get("time_zone", UTC)
            )
        except Exception:
            schedule_start_time = pd.Timestamp.now(tz=UTC)

        # heatpump_room_coupled_neighbors is entered room-relative (0-based
        # within the Rooms tab, matching what the UI actually shows the
        # user) but optimization.py keys predicted_temps by absolute
        # def_load_cfg/p_deferrable index - translate here, once, using the
        # same base offset room_load_indices below will use.
        room_index_base = num_def_loads

        # Learned inter-room thermal coupling (opt-in, see
        # arx_model_coupling_source's own description). Only
        # loaded - and only ever overrides a pair - when the user has
        # explicitly opted in; the default "informational" never touches
        # this file, matching how HybridHeatPumpLR/refit_hybrid_heatpump_model
        # never influence dispatch either. A non-positive learned value is
        # treated the same as a non-positive manual one: skipped, not
        # applied (see the per-pair loop below).
        learned_coupling: dict[tuple[str, str], float] = {}
        if optim_conf.get("arx_model_coupling_source", "informational") == "auto_dispatch":
            coupling_blob = await load_json_blob(
                emhass_conf, "arx_model_coupling.json", logger, default={}
            )
            pairs = coupling_blob.get("pairs", []) if isinstance(coupling_blob, dict) else []
            for pair in pairs:
                try:
                    room_a = str(pair["room_a"]).strip()
                    room_b = str(pair["room_b"]).strip()
                    learned_g = float(pair["conductance_kw_per_k"])
                except (KeyError, TypeError, ValueError):
                    continue
                if room_a and room_b and learned_g > 0:
                    learned_coupling[tuple(sorted((room_a, room_b)))] = learned_g

        # ARX-model dispatch coefficients (opt-in per room via
        # heatpump_room_self_learning_only). Same "small derived JSON blob,
        # gated, never crashes on absence" pattern as learned_coupling above -
        # only loaded when at least one room is flagged, so a config with no
        # flagged rooms (the shipped default) pays zero cost here.
        dispatch_coeffs: dict = {}
        house_elec_coeffs: dict | None = None
        if any(bool(v) for v in room_self_learning):
            dispatch_blob = await load_json_blob(
                emhass_conf, "arx_model_room_dispatch_coefficients.json", logger, default={}
            )
            dispatch_coeffs = dispatch_blob.get("rooms", {}) if isinstance(dispatch_blob, dict) else {}
            # Whole-house electric-draw regression (theta_elec_) - only
            # relevant to a room using weather_curve's exact-MILP dispatch
            # (below), which needs it to price the real predicted
            # electricity draw for whatever supply temperature it picks,
            # instead of treating p_deferrable[k] as a free decision
            # variable (see optimization.py's own docstring for why).
            house_elec_coeffs = (
                dispatch_blob.get("house_elec") if isinstance(dispatch_blob, dict) else None
            )
            if not isinstance(house_elec_coeffs, dict) or not house_elec_coeffs.get("theta"):
                house_elec_coeffs = None

        # RC-physics dispatch coefficients (opt-in per room via
        # heatpump_room_rc_model_only) - same "small derived JSON blob,
        # gated, never crashes on absence" pattern as dispatch_coeffs above.
        # Unlike the ARX model's own per-room coefficients, RC is a
        # single house-wide fitted model (rc_model_params.json,
        # see thermal_mass_physics.PARAM_NAMES) - every RC-flagged room
        # shares the exact same params dict.
        rc_physics_params: dict | None = None
        if any(bool(v) for v in room_rc_model_only):
            rc_blob = await load_json_blob(
                emhass_conf, "rc_model_params.json", logger, default=None
            )
            if isinstance(rc_blob, dict) and isinstance(rc_blob.get("params"), dict):
                rc_physics_params = rc_blob["params"]
        room_name_to_index = {
            str(n).strip(): room_index_base + i
            for i, n in enumerate(room_names)
            if str(n).strip()
        }

        for i in range(num_rooms):
            name = str(room_names[i]).strip()
            if not name:
                continue
            target_temp = float(room_target[i])
            room_min_temps = [float(room_min[i])] * horizon_steps
            room_max_temps = [float(room_max[i])] * horizon_steps
            if week_schedule:
                try:
                    room_min_temps, room_max_temps = flatten_room_schedule(
                        week_schedule,
                        name,
                        schedule_start_time,
                        step_td,
                        horizon_steps,
                        default_min=float(room_min[i]),
                        default_max=float(room_max[i]),
                    )
                except Exception as e:
                    logger.warning("Failed to apply comfort schedule for room %s: %s", name, e)
            neighbor_indices_relative = [
                int(v) for v in _parse_profile_to_float_list(room_coupled_neighbors_raw[i])
            ]
            conductance_raw = _parse_profile_to_float_list(room_coupling_conductance_raw[i])
            if len(conductance_raw) != len(neighbor_indices_relative):
                logger.warning(
                    "Room %s: coupled_neighbors has %d entries but "
                    "coupling_conductance has %d - truncating to the shorter "
                    "of the two rather than guessing the missing pairing.",
                    name,
                    len(neighbor_indices_relative),
                    len(conductance_raw),
                )
            pair_count = min(len(neighbor_indices_relative), len(conductance_raw))
            coupled_neighbors = []
            coupling_conductance = []
            for rel_idx, conductance in zip(
                neighbor_indices_relative[:pair_count], conductance_raw[:pair_count], strict=True
            ):
                if rel_idx < 0 or rel_idx >= num_rooms or rel_idx == i or conductance <= 0:
                    logger.warning(
                        "Room %s: coupled_neighbors entry %d (conductance %s) is out of "
                        "range, self-referencing, or non-positive (num_rooms=%d) - skipped.",
                        name,
                        rel_idx,
                        conductance,
                        num_rooms,
                    )
                    continue
                neighbor_name = str(room_names[rel_idx]).strip()
                pair_key = tuple(sorted((name, neighbor_name))) if neighbor_name else None
                if pair_key is not None and pair_key in learned_coupling:
                    learned_g = learned_coupling[pair_key]
                    logger.debug(
                        "Room %s <-> %s: using ARX-model learned coupling "
                        "%.4f kW/K (manual value %.4f kW/K overridden - "
                        "arx_model_coupling_source=auto_dispatch).",
                        name,
                        neighbor_name,
                        learned_g,
                        conductance,
                    )
                    conductance = learned_g
                coupled_neighbors.append(room_index_base + rel_idx)
                coupling_conductance.append(conductance)
            thermal_cfg = {
                "name": name,
                "supply_temperature": float(room_supply_temp[i]),
                "volume": max(0.05, float(room_volume[i])),
                "start_temperature": target_temp,
                "min_temperatures": room_min_temps,
                "max_temperatures": room_max_temps,
                "indoor_target_temperature": target_temp,
                "shared_power_group": int(room_shared_group[i]),
                "coupled_neighbors": coupled_neighbors,
                "coupling_conductance_kw_per_k": coupling_conductance,
                "_source": "room_auto",
            }
            # weather_curve mode: a real, per-timestep supply-temperature
            # curve (reused by resolve_thermal_battery_cop's existing
            # heating_curve support, utils.py, regardless of self-learning
            # status - a more realistic COP estimate than the flat
            # "supply_temperature" constant above, for free). Only a room
            # that ALSO has self_learning_only on, in a heat-source group
            # with exactly one member, gets the further step of treating
            # supply temperature as a genuine MILP decision variable (see
            # optimization.py) - anything else falls back to today's
            # two-pass dispatch, with a visible reason surfaced in this
            # room's own refit/tune/forecast result (not just a debug log)
            # so switching this mode on never silently gives less than the
            # user thinks they're getting.
            room_unit_index = int(room_unit[i]) if i < len(room_unit) else 0
            unit = _resolve_room_heatpump_unit(heatpump_units, room_unit_index, name, logger)
            supply_temp_mode = unit["control_mode"]
            if supply_temp_mode == "weather_curve":
                thermal_cfg["heating_curve"] = {
                    "slope": unit["curve_slope"],
                    "offset": unit["curve_intercept"],
                    "min_supply": unit["supply_temp_min"],
                    "max_supply": unit["supply_temp_max"],
                }
            # Every room (and the dispatch load below) counts toward the
            # shared heat pump's aggregate duty signal, regardless of
            # whether this specific room is self-learning-flagged -
            # optimization.py::_build_aggregate_heatpump_duty_expr sums
            # this marker to reconstruct room_load_indices without
            # needing to see passed_data.
            thermal_cfg["heatpump_group_member"] = True
            # This room's resolved unit's own nominal power - read by
            # optimization.py::_add_shared_heatpump_group_constraints to
            # cap a shared_power_group against the ACTUAL unit those rooms
            # share, instead of a single house-wide value (see
            # heatpump_room_unit's own docstring for why control_mode/
            # nominal_power are unit properties, not per-room ones).
            thermal_cfg["heatpump_unit_nominal_power"] = unit["nominal_power"]
            thermal_cfg["heatpump_unit_name"] = unit["name"]
            if room_self_learning[i]:
                fitted = dispatch_coeffs.get(name)
                if fitted is None:
                    logger.warning(
                        "Room %s: heatpump_room_self_learning_only is set but no fitted "
                        "ARX-model dispatch coefficients exist yet for this room "
                        "(run the arx-model-refit action first) - falling back to "
                        "the physics/simple model until the next successful refit.",
                        name,
                    )
                else:
                    neighbor_indices: dict[str, int] = {}
                    kept_features: list = []
                    kept_theta: list = []
                    for fname, coef in zip(
                        fitted.get("feature_names", []), fitted.get("theta", []), strict=True
                    ):
                        if fname.startswith("neighbor_diff::"):
                            neighbor_name = fname.removeprefix("neighbor_diff::")
                            neighbor_idx = room_name_to_index.get(neighbor_name)
                            if neighbor_idx is None:
                                logger.warning(
                                    "Room %s: fitted ARX model references "
                                    "neighbor '%s' which is no longer configured - that term is "
                                    "dropped from live dispatch (fidelity will improve after the "
                                    "next refit).",
                                    name,
                                    neighbor_name,
                                )
                                continue
                            neighbor_indices[neighbor_name] = neighbor_idx
                        kept_features.append(fname)
                        kept_theta.append(coef)
                    thermal_cfg["self_learning_dispatch"] = {
                        "feature_names": kept_features,
                        "theta": kept_theta,
                        "neighbor_indices": neighbor_indices,
                    }
                    # weather_curve's exact-MILP dispatch (optimization.py)
                    # needs BOTH this room's own fitted temperature equation
                    # (just built above) AND the whole-house electric-draw
                    # regression (house_elec_coeffs, loaded once above) to
                    # price real predicted electricity for whatever supply
                    # temperature it picks - and only applies to a
                    # single-member heat-source group (see Scope in the
                    # design plan). Checked here, not earlier, so every
                    # failure reason (no fit yet vs. no house_elec yet vs.
                    # multi-member group) can be distinguished in the
                    # visible fallback note instead of a generic one.
                    if supply_temp_mode == "weather_curve":
                        if heat_source_group_member_count != 1:
                            thermal_cfg["dispatch_mode_fallback_reason"] = (
                                f"two-pass (fallback - {heat_source_group_member_count} "
                                "shared heat-source group members, weather-curve MILP "
                                "needs exactly 1)"
                            )
                        elif house_elec_coeffs is None:
                            thermal_cfg["dispatch_mode_fallback_reason"] = (
                                "two-pass (fallback - no whole-house electric-draw "
                                "coefficients yet, run arx-model-refit again)"
                            )
                        else:
                            thermal_cfg["supply_temp_is_decision_variable"] = True
                            thermal_cfg["self_learning_dispatch_elec"] = {
                                "feature_names": list(house_elec_coeffs.get("feature_names", [])),
                                "theta": [float(c) for c in house_elec_coeffs.get("theta", [])],
                            }
                if room_rc_model_only[i]:
                    logger.warning(
                        "Room %s: both heatpump_room_self_learning_only and "
                        "heatpump_room_rc_model_only are set - the ARX model "
                        "takes priority, heatpump_room_rc_model_only is ignored for this room.",
                        name,
                    )
            elif room_rc_model_only[i]:
                if rc_physics_params is None:
                    logger.warning(
                        "Room %s: heatpump_room_rc_model_only is set but no fitted "
                        "RC-physics model exists yet (run rc-model-refit or "
                        "rc-model-tune first) - falling back to the physics/simple "
                        "model until the next successful refit.",
                        name,
                    )
                else:
                    thermal_cfg["rc_physics_dispatch"] = {"params": dict(rc_physics_params)}
            if (
                supply_temp_mode == "weather_curve"
                and room_self_learning[i]
                and "dispatch_mode_fallback_reason" not in thermal_cfg
                and not thermal_cfg.get("supply_temp_is_decision_variable")
            ):
                thermal_cfg["dispatch_mode_fallback_reason"] = (
                    "two-pass (fallback - weather_curve requires "
                    "heatpump_room_self_learning_only with a fitted model for this room)"
                )
            if use_physics:
                # All 4 required keys set together: _add_thermal_battery_constraints
                # only takes the physics branch when every one of them is
                # present, so a partial set would silently fall through to
                # the degree-day branch and then KeyError on
                # hc["specific_heating_demand"].
                thermal_cfg.update(
                    {
                        "u_value": float(room_u_value[i]),
                        "envelope_area": float(room_envelope_area[i]),
                        "ventilation_rate": float(room_ventilation_rate[i]),
                        "heated_volume": max(0.05, float(room_volume[i])),
                        "window_area": float(room_window_area[i]),
                        "shgc": float(room_shgc[i]),
                        "internal_gains_factor": float(room_internal_gains_factor[i]),
                        "thermal_inertia_time_constant": float(room_thermal_inertia[i]),
                        "carnot_efficiency": float(room_carnot_efficiency[i]),
                        # Static config only - live blind *position* is resolved
                        # per-solve instead, via command_line.py::_build_room_blind_positions
                        # (mirrors the existing supply_temperature-static vs.
                        # def_init_temp-live split for start temperature).
                        "blind_type": str(room_blind_type[i]).strip().lower() or "none",
                    }
                )
            else:
                thermal_cfg["custom_heating_demand_profile"] = [0.0] * horizon_steps
            def_load_cfg.append({"thermal_battery": thermal_cfg})
            _append_generic_vectors(float(room_power[i]), semi_cont=False, phase=room_phase[i])

            slug_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"room_{i + 1}"
            passed_data = params.setdefault("passed_data", {})
            passed_data.setdefault("custom_predicted_temperature_id", []).append(
                {
                    "entity_id": f"sensor.temp_predicted{num_def_loads}",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                    "friendly_name": f"Predicted temperature {num_def_loads}",
                }
            )
            passed_data.setdefault("custom_room_target_temp_id", []).append(
                {
                    "entity_id": f"sensor.room_target_temp_{slug_name}",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                    "friendly_name": f"{name} Target Temperature",
                }
            )
            # Always appended, one entry per room (same convention as
            # custom_room_target_temp_id above, so list position stays
            # aligned with room_load_indices's own enumeration order for
            # command_line.py::_publish_room_supply_temp_target) - only
            # actually PUBLISHED for a room whose solved results carry a
            # supply_temp_target_heater{k} column, i.e. one actually using
            # weather_curve's exact-MILP dispatch (see optimization.py::
            # _add_self_learning_dispatch_milp_constraints). An additional,
            # optional signal alongside custom_room_target_temp_id, letting
            # the user choose which one to wire into their own automation
            # instead of (or alongside) the indoor-temperature target.
            passed_data.setdefault("custom_room_supply_temp_target_id", []).append(
                {
                    "entity_id": f"sensor.room_supply_temp_target_{slug_name}",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                    "friendly_name": f"{name} Supply Temperature Target",
                }
            )

            room_load_indices[name] = num_def_loads
            num_def_loads += 1

    # --- Whole-house heat pump dispatch load (only if a real control entity
    # is configured - otherwise there's nothing for an automation to drive) ---
    dispatch_entity = str(optim_conf.get("heatpump_dispatch_control_entity", "") or "").strip()
    dispatch_load_index = None
    if dispatch_entity:
        target_temp = float(optim_conf.get("heatpump_dispatch_target_temperature", 20.0))
        thermal_cfg = {
            "name": "heatpump_dispatch",
            "supply_temperature": float(optim_conf.get("heatpump_dispatch_supply_temperature", 35.0)),
            "volume": max(0.05, float(optim_conf.get("heatpump_dispatch_volume", 20.0))),
            "start_temperature": target_temp,
            "min_temperatures": [float(optim_conf.get("heatpump_dispatch_min_temperature", 18.0))]
            * horizon_steps,
            "max_temperatures": [float(optim_conf.get("heatpump_dispatch_max_temperature", 22.0))]
            * horizon_steps,
            "indoor_target_temperature": target_temp,
            "custom_heating_demand_profile": [0.0] * horizon_steps,
            "_source": "heatpump_dispatch_auto",
            "heatpump_group_member": True,
        }
        def_load_cfg.append({"thermal_battery": thermal_cfg})
        _append_generic_vectors(
            float(optim_conf.get("heatpump_dispatch_nominal_power", 3000.0)), semi_cont=True
        )

        passed_data = params.setdefault("passed_data", {})
        passed_data.setdefault("custom_predicted_temperature_id", []).append(
            {
                "entity_id": f"sensor.temp_predicted{num_def_loads}",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "friendly_name": f"Predicted temperature {num_def_loads}",
            }
        )
        passed_data["custom_heatpump_dispatch_target_id"] = {
            "entity_id": "sensor.heatpump_dispatch_target",
            "device_class": "",
            "unit_of_measurement": "",
            "friendly_name": "Heat Pump Dispatch Target",
        }

        dispatch_load_index = num_def_loads
        num_def_loads += 1

    if not room_load_indices and dispatch_load_index is None:
        return

    optim_conf["def_load_config"] = def_load_cfg
    optim_conf["number_of_deferrable_loads"] = num_def_loads
    params.setdefault("passed_data", {})
    params["passed_data"]["room_load_indices"] = room_load_indices
    if dispatch_load_index is not None:
        params["passed_data"]["heatpump_dispatch_load_index"] = dispatch_load_index


def _append_rc_model_forecast_targets(params: dict, logger: logging.Logger) -> None:
    """Register the entity definitions for the rc-model-forecast sensors.

    Not a deferrable load - this feature never touches optim_conf/def_load_config,
    it only registers where command_line.compute_rc_model_forecast should publish
    its result sensors (indoor_temp_forecast, heating_needed_by, and - opt-in,
    see below - heating_electric_power_forecast) once rc_model_forecast_enabled
    is set. No-op when disabled.
    """
    optim_conf = params.get("optim_conf", {})
    if not optim_conf.get("rc_model_forecast_enabled", False):
        return

    passed_data = params.setdefault("passed_data", {})
    passed_data["custom_indoor_temp_forecast_id"] = {
        "entity_id": "sensor.indoor_temp_forecast",
        "device_class": "temperature",
        "unit_of_measurement": "°C",
        "friendly_name": "Indoor Temperature Forecast",
    }
    passed_data["custom_heating_needed_by_id"] = {
        "entity_id": "sensor.heating_needed_by",
        "device_class": "",
        "unit_of_measurement": "",
        "friendly_name": "Heating Needed By",
    }
    # Only registered when the RC model is actually being fit against real
    # electric-power data (see _fit_temperature_params's own
    # fit_electric_power docstring) - otherwise emitter_power_scale_w sits
    # at its DEFAULT_X0 seed of 0.0 and this sensor would just publish an
    # all-zero curve nobody asked for.
    if optim_conf.get("rc_model_refit_fit_electric_power_enabled", False):
        passed_data["custom_heating_electric_power_forecast_id"] = {
            "entity_id": "sensor.heating_electric_power_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Heating Electric Power Forecast",
        }
    logger.debug("RC model forecast targets registered")


def _append_hybrid_heatpump_forecast_targets(params: dict, logger: logging.Logger) -> None:
    """Register the entity definitions for the hybrid heat pump forecast sensors.

    Standalone sibling of _append_rc_model_forecast_targets, for a different
    feature (command_line.compute_hybrid_heatpump_forecast) - registers where
    to publish the predicted electric-power/gas-consumption sensors once
    hybrid_heatpump_forecast_enabled is set. No-op when disabled.
    """
    optim_conf = params.get("optim_conf", {})
    if not optim_conf.get("hybrid_heatpump_forecast_enabled", False):
        return

    passed_data = params.setdefault("passed_data", {})
    passed_data["custom_hybrid_electric_forecast_id"] = {
        "entity_id": "sensor.hybrid_heatpump_electric_forecast",
        "device_class": "power",
        "unit_of_measurement": "W",
        "friendly_name": "Hybrid Heat Pump Electric Power Forecast",
    }
    passed_data["custom_hybrid_gas_forecast_id"] = {
        "entity_id": "sensor.hybrid_heatpump_gas_forecast",
        "device_class": "gas",
        "unit_of_measurement": "m³",
        "friendly_name": "Hybrid Heat Pump Gas Consumption Forecast",
    }
    logger.debug("Hybrid heat pump forecast targets registered")


def compute_aggregate_heatpump_duty(
    opt_res: pd.DataFrame,
    room_load_indices: dict[str, int],
    dispatch_load_index: int | None,
    heatpump_nominal_power: float,
) -> pd.Series:
    """Aggregate the heat pump's own duty (0-1 part-load fraction) across every
    room/dispatch load it drives, from an already-solved optimization result.

    duty[t] = clip(sum(P_deferrable{k}[t] for k in the heat-pump-driven load
    indices) / heatpump_nominal_power, 0, 1). Deliberately keyed off
    room_load_indices/dispatch_load_index (populated by
    _append_room_thermal_loads) rather than heatpump_room_shared_group: a
    single-room house has group 0 (no explicit sharing declared) but still
    has exactly one physical pump needing a duty signal.

    Used to feed both refit_hybrid_heatpump_model/compute_hybrid_heatpump_forecast
    and the ARX model's "heatpump_duty" input at forecast
    time, replacing the single whole-house heatpump_duty_sensor reading with
    a real, multi-room-aware trajectory once a solved plan exists.

    :param opt_res: A solved optimization result DataFrame (or the loaded
        opt_res_latest.csv) with P_deferrable{k} columns.
    :param room_load_indices: room name -> deferrable load index, from
        params["passed_data"]["room_load_indices"].
    :param dispatch_load_index: the whole-house heat pump dispatch load's
        index, from params["passed_data"].get("heatpump_dispatch_load_index"),
        or None if not configured.
    :param heatpump_nominal_power: plant_conf["heatpump_nominal_power"], in
        watts. Returns an all-zero Series if this is not positive.
    :return: A Series indexed the same as opt_res, clipped to [0, 1].
    """
    load_indices = set(room_load_indices.values())
    if dispatch_load_index is not None:
        load_indices.add(int(dispatch_load_index))
    cols = [f"P_deferrable{k}" for k in load_indices if f"P_deferrable{k}" in opt_res.columns]
    if not cols or heatpump_nominal_power <= 0:
        return pd.Series(0.0, index=opt_res.index)
    total_power = opt_res[cols].sum(axis=1)
    return (total_power / heatpump_nominal_power).clip(lower=0.0, upper=1.0)


def resolve_incremental_series(
    series: pd.Series,
    name: str,
    logger: logging.Logger,
    negative_fraction_threshold: float = 0.05,
    rate_dt_hours: float | None = None,
) -> pd.Series:
    """Detect whether ``series`` is a cumulative meter reading (e.g. a
    lifetime gas/energy totalizer - the standard Home Assistant convention
    for a ``state_class: total_increasing`` sensor) and, if so, convert it
    to a per-row incremental delta; otherwise return it unchanged.

    Detection heuristic: genuinely already-incremental data (real
    consumption per interval) fluctuates constantly - a heat pump/boiler
    cycles on and off - so across a long real-world window it is essentially
    never monotonically non-decreasing by chance. A cumulative counter, by
    definition, can only stay flat or increase, except at rare meter resets.
    So if only a small minority of consecutive samples actually decrease,
    treat the series as cumulative and diff it; meter-reset dips (large
    negative diffs) are clipped to 0 rather than treated as negative
    consumption.

    Used wherever a "consumption per interval" style column is built from a
    raw retrieved sensor (heatpump_gas_meter_sensor being the concrete
    case - HA gas/energy meters are almost always cumulative by convention,
    while a refit's own fit target expects a per-interval delta), both for
    a full training-window column and for a single live "last value" seed
    (call with the raw series and take ``.iloc[-1]`` of the result).

    :param series: Raw retrieved sensor values, already time-ordered.
    :param name: Column/sensor name, used only for logging.
    :param negative_fraction_threshold: Maximum fraction of negative
        consecutive diffs still consistent with "cumulative meter with the
        occasional reset" - above this, the series is treated as already
        incremental and returned unchanged.
    :param rate_dt_hours: Only relevant when the series IS detected as
        cumulative. If given, the per-interval delta is additionally divided
        by this timestep (hours) and scaled x1000 to turn "energy used this
        interval" (assumed kWh, the near-universal HA convention for a
        cumulative electricity meter) into an average power in W - for a
        target that itself wants a rate (heatpump_power_sensor), unlike gas
        consumption which already wants a plain per-interval delta (leave
        None there). Verify this if a detected meter's cumulative unit isn't
        actually kWh.
    :return: The original series (unchanged) if not detected as cumulative,
        otherwise a same-length delta (or, with rate_dt_hours, rate) series
        (first value 0.0, meter resets clipped to 0.0).
    """
    s = series.astype(float)
    if len(s) < 3:
        return s
    diffs = s.diff()
    finite_diffs = diffs.dropna()
    if finite_diffs.empty:
        return s
    negative_fraction = float((finite_diffs < -1e-9).mean())
    if negative_fraction > negative_fraction_threshold:
        # Fluctuates too much to be a cumulative counter - already incremental.
        return s
    if not (finite_diffs > 1e-9).any():
        # Perfectly flat (or only ever equal) - a real cumulative counter
        # still ticks up at least occasionally over a long window; a series
        # that never once increases either is already a rate/instantaneous
        # reading sitting at a stable value, not a stalled totalizer.
        return s
    logger.info(
        "%s looks like a cumulative meter reading (only %.1f%% of consecutive "
        "samples decrease) - converting to a per-interval delta before use.",
        name,
        negative_fraction * 100.0,
    )
    delta = diffs.clip(lower=0.0)
    delta.iloc[0] = 0.0
    if rate_dt_hours is not None and rate_dt_hours > 0:
        delta = delta / rate_dt_hours * 1000.0
        logger.info(
            "%s: converting the per-interval kWh delta to an average power in W "
            "(divided by dt_hours=%.3f, assuming the raw cumulative meter reports "
            "in kWh) - verify this if your meter's cumulative unit isn't kWh.",
            name,
            rate_dt_hours,
        )
    return delta


def _append_arx_model_forecast_targets(params: dict, logger: logging.Logger) -> None:
    """Register the entity definitions for the ARX-model forecast
    sensors: one whole-house electric-power entity, one whole-house gas
    entity, and - unlike _append_hybrid_heatpump_forecast_targets, since
    this model uniquely predicts room temperature too - a *list* of one
    per-room temperature-forecast entity per configured room, mirroring how
    custom_predicted_temperature_id is already a per-room list built in
    _append_room_thermal_loads. No-op when disabled.
    """
    optim_conf = params.get("optim_conf", {})
    if not optim_conf.get("arx_model_forecast_enabled", False):
        return

    passed_data = params.setdefault("passed_data", {})
    passed_data["custom_arx_model_electric_forecast_id"] = {
        "entity_id": "sensor.arx_model_electric_forecast",
        "device_class": "power",
        "unit_of_measurement": "W",
        "friendly_name": "ARX Model Electric Power Forecast",
    }
    passed_data["custom_arx_model_gas_forecast_id"] = {
        "entity_id": "sensor.arx_model_gas_forecast",
        "device_class": "gas",
        "unit_of_measurement": "m³",
        "friendly_name": "ARX Model Gas Consumption Forecast",
    }
    room_names = optim_conf.get("heatpump_room_names", []) or []
    temp_forecast_ids = []
    for name in room_names:
        name = str(name).strip()
        if not name:
            continue
        slug = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "room"
        temp_forecast_ids.append(
            {
                "entity_id": f"sensor.arx_model_temp_forecast_{slug}",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "friendly_name": f"{name} ARX Model Temperature Forecast",
            }
        )
    passed_data["custom_arx_model_temp_forecast_id"] = temp_forecast_ids
    logger.debug("ARX-model forecast targets registered (%d rooms)", len(temp_forecast_ids))


async def _append_ev_deferrable_loads(params: dict, logger: logging.Logger) -> None:
    """Map configured EV chargers into plain semi-continuous deferrable loads.

    Unlike boilers/rooms, EV charging has no thermal model - it's a bare
    power dispatch problem bounded by the charger's real min/max power per
    phase mode. Registers params["passed_data"]["ev_load_indices"] (charger
    name -> deferrable load index) plus custom_ev_charge_mode_target_id /
    custom_ev_phase_target_id entity definitions consumed by the publish
    stage (see _translate_ev_power_to_mode / _publish_ev_targets in
    command_line.py).
    """
    optim_conf = params.get("optim_conf", {})
    _strip_auto_appended_loads(optim_conf, {"ev_auto"})
    if not optim_conf.get("set_use_ev_charger", False):
        return

    num_chargers = int(optim_conf.get("number_of_ev_chargers", 0) or 0)
    if num_chargers <= 0:
        return

    num_def_loads = int(optim_conf.get("number_of_deferrable_loads", 0) or 0)
    base_default_lists = {
        "nominal_power_of_deferrable_loads": 0.0,
        "minimum_power_of_deferrable_loads": 0.0,
        "operating_hours_of_each_deferrable_load": 0,
        "start_timesteps_of_each_deferrable_load": 0,
        "end_timesteps_of_each_deferrable_load": 0,
        "set_deferrable_startup_penalty": 0.0,
        "set_deferrable_load_single_constant": False,
        "treat_deferrable_load_as_semi_cont": False,
        "load_type": "fixed_power_non_splittable",
        "load_dispatch_mode": "hours",
        "required_energy_kwh_of_each_deferrable_load": 0.0,
        "load_phase": "",
    }
    for key, default in base_default_lists.items():
        optim_conf[key] = check_def_loads(num_def_loads, optim_conf, default, key, logger)

    def_load_cfg = optim_conf.get("def_load_config", []) or []
    while len(def_load_cfg) < num_def_loads:
        def_load_cfg.append({})

    ev_names = check_def_loads(num_chargers, optim_conf, "", "ev_charger_names", logger)
    ev_min_1p = check_def_loads(
        num_chargers, optim_conf, 1380.0, "ev_charge_power_min_1_phase", logger
    )
    ev_max_3p = check_def_loads(
        num_chargers, optim_conf, 11000.0, "ev_charge_power_max_3_phase", logger
    )
    ev_phase = check_def_loads(num_chargers, optim_conf, "", "ev_charger_phase", logger)

    ev_load_indices: dict[str, int] = {}
    for i in range(num_chargers):
        name = str(ev_names[i]).strip()
        if not name:
            continue

        def_load_cfg.append({"_source": "ev_auto", "name": name})
        optim_conf["nominal_power_of_deferrable_loads"].append(max(0.0, float(ev_max_3p[i])))
        optim_conf["minimum_power_of_deferrable_loads"].append(max(0.0, float(ev_min_1p[i])))
        optim_conf["operating_hours_of_each_deferrable_load"].append(0)
        optim_conf["start_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["end_timesteps_of_each_deferrable_load"].append(0)
        optim_conf["set_deferrable_startup_penalty"].append(0.0)
        optim_conf["set_deferrable_load_single_constant"].append(False)
        optim_conf["treat_deferrable_load_as_semi_cont"].append(True)
        optim_conf["load_type"].append("fixed_power_non_splittable")
        optim_conf["load_dispatch_mode"].append("hours")
        optim_conf["required_energy_kwh_of_each_deferrable_load"].append(0.0)
        optim_conf["load_phase"].append(str(ev_phase[i]).strip())

        slug_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"ev_{i + 1}"
        passed_data = params.setdefault("passed_data", {})
        passed_data.setdefault("custom_deferrable_forecast_id", []).append(
            {
                "entity_id": f"sensor.p_{slug_name}",
                "device_class": "power",
                "unit_of_measurement": "W",
                "friendly_name": f"{name} Power",
            }
        )
        passed_data.setdefault("custom_ev_charge_mode_target_id", []).append(
            {
                "entity_id": f"sensor.ev_charge_mode_target_{slug_name}",
                "device_class": "",
                "unit_of_measurement": "",
                "friendly_name": f"{name} Charge Mode Target",
            }
        )
        passed_data.setdefault("custom_ev_phase_target_id", []).append(
            {
                "entity_id": f"sensor.ev_phase_target_{slug_name}",
                "device_class": "",
                "unit_of_measurement": "",
                "friendly_name": f"{name} Phase Target",
            }
        )

        ev_load_indices[name] = num_def_loads
        num_def_loads += 1

    if not ev_load_indices:
        return

    optim_conf["def_load_config"] = def_load_cfg
    optim_conf["number_of_deferrable_loads"] = num_def_loads
    params.setdefault("passed_data", {})
    params["passed_data"]["ev_load_indices"] = ev_load_indices


async def _resolve_manual_committed_loads(params: dict, logger: logging.Logger) -> None:
    """Wire up manually-started appliances (washing machine, dishwasher - no
    smart-plug control, only a physical delay-start timer) onto their own
    existing deferrable load slot, flagged per-load via
    optim_conf["is_manual_load"] instead of living in a separate config
    section. The load keeps using its own load_names/nominal_power_of_deferrable_loads/
    operating_hours_of_each_deferrable_load - nothing is duplicated or
    appended here.

    Purely structural: this only forces the single-contiguous-block behaviour
    a pinned manual commitment needs (set_deferrable_load_single_constant,
    treat_deferrable_load_as_semi_cont). Whether a load actually needs to run
    this cycle - and, once a start time has been committed to and shown to
    the user, keeping that exact window pinned across re-optimizations -
    depends on live sensor data (the ready input_boolean) and persisted state
    (data/manual_load_commitments.json), neither of which is available yet
    at this build-time stage. That live handling is done once per solve in
    command_line._apply_manual_load_runtime_overrides, right before the
    optimization runs.

    Records params["passed_data"]["manual_load_indices"] (name -> dict with
    the deferrable load index plus the per-load config the runtime-override
    step and the publish step both need) so those later stages don't have to
    re-derive it from raw config.
    """
    optim_conf = params.get("optim_conf", {})
    if not optim_conf.get("manual_load_enabled", False):
        return

    is_manual = optim_conf.get("is_manual_load", []) or []
    num_loads = len(is_manual)
    if num_loads <= 0 or not any(is_manual):
        return

    load_names = optim_conf.get("load_names", []) or []
    nominal_power = optim_conf.get("nominal_power_of_deferrable_loads", []) or []
    operating_hours = optim_conf.get("operating_hours_of_each_deferrable_load", []) or []
    manual_deadline = optim_conf.get("manual_load_deadline_hour", []) or []
    single_constant = optim_conf.setdefault("set_deferrable_load_single_constant", [])
    semi_cont = optim_conf.setdefault("treat_deferrable_load_as_semi_cont", [])

    # These three are live HA sensor entity ids, so they're mapped to
    # retrieve_hass_conf (not optim_conf) in associations.csv, matching
    # heatpump_room_temp_sensors. Sized to the same per-deferrable-load
    # indexing as is_manual_load.
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    manual_ready_sensor = check_def_loads(
        num_loads, retrieve_hass_conf, "", "manual_load_ready_sensor", logger
    )
    manual_confirm_sensor = check_def_loads(
        num_loads, retrieve_hass_conf, "", "manual_load_confirm_power_sensor", logger
    )
    # Padded here (not used by this function's own output) purely so it's
    # normalized to the same per-load indexing before _resolve_load_profiles
    # reads it - only meaningful for manual loads, since only a human can
    # know in advance which program they're about to run.
    check_def_loads(num_loads, retrieve_hass_conf, "", "manual_load_program_select_sensor", logger)

    manual_load_indices: dict[str, dict] = {}
    for k in range(num_loads):
        if not is_manual[k]:
            continue

        name = str(load_names[k]).strip() if k < len(load_names) else ""
        if not name:
            name = f"appliance_{k + 1}"

        # nominal_power_of_deferrable_loads[k] may already be a resolved
        # sequence (program_based load_type, or a WashData profile resolved
        # later this same cycle - see command_line._resolve_load_profiles)
        # rather than a flat scalar; manual_load_indices["nominal_power"] is only
        # ever used as a rough confirm-power-sensor threshold downstream, so
        # a sequence's peak is a safe stand-in for its (non-existent) single
        # nominal value.
        raw_power = nominal_power[k] if k < len(nominal_power) else 0.0
        power_w = max(0.0, float(max(raw_power))) if isinstance(raw_power, list) else max(
            0.0, float(raw_power)
        )
        duration_h = max(0.0, float(operating_hours[k])) if k < len(operating_hours) else 0.0

        # A pinned manual commitment needs a single contiguous on/off block,
        # regardless of whatever this load was otherwise configured as.
        if k < len(single_constant):
            single_constant[k] = True
        if k < len(semi_cont):
            semi_cont[k] = True

        slug_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or f"manual_load_{k + 1}"
        passed_data = params.setdefault("passed_data", {})
        passed_data.setdefault("custom_manual_load_action_id", []).append(
            {
                "entity_id": f"sensor.manual_load_action_{slug_name}",
                "device_class": "",
                "unit_of_measurement": "",
                "friendly_name": f"{name} Action",
            }
        )

        manual_load_indices[name] = {
            "k": k,
            "ready_sensor": str(manual_ready_sensor[k] or ""),
            "confirm_power_sensor": str(manual_confirm_sensor[k] or ""),
            "nominal_power": power_w,
            "duration_hours": duration_h,
            "deadline_hour": str(manual_deadline[k] or "") if k < len(manual_deadline) else "",
        }

    if not manual_load_indices:
        return

    params.setdefault("passed_data", {})
    params["passed_data"]["manual_load_indices"] = manual_load_indices


def get_days_list(days_to_retrieve: int) -> pd.DatetimeIndex:
    """
    Get list of past days from today to days_to_retrieve.

    :param days_to_retrieve: Total number of days to retrieve from the past
    :type days_to_retrieve: int
    :return: The list of days
    :rtype: pd.DatetimeIndex

    """
    today = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    d = (today - timedelta(days=days_to_retrieve)).isoformat()
    days_list = pd.date_range(start=d, end=today.isoformat(), freq="D").normalize()
    return days_list


def add_date_features(
    data: pd.DataFrame,
    timestamp: str | None = None,
    date_features: list[str] | None = None,
) -> pd.DataFrame:
    """Add date-related features from a DateTimeIndex or a timestamp column.

    :param data: The input DataFrame.
    :type data: pd.DataFrame
    :param timestamp: The column containing the timestamp (optional if DataFrame has a DateTimeIndex).
    :type timestamp: Optional[str]
    :param date_features: List of date features to extract (default: all).
    :type date_features: Optional[List[str]]
    :return: The DataFrame with added date features.
    :rtype: pd.DataFrame
    """

    df = copy.deepcopy(data)  # Avoid modifying the original DataFrame

    # If no specific features are requested, extract all by default
    default_features = ["year", "month", "day_of_week", "day_of_year", "day", "hour"]
    date_features = date_features or default_features

    # Determine whether to use index or a timestamp column
    if timestamp:
        df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
        source = df[timestamp].dt
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTimeIndex or a valid timestamp column.")
        source = df.index

    # Extract date features
    if "year" in date_features:
        df["year"] = source.year
    if "month" in date_features:
        df["month"] = source.month
    if "day_of_week" in date_features:
        df["day_of_week"] = source.dayofweek
    if "day_of_year" in date_features:
        df["day_of_year"] = source.dayofyear
    if "day" in date_features:
        df["day"] = source.day
    if "hour" in date_features:
        df["hour"] = source.hour

    return df


def set_df_index_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set the freq of a DataFrame DateTimeIndex.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Input DataFrame with freq defined
    :rtype: pd.DataFrame

    """
    idx_diff = np.diff(df.index)
    # Sometimes there are zero values in this list.
    idx_diff = idx_diff[np.nonzero(idx_diff)]
    sampling = pd.to_timedelta(np.median(idx_diff))
    df = df[~df.index.duplicated()]
    return df.asfreq(sampling)


def parse_export_time_range(
    start_time: str,
    end_time: str | None,
    time_zone: pd.Timestamp.tz,
    logger: logging.Logger,
) -> tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]:
    """
    Parse and validate start_time and end_time for export operations.

    :param start_time: Start time string in ISO format
    :type start_time: str
    :param end_time: End time string in ISO format (optional)
    :type end_time: str | None
    :param time_zone: Timezone for localization
    :type time_zone: pd.Timestamp.tz
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Tuple of (start_dt, end_dt) or (False, False) on error
    :rtype: tuple[pd.Timestamp, pd.Timestamp] | tuple[bool, bool]
    """
    try:
        start_dt = pd.to_datetime(start_time)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(time_zone)
    except Exception as e:
        logger.error(f"Invalid start_time format: {start_time}. Error: {e}")
        logger.error("Use format like '2024-01-01' or '2024-01-01 00:00:00'")
        return False, False

    if end_time:
        try:
            end_dt = pd.to_datetime(end_time)
            if end_dt.tz is None:
                end_dt = end_dt.tz_localize(time_zone)
        except Exception as e:
            logger.error(f"Invalid end_time format: {end_time}. Error: {e}")
            return False, False
    else:
        end_dt = pd.Timestamp.now(tz=time_zone)
        logger.info(f"No end_time specified, using current time: {end_dt}")

    return start_dt, end_dt


def clean_sensor_column_names(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Clean sensor column names by removing 'sensor.' prefix.

    :param df: Input DataFrame with sensor columns
    :type df: pd.DataFrame
    :param timestamp_col: Name of timestamp column to preserve
    :type timestamp_col: str
    :return: DataFrame with cleaned column names
    :rtype: pd.DataFrame
    """
    column_mapping = {}
    for col in df.columns:
        if col != timestamp_col and col.startswith("sensor."):
            column_mapping[col] = col.replace("sensor.", "")
    return df.rename(columns=column_mapping)


def handle_nan_values(
    df: pd.DataFrame,
    handle_nan: str,
    timestamp_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Handle NaN values in DataFrame according to specified strategy.

    :param df: Input DataFrame
    :type df: pd.DataFrame
    :param handle_nan: Strategy for handling NaN values
    :type handle_nan: str
    :param timestamp_col: Name of timestamp column to exclude from processing
    :type timestamp_col: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: DataFrame with NaN values handled
    :rtype: pd.DataFrame
    """
    nan_count_before = df.isna().sum().sum()
    if nan_count_before == 0:
        return df

    logger.info(f"Found {nan_count_before} NaN values, applying handle_nan method: {handle_nan}")

    if handle_nan == "drop":
        df = df.dropna()
        logger.info(f"Dropped rows with NaN. Remaining rows: {len(df)}")
    elif handle_nan == "fill_zero":
        # Exclude timestamp_col from fillna to avoid unintended changes
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].fillna(0)
        logger.info("Filled NaN values with 0 (excluding timestamp)")
    elif handle_nan == "interpolate":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude timestamp_col from interpolation
        interp_cols = [col for col in numeric_cols if col != timestamp_col]
        df[interp_cols] = df[interp_cols].interpolate(method="linear", limit_direction="both")
        df[interp_cols] = df[interp_cols].ffill().bfill()
        logger.info("Interpolated NaN values (excluding timestamp)")
    elif handle_nan == "forward_fill":
        # Exclude timestamp_col from forward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].ffill()
        logger.info("Forward filled NaN values (excluding timestamp)")
    elif handle_nan == "backward_fill":
        # Exclude timestamp_col from backward fill
        fill_cols = [col for col in df.columns if col != timestamp_col]
        df[fill_cols] = df[fill_cols].bfill()
        logger.info("Backward filled NaN values (excluding timestamp)")
    elif handle_nan == "keep":
        logger.info("Keeping NaN values as-is")
    else:
        logger.warning(f"Unknown handle_nan option '{handle_nan}', keeping NaN values")

    return df


def resample_and_filter_data(
    df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    resample_freq: str,
    logger: logging.Logger,
) -> pd.DataFrame | bool:
    """
    Filter DataFrame to time range and resample to specified frequency.

    :param df: Input DataFrame with datetime index
    :type df: pd.DataFrame
    :param start_dt: Start datetime for filtering
    :type start_dt: pd.Timestamp
    :param end_dt: End datetime for filtering
    :type end_dt: pd.Timestamp
    :param resample_freq: Resampling frequency string (e.g., '1h', '30min')
    :type resample_freq: str
    :param logger: Logger object
    :type logger: logging.Logger
    :return: Resampled DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Validate that DataFrame index is datetime and properly localized
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}")
        return False

    # Check if timezone aware and matches expected timezone
    if df.index.tz is None:
        logger.warning("DataFrame index is timezone-naive, localizing to match start/end times")
        df = df.copy()
        df.index = df.index.tz_localize(start_dt.tz)
    elif df.index.tz != start_dt.tz:
        logger.warning(
            f"DataFrame timezone ({df.index.tz}) differs from filter timezone ({start_dt.tz}), converting"
        )
        df = df.copy()
        df.index = df.index.tz_convert(start_dt.tz)

    # Filter to exact time range
    df_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]

    if df_filtered.empty:
        logger.error("No data in the specified time range after filtering")
        return False

    logger.info(f"Retrieved {len(df_filtered)} data points")

    # Resample to specified frequency
    logger.info(f"Resampling data to frequency: {resample_freq}")
    try:
        df_resampled = df_filtered.resample(resample_freq).mean()
        df_resampled = df_resampled.dropna(how="all")

        if df_resampled.empty:
            logger.error("No data after resampling. Check frequency and data availability.")
            return False

        logger.info(f"After resampling: {len(df_resampled)} data points")
        return df_resampled

    except Exception as e:
        logger.error(f"Error during resampling: {e}")
        return False


@contextmanager
def stage_timer(stage_times: dict, name: str, logger: logging.Logger | None = None):
    """Record wall-clock elapsed time of a block in ``stage_times[name]``.

    Emits one DEBUG line per stage when ``logger`` is provided. Works across
    ``await`` because ``time.perf_counter()`` is a plain monotonic read.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        stage_times[name] = dt
        if logger is not None:
            logger.debug(f"Stage [{name}] completed in {dt:.3f}s")


def log_runtime_banner(logger, optim_conf: dict | None = None):
    """Log a single INFO line with EMHASS/Python/CVXPY/platform info for bug-report reproducibility.

    When ``optim_conf`` is provided, the configured solver (or the EMHASS
    default ``Highs`` when the key is absent) is shown — this matches the
    solver the LP actually uses (see ``optimization.py`` constructor).
    When ``optim_conf`` is missing entirely (early-fail paths), falls back
    to ``cvxpy.installed_solvers()[0]``.
    """
    try:
        import platform as _plat
        from importlib.metadata import version as _pkg_version

        import cvxpy as _cvx

        _ver = _pkg_version("emhass")
        if isinstance(optim_conf, dict):
            solver = str(optim_conf.get("lp_solver", "Highs"))
        else:
            solvers = _cvx.installed_solvers()
            solver = solvers[0] if solvers else "none"
        logger.info(
            f"EMHASS {_ver} | Python {_plat.python_version()} | "
            f"CVXPY {_cvx.__version__} ({solver}) | "
            f"{_plat.system()}-{_plat.machine()}"
        )
    except Exception as err:
        try:
            from importlib.metadata import version as _pkg_version

            _ver = _pkg_version("emhass")
            logger.info(f"EMHASS {_ver} (runtime info unavailable: {err})")
        except Exception:
            logger.info("EMHASS (runtime info unavailable)")
