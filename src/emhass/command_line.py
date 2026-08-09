#!/usr/bin/env python3

import argparse
import asyncio
import copy
import json
import logging
import os
import pathlib
import pickle
import threading
import time as _time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib.metadata import version
from math import ceil

try:
    from datetime import UTC
except ImportError:
    # Python 3.10 compatibility
    from datetime import timezone

    UTC = timezone.utc

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import last_run, plan_store, utils
from emhass.battery_identification import BatteryIdentification
from emhass.forecast import Forecast
from emhass.forecast_calibration import (
    CALIBRATION_DEFAULT_DAYS,
    CALIBRATION_TEST_DAYS,
    CALIBRATION_VAL_DAYS,
    compute_forecast_calibration,
)
from emhass.machine_learning_forecaster import MLForecaster
from emhass.machine_learning_regressor import MLRegressor
from emhass.optimization import Optimization
from emhass.persistence import load_json_blob, save_json_blob
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import log_runtime_banner, stage_timer

default_csv_filename = "opt_res_latest.csv"
default_pkl_suffix = "_mlf.pkl"
default_metadata_json = "metadata.json"
test_df_literal = "test_df_final.pkl"
EMHASS_SCHEMA_VERSION = "1.0"


def _record_optim_snapshot(
    input_data_dict: dict,
    action: str,
    opt_res,
    t0_monotonic: float,
    logger: logging.Logger,
) -> None:
    """Persist a last_run snapshot after an optim wrapper completes.

    Best-effort: any failure is logged with a traceback but does not
    propagate so the wrapper's return path stays intact.
    """
    try:
        optim_status = (
            opt_res["optim_status"].iloc[0]
            if isinstance(opt_res, pd.DataFrame) and "optim_status" in opt_res
            else "Unknown"
        )
        ts = last_run.record(
            input_data_dict["emhass_conf"]["data_path"],
            action=action,
            stage_times=input_data_dict["stage_times"],
            optim_status=optim_status,
            infeasible=(optim_status == "Infeasible"),
            duration_total_seconds=_time.monotonic() - t0_monotonic,
            schema_version=EMHASS_SCHEMA_VERSION,
        )
        # Publish the structured plan ONLY for a successful (Optimal) run, reusing
        # the SAME timestamp last_run stamped so /api/v1/plan's generated_at matches
        # /api/v1/last-run for that run. Only the timestamp is shared, not the
        # verdict: a failed/infeasible run is still recorded by last_run (status
        # error/infeasible) but must not surface on /api/v1/plan as status "ok" —
        # the plan endpoint keeps serving the last VALID plan (or no-run). Gating on
        # optim_status == "Optimal" mirrors last_run's own "ok" criterion, so the
        # two endpoints stay consistent (plan published iff last-run is "ok").
        if optim_status == "Optimal":
            plan_store.record(
                input_data_dict["emhass_conf"]["data_path"],
                plan=plan_store.serialize(opt_res),
                generated_at=ts,
                schema_version=EMHASS_SCHEMA_VERSION,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("last_run: failed to record %s snapshot", action, exc_info=exc)


@dataclass
class SetupContext:
    """
    A dataclass that serves as a context container for optimization preparation helpers.
    This context object encapsulates all necessary configuration and utility objects
    required for setting up and preparing optimization tasks.

    Attributes:
        retrieve_hass_conf (dict): Configuration dictionary for Home Assistant data retrieval.
        optim_conf (dict): Configuration dictionary for optimization parameters.
        plant_conf (dict): Configuration dictionary for plant/system parameters.
        emhass_conf (dict): Configuration dictionary for EMHASS settings.
        params (dict): Additional parameters dictionary.
        logger (logging.Logger): Logger instance for logging messages.
        get_data_from_file (bool): Flag indicating whether to retrieve data from file instead of live source.
        rh (RetrieveHass): RetrieveHass instance for retrieving Home Assistant data.
        fcst (Forecast | None): Optional Forecast object for weather or energy forecasting. Defaults to None.
    """

    retrieve_hass_conf: dict
    optim_conf: dict
    plant_conf: dict
    emhass_conf: dict
    params: dict
    logger: logging.Logger
    get_data_from_file: bool
    rh: RetrieveHass
    fcst: Forecast | None = None


@dataclass
class PublishContext:
    """
    Context object for data publishing helpers.

    Attributes:
        input_data_dict (dict): Dictionary containing input data with keys 'rh' (RetrieveHass),
            'opt' (Optimization - may be None for publish-data), and 'fcst' (Forecast) objects.
        params (dict): Parameters dictionary for publishing configuration.
        idx (int): Index identifier for the current publishing operation.
        common_kwargs (dict): Common keyword arguments shared across publishing helpers.
        logger (logging.Logger): Logger instance for recording publishing operations.
    """

    input_data_dict: dict
    params: dict
    idx: int
    common_kwargs: dict
    logger: logging.Logger

    @property
    def rh(self) -> RetrieveHass:
        return self.input_data_dict["rh"]

    @property
    def opt(self) -> Optimization:
        return self.input_data_dict["opt"]

    @property
    def optim_conf(self) -> dict:
        """Access optim_conf directly from input_data_dict (works even when opt is None)."""
        return self.input_data_dict["optim_conf"]

    @property
    def plant_conf(self) -> dict:
        """Access plant_conf directly from input_data_dict (works even when opt is None)."""
        return self.input_data_dict["plant_conf"]

    @property
    def fcst(self) -> Forecast:
        return self.input_data_dict["fcst"]

    @property
    def emhass_conf(self) -> dict:
        return self.input_data_dict["emhass_conf"]


@dataclass(frozen=True)
class OptimizationCacheKey:
    """
    Frozen dataclass representing configuration fields that affect optimization structure.

    Changes to any of these fields require rebuilding the optimization problem.
    Using a frozen dataclass makes the cache key explicit, hashable, and easy to extend.
    """

    number_of_deferrable_loads: int
    set_use_battery: bool
    set_use_pv: bool
    treat_deferrable_load_as_semi_cont: tuple
    set_deferrable_load_single_constant: tuple
    set_deferrable_startup_penalty: tuple
    deferrable_load_max_cost: tuple
    set_deferrable_max_startups: tuple
    set_deferrable_load_as_timeseries: tuple
    nominal_power_of_deferrable_loads: tuple
    def_load_config_structure: tuple  # (index, type) tuples for each load
    deferrable_load_groups: tuple
    shared_thermal_tanks: tuple  # shared-tank multi-source topology structure
    is_electric_load: tuple  # per-load electric-bus membership flag
    inverter_is_hybrid: bool
    compute_curtailment: bool
    optimization_time_step_s: float | None
    delta_forecast_daily_s: float | None
    num_timesteps: int | None
    costfun: str
    plant_conf_hash: str
    optim_conf_structural_hash: str  # Hash of optim_conf keys that affect problem structure


class OptimizationCache:
    """
    In-memory cache for Optimization objects to enable warm-starting.

    Warm-starting reuses the previous solution as a starting point for the solver,
    which can significantly speed up repeated MPC optimizations where consecutive
    problems are similar.

    The cache is invalidated when configuration changes that affect the optimization
    structure (number of variables, constraints, etc.).

    Thread-safe: Uses a lock to prevent race conditions when multiple optimizations
    run concurrently in async code.
    """

    _instance: "Optimization | None" = None
    _cache_key: OptimizationCacheKey | None = None
    _last_used: datetime | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def _compute_cache_key(
        cls,
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        num_timesteps: int | None = None,
    ) -> OptimizationCacheKey:
        """
        Compute a cache key from configuration that affects optimization structure.

        Returns a frozen dataclass that can be directly compared for equality.
        Changes to any field require rebuilding the optimization problem.
        """

        def to_seconds(val):
            """Convert Timedelta/timedelta to seconds."""
            if val is None:
                return None
            return val.total_seconds() if hasattr(val, "total_seconds") else float(val)

        def to_tuple(val):
            """Convert lists/arrays to tuples for hashability."""
            if val is None:
                return ()
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, list | tuple):
                # Handle nested lists (e.g., nominal_power with sequences)
                return tuple(tuple(v) if isinstance(v, list | tuple) else v for v in val)
            return (val,)

        def config_hash(cfg: dict, exclude_keys: set | None = None) -> str:
            """Create a stable hash of config dict for cache key comparison.

            Args:
                cfg: The config dict to hash
                exclude_keys: Keys to exclude from hash (runtime params)
            """
            import hashlib

            if exclude_keys is None:
                exclude_keys = set()
            # Sort keys for deterministic ordering, exclude runtime parameters
            sorted_items = sorted(
                ((k, v) for k, v in cfg.items() if k not in exclude_keys),
                key=lambda x: str(x[0]),
            )
            config_str = str(sorted_items)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Runtime parameters that should NOT affect cache key
        # These change between MPC iterations but don't affect problem structure
        thermal_runtime_keys = {
            "start_temperature",
            "desired_temperatures",
            "indoor_target_temperature",  # thermal_battery runtime param
            "q_input_initial",  # thermal inertia warm-start override
            "draw_off_demand",  # hot water tank daily profile (updates heating_demand param)
        }
        # Plant parameters that are updated dynamically (no rebuild needed)
        plant_runtime_keys = {
            "soc_init",
            "battery_target_state_of_charge",
            "battery_charge_power_max",
            "battery_discharge_power_max",
        }
        # Optim conf parameters that don't affect problem structure
        # (parameterized via CVXPY Parameters, solver options, or forecast method selection)
        optim_conf_runtime_keys = {
            # Parameterized via CVXPY Parameters
            "operating_hours_of_each_deferrable_load",
            "operating_timesteps_of_each_deferrable_load",
            "start_timesteps_of_each_deferrable_load",
            "end_timesteps_of_each_deferrable_load",
            "required_energy_kwh_of_each_deferrable_load",
            "load_dispatch_mode",
            "def_current_state",
            # Per-call elapsed on-time for min-on remainder (issue #952); value
            # is read via cp.Parameter so no rebuild on cache hit.
            "def_current_on_timesteps",
            # Per-call elapsed off-time for min-off remainder (#952 follow-on); value
            # is read via cp.Parameter so no rebuild on cache hit.
            "def_current_off_timesteps",
            # Per-call current power in watts (issue #605); pin value is a cp.Parameter.
            "def_current_power",
            # Per-call completed operating timesteps today (issue #983); decrements
            # required_timesteps + target_energy via cp.Parameter (no rebuild on cache hit).
            "def_current_operating_timesteps",
            "minimum_power_of_deferrable_loads",
            "cost_forecast_per_deferrable_load",
            # shared_thermal_tanks has its own structural hash field above
            "shared_thermal_tanks",
            # heat_topology is compiled down to the structural fields which ARE
            # part of the cache key (def_load_config_structure,
            # deferrable_load_groups, shared_thermal_tanks). The raw
            # heat_topology itself is excluded to avoid double-counting.
            "heat_topology",
            # Solver options (updated on cache hit)
            "lp_solver_timeout",
            "lp_solver_mip_rel_gap",
            "num_threads",
            "lp_solver",
            # Forecast method selection (not optimization structure)
            "weather_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
            "load_forecast_method",
            # Already handled by explicit fields or separate hash
            "def_load_config",
            "delta_forecast_daily",
        }
        # Extract def_load_config structure (which loads are thermal/thermal_battery/standard)
        # Include hash of thermal config contents to detect parameter changes
        def_load_config = optim_conf.get("def_load_config", []) or []
        def_structure = []
        for i, cfg in enumerate(def_load_config):
            cfg = cfg or {}
            if "thermal_config" in cfg:
                # Include hash of thermal_config contents to detect parameter changes
                # Exclude runtime parameters (start_temperature, desired_temperatures) from hash
                thermal_hash = config_hash(cfg["thermal_config"], thermal_runtime_keys)
                load_type = f"thermal_config:{thermal_hash}"
            elif "thermal_battery" in cfg:
                # Include hash of thermal_battery contents to detect parameter changes
                # Exclude runtime parameters (start_temperature, desired_temperatures) from hash
                thermal_hash = config_hash(cfg["thermal_battery"], thermal_runtime_keys)
                load_type = f"thermal_battery:{thermal_hash}"
            else:
                load_type = "standard"
            def_structure.append((i, load_type))

        return OptimizationCacheKey(
            number_of_deferrable_loads=optim_conf.get("number_of_deferrable_loads", 0),
            set_use_battery=optim_conf.get("set_use_battery", False),
            set_use_pv=optim_conf.get("set_use_pv", True),
            treat_deferrable_load_as_semi_cont=to_tuple(
                optim_conf.get("treat_deferrable_load_as_semi_cont", [])
            ),
            set_deferrable_load_single_constant=to_tuple(
                optim_conf.get("set_deferrable_load_single_constant", [])
            ),
            set_deferrable_startup_penalty=to_tuple(
                optim_conf.get("set_deferrable_startup_penalty", [])
            ),
            deferrable_load_max_cost=to_tuple(optim_conf.get("deferrable_load_max_cost", [])),
            set_deferrable_max_startups=to_tuple(optim_conf.get("set_deferrable_max_startups", [])),
            set_deferrable_load_as_timeseries=to_tuple(
                optim_conf.get("set_deferrable_load_as_timeseries", [])
            ),
            # Note: The following are parameterized and don't require rebuild:
            # - start_timesteps and end_timesteps (via window masks)
            # - operating_hours_of_each_deferrable_load (via Big-M energy constraints)
            nominal_power_of_deferrable_loads=to_tuple(
                optim_conf.get("nominal_power_of_deferrable_loads", [])
            ),
            def_load_config_structure=tuple(def_structure),
            deferrable_load_groups=tuple(
                (tuple(g.get("names", [])), g.get("max_power"), g.get("mutual_exclusion", False))
                for g in optim_conf.get("deferrable_load_groups", [])
            ),
            # shared_thermal_tanks change problem structure (new tank state
            # variable + dynamics constraints), so include a structure hash.
            shared_thermal_tanks=tuple(
                (
                    t.get("id", ""),
                    tuple(int(k) for k in t.get("load_ids", [])),
                    config_hash(
                        {
                            k: v
                            for k, v in t.items()
                            if k not in {"start_temperature", "draw_off_demand"}
                        },
                    ),
                )
                for t in optim_conf.get("shared_thermal_tanks", []) or []
            ),
            # is_electric_load changes p_def_sum membership, hence the electric
            # power balance shape, hence structural.
            is_electric_load=to_tuple(optim_conf.get("is_electric_load", [])),
            inverter_is_hybrid=plant_conf.get("inverter_is_hybrid", False),
            compute_curtailment=plant_conf.get("compute_curtailment", False),
            optimization_time_step_s=to_seconds(retrieve_hass_conf.get("optimization_time_step")),
            delta_forecast_daily_s=to_seconds(optim_conf.get("delta_forecast_daily")),
            num_timesteps=num_timesteps,
            costfun=costfun,
            plant_conf_hash=config_hash(plant_conf, plant_runtime_keys),
            optim_conf_structural_hash=config_hash(optim_conf, optim_conf_runtime_keys),
        )

    @classmethod
    def get(
        cls,
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        logger: logging.Logger,
        num_timesteps: int | None = None,
    ) -> "Optimization | None":
        """
        Get cached Optimization object if configuration matches.

        Returns None if cache is empty or configuration has changed.
        Thread-safe via internal locking.
        """
        cache_key = cls._compute_cache_key(
            optim_conf, plant_conf, costfun, retrieve_hass_conf, num_timesteps
        )

        with cls._lock:
            if cls._instance is not None and cls._cache_key == cache_key:
                age = datetime.now() - cls._last_used if cls._last_used else timedelta(0)
                logger.debug(
                    f"OptimizationCache HIT: Reusing cached optimization object "
                    f"(age={age.total_seconds():.1f}s) - warm-start enabled"
                )
                cls._last_used = datetime.now()
                return cls._instance

            if cls._instance is not None:
                # Log which fields changed for debugging
                if cls._cache_key is not None:
                    changed_fields = []
                    for field in cache_key.__dataclass_fields__:
                        old_val = getattr(cls._cache_key, field)
                        new_val = getattr(cache_key, field)
                        if old_val != new_val:
                            changed_fields.append(f"{field}: {old_val!r} -> {new_val!r}")
                    if changed_fields:
                        logger.debug(
                            f"OptimizationCache MISS: Config changed - {', '.join(changed_fields)}"
                        )
                    else:
                        logger.debug("OptimizationCache MISS: Config changed (unknown diff)")
                else:
                    logger.debug("OptimizationCache MISS: Config changed, rebuilding optimization")
            else:
                logger.debug("OptimizationCache MISS: Empty cache, building new optimization")

            return None

    @classmethod
    def put(
        cls,
        opt: "Optimization",
        optim_conf: dict,
        plant_conf: dict,
        costfun: str,
        retrieve_hass_conf: dict,
        logger: logging.Logger,
        num_timesteps: int | None = None,
    ) -> None:
        """Store Optimization object in cache. Thread-safe via internal locking."""
        cache_key = cls._compute_cache_key(
            optim_conf, plant_conf, costfun, retrieve_hass_conf, num_timesteps
        )
        with cls._lock:
            cls._instance = opt
            cls._cache_key = cache_key
            cls._last_used = datetime.now()
            logger.debug(
                f"OptimizationCache: Stored optimization object "
                f"(loads={cache_key.number_of_deferrable_loads}, battery={cache_key.set_use_battery})"
            )

    @classmethod
    def clear(cls, logger: logging.Logger | None = None) -> None:
        """Clear the cache (e.g., for testing or explicit invalidation). Thread-safe."""
        with cls._lock:
            cls._instance = None
            cls._cache_key = None
            cls._last_used = None
            if logger:
                logger.debug("OptimizationCache: CLEARED")

    @classmethod
    def get_stats(cls) -> dict:
        """Get cache statistics for debugging. Thread-safe."""
        with cls._lock:
            return {
                "has_instance": cls._instance is not None,
                "cache_key": cls._cache_key,
                "last_used": cls._last_used.isoformat() if cls._last_used else None,
            }


async def _retrieve_from_file(
    emhass_conf: dict,
    test_df_literal: str,
    rh: RetrieveHass,
    retrieve_hass_conf: dict,
    optim_conf: dict,
) -> tuple[bool, object]:
    """Helper to retrieve data from a pickle file and configure variables."""
    async with aiofiles.open(emhass_conf["data_path"] / test_df_literal, "rb") as inp:
        content = await inp.read()
        rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
        rh.var_list = var_list
    # Assign variables based on set_type
    retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(var_list[0])
    if optim_conf.get("set_use_pv", True):
        retrieve_hass_conf["sensor_power_photovoltaics"] = str(var_list[1])
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
        ]
        retrieve_hass_conf["sensor_replace_zero"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            var_list[2],
        ]
    else:
        retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_load_no_var_loads"]
        ]
        retrieve_hass_conf["sensor_replace_zero"] = []
    return True, days_list


def _append_entity_ids(var_list: list, value) -> None:
    """
    Append a battery_id sensor config value to ``var_list``.

    A list (N>1, already resolved and duplicate-checked by
    :func:`_resolve_battery_sensor_lists`) is appended element-wise, deduped
    against ``var_list``. The resolver only rejects a within-list or
    cross-list duplicate, not a battery sensor equal to the load sensor
    already at ``var_list[0]``, so this guard is genuinely reachable there,
    not redundant with it. A bare string (N=1) is appended unconditionally.
    """
    if isinstance(value, list):
        for entity_id in value:
            if entity_id not in var_list:
                var_list.append(entity_id)
    else:
        var_list.append(value)


def _append_battery_id_sensors(var_list: list, retrieve_hass_conf: dict) -> None:
    """Append the battery power and SoC sensor config values to ``var_list``."""
    _append_entity_ids(var_list, retrieve_hass_conf["sensor_power_battery"])
    _append_entity_ids(var_list, retrieve_hass_conf["sensor_battery_state_of_charge"])


async def _retrieve_from_hass(
    set_type: str,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    logger: logging.Logger | None,
) -> tuple[bool, object]:
    """Helper to retrieve live data from Home Assistant."""
    # Determine days_list based on set_type
    if set_type in ("perfect-optim", "adjust_pv", "battery_id"):
        days_list = utils.get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
    elif set_type == "naive-mpc-optim":
        days_list = utils.get_days_list(1)
    else:
        days_list = None  # Not needed for dayahead
    var_list = [retrieve_hass_conf["sensor_power_load_no_var_loads"]]
    if set_type == "battery_id":
        # Battery identification needs signed battery power and measured SoC,
        # one pair per battery. The load sensor stays at var_list[0] so
        # prepare_data's load handling is unchanged; the battery columns are
        # passed to prepare_data as protected_columns so its set_zero_min clip
        # cannot destroy the discharge direction or a measured 0% SoC (#1041).
        # A list value (N>1, already resolved by _identify_battery_impl) is
        # appended per-id, deduped against var_list; a bare string (N=1) is
        # the plain single-sensor case.
        _append_battery_id_sensors(var_list, retrieve_hass_conf)
        if logger:
            logger.debug(f"Variable list for battery_id retrieval: {var_list}")
    elif optim_conf.get("set_use_pv", True):
        var_list.append(retrieve_hass_conf["sensor_power_photovoltaics"])
        if optim_conf.get("set_use_adjusted_pv", True):
            var_list.append(retrieve_hass_conf["sensor_power_photovoltaics_forecast"])
    if optim_conf.get("set_use_heatpump", False):
        # Live room / heat-pump temperature sensors, used to override each
        # thermal load's starting temperature (see _build_def_init_temp)
        # instead of always starting from the static config value.
        for entity_id in retrieve_hass_conf.get("heatpump_room_temp_sensors", []) or []:
            if entity_id and entity_id not in var_list:
                var_list.append(entity_id)
        indoor_sensor = retrieve_hass_conf.get("heatpump_indoor_temp_sensor", "")
        if indoor_sensor and indoor_sensor not in var_list:
            var_list.append(indoor_sensor)
    # manual_load_ready_sensor / manual_load_confirm_power_sensor are
    # deliberately NOT added to var_list here: they're a "what is this right
    # now" lookup, not historical data, so they're read via a direct
    # RetrieveHass.get_current_state() REST call in
    # _apply_manual_load_runtime_overrides instead - see that function's
    # docstring for why (routing through use_influxdb depends on the user's
    # InfluxDB integration recording that entity's domain, which many
    # setups don't for input_boolean/switch helpers).
    if logger:
        logger.debug(f"Variable list for data retrieval: {var_list}")
    success = await rh.get_data(
        days_list, var_list, minimal_response=False, significant_changes_only=False
    )
    return success, days_list


def _build_def_init_temp(input_data_dict: dict, logger: logging.Logger) -> list | None:
    """Build the per-load def_init_temp override list from live HA sensor data
    for rooms and the whole-house heat pump dispatch load created by
    _append_room_thermal_loads. Returns None if heat pump loads aren't in use,
    otherwise a list of length number_of_deferrable_loads with None everywhere
    except room/dispatch indices, where it holds the latest real sensor value.
    """
    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    passed_data = params.get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    dispatch_load_index = passed_data.get("heatpump_dispatch_load_index")
    if not room_load_indices and dispatch_load_index is None:
        return None

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    def_init_temp: list = [None] * num_def_loads
    rh = input_data_dict["rh"]
    df_final = getattr(rh, "df_final", None)
    if df_final is None:
        return def_init_temp

    def _latest_sensor_value(entity_id: str) -> float | None:
        if not entity_id or entity_id not in df_final.columns:
            return None
        series = df_final[entity_id].dropna()
        if series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except (TypeError, ValueError):
            return None

    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_sensors = retrieve_hass_conf.get("heatpump_room_temp_sensors", []) or []
    for name, k in room_load_indices.items():
        if k >= len(def_init_temp):
            continue
        if name not in room_names:
            continue
        i = room_names.index(name)
        entity_id = room_sensors[i] if i < len(room_sensors) else None
        value = _latest_sensor_value(entity_id)
        if value is not None:
            def_init_temp[k] = value
        else:
            logger.debug(f"No live temperature sensor value found for room '{name}'")

    if dispatch_load_index is not None and dispatch_load_index < len(def_init_temp):
        indoor_sensor = retrieve_hass_conf.get("heatpump_indoor_temp_sensor", "")
        value = _latest_sensor_value(indoor_sensor)
        if value is not None:
            def_init_temp[dispatch_load_index] = value
        else:
            logger.debug("No live indoor temperature sensor value found for heat pump dispatch")

    return def_init_temp


def _timestep_index_from_timestamp(
    ts: pd.Timestamp, horizon_start: pd.Timestamp, time_step: pd.Timedelta
) -> int:
    """Convert an absolute timestamp into a timestep index relative to
    horizon_start (the first timestamp of this solve's forecast horizon),
    clamped at 0. Used to re-express a persisted manual-load commitment
    (an absolute committed_start_iso) in the relative indexing that
    start_timesteps_of_each_deferrable_load/end_timesteps_of_each_deferrable_load
    expect - necessary on every re-solve since the horizon rolls forward.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    hs = horizon_start
    if hs.tzinfo is None:
        hs = hs.tz_localize(UTC)
    else:
        hs = hs.tz_convert(UTC)
    return max(0, int(round((ts - hs) / time_step)))


def _next_deadline_timestamp(deadline_hour: str, horizon_start: pd.Timestamp) -> pd.Timestamp | None:
    """Resolve a "HH:MM" deadline into the next absolute occurrence at/after
    horizon_start (today's date at that time, or tomorrow's if today's has
    already passed). Returns None for an unset/unparseable deadline.
    """
    try:
        hour_str, minute_str = str(deadline_hour).split(":")[:2]
        hour, minute = int(hour_str), int(minute_str)
    except (ValueError, AttributeError):
        return None
    candidate = horizon_start.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    if candidate <= horizon_start:
        candidate += pd.Timedelta(days=1)
    return candidate


async def _apply_manual_load_runtime_overrides(input_data_dict: dict, logger: logging.Logger) -> None:
    """Live per-cycle handling for manually-committed loads (washer/dishwasher
    with only a physical delay-start timer, see manual_load_enabled): reads
    each load's "ready" input_boolean and optional confirmation power sensor
    via a direct RetrieveHass.get_current_state() REST call (deliberately not
    routed through use_influxdb - a current-value lookup is the wrong fit for
    a historical-data backend, and depends on the user's InfluxDB integration
    actually recording that entity's domain, which many setups don't for
    input_boolean/switch helpers), manages the persisted commitment
    (data/manual_load_commitments.json) lifecycle, and mutates optim_conf's
    runtime-only deferrable-load keys (operating_hours_of_each_deferrable_load /
    start_timesteps_of_each_deferrable_load / end_timesteps_of_each_deferrable_load
    - all cheap CVXPY Parameter updates, see optim_conf_runtime_keys in
    set_input_data_dict) in place.

    A load with an existing future commitment is pinned to the exact window
    already shown to the user - a re-optimization must never move it. A load
    that's ready but not yet committed gets a flexible window (optionally
    bounded by manual_load_deadline_hour) so the solver can find one; the
    actual chosen start is persisted as a new commitment after the solve (see
    _maybe_record_manual_load_commitments in publish_data).
    """
    params = input_data_dict["params"]
    optim_conf = params.get("optim_conf", {})
    passed_data = params.get("passed_data", {})
    manual_load_indices = passed_data.get("manual_load_indices", {})
    if not manual_load_indices:
        return

    fcst = input_data_dict.get("fcst")
    forecast_dates = getattr(fcst, "forecast_dates", None) if fcst is not None else None
    if forecast_dates is None or len(forecast_dates) == 0:
        return
    horizon_start = forecast_dates[0]
    horizon_len = len(forecast_dates)

    time_step = params.get("retrieve_hass_conf", {}).get(
        "optimization_time_step", pd.to_timedelta(30, "min")
    )
    if isinstance(time_step, (int, float)):
        time_step = pd.to_timedelta(time_step, "minutes")
    step_hours = time_step / pd.Timedelta(hours=1)
    if step_hours <= 0:
        return

    rh = input_data_dict["rh"]

    emhass_conf = input_data_dict["emhass_conf"]
    commitments = await load_json_blob(
        emhass_conf, "manual_load_commitments.json", logger, default={}
    )
    if not isinstance(commitments, dict):
        commitments = {}
    commitments_changed = False

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    op_hours = optim_conf.setdefault(
        "operating_hours_of_each_deferrable_load", [0] * num_def_loads
    )
    start_ts_list = optim_conf.setdefault(
        "start_timesteps_of_each_deferrable_load", [0] * num_def_loads
    )
    end_ts_list = optim_conf.setdefault(
        "end_timesteps_of_each_deferrable_load", [0] * num_def_loads
    )

    now = pd.Timestamp.now(tz=UTC)
    for name, load_info in manual_load_indices.items():
        k = load_info["k"]
        if k >= len(op_hours):
            continue

        commitment = commitments.get(name)
        committed_start = None
        if isinstance(commitment, dict) and commitment.get("committed_start_iso"):
            try:
                committed_start = pd.Timestamp(commitment["committed_start_iso"])
                committed_start = (
                    committed_start.tz_localize(UTC)
                    if committed_start.tzinfo is None
                    else committed_start.tz_convert(UTC)
                )
            except (ValueError, TypeError):
                committed_start = None

        # --- Clear a commitment once it's actually satisfied ---
        if committed_start is not None:
            duration_h = float(load_info.get("duration_hours", 0.0) or 0.0)
            confirm_sensor = load_info.get("confirm_power_sensor", "")
            cleared = False
            if confirm_sensor:
                confirm_value = await rh.get_current_state(confirm_sensor)
                nominal_power = float(load_info.get("nominal_power", 0.0) or 0.0)
                threshold = max(0.1 * nominal_power, 20.0)
                if confirm_value is not None and confirm_value >= threshold:
                    cleared = True
                    logger.info(
                        "Manual load '%s' confirmed running via %s, clearing commitment",
                        name,
                        confirm_sensor,
                    )
            else:
                # No confirmation sensor available: fall back to a best-effort
                # clear once the committed window plus its duration has fully
                # elapsed (a small grace period absorbs clock/round-trip skew).
                elapsed_deadline = committed_start + pd.Timedelta(hours=duration_h) + pd.Timedelta(minutes=15)
                if now >= elapsed_deadline:
                    cleared = True
                    logger.info(
                        "Manual load '%s' commitment window elapsed with no confirmation sensor, "
                        "clearing (best-effort)",
                        name,
                    )
            if cleared:
                del commitments[name]
                commitments_changed = True
                committed_start = None

        # --- Apply this cycle's runtime overrides ---
        ready = await rh.get_current_state(load_info.get("ready_sensor", "")) == 1.0
        commitment_idx = None
        if committed_start is not None:
            commitment_idx = _timestep_index_from_timestamp(committed_start, horizon_start, time_step)
            if commitment_idx >= horizon_len:
                commitment_idx = None  # commitment is beyond this solve's horizon

        if not ready and commitment_idx is None:
            op_hours[k] = 0
            start_ts_list[k] = 0
            end_ts_list[k] = 0
            continue

        nominal_power_field = optim_conf.get("nominal_power_of_deferrable_loads", [])
        is_sequence = k < len(nominal_power_field) and isinstance(nominal_power_field[k], list)
        if is_sequence:
            # A learned power profile (e.g. WashData) was resolved for this
            # load this cycle - see _resolve_manual_load_profiles, which runs
            # before this function and already mutated
            # optim_conf["nominal_power_of_deferrable_loads"][k] into a list.
            # The exact-pin mechanism requires end - start == sequence_length
            # exactly for only one candidate start offset to stay feasible,
            # so duration_steps must come from the resolved sequence's own
            # length, not the load's flat operating-hours fallback below.
            sequence_length = len(nominal_power_field[k])
            duration_steps = max(1, sequence_length)
            op_hours[k] = sequence_length  # a step count, not hours - matches the
            # convention _normalize_deferrable_load_categories already uses
            # for program_based loads.
        else:
            duration_h = float(load_info.get("duration_hours", 0.0) or 0.0)
            op_hours[k] = duration_h
            duration_steps = max(1, ceil(duration_h / step_hours))

        if commitment_idx is not None:
            start_ts_list[k] = commitment_idx
            end_ts_list[k] = commitment_idx + duration_steps
        else:
            start_ts_list[k] = 0
            deadline_hour = load_info.get("deadline_hour", "")
            deadline_ts = _next_deadline_timestamp(deadline_hour, horizon_start) if deadline_hour else None
            end_ts_list[k] = (
                _timestep_index_from_timestamp(deadline_ts, horizon_start, time_step)
                if deadline_ts is not None
                else 0
            )

    optim_conf["operating_hours_of_each_deferrable_load"] = op_hours
    optim_conf["start_timesteps_of_each_deferrable_load"] = start_ts_list
    optim_conf["end_timesteps_of_each_deferrable_load"] = end_ts_list

    if commitments_changed:
        await save_json_blob(
            emhass_conf, "manual_load_commitments.json", commitments, logger
        )


async def _resolve_manual_load_profiles(
    rh: RetrieveHass,
    optim_conf: dict,
    params_optim_conf: dict,
    retrieve_hass_conf: dict,
    params: dict,
    logger: logging.Logger,
) -> None:
    """Per-cycle learned power-profile resolution for manually-committed
    loads (see manual_load_enabled / manual_load_profile_sensor, e.g. the
    WashData ha_washdata integration's per-program profile sensors). Runs
    inside set_input_data_dict, before Forecast/OptimizationCache/Optimization
    are built - unlike every other manual_load_* field, this is NOT frozen
    at config-save time: it's read fresh on every action call so a profile
    that WashData refines over more cycles is picked up automatically.

    For each manual load with a configured profile_sensor, fetches that
    entity's attributes fresh via RetrieveHass.get_entity_state_and_attributes
    (a direct REST call, deliberately bypassing InfluxDB - see that method's
    docstring), and - only on a fully valid read - swaps that load's flat
    nominal_power_of_deferrable_loads[k] scalar for the profile's resampled
    Watt sequence, mirroring the pre-existing load_type == "program_based"
    mechanism in _normalize_deferrable_load_categories.

    Mutates BOTH optim_conf (the object about to be used to build/cache the
    Optimization instance - see set_input_data_dict, where this and
    params["optim_conf"] are distinct dict objects by this point) and
    params_optim_conf (params["optim_conf"]) with the same values, so the
    resolved sequence is visible both to the solver's cache key/constraints
    and to _apply_manual_load_runtime_overrides / naive_mpc_optim /
    dayahead_forecast_optim, which all read params["optim_conf"].

    Any failure (missing/unavailable entity, no power_profile attribute,
    invalid power_profile_interval_min) is caught and logged; that load's
    existing flat scalar values (already set by _resolve_manual_committed_loads)
    are left untouched, so it gracefully falls back to the flat model.
    """
    manual_load_indices = params.get("passed_data", {}).get("manual_load_indices", {})
    if not manual_load_indices:
        return

    time_step = retrieve_hass_conf.get("optimization_time_step")
    if isinstance(time_step, (int, float)):
        time_step = pd.to_timedelta(time_step, "minutes")
    if not isinstance(time_step, pd.Timedelta) or time_step <= pd.Timedelta(0):
        return
    target_step_min = time_step / pd.Timedelta(minutes=1)

    for name, load_info in manual_load_indices.items():
        profile_sensor = str(load_info.get("profile_sensor", "") or "").strip()
        if not profile_sensor:
            continue
        k = load_info["k"]
        try:
            payload = await rh.get_entity_state_and_attributes(profile_sensor)
            if not payload:
                logger.debug(
                    "Manual load '%s': profile sensor %s unavailable, "
                    "falling back to flat nominal_power/duration_hours",
                    name,
                    profile_sensor,
                )
                continue
            attributes = payload.get("attributes") or {}
            sequence = utils._parse_profile_to_float_list(attributes.get("power_profile"))
            if not sequence:
                logger.info(
                    "Manual load '%s': profile sensor %s has no valid power_profile yet "
                    "(likely hasn't learned enough cycles), falling back",
                    name,
                    profile_sensor,
                )
                continue
            try:
                source_interval = float(attributes.get("power_profile_interval_min"))
            except (TypeError, ValueError):
                source_interval = None
            if not source_interval or source_interval <= 0:
                logger.warning(
                    "Manual load '%s': profile sensor %s missing/invalid "
                    "power_profile_interval_min, falling back",
                    name,
                    profile_sensor,
                )
                continue

            resampled = utils._resample_power_profile(sequence, source_interval, target_step_min)
            if not resampled:
                continue

            for oc in (optim_conf, params_optim_conf):
                nom = oc.get("nominal_power_of_deferrable_loads")
                if isinstance(nom, list) and k < len(nom):
                    nom[k] = list(resampled)
                op_hours = oc.get("operating_hours_of_each_deferrable_load")
                if isinstance(op_hours, list) and k < len(op_hours):
                    op_hours[k] = len(resampled)
                dispatch = oc.get("load_dispatch_mode")
                if isinstance(dispatch, list) and k < len(dispatch):
                    dispatch[k] = "program"

            logger.info(
                "Manual load '%s': resolved learned power profile from %s "
                "(%d steps at %.1f min, resampled from %.1f min)",
                name,
                profile_sensor,
                len(resampled),
                target_step_min,
                source_interval,
            )
        except Exception as e:  # a WashData/profile-sensor hiccup must never break optimization
            logger.warning(
                "Manual load '%s': error resolving profile sensor %s (%s), falling back",
                name,
                profile_sensor,
                e,
            )
            continue


async def retrieve_home_assistant_data(
    set_type: str,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: str,
    logger: logging.Logger | None = None,
) -> tuple[bool, pd.DataFrame | None, list | None]:
    """Retrieve data from Home Assistant or file and prepare it for optimization."""

    if get_data_from_file:
        success, days_list = await _retrieve_from_file(
            emhass_conf, test_df_literal, rh, retrieve_hass_conf, optim_conf
        )
    else:
        success, days_list = await _retrieve_from_hass(
            set_type, retrieve_hass_conf, optim_conf, rh, logger
        )
    if not success:
        return False, None, days_list
    protected_columns = None
    if set_type == "battery_id":
        # The identifier needs both flow directions of the signed battery
        # power sensor and any legitimately measured 0% SoC sample, so these
        # columns are exempt from the set_zero_min clip (#1041). Columns not
        # present in the retrieved frame are ignored by prepare_data.
        protected_columns = []
        _append_battery_id_sensors(protected_columns, retrieve_hass_conf)
    rh.prepare_data(
        retrieve_hass_conf["sensor_power_load_no_var_loads"],
        load_negative=retrieve_hass_conf["load_negative"],
        set_zero_min=retrieve_hass_conf["set_zero_min"],
        var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
        var_interp=retrieve_hass_conf["sensor_linear_interp"],
        protected_columns=protected_columns,
    )
    return True, rh.df_final.copy(), days_list


def is_model_outdated(
    model_path: pathlib.Path,
    max_age_hours: int,
    logger: logging.Logger,
    label: str = "Adjusted PV model",
) -> bool:
    """
    Check if the saved model file is outdated based on its modification time.

    Format-agnostic: only the file mtime is inspected, so this serves both the
    adjusted-PV regressor pickle and the battery-identification JSON.

    :param model_path: Path to the saved model file.
    :type model_path: pathlib.Path
    :param max_age_hours: Maximum age in hours before model is considered outdated.
    :type max_age_hours: int
    :param logger: Logger object for logging information.
    :type logger: logging.Logger
    :param label: Human-readable name of the artifact, used in the log lines so
        the message matches the caller (e.g. "Battery identification model").
    :type label: str
    :return: True if model is outdated or doesn't exist, False otherwise.
    :rtype: bool
    """
    if not model_path.exists():
        logger.info(f"{label} file does not exist, will train new model")
        return True

    if max_age_hours <= 0:
        logger.info(f"{label} max age is set to 0, forcing model re-fit")
        return True

    model_mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    model_age = datetime.now() - model_mtime
    max_age = timedelta(hours=max_age_hours)

    if model_age > max_age:
        logger.info(
            f"{label} is outdated (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h), will train new model"
        )
        return True
    else:
        logger.info(
            f"Using existing {label} (age: {model_age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h)"
        )
        return False


async def _retrieve_and_fit_pv_model(
    fcst: Forecast,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> bool:
    """
    Helper function to retrieve data and fit the PV adjustment model.

    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # Retrieve data from Home Assistant
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "adjust_pv",
        get_data_from_file,
        retrieve_hass_conf,
        optim_conf,
        rh,
        emhass_conf,
        test_df_literal,
    )
    if not success:
        return False
    # Best-effort retrieval of the curtailment history: timesteps where PV was
    # curtailed must not train the adjustment model (issue #1026). Any failure
    # here (no history, entity missing) falls back to unfiltered training.
    curtailment_series = None
    plant_conf = getattr(fcst, "plant_conf", None) or {}
    if plant_conf.get("compute_curtailment", False) and not get_data_from_file:
        params = getattr(fcst, "params", None) or {}
        curtailment_entity = (
            params.get("passed_data", {})
            .get("custom_pv_curtailment_id", {})
            .get("entity_id", "sensor.p_pv_curtailment")
        )
        try:
            success_curtailment = await rh.get_data(days_list, [curtailment_entity])
            if success_curtailment is not False and curtailment_entity in rh.df_final.columns:
                curtailment_series = rh.df_final[curtailment_entity].copy()
            else:
                fcst.logger.info(
                    f"No history for curtailment entity {curtailment_entity}, "
                    "training the PV adjustment on unfiltered data."
                )
        except Exception as e:
            fcst.logger.info(
                f"Could not retrieve curtailment history ({type(e).__name__}: {e}), "
                "training the PV adjustment on unfiltered data."
            )
    # Call data preparation method
    fcst.adjust_pv_forecast_data_prep(df_input_data, curtailment_series=curtailment_series)
    n_splits = 5
    x_adjust_pv = getattr(fcst, "x_adjust_pv", None)
    if x_adjust_pv is not None and len(x_adjust_pv) <= n_splits:
        fcst.logger.warning(
            f"Not enough data to fit the PV model (found {len(x_adjust_pv)} samples, "
            f"require > {n_splits}). Falling back to unadjusted PV forecast."
        )
        return False
    # Call the fit method
    await fcst.adjust_pv_forecast_fit(
        n_splits=n_splits,
        regression_model=optim_conf["adjusted_pv_regression_model"],
    )
    return True


async def adjust_pv_forecast(
    logger: logging.Logger,
    fcst: Forecast,
    p_pv_forecast: pd.Series,
    get_data_from_file: bool,
    retrieve_hass_conf: dict,
    optim_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    test_df_literal: pd.DataFrame,
) -> pd.Series:
    """
    Adjust the photovoltaic (PV) forecast using historical data and a regression model.

    This method retrieves historical data, prepares it for model fitting, trains a regression
    model, and adjusts the provided PV forecast based on the trained model.

    :param logger: Logger object for logging information and errors.
    :type logger: logging.Logger
    :param fcst: Forecast object used for PV forecast adjustment.
    :type fcst: Forecast
    :param p_pv_forecast: The initial PV forecast to be adjusted.
    :type p_pv_forecast: pd.Series
    :param get_data_from_file: Whether to retrieve data from a file instead of Home Assistant.
    :type get_data_from_file: bool
    :param retrieve_hass_conf: Configuration dictionary for retrieving data from Home Assistant.
    :type retrieve_hass_conf: dict
    :param optim_conf: Configuration dictionary for optimization settings.
    :type optim_conf: dict
    :param rh: RetrieveHass object for interacting with Home Assistant.
    :type rh: RetrieveHass
    :param emhass_conf: Configuration dictionary for emhass paths and settings.
    :type emhass_conf: dict
    :param test_df_literal: DataFrame containing test data for debugging purposes.
    :type test_df_literal: pd.DataFrame
    :return: The adjusted PV forecast as a pandas Series.
    :rtype: pd.Series
    """
    # Normalize data_path to Path object for safety (handles both str and Path types)
    data_path = pathlib.Path(emhass_conf["data_path"])
    model_filename = "adjust_pv_regressor.pkl"
    model_path = data_path / model_filename
    max_age_hours = optim_conf.get("adjusted_pv_model_max_age", 24)
    # Check if model needs to be re-fitted
    if is_model_outdated(model_path, max_age_hours, logger):
        logger.info("Adjusting PV forecast, retrieving history data for model fit")
        success = await _retrieve_and_fit_pv_model(
            fcst,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            logger.warning(
                "Could not train adjusted PV model, falling back to unadjusted PV forecast."
            )
            return p_pv_forecast
    else:
        # Load existing model
        logger.info("Loading existing adjusted PV model from file")
        try:
            async with aiofiles.open(model_path, "rb") as inp:
                content = await inp.read()
                fcst.model_adjust_pv = pickle.loads(content)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
            logger.error(f"Failed to load existing adjusted PV model: {type(e).__name__}: {str(e)}")
            logger.warning(
                "Model file may be corrupted or incompatible. Falling back to re-fitting the model."
            )
            # Use helper function to retrieve data and re-fit model
            success = await _retrieve_and_fit_pv_model(
                fcst,
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
            )
            if not success:
                logger.error(
                    "Failed to retrieve data for model re-fit after load error. Falling back to unadjusted forecast."
                )
                return p_pv_forecast
            logger.info("Successfully re-fitted model after load failure")
        except Exception as e:
            logger.error(
                f"Unexpected error loading adjusted PV model: {type(e).__name__}: {str(e)}"
            )
            logger.error("Cannot recover from this error")
            return False
    # Call the predict method
    p_pv_forecast_in = p_pv_forecast.rename("forecast").to_frame()
    try:
        p_pv_forecast_out = fcst.adjust_pv_forecast_predict(forecasted_pv=p_pv_forecast_in)
    except ValueError as e:
        # A model persisted by an older version may have been trained on a
        # different feature set (e.g. the raw integer "hour" feature that was
        # replaced by the cyclic hour encoding). scikit-learn then raises a
        # ValueError on predict (feature-name mismatch). Re-fit once with the
        # current feature set and retry; other exception types propagate.
        logger.warning(
            f"Adjusted PV model prediction failed ({type(e).__name__}: {e}). "
            "The saved model may predate a feature-set change. Re-fitting."
        )
        success = await _retrieve_and_fit_pv_model(
            fcst,
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
        )
        if not success:
            logger.warning(
                "Could not re-fit the adjusted PV model, falling back to unadjusted PV forecast."
            )
            return p_pv_forecast
        p_pv_forecast_out = fcst.adjust_pv_forecast_predict(forecasted_pv=p_pv_forecast_in)
    # Update the PV forecast
    return p_pv_forecast_out["adjusted_forecast"].rename(None)


# Suggest-tier HA sensor entity ids (fixed; not user-configurable).
BATTERY_ID_CAPACITY_SENSOR = "sensor.battery_identified_capacity"
BATTERY_ID_RTE_SENSOR = "sensor.battery_identified_round_trip_efficiency"


def _batt_conf_val(value, k: int | None):
    """
    Scalar-or-list read for a plant_conf battery value (#1032 array-ifies 9
    plant_conf battery params at number_of_batteries > 1).

    k=None (N=1) always returns ``value`` unchanged, so every call site's N=1
    output is identical to master regardless of this helper's existence. This
    is unrelated to the sensor-key list/bare-string ambiguity (CONTRACT.md's
    SCOPE NOTE on invariant 1): this helper only ever reads plant_conf battery
    params, which keep #1032's own scalar-at-N=1 normalisation. At N>1, k
    selects index k of a per-battery list; a value that is still a bare
    scalar despite k being given is returned as-is (defensive: normal configs
    are already array-ified by check_batt_params before plant_conf reaches
    here, but a hand-built plant_conf, e.g. in a test fixture, may not be).
    """
    if k is None or not isinstance(value, list):
        return value
    return value[k]


def _resolve_battery_sensor_lists(
    retrieve_hass_conf: dict, num_batteries: int, logger: logging.Logger
) -> tuple[list, list] | None:
    """
    Resolve sensor_power_battery / sensor_battery_state_of_charge into exact-
    length per-battery lists, index-matched to the battery config lists.

    Deliberately NO scalar broadcast at num_batteries > 1: one HA sensor
    cannot identify two independent batteries, unlike the numeric plant_conf
    params check_batt_params fans out. At N=1 a bare scalar (today's only
    supported shape) and a length-1 list both resolve to a single-element
    list. Anything else at N>1 (a scalar, or a list of the wrong length) is a
    misconfiguration. So is any list entry that is not a non-empty entity-id
    string, a duplicate id within one list (two batteries sharing a meter), or
    an id shared between the power and SOC lists (one entity cannot be both
    signals). Every rejection returns None after logging one precise warning
    naming the offending key and what's wrong, so the caller can skip cleanly
    before ever touching retrieval.
    """
    resolved: dict[str, list] = {}
    for key in ("sensor_power_battery", "sensor_battery_state_of_charge"):
        raw = retrieve_hass_conf.get(key)
        if isinstance(raw, list):
            if len(raw) != num_batteries:
                logger.warning(
                    "Battery identification: '%s' is a list of length %d but "
                    "number_of_batteries=%d requires exactly %d entity ids "
                    "(one per battery); skipping.",
                    key,
                    len(raw),
                    num_batteries,
                    num_batteries,
                )
                return None
            for idx, entry in enumerate(raw):
                if not isinstance(entry, str) or not entry:
                    logger.warning(
                        "Battery identification: '%s'[%d] is %r, expected a "
                        "non-empty entity-id string; skipping.",
                        key,
                        idx,
                        entry,
                    )
                    return None
            if num_batteries > 1:
                seen: set[str] = set()
                for entry in raw:
                    if entry in seen:
                        logger.warning(
                            "Battery identification: '%s' has duplicate entity id "
                            "%r; one sensor cannot identify two batteries; skipping.",
                            key,
                            entry,
                        )
                        return None
                    seen.add(entry)
            resolved[key] = list(raw)
        else:
            if num_batteries > 1:
                logger.warning(
                    "Battery identification: '%s' is a single value (%r) but "
                    "number_of_batteries=%d requires a list of %d entity ids "
                    "(one per battery, no broadcast for per-battery sensors); "
                    "skipping.",
                    key,
                    raw,
                    num_batteries,
                    num_batteries,
                )
                return None
            resolved[key] = [raw]
    power_list = resolved["sensor_power_battery"]
    soc_list = resolved["sensor_battery_state_of_charge"]
    if num_batteries > 1:
        overlap = set(power_list) & set(soc_list)
        if overlap:
            logger.warning(
                "Battery identification: entity id(s) %s used as both a power "
                "and a state-of-charge sensor; one entity cannot be both "
                "signals; skipping.",
                sorted(overlap),
            )
            return None
    return power_list, soc_list


# On-disk persistence: flat v1 payload at N=1 (same JSON shape as master for
# both a bare-string and a length-1-list sensor config - see CONTRACT.md's
# SCOPE NOTE on invariant 1); a schema_version=2 container of per-battery
# v1-style payloads at N>1, keyed by str(k), so one battery's freshness/
# failure never touches another's entry.
_BATTERY_ID_SCHEMA_VERSION_V2 = 2


def _load_battery_identification_container(json_path: pathlib.Path, logger: logging.Logger) -> dict:
    """
    Read the persisted battery-identification JSON as a v2 per-battery container.

    Tolerant of every "not a usable v2 container" shape: missing file,
    unreadable JSON, a flat v1 payload (has "status" at top level, no
    "batteries" dict), and a foreign/future schema_version all normalise to an
    empty container, so every battery in the per-k loop is treated as
    absent/stale and re-fits - exactly like a missing file does today. A v1
    (or v3+) file is never partially parsed as v2.
    """
    empty = {"schema_version": _BATTERY_ID_SCHEMA_VERSION_V2, "batteries": {}}
    if not json_path.exists():
        return empty
    try:
        payload = json.loads(json_path.read_bytes())
    except (KeyError, ValueError, OSError) as e:
        logger.warning(
            f"Battery identification result unreadable ({type(e).__name__}); will re-fit."
        )
        return empty
    if not isinstance(payload, dict) or not isinstance(payload.get("batteries"), dict):
        return empty
    if payload.get("schema_version") != _BATTERY_ID_SCHEMA_VERSION_V2:
        return empty
    return {
        "schema_version": _BATTERY_ID_SCHEMA_VERSION_V2,
        "batteries": dict(payload["batteries"]),
    }


def _atomic_json_write(json_path: pathlib.Path, data: dict) -> None:
    """
    Write ``data`` as JSON to ``json_path`` atomically (tmp + os.replace).

    Unique temp name (pid + uuid) so overlapping identify_battery calls in
    the same process never share a tmp file before the replace.
    """
    tmp_path = json_path.with_name(
        f"battery_identification.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with open(tmp_path, "w", encoding="utf-8") as outp:
        json.dump(data, outp, indent=2)
    os.replace(tmp_path, json_path)


def _write_battery_identification_container(
    json_path: pathlib.Path, logger: logging.Logger, k: int, entry: dict
) -> None:
    """
    Read-modify-write a single battery's entry into the on-disk v2 container.

    Re-reads the container from disk immediately before merging, rather than
    writing back an in-memory copy loaded at the start of the run: two
    concurrent identify_battery runs (e.g. a dayahead call racing an MPC cron)
    both loading before either writes would otherwise let the second writer's
    whole-container overwrite silently drop the first writer's entry. Re-
    reading here shrinks that lost-update window to the time between this
    read and this write, not the whole per-k loop. The write itself is still
    atomic (tmp + os.replace), so a crash mid-write can never corrupt the file
    or lose any OTHER battery's already-persisted entry.
    """
    container = _load_battery_identification_container(json_path, logger)
    container["batteries"][str(k)] = entry
    _atomic_json_write(json_path, container)


def _battery_fit_is_stale(
    logger: logging.Logger,
    entry,
    current_power: str,
    current_soc: str,
    max_age_hours: float,
    k: int,
) -> bool:
    """
    Per-battery counterpart to :func:`is_model_outdated`: freshness comes from
    the stored entry's own ``fitted_at`` field, not the shared file's mtime, so
    one battery's fresh fit can never suppress another battery's retry.

    ``entry`` is whatever the persisted container has under ``batteries[str(k)]``
    - possibly ``None`` (missing), possibly corrupt (hand-edited or from a
    future incompatible writer). Any shape other than a well-formed dict with
    ``status == "ok"`` is treated as absent/stale for THIS battery only: this
    function must never raise, since it runs inside an eager per-k scan
    (`stale_ks`) computed before the main loop, and one corrupt entry must not
    abort every other battery's fresh-cache read for the whole cycle. This
    mirrors the N=1 cache-hit branch's own ``payload.get("status") == "ok"``
    gate, which a persisted entry always satisfies today (only a successful
    fit is ever written) but a hand-edited or foreign entry may not.

    The stored ``sensors`` pair is compared against the currently resolved
    (power, soc) entity ids: missing (e.g. an entry written before this field
    existed) or mismatched (the lists were edited or reordered since the fit)
    both count as stale, so a cached result is never served for a different
    sensor pair than it was fitted from.
    """
    label = f"Battery {k} identification model"
    if not isinstance(entry, dict):
        if entry is not None:
            logger.warning(
                f"{label} persisted entry is not a usable object "
                f"({type(entry).__name__}); will re-fit."
            )
        else:
            logger.info(f"{label} has no recorded fit, will train new model")
        return True
    if entry.get("status") != "ok":
        logger.warning(
            f"{label} persisted entry has status {entry.get('status')!r}, not 'ok'; will re-fit."
        )
        return True
    sensors = entry.get("sensors")
    if (
        not isinstance(sensors, dict)
        or sensors.get("power") != current_power
        or sensors.get("soc") != current_soc
    ):
        logger.info(f"{label} sensor binding is missing or changed, will re-fit")
        return True
    fitted_at = entry.get("fitted_at")
    if not fitted_at:
        logger.info(f"{label} has no recorded fit, will train new model")
        return True
    if max_age_hours <= 0:
        logger.info(f"{label} max age is set to 0, forcing model re-fit")
        return True
    try:
        fitted_dt = datetime.fromisoformat(fitted_at)
    except (ValueError, TypeError):
        logger.warning(f"{label} fitted_at is unparsable ({fitted_at!r}); will re-fit.")
        return True
    if fitted_dt.tzinfo is None:
        fitted_dt = fitted_dt.replace(tzinfo=UTC)
    age = datetime.now(UTC) - fitted_dt
    max_age = timedelta(hours=max_age_hours)
    if age > max_age:
        logger.info(
            f"{label} is outdated (age: {age.total_seconds() / 3600:.1f}h, "
            f"max: {max_age_hours}h), will train new model"
        )
        return True
    logger.info(
        f"Using existing {label} (age: {age.total_seconds() / 3600:.1f}h, max: {max_age_hours}h)"
    )
    return False


def _log_battery_identification_summary(
    logger: logging.Logger, payload: dict, plant_conf: dict, k: int | None = None
) -> None:
    """
    Log the identified values and, in reported units, how they compare to config.

    k=None: N=1, wording byte-identical to master for a bare-string sensor
    config; a length-1-list config (e.g. the config UI's saved shape, see
    CONTRACT.md's SCOPE NOTE) reaches this with the same payload and produces
    the same wording either way. k=<int>: N>1, per-battery plant_conf reads
    via :func:`_batt_conf_val` and a "Battery {k} " infix, matching every
    other per-battery log line in this module (e.g. the guardrail-failure
    warning in the N>1 loop).
    """
    cap = payload.get("capacity_kwh", {})
    rte = payload.get("round_trip_efficiency", {})
    configured_cap_kwh = (
        float(_batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k)) / 1000.0
    )
    configured_eta = float(_batt_conf_val(plant_conf.get("battery_charge_efficiency", 0), k))
    battery_tag = "" if k is None else f" {k}"
    logger.info(
        "Battery%s identification: capacity %.2f kWh (CI [%s, %s]) vs configured %.2f kWh; "
        "round-trip efficiency %.3f (one-way %.3f) vs configured one-way %.3f.",
        battery_tag,
        cap.get("value") or float("nan"),
        cap.get("ci_low"),
        cap.get("ci_high"),
        configured_cap_kwh,
        rte.get("value") or float("nan"),
        payload.get("eta_charge_symmetric") or float("nan"),
        configured_eta,
    )


def _log_battery_identification_recommendation(
    logger: logging.Logger, payload: dict, plant_conf: dict, k: int | None = None
) -> None:
    """
    Log a plain 'consider updating X from A to B' recommendation (suggest tier).

    k=None: N=1, log wording unchanged from master (independent of whether the
    sensor config is a bare string or a length-1 list). k=<int>: N>1,
    per-battery plant_conf reads via :func:`_batt_conf_val` and the same
    "Battery {k} " infix as :func:`_log_battery_identification_summary`.
    """
    cap = payload.get("capacity_kwh", {})
    identified_cap_kwh = cap.get("value")
    configured_cap_kwh = (
        float(_batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k) or 0) / 1000.0
    )
    identified_eta = payload.get("eta_charge_symmetric")
    configured_eta = _batt_conf_val(plant_conf.get("battery_charge_efficiency"), k)
    battery_tag = "" if k is None else f" {k}"
    logger.info(
        "Battery%s identification recommendation: consider updating "
        "battery_nominal_energy_capacity from %.2f kWh to %.2f kWh, and "
        "battery_charge_efficiency / battery_discharge_efficiency from %s to %s "
        "(symmetric sqrt of the identified round-trip efficiency).",
        battery_tag,
        configured_cap_kwh,
        identified_cap_kwh if identified_cap_kwh is not None else float("nan"),
        configured_eta,
        identified_eta,
    )


async def _publish_battery_identification(
    rh: RetrieveHass, payload: dict, logger: logging.Logger, k: int | None = None
) -> None:
    """
    Publish the two read-only advisory sensors (suggest tier only).

    Attributes carry the confidence interval, sample counts, the last successful
    fit time, and the assumptions, so a user can judge trust from the sensor
    itself. ``fitted_at`` reflects the last SUCCESSFUL fit, never the publish time.

    k=None: N=1, exactly today's two fixed entity ids (zero new entities) -
    true for both a bare-string and a length-1-list sensor config, since
    publish only depends on ``k``, never on the sensor config itself.
    k=<int>: N>1, entity ids suffixed ``_battery<k>`` and friendly names
    suffixed ``Battery {k}``, mirroring the #1032 ``_publish_battery_data``
    per-battery convention (no separator before the digit).
    """
    cap = payload.get("capacity_kwh", {})
    rte = payload.get("round_trip_efficiency", {})
    common = {
        "fitted_at": payload.get("fitted_at"),
        "assumptions": payload.get("assumptions"),
        "n_charge_segments": payload.get("n_charge_segments"),
        "n_discharge_segments": payload.get("n_discharge_segments"),
    }
    cap_entity = (
        BATTERY_ID_CAPACITY_SENSOR if k is None else f"{BATTERY_ID_CAPACITY_SENSOR}_battery{k}"
    )
    rte_entity = BATTERY_ID_RTE_SENSOR if k is None else f"{BATTERY_ID_RTE_SENSOR}_battery{k}"
    name_suffix = "" if k is None else f" Battery {k}"
    await rh.post_scalar_sensor(
        cap_entity,
        cap.get("value"),
        {
            "friendly_name": f"Battery identified capacity{name_suffix}",
            "unit_of_measurement": "kWh",
            "device_class": "energy_storage",
            "ci_low": cap.get("ci_low"),
            "ci_high": cap.get("ci_high"),
            "method": cap.get("method"),
            "crosscheck_theil_sen_kwh": cap.get("crosscheck_theil_sen_kwh"),
            **common,
        },
    )
    await rh.post_scalar_sensor(
        rte_entity,
        rte.get("value"),
        {
            "friendly_name": f"Battery identified round-trip efficiency{name_suffix}",
            "one_way_efficiency_sqrt": payload.get("eta_charge_symmetric"),
            "ci_low": rte.get("ci_low"),
            "ci_high": rte.get("ci_high"),
            "crosscheck_energy_balance": rte.get("crosscheck_energy_balance"),
            **common,
        },
    )


async def identify_battery(
    logger: logging.Logger,
    optim_conf: dict,
    plant_conf: dict,
    retrieve_hass_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    get_data_from_file: bool,
    test_df_literal: str,
) -> None:
    """
    Opt-in battery self-identification (observe/suggest). Structural twin of
    :func:`adjust_pv_forecast`: cadence-gated on a persisted artifact, retrieves
    HA history only when the estimate is stale, and NEVER raises - any failure
    logs a warning and returns, leaving the configured battery values in force.

    v1 never touches ``plant_conf``. In the ``observe`` tier it writes the
    estimate to a JSON under ``data_path`` and logs it; in the ``suggest``
    tier it additionally publishes two read-only HA sensors at N=1, or 2N of
    them (one capacity + one round-trip-efficiency sensor per battery) at
    N>1.

    At ``number_of_batteries`` > 1 each battery is identified independently
    (own config reads, own retrieval columns, own persisted entry, own
    publish): one pack can fit and publish while another still lacks enough
    cycles. See :func:`_identify_battery_impl` for the per-battery loop.

    :param logger: Logger.
    :type logger: logging.Logger
    :param optim_conf: Optimization config (holds the three feature params).
    :type optim_conf: dict
    :param plant_conf: Plant config; read-only here, used for the sanity bound
        and the "configured vs identified" comparison.
    :type plant_conf: dict
    :param retrieve_hass_conf: Retrieve config (holds sensor_power_battery and
        sensor_battery_state_of_charge - a bare entity-id string at N=1, or a
        list of ``number_of_batteries`` entity ids at N>1). At N>1,
        ``_identify_battery_impl`` temporarily mutates these two keys in place
        for the duration of the retrieval ``await`` and restores the original
        values in a ``finally``, so this dict is briefly not what the caller
        put in it if this coroutine is inspected concurrently.
    :type retrieve_hass_conf: dict
    :param rh: RetrieveHass instance.
    :type rh: RetrieveHass
    :param emhass_conf: emhass paths.
    :type emhass_conf: dict
    :param get_data_from_file: Whether history comes from a file instead of HA.
    :type get_data_from_file: bool
    :param test_df_literal: Test data filename for file mode.
    :type test_df_literal: str
    :rtype: None
    """
    # Never-raise boundary: this is an advisory side-feature and must never be
    # able to break an optimization run, so ANY unexpected error is swallowed
    # with a warning, leaving the configured battery values in force.
    try:
        await _identify_battery_impl(
            logger,
            optim_conf,
            plant_conf,
            retrieve_hass_conf,
            rh,
            emhass_conf,
            get_data_from_file,
            test_df_literal,
        )
    except Exception as e:
        logger.warning(
            f"Battery identification failed unexpectedly ({type(e).__name__}: {e}); "
            "keeping configured battery values.",
            exc_info=True,
        )


async def _identify_battery_impl(
    logger: logging.Logger,
    optim_conf: dict,
    plant_conf: dict,
    retrieve_hass_conf: dict,
    rh: RetrieveHass,
    emhass_conf: dict,
    get_data_from_file: bool,
    test_df_literal: str,
) -> None:
    """Implementation of :func:`identify_battery`; wrapped for the never-raise guarantee."""
    num_batteries = utils.validate_num_batteries(plant_conf)

    data_path = pathlib.Path(emhass_conf["data_path"])
    json_path = data_path / "battery_identification.json"
    max_age_hours = optim_conf.get("battery_identification_model_max_age", 24)
    tier = optim_conf.get("battery_identification_trust_tier", "observe")

    if num_batteries == 1:
        # Flat v1 shape, same as master for a bare-string sensor config (a
        # length-1-list config, e.g. the config UI's saved shape, takes the
        # equivalent list-handling path further below - see CONTRACT.md's
        # SCOPE NOTE on invariant 1). In particular: the freshness gate and
        # cached publish come FIRST, exactly like base, WITHOUT ever
        # consulting sensor_power_battery/sensor_battery_state_of_charge -
        # base only read those after a successful retrieval, on the re-fit
        # path, so a cache hit must stay indifferent to whatever (even
        # malformed) shape those two keys happen to be in.
        if not is_model_outdated(
            json_path, max_age_hours, logger, label="Battery identification model"
        ):
            try:
                with open(json_path, "rb") as inp:
                    payload = json.loads(inp.read())
            except (KeyError, ValueError, OSError) as e:
                logger.warning(
                    f"Battery identification result unreadable ({type(e).__name__}); will re-fit."
                )
                payload = None
            if payload is not None and payload.get("status") == "ok":
                _log_battery_identification_summary(logger, payload, plant_conf)
                if tier == "suggest":
                    _log_battery_identification_recommendation(logger, payload, plant_conf)
                    await _publish_battery_identification(rh, payload, logger)
                return
            # Fall through to a re-fit on an unreadable or non-ok cached file
            # (also covers a v2 container left over from a reverted N>1 run:
            # it has no top-level "status", so it reads as not-ok here).

        # Refit path only: resolve the sensor keys now, matching where base
        # first read power_col (after a successful retrieval was decided on,
        # never on the cache-hit path above).
        resolved = _resolve_battery_sensor_lists(retrieve_hass_conf, num_batteries, logger)
        if resolved is None:
            return
        power_list, soc_list = resolved

        logger.info("Battery identification: retrieving history for a fresh fit")
        success, df, _ = await retrieve_home_assistant_data(
            "battery_id",
            get_data_from_file,
            retrieve_hass_conf,
            optim_conf,
            rh,
            emhass_conf,
            test_df_literal,
            logger,
        )
        if not success or df is None:
            logger.warning(
                "Battery identification: could not retrieve history; keeping configured battery values."
            )
            return
        power_col = power_list[0]
        soc_col = soc_list[0]
        if power_col not in df.columns or soc_col not in df.columns:
            logger.warning(
                f"Battery identification: sensors '{power_col}'/'{soc_col}' missing from retrieved "
                "history; keeping configured battery values."
            )
            return
        configured_capacity_wh = float(plant_conf.get("battery_nominal_energy_capacity", 0) or 0)
        result = BatteryIdentification(logger).identify(
            df, power_col, soc_col, configured_capacity_wh
        )
        for msg in result.messages:
            logger.info("Battery identification: %s", msg)
        if not result.is_ok:
            logger.warning(
                f"Battery identification did not pass guardrails (status={result.status}); "
                "keeping configured battery values. Existing results file left untouched."
            )
            return

        # Persist ONLY a successful estimate, atomically. A failed fit must not
        # bump the file mtime (which would suppress retries for max_age_hours).
        payload = result.to_dict()
        payload["fitted_at"] = datetime.now(UTC).isoformat()
        payload["trust_tier"] = tier
        payload["configured_at_fit_time"] = {
            "battery_nominal_energy_capacity": plant_conf.get("battery_nominal_energy_capacity"),
            "battery_charge_efficiency": plant_conf.get("battery_charge_efficiency"),
            "battery_discharge_efficiency": plant_conf.get("battery_discharge_efficiency"),
        }
        _atomic_json_write(json_path, payload)

        _log_battery_identification_summary(logger, payload, plant_conf)
        if tier == "suggest":
            _log_battery_identification_recommendation(logger, payload, plant_conf)
            await _publish_battery_identification(rh, payload, logger)
        return

    # N > 1: schema_version=2 container, one entry per battery. Each battery is
    # independent end to end (own freshness, own fit, own persisted entry, own
    # publish) under the one global trust tier; a v1 flat file left over from a
    # reverted N=1 run is treated as absent (re-fit every battery), never
    # partially parsed. Resolver-first here (unlike N=1): the mutation below
    # needs the resolved lists before any retrieval, and the sensor pair is
    # also an input to the per-battery freshness check.
    resolved = _resolve_battery_sensor_lists(retrieve_hass_conf, num_batteries, logger)
    if resolved is None:
        # Warning already logged by the resolver, naming the offending key.
        return
    power_list, soc_list = resolved

    container = _load_battery_identification_container(json_path, logger)
    batteries = container["batteries"]
    stale_ks = [
        k
        for k in range(num_batteries)
        if _battery_fit_is_stale(
            logger, batteries.get(str(k)), power_list[k], soc_list[k], max_age_hours, k
        )
    ]

    df = None
    if stale_ks:
        # One batched retrieval covering only the currently-stale batteries'
        # sensors, not the full lists: the per-k loop below only ever reads
        # power_list[k]/soc_list[k] for a stale k, so an already-fresh
        # battery's columns would just be fetched and never read. Restricting
        # the mutation to the stale subset means a fresh battery's sensor
        # going away (or lacking history) can never affect a stale sibling's
        # re-fit, and an unreachable sensor among the stale batteries defers
        # only those batteries, not the fresh ones (which are never part of
        # this batch). _retrieve_from_hass's append site reads
        # sensor_power_battery/sensor_battery_state_of_charge straight off
        # retrieve_hass_conf; presenting the resolved subset here - restored
        # immediately after - is the narrowest way to get the stale
        # batteries' entity ids into one var_list without threading a new
        # argument through retrieve_home_assistant_data/_retrieve_from_hass,
        # which are shared with every other set_type.
        logger.info("Battery identification: retrieving history for a fresh fit")
        stale_power = [power_list[k] for k in stale_ks]
        stale_soc = [soc_list[k] for k in stale_ks]
        original_power = retrieve_hass_conf.get("sensor_power_battery")
        original_soc = retrieve_hass_conf.get("sensor_battery_state_of_charge")
        retrieve_hass_conf["sensor_power_battery"] = stale_power
        retrieve_hass_conf["sensor_battery_state_of_charge"] = stale_soc
        try:
            success, df, _ = await retrieve_home_assistant_data(
                "battery_id",
                get_data_from_file,
                retrieve_hass_conf,
                optim_conf,
                rh,
                emhass_conf,
                test_df_literal,
                logger,
            )
        finally:
            retrieve_hass_conf["sensor_power_battery"] = original_power
            retrieve_hass_conf["sensor_battery_state_of_charge"] = original_soc
        if not success:
            df = None

    for k in range(num_batteries):
        if k not in stale_ks:
            # Fresh cached entry: log/publish from it, no re-fit.
            entry = batteries[str(k)]
            _log_battery_identification_summary(logger, entry, plant_conf, k=k)
            if tier == "suggest":
                _log_battery_identification_recommendation(logger, entry, plant_conf, k=k)
                await _publish_battery_identification(rh, entry, logger, k=k)
            continue
        if df is None:
            logger.warning(
                f"Battery {k} identification: could not retrieve history; "
                "keeping configured battery values."
            )
            continue
        power_col = power_list[k]
        soc_col = soc_list[k]
        if power_col not in df.columns or soc_col not in df.columns:
            logger.warning(
                f"Battery {k} identification: sensors '{power_col}'/'{soc_col}' missing from "
                "retrieved history; keeping configured battery values."
            )
            continue
        configured_capacity_wh = float(
            _batt_conf_val(plant_conf.get("battery_nominal_energy_capacity", 0), k) or 0
        )
        result = BatteryIdentification(logger).identify(
            df, power_col, soc_col, configured_capacity_wh
        )
        for msg in result.messages:
            logger.info("Battery %d identification: %s", k, msg)
        if not result.is_ok:
            logger.warning(
                f"Battery {k} identification did not pass guardrails (status={result.status}); "
                "keeping configured battery values. Existing entry left untouched."
            )
            continue

        # Persist ONLY a successful estimate for battery k, read-modify-write,
        # atomic. A failed fit for battery k (above) never touches battery k's
        # (or any other battery's) previously stored entry. "sensors" binds
        # this entry to the exact pair it was fitted from, so a later run
        # that edits or reorders the lists can never serve it as a stale-free
        # cache hit for the wrong sensor pair (see _battery_fit_is_stale).
        new_entry = result.to_dict()
        new_entry["fitted_at"] = datetime.now(UTC).isoformat()
        new_entry["trust_tier"] = tier
        new_entry["sensors"] = {"power": power_col, "soc": soc_col}
        new_entry["configured_at_fit_time"] = {
            "battery_nominal_energy_capacity": _batt_conf_val(
                plant_conf.get("battery_nominal_energy_capacity"), k
            ),
            "battery_charge_efficiency": _batt_conf_val(
                plant_conf.get("battery_charge_efficiency"), k
            ),
            "battery_discharge_efficiency": _batt_conf_val(
                plant_conf.get("battery_discharge_efficiency"), k
            ),
        }
        _write_battery_identification_container(json_path, logger, k, new_entry)

        _log_battery_identification_summary(logger, new_entry, plant_conf, k=k)
        if tier == "suggest":
            _log_battery_identification_recommendation(logger, new_entry, plant_conf, k=k)
            await _publish_battery_identification(rh, new_entry, logger, k=k)


async def _prepare_perfect_optim(ctx: SetupContext):
    """Helper to prepare data for perfect optimization."""
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "perfect-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    if not success:
        return None
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
    }


async def _get_dayahead_pv_forecast(ctx: SetupContext):
    """Helper to retrieve and optionally adjust PV forecast."""
    # Check if we should calculate PV forecast
    if not (
        ctx.optim_conf["set_use_pv"]
        or ctx.optim_conf.get("weather_forecast_method", None) == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    p_pv_forecast = ctx.fcst.get_power_from_weather(df_weather)
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


def _apply_df_freq_horizon(
    df: pd.DataFrame, retrieve_hass_conf: dict, prediction_horizon: int | None
) -> pd.DataFrame:
    """Helper to apply frequency adjustment and prediction horizon slicing."""
    # Handle Frequency
    if retrieve_hass_conf.get("optimization_time_step"):
        step = retrieve_hass_conf["optimization_time_step"]
        if not isinstance(step, pd._libs.tslibs.timedeltas.Timedelta):
            step = pd.to_timedelta(step, "minute")
        df = df[~df.index.duplicated(keep="last")]
        df = df.asfreq(step)
    else:
        df = utils.set_df_index_freq(df)
    # Handle Prediction Horizon
    if prediction_horizon:
        # Slice the dataframe up to the horizon
        df = copy.deepcopy(df)[df.index[0] : df.index[min(prediction_horizon, len(df)) - 1]]
    return df


async def _prepare_dayahead_optim(ctx: SetupContext, stage_times: dict | None = None):
    """Helper to prepare data for day-ahead optimization.

    :param stage_times: Optional dict to record per-stage elapsed times (seconds).
    :type stage_times: dict, optional
    """
    if stage_times is None:
        stage_times = {}
    # Get PV Forecast
    with stage_timer(stage_times, "pv_forecast", ctx.logger):
        p_pv_forecast, df_weather = await _get_dayahead_pv_forecast(ctx)
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    with stage_timer(stage_times, "load_forecast", ctx.logger):
        p_load_forecast = await ctx.fcst.get_load_forecast(
            days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
            method=ctx.optim_conf["load_forecast_method"],
        )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        ctx.logger.error("Unable to get load forecast.")
        return None
    # Build Input DataFrame
    df_input_data_dayahead = pd.DataFrame(
        np.transpose(np.vstack([p_pv_forecast.values, p_load_forecast.values])),
        index=p_pv_forecast.index,
        columns=["p_pv_forecast", "p_load_forecast"],
    )
    # Apply Frequency and Prediction Horizon
    # Use explicitly passed horizon, avoiding JSON re-parsing
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _get_naive_mpc_history(ctx: SetupContext):
    """Helper to retrieve historical data for Naive MPC."""
    # Check if we need to skip historical data retrieval
    is_list_forecast = ctx.optim_conf.get("load_forecast_method") == "list"
    is_list_weather = ctx.optim_conf.get("weather_forecast_method") == "list"
    no_pv = not ctx.optim_conf["set_use_pv"]

    if (is_list_forecast and is_list_weather) or (is_list_forecast and no_pv):
        return True, None, None, False  # success, df, days_list, set_mix_forecast
    # Retrieve data from Home Assistant
    success, df_input_data, days_list = await retrieve_home_assistant_data(
        "naive-mpc-optim",
        ctx.get_data_from_file,
        ctx.retrieve_hass_conf,
        ctx.optim_conf,
        ctx.rh,
        ctx.emhass_conf,
        test_df_literal,
        ctx.logger,
    )
    return success, df_input_data, days_list, True


async def _get_naive_mpc_pv_forecast(ctx: SetupContext, set_mix_forecast, df_input_data):
    """Helper to generate PV forecast for Naive MPC."""
    # If PV is disabled and no weather list, return zero series
    if not (
        ctx.optim_conf["set_use_pv"] or ctx.optim_conf.get("weather_forecast_method") == "list"
    ):
        return pd.Series(0, index=ctx.fcst.forecast_dates), None
    # Get weather forecast
    df_weather = await ctx.fcst.get_weather_forecast(
        method=ctx.optim_conf["weather_forecast_method"]
    )
    if isinstance(df_weather, bool) and not df_weather:
        return None, None
    # Calculate PV power
    p_pv_forecast = ctx.fcst.get_power_from_weather(
        df_weather, set_mix_forecast=set_mix_forecast, df_now=df_input_data
    )
    # Adjust PV forecast if needed
    if ctx.optim_conf["set_use_adjusted_pv"]:
        p_pv_forecast = await adjust_pv_forecast(
            ctx.logger,
            ctx.fcst,
            p_pv_forecast,
            ctx.get_data_from_file,
            ctx.retrieve_hass_conf,
            ctx.optim_conf,
            ctx.rh,
            ctx.emhass_conf,
            test_df_literal,
        )
    return p_pv_forecast, df_weather


async def _prepare_naive_mpc_optim(ctx: SetupContext, stage_times: dict | None = None):
    """Helper to prepare data for Naive MPC optimization.

    :param stage_times: Optional dict to record per-stage elapsed times (seconds).
    :type stage_times: dict, optional
    """
    if stage_times is None:
        stage_times = {}
    # Retrieve Historical Data
    success, df_input_data, days_list, set_mix_forecast = await _get_naive_mpc_history(ctx)
    if not success:
        return None
    # Get PV Forecast
    with stage_timer(stage_times, "pv_forecast", ctx.logger):
        p_pv_forecast, df_weather = await _get_naive_mpc_pv_forecast(
            ctx, set_mix_forecast, df_input_data
        )
    if p_pv_forecast is None:
        return None
    # Get Load Forecast
    with stage_timer(stage_times, "load_forecast", ctx.logger):
        p_load_forecast = await ctx.fcst.get_load_forecast(
            days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
            method=ctx.optim_conf["load_forecast_method"],
            set_mix_forecast=set_mix_forecast,
            df_now=df_input_data,
        )
    if isinstance(p_load_forecast, bool) and not p_load_forecast:
        return None
    # Build and Format Input DataFrame
    df_input_data_dayahead = pd.concat([p_pv_forecast, p_load_forecast], axis=1)
    df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]
    # Reuse freq/horizon helper
    prediction_horizon = ctx.params["passed_data"].get("prediction_horizon")
    df_input_data_dayahead = _apply_df_freq_horizon(
        df_input_data_dayahead, ctx.retrieve_hass_conf, prediction_horizon
    )
    return {
        "df_input_data": df_input_data,
        "days_list": days_list,
        "df_input_data_dayahead": df_input_data_dayahead,
        "df_weather": df_weather,
        "p_pv_forecast": p_pv_forecast,
        "p_load_forecast": p_load_forecast,
    }


async def _prepare_ml_fit_predict(ctx: SetupContext):
    """Helper to prepare data for ML fit/predict/tune."""
    days_to_retrieve = ctx.params["passed_data"]["historic_days_to_retrieve"]
    model_type = ctx.params["passed_data"]["model_type"]
    var_model = ctx.params["passed_data"]["var_model"]
    if ctx.get_data_from_file:
        filename = model_type + ".pkl"
        filename_path = ctx.emhass_conf["data_path"] / filename
        async with aiofiles.open(filename_path, "rb") as inp:
            content = await inp.read()
            df_input_data, _, _, _ = pickle.loads(content)
        df_input_data = df_input_data[df_input_data.index[-1] - pd.offsets.Day(days_to_retrieve) :]
        return {"df_input_data": df_input_data}
    else:
        days_list = utils.get_days_list(days_to_retrieve)
        var_list = [var_model]
        if not await ctx.rh.get_data(days_list, var_list):
            return None
        ctx.rh.prepare_data(
            var_model,
            load_negative=ctx.retrieve_hass_conf.get("load_negative", False),
            set_zero_min=ctx.retrieve_hass_conf.get("set_zero_min", True),
            var_replace_zero=ctx.retrieve_hass_conf.get("sensor_replace_zero", []),
            var_interp=ctx.retrieve_hass_conf.get("sensor_linear_interp", []),
            skip_renaming=True,
        )
        return {"df_input_data": ctx.rh.df_final.copy()}


def _prepare_regressor_fit(ctx: SetupContext):
    """Helper to prepare data for Regressor fit/predict."""
    csv_file = ctx.params["passed_data"].get("csv_file", None)
    if not csv_file:
        ctx.logger.error("csv_file is required for regressor actions but was not provided.")
        return None
    if ctx.get_data_from_file:
        base_path = ctx.emhass_conf["data_path"]
        filename_path = pathlib.Path(base_path) / csv_file
    else:
        filename_path = ctx.emhass_conf["data_path"] / csv_file
    if filename_path.is_file():
        df_input_data = pd.read_csv(filename_path, parse_dates=True)
    else:
        ctx.logger.error(
            f"The CSV file {csv_file} was not found in path: {ctx.emhass_conf['data_path']}"
        )
        return None
    # Validate columns
    required_columns = []
    if "features" in ctx.params["passed_data"]:
        required_columns.extend(ctx.params["passed_data"]["features"])
    if "target" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["target"])
    if "timestamp" in ctx.params["passed_data"]:
        required_columns.append(ctx.params["passed_data"]["timestamp"])
    if not set(required_columns).issubset(df_input_data.columns):
        ctx.logger.error(
            f"The csv file does not contain the required columns: {', '.join(required_columns)}"
        )
        return None
    return {"df_input_data": df_input_data}


async def set_input_data_dict(
    emhass_conf: dict,
    costfun: str,
    params: str,
    runtimeparams: str,
    set_type: str,
    logger: logging.Logger,
    get_data_from_file: bool | None = False,
) -> dict:
    """
    Set up some of the data needed for the different actions.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param costfun: The type of cost function to use for optimization problem
    :type costfun: str
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param set_type: Set the type of setup based on following type of optimization
    :type set_type: str
    :param logger: The passed logger object
    :type logger: logging object
    :param get_data_from_file: Use data from saved CSV file (useful for debug)
    :type get_data_from_file: bool, optional
    :return: A dictionnary with multiple data used by the action functions
    :rtype: dict

    """
    stage_times = {}
    logger.info("Setting up needed data")
    normalized_set_type = str(set_type).strip().lower()
    # Parse Parameters
    if (params is not None) and (params != "null"):
        if isinstance(params, str):
            params = dict(orjson.loads(params))
    else:
        params = {}
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    if type(retrieve_hass_conf) is bool:
        return False
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        emhass_conf,
    )
    log_runtime_banner(logger, optim_conf=optim_conf)
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    # Initialize Core Objects
    rh = RetrieveHass(
        retrieve_hass_conf["hass_url"],
        retrieve_hass_conf["long_lived_token"],
        retrieve_hass_conf["optimization_time_step"],
        retrieve_hass_conf["time_zone"],
        params,
        emhass_conf,
        logger,
        get_data_from_file=get_data_from_file,
    )

    def _resolve_test_data_file() -> pathlib.Path | None:
        """Resolve a valid location for test_df_final.pkl in offline test mode."""
        candidates = [
            emhass_conf["data_path"] / test_df_literal,
            emhass_conf["root_path"].parent.parent / "data" / test_df_literal,
            pathlib.Path.cwd() / "data" / test_df_literal,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    # Retrieve HA config when required by action.
    if normalized_set_type != "thermal-two-stage-plan":
        if get_data_from_file:
            test_data_path = _resolve_test_data_file()
            if test_data_path is None:
                logger.error(
                    f"Offline test data not found. Expected '{test_df_literal}' in data paths."
                )
                return False
            async with aiofiles.open(test_data_path, "rb") as inp:
                content = await inp.read()
                _, _, _, rh.ha_config = pickle.loads(content)
        elif not await rh.get_ha_config():
            return False
        if isinstance(params, dict):
            params_str = orjson.dumps(params).decode("utf-8")
            params = utils.update_params_with_ha_config(params_str, rh.ha_config)
        else:
            params = utils.update_params_with_ha_config(params, rh.ha_config)
    if isinstance(params, str):
        params = dict(orjson.loads(params))
    costfun = optim_conf.get("costfun", costfun)
    # Two-tier guard:
    #   - actions_without_fcst_or_opt: read saved results only; build neither.
    #   - actions_skip_optim_cache: need a Forecast object but no Optimization.
    #     Keeping these out of the OptimizationCache path stops them poisoning
    #     the cache key with config-default values that a subsequent
    #     naive-mpc-optim call would then miss against.
    actions_without_fcst_or_opt = [
        "publish-data",
        "export-influxdb-to-csv",
        "thermal-two-stage-plan",
    ]
    actions_skip_optim_cache = [
        "forecast-model-fit",
        "forecast-model-predict",
        "forecast-model-tune",
        "forecast-calibration",
        "heating-need-forecast",
        "heating-model-refit",
    ]
    # Resolve any manually-committed load's learned power profile (e.g. from
    # WashData) fresh for this action - must happen before Forecast/
    # OptimizationCache/Optimization are built below, since a resolved
    # profile changes optim_conf's structure (see _resolve_manual_load_profiles).
    if (
        optim_conf.get("manual_load_enabled", False)
        and normalized_set_type not in actions_without_fcst_or_opt
        and normalized_set_type not in actions_skip_optim_cache
    ):
        await _resolve_manual_load_profiles(
            rh, optim_conf, params.get("optim_conf", {}), retrieve_hass_conf, params, logger
        )
    if normalized_set_type in actions_without_fcst_or_opt:
        fcst = None
        opt = None
        logger.debug(f"Skipping Optimization creation for action: {set_type}")
    else:
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        if normalized_set_type in actions_skip_optim_cache:
            opt = None
            logger.debug(f"Skipping OptimizationCache for action: {set_type}")
        else:
            # Try to get cached Optimization object for warm-starting
            _num_ts = len(fcst.forecast_dates)
            opt = OptimizationCache.get(
                optim_conf, plant_conf, costfun, retrieve_hass_conf, logger, _num_ts
            )
            if opt is None:
                # Cache miss - create new Optimization object
                opt = Optimization(
                    retrieve_hass_conf,
                    optim_conf,
                    plant_conf,
                    fcst.var_load_cost,
                    fcst.var_prod_price,
                    costfun,
                    emhass_conf,
                    logger,
                    num_timesteps=_num_ts,
                )
                # Store in cache for future warm-starts
                OptimizationCache.put(
                    opt, optim_conf, plant_conf, costfun, retrieve_hass_conf, logger, _num_ts
                )
            else:
                # Cache hit - update references that may have changed
                # (logger, var names from forecast, and runtime-configurable optim_conf values)
                opt.logger = logger
                opt.var_load_cost = fcst.var_load_cost
                opt.var_prod_price = fcst.var_prod_price
                # Update internal config dictionaries to prevent stale lookups
                # for runtime parameters (like battery_target_state_of_charge)
                opt.plant_conf = plant_conf
                opt.optim_conf = optim_conf
                # Update CVXPY Parameters for thermal start temperatures
                # This is critical: updating optim_conf alone doesn't change baked-in constraint values
                opt.update_thermal_start_temps(optim_conf)
                # Same idea for battery power limits — they participate in
                # constraints via cp.Parameter and need the runtime value
                # propagated even on a cache hit.
                opt.update_battery_power_limits(plant_conf)
            # Update runtime-configurable solver options from optim_conf
            # These don't affect problem structure, so they're safe to update on cached object
            runtime_solver_opts = [
                "lp_solver_timeout",
                "lp_solver_mip_rel_gap",
                "num_threads",
            ]
            for key in runtime_solver_opts:
                if key in optim_conf:
                    opt.optim_conf[key] = optim_conf[key]
    # Create SetupContext
    ctx = SetupContext(
        retrieve_hass_conf=retrieve_hass_conf,
        optim_conf=optim_conf,
        plant_conf=plant_conf,
        emhass_conf=emhass_conf,
        params=params,
        logger=logger,
        get_data_from_file=get_data_from_file,
        rh=rh,
        fcst=fcst,
    )
    # Initialize Default Return Data
    data_results = {
        "df_input_data": None,
        "df_input_data_dayahead": None,
        "df_weather": None,
        "p_pv_forecast": None,
        "p_load_forecast": None,
        "days_list": None,
    }
    # Delegate to Helpers based on set_type
    result = None
    if set_type == "dayahead-optim":
        # Dayahead uses granular per-stage timing inside _prepare_dayahead_optim;
        # no coarse outer wrap here to avoid double-counting.
        result = await _prepare_dayahead_optim(ctx, stage_times=stage_times)
    elif set_type == "perfect-optim":
        # Perfect uses historical HA data; no input_data stage timing —
        # price_prep / optim_solve / publish are timed inside perfect_forecast_optim.
        result = await _prepare_perfect_optim(ctx)
    elif set_type == "naive-mpc-optim":
        # Naive MPC uses granular per-stage timing inside _prepare_naive_mpc_optim;
        # no coarse outer wrap here to avoid double-counting.
        result = await _prepare_naive_mpc_optim(ctx, stage_times=stage_times)
    elif set_type in ["forecast-model-fit", "forecast-model-predict", "forecast-model-tune"]:
        result = await _prepare_ml_fit_predict(ctx)
    elif set_type == "forecast-calibration":
        # The calibration action retrieves its own (longer) history window inside
        # forecast_calibration(); no ML-prep here.
        result = {}
    elif set_type == "heating-need-forecast":
        # Retrieves its own live indoor-temperature reading and weather forecast
        # inside compute_heating_forecast(); no generic prep needed here.
        result = {}
    elif set_type == "heating-model-refit":
        # Retrieves its own (long) history window inside refit_heating_model();
        # no generic prep needed here.
        result = {}
    elif set_type == "regressor-model-fit":
        result = _prepare_regressor_fit(ctx)
    elif set_type == "regressor-model-predict":
        if get_data_from_file:
            result = _prepare_regressor_fit(ctx)
        else:
            result = {}
    elif set_type == "thermal-two-stage-plan":
        result = {}
    elif set_type == "publish-data" or set_type == "export-influxdb-to-csv":
        result = {}
    else:
        logger.error(f"The passed action set_type parameter '{set_type}' is not valid")
        result = {}
    if result is None:
        return False
    # Opt-in battery self-identification (observe/suggest). Runs alongside the
    # optimization prep for the two live optim actions; never affects the
    # optimizer in v1. Cadence-gated and never raises, so it cannot break an
    # optimization run.
    if (
        set_type in ("dayahead-optim", "naive-mpc-optim")
        and optim_conf.get("set_use_battery", False)
        and optim_conf.get("set_use_battery_identification", False)
    ):
        await identify_battery(
            logger,
            optim_conf,
            plant_conf,
            retrieve_hass_conf,
            rh,
            emhass_conf,
            get_data_from_file,
            test_df_literal,
        )
    data_results.update(result)
    # Build Final Dictionary
    input_data_dict = {
        "emhass_conf": emhass_conf,
        "retrieve_hass_conf": retrieve_hass_conf,
        "optim_conf": optim_conf,
        "plant_conf": plant_conf,
        "rh": rh,
        "opt": opt,
        "fcst": fcst,
        "costfun": costfun,
        "params": params,
        "stage_times": stage_times,
        **data_results,
    }
    return input_data_dict


async def weather_forecast_cache(
    emhass_conf: dict, params: str, runtimeparams: str, logger: logging.Logger
) -> bool:
    """
    Perform a call to get forecast function, intend to save results to cache.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :type emhass_conf: dict
    :param params: Configuration parameters passed from data/options.json
    :type params: str
    :param runtimeparams: Runtime optimization parameters passed as a dictionary
    :type runtimeparams: str
    :param logger: The passed logger object
    :type logger: logging object
    :return: A bool for function completion
    :rtype: bool

    """
    # Parsing yaml
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
    # Treat runtimeparams
    (
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
    ) = await utils.treat_runtimeparams(
        runtimeparams,
        params,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "forecast",
        logger,
        emhass_conf,
    )
    # Make sure weather_forecast_cache is true
    if (params is not None) and (params != "null"):
        params = orjson.loads(params)
    else:
        params = {}
    params["passed_data"]["weather_forecast_cache"] = True
    params = orjson.dumps(params).decode("utf-8")
    # Create Forecast object
    fcst = Forecast(retrieve_hass_conf, optim_conf, plant_conf, params, emhass_conf, logger)
    result = await fcst.get_weather_forecast(optim_conf["weather_forecast_method"])
    if isinstance(result, bool) and not result:
        return False

    return True


def _log_optimization_summary(input_data_dict: dict, logger: logging.Logger) -> None:
    """Emit the one-line optimization summary (total elapsed + top stage).

    Reads per-stage timings recorded by the orchestrators in ``input_data_dict["stage_times"]``.
    No-op if no stages were recorded.
    """
    stage_times = input_data_dict.get("stage_times", {})
    if not stage_times:
        return
    total = sum(stage_times.values())
    top_name, top_s = max(stage_times.items(), key=lambda x: x[1])
    pct = int(100 * top_s / total) if total > 0 else 0
    logger.info(f"Optimization completed in {total:.1f}s (top: {top_name}={top_s:.1f}s, {pct}%)")


async def perfect_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = True,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the perfect forecast optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    logger.info("Performing perfect forecast optimization")
    # Load cost and prod price forecast
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data = input_data_dict["fcst"].get_load_cost_forecast(
            input_data_dict["df_input_data"],
            method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
            list_and_perfect=True,
        )
        if isinstance(df_input_data, bool) and not df_input_data:
            return False
        df_input_data = input_data_dict["fcst"].get_prod_price_forecast(
            df_input_data,
            method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
            list_and_perfect=True,
        )
    if isinstance(df_input_data, bool) and not df_input_data:
        return False
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res = input_data_dict["opt"].perform_perfect_forecast_optim(
            df_input_data, input_data_dict["days_list"]
        )
    # Save CSV file for analysis
    if save_data_to_file:
        filename = "opt_res_perfect_optim_" + input_data_dict["costfun"] + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )
    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save perfect results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(input_data_dict, last_run.ACTION_PERFECT_OPTIM, opt_res, _t0, logger)

    return opt_res


def prepare_forecast_and_weather_data(
    input_data_dict: dict,
    logger: logging.Logger,
    warn_on_resolution: bool = False,
) -> pd.DataFrame | bool:
    """
    Prepare forecast data with load costs, production prices, outdoor temperature, and GHI.

    This helper function eliminates duplication between dayahead_forecast_optim and naive_mpc_optim.

    :param input_data_dict: Dictionary with forecast and input data
    :type input_data_dict: dict
    :param logger: Logger object
    :type logger: logging.Logger
    :param warn_on_resolution: Whether to warn about GHI resolution mismatch
    :type warn_on_resolution: bool
    :return: Prepared DataFrame or False on error
    :rtype: pd.DataFrame | bool
    """
    # Get load cost forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_load_cost_forecast(
        input_data_dict["df_input_data_dayahead"],
        method=input_data_dict["fcst"].optim_conf["load_cost_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Get production price forecast
    df_input_data_dayahead = input_data_dict["fcst"].get_prod_price_forecast(
        df_input_data_dayahead,
        method=input_data_dict["fcst"].optim_conf["production_price_forecast_method"],
    )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False

    # Add outdoor temperature if provided
    passed_outdoor_temp = input_data_dict["params"]["passed_data"].get(
        "outdoor_temperature_forecast"
    )

    if passed_outdoor_temp is not None:
        forecast_len = len(df_input_data_dayahead)

        # If the passed forecast is shorter than the horizon, pad it with the last value to prevent Pandas crashes
        if len(passed_outdoor_temp) < forecast_len:
            logger.warning(
                "Passed outdoor_temperature_forecast length (%s) "
                "is shorter than the prediction horizon (%s). Padding with the last value.",
                len(passed_outdoor_temp),
                forecast_len,
            )
            last_val = passed_outdoor_temp[-1] if len(passed_outdoor_temp) > 0 else 15.0
            passed_outdoor_temp = passed_outdoor_temp + [last_val] * (
                forecast_len - len(passed_outdoor_temp)
            )

        # If it's longer (e.g. 48h data for 13h horizon), slice it securely
        df_input_data_dayahead["outdoor_temperature_forecast"] = passed_outdoor_temp[:forecast_len]

    # Auto-fallback to temp_air from Open-Meteo weather forecast
    elif (
        input_data_dict["df_weather"] is not None
        and "temp_air" in input_data_dict["df_weather"].columns
    ):
        dayahead_index = df_input_data_dayahead.index
        weather_series = input_data_dict["df_weather"]["temp_air"].copy()

        # Handle Timezone Mismatches
        # If optimization is naive but weather is aware -> Strip weather TZ
        if dayahead_index.tz is None and weather_series.index.tz is not None:
            weather_series.index = weather_series.index.tz_localize(None)
        # If optimization is aware but weather is naive -> Localize weather
        elif dayahead_index.tz is not None and weather_series.index.tz is None:
            weather_series.index = weather_series.index.tz_localize(dayahead_index.tz)
        # If both are aware -> Convert weather to optimization TZ
        elif dayahead_index.tz is not None and weather_series.index.tz is not None:
            weather_series.index = weather_series.index.tz_convert(dayahead_index.tz)

        # Robust Reindexing (The fix for "Found 48 NaNs")
        # method='nearest' snaps 10:00:15 to 10:00:00
        # tolerance='1h' prevents filling data from too far away
        df_input_data_dayahead["outdoor_temperature_forecast"] = weather_series.reindex(
            dayahead_index, method="nearest", tolerance=pd.Timedelta("1h")
        )

        # Final safety fill (forward/backward) for any remaining gaps
        if df_input_data_dayahead["outdoor_temperature_forecast"].isnull().any():
            df_input_data_dayahead["outdoor_temperature_forecast"] = (
                df_input_data_dayahead["outdoor_temperature_forecast"]
                .fillna(method="ffill")
                .fillna(method="bfill")
            )

    # Merge GHI (Global Horizontal Irradiance) from weather forecast if available
    if input_data_dict["df_weather"] is not None and "ghi" in input_data_dict["df_weather"].columns:
        dayahead_index = df_input_data_dayahead.index
        ghi_series = input_data_dict["df_weather"]["ghi"].copy()

        # Handle Timezone Mismatches (Same as above)
        if dayahead_index.tz is None and ghi_series.index.tz is not None:
            ghi_series.index = ghi_series.index.tz_localize(None)
        elif dayahead_index.tz is not None and ghi_series.index.tz is None:
            ghi_series.index = ghi_series.index.tz_localize(dayahead_index.tz)
        elif dayahead_index.tz is not None and ghi_series.index.tz is not None:
            ghi_series.index = ghi_series.index.tz_convert(dayahead_index.tz)

        # Check time resolution if requested
        if (
            warn_on_resolution
            and len(input_data_dict["df_weather"].index) > 1
            and len(dayahead_index) > 1
        ):
            weather_index = input_data_dict["df_weather"].index
            weather_freq = (weather_index[1] - weather_index[0]).total_seconds()
            dayahead_freq = (dayahead_index[1] - dayahead_index[0]).total_seconds()
            if weather_freq > 2 * dayahead_freq:
                logger.warning(
                    "Weather data time resolution (%.0fs) is much coarser than dayahead index (%.0fs). "
                    "Step changes in GHI may occur.",
                    weather_freq,
                    dayahead_freq,
                )

        # Robust Reindexing
        df_input_data_dayahead["ghi"] = ghi_series.reindex(
            dayahead_index, method="nearest", tolerance=pd.Timedelta("1h")
        )

        # Final safety fill
        if df_input_data_dayahead["ghi"].isnull().any():
            df_input_data_dayahead["ghi"] = (
                df_input_data_dayahead["ghi"].fillna(method="ffill").fillna(method="bfill")
            )

        logger.debug(
            "Merged GHI data into optimization input: mean=%.1f W/m², max=%.1f W/m²",
            df_input_data_dayahead["ghi"].mean(),
            df_input_data_dayahead["ghi"].max(),
        )

    return df_input_data_dayahead


async def dayahead_forecast_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the day-ahead optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    await _apply_manual_load_runtime_overrides(input_data_dict, logger)
    soc_init = input_data_dict["params"]["passed_data"].get("soc_init")
    soc_final = input_data_dict["params"]["passed_data"].get("soc_final")
    logger.info(
        f"Performing day-ahead forecast optimization with soc_init: {soc_init}, soc_final: {soc_final}"
    )
    # Prepare forecast data with costs, prices, outdoor temp, and GHI
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data_dayahead = prepare_forecast_and_weather_data(
            input_data_dict, logger, warn_on_resolution=False
        )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    # Read these from params["optim_conf"] rather than relying on
    # self.optim_conf inside perform_optimization's fallback: params and the
    # opt object's own optim_conf are different dict objects by this point
    # in set_input_data_dict (see _apply_manual_load_runtime_overrides /
    # _resolve_manual_load_profiles, which mutate params["optim_conf"]) -
    # without passing these through explicitly, per-cycle overrides (manual
    # load window pinning, resolved WashData profiles) would never reach the
    # solver here, same as naive_mpc_optim already does below.
    def_total_hours = input_data_dict["params"]["optim_conf"].get(
        "operating_hours_of_each_deferrable_load", None
    )
    def_total_timestep = input_data_dict["params"]["optim_conf"].get(
        "operating_timesteps_of_each_deferrable_load", None
    )
    def_start_timestep = input_data_dict["params"]["optim_conf"].get(
        "start_timesteps_of_each_deferrable_load"
    )
    def_end_timestep = input_data_dict["params"]["optim_conf"].get(
        "end_timesteps_of_each_deferrable_load"
    )
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res_dayahead = input_data_dict["opt"].perform_dayahead_forecast_optim(
            df_input_data_dayahead,
            input_data_dict["p_pv_forecast"],
            input_data_dict["p_load_forecast"],
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            stage_times=input_data_dict["stage_times"],
            def_init_temp=_build_def_init_temp(input_data_dict, logger),
        )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_dayahead.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save day_ahead results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(
        input_data_dict, last_run.ACTION_DAYAHEAD_OPTIM, opt_res_dayahead, _t0, logger
    )

    return opt_res_dayahead


async def naive_mpc_optim(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    debug: bool | None = False,
) -> pd.DataFrame:
    """
    Perform a call to the naive Model Predictive Controller optimization routine.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: Save optimization results to CSV file
    :type save_data_to_file: bool, optional
    :param debug: A debug option useful for unittests
    :type debug: bool, optional
    :return: The output data of the optimization
    :rtype: pd.DataFrame

    """
    _t0 = _time.monotonic()
    await _apply_manual_load_runtime_overrides(input_data_dict, logger)
    logger.info("Performing naive MPC optimization")
    # Prepare forecast data with costs, prices, outdoor temp, and GHI (with resolution warning)
    with stage_timer(input_data_dict["stage_times"], "price_prep", logger):
        df_input_data_dayahead = prepare_forecast_and_weather_data(
            input_data_dict, logger, warn_on_resolution=True
        )
    if isinstance(df_input_data_dayahead, bool) and not df_input_data_dayahead:
        return False
    # The specifics params for the MPC at runtime
    prediction_horizon = min(
        input_data_dict["params"]["passed_data"]["prediction_horizon"],
        len(df_input_data_dayahead),
    )
    soc_init = input_data_dict["params"]["passed_data"]["soc_init"]
    soc_final = input_data_dict["params"]["passed_data"]["soc_final"]
    soc_target = input_data_dict["params"]["passed_data"].get("soc_target", None)
    soc_target_timestep = input_data_dict["params"]["passed_data"].get("soc_target_timestep", None)
    current_period_peak = input_data_dict["params"]["passed_data"].get("current_period_peak", None)
    def_total_hours = input_data_dict["params"]["optim_conf"].get(
        "operating_hours_of_each_deferrable_load", None
    )
    def_total_timestep = input_data_dict["params"]["optim_conf"].get(
        "operating_timesteps_of_each_deferrable_load", None
    )
    def_start_timestep = input_data_dict["params"]["optim_conf"].get(
        "start_timesteps_of_each_deferrable_load"
    )
    def_end_timestep = input_data_dict["params"]["optim_conf"].get(
        "end_timesteps_of_each_deferrable_load"
    )
    with stage_timer(input_data_dict["stage_times"], "optim_solve", logger):
        opt_res_naive_mpc = input_data_dict["opt"].perform_naive_mpc_optim(
            df_input_data_dayahead,
            input_data_dict["p_pv_forecast"],
            input_data_dict["p_load_forecast"],
            prediction_horizon,
            soc_init,
            soc_final,
            soc_target=soc_target,
            soc_target_timestep=soc_target_timestep,
            current_period_peak=current_period_peak,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            stage_times=input_data_dict["stage_times"],
            def_init_temp=_build_def_init_temp(input_data_dict, logger),
        )
    # Save CSV file for publish_data
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_naive_mpc_" + today.strftime("%Y_%m_%d") + ".csv"
    else:  # Just save the latest optimization results
        filename = default_csv_filename
    if not debug:
        opt_res_naive_mpc.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / filename,
            index_label="timestamp",
        )

    if not isinstance(input_data_dict["params"], dict):
        params = orjson.loads(input_data_dict["params"])
    else:
        params = input_data_dict["params"]

    # if continual_publish, save mpc results to data_path/entities json
    if input_data_dict["retrieve_hass_conf"].get("continual_publish", False) or params[
        "passed_data"
    ].get("entity_save", False):
        with stage_timer(input_data_dict["stage_times"], "publish", logger):
            # Trigger the publish function, save entity data and not post to HA
            await publish_data(input_data_dict, logger, entity_save=True, dont_post=True)

    _log_optimization_summary(input_data_dict, logger)
    _record_optim_snapshot(
        input_data_dict, last_run.ACTION_NAIVE_MPC_OPTIM, opt_res_naive_mpc, _t0, logger
    )

    return opt_res_naive_mpc


def _get_weather_features(input_data_dict: dict) -> list[str]:
    """Read the configured mlforecaster weather covariate columns (empty when unset)."""
    return list(input_data_dict["params"]["passed_data"].get("mlforecaster_weather_features") or [])


async def _attach_weather_covariates(
    input_data_dict: dict, data: pd.DataFrame, weather_features: list[str], logger: logging.Logger
) -> pd.DataFrame:
    """Attach the configured weather covariate columns onto a load DataFrame (in place, returned).

    Used for the training data (fit/tune) so the columns are aligned to the load history. A no-op
    that returns ``data`` unchanged when no ``weather_features`` are configured.
    """
    if not weather_features:
        return data
    covariates = await input_data_dict["fcst"].get_weather_covariates(data.index, weather_features)
    for column in weather_features:
        data[column] = covariates[column].to_numpy()
    logger.info("Attached %s weather covariate(s) to the load data", len(weather_features))
    return data


async def forecast_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecaster]:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :return: The DataFrame containing the forecast data results without and with backtest and the `mlforecaster` object
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, mlforecaster]
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    var_model = input_data_dict["params"]["passed_data"]["var_model"]
    sklearn_model = input_data_dict["params"]["passed_data"]["sklearn_model"]
    num_lags = input_data_dict["params"]["passed_data"]["num_lags"]
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    perform_backtest = input_data_dict["params"]["passed_data"]["perform_backtest"]
    # Optionally attach weather covariates (aligned to the load history) for the model to use.
    weather_features = _get_weather_features(input_data_dict)
    data = await _attach_weather_covariates(input_data_dict, data, weather_features, logger)
    # The ML forecaster object
    mlf = MLForecaster(
        data,
        model_type,
        var_model,
        sklearn_model,
        num_lags,
        input_data_dict["emhass_conf"],
        logger,
        weather_features=weather_features,
    )
    # Fit the ML model
    df_pred, df_pred_backtest = await mlf.fit(
        split_date_delta=split_date_delta, perform_backtest=perform_backtest
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("saved model to " + str(filename_path))
    return df_pred, df_pred_backtest, mlf


async def forecast_calibration(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Compute an on-demand load forecast calibration report from HA history.

    Retrieves the load history, then compares the built-in load forecast methods
    (naive, typical, mlforecaster) against the realised load over held-out
    test/val windows. Reporting only: no model or artifact is saved and no
    optimization is affected.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: The calibration result dict, or None when there is not enough history
    :rtype: dict | None
    """
    passed_data = input_data_dict["params"]["passed_data"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    rh = input_data_dict["rh"]
    var_model = passed_data.get("var_model") or retrieve_hass_conf.get(
        "sensor_power_load_no_var_loads", "sensor.power_load_no_var_loads"
    )

    # The calibration day windows are runtime-overridable, report-only knobs (never
    # read from the config GUI). Each falls back to its module default, so an empty
    # request reproduces the standard 90 / 14 / 14 day report. A non-positive value
    # is treated as "not set" and falls back to the default; an over-short retrieval
    # window is caught by compute_forecast_calibration's minimum-history gate, which
    # returns a clean error rather than crashing.
    def _positive_or_default(key: str, default: int) -> int:
        value = int(passed_data.get(key) or 0)
        return value if value > 0 else default

    days_to_retrieve = _positive_or_default(
        "calibration_days_to_retrieve", CALIBRATION_DEFAULT_DAYS
    )
    test_days = _positive_or_default("calibration_test_days", CALIBRATION_TEST_DAYS)
    val_days = _positive_or_default("calibration_val_days", CALIBRATION_VAL_DAYS)

    days_list = utils.get_days_list(days_to_retrieve)
    if not await rh.get_data(days_list, [var_model]):
        logger.error("Forecast calibration: failed to retrieve load history from Home Assistant")
        return None
    rh.prepare_data(
        var_model,
        load_negative=retrieve_hass_conf.get("load_negative", False),
        set_zero_min=retrieve_hass_conf.get("set_zero_min", True),
        var_replace_zero=retrieve_hass_conf.get("sensor_replace_zero", []),
        var_interp=retrieve_hass_conf.get("sensor_linear_interp", []),
        skip_renaming=True,
    )
    load = rh.df_final[var_model].copy()
    # The mlforecaster row is fit fresh in memory with a fast, stable
    # LinearRegression (not the user's configured/saved model), so the report is
    # quick and reproducible and never depends on a slow estimator (e.g. SVR/MLP).
    result = await compute_forecast_calibration(
        load,
        rh.freq,
        input_data_dict["emhass_conf"],
        logger,
        sklearn_model="LinearRegression",
        test_days=test_days,
        val_days=val_days,
        var_model=var_model,
    )
    if result.get("error"):
        return None
    return result


async def compute_heating_forecast(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Forecast indoor temperature forward from now, assuming heating stays off.

    Uses the fitted thermal-mass physics model (scripts/thermal_mass_physics_model.py,
    emhass.thermal.thermal_mass_physics) to simulate open-loop from the current live
    indoor temperature through a real weather forecast, answering "if the heat pump
    stays off, when does the house drop below comfort". Publishes
    sensor.indoor_temp_forecast (the full predicted curve) and
    sensor.heating_needed_by (the first crossing timestamp, or a 'beyond_horizon'
    sentinel). EMHASS never calls a device service here - these are informational
    forecast sensors only, same "publish only" pattern as the rest of this fork.

    Requires the optional `thermal` extra (torch/scikit-learn) - importing
    emhass.thermal pulls that in transitively, same as thermal-two-stage-plan.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when disabled/not yet fit/no data
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]

    if not optim_conf.get("heating_forecast_enabled", False):
        logger.debug("heating-need-forecast: disabled (heating_forecast_enabled=False)")
        return None

    fitted = await load_json_blob(emhass_conf, "thermal_physics_params.json", logger, default=None)
    if not fitted or "params" not in fitted:
        logger.error(
            "heating-need-forecast: no fitted model found (data/thermal_physics_params.json). "
            "Run scripts/thermal_mass_physics_model.py at least once."
        )
        return None

    from emhass.thermal.thermal_mass_physics import (
        PARAM_NAMES,
        _infer_timestep_hours,
        _prepare_inputs,
        _simulate_open_loop,
    )

    try:
        params = np.array([fitted["params"][name] for name in PARAM_NAMES], dtype=float)
    except KeyError as e:
        logger.error("heating-need-forecast: fitted params missing key %s", e)
        return None
    # Zero the wind-*direction* terms explicitly: Open-Meteo's forecast has no
    # wind-direction field, and these were tiny fitted contributors (~0.0002).
    # Explicit zeroing (rather than defaulting wind_bearing to 0, which makes
    # cos(0)=1 and would apply the full ua_wind_cos coefficient) is a documented
    # simplification, not an accident.
    params[PARAM_NAMES.index("ua_wind_sin_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("ua_wind_cos_per_h_per_speed")] = 0.0

    indoor_sensor = retrieve_hass_conf.get("heatpump_indoor_temp_sensor", "")
    if not indoor_sensor:
        logger.error("heating-need-forecast: heatpump_indoor_temp_sensor is not configured")
        return None

    days_list = utils.get_days_list(2)
    if not await rh.get_data(days_list, [indoor_sensor]):
        logger.error(
            "heating-need-forecast: failed to retrieve live indoor temperature from Home Assistant"
        )
        return None
    rh.prepare_data(
        indoor_sensor,
        load_negative=False,
        set_zero_min=False,
        var_replace_zero=[],
        var_interp=[indoor_sensor],
        skip_renaming=True,
    )
    indoor_history = rh.df_final[indoor_sensor].dropna()
    if indoor_history.empty:
        logger.error("heating-need-forecast: no live indoor temperature data available")
        return None
    current_indoor_temp = float(indoor_history.iloc[-1])

    df_weather = await input_data_dict["fcst"].get_weather_forecast(
        method=optim_conf.get("weather_forecast_method", "open-meteo")
    )
    if isinstance(df_weather, bool) and not df_weather:
        logger.error("heating-need-forecast: failed to retrieve a weather forecast")
        return None
    if df_weather is None or len(df_weather) == 0:
        logger.error("heating-need-forecast: weather forecast is empty")
        return None

    horizon_hours = float(optim_conf.get("heating_forecast_horizon_hours", 72))
    step_minutes = retrieve_hass_conf["optimization_time_step"].total_seconds() / 60.0
    requested_steps = int(round(horizon_hours * 60.0 / step_minutes)) if step_minutes else 0
    if requested_steps and len(df_weather) < requested_steps:
        logger.warning(
            "heating-need-forecast: weather forecast only covers %d of the requested %d "
            "steps (%.0fh horizon) - pass a larger 'delta_forecast_daily' runtime param "
            "on the triggering call to lengthen it.",
            len(df_weather),
            requested_steps,
            horizon_hours,
        )

    df_physics_input = pd.DataFrame(
        {
            "outdoor_temp": df_weather["temp_air"],
            "wind_speed": df_weather["wind_speed"],
            "ghi": df_weather["ghi"],
            "dni": df_weather["dni"],
            "dhi": df_weather["dhi"],
            "heatpump_duty": 0.0,
        },
        index=df_weather.index,
    )
    thermal_inputs = _prepare_inputs(
        df_physics_input,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    dt_h = _infer_timestep_hours(df_weather.index)
    sim = _simulate_open_loop(
        thermal_inputs,
        params,
        dt_h=dt_h,
        initial_air=current_indoor_temp,
        initial_mass=current_indoor_temp,
        initial_q_emit=0.0,
    )

    safety_margin = float(optim_conf.get("heating_forecast_safety_margin_c", 0.5))
    comfort_min = float(optim_conf.get("heating_forecast_comfort_min_temp", 19.0))
    adjusted = sim.room - safety_margin
    below = np.where(adjusted < comfort_min)[0]
    heating_needed_by = df_weather.index[int(below[0])].isoformat() if len(below) else "beyond_horizon"

    passed_data = input_data_dict["params"]["passed_data"]
    temp_forecast_entity = passed_data.get("custom_indoor_temp_forecast_id")
    needed_by_entity = passed_data.get("custom_heating_needed_by_id")
    if temp_forecast_entity is None or needed_by_entity is None:
        logger.error(
            "heating-need-forecast: target entities not registered "
            "(heating_forecast_enabled was True at optim time but isn't now?)"
        )
        return None

    common_kwargs = {
        "publish_prefix": passed_data.get("publish_prefix", ""),
        "save_entities": False,
        "dont_post": passed_data.get("dont_post", False),
    }
    temp_series = pd.Series(sim.room, index=df_weather.index)
    await rh.post_data(
        temp_series,
        0,
        temp_forecast_entity["entity_id"],
        temp_forecast_entity["device_class"],
        temp_forecast_entity["unit_of_measurement"],
        temp_forecast_entity["friendly_name"],
        type_var="temperature",
        **common_kwargs,
    )
    needed_by_series = pd.Series([heating_needed_by] * len(df_weather), index=df_weather.index)
    await rh.post_data(
        needed_by_series,
        0,
        needed_by_entity["entity_id"],
        needed_by_entity["device_class"],
        needed_by_entity["unit_of_measurement"],
        needed_by_entity["friendly_name"],
        type_var="forecast_event",
        **common_kwargs,
    )

    result = {
        "heating_needed_by": heating_needed_by,
        "current_indoor_temp": current_indoor_temp,
        "comfort_min_temp": comfort_min,
        "safety_margin_c": safety_margin,
        "horizon_hours": horizon_hours,
        "forecast_steps": len(df_weather),
    }
    await save_json_blob(emhass_conf, "heating_forecast_last_run.json", result, logger)
    logger.info("heating-need-forecast: heating_needed_by=%s", heating_needed_by)
    return result


# Maps each ThermalInputs/_prepare_inputs column name to the retrieve_hass_conf
# key naming its live entity_id. Only heatpump_indoor_temp_sensor (room_temp,
# the fit target) is required; every other column is best-effort - a missing
# sensor just falls back to _prepare_inputs' own static default, matching how
# compute_heating_forecast already treats heatpump_duty (forced to 0) as an
# acceptable simplification rather than a hard failure.
_REFIT_SENSOR_COLUMN_MAP = {
    "heatpump_indoor_temp_sensor": "room_temp",
    "heatpump_power_sensor": "electric_power",
    "heatpump_gas_meter_sensor": "gas_consumption",
    "heatpump_duty_sensor": "heatpump_duty",
    "heatpump_flow_temp_sensor": "supply_temp",
    "heatpump_outdoor_temp_sensor": "outdoor_temp",
    "heatpump_weather_wind_speed_sensor": "wind_speed",
    "heatpump_weather_wind_direction_sensor": "wind_bearing",
    "heatpump_weather_ghi_sensor": "ghi",
    "heatpump_weather_dni_sensor": "dni",
    "heatpump_weather_dhi_sensor": "dhi",
}
_REFIT_MIN_ROWS = 500  # a handful of days at 15-30min resolution - below this, don't even try


async def refit_heating_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit the thermal-mass physics model against fresh Home Assistant history
    and deploy it for heating-need-forecast to use.

    Intended to be triggered on a schedule (weekly or so) via a Home Assistant
    automation, same "externally triggered, no scheduler inside EMHASS" pattern
    as forecast-model-fit/tune. Pulls a rolling window of history through
    RetrieveHass.get_data() - routed to InfluxDB when use_influxdb is
    configured, since the HA recorder's own retention (purge_keep_days,
    typically 10 days) is far shorter than the multi-week window a physics
    refit needs (see docs/passing_data.md). A newly-fit model only replaces
    the deployed one if its fit quality clears heating_model_refit_max_mae_c -
    a bad fit (e.g. from a sensor outage during the window) is logged and
    discarded, leaving the previous parameters in place.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when disabled/failed
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]

    if not optim_conf.get("heating_model_refit_enabled", False):
        logger.debug("heating-model-refit: disabled (heating_model_refit_enabled=False)")
        return None
    if not retrieve_hass_conf.get("use_influxdb", False):
        logger.error(
            "heating-model-refit: use_influxdb is not enabled. The refit window "
            "(heating_model_refit_window_days) is normally far longer than Home "
            "Assistant's own recorder retention - configure InfluxDB rather than "
            "risk silently fitting on a truncated REST window."
        )
        return None

    indoor_sensor = retrieve_hass_conf.get("heatpump_indoor_temp_sensor", "")
    if not indoor_sensor:
        logger.error("heating-model-refit: heatpump_indoor_temp_sensor is not configured")
        return None

    sensor_map: dict[str, str] = {}
    for conf_key, column in _REFIT_SENSOR_COLUMN_MAP.items():
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if entity_id:
            sensor_map[entity_id] = column
        elif conf_key != "heatpump_indoor_temp_sensor":
            logger.warning(
                "heating-model-refit: %s is not configured - '%s' will use its static "
                "default for this refit.",
                conf_key,
                column,
            )

    from emhass.thermal.thermal_mass_physics import (
        PARAM_NAMES,
        _fit_temperature_params,
        _infer_timestep_hours,
        _prepare_inputs,
    )

    window_days = int(optim_conf.get("heating_model_refit_window_days", 60))
    days_list = utils.get_days_list(window_days)
    if not await rh.get_data(days_list, list(sensor_map.keys())):
        logger.error("heating-model-refit: failed to retrieve history from Home Assistant/InfluxDB")
        return None

    df_raw = rh.df_final.rename(columns=sensor_map)
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    if "room_temp" not in df_raw.columns:
        # InfluxDB returned no data at all for heatpump_indoor_temp_sensor - rename()
        # is a no-op for a column that was never fetched in the first place.
        logger.error("heating-model-refit: no room_temp data retrieved from InfluxDB")
        return None
    n_rows = int(df_raw["room_temp"].notna().sum()) if "room_temp" in df_raw.columns else 0
    if n_rows < _REFIT_MIN_ROWS:
        logger.error(
            "heating-model-refit: only %d room_temp data points retrieved over %d "
            "days (need at least %d) - aborting rather than fitting on too little data.",
            n_rows,
            window_days,
            _REFIT_MIN_ROWS,
        )
        return None

    thermal_inputs = _prepare_inputs(
        df_raw,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
        facade_azimuth_deg=180.0,
        facade_tilt_deg=90.0,
        solar_horizontal_weight=0.35,
        solar_facade_weight=0.65,
    )
    dt_h = _infer_timestep_hours(df_raw.index)
    segment_len = max(1, round(24.0 / dt_h))  # ~24h segments, matching the original fit

    params, fit_info = _fit_temperature_params(
        thermal_inputs, dt_h=dt_h, segment_len=segment_len, max_nfev=300
    )

    max_mae = float(optim_conf.get("heating_model_refit_max_mae_c", 1.5))
    fit_mae = fit_info["fit_mae_c"]
    if fit_mae > max_mae:
        logger.error(
            "heating-model-refit: fit MAE %.3f°C exceeds heating_model_refit_max_mae_c "
            "(%.3f°C) - keeping the previously deployed model, not overwriting.",
            fit_mae,
            max_mae,
        )
        return {"deployed": False, "fit_mae_c": fit_mae, "max_mae_c": max_mae, "n_rows": n_rows}

    params_dict = {name: float(value) for name, value in zip(PARAM_NAMES, params, strict=True)}
    deployed = await save_json_blob(
        emhass_conf,
        "thermal_physics_params.json",
        {
            "params": params_dict,
            "fit_info": fit_info,
            "source": "auto-refit",
            "refit_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
            "window_days": window_days,
            "n_rows": n_rows,
        },
        logger,
    )
    result = {
        "deployed": deployed,
        "fit_mae_c": fit_mae,
        "max_mae_c": max_mae,
        "n_rows": n_rows,
        "window_days": window_days,
    }
    logger.info(
        "heating-model-refit: deployed=%s fit_mae_c=%.3f (n_rows=%d, window_days=%d)",
        deployed,
        fit_mae,
        n_rows,
        window_days,
    )
    return result


async def forecast_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    use_last_window: bool | None = True,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> pd.DataFrame:
    r"""Perform a forecast model predict using a previously trained skforecast model.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param use_last_window: True if the 'last_window' option should be used for the \
        custom machine learning forecast model. The 'last_window=True' means that the data \
        that will be used to generate the new forecast will be freshly retrieved from \
        Home Assistant. This data is needed because the forecast model is an auto-regressive \
        model with lags. If 'False' then the data using during the model train is used. Defaults to True
    :type use_last_window: Optional[bool], optional
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast prediction data
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this predict method",
            )
            return
    # Make predictions
    if use_last_window:
        data_last_window = copy.deepcopy(input_data_dict["df_input_data"])
    else:
        data_last_window = None
    # When the model was trained with weather covariates, supply the future weather over the
    # forecast horizon so the recursive predict has the exog columns it expects.
    weather_future = await input_data_dict["fcst"]._build_weather_future(data_last_window, mlf)
    predictions = await mlf.predict(data_last_window, weather_future=weather_future)
    # Publish data to a Home Assistant sensor
    model_predict_publish = input_data_dict["params"]["passed_data"]["model_predict_publish"]
    model_predict_entity_id = input_data_dict["params"]["passed_data"]["model_predict_entity_id"]
    model_predict_device_class = input_data_dict["params"]["passed_data"][
        "model_predict_device_class"
    ]
    model_predict_unit_of_measurement = input_data_dict["params"]["passed_data"][
        "model_predict_unit_of_measurement"
    ]
    model_predict_friendly_name = input_data_dict["params"]["passed_data"][
        "model_predict_friendly_name"
    ]
    publish_prefix = input_data_dict["params"]["passed_data"]["publish_prefix"]
    if model_predict_publish is True:
        # Estimate the current index
        now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
            second=0, microsecond=0
        )
        if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
            idx_closest = predictions.index.get_indexer([now_precise], method="ffill")[0]
        elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
            idx_closest = predictions.index.get_indexer([now_precise], method="bfill")[0]
        if idx_closest == -1:
            idx_closest = predictions.index.get_indexer([now_precise], method="nearest")[0]
        # Publish Load forecast
        await input_data_dict["rh"].post_data(
            predictions,
            idx_closest,
            model_predict_entity_id,
            model_predict_device_class,
            model_predict_unit_of_measurement,
            model_predict_friendly_name,
            type_var="mlforecaster",
            publish_prefix=publish_prefix,
        )
    return predictions


async def forecast_model_tune(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlf: MLForecaster | None = None,
) -> tuple[pd.DataFrame, MLForecaster]:
    """Tune a forecast model hyperparameters using bayesian optimization.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    :param mlf: The 'mlforecaster' object previously trained. This is mainly used for debug \
        and unit testing. In production the actual model will be read from a saved pickle file. Defaults to None
    :type mlf: Optional[mlforecaster], optional
    :return: The DataFrame containing the forecast data results using the optimized model
    :rtype: pd.DataFrame
    """
    # Load model
    model_type = input_data_dict["params"]["passed_data"]["model_type"]
    filename = model_type + default_pkl_suffix
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlf = pickle.loads(content)
                logger.debug("loaded saved model from " + str(filename_path))
        else:
            logger.error(
                "The ML forecaster file ("
                + str(filename_path)
                + ") was not found, please run a model fit method before this tune method",
            )
            return None, None
    # Tune the model
    split_date_delta = input_data_dict["params"]["passed_data"]["split_date_delta"]
    if debug:
        n_trials = 5
    else:
        n_trials = input_data_dict["params"]["passed_data"]["n_trials"]
    df_pred_optim = await mlf.tune(
        split_date_delta=split_date_delta, n_trials=n_trials, debug=debug
    )
    # Save model
    if not debug:
        filename = model_type + default_pkl_suffix
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlf, pickle.HIGHEST_PROTOCOL))
            logger.debug("Saved model to " + str(filename_path))
    return df_pred_optim, mlf


async def regressor_model_fit(
    input_data_dict: dict, logger: logging.Logger, debug: bool | None = False
) -> MLRegressor:
    """Perform a forecast model fit from training data retrieved from Home Assistant.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    data = copy.deepcopy(input_data_dict["df_input_data"])
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    if "regression_model" in input_data_dict["params"]["passed_data"]:
        regression_model = input_data_dict["params"]["passed_data"]["regression_model"]
    else:
        logger.error("parameter: 'regression_model' not passed")
        return False
    if "features" in input_data_dict["params"]["passed_data"]:
        features = input_data_dict["params"]["passed_data"]["features"]
    else:
        logger.error("parameter: 'features' not passed")
        return False
    if "target" in input_data_dict["params"]["passed_data"]:
        target = input_data_dict["params"]["passed_data"]["target"]
    else:
        logger.error("parameter: 'target' not passed")
        return False
    if "timestamp" in input_data_dict["params"]["passed_data"]:
        timestamp = input_data_dict["params"]["passed_data"]["timestamp"]
    else:
        logger.error("parameter: 'timestamp' not passed")
        return False
    if "date_features" in input_data_dict["params"]["passed_data"]:
        date_features = input_data_dict["params"]["passed_data"]["date_features"]
    else:
        logger.error("parameter: 'date_features' not passed")
        return False
    # The MLRegressor object
    mlr = MLRegressor(data, model_type, regression_model, features, target, timestamp, logger)
    # Fit the ML model
    fit = await mlr.fit(date_features=date_features)
    if not fit:
        return False
    # Save model
    if not debug:
        filename = model_type + "_mlr.pkl"
        filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(mlr, pickle.HIGHEST_PROTOCOL))
    return mlr


async def regressor_model_predict(
    input_data_dict: dict,
    logger: logging.Logger,
    debug: bool | None = False,
    mlr: MLRegressor | None = None,
) -> np.ndarray:
    """Perform a prediction from csv file.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param debug: True to debug, useful for unit testing, defaults to False
    :type debug: Optional[bool], optional
    """
    if "model_type" in input_data_dict["params"]["passed_data"]:
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
    else:
        logger.error("parameter: 'model_type' not passed")
        return False
    filename = model_type + "_mlr.pkl"
    filename_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not debug:
        if filename_path.is_file():
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                mlr = pickle.loads(content)
        else:
            logger.error(
                "The ML forecaster file was not found, please run a model fit method before this predict method",
            )
            return False
    if "new_values" in input_data_dict["params"]["passed_data"]:
        new_values = input_data_dict["params"]["passed_data"]["new_values"]
    else:
        logger.error("parameter: 'new_values' not passed")
        return False
    # Predict from csv file
    prediction = await mlr.predict(new_values)
    mlr_predict_entity_id = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_entity_id", "sensor.mlr_predict"
    )
    mlr_predict_device_class = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_device_class", "power"
    )
    mlr_predict_unit_of_measurement = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_unit_of_measurement", "W"
    )
    mlr_predict_friendly_name = input_data_dict["params"]["passed_data"].get(
        "mlr_predict_friendly_name", "mlr predictor"
    )
    # Publish prediction
    idx = 0
    if not debug:
        await input_data_dict["rh"].post_data(
            prediction,
            idx,
            mlr_predict_entity_id,
            mlr_predict_device_class,
            mlr_predict_unit_of_measurement,
            mlr_predict_friendly_name,
            type_var="mlregressor",
        )
    return prediction


def _read_optional_list(value: object) -> list[str] | None:
    """Parse optional model list from list or comma-separated string."""
    if value is None:
        return None
    if isinstance(value, list):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out or None
    if isinstance(value, str):
        out = [v.strip() for v in value.split(",") if v.strip()]
        return out or None
    return None


async def thermal_two_stage_plan(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = True,
) -> pd.DataFrame:
    """Run a two-stage (coarse/fine) thermal planning workflow from CSV input."""
    # Imported lazily: emhass.thermal pulls in torch/scikit-learn, which are not
    # required for the rest of EMHASS and are not declared as core dependencies.
    from emhass.thermal import ModelRegistry, build_two_stage_optimization_plan, load_target_registries

    params = input_data_dict.get("params", {})
    if isinstance(params, str):
        params = orjson.loads(params)

    passed = params.get("passed_data", {})
    optim_conf = input_data_dict.get("optim_conf", {})
    retrieve_hass_conf = input_data_dict.get("retrieve_hass_conf", {})

    data_csv = (
        passed.get("thermal_data_csv_path")
        or optim_conf.get("heatpump_two_stage_data_csv")
        or passed.get("csv_file")
    )
    model_dir = (
        passed.get("thermal_model_dir")
        or optim_conf.get("heatpump_two_stage_model_dir")
        or optim_conf.get("heatpump_ml_model_path")
    )
    if not data_csv:
        raise ValueError(
            "thermal_data_csv_path is required (runtime or config heatpump_two_stage_data_csv)"
        )
    if not model_dir:
        raise ValueError(
            "thermal_model_dir is required (runtime or config heatpump_two_stage_model_dir)"
        )

    data_path = pathlib.Path(data_csv)
    if not data_path.is_absolute():
        data_path = input_data_dict["emhass_conf"]["root_path"] / data_path
    model_path = pathlib.Path(model_dir)
    if not model_path.is_absolute():
        model_path = input_data_dict["emhass_conf"]["root_path"] / model_path

    if not data_path.exists():
        raise FileNotFoundError(f"Thermal data CSV not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Thermal model directory not found: {model_path}")

    timestamp_col = passed.get("thermal_timestamp_col") or "timestamp"
    target_col = passed.get("thermal_target_col") or "room_temp"
    outdoor_col = passed.get("thermal_outdoor_col") or "outdoor_temp"

    horizon = int(
        passed.get("thermal_horizon")
        or optim_conf.get("heatpump_two_stage_horizon")
        or optim_conf.get("heatpump_pinn_lookahead")
        or 144
    )
    top_k = int(passed.get("thermal_top_k") or optim_conf.get("heatpump_two_stage_top_k") or 3)
    target_room_temp_min = float(passed.get("thermal_target_room_temp_min") or 20.0)
    target_room_temp_max = float(passed.get("thermal_target_room_temp_max") or 22.0)
    price_weight = float(passed.get("thermal_price_weight") or 0.8)
    comfort_weight = float(passed.get("thermal_comfort_weight") or 5.0)
    energy_weight = float(passed.get("thermal_energy_weight") or 1.0)

    coarse_models = _read_optional_list(
        passed.get("thermal_coarse_models") or optim_conf.get("heatpump_two_stage_coarse_models")
    )
    fine_models = _read_optional_list(
        passed.get("thermal_fine_models") or optim_conf.get("heatpump_two_stage_fine_models")
    )

    lat = float(retrieve_hass_conf.get("Latitude") or passed.get("thermal_latitude") or 52.1202)
    lon = float(retrieve_hass_conf.get("Longitude") or passed.get("thermal_longitude") or 4.4899)

    df = pd.read_csv(data_path)
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column '{timestamp_col}' in {data_path}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)

    target_registries = load_target_registries(model_path)
    registry = target_registries if target_registries else ModelRegistry.load(model_path)

    price_col = passed.get("thermal_price_col")
    if not price_col:
        for candidate in (
            "sensor.current_electricity_market_price",
            "current_electricity_market_price",
            "day_ahead_price",
            "electricity_price",
        ):
            if candidate in df.columns:
                price_col = candidate
                break
    price_series = None
    if price_col and price_col in df.columns:
        price_series = df[price_col]

    gas_price_method = (
        passed.get("thermal_gas_price_forecast_method")
        or optim_conf.get("thermal_gas_price_forecast_method")
        or "constant"
    )
    gas_price_col = passed.get("thermal_gas_price_col") or optim_conf.get("thermal_gas_price_col")
    if not gas_price_col:
        for candidate in (
            "gas_price",
            "thermal_gas_price",
            "current_gas_price",
        ):
            if candidate in df.columns:
                gas_price_col = candidate
                break

    gas_price_series = None
    if str(gas_price_method).strip().lower() == "csv" and gas_price_col and gas_price_col in df.columns:
        gas_price_series = df[gas_price_col]

    two_stage = build_two_stage_optimization_plan(
        df,
        registry,
        coarse_models=coarse_models,
        fine_models=fine_models,
        top_k=top_k,
        horizon=horizon,
        latitude=lat,
        longitude=lon,
        target_col=target_col,
        outdoor_temp_col=outdoor_col,
        price_forecast=price_series,
        target_room_temp_min=target_room_temp_min,
        target_room_temp_max=target_room_temp_max,
        price_weight=price_weight,
        comfort_weight=comfort_weight,
        energy_weight=energy_weight,
    )
    best_plan = two_stage["best_plan"]
    forecast = best_plan["forecast"]
    neutral = best_plan["neutral"]
    price_aware = best_plan.get("price_aware")

    out = pd.DataFrame(index=forecast["index"])
    out["selected_model"] = two_stage["best_model"]
    out["predicted_temp_heater0"] = np.asarray(forecast["predicted_room_temp"], dtype=float)
    if forecast.get("actual_room_temp") is not None:
        out["actual_room_temp"] = np.asarray(forecast["actual_room_temp"], dtype=float)
    if forecast.get("predicted_electric_power") is not None:
        out["predicted_electric_power"] = np.asarray(forecast["predicted_electric_power"], dtype=float)
    if forecast.get("actual_electric_power") is not None:
        out["actual_electric_power"] = np.asarray(forecast["actual_electric_power"], dtype=float)
    if forecast.get("predicted_gas_consumption") is not None:
        out["predicted_gas_consumption"] = np.asarray(forecast["predicted_gas_consumption"], dtype=float)
    if forecast.get("actual_gas_consumption") is not None:
        out["actual_gas_consumption"] = np.asarray(forecast["actual_gas_consumption"], dtype=float)
    out["outdoor_temp"] = np.asarray(forecast["outdoor_temp"], dtype=float)
    out["baseline_curve"] = np.asarray(neutral["baseline_curve"], dtype=float)
    out["setpoint_min"] = np.asarray(neutral["setpoint_min"], dtype=float)
    out["setpoint_max"] = np.asarray(neutral["setpoint_max"], dtype=float)
    # When a day-ahead price series is available, expose the price-aware setpoint
    # as operational setpoint_optimal while keeping neutral profile for reference.
    if price_aware is not None and "setpoint_price_aware" in price_aware:
        out["setpoint_optimal"] = np.asarray(price_aware["setpoint_price_aware"], dtype=float)
        out["setpoint_neutral"] = np.asarray(neutral["setpoint_optimal"], dtype=float)
    else:
        out["setpoint_optimal"] = np.asarray(neutral["setpoint_optimal"], dtype=float)
    out["cv_estimated_electricity_kwh"] = np.asarray(
        neutral["cv_estimated_electricity_kwh"],
        dtype=float,
    )
    out["cv_estimated_gas_kwh"] = np.asarray(neutral["cv_estimated_gas_kwh"], dtype=float)

    # Energy cost view for diagnostics/reporting.
    # Electricity uses day-ahead price series when available.
    if price_series is not None:
        elec_price = price_series.reindex(out.index).ffill().bfill()
        out["electricity_price"] = np.asarray(elec_price, dtype=float)
    else:
        out["electricity_price"] = np.nan

    if gas_price_series is not None:
        gas_price = gas_price_series.reindex(out.index).ffill().bfill()
        out["gas_price"] = np.asarray(gas_price, dtype=float)
    else:
        gas_price = float(
            passed.get("thermal_gas_price")
            or optim_conf.get("thermal_gas_price")
            or 1.40
        )
        out["gas_price"] = gas_price
    out["cv_estimated_electricity_cost"] = out["cv_estimated_electricity_kwh"] * out["electricity_price"]
    out["cv_estimated_gas_cost"] = out["cv_estimated_gas_kwh"] * out["gas_price"]
    out["cv_estimated_total_cost"] = out["cv_estimated_electricity_cost"].fillna(0.0) + out["cv_estimated_gas_cost"]
    if price_aware is not None and "setpoint_price_aware" in price_aware:
        out["setpoint_price_aware"] = np.asarray(price_aware["setpoint_price_aware"], dtype=float)

    out["optim_status"] = "Two-stage thermal plan"

    if save_data_to_file:
        out.to_csv(
            input_data_dict["emhass_conf"]["data_path"] / default_csv_filename,
            index_label="timestamp",
        )

    logger.info(
        "Two-stage thermal plan generated using model=%s horizon=%d",
        two_stage["best_model"],
        len(out),
    )
    return out


async def export_influxdb_to_csv(
    input_data_dict: dict | None,
    logger: logging.Logger,
    emhass_conf: dict | None = None,
    params: str | None = None,
    runtimeparams: str | None = None,
) -> bool:
    """Export data from InfluxDB to CSV file.

    This function can be called in two ways:
    1. With input_data_dict (from web_server via set_input_data_dict)
    2. Without input_data_dict (direct call from command line or web_server before set_input_data_dict)

    :param input_data_dict: Dictionary containing configuration and parameters (optional)
    :type input_data_dict: dict | None
    :param logger: Logger object
    :type logger: logging.Logger
    :param emhass_conf: Dictionary containing EMHASS configuration paths (used when input_data_dict is None)
    :type emhass_conf: dict | None
    :param params: JSON string of params (used when input_data_dict is None)
    :type params: str | None
    :param runtimeparams: JSON string of runtime parameters (used when input_data_dict is None)
    :type runtimeparams: str | None
    :return: Success status
    :rtype: bool
    """
    # Handle two calling modes
    if input_data_dict is None:
        # Direct mode: parse params and create RetrieveHass
        if emhass_conf is None or params is None:
            logger.error("emhass_conf and params are required when input_data_dict is None")
            return False
        # Parse params
        if isinstance(params, str):
            params = orjson.loads(params)
        if isinstance(runtimeparams, str):
            runtimeparams = orjson.loads(runtimeparams)
        # Get configuration
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        if isinstance(retrieve_hass_conf, bool):
            return False
        # Treat runtime params
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8") if runtimeparams else "{}",
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "export-influxdb-to-csv",
            logger,
            emhass_conf,
        )
        # Parse params again if it's a string
        if isinstance(params, str):
            params = orjson.loads(params)
        # Create RetrieveHass object
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        time_zone = rh.time_zone
        data_path = emhass_conf["data_path"]
    else:
        # Standard mode: use input_data_dict
        params = input_data_dict["params"]
        if isinstance(params, str):
            params = orjson.loads(params)
        rh = input_data_dict["rh"]
        time_zone = rh.time_zone
        data_path = input_data_dict["emhass_conf"]["data_path"]
    # Extract parameters from passed_data
    if "sensor_list" not in params.get("passed_data", {}):
        logger.error("parameter: 'sensor_list' not passed")
        return False
    sensor_list = params["passed_data"]["sensor_list"]
    if "csv_filename" not in params.get("passed_data", {}):
        logger.error("parameter: 'csv_filename' not passed")
        return False
    csv_filename = params["passed_data"]["csv_filename"]
    if "start_time" not in params.get("passed_data", {}):
        logger.error("parameter: 'start_time' not passed")
        return False
    start_time = params["passed_data"]["start_time"]
    # Optional parameters with defaults
    end_time = params["passed_data"].get("end_time", None)
    resample_freq = params["passed_data"].get("resample_freq", "1h")
    timestamp_col = params["passed_data"].get("timestamp_col_name", "timestamp")
    decimal_places = params["passed_data"].get("decimal_places", 2)
    handle_nan = params["passed_data"].get("handle_nan", "keep")
    # Check if InfluxDB is enabled
    if not rh.use_influxdb:
        logger.error(
            "InfluxDB is not enabled in configuration. Set use_influxdb: true in config.json"
        )
        return False
    # Parse time range
    start_dt, end_dt = utils.parse_export_time_range(start_time, end_time, time_zone, logger)
    if start_dt is False:
        return False
    # Create days list for data retrieval
    days_list = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D", tz=time_zone)
    if len(days_list) == 0:
        logger.error("No days to retrieve. Check start_time and end_time.")
        return False
    logger.info(
        f"Retrieving {len(sensor_list)} sensors from {start_dt} to {end_dt} ({len(days_list)} days)"
    )
    logger.info(f"Sensors: {sensor_list}")
    # Retrieve data from InfluxDB
    success = rh.get_data(days_list, sensor_list)
    if not success or rh.df_final is None or rh.df_final.empty:
        logger.error("Failed to retrieve data from InfluxDB")
        return False
    # Filter and resample data
    df_export = utils.resample_and_filter_data(rh.df_final, start_dt, end_dt, resample_freq, logger)
    if df_export is False:
        return False
    # Reset index to make timestamp a column
    # Handle custom index names by renaming the index first
    df_export = df_export.rename_axis(timestamp_col).reset_index()
    # Clean column names
    df_export = utils.clean_sensor_column_names(df_export, timestamp_col)
    # Handle NaN values
    df_export = utils.handle_nan_values(df_export, handle_nan, timestamp_col, logger)
    # Round numeric columns to specified decimal places
    numeric_cols = df_export.select_dtypes(include=[np.number]).columns
    df_export[numeric_cols] = df_export[numeric_cols].round(decimal_places)
    # Save to CSV
    csv_path = pathlib.Path(data_path) / csv_filename
    df_export.to_csv(csv_path, index=False)
    logger.info(f"✓ Successfully exported to {csv_filename}")
    logger.info(f"  Rows: {df_export.shape[0]}")
    logger.info(f"  Columns: {list(df_export.columns)}")
    logger.info(
        f"  Time range: {df_export[timestamp_col].min()} to {df_export[timestamp_col].max()}"
    )
    logger.info(f"  File location: {csv_path}")
    return True


def _get_params(input_data_dict: dict) -> dict:
    """Helper to extract params from input_data_dict."""
    if input_data_dict:
        if not isinstance(input_data_dict.get("params", {}), dict):
            return orjson.loads(input_data_dict["params"])
        return input_data_dict.get("params", {})
    return {}


async def _publish_from_saved_entities(
    input_data_dict: dict, logger: logging.Logger, params: dict
) -> pd.DataFrame | None:
    """
    Helper to publish data from saved entity JSON files if publish_prefix is set.
    Returns DataFrame if successful, None if fallback to CSV is needed.
    """
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    entity_path = input_data_dict["emhass_conf"]["data_path"] / "entities"
    if not entity_path.exists() or not os.listdir(entity_path):
        logger.warning(f"No saved entity json files in path: {entity_path}")
        logger.warning("Falling back to opt_res_latest")
        return None
    entity_path_contents = os.listdir(entity_path)
    matches_prefix = any(publish_prefix in entity for entity in entity_path_contents)
    if not (matches_prefix or publish_prefix == "all"):
        logger.warning(f"No saved entity json files that match prefix: {publish_prefix}")
        logger.warning("Falling back to opt_res_latest")
        return None
    opt_res_list = []
    opt_res_list_names = []
    for entity in entity_path_contents:
        if entity == default_metadata_json:
            continue
        if publish_prefix == "all" or publish_prefix in entity:
            entity_data = await publish_json(entity, input_data_dict, entity_path, logger)
            if isinstance(entity_data, bool):
                return None  # Error occurred
            opt_res_list.append(entity_data)
            opt_res_list_names.append(entity.replace(".json", ""))
    opt_res = pd.concat(opt_res_list, axis=1)
    opt_res.columns = opt_res_list_names
    return opt_res


def _load_opt_res_latest(
    input_data_dict: dict, logger: logging.Logger, save_data_to_file: bool
) -> pd.DataFrame | None:
    """Helper to load the optimization results DataFrame from CSV."""
    if save_data_to_file:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        filename = "opt_res_dayahead_" + today.strftime("%Y_%m_%d") + ".csv"
    else:
        filename = default_csv_filename
    file_path = input_data_dict["emhass_conf"]["data_path"] / filename
    if not file_path.exists():
        logger.error("File not found error, run an optimization task first.")
        return None
    opt_res_latest = pd.read_csv(file_path, index_col="timestamp")
    opt_res_latest.index = pd.to_datetime(opt_res_latest.index, utc=True).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    # Infer the index frequency from the saved data itself rather than asserting
    # the current request's optimization_time_step onto it (#976): the CSV may
    # have been written by a run with a different runtime optimization_time_step,
    # and pandas raises on the mismatch. The publish path only needs the
    # timestamps for nearest-index matching. Frames with fewer than 2 rows carry
    # no inferable spacing, so leave their freq unset.
    if len(opt_res_latest.index) > 1:
        opt_res_latest = utils.set_df_index_freq(opt_res_latest)
    return opt_res_latest


def _get_closest_index(retrieve_hass_conf: dict, index: pd.DatetimeIndex) -> int:
    """Helper to find the closest index in the DataFrame to the current time."""
    now_precise = datetime.now(retrieve_hass_conf["time_zone"]).replace(second=0, microsecond=0)
    now_ts = pd.Timestamp(now_precise)
    if index.tz is None and now_ts.tz is not None:
        now_ts = now_ts.tz_localize(None)
    elif index.tz is not None and now_ts.tz is None:
        now_ts = now_ts.tz_localize(index.tz)
    method = retrieve_hass_conf.get("method_ts_round", "nearest")
    if method == "nearest":
        return index.get_indexer([now_ts], method="nearest")[0]
    elif method == "first":
        return index.get_indexer([now_ts], method="ffill")[0]
    elif method == "last":
        return index.get_indexer([now_ts], method="bfill")[0]
    return index.get_indexer([now_ts], method="nearest")[0]


async def _publish_standard_forecasts(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish PV, Load, Curtailment, and Hybrid Inverter data."""
    cols = []
    # Load Forecast
    custom_load = ctx.params["passed_data"]["custom_load_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_Load"],
        ctx.idx,
        custom_load["entity_id"],
        "power",
        custom_load["unit_of_measurement"],
        custom_load["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_Load")
    # PV Forecast
    if "P_PV" in opt_res_latest.columns:
        custom_pv = ctx.params["passed_data"]["custom_pv_forecast_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_PV"],
            ctx.idx,
            custom_pv["entity_id"],
            "power",
            custom_pv["unit_of_measurement"],
            custom_pv["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_PV")
    # Curtailment
    if ctx.plant_conf["compute_curtailment"]:
        custom_curt = ctx.params["passed_data"]["custom_pv_curtailment_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_PV_curtailment"],
            ctx.idx,
            custom_curt["entity_id"],
            "power",
            custom_curt["unit_of_measurement"],
            custom_curt["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_PV_curtailment")
    # Hybrid Inverter
    if ctx.plant_conf["inverter_is_hybrid"]:
        custom_inv = ctx.params["passed_data"]["custom_hybrid_inverter_id"]
        await ctx.rh.post_data(
            opt_res_latest["P_hybrid_inverter"],
            ctx.idx,
            custom_inv["entity_id"],
            "power",
            custom_inv["unit_of_measurement"],
            custom_inv["friendly_name"],
            type_var="power",
            **ctx.common_kwargs,
        )
        cols.append("P_hybrid_inverter")
    return cols


async def _publish_deferrable_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish data for all deferrable loads."""
    cols = []
    custom_def = ctx.params["passed_data"]["custom_deferrable_forecast_id"]
    for k in range(ctx.optim_conf["number_of_deferrable_loads"]):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns:
            ctx.logger.error(f"{col_name} was not found in results DataFrame.")
            continue
        await ctx.rh.post_data(
            opt_res_latest[col_name],
            ctx.idx,
            custom_def[k]["entity_id"],
            "power",
            custom_def[k]["unit_of_measurement"],
            custom_def[k]["friendly_name"],
            type_var="deferrable",
            **ctx.common_kwargs,
        )
        cols.append(col_name)
    return cols


# Fraction of nominal power below which a deferrable load is treated as idle
# ('off') and at/above which it is treated as fully on. Scaling the bands with
# nominal_power keeps the labels meaningful for both small and large loads.
_DEFERRABLE_OFF_FRACTION = 0.01
_DEFERRABLE_ON_FRACTION = 0.99
# Floor for the 'off' band (in W) used when nominal_power is unknown/non-positive,
# so a tiny solver residual still reads as 'off'.
_DEFERRABLE_OFF_FLOOR_W = 1.0


def _deferrable_power_to_state(power: float, nominal_power: float) -> str:
    """Map a scheduled deferrable power to an interpretable command label.

    Generic and convention-free: 'off' when the load is essentially idle, 'on'
    when it runs at (near) its nominal power, and 'variable' for any modulated
    level in between. Users layer their own logic (e.g. mapping 'on' to a switch)
    on top of this label rather than re-deriving it from the raw power forecast.

    The 'off' and 'on' bands are derived from ``nominal_power`` so they scale with
    the load; when ``nominal_power`` is unknown the 'off' band falls back to a
    small fixed floor and only 'off'/'variable' can be distinguished.
    """
    if not np.isfinite(power):
        return "off"
    if nominal_power and nominal_power > 0:
        off_threshold = max(_DEFERRABLE_OFF_FRACTION * nominal_power, _DEFERRABLE_OFF_FLOOR_W)
        if power <= off_threshold:
            return "off"
        if power >= _DEFERRABLE_ON_FRACTION * nominal_power:
            return "on"
        return "variable"
    # Nominal power unknown: only the idle floor is meaningful.
    if power <= _DEFERRABLE_OFF_FLOOR_W:
        return "off"
    return "variable"


async def _publish_deferrable_states(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish one interpretable command sensor per deferrable load (opt-in).

    Gated by the ``publish_deferrable_load_states`` option (default off, so the
    zero-config behaviour is unchanged). For each deferrable load this posts a
    string-state sensor ('on'/'off'/'variable') for the current timestep plus the
    full optimized plan as a 'schedule' attribute — a generic, glue-agnostic
    command surface that automations can act on directly.
    """
    cols = []
    if not ctx.optim_conf.get("publish_deferrable_load_states", False):
        return cols
    custom_state = ctx.params["passed_data"].get("custom_deferrable_state_id")
    if not custom_state:
        return cols
    number_of_deferrable_loads = ctx.optim_conf["number_of_deferrable_loads"]
    if len(custom_state) < number_of_deferrable_loads:
        # A short custom_deferrable_state_id means some loads get no command
        # sensor. Warn once rather than silently dropping them, so a mis-sized
        # runtime override is visible instead of failing quietly.
        ctx.logger.warning(
            "custom_deferrable_state_id has %d entries but there are %d deferrable "
            "loads; command sensors will only be published for the first %d load(s).",
            len(custom_state),
            number_of_deferrable_loads,
            len(custom_state),
        )
    nominal = ctx.optim_conf.get("nominal_power_of_deferrable_loads", [])
    for k in range(number_of_deferrable_loads):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns or k >= len(custom_state):
            continue
        nominal_power = nominal[k] if k < len(nominal) else 0.0
        states = opt_res_latest[col_name].apply(
            lambda power, npow=nominal_power: _deferrable_power_to_state(power, npow)
        )
        await ctx.rh.post_data(
            states,
            ctx.idx,
            custom_state[k]["entity_id"],
            custom_state[k]["device_class"],
            custom_state[k]["unit_of_measurement"],
            custom_state[k]["friendly_name"],
            type_var="categorical",
            **ctx.common_kwargs,
        )
    # The command states are derived (not opt_res columns), so they are
    # published as a side-effect without being added to the returned column set.
    return cols


async def _publish_thermal_variable(
    rh, opt_res_latest, idx, k, custom_ids, col_prefix, type_var, unit_type, kwargs
) -> str | None:
    """Helper to publish a single thermal variable if valid."""
    if custom_ids and k < len(custom_ids):
        col_name = f"{col_prefix}{k}"
        if col_name in opt_res_latest.columns:
            entity_conf = custom_ids[k]
            await rh.post_data(
                opt_res_latest[col_name],
                idx,
                entity_conf["entity_id"],
                unit_type,
                entity_conf["unit_of_measurement"],
                entity_conf["friendly_name"],
                type_var=type_var,
                **kwargs,
            )
            return col_name
    return None


async def _publish_thermal_loads(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish predicted temperature and heating demand for thermal loads."""
    cols = []
    if "custom_predicted_temperature_id" not in ctx.params["passed_data"]:
        return cols
    custom_temp = ctx.params["passed_data"]["custom_predicted_temperature_id"]
    custom_heat = ctx.params["passed_data"].get("custom_heating_demand_id")
    def_load_config = ctx.optim_conf.get("def_load_config", [])
    if not isinstance(def_load_config, list):
        def_load_config = []
    for k in range(ctx.optim_conf["number_of_deferrable_loads"]):
        if k >= len(def_load_config):
            continue
        load_cfg = def_load_config[k]
        if "thermal_config" not in load_cfg and "thermal_battery" not in load_cfg:
            continue
        col_t = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_temp,
            "predicted_temp_heater",
            "temperature",
            "temperature",
            ctx.common_kwargs,
        )
        if col_t:
            cols.append(col_t)
        col_h = await _publish_thermal_variable(
            ctx.rh,
            opt_res_latest,
            ctx.idx,
            k,
            custom_heat,
            "heating_demand_heater",
            "energy",
            "energy",
            ctx.common_kwargs,
        )
        if col_h:
            cols.append(col_h)
    return cols


async def _publish_room_targets(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish each room's target temperature - the top of its currently
    scheduled comfort band - for a companion HA automation to apply to the
    real thermostat (e.g. via climate.set_temperature). EMHASS never calls
    the HA service directly; it only publishes this target sensor.
    """
    cols = []
    room_load_indices = ctx.params["passed_data"].get("room_load_indices", {})
    if not room_load_indices:
        return cols
    custom_target = ctx.params["passed_data"].get("custom_room_target_temp_id", [])
    def_load_config = ctx.optim_conf.get("def_load_config", [])

    for i, (name, k) in enumerate(room_load_indices.items()):
        if i >= len(custom_target) or k >= len(def_load_config):
            continue
        hc = def_load_config[k].get("thermal_battery", {}) if isinstance(def_load_config[k], dict) else {}
        max_temps = hc.get("max_temperatures", [])
        if not max_temps:
            continue
        n = min(len(max_temps), len(opt_res_latest))
        if n == 0:
            continue
        target_series = pd.Series(max_temps[:n], index=opt_res_latest.index[:n])
        col_name = f"room_target_temp_{name}"
        opt_res_latest[col_name] = target_series
        entity_conf = custom_target[i]
        idx = min(ctx.idx, n - 1)
        await ctx.rh.post_data(
            target_series,
            idx,
            entity_conf["entity_id"],
            entity_conf["device_class"],
            entity_conf["unit_of_measurement"],
            entity_conf["friendly_name"],
            type_var="temperature",
            **ctx.common_kwargs,
        )
        cols.append(col_name)
    return cols


async def _publish_heatpump_dispatch_target(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish the whole-house heat pump's target on/off dispatch state for a
    companion HA automation to apply via switch.turn_on/switch.turn_off on
    the configured heatpump_dispatch_control_entity (e.g. switch.climate_control).
    EMHASS never calls the HA service directly; it only publishes this target sensor.
    """
    cols = []
    dispatch_k = ctx.params["passed_data"].get("heatpump_dispatch_load_index")
    if dispatch_k is None:
        return cols
    power_col_name = f"P_deferrable{dispatch_k}"
    if power_col_name not in opt_res_latest.columns:
        return cols
    entity_conf = ctx.params["passed_data"].get("custom_heatpump_dispatch_target_id")
    if not entity_conf:
        return cols
    state_series = opt_res_latest[power_col_name].apply(lambda p: "on" if float(p) > 1.0 else "off")
    opt_res_latest["heatpump_dispatch_target"] = state_series
    await ctx.rh.post_data(
        state_series,
        ctx.idx,
        entity_conf["entity_id"],
        entity_conf["device_class"],
        entity_conf["unit_of_measurement"],
        entity_conf["friendly_name"],
        type_var="heatpump_dispatch",
        **ctx.common_kwargs,
    )
    cols.append("heatpump_dispatch_target")
    return cols


def _translate_ev_power_to_mode(
    power_w: float, min_1p: float, max_1p: float, min_3p: float, max_3p: float
) -> tuple[str, str]:
    """Heuristic translation from the optimizer's continuous EV charging
    power to one of myenergi's 4 discrete charge modes + phase setting.

    This is an explicit approximation: the myenergi Zappi (and similar
    EVSEs) has no continuous-power service, only discrete
    Stopped/Eco/Eco+/Fast modes plus a 1/3-phase select, so any mapping from
    a continuous MILP power output onto those modes is inherently lossy.
    """
    if power_w < min_1p / 2:
        return "stopped", "1_phase"
    if power_w >= max_3p:
        return "fast", "3_phase"
    midpoint = (min_1p + max_1p) / 2
    if power_w >= midpoint:
        return "eco_plus", ("1_phase" if power_w <= max_1p else "3_phase")
    return "eco", "1_phase"


def _ev_conf_value(optim_conf: dict, key: str, i: int, default: float) -> float:
    values = optim_conf.get(key, [])
    try:
        return float(values[i]) if i < len(values) else default
    except (TypeError, ValueError):
        return default


def _ev_display_value(optim_conf: dict, key: str, i: int, default: str) -> str:
    values = optim_conf.get(key, [])
    if i < len(values) and values[i]:
        return str(values[i])
    return default


async def _publish_ev_targets(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish each EV charger's target charge mode and phase, translated
    from the optimizer's continuous power output via _translate_ev_power_to_mode.
    EMHASS never calls the myenergi service directly; a companion HA
    automation reads these sensors and calls select.select_option.
    """
    cols = []
    ev_load_indices = ctx.params["passed_data"].get("ev_load_indices", {})
    if not ev_load_indices:
        return cols
    custom_mode = ctx.params["passed_data"].get("custom_ev_charge_mode_target_id", [])
    custom_phase = ctx.params["passed_data"].get("custom_ev_phase_target_id", [])
    mode_default_labels = {"stopped": "Stopped", "fast": "Fast", "eco": "Eco", "eco_plus": "Eco+"}
    mode_config_keys = {
        "stopped": "ev_charge_mode_stopped_value",
        "fast": "ev_charge_mode_fast_value",
        "eco": "ev_charge_mode_eco_value",
        "eco_plus": "ev_charge_mode_ecoplus_value",
    }
    phase_default_labels = {"1_phase": "1_phase", "3_phase": "3_phase"}
    phase_config_keys = {
        "1_phase": "ev_phase_select_value_1_phase",
        "3_phase": "ev_phase_select_value_3_phase",
    }

    for i, (name, k) in enumerate(ev_load_indices.items()):
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns:
            continue
        min_1p = _ev_conf_value(ctx.optim_conf, "ev_charge_power_min_1_phase", i, 1380.0)
        max_1p = _ev_conf_value(ctx.optim_conf, "ev_charge_power_max_1_phase", i, 3680.0)
        min_3p = _ev_conf_value(ctx.optim_conf, "ev_charge_power_min_3_phase", i, 4140.0)
        max_3p = _ev_conf_value(ctx.optim_conf, "ev_charge_power_max_3_phase", i, 11000.0)

        modes = []
        phases = []
        for power_w in opt_res_latest[col_name]:
            mode, phase = _translate_ev_power_to_mode(
                float(power_w), min_1p, max_1p, min_3p, max_3p
            )
            modes.append(
                _ev_display_value(
                    ctx.optim_conf, mode_config_keys[mode], i, mode_default_labels[mode]
                )
            )
            phases.append(
                _ev_display_value(
                    ctx.optim_conf, phase_config_keys[phase], i, phase_default_labels[phase]
                )
            )
        mode_series = pd.Series(modes, index=opt_res_latest.index)
        phase_series = pd.Series(phases, index=opt_res_latest.index)
        mode_col_name = f"ev_charge_mode_target_{name}"
        phase_col_name = f"ev_phase_target_{name}"
        opt_res_latest[mode_col_name] = mode_series
        opt_res_latest[phase_col_name] = phase_series

        if i < len(custom_mode):
            entity_conf = custom_mode[i]
            await ctx.rh.post_data(
                mode_series,
                ctx.idx,
                entity_conf["entity_id"],
                entity_conf["device_class"],
                entity_conf["unit_of_measurement"],
                entity_conf["friendly_name"],
                type_var="ev_charge_mode",
                **ctx.common_kwargs,
            )
            cols.append(mode_col_name)
        if i < len(custom_phase):
            entity_conf = custom_phase[i]
            await ctx.rh.post_data(
                phase_series,
                ctx.idx,
                entity_conf["entity_id"],
                entity_conf["device_class"],
                entity_conf["unit_of_measurement"],
                entity_conf["friendly_name"],
                type_var="ev_phase_target",
                **ctx.common_kwargs,
            )
            cols.append(phase_col_name)
    return cols


async def _maybe_record_manual_load_commitments(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> dict:
    """Trust-the-plan write-back for manually-committed loads (see
    manual_load_enabled): the first time a ready appliance has no persisted
    commitment yet, read the just-solved plan's chosen start for that load
    and persist it to data/manual_load_commitments.json. Once a commitment
    exists it is never overwritten here - only
    _apply_manual_load_runtime_overrides clears it (on confirmation or
    deadline elapse), which is what keeps a shown-to-the-user plan from
    moving on a later re-optimization.

    Returns the (possibly updated) commitments dict so
    _publish_manual_load_actions doesn't have to re-read the file.
    """
    manual_load_indices = ctx.params.get("passed_data", {}).get("manual_load_indices", {})
    if not manual_load_indices:
        return {}

    commitments = await load_json_blob(
        ctx.emhass_conf, "manual_load_commitments.json", ctx.logger, default={}
    )
    if not isinstance(commitments, dict):
        commitments = {}
    changed = False

    for name, load_info in manual_load_indices.items():
        if isinstance(commitments.get(name), dict) and commitments[name].get("committed_start_iso"):
            continue  # already committed - the whole point is to never move it
        k = load_info["k"]
        col_name = f"P_deferrable{k}"
        if col_name not in opt_res_latest.columns:
            continue
        nominal_power = float(load_info.get("nominal_power", 0.0) or 0.0)
        active_threshold = max(0.1 * nominal_power, 20.0)
        active = opt_res_latest[col_name][opt_res_latest[col_name] >= active_threshold]
        if active.empty:
            continue  # solver didn't schedule it this cycle (e.g. not ready yet)
        start_time = active.index[0]
        if not isinstance(start_time, pd.Timestamp):
            continue
        start_time = (
            start_time.tz_localize(UTC) if start_time.tzinfo is None else start_time.tz_convert(UTC)
        )
        commitments[name] = {
            "committed_start_iso": start_time.isoformat(),
            "created_at_iso": pd.Timestamp.now(tz=UTC).isoformat(),
        }
        changed = True
        ctx.logger.info(
            "Manual load '%s' committed to start at %s", name, commitments[name]["committed_start_iso"]
        )

    if changed:
        await save_json_blob(ctx.emhass_conf, "manual_load_commitments.json", commitments, ctx.logger)
    return commitments


def _format_manual_load_action(committed_start: pd.Timestamp | None, now: pd.Timestamp) -> str:
    """Human-readable instruction for the manual-load action sensor - the
    user asked to see how to set the appliance's physical delay-start timer,
    not a raw timestamp.
    """
    if committed_start is None:
        return "waiting"
    remaining = committed_start - now
    total_minutes = int(remaining.total_seconds() // 60)
    if total_minutes <= 0:
        return "Start now"
    hours, minutes = divmod(total_minutes, 60)
    if hours > 0:
        return f"Set timer to {hours}h {minutes}m"
    return f"Set timer to {minutes}m"


async def _publish_manual_load_actions(ctx: PublishContext, commitments: dict) -> None:
    """Publish each manual load's human-readable timer instruction
    (sensor.manual_load_action_<name>): "waiting" with nothing committed yet,
    "Set timer to Xh Ym" once a start has been committed, "Start now" once
    that committed time has arrived. Derived state, not an opt_res column -
    published as a side-effect, same as _publish_deferrable_states.
    """
    manual_load_indices = ctx.params.get("passed_data", {}).get("manual_load_indices", {})
    if not manual_load_indices:
        return
    custom_action = ctx.params.get("passed_data", {}).get("custom_manual_load_action_id", [])
    now = pd.Timestamp.now(tz=UTC)

    for i, name in enumerate(manual_load_indices.keys()):
        if i >= len(custom_action):
            continue
        commitment = commitments.get(name)
        committed_start = None
        if isinstance(commitment, dict) and commitment.get("committed_start_iso"):
            try:
                committed_start = pd.Timestamp(commitment["committed_start_iso"])
                committed_start = (
                    committed_start.tz_localize(UTC)
                    if committed_start.tzinfo is None
                    else committed_start.tz_convert(UTC)
                )
            except (ValueError, TypeError):
                committed_start = None
        state = _format_manual_load_action(committed_start, now)
        entity_conf = custom_action[i]
        await ctx.rh.post_data(
            pd.Series([state], index=[now]),
            0,
            entity_conf["entity_id"],
            entity_conf["device_class"],
            entity_conf["unit_of_measurement"],
            entity_conf["friendly_name"],
            type_var="categorical",
            **ctx.common_kwargs,
        )


def _has_contiguous_hold(series: pd.Series, target: float, hold_steps: int) -> bool:
    """Return True if `series` contains a run of >= hold_steps consecutive
    values that are all >= target. Mirrors the contiguous-window requirement
    enforced by the legionella constraint in Optimization._add_thermal_battery_constraints.
    """
    if hold_steps <= 0 or series.empty:
        return False
    run = 0
    for value in series.to_numpy():
        if pd.notna(value) and value >= target:
            run += 1
            if run >= hold_steps:
                return True
        else:
            run = 0
    return False


async def _maybe_record_legionella_completion(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> None:
    """Mark a boiler's legionella cycle as completed (write-back last_run_iso)
    when the just-solved plan actually achieves a contiguous hold at/above the
    legionella target temperature. This is a "trust-the-plan" write-back: it
    marks completion once the optimizer has committed to a compliant plan,
    not once a real sensor confirms the tank reached temperature.
    """
    def_load_config = ctx.optim_conf.get("def_load_config", [])
    if not isinstance(def_load_config, list):
        return
    time_step = ctx.input_data_dict["retrieve_hass_conf"]["optimization_time_step"]
    time_step_hours = time_step.total_seconds() / 3600.0
    if time_step_hours <= 0:
        return

    boiler_indices: list[int] = []
    completions: list[str] = []
    for k, load_cfg in enumerate(def_load_config):
        hc = load_cfg.get("thermal_battery") if isinstance(load_cfg, dict) else None
        if not hc or not bool(hc.get("legionella_due", False)):
            continue
        col_name = f"predicted_temp_heater{k}"
        if col_name not in opt_res_latest.columns:
            continue
        legio_target = float(hc.get("legionella_target_temperature", 60.0))
        hold_hours = float(hc.get("legionella_hold_hours", 0.0) or 0.0)
        hold_steps = max(1, ceil(hold_hours / time_step_hours))
        if _has_contiguous_hold(opt_res_latest[col_name], legio_target, hold_steps):
            boiler_indices.append(k)
            completions.append(pd.Timestamp.now(tz=UTC).isoformat())
        else:
            ctx.logger.debug(
                "Legionella cycle for load %s planned but not yet confirmed "
                "in the solved plan (no contiguous hold found).",
                k,
            )

    if not boiler_indices:
        return

    # boiler index (0-based, among boilers only) is separate from the
    # deferrable-load index k. Map back to boiler position using the append
    # order recorded by _append_boiler_thermal_battery_loads (marked
    # "_source": "boiler_auto"), not just "any thermal_battery entry" -
    # a user-defined (non-boiler) thermal_battery load could sit at a lower
    # index and would otherwise throw this mapping off.
    boiler_last_run = list(ctx.optim_conf.get("boiler_legionella_last_run_iso", []))
    def_load_boiler_indices = [
        k
        for k, load_cfg in enumerate(def_load_config)
        if isinstance(load_cfg, dict)
        and load_cfg.get("thermal_battery", {}).get("_source") == "boiler_auto"
    ]
    for k, iso_ts in zip(boiler_indices, completions, strict=False):
        if k in def_load_boiler_indices:
            boiler_pos = def_load_boiler_indices.index(k)
            if boiler_pos < len(boiler_last_run):
                boiler_last_run[boiler_pos] = iso_ts

    await save_json_blob(
        ctx.emhass_conf,
        "boiler_runtime_state.json",
        {"boiler_legionella_last_run_iso": boiler_last_run},
        ctx.logger,
    )
    ctx.logger.info(
        "Legionella cycle completion recorded for boiler load(s): %s", boiler_indices
    )


_BATTERY_POWER_ENTITY_TEMPLATE = "sensor.p_batt_forecast_battery{k}"
_BATTERY_SOC_ENTITY_TEMPLATE = "sensor.soc_batt_forecast_battery{k}"
_DEFAULT_SOC_ENTITY_ID = "sensor.soc_batt_forecast"


async def _publish_battery_data(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Battery Power (fleet total; per-battery at N>1) and SOC
    (bare at N=1, per-battery only at N>1)."""
    cols = []
    if not ctx.optim_conf["set_use_battery"]:
        return cols
    if "P_batt" not in opt_res_latest.columns:
        ctx.logger.error("P_batt was not found in results DataFrame.")
        return cols
    # Power - fleet total. custom_batt_forecast_id overrides this one entity at
    # any N: unchanged code path, so this stays a true no-op at N=1.
    custom_batt = ctx.params["passed_data"]["custom_batt_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_batt"],
        ctx.idx,
        custom_batt["entity_id"],
        "power",
        custom_batt["unit_of_measurement"],
        custom_batt["friendly_name"],
        type_var="batt",
        **ctx.common_kwargs,
    )
    cols.append("P_batt")

    n_batt = utils.validate_num_batteries(ctx.plant_conf)
    if n_batt == 1:
        # N=1: exactly today's entity set, zero new entities. Still
        # overridable via custom_batt_soc_forecast_id.
        # Guard the bare SOC_opt read the same way P_batt is guarded above: a
        # stale N>1 results frame (fleet P_batt + SOC_opt_0/1, no bare
        # SOC_opt) replayed under a config reverted to N=1 must warn+skip
        # SOC, never KeyError.
        if "SOC_opt" not in opt_res_latest.columns:
            ctx.logger.error("SOC_opt was not found in results DataFrame.")
            return cols
        custom_soc = ctx.params["passed_data"]["custom_batt_soc_forecast_id"]
        await ctx.rh.post_data(
            opt_res_latest["SOC_opt"] * 100,
            ctx.idx,
            custom_soc["entity_id"],
            "battery",
            custom_soc["unit_of_measurement"],
            custom_soc["friendly_name"],
            type_var="SOC",
            **ctx.common_kwargs,
        )
        cols.append("SOC_opt")
        return cols

    # N > 1: no bare fleet SOC sensor - SOC has no meaningful fleet aggregate.
    # custom_batt_soc_forecast_id has no natural single target here (it
    # customizes the bare sensor that no longer exists at N>1), so a runtime
    # override is ignored with a one-time warning rather than silently
    # reinterpreted as a prefix. This keeps every per-battery entity id
    # exactly the pinned sensor.*_battery<K> name regardless of a legacy
    # single-battery override, adding no new config surface.
    custom_soc = ctx.params["passed_data"].get("custom_batt_soc_forecast_id") or {}
    if custom_soc.get("entity_id") not in (None, _DEFAULT_SOC_ENTITY_ID):
        ctx.logger.warning(
            "custom_batt_soc_forecast_id override (%s) has no effect when "
            "number_of_batteries=%d: SOC has no meaningful fleet aggregate, so "
            "per-battery SOC always publishes on the fixed "
            "sensor.soc_batt_forecast_battery<K> entity ids.",
            custom_soc.get("entity_id"),
            n_batt,
        )

    for k in range(n_batt):
        p_col = f"P_batt_{k}"
        if p_col in opt_res_latest.columns:
            await ctx.rh.post_data(
                opt_res_latest[p_col],
                ctx.idx,
                _BATTERY_POWER_ENTITY_TEMPLATE.format(k=k),
                "power",
                "W",
                f"Battery Power Forecast Battery {k}",
                type_var="batt",
                **ctx.common_kwargs,
            )
            cols.append(p_col)
        else:
            ctx.logger.error(f"{p_col} was not found in results DataFrame.")

        soc_col = f"SOC_opt_{k}"
        if soc_col in opt_res_latest.columns:
            await ctx.rh.post_data(
                opt_res_latest[soc_col] * 100,
                ctx.idx,
                _BATTERY_SOC_ENTITY_TEMPLATE.format(k=k),
                "battery",
                "%",
                f"Battery SOC Forecast Battery {k}",
                type_var="SOC",
                **ctx.common_kwargs,
            )
            cols.append(soc_col)
        else:
            ctx.logger.error(f"{soc_col} was not found in results DataFrame.")
    return cols


async def _publish_grid_and_costs(ctx: PublishContext, opt_res_latest: pd.DataFrame) -> list[str]:
    """Publish Grid Power, Costs, and Optimization Status."""
    cols = []
    # Grid
    custom_grid = ctx.params["passed_data"]["custom_grid_forecast_id"]
    await ctx.rh.post_data(
        opt_res_latest["P_grid"],
        ctx.idx,
        custom_grid["entity_id"],
        "power",
        custom_grid["unit_of_measurement"],
        custom_grid["friendly_name"],
        type_var="power",
        **ctx.common_kwargs,
    )
    cols.append("P_grid")
    # Cost Function
    custom_cost = ctx.params["passed_data"]["custom_cost_fun_id"]
    col_cost_fun = [i for i in opt_res_latest.columns if "cost_fun_" in i]
    await ctx.rh.post_data(
        opt_res_latest[col_cost_fun],
        ctx.idx,
        custom_cost["entity_id"],
        "monetary",
        custom_cost["unit_of_measurement"],
        custom_cost["friendly_name"],
        type_var="cost_fun",
        **ctx.common_kwargs,
    )
    # Optim Status
    custom_status = ctx.params["passed_data"]["custom_optim_status_id"]
    if "optim_status" not in opt_res_latest:
        opt_res_latest["optim_status"] = "Optimal"
        ctx.logger.warning("no optim_status in opt_res_latest")
    status_val = opt_res_latest["optim_status"]
    await ctx.rh.post_data(
        status_val,
        ctx.idx,
        custom_status["entity_id"],
        "",
        "",
        custom_status["friendly_name"],
        type_var="optim_status",
        **ctx.common_kwargs,
    )
    cols.append("optim_status")
    # Unit Costs
    for key, var_name in [
        ("custom_unit_load_cost_id", "unit_load_cost"),
        ("custom_unit_prod_price_id", "unit_prod_price"),
    ]:
        custom_id = ctx.params["passed_data"][key]
        await ctx.rh.post_data(
            opt_res_latest[var_name],
            ctx.idx,
            custom_id["entity_id"],
            "monetary",
            custom_id["unit_of_measurement"],
            custom_id["friendly_name"],
            type_var=var_name,
            **ctx.common_kwargs,
        )
        cols.append(var_name)
    return cols


async def publish_data(
    input_data_dict: dict,
    logger: logging.Logger,
    save_data_to_file: bool | None = False,
    opt_res_latest: pd.DataFrame | None = None,
    entity_save: bool | None = False,
    dont_post: bool | None = False,
) -> pd.DataFrame:
    """
    Publish the data obtained from the optimization results.

    :param input_data_dict:  A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging object
    :param save_data_to_file: If True we will read data from optimization results in dayahead CSV file
    :type save_data_to_file: bool, optional
    :return: The output data of the optimization readed from a CSV file in the data folder
    :rtype: pd.DataFrame
    :param entity_save: Save built entities to data_path/entities
    :type entity_save: bool, optional
    :param dont_post: Do not post to Home Assistant. Works with entity_save
    :type dont_post: bool, optional

    """
    logger.info("Publishing data to HASS instance")
    # Parse Parameters
    params = _get_params(input_data_dict)
    # Check for Entity Publishing (Prefix mode)
    publish_prefix = params["passed_data"].get("publish_prefix", "")
    if not save_data_to_file and publish_prefix != "" and not dont_post:
        opt_res = await _publish_from_saved_entities(input_data_dict, logger, params)
        if opt_res is not None:
            opt_res.attrs["emhass_schema_version"] = EMHASS_SCHEMA_VERSION
            return opt_res
    # Load Optimization Results (if not passed)
    if opt_res_latest is None:
        opt_res_latest = _load_opt_res_latest(input_data_dict, logger, save_data_to_file)
        if opt_res_latest is None:
            return None
    # A failed/infeasible optimization yields a results frame with only the
    # optim_status column (see Optimization.perform_optimization); the forecast
    # and battery columns are absent. Publishing it would crash on the first
    # lookup (e.g. opt_res_latest["P_Load"]). Surface the failure instead.
    if "P_Load" not in opt_res_latest.columns:
        status = "unknown"
        if "optim_status" in opt_res_latest.columns and not opt_res_latest.empty:
            status = opt_res_latest["optim_status"].iloc[0]
        logger.error(
            "Optimization result is incomplete (status: %s); nothing to publish. "
            "Run a successful optimization before publishing.",
            status,
        )
        return None
    # Determine Closest Index
    idx_closest = _get_closest_index(input_data_dict["retrieve_hass_conf"], opt_res_latest.index)
    # Create Context
    common_kwargs = {
        "publish_prefix": publish_prefix,
        "save_entities": entity_save,
        "dont_post": dont_post,
    }
    ctx = PublishContext(
        input_data_dict=input_data_dict,
        params=params,
        idx=idx_closest,
        common_kwargs=common_kwargs,
        logger=logger,
    )
    # Publish Data Components
    cols_published = []
    cols_published.extend(await _publish_standard_forecasts(ctx, opt_res_latest))
    cols_published.extend(await _publish_deferrable_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_deferrable_states(ctx, opt_res_latest))
    cols_published.extend(await _publish_thermal_loads(ctx, opt_res_latest))
    cols_published.extend(await _publish_room_targets(ctx, opt_res_latest))
    cols_published.extend(await _publish_heatpump_dispatch_target(ctx, opt_res_latest))
    cols_published.extend(await _publish_ev_targets(ctx, opt_res_latest))
    await _maybe_record_legionella_completion(ctx, opt_res_latest)
    manual_load_commitments = await _maybe_record_manual_load_commitments(ctx, opt_res_latest)
    await _publish_manual_load_actions(ctx, manual_load_commitments)
    cols_published.extend(await _publish_battery_data(ctx, opt_res_latest))
    cols_published.extend(await _publish_grid_and_costs(ctx, opt_res_latest))
    # Return Summary DataFrame
    opt_res = opt_res_latest[cols_published].loc[[opt_res_latest.index[idx_closest]]]
    opt_res.attrs["emhass_schema_version"] = EMHASS_SCHEMA_VERSION
    return opt_res


async def continual_publish(
    input_data_dict: dict, entity_path: pathlib.Path, logger: logging.Logger
):
    """
    If continual_publish is true and a entity file saved in /data_path/entities, continually publish sensor on freq rate, updating entity current state value based on timestamp

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    """
    logger.info("Continual publish thread service started")
    freq = input_data_dict["retrieve_hass_conf"].get(
        "optimization_time_step", pd.to_timedelta(1, "minutes")
    )
    while True:
        # Sleep for x seconds (using current time as a reference for time left)
        time_zone = input_data_dict["retrieve_hass_conf"]["time_zone"]
        timestamp_diff = freq.total_seconds() - (datetime.now(time_zone).timestamp() % 60)
        sleep_seconds = max(0.0, min(timestamp_diff, 60.0))
        await asyncio.sleep(sleep_seconds)
        # Delegate processing to helper function to reduce complexity. A
        # transient failure in a single cycle (e.g. a half-written entity file
        # read mid-publish) must not kill the background task: log it and retry
        # on the next interval, otherwise published sensors freeze until restart.
        try:
            freq = await _publish_and_update_freq(input_data_dict, entity_path, logger, freq)
        except asyncio.CancelledError:
            # Task cancellation (e.g. shutdown) must propagate, never be retried.
            # CancelledError is a BaseException so the broad except below would
            # not catch it anyway; this makes that intent explicit.
            raise
        except Exception:
            logger.exception("continual_publish cycle failed; retrying next interval")
    return False


async def _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq):
    """
    Helper to process entity publishing and frequency updates.
    Returns the (potentially updated) frequency.
    """
    # Guard clause: if path doesn't exist, do nothing and return current freq
    if not os.path.exists(entity_path):
        return current_freq
    entity_path_contents = os.listdir(entity_path)
    # Guard clause: if directory is empty, do nothing
    if not entity_path_contents:
        return current_freq
    # Loop through all saved entity files
    for entity in entity_path_contents:
        # Skip metadata and any in-flight atomic-write temp files: entity and
        # metadata files are committed via a "<name>.<...>.tmp" + os.replace, and
        # that temp file can be momentarily visible to this directory listing.
        if entity != default_metadata_json and not entity.endswith(".tmp"):
            await publish_json(
                entity,
                input_data_dict,
                entity_path,
                logger,
                "continual_publish",
            )
    # Retrieve entity metadata from file
    metadata_file = entity_path / default_metadata_json
    if os.path.isfile(metadata_file):
        async with aiofiles.open(metadata_file) as file:
            content = await file.read()
            metadata = orjson.loads(content)
            # Check if freq should be shorter
            if metadata.get("lowest_time_step") is not None:
                return pd.to_timedelta(metadata["lowest_time_step"], "minutes")
    return current_freq


async def publish_json(
    entity: dict,
    input_data_dict: dict,
    entity_path: pathlib.Path,
    logger: logging.Logger,
    reference: str | None = "",
):
    """
    Extract saved entity data from .json (in data_path/entities), build entity, post results to post_data

    :param entity: json file containing entity data
    :type entity: dict
    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param entity_path: Path for entities folder in data_path
    :type entity_path: Path
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param reference: String for identifying who ran the function
    :type reference: str, optional

    """
    # Retrieve entity metadata from file
    if os.path.isfile(entity_path / default_metadata_json):
        async with aiofiles.open(entity_path / default_metadata_json) as file:
            content = await file.read()
            metadata = orjson.loads(content)
    else:
        logger.error(f"unable to locate metadata.json in: {entity_path}")
        return False
    # Round current timecode (now)
    now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
        second=0, microsecond=0
    )
    # Retrieve entity data from file
    entity_data = pd.read_json(entity_path / entity, orient="index")
    # Remove ".json" from string for entity_id
    entity_id = entity.replace(".json", "")
    # Adjust Dataframe from received entity json file
    entity_data.columns = [metadata[entity_id]["name"]]
    entity_data.index.name = "timestamp"
    entity_data.index = pd.to_datetime(entity_data.index).tz_convert(
        input_data_dict["retrieve_hass_conf"]["time_zone"]
    )
    entity_data.index.freq = pd.to_timedelta(
        int(metadata[entity_id]["optimization_time_step"]), "minutes"
    )
    # Calculate the current state value
    if input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "nearest":
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "first":
        idx_closest = entity_data.index.get_indexer([now_precise], method="ffill")[0]
    elif input_data_dict["retrieve_hass_conf"]["method_ts_round"] == "last":
        idx_closest = entity_data.index.get_indexer([now_precise], method="bfill")[0]
    if idx_closest == -1:
        idx_closest = entity_data.index.get_indexer([now_precise], method="nearest")[0]
    # Call post data
    if reference == "continual_publish":
        logger.debug("Auto Published sensor:")
        logger_levels = "DEBUG"
    else:
        logger_levels = "INFO"
    # post/save entity
    await input_data_dict["rh"].post_data(
        data_df=entity_data[metadata[entity_id]["name"]],
        idx=idx_closest,
        entity_id=entity_id,
        device_class=dict.get(metadata[entity_id], "device_class"),
        unit_of_measurement=metadata[entity_id]["unit_of_measurement"],
        friendly_name=metadata[entity_id]["friendly_name"],
        type_var=metadata[entity_id].get("type_var", ""),
        save_entities=False,
        logger_levels=logger_levels,
    )
    return entity_data[metadata[entity_id]["name"]]


async def main():
    r"""Define the main command line entry function.

    This function may take several arguments as inputs. You can type `emhass --help` to see the list of options:

    - action: Set the desired action, options are: perfect-optim, dayahead-optim,
      naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune,
      forecast-calibration

    - config: Define path to the config.yaml file

    - costfun: Define the type of cost function, options are: profit, cost, self-consumption

    - log2file: Define if we should log to a file or not

    - params: Configuration parameters passed from data/options.json if using the add-on

    - runtimeparams: Pass runtime optimization parameters as dictionnary

    - debug: Use True for testing purposes

    """
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        help="Set the desired action, options are: perfect-optim, dayahead-optim,\
        naive-mpc-optim, publish-data, forecast-model-fit, forecast-model-predict, forecast-model-tune,\
        forecast-calibration, heating-need-forecast, heating-model-refit",
    )
    parser.add_argument(
        "--config", type=str, help="Define path to the config.json/defaults.json file"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="String of configuration parameters passed",
    )
    parser.add_argument("--data", type=str, help="Define path to the Data files (.csv & .pkl)")
    parser.add_argument("--root", type=str, help="Define path emhass root")
    parser.add_argument(
        "--costfun",
        type=str,
        default="profit",
        help="Define the type of cost function, options are: profit, cost, self-consumption",
    )
    parser.add_argument(
        "--log2file",
        type=bool,
        default=False,
        help="Define if we should log to a file or not",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        default=None,
        help="Define secret parameter file (secrets_emhass.yaml) path",
    )
    parser.add_argument(
        "--runtimeparams",
        type=str,
        default=None,
        help="Pass runtime optimization parameters as dictionnary",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Use True for testing purposes",
    )
    args = parser.parse_args()

    # The path to the configuration files
    if args.config is not None:
        config_path = pathlib.Path(args.config)
    else:
        config_path = pathlib.Path(str(utils.get_root(__file__, num_parent=3) / "config.json"))
    if args.data is not None:
        data_path = pathlib.Path(args.data)
    else:
        data_path = config_path.parent / "data/"
    if args.root is not None:
        root_path = pathlib.Path(args.root)
    else:
        root_path = utils.get_root(__file__, num_parent=1)
    if args.secrets is not None:
        secrets_path = pathlib.Path(args.secrets)
    else:
        secrets_path = pathlib.Path(config_path.parent / "secrets_emhass.yaml")

    associations_path = root_path / "data/associations.csv"
    defaults_path = root_path / "data/config_defaults.json"

    emhass_conf = {}
    emhass_conf["config_path"] = config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["defaults_path"] = defaults_path
    # create logger
    logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=bool(args.log2file))

    # Check paths
    logger.debug("config path: " + str(config_path))
    logger.debug("data path: " + str(data_path))
    logger.debug("root path: " + str(root_path))
    if not associations_path.exists():
        logger.error("Could not find associations.csv file in: " + str(associations_path))
        logger.error("Try setting config file path with --associations")
        return False
    if not config_path.exists():
        logger.warning("Could not find config.json file in: " + str(config_path))
        logger.warning("Try setting config file path with --config")
    if not secrets_path.exists():
        logger.warning("Could not find secrets file in: " + str(secrets_path))
        logger.warning("Try setting secrets file path with --secrets")
    if not os.path.isdir(data_path):
        logger.error("Could not find data folder in: " + str(data_path))
        logger.error("Try setting data path with --data")
        return False
    if not os.path.isdir(root_path):
        logger.error("Could not find emhass/src folder in: " + str(root_path))
        logger.error("Try setting emhass root path with --root")
        return False

    # Additional argument
    try:
        parser.add_argument(
            "--version",
            action="version",
            version="%(prog)s " + version("emhass"),
        )
        args = parser.parse_args()
    except Exception:
        logger.info(
            "Version not found for emhass package. Or importlib exited with PackageNotFoundError.",
        )

    # Setup config
    config = {}
    # Check if passed config file is yaml of json, build config accordingly
    if config_path.exists():
        # Safe: Use pathlib's suffix instead of regex to avoid ReDoS
        file_extension = config_path.suffix.lstrip(".").lower()

        if file_extension:
            match file_extension:
                case "json":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path
                    )
                case "yaml" | "yml":
                    config = await utils.build_config(
                        emhass_conf, logger, defaults_path, config_path=config_path
                    )
                case _:
                    logger.warning(
                        f"Unsupported config file format: .{file_extension}, building parameters with only defaults"
                    )
                    config = await utils.build_config(emhass_conf, logger, defaults_path)
        else:
            logger.warning("Config file has no extension, building parameters with only defaults")
            config = await utils.build_config(emhass_conf, logger, defaults_path)
    else:
        # If unable to find config file, use only defaults_config.json
        logger.warning("Unable to obtain config.json file, building parameters with only defaults")
        config = await utils.build_config(emhass_conf, logger, defaults_path)
    if type(config) is bool and not config:
        raise Exception("Failed to find default config")

    # Obtain secrets from secrets_emhass.yaml?
    params_secrets = {}
    emhass_conf, built_secrets = await utils.build_secrets(
        emhass_conf, logger, secrets_path=secrets_path
    )
    params_secrets.update(built_secrets)

    # Build params
    params = await utils.build_params(emhass_conf, params_secrets, config, logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building parameters")
    # Add any passed params from args to params
    if args.params:
        params.update(orjson.loads(args.params))

    input_data_dict = await set_input_data_dict(
        emhass_conf,
        args.costfun,
        orjson.dumps(params).decode("utf-8"),
        args.runtimeparams,
        args.action,
        logger,
        args.debug,
    )
    if type(input_data_dict) is bool:
        raise Exception("A error has occurred while creating action objects")

    # Perform selected action
    if args.action == "perfect-optim":
        opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "dayahead-optim":
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "naive-mpc-optim":
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=args.debug)
    elif args.action == "forecast-model-fit":
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=args.debug
        )
        opt_res = None
    elif args.action == "forecast-model-predict":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=args.debug, mlf=mlf)
        opt_res = None
    elif args.action == "forecast-model-tune":
        if args.debug:
            _, _, mlf = await forecast_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlf = None
        df_pred_optim, mlf = await forecast_model_tune(
            input_data_dict, logger, debug=args.debug, mlf=mlf
        )
        opt_res = None
    elif args.action == "forecast-calibration":
        await forecast_calibration(input_data_dict, logger)
        opt_res = None
    elif args.action == "heating-need-forecast":
        await compute_heating_forecast(input_data_dict, logger)
        opt_res = None
    elif args.action == "heating-model-refit":
        await refit_heating_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "regressor-model-fit":
        mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        opt_res = None
    elif args.action == "regressor-model-predict":
        if args.debug:
            mlr = await regressor_model_fit(input_data_dict, logger, debug=args.debug)
        else:
            mlr = None
        prediction = await regressor_model_predict(
            input_data_dict, logger, debug=args.debug, mlr=mlr
        )
        opt_res = None
    elif args.action == "export-influxdb-to-csv":
        success = await export_influxdb_to_csv(input_data_dict, logger)
        opt_res = None
    elif args.action == "publish-data":
        opt_res = await publish_data(input_data_dict, logger)
    else:
        logger.error("The passed action argument is not valid")
        logger.error(
            "Try setting --action: perfect-optim, dayahead-optim, naive-mpc-optim, forecast-model-fit, forecast-model-predict, forecast-model-tune, forecast-calibration, heating-need-forecast, export-influxdb-to-csv or publish-data"
        )
        opt_res = None
    logger.info(opt_res)
    # Flush the logger
    ch.close()
    logger.removeHandler(ch)
    if (
        args.action == "perfect-optim"
        or args.action == "dayahead-optim"
        or args.action == "naive-mpc-optim"
        or args.action == "publish-data"
    ):
        return opt_res
    elif args.action == "forecast-model-fit":
        return df_fit_pred, df_fit_pred_backtest, mlf
    elif args.action == "forecast-model-predict":
        return df_pred
    elif args.action == "regressor-model-fit":
        return mlr
    elif args.action == "regressor-model-predict":
        return prediction
    elif args.action == "export-influxdb-to-csv":
        return success
    elif args.action == "forecast-model-tune":
        return df_pred_optim, mlf
    else:
        return opt_res


def main_sync():
    """Sync wrapper for async main function - used as CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
