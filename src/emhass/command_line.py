#!/usr/bin/env python3

import argparse
import asyncio
import copy
import json
import logging
import os
import pathlib
import pickle
import re
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

    UTC = timezone.utc  # noqa: UP017 - this *is* the fallback for when datetime.UTC doesn't exist

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
from emhass.persistence import (
    load_json_blob,
    load_pickle_blob,
    save_json_blob,
    save_pickle_blob,
)
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
        # "adjust_pv" (the PV-forecast-adjustment fit/refit path, see
        # _retrieve_and_fit_pv_model) always needs the forecast sensor to
        # compare against actual production, regardless of
        # set_use_adjusted_pv - that flag only controls whether an
        # ALREADY-fitted adjustment gets APPLIED to future forecasts, not
        # whether this fitting step itself can run (a user may want to fit/
        # inspect the model before ever turning the flag on). Every other
        # set_type only needs it when set_use_adjusted_pv is on - no point
        # fetching it if nothing downstream will apply the correction.
        if set_type == "adjust_pv" or optim_conf.get("set_use_adjusted_pv", True):
            var_list.append(retrieve_hass_conf["sensor_power_photovoltaics_forecast"])
    if set_type != "battery_id":
        # Per-phase load/PV sensors (see number_of_phases/_add_phase_balance_constraints)
        # - purely opt-in via the sensor fields themselves, no extra
        # number_of_phases check needed here: an unconfigured (blank) entry
        # simply contributes nothing.
        for conf_key in ("sensor_power_load_phase", "sensor_power_photovoltaics_phase"):
            for entity_id in retrieve_hass_conf.get(conf_key, []) or []:
                if entity_id and entity_id not in var_list:
                    var_list.append(entity_id)
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
        # Live per-room blind/window/door sensors - feed
        # _build_room_blind_positions/_build_room_opening_open/_build_room_door_open.
        # Previously missing here entirely (a real bug: those builders read
        # rh.df_final, but nothing ever requested these entities from HA/
        # InfluxDB, so they silently always fell back to "closed"/no data).
        for conf_key in (
            "heatpump_room_blind_sensors",
            "heatpump_room_window_sensors",
            "heatpump_room_door_sensors",
        ):
            for entity_id in retrieve_hass_conf.get(conf_key, []) or []:
                if entity_id and entity_id not in var_list:
                    var_list.append(entity_id)
        # Live whole-house heat-pump power/duty - needed by the Kalman
        # opening detector's predict step (see _build_room_kalman_opening_open)
        # to know how much heat was actually delivered since the last cycle.
        power_sensor = retrieve_hass_conf.get("heatpump_power_sensor", "")
        if power_sensor and power_sensor not in var_list:
            var_list.append(power_sensor)
        duty_sensor = retrieve_hass_conf.get("heatpump_duty_sensor", "")
        if duty_sensor and duty_sensor not in var_list:
            var_list.append(duty_sensor)
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


def _build_room_blind_positions(input_data_dict: dict, logger: logging.Logger) -> list | None:
    """Build the per-load room_blind_positions override list from live HA
    sensor data (heatpump_room_blind_sensors), mirroring _build_def_init_temp
    exactly. Returns None if no rooms are in use, otherwise a list of length
    number_of_deferrable_loads with None everywhere except room indices,
    where it holds the latest real sensor value (clipped to [0,1] - a raw
    HA cover.* entity's native position is often 0-100 and/or the opposite
    polarity, see heatpump_room_blind_sensors's own param_definitions.json
    description; this is a defensive safety net, not a fix).

    Only room loads get a value - the whole-house heat pump dispatch load
    (unlike def_init_temp) has no window/blind concept of its own.
    """
    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    passed_data = params.get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    if not room_load_indices:
        return None

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    room_blind_positions: list = [None] * num_def_loads
    rh = input_data_dict["rh"]
    df_final = getattr(rh, "df_final", None)
    if df_final is None:
        return room_blind_positions

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
    room_blind_sensors = retrieve_hass_conf.get("heatpump_room_blind_sensors", []) or []
    for name, k in room_load_indices.items():
        if k >= len(room_blind_positions):
            continue
        if name not in room_names:
            continue
        i = room_names.index(name)
        entity_id = room_blind_sensors[i] if i < len(room_blind_sensors) else None
        value = _latest_sensor_value(entity_id)
        if value is None:
            continue
        if value < 0.0 or value > 1.0:
            logger.warning(
                "Room %s: blind position sensor value %.3f is outside [0, 1] - clipping. "
                "See heatpump_room_blind_sensors's description for the expected 0(open)-1(closed) "
                "convention; a raw Home Assistant cover entity likely needs normalizing first.",
                name,
                value,
            )
        room_blind_positions[k] = min(1.0, max(0.0, value))

    return room_blind_positions


def _build_room_binary_open_state(
    input_data_dict: dict, logger: logging.Logger, entity_maps: list[dict[str, str]]
) -> list[bool] | None:
    """Shared engine for _build_room_opening_open/_build_room_door_open: OR
    together the live boolean state of every entity map passed in (each a
    room name -> its configured sensor entity_id, e.g. from
    _resolve_room_window_entity_map/_resolve_room_door_entity_map), across
    every room. Mirrors _build_room_blind_positions's overall shape, with
    two deliberate differences: values are interpreted as booleans via
    >= 0.5 (safe since HA binary_sensor on/off states are already coerced to
    1.0/0.0 by retrieve_hass.py's history processing), and there is no
    static-config fallback - absence of a live reading always means False
    (closed), never None, so every slot in the returned list is a plain
    bool.
    """
    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    passed_data = params.get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    if not room_load_indices:
        return None

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    room_open_state: list[bool] = [False] * num_def_loads
    rh = input_data_dict["rh"]
    df_final = getattr(rh, "df_final", None)
    if df_final is None:
        return room_open_state

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

    for name, k in room_load_indices.items():
        if k >= len(room_open_state):
            continue
        for entity_map in entity_maps:
            entity_id = entity_map.get(name)
            if not entity_id:
                continue
            value = _latest_sensor_value(entity_id)
            if value is not None and value >= 0.5:
                room_open_state[k] = True
                break

    return room_open_state


def _build_room_opening_open(input_data_dict: dict, logger: logging.Logger) -> list[bool] | None:
    """Per-load live "window OR door is open right now" state, feeding the
    shared pause-heating + extra-ventilation-loss thermal effect (see
    room_opening_open in optimization.py). True whenever either the room's
    configured window sensor or its door sensor currently reads open.
    """
    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    window_map = _resolve_room_window_entity_map(optim_conf, retrieve_hass_conf)
    door_map = _resolve_room_door_entity_map(optim_conf, retrieve_hass_conf)
    return _build_room_binary_open_state(input_data_dict, logger, [window_map, door_map])


def _build_room_door_open(input_data_dict: dict, logger: logging.Logger) -> list[bool] | None:
    """Per-load live "door is open right now" state, feeding the
    door-specific coupling-conductance boost to declared neighbors (see
    room_door_open in optimization.py) - deliberately door-only, unlike
    room_opening_open above which also considers the window sensor.
    """
    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    door_map = _resolve_room_door_entity_map(optim_conf, retrieve_hass_conf)
    return _build_room_binary_open_state(input_data_dict, logger, [door_map])


async def _build_room_kalman_opening_open(
    input_data_dict: dict, logger: logging.Logger, df_input_data_dayahead: pd.DataFrame
) -> list[bool] | None:
    """Per-load "probably open" state inferred purely from thermal
    behaviour (no HA window/door sensor required) - a per-room scalar
    Kalman filter comparing live observed room temperature against a
    one-step prediction from the room's own existing thermal model. See
    emhass.thermal.opening_kalman_detector for the filter math.

    Always runs (regardless of whether a real sensor is configured for that
    room) - the caller, _build_room_opening_open_with_kalman_fallback, OR's
    this with the sensor-based reading. Only ever feeds room_opening_open,
    never room_door_open (see that function's own docstring for why).

    Persists each room's (x, p, last_update_iso) across dispatch cycles in
    kalman_opening_detector_state.json (see persistence.py) - loaded and
    saved synchronously here, not deferred to the publish_data flow (that
    flow never runs for a bare naive-mpc-optim call without continual_publish/
    entity_save, so deferring the save would silently never persist for a
    large share of real deployments).

    Only usable during naive-mpc-optim - like _build_room_opening_open, this
    returns an all-False no-op when rh.df_final doesn't exist yet
    (dayahead-optim never fetches live HA data at all).
    """
    from emhass.thermal.opening_kalman_detector import (
        KALMAN_STATE_MAX_GAP_HOURS,
        PHYSICS_KALMAN_Q_C2,
        PHYSICS_KALMAN_R_C2,
        SELF_LEARNING_KALMAN_FALLBACK_R_C2,
        SELF_LEARNING_KALMAN_Q_FRACTION_OF_R,
        SELF_LEARNING_KALMAN_R_FLOOR_C2,
        cold_start_state,
        kalman_predict_update,
        predict_next_room_temperature_physics_family,
        predict_next_room_temperature_self_learning,
    )
    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    plant_conf = params.get("plant_conf", {})
    passed_data = params.get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    if not room_load_indices:
        return None

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    room_open_state: list[bool] = [False] * num_def_loads

    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]
    df_final = getattr(rh, "df_final", None)
    if df_final is None:
        return room_open_state

    def _latest_value(entity_id: str | None) -> float | None:
        if not entity_id or entity_id not in df_final.columns:
            return None
        series = df_final[entity_id].dropna()
        if series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except (TypeError, ValueError):
            return None

    # Live "how much heat was actually delivered" - a whole-house signal
    # (see heatpump_power_sensor/heatpump_duty_sensor's own config shape),
    # shared across every room's own predictor, mirroring how
    # _build_aggregate_heatpump_duty_expr already feeds one shared duty
    # signal into every self-learning room's fitted equation.
    duty_sensor = retrieve_hass_conf.get("heatpump_duty_sensor", "")
    duty_now = _latest_value(duty_sensor)
    power_now = None
    power_sensor = retrieve_hass_conf.get("heatpump_power_sensor", "")
    if power_sensor and power_sensor in df_final.columns:
        series = df_final[power_sensor].dropna()
        if not series.empty:
            delta = utils.resolve_incremental_series(
                series,
                "heatpump_power_sensor",
                logger,
                rate_dt_hours=_infer_timestep_hours(df_final.index),
            )
            try:
                power_now = float(delta.iloc[-1])
            except (TypeError, ValueError):
                power_now = None

    if duty_now is None and power_now is None:
        # Hard no-op, NOT a zero-fallback: treating an unresolved
        # heat-added reading as 0.0 would make every physics-family room
        # look colder-than-predicted every single cycle - a systematic
        # false-positive source, not a safe default.
        logger.debug(
            "Kalman opening detector: no live heatpump_power_sensor/"
            "heatpump_duty_sensor reading available this cycle - skipping."
        )
        return room_open_state

    heatpump_nominal_power = float(plant_conf.get("heatpump_nominal_power", 0.0) or 0.0)
    if duty_now is not None:
        duty_live = max(0.0, min(1.0, duty_now))
    elif heatpump_nominal_power > 0:
        duty_live = max(0.0, min(1.0, power_now / heatpump_nominal_power))
    else:
        logger.debug(
            "Kalman opening detector: only a live power reading is available "
            "and plant_conf.heatpump_nominal_power is unset - cannot derive "
            "a duty fraction, skipping this cycle."
        )
        return room_open_state

    state_blob = await load_json_blob(
        emhass_conf, "kalman_opening_detector_state.json", logger, default={}
    )
    rooms_state = dict(state_blob.get("rooms", {})) if isinstance(state_blob, dict) else {}
    new_rooms_state = dict(rooms_state)

    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_temp_sensors = retrieve_hass_conf.get("heatpump_room_temp_sensors", []) or []
    def_load_config = optim_conf.get("def_load_config", []) or []
    nominal_powers = optim_conf.get("nominal_power_of_deferrable_loads", []) or []
    blind_entity_map = _resolve_room_blind_entity_map(optim_conf, retrieve_hass_conf)

    outdoor_now = None
    wind_now = 0.0
    dni_now = 0.0
    dhi_now = 0.0
    if len(df_input_data_dayahead):
        row0 = df_input_data_dayahead.iloc[0]
        if "outdoor_temperature_forecast" in df_input_data_dayahead.columns:
            outdoor_now = float(row0["outdoor_temperature_forecast"])
        if "wind_speed" in df_input_data_dayahead.columns:
            wind_now = float(row0["wind_speed"])
        if "dni" in df_input_data_dayahead.columns:
            dni_now = float(row0["dni"])
        if "dhi" in df_input_data_dayahead.columns:
            dhi_now = float(row0["dhi"])

    now = pd.Timestamp.now(tz="UTC")
    # Self-learning model/coefficients are shared across rooms - loaded at
    # most once per cycle, only if at least one room actually needs them.
    self_learning_model = None
    self_learning_model_loaded = False
    dispatch_coeffs: dict = {}

    for name, k in room_load_indices.items():
        if k >= len(room_open_state) or name not in room_names:
            continue
        i = room_names.index(name)
        entity_id = room_temp_sensors[i] if i < len(room_temp_sensors) else None
        z_now = _latest_value(entity_id)
        if z_now is None:
            continue

        hc = {}
        if k < len(def_load_config) and isinstance(def_load_config[k], dict):
            hc = def_load_config[k].get("thermal_battery", {}) or {}
        is_self_learning = bool(hc.get("self_learning_dispatch"))

        prior = rooms_state.get(name)
        elapsed_hours = None
        if prior:
            try:
                elapsed_hours = (
                    now - pd.Timestamp(prior["last_update_iso"])
                ).total_seconds() / 3600.0
            except (KeyError, ValueError, TypeError):
                elapsed_hours = None

        default_r = SELF_LEARNING_KALMAN_FALLBACK_R_C2 if is_self_learning else PHYSICS_KALMAN_R_C2
        if prior is None or elapsed_hours is None or not (0 < elapsed_hours <= KALMAN_STATE_MAX_GAP_HOURS):
            x0, p0 = cold_start_state(z_now, default_r)
            new_rooms_state[name] = {"x": x0, "p": p0, "last_update_iso": now.isoformat()}
            continue

        if is_self_learning:
            if not self_learning_model_loaded:
                self_learning_model = await load_pickle_blob(
                    emhass_conf, "self_learning_physics_model.pkl", logger, default=None
                )
                coeffs_blob = await load_json_blob(
                    emhass_conf,
                    "self_learning_physics_room_dispatch_coefficients.json",
                    logger,
                    default={},
                )
                dispatch_coeffs = (
                    coeffs_blob.get("rooms", {}) if isinstance(coeffs_blob, dict) else {}
                )
                self_learning_model_loaded = True
            if self_learning_model is None:
                continue
            residual_std = dispatch_coeffs.get(name, {}).get("residual_std_c")
            r = max(
                SELF_LEARNING_KALMAN_R_FLOOR_C2,
                (residual_std**2) if residual_std else SELF_LEARNING_KALMAN_FALLBACK_R_C2,
            )
            q = SELF_LEARNING_KALMAN_Q_FRACTION_OF_R * r
            idx = pd.DatetimeIndex([now])
            supply_now = _latest_value(retrieve_hass_conf.get("heatpump_flow_temp_sensor", ""))
            df_house_fc = pd.DataFrame(
                {
                    "outdoor_temp": [outdoor_now if outdoor_now is not None else 10.0],
                    "wind_speed": [wind_now],
                    "dni": [dni_now],
                    "dhi": [dhi_now],
                    "heatpump_duty": [duty_live],
                    "supply_temp": [supply_now if supply_now is not None else z_now + 5.0],
                    "group_duty": [duty_live],
                },
                index=idx,
            )
            df_room_fc = df_house_fc.copy()
            blind_entity_id = blind_entity_map.get(name)
            if blind_entity_id:
                blind_now = _latest_value(blind_entity_id)
                if blind_now is not None:
                    df_room_fc["blind_position"] = [blind_now]
            x_pred = predict_next_room_temperature_self_learning(
                self_learning_model, name, df_house_fc, df_room_fc, prior["x"]
            )
            if x_pred is None:
                continue
        else:
            r = PHYSICS_KALMAN_R_C2
            q = PHYSICS_KALMAN_Q_C2
            nominal_power_w = float(nominal_powers[k]) if k < len(nominal_powers) else 0.0
            if isinstance(nominal_powers[k] if k < len(nominal_powers) else None, list):
                nominal_power_w = float(max(nominal_powers[k]))
            if nominal_power_w <= 0 or outdoor_now is None:
                continue
            x_pred = predict_next_room_temperature_physics_family(
                current_temp=prior["x"],
                duty=duty_live,
                outdoor_temp=outdoor_now,
                nominal_power_w=nominal_power_w,
                dt_hours=elapsed_hours,
                volume=float(hc.get("volume", 15.0) or 15.0),
                supply_temperature=float(hc.get("supply_temperature", 35.0) or 35.0),
                carnot_efficiency=float(hc.get("carnot_efficiency", 0.4) or 0.4),
                base_loss=float(hc.get("thermal_loss", 0.045) or 0.045),
                heating_demand_kwh=0.0,
                sense=str(hc.get("sense") or "heat"),
            )

        result = kalman_predict_update(prior["x"], prior["p"], x_pred, z_now, q, r)
        room_open_state[k] = result.is_open
        new_rooms_state[name] = {
            "x": result.x_new,
            "p": result.p_new,
            "last_update_iso": now.isoformat(),
        }

    await save_json_blob(
        emhass_conf, "kalman_opening_detector_state.json", {"rooms": new_rooms_state}, logger
    )
    return room_open_state


async def _build_room_opening_open_with_kalman_fallback(
    input_data_dict: dict, logger: logging.Logger, df_input_data_dayahead: pd.DataFrame
) -> list[bool] | None:
    """OR's the sensor-based room_opening_open (window OR door sensor) with
    the always-on Kalman-inferred signal (_build_room_kalman_opening_open) -
    either signal being "open" wins. This is the scope boundary: Kalman
    detection only ever feeds room_opening_open (pause + extra ventilation
    loss), NEVER room_door_open (neighbor-coupling boost) - a single room's
    own residual can't distinguish "my window is open" from "my door is
    open to a colder neighbor" without jointly modelling the neighbor too.
    room_door_open keeps using _build_room_door_open unchanged, sensor-only.
    """
    sensor_based = _build_room_opening_open(input_data_dict, logger)
    kalman_based = await _build_room_kalman_opening_open(
        input_data_dict, logger, df_input_data_dayahead
    )
    if sensor_based is None and kalman_based is None:
        return None
    num_def_loads = input_data_dict["params"]["optim_conf"].get("number_of_deferrable_loads", 0)
    sensor_based = sensor_based if sensor_based is not None else [False] * num_def_loads
    kalman_based = kalman_based if kalman_based is not None else [False] * num_def_loads
    return [a or b for a, b in zip(sensor_based, kalman_based)]


async def _build_room_kalman_blind_position(
    input_data_dict: dict, logger: logging.Logger, df_input_data_dayahead: pd.DataFrame
) -> list[float | None] | None:
    """Per-room continuous (0-1) blind/shading-position ESTIMATE inferred
    purely from thermal behaviour (no heatpump_room_blind_sensors entry
    required) - a per-room scalar Kalman filter over a PERSISTENCE state
    model (position has no independent per-cycle predictor the way
    temperature does - see emhass.thermal.blind_kalman_detector's own
    module docstring for the full algebraic derivation, and for why this
    only ever runs for a room that is CURRENTLY self-learning-dispatching
    (hc["self_learning_dispatch"] present) with an already-identified
    (nonzero) blind_x_dni coefficient - physics-family rooms are
    structurally out of scope, see that same module docstring). A room with
    a real blind sensor configured is skipped UNLESS
    heatpump_room_blind_infer_additional opts it in (that room has
    additional un-sensored shading) - see
    _build_room_blind_positions_with_kalman_fallback for how the two are
    then combined (max, never weakening the real reading).

    The RETURNED (dispatch-facing) value is withheld unless BOTH the live
    filter has converged enough to clear its own confidence gate AND
    self_learning_physics_blind_estimate_source is "auto_dispatch" (default
    "informational" - see _build_room_blind_positions_with_kalman_fallback,
    the caller that actually wires this into room_blind_positions).
    Regardless of that gate, this function always computes/persists/
    publishes an informational sensor.room_blind_position_estimate_<room>
    reading whenever the filter actually runs (i.e. every cycle past the
    initial cold start, whether or not that particular cycle was
    informative) - the "informational,
    always on, builds trust over time" surface the graduated rollout
    depends on.

    Persists each room's (x, p, last_update_iso, last_room_temp_c) across
    dispatch cycles in kalman_blind_detector_state.json - last_room_temp_c
    is the one field _build_room_kalman_opening_open's own state doesn't
    need (that detector's own filtered x IS a temperature, reusable
    directly as next cycle's current_temp; this detector's x is a
    position, so the raw previous actual reading is kept separately -
    closer to predict_one_step_history's own "true previous actual, never
    filtered" convention than the opening detector's live filter is).

    Only usable during naive-mpc-optim - like the opening detector, this
    returns an all-None no-op when rh.df_final doesn't exist yet
    (dayahead-optim never fetches live HA data at all).
    """
    from emhass.thermal.blind_kalman_detector import (
        BLIND_DNI_INFORMATIVE_FLOOR_WM2,
        BLIND_KALMAN_BETA_EPSILON,
        BLIND_KALMAN_DISPATCH_MAX_P,
        BLIND_KALMAN_Q,
        blind_cold_start_state,
        predict_room_temperature_blind_open_baseline,
        resolve_blind_measurement_noise,
    )
    from emhass.thermal.opening_kalman_detector import (
        KALMAN_STATE_MAX_GAP_HOURS,
        SELF_LEARNING_KALMAN_FALLBACK_R_C2,
        kalman_predict_update,
    )
    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    params = input_data_dict["params"]
    optim_conf = params["optim_conf"]
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    plant_conf = params.get("plant_conf", {})
    passed_data = params.get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    if not room_load_indices:
        return None

    num_def_loads = optim_conf.get("number_of_deferrable_loads", 0)
    room_blind_position_state: list[float | None] = [None] * num_def_loads
    # Graduated-trust rollout gate (see _build_room_blind_positions_with_kalman_fallback
    # and param_definitions.json's own description): "informational" (default)
    # means this function still runs its full compute/persist/publish cycle
    # below, unconditionally - only the RETURNED (dispatch-facing) value is
    # withheld.
    dispatch_source = optim_conf.get("self_learning_physics_blind_estimate_source", "informational")

    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]
    df_final = getattr(rh, "df_final", None)
    if df_final is None:
        return room_blind_position_state

    def _latest_value(entity_id: str | None) -> float | None:
        if not entity_id or entity_id not in df_final.columns:
            return None
        series = df_final[entity_id].dropna()
        if series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except (TypeError, ValueError):
            return None

    duty_sensor = retrieve_hass_conf.get("heatpump_duty_sensor", "")
    duty_now = _latest_value(duty_sensor)
    power_now = None
    power_sensor = retrieve_hass_conf.get("heatpump_power_sensor", "")
    if power_sensor and power_sensor in df_final.columns:
        series = df_final[power_sensor].dropna()
        if not series.empty:
            delta = utils.resolve_incremental_series(
                series,
                "heatpump_power_sensor",
                logger,
                rate_dt_hours=_infer_timestep_hours(df_final.index),
            )
            try:
                power_now = float(delta.iloc[-1])
            except (TypeError, ValueError):
                power_now = None

    if duty_now is None and power_now is None:
        logger.debug(
            "Kalman blind detector: no live heatpump_power_sensor/"
            "heatpump_duty_sensor reading available this cycle - skipping."
        )
        return room_blind_position_state

    heatpump_nominal_power = float(plant_conf.get("heatpump_nominal_power", 0.0) or 0.0)
    if duty_now is not None:
        duty_live = max(0.0, min(1.0, duty_now))
    elif heatpump_nominal_power > 0:
        duty_live = max(0.0, min(1.0, power_now / heatpump_nominal_power))
    else:
        logger.debug(
            "Kalman blind detector: only a live power reading is available "
            "and plant_conf.heatpump_nominal_power is unset - cannot derive "
            "a duty fraction, skipping this cycle."
        )
        return room_blind_position_state

    state_blob = await load_json_blob(
        emhass_conf, "kalman_blind_detector_state.json", logger, default={}
    )
    rooms_state = dict(state_blob.get("rooms", {})) if isinstance(state_blob, dict) else {}
    new_rooms_state = dict(rooms_state)

    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_temp_sensors = retrieve_hass_conf.get("heatpump_room_temp_sensors", []) or []
    def_load_config = optim_conf.get("def_load_config", []) or []
    blind_entity_map = _resolve_room_blind_entity_map(optim_conf, retrieve_hass_conf)
    # Per-room opt-in: run this detector even for a room with a real blind
    # sensor, when that room has ADDITIONAL un-sensored shading - see
    # heatpump_room_blind_infer_additional / _em_relabel_blind_position's
    # own docstring for the retroactive sibling of this same idea. Same
    # lightweight inline zip pattern used there, no dedicated resolver.
    _blind_infer_additional_list = optim_conf.get("heatpump_room_blind_infer_additional", []) or []
    infer_additional_blind = {
        str(n).strip(): bool(_blind_infer_additional_list[i])
        for i, n in enumerate(room_names)
        if str(n).strip() and i < len(_blind_infer_additional_list)
    }

    outdoor_now = None
    wind_now = 0.0
    dni_now = 0.0
    dhi_now = 0.0
    if len(df_input_data_dayahead):
        row0 = df_input_data_dayahead.iloc[0]
        if "outdoor_temperature_forecast" in df_input_data_dayahead.columns:
            outdoor_now = float(row0["outdoor_temperature_forecast"])
        if "wind_speed" in df_input_data_dayahead.columns:
            wind_now = float(row0["wind_speed"])
        if "dni" in df_input_data_dayahead.columns:
            dni_now = float(row0["dni"])
        if "dhi" in df_input_data_dayahead.columns:
            dhi_now = float(row0["dhi"])

    now = pd.Timestamp.now(tz="UTC")
    # Self-learning model/coefficients are shared across rooms - loaded at
    # most once per cycle, only if at least one room actually needs them.
    # A SEPARATE load from _build_room_kalman_opening_open's own cached
    # copy (accepted inefficiency - see blind_kalman_detector.py's own
    # module docstring/the feature's design plan for why: keeping the two
    # detectors fully independent is worth one extra small local-file read
    # per cycle).
    self_learning_model = None
    self_learning_model_loaded = False
    dispatch_coeffs: dict = {}

    for name, k in room_load_indices.items():
        if k >= len(room_blind_position_state) or name not in room_names:
            continue
        if name in blind_entity_map and not infer_additional_blind.get(name, False):
            continue  # a real sensor is configured - never touched by this detector
        i = room_names.index(name)
        entity_id = room_temp_sensors[i] if i < len(room_temp_sensors) else None
        z_now = _latest_value(entity_id)
        if z_now is None:
            continue

        hc = {}
        if k < len(def_load_config) and isinstance(def_load_config[k], dict):
            hc = def_load_config[k].get("thermal_battery", {}) or {}
        sl_dispatch = hc.get("self_learning_dispatch")
        if not sl_dispatch:
            continue  # only ever runs for a room currently self-learning-dispatching
        feature_names = sl_dispatch.get("feature_names", [])
        theta = sl_dispatch.get("theta", [])
        if "blind_x_dni" not in feature_names:
            continue
        beta = float(theta[feature_names.index("blind_x_dni")])
        if abs(beta) < BLIND_KALMAN_BETA_EPSILON:
            continue

        prior = rooms_state.get(name)
        elapsed_hours = None
        if prior:
            try:
                elapsed_hours = (
                    now - pd.Timestamp(prior["last_update_iso"])
                ).total_seconds() / 3600.0
            except (KeyError, ValueError, TypeError):
                elapsed_hours = None

        if (
            prior is None
            or elapsed_hours is None
            or not (0 < elapsed_hours <= KALMAN_STATE_MAX_GAP_HOURS)
        ):
            x0, p0 = blind_cold_start_state()
            new_rooms_state[name] = {
                "x": x0,
                "p": p0,
                "last_update_iso": now.isoformat(),
                "last_room_temp_c": z_now,
            }
            continue

        if dni_now <= BLIND_DNI_INFORMATIVE_FLOOR_WM2:
            # No sun right now - no information about position this cycle,
            # and nothing else below needs the fitted model at all. Belief
            # persists unchanged, uncertainty grows by q (predict-only, no
            # update - see blind_kalman_detector.py's own
            # kalman_forward_filter_with_persistence for the offline
            # equivalent of this same branch).
            x_new, p_new = prior["x"], prior["p"] + BLIND_KALMAN_Q
        else:
            if not self_learning_model_loaded:
                self_learning_model = await load_pickle_blob(
                    emhass_conf, "self_learning_physics_model.pkl", logger, default=None
                )
                coeffs_blob = await load_json_blob(
                    emhass_conf,
                    "self_learning_physics_room_dispatch_coefficients.json",
                    logger,
                    default={},
                )
                dispatch_coeffs = (
                    coeffs_blob.get("rooms", {}) if isinstance(coeffs_blob, dict) else {}
                )
                self_learning_model_loaded = True
            if self_learning_model is None:
                continue

            idx = pd.DatetimeIndex([now])
            supply_now = _latest_value(retrieve_hass_conf.get("heatpump_flow_temp_sensor", ""))
            df_house_fc = pd.DataFrame(
                {
                    "outdoor_temp": [outdoor_now if outdoor_now is not None else 10.0],
                    "wind_speed": [wind_now],
                    "dni": [dni_now],
                    "dhi": [dhi_now],
                    "heatpump_duty": [duty_live],
                    "supply_temp": [supply_now if supply_now is not None else z_now + 5.0],
                    "group_duty": [duty_live],
                },
                index=idx,
            )
            df_room_fc = df_house_fc.copy()
            x_pred_open = predict_room_temperature_blind_open_baseline(
                self_learning_model, name, df_house_fc, df_room_fc, prior["last_room_temp_c"]
            )
            if x_pred_open is None:
                continue
            residual = z_now - x_pred_open

            residual_std = dispatch_coeffs.get(name, {}).get("residual_std_c")
            residual_std_c = (
                residual_std if residual_std else SELF_LEARNING_KALMAN_FALLBACK_R_C2**0.5
            )
            r = resolve_blind_measurement_noise(residual_std_c, beta, dni_now)
            raw_z = min(1.0, max(0.0, residual / (beta * dni_now)))
            result = kalman_predict_update(
                prior["x"],
                prior["p"],
                x_pred=prior["x"],
                z_measured=raw_z,
                q=BLIND_KALMAN_Q,
                r=r,
            )
            x_new, p_new = result.x_new, result.p_new

        new_rooms_state[name] = {
            "x": x_new,
            "p": p_new,
            "last_update_iso": now.isoformat(),
            "last_room_temp_c": z_now,
        }
        confident = p_new < BLIND_KALMAN_DISPATCH_MAX_P
        room_blind_position_state[k] = (
            x_new if (confident and dispatch_source == "auto_dispatch") else None
        )

        await rh.post_data(
            pd.Series([round(x_new, 4)]),
            0,
            f"sensor.room_blind_position_estimate_{_slugify_room_name(name)}",
            "",
            "",
            f"{name} Blind Position Estimate",
            type_var="mlregressor",
        )

    await save_json_blob(
        emhass_conf, "kalman_blind_detector_state.json", {"rooms": new_rooms_state}, logger
    )
    return room_blind_position_state


async def _build_room_blind_positions_with_kalman_fallback(
    input_data_dict: dict, logger: logging.Logger, df_input_data_dayahead: pd.DataFrame
) -> list | None:
    """Precedence merge of the real-sensor blind position
    (_build_room_blind_positions) and the live Kalman-estimated one
    (_build_room_kalman_blind_position) - a precedence merge, NOT a plain
    boolean OR like the opening detector's own fallback wrapper, since
    blind position is a graduated-trust CONTINUOUS override, not two
    independent binary signals where either one being "true" wins.

    A room's real sensor reading is always a FLOOR, never weakened: for a
    room with no real sensor, the Kalman estimate is used outright (subject
    to its own confidence gate/self_learning_physics_blind_estimate_source
    check, both already inside _build_room_kalman_blind_position). For a
    partially-sensored room opted in via heatpump_room_blind_infer_additional
    (that detector then also runs for it instead of skipping it - see its
    own docstring), BOTH a real reading and a Kalman estimate can be present
    at once - take the max (more-shaded) of the two, since a second,
    un-sensored closed blind can only add MORE shading than the real sensor
    alone reports, never less. For every other (non-opted-in) sensored
    room, the Kalman estimate is always None here exactly as before, so
    this reduces to "real sensor always wins" unchanged.
    """
    sensor_based = _build_room_blind_positions(input_data_dict, logger)
    kalman_based = await _build_room_kalman_blind_position(
        input_data_dict, logger, df_input_data_dayahead
    )
    if sensor_based is None and kalman_based is None:
        return None
    num_def_loads = input_data_dict["params"]["optim_conf"].get("number_of_deferrable_loads", 0)
    sensor_based = sensor_based if sensor_based is not None else [None] * num_def_loads
    kalman_based = kalman_based if kalman_based is not None else [None] * num_def_loads

    def _combine(s: float | None, k: float | None) -> float | None:
        if s is not None and k is not None:
            return max(s, k)
        return s if s is not None else k

    return [_combine(s, k) for s, k in zip(sensor_based, kalman_based)]


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
            # load this cycle - see _resolve_load_profiles, which runs
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


async def _resolve_load_profiles(
    rh: RetrieveHass,
    optim_conf: dict,
    params_optim_conf: dict,
    retrieve_hass_conf: dict,
    params: dict,
    logger: logging.Logger,
) -> None:
    """Per-cycle WashData program discovery, for any deferrable load with
    load_washdata_enabled set AND a configured load_washdata_device -
    independent of is_manual_load (see associations.csv: "being a washing
    machine" and "being manually dispatched" are orthogonal). Runs inside
    set_input_data_dict, before Forecast/OptimizationCache/Optimization are
    built - unlike most other per-load fields, this is NOT frozen at
    config-save time: it's read fresh on every action call so a profile that
    WashData refines over more cycles is picked up automatically.

    load_washdata_enabled is checked explicitly (not just a non-empty
    device string) so that disabling the UI checkbox reliably turns this
    off even if load_washdata_device[k] still holds a previously-picked
    value underneath the now-hidden dropdown.

    For each enabled load with a configured device slug (e.g. "wasmachine"), fetches
    every entity via RetrieveHass.get_all_states() (a direct REST call,
    deliberately bypassing InfluxDB - see _fetch_ha_entity_payload's
    docstring) and discovers every learned program from the naming
    convention WashData's ha_washdata integration uses:
    sensor.<device>_profiel_<program>_aantal, whose attributes carry
    power_profile/power_profile_interval_min and whose numeric state is a
    run count. Ambiguity between multiple discovered programs is resolved
    two ways:
      - A manual load (is_manual_load[k]) with manual_load_program_select_sensor
        configured: read that entity's current option (e.g. WashData's own
        select.<device>_cyclusprogramma, which the human sets to the program
        they're about to run) and match it to a discovered program by slug -
        only a manual load can know this in advance, since a human is
        physically choosing it.
      - Otherwise (no select match, select unset/left on "auto_detect", or
        the load isn't manual): fall back to the most-used discovered
        program (highest run count), so an automatically-dispatched load can
        still benefit from a real learned power shape instead of a
        hand-typed load_programs guess.

    Only on a fully valid resolution does this swap that load's flat
    nominal_power_of_deferrable_loads[k] scalar for the resampled Watt
    sequence, mirroring the pre-existing load_type == "program_based"
    mechanism in _normalize_deferrable_load_categories.

    Mutates BOTH optim_conf (the object about to be used to build/cache the
    Optimization instance - see set_input_data_dict, where this and
    params["optim_conf"] are distinct dict objects by this point) and
    params_optim_conf (params["optim_conf"]) with the same values, so the
    resolved sequence is visible both to the solver's cache key/constraints
    and to _apply_manual_load_runtime_overrides / naive_mpc_optim /
    dayahead_forecast_optim, which all read params["optim_conf"].

    Any failure (HA unreachable, no learned program yet, missing/invalid
    profile attributes) is caught and logged per-load; that load's existing
    flat scalar values are left untouched, so it gracefully falls back to
    the flat model.
    """
    devices = optim_conf.get("load_washdata_device", []) or []
    enabled_flags = optim_conf.get("load_washdata_enabled", []) or []

    def _is_enabled(idx: int) -> bool:
        return bool(idx < len(enabled_flags) and enabled_flags[idx])

    if not any(
        _is_enabled(i) and str(device or "").strip()
        for i, device in enumerate(devices)
    ):
        return

    time_step = retrieve_hass_conf.get("optimization_time_step")
    if isinstance(time_step, (int, float)):
        time_step = pd.to_timedelta(time_step, "minutes")
    if not isinstance(time_step, pd.Timedelta) or time_step <= pd.Timedelta(0):
        return
    target_step_min = time_step / pd.Timedelta(minutes=1)

    is_manual_load = optim_conf.get("is_manual_load", []) or []
    program_select_sensors = retrieve_hass_conf.get("manual_load_program_select_sensor", []) or []

    all_states = None  # fetched lazily, once, only if a device is actually configured
    for k, device in enumerate(devices):
        device = str(device or "").strip()
        if not device or not _is_enabled(k):
            continue
        try:
            if all_states is None:
                all_states = await rh.get_all_states()
            prefix = f"sensor.{device}_profiel_"
            suffix = "_aantal"
            programs = []
            for state_obj in all_states:
                entity_id = str(state_obj.get("entity_id", ""))
                if not entity_id.startswith(prefix) or not entity_id.endswith(suffix):
                    continue
                attributes = state_obj.get("attributes") or {}
                sequence = utils._parse_profile_to_float_list(attributes.get("power_profile"))
                if not sequence:
                    continue
                try:
                    source_interval = float(attributes.get("power_profile_interval_min"))
                except (TypeError, ValueError):
                    source_interval = None
                if not source_interval or source_interval <= 0:
                    continue
                resampled = utils._resample_power_profile(sequence, source_interval, target_step_min)
                if not resampled:
                    continue
                try:
                    count = float(state_obj.get("state"))
                except (TypeError, ValueError):
                    count = 0.0
                slug = entity_id[len(prefix) : -len(suffix)]
                programs.append({"slug": slug, "power_pattern": resampled, "count": count})

            if not programs:
                logger.debug(
                    "Load %d: no learned WashData programs found yet for device '%s', "
                    "falling back to configured flat power/duration",
                    k,
                    device,
                )
                continue

            chosen = None
            if k < len(is_manual_load) and is_manual_load[k]:
                select_sensor = (
                    str(program_select_sensors[k]).strip()
                    if k < len(program_select_sensors)
                    else ""
                )
                if select_sensor:
                    payload = await rh.get_entity_state_and_attributes(select_sensor)
                    selected = str((payload or {}).get("state") or "").strip()
                    if selected and selected.lower() != "auto_detect":
                        selected_slug = re.sub(r"[^a-z0-9_]+", "_", selected.lower()).strip("_")
                        chosen = next((p for p in programs if p["slug"] == selected_slug), None)
                        if chosen is None:
                            logger.warning(
                                "Load %d: %s = '%s' doesn't match any discovered WashData "
                                "program for device '%s' (have: %s), falling back to most-used",
                                k,
                                select_sensor,
                                selected,
                                device,
                                ", ".join(p["slug"] for p in programs),
                            )

            if chosen is None:
                chosen = max(programs, key=lambda p: p["count"])

            resampled = chosen["power_pattern"]
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
                "Load %d ('%s'): resolved WashData program '%s' (%d steps at %.1f min, "
                "%d program(s) discovered)",
                k,
                device,
                chosen["slug"],
                len(resampled),
                target_step_min,
                len(programs),
            )
        except Exception as e:  # a WashData/HA hiccup must never break optimization
            logger.warning(
                "Load %d: error resolving WashData device '%s' (%s), falling back",
                k,
                device,
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
    # retrieve_home_assistant_data can return success=True even when one of
    # the requested entities had no data at all in HA/InfluxDB over the
    # window (e.g. sensor_power_photovoltaics_forecast never having been
    # published yet, or its history not retained that far back) - the
    # resulting DataFrame simply lacks that column rather than filling it
    # with NaN. adjust_pv_forecast_data_prep indexes both columns
    # unconditionally, so check here and fail soft with a clear message
    # instead of an unhandled KeyError crashing the whole action.
    missing_cols = [c for c in (fcst.var_pv, fcst.var_pv_forecast) if c not in df_input_data.columns]
    if missing_cols:
        fcst.logger.warning(
            f"No historical data available for {missing_cols} over the retrieved window - "
            "the PV adjustment model needs actual production and previously-published "
            "forecast history to compare against. Falling back to unadjusted PV forecast."
        )
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


async def refit_adjust_pv_forecast_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Force an immediate re-fit of adjust_pv_forecast's own regression
    model (model_adjust_pv), bypassing the is_model_outdated staleness
    check adjust_pv_forecast itself uses during regular dayahead/MPC
    cycles (adjusted_pv_model_max_age, default 24h) - lets a user
    manually refresh the model on demand (e.g. right after changing
    adjusted_pv_regression_model) instead of waiting for it to age out.

    Reuses _retrieve_and_fit_pv_model unchanged, which already retrieves
    history, fits, and persists the model to disk (inside
    Forecast.adjust_pv_forecast_fit) - this action always runs live
    (get_data_from_file=False) since it's only ever user-triggered.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when failed
    :rtype: dict | None
    """
    fcst = input_data_dict["fcst"]
    rh = input_data_dict["rh"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    optim_conf = input_data_dict["optim_conf"]
    emhass_conf = input_data_dict["emhass_conf"]

    success = await _retrieve_and_fit_pv_model(
        fcst, False, retrieve_hass_conf, optim_conf, rh, emhass_conf, test_df_literal
    )
    if not success:
        logger.error("adjust-pv-forecast-refit: failed to fit the PV forecast adjustment model")
        return None

    n_samples = len(getattr(fcst, "x_adjust_pv", None) or [])
    logger.info("adjust-pv-forecast-refit: model refit successfully (%d samples)", n_samples)
    return {
        "regression_model": optim_conf["adjusted_pv_regression_model"],
        "n_samples": n_samples,
    }


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
        "hybrid-heatpump-forecast",
        "hybrid-heatpump-model-refit",
        "self-learning-physics-forecast",
        "self-learning-physics-refit",
        "thermal-models-forecast",
        "thermal-models-refit",
        "thermal-models-tune",
        "pv-horizon-refit",
        "pv-forecast-test",
        "adjust-pv-forecast-refit",
        "load-forecast-test",
        "load-quantile-spread-refit",
    ]
    # Resolve any configured load's learned WashData power profile fresh for
    # this action - independent of is_manual_load - must happen before
    # Forecast/OptimizationCache/Optimization are built below, since a
    # resolved profile changes optim_conf's structure (see
    # _resolve_load_profiles).
    if (
        normalized_set_type not in actions_without_fcst_or_opt
        and normalized_set_type not in actions_skip_optim_cache
    ):
        await _resolve_load_profiles(
            rh, optim_conf, params.get("optim_conf", {}), retrieve_hass_conf, params, logger
        )
    if normalized_set_type in actions_without_fcst_or_opt:
        fcst = None
        opt = None
        logger.debug(f"Skipping Optimization creation for action: {set_type}")
    else:
        if optim_conf.get("pv_horizon_learning_enabled", False):
            # A missing/never-refit profile means _apply_pv_horizon_mask
            # (forecast.py) silently no-ops - see pv-horizon-refit's own
            # docstring.
            horizon_state = await load_json_blob(
                emhass_conf, "pv_horizon_profile.json", logger, default=None
            )
            plant_conf["pv_horizon_profile"] = (horizon_state or {}).get("profile")
            plant_conf["pv_horizon_partial_transmittance"] = (horizon_state or {}).get(
                "profile_partial_transmittance"
            )
            plant_conf["pv_horizon_sun_path_envelope"] = (horizon_state or {}).get("sun_path_envelope")
            plant_conf["pv_horizon_diffuse_transmission_factor"] = (horizon_state or {}).get(
                "diffuse_transmission_factor"
            )
        if optim_conf.get("open_meteo_pv_ensemble_enabled", False):
            # Read whatever the tracker last persisted (cold start -> {} ->
            # equal weighting, which _select_percentile_member_weather
            # already treats as a plain unweighted percentile). This
            # cycle's own P10 blend uses this snapshot; the update below
            # (after fcst exists) persists a fresh one for the *next*
            # cycle to pick up - a one-cycle-old weight snapshot is fine
            # for something that only meaningfully changes once a day.
            scores_state = await load_json_blob(
                emhass_conf, "pv_ensemble_model_scores.json", logger, default=None
            )
            plant_conf["pv_ensemble_model_weights"] = (scores_state or {}).get("scores", {})
        # A missing/never-refit blob means _get_historical_daily_load_spread
        # (forecast.py) falls back to the generic bundled reference dataset
        # per-bucket, same as before load-quantile-spread-refit existed -
        # loaded unconditionally (cheap; only used at all when
        # load_forecast_quantile_bias > 0).
        load_spread_state = await load_json_blob(
            emhass_conf, "load_quantile_spread.json", logger, default=None
        )
        plant_conf["load_quantile_spread"] = load_spread_state
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        if optim_conf.get("open_meteo_pv_ensemble_enabled", False) and normalized_set_type in (
            "dayahead-optim",
            "naive-mpc-optim",
        ):
            # Resolve any matured predictions from previous cycles and log a
            # fresh one for tomorrow - see _update_pv_ensemble_model_scores's
            # own docstring. Needs fcst (for _calculate_pvlib_power on each
            # candidate model's own bare forecast), so this can only run
            # after Forecast is constructed above - see this block's own
            # comment on plant_conf["pv_ensemble_model_weights"] for why
            # that's fine (feeds next cycle, not this one).
            await _update_pv_ensemble_model_scores(fcst, rh, retrieve_hass_conf, emhass_conf, logger)
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
    elif set_type == "hybrid-heatpump-forecast":
        # Retrieves its own live sensor readings and weather forecast inside
        # compute_hybrid_heatpump_forecast(); no generic prep needed here.
        result = {}
    elif set_type == "hybrid-heatpump-model-refit":
        # Retrieves its own (long) history window inside
        # refit_hybrid_heatpump_model(); no generic prep needed here.
        result = {}
    elif set_type == "self-learning-physics-forecast":
        # Retrieves its own live sensor readings and weather forecast inside
        # compute_self_learning_physics_forecast(); no generic prep needed here.
        result = {}
    elif set_type == "self-learning-physics-refit":
        # Retrieves its own (long) history window inside
        # refit_self_learning_physics_model(); no generic prep needed here.
        result = {}
    elif set_type == "pv-horizon-refit":
        # Retrieves its own (long) actual-PV + Open-Meteo historical-weather
        # window inside refit_pv_horizon_model(); no generic prep needed here.
        result = {}
    elif set_type == "pv-forecast-test":
        # Same helper _prepare_dayahead_optim itself calls for the PV leg of
        # a normal dayahead/MPC cycle - computing just the PV forecast here
        # instead of running a full optimization.
        p_pv_forecast, df_weather = await _get_dayahead_pv_forecast(ctx)
        result = {"p_pv_forecast": p_pv_forecast, "df_weather": df_weather}
        # P10/P50/P90 side by side, so the preview shows the actual forecast
        # spread instead of just one blended number - reuses the ensemble
        # pool get_weather_forecast already fetched above (inside
        # _get_dayahead_pv_forecast), so this is free of extra network
        # calls. None (silently omitted below) when open_meteo_pv_ensemble_
        # enabled is off or every candidate model's fetch failed.
        if df_weather is not None:
            quantiles = ctx.fcst.get_pv_ensemble_quantile_forecast(df_weather)
            if quantiles is not None:
                result["p_pv_forecast_p10"] = quantiles["p10"]
                result["p_pv_forecast_p50"] = quantiles["p50"]
                result["p_pv_forecast_p90"] = quantiles["p90"]
    elif set_type == "adjust-pv-forecast-refit":
        # Retrieves its own history window inside refit_adjust_pv_forecast_model();
        # no generic prep needed here, same minimal pattern as pv-horizon-refit above.
        result = {}
    elif set_type == "load-forecast-test":
        # P10/P50/P90 side by side, so the preview shows the actual forecast
        # spread instead of just one blended number. Computes the plain
        # point forecast once (bias forced to 0.0) then derives P10/P90
        # from it via get_load_quantile_forecast's THR reconciliation -
        # avoids re-running the whole (possibly expensive) load-forecast
        # method a second/third time just to extract a biased blend, and
        # gets P10 for free alongside the existing P90.
        original_bias = ctx.optim_conf.get("load_forecast_quantile_bias", 0.0)
        ctx.optim_conf["load_forecast_quantile_bias"] = 0.0
        p_load_forecast_p50 = await ctx.fcst.get_load_forecast(
            days_min_load_forecast=ctx.optim_conf["delta_forecast_daily"].days,
            method=ctx.optim_conf["load_forecast_method"],
        )
        ctx.optim_conf["load_forecast_quantile_bias"] = original_bias
        quantiles = await ctx.fcst.get_load_quantile_forecast(p_load_forecast_p50)
        result = {
            "p_load_forecast_p10": quantiles["p10"],
            "p_load_forecast_p50": quantiles["p50"],
            "p_load_forecast_p90": quantiles["p90"],
        }
    elif set_type == "load-quantile-spread-refit":
        # Retrieves its own (long) actual-load history window inside
        # refit_load_quantile_spread_model(); no generic prep needed here,
        # same minimal pattern as pv-horizon-refit above.
        result = {}
    elif set_type == "thermal-models-refit":
        # Delegates to whichever of the three refit_* functions above are
        # enabled, each of which retrieves its own history window; no
        # generic prep needed here.
        result = {}
    elif set_type == "thermal-models-tune":
        # Delegates to tune_self_learning_physics_model, which retrieves
        # its own history window; no generic prep needed here.
        result = {}
    elif set_type == "thermal-models-forecast":
        # Delegates to whichever of the three compute_*_forecast functions
        # above are enabled, each of which retrieves its own live sensor
        # readings/weather forecast; no generic prep needed here.
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


def _merge_weather_column(
    input_data_dict: dict,
    df_input_data_dayahead: pd.DataFrame,
    column: str,
    warn_on_resolution: bool,
    logger: logging.Logger,
) -> None:
    """Merge one column (e.g. "ghi", "wind_speed", "dni", "dhi") from the
    weather forecast (input_data_dict["df_weather"]) onto df_input_data_dayahead
    in place, if that column exists on the weather frame. Same tz-alignment /
    nearest-reindex-with-1h-tolerance / forward-then-backward-fill logic
    originally written for GHI alone (prepare_forecast_and_weather_data) -
    factored out here so wind_speed/dni/dhi (needed by the self-learning-physics
    dispatch equation, see optimization.py::_add_self_learning_dispatch_constraints)
    reach data_opt the same reliable way GHI already does, instead of never
    reaching optimization.py at all (a gap that existed before this feature).
    """
    if input_data_dict["df_weather"] is None or column not in input_data_dict["df_weather"].columns:
        return
    dayahead_index = df_input_data_dayahead.index
    series = input_data_dict["df_weather"][column].copy()

    # Handle Timezone Mismatches
    if dayahead_index.tz is None and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    elif dayahead_index.tz is not None and series.index.tz is None:
        series.index = series.index.tz_localize(dayahead_index.tz)
    elif dayahead_index.tz is not None and series.index.tz is not None:
        series.index = series.index.tz_convert(dayahead_index.tz)

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
                "Step changes in %s may occur.",
                weather_freq,
                dayahead_freq,
                column,
            )

    # Robust Reindexing
    df_input_data_dayahead[column] = series.reindex(
        dayahead_index, method="nearest", tolerance=pd.Timedelta("1h")
    )

    # Final safety fill
    if df_input_data_dayahead[column].isnull().any():
        df_input_data_dayahead[column] = (
            df_input_data_dayahead[column].fillna(method="ffill").fillna(method="bfill")
        )

    logger.debug(
        "Merged %s data into optimization input: mean=%.3g, max=%.3g",
        column,
        df_input_data_dayahead[column].mean(),
        df_input_data_dayahead[column].max(),
    )


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

    # Merge GHI (Global Horizontal Irradiance), plus wind_speed/dni/dhi (needed
    # by the self-learning-physics dispatch equation, see
    # optimization.py::_add_self_learning_dispatch_constraints - these three
    # previously never reached data_opt at all) from the weather forecast.
    for _weather_col in ("ghi", "wind_speed", "dni", "dhi"):
        _merge_weather_column(input_data_dict, df_input_data_dayahead, _weather_col, warn_on_resolution, logger)

    # Solar elevation (needed by the physics-family awning-type blind-shading
    # formula, see utils.calculate_shaded_window_irradiance) AND azimuth
    # (needed by the self-learning-physics model's own dni_x_sun_az_sin/cos
    # features, see self_learning_physics.py's module docstring) - both
    # computed directly from timestamps/location via pvlib, not fetched from
    # a weather API, so they're merged on the DataFrame's own index rather
    # than looped through _merge_weather_column like the fetched columns
    # above.
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    solar_angles = Forecast.compute_solar_angles(
        df_input_data_dayahead,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    df_input_data_dayahead["solar_elevation"] = solar_angles["solar_elevation"]
    # sun_alt_sin/sun_az_sin/sun_az_cos: the exact sin/cos convention
    # self_learning_physics.py::_physics_features reads (and
    # feature_engineering.py::add_solar_features already uses elsewhere in
    # this codebase, kept consistent here). sun_alt_cos is additionally
    # needed by optimization.py::_add_rc_physics_dispatch_constraints's own
    # plane-of-array projection (thermal_mass_physics._facade_poa_vectorized) -
    # self-learning-physics's own dispatch equation never needed it, so it
    # was never merged here before RC dispatch existed.
    alt_rad = np.radians(solar_angles["solar_elevation"].to_numpy(dtype=float))
    az_rad = np.radians(solar_angles["solar_azimuth"].to_numpy(dtype=float))
    df_input_data_dayahead["sun_alt_sin"] = np.sin(alt_rad)
    df_input_data_dayahead["sun_alt_cos"] = np.cos(alt_rad)
    df_input_data_dayahead["sun_az_sin"] = np.sin(az_rad)
    df_input_data_dayahead["sun_az_cos"] = np.cos(az_rad)

    # Per-phase load/PV split for the additive phase-balance safety
    # constraint (see optimization.py::_add_phase_balance_constraints) -
    # a ratio-based split of the ALREADY-computed aggregate p_load_forecast/
    # p_pv_forecast columns above by each phase's historical share
    # (utils.compute_phase_power_shares), not a separate per-phase forecast
    # pipeline - the tuned aggregate forecast itself is untouched. Only
    # computed when number_of_phases > 1 - a pure no-op otherwise, matching
    # every other single-phase deployment's byte-identical behavior.
    plant_conf = input_data_dict.get("plant_conf", {}) or {}
    n_phases = int(plant_conf.get("number_of_phases", 1) or 1)
    if n_phases > 1:
        phase_labels = [f"L{i + 1}" for i in range(n_phases)]
        df_history = input_data_dict.get("df_input_data")
        load_phase_sensors = retrieve_hass_conf.get("sensor_power_load_phase", []) or []
        pv_phase_sensors = retrieve_hass_conf.get("sensor_power_photovoltaics_phase", []) or []

        load_share = None
        if len(load_phase_sensors) == n_phases:
            load_share = utils.compute_phase_power_shares(df_history, load_phase_sensors, logger)
        elif any(load_phase_sensors):
            logger.warning(
                "sensor_power_load_phase has %d entries but number_of_phases=%d - "
                "ignoring it (provide exactly %d entity ids, one per phase).",
                len(load_phase_sensors),
                n_phases,
                n_phases,
            )
        if load_share is None:
            logger.warning(
                "Phase balancing is on (number_of_phases=%d) but no usable per-phase "
                "load sensors are configured - the per-phase safety constraint will "
                "only cover phase-assigned deferrable loads/battery, not your "
                "uncontrolled household base load; fuse-overload protection is not "
                "guaranteed.",
                n_phases,
            )

        pv_share = None
        if len(pv_phase_sensors) == n_phases:
            pv_share = utils.compute_phase_power_shares(df_history, pv_phase_sensors, logger)
        elif any(pv_phase_sensors):
            logger.warning(
                "sensor_power_photovoltaics_phase has %d entries but "
                "number_of_phases=%d - ignoring it (provide exactly %d entity ids, "
                "one per phase).",
                len(pv_phase_sensors),
                n_phases,
                n_phases,
            )
        if pv_share is None:
            # PV, unlike the uncontrolled load, has a reasonable default: one
            # central inverter usually balances its own AC output across its
            # legs by hardware design.
            pv_share = [1.0 / n_phases] * n_phases

        for i, label in enumerate(phase_labels):
            df_input_data_dayahead[f"p_load_phase_{label}"] = (
                (load_share[i] if load_share is not None else 0.0)
                * df_input_data_dayahead["p_load_forecast"]
            )
            df_input_data_dayahead[f"p_pv_phase_{label}"] = (
                pv_share[i] * df_input_data_dayahead["p_pv_forecast"]
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
    # _resolve_load_profiles, which mutate params["optim_conf"]) -
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
    room_opening_open = await _build_room_opening_open_with_kalman_fallback(
        input_data_dict, logger, df_input_data_dayahead
    )
    room_blind_positions = await _build_room_blind_positions_with_kalman_fallback(
        input_data_dict, logger, df_input_data_dayahead
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
            room_blind_positions=room_blind_positions,
            room_opening_open=room_opening_open,
            room_door_open=_build_room_door_open(input_data_dict, logger),
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
    room_opening_open = await _build_room_opening_open_with_kalman_fallback(
        input_data_dict, logger, df_input_data_dayahead
    )
    room_blind_positions = await _build_room_blind_positions_with_kalman_fallback(
        input_data_dict, logger, df_input_data_dayahead
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
            room_blind_positions=room_blind_positions,
            room_opening_open=room_opening_open,
            room_door_open=_build_room_door_open(input_data_dict, logger),
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
    sentinel) - plus, opt-in (heating_model_refit_fit_electric_power_enabled,
    see utils.py::_append_heating_forecast_targets and
    _fit_temperature_params's own fit_electric_power docstring),
    sensor.heating_electric_power_forecast. EMHASS never calls a device
    service here - these are informational forecast sensors only, same
    "publish only" pattern as the rest of this fork.

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
    # Optional - held flat across the whole forecast horizon below (same
    # simplification refit_self_learning_physics_model's own forecast path
    # already uses for blind position: no per-room forecast infra, blinds
    # change state rarely, so the last live reading is a fair proxy).
    blind_sensor = retrieve_hass_conf.get("heatpump_blind_position_sensor", "")

    days_list = utils.get_days_list(2)
    sensors_to_fetch = [indoor_sensor, blind_sensor] if blind_sensor else [indoor_sensor]
    if not await rh.get_data(days_list, sensors_to_fetch):
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

    current_blind_position = 0.0
    if blind_sensor and blind_sensor in rh.df_final.columns:
        blind_history = rh.df_final[blind_sensor].dropna()
        if not blind_history.empty:
            current_blind_position = float(np.clip(blind_history.iloc[-1], 0.0, 1.0))

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
            # Held flat at the last live reading (0.0/open when unconfigured)
            # across the whole horizon - see the fetch above for why.
            "blind_position": current_blind_position,
        },
        index=df_weather.index,
    )
    thermal_inputs = _prepare_inputs(
        df_physics_input,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    dt_h = _infer_timestep_hours(df_weather.index)
    # facade2/facade3 weights are configured constants, never fitted (see
    # thermal_mass_physics.py's own module docstring) - re-read from config
    # here rather than persisted params, same as refit_heating_model's own
    # treatment.
    facade2_weight = float(retrieve_hass_conf.get("heatpump_facade2_weight", "") or 0.0)
    facade3_weight = float(retrieve_hass_conf.get("heatpump_facade3_weight", "") or 0.0)
    sim = _simulate_open_loop(
        thermal_inputs,
        params,
        dt_h=dt_h,
        initial_air=current_indoor_temp,
        initial_mass=current_indoor_temp,
        initial_q_emit=0.0,
        facade2_weight=facade2_weight,
        facade3_weight=facade3_weight,
    )

    safety_margin = float(optim_conf.get("heating_forecast_safety_margin_c", 0.5))
    comfort_min = float(optim_conf.get("heating_forecast_comfort_min_temp", 19.0))
    adjusted = sim.room - safety_margin
    below = np.where(adjusted < comfort_min)[0]
    heating_needed_by = df_weather.index[int(below[0])].isoformat() if len(below) else "beyond_horizon"

    passed_data = input_data_dict["params"]["passed_data"]
    temp_forecast_entity = passed_data.get("custom_indoor_temp_forecast_id")
    needed_by_entity = passed_data.get("custom_heating_needed_by_id")
    # Opt-in (see utils.py::_append_heating_forecast_targets) - only
    # registered when heating_model_refit_fit_electric_power_enabled is on,
    # so None here just means "not opted in", not an error.
    electric_forecast_entity = passed_data.get("custom_heating_electric_power_forecast_id")
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
    # electric_pred is always present on sim (see thermal_mass_physics.py's
    # own SimResult) but only actually meaningful once
    # heating_model_refit_fit_electric_power_enabled has been on for at
    # least one refit/tune - otherwise emitter_power_scale_w sits at its
    # DEFAULT_X0 seed of 0.0 and this is just an all-zero curve, which is
    # exactly why the entity itself is only registered (see
    # utils.py::_append_heating_forecast_targets) when that flag is on.
    if electric_forecast_entity is not None:
        electric_series = pd.Series(sim.electric_pred, index=df_weather.index)
        await rh.post_data(
            electric_series,
            0,
            electric_forecast_entity["entity_id"],
            electric_forecast_entity["device_class"],
            electric_forecast_entity["unit_of_measurement"],
            electric_forecast_entity["friendly_name"],
            type_var="power",
            **common_kwargs,
        )

    result = {
        "heating_needed_by": heating_needed_by,
        "current_indoor_temp": current_indoor_temp,
        "comfort_min_temp": comfort_min,
        "safety_margin_c": safety_margin,
        "horizon_hours": horizon_hours,
        "forecast_steps": len(df_weather),
        "mean_electric_power_forecast_w": (
            float(np.mean(sim.electric_pred)) if electric_forecast_entity is not None else None
        ),
    }
    await save_json_blob(emhass_conf, "heating_forecast_last_run.json", result, logger)
    logger.info("heating-need-forecast: heating_needed_by=%s", heating_needed_by)
    # Added AFTER the JSON persist above (orjson.dumps can't serialize a
    # DataFrame) - the forecasted curve itself, plus a flat comfort_min_temp
    # reference line directly visualizing what heating_needed_by means (the
    # curve's first crossing of that line) - rendered by
    # get_injection_dict_thermal_models/get_forecast_trend_plot_html.
    result["indoor_temp_forecast_df"] = pd.DataFrame(
        {"forecast": temp_series, "comfort_min_temp": comfort_min}, index=df_weather.index
    )
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
    # Gates the window-transmitted solar pathway only (see
    # thermal_mass_physics.py's own module docstring) - falls back to
    # _prepare_inputs's own 0.0 (fully open) default when unconfigured,
    # exactly recovering pre-blind-support behavior.
    "heatpump_blind_position_sensor": "blind_position",
    # Gates the extra ventilation-loss term only (see
    # thermal_mass_physics.py's own module docstring) - falls back to
    # _prepare_inputs's own 0.0 (closed) default when unconfigured.
    "heatpump_door_window_sensor": "door_open",
}
_REFIT_MIN_ROWS = 500  # a handful of days at 15-30min resolution - below this, don't even try


async def _fill_missing_weather_from_open_meteo(
    df_raw: pd.DataFrame,
    retrieve_hass_conf: dict,
    days_list: pd.date_range,
    weather_columns: set[str],
    fcst: Forecast,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Fill outdoor_temp/wind_speed/wind_bearing/ghi/dni/dhi for a refit's
    df_raw from Open-Meteo's Historical Weather API, shared by all three
    thermal-model refits (RC physics, hybrid heat pump, self-learning-physics).

    Controlled by heatpump_weather_use_own_sensors (default True): when True,
    only columns missing entirely (sensor unconfigured) or with gaps (partial
    HA/InfluxDB history) are filled - a fully-covered sensor column is left
    untouched. When False, every column in weather_columns is replaced
    wholesale from Open-Meteo regardless of what sensors are configured -
    lets a specifically unreliable sensor (e.g. one exposed to direct sun,
    reading high) be overridden without unconfiguring it.

    A failed Open-Meteo fetch is logged and swallowed - the refit continues
    with whatever df_raw already had (its own static-default fallback for
    a still-missing column), never crashes the refit over this.

    :param df_raw: The refit's own HA/InfluxDB-sourced history, already
        renamed to internal column names (rh.df_final.rename(columns=sensor_map)).
    :type df_raw: pd.DataFrame
    :param retrieve_hass_conf: Live-HA config dict (for the toggle + Latitude/Longitude,
        already on fcst).
    :type retrieve_hass_conf: dict
    :param days_list: Same days_list already used for this refit's own rh.get_data call.
    :type days_list: pd.date_range
    :param weather_columns: The weather columns this particular refit's own
        *_SENSOR_COLUMN_MAP carries (a subset of
        Forecast.OPEN_METEO_HISTORICAL_WEATHER_VARS's values).
    :type weather_columns: set[str]
    :param fcst: The shared Forecast instance (for lat/lon/time_zone/freq).
    :type fcst: Forecast
    :param logger: The logger object
    :type logger: logging.Logger
    :return: df_raw, with the needed weather columns filled in.
    :rtype: pd.DataFrame
    """
    use_own = retrieve_hass_conf.get("heatpump_weather_use_own_sensors", True)
    needed = {
        c
        for c in weather_columns
        if not use_own or c not in df_raw.columns or df_raw[c].isna().any()
    }
    if not needed:
        return df_raw
    try:
        om_df = await fcst.get_historical_weather_from_open_meteo(days_list, list(needed))
    except Exception:
        logger.warning(
            "Could not fetch Open-Meteo historical weather to fill %s - continuing "
            "with whatever data is already available for this refit.",
            sorted(needed),
            exc_info=True,
        )
        return df_raw
    om_df = om_df.reindex(df_raw.index, method="nearest", tolerance=fcst.freq)
    for col in needed:
        if col not in om_df.columns:
            continue
        if use_own and col in df_raw.columns:
            # Own-sensors mode: only fill actual gaps, never overwrite a
            # real reading.
            df_raw[col] = df_raw[col].fillna(om_df[col])
        else:
            # Either the column doesn't exist yet (sensor unconfigured), or
            # heatpump_weather_use_own_sensors is off - wholesale replace.
            df_raw[col] = om_df[col]
    return df_raw


# EM-style (fit -> teacher-forced residuals -> relabel -> refit) retroactive
# door/window-opening and blind-position relabeling for the RC model - the
# same small-fixed-iteration-count philosophy as self-learning-physics's own
# _OPENING_RELABEL_DEFAULT_ITERATIONS/_BLIND_RELABEL_DEFAULT_ITERATIONS
# (see command_line.py's own module-level constants near
# _em_relabel_opening_open), reused here as separate constants since the RC
# model's own fit cost (3-restart least_squares, not RLS) is different
# enough that these may need independent tuning later.
_RC_DOOR_RELABEL_DEFAULT_ITERATIONS = 2
_RC_BLIND_RELABEL_DEFAULT_ITERATIONS = 2
# Same rationale as self-learning-physics's own _BLIND_RELABEL_MIN_INFORMATIVE_ROWS:
# blind position is only ever observable when there is sun to block or not -
# too little sunny history in the window means skip rather than guess.
_RC_BLIND_RELABEL_MIN_INFORMATIVE_ROWS = 50


def _fit_score_rc_model(
    df_raw: pd.DataFrame,
    n_rows: int,
    prepare_kwargs: dict,
    dt_h: float,
    segment_len: int,
    regularization_overrides: dict[str, float] | None = None,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    phase_offsets: list[int] | None = None,
    warm_start_from: np.ndarray | None = None,
    fit_electric_power: bool = False,
) -> dict:
    """Split df_raw 70/15/15 chronologically and fit+score the RC model
    exactly as refit_heating_model's own established discipline: fit on
    train, score on held-out val (the only number that gates deploy),
    retrain on train+val for the actually-deployed params, score once on
    test (informational only, never gates anything). Pure fit/score - no
    persistence, no deploy decision - so refit_heating_model can call this
    once for the baseline (today's exact data, unrelabeled) and once for a
    relabel-enhanced version, and pick whichever wins on held-out val,
    mirroring self-learning-physics-refit's own per-room baseline-vs-
    enhanced auto-selection.

    :param phase_offsets: forwarded to _fit_temperature_params's own
        phase_offsets - when given more than one offset, the fit is a
        SINGLE joint optimization whose residual vector spans every phase
        at once (rather than several independent fits), so one shared
        parameter set has to explain the data under every segment-
        boundary alignment (the "multiple shooting" interval layout, see
        _fit_temperature_params's own docstring). Defaults to None,
        today's exact single-fixed-phase behavior. When more than one
        offset is given, params_train is ALSO scored against df_val at
        every one of those offsets (see "phase_val_maes"/
        "phase_val_mae_mean" below) - a robustness diagnostic, not a
        selection mechanism (there's only one fitted params_final either
        way).
    :param warm_start_from: forwarded to _fit_temperature_params's own
        warm_start_from - see that parameter's own docstring.
    :param fit_electric_power: forwarded to _fit_temperature_params's own
        fit_electric_power - see that parameter's own docstring.
    :return: {"val_mae", "test_mae", "params_final", "fit_info", "n_val_rows"} -
        val_mae is float("inf") when there aren't enough rows for a
        meaningful held-out split, so callers can compare val_mae across
        sources without a separate validity check. When phase_offsets has
        more than one entry, also includes "phase_val_maes" (val_mae at
        each offset, in offset order) and "phase_val_mae_mean".
    """
    from emhass.thermal.thermal_mass_physics import _fit_temperature_params, _prepare_inputs, _simulate_segmented

    i_train_end = max(1, int(round(n_rows * 0.70)))
    i_val_end = max(i_train_end + 1, int(round(n_rows * 0.85)))
    split1, split2 = df_raw.index[i_train_end], df_raw.index[min(i_val_end, n_rows - 1)]
    df_train = df_raw[df_raw.index < split1]
    df_val = df_raw[(df_raw.index >= split1) & (df_raw.index < split2)]
    df_test = df_raw[df_raw.index >= split2]
    df_trainval = df_raw[df_raw.index < split2]
    if len(df_val) < 10:
        return {
            "val_mae": float("inf"),
            "test_mae": None,
            "params_final": None,
            "fit_info": {},
            "n_val_rows": len(df_val),
        }

    def _score(inputs, params) -> float:
        pred = _simulate_segmented(
            inputs,
            params,
            dt_h=dt_h,
            segment_len=segment_len,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
        )
        finite = np.isfinite(inputs.room)
        return float(np.mean(np.abs(pred[finite] - inputs.room[finite])))

    thermal_inputs_train = _prepare_inputs(df_train, **prepare_kwargs)
    params_train, _fit_info_train = _fit_temperature_params(
        thermal_inputs_train,
        dt_h=dt_h,
        segment_len=segment_len,
        max_nfev=300,
        regularization_overrides=regularization_overrides,
        facade2_weight=facade2_weight,
        facade3_weight=facade3_weight,
        phase_offsets=phase_offsets,
        warm_start_from=warm_start_from,
        fit_electric_power=fit_electric_power,
    )
    val_mae = _score(_prepare_inputs(df_val, **prepare_kwargs), params_train)

    thermal_inputs_trainval = _prepare_inputs(df_trainval, **prepare_kwargs)
    params_final, fit_info = _fit_temperature_params(
        thermal_inputs_trainval,
        dt_h=dt_h,
        segment_len=segment_len,
        max_nfev=300,
        regularization_overrides=regularization_overrides,
        facade2_weight=facade2_weight,
        facade3_weight=facade3_weight,
        phase_offsets=phase_offsets,
        warm_start_from=warm_start_from,
        fit_electric_power=fit_electric_power,
    )
    test_mae = _score(_prepare_inputs(df_test, **prepare_kwargs), params_final) if len(df_test) >= 10 else None

    result = {
        "val_mae": val_mae,
        "test_mae": test_mae,
        "params_final": params_final,
        "fit_info": fit_info,
        "n_val_rows": len(df_val),
    }
    # Robustness diagnostic only - there's a single params_train/params_final
    # either way (the joint fit above already had to explain every phase at
    # once), this just reports how much a single FIXED phase (today's
    # default) would have distorted the apparent val score.
    if phase_offsets and len(phase_offsets) > 1:
        phase_val_maes = [
            _score(_prepare_inputs(df_val.iloc[off:], **prepare_kwargs), params_train)
            if len(df_val) - off >= 10
            else val_mae
            for off in phase_offsets
        ]
        result["phase_val_maes"] = phase_val_maes
        result["phase_val_mae_mean"] = float(np.mean(phase_val_maes))
    return result


def _em_relabel_door_open_rc(
    df: pd.DataFrame,
    prepare_kwargs: dict,
    dt_h: float,
    segment_len: int,
    n_iterations: int,
    logger: logging.Logger,
    regularization_overrides: dict[str, float] | None = None,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    phase_offsets: list[int] | None = None,
    warm_start_from: np.ndarray | None = None,
    fit_electric_power: bool = False,
) -> pd.DataFrame:
    """EM-style retroactive door/window-opening relabeling for the RC
    model - the RC-model sibling of _em_relabel_opening_open, adapted for a
    stateful ODE model instead of a stateless one-step regression (see
    thermal_mass_physics_kalman.py's own module docstring for why the
    predictor itself had to be rewritten, not just reused).

    Simpler than the blind case below: door/window detection is a binary
    "was there an unexplained mismatch" gate (kalman_forward_filter_array +
    smoothed_opening_flags, imported UNCHANGED from
    opening_kalman_detector.py), not an algebraic inversion into a
    magnitude - so there's no bootstrap-vs-calibrate split to worry about.

    Only ever called when heatpump_door_window_sensor is unconfigured (see
    refit_heating_model) - a room with a real sensor is never touched here.

    :param phase_offsets: forwarded, unchanged, to every internal
        _fit_temperature_params call below - the inferred door/window
        labels feed everything downstream, so a single fixed segment
        phase can bias them just as much as it biases the final scoring
        fit (see _fit_score_rc_model's own phase_offsets docstring).
        predict_one_step_history itself needs no phase-awareness: it's
        fully teacher-forced from real data every step, with no
        segmentation/multiple-shooting concept at all.
    :param warm_start_from: forwarded, unchanged, to every internal
        _fit_temperature_params call below - see that parameter's own
        docstring.
    :param fit_electric_power: forwarded, unchanged, to every internal
        _fit_temperature_params call below - see that parameter's own
        docstring.
    :return: df with its "door_open" column replaced by the final
        iteration's inferred 0/1 flags (unchanged in every other column).
    """
    from emhass.thermal.opening_kalman_detector import (
        SELF_LEARNING_KALMAN_FALLBACK_R_C2,
        SELF_LEARNING_KALMAN_Q_FRACTION_OF_R,
        SELF_LEARNING_KALMAN_R_FLOOR_C2,
        cold_start_state,
        kalman_forward_filter_array,
        kalman_rts_smooth,
        smoothed_opening_flags,
    )
    from emhass.thermal.thermal_mass_physics import _fit_temperature_params, _prepare_inputs
    from emhass.thermal.thermal_mass_physics_kalman import predict_one_step_history

    blended = df.copy()
    blended["door_open"] = 0.0
    if n_iterations <= 0:
        return blended

    params, _ = _fit_temperature_params(
        _prepare_inputs(blended, **prepare_kwargs),
        dt_h=dt_h,
        segment_len=segment_len,
        max_nfev=300,
        regularization_overrides=regularization_overrides,
        facade2_weight=facade2_weight,
        facade3_weight=facade3_weight,
        phase_offsets=phase_offsets,
        warm_start_from=warm_start_from,
        fit_electric_power=fit_electric_power,
    )
    for _iteration in range(n_iterations):
        inputs = _prepare_inputs(blended, **prepare_kwargs)
        pred, _sensitivity, _q_solar = predict_one_step_history(
            inputs,
            params,
            dt_h,
            force_door_zero=True,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
        )
        actual = inputs.room
        residual = actual - pred
        finite = residual[np.isfinite(residual)]
        if len(finite) >= 2:
            # Scaled-MAD, not plain std - see _em_relabel_opening_open's own
            # comment: this is bootstrapping from unlabeled data, a plain
            # std would be inflated by the very anomalies being looked for.
            mad = float(np.median(np.abs(finite - np.median(finite))))
            r = max(SELF_LEARNING_KALMAN_R_FLOOR_C2, (1.4826 * mad) ** 2)
        else:
            r = SELF_LEARNING_KALMAN_FALLBACK_R_C2
        q = SELF_LEARNING_KALMAN_Q_FRACTION_OF_R * r

        x0, p0 = cold_start_state(float(actual[0]), r)
        trajectory = kalman_forward_filter_array(x0, p0, pred, actual, q, r)
        _, p_smooth = kalman_rts_smooth(trajectory)
        is_open = smoothed_opening_flags(trajectory, p_smooth, r)
        blended["door_open"] = is_open.astype(float)

        params, _ = _fit_temperature_params(
            _prepare_inputs(blended, **prepare_kwargs),
            dt_h=dt_h,
            segment_len=segment_len,
            max_nfev=300,
            regularization_overrides=regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )

    logger.info(
        "heating-model-refit: door/window-opening relabeling complete over %d "
        "iteration(s) - %d/%d steps flagged open",
        n_iterations,
        int(blended["door_open"].sum()),
        len(blended),
    )
    return blended


def _em_relabel_blind_position_rc(
    df: pd.DataFrame,
    prepare_kwargs: dict,
    dt_h: float,
    segment_len: int,
    n_iterations: int,
    logger: logging.Logger,
    regularization_overrides: dict[str, float] | None = None,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    phase_offsets: list[int] | None = None,
    warm_start_from: np.ndarray | None = None,
    fit_electric_power: bool = False,
) -> pd.DataFrame:
    """EM-style retroactive blind-position relabeling for the RC model -
    the RC-model sibling of _em_relabel_blind_position, adapted for a
    stateful ODE model instead of a stateless one-step regression.

    Unlike self-learning-physics's blind_x_dni (a separate regression
    coefficient that starts at exactly 0, unidentified, until real
    variance exists in the feature - see blind_kalman_detector.py's own
    bootstrap_raw_blind_signal_from_residual), RC's sensitivity is derived
    from solar_gain_c_per_h and friends, CORE model parameters always fit
    meaningfully from real solar data regardless of blind state - so this
    has no bootstrap-vs-calibrate split, every iteration uses the same
    algebraic inversion (see thermal_mass_physics_kalman.py's own module
    docstring).

    Only ever called when heatpump_blind_position_sensor is unconfigured
    (see refit_heating_model) - a room with a real sensor is never touched
    here.

    :param phase_offsets: forwarded, unchanged, to every internal
        _fit_temperature_params call below - see _em_relabel_door_open_rc's
        own phase_offsets docstring for why the intermediate fits need
        this just as much as the final scoring fit does.
    :param warm_start_from: forwarded, unchanged, to every internal
        _fit_temperature_params call below - see that parameter's own
        docstring.
    :param fit_electric_power: forwarded, unchanged, to every internal
        _fit_temperature_params call below - see that parameter's own
        docstring.
    :return: df with its "blind_position" column replaced by the final
        iteration's inferred 0-1 curve (unchanged in every other column),
        OR the input df unchanged (still zeroed) if there's too little
        sunny history in the window to say anything.
    """
    from emhass.thermal.blind_kalman_detector import (
        BLIND_KALMAN_Q,
        blind_cold_start_state,
        kalman_forward_filter_with_persistence,
        smoothed_blind_position,
    )
    from emhass.thermal.opening_kalman_detector import kalman_rts_smooth
    from emhass.thermal.thermal_mass_physics import _fit_temperature_params, _prepare_inputs
    from emhass.thermal.thermal_mass_physics_kalman import (
        invert_blind_position_from_residual,
        predict_one_step_history,
        resolve_measurement_noise,
    )

    blended = df.copy()
    blended["blind_position"] = 0.0
    if n_iterations <= 0:
        return blended

    inputs0 = _prepare_inputs(blended, **prepare_kwargs)
    # Pre-fit gate on raw ghi (not q_solar - that now depends on
    # facade_azimuth_deg/facade_tilt_deg, which haven't been fit yet at
    # this point) - 50 W/m2 matches blind_kalman_detector.py's own
    # BLIND_DNI_INFORMATIVE_FLOOR_WM2, the same "well under a typical
    # clear-sky noon reading, high enough to exclude dawn/dusk/overcast"
    # reasoning applied to ghi instead of dni.
    n_informative = int((inputs0.ghi > 50.0).sum())
    if n_informative < _RC_BLIND_RELABEL_MIN_INFORMATIVE_ROWS:
        logger.info(
            "heating-model-refit: too little sunny history (%d informative rows) "
            "to relabel blind position - skipping",
            n_informative,
        )
        return blended

    params, _ = _fit_temperature_params(
        inputs0,
        dt_h=dt_h,
        segment_len=segment_len,
        max_nfev=300,
        regularization_overrides=regularization_overrides,
        facade2_weight=facade2_weight,
        facade3_weight=facade3_weight,
        phase_offsets=phase_offsets,
        warm_start_from=warm_start_from,
        fit_electric_power=fit_electric_power,
    )
    for _iteration in range(n_iterations):
        inputs = _prepare_inputs(blended, **prepare_kwargs)
        pred, sensitivity, q_solar = predict_one_step_history(
            inputs,
            params,
            dt_h,
            force_blind_zero=True,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
        )
        actual = inputs.room
        residual = actual - pred
        finite = residual[np.isfinite(residual)]
        residual_std_c = (
            float(1.4826 * np.median(np.abs(finite - np.median(finite)))) if len(finite) >= 2 else 0.3
        )
        raw = invert_blind_position_from_residual(residual, sensitivity, q_solar)
        r = resolve_measurement_noise(residual_std_c, sensitivity)

        x0, p0 = blind_cold_start_state()
        trajectory = kalman_forward_filter_with_persistence(x0, p0, raw, BLIND_KALMAN_Q, r)
        x_smooth, _ = kalman_rts_smooth(trajectory)
        position = smoothed_blind_position(x_smooth)
        blended["blind_position"] = position

        params, _ = _fit_temperature_params(
            _prepare_inputs(blended, **prepare_kwargs),
            dt_h=dt_h,
            segment_len=segment_len,
            max_nfev=300,
            regularization_overrides=regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )

    logger.info(
        "heating-model-refit: blind-position relabeling complete over %d iteration(s) "
        "- mean inferred position=%.2f (%d informative rows)",
        n_iterations,
        float(blended["blind_position"].mean()),
        n_informative,
    )
    return blended


async def _run_heating_model_refit(
    input_data_dict: dict,
    logger: logging.Logger,
    *,
    warm_start_from: np.ndarray | None = None,
) -> dict | None:
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

    Shared body for both refit_heating_model (warm_start_from=None, today's
    exact behavior) and tune_heating_model (warm_start_from=the currently-
    deployed params) - see warm_start_from's own docstring on
    _fit_temperature_params for what threading it through changes.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :param warm_start_from: forwarded, unchanged, to every internal
        _fit_temperature_params call (the final scoring fits AND the
        EM-relabel loops' own internal fits) - see that parameter's own
        docstring.
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
        _infer_timestep_hours,
        mass_tau_h_anchor_from_building_class,
        tau_emit_h_anchor_from_emitter_type,
    )

    window_days = int(optim_conf.get("heating_model_refit_window_days", 60))
    days_list = utils.get_days_list(window_days)
    if not await rh.get_data(days_list, list(sensor_map.keys())):
        logger.error("heating-model-refit: failed to retrieve history from Home Assistant/InfluxDB")
        return None

    df_raw = rh.df_final.rename(columns=sensor_map)
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df_raw = await _fill_missing_weather_from_open_meteo(
        df_raw,
        retrieve_hass_conf,
        days_list,
        {"outdoor_temp", "wind_speed", "wind_bearing", "ghi", "dni", "dhi"},
        input_data_dict["fcst"],
        logger,
    )
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

    # Held-out chronological 70/15/15 split, fit, and scoring all live in
    # _fit_score_rc_model (same convention - and same "test is touched
    # exactly once, never for a decision" discipline - as
    # refit_self_learning_physics_model's own split) so it can be run once
    # for the baseline data and, when relabeling is enabled below, once
    # more for the relabel-enhanced data - whichever wins on held-out val
    # is what gets deployed, exactly mirroring self-learning-physics-
    # refit's own per-room baseline-vs-enhanced auto-selection.
    prepare_kwargs = {
        "latitude": float(retrieve_hass_conf["Latitude"]),
        "longitude": float(retrieve_hass_conf["Longitude"]),
    }
    dt_h = _infer_timestep_hours(df_raw.index)
    segment_len = max(1, round(24.0 / dt_h))  # ~24h segments, matching the original fit

    # facade_azimuth_deg/facade_tilt_deg (and facade2/facade3's own pairs -
    # optional extra orientation slots, e.g. a dakraam or a secondary
    # window facing a different way than the primary facade - see
    # thermal_mass_physics.py's own module docstring) always stay genuinely
    # fittable - a configured value is never hard-pinned. When configured,
    # it becomes a MUCH stronger regularisation anchor than the mild
    # default pull (see _fit_temperature_params's own
    # regularization_overrides docstring: _CONFIGURED_PRIOR_REG_WEIGHT
    # vs _DEFAULT_PRIOR_REG_WEIGHT) - "hard to move away from", not
    # "impossible to move away from", so real, sustained data can still
    # correct a wrong estimate. Weight (facade2_weight/facade3_weight) is
    # different: always a hard configured constant, never fitted at all -
    # it isn't something a temperature-only fit can identify - defaulting
    # to 0.0 (slot disabled, exactly today's single-orientation behavior)
    # when unconfigured.
    regularization_overrides: dict[str, float] = {}
    for slot in ("facade", "facade2", "facade3"):
        azimuth_str = retrieve_hass_conf.get(f"heatpump_{slot}_azimuth_deg", "")
        tilt_str = retrieve_hass_conf.get(f"heatpump_{slot}_tilt_deg", "")
        if azimuth_str:
            regularization_overrides[f"{slot}_azimuth_deg"] = float(azimuth_str)
        if tilt_str:
            regularization_overrides[f"{slot}_tilt_deg"] = float(tilt_str)
    facade2_weight = float(retrieve_hass_conf.get("heatpump_facade2_weight", "") or 0.0)
    facade3_weight = float(retrieve_hass_conf.get("heatpump_facade3_weight", "") or 0.0)

    # Same soft-anchor treatment (never a hard pin) for building thermal
    # mass and heat-emitter response time - see
    # thermal_mass_physics.py's own mass_tau_h_anchor_from_building_class/
    # tau_emit_h_anchor_from_emitter_type for the ISO 13790 table / HVAC
    # rule-of-thumb estimates behind these, and why floor area itself is
    # never needed. Both return None (no override added - fully free,
    # today's exact unconfigured behavior) for an empty or unrecognised
    # config value.
    mass_class_anchor = mass_tau_h_anchor_from_building_class(
        retrieve_hass_conf.get("heatpump_building_mass_class", "")
    )
    if mass_class_anchor is not None:
        regularization_overrides["mass_tau_h"] = mass_class_anchor
    emitter_anchor = tau_emit_h_anchor_from_emitter_type(retrieve_hass_conf.get("heatpump_emitter_type", ""))
    if emitter_anchor is not None:
        regularization_overrides["tau_emit_h"] = emitter_anchor

    # Opt-in, off by default: fitting at a single FIXED segment-start phase
    # (today's only behavior) risks the "multiple shooting" segmentation
    # quietly biasing toward whatever time-of-day happens to land on
    # segment boundaries - see _fit_temperature_params's own phase_offsets
    # docstring. Built ONCE here and threaded through every fit in the
    # chain below (the baseline/door_only/blind_only/both scoring fits AND
    # the EM-relabel functions' own internal fits) - a single JOINT fit per
    # call, not num_phases independent ones, so the added cost is roughly
    # linear in num_phases (residual-vector length), not a multiplied
    # restart count - still opt-in for a routinely-scheduled action, same
    # precedent as door/blind relabeling.
    phase_robust_enabled = bool(optim_conf.get("heating_model_refit_phase_robust_enabled", False))
    num_phases = int(optim_conf.get("heating_model_refit_num_phases", 4))
    phase_offsets = (
        [round(i * segment_len / num_phases) for i in range(num_phases)] if phase_robust_enabled else None
    )

    # Opt-in, off by default: adds a joint electric-power residual block to
    # every fit in the chain (see _fit_temperature_params's own
    # fit_electric_power docstring) - only meaningful (and only gated on)
    # when heatpump_power_sensor is actually configured, same "no sensor,
    # no target" precedence rule real sensors get elsewhere in this file.
    fit_electric_power = bool(
        optim_conf.get("heating_model_refit_fit_electric_power_enabled", False)
    ) and bool(retrieve_hass_conf.get("heatpump_power_sensor", ""))

    def _fit_score(df: pd.DataFrame, rows: int) -> dict:
        return _fit_score_rc_model(
            df,
            rows,
            prepare_kwargs,
            dt_h,
            segment_len,
            regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )

    baseline = _fit_score(df_raw, n_rows)
    if phase_robust_enabled and "phase_val_maes" in baseline:
        logger.info(
            "heating-model-refit: baseline phase-robustness - val_mae per phase %s (mean %.3f)",
            [round(v, 3) for v in baseline["phase_val_maes"]],
            baseline["phase_val_mae_mean"],
        )
    if baseline["val_mae"] == float("inf"):
        logger.error(
            "heating-model-refit: too few validation rows (%d) after a 70/15/15 "
            "chronological split of %d rows - aborting.",
            baseline["n_val_rows"],
            n_rows,
        )
        return None

    # Only when the matching real sensor is unconfigured - a room with a
    # real reading is never touched by inference, same precedence rule
    # self-learning-physics's own relabeling establishes.
    door_relabel_enabled = bool(
        optim_conf.get("heating_model_refit_door_relabel_enabled", False)
    ) and not retrieve_hass_conf.get("heatpump_door_window_sensor", "")
    blind_relabel_enabled = bool(
        optim_conf.get("heating_model_refit_blind_relabel_enabled", False)
    ) and not retrieve_hass_conf.get("heatpump_blind_position_sensor", "")

    # Every ENABLED sub-combination is fit and scored independently, not
    # just "baseline vs both-enabled-together" - real data on this feature
    # showed why: door relabeling alone can look better than baseline on
    # val while generalizing clearly worse on held-out test (a sign of
    # overfitting a noisy channel), and combining a good channel (blind)
    # with a bad one (door) can score BEST on val of all candidates while
    # being the WORST on test - val alone can't detect that combination
    # trap. Comparing every enabled combination on val, rather than only
    # the fully-combined one, lets a genuinely-good channel (e.g. blind)
    # win on its own even when a co-enabled bad channel (e.g. door) would
    # otherwise have dragged the only-available "enhanced" candidate down.
    candidates: list[tuple[str, dict]] = [("baseline", baseline)]
    n_door_iter = int(
        optim_conf.get("heating_model_refit_door_relabel_iterations", _RC_DOOR_RELABEL_DEFAULT_ITERATIONS)
    )
    n_blind_iter = int(
        optim_conf.get("heating_model_refit_blind_relabel_iterations", _RC_BLIND_RELABEL_DEFAULT_ITERATIONS)
    )
    if door_relabel_enabled:
        df_door_only = _em_relabel_door_open_rc(
            df_raw.copy(),
            prepare_kwargs,
            dt_h,
            segment_len,
            n_door_iter,
            logger,
            regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )
        candidates.append(("door_only", _fit_score(df_door_only, n_rows)))
    if blind_relabel_enabled:
        df_blind_only = _em_relabel_blind_position_rc(
            df_raw.copy(),
            prepare_kwargs,
            dt_h,
            segment_len,
            n_blind_iter,
            logger,
            regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )
        candidates.append(("blind_only", _fit_score(df_blind_only, n_rows)))
    if door_relabel_enabled and blind_relabel_enabled:
        # Same order as the door_only/blind_only passes above (door first,
        # then blind) - tested empirically against the reverse order on
        # real data; door-first scored better, see command_line.py's own
        # git history for the comparison.
        df_both = _em_relabel_door_open_rc(
            df_raw.copy(),
            prepare_kwargs,
            dt_h,
            segment_len,
            n_door_iter,
            logger,
            regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )
        df_both = _em_relabel_blind_position_rc(
            df_both,
            prepare_kwargs,
            dt_h,
            segment_len,
            n_blind_iter,
            logger,
            regularization_overrides,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            phase_offsets=phase_offsets,
            warm_start_from=warm_start_from,
            fit_electric_power=fit_electric_power,
        )
        candidates.append(("both", _fit_score(df_both, n_rows)))

    if len(candidates) > 1:
        logger.info(
            "heating-model-refit: relabel comparison - %s",
            ", ".join(f"{label} val_mae={c['val_mae']:.3f}" for label, c in candidates),
        )
    relabel_source, chosen = min(candidates, key=lambda kv: kv[1]["val_mae"])

    val_mae = chosen["val_mae"]
    max_mae = float(optim_conf.get("heating_model_refit_max_mae_c", 1.5))
    if val_mae > max_mae:
        logger.error(
            "heating-model-refit: held-out validation MAE %.3f°C exceeds "
            "heating_model_refit_max_mae_c (%.3f°C) - keeping the previously "
            "deployed model, not overwriting.",
            val_mae,
            max_mae,
        )
        return {
            "deployed": False,
            "val_mae_c": val_mae,
            "max_mae_c": max_mae,
            "n_rows": n_rows,
            "relabel_source": relabel_source,
        }

    params_final = chosen["params_final"]
    fit_info = chosen["fit_info"]
    test_mae = chosen["test_mae"]

    params_dict = {name: float(value) for name, value in zip(PARAM_NAMES, params_final, strict=True)}
    deployed = await save_json_blob(
        emhass_conf,
        "thermal_physics_params.json",
        {
            "params": params_dict,
            "fit_info": fit_info,
            "val_mae_c": val_mae,
            "test_mae_c": test_mae,
            "source": "auto-tune" if warm_start_from is not None else "auto-refit",
            "relabel_source": relabel_source,
            "refit_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
            "window_days": window_days,
            "n_rows": n_rows,
        },
        logger,
        keep_previous=True,
    )
    result = {
        "deployed": deployed,
        "val_mae_c": val_mae,
        "test_mae_c": test_mae,
        "max_mae_c": max_mae,
        "n_rows": n_rows,
        "window_days": window_days,
        "relabel_source": relabel_source,
    }
    logger.info(
        "heating-model-refit: honest held-out test MAE (retrained on train+val, NEVER "
        "used for any deploy decision) - deployed=%s val_mae_c=%.3f test_mae_c=%s "
        "source=%s (n_rows=%d, window_days=%d)",
        deployed,
        val_mae,
        f"{test_mae:.3f}" if test_mae is not None else "n/a",
        relabel_source,
        n_rows,
        window_days,
    )
    return result


async def refit_heating_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit the thermal-mass physics model against fresh Home Assistant
    history and deploy it for heating-need-forecast to use.

    Thin wrapper around _run_heating_model_refit's own shared body -
    today's exact, unchanged behavior (warm_start_from=None: every fit in
    the chain uses the usual 3-restart hedge). See tune_heating_model for
    the cheaper, warm-started sibling.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when disabled/failed
    :rtype: dict | None
    """
    return await _run_heating_model_refit(input_data_dict, logger, warm_start_from=None)


async def tune_heating_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Cheaper sibling of refit_heating_model: warm-starts every fit in the
    chain from the currently-deployed thermal_physics_params.json instead
    of the usual 3-restart (default/fast/slow) hedge, collapsing each fit
    down to a single restart seeded at the current parameters (see
    _fit_temperature_params's own warm_start_from docstring for why this
    is safe to do here - we're already confident in the neighborhood,
    that's the whole point of tuning rather than blindly re-exploring).

    Same optim_conf prerequisites as refit_heating_model
    (heating_model_refit_enabled, use_influxdb, heatpump_indoor_temp_sensor)
    and the same heating_model_refit_max_mae_c deploy gate - tuning has
    identical prerequisites to refitting, no separate enable flag, matching
    tune_self_learning_physics_model's own precedent. Falls back to a full
    refit (warm_start_from=None) when nothing has ever been deployed yet -
    there's nothing to warm-start from on a room's very first fit.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when disabled/failed
    :rtype: dict | None
    """
    from emhass.thermal.thermal_mass_physics import PARAM_NAMES

    emhass_conf = input_data_dict["emhass_conf"]
    fitted = await load_json_blob(emhass_conf, "thermal_physics_params.json", logger, default=None)
    warm_start_from = None
    if fitted and "params" in fitted:
        try:
            warm_start_from = np.array([fitted["params"][name] for name in PARAM_NAMES], dtype=float)
        except KeyError as e:
            logger.warning(
                "heating-model-tune: thermal_physics_params.json is missing parameter %s - "
                "falling back to a full (non-warm-started) refit.",
                e,
            )
    else:
        logger.info(
            "heating-model-tune: no currently-deployed model found - falling back to a full refit "
            "(nothing to warm-start from on the very first fit)."
        )
    return await _run_heating_model_refit(input_data_dict, logger, warm_start_from=warm_start_from)


async def refit_load_quantile_spread_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit the load forecast's P10/P90 daily-spread ratios from the
    user's own historical load data.

    _get_historical_daily_load_spread (forecast.py) otherwise falls back
    to the generic bundled reference dataset (long_train_data.pkl) for
    this - a scale-invariant ratio, but still borrowed from whatever
    household that reference data was originally collected from, not the
    user's own day-to-day variability. This retrieves a long window of
    the user's own sensor_power_load_no_var_loads history and computes
    the same (month, day-of-week) -> (day-of-week, any month) -> no-op
    bucketed quantile-ratio cascade directly from it, once, so every
    subsequent live cycle's _get_historical_daily_load_spread call can
    prefer these real per-household ratios at zero extra retrieval cost -
    the same "refit once over a long window, reuse cheaply every cycle"
    pattern already used for pv_horizon_profile, thermal model
    parameters, etc.

    Persists load_quantile_spread.json ({"month_weekday_buckets": {...},
    "weekday_buckets": {...}, "n_days_total": N}), loaded into
    plant_conf["load_quantile_spread"] at the start of every cycle (see
    set_input_data_dict) - a bucket with too few days to trust
    (MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD) is simply absent, so
    _get_historical_daily_load_spread's own fallback to the generic
    reference dataset still applies per-bucket, not just before this
    refit has ever run.

    :param input_data_dict: The setup dictionary (needs retrieve_hass_conf,
        optim_conf, emhass_conf, rh).
    :param logger: The passed logger object.
    :return: The persisted result dict, or None on failure.
    :rtype: dict | None
    """
    from emhass.forecast import MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD

    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    optim_conf = input_data_dict["optim_conf"]
    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]

    load_sensor = retrieve_hass_conf.get("sensor_power_load_no_var_loads", "")
    if not load_sensor:
        logger.error("load-quantile-spread-refit: sensor_power_load_no_var_loads is not configured")
        return None

    window_days = int(optim_conf.get("load_quantile_spread_refit_window_days", 180))
    days_list = utils.get_days_list(window_days)
    if not await rh.get_data(days_list, [load_sensor]):
        logger.error("load-quantile-spread-refit: failed to retrieve history from Home Assistant/InfluxDB")
        return None
    if load_sensor not in rh.df_final.columns:
        logger.error("load-quantile-spread-refit: no data retrieved for %s", load_sensor)
        return None
    load = rh.df_final[load_sensor].dropna()
    if load.empty:
        logger.error("load-quantile-spread-refit: no valid load data in the fetched window")
        return None

    # groupby(date), not resample("D"): resample would silently insert a
    # fake 0.0-sum row for any calendar day absent from the data (a real
    # sensor-history gap), corrupting the bucket with fabricated
    # zero-consumption days - same discipline
    # _get_historical_daily_load_spread's own generic-reference fallback
    # already uses.
    daily_totals = load.groupby(load.index.date).sum()
    daily_totals.index = pd.DatetimeIndex(daily_totals.index)

    month_weekday_buckets: dict[str, dict] = {}
    weekday_buckets: dict[str, dict] = {}
    for weekday in range(7):
        weekday_bucket = daily_totals[daily_totals.index.dayofweek == weekday]
        if len(weekday_bucket) >= MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD:
            median = weekday_bucket.median()
            if median != 0:
                weekday_buckets[str(weekday)] = {
                    "p10_ratio": float(weekday_bucket.quantile(0.1) / median),
                    "p90_ratio": float(weekday_bucket.quantile(0.9) / median),
                    "n": int(len(weekday_bucket)),
                }
        for month in range(1, 13):
            month_bucket = weekday_bucket[weekday_bucket.index.month == month]
            if len(month_bucket) >= MIN_DAYS_FOR_LOAD_QUANTILE_SPREAD:
                median = month_bucket.median()
                if median != 0:
                    month_weekday_buckets[f"{month}_{weekday}"] = {
                        "p10_ratio": float(month_bucket.quantile(0.1) / median),
                        "p90_ratio": float(month_bucket.quantile(0.9) / median),
                        "n": int(len(month_bucket)),
                    }

    result = {
        "month_weekday_buckets": month_weekday_buckets,
        "weekday_buckets": weekday_buckets,
        "n_days_total": int(len(daily_totals)),
    }
    saved = await save_json_blob(emhass_conf, "load_quantile_spread.json", result, logger)
    if not saved:
        logger.error("load-quantile-spread-refit: failed to persist load_quantile_spread.json")
        return None
    return result


_PV_HORIZON_MIN_OBSERVATIONS = 40  # a small multiple of pv_shading_kalman's own per-anchor minimum


async def refit_pv_horizon_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Learn a per-direction PV shading/horizon profile from historical
    production, reusing the Kalman innovation-gate math already used for
    sensorless window/door detection (see pv_shading_kalman.py's own
    module docstring for why this is not a literal continuously-tracked
    Kalman state).

    Compares actual PV output (sensor_power_photovoltaics) against an
    unobstructed clear-sky PVLib simulation
    (Forecast._calculate_pvlib_power with apply_horizon_mask=False - it
    must never mask against a profile it may itself be in the middle of
    updating) driven by Open-Meteo historical weather
    (Forecast.get_historical_weather_from_open_meteo, added earlier this
    session), over a pv_horizon_refit_window_days window. The persisted
    profile (pv_horizon_profile.json) is only ever read at forecast time
    when pv_horizon_learning_enabled is true (see
    Forecast._apply_pv_horizon_mask) - this action always runs when
    invoked regardless of that flag, so a user can inspect a learned
    profile before switching it on.

    When sensor_power_photovoltaics_per_panel is configured (one entity_id
    per physical panel, e.g. optimizers/microinverters), this also learns a
    profile per panel - useful to localize a fixed, partial obstruction (a
    chimney affecting only some panels) that a single combined production
    sensor cannot distinguish from a full-width one. Diagnostics only (shown
    in the pv-horizon-refit result, not applied to the forecast); currently
    only supported with a single PV orientation group.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when failed
    :rtype: dict | None
    """
    from pvlib.irradiance import get_total_irradiance

    from emhass.pv_shading_kalman import (
        AZIMUTH_RENDER_SPACING_DEG,
        aggregate_horizon_profile,
        aggregate_partial_transmittance_surface,
        classify_hard_object_instants,
        classify_shaded_instants,
        compute_geometrically_blind_azimuths,
        compute_self_shading_curve,
        compute_sun_path_envelope,
        estimate_empirical_diffuse_transmission_factor,
    )

    optim_conf = input_data_dict["optim_conf"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    emhass_conf = input_data_dict["emhass_conf"]
    rh = input_data_dict["rh"]
    fcst = input_data_dict["fcst"]

    pv_sensor = retrieve_hass_conf.get("sensor_power_photovoltaics", "")
    if not pv_sensor:
        logger.error("pv-horizon-refit: sensor_power_photovoltaics is not configured")
        return None
    panel_sensors = [
        s for s in retrieve_hass_conf.get("sensor_power_photovoltaics_per_panel", []) if s
    ]

    window_days = int(optim_conf.get("pv_horizon_refit_window_days", 60))
    days_list = utils.get_days_list(window_days)
    if not await rh.get_data(days_list, [pv_sensor, *panel_sensors]):
        logger.error("pv-horizon-refit: failed to retrieve history from Home Assistant/InfluxDB")
        return None
    if pv_sensor not in rh.df_final.columns:
        logger.error("pv-horizon-refit: no data retrieved for %s", pv_sensor)
        return None
    actual = rh.df_final[pv_sensor].dropna()
    if actual.empty:
        logger.error("pv-horizon-refit: no valid PV production data in the fetched window")
        return None

    try:
        weather = await fcst.get_historical_weather_from_open_meteo(
            days_list, ["ghi", "dni", "dhi", "outdoor_temp", "wind_speed"]
        )
    except Exception:
        logger.error(
            "pv-horizon-refit: failed to fetch historical weather from Open-Meteo", exc_info=True
        )
        return None
    weather = weather.rename(columns={"outdoor_temp": "temp_air"})
    # Open-Meteo's historical archive is hourly-only - a plain nearest-value
    # reindex duplicates each hour's reading across every sub-hourly actual
    # production timestamp, producing an artificial step in the "expected"
    # clear-sky simulation exactly on the hour (real irradiance never jumps
    # like that). Interpolating linearly in time between the two surrounding
    # real hourly readings removes that step - it doesn't invent information
    # about what truly happened between those two readings (a brief passing
    # cloud can still be missed), but it stops a purely artificial
    # discontinuity from masquerading as a shading-relevant signal.
    combined_index = weather.index.union(actual.index)
    weather = (
        weather.reindex(combined_index).interpolate(method="time").reindex(actual.index).dropna()
    )

    common_index = actual.index.intersection(weather.index)
    if len(common_index) < _PV_HORIZON_MIN_OBSERVATIONS:
        logger.error(
            "pv-horizon-refit: only %d aligned rows of actual production + weather over %d "
            "days - too little to learn anything from.",
            len(common_index),
            window_days,
        )
        return None
    actual = actual.loc[common_index]
    weather = weather.loc[common_index]

    expected = fcst._calculate_pvlib_power(weather, apply_horizon_mask=False).reindex(common_index)
    # shaded (broad gate - any statistically significant deficit) feeds
    # the new partial-transmittance surface below; hard_blocked (strict,
    # >=95%-blocked gate) feeds aggregate_horizon_profile - the "solid
    # obstruction" horizon proper. See classify_hard_object_instants's own
    # docstring for why these are deliberately two different tests.
    shaded = classify_shaded_instants(actual, expected)
    hard_blocked = classify_hard_object_instants(actual, expected)
    angles = Forecast.compute_solar_angles(
        weather, retrieve_hass_conf["Latitude"], retrieve_hass_conf["Longitude"]
    )

    previous = await load_json_blob(emhass_conf, "pv_horizon_profile.json", logger, default=None)
    previous_profile = (previous or {}).get("profile")
    previous_partial_surface = (previous or {}).get("profile_partial_transmittance")
    forgetting_factor = float(optim_conf.get("pv_horizon_refit_forgetting_factor", 0.7))
    profile = aggregate_horizon_profile(
        hard_blocked,
        angles["solar_azimuth"],
        angles["solar_elevation"],
        actual,
        expected,
        previous_profile,
        forgetting_factor,
    )
    # Genuinely partial attenuation (a tree canopy letting a varying
    # fraction of light through depending on exactly where in its canopy
    # the sun sits) - a separate, additional 2D (azimuth x elevation)
    # layer applied on top of the hard-object horizon above, since a
    # single scalar transmittance per azimuth can't represent it. Gated
    # at apply-time against the sun's own real yearly elevation envelope
    # (persisted below), computed once here rather than on every forecast
    # call - it only depends on site latitude/longitude, not on anything
    # that changes between refits.
    partial_surface = aggregate_partial_transmittance_surface(
        shaded,
        hard_blocked,
        angles["solar_azimuth"],
        angles["solar_elevation"],
        actual,
        expected,
        previous_partial_surface,
        forgetting_factor,
    )
    sun_min_curve, sun_max_curve = compute_sun_path_envelope(
        retrieve_hass_conf["Latitude"], retrieve_hass_conf["Longitude"]
    )

    plant_conf = input_data_dict["plant_conf"]  # also reused below (n_groups etc.)
    self_shading_curve_combined = compute_self_shading_curve(
        plant_conf["surface_tilt"][0], plant_conf["surface_azimuth"][0], AZIMUTH_RENDER_SPACING_DEG
    )

    # Empirical diffuse-light attenuation: a real-data alternative to
    # compute_diffuse_transmission_factor's purely theoretical integral
    # over the direct-beam horizon (which can't reflect an obstruction
    # that affects the sky dome differently than the direct-beam model
    # implies). Separates how much of the modeled DIRECT vs. DIFFUSE POA
    # share is actually getting through via a plain regression over
    # instants already confirmed clear of any known direct-beam shading -
    # see estimate_empirical_diffuse_transmission_factor's own docstring.
    poa = get_total_irradiance(
        surface_tilt=plant_conf["surface_tilt"][0],
        surface_azimuth=plant_conf["surface_azimuth"][0],
        solar_zenith=90.0 - angles["solar_elevation"],
        solar_azimuth=angles["solar_azimuth"],
        dni=weather["dni"],
        ghi=weather["ghi"],
        dhi=weather["dhi"],
    )
    poa_global_safe = poa["poa_global"].replace(0.0, np.nan)
    direct_share = (poa["poa_direct"] / poa_global_safe).fillna(0.0)
    diffuse_share = ((poa["poa_sky_diffuse"] + poa["poa_ground_diffuse"]) / poa_global_safe).fillna(0.0)
    confirmed_clear = ~shaded
    previous_diffuse_factors = (previous or {}).get("diffuse_transmission_factor")
    diffuse_transmission_factor = estimate_empirical_diffuse_transmission_factor(
        actual,
        expected,
        direct_share,
        diffuse_share,
        confirmed_clear,
        previous_diffuse_factors,
        forgetting_factor,
    )

    # Per-panel diagnostics: localizes shading to specific panels (e.g. a
    # chimney affecting only some of them) instead of only the combined
    # system total. Two supported plant_conf shapes:
    # - One orientation group: panels are physically identical, so the
    #   system-wide unobstructed simulation already computed above
    #   (expected) divided by the group's module count approximates each
    #   panel's own unobstructed baseline - no separate PVLib run needed.
    # - One orientation-group entry per panel, index-matched to
    #   sensor_power_photovoltaics_per_panel (e.g. a plant configured as
    #   N identical single-module/single-microinverter entries so the
    #   combined forecast caps each panel at its own real inverter rating):
    #   each panel's own exact PVLib simulation is used instead of the
    #   divided approximation.
    # A panel not currently configured/sensored keeps whatever was last
    # persisted for it (same carry-forward philosophy as the season split
    # in aggregate_horizon_profile), rather than being dropped from the file.
    # Geometrically-blind azimuths: purely from tilt/azimuth/site
    # location, no measurement needed - see compute_geometrically_blind_azimuths's
    # own docstring for why this matters (an anchor that can never be
    # tested stays at its cold-start default forever, indistinguishable
    # from a confirmed-clear reading unless told apart ahead of time).
    # Computed at AZIMUTH_RENDER_SPACING_DEG (not the coarser fitting
    # spacing) since these sets are consumed only by render_horizon_polar_grid's
    # fine-resolution chart, never by the live forecast mask. Only
    # well-defined for the combined/aggregate chart when every panel
    # shares one orientation - a multi-group plant has no single "blind
    # for the system as a whole" set.
    n_groups = len(plant_conf["surface_azimuth"])
    blind_azimuths_combined = (
        compute_geometrically_blind_azimuths(
            plant_conf["surface_tilt"][0],
            plant_conf["surface_azimuth"][0],
            retrieve_hass_conf["Latitude"],
            retrieve_hass_conf["Longitude"],
            spacing_deg=AZIMUTH_RENDER_SPACING_DEG,
        )
        if n_groups == 1
        else None
    )

    profile_per_panel = (previous or {}).get("profile_per_panel", {})
    partial_surface_per_panel = (previous or {}).get("profile_partial_transmittance_per_panel", {})
    blind_azimuths_per_panel: dict[str, set[int]] = {}
    self_shading_curve_per_panel: dict[str, dict[float, float | None]] = {}
    if panel_sensors:
        panel_expected_map: dict[str, pd.Series] = {}
        if n_groups == 1:
            module_count = plant_conf["modules_per_string"][0] * plant_conf["strings_per_inverter"][0]
            shared_expected = (expected / module_count).reindex(common_index)
            panel_expected_map = dict.fromkeys(panel_sensors, shared_expected)
            blind_azimuths_per_panel = dict.fromkeys(panel_sensors, blind_azimuths_combined)
            self_shading_curve_per_panel = dict.fromkeys(panel_sensors, self_shading_curve_combined)
        elif n_groups == len(panel_sensors):
            cec_databases = fcst._load_cec_databases()
            for i, sensor in enumerate(panel_sensors):
                panel_expected_map[sensor] = fcst._calculate_pvlib_power_for_index(
                    weather, i, apply_horizon_mask=False, cec_databases=cec_databases
                ).reindex(common_index)
                blind_azimuths_per_panel[sensor] = compute_geometrically_blind_azimuths(
                    plant_conf["surface_tilt"][i],
                    plant_conf["surface_azimuth"][i],
                    retrieve_hass_conf["Latitude"],
                    retrieve_hass_conf["Longitude"],
                    spacing_deg=AZIMUTH_RENDER_SPACING_DEG,
                )
                self_shading_curve_per_panel[sensor] = compute_self_shading_curve(
                    plant_conf["surface_tilt"][i],
                    plant_conf["surface_azimuth"][i],
                    AZIMUTH_RENDER_SPACING_DEG,
                )
        else:
            logger.warning(
                "pv-horizon-refit: sensor_power_photovoltaics_per_panel has %d sensor(s) "
                "but the PV plant config has %d orientation group(s) - expected either "
                "exactly 1 (divided across panels) or exactly %d (one config entry per "
                "panel, index-matched) - skipping per-panel refit.",
                len(panel_sensors),
                n_groups,
                len(panel_sensors),
            )

        # Peer reference for localizing which panel is responsible when
        # something's off, independent of the weather-anchored expectation
        # above (and its Open-Meteo hourly resolution ceiling): the best-
        # performing panel right now is the best available proxy for what
        # unobstructed output currently looks like. Needs at least 2
        # present panels - with only 1, "max across panels" is just that
        # panel itself (every peer-ratio trivially 1.0), which would
        # silently disable per-panel shading detection entirely below.
        present_panels = [s for s in panel_sensors if s in rh.df_final.columns]
        peer_reference = (
            rh.df_final[present_panels].max(axis=1) if len(present_panels) >= 2 else None
        )

        for sensor, panel_expected in panel_expected_map.items():
            if sensor not in rh.df_final.columns:
                continue
            panel_actual = rh.df_final[sensor].dropna()
            panel_common = panel_actual.index.intersection(panel_expected.index)
            if len(panel_common) < _PV_HORIZON_MIN_OBSERVATIONS:
                continue
            # hard_blocked (not the broader classify_shaded_instants) to
            # stay consistent with aggregate_horizon_profile's own
            # "solid obstruction" semantics - same reasoning as the
            # combined profile above. panel_shaded (the broader gate)
            # feeds this panel's own partial-transmittance surface below,
            # mirroring the combined shaded/hard_blocked split exactly.
            panel_hard_blocked = classify_hard_object_instants(
                panel_actual.loc[panel_common], panel_expected.loc[panel_common]
            )
            panel_shaded = classify_shaded_instants(
                panel_actual.loc[panel_common], panel_expected.loc[panel_common]
            )
            if peer_reference is not None:
                peer_common = panel_common.intersection(peer_reference.dropna().index)
                if len(peer_common) >= _PV_HORIZON_MIN_OBSERVATIONS:
                    # A moment where every panel dips together (weather, or
                    # a slow-moving obstruction currently covering the
                    # whole array) fails this peer test for all of them -
                    # AND, not replace, so that case is correctly left to
                    # the combined/system-wide profile above instead of
                    # being misattributed to one specific panel. Applied to
                    # both the hard-object and the broader shaded gate, so
                    # panel_shaded & ~panel_hard_blocked (the partial-
                    # transmittance surface's own input, same shape as the
                    # combined case) isn't corrupted by a whole-array dip
                    # that only the broader gate would otherwise still see.
                    peer_hard_blocked = classify_hard_object_instants(
                        panel_actual.loc[peer_common], peer_reference.loc[peer_common]
                    )
                    peer_shaded = classify_shaded_instants(
                        panel_actual.loc[peer_common], peer_reference.loc[peer_common]
                    )
                    panel_hard_blocked = panel_hard_blocked & peer_hard_blocked.reindex(
                        panel_common, fill_value=False
                    )
                    panel_shaded = panel_shaded & peer_shaded.reindex(
                        panel_common, fill_value=False
                    )
            profile_per_panel[sensor] = aggregate_horizon_profile(
                panel_hard_blocked,
                angles["solar_azimuth"].loc[panel_common],
                angles["solar_elevation"].loc[panel_common],
                panel_actual.loc[panel_common],
                panel_expected.loc[panel_common],
                profile_per_panel.get(sensor),
                forgetting_factor,
            )
            partial_surface_per_panel[sensor] = aggregate_partial_transmittance_surface(
                panel_shaded,
                panel_hard_blocked,
                angles["solar_azimuth"].loc[panel_common],
                angles["solar_elevation"].loc[panel_common],
                panel_actual.loc[panel_common],
                panel_expected.loc[panel_common],
                partial_surface_per_panel.get(sensor),
                forgetting_factor,
            )

    saved = await save_json_blob(
        emhass_conf,
        "pv_horizon_profile.json",
        {
            "profile": profile,
            "profile_per_panel": profile_per_panel,
            "profile_partial_transmittance": partial_surface,
            "profile_partial_transmittance_per_panel": partial_surface_per_panel,
            # String-keyed (JSON object keys must be strings) - converted
            # back to numeric in Forecast._apply_pv_horizon_mask.
            "sun_path_envelope": {
                "min": {str(k): v for k, v in sun_min_curve.items()},
                "max": {str(k): v for k, v in sun_max_curve.items()},
            },
            "diffuse_transmission_factor": diffuse_transmission_factor,
            "last_refit_iso": pd.Timestamp.now(tz="UTC").isoformat(),
        },
        logger,
    )
    if not saved:
        logger.error("pv-horizon-refit: failed to persist pv_horizon_profile.json")
        return None

    logger.info("pv-horizon-refit: updated horizon profile (%d anchors)", len(profile))
    result = {
        "pv_horizon_profile": profile,
        "n_shaded_instants": int(shaded.sum()),
        "n_observations": len(common_index),
        "blind_azimuths_combined": blind_azimuths_combined,
        "pv_horizon_partial_transmittance": partial_surface,
        "self_shading_curve_combined": self_shading_curve_combined,
        "sun_path_envelope": (sun_min_curve, sun_max_curve),
        "diffuse_transmission_factor": diffuse_transmission_factor,
    }
    if profile_per_panel:
        result["pv_horizon_profile_per_panel"] = profile_per_panel
        result["blind_azimuths_per_panel"] = blind_azimuths_per_panel
        result["self_shading_curve_per_panel"] = self_shading_curve_per_panel
        result["pv_horizon_partial_transmittance_per_panel"] = partial_surface_per_panel
    return result


# Forward-accumulating per-model scoring for the ensemble-derived PV P10
# estimate: 0.7 mirrors pv_horizon_refit_forgetting_factor's own established
# "shouldn't take forever to reflect reality, shouldn't be noisy either"
# reasoning (both update roughly once a day, unlike a live per-cycle RLS
# update, which would want something much closer to 1.0). Prediction target
# ~24h out, logged at most once every 20h per model - naive-mpc-optim alone
# could otherwise run every few minutes.
_PV_ENSEMBLE_SCORE_FORGETTING_FACTOR = 0.7
_PV_ENSEMBLE_PREDICTION_HORIZON_HOURS = 24
_PV_ENSEMBLE_MIN_LOG_INTERVAL_HOURS = 20


def _pinball_loss(actual: float, predicted: float, quantile: float) -> float:
    """Pinball (quantile) loss for one prediction at one quantile level.

    Averaging this over several quantiles (10/50/90 here) is a standard,
    literature-supported approximation of CRPS (Continuous Ranked
    Probability Score) - the usual metric for comparing a model's whole
    predictive distribution against a single observation, not just its
    point/mean forecast.
    """
    diff = actual - predicted
    return max(quantile * diff, (quantile - 1) * diff)


async def _update_pv_ensemble_model_scores(
    fcst, rh, retrieve_hass_conf: dict, emhass_conf: dict, logger: logging.Logger
) -> None:
    """Forward-accumulating per-model accuracy tracker for the Open-Meteo
    ensemble-derived PV P10 estimate (see
    Forecast._get_pv_p10_weather_from_ensemble/_select_percentile_member_weather).

    Open-Meteo's Ensemble API does not retain historical member data -
    confirmed empirically this session (past_days up to 90 and explicit
    past start_date all return null for every model/variable, even a
    single week back) - so retroactive backtesting against it is
    impossible. This instead logs each candidate model's own P10/P50/P90
    (from that model's own members alone, unweighted - see
    _select_percentile_member_weather) for ~pv_ensemble_prediction_horizon_hours
    out, resolves all three once real production data is available, and
    blends the resulting accuracy into a slow, per-model rolling score -
    the same forgetting-factor-blend shape self_learning_physics_refit's
    own RLS update already uses elsewhere in this codebase, just for a
    scalar per-model score instead of a fitted coefficient vector.

    Scored via CRPS (Continuous Ranked Probability Score), approximated as
    the mean of the pinball loss at the 10th/50th/90th percentile
    (_pinball_loss) - not a single point/mean forecast's error. A model can
    have an accurate mean forecast but a poorly-calibrated spread (or vice
    versa); since this score is what weights that model's contribution to
    the pooled P10 selection, it needs to reflect the model's whole
    predictive distribution, not just its average.

    Scored on PV *power*, not raw irradiance - every EMHASS PV install
    already has sensor_power_photovoltaics, unlike a local irradiance
    sensor (this session's own motivating case). A model with no score
    yet starts from a neutral 0.5 prior on its first resolution, not 0.0 -
    untested isn't "known bad", it's unknown.

    Persists pv_ensemble_model_scores.json ({"pending": [...], "scores":
    {...}}) for the *next* cycle's plant_conf["pv_ensemble_model_weights"]
    load (set_input_data_dict, right before this cycle's own
    Forecast(...) construction) to pick up - this function needs `fcst`
    itself (for _calculate_pvlib_power on each candidate model's own bare
    forecast), so it can only run after Forecast already exists, one
    cycle later than the weights it produces.

    :param fcst: The just-constructed Forecast instance for this cycle.
    :param rh: The live RetrieveHass instance for this cycle.
    :param retrieve_hass_conf: Live-HA config dict (for sensor_power_photovoltaics).
    :param emhass_conf: Dictionary containing the needed emhass paths.
    :param logger: The passed logger object.
    :rtype: None
    """
    from emhass.forecast import (
        PV_ENSEMBLE_CANDIDATE_MODELS,
        _parse_pv_ensemble_member_arrays,
        _select_percentile_member_weather,
    )
    from emhass.pv_shading_kalman import MIN_EXPECTED_POWER_W

    pv_sensor = retrieve_hass_conf.get("sensor_power_photovoltaics", "")
    if not pv_sensor:
        logger.warning("pv-ensemble-scoring: sensor_power_photovoltaics is not configured, skipping")
        return

    state = await load_json_blob(
        emhass_conf, "pv_ensemble_model_scores.json", logger, default=None
    ) or {}
    scores: dict[str, float] = dict(state.get("scores", {}))
    pending: list[dict] = state.get("pending", [])

    now = pd.Timestamp.now(tz="UTC")
    matured = [p for p in pending if pd.Timestamp(p["target_iso"]) <= now]
    still_pending = [p for p in pending if pd.Timestamp(p["target_iso"]) > now]

    if matured:
        actual_series = None
        if await rh.get_data(utils.get_days_list(2), [pv_sensor]) and pv_sensor in rh.df_final.columns:
            actual_series = rh.df_final[pv_sensor].dropna()
        for entry in matured:
            model = entry.get("model")
            target_ts = pd.Timestamp(entry["target_iso"])
            predicted_p10 = entry.get("predicted_p10")
            predicted_p50 = entry.get("predicted_p50")
            predicted_p90 = entry.get("predicted_p90")
            actual = None
            if actual_series is not None and not actual_series.empty:
                pos = actual_series.index.get_indexer(
                    [target_ts], method="nearest", tolerance=pd.Timedelta("1h")
                )[0]
                if pos != -1:
                    actual = float(actual_series.iloc[pos])
            if actual is None or None in (predicted_p10, predicted_p50, predicted_p90):
                logger.debug(
                    "pv-ensemble-scoring: no actual production, or missing quantile "
                    "predictions (stale pre-CRPS schema?), near %s for %s's pending "
                    "prediction - dropping it unresolved.",
                    target_ts,
                    model,
                )
                continue
            crps = np.mean(
                [
                    _pinball_loss(actual, predicted_p10, 0.10),
                    _pinball_loss(actual, predicted_p50, 0.50),
                    _pinball_loss(actual, predicted_p90, 0.90),
                ]
            )
            normalized_crps = crps / max(actual, MIN_EXPECTED_POWER_W)
            prior = scores.get(model, 0.5)
            scores[model] = _PV_ENSEMBLE_SCORE_FORGETTING_FACTOR * prior + (
                1.0 - _PV_ENSEMBLE_SCORE_FORGETTING_FACTOR
            ) * (1.0 - min(1.0, normalized_crps))

    last_logged_by_model: dict[str, pd.Timestamp] = {}
    for entry in still_pending:
        logged_ts = pd.Timestamp(entry["logged_iso"])
        model = entry.get("model")
        if model not in last_logged_by_model or logged_ts > last_logged_by_model[model]:
            last_logged_by_model[model] = logged_ts

    target_ts = now + pd.Timedelta(hours=_PV_ENSEMBLE_PREDICTION_HORIZON_HOURS)
    for model in PV_ENSEMBLE_CANDIDATE_MODELS:
        last = last_logged_by_model.get(model)
        if last is not None and (now - last) < pd.Timedelta(hours=_PV_ENSEMBLE_MIN_LOG_INTERVAL_HOURS):
            continue
        data = await fcst._fetch_pv_ensemble_model_json(model, forecast_days=2)
        if data is None:
            continue
        hourly = data.get("hourly")
        if not hourly or "time" not in hourly:
            continue
        try:
            times = pd.to_datetime(hourly["time"], unit="s", utc=True)
            pos = int(np.abs((times - target_ts).total_seconds()).argmin())
            model_arrays = _parse_pv_ensemble_member_arrays(hourly)
            if model_arrays is None:
                logger.warning(
                    "pv-ensemble-scoring: model %s returned no usable member columns, skipping it",
                    model,
                )
                continue
            row_arrays = {var: arr[[pos], :] for var, arr in model_arrays.items()}
            n_members = row_arrays["ghi"].shape[1]
            uniform_weights = np.ones(n_members)
            row_index = [times[pos]]
            predicted = {}
            for label, quantile in (("predicted_p10", 10.0), ("predicted_p50", 50.0), ("predicted_p90", 90.0)):
                # Uniform weights: this evaluates model's own spread in
                # isolation, not the pooled cross-model selection.
                weather_row = _select_percentile_member_weather(
                    row_arrays, uniform_weights, quantile, row_index
                )
                predicted[label] = float(fcst._calculate_pvlib_power(weather_row).iloc[0])
        except (KeyError, IndexError, TypeError, ValueError):
            logger.warning(
                "pv-ensemble-scoring: could not build a P10/P50/P90 forecast row for %s",
                model,
                exc_info=True,
            )
            continue
        still_pending.append(
            {
                "model": model,
                "target_iso": times[pos].isoformat(),
                **predicted,
                "logged_iso": now.isoformat(),
            }
        )

    await save_json_blob(
        emhass_conf,
        "pv_ensemble_model_scores.json",
        {"pending": still_pending, "scores": scores},
        logger,
    )


# Standalone sibling of _REFIT_SENSOR_COLUMN_MAP/_REFIT_MIN_ROWS above, for a
# different model (emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR)
# predicting electric power and gas consumption instead of indoor
# temperature. Unlike the physics refit, the sensors this model is built
# around are hard-required (see _HYBRID_HP_REQUIRED_SENSORS below) - fitting
# on a defaulted-to-0 duty/target column would silently produce a garbage
# model rather than a gracefully degraded one.
_HYBRID_HP_SENSOR_COLUMN_MAP = {
    "heatpump_indoor_temp_sensor": "room_temp",
    "heatpump_power_sensor": "electric_power",
    "heatpump_gas_meter_sensor": "gas_consumption",
    "heatpump_duty_sensor": "heatpump_duty",
    "heatpump_flow_temp_sensor": "supply_temp",
    "heatpump_outdoor_temp_sensor": "outdoor_temp",
    "heatpump_weather_wind_speed_sensor": "wind_speed",
    "heatpump_weather_ghi_sensor": "ghi",
}
# heatpump_gas_meter_sensor is deliberately NOT required: a pure-electric
# system (no gas boiler) has nothing to put there. Its absence is what
# decides electric_only mode below - fitting HybridHeatPumpLR on an
# all-zero gas target would otherwise crash (a single-class y is not
# fittable by sklearn's LogisticRegression), so electric_only skips the gas
# model entirely rather than fitting it on fabricated/defaulted data.
_HYBRID_HP_REQUIRED_SENSORS = (
    "heatpump_indoor_temp_sensor",
    "heatpump_power_sensor",
    "heatpump_duty_sensor",
)
_HYBRID_HP_MIN_ROWS = 500  # same rationale as _REFIT_MIN_ROWS
_HYBRID_HP_MIN_GAS_POSITIVE_ROWS = 50  # well above _HurdleGasModel's own <5 degraded-fallback


def _hybrid_heatpump_solar_features(
    df: pd.DataFrame, latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ghi_norm/sun_alt_sin for HybridHeatPumpLR's feature builder.

    Deliberately wired up for real here - the offline benchmark that
    validated HybridHeatPumpLR's reported accuracy (scripts/compare_ensemble.py)
    never populated these two columns, so build_heatpump_features' internal
    defaulting silently left solar_offset at 0 for every row of that
    comparison. Enabling them here is a conscious deviation, confirmed with
    the user: the cited benchmark numbers no longer strictly describe this
    configuration's accuracy.

    ghi_norm is normalised against a fixed 1000 W/m2 reference (matching
    emhass.thermal.feature_engineering's own fallback constant), not a
    training-window max - a window max is a moving target that would blow up
    on a short or mostly-cloudy refit window.
    """
    from emhass.thermal.thermal_mass_physics import _compute_sun_direction_features

    if "ghi" in df.columns:
        ghi = pd.to_numeric(df["ghi"], errors="coerce").fillna(0.0)
    else:
        ghi = pd.Series(0.0, index=df.index, dtype=float)
    ghi_norm = (ghi / 1000.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    sun_alt_sin, _, _, _ = _compute_sun_direction_features(df, latitude=latitude, longitude=longitude)
    return ghi_norm, sun_alt_sin.to_numpy(dtype=float)


async def refit_hybrid_heatpump_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit the physics-informed heat pump electric (+ optional gas) model
    against fresh Home Assistant history and deploy it for
    hybrid-heatpump-forecast to use.

    Predicts electric power (W) and, when heatpump_gas_meter_sensor is
    configured, gas consumption (m3/interval) for a hybrid (electric heat
    pump + gas boiler) system - see
    emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR. With no gas meter
    sensor configured, fits in electric_only mode instead (for a pure
    electric heat pump, no gas boiler) - not gated on heatpump_is_hybrid,
    since that flag being off is exactly the pure-electric case this mode
    exists for. Standalone and fully isolated from
    refit_heating_model/optimization.py: EMHASS's live dispatch has no gas/
    electric split decision to plug into (heatpump_duty, a required input
    feature here, is itself what the optimizer would be solving for), so
    this only produces an informational forecast, published by
    compute_hybrid_heatpump_forecast below - it never influences dispatch.

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

    if not optim_conf.get("hybrid_heatpump_refit_enabled", False):
        logger.debug("hybrid-heatpump-model-refit: disabled (hybrid_heatpump_refit_enabled=False)")
        return None
    electric_only = not str(retrieve_hass_conf.get("heatpump_gas_meter_sensor", "") or "").strip()
    if not retrieve_hass_conf.get("use_influxdb", False):
        logger.error(
            "hybrid-heatpump-model-refit: use_influxdb is not enabled. The refit window "
            "(hybrid_heatpump_refit_window_days) is normally far longer than Home "
            "Assistant's own recorder retention - configure InfluxDB rather than "
            "risk silently fitting on a truncated REST window."
        )
        return None

    sensor_map: dict[str, str] = {}
    for conf_key in _HYBRID_HP_REQUIRED_SENSORS:
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if not entity_id:
            logger.error("hybrid-heatpump-model-refit: %s is not configured", conf_key)
            return None
        sensor_map[entity_id] = _HYBRID_HP_SENSOR_COLUMN_MAP[conf_key]
    for conf_key, column in _HYBRID_HP_SENSOR_COLUMN_MAP.items():
        if conf_key in _HYBRID_HP_REQUIRED_SENSORS:
            continue
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if entity_id:
            sensor_map[entity_id] = column
        else:
            logger.warning(
                "hybrid-heatpump-model-refit: %s is not configured - '%s' will use its "
                "static default for this refit.",
                conf_key,
                column,
            )

    from emhass.thermal.hybrid_heatpump_lr import HybridHeatPumpLR

    window_days = int(optim_conf.get("hybrid_heatpump_refit_window_days", 60))
    days_list = utils.get_days_list(window_days)
    if not await rh.get_data(days_list, list(sensor_map.keys())):
        logger.error("hybrid-heatpump-model-refit: failed to retrieve history from Home Assistant/InfluxDB")
        return None

    df_raw = rh.df_final.rename(columns=sensor_map)
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df_raw = await _fill_missing_weather_from_open_meteo(
        df_raw,
        retrieve_hass_conf,
        days_list,
        {"outdoor_temp", "wind_speed", "ghi"},
        input_data_dict["fcst"],
        logger,
    )
    required_cols = ["room_temp", "electric_power", "heatpump_duty"]
    if not electric_only:
        required_cols.append("gas_consumption")
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        logger.error(
            "hybrid-heatpump-model-refit: no data retrieved from InfluxDB for required column(s): %s",
            ", ".join(missing),
        )
        return None
    df_raw = df_raw.dropna(subset=required_cols)
    n_rows = len(df_raw)
    if n_rows < _HYBRID_HP_MIN_ROWS:
        logger.error(
            "hybrid-heatpump-model-refit: only %d complete data points retrieved over %d "
            "days (need at least %d) - aborting rather than fitting on too little data.",
            n_rows,
            window_days,
            _HYBRID_HP_MIN_ROWS,
        )
        return None

    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    dt_hours = _infer_timestep_hours(df_raw.index)
    df_raw["electric_power"] = utils.resolve_incremental_series(
        df_raw["electric_power"], "electric_power", logger, rate_dt_hours=dt_hours
    )

    n_gas_positive = None
    if not electric_only:
        df_raw["gas_consumption"] = utils.resolve_incremental_series(
            df_raw["gas_consumption"], "gas_consumption", logger
        )
        n_gas_positive = int((df_raw["gas_consumption"] > 0).sum())
        if n_gas_positive < _HYBRID_HP_MIN_GAS_POSITIVE_ROWS:
            logger.error(
                "hybrid-heatpump-model-refit: only %d positive gas-consumption rows retrieved "
                "over %d days (need at least %d) - aborting rather than deploying a gas model "
                "that would just predict a constant mean.",
                n_gas_positive,
                window_days,
                _HYBRID_HP_MIN_GAS_POSITIVE_ROWS,
            )
            return None

    ghi_norm, sun_alt_sin = _hybrid_heatpump_solar_features(
        df_raw,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    df_raw = df_raw.assign(ghi_norm=ghi_norm, sun_alt_sin=sun_alt_sin)

    # Chronological holdout split, used only to score whether the full-window
    # fit below is trustworthy enough to deploy - the deployed model itself
    # is refit on the full window afterward, once the gate passes.
    split_idx = max(1, int(round(n_rows * 0.8)))
    df_train, df_holdout = df_raw.iloc[:split_idx], df_raw.iloc[split_idx:]
    if len(df_holdout) < 10:
        logger.error(
            "hybrid-heatpump-model-refit: too few holdout rows (%d) after an 80/20 "
            "chronological split of %d rows - aborting.",
            len(df_holdout),
            n_rows,
        )
        return None

    y_gas_train = None if electric_only else df_train["gas_consumption"].to_numpy()
    probe_model = HybridHeatPumpLR(electric_only=electric_only)
    probe_model.fit(df_train, df_train["electric_power"].to_numpy(), y_gas_train)
    elec_pred, gas_pred = probe_model.predict(df_holdout)
    electric_mae = float(np.mean(np.abs(elec_pred - df_holdout["electric_power"].to_numpy())))
    gas_mae = None
    if not electric_only:
        gas_mae = float(np.mean(np.abs(gas_pred - df_holdout["gas_consumption"].to_numpy())))

    max_electric_mae = float(optim_conf.get("hybrid_heatpump_refit_max_electric_mae_w", 150.0))
    max_gas_mae = float(optim_conf.get("hybrid_heatpump_refit_max_gas_mae_m3", 0.02))
    result = {
        "electric_only": electric_only,
        "electric_mae_w": electric_mae,
        "max_electric_mae_w": max_electric_mae,
        "gas_mae_m3": gas_mae,
        "max_gas_mae_m3": max_gas_mae if not electric_only else None,
        "n_rows": n_rows,
        "n_gas_positive": n_gas_positive,
        "window_days": window_days,
    }
    fit_too_bad = electric_mae > max_electric_mae or (not electric_only and gas_mae > max_gas_mae)
    if fit_too_bad:
        logger.error(
            "hybrid-heatpump-model-refit: fit MAE electric=%.2fW (max %.2fW) gas=%s "
            "(max %s) - keeping the previously deployed model, not overwriting.",
            electric_mae,
            max_electric_mae,
            "n/a (electric_only)" if electric_only else f"{gas_mae:.5f}m3",
            "n/a" if electric_only else f"{max_gas_mae:.5f}m3",
        )
        result["deployed"] = False
        return result

    y_gas_full = None if electric_only else df_raw["gas_consumption"].to_numpy()
    final_model = HybridHeatPumpLR(electric_only=electric_only)
    final_model.fit(df_raw, df_raw["electric_power"].to_numpy(), y_gas_full)
    deployed = await save_pickle_blob(
        emhass_conf, "hybrid_heatpump_lr_model.pkl", final_model, logger, keep_previous=True
    )
    result["deployed"] = deployed
    logger.info(
        "hybrid-heatpump-model-refit: deployed=%s electric_only=%s electric_mae_w=%.2f "
        "gas_mae_m3=%s (n_rows=%d, window_days=%d)",
        deployed,
        electric_only,
        electric_mae,
        "n/a" if electric_only else f"{gas_mae:.5f}",
        n_rows,
        window_days,
    )
    return result


def _resolve_aggregate_duty_trajectory(
    input_data_dict: dict,
    forecast_index: pd.DatetimeIndex,
    fallback_duty: float,
    logger: logging.Logger,
) -> pd.Series:
    """Resolve a per-timestep heat pump duty trajectory for a forecast
    horizon, preferring the latest solved dispatch plan (utils.
    compute_aggregate_heatpump_duty on opt_res_latest.csv) over a single
    frozen sensor reading.

    Falls back to `fallback_duty` held constant across the whole horizon
    whenever no solved plan exists yet, no room/dispatch loads are
    configured, or heatpump_nominal_power isn't set - this keeps the
    function usable (if less precise) for setups that haven't run an
    optimization yet, or that dispatch the heat pump some other way.
    """
    passed_data = input_data_dict["params"].get("passed_data", {})
    room_load_indices = passed_data.get("room_load_indices", {})
    dispatch_load_index = passed_data.get("heatpump_dispatch_load_index")
    heatpump_nominal_power = float(
        input_data_dict.get("plant_conf", {}).get("heatpump_nominal_power", 0.0) or 0.0
    )

    if (room_load_indices or dispatch_load_index is not None) and heatpump_nominal_power > 0:
        opt_res_latest = _load_opt_res_latest(input_data_dict, logger, save_data_to_file=False)
        if opt_res_latest is not None and not opt_res_latest.empty:
            aggregate_duty = utils.compute_aggregate_heatpump_duty(
                opt_res_latest, room_load_indices, dispatch_load_index, heatpump_nominal_power
            )
            return aggregate_duty.reindex(forecast_index, method="nearest").fillna(fallback_duty)

    return pd.Series(fallback_duty, index=forecast_index)


async def compute_hybrid_heatpump_forecast(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Forecast electric power (and gas consumption, unless the deployed model
    was fit in electric_only mode) forward from now, using the fitted heat
    pump model (see refit_hybrid_heatpump_model above).

    Informational only, same "publish only" pattern as compute_heating_forecast -
    EMHASS never calls a device service here. Heat pump duty is resolved via
    utils.compute_aggregate_heatpump_duty from the latest solved dispatch plan
    (opt_res_latest.csv) when one exists - a real, multi-room-aware duty
    trajectory instead of a single frozen reading - falling back to the last
    observed heatpump_duty_sensor value held constant when no solved plan is
    available yet. Indoor/supply temperature are still held constant (no
    per-room planned-temperature aggregate exists to use instead). The model's
    electric_power_lag1/gas_consumption_lag1 features are resolved via an
    explicit per-step autoregressive loop (each step's own prediction feeds
    the next step's lag-1 input) rather than one batch predict() call, since
    a batch call over rows with no real electric_power/gas_consumption
    history would silently zero those lag features for the whole horizon.

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

    if not optim_conf.get("hybrid_heatpump_forecast_enabled", False):
        logger.debug("hybrid-heatpump-forecast: disabled (hybrid_heatpump_forecast_enabled=False)")
        return None

    model = await load_pickle_blob(emhass_conf, "hybrid_heatpump_lr_model.pkl", logger, default=None)
    if model is None:
        logger.error(
            "hybrid-heatpump-forecast: no fitted model found (data/hybrid_heatpump_lr_model.pkl). "
            "Run the hybrid-heatpump-model-refit action at least once."
        )
        return None

    live_sensor_keys = [
        "heatpump_duty_sensor",
        "heatpump_indoor_temp_sensor",
        "heatpump_flow_temp_sensor",
        "heatpump_power_sensor",
        "heatpump_gas_meter_sensor",
    ]
    live_entities = [retrieve_hass_conf.get(k, "") for k in live_sensor_keys]
    live_entities = [e for e in live_entities if e]
    if not live_entities:
        logger.error("hybrid-heatpump-forecast: no live sensors configured to read the current state from")
        return None

    days_list = utils.get_days_list(2)
    if not await rh.get_data(days_list, live_entities):
        logger.error("hybrid-heatpump-forecast: failed to retrieve live sensor data from Home Assistant")
        return None
    rh.prepare_data(
        live_entities[0],
        load_negative=False,
        set_zero_min=False,
        var_replace_zero=[],
        var_interp=live_entities,
        skip_renaming=True,
    )

    def _last_value(conf_key: str, default: float) -> float:
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if not entity_id or entity_id not in rh.df_final.columns:
            return default
        series = rh.df_final[entity_id].dropna()
        return float(series.iloc[-1]) if not series.empty else default

    def _last_delta_value(conf_key: str, default: float, rate_dt_hours: float | None = None) -> float:
        # Same cumulative-meter detection as the refit's own training data
        # (utils.resolve_incremental_series) - a raw gas/energy totalizer's
        # bare last value would otherwise seed the model with a huge,
        # out-of-distribution "gas/electric used this step" reading.
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if not entity_id or entity_id not in rh.df_final.columns:
            return default
        series = rh.df_final[entity_id].dropna()
        if series.empty:
            return default
        delta = utils.resolve_incremental_series(
            series, conf_key, logger, rate_dt_hours=rate_dt_hours
        )
        return float(delta.iloc[-1])

    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    live_dt_hours = _infer_timestep_hours(rh.df_final.index)
    last_duty = _last_value("heatpump_duty_sensor", 0.0)
    last_room_temp = _last_value("heatpump_indoor_temp_sensor", 20.0)
    last_supply_temp = _last_value("heatpump_flow_temp_sensor", 25.0)
    last_electric = _last_delta_value("heatpump_power_sensor", 0.0, rate_dt_hours=live_dt_hours)
    last_gas = _last_delta_value("heatpump_gas_meter_sensor", 0.0)

    df_weather = await input_data_dict["fcst"].get_weather_forecast(
        method=optim_conf.get("weather_forecast_method", "open-meteo")
    )
    if isinstance(df_weather, bool) and not df_weather:
        logger.error("hybrid-heatpump-forecast: failed to retrieve a weather forecast")
        return None
    if df_weather is None or len(df_weather) == 0:
        logger.error("hybrid-heatpump-forecast: weather forecast is empty")
        return None

    duty_trajectory = _resolve_aggregate_duty_trajectory(
        input_data_dict, df_weather.index, last_duty, logger
    )

    df_forecast = pd.DataFrame(
        {
            "outdoor_temp": df_weather["temp_air"],
            "wind_speed": df_weather["wind_speed"],
            "ghi": df_weather["ghi"],
            "heatpump_duty": duty_trajectory,
            "room_temp": last_room_temp,
            "supply_temp": last_supply_temp,
        },
        index=df_weather.index,
    )
    ghi_norm, sun_alt_sin = _hybrid_heatpump_solar_features(
        df_forecast,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    df_forecast = df_forecast.assign(ghi_norm=ghi_norm, sun_alt_sin=sun_alt_sin)

    n = len(df_forecast)
    elec_preds = np.zeros(n)
    gas_preds = np.zeros(n)
    # Anchor row: represents the last known real state, seeding step 0's lag-1
    # features. Its non-target columns are discarded (only electric_power/
    # gas_consumption from this row are ever read, via shift(1)).
    prev_row = df_forecast.iloc[[0]].copy()
    prev_row["electric_power"] = last_electric
    prev_row["gas_consumption"] = last_gas
    for i in range(n):
        current_row = df_forecast.iloc[[i]].copy()
        current_row["electric_power"] = 0.0
        current_row["gas_consumption"] = 0.0
        window_df = pd.concat([prev_row, current_row])
        elec_pred_arr, gas_pred_arr = model.predict(window_df)
        elec_preds[i] = elec_pred_arr[-1]
        gas_preds[i] = gas_pred_arr[-1]
        prev_row = current_row.copy()
        prev_row["electric_power"] = elec_preds[i]
        prev_row["gas_consumption"] = gas_preds[i]

    electric_only = model.gas_model_ is None

    passed_data = input_data_dict["params"]["passed_data"]
    electric_entity = passed_data.get("custom_hybrid_electric_forecast_id")
    gas_entity = passed_data.get("custom_hybrid_gas_forecast_id")
    if electric_entity is None or (not electric_only and gas_entity is None):
        logger.error(
            "hybrid-heatpump-forecast: target entities not registered "
            "(hybrid_heatpump_forecast_enabled was True at optim time but isn't now?)"
        )
        return None

    common_kwargs = {
        "publish_prefix": passed_data.get("publish_prefix", ""),
        "save_entities": False,
        "dont_post": passed_data.get("dont_post", False),
    }
    await rh.post_data(
        pd.Series(elec_preds, index=df_forecast.index),
        0,
        electric_entity["entity_id"],
        electric_entity["device_class"],
        electric_entity["unit_of_measurement"],
        electric_entity["friendly_name"],
        type_var="power",
        **common_kwargs,
    )
    if not electric_only:
        # No "gas" type_var exists in RetrieveHass.post_data; "energy" is the
        # closest existing publish shape that still carries the full forecast
        # horizon as an attribute list (like "power"/"temperature" do), rather
        # than falling through to the generic single-value else branch.
        await rh.post_data(
            pd.Series(gas_preds, index=df_forecast.index),
            0,
            gas_entity["entity_id"],
            gas_entity["device_class"],
            gas_entity["unit_of_measurement"],
            gas_entity["friendly_name"],
            type_var="energy",
            **common_kwargs,
        )

    result = {
        "electric_only": electric_only,
        "forecast_steps": n,
        "last_duty": last_duty,
        "last_electric_power_w": last_electric,
        "last_gas_consumption_m3": None if electric_only else last_gas,
        "mean_electric_forecast_w": float(np.mean(elec_preds)),
        "mean_gas_forecast_m3": None if electric_only else float(np.mean(gas_preds)),
    }
    await save_json_blob(emhass_conf, "hybrid_heatpump_forecast_last_run.json", result, logger)
    logger.info(
        "hybrid-heatpump-forecast: electric_only=%s mean_electric_forecast_w=%.1f mean_gas_forecast_m3=%s",
        electric_only,
        result["mean_electric_forecast_w"],
        "n/a" if electric_only else f"{result['mean_gas_forecast_m3']:.5f}",
    )
    # Added AFTER the JSON persist above (orjson.dumps can't serialize a
    # Series) - the forecasted curves themselves, rendered by
    # get_injection_dict_thermal_models/get_forecast_trend_plot_html.
    result["electric_forecast_series"] = pd.Series(elec_preds, index=df_forecast.index)
    if not electric_only:
        result["gas_forecast_series"] = pd.Series(gas_preds, index=df_forecast.index)
    return result


# Standalone sibling of _HYBRID_HP_SENSOR_COLUMN_MAP above, for the
# multi-room self-learning-physics model (emhass.thermal.self_learning_physics.
# SelfLearningPhysicsModel). room_temp is deliberately NOT in this map - it's
# resolved per room from heatpump_room_temp_sensors (see
# _resolve_room_temp_entity_map), not a single whole-house sensor.
_SELF_LEARNING_PHYSICS_SENSOR_COLUMN_MAP = {
    "heatpump_power_sensor": "electric_power",
    "heatpump_gas_meter_sensor": "gas_consumption",
    "heatpump_duty_sensor": "heatpump_duty",
    "heatpump_flow_temp_sensor": "supply_temp",
    "heatpump_outdoor_temp_sensor": "outdoor_temp",
    "heatpump_weather_wind_speed_sensor": "wind_speed",
    "heatpump_weather_dni_sensor": "dni",
    "heatpump_weather_dhi_sensor": "dhi",
}
_SELF_LEARNING_PHYSICS_REQUIRED_SENSORS = ("heatpump_power_sensor", "heatpump_duty_sensor")
_SELF_LEARNING_PHYSICS_MIN_ROWS = 500  # same rationale as _REFIT_MIN_ROWS/_HYBRID_HP_MIN_ROWS
# Coarse noise floor for undeclared-pair candidate-coupling suggestions (see
# refit_self_learning_physics_model's candidate-probe pass below) - well
# under a typical real conductance (the shipped test/example values sit in
# the 0.05-0.6 kW/K range) but high enough to filter out near-zero fit
# noise. This is deliberately a coarse heuristic, not a statistical
# significance test: the empirically-confirmed identifiability bias (a
# known-correct coefficient can still come out ~3x off even on clean
# synthetic data) only gets worse with more simultaneous candidate
# neighbors, which is exactly what the probe pass fits - candidates are
# always informational suggestions for a human to sanity-check, never
# auto-applied.
_CANDIDATE_COUPLING_MIN_KW_PER_K = 0.02
# EM-style (fit -> smooth residuals -> relabel -> refit) retroactive
# opening_open relabeling (see _em_relabel_opening_open below): a small
# FIXED iteration count, not a convergence-detection loop, matching this
# codebase's general preference for simplicity over adaptive stopping
# elsewhere (e.g. the refit's own fixed 80/20 holdout split).
_OPENING_RELABEL_DEFAULT_ITERATIONS = 2
# Cap on how many candidate opening events (see _extract_contiguous_open_events
# below) get surfaced per room - mirrors _CANDIDATE_COUPLING_MIN_KW_PER_K's own
# "informational only, never auto-applied" role, just capping list length
# rather than filtering by a magnitude threshold (every surfaced event
# already cleared the Kalman gate itself - see smoothed_opening_flags).
_CANDIDATE_OPENING_EVENT_MAX_PER_ROOM = 5

# EM-style retroactive blind_position relabeling (see _em_relabel_blind_position
# below) - same "small fixed count, not convergence-detection" philosophy as
# _OPENING_RELABEL_DEFAULT_ITERATIONS, but one pass higher: iteration 0 here
# is a qualitatively weaker heuristic bootstrap (no real blind_x_dni
# coefficient identified yet, unlike opening's own iteration 0, which
# already uses real Kalman-gated residuals) - one extra calibrated pass
# meaningfully improves on that weaker start.
_BLIND_RELABEL_DEFAULT_ITERATIONS = 3
# Minimum sunny (dni above blind_kalman_detector.BLIND_DNI_INFORMATIVE_FLOOR_WM2)
# data points a room needs before blind-position relabeling is even
# attempted - blind position is only ever observable when there's sun to
# block or not, so a room with too little (e.g. heatpump_weather_dni_sensor
# left unconfigured, dni a static-zero column) would otherwise get a
# degenerate all-NaN-normalized-to-nothing synthetic column.
_BLIND_RELABEL_MIN_INFORMATIVE_ROWS = 50
# The bootstrap heuristic (bootstrap_raw_blind_signal_from_residual) has no
# principled physical noise model the way the algebraic-inversion branch's
# own resolve_blind_measurement_noise does (that one derives r from real
# measurement uncertainty) - a fixed, moderately-trusting constant is
# enough, since this pass's only job is injecting SOME nonzero-variance
# signal into blind_x_dni, not being precise.
_BLIND_RELABEL_BOOTSTRAP_R = 0.05


def _resolve_room_temp_entity_map(optim_conf: dict, retrieve_hass_conf: dict) -> dict[str, str]:
    """room name -> its heatpump_room_temp_sensors entity_id, for every room
    with both a non-empty name and a configured sensor (unnamed/unsensored
    rooms are simply absent from the result, not an error)."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_sensors = retrieve_hass_conf.get("heatpump_room_temp_sensors", []) or []
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(room_sensors[i]).strip() if i < len(room_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _resolve_room_blind_entity_map(optim_conf: dict, retrieve_hass_conf: dict) -> dict[str, str]:
    """room name -> its heatpump_room_blind_sensors entity_id, for every room
    with both a non-empty name and a configured blind sensor (unnamed/
    unsensored rooms are simply absent from the result, not an error).
    Direct sibling of _resolve_room_temp_entity_map - same single-entity-
    per-room assumption, comma-separated multi-sensor support is not
    implemented here either (matching that existing precedent)."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_blind_sensors = retrieve_hass_conf.get("heatpump_room_blind_sensors", []) or []
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(room_blind_sensors[i]).strip() if i < len(room_blind_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _resolve_room_window_entity_map(optim_conf: dict, retrieve_hass_conf: dict) -> dict[str, str]:
    """room name -> its heatpump_room_window_sensors entity_id, for every room
    with both a non-empty name and a configured window sensor. Direct sibling
    of _resolve_room_blind_entity_map - same single-entity-per-room
    assumption."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_window_sensors = retrieve_hass_conf.get("heatpump_room_window_sensors", []) or []
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(room_window_sensors[i]).strip() if i < len(room_window_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _resolve_room_door_entity_map(optim_conf: dict, retrieve_hass_conf: dict) -> dict[str, str]:
    """room name -> its heatpump_room_door_sensors entity_id, for every room
    with both a non-empty name and a configured door sensor. Direct sibling
    of _resolve_room_blind_entity_map - same single-entity-per-room
    assumption."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    room_door_sensors = retrieve_hass_conf.get("heatpump_room_door_sensors", []) or []
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(room_door_sensors[i]).strip() if i < len(room_door_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _resolve_room_opening_confirm_ready_entity_map(
    optim_conf: dict, retrieve_hass_conf: dict
) -> dict[str, str]:
    """room name -> its heatpump_room_opening_confirm_ready_sensor entity_id
    (an input_boolean the user flips once they've answered the paired
    confirm-answer sensor below). Direct sibling of
    _resolve_room_window_entity_map - same single-entity-per-room
    assumption. Mirrors the mechanism (not the semantics) of the existing
    manual_load_ready_sensor/manual_load_confirm_power_sensor pair."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    ready_sensors = retrieve_hass_conf.get("heatpump_room_opening_confirm_ready_sensor", []) or []
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(ready_sensors[i]).strip() if i < len(ready_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _resolve_room_opening_confirm_answer_entity_map(
    optim_conf: dict, retrieve_hass_conf: dict
) -> dict[str, str]:
    """room name -> its heatpump_room_opening_confirm_answer_sensor entity_id
    (an input_boolean holding the user's yes/was-open (1) vs. no/was-closed
    (0) answer, read once the paired ready sensor above is set). Direct
    sibling of _resolve_room_window_entity_map."""
    room_names = optim_conf.get("heatpump_room_names", []) or []
    answer_sensors = (
        retrieve_hass_conf.get("heatpump_room_opening_confirm_answer_sensor", []) or []
    )
    entity_map: dict[str, str] = {}
    for i, name in enumerate(room_names):
        name = str(name).strip()
        entity_id = str(answer_sensors[i]).strip() if i < len(answer_sensors) else ""
        if name and entity_id:
            entity_map[name] = entity_id
    return entity_map


def _slugify_room_name(name: str) -> str:
    """A room name -> a safe HA entity_id fragment (lowercase, non
    alphanumerics collapsed to single underscores, no leading/trailing
    underscore) - used only for the auto-generated
    sensor.room_opening_confirmation_<slug> entity id (Phase 4's opening-
    confirmation question sensor), unlike every other per-room sensor in
    this codebase, which always uses a user-configured entity_id instead."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "room"


def _parse_room_neighbor_map(optim_conf: dict) -> dict[str, list[str]]:
    """Translate heatpump_room_coupled_neighbors (per-room, comma-separated
    0-based indices into heatpump_room_names - see param_definitions.json)
    into a room-NAME-keyed neighbor map, since
    SelfLearningPhysicsModel/self_learning_physics operate on room names,
    not optimization.py's absolute def_load_config indices."""
    room_names = [str(n).strip() for n in (optim_conf.get("heatpump_room_names", []) or [])]
    raw_neighbors = optim_conf.get("heatpump_room_coupled_neighbors", []) or []
    neighbor_map: dict[str, list[str]] = {}
    for i, name in enumerate(room_names):
        if not name:
            continue
        raw = raw_neighbors[i] if i < len(raw_neighbors) else ""
        neighbors: list[str] = []
        for part in str(raw or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                j = int(part)
            except ValueError:
                continue
            if 0 <= j < len(room_names) and j != i and room_names[j]:
                neighbors.append(room_names[j])
        neighbor_map[name] = neighbors
    return neighbor_map


def _apply_confirmed_opening_overrides(
    opening: pd.Series, overrides: dict[str, float] | None
) -> pd.Series:
    """Stamp user-confirmed opening_open answers onto `opening` by exact
    ISO-timestamp match - confirmed ground truth (the Phase 4 HA
    confirmation loop's own persisted answers; always {} until that lands)
    always wins over any EM-inferred flag. Applied both before iteration 0
    and after every relabeling pass in _em_relabel_opening_open, so a later
    pass can never overwrite a confirmed answer."""
    if not overrides:
        return opening
    result = opening.copy()
    for ts_iso, value in overrides.items():
        try:
            ts = pd.Timestamp(ts_iso)
        except (ValueError, TypeError):
            continue
        if ts in result.index:
            result.loc[ts] = float(value)
    return result


def _em_relabel_opening_open(
    df_raw: pd.DataFrame,
    dfs_by_room: dict[str, pd.DataFrame],
    neighbor_map: dict[str, list[str]],
    window_entity_map: dict[str, str],
    door_entity_map: dict[str, str],
    forgetting_factor: float,
    ridge: float,
    electric_only: bool,
    n_iterations: int,
    confirmed_overrides: dict[str, dict[str, float]],
    logger: logging.Logger,
    infer_additional_opening: dict[str, bool] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """EM-style (fit -> smooth residuals -> relabel -> refit, repeated a
    small FIXED number of times) retroactive opening_open relabeling for
    rooms with NO configured window sensor AND no configured door sensor at
    all, PLUS (opt-in per room, see infer_additional_opening) rooms that DO
    have a real sensor but also have other, un-sensored openings. Only ever
    synthesizes opening_open, never door_open - a single room's own residual
    can't distinguish "my window is open" from "my door is open to a colder
    neighbor room" without jointly modeling the neighbor, the same scope
    boundary already established for the live per-cycle Kalman detector
    (_build_room_kalman_opening_open).

    A room with a real window OR door sensor configured is NEVER touched
    here UNLESS infer_additional_opening[name] is True, for any timestamp,
    at any iteration - eligibility is checked against the CONFIGURED sensor
    maps, not merely whether that sensor's data happened to be present in
    this particular fetch, so an intermittent data gap can never cause a
    non-opted-in sensored room to be synthetically relabeled. For an
    opted-in partially-sensored room, the room's own real opening_open
    reading (already the union of its real window/door sensors, see the
    np.maximum combination upstream in refit_self_learning_physics_model)
    is captured once before iteration 0 as a floor and combined via
    np.maximum with every iteration's inferred signal - a real "open"
    reading is never weakened or overridden, inference only ever ADDS newly
    -discovered open periods on top of it.

    Unlike the live filter (a true online recursion, one dispatch cycle at
    a time), this uses SelfLearningPhysicsModel.predict_one_step_history -
    teacher-forced, vectorized over a room's entire historical window - to
    build each pass's residual trajectory, then the same forward-filter +
    RTS-smoother pipeline as the live detector (opening_kalman_detector.py)
    to turn those residuals into "probably open" flags with the benefit of
    hindsight (past AND future relative to any point).

    User-confirmed answers (confirmed_overrides; {} until the Phase 4 HA
    confirmation loop lands) are merged in before iteration 0 and
    re-applied after every relabeling pass (AFTER the real-sensor floor is
    combined in), so the EM loop's own inference - or a partially-sensored
    room's own real reading - can never overwrite a confirmed answer.

    :param confirmed_overrides: room name -> {timestamp_iso: 0.0/1.0}.
    :param infer_additional_opening: room name -> opted in to inference
        despite having a real sensor configured (heatpump_room_
        opening_infer_additional). Defaults to {} (no room opted in) -
        every room with a real sensor keeps today's exact behavior.
    :return: (blended dfs_by_room - same dict shape/keys as the input, only
        eligible rooms' opening_open column actually changed; diagnostics -
        the LAST iteration's per-room {"is_open", "innovation", "s"} numpy
        arrays, keyed by room name, present only for rooms actually
        relabeled - "is_open" is the newly-discovered component only (with
        any already-real-sensor-known open periods excluded) for a
        partially-sensored room, so candidate-event surfacing never
        re-surfaces something already known - feeds Phase 3's
        candidate-event surfacing).
    """
    from emhass.thermal.opening_kalman_detector import (
        KALMAN_GATE_SIGMA,
        SELF_LEARNING_KALMAN_FALLBACK_R_C2,
        SELF_LEARNING_KALMAN_Q_FRACTION_OF_R,
        SELF_LEARNING_KALMAN_R_FLOOR_C2,
        cold_start_state,
        kalman_forward_filter_array,
        kalman_rts_smooth,
        smoothed_opening_flags,
    )
    from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

    infer_additional_opening = infer_additional_opening or {}
    eligible_rooms = [
        name
        for name in dfs_by_room
        if (name not in window_entity_map and name not in door_entity_map)
        or infer_additional_opening.get(name, False)
    ]
    # Partially-sensored rooms: eligible AND already had real coverage - the
    # subset that needs the real-reading floor/never-override treatment.
    # A fully-unsensored eligible room has no real reading to protect.
    partial_coverage_rooms = {
        name for name in eligible_rooms if name in window_entity_map or name in door_entity_map
    }
    blended = {name: df.copy() for name, df in dfs_by_room.items()}
    # Captured BEFORE the seed/override step below and held constant across
    # every iteration - this is the real sensor(s)' own reading (already
    # np.maximum'd together upstream if this room has both a window and a
    # door sensor), never itself updated by inference.
    real_reference: dict[str, pd.Series] = {
        name: blended[name]["opening_open"].copy()
        for name in partial_coverage_rooms
        if "opening_open" in blended[name].columns
    }
    for name in eligible_rooms:
        if "opening_open" not in blended[name].columns:
            blended[name]["opening_open"] = 0.0
        blended[name]["opening_open"] = _apply_confirmed_opening_overrides(
            blended[name]["opening_open"], confirmed_overrides.get(name)
        )

    if not eligible_rooms or n_iterations <= 0:
        return blended, {}

    def _fit(dfs: dict[str, pd.DataFrame]) -> SelfLearningPhysicsModel:
        model = SelfLearningPhysicsModel(
            forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
        )
        model.fit(
            df_raw,
            dfs,
            df_raw["electric_power"].to_numpy(),
            None if electric_only else df_raw["gas_consumption"].to_numpy(),
            neighbor_map,
        )
        return model

    model = _fit(blended)
    diagnostics: dict[str, dict] = {}
    for iteration in range(n_iterations):
        for name in eligible_rooms:
            df_room = blended[name]
            pred = model.predict_one_step_history(name, df_room, blended)
            actual = df_room["room_temp"].to_numpy(dtype=float)
            residual = actual - pred
            finite = residual[np.isfinite(residual)]
            if len(finite) >= 2:
                # Scaled-MAD (median absolute deviation), not plain np.std:
                # this is bootstrapping from UNLABELED data, so the very
                # undetected anomalies this loop exists to find would
                # otherwise inflate a plain std and self-weaken the gate
                # that's supposed to catch them. 1.4826 is the standard
                # MAD->sigma scale factor for normally-distributed noise.
                mad = float(np.median(np.abs(finite - np.median(finite))))
                r = max(SELF_LEARNING_KALMAN_R_FLOOR_C2, (1.4826 * mad) ** 2)
            else:
                r = SELF_LEARNING_KALMAN_FALLBACK_R_C2
            q = SELF_LEARNING_KALMAN_Q_FRACTION_OF_R * r

            x0, p0 = cold_start_state(float(actual[0]), r)
            trajectory = kalman_forward_filter_array(x0, p0, pred, actual, q, r)
            _, p_smooth = kalman_rts_smooth(trajectory)
            is_open = smoothed_opening_flags(trajectory, p_smooth, r, gate_sigma=KALMAN_GATE_SIGMA)

            combined = is_open
            if name in real_reference:
                # Real sensor is a floor, never weakened - a real "open"
                # reading always survives regardless of what inference says.
                combined = np.maximum(
                    is_open.astype(float), (real_reference[name].to_numpy() >= 0.5).astype(float)
                ).astype(bool)
            new_opening = _apply_confirmed_opening_overrides(
                pd.Series(combined.astype(float), index=df_room.index),
                confirmed_overrides.get(name),
            )
            blended[name] = df_room.assign(opening_open=new_opening)

            if iteration == n_iterations - 1:
                # For a partially-sensored room, only surface the NEWLY
                # discovered component (real sensor said closed, inference
                # says open) - the real sensor's own already-known open
                # periods aren't useful "candidates" to confirm.
                reported_is_open = is_open
                if name in real_reference:
                    reported_is_open = is_open & ~(real_reference[name].to_numpy() >= 0.5)
                diagnostics[name] = {
                    "is_open": reported_is_open,
                    "innovation": trajectory.innovation,
                    "s": trajectory.p_pred + r,
                }

        model = _fit(blended)

    logger.info(
        "self-learning-physics-refit: opening-open relabeling complete for %d "
        "room(s) over %d iteration(s) (%d partially-sensored, inferring "
        "additional openings; %d fully unsensored): %s",
        len(eligible_rooms),
        n_iterations,
        len(partial_coverage_rooms),
        len(eligible_rooms) - len(partial_coverage_rooms),
        ", ".join(eligible_rooms),
    )
    return blended, diagnostics


def _em_relabel_blind_position(
    df_raw: pd.DataFrame,
    dfs_by_room: dict[str, pd.DataFrame],
    neighbor_map: dict[str, list[str]],
    blind_entity_map: dict[str, str],
    forgetting_factor: float,
    ridge: float,
    electric_only: bool,
    n_iterations: int,
    logger: logging.Logger,
    infer_additional_blind: dict[str, bool] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """EM-style (fit -> smooth residuals -> relabel -> refit, repeated a
    small FIXED number of times) retroactive blind_position relabeling for
    self-learning-physics rooms with NO configured blind sensor at all,
    PLUS (opt-in per room, see infer_additional_blind) rooms that DO have a
    real blind sensor but also have other, un-sensored shading - the
    fit-time sibling of the live per-cycle blind Kalman detector
    (_build_room_kalman_blind_position), covering the same rooms and
    reusing the same emhass.thermal.blind_kalman_detector primitives.

    A room with a real blind sensor configured is NEVER touched here UNLESS
    infer_additional_blind[name] is True, for any timestamp, at any
    iteration - eligibility is checked against the CONFIGURED sensor map,
    matching _em_relabel_opening_open's own never-override guarantee
    exactly. For an opted-in partially-sensored room, the room's own real
    blind_position reading (from its real sensor) is captured once before
    iteration 0 as a floor and combined via np.maximum with every
    iteration's inferred signal: a second, un-sensored closed blind can
    only add MORE shading than the real sensor alone reports, never less -
    the real reading is never weakened.

    Unlike opening detection, blind position is only ever OBSERVABLE when
    there's sun (dni above blind_kalman_detector.BLIND_DNI_INFORMATIVE_FLOOR_WM2)
    - a room without enough informative history (e.g. heatpump_weather_dni_sensor
    left unconfigured, so dni is a static-zero column) is skipped entirely
    rather than writing a degenerate synthetic column.

    Bootstrap/calibrate split (see blind_kalman_detector.py's own module
    docstring for the full algebraic derivation): each iteration checks the
    room's CURRENT fitted blind_x_dni coefficient (beta). While beta is
    still unidentified (a room with no real blind history has a constant-
    zero blind_x_dni feature column, so plain RLS can never move it away
    from its ridge-initialized 0), a heuristic bootstrap signal is used
    instead of the exact algebraic inversion - purely to give the next fit
    a nonzero-variance column so beta becomes identifiable. Every
    iteration always inverts against a FRESHLY recomputed "blind fully
    open" baseline prediction using that iteration's own current beta -
    never against "whatever the previous iteration's synthetic column
    said" (see predict_room_temperature_blind_open_baseline).

    :return: (blended dfs_by_room - same dict shape/keys as the input, only
        eligible+informative rooms' blind_position column actually changed;
        diagnostics - the LAST iteration's per-room {"position", "beta",
        "n_informative"}, keyed by room name, present only for rooms
        actually relabeled).
    """
    from emhass.thermal.blind_kalman_detector import (
        BLIND_DNI_INFORMATIVE_FLOOR_WM2,
        BLIND_KALMAN_BETA_EPSILON,
        BLIND_KALMAN_Q,
        blind_cold_start_state,
        bootstrap_raw_blind_signal_from_residual,
        invert_blind_position_from_residual,
        kalman_forward_filter_with_persistence,
        resolve_blind_measurement_noise,
        smoothed_blind_position,
    )
    from emhass.thermal.opening_kalman_detector import kalman_rts_smooth
    from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

    infer_additional_blind = infer_additional_blind or {}
    candidate_rooms = [
        name
        for name in dfs_by_room
        if name not in blind_entity_map or infer_additional_blind.get(name, False)
    ]
    blended = {name: df.copy() for name, df in dfs_by_room.items()}

    eligible_rooms: list[str] = []
    for name in candidate_rooms:
        dni_col = blended[name].get("dni")
        n_informative = (
            int((dni_col > BLIND_DNI_INFORMATIVE_FLOOR_WM2).sum()) if dni_col is not None else 0
        )
        if n_informative < _BLIND_RELABEL_MIN_INFORMATIVE_ROWS:
            logger.warning(
                "self-learning-physics-refit: room %s has only %d informative "
                "(sunny) data point(s) for blind-position relabeling (need at "
                "least %d) - skipped. Configure heatpump_weather_dni_sensor if "
                "this is unexpected.",
                name,
                n_informative,
                _BLIND_RELABEL_MIN_INFORMATIVE_ROWS,
            )
            continue
        eligible_rooms.append(name)

    # Partially-sensored rooms: eligible AND already had real coverage - the
    # subset that needs the real-reading floor/never-override treatment.
    # Captured BEFORE any relabeling and held constant across every
    # iteration; NaN (no real reading at that timestamp) is filled to 0.0
    # (open/no-shading, the same "missing = inert" convention used
    # everywhere else in this feature) rather than left to propagate
    # through np.maximum.
    partial_coverage_rooms = {name for name in eligible_rooms if name in blind_entity_map}
    real_reference: dict[str, np.ndarray] = {
        name: blended[name]["blind_position"].fillna(0.0).to_numpy()
        for name in partial_coverage_rooms
        if "blind_position" in blended[name].columns
    }

    if not eligible_rooms or n_iterations <= 0:
        return blended, {}

    def _fit(dfs: dict[str, pd.DataFrame]) -> SelfLearningPhysicsModel:
        model = SelfLearningPhysicsModel(
            forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
        )
        model.fit(
            df_raw,
            dfs,
            df_raw["electric_power"].to_numpy(),
            None if electric_only else df_raw["gas_consumption"].to_numpy(),
            neighbor_map,
        )
        return model

    model = _fit(blended)
    diagnostics: dict[str, dict] = {}
    for iteration in range(n_iterations):
        for name in eligible_rooms:
            df_room = blended[name]
            room_model = model.room_models_[name]
            beta = float(room_model.theta_temp[room_model.feature_names.index("blind_x_dni")])
            actual = df_room["room_temp"].to_numpy(dtype=float)
            # Eligibility above already guarantees a real "dni" column with
            # enough informative rows for every room reaching this point.
            dni = df_room["dni"].to_numpy(dtype=float)

            if abs(beta) < BLIND_KALMAN_BETA_EPSILON:
                pred = model.predict_one_step_history(name, df_room, blended)
                residual = actual - pred
                raw = bootstrap_raw_blind_signal_from_residual(residual, dni)
                r_for_filter = _BLIND_RELABEL_BOOTSTRAP_R
            else:
                df_room_open = df_room.assign(blind_position=0.0)
                pred_open = model.predict_one_step_history(name, df_room_open, blended)
                residual = actual - pred_open
                finite = residual[np.isfinite(residual)]
                # Same scaled-MAD estimator _em_relabel_opening_open already
                # uses on its own finite residuals - not plain np.std, for
                # the same reason (undetected anomalies would otherwise
                # inflate std and self-weaken the inversion).
                if len(finite) >= 2:
                    residual_std_c = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
                else:
                    residual_std_c = 0.3
                raw = invert_blind_position_from_residual(residual, dni, beta)
                r_for_filter = resolve_blind_measurement_noise(residual_std_c, beta, dni)

            x0, p0 = blind_cold_start_state()
            trajectory = kalman_forward_filter_with_persistence(
                x0, p0, raw, BLIND_KALMAN_Q, r_for_filter
            )
            x_smooth, _ = kalman_rts_smooth(trajectory)
            position = smoothed_blind_position(x_smooth)
            if name in real_reference:
                position = np.maximum(position, real_reference[name])

            blended[name] = df_room.assign(blind_position=position)

            if iteration == n_iterations - 1:
                diagnostics[name] = {
                    "position": position,
                    "beta": beta,
                    "n_informative": int((dni > BLIND_DNI_INFORMATIVE_FLOOR_WM2).sum()),
                }

        model = _fit(blended)

    logger.info(
        "self-learning-physics-refit: blind-position relabeling complete for %d "
        "room(s) over %d iteration(s) (%d partially-sensored, inferring "
        "additional shading; %d fully unsensored): %s",
        len(eligible_rooms),
        n_iterations,
        len(partial_coverage_rooms),
        len(eligible_rooms) - len(partial_coverage_rooms),
        ", ".join(eligible_rooms),
    )
    return blended, diagnostics


def _extract_contiguous_open_events(diagnostics: dict, index: pd.DatetimeIndex) -> list[dict]:
    """Collapse one room's consecutive is_open=True runs (from
    _em_relabel_opening_open's own last-iteration diagnostics) into
    contiguous candidate events - informational only, never auto-applied,
    the same role _CANDIDATE_COUPLING_MIN_KW_PER_K's candidate-coupling
    suggestions already play for undeclared room pairs.

    :param diagnostics: one room's {"is_open", "innovation", "s"} arrays.
    :param index: that same room's DatetimeIndex (same length/order as the
        diagnostics arrays) - used only to render start/end as ISO strings.
    :return: events sorted by confidence (mean_abs_normalized_innovation)
        descending, NOT yet capped to _CANDIDATE_OPENING_EVENT_MAX_PER_ROOM
        - the caller applies that cap.
    """
    is_open = np.asarray(diagnostics["is_open"], dtype=bool)
    innovation = np.asarray(diagnostics["innovation"], dtype=float)
    s = np.asarray(diagnostics["s"], dtype=float)
    # Innovation normalized by its own step's predictive std - makes
    # "confidence" comparable across events of different lengths/noise
    # levels, the same normalization smoothed_opening_flags's own gate uses.
    normalized = np.abs(innovation) / np.sqrt(np.maximum(s, 1e-9))

    events: list[dict] = []
    n = len(is_open)
    t = 0
    while t < n:
        if not is_open[t]:
            t += 1
            continue
        start = t
        while t < n and is_open[t]:
            t += 1
        end = t  # exclusive
        events.append(
            {
                "start_iso": index[start].isoformat(),
                "end_iso": index[end - 1].isoformat(),
                "n_steps": end - start,
                "mean_abs_normalized_innovation": float(np.mean(normalized[start:end])),
            }
        )
    events.sort(key=lambda e: e["mean_abs_normalized_innovation"], reverse=True)
    return events


async def _resolve_opening_confirmations(
    rh, emhass_conf: dict, optim_conf: dict, retrieve_hass_conf: dict, logger: logging.Logger
) -> dict[str, list[dict]]:
    """Poll every room's opening-confirmation ready/answer input_booleans
    (see _resolve_room_opening_confirm_ready_entity_map/_answer_entity_map)
    once per self-learning-physics-refit call - NOT once per dispatch
    cycle, unlike _apply_manual_load_runtime_overrides's own polling: a
    confirmed answer only ever feeds a FUTURE refit, never live dispatch.

    A room's pending confirmation (published by a PRIOR refit's own
    _publish_opening_confirmation_questions call) whose ready sensor now
    reads 1.0 gets resolved into a permanent confirmed range and persisted
    to self_learning_physics_opening_confirmations.json. A read failure
    (ready sensor missing/unavailable, or ready=1 but the answer sensor
    itself unreadable) leaves that entry pending, never silently dropped -
    it will simply be tried again on the next refit.

    :return: room name -> list of confirmed {"start_iso", "end_iso",
        "value"} ranges, accumulated across EVERY refit so far (not just
        this cycle's newly-resolved ones) - permanent ground truth. Feed
        into _em_relabel_opening_open via
        _expand_confirmed_ranges_to_timestamps first (this function returns
        RANGES, not the individual per-timestamp entries that function
        expects).
    """
    blob = await load_json_blob(
        emhass_conf,
        "self_learning_physics_opening_confirmations.json",
        logger,
        default={"rooms": {}},
    )
    rooms_state = blob.get("rooms", {}) if isinstance(blob, dict) else {}
    if not isinstance(rooms_state, dict):
        rooms_state = {}

    ready_map = _resolve_room_opening_confirm_ready_entity_map(optim_conf, retrieve_hass_conf)
    answer_map = _resolve_room_opening_confirm_answer_entity_map(optim_conf, retrieve_hass_conf)
    changed = False
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()

    for room_name, ready_entity in ready_map.items():
        room_state = rooms_state.get(room_name)
        if not isinstance(room_state, dict) or not room_state.get("pending"):
            continue
        pending = room_state["pending"]
        ready_value = await rh.get_current_state(ready_entity)
        if ready_value != 1.0:
            continue  # not answered yet (or read failed) - stays pending
        answer_entity = answer_map.get(room_name)
        answer_value = await rh.get_current_state(answer_entity) if answer_entity else None
        if answer_value is None:
            continue  # ready, but the answer itself couldn't be read - stays pending
        confirmed = room_state.setdefault("confirmed", [])
        confirmed.append(
            {
                "start_iso": pending.get("start_iso"),
                "end_iso": pending.get("end_iso"),
                "value": 1.0 if answer_value >= 0.5 else 0.0,
                "confirmed_ts_iso": now_iso,
            }
        )
        room_state["pending"] = None
        changed = True
        logger.info(
            "self-learning-physics-refit: opening confirmation resolved for room %s "
            "(%s to %s) -> %s",
            room_name,
            pending.get("start_iso"),
            pending.get("end_iso"),
            "open" if answer_value >= 0.5 else "closed",
        )

    if changed:
        await save_json_blob(
            emhass_conf,
            "self_learning_physics_opening_confirmations.json",
            {"rooms": rooms_state},
            logger,
        )

    return {
        room_name: list(room_state.get("confirmed", []))
        for room_name, room_state in rooms_state.items()
        if isinstance(room_state, dict) and room_state.get("confirmed")
    }


def _expand_confirmed_ranges_to_timestamps(
    confirmed_ranges: dict[str, list[dict]], dfs_by_room: dict[str, pd.DataFrame]
) -> dict[str, dict[str, float]]:
    """Expand each room's confirmed [start_iso, end_iso] ground-truth ranges
    (see _resolve_opening_confirmations) into the individual per-timestamp
    {timestamp_iso: 0.0/1.0} dict _em_relabel_opening_open/
    _apply_confirmed_opening_overrides actually expect - one entry per real
    timestep in that room's own history that falls within the range."""
    expanded: dict[str, dict[str, float]] = {}
    for room_name, ranges in confirmed_ranges.items():
        df_room = dfs_by_room.get(room_name)
        if df_room is None or not ranges:
            continue
        per_ts: dict[str, float] = {}
        for rng in ranges:
            try:
                start = pd.Timestamp(rng["start_iso"])
                end = pd.Timestamp(rng["end_iso"])
                value = float(rng["value"])
            except (KeyError, ValueError, TypeError):
                continue
            mask = (df_room.index >= start) & (df_room.index <= end)
            for ts in df_room.index[mask]:
                per_ts[ts.isoformat()] = value
        if per_ts:
            expanded[room_name] = per_ts
    return expanded


async def _publish_opening_confirmation_questions(
    rh,
    emhass_conf: dict,
    optim_conf: dict,
    retrieve_hass_conf: dict,
    candidate_openings: list[dict],
    logger: logging.Logger,
) -> None:
    """Publish at most one pending opening-confirmation question per room
    (mirrors the manual-load flow's own "one pending commitment per load"
    cardinality) - a direct rh.post_data(...) call (like
    forecast_model_predict's own direct publish), NOT routed through
    PublishContext, since this refit action never has an opt_res_latest.
    State is a human-readable question with the room name and event window
    embedded in the string itself - no HA entity supports arbitrary custom
    attributes today (see RetrieveHass.post_data), so there's nowhere else
    for that context to ride.

    Runs LAST in the refit (after Phase 3's candidate_openings already
    exists) - only ever asks about the single highest-confidence candidate
    per room (candidate_openings already arrives sorted+capped per room
    from _extract_contiguous_open_events), and never re-asks about an
    event that's already pending or already confirmed either way.
    """
    if not candidate_openings:
        return
    ready_map = _resolve_room_opening_confirm_ready_entity_map(optim_conf, retrieve_hass_conf)
    answer_map = _resolve_room_opening_confirm_answer_entity_map(optim_conf, retrieve_hass_conf)
    eligible_rooms = [name for name in ready_map if name in answer_map]
    if not eligible_rooms:
        return

    blob = await load_json_blob(
        emhass_conf,
        "self_learning_physics_opening_confirmations.json",
        logger,
        default={"rooms": {}},
    )
    rooms_state = blob.get("rooms", {}) if isinstance(blob, dict) else {}
    if not isinstance(rooms_state, dict):
        rooms_state = {}

    # Best (first, since candidate_openings already arrives confidence-
    # sorted+capped per room) candidate per room.
    best_candidate: dict[str, dict] = {}
    for candidate in candidate_openings:
        room_name = candidate.get("room")
        if room_name in eligible_rooms and room_name not in best_candidate:
            best_candidate[room_name] = candidate

    changed = False
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()
    for room_name in eligible_rooms:
        candidate = best_candidate.get(room_name)
        if candidate is None:
            continue
        room_state = rooms_state.get(room_name)
        if not isinstance(room_state, dict):
            room_state = {"pending": None, "confirmed": []}
            rooms_state[room_name] = room_state
        if room_state.get("pending"):
            continue  # already waiting on an answer - one at a time
        already_confirmed = {
            (c.get("start_iso"), c.get("end_iso")) for c in room_state.get("confirmed", []) or []
        }
        if (candidate["start_iso"], candidate["end_iso"]) in already_confirmed:
            continue  # this exact event was already answered before

        room_state["pending"] = {
            "start_iso": candidate["start_iso"],
            "end_iso": candidate["end_iso"],
            "question_ts_iso": now_iso,
        }
        changed = True

        entity_id = f"sensor.room_opening_confirmation_{_slugify_room_name(room_name)}"
        question = (
            f"Was room '{room_name}' really open (window/door) between "
            f"{candidate['start_iso']} and {candidate['end_iso']}? Set "
            f"{answer_map[room_name]} to your answer, then {ready_map[room_name]} "
            f"to on, to confirm."
        )
        question_df = pd.Series([question], index=pd.date_range(pd.Timestamp.now(tz="UTC"), periods=1))
        await rh.post_data(
            question_df,
            0,
            entity_id,
            "enum",
            "",
            f"{room_name} Opening Confirmation",
            type_var="categorical",
        )
        logger.info(
            "self-learning-physics-refit: published opening-confirmation question "
            "for room %s (%s to %s) on %s",
            room_name,
            candidate["start_iso"],
            candidate["end_iso"],
            entity_id,
        )

    if changed:
        await save_json_blob(
            emhass_conf,
            "self_learning_physics_opening_confirmations.json",
            {"rooms": rooms_state},
            logger,
        )


def _split_rooms_by_time(
    dfs: dict[str, pd.DataFrame], split1, split2
) -> tuple[dict, dict, dict]:
    """Chronological train/val/test split of a {room_name: df} mapping on
    shared timestamp boundaries - shared by refit_self_learning_physics_model
    (relabel-selection probes, the main val split) and
    tune_self_learning_physics_model (the hyperparameter grid search)."""
    return (
        {n: d[d.index < split1] for n, d in dfs.items()},
        {n: d[(d.index >= split1) & (d.index < split2)] for n, d in dfs.items()},
        {n: d[d.index >= split2] for n, d in dfs.items()},
    )


def _open_loop_windows(n_total: int, horizon_steps: int) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) row ranges of length horizon_steps
    spanning a holdout series of n_total rows - shared by _fit_and_score
    (via _make_self_learning_physics_scorer) and
    _score_physics_baseline_room_maes (nested in
    refit_self_learning_physics_model) so all self-learning-physics scoring
    uses an identical, consistent windowing. Drops a short trailing
    remainder (< half a horizon) rather than scoring it as its own tiny,
    noisy window. Degrades to a single [0, n_total) window (today's old
    behavior) when n_total <= horizon_steps.
    """
    windows = [
        (start, min(start + horizon_steps, n_total)) for start in range(0, n_total, horizon_steps)
    ]
    if len(windows) > 1 and (windows[-1][1] - windows[-1][0]) < horizon_steps / 2:
        windows.pop()
    return windows


def _make_self_learning_physics_scorer(electric_only: bool, neighbor_map, horizon_steps: int):
    """Returns a _fit_and_score(model, df_house_fit, rooms_fit, df_house_eval,
    rooms_eval, collect_series=False) -> dict closure bound to this call's
    electric_only/neighbor_map/horizon_steps - fits model on the fit split,
    then scores it open-loop (via _open_loop_windows) on the eval split.
    Shared by refit_self_learning_physics_model (every val/test scoring
    call there) and tune_self_learning_physics_model (one call per grid
    candidate) - this is the correctness-sensitive core (open-loop
    windowing, per-room MAE, residual-std) that must not be duplicated.
    """
    from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

    def _fit_and_score(
        model: SelfLearningPhysicsModel,
        df_house_fit: pd.DataFrame,
        rooms_fit: dict[str, pd.DataFrame],
        df_house_eval: pd.DataFrame,
        rooms_eval: dict[str, pd.DataFrame],
        collect_series: bool = False,
    ) -> dict:
        model.fit(
            df_house_fit,
            rooms_fit,
            df_house_fit["electric_power"].to_numpy(),
            None if electric_only else df_house_fit["gas_consumption"].to_numpy(),
            neighbor_map,
        )
        elec_residual_chunks = []
        gas_residual_chunks = []
        room_residual_chunks: dict[str, list[np.ndarray]] = {name: [] for name in rooms_eval}
        # Only populated when collect_series=True (the honest-test-report's
        # trainval/test call, see below) - every other call site (the
        # relabel-selection probes, the val-scoring probe, tune's grid
        # search) leaves this unused, at zero extra cost, since only that
        # one call's predicted-vs-actual room temperature ever needs to
        # become a plot.
        room_pred_chunks: dict[str, list[pd.Series]] = (
            {name: [] for name in rooms_eval} if collect_series else {}
        )
        room_actual_chunks: dict[str, list[pd.Series]] = (
            {name: [] for name in rooms_eval} if collect_series else {}
        )
        for start, end in _open_loop_windows(len(df_house_eval), horizon_steps):
            chunk_house = df_house_eval.iloc[start:end]
            chunk_rooms = {name: df.iloc[start:end] for name, df in rooms_eval.items()}
            initial_states = {
                name: float(df["room_temp"].iloc[0]) for name, df in chunk_rooms.items() if len(df)
            }
            pred = model.predict_recursive(chunk_house, chunk_rooms, initial_states)
            elec_residual_chunks.append(
                pred["electric_power"] - chunk_house["electric_power"].to_numpy()
            )
            if not electric_only:
                gas_residual_chunks.append(
                    pred["gas_consumption"] - chunk_house["gas_consumption"].to_numpy()
                )
            for name, df_h in chunk_rooms.items():
                if not len(df_h) or name not in pred["room_temp"]:
                    continue
                room_residual_chunks[name].append(pred["room_temp"][name] - df_h["room_temp"].to_numpy())
                if collect_series:
                    room_pred_chunks[name].append(
                        pd.Series(pred["room_temp"][name], index=df_h.index)
                    )
                    room_actual_chunks[name].append(df_h["room_temp"])

        scores = {
            "electric_mae_w": float(np.mean(np.abs(np.concatenate(elec_residual_chunks)))),
        }
        if not electric_only:
            scores["gas_mae_m3"] = float(np.mean(np.abs(np.concatenate(gas_residual_chunks))))
        room_maes = {}
        room_residual_stds = {}
        for name, chunks in room_residual_chunks.items():
            if not chunks:
                continue
            residuals = np.concatenate(chunks)
            room_maes[name] = float(np.mean(np.abs(residuals)))
            # Holdout residual std - the Kalman opening detector's own
            # measurement-noise variance R for this room (see
            # opening_kalman_detector.py's SELF_LEARNING_KALMAN_* constants
            # and _build_room_kalman_opening_open) - same residual array the
            # MAE above is already computed from, no extra fit/score cost.
            room_residual_stds[name] = float(np.std(residuals))
        scores["room_temp_mae_c"] = room_maes
        scores["room_temp_residual_std_c"] = room_residual_stds
        if collect_series:
            scores["room_temp_pred_series"] = {
                name: pd.concat(chunks).sort_index()
                for name, chunks in room_pred_chunks.items()
                if chunks
            }
            scores["room_temp_actual_test_series"] = {
                name: pd.concat(chunks).sort_index()
                for name, chunks in room_actual_chunks.items()
                if chunks
            }
        return scores

    return _fit_and_score


async def _prepare_self_learning_physics_fit_data(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Shared data-preparation preamble for refit_self_learning_physics_model
    and tune_self_learning_physics_model: config/sensor validation,
    historical data retrieval, per-room dataframe construction, and the
    70/15/15 chronological train/val/test split - identical for both
    callers, since tuning is "the same fit pipeline, different
    hyperparameters," not a different data pipeline. The returned
    dfs_by_room is deliberately the pre-relabel "baseline" (this function
    never runs the opt-in opening/blind relabel blocks - those are
    refit-only, see refit_self_learning_physics_model's own docstring;
    tune always searches on baseline data). Log messages below are
    prefixed "self-learning-physics-refit:" even when called from tune,
    since this is refit's own extracted data pipeline being reused
    verbatim. Gated on the same self_learning_physics_refit_enabled flag
    both refit and tune share - tuning has identical prerequisites to
    refitting, no separate config flag.

    Does NOT resolve self_learning_physics_opening_confirm_enabled's
    confirmation loop (that has real side effects - HA publish/persist -
    and must run at most once per refit, never for tune); callers that
    need it (refit only) resolve it themselves, in the same relative
    position it held before this preamble was extracted (right after the
    enabled/use_influxdb checks, before data retrieval).

    :return: None on disabled/misconfigured/insufficient data (already
        logged), else a dict with electric_only, df_raw, dfs_by_room,
        neighbor_map, forgetting_factor, ridge, dt_hours, n_rows,
        window_days, horizon_steps, split1, split2, df_house_train,
        df_house_val, df_house_test, window_entity_map, door_entity_map,
        blind_entity_map, infer_additional_opening, infer_additional_blind.
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    retrieve_hass_conf = input_data_dict["retrieve_hass_conf"]
    rh = input_data_dict["rh"]

    if not optim_conf.get("self_learning_physics_refit_enabled", False):
        logger.debug(
            "self-learning-physics-refit: disabled (self_learning_physics_refit_enabled=False)"
        )
        return None
    if not retrieve_hass_conf.get("use_influxdb", False):
        logger.error(
            "self-learning-physics-refit: use_influxdb is not enabled. The refit window "
            "(self_learning_physics_refit_window_days) is normally far longer than Home "
            "Assistant's own recorder retention - configure InfluxDB rather than risk "
            "silently fitting on a truncated REST window."
        )
        return None

    electric_only = not str(retrieve_hass_conf.get("heatpump_gas_meter_sensor", "") or "").strip()

    sensor_map: dict[str, str] = {}
    for conf_key in _SELF_LEARNING_PHYSICS_REQUIRED_SENSORS:
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if not entity_id:
            logger.error("self-learning-physics-refit: %s is not configured", conf_key)
            return None
        sensor_map[entity_id] = _SELF_LEARNING_PHYSICS_SENSOR_COLUMN_MAP[conf_key]
    for conf_key, column in _SELF_LEARNING_PHYSICS_SENSOR_COLUMN_MAP.items():
        if conf_key in _SELF_LEARNING_PHYSICS_REQUIRED_SENSORS:
            continue
        entity_id = retrieve_hass_conf.get(conf_key, "")
        if entity_id:
            sensor_map[entity_id] = column
        else:
            logger.warning(
                "self-learning-physics-refit: %s is not configured - '%s' will use its "
                "static default for this refit.",
                conf_key,
                column,
            )
    if not electric_only:
        gas_entity = retrieve_hass_conf.get("heatpump_gas_meter_sensor", "")
        sensor_map[gas_entity] = "gas_consumption"

    room_entity_map = _resolve_room_temp_entity_map(optim_conf, retrieve_hass_conf)
    if not room_entity_map:
        logger.error(
            "self-learning-physics-refit: no rooms with a configured heatpump_room_temp_sensors entry"
        )
        return None

    # Per-room blind/window/door sensors (all opt-in), feeding the
    # self-learning-physics model's own learned blind_x_dni/opening_x_outdoor/
    # door_x_neighbor_diff features (self_learning_physics.py::_physics_features).
    # A room with no configured sensor simply doesn't get the corresponding
    # column - _physics_features already defaults each to 0.0 (inert).
    # Resolved here, BEFORE all_entities is built below, so these entity ids
    # actually reach the rh.get_data(...) fetch - a real bug in the original
    # blind-only wiring (it resolved blind_entity_map only after this fetch
    # already ran, so blind_position training data was silently absent from
    # every real refit) that this fix corrects for all three sensor types.
    blind_entity_map = _resolve_room_blind_entity_map(optim_conf, retrieve_hass_conf)
    window_entity_map = _resolve_room_window_entity_map(optim_conf, retrieve_hass_conf)
    door_entity_map = _resolve_room_door_entity_map(optim_conf, retrieve_hass_conf)

    # Per-room opt-in: infer additional un-sensored openings/blinds on top
    # of (never instead of) a room's own real sensor(s) - see
    # _em_relabel_opening_open/_em_relabel_blind_position's own docstrings.
    # Same lightweight "zip against heatpump_room_names by index" pattern
    # heatpump_room_self_learning_only itself already uses elsewhere, no
    # dedicated resolver function needed for a plain per-room bool.
    _infer_additional_room_names = [
        str(n).strip() for n in (optim_conf.get("heatpump_room_names", []) or [])
    ]
    _opening_infer_additional_list = optim_conf.get("heatpump_room_opening_infer_additional", []) or []
    _blind_infer_additional_list = optim_conf.get("heatpump_room_blind_infer_additional", []) or []
    infer_additional_opening = {
        name: bool(_opening_infer_additional_list[i])
        for i, name in enumerate(_infer_additional_room_names)
        if name and i < len(_opening_infer_additional_list)
    }
    infer_additional_blind = {
        name: bool(_blind_infer_additional_list[i])
        for i, name in enumerate(_infer_additional_room_names)
        if name and i < len(_blind_infer_additional_list)
    }

    window_days = int(optim_conf.get("self_learning_physics_refit_window_days", 60))
    days_list = utils.get_days_list(window_days)
    all_entities = list(
        dict.fromkeys(
            [
                *sensor_map.keys(),
                *room_entity_map.values(),
                *blind_entity_map.values(),
                *window_entity_map.values(),
                *door_entity_map.values(),
            ]
        )
    )
    if not await rh.get_data(days_list, all_entities):
        logger.error(
            "self-learning-physics-refit: failed to retrieve history from Home Assistant/InfluxDB"
        )
        return None

    df_raw = rh.df_final.rename(columns=sensor_map)
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
    df_raw = await _fill_missing_weather_from_open_meteo(
        df_raw,
        retrieve_hass_conf,
        days_list,
        {"outdoor_temp", "wind_speed", "dni", "dhi"},
        input_data_dict["fcst"],
        logger,
    )
    required_cols = ["electric_power", "heatpump_duty"]
    if not electric_only:
        required_cols.append("gas_consumption")
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        logger.error(
            "self-learning-physics-refit: no data retrieved for required column(s): %s",
            ", ".join(missing),
        )
        return None
    df_raw = df_raw.dropna(subset=required_cols)
    n_rows = len(df_raw)
    if n_rows < _SELF_LEARNING_PHYSICS_MIN_ROWS:
        logger.error(
            "self-learning-physics-refit: only %d complete data points retrieved over %d "
            "days (need at least %d) - aborting rather than fitting on too little data.",
            n_rows,
            window_days,
            _SELF_LEARNING_PHYSICS_MIN_ROWS,
        )
        return None

    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    dt_hours = _infer_timestep_hours(df_raw.index)
    df_raw["electric_power"] = utils.resolve_incremental_series(
        df_raw["electric_power"], "electric_power", logger, rate_dt_hours=dt_hours
    )
    if not electric_only:
        df_raw["gas_consumption"] = utils.resolve_incremental_series(
            df_raw["gas_consumption"], "gas_consumption", logger
        )

    # No P_deferrable dispatch history exists to compute a real per-room/
    # aggregate duty for training (see utils.compute_aggregate_heatpump_duty's
    # own docstring) - fall back to the single whole-house heatpump_duty
    # column for both each room's own duty and the shared group_duty
    # confound-control feature. An explicit, accepted v1 limitation.
    df_raw = df_raw.assign(group_duty=df_raw["heatpump_duty"])

    # Sun position (altitude/azimuth) for every historical timestamp, feeding
    # self_learning_physics.py's sun_alt_sin/dni_x_sun_az_sin/cos features -
    # deterministic from timestamp+location (via pvlib), never a real
    # sensor, so it's computed directly rather than fetched. Same
    # Forecast.compute_solar_angles + sin/cos conversion
    # prepare_forecast_and_weather_data already uses for the live dispatch
    # path, applied here to df_raw's own historical index instead.
    solar_angles = Forecast.compute_solar_angles(
        df_raw,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    alt_rad = np.radians(solar_angles["solar_elevation"].to_numpy(dtype=float))
    az_rad = np.radians(solar_angles["solar_azimuth"].to_numpy(dtype=float))
    df_raw["sun_alt_sin"] = np.sin(alt_rad)
    df_raw["sun_az_sin"] = np.sin(az_rad)
    df_raw["sun_az_cos"] = np.cos(az_rad)

    dfs_by_room: dict[str, pd.DataFrame] = {}
    for name, entity_id in room_entity_map.items():
        if entity_id not in rh.df_final.columns:
            logger.warning(
                "self-learning-physics-refit: no data retrieved for room %s (%s) - skipped.",
                name,
                entity_id,
            )
            continue
        df_room = df_raw.assign(room_temp=rh.df_final[entity_id].reindex(df_raw.index))
        blind_entity_id = blind_entity_map.get(name)
        if blind_entity_id and blind_entity_id in rh.df_final.columns:
            df_room = df_room.assign(
                blind_position=rh.df_final[blind_entity_id].reindex(df_raw.index)
            )

        # Per-room live window/door history (opt-in), feeding
        # opening_x_outdoor (window OR door) and door_x_neighbor_diff::*
        # (door only) - see self_learning_physics.py::_physics_features.
        # >= 0.5 interpretation matches _build_room_binary_open_state;
        # NaN comparisons evaluate to False, so missing historical readings
        # correctly default to "closed".
        window_entity_id = window_entity_map.get(name)
        door_entity_id = door_entity_map.get(name)
        opening_series = None
        if window_entity_id and window_entity_id in rh.df_final.columns:
            opening_series = (
                rh.df_final[window_entity_id].reindex(df_raw.index) >= 0.5
            ).astype(float)
        if door_entity_id and door_entity_id in rh.df_final.columns:
            door_series = (rh.df_final[door_entity_id].reindex(df_raw.index) >= 0.5).astype(
                float
            )
            opening_series = (
                door_series if opening_series is None else np.maximum(opening_series, door_series)
            )
            df_room = df_room.assign(door_open=door_series)
        if opening_series is not None:
            df_room = df_room.assign(opening_open=opening_series)

        df_room = df_room.dropna(subset=["room_temp"])
        if len(df_room) < _SELF_LEARNING_PHYSICS_MIN_ROWS:
            logger.warning(
                "self-learning-physics-refit: room %s has only %d complete temperature "
                "data points (need at least %d) - skipped.",
                name,
                len(df_room),
                _SELF_LEARNING_PHYSICS_MIN_ROWS,
            )
            continue
        dfs_by_room[name] = df_room

    if not dfs_by_room:
        logger.error("self-learning-physics-refit: no room has enough temperature history to fit")
        return None

    if optim_conf.get("self_learning_physics_coupling_enabled", True):
        neighbor_map = {
            name: [n for n in neighbors if n in dfs_by_room]
            for name, neighbors in _parse_room_neighbor_map(optim_conf).items()
            if name in dfs_by_room
        }
    else:
        # Degrades gracefully to independent single-zone fits per room - no
        # neighbor_diff features at all, matching the field's own description.
        neighbor_map = dict.fromkeys(dfs_by_room, [])

    forgetting_factor = float(optim_conf.get("self_learning_physics_forgetting_factor", 0.995))
    ridge = float(optim_conf.get("self_learning_physics_ridge", 10.0))

    # Chronological train/val/test split on shared timestamp boundaries (not
    # shared row counts - rooms may have slightly different row counts after
    # their own dropna above), so every room's and the whole house's splits
    # refer to the same real time windows. val is used for every model-
    # SELECTION decision below (the deploy-quality gate, the per-room
    # self-learning-vs-physics dispatch comparison) - test is touched
    # exactly once, after that selection is already final, purely to report
    # an honest accuracy number that was never used to decide anything (see
    # the "honest held-out test MAE" block further down). Comparing several
    # candidate configurations against the SAME split and then reporting
    # that split's own score as "how good is this" is the classic leakage
    # this 3-way split exists to avoid.
    i_train_end = max(1, int(round(n_rows * 0.70)))
    i_val_end = max(i_train_end + 1, int(round(n_rows * 0.85)))
    split1, split2 = df_raw.index[i_train_end], df_raw.index[min(i_val_end, n_rows - 1)]
    df_house_train = df_raw[df_raw.index < split1]
    df_house_val = df_raw[(df_raw.index >= split1) & (df_raw.index < split2)]
    df_house_test = df_raw[df_raw.index >= split2]
    if len(df_house_val) < 10:
        logger.error(
            "self-learning-physics-refit: too few validation rows (%d) after a 70/15/15 "
            "chronological split of %d rows - aborting.",
            len(df_house_val),
            n_rows,
        )
        return None

    # Live dispatch (naive-mpc-optim/dayahead-optim) never runs this model
    # open-loop for the length of a whole holdout split - it re-solves with
    # fresh ground truth on a cadence set by delta_forecast_daily, so a
    # single long continuous recursive holdout run lets prediction drift
    # compound over weeks in a way live dispatch never actually experiences,
    # and makes the resulting MAE highly sensitive to whichever few days in
    # that one window happen to be hardest (confirmed empirically: two
    # adjacent 21-day holdout windows scored on the very same frozen model
    # differed by 30%+ - far more than any feature/hyperparameter choice
    # tested moved it). Re-anchoring to the real actual room temperature on
    # that same delta_forecast_daily cadence and averaging residuals across
    # every such short window is both more robust (many independent samples
    # instead of one) and more representative of what "accurate enough to
    # deploy" actually means for this model in practice.
    delta_forecast = optim_conf.get("delta_forecast_daily", pd.Timedelta(days=1))
    if isinstance(delta_forecast, (int, float)):
        delta_forecast = pd.Timedelta(days=delta_forecast)
    horizon_steps = max(1, int(round(delta_forecast / pd.Timedelta(hours=dt_hours))))

    return {
        "electric_only": electric_only,
        "df_raw": df_raw,
        "dfs_by_room": dfs_by_room,
        "neighbor_map": neighbor_map,
        "forgetting_factor": forgetting_factor,
        "ridge": ridge,
        "dt_hours": dt_hours,
        "n_rows": n_rows,
        "window_days": window_days,
        "horizon_steps": horizon_steps,
        "split1": split1,
        "split2": split2,
        "df_house_train": df_house_train,
        "df_house_val": df_house_val,
        "df_house_test": df_house_test,
        "window_entity_map": window_entity_map,
        "door_entity_map": door_entity_map,
        "blind_entity_map": blind_entity_map,
        "infer_additional_opening": infer_additional_opening,
        "infer_additional_blind": infer_additional_blind,
    }


async def refit_self_learning_physics_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit the multi-room self-learning physics model (RLS, online-adaptive
    linear regression predicting electric power, gas consumption, and every
    configured room's own temperature) against fresh Home Assistant history,
    and deploy it for self-learning-physics-forecast to use.

    Standalone sibling of refit_hybrid_heatpump_model - see
    emhass.thermal.self_learning_physics.SelfLearningPhysicsModel for the
    model itself. Like that sibling, this never influences dispatch by
    itself - see heatpump_room_coupling_conductance/
    self_learning_physics_coupling_source for the opt-in, guardrailed path
    that lets a *fitted* coupling coefficient feed the live optimizer.

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

    if not optim_conf.get("self_learning_physics_refit_enabled", False):
        logger.debug(
            "self-learning-physics-refit: disabled (self_learning_physics_refit_enabled=False)"
        )
        return None
    if not retrieve_hass_conf.get("use_influxdb", False):
        logger.error(
            "self-learning-physics-refit: use_influxdb is not enabled. The refit window "
            "(self_learning_physics_refit_window_days) is normally far longer than Home "
            "Assistant's own recorder retention - configure InfluxDB rather than risk "
            "silently fitting on a truncated REST window."
        )
        return None

    # Opt-in (default off) opening-confirmation loop, Phase 4 of the
    # retroactive-relabeling feature: resolve/persist any now-answered
    # confirmations FIRST, before anything else in this refit - a confirmed
    # answer is permanent ground truth for _em_relabel_opening_open (see
    # _expand_confirmed_ranges_to_timestamps below, once dfs_by_room
    # exists). Publishing new questions happens LAST instead, once Phase
    # 3's candidate_openings exists - see _publish_opening_confirmation_questions.
    # Has real side effects (HA publish/persist) - kept out of the shared
    # _prepare_self_learning_physics_fit_data preamble below (also used by
    # tune_self_learning_physics_model, which must never trigger these),
    # called here in the same relative position it held before that
    # preamble was extracted (right after the enabled/use_influxdb checks,
    # before data retrieval).
    confirmed_ranges: dict[str, list[dict]] = {}
    if optim_conf.get("self_learning_physics_opening_confirm_enabled", False):
        confirmed_ranges = await _resolve_opening_confirmations(
            rh, emhass_conf, optim_conf, retrieve_hass_conf, logger
        )

    prep = await _prepare_self_learning_physics_fit_data(input_data_dict, logger)
    if prep is None:
        return None
    electric_only = prep["electric_only"]
    df_raw = prep["df_raw"]
    dfs_by_room = prep["dfs_by_room"]
    neighbor_map = prep["neighbor_map"]
    forgetting_factor = prep["forgetting_factor"]
    ridge = prep["ridge"]
    dt_hours = prep["dt_hours"]
    n_rows = prep["n_rows"]
    window_days = prep["window_days"]
    horizon_steps = prep["horizon_steps"]
    split1 = prep["split1"]
    split2 = prep["split2"]
    df_house_train = prep["df_house_train"]
    df_house_val = prep["df_house_val"]
    df_house_test = prep["df_house_test"]
    window_entity_map = prep["window_entity_map"]
    door_entity_map = prep["door_entity_map"]
    blind_entity_map = prep["blind_entity_map"]
    infer_additional_opening = prep["infer_additional_opening"]
    infer_additional_blind = prep["infer_additional_blind"]

    from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

    # Snapshot of dfs_by_room BEFORE either relabel block below can touch it -
    # the "baseline" variant in the auto-selection comparison further down
    # (see relabel_active), which automatically picks per room between this
    # untouched baseline and the relabel-enhanced dfs_by_room the two blocks
    # below produce, instead of applying relabeling unconditionally to every
    # eligible room.
    dfs_by_room_baseline = dict(dfs_by_room)
    relabel_active = bool(
        optim_conf.get("self_learning_physics_opening_relabel_enabled", False)
    ) or bool(optim_conf.get("self_learning_physics_blind_relabel_enabled", False))

    # Opt-in (default off), retroactive opening_open relabeling: rooms with
    # NO configured window/door sensor at all get an EM-inferred opening_open
    # column, fed through into dfs_by_room BEFORE the train/holdout split
    # below so every downstream fit (probe, holdout scoring, final_model,
    # candidate-coupling probe) sees the blended data - see
    # _em_relabel_opening_open's own docstring for the never-override
    # guarantee and the confirmed_overrides ground-truth precedence.
    # confirmed_overrides comes from the Phase 4 HA confirmation loop's own
    # persisted, permanent ground truth (confirmed_ranges, resolved above,
    # before dfs_by_room existed) - {} when that loop is disabled or has no
    # confirmations yet.
    opening_relabel_diagnostics: dict[str, dict] = {}
    if optim_conf.get("self_learning_physics_opening_relabel_enabled", False):
        n_relabel_iterations = int(
            optim_conf.get(
                "self_learning_physics_opening_relabel_iterations",
                _OPENING_RELABEL_DEFAULT_ITERATIONS,
            )
        )
        confirmed_overrides = _expand_confirmed_ranges_to_timestamps(confirmed_ranges, dfs_by_room)
        dfs_by_room, opening_relabel_diagnostics = _em_relabel_opening_open(
            df_raw,
            dfs_by_room,
            neighbor_map,
            window_entity_map,
            door_entity_map,
            forgetting_factor,
            ridge,
            electric_only,
            n_relabel_iterations,
            confirmed_overrides=confirmed_overrides,
            logger=logger,
            infer_additional_opening=infer_additional_opening,
        )
        # opening_relabel_diagnostics is consumed further down (only when
        # deployed) to build result["candidate_openings"] - see
        # _extract_contiguous_open_events.

    # Opt-in (default off), retroactive blind_position relabeling: rooms
    # with NO configured blind sensor at all get an EM-inferred continuous
    # blind_position column, fed through into dfs_by_room BEFORE the
    # train/holdout split below, same precedent as opening relabeling
    # above (and independently composable with it - see
    # _em_relabel_blind_position's own docstring).
    blind_relabel_diagnostics: dict[str, dict] = {}
    if optim_conf.get("self_learning_physics_blind_relabel_enabled", False):
        n_blind_relabel_iterations = int(
            optim_conf.get(
                "self_learning_physics_blind_relabel_iterations",
                _BLIND_RELABEL_DEFAULT_ITERATIONS,
            )
        )
        dfs_by_room, blind_relabel_diagnostics = _em_relabel_blind_position(
            df_raw,
            dfs_by_room,
            neighbor_map,
            blind_entity_map,
            forgetting_factor,
            ridge,
            electric_only,
            n_blind_relabel_iterations,
            logger=logger,
            infer_additional_blind=infer_additional_blind,
        )
        # blind_relabel_diagnostics is consumed further down (only when
        # deployed) to build result["blind_position_relabel"].

    # split1/split2/df_house_train/df_house_val/df_house_test/horizon_steps
    # all come from prep above (relabel-independent - see
    # _prepare_self_learning_physics_fit_data's own docstring); only the
    # per-room split below depends on dfs_by_room, which the relabel blocks
    # above may have just replaced.
    rooms_train, rooms_val, rooms_test = _split_rooms_by_time(dfs_by_room, split1, split2)

    _fit_and_score = _make_self_learning_physics_scorer(electric_only, neighbor_map, horizon_steps)

    def _score_physics_baseline_room_maes(rooms_eval: dict[str, pd.DataFrame]) -> dict[str, float]:
        """What the physics/simple thermal_battery model (the fallback a
        room takes when self-learning dispatch isn't attached) would have
        predicted for each room's own temperature, over the SAME window/
        starting point and the SAME open-loop windowing (_open_loop_windows)
        the self-learning model is scored on above - used so a room's
        fitted dispatch coefficients only ever get deployed where they're a
        genuine improvement over the fallback, not just individually "good
        enough" (see room-level filtering below, applied when building
        dispatch_blob).

        Always simulates the "simple" family (zero ongoing heating demand,
        matching what utils.py::_append_room_thermal_loads actually builds
        for a room unless heatpump_model_family="physics" is explicitly
        set) - an accepted simplification, since a physics-family room's
        real envelope/solar demand model needs weather inputs (GHI et al.)
        this refit's own data pipeline doesn't pull.
        """
        room_names_list = [str(n).strip() for n in (optim_conf.get("heatpump_room_names", []) or [])]
        room_volumes_list = optim_conf.get("heatpump_room_volume", []) or []
        room_supply_list = optim_conf.get("heatpump_room_supply_temperature", []) or []
        room_carnot_list = optim_conf.get("heatpump_room_carnot_efficiency", []) or []
        room_power_list = optim_conf.get("heatpump_room_nominal_power", []) or []

        physics_maes: dict[str, float] = {}
        for name, df_h in rooms_eval.items():
            if not len(df_h) or name not in room_names_list:
                continue
            i = room_names_list.index(name)
            nominal_power_w = float(room_power_list[i]) if i < len(room_power_list) else 1500.0
            volume = float(room_volumes_list[i]) if i < len(room_volumes_list) else 15.0
            supply_temperature = float(room_supply_list[i]) if i < len(room_supply_list) else 35.0
            carnot_efficiency = float(room_carnot_list[i]) if i < len(room_carnot_list) else 0.4
            residual_chunks = []
            for start, end in _open_loop_windows(len(df_h), horizon_steps):
                chunk = df_h.iloc[start:end]
                try:
                    trajectory = utils.simulate_physics_room_temperature_trajectory(
                        initial_temp=float(chunk["room_temp"].iloc[0]),
                        duty=chunk["heatpump_duty"].to_numpy(),
                        outdoor_temp=chunk["outdoor_temp"].to_numpy(),
                        nominal_power_w=nominal_power_w,
                        dt_hours=dt_hours,
                        volume=volume,
                        supply_temperature=supply_temperature,
                        carnot_efficiency=carnot_efficiency,
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(
                        "self-learning-physics-refit: could not simulate a physics baseline "
                        "for room %s (%s) - skipping the comparison for this room.", name, e,
                    )
                    residual_chunks = []
                    break
                residual_chunks.append(trajectory - chunk["room_temp"].to_numpy())
            if not residual_chunks:
                continue
            physics_maes[name] = float(np.mean(np.abs(np.concatenate(residual_chunks))))
        return physics_maes

    # Per-room auto-selection: baseline (pre-relabel) vs. enhanced (today's
    # relabel-blended dfs_by_room) is now compared on val, per room, instead
    # of applying relabeling unconditionally to every eligible room. Only
    # runs (and only costs extra probe fits) when relabeling was actually
    # opted into - dfs_by_room/rooms_train/rooms_val/rooms_test are
    # reassigned to the winning per-room mix before anything below reads
    # them, so the rest of this function (deploy gate, honest test report,
    # final_model.fit, dispatch_rooms) needs no further changes - it already
    # just consumes whatever dfs_by_room currently is. Ties (or a room
    # missing from the baseline score) go to enhanced: if the richer variant
    # isn't demonstrably worse, prefer it, since the user explicitly opted
    # the relabeling machinery in.
    use_enhanced_for_room: dict[str, bool] = dict.fromkeys(dfs_by_room, True)
    if relabel_active:
        rooms_train_baseline, rooms_val_baseline, _rooms_test_baseline = _split_rooms_by_time(
            dfs_by_room_baseline, split1, split2
        )
        probe_enhanced = SelfLearningPhysicsModel(
            forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
        )
        val_scores_enhanced = _fit_and_score(
            probe_enhanced, df_house_train, rooms_train, df_house_val, rooms_val
        )
        probe_baseline = SelfLearningPhysicsModel(
            forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
        )
        val_scores_baseline = _fit_and_score(
            probe_baseline, df_house_train, rooms_train_baseline, df_house_val, rooms_val_baseline
        )
        for name in dfs_by_room:
            enhanced_mae = val_scores_enhanced["room_temp_mae_c"].get(name)
            baseline_mae = val_scores_baseline["room_temp_mae_c"].get(name)
            if enhanced_mae is None:
                use_enhanced_for_room[name] = False
            elif baseline_mae is None:
                use_enhanced_for_room[name] = True
            else:
                use_enhanced_for_room[name] = enhanced_mae <= baseline_mae
        dfs_by_room = {
            name: (df if use_enhanced_for_room[name] else dfs_by_room_baseline[name])
            for name, df in dfs_by_room.items()
        }
        rooms_train, rooms_val, rooms_test = _split_rooms_by_time(dfs_by_room, split1, split2)
        logger.info(
            "self-learning-physics-refit: relabel auto-selection on val - using the "
            "relabel-enhanced model for %s, the pre-relabel baseline for %s.",
            sorted(n for n, u in use_enhanced_for_room.items() if u) or "none",
            sorted(n for n, u in use_enhanced_for_room.items() if not u) or "none",
        )

    probe_model = SelfLearningPhysicsModel(
        forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
    )
    val_scores = _fit_and_score(probe_model, df_house_train, rooms_train, df_house_val, rooms_val)
    physics_baseline_val_maes = _score_physics_baseline_room_maes(rooms_val)

    max_electric_mae = float(optim_conf.get("self_learning_physics_refit_max_electric_mae_w", 150.0))
    max_gas_mae = float(optim_conf.get("self_learning_physics_refit_max_gas_mae_m3", 0.02))

    fit_too_bad = val_scores["electric_mae_w"] > max_electric_mae
    if not electric_only:
        fit_too_bad = fit_too_bad or val_scores["gas_mae_m3"] > max_gas_mae
    # Room temperature no longer has an absolute threshold - a room's own
    # fitted dispatch coefficients are only ever deployed (see dispatch_blob
    # below) when they beat this physics baseline specifically, which is a
    # per-room decision, not part of whether the whole refit deploys at all.

    result = {
        "electric_only": electric_only,
        "electric_mae_w": val_scores["electric_mae_w"],
        "max_electric_mae_w": max_electric_mae,
        "gas_mae_m3": None if electric_only else val_scores.get("gas_mae_m3"),
        "max_gas_mae_m3": None if electric_only else max_gas_mae,
        "room_temp_mae_c": val_scores["room_temp_mae_c"],
        "room_temp_residual_std_c": val_scores["room_temp_residual_std_c"],
        "room_temp_physics_baseline_mae_c": physics_baseline_val_maes,
        # Honest, held-out test-set report - populated below, only once the
        # val-based gate has already decided to deploy. NEVER used to gate
        # anything (deploy decision, per-room dispatch inclusion) - test is
        # touched exactly once, purely to report an unbiased accuracy
        # number for whatever configuration val already selected. See the
        # "honest held-out test MAE" block below for why this still isn't a
        # truly prospective/forward-looking accuracy measure (it's the most
        # recent slice of the SAME historical fetch, not genuinely-future
        # data collected after this refit ran).
        "electric_test_mae_w": None,
        "gas_test_mae_m3": None,
        "room_temp_test_mae_c": {},
        "room_temp_physics_baseline_test_mae_c": {},
        # One train/test/pred DataFrame per room with test data (see
        # utils.get_room_temp_test_plot_html) - populated alongside
        # room_temp_test_mae_c below, empty when the test split was too
        # small for an honest report.
        "room_temp_test_plot_df": {},
        "rooms_using_self_learning_dispatch": [],
        # Per-room auto-selection result (see relabel_active above) - which
        # rooms are actually using the relabel-enhanced data this refit, vs.
        # the pre-relabel baseline. Empty when neither relabel flag is on, or
        # when every room's own comparison preferred the baseline.
        "rooms_using_relabel_enhancement": (
            sorted(n for n, u in use_enhanced_for_room.items() if u) if relabel_active else []
        ),
        "n_rows": n_rows,
        "n_rooms": len(dfs_by_room),
        "window_days": window_days,
        "candidate_couplings": [],
        "candidate_openings": [],
        "blind_position_relabel": {},
    }
    if fit_too_bad:
        logger.error(
            "self-learning-physics-refit: whole-house fit quality below threshold "
            "(electric_mae_w=%.2f, gas_mae_m3=%s) - keeping the previously deployed "
            "model, not overwriting. Per-room temperature MAEs (self-learning vs. "
            "physics baseline): %s vs. %s.",
            val_scores["electric_mae_w"],
            "n/a" if electric_only else f"{val_scores.get('gas_mae_m3'):.5f}",
            val_scores["room_temp_mae_c"],
            physics_baseline_val_maes,
        )
        result["deployed"] = False
        return result

    # Honest held-out test MAE: refit on train+val (never on test itself)
    # and score once on the test split - reported purely for visibility,
    # never fed back into the deploy gate or the per-room dispatch decision
    # above (both already used val). trainval_model is discarded right
    # after scoring - it is never pickled/saved, unlike final_model below.
    if len(df_house_test) >= 10:
        df_house_trainval = df_raw[df_raw.index < split2]
        rooms_trainval = {n: d[d.index < split2] for n, d in dfs_by_room.items()}
        trainval_model = SelfLearningPhysicsModel(
            forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
        )
        test_scores = _fit_and_score(
            trainval_model, df_house_trainval, rooms_trainval, df_house_test, rooms_test,
            collect_series=True,
        )
        physics_baseline_test_maes = _score_physics_baseline_room_maes(rooms_test)
        result["electric_test_mae_w"] = test_scores["electric_mae_w"]
        if not electric_only:
            result["gas_test_mae_m3"] = test_scores.get("gas_mae_m3")
        result["room_temp_test_mae_c"] = test_scores["room_temp_mae_c"]
        result["room_temp_physics_baseline_test_mae_c"] = physics_baseline_test_maes
        # Train/test/predicted room-temperature plot data (one DataFrame per
        # room, columns exactly "train"/"test"/"pred"), the same shape
        # MLForecaster.fit() already builds for the load forecaster's own
        # train/test/pred chart (see utils.get_room_temp_test_plot_html) -
        # "train" is the real measured temperature over the train+val period
        # trainval_model was actually fit on, "test"/"pred" are the real vs
        # predicted temperature over the never-touched-for-decisions test
        # period above.
        for room_name, pred_series in test_scores.get("room_temp_pred_series", {}).items():
            actual_train = rooms_trainval[room_name]["room_temp"]
            actual_test = test_scores["room_temp_actual_test_series"][room_name]
            plot_index = actual_train.index.union(actual_test.index).union(pred_series.index)
            df_plot = pd.DataFrame(index=plot_index, columns=["train", "test", "pred"], dtype=float)
            df_plot.loc[actual_train.index, "train"] = actual_train.to_numpy()
            df_plot.loc[actual_test.index, "test"] = actual_test.to_numpy()
            df_plot.loc[pred_series.index, "pred"] = pred_series.to_numpy()
            result["room_temp_test_plot_df"][room_name] = df_plot
        logger.info(
            "self-learning-physics-refit: honest held-out test MAE (retrained on "
            "train+val, NEVER used for any deploy decision) - electric=%.2fW "
            "room_temp=%s vs physics %s",
            test_scores["electric_mae_w"],
            test_scores["room_temp_mae_c"],
            physics_baseline_test_maes,
        )
    else:
        logger.warning(
            "self-learning-physics-refit: too few test rows (%d) for an honest test "
            "report - skipping (the deploy decision above is unaffected).",
            len(df_house_test),
        )

    final_model = SelfLearningPhysicsModel(
        forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
    )
    final_model.fit(
        df_raw,
        dfs_by_room,
        df_raw["electric_power"].to_numpy(),
        None if electric_only else df_raw["gas_consumption"].to_numpy(),
        neighbor_map,
    )
    deployed = await save_pickle_blob(
        emhass_conf, "self_learning_physics_model.pkl", final_model, logger, keep_previous=True
    )
    result["deployed"] = deployed

    if deployed:
        # Save the learned coupling coefficients as their own small,
        # human-readable blob (independent of the pickled model itself) so
        # _append_room_thermal_loads can load just this at config-build time
        # without unpickling the whole model - only consulted at all when
        # self_learning_physics_coupling_source == "auto_dispatch" (default
        # "informational" never reads this file).
        room_names = optim_conf.get("heatpump_room_names", []) or []
        room_volumes = optim_conf.get("heatpump_room_volume", []) or []
        room_thermal_mass_kj_per_k = {}
        for i, name in enumerate(room_names):
            name = str(name).strip()
            if not name or name not in dfs_by_room:
                continue
            volume = float(room_volumes[i]) if i < len(room_volumes) else 15.0
            # 2400 kg/m3 * 0.88 kJ/(kg*K): the same density/heat_capacity
            # defaults optimization.py::_add_thermal_battery_constraints
            # itself falls back to when a room doesn't override them (no
            # per-room override field exists for either today).
            room_thermal_mass_kj_per_k[name] = 2400.0 * 0.88 * max(0.05, volume)
        coupling_coefficients = final_model.coupling_coefficients_kw_per_k(
            room_thermal_mass_kj_per_k, dt_hours
        )
        coupling_blob = {
            "pairs": [
                {"room_a": pair[0], "room_b": pair[1], "conductance_kw_per_k": g}
                for pair, g in coupling_coefficients.items()
            ],
            "fitted_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
            "dt_hours": dt_hours,
        }
        await save_json_blob(emhass_conf, "self_learning_physics_coupling.json", coupling_blob, logger)

        # Per-room dispatch coefficients (opt-in, see heatpump_room_self_learning_only):
        # a small, human-readable serialization of every room's OWN fitted
        # temperature-recurrence coefficients (not just its neighbor-diff
        # slice, unlike coupling_blob above) - utils.py::_append_room_thermal_loads
        # loads this and attaches it to a flagged room's thermal_battery
        # config, and optimization.py uses it as that room's actual dispatch
        # equation. Saved unconditionally alongside the other artifacts on
        # every successful deploy (independent of whether any room is
        # currently flagged - a room can be flagged later without needing a
        # fresh refit first, as long as one has run since this file existed).
        #
        # Per-room hard requirement: a room's own fitted equation is only
        # included here when it actually beats the physics/simple model's
        # own MAE for that same room over the same holdout window
        # (_score_physics_baseline_room_maes) - a room that doesn't clear
        # this bar is simply left out of "rooms" below, so
        # utils.py::_append_room_thermal_loads treats it exactly like "no
        # fitted model yet" and falls back to the physics/simple model with
        # its usual warning. Deliberately no separate absolute MAE
        # threshold any more - "better than the alternative" is the only
        # bar that matters here.
        dispatch_rooms = {}
        for room_name, room_model in final_model.room_models_.items():
            self_mae = val_scores["room_temp_mae_c"].get(room_name)
            physics_mae = physics_baseline_val_maes.get(room_name)
            if self_mae is None or physics_mae is None:
                logger.warning(
                    "self-learning-physics-refit: room %s has no comparable holdout score "
                    "(self-learning or physics-baseline MAE missing) - not deploying "
                    "dispatch coefficients for this room this refit.",
                    room_name,
                )
                continue
            if self_mae >= physics_mae:
                logger.info(
                    "self-learning-physics-refit: room %s's fitted model (MAE=%.3f°C) does not "
                    "beat the physics/simple baseline (MAE=%.3f°C) for this room - dispatch stays "
                    "on the physics/simple model, not deployed as self-learning dispatch.",
                    room_name, self_mae, physics_mae,
                )
                continue
            dispatch_rooms[room_name] = {
                "feature_names": list(room_model.feature_names),
                "theta": [float(c) for c in room_model.theta_temp],
                "neighbors": list(room_model.neighbors),
                # This room's holdout residual std (deg C) - the Kalman
                # opening detector's own measurement-noise variance R for
                # this room (residual_std_c ** 2), see
                # opening_kalman_detector.py's SELF_LEARNING_KALMAN_* constants.
                "residual_std_c": val_scores["room_temp_residual_std_c"].get(room_name),
            }
        result["rooms_using_self_learning_dispatch"] = list(dispatch_rooms.keys())
        # Whole-house electric-draw regression (theta_elec_/house_feature_names_) -
        # unlike every room's own theta_temp above, this is never gated on
        # beating a baseline (there's no separate baseline electric-draw
        # model to compare against): always included whenever the fit
        # produced one, which final_model.fit always does unconditionally.
        # Only consumed by utils.py::_append_room_thermal_loads for a room
        # whose resolved heat pump unit has control_mode == "weather_curve"'s
        # exact-MILP dispatch (see optimization.py) - a config with no such
        # room pays zero cost for this being present.
        house_elec_blob = (
            {
                "feature_names": list(final_model.house_feature_names_),
                "theta": [float(c) for c in final_model.theta_elec_],
            }
            if final_model.theta_elec_ is not None
            else None
        )
        dispatch_blob = {
            "rooms": dispatch_rooms,
            "house_elec": house_elec_blob,
            "fitted_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
            "dt_hours": dt_hours,
            # Every configured room's own held-out val_scores MAE -
            # deliberately NOT gated on beating the physics baseline the way
            # "rooms" above is (that gate decides dispatch eligibility, a
            # different question from "how accurate is this room's forecast
            # in absolute terms") - lets a cross-family forecast selector
            # (RC vs self-learning-physics, see heating_forecast_model_selection)
            # read this family's own accuracy without re-fitting anything.
            "room_temp_mae_c": val_scores["room_temp_mae_c"],
        }
        await save_json_blob(
            emhass_conf,
            "self_learning_physics_room_dispatch_coefficients.json",
            dispatch_blob,
            logger,
            keep_previous=True,
        )

        # Candidate-neighbor suggestions (informational only, never applied
        # automatically): probe every OTHER configured room as a candidate
        # neighbor - not just the ones already declared via
        # heatpump_room_coupled_neighbors - so a real-looking but undeclared
        # relationship can at least be surfaced for a human to consider.
        # Gated on the same self_learning_physics_coupling_enabled flag as
        # the declared-pair fit above: if the user has coupling turned off
        # entirely, suggesting new pairs to couple would be inconsistent.
        candidate_couplings: list[dict] = []
        if optim_conf.get("self_learning_physics_coupling_enabled", True) and len(dfs_by_room) > 1:
            declared_pairs = {
                tuple(sorted((name, neighbor)))
                for name, neighbors in neighbor_map.items()
                for neighbor in neighbors
            }
            probe_neighbor_map = {
                name: [other for other in dfs_by_room if other != name] for name in dfs_by_room
            }
            candidate_probe_model = SelfLearningPhysicsModel(
                forgetting_factor=forgetting_factor, ridge=ridge, electric_only=electric_only
            )
            candidate_probe_model.fit(
                df_raw,
                dfs_by_room,
                df_raw["electric_power"].to_numpy(),
                None if electric_only else df_raw["gas_consumption"].to_numpy(),
                probe_neighbor_map,
            )
            probe_coupling = candidate_probe_model.coupling_coefficients_kw_per_k(
                room_thermal_mass_kj_per_k, dt_hours
            )
            for pair, g in probe_coupling.items():
                if pair in declared_pairs or g <= _CANDIDATE_COUPLING_MIN_KW_PER_K:
                    continue
                candidate_couplings.append(
                    {"room_a": pair[0], "room_b": pair[1], "suggested_conductance_kw_per_k": g}
                )
                logger.info(
                    "self-learning-physics-refit: possible undeclared coupling between "
                    "%s and %s (~%.3f kW/K) - informational only, never applied "
                    "automatically. Add both rooms to each other's "
                    "heatpump_room_coupled_neighbors (with a manual "
                    "heatpump_room_coupling_conductance placeholder) yourself if you "
                    "want to test this pairing.",
                    pair[0],
                    pair[1],
                    g,
                )
            if candidate_couplings:
                await save_json_blob(
                    emhass_conf,
                    "self_learning_physics_coupling_candidates.json",
                    {
                        "candidates": candidate_couplings,
                        "fitted_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
                        "dt_hours": dt_hours,
                    },
                    logger,
                )
        result["candidate_couplings"] = candidate_couplings

        # Candidate opening-event suggestions (informational only, never
        # applied automatically - same role as candidate_couplings above):
        # only ever populated for rooms Phase 2's EM relabeling loop
        # actually touched (opening_relabel_diagnostics is {} unless
        # self_learning_physics_opening_relabel_enabled is on), so a
        # sensored room can never appear here.
        candidate_openings: list[dict] = []
        for room_name, diagnostics in opening_relabel_diagnostics.items():
            room_index = dfs_by_room[room_name].index
            events = _extract_contiguous_open_events(diagnostics, room_index)[
                :_CANDIDATE_OPENING_EVENT_MAX_PER_ROOM
            ]
            for event in events:
                candidate_openings.append({"room": room_name, **event})
                logger.info(
                    "self-learning-physics-refit: candidate opening event for room %s "
                    "from %s to %s (%d step(s)) - informational only, never applied "
                    "automatically. Confirm it via the opening-confirmation loop (if "
                    "enabled) or a real heatpump_room_window_sensors/"
                    "heatpump_room_door_sensors entry to make it permanent.",
                    room_name,
                    event["start_iso"],
                    event["end_iso"],
                    event["n_steps"],
                )
        if candidate_openings:
            await save_json_blob(
                emhass_conf,
                "self_learning_physics_opening_candidates.json",
                {
                    "candidates": candidate_openings,
                    "fitted_at_iso": pd.Timestamp.now(tz="UTC").isoformat(),
                    "dt_hours": dt_hours,
                },
                logger,
            )
        result["candidate_openings"] = candidate_openings

        if optim_conf.get("self_learning_physics_opening_confirm_enabled", False):
            await _publish_opening_confirmation_questions(
                rh, emhass_conf, optim_conf, retrieve_hass_conf, candidate_openings, logger
            )

        # Blind-position relabeling result surface - informational, mirrors
        # rooms_using_self_learning_dispatch/candidate_openings' own role.
        # No candidate-event list for this feature (deliberate scope
        # decision, see _em_relabel_blind_position's own docstring): the
        # continuous position curve itself is a strictly richer signal than
        # a derived discrete-event view.
        result["blind_position_relabel"] = {
            name: {
                "mean_position": (
                    float(np.nanmean(diag["position"])) if len(diag["position"]) else None
                ),
                "n_informative_steps": diag["n_informative"],
                "beta_blind_x_dni": diag["beta"],
            }
            for name, diag in blind_relabel_diagnostics.items()
        }

    logger.info(
        "self-learning-physics-refit: deployed=%s electric_only=%s electric_mae_w=%.2f "
        "n_rooms=%d (n_rows=%d, window_days=%d)",
        deployed,
        electric_only,
        val_scores["electric_mae_w"],
        len(dfs_by_room),
        n_rows,
        window_days,
    )
    return result


async def refit_enabled_thermal_models(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Refit whichever of the three heat pump thermal models are actually
    enabled - heating-model-refit (heating_model_refit_enabled),
    hybrid-heatpump-model-refit (hybrid_heatpump_refit_enabled), and
    self-learning-physics-refit (self_learning_physics_refit_enabled) - in
    one call.

    A convenience action for the common case of using exactly one of these
    models (or wanting all configured ones refit together on the same
    schedule): a single button/automation works regardless of which model(s)
    are turned on, instead of needing to know and wire up one automation per
    model. Each individual action above still exists standalone for anyone
    who wants independent refit schedules per model.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: {model_key: result_dict_or_None} for every model whose own
        _enabled flag is set, or None if none of the three are enabled.
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    results: dict[str, dict | None] = {}

    if optim_conf.get("heating_model_refit_enabled", False):
        results["heating_model"] = await refit_heating_model(input_data_dict, logger)
    if optim_conf.get("hybrid_heatpump_refit_enabled", False):
        results["hybrid_heatpump_model"] = await refit_hybrid_heatpump_model(input_data_dict, logger)
    if optim_conf.get("self_learning_physics_refit_enabled", False):
        results["self_learning_physics_model"] = await refit_self_learning_physics_model(
            input_data_dict, logger
        )

    if not results:
        logger.warning(
            "thermal-models-refit: none of heating_model_refit_enabled/"
            "hybrid_heatpump_refit_enabled/self_learning_physics_refit_enabled "
            "is turned on - nothing to refit."
        )
        return None
    return results


_SELF_LEARNING_PHYSICS_TUNE_FF_GRID = [0.95, 0.98, 0.99, 0.995, 0.999]
_SELF_LEARNING_PHYSICS_TUNE_RIDGE_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]


async def tune_self_learning_physics_model(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Grid-search forgetting_factor x ridge for self-learning-physics (25
    candidates - 5x5 - each a cheap RLS fit+open-loop-score via the same
    _fit_and_score scorer refit's own val-scoring probe uses) - picks
    whichever combination minimizes mean per-room room_temp_mae_c on val
    (room-temperature accuracy is what actually matters for dispatch;
    electric/gas MAE are secondary and deliberately not part of the
    objective). Deploys the winner fit on the full data (train+val+test),
    overwriting self_learning_physics_model.pkl - same "tune re-deploys
    immediately, doesn't persist to config" contract forecast-model-tune
    already established (see MLForecaster.tune - the winning hyperparameters
    live only in the re-fit model pickle, no separate config/JSON write).

    Grid search rather than Bayesian/optuna (unlike forecast_model_tune):
    only 2 continuous parameters here (vs. lags + several model-specific
    hyperparameters there), a 5x5 grid fully covers the space, is
    deterministic/reproducible, and each candidate is a cheap RLS fit (no
    backtest-refit loop like skforecast's) - no real sample-efficiency need
    for Bayesian search.

    Deliberately searches on the room's plain/baseline data - relabel
    enhancement (self_learning_physics_opening_relabel_enabled/
    self_learning_physics_blind_relabel_enabled) is an orthogonal concern
    handled only by refit_self_learning_physics_model's own per-room
    auto-selection; a user running both features gets the winning
    forgetting_factor/ridge surfaced in this result, which they can copy
    into config (self_learning_physics_forgetting_factor/
    self_learning_physics_ridge) to also apply them on a subsequent
    relabel-aware plain refit, since tuning doesn't update those config
    defaults itself.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: A summary dict for the web UI, or None when disabled/failed
    :rtype: dict | None
    """
    emhass_conf = input_data_dict["emhass_conf"]

    prep = await _prepare_self_learning_physics_fit_data(input_data_dict, logger)
    if prep is None:
        return None
    electric_only = prep["electric_only"]
    df_raw = prep["df_raw"]
    dfs_by_room = prep["dfs_by_room"]
    neighbor_map = prep["neighbor_map"]
    horizon_steps = prep["horizon_steps"]
    split1 = prep["split1"]
    split2 = prep["split2"]

    from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

    rooms_train, rooms_val, _rooms_test = _split_rooms_by_time(dfs_by_room, split1, split2)
    df_house_train = df_raw[df_raw.index < split1]
    df_house_val = df_raw[(df_raw.index >= split1) & (df_raw.index < split2)]
    fit_and_score = _make_self_learning_physics_scorer(electric_only, neighbor_map, horizon_steps)

    def _mean_room_temp_mae(ff: float, ridge: float) -> float:
        candidate = SelfLearningPhysicsModel(forgetting_factor=ff, ridge=ridge, electric_only=electric_only)
        scores = fit_and_score(candidate, df_house_train, rooms_train, df_house_val, rooms_val)
        return float(np.mean(list(scores["room_temp_mae_c"].values())))

    best: tuple[float, float, float] | None = None  # (mean_val_mae, forgetting_factor, ridge)
    for ff in _SELF_LEARNING_PHYSICS_TUNE_FF_GRID:
        for ridge in _SELF_LEARNING_PHYSICS_TUNE_RIDGE_GRID:
            mean_mae = _mean_room_temp_mae(ff, ridge)
            if best is None or mean_mae < best[0]:
                best = (mean_mae, ff, ridge)

    default_mae = _mean_room_temp_mae(0.995, 10.0)
    best_mae, best_ff, best_ridge = best

    final_model = SelfLearningPhysicsModel(
        forgetting_factor=best_ff, ridge=best_ridge, electric_only=electric_only
    )
    final_model.fit(
        df_raw,
        dfs_by_room,
        df_raw["electric_power"].to_numpy(),
        None if electric_only else df_raw["gas_consumption"].to_numpy(),
        neighbor_map,
    )
    deployed = await save_pickle_blob(
        emhass_conf, "self_learning_physics_model.pkl", final_model, logger
    )

    logger.info(
        "self-learning-physics-tune: best forgetting_factor=%.3f, ridge=%.1f "
        "(val room_temp_mae_c=%.4f, vs. %.4f for the config default) over %d candidates.",
        best_ff,
        best_ridge,
        best_mae,
        default_mae,
        len(_SELF_LEARNING_PHYSICS_TUNE_FF_GRID) * len(_SELF_LEARNING_PHYSICS_TUNE_RIDGE_GRID),
    )

    result = {
        "deployed": deployed,
        "best_forgetting_factor": best_ff,
        "best_ridge": best_ridge,
        "best_val_room_temp_mae_c": round(best_mae, 4),
        "default_val_room_temp_mae_c": round(default_mae, 4),
        "n_candidates_tried": len(_SELF_LEARNING_PHYSICS_TUNE_FF_GRID)
        * len(_SELF_LEARNING_PHYSICS_TUNE_RIDGE_GRID),
        "room_temp_test_plot_df": {},
    }

    # Honest held-out test chart: refit on train+val (never on test itself)
    # with the WINNING hyperparameters, score once on the test split -
    # reused as-is by get_injection_dict_thermal_models (same
    # room_temp_test_plot_df shape refit_self_learning_physics_model's own
    # honest-test-report already builds), purely for visibility. Unlike
    # refit, there's no per-room deploy gate or physics-baseline comparison
    # to also compute here - tuning always deploys its winner (see the
    # module docstring above), so this is just the chart.
    df_house_test = prep["df_house_test"]
    if len(df_house_test) >= 10:
        _, _, rooms_test = _split_rooms_by_time(dfs_by_room, split1, split2)
        df_house_trainval = df_raw[df_raw.index < split2]
        rooms_trainval = {n: d[d.index < split2] for n, d in dfs_by_room.items()}
        trainval_model = SelfLearningPhysicsModel(
            forgetting_factor=best_ff, ridge=best_ridge, electric_only=electric_only
        )
        test_scores = fit_and_score(
            trainval_model, df_house_trainval, rooms_trainval, df_house_test, rooms_test,
            collect_series=True,
        )
        for room_name, pred_series in test_scores.get("room_temp_pred_series", {}).items():
            actual_train = rooms_trainval[room_name]["room_temp"]
            actual_test = test_scores["room_temp_actual_test_series"][room_name]
            plot_index = actual_train.index.union(actual_test.index).union(pred_series.index)
            df_plot = pd.DataFrame(index=plot_index, columns=["train", "test", "pred"], dtype=float)
            df_plot.loc[actual_train.index, "train"] = actual_train.to_numpy()
            df_plot.loc[actual_test.index, "test"] = actual_test.to_numpy()
            df_plot.loc[pred_series.index, "pred"] = pred_series.to_numpy()
            result["room_temp_test_plot_df"][room_name] = df_plot

    return result


async def tune_enabled_thermal_models(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Tune whichever thermal model(s) actually have a tunable-hyperparameter
    or warm-startable surface AND are enabled - self-learning-physics (a
    forgetting_factor x ridge grid search) and heating-model (a warm-started,
    cheaper-than-refit re-fit - see tune_heating_model). hybrid-heatpump has
    neither and stays out. Both gated on the SAME flag tuning shares with
    their own refit (tuning has identical prerequisites to refitting, no
    separate config flag). Structured as a fan-out (not a direct call),
    mirroring refit_enabled_thermal_models, so a future tunable model slots
    in the same way.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: {model_key: result_dict_or_None} for every tunable model whose
        own _enabled flag is set, or None if none are enabled.
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    results: dict[str, dict | None] = {}

    if optim_conf.get("heating_model_refit_enabled", False):
        results["heating_model"] = await tune_heating_model(input_data_dict, logger)
    if optim_conf.get("self_learning_physics_refit_enabled", False):
        results["self_learning_physics_model"] = await tune_self_learning_physics_model(
            input_data_dict, logger
        )

    if not results:
        logger.warning("thermal-models-tune called but no tunable thermal model is enabled")
        return None
    return results


async def _select_heating_forecast_winner(input_data_dict: dict, logger: logging.Logger) -> str:
    """Pick "heating_model" (RC physics) or "self_learning_physics" to
    provide the informational heating-need forecast, when BOTH
    heating_forecast_enabled and self_learning_physics_forecast_enabled are
    on - controlled by heating_forecast_model_selection ("auto" by
    default). A "heating_model"/"self_learning_physics" pin skips the
    comparison entirely and returns that choice directly.

    "auto" compares each family's own LAST-DEPLOYED held-out accuracy -
    no re-fitting, both numbers are already sitting in each family's own
    deploy-time blob: RC's whole-house val_mae_c (thermal_physics_params.json,
    set by refit_heating_model/tune_heating_model) vs. self-learning-
    physics's mean per-room room_temp_mae_c
    (self_learning_physics_room_dispatch_coefficients.json, set by
    refit_self_learning_physics_model) - and returns whichever is lower.
    Mean-across-rooms is a simplification: exactly right for a single-room
    house (this function doesn't itself decide per-room forecasts), a
    documented approximation for multi-room ones. A family that has never
    successfully deployed is treated as worse than any real number, so the
    other one wins by default; if NEITHER has ever deployed, falls back to
    "heating_model" (arbitrary but harmless - compute_heating_forecast's
    own "no fitted model found" guard will just no-op)."""
    optim_conf = input_data_dict["optim_conf"]
    emhass_conf = input_data_dict["emhass_conf"]
    selection = str(optim_conf.get("heating_forecast_model_selection", "auto") or "auto").lower()
    if selection in ("heating_model", "self_learning_physics"):
        return selection

    rc_blob = await load_json_blob(emhass_conf, "thermal_physics_params.json", logger, default=None)
    rc_mae = rc_blob.get("val_mae_c") if rc_blob else None

    slp_blob = await load_json_blob(
        emhass_conf, "self_learning_physics_room_dispatch_coefficients.json", logger, default=None
    )
    room_maes = (slp_blob or {}).get("room_temp_mae_c") or {}
    slp_mae = float(np.mean(list(room_maes.values()))) if room_maes else None

    if slp_mae is None or (rc_mae is not None and rc_mae <= slp_mae):
        winner = "heating_model"
    else:
        winner = "self_learning_physics"
    logger.info(
        "heating-forecast model selection: heating_model val_mae_c=%s vs self_learning_physics "
        "mean room_temp_mae_c=%s - using %s",
        f"{rc_mae:.3f}" if rc_mae is not None else "n/a",
        f"{slp_mae:.3f}" if slp_mae is not None else "n/a",
        winner,
    )
    return winner


async def compute_enabled_thermal_forecasts(input_data_dict: dict, logger: logging.Logger) -> dict | None:
    """Forecast whichever of the three heat pump thermal models are
    actually enabled - heating-need-forecast (heating_forecast_enabled),
    hybrid-heatpump-forecast (hybrid_heatpump_forecast_enabled), and
    self-learning-physics-forecast (self_learning_physics_forecast_enabled)
    - in one call. Predict-side sibling of refit_enabled_thermal_models,
    identical fan-out shape - EXCEPT for heating_model/self_learning_physics:
    when BOTH of those two's own _forecast_enabled flags are on, only the
    WINNER of _select_heating_forecast_winner actually runs (see that
    function's own docstring) - the whole point of choosing a model family
    is one authoritative informational forecast, not two independently
    published, possibly-disagreeing ones. When only one of the two is
    enabled, it just runs directly, unchanged from before this existed.
    hybrid-heatpump stays a plain independent fan-out branch - it predicts
    a different target (electric/gas draw, not room temperature) and was
    never part of this selection to begin with.

    :param input_data_dict: A dictionnary with multiple data used by the action functions
    :type input_data_dict: dict
    :param logger: The passed logger object
    :type logger: logging.Logger
    :return: {model_key: result_dict_or_None} for every model whose own
        _forecast_enabled flag is set, or None if none of the three are
        enabled.
    :rtype: dict | None
    """
    optim_conf = input_data_dict["optim_conf"]
    results: dict[str, dict | None] = {}

    heating_enabled = bool(optim_conf.get("heating_forecast_enabled", False))
    self_learning_enabled = bool(optim_conf.get("self_learning_physics_forecast_enabled", False))
    if heating_enabled and self_learning_enabled:
        winner = await _select_heating_forecast_winner(input_data_dict, logger)
        if winner == "self_learning_physics":
            results["self_learning_physics_model"] = await compute_self_learning_physics_forecast(
                input_data_dict, logger
            )
        else:
            results["heating_model"] = await compute_heating_forecast(input_data_dict, logger)
    elif heating_enabled:
        results["heating_model"] = await compute_heating_forecast(input_data_dict, logger)
    elif self_learning_enabled:
        results["self_learning_physics_model"] = await compute_self_learning_physics_forecast(
            input_data_dict, logger
        )

    if optim_conf.get("hybrid_heatpump_forecast_enabled", False):
        results["hybrid_heatpump_model"] = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

    if not results:
        logger.warning(
            "thermal-models-forecast: none of heating_forecast_enabled/"
            "hybrid_heatpump_forecast_enabled/self_learning_physics_forecast_enabled "
            "is turned on - nothing to forecast."
        )
        return None
    return results


async def compute_self_learning_physics_forecast(
    input_data_dict: dict, logger: logging.Logger
) -> dict | None:
    """Forecast electric power (and gas consumption, unless electric_only),
    plus every configured room's own temperature, forward from now, using
    the fitted self-learning-physics model (see
    refit_self_learning_physics_model above).

    Informational only, same "publish only" pattern as
    compute_hybrid_heatpump_forecast - EMHASS never calls a device service
    here. Heat pump duty is resolved via the same
    _resolve_aggregate_duty_trajectory helper (latest solved dispatch plan
    when one exists, else the last observed heatpump_duty_sensor value held
    constant).

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

    if not optim_conf.get("self_learning_physics_forecast_enabled", False):
        logger.debug(
            "self-learning-physics-forecast: disabled (self_learning_physics_forecast_enabled=False)"
        )
        return None

    model = await load_pickle_blob(emhass_conf, "self_learning_physics_model.pkl", logger, default=None)
    if model is None:
        logger.error(
            "self-learning-physics-forecast: no fitted model found "
            "(data/self_learning_physics_model.pkl). Run the "
            "self-learning-physics-refit action at least once."
        )
        return None

    room_entity_map = _resolve_room_temp_entity_map(optim_conf, retrieve_hass_conf)
    room_names = [name for name in room_entity_map if name in model.room_models_]
    if not room_names:
        logger.error(
            "self-learning-physics-forecast: no configured room matches the fitted model's "
            "own rooms - re-run self-learning-physics-refit after changing the room list."
        )
        return None
    blind_entity_map = _resolve_room_blind_entity_map(optim_conf, retrieve_hass_conf)

    live_sensor_keys = [
        "heatpump_duty_sensor",
        "heatpump_flow_temp_sensor",
        "heatpump_power_sensor",
        "heatpump_gas_meter_sensor",
    ]
    live_entities = [retrieve_hass_conf.get(k, "") for k in live_sensor_keys]
    live_entities = [e for e in live_entities if e]
    live_entities += [room_entity_map[name] for name in room_names]
    live_entities += [blind_entity_map[name] for name in room_names if name in blind_entity_map]
    live_entities = list(dict.fromkeys(live_entities))
    if not live_entities:
        logger.error(
            "self-learning-physics-forecast: no live sensors configured to read the current state from"
        )
        return None

    days_list = utils.get_days_list(2)
    if not await rh.get_data(days_list, live_entities):
        logger.error("self-learning-physics-forecast: failed to retrieve live sensor data from Home Assistant")
        return None
    rh.prepare_data(
        live_entities[0],
        load_negative=False,
        set_zero_min=False,
        var_replace_zero=[],
        var_interp=live_entities,
        skip_renaming=True,
    )

    def _last_value(entity_id: str, default: float) -> float:
        if not entity_id or entity_id not in rh.df_final.columns:
            return default
        series = rh.df_final[entity_id].dropna()
        return float(series.iloc[-1]) if not series.empty else default

    def _last_delta_value(entity_id: str, default: float, rate_dt_hours: float | None = None) -> float:
        # Same cumulative-meter detection as the refit's own training data
        # (utils.resolve_incremental_series) - a raw gas/energy totalizer's
        # bare last value would otherwise seed the model with a huge,
        # out-of-distribution "gas/electric used this step" reading.
        if not entity_id or entity_id not in rh.df_final.columns:
            return default
        series = rh.df_final[entity_id].dropna()
        if series.empty:
            return default
        delta = utils.resolve_incremental_series(
            series, entity_id, logger, rate_dt_hours=rate_dt_hours
        )
        return float(delta.iloc[-1])

    from emhass.thermal.thermal_mass_physics import _infer_timestep_hours

    live_dt_hours = _infer_timestep_hours(rh.df_final.index)
    last_duty = _last_value(retrieve_hass_conf.get("heatpump_duty_sensor", ""), 0.0)
    last_supply_temp = _last_value(retrieve_hass_conf.get("heatpump_flow_temp_sensor", ""), 25.0)
    last_electric = _last_delta_value(
        retrieve_hass_conf.get("heatpump_power_sensor", ""), 0.0, rate_dt_hours=live_dt_hours
    )
    last_gas = _last_delta_value(retrieve_hass_conf.get("heatpump_gas_meter_sensor", ""), 0.0)
    initial_room_states = {
        name: _last_value(room_entity_map[name], 20.0) for name in room_names
    }

    df_weather = await input_data_dict["fcst"].get_weather_forecast(
        method=optim_conf.get("weather_forecast_method", "open-meteo")
    )
    if isinstance(df_weather, bool) and not df_weather:
        logger.error("self-learning-physics-forecast: failed to retrieve a weather forecast")
        return None
    if df_weather is None or len(df_weather) == 0:
        logger.error("self-learning-physics-forecast: weather forecast is empty")
        return None

    duty_trajectory = _resolve_aggregate_duty_trajectory(
        input_data_dict, df_weather.index, last_duty, logger
    )

    df_house_fc = pd.DataFrame(
        {
            "outdoor_temp": df_weather["temp_air"],
            "wind_speed": df_weather["wind_speed"],
            "dni": df_weather["dni"],
            "dhi": df_weather["dhi"],
            "heatpump_duty": duty_trajectory,
            "supply_temp": last_supply_temp,
            "group_duty": duty_trajectory,
        },
        index=df_weather.index,
    )
    # Sun position over the forecast horizon, feeding sun_alt_sin/
    # dni_x_sun_az_sin/cos - deterministic from timestamp+location (via
    # pvlib), so unlike dni/dhi it has zero forecast uncertainty. Same
    # Forecast.compute_solar_angles + sin/cos conversion
    # refit_self_learning_physics_model/prepare_forecast_and_weather_data
    # already use, applied here to the forecast horizon so the published
    # forecast stays consistent with what the model was actually fit on.
    solar_angles = Forecast.compute_solar_angles(
        df_house_fc,
        latitude=float(retrieve_hass_conf["Latitude"]),
        longitude=float(retrieve_hass_conf["Longitude"]),
    )
    alt_rad = np.radians(solar_angles["solar_elevation"].to_numpy(dtype=float))
    az_rad = np.radians(solar_angles["solar_azimuth"].to_numpy(dtype=float))
    df_house_fc["sun_alt_sin"] = np.sin(alt_rad)
    df_house_fc["sun_az_sin"] = np.sin(az_rad)
    df_house_fc["sun_az_cos"] = np.cos(az_rad)
    dfs_by_room_fc = {
        name: df_house_fc.copy() for name in room_names
    }
    # Room's own live blind/shading position, held flat across the whole
    # forecast horizon - same "no per-room forecast infra, hold the last
    # live reading" simplification already used for last_supply_temp/
    # last_duty above. Rooms with no configured blind sensor simply don't
    # get the column - model.predict_recursive's own _physics_features call
    # already defaults it to 0.0 (blind always open) for those.
    for name in room_names:
        blind_entity_id = blind_entity_map.get(name)
        if blind_entity_id:
            dfs_by_room_fc[name]["blind_position"] = _last_value(blind_entity_id, 0.0)

    # Deliberately NOT mirroring the blind_position pattern above:
    # opening_open/door_open (window/door "is it open") are fast, momentary,
    # live-only signals with no way to forecast future events, unlike a
    # blind position which tends to persist for hours. Holding today's live
    # reading flat across the WHOLE future forecast horizon would wrongly
    # assume a window stays open (or closed) for the entire period. Simply
    # never populating these columns here lets _physics_features's own
    # default-to-0.0 fallback make the whole forecast horizon "assumed
    # closed" - the safe direction - while the REFIT training data (built
    # from real historical per-timestamp sensor readings, see
    # refit_self_learning_physics_model) still teaches the model each room's
    # real response to a genuinely open window/door.
    pred = model.predict_recursive(
        df_house_fc,
        dfs_by_room_fc,
        initial_room_states,
        initial_house_elec=last_electric,
        initial_house_gas=last_gas,
    )
    electric_only = model.theta_gas_ is None

    passed_data = input_data_dict["params"]["passed_data"]
    electric_entity = passed_data.get("custom_self_learning_physics_electric_forecast_id")
    gas_entity = passed_data.get("custom_self_learning_physics_gas_forecast_id")
    room_temp_entities = passed_data.get("custom_self_learning_physics_temp_forecast_id", [])
    if electric_entity is None or (not electric_only and gas_entity is None):
        logger.error(
            "self-learning-physics-forecast: target entities not registered "
            "(self_learning_physics_forecast_enabled was True at optim time but isn't now?)"
        )
        return None

    common_kwargs = {
        "publish_prefix": passed_data.get("publish_prefix", ""),
        "save_entities": False,
        "dont_post": passed_data.get("dont_post", False),
    }
    await rh.post_data(
        pd.Series(pred["electric_power"], index=df_house_fc.index),
        0,
        electric_entity["entity_id"],
        electric_entity["device_class"],
        electric_entity["unit_of_measurement"],
        electric_entity["friendly_name"],
        type_var="power",
        **common_kwargs,
    )
    if not electric_only:
        await rh.post_data(
            pd.Series(pred["gas_consumption"], index=df_house_fc.index),
            0,
            gas_entity["entity_id"],
            gas_entity["device_class"],
            gas_entity["unit_of_measurement"],
            gas_entity["friendly_name"],
            type_var="energy",
            **common_kwargs,
        )

    room_temp_entity_by_name = dict(zip(room_names, room_temp_entities, strict=False))
    mean_room_temps: dict[str, float] = {}
    for name in room_names:
        entity_conf = room_temp_entity_by_name.get(name)
        if entity_conf is None:
            continue
        series = pd.Series(pred["room_temp"][name], index=df_house_fc.index)
        mean_room_temps[name] = float(series.mean())
        await rh.post_data(
            series,
            0,
            entity_conf["entity_id"],
            entity_conf["device_class"],
            entity_conf["unit_of_measurement"],
            entity_conf["friendly_name"],
            type_var="temperature",
            **common_kwargs,
        )

    result = {
        "electric_only": electric_only,
        "forecast_steps": len(df_house_fc),
        "n_rooms": len(room_names),
        "mean_electric_forecast_w": float(np.mean(pred["electric_power"])),
        "mean_gas_forecast_m3": None if electric_only else float(np.mean(pred["gas_consumption"])),
        "mean_room_temps_c": mean_room_temps,
    }
    await save_json_blob(emhass_conf, "self_learning_physics_forecast_last_run.json", result, logger)
    logger.info(
        "self-learning-physics-forecast: electric_only=%s mean_electric_forecast_w=%.1f n_rooms=%d",
        electric_only,
        result["mean_electric_forecast_w"],
        result["n_rooms"],
    )
    # Added AFTER the JSON persist above (orjson.dumps can't serialize a
    # Series/dict-of-Series) - the forecasted curves themselves, rendered by
    # get_injection_dict_thermal_models/get_forecast_trend_plot_html.
    result["electric_forecast_series"] = pd.Series(pred["electric_power"], index=df_house_fc.index)
    if not electric_only:
        result["gas_forecast_series"] = pd.Series(pred["gas_consumption"], index=df_house_fc.index)
    result["room_temp_forecast_df"] = {
        name: pd.Series(pred["room_temp"][name], index=df_house_fc.index) for name in room_names
    }
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
    from emhass.thermal import (
        ModelRegistry,
        build_two_stage_optimization_plan,
        load_target_registries,
    )

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


async def _publish_room_supply_temp_target(
    ctx: PublishContext, opt_res_latest: pd.DataFrame
) -> list[str]:
    """Publish each room's solved supply/flow-temperature target - only for
    a room actually dispatched via weather_curve's exact-MILP path (see
    optimization.py::_add_self_learning_dispatch_milp_constraints), i.e.
    whose solved results carry a supply_temp_target_heater{k} column. An
    additional, optional signal alongside _publish_room_targets's indoor-
    temperature target, letting the user choose which one to wire into
    their own automation - EMHASS never calls the HA service directly, it
    only publishes this target sensor.

    Deliberately not routed through _publish_thermal_variable (unlike
    _publish_thermal_loads): custom_room_supply_temp_target_id is indexed
    by room-enumeration position i (one entry per room, same convention as
    custom_room_target_temp_id - see utils.py::_append_room_thermal_loads),
    while the results column itself is indexed by k, the room's absolute
    def_load_config/P_deferrable index - the two only coincide when rooms
    are the only deferrable loads configured. Same i/k split
    _publish_room_targets already uses for this exact reason.
    """
    cols = []
    room_load_indices = ctx.params["passed_data"].get("room_load_indices", {})
    if not room_load_indices:
        return cols
    custom_target = ctx.params["passed_data"].get("custom_room_supply_temp_target_id", [])

    for i, (_name, k) in enumerate(room_load_indices.items()):
        if i >= len(custom_target):
            continue
        col_name = f"supply_temp_target_heater{k}"
        if col_name not in opt_res_latest.columns:
            continue
        entity_conf = custom_target[i]
        await ctx.rh.post_data(
            opt_res_latest[col_name],
            ctx.idx,
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
    cols_published.extend(await _publish_room_supply_temp_target(ctx, opt_res_latest))
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
        forecast-calibration, heating-need-forecast, heating-model-refit,\
        hybrid-heatpump-forecast, hybrid-heatpump-model-refit,\
        self-learning-physics-forecast, self-learning-physics-refit, thermal-models-refit,\
        thermal-models-tune, thermal-models-forecast",
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
    elif args.action == "hybrid-heatpump-forecast":
        await compute_hybrid_heatpump_forecast(input_data_dict, logger)
        opt_res = None
    elif args.action == "hybrid-heatpump-model-refit":
        await refit_hybrid_heatpump_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "self-learning-physics-forecast":
        await compute_self_learning_physics_forecast(input_data_dict, logger)
        opt_res = None
    elif args.action == "self-learning-physics-refit":
        await refit_self_learning_physics_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "pv-horizon-refit":
        await refit_pv_horizon_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "pv-forecast-test":
        logger.info(input_data_dict["p_pv_forecast"])
        opt_res = None
    elif args.action == "adjust-pv-forecast-refit":
        await refit_adjust_pv_forecast_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "load-forecast-test":
        logger.info(input_data_dict["p_load_forecast_p10"])
        logger.info(input_data_dict["p_load_forecast_p50"])
        logger.info(input_data_dict["p_load_forecast_p90"])
        opt_res = None
    elif args.action == "load-quantile-spread-refit":
        await refit_load_quantile_spread_model(input_data_dict, logger)
        opt_res = None
    elif args.action == "thermal-models-refit":
        await refit_enabled_thermal_models(input_data_dict, logger)
        opt_res = None
    elif args.action == "thermal-models-tune":
        await tune_enabled_thermal_models(input_data_dict, logger)
        opt_res = None
    elif args.action == "thermal-models-forecast":
        await compute_enabled_thermal_forecasts(input_data_dict, logger)
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
