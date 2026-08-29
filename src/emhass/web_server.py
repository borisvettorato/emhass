#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import pickle
import platform
import re
import shutil
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import aiofiles
import jinja2
import orjson
import uvicorn
import yaml
from markupsafe import Markup
from quart import Quart, make_response, request
from quart import logging as log

from emhass import last_run, plan_store
from emhass.command_line import (
    EMHASS_SCHEMA_VERSION,
    compute_enabled_thermal_forecasts,
    compute_heating_forecast,
    compute_hybrid_heatpump_forecast,
    compute_self_learning_physics_forecast,
    continual_publish,
    dayahead_forecast_optim,
    export_influxdb_to_csv,
    forecast_calibration,
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    naive_mpc_optim,
    perfect_forecast_optim,
    publish_data,
    refit_adjust_pv_forecast_model,
    refit_enabled_thermal_models,
    refit_heating_model,
    refit_hybrid_heatpump_model,
    refit_pv_horizon_model,
    refit_self_learning_physics_model,
    regressor_model_fit,
    regressor_model_predict,
    set_input_data_dict,
    thermal_two_stage_plan,
    tune_enabled_thermal_models,
    weather_forecast_cache,
)
from emhass.connection_manager import close_global_connection, get_websocket_client, is_connected
from emhass.persistence import load_json_blob, save_json_blob
from emhass.pv_shading_kalman import SEASON_LABELS
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import (
    build_config,
    build_legacy_config_params,
    build_params,
    build_secrets,
    get_days_list,
    get_injection_dict,
    get_injection_dict_forecast_calibration,
    get_injection_dict_forecast_model_fit,
    get_injection_dict_forecast_model_tune,
    get_injection_dict_thermal_models,
    get_injection_dict_thermal_two_stage,
    get_keys_to_mask,
    get_room_temp_test_plot_html,
    param_to_config,
    render_horizon_polar_grid,
)

app = Quart(__name__)

emhass_conf: dict[str, Path] = {}
entity_path: Path = Path()
params_secrets: dict[str, str | float] = {}
continual_publish_thread: list = []
injection_dict: dict = {}

templates = jinja2.Environment(
    autoescape=True,
    loader=jinja2.PackageLoader("emhass", "templates"),
)

action_log_str = "action_logs.txt"
injection_dict_file = "injection_dict.pkl"
params_file = "params.pkl"
error_msg_associations_file = "Unable to obtain associations file"


# Add custom filter for trusted HTML content
def mark_safe(value):
    """Mark pre-rendered HTML plots as safe (use only for trusted content)"""
    if value is None:
        return ""
    return Markup(value)


templates.filters["mark_safe"] = mark_safe


def _health_verdict(has_run: bool, stale: bool) -> tuple[str, int]:
    """Map run-existence + staleness to (status, http_code).

    Recency-only: last-run correctness (infeasible/error) does NOT affect the
    health verdict. ``has_run`` is False when no optimization has ever completed;
    ``stale`` is True only when a freshness window was requested and the last run
    falls outside it.
    """
    if not has_run:
        return "degraded", 503
    if stale:
        return "degraded", 503
    return "ok", 200


# Register async startup and shutdown handlers
@app.before_serving
async def before_serving():
    """Initialize EMHASS before starting to serve requests."""
    # Capture boot timestamp first, so /healthz reports it even if initialize() fails.
    app.config["boot_ts"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    # Initialize the application
    try:
        await initialize()
        app.logger.info("Full initialization completed")
    except Exception as e:
        app.logger.warning(f"Full initialization failed (this is normal in test environments): {e}")
        app.logger.info("Continuing without WebSocket connection...")
        # The initialize() function already sets up all necessary components except WebSocket
        # So we can continue serving requests even if WebSocket connection fails


@app.after_serving
async def after_serving():
    """Clean up resources after serving."""
    try:
        # Only close WebSocket connection if it was established
        if is_connected():
            await close_global_connection()
            app.logger.info("WebSocket connection closed")
        else:
            app.logger.info("No WebSocket connection to close")
    except Exception as e:
        app.logger.warning(f"WebSocket shutdown failed: {e}")
    app.logger.info("Quart shutdown complete")


async def check_file_log(ref_string: str | None = None) -> bool:
    """
    Check logfile for error, anything after string match if provided.

    :param ref_string: String to reduce log area to check for errors. Use to reduce log to check anything after string match (ie. an action).
    :type ref_string: str
    :return: Boolean return if error was found in logs
    :rtype: bool

    """
    log_array: list[str] = []

    if ref_string is not None:
        log_array = await grab_log(
            ref_string
        )  # grab reduced log array (everything after string match)
    else:
        if (emhass_conf["data_path"] / action_log_str).exists():
            async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str)) as fp:
                content = await fp.read()
                log_array = content.splitlines()
        else:
            app.logger.debug("Unable to obtain {action_log_str}")
            return False

    for log_string in log_array:
        if log_string.split(" ", 1)[0] == "ERROR":
            return True
    return False


async def grab_log(ref_string: str | None = None) -> list[str]:
    """
    Find string in logs, append all lines after into list to return.

    :param ref_string: String used to string match log.
    :type ref_string: str
    :return: List of lines in log after string match.
    :rtype: list

    """
    is_found = []
    output = []
    if (emhass_conf["data_path"] / action_log_str).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str)) as fp:
            content = await fp.read()
            log_array = content.splitlines()
        # Find all string matches, log key (line Number) in is_found
        for x in range(len(log_array) - 1):
            if re.search(ref_string, log_array[x]):
                is_found.append(x)
        if len(is_found) != 0:
            # Use last item in is_found to extract action logs
            for x in range(is_found[-1], len(log_array)):
                output.append(log_array[x])
    return output


# Clear the log file
async def clear_file_log():
    """
    Clear the contents of the log file

    """
    if (emhass_conf["data_path"] / action_log_str).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / action_log_str), "w") as fp:
            await fp.write("")


@app.route("/")
@app.route("/index")
async def index():
    """
    Render initial index page and serve to web server.
    Appends plot tables saved from previous optimization into index.html, then serves.
    """
    app.logger.info("EMHASS server online, serving index.html...")

    # Load cached dict (if exists), to present generated plot tables
    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            try:
                injection_dict = pickle.loads(content)
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                app.logger.warning(
                    "The data container file is empty or incomplete (possible write race condition). "
                    "Please launch an optimization task."
                )
                injection_dict = {}
    else:
        app.logger.info(
            "The data container dictionary is empty... Please launch an optimization task"
        )
        injection_dict = {}

    # The thermostat nav link only makes sense once a heat pump is configured.
    if (emhass_conf["data_path"] / params_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "rb") as fid:
            content = await fid.read()
            try:
                _, params = pickle.loads(content)
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                params = {}
    else:
        params = {}
    use_heatpump = bool(params.get("optim_conf", {}).get("set_use_heatpump", False))

    template = templates.get_template("index.html")
    return await make_response(
        template.render(injection_dict=injection_dict, use_heatpump=use_heatpump)
    )


@app.route("/thermal-comfort", methods=["GET"])
async def thermal_comfort():
    """Serve the standalone thermal comfort UI as a static page."""
    app.logger.info("serving thermal_comfort.html...")
    return await app.send_static_file("thermal_comfort.html")


_VALID_SCHEDULE_DAYS = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


def _validate_room_schedule_payload(payload: object) -> str | None:
    """Validate a thermal_comfort.html schedule payload shape.

    Returns an error message string if invalid, or None if valid.
    """
    if not isinstance(payload, dict):
        return "payload must be a JSON object"
    week_schedule = payload.get("weekSchedule", {})
    if not isinstance(week_schedule, dict):
        return "weekSchedule must be an object"
    for day, rooms in week_schedule.items():
        if day not in _VALID_SCHEDULE_DAYS:
            return f"unknown day '{day}'"
        if not isinstance(rooms, dict):
            return f"weekSchedule['{day}'] must be an object"
        for room_name, slots in rooms.items():
            if not isinstance(slots, list):
                return f"weekSchedule['{day}']['{room_name}'] must be an array"
            for entry in slots:
                if not isinstance(entry, dict):
                    return f"weekSchedule['{day}']['{room_name}'] entries must be objects"
                temp_min = entry.get("temp_min")
                temp_max = entry.get("temp_max")
                if not isinstance(temp_min, int | float) or not isinstance(temp_max, int | float):
                    return (
                        f"weekSchedule['{day}']['{room_name}'] entries need "
                        "numeric temp_min/temp_max"
                    )
                if temp_min > temp_max:
                    return f"weekSchedule['{day}']['{room_name}'] has temp_min > temp_max"
    presets = payload.get("presets", {})
    if not isinstance(presets, dict):
        return "presets must be an object"
    return None


@app.route("/room-schedule", methods=["GET"])
async def get_room_schedule():
    """Return the persisted per-room weekly comfort schedule, if any."""
    blob = await load_json_blob(
        emhass_conf,
        "room_thermal_schedule.json",
        app.logger,
        default={"weekSchedule": {}, "presets": {}},
    )
    return await make_response(blob, 200)


@app.route("/room-schedule", methods=["POST"])
async def save_room_schedule():
    """Persist the per-room weekly comfort schedule from thermal_comfort.html."""
    try:
        payload = await request.get_json(force=True)
    except Exception:
        return await make_response({"error": "invalid JSON body"}, 400)
    error = _validate_room_schedule_payload(payload)
    if error:
        return await make_response({"error": error}, 400)
    ok = await save_json_blob(
        emhass_conf,
        "room_thermal_schedule.json",
        {"weekSchedule": payload.get("weekSchedule", {}), "presets": payload.get("presets", {})},
        app.logger,
    )
    if not ok:
        return await make_response({"error": "failed to save schedule"}, 500)
    return await make_response({"status": "saved"}, 201)


@app.route("/configuration", methods=["GET", "POST"])
async def configuration():
    """
    Configuration page actions:
    Render and serve configuration page html
    """
    # Define the list of secret parameters managed by the UI
    secret_params = get_keys_to_mask()

    if request.method == "POST":
        app.logger.info("Saving configuration/secrets...")
        form_data = await request.form

        # Load existing secrets
        secrets = {}
        # Ensure we have the path from config, fallback to default if missing
        secrets_path = emhass_conf.get("secrets_path", Path("/app/secrets_emhass.yaml"))

        # Try to load existing secrets to preserve others (Async)
        if secrets_path.exists():
            try:
                async with aiofiles.open(secrets_path) as file:
                    content = await file.read()
                    loaded = yaml.safe_load(content)
                    if loaded:
                        secrets = loaded
            except Exception as e:
                app.logger.error(f"Error reading secrets file: {e}")

        # Update secrets with form data
        updated = False
        for key in secret_params:
            if key in form_data:
                value = form_data[key]
                if value != "***":
                    secrets[key] = value
                    updated = True

        # Save to file if changes were made (Async)
        if updated:
            try:
                async with aiofiles.open(secrets_path, "w") as file:
                    # dump returns string if stream is None
                    content = yaml.dump(secrets, default_flow_style=False)
                    await file.write(content)

                app.logger.info("Secrets saved successfully.")

                # Update the global params_secrets
                global params_secrets
                params_secrets.update(secrets)

            except Exception as e:
                app.logger.error(f"Error saving secrets file: {e}")

    app.logger.info("serving configuration.html...")

    # get params
    if (emhass_conf["data_path"] / params_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "rb") as fid:
            content = await fid.read()
            try:
                _, params = pickle.loads(content)  # Don't overwrite emhass_conf["config_path"]
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                params = {}
    else:
        params = {}

    use_heatpump = bool(params.get("optim_conf", {}).get("set_use_heatpump", False))

    template = templates.get_template("configuration.html")
    return await make_response(
        template.render(config=params, use_heatpump=use_heatpump)
    )


@app.route("/template", methods=["GET"])
async def template_action():
    """
    template page actions:
    Render and serve template html
    """
    app.logger.info(" >> Sending rendered template data")

    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            try:
                injection_dict = pickle.loads(content)
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                app.logger.warning(
                    "The data container file is empty or incomplete (possible write race condition). "
                    "Please launch an optimization task."
                )
                injection_dict = {}
    else:
        app.logger.warning("Unable to obtain plot data from {injection_dict_file}")
        app.logger.warning("Try running an launch an optimization task")
        injection_dict = {}

    template = templates.get_template("template.html")
    return await make_response(template.render(injection_dict=injection_dict))


@app.route("/get-config", methods=["GET"])
async def parameter_get():
    """
    Get request action that builds, formats and sends config as json (config.json format)

    """
    app.logger.debug("Obtaining current saved parameters as config")
    # Build config from all possible sources (inc. legacy yaml config)
    config = await build_config(
        emhass_conf,
        app.logger,
        str(emhass_conf["defaults_path"]),
        str(emhass_conf["config_path"]),
        str(emhass_conf["legacy_config_path"]),
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params (converting legacy json parameters from options.json if any)
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = param_to_config(params, app.logger)
    # Send config
    return await make_response(return_config, 200)


@app.route("/get-washdata-devices", methods=["GET"])
async def get_washdata_devices():
    """
    Discover WashData (ha_washdata custom integration) device slugs
    reachable on the connected Home Assistant instance, for the
    load_washdata_device config UI picker. Looks for
    binary_sensor.<device>_actief entities - WashData's own per-device
    "is running now" sensor, present as soon as a device is being
    monitored, even before it has learned any program yet.
    """
    app.logger.debug("Discovering WashData devices")
    config = await build_config(
        emhass_conf,
        app.logger,
        str(emhass_conf["defaults_path"]),
        str(emhass_conf["config_path"]),
        str(emhass_conf["legacy_config_path"]),
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    rh = RetrieveHass(
        retrieve_hass_conf.get("hass_url", ""),
        retrieve_hass_conf.get("long_lived_token", ""),
        retrieve_hass_conf.get("optimization_time_step", 30),
        retrieve_hass_conf.get("time_zone", ""),
        params,
        emhass_conf,
        app.logger,
    )
    states = await rh.get_all_states()
    suffix = "_actief"
    devices = sorted(
        {
            entity_id.split(".", 1)[1][: -len(suffix)]
            for state in states
            if (entity_id := state.get("entity_id", "")).startswith("binary_sensor.")
            and entity_id.endswith(suffix)
        }
    )
    return await make_response(devices, 200)


@app.route("/get-ha-entities", methods=["GET"])
async def get_ha_entities():
    """
    Fetch every entity known to the connected Home Assistant instance, as a
    lightweight list for the config UI's entity-picker suggestions (any
    "(HA entity)" field - temperature/power/humidity sensors, door/window
    binary sensors, switches, selects, etc.). Only entity_id/friendly_name/
    device_class/unit_of_measurement are kept - the frontend filters by
    domain (from the entity_id prefix), device_class and unit per-field,
    client-side, so a single fetch covers every suggestible field on the
    page.
    """
    app.logger.debug("Fetching Home Assistant entities for config UI suggestions")
    config = await build_config(
        emhass_conf,
        app.logger,
        str(emhass_conf["defaults_path"]),
        str(emhass_conf["config_path"]),
        str(emhass_conf["legacy_config_path"]),
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    rh = RetrieveHass(
        retrieve_hass_conf.get("hass_url", ""),
        retrieve_hass_conf.get("long_lived_token", ""),
        retrieve_hass_conf.get("optimization_time_step", 30),
        retrieve_hass_conf.get("time_zone", ""),
        params,
        emhass_conf,
        app.logger,
    )
    states = await rh.get_all_states()
    entities = [
        {
            "entity_id": entity_id,
            "friendly_name": (state.get("attributes") or {}).get("friendly_name", ""),
            "device_class": (state.get("attributes") or {}).get("device_class", ""),
            "unit_of_measurement": (state.get("attributes") or {}).get(
                "unit_of_measurement", ""
            ),
        }
        for state in states
        if (entity_id := state.get("entity_id", ""))
    ]
    return await make_response(entities, 200)


@app.route("/get-room-temperature-forecast", methods=["GET"])
async def get_room_temperature_forecast():
    """
    Fetch each room's measured temperature history (back to yesterday) and
    live predicted-temperature forecast, straight from Home Assistant, for
    the Thermal Comfort ("Thermostat") page's Temperature Profile chart -
    one continuous, source-agnostic list of {date, value} points per room.

    History comes from the room's own heatpump_room_temp_sensors entity
    (same RetrieveHass.get_data() used by every EMHASS forecaster/tuner,
    e.g. the heating-need-forecast action). Forecast comes from
    sensor.temp_predicted{k}, published with the full remaining forecast
    horizon as its 'predicted_temperatures' attribute on every optimization
    run (see RetrieveHass.post_data); k is each room's own load index, from
    passed_data.room_load_indices (built by _append_room_thermal_loads).
    """
    app.logger.debug("Fetching room temperature history and forecasts")
    config = await build_config(
        emhass_conf,
        app.logger,
        str(emhass_conf["defaults_path"]),
        str(emhass_conf["config_path"]),
        str(emhass_conf["legacy_config_path"]),
    )
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    retrieve_hass_conf = params.get("retrieve_hass_conf", {})
    rh = RetrieveHass(
        retrieve_hass_conf.get("hass_url", ""),
        retrieve_hass_conf.get("long_lived_token", ""),
        retrieve_hass_conf.get("optimization_time_step", 30),
        retrieve_hass_conf.get("time_zone", ""),
        params,
        emhass_conf,
        app.logger,
    )
    room_names = retrieve_hass_conf.get("heatpump_room_names", [])
    room_temp_sensors = retrieve_hass_conf.get("heatpump_room_temp_sensors", [])
    room_load_indices = params.get("passed_data", {}).get("room_load_indices", {})

    result: dict[str, list] = {}

    # History: one batched get_data() call for every room's sensor at once.
    sensor_by_room = {
        name: room_temp_sensors[i]
        for i, name in enumerate(room_names)
        if i < len(room_temp_sensors) and room_temp_sensors[i]
    }
    if sensor_by_room:
        days_list = get_days_list(2)
        if await rh.get_data(days_list, list(sensor_by_room.values())):
            for room_name, entity_id in sensor_by_room.items():
                if entity_id not in rh.df_final.columns:
                    continue
                series = rh.df_final[entity_id].dropna()
                result[room_name] = [
                    {"date": ts.isoformat(), "value": float(v)} for ts, v in series.items()
                ]

    # Forecast: appended after each room's history.
    for room_name, k in room_load_indices.items():
        entity_id = f"sensor.temp_predicted{k}"
        payload = await rh.get_entity_state_and_attributes(entity_id)
        if not payload:
            continue
        raw = (payload.get("attributes") or {}).get("predicted_temperatures")
        if not raw:
            continue
        value_key = entity_id.split("sensor.")[1]
        points = []
        for entry in raw:
            try:
                points.append({"date": entry["date"], "value": float(entry[value_key])})
            except (KeyError, TypeError, ValueError):
                continue
        if points:
            result.setdefault(room_name, []).extend(points)

    return await make_response(result, 200)


# Get default Config
@app.route("/get-config/defaults", methods=["GET"])
async def config_get():
    """
    Get request action, retrieves and sends default configuration

    """
    app.logger.debug("Obtaining default parameters")
    # Build config, passing only default file
    config = await build_config(emhass_conf, app.logger, str(emhass_conf["defaults_path"]))
    if type(config) is bool and not config:
        return await make_response(["failed to retrieve default config file"], 500)
    # Format parameters in config with params
    params = await build_params(emhass_conf, {}, config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    return_config = param_to_config(params, app.logger)
    # Send params
    return await make_response(return_config, 200)


# Get YAML-to-JSON config
@app.route("/get-json", methods=["POST"])
async def json_convert():
    """
    Post request action, receives yaml config (config_emhass.yaml or EMHASS-Add-on config page) and converts to config json format.

    """
    app.logger.info("Attempting to convert YAML to JSON")
    data = await request.get_data()
    yaml_config = yaml.safe_load(data)

    # If filed to Parse YAML
    if yaml_config is None:
        return await make_response(["failed to Parse YAML from data"], 400)
    # Test YAML is legacy config format (from config_emhass.yaml)
    test_legacy_config = await build_legacy_config_params(emhass_conf, yaml_config, app.logger)
    if test_legacy_config:
        yaml_config = test_legacy_config
    # Format YAML to params (format params. check if params match legacy option.json format)
    params = await build_params(emhass_conf, {}, yaml_config, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)
    # Covert formatted parameters from params back into config.json format
    config = param_to_config(params, app.logger)
    # convert json to str
    config = orjson.dumps(config).decode()

    # Send params
    return await make_response(config, 200)


@app.route("/set-config", methods=["POST"])
async def parameter_set():
    """
    Receive JSON config, and save config to file (config.json and param.pkl)

    """
    config = {}
    if not emhass_conf["defaults_path"]:
        return await make_response(["Unable to Obtain defaults_path from emhass_conf"], 500)
    if not emhass_conf["config_path"]:
        return await make_response(["Unable to Obtain config_path from emhass_conf"], 500)

    # Load defaults as a reference point (for sorting) and a base to override
    if (
        os.path.exists(emhass_conf["defaults_path"])
        and Path(emhass_conf["defaults_path"]).is_file()
    ):
        async with aiofiles.open(str(emhass_conf["defaults_path"])) as data:
            content = await data.read()
            config = orjson.loads(content)
    else:
        app.logger.warning(
            "Unable to obtain default config. only parameters passed from request will be saved to config.json"
        )

    # Retrieve sent config json
    request_data = await request.get_json(force=True)

    # check if data is empty
    if len(request_data) == 0:
        return await make_response(["failed to retrieve config json"], 400)

    # def_load_config is server-only bookkeeping (see build_params's own
    # passthrough, utils.py) - the config-page frontend has no concept of
    # it and never sends it back, so it must be carried forward here from
    # the config.json ABOUT TO BE OVERWRITTEN below, not from request_data.
    # Without this, _strip_auto_appended_loads (utils.py) never has
    # anything to strip on a real browser Save (only defaults_path/
    # request_data feed build_params here, and neither ever has this key),
    # so every Save silently re-appends another copy of each EV/room/
    # boiler-derived load on top of the last.
    if "def_load_config" not in request_data and os.path.exists(emhass_conf["config_path"]):
        async with aiofiles.open(str(emhass_conf["config_path"])) as existing:
            existing_config = orjson.loads(await existing.read())
        if existing_config.get("def_load_config") is not None:
            request_data["def_load_config"] = existing_config["def_load_config"]

    # Format config by converting to params (format params. check if params match legacy option.json format. If so format)
    params = await build_params(emhass_conf, params_secrets, request_data, app.logger)
    if type(params) is bool and not params:
        return await make_response([error_msg_associations_file], 500)

    # Covert formatted parameters from params back into config.json format.
    # Overwrite existing default parameters in config
    config.update(param_to_config(params, app.logger))

    # Save config to config.json
    if os.path.exists(emhass_conf["config_path"].parent):
        async with aiofiles.open(str(emhass_conf["config_path"]), "w") as f:
            await f.write(orjson.dumps(config, option=orjson.OPT_INDENT_2).decode())
    else:
        return await make_response(["Unable to save config file"], 500)

    # Save params with updated config
    if os.path.exists(emhass_conf["data_path"]):
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "wb") as fid:
            content = pickle.dumps(
                (
                    emhass_conf["config_path"],
                    await build_params(emhass_conf, params_secrets, config, app.logger),
                )
            )
            await fid.write(content)
    else:
        return await make_response(["Unable to save params file, missing data_path"], 500)

    app.logger.info("Saved parameters from webserver")
    return await make_response({}, 200)


async def _load_params_and_runtime(request, emhass_conf, logger):
    """
    Loads configuration parameters from pickle and runtime parameters from the request.
    Returns a tuple (params, costfun, runtimeparams) or raises an exception/returns None on failure.
    """
    action_str = " >> Obtaining params: "
    logger.info(action_str)

    # Load params.pkl
    params = None
    costfun = "profit"
    params_path = emhass_conf["data_path"] / params_file

    if params_path.exists():
        async with aiofiles.open(str(params_path), "rb") as fid:
            content = await fid.read()
            try:
                _, params = pickle.loads(content)  # Don't overwrite emhass_conf["config_path"]
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                logger.error(
                    "params.pkl is corrupted or truncated (race condition); cannot proceed"
                )
                return None, None, None
            # Set local costfun variable
            if params.get("optim_conf") is not None:
                costfun = params["optim_conf"].get("costfun", "profit")
            params = orjson.dumps(params).decode()
    else:
        logger.error("Unable to find params.pkl file")
        return None, None, None

    # Load runtime params
    try:
        runtimeparams = await request.get_json(force=True)
        if runtimeparams:
            logger.info("Passed runtime parameters: " + str(runtimeparams))
        else:
            runtimeparams = {}
    except Exception as e:
        logger.error(f"Error parsing runtime params JSON: {e}")
        logger.error("Check your payload for syntax errors (e.g., use 'false' instead of 'False')")
        runtimeparams = {}

    runtimeparams = orjson.dumps(runtimeparams).decode()

    return params, costfun, runtimeparams


async def _handle_action_dispatch(
    action_name, input_data_dict, emhass_conf, params, runtimeparams, logger
):
    """
    Dispatches the specific logic based on the action_name.
    Returns (response_msg, status_code).
    """
    # Actions that don't require input_data_dict or have specific flows
    if action_name == "weather-forecast-cache":
        action_str = " >> Performing weather forecast, try to caching result"
        logger.info(action_str)
        await weather_forecast_cache(emhass_conf, params, runtimeparams, logger)
        return "EMHASS >> Weather Forecast has run and results possibly cached... \n", 200

    if action_name == "export-influxdb-to-csv":
        action_str = " >> Exporting InfluxDB data to CSV..."
        logger.info(action_str)
        success = await export_influxdb_to_csv(None, logger, emhass_conf, params, runtimeparams)
        if success:
            return "EMHASS >> Action export-influxdb-to-csv executed successfully... \n", 200
        return await grab_log(action_str), 400

    # Actions requiring input_data_dict
    if action_name == "publish-data":
        action_str = " >> Publishing data..."
        logger.info(action_str)
        _ = await publish_data(input_data_dict, logger)
        return "EMHASS >> Action publish-data executed... \n", 200

    # Mapping for optimization actions to their functions
    optim_actions = {
        "perfect-optim": perfect_forecast_optim,
        "dayahead-optim": dayahead_forecast_optim,
        "naive-mpc-optim": naive_mpc_optim,
    }

    if action_name in optim_actions:
        action_str = f" >> Performing {action_name}..."
        logger.info(action_str)
        opt_res = await optim_actions[action_name](input_data_dict, logger)
        injection_dict = get_injection_dict(opt_res)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return f"EMHASS >> Action {action_name} executed... \n", 200

    # Delegate Machine Learning actions to helper
    ml_response = await _handle_ml_actions(action_name, input_data_dict, emhass_conf, logger)
    if ml_response:
        return ml_response

    # Fallback for invalid action
    logger.error("ERROR: passed action is not valid")
    return "EMHASS >> ERROR: Passed action is not valid... \n", 400


async def _handle_ml_actions(action_name, input_data_dict, emhass_conf, logger):
    """
    Helper function to handle Machine Learning specific actions.
    Returns (msg, status) if action is handled, otherwise None.
    """
    # forecast-model-fit
    if action_name == "forecast-model-fit":
        action_str = " >> Performing a machine learning forecast model fit..."
        logger.info(action_str)
        df_fit_pred, _, mlf = await forecast_model_fit(input_data_dict, logger)
        injection_dict = get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-fit executed... \n", 200

    # forecast-model-predict
    if action_name == "forecast-model-predict":
        action_str = " >> Performing a machine learning forecast model predict..."
        logger.info(action_str)
        df_pred = await forecast_model_predict(input_data_dict, logger)
        if df_pred is None:
            return await grab_log(action_str), 400

        table1 = df_pred.reset_index().to_html(classes="mystyle", index=False)
        injection_dict = {
            "title": "<h2>Custom machine learning forecast model predict</h2>",
            "subsubtitle0": "<h4>Performed a prediction using a pre-trained model</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-predict executed... \n", 200

    # forecast-model-tune
    if action_name == "forecast-model-tune":
        action_str = " >> Performing a machine learning forecast model tune..."
        logger.info(action_str)
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, logger)
        if df_pred_optim is None or mlf is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_forecast_model_tune(df_pred_optim, mlf)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-model-tune executed... \n", 200

    # forecast-calibration
    if action_name == "forecast-calibration":
        action_str = " >> Performing a load forecast calibration..."
        logger.info(action_str)
        result = await forecast_calibration(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_forecast_calibration(result)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action forecast-calibration executed... \n", 200

    # heating-need-forecast
    if action_name == "heating-need-forecast":
        action_str = " >> Performing a heating-need forecast..."
        logger.info(action_str)
        result = await compute_heating_forecast(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Heating-need forecast</h2>",
            "subsubtitle0": f"<h4>Heating needed by: {result['heating_needed_by']}</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action heating-need-forecast executed... \n", 200

    # heating-model-refit
    if action_name == "heating-model-refit":
        action_str = " >> Performing a heating-model refit..."
        logger.info(action_str)
        result = await refit_heating_model(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Heating-model refit</h2>",
            "subsubtitle0": f"<h4>Deployed: {result['deployed']}</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action heating-model-refit executed... \n", 200

    # hybrid-heatpump-model-refit
    if action_name == "hybrid-heatpump-model-refit":
        action_str = " >> Performing a hybrid heat pump gas/electric model refit..."
        logger.info(action_str)
        result = await refit_hybrid_heatpump_model(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Hybrid heat pump model refit</h2>",
            "subsubtitle0": f"<h4>Deployed: {result['deployed']}</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action hybrid-heatpump-model-refit executed... \n", 200

    # hybrid-heatpump-forecast
    if action_name == "hybrid-heatpump-forecast":
        action_str = " >> Performing a hybrid heat pump gas/electric forecast..."
        logger.info(action_str)
        result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Hybrid heat pump forecast</h2>",
            "subsubtitle0": f"<h4>Mean electric forecast: {result['mean_electric_forecast_w']:.1f} W</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action hybrid-heatpump-forecast executed... \n", 200

    # self-learning-physics-refit
    if action_name == "self-learning-physics-refit":
        action_str = " >> Performing a self-learning-physics model refit..."
        logger.info(action_str)
        result = await refit_self_learning_physics_model(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        # room_temp_test_plot_df holds one DataFrame per room (rendered as
        # its own chart below), not a scalar/short value worth a table row.
        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>"
            for key, value in result.items()
            if key != "room_temp_test_plot_df"
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Self-learning-physics model refit</h2>",
            "subsubtitle0": f"<h4>Deployed: {result['deployed']}</h4>",
            "table1": table1,
        }
        # Train/test/predicted room-temperature chart(s) - the same kind of
        # train/test/pred visual the load forecaster's own fit action shows
        # (get_injection_dict_forecast_model_fit), one per room with a real
        # honest held-out test result this refit.
        for i, (room_name, df_plot) in enumerate(result.get("room_temp_test_plot_df", {}).items()):
            injection_dict[f"subsubtitle{i + 1}"] = (
                f"<h4>{room_name}: honest held-out test - actual vs predicted room temperature</h4>"
            )
            injection_dict[f"figure_{i}"] = get_room_temp_test_plot_html(df_plot, room_name)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action self-learning-physics-refit executed... \n", 200

    # pv-horizon-refit
    if action_name == "pv-horizon-refit":
        action_str = " >> Performing a PV shading/horizon model refit..."
        logger.info(action_str)
        result = await refit_pv_horizon_model(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        _table1_excluded_keys = (
            "pv_horizon_profile",
            "pv_horizon_profile_per_panel",
            "blind_azimuths_combined",
            "blind_azimuths_per_panel",
        )
        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>"
            for key, value in result.items()
            if key not in _table1_excluded_keys
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>PV horizon/shading model refit</h2>",
            "subsubtitle0": f"<h4>{result['n_shaded_instants']} shaded instant(s) over {result['n_observations']} observations</h4>",
            "table1": table1,
            "subsubtitle1": (
                "<h5>Each chart: angle is the compass direction the sun comes from (N up, "
                "clockwise), bar length is how high you'd need to look before the "
                "obstruction clears, color is how much light still gets through below "
                "that. Grey wedges are directions the sun can never test for that panel "
                "(self-shaded by its own tilt, or the sun never reaches there at this "
                "latitude) - not a confirmed-clear reading.</h5>"
            ),
        }
        profile = result["pv_horizon_profile"]
        profile_per_panel = result.get("pv_horizon_profile_per_panel") or {}
        blind_azimuths_combined = result.get("blind_azimuths_combined")
        blind_azimuths_per_panel = result.get("blind_azimuths_per_panel") or {}
        seasons_present = sorted(
            {season for seasons in profile.values() for season in seasons},
            key=SEASON_LABELS.index,
        )
        for i, season in enumerate(seasons_present):
            injection_dict[f"subsubtitle_season_{i}"] = f"<h4>{season.capitalize()}</h4>"
            injection_dict[f"figure_{i}"] = render_horizon_polar_grid(
                profile,
                profile_per_panel,
                season,
                blind_azimuths_per_panel=blind_azimuths_per_panel,
                blind_azimuths_combined=blind_azimuths_combined,
            )
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action pv-horizon-refit executed... \n", 200

    # pv-forecast-test
    if action_name == "pv-forecast-test":
        action_str = " >> Computing a PV forecast preview (no optimization)..."
        logger.info(action_str)
        p_pv_forecast = input_data_dict.get("p_pv_forecast")
        if p_pv_forecast is None:
            return await grab_log(action_str), 400

        table1 = (
            p_pv_forecast.rename("p_pv_forecast_w")
            .reset_index()
            .to_html(classes="mystyle", index=False)
        )
        injection_dict = {
            "title": "<h2>PV forecast preview</h2>",
            "subsubtitle0": "<h4>PV power forecast, without running a full optimization</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action pv-forecast-test executed... \n", 200

    # adjust-pv-forecast-refit
    if action_name == "adjust-pv-forecast-refit":
        action_str = " >> Forcing a refit of the PV forecast adjustment model..."
        logger.info(action_str)
        result = await refit_adjust_pv_forecast_model(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>PV forecast adjustment model refit</h2>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action adjust-pv-forecast-refit executed... \n", 200

    # load-forecast-test
    if action_name == "load-forecast-test":
        action_str = " >> Computing a load forecast preview (P50 and P90, no optimization)..."
        logger.info(action_str)
        p_load_forecast_p50 = input_data_dict.get("p_load_forecast_p50")
        p_load_forecast_p90 = input_data_dict.get("p_load_forecast_p90")
        if p_load_forecast_p50 is None or p_load_forecast_p90 is None:
            return await grab_log(action_str), 400

        table1 = (
            p_load_forecast_p50.rename("p_load_forecast_p50_w")
            .reset_index()
            .to_html(classes="mystyle", index=False)
        )
        table2 = (
            p_load_forecast_p90.rename("p_load_forecast_p90_w")
            .reset_index()
            .to_html(classes="mystyle", index=False)
        )
        injection_dict = {
            "title": "<h2>Load forecast preview</h2>",
            "subsubtitle0": "<h4>Load power forecast at bias=0 (P50) vs. bias=1 (THR-reconciled P90), without running a full optimization</h4>",
            "table1": table1,
            "table2": table2,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action load-forecast-test executed... \n", 200

    # thermal-models-refit/-tune/-forecast (consolidated: run every enabled
    # thermal model in one call, instead of needing a separate button/
    # automation per model - heating-model-refit/hybrid-heatpump-model-
    # refit/self-learning-physics-refit above, and self-learning-physics-
    # forecast below, remain available individually for independent
    # per-model schedules). Full per-model detail (not just a deployed?
    # summary) via the shared get_injection_dict_thermal_models helper, so
    # none of these three lose detail vs. calling the individual actions.
    if action_name == "thermal-models-refit":
        action_str = " >> Performing a refit of every enabled thermal model..."
        logger.info(action_str)
        results = await refit_enabled_thermal_models(input_data_dict, logger)
        if results is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_thermal_models(results, "<h2>Thermal models refit</h2>")
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action thermal-models-refit executed... \n", 200

    if action_name == "thermal-models-tune":
        action_str = " >> Performing a tune of every tunable, enabled thermal model..."
        logger.info(action_str)
        results = await tune_enabled_thermal_models(input_data_dict, logger)
        if results is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_thermal_models(results, "<h2>Thermal models tune</h2>")
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action thermal-models-tune executed... \n", 200

    if action_name == "thermal-models-forecast":
        action_str = " >> Performing a forecast of every enabled thermal model..."
        logger.info(action_str)
        results = await compute_enabled_thermal_forecasts(input_data_dict, logger)
        if results is None:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_thermal_models(results, "<h2>Thermal models forecast</h2>")
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action thermal-models-forecast executed... \n", 200

    # self-learning-physics-forecast
    if action_name == "self-learning-physics-forecast":
        action_str = " >> Performing a self-learning-physics forecast..."
        logger.info(action_str)
        result = await compute_self_learning_physics_forecast(input_data_dict, logger)
        if result is None:
            return await grab_log(action_str), 400

        table1 = "<table class='mystyle'><tbody>" + "".join(
            f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in result.items()
        ) + "</tbody></table>"
        injection_dict = {
            "title": "<h2>Self-learning-physics forecast</h2>",
            "subsubtitle0": f"<h4>Mean electric forecast: {result['mean_electric_forecast_w']:.1f} W</h4>",
            "table1": table1,
        }
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action self-learning-physics-forecast executed... \n", 200

    # regressor-model-fit
    if action_name == "regressor-model-fit":
        action_str = " >> Performing a machine learning regressor fit..."
        logger.info(action_str)
        await regressor_model_fit(input_data_dict, logger)
        return "EMHASS >> Action regressor-model-fit executed... \n", 200

    # regressor-model-predict
    if action_name == "regressor-model-predict":
        action_str = " >> Performing a machine learning regressor predict..."
        logger.info(action_str)
        await regressor_model_predict(input_data_dict, logger)
        return "EMHASS >> Action regressor-model-predict executed... \n", 200

    # thermal-two-stage-plan
    if action_name == "thermal-two-stage-plan":
        action_str = " >> Performing thermal two-stage planning..."
        logger.info(action_str)
        df_plan = await thermal_two_stage_plan(input_data_dict, logger)
        if df_plan is None or df_plan.empty:
            return await grab_log(action_str), 400

        injection_dict = get_injection_dict_thermal_two_stage(df_plan)
        await _save_injection_dict(injection_dict, emhass_conf["data_path"])
        return "EMHASS >> Action thermal-two-stage-plan executed... \n", 201

    return None


async def _save_injection_dict(injection_dict, data_path):
    """Helper to save injection dict to pickle."""
    async with aiofiles.open(str(data_path / injection_dict_file), "wb") as fid:
        content = pickle.dumps(injection_dict)
        await fid.write(content)


@app.route("/api/v1/last-run", methods=["GET"])
async def api_v1_last_run():
    """Return metadata about the most recent optimization run.

    Always-200 envelope with status enum:
      - "no-run": no optim has completed yet (or state lost on restart with no disk file)
      - "ok": last solve succeeded
      - "infeasible": solver returned Infeasible
      - "error": run errored out

    Other fields are populated for status != "no-run"; null otherwise.

    Schema: docs/api/v1/last-run.schema.json (JSON Schema draft 2020-12).
    """
    snap = last_run.read(emhass_conf["data_path"])
    if snap is None:
        response_body = {
            "status": "no-run",
            "timestamp": None,
            "action": None,
            "stage_times": None,
            "duration_total_seconds": None,
            "emhass_version": last_run.emhass_version(),
            "schema_version": EMHASS_SCHEMA_VERSION,
            "infeasible": None,
            "error_message": None,
        }
    else:
        response_body = snap

    response = await make_response(orjson.dumps(response_body))
    response.headers["Content-Type"] = "application/json"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/v1/plan", methods=["GET"])
async def api_v1_plan():
    """Return the latest optimization plan as structured JSON.

    Always-200 envelope with status enum:
      - "no-run": no optim has completed yet (state lost on restart with no disk file)
      - "ok": a plan is available

    Schema: docs/api/v1/plan.schema.json (JSON Schema draft 2020-12).
    """
    snap = plan_store.read(emhass_conf["data_path"])
    if snap is None:
        response_body = {
            "status": "no-run",
            "generated_at": None,
            "emhass_schema_version": EMHASS_SCHEMA_VERSION,
            "plan": None,
        }
    else:
        response_body = {"status": "ok", **snap}

    response = await make_response(orjson.dumps(response_body))
    response.headers["Content-Type"] = "application/json"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/healthz", methods=["GET"])
async def healthz():
    """Liveness/readiness probe for container watchdogs.

    HTTP 200 status:"ok"       -> a run exists (and is within ?max_age_seconds= if given)
    HTTP 503 status:"degraded" -> no run yet, or the run is older than the requested window

    Recency-only: an infeasible/errored last solve is still a healthy server; the raw
    result is reported as last_run_status for diagnostics. Unauthenticated, read-only.
    Schema: docs/api/healthz.schema.json (JSON Schema draft 2020-12).
    """
    raw = request.args.get("max_age_seconds")
    try:
        max_age = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        max_age = None  # graceful-ignore: a 400 would read as "unhealthy" to a watchdog
    snap = last_run.read(emhass_conf["data_path"])
    has_run = snap is not None
    stale = (
        max_age is not None
        and has_run
        and not last_run.is_recent(emhass_conf["data_path"], max_age)
    )
    status, code = _health_verdict(has_run, stale)
    body = {
        "status": status,
        "boot_ts": app.config.get("boot_ts"),
        "last_run_ts": snap.get("timestamp") if snap else None,
        "last_run_status": snap.get("status") if snap else None,
        "versions": {
            "emhass": last_run.emhass_version(),
            "python": platform.python_version(),
            "schema_version": EMHASS_SCHEMA_VERSION,
        },
    }
    response = await make_response(orjson.dumps(body), code)
    response.headers["Content-Type"] = "application/json"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/action/<action_name>", methods=["POST"])
async def action_call(action_name: str):
    """
    Receive Post action, run action according to passed slug(action_name)
    """
    global continual_publish_thread
    global injection_dict

    # Load Parameters
    params, costfun, runtimeparams = await _load_params_and_runtime(
        request, emhass_conf, app.logger
    )
    if params is None:
        return await make_response(await grab_log(" >> Obtaining params: "), 400)

    # Check for actions that do not need input_data_dict
    if action_name in ["weather-forecast-cache", "export-influxdb-to-csv"]:
        msg, status = await _handle_action_dispatch(
            action_name, None, emhass_conf, params, runtimeparams, app.logger
        )
        if status == 400:
            return await make_response(msg, status)

        # Check logs for these specific actions
        action_str = f" >> Performing {action_name}..."
        if not await check_file_log(action_str):
            return await make_response(msg, status)
        return await make_response(await grab_log(action_str), 400)

    # Set Input Data Dict (Common for all other actions)
    # offline_test_mode is only honored when explicitly enabled server-side
    # (EMHASS_ALLOW_OFFLINE_TEST_MODE), so a client can't silently force live
    # actions to run against bundled test data instead of Home Assistant.
    offline_test_mode = False
    if os.environ.get("EMHASS_ALLOW_OFFLINE_TEST_MODE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        try:
            runtime_dict = orjson.loads(runtimeparams) if isinstance(runtimeparams, str) else {}
            if isinstance(runtime_dict, dict):
                offline_test_mode = bool(runtime_dict.get("offline_test_mode", False))
        except Exception:
            offline_test_mode = False

    if offline_test_mode:
        app.logger.info("Offline test mode enabled: using local test data instead of Home Assistant")
        offline_test_file = emhass_conf["data_path"] / "test_df_final.pkl"
        if not offline_test_file.exists():
            fallback_test_file = emhass_conf["root_path"].parent.parent / "data" / "test_df_final.pkl"
            if fallback_test_file.exists():
                shutil.copy2(fallback_test_file, offline_test_file)
                app.logger.info(
                    f"Offline test data copied to data_path: {offline_test_file}"
                )
            else:
                app.logger.warning(
                    "Offline test mode requested but test_df_final.pkl was not found in fallback data path"
                )

    action_str = " >> Setting input data dict"
    app.logger.info(action_str)
    input_data_dict = await set_input_data_dict(
        emhass_conf,
        costfun,
        params,
        runtimeparams,
        action_name,
        app.logger,
        get_data_from_file=offline_test_mode,
    )

    if not input_data_dict:
        return await make_response(await grab_log(action_str), 400)

    # Handle Continual Publish Threading
    rh_handed_to_thread = False
    if len(continual_publish_thread) == 0 and input_data_dict["retrieve_hass_conf"].get(
        "continual_publish", False
    ):
        rh_handed_to_thread = True
        continual_loop = app.add_background_task(
            continual_publish, input_data_dict, entity_path, app.logger
        )
        continual_publish_thread.append(continual_loop)

    # Execute Action
    try:
        msg, status = await _handle_action_dispatch(
            action_name, input_data_dict, emhass_conf, params, runtimeparams, app.logger
        )
    finally:
        # Close HTTP session unless this rh was handed to the continual_publish thread.
        # Each call to set_input_data_dict creates a fresh rh, so closing here only
        # affects this request's rh, not any previously started background thread's.
        if not rh_handed_to_thread and "rh" in input_data_dict:
            await input_data_dict["rh"].close()

    # Final Log Check & Response
    if status == 200:
        if not await check_file_log(" >> "):
            return await make_response(msg, 200)
        return await make_response(await grab_log(" >> "), 400)

    return await make_response(msg, status)


async def _setup_paths() -> tuple[Path, Path, Path, Path, Path, Path]:
    """Helper to set up environment paths and update emhass_conf."""
    # Find env's, not not set defaults
    DATA_PATH = os.getenv("DATA_PATH", default="/data/")
    ROOT_PATH = os.getenv("ROOT_PATH", default=str(Path(__file__).parent))
    CONFIG_PATH = os.getenv("CONFIG_PATH", default="/share/config.json")
    OPTIONS_PATH = os.getenv("OPTIONS_PATH", default="/data/options.json")
    DEFAULTS_PATH = os.getenv("DEFAULTS_PATH", default=ROOT_PATH + "/data/config_defaults.json")
    ASSOCIATIONS_PATH = os.getenv("ASSOCIATIONS_PATH", default=ROOT_PATH + "/data/associations.csv")
    LEGACY_CONFIG_PATH = os.getenv("LEGACY_CONFIG_PATH", default="/app/config_emhass.yaml")
    # Define the paths
    config_path = Path(CONFIG_PATH)
    options_path = Path(OPTIONS_PATH)
    defaults_path = Path(DEFAULTS_PATH)
    associations_path = Path(ASSOCIATIONS_PATH)
    legacy_config_path = Path(LEGACY_CONFIG_PATH)
    data_path = Path(DATA_PATH)
    root_path = Path(ROOT_PATH)
    # Add paths to emhass_conf
    emhass_conf["config_path"] = config_path
    emhass_conf["options_path"] = options_path
    emhass_conf["defaults_path"] = defaults_path
    emhass_conf["associations_path"] = associations_path
    emhass_conf["legacy_config_path"] = legacy_config_path
    emhass_conf["data_path"] = data_path
    emhass_conf["root_path"] = root_path
    return (
        config_path,
        options_path,
        defaults_path,
        associations_path,
        legacy_config_path,
        root_path,
    )


async def _build_configuration(
    config_path: Path, legacy_config_path: Path, defaults_path: Path
) -> tuple[dict, str, str]:
    """Helper to build configuration and local variables."""
    config = {}
    # Combine parameters from configuration sources (if exists)
    built_config = await build_config(
        emhass_conf,
        app.logger,
        str(defaults_path),
        str(config_path) if config_path.exists() else None,
        str(legacy_config_path) if legacy_config_path.exists() else None,
    )
    # Catch the False return BEFORE trying to update the dictionary
    if type(built_config) is bool and not built_config:
        raise Exception("Failed to find default config")
    config.update(built_config)
    # Set local variables
    costfun = os.getenv("LOCAL_COSTFUN", config.get("costfun", "profit"))
    logging_level = os.getenv("LOGGING_LEVEL", config.get("logging_level", "INFO"))
    # Temporary set logging level if debug
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)

    return config, costfun, logging_level


async def _setup_secrets(args: dict | None, options_path: Path) -> str:
    """Helper to parse arguments and build secrets."""
    ## Secrets
    # Argument
    argument = {}
    no_response = False
    if args is not None:
        if args.get("url", None):
            argument["url"] = args["url"]
        if args.get("key", None):
            argument["key"] = args["key"]
        if args.get("no_response", None):
            no_response = args["no_response"]

    # Define secrets_path and save to emhass_conf
    secrets_path = Path(os.getenv("SECRETS_PATH", default="/app/secrets_emhass.yaml"))

    # Store it in the global config so configuration() can use it later
    global emhass_conf
    emhass_conf["secrets_path"] = secrets_path

    # Combine secrets from ENV, Arguments/ARG, Secrets file (secrets_emhass.yaml), options (options.json from addon configuration file) and/or Home Assistant Standalone API (if exist)
    emhass_conf, secrets = await build_secrets(
        emhass_conf,
        app.logger,
        secrets_path=secrets_path,  # Use the variable we defined above
        options_path=str(options_path),
        argument=argument,
        no_response=bool(no_response),
    )
    params_secrets.update(secrets)
    return params_secrets.get("server_ip", "0.0.0.0")


def _validate_data_path(root_path: Path) -> None:
    """Helper to validate and create the data path if necessary."""
    # Check if data path exists
    if not os.path.isdir(emhass_conf["data_path"]):
        app.logger.warning("Unable to find data_path: " + str(emhass_conf["data_path"]))
        if os.path.isdir(Path("/data/")):
            emhass_conf["data_path"] = Path("/data/")
        else:
            Path(root_path / "data/").mkdir(parents=True, exist_ok=True)
            emhass_conf["data_path"] = root_path / "data/"
        app.logger.info("data_path has been set to " + str(emhass_conf["data_path"]))


async def _load_injection_dict() -> dict | None:
    """Helper to load the injection dictionary."""
    # Initialize this global dict
    if (emhass_conf["data_path"] / injection_dict_file).exists():
        async with aiofiles.open(str(emhass_conf["data_path"] / injection_dict_file), "rb") as fid:
            content = await fid.read()
            try:
                return pickle.loads(content)
            except (EOFError, pickle.UnpicklingError, UnicodeDecodeError):
                # File truncated due to write race condition; treat as not yet available
                return None
    else:
        return None


async def _build_and_save_params(
    config: dict, costfun: str, logging_level: str, config_path: Path
) -> dict:
    """Helper to build parameters and save them to a pickle file."""
    # Build params from config and param_secrets (migrate params to correct config catagories), save result to params.pkl
    params = await build_params(emhass_conf, params_secrets, config, app.logger)
    if type(params) is bool:
        raise Exception("A error has occurred while building params")
    # Update params with local variables
    params["optim_conf"]["costfun"] = costfun
    params["optim_conf"]["logging_level"] = logging_level
    # Save params to file for later reference (use emhass_conf["config_path"] which may have been updated by build_secrets)
    if os.path.exists(str(emhass_conf["data_path"])):
        async with aiofiles.open(str(emhass_conf["data_path"] / params_file), "wb") as fid:
            content = pickle.dumps((emhass_conf["config_path"], params))
            await fid.write(content)
    else:
        raise Exception("missing: " + str(emhass_conf["data_path"]))
    return params


async def _configure_logging(logging_level: str) -> None:
    """Helper to configure logging handlers and levels."""
    # Define loggers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log.default_handler.setFormatter(formatter)
    # Action file logger
    file_logger = logging.FileHandler(str(emhass_conf["data_path"] / action_log_str))
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    file_logger.setFormatter(formatter)  # add format to Handler
    if logging_level == "DEBUG":
        app.logger.setLevel(logging.DEBUG)
        file_logger.setLevel(logging.DEBUG)
    elif logging_level == "INFO":
        app.logger.setLevel(logging.INFO)
        file_logger.setLevel(logging.INFO)
    elif logging_level == "WARNING":
        app.logger.setLevel(logging.WARNING)
        file_logger.setLevel(logging.WARNING)
    elif logging_level == "ERROR":
        app.logger.setLevel(logging.ERROR)
        file_logger.setLevel(logging.ERROR)
    else:
        app.logger.setLevel(logging.DEBUG)
        file_logger.setLevel(logging.DEBUG)
    app.logger.propagate = False
    app.logger.addHandler(file_logger)
    # Clear Action File logger file, ready for new instance
    await clear_file_log()


def _cleanup_entities() -> Path:
    """Helper to remove entity/metadata files."""
    # If entity_path exists, remove any entity/metadata files
    ent_path = emhass_conf["data_path"] / "entities"
    if os.path.exists(ent_path):
        entity_path_contents = os.listdir(ent_path)
        if len(entity_path_contents) > 0:
            for entity in entity_path_contents:
                os.remove(ent_path / entity)
    return ent_path


async def _initialize_connections(params: dict) -> None:
    """Helper to initialize WebSocket or InfluxDB connections."""
    # Initialize persistent WebSocket connection only if use_websocket is enabled
    use_websocket = params.get("retrieve_hass_conf", {}).get("use_websocket", False)
    use_influxdb = params.get("retrieve_hass_conf", {}).get("use_influxdb", False)
    # Initialize persistent WebSocket connection if enabled
    if use_websocket:
        app.logger.info("WebSocket mode enabled - initializing connection...")
        try:
            await get_websocket_client(
                hass_url=params_secrets["hass_url"],
                token=params_secrets["long_lived_token"],
                logger=app.logger,
            )
            app.logger.info("WebSocket connection established")
            # WebSocket shutdown is already handled by @app.after_serving
        except Exception as ws_error:
            app.logger.warning(f"WebSocket connection failed: {ws_error}")
            app.logger.info("Continuing without WebSocket connection...")
            # Re-raise the exception so before_serving can handle it
            raise
    # Log InfluxDB mode if enabled (No persistent connection init required here)
    elif use_influxdb:
        app.logger.info("InfluxDB mode enabled - using InfluxDB for data retrieval")
    # Default to REST API if neither is enabled
    else:
        app.logger.info("WebSocket and InfluxDB modes disabled - using REST API for data retrieval")


async def initialize(args: dict | None = None):
    global emhass_conf, params_secrets, continual_publish_thread, injection_dict, entity_path
    # Grab the logging level early from ENV so initialization functions can log properly
    early_log_level = os.getenv("LOGGING_LEVEL", "INFO")
    normalized_log_level = early_log_level.upper()
    log_level = getattr(logging, normalized_log_level, logging.INFO)
    app.logger.setLevel(log_level)
    # Setup paths
    (
        config_path,
        options_path,
        defaults_path,
        _,
        legacy_config_path,
        root_path,
    ) = await _setup_paths()
    # Setup Secrets (must run BEFORE build_configuration to allow options.json to override config_path)
    server_ip = await _setup_secrets(args, options_path)
    # Build configuration (now uses potentially updated emhass_conf["config_path"] from options.json)
    config, costfun, logging_level = await _build_configuration(
        emhass_conf["config_path"],
        emhass_conf.get("legacy_config_path", legacy_config_path),
        defaults_path,
    )
    # Validate Data Path
    _validate_data_path(root_path)
    # Load Injection Dict
    injection_dict = await _load_injection_dict()
    # Build and Save Params
    params = await _build_and_save_params(config, costfun, logging_level, config_path)
    # Configure Logging
    await _configure_logging(logging_level)
    # Cleanup Entities
    entity_path = _cleanup_entities()
    # Initialize Continual Publish Thread
    # Initialise continual publish thread list
    continual_publish_thread = []
    # Log Startup Info
    # Logging
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Launching the emhass webserver at: http://" + server_ip + ":" + str(port))
    app.logger.info(
        "Home Assistant data fetch will be performed using url: " + params_secrets["hass_url"]
    )
    app.logger.info("The data path is: " + str(emhass_conf["data_path"]))
    app.logger.info("The config path is: " + str(emhass_conf["config_path"]))
    app.logger.info("The logging is: " + str(logging_level))
    try:
        app.logger.info("Using core emhass version: " + version("emhass"))
    except PackageNotFoundError:
        app.logger.info("Using development emhass version")
    # Initialize Connections (WebSocket/InfluxDB)
    await _initialize_connections(params)
    app.logger.info("Initialization complete")


async def main() -> None:
    """
    Main function to handle command line arguments.

    Note: In production, the app should be run via gunicorn with uvicorn workers:
    gunicorn emhass.web_server:app -c gunicorn.conf.py -k uvicorn.workers.UvicornWorker
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="HA URL")
    parser.add_argument("--key", type=str, help="HA long‑lived token")
    parser.add_argument("--no_response", action="store_true")
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    # Initialize the app before starting server
    await initialize(args_dict)
    # For direct execution (development/testing), use uvicorn programmatically
    host = params_secrets.get("server_ip", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    app.logger.info(f"Starting server directly on {host}:{port}")
    # Use uvicorn.Server to run within existing event loop
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
