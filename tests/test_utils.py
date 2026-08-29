#!/usr/bin/env python

import json
import logging
import pathlib
import unittest
from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import orjson
import pandas as pd
import pytz

from emhass import utils
from emhass.utils import treat_runtimeparams

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["options_path"] = root / "options.json"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["secrets_path"] = root / "secrets_emhass(example).yaml"
emhass_conf["legacy_config_path"] = (
    pathlib.Path(utils.get_root(__file__, num_parent=1)) / "config_emhass.yaml"
)
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# Create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestUtils(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
        print(emhass_conf["legacy_config_path"])
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            # Add Altitude secret manually for testing get_yaml_parse
            secrets["Altitude"] = 8000.0
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )

        return params

    async def asyncSetUp(self):
        params = await TestUtils.get_test_params()
        # Add runtime parameters for forecast lists
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        self.runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        self.params_json = orjson.dumps(params).decode("utf-8")
        # Create dummy data resembling optimization output
        generator = np.random.default_rng(42)
        dates = pd.date_range(start="2024-01-01", periods=24, freq="1h")
        self.df = pd.DataFrame(index=dates)
        self.df["P_PV"] = generator.standard_normal(24) * 1000
        self.df["P_Load"] = generator.standard_normal(24) * 500
        self.df["optim_status"] = "Optimal"
        self.df["cost_fun_profit"] = 0.5

    async def test_build_config(self):
        # Test building with the different config methods
        config = {}
        params = {}
        # Test with defaults
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(
            config["load_peak_hour_periods"],
            {
                "period_hp_1": [{"start": "02:54"}, {"end": "15:24"}],
                "period_hp_2": [{"start": "17:24"}, {"end": "20:24"}],
            },
        )
        self.assertEqual(
            params["retrieve_hass_conf"]["sensor_replace_zero"],
            ["sensor.power_photovoltaics", "sensor.p_pv_forecast"],
        )
        # Test with config.json
        config = await utils.build_config(
            emhass_conf,
            logger,
            emhass_conf["defaults_path"],
            emhass_conf["config_path"],
        )
        params = await utils.build_params(emhass_conf, {}, config, logger)

    async def test_build_params_heatpump_model_family_default_and_explicit_selection(self):
        """A fresh install defaults to "simple" (today's thermal-loss-only
        behavior, unchanged). An explicit "physics" selection - including one
        already stored from before physics fields were wired to live dispatch
        - must be honored as-is, not silently rewritten: there is no way to
        tell an old dead selection apart from a deliberate new one, so
        build_params must never touch this value itself."""
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(params["optim_conf"]["heatpump_model_family"], "simple")

        config["heatpump_model_family"] = "physics"
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(params["optim_conf"]["heatpump_model_family"], "physics")

        # Test with legacy config_emhass yaml
        config = await utils.build_config(
            emhass_conf,
            logger,
            emhass_conf["defaults_path"],
            legacy_config_path=emhass_conf["legacy_config_path"],
        )
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(
            params["retrieve_hass_conf"]["sensor_replace_zero"], ["sensor.power_photovoltaics"]
        )
        self.assertEqual(
            config["load_peak_hour_periods"],
            {
                "period_hp_1": [{"start": "02:54"}, {"end": "15:24"}],
                "period_hp_2": [{"start": "17:24"}, {"end": "20:24"}],
            },
        )
        self.assertEqual(params["plant_conf"]["battery_charge_efficiency"], 0.95)

    async def test_build_params_def_load_config_survives_config_round_trip_without_duplicating(self):
        """Full regression test for the real bug this session fixed: the
        config page's own GET /get-config -> edit -> POST /set-config flow
        (web_server.py's parameter_get/parameter_set) runs build_params ->
        param_to_config -> (save) -> build_params again on the SAME
        already-derived config. Before this fix, def_load_config never
        survived that round trip (no associations.csv row for it), so
        _strip_auto_appended_loads always saw an empty list and every
        round trip silently appended another copy of each EV/room/boiler
        load on top of the last. Simulates 4 repeated Save clicks in a
        row and asserts the deferrable-load count never grows past the
        first, correctly-derived value."""
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        config["set_use_ev_charger"] = True
        config["number_of_ev_chargers"] = 1
        config["ev_charger_names"] = ["Zappi"]
        config["set_use_boiler"] = True
        config["number_of_boilers"] = 1
        config["boiler_names"] = ["dhw_tank"]

        counts = []
        for _ in range(4):
            params = await utils.build_params(emhass_conf, {}, dict(config), logger)
            counts.append(params["optim_conf"]["number_of_deferrable_loads"])
            for key in utils._AUTO_LOAD_ARRAY_KEYS:
                self.assertEqual(
                    len(params["optim_conf"][key]),
                    counts[-1],
                    f"{key} length diverged from number_of_deferrable_loads",
                )
            config = utils.param_to_config(params, logger)

        self.assertEqual(
            len(set(counts)), 1, f"number_of_deferrable_loads grew across save round trips: {counts}"
        )
        # config_defaults.json's own baseline manual load ("load_1") plus
        # the 1 EV charger and 1 boiler configured above, present exactly
        # once each - not duplicated across any of the 4 rounds.
        self.assertEqual(counts[0], 3)

    async def test_get_yaml_parse(self):
        # Test get_yaml_parse with only secrets
        params = {}
        updated_emhass_conf, secrets = await utils.build_secrets(emhass_conf, logger)
        emhass_conf.update(updated_emhass_conf)
        params.update(await utils.build_params(emhass_conf, secrets, {}, logger))
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            orjson.dumps(params).decode("utf-8"), logger
        )
        self.assertIsInstance(retrieve_hass_conf, dict)
        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(plant_conf, dict)
        self.assertEqual(retrieve_hass_conf["Altitude"], 4807.8)
        # Test get_yaml_parse with built params in get_test_params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        self.assertEqual(retrieve_hass_conf["Altitude"], 8000.0)

    @patch("emhass.utils._get_now")
    def test_get_forecast_dates_standard_day(self, mock_ts_now):
        """
        Tests the forecast date generation on a standard 24-hour day.
        """
        # 1. Define parameters for this specific test
        time_zone = pytz.timezone("Australia/Sydney")
        freq = 60  # in minutes
        delta_forecast = 1  # in days

        # 2. Define the mock 'now' and the expected results
        mock_now = datetime(2025, 10, 11, 7, 0, 0)
        expected_start = "2025-10-11T07:00:00"
        expected_end = "2025-10-12T06:00:00"
        expected_range = pd.date_range(
            start=expected_start, end=expected_end, freq=f"{freq}min", tz=time_zone
        )
        expected_dates = [ts.isoformat() for ts in expected_range]

        # 3. Set the return value for the mock - tz-aware UTC instant equal to the local time
        mock_ts_now.return_value = time_zone.localize(mock_now).astimezone(UTC)

        actual_dates = utils.get_forecast_dates(freq, delta_forecast, time_zone)

        # 4. Perform assertions
        self.assertIsInstance(actual_dates, list)
        self.assertEqual(len(actual_dates), 24)
        self.assertListEqual(actual_dates, expected_dates)

    @patch("emhass.utils._get_now")
    def test_get_forecast_dates_dst_crossing(self, mock_ts_now):
        """
        Tests the forecast date generation on a day with a DST transition (23 hours).
        """
        # 1. Define parameters for this specific test
        time_zone = pytz.timezone("Australia/Sydney")
        freq = 60  # in minutes
        delta_forecast = 1  # in days

        # 2. Define mock 'now' and expected results
        mock_now = datetime(2025, 10, 4, 23, 0, 0)
        expected_start = "2025-10-04T23:00:00"
        expected_end = "2025-10-05T22:00:00"
        expected_range = pd.date_range(
            start=expected_start, end=expected_end, freq=f"{freq}min", tz=time_zone
        )
        expected_dates = [ts.isoformat() for ts in expected_range]

        # 3. Set the return value for the mock - tz-aware UTC instant equal to the local time
        mock_ts_now.return_value = time_zone.localize(mock_now).astimezone(UTC)

        actual_dates = utils.get_forecast_dates(freq, delta_forecast, time_zone)
        # 4. Perform assertions
        self.assertIsInstance(actual_dates, list)
        self.assertEqual(len(actual_dates), 23)  # This day correctly has 23 hours
        self.assertListEqual(actual_dates, expected_dates)
        self.assertIn("+10:00", actual_dates[2])
        self.assertIn("+11:00", actual_dates[3])

    def test_get_forecast_dates_host_tz_differs_from_config(self):
        """Issue #984: forecast dates must be correct when host clock is UTC but EMHASS tz differs.

        Simulates a UTC host by patching emhass.utils.datetime so both call forms
        (datetime.now() and datetime.now(tz)) behave as on a UTC host. The fix converts
        the aware UTC instant to the config tz (tz_convert) instead of localizing a naive
        wall clock (which mislabels UTC wall-clock time as local time).
        """

        class _FakeDT(datetime):
            """Subclass of datetime so isinstance checks in utils still work."""

            @classmethod
            def now(cls, tz=None):
                base = datetime(2026, 6, 15, 11, 17, 0)  # UTC wall clock
                if tz is None:
                    # Base-code path: old _get_now() calls datetime.now() with no tz.
                    # Kept so this same test runs (and fails) against the unfixed code.
                    return base
                return base.replace(tzinfo=UTC).astimezone(tz)

        brisbane = pytz.timezone("Australia/Brisbane")

        # --- host != config tz case (UTC host, Brisbane config) ---
        with patch("emhass.utils.datetime", _FakeDT):
            dates = utils.get_forecast_dates(5, 2, brisbane)
        # Correct local now: 11:17 UTC -> 21:17 AEST (+10); floored to 5min -> 21:15
        self.assertEqual(dates[0], "2026-06-15T21:15:00+10:00")

        # --- control: host clock == config tz (Brisbane host, Brisbane config) ---
        class _FakeDTLocal(datetime):
            """Simulates a Brisbane host: naive now() is 21:17 local; aware now() is consistent."""

            @classmethod
            def now(cls, tz=None):
                local_naive = datetime(2026, 6, 15, 21, 17, 0)
                if tz is None:
                    # Base-code path (datetime.now() with no tz); kept for base-safety.
                    return local_naive
                # Express the same instant tz-aware in the requested tz
                return brisbane.localize(local_naive).astimezone(tz)

        with patch("emhass.utils.datetime", _FakeDTLocal):
            dates_local = utils.get_forecast_dates(5, 2, brisbane)
        # Same correct local start expected
        self.assertEqual(dates_local[0], "2026-06-15T21:15:00+10:00")

    async def test_treat_runtimeparams(self):
        # Test dayahead runtime params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertIsInstance(params["passed_data"]["pv_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_cost_forecast"], list)
        self.assertIsInstance(params["passed_data"]["prod_price_forecast"], list)
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertEqual(optim_conf["load_forecast_method"], "list")
        self.assertEqual(optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(optim_conf["production_price_forecast_method"], "list")
        # Test naive MPC runtime params
        set_type = "naive-mpc-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertEqual(params["passed_data"]["prediction_horizon"], 10)
        self.assertEqual(
            params["passed_data"]["soc_init"], plant_conf["battery_target_state_of_charge"]
        )
        self.assertEqual(
            params["passed_data"]["soc_final"], plant_conf["battery_target_state_of_charge"]
        )
        self.assertEqual(
            params["optim_conf"]["operating_hours_of_each_deferrable_load"],
            optim_conf["operating_hours_of_each_deferrable_load"],
        )
        # Test passing optimization and plant configuration parameters at runtime
        runtimeparams = orjson.loads(self.runtimeparams_json)
        runtimeparams.update({"number_of_deferrable_loads": 3})
        runtimeparams.update({"nominal_power_of_deferrable_loads": [3000.0, 750.0, 2500.0]})
        runtimeparams.update({"operating_hours_of_each_deferrable_load": [5, 8, 10]})
        runtimeparams.update({"treat_deferrable_load_as_semi_cont": [True, True, True]})
        runtimeparams.update({"set_deferrable_load_single_constant": [False, False, False]})
        runtimeparams.update({"weight_battery_discharge": 2.0})
        runtimeparams.update({"weight_battery_charge": 2.0})
        runtimeparams.update({"solcast_api_key": "yoursecretsolcastapikey"})
        runtimeparams.update({"solcast_rooftop_id": "yourrooftopid"})
        runtimeparams.update({"solar_forecast_kwp": 5.0})
        runtimeparams.update({"battery_target_state_of_charge": 0.4})
        runtimeparams.update({"publish_prefix": "emhass_"})
        runtimeparams.update({"custom_pv_forecast_id": "my_custom_pv_forecast_id"})
        runtimeparams.update({"custom_load_forecast_id": "my_custom_load_forecast_id"})
        runtimeparams.update({"custom_batt_forecast_id": "my_custom_batt_forecast_id"})
        runtimeparams.update({"custom_batt_soc_forecast_id": "my_custom_batt_soc_forecast_id"})
        runtimeparams.update({"custom_grid_forecast_id": "my_custom_grid_forecast_id"})
        runtimeparams.update({"custom_cost_fun_id": "my_custom_cost_fun_id"})
        runtimeparams.update({"custom_optim_status_id": "my_custom_optim_status_id"})
        runtimeparams.update({"custom_unit_load_cost_id": "my_custom_unit_load_cost_id"})
        runtimeparams.update({"custom_unit_prod_price_id": "my_custom_unit_prod_price_id"})
        runtimeparams.update({"custom_deferrable_forecast_id": "my_custom_deferrable_forecast_id"})
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertIsInstance(params["passed_data"]["pv_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_cost_forecast"], list)
        self.assertIsInstance(params["passed_data"]["prod_price_forecast"], list)
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 3)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [3000.0, 750.0, 2500.0])
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"], [5, 8, 10])
        self.assertEqual(optim_conf["treat_deferrable_load_as_semi_cont"], [True, True, True])
        self.assertEqual(optim_conf["set_deferrable_load_single_constant"], [False, False, False])
        self.assertEqual(optim_conf["weight_battery_discharge"], 2.0)
        self.assertEqual(optim_conf["weight_battery_charge"], 2.0)
        self.assertEqual(retrieve_hass_conf["solcast_api_key"], "yoursecretsolcastapikey")
        self.assertEqual(retrieve_hass_conf["solcast_rooftop_id"], "yourrooftopid")
        self.assertEqual(retrieve_hass_conf["solar_forecast_kwp"], 5.0)
        self.assertEqual(plant_conf["battery_target_state_of_charge"], 0.4)
        self.assertEqual(params["passed_data"]["publish_prefix"], "emhass_")
        self.assertEqual(params["passed_data"]["custom_pv_forecast_id"], "my_custom_pv_forecast_id")
        self.assertEqual(
            params["passed_data"]["custom_load_forecast_id"], "my_custom_load_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_batt_forecast_id"], "my_custom_batt_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_batt_soc_forecast_id"], "my_custom_batt_soc_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_grid_forecast_id"], "my_custom_grid_forecast_id"
        )
        self.assertEqual(params["passed_data"]["custom_cost_fun_id"], "my_custom_cost_fun_id")
        self.assertEqual(
            params["passed_data"]["custom_optim_status_id"], "my_custom_optim_status_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_unit_load_cost_id"], "my_custom_unit_load_cost_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_unit_prod_price_id"], "my_custom_unit_prod_price_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_deferrable_forecast_id"],
            "my_custom_deferrable_forecast_id",
        )

    async def test_treat_runtimeparams_forecast_calibration_day_windows(self):
        """The forecast-calibration day windows are runtime-overridable passed_data
        keys: valid values land as ints, a non-integer is dropped (not fatal), and
        omitting them leaves them absent so the action uses its defaults."""
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        runtimeparams = orjson.loads(self.runtimeparams_json)
        runtimeparams.update(
            {
                "calibration_days_to_retrieve": "120",  # string int -> cast
                "calibration_test_days": 21,
                "calibration_val_days": "not-a-number",  # invalid -> ignored
            }
        )
        (params, _, _, _) = await utils.treat_runtimeparams(
            runtimeparams,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "forecast-calibration",
            logger,
            emhass_conf,
        )
        passed = orjson.loads(params)["passed_data"]
        self.assertEqual(passed["calibration_days_to_retrieve"], 120)
        self.assertIsInstance(passed["calibration_days_to_retrieve"], int)
        self.assertEqual(passed["calibration_test_days"], 21)
        # invalid value is skipped, never raised, and the config GUI is untouched
        self.assertNotIn("calibration_val_days", passed)

    async def test_treat_runtimeparams_forecast_calibration_defaults_absent(self):
        """With no calibration keys passed, none appear in passed_data, so the
        action falls back to its 90 / 14 / 14 defaults (true no-op)."""
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        (params, _, _, _) = await utils.treat_runtimeparams(
            orjson.loads(self.runtimeparams_json),
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "forecast-calibration",
            logger,
            emhass_conf,
        )
        passed = orjson.loads(params)["passed_data"]
        for key in (
            "calibration_days_to_retrieve",
            "calibration_test_days",
            "calibration_val_days",
        ):
            self.assertNotIn(key, passed)

    @patch("emhass.utils._get_now")
    async def test_treat_runtimeparams_dict_forecast_holds_last_value(self, mock_now):
        """Regression for issue #1003.

        A dict forecast (load_cost_forecast / prod_price_forecast) must hold each
        value until the next provided point (step semantics). A point defined
        BEFORE the forecast horizon start must anchor the leading slots; it must
        not be discarded so that the trailing back-fill paints those slots with
        the NEXT value. Before the fix, reindex(method="nearest") dropped the
        pre-horizon anchor and the bfill filled the leading slots with the next
        value, so the load price before the first in-horizon point read 0 and the
        production price before its first point read the later peak value.
        """
        params = await TestUtils.get_test_params()
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            orjson.dumps(params).decode("utf-8"), logger
        )
        time_zone = retrieve_hass_conf["time_zone"]

        # Pin "now" to 09:00 local so the 30 min forecast grid starts there.
        mock_local = pd.Timestamp("2026-06-26T09:00:00", tz=time_zone)
        mock_now.return_value = mock_local.tz_convert(UTC)

        def iso(hour, minute=0):
            return pd.Timestamp(f"2026-06-26T{hour:02d}:{minute:02d}:00", tz=time_zone).isoformat()

        horizon = 12  # 12 x 30 min = 09:00 .. 14:30
        runtimeparams = {
            "prediction_horizon": horizon,
            # First point (08:00) is one hour before the horizon start (09:00).
            "load_cost_forecast": {iso(8): 0.308, iso(11): 0.0, iso(14): 0.308},
            "prod_price_forecast": {iso(8): 0.0, iso(13): 0.1, iso(21): 0.0},
        }
        set_type = "naive-mpc-optim"
        params_out, _, _, _ = await utils.treat_runtimeparams(
            runtimeparams,
            orjson.dumps(params).decode("utf-8"),
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        passed = orjson.loads(params_out)["passed_data"]
        load_cost = passed["load_cost_forecast"]
        prod_price = passed["prod_price_forecast"]

        self.assertEqual(len(load_cost), horizon)
        self.assertEqual(len(prod_price), horizon)

        # Leading slots hold the pre-horizon anchor (the bug back-filled the next
        # value here: load_cost[0] -> 0.0 and prod_price[0] -> 0.1).
        self.assertAlmostEqual(load_cost[0], 0.308)
        self.assertAlmostEqual(prod_price[0], 0.0)

        # Full step profile across the horizon.
        # load_cost: 0.308 until 11:00 (idx 4), 0.0 until 14:00 (idx 10), then 0.308.
        for idx in range(0, 4):
            self.assertAlmostEqual(load_cost[idx], 0.308)
        for idx in range(4, 10):
            self.assertAlmostEqual(load_cost[idx], 0.0)
        for idx in range(10, 12):
            self.assertAlmostEqual(load_cost[idx], 0.308)
        # prod_price: 0.0 until 13:00 (idx 8), then 0.1 to the end of the horizon.
        for idx in range(0, 8):
            self.assertAlmostEqual(prod_price[idx], 0.0)
        for idx in range(8, 12):
            self.assertAlmostEqual(prod_price[idx], 0.1)

    async def test_treat_runtimeparams_failed(self):
        # Test treatment of nan values
        params = await TestUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, "nan", 7, 8, 9, 10],
            "load_power_forecast": [1, 2, "nan", 4, 5, 6, 7, 8, 9, 10],
            "load_cost_forecast": [1, 2, 3, 4, 5, 6, 7, 8, "nan", 10],
            "prod_price_forecast": [1, 2, 3, 4, "nan", 6, 7, 8, 9, 10],
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
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

        self.assertGreater(
            len([x for x in runtimeparams["pv_power_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["load_power_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["load_cost_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["prod_price_forecast"] if not str(x).isdigit()]), 0
        )
        # Test list embedded into a string
        params = await TestUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "load_power_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "load_cost_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "prod_price_forecast": "[1,2,3,4,5,6,7,8,9,10]",
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
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
        self.assertIsInstance(runtimeparams["pv_power_forecast"], list)
        self.assertIsInstance(runtimeparams["load_power_forecast"], list)
        self.assertIsInstance(runtimeparams["load_cost_forecast"], list)
        self.assertIsInstance(runtimeparams["prod_price_forecast"], list)
        # Test string of numbers
        params = await TestUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": "1,2,3,4,5,6,7,8,9,10",
            "load_power_forecast": "1,2,3,4,5,6,7,8,9,10",
            "load_cost_forecast": "1,2,3,4,5,6,7,8,9,10",
            "prod_price_forecast": "1,2,3,4,5,6,7,8,9,10",
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
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
        self.assertIsInstance(runtimeparams["pv_power_forecast"], str)
        self.assertIsInstance(runtimeparams["load_power_forecast"], str)
        self.assertIsInstance(runtimeparams["load_cost_forecast"], str)
        self.assertIsInstance(runtimeparams["prod_price_forecast"], str)

    async def test_update_params_with_ha_config(self):
        # Test dayahead runtime params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        ha_config = {"currency": "USD", "unit_system": {"temperature": "°F"}}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )

    async def test_update_params_with_ha_config_special_case(self):
        # Test special passed runtime params
        runtimeparams = {
            "prediction_horizon": 28,
            "pv_power_forecast": [
                523,
                873,
                1059,
                1195,
                1291,
                1352,
                1366,
                1327,
                1254,
                1150,
                1004,
                813,
                589,
                372,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                153,
                228,
                301,
                363,
                407,
                438,
                456,
                458,
                443,
                417,
                381,
                332,
                269,
                195,
                123,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "num_def_loads": 2,
            "P_deferrable_nom": [0, 0],
            "def_total_hours": [0, 0],
            "treat_def_as_semi_cont": [1, 1],
            "set_def_constant": [1, 1],
            "def_start_timestep": [0, 0],
            "def_end_timestep": [0, 0],
            "soc_init": 0.64,
            "soc_final": 0.9,
            "load_cost_forecast": [
                0.2751,
                0.2751,
                0.2729,
                0.2729,
                0.2748,
                0.2748,
                0.2746,
                0.2746,
                0.2815,
                0.2815,
                0.2841,
                0.2841,
                0.282,
                0.282,
                0.288,
                0.288,
                0.29,
                0.29,
                0.2841,
                0.2841,
                0.2747,
                0.2747,
                0.2677,
                0.2677,
                0.2628,
                0.2628,
                0.2532,
                0.2532,
            ],
            "prod_price_forecast": [
                0.1213,
                0.1213,
                0.1192,
                0.1192,
                0.121,
                0.121,
                0.1208,
                0.1208,
                0.1274,
                0.1274,
                0.1298,
                0.1298,
                0.1278,
                0.1278,
                0.1335,
                0.1335,
                0.1353,
                0.1353,
                0.1298,
                0.1298,
                0.1209,
                0.1209,
                0.1143,
                0.1143,
                0.1097,
                0.1097,
                0.1007,
                0.1007,
            ],
            "alpha": 1,
            "beta": 0,
            "load_power_forecast": [
                399,
                300,
                400,
                600,
                300,
                200,
                200,
                200,
                200,
                300,
                300,
                200,
                400,
                200,
                200,
                400,
                400,
                400,
                300,
                300,
                300,
                600,
                800,
                500,
                400,
                400,
                500,
                500,
                2400,
                2300,
                2400,
                2400,
                2300,
                2400,
                2400,
                2400,
                2300,
                2400,
                2400,
                200,
                200,
                300,
                300,
                300,
                300,
                300,
                300,
                300,
            ],
        }
        params_ = orjson.loads(self.params_json)
        params_["passed_data"].update(runtimeparams)

        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params_json = orjson.dumps(params_).decode()

        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        ha_config = {"currency": "USD"}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )
        # Test with 0 deferrable loads
        runtimeparams = {
            "prediction_horizon": 28,
            "pv_power_forecast": [
                523,
                873,
                1059,
                1195,
                1291,
                1352,
                1366,
                1327,
                1254,
                1150,
                1004,
                813,
                589,
                372,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                153,
                228,
                301,
                363,
                407,
                438,
                456,
                458,
                443,
                417,
                381,
                332,
                269,
                195,
                123,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "num_def_loads": 0,
            "def_start_timestep": [0, 0],
            "def_end_timestep": [0, 0],
            "soc_init": 0.64,
            "soc_final": 0.9,
            "load_cost_forecast": [
                0.2751,
                0.2751,
                0.2729,
                0.2729,
                0.2748,
                0.2748,
                0.2746,
                0.2746,
                0.2815,
                0.2815,
                0.2841,
                0.2841,
                0.282,
                0.282,
                0.288,
                0.288,
                0.29,
                0.29,
                0.2841,
                0.2841,
                0.2747,
                0.2747,
                0.2677,
                0.2677,
                0.2628,
                0.2628,
                0.2532,
                0.2532,
            ],
            "prod_price_forecast": [
                0.1213,
                0.1213,
                0.1192,
                0.1192,
                0.121,
                0.121,
                0.1208,
                0.1208,
                0.1274,
                0.1274,
                0.1298,
                0.1298,
                0.1278,
                0.1278,
                0.1335,
                0.1335,
                0.1353,
                0.1353,
                0.1298,
                0.1298,
                0.1209,
                0.1209,
                0.1143,
                0.1143,
                0.1097,
                0.1097,
                0.1007,
                0.1007,
            ],
            "alpha": 1,
            "beta": 0,
            "load_power_forecast": [
                399,
                300,
                400,
                600,
                300,
                200,
                200,
                200,
                200,
                300,
                300,
                200,
                400,
                200,
                200,
                400,
                400,
                400,
                300,
                300,
                300,
                600,
                800,
                500,
                400,
                400,
                500,
                500,
                2400,
                2300,
                2400,
                2400,
                2300,
                2400,
                2400,
                2400,
                2300,
                2400,
                2400,
                200,
                200,
                300,
                300,
                300,
                300,
                300,
                300,
                300,
            ],
        }
        params_ = orjson.loads(self.params_json)
        params_["passed_data"].update(runtimeparams)
        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params_json = orjson.dumps(params_).decode()
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        ha_config = {"currency": "USD"}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )

    async def test_build_secrets(self):
        # Test the build_secrets defaults from get_test_params()
        params = await TestUtils.get_test_params()
        expected_keys = [
            "retrieve_hass_conf",
            "params_secrets",
            "optim_conf",
            "plant_conf",
            "passed_data",
        ]
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from options.json
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf,
            logger,
            options_path=emhass_conf["options_path"],
            secrets_path="",
            no_response=True,
        )
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from secrets_emhass(example).yaml
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf, logger, secrets_path=emhass_conf["secrets_path"]
        )
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from arguments (command_line cli)
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf, logger, {"url": "test.url", "key": "test.key"}, secrets_path=""
        )
        logger.debug("Obtaining long_lived_token from passed argument")
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "test.url")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "test.key")

    def test_get_injection_dict_with_thermal(self):
        # Add thermal columns to dummy df
        self.df["predicted_temp_heater1"] = 21.0
        self.df["target_temp_heater1"] = 22.0
        # Run function
        injection_dict = utils.get_injection_dict(self.df.copy())
        # Verify Keys
        self.assertIn("figure_0", injection_dict, "Powers plot missing")
        self.assertIn("figure_thermal", injection_dict, "Thermal plot missing")
        self.assertIn("figure_2", injection_dict, "Cost plot missing")
        # Verify Content
        self.assertIn("Thermal loads temperature schedule", injection_dict["figure_thermal"])
        self.assertIn("Temperature (&deg;C)", injection_dict["figure_thermal"])

    def test_get_injection_dict_without_thermal(self):
        # Ensure no thermal columns
        cols = [c for c in self.df.columns if "heater" not in c]
        df_clean = self.df[cols].copy()
        # Run function
        injection_dict = utils.get_injection_dict(df_clean)
        # Verify Thermal is NOT present
        self.assertNotIn("figure_thermal", injection_dict)
        self.assertIn("figure_0", injection_dict)

    def test_get_room_temp_test_plot_html(self):
        """Mirrors get_injection_dict_forecast_model_fit's own train/test/
        pred chart shape (see machine_learning_forecaster.py's df_pred) -
        a DataFrame with columns exactly train/test/pred, some NaN
        (outside each column's own segment), turned into an embeddable
        Plotly HTML fragment."""
        idx = pd.date_range("2026-01-01", periods=6, freq="30min", tz="UTC")
        df_plot = pd.DataFrame(
            {
                "train": [20.0, 20.5, 21.0, np.nan, np.nan, np.nan],
                "test": [np.nan, np.nan, np.nan, 21.5, 22.0, 22.5],
                "pred": [np.nan, np.nan, np.nan, 21.3, 21.9, 22.4],
            },
            index=idx,
        )
        html = utils.get_room_temp_test_plot_html(df_plot, "Woonkamer")
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)
        self.assertIn("Woonkamer", html)
        # A real Plotly HTML fragment, not just an arbitrary string.
        self.assertIn("plotly", html.lower())

    def test_get_forecast_trend_plot_html(self):
        """Generalized sibling of get_room_temp_test_plot_html for the
        predict-side forecast charts (compute_heating_forecast/
        compute_hybrid_heatpump_forecast/compute_self_learning_physics_forecast) -
        no fixed train/test/pred column triple, just whatever forecast
        column(s) are passed (a single forecasted series, or a forecast
        alongside a flat reference line)."""
        idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
        # Single-column case (e.g. a room temperature forecast series).
        df_single = pd.DataFrame({"forecast": [20.0, 20.5, 21.0, 21.5]}, index=idx)
        html_single = utils.get_forecast_trend_plot_html(df_single, "Woonkamer temperature (°C)")
        self.assertIsInstance(html_single, str)
        self.assertIn("plotly", html_single.lower())
        # Multi-column case (heating-need-forecast's own forecast + flat
        # comfort_min_temp reference line).
        df_multi = pd.DataFrame(
            {"forecast": [19.5, 19.2, 18.9, 18.6], "comfort_min_temp": [19.0] * 4}, index=idx
        )
        html_multi = utils.get_forecast_trend_plot_html(df_multi, "Indoor temperature (°C)")
        self.assertIsInstance(html_multi, str)
        self.assertIn("plotly", html_multi.lower())

    def test_render_horizon_polar_grid_includes_combined_and_every_panel(self):
        """Replaces pv-horizon-refit's old flat per-(panel, azimuth, season)
        table - one chart per panel plus one combined, as a grid of Plotly
        polar bar charts. Panel labels are arbitrary strings under test,
        chosen so they can't collide with anything in Plotly's own bundled
        JS source (unlike a generic marker like the literal word
        "barpolar", which the library's own code also contains)."""
        profile = {
            "0": {"summer": {"elevation": 10.0, "transmittance": 0.2}},
            "90": {"summer": {"elevation": 5.0, "transmittance": 0.1}},
        }
        profile_per_panel = {
            "panel_alpha_zz": {"0": {"summer": {"elevation": 12.0, "transmittance": 0.3}}},
            "panel_beta_zz": {"0": {"summer": {"elevation": 8.0, "transmittance": 0.15}}},
        }

        html = utils.render_horizon_polar_grid(profile, profile_per_panel, "summer")

        self.assertIsInstance(html, str)
        self.assertIn("plotly", html.lower())
        self.assertIn("Combined (all panels)", html)
        self.assertIn("panel_alpha_zz", html)
        self.assertIn("panel_beta_zz", html)

    def test_render_horizon_polar_grid_wedges_are_full_radius_stacked_bands(self):
        """Every azimuth is a complete center-to-rim wedge (a 'vakje'), not
        a bar that stops short - an open-sky band from the center out to
        (90 - elevation), then the transmittance-colored band stacked on
        top of that for the remaining `elevation` degrees out to the rim
        (center=zenith, rim=horizon). The blocked band's own r therefore
        stays numerically equal to elevation - hover still shows the true
        value with no separate transform needed."""
        import re

        profile = {
            "0": {"summer": {"elevation": 0.0, "transmittance": 0.0}},  # fully clear
            "90": {"summer": {"elevation": 18.4, "transmittance": 0.15}},  # partial
        }

        html = utils.render_horizon_polar_grid(profile, {}, "summer")

        match = re.search(r"Plotly\.newPlot\(\s*\"[^\"]+\",\s*(\[.*?\]),\s*\{", html, re.DOTALL)
        traces = json.loads(match.group(1))
        clear_trace = next(t for t in traces if t.get("marker", {}).get("color") == "#eaf4fb")
        blocked_trace = next(t for t in traces if t.get("marker", {}).get("color") == [0.0, 0.15])

        self.assertEqual(clear_trace["theta"], [0, 90])
        self.assertEqual(clear_trace["r"], [90.0, 71.6])
        self.assertIsNone(clear_trace.get("base"))
        self.assertEqual(blocked_trace["r"], [0.0, 18.4])
        self.assertEqual(blocked_trace["base"], [90.0, 71.6])
        # Clear band + blocked band always reach exactly the rim (90) together.
        for clear_r, blocked_r in zip(clear_trace["r"], blocked_trace["r"], strict=True):
            self.assertAlmostEqual(clear_r + blocked_r, 90.0, places=6)

    def test_render_horizon_polar_grid_negative_elevation_does_not_crash(self):
        """A bin with no obstruction detected has a negative learned
        elevation (see pv_shading_kalman.py) - must clamp to 0 rather than
        produce an invalid negative-radius bar."""
        profile = {
            "0": {"summer": {"elevation": -8.6, "transmittance": 0.0}},
            "90": {"summer": {"elevation": 11.6, "transmittance": 0.14}},
        }

        html = utils.render_horizon_polar_grid(profile, {}, "summer")

        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)

    def test_render_horizon_polar_grid_missing_season_renders_empty_not_crash(self):
        """A season with no data anywhere yet (e.g. only "summer" exists
        until enough of the year has passed) must render an empty chart,
        not raise."""
        profile = {"0": {"summer": {"elevation": 10.0, "transmittance": 0.2}}}
        profile_per_panel = {
            "panel_gamma_zz": {"0": {"summer": {"elevation": 9.0, "transmittance": 0.1}}}
        }

        html = utils.render_horizon_polar_grid(profile, profile_per_panel, "winter")

        self.assertIsInstance(html, str)
        self.assertIn("panel_gamma_zz", html)

    def test_render_horizon_polar_grid_without_per_panel_sensors(self):
        """No per-panel sensors configured (profile_per_panel empty/falsy) -
        must still render the combined chart alone."""
        profile = {"0": {"summer": {"elevation": 10.0, "transmittance": 0.2}}}

        html = utils.render_horizon_polar_grid(profile, {}, "summer")

        self.assertIn("Combined (all panels)", html)

    def test_render_horizon_polar_grid_marks_blind_azimuths(self):
        """A geometrically-blind bin (self-shaded, or the sun never
        reaches there at this latitude) must render distinctly from a
        confirmed-clear reading, not silently look identical to it."""
        profile = {"0": {"summer": {"elevation": 0.0, "transmittance": 0.0}}}
        profile_per_panel = {
            "panel_delta_zz": {"0": {"summer": {"elevation": 0.0, "transmittance": 0.0}}}
        }

        html = utils.render_horizon_polar_grid(
            profile,
            profile_per_panel,
            "summer",
            blind_azimuths_per_panel={"panel_delta_zz": {0, 15}},
            blind_azimuths_combined={0},
        )

        self.assertIn("no direct sun ever reaches", html)
        self.assertIn("lightgrey", html)

    def test_render_horizon_polar_grid_blind_params_default_to_no_marking(self):
        """Omitting the new blind-azimuth params (existing call sites from
        before this feature) must not error and must not mark anything."""
        profile = {"0": {"summer": {"elevation": 10.0, "transmittance": 0.2}}}
        profile_per_panel = {
            "panel_epsilon_zz": {"0": {"summer": {"elevation": 8.0, "transmittance": 0.1}}}
        }

        html = utils.render_horizon_polar_grid(profile, profile_per_panel, "summer")

        self.assertNotIn("no direct sun ever reaches", html)

    def test_get_injection_dict_thermal_models(self):
        """Shared by thermal-models-refit/-tune/-forecast (see
        web_server.py) - one <h4> + full key/value table per model that
        actually ran, a 'no result' note for a model that declined (None),
        and per-room honest-test charts (via get_room_temp_test_plot_html)
        for any room_temp_test_plot_df a result carries (refit-only in
        practice, but the helper itself doesn't special-case which model
        key it came from)."""
        idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
        df_plot = pd.DataFrame(
            {
                "train": [20.0, 20.5, np.nan, np.nan],
                "test": [np.nan, np.nan, 21.0, 21.5],
                "pred": [np.nan, np.nan, 20.9, 21.4],
            },
            index=idx,
        )
        results = {
            "heating_model": {"deployed": True, "val_mae_c": 0.42, "test_mae_c": 0.51},
            "hybrid_heatpump_model": None,
            "self_learning_physics_model": {
                "deployed": True,
                "electric_mae_w": 12.3,
                "room_temp_test_plot_df": {"Woonkamer": df_plot},
            },
        }

        injection_dict = utils.get_injection_dict_thermal_models(
            results, "<h2>Thermal models refit</h2>"
        )

        self.assertEqual(injection_dict["title"], "<h2>Thermal models refit</h2>")
        # One subsubtitle+table pair for heating_model (deployed dict).
        heating_titles = [v for k, v in injection_dict.items() if "Heating model" in str(v)]
        self.assertTrue(heating_titles)
        heating_tables = [
            v for k, v in injection_dict.items() if k.startswith("table") and "val_mae_c" in str(v)
        ]
        self.assertTrue(heating_tables)
        # A "no result" note for the declined (None) hybrid_heatpump_model,
        # no table for it.
        self.assertTrue(
            any("Hybrid heat pump model: no result" in str(v) for v in injection_dict.values())
        )
        # A room chart for self_learning_physics_model's own room, rendered
        # as a real Plotly fragment (not just present as a raw DataFrame).
        figure_values = [v for k, v in injection_dict.items() if k.startswith("figure_")]
        self.assertEqual(len(figure_values), 1)
        self.assertIn("plotly", figure_values[0].lower())
        self.assertIn("Woonkamer", figure_values[0])
        # room_temp_test_plot_df itself must never leak into a table cell -
        # it's a DataFrame, not a scalar/short value worth a table row.
        for key, value in injection_dict.items():
            if key.startswith("table"):
                self.assertNotIn("room_temp_test_plot_df", str(value))

    def test_get_injection_dict_thermal_models_forecast_charts(self):
        """Predict-side sibling of test_get_injection_dict_thermal_models -
        indoor_temp_forecast_df (heating_model), electric_forecast_series/
        gas_forecast_series (hybrid_heatpump_model), and room_temp_forecast_df
        (self_learning_physics_model, dict of room -> Series) each render as
        their own real Plotly chart via get_forecast_trend_plot_html, and
        none of them leak into the generic key/value table."""
        idx = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
        results = {
            "heating_model": {
                "heating_needed_by": "beyond_horizon",
                "indoor_temp_forecast_df": pd.DataFrame(
                    {"forecast": [19.5, 19.2, 18.9, 18.6], "comfort_min_temp": [19.0] * 4}, index=idx
                ),
            },
            "hybrid_heatpump_model": {
                "mean_electric_forecast_w": 400.0,
                "electric_forecast_series": pd.Series([400.0] * 4, index=idx),
                "gas_forecast_series": pd.Series([0.02] * 4, index=idx),
            },
            "self_learning_physics_model": {
                "mean_electric_forecast_w": 350.0,
                "room_temp_forecast_df": {
                    "Woonkamer": pd.Series([21.0, 21.1, 21.2, 21.3], index=idx),
                },
            },
        }

        injection_dict = utils.get_injection_dict_thermal_models(
            results, "<h2>Thermal models forecast</h2>"
        )

        figure_values = [v for k, v in injection_dict.items() if k.startswith("figure_")]
        # 1 (heating) + 2 (hybrid electric/gas) + 1 (self-learning-physics room)
        self.assertEqual(len(figure_values), 4)
        for html in figure_values:
            self.assertIn("plotly", html.lower())
        # Chart-carrying keys must never leak into a generic table cell.
        for key, value in injection_dict.items():
            if key.startswith("table"):
                text = str(value)
                self.assertNotIn("indoor_temp_forecast_df", text)
                self.assertNotIn("electric_forecast_series", text)
                self.assertNotIn("gas_forecast_series", text)
                self.assertNotIn("room_temp_forecast_df", text)
                # A plain summary stat must still show up normally.
                self.assertTrue(
                    "400.0" in text or "350.0" in text or "beyond_horizon" in text
                )

    async def test_treat_runtimeparams_historic_days_to_retrieve(self):
        # Setup base configuration
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "forecast-model-fit"
        # Case 1: Parameter NOT provided in runtimeparams
        # Should fallback to config default (usually 2), which is < 9, so it should be forced to 9.
        runtimeparams_empty = {}
        runtimeparams_json_1 = orjson.dumps(runtimeparams_empty).decode("utf-8")
        params_1, _, _, _ = await utils.treat_runtimeparams(
            runtimeparams_json_1,
            self.params_json,
            retrieve_hass_conf.copy(),
            optim_conf.copy(),
            plant_conf.copy(),
            set_type,
            logger,
            emhass_conf,
        )
        params_1 = orjson.loads(params_1)
        self.assertEqual(
            params_1["passed_data"]["historic_days_to_retrieve"],
            9,
            "If not provided (and default < 9), should be forced to 9",
        )
        # Case 2: Provided but < 9 (e.g. 5 days)
        # The user specifically asks for 5 days. Since 5 < 9, validation should force it to 9.
        runtimeparams_low = {"historic_days_to_retrieve": 5}
        runtimeparams_json_2 = orjson.dumps(runtimeparams_low).decode("utf-8")
        params_2, _, _, _ = await utils.treat_runtimeparams(
            runtimeparams_json_2,
            self.params_json,
            retrieve_hass_conf.copy(),
            optim_conf.copy(),
            plant_conf.copy(),
            set_type,
            logger,
            emhass_conf,
        )
        params_2 = orjson.loads(params_2)
        self.assertEqual(
            params_2["passed_data"]["historic_days_to_retrieve"],
            9,
            "If provided value is < 9, should be overridden to 9",
        )
        # Case 3: Provided and >= 9 (e.g. 26 days)
        # This is the fix verification. It should NOT be overridden.
        runtimeparams_high = {"historic_days_to_retrieve": 26}
        runtimeparams_json_3 = orjson.dumps(runtimeparams_high).decode("utf-8")
        params_3, _, _, _ = await utils.treat_runtimeparams(
            runtimeparams_json_3,
            self.params_json,
            retrieve_hass_conf.copy(),
            optim_conf.copy(),
            plant_conf.copy(),
            set_type,
            logger,
            emhass_conf,
        )
        params_3 = orjson.loads(params_3)
        self.assertEqual(
            params_3["passed_data"]["historic_days_to_retrieve"],
            26,
            "If provided value is >= 9, it should be respected",
        )

    async def test_treat_runtimeparams_power_limits_parsing(self):
        """Test parsing of power limits (scalar, list, stringified list)."""
        params = await TestUtils.get_test_params()
        params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
            params["retrieve_hass_conf"]["optimization_time_step"], "minutes"
        )
        params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(
            days=params["optim_conf"]["delta_forecast_daily"]
        )
        retrieve_hass_conf = params["retrieve_hass_conf"]
        optim_conf = params["optim_conf"]
        plant_conf = params["plant_conf"]
        # Test Scalars (should remain scalars)
        runtimeparams = json.dumps(
            {"maximum_power_from_grid": 5000, "maximum_power_to_grid": 2000.5}
        )
        _, _, _, plant_conf_out = await treat_runtimeparams(
            runtimeparams,
            deepcopy(params),
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )
        self.assertEqual(plant_conf_out["maximum_power_from_grid"], 5000)
        self.assertEqual(plant_conf_out["maximum_power_to_grid"], 2000.5)
        # Test Lists (should remain lists)
        runtimeparams = json.dumps(
            {"maximum_power_from_grid": [1000, 2000, 3000], "maximum_power_to_grid": [4000, 5000]}
        )
        _, _, _, plant_conf_out = await treat_runtimeparams(
            runtimeparams,
            deepcopy(params),
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )
        self.assertIsInstance(plant_conf_out["maximum_power_from_grid"], list)
        self.assertListEqual(plant_conf_out["maximum_power_from_grid"], [1000, 2000, 3000])
        self.assertListEqual(plant_conf_out["maximum_power_to_grid"], [4000, 5000])
        # Test Stringified Lists (should be parsed into lists)
        runtimeparams = json.dumps(
            {"maximum_power_from_grid": "[100, 200, 300]", "maximum_power_to_grid": "[400, 500]"}
        )
        _, _, _, plant_conf_out = await treat_runtimeparams(
            runtimeparams,
            deepcopy(params),
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )
        self.assertIsInstance(plant_conf_out["maximum_power_from_grid"], list)
        self.assertListEqual(plant_conf_out["maximum_power_from_grid"], [100, 200, 300])
        self.assertListEqual(plant_conf_out["maximum_power_to_grid"], [400, 500])

    async def test_treat_runtimeparams_power_limits_invalid(self):
        """Test invalid power limit strings triggers warning but doesn't crash."""
        params = await TestUtils.get_test_params()
        params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
            params["retrieve_hass_conf"]["optimization_time_step"], "minutes"
        )
        params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(
            days=params["optim_conf"]["delta_forecast_daily"]
        )
        retrieve_hass_conf = params["retrieve_hass_conf"]
        optim_conf = params["optim_conf"]
        plant_conf = params["plant_conf"]
        # Default values from config to verify fallback
        default_from_grid = plant_conf.get("maximum_power_from_grid")
        runtimeparams = json.dumps({"maximum_power_from_grid": "this-is-not-a-list"})
        # Capture logs to verify warning
        with self.assertLogs(logger, level="WARNING") as cm:
            _, _, _, plant_conf_out = await treat_runtimeparams(
                runtimeparams,
                deepcopy(params),
                retrieve_hass_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                logger,
                emhass_conf,
            )
        # Verify the warning message was logged
        self.assertTrue(any("Could not parse maximum_power_from_grid" in o for o in cm.output))
        # Verify it fell back to default or kept original value (depending on logic, usually implies no change)
        self.assertEqual(plant_conf_out["maximum_power_from_grid"], default_from_grid)

    async def test_treat_runtimeparams_preserves_out_of_band_soc_init(self):
        """Naive MPC should preserve the real initial SOC even if it is out of bounds."""
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        runtimeparams = {
            "prediction_horizon": 10,
            "soc_init": 0.05,
            "soc_final": 0.6,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        params_out, _, _, _ = await treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )
        params_out = orjson.loads(params_out)

        self.assertEqual(params_out["passed_data"]["soc_init"], 0.05)
        self.assertEqual(params_out["passed_data"]["soc_final"], 0.6)

    async def test_treat_runtimeparams_preserves_high_out_of_band_soc_init(self):
        """Naive MPC should preserve a high initial SOC that starts above soc_max."""
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        runtimeparams = {
            "prediction_horizon": 10,
            "soc_init": 0.95,
            "soc_final": 0.6,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        params_out, _, _, _ = await treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )
        params_out = orjson.loads(params_out)

        self.assertEqual(params_out["passed_data"]["soc_init"], 0.95)
        self.assertEqual(params_out["passed_data"]["soc_final"], 0.6)

    async def test_treat_runtimeparams_ignore_pv_feedback_during_curtailment(self):
        """Wiring for ignore_pv_feedback_during_curtailment runtime flag (#818).

        The read site in forecast.py reads from params["passed_data"]; this
        test pins the runtime → passed_data path for the four realistic input
        shapes: missing key (default False), JSON bool true, string "true",
        string "false". The string cases document the bool() coerce behaviour.
        """
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        async def run(runtimeparams_dict):
            runtimeparams_json = orjson.dumps(runtimeparams_dict).decode("utf-8")
            params_out, _, _, _ = await treat_runtimeparams(
                runtimeparams_json,
                params_json,
                retrieve_hass_conf,
                optim_conf,
                plant_conf,
                "naive-mpc-optim",
                logger,
                emhass_conf,
            )
            return orjson.loads(params_out)

        # Case 1: key absent -> default False
        out = await run({"prediction_horizon": 10})
        self.assertIs(out["passed_data"]["ignore_pv_feedback_during_curtailment"], False)

        # Case 2: JSON bool true -> True
        out = await run({"prediction_horizon": 10, "ignore_pv_feedback_during_curtailment": True})
        self.assertIs(out["passed_data"]["ignore_pv_feedback_during_curtailment"], True)

        # Case 3: string "true" -> bool() coerce -> True
        out = await run({"prediction_horizon": 10, "ignore_pv_feedback_during_curtailment": "true"})
        self.assertIs(out["passed_data"]["ignore_pv_feedback_during_curtailment"], True)

        # Case 4: string "false" -> bool() coerce -> True (Python bool("false") is True)
        # Documents the known limitation of bool() on non-empty strings;
        # JSON bool transport is the supported shape.
        out = await run(
            {"prediction_horizon": 10, "ignore_pv_feedback_during_curtailment": "false"}
        )
        self.assertIs(out["passed_data"]["ignore_pv_feedback_during_curtailment"], True)

    async def test_treat_runtimeparams_handles_string_null_heat_topology(self):
        """String "null" heat_topology must not crash; warning logged; no compiled fields merged."""
        params = await TestUtils.get_test_params()
        params["optim_conf"]["heat_topology"] = "null"
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        original_num_loads = optim_conf["number_of_deferrable_loads"]

        runtimeparams_json = orjson.dumps({}).decode("utf-8")

        with self.assertLogs(logger, level="WARNING") as log_cm:
            _, _, optim_conf_out, _ = await treat_runtimeparams(
                runtimeparams_json,
                params_json,
                retrieve_hass_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                logger,
                emhass_conf,
            )

        self.assertEqual(optim_conf_out["number_of_deferrable_loads"], original_num_loads)
        self.assertTrue(any("heat_topology" in m for m in log_cm.output))

    async def test_treat_runtimeparams_compiles_valid_heat_topology(self):
        """Valid dict heat_topology is compiled and merged into optim_conf."""
        params = await TestUtils.get_test_params()
        topo = {
            "sources": [
                {
                    "id": "boiler",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 10000,
                    "min_power": 2000,
                }
            ],
            "storage": [
                {
                    "id": "tank",
                    "volume": 0.1,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [60] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "boiler", "to": "tank"}],
        }
        params["optim_conf"]["heat_topology"] = topo
        params["optim_conf"]["heatpump_config_mode"] = "graph_topology"
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        runtimeparams_json = orjson.dumps({}).decode("utf-8")
        _, _, optim_conf_out, _ = await treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

        self.assertEqual(optim_conf_out["number_of_deferrable_loads"], 1)
        self.assertEqual(len(optim_conf_out["def_load_config"]), 1)

    async def test_treat_runtimeparams_bool_coercion(self):
        """_cast_bool None-guard and scalar-padding paths must be covered.

        def_current_state=None hits `if value is None` in _cast_bool and the
        scalar else-branch (padded to n_loads).  set_deferrable_load_single_constant
        with a scalar bool hits its scalar else-branch identically.
        """
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        runtimeparams_json = orjson.dumps(
            {
                "def_current_state": None,
                "set_deferrable_load_single_constant": False,
            }
        ).decode("utf-8")
        _, _, optim_conf_out, _ = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )
        n = len(optim_conf_out["nominal_power_of_deferrable_loads"])
        self.assertEqual(optim_conf_out["def_current_state"], [False] * n)
        self.assertEqual(optim_conf_out["set_deferrable_load_single_constant"], [False] * n)

    def test_param_to_config(self):
        """Test converting built params back to a flat config dictionary and masking secrets."""
        # Create a mock parameter dictionary with the required categories
        mock_param = {
            "retrieve_hass_conf": {
                "hass_url": "http://secret",  # This should be masked
                "optimization_time_step": 30,
                "time_zone": "Europe/Paris",
            },
            "optim_conf": {"set_use_battery": True, "costfun": "profit"},
            "plant_conf": {"battery_capacity": 10.0},
        }
        # Execute
        result_config = utils.param_to_config(mock_param, logger)
        # Verify structure was flattened correctly
        self.assertIn("optimization_time_step", result_config)
        self.assertIn("set_use_battery", result_config)
        self.assertIn("battery_capacity", result_config)
        # Verify secrets were excluded
        self.assertNotIn("hass_url", result_config)
        # Verify values transferred correctly
        self.assertEqual(result_config["costfun"], "profit")
        self.assertTrue(result_config["set_use_battery"])

    def test_check_def_loads(self):
        """Test padding of deferrable load parameter lists."""
        default_val = 5
        # Case 1: Needs padding (num_def_loads > list length)
        parameter = {"operating_hours": [3, 4]}
        result1 = utils.check_def_loads(4, parameter, default_val, "operating_hours", logger)
        self.assertEqual(len(result1), 4)
        self.assertEqual(result1, [3, 4, 5, 5])
        # Case 2: No padding needed (num_def_loads == list length)
        parameter = {"operating_hours": [3, 4]}  # Reset
        result2 = utils.check_def_loads(2, parameter, default_val, "operating_hours", logger)
        self.assertEqual(len(result2), 2)
        self.assertEqual(result2, [3, 4])
        # Case 3: Missing key -> fill with defaults instead of raising KeyError (#929)
        parameter = {"other_param": "test"}
        result3 = utils.check_def_loads(2, parameter, default_val, "missing_key", logger)
        self.assertEqual(result3, [5, 5])
        self.assertEqual(parameter["missing_key"], [5, 5])
        # Case 4: Explicit JSON null (None) -> fill with defaults, no crash
        parameter = {"deferrable_load_max_cost": None}
        result4 = utils.check_def_loads(3, parameter, 0.0, "deferrable_load_max_cost", logger)
        self.assertEqual(result4, [0.0, 0.0, 0.0])
        # Case 5 (#929 regression): a per-load list shorter than the load count
        # (e.g. the length-2 shipped default vs number_of_deferrable_loads=3) must pad
        # SILENTLY. Enlarging-to-fit is the function's documented job, not a warning, so
        # it is logged at DEBUG and never at WARNING.
        parameter = {"deferrable_load_max_cost": [0.0, 0.0]}
        with self.assertLogs(logger, level="DEBUG") as cm:
            result5 = utils.check_def_loads(3, parameter, 0.0, "deferrable_load_max_cost", logger)
        self.assertEqual(result5, [0.0, 0.0, 0.0])
        self.assertTrue(any(r.levelno == logging.DEBUG for r in cm.records))
        self.assertFalse(any(r.levelno >= logging.WARNING for r in cm.records))

    def test_normalize_deferrable_load_categories_dispatch_modes(self):
        """Program loads force sequence mode; energy_kwh targets are normalized to float."""
        params = {
            "optim_conf": {
                "number_of_deferrable_loads": 2,
                "load_type": ["program_based", "fixed_power_non_splittable"],
                "load_programs": [
                    '[{"name":"eco","power_pattern":"100,200,300"}]',
                    "[]",
                ],
                "load_dispatch_mode": ["hours", "energy_kwh"],
                "required_energy_kwh_of_each_deferrable_load": ["1.2", "2.5"],
                "nominal_power_of_deferrable_loads": [1500.0, 1000.0],
                "operating_hours_of_each_deferrable_load": [0, 0],
                "treat_deferrable_load_as_semi_cont": [True, True],
            }
        }

        utils._normalize_deferrable_load_categories(params, logger)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["load_dispatch_mode"][0], "program")
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"][0], [100.0, 200.0, 300.0])
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"][0], 3)
        self.assertEqual(optim_conf["required_energy_kwh_of_each_deferrable_load"][1], 2.5)

    def test_strip_auto_appended_loads_removes_matching_and_keeps_others(self):
        """Only entries whose own _source, or nested thermal_battery._source,
        is in the given marker set are removed - a manually-declared load
        (no _source at all) and an unrelated marker survive untouched, at
        their original relative order, with number_of_deferrable_loads
        decremented to match."""
        optim_conf = {
            "number_of_deferrable_loads": 3,
            "def_load_config": [
                {},  # manual load, no _source
                {"thermal_battery": {"name": "Living Room", "_source": "room_auto"}},
                {"_source": "ev_auto", "name": "Zappi"},
            ],
            "nominal_power_of_deferrable_loads": [3000.0, 1500.0, 11000.0],
            "load_type": ["fixed_power_non_splittable"] * 3,
        }

        utils._strip_auto_appended_loads(optim_conf, {"ev_auto"})

        self.assertEqual(optim_conf["number_of_deferrable_loads"], 2)
        self.assertEqual(len(optim_conf["def_load_config"]), 2)
        self.assertNotIn("_source", optim_conf["def_load_config"][0])
        self.assertEqual(
            optim_conf["def_load_config"][1]["thermal_battery"]["_source"], "room_auto"
        )
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [3000.0, 1500.0])
        self.assertEqual(len(optim_conf["load_type"]), 2)

    def test_strip_auto_appended_loads_noop_when_nothing_matches(self):
        optim_conf = {
            "number_of_deferrable_loads": 1,
            "def_load_config": [{"thermal_battery": {"name": "Living Room", "_source": "room_auto"}}],
            "nominal_power_of_deferrable_loads": [1500.0],
        }
        before = {k: (list(v) if isinstance(v, list) else v) for k, v in optim_conf.items()}

        utils._strip_auto_appended_loads(optim_conf, {"ev_auto"})

        self.assertEqual(optim_conf["def_load_config"], before["def_load_config"])
        self.assertEqual(optim_conf["number_of_deferrable_loads"], before["number_of_deferrable_loads"])
        self.assertEqual(
            optim_conf["nominal_power_of_deferrable_loads"],
            before["nominal_power_of_deferrable_loads"],
        )

    def test_prune_orphaned_deferrable_load_slots_removes_empty_untagged_slot(self):
        """A def_load_config slot beyond load_names with no _source marker
        and no configured value anywhere (nominal power, manual flag,
        WashData flag, required energy, operating hours all falsy/absent)
        is leftover corruption from the pre-marker-system save-cycle bug -
        it gets dropped, with every parallel per-load array shrinking in
        lockstep and number_of_deferrable_loads decremented to match. This
        is the exact shape reported by a real user: 2 named manual loads,
        1 orphaned empty slot, 1 real room_auto load."""
        optim_conf = {
            "number_of_deferrable_loads": 4,
            "load_names": ["Vaatwasser", "Wasmachine"],
            "def_load_config": [
                {},
                {},
                {},
                {"thermal_battery": {"name": "Woonkamer", "_source": "room_auto"}},
            ],
            "nominal_power_of_deferrable_loads": [3000.0, 3000.0, 0.0, 1500.0],
            "is_manual_load": [True, True, False, False],
            "load_washdata_enabled": [True, True, False, False],
            "required_energy_kwh_of_each_deferrable_load": [0.0, 0.0, 0.0, 0.0],
            "operating_hours_of_each_deferrable_load": [4, 0, 0, 0],
            "load_type": ["program_based", "program_based", "fixed_power_non_splittable", "fixed_power_non_splittable"],
        }

        utils._prune_orphaned_deferrable_load_slots(optim_conf, logger)

        self.assertEqual(optim_conf["number_of_deferrable_loads"], 3)
        self.assertEqual(len(optim_conf["def_load_config"]), 3)
        self.assertEqual(
            optim_conf["def_load_config"][2]["thermal_battery"]["name"], "Woonkamer"
        )
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [3000.0, 3000.0, 1500.0])
        self.assertEqual(optim_conf["is_manual_load"], [True, True, False])
        self.assertEqual(len(optim_conf["load_type"]), 3)

    def test_prune_orphaned_deferrable_load_slots_keeps_tagged_entry(self):
        """An entry beyond load_names carrying a recognized _source marker
        is never touched by pruning, even though it's outside the named
        range - it stays owned by _strip_auto_appended_loads instead."""
        optim_conf = {
            "number_of_deferrable_loads": 2,
            "load_names": ["Vaatwasser"],
            "def_load_config": [
                {},
                {"_source": "ev_auto", "name": "ev_1"},
            ],
            "nominal_power_of_deferrable_loads": [3000.0, 0.0],
        }

        utils._prune_orphaned_deferrable_load_slots(optim_conf, logger)

        self.assertEqual(optim_conf["number_of_deferrable_loads"], 2)
        self.assertEqual(len(optim_conf["def_load_config"]), 2)
        self.assertEqual(optim_conf["def_load_config"][1]["_source"], "ev_auto")

    def test_prune_orphaned_deferrable_load_slots_keeps_slot_with_real_signal(self):
        """A slot beyond load_names with an empty def_load_config entry but
        a real configured value elsewhere (nonzero nominal power) is not an
        orphan - e.g. a manual load configured only via the array fields,
        with no thermal_battery. Left untouched."""
        optim_conf = {
            "number_of_deferrable_loads": 2,
            "load_names": ["Vaatwasser"],
            "def_load_config": [{}, {}],
            "nominal_power_of_deferrable_loads": [3000.0, 1200.0],
        }

        utils._prune_orphaned_deferrable_load_slots(optim_conf, logger)

        self.assertEqual(optim_conf["number_of_deferrable_loads"], 2)
        self.assertEqual(len(optim_conf["def_load_config"]), 2)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [3000.0, 1200.0])

    def test_prune_orphaned_deferrable_load_slots_noop_when_nothing_orphaned(self):
        optim_conf = {
            "number_of_deferrable_loads": 1,
            "load_names": ["Vaatwasser"],
            "def_load_config": [{}],
            "nominal_power_of_deferrable_loads": [3000.0],
        }
        before = {k: (list(v) if isinstance(v, list) else v) for k, v in optim_conf.items()}

        utils._prune_orphaned_deferrable_load_slots(optim_conf, logger)

        self.assertEqual(optim_conf["def_load_config"], before["def_load_config"])
        self.assertEqual(
            optim_conf["number_of_deferrable_loads"], before["number_of_deferrable_loads"]
        )
        self.assertEqual(
            optim_conf["nominal_power_of_deferrable_loads"],
            before["nominal_power_of_deferrable_loads"],
        )

    async def test_append_boiler_thermal_battery_loads(self):
        """Boiler configuration should append thermal_battery loads with legionella metadata."""
        params = {
            "retrieve_hass_conf": {
                "optimization_time_step": pd.to_timedelta(30, "min"),
            },
            "optim_conf": {
                "set_use_boiler": True,
                "number_of_boilers": 1,
                "boiler_names": ["dhw_tank"],
                "boiler_type": ["hpboiler"],
                "boiler_nominal_power": [1500.0],
                "boiler_volume_l": [180.0],
                "boiler_supply_temperature": [55.0],
                "boiler_start_temperature": [49.0],
                "boiler_target_temperature": [52.0],
                "boiler_min_temperature": [45.0],
                "boiler_max_temperature": [60.0],
                "boiler_loss_factor": [0.02],
                "boiler_dhw_draw_kwh_forecast": ["0.2,0.1,0.0"],
                "boiler_legionella_interval_days": [7],
                "boiler_legionella_target_temp": [60.0],
                "boiler_legionella_hold_hours": [0.5],
                "boiler_legionella_last_run_iso": [""],
                "boiler_legionella_force_resistive": [True],
                "boiler_coupled_heatpump_load_index": [-1],
                "boiler_hp_shared_max_power": [0.0],
                "boiler_uncertainty_margin_kwh": [0.2],
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 1)
        self.assertEqual(len(optim_conf["def_load_config"]), 1)
        thermal_cfg = optim_conf["def_load_config"][0]["thermal_battery"]
        self.assertEqual(thermal_cfg["name"], "dhw_tank")
        self.assertEqual(thermal_cfg["boiler_type"], "resistive")
        self.assertTrue(thermal_cfg["legionella_due"])
        self.assertGreaterEqual(thermal_cfg["legionella_target_temperature"], 60.0)

    async def test_append_boiler_thermal_battery_loads_idempotent_across_repeated_calls(self):
        """Regression test for the real bug this session fixed: calling this
        function twice on the SAME optim_conf (simulating the config page's
        GET /get-config -> edit -> POST /set-config round trip, which both
        run build_params - and therefore this function - against an
        already-derived config) must not double the boiler's appended load.
        Before the fix, number_of_deferrable_loads/every parallel array/
        def_load_config would all grow by 1 extra boiler entry per call."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_boiler": True,
                "number_of_boilers": 1,
                "boiler_names": ["dhw_tank"],
                "boiler_type": ["resistive"],
                "boiler_nominal_power": [1500.0],
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)
        after_first = dict(params["optim_conf"])
        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)
        after_second = params["optim_conf"]

        self.assertEqual(after_second["number_of_deferrable_loads"], 1)
        self.assertEqual(len(after_second["def_load_config"]), 1)
        self.assertEqual(
            after_second["nominal_power_of_deferrable_loads"],
            after_first["nominal_power_of_deferrable_loads"],
        )

    async def test_append_boiler_thermal_battery_loads_disabling_cleans_up_stale_entry(self):
        """Turning set_use_boiler off must remove the previously-appended
        boiler entry, not leave it orphaned in def_load_config forever."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_boiler": True,
                "number_of_boilers": 1,
                "boiler_names": ["dhw_tank"],
                "boiler_type": ["resistive"],
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }
        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)
        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 1)

        params["optim_conf"]["set_use_boiler"] = False
        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)

        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 0)
        self.assertEqual(params["optim_conf"]["def_load_config"], [])

    async def test_append_boiler_thermal_battery_loads_overlays_persisted_last_run(self):
        """A backend-persisted legionella_last_run_iso should override the
        config-supplied default, so a completed cycle can clear legionella_due
        without ever rewriting config.json."""
        import tempfile

        from emhass.persistence import save_json_blob

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_emhass_conf = dict(emhass_conf)
            tmp_emhass_conf["data_path"] = pathlib.Path(tmp_dir)

            recent_iso = pd.Timestamp.now(tz="UTC").isoformat()
            await save_json_blob(
                tmp_emhass_conf,
                "boiler_runtime_state.json",
                {"boiler_legionella_last_run_iso": [recent_iso]},
                logger,
            )

            params = {
                "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
                "optim_conf": {
                    "set_use_boiler": True,
                    "number_of_boilers": 1,
                    "boiler_names": ["dhw_tank"],
                    "boiler_legionella_interval_days": [7],
                    "boiler_legionella_last_run_iso": [""],
                    "delta_forecast_daily": pd.to_timedelta(1, "days"),
                    "number_of_deferrable_loads": 0,
                    "nominal_power_of_deferrable_loads": [],
                    "minimum_power_of_deferrable_loads": [],
                    "operating_hours_of_each_deferrable_load": [],
                    "start_timesteps_of_each_deferrable_load": [],
                    "end_timesteps_of_each_deferrable_load": [],
                    "set_deferrable_startup_penalty": [],
                    "set_deferrable_load_single_constant": [],
                    "treat_deferrable_load_as_semi_cont": [],
                    "load_type": [],
                    "load_dispatch_mode": [],
                    "required_energy_kwh_of_each_deferrable_load": [],
                    "def_load_config": [],
                },
            }

            await utils._append_boiler_thermal_battery_loads(params, logger, tmp_emhass_conf)

            thermal_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
            # A legionella cycle within the interval should no longer be due.
            self.assertFalse(thermal_cfg["legionella_due"])

    async def test_append_room_thermal_loads_creates_def_load_config_entries(self):
        """Configured rooms and the heat pump dispatch unit should each become
        their own thermal_battery deferrable load, with index bookkeeping in
        passed_data for later stages (schedule flattening, publishing)."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_heatpump": True,
                "heatpump_number_of_rooms": 1,
                "heatpump_room_names": ["Living Room"],
                "heatpump_room_min_temperature": [18.0],
                "heatpump_room_max_temperature": [24.0],
                "heatpump_room_target_temperature": [21.0],
                "heatpump_room_nominal_power": [1500.0],
                "heatpump_room_supply_temperature": [35.0],
                "heatpump_room_volume": [15.0],
                "heatpump_room_shared_group": [0],
                "heatpump_dispatch_control_entity": "switch.climate_control",
                "heatpump_dispatch_min_temperature": 18.0,
                "heatpump_dispatch_max_temperature": 22.0,
                "heatpump_dispatch_target_temperature": 20.0,
                "heatpump_dispatch_nominal_power": 3000.0,
                "heatpump_dispatch_supply_temperature": 35.0,
                "heatpump_dispatch_volume": 20.0,
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 2)
        self.assertEqual(len(optim_conf["def_load_config"]), 2)

        room_cfg = optim_conf["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["name"], "Living Room")
        self.assertEqual(room_cfg["_source"], "room_auto")

        dispatch_cfg = optim_conf["def_load_config"][1]["thermal_battery"]
        self.assertEqual(dispatch_cfg["_source"], "heatpump_dispatch_auto")

        self.assertEqual(params["passed_data"]["room_load_indices"], {"Living Room": 0})
        self.assertEqual(params["passed_data"]["heatpump_dispatch_load_index"], 1)

    def _weather_curve_base_params(self, **optim_conf_overrides):
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_heatpump": True,
                "heatpump_number_of_rooms": 1,
                "heatpump_room_names": ["Woonkamer"],
                "heatpump_room_min_temperature": [18.0],
                "heatpump_room_max_temperature": [24.0],
                "heatpump_room_target_temperature": [21.0],
                "heatpump_room_nominal_power": [1500.0],
                "heatpump_room_supply_temperature": [35.0],
                "heatpump_number_of_units": 1,
                "heatpump_unit_control_mode": ["weather_curve"],
                "heatpump_unit_curve_slope": [-1.0],
                "heatpump_unit_curve_intercept": [40.0],
                "heatpump_unit_supply_temp_min": [20.0],
                "heatpump_unit_supply_temp_max": [70.0],
                "heatpump_room_volume": [15.0],
                "heatpump_room_shared_group": [0],
                "heatpump_room_self_learning_only": [True],
                "heatpump_dispatch_control_entity": "",
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }
        params["optim_conf"].update(optim_conf_overrides)
        return params

    #: Minimal but realistic fitted-coefficients fixture, shared by the
    #: weather_curve MILP tests below - a room's own theta_temp fit plus
    #: the whole-house theta_elec_ fit, matching the two artifacts
    #: refit_self_learning_physics_model actually persists together.
    _WOONKAMER_DISPATCH_BLOB = {
        "rooms": {
            "Woonkamer": {
                "feature_names": ["bias", "room_last", "duty", "delta_supply"],
                "theta": [1.0, 0.9, 0.5, 0.05],
                "neighbors": [],
            }
        },
        "house_elec": {
            "feature_names": ["bias", "duty", "delta_supply", "duty_x_delta_supply"],
            "theta": [50.0, 400.0, 0.0, 20.0],
        },
    }

    async def test_append_room_thermal_loads_weather_curve_single_member_is_decision_variable(self):
        """A self-learning-only room in weather_curve mode, alone in its
        heat-source group (no dispatch entity, only room in the list), with
        both a fitted room-temperature equation AND whole-house electric-
        draw coefficients available, gets the full MILP marker plus the
        heating_curve dict resolve_thermal_battery_cop already knows how to
        consume."""
        params = self._weather_curve_base_params()
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": self._WOONKAMER_DISPATCH_BLOB}
            )
        )

        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(
            hc["heating_curve"],
            {"slope": -1.0, "offset": 40.0, "min_supply": 20.0, "max_supply": 70.0},
        )
        self.assertTrue(hc["supply_temp_is_decision_variable"])
        self.assertEqual(
            hc["self_learning_dispatch_elec"]["feature_names"],
            ["bias", "duty", "delta_supply", "duty_x_delta_supply"],
        )
        self.assertEqual(hc["self_learning_dispatch_elec"]["theta"], [50.0, 400.0, 0.0, 20.0])
        self.assertNotIn("dispatch_mode_fallback_reason", hc)

    async def test_append_room_thermal_loads_multi_unit_resolves_per_room(self):
        """Two rooms on two DIFFERENT heat pump units (heatpump_room_unit
        0/1) must each resolve control_mode/curve/supply-temp-bounds and
        nominal_power from THEIR OWN unit, not from a single shared/global
        value - the core property this refactor exists for. Room 0's unit
        is weather_curve (gets a heating_curve dict); room 1's unit is
        fixed (gets none) - proving control_mode itself is genuinely
        per-unit, not just the numbers that feed it."""
        params = self._weather_curve_base_params(
            heatpump_number_of_rooms=2,
            heatpump_room_names=["Kamer1", "Kamer2"],
            heatpump_room_min_temperature=[18.0, 18.0],
            heatpump_room_max_temperature=[24.0, 24.0],
            heatpump_room_target_temperature=[21.0, 21.0],
            heatpump_room_nominal_power=[1500.0, 1500.0],
            heatpump_room_supply_temperature=[35.0, 35.0],
            heatpump_room_volume=[15.0, 15.0],
            heatpump_room_shared_group=[0, 0],
            heatpump_room_self_learning_only=[False, False],
            heatpump_number_of_units=2,
            heatpump_unit_name=["Unit A", "Unit B"],
            heatpump_unit_nominal_power=[3000.0, 5000.0],
            heatpump_unit_control_mode=["weather_curve", "fixed"],
            heatpump_unit_curve_slope=[-1.2, -1.0],
            heatpump_unit_curve_intercept=[45.0, 40.0],
            heatpump_unit_supply_temp_min=[22.0, 20.0],
            heatpump_unit_supply_temp_max=[65.0, 70.0],
            heatpump_room_unit=[0, 1],
        )

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        def_load_config = params["optim_conf"]["def_load_config"]
        hc0 = def_load_config[0]["thermal_battery"]
        hc1 = def_load_config[1]["thermal_battery"]

        self.assertEqual(
            hc0["heating_curve"],
            {"slope": -1.2, "offset": 45.0, "min_supply": 22.0, "max_supply": 65.0},
        )
        self.assertEqual(hc0["heatpump_unit_nominal_power"], 3000.0)
        self.assertEqual(hc0["heatpump_unit_name"], "Unit A")

        self.assertNotIn("heating_curve", hc1)
        self.assertEqual(hc1["heatpump_unit_nominal_power"], 5000.0)
        self.assertEqual(hc1["heatpump_unit_name"], "Unit B")

        # The aggregate written back to plant_conf is the SUM across both
        # units (used by _build_aggregate_heatpump_duty_expr and friends
        # for the whole-house duty signal) - not either unit's own value.
        self.assertEqual(params["plant_conf"]["heatpump_nominal_power"], 8000.0)

    async def test_append_room_thermal_loads_migrates_legacy_nominal_power(self):
        """A config saved before the Heat Pump Units section existed has no
        heatpump_unit_nominal_power at all, but its raw config.json may
        still carry the old global heatpump_nominal_power value (routed
        read-only into optim_conf via associations.csv - see
        _load_heatpump_units's own docstring). Unit 0 must seed its
        default from that value, not silently reset a real, already-
        configured nominal power back to the generic 3000W default."""
        params = self._weather_curve_base_params(heatpump_nominal_power=4200.0)

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(hc["heatpump_unit_nominal_power"], 4200.0)

    async def test_append_room_thermal_loads_heatpump_unit_out_of_range_falls_back_with_warning(self):
        """A room referencing a heatpump_room_unit index beyond the
        configured unit list (e.g. a unit was removed after the room was
        pointed at it) must fall back to unit 0 with a visible warning,
        never crash the whole config build over one room's stale
        reference."""
        params = self._weather_curve_base_params(
            heatpump_unit_control_mode=["fixed"],
            heatpump_room_unit=[5],
        )

        with self.assertLogs(logger, level="WARNING") as log_ctx:
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(hc["heatpump_unit_nominal_power"], 3000.0)
        self.assertTrue(
            any("out of range" in msg.lower() for msg in log_ctx.output),
            log_ctx.output,
        )

    async def test_append_room_thermal_loads_weather_curve_multi_member_falls_back_visibly(self):
        """The same fitted room, but sharing its heat-source group with a
        whole-house dispatch load (2 members) - the exact-MILP path only
        supports a single-member group, so it must fall back to today's
        two-pass dispatch with a VISIBLE reason (not just a debug log)
        rather than silently downgrading, even though a fit exists."""
        params = self._weather_curve_base_params(
            heatpump_dispatch_control_entity="switch.climate_control",
            heatpump_dispatch_nominal_power=3000.0,
        )
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": self._WOONKAMER_DISPATCH_BLOB}
            )
        )

        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertIn("heating_curve", hc)
        self.assertNotIn("supply_temp_is_decision_variable", hc)
        self.assertIn("2", hc["dispatch_mode_fallback_reason"])
        self.assertIn("two-pass", hc["dispatch_mode_fallback_reason"])

    async def test_append_room_thermal_loads_weather_curve_missing_house_elec_falls_back_visibly(self):
        """A single-member group with a fitted room-temperature equation but
        NO whole-house electric-draw coefficients yet (e.g. refit ran before
        this feature existed) - can't price predicted electricity, so it
        must fall back with its own distinguishable reason rather than
        silently reusing the multi-member message."""
        params = self._weather_curve_base_params()
        blob_without_house_elec = {"rooms": self._WOONKAMER_DISPATCH_BLOB["rooms"]}
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": blob_without_house_elec}
            )
        )

        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertNotIn("supply_temp_is_decision_variable", hc)
        self.assertIn("electric-draw", hc["dispatch_mode_fallback_reason"])
        self.assertIn("two-pass", hc["dispatch_mode_fallback_reason"])

    async def test_append_room_thermal_loads_weather_curve_without_self_learning_only(self):
        """weather_curve mode without self_learning_only still gets the
        richer heating_curve (a free COP-estimate improvement for the
        regular, non-self-learning dispatch path), but neither the MILP
        marker nor a fallback note - there's nothing to fall back FROM."""
        params = self._weather_curve_base_params(heatpump_room_self_learning_only=[False])

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertIn("heating_curve", hc)
        self.assertNotIn("supply_temp_is_decision_variable", hc)
        self.assertNotIn("dispatch_mode_fallback_reason", hc)

    async def test_append_room_thermal_loads_weather_curve_no_fit_yet_falls_back_visibly(self):
        """self_learning_only + weather_curve, but no refit has ever
        produced a fitted equation for this room - must fall back with a
        reason distinguishable from the multi-member/missing-house-elec
        cases (a room can fix this one just by running the refit action)."""
        params = self._weather_curve_base_params()

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        hc = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertNotIn("supply_temp_is_decision_variable", hc)
        self.assertIn("fitted model", hc["dispatch_mode_fallback_reason"])
        self.assertIn("two-pass", hc["dispatch_mode_fallback_reason"])

    async def test_append_room_thermal_loads_registers_supply_temp_target_entity_per_room(self):
        """custom_room_supply_temp_target_id must be appended once per room,
        same cardinality/order as custom_room_target_temp_id (see
        command_line.py::_publish_room_supply_temp_target's own docstring
        for why: its list position must stay aligned with
        room_load_indices's enumeration order) - registered regardless of
        whether this particular room actually qualifies for MILP dispatch,
        since the publisher itself already skips a room with no
        supply_temp_target_heater{k} results column."""
        params = self._weather_curve_base_params(heatpump_room_self_learning_only=[False])

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        passed_data = params["passed_data"]
        self.assertEqual(len(passed_data["custom_room_target_temp_id"]), 1)
        self.assertEqual(len(passed_data["custom_room_supply_temp_target_id"]), 1)
        self.assertEqual(
            passed_data["custom_room_supply_temp_target_id"][0]["entity_id"],
            "sensor.room_supply_temp_target_woonkamer",
        )

    async def test_append_room_thermal_loads_idempotent_across_repeated_calls(self):
        """Same regression as the boiler/EV siblings - calling this function
        twice on the same optim_conf (the config page's GET->edit->POST
        round trip) must not double the room + dispatch entries."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_heatpump": True,
                "heatpump_number_of_rooms": 1,
                "heatpump_room_names": ["Living Room"],
                "heatpump_room_nominal_power": [1500.0],
                "heatpump_dispatch_control_entity": "switch.climate_control",
                "heatpump_dispatch_nominal_power": 3000.0,
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_room_thermal_loads(params, logger, emhass_conf)
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 2)
        self.assertEqual(len(optim_conf["def_load_config"]), 2)
        self.assertEqual(len(optim_conf["nominal_power_of_deferrable_loads"]), 2)
        self.assertEqual(params["passed_data"]["room_load_indices"], {"Living Room": 0})
        self.assertEqual(params["passed_data"]["heatpump_dispatch_load_index"], 1)

    async def test_append_room_thermal_loads_noop_without_configured_hardware(self):
        """set_use_heatpump alone shouldn't append phantom rooms/dispatch loads
        when no room name or dispatch entity is actually configured."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_heatpump": True,
                "heatpump_number_of_rooms": 0,
                "heatpump_dispatch_control_entity": "",
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 0)
        self.assertEqual(params["optim_conf"]["def_load_config"], [])

    @staticmethod
    def _base_room_params(**overrides):
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_heatpump": True,
                "heatpump_number_of_rooms": 1,
                "heatpump_room_names": ["Living Room"],
                "heatpump_room_min_temperature": [18.0],
                "heatpump_room_max_temperature": [24.0],
                "heatpump_room_target_temperature": [21.0],
                "heatpump_room_nominal_power": [1500.0],
                "heatpump_room_supply_temperature": [35.0],
                "heatpump_room_volume": [15.0],
                "heatpump_room_shared_group": [0],
                "heatpump_dispatch_control_entity": "",
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }
        params["optim_conf"].update(overrides)
        return params

    async def test_append_room_thermal_loads_simple_family_preserves_zero_list_default(self):
        """Regression guard: heatpump_model_family unset (or "simple") must
        keep today's exact thermal-loss-only behavior - no physics keys, a
        zero-list custom_heating_demand_profile."""
        params = self._base_room_params()
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["custom_heating_demand_profile"], [0.0] * 48)
        for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]:
            self.assertNotIn(key, room_cfg)

    async def test_append_room_thermal_loads_machine_learning_family_preserves_zero_list_default(self):
        """machine_learning/deep_learning are selectable but not yet wired to
        live dispatch - they must behave exactly like "simple" for now."""
        params = self._base_room_params(heatpump_model_family="machine_learning")
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["custom_heating_demand_profile"], [0.0] * 48)

    async def test_append_room_thermal_loads_physics_family_populates_envelope_fields(self):
        """heatpump_model_family="physics" must populate all 8 real envelope/
        RC-model keys atomically and omit custom_heating_demand_profile
        entirely, so optimization.py's physics branch (gated on all 4 core
        keys being present) actually runs instead of silently falling
        through."""
        params = self._base_room_params(
            heatpump_model_family="physics",
            heatpump_room_u_value=[0.4],
            heatpump_room_envelope_area=[50.0],
            heatpump_room_ventilation_rate=[0.6],
            heatpump_room_window_area=[8.0],
            heatpump_room_shgc=[0.55],
            heatpump_room_internal_gains_factor=[150.0],
            heatpump_room_thermal_inertia_time_constant=[3.0],
            heatpump_room_carnot_efficiency=[0.42],
            heatpump_room_blind_type=["screen"],
        )
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertNotIn("custom_heating_demand_profile", room_cfg)
        self.assertEqual(room_cfg["u_value"], 0.4)
        self.assertEqual(room_cfg["envelope_area"], 50.0)
        self.assertEqual(room_cfg["ventilation_rate"], 0.6)
        self.assertEqual(room_cfg["heated_volume"], 15.0)  # reuses heatpump_room_volume
        self.assertEqual(room_cfg["window_area"], 8.0)
        self.assertEqual(room_cfg["shgc"], 0.55)
        self.assertEqual(room_cfg["internal_gains_factor"], 150.0)
        self.assertEqual(room_cfg["thermal_inertia_time_constant"], 3.0)
        self.assertEqual(room_cfg["carnot_efficiency"], 0.42)
        self.assertEqual(room_cfg["blind_type"], "screen")

    async def test_append_room_thermal_loads_physics_family_blind_type_defaults_to_none(self):
        """heatpump_room_blind_type must default to 'none' (inert) when the
        user hasn't set it, matching this codebase's "new features default
        off" convention throughout."""
        params = self._base_room_params(
            heatpump_model_family="physics",
            heatpump_room_u_value=[0.4],
            heatpump_room_envelope_area=[50.0],
            heatpump_room_ventilation_rate=[0.6],
            heatpump_room_window_area=[8.0],
            heatpump_room_shgc=[0.55],
        )
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["blind_type"], "none")

    async def test_append_room_thermal_loads_graph_topology_mode_is_noop(self):
        """heatpump_config_mode="graph_topology" makes the room list inert -
        heating is configured as a heat_topology graph instead, and the two
        mechanisms must never both append loads into the same MILP."""
        params = self._base_room_params(heatpump_config_mode="graph_topology")
        await utils._append_room_thermal_loads(params, logger, emhass_conf)

        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 0)
        self.assertEqual(params["optim_conf"]["def_load_config"], [])

    @staticmethod
    def _two_room_coupling_params(**overrides):
        """Two rooms, room 0 manually coupled to room 1 (room-relative index
        1) at 0.05 kW/K - the minimum config needed to exercise the learned-
        coupling opt-in override path in _append_room_thermal_loads."""
        defaults = {
            "heatpump_number_of_rooms": 2,
            "heatpump_room_names": ["Living Room", "Bedroom"],
            "heatpump_room_min_temperature": [18.0, 18.0],
            "heatpump_room_max_temperature": [24.0, 24.0],
            "heatpump_room_target_temperature": [21.0, 21.0],
            "heatpump_room_nominal_power": [1500.0, 1500.0],
            "heatpump_room_supply_temperature": [35.0, 35.0],
            "heatpump_room_volume": [15.0, 15.0],
            "heatpump_room_shared_group": [0, 0],
            "heatpump_room_coupled_neighbors": ["1", ""],
            "heatpump_room_coupling_conductance": ["0.05", ""],
        }
        defaults.update(overrides)
        return TestUtils._base_room_params(**defaults)

    @staticmethod
    def _mock_load_json_blob_side_effect(coupling_response):
        """_append_room_thermal_loads also unconditionally loads
        room_thermal_schedule.json via the same load_json_blob helper - a
        blanket mock would make it impossible to tell "the coupling blob
        specifically was never requested" from "some other blob load was
        skipped". This routes only the coupling-blob filename to a caller-
        supplied response and everything else to its own `default`, exactly
        like the real load_json_blob does for a missing file."""

        async def _side_effect(_emhass_conf, filename, _logger, default=None):
            if filename == "self_learning_physics_coupling.json":
                return coupling_response
            return default

        return _side_effect

    async def test_room_coupling_informational_default_never_touches_conductance(self):
        """self_learning_physics_coupling_source defaults to 'informational'
        (absent entirely from optim_conf here, matching a config that
        predates this feature) - the learned-coupling blob must never even
        be read, and the manually-entered conductance must survive
        untouched, mirroring refit_hybrid_heatpump_model's own isolation
        from dispatch. The mocked response below (if it *were* read) would
        change the outcome to 0.09 - since it stays 0.05, the blob was
        genuinely never consulted, not just coincidentally ignored."""
        params = self._two_room_coupling_params()
        coupling_blob = {
            "pairs": [
                {"room_a": "Bedroom", "room_b": "Living Room", "conductance_kw_per_k": 0.09}
            ]
        }

        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_side_effect(coupling_blob))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        requested_filenames = [call.args[1] for call in mock_load.await_args_list]
        self.assertNotIn("self_learning_physics_coupling.json", requested_filenames)
        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["coupling_conductance_kw_per_k"], [0.05])

    async def test_room_coupling_auto_dispatch_overrides_declared_pair(self):
        """With the explicit opt-in, a learned coefficient for an already-
        manually-declared pair overrides the manual value for that pair -
        the room-name pair keying must be order-independent (room_a/room_b
        sorted, same as the config's own room_a < room_b convention)."""
        params = self._two_room_coupling_params(
            self_learning_physics_coupling_source="auto_dispatch"
        )
        coupling_blob = {
            "pairs": [
                {"room_a": "Bedroom", "room_b": "Living Room", "conductance_kw_per_k": 0.09}
            ]
        }

        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_side_effect(coupling_blob))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        requested_filenames = [call.args[1] for call in mock_load.await_args_list]
        self.assertIn("self_learning_physics_coupling.json", requested_filenames)
        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["coupling_conductance_kw_per_k"], [0.09])

    async def test_room_coupling_auto_dispatch_ignores_pair_without_manual_declaration(self):
        """A learned coefficient can only ever override an already-declared
        (positive-conductance) manual pair - it must never create
        dispatch-affecting coupling for a pair the user never entered
        manually in the first place, even under the auto_dispatch opt-in."""
        params = self._two_room_coupling_params(
            heatpump_room_coupled_neighbors=["", ""],
            heatpump_room_coupling_conductance=["", ""],
            self_learning_physics_coupling_source="auto_dispatch",
        )
        coupling_blob = {
            "pairs": [
                {"room_a": "Bedroom", "room_b": "Living Room", "conductance_kw_per_k": 0.09}
            ]
        }

        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_side_effect(coupling_blob))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["coupled_neighbors"], [])
        self.assertEqual(room_cfg["coupling_conductance_kw_per_k"], [])

    @staticmethod
    def _mock_load_json_blob_routing(responses: dict[str, dict]):
        """Generalizes _mock_load_json_blob_side_effect to route several
        filenames at once (self_learning_physics_coupling.json AND
        self_learning_physics_room_dispatch_coefficients.json can both be
        loaded in the same _append_room_thermal_loads call) - anything not
        in `responses` falls through to the real load_json_blob's own
        `default`, same as the single-filename helper above."""

        async def _side_effect(_emhass_conf, filename, _logger, default=None):
            if filename in responses:
                return responses[filename]
            return default

        return _side_effect

    async def test_self_learning_dispatch_loads_and_translates_coefficients(self):
        """heatpump_room_self_learning_only=True for a room with a matching
        entry in the dispatch-coefficients artifact must attach
        self_learning_dispatch to that room's thermal_battery config, with
        any neighbor_diff::<name> feature's name resolved to the neighbor's
        current absolute def_load_config index (not left as a bare name)."""
        params = self._two_room_coupling_params(
            heatpump_room_self_learning_only=[True, False]
        )
        dispatch_blob = {
            "rooms": {
                "Living Room": {
                    "feature_names": ["bias", "room_last", "duty", "neighbor_diff::Bedroom"],
                    "theta": [15.0, 0.9, 4.0, 0.2],
                    "neighbors": ["Bedroom"],
                }
            }
        }

        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": dispatch_blob}
            )
        )
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        sl = room_cfg["self_learning_dispatch"]
        self.assertEqual(
            sl["feature_names"], ["bias", "room_last", "duty", "neighbor_diff::Bedroom"]
        )
        self.assertEqual(sl["theta"], [15.0, 0.9, 4.0, 0.2])
        # Bedroom is room-relative index 1, room_index_base is 0 for the
        # first room appended (num_def_loads starts at 0 in this params
        # dict) - so absolute index 1.
        self.assertEqual(sl["neighbor_indices"], {"Bedroom": 1})
        # Room 1 (Bedroom) isn't flagged - must never get a self_learning_dispatch key.
        room_1_cfg = params["optim_conf"]["def_load_config"][1]["thermal_battery"]
        self.assertNotIn("self_learning_dispatch", room_1_cfg)

    async def test_self_learning_dispatch_missing_artifact_warns_and_falls_back(self):
        """Flag set but no fitted model covers this room yet (no refit run,
        or the artifact simply doesn't mention this room's name) - must
        warn and leave self_learning_dispatch unset, never crash."""
        params = self._two_room_coupling_params(
            heatpump_room_self_learning_only=[True, False]
        )
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": {"rooms": {}}}
            )
        )
        with patch("emhass.utils.load_json_blob", mock_load), self.assertLogs(
            logger, level="WARNING"
        ) as log_ctx:
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertNotIn("self_learning_dispatch", room_cfg)
        self.assertTrue(
            any("no fitted" in msg.lower() for msg in log_ctx.output),
            log_ctx.output,
        )

    async def test_self_learning_dispatch_stale_neighbor_dropped_rest_kept(self):
        """A fitted model referencing a neighbor room that's no longer
        configured must drop only that one neighbor_diff feature, keeping
        every other coefficient (dropping one column of a linear model
        doesn't invalidate the others)."""
        params = self._two_room_coupling_params(
            heatpump_room_self_learning_only=[True, False]
        )
        dispatch_blob = {
            "rooms": {
                "Living Room": {
                    "feature_names": ["bias", "duty", "neighbor_diff::Attic"],
                    "theta": [15.0, 4.0, 0.2],
                    "neighbors": ["Attic"],
                }
            }
        }
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"self_learning_physics_room_dispatch_coefficients.json": dispatch_blob}
            )
        )
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        sl = room_cfg["self_learning_dispatch"]
        self.assertEqual(sl["feature_names"], ["bias", "duty"])
        self.assertEqual(sl["theta"], [15.0, 4.0])
        self.assertEqual(sl["neighbor_indices"], {})

    async def test_heatpump_group_member_stamped_on_every_room_and_dispatch_load(self):
        """heatpump_group_member (consumed by
        optimization.py::_build_aggregate_heatpump_duty_expr) must be True
        on every room's thermal_battery config AND the whole-house dispatch
        load, regardless of whether any room is self-learning-flagged - the
        aggregate duty signal is a property of the shared physical heat
        pump, not of this feature."""
        params = self._two_room_coupling_params(
            heatpump_dispatch_control_entity="switch.climate_control"
        )
        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_routing({}))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        def_load_config = params["optim_conf"]["def_load_config"]
        self.assertEqual(len(def_load_config), 3)  # 2 rooms + 1 dispatch load
        for cfg in def_load_config:
            self.assertTrue(cfg["thermal_battery"]["heatpump_group_member"])

    async def test_self_learning_dispatch_artifact_never_loaded_when_no_room_flagged(self):
        """No room flagged at all - the dispatch-coefficients artifact must
        never even be requested (same zero-cost-when-unused guarantee as
        the learned-coupling blob)."""
        params = self._two_room_coupling_params()
        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_routing({}))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        requested_filenames = [call.args[1] for call in mock_load.await_args_list]
        self.assertNotIn(
            "self_learning_physics_room_dispatch_coefficients.json", requested_filenames
        )

    async def test_rc_physics_dispatch_loads_fitted_params(self):
        """heatpump_room_rc_physics_only=True for a room, with a valid
        thermal_physics_params.json artifact present, must attach
        rc_physics_dispatch (a plain copy of the artifact's own "params"
        dict, house-wide - not per-room) to that room's thermal_battery
        config."""
        params = self._two_room_coupling_params(
            heatpump_room_rc_physics_only=[True, False]
        )
        rc_blob = {"params": {"tau_emit_h": 2.5, "bias_c_per_h": 0.1, "mass_tau_h": 48.0}}
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {"thermal_physics_params.json": rc_blob}
            )
        )
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(room_cfg["rc_physics_dispatch"]["params"], rc_blob["params"])
        # Room 1 isn't flagged - must never get an rc_physics_dispatch key.
        room_1_cfg = params["optim_conf"]["def_load_config"][1]["thermal_battery"]
        self.assertNotIn("rc_physics_dispatch", room_1_cfg)

    async def test_rc_physics_dispatch_missing_artifact_warns_and_falls_back(self):
        """Flag set but no fitted RC model exists yet (no refit/tune run) -
        must warn and leave rc_physics_dispatch unset, never crash."""
        params = self._two_room_coupling_params(
            heatpump_room_rc_physics_only=[True, False]
        )
        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_routing({}))
        with patch("emhass.utils.load_json_blob", mock_load), self.assertLogs(
            logger, level="WARNING"
        ) as log_ctx:
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertNotIn("rc_physics_dispatch", room_cfg)
        self.assertTrue(
            any("no fitted rc-physics model" in msg.lower() for msg in log_ctx.output),
            log_ctx.output,
        )

    async def test_rc_physics_dispatch_self_learning_priority_when_both_flagged(self):
        """A room with BOTH heatpump_room_self_learning_only and
        heatpump_room_rc_physics_only set must dispatch via self-learning
        only - RC's own artifact must never even be attached, and a warning
        must explain why."""
        params = self._two_room_coupling_params(
            heatpump_room_self_learning_only=[True, False],
            heatpump_room_rc_physics_only=[True, False],
        )
        dispatch_blob = {
            "rooms": {
                "Living Room": {
                    "feature_names": ["bias", "room_last", "duty"],
                    "theta": [15.0, 0.9, 4.0],
                }
            }
        }
        rc_blob = {"params": {"tau_emit_h": 2.5}}
        mock_load = AsyncMock(
            side_effect=self._mock_load_json_blob_routing(
                {
                    "self_learning_physics_room_dispatch_coefficients.json": dispatch_blob,
                    "thermal_physics_params.json": rc_blob,
                }
            )
        )
        with patch("emhass.utils.load_json_blob", mock_load), self.assertLogs(
            logger, level="WARNING"
        ) as log_ctx:
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        room_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertIn("self_learning_dispatch", room_cfg)
        self.assertNotIn("rc_physics_dispatch", room_cfg)
        self.assertTrue(
            any("takes priority" in msg.lower() for msg in log_ctx.output),
            log_ctx.output,
        )

    async def test_rc_physics_dispatch_artifact_never_loaded_when_no_room_flagged(self):
        """No room flagged at all - thermal_physics_params.json must never
        even be requested (same zero-cost-when-unused guarantee as the
        self-learning dispatch-coefficients blob)."""
        params = self._two_room_coupling_params()
        mock_load = AsyncMock(side_effect=self._mock_load_json_blob_routing({}))
        with patch("emhass.utils.load_json_blob", mock_load):
            await utils._append_room_thermal_loads(params, logger, emhass_conf)

        requested_filenames = [call.args[1] for call in mock_load.await_args_list]
        self.assertNotIn("thermal_physics_params.json", requested_filenames)

    async def test_append_boiler_thermal_battery_loads_resistive_uses_flat_efficiency(self):
        """resolve_thermal_battery_cop only takes the flat constant-efficiency
        branch when "efficiency" is present in hc - a resistive boiler set to
        "carnot_efficiency" instead falls into the heat-pump Carnot-lift
        formula and computes a COP well above 1.0, which is physically wrong
        for resistive heating."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_boiler": True,
                "number_of_boilers": 1,
                "boiler_names": ["dhw_tank"],
                "boiler_type": ["resistive"],
                "boiler_legionella_interval_days": [7],
                "boiler_legionella_last_run_iso": [pd.Timestamp.now(tz="UTC").isoformat()],
                "boiler_legionella_force_resistive": [False],
                "delta_forecast_daily": pd.to_timedelta(1, "days"),
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_boiler_thermal_battery_loads(params, logger, emhass_conf)

        thermal_cfg = params["optim_conf"]["def_load_config"][0]["thermal_battery"]
        self.assertEqual(thermal_cfg["boiler_type"], "resistive")
        self.assertEqual(thermal_cfg.get("efficiency"), 1.0)
        self.assertNotIn("carnot_efficiency", thermal_cfg)
        cop = utils.resolve_thermal_battery_cop(thermal_cfg, None, length=4)
        self.assertTrue((cop == 1.0).all())

    def test_append_heating_forecast_targets_registers_entities_when_enabled(self):
        params = {"optim_conf": {"heating_forecast_enabled": True}}

        utils._append_heating_forecast_targets(params, logger)

        passed_data = params["passed_data"]
        self.assertEqual(
            passed_data["custom_indoor_temp_forecast_id"]["entity_id"],
            "sensor.indoor_temp_forecast",
        )
        self.assertEqual(
            passed_data["custom_heating_needed_by_id"]["entity_id"],
            "sensor.heating_needed_by",
        )

    def test_append_heating_forecast_targets_noop_when_disabled(self):
        params = {"optim_conf": {"heating_forecast_enabled": False}}

        utils._append_heating_forecast_targets(params, logger)

        self.assertNotIn("passed_data", params)
        self.assertNotIn("passed_data", params)

    @staticmethod
    def _make_week_schedule(day_room_map):
        """Build a weekSchedule dict where each slot's temp_min encodes its
        own slot index, for easy assertion of which slot got looked up."""
        schedule = {}
        for day, room in day_room_map:
            schedule.setdefault(day, {})[room] = [
                {"slot": s, "temp_min": float(s), "temp_max": float(s) + 1.0} for s in range(48)
            ]
        return schedule

    def test_flatten_room_schedule_midweek(self):
        """A plain midweek lookup should return the exact slot's band."""
        week_schedule = self._make_week_schedule([("Wednesday", "Living Room")])
        # Wednesday 10:00 -> slot 20; Wednesday 10:30 -> slot 21.
        start = pd.Timestamp("2026-01-07 10:00:00", tz="UTC")  # a Wednesday
        min_temps, max_temps = utils.flatten_room_schedule(
            week_schedule, "Living Room", start, pd.to_timedelta(30, "min"), 2
        )
        self.assertEqual(min_temps, [20.0, 21.0])
        self.assertEqual(max_temps, [21.0, 22.0])

    def test_flatten_room_schedule_midnight_crossing(self):
        """Horizon spanning midnight within the same week should switch days
        automatically, since each step derives its own day/slot."""
        week_schedule = self._make_week_schedule(
            [("Saturday", "Living Room"), ("Sunday", "Living Room")]
        )
        # Saturday 23:45 -> Saturday slot 47; +30min -> Sunday slot 0; +30min -> Sunday slot 1.
        start = pd.Timestamp("2026-01-03 23:45:00", tz="UTC")  # a Saturday
        min_temps, _ = utils.flatten_room_schedule(
            week_schedule, "Living Room", start, pd.to_timedelta(30, "min"), 3
        )
        self.assertEqual(min_temps, [47.0, 0.0, 1.0])

    def test_flatten_room_schedule_week_boundary(self):
        """Horizon spanning Sunday -> Monday should pick up the new week's Monday schedule."""
        week_schedule = self._make_week_schedule(
            [("Sunday", "Living Room"), ("Monday", "Living Room")]
        )
        start = pd.Timestamp("2026-01-04 23:45:00", tz="UTC")  # a Sunday
        min_temps, _ = utils.flatten_room_schedule(
            week_schedule, "Living Room", start, pd.to_timedelta(30, "min"), 3
        )
        self.assertEqual(min_temps, [47.0, 0.0, 1.0])

    def test_flatten_room_schedule_missing_falls_back_to_default(self):
        """A room/day with no saved schedule should fall back to the static default."""
        min_temps, max_temps = utils.flatten_room_schedule(
            {}, "Living Room", pd.Timestamp.now(tz="UTC"), pd.to_timedelta(30, "min"), 4,
            default_min=17.5, default_max=23.5,
        )
        self.assertEqual(min_temps, [17.5] * 4)
        self.assertEqual(max_temps, [23.5] * 4)

    async def test_append_ev_deferrable_loads_sets_semi_cont_bounds(self):
        """A configured EV charger should become a semi-continuous deferrable
        load bounded by its real min/max power, with target-sensor bookkeeping
        for the publish stage."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_ev_charger": True,
                "number_of_ev_chargers": 1,
                "ev_charger_names": ["Zappi"],
                "ev_charge_power_min_1_phase": [1380.0],
                "ev_charge_power_max_3_phase": [11000.0],
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_ev_deferrable_loads(params, logger)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 1)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [11000.0])
        self.assertEqual(optim_conf["minimum_power_of_deferrable_loads"], [1380.0])
        self.assertTrue(optim_conf["treat_deferrable_load_as_semi_cont"][0])
        self.assertEqual(params["passed_data"]["ev_load_indices"], {"Zappi": 0})
        self.assertEqual(
            params["passed_data"]["custom_ev_charge_mode_target_id"][0]["entity_id"],
            "sensor.ev_charge_mode_target_zappi",
        )

    async def test_append_ev_deferrable_loads_idempotent_across_repeated_calls(self):
        """Same regression as the room/boiler siblings - calling this
        function twice on the same optim_conf (the config page's
        GET->edit->POST round trip) must not double the EV charger entry."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_ev_charger": True,
                "number_of_ev_chargers": 1,
                "ev_charger_names": ["Zappi"],
                "ev_charge_power_min_1_phase": [1380.0],
                "ev_charge_power_max_3_phase": [11000.0],
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }

        await utils._append_ev_deferrable_loads(params, logger)
        await utils._append_ev_deferrable_loads(params, logger)

        optim_conf = params["optim_conf"]
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 1)
        self.assertEqual(len(optim_conf["def_load_config"]), 1)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [11000.0])
        self.assertEqual(params["passed_data"]["ev_load_indices"], {"Zappi": 0})

    async def test_append_ev_deferrable_loads_disabling_cleans_up_stale_entry(self):
        """Turning set_use_ev_charger off must remove the previously-
        appended charger entry, not leave it orphaned forever."""
        params = {
            "retrieve_hass_conf": {"optimization_time_step": pd.to_timedelta(30, "min")},
            "optim_conf": {
                "set_use_ev_charger": True,
                "number_of_ev_chargers": 1,
                "ev_charger_names": ["Zappi"],
                "number_of_deferrable_loads": 0,
                "nominal_power_of_deferrable_loads": [],
                "minimum_power_of_deferrable_loads": [],
                "operating_hours_of_each_deferrable_load": [],
                "start_timesteps_of_each_deferrable_load": [],
                "end_timesteps_of_each_deferrable_load": [],
                "set_deferrable_startup_penalty": [],
                "set_deferrable_load_single_constant": [],
                "treat_deferrable_load_as_semi_cont": [],
                "load_type": [],
                "load_dispatch_mode": [],
                "required_energy_kwh_of_each_deferrable_load": [],
                "def_load_config": [],
            },
        }
        await utils._append_ev_deferrable_loads(params, logger)
        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 1)

        params["optim_conf"]["set_use_ev_charger"] = False
        await utils._append_ev_deferrable_loads(params, logger)

        self.assertEqual(params["optim_conf"]["number_of_deferrable_loads"], 0)
        self.assertEqual(params["optim_conf"]["def_load_config"], [])

    async def test_resolve_manual_committed_loads_flags_existing_slot(self):
        """Marking an *existing* deferrable load as manual (is_manual_load)
        reuses that load's own name/nominal power/operating hours - no
        separate slot is appended and number_of_deferrable_loads is
        unchanged. It's forced to single-constant semi-continuous so a
        pinned commitment (decided live, per-cycle, by
        command_line._apply_manual_load_runtime_overrides) stays one
        contiguous block."""
        params = {
            "retrieve_hass_conf": {
                "optimization_time_step": pd.to_timedelta(30, "min"),
                "manual_load_ready_sensor": ["input_boolean.dishwasher_ready"],
                "manual_load_confirm_power_sensor": ["sensor.dishwasher_power"],
            },
            "optim_conf": {
                "manual_load_enabled": True,
                "is_manual_load": [True],
                "manual_load_deadline_hour": ["22:00"],
                "number_of_deferrable_loads": 1,
                "load_names": ["Dishwasher"],
                "nominal_power_of_deferrable_loads": [1800.0],
                "operating_hours_of_each_deferrable_load": [2.5],
                "minimum_power_of_deferrable_loads": [0.0],
                "start_timesteps_of_each_deferrable_load": [0],
                "end_timesteps_of_each_deferrable_load": [0],
                "set_deferrable_startup_penalty": [0.0],
                "set_deferrable_load_single_constant": [False],
                "treat_deferrable_load_as_semi_cont": [False],
                "load_type": ["fixed_power_non_splittable"],
                "load_dispatch_mode": ["hours"],
                "required_energy_kwh_of_each_deferrable_load": [0.0],
            },
        }

        await utils._resolve_manual_committed_loads(params, logger)

        optim_conf = params["optim_conf"]
        # No slot appended - the load stays at its configured index/count.
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 1)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [1800.0])
        self.assertTrue(optim_conf["treat_deferrable_load_as_semi_cont"][0])
        self.assertTrue(optim_conf["set_deferrable_load_single_constant"][0])
        load_info = params["passed_data"]["manual_load_indices"]["Dishwasher"]
        self.assertEqual(load_info["k"], 0)
        self.assertEqual(load_info["ready_sensor"], "input_boolean.dishwasher_ready")
        self.assertEqual(load_info["confirm_power_sensor"], "sensor.dishwasher_power")
        self.assertEqual(load_info["nominal_power"], 1800.0)
        self.assertEqual(load_info["duration_hours"], 2.5)
        self.assertEqual(load_info["deadline_hour"], "22:00")
        self.assertEqual(
            params["passed_data"]["custom_manual_load_action_id"][0]["entity_id"],
            "sensor.manual_load_action_dishwasher",
        )
        # The base load's own forecast sensor is registered elsewhere
        # (treat_runtimeparams), never here - no duplicate entry.
        self.assertNotIn("custom_deferrable_forecast_id", params["passed_data"])

    async def test_resolve_manual_committed_loads_noop_when_disabled(self):
        params = {
            "retrieve_hass_conf": {},
            "optim_conf": {"manual_load_enabled": False},
        }
        await utils._resolve_manual_committed_loads(params, logger)
        self.assertNotIn("passed_data", params)

    async def test_resolve_manual_committed_loads_noop_when_none_flagged(self):
        params = {
            "retrieve_hass_conf": {},
            "optim_conf": {
                "manual_load_enabled": True,
                "is_manual_load": [False, False],
                "load_names": ["dishwasher", "washing_machine"],
                "number_of_deferrable_loads": 2,
            },
        }
        await utils._resolve_manual_committed_loads(params, logger)
        self.assertNotIn("passed_data", params)

    async def test_resolve_manual_committed_loads_handles_sequence_nominal_power(self):
        """A manual load can also be program_based - once
        _normalize_deferrable_load_categories resolves load_programs into a
        sequence, nominal_power_of_deferrable_loads[k] is a list rather than
        a flat scalar. Must not crash, and should fall back to the
        sequence's peak as a rough confirm-power-sensor threshold."""
        params = {
            "retrieve_hass_conf": {},
            "optim_conf": {
                "manual_load_enabled": True,
                "is_manual_load": [True],
                "load_names": ["Dishwasher"],
                "number_of_deferrable_loads": 1,
                "nominal_power_of_deferrable_loads": [[300.0, 900.0, 150.0]],
                "operating_hours_of_each_deferrable_load": [3],
            },
        }
        await utils._resolve_manual_committed_loads(params, logger)
        load_info = params["passed_data"]["manual_load_indices"]["Dishwasher"]
        self.assertEqual(load_info["nominal_power"], 900.0)

    def test_resample_power_profile_downsample_exact_multiple(self):
        """15min -> 30min, exact multiple: plain pairwise average."""
        profile = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        result = utils._resample_power_profile(profile, 15.0, 30.0)
        self.assertEqual(result, [150.0, 350.0, 550.0])

    def test_resample_power_profile_downsample_with_singleton_tail(self):
        """15min -> 30min on an odd-length (9-element) profile: the last
        30min bin only has one 15min source block behind it."""
        profile = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        result = utils._resample_power_profile(profile, 15.0, 30.0)
        self.assertEqual(result, [15.0, 35.0, 55.0, 75.0, 90.0])

    def test_resample_power_profile_upsample(self):
        """15min -> 5min: each source value repeated across its 3 sub-bins."""
        profile = [100.0, 200.0, 300.0]
        result = utils._resample_power_profile(profile, 15.0, 5.0)
        self.assertEqual(
            result, [100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 300.0, 300.0, 300.0]
        )

    def test_resample_power_profile_equal_resolution_passthrough(self):
        profile = [1.0, 2.0, 3.0]
        result = utils._resample_power_profile(profile, 30.0, 30.0)
        self.assertEqual(result, profile)

    def test_resample_power_profile_single_element_passthrough(self):
        result = utils._resample_power_profile([42.0], 15.0, 30.0)
        self.assertEqual(result, [42.0])

    def test_resample_power_profile_empty_passthrough(self):
        self.assertEqual(utils._resample_power_profile([], 15.0, 30.0), [])

    def test_resample_power_profile_degenerate_interval_passthrough(self):
        profile = [1.0, 2.0, 3.0]
        self.assertEqual(utils._resample_power_profile(profile, 0.0, 30.0), profile)
        self.assertEqual(utils._resample_power_profile(profile, 15.0, 0.0), profile)

    async def test_save_load_json_blob_roundtrip(self):
        """save_json_blob followed by load_json_blob should return the same data."""
        import tempfile

        from emhass.persistence import load_json_blob, save_json_blob

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_emhass_conf = {"data_path": pathlib.Path(tmp_dir)}
            data = {"weekSchedule": {"Monday": {"Living Room": [1, 2, 3]}}, "n": 42}

            ok = await save_json_blob(tmp_emhass_conf, "test_blob.json", data, logger)
            self.assertTrue(ok)

            loaded = await load_json_blob(tmp_emhass_conf, "test_blob.json", logger)
            self.assertEqual(loaded, data)

    async def test_load_json_blob_missing_file_returns_default(self):
        """load_json_blob should return the default (never raise) when the file is absent."""
        import tempfile

        from emhass.persistence import load_json_blob

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_emhass_conf = {"data_path": pathlib.Path(tmp_dir)}

            loaded = await load_json_blob(
                tmp_emhass_conf, "does_not_exist.json", logger, default={"a": 1}
            )
            self.assertEqual(loaded, {"a": 1})

            loaded_default = await load_json_blob(tmp_emhass_conf, "does_not_exist.json", logger)
            self.assertEqual(loaded_default, {})

    def test_parse_export_time_range(self):
        """Test timestamp parsing and validation for data exports."""
        time_zone = pytz.timezone("Europe/Paris")
        # Case 1: Valid start and end times
        start_time = "2024-01-01 00:00:00"
        end_time = "2024-01-02 00:00:00"
        start_dt, end_dt = utils.parse_export_time_range(start_time, end_time, time_zone, logger)
        self.assertIsInstance(start_dt, pd.Timestamp)
        self.assertIsInstance(end_dt, pd.Timestamp)
        self.assertEqual(str(start_dt.tz), "Europe/Paris")
        # Case 2: Missing end time (should default to now)
        start_dt, end_dt = utils.parse_export_time_range(start_time, None, time_zone, logger)
        self.assertIsInstance(start_dt, pd.Timestamp)
        self.assertIsInstance(end_dt, pd.Timestamp)
        self.assertAlmostEqual(
            end_dt.timestamp(), pd.Timestamp.now(tz=time_zone).timestamp(), delta=5.0
        )
        # Case 3: Invalid start time
        start_dt, end_dt = utils.parse_export_time_range(
            "invalid-date", end_time, time_zone, logger
        )
        self.assertFalse(start_dt)
        self.assertFalse(end_dt)
        # Case 4: Invalid end time
        start_dt, end_dt = utils.parse_export_time_range(
            start_time, "invalid-date", time_zone, logger
        )
        self.assertFalse(start_dt)
        self.assertFalse(end_dt)

    def test_handle_nan_values(self):
        """Test NaN handling strategies for dataframes."""
        # Create a test dataframe with NaNs
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="1h"),
                "value1": [1.0, np.nan, np.nan, 4.0],
                "value2": [10.0, 20.0, np.nan, 40.0],
            }
        )
        # Case 1: Drop
        df_drop = utils.handle_nan_values(df.copy(), "drop", "timestamp", logger)
        self.assertEqual(len(df_drop), 2)
        # Case 2: Fill Zero
        df_zero = utils.handle_nan_values(df.copy(), "fill_zero", "timestamp", logger)
        self.assertEqual(df_zero["value1"].iloc[1], 0.0)
        # Case 3: Interpolate
        df_interp = utils.handle_nan_values(df.copy(), "interpolate", "timestamp", logger)
        self.assertEqual(df_interp["value1"].iloc[1], 2.0)
        self.assertEqual(df_interp["value1"].iloc[2], 3.0)
        # Case 4: Forward Fill
        df_ffill = utils.handle_nan_values(df.copy(), "forward_fill", "timestamp", logger)
        self.assertEqual(df_ffill["value1"].iloc[1], 1.0)
        self.assertEqual(df_ffill["value1"].iloc[2], 1.0)
        # Case 5: Backward Fill
        df_bfill = utils.handle_nan_values(df.copy(), "backward_fill", "timestamp", logger)
        self.assertEqual(df_bfill["value1"].iloc[1], 4.0)
        self.assertEqual(df_bfill["value1"].iloc[2], 4.0)
        # Case 6: No NaNs present (should return immediately)
        df_clean = df.dropna()
        df_result = utils.handle_nan_values(df_clean, "drop", "timestamp", logger)
        self.assertEqual(len(df_result), 2)

    async def test_naive_mpc_horizon_extends_forecast_window(self):
        """RED regression test: naive-mpc with prediction_horizon > default window must
        expand delta_forecast_daily to cover the full horizon.

        Bug: forecast_dates is built from delta_forecast_daily (config default = 1 day = 48
        steps at 30 min) BEFORE prediction_horizon is parsed.  The subsequent slice
          forecast_dates = copy.deepcopy(forecast_dates)[0:prediction_horizon]
        is a no-op when prediction_horizon > len(forecast_dates), silently leaving the
        window at 1 day with no warning.

        Fix (Phase 1): once prediction_horizon is known inside the naive-mpc-optim branch,
        if it needs more steps than delta_forecast provides, raise delta_forecast to
        ceil(prediction_horizon * optimization_time_step_minutes / 1440) and update
        params["optim_conf"]["delta_forecast_daily"].

        Observable: returned optim_conf["delta_forecast_daily"].days
        """
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        # --- PRIMARY assertion (RED today) ---
        # prediction_horizon=72 steps @ 30 min = 36 h = 2 days
        # No delta_forecast_daily in runtimeparams → config default of 1 day
        # After the fix, delta_forecast_daily must be extended to 2 days.
        runtimeparams_wide = {
            "prediction_horizon": 72,
            "optimization_time_step": 30,
        }
        runtimeparams_wide_json = orjson.dumps(runtimeparams_wide).decode("utf-8")

        _, _, optim_conf_wide, _ = await treat_runtimeparams(
            runtimeparams_wide_json,
            params_json,
            retrieve_hass_conf.copy(),
            optim_conf.copy(),
            plant_conf.copy(),
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )

        actual_days_wide = optim_conf_wide["delta_forecast_daily"].days
        self.assertEqual(
            actual_days_wide,
            2,
            f"prediction_horizon=72 @ 30 min = 36 h requires delta_forecast_daily=2 days, "
            f"but got {actual_days_wide} day(s). "
            f"This is the silent-truncation bug: forecast_dates is built before "
            f"prediction_horizon is parsed, so the horizon is never extended.",
        )

    async def test_naive_mpc_horizon_unchanged_when_within_one_day(self):
        """Backwards-compat guard (must PASS before AND after the Phase 1 fix):
        a naive-mpc prediction_horizon that fits inside the default 1-day window
        must leave delta_forecast_daily untouched (no spurious extension).

        Kept as its own test (not appended to the RED extend-test) so it always
        runs: assertEqual short-circuits, so a guard sharing a method with a RED
        assertion would never execute during the RED phase.
        """
        params = await TestUtils.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

        # prediction_horizon=24 steps @ 30 min = 12 h < 1 day -> no change needed
        runtimeparams_narrow = {
            "prediction_horizon": 24,
            "optimization_time_step": 30,
        }
        runtimeparams_narrow_json = orjson.dumps(runtimeparams_narrow).decode("utf-8")

        _, _, optim_conf_narrow, _ = await treat_runtimeparams(
            runtimeparams_narrow_json,
            params_json,
            retrieve_hass_conf.copy(),
            optim_conf.copy(),
            plant_conf.copy(),
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )

        actual_days_narrow = optim_conf_narrow["delta_forecast_daily"].days
        self.assertEqual(
            actual_days_narrow,
            1,
            f"prediction_horizon=24 @ 30 min = 12 h fits within 1 day; "
            f"delta_forecast_daily must remain 1, but got {actual_days_narrow}.",
        )

    def test_resample_and_filter_data(self):
        """Test time range filtering and data resampling."""
        time_zone = pytz.timezone("Europe/Paris")

        # Create a dummy 5-minute dataset
        idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz=time_zone)
        df = pd.DataFrame({"value": range(100)}, index=idx)

        start_dt = idx[10]  # 2024-01-01 00:50:00
        end_dt = idx[60]  # 2024-01-01 05:00:00

        # Case 1: Valid resample from 5min to 30min
        df_resampled = utils.resample_and_filter_data(df, start_dt, end_dt, "30min", logger)
        self.assertIsInstance(df_resampled, pd.DataFrame)
        # Verify index frequency is correctly applied
        self.assertEqual(df_resampled.index.freq.freqstr, "30min")
        # Verify filtering worked
        self.assertEqual(df_resampled.index[0], start_dt.floor("30min"))

        # Case 2: Invalid index type
        df_invalid_index = df.reset_index()
        res = utils.resample_and_filter_data(df_invalid_index, start_dt, end_dt, "30min", logger)
        self.assertFalse(res)

        # Case 3: Empty after filtering
        future_start = pd.Timestamp("2025-01-01", tz=time_zone)
        future_end = pd.Timestamp("2025-01-02", tz=time_zone)
        res = utils.resample_and_filter_data(df, future_start, future_end, "30min", logger)
        self.assertFalse(res)

        # Case 4: Naive timezone handling (should auto-localize)
        df_naive = pd.DataFrame(
            {"value": range(100)}, index=pd.date_range("2024-01-01", periods=100, freq="5min")
        )
        # Using a slice that overlaps with the naive index dates
        res = utils.resample_and_filter_data(df_naive, start_dt, end_dt, "30min", logger)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(res.index.tz, time_zone)


class TestHeatingDemand(unittest.TestCase):
    def test_calculate_heating_demand_basic(self):
        """Test heating demand calculation with basic parameters."""
        specific_heating_demand = 100.0  # kWh/m²/year
        floor_area = 150.0  # m²
        # Outdoor temps: cold weather requiring heating
        outdoor_temps = np.array([5.0, 10.0, 15.0, 8.0, 12.0, 6.0, 9.0, 11.0, 7.0, 13.0])
        base_temperature = 18.0
        annual_reference_hdd = 3000.0
        optimization_time_step = 30  # minutes

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            annual_reference_hdd,
            optimization_time_step,
        )

        # Verify output is numpy array
        self.assertIsInstance(heating_demand, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(heating_demand), len(outdoor_temps))
        # Verify all values are non-negative
        self.assertTrue(np.all(heating_demand >= 0.0))

        # Manual verification for first timestep: outdoor_temp = 5°C
        # HDD = max(18 - 5, 0) = 13 degree-days
        # HDD scaled to 30 min = 13 * (0.5 / 24) = 0.270833
        # heating_demand = 100 * 150 * (0.270833 / 3000) = 1.354 kWh
        hdd_first = max(base_temperature - outdoor_temps[0], 0.0)
        hours_per_timestep = optimization_time_step / 60.0
        hdd_scaled = hdd_first * (hours_per_timestep / 24.0)
        expected_demand = specific_heating_demand * floor_area * (hdd_scaled / annual_reference_hdd)
        self.assertAlmostEqual(heating_demand[0], expected_demand, places=6)

    def test_calculate_heating_demand_no_heating_needed(self):
        """Test heating demand when outdoor temp exceeds base temperature."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        # Summer temperatures - all above base temperature
        outdoor_temps = np.array([20.0, 25.0, 22.0, 24.0, 28.0])
        base_temperature = 18.0

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, base_temperature
        )

        # All heating demand should be zero when outdoor temp >= base temp
        self.assertTrue(np.allclose(heating_demand, 0.0))

    def test_calculate_heating_demand_pandas_series(self):
        """Test heating demand with pandas Series input."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_array = np.array([5.0, 10.0, 15.0, 8.0, 12.0])
        outdoor_temps_series = pd.Series(outdoor_temps_array)

        heating_demand_array = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array
        )
        heating_demand_series = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_series
        )

        # Results should be identical regardless of input type
        np.testing.assert_array_almost_equal(heating_demand_array, heating_demand_series)

    def test_calculate_heating_demand_different_timestep(self):
        """Test heating demand with different optimization time steps."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps = np.array([10.0, 12.0, 8.0])
        base_temperature = 18.0

        # Compare 30-minute vs 60-minute timesteps
        demand_30min = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            optimization_time_step=30,
        )
        demand_60min = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            optimization_time_step=60,
        )

        # 60-minute timestep should have exactly double the demand of 30-minute
        np.testing.assert_array_almost_equal(demand_60min, demand_30min * 2.0)

    def test_calculate_heating_demand_different_reference_hdd(self):
        """Test heating demand with different annual reference HDD values."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps = np.array([5.0, 10.0, 15.0])

        # Compare different reference HDD values
        demand_hdd_3000 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, annual_reference_hdd=3000.0
        )
        demand_hdd_1500 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, annual_reference_hdd=1500.0
        )

        # Half the reference HDD should double the heating demand
        np.testing.assert_array_almost_equal(demand_hdd_1500, demand_hdd_3000 * 2.0)

    def test_calculate_heating_demand_at_base_temperature(self):
        """Test heating demand exactly at base temperature (boundary condition)."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        # Outdoor temp exactly at base temperature
        outdoor_temps = np.array([18.0, 18.0, 18.0])
        base_temperature = 18.0

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, base_temperature
        )

        # At base temperature, HDD should be zero, so heating demand should be zero
        self.assertTrue(np.allclose(heating_demand, 0.0))

    def test_calculate_heating_demand_realistic_scenario(self):
        """Test heating demand with realistic winter scenario."""
        # Realistic parameters for Central European home
        specific_heating_demand = 80.0  # kWh/m²/year (modern insulated home)
        floor_area = 120.0  # m² (typical family home)
        # Typical winter week hourly temperatures (°C)
        outdoor_temps = np.array([2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0])
        base_temperature = 18.0
        annual_reference_hdd = 2800.0  # Typical for Central Europe
        optimization_time_step = 60  # 1-hour timestep

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            annual_reference_hdd,
            optimization_time_step,
        )

        # Verify all values are positive (cold weather)
        self.assertTrue(np.all(heating_demand > 0.0))

        # Verify coldest temperature has highest demand
        coldest_idx = np.argmin(outdoor_temps)
        self.assertEqual(coldest_idx, np.argmax(heating_demand))

        # Verify warmer temperature has lower demand
        warmest_idx = np.argmax(outdoor_temps)
        self.assertEqual(warmest_idx, np.argmin(heating_demand))

    def test_calculate_heating_demand_auto_infer_timestep(self):
        """Test automatic inference of optimization_time_step from pandas Series index."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_values = np.array([5.0, 10.0, 15.0, 8.0, 12.0])

        # Create pandas Series with 30-minute DatetimeIndex
        start_date = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        date_range_30min = pd.date_range(
            start=start_date, periods=len(outdoor_temps_values), freq="30min"
        )
        outdoor_temps_30min = pd.Series(outdoor_temps_values, index=date_range_30min)

        # Create pandas Series with 60-minute DatetimeIndex
        date_range_60min = pd.date_range(
            start=start_date, periods=len(outdoor_temps_values), freq="60min"
        )
        outdoor_temps_60min = pd.Series(outdoor_temps_values, index=date_range_60min)

        # Test auto-inference (should infer 30 min from Series)
        demand_auto_30 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_30min
        )

        # Test explicit 30 min parameter (should match auto-inference)
        demand_explicit_30 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_30min, optimization_time_step=30
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(demand_auto_30, demand_explicit_30)

        # Test auto-inference with 60-minute frequency
        demand_auto_60 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_60min
        )

        # Test explicit 60 min parameter
        demand_explicit_60 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_60min, optimization_time_step=60
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(demand_auto_60, demand_explicit_60)

        # Verify 60-min is double the demand of 30-min (when auto-inferred)
        np.testing.assert_array_almost_equal(demand_auto_60, demand_auto_30 * 2.0)

    def test_calculate_heating_demand_fallback_to_default(self):
        """Test fallback to default 30-minute timestep when not inferrable."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_array = np.array([5.0, 10.0, 15.0])

        # Test with numpy array (should fall back to 30 min)
        demand_array = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array
        )

        # Test explicit 30 min parameter
        demand_explicit = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array, optimization_time_step=30
        )

        # Results should be identical (both use 30 min)
        np.testing.assert_array_almost_equal(demand_array, demand_explicit)

        # Test with pandas Series without DatetimeIndex (should fall back to 30 min)
        outdoor_temps_series_no_dt = pd.Series(outdoor_temps_array)
        demand_series_no_dt = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_series_no_dt
        )

        # Should also match explicit 30 min
        np.testing.assert_array_almost_equal(demand_series_no_dt, demand_explicit)

    def test_calculate_heating_demand_physics_no_solar_basic_monotonic(self):
        """No solar gains: zero demand when outdoor >= indoor, higher demand for colder steps."""
        indoor_temp = 21.0
        # Outdoor temps: some above, some below indoor
        outdoor_temps = np.array([22.0, 21.0, 20.0, 15.0, 10.0, 5.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³

        demand = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # 1) Non-negative demand
        self.assertTrue(np.all(demand >= 0.0), "All heating demands should be non-negative")

        # 2) Zero demand when outdoor >= indoor (first two steps)
        self.assertEqual(demand[0], 0.0, "No heating needed when outdoor (22°C) > indoor (21°C)")
        self.assertEqual(demand[1], 0.0, "No heating needed when outdoor (21°C) = indoor (21°C)")

        # 3) Positive demand when outdoor < indoor
        self.assertGreater(demand[2], 0.0, "Heating needed when outdoor (20°C) < indoor (21°C)")
        self.assertGreater(demand[3], 0.0, "Heating needed when outdoor (15°C) < indoor (21°C)")

        # 4) Colder timesteps yield higher demand (monotonic relationship)
        colder_indices = [2, 3, 4, 5]
        for i in range(len(colder_indices) - 1):
            idx_warmer = colder_indices[i]
            idx_colder = colder_indices[i + 1]
            self.assertGreaterEqual(
                demand[idx_colder],
                demand[idx_warmer],
                msg=f"Demand at colder step {idx_colder} ({outdoor_temps[idx_colder]}°C) "
                f"should be >= step {idx_warmer} ({outdoor_temps[idx_warmer]}°C)",
            )

    def test_calculate_heating_demand_physics_accepts_per_timestep_ventilation_rate_array(self):
        """ventilation_rate may be a per-timestep array (e.g. a fixed extra
        ACH added only at the timestep an open window/door is detected,
        see optimization.py's OPENING_EXTRA_ACH) instead of a single scalar -
        plain elementwise broadcasting, no other change to the function."""
        indoor_temp = 21.0
        outdoor_temps = np.array([5.0, 5.0, 5.0, 5.0])
        optimization_time_step = 30  # minutes
        u_value = 0.35
        envelope_area = 380.0
        heated_volume = 240.0
        base_ventilation_rate = 0.4

        demand_scalar = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=base_ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # Boost only the first timestep's ventilation rate.
        ventilation_rate_arr = np.full(4, base_ventilation_rate)
        ventilation_rate_arr[0] += 8.0
        demand_array = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate_arr,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # Boosted timestep: strictly higher demand.
        self.assertGreater(demand_array[0], demand_scalar[0])
        # Every other (unboosted) timestep: unaffected.
        np.testing.assert_array_almost_equal(demand_array[1:], demand_scalar[1:])

    def test_calculate_heating_demand_physics_with_solar_gains_reduces_demand(self):
        """Solar gains reduce demand vs. no-solar case, and demand never becomes negative."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³
        window_area = 28.0  # m²
        shgc = 0.6  # Solar Heat Gain Coefficient

        # Simple GHI profile with some non-zero irradiance
        solar_irradiance = np.array([0.0, 200.0, 400.0, 0.0])  # W/m²

        demand_no_solar = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        demand_with_solar = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=solar_irradiance,
            window_area=window_area,
            shgc=shgc,
        )

        # Demand must never be negative
        self.assertTrue(
            np.all(demand_with_solar >= 0.0), "Demand with solar gains should never be negative"
        )

        # With solar gains, demand should not increase at any timestep
        self.assertTrue(
            np.all(demand_with_solar <= demand_no_solar),
            msg=f"Demand with solar gains should be <= no-solar demand at all timesteps.\n"
            f"no_solar={demand_no_solar}, with_solar={demand_with_solar}",
        )

        # For timesteps with non-zero irradiance, some reduction is expected
        self.assertLess(
            np.sum(demand_with_solar[solar_irradiance > 0.0]),
            np.sum(demand_no_solar[solar_irradiance > 0.0]),
            "Solar irradiance should reduce total heating demand during sunny periods",
        )

    def test_calculate_heating_demand_physics_scaling_with_timestep(self):
        """Sanity check: total demand scales appropriately with optimization_time_step."""
        indoor_temp = 21.0
        outdoor_temps = np.array([5.0, 5.0, 5.0, 5.0])  # constant cold
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³

        # Case 1: 30-minute timestep
        demand_30min = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=30,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # Case 2: 60-minute timestep with same temperatures
        demand_60min = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=60,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        total_30 = np.sum(demand_30min)
        total_60 = np.sum(demand_60min)

        # For a purely linear time scaling, 60-minute steps should yield about 2× 30-minute steps
        # (depending on implementation details, allow a small numerical tolerance).
        self.assertAlmostEqual(
            total_60,
            2.0 * total_30,
            delta=0.01 * total_60,
            msg=f"60-minute timestep total ({total_60:.3f}) should be ~2x 30-minute total ({total_30:.3f})",
        )

    def test_calculate_heating_demand_physics_with_internal_gains_reduces_demand(self):
        """Internal gains from electrical load reduce heating demand, and demand never becomes negative."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³

        # Electrical load profile in W
        load_forecast = np.array([1000.0, 2000.0, 3000.0, 1500.0])
        internal_gains_factor = 0.7  # 70% of electrical load becomes heat

        demand_no_internal = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
        )

        demand_with_internal = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            internal_gains_forecast=load_forecast,
            internal_gains_factor=internal_gains_factor,
        )

        # Demand must never be negative
        self.assertTrue(
            np.all(demand_with_internal >= 0.0),
            "Demand with internal gains should never be negative",
        )

        # With internal gains, demand should not increase at any timestep
        self.assertTrue(
            np.all(demand_with_internal <= demand_no_internal),
            msg=f"Demand with internal gains should be <= no-internal demand at all timesteps.\n"
            f"no_internal={demand_no_internal}, with_internal={demand_with_internal}",
        )

        # Total demand should be reduced
        self.assertLess(
            np.sum(demand_with_internal),
            np.sum(demand_no_internal),
            "Internal gains should reduce total heating demand",
        )

    def test_calculate_heating_demand_physics_with_both_solar_and_internal_gains(self):
        """Both solar and internal gains reduce heating demand cumulatively."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³
        window_area = 28.0  # m²
        shgc = 0.6

        # Solar and load profiles
        solar_irradiance = np.array([0.0, 200.0, 400.0, 0.0])  # W/m²
        load_forecast = np.array([1000.0, 2000.0, 2500.0, 1000.0])  # W
        internal_gains_factor = 0.7

        demand_no_gains = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
        )

        demand_solar_only = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=solar_irradiance,
            window_area=window_area,
            shgc=shgc,
        )

        demand_internal_only = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            internal_gains_forecast=load_forecast,
            internal_gains_factor=internal_gains_factor,
        )

        demand_both_gains = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=solar_irradiance,
            window_area=window_area,
            shgc=shgc,
            internal_gains_forecast=load_forecast,
            internal_gains_factor=internal_gains_factor,
        )

        # Demand must never be negative
        self.assertTrue(
            np.all(demand_both_gains >= 0.0),
            "Demand with both gains should never be negative",
        )

        # Per-timestep checks: gains must never increase demand at any timestep
        self.assertTrue(
            np.all(demand_both_gains <= demand_no_gains),
            f"Combined gains should not increase demand at any timestep:\n"
            f"no_gains={demand_no_gains}, both_gains={demand_both_gains}",
        )
        self.assertTrue(
            np.all(demand_both_gains <= demand_solar_only),
            f"Combined gains should not increase demand vs solar-only at any timestep:\n"
            f"solar_only={demand_solar_only}, both_gains={demand_both_gains}",
        )
        self.assertTrue(
            np.all(demand_both_gains <= demand_internal_only),
            f"Combined gains should not increase demand vs internal-only at any timestep:\n"
            f"internal_only={demand_internal_only}, both_gains={demand_both_gains}",
        )

        # Total demand should also be reduced (sum check)
        self.assertLess(
            np.sum(demand_both_gains),
            np.sum(demand_solar_only),
            "Combined gains should reduce total demand more than solar only",
        )
        self.assertLess(
            np.sum(demand_both_gains),
            np.sum(demand_internal_only),
            "Combined gains should reduce total demand more than internal only",
        )
        self.assertLess(
            np.sum(demand_both_gains),
            np.sum(demand_no_gains),
            "Combined gains should reduce total demand vs no gains",
        )

    def test_calculate_heating_demand_physics_internal_gains_factor_zero(self):
        """Factor of 0 should have no effect (backwards compatibility)."""
        indoor_temp = 21.0
        outdoor_temps = np.array([5.0, 5.0, 5.0, 5.0])
        optimization_time_step = 30
        u_value = 0.35
        envelope_area = 380.0
        ventilation_rate = 0.4
        heated_volume = 240.0
        load_forecast = np.array([2000.0, 3000.0, 4000.0, 2500.0])  # W

        demand_no_internal = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
        )

        demand_with_zero_factor = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            internal_gains_forecast=load_forecast,
            internal_gains_factor=0.0,
        )

        # With factor=0, demand should be identical to no internal gains
        np.testing.assert_array_almost_equal(
            demand_no_internal,
            demand_with_zero_factor,
            decimal=10,
            err_msg="Factor=0 should produce identical results to no internal gains",
        )

    def test_calculate_heating_demand_physics_internal_gains_with_pandas_series(self):
        """Internal gains should work with pandas Series input."""
        indoor_temp = 21.0
        outdoor_temps = np.array([5.0, 5.0, 5.0, 5.0])
        optimization_time_step = 30
        u_value = 0.35
        envelope_area = 380.0
        ventilation_rate = 0.4
        heated_volume = 240.0
        load_array = np.array([2000.0, 3000.0, 4000.0, 2500.0])  # W
        load_series = pd.Series(load_array)
        internal_gains_factor = 0.8

        demand_from_array = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            internal_gains_forecast=load_array,
            internal_gains_factor=internal_gains_factor,
        )

        demand_from_series = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            internal_gains_forecast=load_series,
            internal_gains_factor=internal_gains_factor,
        )

        np.testing.assert_array_almost_equal(
            demand_from_array,
            demand_from_series,
            decimal=10,
            err_msg="Results should be identical for array and Series input",
        )

    def test_calculate_heating_demand_physics_internal_gains_mismatched_lengths(self):
        """Mismatched internal gains and outdoor temperature forecasts raise ValueError."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])  # 4 elements
        optimization_time_step = 60
        u_value = 0.35
        envelope_area = 380.0
        ventilation_rate = 0.4
        heated_volume = 240.0

        # Internal gains forecast with different length (3 instead of 4)
        load_forecast_wrong_length = np.array([1000.0, 2000.0, 3000.0])  # 3 elements (W)
        internal_gains_factor = 0.7

        with self.assertRaises(ValueError) as context:
            utils.calculate_heating_demand_physics(
                u_value=u_value,
                envelope_area=envelope_area,
                ventilation_rate=ventilation_rate,
                heated_volume=heated_volume,
                indoor_target_temperature=indoor_temp,
                outdoor_temperature_forecast=outdoor_temps,
                optimization_time_step=optimization_time_step,
                internal_gains_forecast=load_forecast_wrong_length,
                internal_gains_factor=internal_gains_factor,
            )

        self.assertIn("internal_gains_forecast length", str(context.exception))
        self.assertIn("outdoor_temperature_forecast length", str(context.exception))

    def test_calculate_heating_demand_physics_internal_gains_factor_out_of_range(self):
        """Factor outside [0, 1] range raises ValueError."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60
        u_value = 0.35
        envelope_area = 380.0
        ventilation_rate = 0.4
        heated_volume = 240.0
        load_forecast = np.array([1000.0, 2000.0, 3000.0, 1500.0])  # W

        # Test factor > 1
        with self.assertRaises(ValueError) as context:
            utils.calculate_heating_demand_physics(
                u_value=u_value,
                envelope_area=envelope_area,
                ventilation_rate=ventilation_rate,
                heated_volume=heated_volume,
                indoor_target_temperature=indoor_temp,
                outdoor_temperature_forecast=outdoor_temps,
                optimization_time_step=optimization_time_step,
                internal_gains_forecast=load_forecast,
                internal_gains_factor=1.5,  # Invalid: > 1
            )

        self.assertIn("internal_gains_factor must be between 0 and 1", str(context.exception))

        # Test factor < 0 should also raise ValueError
        with self.assertRaises(ValueError) as context_neg:
            utils.calculate_heating_demand_physics(
                u_value=u_value,
                envelope_area=envelope_area,
                ventilation_rate=ventilation_rate,
                heated_volume=heated_volume,
                indoor_target_temperature=indoor_temp,
                outdoor_temperature_forecast=outdoor_temps,
                optimization_time_step=optimization_time_step,
                internal_gains_forecast=load_forecast,
                internal_gains_factor=-0.5,  # Invalid: < 0
            )

        self.assertIn("internal_gains_factor must be between 0 and 1", str(context_neg.exception))

    def test_calculate_heating_demand_physics_internal_gains_warns_on_low_values(self):
        """Warning should be raised when values look like kW instead of W."""
        import warnings

        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60
        u_value = 0.35
        envelope_area = 380.0
        ventilation_rate = 0.4
        heated_volume = 240.0
        # Values that look like kW (1-5 range) instead of W (1000-5000 range)
        load_forecast_kw_mistake = np.array([1.0, 2.0, 3.0, 1.5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            utils.calculate_heating_demand_physics(
                u_value=u_value,
                envelope_area=envelope_area,
                ventilation_rate=ventilation_rate,
                heated_volume=heated_volume,
                indoor_target_temperature=indoor_temp,
                outdoor_temperature_forecast=outdoor_temps,
                optimization_time_step=optimization_time_step,
                internal_gains_forecast=load_forecast_kw_mistake,
                internal_gains_factor=0.7,
            )

            # Verify warning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertIn("very low", str(w[0].message))
            self.assertIn("Watts, not kilowatts", str(w[0].message))

    def test_calculate_heating_demand_physics_cooling_sense_targets_hot_steps(self):
        """sense='cool': demand magnitude lands on HOT steps, not cold ones (#994).

        Base-safe: if the running build does not accept a ``sense`` argument the
        call falls back to the default (heating) behaviour, under which the
        magnitude ordering is INVERTED, so this test fails on the behavioural
        assertion rather than on a TypeError.
        """
        import inspect

        indoor_temp = 24.0
        # Hot day, cool night profile: steps 0-1 hot (outdoor > indoor),
        # steps 2-3 cool (outdoor < indoor).
        outdoor_temps = np.array([34.0, 30.0, 20.0, 16.0])
        kwargs = {
            "u_value": 0.30,
            "envelope_area": 342.0,
            "ventilation_rate": 0.3,
            "heated_volume": 597.0,
            "indoor_target_temperature": indoor_temp,
            "outdoor_temperature_forecast": outdoor_temps,
            "optimization_time_step": 60,
        }
        supports_sense = (
            "sense" in inspect.signature(utils.calculate_heating_demand_physics).parameters
        )
        if supports_sense:
            kwargs["sense"] = "cool"

        demand = utils.calculate_heating_demand_physics(**kwargs)

        hot_magnitude = abs(demand[0]) + abs(demand[1])
        cool_magnitude = abs(demand[2]) + abs(demand[3])

        # Cooling need must be driven by the HOT steps, not the cool ones.
        self.assertGreater(
            hot_magnitude,
            cool_magnitude,
            msg=(
                "Cooling demand should be non-zero when it is hot outside, near-zero "
                f"when it is cool (got hot={hot_magnitude:.4f} cool={cool_magnitude:.4f})"
            ),
        )

    def test_calculate_heating_demand_physics_cooling_sign_and_solar(self):
        """sense='cool': demand is signed as a heat gain (<= 0) and solar adds load.

        Skipped on builds without ``sense`` support so the suite stays green on
        base; the behavioural RED proof above is the base-sensitive one.
        """
        import inspect

        if "sense" not in inspect.signature(utils.calculate_heating_demand_physics).parameters:
            self.skipTest("running build has no cooling sense support")

        indoor_temp = 24.0
        outdoor_temps = np.array([34.0, 34.0, 34.0, 34.0])
        base_kwargs = {
            "u_value": 0.30,
            "envelope_area": 342.0,
            "ventilation_rate": 0.3,
            "heated_volume": 597.0,
            "indoor_target_temperature": indoor_temp,
            "outdoor_temperature_forecast": outdoor_temps,
            "optimization_time_step": 60,
            "sense": "cool",
        }

        demand_no_solar = utils.calculate_heating_demand_physics(**base_kwargs)
        # Signed as a heat gain: cooling demand is never positive.
        self.assertTrue(
            np.all(demand_no_solar <= 0.0),
            "Cooling demand should be returned as a heat gain (<= 0)",
        )

        # Solar gains ADD to the cooling load (push demand more negative).
        demand_with_solar = utils.calculate_heating_demand_physics(
            **base_kwargs,
            solar_irradiance_forecast=np.array([600.0, 600.0, 600.0, 600.0]),
            window_area=12.0,
            shgc=0.6,
        )
        self.assertTrue(
            np.all(demand_with_solar <= demand_no_solar + 1e-9),
            "Solar gains should increase cooling load (more negative demand)",
        )

    def test_calculate_heating_demand_physics_heating_sense_is_noop(self):
        """Default and explicit sense='heat' must be byte-identical (true no-op)."""
        import inspect

        indoor_temp = 21.0
        outdoor_temps = np.array([22.0, 15.0, 10.0, 5.0])
        kwargs = {
            "u_value": 0.35,
            "envelope_area": 380.0,
            "ventilation_rate": 0.4,
            "heated_volume": 240.0,
            "indoor_target_temperature": indoor_temp,
            "outdoor_temperature_forecast": outdoor_temps,
            "optimization_time_step": 60,
            "solar_irradiance_forecast": np.array([0.0, 200.0, 400.0, 0.0]),
            "window_area": 28.0,
            "shgc": 0.6,
        }
        demand_default = utils.calculate_heating_demand_physics(**kwargs)

        if "sense" in inspect.signature(utils.calculate_heating_demand_physics).parameters:
            demand_heat = utils.calculate_heating_demand_physics(**kwargs, sense="heat")
            np.testing.assert_array_equal(demand_default, demand_heat)

        # Heating demand stays non-negative regardless.
        self.assertTrue(np.all(demand_default >= 0.0))

    def test_calculate_cop_heatpump(self):
        """Test heat pump COP calculation utility function with Carnot-based formula."""
        # Test basic calculation with example outdoor temperatures
        supply_temp = 35.0  # °C
        carnot_efficiency = 0.4  # Typical value for real heat pumps (40% of Carnot)
        outdoor_temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

        cops = utils.calculate_cop_heatpump(supply_temp, carnot_efficiency, outdoor_temps)

        # Verify output is numpy array
        self.assertIsInstance(cops, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(cops), len(outdoor_temps))

        # Manually verify first value using Carnot formula:
        # COP = carnot_efficiency * T_supply_kelvin / (T_supply_kelvin - T_outdoor_kelvin)
        # COP = 0.4 * (35 + 273.15) / |(35 + 273.15) - (0 + 273.15)|
        # COP = 0.4 * 308.15 / 35 = 3.521...
        supply_kelvin = supply_temp + 273.15
        outdoor_kelvin = outdoor_temps[0] + 273.15
        expected_first_cop = carnot_efficiency * supply_kelvin / abs(supply_kelvin - outdoor_kelvin)
        self.assertAlmostEqual(cops[0], expected_first_cop, places=6)

        # Verify all COPs are non-negative
        self.assertTrue(np.all(cops >= 0.0))

        # Test with pandas Series input
        outdoor_temps_series = pd.Series(outdoor_temps)
        cops_from_series = utils.calculate_cop_heatpump(
            supply_temp, carnot_efficiency, outdoor_temps_series
        )
        np.testing.assert_array_almost_equal(cops, cops_from_series)

        # Test that COP decreases as temperature difference increases
        # When outdoor temp gets further from supply temp, COP should decrease
        outdoor_increasing = np.array([30.0, 25.0, 20.0, 15.0, 10.0])  # Getting colder
        cops_decreasing = utils.calculate_cop_heatpump(
            supply_temp, carnot_efficiency, outdoor_increasing
        )
        # Each successive COP should be lower as temp difference increases
        for i in range(len(cops_decreasing) - 1):
            self.assertGreaterEqual(cops_decreasing[i], cops_decreasing[i + 1])

        # Test with different carnot_efficiency values
        carnot_eff_high = 0.5
        cops_high_eff = utils.calculate_cop_heatpump(supply_temp, carnot_eff_high, outdoor_temps)
        # Higher Carnot efficiency should give proportionally higher COPs (subject to 8.0 cap)
        expected_ratio = carnot_eff_high / carnot_efficiency
        expected_cops_uncapped = cops * expected_ratio
        expected_cops_capped = np.minimum(expected_cops_uncapped, 8.0)
        np.testing.assert_array_almost_equal(cops_high_eff, expected_cops_capped)

        # Test realistic scenario: heat pump at 35°C supply, 5°C outdoor
        # COP = 0.4 * 308.15 / |308.15 - 278.15| = 0.4 * 308.15 / 30 = 4.108
        cop_realistic = utils.calculate_cop_heatpump(35.0, 0.4, np.array([5.0]))
        expected_realistic = 0.4 * (35 + 273.15) / abs((35 + 273.15) - (5 + 273.15))
        self.assertAlmostEqual(cop_realistic[0], expected_realistic, places=6)
        # Typical heat pump COP should be in range 2-6 for normal conditions
        self.assertGreater(cop_realistic[0], 2.0)
        self.assertLess(cop_realistic[0], 6.0)

    def test_calculate_cop_heatpump_edge_case_warning(self):
        """Test COP calculation logs warning when outdoor temp >= supply temp."""

        # Test case where outdoor temps exceed or equal supply temp
        supply_temp = 30.0
        carnot_eff = 0.4
        # Mix of normal and problematic outdoor temps
        outdoor_temps = np.array([5.0, 10.0, 30.0, 35.0, 40.0])  # Last 3 >= supply

        # Capture log messages
        with self.assertLogs("emhass.utils", level="WARNING") as log_context:
            cops = utils.calculate_cop_heatpump(supply_temp, carnot_eff, outdoor_temps)

            # Verify warning was logged
            self.assertTrue(
                any(
                    "outdoor temperature >= supply temperature" in msg for msg in log_context.output
                ),
                "Should log warning about non-physical temperature scenario",
            )

        # Verify result is still valid (uses COP=1.0 for non-physical scenarios)
        self.assertIsInstance(cops, np.ndarray)
        self.assertEqual(len(cops), len(outdoor_temps))
        self.assertTrue(np.all(cops >= 1.0), "All COPs should be >= 1.0 (lower bound)")
        self.assertTrue(np.all(cops <= 8.0), "All COPs should be <= 8.0 (upper bound)")
        self.assertTrue(np.all(np.isfinite(cops)), "All COPs should be finite (no inf/nan)")
        # Non-physical scenarios (outdoor >= supply) should get COP=1.0 (direct electric heating)
        # outdoor_temps = [5, 10, 30, 35, 40], supply = 30
        # Valid: cops[0], cops[1]  (5 < 30, 10 < 30)
        # Invalid: cops[2], cops[3], cops[4]  (30 >= 30, 35 > 30, 40 > 30)
        self.assertEqual(cops[2], 1.0, "Boundary case (equal temps) should have COP=1.0")
        self.assertEqual(
            cops[3], 1.0, "Non-physical scenario (outdoor > supply) should have COP=1.0"
        )
        self.assertEqual(
            cops[4], 1.0, "Non-physical scenario (outdoor > supply) should have COP=1.0"
        )
        # Valid scenarios should have reasonable COP > 1.0
        self.assertGreater(cops[0], 1.0, "Valid scenario should have COP > 1.0")
        self.assertGreater(cops[1], 1.0, "Valid scenario should have COP > 1.0")

    def test_calculate_cop_heatpump_cooling_mode(self):
        """Cooling mode uses the inverted Carnot lift (outdoor - supply)."""
        supply_temp = 18.5
        carnot_eff = 0.45
        outdoor_temps = np.array([24.0, 30.0])

        cops = utils.calculate_cop_heatpump(
            supply_temp,
            carnot_eff,
            outdoor_temps,
            mode="cool",
        )

        supply_kelvin = supply_temp + 273.15
        expected = carnot_eff * supply_kelvin / ((outdoor_temps + 273.15) - supply_kelvin)
        expected = np.clip(expected, 1.0, 8.0)

        np.testing.assert_allclose(cops, expected)
        self.assertTrue(np.all(cops > 1.0))

    def test_calculate_cop_heatpump_cooling_mode_warning(self):
        """Cooling mode warns and clamps to COP=1.0 when outdoor <= supply."""
        supply_temp = 18.5
        carnot_eff = 0.45
        outdoor_temps = np.array([15.0, 18.5, 22.0])

        with self.assertLogs("emhass.utils", level="WARNING") as log_context:
            cops = utils.calculate_cop_heatpump(
                supply_temp,
                carnot_eff,
                outdoor_temps,
                mode="cool",
            )

            self.assertTrue(
                any(
                    "outdoor temperature <= supply temperature" in msg for msg in log_context.output
                )
            )

        self.assertEqual(cops[0], 1.0)
        self.assertEqual(cops[1], 1.0)
        self.assertGreater(cops[2], 1.0)

    def test_calculate_cop_heatpump_invalid_mode_raises(self):
        """Invalid mode must fail explicitly instead of silently falling back."""
        with self.assertRaises(ValueError) as ctx:
            utils.calculate_cop_heatpump(
                supply_temperature=35.0,
                carnot_efficiency=0.4,
                outdoor_temperature_forecast=np.array([5.0, 10.0]),
                mode=" typo ",
            )
        self.assertIn("COP calculation", str(ctx.exception))
        self.assertIn("Expected 'heat' or 'cool'", str(ctx.exception))

    def test_normalize_heat_cool_mode(self):
        self.assertEqual(utils.normalize_heat_cool_mode(" HeAt "), "heat")
        self.assertEqual(utils.normalize_heat_cool_mode(" COOL "), "cool")

        with self.assertRaises(ValueError) as ctx:
            utils.normalize_heat_cool_mode(" typo ", field_name="sense", context="thermal_battery")
        self.assertIn("thermal_battery", str(ctx.exception))
        self.assertIn("invalid sense", str(ctx.exception))

    def test_calculate_thermal_loss_signed(self):
        """Test thermal loss sign-switching utility function based on Langer & Volling (2020)."""
        # Test basic calculation with temperatures crossing the indoor threshold
        indoor_temp = 20.0
        base_loss = 0.045
        # Outdoor temps: some below indoor (loss), some above indoor (gain)
        outdoor_temps = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

        losses = utils.calculate_thermal_loss_signed(outdoor_temps, indoor_temp, base_loss)

        # Verify output is numpy array
        self.assertIsInstance(losses, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(losses), len(outdoor_temps))

        # Verify sign switching based on temperature threshold
        # When outdoor < indoor: Hot(h) = 0, Loss = base_loss * (1 - 2*0) = +base_loss (positive loss)
        # When outdoor >= indoor: Hot(h) = 1, Loss = base_loss * (1 - 2*1) = -base_loss (negative loss/gain)
        self.assertAlmostEqual(losses[0], base_loss, places=6)  # 10°C < 20°C: +loss
        self.assertAlmostEqual(losses[1], base_loss, places=6)  # 15°C < 20°C: +loss
        self.assertAlmostEqual(losses[2], -base_loss, places=6)  # 20°C >= 20°C: -loss (gain)
        self.assertAlmostEqual(losses[3], -base_loss, places=6)  # 25°C >= 20°C: -loss (gain)
        self.assertAlmostEqual(losses[4], -base_loss, places=6)  # 30°C >= 20°C: -loss (gain)

        # Test with pandas Series input
        outdoor_temps_series = pd.Series(outdoor_temps)
        losses_from_series = utils.calculate_thermal_loss_signed(
            outdoor_temps_series, indoor_temp, base_loss
        )
        np.testing.assert_array_almost_equal(losses, losses_from_series)

        # Test winter scenario: all outdoor temps below indoor (all positive losses)
        outdoor_winter = np.array([0.0, 5.0, 10.0, 15.0])
        losses_winter = utils.calculate_thermal_loss_signed(outdoor_winter, indoor_temp, base_loss)
        self.assertTrue(np.all(losses_winter > 0))
        self.assertTrue(np.allclose(losses_winter, base_loss))

        # Test summer scenario: all outdoor temps above indoor (all negative losses)
        outdoor_summer = np.array([25.0, 30.0, 35.0, 40.0])
        losses_summer = utils.calculate_thermal_loss_signed(outdoor_summer, indoor_temp, base_loss)
        self.assertTrue(np.all(losses_summer < 0))
        self.assertTrue(np.allclose(losses_summer, -base_loss))

        # Test with different base_loss value
        base_loss_2 = 0.1
        losses_2 = utils.calculate_thermal_loss_signed(outdoor_temps, indoor_temp, base_loss_2)
        # Verify magnitude is scaled by base_loss
        expected_ratio = base_loss_2 / base_loss
        np.testing.assert_array_almost_equal(losses_2, losses * expected_ratio)

        # Test formula correctness per Langer & Volling (2020) Equation B.13
        # Loss+/- = base_loss * (1 - 2 * Hot(h))
        # Manual verification for outdoor_temp = 18°C (< 20°C indoor)
        loss_manual_cold = base_loss * (1 - 2 * 0)
        self.assertAlmostEqual(loss_manual_cold, base_loss, places=6)

        # Manual verification for outdoor_temp = 22°C (>= 20°C indoor)
        loss_manual_warm = base_loss * (1 - 2 * 1)
        self.assertAlmostEqual(loss_manual_warm, -base_loss, places=6)


class TestResolveThermalBatteryCop(unittest.TestCase):
    """Tests for the dispatch helper that picks between Carnot COP and flat efficiency."""

    def test_efficiency_mode_returns_flat_array(self):
        """When 'efficiency' is set, return a flat conversion-factor array."""
        hc = {"efficiency": 0.9}
        outdoor = np.array([0.0, 5.0, 10.0, 15.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=4)
        np.testing.assert_array_almost_equal(cops, np.full(4, 0.9))

    def test_efficiency_mode_ignores_outdoor_temperature(self):
        """Flat efficiency does not vary with outdoor temperature."""
        hc = {"efficiency": 0.85}
        hot = utils.resolve_thermal_battery_cop(hc, np.array([20.0] * 6), length=6)
        cold = utils.resolve_thermal_battery_cop(hc, np.array([-15.0] * 6), length=6)
        np.testing.assert_array_almost_equal(hot, cold)
        self.assertTrue(np.all(hot == 0.85))

    def test_efficiency_mode_does_not_require_supply_temperature(self):
        """Constant-efficiency mode works without a supply_temperature field."""
        hc = {"efficiency": 0.9}  # no supply_temperature, no carnot_efficiency
        cops = utils.resolve_thermal_battery_cop(hc, np.array([5.0, 5.0]), length=2)
        np.testing.assert_array_almost_equal(cops, np.array([0.9, 0.9]))

    def test_heatpump_mode_falls_back_to_carnot(self):
        """When 'efficiency' is not set, fall back to the existing Carnot COP."""
        hc = {"supply_temperature": 35.0, "carnot_efficiency": 0.4}
        outdoor = np.array([5.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=1)
        expected = utils.calculate_cop_heatpump(35.0, 0.4, outdoor)
        np.testing.assert_array_almost_equal(cops, expected)

    def test_heatpump_mode_default_carnot_efficiency(self):
        """carnot_efficiency defaults to 0.4 when not set."""
        hc_explicit = {"supply_temperature": 35.0, "carnot_efficiency": 0.4}
        hc_implicit = {"supply_temperature": 35.0}
        outdoor = np.array([0.0, 5.0, 10.0])
        cops_explicit = utils.resolve_thermal_battery_cop(hc_explicit, outdoor, length=3)
        cops_implicit = utils.resolve_thermal_battery_cop(hc_implicit, outdoor, length=3)
        np.testing.assert_array_almost_equal(cops_explicit, cops_implicit)

    def test_heatpump_mode_cooling_sense_uses_cooling_cop(self):
        """resolve_thermal_battery_cop should forward sense='cool' to COP calc."""
        hc = {
            "sense": "cool",
            "supply_temperature": 18.5,
            "carnot_efficiency": 0.45,
        }
        outdoor = np.array([24.0, 30.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=2)
        expected = utils.calculate_cop_heatpump(18.5, 0.45, outdoor, mode="cool")
        np.testing.assert_array_almost_equal(cops, expected)

    def test_heatpump_mode_invalid_sense_raises(self):
        hc = {
            "sense": "invalid",
            "supply_temperature": 18.5,
            "carnot_efficiency": 0.45,
        }
        outdoor = np.array([24.0, 30.0])
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_thermal_battery_cop(hc, outdoor, length=2)
        self.assertIn("thermal_battery", str(ctx.exception))
        self.assertIn("invalid sense", str(ctx.exception))

    def test_sense_null_falls_back_to_heat(self):
        outdoor = np.array([0.0, 5.0])
        expected = utils.calculate_cop_heatpump(35.0, 0.4, outdoor, mode="heat")
        hc = {"sense": None, "supply_temperature": 35.0}
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=2)
        np.testing.assert_array_almost_equal(cops, expected)

    def test_missing_both_efficiency_and_supply_temperature_raises(self):
        """At least one of efficiency or supply_temperature must be set."""
        hc = {"carnot_efficiency": 0.4}
        outdoor = np.array([5.0])
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_thermal_battery_cop(hc, outdoor, length=1)
        self.assertIn("efficiency", str(ctx.exception))
        self.assertIn("supply_temperature", str(ctx.exception))

    def test_nonpositive_efficiency_raises(self):
        """efficiency must be strictly positive."""
        for bad in (0.0, -0.5):
            hc = {"efficiency": bad}
            with self.assertRaises(ValueError) as ctx:
                utils.resolve_thermal_battery_cop(hc, np.array([5.0]), length=1)
            self.assertIn("positive", str(ctx.exception))

    def test_length_truncation_in_heatpump_mode(self):
        """When `length` is given, the returned array is truncated."""
        hc = {"supply_temperature": 35.0}
        outdoor = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=3)
        self.assertEqual(len(cops), 3)

    def test_length_none_returns_full_forecast(self):
        """When `length` is None, return the full forecast length."""
        hc = {"supply_temperature": 35.0}
        outdoor = np.array([0.0, 5.0, 10.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor, length=None)
        self.assertEqual(len(cops), 3)

    def test_efficiency_mode_accepts_outdoor_none_when_length_given(self):
        """Constant-efficiency mode ignores outdoor; passing None is allowed
        when `length` is supplied explicitly."""
        hc = {"efficiency": 0.9}
        cops = utils.resolve_thermal_battery_cop(hc, None, length=4)
        np.testing.assert_array_almost_equal(cops, np.full(4, 0.9))

    def test_efficiency_mode_outdoor_none_without_length_raises(self):
        """Constant-efficiency mode with outdoor=None and length=None is
        ambiguous and raises with a clear message."""
        hc = {"efficiency": 0.9}
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_thermal_battery_cop(hc, None, length=None)
        self.assertIn("length", str(ctx.exception))

    def test_heatpump_mode_outdoor_none_raises(self):
        """Heat-pump mode requires outdoor temperature; None must raise."""
        hc = {"supply_temperature": 35.0}
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_thermal_battery_cop(hc, None, length=3)
        self.assertIn("outdoor_temperature_forecast", str(ctx.exception))


class TestSimulatePhysicsRoomTemperatureTrajectory(unittest.TestCase):
    """Tests for the open-loop physics/RC simulation used to score the
    self-learning-physics refit's per-room physics baseline."""

    def test_matches_hand_composed_recurrence(self):
        """The recursive result must match COP/loss building blocks composed
        by hand, step by step - proves the wiring, not just the shape."""
        initial_temp = 20.0
        duty = np.array([0.5, 0.5, 0.5, 0.5])
        outdoor_temp = np.array([5.0, 5.0, 5.0, 5.0])
        nominal_power_w = 1500.0
        dt_hours = 0.5

        result = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=initial_temp,
            duty=duty,
            outdoor_temp=outdoor_temp,
            nominal_power_w=nominal_power_w,
            dt_hours=dt_hours,
        )

        cops = utils.calculate_cop_heatpump(35.0, 0.4, outdoor_temp)
        losses = utils.calculate_thermal_loss_signed(outdoor_temp, initial_temp, 0.045)
        conversion = 3600.0 / (2400.0 * 0.88 * 15.0)
        power_w = duty * nominal_power_w
        expected = np.zeros(4)
        expected[0] = initial_temp
        for t in range(3):
            heat_in_kwh = cops[t] * power_w[t] / 1000.0 * dt_hours
            expected[t + 1] = expected[t] + conversion * (heat_in_kwh - 0.0 - losses[t])

        np.testing.assert_array_almost_equal(result, expected)

    def test_initial_value_preserved(self):
        """The first element must always equal initial_temp exactly."""
        result = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=21.3,
            duty=np.array([0.0, 1.0, 0.2]),
            outdoor_temp=np.array([-5.0, 0.0, 5.0]),
            nominal_power_w=1000.0,
            dt_hours=1.0,
        )
        self.assertEqual(result[0], 21.3)

    def test_length_matches_input(self):
        for n in (1, 2, 5):
            result = utils.simulate_physics_room_temperature_trajectory(
                initial_temp=20.0,
                duty=np.zeros(n),
                outdoor_temp=np.full(n, 10.0),
                nominal_power_w=1000.0,
                dt_hours=1.0,
            )
            self.assertEqual(len(result), n)

    def test_single_step_returns_only_initial_temp(self):
        """With a single row there is nothing to recurse over."""
        result = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=19.5,
            duty=np.array([0.8]),
            outdoor_temp=np.array([-2.0]),
            nominal_power_w=2000.0,
            dt_hours=1.0,
        )
        np.testing.assert_array_almost_equal(result, np.array([19.5]))

    def test_strong_heating_raises_temperature_despite_cold_outdoor(self):
        """Enough heating power must win against a cold-outdoor loss term."""
        result = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=18.0,
            duty=np.full(6, 1.0),
            outdoor_temp=np.full(6, -5.0),
            nominal_power_w=3000.0,
            dt_hours=1.0,
        )
        self.assertTrue(np.all(np.diff(result) > 0))

    def test_zero_duty_cold_outdoor_temperature_decreases(self):
        """With no heating power at all, a colder outdoor must cool the room."""
        result = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=20.0,
            duty=np.zeros(6),
            outdoor_temp=np.full(6, -5.0),
            nominal_power_w=1500.0,
            dt_hours=1.0,
        )
        self.assertTrue(np.all(np.diff(result) < 0))

    def test_cooling_sense_flips_heating_direction(self):
        """sense='cool' must actively counteract passive heat gain, unlike
        sense='heat' (or zero duty) under the same hot-outdoor scenario."""
        kwargs = {
            "initial_temp": 25.0,
            "duty": np.full(6, 1.0),
            "outdoor_temp": np.full(6, 30.0),
            "nominal_power_w": 3000.0,
            "dt_hours": 1.0,
            "supply_temperature": 18.0,
            "carnot_efficiency": 0.4,
        }
        cooling = utils.simulate_physics_room_temperature_trajectory(sense="cool", **kwargs)
        no_op = utils.simulate_physics_room_temperature_trajectory(
            **{**kwargs, "duty": np.zeros(6)}, sense="cool"
        )
        self.assertTrue(np.all(np.diff(cooling) < 0))
        self.assertTrue(np.all(np.diff(no_op) > 0))

    def test_heating_demand_kwh_lowers_trajectory(self):
        """A nonzero ongoing heating demand must pull the trajectory down
        relative to the same run with the default zero-demand assumption."""
        kwargs = {
            "initial_temp": 20.0,
            "duty": np.full(5, 0.6),
            "outdoor_temp": np.full(5, 2.0),
            "nominal_power_w": 1500.0,
            "dt_hours": 0.5,
        }
        without_demand = utils.simulate_physics_room_temperature_trajectory(**kwargs)
        with_demand = utils.simulate_physics_room_temperature_trajectory(
            heating_demand_kwh=np.full(5, 0.2), **kwargs
        )
        self.assertTrue(np.all(with_demand[1:] < without_demand[1:]))

    def test_default_supply_temperature_and_carnot_efficiency_match_explicit(self):
        explicit = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=20.0,
            duty=np.full(4, 0.5),
            outdoor_temp=np.full(4, 5.0),
            nominal_power_w=1500.0,
            dt_hours=0.5,
            supply_temperature=35.0,
            carnot_efficiency=0.4,
        )
        implicit = utils.simulate_physics_room_temperature_trajectory(
            initial_temp=20.0,
            duty=np.full(4, 0.5),
            outdoor_temp=np.full(4, 5.0),
            nominal_power_w=1500.0,
            dt_hours=0.5,
        )
        np.testing.assert_array_almost_equal(explicit, implicit)

    def test_nonpositive_density_heat_capacity_or_volume_raises(self):
        base = {
            "initial_temp": 20.0,
            "duty": np.array([0.5]),
            "outdoor_temp": np.array([5.0]),
            "nominal_power_w": 1500.0,
            "dt_hours": 0.5,
        }
        for bad_kwargs in (
            {"density": 0.0},
            {"density": -1.0},
            {"heat_capacity": 0.0},
            {"volume": 0.0},
            {"volume": -15.0},
        ):
            with self.assertRaises(ValueError):
                utils.simulate_physics_room_temperature_trajectory(**{**base, **bad_kwargs})

    def test_invalid_sense_raises(self):
        with self.assertRaises(ValueError) as ctx:
            utils.simulate_physics_room_temperature_trajectory(
                initial_temp=20.0,
                duty=np.array([0.5]),
                outdoor_temp=np.array([5.0]),
                nominal_power_w=1500.0,
                dt_hours=0.5,
                sense="invalid",
            )
        self.assertIn("invalid sense", str(ctx.exception))


class TestCalculateShadedWindowIrradiance(unittest.TestCase):
    """Tests for the direct/diffuse solar decomposition + blind-shading
    helper used by the physics-family room heating-demand formula."""

    def test_none_type_passes_through_unattenuated(self):
        dni = np.array([100.0, 200.0])
        dhi = np.array([50.0, 60.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=0.7, blind_type="none"
        )
        np.testing.assert_array_almost_equal(result, [150.0, 260.0])

    def test_unset_type_defaults_to_no_shading(self):
        dni = np.array([100.0])
        dhi = np.array([50.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type=""
        )
        np.testing.assert_array_almost_equal(result, [150.0])

    def test_screen_blocks_direct_proportional_to_position(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=0.5, blind_type="screen"
        )
        np.testing.assert_array_almost_equal(result, [130.0])

    def test_screen_is_angle_independent(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        low_sun = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=0.5, blind_type="screen",
            solar_elevation_deg=np.array([5.0]),
        )
        high_sun = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=0.5, blind_type="screen",
            solar_elevation_deg=np.array([60.0]),
        )
        np.testing.assert_array_almost_equal(low_sun, high_sun)
        np.testing.assert_array_almost_equal(low_sun, [130.0])

    def test_awning_zero_effect_below_low_elevation(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type="awning",
            solar_elevation_deg=np.array([10.0]),  # below default low=20
        )
        np.testing.assert_array_almost_equal(result, [230.0])  # no shading at all

    def test_awning_full_effect_above_high_elevation(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type="awning",
            solar_elevation_deg=np.array([50.0]),  # above default high=45
        )
        np.testing.assert_array_almost_equal(result, [30.0])  # direct fully blocked

    def test_awning_linear_ramp_between_thresholds(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type="awning",
            solar_elevation_deg=np.array([32.5]),  # midpoint of default 20-45
        )
        np.testing.assert_array_almost_equal(result, [130.0])  # 50% blocked

    def test_awning_also_scales_with_blind_position(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=0.4, blind_type="awning",
            solar_elevation_deg=np.array([50.0]),  # full elevation factor
        )
        np.testing.assert_array_almost_equal(result, [30.0 + 200.0 * 0.6])

    def test_awning_without_elevation_degrades_to_no_shading(self):
        dni = np.array([200.0])
        dhi = np.array([30.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type="awning",
            solar_elevation_deg=None,
        )
        np.testing.assert_array_almost_equal(result, [230.0])

    def test_diffuse_never_attenuated_by_any_type(self):
        # Invariant: changing dhi by some delta must change the result by
        # exactly that same delta, for every blind_type/position/elevation
        # combination - proving diffuse always passes through with a fixed
        # coefficient of 1, completely independent of shading.
        dni = np.array([200.0])
        for blind_type, elev, position in (
            ("none", None, 1.0),
            ("screen", None, 1.0),
            ("awning", np.array([50.0]), 1.0),  # full elevation + position -> max shading
        ):
            low_dhi = utils.calculate_shaded_window_irradiance(
                dni, np.array([10.0]), blind_position=position, blind_type=blind_type,
                solar_elevation_deg=elev,
            )
            high_dhi = utils.calculate_shaded_window_irradiance(
                dni, np.array([60.0]), blind_position=position, blind_type=blind_type,
                solar_elevation_deg=elev,
            )
            np.testing.assert_array_almost_equal(high_dhi - low_dhi, [50.0])

    def test_blind_position_below_zero_raises(self):
        with self.assertRaises(ValueError):
            utils.calculate_shaded_window_irradiance(
                np.array([100.0]), np.array([50.0]), blind_position=-0.1, blind_type="screen"
            )

    def test_blind_position_above_one_raises(self):
        with self.assertRaises(ValueError):
            utils.calculate_shaded_window_irradiance(
                np.array([100.0]), np.array([50.0]), blind_position=1.1, blind_type="screen"
            )

    def test_unrecognized_blind_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            utils.calculate_shaded_window_irradiance(
                np.array([100.0]), np.array([50.0]), blind_position=0.5, blind_type="curtain"
            )
        self.assertIn("curtain", str(ctx.exception))

    def test_mismatched_dni_dhi_length_raises(self):
        with self.assertRaises(ValueError):
            utils.calculate_shaded_window_irradiance(
                np.array([100.0, 200.0, 300.0]), np.array([50.0, 60.0]),
                blind_position=0.5, blind_type="none",
            )

    def test_mismatched_elevation_length_raises(self):
        with self.assertRaises(ValueError):
            utils.calculate_shaded_window_irradiance(
                np.array([100.0, 200.0]), np.array([50.0, 60.0]),
                blind_position=0.5, blind_type="awning",
                solar_elevation_deg=np.array([30.0]),
            )

    def test_invalid_elevation_threshold_order_raises(self):
        with self.assertRaises(ValueError):
            utils.calculate_shaded_window_irradiance(
                np.array([100.0]), np.array([50.0]), blind_position=0.5, blind_type="awning",
                solar_elevation_deg=np.array([30.0]),
                awning_elevation_low_deg=45.0,
                awning_elevation_high_deg=20.0,
            )

    def test_custom_elevation_thresholds_shift_the_ramp(self):
        dni = np.array([200.0])
        dhi = np.array([0.0])
        result = utils.calculate_shaded_window_irradiance(
            dni, dhi, blind_position=1.0, blind_type="awning",
            solar_elevation_deg=np.array([15.0]),
            awning_elevation_low_deg=0.0,
            awning_elevation_high_deg=30.0,
        )
        # 15 is the midpoint of a custom 0-30 range -> 50% blocked.
        np.testing.assert_array_almost_equal(result, [100.0])


class TestCalculateSurfaceSolarGain(unittest.TestCase):
    """Tests for the pool/outdoor-thermal-mass solar absorption helper."""

    def test_returns_none_when_absorption_area_unset(self):
        hc = {}  # no solar_absorption_area
        result = utils.calculate_surface_solar_gain(
            hc, np.array([500.0, 600.0]), optimization_time_step_minutes=30, length=2
        )
        self.assertIsNone(result)

    def test_returns_none_when_absorption_area_zero(self):
        hc = {"solar_absorption_area": 0.0}
        result = utils.calculate_surface_solar_gain(
            hc, np.array([500.0]), optimization_time_step_minutes=30, length=1
        )
        self.assertIsNone(result)

    def test_returns_none_when_ghi_forecast_none(self):
        hc = {"solar_absorption_area": 30.0}
        result = utils.calculate_surface_solar_gain(
            hc, None, optimization_time_step_minutes=30, length=4
        )
        self.assertIsNone(result)

    def test_computes_expected_gain(self):
        """100 W/m² over 30 min on a 30 m² pool at 0.7 absorption =
        100 * 30 * 0.7 / 1000 * 0.5 = 1.05 kWh per timestep."""
        hc = {"solar_absorption_area": 30.0, "solar_absorption_factor": 0.7}
        result = utils.calculate_surface_solar_gain(
            hc, np.array([100.0, 200.0]), optimization_time_step_minutes=30, length=2
        )
        np.testing.assert_array_almost_equal(result, np.array([1.05, 2.10]))

    def test_default_absorption_factor_is_0_7(self):
        hc = {"solar_absorption_area": 30.0}
        result = utils.calculate_surface_solar_gain(
            hc, np.array([1000.0]), optimization_time_step_minutes=60, length=1
        )
        # 1000 * 30 * 0.7 / 1000 * 1 = 21 kWh
        np.testing.assert_array_almost_equal(result, np.array([21.0]))

    def test_length_pads_with_zero(self):
        """When ghi is shorter than length, pad with zeros (no solar at night)."""
        hc = {"solar_absorption_area": 10.0, "solar_absorption_factor": 0.5}
        result = utils.calculate_surface_solar_gain(
            hc, np.array([200.0]), optimization_time_step_minutes=60, length=3
        )
        # First slot: 200*10*0.5/1000*1 = 1.0 kWh; remaining padded to zero
        np.testing.assert_array_almost_equal(result, np.array([1.0, 0.0, 0.0]))

    def test_length_truncates_excess(self):
        hc = {"solar_absorption_area": 10.0, "solar_absorption_factor": 1.0}
        ghi = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        result = utils.calculate_surface_solar_gain(
            hc, ghi, optimization_time_step_minutes=60, length=2
        )
        self.assertEqual(len(result), 2)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0]))

    def test_negative_absorption_factor_raises(self):
        hc = {"solar_absorption_area": 10.0, "solar_absorption_factor": -0.1}
        with self.assertRaises(ValueError) as ctx:
            utils.calculate_surface_solar_gain(
                hc, np.array([100.0]), optimization_time_step_minutes=60, length=1
            )
        self.assertIn(">= 0", str(ctx.exception))

    def test_zero_absorption_factor_returns_zero_array(self):
        """A fully-covered pool (factor=0) absorbs nothing."""
        hc = {"solar_absorption_area": 30.0, "solar_absorption_factor": 0.0}
        result = utils.calculate_surface_solar_gain(
            hc, np.array([500.0, 600.0]), optimization_time_step_minutes=30, length=2
        )
        np.testing.assert_array_almost_equal(result, np.zeros(2))


class TestApplyHeatingCurve(unittest.TestCase):
    """Heating-curve: T_supply = clip(offset - slope*T_outdoor, min, max)."""

    def test_linear_curve_negative_slope_clipped_high(self):
        """Cold outdoor should raise supply T, clipped at max_supply."""
        curve = {"slope": 1.5, "offset": 35.0, "min_supply": 25.0, "max_supply": 55.0}
        outdoor = np.array([-10.0, 0.0, 5.0, 15.0, 25.0])
        supply = utils.apply_heating_curve(curve, outdoor)
        # at -10: 35 + 15 = 50 (within bounds)
        # at  0: 35 (within bounds)
        # at  5: 27.5
        # at 15: 12.5 -> clipped to 25
        # at 25: -2.5 -> clipped to 25
        np.testing.assert_array_almost_equal(supply, [50.0, 35.0, 27.5, 25.0, 25.0])

    def test_curve_clipped_at_max(self):
        """Very cold outdoor should clip at max_supply."""
        curve = {"slope": 2.0, "offset": 40.0, "min_supply": 30.0, "max_supply": 50.0}
        outdoor = np.array([-20.0, -10.0, 0.0])
        supply = utils.apply_heating_curve(curve, outdoor)
        # at -20: 40 + 40 = 80 -> clipped to 50
        # at -10: 40 + 20 = 60 -> clipped to 50
        # at   0: 40 (within bounds)
        np.testing.assert_array_almost_equal(supply, [50.0, 50.0, 40.0])

    def test_default_bounds(self):
        """min_supply defaults to 25, max_supply to 70."""
        curve = {"slope": 1.0, "offset": 30.0}
        outdoor = np.array([-50.0, 50.0])
        supply = utils.apply_heating_curve(curve, outdoor)
        # -50: 30+50=80 -> clipped to 70 default
        #  50: 30-50=-20 -> clipped to 25 default
        np.testing.assert_array_almost_equal(supply, [70.0, 25.0])

    def test_inverted_bounds_raises(self):
        """min_supply >= max_supply is a config error."""
        curve = {"slope": 1.0, "offset": 30.0, "min_supply": 60.0, "max_supply": 40.0}
        with self.assertRaises(ValueError) as ctx:
            utils.apply_heating_curve(curve, np.array([5.0]))
        self.assertIn("min_supply", str(ctx.exception))

    def test_accepts_pandas_series(self):
        """pd.Series input should produce numpy output."""
        curve = {"slope": 1.0, "offset": 30.0, "min_supply": 25.0, "max_supply": 50.0}
        outdoor = pd.Series([0.0, 10.0, 20.0])
        supply = utils.apply_heating_curve(curve, outdoor)
        np.testing.assert_array_almost_equal(supply, [30.0, 25.0, 25.0])


class TestResolveMinTemperatures(unittest.TestCase):
    """Weather-compensated min buffer T floor (radiator emission floor)."""

    def test_static_only(self):
        """A config with only static `min_temperatures` returns it unchanged."""
        cfg = {"min_temperatures": [25.0] * 48}
        out = utils.resolve_min_temperatures(cfg, None, length=48)
        self.assertEqual(out, [25.0] * 48)

    def test_curve_only(self):
        """A config with only `min_temperature_curve` returns per-slot derived floor."""
        cfg = {
            "min_temperature_curve": {
                "slope": 1.0,
                "offset": 35.0,
                "min_supply": 30.0,
                "max_supply": 55.0,
            }
        }
        outdoor = np.array([-5.0, 0.0, 5.0, 15.0, 25.0])
        out = utils.resolve_min_temperatures(cfg, outdoor, length=5)
        # at -5: 35-(-5)=40 -> 40
        # at  0: 35 -> 35
        # at  5: 30 -> 30
        # at 15: 20 -> clipped to 30
        # at 25: 10 -> clipped to 30
        self.assertEqual(out, [40.0, 35.0, 30.0, 30.0, 30.0])

    def test_curve_and_static_max_wins_elementwise(self):
        """When both are set, element-wise max is taken (more conservative floor wins)."""
        cfg = {
            "min_temperatures": [20.0, 30.0, 40.0, 50.0, 60.0],
            "min_temperature_curve": {
                "slope": 1.0,
                "offset": 35.0,
                "min_supply": 30.0,
                "max_supply": 55.0,
            },
        }
        # curve at outdoor = -5,0,5,15,25 -> [40, 35, 30, 30, 30]
        # static                          -> [20, 30, 40, 50, 60]
        # elementwise max                 -> [40, 35, 40, 50, 60]
        outdoor = np.array([-5.0, 0.0, 5.0, 15.0, 25.0])
        out = utils.resolve_min_temperatures(cfg, outdoor, length=5)
        self.assertEqual(out, [40.0, 35.0, 40.0, 50.0, 60.0])

    def test_floor_of_30_via_curve_min_supply(self):
        """User-friendly pattern: curve with min_supply=30 keeps buffer at 30 even in summer."""
        cfg = {
            "min_temperature_curve": {
                "slope": 1.0,
                "offset": 35.0,
                "min_supply": 30.0,
                "max_supply": 50.0,
            }
        }
        # Summer outdoor: 20, 25, 30 °C - curve says 15, 10, 5 -> all clipped to 30
        out = utils.resolve_min_temperatures(cfg, np.array([20.0, 25.0, 30.0]), length=3)
        self.assertEqual(out, [30.0, 30.0, 30.0])

    def test_neither_set_returns_empty(self):
        """Tank with no min config returns empty list (caller raises)."""
        cfg = {}
        out = utils.resolve_min_temperatures(cfg, np.array([5.0]), length=1)
        self.assertEqual(out, [])

    def test_curve_without_outdoor_raises(self):
        """min_temperature_curve requires outdoor temperature input."""
        cfg = {"min_temperature_curve": {"slope": 1.0, "offset": 35.0}}
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_min_temperatures(cfg, None, length=1)
        self.assertIn("outdoor", str(ctx.exception))

    def test_short_static_list_padded(self):
        """Static list shorter than horizon is padded with its last value."""
        cfg = {"min_temperatures": [25.0, 28.0]}
        outdoor = None  # curve absent so outdoor not needed
        out = utils.resolve_min_temperatures(cfg, outdoor, length=4)
        self.assertEqual(out, [25.0, 28.0, 28.0, 28.0])

    def test_scalar_min_temperature_normalised(self):
        """A scalar (int / float) under the singular `min_temperature` key is
        accepted and treated as a one-element list (padded to horizon)."""
        # float scalar
        cfg = {"min_temperature": 20.0}
        out = utils.resolve_min_temperatures(cfg, None, length=3)
        self.assertEqual(out, [20.0, 20.0, 20.0])

        # int scalar
        cfg_int = {"min_temperature": 18}
        out_int = utils.resolve_min_temperatures(cfg_int, None, length=2)
        self.assertEqual(out_int, [18.0, 18.0])


class TestResolveThermalBatteryCopHeatingCurve(unittest.TestCase):
    """resolve_thermal_battery_cop with heating_curve."""

    def test_heating_curve_produces_per_slot_cop_variation(self):
        """COP should differ between cold and mild slots when heating curve drops supply T."""
        hc = {
            "heating_curve": {"slope": 1.0, "offset": 30.0, "min_supply": 25.0, "max_supply": 55.0},
            "carnot_efficiency": 0.45,
        }
        # Cold morning, mild noon
        outdoor = np.array([-5.0, -5.0, 0.0, 5.0, 10.0, 15.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor)
        # At -5 outdoor: supply = 35, ΔT = 40 -> COP = 0.45 * 308 / 40 ≈ 3.47
        # At 15 outdoor: supply = 25 (clipped), ΔT = 10 -> COP = 0.45 * 298 / 10 ≈ 13.4 -> capped at 8.0
        # COP should increase as outdoor rises (closer to supply T)
        self.assertLess(cops[0], cops[-1])
        self.assertGreater(cops[-1], 5.0)  # mild day, high COP

    def test_heating_curve_takes_precedence_over_constant(self):
        """If both heating_curve and supply_temperature are set, heating_curve wins."""
        hc = {
            "heating_curve": {"slope": 1.0, "offset": 30.0, "min_supply": 25.0, "max_supply": 55.0},
            "supply_temperature": 55.0,  # would give much lower COP - should be ignored
            "carnot_efficiency": 0.4,
        }
        outdoor = np.array([10.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor)
        # heating_curve at 10 outdoor: supply = 25 (clipped), so high COP
        # If supply_temperature=55 had been used: COP = 0.4 * 328 / 45 ≈ 2.92
        self.assertGreater(cops[0], 4.0)

    def test_no_heating_curve_falls_back_to_supply_temperature(self):
        """Backward compatibility: configs without heating_curve still work."""
        hc = {"supply_temperature": 55.0, "carnot_efficiency": 0.4}
        outdoor = np.array([5.0, 10.0])
        cops = utils.resolve_thermal_battery_cop(hc, outdoor)
        # At 5°C outdoor, 55°C supply: COP = 0.4 * 328.15 / 50 ≈ 2.625
        self.assertAlmostEqual(cops[0], 2.625, places=2)

    def test_missing_both_raises_with_clear_message(self):
        """When neither supply_temperature nor heating_curve nor efficiency is set, raise."""
        hc = {"carnot_efficiency": 0.4}
        with self.assertRaises(ValueError) as ctx:
            utils.resolve_thermal_battery_cop(hc, np.array([5.0]))
        msg = str(ctx.exception)
        self.assertIn("supply_temperature", msg)
        self.assertIn("heating_curve", msg)


class TestCompileHeatTopology(unittest.TestCase):
    """Tests for the graph -> primitives compiler."""

    def test_minimal_single_source_single_storage(self):
        topo = {
            "sources": [
                {
                    "id": "boiler",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "cost_track": "gas",
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.05,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.06,
                }
            ],
            "flows": [{"from": "boiler", "to": "buf"}],
            "cost_tracks": {"gas": [0.085] * 48},
        }
        out = utils.compile_heat_topology(topo)
        self.assertEqual(out["number_of_deferrable_loads"], 1)
        self.assertEqual(out["nominal_power_of_deferrable_loads"], [25000.0])
        self.assertEqual(out["minimum_power_of_deferrable_loads"], [8000.0])
        self.assertEqual(out["def_load_config"][0]["thermal_source"]["efficiency"], 0.92)
        self.assertEqual(out["shared_thermal_tanks"][0]["id"], "buf")
        self.assertEqual(out["shared_thermal_tanks"][0]["load_ids"], [0])
        self.assertEqual(out["cost_forecast_per_deferrable_load"][0], [0.085] * 48)

    def test_two_sources_one_storage(self):
        """HP + gas both feed the same DHW tank."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 55,
                    "carnot_efficiency": 0.40,
                    "nominal_power": 3500,
                    "min_power": 800,
                    "cost_track": "retail",
                },
                {
                    "id": "gas",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "cost_track": "gas_flat",
                },
            ],
            "storage": [
                {
                    "id": "dhw",
                    "volume": 0.20,
                    "start_temperature": 51,
                    "min_temperature": [48] * 48,
                    "max_temperature": [62] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "consumers": [
                {
                    "id": "drw",
                    "type": "profile",
                    "target": "dhw",
                    "profile": [0] * 14 + [1.0] + [0] * 33,
                }
            ],
            "flows": [
                {"from": "hp", "to": "dhw"},
                {"from": "gas", "to": "dhw"},
            ],
            "cost_tracks": {"retail": [0.25] * 48, "gas_flat": [0.085] * 48},
        }
        out = utils.compile_heat_topology(topo)
        self.assertEqual(out["number_of_deferrable_loads"], 2)
        # Tank has both loads
        self.assertEqual(out["shared_thermal_tanks"][0]["load_ids"], [0, 1])
        # Draw profile passed through
        self.assertEqual(sum(out["shared_thermal_tanks"][0]["draw_off_demand"]), 1.0)
        # Per-source cost tracks
        self.assertEqual(out["cost_forecast_per_deferrable_load"][0][0], 0.25)
        self.assertEqual(out["cost_forecast_per_deferrable_load"][1][0], 0.085)

    def test_actuator_group_emits_deferrable_group(self):
        """One physical boiler serving two tanks via mutex."""
        topo = {
            "sources": [
                {
                    "id": "g",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "cost_track": "gas",
                }
            ],
            "storage": [
                {
                    "id": "dhw",
                    "volume": 0.2,
                    "start_temperature": 50,
                    "min_temperature": [45] * 48,
                    "max_temperature": [60] * 48,
                    "thermal_loss": 0.05,
                },
                {
                    "id": "buf",
                    "volume": 0.1,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.06,
                },
            ],
            "flows": [
                {"from": "g", "to": "dhw"},
                {"from": "g", "to": "buf"},
            ],
            "actuator_groups": [
                {
                    "flows": [["g", "dhw"], ["g", "buf"]],
                    "mutual_exclusion": True,
                    "max_combined_power": 25000,
                }
            ],
            "cost_tracks": {"gas": [0.085] * 48},
        }
        out = utils.compile_heat_topology(topo)
        self.assertEqual(len(out["deferrable_load_groups"]), 1)
        g = out["deferrable_load_groups"][0]
        self.assertEqual(set(g["names"]), {"deferrable0", "deferrable1"})
        self.assertTrue(g["mutual_exclusion"])
        self.assertEqual(g["max_power"], 25000.0)

    def test_two_profile_consumers_on_same_storage_pad_to_max_length(self):
        """When two profile consumers target the same storage with different
        profile lengths, the merged profile must preserve the LONGER input
        instead of silently truncating to the first profile's length."""
        topo = {
            "sources": [
                {
                    "id": "g",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "cost_track": "gas",
                }
            ],
            "storage": [
                {
                    "id": "dhw",
                    "volume": 0.2,
                    "start_temperature": 50,
                    "min_temperature": [45] * 48,
                    "max_temperature": [60] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "consumers": [
                # First profile: 24 slots
                {"id": "morning", "type": "profile", "target": "dhw", "profile": [0.1] * 24},
                # Second profile: 48 slots - must NOT be truncated to 24.
                {"id": "evening", "type": "profile", "target": "dhw", "profile": [0.05] * 48},
            ],
            "flows": [{"from": "g", "to": "dhw"}],
            "cost_tracks": {"gas": [0.085] * 48},
        }
        out = utils.compile_heat_topology(topo)
        merged = out["shared_thermal_tanks"][0]["draw_off_demand"]
        self.assertEqual(len(merged), 48)
        # First 24 slots: 0.1 + 0.05 = 0.15
        self.assertAlmostEqual(merged[0], 0.15)
        self.assertAlmostEqual(merged[23], 0.15)
        # Slots 24..47: only the second profile contributes
        self.assertAlmostEqual(merged[24], 0.05)
        self.assertAlmostEqual(merged[47], 0.05)

    def test_unknown_source_type_error_includes_id_and_index(self):
        """The error for an unrecognised source type must include both the
        offending source's id AND its index in topology.sources to aid
        diagnosis when multiple sources are present."""
        topo = {
            "sources": [
                {
                    "id": "ok",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                },
                {
                    "id": "bad",
                    "type": "nuclear",
                    "efficiency": 0.99,
                    "nominal_power": 1000000,
                    "min_power": 10000,
                },
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.1,
                    "start_temperature": 30,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [
                {"from": "ok", "to": "buf"},
                {"from": "bad", "to": "buf"},
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        msg = str(ctx.exception)
        self.assertIn("'bad'", msg)
        self.assertIn("[1]", msg)
        self.assertIn("nuclear", msg)

    def test_unknown_source_id_in_flow_raises(self):
        topo = {
            "sources": [
                {
                    "id": "boiler",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.1,
                    "start_temperature": 30,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "WRONG", "to": "buf"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        self.assertIn("WRONG", str(ctx.exception))

    def test_unknown_storage_id_in_consumer_raises(self):
        topo = {
            "sources": [
                {
                    "id": "b",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                }
            ],
            "storage": [
                {
                    "id": "ok",
                    "volume": 0.1,
                    "start_temperature": 30,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "consumers": [{"id": "x", "type": "profile", "target": "GHOST", "profile": [0] * 48}],
            "flows": [{"from": "b", "to": "ok"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        self.assertIn("GHOST", str(ctx.exception))

    def test_electric_flag_auto_default_by_source_type(self):
        """Source type 'gas' / 'oil' / 'district' defaults electric=False;
        type 'heatpump' / 'electric' defaults electric=True."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 55,
                    "carnot_efficiency": 0.40,
                    "nominal_power": 3500,
                    "min_power": 800,
                },
                {
                    "id": "gas",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 8000,
                },
                {
                    "id": "oil",
                    "type": "oil",
                    "efficiency": 0.88,
                    "nominal_power": 30000,
                    "min_power": 10000,
                },
                {
                    "id": "dh",
                    "type": "district",
                    "efficiency": 0.95,
                    "nominal_power": 15000,
                    "min_power": 5000,
                },
                {
                    "id": "el",
                    "type": "electric",
                    "efficiency": 1.0,
                    "nominal_power": 2000,
                    "min_power": 0,
                },
            ],
            "storage": [
                {
                    "id": "tank",
                    "volume": 0.2,
                    "start_temperature": 50,
                    "min_temperature": [45] * 48,
                    "max_temperature": [62] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": s, "to": "tank"} for s in ("hp", "gas", "oil", "dh", "el")],
        }
        out = utils.compile_heat_topology(topo)
        self.assertEqual(
            out["is_electric_load"],
            [True, False, False, False, True],
            "Expected HP/electric → True, gas/oil/district → False",
        )

    def test_electric_flag_explicit_override_wins(self):
        """An explicit `electric: true|false` on a source overrides the type default."""
        topo = {
            "sources": [
                # Gas but flagged as electric (e.g., gas-fired heat pump with
                # large parasitic electric draw - hypothetical override case)
                {
                    "id": "weird",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "electric": True,
                },
            ],
            "storage": [
                {
                    "id": "tank",
                    "volume": 0.2,
                    "start_temperature": 50,
                    "min_temperature": [45] * 48,
                    "max_temperature": [62] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "weird", "to": "tank"}],
        }
        out = utils.compile_heat_topology(topo)
        self.assertEqual(out["is_electric_load"], [True])

    def test_storage_soft_comfort_fields_propagate(self):
        """desired_temperatures + overshoot_temperature + penalty_factor + comfort_sense
        on a storage block should flow through to the compiled shared_thermal_tank."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 35,
                    "carnot_efficiency": 0.4,
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buffer",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                    "desired_temperatures": [30.0] * 48,
                    "overshoot_temperature": 40.0,
                    "penalty_factor": 5.0,
                    "comfort_sense": "heat",
                }
            ],
            "flows": [{"from": "hp", "to": "buffer"}],
        }
        out = utils.compile_heat_topology(topo)
        tank = out["shared_thermal_tanks"][0]
        self.assertEqual(tank["desired_temperatures"], [30.0] * 48)
        self.assertEqual(tank["overshoot_temperature"], 40.0)
        self.assertEqual(tank["penalty_factor"], 5.0)
        self.assertEqual(tank["sense"], "heat")

    def test_storage_comfort_sense_propagates_to_source(self):
        """A heat-pump source feeding a `comfort_sense: cool` storage must receive
        sense="cool" on its compiled thermal_source block, so
        resolve_thermal_battery_cop computes the cooling COP instead of defaulting
        to "heat" and clamping the COP to 1.0 on a warm day (outdoor >= supply)."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 18,
                    "carnot_efficiency": 0.35,
                    "nominal_power": 2100,
                    "min_power": 250,
                }
            ],
            "storage": [
                {
                    "id": "zone",
                    "volume": 0.2,
                    "start_temperature": 24,
                    "min_temperatures": [10] * 48,
                    "max_temperatures": [28] * 48,
                    "comfort_sense": "cool",
                }
            ],
            "flows": [{"from": "hp", "to": "zone"}],
        }
        out = utils.compile_heat_topology(topo)
        source = out["def_load_config"][0]["thermal_source"]
        self.assertEqual(source.get("sense"), "cool")
        # On a warm day the cooling COP must not collapse to the clamped 1.0.
        cop = utils.resolve_thermal_battery_cop(source, [31.0] * 48, length=48)
        self.assertGreater(cop[0], 1.5)

    def test_storage_soft_comfort_scalar_desired_temperature(self):
        """Scalar `desired_temperature` is accepted and stored as a float."""
        topo = {
            "sources": [
                {
                    "id": "g",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 4000,
                }
            ],
            "storage": [
                {
                    "id": "buffer",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                    "desired_temperature": 35.0,
                    "overshoot_temperature": 45.0,
                }
            ],
            "flows": [{"from": "g", "to": "buffer"}],
        }
        out = utils.compile_heat_topology(topo)
        tank = out["shared_thermal_tanks"][0]
        self.assertEqual(tank["desired_temperatures"], 35.0)
        self.assertEqual(tank["overshoot_temperature"], 45.0)

    def test_storage_soft_comfort_omitted_does_not_add_fields(self):
        """When no soft-comfort fields are set, nothing extra is added to the tank."""
        topo = {
            "sources": [
                {
                    "id": "g",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 4000,
                }
            ],
            "storage": [
                {
                    "id": "buffer",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "g", "to": "buffer"}],
        }
        out = utils.compile_heat_topology(topo)
        tank = out["shared_thermal_tanks"][0]
        self.assertNotIn("desired_temperatures", tank)
        self.assertNotIn("overshoot_temperature", tank)
        self.assertNotIn("penalty_factor", tank)
        self.assertNotIn("sense", tank)

    def test_storage_comfort_sense_invalid_raises(self):
        """An invalid comfort_sense should raise ValueError."""
        topo = {
            "sources": [
                {
                    "id": "g",
                    "type": "gas",
                    "efficiency": 0.92,
                    "nominal_power": 25000,
                    "min_power": 4000,
                }
            ],
            "storage": [
                {
                    "id": "buffer",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                    "comfort_sense": "freeze",
                }
            ],
            "flows": [{"from": "g", "to": "buffer"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        self.assertIn("comfort_sense", str(ctx.exception))

    def test_heating_curve_propagates_through_compiler(self):
        """`heating_curve` on a heatpump source should flow through to def_load_config."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "heating_curve": {
                        "slope": 1.0,
                        "offset": 30.0,
                        "min_supply": 28.0,
                        "max_supply": 50.0,
                    },
                    "carnot_efficiency": 0.45,
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        out = utils.compile_heat_topology(topo)
        ts = out["def_load_config"][0]["thermal_source"]
        self.assertIn("heating_curve", ts)
        self.assertEqual(ts["heating_curve"]["slope"], 1.0)
        self.assertEqual(ts["heating_curve"]["offset"], 30.0)
        self.assertEqual(ts["heating_curve"]["min_supply"], 28.0)
        self.assertEqual(ts["heating_curve"]["max_supply"], 50.0)
        # Constant supply_temperature should NOT be set when heating_curve is given
        self.assertNotIn("supply_temperature", ts)
        # Carnot efficiency still propagates
        self.assertEqual(ts["carnot_efficiency"], 0.45)

    def test_heating_curve_missing_required_field_raises(self):
        """heating_curve must include slope and offset."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "heating_curve": {"slope": 1.0},  # missing offset
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        self.assertIn("heating_curve", str(ctx.exception))
        self.assertIn("offset", str(ctx.exception))

    def test_heatpump_without_supply_or_curve_raises(self):
        """A heatpump source must specify supply_temperature or heating_curve."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        msg = str(ctx.exception)
        self.assertIn("supply_temperature", msg)
        self.assertIn("heating_curve", msg)

    def test_min_temperature_curve_propagates(self):
        """min_temperature_curve on storage should flow through to the compiled tank."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 55,
                    "carnot_efficiency": 0.4,
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [60] * 48,
                    "thermal_loss": 0.06,
                    "min_temperature_curve": {
                        "slope": 1.0,
                        "offset": 35,
                        "min_supply": 30,
                        "max_supply": 55,
                    },
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        out = utils.compile_heat_topology(topo)
        tank = out["shared_thermal_tanks"][0]
        self.assertIn("min_temperature_curve", tank)
        self.assertEqual(tank["min_temperature_curve"]["slope"], 1.0)
        self.assertEqual(tank["min_temperature_curve"]["offset"], 35.0)
        self.assertEqual(tank["min_temperature_curve"]["min_supply"], 30.0)
        self.assertEqual(tank["min_temperature_curve"]["max_supply"], 55.0)
        # Static absolute floor is still there for safety
        self.assertEqual(tank["min_temperatures"], [20] * 48)

    def test_min_temperature_curve_missing_slope_raises(self):
        """min_temperature_curve must include slope + offset."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 55,
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [20] * 48,
                    "max_temperature": [60] * 48,
                    "thermal_loss": 0.06,
                    "min_temperature_curve": {"offset": 35},  # missing slope
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        msg = str(ctx.exception)
        self.assertIn("min_temperature_curve", msg)
        self.assertIn("slope", msg)

    def test_constant_supply_temperature_still_works(self):
        """Back-compat: heatpump with supply_temperature only (no curve) still compiles."""
        topo = {
            "sources": [
                {
                    "id": "hp",
                    "type": "heatpump",
                    "supply_temperature": 55.0,
                    "carnot_efficiency": 0.4,
                    "nominal_power": 10000,
                    "min_power": 1000,
                }
            ],
            "storage": [
                {
                    "id": "buf",
                    "volume": 0.2,
                    "start_temperature": 35,
                    "min_temperature": [25] * 48,
                    "max_temperature": [55] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "hp", "to": "buf"}],
        }
        out = utils.compile_heat_topology(topo)
        ts = out["def_load_config"][0]["thermal_source"]
        self.assertEqual(ts["supply_temperature"], 55.0)
        self.assertNotIn("heating_curve", ts)

    def test_cost_track_not_found_raises(self):
        topo = {
            "sources": [
                {
                    "id": "b",
                    "type": "gas",
                    "efficiency": 0.9,
                    "nominal_power": 25000,
                    "min_power": 8000,
                    "cost_track": "missing",
                }
            ],
            "storage": [
                {
                    "id": "ok",
                    "volume": 0.1,
                    "start_temperature": 30,
                    "min_temperature": [25] * 48,
                    "max_temperature": [50] * 48,
                    "thermal_loss": 0.05,
                }
            ],
            "flows": [{"from": "b", "to": "ok"}],
            "cost_tracks": {"gas": [0.085] * 48},
        }
        with self.assertRaises(ValueError) as ctx:
            utils.compile_heat_topology(topo)
        self.assertIn("missing", str(ctx.exception))

    def test_compile_heat_topology_rejects_non_dict(self):
        """Non-dict inputs (string "null", None, "") must return {} without raising."""
        self.assertEqual(utils.compile_heat_topology("null"), {})
        self.assertEqual(utils.compile_heat_topology(None), {})
        self.assertEqual(utils.compile_heat_topology(""), {})

    def test_compile_heat_topology_rejects_empty_dict(self):
        """Empty dict must return {} without raising."""
        self.assertEqual(utils.compile_heat_topology({}), {})


class TestRuntimeBanner(unittest.TestCase):
    def test_log_runtime_banner_logs_info(self):
        from emhass.utils import log_runtime_banner

        test_logger = logging.getLogger("emhass-test-banner")
        with self.assertLogs("emhass-test-banner", level="INFO") as cm:
            log_runtime_banner(test_logger)
        self.assertEqual(len(cm.output), 1, f"Expected one INFO record, got {len(cm.output)}")
        msg = cm.records[0].getMessage()
        self.assertRegex(
            msg,
            r"^EMHASS \S+ \| Python \S+ \| CVXPY \S+ \(\S+\) \| \S+-\S+$",
            f"Banner format mismatch: {msg!r}",
        )

    def test_log_runtime_banner_survives_introspection_failure(self):
        import unittest.mock

        from emhass.utils import log_runtime_banner

        test_logger = logging.getLogger("emhass-test-banner-fail")
        with unittest.mock.patch(
            "cvxpy.installed_solvers",
            side_effect=RuntimeError("simulated solver-introspection failure"),
        ):
            with self.assertLogs("emhass-test-banner-fail", level="INFO") as cm:
                log_runtime_banner(test_logger)  # must not raise
        self.assertEqual(len(cm.output), 1)
        self.assertIn("runtime info unavailable", cm.records[0].getMessage())

    def test_log_runtime_banner_uses_active_solver_from_optim_conf(self):
        from emhass.utils import log_runtime_banner

        test_logger = logging.getLogger("emhass-test-banner-active")
        with self.assertLogs("emhass-test-banner-active", level="INFO") as cm:
            log_runtime_banner(test_logger, optim_conf={"lp_solver": "COIN_CMD"})
        self.assertEqual(len(cm.output), 1, f"Expected one INFO record, got {len(cm.output)}")
        msg = cm.records[0].getMessage()
        self.assertIn("COIN_CMD", msg, f"Expected active solver in banner: {msg!r}")
        self.assertRegex(
            msg,
            r"^EMHASS \S+ \| Python \S+ \| CVXPY \S+ \(COIN_CMD\) \| \S+-\S+$",
            f"Banner format mismatch: {msg!r}",
        )

    def test_log_runtime_banner_defaults_to_highs_when_key_missing(self):
        # Mirrors optimization.py default: when lp_solver is not set in optim_conf,
        # the LP uses "Highs". Banner must match reality.
        from emhass.utils import log_runtime_banner

        test_logger = logging.getLogger("emhass-test-banner-default")
        with self.assertLogs("emhass-test-banner-default", level="INFO") as cm:
            log_runtime_banner(test_logger, optim_conf={})
        self.assertEqual(len(cm.output), 1, f"Expected one INFO record, got {len(cm.output)}")
        msg = cm.records[0].getMessage()
        self.assertIn("Highs", msg, f"Expected default Highs in banner: {msg!r}")

    def test_log_runtime_banner_double_fallback_when_version_lookup_fails(self):
        # Covers the inner except: outer introspection AND importlib.metadata.version
        # both fail. Banner must still emit one INFO and not raise.
        import unittest.mock

        from emhass.utils import log_runtime_banner

        test_logger = logging.getLogger("emhass-test-banner-double-fail")
        with unittest.mock.patch(
            "cvxpy.installed_solvers",
            side_effect=RuntimeError("primary failure"),
        ):
            with unittest.mock.patch(
                "importlib.metadata.version",
                side_effect=RuntimeError("version lookup failure"),
            ):
                with self.assertLogs("emhass-test-banner-double-fail", level="INFO") as cm:
                    log_runtime_banner(test_logger)  # must not raise
        self.assertEqual(len(cm.output), 1)
        self.assertIn("runtime info unavailable", cm.records[0].getMessage())


class TestResolveIncrementalSeries(unittest.TestCase):
    """Tests for the cumulative-meter-to-delta auto-detection helper used to
    turn a raw HA gas/energy totalizer into a per-interval consumption
    series before it's used as a refit's fit target."""

    def test_cumulative_meter_is_converted_to_delta(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        # Monotonically non-decreasing, like a real lifetime gas totalizer.
        raw = pd.Series([2011.900, 2011.910, 2011.912, 2011.930, 2011.935])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), [0.0, 0.010, 0.002, 0.018, 0.005]
        )

    def test_already_incremental_series_is_returned_unchanged(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        # Fluctuates constantly (heating cycling on/off) - already a delta.
        raw = pd.Series([0.0, 0.4, 0.0, 0.6, 0.1, 0.0, 0.3, 0.0, 0.5, 0.2])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        np.testing.assert_array_almost_equal(result.to_numpy(), raw.to_numpy())

    def test_meter_reset_is_clipped_to_zero_not_negative(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        # A single reset dip amid an otherwise steadily-rising counter - long
        # enough that the one reset stays a small minority of diffs (2%),
        # matching how a real, mostly-monotonic meter with an occasional
        # reset would look over many samples.
        raw = pd.Series([100.0 + 0.5 * i for i in range(24)] + [5.0 + 0.5 * i for i in range(24)])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        self.assertTrue((result >= 0.0).all())
        # The reset step itself (111.5 -> 5.0) must be clipped to 0, not -106.5.
        self.assertEqual(result.iloc[24], 0.0)

    def test_first_value_is_always_zero_for_a_converted_series(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([50.0, 50.2, 50.5, 50.9])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        self.assertEqual(result.iloc[0], 0.0)

    def test_short_series_returned_unchanged(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([10.0, 10.5])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        np.testing.assert_array_almost_equal(result.to_numpy(), raw.to_numpy())

    def test_constant_series_returned_unchanged_not_all_zero_delta(self):
        # A flat series (e.g. a steady instantaneous power reading, or no
        # InfluxDB variation at all) has no negative diffs, but also never
        # once increases - a real cumulative counter still ticks up
        # occasionally over a long window, so "never increases" must NOT be
        # treated as cumulative (that would wrongly zero out an
        # already-correct constant rate/instantaneous reading).
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([42.0, 42.0, 42.0, 42.0, 42.0])
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        np.testing.assert_array_almost_equal(result.to_numpy(), raw.to_numpy())

    def test_negative_fraction_threshold_is_respected(self):
        logger = logging.getLogger("emhass-test-resolve-incremental")
        # Exactly at the boundary: only one clearly-negative diff among many
        # positive ones should still count as "mostly cumulative" and convert.
        raw = pd.Series([1.0] + list(range(2, 41)) + [1.0] + list(range(2, 41)))
        result = utils.resolve_incremental_series(raw, "gas_consumption", logger)
        # Converted (not identical to raw), since negative fraction is tiny.
        self.assertFalse(np.allclose(result.to_numpy(), raw.to_numpy()))

    def test_rate_dt_hours_converts_cumulative_kwh_to_average_power_w(self):
        # A cumulative electricity meter in kWh, rising by 0.5 kWh every
        # 30-minute (0.5h) step -> average power should come out to 1000 W.
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([100.0, 100.5, 101.0, 101.5, 102.0])
        result = utils.resolve_incremental_series(
            raw, "electric_power", logger, rate_dt_hours=0.5
        )
        np.testing.assert_array_almost_equal(
            result.to_numpy(), [0.0, 1000.0, 1000.0, 1000.0, 1000.0]
        )

    def test_rate_dt_hours_ignored_when_not_detected_as_cumulative(self):
        # Already-instantaneous power fluctuating around a level - must be
        # returned completely unchanged, not scaled by rate_dt_hours.
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([300.0, 280.0, 310.0, 290.0, 305.0, 295.0])
        result = utils.resolve_incremental_series(
            raw, "electric_power", logger, rate_dt_hours=0.5
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), raw.to_numpy())

    def test_rate_dt_hours_ignored_for_constant_power_reading(self):
        # A steady instantaneous power reading (e.g. heat pump running at a
        # fixed duty for a while) must stay exactly as-is, not get zeroed
        # out by being mistaken for a stalled cumulative counter.
        logger = logging.getLogger("emhass-test-resolve-incremental")
        raw = pd.Series([300.0, 300.0, 300.0, 300.0, 300.0])
        result = utils.resolve_incremental_series(
            raw, "electric_power", logger, rate_dt_hours=0.5
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), raw.to_numpy())


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
