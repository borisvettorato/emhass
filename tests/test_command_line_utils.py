#!/usr/bin/env python

import asyncio
import copy
import json
import os
import pathlib
import pickle
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.command_line import (
    _CANDIDATE_OPENING_EVENT_MAX_PER_ROOM,
    OptimizationCache,
    OptimizationCacheKey,
    PublishContext,
    SetupContext,
    _apply_df_freq_horizon,
    _apply_manual_load_runtime_overrides,
    _build_room_blind_positions,
    _build_room_door_open,
    _build_room_kalman_opening_open,
    _build_room_opening_open,
    _build_room_opening_open_with_kalman_fallback,
    _em_relabel_opening_open,
    _expand_confirmed_ranges_to_timestamps,
    _extract_contiguous_open_events,
    _format_manual_load_action,
    _load_opt_res_latest,
    _maybe_record_manual_load_commitments,
    _next_deadline_timestamp,
    _prepare_dayahead_optim,
    _publish_and_update_freq,
    _publish_manual_load_actions,
    _publish_opening_confirmation_questions,
    _resolve_opening_confirmations,
    _resolve_room_blind_entity_map,
    _resolve_room_door_entity_map,
    _resolve_room_window_entity_map,
    _slugify_room_name,
    _timestep_index_from_timestamp,
    _translate_ev_power_to_mode,
    adjust_pv_forecast,
    compute_heating_forecast,
    compute_hybrid_heatpump_forecast,
    compute_self_learning_physics_forecast,
    continual_publish,
    dayahead_forecast_optim,
    export_influxdb_to_csv,
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    is_model_outdated,
    main,
    naive_mpc_optim,
    perfect_forecast_optim,
    prepare_forecast_and_weather_data,
    publish_data,
    publish_json,
    refit_heating_model,
    refit_hybrid_heatpump_model,
    refit_self_learning_physics_model,
    regressor_model_fit,
    regressor_model_predict,
    retrieve_home_assistant_data,
    set_input_data_dict,
)
from emhass.forecast import Forecast

# Sentinel distinguishing "no washdata_device argument given" (don't
# configure load_washdata_device at all) from "configured, but no matching
# entities were discovered" (an explicit fallback-path test case, passed as
# washdata_states=[]).
_UNSET = object()


def _washdata_program_state(device, program_slug, power_profile, interval_min, count):
    """Build a fake /api/states entry matching WashData's ha_washdata naming
    convention (sensor.<device>_profiel_<program>_aantal), for mocking
    RetrieveHass.get_all_states() in the discovery tests below."""
    return {
        "entity_id": f"sensor.{device}_profiel_{program_slug}_aantal",
        "state": str(count),
        "attributes": {
            "power_profile": power_profile,
            "power_profile_interval_min": interval_min,
        },
    }

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

rng = np.random.default_rng()


class TestCommandLineAsyncUtils(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params(set_use_pv=False):
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
            if set_use_pv:
                params["optim_conf"]["set_use_pv"] = True
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        # Add runtime parameters for forecast lists
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        self.runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        self.params_json = orjson.dumps(params).decode("utf-8")

    # Test input data for actions (using data from file)
    async def test_set_input_data_dict(self):
        costfun = "profit"
        # Test dayahead
        action = "dayahead-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertIs(input_data_dict["df_input_data"], None)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(input_data_dict["fcst"].optim_conf["weather_forecast_method"], "list")
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_forecast_method"], "list")
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
        )
        # Test publish data
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIs(input_data_dict["df_input_data"], None)
        self.assertIs(input_data_dict["df_input_data_dayahead"], None)
        self.assertIs(input_data_dict["p_pv_forecast"], None)
        self.assertIs(input_data_dict["p_load_forecast"], None)
        # Test naive mpc
        action = "naive-mpc-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
        )  # The default value for prediction_horizon
        # Test Naive mpc with a shorter forecast =
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "load_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "load_cost_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prod_price_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prediction_horizon": 10,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
        )  # The default value for prediction_horizon
        # Test naive mpc with a shorter forecast and prediction horizon = 10
        action = "naive-mpc-optim"
        runtimeparams["prediction_horizon"] = 10
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
        )  # The fixed value for prediction_horizon
        # Test passing just load cost and prod price as lists
        action = "dayahead-optim"
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
        )

    # Test day-ahead optimization
    async def test_webserver_get_injection_dict(self):
        costfun = "profit"
        action = "dayahead-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Create a dummy result matching the index of the input
        mock_res = pd.DataFrame(index=input_data_dict["df_input_data_dayahead"].index)
        mock_res["p_grid"] = 0.0
        mock_res["p_pv"] = 0.0
        mock_res["cost_fun_profit"] = 0.0
        mock_res["optim_status"] = "Optimal"
        input_data_dict["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        injection_dict = utils.get_injection_dict(opt_res)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["table1"], str)
        self.assertIsInstance(injection_dict["table2"], str)

    # Test data formatting of dayahead optimization with load cost and prod price as lists
    async def test_dayahead_forecast_optim(self):
        # Test dataframe output of profit dayahead optimization
        costfun = "profit"
        action = "dayahead-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        mock_res = pd.DataFrame(index=input_data_dict["df_input_data_dayahead"].index)
        # We need to populate columns that might be checked or used
        mock_res["p_grid"] = 0.0
        mock_res["p_pv"] = 0.0
        input_data_dict["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), len(params["passed_data"]["pv_power_forecast"]))
        # Test dayahead output, passing just load cost and prod price as runtime lists (costfun=profit)
        action = "dayahead-optim"
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # This specific test checks if unit_load_cost matches the passed list,
        # so we must populate it in our mock.
        mock_res_2 = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="30min"))
        mock_res_2["unit_load_cost"] = runtimeparams["load_cost_forecast"]
        mock_res_2["unit_prod_price"] = runtimeparams["prod_price_forecast"]
        mock_res_2["p_grid"] = 0.0
        input_data_dict["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res_2)
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
        )
        self.assertEqual(
            opt_res["unit_load_cost"].values.tolist(),
            runtimeparams["load_cost_forecast"],
        )
        self.assertEqual(
            opt_res["unit_prod_price"].values.tolist(),
            runtimeparams["prod_price_forecast"],
        )
        # Test dayahead output, using set_use_adjusted_pv = True
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["set_use_adjusted_pv"] = True
        params["optim_conf"]["set_use_pv"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Re-use the simple mock from Pass 1 logic
        input_data_dict["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)

    async def test_dayahead_forecast_optim_passes_kalman_merged_room_opening_open(self):
        """The solver call must receive whatever
        _build_room_opening_open_with_kalman_fallback returns for
        room_opening_open - not the bare sensor-only builder."""
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            self.params_json,
            self.runtimeparams_json,
            "dayahead-optim",
            logger,
            get_data_from_file=True,
        )
        mock_res = pd.DataFrame(index=input_data_dict["df_input_data_dayahead"].index)
        mock_res["p_grid"] = 0.0
        mock_res["p_pv"] = 0.0
        input_data_dict["opt"].perform_dayahead_forecast_optim = MagicMock(return_value=mock_res)
        sentinel = [True, False]

        with patch(
            "emhass.command_line._build_room_opening_open_with_kalman_fallback",
            AsyncMock(return_value=sentinel),
        ) as mock_builder:
            await dayahead_forecast_optim(input_data_dict, logger, debug=True)

        mock_builder.assert_awaited_once()
        call_kwargs = input_data_dict["opt"].perform_dayahead_forecast_optim.call_args.kwargs
        self.assertEqual(call_kwargs["room_opening_open"], sentinel)

    # Test dataframe output of perfect forecast optimization
    async def test_perfect_forecast_optim(self):
        costfun = "profit"
        action = "perfect-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        with self.assertLogs(logger, level="INFO") as cm:
            opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertIn("cost_fun_" + input_data_dict["costfun"], opt_res.columns)
        self.assertIn(
            "Optimization completed in",
            "\n".join(cm.output),
            "Summary line missing — expected one INFO record from orchestrator",
        )

    # Test naive-mpc with prediction_horizon=72 auto-extends delta_forecast_daily to 2 days
    async def test_naive_mpc_autoextends_horizon_end_to_end(self):
        """Integration test: prediction_horizon=72 (36 h, 2 days at 30-min steps) with
        NO delta_forecast_daily override must cause set_input_data_dict to bump
        delta_forecast_daily from 1 → 2 days, give 72 forecast_dates, and let
        naive_mpc_optim return a 72-row result with no NaNs."""
        costfun = "profit"
        action = "naive-mpc-optim"
        # 72 elements: forecast lists must cover the full 72-step horizon
        runtimeparams = {
            "prediction_horizon": 72,
            "pv_power_forecast": list(range(1, 73)),
            "load_power_forecast": list(range(1, 73)),
            "load_cost_forecast": [0.15] * 72,
            "prod_price_forecast": [0.05] * 72,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params = copy.deepcopy(await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True))
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")

        idd = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(idd, dict)

        # THE KEY INVARIANT: delta_forecast_daily bumped to 2 days by the auto-extend logic
        self.assertEqual(
            idd["fcst"].optim_conf["delta_forecast_daily"].days,
            2,
            "delta_forecast_daily must be bumped to 2 days to cover a 72-step horizon",
        )
        # forecast_dates must cover the full 72-step window
        self.assertEqual(
            len(idd["fcst"].forecast_dates),
            72,
            f"forecast_dates must have exactly 72 entries; got {len(idd['fcst'].forecast_dates)}",
        )

        opt_res = await naive_mpc_optim(idd, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(len(opt_res), 72, f"opt_res must have 72 rows; got {len(opt_res)}")
        self.assertEqual(
            opt_res.isnull().sum().sum(),
            0,
            "opt_res must contain no NaN values",
        )

    # Test naive mpc optimization
    async def test_naive_mpc_optim(self):
        # Test mpc optimization
        costfun = "profit"
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)
        # Test mpc optimization with runtime parameters similar to the documentation
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.6,
            "operating_hours_of_each_deferrable_load": [1, 3],
            "start_timesteps_of_each_deferrable_load": [-3, 0],
            "end_timesteps_of_each_deferrable_load": [8, 0],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "naive"
        params["optim_conf"]["load_cost_forecast_method"] = "hp_hc_periods"
        params["optim_conf"]["production_price_forecast_method"] = "constant"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)
        # Test publish after passing the forecast as list
        # with method_ts_round=first
        costfun = "profit"
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["retrieve_hass_conf"]["method_ts_round"] = "first"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            None,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res_first = await publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertEqual(len(opt_res_first), 1)
        # test mpc and publish with method_ts_round=last and set_use_battery=true
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["retrieve_hass_conf"]["method_ts_round"] = "last"
        params["optim_conf"]["set_use_battery"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            None,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res_last = await publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertEqual(len(opt_res_last), 1)

        # Check if status is published
        from datetime import datetime

        now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
            second=0, microsecond=0
        )
        idx_closest = opt_res.index.get_indexer([now_precise], method="nearest")[0]
        custom_cost_fun_id = {
            "entity_id": "sensor.optim_status",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        }
        publish_prefix = ""
        response, data = await input_data_dict["rh"].post_data(
            opt_res["optim_status"],
            idx_closest,
            custom_cost_fun_id["entity_id"],
            "",
            custom_cost_fun_id["unit_of_measurement"],
            custom_cost_fun_id["friendly_name"],
            type_var="optim_status",
            publish_prefix=publish_prefix,
        )
        self.assertTrue(hasattr(response, "__class__"))
        self.assertEqual(data["attributes"]["friendly_name"], "EMHASS optimization status")
        # When using set_use_adjusted_pv = True
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["optim_conf"]["set_use_adjusted_pv"] = True
        params["optim_conf"]["set_use_pv"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)

    # Test outputs of fit, predict and tune
    async def test_forecast_model_fit_predict_tune(self):
        costfun = "profit"
        action = "forecast-model-fit"
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
            "model_predict_publish": True,
            "model_predict_entity_id": "sensor.p_load_forecast_knn",
            "model_predict_unit_of_measurement": "W",
            "model_predict_friendly_name": "Load Power Forecast KNN regressor",
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(input_data_dict["params"]["passed_data"]["model_type"], "long_train_data")
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["sklearn_model"], "KNeighborsRegressor"
        )
        self.assertIs(input_data_dict["params"]["passed_data"]["perform_backtest"], False)
        default_file_path = emhass_conf["data_path"] / "load_forecast.pkl"
        created_dummy = False
        if default_file_path.exists():
            default_file_path.unlink()
        idx = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="30min")
        df_dummy = pd.DataFrame({"sensor.power_load_no_var_loads": [100.0] * 60}, index=idx)
        dummy_data = (df_dummy, None, None, None)
        with default_file_path.open("wb") as f:
            pickle.dump(dummy_data, f)
        created_dummy = True
        try:
            input_data_dict = await set_input_data_dict(
                emhass_conf,
                costfun,
                self.params_json,
                self.runtimeparams_json,
                action,
                logger,
                get_data_from_file=True,
            )
        finally:
            if created_dummy and default_file_path.exists():
                default_file_path.unlink()
        self.assertEqual(input_data_dict["params"]["passed_data"]["model_type"], "load_forecast")
        self.assertIsInstance(input_data_dict["df_input_data"], pd.DataFrame)
        idx_fresh = pd.date_range(end=pd.Timestamp.now(), periods=48 * 10, freq="30min")
        df_fresh = pd.DataFrame(
            {"sensor.power_load_no_var_loads": rng.random(len(idx_fresh)) * 100},
            index=idx_fresh,
        )
        df_fresh = utils.set_df_index_freq(df_fresh)
        input_data_dict["df_input_data"] = df_fresh
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=True
        )
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertIs(df_fit_pred_backtest, None)
        injection_dict = utils.get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)
        # Re-inject fresh data for predict
        input_data_dict["df_input_data"] = df_fresh
        df_pred = await forecast_model_predict(
            input_data_dict, logger, use_last_window=False, debug=True, mlf=mlf
        )
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(df_pred.isnull().sum().sum(), 0)
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(df_pred.isnull().sum().sum(), 0)
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertIs(mlf.is_tuned, True)
        injection_dict = utils.get_injection_dict_forecast_model_tune(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)

    # Test data formatting of regressor model fit amd predict
    async def test_regressor_model_fit_predict(self):
        costfun = "profit"
        action = "regressor-model-fit"  # fit and predict methods
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["model_type"],
            "heating_hours_degreeday",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["regression_model"],
            "LassoRegression",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["csv_file"],
            "heating_prediction.csv",
        )
        mlr = await regressor_model_fit(input_data_dict, logger, debug=True)

        # def test_regressor_model_predict(self):
        costfun = "profit"
        action = "regressor-model-predict"  # predict methods
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")

        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["model_type"],
            "heating_hours_degreeday",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["mlr_predict_friendly_name"],
            "Predicted hours",
        )

        await regressor_model_predict(input_data_dict, logger, debug=True, mlr=mlr)

    # CLI test action that does not exist
    async def test_main_wrong_action(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "test",
                "--config",
                str(emhass_conf["config_path"]),
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
            self.assertIsNone(opt_res)

    # CLI test action perfect-optim action
    async def test_main_perfect_forecast_optim(self):
        test_params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        # We patch sys.argv to simulate CLI args
        # AND we patch the Optimization method to return a dummy result instantly
        with (
            patch(
                "sys.argv",
                [
                    "main",
                    "--action",
                    "perfect-optim",
                    "--config",
                    str(emhass_conf["config_path"]),
                    "--debug",
                    "True",
                    "--params",
                    orjson.dumps(test_params).decode("utf-8"),
                ],
            ),
            patch("emhass.optimization.Optimization.perform_perfect_forecast_optim") as mock_optim,
        ):
            # Setup the mock return value to satisfy assertions
            # Create a dataframe with a timezone-aware index (required by assertions)
            idx = pd.date_range("2024-01-01", periods=48, freq="30min", tz="Europe/Paris")
            mock_df = pd.DataFrame(index=idx)
            mock_df["cost_fun_profit"] = 0.0  # Add column expected by logical checks
            mock_df["p_grid"] = 0.0
            mock_optim.return_value = mock_df
            opt_res = await main()

        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            opt_res.index.dtype,
            pd.core.dtypes.dtypes.DatetimeTZDtype,
        )

    # CLI test dayahead forecast optimzation action
    async def test_main_dayahead_forecast_optim(self):
        # --- FIX: Mock Optimization class method using patch ---
        # Because we call main(), we can't access input_data_dict directly.
        # We must patch the class method itself.
        with (
            patch(
                "sys.argv",
                [
                    "main",
                    "--action",
                    "dayahead-optim",
                    "--config",
                    str(emhass_conf["config_path"]),
                    "--params",
                    self.params_json,
                    "--runtimeparams",
                    self.runtimeparams_json,
                    "--debug",
                    "True",
                ],
            ),
            patch("emhass.optimization.Optimization.perform_dayahead_forecast_optim") as mock_optim,
        ):
            # Setup the mock return value
            mock_df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=48, freq="30min"))
            mock_df["p_grid"] = 0.0
            mock_optim.return_value = mock_df
            opt_res = await main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)

    # CLI test naive mpc optimzation action
    async def test_main_naive_mpc_optim(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "naive-mpc-optim",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                self.params_json,
                "--runtimeparams",
                self.runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)

    # CLI test forecast model fit action
    async def test_main_forecast_model_fit(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-fit",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_fit_pred, df_fit_pred_backtest, _ = await main()
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertIs(df_fit_pred_backtest, None)

    # CLI test forecast model predict action
    async def test_main_forecast_model_predict(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-predict",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_pred = await main()
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(df_pred.isnull().sum().sum(), 0)

    # CLI test forecast model tune action
    async def test_main_forecast_model_tune(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-tune",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_pred_optim, mlf = await main()
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertIs(mlf.is_tuned, True)

    # CLI test regressor model fit action
    async def test_main_regressor_model_fit(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-fit",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            await main()

    # CLI test regressor model predict action
    async def test_main_regressor_model_predict(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-predict",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            prediction = await main()
        self.assertIsInstance(prediction, np.ndarray)

    # CLI test publish data action
    async def test_main_publish_data(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "publish-data",
                "--config",
                str(emhass_conf["config_path"]),
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
            self.assertFalse(opt_res.empty)

    # Test export_influxdb_to_csv
    async def test_export_influxdb_to_csv(self):
        costfun = "profit"
        action = "export-influxdb-to-csv"
        # Test Success Case
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "sensor_list": [
                "sensor.power_load_no_var_loads",
                "sensor.power_photovoltaics",
            ],
            "csv_filename": "test_export.csv",
            "start_time": "2025-11-10",
            "end_time": "2025-11-11",
            "resample_freq": "30min",
            "handle_nan": "interpolate",
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,  # Use True to avoid HA calls
        )
        # Mock rh.use_influxdb
        input_data_dict["rh"].use_influxdb = True
        # Create mock data
        index = pd.date_range(
            start="2025-11-10",
            end="2025-11-12",
            freq="10min",
            tz=input_data_dict["rh"].time_zone,
        )
        data = {
            "sensor.power_load_no_var_loads": rng.random(len(index)) * 1000,
            "sensor.power_photovoltaics": rng.random(len(index)) * 5000,
        }
        df_final_mock = pd.DataFrame(data, index=index)
        # Add some NaNs to test handle_nan
        df_final_mock.iloc[5:10, 0] = np.nan
        # Mock rh.get_data
        input_data_dict["rh"].get_data = Mock(return_value=True)
        input_data_dict["rh"].df_final = df_final_mock
        # Mock the final to_csv call to avoid writing a file
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            success = await export_influxdb_to_csv(input_data_dict, logger)
            self.assertTrue(success)
            # Check if to_csv was called
            mock_to_csv.assert_called_once()
            # Check call args
            args, kwargs = mock_to_csv.call_args
            self.assertFalse(kwargs["index"], False)
            self.assertIsInstance(args[0], pathlib.Path)
            self.assertEqual(args[0].name, "test_export.csv")
        # Test InfluxDB Disabled
        input_data_dict["rh"].use_influxdb = False
        success = await export_influxdb_to_csv(input_data_dict, logger)
        self.assertFalse(success)
        # Test Missing Params (e.g., sensor_list)
        params_no_sensors = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams_no_sensors = {
            "csv_filename": "test_export.csv",
            "start_time": "2025-11-10",
        }
        runtimeparams_no_sensors_json = orjson.dumps(runtimeparams_no_sensors).decode("utf-8")
        params_no_sensors["passed_data"] = runtimeparams_no_sensors
        params_no_sensors_json = orjson.dumps(params_no_sensors).decode("utf-8")
        input_data_dict_no_sensors = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_no_sensors_json,
            runtimeparams_no_sensors_json,
            action,
            logger,
            get_data_from_file=True,
        )
        input_data_dict_no_sensors["rh"].use_influxdb = True
        # This should fail inside export_influxdb_to_csv due to missing 'sensor_list'
        success = await export_influxdb_to_csv(input_data_dict_no_sensors, logger)
        self.assertFalse(success)
        # Test rh.get_data fails
        input_data_dict["rh"].use_influxdb = True  # Reset from test 2
        input_data_dict["rh"].get_data = Mock(return_value=False)  # Mock get_data to fail
        input_data_dict["rh"].df_final = None
        success = await export_influxdb_to_csv(input_data_dict, logger)
        self.assertFalse(success)

    # Test that runtime costfun parameter overrides config costfun parameter
    async def test_costfun_runtime_override(self):
        """Test that runtime costfun parameter correctly overrides config costfun parameter."""
        # Build params with default config
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        # Set costfun in config to 'profit'
        params["optim_conf"]["costfun"] = "profit"
        # Add runtime parameters with costfun override
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
            "costfun": "cost",  # Override to 'cost'
        }
        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        # The costfun passed to set_input_data_dict is from the config (before runtime params)
        costfun_from_config = "profit"
        action = "dayahead-optim"
        # Call set_input_data_dict
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # This is 'profit' from config
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Check that the costfun in input_data_dict is the runtime parameter value ('cost')
        self.assertEqual(
            input_data_dict["costfun"],
            "cost",
            "Runtime parameter 'costfun' should override config parameter",
        )
        # Also test with 'self-consumption' as another option
        runtimeparams["costfun"] = "self-consumption"
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # Still 'profit' from config
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(
            input_data_dict["costfun"],
            "self-consumption",
            "Runtime parameter 'costfun' should override config parameter for self-consumption",
        )
        # Also test when costfun is NOT provided as runtime parameter
        runtimeparams_no_costfun = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
            # No costfun parameter
        }
        runtimeparams_no_costfun_json = orjson.dumps(runtimeparams_no_costfun).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # 'profit' from config
            params_json,
            runtimeparams_no_costfun_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # When no runtime costfun is provided, should use config value
        self.assertEqual(
            input_data_dict["costfun"],
            "profit",
            "Should use config parameter when runtime parameter is not provided",
        )

    def test_is_model_outdated(self):
        """Test the is_model_outdated function for various scenarios."""
        # Test 1: Non-existent file should return True
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_path = pathlib.Path(tmpdir) / "nonexistent_model.pkl"
            result = is_model_outdated(non_existent_path, 24, logger)
            self.assertTrue(result, "Should return True for non-existent file")
        # Test 2: max_age_hours = 0 should force refit (return True)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            result = is_model_outdated(tmp_path, 0, logger)
            self.assertTrue(result, "Should return True when max_age_hours = 0")
        finally:
            tmp_path.unlink()
        # Test 3: Fresh model (just created) should return False
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertFalse(result, "Should return False for fresh model")
        finally:
            tmp_path.unlink()
        # Test 4: Old model (simulated old modification time) should return True
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            # Set modification time to 48 hours ago
            old_time = (datetime.now() - timedelta(hours=48)).timestamp()
            os.utime(tmp_path, (old_time, old_time))
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertTrue(result, "Should return True for model older than max_age")
        finally:
            tmp_path.unlink()
        # Test 5: Model just under the threshold should return False
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            # Set modification time to 23 hours ago (just under 24h threshold)
            recent_time = (datetime.now() - timedelta(hours=23)).timestamp()
            os.utime(tmp_path, (recent_time, recent_time))
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertFalse(result, "Should return False for model just under max_age threshold")
        finally:
            tmp_path.unlink()

    async def test_adjusted_pv_model_max_age_runtime_override(self):
        """Test that runtime adjusted_pv_model_max_age parameter overrides config parameter."""
        # Build params with default config
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        # Set adjusted_pv_model_max_age in config to 24
        params["optim_conf"]["adjusted_pv_model_max_age"] = 24
        # Add runtime parameters with adjusted_pv_model_max_age override
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "adjusted_pv_model_max_age": 6,  # Override to 6 hours
        }
        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        costfun = "profit"
        action = "dayahead-optim"
        # Call set_input_data_dict
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Check that adjusted_pv_model_max_age was overridden in the forecast object
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["adjusted_pv_model_max_age"],
            6,
            "Runtime parameter 'adjusted_pv_model_max_age' should override config parameter",
        )
        # Test with different value
        runtimeparams["adjusted_pv_model_max_age"] = 0  # Force refit
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["adjusted_pv_model_max_age"],
            0,
            "Runtime parameter should override with value 0 (force refit)",
        )

    async def test_adjust_pv_forecast_corrupted_model_recovery(self):
        """Test that adjust_pv_forecast gracefully handles corrupted model files."""
        # Create a corrupted pickle file using tempfile for the path, then write async
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pkl")
        os.close(tmp_fd)  # Close the file descriptor immediately
        tmp_path = pathlib.Path(tmp_name)
        # Write corrupted data asynchronously
        async with aiofiles.open(tmp_path, "wb") as tmp:
            await tmp.write(b"This is not a valid pickle file!")
        try:
            # Setup mock objects
            fcst = MagicMock(spec=Forecast)
            p_pv_forecast = pd.Series([100, 200, 300], name="P_PV")
            test_emhass_conf = {
                "data_path": tmp_path.parent,
            }
            test_optim_conf = {
                "adjusted_pv_model_max_age": 24,
                "adjusted_pv_regression_model": "LassoRegression",
            }
            test_retrieve_hass_conf = {}
            rh = MagicMock()
            # Rename temp file to expected model name
            model_path = tmp_path.parent / "adjust_pv_regressor.pkl"
            tmp_path.rename(model_path)
            # Mock the data retrieval and fit methods
            with patch("emhass.command_line.retrieve_home_assistant_data") as mock_retrieve:
                mock_retrieve.return_value = (True, pd.DataFrame(), None)
                fcst.adjust_pv_forecast_data_prep = MagicMock()
                fcst.adjust_pv_forecast_fit = AsyncMock()
                fcst.adjust_pv_forecast_predict = MagicMock(
                    return_value=pd.DataFrame({"adjusted_forecast": [100, 200, 300]})
                )
                # Call adjust_pv_forecast - should handle corruption and re-fit
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast,
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                # Verify that it called re-fit after detecting corruption
                fcst.adjust_pv_forecast_fit.assert_called_once()
                self.assertIsNotNone(result, "Should return valid result after recovery")
        finally:
            # Cleanup - unlink_missing_ok handles non-existent files safely
            model_path.unlink(missing_ok=True)

    async def test_adjust_pv_forecast_stale_feature_model_refit(self):
        """A saved model trained on an older feature set fails on predict:
        adjust_pv_forecast must re-fit once and retry instead of erroring."""
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pkl")
        os.close(tmp_fd)
        tmp_path = pathlib.Path(tmp_name)
        # A valid pickle (so the load path succeeds); predict fails later
        async with aiofiles.open(tmp_path, "wb") as tmp:
            await tmp.write(pickle.dumps({"legacy": "model"}))
        try:
            fcst = MagicMock(spec=Forecast)
            p_pv_forecast = pd.Series([100, 200, 300], name="P_PV")
            test_emhass_conf = {
                "data_path": tmp_path.parent,
            }
            test_optim_conf = {
                "adjusted_pv_model_max_age": 24,
                "adjusted_pv_regression_model": "LassoRegression",
            }
            test_retrieve_hass_conf = {}
            rh = MagicMock()
            model_path = tmp_path.parent / "adjust_pv_regressor.pkl"
            tmp_path.rename(model_path)
            with patch("emhass.command_line.retrieve_home_assistant_data") as mock_retrieve:
                mock_retrieve.return_value = (True, pd.DataFrame(), None)
                fcst.adjust_pv_forecast_data_prep = MagicMock()
                fcst.adjust_pv_forecast_fit = AsyncMock()
                # First predict raises like scikit-learn does on a feature-name
                # mismatch; after the re-fit the retry succeeds
                fcst.adjust_pv_forecast_predict = MagicMock(
                    side_effect=[
                        ValueError("The feature names should match those that were passed"),
                        pd.DataFrame({"adjusted_forecast": [100, 200, 300]}),
                    ]
                )
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast,
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                fcst.adjust_pv_forecast_fit.assert_called_once()
                self.assertEqual(fcst.adjust_pv_forecast_predict.call_count, 2)
                self.assertIsNotNone(result, "Should return valid result after re-fit")
        finally:
            model_path.unlink(missing_ok=True)

    async def test_adjusted_pv_model_max_age_affects_model_refit_behavior(self):
        """
        Test that adjusted_pv_model_max_age controls whether a cached model is reused
        vs. refit within adjust_pv_forecast.
        """
        # Create a temporary data_path with a synthetic PV model file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = pathlib.Path(tmpdir)
            model_path = data_path / "adjust_pv_regressor.pkl"
            # Create a simple picklable object to represent a valid model
            # (Using a dict instead of MagicMock since MagicMock isn't picklable)
            mock_model = {"model_type": "test", "params": [1, 2, 3]}
            async with aiofiles.open(model_path, "wb") as f:
                await f.write(pickle.dumps(mock_model))
            # Setup test objects
            fcst = MagicMock(spec=Forecast)
            p_pv_forecast = pd.Series([100, 200, 300], name="P_PV")
            test_emhass_conf = {
                "data_path": data_path,
            }
            test_retrieve_hass_conf = {}
            rh = MagicMock()
            # Mock the data retrieval to avoid real I/O
            with patch("emhass.command_line.retrieve_home_assistant_data") as mock_retrieve:
                mock_retrieve.return_value = (True, pd.DataFrame(), None)
                fcst.adjust_pv_forecast_data_prep = MagicMock()
                fcst.adjust_pv_forecast_fit = AsyncMock()
                fcst.adjust_pv_forecast_predict = MagicMock(
                    return_value=pd.DataFrame({"adjusted_forecast": [100, 200, 300]})
                )
                # Test Case 1: Fresh model with large max_age -> should load existing, no refit
                test_optim_conf_fresh = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }
                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_fresh,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                # Should NOT call fit when model is fresh
                fcst.adjust_pv_forecast_fit.assert_not_called()
                # Should NOT retrieve data when model is fresh
                mock_retrieve.assert_not_called()
                self.assertIsNotNone(result, "Should return valid result using cached model")
                # Test Case 2: max_age = 0 -> should force refit
                test_optim_conf_force = {
                    "adjusted_pv_model_max_age": 0,
                    "adjusted_pv_regression_model": "LassoRegression",
                }
                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_force,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                # Should call fit when max_age = 0
                fcst.adjust_pv_forecast_fit.assert_called_once()
                # Should retrieve data when refitting
                mock_retrieve.assert_called_once()
                self.assertIsNotNone(result, "Should return valid result after forced refit")
                # Test Case 3: Old model (48h old) with max_age=24 -> should refit
                # Set model file modification time to 48 hours ago
                old_time = (datetime.now() - timedelta(hours=48)).timestamp()
                os.utime(model_path, (old_time, old_time))
                test_optim_conf_stale = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }
                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_stale,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                # Should call fit when model is stale
                fcst.adjust_pv_forecast_fit.assert_called_once()
                # Should retrieve data when refitting
                mock_retrieve.assert_called_once()
                self.assertIsNotNone(
                    result, "Should return valid result after refitting stale model"
                )
                # Test Case 4: Model just under threshold (23h old, max_age=24) -> should reuse
                # Set model file modification time to 23 hours ago
                recent_time = (datetime.now() - timedelta(hours=23)).timestamp()
                os.utime(model_path, (recent_time, recent_time))
                test_optim_conf_under = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }
                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_under,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )
                # Should NOT call fit when model is just under threshold
                fcst.adjust_pv_forecast_fit.assert_not_called()
                # Should NOT retrieve data when model is still fresh enough
                mock_retrieve.assert_not_called()
                self.assertIsNotNone(
                    result,
                    "Should return valid result using cached model just under threshold",
                )

    async def test_retrieve_from_hass_naive_mpc(self):
        """
        Test the _retrieve_from_hass helper specifically for the 'naive-mpc-optim' path
        to cover the days_list=1 assignment and debug logging.
        """
        # Prepare params to trigger the specific if/else blocks
        optim_conf = {"set_use_pv": True, "set_use_adjusted_pv": True}
        retrieve_hass_conf = {
            "historic_days_to_retrieve": 2,
            "sensor_power_load_no_var_loads": "sensor.load",
            "sensor_power_photovoltaics": "sensor.pv",
            "sensor_power_photovoltaics_forecast": "sensor.pv_forecast",
            "load_negative": False,
            "set_zero_min": True,
            "sensor_replace_zero": [],
            "sensor_linear_interp": [],
        }
        # Mock the RetrieveHass object
        mock_rh = Mock()
        mock_rh.get_data = AsyncMock(return_value=True)
        # Mock prepare_data so it doesn't fail if called
        mock_rh.prepare_data = Mock()
        mock_rh.df_final = pd.DataFrame()  # Ensure df_final exists for copy()
        # Mock logger to verify debug call
        mock_logger = Mock()
        # Execute
        success, days_list, _ = await retrieve_home_assistant_data(
            set_type="naive-mpc-optim",  # triggers the elif set_type == "naive-mpc-optim"
            get_data_from_file=False,  # triggers _retrieve_from_hass
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            rh=mock_rh,
            emhass_conf={},
            test_df_literal="test.pkl",
            logger=mock_logger,
        )
        # Assertions
        self.assertTrue(success)
        # Verify the specific logger path was hit
        mock_logger.debug.assert_called()
        call_args = str(mock_logger.debug.call_args)
        self.assertIn("Variable list for data retrieval", call_args)
        # Non-battery_id retrieval must not protect any column from the
        # set_zero_min treatment (#1041)
        self.assertIsNone(mock_rh.prepare_data.call_args.kwargs.get("protected_columns"))

    async def test_retrieve_from_hass_heatpump_sensors_include_blind_window_door_power_duty(
        self,
    ):
        """Regression test for a real, confirmed bug: heatpump_room_blind_sensors/
        heatpump_room_window_sensors/heatpump_room_door_sensors were never
        added to the naive-mpc-optim live fetch var_list at all, so the
        sensor-based blind/window/door detection never actually received
        real sensor data in production. heatpump_power_sensor/heatpump_duty_sensor
        are new additions needed by the Kalman opening detector's predict step."""
        optim_conf = {"set_use_pv": False, "set_use_heatpump": True}
        retrieve_hass_conf = {
            "historic_days_to_retrieve": 2,
            "sensor_power_load_no_var_loads": "sensor.load",
            "load_negative": False,
            "set_zero_min": True,
            "sensor_replace_zero": [],
            "sensor_linear_interp": [],
            "heatpump_room_temp_sensors": ["sensor.room_temp"],
            "heatpump_indoor_temp_sensor": "sensor.indoor_temp",
            "heatpump_room_blind_sensors": ["cover.living_room_blind"],
            "heatpump_room_window_sensors": ["binary_sensor.living_room_window"],
            "heatpump_room_door_sensors": ["binary_sensor.living_room_door"],
            "heatpump_power_sensor": "sensor.hp_power",
            "heatpump_duty_sensor": "sensor.hp_duty",
        }
        mock_rh = Mock()
        mock_rh.get_data = AsyncMock(return_value=True)
        mock_rh.prepare_data = Mock()
        mock_rh.df_final = pd.DataFrame()

        await retrieve_home_assistant_data(
            set_type="naive-mpc-optim",
            get_data_from_file=False,
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            rh=mock_rh,
            emhass_conf={},
            test_df_literal="test.pkl",
            logger=logger,
        )

        var_list = mock_rh.get_data.call_args.args[1]
        self.assertIn("cover.living_room_blind", var_list)
        self.assertIn("binary_sensor.living_room_window", var_list)
        self.assertIn("binary_sensor.living_room_door", var_list)
        self.assertIn("sensor.hp_power", var_list)
        self.assertIn("sensor.hp_duty", var_list)

    async def test_retrieve_from_hass_heatpump_sensors_absent_without_set_use_heatpump(self):
        optim_conf = {"set_use_pv": False, "set_use_heatpump": False}
        retrieve_hass_conf = {
            "historic_days_to_retrieve": 2,
            "sensor_power_load_no_var_loads": "sensor.load",
            "load_negative": False,
            "set_zero_min": True,
            "sensor_replace_zero": [],
            "sensor_linear_interp": [],
            "heatpump_room_blind_sensors": ["cover.living_room_blind"],
            "heatpump_room_window_sensors": ["binary_sensor.living_room_window"],
            "heatpump_room_door_sensors": ["binary_sensor.living_room_door"],
            "heatpump_power_sensor": "sensor.hp_power",
            "heatpump_duty_sensor": "sensor.hp_duty",
        }
        mock_rh = Mock()
        mock_rh.get_data = AsyncMock(return_value=True)
        mock_rh.prepare_data = Mock()
        mock_rh.df_final = pd.DataFrame()

        await retrieve_home_assistant_data(
            set_type="naive-mpc-optim",
            get_data_from_file=False,
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            rh=mock_rh,
            emhass_conf={},
            test_df_literal="test.pkl",
            logger=logger,
        )

        var_list = mock_rh.get_data.call_args.args[1]
        self.assertNotIn("cover.living_room_blind", var_list)
        self.assertNotIn("binary_sensor.living_room_window", var_list)
        self.assertNotIn("binary_sensor.living_room_door", var_list)
        self.assertNotIn("sensor.hp_power", var_list)
        self.assertNotIn("sensor.hp_duty", var_list)

    async def test_retrieve_from_hass_battery_id_protected_columns(self):
        """
        battery_id retrieval must pass the battery power and SoC sensors to
        prepare_data as protected_columns, so the set_zero_min clip cannot
        destroy the discharge direction or a measured 0% SoC (#1041).
        Covers the bare-string (N=1) and list (N>1) config forms.
        """
        optim_conf = {"set_use_pv": True, "set_use_adjusted_pv": True}
        base_conf = {
            "historic_days_to_retrieve": 2,
            "sensor_power_load_no_var_loads": "sensor.load",
            "sensor_power_photovoltaics": "sensor.pv",
            "sensor_power_photovoltaics_forecast": "sensor.pv_forecast",
            "load_negative": False,
            "set_zero_min": True,
            "sensor_replace_zero": [],
            "sensor_linear_interp": [],
        }
        cases = [
            (
                "sensor.batt_power",
                "sensor.batt_soc",
                ["sensor.batt_power", "sensor.batt_soc"],
            ),
            (
                ["sensor.batt_power1", "sensor.batt_power2"],
                ["sensor.batt_soc1", "sensor.batt_soc2"],
                [
                    "sensor.batt_power1",
                    "sensor.batt_power2",
                    "sensor.batt_soc1",
                    "sensor.batt_soc2",
                ],
            ),
        ]
        for power_cfg, soc_cfg, expected in cases:
            with self.subTest(power_cfg=power_cfg, soc_cfg=soc_cfg):
                retrieve_hass_conf = dict(base_conf)
                retrieve_hass_conf["sensor_power_battery"] = power_cfg
                retrieve_hass_conf["sensor_battery_state_of_charge"] = soc_cfg
                mock_rh = Mock()
                mock_rh.get_data = AsyncMock(return_value=True)
                mock_rh.prepare_data = Mock()
                mock_rh.df_final = pd.DataFrame()
                success, _, _ = await retrieve_home_assistant_data(
                    set_type="battery_id",
                    get_data_from_file=False,
                    retrieve_hass_conf=retrieve_hass_conf,
                    optim_conf=optim_conf,
                    rh=mock_rh,
                    emhass_conf={},
                    test_df_literal="test.pkl",
                    logger=Mock(),
                )
                self.assertTrue(success)
                self.assertEqual(
                    mock_rh.prepare_data.call_args.kwargs.get("protected_columns"),
                    expected,
                )

    async def test_adjust_pv_forecast_generic_exception(self):
        """
        Test the catch-all Exception block in adjust_pv_forecast.
        This simulates a non-pickle/EOF error (like a Runtime error) during model load.
        """
        mock_logger = Mock()
        mock_fcst = Mock()
        mock_rh = Mock()
        # 1. Force is_model_outdated to False so it attempts to load
        # 2. Mock aiofiles to return bytes
        # 3. Mock pickle.loads to raise a generic Exception (not one of the specific caught ones)
        with (
            patch("emhass.command_line.is_model_outdated", return_value=False),
            patch("emhass.command_line.aiofiles.open") as mock_file,
            patch("pickle.loads", side_effect=Exception("Generic catastrophe")),
        ):
            # Setup mock file context
            mock_file_handle = AsyncMock()
            mock_file.return_value.__aenter__.return_value = mock_file_handle
            mock_file_handle.read.return_value = b"some bytes"
            # Execute
            result = await adjust_pv_forecast(
                logger=mock_logger,
                fcst=mock_fcst,
                p_pv_forecast=pd.Series([1, 2]),
                get_data_from_file=False,
                retrieve_hass_conf={},
                optim_conf={"adjusted_pv_model_max_age": 1},
                rh=mock_rh,
                emhass_conf={"data_path": "."},
                test_df_literal=pd.DataFrame(),
            )
            # Assertions
            self.assertFalse(result, "Should return False on generic exception")
            # Verify we hit the specific exception block
            # logger.error(f"Unexpected error loading adjusted PV model: ...")
            # logger.error("Cannot recover from this error")
            error_logs = [str(call) for call in mock_logger.error.mock_calls]
            self.assertTrue(any("Unexpected error loading" in log for log in error_logs))
            self.assertTrue(any("Cannot recover" in log for log in error_logs))

    async def test_publish_thermal_loads(self):
        """
        Test _publish_thermal_loads with a configured thermal load.
        """
        # Setup thermal config in optim_conf
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["def_load_config"] = [{"thermal_config": {"model_type": "ideal"}}]
        params["optim_conf"]["number_of_deferrable_loads"] = 1
        # Setup passed_data with thermal IDs
        runtimeparams = {
            "custom_predicted_temperature_id": [
                {"entity_id": "sensor.temp", "unit_of_measurement": "C", "friendly_name": "Temp"}
            ],
            "custom_heating_demand_id": [
                {"entity_id": "sensor.heat", "unit_of_measurement": "W", "friendly_name": "Heat"}
            ],
        }
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "publish-data",
            logger,
            get_data_from_file=True,
        )
        # Mock the optimization results DataFrame to include thermal columns AND standard columns
        idx = pd.date_range(end=pd.Timestamp.now(tz="Europe/Paris"), periods=1, freq="30min")
        mock_df = pd.DataFrame(
            {
                "predicted_temp_heater0": [20.5],
                "heating_demand_heater0": [1000.0],
                "P_PV": [0.0],
                "P_Load": [0.0],
                "P_grid": [0.0],
                "optim_status": ["Optimal"],
                "unit_load_cost": [0.1],
                "unit_prod_price": [0.05],
            },
            index=idx,
        )
        # Mock rh.post_data
        input_data_dict["rh"].post_data = AsyncMock(return_value=True)
        # Patch _get_closest_index to return 0 to bypass timestamp matching issues
        with patch("emhass.command_line._get_closest_index", return_value=0):
            # Execute
            await publish_data(input_data_dict, logger, opt_res_latest=mock_df)
        # Verify calls for thermal data
        call_args_list = input_data_dict["rh"].post_data.call_args_list
        found_temp = any("sensor.temp" in str(args) for args in call_args_list)
        found_heat = any("sensor.heat" in str(args) for args in call_args_list)
        self.assertTrue(found_temp, "Should publish predicted temperature")
        self.assertTrue(found_heat, "Should publish heating demand")

    def test_translate_ev_power_to_mode_boundaries(self):
        """Table-driven test of the continuous-power -> discrete myenergi
        mode/phase heuristic across its boundary conditions."""
        min_1p, max_1p, min_3p, max_3p = 1380.0, 3680.0, 4140.0, 11000.0
        cases = [
            (0.0, "stopped", "1_phase"),
            (min_1p / 2 - 1, "stopped", "1_phase"),
            (min_1p / 2, "eco", "1_phase"),
            ((min_1p + max_1p) / 2 - 1, "eco", "1_phase"),
            ((min_1p + max_1p) / 2, "eco_plus", "1_phase"),
            (max_1p, "eco_plus", "1_phase"),
            (max_1p + 1, "eco_plus", "3_phase"),
            (max_3p - 1, "eco_plus", "3_phase"),
            (max_3p, "fast", "3_phase"),
            (max_3p + 5000, "fast", "3_phase"),
        ]
        for power_w, expected_mode, expected_phase in cases:
            with self.subTest(power_w=power_w):
                mode, phase = _translate_ev_power_to_mode(
                    power_w, min_1p, max_1p, min_3p, max_3p
                )
                self.assertEqual(mode, expected_mode)
                self.assertEqual(phase, expected_phase)

    async def test_publish_room_heatpump_ev_targets(self):
        """Test the Phase 3 publish helpers (room target temp, heat pump
        dispatch on/off, EV charge mode/phase) end-to-end via publish_data."""
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["def_load_config"] = [
            {
                "thermal_battery": {
                    "min_temperatures": [18.0],
                    "max_temperatures": [21.5],
                }
            },
            {},
            {},
        ]
        params["optim_conf"]["number_of_deferrable_loads"] = 3
        params["passed_data"] = {
            "room_load_indices": {"Living Room": 0},
            "heatpump_dispatch_load_index": 1,
            "ev_load_indices": {"Zappi": 2},
            "custom_room_target_temp_id": [
                {
                    "entity_id": "sensor.room_target_temp_living_room",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                    "friendly_name": "Living Room Target Temperature",
                }
            ],
            "custom_heatpump_dispatch_target_id": {
                "entity_id": "sensor.heatpump_dispatch_target",
                "device_class": "",
                "unit_of_measurement": "",
                "friendly_name": "Heat Pump Dispatch Target",
            },
            "custom_ev_charge_mode_target_id": [
                {
                    "entity_id": "sensor.ev_charge_mode_target_zappi",
                    "device_class": "",
                    "unit_of_measurement": "",
                    "friendly_name": "Zappi Charge Mode Target",
                }
            ],
            "custom_ev_phase_target_id": [
                {
                    "entity_id": "sensor.ev_phase_target_zappi",
                    "device_class": "",
                    "unit_of_measurement": "",
                    "friendly_name": "Zappi Phase Target",
                }
            ],
        }
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "publish-data",
            logger,
            get_data_from_file=True,
        )
        idx = pd.date_range(end=pd.Timestamp.now(tz="Europe/Paris"), periods=1, freq="30min")
        mock_df = pd.DataFrame(
            {
                "P_deferrable0": [500.0],
                "P_deferrable1": [2000.0],
                "P_deferrable2": [9000.0],
                "P_PV": [0.0],
                "P_Load": [0.0],
                "P_grid": [0.0],
                "optim_status": ["Optimal"],
                "unit_load_cost": [0.1],
                "unit_prod_price": [0.05],
            },
            index=idx,
        )
        input_data_dict["rh"].post_data = AsyncMock(return_value=True)
        with patch("emhass.command_line._get_closest_index", return_value=0):
            await publish_data(input_data_dict, logger, opt_res_latest=mock_df)

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        published_entities = [args[0][2] for args in call_args_list if len(args[0]) > 2]

        self.assertIn("sensor.room_target_temp_living_room", published_entities)
        self.assertIn("sensor.heatpump_dispatch_target", published_entities)
        self.assertIn("sensor.ev_charge_mode_target_zappi", published_entities)
        self.assertIn("sensor.ev_phase_target_zappi", published_entities)

    @staticmethod
    def _fake_fitted_params():
        from emhass.thermal.thermal_mass_physics import DEFAULT_X0, PARAM_NAMES

        return {"params": dict(zip(PARAM_NAMES, DEFAULT_X0.tolist(), strict=True))}

    async def _build_heating_forecast_input_data_dict(self):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["heating_forecast_enabled"] = True
        params["optim_conf"]["heating_forecast_horizon_hours"] = 24
        params["optim_conf"]["heating_forecast_comfort_min_temp"] = 19.0
        params["optim_conf"]["heating_forecast_safety_margin_c"] = 0.5
        params["retrieve_hass_conf"]["heatpump_indoor_temp_sensor"] = "sensor.indoor_temperature"
        # _append_heating_forecast_targets only runs inside build_params (i.e. when
        # heating_forecast_enabled is already True *before* the config pipeline
        # builds this params blob); set_input_data_dict doesn't re-run it on an
        # already-built params dict. Register the same entities by hand here,
        # matching how test_publish_room_heatpump_ev_targets does it above -
        # the registration function itself has its own dedicated test in
        # tests/test_utils.py.
        params.setdefault("passed_data", {})
        params["passed_data"]["custom_indoor_temp_forecast_id"] = {
            "entity_id": "sensor.indoor_temp_forecast",
            "device_class": "temperature",
            "unit_of_measurement": "°C",
            "friendly_name": "Indoor Temperature Forecast",
        }
        params["passed_data"]["custom_heating_needed_by_id"] = {
            "entity_id": "sensor.heating_needed_by",
            "device_class": "",
            "unit_of_measurement": "",
            "friendly_name": "Heating Needed By",
        }
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "heating-need-forecast",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=1, freq="30min")
        rh.get_data = AsyncMock(return_value=True)
        rh.prepare_data = Mock()
        rh.df_final = pd.DataFrame({"sensor.indoor_temperature": [20.0]}, index=idx)
        rh.post_data = AsyncMock(return_value=True)

        weather_idx = pd.date_range(
            start=pd.Timestamp.now(tz="UTC"), periods=48, freq="30min"
        )
        df_weather = pd.DataFrame(
            {
                "temp_air": -5.0,
                "wind_speed": 5.0,
                "ghi": 0.0,
                "dni": 0.0,
                "dhi": 0.0,
            },
            index=weather_idx,
        )
        input_data_dict["fcst"].get_weather_forecast = AsyncMock(return_value=df_weather)
        return input_data_dict

    async def test_compute_heating_forecast_disabled_returns_none(self):
        input_data_dict = await self._build_heating_forecast_input_data_dict()
        input_data_dict["optim_conf"]["heating_forecast_enabled"] = False

        result = await compute_heating_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_heating_forecast_missing_fit_returns_none(self):
        input_data_dict = await self._build_heating_forecast_input_data_dict()

        with patch(
            "emhass.command_line.load_json_blob", AsyncMock(return_value=None)
        ):
            result = await compute_heating_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_heating_forecast_publishes_both_sensors(self):
        input_data_dict = await self._build_heating_forecast_input_data_dict()

        with patch(
            "emhass.command_line.load_json_blob",
            AsyncMock(return_value=self._fake_fitted_params()),
        ):
            result = await compute_heating_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        self.assertEqual(len(call_args_list), 2)
        published_entities = {args[2]: kwargs.get("type_var") for args, kwargs in call_args_list}
        self.assertEqual(
            published_entities.get("sensor.indoor_temp_forecast"), "temperature"
        )
        self.assertEqual(
            published_entities.get("sensor.heating_needed_by"), "forecast_event"
        )
        # Outdoor is -5degC for the whole horizon with heating forced off: the
        # 19degC comfort floor (minus the 0.5degC safety margin) must be
        # crossed well within a 24h horizon, not "beyond_horizon".
        self.assertNotEqual(result["heating_needed_by"], "beyond_horizon")

    async def _build_refit_input_data_dict(self, n_rows: int = 2000):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["heating_model_refit_enabled"] = True
        params["optim_conf"]["heating_model_refit_window_days"] = 60
        params["optim_conf"]["heating_model_refit_max_mae_c"] = 1.5
        params["retrieve_hass_conf"]["use_influxdb"] = True
        params["retrieve_hass_conf"]["heatpump_indoor_temp_sensor"] = "sensor.indoor_temperature"
        params["retrieve_hass_conf"]["heatpump_power_sensor"] = "sensor.kwh_meter"
        params["retrieve_hass_conf"]["heatpump_outdoor_temp_sensor"] = "sensor.outdoor_temperature"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "heating-model-refit",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        rh.get_data = AsyncMock(return_value=True)
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_rows, freq="15min")
        rh.df_final = pd.DataFrame(
            {
                "sensor.indoor_temperature": 20.0 + 0.1 * np.sin(np.linspace(0, 40, n_rows)),
                "sensor.kwh_meter": 300.0,
                "sensor.outdoor_temperature": 5.0,
            },
            index=idx,
        )
        return input_data_dict

    async def test_refit_heating_model_disabled_returns_none(self):
        input_data_dict = await self._build_refit_input_data_dict()
        input_data_dict["optim_conf"]["heating_model_refit_enabled"] = False

        result = await refit_heating_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_heating_model_requires_influxdb(self):
        input_data_dict = await self._build_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["use_influxdb"] = False

        result = await refit_heating_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_heating_model_too_few_rows_returns_none(self):
        input_data_dict = await self._build_refit_input_data_dict(n_rows=10)

        result = await refit_heating_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_heating_model_deploys_good_fit(self):
        from emhass.thermal.thermal_mass_physics import DEFAULT_X0, PARAM_NAMES

        input_data_dict = await self._build_refit_input_data_dict()
        fake_params = DEFAULT_X0.copy()
        fake_fit_info = {"fit_mae_c": 0.3, "nfev": 5, "cost": 1.0, "success": True, "status": 2}

        with (
            patch(
                "emhass.thermal.thermal_mass_physics._fit_temperature_params",
                return_value=(fake_params, fake_fit_info),
            ),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await refit_heating_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["deployed"])
        self.assertEqual(result["fit_mae_c"], 0.3)
        mock_save.assert_awaited_once()
        saved_filename = mock_save.call_args[0][1]
        saved_payload = mock_save.call_args[0][2]
        self.assertEqual(saved_filename, "thermal_physics_params.json")
        self.assertEqual(set(saved_payload["params"].keys()), set(PARAM_NAMES))

    async def test_refit_heating_model_rejects_bad_fit(self):
        input_data_dict = await self._build_refit_input_data_dict()
        fake_params = None
        bad_fit_info = {"fit_mae_c": 5.0, "nfev": 5, "cost": 1.0, "success": True, "status": 2}

        with (
            patch(
                "emhass.thermal.thermal_mass_physics._fit_temperature_params",
                return_value=(fake_params, bad_fit_info),
            ),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await refit_heating_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(result["deployed"])
        mock_save.assert_not_awaited()

    # ------------------------------------------------------------------
    # Hybrid heat pump gas/electric model (standalone sibling of the
    # physics refit/forecast above, for emhass.thermal.hybrid_heatpump_lr)
    # ------------------------------------------------------------------

    class _FakeHybridModel:
        """Stand-in for HybridHeatPumpLR: exercises refit_hybrid_heatpump_model's
        own control flow (gating, MAE computation, threshold, save) without
        depending on sklearn's actual fit quality on synthetic data."""

        def __init__(self, elec_value=300.0, gas_value=0.0, electric_only=False):
            self.elec_value = elec_value
            self.gas_value = gas_value
            self.electric_only = electric_only
            # Mirrors the real class: gas_model_ is the runtime source of
            # truth compute_hybrid_heatpump_forecast checks to decide
            # whether to publish/score a gas prediction at all.
            self.gas_model_ = None if electric_only else "stub"
            # Records the "current step" duty seen on each predict() call, so
            # tests can confirm a dynamic per-step trajectory was actually
            # fed in (not a single frozen value) - see
            # test_compute_hybrid_heatpump_forecast_uses_aggregate_duty_trajectory.
            self.seen_duties = []

        def fit(self, df, y_elec, y_gas):
            return self

        def predict(self, df):
            n = len(df)
            self.seen_duties.append(float(df["heatpump_duty"].iloc[-1]))
            gas = np.zeros(n) if self.electric_only else np.full(n, self.gas_value)
            return np.full(n, self.elec_value), gas

    async def _build_hybrid_refit_input_data_dict(self, n_rows: int = 2000, n_gas_positive: int = 200):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["hybrid_heatpump_refit_enabled"] = True
        params["optim_conf"]["heatpump_is_hybrid"] = True
        params["optim_conf"]["hybrid_heatpump_refit_window_days"] = 60
        params["optim_conf"]["hybrid_heatpump_refit_max_electric_mae_w"] = 150.0
        params["optim_conf"]["hybrid_heatpump_refit_max_gas_mae_m3"] = 0.02
        params["retrieve_hass_conf"]["use_influxdb"] = True
        params["retrieve_hass_conf"]["heatpump_indoor_temp_sensor"] = "sensor.indoor_temperature"
        params["retrieve_hass_conf"]["heatpump_power_sensor"] = "sensor.kwh_meter"
        params["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = "sensor.gas_meter"
        params["retrieve_hass_conf"]["heatpump_duty_sensor"] = "sensor.hp_duty"
        params["retrieve_hass_conf"]["heatpump_outdoor_temp_sensor"] = "sensor.outdoor_temperature"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "hybrid-heatpump-model-refit",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        rh.get_data = AsyncMock(return_value=True)
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_rows, freq="15min")
        # Spread positive gas rows evenly across the window so both the
        # chronological train and holdout splits naturally contain some.
        gas = np.zeros(n_rows)
        if n_gas_positive:
            step = max(1, n_rows // n_gas_positive)
            gas[::step] = 0.01
        rh.df_final = pd.DataFrame(
            {
                "sensor.indoor_temperature": 20.0,
                "sensor.kwh_meter": 300.0,
                "sensor.gas_meter": gas,
                "sensor.hp_duty": 0.5,
                "sensor.outdoor_temperature": 5.0,
            },
            index=idx,
        )
        return input_data_dict

    async def test_refit_hybrid_heatpump_model_disabled_returns_none(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict()
        input_data_dict["optim_conf"]["hybrid_heatpump_refit_enabled"] = False

        result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_hybrid_heatpump_model_not_gated_on_is_hybrid(self):
        # heatpump_is_hybrid is deliberately NOT read by this feature - a
        # pure-electric household has it False and must still be able to
        # refit an electric-only model. Uses the real HybridHeatPumpLR (not
        # _FakeHybridModel, which doesn't branch on electric_only and so
        # can't catch a regression back to fitting the gas model on absent
        # data - the whole point of this test).
        input_data_dict = await self._build_hybrid_refit_input_data_dict()
        input_data_dict["optim_conf"]["heatpump_is_hybrid"] = False
        input_data_dict["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = ""

        with patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)) as mock_save:
            result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["electric_only"])
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["gas_mae_m3"])
        mock_save.assert_awaited_once()

    async def test_refit_hybrid_heatpump_model_electric_only_skips_gas_positive_gate(self):
        # n_gas_positive=0 would fail the gas-positive-rows gate in hybrid
        # mode (see test_refit_hybrid_heatpump_model_too_few_gas_positive_rows_returns_none
        # below) - with no gas sensor configured that gate must not even run.
        input_data_dict = await self._build_hybrid_refit_input_data_dict(n_gas_positive=0)
        input_data_dict["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = ""

        with (
            patch(
                "emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR",
                lambda *a, **kw: self._FakeHybridModel(elec_value=300.0, gas_value=0.0),
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["electric_only"])
        self.assertTrue(result["deployed"])

    async def test_refit_hybrid_heatpump_model_requires_influxdb(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["use_influxdb"] = False

        result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_hybrid_heatpump_model_requires_all_hard_sensors(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["heatpump_duty_sensor"] = ""

        result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_hybrid_heatpump_model_too_few_rows_returns_none(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict(n_rows=10, n_gas_positive=0)

        result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_hybrid_heatpump_model_too_few_gas_positive_rows_returns_none(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict(n_gas_positive=5)

        result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_hybrid_heatpump_model_deploys_good_fit(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict()

        with (
            patch(
                "emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR",
                lambda *a, **kw: self._FakeHybridModel(elec_value=300.0, gas_value=0.0),
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(result["electric_only"])
        self.assertTrue(result["deployed"])
        self.assertLess(result["electric_mae_w"], 150.0)
        self.assertLess(result["gas_mae_m3"], 0.02)
        mock_save.assert_awaited_once()
        saved_filename = mock_save.call_args[0][1]
        self.assertEqual(saved_filename, "hybrid_heatpump_lr_model.pkl")

    async def test_refit_hybrid_heatpump_model_rejects_bad_fit(self):
        input_data_dict = await self._build_hybrid_refit_input_data_dict()

        with (
            patch(
                "emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR",
                lambda *a, **kw: self._FakeHybridModel(elec_value=10000.0, gas_value=5.0),
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await refit_hybrid_heatpump_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(result["deployed"])
        mock_save.assert_not_awaited()

    async def _build_hybrid_forecast_input_data_dict(self):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["hybrid_heatpump_forecast_enabled"] = True
        params["retrieve_hass_conf"]["heatpump_duty_sensor"] = "sensor.hp_duty"
        params["retrieve_hass_conf"]["heatpump_indoor_temp_sensor"] = "sensor.indoor_temperature"
        params["retrieve_hass_conf"]["heatpump_flow_temp_sensor"] = "sensor.flow_temperature"
        params["retrieve_hass_conf"]["heatpump_power_sensor"] = "sensor.kwh_meter"
        params["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = "sensor.gas_meter"
        # _append_hybrid_heatpump_forecast_targets only runs inside build_params
        # (i.e. when hybrid_heatpump_forecast_enabled is already True *before*
        # the config pipeline builds this params blob) - register the same
        # entities by hand here, matching _build_heating_forecast_input_data_dict.
        params.setdefault("passed_data", {})
        params["passed_data"]["custom_hybrid_electric_forecast_id"] = {
            "entity_id": "sensor.hybrid_heatpump_electric_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Hybrid Heat Pump Electric Power Forecast",
        }
        params["passed_data"]["custom_hybrid_gas_forecast_id"] = {
            "entity_id": "sensor.hybrid_heatpump_gas_forecast",
            "device_class": "gas",
            "unit_of_measurement": "m³",
            "friendly_name": "Hybrid Heat Pump Gas Consumption Forecast",
        }
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "hybrid-heatpump-forecast",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=1, freq="30min")
        rh.get_data = AsyncMock(return_value=True)
        rh.prepare_data = Mock()
        rh.df_final = pd.DataFrame(
            {
                "sensor.hp_duty": [0.6],
                "sensor.indoor_temperature": [20.0],
                "sensor.flow_temperature": [35.0],
                "sensor.kwh_meter": [350.0],
                "sensor.gas_meter": [0.0],
            },
            index=idx,
        )
        rh.post_data = AsyncMock(return_value=True)

        weather_idx = pd.date_range(start=pd.Timestamp.now(tz="UTC"), periods=48, freq="30min")
        df_weather = pd.DataFrame(
            {"temp_air": -5.0, "wind_speed": 5.0, "ghi": 0.0},
            index=weather_idx,
        )
        input_data_dict["fcst"].get_weather_forecast = AsyncMock(return_value=df_weather)
        return input_data_dict

    async def test_compute_hybrid_heatpump_forecast_disabled_returns_none(self):
        input_data_dict = await self._build_hybrid_forecast_input_data_dict()
        input_data_dict["optim_conf"]["hybrid_heatpump_forecast_enabled"] = False

        result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_hybrid_heatpump_forecast_missing_model_returns_none(self):
        input_data_dict = await self._build_hybrid_forecast_input_data_dict()

        with patch(
            "emhass.command_line.load_pickle_blob", AsyncMock(return_value=None)
        ):
            result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_hybrid_heatpump_forecast_publishes_both_sensors(self):
        input_data_dict = await self._build_hybrid_forecast_input_data_dict()
        fake_model = self._FakeHybridModel(elec_value=400.0, gas_value=0.02)

        with patch(
            "emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)
        ):
            result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertEqual(result["forecast_steps"], 48)

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        self.assertEqual(len(call_args_list), 2)
        published_entities = {args[2]: kwargs.get("type_var") for args, kwargs in call_args_list}
        self.assertEqual(
            published_entities.get("sensor.hybrid_heatpump_electric_forecast"), "power"
        )
        self.assertEqual(
            published_entities.get("sensor.hybrid_heatpump_gas_forecast"), "energy"
        )
        # Autoregressive loop always feeds the fake model's own constant
        # prediction back as next step's lag - result should equal that
        # constant for every step, not drift or default to 0.
        self.assertFalse(result["electric_only"])
        self.assertAlmostEqual(result["mean_electric_forecast_w"], 400.0)
        self.assertAlmostEqual(result["mean_gas_forecast_m3"], 0.02)

    async def test_compute_hybrid_heatpump_forecast_uses_aggregate_duty_trajectory(self):
        # With a solved dispatch plan available, the duty fed into the model
        # must follow that plan's per-step P_deferrable0/heatpump_nominal_power
        # trajectory - not the single frozen heatpump_duty_sensor reading
        # (which test_compute_hybrid_heatpump_forecast_publishes_both_sensors
        # implicitly covers via the no-solved-plan fallback path).
        input_data_dict = await self._build_hybrid_forecast_input_data_dict()
        input_data_dict["params"]["passed_data"]["room_load_indices"] = {"room_1": 0}
        input_data_dict["plant_conf"]["heatpump_nominal_power"] = 1000.0
        fake_model = self._FakeHybridModel(elec_value=400.0, gas_value=0.0)

        # Build a synthetic solved-plan DataFrame spanning the same window
        # the forecast's own weather data will use, with a P_deferrable0
        # trajectory that clearly varies (low -> high -> low).
        n = 48
        plan_idx = pd.date_range(start=pd.Timestamp.now(tz="UTC"), periods=n, freq="30min")
        ramp = np.concatenate([np.linspace(0, 1000, n // 2), np.linspace(1000, 0, n - n // 2)])
        opt_res_latest = pd.DataFrame({"P_deferrable0": ramp}, index=plan_idx)

        with (
            patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)),
            patch("emhass.command_line._load_opt_res_latest", return_value=opt_res_latest),
        ):
            result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        # The recorded per-step duties must show real variation, not be a
        # single repeated constant.
        seen = fake_model.seen_duties
        self.assertGreater(len(seen), 1)
        self.assertGreater(max(seen) - min(seen), 0.3)

    async def test_compute_hybrid_heatpump_forecast_electric_only_skips_gas_publish(self):
        input_data_dict = await self._build_hybrid_forecast_input_data_dict()
        fake_model = self._FakeHybridModel(elec_value=400.0, electric_only=True)

        with patch(
            "emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)
        ):
            result = await compute_hybrid_heatpump_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["electric_only"])
        self.assertIsNone(result["mean_gas_forecast_m3"])
        self.assertIsNone(result["last_gas_consumption_m3"])

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        self.assertEqual(len(call_args_list), 1)
        published_entities = {args[2]: kwargs.get("type_var") for args, kwargs in call_args_list}
        self.assertEqual(
            published_entities.get("sensor.hybrid_heatpump_electric_forecast"), "power"
        )
        self.assertNotIn("sensor.hybrid_heatpump_gas_forecast", published_entities)

    # ------------------------------------------------------------------
    # Multi-room self-learning-physics model (standalone sibling of the
    # hybrid heat pump gas/electric model above, for
    # emhass.thermal.self_learning_physics.SelfLearningPhysicsModel) -
    # electric/gas plus every room's own temperature, with optional learned
    # inter-room coupling persisted to self_learning_physics_coupling.json.
    # ------------------------------------------------------------------

    class _FakeSelfLearningPhysicsModel:
        """Stand-in for SelfLearningPhysicsModel: exercises
        refit_self_learning_physics_model's own control flow (gating,
        whole-house + per-room MAE computation, threshold, save, coupling-
        blob persistence) without depending on the real RLS fit's quality
        on synthetic data."""

        def __init__(
            self, elec_value=300.0, gas_value=0.0, room_temp_value=20.0,
            electric_only=False, coupling=None, pair_conductance_kw_per_k=None,
        ):
            self.elec_value = elec_value
            self.gas_value = gas_value
            self.room_temp_value = room_temp_value
            self.electric_only = electric_only
            self.theta_gas_ = None if electric_only else "stub"
            # coupling: a fixed dict returned regardless of what .fit() was
            # last called with - simplest for tests that only care about the
            # *declared*-pair coupling blob.
            # pair_conductance_kw_per_k: derives the returned dict from
            # whichever neighbor_map .fit() was *most recently* called with
            # - needed for the candidate-probe tests below, since
            # refit_self_learning_physics_model calls .fit() a second time
            # with an all-pairs neighbor_map (on the same shared fake
            # instance, since SelfLearningPhysicsModel is patched to a
            # lambda returning this one object) specifically to probe
            # undeclared pairs.
            self._coupling = coupling
            self._pair_conductance = pair_conductance_kw_per_k
            self._last_neighbor_map: dict[str, list[str]] = {}
            self._last_room_names: list[str] = []
            self._last_dfs_by_room: dict = {}

        def fit(self, df_house, dfs_by_room, y_elec, y_gas, neighbor_map):
            self._last_neighbor_map = {k: list(v) for k, v in neighbor_map.items()}
            self._last_room_names = list(dfs_by_room.keys())
            self._last_dfs_by_room = dfs_by_room
            return self

        @property
        def room_models_(self):
            """Minimal stand-in for the real _RoomModel dict - only shape
            (feature_names/theta_temp/neighbors) needs to be real, since
            refit_self_learning_physics_model's own dispatch-coefficients
            export (see command_line.py, saves
            self_learning_physics_room_dispatch_coefficients.json) just
            serializes these three attributes verbatim, it doesn't inspect
            their values."""
            from types import SimpleNamespace

            return {
                name: SimpleNamespace(
                    feature_names=["bias"],
                    theta_temp=[0.0],
                    neighbors=list(self._last_neighbor_map.get(name, [])),
                )
                for name in self._last_room_names
            }

        def predict_recursive(
            self, df_house_fc, dfs_by_room_fc, initial_room_states,
            initial_house_elec=0.0, initial_house_gas=0.0,
        ):
            n = len(df_house_fc)
            elec = np.full(n, self.elec_value)
            gas = None if self.electric_only else np.full(n, self.gas_value)
            room_temp = {name: np.full(n, self.room_temp_value) for name in dfs_by_room_fc}
            return {"room_temp": room_temp, "electric_power": elec, "gas_consumption": gas}

        def coupling_coefficients_kw_per_k(self, room_thermal_mass_kj_per_k, dt_hours):
            if self._coupling is not None:
                return self._coupling
            if self._pair_conductance is None:
                return {}
            pairs = set()
            for name, neighbors in self._last_neighbor_map.items():
                for neighbor in neighbors:
                    pairs.add(tuple(sorted((name, neighbor))))
            return dict.fromkeys(pairs, self._pair_conductance)

    class _FakeSelfLearningPhysicsForecastModel:
        """Stand-in used by compute_self_learning_physics_forecast tests -
        unlike the refit-side fake above, this one needs a populated
        room_models_ (the forecast function filters the configured room
        list down to whichever rooms the *fitted* model actually covers)."""

        def __init__(
            self, room_names, elec_value=400.0, gas_value=0.02, room_temp_value=21.0,
            electric_only=False,
        ):
            self.room_models_ = dict.fromkeys(room_names, object())
            self.theta_gas_ = None if electric_only else "stub"
            self.elec_value = elec_value
            self.gas_value = gas_value
            self.room_temp_value = room_temp_value
            self.electric_only = electric_only
            # Records the whole-horizon duty column seen on the (single)
            # predict_recursive call - unlike HybridHeatPumpLR's predict(),
            # which command_line.py calls once per step in an autoregressive
            # loop, SelfLearningPhysicsModel does its own per-row recursion
            # *inside* predict_recursive, so command_line.py only ever calls
            # it once per forecast with the whole horizon's DataFrame - see
            # test_compute_self_learning_physics_forecast_uses_aggregate_duty_trajectory.
            self.seen_duties = []
            # Records each room's blind_position column (if present) as seen
            # on the (single) predict_recursive call - see
            # test_compute_self_learning_physics_forecast_holds_blind_position_flat.
            self.seen_blind_positions: dict = {}
            # Records whether opening_open/door_open columns were present at
            # all on the (single) predict_recursive call - see
            # test_compute_self_learning_physics_forecast_never_populates_opening_or_door_open.
            self.seen_opening_open_columns: dict = {}
            self.seen_door_open_columns: dict = {}

        def predict_recursive(
            self, df_house_fc, dfs_by_room_fc, initial_room_states,
            initial_house_elec=0.0, initial_house_gas=0.0,
        ):
            n = len(df_house_fc)
            self.seen_duties = df_house_fc["heatpump_duty"].tolist()
            self.seen_blind_positions = {
                name: (
                    df["blind_position"].tolist() if "blind_position" in df.columns else None
                )
                for name, df in dfs_by_room_fc.items()
            }
            self.seen_opening_open_columns = {
                name: "opening_open" in df.columns for name, df in dfs_by_room_fc.items()
            }
            self.seen_door_open_columns = {
                name: "door_open" in df.columns for name, df in dfs_by_room_fc.items()
            }
            elec = np.full(n, self.elec_value)
            gas = None if self.electric_only else np.full(n, self.gas_value)
            room_temp = {name: np.full(n, self.room_temp_value) for name in dfs_by_room_fc}
            return {"room_temp": room_temp, "electric_power": elec, "gas_consumption": gas}

    async def _build_self_learning_physics_refit_input_data_dict(
        self,
        n_rows: int = 2000,
        room_names: tuple[str, ...] = ("Living Room", "Bedroom"),
        with_gas: bool = True,
        with_coupling: bool = False,
        with_blind: bool = False,
        with_window: bool = False,
        with_door: bool = False,
    ):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["self_learning_physics_refit_enabled"] = True
        params["optim_conf"]["self_learning_physics_refit_window_days"] = 60
        params["optim_conf"]["self_learning_physics_refit_max_electric_mae_w"] = 150.0
        params["optim_conf"]["self_learning_physics_refit_max_gas_mae_m3"] = 0.02
        params["optim_conf"]["heatpump_room_names"] = list(room_names)
        params["optim_conf"]["heatpump_room_volume"] = [15.0] * len(room_names)
        params["optim_conf"]["heatpump_room_coupled_neighbors"] = (
            ["1", "0"] if with_coupling else [""] * len(room_names)
        )
        params["retrieve_hass_conf"]["use_influxdb"] = True
        params["retrieve_hass_conf"]["heatpump_power_sensor"] = "sensor.kwh_meter"
        params["retrieve_hass_conf"]["heatpump_duty_sensor"] = "sensor.hp_duty"
        params["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = (
            "sensor.gas_meter" if with_gas else ""
        )
        params["retrieve_hass_conf"]["heatpump_outdoor_temp_sensor"] = "sensor.outdoor_temperature"
        room_sensors = [f"sensor.room_temp_{i}" for i in range(len(room_names))]
        params["retrieve_hass_conf"]["heatpump_room_temp_sensors"] = room_sensors
        # Only the FIRST room gets a configured blind sensor - lets tests
        # confirm the second (unconfigured) room simply doesn't get a
        # blind_position column at all, rather than one full of NaN/0.
        blind_sensors = [""] * len(room_names)
        if with_blind and room_names:
            blind_sensors[0] = "cover.living_room_blind_position"
        params["retrieve_hass_conf"]["heatpump_room_blind_sensors"] = blind_sensors
        # Same "first room only" convention as blind, for window and door
        # sensors - independent columns, so all three may be configured at
        # once without conflict.
        window_sensors = [""] * len(room_names)
        if with_window and room_names:
            window_sensors[0] = "binary_sensor.living_room_window"
        params["retrieve_hass_conf"]["heatpump_room_window_sensors"] = window_sensors
        door_sensors = [""] * len(room_names)
        if with_door and room_names:
            door_sensors[0] = "binary_sensor.living_room_door"
        params["retrieve_hass_conf"]["heatpump_room_door_sensors"] = door_sensors
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "self-learning-physics-refit",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        rh.get_data = AsyncMock(return_value=True)
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_rows, freq="15min")
        data = {
            "sensor.kwh_meter": 300.0,
            "sensor.hp_duty": 0.5,
            "sensor.outdoor_temperature": 5.0,
        }
        if with_gas:
            data["sensor.gas_meter"] = 0.0
        for i, sensor in enumerate(room_sensors):
            data[sensor] = 20.0 + i
        if with_blind and room_names:
            data["cover.living_room_blind_position"] = 0.4
        if with_window and room_names:
            data["binary_sensor.living_room_window"] = 1.0  # open
        if with_door and room_names:
            data["binary_sensor.living_room_door"] = 0.0  # closed
        rh.df_final = pd.DataFrame(data, index=idx)
        return input_data_dict

    async def test_refit_self_learning_physics_model_disabled_returns_none(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_refit_enabled"] = False

        result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_self_learning_physics_model_requires_influxdb(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["use_influxdb"] = False

        result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_self_learning_physics_model_requires_required_sensors(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["heatpump_duty_sensor"] = ""

        result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_self_learning_physics_model_requires_rooms_with_temp_sensors(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["retrieve_hass_conf"]["heatpump_room_temp_sensors"] = ["", ""]

        result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_self_learning_physics_model_too_few_rows_returns_none(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(n_rows=10)

        result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNone(result)

    async def test_refit_self_learning_physics_model_deploys_good_fit_hybrid(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=True)
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)) as mock_save_pkl,
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(result["electric_only"])
        self.assertTrue(result["deployed"])
        self.assertLess(result["electric_mae_w"], 150.0)
        self.assertLess(result["gas_mae_m3"], 0.02)
        self.assertEqual(result["n_rooms"], 2)
        mock_save_pkl.assert_awaited_once()
        self.assertEqual(mock_save_pkl.call_args[0][1], "self_learning_physics_model.pkl")
        # Two JSON blobs saved on every successful deploy: the (possibly
        # empty) coupling blob and the per-room dispatch-coefficients blob
        # (see test_refit_self_learning_physics_model_saves_dispatch_coefficients_blob
        # below for the latter's own content).
        saved_json_filenames = [call.args[1] for call in mock_save_json.await_args_list]
        self.assertIn("self_learning_physics_coupling.json", saved_json_filenames)
        self.assertIn(
            "self_learning_physics_room_dispatch_coefficients.json", saved_json_filenames
        )

    async def test_refit_self_learning_physics_model_converts_cumulative_gas_meter(self):
        """A raw cumulative gas totalizer (the standard HA convention for a
        state_class=total_increasing sensor - see
        utils.resolve_incremental_series) must be converted to a per-
        interval delta before it's used as the gas fit target. Without that
        conversion, a correctly-scaled small prediction would be compared
        against a raw ~2000 m3 lifetime meter reading, producing a massive,
        meaningless gas MAE that fails the deploy gate no matter how good
        the underlying fit actually is - this reproduces that real bug."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=True)
        n_rows = len(input_data_dict["rh"].df_final)
        # A realistic lifetime gas totalizer: starts around 2000 m3, rises
        # slowly and steadily (constant per-row delta of ~0.0025 m3).
        cumulative_gas = 2000.0 + np.linspace(0.0, 5.0, n_rows)
        input_data_dict["rh"].df_final["sensor.gas_meter"] = cumulative_gas

        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0025, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        # Without the conversion this would be on the order of 2000+ (a
        # ~0.0025 prediction against a ~2000-2005 raw meter reading).
        self.assertLess(result["gas_mae_m3"], 1.0)
        self.assertTrue(result["deployed"])

    async def test_refit_self_learning_physics_model_populates_blind_position_for_configured_room_only(
        self,
    ):
        """Only the room with a configured heatpump_room_blind_sensors entry
        should get a 'blind_position' column in its refit training
        DataFrame - the other room (no configured blind sensor) should have
        no such column at all, since self_learning_physics.py's own
        _physics_features already defaults a missing column to 0.0 (blind
        always open = inert) rather than needing an explicit all-zero
        column here."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(
            with_gas=True, with_blind=True
        )
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        dfs_by_room = fake_model._last_dfs_by_room
        self.assertIn("Living Room", dfs_by_room)
        self.assertIn("Bedroom", dfs_by_room)
        self.assertIn("blind_position", dfs_by_room["Living Room"].columns)
        self.assertTrue(
            (dfs_by_room["Living Room"]["blind_position"] == 0.4).all(),
            "Living Room's blind_position column should hold the configured "
            "sensor's constant reading",
        )
        self.assertNotIn(
            "blind_position",
            dfs_by_room["Bedroom"].columns,
            "Bedroom has no configured blind sensor and should not get a "
            "blind_position column at all",
        )

    async def test_refit_self_learning_physics_model_populates_opening_and_door_open_columns(self):
        """Only the room with configured window/door sensors gets
        'opening_open'/'door_open' training columns - opening_open is the OR
        of window and door history, door_open reflects the door alone. The
        other room gets neither column at all (defaults to 0.0/closed via
        _physics_features)."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(
            with_gas=True, with_window=True, with_door=True
        )
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        dfs_by_room = fake_model._last_dfs_by_room
        # Living Room: window=open(1.0), door=closed(0.0) -> opening_open=1.0
        # (OR'd), door_open=0.0.
        self.assertIn("opening_open", dfs_by_room["Living Room"].columns)
        self.assertTrue((dfs_by_room["Living Room"]["opening_open"] == 1.0).all())
        self.assertIn("door_open", dfs_by_room["Living Room"].columns)
        self.assertTrue((dfs_by_room["Living Room"]["door_open"] == 0.0).all())
        self.assertNotIn("opening_open", dfs_by_room["Bedroom"].columns)
        self.assertNotIn("door_open", dfs_by_room["Bedroom"].columns)

    async def test_refit_self_learning_physics_model_fetches_blind_window_door_entities(self):
        """Regression test for a real, confirmed pre-existing bug: the
        blind/window/door entity maps must be resolved BEFORE all_entities
        is built, so their entity ids actually reach rh.get_data's fetch
        list - previously blind_entity_map was resolved only after that
        fetch already ran, so blind_position training data was silently
        absent from every real refit."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(
            with_gas=True, with_blind=True, with_window=True, with_door=True
        )
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        rh = input_data_dict["rh"]
        rh.get_data.assert_awaited_once()
        fetched_entities = rh.get_data.await_args.args[1]
        self.assertIn("cover.living_room_blind_position", fetched_entities)
        self.assertIn("binary_sensor.living_room_window", fetched_entities)
        self.assertIn("binary_sensor.living_room_door", fetched_entities)

    async def test_refit_self_learning_physics_model_converts_cumulative_electric_meter(self):
        """A raw cumulative electricity meter (kWh totalizer) fed into
        heatpump_power_sensor must be converted to an average power in W
        (delta / dt_hours * 1000) before it's used as the electric fit
        target - heatpump_power_sensor's own documented contract is
        real-time power in watts, not cumulative energy."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=False)
        n_rows = len(input_data_dict["rh"].df_final)
        # A cumulative kWh totalizer rising by exactly 0.075 kWh every 15-min
        # (0.25h) step -> average power = 0.075/0.25*1000 = 300 W.
        cumulative_kwh = 1000.0 + np.arange(n_rows) * 0.075
        input_data_dict["rh"].df_final["sensor.kwh_meter"] = cumulative_kwh

        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, room_temp_value=20.5, electric_only=True
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        # Without the conversion this would be enormous (a ~300 W prediction
        # against a raw ~1000-1150 kWh cumulative reading).
        self.assertLess(result["electric_mae_w"], 10.0)
        self.assertTrue(result["deployed"])

    async def test_refit_self_learning_physics_model_saves_dispatch_coefficients_blob(self):
        """The per-room dispatch-coefficients artifact (consumed by
        utils.py::_append_room_thermal_loads for a heatpump_room_self_learning_only
        room) must contain every fitted room's own feature_names/theta/
        neighbors, verbatim from room_models_ - and must NOT be saved when
        the fit is rejected (same quality gate as the pickle/coupling blob)."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=True)
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertTrue(result["deployed"])
        dispatch_call = next(
            call for call in mock_save_json.call_args_list
            if call.args[1] == "self_learning_physics_room_dispatch_coefficients.json"
        )
        saved_payload = dispatch_call.args[2]
        self.assertEqual(set(saved_payload["rooms"].keys()), {"Living Room", "Bedroom"})
        for room_payload in saved_payload["rooms"].values():
            self.assertEqual(room_payload["feature_names"], ["bias"])
            self.assertEqual(room_payload["theta"], [0.0])
            self.assertEqual(room_payload["neighbors"], [])

    async def test_refit_self_learning_physics_model_saves_residual_std_c(self):
        """The dispatch-coefficients blob and the refit's own result dict
        must both carry a per-room holdout residual std (residual_std_c) -
        the Kalman opening detector's own measurement-noise variance R for
        that room (see opening_kalman_detector.py). The fixture's holdout
        room_temp is a flat constant per room and the fake model predicts a
        flat constant too, so the residual is itself perfectly constant -
        std must be exactly 0.0, a precise, deterministic check."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=True)
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertTrue(result["deployed"])
        self.assertIn("room_temp_residual_std_c", result)
        for std in result["room_temp_residual_std_c"].values():
            self.assertAlmostEqual(std, 0.0)

        dispatch_call = next(
            call for call in mock_save_json.call_args_list
            if call.args[1] == "self_learning_physics_room_dispatch_coefficients.json"
        )
        saved_payload = dispatch_call.args[2]
        for room_payload in saved_payload["rooms"].values():
            self.assertIn("residual_std_c", room_payload)
            self.assertAlmostEqual(room_payload["residual_std_c"], 0.0)

    async def test_refit_self_learning_physics_model_filters_rooms_that_lose_to_physics_baseline(self):
        """Whole-model deploy only depends on electric/gas MAE now - but a
        room's own coefficients are only exported into the dispatch blob
        when its self-learning MAE actually beats what the physics/simple
        fallback would have predicted for that SAME room over the SAME
        holdout window (see _score_physics_baseline_room_maes). A room
        whose self-learning fit is far worse than its own physics baseline
        must be excluded, even though the whole-house model still deploys
        and the other room (an accurate fit) is still included."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=True)

        class _PerRoomFakeModel:
            def __init__(self):
                self.theta_gas_ = "stub"
                self._last_room_names: list[str] = []

            def fit(self, df_house, dfs_by_room, y_elec, y_gas, neighbor_map):
                self._last_room_names = list(dfs_by_room.keys())
                return self

            @property
            def room_models_(self):
                from types import SimpleNamespace

                return {
                    name: SimpleNamespace(feature_names=["bias"], theta_temp=[0.0], neighbors=[])
                    for name in self._last_room_names
                }

            def predict_recursive(
                self, df_house_fc, dfs_by_room_fc, initial_room_states,
                initial_house_elec=0.0, initial_house_gas=0.0,
            ):
                n = len(df_house_fc)
                room_temp = {}
                for name in dfs_by_room_fc:
                    # "Living Room"'s real holdout history is a flat 20.0 -
                    # predicting it exactly beats any physics baseline that
                    # drifts away from it. "Bedroom"'s real history is a
                    # flat 21.0 - predicting a wildly wrong flat 100.0 must
                    # lose even to a physics baseline that itself drifts.
                    room_temp[name] = np.full(n, 20.0 if name == "Living Room" else 100.0)
                return {
                    "room_temp": room_temp,
                    "electric_power": np.full(n, 300.0),
                    "gas_consumption": np.full(n, 0.0),
                }

            def coupling_coefficients_kw_per_k(self, room_thermal_mass_kj_per_k, dt_hours):
                return {}

        fake_model = _PerRoomFakeModel()

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertTrue(result["deployed"])
        self.assertIn("Living Room", result["room_temp_physics_baseline_mae_c"])
        self.assertIn("Bedroom", result["room_temp_physics_baseline_mae_c"])
        self.assertIn("Living Room", result["rooms_using_self_learning_dispatch"])
        self.assertNotIn("Bedroom", result["rooms_using_self_learning_dispatch"])

        dispatch_call = next(
            call for call in mock_save_json.call_args_list
            if call.args[1] == "self_learning_physics_room_dispatch_coefficients.json"
        )
        saved_payload = dispatch_call.args[2]
        self.assertIn("Living Room", saved_payload["rooms"])
        self.assertNotIn("Bedroom", saved_payload["rooms"])

    async def test_refit_self_learning_physics_model_rejects_bad_fit_skips_dispatch_blob(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=10000.0, gas_value=5.0, room_temp_value=100.0
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertFalse(result["deployed"])
        mock_save_json.assert_not_awaited()

    async def test_refit_self_learning_physics_model_saves_coupling_blob_with_learned_pairs(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_coupling=True)
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5,
            coupling={("Bedroom", "Living Room"): 0.055},
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertTrue(result["deployed"])
        coupling_call = next(
            call for call in mock_save_json.call_args_list
            if call.args[1] == "self_learning_physics_coupling.json"
        )
        saved_payload = coupling_call.args[2]
        self.assertEqual(
            saved_payload["pairs"],
            [{"room_a": "Bedroom", "room_b": "Living Room", "conductance_kw_per_k": 0.055}],
        )

    async def test_refit_self_learning_physics_model_surfaces_undeclared_pair_as_candidate(self):
        # 3 rooms: Living Room <-> Bedroom is manually declared (a placeholder
        # conductance); Attic has no declared neighbor at all. The fake's
        # coupling estimate is the same magnitude for every pair it's asked
        # about, so the declared_pairs filter in
        # refit_self_learning_physics_model itself is what decides which
        # pairs surface as "candidates" - here, both of Attic's undeclared
        # pairs (with Bedroom and with Living Room), but not the already-
        # declared Living Room <-> Bedroom pair.
        room_names = ("Living Room", "Bedroom", "Attic")
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(
            room_names=room_names, with_coupling=False
        )
        input_data_dict["optim_conf"]["heatpump_room_coupled_neighbors"] = ["1", "0", ""]
        input_data_dict["optim_conf"]["heatpump_room_coupling_conductance"] = ["0.05", "0.05", ""]
        fake_model = self._FakeSelfLearningPhysicsModel(
            room_temp_value=20.5, pair_conductance_kw_per_k=0.4
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertTrue(result["deployed"])
        candidates = result["candidate_couplings"]
        candidate_pairs = {frozenset((c["room_a"], c["room_b"])) for c in candidates}
        self.assertEqual(
            candidate_pairs,
            {frozenset({"Attic", "Bedroom"}), frozenset({"Attic", "Living Room"})},
        )
        self.assertNotIn(frozenset({"Living Room", "Bedroom"}), candidate_pairs)
        for candidate in candidates:
            self.assertAlmostEqual(candidate["suggested_conductance_kw_per_k"], 0.4)

        saved_filenames = [call.args[1] for call in mock_save_json.await_args_list]
        self.assertIn("self_learning_physics_coupling_candidates.json", saved_filenames)

    async def test_refit_self_learning_physics_model_filters_weak_candidate_below_noise_floor(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(
            with_coupling=False
        )
        fake_model = self._FakeSelfLearningPhysicsModel(
            room_temp_value=20.5, pair_conductance_kw_per_k=0.005
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertEqual(result["candidate_couplings"], [])
        saved_filenames = [call.args[1] for call in mock_save_json.await_args_list]
        self.assertNotIn("self_learning_physics_coupling_candidates.json", saved_filenames)

    async def test_refit_self_learning_physics_model_rejects_bad_fit(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=10000.0, gas_value=5.0, room_temp_value=100.0
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)) as mock_save_pkl,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(result["deployed"])
        mock_save_pkl.assert_not_awaited()

    async def test_refit_self_learning_physics_model_electric_only_skips_gas_mae(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict(with_gas=False)
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, room_temp_value=20.5, electric_only=True
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["electric_only"])
        self.assertTrue(result["deployed"])
        self.assertIsNone(result["gas_mae_m3"])

    async def test_refit_self_learning_physics_model_opening_relabel_disabled_by_default(self):
        """self_learning_physics_opening_relabel_enabled defaults to False -
        _em_relabel_opening_open must never even be called, let alone change
        anything, unless a config explicitly opts in."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line._em_relabel_opening_open") as mock_relabel,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["deployed"])
        mock_relabel.assert_not_called()

    async def test_refit_self_learning_physics_model_opening_relabel_enabled_feeds_final_fit(self):
        """When enabled, _em_relabel_opening_open's returned (blended)
        dfs_by_room must actually reach the deployed model's own .fit() call
        - not just an earlier probe pass - proving the relabeled data isn't
        silently dropped before the split/final fit."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_opening_relabel_enabled"] = True
        input_data_dict["optim_conf"]["self_learning_physics_opening_relabel_iterations"] = 1
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        def _fake_relabel(df_raw, dfs_by_room, *args, **kwargs):
            blended = {name: df.assign(_relabel_marker=1.0) for name, df in dfs_by_room.items()}
            return blended, {}

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch(
                "emhass.command_line._em_relabel_opening_open", side_effect=_fake_relabel
            ) as mock_relabel,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        mock_relabel.assert_called_once()
        self.assertTrue(
            all(
                "_relabel_marker" in df.columns for df in fake_model._last_dfs_by_room.values()
            ),
            "The EM-relabeled dfs_by_room must reach the deployed model's own "
            ".fit() call, not just an earlier probe.",
        )

    async def test_refit_self_learning_physics_model_surfaces_candidate_opening_events(self):
        """Phase 3: a contiguous is_open run in Phase 2's diagnostics for an
        unsensored room must surface as a result["candidate_openings"] entry
        and get persisted to its own blob - informational only, mirroring
        candidate_couplings' own already-tested behaviour above."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_opening_relabel_enabled"] = True
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        def _fake_relabel(df_raw, dfs_by_room, *args, **kwargs):
            n = len(dfs_by_room["Bedroom"])
            is_open = np.zeros(n, dtype=bool)
            is_open[10:15] = True  # one contiguous 5-step candidate event
            diagnostics = {
                "Bedroom": {
                    "is_open": is_open,
                    "innovation": np.full(n, 0.5),
                    "s": np.full(n, 0.01),
                }
            }
            return dfs_by_room, diagnostics

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch(
                "emhass.command_line.save_json_blob", AsyncMock(return_value=True)
            ) as mock_save_json,
            patch("emhass.command_line._em_relabel_opening_open", side_effect=_fake_relabel),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["deployed"])
        self.assertEqual(len(result["candidate_openings"]), 1)
        candidate = result["candidate_openings"][0]
        self.assertEqual(candidate["room"], "Bedroom")
        self.assertEqual(candidate["n_steps"], 5)

        saved_filenames = [call.args[1] for call in mock_save_json.await_args_list]
        self.assertIn("self_learning_physics_opening_candidates.json", saved_filenames)

    async def test_refit_self_learning_physics_model_no_candidate_openings_when_relabel_disabled(
        self,
    ):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch(
                "emhass.command_line.save_json_blob", AsyncMock(return_value=True)
            ) as mock_save_json,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertEqual(result["candidate_openings"], [])
        saved_filenames = [call.args[1] for call in mock_save_json.await_args_list]
        self.assertNotIn("self_learning_physics_opening_candidates.json", saved_filenames)

    async def test_refit_self_learning_physics_model_caps_candidate_openings_per_room(self):
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_opening_relabel_enabled"] = True
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        def _fake_relabel(df_raw, dfs_by_room, *args, **kwargs):
            n = len(dfs_by_room["Bedroom"])
            is_open = np.zeros(n, dtype=bool)
            # 8 separate, well-spaced single-step events - more than the cap.
            for i in range(8):
                is_open[10 + i * 20] = True
            diagnostics = {
                "Bedroom": {
                    "is_open": is_open,
                    "innovation": np.full(n, 0.5),
                    "s": np.full(n, 0.01),
                }
            }
            return dfs_by_room, diagnostics

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line._em_relabel_opening_open", side_effect=_fake_relabel),
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertEqual(len(result["candidate_openings"]), _CANDIDATE_OPENING_EVENT_MAX_PER_ROOM)

    async def test_refit_self_learning_physics_model_opening_confirm_disabled_by_default(self):
        """self_learning_physics_opening_confirm_enabled defaults to False -
        neither the resolve nor the publish half of Phase 4's HA
        confirmation loop should ever run unless a config explicitly opts
        in."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch(
                "emhass.command_line._resolve_opening_confirmations", AsyncMock()
            ) as mock_resolve,
            patch(
                "emhass.command_line._publish_opening_confirmation_questions", AsyncMock()
            ) as mock_publish,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        mock_resolve.assert_not_called()
        mock_publish.assert_not_called()

    async def test_refit_self_learning_physics_model_opening_confirm_enabled_resolves_and_publishes(
        self,
    ):
        """When enabled: _resolve_opening_confirmations must run early and
        its result must actually reach _em_relabel_opening_open's
        confirmed_overrides argument (not get dropped along the way), and
        _publish_opening_confirmation_questions must run last with exactly
        the same candidate_openings the result dict itself reports."""
        input_data_dict = await self._build_self_learning_physics_refit_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_opening_relabel_enabled"] = True
        input_data_dict["optim_conf"]["self_learning_physics_opening_confirm_enabled"] = True
        fake_model = self._FakeSelfLearningPhysicsModel(
            elec_value=300.0, gas_value=0.0, room_temp_value=20.5
        )

        captured = {}

        def _fake_relabel(df_raw, dfs_by_room, *args, confirmed_overrides=None, **kwargs):
            captured["confirmed_overrides"] = confirmed_overrides
            n = len(dfs_by_room["Bedroom"])
            is_open = np.zeros(n, dtype=bool)
            is_open[10:13] = True
            diagnostics = {
                "Bedroom": {
                    "is_open": is_open,
                    "innovation": np.full(n, 0.5),
                    "s": np.full(n, 0.01),
                }
            }
            return dfs_by_room, diagnostics

        async def _fake_resolve(rh, emhass_conf, optim_conf, retrieve_hass_conf, logger):
            # A range spanning the whole refit window, so every timestamp
            # in Bedroom's history gets expanded into an override.
            return {
                "Bedroom": [
                    {
                        "start_iso": "2020-01-01T00:00:00+00:00",
                        "end_iso": "2030-01-01T00:00:00+00:00",
                        "value": 0.0,
                    }
                ]
            }

        with (
            patch(
                "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
                lambda *a, **kw: fake_model,
            ),
            patch("emhass.command_line.save_pickle_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch("emhass.command_line._em_relabel_opening_open", side_effect=_fake_relabel),
            patch(
                "emhass.command_line._resolve_opening_confirmations", side_effect=_fake_resolve
            ) as mock_resolve,
            patch(
                "emhass.command_line._publish_opening_confirmation_questions", AsyncMock()
            ) as mock_publish,
        ):
            result = await refit_self_learning_physics_model(input_data_dict, logger)

        self.assertIsNotNone(result)
        mock_resolve.assert_awaited_once()
        self.assertTrue(captured["confirmed_overrides"])
        self.assertTrue(
            all(v == 0.0 for v in captured["confirmed_overrides"]["Bedroom"].values())
        )

        mock_publish.assert_awaited_once()
        published_candidates = mock_publish.call_args[0][4]
        self.assertEqual(published_candidates, result["candidate_openings"])

    async def _build_self_learning_physics_forecast_input_data_dict(
        self,
        room_names: tuple[str, ...] = ("Living Room", "Bedroom"),
        with_gas: bool = True,
        with_blind: bool = False,
        with_window: bool = False,
        with_door: bool = False,
    ):
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["self_learning_physics_forecast_enabled"] = True
        params["optim_conf"]["heatpump_room_names"] = list(room_names)
        params["retrieve_hass_conf"]["heatpump_duty_sensor"] = "sensor.hp_duty"
        params["retrieve_hass_conf"]["heatpump_flow_temp_sensor"] = "sensor.flow_temperature"
        params["retrieve_hass_conf"]["heatpump_power_sensor"] = "sensor.kwh_meter"
        params["retrieve_hass_conf"]["heatpump_gas_meter_sensor"] = (
            "sensor.gas_meter" if with_gas else ""
        )
        room_sensors = [f"sensor.room_temp_{i}" for i in range(len(room_names))]
        params["retrieve_hass_conf"]["heatpump_room_temp_sensors"] = room_sensors
        # Only the FIRST room gets a configured blind sensor, same convention
        # as _build_self_learning_physics_refit_input_data_dict's with_blind.
        blind_sensors = [""] * len(room_names)
        if with_blind and room_names:
            blind_sensors[0] = "cover.living_room_blind_position"
        params["retrieve_hass_conf"]["heatpump_room_blind_sensors"] = blind_sensors
        window_sensors = [""] * len(room_names)
        if with_window and room_names:
            window_sensors[0] = "binary_sensor.living_room_window"
        params["retrieve_hass_conf"]["heatpump_room_window_sensors"] = window_sensors
        door_sensors = [""] * len(room_names)
        if with_door and room_names:
            door_sensors[0] = "binary_sensor.living_room_door"
        params["retrieve_hass_conf"]["heatpump_room_door_sensors"] = door_sensors

        # _append_self_learning_physics_forecast_targets only runs inside
        # build_params (i.e. when self_learning_physics_forecast_enabled is
        # already True *before* the config pipeline builds this params blob)
        # - register the same entities by hand, matching
        # _build_hybrid_forecast_input_data_dict's own approach.
        params.setdefault("passed_data", {})
        params["passed_data"]["custom_self_learning_physics_electric_forecast_id"] = {
            "entity_id": "sensor.self_learning_physics_electric_forecast",
            "device_class": "power",
            "unit_of_measurement": "W",
            "friendly_name": "Self-Learning-Physics Electric Power Forecast",
        }
        params["passed_data"]["custom_self_learning_physics_gas_forecast_id"] = {
            "entity_id": "sensor.self_learning_physics_gas_forecast",
            "device_class": "gas",
            "unit_of_measurement": "m³",
            "friendly_name": "Self-Learning-Physics Gas Consumption Forecast",
        }
        params["passed_data"]["custom_self_learning_physics_temp_forecast_id"] = [
            {
                "entity_id": f"sensor.self_learning_physics_temp_forecast_{i}",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "friendly_name": f"{name} Self-Learning-Physics Temperature Forecast",
            }
            for i, name in enumerate(room_names)
        ]

        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "self-learning-physics-forecast",
            logger,
            get_data_from_file=True,
        )
        rh = input_data_dict["rh"]
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=1, freq="30min")
        rh.get_data = AsyncMock(return_value=True)
        rh.prepare_data = Mock()
        data = {
            "sensor.hp_duty": [0.6],
            "sensor.flow_temperature": [35.0],
            "sensor.kwh_meter": [350.0],
        }
        if with_gas:
            data["sensor.gas_meter"] = [0.0]
        for i, sensor in enumerate(room_sensors):
            data[sensor] = [20.0 + i]
        if with_blind and room_names:
            data["cover.living_room_blind_position"] = [0.7]
        if with_window and room_names:
            data["binary_sensor.living_room_window"] = [1.0]  # open
        if with_door and room_names:
            data["binary_sensor.living_room_door"] = [1.0]  # open
        rh.df_final = pd.DataFrame(data, index=idx)
        rh.post_data = AsyncMock(return_value=True)

        weather_idx = pd.date_range(start=pd.Timestamp.now(tz="UTC"), periods=48, freq="30min")
        df_weather = pd.DataFrame(
            {"temp_air": -5.0, "wind_speed": 5.0, "ghi": 0.0, "dni": 0.0, "dhi": 0.0},
            index=weather_idx,
        )
        input_data_dict["fcst"].get_weather_forecast = AsyncMock(return_value=df_weather)
        return input_data_dict

    async def test_compute_self_learning_physics_forecast_disabled_returns_none(self):
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict()
        input_data_dict["optim_conf"]["self_learning_physics_forecast_enabled"] = False

        result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_self_learning_physics_forecast_missing_model_returns_none(self):
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict()

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=None)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNone(result)
        input_data_dict["rh"].post_data.assert_not_called()

    async def test_compute_self_learning_physics_forecast_publishes_whole_house_and_per_room(self):
        room_names = ("Living Room", "Bedroom")
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names
        )
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            room_names, elec_value=400.0, gas_value=0.02, room_temp_value=21.5
        )

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertEqual(result["forecast_steps"], 48)
        self.assertEqual(result["n_rooms"], 2)
        self.assertFalse(result["electric_only"])
        self.assertAlmostEqual(result["mean_electric_forecast_w"], 400.0)
        self.assertAlmostEqual(result["mean_gas_forecast_m3"], 0.02)
        self.assertAlmostEqual(result["mean_room_temps_c"]["Living Room"], 21.5)
        self.assertAlmostEqual(result["mean_room_temps_c"]["Bedroom"], 21.5)

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        self.assertEqual(len(call_args_list), 4)  # electric + gas + 2 rooms
        published_entities = {args[2]: kwargs.get("type_var") for args, kwargs in call_args_list}
        self.assertEqual(
            published_entities.get("sensor.self_learning_physics_electric_forecast"), "power"
        )
        self.assertEqual(
            published_entities.get("sensor.self_learning_physics_gas_forecast"), "energy"
        )
        self.assertEqual(
            published_entities.get("sensor.self_learning_physics_temp_forecast_0"), "temperature"
        )
        self.assertEqual(
            published_entities.get("sensor.self_learning_physics_temp_forecast_1"), "temperature"
        )

    async def test_compute_self_learning_physics_forecast_electric_only_skips_gas_publish(self):
        room_names = ("Living Room",)
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names, with_gas=False
        )
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            room_names, elec_value=400.0, room_temp_value=20.0, electric_only=True
        )

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertTrue(result["electric_only"])
        self.assertIsNone(result["mean_gas_forecast_m3"])

        call_args_list = input_data_dict["rh"].post_data.call_args_list
        self.assertEqual(len(call_args_list), 2)  # electric + 1 room, no gas
        published_entities = {args[2]: kwargs.get("type_var") for args, kwargs in call_args_list}
        self.assertNotIn("sensor.self_learning_physics_gas_forecast", published_entities)

    async def test_compute_self_learning_physics_forecast_holds_blind_position_flat(self):
        """A room's current blind reading is a single live snapshot (unlike
        duty, which follows a solved per-step dispatch trajectory) - it
        should be held constant across the whole forecast horizon, mirroring
        how last_supply_temp/duty are already held flat elsewhere in this
        function. The other room (no configured blind sensor) should get no
        blind_position column at all, exactly like the refit side."""
        room_names = ("Living Room", "Bedroom")
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names, with_blind=True
        )
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            room_names, elec_value=400.0, gas_value=0.02, room_temp_value=21.5
        )

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        living_room_blinds = fake_model.seen_blind_positions["Living Room"]
        self.assertIsNotNone(living_room_blinds)
        self.assertEqual(len(living_room_blinds), 48)
        self.assertTrue(
            all(v == 0.7 for v in living_room_blinds),
            "Living Room's blind_position should stay at the live sensor's "
            "current reading across the entire forecast horizon",
        )
        self.assertIsNone(
            fake_model.seen_blind_positions["Bedroom"],
            "Bedroom has no configured blind sensor and should not get a "
            "blind_position column at all",
        )

    async def test_compute_self_learning_physics_forecast_never_populates_opening_or_door_open(
        self,
    ):
        """Deliberate contrast with blind_position (held flat above):
        opening_open/door_open must NEVER be populated at forecast time at
        all, even when window/door sensors are configured and currently
        reading 'open' - a live-only momentary signal has no valid forecast
        for future timesteps, so the published forecast horizon must be left
        to _physics_features's own default-to-0.0 ('assumed closed')
        fallback instead of being held flat like blind_position."""
        room_names = ("Living Room", "Bedroom")
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names, with_window=True, with_door=True
        )
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            room_names, elec_value=400.0, gas_value=0.02, room_temp_value=21.5
        )

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        self.assertFalse(fake_model.seen_opening_open_columns["Living Room"])
        self.assertFalse(fake_model.seen_door_open_columns["Living Room"])
        self.assertFalse(fake_model.seen_opening_open_columns["Bedroom"])
        self.assertFalse(fake_model.seen_door_open_columns["Bedroom"])

    async def test_compute_self_learning_physics_forecast_uses_aggregate_duty_trajectory(self):
        # Same rationale as
        # test_compute_hybrid_heatpump_forecast_uses_aggregate_duty_trajectory:
        # with a solved dispatch plan available, the duty fed into the model
        # must follow that plan's per-step trajectory, not a single frozen
        # heatpump_duty_sensor reading.
        room_names = ("Living Room",)
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names
        )
        input_data_dict["params"]["passed_data"]["room_load_indices"] = {"room_1": 0}
        input_data_dict["plant_conf"]["heatpump_nominal_power"] = 1000.0
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            room_names, elec_value=400.0, gas_value=0.0, room_temp_value=20.0
        )

        n = 48
        plan_idx = pd.date_range(start=pd.Timestamp.now(tz="UTC"), periods=n, freq="30min")
        ramp = np.concatenate([np.linspace(0, 1000, n // 2), np.linspace(1000, 0, n - n // 2)])
        opt_res_latest = pd.DataFrame({"P_deferrable0": ramp}, index=plan_idx)

        with (
            patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)),
            patch("emhass.command_line._load_opt_res_latest", return_value=opt_res_latest),
        ):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNotNone(result)
        seen = fake_model.seen_duties
        self.assertGreater(len(seen), 1)
        self.assertGreater(max(seen) - min(seen), 0.3)

    async def test_compute_self_learning_physics_forecast_room_not_in_model_returns_none(self):
        # The config lists a room the fitted model doesn't cover (e.g. added
        # after the last refit) - must fail loudly rather than silently
        # forecasting a subset of the configured rooms.
        room_names = ("Living Room", "Attic")
        input_data_dict = await self._build_self_learning_physics_forecast_input_data_dict(
            room_names=room_names
        )
        fake_model = self._FakeSelfLearningPhysicsForecastModel(
            ("Kitchen",), elec_value=400.0, room_temp_value=20.0
        )

        with patch("emhass.command_line.load_pickle_blob", AsyncMock(return_value=fake_model)):
            result = await compute_self_learning_physics_forecast(input_data_dict, logger)

        self.assertIsNone(result)

    async def _build_manual_load_input_data_dict(
        self,
        ready=True,
        confirm_sensor=False,
        washdata_device=_UNSET,
        washdata_states=None,
        program_select_value=None,
    ):
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        # Flag the existing default "dishwasher" deferrable load (index 0) as
        # manual, instead of a separate manual-loads section - reuses its own
        # load_names/nominal_power_of_deferrable_loads/operating_hours entry.
        config["manual_load_enabled"] = True
        config["is_manual_load"] = [True, False]
        config["load_names"] = ["Dishwasher", "washing_machine"]
        config["nominal_power_of_deferrable_loads"] = [1800.0, 750.0]
        config["operating_hours_of_each_deferrable_load"] = [2.0, 0]
        config["manual_load_ready_sensor"] = ["input_boolean.dishwasher_ready", ""]
        config["manual_load_deadline_hour"] = ["", ""]
        config["manual_load_confirm_power_sensor"] = (
            ["sensor.dishwasher_power", ""] if confirm_sensor else ["", ""]
        )
        configure_washdata = washdata_device is not _UNSET
        config["load_washdata_enabled"] = [configure_washdata, False]
        config["load_washdata_device"] = [washdata_device, ""] if configure_washdata else ["", ""]
        configure_program_select = program_select_value is not None
        config["manual_load_program_select_sensor"] = (
            ["select.dishwasher_cyclusprogramma", ""] if configure_program_select else ["", ""]
        )
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params.setdefault("passed_data", {})
        params["passed_data"].update(
            {
                "pv_power_forecast": [i + 1 for i in range(48)],
                "load_power_forecast": [i + 1 for i in range(48)],
                "load_cost_forecast": [i + 1 for i in range(48)],
                "prod_price_forecast": [i + 1 for i in range(48)],
            }
        )
        params_json = orjson.dumps(params).decode("utf-8")

        async def _fake_entity_fetch(entity_id):
            if entity_id == "select.dishwasher_cyclusprogramma":
                return {"state": program_select_value}
            return None

        # _resolve_load_profiles runs INSIDE set_input_data_dict (before
        # Forecast/Optimization are built), on the rh instance it constructs
        # internally - so both fetches have to be patched at the class
        # level, wrapping this call, not on an rh object we don't have yet.
        with (
            patch(
                "emhass.retrieve_hass.RetrieveHass.get_all_states",
                AsyncMock(return_value=washdata_states if washdata_states is not None else []),
            ),
            patch(
                "emhass.retrieve_hass.RetrieveHass.get_entity_state_and_attributes",
                AsyncMock(side_effect=_fake_entity_fetch),
            ),
        ):
            input_data_dict = await set_input_data_dict(
                emhass_conf,
                "profit",
                params_json,
                None,
                "dayahead-optim",
                logger,
                get_data_from_file=True,
            )
        rh = input_data_dict["rh"]
        # get_current_state() is a direct REST call (deliberately bypassing
        # use_influxdb/df_final - see _apply_manual_load_runtime_overrides),
        # so it's mocked directly rather than via rh.df_final. Backed by a
        # mutable dict on rh so individual tests can change a value (e.g.
        # the confirm-power-sensor reading) after construction.
        state_values = {"input_boolean.dishwasher_ready": 1.0 if ready else 0.0}
        if confirm_sensor:
            state_values["sensor.dishwasher_power"] = 0.0
        rh._test_state_values = state_values
        rh.get_current_state = AsyncMock(side_effect=lambda entity_id: rh._test_state_values.get(entity_id))
        return input_data_dict

    async def test_resolve_load_profile_success(self):
        """A single discovered WashData program is resolved into a resampled
        sequence, mutated into BOTH optim_conf (the object used to build the
        solver) and params["optim_conf"] (what downstream pinning logic
        reads) - the dual-mutation the whole feature depends on."""
        states = [
            _washdata_program_state(
                "wasmachine", "katoen_40", [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 15, 3
            )
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine", washdata_states=states
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        expected = [150.0, 350.0, 550.0]  # 15min -> 30min (default optimization_time_step)

        for optim_conf in (
            input_data_dict["optim_conf"],
            input_data_dict["params"]["optim_conf"],
        ):
            self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"][k], expected)
            self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"][k], 3)
            self.assertEqual(optim_conf["load_dispatch_mode"][k], "program")

    async def test_resolve_load_profile_no_programs_discovered_falls_back(self):
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine", washdata_states=[]
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]

        for optim_conf in (
            input_data_dict["optim_conf"],
            input_data_dict["params"]["optim_conf"],
        ):
            self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"][k], 1800.0)

    async def test_resolve_load_profile_no_power_profile_attr_falls_back(self):
        states = [
            {
                "entity_id": "sensor.wasmachine_profiel_eco_aantal",
                "state": "1",
                "attributes": {"average_length_min": 125},
            }
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine", washdata_states=states
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        self.assertEqual(
            input_data_dict["optim_conf"]["nominal_power_of_deferrable_loads"][k], 1800.0
        )

    async def test_resolve_load_profile_invalid_interval_falls_back(self):
        states = [_washdata_program_state("wasmachine", "eco", [100.0, 200.0], 0, 1)]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine", washdata_states=states
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        self.assertEqual(
            input_data_dict["optim_conf"]["nominal_power_of_deferrable_loads"][k], 1800.0
        )

    async def test_resolve_load_profile_multiple_programs_picks_most_used(self):
        """With no program-select sensor configured, the discovered program
        with the highest run count ("aantal") wins."""
        states = [
            _washdata_program_state("wasmachine", "eco_20", [50.0, 60.0], 30, 1),
            _washdata_program_state(
                "wasmachine", "katoen_40", [100.0, 200.0, 300.0], 30, 5
            ),
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine", washdata_states=states
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        self.assertEqual(
            input_data_dict["optim_conf"]["operating_hours_of_each_deferrable_load"][k], 3
        )

    async def test_resolve_load_profile_program_select_pins_exact_program(self):
        """A manual load with manual_load_program_select_sensor configured
        (e.g. WashData's own select.<device>_cyclusprogramma) uses that
        program even when it's not the most-used one."""
        states = [
            _washdata_program_state(
                "wasmachine", "eco_20", [50.0, 60.0, 70.0], 30, 5
            ),  # most-used, but not selected
            _washdata_program_state("wasmachine", "katoen_40", [100.0, 200.0], 30, 1),
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine",
            washdata_states=states,
            program_select_value="Katoen 40",
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        self.assertEqual(
            input_data_dict["optim_conf"]["operating_hours_of_each_deferrable_load"][k], 2
        )

    async def test_resolve_load_profile_program_select_auto_detect_falls_back_to_most_used(self):
        states = [
            _washdata_program_state("wasmachine", "eco_20", [50.0, 60.0], 30, 1),
            _washdata_program_state(
                "wasmachine", "katoen_40", [100.0, 200.0, 300.0], 30, 5
            ),
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            washdata_device="wasmachine",
            washdata_states=states,
            program_select_value="auto_detect",
        )
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        self.assertEqual(
            input_data_dict["optim_conf"]["operating_hours_of_each_deferrable_load"][k], 3
        )

    async def test_resolve_load_profile_works_for_non_manual_load(self):
        """load_washdata_device is independent of is_manual_load - an
        automatically-dispatched load (is_manual_load=False) must also get
        its profile resolved. This is the orthogonality the feature exists
        for: "being a washing machine" and "being manually dispatched" are
        separate properties."""
        states = [
            _washdata_program_state(
                "washing_machine", "katoen_40", [100.0, 200.0, 300.0, 400.0], 30, 2
            )
        ]
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        config["number_of_deferrable_loads"] = 2
        config["is_manual_load"] = [False, False]
        config["load_washdata_enabled"] = [False, True]
        config["load_washdata_device"] = ["", "washing_machine"]
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params.setdefault("passed_data", {})
        params["passed_data"].update(
            {
                "pv_power_forecast": [i + 1 for i in range(48)],
                "load_power_forecast": [i + 1 for i in range(48)],
                "load_cost_forecast": [i + 1 for i in range(48)],
                "prod_price_forecast": [i + 1 for i in range(48)],
            }
        )
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "emhass.retrieve_hass.RetrieveHass.get_all_states",
            AsyncMock(return_value=states),
        ):
            input_data_dict = await set_input_data_dict(
                emhass_conf,
                "profit",
                params_json,
                None,
                "dayahead-optim",
                logger,
                get_data_from_file=True,
            )
        self.assertEqual(
            input_data_dict["optim_conf"]["nominal_power_of_deferrable_loads"][1],
            [100.0, 200.0, 300.0, 400.0],
        )
        self.assertEqual(
            input_data_dict["optim_conf"]["load_dispatch_mode"][1], "program"
        )
        # Never flagged manual, so no manual-load bookkeeping was created.
        self.assertEqual(
            input_data_dict["params"]["passed_data"].get("manual_load_indices", {}), {}
        )

    async def test_manual_load_runtime_overrides_pins_resolved_profile_exactly(self):
        """Once a profile is resolved, the exact-pin window width must equal
        the resolved sequence length, not the old hours-derived value -
        required for only one candidate start offset to stay feasible."""
        states = [
            _washdata_program_state(
                "wasmachine", "katoen_40", [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], 15, 3
            )
        ]
        input_data_dict = await self._build_manual_load_input_data_dict(
            ready=False, washdata_device="wasmachine", washdata_states=states
        )
        params = input_data_dict["params"]
        k = params["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        horizon_start = input_data_dict["fcst"].forecast_dates[0]
        committed_start = horizon_start + pd.Timedelta(hours=3)
        commitments = {"Dishwasher": {"committed_start_iso": committed_start.isoformat()}}

        with patch("emhass.command_line.load_json_blob", AsyncMock(return_value=commitments)):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        optim_conf = params["optim_conf"]
        start = optim_conf["start_timesteps_of_each_deferrable_load"][k]
        end = optim_conf["end_timesteps_of_each_deferrable_load"][k]
        self.assertEqual(end - start, 3)  # resolved sequence length, not ceil(2.0h / 0.5h) == 4

    async def test_manual_load_runtime_overrides_ready_no_commitment(self):
        """A newly-requested, uncommitted load gets a flexible (full-horizon)
        window this cycle so the solver can pick a placement."""
        input_data_dict = await self._build_manual_load_input_data_dict(ready=True)
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]

        with patch("emhass.command_line.load_json_blob", AsyncMock(return_value={})):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        optim_conf = input_data_dict["params"]["optim_conf"]
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"][k], 2.0)
        self.assertEqual(optim_conf["start_timesteps_of_each_deferrable_load"][k], 0)
        self.assertEqual(optim_conf["end_timesteps_of_each_deferrable_load"][k], 0)

    async def test_manual_load_runtime_overrides_not_ready_no_commitment(self):
        """A load that hasn't been requested and has no commitment stays idle."""
        input_data_dict = await self._build_manual_load_input_data_dict(ready=False)
        k = input_data_dict["params"]["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]

        with patch("emhass.command_line.load_json_blob", AsyncMock(return_value={})):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        optim_conf = input_data_dict["params"]["optim_conf"]
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"][k], 0)

    async def test_manual_load_runtime_overrides_pins_existing_commitment(self):
        """The core guarantee: once a start time has been committed to (and
        shown to the user), a re-optimization must not move it - regardless
        of the live ready-sensor value this cycle."""
        input_data_dict = await self._build_manual_load_input_data_dict(ready=False)
        params = input_data_dict["params"]
        k = params["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        horizon_start = input_data_dict["fcst"].forecast_dates[0]
        committed_start = horizon_start + pd.Timedelta(hours=3)
        commitments = {"Dishwasher": {"committed_start_iso": committed_start.isoformat()}}

        with patch("emhass.command_line.load_json_blob", AsyncMock(return_value=commitments)):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        optim_conf = params["optim_conf"]
        time_step = params["retrieve_hass_conf"]["optimization_time_step"]
        expected_start = round(pd.Timedelta(hours=3) / time_step)
        expected_duration_steps = round(pd.Timedelta(hours=2) / time_step)
        self.assertEqual(optim_conf["start_timesteps_of_each_deferrable_load"][k], expected_start)
        self.assertEqual(
            optim_conf["end_timesteps_of_each_deferrable_load"][k],
            expected_start + expected_duration_steps,
        )
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"][k], 2.0)

    async def test_manual_load_runtime_overrides_clears_via_confirm_sensor(self):
        """Once the confirmation power sensor shows the appliance actually
        drawing power, the commitment is cleared (and not re-pinned)."""
        input_data_dict = await self._build_manual_load_input_data_dict(
            ready=False, confirm_sensor=True
        )
        input_data_dict["rh"]._test_state_values["sensor.dishwasher_power"] = 1000.0
        params = input_data_dict["params"]
        horizon_start = input_data_dict["fcst"].forecast_dates[0]
        committed_start = horizon_start - pd.Timedelta(minutes=10)
        commitments = {"Dishwasher": {"committed_start_iso": committed_start.isoformat()}}

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=commitments)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        mock_save.assert_awaited_once()
        saved_payload = mock_save.call_args[0][2]
        self.assertNotIn("Dishwasher", saved_payload)
        k = params["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        # ready=False and the commitment was just cleared -> idle this cycle.
        self.assertEqual(params["optim_conf"]["operating_hours_of_each_deferrable_load"][k], 0)

    async def test_manual_load_runtime_overrides_clears_via_deadline_elapsed(self):
        """With no confirmation sensor configured, a commitment whose window
        (start + duration + grace) has fully elapsed is cleared best-effort."""
        input_data_dict = await self._build_manual_load_input_data_dict(ready=False)
        horizon_start = input_data_dict["fcst"].forecast_dates[0]
        committed_start = horizon_start - pd.Timedelta(hours=5)
        commitments = {"Dishwasher": {"committed_start_iso": committed_start.isoformat()}}

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=commitments)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _apply_manual_load_runtime_overrides(input_data_dict, logger)

        mock_save.assert_awaited_once()
        saved_payload = mock_save.call_args[0][2]
        self.assertNotIn("Dishwasher", saved_payload)

    async def test_timestep_index_from_timestamp(self):
        horizon_start = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
        ts = horizon_start + pd.Timedelta(hours=1, minutes=30)
        idx = _timestep_index_from_timestamp(ts, horizon_start, pd.Timedelta(minutes=30))
        self.assertEqual(idx, 3)
        # Clamped at 0 for a timestamp before the horizon start.
        idx_past = _timestep_index_from_timestamp(
            horizon_start - pd.Timedelta(hours=1), horizon_start, pd.Timedelta(minutes=30)
        )
        self.assertEqual(idx_past, 0)

    async def test_resolve_room_blind_entity_map_skips_unnamed_or_unsensored_rooms(self):
        optim_conf = {"heatpump_room_names": ["Living Room", "", "Bedroom"]}
        retrieve_hass_conf = {
            "heatpump_room_blind_sensors": ["cover.living_room_blind", "cover.orphan_blind", ""]
        }
        entity_map = _resolve_room_blind_entity_map(optim_conf, retrieve_hass_conf)
        self.assertEqual(entity_map, {"Living Room": "cover.living_room_blind"})

    async def test_build_room_blind_positions_maps_room_to_load_index(self):
        from types import SimpleNamespace

        input_data_dict = {
            "params": {
                "optim_conf": {
                    "number_of_deferrable_loads": 3,
                    "heatpump_room_names": ["Living Room", "Bedroom"],
                },
                "retrieve_hass_conf": {
                    "heatpump_room_blind_sensors": ["cover.living_room_blind", ""],
                },
                "passed_data": {
                    "room_load_indices": {"Living Room": 0, "Bedroom": 2},
                },
            },
            "rh": SimpleNamespace(
                df_final=pd.DataFrame({"cover.living_room_blind": [0.2, 0.3]})
            ),
        }

        result = _build_room_blind_positions(input_data_dict, logger)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.3)  # Living Room's latest value
        self.assertIsNone(result[1])  # not a room load
        self.assertIsNone(result[2])  # Bedroom has no configured blind sensor

    async def test_build_room_blind_positions_clips_out_of_range_values(self):
        from types import SimpleNamespace

        input_data_dict = {
            "params": {
                "optim_conf": {
                    "number_of_deferrable_loads": 1,
                    "heatpump_room_names": ["Living Room"],
                },
                "retrieve_hass_conf": {
                    "heatpump_room_blind_sensors": ["cover.living_room_blind"],
                },
                "passed_data": {"room_load_indices": {"Living Room": 0}},
            },
            "rh": SimpleNamespace(
                # A raw HA cover.* entity in its native 0-100 convention -
                # exactly the case heatpump_room_blind_sensors's own
                # description warns needs normalizing first.
                df_final=pd.DataFrame({"cover.living_room_blind": [70.0]})
            ),
        }

        result = _build_room_blind_positions(input_data_dict, logger)

        self.assertEqual(result, [1.0])  # clipped, not silently used as-is

    async def test_build_room_blind_positions_no_rooms_returns_none(self):
        input_data_dict = {
            "params": {
                "optim_conf": {"number_of_deferrable_loads": 1},
                "retrieve_hass_conf": {},
                "passed_data": {},
            },
            "rh": None,
        }
        self.assertIsNone(_build_room_blind_positions(input_data_dict, logger))

    async def test_resolve_room_window_and_door_entity_maps_skip_unnamed_or_unsensored_rooms(self):
        optim_conf = {"heatpump_room_names": ["Living Room", "", "Bedroom"]}
        retrieve_hass_conf = {
            "heatpump_room_window_sensors": [
                "binary_sensor.living_room_window",
                "binary_sensor.orphan_window",
                "",
            ],
            "heatpump_room_door_sensors": ["", "", "binary_sensor.bedroom_door"],
        }
        window_map = _resolve_room_window_entity_map(optim_conf, retrieve_hass_conf)
        door_map = _resolve_room_door_entity_map(optim_conf, retrieve_hass_conf)
        self.assertEqual(window_map, {"Living Room": "binary_sensor.living_room_window"})
        self.assertEqual(door_map, {"Bedroom": "binary_sensor.bedroom_door"})

    async def test_build_room_opening_open_ors_window_and_door_per_room(self):
        from types import SimpleNamespace

        input_data_dict = {
            "params": {
                "optim_conf": {
                    "number_of_deferrable_loads": 3,
                    "heatpump_room_names": ["Living Room", "Bedroom", "Kitchen"],
                },
                "retrieve_hass_conf": {
                    "heatpump_room_window_sensors": [
                        "binary_sensor.lr_window",
                        "",
                        "",
                    ],
                    "heatpump_room_door_sensors": [
                        "",
                        "binary_sensor.br_door",
                        "",
                    ],
                },
                "passed_data": {
                    "room_load_indices": {"Living Room": 0, "Bedroom": 1, "Kitchen": 2},
                },
            },
            "rh": SimpleNamespace(
                df_final=pd.DataFrame(
                    {
                        "binary_sensor.lr_window": [1.0],  # open
                        "binary_sensor.br_door": [0.0],  # closed
                    }
                )
            ),
        }

        result = _build_room_opening_open(input_data_dict, logger)

        self.assertEqual(len(result), 3)
        self.assertTrue(result[0])  # Living Room: window open
        self.assertFalse(result[1])  # Bedroom: door closed
        self.assertFalse(result[2])  # Kitchen: no sensor configured -> False, never None
        self.assertIsInstance(result[0], bool)

    async def test_build_room_door_open_ignores_window_sensor(self):
        from types import SimpleNamespace

        input_data_dict = {
            "params": {
                "optim_conf": {
                    "number_of_deferrable_loads": 1,
                    "heatpump_room_names": ["Living Room"],
                },
                "retrieve_hass_conf": {
                    "heatpump_room_window_sensors": ["binary_sensor.lr_window"],
                    "heatpump_room_door_sensors": ["binary_sensor.lr_door"],
                },
                "passed_data": {"room_load_indices": {"Living Room": 0}},
            },
            "rh": SimpleNamespace(
                df_final=pd.DataFrame(
                    {
                        "binary_sensor.lr_window": [1.0],  # open - must be ignored here
                        "binary_sensor.lr_door": [0.0],  # closed
                    }
                )
            ),
        }

        result = _build_room_door_open(input_data_dict, logger)

        self.assertEqual(result, [False])

    async def test_build_room_opening_open_no_rooms_returns_none(self):
        input_data_dict = {
            "params": {
                "optim_conf": {"number_of_deferrable_loads": 1},
                "retrieve_hass_conf": {},
                "passed_data": {},
            },
            "rh": None,
        }
        self.assertIsNone(_build_room_opening_open(input_data_dict, logger))
        self.assertIsNone(_build_room_door_open(input_data_dict, logger))

    def _kalman_input_data_dict(
        self,
        room_temp: float = 20.0,
        duty_sensor_value: float | None = 0.0,
        power_sensor_value: float | None = None,
        with_self_learning: bool = False,
        df_final: pd.DataFrame | None = None,
    ) -> dict:
        """Minimal, hand-built input_data_dict for _build_room_kalman_opening_open,
        mirroring the direct-dict style already used for
        _build_room_blind_positions's own tests - one physics-family room
        ("Living Room", load index 0), simple/degree-day family (no
        u_value/envelope_area/ventilation_rate/heated_volume) unless
        with_self_learning is set, in which case its thermal_battery dict
        carries a self_learning_dispatch key (routing only - the fitted
        model itself is mocked separately via load_pickle_blob in tests
        that need it)."""
        from types import SimpleNamespace

        hc: dict = {
            "start_temperature": room_temp,
            "supply_temperature": 35.0,
            "volume": 15.0,
        }
        if with_self_learning:
            hc["self_learning_dispatch"] = {
                "feature_names": ["bias"],
                "theta": [20.0],
                "neighbor_indices": {},
            }
        data = {"sensor.room_temp": [room_temp]}
        if duty_sensor_value is not None:
            data["sensor.hp_duty"] = [duty_sensor_value]
        if power_sensor_value is not None:
            data["sensor.hp_power"] = [power_sensor_value]
        if df_final is None:
            df_final = pd.DataFrame(data)
        return {
            "params": {
                "optim_conf": {
                    "number_of_deferrable_loads": 1,
                    "heatpump_room_names": ["Living Room"],
                    "nominal_power_of_deferrable_loads": [1500.0],
                    "def_load_config": [{"thermal_battery": hc}],
                },
                "retrieve_hass_conf": {
                    "heatpump_duty_sensor": "sensor.hp_duty",
                    "heatpump_power_sensor": "sensor.hp_power",
                    "heatpump_room_temp_sensors": ["sensor.room_temp"],
                },
                "plant_conf": {"heatpump_nominal_power": 3000.0},
                "passed_data": {"room_load_indices": {"Living Room": 0}},
            },
            "rh": SimpleNamespace(df_final=df_final),
            "emhass_conf": {},
        }

    def _kalman_df_input_data_dayahead(self, outdoor_temp: float = 5.0) -> pd.DataFrame:
        return pd.DataFrame({"outdoor_temperature_forecast": [outdoor_temp]})

    async def test_build_room_kalman_opening_open_no_df_final_is_noop(self):
        from types import SimpleNamespace

        input_data_dict = self._kalman_input_data_dict()
        input_data_dict["rh"] = SimpleNamespace(df_final=None)
        df_dayahead = self._kalman_df_input_data_dayahead()

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock()) as mock_load,
            patch("emhass.command_line.save_json_blob", AsyncMock()) as mock_save,
        ):
            result = await _build_room_kalman_opening_open(input_data_dict, logger, df_dayahead)

        self.assertEqual(result, [False])
        mock_load.assert_not_called()
        mock_save.assert_not_called()

    async def test_build_room_kalman_opening_open_missing_power_and_duty_is_hard_noop(self):
        """Neither heatpump_power_sensor nor heatpump_duty_sensor resolves -
        must be a hard no-op (all False), NOT a zero-fallback, and must
        never attempt to load/save persisted state."""
        input_data_dict = self._kalman_input_data_dict(
            duty_sensor_value=None, power_sensor_value=None
        )
        df_dayahead = self._kalman_df_input_data_dayahead()

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock()) as mock_load,
            patch("emhass.command_line.save_json_blob", AsyncMock()) as mock_save,
        ):
            result = await _build_room_kalman_opening_open(input_data_dict, logger, df_dayahead)

        self.assertEqual(result, [False])
        mock_load.assert_not_called()
        mock_save.assert_not_called()

    async def test_build_room_kalman_opening_open_cold_start_never_flags_but_persists(self):
        input_data_dict = self._kalman_input_data_dict(room_temp=20.0, duty_sensor_value=0.0)
        df_dayahead = self._kalman_df_input_data_dayahead()

        with (
            patch(
                "emhass.command_line.load_json_blob", AsyncMock(return_value={})
            ) as mock_load,
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await _build_room_kalman_opening_open(input_data_dict, logger, df_dayahead)

        self.assertEqual(result, [False], "A room's first-ever cycle must never be flagged open")
        mock_load.assert_awaited_once()
        mock_save.assert_awaited_once()
        saved_state = mock_save.call_args.args[2]
        self.assertIn("Living Room", saved_state["rooms"])
        self.assertAlmostEqual(saved_state["rooms"]["Living Room"]["x"], 20.0)

    async def test_build_room_kalman_opening_open_sustained_gap_flags_open_on_second_cycle(self):
        """Cycle 1 cold-starts at 20.0 (never flagged). Cycle 2, moments
        later, the SAME room's live reading has dropped 5C (a live window
        opening) - with duty=0 and a near-zero elapsed dt, the physics
        predictor's own one-step prediction stays close to 20.0, so this
        large a gap must cross the gate and flag is_open=True."""
        input_data_dict = self._kalman_input_data_dict(room_temp=20.0, duty_sensor_value=0.0)
        df_dayahead = self._kalman_df_input_data_dayahead()

        with (
            patch(
                "emhass.command_line.load_json_blob", AsyncMock(return_value={})
            ),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result_1 = await _build_room_kalman_opening_open(input_data_dict, logger, df_dayahead)
        self.assertEqual(result_1, [False])
        persisted_state = mock_save.call_args.args[2]

        input_data_dict_2 = self._kalman_input_data_dict(room_temp=15.0, duty_sensor_value=0.0)
        with (
            patch(
                "emhass.command_line.load_json_blob",
                AsyncMock(return_value=persisted_state),
            ),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result_2 = await _build_room_kalman_opening_open(
                input_data_dict_2, logger, df_dayahead
            )

        self.assertEqual(result_2, [True], "A sudden sustained 5C gap must flag the room open")

    async def test_build_room_kalman_opening_open_self_learning_routes_to_pickled_model(self):
        """A room whose def_load_config carries self_learning_dispatch must
        route to the self-learning branch - verified by asserting
        load_pickle_blob (only reached by that branch) actually gets
        called, unlike the physics-family branch which never touches it."""
        input_data_dict = self._kalman_input_data_dict(
            room_temp=20.0, duty_sensor_value=0.3, with_self_learning=True
        )
        df_dayahead = self._kalman_df_input_data_dayahead()

        class _FakeModel:
            room_models_ = {"Living Room": object()}

            def predict_recursive(self, df_house_fc, dfs_by_room_fc, initial_room_states):
                n = len(df_house_fc)
                return {
                    "room_temp": {"Living Room": np.full(n, 20.0)},
                    "electric_power": np.zeros(n),
                    "gas_consumption": None,
                }

        # First cycle: cold start (no prior state yet) - load_pickle_blob
        # should NOT be reached at all this cycle (cold start returns before
        # any model is needed).
        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value={})),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
            patch("emhass.command_line.load_pickle_blob", AsyncMock()) as mock_load_pickle,
        ):
            await _build_room_kalman_opening_open(input_data_dict, logger, df_dayahead)
        mock_load_pickle.assert_not_called()
        persisted_state = mock_save.call_args.args[2]

        # Second cycle: a real prior state exists now - the self-learning
        # branch should be reached and load_pickle_blob called.
        input_data_dict_2 = self._kalman_input_data_dict(
            room_temp=20.0, duty_sensor_value=0.3, with_self_learning=True
        )
        with (
            patch(
                "emhass.command_line.load_json_blob",
                AsyncMock(return_value=persisted_state),
            ),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
            patch(
                "emhass.command_line.load_pickle_blob", AsyncMock(return_value=_FakeModel())
            ) as mock_load_pickle_2,
        ):
            result_2 = await _build_room_kalman_opening_open(
                input_data_dict_2, logger, df_dayahead
            )
        mock_load_pickle_2.assert_awaited()
        self.assertEqual(result_2, [False])  # matching prediction, no anomaly

    async def test_build_room_opening_open_with_kalman_fallback_or_merges(self):
        """sensor-open OR kalman-open -> open; both closed -> closed; a room
        with no window/door sensor configured at all but Kalman flags it ->
        still open (validates the 'always runs' design)."""
        input_data_dict = self._kalman_input_data_dict(room_temp=20.0, duty_sensor_value=0.0)
        df_dayahead = self._kalman_df_input_data_dayahead()

        with patch(
            "emhass.command_line._build_room_kalman_opening_open",
            AsyncMock(return_value=[True]),
        ):
            result = await _build_room_opening_open_with_kalman_fallback(
                input_data_dict, logger, df_dayahead
            )
        self.assertEqual(result, [True], "Kalman-open alone (no sensor configured) must win")

        with patch(
            "emhass.command_line._build_room_kalman_opening_open",
            AsyncMock(return_value=[False]),
        ):
            result = await _build_room_opening_open_with_kalman_fallback(
                input_data_dict, logger, df_dayahead
            )
        self.assertEqual(result, [False])

    async def test_next_deadline_timestamp(self):
        horizon_start = pd.Timestamp("2026-01-01T10:00:00", tz="UTC")
        # Deadline later today.
        deadline = _next_deadline_timestamp("22:00", horizon_start)
        self.assertEqual(deadline, pd.Timestamp("2026-01-01T22:00:00", tz="UTC"))
        # Deadline already passed today -> rolls to tomorrow.
        deadline2 = _next_deadline_timestamp("06:00", horizon_start)
        self.assertEqual(deadline2, pd.Timestamp("2026-01-02T06:00:00", tz="UTC"))
        # Unparseable/empty -> None.
        self.assertIsNone(_next_deadline_timestamp("", horizon_start))
        self.assertIsNone(_next_deadline_timestamp("not-a-time", horizon_start))

    async def test_format_manual_load_action(self):
        now = pd.Timestamp("2026-01-01T10:00:00", tz="UTC")
        self.assertEqual(_format_manual_load_action(None, now), "waiting")
        self.assertEqual(
            _format_manual_load_action(now - pd.Timedelta(minutes=1), now), "Start now"
        )
        self.assertEqual(
            _format_manual_load_action(now + pd.Timedelta(hours=1, minutes=30), now),
            "Set timer to 1h 30m",
        )
        self.assertEqual(
            _format_manual_load_action(now + pd.Timedelta(minutes=20), now), "Set timer to 20m"
        )

    async def test_maybe_record_manual_load_commitments_creates_and_never_overwrites(self):
        input_data_dict = await self._build_manual_load_input_data_dict(ready=True)
        params = input_data_dict["params"]
        k = params["passed_data"]["manual_load_indices"]["Dishwasher"]["k"]
        idx = pd.date_range(start=pd.Timestamp.now(tz="UTC"), periods=4, freq="30min")
        opt_res = pd.DataFrame({f"P_deferrable{k}": [0.0, 1800.0, 1800.0, 0.0]}, index=idx)
        ctx = PublishContext(
            input_data_dict=input_data_dict,
            params=params,
            idx=0,
            common_kwargs={},
            logger=logger,
        )

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value={})),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            commitments = await _maybe_record_manual_load_commitments(ctx, opt_res)

        self.assertIn("Dishwasher", commitments)
        self.assertEqual(
            commitments["Dishwasher"]["committed_start_iso"], idx[1].isoformat()
        )
        mock_save.assert_awaited_once()

        # A second call with an already-persisted commitment must not move it,
        # even if the solved plan this time suggests a different start.
        existing = {"Dishwasher": {"committed_start_iso": idx[1].isoformat()}}
        opt_res_2 = pd.DataFrame({f"P_deferrable{k}": [1800.0, 1800.0, 0.0, 0.0]}, index=idx)
        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=existing)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save_2,
        ):
            commitments_2 = await _maybe_record_manual_load_commitments(ctx, opt_res_2)

        self.assertEqual(
            commitments_2["Dishwasher"]["committed_start_iso"], idx[1].isoformat()
        )
        mock_save_2.assert_not_awaited()

    async def test_publish_manual_load_actions(self):
        input_data_dict = await self._build_manual_load_input_data_dict(ready=True)
        params = input_data_dict["params"]
        input_data_dict["rh"].post_data = AsyncMock(return_value=True)
        ctx = PublishContext(
            input_data_dict=input_data_dict,
            params=params,
            idx=0,
            common_kwargs={},
            logger=logger,
        )
        now = pd.Timestamp.now(tz="UTC")
        commitments = {
            "Dishwasher": {
                "committed_start_iso": (now + pd.Timedelta(hours=1)).isoformat()
            }
        }

        await _publish_manual_load_actions(ctx, commitments)

        input_data_dict["rh"].post_data.assert_awaited_once()
        call_args, call_kwargs = input_data_dict["rh"].post_data.call_args
        self.assertEqual(call_args[2], "sensor.manual_load_action_dishwasher")
        self.assertEqual(call_kwargs.get("type_var"), "categorical")
        published_series = call_args[0]
        self.assertEqual(published_series.iloc[0], "Set timer to 1h 0m")

    async def test_regressor_preparation_errors(self):
        """
        Test logger error paths in _prepare_regressor_fit (missing CSV, missing columns).
        """
        # Case 1: No csv_file in params
        # Use get_test_params to ensure proper structure
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["passed_data"] = {}
        params_json = orjson.dumps(params).decode("utf-8")
        # We use set_input_data_dict which calls _prepare_regressor_fit
        # This should return False (failed setup) because csv_file is missing
        res = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            None,
            "regressor-model-fit",
            logger,
            get_data_from_file=True,
        )
        self.assertFalse(res, "Should fail when csv_file is missing")
        # Case 2: CSV file missing on disk
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["passed_data"] = {"csv_file": "missing.csv"}
        params_json = orjson.dumps(params).decode("utf-8")
        with patch("pathlib.Path.is_file", return_value=False):
            res = await set_input_data_dict(
                emhass_conf,
                "profit",
                params_json,
                None,
                "regressor-model-fit",
                logger,
                get_data_from_file=True,
            )
            self.assertFalse(res, "Should fail when file does not exist")
        # Case 3: CSV exists but missing required columns
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["passed_data"] = {
            "csv_file": "exists.csv",
            "features": ["required_col"],
            "target": "target_col",
        }
        params_json = orjson.dumps(params).decode("utf-8")
        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pandas.read_csv", return_value=pd.DataFrame({"wrong_col": [1]})),
        ):
            res = await set_input_data_dict(
                emhass_conf,
                "profit",
                params_json,
                None,
                "regressor-model-fit",
                logger,
                get_data_from_file=True,
            )
            self.assertFalse(res, "Should fail when columns are missing")

    async def test_prepare_forecast_and_weather_data(self):
        """
        Test the standalone prepare_forecast_and_weather_data helper method.
        Covers the padding, slicing, timezone mismatches, and GHI resolution warnings.
        """
        # Setup base DataFrames
        dayahead_idx = pd.date_range("2025-01-01", periods=10, freq="30min", tz="UTC")
        df_input_data_dayahead = pd.DataFrame({"P_PV": [0.0] * 10}, index=dayahead_idx)
        # Setup input_data_dict
        input_data_dict = {
            "fcst": MagicMock(),
            "df_input_data_dayahead": df_input_data_dayahead,
            "params": {"passed_data": {}},
            "df_weather": None,
            "retrieve_hass_conf": {"Latitude": 45.83, "Longitude": 6.86},
        }
        # Mock the forecast methods to just return the passed DataFrame
        input_data_dict["fcst"].get_load_cost_forecast.return_value = df_input_data_dayahead.copy()
        input_data_dict["fcst"].get_prod_price_forecast.return_value = df_input_data_dayahead.copy()
        # Test 1: Passed outdoor temp is longer than horizon (Slice Test)
        input_data_dict["params"]["passed_data"]["outdoor_temperature_forecast"] = [
            20.0
        ] * 15  # 15 passed > 10 horizon
        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)
        self.assertIsInstance(res_df, pd.DataFrame)
        self.assertEqual(
            len(res_df["outdoor_temperature_forecast"]), 10, "Should have sliced to exactly 10"
        )
        # Test 2: Passed outdoor temp is shorter than horizon (Pad Test)
        # 5 passed < 10 horizon, last value is 25.0
        input_data_dict["params"]["passed_data"]["outdoor_temperature_forecast"] = [
            18.0,
            19.0,
            20.0,
            21.0,
            25.0,
        ]
        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)
        self.assertEqual(
            len(res_df["outdoor_temperature_forecast"]), 10, "Should have padded to exactly 10"
        )
        self.assertEqual(
            res_df["outdoor_temperature_forecast"].iloc[-1],
            25.0,
            "Padded values should match the last passed value",
        )
        # Test 3: Fallback to df_weather with Timezone conversion and GHI
        input_data_dict["params"]["passed_data"] = {}  # Remove passed outdoor temp
        # Weather index is timezone naive, dayahead is UTC
        weather_idx = pd.date_range("2025-01-01", periods=5, freq="2h")
        df_weather = pd.DataFrame(
            {"temp_air": [10.0, 11.0, 12.0, 13.0, 14.0], "ghi": [100, 200, 300, 400, 500]},
            index=weather_idx,
        )
        input_data_dict["df_weather"] = df_weather
        # We also want to test the resolution warning (warn_on_resolution=True)
        # dayahead is 30m, weather is 1h -> weather_freq > 2 * dayahead_freq will trigger the warning
        with self.assertLogs(logger, level="WARNING") as cm:
            res_df = prepare_forecast_and_weather_data(
                input_data_dict, logger, warn_on_resolution=True
            )
        # Verify timezone conversion worked and NaNs were safely filled
        self.assertIsInstance(res_df, pd.DataFrame)
        self.assertIn("temp_air", df_weather.columns)
        self.assertIn("outdoor_temperature_forecast", res_df.columns)
        self.assertIn("ghi", res_df.columns)
        self.assertEqual(
            res_df["outdoor_temperature_forecast"].isnull().sum(),
            0,
            "Forward/Backward fill should have caught all NaNs",
        )
        self.assertEqual(res_df["ghi"].isnull().sum(), 0)
        # Verify the resolution warning was actually triggered
        warning_logs = str(cm.output)
        self.assertTrue(
            "much coarser than dayahead" in warning_logs, "Resolution warning should have triggered"
        )
        # Test 4: Timezone mismatch (Dayahead Naive, Weather Aware)
        # Make dayahead naive
        input_data_dict["df_input_data_dayahead"].index = input_data_dict[
            "df_input_data_dayahead"
        ].index.tz_localize(None)
        input_data_dict["fcst"].get_load_cost_forecast.return_value = input_data_dict[
            "df_input_data_dayahead"
        ].copy()
        input_data_dict["fcst"].get_prod_price_forecast.return_value = input_data_dict[
            "df_input_data_dayahead"
        ].copy()
        # Make weather aware
        df_weather.index = df_weather.index.tz_localize("Europe/Paris")
        input_data_dict["df_weather"] = df_weather
        # Execution shouldn't crash
        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)
        self.assertIsInstance(res_df, pd.DataFrame)
        # Result index should remain naive
        self.assertIsNone(res_df.index.tz)

    async def test_prepare_forecast_and_weather_data_merges_wind_dni_dhi(self):
        """wind_speed/dni/dhi (needed by the self-learning-physics dispatch
        equation, see optimization.py::_add_self_learning_dispatch_constraints)
        must reach data_opt the same way ghi already does - previously these
        three never reached data_opt at all (_merge_weather_column's own
        docstring notes this as the gap this refactor closed)."""
        dayahead_idx = pd.date_range("2025-01-01", periods=5, freq="30min", tz="UTC")
        df_input_data_dayahead = pd.DataFrame({"P_PV": [0.0] * 5}, index=dayahead_idx)
        weather_idx = pd.date_range("2025-01-01", periods=5, freq="30min", tz="UTC")
        df_weather = pd.DataFrame(
            {
                "temp_air": [10.0] * 5,
                "ghi": [100.0] * 5,
                "wind_speed": [3.0, 3.5, 4.0, 4.5, 5.0],
                "dni": [50.0, 60.0, 70.0, 80.0, 90.0],
                "dhi": [10.0, 12.0, 14.0, 16.0, 18.0],
            },
            index=weather_idx,
        )
        input_data_dict = {
            "fcst": MagicMock(),
            "df_input_data_dayahead": df_input_data_dayahead,
            "params": {"passed_data": {}},
            "df_weather": df_weather,
            "retrieve_hass_conf": {"Latitude": 45.83, "Longitude": 6.86},
        }
        input_data_dict["fcst"].get_load_cost_forecast.return_value = df_input_data_dayahead.copy()
        input_data_dict["fcst"].get_prod_price_forecast.return_value = df_input_data_dayahead.copy()

        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)

        for column, expected in (
            ("wind_speed", [3.0, 3.5, 4.0, 4.5, 5.0]),
            ("dni", [50.0, 60.0, 70.0, 80.0, 90.0]),
            ("dhi", [10.0, 12.0, 14.0, 16.0, 18.0]),
        ):
            self.assertIn(column, res_df.columns)
            self.assertEqual(res_df[column].isnull().sum(), 0)
            np.testing.assert_allclose(res_df[column].to_numpy(), expected)

    async def test_prepare_forecast_and_weather_data_missing_wind_dni_dhi_columns_are_skipped(self):
        """When df_weather doesn't have wind_speed/dni/dhi at all (e.g. a
        weather source that only ever provided ghi), _merge_weather_column
        must silently no-op for those columns rather than crash."""
        dayahead_idx = pd.date_range("2025-01-01", periods=5, freq="30min", tz="UTC")
        df_input_data_dayahead = pd.DataFrame({"P_PV": [0.0] * 5}, index=dayahead_idx)
        df_weather = pd.DataFrame(
            {"temp_air": [10.0] * 5, "ghi": [100.0] * 5}, index=dayahead_idx
        )
        input_data_dict = {
            "fcst": MagicMock(),
            "df_input_data_dayahead": df_input_data_dayahead,
            "params": {"passed_data": {}},
            "df_weather": df_weather,
            "retrieve_hass_conf": {"Latitude": 45.83, "Longitude": 6.86},
        }
        input_data_dict["fcst"].get_load_cost_forecast.return_value = df_input_data_dayahead.copy()
        input_data_dict["fcst"].get_prod_price_forecast.return_value = df_input_data_dayahead.copy()

        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)

        self.assertIn("ghi", res_df.columns)
        self.assertNotIn("wind_speed", res_df.columns)
        self.assertNotIn("dni", res_df.columns)
        self.assertNotIn("dhi", res_df.columns)

    async def test_prepare_forecast_and_weather_data_adds_solar_elevation(self):
        """solar_elevation (needed by the physics-family awning-type
        blind-shading formula, see utils.calculate_shaded_window_irradiance)
        is computed directly from timestamps/location via
        Forecast.compute_solar_angles, not fetched from a weather API - it
        should be present and vary between a midday and a midnight
        timestamp for a fixed location."""
        dayahead_idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2025-06-21T12:00:00", tz="UTC"),
                pd.Timestamp("2025-06-21T00:00:00", tz="UTC"),
            ]
        )
        df_input_data_dayahead = pd.DataFrame({"P_PV": [0.0, 0.0]}, index=dayahead_idx)
        input_data_dict = {
            "fcst": MagicMock(),
            "df_input_data_dayahead": df_input_data_dayahead,
            "params": {"passed_data": {}},
            "df_weather": None,
            # Grenoble, France - well north of the equator so a summer
            # midday sun sits clearly above the horizon and midnight clearly
            # below it.
            "retrieve_hass_conf": {"Latitude": 45.19, "Longitude": 5.73},
        }
        input_data_dict["fcst"].get_load_cost_forecast.return_value = df_input_data_dayahead.copy()
        input_data_dict["fcst"].get_prod_price_forecast.return_value = df_input_data_dayahead.copy()

        res_df = prepare_forecast_and_weather_data(input_data_dict, logger)

        self.assertIn("solar_elevation", res_df.columns)
        self.assertGreater(res_df["solar_elevation"].iloc[0], 30.0)  # midday, well above horizon
        self.assertLess(res_df["solar_elevation"].iloc[1], 0.0)  # midnight, below horizon

    async def test_weather_forecast_methods(self):
        """
        Test logic in _get_dayahead_pv_forecast regarding weather method switching.
        """
        # Test Method = List (should skip normal weather forecast fetch)
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["set_use_pv"] = True
        params["optim_conf"]["delta_forecast_daily"] = pd.Timedelta(
            days=params["optim_conf"]["delta_forecast_daily"]
        )
        mock_fcst = Mock()
        mock_fcst.forecast_dates = pd.date_range("2024-01-01", periods=1)
        mock_fcst.get_weather_forecast = AsyncMock(return_value=pd.DataFrame())
        mock_fcst.get_power_from_weather = Mock(return_value=pd.Series([0]))
        mock_fcst.get_load_forecast = AsyncMock(return_value=pd.Series([0]))
        # Create SetupContext manually to bypass set_input_data_dict complexity
        ctx = SetupContext(
            retrieve_hass_conf=params["retrieve_hass_conf"],
            optim_conf=params["optim_conf"],
            plant_conf={},
            emhass_conf=emhass_conf,
            params=params,
            logger=logger,
            get_data_from_file=False,
            rh=Mock(),
            fcst=mock_fcst,
        )
        await _prepare_dayahead_optim(ctx)
        # get_weather_forecast should be called with method='list'
        mock_fcst.get_weather_forecast.assert_called_with(method="list")
        # Test Method != List (e.g. scrapper), ensuring it returns None if weather fails
        ctx.optim_conf["weather_forecast_method"] = "scrapper"
        mock_fcst.get_weather_forecast = AsyncMock(return_value=False)  # Simulate failure
        res = await _prepare_dayahead_optim(ctx)
        self.assertIsNone(res, "Should return None if weather forecast fails")

    async def test_thermal_config_runtime_overrides(self):
        """
        Test that thermal config parameters (def_load_config and heater overrides)
        are correctly processed for non-MPC actions (e.g. dayahead-optim).
        """
        costfun = "profit"
        action = "dayahead-optim"
        params = await TestCommandLineAsyncUtils.get_test_params()
        # Base thermal config passed in runtime (simulating what the add-on does)
        runtime_def_load_config = [
            {
                "thermal_config": {
                    "model_type": "thermal_battery",
                    "start_temperature": 20.0,
                    "desired_temperatures": [21.0] * 48,
                }
            }
        ]
        # Overrides passed in runtime
        runtimeparams = {
            "def_load_config": runtime_def_load_config,
            "heater_start_temperatures": [25.5],
            "heater_desired_temperatures": [[22.5] * 48],
            # Required forecasts to pass validation
            "pv_power_forecast": [1] * 48,
            "load_power_forecast": [1] * 48,
            "load_cost_forecast": [1] * 48,
            "prod_price_forecast": [1] * 48,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params_json = orjson.dumps(params).decode("utf-8")
        # Execute
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Assertions
        optim_conf = input_data_dict["params"]["optim_conf"]
        # Verify def_load_config was copied from runtimeparams (previously ignored in dayahead)
        self.assertIn("def_load_config", optim_conf)
        self.assertEqual(len(optim_conf["def_load_config"]), 1)
        thermal_config = optim_conf["def_load_config"][0]["thermal_config"]
        # Verify start_temperature override applied (20.0 -> 25.5)
        self.assertEqual(thermal_config["start_temperature"], 25.5)
        # Verify desired_temperatures override applied (21.0 -> 22.5)
        self.assertEqual(thermal_config["desired_temperatures"], [22.5] * 48)

    @patch("emhass.command_line.publish_json", new_callable=AsyncMock)
    @patch("emhass.command_line.aiofiles.open")
    @patch("os.path.isfile")
    @patch("os.listdir")
    @patch("os.path.exists")
    async def test_publish_and_update_freq(
        self, mock_exists, mock_listdir, mock_isfile, mock_aio_open, mock_publish_json
    ):
        """Test the background loop helper that checks for cached entities and updates frequency."""
        input_data_dict = {}
        entity_path = pathlib.Path("/mock/entities")
        logger = MagicMock()
        current_freq = pd.Timedelta(minutes=30)
        # Test 1: Directory does not exist
        mock_exists.return_value = False
        res = await _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq)
        self.assertEqual(res, current_freq)
        # Test 2: Directory is empty
        mock_exists.return_value = True
        mock_listdir.return_value = []
        res = await _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq)
        self.assertEqual(res, current_freq)
        # Test 3: Directory has files, metadata exists, new frequency returned.
        # The listing also contains an in-flight atomic-write temp file, which
        # must be skipped (publishing it would derive a bogus entity_id and
        # KeyError in publish_json).
        mock_listdir.return_value = [
            "sensor1.json",
            "metadata.json",
            "sensor1.json.1234.deadbeef.tmp",
        ]
        mock_isfile.return_value = True
        # Mock reading the metadata.json file
        mock_file_handle = AsyncMock()
        mock_file_handle.read.return_value = orjson.dumps({"lowest_time_step": 15})
        mock_aio_open.return_value.__aenter__.return_value = mock_file_handle
        res = await _publish_and_update_freq(input_data_dict, entity_path, logger, current_freq)
        # publish_json should only be called for "sensor1.json", NOT
        # "metadata.json" nor the ".tmp" temp file.
        mock_publish_json.assert_called_once_with(
            "sensor1.json", input_data_dict, entity_path, logger, "continual_publish"
        )
        # Expected new frequency based on the mocked lowest_time_step
        self.assertEqual(res, pd.Timedelta(minutes=15))

    @patch("emhass.command_line.pd.read_json")
    @patch("emhass.command_line.aiofiles.open")
    @patch("os.path.isfile")
    @patch("emhass.command_line.datetime")
    async def test_publish_json(self, mock_datetime, mock_isfile, mock_aio_open, mock_read_json):
        """Test the individual JSON file extraction and posting mechanism."""
        entity_path = pathlib.Path("/mock/entities")
        entity_file = "sensor_test.json"
        entity_id = "sensor_test"
        mock_rh = AsyncMock()
        input_data_dict = {
            "retrieve_hass_conf": {
                "time_zone": "UTC",
                "method_ts_round": "nearest",  # We will change this to test all branches
            },
            "rh": mock_rh,
        }
        # Test 1: Metadata file is missing
        mock_isfile.return_value = False
        res = await publish_json(entity_file, input_data_dict, entity_path, logger)
        self.assertFalse(res)
        # Test 2: Successful publish with 'nearest' rounding
        mock_isfile.return_value = True
        # Setup mock metadata JSON payload
        mock_metadata = {
            entity_id: {
                "name": "Test_Sensor",
                "optimization_time_step": 30,
                "unit_of_measurement": "W",
                "friendly_name": "Testing Sensor",
                "device_class": "power",
                "type_var": "custom",
            }
        }
        mock_file_handle = AsyncMock()
        mock_file_handle.read.return_value = orjson.dumps(mock_metadata)
        mock_aio_open.return_value.__aenter__.return_value = mock_file_handle
        # Setup mocked Pandas DataFrame representing the saved sensor data
        idx = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
        mock_df = pd.DataFrame({0: [100.0, 200.0, 300.0]}, index=idx)
        mock_read_json.return_value = mock_df
        # Mock datetime.now() to match the middle index (2024-01-01 00:30:00)
        mock_now = MagicMock()
        mock_now.replace.return_value = idx[1]
        mock_datetime.now.return_value = mock_now
        # Execute nearest
        res = await publish_json(
            entity_file, input_data_dict, entity_path, logger, "continual_publish"
        )
        # Verify formatting and post_data execution
        self.assertIsInstance(res, pd.Series)
        self.assertEqual(res.name, "Test_Sensor")
        mock_rh.post_data.assert_called_once()
        # Verify post_data arguments
        call_args = mock_rh.post_data.call_args[1]
        self.assertEqual(call_args["idx"], 1)  # Middle index matched
        self.assertEqual(call_args["entity_id"], entity_id)
        self.assertEqual(
            call_args["logger_levels"], "DEBUG"
        )  # Because reference='continual_publish'
        # Test 3: Coverage for 'first' rounding
        input_data_dict["retrieve_hass_conf"]["method_ts_round"] = "first"
        mock_rh.reset_mock()
        await publish_json(entity_file, input_data_dict, entity_path, logger)
        self.assertEqual(mock_rh.post_data.call_args[1]["idx"], 1)
        self.assertEqual(mock_rh.post_data.call_args[1]["logger_levels"], "INFO")  # Blank reference
        # Test 4: Coverage for 'last' rounding
        input_data_dict["retrieve_hass_conf"]["method_ts_round"] = "last"
        mock_rh.reset_mock()
        await publish_json(entity_file, input_data_dict, entity_path, logger)
        self.assertEqual(mock_rh.post_data.call_args[1]["idx"], 1)

    @patch("emhass.command_line.asyncio.sleep", new_callable=AsyncMock)
    @patch("emhass.command_line._publish_and_update_freq")
    async def test_continual_publish_survives_cycle_exception(self, mock_update, _mock_sleep):
        """A transient failure in a single publish cycle (e.g. issue #1000's
        empty-file read raising ValueError) must not kill the background task:
        the loop logs it and continues on the next interval. RED on base, where
        the exception propagates out of continual_publish and publishing stops
        until restart."""
        input_data_dict = {
            "retrieve_hass_conf": {
                "time_zone": UTC,
                "optimization_time_step": pd.Timedelta(minutes=5),
            }
        }
        entity_path = pathlib.Path("/mock/entities")
        logger = MagicMock()

        count = 0

        async def fake_update(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                # The exact failure reported in issue #1000.
                raise ValueError("Expected object or value")
            # Break out of the otherwise-infinite loop on the second iteration.
            # CancelledError is a BaseException, so a correct ``except Exception``
            # guard does not swallow it.
            raise asyncio.CancelledError

        mock_update.side_effect = fake_update

        with self.assertRaises(asyncio.CancelledError):
            await continual_publish(input_data_dict, entity_path, logger)

        # Reaching a second iteration proves the first exception was swallowed
        # and the loop kept running.
        self.assertEqual(count, 2)
        # The transient failure is logged (with traceback) exactly once, and the
        # message is pinned so a future refactor cannot silently drop it.
        logger.exception.assert_called_once()
        self.assertIn(
            "continual_publish cycle failed; retrying next interval",
            logger.exception.call_args[0][0],
        )


class TestEmRelabelOpeningOpen(unittest.TestCase):
    """Direct unit tests for _em_relabel_opening_open (Phase 2's EM-style
    fit -> smooth-residuals -> relabel -> refit loop, see command_line.py's
    own module docstring on it) - exercised standalone here, independent of
    the fuller refit_self_learning_physics_model integration (covered by
    test_refit_self_learning_physics_model_opening_relabel_disabled_by_default/
    ..._enabled_feeds_final_fit in TestCommandLineAsyncUtils above, which
    only check the wiring, not this function's own inference quality)."""

    def _build_frames(self, n: int = 500, seed: int = 0, hidden_event: tuple[int, int] | None = None):
        """Two rooms: 'Sensored' (has a real opening_open column and a
        configured window sensor - must NEVER be touched by relabeling) and
        'Unsensored' (no sensor at all - the only room _em_relabel_opening_open
        may touch). hidden_event, if given, is a (start_idx, end_idx) range
        where Unsensored's TRUE room_temp includes extra, unlabeled heat
        loss - simulating a real open window with no sensor to ever record
        it, the exact scenario this function exists to catch."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
        outdoor = 5.0 + 2.0 * np.sin(np.linspace(0, 12 * np.pi, n))
        duty = np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 8 * np.pi, n)), 0.0, 1.0)
        df_raw = pd.DataFrame(
            {
                "electric_power": 300.0 + 50.0 * duty,
                "heatpump_duty": duty,
                "group_duty": duty,
                "outdoor_temp": outdoor,
                "supply_temp": 35.0,
                "wind_speed": 1.0,
                "dni": 0.0,
                "dhi": 0.0,
                "sun_alt_sin": 0.0,
            },
            index=idx,
        )

        event_mask = None
        if hidden_event is not None:
            event_mask = np.zeros(n, dtype=bool)
            event_mask[hidden_event[0] : hidden_event[1]] = True

        def _simulate_room_temp(extra_loss_mask):
            # Mean-reverting toward a duty-driven target, matching the
            # duty/delta_env structure _physics_features itself fits on -
            # bounded and numerically stable, unlike a pure cumulative-loss
            # simulation. extra_loss_mask pulls temp further toward outdoor
            # on top of that, simulating a real open window's extra
            # ventilation loss with no sensor to ever record it.
            temp = np.zeros(n)
            temp[0] = 20.0
            for t in range(1, n):
                target = 18.0 + 4.0 * duty[t - 1]
                base = 0.10 * (target - temp[t - 1])
                extra = 0.0
                if extra_loss_mask is not None and extra_loss_mask[t - 1]:
                    extra = 0.20 * (outdoor[t - 1] - temp[t - 1])
                noise = rng.normal(0.0, 0.03)
                temp[t] = temp[t - 1] + base + extra + noise
            return temp

        df_sensored = df_raw.copy()
        df_sensored["room_temp"] = _simulate_room_temp(None)
        df_sensored["opening_open"] = 0.0  # real sensor: always closed here

        df_unsensored = df_raw.copy()
        df_unsensored["room_temp"] = _simulate_room_temp(event_mask)

        dfs_by_room = {"Sensored": df_sensored, "Unsensored": df_unsensored}
        neighbor_map = {"Sensored": [], "Unsensored": []}
        window_entity_map = {"Sensored": "binary_sensor.sensored_window"}
        door_entity_map: dict[str, str] = {}
        return df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, event_mask

    def test_never_touches_room_with_configured_sensor(self):
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames()
        )
        original_sensored_opening = dfs_by_room["Sensored"]["opening_open"].copy()

        blended, diagnostics = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=2, confirmed_overrides={}, logger=logger,
        )

        pd.testing.assert_series_equal(
            blended["Sensored"]["opening_open"], original_sensored_opening
        )
        self.assertNotIn("Sensored", diagnostics)
        self.assertIn("Unsensored", diagnostics)

    def test_no_eligible_rooms_is_a_noop(self):
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames()
        )
        # Give the ONLY otherwise-eligible room a configured door sensor too
        # - now every room has a real sensor, nothing left to relabel.
        door_entity_map = {"Unsensored": "binary_sensor.unsensored_door"}

        blended, diagnostics = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=2, confirmed_overrides={}, logger=logger,
        )

        self.assertEqual(diagnostics, {})
        self.assertNotIn("opening_open", blended["Unsensored"].columns)

    def test_confirmed_overrides_always_win(self):
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames()
        )
        ts_no = dfs_by_room["Unsensored"].index[10].isoformat()
        ts_yes = dfs_by_room["Unsensored"].index[20].isoformat()
        confirmed = {"Unsensored": {ts_no: 0.0, ts_yes: 1.0}}

        blended, _ = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=2, confirmed_overrides=confirmed, logger=logger,
        )

        opening = blended["Unsensored"]["opening_open"]
        self.assertEqual(opening.loc[pd.Timestamp(ts_no)], 0.0)
        self.assertEqual(opening.loc[pd.Timestamp(ts_yes)], 1.0)

    def test_exactly_one_plus_n_iterations_model_constructions(self):
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames()
        )
        from emhass.thermal.self_learning_physics import (
            SelfLearningPhysicsModel as RealSelfLearningPhysicsModel,
        )

        construction_count = {"n": 0}

        def _counting_ctor(*args, **kwargs):
            construction_count["n"] += 1
            return RealSelfLearningPhysicsModel(*args, **kwargs)

        with patch(
            "emhass.thermal.self_learning_physics.SelfLearningPhysicsModel",
            _counting_ctor,
        ):
            _em_relabel_opening_open(
                df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
                forgetting_factor=0.995, ridge=10.0, electric_only=True,
                n_iterations=3, confirmed_overrides={}, logger=logger,
            )

        # 1 baseline fit + one refit per relabeling iteration.
        self.assertEqual(construction_count["n"], 1 + 3)

    def test_diagnostics_shape_matches_room_history_length(self):
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames(n=300)
        )

        _, diagnostics = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=1, confirmed_overrides={}, logger=logger,
        )

        diag = diagnostics["Unsensored"]
        self.assertEqual(len(diag["is_open"]), 300)
        self.assertEqual(len(diag["innovation"]), 300)
        self.assertEqual(len(diag["s"]), 300)

    def test_hidden_opening_event_is_relabeled_and_strengthens_fitted_coefficient(self):
        """The scenario this whole feature exists for: an unsensored room
        with a hidden, real opening event baked into its true room_temp
        history (extra heat loss during a known window, never recorded by
        any sensor). After relabeling: (1) the inferred opening_open flag
        must be clearly more active during the true event window than
        outside it, and (2) a model fit on the relabeled data must learn a
        materially larger-magnitude opening_x_outdoor coefficient than a
        single-pass fit on the original, unlabeled data (whose
        opening_x_outdoor feature column is all-zero, so its own fitted
        coefficient for that term stays at its ridge-initialized ~0)."""
        event = (150, 220)
        df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map, _ = (
            self._build_frames(n=500, hidden_event=event)
        )

        from emhass.thermal.self_learning_physics import SelfLearningPhysicsModel

        baseline_model = SelfLearningPhysicsModel(
            forgetting_factor=0.995, ridge=10.0, electric_only=True
        )
        baseline_model.fit(
            df_raw, dfs_by_room, df_raw["electric_power"].to_numpy(), None, neighbor_map
        )
        baseline_room = baseline_model.room_models_["Unsensored"]
        baseline_coef = abs(
            baseline_room.theta_temp[baseline_room.feature_names.index("opening_x_outdoor")]
        )

        blended, diagnostics = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map, door_entity_map,
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=2, confirmed_overrides={}, logger=logger,
        )

        is_open = np.asarray(diagnostics["Unsensored"]["is_open"], dtype=bool)
        in_event_rate = is_open[event[0] : event[1]].mean()
        outside_event_rate = np.concatenate(
            [is_open[: event[0]], is_open[event[1] :]]
        ).mean()
        self.assertGreater(
            in_event_rate,
            outside_event_rate,
            "The hidden opening event should be flagged open noticeably more "
            "often than the rest of the (truly closed) history.",
        )

        relabeled_model = SelfLearningPhysicsModel(
            forgetting_factor=0.995, ridge=10.0, electric_only=True
        )
        relabeled_model.fit(
            df_raw, blended, df_raw["electric_power"].to_numpy(), None, neighbor_map
        )
        relabeled_room = relabeled_model.room_models_["Unsensored"]
        relabeled_coef = abs(
            relabeled_room.theta_temp[relabeled_room.feature_names.index("opening_x_outdoor")]
        )
        self.assertGreater(
            relabeled_coef,
            baseline_coef,
            "Fitting on the EM-relabeled data should learn a stronger "
            "opening_x_outdoor effect than a single-pass fit on unlabeled data.",
        )


class TestExtractContiguousOpenEvents(unittest.TestCase):
    """Direct unit tests for _extract_contiguous_open_events (Phase 3's
    candidate-event extraction, mirroring the already-shipped
    candidate-coupling suggestions pattern - informational only, never
    auto-applied). See the integration tests in TestCommandLineAsyncUtils
    (test_refit_self_learning_physics_model_surfaces_candidate_opening_events
    et al.) for how this feeds into refit_self_learning_physics_model's own
    result["candidate_openings"]."""

    def _diagnostics(self, is_open, innovation=None, s=None):
        n = len(is_open)
        return {
            "is_open": np.asarray(is_open, dtype=bool),
            "innovation": np.asarray(innovation if innovation is not None else [1.0] * n),
            "s": np.asarray(s if s is not None else [1.0] * n),
        }

    def test_no_open_steps_returns_empty_list(self):
        diagnostics = self._diagnostics([False, False, False])
        index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")

        events = _extract_contiguous_open_events(diagnostics, index)

        self.assertEqual(events, [])

    def test_single_contiguous_run_becomes_one_event(self):
        diagnostics = self._diagnostics([False, True, True, True, False])
        index = pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC")

        events = _extract_contiguous_open_events(diagnostics, index)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["n_steps"], 3)
        self.assertEqual(event["start_iso"], index[1].isoformat())
        self.assertEqual(event["end_iso"], index[3].isoformat())

    def test_two_separate_runs_become_two_events(self):
        diagnostics = self._diagnostics([True, False, True, True])
        index = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")

        events = _extract_contiguous_open_events(diagnostics, index)

        self.assertEqual(len(events), 2)
        self.assertEqual({e["n_steps"] for e in events}, {1, 2})

    def test_run_touching_the_end_of_history_is_still_captured(self):
        diagnostics = self._diagnostics([False, False, True])
        index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")

        events = _extract_contiguous_open_events(diagnostics, index)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["n_steps"], 1)

    def test_sorted_by_confidence_descending(self):
        # Two single-step events; the second has a much larger normalized
        # innovation (larger |innovation| for the same s) - should sort first.
        is_open = [True, False, True]
        innovation = [0.1, 0.0, 5.0]
        s = [1.0, 1.0, 1.0]
        diagnostics = self._diagnostics(is_open, innovation, s)
        index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")

        events = _extract_contiguous_open_events(diagnostics, index)

        self.assertEqual(len(events), 2)
        self.assertGreater(
            events[0]["mean_abs_normalized_innovation"],
            events[1]["mean_abs_normalized_innovation"],
        )
        self.assertEqual(events[0]["start_iso"], index[2].isoformat())


class TestSlugifyRoomName(unittest.TestCase):
    def test_lowercases_and_collapses_non_alphanumerics(self):
        self.assertEqual(_slugify_room_name("Living Room"), "living_room")
        self.assertEqual(_slugify_room_name("Kids' Room #2"), "kids_room_2")
        self.assertEqual(_slugify_room_name("  Attic  "), "attic")

    def test_empty_or_all_symbol_name_falls_back_to_room(self):
        self.assertEqual(_slugify_room_name(""), "room")
        self.assertEqual(_slugify_room_name("###"), "room")


class TestExpandConfirmedRangesToTimestamps(unittest.TestCase):
    def test_expands_range_into_one_entry_per_timestep(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC")
        df_room = pd.DataFrame({"room_temp": [20.0] * 5}, index=idx)
        confirmed_ranges = {
            "Bedroom": [
                {"start_iso": idx[1].isoformat(), "end_iso": idx[3].isoformat(), "value": 1.0}
            ]
        }

        expanded = _expand_confirmed_ranges_to_timestamps(confirmed_ranges, {"Bedroom": df_room})

        self.assertEqual(
            expanded["Bedroom"],
            {idx[1].isoformat(): 1.0, idx[2].isoformat(): 1.0, idx[3].isoformat(): 1.0},
        )

    def test_room_not_in_dfs_by_room_is_skipped(self):
        confirmed_ranges = {
            "Ghost": [
                {
                    "start_iso": "2026-01-01T00:00:00+00:00",
                    "end_iso": "2026-01-01T01:00:00+00:00",
                    "value": 1.0,
                }
            ]
        }

        expanded = _expand_confirmed_ranges_to_timestamps(confirmed_ranges, {})

        self.assertEqual(expanded, {})

    def test_malformed_range_entry_is_skipped_not_raised(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
        df_room = pd.DataFrame({"room_temp": [20.0] * 3}, index=idx)
        confirmed_ranges = {
            "Bedroom": [{"start_iso": "not-a-timestamp", "end_iso": "also-not", "value": 1.0}]
        }

        expanded = _expand_confirmed_ranges_to_timestamps(confirmed_ranges, {"Bedroom": df_room})

        self.assertEqual(expanded, {})


class TestResolveOpeningConfirmations(unittest.IsolatedAsyncioTestCase):
    """Direct unit tests for _resolve_opening_confirmations (Phase 4's poll/
    resolve half of the HA confirmation loop) - runs once per refit, never
    once per dispatch cycle (unlike _apply_manual_load_runtime_overrides's
    own live polling)."""

    def _confs(
        self,
        room_names=("Bedroom",),
        ready=("input_boolean.bedroom_ready",),
        answer=("input_boolean.bedroom_answer",),
    ):
        optim_conf = {"heatpump_room_names": list(room_names)}
        retrieve_hass_conf = {
            "heatpump_room_opening_confirm_ready_sensor": list(ready),
            "heatpump_room_opening_confirm_answer_sensor": list(answer),
        }
        return optim_conf, retrieve_hass_conf

    async def test_no_pending_entries_is_a_noop(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(return_value=None)
        blob = {"rooms": {"Bedroom": {"pending": None, "confirmed": []}}}

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(result, {})
        mock_save.assert_not_called()
        rh.get_current_state.assert_not_called()

    async def test_ready_and_answer_resolve_pending_to_confirmed_open(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(side_effect=[1.0, 1.0])  # ready, then answer
        blob = {
            "rooms": {
                "Bedroom": {
                    "pending": {
                        "start_iso": "2026-01-01T00:00:00+00:00",
                        "end_iso": "2026-01-01T01:00:00+00:00",
                    },
                    "confirmed": [],
                }
            }
        }

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(len(result["Bedroom"]), 1)
        self.assertEqual(result["Bedroom"][0]["value"], 1.0)
        mock_save.assert_awaited_once()
        saved_blob = mock_save.call_args[0][2]
        self.assertIsNone(saved_blob["rooms"]["Bedroom"]["pending"])

    async def test_ready_but_answer_false_resolves_to_closed(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(side_effect=[1.0, 0.0])
        blob = {
            "rooms": {
                "Bedroom": {
                    "pending": {
                        "start_iso": "2026-01-01T00:00:00+00:00",
                        "end_iso": "2026-01-01T01:00:00+00:00",
                    },
                    "confirmed": [],
                }
            }
        }

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(result["Bedroom"][0]["value"], 0.0)

    async def test_ready_sensor_unreadable_leaves_entry_pending(self):
        """A read failure must never silently drop the pending entry - it
        will simply be retried on the next refit."""
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(return_value=None)
        pending = {
            "start_iso": "2026-01-01T00:00:00+00:00",
            "end_iso": "2026-01-01T01:00:00+00:00",
        }
        blob = {"rooms": {"Bedroom": {"pending": dict(pending), "confirmed": []}}}

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(result, {})
        mock_save.assert_not_called()
        self.assertEqual(blob["rooms"]["Bedroom"]["pending"], pending)

    async def test_ready_true_but_answer_unreadable_leaves_entry_pending(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(side_effect=[1.0, None])
        pending = {
            "start_iso": "2026-01-01T00:00:00+00:00",
            "end_iso": "2026-01-01T01:00:00+00:00",
        }
        blob = {"rooms": {"Bedroom": {"pending": dict(pending), "confirmed": []}}}

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(result, {})
        mock_save.assert_not_called()

    async def test_accumulates_previously_confirmed_ranges_across_refits(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.get_current_state = AsyncMock(return_value=None)
        blob = {
            "rooms": {
                "Bedroom": {
                    "pending": None,
                    "confirmed": [
                        {
                            "start_iso": "2026-01-01T00:00:00+00:00",
                            "end_iso": "2026-01-01T01:00:00+00:00",
                            "value": 1.0,
                        }
                    ],
                }
            }
        }

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)),
        ):
            result = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )

        self.assertEqual(len(result["Bedroom"]), 1)
        self.assertEqual(result["Bedroom"][0]["value"], 1.0)


class TestPublishOpeningConfirmationQuestions(unittest.IsolatedAsyncioTestCase):
    """Direct unit tests for _publish_opening_confirmation_questions (Phase
    4's publish half) - a direct rh.post_data(...) call, never routed
    through PublishContext (this refit action never has an opt_res_latest)."""

    def _confs(
        self,
        room_names=("Bedroom",),
        ready=("input_boolean.bedroom_ready",),
        answer=("input_boolean.bedroom_answer",),
    ):
        optim_conf = {"heatpump_room_names": list(room_names)}
        retrieve_hass_conf = {
            "heatpump_room_opening_confirm_ready_sensor": list(ready),
            "heatpump_room_opening_confirm_answer_sensor": list(answer),
        }
        return optim_conf, retrieve_hass_conf

    def _candidate(self, room="Bedroom", start="2026-01-01T00:00:00+00:00", end="2026-01-01T01:00:00+00:00"):
        return {
            "room": room,
            "start_iso": start,
            "end_iso": end,
            "n_steps": 3,
            "mean_abs_normalized_innovation": 4.0,
        }

    async def test_publishes_question_for_fresh_candidate(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.post_data = AsyncMock()
        candidate_openings = [self._candidate()]

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value={"rooms": {}})),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _publish_opening_confirmation_questions(
                rh, {}, optim_conf, retrieve_hass_conf, candidate_openings, logger
            )

        rh.post_data.assert_awaited_once()
        call_args = rh.post_data.call_args
        self.assertEqual(call_args[0][2], "sensor.room_opening_confirmation_bedroom")
        self.assertEqual(call_args[1]["type_var"], "categorical")
        mock_save.assert_awaited_once()
        saved_blob = mock_save.call_args[0][2]
        self.assertIsNotNone(saved_blob["rooms"]["Bedroom"]["pending"])

    async def test_no_duplicate_publish_when_already_pending(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.post_data = AsyncMock()
        candidate_openings = [self._candidate()]
        blob = {
            "rooms": {
                "Bedroom": {
                    "pending": {
                        "start_iso": "2025-12-01T00:00:00+00:00",
                        "end_iso": "2025-12-01T01:00:00+00:00",
                        "question_ts_iso": "2025-12-01T00:00:00+00:00",
                    },
                    "confirmed": [],
                }
            }
        }

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _publish_opening_confirmation_questions(
                rh, {}, optim_conf, retrieve_hass_conf, candidate_openings, logger
            )

        rh.post_data.assert_not_called()
        mock_save.assert_not_called()

    async def test_no_republish_for_already_confirmed_event(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.post_data = AsyncMock()
        event = self._candidate()
        blob = {
            "rooms": {
                "Bedroom": {
                    "pending": None,
                    "confirmed": [
                        {
                            "start_iso": event["start_iso"],
                            "end_iso": event["end_iso"],
                            "value": 0.0,
                            "confirmed_ts_iso": "2025-12-01T00:00:00+00:00",
                        }
                    ],
                }
            }
        }

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value=blob)),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _publish_opening_confirmation_questions(
                rh, {}, optim_conf, retrieve_hass_conf, [event], logger
            )

        rh.post_data.assert_not_called()
        mock_save.assert_not_called()

    async def test_no_candidates_is_a_noop(self):
        optim_conf, retrieve_hass_conf = self._confs()
        rh = MagicMock()
        rh.post_data = AsyncMock()

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock()) as mock_load,
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _publish_opening_confirmation_questions(
                rh, {}, optim_conf, retrieve_hass_conf, [], logger
            )

        rh.post_data.assert_not_called()
        mock_save.assert_not_called()
        mock_load.assert_not_called()

    async def test_room_without_both_ready_and_answer_sensors_is_skipped(self):
        optim_conf, retrieve_hass_conf = self._confs(ready=("",))
        rh = MagicMock()
        rh.post_data = AsyncMock()
        candidate_openings = [self._candidate()]

        with (
            patch("emhass.command_line.load_json_blob", AsyncMock(return_value={"rooms": {}})),
            patch("emhass.command_line.save_json_blob", AsyncMock(return_value=True)) as mock_save,
        ):
            await _publish_opening_confirmation_questions(
                rh, {}, optim_conf, retrieve_hass_conf, candidate_openings, logger
            )

        rh.post_data.assert_not_called()
        mock_save.assert_not_called()


class TestOpeningConfirmationFeedsEmRelabel(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_no_is_never_re_flagged_by_em_relabel(self):
        """Combined Phase 2 + Phase 4 end-to-end: a permanently confirmed_no
        answer for a specific event window must flow
        _resolve_opening_confirmations -> _expand_confirmed_ranges_to_timestamps
        -> _em_relabel_opening_open's confirmed_overrides and win over
        whatever the EM loop's own inference would otherwise have flagged
        there."""
        idx = pd.date_range("2026-01-01", periods=200, freq="30min", tz="UTC")
        rng = np.random.default_rng(0)
        outdoor = 5.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, 200))
        duty = np.clip(0.5 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, 200)), 0.0, 1.0)
        df_raw = pd.DataFrame(
            {
                "electric_power": 300.0 + 50.0 * duty,
                "heatpump_duty": duty,
                "group_duty": duty,
                "outdoor_temp": outdoor,
                "supply_temp": 35.0,
                "wind_speed": 1.0,
                "dni": 0.0,
                "dhi": 0.0,
                "sun_alt_sin": 0.0,
            },
            index=idx,
        )
        # A real anomaly at steps 50-69 - exactly what the confirmed_no
        # answer below will insist did NOT happen.
        temp = np.zeros(200)
        temp[0] = 20.0
        for t in range(1, 200):
            target = 18.0 + 4.0 * duty[t - 1]
            base = 0.10 * (target - temp[t - 1])
            extra = 0.25 * (outdoor[t - 1] - temp[t - 1]) if 50 <= t - 1 < 70 else 0.0
            temp[t] = temp[t - 1] + base + extra + rng.normal(0.0, 0.03)
        df_room = df_raw.assign(room_temp=temp)
        dfs_by_room = {"Bedroom": df_room}
        neighbor_map = {"Bedroom": []}

        rh = MagicMock()
        rh.get_current_state = AsyncMock(return_value=None)
        confirmed_blob = {
            "rooms": {
                "Bedroom": {
                    "pending": None,
                    "confirmed": [
                        {
                            "start_iso": idx[50].isoformat(),
                            "end_iso": idx[69].isoformat(),
                            "value": 0.0,
                        }
                    ],
                }
            }
        }
        optim_conf = {"heatpump_room_names": ["Bedroom"]}
        retrieve_hass_conf = {
            "heatpump_room_opening_confirm_ready_sensor": [""],
            "heatpump_room_opening_confirm_answer_sensor": [""],
        }

        with patch("emhass.command_line.load_json_blob", AsyncMock(return_value=confirmed_blob)):
            confirmed_ranges = await _resolve_opening_confirmations(
                rh, {}, optim_conf, retrieve_hass_conf, logger
            )
        confirmed_overrides = _expand_confirmed_ranges_to_timestamps(confirmed_ranges, dfs_by_room)

        blended, _ = _em_relabel_opening_open(
            df_raw, dfs_by_room, neighbor_map, window_entity_map={}, door_entity_map={},
            forgetting_factor=0.995, ridge=10.0, electric_only=True,
            n_iterations=2, confirmed_overrides=confirmed_overrides, logger=logger,
        )

        opening = blended["Bedroom"]["opening_open"]
        confirmed_window = opening.loc[idx[50] : idx[69]]
        self.assertTrue(
            (confirmed_window == 0.0).all(),
            "A confirmed_no answer must force opening_open=0.0 for the whole "
            "confirmed window, even though the EM loop's own inference "
            "would otherwise have flagged much of it open.",
        )


class TestCommandLineTimezoneLogic(unittest.IsolatedAsyncioTestCase):
    """
    Test class to verify Timezone alignment in command_line.py.
    Uses real configuration loading to ensure all Forecast parameters are present.
    """

    @staticmethod
    async def get_test_params():
        params = {}
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults.json does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        # Load real parameters and configuration
        params = await self.get_test_params()
        self.params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = (
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        )
        self.emhass_conf = emhass_conf
        self.logger = logger

    def test_prepare_forecast_and_weather_data_with_open_meteo(self):
        """
        Test that Open-Meteo weather data is correctly aligned with the Optimization Index
        even when timestamps do not match perfectly (e.g. 15s drift).
        """
        # Initialize Forecast object using the REAL loaded configurations
        fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            json.loads(self.params_json),
            self.emhass_conf,
            self.logger,
        )

        # Simulate "Dayahead" Data (Optimization Window)
        # CRITICAL FIX: Must be Timezone-Aware to pass get_load_cost_forecast validations
        tz = self.retrieve_hass_conf["time_zone"]
        now_optim = pd.Timestamp.now(tz=tz).floor("30min")
        index_optim = pd.date_range(start=now_optim, periods=48, freq="30min")

        df_input_data_dayahead = pd.DataFrame(index=index_optim)
        df_input_data_dayahead["p_load_forecast"] = 1000
        df_input_data_dayahead["p_pv_forecast"] = 0

        # Simulate "Open-Meteo" Weather Data
        # We shift it by 15 seconds to simulate the misalignment that causes NaNs
        # without the new robust reindexing logic.
        now_weather = now_optim + pd.Timedelta(seconds=15)
        index_weather = pd.date_range(start=now_weather, periods=48, freq="30min")

        df_weather = pd.DataFrame(index=index_weather)
        # Fill with a recognizable pattern (20.0, 20.5, 21.0...)
        df_weather["temp_air"] = [20 + (i * 0.5) for i in range(48)]
        # GHI is required by the function logic
        df_weather["ghi"] = 0

        # Construct Input Dictionary
        input_data_dict = {
            "fcst": fcst,
            "df_input_data_dayahead": df_input_data_dayahead,
            "df_weather": df_weather,
            "params": {
                "passed_data": {}
            },  # No explicit outdoor_temp passed, forcing fallback logic
            "retrieve_hass_conf": self.retrieve_hass_conf,
        }

        # Execute the function under test
        df_result = prepare_forecast_and_weather_data(
            input_data_dict, self.logger, warn_on_resolution=False
        )

        # 6. Assertions
        # A. Function should not return False
        self.assertFalse(isinstance(df_result, bool) and not df_result)

        # B. Check column existence
        self.assertIn("outdoor_temperature_forecast", df_result.columns)

        # C. Check for NaNs (The Critical Fix Verification)
        nan_count = df_result["outdoor_temperature_forecast"].isna().sum()
        self.assertEqual(
            0,
            nan_count,
            f"Found {nan_count} NaNs. The alignment fix in command_line.py is not working.",
        )

        # D. Check Data Integrity
        # Since we offset by 15s (very small vs 30min step), the interpolated value
        # should be extremely close to the source value (20.0).
        first_val = df_result["outdoor_temperature_forecast"].iloc[0]
        self.assertAlmostEqual(
            20.0,
            first_val,
            delta=0.5,
            msg="Mapped temperature value diverged significantly from source.",
        )


class TestSchemaVersion(unittest.IsolatedAsyncioTestCase):
    """Cover EMHASS_SCHEMA_VERSION constant and the publish_data early-return attach."""

    def test_constant_value(self):
        from emhass.command_line import EMHASS_SCHEMA_VERSION

        self.assertEqual(EMHASS_SCHEMA_VERSION, "1.0")

    async def test_publish_data_attaches_schema_version_on_saved_entities_path(self):
        from emhass.command_line import EMHASS_SCHEMA_VERSION

        mock_df = pd.DataFrame({"P_grid": [0.0]})
        with (
            patch(
                "emhass.command_line._get_params",
                return_value={"passed_data": {"publish_prefix": "test_"}},
            ),
            patch(
                "emhass.command_line._publish_from_saved_entities",
                new=AsyncMock(return_value=mock_df),
            ),
        ):
            result = await publish_data({}, logger)
        self.assertEqual(result.attrs["emhass_schema_version"], EMHASS_SCHEMA_VERSION)


class TestPublishInfeasibleGuard(unittest.IsolatedAsyncioTestCase):
    """Regression test for #875.

    When the optimization comes back infeasible, perform_optimization returns a
    frame containing only the optim_status column. publish_data used to crash with
    KeyError: 'P_Load' (reported by g1za and ztega). It should now log and return
    None instead.
    """

    async def test_publish_data_infeasible_returns_none(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
        infeasible_res = pd.DataFrame({"optim_status": ["Infeasible"] * 3}, index=idx)
        with patch(
            "emhass.command_line._get_params",
            return_value={"passed_data": {"publish_prefix": ""}},
        ):
            result = await publish_data({}, logger, opt_res_latest=infeasible_res)
        self.assertIsNone(result)


class TestOptimizationCache(unittest.TestCase):
    """Unit tests for the OptimizationCache warm-starting functionality."""

    def setUp(self):
        """Clear the cache before each test."""
        OptimizationCache.clear()
        self.logger = logger

        # Base configuration for testing
        self.optim_conf = {
            "number_of_deferrable_loads": 2,
            "set_use_battery": True,
            "set_use_pv": True,
            "treat_deferrable_load_as_semi_cont": [False, False],
            "set_deferrable_load_single_constant": [False, False],
            "set_deferrable_startup_penalty": [0, 0],
            "set_deferrable_load_as_timeseries": [False, False],
            "delta_forecast_daily": pd.Timedelta(days=1),
            # Deferrable load constraint parameters
            "nominal_power_of_deferrable_loads": [1000, 2000],
            "operating_hours_of_each_deferrable_load": [3, 5],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [48, 48],
        }
        self.plant_conf = {
            "battery_capacity": 10.0,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
        }
        self.retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=30),
        }
        self.costfun = "profit"

    def tearDown(self):
        """Clear the cache after each test."""
        OptimizationCache.clear()

    def test_cache_miss_empty(self):
        """Test that an empty cache returns None."""
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNone(result)

    def test_cache_hit_same_config(self):
        """Test that the same config returns the cached object."""
        # Create a mock Optimization object
        mock_opt = MagicMock()
        mock_opt.name = "test_optimization"

        # Store in cache
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Retrieve from cache
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)
        self.assertEqual(result.name, "test_optimization")

    def test_cache_miss_config_changed(self):
        """Test that changing config invalidates the cache."""
        # Store with original config
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify config - change number of deferrable loads
        modified_optim_conf = self.optim_conf.copy()
        modified_optim_conf["number_of_deferrable_loads"] = 5

        # Should return None (cache miss due to config change)
        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        self.assertIsNone(result)

    def test_cache_miss_battery_config_changed(self):
        """Test that changing battery config invalidates the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify plant config - change battery capacity
        modified_plant_conf = self.plant_conf.copy()
        modified_plant_conf["battery_capacity"] = 20.0

        result = OptimizationCache.get(
            self.optim_conf,
            modified_plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        self.assertIsNone(result)

    def test_cache_miss_costfun_changed(self):
        """Test that changing cost function invalidates the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Change cost function
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            "self-consumption",  # Different costfun
            self.retrieve_hass_conf,
            self.logger,
        )

        self.assertIsNone(result)

    def test_cache_miss_time_step_changed(self):
        """Test that changing optimization time step invalidates the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify time step
        modified_retrieve_conf = self.retrieve_hass_conf.copy()
        modified_retrieve_conf["optimization_time_step"] = pd.Timedelta(minutes=15)

        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            modified_retrieve_conf,
            self.logger,
        )

        self.assertIsNone(result)

    def test_cache_miss_nominal_power_changed(self):
        """Test that changing nominal power of deferrable loads invalidates the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify nominal power for load 0
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["nominal_power_of_deferrable_loads"] = [
            1500,
            2000,
        ]  # Changed from [1000, 2000]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        self.assertIsNone(result)

    def test_cache_hit_operating_hours_changed(self):
        """Test that changing operating hours does NOT invalidate the cache.

        Operating hours are now parameterized via Big-M energy constraints, so the
        cached problem can be reused even when operating hours change. The actual
        operating hours are passed as parameter values before solving.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify operating hours for load 1
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["operating_hours_of_each_deferrable_load"] = [
            3,
            8,
        ]  # Changed from [3, 5]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should still return cached object since operating hours are parameterized
        self.assertEqual(result, mock_opt)

    def test_cache_hit_operating_timesteps_changed(self):
        """Test that changing operating timesteps does NOT invalidate the cache.

        operating_timesteps_of_each_deferrable_load is parameterised via
        param_target_energy and param_required_timesteps (see optimization.py
        ~line 2980-3007). Small tick-to-tick shifts in the operating-time
        requirement (e.g. hot-water-hours-needed translating to 37 vs 38
        timesteps) should NOT trigger a problem rebuild. This is a regression
        test for the fix that added operating_timesteps to
        optim_conf_runtime_keys.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify operating timesteps for load 1
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["operating_timesteps_of_each_deferrable_load"] = [
            6,
            16,
        ]  # was implicit/None before, now varies

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should still return cached object since operating timesteps are parameterized
        self.assertEqual(result, mock_opt)

    def test_cache_hit_start_timestep_changed(self):
        """Test that changing start timesteps does NOT invalidate the cache.

        Start/end timesteps are now parameterized via window masks, so the
        problem structure doesn't change when they change. This enables
        warm-starting for MPC where time windows shift each iteration.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify start timestep for load 0 - should still hit cache
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["start_timesteps_of_each_deferrable_load"] = [
            10,
            0,
        ]  # Changed from [0, 0]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Cache should HIT because start_timesteps are now parameterized
        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)

    def test_cache_hit_end_timestep_changed(self):
        """Test that changing end timesteps does NOT invalidate the cache.

        Start/end timesteps are now parameterized via window masks, so the
        problem structure doesn't change when they change. This enables
        warm-starting for MPC where time windows shift each iteration.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Modify end timestep for load 1 - should still hit cache
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["end_timesteps_of_each_deferrable_load"] = [
            48,
            24,
        ]  # Changed from [48, 48]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Cache should HIT because end_timesteps are now parameterized
        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)

    def test_cache_key_deterministic(self):
        """Test that the same config produces the same cache key."""
        key1 = OptimizationCache._compute_cache_key(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
        )
        key2 = OptimizationCache._compute_cache_key(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
        )

        self.assertEqual(key1, key2)
        # Key should be an OptimizationCacheKey dataclass instance
        self.assertIsInstance(key1, OptimizationCacheKey)

    def test_cache_key_different_for_different_config(self):
        """Test that different configs produce different cache keys."""
        key1 = OptimizationCache._compute_cache_key(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
        )

        modified_optim_conf = self.optim_conf.copy()
        modified_optim_conf["set_use_battery"] = False

        key2 = OptimizationCache._compute_cache_key(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
        )

        self.assertNotEqual(key1, key2)

    def test_cache_clear(self):
        """Test that clear() empties the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Verify it's cached
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNotNone(result)

        # Clear the cache
        OptimizationCache.clear(self.logger)

        # Verify cache is empty
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNone(result)

    def test_cache_get_stats(self):
        """Test that get_stats() returns correct information."""
        # Empty cache stats
        stats = OptimizationCache.get_stats()
        self.assertFalse(stats["has_instance"])
        self.assertIsNone(stats["cache_key"])
        self.assertIsNone(stats["last_used"])

        # After storing
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        stats = OptimizationCache.get_stats()
        self.assertTrue(stats["has_instance"])
        self.assertIsNotNone(stats["cache_key"])
        self.assertIsNotNone(stats["last_used"])

    def test_cache_last_used_updated_on_hit(self):
        """Test that last_used timestamp is updated on cache hit."""
        from datetime import datetime, timedelta

        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Set _last_used to a known earlier time (1 hour ago) to avoid timing issues
        past_time = datetime.now() - timedelta(hours=1)
        OptimizationCache._last_used = past_time

        # Access cache (hit) - this should update _last_used to current time
        OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        second_used = OptimizationCache._last_used

        # second_used should be much more recent than our artificially set past_time
        self.assertGreater(second_used, past_time)
        # And it should be within the last few seconds (not the 1 hour ago we set)
        self.assertLess((datetime.now() - second_used).total_seconds(), 5)

    def test_cache_handles_none_values_in_config(self):
        """Test that cache handles None values in configuration gracefully."""
        optim_conf_with_none = self.optim_conf.copy()
        optim_conf_with_none["delta_forecast_daily"] = None

        retrieve_conf_with_none = self.retrieve_hass_conf.copy()
        retrieve_conf_with_none["optimization_time_step"] = None

        # Should not raise an exception
        key = OptimizationCache._compute_cache_key(
            optim_conf_with_none,
            self.plant_conf,
            self.costfun,
            retrieve_conf_with_none,
        )
        self.assertIsNotNone(key)
        # Key should be an OptimizationCacheKey dataclass instance
        self.assertIsInstance(key, OptimizationCacheKey)

    def test_cache_miss_def_load_config_changed(self):
        """Test that changing def_load_config structure invalidates the cache.

        def_load_config determines which constraint branches are taken
        (standard vs thermal_config vs thermal_battery), so changes to it
        require rebuilding the optimization problem.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Add a thermal_config to a load - this changes constraint structure
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["def_load_config"] = [
            {"thermal_config": {"heating_rate": 5.0}},  # Changed: now has thermal_config
            {},  # Standard load
        ]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should be cache MISS because def_load_config structure changed
        self.assertIsNone(result)

    def test_cache_miss_def_load_config_thermal_battery_added(self):
        """Test that adding thermal_battery to def_load_config invalidates the cache."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Add a thermal_battery to a load - this changes constraint structure
        modified_optim_conf = copy.deepcopy(self.optim_conf)
        modified_optim_conf["def_load_config"] = [
            {},  # Standard load
            {"thermal_battery": {"volume": 10.0}},  # Changed: now has thermal_battery
        ]

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should be cache MISS because def_load_config structure changed
        self.assertIsNone(result)

    def test_cache_hit_thermal_start_temperature_changed(self):
        """Test that changing start_temperature does NOT invalidate cache.

        start_temperature is a runtime parameter that changes between MPC
        iterations. It should not cause a cache miss - instead, the cached
        object's optim_conf should be updated with the new value.
        """
        mock_opt = MagicMock()
        # Set up a thermal_config with initial start_temperature
        optim_conf_with_thermal = copy.deepcopy(self.optim_conf)
        optim_conf_with_thermal["def_load_config"] = [
            {
                "thermal_config": {
                    "heating_rate": 5.0,
                    "cooling_constant": 0.1,
                    "start_temperature": 45.0,  # Initial temperature
                    "desired_temperatures": [50.0] * 10,
                }
            },
        ]

        OptimizationCache.put(
            mock_opt,
            optim_conf_with_thermal,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Change only the runtime parameters (start_temperature, desired_temperatures)
        modified_optim_conf = copy.deepcopy(optim_conf_with_thermal)
        modified_optim_conf["def_load_config"][0]["thermal_config"]["start_temperature"] = (
            42.5  # Different temperature
        )
        modified_optim_conf["def_load_config"][0]["thermal_config"]["desired_temperatures"] = [
            55.0
        ] * 10  # Different desired temps

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should be cache HIT - runtime params don't affect structure
        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)

    def test_cache_miss_thermal_structural_param_changed(self):
        """Test that changing structural thermal params DOES invalidate cache.

        Parameters like heating_rate, cooling_constant affect the constraint
        structure and should cause a cache miss when changed.
        """
        mock_opt = MagicMock()
        # Set up a thermal_config
        optim_conf_with_thermal = copy.deepcopy(self.optim_conf)
        optim_conf_with_thermal["def_load_config"] = [
            {
                "thermal_config": {
                    "heating_rate": 5.0,
                    "cooling_constant": 0.1,
                    "start_temperature": 45.0,
                }
            },
        ]

        OptimizationCache.put(
            mock_opt,
            optim_conf_with_thermal,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Change a structural parameter (heating_rate)
        modified_optim_conf = copy.deepcopy(optim_conf_with_thermal)
        modified_optim_conf["def_load_config"][0]["thermal_config"]["heating_rate"] = (
            10.0  # Different heating rate
        )

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Should be cache MISS - structural param changed
        self.assertIsNone(result)

    def test_cache_key_has_no_battery_capacity_field(self):
        """Test that OptimizationCacheKey no longer has a battery_capacity field."""
        self.assertNotIn("battery_capacity", OptimizationCacheKey.__dataclass_fields__)

    def test_cache_miss_on_optim_conf_structural_change(self):
        """Test that changing structural optim_conf params causes cache MISS."""
        mock_opt = MagicMock()
        optim_conf = copy.deepcopy(self.optim_conf)
        optim_conf["set_nocharge_from_grid"] = False

        OptimizationCache.put(
            mock_opt,
            optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Change structural param
        modified_optim_conf = copy.deepcopy(optim_conf)
        modified_optim_conf["set_nocharge_from_grid"] = True

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNone(result)

    def test_cache_hit_on_optim_conf_runtime_change(self):
        """Test that changing runtime-only optim_conf params still gives cache HIT."""
        mock_opt = MagicMock()
        optim_conf = copy.deepcopy(self.optim_conf)
        optim_conf["lp_solver_timeout"] = 30

        OptimizationCache.put(
            mock_opt,
            optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )

        # Change runtime param only
        modified_optim_conf = copy.deepcopy(optim_conf)
        modified_optim_conf["lp_solver_timeout"] = 60

        result = OptimizationCache.get(
            modified_optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)

    def test_cache_miss_when_num_timesteps_changes(self):
        """Test that changing num_timesteps causes a cache miss.

        When the forecast crosses a DST boundary the number of timesteps in the
        optimisation window can differ from a normal day (e.g. 668 vs 672 for a
        7-day 15-min forecast crossing spring-forward).  Passing a different
        num_timesteps must invalidate the cached problem so the optimizer is
        rebuilt with the correct horizon.
        """
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
            num_timesteps=672,  # normal (non-DST) horizon
        )

        # DST spring-forward shrinks the window by one hour (4 slots at 15 min)
        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
            num_timesteps=668,  # DST-adjusted horizon
        )

        self.assertIsNone(result)

    def test_cache_hit_same_num_timesteps(self):
        """Test that the same num_timesteps still produces a cache hit."""
        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
            num_timesteps=668,
        )

        result = OptimizationCache.get(
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
            num_timesteps=668,
        )

        self.assertIsNotNone(result)
        self.assertIs(result, mock_opt)


class TestDstFixes(unittest.TestCase):
    """Unit tests for DST-boundary fixes in _apply_df_freq_horizon and _load_opt_res_latest."""

    def setUp(self):
        self.retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=15),
            "time_zone": "Europe/Paris",
        }

    def _make_df(self, n: int) -> pd.DataFrame:
        """Build a simple DataFrame with a 15-min UTC index of length n."""
        idx = pd.date_range("2025-03-30 00:00", periods=n, freq="15min", tz="UTC")
        return pd.DataFrame({"P_pv": range(n), "P_load": range(n)}, index=idx)

    def test_apply_df_freq_horizon_clamps_to_df_length(self):
        """_apply_df_freq_horizon must not raise IndexError when prediction_horizon > len(df).

        Root cause: across a spring-forward DST boundary a 7-day 15-min forecast
        produces 668 rows (not 672).  The MPC caller may still pass horizon=672
        (the non-DST default).  The fix clamps the slice to min(horizon, len(df)).
        """
        df = self._make_df(668)  # DST-shortened horizon

        # Must not raise an IndexError
        result = _apply_df_freq_horizon(df, self.retrieve_hass_conf, prediction_horizon=672)

        self.assertEqual(len(result), 668)
        self.assertEqual(result.index[0], df.index[0])
        self.assertEqual(result.index[-1], df.index[-1])

    def test_apply_df_freq_horizon_normal_day(self):
        """On a normal day _apply_df_freq_horizon slices exactly to the horizon."""
        df = self._make_df(672)

        result = _apply_df_freq_horizon(df, self.retrieve_hass_conf, prediction_horizon=672)

        self.assertEqual(len(result), 672)

    def test_apply_df_freq_horizon_none_horizon_returns_full_df(self):
        """When prediction_horizon is None the full DataFrame is returned."""
        df = self._make_df(668)

        result = _apply_df_freq_horizon(df, self.retrieve_hass_conf, prediction_horizon=None)

        self.assertEqual(len(result), 668)

    def test_load_opt_res_latest_handles_mixed_tz_csv(self):
        """_load_opt_res_latest must parse a CSV whose index has mixed UTC offsets.

        Across a spring-forward DST transition timestamps written with +01:00 and
        +02:00 offsets appear in the same CSV.  The old code raised
        'ValueError: Mixed timezones detected'.  The fix uses
        pd.to_datetime(..., utc=True).tz_convert(tz) which handles mixed offsets.
        """
        import pytz

        paris_tz = pytz.timezone("Europe/Paris")

        # Simulate the production case: 8 timestamps that are exactly 15 min apart
        # in UTC, straddling the spring-forward DST boundary (2025-03-30 02:00 Paris
        # = 01:00 UTC).  After tz_convert(Paris) the first 4 show +01:00 and the
        # last 4 show +02:00, giving a mixed-offset index when serialised to CSV.
        idx_utc = pd.date_range("2025-03-30 00:00", periods=8, freq="15min", tz="UTC")
        idx_paris = idx_utc.tz_convert("Europe/Paris")
        # Verify the test assumption: both offsets must be present
        assert "+01:00" in str(idx_paris[0]) and "+02:00" in str(idx_paris[-1])

        df = pd.DataFrame({"P_pv": range(8), "P_load": range(8)}, index=idx_paris)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            # _load_opt_res_latest(..., save_data_to_file=False) looks for default_csv_filename
            csv_path = tmp_path / "opt_res_latest.csv"
            df.index.name = "timestamp"
            df.to_csv(csv_path)

            # Build a minimal input_data_dict
            input_data_dict = {
                "emhass_conf": {"data_path": tmp_path},
                "retrieve_hass_conf": {
                    "time_zone": paris_tz,
                    "optimization_time_step": pd.Timedelta(minutes=15),
                },
            }

            result = _load_opt_res_latest(input_data_dict, logger, save_data_to_file=False)

        # If _load_opt_res_latest returned None it means the mixed-TZ ValueError was
        # raised (or file not found).  The fix makes it return a valid DataFrame.
        self.assertIsNotNone(result, "_load_opt_res_latest returned None; mixed-TZ parse failed")
        self.assertEqual(len(result), 8)
        # After tz_convert all timestamps must be in Europe/Paris.
        # pytz returns different DstTzInfo objects for CET/CEST, so compare by zone name.
        zones = {getattr(ts.tzinfo, "zone", str(ts.tzinfo)) for ts in result.index}
        self.assertEqual(zones, {"Europe/Paris"}, f"Unexpected timezone(s) in result: {zones}")


class TestLoadOptResLatestFreqInference(unittest.TestCase):
    """Unit tests for #976: _load_opt_res_latest must infer the index frequency
    from the saved CSV instead of asserting the current request's
    optimization_time_step onto it.

    An optimization run with a runtime optimization_time_step (e.g. 60) writes
    an hourly CSV; a later publish-data call whose body does not repeat that
    key falls back to the config value (e.g. 30 min) and the old freq
    assignment raised 'Inferred frequency h from passed values does not
    conform to passed frequency 30min'. The publish path only uses the
    timestamps for nearest-index matching, so the frequency baked into the
    saved data is the correct one.
    """

    @staticmethod
    def _write_csv(tmp_path: pathlib.Path, periods: int, freq: str) -> pd.DataFrame:
        idx = pd.date_range("2026-08-01 00:00", periods=periods, freq=freq, tz="UTC")
        df = pd.DataFrame({"P_Load": range(periods)}, index=idx)
        df.index.name = "timestamp"
        df.to_csv(tmp_path / "opt_res_latest.csv")
        return df

    @staticmethod
    def _input_data_dict(tmp_path: pathlib.Path, step_minutes: int) -> dict:
        import pytz

        return {
            "emhass_conf": {"data_path": tmp_path},
            "retrieve_hass_conf": {
                "time_zone": pytz.timezone("Europe/Paris"),
                "optimization_time_step": pd.Timedelta(minutes=step_minutes),
            },
        }

    def _load_result(self, periods: int, freq: str, step_minutes: int) -> pd.DataFrame | None:
        """Write a CSV and load it back through _load_opt_res_latest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            self._write_csv(tmp_path, periods=periods, freq=freq)
            return _load_opt_res_latest(
                self._input_data_dict(tmp_path, step_minutes=step_minutes),
                logger,
                save_data_to_file=False,
            )

    def test_mismatched_step_loads_and_infers_freq(self):
        """An hourly CSV must load under a 30-min config instead of raising.

        This is the #976 repro: naive-mpc-optim wrote the CSV with a runtime
        optimization_time_step of 60; publish-data then arrived using the
        config default of 30 min.
        """
        result = self._load_result(periods=8, freq="60min", step_minutes=30)

        self.assertIsNotNone(result, "_load_opt_res_latest returned None for an hourly CSV")
        self.assertEqual(len(result), 8)
        # The frequency must come from the data, not the request config.
        self.assertEqual(result.index.freq, pd.tseries.frequencies.to_offset("60min"))
        self.assertListEqual(list(result["P_Load"]), list(range(8)))

    def test_matching_step_keeps_frame_and_freq(self):
        """Counterfactual: the healthy matching-step path must be unchanged."""
        result = self._load_result(periods=8, freq="30min", step_minutes=30)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        self.assertEqual(result.index.freq, pd.tseries.frequencies.to_offset("30min"))
        self.assertListEqual(list(result["P_Load"]), list(range(8)))

    def test_single_row_csv_does_not_crash(self):
        """A frame with fewer than 2 rows carries no inferable spacing; it must
        load without crashing (the downstream P_Load/optim_status guard in
        publish_data handles degenerate frames)."""
        result = self._load_result(periods=1, freq="30min", step_minutes=30)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)


class TestOptimizationCacheIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for CLI warm-start flow using actual naive_mpc_optim calls."""

    @staticmethod
    async def get_test_params():
        """Build params with default config."""
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
            params["optim_conf"]["set_use_pv"] = True
        else:
            raise Exception(
                "config_defaults does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        """Set up test fixtures and clear the cache."""
        OptimizationCache.clear()
        self.params = await TestOptimizationCacheIntegration.get_test_params()

    def tearDown(self):
        """Clear cache after each test."""
        OptimizationCache.clear()

    def _make_mpc_inputs(self, runtimeparams, costfun="profit"):
        """Build params_json and runtimeparams_json for an MPC integration test.

        Sets forecast methods to "list" and attaches runtimeparams as passed_data.
        Returns (params_json, runtimeparams_json).
        """
        params = copy.deepcopy(self.params)
        params["passed_data"] = runtimeparams
        for key in (
            "weather_forecast_method",
            "load_forecast_method",
            "load_cost_forecast_method",
            "production_price_forecast_method",
        ):
            params["optim_conf"][key] = "list"
        if costfun != "profit":
            params["optim_conf"]["costfun"] = costfun
        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        return params_json, runtimeparams_json

    async def _run_set_input(self, runtimeparams, costfun="profit", action="naive-mpc-optim"):
        """Call set_input_data_dict with boilerplate handled. Returns input_data_dict."""
        params_json, runtimeparams_json = self._make_mpc_inputs(runtimeparams, costfun)
        return await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

    async def test_mpc_cache_hit_on_repeated_calls(self):
        """Test that repeated MPC calls with same config reuse the cached Optimization object.

        Note: set_input_data_dict creates and caches the Optimization object, so the cache
        is populated after the first set_input_data_dict call, not after naive_mpc_optim.
        """
        costfun = "profit"
        action = "naive-mpc-optim"

        # Set up runtime parameters for a 10-step MPC
        runtimeparams = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200 + i * 10 for i in range(10)],
            "load_cost_forecast": [0.15 + i * 0.01 for i in range(10)],
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        params = copy.deepcopy(self.params)
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        params_json = orjson.dumps(params).decode("utf-8")

        # Verify cache is empty before first call
        stats_before = OptimizationCache.get_stats()
        self.assertFalse(stats_before["has_instance"])

        # First call - set_input_data_dict creates and caches the Optimization object
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        # Cache should now be populated (set_input_data_dict creates the Optimization)
        stats_after_setup = OptimizationCache.get_stats()
        self.assertTrue(stats_after_setup["has_instance"])
        first_cache_key = stats_after_setup["cache_key"]

        opt_res1 = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res1, pd.DataFrame)

        # Second call with same config - should reuse cached Optimization (cache hit)
        # Change forecast values slightly (these don't affect problem structure)
        runtimeparams2 = {
            "pv_power_forecast": [150 * (i + 1) for i in range(10)],  # Different values
            "load_power_forecast": [250 + i * 10 for i in range(10)],
            "load_cost_forecast": [0.20 + i * 0.01 for i in range(10)],
            "prod_price_forecast": [0.08] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.6,  # Different SOC
            "soc_final": 0.6,
        }
        runtimeparams_json2 = orjson.dumps(runtimeparams2).decode("utf-8")
        params["passed_data"] = runtimeparams2
        params_json2 = orjson.dumps(params).decode("utf-8")

        input_data_dict2 = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json2,
            runtimeparams_json2,
            action,
            logger,
            get_data_from_file=True,
        )

        opt_res2 = await naive_mpc_optim(input_data_dict2, logger, debug=True)
        self.assertIsInstance(opt_res2, pd.DataFrame)

        stats_after_second = OptimizationCache.get_stats()
        self.assertTrue(stats_after_second["has_instance"])
        # Cache key should be the same (hit)
        self.assertEqual(stats_after_second["cache_key"], first_cache_key)

    async def test_mpc_cache_hit_with_different_time_windows(self):
        """Test that changing start/end timesteps results in cache HIT (parameterized)."""
        costfun = "profit"
        action = "naive-mpc-optim"

        runtimeparams = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
            "operating_hours_of_each_deferrable_load": [2, 3],
            "start_timesteps_of_each_deferrable_load": [0, 0],
            "end_timesteps_of_each_deferrable_load": [10, 10],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        params = copy.deepcopy(self.params)
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        params_json = orjson.dumps(params).decode("utf-8")

        # First call
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res1 = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res1, pd.DataFrame)

        stats_after_first = OptimizationCache.get_stats()
        first_cache_key = stats_after_first["cache_key"]

        # Second call with different time windows (simulating MPC rolling horizon)
        runtimeparams2 = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
            "operating_hours_of_each_deferrable_load": [2, 3],
            "start_timesteps_of_each_deferrable_load": [2, 1],  # Changed!
            "end_timesteps_of_each_deferrable_load": [8, 9],  # Changed!
        }
        runtimeparams_json2 = orjson.dumps(runtimeparams2).decode("utf-8")
        params["passed_data"] = runtimeparams2
        params_json2 = orjson.dumps(params).decode("utf-8")

        input_data_dict2 = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json2,
            runtimeparams_json2,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res2 = await naive_mpc_optim(input_data_dict2, logger, debug=True)
        self.assertIsInstance(opt_res2, pd.DataFrame)

        stats_after_second = OptimizationCache.get_stats()
        # Cache key should be the same (time windows are parameterized, not in key)
        self.assertEqual(stats_after_second["cache_key"], first_cache_key)

    async def test_mpc_cache_miss_on_structural_change(self):
        """Test that changing structural config (e.g., costfun) causes cache MISS."""
        action = "naive-mpc-optim"

        runtimeparams = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        params = copy.deepcopy(self.params)
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        params_json = orjson.dumps(params).decode("utf-8")

        # First call with costfun="profit"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res1 = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res1, pd.DataFrame)

        stats_after_first = OptimizationCache.get_stats()
        first_cache_key = stats_after_first["cache_key"]

        # Second call with costfun="cost" - should cause cache miss
        # Note: costfun from optim_conf takes precedence, so we must update it in params
        params2 = copy.deepcopy(self.params)
        params2["passed_data"] = runtimeparams
        params2["optim_conf"]["weather_forecast_method"] = "list"
        params2["optim_conf"]["load_forecast_method"] = "list"
        params2["optim_conf"]["load_cost_forecast_method"] = "list"
        params2["optim_conf"]["production_price_forecast_method"] = "list"
        params2["optim_conf"]["costfun"] = "cost"  # This is what changes the costfun
        params_json2 = orjson.dumps(params2).decode("utf-8")

        input_data_dict2 = await set_input_data_dict(
            emhass_conf,
            "cost",
            params_json2,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res2 = await naive_mpc_optim(input_data_dict2, logger, debug=True)
        self.assertIsInstance(opt_res2, pd.DataFrame)

        stats_after_second = OptimizationCache.get_stats()
        # Cache key should be different (costfun changed)
        self.assertNotEqual(stats_after_second["cache_key"], first_cache_key)

    async def test_mpc_multiple_iterations_simulate_rolling_horizon(self):
        """Simulate multiple MPC iterations as in real rolling-horizon operation."""
        costfun = "profit"
        action = "naive-mpc-optim"

        params = copy.deepcopy(self.params)
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"

        # Simulate 4 MPC iterations with shifting time windows
        cache_keys = []
        for iteration in range(4):
            runtimeparams = {
                "pv_power_forecast": [100 * (i + 1 + iteration) for i in range(10)],
                "load_power_forecast": [200 + iteration * 5] * 10,
                "load_cost_forecast": [0.15 + iteration * 0.01] * 10,
                "prod_price_forecast": [0.05] * 10,
                "prediction_horizon": 10,
                "soc_init": 0.5 + iteration * 0.05,
                "soc_final": 0.5,
                "operating_hours_of_each_deferrable_load": [2, 3],
                # Simulate rolling horizon: windows shift each iteration
                "start_timesteps_of_each_deferrable_load": [iteration, iteration],
                "end_timesteps_of_each_deferrable_load": [10, 10],
            }
            runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
            params["passed_data"] = runtimeparams
            params_json = orjson.dumps(params).decode("utf-8")

            input_data_dict = await set_input_data_dict(
                emhass_conf,
                costfun,
                params_json,
                runtimeparams_json,
                action,
                logger,
                get_data_from_file=True,
            )
            opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
            self.assertIsInstance(opt_res, pd.DataFrame)
            self.assertEqual(len(opt_res), 10)

            stats = OptimizationCache.get_stats()
            cache_keys.append(stats["cache_key"])

        # All iterations should have the same cache key (cache was reused)
        self.assertTrue(all(key == cache_keys[0] for key in cache_keys))

    async def test_thermal_start_temperature_constraint_updates_on_cache_hit(self):
        """Verify that thermal start_temperature constraint value actually changes on cache hit.

        This test ensures that:
        1. The CVXPY constraint predicted_temp[0] == start_temperature uses a cp.Parameter
        2. On cache hit, updating start_temperature actually changes the constraint result
        3. The optimization result reflects the new start_temperature, not the old baked-in value

        This catches the bug where start_temperature was a raw float baked into constraints
        at problem build time, causing cache hits to use stale temperature values.
        """
        from emhass.optimization import Optimization

        # Create minimal configs for a thermal load optimization
        retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=30),
            "time_zone": "UTC",
            "sensor_power_photovoltaics": "pv",
            "sensor_power_load_no_var_loads": "load",
        }

        plant_conf = {
            "pv_module_model": None,
            "pv_inverter_model": None,
            "surface_tilt": 30,
            "surface_azimuth": 180,
            "modules_per_string": 10,
            "strings_per_inverter": 1,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
            "set_use_battery": False,
            "battery_dynamic_max": 1.0,
            "battery_dynamic_min": -1.0,
            "battery_discharge_power_max": 1000,
            "battery_charge_power_max": 1000,
            "battery_nominal_energy_capacity": 5000,
            "battery_minimum_state_of_charge": 0.1,
            "battery_maximum_state_of_charge": 0.9,
            "battery_discharge_efficiency": 0.95,
            "battery_charge_efficiency": 0.95,
        }

        n_timesteps = 48
        initial_start_temp = 22.0
        updated_start_temp = 18.5

        optim_conf = {
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [2000],
            "operating_hours_of_each_deferrable_load": [8],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [n_timesteps],
            "treat_deferrable_load_as_semi_cont": [False],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0],
            "set_use_battery": False,
            "set_total_pv_sell": False,
            "delta_forecast_daily": 1,
            "def_load_config": [
                {
                    "thermal_config": {
                        "start_temperature": initial_start_temp,
                        "cooling_constant": 0.1,
                        "heating_rate": 2.0,
                        "overshoot_temperature": None,
                        "desired_temperatures": [],
                        "min_temperatures": [19.0] * n_timesteps,
                        "max_temperatures": [25.0] * n_timesteps,
                        "sense": "heat",
                        "thermal_inertia": 0.0,
                    }
                }
            ],
        }

        # Create test data
        start = pd.Timestamp.now(tz="UTC")
        idx = pd.date_range(start=start, periods=n_timesteps, freq="30min", tz="UTC")
        data_opt = pd.DataFrame(
            {
                "outdoor_temperature_forecast": [10.0] * n_timesteps,
                "pv_forecast": [500.0] * n_timesteps,
                "load_forecast": [1000.0] * n_timesteps,
            },
            index=idx,
        )
        p_pv = np.array([500.0] * n_timesteps)
        p_load = np.array([1000.0] * n_timesteps)
        unit_load_cost = np.array([0.15] * n_timesteps)
        unit_prod_price = np.array([0.05] * n_timesteps)

        # Create Optimization object
        opt = Optimization(
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            plant_conf=plant_conf,
            var_load_cost="unit_load_cost",
            var_prod_price="unit_prod_price",
            costfun="profit",
            emhass_conf=emhass_conf,
            logger=logger,
        )

        # First optimization with initial_start_temp
        opt.perform_optimization(
            data_opt=data_opt,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        self.assertEqual(opt.optim_status, "Optimal")
        temp_var_1 = opt.predicted_temps.get(0)
        self.assertIsNotNone(temp_var_1)
        # Capture the actual value (not just the Variable reference, which gets overwritten)
        first_temp_value = float(temp_var_1.value[0])
        self.assertAlmostEqual(
            first_temp_value,
            initial_start_temp,
            places=2,
            msg=f"First call: predicted_temp[0] should be {initial_start_temp}",
        )

        # Simulate cache hit: update optim_conf and call update_thermal_start_temps
        # This is what command_line.py does on cache hit
        opt.optim_conf["def_load_config"][0]["thermal_config"]["start_temperature"] = (
            updated_start_temp
        )
        optim_conf_updated = copy.deepcopy(optim_conf)
        optim_conf_updated["def_load_config"][0]["thermal_config"]["start_temperature"] = (
            updated_start_temp
        )
        opt.update_thermal_start_temps(optim_conf_updated)

        # Second optimization (reuses cached problem structure)
        opt.perform_optimization(
            data_opt=data_opt,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        self.assertEqual(opt.optim_status, "Optimal")
        temp_var_2 = opt.predicted_temps.get(0)
        self.assertIsNotNone(temp_var_2)
        second_temp_value = float(temp_var_2.value[0])
        self.assertAlmostEqual(
            second_temp_value,
            updated_start_temp,
            places=2,
            msg=f"Second call (cache hit): predicted_temp[0] should be {updated_start_temp}, "
            f"not the old baked-in value {initial_start_temp}",
        )

        # Verify the constraint actually changed (not just a coincidence)
        self.assertNotAlmostEqual(
            first_temp_value,
            second_temp_value,
            places=1,
            msg="The two optimization results should have different starting temperatures",
        )

    async def test_thermal_outdoor_temp_updates_on_cache_hit(self):
        """Verify that outdoor_temp forecast updates affect thermal dynamics on cache hit.

        This test ensures that changing weather forecasts between MPC iterations
        actually affects the thermal model behavior, not using stale baked-in values.
        """
        from emhass.optimization import Optimization

        retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=30),
            "time_zone": "UTC",
            "sensor_power_photovoltaics": "pv",
            "sensor_power_load_no_var_loads": "load",
        }

        plant_conf = {
            "pv_module_model": None,
            "pv_inverter_model": None,
            "surface_tilt": 30,
            "surface_azimuth": 180,
            "modules_per_string": 10,
            "strings_per_inverter": 1,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
            "set_use_battery": False,
            "battery_dynamic_max": 1.0,
            "battery_dynamic_min": -1.0,
            "battery_discharge_power_max": 1000,
            "battery_charge_power_max": 1000,
            "battery_nominal_energy_capacity": 5000,
            "battery_minimum_state_of_charge": 0.1,
            "battery_maximum_state_of_charge": 0.9,
            "battery_discharge_efficiency": 0.95,
            "battery_charge_efficiency": 0.95,
        }

        n_timesteps = 10

        optim_conf = {
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [2000],
            "operating_hours_of_each_deferrable_load": [4],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [n_timesteps],
            "treat_deferrable_load_as_semi_cont": [False],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0],
            "set_use_battery": False,
            "set_total_pv_sell": False,
            "delta_forecast_daily": 1,
            "def_load_config": [
                {
                    "thermal_config": {
                        "start_temperature": 20.0,
                        "cooling_constant": 0.1,
                        "heating_rate": 2.0,
                        "min_temperatures": [18.0] * n_timesteps,
                        "max_temperatures": [25.0] * n_timesteps,
                        "sense": "heat",
                        "thermal_inertia": 0.0,
                    }
                }
            ],
        }

        # Create Optimization object
        opt = Optimization(
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            plant_conf=plant_conf,
            var_load_cost="unit_load_cost",
            var_prod_price="unit_prod_price",
            costfun="profit",
            emhass_conf=emhass_conf,
            logger=logger,
        )

        p_pv = np.array([500.0] * n_timesteps)
        p_load = np.array([1000.0] * n_timesteps)
        unit_load_cost = np.array([0.15] * n_timesteps)
        unit_prod_price = np.array([0.05] * n_timesteps)

        # First call with cold outdoor temp (10°C) - more heating needed
        start1 = pd.Timestamp.now(tz="UTC")
        idx1 = pd.date_range(start=start1, periods=n_timesteps, freq="30min", tz="UTC")
        data_opt_cold = pd.DataFrame(
            {"outdoor_temperature_forecast": [10.0] * n_timesteps},
            index=idx1,
        )

        opt.perform_optimization(
            data_opt=data_opt_cold,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        # Capture outdoor_temp parameter value after first call
        outdoor_temp_param_first = opt.param_thermal[0]["outdoor_temp"].value.copy()

        # Second call with warm outdoor temp (25°C) - less heating needed
        data_opt_warm = pd.DataFrame(
            {"outdoor_temperature_forecast": [25.0] * n_timesteps},
            index=idx1,
        )

        opt.perform_optimization(
            data_opt=data_opt_warm,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        # Capture outdoor_temp parameter value after second call
        outdoor_temp_param_second = opt.param_thermal[0]["outdoor_temp"].value.copy()

        # Verify the outdoor_temp parameter was updated
        self.assertAlmostEqual(outdoor_temp_param_first[0], 10.0, places=1)
        self.assertAlmostEqual(outdoor_temp_param_second[0], 25.0, places=1)
        self.assertNotAlmostEqual(
            outdoor_temp_param_first[0],
            outdoor_temp_param_second[0],
            places=1,
            msg="Outdoor temp parameter should update between calls",
        )

    async def test_thermal_battery_params_update_on_cache_hit(self):
        """Verify thermal_battery derived parameters update correctly on cache hit.

        Tests that heating_demand, thermal_losses, and heatpump_cops are recomputed
        when outdoor_temp or indoor_target_temperature changes.
        """
        from emhass.optimization import Optimization

        retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=30),
            "time_zone": "UTC",
            "sensor_power_photovoltaics": "pv",
            "sensor_power_load_no_var_loads": "load",
        }

        plant_conf = {
            "pv_module_model": None,
            "pv_inverter_model": None,
            "surface_tilt": 30,
            "surface_azimuth": 180,
            "modules_per_string": 10,
            "strings_per_inverter": 1,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
            "set_use_battery": False,
            "battery_dynamic_max": 1.0,
            "battery_dynamic_min": -1.0,
            "battery_discharge_power_max": 1000,
            "battery_charge_power_max": 1000,
            "battery_nominal_energy_capacity": 5000,
            "battery_minimum_state_of_charge": 0.1,
            "battery_maximum_state_of_charge": 0.9,
            "battery_discharge_efficiency": 0.95,
            "battery_charge_efficiency": 0.95,
        }

        n_timesteps = 10

        optim_conf = {
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [3000],
            "operating_hours_of_each_deferrable_load": [4],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [n_timesteps],
            "treat_deferrable_load_as_semi_cont": [False],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0],
            "set_use_battery": False,
            "set_total_pv_sell": False,
            "delta_forecast_daily": 1,
            "def_load_config": [
                {
                    "thermal_battery": {
                        "start_temperature": 22.0,
                        "indoor_target_temperature": 22.0,
                        "volume": 8,
                        "u_value": 0.231,
                        "envelope_area": 314.0,
                        "ventilation_rate": 0.41,
                        "heated_volume": 356.0,
                        "carnot_efficiency": 0.39,
                        "supply_temperature": 30.0,
                        "min_temperatures": [21.0] * n_timesteps,
                        "max_temperatures": [24.0] * n_timesteps,
                    }
                }
            ],
        }

        opt = Optimization(
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            plant_conf=plant_conf,
            var_load_cost="unit_load_cost",
            var_prod_price="unit_prod_price",
            costfun="profit",
            emhass_conf=emhass_conf,
            logger=logger,
        )

        p_pv = np.array([500.0] * n_timesteps)
        p_load = np.array([1000.0] * n_timesteps)
        unit_load_cost = np.array([0.15] * n_timesteps)
        unit_prod_price = np.array([0.05] * n_timesteps)

        start1 = pd.Timestamp.now(tz="UTC")
        idx1 = pd.date_range(start=start1, periods=n_timesteps, freq="30min", tz="UTC")

        # First call with cold outdoor temp (0°C)
        data_opt_cold = pd.DataFrame(
            {"outdoor_temperature_forecast": [0.0] * n_timesteps},
            index=idx1,
        )

        opt.perform_optimization(
            data_opt=data_opt_cold,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        # Capture derived parameter values after first call
        cops_first = opt.param_thermal[0]["heatpump_cops"].value.copy()
        heating_demand_first = opt.param_thermal[0]["heating_demand"].value.copy()

        # Second call with warm outdoor temp (15°C)
        data_opt_warm = pd.DataFrame(
            {"outdoor_temperature_forecast": [15.0] * n_timesteps},
            index=idx1,
        )

        opt.perform_optimization(
            data_opt=data_opt_warm,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        # Capture derived parameter values after second call
        cops_second = opt.param_thermal[0]["heatpump_cops"].value.copy()
        heating_demand_second = opt.param_thermal[0]["heating_demand"].value.copy()

        # Verify COPs changed (warmer outdoor = higher COP)
        self.assertGreater(
            cops_second[0],
            cops_first[0],
            "Heatpump COP should be higher with warmer outdoor temp",
        )

        # Verify heating demand changed (warmer outdoor = lower heating demand)
        self.assertLess(
            heating_demand_second[0],
            heating_demand_first[0],
            "Heating demand should be lower with warmer outdoor temp",
        )

    async def test_thermal_min_max_temps_update_on_cache_hit(self):
        """Verify min/max temperature constraints update on cache hit.

        Tests that changing min_temperatures/max_temperatures in config
        actually affects the optimization constraints.
        """
        from emhass.optimization import Optimization

        retrieve_hass_conf = {
            "optimization_time_step": pd.Timedelta(minutes=30),
            "time_zone": "UTC",
            "sensor_power_photovoltaics": "pv",
            "sensor_power_load_no_var_loads": "load",
        }

        plant_conf = {
            "pv_module_model": None,
            "pv_inverter_model": None,
            "surface_tilt": 30,
            "surface_azimuth": 180,
            "modules_per_string": 10,
            "strings_per_inverter": 1,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
            "set_use_battery": False,
            "battery_dynamic_max": 1.0,
            "battery_dynamic_min": -1.0,
            "battery_discharge_power_max": 1000,
            "battery_charge_power_max": 1000,
            "battery_nominal_energy_capacity": 5000,
            "battery_minimum_state_of_charge": 0.1,
            "battery_maximum_state_of_charge": 0.9,
            "battery_discharge_efficiency": 0.95,
            "battery_charge_efficiency": 0.95,
        }

        n_timesteps = 10
        initial_min_temp = 18.0
        updated_min_temp = 22.0

        optim_conf = {
            "number_of_deferrable_loads": 1,
            "nominal_power_of_deferrable_loads": [2000],
            "operating_hours_of_each_deferrable_load": [4],
            "start_timesteps_of_each_deferrable_load": [0],
            "end_timesteps_of_each_deferrable_load": [n_timesteps],
            "treat_deferrable_load_as_semi_cont": [False],
            "set_deferrable_load_single_constant": [False],
            "set_deferrable_startup_penalty": [0],
            "set_use_battery": False,
            "set_total_pv_sell": False,
            "delta_forecast_daily": 1,
            "def_load_config": [
                {
                    "thermal_config": {
                        "start_temperature": 20.0,
                        "cooling_constant": 0.1,
                        "heating_rate": 2.0,
                        "min_temperatures": [initial_min_temp] * n_timesteps,
                        "max_temperatures": [26.0] * n_timesteps,
                        "sense": "heat",
                        "thermal_inertia": 0.0,
                    }
                }
            ],
        }

        opt = Optimization(
            retrieve_hass_conf=retrieve_hass_conf,
            optim_conf=optim_conf,
            plant_conf=plant_conf,
            var_load_cost="unit_load_cost",
            var_prod_price="unit_prod_price",
            costfun="profit",
            emhass_conf=emhass_conf,
            logger=logger,
        )

        p_pv = np.array([500.0] * n_timesteps)
        p_load = np.array([1000.0] * n_timesteps)
        unit_load_cost = np.array([0.15] * n_timesteps)
        unit_prod_price = np.array([0.05] * n_timesteps)

        start1 = pd.Timestamp.now(tz="UTC")
        idx1 = pd.date_range(start=start1, periods=n_timesteps, freq="30min", tz="UTC")
        data_opt = pd.DataFrame(
            {"outdoor_temperature_forecast": [10.0] * n_timesteps},
            index=idx1,
        )

        # First call with initial min_temp
        opt.perform_optimization(
            data_opt=data_opt,
            p_pv=p_pv,
            p_load=p_load,
            unit_load_cost=unit_load_cost,
            unit_prod_price=unit_prod_price,
        )

        min_temps_first = opt.param_thermal[0]["min_temps"].value.copy()

        # Update min_temperatures in config and call update_thermal_params
        opt.optim_conf["def_load_config"][0]["thermal_config"]["min_temperatures"] = [
            updated_min_temp
        ] * n_timesteps
        opt.update_thermal_params(opt.optim_conf, data_opt, p_load)

        min_temps_second = opt.param_thermal[0]["min_temps"].value.copy()

        # Verify min_temps parameter was updated
        self.assertAlmostEqual(min_temps_first[1], initial_min_temp, places=1)
        self.assertAlmostEqual(min_temps_second[1], updated_min_temp, places=1)

    async def test_mpc_cache_plant_conf_updates(self):
        """
        Verify that structural plant_conf changes trigger a cache miss,
        and non-structural plant_conf changes update correctly on a hit.
        """
        costfun = "profit"
        action = "naive-mpc-optim"  # Switched to MPC to use shorter prediction horizon
        # Set up the base runtime parameters with lists to bypass PVLib/pickles
        base_runtimeparams = {
            "pv_power_forecast": [100] * 10,
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
        }
        # Base run
        runtimeparams_1 = base_runtimeparams.copy()
        runtimeparams_1.update(
            {"battery_target_state_of_charge": 0.5, "battery_minimum_state_of_charge": 0.2}
        )
        params_1 = copy.deepcopy(self.params)
        params_1["passed_data"] = runtimeparams_1
        # Explicitly bypass forecast downloads/computations
        params_1["optim_conf"]["weather_forecast_method"] = "list"
        params_1["optim_conf"]["load_forecast_method"] = "list"
        params_1["optim_conf"]["load_cost_forecast_method"] = "list"
        params_1["optim_conf"]["production_price_forecast_method"] = "list"

        input_data_1 = await set_input_data_dict(
            emhass_conf,
            costfun,
            orjson.dumps(params_1).decode("utf-8"),
            orjson.dumps(runtimeparams_1).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )
        opt_1 = input_data_1["opt"]
        # Cache Hit: Change target SOC (Should NOT trigger miss, but SHOULD update plant_conf)
        runtimeparams_2 = base_runtimeparams.copy()
        runtimeparams_2.update(
            {"battery_target_state_of_charge": 0.8, "battery_minimum_state_of_charge": 0.2}
        )
        params_2 = copy.deepcopy(params_1)
        params_2["passed_data"] = runtimeparams_2
        input_data_2 = await set_input_data_dict(
            emhass_conf,
            costfun,
            orjson.dumps(params_2).decode("utf-8"),
            orjson.dumps(runtimeparams_2).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )
        opt_2 = input_data_2["opt"]
        self.assertIs(opt_1, opt_2, "Target SOC change should result in Cache Hit")
        self.assertEqual(
            opt_2.plant_conf["battery_target_state_of_charge"],
            0.8,
            "Stale plant_conf on cache hit!",
        )
        # Cache Miss: Change Minimum SOC limit (Must rebuild CVXPY constraint)
        runtimeparams_3 = base_runtimeparams.copy()
        runtimeparams_3.update(
            {"battery_target_state_of_charge": 0.8, "battery_minimum_state_of_charge": 0.4}
        )
        params_3 = copy.deepcopy(params_1)
        params_3["passed_data"] = runtimeparams_3
        input_data_3 = await set_input_data_dict(
            emhass_conf,
            costfun,
            orjson.dumps(params_3).decode("utf-8"),
            orjson.dumps(runtimeparams_3).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )
        opt_3 = input_data_3["opt"]
        self.assertIsNot(
            opt_2, opt_3, "Min SOC change must trigger Cache Miss to rebuild constraints"
        )

    async def test_cache_miss_on_battery_dynamic_change(self):
        """Test that changing set_battery_dynamic causes cache MISS."""
        base_rt = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }

        opt_1 = (await self._run_set_input(base_rt))["opt"]
        opt_2 = (await self._run_set_input({**base_rt, "set_battery_dynamic": True}))["opt"]

        self.assertIsNot(opt_1, opt_2, "set_battery_dynamic change must cause cache MISS")

    async def test_cache_miss_on_weight_battery_change(self):
        """Test that changing weight_battery_discharge causes cache MISS."""
        base_rt = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }

        opt_1 = (await self._run_set_input(base_rt))["opt"]
        opt_2 = (await self._run_set_input({**base_rt, "weight_battery_discharge": 2.0}))["opt"]

        self.assertIsNot(opt_1, opt_2, "weight_battery_discharge change must cause cache MISS")

    async def test_cache_miss_on_grid_policy_change(self):
        """Test that changing set_nocharge_from_grid causes cache MISS."""
        base_rt = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }

        opt_1 = (await self._run_set_input(base_rt))["opt"]
        opt_2 = (await self._run_set_input({**base_rt, "set_nocharge_from_grid": True}))["opt"]

        self.assertIsNot(opt_1, opt_2, "set_nocharge_from_grid change must cause cache MISS")

    async def test_def_current_state_updates_on_cache_hit(self):
        """Test that def_current_state is updated via CVXPY Parameters on cache hit."""
        # config_defaults.json now ships a single deferrable load; this test
        # exercises a 2-load def_current_state array, so pad optim_conf the
        # same way check_def_loads/build_params would for number_of_deferrable_loads=2.
        self.params["optim_conf"]["number_of_deferrable_loads"] = 2
        for name, default in utils.DEF_LOAD_ARRAY_PARAMS.items():
            self.params["optim_conf"][name] = utils.check_def_loads(
                2, self.params["optim_conf"], default, name, logger
            )
        base_rt = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
            "def_current_state": [False, False],
        }

        # First call
        input_data_dict = await self._run_set_input(base_rt)
        opt_1 = input_data_dict["opt"]
        await naive_mpc_optim(input_data_dict, logger, debug=True)

        # Second call with def_current_state changed (should be cache HIT)
        input_data_dict2 = await self._run_set_input(
            {**base_rt, "def_current_state": [True, False]}
        )
        opt_2 = input_data_dict2["opt"]

        # Should be cache HIT (same object)
        self.assertIs(opt_1, opt_2, "def_current_state change should NOT cause cache MISS")

        # Run optimization to trigger parameter update in perform_optimization
        await naive_mpc_optim(input_data_dict2, logger, debug=True)

        # Verify the CVXPY parameter was updated
        self.assertEqual(opt_2.param_def_current_state[0].value, 1.0)
        self.assertEqual(opt_2.param_def_current_state[1].value, 0.0)

    async def test_cache_hit_on_battery_power_limit_change(self):
        """Test that changing battery_discharge_power_max keeps cache HIT.

        Both battery power limits live in plant_runtime_keys and are wired
        through to cp.Parameters whose .value is updated per solve, so a
        change shouldn't invalidate the cached Optimization instance.
        """
        base_rt = {
            "pv_power_forecast": [100 * (i + 1) for i in range(10)],
            "load_power_forecast": [200] * 10,
            "load_cost_forecast": [0.15] * 10,
            "prod_price_forecast": [0.05] * 10,
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.5,
        }

        opt_1 = (await self._run_set_input(base_rt))["opt"]
        opt_2 = (await self._run_set_input({**base_rt, "battery_discharge_power_max": 9999}))["opt"]

        self.assertIs(opt_1, opt_2, "battery_discharge_power_max change must keep cache HIT")
        # And the new value has been propagated to the CVXPY Parameter.
        # #610: param_battery_discharge_power_max is now a per-battery list
        # (index 0 at number_of_batteries==1, uniformly indexed like every
        # other per-battery Parameter/Variable in optimization.py).
        self.assertAlmostEqual(float(opt_2.param_battery_discharge_power_max[0].value), 9999.0)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
