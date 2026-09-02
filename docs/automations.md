# Home Assistant Automations

To automate EMHASS with Home Assistant, we will need to define some shell commands in the Home Assistant `configuration.yaml` file and some basic automations in the `automations.yaml` file.
In the next few paragraphs, we are going to consider the `dayahead-optim` optimization strategy, which is also the first that was implemented, and we will also cover how to publish the optimization  results.  
Additional optimization strategies were developed later, that can be used in combination with/replace the `dayahead-optim` strategy, such as MPC, or to expand the functionalities such as the Machine Learning method to predict your household consumption. Each of them has some specificities and features and will be considered in dedicated sections.

## Dayahead Optimization - Method 1) Add-on and docker standalone

We can use the `shell_command` integration in `configuration.yaml`:
```yaml
shell_command:
  dayahead_optim: "curl -i -H \"Content-Type:application/json\" -X POST -d '{}' http://localhost:5000/action/dayahead-optim"
  publish_data: "curl -i -H \"Content-Type:application/json\" -X POST -d '{}' http://localhost:5000/action/publish-data"
```
An alternative that will be useful when passing data at runtime (see dedicated section), we can use the the `rest_command` instead:
```yaml
rest_command:
  url: http://127.0.0.1:5000/action/dayahead-optim
  method: POST
  headers:
    content-type: application/json
  payload: >-
    {}
```

## Dayahead Optimization - Method 2) Legacy method using a Python virtual environment

In `configuration.yaml`:
```yaml
shell_command:
  dayahead_optim: ~/emhass/scripts/dayahead_optim.sh
  publish_data: ~/emhass/scripts/publish_data.sh
```
Create the file `dayahead_optim.sh` with the following content:
```bash
#!/bin/bash
. ~/emhassenv/bin/activate
emhass --action 'dayahead-optim' --config ~/emhass/config.json
```
And the file `publish_data.sh` with the following content:
```bash
#!/bin/bash
. ~/emhassenv/bin/activate
emhass --action 'publish-data' --config ~/emhass/config.json
```
Then specify user rights and make the files executables:
```bash
sudo chmod -R 755 ~/emhass/scripts/dayahead_optim.sh
sudo chmod -R 755 ~/emhass/scripts/publish_data.sh
sudo chmod +x ~/emhass/scripts/dayahead_optim.sh
sudo chmod +x ~/emhass/scripts/publish_data.sh
```

## Common for any installation method

### Options 1, Home Assistant automate publish

In `automations.yaml`:
```yaml
- alias: EMHASS day-ahead optimization
  trigger:
    platform: time
    at: '05:30:00'
  action:
  - service: shell_command.dayahead_optim
- alias: EMHASS publish data
  trigger:
  - minutes: /5
    platform: time_pattern
  action:
  - service: shell_command.publish_data
```
In these automations the day-ahead optimization is performed once a day, every day at 5:30am, and the data *(output of automation)* is published every 5 minutes.

### Option 2, EMHASS automated publish 

In `automations.yaml`:
```yaml
- alias: EMHASS day-ahead optimization
  trigger:
    platform: time
    at: '05:30:00'
  action:
  - service: shell_command.dayahead_optim
  - service: shell_command.publish_data
```
in configuration page/`config.json` 
```json
"method_ts_round": "first"
"continual_publish": true
```
In this automation, the day-ahead optimization is performed once a day, every day at 5:30am. 
If the `optimization_time_step` parameter is set to `30` *(default)* in the configuration, the results of the day-ahead optimization will generate 48 values *(for each entity)*, a value for every 30 minutes in a day *(i.e. 24 hrs x 2)*.

Setting the parameter `continual_publish` to `true` in the configuration page will allow EMHASS to store the optimization results as entities/sensors into separate json files. `continual_publish` will periodically (every `optimization_time_step` amount of minutes) run a publish, and publish the optimization results of each generated entities/sensors to Home Assistant. The current state of the sensor/entity being updated every time publish runs, selecting one of the 48 stored values, by comparing the stored values' timestamps, the current timestamp and [`'method_ts_round': "first"`](https://emhass.readthedocs.io/en/latest/publish_data.html#the-publish-data-specificities) to select the optimal stored value for the current state.

option 1 and 2 are very similar, however, option 2 (`continual_publish`) will require a CPU thread to constantly be run inside of EMHASS, lowering efficiency. The reason why you may pick one over the other is explained in more detail below in [continual_publish](https://emhass.readthedocs.io/en/latest/publish_data.html#continual-publish-emhass-automation).

Lastly, we can link an EMHASS published entity/sensor's current state to a Home Assistant entity on/off switch, controlling a desired controllable load. 
For example, imagine that I want to control my water heater. I can use a published `deferrable` EMHASS entity to control my water heater's desired behavior. In this case, we could use an automation like the below, to control the desired water heater on and off:
  
on:
```yaml
automation:
- alias: Water Heater Optimized ON
  trigger:
  - minutes: /5
    platform: time_pattern
  condition:
  - condition: numeric_state
    entity_id: sensor.p_deferrable0
    above: 0.1
  action:
    - service: homeassistant.turn_on
      entity_id: switch.water_heater_switch
```
off:
```yaml
automation:
- alias: Water Heater Optimized OFF
  trigger:
  - minutes: /5
    platform: time_pattern
  condition:
  - condition: numeric_state
    entity_id: sensor.p_deferrable0
    below: 0.1
  action:
    - service: homeassistant.turn_off
      entity_id: switch.water_heater_switch
```
These automations will turn on and off the Home Assistant entity `switch.water_heater_switch` using the current state from the EMHASS entity `sensor.p_deferrable0`. `sensor.p_deferrable0`  being the entity generated from the EMHASS day-ahead optimization and published by examples above. The `sensor.p_deferrable0` entity's current state is updated every 30 minutes (or `optimization_time_step` minutes) via an automated publish option 1 or 2. *(selecting one of the 48 stored data values)*

## RC model forecast

`rc-model-forecast` is a report-only action (it never calls a device service) that simulates the indoor temperature forward assuming heating stays off, using the fitted thermal-mass physics model (see `scripts/thermal_mass_physics_model.py` - run it at least once to produce `data/rc_model_params.json` before enabling this). It publishes `sensor.indoor_temp_forecast` (the predicted curve, as a `predicted_temperatures` attribute) and `sensor.heating_needed_by` (the first timestamp the forecast crosses `rc_model_forecast_comfort_min_temp`, or `"beyond_horizon"`).

It needs `rc_model_forecast_enabled: true` in your config, and a weather forecast that actually reaches as far as `rc_model_forecast_horizon_hours` (default 72h). EMHASS's day-ahead weather window is controlled by `delta_forecast_daily`, which is read once when the Forecast object is built - **pass it explicitly in the request body** so it isn't left at the default 1-day window:
```yaml
rest_command:
  rc_model_forecast:
    url: http://127.0.0.1:5000/action/rc-model-forecast
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {"delta_forecast_daily": 3}
```
Keep the `3` here in sync with `rc_model_forecast_horizon_hours / 24` in your config - if they drift apart, EMHASS logs a warning (not an error) and simply forecasts as far as the data actually reaches.

In `automations.yaml`, trigger it a few times a day (there's no need to run it as often as `dayahead-optim` - the forecast only meaningfully changes as the weather forecast itself updates):
```yaml
- alias: EMHASS rc-model forecast
  trigger:
    platform: time_pattern
    hours: '/6'
  action:
  - service: rest_command.rc_model_forecast
```

Unlike the room/heat-pump/EV target sensors elsewhere in this fork, `sensor.indoor_temp_forecast` and `sensor.heating_needed_by` are purely informational - nothing in Home Assistant needs to *consume* them for this feature to be useful (you'd typically just look at the dashboard, or add your own notification automation on `sensor.heating_needed_by`), so no staleness watchdog is needed here: there's no device that could get stuck in a stale commanded state.

## Weekly model auto-refit

`rc-model-refit` periodically refits the thermal-mass physics model against fresh history and deploys it to `data/rc_model_params.json` - the same file `scripts/thermal_mass_physics_model.py --deploy-path` writes when you run it by hand, and the same file `rc-model-forecast` reads. Like every other EMHASS action, there's no scheduler inside EMHASS itself - you trigger it externally, same as `dayahead-optim`.

**Requires InfluxDB**, not just Home Assistant's own recorder: `rc_model_refit_window_days` (default 60) is normally far longer than the recorder's own retention (`purge_keep_days`, often 10 days by default) - see [InfluxDB as a data source](passing_data.md#influxdb-as-a-data-source) for why REST/WebSocket silently degrades to low-resolution stats beyond that window. Set `use_influxdb: true` plus `influxdb_host`/`influxdb_database`/etc. in your config, and add `influxdb_username`/`influxdb_password` to `secrets_emhass.yaml` (see the template file) - EMHASS routes every history pull through InfluxDB automatically once this is set, no other change needed.

You'll also need to point EMHASS at where each training signal actually lives in Home Assistant: at least one Rooms-tab `heatpump_room_temp_sensors` entry (required - its first configured entry is the fit target) plus the optional `heatpump_power_sensor`, `heatpump_gas_meter_sensor`, `heatpump_flow_temp_sensor`, `heatpump_outdoor_temp_sensor`, `heatpump_duty_sensor`, `heatpump_weather_wind_speed_sensor`, `heatpump_weather_wind_direction_sensor`, `heatpump_weather_ghi_sensor`/`dni_sensor`/`dhi_sensor` - any left unset just falls back to a static default for that signal in the fit, matching how `rc-model-forecast` already treats a few of these.

A refit takes real time (~35s for a ~60-day window fit on the reference hardware this was validated on) - use a generous timeout on the triggering `rest_command` so it isn't cut off mid-fit:
```yaml
rest_command:
  rc_model_refit:
    url: http://127.0.0.1:5000/action/rc-model-refit
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
    timeout: 120
```
```yaml
- alias: EMHASS weekly rc-model refit
  trigger:
    platform: time
    at: '03:00:00'
  condition:
    condition: time
    weekday:
      - sun
  action:
  - service: rest_command.rc_model_refit
```
A refit that fits worse than `rc_model_refit_max_mae_c` (default 1.5°C) is logged as an error and discarded - the previously deployed parameters stay in place, so a bad refit (e.g. a sensor outage during the window) can't silently make `rc-model-forecast` worse.

## Heat pump electric/gas forecast (hybrid or pure-electric)

`hybrid-heatpump-model-refit` and `hybrid-heatpump-forecast` are a standalone pair of actions - same shape as `rc-model-refit`/`rc-model-forecast` above, for a different fitted model (`emhass.thermal.hybrid_heatpump_lr.HybridHeatPumpLR`) that predicts electric power (and gas consumption, for a hybrid system) instead of indoor temperature. **This is informational only and never influences dispatch** - EMHASS's optimizer has no gas/electric split decision to plug this into (the model needs the heat pump's duty as an input, which is exactly what the optimizer would otherwise be solving for), so it only publishes forecast sensors for your own dashboards/automations.

Works for both a hybrid system (electric heat pump + gas boiler) and a pure-electric heat pump - the mode is decided purely by whether `heatpump_gas_meter_sensor` is configured, not by `heatpump_is_hybrid` (that flag isn't read by this feature at all). Leave `heatpump_gas_meter_sensor` empty for a pure-electric refit: the gas model is skipped entirely (not fit on fabricated zero data), `hybrid-heatpump-forecast` then only publishes the electric sensor, and `hybrid_heatpump_refit_max_gas_mae_m3` is ignored.

`heatpump_gas_meter_sensor` can point at either a raw cumulative meter totalizer (the usual HA `state_class: total_increasing` convention - a lifetime running total that only ever rises) or an already-incremental per-interval consumption sensor - both the refit's training data and its live forecast seed auto-detect a cumulative reading (`utils.resolve_incremental_series`: a real per-interval delta fluctuates constantly as the burner cycles on/off, so a series that's almost always non-decreasing is assumed to be a raw totalizer instead) and convert it to a delta before use, clipping meter resets to zero rather than treating them as negative consumption.

`heatpump_power_sensor` gets the same auto-detection, but converts differently since it wants an average power in W (not a plain per-interval delta like gas): a detected cumulative reading is divided by the refit's own inferred timestep and scaled assuming the meter's cumulative unit is kWh (the near-universal HA convention for an energy totalizer) - so an already-instantaneous power sensor (the normal case) and a cumulative kWh energy meter both work without any extra configuration. A perfectly flat/constant reading is never treated as cumulative either way (a real totalizer still ticks up occasionally over a long window; something that never once increases is already a stable rate reading).

`hybrid-heatpump-model-refit` needs `hybrid_heatpump_refit_enabled: true`, `use_influxdb: true` (same rationale as the physics refit - the refit window is normally longer than the recorder's own retention), and, unlike the physics refit's mostly-optional sensor list, **hard-requires** at least one Rooms-tab `heatpump_room_temp_sensors` entry, `heatpump_power_sensor` and `heatpump_duty_sensor` to be configured (they're the fit targets/inputs this model is built around, not optional context) - plus `heatpump_gas_meter_sensor` too, for a hybrid (non-electric-only) refit. A refit is only deployed if its electric-power MAE (`hybrid_heatpump_refit_max_electric_mae_w`, default 150 W) - and, for a hybrid refit, its gas-consumption MAE (`hybrid_heatpump_refit_max_gas_mae_m3`, default 0.02 m³) too - measured on a held-out chronological slice of the refit window, clears the threshold(s); otherwise the previous model is left in place, same protective shape as `rc_model_refit_max_mae_c`.

```yaml
rest_command:
  hybrid_heatpump_model_refit:
    url: http://127.0.0.1:5000/action/hybrid-heatpump-model-refit
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
    timeout: 120
```

`hybrid-heatpump-forecast` needs `hybrid_heatpump_forecast_enabled: true` and a previously-deployed model (run the refit action at least once). It publishes `sensor.hybrid_heatpump_electric_forecast` (W), plus `sensor.hybrid_heatpump_gas_forecast` (m³) only when the deployed model was fit with a gas sensor configured. Known simplification: since EMHASS has no "planned duty" schedule to read for a generic thermal_battery load, the last observed `heatpump_duty_sensor` reading is held constant across the whole forecast horizon - treat this as an informational estimate, not a claim about what will actually run.

```yaml
rest_command:
  hybrid_heatpump_forecast:
    url: http://127.0.0.1:5000/action/hybrid-heatpump-forecast
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
```

## Multi-room ARX model (electric/gas + per-room temperature, learned coupling)

`arx-model-refit` and `arx-model-forecast` are a second, independent action pair alongside `hybrid-heatpump-model-refit`/`hybrid-heatpump-forecast` above - same "refit against InfluxDB history, then forecast" shape, but built around a different fitted model (`emhass.thermal.arx_model.ArxModel`) that is online-adaptive (Recursive Least Squares with a forgetting factor) and, unlike the hybrid model, predicts **each configured room's own temperature** in addition to whole-house electric power (and gas, for a hybrid system) - genuinely closed-loop, in that each forecast step's own predicted temperature feeds the next step's input, for every room at once. **Informational only by default** (electric/gas forecast, and every room's temperature forecast) - see "Inter-room thermal coupling" below for the coupling-only opt-in, and "Self-learning dispatch" further down for the one path where a room's fitted model genuinely drives real dispatch.

Works for both hybrid and pure-electric heat pumps, same convention as the model above: leave `heatpump_gas_meter_sensor` empty and the gas fit/forecast is skipped entirely.

`arx-model-refit` needs `arx_model_refit_enabled: true`, `use_influxdb: true`, `heatpump_power_sensor` and `heatpump_duty_sensor` configured (hard-required, same as the hybrid refit), and at least one room with a `heatpump_room_temp_sensors` entry - rooms with too little history are skipped individually rather than failing the whole refit. The whole-house model is only deployed if electric MAE (`arx_model_refit_max_electric_mae_w`, default 150 W) and gas MAE (`arx_model_refit_max_gas_mae_m3`, default 0.02 m³, hybrid only) clear their thresholds on a chronological holdout slice of `arx_model_refit_window_days` (default 60) - otherwise the previously deployed model is left in place entirely. `arx_model_forgetting_factor` (default 0.995) and `arx_model_ridge` (default 10.0) tune the RLS fit itself.

Room temperature has no absolute accuracy threshold - each room is judged relative to its own physics/simple-model alternative instead, see "Self-learning dispatch" below.

```yaml
rest_command:
  arx_model_refit:
    url: http://127.0.0.1:5000/action/arx-model-refit
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
    timeout: 120
```

`arx-model-forecast` needs `arx_model_forecast_enabled: true` and a previously-deployed model. It publishes a whole-house electric forecast sensor (plus gas, for a hybrid deployment) and one per-room temperature forecast sensor (`sensor.arx_model_temp_forecast_<room>`) for every room the deployed model covers. Like `hybrid-heatpump-forecast`, it reads the same dynamic, multi-room-aware aggregate duty trajectory (Part A - the combined dispatched power of every heat-pump-driven load in the latest solved plan, divided by `heatpump_nominal_power`) rather than holding the last observed duty constant.

```yaml
rest_command:
  arx_model_forecast:
    url: http://127.0.0.1:5000/action/arx-model-forecast
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
```

### One refit button instead of one per model

Most installations only ever turn on exactly one of the three thermal-model refits above (`rc-model-refit`, `hybrid-heatpump-model-refit`, `arx-model-refit`) - wiring up a separate automation/button per model is unnecessary busywork in that case, and it's just as unnecessary to know which of the three actions corresponds to whichever model you enabled. `thermal-models-refit` is a single consolidated action that checks all three `*_refit_enabled` flags and runs whichever are actually turned on, skipping the rest - one button/automation regardless of which model(s) you use, or all of them at once if more than one is enabled:

```yaml
rest_command:
  thermal_models_refit:
    url: http://127.0.0.1:5000/action/thermal-models-refit
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
```

The three individual actions remain available and unchanged for anyone who wants independent refit schedules per model (e.g. a faster cadence for `arx-model-refit` than `hybrid-heatpump-model-refit`).

Two symmetric consolidated actions exist alongside `thermal-models-refit`, same "one button regardless of which model(s) you use" shape:

- **`thermal-models-forecast`** checks all three `*_forecast_enabled` flags and runs whichever are turned on (`rc-model-forecast`, `hybrid-heatpump-forecast`, `arx-model-forecast`) - the predict-side sibling of `thermal-models-refit`.
- **`thermal-models-tune`** checks `arx_model_refit_enabled` and, if on, grid-searches the ARX model's `forgetting_factor`/`ridge` (25 candidates, scored on the same val split refit uses) and deploys the winner immediately - the only tunable thermal model today, since rc-model and hybrid-heatpump are direct fits with no hyperparameters to search. Tuning does not update `arx_model_forgetting_factor`/`arx_model_ridge` in config, so a later plain refit reverts to those config defaults unless you copy the winning values (returned in the result) into config yourself.

```yaml
rest_command:
  thermal_models_tune:
    url: http://127.0.0.1:5000/action/thermal-models-tune
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
  thermal_models_forecast:
    url: http://127.0.0.1:5000/action/thermal-models-forecast
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
```

These are also the three buttons shown on the dashboard's Advanced panel ("Fit thermal model(s)" / "Tune thermal model(s)" / "Predict thermal model(s)") - it no longer shows one button per individual model.

### Inter-room thermal coupling: manual vs. learned

If you've configured `heatpump_room_coupled_neighbors`/`heatpump_room_coupling_conductance` (manually-entered conductances that already affect real dispatch by warming/cooling one room's own thermal-battery constraint from its neighbors' temperatures), `arx_model_coupling_enabled: true` (default) additionally *learns* a conductance for the same room pairs from history, via a neighbor-temperature-difference feature in each room's own RLS fit. This learned value is:

- **Published/logged only by default** (`arx_model_coupling_source: "informational"`) - it never touches `heatpump_room_coupling_conductance` and never affects dispatch; your manually-entered value keeps doing exactly what it already does.
- **Opt-in to real dispatch** by setting `arx_model_coupling_source: "auto_dispatch"` - on the next optimization run, any room pair present in the freshly-refit `arx_model_coupling.json` overrides the manual conductance for that pair specifically; pairs the model hasn't learned yet (or that fail the refit's own quality gates) keep the manual value.

Be deliberate before opting in: a room pair held at a near-constant temperature difference (e.g. both rooms following the same static schedule) produces a statistically unreliable coefficient regardless of how much history you feed it - this is a real identifiability limitation of fitting a coupling term from data where the two rooms rarely decouple from each other, not just a tuning problem. Compare the learned value against your manually-entered one for a while (both are visible in the refit's own result) before switching a pair over to `auto_dispatch`.

### Candidate coupling suggestions (undeclared pairs)

The coupling fit above only ever considers pairs you've already declared via `heatpump_room_coupled_neighbors` - EMHASS doesn't invent room topology on its own. Every refit (when `arx_model_coupling_enabled` is on and you have more than one room) also runs a second, purely diagnostic pass that probes *every other* configured room as a candidate neighbor, regardless of whether you've declared it - so a real-looking but undeclared relationship at least gets surfaced instead of silently ignored.

This is informational only, one step further removed from dispatch than the `auto_dispatch` coupling above - a candidate is never added to `heatpump_room_coupled_neighbors` for you, however strong it looks:

- Candidates clearing a coarse noise floor (~0.02 kW/K - filters near-zero fit noise, not a statistical significance test) are logged and returned in the refit action's own result (`candidate_couplings`), and saved to `data/arx_model_coupling_candidates.json` for later inspection.
- **Take these with an extra grain of salt** compared to the already-cautious `auto_dispatch` path above: probing every other room at once as a candidate neighbor multiplies the same identifiability problem across more simultaneous regressors, so a candidate's magnitude is a rougher estimate than a declared pair's own learned coefficient.
- To actually test a candidate, add both rooms to each other's `heatpump_room_coupled_neighbors` yourself, with a manual `heatpump_room_coupling_conductance` placeholder (any positive number - see "Inter-room thermal coupling" above for why a pair needs a manual entry to become eligible at all) - only then can that pair either keep your manual value (informational, the default) or be refined by a real declared-pair fit under `auto_dispatch`.

### Self-learning dispatch: letting the fitted model drive a room, not just forecast it

Each room's tab in the config UI has a **"Self-learning only"** toggle (`heatpump_room_self_learning_only`). Turning it on for a room is a statement of intent - "let the fitted ARX model actually dispatch this room once it's proven itself" - but whether it actually does so is decided automatically, per room, on every `arx-model-refit`, never by the toggle alone:

- `arx-model-refit` fits the room's own equation as usual, then separately simulates what the room's own physics/simple thermal_battery model (the same open-loop RC recurrence the optimizer would otherwise use) would have predicted over that *same* holdout window and starting point.
- The room's fitted coefficients are written into the live dispatch artifact (`arx_model_room_dispatch_coefficients.json`) **only if** its measured MAE actually beats that physics-baseline MAE for that room - a hard, per-room requirement, not an informational suggestion. A room that doesn't clear this bar simply keeps dispatching on its ordinary physics/simple thermal_battery configuration until a later refit does clear it; nothing crashes and no manual intervention is needed either way.
- Once a room does clear the bar, the optimizer solves that room's comfort-bound/legionella/desired-temperature constraints directly against the fitted equation (room's own history-derived response to duty, weather, and any learned neighbor coupling) instead of the RC recurrence - "Room supply temperature" and any physics envelope fields for that room become informational only, since the fitted equation no longer uses them.
- The physics-baseline simulation always assumes the "simple" family (no ongoing envelope/solar heating demand) and a static supply temperature/COP, regardless of the room's actual configured model family - a deliberate v1 simplification, since a physics-family baseline would need weather inputs this refit's data pipeline doesn't currently pull. This makes the bar a bit easier to clear for physics-family rooms than a fully-faithful baseline would, but never wrong in the unsafe direction: it can only cause a room to switch to self-learning dispatch a little earlier, never dispatch on stale/unfit coefficients.
- Both figures are visible in the refit action's own result: `room_temp_physics_baseline_mae_c` (per room) and `rooms_using_self_learning_dispatch` (the rooms currently cleared for live dispatch).

### Sun-shading: blinds attenuate only the direct solar component

Two independent implementations share the same underlying rule - shading only ever reduces the *direct* solar component reaching a window, never the diffuse component arriving from the rest of the sky dome - but apply it differently depending on the room's model family:

- **Self-learning rooms** (`heatpump_room_self_learning_only`) never need a manually-specified blocking percentage or window orientation at all. A `blind_x_dni` interaction feature (`blind_position * dni`) lets each room's own RLS fit empirically discover how effective *that room's actual* blind is - a room whose blind barely cuts gain fits a small coefficient, a room with an efficient screen fits a large one, purely from history. Window orientation itself is already implicit in the model's existing `dni`/`dhi` coefficients (a room with more southward glass naturally shows a bigger fitted DNI response), so no azimuth field is needed anywhere in this path. A room with no configured blind sensor simply never gets a `blind_position` column - `blind_x_dni` defaults to 0 (always open, inert) rather than needing an explicit all-zero column.
- **Physics-family rooms** (static U-value/envelope formula) use an explicit per-room `heatpump_room_blind_type` (`none`/`screen`/`awning`) together with the live `heatpump_room_blind_sensors` position (0=open, 1=fully closed - see that field's own description before feeding it from a raw Home Assistant `cover.*` entity, whose native range/polarity is usually the opposite and will likely need a normalizing template sensor first). `screen` blocks a fixed fraction of the direct component regardless of sun angle, since it sits flush on the glass; `awning`/knikarmscherm only engages once the sun is high enough in the sky (little to no effect in mornings, evenings, or winter), since it projects outward above the window rather than covering it. Solar elevation for the awning formula is computed via `pvlib` from your configured `Latitude`/`Longitude`, not fetched from a weather API.

Both paths read the weather forecast's direct/diffuse decomposition (`dni`/`dhi`) rather than a single flat GHI figure - if your weather source has only ever supplied GHI historically, configure `heatpump_weather_dni_sensor`/`heatpump_weather_dhi_sensor` (refit) and/or check that your day-ahead weather provider returns `dni`/`dhi` (live dispatch) so shading has something real to act on; without them, shading has no measurable effect regardless of `blind_type`/`blind_x_dni`.

A room's live blind position only takes effect on a *fresh* optimization solve, not a warm-started cache-hit re-solve within the same MPC cycle - the same limitation the initial-temperature override (`def_init_temp`) already has, for the same reason.

### Open windows and doors: pause heating, add ventilation loss, and (doors only) boost neighbor coupling

`heatpump_room_window_sensors`/`heatpump_room_door_sensors` (one HA `binary_sensor` per room, open/closed) are live, current-moment-only signals - unlike blind position, a window/door's open state can't be forecast for future timesteps, so both only ever affect the *current* dispatch step, never later ones in the same solve:

- **A room's window OR door being open right now** pauses that room's heating for the current step (its heat-pump power is forced to zero) and adds a fixed extra ventilation/infiltration loss to the current step's heating-demand calculation (physics family), so the optimizer doesn't keep "fighting" an open opening. The room's comfort-temperature bound is automatically relaxed for that same step, so pausing heat input there can never make the solve infeasible. Self-learning rooms instead learn this effect empirically via an `opening_x_outdoor` interaction feature, fit from real historical window/door-open events - no fixed loss constant needed on that path.
- **A room's door being open right now**, specifically, *additionally* boosts its thermal-coupling conductance to any declared neighbor room(s) (`heatpump_room_coupled_neighbors`) for the current step - air mixes far more freely through an open doorway than a closed one. This has simply no effect for a room with no declared neighbors (nothing to boost), regardless of whether the door itself is open. Self-learning rooms get the equivalent as a `door_x_neighbor_diff::<neighbor>` interaction feature per declared neighbor.
- Both mechanisms refresh on every solve, including a warm-started cache-hit re-solve - **except** the door-coupling boost, which (like blind position) only takes effect on a fresh optimization solve, since the underlying coupling conductance is baked into the problem at build time rather than a per-solve-refreshable parameter.
- At forecast time (`compute_arx_model_forecast`), unlike blind position (held flat at its live reading for the whole horizon), these signals are deliberately never held flat - a momentary "is it open right now" reading has no valid forecast for future steps, so the whole published forecast horizon assumes closed.

### Sensorless detection: inferring "probably open" without a window/door sensor

`heatpump_room_window_sensors`/`heatpump_room_door_sensors` are optional - many real installations don't have a smart contact sensor on every window and door. A per-room Kalman filter (`emhass.thermal.opening_kalman_detector`) **always runs alongside** the sensor-based detection above, for every room, every `naive-mpc-optim` cycle - it compares the room's live observed temperature against a one-step prediction from that room's own existing thermal model (the physics/simple formula, or a fitted ARX model), and flags "probably open" when the mismatch ("innovation") is statistically large relative to that model's own noise level (a standard 3-sigma innovation gate). The two signals are OR'd - either one reporting "open" pauses heating and adds the extra ventilation loss (see above); a real sensor is never overridden or second-guessed, only ever agreed with or supplemented.

- **Only ever feeds the shared window/door signal, never the door-specific neighbor-coupling boost** - a single room's own temperature residual can't tell "my window is open to the cold outside" apart from "my door is open to a colder neighbor room" without also modelling the neighbor, so that stays sensor-only.
- **Needs `heatpump_power_sensor`/`heatpump_duty_sensor` configured** (live, real-time values - the same fields used elsewhere in this doc) - the filter's prediction step needs to know how much heat was actually delivered since the last cycle. Without either resolving, the detector cleanly no-ops that cycle rather than guessing.
- **Self-learning rooms get a real, per-room noise estimate** (`residual_std_c`, captured automatically during `arx-model-refit` from that room's own holdout residual spread) - physics/simple rooms use a fixed, conservative default, since there's no fitted history to calibrate from.
- Each room's filter state persists across dispatch cycles in `kalman_opening_detector_state.json` - a gap longer than 3 hours (restart, missed cycle, first-ever run) reseeds it from the live reading rather than trusting a stale belief, and never flags "open" on that reseed cycle.
- Like the sensor-based detection above, this is a `naive-mpc-optim`-only mechanism - `dayahead-optim` never fetches live HA data at all, so there is nothing for the filter to compare against there.

### Retroactive relabeling: turning sensorless detection into permanent ground truth

The sensorless Kalman detector above only ever looks *backward in time from now*, one dispatch cycle at a time - it can miss a real opening (its own uncertainty is still building) or flag a false one. At `arx-model-refit` time, a room's *entire* historical window is already available - past **and** future relative to any point - so a smoother (the same Kalman math, run backward as well as forward) can retroactively relabel "probably open" periods with noticeably more confidence, and feed that back into the room's own fitted model. Three independent opt-ins build on each other:

**1. `arx_model_opening_relabel_enabled` (default off)** - for every room with **neither** `heatpump_room_window_sensors` **nor** `heatpump_room_door_sensors` configured, each refit runs a small, fixed number of fit → smooth-residuals → relabel → refit passes (`arx_model_opening_relabel_iterations`, default 2 - not a convergence-detection loop, deliberately simple) against that room's own history, synthesizing an `opening_open` column exactly like a real sensor would have produced. A room with either sensor configured is **never** touched by this, for any timestamp, at any iteration - a real reading always wins, this only ever fills in where none exists.

- Only ever synthesizes the shared window/door signal (`opening_open`), same scope boundary as the live Kalman detector - a single room's residual still can't distinguish "my window is open" from "my door is open to a colder neighbor room" without jointly modelling the neighbor.
- The relabeled data feeds every downstream fit in that refit (holdout scoring, the deployed model, the candidate-coupling probe) - a room that clears the "Self-learning dispatch" bar afterward genuinely learned its `opening_x_outdoor` coefficient from these inferred events, the same as it would from a real sensor's history.

**2. Candidate opening events (always on once relabeling is)** - each refit collapses the relabeling pass's own contiguous "probably open" runs into events (`{start, end, n_steps, confidence}`), sorted by confidence and capped at 5 per room, returned in the refit result (`candidate_openings`) and saved to `data/arx_model_opening_candidates.json`. **Informational only, exactly like the candidate-coupling suggestions above** - nothing is applied automatically just because an event was surfaced.

**3. `arx_model_opening_confirm_enabled` (default off, requires relabeling itself enabled)** - closes the loop by asking *you*. Each refit:

- **First**, polls every room's confirmation `input_boolean` pair (see below) for a newly-answered pending question, and permanently records the answer to `data/arx_model_opening_confirmations.json`. A confirmed answer is treated as ground truth forever after: it's re-applied before *and* after every future relabeling pass, so the EM loop's own inference can never overwrite what you've confirmed - this is genuinely how the system "learns where the doors, windows, and blinds are" over time, rather than re-guessing from scratch every refit.
- **Last** (once this refit's own candidate events exist), publishes at most one new question per room - the single highest-confidence candidate that isn't already pending or already answered - as `sensor.room_opening_confirmation_<room>`, a plain text sensor asking e.g. *"Was room 'Attic' really open (window/door) between 14:00 and 15:30? Set input_boolean.attic_opening_answer to your answer, then input_boolean.attic_opening_ready to on, to confirm."*

This needs two per-room `input_boolean` helpers (mirroring the mechanism of the existing manual-load ready/confirm pair, applied to a different question): `heatpump_room_opening_confirm_answer_sensor` (your yes/no answer) and `heatpump_room_opening_confirm_ready_sensor` (flip to `on` once you've set the answer, to submit it). For example, in `configuration.yaml`:

```yaml
input_boolean:
  attic_opening_answer:
    name: Attic opening confirmation - answer (on = was open)
    icon: mdi:door-open
  attic_opening_ready:
    name: Attic opening confirmation - submit
    icon: mdi:check
```

Then, on the Attic room's own row in the config UI, set `heatpump_room_opening_confirm_answer_sensor` to `input_boolean.attic_opening_answer` and `heatpump_room_opening_confirm_ready_sensor` to `input_boolean.attic_opening_ready`. See `homeassistant_automations/room_opening_confirm_notify.yaml` for a ready-made actionable notification that sets both booleans for you from a single tap, instead of requiring the Home Assistant app.

**Refit-cadence, not dispatch-cadence** - unlike every other automation on this page, the confirmation loop only ever polls/publishes once per `arx-model-refit` call (nightly/weekly, typically), never once per `naive-mpc-optim` dispatch cycle. A confirmed answer only ever feeds a *future* refit; there is nothing for it to do in between.

### Sensorless blind-position estimation: a continuous 0-1 inference, not a flag

The same Kalman-filter idea extends to blind/shading position - but unlike a window/door's binary open/closed state, a blind's position is continuous (0=open..1=fully closed), and it's only ever *observable* when there's actually sun to be blocked or not: at night or under heavy cloud, closing or opening the blind makes literally zero difference to the room's temperature, so there is genuinely no information to infer from at those times - a physical fact about the problem, not a limitation of the filter. **ARX-model rooms only** - a physics-family room's own live one-step predictor has no solar term at all, so this kind of inference is structurally impossible there without first extending that predictor; physics-family rooms keep relying on `heatpump_room_blind_sensors`/`heatpump_room_blind_type` alone, unchanged.

- **`arx_model_blind_relabel_enabled`** (default off, refit-cadence): for every self-learning room with **no** `heatpump_room_blind_sensors` entry configured, each refit retroactively infers a continuous blind-position curve from that room's own sunny-period thermal history - the same fit → smooth-residuals → relabel → refit loop as opening detection, `arx_model_blind_relabel_iterations` times (default 3, one more than opening's own default: the first pass here is a genuinely weaker bootstrap, since there's no fitted `blind_x_dni` coefficient to invert against yet - see the source for the full derivation). A room with too little sunny history in the refit window (e.g. `heatpump_weather_dni_sensor` left unconfigured) is skipped for that refit rather than guessed at. Feeds back into that room's own `blind_x_dni` feature, same as a real sensor's history would.
- **Live, per-cycle estimation** (always on for a self-learning room that clears the above once its `blind_x_dni` coefficient is genuinely identified - no separate flag needed for the live half): every `naive-mpc-optim` cycle publishes `sensor.room_blind_position_estimate_<room>`, a plain 0-1 reading, purely informational by default - compare it against your own knowledge of when the blinds were actually up or down, for as long as you like, before trusting it with anything.
- **`arx_model_blind_estimate_source`** (`"informational"` default / `"auto_dispatch"` opt-in, same shape as `arx_model_coupling_source` above) - the rollout gate. **Unlike an open window, where pausing heat is always the safe response regardless of whether the window is really open, a wrong blind-position guess has no safe direction**: overestimating closure underheats a genuinely sunny room, underestimating overheats (wastes money on) a genuinely shaded one. This is why the default is informational-only, and why `auto_dispatch` additionally requires the live filter to have converged through several consecutive sunny, consistent readings before a room's estimate is ever allowed to override the static config fallback for a solve - a single lucky reading is never enough. A room's real `heatpump_room_blind_sensors` value, when configured, always takes precedence over any estimate regardless of this setting.

## Rate-aware setpoint tracking: follow the optimizer's planned pace, not just its target

If you're driving a room's heating with a fast local loop of your own (a PID on a mixing valve or TRV, for example), pointing that loop at a single static target can fight the optimizer's own intent. EMHASS may deliberately want a room to drift down slowly through an expensive price window and rise quickly once prices drop - a local loop that only ever sees "the target is 20°C" has no way to know it should currently be moving slowly rather than as fast as it can. EMHASS itself never commands the valve/PID directly (see the publish-only pattern throughout this page) - it publishes a plan; realizing that plan's *pace* physically is still your local loop's job, it just needs the right signal to follow.

EMHASS already publishes the optimizer's full planned trajectory, not just its current value: `sensor.temp_predicted{k}` (the entity behind the `predicted_temp_heater{k}` result column, published by `_publish_thermal_loads`) carries a `predicted_temperatures` attribute holding every future step of the solved plan, the same mechanism `sensor.indoor_temp_forecast` from `rc-model-forecast` above already uses. **This is the entity a rate-aware local controller should read** - not `sensor.room_target_temp_<name>` (published by `_publish_room_targets`), which is a static comfort-band ceiling (the top of that room's currently scheduled `max_temperatures`), not the optimizer's actual planned pace.

Example: blend toward the next scheduled step instead of jumping straight to it, so a local PID's setpoint starts moving ahead of the optimizer's own step boundary:

```yaml
automation:
- alias: Ramp local PID setpoint toward the optimizer's next step
  trigger:
  - minutes: /5
    platform: time_pattern
  action:
  - service: input_number.set_value
    target:
      entity_id: input_number.living_room_pid_setpoint
    data:
      value: >-
        {% set traj = state_attr('sensor.temp_predicted0', 'predicted_temperatures') %}
        {% if traj and traj | length > 1 %}
          {% set now_v = traj[0]['temp_predicted0'] | float %}
          {% set next_v = traj[1]['temp_predicted0'] | float %}
          {{ ((now_v + next_v) / 2) | round(2) }}
        {% else %}
          {{ states('sensor.temp_predicted0') | float }}
        {% endif %}
```

(`temp_predicted0` is deferrable-load index `0`'s predicted-temperature entity - adjust to whichever index your room actually is. Each `predicted_temperatures` list entry is `{"date": <ISO timestamp>, "temp_predicted0": <value>}`, ordered from the current step forward, so `traj[0]` is "now" and `traj[1]` is the next scheduled step.)

A further refinement - modeling the heat pump/TRV's own physical response speed *inside* the optimizer, so the published trajectory is inherently something the hardware can actually track step-for-step (rate-constrained MPC) - is a real idea for a future round, not implemented here.

## Manually-committed loads (washer/dishwasher with no smart-plug control)

`manual_load_enabled` handles appliances that can't be safely dispatched at all - a washing machine or dishwasher whose only remote control is a smart plug that measures power but can't switch it (cutting power resets the appliance's program), leaving a physical delay-start timer as the only way to schedule it. EMHASS can still compute *when* to start it, cost/solar-optimally, the same way it treats any other deferrable load - it just can't press the button, so it tells you what to set the timer to instead, and **that decision doesn't move once made**: unlike every other deferrable load, this one is deliberately never re-optimized after a start time has been chosen and shown to you.

This is not a separate section of loads - it's a per-load property on an *existing* entry in **Deferrable Loads**. Turn on `manual_load_enabled` (the section's master switch), then on that appliance's own load tab tick `is_manual_load`. It reuses that same tab's `load_names`/`nominal_power_of_deferrable_loads`/`operating_hours_of_each_deferrable_load` - nothing needs to be entered twice - plus:
- `manual_load_ready_sensor` - the entity ID of a Home Assistant `input_boolean` you flip on to say "I want to run this today". Create one per appliance, e.g.:
  ```yaml
  input_boolean:
    dishwasher_ready:
      name: Dishwasher ready to run
      icon: mdi:dishwasher
  ```
- `manual_load_deadline_hour` (optional) - a `"HH:MM"` latest-finish time for the day; leave empty to let EMHASS place it anywhere in the optimization horizon.
- `manual_load_confirm_power_sensor` (optional) - if your smart plug's power sensor is configured here, EMHASS uses it only to detect the appliance actually running (to clear the commitment automatically) - never to control it. Without one, EMHASS falls back to clearing the commitment once its window has elapsed (best-effort).
- `manual_load_program_select_sensor` (optional, see WashData below) - the entity ID of a `select` you set to the program you're about to run.

```yaml
load_names: ["dishwasher", "Wasmachine"]
is_manual_load: [false, true]
manual_load_ready_sensor: ["", "input_boolean.wasmachine_ready"]
```

`manual_load_ready_sensor`/`manual_load_confirm_power_sensor`/`manual_load_program_select_sensor`/`manual_load_deadline_hour` are indexed the same way as every other Deferrable Loads array field - one entry per load, only meaningful where `is_manual_load` is true for that index. They're always read via a direct Home Assistant REST state lookup, even if you have `use_influxdb: true` set - unlike the training-data pulls elsewhere in this fork, "what is this entity's value right now" is never routed through InfluxDB, so you don't need your InfluxDB integration to be recording `input_boolean`/`select` helper domains for this feature to work.

### WashData: real learned power profiles instead of a flat guess

"Being a washing machine" and "being manually dispatched" are independent properties - `load_washdata_device` works on *any* Deferrable Loads tab, whether or not `is_manual_load` is set, so an automatically-dispatched load (a load a smart plug can actually switch) can equally benefit from a real learned power shape instead of a hand-typed `load_programs` guess.

Set `load_washdata_device` to the device slug used by the [WashData](https://github.com/3dg1luk43/ha_washdata) `ha_washdata` custom integration for that appliance (e.g. `"wasmachine"`, matching its `sensor.wasmachine_profiel_<program>_aantal` / `binary_sensor.wasmachine_actief` entities). **Unlike most other per-load fields, this is read fresh on every optimization cycle, never frozen at config-save time**: EMHASS discovers every `sensor.wasmachine_profiel_<program>_aantal` entity WashData has learned so far (there's one per distinct program once it's been run a few times), and:
- With no learned program yet, it falls back to this load's flat `nominal_power_of_deferrable_loads`/`operating_hours_of_each_deferrable_load`, no error.
- With exactly one learned program, or several and no program pinned (see below), it uses whichever has the highest run count ("aantal") - the program you actually use most.
- If the load is also manually-committed and `manual_load_program_select_sensor` points at a `select` entity (WashData already provides one, e.g. `select.wasmachine_cyclusprogramma`) that's set to something other than `"auto_detect"`, EMHASS matches that option against the discovered programs and pins the plan to that exact one instead - only a manual load can know this in advance, since a human is physically choosing the program on the dial.

```yaml
load_names: ["dishwasher", "Wasmachine"]
is_manual_load: [false, true]
load_washdata_device: ["", "wasmachine"]
manual_load_ready_sensor: ["", "input_boolean.wasmachine_ready"]
manual_load_program_select_sensor: ["", "select.wasmachine_cyclusprogramma"]
```

Either way, it still only *advises* - you still choose the actual wash program on the machine's own dial.

Flow: flip the `input_boolean` on → the next `dayahead-optim` or `naive-mpc-optim` run picks an optimal start (using the WashData-learned profile shape when configured) and publishes `sensor.manual_load_action_<name>` with a human-readable instruction ("Set timer to 2h 15m", later "Start now") - see `homeassistant_automations/manual_load_notify.yaml` for a notification example. Every re-optimization after that keeps the exact same window; nothing you do (short of confirming the appliance ran, or the deadline passing) changes it. `sensor.p_<name>` (the same per-load power sensor every deferrable load gets) shows the planned power draw alongside the regular deferrable loads.