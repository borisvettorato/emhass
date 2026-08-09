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

## Heating-need forecast

`heating-need-forecast` is a report-only action (it never calls a device service) that simulates the indoor temperature forward assuming heating stays off, using the fitted thermal-mass physics model (see `scripts/thermal_mass_physics_model.py` - run it at least once to produce `data/thermal_physics_params.json` before enabling this). It publishes `sensor.indoor_temp_forecast` (the predicted curve, as a `predicted_temperatures` attribute) and `sensor.heating_needed_by` (the first timestamp the forecast crosses `heating_forecast_comfort_min_temp`, or `"beyond_horizon"`).

It needs `heating_forecast_enabled: true` in your config, and a weather forecast that actually reaches as far as `heating_forecast_horizon_hours` (default 72h). EMHASS's day-ahead weather window is controlled by `delta_forecast_daily`, which is read once when the Forecast object is built - **pass it explicitly in the request body** so it isn't left at the default 1-day window:
```yaml
rest_command:
  heating_need_forecast:
    url: http://127.0.0.1:5000/action/heating-need-forecast
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {"delta_forecast_daily": 3}
```
Keep the `3` here in sync with `heating_forecast_horizon_hours / 24` in your config - if they drift apart, EMHASS logs a warning (not an error) and simply forecasts as far as the data actually reaches.

In `automations.yaml`, trigger it a few times a day (there's no need to run it as often as `dayahead-optim` - the forecast only meaningfully changes as the weather forecast itself updates):
```yaml
- alias: EMHASS heating-need forecast
  trigger:
    platform: time_pattern
    hours: '/6'
  action:
  - service: rest_command.heating_need_forecast
```

Unlike the room/heat-pump/EV target sensors elsewhere in this fork, `sensor.indoor_temp_forecast` and `sensor.heating_needed_by` are purely informational - nothing in Home Assistant needs to *consume* them for this feature to be useful (you'd typically just look at the dashboard, or add your own notification automation on `sensor.heating_needed_by`), so no staleness watchdog is needed here: there's no device that could get stuck in a stale commanded state.

## Weekly model auto-refit

`heating-model-refit` periodically refits the thermal-mass physics model against fresh history and deploys it to `data/thermal_physics_params.json` - the same file `scripts/thermal_mass_physics_model.py --deploy-path` writes when you run it by hand, and the same file `heating-need-forecast` reads. Like every other EMHASS action, there's no scheduler inside EMHASS itself - you trigger it externally, same as `dayahead-optim`.

**Requires InfluxDB**, not just Home Assistant's own recorder: `heating_model_refit_window_days` (default 60) is normally far longer than the recorder's own retention (`purge_keep_days`, often 10 days by default) - see [InfluxDB as a data source](passing_data.md#influxdb-as-a-data-source) for why REST/WebSocket silently degrades to low-resolution stats beyond that window. Set `use_influxdb: true` plus `influxdb_host`/`influxdb_database`/etc. in your config, and add `influxdb_username`/`influxdb_password` to `secrets_emhass.yaml` (see the template file) - EMHASS routes every history pull through InfluxDB automatically once this is set, no other change needed.

You'll also need to point EMHASS at where each training signal actually lives in Home Assistant: `heatpump_indoor_temp_sensor` (required - the fit target) plus the optional `heatpump_power_sensor`, `heatpump_gas_meter_sensor`, `heatpump_flow_temp_sensor`, `heatpump_outdoor_temp_sensor`, `heatpump_duty_sensor`, `heatpump_weather_wind_speed_sensor`, `heatpump_weather_wind_direction_sensor`, `heatpump_weather_ghi_sensor`/`dni_sensor`/`dhi_sensor` - any left unset just falls back to a static default for that signal in the fit, matching how `heating-need-forecast` already treats a few of these.

A refit takes real time (~35s for a ~60-day window fit on the reference hardware this was validated on) - use a generous timeout on the triggering `rest_command` so it isn't cut off mid-fit:
```yaml
rest_command:
  heating_model_refit:
    url: http://127.0.0.1:5000/action/heating-model-refit
    method: POST
    headers:
      content-type: application/json
    payload: >-
      {}
    timeout: 120
```
```yaml
- alias: EMHASS weekly heating-model refit
  trigger:
    platform: time
    at: '03:00:00'
  condition:
    condition: time
    weekday:
      - sun
  action:
  - service: rest_command.heating_model_refit
```
A refit that fits worse than `heating_model_refit_max_mae_c` (default 1.5°C) is logged as an error and discarded - the previously deployed parameters stay in place, so a bad refit (e.g. a sensor outage during the window) can't silently make `heating-need-forecast` worse.

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