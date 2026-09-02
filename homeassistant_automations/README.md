# EMHASS → Home Assistant device automations

These automations are the "executor" half of the EMHASS "planner publishes,
Home Assistant executes" pattern used throughout this fork (see also the
upstream EMHASS `dhw_walkthrough.md` cookbook, and how EMHASS already
publishes `sensor.p_deferrable0` etc. for every ordinary deferrable load).

**EMHASS itself never calls a Home Assistant service.** It only publishes a
target-state sensor after each `dayahead-optim` / `naive-mpc-optim` /
`publish-data` run. Each automation here:

1. Watches one EMHASS-published target sensor.
2. Calls the real device's service to apply it.
3. Includes a **staleness watchdog**: if EMHASS hasn't updated the target
   sensor recently, the automation reverts the device to a safe default
   instead of leaving it stuck on the last command forever.

This keeps all "does this actually control my house" logic in Home
Assistant's own (mature, independently testable) automation engine, not in
EMHASS. If EMHASS has a bug or is offline, the *worst case* is "the device
falls back to its safe default a bit late" - never "EMHASS mis-commands a
device directly."

## Install

Copy the three files below into your Home Assistant `automations.yaml`
(or import each one individually via Settings → Automations → Edit in
YAML). Adjust the `entity_id`s if yours differ from what's listed - these
were confirmed against a real instance during development, but always
double-check against **Developer Tools → States** on your own system first.

## Tuning the watchdog interval

Every watchdog here polls every 5 minutes (`time_pattern: minutes: "/5"`)
and reverts if the EMHASS target sensor hasn't updated in **45 minutes**
(`2700` seconds). That default assumes EMHASS republishes at least every
~15-30 minutes (e.g. via `naive-mpc-optim` on a schedule, or
`continual_publish: true`). If your `continual_publish` interval or your
`dayahead-optim`/`naive-mpc-optim` cron schedule is longer, **raise the
threshold accordingly** - a watchdog interval shorter than your actual
publish cadence will trigger on every normal cycle. If you're not sure what
your interval is, check `optimization_time_step` and your automation/cron
that calls the EMHASS `/action/...` endpoints.

## Files

- `living_room_thermostat.yaml` - drives `climate.thermostaat_woonkamer` from
  `sensor.room_target_temp_living_room`. Watchdog fallback: 18°C setback
  (never "off" - the PID thermostat has its own internal safety, but a
  target it never received doesn't).
- `heatpump_dispatch.yaml` - drives `switch.climate_control` (the Daikin
  Altherma) from `sensor.heatpump_dispatch_target`. Watchdog fallback: turn
  the switch back **on**, handing control back to the heat pump's own
  native weather-compensated regulation - the safe direction for heating
  equipment is "resume normal operation," not "stay off."
- `ev_charger_myenergi.yaml` - drives the myenergi Zappi's
  `select.myenergi_zappi_charge_mode` / `select.myenergi_zappi_phase_setting`
  from `sensor.ev_charge_mode_target_zappi` / `sensor.ev_phase_target_zappi`.
  Watchdog fallback: **Stopped** - the safe direction for an EV charger is to
  stop drawing power, the opposite of the heat pump.

- `manual_load_notify.yaml` - notifies about
  `sensor.manual_load_action_<name>` (manual_load_enabled), the
  human-readable timer instruction for appliances with no smart-plug control
  (washing machine/dishwasher). Not a device-executor, no watchdog - see
  below.
- `room_opening_confirm_notify.yaml` - actionable (tap Yes/No) notification
  for `sensor.room_opening_confirmation_<room>`
  (`arx_model_opening_confirm_enabled`), the retroactive
  opening-detection confirmation loop - see "Retroactive relabeling" in
  `docs/automations.md`. Your tap sets the two per-room confirmation
  `input_boolean`s EMHASS is already polling on its own refit-cadence
  schedule. Not a device-executor, no watchdog - same reasoning as
  `manual_load_notify.yaml` below.

**Not listed here on purpose:** the `rc-model-forecast` action (see
`docs/automations.md`) publishes `sensor.indoor_temp_forecast` /
`sensor.heating_needed_by`, but doesn't control any device - it's an
informational forecast, not a target sensor for an executor automation. No
watchdog applies: there's no commanded state to get stuck in. The same is
true for `manual_load_notify.yaml`'s `sensor.manual_load_action_<name>` and
`room_opening_confirm_notify.yaml`'s `sensor.room_opening_confirmation_<room>`
- EMHASS never controls these appliances/rooms directly, it only tells you
what to do (manual load) or asks you a question (opening confirmation).
