# EMHASS-experimental

Shadow-test build of EMHASS from the `experimental-changes` branch
(`https://github.com/borisvettorato/emhass/tree/experimental-changes`).
Built and published by `.github/workflows/publish_docker-experimental.yaml`
on every push to that branch (or a manual "Run workflow" dispatch), tagged
`ghcr.io/borisvettorato/emhass:experimental`.

This is a separate add-on (slug `emhass-experimental`, its own image tag)
from your existing `emhass` (stable) and `emhass-test` (official upstream
test) add-ons - installing/running it does not touch or overwrite either.
Home Assistant also keeps each add-on's `/config` storage isolated per
slug, so this add-on's `config.json`/`data/` never collide with your
production add-on's.

## Shadow-testing checklist

- Configure this add-on with your real `hass_url`/`long_lived_token`/etc.
  (same values as your production add-on) so it sees live data.
- Set up the periodic optimization triggers (`dayahead-optim`,
  `naive-mpc-optim`, `publish-data` - see `docs/automations.md`) pointing
  at **this add-on's** port/URL, not your production one.
- Do **not** install the executor automations
  (`homeassistant_automations/heatpump_dispatch.yaml`,
  `ev_charger_myenergi.yaml`) yet - those are the only things that ever
  call a real Home Assistant service. Everything else this add-on
  publishes is purely informational until an executor automation acts
  on it.
- Watch the published sensors on a dashboard for a while before deciding
  whether to promote `experimental-changes` into your stable/test channel.
