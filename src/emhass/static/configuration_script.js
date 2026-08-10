//javascript file for dynamically processing configuration page

//used static files
//param_definitions.json : stores information about parameters (E.g. their defaults, their type, and what parameter section to be in)
//configuration_list.html : template html to act as a base for the list view. (Params get dynamically added after)

//Div layout
/* <div configuration-container>
  <div class="section-card">
    <div class="section-card-header"> POSSIBLE HEADER INPUT HERE WITH PARAMETER ID</div>
    <div class="section-body">
      <div id="PARAMETER-NAME" class="param">
        <div class="param-input">input/s here</div>
      </div>
    </div>
  </div>
</div>; */

//#610: the 15 per-battery array params in the "Battery" section (plant_conf +
//optim_conf, see utils.py BATT_ARRAY_PARAMS_PLANT_CONF/BATT_ARRAY_PARAMS_OPTIM_CONF/
//BATT_WEIGHT_PARAMS). Kept as an explicit list (not "every array.* param in the
//Battery section") because number_of_batteries itself is int (not array.*), so a
//generic filter would already exclude it - this list exists so the
//"number_of_batteries" header case below only grows/shrinks these, never the
//section's other array.* siblings if any get added later.
const BATTERY_ARRAY_PARAMS = [
  "weight_battery_discharge",
  "weight_battery_charge",
  "battery_discharge_power_max",
  "battery_charge_power_max",
  "battery_discharge_efficiency",
  "battery_charge_efficiency",
  "battery_nominal_energy_capacity",
  "battery_minimum_state_of_charge",
  "battery_maximum_state_of_charge",
  "battery_target_state_of_charge",
  "battery_stress_cost",
  "battery_soc_deficit_threshold",
  "battery_soc_deficit_cost",
  "battery_soc_surplus_threshold",
  "battery_soc_surplus_cost",
];

//on page reload
window.onload = async function () {
  ///fetch configuration parameters from definitions json file
  let param_definitions = await getParamDefinitions();
  //obtain configuration from emhass (pull)
  let config = await obtainConfig();
  //obtain configuration_list.html html as a template to dynamically to render parameters in a list view (parameters as input items)
  let list_html = await getListHTML();
  //load list parameter page (default)
  loadConfigurationListView(param_definitions, config, list_html);

  //add event listener to save button
  document
    .getElementById("save")
    .addEventListener("click", () => saveConfiguration(param_definitions));

  //add event listener to yaml button (convert yaml to json in box view)
  document.getElementById("yaml").addEventListener("click", () => yamlToJson());
  //hide yaml button by default (display in box view)
  document.getElementById("yaml").style.display = "none";

  //add event listener to defaults button
  document
    .getElementById("defaults")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, true)
    );

  //add event listener to json-toggle button (toggle between json box and list view)
  document
    .getElementById("json-toggle")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, false)
    );
};

//obtain file containing information about parameters (definitions)
// Cache-busting query string (bump whenever param_definitions.json content
// changes) - this is a plain fetch() with no <script src> tag to carry a
// version param, so the browser's HTTP cache would otherwise keep serving a
// stale copy indefinitely after an update (#WashData toggle/dropdown fields
// silently missing on an existing install was exactly this bug).
async function getParamDefinitions() {
  const response = await fetch(`static/data/param_definitions.json?version=6`);
  if (response.status !== 200 && response.status !== 201) {
    //alert error in alert box
    errorAlert("Unable to obtain definitions file");
    return {};
  }
  const param_definitions = await response.json();
  return await param_definitions;
}

//obtain emhass config (from saved params extracted/simplified into the config format)
async function obtainConfig() {
  config = {};
  const response = await fetch(`get-config`, {
    method: "GET",
  });
  let  response_status = response.status; //return status
  //if request failed
  if (response_status !== 200 && response_status !== 201) {
    showChangeStatus(response_status, await response.json());
    return {};
  }
  //else extract json rom data
  let blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  showChangeStatus(response_status, {});
  return config;
}

//obtain emhass default config (to present the default parameters in view)
async function ObtainDefaultConfig() {
  config = {};
  const response = await fetch(`get-config/defaults`, {
    method: "GET",
  });
  //if request failed
  let response_status = response.status; //return status
  if (response_status !== 200 && response_status !== 201) {
    showChangeStatus(response_status, await response.json());
    return {};
  }
  //else extract json rom data
  let blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  showChangeStatus(response_status, {});
  return config;
}

//get html data from configuration_list.html (list template)
async function getListHTML() {
  const response = await fetch(`static/configuration_list.html`);
  if (response.status !== 200 && response.status !== 201) {
    errorAlert("Unable to obtain configuration_list.html file");
    return {};
  }
  let blob = await response.blob(); //get data blob
  let htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  return htmlTemplateData;
}

function normalizeIndexedNames(countParamId, namesParamId, prefix, zeroPad = false) {
  const countElement = document.getElementById(countParamId);
  const namesElement = document.getElementById(namesParamId);
  if (!countElement || !namesElement) return;

  const count = Math.max(1, Number.parseInt(countElement.value || "1"));
  const nameInputs = Array.from(namesElement.querySelectorAll(".param_input"));
  const autoNamePattern = new RegExp(`^${prefix}_[0-9]+$`);

  for (let i = 0; i < Math.min(count, nameInputs.length); i++) {
    const input = nameInputs[i];
    const current = (input.value || "").trim();
    if (!current || autoNamePattern.test(current)) {
      const suffix = zeroPad ? String(i + 1).padStart(2, "0") : String(i + 1);
      input.value = `${prefix}_${suffix}`;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }
}

function applyLoadTypeVisibility() {
  const loadSection = document.getElementById("Deferrable Loads");
  if (!loadSection) return;
  const loadBody = loadSection.querySelector(".section-body");
  if (!loadBody) return;

  const activeIndex = Number.parseInt(loadBody.dataset.activeIndex || "0");
  const loadTypeDiv = document.getElementById("load_type");
  const loadTypeSelect = loadTypeDiv
    ? loadTypeDiv.querySelectorAll("select")[activeIndex]
    : null;
  const loadType = loadTypeSelect ? loadTypeSelect.value : "program_based";

  // dispatch_mode is only user-configurable for non-program types
  // (program_based always forces dispatch_mode="program" in the backend)
  const isProgramBased = loadType === "program_based";

  const alwaysVisible = [
    "load_names",
    "start_timesteps_of_each_deferrable_load",
    "end_timesteps_of_each_deferrable_load",
    "load_type"
  ];
  // set_deferrable_max_startups/def_minimum_on_time/def_minimum_off_time/
  // treat_deferrable_load_as_semi_cont/set_deferrable_load_single_constant
  // are real solver constraints only in optimization.py's non-sequence-load
  // branch (see _add_deferrable_load_constraints) - program_based loads are
  // shaped by convolution instead and never reach that code path (the
  // single-constant block is even explicitly gated on "not is_sequence_load"),
  // so these fields are silent no-ops for program_based and shouldn't be shown.
  const nonProgramExtras = [
    "set_deferrable_max_startups",
    "def_minimum_on_time",
    "def_minimum_off_time",
    "treat_deferrable_load_as_semi_cont",
    "set_deferrable_load_single_constant"
  ];
  const visibleByType = {
    program_based: [
      "load_programs",
      "load_washdata_enabled",
      "load_washdata_device"
    ],
    fixed_power_splittable: [
      "load_dispatch_mode",
      "nominal_power_of_deferrable_loads",
      "operating_hours_of_each_deferrable_load",
      "set_deferrable_startup_penalty",
      ...nonProgramExtras
    ],
    fixed_power_non_splittable: [
      "load_dispatch_mode",
      "nominal_power_of_deferrable_loads",
      "operating_hours_of_each_deferrable_load",
      ...nonProgramExtras
    ],
    variable_power_variable_time: [
      "load_dispatch_mode",
      "nominal_power_of_deferrable_loads",
      "minimum_power_of_deferrable_loads",
      "operating_hours_of_each_deferrable_load",
      ...nonProgramExtras
    ]
  };

  const managedParams = new Set([
    ...alwaysVisible,
    "load_dispatch_mode",
    ...Object.values(visibleByType).flat()
  ]);
  const visibleNow = new Set([
    ...alwaysVisible,
    ...(visibleByType[loadType] || visibleByType.program_based)
  ]);

  managedParams.forEach((id) => {
    const div = document.getElementById(id);
    if (!div) return;
    div.style.display = visibleNow.has(id) ? "" : "none";
  });

  const dispatchModeDiv = document.getElementById("load_dispatch_mode");
  const dispatchSelect = dispatchModeDiv
    ? dispatchModeDiv.querySelectorAll("select")[activeIndex]
    : null;
  const dispatchMode = isProgramBased ? "program" : (dispatchSelect ? dispatchSelect.value : "hours");
  const energyTargetDiv = document.getElementById("required_energy_kwh_of_each_deferrable_load");
  if (energyTargetDiv) {
    energyTargetDiv.style.display = dispatchMode === "energy_kwh" ? "" : "none";
  }
}

// Mirrors applyLoadTypeVisibility(): shows the ready/confirm/profile/deadline
// sensor fields on a Deferrable Loads tab only when that tab's
// "Manually-committed load" (is_manual_load) checkbox is on.
function applyManualLoadVisibility() {
  const loadSection = document.getElementById("Deferrable Loads");
  if (!loadSection) return;
  const loadBody = loadSection.querySelector(".section-body");
  if (!loadBody) return;

  // "Manually-committed load" is itself only meaningful once the section-level
  // "Enable manually-committed loads" header toggle is on.
  const manualLoadEnabledInput = document.getElementById("manual_load_enabled");
  const manualLoadEnabled = manualLoadEnabledInput ? manualLoadEnabledInput.checked : false;

  const manualDiv = document.getElementById("is_manual_load");
  if (manualDiv) manualDiv.style.display = manualLoadEnabled ? "" : "none";

  const activeIndex = Number.parseInt(loadBody.dataset.activeIndex || "0");
  const manualCheckbox = manualDiv
    ? manualDiv.querySelectorAll("input[type='checkbox']")[activeIndex]
    : null;
  const isManual = manualLoadEnabled && manualCheckbox ? manualCheckbox.checked : false;

  const manualFields = [
    "manual_load_ready_sensor",
    "manual_load_confirm_power_sensor",
    "manual_load_program_select_sensor",
    "manual_load_deadline_hour",
  ];
  manualFields.forEach((id) => {
    const div = document.getElementById(id);
    if (!div) return;
    div.style.display = isManual ? "" : "none";
  });
}

// Mirrors applyManualLoadVisibility(): the device picker itself only shows
// when the "Use a WashData device" (load_washdata_enabled) checkbox on this
// tab is on - applyLoadTypeVisibility() already hides both for non-program
// load types, this only handles the checkbox layered on top of that.
// When actually linked to a device, EMHASS resolves that load's power
// shape/duration live from WashData every cycle (see
// command_line._resolve_load_profiles), so the fields for hand-specifying
// them become moot and are hidden too. Only ever hides (never re-shows) the
// managed fields below: with the checkbox off this leaves whatever
// applyLoadTypeVisibility already decided untouched.
function applyWashdataVisibility() {
  const loadSection = document.getElementById("Deferrable Loads");
  if (!loadSection) return;
  const loadBody = loadSection.querySelector(".section-body");
  if (!loadBody) return;

  const activeIndex = Number.parseInt(loadBody.dataset.activeIndex || "0");
  const enabledDiv = document.getElementById("load_washdata_enabled");
  const enabledCheckbox = enabledDiv
    ? enabledDiv.querySelectorAll("input[type='checkbox']")[activeIndex]
    : null;
  const washdataEnabled = enabledCheckbox ? enabledCheckbox.checked : false;

  const deviceDiv = document.getElementById("load_washdata_device");
  if (deviceDiv) deviceDiv.style.display = washdataEnabled ? "" : "none";
  if (!washdataEnabled) return;

  const washdataManagedFields = [
    "nominal_power_of_deferrable_loads",
    "operating_hours_of_each_deferrable_load",
    "load_dispatch_mode",
    "load_programs",
    "minimum_power_of_deferrable_loads",
    "required_energy_kwh_of_each_deferrable_load",
    "set_deferrable_startup_penalty",
    "set_deferrable_max_startups",
    "def_minimum_on_time",
    "def_minimum_off_time",
  ];
  washdataManagedFields.forEach((id) => {
    const div = document.getElementById(id);
    if (div) div.style.display = "none";
  });
}

// Fetches the WashData devices discovered on the connected HA instance
// (see /get-washdata-devices) and rebuilds every load_washdata_device
// <select>'s options from that live list - so the user picks a real,
// currently-detected appliance instead of typing a slug by hand. Each
// select's previously-saved value is preserved as its own option even if
// it's no longer in the live list (HA unreachable when the page loaded, or
// the device was renamed) so a valid saved config is never silently wiped.
//
// savedConfig (optional) is the just-loaded config object: on first render,
// buildParamElement() only knew about the placeholder select_options: [""],
// so a real saved device slug that doesn't match that one static option
// gets silently collapsed to "" in the freshly-built DOM before this
// function ever runs - reading select.value alone would already have lost
// it. Falls back to the live select.value for any index savedConfig
// doesn't cover (newly-added load tabs, or later calls after the user has
// already interacted with the page).
async function attachWashdataDeviceSuggestions(savedConfig) {
  const div = document.getElementById("load_washdata_device");
  if (!div) return;
  const selects = div.querySelectorAll("select.param_input");
  if (selects.length === 0) return;

  const savedValues = Array.isArray(savedConfig?.load_washdata_device)
    ? savedConfig.load_washdata_device
    : [];

  let devices = [];
  try {
    const response = await fetch("get-washdata-devices");
    if (response.ok) {
      devices = await response.json();
    }
  } catch (e) {
    console.warn("Could not fetch WashData devices for suggestions", e);
  }

  selects.forEach((select, index) => {
    // Live DOM value wins whenever it's set (never clobber an unsaved edit
    // the user just made); savedConfig is only a fallback for the "freshly
    // rendered, not-yet-populated select" case described above.
    const currentValue = (select.value || savedValues[index] || "").trim();
    const optionValues = new Set(["", ...devices]);
    if (currentValue) optionValues.add(currentValue);

    select.innerHTML = "";
    optionValues.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value || "— select a device —";
      select.appendChild(option);
    });
    select.value = currentValue && optionValues.has(currentValue) ? currentValue : "";
  });
}

// Per-field filter used by attachEntitySuggestions() below: domain is the
// entity_id prefix(es) that make sense for that field (e.g. "sensor",
// "binary_sensor", "switch", "select", "input_boolean"), deviceClass/unit
// (when given) narrow further by HA's device_class/unit_of_measurement
// attributes - e.g. only offering entities showing degrees for a
// temperature field. When either is specified, matching one of them is
// required (not just "not actively wrong"): a real HA instance has plenty
// of sensor-domain entities with neither device_class nor unit_of_measurement
// set (template sensors, some integrations), and letting all of those ride
// along on domain alone defeated the point of filtering - it looked like
// "every sensor" instead of just the relevant ones. Fields with no
// deviceClass/unit at all (e.g. a select-entity field) stay domain-only, and
// attachEntitySuggestions() still falls back to every entity of the right
// domain if the strict filter excludes everything for a given field, so this
// never produces an empty, useless list. Covers every "(HA entity)"-style
// field across the config UI, so one fetch of /get-ha-entities can suggest
// for all of them.
const DEG = ["°C", "°F"];
const ENTITY_SUGGESTION_FILTERS = {
  // Local
  sensor_power_photovoltaics: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },
  sensor_power_photovoltaics_forecast: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },
  sensor_power_battery: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },
  sensor_battery_state_of_charge: { domain: ["sensor"], deviceClass: ["battery"], unit: ["%"] },
  sensor_power_load_no_var_loads: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },
  sensor_replace_zero: { domain: ["sensor"] },
  sensor_linear_interp: { domain: ["sensor"] },

  // Deferrable Loads
  manual_load_ready_sensor: { domain: ["input_boolean", "binary_sensor"] },
  manual_load_program_select_sensor: { domain: ["select", "input_select"] },
  manual_load_confirm_power_sensor: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },

  // Heat Pump
  heatpump_indoor_temp_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  heatpump_dispatch_control_entity: { domain: ["switch", "input_boolean"] },
  heatpump_cop_source_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  heatpump_weather_ghi_sensor: { domain: ["sensor"], deviceClass: ["irradiance"], unit: ["W/m²"] },
  heatpump_weather_dni_sensor: { domain: ["sensor"], deviceClass: ["irradiance"], unit: ["W/m²"] },
  heatpump_weather_dhi_sensor: { domain: ["sensor"], deviceClass: ["irradiance"], unit: ["W/m²"] },
  heatpump_weather_wind_speed_sensor: { domain: ["sensor"], deviceClass: ["wind_speed"], unit: ["m/s", "km/h", "mph", "kn"] },
  heatpump_weather_wind_direction_sensor: { domain: ["sensor"], deviceClass: ["wind_direction"], unit: ["°"] },
  heatpump_weather_humidity_sensor: { domain: ["sensor"], deviceClass: ["humidity"], unit: ["%"] },
  heatpump_power_sensor: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },
  heatpump_flow_temp_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  heatpump_gas_meter_sensor: { domain: ["sensor"], deviceClass: ["gas"] },
  heatpump_outdoor_temp_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  heatpump_duty_sensor: { domain: ["binary_sensor", "sensor"] },
  heatpump_target_temp_sensor: { domain: ["sensor", "number"], deviceClass: ["temperature"], unit: DEG },

  // Rooms
  heatpump_room_temp_sensors: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  heatpump_room_valve_sensors: { domain: ["sensor", "number"], unit: ["%"] },
  heatpump_room_blind_sensors: { domain: ["sensor", "cover"], unit: ["%"] },
  heatpump_room_window_sensors: { domain: ["binary_sensor"], deviceClass: ["window"] },
  heatpump_room_door_sensors: { domain: ["binary_sensor"], deviceClass: ["door"] },

  // Boiler
  boiler_temp_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  boiler_target_temp_sensor: { domain: ["sensor"], deviceClass: ["temperature"], unit: DEG },
  boiler_power_sensor: { domain: ["sensor"], deviceClass: ["power"], unit: ["W", "kW"] },

  // EV Charging
  ev_charge_mode_service: { domain: ["select", "input_select"] },
  ev_phase_select_entity: { domain: ["select", "input_select"] },
};

// True if entity is plausibly right for filter: domain must match, then
// device_class/unit are OR'd soft-positive signals - see comment above.
function entityMatchesFilter(entity, filter) {
  const domain = (entity.entity_id || "").split(".")[0];
  if (!filter.domain.includes(domain)) return false;

  const hasDeviceClassFilter = Array.isArray(filter.deviceClass);
  const hasUnitFilter = Array.isArray(filter.unit);
  if (!hasDeviceClassFilter && !hasUnitFilter) return true;

  const deviceClassMatches = hasDeviceClassFilter && !!entity.device_class && filter.deviceClass.includes(entity.device_class);
  const unitMatches = hasUnitFilter && !!entity.unit_of_measurement && filter.unit.includes(entity.unit_of_measurement);

  // A device_class/unit filter is specified for this field, so require an
  // actual match on one of them - an entity with neither attribute set no
  // longer gets the benefit of the doubt here (that used to let every
  // metadata-less sensor in a HA instance ride along with the real
  // temperature sensors, drowning out the filtering). attachEntitySuggestions()
  // still falls back to every entity of the right domain if this ends up
  // excluding everything for a given field, so a too-strict guess still
  // degrades to a useful list rather than an empty one.
  return deviceClassMatches || unitMatches;
}

// Fetches every HA entity once (see /get-ha-entities) and, for every field
// in ENTITY_SUGGESTION_FILTERS that's actually rendered on the page, offers
// matching entity_ids as native <datalist> suggestions on its text input(s)
// - a "pick from a list" feel while keeping free text as the fallback
// (unusual device_class/unit, custom/template entity, or HA unreachable
// when the page loaded). Deliberately kept as free-text + datalist rather
// than a <select> (like WashData's device picker): unlike a WashData
// device, which must match a real detected device exactly, any of these
// fields may legitimately need an entity the filter doesn't anticipate.
async function attachEntitySuggestions() {
  const fieldIds = Object.keys(ENTITY_SUGGESTION_FILTERS).filter((id) =>
    document.getElementById(id)
  );
  if (fieldIds.length === 0) return;

  let entities = [];
  try {
    const response = await fetch("get-ha-entities");
    if (response.ok) {
      entities = await response.json();
    }
  } catch (e) {
    console.warn("Could not fetch Home Assistant entities for suggestions", e);
    return;
  }
  if (!entities.length) return;

  fieldIds.forEach((fieldId) => {
    const filter = ENTITY_SUGGESTION_FILTERS[fieldId];
    const div = document.getElementById(fieldId);
    if (!div) return;
    const inputs = div.querySelectorAll("input.param_input[type='text']");
    if (inputs.length === 0) return;

    const matches = entities.filter((entity) => entityMatchesFilter(entity, filter));
    // Fall back to every entity of the right domain if the device_class/unit
    // filter matched nothing at all (a stricter guess than this HA
    // instance's actual entities support) - better a broad, useful list
    // than an empty one.
    const candidates = matches.length
      ? matches
      : entities.filter((entity) => filter.domain.includes((entity.entity_id || "").split(".")[0]));

    const listId = `entity-suggestions-${fieldId}`;
    let datalist = document.getElementById(listId);
    if (!datalist) {
      datalist = document.createElement("datalist");
      datalist.id = listId;
      document.body.appendChild(datalist);
    }
    datalist.innerHTML = "";
    candidates.forEach((entity) => {
      const option = document.createElement("option");
      option.value = entity.entity_id;
      if (entity.friendly_name) option.label = entity.friendly_name;
      datalist.appendChild(option);
    });

    inputs.forEach((input) => {
      input.setAttribute("list", listId);
      if (!input.value && candidates.length) {
        input.setAttribute("placeholder", `e.g. ${candidates[0].entity_id}`);
      }
    });
  });
}

function setupLoadProgramTabs() {
  const loadSection = document.getElementById("Deferrable Loads");
  if (!loadSection) return;
  const loadBody = loadSection.querySelector(".section-body");
  if (!loadBody) return;

  const activeIndex = Number.parseInt(loadBody.dataset.activeIndex || "0");
  const loadTypeDiv = document.getElementById("load_type");
  const loadTypeSelect = loadTypeDiv
    ? loadTypeDiv.querySelectorAll("select")[activeIndex]
    : null;
  const loadType = loadTypeSelect ? loadTypeSelect.value : "program_based";

  const programsDiv = document.getElementById("load_programs");
  if (!programsDiv) return;

  const oldEditor = programsDiv.querySelector(".load-programs-editor");
  if (oldEditor) oldEditor.remove();

  if (loadType !== "program_based") return;

  const programInputs = Array.from(programsDiv.querySelectorAll(".param_input"));
  if (!programInputs.length || !programInputs[activeIndex]) return;

  const sourceInput = programInputs[activeIndex];
  sourceInput.style.display = "none";

  const parsePrograms = (raw) => {
    try {
      const parsed = JSON.parse(raw || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.map((item, idx) => ({
        name: typeof item?.name === "string" && item.name.trim() ? item.name : `program_${idx + 1}`,
        power_pattern: typeof item?.power_pattern === "string" ? item.power_pattern : ""
      }));
    } catch {
      return [];
    }
  };

  let programs = parsePrograms(sourceInput.value);
  if (!programs.length) {
    programs = [{ name: "program_1", power_pattern: "" }];
  }

  const editor = document.createElement("div");
  editor.className = "load-programs-editor";
  programsDiv.appendChild(editor);

  const serialize = () => {
    sourceInput.value = JSON.stringify(programs);
    sourceInput.dispatchEvent(new Event("input", { bubbles: true }));
  };

  const render = (activeProgramIndex = 0) => {
    editor.innerHTML = "";

    // --- Header row (styled like section-card-header) ---
    const headerRow = document.createElement("div");
    headerRow.className = "load-program-header-row";

    const headerLabel = document.createElement("span");
    headerLabel.className = "load-program-header-label";
    headerLabel.textContent = "Programs";
    headerRow.appendChild(headerLabel);

    const countInput = document.createElement("input");
    countInput.type = "number";
    countInput.min = "1";
    countInput.value = String(programs.length);
    countInput.className = "load-program-count-input";
    countInput.addEventListener("change", () => {
      const newCount = Math.max(1, Number.parseInt(countInput.value) || 1);
      countInput.value = String(newCount);
      while (programs.length < newCount) {
        programs.push({ name: `program_${programs.length + 1}`, power_pattern: "" });
      }
      while (programs.length > newCount) {
        programs.pop();
      }
      serialize();
      render(Math.min(activeProgramIndex, programs.length - 1));
    });
    headerRow.appendChild(countInput);
    editor.appendChild(headerRow);

    // --- Tab bar (like subtabs-bar indexed-tabs) ---
    const tabs = document.createElement("div");
    tabs.className = "subtabs-bar load-program-tabs";
    programs.forEach((program, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "subtab-btn load-program-tab-btn";
      if (idx === activeProgramIndex) btn.classList.add("active");
      btn.textContent = program.name || `program_${idx + 1}`;
      btn.addEventListener("click", () => render(idx));
      tabs.appendChild(btn);
    });
    editor.appendChild(tabs);

    // --- Fields for active program ---
    const active = programs[activeProgramIndex];

    const nameLabel = document.createElement("label");
    nameLabel.className = "load-program-field-label";
    nameLabel.textContent = "Program name";
    editor.appendChild(nameLabel);

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "param_input load-program-text-input";
    nameInput.value = active.name;
    nameInput.addEventListener("input", () => {
      active.name = nameInput.value;
      // Update tab label live without full re-render
      const tabBtns = Array.from(tabs.querySelectorAll(".load-program-tab-btn"));
      if (tabBtns[activeProgramIndex]) {
        tabBtns[activeProgramIndex].textContent = nameInput.value || `program_${activeProgramIndex + 1}`;
      }
      serialize();
    });
    editor.appendChild(nameInput);

    const patternLabel = document.createElement("label");
    patternLabel.className = "load-program-field-label";
    patternLabel.textContent = "Power pattern per timestep (W)";
    editor.appendChild(patternLabel);

    const patternInput = document.createElement("input");
    patternInput.type = "text";
    patternInput.className = "param_input load-program-text-input";
    patternInput.placeholder = "200,400,1000,200,400";
    patternInput.value = active.power_pattern;

    const sparklineContainer = document.createElement("div");
    sparklineContainer.className = "load-program-sparkline";
    editor.appendChild(patternInput);
    editor.appendChild(sparklineContainer);

    const updateSparkline = (raw) => {
      const values = raw
        .split(",")
        .map((s) => parseFloat(s.trim()))
        .filter((v) => !isNaN(v) && v >= 0);
      sparklineContainer.innerHTML = "";
      if (values.length < 2) {
        sparklineContainer.style.display = "none";
        return;
      }
      sparklineContainer.style.display = "";
      const W = 320, H = 60, pad = 4;
      const maxV = Math.max(...values);
      const range = maxV || 1;
      const barW = Math.max(2, (W - pad * 2) / values.length - 1);
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      svg.setAttribute("width", W);
      svg.setAttribute("height", H);
      svg.style.cssText = "display:block;width:100%;max-width:400px;height:auto;";
      values.forEach((v, i) => {
        const x = pad + i * ((W - pad * 2) / values.length);
        const barHeight = Math.max(2, (v / range) * (H - pad * 2));
        const y = H - pad - barHeight;
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("x", x.toFixed(1));
        rect.setAttribute("y", y.toFixed(1));
        rect.setAttribute("width", barW.toFixed(1));
        rect.setAttribute("height", barHeight.toFixed(1));
        rect.setAttribute("rx", "2");
        rect.setAttribute("class", "sparkline-bar");
        const title = document.createElementNS(ns, "title");
        title.textContent = `Step ${i + 1}: ${v} W`;
        rect.appendChild(title);
        svg.appendChild(rect);
      });
      const baseline = document.createElementNS(ns, "line");
      baseline.setAttribute("x1", pad);
      baseline.setAttribute("y1", H - pad);
      baseline.setAttribute("x2", W - pad);
      baseline.setAttribute("y2", H - pad);
      baseline.setAttribute("class", "sparkline-baseline");
      svg.appendChild(baseline);
      sparklineContainer.appendChild(svg);
    };

    updateSparkline(active.power_pattern);

    patternInput.addEventListener("input", () => {
      active.power_pattern = patternInput.value;
      serialize();
      updateSparkline(patternInput.value);
    });
  };

  serialize();
  render(0);
}

function applyEVVisibility() {
  const evSection = document.getElementById("EV Charging");
  if (!evSection) return;

  const evBody = evSection.querySelector(".section-body");
  const activeIndex = evBody ? Number.parseInt(evBody.dataset.activeIndex || "0") : 0;
  const phaseModeDiv = document.getElementById("ev_phase_mode");
  const phaseModeSelect = phaseModeDiv
    ? phaseModeDiv.querySelectorAll("select")[activeIndex]
    : null;
  const phaseMode = phaseModeSelect ? phaseModeSelect.value : "auto_1_or_3_phase";

  const alwaysVisible = [
    "ev_charger_names",
    "ev_phase_mode",
    "ev_charge_mode_service",
    "ev_charge_mode_fast_value",
    "ev_charge_mode_eco_value",
    "ev_charge_mode_ecoplus_value",
    "ev_charge_mode_stopped_value",
    "ev_charge_mode_variable_value"
  ];

  const visibleByPhaseMode = {
    "1_phase": [
      "ev_charge_power_min_1_phase",
      "ev_charge_power_max_1_phase"
    ],
    "3_phase": [
      "ev_charge_power_min_3_phase",
      "ev_charge_power_max_3_phase"
    ],
    "auto_1_or_3_phase": [
      "ev_phase_select_entity",
      "ev_phase_select_value_1_phase",
      "ev_phase_select_value_3_phase",
      "ev_phase_select_value_auto",
      "ev_charge_power_min_1_phase",
      "ev_charge_power_max_1_phase",
      "ev_charge_power_min_3_phase",
      "ev_charge_power_max_3_phase"
    ]
  };

  const managedParams = new Set([
    ...alwaysVisible,
    ...Object.values(visibleByPhaseMode).flat()
  ]);
  const visibleNow = new Set([
    ...alwaysVisible,
    ...(visibleByPhaseMode[phaseMode] || visibleByPhaseMode["auto_1_or_3_phase"])
  ]);

  managedParams.forEach((id) => {
    const div = document.getElementById(id);
    if (!div) return;
    div.style.display = visibleNow.has(id) ? "" : "none";
  });
}

function applyHeatpumpModelVisibility() {
  const modelDiv = document.getElementById("heatpump_model_family");
  if (!modelDiv) return;
  const modelSelect = modelDiv.querySelector("select");
  if (!modelSelect) return;

  const family = modelSelect.value;
  // Per-room physics fields live on the Rooms tab, not here - Rooms renders
  // them as flat per-room arrays (not indexed-tab fields, so no
  // setupIndexedSectionTabs changes needed), gated the same way by family.
  const physicsFields = [
    "heatpump_room_u_value",
    "heatpump_room_envelope_area",
    "heatpump_room_ventilation_rate",
    "heatpump_room_window_area",
    "heatpump_room_shgc",
    "heatpump_room_internal_gains_factor",
    "heatpump_room_thermal_inertia_time_constant",
    "heatpump_room_carnot_efficiency"
  ];
  const mlFields = [
    "heatpump_ml_model_name",
    "heatpump_ml_model_path",
    "heatpump_two_stage_data_csv",
    "heatpump_two_stage_model_dir",
    "heatpump_two_stage_horizon",
    "heatpump_two_stage_top_k",
    "heatpump_two_stage_coarse_models",
    "heatpump_two_stage_fine_models"
  ];
  const dlFields = [
    "heatpump_dl_model_name",
    "heatpump_use_pinn",
    "heatpump_pinn_auto_train",
    "heatpump_pinn_lookahead"
  ];

  const setVisibility = (ids, visible) => {
    ids.forEach((id) => {
      const div = document.getElementById(id);
      if (div) div.style.display = visible ? "" : "none";
    });
  };

  setVisibility(physicsFields, family === "physics");
  setVisibility(mlFields, family === "machine_learning");
  setVisibility(dlFields, family === "deep_learning");

  const pinnToggleDiv = document.getElementById("heatpump_use_pinn");
  if (pinnToggleDiv && family === "deep_learning") {
    const pinnInput = pinnToggleDiv.querySelector("input[type='checkbox']");
    if (pinnInput) {
      const pinnOnly = ["heatpump_pinn_auto_train", "heatpump_pinn_lookahead"];
      pinnOnly.forEach((id) => {
        const div = document.getElementById(id);
        if (div) div.style.display = pinnInput.checked ? "" : "none";
      });
    }
  }
}

// heat_topology/is_electric_load (graph_topology mode) vs. heatpump_model_family
// + the Rooms tab (room_list mode, the default) are mutually exclusive ways
// to configure heating - only one is shown at a time.
function applyHeatpumpConfigModeVisibility() {
  const modeDiv = document.getElementById("heatpump_config_mode");
  if (!modeDiv) return;
  const modeSelect = modeDiv.querySelector("select");
  if (!modeSelect) return;

  const isTopology = modeSelect.value === "graph_topology";
  const topologyFields = ["heat_topology", "is_electric_load"];
  topologyFields.forEach((id) => {
    const div = document.getElementById(id);
    if (div) div.style.display = isTopology ? "" : "none";
  });

  const modelFamilyDiv = document.getElementById("heatpump_model_family");
  if (modelFamilyDiv) modelFamilyDiv.style.display = isTopology ? "none" : "";

  // graph_topology mode ignores the room list entirely (per this field's own
  // description above) - hide the Rooms sub-tab too, and bounce off it to
  // Heat Pump if it was the active sub-tab when the mode switched.
  const roomsTabBtn = document.querySelector('.thermal-subtabs .subtab-btn[data-target="Rooms"]');
  if (roomsTabBtn) {
    roomsTabBtn.style.display = isTopology ? "none" : "";
    if (isTopology && roomsTabBtn.classList.contains("active")) {
      const heatPumpTabBtn = document.querySelector('.thermal-subtabs .subtab-btn[data-target="Heat Pump"]');
      if (heatPumpTabBtn) heatPumpTabBtn.click();
    }
  }
}

function applyHybridTariffVisibility() {
  const hybridToggleDiv = document.getElementById("heatpump_is_hybrid");
  const gasPriceMethodDiv = document.getElementById("thermal_gas_price_forecast_method");
  const gasPriceDiv = document.getElementById("thermal_gas_price");
  const gasPriceColDiv = document.getElementById("thermal_gas_price_col");

  const hybridInput = hybridToggleDiv
    ? hybridToggleDiv.querySelector("input[type='checkbox']")
    : null;
  const gasPriceMethodInput = gasPriceMethodDiv
    ? gasPriceMethodDiv.querySelector("select")
    : null;

  const isHybrid = hybridInput ? hybridInput.checked : false;
  const gasPriceMethod = gasPriceMethodInput ? gasPriceMethodInput.value : "constant";

  if (gasPriceMethodDiv) gasPriceMethodDiv.style.display = isHybrid ? "" : "none";
  if (gasPriceDiv) gasPriceDiv.style.display = isHybrid && gasPriceMethod === "constant" ? "" : "none";
  if (gasPriceColDiv) gasPriceColDiv.style.display = isHybrid && gasPriceMethod === "csv" ? "" : "none";
}

// set_use_battery (Battery section header) -> set_use_battery_identification
// (Solar System (PV)) -> sensor_power_battery/sensor_battery_state_of_charge
// (Local). A plain "requires" can't express either link: set_use_battery is a
// header toggle with no .param_input class, and the sensor fields live in a
// section built BEFORE Solar System (PV), so a forward-referencing "requires"
// would silently fail to attach.
function applyBatteryIdentificationVisibility() {
  const useBatteryInput = document.getElementById("set_use_battery");
  const useBattery = useBatteryInput ? useBatteryInput.checked : false;

  const identificationDiv = document.getElementById("set_use_battery_identification");
  if (identificationDiv) identificationDiv.style.display = useBattery ? "" : "none";

  const identificationCheckbox = identificationDiv
    ? identificationDiv.querySelector("input[type='checkbox']")
    : null;
  const identificationEnabled = useBattery && identificationCheckbox ? identificationCheckbox.checked : false;

  ["sensor_power_battery", "sensor_battery_state_of_charge"].forEach((id) => {
    const div = document.getElementById(id);
    if (div) div.style.display = identificationEnabled ? "" : "none";
  });
}

// Boiler is not an indexed-tab section (unlike Deferrable Loads/Rooms/EV
// Charging) - every boiler's array.select/array.* inputs render as a flat,
// simultaneously-visible list, one input per boiler, all sharing the same
// index order as boiler_type's own selects. So per-field gating here hides
// individual input rows by index instead of hiding a whole tab/param div.
function applyBoilerVisibility() {
  const boilerTypeDiv = document.getElementById("boiler_type");
  if (!boilerTypeDiv) return;
  const typeSelects = Array.from(boilerTypeDiv.querySelectorAll("select.param_input"));
  if (!typeSelects.length) return;

  const applyPerIndex = (fieldId, predicate) => {
    const div = document.getElementById(fieldId);
    if (!div) return;
    const inputs = Array.from(div.querySelectorAll(".param_input"));
    typeSelects.forEach((select, index) => {
      const input = inputs[index];
      if (!input) return;
      input.style.display = predicate(select.value) ? "" : "none";
    });
  };

  // Only relevant when this boiler shares its physical unit with a
  // space-heating load (see boiler_type's own description).
  applyPerIndex("boiler_coupled_heatpump_load_index", (type) => type === "hp_tank_zone");
  applyPerIndex("boiler_hp_shared_max_power", (type) => type === "hp_tank_zone");
  // Only used to estimate COP for heat-pump based boiler types - meaningless
  // for resistive (flat COP=1.0).
  applyPerIndex("boiler_supply_temperature", (type) => type !== "resistive");
}

function setupWeatherCurvePreview() {
  const interceptInput = document.getElementById("heatpump_curve_intercept")?.querySelector(".param_input");
  const slopeInput = document.getElementById("heatpump_curve_slope")?.querySelector(".param_input");
  const minFlowInput = document.getElementById("heatpump_supply_temp_min")?.querySelector(".param_input");
  const maxFlowInput = document.getElementById("heatpump_supply_temp_max")?.querySelector(".param_input");
  const anchor = document.getElementById("heatpump_curve_intercept");
  if (!interceptInput || !slopeInput || !minFlowInput || !maxFlowInput || !anchor) return;

  let preview = document.getElementById("weather-curve-preview");
  if (!preview) {
    preview = document.createElement("div");
    preview.id = "weather-curve-preview";
    preview.className = "param-preview";
    anchor.appendChild(preview);
  }

  const update = () => {
    const intercept = Number.parseFloat(interceptInput.value || "40");
    const slope = Number.parseFloat(slopeInput.value || "-1");
    const minFlow = Number.parseFloat(minFlowInput.value || "20");
    const maxFlow = Number.parseFloat(maxFlowInput.value || "60");

    const sampleOutdoor = [-10, 0, 10, 20];
    const points = sampleOutdoor.map((tOut) => {
      const raw = slope * tOut + intercept;
      const clipped = Math.max(minFlow, Math.min(maxFlow, raw));
      return `${tOut}C -> ${clipped.toFixed(1)}C`;
    });

    preview.innerHTML = [
      `<strong>Weather curve preview</strong>`,
      `Formula: flow = ${slope.toFixed(2)} * outdoor + ${intercept.toFixed(1)}`,
      `Flow limits: ${minFlow.toFixed(1)}C to ${maxFlow.toFixed(1)}C`,
      `Samples: ${points.join(" | ")}`
    ].join("<br>");
  };

  [interceptInput, slopeInput, minFlowInput, maxFlowInput].forEach((input) => {
    input.addEventListener("input", update);
  });
  update();
}

//load list configuration view
function loadConfigurationListView(param_definitions, config, list_html) {
  if (list_html == null || config == null || param_definitions == null) {
    return 1;
  }

  //list parameters used in the section headers
  //#610: number_of_batteries added to auto-sync the 15 per-battery array params
  //(mirrors number_of_deferrable_loads), see the "number_of_batteries" case in headerElement
  let header_input_list = [
    "set_use_battery",
    "set_use_pv",
    "number_of_deferrable_loads",
    "number_of_batteries",
    "set_use_heatpump",
    "set_use_boiler",
    "heatpump_number_of_rooms",
    "number_of_ev_chargers",
    "set_use_ev_charger",
    "manual_load_enabled"
  ];

  //get the main container and append list template html
  document.getElementById("configuration-container").innerHTML = list_html;

  //loop through configuration sections ('Local','System','Tariff','Solar System (PV)') in definitions file
  for (let section in param_definitions) {
    // build each section by adding parameters with their corresponding input elements
    buildParamContainers(
      section,
      param_definitions[section],
      config,
      header_input_list
    );

    //after sections have been built, add event listeners for section header inputs
    //loop though headers
    for (let header_input_param of header_input_list) {
      if (param_definitions[section].hasOwnProperty(header_input_param)) {
        //grab default from definitions file
        let value = param_definitions[section][header_input_param]["default_value"];
        //find input element (using the parameter name as the input element ID)
        let header_input_element = document.getElementById(header_input_param);
        if (header_input_element !== null) {
          //add event listener to element (trigger on input change)
          header_input_element.addEventListener("input", (e) =>
            headerElement(e.target, param_definitions, config)
          );
          //check the EMHASS config to see if it contains a stored param value
          //else keep default
          value = checkConfigParam(value, config, header_input_param);
          //set value of input
          header_input_element.value = value;
          //checkboxes (for Booleans) also set value to "checked"
          if (header_input_element.type == "checkbox") {
            header_input_element.checked = value;
          }
          //manually trigger the header parameter input event listener for setting up initial section state
          headerElement(header_input_element, param_definitions, config);
        }
      }
    }
  }

  // Dynamic hiding for InfluxDB options
  const use_influx_param = "use_influxdb";
  const influx_related_params = [
    "influxdb_host",
    "influxdb_port",
    "influxdb_database",
    "influxdb_measurement",
    "influxdb_retention_policy",
    "influxdb_use_ssl",
    "influxdb_verify_ssl"
  ];

  const influx_toggle_div = document.getElementById(use_influx_param);
  if (influx_toggle_div) {
    // The actual input is inside the div with the ID
    const influx_input = influx_toggle_div.querySelector("input");
    if (influx_input) {
      const toggleInfluxVisibility = () => {
        const isChecked = influx_input.checked;
        influx_related_params.forEach(paramId => {
          const paramDiv = document.getElementById(paramId);
          if (paramDiv) {
            paramDiv.style.display = isChecked ? "" : "none";
          }
        });
      };

      // Add listener and set initial state
      influx_input.addEventListener("change", toggleInfluxVisibility);
      toggleInfluxVisibility();
    }
  }

  setupSectionTabs();

  const model_family_div = document.getElementById("heatpump_model_family");
  if (model_family_div) {
    const model_select = model_family_div.querySelector("select");
    if (model_select) {
      model_select.addEventListener("change", applyHeatpumpModelVisibility);
      applyHeatpumpModelVisibility();
    }
  }

  const config_mode_div = document.getElementById("heatpump_config_mode");
  if (config_mode_div) {
    const config_mode_select = config_mode_div.querySelector("select");
    if (config_mode_select) {
      config_mode_select.addEventListener("change", applyHeatpumpConfigModeVisibility);
      applyHeatpumpConfigModeVisibility();
    }
  }

  const pinn_toggle_div = document.getElementById("heatpump_use_pinn");
  if (pinn_toggle_div) {
    const pinn_input = pinn_toggle_div.querySelector("input[type='checkbox']");
    if (pinn_input) {
      pinn_input.addEventListener("change", applyHeatpumpModelVisibility);
    }
  }

  // Heat Pump: hybride toggle → show/hide gasmeter sensor
  const hybrid_toggle_div = document.getElementById("heatpump_is_hybrid");
  if (hybrid_toggle_div) {
    const hybrid_input = hybrid_toggle_div.querySelector("input[type='checkbox']");
    if (hybrid_input) {
      const toggleHybridVisibility = () => {
        const isHybrid = hybrid_input.checked;
        const gasDiv = document.getElementById("heatpump_gas_meter_sensor");
        if (gasDiv) gasDiv.style.display = isHybrid ? "" : "none";
        applyHybridTariffVisibility();
      };
      hybrid_input.addEventListener("change", toggleHybridVisibility);
      toggleHybridVisibility();
    }
  }

  const gas_price_method_div = document.getElementById("thermal_gas_price_forecast_method");
  if (gas_price_method_div) {
    const gas_price_method_select = gas_price_method_div.querySelector("select");
    if (gas_price_method_select) {
      gas_price_method_select.addEventListener("change", applyHybridTariffVisibility);
      applyHybridTariffVisibility();
    }
  }

  // Battery self-identification: set_use_battery (header) and its own toggle → sensor fields
  const battery_identification_div = document.getElementById("set_use_battery_identification");
  if (battery_identification_div) {
    const battery_identification_checkbox = battery_identification_div.querySelector("input[type='checkbox']");
    if (battery_identification_checkbox) {
      battery_identification_checkbox.addEventListener("change", applyBatteryIdentificationVisibility);
    }
  }
  applyBatteryIdentificationVisibility();

  // Boiler: type (per row) → show/hide the hp_tank_zone-only/heat-pump-only fields
  const boiler_type_div = document.getElementById("boiler_type");
  if (boiler_type_div) {
    // Delegated listener: also covers selects added later via the field's own +/- buttons.
    boiler_type_div.addEventListener("change", applyBoilerVisibility);
    document.querySelectorAll(".input-plus.boiler_type, .input-minus.boiler_type").forEach((btn) => {
      btn.addEventListener("click", () => applyBoilerVisibility());
    });
    applyBoilerVisibility();
  }

  // Heat Pump: control mode → show/hide stooklijst params & thermostat sensor
  const control_mode_div = document.getElementById("heatpump_control_mode");
  if (control_mode_div) {
    const control_select = control_mode_div.querySelector("select");
    if (control_select) {
      const toggleControlModeVisibility = () => {
        const mode = control_select.value;
        const weatherParams = ["heatpump_curve_intercept", "heatpump_curve_slope", "heatpump_max_deviation"];
        weatherParams.forEach(id => {
          const div = document.getElementById(id);
          if (div) div.style.display = mode === "weather_curve" ? "" : "none";
        });
        const thermostatDiv = document.getElementById("heatpump_target_temp_sensor");
        if (thermostatDiv) thermostatDiv.style.display = mode === "thermostat_sensor" ? "" : "none";
      };
      control_select.addEventListener("change", toggleControlModeVisibility);
      toggleControlModeVisibility();
    }
  }

  setupWeatherCurvePreview();

  const ev_phase_mode_div = document.getElementById("ev_phase_mode");
  if (ev_phase_mode_div) {
    const ev_phase_selects = ev_phase_mode_div.querySelectorAll("select");
    ev_phase_selects.forEach((select) => {
      select.addEventListener("change", applyEVVisibility);
    });
  }

  const load_type_div = document.getElementById("load_type");
  if (load_type_div) {
    const load_type_selects = load_type_div.querySelectorAll("select");
    load_type_selects.forEach((select) => {
      select.addEventListener("change", () => {
        applyLoadTypeVisibility();
        applyWashdataVisibility();
        setupLoadProgramTabs();
      });
    });
  }

  const load_dispatch_mode_div = document.getElementById("load_dispatch_mode");
  if (load_dispatch_mode_div) {
    const load_dispatch_selects = load_dispatch_mode_div.querySelectorAll("select");
    load_dispatch_selects.forEach((select) => {
      select.addEventListener("change", () => {
        applyLoadTypeVisibility();
      });
    });
  }

  const is_manual_load_div = document.getElementById("is_manual_load");
  if (is_manual_load_div) {
    const is_manual_load_checkboxes = is_manual_load_div.querySelectorAll("input[type='checkbox']");
    is_manual_load_checkboxes.forEach((checkbox) => {
      checkbox.addEventListener("change", applyManualLoadVisibility);
    });
  }

  const load_washdata_enabled_div = document.getElementById("load_washdata_enabled");
  if (load_washdata_enabled_div) {
    const load_washdata_enabled_checkboxes = load_washdata_enabled_div.querySelectorAll("input[type='checkbox']");
    load_washdata_enabled_checkboxes.forEach((checkbox) => {
      checkbox.addEventListener("change", applyWashdataVisibility);
    });
  }
  const load_washdata_device_div = document.getElementById("load_washdata_device");
  if (load_washdata_device_div) {
    const load_washdata_device_selects = load_washdata_device_div.querySelectorAll("select.param_input");
    load_washdata_device_selects.forEach((select) => {
      select.addEventListener("change", () => {
        applyLoadTypeVisibility();
        applyWashdataVisibility();
      });
    });
  }
  attachWashdataDeviceSuggestions(config);
  attachEntitySuggestions();

  setupIndexedSectionTabs("Deferrable Loads", "number_of_deferrable_loads", "Load", "load_names", [
    "load_names",
    "load_type",
    "load_washdata_enabled",
    "load_washdata_device",
    "is_manual_load",
    "manual_load_ready_sensor",
    "manual_load_confirm_power_sensor",
    "manual_load_program_select_sensor",
    "manual_load_deadline_hour",
    "start_timesteps_of_each_deferrable_load",
    "end_timesteps_of_each_deferrable_load",
    "load_dispatch_mode",
    "load_programs",
    "required_energy_kwh_of_each_deferrable_load",
    "nominal_power_of_deferrable_loads",
    "minimum_power_of_deferrable_loads",
    "operating_hours_of_each_deferrable_load",
    "set_deferrable_startup_penalty",
    "treat_deferrable_load_as_semi_cont",
    "set_deferrable_load_single_constant",
    "set_deferrable_max_startups",
    "def_minimum_on_time",
    "def_minimum_off_time",
    "deferrable_load_max_cost",
    "cost_forecast_per_deferrable_load",
  ], () => {
    applyLoadTypeVisibility();
    applyManualLoadVisibility();
    applyWashdataVisibility();
    setupLoadProgramTabs();
  });
  setupIndexedSectionTabs("Rooms", "heatpump_number_of_rooms", "Room", "heatpump_room_names", [
    "heatpump_room_names",
    "heatpump_room_temp_sensors",
    "heatpump_room_valve_sensors",
    "heatpump_room_valve_mode",
    "heatpump_room_blind_sensors",
    "heatpump_room_window_sensors",
    "heatpump_room_door_sensors",
    "heatpump_room_min_temperature",
    "heatpump_room_max_temperature",
    "heatpump_room_target_temperature",
    "heatpump_room_nominal_power",
    "heatpump_room_supply_temperature",
    "heatpump_room_volume",
    "heatpump_room_shared_group",
    "heatpump_room_u_value",
    "heatpump_room_envelope_area",
    "heatpump_room_ventilation_rate",
    "heatpump_room_window_area",
    "heatpump_room_shgc",
    "heatpump_room_internal_gains_factor",
    "heatpump_room_thermal_inertia_time_constant",
    "heatpump_room_carnot_efficiency"
  ]);
  setupIndexedSectionTabs("EV Charging", "number_of_ev_chargers", "Charger", "ev_charger_names", [
    "ev_charger_names",
    "ev_phase_mode",
    "ev_charge_mode_service",
    "ev_phase_select_entity",
    "ev_charge_mode_stopped_value",
    "ev_charge_mode_fast_value",
    "ev_charge_mode_eco_value",
    "ev_charge_mode_ecoplus_value",
    "ev_charge_mode_variable_value",
    "ev_phase_select_value_1_phase",
    "ev_phase_select_value_3_phase",
    "ev_phase_select_value_auto",
    "ev_charge_power_min_1_phase",
    "ev_charge_power_max_1_phase",
    "ev_charge_power_min_3_phase",
    "ev_charge_power_max_3_phase"
  ], applyEVVisibility);

  normalizeIndexedNames("number_of_deferrable_loads", "load_names", "load");
  normalizeIndexedNames("heatpump_number_of_rooms", "heatpump_room_names", "room");
  normalizeIndexedNames("number_of_ev_chargers", "ev_charger_names", "ev");

  applyLoadTypeVisibility();
  applyManualLoadVisibility();
  applyWashdataVisibility();
  setupLoadProgramTabs();
  applyEVVisibility();
}

function setupSectionTabs() {
  const container = document.getElementById("configuration-container");
  if (!container) return;

  const cards = Array.from(container.querySelectorAll(".section-card"));
  if (!cards.length) return;

  const thermalSections = ["Heat Pump", "Boiler", "Rooms"];
  const sectionMap = new Map(cards.map((card) => [card.id, card]));
  const topLevelSections = [
    { id: "Local", label: "Local" },
    { id: "System", label: "System" },
    { id: "Tariff", label: "Tariff" },
    { id: "Solar System (PV)", label: "Solar System (PV)" },
    { id: "Battery", label: "Battery" },
    { id: "Thermal", label: "Thermal" },
    { id: "EV Charging", label: "EV Charging" },
    { id: "Deferrable Loads", label: "Loads" }
  ];

  const existingTabs = container.querySelector(".section-tabs");
  if (existingTabs) existingTabs.remove();
  const existingThermalTabs = container.querySelector(".thermal-subtabs");
  if (existingThermalTabs) existingThermalTabs.remove();

  const tabs = document.createElement("div");
  tabs.className = "section-tabs";

  const thermalTabs = document.createElement("div");
  thermalTabs.className = "subtabs-bar thermal-subtabs";
  thermalTabs.style.display = "none";

  const activateThermalSubtab = (targetId) => {
    cards.forEach((card) => {
      if (thermalSections.includes(card.id)) {
        card.style.display = card.id === targetId ? "block" : "none";
      }
    });
    Array.from(thermalTabs.querySelectorAll(".subtab-btn")).forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.target === targetId);
    });
  };

  thermalSections.forEach((sectionId) => {
    if (!sectionMap.has(sectionId)) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "subtab-btn";
    btn.dataset.target = sectionId;
    btn.textContent = sectionId;
    btn.addEventListener("click", () => activateThermalSubtab(sectionId));
    thermalTabs.appendChild(btn);
  });

  const activateTab = (targetId) => {
    cards.forEach((card) => {
      card.style.display = "none";
    });

    if (targetId === "Thermal") {
      thermalTabs.style.display = "flex";
      activateThermalSubtab("Heat Pump");
    } else {
      thermalTabs.style.display = "none";
      const card = sectionMap.get(targetId);
      if (card) {
        card.style.display = "block";
      }
    }

    Array.from(tabs.querySelectorAll(".section-tab-btn")).forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.target === targetId);
    });
  };

  topLevelSections.forEach((section) => {
    if (section.id !== "Thermal" && !sectionMap.has(section.id)) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "section-tab-btn";
    btn.dataset.target = section.id;
    btn.textContent = section.label;
    btn.addEventListener("click", () => activateTab(section.id));
    tabs.appendChild(btn);
  });

  container.insertBefore(tabs, cards[0]);
  container.insertBefore(thermalTabs, cards[0]);

  activateTab("Local");
}

function setupIndexedSectionTabs(sectionId, countParamId, tabLabelPrefix, namesParamId = null, targetParamIds = null, onActivate = null) {
  const section = document.getElementById(sectionId);
  if (!section) return;

  const body = section.querySelector(".section-body");
  if (!body) return;

  const countElement = document.getElementById(countParamId);
  if (!countElement) return;

  const count = Math.max(1, Number.parseInt(countElement.value || "1"));

  const oldTabs = body.querySelector(".subtabs-bar.indexed-tabs");
  if (oldTabs) oldTabs.remove();

  const tabsBar = document.createElement("div");
  tabsBar.className = "subtabs-bar indexed-tabs";

  const namesElement = namesParamId ? document.getElementById(namesParamId) : null;
  const getNameAtIndex = (index) => {
    if (!namesElement) return `${tabLabelPrefix} ${index + 1}`;
    const nameInputs = Array.from(namesElement.querySelectorAll(".param_input"));
    if (!nameInputs.length) return `${tabLabelPrefix} ${index + 1}`;
    const nameValue = (nameInputs[index] && nameInputs[index].value) ? nameInputs[index].value.trim() : "";
    return nameValue || `${tabLabelPrefix} ${index + 1}`;
  };

  const updateTabLabels = () => {
    Array.from(tabsBar.querySelectorAll(".subtab-btn")).forEach((btn, index) => {
      btn.textContent = getNameAtIndex(index);
    });
  };

  const activateIndex = (targetIndex) => {
    body.dataset.activeIndex = String(targetIndex);
    const params = Array.from(body.querySelectorAll(".param"));
    params.forEach(param => {
      if (targetParamIds && !targetParamIds.includes(param.id)) return;
      const inputs = Array.from(param.querySelectorAll(".param_input"));
      if (inputs.length <= 1) return;
      inputs.forEach((input, idx) => {
        const wrapper = input.parentElement && input.parentElement.classList.contains("switch")
          ? input.parentElement
          : input;
        if (wrapper) {
          wrapper.style.display = idx === targetIndex ? "" : "none";
        }
      });
    });

    Array.from(tabsBar.querySelectorAll(".subtab-btn")).forEach(btn => {
      btn.classList.toggle("active", Number.parseInt(btn.dataset.index) === targetIndex);
    });

    if (onActivate) onActivate(targetIndex);
  };

  for (let i = 0; i < count; i++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "subtab-btn";
    btn.dataset.index = i;
    btn.textContent = getNameAtIndex(i);
    btn.addEventListener("click", () => activateIndex(i));
    tabsBar.appendChild(btn);
  }

  body.insertBefore(tabsBar, body.firstChild);
  if (namesElement) {
    Array.from(namesElement.querySelectorAll(".param_input")).forEach((input) => {
      input.addEventListener("input", updateTabLabels);
    });
  }
  updateTabLabels();
  activateIndex(0);
}

//build sections body, containing parameter/param containers (containing parameter/param inputs)
function buildParamContainers(
  section,
  section_parameters_definitions,
  config,
  header_input_list
) {
  //get the section container element
  let SectionContainer = document.getElementById(section);
  //get the body container inside the section (where the parameters will be appended)
  let SectionParamElement = SectionContainer.getElementsByClassName("section-body");
  if (SectionContainer == null || SectionParamElement.length == 0) {
    console.error("Unable to find Section container or Section Body");
    return 0;
  }

  //loop though the sections parameters in definition file, generate and append param (div) elements for the section
  for (const [
    parameter_definition_name,
    parameter_definition_object,
  ] of Object.entries(section_parameters_definitions)) {
    //check parameter definitions have the required key values
    if (
      !("friendly_name" in parameter_definition_object) ||
      !("Description" in parameter_definition_object) ||
      !("input" in parameter_definition_object) ||
      !("default_value" in parameter_definition_object)
    ) {
      console.log(
        parameter_definition_name +
          " is missing some required values in the definitions file"
      );
      continue;
    }
    if (
      parameter_definition_object["input"] === "select" &&
      !("select_options" in parameter_definition_object)
    ) {
      console.log(
        parameter_definition_name +
          " is missing select_options values in the definitions file"
      );
      continue;
    }

    //check if param is set in the section header, if so skip building param
    if (header_input_list.includes(parameter_definition_name)) {
      continue;
    }

    //if parameter type == array.* and not in "Deferrable Loads" or "Battery" section,
    //append plus and minus buttons in param div. Battery is excluded the same way
    //Deferrable Loads is (#610): the 15 per-battery array params
    //(BATTERY_ARRAY_PARAMS) are length-managed exclusively by the
    //number_of_batteries header count below, so no free +/- button can desync a
    //param's length from number_of_batteries and crash check_batt_params on save.
    let array_buttons = "";
    if (
      parameter_definition_object["input"].search("array.") > -1 &&
      section != "Deferrable Loads" &&
      section != "Rooms" &&
      section != "Battery"
    ) {
      array_buttons = `
                  <button type="button" class="input-plus ${parameter_definition_name}">+</button>
                  <button type="button" class="input-minus ${parameter_definition_name}">-</button>
                  <br>
                  `;
    }

    //generates and appends param container into section
    //buildParamElement() builds the parameter input/s and returns html to append in param-input
    SectionParamElement[0].innerHTML += `
          <div class="param" id="${parameter_definition_name}">
             <h5>${
               parameter_definition_object["friendly_name"]
             }:</h5> <i>${parameter_definition_name}</i> </br>
              ${array_buttons}
             <div class="param-input"> 
                  ${buildParamElement(
                    parameter_definition_object,
                    parameter_definition_name,
                    config
                  )}
             </div>
              <p>${parameter_definition_object["Description"]}</p>
          </div>
          `;
  }

  //after looping though, build and appending the parameters in the corresponding section:
  //create add button (array plus) event listeners
  let plus = SectionContainer.querySelectorAll(".input-plus");
  plus.forEach(function (answer) {
    answer.addEventListener("click", () =>
      plusElements(answer.classList[1], param_definitions, section, {})
    );
  });

  //create subtract button (array minus) event listeners
  let minus = SectionContainer.querySelectorAll(".input-minus");
  minus.forEach(function (answer) {
    answer.addEventListener("click", () => minusElements(answer.classList[1]));
  });

  //check initial checkbox state, check "value" of input and match to "checked" value
  let checkbox = SectionContainer.querySelectorAll("input[type='checkbox']");
  checkbox.forEach(function (answer) {
    let value = answer.value === "true";
    answer.checked = value;
  });

  //loop though sections params again, check if param has a requirement, if so add a event listener to the required param input
  //if required param gets changed, trigger function to check if that required parameter matches the required value for the param
  //if false, add css class to param element to shadow it, to show that its unaccessible
  for (const [
    parameter_definition_name,
    parameter_definition_object,
  ] of Object.entries(section_parameters_definitions)) {
    //check if param has a requirement from definitions file
    if ("requires" in parameter_definition_object) {
      // get param requirement element
      const requirement_element = document.getElementById(
        Object.keys(parameter_definition_object["requires"])[0]
      );
      if (requirement_element == null) {
        console.debug(
          "unable to find " +
            Object.keys(parameter_definition_object["requires"])[0] +
            " param div container element"
        );
        continue;
      }

      // get param element that has requirement
      const param_element = document.getElementById(parameter_definition_name);
      if (param_element == null) {
        console.debug(
          "unable to find " +
            parameter_definition_name +
            " param div container element"
        );
        continue;
      }

      //obtain required param inputs, add event listeners
      let requirement_inputs =
        requirement_element.getElementsByClassName("param_input");
      //grab required value
      const requirement_value = Object.values(
        parameter_definition_object["requires"]
      )[0];

      //for all required inputs
      for (const input of requirement_inputs) {
        //if listener not already attached
        if (input.getAttribute("listener") !== "true") {
          //create event listener with arguments referencing the required param. param with requirement and required value
          input.addEventListener("input", () =>
            checkRequirements(input, param_element, requirement_value)
          );
          //manually run function to gain initial param element initial state
          checkRequirements(input, param_element, requirement_value);
        }
      }
    }
  }
}

//create html input element/s for a param container (called by buildParamContainers)
function buildParamElement(
  parameter_definition_object,
  parameter_definition_name,
  config
) {
  let type = "";
  let inputs = "";
  let type_specific_html = "";
  let type_specific_html_end = "";
  let placeholder = ""

  //switch statement to adjust generated html according to the parameter data type (definitions in definitions file)
  switch (parameter_definition_object["input"]) {
    case "array.int":
    //number
    case "int":
      type = "number";
      placeholder = Number.parseInt(parameter_definition_object["default_value"]);
      break;
    case "array.float":
    case "float":
      type = "number";
      placeholder = Number.parseFloat(parameter_definition_object["default_value"]);
      break;
    //text (string)
    case "array.string":
    case "string":
      type = "text";
      placeholder = parameter_definition_object["default_value"];
      break;
    case "array.time":
    //time ("00:00")
    case "time":
      type = "time";
      break;
    //checkbox (boolean)
    case "array.boolean":
    case "boolean":
      type = "checkbox";
      type_specific_html = `
              <label class="switch">
              `;
      type_specific_html_end = `
              <span class="slider"></span>
              </label>
              `;
      placeholder = parameter_definition_object["default_value"] === "true";
      break;
    //selects (pick)
    case "array.select":
      placeholder = parameter_definition_object["default_value"];
      break;
    case "select":
      //format selects later
      break;
    case "object":
    case "array.array.float":
      type = "text";
      placeholder = parameter_definition_object["default_value"] === null
        ? ""
        : JSON.stringify(parameter_definition_object["default_value"]);
      break;
  }

  //check default values saved in param definitions
  //definitions default value is used if none is found in the configs, or an array element has been added in the ui (deferrable load number increase or plus button pressed)
  //check if a param value is saved in the config file (if so overwrite definition default)
  let value = checkConfigParam(placeholder, config, parameter_definition_name);

  //generate and return param input html,
  //check if param value is not an object, if so assume its a single value.
  if (typeof value !== "object") {
    //if select, generate and return select elements instead of input
    if (parameter_definition_object["input"] == "select" || parameter_definition_object["input"] == "array.select") {
      let inputs = `<select class="param_input">`;
      for (const options of parameter_definition_object["select_options"]) {
        let selected = ""
        //if item in select is the same as the config value, then append "selected" tag
        if (options==value) {selected = `selected="selected"`}
        inputs += `<option ${selected}>${options}</option>`;
      }
      inputs += `</select>`;
      return inputs;
    }
    // generate param input html and return
    else {
      return `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder=${parameter_definition_object["default_value"]} value=${value} >
          ${type_specific_html_end}
          `;
    }
  }
  // else if object, loop though array of values, generate input element per value, and and return
  else {
    if (parameter_definition_object["input"] == "array.select") {
      let inputs = "";
      for (let param of value) {
        inputs += `<select class="param_input">`;
        for (const options of parameter_definition_object["select_options"]) {
          let selected = "";
          if (options == param) {selected = `selected="selected"`}
          inputs += `<option ${selected}>${options}</option>`;
        }
        inputs += `</select>`;
      }
      return inputs;
    }
    // null default: render a single empty input so the section keeps rendering
    if (value === null) {
      return `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder="${placeholder}" value="">
          ${type_specific_html_end}
          `;
    }
    // The nested-object path below is designed only for load_peak_hour_periods.
    if (parameter_definition_object["input"] === "object") {
      return `<input class="param_input" type="text" placeholder="${placeholder}" value="${JSON.stringify(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')}">`;
    }
    //for items such as load_peak_hour_periods (object of objects with arrays)
    // exclude null: typeof null === "object" is a JS gotcha — null elements must fall to the array branch
    if (typeof Object.values(value)[0] === "object" && Object.values(value)[0] !== null) {
      for (let param of Object.values(value)) {
        for (let items of Object.values(param)) {
          inputs += `<input class="param_input" type="${type}" placeholder=${Object.values(items)[0]} value=${
            Object.values(items)[0]
          }>`;
        }
        inputs += `</br>`;
      }
      return inputs;
    }
    // array of values
    else {
      let inputs = "";
      for (let param of value) {
        inputs += `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder=${parameter_definition_object["default_value"]} value=${param}>
          ${type_specific_html_end}
          `;
      }
      return inputs;
    }
  }
}

//add param inputs in param div container (for type array)
function plusElements(
  parameter_definition_name,
  param_definitions,
  section,
  config
) {
  let param_element = document.getElementById(parameter_definition_name);
  if (param_element == null) {
    console.log(
      "Unable to find " + parameter_definition_name + " param div container"
    );
    return 1;
  }
  let param_input_container =
    param_element.getElementsByClassName("param-input")[0];
  // Add a copy of the param element
  param_input_container.innerHTML += buildParamElement(
    param_definitions[section][parameter_definition_name],
    parameter_definition_name,
    config
  );
}

//Remove param inputs in param div container (minimum 1)
function minusElements(param) {
  let param_element = document.getElementById(param);
  let param_input
  if (param_element == null) {
    console.log(
      "Unable to find " + param + " param div container"
    );
    return 1;
  }
  let param_input_list = param_element.getElementsByTagName("input");
  if (param_input_list.length == 0) {
    param_input_list = param_element.getElementsByTagName("select");
  }
  if (param_input_list.length == 0) {
    console.log(
      "Unable to find " + param + " param input/s"
    );
    return 1;
  }

  //verify if input is a boolean (if so remove parent slider/switch element with input)
  if (
    param_input_list[param_input_list.length - 1].parentNode.tagName === "LABEL"
  ) {
    param_input = param_input_list[param_input_list.length - 1].parentNode;
  } else {
    param_input = param_input_list[param_input_list.length - 1];
  }

  //if param is "load_peak_hour_periods", remove both start and end param inputs as well as the line brake tag separating the inputs
  if (param == "load_peak_hour_periods") {
    if (param_input_list.length > 2) {
      let brs = document.getElementById(param).getElementsByTagName("br");
      param_input_list[param_input_list.length - 1].remove();
      param_input_list[param_input_list.length - 1].remove();
      brs[brs.length - 1].remove();
    }
  } else if (param_input_list.length > 1) {
    param_input.remove();
  }
}

//check requirement_element inputs,
//if requirement_element don't match requirement_value, add .requirement-disable class to param_element
//else remove class
function checkRequirements(
  requirement_element,
  param_element,
  requirement_value
) {
  let requirement_element_value
  //get current value of required element
  if (requirement_element.type == "checkbox") {
    requirement_element_value = requirement_element.checked;
  } else {
    requirement_element_value = requirement_element.value;
  }

  //fully hide the param (not just dim it) when its requirement isn't met,
  //so "only visible when needed" holds for every field with a "requires" key
  //in param_definitions.json, without needing a bespoke apply*Visibility()
  //function per field.
  param_element.style.display = requirement_element_value == requirement_value ? "" : "none";
}

//on header input change, execute accordingly
function headerElement(element, param_definitions, config) {
  //obtain section body element
  let section_card = element.closest(".section-card");
  let param_list 
  let difference
  if (section_card == null) {
    console.log("Unable to obtain section-card");
    return 1;
  }
  let param_container = section_card.getElementsByClassName("section-body");
  if (param_container.length > 0) {
    param_container = section_card.getElementsByClassName("section-body")[0];
  } else {
    console.log("Unable to obtain section-body");
    return 1;
  }

  switch (element.id) {
    //if set_use_battery, add or remove battery section (inc. params)
    case "set_use_battery":
      if (element.checked) {
        param_container.innerHTML = "";
        //#610: also exclude number_of_batteries here (its own header case below
        //owns it) - otherwise this rebuild would render a SECOND, duplicate
        //"number_of_batteries" .param row inside the section body alongside the
        //real header input, since this rebuild's own exclusion list is separate
        //from the outer header_input_list used at initial page load.
        buildParamContainers("Battery", param_definitions["Battery"], config, [
          "set_use_battery",
          "number_of_batteries",
        ]);
        element.checked = true;
      } else {
        param_container.innerHTML = "";
      }
      applyBatteryIdentificationVisibility();
      break;

    //if set_use_pv, add or remove PV section (inc. related params)
    case "set_use_pv":
      if (element.checked) {
        param_container.innerHTML = "";
        buildParamContainers("Solar System (PV)", param_definitions["Solar System (PV)"], config, [
          "set_use_pv",
        ]);
        element.checked = true;
      } else {
        param_container.innerHTML = "";
      }
      break;

    //if set_use_heatpump, add or remove Heat Pump section (inc. related params)
    case "set_use_heatpump":
      // Keep Heat Pump parameters visible at all times so users can preconfigure
      // thermal settings even before enabling optimization.
      break;

    //if set_use_boiler, keep section visible and let users preconfigure boiler setup
    case "set_use_boiler":
      break;

    //if set_use_ev_charger, keep section visible and let users preconfigure EV setup
    case "set_use_ev_charger":
      break;

    //if manual_load_enabled, show/hide the per-load "Manually-committed load"
    //toggle (and, transitively, its own dependent sensor fields) accordingly
    case "manual_load_enabled":
      applyManualLoadVisibility();
      break;

    //if number_of_deferrable_loads, the number of inputs in the "Deferrable Loads" section should add up to number_of_deferrable_loads value in header
    case "number_of_deferrable_loads":
      //get a list of param in section
      param_list = param_container.getElementsByClassName("param");
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_deferrable_loads"
        );
        return 1;
      }
      //calculate how much off the fist parameters input elements amount to is, compering to the number_of_deferrable_loads value
      const firstLoadParam = param_container.querySelector(".param");
      if (!firstLoadParam) {
        return 1;
      }
      difference =
        Number.parseInt(element.value) -
        firstLoadParam.querySelectorAll(".param_input").length;
      //add elements based on how many elements are missing
      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            //append element, do not pass config to obtain default parameter from definitions file
            plusElements(param.id, param_definitions, "Deferrable Loads", {});
          }
        }
      }
      //subtract elements based how many elements its over
      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id);
          }
        }
      }
      setupIndexedSectionTabs("Deferrable Loads", "number_of_deferrable_loads", "Load", "load_names", [
        "load_names",
        "load_type",
        "load_washdata_enabled",
        "load_washdata_device",
        "is_manual_load",
        "manual_load_ready_sensor",
        "manual_load_confirm_power_sensor",
        "manual_load_program_select_sensor",
        "manual_load_deadline_hour",
        "start_timesteps_of_each_deferrable_load",
        "end_timesteps_of_each_deferrable_load",
        "load_dispatch_mode",
        "load_programs",
        "required_energy_kwh_of_each_deferrable_load",
        "nominal_power_of_deferrable_loads",
        "minimum_power_of_deferrable_loads",
        "operating_hours_of_each_deferrable_load",
        "set_deferrable_startup_penalty",
        "treat_deferrable_load_as_semi_cont",
        "set_deferrable_load_single_constant",
        "set_deferrable_max_startups",
        "def_minimum_on_time",
        "def_minimum_off_time",
        "deferrable_load_max_cost",
        "cost_forecast_per_deferrable_load"
      ], () => {
        applyLoadTypeVisibility();
        applyManualLoadVisibility();
        applyWashdataVisibility();
        setupLoadProgramTabs();
      });
      normalizeIndexedNames("number_of_deferrable_loads", "load_names", "load");
      attachWashdataDeviceSuggestions(config);
      attachEntitySuggestions();
      applyLoadTypeVisibility();
      applyManualLoadVisibility();
      applyWashdataVisibility();
      setupLoadProgramTabs();
      break;

    //if heatpump_number_of_rooms, number of room array fields should match
    case "heatpump_number_of_rooms":
      param_list = param_container.getElementsByClassName("param");
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in heatpump_number_of_rooms"
        );
        return 1;
      }
      const firstRoomParam = param_container.querySelector(".param");
      if (!firstRoomParam) {
        return 1;
      }
      difference =
        Number.parseInt(element.value) -
        firstRoomParam.querySelectorAll(".param_input").length;

      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            plusElements(param.id, param_definitions, "Rooms", {});
          }
        }
      }

      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id);
          }
        }
      }
      setupIndexedSectionTabs("Rooms", "heatpump_number_of_rooms", "Room", "heatpump_room_names", [
        "heatpump_room_names",
        "heatpump_room_temp_sensors",
        "heatpump_room_valve_sensors",
        "heatpump_room_valve_mode",
        "heatpump_room_blind_sensors",
        "heatpump_room_window_sensors",
        "heatpump_room_door_sensors",
        "heatpump_room_min_temperature",
        "heatpump_room_max_temperature",
        "heatpump_room_target_temperature",
        "heatpump_room_nominal_power",
        "heatpump_room_supply_temperature",
        "heatpump_room_volume",
        "heatpump_room_shared_group",
        "heatpump_room_u_value",
        "heatpump_room_envelope_area",
        "heatpump_room_ventilation_rate",
        "heatpump_room_window_area",
        "heatpump_room_shgc",
        "heatpump_room_internal_gains_factor",
        "heatpump_room_thermal_inertia_time_constant",
        "heatpump_room_carnot_efficiency"
      ]);
      normalizeIndexedNames("heatpump_number_of_rooms", "heatpump_room_names", "room");
      attachEntitySuggestions();
      break;

    //if number_of_ev_chargers, number of EV charger array fields should match
    case "number_of_ev_chargers":
      param_list = param_container.getElementsByClassName("param");
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_ev_chargers"
        );
        return 1;
      }
      const firstEVParam = param_container.querySelector(".param");
      if (!firstEVParam) {
        return 1;
      }
      difference =
        Number.parseInt(element.value) -
        firstEVParam.querySelectorAll(".param_input").length;

      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            plusElements(param.id, param_definitions, "EV Charging", {});
          }
        }
      }

      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id);
          }
        }
      }
      setupIndexedSectionTabs("EV Charging", "number_of_ev_chargers", "Charger", "ev_charger_names", [
        "ev_charger_names",
        "ev_phase_mode",
        "ev_charge_mode_service",
        "ev_phase_select_entity",
        "ev_charge_mode_stopped_value",
        "ev_charge_mode_fast_value",
        "ev_charge_mode_eco_value",
        "ev_charge_mode_ecoplus_value",
        "ev_charge_mode_variable_value",
        "ev_phase_select_value_1_phase",
        "ev_phase_select_value_3_phase",
        "ev_phase_select_value_auto",
        "ev_charge_power_min_1_phase",
        "ev_charge_power_max_1_phase",
        "ev_charge_power_min_3_phase",
        "ev_charge_power_max_3_phase"
      ], applyEVVisibility);
      normalizeIndexedNames("number_of_ev_chargers", "ev_charger_names", "ev");
      applyEVVisibility();
      attachEntitySuggestions();
      break;

    //#610: the 15 per-battery array params
    //(BATTERY_ARRAY_PARAMS) in the "Battery" section should add up to
    //number_of_batteries, mirroring the number_of_deferrable_loads case above.
    //Unlike "Deferrable Loads", "Battery" also holds global scalar flags
    //(set_use_battery, set_nocharge_from_grid, set_battery_dynamic, ...) that
    //must NOT be grown/shrunk by this count, so param_list is filtered down to
    //BATTERY_ARRAY_PARAMS rather than every ".param" in the section.
    case "number_of_batteries":
      //get a list of only the battery array params in section
      param_list = Array.from(
        param_container.getElementsByClassName("param")
      ).filter((p) => BATTERY_ARRAY_PARAMS.includes(p.id));
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_batteries"
        );
        return 1;
      }
      //calculate how much off the first battery array param's input elements amount to is, comparing to the number_of_batteries value
      difference =
        Number.parseInt(element.value) -
        param_list[0].querySelectorAll("input").length;
      //add elements based on how many elements are missing
      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            //append element, do not pass config to obtain default parameter from definitions file
            plusElements(param.id, param_definitions, "Battery", {});
          }
        }
      }
      //subtract elements based how many elements its over
      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id);
          }
        }
      }
      break;
  }
}

//checks parameter value in config, updates value if exists
function checkConfigParam(value, config, parameter_definition_name) {
  if (config !== null && config !== undefined) {
    //check if parameter has a saved value
    if (parameter_definition_name in config) {
      value = config[parameter_definition_name];
    }
  }
  return value;
}

function ensureArrayLength(values, length, defaultValue) {
  const out = Array.isArray(values) ? values.slice(0, length) : [];
  while (out.length < length) {
    out.push(defaultValue);
  }
  return out;
}

function parseProgramPowerSequence(raw) {
  if (raw == null) return [];

  const toNumeric = (list) => list
    .map((v) => Number.parseFloat(v))
    .filter((v) => Number.isFinite(v) && v >= 0);

  if (Array.isArray(raw)) {
    if (raw.length > 0 && typeof raw[0] === "object" && raw[0] !== null) {
      for (const item of raw) {
        const seq = parseProgramPowerSequence(item.power_pattern ?? item.sequence ?? item);
        if (seq.length) return seq;
      }
      return [];
    }
    return toNumeric(raw);
  }

  if (typeof raw === "object") {
    if (raw.power_pattern != null) {
      return parseProgramPowerSequence(raw.power_pattern);
    }
    if (raw.programs != null) {
      return parseProgramPowerSequence(raw.programs);
    }
    return [];
  }

  if (typeof raw === "number") {
    return Number.isFinite(raw) && raw >= 0 ? [raw] : [];
  }

  if (typeof raw === "string") {
    const text = raw.trim();
    if (!text) return [];
    try {
      const parsed = JSON.parse(text);
      return parseProgramPowerSequence(parsed);
    } catch {
      return toNumeric(text.split(",").map((s) => s.trim()).filter(Boolean));
    }
  }

  return [];
}

function normalizeDeferrableLoadConfig(config) {
  const numLoads = Number.parseInt(config.number_of_deferrable_loads || "0");
  if (!Number.isFinite(numLoads) || numLoads <= 0) return;

  config.load_type = ensureArrayLength(config.load_type, numLoads, "fixed_power_non_splittable");
  config.load_dispatch_mode = ensureArrayLength(config.load_dispatch_mode, numLoads, "hours");
  config.load_programs = ensureArrayLength(config.load_programs, numLoads, "[]");
  config.required_energy_kwh_of_each_deferrable_load = ensureArrayLength(
    config.required_energy_kwh_of_each_deferrable_load,
    numLoads,
    0.0
  );
  config.nominal_power_of_deferrable_loads = ensureArrayLength(
    config.nominal_power_of_deferrable_loads,
    numLoads,
    0
  );
  config.operating_hours_of_each_deferrable_load = ensureArrayLength(
    config.operating_hours_of_each_deferrable_load,
    numLoads,
    0
  );
  config.treat_deferrable_load_as_semi_cont = ensureArrayLength(
    config.treat_deferrable_load_as_semi_cont,
    numLoads,
    true
  );
  config.load_washdata_enabled = ensureArrayLength(config.load_washdata_enabled, numLoads, false);
  config.load_washdata_device = ensureArrayLength(config.load_washdata_device, numLoads, "");
  config.is_manual_load = ensureArrayLength(config.is_manual_load, numLoads, false);
  config.manual_load_ready_sensor = ensureArrayLength(config.manual_load_ready_sensor, numLoads, "");
  config.manual_load_confirm_power_sensor = ensureArrayLength(
    config.manual_load_confirm_power_sensor,
    numLoads,
    ""
  );
  config.manual_load_program_select_sensor = ensureArrayLength(
    config.manual_load_program_select_sensor,
    numLoads,
    ""
  );
  config.manual_load_deadline_hour = ensureArrayLength(config.manual_load_deadline_hour, numLoads, "");

  for (let i = 0; i < numLoads; i++) {
    const type = config.load_type[i];
    let mode = String(config.load_dispatch_mode[i] || "").trim();
    if (!["hours", "program", "energy_kwh"].includes(mode)) {
      mode = type === "program_based" ? "program" : "hours";
    }
    config.load_dispatch_mode[i] = mode;

    if (type === "program_based") {
      const sequence = parseProgramPowerSequence(config.load_programs[i]);
      if (sequence.length) {
        config.nominal_power_of_deferrable_loads[i] = sequence;
        config.operating_hours_of_each_deferrable_load[i] = sequence.length;
        config.load_dispatch_mode[i] = "program";
      } else if (config.load_dispatch_mode[i] === "program") {
        config.load_dispatch_mode[i] = "hours";
      }
      config.treat_deferrable_load_as_semi_cont[i] = true;
      continue;
    }

    const nominal = config.nominal_power_of_deferrable_loads[i];
    const nominalScalar = Array.isArray(nominal)
      ? Number.parseFloat(nominal[0] || 0)
      : Number.parseFloat(nominal || 0);
    config.nominal_power_of_deferrable_loads[i] = Number.isFinite(nominalScalar)
      ? nominalScalar
      : 0;

    if (type === "fixed_power_splittable" || type === "variable_power_variable_time") {
      config.treat_deferrable_load_as_semi_cont[i] = false;
    } else {
      config.treat_deferrable_load_as_semi_cont[i] = true;
    }
  }
}

//send all parameter input values to EMHASS, to save to config.json and param.pkl
async function saveConfiguration(param_definitions) {
  //start wth none
  let config = {};
  let param_inputs
  let param_element

  //if section-cards (config sections/list) exists
  let config_card = document.getElementsByClassName("section-card");
  //check if page is in list or box view
  let config_box_element = document.getElementById("config-box");

  //if true, in list view
  if (Boolean(config_card.length)) {
    //retrieve params and their input/s by looping though param_definitions list
    //loop through the sections
    for (const [, section_object] of Object.entries(
      param_definitions
    )) {
      //loop through parameters
      for (let [
        parameter_definition_name,
        parameter_definition_object,
      ] of Object.entries(section_object)) {
        let param_values = []; //stores the obtained param input values
        let param_array = false;
        //get param container
        param_element = document.getElementById(parameter_definition_name);
        if (param_element == null) {
          console.debug(
            "unable to find " +
              parameter_definition_name +
              " param div container element, skipping this param"
          );
        }
        //extract input/s and their value/s from param container div
        else {
          if (param_element.tagName !== "INPUT") {
            param_inputs = param_element.getElementsByClassName("param_input");
          } else {
            //check if param_element is also param_input (ex. for header parameters)
            param_inputs = [param_element];
          }

          // loop though param_inputs, extract the element/s values
          for (let input of param_inputs) {
            switch (input.type) {
              case "number":
                param_values.push(Number.parseFloat(input.value));
                break;
              case "checkbox":
                param_values.push(input.checked);
                break;
              default:
                param_values.push(input.value);
                break;
            }
          }
          //obtain param input type from param_definitions, check if param should be formatted as an array
          param_array = Boolean(
            !parameter_definition_object["input"].search("array")
          );

          //build parameters using values extracted from param_inputs

          // object-type: JSON.parse the text-box value; treat "" and "null" as JSON null
          if (parameter_definition_object["input"] === "object") {
            const raw = (param_values[0] ?? "").toString();
            if (raw === "" || raw === "null") {
              config[parameter_definition_name] = null;
            } else {
              try {
                config[parameter_definition_name] = JSON.parse(raw);
              } catch (_) {
                errorAlert(parameter_definition_name + ": invalid JSON — please check the value and try again.");
                return 0;
              }
            }
            continue;
          }

          // If time with 2 sets (load_peak_hour_periods)
          if (
            parameter_definition_object["input"] == "array.time" &&
            param_values.length % 2 === 0
          ) {
            config[parameter_definition_name] = {};
            for (let i = 0; i < param_values.length; i++) {
              config[parameter_definition_name][
                "period_hp_" +
                  (Object.keys(config[parameter_definition_name]).length + 1)
              ] = [{ start: param_values[i] }, { end: param_values[++i] }];
            }
            continue;
          }

          //single value
          if (param_values.length && !param_array) {
            config[parameter_definition_name] = param_values[0];
          }

          //array value
          else if (param_values.length) {
            config[parameter_definition_name] = param_values;
          }
        }
      }
    }
  }

  //if box view, extract json from box view
  else if (config_box_element !== null) {
    //try and parse json from box
    try {
      config = JSON.parse(config_box_element.value);
    } catch (error) {
      //if json error, show in alert box
      document.getElementById("alert-text").textContent =
        "\r\n" +
        error +
        "\r\n" +
        "JSON Error: String values may not be wrapped in quotes";
      document.getElementById("alert").style.display = "block";
      document.getElementById("alert").style.textAlign = "center";
      return 0;
    }
  }
  // else, cant find box or list view
  else {
    errorAlert("There has been an error verifying box or list view");
  }

  normalizeDeferrableLoadConfig(config);

  //finally, send built config to emhass
  const response = await fetch(`set-config`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
  showChangeStatus(response.status, await response.json());
}

//Toggle between box (json) and list view
async function ToggleView(param_definitions, list_html, default_reset) {
  let selected = "";
  config = {};

  //find out if list or box view is active
  let configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //get yaml button
  let yaml_button = document.getElementById("yaml");
  if (yaml_button == null) {
    console.log("Unable to obtain yaml button");
  }

  // if section-cards (config sections/list) exists
  let config_card = configuration_container.getElementsByClassName("section-card");
  //selected view (0 = box)
  let selected_view = Boolean(config_card.length);

  //if default_reset is passed do not switch views, instead reinitialize view with default config as values
  if (default_reset) {
    selected_view = !selected_view;
    //obtain default config as config (when pressing the default button)
    config = await ObtainDefaultConfig();
  } else {
    //obtain latest config
    config = await obtainConfig();
  }

  //if array is empty assume json box is selected
  if (selected_view) {
    selected = "list";
  } else {
    selected = "box";
  }
  //remove contents of current view
  configuration_container.innerHTML = "";
  //build new view
  switch (selected) {
    case "box":
      //load list
      loadConfigurationListView(param_definitions, config, list_html);
      yaml_button.style.display = "none";
      break;
    case "list":
      //load box
      loadConfigurationBoxPage(config);
      yaml_button.style.display = "block";
      break;
  }
}

//load box (json textarea) view
async function loadConfigurationBoxPage(config) {
  //get configuration container element
  let configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //append configuration container with textbox area
  configuration_container.innerHTML = `
      <textarea id="config-box" rows="30" placeholder="{}"></textarea>
      `;
  //set created textarea box with retrieved config
  document.getElementById("config-box").innerHTML = JSON.stringify(
    config,
    null,
    2
  );
}

//function in control of status icons and alert box from a fetch request
async function showChangeStatus(status, logJson) {
  let loading = document.getElementById("loader"); //element showing statuses
  if (loading === null) {
    console.log("unable to find loader element");
    return 1;
  }
  if (status === 200 || status === 201) {
    //if status is 200 or 201, then show a tick
    loading.innerHTML = `<p class=tick>&#x2713;</p>`;
  } else {
    //then show a cross
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=cross>&#x292C;</p>`; //show cross icon to indicate an error
    if (logJson.length != 0 && document.getElementById("alert-text") !== null) {
      document.getElementById("alert-text").textContent =
        "\r\n\u2022 " + logJson.join("\r\n\u2022 "); //show received log data in alert box
      document.getElementById("alert").style.display = "block";
      document.getElementById("alert").style.textAlign = "left";
    }
  }
  //remove tick/cross after some time
  setTimeout(() => {
    loading.innerHTML = "";
  }, 4000);
}

//simple function to write text to the alert box
async function errorAlert(text) {
  if (
    document.getElementById("alert-text") !== null &&
    document.getElementById("alert") !== null
  ) {
    document.getElementById("alert-text").textContent = "\r\n" + text + "\r\n";
    document.getElementById("alert").style.display = "block";
    document.getElementById("alert").style.textAlign = "left";
  }
  return 0;
}

//convert yaml box into json box
async function yamlToJson() {
  //get box element
  let config_box_element = document.getElementById("config-box");
  if (config_box_element == null) {
    errorAlert("Unable to obtain config box");
  } else {
    const response = await fetch(`get-json`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: config_box_element.value,
    });
    let response_status = response.status; //return status
    if (response_status == 201) {
      showChangeStatus(response_status, {});
      let blob = await response.blob(); //get data blob
      config = await new Response(blob).json(); //obtain json from blob
      config_box_element.value = JSON.stringify(config, null, 2);
    } else {
      showChangeStatus(response_status, await response.json());
    }
  }
  return 0;
}