"""Thermal-mass physics core: a lumped RC-network model of indoor room temperature.

Main states:
- T_air: predicted indoor air temperature
- T_mass: hidden building/floor thermal mass temperature
- Q_emit: delayed heat-emitter state driven by duty * max(supply - T_air, 0)
- T_wall: hidden EXTERIOR-wall-surface temperature, sun-heated independent of
  blinds (see below) - a room's own thermal order this way ends up in the
  3R3C/4R4C family used in the building-grey-box literature (see e.g.
  Reynders et al. 2014, "impact of the model structure on the identified
  parameters and the predictive performance of RC building models" - our
  own topology is a "star" (T_air couples to both outdoor and T_mass
  directly, T_mass to T_air and now T_wall) rather than the more common
  "series" outdoor-mass-air chain, and couplings are independently-gained
  rather than reciprocal single-resistor values - a data-driven state-space
  model inspired by RC networks, not a strictly circuit-equivalent one).

Two distinct solar-gain pathways, split because they differ in BOTH timing
and controllability:
- Window-transmitted gain ("solar_*" params) - the ONLY pathway gated by
  blind_position (0=open..1=closed), since interior blinds sit between the
  window and the room and can only ever block light that would otherwise
  pass THROUGH the window. Itself split, via window_solar_radiative_fraction,
  into a convective share (fed straight into T_air's own rate, no lag - a
  room's light interior surfaces/furniture re-radiate to air quickly) and a
  radiative share that lands on T_mass instead - the convective/radiative
  split ISO 13790's 5R1C model and most grey-box RC building-identification
  papers use, since direct-beam sun through a window typically strikes the
  floor (real thermal mass), not air. At fraction=0 this collapses to the
  pre-split behavior (100% convective, straight to T_air).
- Opaque-exterior-wall-absorbed gain ("wall_*" params) - heats the
  building's own outer surface directly, completely unaffected by any
  interior shading device (blinds have no way to reach the outside of the
  building), and reaches the room only after conducting inward through the
  wall's own thermal mass (wall_tau_h) and partially dragging T_mass
  (wall_to_mass_weight) rather than T_air directly.

Solar gain can be summed from up to 3 independently-oriented facades
("facade1"/facade2/facade3 - facade1 is the original, always-active
`facade_azimuth_deg`/`facade_tilt_deg` pair; facade2/facade3 are optional
extra slots, each contributing `facadeN_weight * poa_N` to the combined
plane-of-array signal before the horizontal/facade blend). This mirrors
ISO 13790's own 5R1C reference method (and its Modelica Buildings library
implementation), which computes solar gain separately per window/wall
orientation group and sums the contributions before distributing into the
zone's air/mass nodes - the natural way to represent a room whose windows
don't all face the same direction (e.g. a big south window plus a
near-horizontal dakraam plus a north window). facade2_weight/facade3_weight
are configured constants (proportional to each slot's relative window
area), never fitted - unlike orientation itself, weight isn't something a
temperature-only fit can identify, and turning it into another free
parameter would only make an already data-hungry fit harder. Both default
to 0.0 (slot disabled), so a house that never configures them gets exactly
the single-orientation behavior.

Orientation itself (all 6 of facade1/facade2/facade3's own azimuth/tilt)
stays genuinely self-learning even when the user configures a known/
estimated value - see _fit_temperature_params's own regularization_overrides
parameter. A configured value is never hard-pinned (that would mean a
wrong guess or typo can never be corrected, and this codebase's whole
design principle - state explicitly by the user this feature was built
for - is "configurable when known, but still self-learning, including
finding a better answer than the configured one if real data disagrees
strongly enough"); instead it becomes a much stronger regularisation
anchor than the mild default pull an unconfigured slot gets, i.e. "hard
to move away from" rather than "impossible to move away from". This is
the same anchor+weight shape category-based priors use too: mass_tau_h
(building-mass-class -> mass_tau_h_anchor_from_building_class, ISO/FDIS
13790:2007 Table 12's Cm-per-floor-area figures, used only as a RATIO
against the "medium" class - floor area itself cancels exactly out of that
ratio, see mass_tau_h_anchor_from_building_class's own docstring, so it is
never needed as an input) and tau_emit_h (heat-emitter type ->
tau_emit_h_anchor_from_emitter_type, rough HVAC vuistregel response-time
estimates - convector/radiator tens of minutes, floor heating several
hours - meaningfully weaker grounding than the ISO table, and only sound
for a zone where a SINGLE emitter type covers every room this sensor
measures, since there is only one fitted tau_emit_h for the whole zone).
Unlike the 6 facade-orientation terms (which always have SOME default
anchor, even unconfigured), mass_tau_h/tau_emit_h get no regularisation
term at all unless the user actually configures a value - fully free,
exactly today's pre-existing behavior, when left unset. wall_tau_h (the
separate EXTERIOR wall surface - see above) deliberately has no such
prior: it would need wall area (far less commonly known than floor area)
and a wall-construction kappa from a different table (ISO 13786), not
sourced here.

This module holds the simulation core (inputs container, fit-parameter schema,
the open-loop stepper) plus the fitting routine (_fit_temperature_params) so
both can be shared between the offline research script
(scripts/thermal_mass_physics_model.py) and the live EMHASS actions
(command_line.compute_heating_forecast, command_line.refit_heating_model).
CSV/report I/O and plotting stay in the script - this module has no
dependency on sklearn or any CLI/file-loading code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from pvlib.location import Location
except Exception:  # pragma: no cover - fallback for environments without pvlib
    Location = None


@dataclass
class ThermalInputs:
    index: pd.DatetimeIndex
    room: np.ndarray
    electric: np.ndarray
    gas: np.ndarray
    duty: np.ndarray
    supply: np.ndarray
    outdoor: np.ndarray
    wind_speed: np.ndarray
    wind_sin: np.ndarray
    wind_cos: np.ndarray
    sun_alt_sin: np.ndarray
    sun_alt_cos: np.ndarray
    sun_az_sin: np.ndarray
    sun_az_cos: np.ndarray
    heatpump_duty: np.ndarray
    # Room's own live blind/shading position (0=open..1=closed) - only ever
    # gates the WINDOW solar pathway (see module docstring), never the
    # exterior-wall one. Defaults to None (resolved to all-zeros/open by
    # _simulate_open_loop) both when no sensor is configured (via
    # _prepare_inputs's own 0.0 default) AND for any caller outside this
    # module that still constructs ThermalInputs directly without knowing
    # about this field (e.g. scripts/cvxpy_state_space_thermal_model.py's
    # own, unrelated state-space model) - exactly recovering pre-blind-
    # support behavior either way.
    blind_position: np.ndarray | None = None
    # Door/window open state (0=closed..1=open) - an extra ventilation-loss
    # gate on top of the existing wind-driven envelope loss (see
    # door_open_extra_loss_per_h). Same optional/None-default treatment as
    # blind_position, for the same backward-compat reasons.
    door_open: np.ndarray | None = None
    # Raw irradiance components - q_solar (the facade-projected, blended
    # solar proxy) is no longer precomputed once outside the fit loop,
    # since facade_azimuth_deg/facade_tilt_deg are now FITTABLE parameters
    # (see PARAM_NAMES) rather than fixed kwargs, so the plane-of-array
    # projection has to happen per-params-guess inside
    # _simulate_open_loop/_simulate_segmented instead. Optional/None-default
    # for the same backward-compat reasons as blind_position/door_open.
    ghi: np.ndarray | None = None
    dni: np.ndarray | None = None
    dhi: np.ndarray | None = None
    # Legacy field, kept ONLY so scripts/cvxpy_state_space_thermal_model.py's
    # own direct ThermalInputs(...) construction (an unrelated state-space
    # model, still passes q_solar= explicitly at 2 call sites) doesn't break -
    # unused by this module's own simulate/fit functions going forward.
    q_solar: np.ndarray | None = None


@dataclass
class SimResult:
    room: np.ndarray
    air_before: np.ndarray
    mass: np.ndarray
    q_emit: np.ndarray
    wall: np.ndarray
    q_solar: np.ndarray
    loss_coeff: np.ndarray
    # Predicted electric power (W) - q_emit * emitter_power_scale_w / COP,
    # see _cop_carnot_vectorized. Computed unconditionally (cheap, one
    # vectorized expression after the main per-timestep loop) even when
    # fit_electric_power is off - harmless in that case since
    # emitter_power_scale_w sits at its DEFAULT_X0 seed of 0.0, giving an
    # all-zero array no caller reads. When fit_gas_consumption is True,
    # this becomes the heat-pump-delivered share only (capacity-capped) -
    # see gas_pred below.
    electric_pred: np.ndarray
    # Predicted gas consumption (m3/interval) - the capacity-split's gas
    # share, converted via boiler_efficiency/GAS_CALORIFIC_VALUE_WH_PER_M3.
    # None unless fit_gas_consumption=True was passed to
    # _simulate_open_loop - see that parameter's own docstring for the
    # bivalent-parallel capacity-split this and electric_pred (above)
    # jointly implement.
    gas_pred: np.ndarray | None = None


PARAM_NAMES = [
    "tau_emit_h",
    "emit_gain_per_h",
    "ua_base_per_h",
    "ua_wind_per_h_per_speed",
    "ua_wind_sin_per_h_per_speed",
    "ua_wind_cos_per_h_per_speed",
    "mass_tau_h",
    "mass_gain_per_h",
    "solar_gain_c_per_h",
    "solar_alt_sin_gain_c_per_h",
    "solar_alt_cos_gain_c_per_h",
    "solar_az_sin_gain_c_per_h",
    "solar_az_cos_gain_c_per_h",
    "bias_c_per_h",
    # Appended (not inserted) so _fit_temperature_params's own regularisation
    # array, which indexes the first 14 params by fixed position, needs no
    # changes. See module docstring for what these represent.
    "wall_tau_h",
    "wall_solar_gain_c",
    "wall_to_mass_weight",
    # Extra ventilation-loss coefficient while door_open is nonzero (see
    # ThermalInputs.door_open) - additive on top of direction_loss, same
    # "append, weakly identified without real signal" treatment as
    # wall_to_mass_weight above.
    "door_open_extra_loss_per_h",
    # Fraction (0-1) of window-transmitted solar heat (solar_gain_c_per_h
    # and its 4 sun-direction harmonics above) that lands on interior
    # thermal mass (floor/furniture) rather than air directly - the
    # convective/radiative split ISO 13790's 5R1C model and most grey-box
    # RC building-identification papers use (direct-beam sun typically
    # strikes the floor, which has real thermal mass, not air). At 0.0
    # this exactly recovers the pre-split behavior (100% convective,
    # straight into d_air_dt) - see _simulate_open_loop's own comments for
    # where this actually splits the term. Unlike wall_to_mass_weight this
    # is NOT regularised toward 0: window solar is a strong, frequently-
    # observed signal (unlike the sparse wall/door channels), so it's
    # treated as a core, freely-fit parameter like solar_gain_c_per_h
    # itself.
    "window_solar_radiative_fraction",
    # Assumed window/facade orientation for the plane-of-array solar
    # projection (see _compute_poa_solar below) - previously a fixed
    # kwarg (facade_azimuth_deg=180/facade_tilt_deg=90, i.e. due south,
    # vertical) hardcoded at both command_line.py call sites, with no way
    # to either tell the model the real orientation or let it learn one.
    # Now genuinely fittable: defaults recover that exact old assumption
    # (so an unfit/never-refit deployment behaves identically to today),
    # and refit_heating_model can instead FIX these via
    # _fit_temperature_params's fixed_overrides when the user has
    # configured a real, known orientation (heatpump_facade_azimuth_deg/
    # heatpump_facade_tilt_deg) - a known fact always wins over inference,
    # same precedence principle as door/blind sensors overriding
    # relabeling. Mildly regularised toward the south/vertical default
    # below (same treatment as wall_to_mass_weight) as a safety net for
    # a fit too weak to truly pin down orientation - not a hard constraint.
    "facade_azimuth_deg",
    "facade_tilt_deg",
    # Two optional extra orientation slots ("facade2"/"facade3"), each only
    # contributing to q_solar when its matching facadeN_weight kwarg (a
    # configured constant, never fitted - see module docstring) is nonzero.
    # Same [0,360]/[0,90] bounds and south/vertical-default regularization
    # treatment as facade_azimuth_deg/facade_tilt_deg above, just applied to
    # each slot's OWN DEFAULT_X0 seed rather than a hardcoded 180/90 - there's
    # no universally "correct" default for a secondary orientation the way
    # south is a reasonable prior for a single dominant one.
    "facade2_azimuth_deg",
    "facade2_tilt_deg",
    "facade3_azimuth_deg",
    "facade3_tilt_deg",
    # Electric-power bridge (appended, not inserted, same convention as
    # above) - OFF by default (see _fit_temperature_params's own
    # fit_electric_power docstring): both stay pinned at their DEFAULT_X0
    # seed with zero effect on the fit unless fit_electric_power=True
    # actually appends an electric-power residual block, since with no
    # such block they have zero gradient contribution to explore. Carnot-
    # lift COP formula and default mirror utils.calculate_cop_heatpump/
    # heatpump_room_carnot_efficiency's own existing 0.4 default - see
    # that function's docstring for the "typical 0.35-0.50" real-world-
    # efficiency range this is drawn from.
    "carnot_efficiency",
    # Converts q_emit (a duty x temp-lift PROXY, not literal Watts - this
    # model folds the room's own effective thermal capacitance into
    # emit_gain_per_h/mass_gain_per_h/etc. rather than tracking it as its
    # own state) into real thermal Watts. q_emit's own natural scale
    # varies a lot house to house, so this bound is deliberately wide -
    # to be tightened against real data, same "validate empirically,
    # don't just guess once" precedent as mass_tau_h/mass_gain_per_h/
    # wall_solar_gain_c's own UPPER_BOUNDS history below.
    "emitter_power_scale_w",
    # How much duty's effectiveness at producing heat scales with the
    # instantaneous Carnot-lift COP (see _cop_carnot_vectorized), relative
    # to a fixed reference COP evaluated at _COP_REFERENCE_OUTDOOR_C (not a
    # separately fitted/persisted "fit-window mean COP" - deliberately
    # deterministic from carnot_efficiency + supply_temperature alone, so it
    # needs no new persisted state and stays consistent between fit-time and
    # dispatch-time). Appended (not inserted), same convention as every
    # parameter above. Deliberately FITTABLE rather than a hand-picked
    # constant: emit_gain_per_h was already fit against real historical
    # duty/room_temp data, so whatever COP variation happened during that
    # fit window is already smeared into it - a hand-picked COP multiplier
    # bolted on afterward would risk double-counting that effect, whereas
    # letting the fit itself determine cop_sensitivity jointly with
    # emit_gain_per_h avoids that (the fit only pushes this away from 0 if
    # real temperature-residual data actually supports it, same
    # "regularized toward a neutral default" treatment as
    # wall_to_mass_weight/door_open_extra_loss below). 0.0 (this param's own
    # DEFAULT_X0 seed) means cop_scale=1.0 always - byte-identical to the
    # pre-this-parameter behavior, unlike carnot_efficiency/
    # emitter_power_scale_w this needs no fit_electric_power gate, since it
    # affects room temperature directly (the always-active core residual),
    # not just the opt-in electric-power one.
    "cop_sensitivity",
    # Bivalent-parallel gas-boiler bridge (appended, same convention as
    # every parameter above) - gated behind the NEW fit_gas_consumption
    # flag (a real bool parameter threaded through _simulate_open_loop/
    # _simulate_segmented/_fit_temperature_params, NOT inferred from these
    # values - see the module's own gas-split docstring below for why a
    # sentinel-default approach was rejected). Requires
    # fit_electric_power=True too (the split needs the COP/electric
    # machinery already active). heatpump_capacity_ref_w/
    # heatpump_capacity_slope_w_per_c together form a linear heat-pump
    # max-heat-output curve (heat pumps deliver less peak capacity as it
    # gets colder) anchored at the SAME _COP_REFERENCE_OUTDOOR_C (5degC)
    # cop_sensitivity already uses - demand up to that curve is served by
    # the heat pump alone; demand beyond it is made up by gas, in Watts of
    # THERMAL heat (same units emitter_power_scale_w already converts
    # q_emit into) - never a third "heat pump fully off" stage. All three
    # pinned at their own DEFAULT_X0 via fixed_overrides when
    # fit_gas_consumption is off, same "harmless when off" treatment
    # carnot_efficiency/emitter_power_scale_w already get.
    "heatpump_capacity_ref_w",
    "heatpump_capacity_slope_w_per_c",
    # Gas boiler combustion efficiency - converts the gas-delivered share
    # of the heat split (Watts thermal) into gas consumption (m3/interval)
    # via GAS_CALORIFIC_VALUE_WH_PER_M3 below. Regularised toward its own
    # 0.90 DEFAULT_X0 when fit_gas_consumption is on (see
    # _fit_temperature_params) - capacity and efficiency both scale the
    # gas residual, an unregularised joint fit has the exact same flat-
    # valley degeneracy already documented for carnot_efficiency/
    # emitter_power_scale_w.
    "boiler_efficiency",
]

LOWER_BOUNDS = np.array(
    [0.25, 0.0, 0.0, 0.0, -0.02, -0.02, 2.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -0.20, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, -0.3, 0.0, -500.0, 0.5],
    dtype=float,
)
UPPER_BOUNDS = np.array(
    # mass_tau_h/mass_gain_per_h/wall_solar_gain_c were widened once (240->500,
    # 0.40->0.80, 30->50) after a real refit pinned all three at their old
    # bounds with window_solar_radiative_fraction newly feeding real solar
    # heat into T_mass - but a re-test at the wider bounds just ran further
    # (mass_tau_h~458, wall_solar_gain_c~47.5, window_solar_radiative_fraction
    # ~0.99) with NO clear MAE improvement - the classic signature of a
    # genuine flat/unidentifiable direction (once mass_tau_h is huge, mass
    # barely moves within the window, so window_solar_radiative_fraction and
    # wall_to_mass_weight can drift arbitrarily without affecting the fit),
    # not a too-tight bound. Reverted to the original, more physically
    # plausible values. facade_azimuth_deg spans the full compass (0-360);
    # facade_tilt_deg spans horizontal-to-vertical (0-90, pvlib's own range) -
    # same bounds reused for facade2/facade3. carnot_efficiency capped at 0.7
    # (calculate_cop_heatpump's own "typical 0.35-0.50" range, widened for
    # safety); emitter_power_scale_w capped at 20000 (generously covers
    # residential heat-pump nominal power divided by q_emit's own plausible
    # scale) - both unvalidated placeholders pending a real-data check.
    # cop_sensitivity capped at +-0.3: at a typical ~4-unit COP spread (COP
    # roughly 1-8 per _cop_carnot_vectorized's own clip), this keeps
    # cop_scale within a generous but bounded ~0.1-2.2 range rather than
    # letting a single weakly-identified coefficient swing emit_raw by an
    # unbounded factor. heatpump_capacity_ref_w capped at 30000 (generous
    # residential range, same "unvalidated placeholder" caveat as
    # emitter_power_scale_w above); heatpump_capacity_slope_w_per_c at
    # +-500 (a few hundred W of capacity change per degC is a plausible
    # real range, sign not assumed - a real air-source heat pump loses
    # capacity as it gets colder, i.e. a positive fitted value, but the
    # fit decides, not a hardcoded assumption); boiler_efficiency at 1.0
    # (plain combustion-efficiency convention, not a condensing-boiler
    # LHV>100% edge case).
    [12.0, 0.8, 0.25, 0.03, 0.02, 0.02, 240.0, 0.40, 3.0, 2.0, 2.0, 2.0, 2.0, 0.20, 48.0, 30.0, 1.0, 1.0, 1.0, 360.0, 90.0, 360.0, 90.0, 360.0, 90.0, 0.7, 20000.0, 0.3, 30000.0, 500.0, 1.0],
    dtype=float,
)
DEFAULT_X0 = np.array(
    # facade2 seeded at due-north/vertical, facade3 at due-east/vertical -
    # arbitrary but deliberately DIFFERENT from facade1's south seed and
    # from each other, purely so an enabled-but-unconfigured slot doesn't
    # start the search exactly on top of facade1's own optimum. Irrelevant
    # whenever the matching facadeN_weight stays at its own 0.0 default
    # (slot disabled). carnot_efficiency seeded at 0.4, matching
    # heatpump_room_carnot_efficiency's own existing default;
    # emitter_power_scale_w seeded at 0.0 - harmless (electric_pred is
    # just 0.0 everywhere) whenever fit_electric_power stays off, and a
    # neutral starting point for the search when it's on. cop_sensitivity
    # seeded at 0.0 - cop_scale=1.0 everywhere, exactly recovering pre-
    # cop_sensitivity behavior until a real fit finds a nonzero value.
    # heatpump_capacity_ref_w seeded at 8000 (a plausible mid-size
    # residential heat pump, purely a starting point when fit_gas_consumption
    # is on - pinned exactly here via fixed_overrides, never searched, when
    # it's off); heatpump_capacity_slope_w_per_c at 0.0 (no temperature
    # dependence assumed until real data supports one); boiler_efficiency
    # at 0.90 (typical modern condensing-boiler combustion efficiency).
    [2.5, 0.08, 0.025, 0.0015, 0.0, 0.0, 48.0, 0.04, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 0.3, 0.0, 0.5, 180.0, 90.0, 0.0, 90.0, 90.0, 90.0, 0.4, 0.0, 0.0, 8000.0, 0.0, 0.90],
    dtype=float,
)

# Fixed reference outdoor temperature (degC) cop_sensitivity's own COP
# scaling is anchored against (see PARAM_NAMES's own cop_sensitivity
# docstring for why this is a deterministic anchor rather than a persisted
# fit-window statistic) - matches calculate_cop_heatpump's own docstring
# example of a "typical mild" outdoor point, nothing more significant than
# that; any fixed, reasonable anchor works equally well since cop_sensitivity
# itself is fit to compensate for whatever anchor is chosen.
_COP_REFERENCE_OUTDOOR_C = 5.0

# Natural-gas higher heating value, Wh per m3 - Dutch/Groningen-quality gas
# (~9.77 kWh/m3), the near-universal reference for a Dutch installation's
# own gas meter reading. A fixed, known physical fact (same "configured
# constant, never fitted" category as hybrid_heatpump_lr.py's own
# bivalent_point/hp_rated_norm), not something to learn from data - could
# become a config knob for a different gas quality/region later, no
# evidence that's needed yet.
GAS_CALORIFIC_VALUE_WH_PER_M3 = 9770.0

# Regularisation weights for any parameter named in regularization_overrides
# (see _fit_temperature_params's own docstring): the usual mild pull toward
# a default anchor, vs. 10x stronger when the user has actually configured
# a value for that specific parameter - "hard to move away from", not a
# hard pin (fixed_overrides remains the mechanism for an absolute pin).
# Originally added for the 6 facade-orientation parameters, reused as-is
# for mass_tau_h/tau_emit_h below - same anchor+weight shape, just a
# different physical quantity and a different source for the anchor value.
_DEFAULT_PRIOR_REG_WEIGHT = 0.03
_CONFIGURED_PRIOR_REG_WEIGHT = 0.3

# Relative weight of the (opt-in) electric-power residual block against the
# temperature residual block once both are normalized to comparable scales
# (degC for temperature; MAD-normalized for electric power - see
# _fit_temperature_params's own fit_electric_power docstring). 1.0 (equal
# footing) is a placeholder pending real-data validation, same "pick a
# reasonable default now, validate before treating it as final" precedent
# as this module's own phase-robust/warm-start additions.
_ENERGY_FIT_WEIGHT = 1.0

# ISO/FDIS 13790:2007 Table 12 (Annex 12.3.1.2, read directly from the
# standard) - internal heat capacity Cm per unit floor area, J/(K*m2), by
# building "heat capacity class". Cm = BUILDING_MASS_CLASS_CM[class] *
# floor_area - but floor_area is NEVER needed here: without an independently
# known envelope heat-loss rate (H_tr+H_ve - circular, since ua_base_per_h
# is exactly what THIS model fits that from), an absolute tau = Cm/(3600*H)
# can't be computed, so the only sound use of this table is a RATIO against
# BUILDING_MASS_CLASS_CM["medium"] (DEFAULT_X0's own implicit class) - and
# floor_area cancels exactly out of that ratio for the same building. Maps
# to mass_tau_h specifically (T_mass = INTERIOR thermal mass/floor/
# furniture) - NOT wall_tau_h (the separate EXTERIOR wall surface, whose
# own thermal response depends on wall area and a wall-construction-
# specific kappa from ISO 13786, a different table not sourced here).
BUILDING_MASS_CLASS_CM = {
    "very_light": 80_000.0,
    "light": 110_000.0,
    "medium": 165_000.0,
    "heavy": 260_000.0,
    "very_heavy": 370_000.0,
}
# Rough HVAC vuistregel (rule-of-thumb) heat-emitter response-time
# estimates - NOT from a standard, much weaker grounding than
# BUILDING_MASS_CLASS_CM above. Convector/radiator: consumer-guide
# reported warm-up/cool-down times of tens of minutes; floor heating:
# hours (thick screed, large thermal mass) - see module docstring.
EMITTER_TAU_H_ESTIMATE = {
    "convector": 0.5,
    "radiator": 0.75,
    "floor_heating": 5.0,
}


def mass_tau_h_anchor_from_building_class(building_class: str) -> float | None:
    """DEFAULT_X0's own mass_tau_h scaled by this class's Cm ratio to
    "medium" - None for "" or an unrecognised string, so callers can skip
    adding a regularization_overrides entry entirely."""
    cm = BUILDING_MASS_CLASS_CM.get(building_class)
    if cm is None:
        return None
    return float(DEFAULT_X0[PARAM_NAMES.index("mass_tau_h")] * cm / BUILDING_MASS_CLASS_CM["medium"])


def tau_emit_h_anchor_from_emitter_type(emitter_type: str) -> float | None:
    """EMITTER_TAU_H_ESTIMATE[emitter_type] - None for "" or an
    unrecognised string. Only meaningful when a SINGLE emitter type covers
    the whole zone this indoor-temperature sensor measures - a mixed zone
    (e.g. different rooms on convector/floor heating/radiator, all feeding
    the same fitted tau_emit_h) has no single right answer here, so callers
    should leave this unconfigured rather than guess for that case."""
    return EMITTER_TAU_H_ESTIMATE.get(emitter_type)


def _infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.25
    diffs = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.25
    return float(np.median(diffs))


def _series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        return dt_index.tz_localize("UTC")
    return dt_index.tz_convert("UTC")


def _compute_sun_direction_features(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return sin/cos decompositions of solar altitude and azimuth."""
    if Location is None:
        zeros = pd.Series(0.0, index=df.index, dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()

    try:
        location = Location(latitude=latitude, longitude=longitude, tz="UTC")
        solar_position = location.get_solarposition(_utc_index(df.index))
        altitude_rad = np.radians(90.0 - solar_position["apparent_zenith"].to_numpy(dtype=float))
        azimuth_rad = np.radians(solar_position["azimuth"].to_numpy(dtype=float))
        alt_sin = pd.Series(np.sin(altitude_rad), index=df.index).clip(lower=0.0)
        alt_cos = pd.Series(np.cos(altitude_rad), index=df.index)
        az_sin = pd.Series(np.sin(azimuth_rad), index=df.index)
        az_cos = pd.Series(np.cos(azimuth_rad), index=df.index)
        return alt_sin, alt_cos, az_sin, az_cos
    except Exception:
        zeros = pd.Series(0.0, index=df.index, dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()


def _facade_trig(facade_azimuth_deg: float, facade_tilt_deg: float) -> tuple[float, float, float, float]:
    """(cos_tilt, sin_tilt, cos_az, sin_az) for a candidate facade
    orientation - computed ONCE per simulate/residuals call (facade
    orientation is now a fitted parameter, constant across all timesteps
    within one call), not per-timestep, matching _facade_poa_scalar/
    _facade_poa_vectorized's own "params are per-call constants" contract.
    """
    tilt_rad = np.radians(facade_tilt_deg)
    az_rad = np.radians(facade_azimuth_deg)
    return float(np.cos(tilt_rad)), float(np.sin(tilt_rad)), float(np.cos(az_rad)), float(np.sin(az_rad))


def _cop_carnot_vectorized(
    carnot_efficiency: float, supply_c: np.ndarray, outdoor_c: np.ndarray
) -> np.ndarray:
    """Carnot-lift COP - the SAME formula as utils.calculate_cop_heatpump
    (COP = carnot_efficiency * T_supply_K / (T_supply_K - T_outdoor_K),
    clipped to [1, 8]) - deliberately reimplemented inline rather than
    calling that function directly: it logs a warning (via its own module
    logger) on every non-physical timestep, fine for a one-off forecast
    call but this runs inside _simulate_open_loop/_simulate_segmented,
    themselves inside least_squares's residual function - evaluated
    potentially thousands of times per fit (Jacobian finite differences x
    however many restarts). No warning here for the same reason
    _facade_poa_scalar's own docstring gives for its plain min/max: hot-
    path performance, not because the non-physical case doesn't matter -
    it's silently clamped to COP=1.0 instead, same as the source function.
    """
    supply_k = supply_c + 273.15
    outdoor_k = outdoor_c + 273.15
    lift = supply_k - outdoor_k
    cop = np.where(lift > 0.0, carnot_efficiency * supply_k / np.maximum(lift, 1e-6), 1.0)
    return np.clip(cop, 1.0, 8.0)


def _split_heat_pump_gas_w(
    q_heat_total_w: np.ndarray,
    outdoor: np.ndarray,
    heatpump_capacity_ref_w: float,
    heatpump_capacity_slope_w_per_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bivalent-parallel split of total delivered heat (Watts, thermal -
    the SAME units emitter_power_scale_w already converts q_emit into)
    into a heat-pump-delivered share and a gas-delivered share - shared
    by _simulate_open_loop and _simulate_segmented so the physics is
    defined exactly once, not duplicated between the single-call and
    batched-fitting code paths.

    Model: the heat pump alone serves demand up to its own linear,
    outdoor-temperature-dependent max capacity (heatpump_capacity_ref_w
    at _COP_REFERENCE_OUTDOOR_C, sloped by heatpump_capacity_slope_w_per_c
    per degree - real air-source heat pumps deliver less peak capacity as
    it gets colder); gas makes up exactly the remainder. No third "heat
    pump fully off" stage - the heat pump always contributes whatever it
    can, right down to demand that never exceeds its own capacity at all
    (gas_delivered_w == 0 throughout in that case).

    Shape-agnostic via plain numpy broadcasting - works identically for
    _simulate_open_loop's 1D arrays and _simulate_segmented's batched 2D
    ones.

    :return: (hp_delivered_w, gas_delivered_w), both same shape as
        q_heat_total_w, always non-negative, always summing back to it.
    """
    hp_capacity_w = np.clip(
        heatpump_capacity_ref_w
        + heatpump_capacity_slope_w_per_c * (outdoor - _COP_REFERENCE_OUTDOOR_C),
        0.0,
        None,
    )
    hp_delivered_w = np.minimum(q_heat_total_w, hp_capacity_w)
    gas_delivered_w = q_heat_total_w - hp_delivered_w
    return hp_delivered_w, gas_delivered_w


def _facade_poa_scalar(
    ghi: float,
    dni: float,
    dhi: float,
    sun_alt_sin: float,
    sun_alt_cos: float,
    sun_az_sin: float,
    sun_az_cos: float,
    cos_tilt: float,
    sin_tilt: float,
    cos_az: float,
    sin_az: float,
) -> float:
    """Plane-of-array irradiance for one timestep, exact closed form of
    pvlib.irradiance.get_total_irradiance's own default (model='isotropic',
    albedo=0.25) - verified against pvlib directly (see
    test_facade_poa_matches_pvlib_isotropic_model). sun_alt_sin/sun_alt_cos
    are cos(zenith)/sin(zenith) (altitude = 90-zenith); sun_az_sin/sun_az_cos
    are sin/cos of solar azimuth - both already computed once in
    _prepare_inputs, unaffected by facade orientation. Plain Python
    min/max, not np.clip/np.maximum, for the same per-timestep-scalar
    performance reason as the rest of _simulate_open_loop (see its own
    np.clip comment).
    """
    cos_aoi = cos_tilt * sun_alt_sin + sin_tilt * sun_alt_cos * (cos_az * sun_az_cos + sin_az * sun_az_sin)
    poa_direct = dni * max(0.0, cos_aoi)
    poa_sky = dhi * (1.0 + cos_tilt) * 0.5
    poa_ground = ghi * 0.25 * (1.0 - cos_tilt) * 0.5
    return max(0.0, poa_direct + poa_sky + poa_ground)


def _facade_poa_vectorized(
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    sun_alt_sin: np.ndarray,
    sun_alt_cos: np.ndarray,
    sun_az_sin: np.ndarray,
    sun_az_cos: np.ndarray,
    cos_tilt: float,
    sin_tilt: float,
    cos_az: float,
    sin_az: float,
) -> np.ndarray:
    """Array/vectorized sibling of _facade_poa_scalar - identical formula,
    used by _simulate_segmented's own batched (n_segments, segment_len)
    arrays."""
    cos_aoi = cos_tilt * sun_alt_sin + sin_tilt * sun_alt_cos * (cos_az * sun_az_cos + sin_az * sun_az_sin)
    poa_direct = dni * np.maximum(0.0, cos_aoi)
    poa_sky = dhi * (1.0 + cos_tilt) * 0.5
    poa_ground = ghi * 0.25 * (1.0 - cos_tilt) * 0.5
    return np.maximum(0.0, poa_direct + poa_sky + poa_ground)


def _prepare_inputs(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
) -> ThermalInputs:
    """facade_azimuth_deg/facade_tilt_deg/solar_horizontal_weight/
    solar_facade_weight are no longer accepted here - facade orientation
    is now a FITTED parameter (see PARAM_NAMES), so the plane-of-array
    projection has to happen inside _simulate_open_loop/_simulate_segmented
    per params-guess, not once upfront. This function now only carries the
    RAW ghi/dni/dhi (plus sun position, unaffected by facade orientation)
    - see _facade_poa_scalar/_facade_poa_vectorized for where they're
    actually combined.
    """
    room = _series(df, "room_temp", 20.0).to_numpy(dtype=float)
    electric = _series(df, "electric_power", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    gas = _series(df, "gas_consumption", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    duty = _series(df, "heatpump_duty", 0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    supply_default = pd.Series(25.0, index=df.index, dtype=float)
    supply = pd.to_numeric(df.get("supply_temp", supply_default), errors="coerce").ffill().fillna(25.0)
    outdoor = _series(df, "outdoor_temp", 10.0).to_numpy(dtype=float)
    wind_speed = _series(df, "wind_speed", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    wind_bearing = np.radians(_series(df, "wind_bearing", 0.0).to_numpy(dtype=float))
    ghi = _series(df, "ghi", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    dni = _series(df, "dni", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    dhi = _series(df, "dhi", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    sun_alt_sin, sun_alt_cos, sun_az_sin, sun_az_cos = _compute_sun_direction_features(
        df,
        latitude=latitude,
        longitude=longitude,
    )
    blind_position = _series(df, "blind_position", 0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    door_open = _series(df, "door_open", 0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    return ThermalInputs(
        index=pd.DatetimeIndex(df.index),
        room=room,
        electric=electric,
        gas=gas,
        duty=duty,
        supply=supply.to_numpy(dtype=float),
        outdoor=outdoor,
        wind_speed=wind_speed,
        wind_sin=np.sin(wind_bearing),
        wind_cos=np.cos(wind_bearing),
        sun_alt_sin=sun_alt_sin.to_numpy(dtype=float),
        sun_alt_cos=sun_alt_cos.to_numpy(dtype=float),
        sun_az_sin=sun_az_sin.to_numpy(dtype=float),
        sun_az_cos=sun_az_cos.to_numpy(dtype=float),
        heatpump_duty=duty,
        blind_position=blind_position,
        door_open=door_open,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
    )


def _simulate_open_loop(
    inputs: ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    initial_air: float,
    initial_mass: float | None = None,
    initial_q_emit: float = 0.0,
    initial_wall: float | None = None,
    horizontal_weight: float = 0.35,
    facade_weight: float = 0.65,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    fit_gas_consumption: bool = False,
) -> SimResult:
    (
        tau_emit,
        emit_gain,
        ua_base,
        ua_wind,
        ua_wind_sin,
        ua_wind_cos,
        mass_tau,
        mass_gain,
        solar_gain,
        solar_alt_sin_gain,
        solar_alt_cos_gain,
        solar_az_sin_gain,
        solar_az_cos_gain,
        bias,
        wall_tau,
        wall_solar_gain,
        wall_to_mass_weight,
        door_open_extra_loss,
        window_solar_radiative_fraction,
        facade_azimuth_deg,
        facade_tilt_deg,
        facade2_azimuth_deg,
        facade2_tilt_deg,
        facade3_azimuth_deg,
        facade3_tilt_deg,
        carnot_efficiency,
        emitter_power_scale_w,
        cop_sensitivity,
        heatpump_capacity_ref_w,
        heatpump_capacity_slope_w_per_c,
        boiler_efficiency,
    ) = params
    n = len(inputs.room)
    # Callers outside this module that still construct ThermalInputs
    # directly (e.g. scripts/cvxpy_state_space_thermal_model.py's own,
    # unrelated state-space model) leave blind_position at its dataclass
    # default of None - resolve once here rather than per-step, exactly
    # recovering the pre-blind-support "always fully open" computation.
    blind_position = inputs.blind_position if inputs.blind_position is not None else np.zeros(n)
    door_open = inputs.door_open if inputs.door_open is not None else np.zeros(n)
    ghi = inputs.ghi if inputs.ghi is not None else np.zeros(n)
    dni = inputs.dni if inputs.dni is not None else np.zeros(n)
    dhi = inputs.dhi if inputs.dhi is not None else np.zeros(n)
    # cop_scale: how much emit_raw (duty*lift) is amplified/dampened by the
    # instantaneous COP relative to a fixed reference COP - see
    # PARAM_NAMES's own cop_sensitivity docstring. Vectorized and computed
    # ONCE per call (not per-timestep), same "params are per-call constants"
    # treatment as _facade_trig above; cop_sensitivity=0.0 makes this an
    # all-ones array, a cheap no-op multiply.
    cop_arr = _cop_carnot_vectorized(carnot_efficiency, inputs.supply, inputs.outdoor)
    cop_ref_arr = _cop_carnot_vectorized(
        carnot_efficiency, inputs.supply, np.full(n, _COP_REFERENCE_OUTDOOR_C)
    )
    cop_scale_arr = np.clip(1.0 + cop_sensitivity * (cop_arr - cop_ref_arr), 0.1, None)
    cos_tilt, sin_tilt, cos_az, sin_az = _facade_trig(facade_azimuth_deg, facade_tilt_deg)
    # facade2/facade3 trig only computed when actually weighted in - a
    # disabled slot (the common case, weight defaults to 0.0) costs nothing
    # extra in the hot per-timestep loop below.
    if facade2_weight != 0.0:
        cos_tilt2, sin_tilt2, cos_az2, sin_az2 = _facade_trig(facade2_azimuth_deg, facade2_tilt_deg)
    if facade3_weight != 0.0:
        cos_tilt3, sin_tilt3, cos_az3, sin_az3 = _facade_trig(facade3_azimuth_deg, facade3_tilt_deg)
    pred_room = np.zeros(n, dtype=float)
    air_before = np.zeros(n, dtype=float)
    mass_series = np.zeros(n, dtype=float)
    q_emit_series = np.zeros(n, dtype=float)
    wall_series = np.zeros(n, dtype=float)
    q_solar_series = np.zeros(n, dtype=float)
    loss_series = np.zeros(n, dtype=float)

    air = float(initial_air)
    mass = float(initial_air if initial_mass is None else initial_mass)
    q_emit = float(initial_q_emit)
    wall = float(initial_air if initial_wall is None else initial_wall)
    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau, 1e-6), 0.0, 1.0))
    wall_alpha = float(np.clip(dt_h / max(wall_tau, 1e-6), 0.0, 1.0))

    for i in range(n):
        air_before[i] = air
        emit_raw = inputs.duty[i] * max(inputs.supply[i] - air, 0.0) * cop_scale_arr[i]
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)

        # Plane-of-array solar proxy - now computed per-timestep here
        # (rather than once upfront) since facade_azimuth_deg/
        # facade_tilt_deg are fitted parameters, not fixed kwargs - see
        # _facade_poa_scalar and the module docstring for the ISO 13790
        # rationale for the horizontal/facade blend.
        poa = _facade_poa_scalar(
            ghi[i], dni[i], dhi[i],
            inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
            cos_tilt, sin_tilt, cos_az, sin_az,
        )
        # facade2/facade3 - optional extra orientations (e.g. a dakraam or a
        # secondary window facing a different way), each summed in
        # proportional to its own configured (never fitted) weight - see
        # module docstring.
        if facade2_weight != 0.0:
            poa += facade2_weight * _facade_poa_scalar(
                ghi[i], dni[i], dhi[i],
                inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
                cos_tilt2, sin_tilt2, cos_az2, sin_az2,
            )
        if facade3_weight != 0.0:
            poa += facade3_weight * _facade_poa_scalar(
                ghi[i], dni[i], dhi[i],
                inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
                cos_tilt3, sin_tilt3, cos_az3, sin_az3,
            )
        q_solar_i = max(0.0, horizontal_weight * ghi[i] + facade_weight * poa) / 1000.0
        q_solar_series[i] = q_solar_i

        # Exterior wall surface: relaxes toward an outdoor-plus-solar-excess
        # equilibrium (a sun-exposed wall genuinely runs hotter than ambient
        # air) with its OWN time constant - never gated by blind_position,
        # since interior shading cannot affect what hits the outside of the
        # building (see module docstring). Updated before mass so mass's own
        # target below can pull toward the freshly-computed wall value this
        # same step, matching how q_emit/mass already feed d_air_dt within
        # the same iteration they're updated in.
        wall_target = inputs.outdoor[i] + wall_solar_gain * q_solar_i
        wall = wall + wall_alpha * (wall_target - wall)

        direction_loss = (
            ua_base
            + ua_wind * inputs.wind_speed[i]
            + ua_wind_sin * inputs.wind_speed[i] * inputs.wind_sin[i]
            + ua_wind_cos * inputs.wind_speed[i] * inputs.wind_cos[i]
        )
        # Extra ventilation loss while a door/window is open - additive on
        # top of the envelope loss, gated 0-1 like blind_position gates
        # window_solar_total below. door_open defaults to 0.0 (closed) when
        # unconfigured, exactly recovering pre-door-support behavior.
        loss_coeff = max(0.0, float(direction_loss)) + door_open_extra_loss * door_open[i]
        solar_direction_gain = (
            solar_gain
            + solar_alt_sin_gain * inputs.sun_alt_sin[i]
            + solar_alt_cos_gain * inputs.sun_alt_cos[i]
            + solar_az_sin_gain * inputs.sun_az_sin[i]
            + solar_az_cos_gain * inputs.sun_az_cos[i]
        )
        # Window-transmitted gain - the ONLY solar pathway a closed blind
        # can block (see module docstring). blind_position defaults to 0.0
        # (open) when unconfigured, so (1 - blind_position) == 1 exactly
        # recovers the pre-blind-support computation. Split into a
        # convective share (straight into d_air_dt, as before) and a
        # radiative share that lands on interior thermal mass instead -
        # the split ISO 13790's 5R1C model and most grey-box RC papers use,
        # since direct-beam sun through a window typically strikes the
        # floor (real thermal mass), not air. At
        # window_solar_radiative_fraction == 0 this is exactly the old,
        # 100%-convective behavior.
        window_solar_total = (
            max(0.0, float(solar_direction_gain)) * q_solar_i * (1.0 - blind_position[i])
        )
        window_solar_convective = (1.0 - window_solar_radiative_fraction) * window_solar_total
        window_solar_radiative = window_solar_radiative_fraction * window_solar_total

        # mass_target == air when wall_to_mass_weight == 0, so the relax-
        # toward-target part is an exact superset of the pre-wall behavior
        # (mass = mass + mass_alpha*(air-mass)) - backward compatible at
        # weight=0. The radiative solar share is added as a direct heat
        # nudge (dt_h-scaled, matching d_air_dt's own units) rather than
        # folded into the relax-to-target itself, since it is a rate-like
        # quantity (like q_emit's contribution to d_air_dt), not a
        # temperature-like target the way wall_target is.
        mass_target = air + wall_to_mass_weight * (wall - air)
        mass = mass + mass_alpha * (mass_target - mass) + dt_h * window_solar_radiative

        d_air_dt = (
            emit_gain * q_emit
            + window_solar_convective
            - loss_coeff * (air - inputs.outdoor[i])
            + mass_gain * (mass - air)
            + bias
        )
        # Plain Python min/max, not np.clip: profiling a real 60-day refit
        # showed np.clip on a single scalar (called here once per timestep,
        # i.e. ~150k+ times over a realistic fit) pays numpy's full ufunc
        # dispatch overhead every call - ~35% of total fit wall-clock time
        # in that profile - for a 1-element operation plain Python handles
        # far more cheaply. Identical result for scalar input either way.
        air = min(35.0, max(5.0, air + dt_h * d_air_dt))
        pred_room[i] = air
        mass_series[i] = mass
        q_emit_series[i] = q_emit
        wall_series[i] = wall
        loss_series[i] = loss_coeff

    # Vectorized, AFTER the per-timestep loop (not inside it) - q_emit_series
    # is already a full array by this point, and this is a cheap one-shot
    # expression, no reason to pay per-iteration Python overhead for it.
    # Room temperature (pred_room, computed above) never depends on this -
    # the room doesn't care which source delivered a given Watt, so the
    # heat-pump-vs-gas SPLIT below is a pure accounting exercise on top of
    # the already-finished simulation, not a physical feedback.
    cop_series = _cop_carnot_vectorized(carnot_efficiency, inputs.supply, inputs.outdoor)
    q_heat_total_w = q_emit_series * emitter_power_scale_w
    gas_pred_series = None
    if fit_gas_consumption:
        hp_delivered_w, gas_delivered_w = _split_heat_pump_gas_w(
            q_heat_total_w, inputs.outdoor, heatpump_capacity_ref_w, heatpump_capacity_slope_w_per_c
        )
        electric_pred_series = hp_delivered_w / cop_series
        gas_pred_series = gas_delivered_w * dt_h / (boiler_efficiency * GAS_CALORIFIC_VALUE_WH_PER_M3)
    else:
        electric_pred_series = q_heat_total_w / cop_series

    return SimResult(
        room=pred_room,
        air_before=air_before,
        mass=mass_series,
        q_emit=q_emit_series,
        wall=wall_series,
        q_solar=q_solar_series,
        loss_coeff=loss_series,
        electric_pred=electric_pred_series,
        gas_pred=gas_pred_series,
    )


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _simulate_segmented(
    inputs: ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    segment_len: int,
    horizontal_weight: float = 0.35,
    facade_weight: float = 0.65,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    return_electric: bool = False,
    return_gas: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized across segments, not a per-segment Python loop calling
    _simulate_open_loop repeatedly (the original implementation, and still
    exactly what a single non-repeated call - e.g.
    command_line.compute_heating_forecast's own one-shot whole-horizon
    simulation, nothing to batch - should keep using directly).

    Every segment is independent by construction: each seeds its own
    T_air/T_mass/T_wall/Q_emit fresh from the room's own actual history at
    the segment's own start (see initial_* below) - NEVER chained from a
    previous segment's own simulated output. That independence is exactly
    what makes batching legal: instead of re-running the full
    segment_len-step Python loop from scratch for every one of
    (n_rows // segment_len) segments (thousands of scalar iterations for a
    real refit window), every segment's state is carried as one element of
    a length-(n_segments) array, and a SINGLE segment_len-step Python loop
    updates every segment at once via plain numpy elementwise ops -
    segment_len (a few dozen) iterations total, not the full row count.
    Profiling a real 60-day refit showed this (combined with the
    scalar-np.clip fix in _simulate_open_loop) cut _fit_temperature_params
    wall-clock time by roughly an order of magnitude; the underlying
    equations and their result are unchanged - see
    test_simulate_segmented_matches_manual_per_segment_loop for the
    numerical proof this is bit-for-bit consistent with the straightforward
    per-segment loop it replaces.

    Only full-length segments are batched this way; a final, shorter tail
    segment (when len(inputs.room) isn't an exact multiple of segment_len)
    is simulated the simple way via _simulate_open_loop directly - simpler
    and safer than padding a ragged batch, and it's at most one extra
    (cheap, O(segment_len)) call regardless of dataset size.

    :param return_electric: when True, also returns the predicted
        electric-power array (see _cop_carnot_vectorized) as a second
        return value - default False keeps the original single-array
        return untouched for every existing caller that doesn't need it.
    :param return_gas: when True (only meaningful together with
        return_electric=True - the split needs the COP/electric machinery
        active), also returns the predicted gas-consumption array as a
        third return value, and electric_pred becomes the capacity-capped
        heat-pump-delivered share rather than the whole demand - see
        _split_heat_pump_gas_w's own docstring for the bivalent-parallel
        model this implements. Default False keeps electric_pred's
        formula byte-identical to today whenever unused.
    """
    (
        tau_emit,
        emit_gain,
        ua_base,
        ua_wind,
        ua_wind_sin,
        ua_wind_cos,
        mass_tau,
        mass_gain,
        solar_gain,
        solar_alt_sin_gain,
        solar_alt_cos_gain,
        solar_az_sin_gain,
        solar_az_cos_gain,
        bias,
        wall_tau,
        wall_solar_gain,
        wall_to_mass_weight,
        door_open_extra_loss,
        window_solar_radiative_fraction,
        facade_azimuth_deg,
        facade_tilt_deg,
        facade2_azimuth_deg,
        facade2_tilt_deg,
        facade3_azimuth_deg,
        facade3_tilt_deg,
        carnot_efficiency,
        emitter_power_scale_w,
        cop_sensitivity,
        heatpump_capacity_ref_w,
        heatpump_capacity_slope_w_per_c,
        boiler_efficiency,
    ) = params

    n = len(inputs.room)
    pred = np.zeros(n, dtype=float)
    electric_pred = np.zeros(n, dtype=float) if return_electric else None
    gas_pred = np.zeros(n, dtype=float) if return_gas else None
    n_full_segments = n // segment_len if segment_len > 0 else 0
    n_batched = n_full_segments * segment_len
    blind_position = inputs.blind_position if inputs.blind_position is not None else np.zeros(n)
    door_open = inputs.door_open if inputs.door_open is not None else np.zeros(n)
    ghi = inputs.ghi if inputs.ghi is not None else np.zeros(n)
    dni = inputs.dni if inputs.dni is not None else np.zeros(n)
    dhi = inputs.dhi if inputs.dhi is not None else np.zeros(n)
    # See _simulate_open_loop's own cop_scale comment - same formula,
    # vectorized once up front here too (used both for the per-segment
    # q_emit warm-start below and the batched per-timestep loop).
    cop_scale_arr = np.clip(
        1.0
        + cop_sensitivity
        * (
            _cop_carnot_vectorized(carnot_efficiency, inputs.supply, inputs.outdoor)
            - _cop_carnot_vectorized(
                carnot_efficiency, inputs.supply, np.full(n, _COP_REFERENCE_OUTDOOR_C)
            )
        ),
        0.1,
        None,
    )
    cos_tilt, sin_tilt, cos_az, sin_az = _facade_trig(facade_azimuth_deg, facade_tilt_deg)
    if facade2_weight != 0.0:
        cos_tilt2, sin_tilt2, cos_az2, sin_az2 = _facade_trig(facade2_azimuth_deg, facade2_tilt_deg)
    if facade3_weight != 0.0:
        cos_tilt3, sin_tilt3, cos_az3, sin_az3 = _facade_trig(facade3_azimuth_deg, facade3_tilt_deg)

    if n_full_segments > 0:
        seg_starts = np.arange(n_full_segments) * segment_len
        prev_idx = np.maximum(0, seg_starts - 1)
        air = inputs.room[prev_idx].astype(float)
        mass = air.copy()
        wall = air.copy()
        q_emit = (
            inputs.duty[prev_idx]
            * np.maximum(inputs.supply[prev_idx] - air, 0.0)
            * cop_scale_arr[prev_idx]
        )

        emit_alpha = min(1.0, max(0.0, dt_h / max(tau_emit, 1e-6)))
        mass_alpha = min(1.0, max(0.0, dt_h / max(mass_tau, 1e-6)))
        wall_alpha = min(1.0, max(0.0, dt_h / max(wall_tau, 1e-6)))

        def _batch(arr: np.ndarray) -> np.ndarray:
            return arr[:n_batched].reshape(n_full_segments, segment_len)

        duty_b = _batch(inputs.duty)
        supply_b = _batch(inputs.supply)
        outdoor_b = _batch(inputs.outdoor)
        wind_speed_b = _batch(inputs.wind_speed)
        wind_sin_b = _batch(inputs.wind_sin)
        wind_cos_b = _batch(inputs.wind_cos)
        ghi_b = _batch(ghi)
        dni_b = _batch(dni)
        dhi_b = _batch(dhi)
        sun_alt_sin_b = _batch(inputs.sun_alt_sin)
        sun_alt_cos_b = _batch(inputs.sun_alt_cos)
        sun_az_sin_b = _batch(inputs.sun_az_sin)
        sun_az_cos_b = _batch(inputs.sun_az_cos)
        blind_b = _batch(blind_position)
        door_open_b = _batch(door_open)
        cop_scale_b = _batch(cop_scale_arr)

        pred_batch = np.zeros((n_full_segments, segment_len), dtype=float)
        q_emit_batch = np.zeros((n_full_segments, segment_len), dtype=float) if return_electric else None
        for t in range(segment_len):
            emit_raw = duty_b[:, t] * np.maximum(supply_b[:, t] - air, 0.0) * cop_scale_b[:, t]
            q_emit = q_emit + emit_alpha * (emit_raw - q_emit)
            if return_electric:
                q_emit_batch[:, t] = q_emit

            poa = _facade_poa_vectorized(
                ghi_b[:, t], dni_b[:, t], dhi_b[:, t],
                sun_alt_sin_b[:, t], sun_alt_cos_b[:, t], sun_az_sin_b[:, t], sun_az_cos_b[:, t],
                cos_tilt, sin_tilt, cos_az, sin_az,
            )
            if facade2_weight != 0.0:
                poa = poa + facade2_weight * _facade_poa_vectorized(
                    ghi_b[:, t], dni_b[:, t], dhi_b[:, t],
                    sun_alt_sin_b[:, t], sun_alt_cos_b[:, t], sun_az_sin_b[:, t], sun_az_cos_b[:, t],
                    cos_tilt2, sin_tilt2, cos_az2, sin_az2,
                )
            if facade3_weight != 0.0:
                poa = poa + facade3_weight * _facade_poa_vectorized(
                    ghi_b[:, t], dni_b[:, t], dhi_b[:, t],
                    sun_alt_sin_b[:, t], sun_alt_cos_b[:, t], sun_az_sin_b[:, t], sun_az_cos_b[:, t],
                    cos_tilt3, sin_tilt3, cos_az3, sin_az3,
                )
            q_solar_t = np.maximum(0.0, horizontal_weight * ghi_b[:, t] + facade_weight * poa) / 1000.0

            wall_target = outdoor_b[:, t] + wall_solar_gain * q_solar_t
            wall = wall + wall_alpha * (wall_target - wall)

            direction_loss = (
                ua_base
                + ua_wind * wind_speed_b[:, t]
                + ua_wind_sin * wind_speed_b[:, t] * wind_sin_b[:, t]
                + ua_wind_cos * wind_speed_b[:, t] * wind_cos_b[:, t]
            )
            loss_coeff = np.maximum(0.0, direction_loss) + door_open_extra_loss * door_open_b[:, t]
            solar_direction_gain = (
                solar_gain
                + solar_alt_sin_gain * sun_alt_sin_b[:, t]
                + solar_alt_cos_gain * sun_alt_cos_b[:, t]
                + solar_az_sin_gain * sun_az_sin_b[:, t]
                + solar_az_cos_gain * sun_az_cos_b[:, t]
            )
            # Convective/radiative split - see _simulate_open_loop's own
            # comments for the physical rationale.
            window_solar_total = (
                np.maximum(0.0, solar_direction_gain) * q_solar_t * (1.0 - blind_b[:, t])
            )
            window_solar_convective = (1.0 - window_solar_radiative_fraction) * window_solar_total
            window_solar_radiative = window_solar_radiative_fraction * window_solar_total

            mass_target = air + wall_to_mass_weight * (wall - air)
            mass = mass + mass_alpha * (mass_target - mass) + dt_h * window_solar_radiative

            d_air_dt = (
                emit_gain * q_emit
                + window_solar_convective
                - loss_coeff * (air - outdoor_b[:, t])
                + mass_gain * (mass - air)
                + bias
            )
            air = np.clip(air + dt_h * d_air_dt, 5.0, 35.0)
            pred_batch[:, t] = air

        pred[:n_batched] = pred_batch.reshape(-1)
        if return_electric:
            cop_batch = _cop_carnot_vectorized(carnot_efficiency, supply_b, outdoor_b)
            q_heat_total_batch = q_emit_batch * emitter_power_scale_w
            if return_gas:
                hp_delivered_batch, gas_delivered_batch = _split_heat_pump_gas_w(
                    q_heat_total_batch, outdoor_b, heatpump_capacity_ref_w, heatpump_capacity_slope_w_per_c
                )
                electric_pred[:n_batched] = (hp_delivered_batch / cop_batch).reshape(-1)
                gas_pred[:n_batched] = (
                    gas_delivered_batch * dt_h / (boiler_efficiency * GAS_CALORIFIC_VALUE_WH_PER_M3)
                ).reshape(-1)
            else:
                electric_pred[:n_batched] = (q_heat_total_batch / cop_batch).reshape(-1)

    if n_batched < n:
        start = n_batched
        sub = ThermalInputs(
            index=inputs.index[start:n],
            room=inputs.room[start:n],
            electric=inputs.electric[start:n],
            gas=inputs.gas[start:n],
            duty=inputs.duty[start:n],
            supply=inputs.supply[start:n],
            outdoor=inputs.outdoor[start:n],
            wind_speed=inputs.wind_speed[start:n],
            wind_sin=inputs.wind_sin[start:n],
            wind_cos=inputs.wind_cos[start:n],
            sun_alt_sin=inputs.sun_alt_sin[start:n],
            sun_alt_cos=inputs.sun_alt_cos[start:n],
            sun_az_sin=inputs.sun_az_sin[start:n],
            sun_az_cos=inputs.sun_az_cos[start:n],
            heatpump_duty=inputs.heatpump_duty[start:n],
            blind_position=blind_position[start:n],
            door_open=door_open[start:n],
            ghi=ghi[start:n],
            dni=dni[start:n],
            dhi=dhi[start:n],
        )
        initial_air = float(inputs.room[max(0, start - 1)])
        initial_q_emit = float(
            inputs.duty[max(0, start - 1)]
            * max(inputs.supply[max(0, start - 1)] - initial_air, 0.0)
            * cop_scale_arr[max(0, start - 1)]
        )
        sim = _simulate_open_loop(
            sub,
            params,
            dt_h=dt_h,
            initial_air=initial_air,
            initial_mass=initial_air,
            initial_q_emit=initial_q_emit,
            initial_wall=initial_air,
            horizontal_weight=horizontal_weight,
            facade_weight=facade_weight,
            facade2_weight=facade2_weight,
            facade3_weight=facade3_weight,
            fit_gas_consumption=return_gas,
        )
        pred[start:n] = sim.room
        if return_electric:
            electric_pred[start:n] = sim.electric_pred
        if return_gas:
            gas_pred[start:n] = sim.gas_pred

    if return_gas:
        return pred, electric_pred, gas_pred
    if return_electric:
        return pred, electric_pred
    return pred


def _slice_inputs(inputs: ThermalInputs, start: int) -> ThermalInputs:
    """Drop the first ``start`` rows of every field in ``inputs``.

    Generalizes the per-field tail-segment slicing ``_simulate_segmented``
    already does above (fixed at its own batched/unbatched boundary) to an
    arbitrary start row. Since ``_simulate_segmented`` always treats
    whatever row is first in its input as the start of segment 0, slicing
    here is exactly equivalent to shifting where every segment boundary
    falls (the "multiple shooting" interval layout) - used by
    ``_fit_temperature_params``'s own ``phase_offsets`` to build
    phase-shifted views without re-running ``_prepare_inputs``.
    """

    def _opt(arr: np.ndarray | None) -> np.ndarray | None:
        return arr[start:] if arr is not None else None

    return ThermalInputs(
        index=inputs.index[start:],
        room=inputs.room[start:],
        electric=inputs.electric[start:],
        gas=inputs.gas[start:],
        duty=inputs.duty[start:],
        supply=inputs.supply[start:],
        outdoor=inputs.outdoor[start:],
        wind_speed=inputs.wind_speed[start:],
        wind_sin=inputs.wind_sin[start:],
        wind_cos=inputs.wind_cos[start:],
        sun_alt_sin=inputs.sun_alt_sin[start:],
        sun_alt_cos=inputs.sun_alt_cos[start:],
        sun_az_sin=inputs.sun_az_sin[start:],
        sun_az_cos=inputs.sun_az_cos[start:],
        heatpump_duty=inputs.heatpump_duty[start:],
        blind_position=_opt(inputs.blind_position),
        door_open=_opt(inputs.door_open),
        ghi=_opt(inputs.ghi),
        dni=_opt(inputs.dni),
        dhi=_opt(inputs.dhi),
        q_solar=_opt(inputs.q_solar),
    )


def _fit_temperature_params(
    inputs: ThermalInputs,
    *,
    dt_h: float,
    segment_len: int,
    max_nfev: int,
    horizontal_weight: float = 0.35,
    facade_weight: float = 0.65,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
    fixed_overrides: dict[str, float] | None = None,
    regularization_overrides: dict[str, float] | None = None,
    phase_offsets: list[int] | None = None,
    warm_start_from: np.ndarray | None = None,
    fit_electric_power: bool = False,
    fit_gas_consumption: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Fit the physics parameters against ``inputs.room`` (and, opt-in, also
    ``inputs.electric``) via segmented open-loop least-squares (3 restarts,
    best-of picked by fit MAE).

    :param fit_electric_power: when True, appends a SECOND residual block -
        predicted electric power (``q_emit * emitter_power_scale_w / COP``,
        see ``_cop_carnot_vectorized``) vs. ``inputs.electric`` - to the
        SAME residual vector the temperature block already builds, so one
        shared parameter set (including the two new PARAM_NAMES entries,
        ``carnot_efficiency``/``emitter_power_scale_w``) has to explain
        BOTH the room's temperature trajectory AND its real electric-power
        draw at once - a genuine joint fit, not a post-hoc division after
        an unrelated temperature-only fit. The electric residual is
        normalized by a robust (median-absolute-deviation) scale of
        ``inputs.electric`` itself before concatenation - same "scaled-MAD,
        not plain std" convention already used for this module's own
        EM-relabeling noise-floor estimates - then multiplied by
        ``_ENERGY_FIT_WEIGHT``, so a large-magnitude Watt residual doesn't
        drown out (or get drowned out by) the small-magnitude degC
        temperature residuals it's concatenated with. Defaults to False:
        with no electric residual block appended, ``carnot_efficiency``/
        ``emitter_power_scale_w`` have zero gradient contribution and stay
        at their ``DEFAULT_X0`` seed - today's exact behavior, unchanged.
    :param fit_gas_consumption: when True (only meaningful together with
        ``fit_electric_power=True`` - the split needs the COP/electric
        machinery already active), appends a THIRD residual block -
        predicted gas consumption vs. ``inputs.gas`` - to the same
        residual vector, same MAD-normalized/``_ENERGY_FIT_WEIGHT``-scaled
        treatment as the electric block. The electric block's own formula
        also changes: instead of the whole demand, it becomes the
        bivalent-parallel-capped heat-pump-delivered share (see
        ``_split_heat_pump_gas_w``'s own docstring) - the three new
        PARAM_NAMES entries (``heatpump_capacity_ref_w``/
        ``heatpump_capacity_slope_w_per_c``/``boiler_efficiency``) get
        real gradient contribution only when this is True; pinned at
        their own ``DEFAULT_X0`` otherwise, same "harmless when off"
        treatment as ``fit_electric_power``.
    :param warm_start_from: optional full (PARAM_NAMES-order, one value per
        entry) starting point - e.g. the currently-deployed parameters, for a
        cheaper re-tune rather than a from-scratch refit. When given, this
        REPLACES the usual 3-restart hedge (``x0_default``/``x0_fast``/
        ``x0_slow``) with a single restart seeded here - the exploratory
        east/west-facade hedge exists to discover the right basin when
        starting cold; when warm-starting from an already-good fit, that
        exploration is no longer the point, cutting straight to a single
        confident restart is what actually makes tuning cheaper than a
        full refit. ``max_nfev`` is unchanged - the one restart still gets
        a full budget to converge precisely. Composes with ``phase_offsets``
        unchanged (orthogonal - one controls how many restarts, the other
        how many phases each restart's residual vector spans).
    :param phase_offsets: optional list of row offsets (e.g.
        ``[0, 12, 24, 36]`` for 4 evenly-spaced 24h-segment phases at
        dt_h=0.5) at which to evaluate the fit JOINTLY - one shared
        parameter set must explain the data under every segment-boundary
        alignment at once, since ``_simulate_segmented`` always starts
        segment 0 at whatever row is first in its input (see
        ``_slice_inputs``). Defaults to ``None``, treated exactly like
        ``[0]`` - today's single-fixed-phase behavior, byte-for-byte
        unchanged. A single ``least_squares`` call still does only 3
        restarts regardless of how many phases are given; each restart's
        residual vector just grows by a factor of ``len(phase_offsets)``,
        which is far cheaper than re-running the whole fit once per phase
        and picking the best (that "best-of-N" approach also risks
        selection bias: picking whichever phase happens to score best on
        the very data used to judge it).
    :param fixed_overrides: optional {param_name: value} - excludes those
        parameters from the search entirely rather than trying to pin them
        via zero-width least_squares bounds (a known scipy TRF edge case
        avoided entirely this way). The returned params array always
        reflects the override exactly - a known fact never drifts, same
        precedence principle as a real door/blind sensor overriding
        inference elsewhere in this codebase. Generic/reusable - not
        currently populated by any facade-orientation caller (see
        regularization_overrides below for that case specifically).
    :param regularization_overrides: optional {param_name: value} for the 6
        facade-orientation parameters (facade_azimuth_deg/facade_tilt_deg,
        facade2_*, facade3_*), plus mass_tau_h/tau_emit_h (a building-mass-
        class/emitter-type based prior - see mass_tau_h_anchor_from_building_class/
        tau_emit_h_anchor_from_emitter_type) - unlike fixed_overrides, the
        parameter STAYS in the free/fittable search, just regularised MUCH
        more strongly (_CONFIGURED_PRIOR_REG_WEIGHT, 10x the default
        _DEFAULT_PRIOR_REG_WEIGHT) toward the given value instead of the
        usual default anchor (south/vertical for facade1, each slot's own
        DEFAULT_X0 seed for facade2/facade3; NO default anchor at all for
        mass_tau_h/tau_emit_h - see the regularisation array below, they
        stay fully unregularised/free exactly like today whenever absent
        from this dict). A known/estimated value becomes "hard to move
        away from" rather than "impossible to move away from" - resists
        the kind of wandering seen on short/noisy refit windows while
        still letting strong, sustained real data correct a wrong
        estimate, matching this codebase's "configurable, but still
        genuinely self-learning" design principle (see module docstring).
    """
    offsets = list(phase_offsets) if phase_offsets else [0]
    phase_inputs = [inputs if off == 0 else _slice_inputs(inputs, off) for off in offsets]
    phase_finite = [np.isfinite(sub.room) for sub in phase_inputs]
    n_data_residuals = sum(int(f.sum()) for f in phase_finite)
    if fit_electric_power:
        # Robust (MAD-based) scale of the ORIGINAL, unsliced electric-power
        # series - computed once here (not per phase, not per residuals()
        # call) since it's a fixed normalization constant, not something
        # that depends on params or which phase is being evaluated. Same
        # house's own sensor across every phase, so one shared scale is
        # both correct and cheaper than recomputing it per phase.
        electric_finite = np.isfinite(inputs.electric)
        electric_vals = inputs.electric[electric_finite]
        electric_median = float(np.median(electric_vals)) if len(electric_vals) else 0.0
        electric_scale = (
            float(1.4826 * np.median(np.abs(electric_vals - electric_median)))
            if len(electric_vals) >= 2
            else 1.0
        )
        electric_scale = max(electric_scale, 1.0)  # avoid dividing by ~0 for an all-flat sensor
        n_data_residuals += sum(len(sub.electric) for sub in phase_inputs)
    if fit_gas_consumption:
        # Same robust-scale treatment as electric_scale above, against
        # inputs.gas instead.
        gas_finite = np.isfinite(inputs.gas)
        gas_vals = inputs.gas[gas_finite]
        gas_median = float(np.median(gas_vals)) if len(gas_vals) else 0.0
        gas_scale = (
            float(1.4826 * np.median(np.abs(gas_vals - gas_median))) if len(gas_vals) >= 2 else 1.0
        )
        gas_scale = max(gas_scale, 1.0)
        n_data_residuals += sum(len(sub.gas) for sub in phase_inputs)
    fixed_overrides = dict(fixed_overrides or {})
    if not fit_electric_power:
        # With no electric residual block, carnot_efficiency/
        # emitter_power_scale_w are a genuinely flat direction (truly zero
        # gradient, not just weakly identified) - least_squares's
        # trust-region iterations aren't guaranteed to leave a flat
        # direction exactly at x0 (the same "can drift on pure numerical
        # noise" phenomenon this module has already documented for
        # mass_tau_h - see UPPER_BOUNDS's own comment), so pin them out of
        # the free-parameter search entirely via fixed_overrides rather
        # than trusting the optimizer to leave them alone. A caller-
        # supplied fixed_overrides value for either name (unlikely, but
        # possible) is respected and not clobbered.
        fixed_overrides.setdefault("carnot_efficiency", float(DEFAULT_X0[PARAM_NAMES.index("carnot_efficiency")]))
        fixed_overrides.setdefault(
            "emitter_power_scale_w", float(DEFAULT_X0[PARAM_NAMES.index("emitter_power_scale_w")])
        )
    if not fit_gas_consumption:
        # Same "genuinely flat direction, pin rather than trust the
        # optimizer to leave it alone" treatment as carnot_efficiency/
        # emitter_power_scale_w above.
        for _gas_param_name in (
            "heatpump_capacity_ref_w",
            "heatpump_capacity_slope_w_per_c",
            "boiler_efficiency",
        ):
            fixed_overrides.setdefault(_gas_param_name, float(DEFAULT_X0[PARAM_NAMES.index(_gas_param_name)]))
    regularization_overrides = regularization_overrides or {}
    fixed_indices = sorted(PARAM_NAMES.index(name) for name in fixed_overrides)
    free_indices = [i for i in range(len(PARAM_NAMES)) if i not in fixed_indices]

    def _build_full(free_values: np.ndarray) -> np.ndarray:
        full = np.zeros(len(PARAM_NAMES), dtype=float)
        for idx in fixed_indices:
            full[idx] = fixed_overrides[PARAM_NAMES[idx]]
        full[free_indices] = free_values
        return full

    def _prior_reg_term(params: np.ndarray, idx: int, default_anchor: float, scale: float) -> float:
        """Regularisation term for a parameter that ALWAYS has some default
        anchor even when unconfigured (the 6 facade-orientation params -
        see regularization_overrides docstring above): anchor+weight both
        depend on whether this specific parameter was configured. mass_tau_h/
        tau_emit_h are handled separately below (no term at all unless
        configured, not even at the default weight)."""
        name = PARAM_NAMES[idx]
        if name in regularization_overrides:
            anchor, weight = regularization_overrides[name], _CONFIGURED_PRIOR_REG_WEIGHT
        else:
            anchor, weight = default_anchor, _DEFAULT_PRIOR_REG_WEIGHT
        return ((params[idx] - anchor) / scale) * weight

    def residuals(free_values: np.ndarray) -> np.ndarray:
        params = _build_full(free_values)
        res_pieces = []
        for sub, sub_finite in zip(phase_inputs, phase_finite):
            sim_out = _simulate_segmented(
                sub,
                params,
                dt_h=dt_h,
                segment_len=segment_len,
                horizontal_weight=horizontal_weight,
                facade_weight=facade_weight,
                facade2_weight=facade2_weight,
                facade3_weight=facade3_weight,
                return_electric=fit_electric_power,
                return_gas=fit_gas_consumption,
            )
            if fit_gas_consumption:
                pred, pred_electric, pred_gas = sim_out
            elif fit_electric_power:
                pred, pred_electric = sim_out
                pred_gas = None
            else:
                pred, pred_electric, pred_gas = sim_out, None, None
            res_pieces.append(pred[sub_finite] - sub.room[sub_finite])
            if fit_electric_power:
                res_pieces.append((pred_electric - sub.electric) / electric_scale * _ENERGY_FIT_WEIGHT)
            if fit_gas_consumption:
                res_pieces.append((pred_gas - sub.gas) / gas_scale * _ENERGY_FIT_WEIGHT)
        res = np.concatenate(res_pieces)
        # Light regularisation keeps weakly identified terms from doing wild things.
        # Added ONCE below (not per phase) - it depends only on params, not
        # on data, so repeating it per phase would just up-weight it by
        # len(phase_inputs) relative to the data-fit terms.
        reg_list = [
            (params[4] / 0.02) * 0.03,
            (params[5] / 0.02) * 0.03,
            (params[13] / 0.20) * 0.03,
            *((params[9:13] / 2.0) * 0.03),
            # wall_to_mass_weight (index 16): the newest, least-grounded
            # cross-term - regularised toward 0 (== "mass ignores wall,
            # exactly today's behavior") for the same reason the
            # directional solar cross-terms above are, and unlike
            # wall_tau_h/wall_solar_gain_c which behave like the
            # existing (unregularised) mass_tau_h/solar_gain_c_per_h.
            (params[16] / 1.0) * 0.03,
            # door_open_extra_loss_per_h (index 17): a constant-zero
            # feature column whenever no door/window sensor or
            # relabeling is in play (door_open defaults to 0.0
            # throughout) - same "weakly identified without real
            # signal, regularise toward 0" treatment as
            # wall_to_mass_weight above.
            (params[17] / 1.0) * 0.03,
            # cop_sensitivity (index 27): regularised toward 0 (== "duty's
            # effectiveness is COP-independent, exactly today's pre-
            # cop_sensitivity behavior") for the same "weakly identified
            # without real signal, don't let it drift" reason as
            # wall_to_mass_weight/door_open_extra_loss above - a real
            # temperature-residual improvement has to outweigh this pull
            # before cop_sensitivity moves away from 0. Scaled by its own
            # bound width (0.3) rather than 1.0, matching how this term's
            # natural magnitude compares to the other two above.
            (params[27] / 0.3) * 0.03,
            # facade_azimuth_deg/facade_tilt_deg (indices 19/20) and
            # facade2/facade3's own pairs (21-24): regularised via
            # _prior_reg_term - mildly toward a default anchor
            # (south/vertical for facade1, each slot's own DEFAULT_X0
            # seed for facade2/facade3, matching the old hardcoded
            # assumption) when unconfigured, or MUCH more strongly
            # toward the user's own configured value when present in
            # regularization_overrides (see that parameter's own
            # docstring above) - "hard to move away from", not a hard
            # pin. Only meaningful for facade2/facade3 while the
            # matching facadeN_weight is nonzero (a disabled slot's
            # value doesn't affect _simulate_segmented's output at all,
            # so this term just contributes a fixed, harmless constant
            # either way).
            _prior_reg_term(params, 19, 180.0, 180.0),
            _prior_reg_term(params, 20, 90.0, 90.0),
            _prior_reg_term(params, 21, DEFAULT_X0[21], 180.0),
            _prior_reg_term(params, 22, DEFAULT_X0[22], 90.0),
            _prior_reg_term(params, 23, DEFAULT_X0[23], 180.0),
            _prior_reg_term(params, 24, DEFAULT_X0[24], 90.0),
        ]
        # mass_tau_h (index 6)/tau_emit_h (index 0): UNLIKE the facade terms
        # above, these get NO regularisation term at all when unconfigured -
        # today's exact behavior (fully free, no default pull) - only added
        # when the user has actually configured a building-mass-class/
        # emitter-type prior (see mass_tau_h_anchor_from_building_class/
        # tau_emit_h_anchor_from_emitter_type and the module docstring).
        if "mass_tau_h" in regularization_overrides:
            reg_list.append(_prior_reg_term(params, 6, DEFAULT_X0[6], DEFAULT_X0[6]))
        if "tau_emit_h" in regularization_overrides:
            reg_list.append(_prior_reg_term(params, 0, DEFAULT_X0[0], DEFAULT_X0[0]))
        if fit_electric_power:
            # carnot_efficiency and emitter_power_scale_w are NOT
            # independently identifiable from electric-power data alone -
            # electric_pred is proportional to emitter_power_scale_w /
            # carnot_efficiency (COP itself is proportional to
            # carnot_efficiency), so any pair with the same ratio predicts
            # identically, a genuine flat valley (confirmed empirically:
            # an unregularised joint fit landed carnot_efficiency far from
            # its true generating value while still fitting electric power
            # well). Mildly anchoring carnot_efficiency toward a sensible
            # physical default (same _prior_reg_term/regularization_overrides
            # mechanism as the 6 facade-orientation params - a caller can
            # pass a known value from a heat pump's own datasheet via
            # regularization_overrides["carnot_efficiency"] for a much
            # stronger anchor) breaks the degeneracy by letting
            # emitter_power_scale_w (left unregularised, same treatment as
            # mass_tau_h/solar_gain_c_per_h) absorb the true scale instead.
            carnot_idx = PARAM_NAMES.index("carnot_efficiency")
            reg_list.append(_prior_reg_term(params, carnot_idx, DEFAULT_X0[carnot_idx], 0.4))
        if fit_gas_consumption:
            # boiler_efficiency: the exact same flat-valley degeneracy as
            # carnot_efficiency/emitter_power_scale_w above - gas_pred is
            # proportional to gas_delivered_w / boiler_efficiency, and
            # gas_delivered_w itself depends on heatpump_capacity_ref_w/
            # heatpump_capacity_slope_w_per_c, so a bigger capacity + lower
            # efficiency can fit similarly to a smaller capacity + higher
            # efficiency. Anchored toward its own 0.90 DEFAULT_X0 (same
            # regularization_overrides mechanism, a real datasheet/nameplate
            # value can override), letting heatpump_capacity_ref_w (left
            # unregularised, same "absorbs the true scale" treatment as
            # emitter_power_scale_w) absorb the true capacity instead.
            boiler_idx = PARAM_NAMES.index("boiler_efficiency")
            reg_list.append(_prior_reg_term(params, boiler_idx, DEFAULT_X0[boiler_idx], 0.2))
            # heatpump_capacity_slope_w_per_c: weakly identified without a
            # wide outdoor-temperature range inside the fit window - same
            # "regularised toward a neutral default, real sustained data
            # can still move it" treatment as cop_sensitivity above.
            slope_idx = PARAM_NAMES.index("heatpump_capacity_slope_w_per_c")
            reg_list.append((params[slope_idx] / 500.0) * 0.03)
        reg = np.array(reg_list, dtype=float)
        return np.concatenate([res, reg])

    x0_fast = DEFAULT_X0.copy()
    x0_fast[:9] = np.array([1.0, 0.12, 0.035, 0.001, 0.0, 0.0, 24.0, 0.06, 0.25], dtype=float)
    x0_slow = DEFAULT_X0.copy()
    x0_slow[:9] = np.array([5.0, 0.04, 0.018, 0.0025, 0.0, 0.0, 96.0, 0.025, 0.55], dtype=float)
    x0_default = DEFAULT_X0.copy()
    # Hedge across plausible facade orientations using the same 3 restarts
    # (rather than adding a 4th) - a single local search seeded at due
    # south could get stuck if the real orientation is far away (e.g.
    # north-facing). Kept UNCONDITIONALLY, even when facade_azimuth_deg is
    # configured (regularization_overrides): least_squares is a local
    # optimizer that cannot escape whichever basin it starts in, so if the
    # configured estimate is wrong, at least one restart still needs a
    # genuine chance to discover the real direction and win the best-of-3
    # comparison on its own (lower overall score) - collapsing every
    # restart onto the (possibly wrong) anchor would make
    # regularization_overrides behave like a de-facto hard pin in
    # practice, defeating its whole purpose. Only the PRIMARY restart
    # (x0_default) is biased toward a configured value - a reasonable
    # "start where we best guess" without sacrificing the other two
    # restarts' exploratory role.
    az1_idx = PARAM_NAMES.index("facade_azimuth_deg")
    x0_fast[az1_idx] = 90.0  # east
    x0_slow[az1_idx] = 270.0  # west
    if "facade_azimuth_deg" in regularization_overrides:
        x0_default[az1_idx] = regularization_overrides["facade_azimuth_deg"]
    starts = [np.asarray(warm_start_from, dtype=float)] if warm_start_from is not None else [
        x0_default,
        x0_fast,
        x0_slow,
    ]

    lb_free = LOWER_BOUNDS[free_indices]
    ub_free = UPPER_BOUNDS[free_indices]

    best = None
    for x0_full in starts:
        x0_full = np.clip(x0_full, LOWER_BOUNDS, UPPER_BOUNDS)
        result = least_squares(
            residuals,
            x0=x0_full[free_indices],
            bounds=(lb_free, ub_free),
            loss="soft_l1",
            f_scale=0.30,
            max_nfev=max_nfev,
            verbose=0,
        )
        score = float(np.mean(np.abs(residuals(result.x)[:n_data_residuals])))
        if best is None or score < best[0]:
            best = (score, result)

    assert best is not None
    result = best[1]
    return _build_full(result.x), {
        "fit_mae_c": float(best[0]),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "success": bool(result.success),
        "status": int(result.status),
    }
