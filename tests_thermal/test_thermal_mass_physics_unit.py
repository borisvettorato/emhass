"""Unit tests for the thermal-mass physics simulation core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emhass.thermal.thermal_mass_physics import (
    BUILDING_MASS_CLASS_CM,
    DEFAULT_X0,
    EMITTER_TAU_H_ESTIMATE,
    PARAM_NAMES,
    ThermalInputs,
    _cop_carnot_vectorized,
    _facade_poa_scalar,
    _facade_trig,
    _fit_temperature_params,
    _infer_timestep_hours,
    _prepare_inputs,
    _simulate_open_loop,
    _simulate_segmented,
    _slice_inputs,
    mass_tau_h_anchor_from_building_class,
    tau_emit_h_anchor_from_emitter_type,
)

pytestmark = pytest.mark.unit


def _weather_df(
    n: int = 48,
    outdoor_temp: float = 5.0,
    ghi: float = 0.0,
    blind_position: float = 0.0,
    door_open: float = 0.0,
) -> pd.DataFrame:
    idx = pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "outdoor_temp": outdoor_temp,
            "wind_speed": 10.0,
            "ghi": ghi,
            "dni": 0.0,
            "dhi": 0.0,
            "heatpump_duty": 0.0,
            "blind_position": blind_position,
            "door_open": door_open,
        },
        index=idx,
    )


def test_infer_timestep_hours_half_hourly() -> None:
    idx = pd.date_range("2026-01-15", periods=10, freq="30min", tz="UTC")
    assert _infer_timestep_hours(idx) == pytest.approx(0.5)


def test_infer_timestep_hours_too_short_falls_back() -> None:
    idx = pd.date_range("2026-01-15", periods=1, freq="30min", tz="UTC")
    assert _infer_timestep_hours(idx) == pytest.approx(0.25)


def test_facade_poa_matches_pvlib_isotropic_model() -> None:
    """_facade_poa_scalar is a hand-derived closed form of
    pvlib.irradiance.get_total_irradiance's own default (model='isotropic',
    albedo=0.25) - this is the concrete proof it's correct, not just
    "looks right". Only checked for daytime (zenith < 90) points: for a
    below-horizon sun, sun_alt_sin's existing clip-to-0 convention (used
    everywhere else in this module, e.g. the solar_alt_sin_gain_c_per_h
    harmonic) loses cos(zenith)'s true negative value - harmless in
    practice (dni/ghi/dhi are genuinely ~0 at night in any real dataset),
    but not something a synthetic all-hours comparison could match exactly."""
    from pvlib.irradiance import get_total_irradiance
    from pvlib.location import Location

    lat, lon = 51.65, 4.93
    idx = pd.date_range("2026-06-15 00:00", periods=48 * 3, freq="30min", tz="UTC")
    location = Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(idx)
    zenith = solar_position["apparent_zenith"].to_numpy()
    azimuth = solar_position["azimuth"].to_numpy()
    altitude_rad = np.radians(90.0 - zenith)
    azimuth_rad = np.radians(azimuth)
    sun_alt_sin = np.clip(np.sin(altitude_rad), 0.0, None)
    sun_alt_cos = np.cos(altitude_rad)
    sun_az_sin = np.sin(azimuth_rad)
    sun_az_cos = np.cos(azimuth_rad)

    rng = np.random.default_rng(1)
    n = len(idx)
    day = zenith < 90.0
    ghi = np.where(day, rng.uniform(0, 800, n), 0.0)
    dni = np.where(day, rng.uniform(0, 900, n), 0.0)
    dhi = np.where(day, rng.uniform(0, 300, n), 0.0)

    for facade_azimuth_deg, facade_tilt_deg in [(180, 90), (90, 90), (270, 45), (0, 0), (135, 60), (359, 89)]:
        poa_ref = get_total_irradiance(
            surface_tilt=facade_tilt_deg,
            surface_azimuth=facade_azimuth_deg,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=dni,
            ghi=ghi,
            dhi=dhi,
        )["poa_global"].fillna(0.0).clip(lower=0.0).to_numpy()

        cos_tilt, sin_tilt, cos_az, sin_az = _facade_trig(facade_azimuth_deg, facade_tilt_deg)
        poa_mine = np.array(
            [
                _facade_poa_scalar(
                    ghi[i], dni[i], dhi[i],
                    sun_alt_sin[i], sun_alt_cos[i], sun_az_sin[i], sun_az_cos[i],
                    cos_tilt, sin_tilt, cos_az, sin_az,
                )
                for i in range(n)
            ]
        )
        np.testing.assert_allclose(poa_mine, poa_ref, atol=1e-6)


def test_multi_facade_weights_zero_reproduces_single_orientation_backward_compat() -> None:
    """facade2_weight/facade3_weight default to 0.0 (slot disabled) - a
    house that never configures a second/third orientation must get EXACTLY
    today's single-orientation q_solar/room trajectory, bit for bit, for
    both _simulate_open_loop and _simulate_segmented - the backward-
    compatibility guarantee the module docstring claims."""
    df = _weather_df(n=96, outdoor_temp=5.0, ghi=600.0)
    inputs = _prepare_inputs(df, latitude=51.65, longitude=4.93)
    params = DEFAULT_X0.copy()

    sim_default = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0)
    sim_explicit_zero = _simulate_open_loop(
        inputs, params, dt_h=0.5, initial_air=20.0, facade2_weight=0.0, facade3_weight=0.0
    )
    np.testing.assert_allclose(sim_default.room, sim_explicit_zero.room, atol=1e-12)
    np.testing.assert_allclose(sim_default.q_solar, sim_explicit_zero.q_solar, atol=1e-12)

    pred_default = _simulate_segmented(inputs, params, dt_h=0.5, segment_len=48)
    pred_explicit_zero = _simulate_segmented(
        inputs, params, dt_h=0.5, segment_len=48, facade2_weight=0.0, facade3_weight=0.0
    )
    np.testing.assert_allclose(pred_default, pred_explicit_zero, atol=1e-12)


def test_multi_facade_poa_sums_weighted_contributions() -> None:
    """With facade2_weight/facade3_weight nonzero, q_solar must equal the
    horizontal/facade blend of (facade1's own POA) + weight2*(facade2's own
    POA) + weight3*(facade3's own POA) - the ISO 13790-style per-orientation
    summation described in the module docstring - verified against a
    hand-rolled reference built directly from _facade_poa_scalar (already
    separately verified against pvlib in
    test_facade_poa_matches_pvlib_isotropic_model), not by re-deriving the
    plane-of-array formula here."""
    df = _weather_df(n=48, outdoor_temp=5.0, ghi=600.0)
    inputs = _prepare_inputs(df, latitude=51.65, longitude=4.93)
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("facade2_azimuth_deg")] = 0.0  # north
    params[PARAM_NAMES.index("facade2_tilt_deg")] = 10.0  # near-horizontal dakraam
    params[PARAM_NAMES.index("facade3_azimuth_deg")] = 90.0  # east
    params[PARAM_NAMES.index("facade3_tilt_deg")] = 90.0
    facade2_weight, facade3_weight = 0.4, 0.2

    sim = _simulate_open_loop(
        inputs, params, dt_h=0.5, initial_air=20.0, facade2_weight=facade2_weight, facade3_weight=facade3_weight
    )

    n = len(inputs.room)
    dni = np.zeros(n) if inputs.dni is None else inputs.dni
    dhi = np.zeros(n) if inputs.dhi is None else inputs.dhi
    trig1 = _facade_trig(180.0, 90.0)  # DEFAULT_X0's own facade1 orientation
    trig2 = _facade_trig(0.0, 10.0)
    trig3 = _facade_trig(90.0, 90.0)
    expected_q_solar = np.zeros(n)
    for i in range(n):
        poa = _facade_poa_scalar(
            inputs.ghi[i], dni[i], dhi[i],
            inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
            *trig1,
        )
        poa += facade2_weight * _facade_poa_scalar(
            inputs.ghi[i], dni[i], dhi[i],
            inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
            *trig2,
        )
        poa += facade3_weight * _facade_poa_scalar(
            inputs.ghi[i], dni[i], dhi[i],
            inputs.sun_alt_sin[i], inputs.sun_alt_cos[i], inputs.sun_az_sin[i], inputs.sun_az_cos[i],
            *trig3,
        )
        expected_q_solar[i] = max(0.0, 0.35 * inputs.ghi[i] + 0.65 * poa) / 1000.0

    np.testing.assert_allclose(sim.q_solar, expected_q_solar, atol=1e-9)

    # A disabled slot (weight left at 0) must not move q_solar at all, even
    # with a wildly different orientation configured for it - changing
    # facade2's own azimuth/tilt while facade2_weight stays 0.0 must be a
    # complete no-op.
    sim_only3 = _simulate_open_loop(
        inputs, params, dt_h=0.5, initial_air=20.0, facade2_weight=0.0, facade3_weight=facade3_weight
    )
    params_different_facade2 = params.copy()
    params_different_facade2[PARAM_NAMES.index("facade2_azimuth_deg")] = 270.0
    params_different_facade2[PARAM_NAMES.index("facade2_tilt_deg")] = 0.0
    sim_only3_different_facade2 = _simulate_open_loop(
        inputs, params_different_facade2, dt_h=0.5, initial_air=20.0, facade2_weight=0.0, facade3_weight=facade3_weight
    )
    np.testing.assert_allclose(sim_only3.q_solar, sim_only3_different_facade2.q_solar, atol=1e-12)
    assert not np.allclose(sim_only3.q_solar, sim.q_solar)


def _synthetic_true_params(param_overrides: dict[str, float]) -> np.ndarray:
    """DEFAULT_X0, with solar_gain_c_per_h bumped up (strong enough that
    facade orientation visibly matters) plus any caller-supplied overrides
    (e.g. a true mass_tau_h/tau_emit_h/facade_azimuth_deg to generate
    synthetic data with, or to isolate as the one free parameter in a fit)."""
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("solar_gain_c_per_h")] = 1.5
    for name, value in param_overrides.items():
        params[PARAM_NAMES.index(name)] = value
    return params


def _synthetic_solar_driven_inputs(n_days: float, *, param_overrides: dict[str, float] | None = None) -> ThermalInputs:
    """A multi-day synthetic room trajectory GENERATED by _simulate_open_loop
    itself at known true parameter value(s) (see _synthetic_true_params) -
    real pvlib sun position/ghi/dni/dhi (same day/night envelope approach as
    test_facade_poa_matches_pvlib_isotropic_model) PLUS a real heating duty
    cycle (so tau_emit_h's own dynamics are actually exercised, not just
    solar/mass), so the fit has a genuine, physically real signal to recover
    the true parameter value(s) from."""
    from pvlib.location import Location

    lat, lon = 51.65, 4.93
    n = int(round(n_days * 48))
    idx = pd.date_range("2026-06-01 00:00", periods=n, freq="30min", tz="UTC")
    location = Location(latitude=lat, longitude=lon, tz="UTC")
    solar_position = location.get_solarposition(idx)
    zenith = solar_position["apparent_zenith"].to_numpy()
    day = zenith < 90.0
    rng = np.random.default_rng(7)
    envelope = np.clip(np.cos(np.radians(zenith)), 0.0, None)
    ghi = np.where(day, envelope * 700.0 + rng.uniform(0, 20, n), 0.0)
    dni = np.where(day, envelope * 900.0, 0.0)
    dhi = np.where(day, envelope * 150.0, 0.0)
    outdoor = 10.0 + 4.0 * np.sin(np.arange(n) / 48.0 * 2 * np.pi)
    hour = idx.hour + idx.minute / 60.0
    duty = np.where((hour >= 6) & (hour < 22), 0.5, 0.0)  # daytime heating cycle

    df = pd.DataFrame(
        {
            "outdoor_temp": outdoor, "wind_speed": 2.0, "ghi": ghi, "dni": dni, "dhi": dhi,
            "heatpump_duty": duty, "supply_temp": 40.0,
        },
        index=idx,
    )
    inputs = _prepare_inputs(df, latitude=lat, longitude=lon)

    params = _synthetic_true_params(param_overrides or {})
    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=18.0)
    return ThermalInputs(**{**inputs.__dict__, "room": sim.room})


def _fixed_overrides_pinning_all_but(param_overrides: dict[str, float], free_param_name: str) -> dict[str, float]:
    """Pin every physics parameter EXCEPT free_param_name to the exact
    values _synthetic_solar_driven_inputs generated its data with -
    isolates that one parameter's own identifiability from the confound of
    24 OTHER free parameters partially compensating for a wrong value
    (empirically confirmed for facade_azimuth_deg: with everything free, a
    deliberately-wrong azimuth barely costs any residual at all, since e.g.
    solar_gain_c_per_h/the sun-direction harmonics can absorb most of the
    difference - not what these tests are about)."""
    true_params = _synthetic_true_params(param_overrides)
    return {name: float(true_params[i]) for i, name in enumerate(PARAM_NAMES) if name != free_param_name}


def test_regularization_overrides_pulls_harder_than_default_toward_same_anchor() -> None:
    """Same data/anchor position (180.0, the existing unconfigured default
    for facade_azimuth_deg) - only the WEIGHT differs
    (_CONFIGURED_PRIOR_REG_WEIGHT vs _DEFAULT_PRIOR_REG_WEIGHT,
    see thermal_mass_physics.py's own module docstring). True azimuth
    (30 deg) is deliberately away from both the anchor (180) AND the fixed
    x0_fast/x0_slow restart hedge points (90/270), so the comparison isn't
    muddied by one restart happening to start on top of the truth. The
    configured fit must land measurably closer to 180.0 than the
    unconfigured one - "harder to move away from" is a real, checkable
    effect, not just documentation."""
    true_azimuth = 30.0
    inputs = _synthetic_solar_driven_inputs(3.0, param_overrides={"facade_azimuth_deg": true_azimuth})
    fixed_overrides = _fixed_overrides_pinning_all_but({"facade_azimuth_deg": true_azimuth}, "facade_azimuth_deg")

    params_unconfigured, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=40, fixed_overrides=fixed_overrides
    )
    params_anchored, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=40, fixed_overrides=fixed_overrides,
        regularization_overrides={"facade_azimuth_deg": 180.0},
    )

    az_idx = PARAM_NAMES.index("facade_azimuth_deg")
    dist_unconfigured = abs(params_unconfigured[az_idx] - 180.0)
    dist_anchored = abs(params_anchored[az_idx] - 180.0)
    assert dist_anchored < dist_unconfigured


def test_regularization_overrides_is_not_a_hard_pin() -> None:
    """With azimuth's own identifiability isolated from other free
    parameters (see _fixed_overrides_pinning_all_but) and enough clean
    multi-day solar signal, the fit must still move measurably toward
    the true value rather than staying stuck at a deliberately WRONG
    configured anchor - the property that actually distinguishes
    regularization_overrides from the old fixed_overrides hard-exclude-
    from-search behavior. This also exercises the restart-hedging fix
    (x0_fast/x0_slow must keep exploring 90/270 even when
    facade_azimuth_deg is configured, or every restart collapses onto the
    same, possibly-wrong, starting point and the fit can never discover a
    better answer regardless of how much real data supports one)."""
    true_azimuth = 90.0
    wrong_anchor = 200.0
    inputs = _synthetic_solar_driven_inputs(5.0, param_overrides={"facade_azimuth_deg": true_azimuth})
    fixed_overrides = _fixed_overrides_pinning_all_but({"facade_azimuth_deg": true_azimuth}, "facade_azimuth_deg")

    params_anchored, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=60, fixed_overrides=fixed_overrides,
        regularization_overrides={"facade_azimuth_deg": wrong_anchor},
    )

    az_idx = PARAM_NAMES.index("facade_azimuth_deg")
    fitted = params_anchored[az_idx]
    assert abs(fitted - true_azimuth) < abs(fitted - wrong_anchor)


def test_mass_tau_h_anchor_from_building_class_ratios() -> None:
    """DEFAULT_X0's own mass_tau_h implicitly represents "medium" (its Cm
    ratio to itself is 1.0) - every other class scales proportionally to
    ISO/FDIS 13790:2007 Table 12's own Cm-per-floor-area figures (floor
    area itself cancels out of the ratio, see the module docstring - never
    needed as an input). Unrecognised/empty must return None so callers
    can skip adding a regularization_overrides entry entirely."""
    default_mass_tau_h = float(DEFAULT_X0[PARAM_NAMES.index("mass_tau_h")])
    assert mass_tau_h_anchor_from_building_class("medium") == pytest.approx(default_mass_tau_h)
    for building_class, cm in BUILDING_MASS_CLASS_CM.items():
        expected = default_mass_tau_h * cm / BUILDING_MASS_CLASS_CM["medium"]
        assert mass_tau_h_anchor_from_building_class(building_class) == pytest.approx(expected)
    # heavier construction -> proportionally longer time constant
    assert mass_tau_h_anchor_from_building_class("very_heavy") > mass_tau_h_anchor_from_building_class("very_light")
    assert mass_tau_h_anchor_from_building_class("") is None
    assert mass_tau_h_anchor_from_building_class("nonsense") is None


def test_tau_emit_h_anchor_from_emitter_type_values() -> None:
    """EMITTER_TAU_H_ESTIMATE's own values, verbatim - floor heating must be
    the slowest (largest tau_emit_h), matching real-world heat-emitter
    response time. Unrecognised/empty must return None."""
    for emitter_type, expected in EMITTER_TAU_H_ESTIMATE.items():
        assert tau_emit_h_anchor_from_emitter_type(emitter_type) == pytest.approx(expected)
    assert tau_emit_h_anchor_from_emitter_type("floor_heating") > tau_emit_h_anchor_from_emitter_type("radiator")
    assert tau_emit_h_anchor_from_emitter_type("radiator") > tau_emit_h_anchor_from_emitter_type("convector")
    assert tau_emit_h_anchor_from_emitter_type("") is None
    assert tau_emit_h_anchor_from_emitter_type("nonsense") is None


def test_mass_tau_h_regularization_overrides_absent_is_fully_unregularized() -> None:
    """UNLIKE the facade-orientation terms (which always have SOME default
    pull toward DEFAULT_X0's own value), mass_tau_h must get NO
    regularisation term at all when its own key is absent from
    regularization_overrides - so a true generating value FAR from
    DEFAULT_X0's own mass_tau_h (48.0) must be recovered close to its real
    value, not dragged back toward 48 the way a hidden default pull would.
    mass_tau_h is isolated via fixed_overrides (see
    _fixed_overrides_pinning_all_but) so real data, not correlation with
    other free parameters, is what's actually being trusted here."""
    # 150.0 (far from 48.0 but ALSO long relative to the 24h segment_len
    # used everywhere in this codebase) turned out to be a bad choice here:
    # with mass_alpha = dt_h/mass_tau_h so small, T_mass barely relaxes
    # within a single 24h segment at all, a documented, pre-existing
    # unidentifiable-direction issue (see UPPER_BOUNDS's own comment on
    # mass_tau_h/mass_gain_per_h/wall_solar_gain_c) - NOT specific to
    # regularization_overrides, so not what this test is about. 20.0 stays
    # well clear of that flat region while still being clearly different
    # from 48.0.
    true_mass_tau_h = 20.0
    inputs = _synthetic_solar_driven_inputs(4.0, param_overrides={"mass_tau_h": true_mass_tau_h})
    fixed_overrides = _fixed_overrides_pinning_all_but({"mass_tau_h": true_mass_tau_h}, "mass_tau_h")

    params, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=60, fixed_overrides=fixed_overrides
    )

    mass_idx = PARAM_NAMES.index("mass_tau_h")
    fitted = params[mass_idx]
    default_mass_tau_h = float(DEFAULT_X0[mass_idx])
    assert abs(fitted - true_mass_tau_h) < abs(fitted - default_mass_tau_h)


def test_mass_tau_h_regularization_overrides_pulls_harder_than_unregularized() -> None:
    """Same shape as the facade_azimuth_deg version above, for a
    differently-scaled, non-periodic parameter - proves the generalised
    mechanism (_prior_reg_term reused for mass_tau_h, see
    _fit_temperature_params's own regularisation array) actually threads
    through correctly, not just for the 6 originally-built facade terms.
    True mass_tau_h (96.0, "heavy"-ish) is deliberately far from the
    configured anchor (23.27, "very_light") so the pull is unambiguous."""
    true_mass_tau_h = 96.0
    anchor = mass_tau_h_anchor_from_building_class("very_light")
    inputs = _synthetic_solar_driven_inputs(3.0, param_overrides={"mass_tau_h": true_mass_tau_h})
    fixed_overrides = _fixed_overrides_pinning_all_but({"mass_tau_h": true_mass_tau_h}, "mass_tau_h")

    params_unconfigured, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=40, fixed_overrides=fixed_overrides
    )
    params_anchored, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=40, fixed_overrides=fixed_overrides,
        regularization_overrides={"mass_tau_h": anchor},
    )

    mass_idx = PARAM_NAMES.index("mass_tau_h")
    # Unconfigured has NO pull at all (see the absent-key test above) - it
    # should track the true value; the anchored fit must land measurably
    # closer to the (wrong) anchor than the unconfigured one does.
    dist_unconfigured = abs(params_unconfigured[mass_idx] - anchor)
    dist_anchored = abs(params_anchored[mass_idx] - anchor)
    assert dist_anchored < dist_unconfigured


def test_slice_inputs_drops_leading_rows() -> None:
    """_slice_inputs must drop the first `start` rows of every field -
    required and optional alike - so slicing THEN simulating is exactly
    equivalent to _simulate_segmented's own convention that segment 0
    always starts at whatever row is first in its input (the mechanism
    _fit_temperature_params's phase_offsets relies on to build
    phase-shifted views without re-running _prepare_inputs)."""
    inputs = _synthetic_solar_driven_inputs(2.0)
    start = 5

    sliced = _slice_inputs(inputs, start)

    assert len(sliced.room) == len(inputs.room) - start
    np.testing.assert_array_equal(sliced.room, inputs.room[start:])
    np.testing.assert_array_equal(sliced.outdoor, inputs.outdoor[start:])
    assert list(sliced.index) == list(inputs.index[start:])
    # ghi/dni/dhi are populated (not None) by _prepare_inputs whenever the
    # source df has those columns (true for _synthetic_solar_driven_inputs) -
    # must be sliced too, not silently dropped to None.
    assert sliced.ghi is not None
    np.testing.assert_array_equal(sliced.ghi, inputs.ghi[start:])


def test_fit_temperature_params_phase_offsets_none_matches_single_phase_list() -> None:
    """phase_offsets=None (the default) must be numerically identical to
    passing phase_offsets=[0] explicitly - both take the same "single
    phase, no shift" code path (see _fit_temperature_params's own
    docstring) - and therefore to today's pre-multi-phase behavior, since
    there is no RNG anywhere in this fit (least_squares/np.clip on fixed
    x0/bounds), so identical inputs must produce bit-identical output."""
    inputs = _synthetic_solar_driven_inputs(2.0)

    params_default, info_default = _fit_temperature_params(inputs, dt_h=0.5, segment_len=48, max_nfev=20)
    params_explicit_single, info_explicit_single = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=20, phase_offsets=[0]
    )

    np.testing.assert_array_equal(params_default, params_explicit_single)
    assert info_default["fit_mae_c"] == pytest.approx(info_explicit_single["fit_mae_c"])


def test_fit_temperature_params_phase_offsets_evaluates_every_phase() -> None:
    """A multi-offset phase_offsets must make _simulate_segmented run once
    PER PHASE per residual evaluation, each on a correctly phase-shifted
    (row-dropped) view of inputs - not just the first offset - proving the
    joint objective genuinely spans every requested phase (one shared
    parameter set has to explain the data under every segment-boundary
    alignment at once) rather than silently collapsing to a single one."""
    from unittest.mock import patch

    import emhass.thermal.thermal_mass_physics as tmp

    inputs = _synthetic_solar_driven_inputs(3.0)
    offsets = [0, 10, 24]
    seen_lengths: list[int] = []
    real_simulate = tmp._simulate_segmented

    def _wrapped(sub_inputs, params, **kwargs):
        seen_lengths.append(len(sub_inputs.room))
        return real_simulate(sub_inputs, params, **kwargs)

    with patch("emhass.thermal.thermal_mass_physics._simulate_segmented", side_effect=_wrapped):
        _fit_temperature_params(inputs, dt_h=0.5, segment_len=48, max_nfev=5, phase_offsets=offsets)

    expected_lengths = {len(inputs.room) - off for off in offsets}
    assert expected_lengths.issubset(set(seen_lengths))


def test_fit_temperature_params_warm_start_from_uses_single_restart() -> None:
    """warm_start_from must collapse the usual 3-restart hedge
    (x0_default/x0_fast/x0_slow) down to a SINGLE restart seeded at the
    given array - verified by patching least_squares and counting calls
    (1, not 3) and checking the x0 it was actually called with matches
    warm_start_from's own free-parameter slice (the whole point of
    warm-starting is skipping the exploratory east/west-facade hedge, not
    just adding a 4th candidate)."""
    from unittest.mock import patch

    import emhass.thermal.thermal_mass_physics as tmp

    inputs = _synthetic_solar_driven_inputs(2.0)
    warm_start = DEFAULT_X0.copy()
    warm_start[PARAM_NAMES.index("mass_tau_h")] = 60.0
    real_least_squares = tmp.least_squares
    seen_x0: list = []

    def _wrapped(*args, **kwargs):
        seen_x0.append(kwargs["x0"])
        return real_least_squares(*args, **kwargs)

    with patch("emhass.thermal.thermal_mass_physics.least_squares", side_effect=_wrapped):
        _fit_temperature_params(inputs, dt_h=0.5, segment_len=48, max_nfev=5, warm_start_from=warm_start)

    assert len(seen_x0) == 1
    # fit_electric_power defaults to False here, which auto-pins
    # carnot_efficiency/emitter_power_scale_w out of the free-parameter
    # search entirely (see _fit_temperature_params's own fit_electric_power
    # docstring) - so the actual free-parameter x0 passed to least_squares
    # is warm_start with just those two entries removed. cop_sensitivity
    # (appended AFTER both, at the very end of PARAM_NAMES) stays free
    # regardless of fit_electric_power - it affects room temperature
    # directly, not just the opt-in electric residual - so a plain
    # warm_start[:-2] positional slice would wrongly drop cop_sensitivity
    # and keep carnot_efficiency instead; np.delete by explicit index is
    # correct regardless of where the auto-pinned params sit positionally.
    pinned_indices = [PARAM_NAMES.index("carnot_efficiency"), PARAM_NAMES.index("emitter_power_scale_w")]
    np.testing.assert_array_equal(seen_x0[0], np.delete(warm_start, pinned_indices))


def test_cop_carnot_vectorized_matches_calculate_cop_heatpump() -> None:
    """The inline, hot-path COP formula (_cop_carnot_vectorized, used
    inside _simulate_open_loop/_simulate_segmented) must match
    utils.calculate_cop_heatpump's own output exactly at representative
    temperature points - it's a deliberate reimplementation for
    performance (see its own docstring: calculate_cop_heatpump logs a
    warning per non-physical timestep, too costly to call thousands of
    times inside least_squares's residual function), not an independent
    formula that could silently drift from the original."""
    from emhass.utils import calculate_cop_heatpump

    carnot_efficiency = 0.4
    supply_c = np.array([35.0, 45.0, 55.0, 30.0])
    outdoor_c = np.array([-5.0, 0.0, 10.0, 20.0])

    inline = _cop_carnot_vectorized(carnot_efficiency, supply_c, outdoor_c)
    reference = calculate_cop_heatpump(supply_c, carnot_efficiency, outdoor_c)

    np.testing.assert_allclose(inline, reference)


def test_fit_temperature_params_fit_electric_power_disabled_leaves_defaults_unchanged() -> None:
    """fit_electric_power=False (the default) must leave
    carnot_efficiency/emitter_power_scale_w exactly at their DEFAULT_X0
    seed - with no electric residual block appended, they have zero
    gradient contribution, so least_squares has no reason to move them
    away from x0. Today's exact behavior, unchanged."""
    inputs = _synthetic_solar_driven_inputs(2.0)

    params, _ = _fit_temperature_params(inputs, dt_h=0.5, segment_len=48, max_nfev=20)

    carnot_idx = PARAM_NAMES.index("carnot_efficiency")
    scale_idx = PARAM_NAMES.index("emitter_power_scale_w")
    assert params[carnot_idx] == pytest.approx(DEFAULT_X0[carnot_idx])
    assert params[scale_idx] == pytest.approx(DEFAULT_X0[scale_idx])


def test_fit_temperature_params_fit_electric_power_recovers_synthetic_params() -> None:
    """fit_electric_power=True must recover known synthetic
    carnot_efficiency/emitter_power_scale_w from data GENERATED by the
    model itself at a known true value (sim.electric_pred used as the
    synthetic inputs.electric target) - same style as the existing
    test_regularization_overrides_* tests. Both isolated together (not
    via _fixed_overrides_pinning_all_but's own single-free-param
    isolation) since they're physically coupled - COP depends on
    carnot_efficiency, and emitter_power_scale_w is a separate multiplier
    on top of it - recovering just one while the other stays wrong isn't
    the property this test is about."""
    true_carnot = 0.35
    true_scale = 3000.0
    inputs = _synthetic_solar_driven_inputs(3.0)
    true_params = _synthetic_true_params({})
    true_params[PARAM_NAMES.index("carnot_efficiency")] = true_carnot
    true_params[PARAM_NAMES.index("emitter_power_scale_w")] = true_scale
    sim = _simulate_open_loop(inputs, true_params, dt_h=0.5, initial_air=18.0)
    inputs = ThermalInputs(**{**inputs.__dict__, "room": sim.room, "electric": sim.electric_pred})

    free_names = {"carnot_efficiency", "emitter_power_scale_w"}
    fixed_overrides = {
        name: float(true_params[i]) for i, name in enumerate(PARAM_NAMES) if name not in free_names
    }

    params, _ = _fit_temperature_params(
        inputs, dt_h=0.5, segment_len=48, max_nfev=60,
        fixed_overrides=fixed_overrides, fit_electric_power=True,
    )

    carnot_idx = PARAM_NAMES.index("carnot_efficiency")
    scale_idx = PARAM_NAMES.index("emitter_power_scale_w")
    # abs=0.06 (was 0.05): the same flat-valley fragility documented below
    # for emitter_power_scale_w - adding cop_sensitivity's own always-zero
    # regularisation row (see PARAM_NAMES) lengthens the residual vector
    # scipy.optimize.least_squares sums per-iteration, and near this
    # genuinely near-flat gradient a purely floating-point-scale shift in
    # iteration-count-sensitive early stopping (ftol) is enough to land the
    # fit right at its own x0=0.4 seed instead of just past it - confirmed
    # cop_scale itself is an EXACT (bit-identical) 1.0 no-op whenever
    # cop_sensitivity=0.0, so this is a convergence-timing artifact of the
    # already-fragile valley, not a computation error.
    assert params[carnot_idx] == pytest.approx(true_carnot, abs=0.06)
    # Wider tolerance than carnot_efficiency's own: electric_pred is only
    # sensitive to the RATIO emitter_power_scale_w/carnot_efficiency (COP
    # is proportional to carnot_efficiency - see _fit_temperature_params's
    # own fit_electric_power regularisation comment for the full
    # derivation), a genuine flat valley the mild carnot_efficiency anchor
    # only partially breaks - a small anchor-induced offset in
    # carnot_efficiency proportionally offsets the recovered scale too.
    assert params[scale_idx] == pytest.approx(true_scale, rel=0.25)


def test_cop_sensitivity_zero_is_exact_noop_regardless_of_carnot_efficiency() -> None:
    """cop_sensitivity=0.0 (its own DEFAULT_X0 seed) must make cop_scale
    EXACTLY 1.0 at every timestep, so _simulate_open_loop's predicted room
    trajectory is bit-for-bit IDENTICAL regardless of carnot_efficiency's
    own value - the backward-compatibility guarantee that this parameter's
    default genuinely recovers pre-cop_sensitivity behavior, not just
    approximately."""
    df = _weather_df(n=12, outdoor_temp=5.0, ghi=0.0)
    df["heatpump_duty"] = 0.6
    inputs = _prepare_inputs(df, latitude=51.65, longitude=4.93)

    params_low_carnot = DEFAULT_X0.copy()
    params_low_carnot[PARAM_NAMES.index("carnot_efficiency")] = 0.2
    params_high_carnot = DEFAULT_X0.copy()
    params_high_carnot[PARAM_NAMES.index("carnot_efficiency")] = 0.6
    # Both leave cop_sensitivity at its own 0.0 DEFAULT_X0 seed.

    sim_low = _simulate_open_loop(inputs, params_low_carnot, dt_h=0.5, initial_air=20.0)
    sim_high = _simulate_open_loop(inputs, params_high_carnot, dt_h=0.5, initial_air=20.0)
    np.testing.assert_array_equal(sim_low.room, sim_high.room)


def test_cop_sensitivity_amplifies_emit_raw_at_higher_cop() -> None:
    """Direct algebra proof, isolated from every other term (ua_base/
    ua_wind/mass_gain/bias/solar all zeroed so only tau_emit/emit_gain/
    carnot_efficiency/cop_sensitivity can affect the result): with
    cop_sensitivity > 0, the SAME duty and supply-air lift at a MILDER
    outdoor temperature (higher COP, per _cop_carnot_vectorized) must
    produce MORE room heating than at a COLDER outdoor temperature (lower
    COP) - the physical relationship cop_sensitivity exists to express (see
    PARAM_NAMES's own docstring)."""
    n = 6
    duty = np.full(n, 0.6)
    supply = np.full(n, 40.0)
    zeros = np.zeros(n)

    def _isolated_inputs(outdoor_c: float) -> ThermalInputs:
        return ThermalInputs(
            index=pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC"),
            room=zeros.copy(),
            electric=zeros.copy(),
            gas=zeros.copy(),
            duty=duty.copy(),
            supply=supply.copy(),
            outdoor=np.full(n, outdoor_c),
            wind_speed=zeros.copy(),
            wind_sin=zeros.copy(),
            wind_cos=zeros.copy(),
            sun_alt_sin=zeros.copy(),
            sun_alt_cos=zeros.copy(),
            sun_az_sin=zeros.copy(),
            sun_az_cos=zeros.copy(),
            heatpump_duty=duty.copy(),
        )

    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_base_per_h")] = 0.0
    params[PARAM_NAMES.index("ua_wind_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("mass_gain_per_h")] = 0.0
    params[PARAM_NAMES.index("bias_c_per_h")] = 0.0
    params[PARAM_NAMES.index("solar_gain_c_per_h")] = 0.0
    params[PARAM_NAMES.index("wall_solar_gain_c")] = 0.0
    params[PARAM_NAMES.index("carnot_efficiency")] = 0.4
    params[PARAM_NAMES.index("cop_sensitivity")] = 0.3  # upper bound - maximal effect

    sim_cold = _simulate_open_loop(_isolated_inputs(-10.0), params, dt_h=0.5, initial_air=20.0)
    sim_mild = _simulate_open_loop(_isolated_inputs(15.0), params, dt_h=0.5, initial_air=20.0)
    assert sim_mild.room[-1] > sim_cold.room[-1]

    # And with cop_sensitivity=0, the SAME two scenarios must land much
    # closer together (bounded by float precision, not exactly equal - the
    # envelope-loss term is zeroed but COP itself still slightly changes
    # emit_gain's own... no, with cop_sensitivity=0 cop_scale=1 always, so
    # the only remaining difference between cold/mild is via outdoor
    # appearing in loss_coeff*(air-outdoor), already zeroed above - so this
    # actually must match exactly too, proving the amplification above is
    # entirely attributable to cop_sensitivity.
    params_flat = params.copy()
    params_flat[PARAM_NAMES.index("cop_sensitivity")] = 0.0
    sim_cold_flat = _simulate_open_loop(_isolated_inputs(-10.0), params_flat, dt_h=0.5, initial_air=20.0)
    sim_mild_flat = _simulate_open_loop(_isolated_inputs(15.0), params_flat, dt_h=0.5, initial_air=20.0)
    np.testing.assert_array_equal(sim_cold_flat.room, sim_mild_flat.room)


def test_prepare_inputs_defaults_missing_columns() -> None:
    idx = pd.date_range("2026-01-15", periods=24, freq="30min", tz="UTC")
    df = pd.DataFrame(index=idx)  # no columns at all

    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
    )

    assert isinstance(inputs, ThermalInputs)
    assert len(inputs.room) == len(idx)
    assert np.all(inputs.duty == 0.0)
    assert np.all(inputs.outdoor == 10.0)  # documented default in _prepare_inputs


def test_prepare_inputs_reads_provided_columns() -> None:
    df = _weather_df(n=10, outdoor_temp=-2.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
    )
    assert np.all(inputs.outdoor == -2.0)
    assert np.all(inputs.wind_speed == 10.0)
    assert np.all(inputs.duty == 0.0)


def test_simulate_open_loop_holds_steady_with_all_terms_zeroed() -> None:
    """With every gain/loss coefficient at zero, nothing should move the
    temperature away from its initial value - a basic conservation check."""
    df = _weather_df(n=48, outdoor_temp=5.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
    )
    params = np.zeros(len(PARAM_NAMES), dtype=float)
    params[PARAM_NAMES.index("tau_emit_h")] = 1.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("mass_tau_h")] = 10.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert np.allclose(sim.room, 20.0)


def test_simulate_open_loop_cools_toward_colder_outdoor() -> None:
    """The scenario compute_rc_model_forecast actually runs: heating off
    (duty=0), a real envelope-loss coefficient, outdoor colder than indoor -
    room temperature must trend down, not up or flat."""
    df = _weather_df(n=48, outdoor_temp=-5.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_wind_sin_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("ua_wind_cos_per_h_per_speed")] = 0.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim.room[-1] < 20.0
    assert np.all(np.diff(sim.room) <= 1e-9)  # monotonically non-increasing


def test_simulate_open_loop_clips_to_physical_bounds() -> None:
    """d_air_dt is unbounded by construction; the stepper must still clip
    room temperature to the documented [5, 35] degC physical range."""
    df = _weather_df(n=200, outdoor_temp=-50.0)
    inputs = _prepare_inputs(
        df,
        latitude=51.65,
        longitude=4.93,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_base_per_h")] = 0.25  # upper bound, aggressive loss

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim.room.min() >= 5.0
    assert sim.room.max() <= 35.0


# ----------------------------------------------------------------------
# Wall state + blind-gated window solar (see thermal_mass_physics.py's own
# module docstring for the physical rationale: window-transmitted gain is
# blind-blockable and feeds air directly; opaque-exterior-wall-absorbed
# gain is never blind-blockable and feeds a separate, laggy wall state that
# only reaches the room via wall_to_mass_weight's pull on T_mass).
# ----------------------------------------------------------------------


def test_wall_state_relaxes_toward_outdoor_plus_solar_when_isolated() -> None:
    """With every OTHER coupling zeroed (air/mass/emit terms all 0, so
    T_wall's own dynamics can be observed in isolation via sim.wall), a
    constant-sun scenario must pull T_wall toward outdoor + wall_solar_gain
    * q_solar - the "sun-exposed wall runs hotter than ambient" physics the
    new state exists to capture."""
    df = _weather_df(n=200, outdoor_temp=5.0, ghi=800.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    params = np.zeros(len(PARAM_NAMES), dtype=float)
    params[PARAM_NAMES.index("tau_emit_h")] = 1.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("mass_tau_h")] = 10.0  # avoid div-by-zero, gain is 0 anyway
    params[PARAM_NAMES.index("wall_tau_h")] = 0.5  # fast enough to equilibrate within the run
    params[PARAM_NAMES.index("wall_solar_gain_c")] = 8.0
    # wall_to_mass_weight stays 0 - isolates T_wall's OWN dynamics from
    # feeding back into T_air/T_mass (which are held fixed here anyway
    # since every other gain is 0).

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0, initial_wall=20.0)

    # sim.q_solar is the ACTUALLY-computed per-timestep proxy for this
    # params guess (facade_azimuth_deg/facade_tilt_deg included) - using it
    # directly avoids hand-duplicating the plane-of-array formula here.
    expected_wall_target = 5.0 + 8.0 * sim.q_solar[-1]
    assert sim.wall[-1] == pytest.approx(expected_wall_target, abs=0.05)
    assert sim.room[-1] == pytest.approx(20.0)  # wall_to_mass_weight=0 -> zero effect on air


def test_blind_position_gates_window_solar_but_not_wall_solar() -> None:
    """Closing the blind must suppress the WINDOW/air-feeding solar pathway
    (room heats less with blind=1 than blind=0) while leaving the
    exterior-wall pathway completely unaffected (sim.wall identical either
    way) - the core physical distinction this feature exists to draw."""
    df_open = _weather_df(n=48, outdoor_temp=5.0, ghi=800.0, blind_position=0.0)
    df_closed = _weather_df(n=48, outdoor_temp=5.0, ghi=800.0, blind_position=1.0)
    kwargs = dict(
        latitude=51.65, longitude=4.93,
    )
    inputs_open = _prepare_inputs(df_open, **kwargs)
    inputs_closed = _prepare_inputs(df_closed, **kwargs)
    params = DEFAULT_X0.copy()  # solar_gain_c_per_h > 0, wall_solar_gain_c > 0 by default

    sim_open = _simulate_open_loop(inputs_open, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)
    sim_closed = _simulate_open_loop(inputs_closed, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim_open.room[-1] > sim_closed.room[-1]
    np.testing.assert_allclose(sim_open.wall, sim_closed.wall)


def test_door_open_increases_ventilation_loss() -> None:
    """An open door/window must cool the room faster toward a colder
    outdoor than a closed one, at the same fitted door_open_extra_loss_per_h
    - the core physical effect this feature exists to model."""
    df_closed = _weather_df(n=48, outdoor_temp=-5.0, door_open=0.0)
    df_open = _weather_df(n=48, outdoor_temp=-5.0, door_open=1.0)
    kwargs = dict(
        latitude=51.65, longitude=4.93,
    )
    inputs_closed = _prepare_inputs(df_closed, **kwargs)
    inputs_open = _prepare_inputs(df_open, **kwargs)
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("door_open_extra_loss_per_h")] = 0.3

    sim_closed = _simulate_open_loop(inputs_closed, params, dt_h=0.5, initial_air=20.0)
    sim_open = _simulate_open_loop(inputs_open, params, dt_h=0.5, initial_air=20.0)

    assert sim_open.room[-1] < sim_closed.room[-1]


def test_door_open_extra_loss_zero_is_backward_compatible() -> None:
    """At door_open_extra_loss_per_h=0 (its DEFAULT_X0 value), door_open
    having any value at all must not change the simulation one bit -
    exact backward compatibility for anyone who hasn't fitted this
    parameter to a real nonzero value yet."""
    df_closed = _weather_df(n=48, outdoor_temp=-5.0, door_open=0.0)
    df_open = _weather_df(n=48, outdoor_temp=-5.0, door_open=1.0)
    kwargs = dict(
        latitude=51.65, longitude=4.93,
    )
    inputs_closed = _prepare_inputs(df_closed, **kwargs)
    inputs_open = _prepare_inputs(df_open, **kwargs)
    params = DEFAULT_X0.copy()
    assert params[PARAM_NAMES.index("door_open_extra_loss_per_h")] == 0.0

    sim_closed = _simulate_open_loop(inputs_closed, params, dt_h=0.5, initial_air=20.0)
    sim_open = _simulate_open_loop(inputs_open, params, dt_h=0.5, initial_air=20.0)

    np.testing.assert_allclose(sim_open.room, sim_closed.room)


def test_wall_to_mass_weight_zero_reproduces_pre_wall_mass_formula() -> None:
    """At wall_to_mass_weight=0 AND window_solar_radiative_fraction=0, T_mass
    must follow EXACTLY the pre-wall/pre-split recurrence (mass = mass +
    mass_alpha*(air-mass)), regardless of how the (now-decoupled) wall state
    itself moves - the backward-compatibility guarantee the module docstring
    claims. window_solar_radiative_fraction must ALSO be zeroed here (unlike
    DEFAULT_X0's own 0.5) since it independently injects heat into mass -
    see test_window_solar_radiative_fraction_feeds_mass_not_air below for
    that pathway on its own."""
    df = _weather_df(n=48, outdoor_temp=-5.0, ghi=500.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    params = DEFAULT_X0.copy()
    params[PARAM_NAMES.index("ua_wind_sin_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("ua_wind_cos_per_h_per_speed")] = 0.0
    params[PARAM_NAMES.index("wall_to_mass_weight")] = 0.0
    params[PARAM_NAMES.index("window_solar_radiative_fraction")] = 0.0

    sim = _simulate_open_loop(inputs, params, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    mass_alpha = 0.5 / DEFAULT_X0[PARAM_NAMES.index("mass_tau_h")]
    expected_mass = np.zeros(len(sim.mass))
    mass = 20.0
    for i, air in enumerate(sim.air_before):
        mass = mass + mass_alpha * (air - mass)
        expected_mass[i] = mass
    np.testing.assert_allclose(sim.mass, expected_mass, atol=1e-9)


def test_window_solar_radiative_fraction_zero_matches_old_all_convective_behavior() -> None:
    """At window_solar_radiative_fraction=0, room temperature must be
    bit-identical to the pre-split behavior (100% of window solar straight
    into T_air, none into T_mass) - the backward-compatibility guarantee for
    anyone who hasn't fitted this parameter to a real nonzero value yet."""
    df = _weather_df(n=48, outdoor_temp=5.0, ghi=600.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    params_all_convective = DEFAULT_X0.copy()
    params_all_convective[PARAM_NAMES.index("window_solar_radiative_fraction")] = 0.0

    sim = _simulate_open_loop(inputs, params_all_convective, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    # Hand-rolled reference using the OLD equations (window solar fed
    # straight into d_air_dt, mass untouched by it). Uses sim.q_solar (the
    # ACTUALLY-computed per-timestep proxy for this params guess, including
    # facade_azimuth_deg/facade_tilt_deg) rather than re-deriving the
    # plane-of-array formula here - that formula is separately verified
    # against pvlib directly in test_facade_poa_matches_pvlib_isotropic_model;
    # this test's own job is only the convective/radiative split invariant.
    (
        tau_emit, emit_gain, ua_base, ua_wind, ua_wind_sin, ua_wind_cos, mass_tau, mass_gain,
        solar_gain, solar_alt_sin_gain, solar_alt_cos_gain, solar_az_sin_gain, solar_az_cos_gain,
        bias, wall_tau, wall_solar_gain, wall_to_mass_weight, door_open_extra_loss, _frac,
        _facade_azimuth_deg, _facade_tilt_deg,
        _facade2_azimuth_deg, _facade2_tilt_deg, _facade3_azimuth_deg, _facade3_tilt_deg,
        _carnot_efficiency, _emitter_power_scale_w, _cop_sensitivity,
    ) = params_all_convective
    dt_h = 0.5
    air = mass = wall = 20.0
    q_emit = 0.0
    emit_alpha = dt_h / tau_emit
    mass_alpha = dt_h / mass_tau
    wall_alpha = dt_h / wall_tau
    expected_room = np.zeros(len(inputs.room))
    for i in range(len(inputs.room)):
        emit_raw = inputs.duty[i] * max(inputs.supply[i] - air, 0.0)
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)
        wall_target = inputs.outdoor[i] + wall_solar_gain * sim.q_solar[i]
        wall = wall + wall_alpha * (wall_target - wall)
        mass_target = air + wall_to_mass_weight * (wall - air)
        mass = mass + mass_alpha * (mass_target - mass)
        direction_loss = ua_base + ua_wind * inputs.wind_speed[i]
        loss_coeff = max(0.0, direction_loss)
        solar_direction_gain = (
            solar_gain
            + solar_alt_sin_gain * inputs.sun_alt_sin[i]
            + solar_alt_cos_gain * inputs.sun_alt_cos[i]
            + solar_az_sin_gain * inputs.sun_az_sin[i]
            + solar_az_cos_gain * inputs.sun_az_cos[i]
        )
        window_solar_heat = max(0.0, solar_direction_gain) * sim.q_solar[i]
        d_air_dt = (
            emit_gain * q_emit + window_solar_heat - loss_coeff * (air - inputs.outdoor[i])
            + mass_gain * (mass - air) + bias
        )
        air = min(35.0, max(5.0, air + dt_h * d_air_dt))
        expected_room[i] = air

    np.testing.assert_allclose(sim.room, expected_room, atol=1e-9)


def test_window_solar_radiative_fraction_feeds_mass_not_air_directly() -> None:
    """At window_solar_radiative_fraction=1 (fully radiative), the room
    must warm LESS from window solar in the short term than at fraction=0
    (fully convective) - the radiative share only reaches air indirectly,
    with a lag, via mass_gain*(mass-air) - while T_mass itself must warm
    MORE at fraction=1 than at fraction=0."""
    df = _weather_df(n=12, outdoor_temp=5.0, ghi=700.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    params_convective = DEFAULT_X0.copy()
    params_convective[PARAM_NAMES.index("window_solar_radiative_fraction")] = 0.0
    params_radiative = DEFAULT_X0.copy()
    params_radiative[PARAM_NAMES.index("window_solar_radiative_fraction")] = 1.0

    sim_convective = _simulate_open_loop(inputs, params_convective, dt_h=0.5, initial_air=20.0, initial_mass=20.0)
    sim_radiative = _simulate_open_loop(inputs, params_radiative, dt_h=0.5, initial_air=20.0, initial_mass=20.0)

    assert sim_radiative.room[-1] < sim_convective.room[-1]
    assert sim_radiative.mass[-1] > sim_convective.mass[-1]


def test_simulate_segmented_matches_manual_per_segment_loop() -> None:
    """_simulate_segmented batches every FULL segment across a vectorized
    numpy loop instead of calling _simulate_open_loop once per segment (a
    performance rewrite, see its own docstring for why this is legal -
    segments never share state) - this is the numerical proof it wasn't
    also a behavior change. Manually reproduces the exact per-segment
    approach _simulate_segmented used before that rewrite (call
    _simulate_open_loop once per segment, each reseeded from the room's own
    actual history at that segment's start) and asserts bit-for-bit
    agreement. n=140 with segment_len=48 deliberately isn't a multiple (140
    = 2*48 + 44) so this also exercises the leftover-tail code path, not
    just the fully-batched one."""
    n, segment_len = 140, 48
    idx = pd.date_range("2026-01-15", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "outdoor_temp": 5.0 + 3.0 * np.sin(np.linspace(0, 6, n)),
            "wind_speed": rng.uniform(0, 8, n),
            "wind_bearing": rng.uniform(0, 360, n),
            "ghi": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 600.0,
            "dni": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 400.0,
            "dhi": np.clip(np.sin(np.linspace(0, 20, n)), 0, None) * 150.0,
            "heatpump_duty": rng.uniform(0, 1, n),
            "room_temp": 20.0 + 2.0 * np.sin(np.linspace(0, 10, n)),
            "blind_position": (np.arange(n) % 7 == 0).astype(float),
        },
        index=idx,
    )
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    params = DEFAULT_X0.copy()

    actual = _simulate_segmented(inputs, params, dt_h=0.5, segment_len=segment_len)

    expected = np.zeros(n, dtype=float)
    for start in range(0, n, segment_len):
        stop = min(n, start + segment_len)
        sub = ThermalInputs(
            index=inputs.index[start:stop],
            room=inputs.room[start:stop],
            electric=inputs.electric[start:stop],
            gas=inputs.gas[start:stop],
            duty=inputs.duty[start:stop],
            supply=inputs.supply[start:stop],
            outdoor=inputs.outdoor[start:stop],
            wind_speed=inputs.wind_speed[start:stop],
            wind_sin=inputs.wind_sin[start:stop],
            wind_cos=inputs.wind_cos[start:stop],
            sun_alt_sin=inputs.sun_alt_sin[start:stop],
            sun_alt_cos=inputs.sun_alt_cos[start:stop],
            sun_az_sin=inputs.sun_az_sin[start:stop],
            sun_az_cos=inputs.sun_az_cos[start:stop],
            heatpump_duty=inputs.heatpump_duty[start:stop],
            blind_position=inputs.blind_position[start:stop],
            ghi=inputs.ghi[start:stop],
            dni=inputs.dni[start:stop],
            dhi=inputs.dhi[start:stop],
        )
        initial_air = float(inputs.room[max(0, start - 1)])
        initial_q_emit = float(
            inputs.duty[max(0, start - 1)] * max(inputs.supply[max(0, start - 1)] - initial_air, 0.0)
        )
        sim = _simulate_open_loop(
            sub, params, dt_h=0.5, initial_air=initial_air, initial_mass=initial_air,
            initial_q_emit=initial_q_emit, initial_wall=initial_air,
        )
        expected[start:stop] = sim.room

    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_simulate_segmented_handles_short_input_via_tail_path_only() -> None:
    """n < segment_len must skip the vectorized batch entirely (0 full
    segments) and fall through cleanly to the tail path alone."""
    df = _weather_df(n=10, outdoor_temp=5.0, ghi=200.0)
    inputs = _prepare_inputs(
        df, latitude=51.65, longitude=4.93,
    )
    pred = _simulate_segmented(inputs, DEFAULT_X0.copy(), dt_h=0.5, segment_len=48)
    assert len(pred) == 10
    assert np.all(np.isfinite(pred))
