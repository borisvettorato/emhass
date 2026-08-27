"""Sensorless blind-position and door/window-opening inference for the RC
thermal-mass physics model (thermal_mass_physics.py) - the RC-model sibling
of blind_kalman_detector.py/opening_kalman_detector.py, which serve the
same purpose for the self-learning-physics model family only.

Why this needs its own predictor rather than reusing the self-learning-
physics one: self-learning-physics is a stateless one-step linear
regression (theta @ features), so "predict assuming blind=0" is a trivial
one-row recompute, and blind_x_dni's fitted coefficient (beta) is a single
scalar used directly in blind_kalman_detector.invert_blind_position_from_residual.
The RC model is a stateful ODE simulation (T_air/T_mass/T_wall/Q_emit
accumulate across a segment) with no single fitted "beta" for blind
position - instead, blind_position and door_open each enter d_air_dt (or,
for blind's radiative share, T_mass's own update) LINEARLY at each
individual timestep (see thermal_mass_physics.py's window_solar_total/
loss_coeff terms), so the right adaptation is a teacher-forced one-step-
ahead predictor (predict_one_step_history below) with an ANALYTICALLY-
COMPUTED, time-varying sensitivity - not a fitted scalar, but the same
role. Since blind's radiative share (window_solar_radiative_fraction, see
thermal_mass_physics.py's own module docstring for the ISO 13790
rationale) reaches T_air only through T_mass, the sensitivity is not a
plain "dt_h * solar term" any more - it has a small extra piece for the
radiative share's SAME-STEP contribution via the already-updated mass
(mass is updated before d_air_dt within one step) - see
predict_one_step_history's own inline derivation.

Two different detection strategies, matching the physical difference
between the two quantities (same split as the self-learning-physics
modules this mirrors):
- Door/window (binary, "probably open right now"): a 3-sigma innovation
  gate on the teacher-forced residual - genuinely simpler than blind, no
  algebraic inversion needed, since detection only needs "was there an
  unexplained mismatch", not a magnitude. Reuses
  opening_kalman_detector.kalman_forward_filter_array/kalman_rts_smooth/
  smoothed_opening_flags UNCHANGED - this module only supplies the RC-
  specific one-step predictor those functions need as input.
- Blind (continuous 0-1 position): algebraic inversion of the residual
  against the analytic sensitivity above, mirroring
  blind_kalman_detector.invert_blind_position_from_residual's role exactly
  but generalized from a fixed scalar beta to a per-timestep sensitivity
  array (see invert_blind_position_from_residual/resolve_measurement_noise
  below). Reuses blind_kalman_detector.kalman_forward_filter_with_persistence/
  blind_cold_start_state/smoothed_blind_position UNCHANGED - only the
  inversion/noise-resolution math (which assumes a scalar beta there) is
  reimplemented here for an array sensitivity.

Unlike self-learning-physics's blind_x_dni (a SEPARATE regression
coefficient that starts at exactly 0 - unidentified - until real variance
exists in the feature), RC's sensitivity is derived from solar_gain_c_per_h
and friends, which are CORE model parameters always fit meaningfully from
real solar data regardless of blind state - so there is no "beta stuck at
0, need a bootstrap heuristic" problem here the way blind_kalman_detector.py
has (see bootstrap_raw_blind_signal_from_residual there) - this module has
no bootstrap-phase equivalent, by design, not by omission.

This module is pure math - zero HA/live-fetch/persistence code, matching
this package's existing thermal/*.py convention. All refit orchestration
(the EM relabel loop) lives in command_line.py.
"""

from __future__ import annotations

import numpy as np

from emhass.thermal.thermal_mass_physics import ThermalInputs, _facade_poa_scalar, _facade_trig

# Physical informativeness gate on RC's own solar proxy (q_solar, already
# scaled roughly to a 0-1 range by the same horizontal/facade blend
# thermal_mass_physics.py's own _simulate_open_loop uses) -
# below this, dividing residual by sensitivity amplifies ordinary model
# noise into a meaninglessly large position estimate. Deliberately gates on
# the PHYSICAL quantity (is there sun) rather than on the derived
# sensitivity magnitude (which also depends on the current fit's solar
# coefficients) - keeps the gate's meaning independent of fit quality,
# mirroring blind_kalman_detector.BLIND_DNI_INFORMATIVE_FLOOR_WM2's own
# role one level up (this is q_solar's own already-normalized scale, not a
# raw W/m2 value, hence the much smaller threshold).
RC_BLIND_INFORMATIVE_Q_SOLAR_FLOOR = 0.05

# Pure divide-by-zero safety floor on the sensitivity itself, NOT a trust
# threshold (that role is RC_BLIND_INFORMATIVE_Q_SOLAR_FLOOR above) -
# mirrors BLIND_KALMAN_BETA_EPSILON's purely-defensive role.
RC_BLIND_SENSITIVITY_EPSILON = 1e-9

# Same floor/ceiling role as blind_kalman_detector.BLIND_KALMAN_R_FLOOR/
# CEILING - the floor prevents r->0 (and an overconfident single-cycle
# jump) at very high sensitivity; the ceiling is a defensive numerical
# bound only (the q_solar floor above already excludes the worst cases).
RC_BLIND_KALMAN_R_FLOOR = 0.0025
RC_BLIND_KALMAN_R_CEILING = 1.0


def predict_one_step_history(
    inputs: ThermalInputs,
    params: np.ndarray,
    dt_h: float,
    *,
    force_blind_zero: bool = False,
    force_door_zero: bool = False,
    horizontal_weight: float = 0.35,
    facade_weight: float = 0.65,
    facade2_weight: float = 0.0,
    facade3_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teacher-forced one-step-ahead room-temperature prediction across the
    whole window, plus the analytic sensitivity of that prediction to
    blind_position - the RC-model sibling of self_learning_physics.py's own
    predict_one_step_history (same "T_air is always reset to the real
    observed value every step" discipline, so error can never compound the
    way a closed-loop/open-loop simulation's would over a multi-week
    window).

    pred[i] predicts inputs.room[i], using state "as of just before i"
    (T_air = the ACTUAL previous reading, never a predicted one) and this
    module's own driving inputs at index max(0, i-1). Index 0 has no real
    predecessor, so it bootstraps from its own reading as a same-value
    "predecessor" (a real, locally-grounded value, same spirit as - and no
    worse a boundary condition than - self_learning_physics.py's own
    hardcoded-fallback treatment of its first row).

    Hidden states (T_mass/T_wall/Q_emit) are NOT teacher-forced (no ground
    truth exists for them) - they evolve via the model's own recursion,
    seeded from the actual air history. This is a deliberate, bounded
    approximation: mass_tau_h/wall_tau_h are both long relative to dt_h in
    every real fit seen so far (tens to hundreds of hours vs. a ~0.5h
    step), so this drift is slow, and T_air itself - the only state ever
    compared against ground truth - never compounds error since it is
    reset every single step.

    :param force_blind_zero: Use blind_position=0.0 (assume fully open)
        throughout, regardless of inputs.blind_position - used when THIS
        function's caller is detecting blind position itself (comparing
        against this "assumed open" baseline). When False, uses whatever
        inputs.blind_position already holds (the current best estimate,
        e.g. from a real sensor or a prior relabel iteration) - used when
        detecting door_open instead, so a sunny/shaded period isn't
        misattributed to a door event.
    :param force_door_zero: Same role as force_blind_zero, for door_open.
    :return: (pred, sensitivity_blind, q_solar) - all length len(inputs.room).
        sensitivity_blind[i] = d(pred[i])/d(blind_position at the driving
        index) - always computed (cheap byproduct), meaningful only when
        force_blind_zero=True (it is the local slope AWAY from that forced
        baseline). q_solar[i] is the actually-computed plane-of-array proxy
        at the driving index (using this call's own facade_azimuth_deg/
        facade_tilt_deg) - a cheap byproduct callers need for
        invert_blind_position_from_residual's own informativeness gate,
        now that q_solar is no longer a precomputed ThermalInputs field.
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
        _carnot_efficiency,
        _emitter_power_scale_w,
        _cop_sensitivity,
    ) = params

    n = len(inputs.room)
    if force_blind_zero:
        blind_position = np.zeros(n)
    else:
        blind_position = inputs.blind_position if inputs.blind_position is not None else np.zeros(n)
    if force_door_zero:
        door_open = np.zeros(n)
    else:
        door_open = inputs.door_open if inputs.door_open is not None else np.zeros(n)
    ghi = inputs.ghi if inputs.ghi is not None else np.zeros(n)
    dni = inputs.dni if inputs.dni is not None else np.zeros(n)
    dhi = inputs.dhi if inputs.dhi is not None else np.zeros(n)
    cos_tilt, sin_tilt, cos_az, sin_az = _facade_trig(facade_azimuth_deg, facade_tilt_deg)
    if facade2_weight != 0.0:
        cos_tilt2, sin_tilt2, cos_az2, sin_az2 = _facade_trig(facade2_azimuth_deg, facade2_tilt_deg)
    if facade3_weight != 0.0:
        cos_tilt3, sin_tilt3, cos_az3, sin_az3 = _facade_trig(facade3_azimuth_deg, facade3_tilt_deg)

    pred = np.zeros(n, dtype=float)
    sensitivity_blind = np.zeros(n, dtype=float)
    q_solar_series = np.zeros(n, dtype=float)

    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau, 1e-6), 0.0, 1.0))
    wall_alpha = float(np.clip(dt_h / max(wall_tau, 1e-6), 0.0, 1.0))

    air = float(inputs.room[0])
    mass = air
    wall = air
    q_emit = float(inputs.duty[0] * max(inputs.supply[0] - air, 0.0))

    for i in range(n):
        src = max(0, i - 1)
        emit_raw = inputs.duty[src] * max(inputs.supply[src] - air, 0.0)
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)

        poa = _facade_poa_scalar(
            ghi[src], dni[src], dhi[src],
            inputs.sun_alt_sin[src], inputs.sun_alt_cos[src], inputs.sun_az_sin[src], inputs.sun_az_cos[src],
            cos_tilt, sin_tilt, cos_az, sin_az,
        )
        if facade2_weight != 0.0:
            poa += facade2_weight * _facade_poa_scalar(
                ghi[src], dni[src], dhi[src],
                inputs.sun_alt_sin[src], inputs.sun_alt_cos[src], inputs.sun_az_sin[src], inputs.sun_az_cos[src],
                cos_tilt2, sin_tilt2, cos_az2, sin_az2,
            )
        if facade3_weight != 0.0:
            poa += facade3_weight * _facade_poa_scalar(
                ghi[src], dni[src], dhi[src],
                inputs.sun_alt_sin[src], inputs.sun_alt_cos[src], inputs.sun_az_sin[src], inputs.sun_az_cos[src],
                cos_tilt3, sin_tilt3, cos_az3, sin_az3,
            )
        q_solar_src = max(0.0, horizontal_weight * ghi[src] + facade_weight * poa) / 1000.0
        q_solar_series[i] = q_solar_src

        wall_target = inputs.outdoor[src] + wall_solar_gain * q_solar_src
        wall = wall + wall_alpha * (wall_target - wall)

        direction_loss = (
            ua_base
            + ua_wind * inputs.wind_speed[src]
            + ua_wind_sin * inputs.wind_speed[src] * inputs.wind_sin[src]
            + ua_wind_cos * inputs.wind_speed[src] * inputs.wind_cos[src]
        )
        loss_coeff = max(0.0, float(direction_loss)) + door_open_extra_loss * door_open[src]
        solar_direction_gain = (
            solar_gain
            + solar_alt_sin_gain * inputs.sun_alt_sin[src]
            + solar_alt_cos_gain * inputs.sun_alt_cos[src]
            + solar_az_sin_gain * inputs.sun_az_sin[src]
            + solar_az_cos_gain * inputs.sun_az_cos[src]
        )
        # Convective/radiative split - mirrors thermal_mass_physics.py's own
        # _simulate_open_loop exactly (see that module's docstring for the
        # ISO 13790 rationale).
        s = max(0.0, float(solar_direction_gain)) * q_solar_src
        window_solar_total = s * (1.0 - blind_position[src])
        window_solar_convective = (1.0 - window_solar_radiative_fraction) * window_solar_total
        window_solar_radiative = window_solar_radiative_fraction * window_solar_total

        mass_target = air + wall_to_mass_weight * (wall - air)
        mass = mass + mass_alpha * (mass_target - mass) + dt_h * window_solar_radiative

        d_air_dt = (
            emit_gain * q_emit
            + window_solar_convective
            - loss_coeff * (air - inputs.outdoor[src])
            + mass_gain * (mass - air)
            + bias
        )
        pred[i] = min(35.0, max(5.0, air + dt_h * d_air_dt))
        # d(pred[i])/d(blind_position[src]), derived from the chain above:
        # blind_position gates s*(1-b) -> both the convective share (direct
        # -dt_h*(1-frac)*s contribution to pred) AND, via the now-updated
        # mass THIS SAME STEP, an indirect -dt_h*(dt_h*mass_gain*frac)*s
        # contribution (mass is updated BEFORE d_air_dt uses it, so a
        # radiative nudge to mass already shows up in this step's pred, not
        # just future ones). At frac=0 this reduces exactly to the old
        # dt_h*s formula.
        sensitivity_blind[i] = dt_h * s * ((1.0 - window_solar_radiative_fraction) + mass_gain * dt_h * window_solar_radiative_fraction)

        # Teacher-force: the NEXT iteration's starting air is the ACTUAL
        # observed value at this index, never this step's own prediction.
        air = float(inputs.room[i])

    return pred, sensitivity_blind, q_solar_series


def invert_blind_position_from_residual(
    residual: np.ndarray,
    sensitivity: np.ndarray,
    q_solar: np.ndarray,
    q_solar_floor: float = RC_BLIND_INFORMATIVE_Q_SOLAR_FLOOR,
) -> np.ndarray:
    """Exact algebraic inversion (see module docstring): raw(t) =
    -residual(t) / sensitivity(t) for q_solar(t) above q_solar_floor, else
    NaN - clipped to [0, 1] (NaN preserved through np.clip).

    Sign check: closing the blind cuts solar gain, so a real closed blind +
    sun means actual < pred_open_baseline (cooler than "assumed open"), so
    residual < 0; sensitivity >= 0 by construction (see
    predict_one_step_history); raw = -residual/sensitivity =
    -(negative)/(positive) = positive - correctly trends toward "closed" (1).
    """
    residual = np.asarray(residual, dtype=float)
    sensitivity = np.asarray(sensitivity, dtype=float)
    q_solar = np.asarray(q_solar, dtype=float)
    raw = np.full(residual.shape, np.nan)
    mask = (q_solar > q_solar_floor) & (sensitivity > RC_BLIND_SENSITIVITY_EPSILON)
    raw[mask] = -residual[mask] / sensitivity[mask]
    return np.clip(raw, 0.0, 1.0)


def resolve_measurement_noise(
    residual_std_c: float, sensitivity: np.ndarray | float
) -> np.ndarray | float:
    """r(t) = clip((residual_std_c / sensitivity(t))**2, R_FLOOR, R_CEILING) -
    the array-sensitivity generalization of
    blind_kalman_detector.resolve_blind_measurement_noise (which divides by
    |beta|*dni instead of this module's own single combined sensitivity
    term). Robust to sensitivity containing zeros (maps to the ceiling,
    never raises) - callers are not required to pre-filter to informative
    timesteps only.
    """
    sensitivity_arr = np.asarray(sensitivity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (residual_std_c / sensitivity_arr) ** 2
    r = np.nan_to_num(r, nan=RC_BLIND_KALMAN_R_CEILING, posinf=RC_BLIND_KALMAN_R_CEILING)
    r = np.clip(r, RC_BLIND_KALMAN_R_FLOOR, RC_BLIND_KALMAN_R_CEILING)
    return float(r) if np.ndim(sensitivity) == 0 else r
