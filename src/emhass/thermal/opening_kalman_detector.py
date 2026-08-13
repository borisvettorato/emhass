"""
Opening Kalman Detector - Sensorless "window/door probably open" inference
============================================================================

A per-room scalar Kalman filter that infers whether a window/door is open
RIGHT NOW purely from thermal behaviour - comparing a room's live observed
temperature against what its existing thermal model (physics-family formula
or a fitted self-learning-physics model) predicted for "now", and flagging a
statistically large mismatch ("innovation") as "probably open". This runs
ALONGSIDE the sensor-based room_opening_open mechanism in command_line.py
(see _build_room_opening_open_with_kalman_fallback) - always on, OR'd with
any real heatpump_room_window_sensors/heatpump_room_door_sensors reading,
never a fallback-only mode.

Scalar (1-D state = room temperature, 1-D measurement) is the right minimal
formulation here - there is deliberately no cross-room coupling in the
filter itself, so this module only ever feeds the shared "pause heating +
extra ventilation loss" room_opening_open signal, never the neighbor-
coupling-boost room_door_open signal (a single room's own residual can't
distinguish "my window is open to the cold outside" from "my door is open to
a colder neighbor room" without jointly modelling the neighbor too).

The "predict" step is an external nonlinear one-step physics/learned model
rather than a linear F*x, so P_pred = P_prev + Q is the standard Extended-
Kalman-Filter simplification with Jacobian(d predict / d x_prev) ~= 1 - an
excellent approximation here since a room's own RC time constant is always
much greater than one dispatch timestep, so the predictor's sensitivity to
its own starting temperature really is ~=1.

This module is pure math - zero HA/live-fetch/persistence code, matching
this package's existing thermal/*.py convention (self_learning_physics.py,
thermal_mass_physics.py). All HA/persistence orchestration lives in
command_line.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emhass.utils import simulate_physics_room_temperature_trajectory

# Normalized-innovation fault-detection gate: |innovation| > GATE_SIGMA*sqrt(S)
# flags "probably open". 3.0 is the standard ~99.7%-confidence threshold for
# a 1-DOF residual - large enough that ordinary sensor/model noise rarely
# crosses it, small enough that a genuinely open window/door (a sustained,
# repeating residual) reliably crosses it within 1-2 cycles.
KALMAN_GATE_SIGMA = 3.0

# Physics/simple-family rooms have no fitted model to derive a data-driven
# noise estimate from, so both R and Q are fixed, non-configurable defaults
# (mirroring OPENING_EXTRA_ACH/DOOR_OPEN_COUPLING_MULTIPLIER's own precedent
# in optimization.py). R = 0.09 C^2 (~0.3 C stdev): typical sensor + model
# mismatch for a static U-value formula. Q = 0.01 C^2 (~0.1 C stdev) per
# cycle: normal unexplained drift even with everything closed.
PHYSICS_KALMAN_R_C2 = 0.09
PHYSICS_KALMAN_Q_C2 = 0.01

# Self-learning-family rooms get a real, per-room, data-driven R from the
# refit's own holdout residual std (see command_line.py::refit_self_learning_physics_model's
# residual_std_c capture) - Q is derived as a fraction of that R rather than
# an independent constant, since a better-fit room (smaller R) should also
# track genuine drift more tightly (smaller Q).
SELF_LEARNING_KALMAN_Q_FRACTION_OF_R = 0.2
# Fallback R for a room fitted before residual_std_c existed (an older
# persisted self_learning_physics_room_dispatch_coefficients.json).
SELF_LEARNING_KALMAN_FALLBACK_R_C2 = PHYSICS_KALMAN_R_C2
# Floor on the self-learning R, to prevent a suspiciously-tiny fitted
# residual_std_c from producing R~=0 -> K~=1 -> every reading flagged.
SELF_LEARNING_KALMAN_R_FLOOR_C2 = 0.0004

# A room's persisted Kalman state older than this is treated as stale
# (EMHASS restart, missed cycle, first-ever run) and reseeded via
# cold_start_state rather than trusted as a prior belief.
KALMAN_STATE_MAX_GAP_HOURS = 3.0


@dataclass
class KalmanStepResult:
    """One scalar Kalman predict+update step's full intermediate state -
    exposed in full (not just is_open) so callers/tests can inspect/assert
    on x_new/p_new for persistence and on innovation/s for diagnostics."""

    x_pred: float
    p_pred: float
    innovation: float
    s: float
    k: float
    x_new: float
    p_new: float
    is_open: bool


def kalman_predict_update(
    x_prev: float,
    p_prev: float,
    x_pred: float,
    z_measured: float,
    q: float,
    r: float,
    gate_sigma: float = KALMAN_GATE_SIGMA,
) -> KalmanStepResult:
    """One scalar Kalman filter predict+update cycle.

    x_pred is supplied by the caller (computed externally by
    predict_next_room_temperature_physics_family/_self_learning, an external
    nonlinear one-step model - not derived here from x_prev via a linear
    transition matrix). p_prev is the previous cycle's posterior variance;
    q/r are this room's process/measurement noise variances (see the
    module-level PHYSICS_KALMAN_*/SELF_LEARNING_KALMAN_* constants for how
    callers should resolve them per room family).

    :param x_prev: Previous cycle's posterior state estimate (deg C).
    :param p_prev: Previous cycle's posterior variance (deg C^2).
    :param x_pred: This cycle's externally-computed one-step prediction (deg C).
    :param z_measured: This cycle's live observed room temperature (deg C).
    :param q: Process noise variance (deg C^2).
    :param r: Measurement/model noise variance (deg C^2).
    :param gate_sigma: Number of standard deviations of the innovation
        covariance S beyond which the reading is flagged "probably open".
    :return: KalmanStepResult with every intermediate quantity.
    """
    p_pred = p_prev + q
    innovation = z_measured - x_pred
    s = p_pred + r
    k = p_pred / s if s > 0 else 0.0
    x_new = x_pred + k * innovation
    p_new = (1.0 - k) * p_pred
    is_open = abs(innovation) > gate_sigma * (s**0.5)
    return KalmanStepResult(
        x_pred=x_pred,
        p_pred=p_pred,
        innovation=innovation,
        s=s,
        k=k,
        x_new=x_new,
        p_new=p_new,
        is_open=is_open,
    )


def cold_start_state(z_measured: float, r: float) -> tuple[float, float]:
    """Initial (x, p) for a room with no persisted prior state (first-ever
    run, or a gap exceeding KALMAN_STATE_MAX_GAP_HOURS) - seed the state at
    the live measurement with variance equal to the measurement noise itself,
    since there is no better prior belief to start from. Callers should never
    flag "open" on the same cycle a room is cold-started - there's nothing
    yet to compare the measurement against.
    """
    return z_measured, r


@dataclass
class ForwardFilterTrajectory:
    """Every step's forward predict+update quantities, kept in FULL across
    an entire array - unlike the live per-dispatch-cycle filter (see
    command_line.py::_build_room_kalman_opening_open), which only ever
    persists the latest (x, p) between cycles, the RTS backward pass below
    needs p_filt[t] AND p_pred[t+1] for every t, so the full trajectory
    must be retained. This is a fit-time/offline structure only - never
    used by the live dispatch-time filter."""

    x_pred: np.ndarray
    p_pred: np.ndarray
    x_filt: np.ndarray
    p_filt: np.ndarray
    innovation: np.ndarray


def kalman_forward_filter_array(
    x0: float,
    p0: float,
    x_pred: np.ndarray,
    z_measured: np.ndarray,
    q: float,
    r: float,
) -> ForwardFilterTrajectory:
    """Run kalman_predict_update once per element of x_pred/z_measured,
    keeping every step's own (x_pred, p_pred, x_filt, p_filt, innovation) -
    the offline/fit-time sibling of the live per-cycle loop, which only
    ever keeps the latest state for persistence. Reuses kalman_predict_update
    unchanged for the per-step math itself - not a second copy of the
    recurrence.

    :param x0: Prior state estimate before the first element (deg C).
    :param p0: Prior variance before the first element (deg C^2).
    :param x_pred: Externally-computed one-step predictions, one per
        timestep (deg C) - e.g. from predict_one_step_history.
    :param z_measured: Observed room temperature, one per timestep (deg C).
    :return: ForwardFilterTrajectory, same length as x_pred/z_measured.
    """
    n = len(x_pred)
    out_x_pred = np.zeros(n)
    out_p_pred = np.zeros(n)
    out_x_filt = np.zeros(n)
    out_p_filt = np.zeros(n)
    out_innovation = np.zeros(n)
    x_prev, p_prev = x0, p0
    for t in range(n):
        result = kalman_predict_update(
            x_prev=x_prev,
            p_prev=p_prev,
            x_pred=float(x_pred[t]),
            z_measured=float(z_measured[t]),
            q=q,
            r=r,
        )
        out_x_pred[t] = result.x_pred
        out_p_pred[t] = result.p_pred
        out_x_filt[t] = result.x_new
        out_p_filt[t] = result.p_new
        out_innovation[t] = result.innovation
        x_prev, p_prev = result.x_new, result.p_new
    return ForwardFilterTrajectory(
        x_pred=out_x_pred,
        p_pred=out_p_pred,
        x_filt=out_x_filt,
        p_filt=out_p_filt,
        innovation=out_innovation,
    )


def kalman_rts_smooth(trajectory: ForwardFilterTrajectory) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel backward smoothing pass over a full forward
    trajectory (see ForwardFilterTrajectory/kalman_forward_filter_array) -
    at fit time the WHOLE historical window is already available, so each
    point's estimate can be refined using both past AND future measurements,
    unlike the live per-cycle filter which only ever has the past.

    Base case: x_smooth[-1] = x_filt[-1], p_smooth[-1] = p_filt[-1] (the
    last point has no "future" to incorporate). For t from n-2 down to 0:
        C[t] = p_filt[t] / p_pred[t+1]
        x_smooth[t] = x_filt[t] + C[t] * (x_smooth[t+1] - x_pred[t+1])
        p_smooth[t] = p_filt[t] + C[t]**2 * (p_smooth[t+1] - p_pred[t+1])

    Same F~=1 linearization kalman_predict_update's own p_pred = p_prev + q
    already relies on (see this module's own docstring on the EKF-Jacobian
    argument) - these are the textbook linear-Gaussian RTS equations under
    that existing approximation, not a new modeling assumption.

    :return: (x_smooth, p_smooth), same length/order as trajectory.
    """
    n = len(trajectory.x_filt)
    x_smooth = np.zeros(n)
    p_smooth = np.zeros(n)
    x_smooth[-1] = trajectory.x_filt[-1]
    p_smooth[-1] = trajectory.p_filt[-1]
    for t in range(n - 2, -1, -1):
        p_pred_next = trajectory.p_pred[t + 1]
        c = trajectory.p_filt[t] / p_pred_next if p_pred_next > 0 else 0.0
        x_smooth[t] = trajectory.x_filt[t] + c * (x_smooth[t + 1] - trajectory.x_pred[t + 1])
        p_smooth[t] = trajectory.p_filt[t] + c**2 * (p_smooth[t + 1] - p_pred_next)
    return x_smooth, p_smooth


def smoothed_opening_flags(
    trajectory: ForwardFilterTrajectory,
    p_smooth: np.ndarray,
    r: float,
    gate_sigma: float = KALMAN_GATE_SIGMA,
) -> np.ndarray:
    """Per-timestep "probably open" boolean flags from a smoothed pass -
    used by the fit-time EM relabeling loop (see
    command_line.py::_em_relabel_opening_open), never by live dispatch.

    Reuses each step's own FORWARD innovation (the external one-step
    model's residual, z[t] - x_pred[t]) but gates it against the SMOOTHED
    uncertainty sqrt(p_smooth[t] + r) instead of the forward-only
    sqrt(p_pred[t] + r) kalman_predict_update itself gates against. Since
    p_smooth[t] &lt;= p_pred[t] always (conditioning on strictly more data -
    both past and future - can only reduce uncertainty, never increase it),
    this gate is never LESS sensitive than what a live forward-only
    detector would have flagged at the time - the right property for a
    higher-confidence, retroactive classification, without needing to
    re-run the external nonlinear predictor a second time against the
    smoothed trajectory.

    :return: Boolean array, same length as trajectory.
    """
    threshold = gate_sigma * np.sqrt(np.maximum(p_smooth, 0.0) + r)
    return np.abs(trajectory.innovation) > threshold


def predict_next_room_temperature_physics_family(
    current_temp: float,
    duty: float,
    outdoor_temp: float,
    nominal_power_w: float,
    dt_hours: float,
    volume: float = 15.0,
    supply_temperature: float = 35.0,
    carnot_efficiency: float = 0.4,
    base_loss: float = 0.045,
    heating_demand_kwh: float = 0.0,
    sense: str = "heat",
) -> float:
    """One-step-ahead room temperature prediction for a physics/simple-
    family room, reusing utils.simulate_physics_room_temperature_trajectory
    (the SAME recurrence real dispatch uses, already used elsewhere for
    exactly this "simulate forward, compare to reality" purpose) via a thin
    2-row wrapper - deliberately not a second hand-rolled copy of the
    formula that could drift out of sync with the real one.

    Assumes the window/door is closed (no OPENING_EXTRA_ACH ventilation
    boost) - this is the whole point: a live measurement diverging from this
    "assumed closed" prediction is the anomaly signal being looked for.

    :param current_temp: Live actual room temperature one cycle ago (deg C).
    :param duty: Fraction of nominal_power_w actually delivered (0-1).
    :param outdoor_temp: Live outdoor temperature (deg C).
    :param nominal_power_w: This room's nominal heat-pump power (W).
    :param dt_hours: Elapsed time since current_temp was measured (hours).
    :param heating_demand_kwh: Precomputed ongoing-loss demand for this one
        step (physics family only; 0.0 for "simple" family, matching
        simulate_physics_room_temperature_trajectory's own default).
    :return: Predicted room temperature one step ahead (deg C).
    """
    trajectory = simulate_physics_room_temperature_trajectory(
        initial_temp=current_temp,
        duty=np.array([duty, duty]),
        outdoor_temp=np.array([outdoor_temp, outdoor_temp]),
        nominal_power_w=nominal_power_w,
        dt_hours=dt_hours,
        volume=volume,
        base_loss=base_loss,
        supply_temperature=supply_temperature,
        carnot_efficiency=carnot_efficiency,
        heating_demand_kwh=np.array([heating_demand_kwh, heating_demand_kwh]),
        sense=sense,
    )
    return float(trajectory[1])


def predict_next_room_temperature_self_learning(
    model,
    room_name: str,
    df_house_fc: pd.DataFrame,
    df_room_fc: pd.DataFrame,
    current_temp: float,
) -> float | None:
    """One-step-ahead room temperature prediction for a self-learning-
    physics room, via a 1-row SelfLearningPhysicsModel.predict_recursive
    call - mathematically exactly a one-step predict (the loop runs once,
    starting from initial_room_states=current_temp, the room's live actual
    previous temperature) rather than a separate reimplementation.

    df_house_fc/df_room_fc must each be exactly 1 row, built the same way
    compute_self_learning_physics_forecast already builds its own forecast
    frames (outdoor_temp/wind_speed/dni/dhi/supply_temp/heatpump_duty/
    group_duty, plus blind_position if configured) - opening_open/door_open
    are deliberately never set here either, matching that function's own
    "assumed closed" forecast-time convention.

    :return: Predicted room temperature (deg C), or None if room_name isn't
        covered by this fitted model (e.g. too little history at refit time).
    """
    if room_name not in model.room_models_:
        return None
    result = model.predict_recursive(
        df_house_fc, {room_name: df_room_fc}, {room_name: current_temp}
    )
    return float(result["room_temp"][room_name][0])
