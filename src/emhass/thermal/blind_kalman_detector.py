"""
Blind Kalman Detector - Sensorless, continuous blind/shading-position inference
=================================================================================

A per-room scalar Kalman filter that infers a self-learning-physics room's
blind/shading position (continuous, 0=open..1=fully closed) purely from
thermal behaviour, for rooms with NO configured heatpump_room_blind_sensors
entry - sibling of opening_kalman_detector.py's window/door detector, but
for a genuinely different physical quantity with a different state model.

Self-learning-physics rooms ONLY (see below for why physics-family rooms
are out of scope), both live (per naive-mpc-optim cycle,
command_line.py::_build_room_kalman_blind_position) and retroactive (per
self-learning-physics-refit, command_line.py::_em_relabel_blind_position).

Why this needs its OWN state model, not opening_kalman_detector.py's:
unlike a window/door's binary open/closed state (where the live filter's
state IS the room's own temperature, and "open" is a threshold test on the
innovation), blind position is a genuinely different physical quantity that
is NEVER directly observed - only ever inferred algebraically from a
residual. The Kalman STATE here is position itself (0-1), tracked via a
persistence/random-walk model (x_pred[t] = x_filt[t-1] - "it probably
hasn't moved since last cycle"; no independent external predictor exists
for position the way one does for temperature), not the room's temperature.

Why blind position is only ever OBSERVABLE when there's sun: the model's
prediction is exactly `bias/duty/... + beta * blind_position * dni` -
closing the blind only ever changes the DIRECT solar gain term, which is
identically zero whenever dni=0 (night, heavy cloud). At those times there
is genuinely no information about blind position in the residual at all -
a physical fact about the observation model, not a limitation of this
filter's design.

Why self-learning-physics rooms only: physics-family rooms' own live one-
step predictor (predict_next_room_temperature_physics_family, this
package's opening_kalman_detector.py) is a thin wrapper around
utils.simulate_physics_room_temperature_trajectory, which has NO solar/
window/blind term at all (it only ever receives a caller-supplied
heating_demand_kwh, defaulting to zero) - there is no residual sensitivity
to blind position whatsoever on that path, so inversion is structurally
impossible without first extending that predictor. Out of scope here.

Exact algebraic derivation (see also command_line.py::_em_relabel_blind_position's
own docstring for how this is used at fit time): SelfLearningPhysicsModel's
prediction is a plain dot product (theta @ x_row) and none of its other
features' nonlinear pieces (delta_supply/delta_env clipping) depend on
blind_position - so for a fixed history row, changing ONLY blind_position
changes the prediction by exactly beta*blind_position*dni, nothing else.
Define pred_open(t) = the model's prediction with blind_position forced to
0 (predict_room_temperature_blind_open_baseline below). Then:

    true_position(t) = (actual_temp(t) - pred_open(t)) / (beta * dni(t))

for dni(t) above an informative floor - see invert_blind_position_from_residual.
Sign check: closing the blind cuts solar gain, so beta fits negative; a
real closed blind + sun means actual < pred_open (cooler than "assumed
open"), so residual < 0, and raw = residual/(beta*dni) = (neg)/(neg*pos) =
positive - correctly trends toward "closed" (1).

For an unsensored room with no history-ever blind_position column, beta is
never identifiable from a zero-variance feature (RLS can't move a constant-
zero column's coefficient) - see bootstrap_raw_blind_signal_from_residual
for the iteration-0-only heuristic that gets around this.

This module is pure math - zero HA/live-fetch/persistence code, matching
opening_kalman_detector.py's own convention. All HA/persistence
orchestration lives in command_line.py. kalman_rts_smooth (RTS backward
pass) is reused UNMODIFIED from opening_kalman_detector.py by callers -
it only ever reads a ForwardFilterTrajectory's x_pred/p_pred/x_filt/p_filt
generically, with no dependency on how x_pred was produced, so it is not
re-imported/re-exported here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emhass.thermal.opening_kalman_detector import (
    ForwardFilterTrajectory,
    kalman_predict_update,
    predict_next_room_temperature_self_learning,
)

# Below this DNI (W/m2), dividing residual by dni amplifies ordinary model
# noise into a meaninglessly large measurement variance - well under a
# typical clear-sky noon DNI (600-1000 W/m2), high enough to exclude dawn/
# dusk/heavily-overcast readings where there's genuinely no information.
BLIND_DNI_INFORMATIVE_FLOOR_WM2 = 50.0

# Blind position has no real process noise when untouched (it's exactly
# constant between user actions) - a small nonzero value lets the filter
# track a real change within a few cycles rather than freezing under an
# over-confident prior.
BLIND_KALMAN_Q = 0.0009

# Floor/ceiling on the per-timestep measurement noise (see
# resolve_blind_measurement_noise) - the floor prevents r->0 (and K~=1, an
# overconfident single-cycle jump) at very high dni/strong beta, mirroring
# SELF_LEARNING_KALMAN_R_FLOOR_C2's own role; the ceiling is a defensive
# numerical bound only (the DNI floor already excludes the worst cases).
BLIND_KALMAN_R_FLOOR = 0.0025
BLIND_KALMAN_R_CEILING = 1.0

# Pure divide-by-zero safety floor on |beta| - NOT a trust threshold. Real
# trust in a live/retroactive estimate is mediated entirely through
# resolve_blind_measurement_noise's own 1/beta^2 scaling (a weak beta
# inflates r, which self-penalizes) and the live dispatch confidence gate
# (see command_line.py's own use of BLIND_KALMAN_DISPATCH_MAX_P).
BLIND_KALMAN_BETA_EPSILON = 1e-6

# Wide "no idea yet" prior variance for a room with no persisted state -
# unlike cold_start_state (opening_kalman_detector.py), blind position is
# never directly observed, so there is no live measurement to seed x from;
# assume open (0.0), matching every other "unknown = open/inert" default
# in this codebase (_physics_features's own blind_position fallback, etc.).
BLIND_KALMAN_COLD_START_P = 1.0

# Live dispatch confidence gate (see command_line.py::
# _build_room_kalman_blind_position/_build_room_blind_positions_with_kalman_fallback):
# only ever reachable after several consecutive INFORMATIVE (sunny) cycles
# of consistent evidence, never from a single reading - deliberately
# conservative, since a wrong blind-position guess has no safe direction
# the way pausing heat for a possibly-open window does.
BLIND_KALMAN_DISPATCH_MAX_P = 0.01


def blind_cold_start_state() -> tuple[float, float]:
    """Initial (x, p) for a room with no persisted prior blind-position
    state (first-ever run, or a stale gap) - see module docstring for why
    this can't reuse opening_kalman_detector.cold_start_state (there is no
    z_measured to seed a position from)."""
    return 0.0, BLIND_KALMAN_COLD_START_P


def predict_room_temperature_blind_open_baseline(
    model,
    room_name: str,
    df_house_fc: pd.DataFrame,
    df_room_fc: pd.DataFrame,
    current_temp: float,
) -> float | None:
    """pred_open(t) from this module's own docstring derivation: this
    room's one-step temperature prediction with blind_position FORCED to
    0.0, regardless of whatever df_room_fc's own "blind_position" column
    (if any) already holds - every inversion in this module compares
    against this specific, consistent baseline. Thin wrapper around
    predict_next_room_temperature_self_learning (opening_kalman_detector.py),
    not a reimplementation.

    :return: Predicted room temperature (deg C) assuming the blind is fully
        open, or None if room_name isn't covered by this fitted model.
    """
    df_room_fc_open = df_room_fc.assign(blind_position=0.0)
    return predict_next_room_temperature_self_learning(
        model, room_name, df_house_fc, df_room_fc_open, current_temp
    )


def _informative_mask(dni: np.ndarray, dni_floor: float) -> np.ndarray:
    return dni > dni_floor


def invert_blind_position_from_residual(
    residual: np.ndarray,
    dni: np.ndarray,
    beta: float,
    dni_floor: float = BLIND_DNI_INFORMATIVE_FLOOR_WM2,
) -> np.ndarray:
    """Exact algebraic inversion (see module docstring): raw(t) =
    residual(t) / (beta * dni(t)) for dni(t) above dni_floor, else NaN -
    clipped to [0, 1] (NaN preserved through np.clip). Only valid once
    beta is meaningfully nonzero - see BLIND_KALMAN_BETA_EPSILON and
    bootstrap_raw_blind_signal_from_residual for the iteration-0 case.
    """
    residual = np.asarray(residual, dtype=float)
    dni = np.asarray(dni, dtype=float)
    raw = np.full(residual.shape, np.nan)
    mask = _informative_mask(dni, dni_floor)
    raw[mask] = residual[mask] / (beta * dni[mask])
    return np.clip(raw, 0.0, 1.0)


def _robust_normalize_to_unit_interval(raw: np.ndarray) -> np.ndarray:
    """5th/95th-percentile clip-then-rescale onto [0, 1] - a single early
    outlier residual can't blow up the whole room's scale. NaN propagates
    naturally through clip/arithmetic (no explicit masking needed).
    Degenerate input (fewer than 2 finite values, or zero-variance) returns
    an all-NaN array of the same shape rather than a meaningless 0/1
    constant."""
    raw = np.asarray(raw, dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size < 2:
        return np.full(raw.shape, np.nan)
    lo, hi = np.nanpercentile(finite, 5.0), np.nanpercentile(finite, 95.0)
    if hi <= lo:
        return np.full(raw.shape, np.nan)
    return (np.clip(raw, lo, hi) - lo) / (hi - lo)


def bootstrap_raw_blind_signal_from_residual(
    residual: np.ndarray,
    dni: np.ndarray,
    dni_floor: float = BLIND_DNI_INFORMATIVE_FLOOR_WM2,
) -> np.ndarray:
    """Iteration-0-ONLY heuristic (see module docstring), used only while
    beta is still unidentified (a zero-variance blind_x_dni feature column
    means RLS has never moved beta away from 0): raw(t) = -residual(t)/
    dni(t) for dni(t) above dni_floor else NaN, then percentile-normalized
    onto [0, 1]. NOT a calibrated position estimate - its only job is to
    give the next fit a nonzero-variance blind_x_dni column so beta becomes
    identifiable, at which point invert_blind_position_from_residual takes
    over.
    """
    residual = np.asarray(residual, dtype=float)
    dni = np.asarray(dni, dtype=float)
    raw = np.full(residual.shape, np.nan)
    mask = _informative_mask(dni, dni_floor)
    raw[mask] = -residual[mask] / dni[mask]
    return _robust_normalize_to_unit_interval(raw)


def resolve_blind_measurement_noise(
    residual_std_c: float, beta: float, dni: np.ndarray | float
) -> np.ndarray | float:
    """r(t) = clip((residual_std_c / (|beta| * dni(t)))**2, R_FLOOR, R_CEILING).
    Unlike every other fixed-constant r in this codebase, this one
    genuinely varies per-timestep - dividing a roughly-fixed temperature
    residual noise by dni and by beta amplifies it, so both a low-sun
    moment AND a weakly-identified beta self-penalize here, without
    needing a separate coefficient-significance test this RLS fit doesn't
    expose. Robust to dni containing zeros / beta being exactly 0 (both
    map to the ceiling, never raise) - callers are not required to
    pre-filter to informative timesteps only.
    """
    dni_arr = np.asarray(dni, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (residual_std_c / (np.abs(beta) * dni_arr)) ** 2
    r = np.nan_to_num(r, nan=BLIND_KALMAN_R_CEILING, posinf=BLIND_KALMAN_R_CEILING)
    r = np.clip(r, BLIND_KALMAN_R_FLOOR, BLIND_KALMAN_R_CEILING)
    return float(r) if np.ndim(dni) == 0 else r


def kalman_forward_filter_with_persistence(
    x0: float,
    p0: float,
    z_measured: np.ndarray,
    q: float,
    r: np.ndarray | float,
) -> ForwardFilterTrajectory:
    """Array-based forward Kalman filter for a PERSISTENCE state model -
    x_pred[t] = x_filt[t-1] computed INTERNALLY each step, unlike
    opening_kalman_detector.kalman_forward_filter_array, whose x_pred is an
    array supplied by the caller from an independent per-t temperature
    predictor (no such independent predictor exists for blind position -
    the only sensible model is "it probably hasn't moved since last
    cycle"). z_measured[t] may be NaN - a genuine missing/uninformative
    (low-DNI) timestep, handled by growing p via +q only, leaving x/
    innovation completely untouched, never coerced to a wrong reading of 0
    - mathematically equivalent to a kalman_predict_update call with
    r=infinity (k=0), just without the numerical risk of an actual
    infinite r. r may be a scalar or a per-timestep array (this
    application's own r varies with dni(t), see
    resolve_blind_measurement_noise).

    :param x0: Prior position estimate before the first element (0-1).
    :param p0: Prior variance before the first element.
    :param z_measured: Raw per-timestep position measurements (0-1, may
        contain NaN for uninformative timesteps).
    :param q: Process noise variance (same units as p0).
    :param r: Measurement noise variance, scalar or one per timestep.
    :return: ForwardFilterTrajectory, same length as z_measured.
    """
    n = len(z_measured)
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), (n,))
    out_x_pred = np.zeros(n)
    out_p_pred = np.zeros(n)
    out_x_filt = np.zeros(n)
    out_p_filt = np.zeros(n)
    out_innovation = np.zeros(n)
    x_prev, p_prev = x0, p0
    for t in range(n):
        z_t = float(z_measured[t])
        if not np.isfinite(z_t):
            p_pred_t = p_prev + q
            out_x_pred[t] = x_prev
            out_p_pred[t] = p_pred_t
            out_x_filt[t] = x_prev
            out_p_filt[t] = p_pred_t
            out_innovation[t] = 0.0
            x_prev, p_prev = x_prev, p_pred_t
            continue
        result = kalman_predict_update(
            x_prev=x_prev,
            p_prev=p_prev,
            x_pred=x_prev,
            z_measured=z_t,
            q=q,
            r=float(r_arr[t]),
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


def smoothed_blind_position(x_smooth: np.ndarray) -> np.ndarray:
    """The RTS-smoothed state IS the answer for blind position - no gating
    step needed (unlike smoothed_opening_flags's binary threshold test).
    Just clip to [0, 1]: the smoother's own linear-Gaussian math has no
    knowledge of that physical bound."""
    return np.clip(x_smooth, 0.0, 1.0)
