#!/usr/bin/env python3

"""
PV Shading/Horizon Kalman Detector
===================================

Learns a per-direction horizon profile (the elevation angle below which
the sun is physically obstructed - trees, chimneys, neighbouring roofs)
from historical PV production, reusing the same scalar Kalman
innovation-gate math already used for sensorless window/door detection
(see emhass.thermal.opening_kalman_detector.kalman_predict_update).

A horizon is a per-azimuth THRESHOLD, not a continuously drifting
quantity, so this module does not run the gate recursively across time -
there is no meaningful "state evolving from one timestep to the next"
here. Instead, kalman_predict_update is applied POINTWISE to each
historical instant ("is actual output anomalously low relative to the
unobstructed clear-sky expectation right now") and the flagged instants
are aggregated per azimuth bin into the horizon profile, refined across
refits with a forgetting-factor blend (the same weighted-blend shape as
self_learning_physics_refit's own incremental update, at a much faster
default rate here - see aggregate_horizon_profile's own docstring for why).

Pure math - zero HA/persistence code, matching this package's existing
thermal/*.py convention (see opening_kalman_detector.py's own module
docstring). All HA/persistence orchestration lives in command_line.py
(see refit_pv_horizon_model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from emhass.thermal.opening_kalman_detector import kalman_predict_update

# 24 bins of 15 degrees each - fine enough to localize a specific
# obstruction (a chimney, a single tree) without so many bins that any one
# of them starves for data over a realistic refit window.
AZIMUTH_BIN_WIDTH_DEG = 15

# Ratio (actual / expected clear-sky) noise variance under normal, unshaded
# conditions - inverter losses, soiling, and clear-sky-model imperfection
# typically keep real output within ~10% of PVLib's clear-sky estimate on
# a genuinely clear day. Applied as a fixed-per-call variance (see
# classify_shaded_instants's own docstring for why this isn't recursive).
SHADING_GATE_R = 0.01

# No temporal drift term - each instant is classified independently
# against the same fixed "expect full output" prior, not filtered forward
# from the previous instant's belief.
SHADING_GATE_Q = 0.0

# Number of standard deviations the ratio must miss by to be flagged
# "shaded" - same default as opening_kalman_detector's own
# KALMAN_GATE_SIGMA, reused for consistency (not re-imported, since a
# different physical unit/meaning here warrants its own name).
SHADING_GATE_SIGMA = 3.0

# Below this expected clear-sky power, the actual/expected ratio's own
# noise floor is too high to trust (near sunrise/sunset/night, or a very
# small system) - excluded from classification entirely rather than risk
# false positives skewing the horizon at exactly the elevations that
# matter most.
MIN_EXPECTED_POWER_W = 50.0

# A bin needs at least this many valid (non-excluded) observations in a
# single refit window before its estimate is trusted enough to update the
# persisted profile - otherwise the sun simply never reached that azimuth
# (enough) this window, and the previous value is kept unchanged rather
# than let a sparse window regress a well-established estimate.
MIN_OBSERVATIONS_PER_BIN = 20


def classify_shaded_instants(actual: pd.Series, expected_clear_sky: pd.Series) -> pd.Series:
    """Per-timestep boolean: True where actual output is anomalously low
    relative to the unobstructed clear-sky expectation at that instant.

    Applies kalman_predict_update POINTWISE - x_prev=x_pred=1.0 ("expect
    full output"), p_prev=SHADING_GATE_R fixed on every call - rather than
    recursively carrying (x, p) forward across timesteps. A horizon is a
    threshold, not a drifting continuous quantity, so there is no
    meaningful per-instant "previous belief" to filter forward; this
    reuses the gate math and statistical philosophy of
    opening_kalman_detector's own innovation gate, applied as a stateless
    per-instant test instead.

    Only a DEFICIT counts as shading - a surplus (actual > expected) is
    weather-model error in the other direction, not evidence of an
    obstruction, so the underlying two-sided gate is narrowed to one side
    here. Timestamps where expected_clear_sky < MIN_EXPECTED_POWER_W are
    excluded (returned as False) since the ratio's own noise floor there
    is too high to classify reliably.

    :param actual: Measured PV power (W), indexed by timestamp.
    :type actual: pd.Series
    :param expected_clear_sky: Unobstructed clear-sky PVLib simulation
        output (W) for the same timestamps.
    :type expected_clear_sky: pd.Series
    :return: Boolean Series, same index as actual/expected_clear_sky.
    :rtype: pd.Series
    """
    ratio = actual / expected_clear_sky.replace(0.0, np.nan)
    valid = expected_clear_sky >= MIN_EXPECTED_POWER_W
    shaded = pd.Series(False, index=actual.index)
    for ts in actual.index[valid]:
        r = ratio.loc[ts]
        if pd.isna(r):
            continue
        result = kalman_predict_update(
            x_prev=1.0,
            p_prev=SHADING_GATE_R,
            x_pred=1.0,
            z_measured=float(r),
            q=SHADING_GATE_Q,
            r=SHADING_GATE_R,
            gate_sigma=SHADING_GATE_SIGMA,
        )
        shaded.loc[ts] = result.is_open and r < 1.0
    return shaded


def aggregate_horizon_profile(
    shaded: pd.Series,
    azimuth: pd.Series,
    elevation: pd.Series,
    previous_profile: dict[str, float] | None,
    forgetting_factor: float,
) -> dict[str, float]:
    """Bin flagged shaded instants by azimuth and blend into a horizon
    profile.

    For each AZIMUTH_BIN_WIDTH_DEG-wide bin with enough valid observations
    this window (see MIN_OBSERVATIONS_PER_BIN): if any were flagged
    shaded, this window's evidence is a LOWER bound on the horizon (the
    highest elevation seen blocked - it's obstructed at least up to
    there). If none were flagged shaded, this window's evidence is an
    UPPER bound instead (the lowest elevation the sun was actually
    observed at, unshaded - the true horizon can't be higher than that,
    or that reading would have been blocked too). Either way this
    window's estimate is blended with previous_profile:
    new = forgetting_factor * previous + (1 - forgetting_factor) * this_window.

    forgetting_factor is deliberately much lower here than a live,
    every-cycle RLS update (e.g. self_learning_physics_refit's own 0.995
    default) - this only runs once per periodic refit (weekly-ish), so a
    value that slow would take the better part of a year to reflect a
    real obstruction. A bin with fewer than MIN_OBSERVATIONS_PER_BIN valid
    observations this window (the sun didn't reach that azimuth enough to
    say anything new) keeps its previous value unchanged instead of being
    blended from too little data.

    :param shaded: Boolean Series from classify_shaded_instants.
    :type shaded: pd.Series
    :param azimuth: Solar azimuth (degrees, 0-360) for the same timestamps.
    :type azimuth: pd.Series
    :param elevation: Solar elevation (degrees) for the same timestamps.
    :type elevation: pd.Series
    :param previous_profile: The persisted profile from the last refit -
        {"<bin_start_deg>": horizon_elevation_deg}, or None on a first-ever
        refit (every bin then starts from an implicit 0.0 - "no known
        obstruction").
    :type previous_profile: dict[str, float] | None
    :param forgetting_factor: Weight on the previous profile, in [0, 1].
    :type forgetting_factor: float
    :return: The updated profile, same shape as previous_profile.
    :rtype: dict[str, float]
    """
    previous_profile = previous_profile or {}
    bins = np.arange(0, 360, AZIMUTH_BIN_WIDTH_DEG)
    profile: dict[str, float] = {}
    for bin_start in bins:
        key = str(int(bin_start))
        in_bin = (azimuth >= bin_start) & (azimuth < bin_start + AZIMUTH_BIN_WIDTH_DEG)
        n_obs = int(in_bin.sum())
        prev_value = float(previous_profile.get(key, 0.0))
        if n_obs < MIN_OBSERVATIONS_PER_BIN:
            profile[key] = prev_value
            continue
        bin_elevations = elevation[in_bin]
        bin_shaded_elevations = bin_elevations[shaded[in_bin]]
        if not bin_shaded_elevations.empty:
            window_value = float(bin_shaded_elevations.max())
        else:
            window_value = float(bin_elevations.min())
        profile[key] = forgetting_factor * prev_value + (1 - forgetting_factor) * window_value
    return profile
