#!/usr/bin/env python3

"""
PV Shading/Horizon Kalman Detector
===================================

Learns a per-direction, per-season horizon profile (the elevation angle
below which the sun is physically obstructed, and the fraction of direct
sun that still gets through below it - trees, chimneys, neighbouring
roofs) from historical PV production, reusing the same scalar Kalman
innovation-gate math already used for sensorless window/door detection
(see emhass.thermal.opening_kalman_detector.kalman_predict_update). The
transmittance fraction lets a partially-transmissive obstruction (a tree
canopy) be told apart from a hard one (a chimney, a roofline) instead of
treating every obstruction as a full block; the season split lets a
deciduous tree's leaf-on/leaf-off difference be learned instead of
averaged away.

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

# Standard meteorological seasons (not astronomical solstice/equinox
# dates) - matches how professional shading assessments describe
# leaf-on/leaf-off measurement splits. A bin's horizon/transmittance is
# learned independently per season: a deciduous tree lets far more direct
# sun through in winter (leaf-off) than summer (leaf-on), and lumping
# both into one estimate would systematically misrepresent both.
SEASON_LABELS = ("winter", "spring", "summer", "autumn")
_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# A (bin, season) cell can clear MIN_OBSERVATIONS_PER_BIN while having
# very few *shaded* instants among them - averaging a transmittance
# estimate from e.g. 2 points is noise. Below this count the elevation
# still updates (still meaningful from just the shaded elevations' max),
# but transmittance is left at its previous value this round.
MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE = 5

# Fraction of expected clear-sky DNI assumed to get through below a
# bin/season's learned horizon elevation, before any shaded instant has
# ever been observed there to measure it directly - matches this
# feature's original (pre-transmittance) hard-block behavior, so a
# freshly-learned obstruction still starts conservative.
DEFAULT_TRANSMITTANCE = 0.0

_COLD_START_ENTRY = {"elevation": 0.0, "transmittance": DEFAULT_TRANSMITTANCE}


def season_labels_for_index(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized month -> meteorological season label for a
    DatetimeIndex. Single source of truth shared by
    aggregate_horizon_profile (fitting) and
    forecast.py::_apply_pv_horizon_mask (applying), so both sides agree
    on which season a given timestamp belongs to.
    """
    return pd.Series(index.month, index=index).map(_SEASON_BY_MONTH)


def normalize_bin_entry(entry) -> dict[str, dict[str, float]]:
    """Normalize one persisted profile bin entry into the current
    season-nested shape, tolerating every format this feature has used:

    - bare float/int (original, pre-transmittance): broadcast to every
      season as a hard block (DEFAULT_TRANSMITTANCE).
    - flat {"elevation":.., "transmittance":..} (pre-season): broadcast
      to every season as a uniform prior - the first time each season is
      actually refit under the season-aware logic it starts from this
      shared prior, then seasons naturally drift apart as real
      per-season evidence arrives.
    - already season-nested {"<season>": {"elevation":.., "transmittance":..}}:
      returned as-is.
    - None/missing: {}.

    Read-only broadcast (the same sub-dict object referenced across
    seasons for the first two cases) is safe: callers only ever read
    from a normalized previous-profile, never mutate it - every new
    profile entry aggregate_horizon_profile produces is a fresh dict.
    """
    if entry is None:
        return {}
    if isinstance(entry, int | float):
        flat = {"elevation": float(entry), "transmittance": DEFAULT_TRANSMITTANCE}
        return dict.fromkeys(SEASON_LABELS, flat)
    if isinstance(entry, dict) and entry and isinstance(next(iter(entry.values())), dict):
        return entry
    return dict.fromkeys(SEASON_LABELS, entry)


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
    actual: pd.Series,
    expected_clear_sky: pd.Series,
    previous_profile: dict | None,
    forgetting_factor: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Bin flagged shaded instants by azimuth and season, and blend into
    a horizon profile.

    For each (AZIMUTH_BIN_WIDTH_DEG-wide azimuth bin, meteorological
    season) cell with enough valid observations this window (see
    MIN_OBSERVATIONS_PER_BIN): if any were flagged shaded, this window's
    evidence is a LOWER bound on the horizon elevation (the highest
    elevation seen blocked - it's obstructed at least up to there), and
    the mean actual/expected ratio among just those shaded instants is
    this window's evidence for the transmittance (how much light still
    gets through below that elevation - 0 for a hard obstruction, higher
    for a tree canopy). If none were flagged shaded, this window's
    elevation evidence is an UPPER bound instead (the lowest elevation
    the sun was actually observed at, unshaded), and there is no
    transmittance evidence at all this round (nothing below the horizon
    was observed to measure). Either way, whichever fields have new
    evidence are blended with previous_profile:
    new = forgetting_factor * previous + (1 - forgetting_factor) * this_window.

    forgetting_factor is deliberately much lower here than a live,
    every-cycle RLS update (e.g. self_learning_physics_refit's own 0.995
    default) - this only runs once per periodic refit (weekly-ish), so a
    value that slow would take the better part of a year to reflect a
    real obstruction. A cell with fewer than MIN_OBSERVATIONS_PER_BIN
    valid observations this window (the common case for 3 of the 4
    seasons on any given refit - a single refit window only ever falls
    within 1-2 seasons) keeps its previous value entirely unchanged
    instead of being blended from too little data; a whole direction
    only converges across many periodic refits spread over a year, same
    as the elevation estimate itself already did before seasons existed.

    :param shaded: Boolean Series from classify_shaded_instants.
    :type shaded: pd.Series
    :param azimuth: Solar azimuth (degrees, 0-360) for the same timestamps.
    :type azimuth: pd.Series
    :param elevation: Solar elevation (degrees) for the same timestamps.
    :type elevation: pd.Series
    :param actual: Measured PV power (W) for the same timestamps.
    :type actual: pd.Series
    :param expected_clear_sky: Unobstructed clear-sky PVLib simulation
        output (W) for the same timestamps.
    :type expected_clear_sky: pd.Series
    :param previous_profile: The persisted profile from the last refit,
        in any format normalize_bin_entry accepts, or None on a
        first-ever refit.
    :type previous_profile: dict | None
    :param forgetting_factor: Weight on the previous profile, in [0, 1].
    :type forgetting_factor: float
    :return: {"<bin_start_deg>": {"<season>": {"elevation": .., "transmittance": ..}}}
    :rtype: dict[str, dict[str, dict[str, float]]]
    """
    previous_profile = previous_profile or {}
    season = season_labels_for_index(elevation.index)
    ratio = (actual / expected_clear_sky).clip(lower=0.0, upper=1.0)
    bins = np.arange(0, 360, AZIMUTH_BIN_WIDTH_DEG)
    profile: dict[str, dict[str, dict[str, float]]] = {}
    for bin_start in bins:
        key = str(int(bin_start))
        in_bin = (azimuth >= bin_start) & (azimuth < bin_start + AZIMUTH_BIN_WIDTH_DEG)
        prev_seasons = normalize_bin_entry(previous_profile.get(key))
        bin_profile = dict(prev_seasons)
        for s in SEASON_LABELS:
            in_cell = in_bin & (season == s)
            n_obs = int(in_cell.sum())
            if n_obs < MIN_OBSERVATIONS_PER_BIN:
                continue
            prev_entry = prev_seasons.get(s, _COLD_START_ENTRY)
            cell_elevations = elevation[in_cell]
            cell_shaded_mask = shaded[in_cell]
            cell_shaded_elevations = cell_elevations[cell_shaded_mask]
            if not cell_shaded_elevations.empty:
                window_elevation = float(cell_shaded_elevations.max())
            else:
                window_elevation = float(cell_elevations.min())
            new_elevation = (
                forgetting_factor * prev_entry["elevation"]
                + (1 - forgetting_factor) * window_elevation
            )
            new_transmittance = prev_entry["transmittance"]
            if cell_shaded_mask.sum() >= MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE:
                window_transmittance = float(ratio[in_cell][cell_shaded_mask].mean())
                new_transmittance = (
                    forgetting_factor * prev_entry["transmittance"]
                    + (1 - forgetting_factor) * window_transmittance
                )
            bin_profile[s] = {"elevation": new_elevation, "transmittance": new_transmittance}
        profile[key] = bin_profile
    return profile
