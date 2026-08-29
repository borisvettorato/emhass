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
are aggregated per azimuth anchor into the horizon profile, refined across
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
from pvlib.irradiance import aoi
from pvlib.solarposition import get_solarposition

from emhass.thermal.opening_kalman_detector import kalman_predict_update

# Spacing between fitted azimuth anchors - fine enough to localize a
# specific obstruction (a chimney, a single tree) without so many anchors
# that any one of them starves for data over a realistic refit window.
# Not a bin boundary: aggregate_horizon_profile fits each anchor from a
# soft, overlapping kernel window (see _azimuth_kernel_weight), and
# interpolate_horizon_profile queries a smooth function of azimuth between
# anchors - there is no discrete "which box am I in" step anywhere. 15 is
# a multiple of 5, so every anchor key a profile persisted before this
# spacing changed from 15 remains a valid anchor going forward - no
# migration needed, it just interpolates coarser until refit again.
AZIMUTH_ANCHOR_SPACING_DEG = 5

# Width of the Gaussian kernel used to softly weight each observation's
# contribution to an anchor (see _azimuth_kernel_weight) - roughly matches
# the span over which a solar panel's own physical width smears a sharp
# obstruction edge into a gradual actual/expected transition, so the
# smoothing reveals a real physical effect rather than manufacturing one.
AZIMUTH_KERNEL_BANDWIDTH_DEG = 10.0

# Observations beyond this many kernel bandwidths get treated as fully
# outside an anchor's window (weight ~1.1% or less there) - keeps each
# anchor's effective window finite (roughly +/-30 degrees) instead of
# every observation on Earth technically contributing an epsilon to every
# anchor. Also reused by interpolate_horizon_profile as the weight of its
# virtual cold-start anchor - see that function's own docstring for why.
AZIMUTH_KERNEL_CUTOFF = 3.0

# Sampling step used when rendering the now-continuous profile as a chart
# (render_horizon_polar_grid) and when recomputing geometric blindness at
# chart resolution - independent of AZIMUTH_ANCHOR_SPACING_DEG (which
# governs FITTING/persistence): fine enough that consecutive wedges are
# visually indistinguishable from a smooth gradient, cheap since it's just
# interpolate_horizon_profile evaluated at a few hundred points.
AZIMUTH_RENDER_SPACING_DEG = 2

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

# An anchor needs at least this much effective weight (the sum of each
# valid, non-excluded observation's kernel weight - see
# _azimuth_kernel_weight, equivalent to a raw count when every observation
# sits dead-center on the anchor) in a single refit window before its
# estimate is trusted enough to update the persisted profile - otherwise
# the sun simply never reached that azimuth (enough) this window, and the
# previous value is kept unchanged rather than let a sparse window regress
# a well-established estimate.
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

# An (anchor, season) cell can clear MIN_OBSERVATIONS_PER_BIN while having
# very little *shaded* effective weight among them - averaging a
# transmittance estimate from e.g. 2 weakly-weighted points is noise.
# Below this weight the elevation still updates (still meaningful from
# just the shaded elevations' max), but transmittance is left at its
# previous value this round.
MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE = 5

# Fraction of expected clear-sky DNI assumed to get through below an
# anchor/season's learned horizon elevation, before any shaded instant has
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
    - None/missing, or an empty dict (an anchor that exists in the profile
      but has never had any season clear MIN_OBSERVATIONS_PER_BIN yet -
      aggregate_horizon_profile can produce exactly this for a fresh
      anchor with no previous_profile): {}.

    The empty-dict case matters: an empty dict is falsy in Python, but
    it's NOT the same as a bare-float/flat entry of 0 - broadcasting {}
    itself to every season (dict.fromkeys(SEASON_LABELS, {})) would make
    every season's .get(some_season, cold_start) return {} instead of
    falling back to cold_start (since the key would exist, just mapped to
    an empty dict), and callers indexing e["elevation"] on that would
    crash. Treating it the same as None/missing is what every caller
    actually wants: "no evidence for this anchor at all".

    Read-only broadcast (the same sub-dict object referenced across
    seasons for the flat/bare-float cases) is safe: callers only ever read
    from a normalized previous-profile, never mutate it - every new
    profile entry aggregate_horizon_profile produces is a fresh dict.
    """
    if entry is None:
        return {}
    if isinstance(entry, dict) and not entry:
        return {}
    if isinstance(entry, int | float):
        flat = {"elevation": float(entry), "transmittance": DEFAULT_TRANSMITTANCE}
        return dict.fromkeys(SEASON_LABELS, flat)
    if isinstance(entry, dict) and isinstance(next(iter(entry.values())), dict):
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


def compute_geometrically_blind_azimuths(
    surface_tilt: float,
    surface_azimuth: float,
    latitude: float,
    longitude: float,
    spacing_deg: float = AZIMUTH_ANCHOR_SPACING_DEG,
) -> set[int]:
    """Azimuth anchors (spacing_deg apart) where direct sunlight can never
    reach this panel's front face, regardless of any external obstruction -
    two purely geometric/astronomical reasons, neither needs any measurement:

    - self-shading: the sun is behind the panel's own tilted plane
      (angle-of-incidence >= 90 degrees).
    - the sun's own path at this latitude/longitude never passes through
      that azimuth above the horizon at all, for any panel.

    Without this, an anchor that never clears MIN_OBSERVATIONS_PER_BIN
    because the panel simply can't ever see that direction stays at its
    cold-start default (elevation=0, transmittance=0) forever - identical,
    in a rendered chart, to a direction that was actually checked and found
    clear. This tells the two apart ahead of any measurement.

    spacing_deg defaults to the same spacing aggregate_horizon_profile fits
    at (for the persisted-profile/live-masking use), but a caller doing its
    own fine-resolution rendering (e.g. render_horizon_polar_grid) can pass
    a smaller value - geometric blindness is an exact yes/no astronomical
    fact at every azimuth, so it's recomputed directly at whatever
    resolution is needed rather than interpolated from a coarser set.

    Sweeps one fixed reference year of 15-minute solar positions (~35k
    points - pure trig via pvlib, no irradiance/weather modeling, cheap)
    and buckets by azimuth; an anchor is blind only if NO timestamp all
    year ever has both solar elevation > 0 (daytime) and angle-of-incidence
    < 90 (front face lit) there.

    :param surface_tilt: Panel tilt from horizontal, degrees.
    :type surface_tilt: float
    :param surface_azimuth: Panel azimuth, degrees (0-360).
    :type surface_azimuth: float
    :param latitude: Site latitude, degrees.
    :type latitude: float
    :param longitude: Site longitude, degrees.
    :type longitude: float
    :param spacing_deg: Spacing between azimuth anchors, degrees.
    :type spacing_deg: float
    :return: The set of azimuth anchor angles that are geometrically blind
        for this panel.
    :rtype: set[int]
    """
    times = pd.date_range("2023-01-01", "2023-12-31 23:45", freq="15min", tz="UTC")
    solpos = get_solarposition(times, latitude, longitude)
    daytime = solpos["elevation"] > 0
    illuminated = aoi(surface_tilt, surface_azimuth, solpos["zenith"], solpos["azimuth"]) < 90
    visible_azimuths = solpos.loc[daytime & illuminated, "azimuth"]
    all_anchors = set(range(0, 360, int(spacing_deg)))
    visible_anchors = {
        int(spacing_deg * (az // spacing_deg)) % 360 for az in visible_azimuths
    }
    return all_anchors - visible_anchors


def _azimuth_kernel_weight(azimuth: pd.Series, anchor_deg: float) -> pd.Series:
    """Gaussian kernel weight of each observation's azimuth relative to
    one azimuth anchor, using CIRCULAR distance (so 359 degrees and
    1 degree are treated as 2 degrees apart, not 358, unlike a plain
    subtraction). Weight decays smoothly towards 0 with distance instead
    of dropping to exactly 0 at a hard boundary - an observation some
    distance away still contributes, just with less weight than one
    sitting exactly on the anchor.

    :param azimuth: Solar azimuth (degrees, 0-360) for each observation.
    :type azimuth: pd.Series
    :param anchor_deg: The anchor's azimuth (degrees).
    :type anchor_deg: float
    :return: Weight in (0, 1], same index as azimuth.
    :rtype: pd.Series
    """
    az_diff = (azimuth - anchor_deg).abs() % 360
    az_dist = np.minimum(az_diff, 360 - az_diff)
    return np.exp(-0.5 * (az_dist / AZIMUTH_KERNEL_BANDWIDTH_DEG) ** 2)


def aggregate_horizon_profile(
    shaded: pd.Series,
    azimuth: pd.Series,
    elevation: pd.Series,
    actual: pd.Series,
    expected_clear_sky: pd.Series,
    previous_profile: dict | None,
    forgetting_factor: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit flagged shaded instants by azimuth and season into a set of
    azimuth anchors, and blend into a horizon profile.

    Each azimuth anchor (AZIMUTH_ANCHOR_SPACING_DEG apart) is fit from a
    SOFT, overlapping window rather than a discrete bin: an observation
    counts towards an anchor if it falls within AZIMUTH_KERNEL_CUTOFF
    bandwidths of that anchor (see _azimuth_kernel_weight), so
    observations near one anchor also contribute to neighbouring anchors
    instead of counting fully for exactly one and not at all for others.
    This makes the fitted values vary gradually between adjacent anchors -
    matching the real underlying physics, since a panel's own physical
    width means a shadow edge sweeps gradually across it even when the
    obstruction itself (a chimney, a roofline) is sharp, rather than
    jumping at some arbitrary boundary as a hard partition would. There is
    no discrete "which box is this observation in" step anywhere in this
    function or in interpolate_horizon_profile (the query-side
    counterpart) - anchors are just densely-spaced sample points on an
    otherwise continuous function of azimuth.

    For each (soft azimuth window, meteorological season) cell with
    enough effective weight this window (see MIN_OBSERVATIONS_PER_BIN -
    now a sum of kernel weights, not a raw count): if any covered
    instants were flagged shaded, this window's evidence is a LOWER
    bound on the horizon elevation (the highest elevation seen blocked -
    it's obstructed at least up to there), and the kernel-weighted mean
    actual/expected ratio among just those shaded instants is this
    window's evidence for the transmittance (how much light still gets
    through below that elevation - 0 for a hard obstruction, higher for
    a tree canopy). The elevation bound itself is a plain max/min over
    the soft-included instants, not weighted - weighting it directly
    (e.g. a weighted quantile) would let a handful of high-weight points
    pull the bound past an instant it actually observed to be shaded/
    clear, undermining the "at least blocked/clear up to here" guarantee
    that makes it useful. If none were flagged shaded, this window's
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
    :return: {"<anchor_deg>": {"<season>": {"elevation": .., "transmittance": ..}}}
    :rtype: dict[str, dict[str, dict[str, float]]]
    """
    previous_profile = previous_profile or {}
    season = season_labels_for_index(elevation.index)
    ratio = (actual / expected_clear_sky).clip(lower=0.0, upper=1.0)
    anchors = np.arange(0, 360, AZIMUTH_ANCHOR_SPACING_DEG)
    weight_cutoff = np.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
    profile: dict[str, dict[str, dict[str, float]]] = {}
    for anchor_deg in anchors:
        key = str(int(anchor_deg))
        weight = _azimuth_kernel_weight(azimuth, float(anchor_deg))
        in_window = weight > weight_cutoff
        prev_seasons = normalize_bin_entry(previous_profile.get(key))
        anchor_profile = dict(prev_seasons)
        for s in SEASON_LABELS:
            in_cell = in_window & (season == s)
            cell_weight = weight[in_cell]
            effective_n = float(cell_weight.sum())
            if effective_n < MIN_OBSERVATIONS_PER_BIN:
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
            shaded_weight = cell_weight[cell_shaded_mask]
            if float(shaded_weight.sum()) >= MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE:
                shaded_ratio = ratio[in_cell][cell_shaded_mask]
                window_transmittance = float(
                    (shaded_weight * shaded_ratio).sum() / shaded_weight.sum()
                )
                new_transmittance = (
                    forgetting_factor * prev_entry["transmittance"]
                    + (1 - forgetting_factor) * window_transmittance
                )
            anchor_profile[s] = {"elevation": new_elevation, "transmittance": new_transmittance}
        profile[key] = anchor_profile
    return profile


def interpolate_horizon_profile(
    profile: dict, azimuth: pd.Series, season: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Continuous query into a persisted horizon profile: for arbitrary
    azimuths (not confined to any grid), returns a kernel-weighted average
    of every anchor's (elevation, transmittance) - the query-side
    counterpart to aggregate_horizon_profile's fitting. Used by both
    Forecast._apply_pv_horizon_mask (the live day-ahead forecast mask) and
    render_horizon_polar_grid (the diagnostic chart), so both agree on
    exactly the same continuous function of azimuth.

    For each present anchor, weight = _azimuth_kernel_weight(azimuth,
    anchor_deg); result = sum(weight * anchor_value) / sum(weight). This is
    a plain weighted average of already-fitted anchor values - NOT the
    conservative max/min bound logic aggregate_horizon_profile uses when
    combining raw, noisy observations. Those are different jobs: fitting
    has to turn noisy raw instants into one trustworthy estimate per
    anchor (where a bound is the right conservative choice), while this
    function only smooths between two already-trustworthy fitted numbers
    (where a weighted average is the right choice - there's no noise left
    to be conservative about, just a gap to interpolate across).

    A virtual "cold start" anchor (elevation=0, transmittance=0) is
    included in every average with a small fixed weight
    (exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2), the same "beyond this many
    bandwidths, treat as not really contributing" threshold
    aggregate_horizon_profile itself uses) - without it, a single learned
    anchor would project its exact value across the ENTIRE circle
    (weighted-average math degenerates to "the only anchor's value,
    everywhere" when nothing else competes in the sum, however far away
    the query is), which is clearly wrong: a chimney learned at due south
    says nothing about due north. With this virtual anchor, the result
    fades smoothly toward the cold-start default as a query moves away
    from every real anchor, and only costs a ~1% pull toward cold-start
    for a query sitting exactly on a well-measured anchor - negligible.

    Works unchanged on a profile with any anchor spacing, including an old
    profile persisted before AZIMUTH_ANCHOR_SPACING_DEG was tightened
    (interpolation is just coarser then, not wrong) - there is no assumed
    grid, only whichever anchor keys are actually present.

    :param profile: The persisted profile - {"<anchor_deg>": {"<season>":
        {"elevation": .., "transmittance": ..}}}, in any format
        normalize_bin_entry accepts.
    :type profile: dict
    :param azimuth: Query solar azimuths (degrees, 0-360).
    :type azimuth: pd.Series
    :param season: Meteorological season label for each query row (see
        season_labels_for_index) - varies per row since a multi-day
        forecast can cross a season boundary.
    :type season: pd.Series
    :return: (elevation, transmittance) Series, same index as azimuth.
    :rtype: tuple[pd.Series, pd.Series]
    """
    anchor_degs = sorted(int(k) for k in profile)
    elevation = pd.Series(_COLD_START_ENTRY["elevation"], index=azimuth.index, dtype=float)
    transmittance = pd.Series(_COLD_START_ENTRY["transmittance"], index=azimuth.index, dtype=float)
    if not anchor_degs:
        return elevation, transmittance
    weights = pd.DataFrame(
        {d: _azimuth_kernel_weight(azimuth, float(d)) for d in anchor_degs}, index=azimuth.index
    )
    cold_start_weight = np.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
    for s in SEASON_LABELS:
        rows = season == s
        if not rows.any():
            continue
        entries = [normalize_bin_entry(profile.get(str(d))).get(s, _COLD_START_ENTRY) for d in anchor_degs]
        elevation_values = np.array([e["elevation"] for e in entries])
        transmittance_values = np.array([e["transmittance"] for e in entries])
        row_weights = weights.loc[rows, anchor_degs].to_numpy()
        weight_sums = row_weights.sum(axis=1) + cold_start_weight
        elevation.loc[rows] = (
            row_weights @ elevation_values + cold_start_weight * _COLD_START_ENTRY["elevation"]
        ) / weight_sums
        transmittance.loc[rows] = (
            row_weights @ transmittance_values + cold_start_weight * _COLD_START_ENTRY["transmittance"]
        ) / weight_sums
    return elevation, transmittance
