#!/usr/bin/env python3

"""
PV Shading/Horizon Kalman Detector
===================================

Learns a per-direction, per-season horizon profile (the elevation angle
below which the sun is physically obstructed by a genuine HARD object,
and the fraction of direct sun that still gets through below it) from
historical PV production, reusing the same scalar Kalman innovation-gate
math already used for sensorless window/door detection (see
emhass.thermal.opening_kalman_detector.kalman_predict_update). The
season split lets a deciduous tree's leaf-on/leaf-off difference be
learned instead of averaged away.

Two independent, additive layers on top of that hard-object horizon:
- Genuinely PARTIAL shading (a tree canopy letting a varying fraction of
  light through depending on exactly where in its canopy the sun sits) -
  a real 2D (azimuth x elevation) transmittance surface
  (aggregate_partial_transmittance_surface / interpolate_partial_transmittance),
  since a single scalar-per-azimuth number can't represent attenuation
  that genuinely varies with elevation too.
- Diffuse-light (sky-dome) attenuation (compute_diffuse_transmission_factor) -
  a real obstruction blocks part of the sky dome's diffuse contribution
  too, not just the direct beam, computed once per season as a closed-
  form isotropic-sky integral rather than per-instant.

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

# Spacing between fitted elevation anchors for the 2D partial-
# transmittance surface (aggregate_partial_transmittance_surface) -
# coarser than AZIMUTH_ANCHOR_SPACING_DEG because elevation is a second
# axis sharing the same finite dataset: a 5-degree elevation grid starves
# for data even with abundant synthetic data (confirmed empirically this
# session - a 5x5 degree grid cleared enough evidence in barely a quarter
# of physically-possible cells).
ELEVATION_ANCHOR_SPACING_DEG = 15

# Width of the Gaussian kernel used to softly weight each observation's
# contribution to an anchor (see _azimuth_kernel_weight) - roughly matches
# the span over which a solar panel's own physical width smears a sharp
# obstruction edge into a gradual actual/expected transition, so the
# smoothing reveals a real physical effect rather than manufacturing one.
AZIMUTH_KERNEL_BANDWIDTH_DEG = 10.0

# Width of the (non-circular - elevation doesn't wrap around) Gaussian
# kernel used to softly weight each observation's contribution to an
# elevation anchor in the 2D partial-transmittance surface - the
# elevation-axis equivalent of AZIMUTH_KERNEL_BANDWIDTH_DEG.
ELEVATION_KERNEL_BANDWIDTH_DEG = 10.0

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

# Fraction of direct light still allowed through before an instant counts
# as a genuine "hard object" (a chimney, a roofline, a solid obstruction)
# rather than merely partial shading (a tree canopy letting a varying
# fraction through) - a much stricter test than classify_shaded_instants's
# own gate, which flags any statistically significant dip. Feeds
# aggregate_horizon_profile's elevation/transmittance fields (the "solid
# obstruction" horizon); genuinely partial attenuation is handled
# separately by aggregate_partial_transmittance_surface.
HARD_OBJECT_RATIO_THRESHOLD = 0.05

# Fraction of expected clear-sky DNI assumed to get through below an
# anchor/season's learned horizon elevation, before any shaded instant has
# ever been observed there to measure it directly - matches this
# feature's original (pre-transmittance) hard-block behavior, so a
# freshly-learned obstruction still starts conservative.
DEFAULT_TRANSMITTANCE = 0.0

_COLD_START_ENTRY = {"elevation": 0.0, "transmittance": DEFAULT_TRANSMITTANCE}

# Minimum confirmed-clear (no known direct-beam shading at all) instants
# a season needs before estimate_empirical_diffuse_transmission_factor
# trusts its own regression - separating a direct-share coefficient from
# a diffuse-share coefficient needs enough rows for the fit to be stable,
# not just statistically present.
MIN_OBSERVATIONS_FOR_DIFFUSE_REGRESSION = 200

# The direct/diffuse POA share only varies naturally with solar elevation
# and real weather variation (haze, humidity) - too little of that
# variation within a season's confirmed-clear window means the two
# regression coefficients aren't really separable (near-collinear
# columns), so the fit is skipped rather than trusted below this bar.
MIN_DIRECT_SHARE_STD_FOR_DIFFUSE_REGRESSION = 0.03


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


def classify_hard_object_instants(actual: pd.Series, expected_clear_sky: pd.Series) -> pd.Series:
    """Per-timestep boolean: True where actual output implies a genuine
    "hard object" (a solid obstruction - a chimney, a roofline) rather
    than merely partial shading - at least HARD_OBJECT_RATIO_THRESHOLD-
    strict a deficit (<=5% of expected clear-sky output still getting
    through, by default).

    A direct ratio threshold, not a statistical gate like
    classify_shaded_instants: whether >=95% of direct light is blocked
    isn't a subtle judgement call the way a 10-20% dip is, so no Kalman
    gate is needed here - just the same MIN_EXPECTED_POWER_W noise-floor
    exclusion classify_shaded_instants itself applies near sunrise/sunset.

    Feeds aggregate_horizon_profile's elevation/transmittance fields (the
    "solid obstruction" horizon). Every hard-blocked instant is also
    "shaded" under classify_shaded_instants's own broader gate, so
    aggregate_partial_transmittance_surface's genuinely-partial evidence
    is defined as shaded-but-not-hard-blocked, never double-counting a
    hard-object instant as partial evidence too.

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
    return valid & (ratio <= HARD_OBJECT_RATIO_THRESHOLD)


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
    hard_blocked: pd.Series,
    azimuth: pd.Series,
    elevation: pd.Series,
    actual: pd.Series,
    expected_clear_sky: pd.Series,
    previous_profile: dict | None,
    forgetting_factor: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit flagged hard-object instants by azimuth and season into a set
    of azimuth anchors, and blend into a horizon profile.

    hard_blocked (classify_hard_object_instants's output - a strict
    >=95%-blocked criterion, HARD_OBJECT_RATIO_THRESHOLD) drives this
    function, NOT classify_shaded_instants's broader "any statistically
    significant dip" gate - this profile is specifically the SOLID-
    OBSTRUCTION ("vaste objecten") horizon: a real geometric edge, not
    wherever partial shading merely starts. Genuinely partial attenuation
    (a tree canopy letting a varying fraction of light through) is fit
    separately by aggregate_partial_transmittance_surface and applied as
    an additional layer on top of this profile - see
    Forecast._apply_pv_horizon_mask.

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
    instants were flagged hard-blocked, this window's evidence is a LOWER
    bound on the horizon elevation (the highest elevation seen blocked -
    it's obstructed at least up to there), and the kernel-weighted mean
    actual/expected ratio among just those hard-blocked instants is this
    window's evidence for the transmittance (how much light still gets
    through below that elevation - close to 0 for a genuine hard
    obstruction, by construction of the >=95%-blocked criterion that
    selected these instants in the first place). The elevation bound
    itself is a plain max/min over the soft-included instants, not
    weighted - weighting it directly (e.g. a weighted quantile) would let
    a handful of high-weight points pull the bound past an instant it
    actually observed to be hard-blocked/clear, undermining the "at least
    blocked/clear up to here" guarantee that makes it useful. If none
    were flagged hard-blocked, elevation is left exactly as it was (0.0/
    true horizon for a direction never once seen hard-blocked, or
    whatever was last confirmed for one that has) - NOT pulled up to the
    lowest elevation the sun happened to reach this window, since
    transmittance stays at its own cold-start default (fully blocked)
    below whatever elevation is set regardless of why, which would
    otherwise render an azimuth nobody had ever seen the sun sit low
    enough to test as a confident "definitely blocked" zone instead of
    "not yet tested" - and there is no transmittance evidence at all this
    round either way (nothing below the horizon was observed to
    measure). Whichever field has new evidence is blended with
    previous_profile:
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

    :param hard_blocked: Boolean Series from classify_hard_object_instants
        (NOT classify_shaded_instants - see the function docstring above).
    :type hard_blocked: pd.Series
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
            cell_hard_blocked_mask = hard_blocked[in_cell]
            cell_hard_blocked_elevations = cell_elevations[cell_hard_blocked_mask]
            if not cell_hard_blocked_elevations.empty:
                window_elevation = float(cell_hard_blocked_elevations.max())
                new_elevation = (
                    forgetting_factor * prev_entry["elevation"]
                    + (1 - forgetting_factor) * window_elevation
                )
            else:
                # No hard-blocked evidence this window - leave elevation
                # exactly as it was (0.0/true horizon for a direction
                # never once seen hard-blocked, or whatever was last
                # confirmed for one that has). An earlier version of this
                # function moved elevation up to the sun's own lowest
                # elevation observed this window as a conservative "at
                # least clear down to here" bound - but transmittance
                # stays at its own cold-start default (fully blocked,
                # DEFAULT_TRANSMITTANCE) below THAT elevation regardless,
                # so an azimuth nobody had ever seen the sun sit low
                # enough to test rendered as a confident "definitely
                # blocked" red zone instead of "not yet tested". Once a
                # direction genuinely has confirmed hard-blocked evidence,
                # a later window without any doesn't erode that boundary
                # either - same "insufficient new evidence, keep the
                # existing value" principle as everywhere else in this
                # function.
                new_elevation = prev_entry["elevation"]
            new_transmittance = prev_entry["transmittance"]
            hard_blocked_weight = cell_weight[cell_hard_blocked_mask]
            if float(hard_blocked_weight.sum()) >= MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE:
                hard_blocked_ratio = ratio[in_cell][cell_hard_blocked_mask]
                window_transmittance = float(
                    (hard_blocked_weight * hard_blocked_ratio).sum() / hard_blocked_weight.sum()
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


def compute_diffuse_transmission_factor(profile: dict, season: str) -> float:
    """Closed-form isotropic-sky-dome diffuse-light attenuation factor for
    one season, derived from the learned per-azimuth hard-object horizon
    elevation h(az) and transmittance t(az) - the fraction of an
    unobstructed sky dome's diffuse contribution still reaching the panel,
    averaged over every azimuth.

    Derivation: for a horizontal reference and an isotropic sky, the
    diffuse view-factor integral integral(cos(theta)*sin(theta), theta,
    0, 90) = 1/2 over the whole hemisphere. Splitting that integral at a
    horizon elevation h gives sin(h)^2/2 for the blocked band [0, h] and
    cos(h)^2/2 for the clear band [h, 90] - so, letting transmittance t
    reduce (not zero) the blocked band's own contribution, one azimuth's
    remaining fraction is t*sin(h)^2 + cos(h)^2 (the two halves' /2
    factors cancel against the /2 normalization). Averaging that over all
    azimuths gives the overall factor.

    Used by Forecast._apply_pv_horizon_mask to attenuate DHI - unlike DNI
    masking, this is NOT conditional on the sun's current position: the
    sky dome (and however much of it is obstructed) is there all the
    time, so the same factor applies to every timestep of a given season,
    not just ones below the sun's own instantaneous horizon.

    :param profile: The persisted horizon profile (see
        aggregate_horizon_profile / interpolate_horizon_profile).
    :type profile: dict
    :param season: Which season to compute the factor for.
    :type season: str
    :return: Diffuse-transmission factor in [0, 1] (1.0 = fully
        unobstructed sky dome).
    :rtype: float
    """
    query_azimuth = pd.Series(np.arange(0, 360, AZIMUTH_RENDER_SPACING_DEG), dtype=float)
    query_season = pd.Series([season] * len(query_azimuth))
    elevation, transmittance = interpolate_horizon_profile(profile, query_azimuth, query_season)
    h_rad = np.radians(elevation.clip(lower=0.0, upper=90.0))
    per_azimuth_factor = transmittance * np.sin(h_rad) ** 2 + np.cos(h_rad) ** 2
    return float(per_azimuth_factor.mean())


def compute_sun_path_envelope(
    latitude: float, longitude: float, fine_step_deg: float = AZIMUTH_RENDER_SPACING_DEG
) -> tuple[dict[float, float | None], dict[float, float | None]]:
    """The sun's own real yearly elevation envelope at each azimuth -
    earliest (lowest) and latest (highest) elevation the sun is ever
    observed at, swept across one fixed reference year at fine_step_deg
    azimuth resolution.

    Used to gate interpolate_partial_transmittance against azimuth/
    elevation combinations the sun has never physically occupied - a
    kernel's finite bandwidth would otherwise "bleed" a value into those
    combinations from real observations just inside the envelope, which
    is physically meaningless (there is no such thing as a measurement
    where the sun was never present) - a real bug caught and fixed in
    this session's own visual prototyping (single_panel_2d_demo.py)
    before this function existed in production.

    Sweeps a full year at 2-minute solar-position resolution (~260k
    points - pure trig via pvlib, cheap) and buckets by azimuth; each
    curve is then lightly smoothed (a small centered rolling average,
    wrapping around 360 degrees) to remove sampling kinks from different
    days handing off the extremum from one azimuth bucket to the next -
    purely cosmetic, the curve stays an honest envelope of the same
    underlying data.

    :param latitude: Site latitude, degrees.
    :type latitude: float
    :param longitude: Site longitude, degrees.
    :type longitude: float
    :param fine_step_deg: Azimuth resolution of the swept curve, degrees.
    :type fine_step_deg: float
    :return: (sun_min_curve, sun_max_curve) - each {azimuth_deg: elevation,
        or None if the sun's path never crosses that azimuth at all}.
    :rtype: tuple[dict[float, float | None], dict[float, float | None]]
    """
    fine_az = np.arange(0, 360, fine_step_deg)
    times = pd.date_range("2023-01-01", "2023-12-31 23:58", freq="2min", tz="UTC")
    solpos = get_solarposition(times, latitude, longitude)
    daytime = solpos[solpos["elevation"] > 0]
    sun_min_curve: dict[float, float | None] = {}
    sun_max_curve: dict[float, float | None] = {}
    for az in fine_az:
        in_bin = daytime[(daytime["azimuth"] >= az) & (daytime["azimuth"] < az + fine_step_deg)]
        if in_bin.empty:
            sun_min_curve[az] = None
            sun_max_curve[az] = None
        else:
            sun_min_curve[az] = float(in_bin["elevation"].min())
            sun_max_curve[az] = float(in_bin["elevation"].max())

    def _smooth(curve: dict[float, float | None], window: int = 5) -> dict[float, float | None]:
        keys = sorted(curve.keys())
        values = [curve[k] for k in keys]
        n = len(values)
        half = window // 2
        smoothed: dict[float, float | None] = {}
        for i, k in enumerate(keys):
            window_vals = [values[(i + o) % n] for o in range(-half, half + 1)]
            present = [v for v in window_vals if v is not None]
            smoothed[k] = float(np.mean(present)) if present else None
        return smoothed

    return _smooth(sun_min_curve), _smooth(sun_max_curve)


def compute_self_shading_curve(
    surface_tilt: float, surface_azimuth: float, fine_step_deg: float = AZIMUTH_RENDER_SPACING_DEG
) -> dict[float, float | None]:
    """The panel's own precise self-shading boundary - at each fine
    azimuth, the elevation below which the sun is behind the panel's own
    tilted plane (angle-of-incidence >= 90 degrees), computed directly
    from geometry rather than approximated from a coarser measured
    profile.

    Complements compute_sun_path_envelope (the sun's own real reach) -
    together they give the two purely-geometric boundaries a rendered
    chart needs, independent of any learned/measured data.
    compute_geometrically_blind_azimuths already answers the coarser,
    binary "is this whole anchor self-shaded" question for the live mask;
    this is its fine-resolution, continuous-boundary counterpart for
    charting (see this session's own single_panel_combined_demo.py
    prototype, which this is a direct port of).

    Sweeps elevation 0-90 degrees (0.5-degree steps - pure trig via
    pvlib.irradiance.aoi, cheap) at each fine azimuth and finds the
    lowest elevation where the front face becomes lit.

    :param surface_tilt: Panel tilt from horizontal, degrees.
    :type surface_tilt: float
    :param surface_azimuth: Panel azimuth, degrees (0-360).
    :type surface_azimuth: float
    :param fine_step_deg: Azimuth resolution of the swept curve, degrees.
    :type fine_step_deg: float
    :return: {azimuth_deg: elevation of the self-shading boundary, or
        None if never self-shaded (illuminated at every elevation) at
        that azimuth, or 0.0 if self-shaded at every elevation}.
    :rtype: dict[float, float | None]
    """
    fine_az = np.arange(0, 360, fine_step_deg)
    elevation_sweep = np.arange(0, 90.01, 0.5)
    zenith_sweep = 90 - elevation_sweep
    curve: dict[float, float | None] = {}
    for az in fine_az:
        illuminated = aoi(surface_tilt, surface_azimuth, zenith_sweep, az) < 90
        if illuminated.all():
            curve[az] = None
        elif not illuminated.any():
            curve[az] = 0.0
        else:
            curve[az] = float(elevation_sweep[np.argmax(illuminated)])
    return curve


def aggregate_partial_transmittance_surface(
    shaded: pd.Series,
    hard_blocked: pd.Series,
    azimuth: pd.Series,
    elevation: pd.Series,
    actual: pd.Series,
    expected_clear_sky: pd.Series,
    previous_surface: dict | None,
    forgetting_factor: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Fit a genuine 2D (azimuth x elevation) partial-transmittance
    surface from instants that are shaded (classify_shaded_instants) but
    NOT a hard object (classify_hard_object_instants) - real, measured
    PARTIAL attenuation (a tree canopy letting a varying fraction of
    light through depending on exactly where in its canopy the sun sits),
    which aggregate_horizon_profile's single scalar-per-azimuth
    transmittance can't represent (it only stores one number regardless
    of elevation).

    Each (azimuth anchor, elevation anchor, season) cell is fit from a
    SOFT, overlapping 2D window - the same circular-azimuth kernel
    aggregate_horizon_profile itself uses (_azimuth_kernel_weight),
    multiplied by a plain (non-circular) Gaussian kernel over elevation -
    so there is no discrete "which box is this observation in" step on
    either axis, the same principle as the 1D model just extended to a
    second axis. Elevation anchors are spaced ELEVATION_ANCHOR_SPACING_DEG
    apart (coarser than azimuth's AZIMUTH_ANCHOR_SPACING_DEG, since a 2D
    grid divides the same finite dataset across more cells).

    Unlike aggregate_horizon_profile's elevation field (a conservative
    max/min bound, appropriate for pinning down a hard edge), this
    surface is a plain kernel-weighted MEAN ratio - there's no edge to
    bound here, just a smoothly-varying partial attenuation to average
    directly, the same shape aggregate_horizon_profile's own transmittance
    field already uses for its own (single-elevation-value) average.

    A cell absent from previous_surface defaults to transmittance=1.0
    (no additional attenuation), not aggregate_horizon_profile's
    cold-start of 0 - this surface only ever REDUCES light on top of
    whatever the hard-object horizon already decided, so "no evidence
    here yet" has to mean "no extra effect", never "fully blocked".

    :param shaded: Boolean Series from classify_shaded_instants (broad
        gate - any statistically significant deficit).
    :type shaded: pd.Series
    :param hard_blocked: Boolean Series from classify_hard_object_instants
        (strict gate - genuinely near-total blocks only) - subtracted out
        so this surface only fits from genuinely PARTIAL evidence, never
        double-counting a hard-object instant.
    :type hard_blocked: pd.Series
    :param azimuth: Solar azimuth (degrees, 0-360) for the same timestamps.
    :type azimuth: pd.Series
    :param elevation: Solar elevation (degrees) for the same timestamps.
    :type elevation: pd.Series
    :param actual: Measured PV power (W) for the same timestamps.
    :type actual: pd.Series
    :param expected_clear_sky: Unobstructed clear-sky PVLib simulation
        output (W) for the same timestamps.
    :type expected_clear_sky: pd.Series
    :param previous_surface: The persisted surface from the last refit -
        {"<azimuth_anchor>": {"<season>": {"<elevation_anchor>":
        transmittance}}} - or None on a first-ever refit.
    :type previous_surface: dict | None
    :param forgetting_factor: Weight on the previous surface, in [0, 1] -
        same value aggregate_horizon_profile is called with.
    :type forgetting_factor: float
    :return: {"<azimuth_anchor>": {"<season>": {"<elevation_anchor>": transmittance}}}
    :rtype: dict[str, dict[str, dict[str, float]]]
    """
    previous_surface = previous_surface or {}
    season = season_labels_for_index(elevation.index)
    ratio = (actual / expected_clear_sky).clip(lower=0.0, upper=1.0)
    partial = shaded & ~hard_blocked
    azimuth_anchors = np.arange(0, 360, AZIMUTH_ANCHOR_SPACING_DEG)
    elevation_anchors = np.arange(0, 90, ELEVATION_ANCHOR_SPACING_DEG)
    azimuth_weight_cutoff = np.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
    surface: dict[str, dict[str, dict[str, float]]] = {}
    for az_anchor in azimuth_anchors:
        az_key = str(int(az_anchor))
        az_weight = _azimuth_kernel_weight(azimuth, float(az_anchor))
        az_in_window = az_weight > azimuth_weight_cutoff
        prev_az_seasons = previous_surface.get(az_key) or {}
        az_surface = {s: dict(v) for s, v in prev_az_seasons.items()}
        for s in SEASON_LABELS:
            prev_elevation_map = prev_az_seasons.get(s) or {}
            season_map = dict(prev_elevation_map)
            season_rows = az_in_window & (season == s) & partial
            for el_anchor in elevation_anchors:
                el_key = str(int(el_anchor))
                el_weight = np.exp(
                    -0.5 * ((elevation - float(el_anchor)) / ELEVATION_KERNEL_BANDWIDTH_DEG) ** 2
                )
                cell_weight = (az_weight * el_weight)[season_rows]
                effective_n = float(cell_weight.sum())
                if effective_n < MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE:
                    continue
                cell_ratio = ratio[season_rows]
                window_transmittance = float((cell_weight * cell_ratio).sum() / cell_weight.sum())
                prev_value = float(prev_elevation_map.get(el_key, 1.0))
                season_map[el_key] = (
                    forgetting_factor * prev_value + (1 - forgetting_factor) * window_transmittance
                )
            if season_map:
                az_surface[s] = season_map
        if az_surface:
            surface[az_key] = az_surface
    return surface


def interpolate_partial_transmittance(
    surface: dict,
    azimuth: pd.Series,
    elevation: pd.Series,
    season: pd.Series,
    sun_min_curve: dict[float, float | None] | None = None,
    sun_max_curve: dict[float, float | None] | None = None,
) -> pd.Series:
    """Continuous 2D query into a persisted partial-transmittance surface:
    for arbitrary (azimuth, elevation) pairs, returns a kernel-weighted
    average of every anchor's transmittance - the query-side counterpart
    to aggregate_partial_transmittance_surface's fitting, mirroring
    interpolate_horizon_profile's own design one axis further.

    Defaults to 1.0 (no additional attenuation) far from any real
    evidence - unlike interpolate_horizon_profile's cold-start of 0, this
    surface only ever REDUCES light on top of the existing hard-object
    horizon, so "nothing measured here" has to mean "no extra effect",
    not "fully blocked". The same virtual-cold-start-anchor trick
    interpolate_horizon_profile uses (a fixed weight competing in every
    average, see AZIMUTH_KERNEL_CUTOFF) keeps a lone real anchor from
    projecting its value across the whole (azimuth, elevation) plane.

    When sun_min_curve/sun_max_curve (see compute_sun_path_envelope) are
    given, a query point outside the sun's own real yearly elevation
    range AT THAT AZIMUTH returns 1.0 unconditionally, bypassing the
    kernel entirely - the elevation kernel's finite bandwidth would
    otherwise bleed a value in from real observations just inside the
    envelope into elevations the sun has never physically occupied at
    that azimuth, which is physically meaningless. Omit them only for
    quick/unit-test convenience; production callers should always supply
    the real envelope.

    :param surface: The persisted surface - {"<azimuth_anchor>":
        {"<season>": {"<elevation_anchor>": transmittance}}}.
    :type surface: dict
    :param azimuth: Query solar azimuths (degrees, 0-360).
    :type azimuth: pd.Series
    :param elevation: Query solar elevations (degrees).
    :type elevation: pd.Series
    :param season: Meteorological season label for each query row.
    :type season: pd.Series
    :param sun_min_curve: {azimuth_deg: lowest elevation the sun is ever
        observed at, or None} from compute_sun_path_envelope, or None to
        skip the physical gate.
    :type sun_min_curve: dict[float, float | None] | None
    :param sun_max_curve: Same shape, highest elevation.
    :type sun_max_curve: dict[float, float | None] | None
    :return: Transmittance Series in (0, 1], same index as azimuth.
    :rtype: pd.Series
    """
    transmittance = pd.Series(1.0, index=azimuth.index, dtype=float)
    cold_start_weight = np.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
    for s in SEASON_LABELS:
        rows = season == s
        if not rows.any():
            continue
        row_azimuth = azimuth[rows]
        row_elevation = elevation[rows]
        weight_sum = pd.Series(cold_start_weight, index=row_azimuth.index)
        weighted_value_sum = pd.Series(cold_start_weight * 1.0, index=row_azimuth.index)
        for az_key, az_seasons in surface.items():
            elevation_map = az_seasons.get(s)
            if not elevation_map:
                continue
            az_weight = _azimuth_kernel_weight(row_azimuth, float(az_key))
            for el_key, value in elevation_map.items():
                el_weight = np.exp(
                    -0.5 * ((row_elevation - float(el_key)) / ELEVATION_KERNEL_BANDWIDTH_DEG) ** 2
                )
                w = az_weight * el_weight
                weight_sum = weight_sum + w
                weighted_value_sum = weighted_value_sum + w * float(value)
        transmittance.loc[rows] = weighted_value_sum / weight_sum

    if sun_min_curve and sun_max_curve:
        fine_azs = np.array(sorted(sun_min_curve.keys()))
        nearest = fine_azs[np.abs(azimuth.to_numpy()[:, None] - fine_azs[None, :]).argmin(axis=1)]
        lo = np.array([sun_min_curve[a] for a in nearest], dtype=float)
        hi = np.array([sun_max_curve[a] for a in nearest], dtype=float)
        el = elevation.to_numpy()
        physically_possible = ~np.isnan(lo) & (el >= lo) & (el <= hi)
        transmittance = transmittance.where(pd.Series(physically_possible, index=azimuth.index), 1.0)
    return transmittance


def estimate_empirical_diffuse_transmission_factor(
    actual: pd.Series,
    expected_clear_sky: pd.Series,
    direct_share: pd.Series,
    diffuse_share: pd.Series,
    confirmed_clear: pd.Series,
    previous_factors: dict[str, float] | None,
    forgetting_factor: float,
) -> dict[str, float]:
    """Empirically measured diffuse-light (sky-dome) attenuation, per
    season - a real-data alternative/upgrade to
    compute_diffuse_transmission_factor's purely theoretical integral
    over the learned direct-beam horizon, which can't reflect an
    obstruction that affects the sky dome differently than the direct-
    beam model implies (including in directions the direct beam never
    tests at all).

    Exploits instants that are CONFIRMED clear of any known direct-beam
    shading (confirmed_clear - typically ~classify_shaded_instants, zero
    evidence of shading at all, not just below the hard-object threshold)
    where the DNI/DHI split still naturally varies (low sun vs. high sun,
    hazier vs. crystal-clear days) - a plain no-intercept linear
    regression separates how much of the DIRECT-attributable share and
    the DIFFUSE-attributable share of the modeled clear-sky power is
    actually getting through, using only real observed variation - no
    artificial DNI=0 scenario needed:

        actual / expected_clear_sky ~= beta_direct * direct_share + beta_diffuse * diffuse_share

    beta_direct is a free sanity check (should land near 1.0, since these
    instants are already confirmed clear of known direct shading, so
    nothing should be attenuating the direct share specifically); only
    beta_diffuse (clipped to [0, 1]) is used, becoming this window's
    empirical diffuse-transmission estimate.

    Same shape as aggregate_horizon_profile: takes the previous full
    per-season dict, returns the updated full dict, a season untouched
    this window (too few confirmed-clear rows, or too little natural
    direct/diffuse variation to separate the two coefficients reliably)
    carries its previous value forward unchanged rather than being
    blended from an unstable fit.

    :param actual: Measured PV power (W), indexed by timestamp.
    :type actual: pd.Series
    :param expected_clear_sky: Unobstructed clear-sky PVLib simulation
        output (W) for the same timestamps.
    :type expected_clear_sky: pd.Series
    :param direct_share: Fraction of modeled POA irradiance attributable
        to the direct beam, per timestamp (poa_direct / poa_global).
    :type direct_share: pd.Series
    :param diffuse_share: Fraction of modeled POA irradiance attributable
        to sky + ground-reflected diffuse light, per timestamp
        ((poa_sky_diffuse + poa_ground_diffuse) / poa_global) -
        direct_share + diffuse_share ~= 1.
    :type diffuse_share: pd.Series
    :param confirmed_clear: Boolean Series - True where there is zero
        evidence of any direct-beam shading at all this instant.
    :type confirmed_clear: pd.Series
    :param previous_factors: {season: factor} from the last refit, or
        None on a first-ever refit.
    :type previous_factors: dict[str, float] | None
    :param forgetting_factor: Weight on the previous factor, in [0, 1] -
        same value the rest of this feature is called with.
    :type forgetting_factor: float
    :return: {season: empirical diffuse-transmission factor in [0, 1]} -
        only seasons that have ever cleared the evidence bar, possibly
        sparse.
    :rtype: dict[str, float]
    """
    previous_factors = dict(previous_factors or {})
    factors = dict(previous_factors)
    season = season_labels_for_index(actual.index)
    valid = confirmed_clear & (expected_clear_sky >= MIN_EXPECTED_POWER_W)
    for s in SEASON_LABELS:
        rows = valid & (season == s)
        n = int(rows.sum())
        if n < MIN_OBSERVATIONS_FOR_DIFFUSE_REGRESSION:
            continue
        if float(direct_share[rows].std()) < MIN_DIRECT_SHARE_STD_FOR_DIFFUSE_REGRESSION:
            continue
        ratio = (actual[rows] / expected_clear_sky[rows]).clip(lower=0.0, upper=1.5)
        design_matrix = np.column_stack([direct_share[rows].to_numpy(), diffuse_share[rows].to_numpy()])
        beta, *_ = np.linalg.lstsq(design_matrix, ratio.to_numpy(), rcond=None)
        window_factor = float(np.clip(beta[1], 0.0, 1.0))
        prev = previous_factors.get(s)
        factors[s] = (
            window_factor
            if prev is None
            else forgetting_factor * prev + (1 - forgetting_factor) * window_factor
        )
    return factors
