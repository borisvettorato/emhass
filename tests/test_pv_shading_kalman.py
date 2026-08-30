#!/usr/bin/env python
"""Unit tests for pv_shading_kalman.py - pure math, no I/O, no live network."""

import math
import unittest

import numpy as np
import pandas as pd

from emhass.pv_shading_kalman import (
    AZIMUTH_ANCHOR_SPACING_DEG,
    AZIMUTH_KERNEL_BANDWIDTH_DEG,
    AZIMUTH_KERNEL_CUTOFF,
    ELEVATION_KERNEL_BANDWIDTH_DEG,
    HARD_OBJECT_RATIO_THRESHOLD,
    MIN_EXPECTED_POWER_W,
    MIN_OBSERVATIONS_PER_BIN,
    MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE,
    _azimuth_kernel_weight,
    aggregate_horizon_profile,
    aggregate_partial_transmittance_surface,
    classify_hard_object_instants,
    classify_shaded_instants,
    compute_diffuse_transmission_factor,
    compute_geometrically_blind_azimuths,
    compute_sun_path_envelope,
    interpolate_horizon_profile,
    interpolate_partial_transmittance,
    normalize_bin_entry,
)


class TestClassifyShadedInstants(unittest.TestCase):
    def test_matching_output_is_never_flagged(self):
        """Actual output tracking the clear-sky expectation closely (small,
        realistic noise) is never flagged as shaded."""
        idx = pd.date_range("2026-06-01 08:00", periods=10, freq="30min", tz="UTC")
        expected = pd.Series([1000.0] * 10, index=idx)
        actual = pd.Series([980.0, 1010.0, 995.0, 1005.0, 990.0] * 2, index=idx)

        shaded = classify_shaded_instants(actual, expected)

        self.assertFalse(shaded.any())

    def test_large_sustained_deficit_is_flagged(self):
        """Actual output far below the clear-sky expectation (a real
        obstruction) is flagged."""
        idx = pd.date_range("2026-06-01 08:00", periods=5, freq="30min", tz="UTC")
        expected = pd.Series([1000.0] * 5, index=idx)
        actual = pd.Series([100.0] * 5, index=idx)  # 90% deficit

        shaded = classify_shaded_instants(actual, expected)

        self.assertTrue(shaded.all())

    def test_surplus_is_never_flagged(self):
        """Actual output ABOVE the clear-sky expectation is weather-model
        error, not shading evidence - never flagged, even if it would trip
        the underlying two-sided gate."""
        idx = pd.date_range("2026-06-01 08:00", periods=3, freq="30min", tz="UTC")
        expected = pd.Series([500.0] * 3, index=idx)
        actual = pd.Series([2000.0] * 3, index=idx)  # far above expected

        shaded = classify_shaded_instants(actual, expected)

        self.assertFalse(shaded.any())

    def test_low_expected_power_is_excluded(self):
        """Timestamps with expected clear-sky power below the noise floor
        (sunrise/sunset/night) are never flagged, regardless of ratio."""
        idx = pd.date_range("2026-06-01 06:00", periods=2, freq="30min", tz="UTC")
        expected = pd.Series([MIN_EXPECTED_POWER_W - 1, MIN_EXPECTED_POWER_W - 1], index=idx)
        actual = pd.Series([0.0, 0.0], index=idx)  # would be a 100% deficit if counted

        shaded = classify_shaded_instants(actual, expected)

        self.assertFalse(shaded.any())

    def test_zero_expected_power_does_not_crash(self):
        """A zero expected value (night) must not raise a division error."""
        idx = pd.date_range("2026-06-01 00:00", periods=2, freq="30min", tz="UTC")
        expected = pd.Series([0.0, 0.0], index=idx)
        actual = pd.Series([0.0, 0.0], index=idx)

        shaded = classify_shaded_instants(actual, expected)

        self.assertFalse(shaded.any())


class TestClassifyHardObjectInstants(unittest.TestCase):
    def test_near_total_block_is_flagged(self):
        """A deficit at or beyond HARD_OBJECT_RATIO_THRESHOLD (>=95%
        blocked, by default) is a genuine hard object."""
        idx = pd.date_range("2026-06-01 08:00", periods=3, freq="30min", tz="UTC")
        expected = pd.Series([1000.0] * 3, index=idx)
        actual = pd.Series([1000.0 * HARD_OBJECT_RATIO_THRESHOLD] * 3, index=idx)  # exactly at the line

        hard_blocked = classify_hard_object_instants(actual, expected)

        self.assertTrue(hard_blocked.all())

    def test_moderate_partial_deficit_is_not_flagged(self):
        """A real, statistically significant deficit that classify_shaded_
        instants WOULD flag, but well short of a genuine hard object
        (a tree canopy, not a chimney) - must NOT be flagged here, that's
        exactly the distinction this function exists to draw."""
        idx = pd.date_range("2026-06-01 08:00", periods=3, freq="30min", tz="UTC")
        expected = pd.Series([1000.0] * 3, index=idx)
        actual = pd.Series([400.0] * 3, index=idx)  # 60% deficit - real, but not >=95%

        self.assertTrue(classify_shaded_instants(actual, expected).all())
        self.assertFalse(classify_hard_object_instants(actual, expected).any())

    def test_low_expected_power_is_excluded(self):
        """Same MIN_EXPECTED_POWER_W noise-floor exclusion as
        classify_shaded_instants - near sunrise/sunset/night, the ratio
        isn't trustworthy regardless of how extreme it looks."""
        idx = pd.date_range("2026-06-01 06:00", periods=2, freq="30min", tz="UTC")
        expected = pd.Series([MIN_EXPECTED_POWER_W - 1] * 2, index=idx)
        actual = pd.Series([0.0, 0.0], index=idx)

        self.assertFalse(classify_hard_object_instants(actual, expected).any())

    def test_zero_expected_power_does_not_crash(self):
        idx = pd.date_range("2026-06-01 00:00", periods=2, freq="30min", tz="UTC")
        expected = pd.Series([0.0, 0.0], index=idx)
        actual = pd.Series([0.0, 0.0], index=idx)

        self.assertFalse(classify_hard_object_instants(actual, expected).any())


class TestNormalizeBinEntry(unittest.TestCase):
    def test_none_normalizes_to_empty_dict(self):
        self.assertEqual(normalize_bin_entry(None), {})

    def test_bare_float_broadcasts_hard_block_to_every_season(self):
        normalized = normalize_bin_entry(12.5)

        self.assertEqual(set(normalized.keys()), {"winter", "spring", "summer", "autumn"})
        for entry in normalized.values():
            self.assertEqual(entry, {"elevation": 12.5, "transmittance": 0.0})

    def test_flat_dict_broadcasts_to_every_season(self):
        normalized = normalize_bin_entry({"elevation": 8.0, "transmittance": 0.4})

        for entry in normalized.values():
            self.assertEqual(entry, {"elevation": 8.0, "transmittance": 0.4})

    def test_season_nested_dict_is_returned_as_is(self):
        entry = {"winter": {"elevation": 8.0, "transmittance": 0.4}}

        self.assertEqual(normalize_bin_entry(entry), entry)

    def test_empty_dict_normalizes_to_empty_dict_not_broadcast(self):
        """An anchor that exists in the profile but has never had any
        season clear MIN_OBSERVATIONS_PER_BIN yet (aggregate_horizon_profile
        can produce exactly this for a fresh anchor with no
        previous_profile) must normalize the same as None/missing - NOT
        get {} broadcast to every season, which would make
        normalized.get(some_season, cold_start) return {} instead of
        falling back to cold_start (the key would exist, just empty), and
        crash any caller indexing e["elevation"] on it."""
        self.assertEqual(normalize_bin_entry({}), {})


class TestAzimuthKernelWeight(unittest.TestCase):
    def test_weight_is_one_at_zero_distance(self):
        weight = _azimuth_kernel_weight(pd.Series([97.5]), 97.5)

        self.assertAlmostEqual(weight.iloc[0], 1.0, places=10)

    def test_weight_decays_smoothly_with_distance(self):
        weight = _azimuth_kernel_weight(pd.Series([107.5]), 97.5)  # 10 degrees away

        expected = np.exp(-0.5 * (10.0 / AZIMUTH_KERNEL_BANDWIDTH_DEG) ** 2)
        self.assertAlmostEqual(weight.iloc[0], expected, places=10)

    def test_weight_is_negligible_beyond_the_cutoff(self):
        far_distance = AZIMUTH_KERNEL_BANDWIDTH_DEG * (AZIMUTH_KERNEL_CUTOFF + 1)
        weight = _azimuth_kernel_weight(pd.Series([97.5 + far_distance]), 97.5)

        self.assertLess(weight.iloc[0], np.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2))

    def test_distance_wraps_around_0_360(self):
        """An observation at 359 degrees is genuinely close (2 degrees) to
        a bin centered at 1 degree - a plain (non-circular) subtraction
        would wrongly compute 358 degrees apart and give it ~zero weight."""
        weight_wrapped = _azimuth_kernel_weight(pd.Series([359.0]), 1.0)
        weight_equivalent = _azimuth_kernel_weight(pd.Series([3.0]), 1.0)  # also 2 degrees away

        self.assertAlmostEqual(weight_wrapped.iloc[0], weight_equivalent.iloc[0], places=10)
        self.assertGreater(weight_wrapped.iloc[0], 0.9)


class TestAggregateHorizonProfile(unittest.TestCase):
    """All fixtures below use June 2026 dates, which fall in the
    'summer' meteorological season - profile[bin]["summer"] is the
    entry under test unless noted otherwise."""

    def _make_series(self, n, azimuth_value, elevations, shaded_mask, actual=None, expected=None):
        idx = pd.date_range("2026-06-01 08:00", periods=n, freq="15min", tz="UTC")
        # azimuth_value is the target anchor itself (anchors are sample
        # points, not bin boundaries) - placing observations exactly there
        # gives kernel weight 1.0, landing fully on that one anchor.
        azimuth = pd.Series([float(azimuth_value)] * n, index=idx)
        elevation = pd.Series(elevations, index=idx)
        shaded = pd.Series(shaded_mask, index=idx)
        actual = pd.Series([1000.0] * n, index=idx) if actual is None else pd.Series(actual, index=idx)
        expected = (
            pd.Series([1000.0] * n, index=idx) if expected is None else pd.Series(expected, index=idx)
        )
        return shaded, azimuth, elevation, actual, expected

    def test_sparse_bin_keeps_previous_value_unchanged(self):
        """A bin with fewer than MIN_OBSERVATIONS_PER_BIN observations this
        window is left exactly as it was - never blended from too little
        data."""
        n = MIN_OBSERVATIONS_PER_BIN - 1
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, azimuth_value=90, elevations=[5.0] * n, shaded_mask=[True] * n
        )
        previous = {"90": {"summer": {"elevation": 12.5, "transmittance": 0.3}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.5
        )

        self.assertEqual(profile["90"]["summer"], {"elevation": 12.5, "transmittance": 0.3})

    def test_observation_between_two_anchors_contributes_to_both(self):
        """An observation sitting exactly halfway between two adjacent
        azimuth anchors contributes partial weight to BOTH, instead of
        being attributed entirely to just the nearer one - the core
        behaviour change from hard bin membership to soft, kernel-weighted
        (overlapping) anchor windows."""
        n = 40
        halfway = 90 + AZIMUTH_ANCHOR_SPACING_DEG / 2
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, halfway, [6.0] * n, shaded_mask=[True] * n
        )

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        # A hard-bin model would attribute this observation to exactly one
        # of the two neighbouring anchors and leave the other untouched.
        self.assertIn("summer", profile["90"])
        self.assertIn("summer", profile[str(90 + AZIMUTH_ANCHOR_SPACING_DEG)])

    def test_min_observations_threshold_uses_effective_weight_not_raw_count(self):
        """Enough raw observations to have cleared the old hard-count
        threshold, but far enough from the anchor that their kernel
        weight is small - the EFFECTIVE (weighted) count must still fall
        short, leaving the anchor's previous value untouched."""
        n = MIN_OBSERVATIONS_PER_BIN + 10  # would have passed a raw count
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 90, [6.0] * n, shaded_mask=[True] * n
        )
        far_azimuth = 90 - AZIMUTH_KERNEL_BANDWIDTH_DEG * 2.5  # well inside the cutoff, low weight
        azimuth = pd.Series([far_azimuth] * n, index=azimuth.index)
        previous = {"90": {"summer": {"elevation": 12.5, "transmittance": 0.3}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.5
        )

        self.assertEqual(profile["90"]["summer"], {"elevation": 12.5, "transmittance": 0.3})

    def test_shaded_observations_raise_the_bin_toward_the_blocked_elevation(self):
        """Enough observations, some flagged shaded up to a real elevation -
        the bin's estimate moves toward that elevation (a lower bound: it's
        blocked at least that high)."""
        n = MIN_OBSERVATIONS_PER_BIN + 10
        elevations = list(np.linspace(2.0, 20.0, n))
        # Flag everything below 15 degrees as shaded.
        shaded_mask = [e < 15.0 for e in elevations]
        shaded, azimuth, elevation, actual, expected = self._make_series(n, 45, elevations, shaded_mask)

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        # forgetting_factor=0.0 -> profile is exactly this window's estimate.
        self.assertAlmostEqual(
            profile["45"]["summer"]["elevation"], max(e for e in elevations if e < 15.0), places=5
        )

    def test_no_shading_observed_gives_an_upper_bound_not_zero(self):
        """A bin with plenty of valid, unshaded observations is evidence
        the horizon is at most the lowest elevation actually observed - not
        an unconditional reset to 0, which would erase a real, previously-
        learned obstruction that this window's sun path simply never
        tested (e.g. a seasonal gap)."""
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = list(np.linspace(10.0, 30.0, n))  # never dips low
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 180, elevations, shaded_mask=[False] * n
        )

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        self.assertAlmostEqual(profile["180"]["summer"]["elevation"], min(elevations), places=5)

    def test_no_shading_observed_leaves_transmittance_untouched(self):
        """No shaded instants this window means zero evidence about what
        happens below a horizon the sun never dipped under - transmittance
        must stay exactly at its previous value, not drift toward any
        default."""
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = list(np.linspace(10.0, 30.0, n))
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 180, elevations, shaded_mask=[False] * n
        )
        previous = {"180": {"summer": {"elevation": 5.0, "transmittance": 0.42}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.0
        )

        self.assertEqual(profile["180"]["summer"]["transmittance"], 0.42)

    def test_forgetting_factor_blends_previous_and_window(self):
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = [8.0] * n
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 270, elevations, shaded_mask=[True] * n
        )
        previous = {"270": {"summer": {"elevation": 20.0, "transmittance": 0.0}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.8
        )

        expected_value = 0.8 * 20.0 + 0.2 * 8.0
        self.assertAlmostEqual(profile["270"]["summer"]["elevation"], expected_value, places=5)

    def test_cold_start_unseen_bin_defaults_to_zero_baseline(self):
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = [1.0] * n
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 0, elevations, shaded_mask=[False] * n
        )

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.5
        )

        # No previous profile -> implicit 0.0 baseline, blended with this
        # window's (small) upper-bound estimate.
        self.assertAlmostEqual(profile["0"]["summer"]["elevation"], 0.5 * 0.0 + 0.5 * 1.0, places=5)

    def test_profile_covers_every_azimuth_bin(self):
        idx = pd.date_range("2026-06-01 08:00", periods=1, freq="15min", tz="UTC")
        shaded = pd.Series([False], index=idx)
        azimuth = pd.Series([10.0], index=idx)
        elevation = pd.Series([5.0], index=idx)
        actual = pd.Series([1000.0], index=idx)
        expected = pd.Series([1000.0], index=idx)

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.5
        )

        self.assertEqual(len(profile), 360 // AZIMUTH_ANCHOR_SPACING_DEG)

    def test_transmittance_computed_from_shaded_instants_ratio(self):
        """A partially-shaded bin (tree canopy: 40% of clear-sky power
        still gets through while shaded) learns a transmittance around
        0.4, not a hard 0."""
        n = MIN_OBSERVATIONS_PER_BIN + 10
        elevations = list(np.linspace(2.0, 20.0, n))
        shaded_mask = [e < 15.0 for e in elevations]
        expected = [1000.0] * n
        actual = [0.4 * 1000.0 if s else 1000.0 for s in shaded_mask]
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 45, elevations, shaded_mask, actual=actual, expected=expected
        )

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        self.assertAlmostEqual(profile["45"]["summer"]["transmittance"], 0.4, places=5)

    def test_transmittance_blends_with_forgetting_factor_across_refits(self):
        n = MIN_OBSERVATIONS_PER_BIN + 10
        elevations = [8.0] * n
        shaded_mask = [True] * n
        actual = [0.6 * 1000.0] * n
        expected = [1000.0] * n
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 90, elevations, shaded_mask, actual=actual, expected=expected
        )
        previous = {"90": {"summer": {"elevation": 8.0, "transmittance": 0.2}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.7
        )

        expected_transmittance = 0.7 * 0.2 + 0.3 * 0.6
        self.assertAlmostEqual(
            profile["90"]["summer"]["transmittance"], expected_transmittance, places=5
        )

    def test_too_few_shaded_instants_leaves_transmittance_unchanged(self):
        """The bin clears MIN_OBSERVATIONS_PER_BIN overall, but fewer than
        MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE of them are actually
        shaded - too little to average a transmittance from, so it's left
        at its previous value even though the elevation still updates."""
        n = MIN_OBSERVATIONS_PER_BIN + 10
        self.assertLess(MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE, n)
        elevations = list(np.linspace(2.0, 20.0, n))
        n_shaded = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE - 1
        shaded_mask = [i < n_shaded for i in range(n)]
        actual = [100.0 if s else 1000.0 for s in shaded_mask]
        expected = [1000.0] * n
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 135, elevations, shaded_mask, actual=actual, expected=expected
        )
        previous = {"135": {"summer": {"elevation": 0.0, "transmittance": 0.55}}}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.0
        )

        self.assertEqual(profile["135"]["summer"]["transmittance"], 0.55)
        # Elevation still updates from the (few) shaded instants' max.
        self.assertAlmostEqual(
            profile["135"]["summer"]["elevation"], elevations[n_shaded - 1], places=5
        )

    def test_legacy_bare_float_previous_profile_is_normalized(self):
        """A previous_profile entry persisted before transmittance/season
        existed (a bare float) is used as a uniform prior instead of
        crashing."""
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = [8.0] * n
        shaded, azimuth, elevation, actual, expected = self._make_series(
            n, 195, elevations, shaded_mask=[True] * n
        )
        previous = {"195": 20.0}

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.8
        )

        # Legacy float normalizes to elevation=20.0, transmittance=0.0 as
        # the prior for every season, including summer.
        expected_elevation = 0.8 * 20.0 + 0.2 * 8.0
        self.assertAlmostEqual(profile["195"]["summer"]["elevation"], expected_elevation, places=5)

    def test_season_spanning_window_updates_each_season_independently(self):
        """A refit window spanning two seasons (winter + spring rows for
        the same bin) updates each season's cell from only its own rows,
        without touching the other season's already-persisted entry."""
        n_per_season = MIN_OBSERVATIONS_PER_BIN + 5
        winter_idx = pd.date_range("2026-01-05", periods=n_per_season, freq="30min", tz="UTC")
        spring_idx = pd.date_range("2026-04-05", periods=n_per_season, freq="30min", tz="UTC")
        idx = winter_idx.append(spring_idx)
        n = len(idx)
        azimuth = pd.Series([60.0] * n, index=idx)  # anchor 60 itself
        elevation = pd.Series([6.0] * n, index=idx)
        shaded = pd.Series([True] * n, index=idx)
        actual = pd.Series([300.0] * n, index=idx)
        expected = pd.Series([1000.0] * n, index=idx)
        previous = {
            "60": {
                "winter": {"elevation": 10.0, "transmittance": 0.1},
                "autumn": {"elevation": 3.0, "transmittance": 0.9},
            }
        }

        profile = aggregate_horizon_profile(
            shaded, azimuth, elevation, actual, expected, previous, forgetting_factor=0.5
        )

        # winter and spring both had enough rows this window -> updated.
        self.assertAlmostEqual(profile["60"]["winter"]["elevation"], 0.5 * 10.0 + 0.5 * 6.0, places=5)
        self.assertAlmostEqual(profile["60"]["spring"]["elevation"], 0.5 * 0.0 + 0.5 * 6.0, places=5)
        # autumn was never touched this window -> carried forward as-is.
        self.assertEqual(profile["60"]["autumn"], {"elevation": 3.0, "transmittance": 0.9})
        # summer was never seen (no previous entry, no rows) -> absent.
        self.assertNotIn("summer", profile["60"])


class TestComputeGeometricallyBlindAzimuths(unittest.TestCase):
    """A bin the sun can never test - self-shading or the sun's own path
    never reaching there at this latitude - stays at its cold-start
    default forever, identical to a confirmed-clear reading. This lets the
    two be told apart ahead of any measurement, purely from geometry."""

    # Amsterdam-ish, used throughout - not asserting on a real system's
    # exact values, just the qualitative geometric relationships below.
    LATITUDE = 52.0
    LONGITUDE = 5.0

    def test_south_facing_tilted_panel_is_blind_near_due_north(self):
        blind = compute_geometrically_blind_azimuths(
            surface_tilt=30, surface_azimuth=180, latitude=self.LATITUDE, longitude=self.LONGITUDE
        )
        # Due north (self-shaded - behind a south-facing tilted panel, and
        # at this latitude the sun barely if ever reaches there anyway).
        self.assertIn(0, blind)
        # Due south (the panel's own facing direction) must never be blind.
        self.assertNotIn(180, blind)

    def test_result_only_contains_known_anchor_points(self):
        blind = compute_geometrically_blind_azimuths(
            surface_tilt=30, surface_azimuth=180, latitude=self.LATITUDE, longitude=self.LONGITUDE
        )
        expected_anchors = set(range(0, 360, AZIMUTH_ANCHOR_SPACING_DEG))
        self.assertTrue(blind.issubset(expected_anchors))

    def test_spacing_deg_param_controls_anchor_resolution(self):
        blind = compute_geometrically_blind_azimuths(
            surface_tilt=30,
            surface_azimuth=180,
            latitude=self.LATITUDE,
            longitude=self.LONGITUDE,
            spacing_deg=2,
        )
        expected_anchors = set(range(0, 360, 2))
        self.assertTrue(blind.issubset(expected_anchors))
        self.assertTrue(all(b % 2 == 0 for b in blind))

    def test_flat_panel_has_no_self_shading_only_latitude_driven_blind_bins(self):
        """A horizontal (tilt=0) panel's angle-of-incidence never exceeds
        90 degrees while the sun is up - it can never self-shade. Any blind
        bins for it must be a subset of a tilted panel's own blind bins at
        the same site (tilting can only ever add self-shading, never remove
        a bin the flat panel already couldn't see)."""
        blind_flat = compute_geometrically_blind_azimuths(
            surface_tilt=0, surface_azimuth=180, latitude=self.LATITUDE, longitude=self.LONGITUDE
        )
        blind_tilted = compute_geometrically_blind_azimuths(
            surface_tilt=30, surface_azimuth=180, latitude=self.LATITUDE, longitude=self.LONGITUDE
        )
        self.assertTrue(blind_flat.issubset(blind_tilted))
        self.assertLess(len(blind_flat), len(blind_tilted))


class TestInterpolateHorizonProfile(unittest.TestCase):
    """The query-side counterpart to aggregate_horizon_profile: a
    continuous function of azimuth, not a lookup into fixed bins."""

    @staticmethod
    def _expected_weighted_average(anchor_values_and_distances, cold_start_value=0.0):
        """Independent re-derivation of interpolate_horizon_profile's own
        documented formula (kernel-weighted average, plus a virtual
        cold-start anchor at cold_start_weight) - used to compute exact
        expected values below rather than hand-transcribing decimals."""
        cold_start_weight = math.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
        numerator = cold_start_weight * cold_start_value
        denominator = cold_start_weight
        for value, distance_deg in anchor_values_and_distances:
            w = math.exp(-0.5 * (distance_deg / AZIMUTH_KERNEL_BANDWIDTH_DEG) ** 2)
            numerator += w * value
            denominator += w
        return numerator / denominator

    def test_single_anchor_value_fades_toward_cold_start_with_distance(self):
        """With only one real anchor, a query exactly on it recovers very
        close to that anchor's value (a small, deliberate pull toward the
        cold-start default - see interpolate_horizon_profile's own
        docstring for why a virtual cold-start anchor is always included),
        while a query far from it (its antipode) fades almost all the way
        back to the cold-start default instead of inheriting that one
        anchor's value unboundedly across the entire circle."""
        profile = {"90": {"summer": {"elevation": 20.0, "transmittance": 0.3}}}
        azimuth = pd.Series([90.0, 270.0])  # on the anchor, and its antipode
        season = pd.Series(["summer"] * 2)

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertAlmostEqual(
            elevation.iloc[0], self._expected_weighted_average([(20.0, 0.0)]), places=5
        )
        self.assertAlmostEqual(
            transmittance.iloc[0], self._expected_weighted_average([(0.3, 0.0)]), places=5
        )
        self.assertGreater(elevation.iloc[0], 19.0)  # close to 20, not exactly - see docstring
        self.assertLess(elevation.iloc[1], 0.5)  # antipode: essentially cold-start (0)

    def test_symmetric_query_averages_two_equidistant_anchors(self):
        """A query azimuth exactly halfway between two anchors gets equal
        weight from both - the result is (very close to) their plain
        average, offset only by the same small cold-start pull as above."""
        profile = {
            "80": {"summer": {"elevation": 10.0, "transmittance": 0.1}},
            "100": {"summer": {"elevation": 30.0, "transmittance": 0.3}},
        }
        azimuth = pd.Series([90.0])
        season = pd.Series(["summer"])

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertAlmostEqual(
            elevation.iloc[0],
            self._expected_weighted_average([(10.0, 10.0), (30.0, 10.0)]),
            places=5,
        )
        self.assertAlmostEqual(
            transmittance.iloc[0],
            self._expected_weighted_average([(0.1, 10.0), (0.3, 10.0)]),
            places=5,
        )

    def test_asymmetric_query_weighs_the_nearer_anchor_more(self):
        """A query azimuth closer to one anchor than the other lands
        strictly between their two values, biased toward the nearer one -
        not a 50/50 split, and not equal to either anchor outright."""
        profile = {
            "80": {"summer": {"elevation": 10.0, "transmittance": 0.1}},
            "100": {"summer": {"elevation": 30.0, "transmittance": 0.3}},
        }
        azimuth = pd.Series([85.0])  # 5deg from 80, 15deg from 100
        season = pd.Series(["summer"])

        elevation, _ = interpolate_horizon_profile(profile, azimuth, season)

        self.assertGreater(elevation.iloc[0], 10.0)
        self.assertLess(elevation.iloc[0], 20.0)  # closer to the 80-anchor's value than the midpoint

    def test_circular_wraparound_treats_0_360_as_adjacent(self):
        """Two anchors symmetric around the 0/360 seam (350 and 10) must
        weigh equally on a query AT the seam (0) - a naive, non-circular
        distance would instead treat anchor 350 as ~350deg away (~zero
        weight) and let anchor 10 dominate."""
        profile = {
            "350": {"summer": {"elevation": 5.0, "transmittance": 0.0}},
            "10": {"summer": {"elevation": 25.0, "transmittance": 0.0}},
        }
        azimuth = pd.Series([0.0])
        season = pd.Series(["summer"])

        elevation, _ = interpolate_horizon_profile(profile, azimuth, season)

        self.assertAlmostEqual(
            elevation.iloc[0],
            self._expected_weighted_average([(5.0, 10.0), (25.0, 10.0)]),
            places=5,
        )

    def test_missing_season_at_an_anchor_falls_back_to_cold_start(self):
        """An anchor with data for a DIFFERENT season than the one queried
        contributes the cold-start default (0, 0), not its other season's
        value - a per-season fallback, same as aggregate_horizon_profile's
        own previous-entry lookup."""
        profile = {"90": {"summer": {"elevation": 50.0, "transmittance": 0.5}}}
        azimuth = pd.Series([90.0])
        season = pd.Series(["winter"])

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertEqual(elevation.iloc[0], 0.0)
        self.assertEqual(transmittance.iloc[0], 0.0)

    def test_anchor_with_empty_dict_entry_does_not_crash(self):
        """An anchor present in the profile but with an empty dict value
        (a fresh anchor aggregate_horizon_profile created with no season
        yet clearing MIN_OBSERVATIONS_PER_BIN - very common for the many
        geometrically-blind anchors on a first-ever refit) must be treated
        as cold-start, not crash - regression test for the same bug class
        normalize_bin_entry's own empty-dict handling now covers."""
        profile = {
            "50": {},  # never any evidence for this anchor
            "90": {"summer": {"elevation": 20.0, "transmittance": 0.2}},
        }
        azimuth = pd.Series([90.0])
        season = pd.Series(["summer"])

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertGreater(elevation.iloc[0], 0.0)
        self.assertGreater(transmittance.iloc[0], 0.0)

    def test_empty_profile_returns_cold_start_without_crashing(self):
        elevation, transmittance = interpolate_horizon_profile(
            {}, pd.Series([45.0, 200.0]), pd.Series(["summer", "winter"])
        )

        self.assertTrue((elevation == 0.0).all())
        self.assertTrue((transmittance == 0.0).all())

    def test_legacy_coarse_15_degree_profile_still_interpolates(self):
        """A profile persisted before AZIMUTH_ANCHOR_SPACING_DEG was
        tightened to 5 degrees (only 15-degree-spaced anchors) still works
        - just coarser - with no migration and no special-casing: there is
        no assumed grid, only whichever anchor keys are actually present."""
        profile = {
            "0": {"summer": {"elevation": 0.0, "transmittance": 0.0}},
            "15": {"summer": {"elevation": 20.0, "transmittance": 0.2}},
        }
        azimuth = pd.Series([7.0])  # between the two legacy anchors
        season = pd.Series(["summer"])

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertGreater(elevation.iloc[0], 0.0)
        self.assertLess(elevation.iloc[0], 20.0)
        self.assertGreater(transmittance.iloc[0], 0.0)
        self.assertLess(transmittance.iloc[0], 0.2)

    def test_season_varies_per_row(self):
        """A multi-day forecast can cross a season boundary - each row is
        interpolated against its OWN season's anchor values, not a single
        season applied to the whole batch."""
        profile = {
            "90": {
                "summer": {"elevation": 20.0, "transmittance": 0.2},
                "winter": {"elevation": 5.0, "transmittance": 0.05},
            }
        }
        azimuth = pd.Series([90.0, 90.0])
        season = pd.Series(["summer", "winter"])

        elevation, transmittance = interpolate_horizon_profile(profile, azimuth, season)

        self.assertAlmostEqual(
            elevation.iloc[0], self._expected_weighted_average([(20.0, 0.0)]), places=5
        )
        self.assertAlmostEqual(
            elevation.iloc[1], self._expected_weighted_average([(5.0, 0.0)]), places=5
        )
        self.assertAlmostEqual(
            transmittance.iloc[0], self._expected_weighted_average([(0.2, 0.0)]), places=5
        )
        self.assertAlmostEqual(
            transmittance.iloc[1], self._expected_weighted_average([(0.05, 0.0)]), places=5
        )


class TestComputeDiffuseTransmissionFactor(unittest.TestCase):
    def test_empty_profile_gives_factor_of_one(self):
        """No anchors at all -> interpolate_horizon_profile returns pure
        cold-start (elevation=0, transmittance=0) everywhere -> every
        azimuth's own remaining fraction is 0*sin(0)^2 + cos(0)^2 = 1.0 -
        a fully unobstructed sky dome."""
        self.assertAlmostEqual(compute_diffuse_transmission_factor({}, "summer"), 1.0, places=6)

    def test_a_real_obstruction_reduces_the_factor_below_one(self):
        profile = {"180": {"summer": {"elevation": 30.0, "transmittance": 0.2}}}

        factor = compute_diffuse_transmission_factor(profile, "summer")

        self.assertLess(factor, 1.0)
        self.assertGreater(factor, 0.0)

    def test_matches_the_documented_closed_form(self):
        """Independent re-derivation of the documented formula - mean over
        azimuth of t*sin(h)^2 + cos(h)^2, using interpolate_horizon_profile
        directly - to catch a transcription error in the production
        formula itself, not just re-assert whatever it currently does."""
        from emhass.pv_shading_kalman import AZIMUTH_RENDER_SPACING_DEG

        profile = {
            "90": {"summer": {"elevation": 25.0, "transmittance": 0.1}},
            "270": {"summer": {"elevation": 15.0, "transmittance": 0.3}},
        }
        query_azimuth = pd.Series(np.arange(0, 360, AZIMUTH_RENDER_SPACING_DEG), dtype=float)
        query_season = pd.Series(["summer"] * len(query_azimuth))
        elevation, transmittance = interpolate_horizon_profile(profile, query_azimuth, query_season)
        h_rad = np.radians(elevation.clip(lower=0.0, upper=90.0))
        expected = float((transmittance * np.sin(h_rad) ** 2 + np.cos(h_rad) ** 2).mean())

        self.assertAlmostEqual(
            compute_diffuse_transmission_factor(profile, "summer"), expected, places=8
        )

    def test_a_bigger_hard_obstruction_reduces_the_factor_more(self):
        """A higher learned horizon elevation (blocks more of the sky
        dome) must reduce the diffuse factor further, all else equal."""
        small = compute_diffuse_transmission_factor(
            {"180": {"summer": {"elevation": 10.0, "transmittance": 0.0}}}, "summer"
        )
        big = compute_diffuse_transmission_factor(
            {"180": {"summer": {"elevation": 60.0, "transmittance": 0.0}}}, "summer"
        )
        self.assertLess(big, small)


class TestComputeSunPathEnvelope(unittest.TestCase):
    """Real solar-geometry sweep - a coarse fine_step_deg keeps these
    tests fast without changing what's being verified (the function's own
    correctness, not any particular resolution)."""

    LATITUDE, LONGITUDE = 52.0, 5.0
    STEP = 30

    def test_returns_one_entry_per_anchor_in_both_curves(self):
        sun_min, sun_max = compute_sun_path_envelope(self.LATITUDE, self.LONGITUDE, self.STEP)

        expected_keys = set(np.arange(0, 360, self.STEP))
        self.assertEqual(set(sun_min.keys()), expected_keys)
        self.assertEqual(set(sun_max.keys()), expected_keys)

    def test_max_is_never_below_min_where_both_are_present(self):
        sun_min, sun_max = compute_sun_path_envelope(self.LATITUDE, self.LONGITUDE, self.STEP)

        for az in sun_min:
            if sun_min[az] is not None and sun_max[az] is not None:
                self.assertGreaterEqual(sun_max[az], sun_min[az])

    def test_due_south_is_reached_at_this_latitude(self):
        """At 52N, the sun crosses due south every single day (at solar
        noon) - that azimuth's envelope must never be None."""
        sun_min, sun_max = compute_sun_path_envelope(self.LATITUDE, self.LONGITUDE, self.STEP)

        self.assertIsNotNone(sun_min[180])
        self.assertIsNotNone(sun_max[180])


class TestAggregatePartialTransmittanceSurface(unittest.TestCase):
    def _make_series(self, n, azimuth_value, elevation_value, ratio_value, hard_blocked_mask=None):
        idx = pd.date_range("2026-06-01 08:00", periods=n, freq="15min", tz="UTC")
        azimuth = pd.Series([float(azimuth_value)] * n, index=idx)
        elevation = pd.Series([float(elevation_value)] * n, index=idx)
        expected = pd.Series([1000.0] * n, index=idx)
        actual = pd.Series([1000.0 * ratio_value] * n, index=idx)
        hard_blocked = pd.Series(hard_blocked_mask or [False] * n, index=idx)
        return azimuth, elevation, actual, expected, hard_blocked

    def test_fits_a_real_partial_obstruction_at_its_anchor(self):
        """Enough genuinely-partial (shaded but not hard-blocked) evidence
        at a known (azimuth anchor, elevation anchor) - the surface learns
        a transmittance close to the true injected ratio there."""
        n = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE + 15
        azimuth, elevation, actual, expected, hard_blocked = self._make_series(
            n, azimuth_value=90, elevation_value=15, ratio_value=0.4
        )
        shaded = pd.Series([True] * n, index=azimuth.index)  # all genuinely partial

        surface = aggregate_partial_transmittance_surface(
            shaded, hard_blocked, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        self.assertAlmostEqual(surface["90"]["summer"]["15"], 0.4, places=5)

    def test_hard_blocked_instants_are_excluded_from_the_partial_fit(self):
        """An instant that IS hard-blocked must never feed this surface,
        even if it's also marked shaded (every hard-blocked instant is
        shaded under the broader gate too) - partial = shaded & ~hard_blocked."""
        n = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE + 15
        azimuth, elevation, actual, expected, hard_blocked = self._make_series(
            n, azimuth_value=90, elevation_value=15, ratio_value=0.01, hard_blocked_mask=[True] * n
        )
        shaded = pd.Series([True] * n, index=azimuth.index)

        surface = aggregate_partial_transmittance_surface(
            shaded, hard_blocked, azimuth, elevation, actual, expected, None, forgetting_factor=0.0
        )

        # No genuinely-partial evidence at all (everything was hard-blocked)
        # -> nothing learned for this azimuth anchor's season.
        self.assertNotIn("summer", surface.get("90", {}))

    def test_insufficient_observations_keeps_previous_value(self):
        n = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE - 1
        azimuth, elevation, actual, expected, hard_blocked = self._make_series(
            n, azimuth_value=90, elevation_value=15, ratio_value=0.4
        )
        shaded = pd.Series([True] * n, index=azimuth.index)
        previous = {"90": {"summer": {"15": 0.7}}}

        surface = aggregate_partial_transmittance_surface(
            shaded, hard_blocked, azimuth, elevation, actual, expected, previous, forgetting_factor=0.5
        )

        self.assertEqual(surface["90"]["summer"]["15"], 0.7)

    def test_forgetting_factor_blends_previous_and_window(self):
        n = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE + 15
        azimuth, elevation, actual, expected, hard_blocked = self._make_series(
            n, azimuth_value=90, elevation_value=15, ratio_value=0.4
        )
        shaded = pd.Series([True] * n, index=azimuth.index)
        previous = {"90": {"summer": {"15": 0.8}}}

        surface = aggregate_partial_transmittance_surface(
            shaded, hard_blocked, azimuth, elevation, actual, expected, previous, forgetting_factor=0.5
        )

        self.assertAlmostEqual(surface["90"]["summer"]["15"], 0.5 * 0.8 + 0.5 * 0.4, places=5)

    def test_a_different_previously_learned_elevation_anchor_is_untouched(self):
        """A season map is a dict of many elevation anchors, not a single
        scalar - updating one anchor this window must not disturb a
        DIFFERENT elevation anchor's own previously-persisted value at the
        same azimuth."""
        n = MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE + 15
        azimuth, elevation, actual, expected, hard_blocked = self._make_series(
            n, azimuth_value=90, elevation_value=15, ratio_value=0.4
        )
        shaded = pd.Series([True] * n, index=azimuth.index)
        previous = {"90": {"summer": {"45": 0.9}}}

        surface = aggregate_partial_transmittance_surface(
            shaded, hard_blocked, azimuth, elevation, actual, expected, previous, forgetting_factor=0.0
        )

        self.assertEqual(surface["90"]["summer"]["45"], 0.9)
        self.assertAlmostEqual(surface["90"]["summer"]["15"], 0.4, places=5)


class TestInterpolatePartialTransmittance(unittest.TestCase):
    @staticmethod
    def _expected_partial_average(anchor_values_and_distances, cold_start_value=1.0):
        """Independent re-derivation of interpolate_partial_transmittance's
        own documented formula - a 2D kernel-weighted average plus a
        virtual cold-start anchor (value 1.0, not interpolate_horizon_
        profile's 0.0 - this surface only ever REDUCES light)."""
        cold_start_weight = math.exp(-0.5 * AZIMUTH_KERNEL_CUTOFF**2)
        numerator = cold_start_weight * cold_start_value
        denominator = cold_start_weight
        for value, az_distance_deg, el_distance_deg in anchor_values_and_distances:
            w_az = math.exp(-0.5 * (az_distance_deg / AZIMUTH_KERNEL_BANDWIDTH_DEG) ** 2)
            w_el = math.exp(-0.5 * (el_distance_deg / ELEVATION_KERNEL_BANDWIDTH_DEG) ** 2)
            w = w_az * w_el
            numerator += w * value
            denominator += w
        return numerator / denominator

    def test_empty_surface_returns_one_everywhere(self):
        transmittance = interpolate_partial_transmittance(
            {}, pd.Series([90.0, 270.0]), pd.Series([10.0, 20.0]), pd.Series(["summer", "winter"])
        )

        self.assertTrue((transmittance == 1.0).all())

    def test_single_anchor_value_fades_toward_one_with_distance(self):
        # 270 (the antipode, 180deg away) underflows past float64's
        # precision floor when added to the cold-start weight - 60deg away
        # is still comfortably closer to 1.0 than the exact match, without
        # hitting that floor.
        surface = {"90": {"summer": {"15": 0.4}}}
        azimuth = pd.Series([90.0, 150.0])
        elevation = pd.Series([15.0, 15.0])
        season = pd.Series(["summer"] * 2)

        transmittance = interpolate_partial_transmittance(surface, azimuth, elevation, season)

        self.assertAlmostEqual(
            transmittance.iloc[0], self._expected_partial_average([(0.4, 0.0, 0.0)]), places=5
        )
        self.assertGreater(transmittance.iloc[1], transmittance.iloc[0])  # far away -> closer to 1.0
        self.assertLess(transmittance.iloc[1], 1.0)

    def test_missing_season_at_an_anchor_falls_back_to_one(self):
        surface = {"90": {"summer": {"15": 0.2}}}
        azimuth = pd.Series([90.0])
        elevation = pd.Series([15.0])
        season = pd.Series(["winter"])

        transmittance = interpolate_partial_transmittance(surface, azimuth, elevation, season)

        self.assertEqual(transmittance.iloc[0], 1.0)

    def test_sun_path_gate_forces_one_outside_the_real_envelope(self):
        """A query elevation outside the sun's own real range at that
        azimuth must return 1.0 unconditionally, even with strong nearby
        anchor evidence - the elevation kernel would otherwise bleed a
        value into a physically impossible combination."""
        surface = {"90": {"summer": {"15": 0.2}}}
        azimuth = pd.Series([90.0])
        elevation = pd.Series([15.0])
        season = pd.Series(["summer"])
        sun_min_curve = {90.0: 20.0}  # the sun is NEVER below 20deg at az=90
        sun_max_curve = {90.0: 60.0}

        transmittance = interpolate_partial_transmittance(
            surface, azimuth, elevation, season, sun_min_curve, sun_max_curve
        )

        self.assertEqual(transmittance.iloc[0], 1.0)

    def test_sun_path_gate_allows_a_physically_possible_query_through(self):
        surface = {"90": {"summer": {"15": 0.2}}}
        azimuth = pd.Series([90.0])
        elevation = pd.Series([15.0])
        season = pd.Series(["summer"])
        sun_min_curve = {90.0: 0.0}
        sun_max_curve = {90.0: 60.0}

        transmittance = interpolate_partial_transmittance(
            surface, azimuth, elevation, season, sun_min_curve, sun_max_curve
        )

        self.assertLess(transmittance.iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
