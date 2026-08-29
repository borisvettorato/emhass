#!/usr/bin/env python
"""Unit tests for pv_shading_kalman.py - pure math, no I/O, no live network."""

import unittest

import numpy as np
import pandas as pd

from emhass.pv_shading_kalman import (
    AZIMUTH_BIN_WIDTH_DEG,
    MIN_EXPECTED_POWER_W,
    MIN_OBSERVATIONS_PER_BIN,
    MIN_SHADED_OBSERVATIONS_FOR_TRANSMITTANCE,
    aggregate_horizon_profile,
    classify_shaded_instants,
    compute_geometrically_blind_azimuths,
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


class TestAggregateHorizonProfile(unittest.TestCase):
    """All fixtures below use June 2026 dates, which fall in the
    'summer' meteorological season - profile[bin]["summer"] is the
    entry under test unless noted otherwise."""

    def _make_series(self, n, azimuth_value, elevations, shaded_mask, actual=None, expected=None):
        idx = pd.date_range("2026-06-01 08:00", periods=n, freq="15min", tz="UTC")
        azimuth = pd.Series([azimuth_value] * n, index=idx)
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

        self.assertEqual(len(profile), 360 // AZIMUTH_BIN_WIDTH_DEG)

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
        azimuth = pd.Series([60] * n, index=idx)
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

    def test_result_only_contains_known_bin_starts(self):
        blind = compute_geometrically_blind_azimuths(
            surface_tilt=30, surface_azimuth=180, latitude=self.LATITUDE, longitude=self.LONGITUDE
        )
        expected_bins = set(range(0, 360, AZIMUTH_BIN_WIDTH_DEG))
        self.assertTrue(blind.issubset(expected_bins))

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


if __name__ == "__main__":
    unittest.main()
