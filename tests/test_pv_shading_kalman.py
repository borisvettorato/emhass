#!/usr/bin/env python
"""Unit tests for pv_shading_kalman.py - pure math, no I/O, no live network."""

import unittest

import numpy as np
import pandas as pd

from emhass.pv_shading_kalman import (
    AZIMUTH_BIN_WIDTH_DEG,
    MIN_EXPECTED_POWER_W,
    MIN_OBSERVATIONS_PER_BIN,
    aggregate_horizon_profile,
    classify_shaded_instants,
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


class TestAggregateHorizonProfile(unittest.TestCase):
    def _make_series(self, n, azimuth_value, elevations, shaded_mask):
        idx = pd.date_range("2026-06-01 08:00", periods=n, freq="15min", tz="UTC")
        azimuth = pd.Series([azimuth_value] * n, index=idx)
        elevation = pd.Series(elevations, index=idx)
        shaded = pd.Series(shaded_mask, index=idx)
        return shaded, azimuth, elevation

    def test_sparse_bin_keeps_previous_value_unchanged(self):
        """A bin with fewer than MIN_OBSERVATIONS_PER_BIN observations this
        window is left exactly as it was - never blended from too little
        data."""
        n = MIN_OBSERVATIONS_PER_BIN - 1
        shaded, azimuth, elevation = self._make_series(
            n, azimuth_value=90, elevations=[5.0] * n, shaded_mask=[True] * n
        )
        previous = {"90": 12.5}

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, previous, forgetting_factor=0.5)

        self.assertEqual(profile["90"], 12.5)

    def test_shaded_observations_raise_the_bin_toward_the_blocked_elevation(self):
        """Enough observations, some flagged shaded up to a real elevation -
        the bin's estimate moves toward that elevation (a lower bound: it's
        blocked at least that high)."""
        n = MIN_OBSERVATIONS_PER_BIN + 10
        elevations = list(np.linspace(2.0, 20.0, n))
        # Flag everything below 15 degrees as shaded.
        shaded_mask = [e < 15.0 for e in elevations]
        shaded, azimuth, elevation = self._make_series(n, 45, elevations, shaded_mask)

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, None, forgetting_factor=0.0)

        # forgetting_factor=0.0 -> profile is exactly this window's estimate.
        self.assertAlmostEqual(profile["45"], max(e for e in elevations if e < 15.0), places=5)

    def test_no_shading_observed_gives_an_upper_bound_not_zero(self):
        """A bin with plenty of valid, unshaded observations is evidence
        the horizon is at most the lowest elevation actually observed - not
        an unconditional reset to 0, which would erase a real, previously-
        learned obstruction that this window's sun path simply never
        tested (e.g. a seasonal gap)."""
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = list(np.linspace(10.0, 30.0, n))  # never dips low
        shaded, azimuth, elevation = self._make_series(n, 180, elevations, shaded_mask=[False] * n)

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, None, forgetting_factor=0.0)

        self.assertAlmostEqual(profile["180"], min(elevations), places=5)

    def test_forgetting_factor_blends_previous_and_window(self):
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = [8.0] * n
        shaded, azimuth, elevation = self._make_series(n, 270, elevations, shaded_mask=[True] * n)
        previous = {"270": 20.0}

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, previous, forgetting_factor=0.8)

        expected_value = 0.8 * 20.0 + 0.2 * 8.0
        self.assertAlmostEqual(profile["270"], expected_value, places=5)

    def test_cold_start_unseen_bin_defaults_to_zero_baseline(self):
        n = MIN_OBSERVATIONS_PER_BIN + 5
        elevations = [1.0] * n
        shaded, azimuth, elevation = self._make_series(n, 0, elevations, shaded_mask=[False] * n)

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, None, forgetting_factor=0.5)

        # No previous profile -> implicit 0.0 baseline, blended with this
        # window's (small) upper-bound estimate.
        self.assertAlmostEqual(profile["0"], 0.5 * 0.0 + 0.5 * 1.0, places=5)

    def test_profile_covers_every_azimuth_bin(self):
        idx = pd.date_range("2026-06-01 08:00", periods=1, freq="15min", tz="UTC")
        shaded = pd.Series([False], index=idx)
        azimuth = pd.Series([10.0], index=idx)
        elevation = pd.Series([5.0], index=idx)

        profile = aggregate_horizon_profile(shaded, azimuth, elevation, None, forgetting_factor=0.5)

        self.assertEqual(len(profile), 360 // AZIMUTH_BIN_WIDTH_DEG)


if __name__ == "__main__":
    unittest.main()
