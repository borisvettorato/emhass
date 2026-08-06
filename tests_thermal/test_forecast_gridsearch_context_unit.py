"""Unit tests for physics-context windowing in forecast_gridsearch."""

from __future__ import annotations

import numpy as np

from emhass.thermal.forecast_gridsearch import create_physics_context_sequences


def test_create_physics_context_sequences_alignment() -> None:
    room = np.array([10, 11, 12, 13, 14, 15], dtype=np.float32)
    outdoor = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    solar = np.array([0, 10, 20, 30, 40, 50], dtype=np.float32)

    # lookback=2, lookahead=2 => max_i = 3
    ctx = create_physics_context_sequences(room, outdoor, solar, lookback=2, lookahead=2)

    assert ctx["room_temp_prev"].shape == (3,)
    assert ctx["outdoor_temp"].shape == (3, 2)
    assert ctx["solar_heat"].shape == (3, 2)

    # i=0 -> start=2, room_prev=room[1]=11, horizon rows [2:4]
    assert ctx["room_temp_prev"][0] == 11
    assert np.all(ctx["outdoor_temp"][0] == np.array([3, 4], dtype=np.float32))
    assert np.all(ctx["solar_heat"][0] == np.array([20, 30], dtype=np.float32))


def test_create_physics_context_sequences_empty_when_insufficient_history() -> None:
    room = np.array([1, 2, 3], dtype=np.float32)
    outdoor = np.array([1, 2, 3], dtype=np.float32)
    solar = np.array([0, 0, 0], dtype=np.float32)

    ctx = create_physics_context_sequences(room, outdoor, solar, lookback=3, lookahead=2)

    assert ctx["room_temp_prev"].shape == (0,)
    assert ctx["outdoor_temp"].shape == (0, 2)
    assert ctx["solar_heat"].shape == (0, 2)
