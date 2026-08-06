"""Unit tests for PINN forecast model behavior.

These tests avoid real-data loading and file I/O.
"""

import numpy as np
import torch
import pytest

from emhass.thermal.pinn_forecaster import PInnThermalForecaster
from emhass.thermal.pinn_model import QuantilePhysicsInformedLSTM


pytestmark = pytest.mark.unit


class TestPINNForecastUnit:
    def test_model_initialization(self, pinn_model):
        assert pinn_model is not None
        assert isinstance(pinn_model, torch.nn.Module)

    def test_forward_pass_shapes_synthetic(self, pinn_model):
        input_size = getattr(pinn_model, "input_size", 37)
        X = torch.randn(1, 192, input_size, dtype=torch.float32)

        output = pinn_model(X)

        assert "q10" in output
        assert "q50" in output
        assert "q90" in output
        assert "physics_params" in output

        assert output["q10"].shape == (1, 144, 3)
        assert output["q50"].shape == (1, 144, 3)
        assert output["q90"].shape == (1, 144, 3)
        assert output["physics_params"].shape == (1, 5)

    def test_physics_params_positive_synthetic(self, pinn_model):
        input_size = getattr(pinn_model, "input_size", 37)
        X = torch.tensor(np.random.randn(1, 192, input_size), dtype=torch.float32)

        output = pinn_model(X)
        physics = output["physics_params"].detach().numpy()[0]

        assert np.all(physics > 0), "Softplus-constrained physics parameters should be positive"

    def test_stateful_forward_returns_state(self):
        pinn_model = QuantilePhysicsInformedLSTM(input_size=37, lookahead=144)
        input_size = getattr(pinn_model, "input_size", 37)
        X = torch.randn(1, 192, input_size, dtype=torch.float32)

        first = pinn_model(X)
        assert "state_out" in first
        h, c = first["state_out"]
        assert h.shape[0] == getattr(pinn_model, "num_layers", 2)
        assert c.shape == h.shape

        second = pinn_model(X, state=(h, c))
        h2, c2 = second["state_out"]
        assert h2.shape == h.shape
        assert c2.shape == c.shape

    def test_forecaster_stateful_stream_and_reset(self):
        class IdentityScaler:
            def inverse_transform(self, arr):
                return arr

        f = PInnThermalForecaster(model_path=None, input_size=37, lookahead=144, device="cpu")
        f.scaler_y = IdentityScaler()

        X = np.random.randn(192, 37).astype(np.float32)

        _ = f.forecast(
            X,
            stateful=True,
            current_timestamp="2026-03-29T10:00:00+00:00",
            max_gap_minutes=30,
        )
        assert f.state is not None

        prev_h = f.state[0].clone()
        _ = f.forecast(
            X,
            stateful=True,
            current_timestamp="2026-03-29T10:15:00+00:00",
            max_gap_minutes=30,
        )
        assert f.state is not None
        assert not torch.equal(f.state[0], prev_h)

        # Gap larger than threshold should force a reset before processing.
        _ = f.forecast(
            X,
            stateful=True,
            current_timestamp="2026-03-29T12:00:00+00:00",
            max_gap_minutes=30,
        )
        assert f.state is not None

        f.reset_state()
        assert f.state is None
        assert f.last_timestamp is None
