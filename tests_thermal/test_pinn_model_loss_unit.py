"""Unit tests for explicit energy-balance term in QuantileLoss."""

from __future__ import annotations

import torch

from emhass.thermal.pinn_model import QuantileLoss


def _build_output(
    room_temp: torch.Tensor,
    elec_power: torch.Tensor,
    gas_flow: torch.Tensor,
    physics_params: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Build q10/q50/q90 consistently to focus this test on physics_balance behavior.
    q50 = torch.stack([room_temp, elec_power, gas_flow], dim=-1)
    return {
        "q10": q50 - 0.1,
        "q50": q50,
        "q90": q50 + 0.1,
        "physics_params": physics_params,
    }


def test_physics_balance_penalizes_inconsistent_energy() -> None:
    # One sample, four timesteps.
    room_temp = torch.tensor([[20.0, 20.2, 20.4, 20.6]], dtype=torch.float32)
    outdoor = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
    solar = torch.zeros_like(room_temp)
    room_prev = torch.tensor([19.8], dtype=torch.float32)

    # Physics params: C, R, alpha, eta, lambda
    physics_params = torch.tensor([[30.0, 12.0, 0.0, 3.0, 0.05]], dtype=torch.float32)

    # Case A: too little energy input for rising room temperature.
    out_low = _build_output(
        room_temp=room_temp,
        elec_power=torch.tensor([[0.05, 0.05, 0.05, 0.05]], dtype=torch.float32),
        gas_flow=torch.zeros_like(room_temp),
        physics_params=physics_params,
    )

    # Case B: enough energy input to better match dynamics.
    out_high = _build_output(
        room_temp=room_temp,
        elec_power=torch.tensor([[4.0, 4.0, 4.0, 4.0]], dtype=torch.float32),
        gas_flow=torch.zeros_like(room_temp),
        physics_params=physics_params,
    )

    criterion = QuantileLoss(weight_physics=0.0, weight_physics_balance=1.0)
    target = out_low["q50"].clone()
    context = {
        "room_temp_prev": room_prev,
        "outdoor_temp": outdoor,
        "solar_heat": solar,
    }

    losses_low = criterion(out_low, target, physics_context=context)
    losses_high = criterion(out_high, target, physics_context=context)

    assert losses_low["physics_balance"].item() > losses_high["physics_balance"].item()


def test_physics_balance_is_zero_without_context() -> None:
    room_temp = torch.tensor([[20.0, 20.0, 20.0]], dtype=torch.float32)
    elec = torch.zeros_like(room_temp)
    gas = torch.zeros_like(room_temp)
    physics_params = torch.tensor([[30.0, 12.0, 0.0, 3.0, 0.05]], dtype=torch.float32)
    output = _build_output(room_temp, elec, gas, physics_params)

    criterion = QuantileLoss(weight_physics=0.0, weight_physics_balance=1.0)
    losses = criterion(output, output["q50"])

    assert torch.isclose(losses["physics_balance"], torch.tensor(0.0))
