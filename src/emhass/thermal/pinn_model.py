"""
Physics-Informed Neural Network (PINN) for thermal forecasting.
"""
import torch
import torch.nn as nn
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class QuantilePhysicsInformedLSTM(nn.Module):
    """
    PINN model with quantile outputs (q10, q50, q90)
    Learns building thermal parameters: C, R, α_solar, η_hp, λ_inf
    """
    
    def __init__(
        self,
        input_size: int,
        hidden: int = 128,
        num_layers: int = 2,
        lookahead: int = 144,  # 36h @ 15min
        targets: int = 3,      # room_temp, electric_power, gas_consumption
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.lookahead = lookahead
        self.targets = targets
        
        # LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        # THREE output heads (quantiles)
        self.fc_q10 = nn.Linear(hidden, lookahead * targets)
        self.fc_q50 = nn.Linear(hidden, lookahead * targets)
        self.fc_q90 = nn.Linear(hidden, lookahead * targets)
        
        # Physics parameters
        self.fc_physics = nn.Linear(hidden, 5)  # C, R, α, η, λ
        
        self.softplus = nn.Softplus()
        
        logger.info(
            f"✅ PINN initialized: "
            f"input={input_size}, hidden={hidden}, layers={num_layers}, lookahead={lookahead}"
        )
    
    def init_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build an initial hidden/cell state for stateful streaming."""
        target_device = device or next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden, device=target_device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden, device=target_device)
        return h0, c0

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, 192, input_size) = 48h @ 15min
            state: Optional LSTM hidden/cell state (h, c)
        
        Returns:
            dict with q10, q50, q90, physics_params, state_out
        """
        # LSTM forward
        if state is not None:
            lstm_out, state_out = self.lstm(x, state)
        else:
            lstm_out, state_out = self.lstm(x)
        hidden_state = lstm_out[:, -1, :]
        
        # Quantile predictions
        q10 = self.fc_q10(hidden_state).reshape(-1, self.lookahead, self.targets)
        q50 = self.fc_q50(hidden_state).reshape(-1, self.lookahead, self.targets)
        q90 = self.fc_q90(hidden_state).reshape(-1, self.lookahead, self.targets)
        
        # Physics parameters (always positive)
        physics_params = self.softplus(self.fc_physics(hidden_state))
        
        return {
            'q10': q10,
            'q50': q50,
            'q90': q90,
            'physics_params': physics_params,
            'state_out': state_out,
        }
    
    def get_physics_params(self) -> Dict[str, float]:
        """Extract learned physics parameters for inspection"""
        return {
            'C': 'Thermal capacity [kJ/K]',
            'R': 'Thermal resistance [K/W]',
            'alpha_solar': 'Solar gain coefficient',
            'eta_hp': 'Heat pump COP',
            'lambda_inf': 'Infiltration rate',
        }


class QuantileLoss(nn.Module):
    """
    Combined loss: quantile regression + physics constraints + ordering
    """
    
    def __init__(
        self,
        weight_physics: float = 0.1,
        weight_physics_balance: float = 0.05,
        timestep_hours: float = 0.25,
        gas_kwh_per_m3: float = 10.5,
        boiler_efficiency: float = 0.90,
    ):
        super().__init__()
        self.weight_physics = weight_physics
        self.weight_physics_balance = weight_physics_balance
        self.timestep_hours = max(float(timestep_hours), 1e-6)
        self.gas_kwh_per_m3 = float(gas_kwh_per_m3)
        self.boiler_efficiency = float(boiler_efficiency)
    
    def quantile_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        quantile: float,
    ) -> torch.Tensor:
        """Quantile regression loss"""
        error = target - pred
        return torch.mean(torch.max((quantile - 1) * error, quantile * error))
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        physics_context: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: dict with q10, q50, q90, physics_params
            target: ground truth (batch, lookahead, targets)
            physics_context: optional exogenous tensors used for explicit
                energy-balance constraints. Expected keys:
                - room_temp_prev: (batch,) or (batch, 1)
                - outdoor_temp:   (batch, lookahead)
                - solar_heat:     (batch, lookahead) [optional]
        
        Returns:
            dict with individual loss components
        """
        q10 = output['q10']
        q50 = output['q50']
        q90 = output['q90']
        physics_params = output['physics_params']
        
        # ============ Quantile losses ============
        loss_q10 = self.quantile_loss(q10, target, quantile=0.10)
        loss_q50 = self.quantile_loss(q50, target, quantile=0.50)
        loss_q90 = self.quantile_loss(q90, target, quantile=0.90)
        
        # ============ Order constraint (q10 < q50 < q90) ============
        order_loss = (
            torch.relu(q10 - q50).mean() +
            torch.relu(q50 - q90).mean()
        )
        
        # ============ Physics constraints ============
        C, R, alpha, eta, lambda_inf = [
            physics_params[:, i] for i in range(5)
        ]
        
        physics_loss = (
            torch.clamp(10 - C, min=0) ** 2 +           # C >= 10
            torch.clamp(1/R - 0.1, min=0) ** 2 +        # R >= 10
            torch.clamp(alpha - 0.5, min=0) ** 2 +      # α upper-bounded
            torch.clamp(3 - eta, min=0) ** 2 +          # η >= 3 (COP)
            torch.clamp(lambda_inf - 0.1, min=0) ** 2   # λ upper-bounded
        ).mean()

        # ============ Explicit energy-balance consistency ============
        # Applied only when the model predicts [room_temp, electric_power, gas_consumption]
        # and exogenous context is supplied by the training loop.
        physics_balance_loss = torch.zeros((), device=q50.device, dtype=q50.dtype)
        if (
            physics_context is not None
            and q50.ndim == 3
            and q50.shape[-1] >= 3
            and "outdoor_temp" in physics_context
            and "room_temp_prev" in physics_context
        ):
            room_temp = q50[:, :, 0]
            elec_power = torch.relu(q50[:, :, 1])
            gas_flow = torch.relu(q50[:, :, 2])

            room_prev = physics_context["room_temp_prev"]
            if room_prev.ndim == 1:
                room_prev = room_prev.unsqueeze(1)
            room_hist = torch.cat([room_prev, room_temp[:, :-1]], dim=1)
            dT_dt = (room_temp - room_hist) / self.timestep_hours

            outdoor = physics_context["outdoor_temp"]
            if outdoor.ndim == 1:
                outdoor = outdoor.unsqueeze(1).expand_as(room_temp)

            solar = physics_context.get("solar_heat")
            if solar is None:
                solar = torch.zeros_like(room_temp)
            elif solar.ndim == 1:
                solar = solar.unsqueeze(1).expand_as(room_temp)

            # Thermal input proxy from predicted electric + gas consumption.
            heat_from_hp = eta.unsqueeze(1) * elec_power
            heat_from_boiler = (self.gas_kwh_per_m3 * self.boiler_efficiency) * gas_flow
            heat_input = heat_from_hp + heat_from_boiler

            heat_loss = (room_temp - outdoor) / torch.clamp(R.unsqueeze(1), min=1e-6)
            solar_gain = alpha.unsqueeze(1) * torch.relu(solar)

            # RC energy balance residual:
            #   C * dT/dt = heat_input + solar_gain - heat_loss
            residual = C.unsqueeze(1) * dT_dt - (heat_input + solar_gain - heat_loss)
            physics_balance_loss = torch.mean(torch.sqrt(residual ** 2 + 1e-6))
        
        # ============ Total loss ============
        total_loss = (
            loss_q10 +
            loss_q50 +
            loss_q90 +
            0.5 * order_loss +
            self.weight_physics * physics_loss +
            self.weight_physics_balance * physics_balance_loss
        )
        
        return {
            'total': total_loss,
            'q10': loss_q10,
            'q50': loss_q50,
            'q90': loss_q90,
            'order': order_loss,
            'physics': physics_loss,
            'physics_balance': physics_balance_loss,
        }