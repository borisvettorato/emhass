"""
Step 1: Test PINN Forecast with REAL DATA + Generate Plot
"""
import pytest
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


pytestmark = pytest.mark.integration


class TestPINNForecastRealData:
    """Test PINN forecasting with real data"""
    
    # test_model_initialization intentionally lives only in
    # test_pinn_forecast_unit.py (a byte-identical isinstance check to the
    # one that used to be here) - this file's own test_forward_pass_real_data/
    # test_quantile_ordering/test_physics_params_positive already exercise
    # this same pinn_model fixture end-to-end (a strictly stronger implicit
    # proof it constructed correctly, including its checkpoint-loading path)
    # so a bare "is it an nn.Module" duplicate added nothing here.

    def test_forward_pass_real_data(self, pinn_model, sample_input_data_real):
        """Test forward pass with REAL data"""
        X = torch.tensor(sample_input_data_real[np.newaxis, :, :], dtype=torch.float32)
        
        logger.debug(f"Input shape: {X.shape}")
        
        output = pinn_model(X)
        
        assert 'q10' in output
        assert 'q50' in output
        assert 'q90' in output
        assert 'physics_params' in output
        
        assert output['q10'].shape == (1, 144, 3)
        assert output['q50'].shape == (1, 144, 3)
        assert output['q90'].shape == (1, 144, 3)
        assert output['physics_params'].shape == (1, 5)
        
        logger.info("✅ Forward pass OK")
    
    def test_quantile_ordering(self, pinn_model, sample_input_data_real):
        """q10 <= q50 <= q90 must hold for the overwhelming majority of the
        144-timestep forecast horizon.

        The model has NO architectural guarantee of this ordering - q10/
        q50/q90 come from 3 independent linear heads (see
        src/emhass/thermal/pinn_model.py's fc_q10/fc_q50/fc_q90) - it's
        only encouraged via a soft relu-penalty term in the training loss
        (order_constraint_loss). So a small violation rate is expected
        even for a well-trained model, hence a rate ceiling rather than
        demanding zero violations. The pinn_model fixture tries to load a
        trained checkpoint and falls back to random-initialized weights
        if that fails (see the fixture's own logging) - a high violation
        rate here is real, useful signal that the checkpoint is missing
        or broken, not something to silently swallow into a log line.
        """
        X = torch.tensor(sample_input_data_real[np.newaxis, :, :], dtype=torch.float32)

        output = pinn_model(X)

        q10 = output['q10'].detach().numpy()[0, :, 0]  # Temperature
        q50 = output['q50'].detach().numpy()[0, :, 0]
        q90 = output['q90'].detach().numpy()[0, :, 0]

        n = len(q10)
        violations_1 = int(np.sum(q10 > q50))
        violations_2 = int(np.sum(q50 > q90))

        logger.debug(f"Quantile ordering violations: q10>q50={violations_1}/{n}, q50>q90={violations_2}/{n}")

        max_violation_rate = 0.10
        assert violations_1 <= max_violation_rate * n, (
            f"q10 > q50 in {violations_1}/{n} timesteps (>{max_violation_rate:.0%}) - "
            "quantile ordering badly violated (untrained model or broken checkpoint?)"
        )
        assert violations_2 <= max_violation_rate * n, (
            f"q50 > q90 in {violations_2}/{n} timesteps (>{max_violation_rate:.0%}) - "
            "quantile ordering badly violated (untrained model or broken checkpoint?)"
        )
        logger.info("✅ Quantile ordering OK")
    
    def test_physics_params_positive(self, pinn_model, sample_input_data_real):
        """Test physics parameters are positive"""
        X = torch.tensor(sample_input_data_real[np.newaxis, :, :], dtype=torch.float32)
        
        output = pinn_model(X)
        physics = output['physics_params'].detach().numpy()[0]
        
        assert np.all(physics > 0), "Physics params should be positive (Softplus enforces this)"
        
        logger.info(
            f"✅ Physics params positive:\n"
            f"   C: {physics[0]:.2f} kJ/K\n"
            f"   R: {physics[1]:.2f} K/W\n"
            f"   α_solar: {physics[2]:.3f}\n"
            f"   η_hp: {physics[3]:.2f}\n"
            f"   λ_inf: {physics[4]:.3f}"
        )
    
    def test_forecast_generates_plot(self, pinn_model, sample_input_data_real, load_real_data):
        """Test forecast generation and plot creation with REAL data"""
        X = torch.tensor(sample_input_data_real[np.newaxis, :, :], dtype=torch.float32)
        
        logger.debug(f"Generating forecast...")
        
        output = pinn_model(X)
        
        # Denormalize
        scaler_y = load_real_data['scaler_y']
        
        q10_norm = output['q10'].detach().numpy()[0, :, 0]  # Temperature
        q50_norm = output['q50'].detach().numpy()[0, :, 0]
        q90_norm = output['q90'].detach().numpy()[0, :, 0]
        
        # Create dummy 3D arrays for denormalization
        q10_3d = np.column_stack([q10_norm, np.zeros(len(q10_norm)), np.zeros(len(q10_norm))])
        q50_3d = np.column_stack([q50_norm, np.zeros(len(q50_norm)), np.zeros(len(q50_norm))])
        q90_3d = np.column_stack([q90_norm, np.zeros(len(q90_norm)), np.zeros(len(q90_norm))])
        
        q10 = scaler_y.inverse_transform(q10_3d)[:, 0]
        q50 = scaler_y.inverse_transform(q50_3d)[:, 0]
        q90 = scaler_y.inverse_transform(q90_3d)[:, 0]
        
        # ============================================================
        # GET ACTUAL TEMPERATURE DATA (last 144 timesteps = 36h)
        # ============================================================
        
        df = load_real_data['df']
        actual_temps = df['room_temp'].values[-144:]  # Last 36h
        
        logger.debug(f"Actual temps range: {actual_temps.min():.1f} - {actual_temps.max():.1f}°C")
        logger.debug(f"Q50 range: {q50.min():.1f} - {q50.max():.1f}°C")
        
        # Generate time array (36h @ 15min)
        hours = np.arange(144) * 0.25
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add quantile bands
        fig.add_trace(go.Scatter(
            x=hours, y=q90,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,200,0)',
            showlegend=False,
            name='Q90',
        ))
        
        fig.add_trace(go.Scatter(
            x=hours, y=q10,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,200,0)',
            name='80% Confidence Interval',
            fillcolor='rgba(0,100,200,0.2)',
        ))
        
        # Add median forecast
        fig.add_trace(go.Scatter(
            x=hours, y=q50,
            mode='lines',
            name='Forecast (Q50)',
            line=dict(color='rgb(0,100,200)', width=2),
        ))
    
        # ← ADD ACTUAL DATA
        fig.add_trace(go.Scatter(
            x=hours, y=actual_temps,
            mode='lines+markers',
            name='Actual Temperature',
            line=dict(color='rgb(255,0,0)', width=2, dash='dash'),
            marker=dict(size=4),
        ))
        
        # Add comfort zone
        fig.add_hrect(y0=19.5, y1=22.0, 
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    annotation_text="Comfort Zone", 
                    annotation_position="right")
        
        fig.update_layout(
            title="PINN 36h Temperature Forecast vs Actual (REAL DATA - Untrained Model)",
            xaxis_title="Hours ahead",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            template="plotly_white",
            height=600,
            width=1200,
        )
        
        # Save plot
        output_path = Path(__file__).parent / "plots" / "forecast_real_data.html"
        output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Plot saved: {output_path}")
        logger.info(f"   Actual: {actual_temps.min():.1f} - {actual_temps.max():.1f}°C")
        logger.info(f"   Forecast Q50: {q50.min():.1f} - {q50.max():.1f}°C")
        logger.info(f"   Forecast Q10-Q90: {q10.min():.1f} - {q90.max():.1f}°C")
        
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])