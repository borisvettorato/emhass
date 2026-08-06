import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch


logger = logging.getLogger(__name__)


def test_optimization_with_real_forecast(pinn_model, sample_input_data_real, load_real_data, optimizer):
    """Test optimization with REAL forecast"""
    
    # Get forecast
    X = torch.tensor(sample_input_data_real[np.newaxis, :, :], dtype=torch.float32)
    output = pinn_model(X)
    
    scaler_y = load_real_data['scaler_y']
    
    # Denormalize temperature forecast (q50)
    q50_norm = output['q50'].detach().numpy()[0, :, 0]
    q50_3d = np.column_stack([q50_norm, np.zeros(144), np.zeros(144)])
    room_temps = scaler_y.inverse_transform(q50_3d)[:, 0]
    
    # Simulate outdoor temps (simplified: use last value + sine variation)
    hours = np.arange(144) * 0.25
    outdoor_temps = 10 + 5 * np.sin(2 * np.pi * hours / 24)
    
    # Optimize
    result = optimizer.get_optimal_setpoint(room_temps, outdoor_temps)
    
    # Plot
    hours_array = np.arange(144) * 0.25
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours_array, y=result['baseline_curve'],
        mode='lines',
        name='HP Baseline Curve',
        line=dict(color='green', dash='dash', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=hours_array, y=result['setpoint_optimal'],
        mode='lines',
        name='Optimal Setpoint',
        line=dict(color='red', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=hours_array, y=room_temps,
        mode='lines',
        name='Room Temp Forecast',
        line=dict(color='orange', width=2),
    ))
    
    fig.update_layout(
        title="Heat Pump Optimization (REAL DATA)",
        xaxis_title="Hours ahead",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        template="plotly_white",
        height=600,
        width=1200,
    )
    
    output_path = Path(__file__).parent / "plots" / "optimization_real_data.html"
    output_path.parent.mkdir(exist_ok=True)
    fig.write_html(str(output_path))
    
    logger.info(f"✅ Optimization plot saved: {output_path}")
    assert output_path.exists()