"""
Add thermal prediction to optimization
Place in: src/emhass/

Integration with existing optimization.py
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from emhass_thermal_predictor import (
    setup_thermal_predictor,
    get_thermal_forecast,
    format_forecast_for_optimization,
    log_forecast_stats
)

logger = logging.getLogger(__name__)

class ThermalOptimizationMixin:
    """Add thermal forecasting to optimization"""
    
    def __init__(self, config: Dict):
        """Initialize with thermal predictor"""
        self.config = config
        self.thermal_predictor = None
        self.last_forecast = None
        
        # Setup thermal predictor if enabled
        if config.get('thermal_model', {}).get('enabled', False):
            self.setup_thermal_modeling()
    
    def setup_thermal_modeling(self):
        """Initialize thermal predictor"""
        logger.info("\n" + "="*80)
        logger.info("🔬 THERMAL MODELING SETUP")
        logger.info("="*80)
        
        thermal_config = self.config.get('thermal_model', {})
        emhass_path = self.config.get('emhass_path', '.')
        
        try:
            self.thermal_predictor = setup_thermal_predictor(
                thermal_config,
                emhass_path=emhass_path
            )
        except Exception as e:
            logger.error(f"Failed to setup thermal modeling: {e}")
            self.thermal_predictor = None
    
    def get_thermal_forecast(self, history_df: pd.DataFrame) -> Optional[Dict]:
        """
        Get 34-hour thermal forecast
        
        Args:
            history_df: Historical data from Home Assistant
        
        Returns:
            Forecast dictionary or None
        """
        if self.thermal_predictor is None:
            return None
        
        logger.info("\n📊 Querying thermal forecast...")
        
        forecast = get_thermal_forecast(self.thermal_predictor, history_df)
        
        if forecast.get('status') == 'success':
            log_forecast_stats(forecast)
            self.last_forecast = forecast
            return forecast
        else:
            logger.error(f"Thermal forecast failed: {forecast.get('error')}")
            return None
    
    def use_thermal_forecast_in_optimization(self, 
                                            day_ahead_prices: np.ndarray,
                                            comfort_min: float = 19.0,
                                            comfort_max: float = 22.0) -> Dict:
        """
        Use thermal forecast to optimize heat pump operation
        
        Args:
            day_ahead_prices: Electricity prices (€/kWh) for 34h
            comfort_min: Minimum comfortable temperature (°C)
            comfort_max: Maximum comfortable temperature (°C)
        
        Returns:
            Optimization recommendation
        """
        if self.last_forecast is None:
            logger.warning("No thermal forecast available")
            return {}
        
        forecast = self.last_forecast
        predictions = forecast.get('forecast', np.array([]))
        physics = forecast.get('physics_params', {})
        
        if len(predictions) == 0:
            return {}
        
        room_temps = predictions[:, 0]
        power_estimates = predictions[:, 1]
        
        logger.info("\n🔄 Optimizing heat pump using thermal predictions...\n")
        
        # Build optimization recommendations
        recommendations = {
            'forecast_room_temps': room_temps.tolist(),
            'forecast_power': power_estimates.tolist(),
            'comfort_range': [comfort_min, comfort_max],
            'predicted_violations': np.sum((room_temps < comfort_min) | (room_temps > comfort_max)),
            'physics_params': physics,
            'low_cost_hours': np.argsort(day_ahead_prices)[:8].tolist(),  # 8 cheapest hours
            'notes': [
                f"Thermal mass: {physics.get('thermal_mass', 0):.2f} (building inertia)",
                f"Conductance: {physics.get('conductance', 0):.4f} (heat loss rate)",
                f"Solar gain today: {physics.get('solar_gain', 0):.2f}",
            ]
        }
        
        logger.info(f"  ✓ Predicted comfort violations: {recommendations['predicted_violations']}")
        logger.info(f"  ✓ Cheapest hours: {recommendations['low_cost_hours']}")
        logger.info(f"  ✓ Using physics-informed optimization\n")
        
        return recommendations

# ============================================================================
# USAGE IN EXISTING OPTIMIZATION.PY
# ============================================================================

"""
In your existing Optimizer class, add:

    def __init__(self, ...):
        # Existing code...
        
        # Add thermal modeling
        if config.get('thermal_model', {}).get('enabled', False):
            self.thermal_predictor = setup_thermal_predictor(config)
    
    def run_optimization(self, ...):
        # Existing code...
        
        # Get thermal forecast
        if self.thermal_predictor:
            forecast = self.get_thermal_forecast(history_df)
            
            if forecast:
                # Use in optimization
                thermal_recs = self.use_thermal_forecast_in_optimization(
                    day_ahead_prices=prices,
                    comfort_min=19.0,
                    comfort_max=22.0
                )
                
                # Incorporate into optimization problem
                # e.g., prioritize cheap hours where forecast shows low heating need
"""