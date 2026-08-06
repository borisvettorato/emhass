"""
Heat pump optimization with PINN forecast
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HeatPumpOptimizer:
    """
    Optimize heat pump setpoints given:
    - PINN temperature forecast
    - Heat pump curve: y = -x + 20 (outdoor_temp → supply_temp)
    - Constraints: max +10°C from curve, heating only (no cooling)
    """
    
    def __init__(
        self,
        curve_intercept: float = 40.0,
        curve_slope: float = -1.0,
        max_deviation: float = 10.0,
        min_supply_temp: float = 10.0,
        max_supply_temp: float = 60.0,
        timestep_hours: float = 0.25,
        thermal_demand_coeff_kw_per_c: float = 0.35,
        cv_pump_power_kw: float = 0.08,
        gas_boiler_efficiency: float = 0.90,
    ):
        """
        Args:
            curve_intercept: Heat pump curve y-intercept (40.0)
            curve_slope: Heat pump curve slope (-1.0)
            max_deviation: Max allowed deviation from curve (+10.0)
            min_supply_temp: Minimum supply temperature (10°C)
            max_supply_temp: Maximum supply temperature (60°C)
            timestep_hours: Time resolution in hours (default 15 min)
            thermal_demand_coeff_kw_per_c: Thermal demand gain per °C lift
            cv_pump_power_kw: Electrical pump power draw when heating is active
            gas_boiler_efficiency: Boiler efficiency used for gas estimate
        """
        self.curve_intercept = curve_intercept
        self.curve_slope = curve_slope
        self.max_deviation = max_deviation
        self.min_supply_temp = min_supply_temp
        self.max_supply_temp = max_supply_temp
        self.timestep_hours = timestep_hours
        self.thermal_demand_coeff_kw_per_c = thermal_demand_coeff_kw_per_c
        self.cv_pump_power_kw = cv_pump_power_kw
        self.gas_boiler_efficiency = gas_boiler_efficiency
        
        logger.info(
            f"✅ HeatPumpOptimizer initialized:\n"
            f"   Curve: y = {curve_slope}*x + {curve_intercept}\n"
            f"   Max deviation: ±{max_deviation}°C\n"
            f"   Supply temp range: {min_supply_temp}-{max_supply_temp}°C"
        )
    
    def get_baseline_curve(self, outdoor_temps: np.ndarray) -> np.ndarray:
        """
        Calculate baseline heat pump curve
        y = -x + 40
        
        Args:
            outdoor_temps: Array of outdoor temperatures
        
        Returns:
            Supply temperatures from curve
        """
        return self.curve_slope * outdoor_temps + self.curve_intercept
    
    def get_optimal_setpoint(
        self,
        room_temp_forecast: np.ndarray,
        outdoor_temp_forecast: np.ndarray,
        target_room_temp_min: float = 19.5,
        target_room_temp_max: float = 22.0,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate optimal supply temperature setpoint
        
        Constraints:
        1. Room temp should stay in [target_min, target_max]
        2. Supply temp must be within ±max_deviation from curve
        3. No cooling (supply_temp <= outdoor_temp + safety_margin)
        4. Supply temp in valid range
        
        Args:
            room_temp_forecast: Predicted room temperatures (144,)
            outdoor_temp_forecast: Outdoor temps (144,)
            target_room_temp_min: Comfort min (19.5°C)
            target_room_temp_max: Comfort max (22.0°C)
        
        Returns:
            dict with setpoints, constraints, status
        """
        n_steps = len(room_temp_forecast)
        
        # Baseline curve
        baseline = self.get_baseline_curve(outdoor_temp_forecast)
        
        # Start from strict weather-curve envelope (±max_deviation)
        lower = baseline - self.max_deviation
        upper = baseline + self.max_deviation

        # Clamp to valid supply range first.
        lower = np.clip(lower, self.min_supply_temp, self.max_supply_temp)
        upper = np.clip(upper, self.min_supply_temp, self.max_supply_temp)

        # Enforce heating-only lower bound while preserving strict upper bound.
        # If infeasible (very warm outdoor), collapse interval to upper.
        heating_floor = outdoor_temp_forecast + 2.0
        setpoint_min = np.maximum(lower, heating_floor)
        setpoint_max = upper.copy()
        infeasible = setpoint_min > setpoint_max
        setpoint_min[infeasible] = setpoint_max[infeasible]

        # Comfort-driven control target:
        # - TOO_COLD  -> near upper bound
        # - TOO_HOT   -> near lower bound
        # - COMFORT   -> around baseline clipped to feasible range
        setpoint_optimal = np.empty(n_steps, dtype=float)
        
        # Check comfort satisfaction
        comfort_status = np.empty(n_steps, dtype=object)
        comfort_mid = 0.5 * (target_room_temp_min + target_room_temp_max)
        comfort_span = max(1e-6, target_room_temp_max - target_room_temp_min)

        for i in range(n_steps):
            room_t = room_temp_forecast[i]
            if room_t >= target_room_temp_min and room_t <= target_room_temp_max:
                comfort_status[i] = "COMFORTABLE"
                # Small centered correction inside comfort band.
                # Negative -> slightly warmer setpoint, positive -> slightly cooler.
                norm = np.clip((comfort_mid - room_t) / comfort_span, -0.5, 0.5)
                candidate = baseline[i] + norm * (setpoint_max[i] - setpoint_min[i])
                setpoint_optimal[i] = np.clip(candidate, setpoint_min[i], setpoint_max[i])
            elif room_t < target_room_temp_min:
                comfort_status[i] = "TOO_COLD"
                setpoint_optimal[i] = setpoint_max[i]
            else:
                comfort_status[i] = "TOO_HOT"
                setpoint_optimal[i] = setpoint_min[i]

        # Estimated boiler energy proxies per timestep.
        # Gas demand proxy grows with supply-outdoor lift and is converted by efficiency.
        lift = np.maximum(0.0, setpoint_optimal - outdoor_temp_forecast)
        thermal_kw = self.thermal_demand_coeff_kw_per_c * lift
        gas_kwh_est = (thermal_kw / max(self.gas_boiler_efficiency, 1e-6)) * self.timestep_hours
        elec_kwh_est = np.where(thermal_kw > 1e-9, self.cv_pump_power_kw * self.timestep_hours, 0.0)
        
        return {
            'setpoint_optimal': setpoint_optimal,
            'setpoint_min': setpoint_min,
            'setpoint_max': setpoint_max,
            'baseline_curve': baseline,
            'outdoor_temp': outdoor_temp_forecast,
            'room_temp_forecast': room_temp_forecast,
            'comfort_status': comfort_status,
            'target_min': target_room_temp_min,
            'target_max': target_room_temp_max,
            'cv_estimated_gas_kwh': gas_kwh_est,
            'cv_estimated_electricity_kwh': elec_kwh_est,
        }
    
    def get_constraint_violations(
        self,
        setpoint: np.ndarray,
        outdoor_temp: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Check for constraint violations
        """
        baseline = self.get_baseline_curve(outdoor_temp)
        deviation = setpoint - baseline
        
        violations = {
            'deviation_exceed': np.abs(deviation) > self.max_deviation,
            'too_high': setpoint > self.max_supply_temp,
            'too_low': setpoint < self.min_supply_temp,
            'cooling': setpoint <= outdoor_temp,  # Heating only
        }
        
        return violations

    def get_price_aware_setpoint(
        self,
        room_temp_forecast: np.ndarray,
        outdoor_temp_forecast: np.ndarray,
        price_forecast: np.ndarray,
        target_room_temp_min: float = 19.5,
        target_room_temp_max: float = 22.0,
        price_weight: float = 0.8,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate a price-aware supply temperature setpoint profile.

        This method builds on top of get_optimal_setpoint bounds and shifts
        setpoints lower when price is high and higher when price is low.
        """
        base = self.get_optimal_setpoint(
            room_temp_forecast=room_temp_forecast,
            outdoor_temp_forecast=outdoor_temp_forecast,
            target_room_temp_min=target_room_temp_min,
            target_room_temp_max=target_room_temp_max,
        )

        n_steps = len(room_temp_forecast)
        if len(price_forecast) != n_steps:
            raise ValueError(
                "price_forecast must have the same length as room_temp_forecast"
            )

        prices = np.asarray(price_forecast, dtype=float)
        prices = np.nan_to_num(prices, nan=np.nanmedian(prices) if np.any(~np.isnan(prices)) else 0.0)

        price_min = float(np.min(prices))
        price_max = float(np.max(prices))
        if price_max - price_min < 1e-9:
            normalized_price = np.zeros_like(prices)
        else:
            normalized_price = (prices - price_min) / (price_max - price_min)

        # Reallocate the same heating effort to cheaper hours first.
        # This follows the rearrangement inequality and tends to minimize
        # sum(price * setpoint) for positive prices.
        neutral_setpoint = np.asarray(base['setpoint_optimal'], dtype=float)
        sorted_prices_idx = np.argsort(prices)  # cheapest first
        sorted_setpoints_desc = np.sort(neutral_setpoint)[::-1]  # highest first

        economic_setpoint = np.empty_like(neutral_setpoint)
        economic_setpoint[sorted_prices_idx] = sorted_setpoints_desc

        # Blend with neutral profile to keep behavior smoother and less aggressive.
        setpoint_price_aware = (
            price_weight * economic_setpoint + (1.0 - price_weight) * neutral_setpoint
        )
        setpoint_price_aware = np.clip(
            setpoint_price_aware,
            base['setpoint_min'],
            base['setpoint_max'],
        )

        result = dict(base)
        result['setpoint_price_aware'] = setpoint_price_aware
        result['price_forecast'] = prices
        result['price_normalized'] = normalized_price
        result['price_weight'] = price_weight
        result['cost_proxy'] = setpoint_price_aware * prices

        return result