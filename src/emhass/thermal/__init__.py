"""
Thermal module for EMHASS - PINN forecasting and optimization
"""
from .pinn_model import QuantilePhysicsInformedLSTM, QuantileLoss
from .pinn_forecaster import PInnThermalForecaster
from .pinn_optimizer import HeatPumpOptimizer
from .forecast_gridsearch import SearchOptions, run_gridsearch, run_two_phase_gridsearch
from .feature_engineering import (
    build_feature_matrix,
    build_sequence_matrix,
    time_based_split,
    recommended_feature_level,
    normalise_sensors,
    add_cyclic_time_features,
    add_solar_features,
    add_wind_features,
    add_physics_features,
    add_lag_features,
    add_rolling_features,
    add_interaction_features,
    MODEL_FEATURE_LEVEL,
    FEATURE_SETS,
    DEFAULT_SENSOR_MAP,
)
from .classical_models import (
    train_all_models,
    train_single_model,
    train_autoregressive_target_registries,
    save_target_registries,
    load_target_registries,
    ModelRegistry,
    TrainResult,
)
from .ml_integration import (
    build_classical_forecast_bundle,
    build_classical_optimization_plan,
    build_two_stage_optimization_plan,
)
from .hybrid_heatpump_lr import (
    HybridHeatPumpLR,
    build_heatpump_features,
)

__all__ = [
    'QuantilePhysicsInformedLSTM',
    'QuantileLoss',
    'PInnThermalForecaster',
    'HeatPumpOptimizer',
    'SearchOptions',
    'run_gridsearch',
    'run_two_phase_gridsearch',
    'build_feature_matrix',
    'build_sequence_matrix',
    'time_based_split',
    'recommended_feature_level',
    'normalise_sensors',
    'add_cyclic_time_features',
    'add_solar_features',
    'add_wind_features',
    'add_physics_features',
    'add_lag_features',
    'add_rolling_features',
    'add_interaction_features',
    'MODEL_FEATURE_LEVEL',
    'FEATURE_SETS',
    'DEFAULT_SENSOR_MAP',
    'train_all_models',
    'train_single_model',
    'train_autoregressive_target_registries',
    'save_target_registries',
    'load_target_registries',
    'ModelRegistry',
    'TrainResult',
    'build_classical_forecast_bundle',
    'build_classical_optimization_plan',
    'build_two_stage_optimization_plan',
    'HybridHeatPumpLR',
    'build_heatpump_features',
]
