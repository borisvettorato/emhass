"""
Thermal module for EMHASS - PINN forecasting and optimization
"""
import logging

logger = logging.getLogger(__name__)

try:
    # torch is the optional "thermal" extra (see pyproject.toml) - the rest
    # of this package (self_learning_physics, hybrid_heatpump_lr,
    # classical_models, feature_engineering, ...) is pure numpy/pandas/
    # sklearn and must stay importable without it. The Dockerfile's base
    # install (`uv pip install .`) does not pull in torch, so importing
    # ANY submodule of emhass.thermal - even a torch-free one - would
    # otherwise crash every deployment that didn't opt into `.[thermal]`.
    from .forecast_gridsearch import SearchOptions, run_gridsearch, run_two_phase_gridsearch
    from .pinn_forecaster import PInnThermalForecaster
    from .pinn_model import QuantileLoss, QuantilePhysicsInformedLSTM
    from .pinn_optimizer import HeatPumpOptimizer
except ImportError as e:
    logger.debug(
        "emhass.thermal: PINN features unavailable (%s) - install the 'thermal' "
        "extra (torch, scikit-learn) to enable them.", e,
    )
    QuantilePhysicsInformedLSTM = None
    QuantileLoss = None
    PInnThermalForecaster = None
    HeatPumpOptimizer = None
    SearchOptions = None
    run_gridsearch = None
    run_two_phase_gridsearch = None
from .classical_models import (  # noqa: E402 - torch-free, deliberately after the try/except above
    ModelRegistry,
    TrainResult,
    load_target_registries,
    save_target_registries,
    train_all_models,
    train_autoregressive_target_registries,
    train_single_model,
)
from .feature_engineering import (  # noqa: E402
    DEFAULT_SENSOR_MAP,
    FEATURE_SETS,
    MODEL_FEATURE_LEVEL,
    add_cyclic_time_features,
    add_interaction_features,
    add_lag_features,
    add_physics_features,
    add_rolling_features,
    add_solar_features,
    add_wind_features,
    build_feature_matrix,
    build_sequence_matrix,
    normalise_sensors,
    recommended_feature_level,
    time_based_split,
)
from .hybrid_heatpump_lr import (  # noqa: E402
    HybridHeatPumpLR,
    build_heatpump_features,
)
from .ml_integration import (  # noqa: E402
    build_classical_forecast_bundle,
    build_classical_optimization_plan,
    build_two_stage_optimization_plan,
)

__all__ = [
    "QuantilePhysicsInformedLSTM",
    "QuantileLoss",
    "PInnThermalForecaster",
    "HeatPumpOptimizer",
    "SearchOptions",
    "run_gridsearch",
    "run_two_phase_gridsearch",
    "build_feature_matrix",
    "build_sequence_matrix",
    "time_based_split",
    "recommended_feature_level",
    "normalise_sensors",
    "add_cyclic_time_features",
    "add_solar_features",
    "add_wind_features",
    "add_physics_features",
    "add_lag_features",
    "add_rolling_features",
    "add_interaction_features",
    "MODEL_FEATURE_LEVEL",
    "FEATURE_SETS",
    "DEFAULT_SENSOR_MAP",
    "train_all_models",
    "train_single_model",
    "train_autoregressive_target_registries",
    "save_target_registries",
    "load_target_registries",
    "ModelRegistry",
    "TrainResult",
    "build_classical_forecast_bundle",
    "build_classical_optimization_plan",
    "build_two_stage_optimization_plan",
    "HybridHeatPumpLR",
    "build_heatpump_features",
]
