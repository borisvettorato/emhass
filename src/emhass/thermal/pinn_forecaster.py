"""
PINN Thermal Forecaster wrapper for EMHASS
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from .feature_engineering import build_feature_matrix, LSTM_WEATHER_CONTROL_FEATURES
from .pinn_model import QuantilePhysicsInformedLSTM, QuantileLoss

logger = logging.getLogger(__name__)


class PInnThermalForecaster:
    """
    Wrapper around PINN model for EMHASS integration
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: int = 37,
        hidden: int = 128,
        lookahead: int = 144,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            input_size: Number of input features (37 for thermal model)
            hidden: LSTM hidden size
            lookahead: Forecast horizon (144 = 36h @ 15min)
            device: cuda or cpu
        """
        self.device = device
        self.lookahead = lookahead
        self.input_size = input_size
        self.state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.last_timestamp: pd.Timestamp | None = None
        
        # Initialize model
        self.model = QuantilePhysicsInformedLSTM(
            input_size=input_size,
            hidden=hidden,
            lookahead=lookahead,
        ).to(device)
        
        # Load checkpoint if provided
        if model_path and Path(model_path).exists():
            self._load_checkpoint(model_path)
        else:
            logger.warning("No model checkpoint provided. Initialize with random weights.")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load saved model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            # Support both old ('columns') and new ('feature_cols') checkpoint format
            self.feature_cols: list[str] = checkpoint.get('feature_cols') or checkpoint.get('columns') or []
            self.target_cols: list[str] = checkpoint.get('target_cols', [])
            self.lookback: int = int(checkpoint.get('lookback', self.lookahead))
            self.feature_level: str = str(checkpoint.get('feature_level', 'standard'))
            self.latitude: float = float(checkpoint.get('latitude', 52.1202))
            self.longitude: float = float(checkpoint.get('longitude', 4.4899))
            self.facade_azimuth_deg: float | None = checkpoint.get('facade_azimuth_deg')
            # Always reset recurrent state after loading model weights.
            self.reset_state()
            logger.info("Loaded checkpoint: %s  features=%d", checkpoint_path, len(self.feature_cols))
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            raise

    def prepare_input_from_df(
        self,
        df: pd.DataFrame,
        lookback: int | None = None,
    ) -> np.ndarray:
        """Build a scaled (lookback, n_features) array from a raw sensor DataFrame.

        Uses the feature list, coordinates and feature level stored in the checkpoint
        so that training and inference are guaranteed to use identical features.

        Args:
            df: Raw sensor DataFrame with a DatetimeIndex.
            lookback: Override window length (defaults to checkpoint ``lookback``).

        Returns:
            Numpy array of shape *(lookback, n_input_features)* ready for :meth:`forecast`.
        """
        n = int(lookback or getattr(self, "lookback", self.lookahead))
        feature_level = getattr(self, "feature_level", "standard")
        latitude = getattr(self, "latitude", 52.1202)
        longitude = getattr(self, "longitude", 4.4899)
        facade_azimuth_deg = getattr(self, "facade_azimuth_deg", None)
        feature_cols = getattr(self, "feature_cols", []) or LSTM_WEATHER_CONTROL_FEATURES
        primary_target = (getattr(self, "target_cols", []) or ["room_temp"])[0]

        feature_df, _ = build_feature_matrix(
            df,
            feature_level=feature_level,
            latitude=latitude,
            longitude=longitude,
            facade_azimuth_deg=facade_azimuth_deg,
            target_col=primary_target,
            drop_na=False,
            include_feature_cols=feature_cols,
        )

        # Ensure every expected column is present
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0

        X_raw = feature_df[feature_cols].fillna(0.0).tail(n).values
        if len(X_raw) != n:
            raise RuntimeError(
                f"prepare_input_from_df: need {n} rows, got {len(X_raw)} after feature engineering"
            )
        return self.scaler_X.transform(X_raw)

    def reset_state(self) -> None:
        """Reset the stored LSTM state for a new sequence/stream."""
        self.state = None
        self.last_timestamp = None

    @staticmethod
    def _to_timestamp(ts: pd.Timestamp | str | None) -> pd.Timestamp | None:
        if ts is None:
            return None
        return pd.Timestamp(ts)

    def maybe_reset_on_gap(
        self,
        current_timestamp: pd.Timestamp | str | None,
        max_gap_minutes: float,
    ) -> bool:
        """Reset state when incoming timestamps are non-monotonic or too far apart."""
        ts = self._to_timestamp(current_timestamp)
        if ts is None:
            return False
        if self.last_timestamp is None:
            self.last_timestamp = ts
            return False

        gap_minutes = (ts - self.last_timestamp).total_seconds() / 60.0
        should_reset = gap_minutes < 0 or gap_minutes > float(max_gap_minutes)
        if should_reset:
            self.reset_state()
        self.last_timestamp = ts
        return should_reset
    
    def forecast(
        self,
        X: np.ndarray,
        return_quantiles: bool = True,
        stateful: bool = False,
        reset_state: bool = False,
        current_timestamp: pd.Timestamp | str | None = None,
        max_gap_minutes: float | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate forecast
        
        Args:
            X: Input data (192, input_size) = 48h @ 15min, normalized
            return_quantiles: Return q10/q50/q90 or only q50
            stateful: Reuse hidden state between consecutive calls
            reset_state: Force a state reset before this forecast
            current_timestamp: Optional timestamp used for gap-aware resets
            max_gap_minutes: Reset when timestamp gap exceeds this value
        
        Returns:
            dict with forecasts (denormalized)
        """
        self.model.eval()
        if reset_state:
            self.reset_state()
        if stateful and max_gap_minutes is not None:
            self.maybe_reset_on_gap(current_timestamp, max_gap_minutes)
        elif current_timestamp is not None:
            self.last_timestamp = self._to_timestamp(current_timestamp)
        
        with torch.no_grad():
            X_tensor = torch.tensor(X[np.newaxis, :, :], dtype=torch.float32).to(self.device)
            output = self.model(X_tensor, state=self.state if stateful else None)

        if stateful and output.get('state_out') is not None:
            h, c = output['state_out']
            self.state = (h.detach(), c.detach())
        
        # Denormalize
        result = {}
        
        if return_quantiles:
            q10 = self._denormalize(output['q10'].cpu().numpy()[0, :, :])
            q50 = self._denormalize(output['q50'].cpu().numpy()[0, :, :])
            q90 = self._denormalize(output['q90'].cpu().numpy()[0, :, :])
            
            result['q10'] = q10  # (144, 3)
            result['q50'] = q50
            result['q90'] = q90
        else:
            q50 = self._denormalize(output['q50'].cpu().numpy()[0, :, :])
            result['q50'] = q50
        
        # Physics parameters
        physics = output['physics_params'].cpu().numpy()[0]
        result['physics_params'] = {
            'C': float(physics[0]),
            'R': float(physics[1]),
            'alpha_solar': float(physics[2]),
            'eta_hp': float(physics[3]),
            'lambda_inf': float(physics[4]),
        }
        
        return result

    def warmup(
        self,
        X: np.ndarray,
        current_timestamp: pd.Timestamp | str | None = None,
        max_gap_minutes: float | None = None,
        reset_state: bool = False,
    ) -> None:
        """Run a forward pass that only updates recurrent state (no forecast output)."""
        self.forecast(
            X,
            return_quantiles=False,
            stateful=True,
            reset_state=reset_state,
            current_timestamp=current_timestamp,
            max_gap_minutes=max_gap_minutes,
        )
    
    def _denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize prediction"""
        return self.scaler_y.inverse_transform(X_norm)
    
    def get_temperature_forecast(
        self,
        X: np.ndarray,
        quantile: str = 'q50',
        stateful: bool = False,
        reset_state: bool = False,
        current_timestamp: pd.Timestamp | str | None = None,
        max_gap_minutes: float | None = None,
    ) -> np.ndarray:
        """
        Get temperature forecast only (index 0 of 3 targets)
        
        Args:
            X: Input data
            quantile: 'q10', 'q50', or 'q90'
            stateful: Reuse hidden state between consecutive calls
            reset_state: Force a state reset before this forecast
            current_timestamp: Optional timestamp used for gap-aware resets
            max_gap_minutes: Reset when timestamp gap exceeds this value
        
        Returns:
            Temperature forecast (144,) for 36 hours
        """
        forecast = self.forecast(
            X,
            return_quantiles=True,
            stateful=stateful,
            reset_state=reset_state,
            current_timestamp=current_timestamp,
            max_gap_minutes=max_gap_minutes,
        )
        return forecast[quantile][:, 0]  # Temperature is index 0