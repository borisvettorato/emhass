"""Thermal-mass physics core: a lumped RC-network model of indoor room temperature.

Main states:
- T_air: predicted indoor air temperature
- T_mass: hidden building/floor thermal mass temperature
- Q_emit: delayed heat-emitter state driven by duty * max(supply - T_air, 0)
- T_wall: hidden EXTERIOR-wall-surface temperature, sun-heated independent of
  blinds (see below) - a room's own thermal order this way ends up in the
  3R3C/4R4C family used in the building-grey-box literature (see e.g.
  Reynders et al. 2014, "impact of the model structure on the identified
  parameters and the predictive performance of RC building models" - our
  own topology is a "star" (T_air couples to both outdoor and T_mass
  directly, T_mass to T_air and now T_wall) rather than the more common
  "series" outdoor-mass-air chain, and couplings are independently-gained
  rather than reciprocal single-resistor values - a data-driven state-space
  model inspired by RC networks, not a strictly circuit-equivalent one).

Two distinct solar-gain pathways, split because they differ in BOTH timing
and controllability:
- Window-transmitted gain (existing "solar_*" params) - lands on interior
  surfaces/air, fed straight into T_air's own rate (no separate lag state -
  a room's light interior surfaces/furniture re-radiate to air quickly
  relative to T_mass/T_wall's own time constants) - and is the ONLY pathway
  gated by blind_position (0=open..1=closed), since interior blinds sit
  between the window and the room and can only ever block light that would
  otherwise pass THROUGH the window.
- Opaque-exterior-wall-absorbed gain (new "wall_*" params) - heats the
  building's own outer surface directly, completely unaffected by any
  interior shading device (blinds have no way to reach the outside of the
  building), and reaches the room only after conducting inward through the
  wall's own thermal mass (wall_tau_h) and partially dragging T_mass
  (wall_to_mass_weight) rather than T_air directly.

This module holds the simulation core (inputs container, fit-parameter schema,
the open-loop stepper) plus the fitting routine (_fit_temperature_params) so
both can be shared between the offline research script
(scripts/thermal_mass_physics_model.py) and the live EMHASS actions
(command_line.compute_heating_forecast, command_line.refit_heating_model).
CSV/report I/O and plotting stay in the script - this module has no
dependency on sklearn or any CLI/file-loading code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    from pvlib.irradiance import get_total_irradiance
    from pvlib.location import Location
except Exception:  # pragma: no cover - fallback for environments without pvlib
    Location = None
    get_total_irradiance = None


@dataclass
class ThermalInputs:
    index: pd.DatetimeIndex
    room: np.ndarray
    electric: np.ndarray
    gas: np.ndarray
    duty: np.ndarray
    supply: np.ndarray
    outdoor: np.ndarray
    wind_speed: np.ndarray
    wind_sin: np.ndarray
    wind_cos: np.ndarray
    q_solar: np.ndarray
    sun_alt_sin: np.ndarray
    sun_alt_cos: np.ndarray
    sun_az_sin: np.ndarray
    sun_az_cos: np.ndarray
    heatpump_duty: np.ndarray
    # Room's own live blind/shading position (0=open..1=closed) - only ever
    # gates the WINDOW solar pathway (see module docstring), never the
    # exterior-wall one. Defaults to None (resolved to all-zeros/open by
    # _simulate_open_loop) both when no sensor is configured (via
    # _prepare_inputs's own 0.0 default) AND for any caller outside this
    # module that still constructs ThermalInputs directly without knowing
    # about this field (e.g. scripts/cvxpy_state_space_thermal_model.py's
    # own, unrelated state-space model) - exactly recovering pre-blind-
    # support behavior either way.
    blind_position: np.ndarray | None = None


@dataclass
class SimResult:
    room: np.ndarray
    air_before: np.ndarray
    mass: np.ndarray
    q_emit: np.ndarray
    wall: np.ndarray
    q_solar: np.ndarray
    loss_coeff: np.ndarray


PARAM_NAMES = [
    "tau_emit_h",
    "emit_gain_per_h",
    "ua_base_per_h",
    "ua_wind_per_h_per_speed",
    "ua_wind_sin_per_h_per_speed",
    "ua_wind_cos_per_h_per_speed",
    "mass_tau_h",
    "mass_gain_per_h",
    "solar_gain_c_per_h",
    "solar_alt_sin_gain_c_per_h",
    "solar_alt_cos_gain_c_per_h",
    "solar_az_sin_gain_c_per_h",
    "solar_az_cos_gain_c_per_h",
    "bias_c_per_h",
    # Appended (not inserted) so _fit_temperature_params's own regularisation
    # array, which indexes the first 14 params by fixed position, needs no
    # changes. See module docstring for what these represent.
    "wall_tau_h",
    "wall_solar_gain_c",
    "wall_to_mass_weight",
]

LOWER_BOUNDS = np.array(
    [0.25, 0.0, 0.0, 0.0, -0.02, -0.02, 2.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -0.20, 0.25, 0.0, 0.0], dtype=float
)
UPPER_BOUNDS = np.array(
    [12.0, 0.8, 0.25, 0.03, 0.02, 0.02, 240.0, 0.40, 3.0, 2.0, 2.0, 2.0, 2.0, 0.20, 48.0, 30.0, 1.0], dtype=float
)
DEFAULT_X0 = np.array(
    [2.5, 0.08, 0.025, 0.0015, 0.0, 0.0, 48.0, 0.04, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 8.0, 0.3], dtype=float
)


def _infer_timestep_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.25
    diffs = index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float) / 3600.0
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.25
    return float(np.median(diffs))


def _series(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").ffill().fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tz is None:
        return dt_index.tz_localize("UTC")
    return dt_index.tz_convert("UTC")


def _compute_sun_direction_features(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return sin/cos decompositions of solar altitude and azimuth."""
    if Location is None:
        zeros = pd.Series(0.0, index=df.index, dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()

    try:
        location = Location(latitude=latitude, longitude=longitude, tz="UTC")
        solar_position = location.get_solarposition(_utc_index(df.index))
        altitude_rad = np.radians(90.0 - solar_position["apparent_zenith"].to_numpy(dtype=float))
        azimuth_rad = np.radians(solar_position["azimuth"].to_numpy(dtype=float))
        alt_sin = pd.Series(np.sin(altitude_rad), index=df.index).clip(lower=0.0)
        alt_cos = pd.Series(np.cos(altitude_rad), index=df.index)
        az_sin = pd.Series(np.sin(azimuth_rad), index=df.index)
        az_cos = pd.Series(np.cos(azimuth_rad), index=df.index)
        return alt_sin, alt_cos, az_sin, az_cos
    except Exception:
        zeros = pd.Series(0.0, index=df.index, dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()


def _compute_solar_gain_proxy(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    facade_azimuth_deg: float,
    facade_tilt_deg: float,
    horizontal_weight: float,
    facade_weight: float,
) -> pd.Series:
    ghi = _series(df, "ghi", 0.0).clip(lower=0.0)
    dni = _series(df, "dni", 0.0).clip(lower=0.0)
    dhi = _series(df, "dhi", 0.0).clip(lower=0.0)

    if Location is None or get_total_irradiance is None:
        return ((horizontal_weight + facade_weight) * ghi).clip(lower=0.0) / 1000.0

    try:
        location = Location(latitude=latitude, longitude=longitude, tz="UTC")
        solar_position = location.get_solarposition(_utc_index(df.index))
        poa = get_total_irradiance(
            surface_tilt=facade_tilt_deg,
            surface_azimuth=facade_azimuth_deg,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=dni,
            ghi=ghi,
            dhi=dhi,
        )["poa_global"].fillna(0.0).clip(lower=0.0)
        q_solar = horizontal_weight * ghi + facade_weight * poa
    except Exception:
        q_solar = (horizontal_weight + facade_weight) * ghi
    return q_solar.clip(lower=0.0) / 1000.0


def _prepare_inputs(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    facade_azimuth_deg: float,
    facade_tilt_deg: float,
    solar_horizontal_weight: float,
    solar_facade_weight: float,
) -> ThermalInputs:
    room = _series(df, "room_temp", 20.0).to_numpy(dtype=float)
    electric = _series(df, "electric_power", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    gas = _series(df, "gas_consumption", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    duty = _series(df, "heatpump_duty", 0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    supply_default = pd.Series(25.0, index=df.index, dtype=float)
    supply = pd.to_numeric(df.get("supply_temp", supply_default), errors="coerce").ffill().fillna(25.0)
    outdoor = _series(df, "outdoor_temp", 10.0).to_numpy(dtype=float)
    wind_speed = _series(df, "wind_speed", 0.0).clip(lower=0.0).to_numpy(dtype=float)
    wind_bearing = np.radians(_series(df, "wind_bearing", 0.0).to_numpy(dtype=float))
    q_solar = _compute_solar_gain_proxy(
        df,
        latitude=latitude,
        longitude=longitude,
        facade_azimuth_deg=facade_azimuth_deg,
        facade_tilt_deg=facade_tilt_deg,
        horizontal_weight=solar_horizontal_weight,
        facade_weight=solar_facade_weight,
    ).to_numpy(dtype=float)
    sun_alt_sin, sun_alt_cos, sun_az_sin, sun_az_cos = _compute_sun_direction_features(
        df,
        latitude=latitude,
        longitude=longitude,
    )
    blind_position = _series(df, "blind_position", 0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    return ThermalInputs(
        index=pd.DatetimeIndex(df.index),
        room=room,
        electric=electric,
        gas=gas,
        duty=duty,
        supply=supply.to_numpy(dtype=float),
        outdoor=outdoor,
        wind_speed=wind_speed,
        wind_sin=np.sin(wind_bearing),
        wind_cos=np.cos(wind_bearing),
        q_solar=q_solar,
        sun_alt_sin=sun_alt_sin.to_numpy(dtype=float),
        sun_alt_cos=sun_alt_cos.to_numpy(dtype=float),
        sun_az_sin=sun_az_sin.to_numpy(dtype=float),
        sun_az_cos=sun_az_cos.to_numpy(dtype=float),
        heatpump_duty=duty,
        blind_position=blind_position,
    )


def _simulate_open_loop(
    inputs: ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    initial_air: float,
    initial_mass: float | None = None,
    initial_q_emit: float = 0.0,
    initial_wall: float | None = None,
) -> SimResult:
    (
        tau_emit,
        emit_gain,
        ua_base,
        ua_wind,
        ua_wind_sin,
        ua_wind_cos,
        mass_tau,
        mass_gain,
        solar_gain,
        solar_alt_sin_gain,
        solar_alt_cos_gain,
        solar_az_sin_gain,
        solar_az_cos_gain,
        bias,
        wall_tau,
        wall_solar_gain,
        wall_to_mass_weight,
    ) = params
    n = len(inputs.room)
    # Callers outside this module that still construct ThermalInputs
    # directly (e.g. scripts/cvxpy_state_space_thermal_model.py's own,
    # unrelated state-space model) leave blind_position at its dataclass
    # default of None - resolve once here rather than per-step, exactly
    # recovering the pre-blind-support "always fully open" computation.
    blind_position = inputs.blind_position if inputs.blind_position is not None else np.zeros(n)
    pred_room = np.zeros(n, dtype=float)
    air_before = np.zeros(n, dtype=float)
    mass_series = np.zeros(n, dtype=float)
    q_emit_series = np.zeros(n, dtype=float)
    wall_series = np.zeros(n, dtype=float)
    loss_series = np.zeros(n, dtype=float)

    air = float(initial_air)
    mass = float(initial_air if initial_mass is None else initial_mass)
    q_emit = float(initial_q_emit)
    wall = float(initial_air if initial_wall is None else initial_wall)
    emit_alpha = float(np.clip(dt_h / max(tau_emit, 1e-6), 0.0, 1.0))
    mass_alpha = float(np.clip(dt_h / max(mass_tau, 1e-6), 0.0, 1.0))
    wall_alpha = float(np.clip(dt_h / max(wall_tau, 1e-6), 0.0, 1.0))

    for i in range(n):
        air_before[i] = air
        emit_raw = inputs.duty[i] * max(inputs.supply[i] - air, 0.0)
        q_emit = q_emit + emit_alpha * (emit_raw - q_emit)

        # Exterior wall surface: relaxes toward an outdoor-plus-solar-excess
        # equilibrium (a sun-exposed wall genuinely runs hotter than ambient
        # air) with its OWN time constant - never gated by blind_position,
        # since interior shading cannot affect what hits the outside of the
        # building (see module docstring). Updated before mass so mass's own
        # target below can pull toward the freshly-computed wall value this
        # same step, matching how q_emit/mass already feed d_air_dt within
        # the same iteration they're updated in.
        wall_target = inputs.outdoor[i] + wall_solar_gain * inputs.q_solar[i]
        wall = wall + wall_alpha * (wall_target - wall)

        # mass_target == air when wall_to_mass_weight == 0, so this is an
        # exact superset of the pre-wall behavior (mass = mass +
        # mass_alpha*(air-mass)) - backward compatible at weight=0.
        mass_target = air + wall_to_mass_weight * (wall - air)
        mass = mass + mass_alpha * (mass_target - mass)

        direction_loss = (
            ua_base
            + ua_wind * inputs.wind_speed[i]
            + ua_wind_sin * inputs.wind_speed[i] * inputs.wind_sin[i]
            + ua_wind_cos * inputs.wind_speed[i] * inputs.wind_cos[i]
        )
        loss_coeff = max(0.0, float(direction_loss))
        solar_direction_gain = (
            solar_gain
            + solar_alt_sin_gain * inputs.sun_alt_sin[i]
            + solar_alt_cos_gain * inputs.sun_alt_cos[i]
            + solar_az_sin_gain * inputs.sun_az_sin[i]
            + solar_az_cos_gain * inputs.sun_az_cos[i]
        )
        # Window-transmitted gain only - the ONLY solar pathway a closed
        # blind can block (see module docstring). blind_position defaults to
        # 0.0 (open) when unconfigured, so (1 - blind_position) == 1 exactly
        # recovers the pre-blind-support computation.
        window_solar_heat = (
            max(0.0, float(solar_direction_gain)) * inputs.q_solar[i] * (1.0 - blind_position[i])
        )
        d_air_dt = (
            emit_gain * q_emit
            + window_solar_heat
            - loss_coeff * (air - inputs.outdoor[i])
            + mass_gain * (mass - air)
            + bias
        )
        # Plain Python min/max, not np.clip: profiling a real 60-day refit
        # showed np.clip on a single scalar (called here once per timestep,
        # i.e. ~150k+ times over a realistic fit) pays numpy's full ufunc
        # dispatch overhead every call - ~35% of total fit wall-clock time
        # in that profile - for a 1-element operation plain Python handles
        # far more cheaply. Identical result for scalar input either way.
        air = min(35.0, max(5.0, air + dt_h * d_air_dt))
        pred_room[i] = air
        mass_series[i] = mass
        q_emit_series[i] = q_emit
        wall_series[i] = wall
        loss_series[i] = loss_coeff

    return SimResult(
        room=pred_room,
        air_before=air_before,
        mass=mass_series,
        q_emit=q_emit_series,
        wall=wall_series,
        q_solar=inputs.q_solar.copy(),
        loss_coeff=loss_series,
    )


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - true
    mse = float(np.mean(err**2))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def _simulate_segmented(
    inputs: ThermalInputs,
    params: np.ndarray,
    *,
    dt_h: float,
    segment_len: int,
) -> np.ndarray:
    """Vectorized across segments, not a per-segment Python loop calling
    _simulate_open_loop repeatedly (the original implementation, and still
    exactly what a single non-repeated call - e.g.
    command_line.compute_heating_forecast's own one-shot whole-horizon
    simulation, nothing to batch - should keep using directly).

    Every segment is independent by construction: each seeds its own
    T_air/T_mass/T_wall/Q_emit fresh from the room's own actual history at
    the segment's own start (see initial_* below) - NEVER chained from a
    previous segment's own simulated output. That independence is exactly
    what makes batching legal: instead of re-running the full
    segment_len-step Python loop from scratch for every one of
    (n_rows // segment_len) segments (thousands of scalar iterations for a
    real refit window), every segment's state is carried as one element of
    a length-(n_segments) array, and a SINGLE segment_len-step Python loop
    updates every segment at once via plain numpy elementwise ops -
    segment_len (a few dozen) iterations total, not the full row count.
    Profiling a real 60-day refit showed this (combined with the
    scalar-np.clip fix in _simulate_open_loop) cut _fit_temperature_params
    wall-clock time by roughly an order of magnitude; the underlying
    equations and their result are unchanged - see
    test_simulate_segmented_matches_manual_per_segment_loop for the
    numerical proof this is bit-for-bit consistent with the straightforward
    per-segment loop it replaces.

    Only full-length segments are batched this way; a final, shorter tail
    segment (when len(inputs.room) isn't an exact multiple of segment_len)
    is simulated the simple way via _simulate_open_loop directly - simpler
    and safer than padding a ragged batch, and it's at most one extra
    (cheap, O(segment_len)) call regardless of dataset size.
    """
    (
        tau_emit,
        emit_gain,
        ua_base,
        ua_wind,
        ua_wind_sin,
        ua_wind_cos,
        mass_tau,
        mass_gain,
        solar_gain,
        solar_alt_sin_gain,
        solar_alt_cos_gain,
        solar_az_sin_gain,
        solar_az_cos_gain,
        bias,
        wall_tau,
        wall_solar_gain,
        wall_to_mass_weight,
    ) = params

    n = len(inputs.room)
    pred = np.zeros(n, dtype=float)
    n_full_segments = n // segment_len if segment_len > 0 else 0
    n_batched = n_full_segments * segment_len
    blind_position = inputs.blind_position if inputs.blind_position is not None else np.zeros(n)

    if n_full_segments > 0:
        seg_starts = np.arange(n_full_segments) * segment_len
        prev_idx = np.maximum(0, seg_starts - 1)
        air = inputs.room[prev_idx].astype(float)
        mass = air.copy()
        wall = air.copy()
        q_emit = inputs.duty[prev_idx] * np.maximum(inputs.supply[prev_idx] - air, 0.0)

        emit_alpha = min(1.0, max(0.0, dt_h / max(tau_emit, 1e-6)))
        mass_alpha = min(1.0, max(0.0, dt_h / max(mass_tau, 1e-6)))
        wall_alpha = min(1.0, max(0.0, dt_h / max(wall_tau, 1e-6)))

        def _batch(arr: np.ndarray) -> np.ndarray:
            return arr[:n_batched].reshape(n_full_segments, segment_len)

        duty_b = _batch(inputs.duty)
        supply_b = _batch(inputs.supply)
        outdoor_b = _batch(inputs.outdoor)
        wind_speed_b = _batch(inputs.wind_speed)
        wind_sin_b = _batch(inputs.wind_sin)
        wind_cos_b = _batch(inputs.wind_cos)
        q_solar_b = _batch(inputs.q_solar)
        sun_alt_sin_b = _batch(inputs.sun_alt_sin)
        sun_alt_cos_b = _batch(inputs.sun_alt_cos)
        sun_az_sin_b = _batch(inputs.sun_az_sin)
        sun_az_cos_b = _batch(inputs.sun_az_cos)
        blind_b = _batch(blind_position)

        pred_batch = np.zeros((n_full_segments, segment_len), dtype=float)
        for t in range(segment_len):
            emit_raw = duty_b[:, t] * np.maximum(supply_b[:, t] - air, 0.0)
            q_emit = q_emit + emit_alpha * (emit_raw - q_emit)

            wall_target = outdoor_b[:, t] + wall_solar_gain * q_solar_b[:, t]
            wall = wall + wall_alpha * (wall_target - wall)

            mass_target = air + wall_to_mass_weight * (wall - air)
            mass = mass + mass_alpha * (mass_target - mass)

            direction_loss = (
                ua_base
                + ua_wind * wind_speed_b[:, t]
                + ua_wind_sin * wind_speed_b[:, t] * wind_sin_b[:, t]
                + ua_wind_cos * wind_speed_b[:, t] * wind_cos_b[:, t]
            )
            loss_coeff = np.maximum(0.0, direction_loss)
            solar_direction_gain = (
                solar_gain
                + solar_alt_sin_gain * sun_alt_sin_b[:, t]
                + solar_alt_cos_gain * sun_alt_cos_b[:, t]
                + solar_az_sin_gain * sun_az_sin_b[:, t]
                + solar_az_cos_gain * sun_az_cos_b[:, t]
            )
            window_solar_heat = (
                np.maximum(0.0, solar_direction_gain) * q_solar_b[:, t] * (1.0 - blind_b[:, t])
            )
            d_air_dt = (
                emit_gain * q_emit
                + window_solar_heat
                - loss_coeff * (air - outdoor_b[:, t])
                + mass_gain * (mass - air)
                + bias
            )
            air = np.clip(air + dt_h * d_air_dt, 5.0, 35.0)
            pred_batch[:, t] = air

        pred[:n_batched] = pred_batch.reshape(-1)

    if n_batched < n:
        start = n_batched
        sub = ThermalInputs(
            index=inputs.index[start:n],
            room=inputs.room[start:n],
            electric=inputs.electric[start:n],
            gas=inputs.gas[start:n],
            duty=inputs.duty[start:n],
            supply=inputs.supply[start:n],
            outdoor=inputs.outdoor[start:n],
            wind_speed=inputs.wind_speed[start:n],
            wind_sin=inputs.wind_sin[start:n],
            wind_cos=inputs.wind_cos[start:n],
            q_solar=inputs.q_solar[start:n],
            sun_alt_sin=inputs.sun_alt_sin[start:n],
            sun_alt_cos=inputs.sun_alt_cos[start:n],
            sun_az_sin=inputs.sun_az_sin[start:n],
            sun_az_cos=inputs.sun_az_cos[start:n],
            heatpump_duty=inputs.heatpump_duty[start:n],
            blind_position=blind_position[start:n],
        )
        initial_air = float(inputs.room[max(0, start - 1)])
        initial_q_emit = float(
            inputs.duty[max(0, start - 1)] * max(inputs.supply[max(0, start - 1)] - initial_air, 0.0)
        )
        sim = _simulate_open_loop(
            sub,
            params,
            dt_h=dt_h,
            initial_air=initial_air,
            initial_mass=initial_air,
            initial_q_emit=initial_q_emit,
            initial_wall=initial_air,
        )
        pred[start:n] = sim.room

    return pred


def _fit_temperature_params(
    inputs: ThermalInputs,
    *,
    dt_h: float,
    segment_len: int,
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Fit the 17 physics parameters against ``inputs.room`` via segmented
    open-loop least-squares (3 restarts, best-of picked by fit MAE)."""
    finite = np.isfinite(inputs.room)

    def residuals(params: np.ndarray) -> np.ndarray:
        pred = _simulate_segmented(inputs, params, dt_h=dt_h, segment_len=segment_len)
        res = pred[finite] - inputs.room[finite]
        # Light regularisation keeps weakly identified terms from doing wild things.
        reg = np.array(
            [
                (params[4] / 0.02) * 0.03,
                (params[5] / 0.02) * 0.03,
                (params[13] / 0.20) * 0.03,
                *((params[9:13] / 2.0) * 0.03),
                # wall_to_mass_weight (index 16): the newest, least-grounded
                # cross-term - regularised toward 0 (== "mass ignores wall,
                # exactly today's behavior") for the same reason the
                # directional solar cross-terms above are, and unlike
                # wall_tau_h/wall_solar_gain_c which behave like the
                # existing (unregularised) mass_tau_h/solar_gain_c_per_h.
                (params[16] / 1.0) * 0.03,
            ],
            dtype=float,
        )
        return np.concatenate([res, reg])

    x0_fast = DEFAULT_X0.copy()
    x0_fast[:9] = np.array([1.0, 0.12, 0.035, 0.001, 0.0, 0.0, 24.0, 0.06, 0.25], dtype=float)
    x0_slow = DEFAULT_X0.copy()
    x0_slow[:9] = np.array([5.0, 0.04, 0.018, 0.0025, 0.0, 0.0, 96.0, 0.025, 0.55], dtype=float)
    starts = [DEFAULT_X0, x0_fast, x0_slow]

    best = None
    for x0 in starts:
        result = least_squares(
            residuals,
            x0=np.clip(x0, LOWER_BOUNDS, UPPER_BOUNDS),
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            loss="soft_l1",
            f_scale=0.30,
            max_nfev=max_nfev,
            verbose=0,
        )
        score = float(np.mean(np.abs(residuals(result.x)[: int(finite.sum())])))
        if best is None or score < best[0]:
            best = (score, result)

    assert best is not None
    result = best[1]
    return result.x, {
        "fit_mae_c": float(best[0]),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "success": bool(result.success),
        "status": int(result.status),
    }
