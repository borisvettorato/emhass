import bz2
import copy
import logging
import os
import pickle
import time
from math import ceil, isfinite

import cvxpy as cp
import numpy as np
import pandas as pd

from emhass import utils

# Keys the thermal model actually reads from a load's thermal_config (issue #943).
# Any other key is silently ignored, so a typo such as the singular
# `min_temperature` (the model reads the list key `min_temperatures`) yields a
# load that never schedules, with no feedback; we warn on unrecognized keys.
THERMAL_CONFIG_KNOWN_KEYS = frozenset(
    {
        "heating_rate",
        "cooling_constant",
        "start_temperature",
        "min_temperatures",
        "max_temperatures",
        "desired_temperatures",
        "overshoot_temperature",
        "penalty_factor",
        "sense",
    }
)
# Common singular typo -> (correct list key, what that key controls). The role
# tailors the guidance so the hint for `desired_temperature` talks about the soft
# target rather than the hard min/max comfort band.
THERMAL_CONFIG_KEY_HINTS = {
    "min_temperature": ("min_temperatures", "the hard comfort band"),
    "max_temperature": ("max_temperatures", "the hard comfort band"),
    "target_temperature": ("min_temperatures/max_temperatures", "the hard comfort band"),
    "desired_temperature": (
        "desired_temperatures",
        "the soft target used with overshoot_temperature",
    ),
}

# Tie-break weight for PV curtailment timing (issue #342): must be far below
# real tariff coefficients (~1e-4 $/W per step) yet large enough for the LP to
# act on (above HiGHS dual feasibility tolerance once multiplied by realistic
# curtailment powers in W).
CURTAILMENT_TIEBREAK_EPS = 1e-7

# Multi-battery symmetry-breaking tie-break (issue #610): with N>1 batteries of
# identical (or near-identical) cost/efficiency, the LP has many equally-optimal
# ways to split charge/discharge across batteries. A tiny index-scaled usage
# tilt breaks the tie at the LP/MILP optimum: it penalizes total throughput
# (charge + discharge combined) scaled by battery index, so on an exact tie
# the lowest-index battery is preferred for both charging and discharging.
# It never overrides a real cost/efficiency difference.
#
# Sizing: the smallest realistic difference this must stay dominated by is a
# 0.1% round-trip-efficiency delta between two otherwise-identical batteries
# (see test_epsilon_dominance in test_multi_battery_optimization.py) at the
# cheapest realistic tariff; 1e-9 is ~5 orders of magnitude below that.
#
# This only guarantees a unique mathematical optimum, not that the solver
# reports it: HiGHS is a MILP solver and stops once it is within
# lp_solver_mip_rel_gap of that optimum (default 0.01), several orders larger
# than this tilt's own contribution to the objective. Within that gap the
# solver is free to return any plan it likes, so strict run-to-run
# determinism needs a tight (or zero) lp_solver_mip_rel_gap - the same knob
# that governs general schedule repeatability, not something specific to
# multi-battery.
BATTERY_TIEBREAK_EPS = 1e-9

# Battery-first priority (issue #834/#1002): when set_battery_first_priority is
# on, importing from the grid while the battery is still above its minimum SoC is
# penalized at this multiple of the prevailing import tariff. Making it a soft
# penalty rather than a hard constraint means the optimizer still prefers to drain
# the battery before importing (the penalty dwarfs any realistic tariff gradient,
# so drain-first wins at any currency/price scale) but can always fall back to
# importing when that is the only feasible option (e.g. recharging to a terminal
# SoC target with no PV), instead of returning infeasible. The gate confines the
# penalty to genuinely avoidable import, so an aggressive factor is safe.
BATTERY_FIRST_IMPORT_PENALTY_FACTOR = 100.0

# Soft terminal-SoC target, the same treatment #1002 gave set_battery_first_priority.
# A hard equality on the horizon's net energy change turns the solve infeasible
# whenever the requested soc_final simply cannot be reached. That collides with
# set_nodischarge_to_grid on AC-coupled systems: when PV already covers the load a
# large SoC shed has no local deficit to discharge into, and export is (correctly)
# closed off, so no schedule exists (#936 vs #795). Enforcing the target through
# non-negative slacks priced far above any realistic tariff keeps it met exactly
# whenever that is possible, while a contradictory target relaxes to the closest
# reachable SoC instead of returning infeasible.
SOC_FINAL_DEVIATION_PENALTY_FACTOR = 100.0

# Open window/door thermal effects: live-only, current-moment signals (see
# room_opening_open/room_door_open threading), so these constants only ever
# apply at the near-term timestep of a solve - never held flat across a
# forecast horizon the way blind_position is. Fixed, non-configurable values,
# mirroring the sun-shading feature's own awning_elevation_low_deg/high_deg
# precedent (utils.py) rather than adding new config surface for them.
#
# Extra air-changes/hour added to a room's ventilation_rate while its window
# or door is reported open. Single-sided natural-ventilation literature puts
# a fully open window/door around 5-15 ACH depending on opening size/wind;
# 8.0 sits centrally without assuming an unusually large or gusty opening.
OPENING_EXTRA_ACH = 8.0

# Multiplier applied to a room-pair's coupling conductance (g, kW/K) while
# either room's door is open. Manually-configured closed-state g values in
# this codebase's own docs typically run 0.05-0.6 kW/K; a 5x multiplier lands
# a typical pairing at ~0.5-1.5 kW/K, consistent with commonly-cited
# open-interior-doorway natural-convection figures.
DOOR_OPEN_COUPLING_MULTIPLIER = 5.0

# Permissive sentinels used to relax a room's min/max comfort-temperature
# bound at the near-term timestep while its window/door is open (so pausing
# heat input there, see OPENING_EXTRA_ACH's own docstring context, can never
# make that one solve infeasible against a comfort bound it has no way to
# meet). Both min_temps/max_temps cp.Parameters are declared without
# nonneg=True, so a negative sentinel is legal.
OPENING_RELAX_MIN_TEMP = -100.0
OPENING_RELAX_MAX_TEMP = 1000.0

# Penalty coefficient (currency-per-degree-per-timestep) for the elastic
# comfort-bound slack _add_thermal_battery_bounds_and_penalty introduces
# only during _perform_optimization_core's already-existing infeasible-retry
# pass (self._soft_comfort_bounds_pass) - see that method's own comment.
# Large enough to dominate any real cost/comfort trade-off in the objective
# (so a genuinely avoidable violation is never "cheaper" than the real fix),
# but a finite value rather than a big-M placeholder, since it directly
# scales the objective and must stay solver-well-conditioned.
_COMFORT_VIOLATION_PENALTY_PER_DEGREE = 1000.0


def _resolve_phase_tag(tag: str, phase_labels: list[str]) -> list[str] | None:
    """Parse a load_phase/battery_phase tag (see _add_phase_balance_constraints)
    into the list of phase labels it refers to.

    A tag is either empty ("", unassigned - excluded from every phase's
    sum), a single label ("L1"), or any "+"-joined combination of 2 or
    more labels ("L1+L2", "L1+L2+L3", any order) for a device that is
    itself wired across more than one phase - a genuinely multi-phase
    heat pump compressor, boiler element, battery inverter, or a fixed
    multi-phase-only EV charger with no single-phase fallback. Power is
    assumed evenly split across whichever phases are named (the standard
    assumption for a symmetric multi-phase load, the same reasoning
    already used for PV's even-split fallback).

    Returns None for "" (unassigned) or when the tag is malformed or
    names any phase that isn't currently active (see phase_labels) - the
    caller should treat None as "exclude entirely" and log once for the
    latter case, mirroring validate_num_phases' own philosophy: a stale/
    typo'd tag should degrade coverage visibly, never silently
    misattribute power to the wrong phase count.
    """
    tag = tag.strip()
    if not tag:
        return None
    labels = [p.strip() for p in tag.split("+") if p.strip()]
    if not labels or any(p not in phase_labels for p in labels):
        return None
    return labels


def _linearize_relu(
    constraints: list, expr, lower_bound: float, upper_bound: float, name: str
) -> tuple[cp.Variable, cp.Variable]:
    """Exact MILP linearization of ``z = max(expr, 0)``, for an affine CVXPY
    expression ``expr`` with known constant bounds ``[lower_bound,
    upper_bound]`` (``lower_bound`` may be negative or positive; the
    formulation is exact either way, including when the bound doesn't
    actually contain 0).

    Introduces one new boolean indicator ``b`` (``b == 1`` <=> ``expr >=
    0`` at any feasible, non-dominated solution) and one new nonneg
    variable ``z``, with exactly 3 linking inequalities:

        z >= expr
        z <= upper_bound * b
        z <= expr - lower_bound * (1 - b)

    No separate constraint is needed to force ``b`` to actually track
    ``expr``'s sign - it falls out for free: if ``expr > 0`` and ``b`` were
    0, the first and second constraints would require ``z >= expr > 0``
    and ``z <= 0`` simultaneously (infeasible), and symmetrically if
    ``expr < 0`` and ``b`` were 1, ``z <= expr - lower_bound*0 = expr < 0``
    would contradict ``z``'s own non-negativity - so any feasible ``b`` is
    already the correct one.

    Used for ``delta_supply = max(supply_temp - room_last, 0)`` and
    ``delta_env = max(room_last - outdoor_temp, 0)`` in
    ``_add_self_learning_dispatch_milp_constraints`` - unlike the
    reference-trajectory-linearized version these replace, both operands
    here are live CVXPY expressions (``supply_temp`` is now a decision
    variable, ``room_last`` a recurrence variable), so this can no longer
    be computed as a plain numpy ``np.clip``.

    :param constraints: Constraint list to append the 3 new inequalities to.
    :param expr: The affine CVXPY expression to take ``max(expr, 0)`` of.
    :param lower_bound: A valid constant lower bound on every element of `expr`.
    :param upper_bound: A valid constant upper bound on every element of `expr`.
    :param name: Unique name prefix for the new `b`/`z` CVXPY variables.
    :return: ``(z, b)`` - the ReLU output and its sign indicator.
    :rtype: tuple[cp.Variable, cp.Variable]
    """
    n = expr.shape[0] if getattr(expr, "shape", None) else 1
    b = cp.Variable(n, boolean=True, name=f"{name}_ind")
    z = cp.Variable(n, nonneg=True, name=f"{name}_relu")
    constraints.append(z >= expr)
    constraints.append(z <= upper_bound * b)
    constraints.append(z <= expr - lower_bound * (1 - b))
    return z, b


def _linearize_binary_times_continuous(
    constraints: list,
    binary_var,
    continuous_expr,
    continuous_lower_bound: float,
    continuous_upper_bound: float,
    name: str,
) -> cp.Variable:
    """Exact MILP linearization of ``z = binary_var * continuous_expr`` -
    the standard McCormick reformulation for a product where one factor is
    genuinely boolean, which (unlike the general continuous x continuous
    case) is exact, not a relaxation:

        z <= continuous_upper_bound * binary_var
        z >= continuous_lower_bound * binary_var
        z <= continuous_expr - continuous_lower_bound * (1 - binary_var)
        z >= continuous_expr - continuous_upper_bound * (1 - binary_var)

    Used for ``duty_x_delta_supply``/``duty_x_delta_env`` in
    ``_add_self_learning_dispatch_milp_constraints``, where `binary_var` is
    the shared heat-source on/off variable and `continuous_expr` is the
    (already-linearized-via-`_linearize_relu`) `delta_supply`/`delta_env`
    output - both always >= 0 in practice, so `continuous_lower_bound` is
    normally 0, but the formulation is exact for any valid bound pair.

    :param constraints: Constraint list to append the 4 new inequalities to.
    :param binary_var: A boolean CVXPY variable.
    :param continuous_expr: An affine CVXPY expression, the other factor.
    :param continuous_lower_bound: A valid constant lower bound on `continuous_expr`.
    :param continuous_upper_bound: A valid constant upper bound on `continuous_expr`.
    :param name: Unique name prefix for the new `z` CVXPY variable.
    :return: `z`, equal to `binary_var * continuous_expr` at every feasible point.
    :rtype: cp.Variable
    """
    n = continuous_expr.shape[0] if getattr(continuous_expr, "shape", None) else 1
    z = cp.Variable(n, name=f"{name}_bxc")
    constraints.append(z <= continuous_upper_bound * binary_var)
    constraints.append(z >= continuous_lower_bound * binary_var)
    constraints.append(z <= continuous_expr - continuous_lower_bound * (1 - binary_var))
    constraints.append(z >= continuous_expr - continuous_upper_bound * (1 - binary_var))
    return z


class Optimization:
    r"""
    Optimize the deferrable load and battery energy dispatch problem using \
    the linear programming optimization technique. All equipement equations, \
    including the battery equations are hence transformed in a linear form.

    This class methods are:

    - perform_optimization

    - perform_perfect_forecast_optim

    - perform_dayahead_forecast_optim

    - perform_naive_mpc_optim

    """

    def __init__(
        self,
        retrieve_hass_conf: dict,
        optim_conf: dict,
        plant_conf: dict,
        var_load_cost: str,
        var_prod_price: str,
        costfun: str,
        emhass_conf: dict,
        logger: logging.Logger,
        opt_time_delta: int | None = 24,
        num_timesteps: int | None = None,
    ) -> None:
        r"""
        Define constructor for Optimization class.

        :param retrieve_hass_conf: Configuration parameters used to retrieve data \
            from hass
        :type retrieve_hass_conf: dict
        :param optim_conf: Configuration parameters used for the optimization task
        :type optim_conf: dict
        :param plant_conf: Configuration parameters used to model the electrical \
            system: PV production, battery, etc.
        :type plant_conf: dict
        :param var_load_cost: The column name for the unit load cost.
        :type var_load_cost: str
        :param var_prod_price: The column name for the unit power production price.
        :type var_prod_price: str
        :param costfun: The type of cost function to use for optimization problem
        :type costfun: str
        :param emhass_conf: Dictionary containing the needed emhass paths
        :type emhass_conf: dict
        :param logger: The passed logger object
        :type logger: logging object
        :param opt_time_delta: The number of hours to optimize. If days_list has \
            more than one day then the optimization will be peformed by chunks of \
            opt_time_delta periods, defaults to 24
        :type opt_time_delta: float, optional

        """
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        # Number of batteries (#610). Read defensively: plant_conf may come
        # from a hand-built dict (tests, or a config predating this feature)
        # that never sets the key, in which case a single battery is the only
        # sensible default. Structural: a change to this count alters the
        # number of decision variables/constraints, so (like
        # number_of_deferrable_loads) it must invalidate any cached problem
        # rather than update a cp.Parameter in place.
        self.n_batt = int(self.plant_conf.get("number_of_batteries", 1))
        # Number of electrical phases (1/2/3) for the additive per-phase
        # power-balance safety constraint (see _add_phase_balance_constraints).
        # <= 1 leaves the whole feature inert - phase_labels stays empty, and
        # every downstream check keys off that emptiness rather than n_phases
        # directly, so a stray >1 value with no labels built yet can never
        # half-activate the feature.
        self.n_phases = int(self.plant_conf.get("number_of_phases", 1) or 1)
        self.phase_labels = [f"L{i + 1}" for i in range(self.n_phases)] if self.n_phases > 1 else []
        self.freq = self.retrieve_hass_conf["optimization_time_step"]
        self.time_zone = self.retrieve_hass_conf["time_zone"]
        self.time_step = self.freq.seconds / 3600  # in hours
        self.var_pv = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        self.var_load = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        self.var_load_new = self.var_load + "_positive"
        self.costfun = costfun
        self.emhass_conf = emhass_conf
        self.logger = logger
        self.var_load_cost = var_load_cost
        self.var_prod_price = var_prod_price
        self.optim_status = None

        # Prioritize config value over default arg
        if "delta_forecast_daily" in self.optim_conf:
            # If configured in days (int/float), convert to timedelta
            val = self.optim_conf["delta_forecast_daily"]
            if isinstance(val, int) or isinstance(val, float):
                self.time_delta = pd.to_timedelta(val, "days")
            else:
                # Assume it is already a timedelta or compatible
                self.time_delta = pd.to_timedelta(val)
        else:
            # Fallback to the argument (default 24h)
            self.time_delta = pd.to_timedelta(opt_time_delta, "hours")

        # Configuration for Solver
        if "num_threads" in optim_conf.keys():
            if optim_conf["num_threads"] == 0:
                self.num_threads = int(os.cpu_count())
            else:
                self.num_threads = int(optim_conf["num_threads"])
        else:
            self.num_threads = int(os.cpu_count())

        # Force HiGHS solver or use configured one, defaulting to Highs if not specified
        if "lp_solver" in optim_conf.keys():
            self.lp_solver = optim_conf["lp_solver"]
        else:
            self.lp_solver = "Highs"  # Default to Highs for speed

        # Mask sensitive data before logging
        conf_to_log = retrieve_hass_conf.copy()
        keys_to_mask = utils.get_keys_to_mask()
        for key in keys_to_mask:
            if key in conf_to_log:
                conf_to_log[key] = "***"
        self.logger.debug(f"Initialized Optimization with retrieve_hass_conf: {conf_to_log}")
        self.logger.debug(f"Optimization configuration: {optim_conf}")
        self.logger.debug(f"Plant configuration: {plant_conf}")
        self.logger.debug(f"Number of threads: {self.num_threads}")

        # CVXPY Initialization
        # Calculate the fixed number of time steps (N)
        # num_timesteps may be passed explicitly to account for DST-adjusted horizons.
        if num_timesteps is not None:
            self.num_timesteps = num_timesteps
        else:
            self.num_timesteps = int(self.time_delta / self.freq)
        self.logger.debug(f"CVXPY: Initialization with {self.num_timesteps} time steps.")

        # Define Parameters (Data holders)
        # These will be updated in perform_optimization without rebuilding the problem
        self.param_pv_forecast = cp.Parameter(self.num_timesteps, name="pv_forecast")
        self.param_load_forecast = cp.Parameter(self.num_timesteps, name="load_forecast")
        # Per-phase load/PV parameters (only when number_of_phases > 1) - see
        # _add_phase_balance_constraints. Populated from the p_load_phase_{lbl}/
        # p_pv_phase_{lbl} columns command_line.py::prepare_forecast_and_weather_data
        # writes onto data_opt, the same optional-column side-channel already
        # used for ghi/wind_speed/dni/dhi.
        self.param_load_forecast_phase = {
            lbl: cp.Parameter(self.num_timesteps, name=f"load_forecast_{lbl}")
            for lbl in self.phase_labels
        }
        self.param_pv_forecast_phase = {
            lbl: cp.Parameter(self.num_timesteps, nonneg=True, name=f"pv_forecast_{lbl}")
            for lbl in self.phase_labels
        }
        self.param_load_cost = cp.Parameter(self.num_timesteps, name="load_cost")
        # Non-negative clip of the import tariff, used only by the battery-first
        # priority penalty (issue #1002). Pricing that penalty off the raw signed
        # tariff would turn it into an unbounded reward in a negative-price slot
        # (routine on day-ahead markets), making the penalty variable run to
        # infinity. A dedicated Parameter (rather than max() baked in at build
        # time) keeps the clip correct across warm-started re-solves.
        self.param_load_cost_pos = cp.Parameter(
            self.num_timesteps, nonneg=True, name="load_cost_pos"
        )
        # Non-negative PV surplus, max(0, PV - load), used as the export ceiling for
        # set_nodischarge_to_grid on AC-coupled systems. Bounding export by raw PV
        # (pre-fix behaviour) lets the battery reach the grid indirectly: it covers
        # the whole load so PV is freed for export (regression of #795, reintroduced
        # by #981). Bounding by the surplus blocks battery-to-grid while still
        # allowing battery-to-load. Dedicated Parameter (not max() baked in at build
        # time) to stay correct across warm-started re-solves.
        self.param_export_ceiling = cp.Parameter(
            self.num_timesteps, nonneg=True, name="export_ceiling"
        )
        # Currency per Wh charged for missing the terminal SoC target. Set per solve
        # from the horizon's highest import tariff so the target stays dominant at any
        # price scale; a Parameter (not a baked-in constant) keeps that correct across
        # warm-started re-solves. See SOC_FINAL_DEVIATION_PENALTY_FACTOR.
        self.param_soc_final_penalty = cp.Parameter(nonneg=True, name="soc_final_penalty")
        self.param_prod_price = cp.Parameter(self.num_timesteps, name="prod_price")

        # Per-deferrable-load cost override parameters. When the user supplies a
        # `cost_forecast_per_deferrable_load[k]` array, that load is priced at its
        # own per-timestep rate (e.g., gas price for a gas-boiler load) instead of
        # the shared electricity tariff. The objective adds an adjustment term
        # `(per_load_cost - load_cost) * p_deferrable[k]` per load. Default values
        # equal `load_cost` for every timestep, making the adjustment a no-op
        # unless the user explicitly overrides.
        num_def_loads = self.optim_conf.get("number_of_deferrable_loads", 0)
        self.param_cost_per_load = [
            cp.Parameter(self.num_timesteps, name=f"cost_per_load_{k}")
            for k in range(num_def_loads)
        ]

        # Per-battery Scalar Parameters (#610). A list of length self.n_batt,
        # one cp.Parameter per battery, indexed k in range(self.n_batt) - this
        # is the uniform indexing scheme the whole battery model below follows.
        # At n_batt == 1 this is a 1-element list, so the N=1 solve is
        # mathematically identical to before (single scalar per Parameter);
        # only the Python container shape differs internally.
        self.param_soc_init = [
            cp.Parameter(nonneg=True, name=f"soc_init_{k}") for k in range(self.n_batt)
        ]
        self.param_soc_final = [
            cp.Parameter(nonneg=True, name=f"soc_final_{k}") for k in range(self.n_batt)
        ]

        # Battery power limits — parameterised so SoC-derated values arriving
        # via runtimeparams update without invalidating the OptimizationCache.
        # One Parameter per battery (update_battery_power_limits loops over k).
        self.param_battery_charge_power_max = [
            cp.Parameter(nonneg=True, name=f"battery_charge_power_max_{k}")
            for k in range(self.n_batt)
        ]
        self.param_battery_discharge_power_max = [
            cp.Parameter(nonneg=True, name=f"battery_discharge_power_max_{k}")
            for k in range(self.n_batt)
        ]
        # Read only the two power-limit keys here (not the full
        # _battery_conf_as_lists(), which also reads weight_battery_charge/
        # discharge - those are irrelevant to Parameter seeding and, unlike
        # the power limits, are not guaranteed present on a hand-built
        # set_use_battery=False config).
        _charge_max_list = self._batt_list(self.plant_conf, "battery_charge_power_max", default=0)
        _discharge_max_list = self._batt_list(
            self.plant_conf, "battery_discharge_power_max", default=0
        )
        for k in range(self.n_batt):
            self.param_battery_charge_power_max[k].value = float(_charge_max_list[k])
            self.param_battery_discharge_power_max[k].value = float(_discharge_max_list[k])

        # SOC recovery parameters
        self._init_soc_recovery_params()

        # Optional intermediate SOC target parameters (issue #553)
        self._init_soc_target_params()

        # Peak grid import already incurred this billing period (issue #623, Phase 2)
        self._init_current_period_peak_param()

        # Initialize deferrable load parameters (window masks and energy constraints)
        self._init_deferrable_load_params()

        # Initialize Variables & Bound Constraints
        self.vars, self.constraints = self._initialize_decision_variables()

        # Note: The self.prob object will be constructed in a subsequent step
        self.prob = None

        # Self-learning-physics dispatch state (see perform_optimization /
        # _perform_two_pass_optimization) - a no-op for every
        # config with no heatpump_room_self_learning_only room, so this is
        # cheap to always initialize rather than lazily via getattr.
        self._self_learning_force_rc_pass = False
        self._sl_reference_trajectories: dict[int, np.ndarray] = {}
        self._sl_reference_signature: dict[int, tuple] = {}
        self._sl_cache_solve_count = 0
        self._sl_last_solve_status: str | None = None

        # RC-physics dispatch state (see perform_optimization /
        # _perform_two_pass_optimization) - the SAME forced-
        # reference-pass mechanism self-learning-physics uses (a room's own
        # q_emit=duty*max(supply-air,0) update has the identical "duty times
        # a live-state-dependent clamp" nonlinearity as self-learning's own
        # duty_x_delta_supply term), reusing _self_learning_force_rc_pass
        # rather than a second flag - both room types need every OTHER room
        # forced onto its ordinary recurrence during the same reference pass.
        # A no-op for every config with no heatpump_room_rc_physics_only
        # room, so cheap to always initialize.
        self._rc_reference_trajectories: dict[int, np.ndarray] = {}
        self._rc_reference_signature: dict[int, tuple] = {}

    def _init_soc_recovery_params(self) -> None:
        """Initialize CVXPY parameters used for out-of-band SOC recovery.

        One set per battery (#610): each battery can independently start out
        of its own [min, max] band and recover once. Lists of length
        self.n_batt, indexed k like every other per-battery Parameter.
        """
        self.param_soc_low_gap = [
            cp.Parameter(nonneg=True, name=f"soc_low_gap_{k}") for k in range(self.n_batt)
        ]
        self.param_soc_high_gap = [
            cp.Parameter(nonneg=True, name=f"soc_high_gap_{k}") for k in range(self.n_batt)
        ]
        self.param_soc_low_required = [
            cp.Parameter(nonneg=True, name=f"soc_low_required_{k}") for k in range(self.n_batt)
        ]
        self.param_soc_high_required = [
            cp.Parameter(nonneg=True, name=f"soc_high_required_{k}") for k in range(self.n_batt)
        ]
        for k in range(self.n_batt):
            self.param_soc_low_gap[k].value = 0.0
            self.param_soc_high_gap[k].value = 0.0
            self.param_soc_low_required[k].value = 0.0
            self.param_soc_high_required[k].value = 0.0

    def _init_soc_target_params(self) -> None:
        """Initialize CVXPY parameters for the optional intermediate SOC target (#553).

        ``param_soc_target_floor`` is a single per-horizon vector giving the
        minimum stored energy (Wh) required at each timestep: the target energy
        at the requested timestep and 0.0 everywhere else. Using one precomputed
        floor vector (rather than a mask * value product of two parameters) keeps
        the problem DPP / warm-start safe — the numeric multiply happens at
        set-time, so no recanonicalisation is forced on each solve. The default
        (all zeros) makes the constraint a no-op, so behaviour is unchanged
        unless a target is explicitly requested. It is a vector param so it must
        be (re)created whenever the horizon length changes. Called from __init__
        and when resizing the optimization problem.

        One vector per battery (#610), list of length self.n_batt: the target
        itself is not yet a per-battery runtime input, so every battery's floor
        is fed the identical target fraction, applied against ITS OWN capacity
        in perform_optimization. The per-battery Parameter exists now so a
        future per-battery target only has to change the value each entry
        receives, not the model structure.
        """
        self.param_soc_target_floor = [
            cp.Parameter(self.num_timesteps, nonneg=True, name=f"soc_target_floor_{k}")
            for k in range(self.n_batt)
        ]
        for k in range(self.n_batt):
            self.param_soc_target_floor[k].value = np.zeros(self.num_timesteps)

    def _init_current_period_peak_param(self) -> None:
        """Initialize the CVXPY parameter for the peak grid import already
        incurred this billing period (issue #623, Phase 2).

        ``param_current_period_peak`` is a single scalar in WATTS (matching
        p_grid_pos / peak_import) used to raise the floor of the ``peak_import``
        epigraph variable so the demand / capacity charge accounts for a peak
        already locked in for the period: once the floor binds, shaving below it
        has zero marginal value, so the solver does not waste battery or
        deferrable flexibility on a peak it cannot reduce.

        Like ``param_soc_target_floor`` it is a ``cp.Parameter`` so its value is
        set per call without forcing a problem rebuild (DPP / warm-start safe).
        Default 0.0 makes the added constraint ``peak_import >= 0`` redundant
        with the variable's own non-negativity and the existing
        ``peak_import >= p_grid_pos`` epigraph, so behaviour is identical to
        Phase 1 unless a value is explicitly passed. Being a scalar (not
        horizon-dependent) it does NOT need re-creation when the horizon
        resizes, so unlike ``_init_soc_target_params`` it is created in
        __init__ only.
        """
        self.param_current_period_peak = cp.Parameter(nonneg=True, name="current_period_peak")
        self.param_current_period_peak.value = 0.0

    def _init_deferrable_load_params(self) -> None:
        """
        Initialize CVXPY parameters for deferrable loads (window masks and energy constraints).

        This method creates:
        - param_window_masks: Allow changing time windows without rebuilding the problem
        - param_target_energy: Target energy for Big-M energy constraints
        - param_energy_active: Flags to enable/disable energy constraints
        - param_required_timesteps: Required timesteps for binary loads
        - param_timesteps_active: Flags to enable/disable timestep constraints

        Called from __init__ and when resizing the optimization problem.
        """
        num_def_loads = self.optim_conf.get("number_of_deferrable_loads", 0)
        n = self.num_timesteps

        # Window Mask Parameters for Deferrable Loads
        # mask[t] = 0 means load must be off at timestep t
        # mask[t] = 1 means load can operate at timestep t
        self.param_window_masks = []
        for k in range(num_def_loads):
            mask = cp.Parameter(n, nonneg=True, name=f"window_mask_{k}")
            mask.value = np.ones(n)  # Default: no restriction
            self.param_window_masks.append(mask)

        # Gate for manually-committed sequence loads (see manual_load_enabled):
        # a plain program_based/sequence load's cp.sum(y)==1 constraint is
        # unconditional (it must always run somewhere), which is correct for
        # a real fixed program but wrong for a manual load that should stay
        # fully idle whenever it hasn't been requested - forcing the window
        # mask to all-zero instead would conflict with cp.sum(y)==1 and make
        # the whole MILP infeasible. Default 1.0 (must-run) so every
        # non-manual program_based sequence load is unaffected; only
        # manual-auto sequence loads ever get this parameterized down to 0.
        self.param_sequence_required = []
        for k in range(num_def_loads):
            req = cp.Parameter(nonneg=True, name=f"seq_required_{k}")
            req.value = 1.0
            self.param_sequence_required.append(req)

        # Energy Constraint Parameters for Deferrable Loads
        # Uses Big-M formulation to enable/disable the constraint
        self.param_target_energy = []  # Target energy in Wh
        self.param_energy_active = []  # 1 = constraint active, 0 = inactive (relaxed via Big-M)
        self.param_required_timesteps = []  # For binary loads: number of timesteps to run
        self.param_timesteps_active = []  # 1 = timestep constraint active, 0 = inactive
        for k in range(num_def_loads):
            # Target energy parameter
            energy_param = cp.Parameter(nonneg=True, name=f"target_energy_{k}")
            energy_param.value = 0.0
            self.param_target_energy.append(energy_param)

            # Energy constraint active flag
            energy_active = cp.Parameter(nonneg=True, name=f"energy_active_{k}")
            energy_active.value = 0.0
            self.param_energy_active.append(energy_active)

            # Required timesteps for binary loads
            timesteps_param = cp.Parameter(nonneg=True, name=f"required_timesteps_{k}")
            timesteps_param.value = 0.0
            self.param_required_timesteps.append(timesteps_param)

            # Timesteps constraint active flag
            timesteps_active = cp.Parameter(nonneg=True, name=f"timesteps_active_{k}")
            timesteps_active.value = 0.0
            self.param_timesteps_active.append(timesteps_active)

        # Deferrable load current state parameters (for startup detection)
        # Allows updating def_current_state without rebuilding constraints.
        # IMPORTANT: Values MUST be exactly 0.0 or 1.0 (binary indicator).
        # Fractional values would weaken the MIP startup/on-off constraints.
        self.param_def_current_state = []
        for k in range(num_def_loads):
            p = cp.Parameter(nonneg=True, name=f"def_current_state_{k}")
            p.value = 0.0
            self.param_def_current_state.append(p)

        # Running lower-bound masks for single-constant loads that are currently running.
        # param_running_lb[k][t] = 1 forces p_def_bin2[k][t] = 1 (load must stay on).
        # param_already_running_sc[k] = 1 suppresses the mandatory startup event so the
        # solver doesn't try to turn the load off and back on to satisfy sum(starts)==1.
        self.param_running_lb = []
        self.param_already_running_sc = []
        for k in range(num_def_loads):
            lb = cp.Parameter(n, nonneg=True, name=f"running_lb_{k}")
            lb.value = np.zeros(n)
            self.param_running_lb.append(lb)
            ar = cp.Parameter(nonneg=True, name=f"already_running_sc_{k}")
            ar.value = 0.0
            self.param_already_running_sc.append(ar)

        # Min-on-time elapsed tracking (for initial-condition remainder, issue #952).
        # param_current_on_timesteps[k]: integer timesteps the load has already been ON
        # at the start of this horizon. Only meaningful when def_current_state[k]=True
        # and def_minimum_on_time[k] > 0. Absent in optim_conf -> no initial force.
        # This is a scalar nonneg Parameter (mirrors param_def_current_state).
        # The CONSTRAINT enforcing remaining = max(0, N - elapsed) ON steps is applied
        # by writing param_running_lb in the per-solve param-update block below.
        self.param_current_on_timesteps = []
        for k in range(num_def_loads):
            cot = cp.Parameter(nonneg=True, name=f"current_on_timesteps_{k}")
            cot.value = 0.0
            self.param_current_on_timesteps.append(cot)

        # Min-off-time elapsed tracking (for initial-condition remainder, #952 follow-on).
        # param_current_off_timesteps[k]: integer timesteps the load has already been OFF
        # at the start of this horizon. Only meaningful when def_current_state[k]=False
        # and def_minimum_off_time[k] > 0. Absent in optim_conf -> no initial force.
        # The CONSTRAINT enforcing remaining = max(0, N - elapsed) OFF steps is applied
        # by writing param_running_ub in the per-solve param-update block below.
        self.param_current_off_timesteps = []
        for k in range(num_def_loads):
            coft = cp.Parameter(nonneg=True, name=f"current_off_timesteps_{k}")
            coft.value = 0.0
            self.param_current_off_timesteps.append(coft)

        # Force-OFF mask: param_running_ub[k] is a per-load length-n CVXPY Parameter
        # vector. Default value 1.0 = no force (upper bound is never tight). When the
        # min-off remainder is active, entries are set to 0.0 to force bin2[k][t] <= 0.
        # Constraint bin2[k] <= param_running_ub[k] is added ONLY for loads with
        # def_minimum_off_time[k] > 0 (so inactive loads never see a trivial bin2<=1
        # constraint). Mirrors param_running_lb but for the OFF direction.
        self.param_running_ub = []
        for k in range(num_def_loads):
            ub = cp.Parameter(n, nonneg=True, name=f"running_ub_{k}")
            ub.value = np.ones(n)
            self.param_running_ub.append(ub)

        # Current-power parameters (issue #605).
        # param_def_current_power[k]: the actual power (W) the load is drawing at t=0.
        # param_def_current_power_active[k]: 1 iff the power-pin constraint should be
        #   tight (i.e. the load is affected AND pin-eligible, see below). Both are set
        #   on every solve by _update_def_current_power_params. Default 0.0 = no-op.
        # _def_current_power_affected[k]: True iff def_current_power changes anything for
        #   load k (drives the t=0 force-on / phantom-startup suppression). Excludes
        #   single_const / sequence / thermal loads entirely (see the update method).
        self.param_def_current_power = []
        self.param_def_current_power_active = []
        self._def_current_power_affected = [False] * num_def_loads
        for k in range(num_def_loads):
            pw = cp.Parameter(nonneg=True, name=f"def_current_power_{k}")
            pw.value = 0.0
            self.param_def_current_power.append(pw)
            active = cp.Parameter(nonneg=True, name=f"def_current_power_active_{k}")
            active.value = 0.0
            self.param_def_current_power_active.append(active)

        # Completed operating-timesteps parameters (issue #983).
        # param_current_operating_timesteps[k]: how many operating timesteps load k has
        # already run today. Used to decrement required_timesteps and target_energy in the
        # per-solve param-update block, clamped at 0. Absent key -> no decrement (no-op).
        self.param_current_operating_timesteps = []
        for k in range(num_def_loads):
            cotp = cp.Parameter(nonneg=True, name=f"current_operating_timesteps_{k}")
            cotp.value = 0.0
            self.param_current_operating_timesteps.append(cotp)

        # Load active parameters: allows deactivating non-thermal loads with 0 operating
        # timesteps without rebuilding the problem. When param_load_active[k] = 0, all
        # binary variables for load k are forced to 0 by constraints, letting the solver
        # presolve them away instantly instead of branching on them.
        self.param_load_active = []
        for k in range(num_def_loads):
            p = cp.Parameter(nonneg=True, name=f"load_active_{k}")
            p.value = 1.0  # Default: all loads active
            self.param_load_active.append(p)
        # Thermal Parameters for warm-starting
        # Dict keyed by load index k, stores all parameters needed for thermal constraints
        # This allows updating runtime values (forecasts, temperatures) without rebuilding constraints
        self.param_thermal = {}
        def_load_config = self.optim_conf.get("def_load_config", []) or []
        for k in range(num_def_loads):
            if k < len(def_load_config) and def_load_config[k]:
                cfg = def_load_config[k]
                if "thermal_config" in cfg:
                    hc = cfg["thermal_config"]
                    if isinstance(hc, dict):
                        for bad_key in (key for key in hc if key not in THERMAL_CONFIG_KNOWN_KEYS):
                            hint = THERMAL_CONFIG_KEY_HINTS.get(bad_key)
                            if hint:
                                correct_key, role = hint
                                self.logger.warning(
                                    "Deferrable load %d thermal_config: unknown key '%s' is "
                                    "ignored; did you mean '%s' (%s)?",
                                    k,
                                    bad_key,
                                    correct_key,
                                    role,
                                )
                            else:
                                self.logger.warning(
                                    "Deferrable load %d thermal_config: unknown key '%s' is "
                                    "ignored. Recognized keys: %s.",
                                    k,
                                    bad_key,
                                    ", ".join(sorted(THERMAL_CONFIG_KNOWN_KEYS)),
                                )
                    init_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    min_temps = hc.get("min_temperatures", [])
                    max_temps = hc.get("max_temperatures", [])
                    desired_temps = hc.get("desired_temperatures", [])

                    self.param_thermal[k] = {
                        "type": "thermal_config",
                        "start_temp": cp.Parameter(name=f"thermal_start_temp_{k}", value=init_temp),
                        "outdoor_temp": cp.Parameter(n, name=f"thermal_outdoor_temp_{k}"),
                        "min_temps": cp.Parameter(n, name=f"thermal_min_temps_{k}"),
                        "max_temps": cp.Parameter(n, name=f"thermal_max_temps_{k}"),
                        "desired_temps": cp.Parameter(n, name=f"thermal_desired_temps_{k}"),
                    }
                    # Initialize with default values
                    self.param_thermal[k]["outdoor_temp"].value = np.full(n, 15.0)
                    self.param_thermal[k]["min_temps"].value = self._pad_temp_array(
                        min_temps, n, 18.0
                    )
                    self.param_thermal[k]["max_temps"].value = self._pad_temp_array(
                        max_temps, n, 26.0
                    )
                    self.param_thermal[k]["desired_temps"].value = self._pad_temp_array(
                        desired_temps, n, 22.0
                    )

                elif "thermal_battery" in cfg:
                    hc = cfg["thermal_battery"]
                    init_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    min_temps = hc.get("min_temperatures", [])
                    max_temps = hc.get("max_temperatures", [])
                    desired_temps = hc.get("desired_temperatures", [])

                    self.param_thermal[k] = {
                        "type": "thermal_battery",
                        "start_temp": cp.Parameter(
                            name=f"thermal_battery_start_temp_{k}", value=init_temp
                        ),
                        "outdoor_temp": cp.Parameter(n, name=f"thermal_battery_outdoor_temp_{k}"),
                        "min_temps": cp.Parameter(n, name=f"thermal_battery_min_temps_{k}"),
                        "max_temps": cp.Parameter(n, name=f"thermal_battery_max_temps_{k}"),
                        "thermal_losses": cp.Parameter(n, name=f"thermal_battery_losses_{k}"),
                        "heating_demand": cp.Parameter(
                            n, name=f"thermal_battery_heating_demand_{k}"
                        ),
                        "heatpump_cops": cp.Parameter(n, name=f"thermal_battery_cops_{k}"),
                        "desired_temps": cp.Parameter(n, name=f"thermal_battery_desired_temps_{k}"),
                    }
                    # Initialize with default values
                    self.param_thermal[k]["outdoor_temp"].value = np.full(n, 15.0)
                    self.param_thermal[k]["min_temps"].value = self._pad_temp_array(
                        min_temps, n, 18.0
                    )
                    self.param_thermal[k]["max_temps"].value = self._pad_temp_array(
                        max_temps, n, 26.0
                    )
                    self.param_thermal[k]["thermal_losses"].value = np.zeros(n)
                    self.param_thermal[k]["heating_demand"].value = np.zeros(n)
                    self.param_thermal[k]["heatpump_cops"].value = np.full(n, 3.0)
                    self.param_thermal[k]["desired_temps"].value = self._pad_temp_array(
                        desired_temps, n, 22.0
                    )

                    # Thermal inertia support (first-order low-pass filter on heat input)
                    # Always define q_input_start so downstream logic can rely on its presence.
                    # tau_hours controls whether inertia dynamics are applied, not whether
                    # this parameter exists.
                    q_input_init = float(hc.get("q_input_initial", 0.0) or 0.0)
                    self.param_thermal[k]["q_input_start"] = cp.Parameter(
                        name=f"thermal_battery_q_input_start_{k}", value=q_input_init
                    )

        # Legacy compatibility - keep param_thermal_start_temps as alias
        self.param_thermal_start_temps = {
            k: (params["type"], params["start_temp"]) for k, params in self.param_thermal.items()
        }

    def _pad_temp_array(self, temp_list: list, n: int, default: float) -> np.ndarray:
        """Pad/truncate temperature list to length n, replacing None with default."""
        if not temp_list:
            return np.full(n, default)
        arr = np.array([default if v is None else float(v) for v in temp_list[:n]])
        if len(arr) < n:
            arr = np.concatenate([arr, np.full(n - len(arr), default)])
        return arr

    def _relax_opening_temp_bounds(
        self, min_arr: np.ndarray, max_arr: np.ndarray, is_open: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Relax a room's min/max comfort-temperature bound at the near-term
        timestep (index 1 - min_temps_param[1:]/max_temps_param[1:] is what
        _add_thermal_battery_bounds_and_penalty actually constrains, index 0
        being the separately-pinned start_temperature) while its window/door
        is reported open right now, so pausing heat input there
        (see param_window_masks) can never make that one solve infeasible
        against a comfort bound it currently has no way to meet.

        Called at every site that assigns min_temps_param/max_temps_param's
        .value (cold-build physics, cache-hit update_thermal_params, and
        self-learning dispatch), always AFTER the array has been freshly
        built/padded for this call, so no stale relaxation ever persists
        across calls where the room's opening is no longer open.
        """
        if is_open and len(min_arr) > 1 and len(max_arr) > 1:
            min_arr = min_arr.copy()
            max_arr = max_arr.copy()
            min_arr[1] = OPENING_RELAX_MIN_TEMP
            max_arr[1] = OPENING_RELAX_MAX_TEMP
        return min_arr, max_arr

    def _persist_q_input(self, k: int, params: dict, hc: dict) -> None:
        """Auto-persist Q_input from previous solve and apply manual override.

        Called on cache hit to carry forward the thermal inertia filter state.
        Only persists when thermal inertia is currently enabled (tau > 0) AND a
        previous solve produced q_input values. If tau was changed to 0, any stale
        q_input_var is cleared to prevent surprising persistence.

        :param k: Deferrable load index
        :param params: The param_thermal[k] dict for this load
        :param hc: The thermal_battery config dict from def_load_config
        """
        tau_hours = float(hc.get("thermal_inertia_time_constant", 0.0) or 0.0)

        if tau_hours > 0 and "q_input_var" in params:
            prev_q = params["q_input_var"].value
            if prev_q is not None and len(prev_q) > 1:
                # Use index 1: in MPC the horizon shifts by one timestep,
                # so prev_q[1] becomes the new initial condition.
                new_q_start = float(prev_q[1])
                self.logger.debug(
                    "Auto-persisting q_input for load %s: %.4f -> %.4f",
                    k,
                    params["q_input_start"].value,
                    new_q_start,
                )
                params["q_input_start"].value = new_q_start
            elif prev_q is None:
                # Previous solve was infeasible — q_input has no values.
                # Fall back to heating demand so the next iteration doesn't
                # stay stuck at q_input_start=0 (which causes a persistent
                # infeasibility loop when start_temp <= min_temp).
                demand = params.get("heating_demand")
                fallback = 0.0
                if (
                    demand is not None
                    and hasattr(demand, "value")
                    and demand.value is not None
                    and len(demand.value) > 0
                ):
                    fallback = max(float(demand.value[0]), 0.0)
                old_val = float(params["q_input_start"].value or 0.0)
                if fallback > 0.0 or old_val < 1e-6:
                    params["q_input_start"].value = fallback
                    if abs(fallback - old_val) > 1e-6:
                        self.logger.warning(
                            "Load %s: previous solve infeasible, resetting "
                            "q_input_start from %.4f to heating demand fallback %.4f",
                            k,
                            old_val,
                            fallback,
                        )
                # Force problem rebuild so the feasibility guard in
                # _add_thermal_battery_constraints re-evaluates with the
                # updated q_input_start.  Without this, the constraint
                # structure from the initial build is reused on warm-start
                # and the guard condition is never re-checked.
                self.prob = None
                # Skip the q_input_initial override below — the recovery
                # value must survive to break the infeasibility loop.
                return
        elif tau_hours == 0 and "q_input_var" in params:
            # Inertia was disabled — clear stale variable reference
            del params["q_input_var"]
            params["q_input_start"].value = 0.0

        # Manual override via config takes priority
        if "q_input_initial" in hc:
            params["q_input_start"].value = float(hc.get("q_input_initial", 0.0) or 0.0)

    def _update_def_current_state_params(self, num_def_loads: int) -> None:
        """Update def_current_state CVXPY Parameters from optim_conf.

        Validates that each entry is a bool or numeric 0/1, raising ValueError
        for unexpected values that would silently weaken MIP constraints.
        Missing entries default to off (0.0).
        """
        if "def_current_state" not in self.optim_conf:
            # Reset all to 0.0 to avoid stale values from previous solves
            for k in range(min(num_def_loads, len(self.param_def_current_state))):
                self.param_def_current_state[k].value = 0.0
            return

        def_state_conf = self.optim_conf["def_current_state"]
        n_conf_states = len(def_state_conf)

        if n_conf_states != num_def_loads:
            self.logger.warning(
                "def_current_state length mismatch: "
                "num_deferrable_loads=%d, len(def_current_state)=%d; "
                "extra entries will be ignored or missing ones assumed off",
                num_def_loads,
                n_conf_states,
            )

        for k in range(num_def_loads):
            state = def_state_conf[k] if k < n_conf_states else False
            # Validate binary: accept bool and numeric 0/1, reject everything else
            if isinstance(state, bool):
                self.param_def_current_state[k].value = float(state)
            elif isinstance(state, int | float) and state in (0, 1, 0.0, 1.0):
                self.param_def_current_state[k].value = float(state)
            else:
                raise ValueError(
                    f"Invalid def_current_state value at index {k}: {state!r}. "
                    "Expected one of {{True, False, 0, 1, 0.0, 1.0}}."
                )

    @staticmethod
    def _coerce_nonneg_timesteps(value, k: int, param_name: str) -> int:
        """Validate a per-load timestep entry into a non-negative int (issue #952).

        Shared by def_minimum_on_time, def_minimum_off_time, def_current_on_timesteps,
        and def_current_off_timesteps so all min-on/off and elapsed-timestep validation
        lives in one place and a malformed value fails loudly with context instead of
        a bare int() error.
        """
        try:
            steps = int(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid {param_name} value at index {k}: {value!r}. "
                "Expected a non-negative integer (timesteps)."
            ) from err
        if steps < 0:
            raise ValueError(f"{param_name}[{k}]={steps} is negative; must be >= 0.")
        return steps

    def _update_def_current_on_timesteps_params(self, num_def_loads: int) -> None:
        """Update def_current_on_timesteps CVXPY Parameters from optim_conf.

        Reads ``optim_conf["def_current_on_timesteps"]`` (a per-load list of
        non-negative integers representing how many timesteps each load has
        already been ON at the start of the horizon) and writes the values
        into ``self.param_current_on_timesteps``.

        This is used to compute the remaining min-on-time steps for a currently-
        running load (issue #952): remaining = max(0, N - elapsed). When the key
        is absent from optim_conf the parameter is reset to 0.0 for all loads,
        which means no initial-run forcing is applied (NOT assumed-zero-elapsed;
        the absent-key path is intentionally a no-op).

        See also: ``_update_def_current_state_params`` (mirrors the same pattern).
        """
        if "def_current_on_timesteps" not in self.optim_conf:
            for k in range(min(num_def_loads, len(self.param_current_on_timesteps))):
                self.param_current_on_timesteps[k].value = 0.0
            return

        cot_conf = self.optim_conf["def_current_on_timesteps"]
        n_conf = len(cot_conf)
        if n_conf != num_def_loads:
            self.logger.warning(
                "def_current_on_timesteps length mismatch: "
                "num_deferrable_loads=%d, len(def_current_on_timesteps)=%d; "
                "extra entries will be ignored or missing ones assumed 0",
                num_def_loads,
                n_conf,
            )

        for k in range(num_def_loads):
            val = cot_conf[k] if k < n_conf else 0
            elapsed = self._coerce_nonneg_timesteps(val, k, "def_current_on_timesteps")
            if k < len(self.param_current_on_timesteps):
                self.param_current_on_timesteps[k].value = float(elapsed)

    def _update_def_current_off_timesteps_params(self, num_def_loads: int) -> None:
        """Update def_current_off_timesteps CVXPY Parameters from optim_conf.

        Reads ``optim_conf["def_current_off_timesteps"]`` (a per-load list of
        non-negative integers representing how many timesteps each load has
        already been OFF at the start of the horizon) and writes the values
        into ``self.param_current_off_timesteps``.

        This is used to compute the remaining min-off-time steps for a currently-
        stopped load (#952 follow-on): remaining = max(0, N - elapsed). When the key
        is absent from optim_conf the parameter is reset to 0.0 for all loads,
        which means no initial-off forcing is applied (NOT assumed-zero-elapsed;
        the absent-key path is intentionally a no-op).

        See also: ``_update_def_current_on_timesteps_params`` (mirrors the same pattern).
        """
        if "def_current_off_timesteps" not in self.optim_conf:
            for k in range(min(num_def_loads, len(self.param_current_off_timesteps))):
                self.param_current_off_timesteps[k].value = 0.0
            return

        coft_conf = self.optim_conf["def_current_off_timesteps"]
        n_conf = len(coft_conf)
        if n_conf != num_def_loads:
            self.logger.warning(
                "def_current_off_timesteps length mismatch: "
                "num_deferrable_loads=%d, len(def_current_off_timesteps)=%d; "
                "extra entries will be ignored or missing ones assumed 0",
                num_def_loads,
                n_conf,
            )

        for k in range(num_def_loads):
            val = coft_conf[k] if k < n_conf else 0
            elapsed = self._coerce_nonneg_timesteps(val, k, "def_current_off_timesteps")
            if k < len(self.param_current_off_timesteps):
                self.param_current_off_timesteps[k].value = float(elapsed)

    @staticmethod
    def _coerce_nonneg_power(value, k: int, param_name: str) -> float:
        """Validate a per-load def_current_power entry into a non-negative float (issue #605).

        A malformed value fails loudly with the param name and index for context.
        """
        try:
            watts = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid {param_name} value at index {k}: {value!r}. "
                "Expected a non-negative number (watts)."
            ) from err
        if watts < 0:
            raise ValueError(f"{param_name}[{k}]={watts} is negative; must be >= 0.")
        return watts

    def _update_def_current_power_params(self, num_def_loads: int) -> None:
        """Update def_current_power CVXPY Parameters from optim_conf (issue #605).

        Reads ``optim_conf["def_current_power"]`` (a per-load list of non-negative
        floats in watts representing the power each load is currently drawing) and
        writes the values into ``self.param_def_current_power`` and
        ``self.param_def_current_power_active``.

        Side-effect: when power[k] > 0, also bumps ``param_def_current_state[k]``
        to max(existing, 1.0) so the t=0 phantom-startup penalty is suppressed
        (mirrors the logic a caller would supply via def_current_state). The
        *input* boolean def_current_state is left untouched; only the internal
        CVXPY Parameter is shared.

        When the key is absent from optim_conf all parameters reset to 0.0, which
        is an exact no-op (no pin, no force-on, no phantom-startup suppression).

        Must be called AFTER ``_update_def_current_state_params`` so the existing
        param_def_current_state value is available for the max(...) bump.
        """
        self._def_current_power_affected = [False] * num_def_loads
        if "def_current_power" not in self.optim_conf:
            for k in range(min(num_def_loads, len(self.param_def_current_power))):
                self.param_def_current_power[k].value = 0.0
                self.param_def_current_power_active[k].value = 0.0
            return

        dcp_conf = self.optim_conf["def_current_power"]
        n_conf = len(dcp_conf)
        if n_conf != num_def_loads:
            self.logger.warning(
                "def_current_power length mismatch: "
                "num_deferrable_loads=%d, len(def_current_power)=%d; "
                "extra entries will be ignored or missing ones assumed 0",
                num_def_loads,
                n_conf,
            )

        # Eligibility is structural (determined at build time from load type).
        # Re-derive it here using the same flags used in the constraint block so the
        # parameters match what was baked into the problem.
        nominal_powers = self.optim_conf.get("nominal_power_of_deferrable_loads", [])
        semi_cont_flags = self.optim_conf.get("treat_deferrable_load_as_semi_cont", [])
        single_const_flags = self.optim_conf.get("set_deferrable_load_single_constant", [])

        for k in range(num_def_loads):
            val = dcp_conf[k] if k < n_conf else 0
            watts = self._coerce_nonneg_power(val, k, "def_current_power")

            if k < len(self.param_def_current_power):
                self.param_def_current_power[k].value = watts

            is_semi_cont = semi_cont_flags[k] if k < len(semi_cont_flags) else False
            is_single_const = single_const_flags[k] if k < len(single_const_flags) else False
            is_sequence_load = k < len(nominal_powers) and isinstance(nominal_powers[k], list)
            is_thermal = k in self.param_thermal

            # A load is AFFECTED by def_current_power only when injecting its t=0
            # power/on-state is meaningful and safe. Excluded entirely:
            #   - single_const: runs as one fixed block; "currently running" is already
            #     handled by def_current_state (which pins the remaining required
            #     timesteps). A below-nominal pin here would fight the required-energy
            #     target and silently relax the MIP, so use def_current_state for these.
            #   - sequence (list-valued nominal power): shaped by convolution, no free
            #     t=0 power variable to pin or force.
            #   - thermal: governed by temperature dynamics, not an on/off binary.
            affected = watts > 0 and not is_single_const and not is_sequence_load and not is_thermal
            if k < len(self._def_current_power_affected):
                self._def_current_power_affected[k] = affected

            # The power PIN additionally needs a free t=0 power variable, so semi_cont
            # is excluded from the pin (its power == nominal*bin); for an affected
            # semi_cont load the t=0 force-on alone injects nominal, which is correct
            # for an on/off device.
            pin_active = affected and not is_semi_cont
            if k < len(self.param_def_current_power_active):
                self.param_def_current_power_active[k].value = 1.0 if pin_active else 0.0

            # Suppress phantom startup: if this load is reported as running now,
            # bump param_def_current_state so t=0 is not counted as a start event.
            if affected and k < len(self.param_def_current_state):
                self.param_def_current_state[k].value = max(
                    self.param_def_current_state[k].value, 1.0
                )

    def _update_def_current_operating_timesteps_params(self, num_def_loads: int) -> None:
        """Update def_current_operating_timesteps CVXPY Parameters from optim_conf (issue #983).

        Reads ``optim_conf["def_current_operating_timesteps"]`` (a per-load list of
        non-negative integers representing how many operating timesteps each must-run load
        has already completed today) and writes the values into
        ``self.param_current_operating_timesteps``.

        When the key is absent from optim_conf all parameters reset to 0.0, which is an
        exact no-op (no decrement applied). The actual decrement of ``required_timesteps``
        and ``target_energy`` is applied in the per-solve param-update loop using the
        parameter values set here.

        A length mismatch is warned and handled gracefully: extra entries are ignored,
        missing ones are assumed 0 (no decrement for that load).
        """
        if "def_current_operating_timesteps" not in self.optim_conf:
            for k in range(min(num_def_loads, len(self.param_current_operating_timesteps))):
                self.param_current_operating_timesteps[k].value = 0.0
            return

        cots_conf = self.optim_conf["def_current_operating_timesteps"]
        n_conf = len(cots_conf)
        if n_conf != num_def_loads:
            self.logger.warning(
                "def_current_operating_timesteps length mismatch: "
                "num_deferrable_loads=%d, len(def_current_operating_timesteps)=%d; "
                "extra entries will be ignored or missing ones assumed 0",
                num_def_loads,
                n_conf,
            )

        for k in range(num_def_loads):
            val = cots_conf[k] if k < n_conf else 0
            elapsed = self._coerce_nonneg_timesteps(val, k, "def_current_operating_timesteps")
            if k < len(self.param_current_operating_timesteps):
                self.param_current_operating_timesteps[k].value = float(elapsed)

    def _batt_list(
        self,
        source: dict,
        key: str,
        *,
        required: bool = False,
        default: float | None = None,
    ) -> list:
        """
        Normalise one plant_conf/optim_conf battery value into a length
        self.n_batt list.

        utils.check_batt_params has already normalised these values before
        they reach this class: at number_of_batteries == 1 the value is left
        a bare scalar (single-battery math untouched); at N > 1 it is already
        an exact-length-N list. This helper only wraps the N==1 scalar into a
        1-element list so the rest of this module can iterate uniformly over
        ``for k in range(self.n_batt)``. It never mutates ``source``.

        ``required=True`` mirrors an existing direct ``source[key]`` read
        (KeyError if missing); ``required=False`` mirrors an existing
        ``source.get(key, default)`` read.
        """
        value = source[key] if required else source.get(key, default)
        if isinstance(value, list):
            if len(value) != self.n_batt:
                raise ValueError(
                    f"{key} has {len(value)} entries but number_of_batteries={self.n_batt}"
                )
            return value
        return [value] * self.n_batt

    def _batt_weight_list(self, value) -> list:
        """
        Normalise weight_battery_charge/weight_battery_discharge into a
        length self.n_batt list, mirroring utils.check_batt_weight_params'
        disambiguation at this module's own boundary.

        At n_batt == 1 the single entry IS the original value untouched
        (scalar or a flat time-series list): wrapping it as index 0 of a
        1-element list is a no-op for every existing single-battery read site
        (they all did ``np.array(weight_dis)`` on the raw value; now they do the
        exact same thing on ``weight_list[0]``).

        At n_batt > 1, utils.check_batt_weight_params has already resolved the
        value into a length-n_batt nested list (each entry a per-battery
        scalar or time series) before it reaches here, so the common case is
        just a pass-through. The remaining branches are a defensive fallback
        for a hand-built config that bypassed utils.py (e.g. a unit test
        constructing plant_conf/optim_conf directly): a bare scalar or a flat
        list not of length n_batt is broadcast/shared to every battery.
        """
        if self.n_batt == 1:
            return [value]
        if isinstance(value, list) and len(value) == self.n_batt:
            return list(value)
        return [value] * self.n_batt

    def _battery_conf_as_lists(self) -> dict:
        """
        Read every per-battery plant_conf/optim_conf value as a length
        self.n_batt list. Called from variable/parameter construction,
        constraint building, the objective, and results extraction, so every
        caller stays in lockstep on the same normalised view. Read-only:
        never mutates self.plant_conf/self.optim_conf.
        """
        return {
            "charge_power_max": self._batt_list(
                self.plant_conf, "battery_charge_power_max", default=0
            ),
            "discharge_power_max": self._batt_list(
                self.plant_conf, "battery_discharge_power_max", default=0
            ),
            "cap": self._batt_list(
                self.plant_conf, "battery_nominal_energy_capacity", required=True
            ),
            "eff_dis": self._batt_list(
                self.plant_conf, "battery_discharge_efficiency", required=True
            ),
            "eff_chg": self._batt_list(self.plant_conf, "battery_charge_efficiency", required=True),
            "soc_min": self._batt_list(
                self.plant_conf, "battery_minimum_state_of_charge", required=True
            ),
            "soc_max": self._batt_list(
                self.plant_conf, "battery_maximum_state_of_charge", required=True
            ),
            "soc_target": self._batt_list(
                self.plant_conf, "battery_target_state_of_charge", required=True
            ),
            "stress_cost": self._batt_list(self.plant_conf, "battery_stress_cost", default=0),
            "soc_deficit_threshold": self._batt_list(
                self.optim_conf, "battery_soc_deficit_threshold", default=0.4
            ),
            "soc_deficit_cost": self._batt_list(
                self.optim_conf, "battery_soc_deficit_cost", default=0.0
            ),
            "soc_surplus_threshold": self._batt_list(
                self.optim_conf, "battery_soc_surplus_threshold", default=0.9
            ),
            "soc_surplus_cost": self._batt_list(
                self.optim_conf, "battery_soc_surplus_cost", default=0.0
            ),
            "weight_dis": self._batt_weight_list(self.optim_conf["weight_battery_discharge"]),
            "weight_chg": self._batt_weight_list(self.optim_conf["weight_battery_charge"]),
        }

    def _normalize_soc_arg(self, value: float | list | None) -> list:
        """
        Normalise a perform_optimization soc_init/soc_final argument into a
        length self.n_batt list (#610). A bare float (or None) broadcasts to
        every battery - the pre-#610 call convention, still exactly what a
        single-battery caller passes today, so at n_batt == 1 this is a true
        no-op ([value] round-trips to the same value at index 0). An explicit
        list must be exactly self.n_batt long (hard error otherwise, matching
        check_batt_params' no-silent-padding stance).
        """
        if value is None:
            return [None] * self.n_batt
        if isinstance(value, list):
            if len(value) != self.n_batt:
                raise ValueError(
                    f"soc_init/soc_final list must have {self.n_batt} entries "
                    f"(number_of_batteries), got {len(value)}"
                )
            return list(value)
        return [value] * self.n_batt

    def _setup_battery_stress_cost(self, k: int, stress_unit_cost: float, max_power: float) -> dict:
        """
        Per-battery variant of _setup_stress_cost (#610). battery_stress_cost is
        a per-battery array but battery_stress_segments stays a single global
        PWL-discretisation knob (a discretisation choice, not a physical
        battery property), so this cannot reuse the generic key-based lookup
        verbatim: it takes the
        already-resolved per-battery stress-cost value directly, reads the
        shared segments knob under the unqualified "battery" key, and gives the
        created Variable a battery-index-qualified name (mirrors the
        deferrable-load ``f"..._{k}"`` naming idiom).
        """
        active = stress_unit_cost > 0 and max_power > 0
        stress_cost_var = None
        if active:
            stress_cost_var = cp.Variable(
                self.num_timesteps, nonneg=True, name=f"battery_stress_cost_{k}"
            )
        return {
            "active": active,
            "vars": stress_cost_var,
            "unit_cost": stress_unit_cost,
            "max_power": max_power,
            "segments": self.plant_conf.get("battery_stress_segments", 10),
        }

    def update_battery_power_limits(self, plant_conf: dict) -> None:
        """
        Update battery charge/discharge power-limit Parameters from plant_conf.

        Called on cache hit to sync runtime power-limit values without
        rebuilding constraints. Mirrors update_thermal_start_temps. One
        Parameter pair per battery (#610); ``plant_conf`` here is the
        possibly-refreshed runtime config, so values are read from it (not
        ``self.plant_conf``) via the same ``_batt_list`` normaliser used
        everywhere else, keyed off the structural ``self.n_batt``.

        :param plant_conf: The plant configuration containing
            battery_charge_power_max / battery_discharge_power_max
        """
        charge_list = self._batt_list(plant_conf, "battery_charge_power_max", default=0)
        discharge_list = self._batt_list(plant_conf, "battery_discharge_power_max", default=0)
        for k in range(self.n_batt):
            new_charge_max = float(charge_list[k] or 0)
            new_discharge_max = float(discharge_list[k] or 0)
            if self.param_battery_charge_power_max[k].value != new_charge_max:
                self.param_battery_charge_power_max[k].value = new_charge_max
            if self.param_battery_discharge_power_max[k].value != new_discharge_max:
                self.param_battery_discharge_power_max[k].value = new_discharge_max

    def update_thermal_start_temps(self, optim_conf: dict) -> None:
        """
        Update thermal start temperature parameters from optim_conf.

        Called on cache hit to sync runtime thermal parameters without rebuilding constraints.
        This is a convenience wrapper that only updates start_temp. For full updates including
        forecasts, use update_thermal_params().

        :param optim_conf: The optimization configuration containing def_load_config
        """
        def_load_config = optim_conf.get("def_load_config", []) or []
        for k, (thermal_type, param) in self.param_thermal_start_temps.items():
            if k < len(def_load_config) and def_load_config[k]:
                cfg = def_load_config[k]
                if thermal_type == "thermal_config" and "thermal_config" in cfg:
                    hc = cfg["thermal_config"]
                    new_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    if param.value != new_temp:
                        self.logger.debug(
                            f"Updating thermal_config start_temp for load {k}: {param.value} -> {new_temp}"
                        )
                        param.value = new_temp
                elif thermal_type == "thermal_battery" and "thermal_battery" in cfg:
                    hc = cfg["thermal_battery"]
                    new_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                    if param.value != new_temp:
                        self.logger.debug(
                            f"Updating thermal_battery start_temp for load {k}: {param.value} -> {new_temp}"
                        )
                        param.value = new_temp

                    if k in self.param_thermal:
                        self._persist_q_input(k, self.param_thermal[k], hc)

    def update_thermal_params(
        self,
        optim_conf: dict,
        data_opt: pd.DataFrame,
        p_load: np.ndarray,
        room_opening_open: list | None = None,
    ) -> None:
        """
        Update all thermal parameters from optim_conf and data_opt.

        Called on cache hit to sync all runtime thermal parameters without rebuilding constraints.
        This includes start_temperature, outdoor_temp forecasts, min/max temps, and derived
        values like thermal_losses, heating_demand, and heatpump_cops.

        :param optim_conf: The optimization configuration containing def_load_config
        :param data_opt: DataFrame with forecast data (outdoor_temperature_forecast, ghi, etc.)
        :param p_load: Load power forecast array (for internal gains calculation)
        :param room_opening_open: Optional per-load live "window OR door is open
            right now" list - see room_opening_open on perform_optimization. Relaxes
            the near-term comfort bound and boosts ventilation loss for a room whose
            opening is currently open, refreshed on this cache-hit path exactly like
            the fresh-build path (unlike room_blind_positions/room_door_open, which
            never reach this function).
        """
        def_load_config = optim_conf.get("def_load_config", []) or []
        n = self.num_timesteps

        for k, params in self.param_thermal.items():
            if k >= len(def_load_config) or not def_load_config[k]:
                continue

            cfg = def_load_config[k]
            thermal_type = params["type"]

            # Get outdoor temperature forecast
            outdoor_temp = self._get_clean_outdoor_temp(data_opt, n)

            if thermal_type == "thermal_config" and "thermal_config" in cfg:
                hc = cfg["thermal_config"]

                # Update start_temperature
                new_start_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                if params["start_temp"].value != new_start_temp:
                    self.logger.debug(
                        f"Updating thermal_config start_temp for load {k}: "
                        f"{params['start_temp'].value} -> {new_start_temp}"
                    )
                params["start_temp"].value = new_start_temp

                # Update outdoor_temp
                params["outdoor_temp"].value = outdoor_temp

                # Update min/max temperatures
                min_temps = hc.get("min_temperatures", [])
                max_temps = hc.get("max_temperatures", [])
                params["min_temps"].value = self._pad_temp_array(min_temps, n, 18.0)
                params["max_temps"].value = self._pad_temp_array(max_temps, n, 26.0)

                # Update desired_temperatures
                desired_temps = hc.get("desired_temperatures", [])
                params["desired_temps"].value = self._pad_temp_array(desired_temps, n, 22.0)

            elif thermal_type == "thermal_battery" and "thermal_battery" in cfg:
                hc = cfg["thermal_battery"]

                # Update start_temperature
                new_start_temp = float(hc.get("start_temperature", 20.0) or 20.0)
                if params["start_temp"].value != new_start_temp:
                    self.logger.debug(
                        f"Updating thermal_battery start_temp for load {k}: "
                        f"{params['start_temp'].value} -> {new_start_temp}"
                    )
                params["start_temp"].value = new_start_temp

                # Update outdoor_temp
                params["outdoor_temp"].value = outdoor_temp

                # Update min/max temperatures
                min_temps = hc.get("min_temperatures", [])
                max_temps = hc.get("max_temperatures", [])
                min_temps_arr = self._pad_temp_array(min_temps, n, 18.0)
                max_temps_arr = self._pad_temp_array(max_temps, n, 26.0)
                opening_open_k = (
                    room_opening_open is not None
                    and k < len(room_opening_open)
                    and room_opening_open[k]
                )
                min_temps_arr, max_temps_arr = self._relax_opening_temp_bounds(
                    min_temps_arr, max_temps_arr, opening_open_k
                )
                params["min_temps"].value = min_temps_arr
                params["max_temps"].value = max_temps_arr

                # Update desired_temperatures
                if "desired_temps" in params:
                    desired_temps_list = hc.get("desired_temperatures", [])
                    params["desired_temps"].value = self._pad_temp_array(
                        desired_temps_list, n, 22.0
                    )

                # Compute derived arrays
                indoor_target_temp = hc.get(
                    "indoor_target_temperature",
                    min_temps[0] if min_temps else 20.0,
                )

                # Conversion factors per timestep (Carnot COP for heat pumps,
                # flat value for constant-efficiency sources like gas boilers).
                cop_hc = self._resolve_boiler_hc_for_cop(k, hc)
                heatpump_cops = utils.resolve_thermal_battery_cop(cop_hc, outdoor_temp, length=n)
                params["heatpump_cops"].value = np.array(heatpump_cops)

                # Thermal losses and heating demand
                base_loss = hc.get("thermal_loss", 0.045)
                draw_off_profile = hc.get("draw_off_demand", None)

                if draw_off_profile is not None and len(draw_off_profile) > 0:
                    # Hot water tank mode: constant standby loss + tiled draw-off
                    params["thermal_losses"].value = np.full(n, base_loss)
                    draw_off_arr = self._tile_profile(draw_off_profile, n)
                    params["heating_demand"].value = draw_off_arr
                else:
                    # Building heating mode: outdoor-temp-dependent losses
                    thermal_losses = utils.calculate_thermal_loss_signed(
                        outdoor_temperature_forecast=outdoor_temp.tolist(),
                        indoor_temperature=new_start_temp,
                        base_loss=base_loss,
                    )
                    params["thermal_losses"].value = np.array(thermal_losses[:n])

                    # Heating demand
                    if all(
                        key in hc
                        for key in [
                            "u_value",
                            "envelope_area",
                            "ventilation_rate",
                            "heated_volume",
                        ]
                    ):
                        window_area = hc.get("window_area", None)
                        shgc = hc.get("shgc", 0.6)
                        internal_gains_factor = hc.get("internal_gains_factor", 0.0)

                        # Solar irradiance (direct/diffuse decomposition +
                        # blind-shading - blind_position is always the static
                        # hc.get("blind_position", 0.0) fallback here, since
                        # this cache-hit refresh path never receives a live
                        # room_blind_positions override, see
                        # _resolve_room_solar_irradiance's own docstring)
                        solar_irradiance = self._resolve_room_solar_irradiance(
                            data_opt, hc, n, window_area, hc.get("blind_position", 0.0)
                        )

                        # Internal gains
                        internal_gains_forecast = None
                        if internal_gains_factor > 0:
                            internal_gains_forecast = p_load

                        # Extra ventilation loss at the near-term step only
                        # while this room's window/door is open right now -
                        # see OPENING_EXTRA_ACH's own module-level docstring.
                        ventilation_rate_arr = np.full(n, hc["ventilation_rate"])
                        if (
                            room_opening_open is not None
                            and k < len(room_opening_open)
                            and room_opening_open[k]
                        ):
                            ventilation_rate_arr[0] += OPENING_EXTRA_ACH

                        heating_demand = utils.calculate_heating_demand_physics(
                            u_value=hc["u_value"],
                            envelope_area=hc["envelope_area"],
                            ventilation_rate=ventilation_rate_arr,
                            heated_volume=hc["heated_volume"],
                            indoor_target_temperature=indoor_target_temp,
                            outdoor_temperature_forecast=outdoor_temp.tolist(),
                            optimization_time_step=int(self.freq.total_seconds() / 60),
                            solar_irradiance_forecast=solar_irradiance,
                            window_area=window_area,
                            shgc=shgc,
                            internal_gains_forecast=internal_gains_forecast,
                            internal_gains_factor=internal_gains_factor,
                            sense=hc.get("sense") or "heat",
                        )
                        params["heating_demand"].value = np.array(heating_demand[:n])
                    else:
                        params["heating_demand"].value = np.zeros(n)

                self._persist_q_input(k, params, hc)

    def _get_clean_outdoor_temp(self, data_opt: pd.DataFrame, n: int) -> np.ndarray:
        """Extract and clean outdoor temperature from data_opt."""
        outdoor_temp = self._get_clean_list("outdoor_temperature_forecast", data_opt)
        if not outdoor_temp or all(x is None for x in outdoor_temp):
            outdoor_temp = self._get_clean_list("temp_air", data_opt)

        if not outdoor_temp or all(x is None for x in outdoor_temp):
            return np.full(n, 15.0)

        outdoor_temp = np.array(
            [15.0 if (x is None or pd.isna(x)) else float(x) for x in outdoor_temp]
        )
        if len(outdoor_temp) < n:
            pad = np.full(n - len(outdoor_temp), 15.0)
            outdoor_temp = np.concatenate((outdoor_temp, pad))
        return outdoor_temp[:n]

    def _get_clean_weather_col(self, data_opt: pd.DataFrame, column: str, n: int, default: float = 0.0) -> np.ndarray:
        """Generalizes _get_clean_outdoor_temp to any weather column (e.g.
        wind_speed/dni/dhi, see command_line.py::prepare_forecast_and_weather_data
        and _merge_weather_column) - used by
        _add_self_learning_dispatch_constraints. Falls back to a constant
        `default` array (never raises) when the column is missing, exactly
        as self_learning_physics.py::_physics_features itself falls back at
        fit/forecast time for the same columns - dispatch-time and fit-time
        behavior stay consistent when a weather source doesn't provide one
        of these.
        """
        values = self._get_clean_list(column, data_opt)
        if not values or all(x is None for x in values):
            return np.full(n, default)
        arr = np.array([default if (x is None or pd.isna(x)) else float(x) for x in values])
        if len(arr) < n:
            arr = np.concatenate((arr, np.full(n - len(arr), default)))
        return arr[:n]

    def _resolve_room_solar_irradiance(
        self, data_opt: pd.DataFrame, hc: dict, n: int, window_area, blind_position: float
    ) -> np.ndarray | None:
        """Effective (shading-adjusted) solar irradiance for a physics-family
        room's calculate_heating_demand_physics call, replacing a raw GHI
        reading with a direct/diffuse decomposition (utils.calculate_shaded_window_irradiance)
        - direct component only ever attenuated by shading, diffuse never is.

        blind_position is passed in explicitly rather than read from `hc`
        here, since it may be a live per-solve override
        (command_line.py::_build_room_blind_positions) that only reaches the
        two cold-build call sites (Sites B/C) - the cache-hit refresh path
        (update_thermal_params, Site A) always passes hc.get("blind_position", 0.0)
        (static, effectively "open/no shading") for the same reason
        def_init_temp doesn't reach that path either - see perform_optimization's
        own docstring for that precedent.

        Returns None when window_area isn't configured, matching the
        existing "only compute solar if window_area is set" guard at every
        call site.
        """
        if window_area is None:
            return None
        dni_arr = self._get_clean_weather_col(data_opt, "dni", n, default=0.0)
        dhi_arr = self._get_clean_weather_col(data_opt, "dhi", n, default=0.0)
        elev_arr = self._get_clean_weather_col(data_opt, "solar_elevation", n, default=0.0)
        blind_type = hc.get("blind_type", "none")
        return utils.calculate_shaded_window_irradiance(
            dni_arr, dhi_arr, float(blind_position), blind_type, elev_arr
        )

    def _prepare_power_limit_array(self, limit_value, limit_name, data_length):
        """
        Convert power limit to numpy array for time-varying constraints.

        Args:
            limit_value: Scalar, list, or array of power limit values
            limit_name: Name of the limit (for logging)
            data_length: Expected length of optimization horizon

        Returns:
            numpy.ndarray: Array of power limits with length = data_length
        """
        if limit_value is None:
            self.logger.error(f"{limit_name} is None, using default value 9000 W")
            return np.full(data_length, 9000.0)

        # Convert to numpy array if it's a list
        if isinstance(limit_value, list):
            limit_array = np.array(limit_value, dtype=float)
        elif isinstance(limit_value, np.ndarray):
            limit_array = limit_value.astype(float)
        else:
            # Scalar value - broadcast to all timesteps
            return np.full(data_length, float(limit_value))

        # Validate length
        if len(limit_array) != data_length:
            self.logger.warning(
                f"{limit_name} length ({len(limit_array)}) doesn't match "
                f"optimization horizon ({data_length}). Using scalar from first value."
            )
            return np.full(data_length, float(limit_array[0]) if len(limit_array) > 0 else 9000.0)

        self.logger.info(f"{limit_name} configured as time-varying with {data_length} values")
        return limit_array

    def _setup_stress_cost(self, cost_conf_key, max_power, var_name_prefix):
        """
        Generic setup for a stress cost (battery or inverter).
        """
        stress_unit_cost = self.plant_conf.get(cost_conf_key, 0)
        active = stress_unit_cost > 0 and max_power > 0

        stress_cost_var = None
        if active:
            self.logger.debug(
                f"Stress cost enabled for {var_name_prefix}. "
                f"Unit Cost: {stress_unit_cost}/kWh at full load {max_power}W."
            )
            stress_cost_var = cp.Variable(
                self.num_timesteps, nonneg=True, name=f"{var_name_prefix}_stress_cost"
            )

        return {
            "active": active,
            "vars": stress_cost_var,
            "unit_cost": stress_unit_cost,
            "max_power": max_power,
            # Defaults to 10 segments if not provided in config
            "segments": self.plant_conf.get(f"{var_name_prefix}_stress_segments", 10),
        }

    def _build_stress_segments(self, max_power, stress_unit_cost, segments):
        """
        Generic builder for Piece-Wise Linear segments for a quadratic cost curve.
        """
        # Cost rate at nominal power (currency/hr)
        max_cost_rate_hr = (max_power / 1000.0) * stress_unit_cost
        max_cost_step = max_cost_rate_hr * self.time_step

        x_points = np.linspace(0, max_power, segments + 1)
        y_points = max_cost_step * (x_points / max_power) ** 2

        seg_params = []
        for k in range(segments):
            x0, x1 = x_points[k], x_points[k + 1]
            y0, y1 = y_points[k], y_points[k + 1]
            slope = (y1 - y0) / (x1 - x0)
            intercept = y0 - slope * x0
            seg_params.append((slope, intercept))
        return seg_params

    def _add_stress_constraints(self, constraints, power_expression, stress_var, seg_params):
        """
        Generic constraint adder for stress costs (Vectorized).

        :param constraints: List to append constraints to
        :param power_expression: CVXPY expression (vector) for the power to be penalized
        :param stress_var: CVXPY variable (vector) for the stress cost
        :param seg_params: List of (slope, intercept) tuples
        """
        for slope, intercept in seg_params:
            # Vectorized constraints for both positive and negative directions (symmetry).
            # This creates a convex envelope around |power_expression|.
            constraints.append(stress_var >= slope * power_expression + intercept)
            constraints.append(stress_var >= -slope * power_expression + intercept)

    def _get_clean_list(self, key, data_opt):
        """Helper to extract list from DataFrame/Series/List safely."""
        val = data_opt.get(key)
        if hasattr(val, "values"):
            return val.values.tolist()
        return val if isinstance(val, list) else []

    def _get_capacity_cost_per_kw(self):
        """Capacity / demand charge rate (currency per kW) for issue #623,
        coerced to a non-negative float.

        ``capacity_cost_per_kw`` is runtime-overridable (see associations.csv),
        and runtime params are copied verbatim, so an HA template typically
        delivers it as a string. Coerce defensively and fall back to 0.0 (feature
        off) on a missing, non-numeric, non-finite or negative value rather than letting
        the ``> 0`` gate crash the problem build.
        """
        raw = self.optim_conf.get("capacity_cost_per_kw", 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self.logger.warning(
                f"Invalid capacity_cost_per_kw value ({raw!r}); "
                "ignoring it (no capacity charge applied)."
            )
            return 0.0
        # not isfinite(...) catches NaN and +/-inf; the second clause catches
        # negatives. An HA template can deliver any of these (incl. the string
        # "inf"), and cvxpy rejects a non-finite value, so fall back to 0.0.
        if not isfinite(value) or value < 0:
            self.logger.warning(
                f"capacity_cost_per_kw must be a finite number >= 0, got {raw!r}; "
                "ignoring it (no capacity charge applied)."
            )
            return 0.0
        return value

    def _initialize_decision_variables(self):
        """
        Initialize all main decision variables for the CVXPY problem.

        Returns:
            vars_dict: Dictionary containing cvxpy Variables
            constraints: List of bounds constraints associated with these variables
        """
        vars_dict = {}
        constraints = []
        n = self.num_timesteps

        # Prepare Power Limits
        max_power_from_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        max_power_to_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Grid power variables
        # P_grid_neg <= 0
        vars_dict["p_grid_neg"] = cp.Variable(n, nonpos=True, name="p_grid_neg")
        # Apply vectorized lower bound constraint
        constraints.append(vars_dict["p_grid_neg"] >= -max_power_to_grid_arr)

        # P_grid_pos >= 0
        vars_dict["p_grid_pos"] = cp.Variable(n, nonneg=True, name="p_grid_pos")
        # Apply vectorized upper bound constraint
        constraints.append(vars_dict["p_grid_pos"] <= max_power_from_grid_arr)

        # Deferrable load variables
        num_deferrable_loads = self.optim_conf["number_of_deferrable_loads"]
        p_deferrable = []
        p_def_bin1 = []
        p_def_start = []
        p_def_bin2 = []
        # p_def_stop[k]: falling-edge binary (1 = load turned OFF at timestep t).
        # Only created (non-None) for loads where def_minimum_off_time[k] > 0.
        # Mirrored to p_def_start; default None = inactive (no min-off constraint).
        p_def_stop = [None] * num_deferrable_loads

        for k in range(num_deferrable_loads):
            # Calculate Upper Bound
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                up_bound = np.max(self.optim_conf["nominal_power_of_deferrable_loads"][k])
            else:
                up_bound = self.optim_conf["nominal_power_of_deferrable_loads"][k]

            # Continuous/Semi-Continuous Power Variable
            var_p_def = cp.Variable(n, nonneg=True, name=f"p_deferrable_{k}")
            p_deferrable.append(var_p_def)

            # Global upper bound (specific semi-continuous logic handled in constraints)
            constraints.append(var_p_def <= up_bound)

            # Binary Variables
            p_def_bin1.append(cp.Variable(n, boolean=True, name=f"p_def_bin1_{k}"))
            p_def_start.append(cp.Variable(n, boolean=True, name=f"p_def_start_{k}"))
            p_def_bin2.append(cp.Variable(n, boolean=True, name=f"p_def_bin2_{k}"))

        vars_dict["p_deferrable"] = p_deferrable
        vars_dict["p_def_bin1"] = p_def_bin1
        vars_dict["p_def_start"] = p_def_start
        vars_dict["p_def_bin2"] = p_def_bin2
        vars_dict["p_def_stop"] = p_def_stop
        vars_dict["group_activity"] = {}

        # Binary indicators for Grid and Battery direction
        vars_dict["D"] = cp.Variable(n, boolean=True, name="D")

        # Battery power variables (#610: one set PER BATTERY, k in
        # range(self.n_batt), mirroring the deferrable-load f"..._{k}" naming
        # idiom at the top of this method). "E" (direction binary) is likewise
        # per-battery; "D" (grid direction) above stays a single shared binary
        # - there is only one grid connection regardless of battery count.
        if self.optim_conf["set_use_battery"]:
            vars_dict["E"] = [
                cp.Variable(n, boolean=True, name=f"E_{k}") for k in range(self.n_batt)
            ]
            vars_dict["p_sto_pos"] = []
            vars_dict["p_sto_neg"] = []
            vars_dict["soc_low_recovered"] = []
            vars_dict["soc_high_recovered"] = []
            vars_dict["soc_deficit_cost"] = []
            vars_dict["soc_surplus_cost"] = []
            for k in range(self.n_batt):
                p_sto_pos_k = cp.Variable(n, nonneg=True, name=f"p_sto_pos_{k}")
                constraints.append(p_sto_pos_k <= self.param_battery_discharge_power_max[k])
                vars_dict["p_sto_pos"].append(p_sto_pos_k)

                p_sto_neg_k = cp.Variable(n, nonpos=True, name=f"p_sto_neg_{k}")
                constraints.append(p_sto_neg_k >= -self.param_battery_charge_power_max[k])
                vars_dict["p_sto_neg"].append(p_sto_neg_k)

                vars_dict["soc_low_recovered"].append(
                    cp.Variable(n, boolean=True, name=f"soc_low_recovered_{k}")
                )
                vars_dict["soc_high_recovered"].append(
                    cp.Variable(n, boolean=True, name=f"soc_high_recovered_{k}")
                )
                vars_dict["soc_deficit_cost"].append(
                    cp.Variable(n, nonneg=True, name=f"soc_deficit_cost_{k}")
                )
                vars_dict["soc_surplus_cost"].append(
                    cp.Variable(n, nonneg=True, name=f"soc_surplus_cost_{k}")
                )
            # Terminal-SoC slacks: the signed miss on the horizon's net energy change,
            # split into two non-negative parts so the deviation stays linear. Priced in
            # the objective rather than forbidden, so an unreachable soc_final degrades
            # to "as close as allowed" instead of infeasible. One pair PER BATTERY
            # (#610): each battery's own target relaxes independently; an aggregate
            # slack would let one battery's overshoot cancel another's undershoot
            # at zero cost.
            vars_dict["soc_final_under"] = [
                cp.Variable(nonneg=True, name=f"soc_final_under_{k}") for k in range(self.n_batt)
            ]
            vars_dict["soc_final_over"] = [
                cp.Variable(nonneg=True, name=f"soc_final_over_{k}") for k in range(self.n_batt)
            ]
            # Battery-first priority gate (issue #834): binary per timestep,
            # 1 = grid import is "free" (unpenalized) in this slot. Only created
            # when the feature is enabled; otherwise it never enters self.vars.
            # battery_first_penalty (issue #1002): nonneg slack = the amount of
            # grid import that happens while the battery is still charged (the
            # gate is 0). Penalized in the objective instead of forbidden, so the
            # feature can never make the problem infeasible. Stays a SINGLE
            # aggregate gate/penalty for N batteries (#610): it gates on
            # aggregate stored energy vs aggregate minimum, not per-battery.
            if self.optim_conf.get("set_battery_first_priority", False):
                vars_dict["battery_first_import_gate"] = cp.Variable(
                    n, boolean=True, name="battery_first_import_gate"
                )
                vars_dict["battery_first_penalty"] = cp.Variable(
                    n, nonneg=True, name="battery_first_penalty"
                )
        else:
            # Create dummy zero variables to preserve logic structure without
            # conditional checks everywhere. A SINGLE dummy set regardless of
            # self.n_batt (#610): downstream code that must stay branch-free
            # (the power balance / DC-bus sums) iterates over the actual list
            # length rather than self.n_batt, so a 1-element all-zero list
            # contributes exactly zero either way.
            vars_dict["E"] = [cp.Variable(n, boolean=True, name="E_dummy")]
            vars_dict["p_sto_pos"] = [cp.Variable(n, name="p_sto_pos_dummy")]
            vars_dict["p_sto_neg"] = [cp.Variable(n, name="p_sto_neg_dummy")]
            constraints.append(vars_dict["p_sto_pos"][0] == 0)
            constraints.append(vars_dict["p_sto_neg"][0] == 0)
            vars_dict["soc_low_recovered"] = [cp.Variable(n, name="soc_low_recovered_dummy")]
            vars_dict["soc_high_recovered"] = [cp.Variable(n, name="soc_high_recovered_dummy")]
            constraints.append(vars_dict["soc_low_recovered"][0] == 0)
            constraints.append(vars_dict["soc_high_recovered"][0] == 0)
            vars_dict["soc_deficit_cost"] = [cp.Variable(n, name="soc_deficit_cost_dummy")]
            constraints.append(vars_dict["soc_deficit_cost"][0] == 0)
            vars_dict["soc_surplus_cost"] = [cp.Variable(n, name="soc_surplus_cost_dummy")]
            constraints.append(vars_dict["soc_surplus_cost"][0] == 0)

        # Self-consumption variable
        if self.costfun == "self-consumption":
            vars_dict["SC"] = cp.Variable(n, nonneg=True, name="SC")

        # Hybrid Inverter variable
        if self.plant_conf["inverter_is_hybrid"]:
            vars_dict["p_hybrid_inverter"] = cp.Variable(n, name="p_hybrid_inverter")

        # Curtailment variable
        vars_dict["p_pv_curtailment"] = cp.Variable(n, nonneg=True, name="p_pv_curtailment")

        # Peak grid-import variable for the capacity / demand charge (issue #623).
        # Opt-in: only created when capacity_cost_per_kw > 0, so when the feature
        # is off the problem is byte-identical to before (no extra variable, no
        # constraint, no objective term). peak_import (W) is a single scalar
        # bounded below by every grid-import timestep, i.e. the epigraph of
        # max(p_grid_pos) over the horizon; the cost on it is added in
        # _build_objective_function. The gate is a static config value so it is
        # part of the OptimizationCache key (a change rebuilds the problem).
        if self._get_capacity_cost_per_kw() > 0:
            vars_dict["peak_import"] = cp.Variable(nonneg=True, name="peak_import")
            constraints.append(vars_dict["peak_import"] >= vars_dict["p_grid_pos"])
            # Floor peak_import at any demand already incurred this billing period
            # (issue #623, Phase 2). With the floor binding, shaving below it has
            # zero marginal value, so the solver does not waste battery /
            # deferrable flexibility on a peak already locked in for the month.
            # The value is a cp.Parameter (W, default 0.0) so it is updated per
            # call without a rebuild (DPP / warm-start safe); default 0.0 makes
            # this redundant with the nonneg bound and the epigraph above, so the
            # plan is identical to Phase 1.
            constraints.append(vars_dict["peak_import"] >= self.param_current_period_peak)

        # Sum of deferrable loads ON THE ELECTRIC BUS. A load flagged with
        # is_electric_load[k] = False (gas boiler, oil burner, district
        # heating) provides heat to its thermal target but does NOT draw
        # electricity from the grid - its p_deferrable is in input-power-
        # equivalent units (gas burn rate, in W) and feeds the thermal
        # balance only. Excluding it from p_def_sum keeps the electric
        # balance honest (no phantom grid draw when the boiler fires).
        is_electric = self.optim_conf.get("is_electric_load", [True] * num_deferrable_loads)
        if num_deferrable_loads > 0:
            electric_loads = [
                p_deferrable[k]
                for k in range(num_deferrable_loads)
                if k >= len(is_electric) or bool(is_electric[k])
            ]
            vars_dict["p_def_sum"] = sum(electric_loads) if electric_loads else np.zeros(n)
        else:
            vars_dict["p_def_sum"] = np.zeros(n)

        return vars_dict, constraints

    def _build_objective_function(
        self,
        batt_stress_conf,
        inv_stress_conf,
        type_self_conso="bigm",
    ):
        """
        Construct the objective function based on configuration using vectorized CVXPY operations.
        Returns a CVXPY expression to be Maximized.
        """
        # Retrieve variables from self.vars (populated in _initialize_decision_variables)
        p_grid_pos = self.vars["p_grid_pos"]
        p_grid_neg = self.vars["p_grid_neg"]
        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        p_def_sum = self.vars["p_def_sum"]
        SC = self.vars.get("SC", None)

        # Retrieve parameters (vectors of length N)
        unit_load_cost = self.param_load_cost
        unit_prod_price = self.param_prod_price
        p_load = self.param_load_forecast

        # Common scaling factor
        # We maximize the negative cost (which is equivalent to minimizing cost)
        # or maximize Profit.
        scale = 0.001 * self.time_step

        # Initialize objective expression
        objective_terms = []

        # Base Cost Function
        if self.costfun == "profit":
            # Profit = Export Income - Import Cost
            # formulated as: -Cost - (Export_Neg_Value * Price)
            # Since p_grid_neg is negative, (Export_Neg * Price) is negative (cost-like).
            # We want to Maximize: -(ImportCost + ExportNeg*Price)
            # = -ImportCost + (-ExportNeg)*Price  <-- Positive Income

            if self.optim_conf["set_total_pv_sell"]:
                # Cost depends on Total Load (Load + Def)
                cost_term = cp.multiply(unit_load_cost, p_load + p_def_sum)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))
            else:
                # Cost depends on Grid Import
                cost_term = cp.multiply(unit_load_cost, p_grid_pos)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))

        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                cost_term = cp.multiply(unit_load_cost, p_load + p_def_sum)
                objective_terms.append(-scale * cp.sum(cost_term))
            else:
                cost_term = cp.multiply(unit_load_cost, p_grid_pos)
                objective_terms.append(-scale * cp.sum(cost_term))

        elif self.costfun == "self-consumption":
            if type_self_conso == "bigm":
                bigm = 1e3
                cost_term = bigm * cp.multiply(unit_load_cost, p_grid_pos)
                prod_term = cp.multiply(unit_prod_price, p_grid_neg)
                objective_terms.append(-scale * cp.sum(cost_term + prod_term))
            elif type_self_conso == "maxmin":
                # Maximize SC
                objective_terms.append(scale * cp.sum(cp.multiply(unit_load_cost, SC)))

        # Battery Cycle Cost and SOC Penalty (#610: summed over every battery
        # k, each with its own weight_dis[k]/weight_chg[k] - a flat scalar or
        # time-series per battery, sliced/broadcast exactly like the single-
        # battery code did on the whole config value).
        if self.optim_conf["set_use_battery"]:
            batt_conf = self._battery_conf_as_lists()
            cycle_cost_terms = []
            for k in range(len(p_sto_pos)):
                # p_sto_neg is negative. -weight*p_sto_neg is a positive penalty value.
                # We subtract this positive penalty from the maximization objective.
                weight_dis_k = batt_conf["weight_dis"][k]
                weight_chg_k = batt_conf["weight_chg"][k]

                # Handle time-varying weights with slicing for resized horizons
                if (
                    isinstance(weight_dis_k, list | np.ndarray)
                    and len(weight_dis_k) > self.num_timesteps
                ):
                    weight_dis_k = weight_dis_k[: self.num_timesteps]
                if (
                    isinstance(weight_chg_k, list | np.ndarray)
                    and len(weight_chg_k) > self.num_timesteps
                ):
                    weight_chg_k = weight_chg_k[: self.num_timesteps]

                cycle_cost_terms.append(
                    cp.multiply(np.array(weight_dis_k), p_sto_pos[k])
                    - cp.multiply(np.array(weight_chg_k), p_sto_neg[k])
                )
            objective_terms.append(-scale * cp.sum(sum(cycle_cost_terms)))

            # Multi-battery symmetry-breaking tie-break (#610). Skipped at
            # n_batt == 1, where the k==0 term would be an exact-zero no-op.
            # p_sto_neg is negative-signed, so (p_sto_pos - p_sto_neg) is total
            # throughput and this term penalizes higher-index usage in BOTH
            # directions: charge and discharge ties alike resolve to the
            # lowest-index battery. Deterministic, and never overrides a real
            # cost/efficiency difference (magnitude derivation on
            # BATTERY_TIEBREAK_EPS at the top of this module).
            if self.n_batt > 1:
                tiebreak_terms = [
                    k * cp.sum(p_sto_pos[k] - p_sto_neg[k]) for k in range(len(p_sto_pos))
                ]
                objective_terms.append(-BATTERY_TIEBREAK_EPS * cp.sum(sum(tiebreak_terms)))

        # Deferrable Load Startup Penalties
        if (
            "set_deferrable_startup_penalty" in self.optim_conf
            and self.optim_conf["set_deferrable_startup_penalty"]
        ):
            p_def_start = self.vars["p_def_start"]
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                penalty = self.optim_conf["set_deferrable_startup_penalty"][k]
                if penalty > 0:
                    nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                    # Vectorized cost calculation for this load's startups
                    startup_cost_vector = cp.multiply(p_def_start[k], unit_load_cost)
                    total_startup_cost = cp.sum(startup_cost_vector)

                    term = -scale * penalty * nominal_power * total_startup_cost
                    objective_terms.append(term)

        # Deferrable Load Max Cost Rewards
        # Add reward for scheduling loads equal to the max_cost..
        # Solver will only schedule if it can do so at lower cost than this reward.
        if hasattr(self, "deferrable_with_max_cost"):
            for k, (max_cost, load_is_scheduled) in self.deferrable_with_max_cost.items():
                # Add reward term: +max_cost * load_is_scheduled
                # This means: if solver schedules the load (load_is_scheduled=1),
                # it gets a reward of 'max_cost'
                reward_term = max_cost * load_is_scheduled
                objective_terms.append(reward_term)

                self.logger.debug(
                    f"Deferrable load {k}: added max cost reward of {max_cost} to objective"
                )

        # Per-load cost overrides. Behavior depends on whether the load is on
        # the electric balance (`is_electric_load[k]`):
        #
        # - Electric load (default): the load's power already enters p_def_sum
        #   and gets charged at unit_load_cost. Apply an ADJUSTMENT term
        #   `(per_load_cost - load_cost) * p_deferrable[k]` so the net cost
        #   becomes `per_load_cost * p_deferrable[k]` instead of the global
        #   retail tariff.
        #
        # - Non-electric load (gas / oil / district): the load was excluded
        #   from p_def_sum, so the base electric cost charges it nothing.
        #   Add the DIRECT cost `per_load_cost * p_deferrable[k]` instead of
        #   an adjustment - otherwise the (cheap_gas - retail) adjustment
        #   becomes a subsidy that pays the optimizer to fire gas.
        if self.costfun in ("profit", "cost") and self.param_cost_per_load:
            p_deferrable = self.vars.get("p_deferrable", None)
            is_electric = self.optim_conf.get(
                "is_electric_load",
                [True] * len(self.param_cost_per_load),
            )
            if p_deferrable is not None:
                for k, param_cost in enumerate(self.param_cost_per_load):
                    if k >= len(p_deferrable):
                        break
                    k_is_electric = k >= len(is_electric) or bool(is_electric[k])
                    if k_is_electric:
                        # Electric load - adjust away the global tariff
                        per_load_term = cp.multiply(param_cost - unit_load_cost, p_deferrable[k])
                    else:
                        # Non-electric load - charge directly at its commodity rate
                        per_load_term = cp.multiply(param_cost, p_deferrable[k])
                    objective_terms.append(-scale * cp.sum(per_load_term))

        # Stress Costs
        # These variables represent a cost to be minimized.
        # Since we are Maximizing the objective, we subtract them.
        if inv_stress_conf and inv_stress_conf["active"]:
            objective_terms.append(-cp.sum(inv_stress_conf["vars"]))

        # batt_stress_conf is now a list of one per-battery stress config dict
        # (#610); each entry is only "active" (has a Variable) when that
        # battery's own battery_stress_cost > 0.
        if batt_stress_conf:
            active_batt_stress_vars = [c["vars"] for c in batt_stress_conf if c["active"]]
            if active_batt_stress_vars:
                self.logger.debug("Adding battery stress cost to objective function")
                objective_terms.append(-cp.sum(sum(active_batt_stress_vars)))

        # SOC Deficit Cost (convert to per Wh) - summed over every battery
        if self.optim_conf["set_use_battery"]:
            soc_deficit_cost = self.vars.get("soc_deficit_cost")
            if soc_deficit_cost is not None:
                self.logger.debug(
                    f"Adding SOC deficit cost {soc_deficit_cost}  to objective function: "
                )
                objective_terms.append(-cp.sum(sum(soc_deficit_cost)))

        # SOC Surplus Cost (high-SoC dwell penalty, mirror of the deficit term)
        if self.optim_conf["set_use_battery"]:
            soc_surplus_cost = self.vars.get("soc_surplus_cost")
            if soc_surplus_cost is not None:
                self.logger.debug(
                    f"Adding SOC surplus cost {soc_surplus_cost}  to objective function: "
                )
                objective_terms.append(-cp.sum(sum(soc_surplus_cost)))

        # Terminal-SoC deviation penalty. param_soc_final_penalty is already in currency
        # per Wh (it folds in the kWh conversion and the dominance factor), so the slacks
        # enter the objective directly. Charging both directions keeps the target an
        # equality rather than a one-sided bound. Summed over the per-battery slack
        # pairs (#610); every battery's miss is priced at the same rate.
        soc_final_under = self.vars.get("soc_final_under")
        if soc_final_under is not None:
            objective_terms.append(
                -self.param_soc_final_penalty
                * (sum(soc_final_under) + sum(self.vars["soc_final_over"]))
            )

        # Battery-first priority penalty (issue #834/#1002). battery_first_penalty
        # is the grid import that occurs while the battery is still above its
        # minimum SoC. Priced at BATTERY_FIRST_IMPORT_PENALTY_FACTOR times the
        # import tariff so draining the battery first is preferred at any tariff
        # scale, while keeping the feature a soft penalty that can never make the
        # problem infeasible. Only present when the feature is enabled. The tariff
        # is clipped to non-negative (param_load_cost_pos): a negative-price slot
        # must not turn this penalty into an unbounded reward on the otherwise
        # upper-unbounded penalty variable.
        battery_first_penalty = self.vars.get("battery_first_penalty")
        if battery_first_penalty is not None:
            objective_terms.append(
                -scale
                * BATTERY_FIRST_IMPORT_PENALTY_FACTOR
                * cp.sum(cp.multiply(self.param_load_cost_pos, battery_first_penalty))
            )

        # Capacity / demand charge (issue #623). A one-time cost on the peak grid
        # import over the optimisation, priced in currency per kW. The peak_import
        # variable only exists when capacity_cost_per_kw > 0 (opt-in; default 0 is
        # a true no-op). This is a peak-POWER charge, so it is NOT scaled by
        # time_step the way the per-timestep energy terms are; peak_import is in W
        # and divided by 1000 to price it in kW. Subtracted because the objective
        # is maximised.
        capacity_cost_per_kw = self._get_capacity_cost_per_kw()
        if capacity_cost_per_kw > 0 and "peak_import" in self.vars:
            objective_terms.append(-capacity_cost_per_kw * (self.vars["peak_import"] / 1000.0))

        # Curtailment timing tie-break (issue #342). p_pv_curtailment carries no cost
        # of its own, so among equal-cost optima the solver may curtail early in the
        # horizon even when storing now and curtailing later is equally cheap. Add a
        # tiny time-decreasing penalty so the latest feasible timesteps are preferred.
        # The weight is normalized by the horizon length and the epsilon is orders of
        # magnitude below any real tariff coefficient, so it breaks ties without ever
        # flipping a real economic decision.
        if self.plant_conf["compute_curtailment"]:
            p_pv_curtailment = self.vars["p_pv_curtailment"]
            tiebreak_weights = np.arange(self.num_timesteps, 0, -1) / self.num_timesteps
            objective_terms.append(
                -CURTAILMENT_TIEBREAK_EPS * cp.sum(cp.multiply(tiebreak_weights, p_pv_curtailment))
            )

        # Sum all terms to create the final objective expression
        return cp.Maximize(cp.sum(objective_terms))

    def _add_main_power_balance_constraints(self, constraints):
        """Add the main power balance constraints (Vectorized)."""
        # Retrieve variables
        p_hybrid_inverter = self.vars.get("p_hybrid_inverter")
        p_def_sum = self.vars["p_def_sum"]
        p_grid_neg = self.vars["p_grid_neg"]
        p_grid_pos = self.vars["p_grid_pos"]
        p_pv_curtailment = self.vars["p_pv_curtailment"]
        # p_sto_pos/p_sto_neg are lists (#610), one entry per battery when
        # set_use_battery is on, a single always-zero dummy entry when off.
        # This choke point folds every battery's power into the shared
        # balance by summing over the ACTUAL list length (never self.n_batt
        # directly), so it stays branch-free regardless of whether the
        # battery feature is on.
        p_sto_pos_list = self.vars["p_sto_pos"]
        p_sto_neg_list = self.vars["p_sto_neg"]
        p_sto_pos_total = sum(p_sto_pos_list)
        p_sto_neg_total = sum(p_sto_neg_list)
        D = self.vars["D"]

        # Retrieve parameters
        p_pv = self.param_pv_forecast
        p_load = self.param_load_forecast

        # Prepare Time-Varying Limits
        # We re-calculate them here to ensure we use the correct time-varying limits
        n = self.num_timesteps
        max_power_from_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        max_power_to_grid_arr = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Main Power Balance Constraints
        if self.plant_conf["inverter_is_hybrid"]:
            constraints.append(
                p_hybrid_inverter - p_def_sum - p_load + p_grid_neg + p_grid_pos == 0
            )
        else:
            if self.plant_conf["compute_curtailment"]:
                constraints.append(
                    p_pv
                    - p_pv_curtailment
                    - p_def_sum
                    - p_load
                    + p_grid_neg
                    + p_grid_pos
                    + p_sto_pos_total
                    + p_sto_neg_total
                    == 0
                )
            else:
                constraints.append(
                    p_pv
                    - p_def_sum
                    - p_load
                    + p_grid_neg
                    + p_grid_pos
                    + p_sto_pos_total
                    + p_sto_neg_total
                    == 0
                )

        # Grid Constraints (Vectorized with Time-Varying Limits)
        # p_grid_pos <= max_from_grid[t] * D[t]
        constraints.append(p_grid_pos <= cp.multiply(max_power_from_grid_arr, D))

        # -p_grid_neg <= max_to_grid[t] * (1 - D[t])
        constraints.append(-p_grid_neg <= cp.multiply(max_power_to_grid_arr, (1 - D)))

    def _add_hybrid_inverter_constraints(self, constraints, inv_stress_conf):
        """Add constraints specific to hybrid inverters (Vectorized)."""
        if not self.plant_conf["inverter_is_hybrid"]:
            return

        # Retrieve main interface variables
        p_hybrid_inverter = self.vars["p_hybrid_inverter"]
        p_pv_curtailment = self.vars["p_pv_curtailment"]
        # #610: fold every battery's power into the DC-bus balance the same
        # way the main balance does - sum over the actual list length so the
        # off-case single dummy entry contributes exactly zero.
        p_sto_pos_total = sum(self.vars["p_sto_pos"])
        p_sto_neg_total = sum(self.vars["p_sto_neg"])
        p_pv = self.param_pv_forecast

        # Determine Inverter Capacity (Configuration Logic)
        p_nom_inverter_output = self.plant_conf.get("inverter_ac_output_max", None)
        p_nom_inverter_input = self.plant_conf.get("inverter_ac_input_max", None)

        # (Legacy lookup logic preserved but runs once during setup)
        if p_nom_inverter_output is None:
            if "pv_inverter_model" in self.plant_conf:
                if isinstance(self.plant_conf["pv_inverter_model"], list):
                    p_nom_inverter_output = 0.0
                    for i in range(len(self.plant_conf["pv_inverter_model"])):
                        if isinstance(self.plant_conf["pv_inverter_model"][i], str):
                            with bz2.BZ2File(
                                self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                                "rb",
                            ) as f:
                                cec_inverters = pickle.load(f)
                            inverter = cec_inverters[self.plant_conf["pv_inverter_model"][i]]
                            p_nom_inverter_output += inverter.Paco
                        else:
                            p_nom_inverter_output += self.plant_conf["pv_inverter_model"][i]
                else:
                    if isinstance(self.plant_conf["pv_inverter_model"], str):
                        with bz2.BZ2File(
                            self.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2",
                            "rb",
                        ) as f:
                            cec_inverters = pickle.load(f)
                        inverter = cec_inverters[self.plant_conf["pv_inverter_model"]]
                        p_nom_inverter_output = inverter.Paco
                    else:
                        p_nom_inverter_output = self.plant_conf["pv_inverter_model"]

            if p_nom_inverter_output is None:
                p_nom_inverter_output = 0  # Fallback

        if p_nom_inverter_input is None:
            p_nom_inverter_input = p_nom_inverter_output

        eff_dc_ac = self.plant_conf.get("inverter_efficiency_dc_ac", 1.0)
        eff_ac_dc = self.plant_conf.get("inverter_efficiency_ac_dc", 1.0)

        p_dc_ac_max = p_nom_inverter_output / eff_dc_ac
        p_ac_dc_max = p_nom_inverter_input * eff_ac_dc

        n = self.num_timesteps

        # Define Internal Variables
        # We define them here and attach to self.vars so they persist for result extraction
        p_dc_ac = cp.Variable(n, nonneg=True, name="p_dc_ac")
        p_ac_dc = cp.Variable(n, nonneg=True, name="p_ac_dc")
        is_dc_sourcing = cp.Variable(n, boolean=True, name="is_dc_sourcing")

        self.vars["p_dc_ac"] = p_dc_ac
        self.vars["p_ac_dc"] = p_ac_dc
        self.vars["is_dc_sourcing"] = is_dc_sourcing

        # Power Balance Constraints (Vectorized)

        # DC Bus Balance
        if self.plant_conf["compute_curtailment"]:
            e_dc_balance = (p_pv - p_pv_curtailment + p_sto_pos_total + p_sto_neg_total) - (
                p_dc_ac - p_ac_dc
            )
        else:
            e_dc_balance = (p_pv + p_sto_pos_total + p_sto_neg_total) - (p_dc_ac - p_ac_dc)

        constraints.append(e_dc_balance == 0)

        # AC Bus Balance
        # p_hybrid == converted_DC_to_AC - converted_AC_to_DC
        constraints.append(
            p_hybrid_inverter == (p_dc_ac * eff_dc_ac) - (p_ac_dc * (1.0 / eff_ac_dc))
        )

        # Enforce Binary Logic (Cannot source and sink DC simultaneously)
        constraints.append(p_ac_dc <= (1 - is_dc_sourcing) * p_ac_dc_max)
        constraints.append(p_dc_ac <= is_dc_sourcing * p_dc_ac_max)

        # Stress Cost
        if inv_stress_conf and inv_stress_conf["active"]:
            seg_params = self._build_stress_segments(
                inv_stress_conf["max_power"],
                inv_stress_conf["unit_cost"],
                inv_stress_conf["segments"],
            )
            self._add_stress_constraints(
                constraints,
                p_hybrid_inverter,  # Power expression
                inv_stress_conf["vars"],  # Stress variable
                seg_params,
            )

    def _add_battery_constraints(self, constraints, batt_stress_conf):
        """Add all battery-related constraints (Vectorized).

        #610: replicated per battery, k in range(self.n_batt) - this method
        only runs when set_use_battery is True (the early return below), so
        every list read here (self.vars["p_sto_pos"], etc.) is the real
        per-battery list, never the off-case single dummy. batt_stress_conf is
        a list of one per-battery stress config dict (see
        _setup_battery_stress_cost), aligned index-for-index with the battery
        lists. Two things stay a SINGLE shared quantity across the whole
        fleet, not per-battery: "D" (grid direction - one grid connection)
        and the battery-first priority gate/penalty (#610: gates on
        AGGREGATE stored energy vs aggregate minimum).
        """
        if not self.optim_conf["set_use_battery"]:
            return

        p_sto_pos = self.vars["p_sto_pos"]
        p_sto_neg = self.vars["p_sto_neg"]
        E = self.vars["E"]  # Binary per battery: 1=Discharge, 0=Charge
        D = self.vars["D"]  # Binary: 1=Import, 0=Export (shared - one grid connection)
        p_pv = self.param_pv_forecast

        batt_conf = self._battery_conf_as_lists()
        cap_list = batt_conf["cap"]
        eff_dis_list = batt_conf["eff_dis"]
        eff_chg_list = batt_conf["eff_chg"]
        soc_min_list = batt_conf["soc_min"]
        soc_max_list = batt_conf["soc_max"]

        # Grid Interaction Constraints (shared: one grid connection for the
        # whole fleet, so these sum battery power over k).

        # No charge from grid: total battery charge power cannot exceed PV production
        if self.optim_conf["set_nocharge_from_grid"]:
            constraints.append(sum(p_sto_neg) + p_pv >= 0)

        # No discharge to grid: prevent battery energy from reaching the grid. Hybrid inverters
        # prioritise PV to the load, so the battery cannot discharge while PV exports (strict E<=D, #796).
        # AC-coupled systems can, so E<=D wrongly forbids battery-to-load during export and makes the
        # solve infeasible when a large SoC must be shed (#936). For them bound grid export to the PV
        # *surplus* (max(0, PV - load), param_export_ceiling), not raw PV: bounding by raw PV lets the
        # battery cover the entire load and free PV for export, i.e. battery-to-grid through a PV detour
        # (#795, reintroduced by #981). Bounding by the surplus blocks battery-to-grid while still
        # allowing battery-to-load.
        if self.optim_conf["set_nodischarge_to_grid"]:
            if self.plant_conf["inverter_is_hybrid"]:
                for k in range(self.n_batt):
                    constraints.append(E[k] <= D)
            else:
                constraints.append(self.vars["p_grid_neg"] + self.param_export_ceiling >= 0)
                if self.plant_conf["compute_curtailment"]:
                    # Curtailed PV cannot be exported either. This stays a SEPARATE bound:
                    # folding p_pv_curtailment into the surplus ceiling would additionally
                    # cap curtailment itself at the surplus, an unrelated restriction that
                    # removes legitimate curtailment freedom (it may exceed the surplus)
                    # and breaks the #342 tie-break placement. Together the two bounds give
                    # export <= min(surplus, PV - curtailment), which is what we want.
                    constraints.append(
                        self.vars["p_grid_neg"] + p_pv - self.vars["p_pv_curtailment"] >= 0
                    )

        # Per-battery constraints. current_stored_energy_list is kept around
        # for the aggregate battery-first gate below. This SOC recursion is
        # hand-duplicated a second time in _build_results_dataframe (there in
        # numpy space over realized values, here as CVXPY expressions) and
        # must stay in lockstep with it.
        current_stored_energy_list = []
        for k in range(self.n_batt):
            cap = cap_list[k]
            eff_dis = eff_dis_list[k]
            eff_chg = eff_chg_list[k]
            max_dis = self.param_battery_discharge_power_max[k]
            max_chg = self.param_battery_charge_power_max[k]  # nonneg cp.Parameter
            soc_init_k = self.param_soc_init[k]
            soc_final_k = self.param_soc_final[k]
            soc_low_recovered_k = self.vars["soc_low_recovered"][k]
            soc_high_recovered_k = self.vars["soc_high_recovered"][k]
            min_energy = soc_min_list[k] * cap
            max_energy = soc_max_list[k] * cap
            recovery_margin = max(cap * 1e-6, 1e-3)
            recovery_big_m_low = cap - min_energy + recovery_margin
            recovery_big_m_high = max_energy + recovery_margin

            # Dynamic Power Limits (Ramp Rate) - each battery against ITS OWN power max
            if self.optim_conf["set_battery_dynamic"]:
                # Use slicing for vectorized ramp constraints: var[t+1] - var[t]
                # p_sto_pos ramp
                ramp_up_limit = self.time_step * self.optim_conf["battery_dynamic_max"] * max_dis
                ramp_down_limit = self.time_step * self.optim_conf["battery_dynamic_min"] * max_dis

                diff_pos = p_sto_pos[k][1:] - p_sto_pos[k][:-1]
                constraints.append(diff_pos <= ramp_up_limit)
                constraints.append(diff_pos >= ramp_down_limit)

                # p_sto_neg ramp (Note: p_sto_neg is negative, max_chg is positive magnitude)
                ramp_up_limit_neg = (
                    self.time_step * self.optim_conf["battery_dynamic_max"] * max_chg
                )
                ramp_down_limit_neg = (
                    self.time_step * self.optim_conf["battery_dynamic_min"] * max_chg
                )

                diff_neg = p_sto_neg[k][1:] - p_sto_neg[k][:-1]
                constraints.append(diff_neg <= ramp_up_limit_neg)
                constraints.append(diff_neg >= ramp_down_limit_neg)

            # Power & Binary Constraints
            # Discharge limit based on binary E[k]
            constraints.append(p_sto_pos[k] <= eff_dis * max_dis * E[k])

            # Charge limit based on binary E[k] (1-E[k])
            # p_sto_neg[k] >= -1/eff * max * (1-E[k])  --> (p_sto_neg is negative)
            constraints.append(p_sto_neg[k] >= -(1 / eff_chg) * max_chg * (1 - E[k]))

            # SOC Constraints (Vectorized Accumulation)

            # Calculate Energy Change per timestep (kWh)
            # Energy out = p_sto_pos / eff_dis
            # Energy in  = p_sto_neg * eff_chg  (p_sto_neg is negative, so this adds negative energy)
            power_flow = (p_sto_pos[k] * (1 / eff_dis)) + (p_sto_neg[k] * eff_chg)
            energy_change = power_flow * self.time_step

            # Calculate Cumulative Energy used/added
            cumulative_energy = cp.cumsum(energy_change)

            # SOC State (kWh) at every timestep t
            # SOC_t = SOC_init - Cumulative_Change
            # (Subtracting because positive flow is Discharge/Depletion)
            current_stored_energy = (soc_init_k * cap) - cumulative_energy
            current_stored_energy_list.append(current_stored_energy)

            # Min/Max SOC bounds with a single recovery transition.
            # Before recovery the trajectory stays on the initial out-of-band side.
            # After recovery the usual hard SOC limits apply and cannot be violated again.
            constraints.append(
                current_stored_energy
                >= min_energy - self.param_soc_low_gap[k] * (1 - soc_low_recovered_k)
            )
            constraints.append(
                current_stored_energy
                <= max_energy + self.param_soc_high_gap[k] * (1 - soc_high_recovered_k)
            )
            constraints.append(soc_low_recovered_k[1:] >= soc_low_recovered_k[:-1])
            constraints.append(soc_high_recovered_k[1:] >= soc_high_recovered_k[:-1])
            constraints.append(soc_low_recovered_k <= self.param_soc_low_required[k])
            constraints.append(soc_high_recovered_k <= self.param_soc_high_required[k])
            constraints.append(soc_low_recovered_k[-1] == self.param_soc_low_required[k])
            constraints.append(soc_high_recovered_k[-1] == self.param_soc_high_required[k])
            constraints.append(
                current_stored_energy[1:]
                >= current_stored_energy[:-1]
                - recovery_big_m_low
                * (soc_low_recovered_k[:-1] + (1 - self.param_soc_low_required[k]))
            )
            constraints.append(
                current_stored_energy[1:]
                <= current_stored_energy[:-1]
                + recovery_big_m_high
                * (soc_high_recovered_k[:-1] + (1 - self.param_soc_high_required[k]))
            )
            constraints.append(
                current_stored_energy
                <= min_energy
                - recovery_margin
                + recovery_big_m_low * soc_low_recovered_k
                + recovery_big_m_low * (1 - self.param_soc_low_required[k])
            )
            constraints.append(
                current_stored_energy
                >= max_energy
                + recovery_margin
                - recovery_big_m_high * soc_high_recovered_k
                - recovery_big_m_high * (1 - self.param_soc_high_required[k])
            )

            # Final SOC Constraint
            # The total energy change over the whole horizon should match init -> final:
            # Total Sum of power flow * dt == (Init - Final) * Capacity.
            # Enforced softly (see SOC_FINAL_DEVIATION_PENALTY_FACTOR): the two non-negative
            # slacks absorb any unreachable remainder and are charged in the objective, so
            # the equality still holds exactly whenever a schedule exists for it. Per
            # battery: each battery's own target relaxes independently.
            total_energy_change = cp.sum(energy_change)
            constraints.append(
                total_energy_change
                == (soc_init_k - soc_final_k) * cap
                + self.vars["soc_final_under"][k]
                - self.vars["soc_final_over"][k]
            )

            # Intermediate SOC target (issue #553): require SoC >= target at the
            # requested timestep, leaving the battery free to discharge afterward.
            # Per-battery precomputed floor vector; zero = no-op, so behaviour is
            # unchanged unless a target is explicitly requested. The identical
            # target fraction is currently applied to every battery's own
            # capacity; param_soc_target_floor is already a per-battery
            # Parameter so a future per-battery target only needs a different
            # value per entry, not a model change.
            constraints.append(current_stored_energy >= self.param_soc_target_floor[k])

            # Stress Cost (per battery: battery_stress_cost[k] gates this battery only)
            stress_conf_k = batt_stress_conf[k] if batt_stress_conf else None
            if stress_conf_k and stress_conf_k["active"]:
                seg_params = self._build_stress_segments(
                    stress_conf_k["max_power"],
                    stress_conf_k["unit_cost"],
                    stress_conf_k["segments"],
                )
                self._add_stress_constraints(
                    constraints,
                    p_sto_pos[k] - p_sto_neg[k],  # Total power magnitude expression
                    stress_conf_k["vars"],
                    seg_params,
                )

            # SOC Deficit Cost (per battery, own threshold/cost)
            soc_deficit_threshold = batt_conf["soc_deficit_threshold"][k]
            soc_deficit_cost_rate = batt_conf["soc_deficit_cost"][k] / 1000.0  # kWh to Wh
            if soc_deficit_threshold > 0 and soc_deficit_cost_rate > 0:
                threshold_energy = soc_deficit_threshold * cap
                soc_deficit_cost_k = self.vars["soc_deficit_cost"][k]
                constraints.append(
                    soc_deficit_cost_k
                    >= (threshold_energy - current_stored_energy)
                    * soc_deficit_cost_rate
                    * self.time_step
                )

            # SOC Surplus Cost (mirror of the deficit penalty above: penalize SoC
            # ABOVE a high threshold to discourage long dwell near full charge).
            soc_surplus_threshold = batt_conf["soc_surplus_threshold"][k]
            soc_surplus_cost_rate = batt_conf["soc_surplus_cost"][k] / 1000.0  # kWh to Wh
            if soc_surplus_threshold > 0 and soc_surplus_cost_rate > 0:
                threshold_energy = soc_surplus_threshold * cap
                soc_surplus_cost_k = self.vars["soc_surplus_cost"][k]
                constraints.append(
                    soc_surplus_cost_k
                    >= (current_stored_energy - threshold_energy)
                    * soc_surplus_cost_rate
                    * self.time_step
                )

        # Battery-first priority (issue #834): on a flat (non time-of-use)
        # tariff, "drain the battery before importing" and "interleave grid
        # import with discharge" are cost-equivalent, so the solver may plan
        # grid imports while the battery is still well above its minimum SoC.
        # When enabled, prefer to drain stored energy before importing. This uses
        # a dedicated binary gate, not the grid-direction binary D: with
        # set_nodischarge_to_grid the constraint E <= D would otherwise force the
        # battery to stop discharging, which is exactly what we want to avoid.
        #
        # This is a SOFT penalty, not a hard constraint (issue #1002). The gate is
        # forced to 0 in any slot where the battery is still above min SoC, and
        # any grid import in such a slot is charged battery_first_penalty and
        # penalized in the objective at BATTERY_FIRST_IMPORT_PENALTY_FACTOR times
        # the import tariff. That dwarfs any realistic tariff gradient, so the
        # solver still drains the battery before importing, but it can always fall
        # back to importing when that is the only feasible option (recharging to a
        # terminal SoC target with no PV, or a load that exceeds the battery's
        # discharge power) instead of returning infeasible as the old hard bound
        # `p_grid_pos <= max_from_grid * import_gate` did.
        #
        # #610: with N batteries there is one shared gate/penalty (not
        # per-battery), gated on AGGREGATE stored energy vs AGGREGATE minimum -
        # "is the fleet as a whole still above its combined floor". At N=1 the
        # sums below collapse to exactly the single-battery expressions.
        if self.optim_conf.get("set_battery_first_priority", False):
            import_gate = self.vars["battery_first_import_gate"]
            battery_first_penalty = self.vars["battery_first_penalty"]
            p_grid_pos = self.vars["p_grid_pos"]
            max_from_grid = self._prepare_power_limit_array(
                self.plant_conf.get("maximum_power_from_grid", 9000),
                "maximum_power_from_grid",
                self.num_timesteps,
            )
            aggregate_stored_energy = sum(current_stored_energy_list)
            aggregate_min_energy = sum(soc_min_list[k] * cap_list[k] for k in range(self.n_batt))
            aggregate_cap = sum(cap_list)
            # For a very lopsided fleet the 1% aggregate tolerance below can
            # exceed the smallest battery's entire usable SoC swing, so its
            # charge state barely moves the shared gate (see docs/config.md).
            if self.n_batt > 1:
                usable_swings = [
                    (soc_max_list[k] - soc_min_list[k]) * cap_list[k] for k in range(self.n_batt)
                ]
                min_swing = min(usable_swings)
                max_swing = max(usable_swings)
                if max_swing > 10 * min_swing:
                    ratio_txt = f"{max_swing / min_swing:.0f}x" if min_swing > 0 else "inf"
                    self.logger.warning(
                        "Batteries are very different in size (%s usable SoC swing); "
                        "the battery-first import gate tracks the fleet's aggregate "
                        "SoC, so the smaller battery may not be drained before grid "
                        "import is allowed. See docs/config.md.",
                        ratio_txt,
                    )
            # 1% aggregate-SoC tolerance so the gate opens cleanly once the
            # fleet has numerically reached its combined minimum, avoiding
            # chatter at the floor (mirrors the single-battery 1% tolerance).
            soc_tolerance_energy = 0.01 * aggregate_cap
            # import_gate = 1 (import unpenalized) is only possible once the
            # AGGREGATE stored energy is at/below the aggregate min + tolerance;
            # otherwise the gate is forced to 0 and any grid import in that slot
            # is penalized.
            constraints.append(
                aggregate_stored_energy - aggregate_min_energy - soc_tolerance_energy
                <= aggregate_cap * (1 - import_gate)
            )
            # battery_first_penalty >= import beyond the free (gated) allowance;
            # nonneg, so it equals max(0, import while the fleet is charged).
            constraints.append(
                battery_first_penalty >= p_grid_pos - cp.multiply(max_from_grid, import_gate)
            )

    def _add_thermal_load_constraints(self, constraints, k, data_opt, def_init_temp):
        """
        Handle constraints for thermal deferrable loads (Vectorized).
        Includes thermal inertia (lag) logic.
        Uses cp.Parameter for runtime values to enable warm-starting on cache hits.
        """
        p_deferrable = self.vars["p_deferrable"][k]
        p_def_bin2 = self.vars["p_def_bin2"][k]

        # Config retrieval
        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_config"]
        required_len = self.num_timesteps

        # Use parameterized values if available (enables warm-start on cache hit)
        if k in self.param_thermal:
            params = self.param_thermal[k]
            start_temperature = params["start_temp"]
            outdoor_temp = params["outdoor_temp"]
            min_temps_param = params["min_temps"]
            max_temps_param = params["max_temps"]
            desired_temps_param = params["desired_temps"]

            # Update param value if def_init_temp override is provided
            if def_init_temp[k] is not None:
                params["start_temp"].value = float(def_init_temp[k])

            # Initialize outdoor temp from data_opt (will be updated on subsequent calls)
            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)
            params["outdoor_temp"].value = outdoor_temp_arr

            # Initialize min/max/desired temps from config
            min_temps_list = hc.get("min_temperatures", [])
            max_temps_list = hc.get("max_temperatures", [])
            desired_temps_list = hc.get("desired_temperatures", [])
            params["min_temps"].value = self._pad_temp_array(min_temps_list, required_len, 18.0)
            params["max_temps"].value = self._pad_temp_array(max_temps_list, required_len, 26.0)
            params["desired_temps"].value = self._pad_temp_array(
                desired_temps_list, required_len, 22.0
            )
        else:
            # Fallback for loads not in param dict (shouldn't happen normally)
            start_temperature = (
                def_init_temp[k]
                if def_init_temp[k] is not None
                else hc.get("start_temperature", 20.0)
            )
            start_temperature = float(start_temperature) if start_temperature is not None else 20.0
            outdoor_temp = self._get_clean_outdoor_temp(data_opt, required_len)
            min_temps_param = None
            max_temps_param = None
            desired_temps_param = None

        # Constants (structural - don't change between MPC iterations)
        cooling_constant = hc["cooling_constant"]
        heating_rate = hc["heating_rate"]
        overshoot_temperature = hc.get("overshoot_temperature", None)
        sense = utils.normalize_heat_cool_mode(
            hc.get("sense") or "heat",
            field_name="sense",
            context=f"Load {k} thermal_config",
        )
        sense_coeff = 1 if sense == "heat" else -1
        nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]

        # Thermal Inertia Logic
        thermal_inertia = hc.get("thermal_inertia", 0.0)
        L = int(thermal_inertia / self.time_step)

        # Define Temperature State Variable
        predicted_temp = cp.Variable(required_len, name=f"temp_load_{k}")

        constraints.append(predicted_temp[0] == start_temperature)

        heat_factor = (heating_rate * self.time_step) / nominal_power
        cool_factor = cooling_constant * self.time_step

        # Main Dynamics (Delayed Power)
        # T[t+1] depends on T[t] and P[t-L]
        constraints.append(
            predicted_temp[1 + L :]
            == predicted_temp[L:-1]
            + (p_deferrable[: -1 - L] * sense_coeff * heat_factor)
            - (cool_factor * (predicted_temp[L:-1] - outdoor_temp[L:-1]))
        )

        # Startup "Dead Zone" Dynamics
        if L > 0:
            constraints.append(
                predicted_temp[1 : 1 + L]
                == predicted_temp[:L] - (cool_factor * (predicted_temp[:L] - outdoor_temp[:L]))
            )

        # Min/Max Temperature Constraints
        # Only add constraints if config actually specifies min/max temps
        # Skip index 0 (already constrained by start_temperature)
        min_temps_config = hc.get("min_temperatures", [])
        max_temps_config = hc.get("max_temperatures", [])

        if min_temps_config:
            if min_temps_param is not None:
                # Use parameter (allows warm-start updates), but only for valid config indices
                valid_indices = [
                    i
                    for i, v in enumerate(min_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    constraints.append(
                        predicted_temp[valid_indices] >= min_temps_param[valid_indices]
                    )
            else:
                valid_indices = [
                    i
                    for i, v in enumerate(min_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    limit_vals = np.array([min_temps_config[i] for i in valid_indices])
                    constraints.append(predicted_temp[valid_indices] >= limit_vals)

        if max_temps_config:
            if max_temps_param is not None:
                valid_indices = [
                    i
                    for i, v in enumerate(max_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    constraints.append(
                        predicted_temp[valid_indices] <= max_temps_param[valid_indices]
                    )
            else:
                valid_indices = [
                    i
                    for i, v in enumerate(max_temps_config)
                    if v is not None and i < required_len and i > 0
                ]
                if valid_indices:
                    limit_vals = np.array([max_temps_config[i] for i in valid_indices])
                    constraints.append(predicted_temp[valid_indices] <= limit_vals)

        # Overshoot Logic
        penalty_expr = 0
        desired_temps_list = hc.get("desired_temperatures", [])

        if desired_temps_list and overshoot_temperature is not None:
            is_overshoot = cp.Variable(required_len, boolean=True, name=f"is_overshoot_{k}")
            big_m = 100
            if sense == "heat":
                constraints.append(
                    predicted_temp - overshoot_temperature - (big_m * is_overshoot) <= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (big_m * (1 - is_overshoot)) >= 0
                )
            else:
                constraints.append(
                    predicted_temp - overshoot_temperature - (-big_m * is_overshoot) >= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (-big_m * (1 - is_overshoot)) <= 0
                )

            constraints.append(is_overshoot[1:] + p_def_bin2[:-1] <= 1)

            # Penalty Calculation
            # Filter for valid indices (not None, within bounds, skip index 0)
            penalty_factor = hc.get("penalty_factor", 10)
            valid_indices = [
                i
                for i, val in enumerate(desired_temps_list)
                if val is not None and i < required_len and i > 0
            ]
            if valid_indices:
                if desired_temps_param is not None:
                    # Use parameter for actual values (allows warm-start value updates)
                    deviation = (
                        predicted_temp[valid_indices] - desired_temps_param[valid_indices]
                    ) * sense_coeff
                else:
                    # Fallback to raw values
                    des_temps = np.array([desired_temps_list[i] for i in valid_indices])
                    deviation = (predicted_temp[valid_indices] - des_temps) * sense_coeff
                penalty_expr = -cp.pos(-deviation * penalty_factor)

        # Semi-Continuous Constraint
        if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
            constraints.append(p_deferrable == p_def_bin2 * nominal_power)

        total_penalty = cp.sum(penalty_expr) if not isinstance(penalty_expr, int) else 0
        return predicted_temp, None, total_penalty

    @staticmethod
    def _tile_profile(profile, required_len):
        """Tile a daily profile (e.g. draw-off demand) to fill the optimization horizon."""
        arr = np.array(profile, dtype=float)
        if len(arr) < required_len:
            repeats = int(np.ceil(required_len / len(arr)))
            arr = np.tile(arr, repeats)
        return arr[:required_len]

    def _resolve_draw_off_demand(self, hc, base_loss, required_len):
        """Return (demand_arr, loss_arr) if hot-water-tank mode (draw_off_demand present), else None."""
        draw_off_profile = hc.get("draw_off_demand", None)
        if draw_off_profile is not None and len(draw_off_profile) > 0:
            demand_arr = self._tile_profile(draw_off_profile, required_len)
            loss_arr = np.full(required_len, base_loss)
            return demand_arr, loss_arr
        return None

    def _resolve_boiler_hc_for_cop(self, k: int, hc: dict) -> dict:
        """For a "hp_tank_zone" boiler coupled to a real heat-pump load
        (boiler_coupled_heatpump_load_index) - same physical heat-pump unit,
        just diverting some of its output to the DHW tank instead of/alongside
        space heating - derive its COP from that heat pump's own
        carnot_efficiency instead of the boiler's flat placeholder value, so
        the same real-world efficiency factor applies to both. Deliberately
        keeps the boiler's own supply_temperature/heating_curve unchanged
        (the DHW tank's own target, typically higher than the space-heating
        supply temperature, so pumping to it should still show a worse COP
        than the coupled load's own).

        Returns hc unchanged when boiler_type isn't "hp_tank_zone" (including
        during a forced-resistive legionella cycle, where boiler_type is
        temporarily "resistive" and the constant-efficiency branch of
        resolve_thermal_battery_cop already applies regardless), when there's
        no valid coupling configured, or when the coupled load has no
        thermal_battery config with a carnot_efficiency to borrow.
        """
        if hc.get("boiler_type") != "hp_tank_zone":
            return hc
        coupled_idx = int(hc.get("coupled_heatpump_load_index", -1) or -1)
        def_load_config = self.optim_conf.get("def_load_config", [])
        if coupled_idx < 0 or coupled_idx == k or coupled_idx >= len(def_load_config):
            return hc
        coupled_hc = def_load_config[coupled_idx].get("thermal_battery")
        if not coupled_hc or "carnot_efficiency" not in coupled_hc:
            return hc
        return {**hc, "carnot_efficiency": coupled_hc["carnot_efficiency"]}

    def _apply_surface_solar_gain(self, hc, data_opt, heating_demand, required_len):
        """Subtract surface solar gain from `heating_demand` when configured.

        Single source of truth used by both the parameterized and fallback
        paths of `_add_thermal_battery_constraints`. No-op when
        `solar_absorption_area` is unset on `hc` or when `heating_demand` is
        None.
        """
        if heating_demand is None:
            return heating_demand
        ghi_arr = data_opt["ghi"].values if "ghi" in data_opt.columns else None
        solar_gain = utils.calculate_surface_solar_gain(
            hc,
            ghi_arr,
            optimization_time_step_minutes=int(self.freq.total_seconds() / 60),
            length=required_len,
        )
        if solar_gain is None:
            return heating_demand
        return heating_demand - solar_gain

    def _add_thermal_battery_constraints(
        self, constraints, k, data_opt, p_load, def_init_temp=None, coupling_flow_vars=None,
        room_blind_positions=None, room_opening_open=None,
    ):
        """
        Handle constraints for thermal battery loads (Vectorized, Legacy Match).
        Uses cp.Parameter for runtime values to enable warm-starting on cache hits.

        coupling_flow_vars: optional {(i, j): cp.Variable} dict of pre-created
        room-to-room thermal coupling flow variables (see
        _get_room_thermal_coupling_pairs/_add_room_thermal_coupling_constraints).
        When load k participates in any pair, its net outgoing flow is folded
        into k's own recurrence equation below - the flow variables' actual
        values are pinned separately, after every room's predicted_temps
        exists (see the caller).
        """
        p_deferrable = self.vars["p_deferrable"][k]

        def_load_config = self.optim_conf["def_load_config"][k]
        hc = def_load_config["thermal_battery"]
        required_len = self.num_timesteps

        # Structural parameters (don't change between MPC iterations).
        # supply_temperature / efficiency / heating_curve requirement is
        # validated by resolve_thermal_battery_cop further down (single
        # source of truth).
        volume = hc["volume"]
        min_temperatures_list = hc["min_temperatures"]
        max_temperatures_list = hc["max_temperatures"]

        if not min_temperatures_list:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'min_temperatures'")
        if not max_temperatures_list:
            raise ValueError(f"Load {k}: thermal_battery requires non-empty 'max_temperatures'")

        density = hc.get("density", 2400)  # kg/m^3 (default: concrete)
        heat_capacity = hc.get("heat_capacity", 0.88)  # kJ/(kg*degC) (default: concrete)
        base_loss = hc.get("thermal_loss", 0.045)  # kW (default: 0.045)
        if density <= 0 or heat_capacity <= 0 or volume <= 0:
            raise ValueError(
                f"Load {k}: thermal_battery requires positive density ({density}), "
                f"heat_capacity ({heat_capacity}), and volume ({volume})"
            )
        conversion = 3600 / (density * heat_capacity * volume)

        # Determine heat-flow direction: +1 for heating (pump adds heat), -1 for cooling (pump removes heat)
        sense = utils.normalize_heat_cool_mode(
            hc.get("sense") or "heat",
            field_name="sense",
            context=f"Load {k} thermal_battery",
        )
        sense_coeff = 1 if sense == "heat" else -1

        # Use parameterized values if available (enables warm-start on cache hit)
        if k in self.param_thermal:
            params = self.param_thermal[k]
            start_temperature = params["start_temp"]
            heatpump_cops = params["heatpump_cops"]
            thermal_losses = params["thermal_losses"]
            heating_demand = params["heating_demand"]
            min_temps_param = params["min_temps"]
            max_temps_param = params["max_temps"]

            # Update param value if def_init_temp override is provided (e.g. a
            # live HA room/heat-pump temperature sensor reading).
            if def_init_temp is not None and def_init_temp[k] is not None:
                params["start_temp"].value = float(def_init_temp[k])

            # Initialize parameter values from data_opt and config
            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)
            params["outdoor_temp"].value = outdoor_temp_arr
            start_temp_float = float(params["start_temp"].value)

            # Compute and set derived parameter values
            cop_hc = self._resolve_boiler_hc_for_cop(k, hc)
            cops = utils.resolve_thermal_battery_cop(cop_hc, outdoor_temp_arr, length=required_len)
            params["heatpump_cops"].value = np.array(cops)

            # Check for hot water tank mode (draw_off_demand present)
            # draw_off_demand units: kWh per timestep (same as heating_demand from
            # calculate_heating_demand / calculate_heating_demand_physics). This is
            # consistent with the thermal dynamics equation where all energy terms are
            # in kWh: conversion * (COP * P_kW * dt_hours - demand_kWh - loss_kWh)
            hot_water = self._resolve_draw_off_demand(hc, base_loss, required_len)
            # Explicit custom demand profile takes priority over both DHW draw-off
            # and the physics/degree-day models below.
            custom_demand = hc.get("custom_heating_demand_profile", None)
            if custom_demand is not None:
                demand = np.array(custom_demand, dtype=float)
                if len(demand) < required_len:
                    demand = np.concatenate((demand, np.zeros(required_len - len(demand))))
                params["heating_demand"].value = demand[:required_len]
                losses = utils.calculate_thermal_loss_signed(
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    indoor_temperature=start_temp_float,
                    base_loss=base_loss,
                )
                params["thermal_losses"].value = np.array(losses[:required_len])
            elif hot_water is not None:
                params["heating_demand"].value, params["thermal_losses"].value = hot_water
            else:
                losses = utils.calculate_thermal_loss_signed(
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    indoor_temperature=start_temp_float,
                    base_loss=base_loss,
                )
                params["thermal_losses"].value = np.array(losses[:required_len])

                # Compute heating demand
                if all(
                    key in hc
                    for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
                ):
                    indoor_target_temp = hc.get(
                        "indoor_target_temperature",
                        min_temperatures_list[0] if min_temperatures_list else 20.0,
                    )
                    window_area = hc.get("window_area", None)
                    shgc = hc.get("shgc", 0.6)
                    internal_gains_factor = hc.get("internal_gains_factor", 0.0)

                    internal_gains_forecast = p_load if internal_gains_factor > 0 else None
                    blind_position_k = (
                        room_blind_positions[k]
                        if room_blind_positions is not None
                        and k < len(room_blind_positions)
                        and room_blind_positions[k] is not None
                        else float(hc.get("blind_position", 0.0))
                    )
                    solar_irradiance = self._resolve_room_solar_irradiance(
                        data_opt, hc, required_len, window_area, blind_position_k
                    )

                    # Extra ventilation loss at the near-term step only while
                    # this room's window/door is open right now.
                    ventilation_rate_arr = np.full(required_len, hc["ventilation_rate"])
                    if (
                        room_opening_open is not None
                        and k < len(room_opening_open)
                        and room_opening_open[k]
                    ):
                        ventilation_rate_arr[0] += OPENING_EXTRA_ACH

                    demand = utils.calculate_heating_demand_physics(
                        u_value=hc["u_value"],
                        envelope_area=hc["envelope_area"],
                        ventilation_rate=ventilation_rate_arr,
                        heated_volume=hc["heated_volume"],
                        indoor_target_temperature=indoor_target_temp,
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        optimization_time_step=int(self.freq.total_seconds() / 60),
                        solar_irradiance_forecast=solar_irradiance,
                        window_area=window_area,
                        shgc=shgc,
                        internal_gains_forecast=internal_gains_forecast,
                        internal_gains_factor=internal_gains_factor,
                        sense=sense,
                    )
                    params["heating_demand"].value = np.array(demand[:required_len])

                    gains_info = []
                    if solar_irradiance is not None:
                        gains_info.append(f"solar (window_area={window_area:.1f}, shgc={shgc:.2f})")
                    if internal_gains_factor > 0:
                        gains_info.append(f"internal (factor={internal_gains_factor:.2f})")
                    gains_str = " with " + " and ".join(gains_info) if gains_info else ""
                    self.logger.debug(
                        "Load %s: Using physics-based heating demand%s "
                        "(u_value=%.2f, envelope_area=%.1f, ventilation_rate=%.2f, heated_volume=%.1f, "
                        "indoor_target_temp=%.1f)",
                        k,
                        gains_str,
                        hc["u_value"],
                        hc["envelope_area"],
                        hc["ventilation_rate"],
                        hc["heated_volume"],
                        indoor_target_temp,
                    )
                else:
                    base_temperature = hc.get("base_temperature", 18.0)
                    annual_reference_hdd = hc.get("annual_reference_hdd", 3000.0)
                    if sense == "cool":
                        self.logger.warning(
                            "Load %s: the degree-day (specific_heating_demand) "
                            "demand model is heating-only; sense='cool' will be "
                            "treated as heating. Configure the physics model "
                            "(u_value, envelope_area, ventilation_rate, "
                            "heated_volume) for cooling demand.",
                            k,
                        )
                    demand = utils.calculate_heating_demand(
                        specific_heating_demand=hc["specific_heating_demand"],
                        floor_area=hc["area"],
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        base_temperature=base_temperature,
                        annual_reference_hdd=annual_reference_hdd,
                        optimization_time_step=int(self.freq.total_seconds() / 60),
                    )
                    params["heating_demand"].value = np.array(demand[:required_len])

            # Surface solar gain (pool, outdoor tank, solar-thermal). Subtracts
            # absorbed irradiance from the residual heating demand. No-op when
            # solar_absorption_area is unset.
            params["heating_demand"].value = self._apply_surface_solar_gain(
                hc, data_opt, params["heating_demand"].value, required_len
            )

            # Set min/max temperature parameters
            min_temps_arr = self._pad_temp_array(min_temperatures_list, required_len, 18.0)
            max_temps_arr = self._pad_temp_array(max_temperatures_list, required_len, 26.0)
            opening_open_k = (
                room_opening_open is not None
                and k < len(room_opening_open)
                and room_opening_open[k]
            )
            min_temps_arr, max_temps_arr = self._relax_opening_temp_bounds(
                min_temps_arr, max_temps_arr, opening_open_k
            )
            params["min_temps"].value = min_temps_arr
            params["max_temps"].value = max_temps_arr

        else:
            # Fallback for loads not in param dict (shouldn't happen normally)
            start_temperature = (
                def_init_temp[k]
                if def_init_temp is not None and def_init_temp[k] is not None
                else hc.get("start_temperature", 20.0)
            )
            start_temperature = float(start_temperature) if start_temperature is not None else 20.0
            start_temp_float = start_temperature

            outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)

            cop_hc = self._resolve_boiler_hc_for_cop(k, hc)
            heatpump_cops = np.array(
                utils.resolve_thermal_battery_cop(cop_hc, outdoor_temp_arr, length=required_len)
            )

            # Check for hot water tank mode (draw_off_demand present)
            # draw_off_demand units: kWh per timestep (see parameterized path comment)
            hot_water = self._resolve_draw_off_demand(hc, base_loss, required_len)
            # Explicit custom demand profile takes priority over both DHW draw-off
            # and the physics/degree-day models below.
            custom_demand = hc.get("custom_heating_demand_profile", None)
            if custom_demand is not None:
                demand = np.array(custom_demand, dtype=float)
                if len(demand) < required_len:
                    demand = np.concatenate((demand, np.zeros(required_len - len(demand))))
                thermal_losses = np.array(
                    utils.calculate_thermal_loss_signed(
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        indoor_temperature=start_temp_float,
                        base_loss=base_loss,
                    )[:required_len]
                )
                heating_demand = np.array(demand[:required_len])
            elif hot_water is not None:
                heating_demand, thermal_losses = hot_water
            else:
                thermal_losses = np.array(
                    utils.calculate_thermal_loss_signed(
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        indoor_temperature=start_temp_float,
                        base_loss=base_loss,
                    )[:required_len]
                )

                # Compute heating demand (simplified fallback)
                if all(
                    key in hc
                    for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
                ):
                    indoor_target_temp = hc.get("indoor_target_temperature", 20.0)
                    # This fallback branch previously never computed solar
                    # gain at all (a pre-existing gap) - now matches the
                    # parameterized path (Site B) exactly, including
                    # direct/diffuse decomposition and blind-shading.
                    window_area = hc.get("window_area", None)
                    shgc = hc.get("shgc", 0.6)
                    blind_position_k = (
                        room_blind_positions[k]
                        if room_blind_positions is not None
                        and k < len(room_blind_positions)
                        and room_blind_positions[k] is not None
                        else float(hc.get("blind_position", 0.0))
                    )
                    solar_irradiance = self._resolve_room_solar_irradiance(
                        data_opt, hc, required_len, window_area, blind_position_k
                    )
                    # Extra ventilation loss at the near-term step only while
                    # this room's window/door is open right now (mirrors
                    # Site B exactly).
                    ventilation_rate_arr = np.full(required_len, hc["ventilation_rate"])
                    if (
                        room_opening_open is not None
                        and k < len(room_opening_open)
                        and room_opening_open[k]
                    ):
                        ventilation_rate_arr[0] += OPENING_EXTRA_ACH
                    demand = utils.calculate_heating_demand_physics(
                        u_value=hc["u_value"],
                        envelope_area=hc["envelope_area"],
                        ventilation_rate=ventilation_rate_arr,
                        heated_volume=hc["heated_volume"],
                        indoor_target_temperature=indoor_target_temp,
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        optimization_time_step=int(self.freq.total_seconds() / 60),
                        solar_irradiance_forecast=solar_irradiance,
                        window_area=window_area,
                        shgc=shgc,
                        sense=sense,
                    )
                else:
                    if sense == "cool":
                        self.logger.warning(
                            "Load %s: the degree-day (specific_heating_demand) "
                            "demand model is heating-only; sense='cool' will be "
                            "treated as heating. Configure the physics model "
                            "(u_value, envelope_area, ventilation_rate, "
                            "heated_volume) for cooling demand.",
                            k,
                        )
                    demand = utils.calculate_heating_demand(
                        specific_heating_demand=hc["specific_heating_demand"],
                        floor_area=hc["area"],
                        outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                        base_temperature=hc.get("base_temperature", 18.0),
                        annual_reference_hdd=hc.get("annual_reference_hdd", 3000.0),
                        optimization_time_step=int(self.freq.total_seconds() / 60),
                    )
                heating_demand = np.array(demand[:required_len])
            # Surface solar gain (fallback path - mirrors parameterized path).
            heating_demand = self._apply_surface_solar_gain(
                hc, data_opt, heating_demand, required_len
            )
            min_temps_param = None
            max_temps_param = None

        # Build constraints using parameters
        predicted_temp_thermal = cp.Variable(required_len, name=f"temp_thermal_batt_{k}")

        constraints.append(predicted_temp_thermal[0] == start_temperature)

        # Room-to-room thermal coupling: net outgoing flow for this room,
        # folded into its own recurrence below (subtracted, same sign
        # convention as thermal_losses). flow_var == g*dt*(T_i - T_j) is
        # pinned separately in _add_room_thermal_coupling_constraints, once
        # every room's predicted_temp_thermal exists - a positive flow means
        # the "i" side of the pair is losing heat to the "j" side.
        coupling_term = 0
        if coupling_flow_vars:
            for (i, j), flow_var in coupling_flow_vars.items():
                if i == k:
                    coupling_term = coupling_term + flow_var[:-1]
                elif j == k:
                    coupling_term = coupling_term - flow_var[:-1]

        # Thermal inertia: first-order low-pass filter on heat input
        tau_hours = float(hc.get("thermal_inertia_time_constant", 0.0) or 0.0)

        if tau_hours < 0:
            raise ValueError(
                f"Load {k}: thermal_inertia_time_constant must be >= 0, got {tau_hours}"
            )
        if tau_hours > 6:
            self.logger.warning(
                "Load %s: thermal_inertia_time_constant=%.1f h is large. "
                "Ensure this value reflects your system's dynamics.",
                k,
                tau_hours,
            )

        if tau_hours > 0:
            alpha = self.time_step / tau_hours
            if alpha > 1.0:
                self.logger.warning(
                    "Load %s: thermal_inertia_time_constant (%.2f h) < time_step (%.2f h), "
                    "clamping filter coefficient to 1.0.",
                    k,
                    tau_hours,
                    self.time_step,
                )
                alpha = 1.0

            # Q_input variable: filtered heat energy per timestep (kWh)
            q_input = cp.Variable(required_len, nonneg=True, name=f"q_input_{k}")

            # Initialize Q_input[0] from CVXPY Parameter (enables warm-start updates)
            params = self.param_thermal.get(k, {})
            q_input_start = params.get("q_input_start", 0.0)

            # Extract scalar values for the feasibility guard.
            q_start_val = 0.0
            if hasattr(q_input_start, "value") and q_input_start.value is not None:
                q_start_val = float(q_input_start.value)
            elif isinstance(q_input_start, int | float):
                q_start_val = float(q_input_start)

            # min_temperatures_list is guaranteed non-empty by the validator above.
            min_temp_0 = float(min_temperatures_list[0])

            if q_start_val < 1e-6 and start_temp_float <= min_temp_0:
                # When q_input_start is near zero AND temperature is at/below the
                # minimum, fixing q_input[0]=0 makes the problem infeasible because
                # the temperature would drop below min at the next timestep.
                # Let the solver choose a feasible initial heat input instead.
                self.logger.debug(
                    "Load %s: releasing q_input[0] constraint "
                    "(q_start=%.4f, start_temp=%.1f, min_temp=%.1f)",
                    k,
                    q_start_val,
                    start_temp_float,
                    min_temp_0,
                )
            else:
                constraints.append(q_input[0] == q_input_start)

            # Raw heat input: COP * P_hp / 1000 * dt (kWh thermal per timestep)
            raw_heat = cp.multiply(heatpump_cops[:-1], p_deferrable[:-1]) / 1000 * self.time_step

            # First-order low-pass filter
            constraints.append(q_input[1:] == q_input[:-1] + alpha * (raw_heat - q_input[:-1]))

            # Temperature uses filtered Q_input instead of raw heat
            # sense_coeff: +1 for heating (pump adds heat), -1 for cooling (pump removes heat)
            # Sign convention: heating_demand is >=0 for heating and <=0 for
            # cooling (calculate_heating_demand_physics returns a signed heat
            # gain), so subtracting it cools the tank when heating and warms it
            # when cooling, matching the thermal_losses sign convention.
            constraints.append(
                predicted_temp_thermal[1:]
                == predicted_temp_thermal[:-1]
                + conversion
                * (
                    sense_coeff * q_input[:-1]
                    - heating_demand[:-1]
                    - thermal_losses[:-1]
                    - coupling_term
                )
            )

            # Store reference for auto-persistence on cache hit
            if k in self.param_thermal:
                self.param_thermal[k]["q_input_var"] = q_input
        else:
            q_input = None
            # Original Langer & Volling equation (backward compatible)
            # sense_coeff: +1 for heating (pump adds heat), -1 for cooling (pump removes heat)
            constraints.append(
                predicted_temp_thermal[1:]
                == predicted_temp_thermal[:-1]
                + conversion
                * (
                    sense_coeff
                    * (cp.multiply(heatpump_cops[:-1], p_deferrable[:-1]) / 1000 * self.time_step)
                    - heating_demand[:-1]
                    - thermal_losses[:-1]
                    - coupling_term
                )
            )

        # Return heating_demand array for result building
        heating_demand_arr = (
            self.param_thermal[k]["heating_demand"].value
            if k in self.param_thermal
            else heating_demand
        )

        penalty_term = self._add_thermal_battery_bounds_and_penalty(
            constraints,
            k,
            hc,
            predicted_temp_thermal,
            required_len,
            min_temps_param,
            max_temps_param,
            min_temperatures_list,
            max_temperatures_list,
            sense,
            p_deferrable,
        )
        return predicted_temp_thermal, heating_demand_arr, q_input, penalty_term

    def _add_thermal_battery_bounds_and_penalty(
        self,
        constraints,
        k,
        hc,
        predicted_temp_thermal,
        required_len,
        min_temps_param,
        max_temps_param,
        min_temperatures_list,
        max_temperatures_list,
        sense,
        p_deferrable,
    ):
        """Shared tail end of thermal_battery constraint-building: min/max
        temperature bounds, legionella hold, and the soft overshoot/desired-
        temperature penalty. Operates purely on predicted_temp_thermal/hc/
        self.param_thermal[k] - independent of which equation produced
        predicted_temp_thermal, so both the physics/RC recurrence
        (_add_thermal_battery_constraints) and the self-learning-physics
        recurrence (_add_self_learning_dispatch_constraints) call this
        identically. Pure extraction from the pre-existing RC-only code path -
        no behavior change for RC rooms.

        :return: penalty_term (cp.Expression or None)
        """
        # Min/Max Temperature Constraints using parameters. Hard by default -
        # comfort bounds must never be quietly traded away for cost when a
        # zero-violation plan exists. self._soft_comfort_bounds_pass (set
        # only by _perform_optimization_core's own already-existing
        # infeasible-retry path, see its own comment) turns these into
        # elastic (slack-variable) constraints instead of dropping the
        # bound entirely: a heavily-penalized nonneg slack absorbs any
        # UNAVOIDABLE violation, so a genuinely-infeasible day still
        # produces a usable plan (comfort pushed as close as physically
        # possible) instead of no plan at all - the same escape hatch
        # _perform_optimization_core's relaxed-LP retry already offers
        # every OTHER load type's own binary logic, extended to comfort
        # bounds too. Never applies on the primary (strict) solve attempt.
        comfort_penalty_terms: list = []
        soft_pass = getattr(self, "_soft_comfort_bounds_pass", False)
        if min_temps_param is not None:
            if soft_pass:
                slack_min = cp.Variable(required_len, nonneg=True, name=f"comfort_slack_min_{k}")
                constraints.append(predicted_temp_thermal[1:] + slack_min[1:] >= min_temps_param[1:])
                comfort_penalty_terms.append(slack_min[1:])
            else:
                constraints.append(predicted_temp_thermal[1:] >= min_temps_param[1:])
        elif valid_indices := [
            i
            for i, v in enumerate(min_temperatures_list)
            if v is not None and i < required_len and i > 0
        ]:
            limit_vals = np.array([min_temperatures_list[i] for i in valid_indices])
            if soft_pass:
                slack_min = cp.Variable(len(valid_indices), nonneg=True, name=f"comfort_slack_min_{k}")
                constraints.append(predicted_temp_thermal[valid_indices] + slack_min >= limit_vals)
                comfort_penalty_terms.append(slack_min)
            else:
                constraints.append(predicted_temp_thermal[valid_indices] >= limit_vals)

        if max_temps_param is not None:
            if soft_pass:
                slack_max = cp.Variable(required_len, nonneg=True, name=f"comfort_slack_max_{k}")
                constraints.append(predicted_temp_thermal[1:] - slack_max[1:] <= max_temps_param[1:])
                comfort_penalty_terms.append(slack_max[1:])
            else:
                constraints.append(predicted_temp_thermal[1:] <= max_temps_param[1:])
        elif valid_indices := [
            i
            for i, v in enumerate(max_temperatures_list)
            if v is not None and i < required_len and i > 0
        ]:
            limit_vals = np.array([max_temperatures_list[i] for i in valid_indices])
            if soft_pass:
                slack_max = cp.Variable(len(valid_indices), nonneg=True, name=f"comfort_slack_max_{k}")
                constraints.append(predicted_temp_thermal[valid_indices] - slack_max <= limit_vals)
                comfort_penalty_terms.append(slack_max)
            else:
                constraints.append(predicted_temp_thermal[valid_indices] <= limit_vals)

        # Legionella hard constraints when due.
        if bool(hc.get("legionella_due", False)):
            legio_target = float(hc.get("legionella_target_temperature", 60.0))
            hold_hours = float(hc.get("legionella_hold_hours", 0.0) or 0.0)
            hold_steps = max(1, int(ceil(hold_hours / self.time_step)))
            if hold_steps <= 1:
                constraints.append(cp.max(predicted_temp_thermal) >= legio_target)
            else:
                # Require a single *contiguous* window of hold_steps timesteps at or
                # above target. Disinfection needs a sustained hold, not scattered
                # timesteps that individually touch the target temperature.
                window_starts = required_len - hold_steps + 1
                if window_starts < 1:
                    # Horizon shorter than the required hold duration: best effort.
                    constraints.append(cp.max(predicted_temp_thermal) >= legio_target)
                else:
                    y = cp.Variable(window_starts, boolean=True, name=f"legio_hold_{k}")
                    big_m = 100.0
                    constraints.append(cp.sum(y) == 1)
                    for t in range(window_starts):
                        window = predicted_temp_thermal[t : t + hold_steps]
                        constraints.append(window >= legio_target - big_m * (1 - y[t]))

        # Soft constraints (overshoot/desired/penalty) - same pattern as thermal_config
        penalty_expr = 0
        desired_temps_list = hc.get("desired_temperatures", [])
        overshoot_temperature = hc.get("overshoot_temperature", None)
        sense_coeff = 1 if sense == "heat" else -1

        if desired_temps_list and overshoot_temperature is not None:
            is_overshoot = cp.Variable(required_len, boolean=True, name=f"is_overshoot_tb_{k}")
            big_m = 100

            if sense == "heat":
                constraints.append(
                    predicted_temp_thermal - overshoot_temperature - (big_m * is_overshoot) <= 0
                )
                constraints.append(
                    predicted_temp_thermal - overshoot_temperature + (big_m * (1 - is_overshoot))
                    >= 0
                )
            else:
                constraints.append(
                    predicted_temp_thermal - overshoot_temperature - (-big_m * is_overshoot) >= 0
                )
                constraints.append(
                    predicted_temp_thermal - overshoot_temperature + (-big_m * (1 - is_overshoot))
                    <= 0
                )

            # Prevent heating when in overshoot — use p_def_bin2 if available, else bound power directly
            if self.optim_conf["treat_deferrable_load_as_semi_cont"][k]:
                p_def_bin2 = self.vars["p_def_bin2"][k]
                constraints.append(is_overshoot[1:] + p_def_bin2[:-1] <= 1)
            else:
                # For non-semi-cont loads, suppress power directly when in overshoot
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    nominal_power = max(nominal_power)
                constraints.append(p_deferrable <= nominal_power * (1 - is_overshoot))

            # Penalty calculation
            penalty_factor = hc.get("penalty_factor", 10)
            if valid_indices := [
                i
                for i, val in enumerate(desired_temps_list)
                if val is not None and i < required_len and i > 0
            ]:
                if k in self.param_thermal and "desired_temps" in self.param_thermal[k]:
                    desired_temps_param = self.param_thermal[k]["desired_temps"]
                    deviation = (
                        predicted_temp_thermal[valid_indices] - desired_temps_param[valid_indices]
                    ) * sense_coeff
                else:
                    des_temps = np.array([desired_temps_list[i] for i in valid_indices])
                    deviation = (predicted_temp_thermal[valid_indices] - des_temps) * sense_coeff

                penalty_expr = -cp.pos(-deviation * penalty_factor)

        if comfort_penalty_terms:
            # Deliberately much larger than the ordinary overshoot
            # penalty_factor above (default 10): this only ever fires
            # during the already-infeasible retry pass, so it must dominate
            # every real cost/comfort trade-off in the objective - a
            # genuinely avoidable violation should never be cheaper than
            # the real fix, only an UNAVOIDABLE one should ever show
            # nonzero slack. Per-degree-per-timestep, not a one-off cost,
            # so it scales with both how far and how long comfort is missed.
            penalty_expr = penalty_expr - _COMFORT_VIOLATION_PENALTY_PER_DEGREE * cp.sum(
                cp.hstack(comfort_penalty_terms)
            )

        return None if isinstance(penalty_expr, int) else cp.sum(penalty_expr)

    def _build_aggregate_heatpump_duty_expr(self):
        """Aggregate heat-pump duty as a native CVXPY affine expression - the
        live-solve equivalent of utils.compute_aggregate_heatpump_duty
        (which computes the same ratio downstream, from an already-solved
        plan's P_deferrable columns, for the hybrid/self-learning forecast
        actions). Fed as BOTH the "duty" and "group_duty" feature for every
        self-learning-flagged room's dispatch equation (see
        _add_self_learning_dispatch_constraints's own docstring: these are
        the exact same underlying signal in the training data this model
        was actually fit on - a single whole-house heatpump_duty_sensor
        reading, never a per-room one - so feeding the same expression into
        both feature slots is faithful to the fit, not an approximation).

        Membership mirrors room_load_indices/heatpump_dispatch_load_index
        (utils.py::compute_aggregate_heatpump_duty's own definition) via the
        heatpump_group_member marker utils.py::_append_room_thermal_loads
        stamps on every room and the dispatch load, flag-independent -
        deliberately not heatpump_room_shared_group (a single-room house has
        shared_group=0 but still has exactly one physical pump needing this
        signal, same rationale already documented for the downstream helper).

        Deliberately NOT clipped to [0, 1] (clipping a decision-variable
        expression is not affine, illegal inside an equality constraint) -
        warns instead when a member's own nominal power could push the
        ratio outside the [0, 1] range the model was actually trained on.
        """
        def_load_config = self.optim_conf.get("def_load_config", []) or []
        members = [
            k
            for k, cfg in enumerate(def_load_config)
            if isinstance(cfg, dict)
            and isinstance(cfg.get("thermal_battery"), dict)
            and cfg["thermal_battery"].get("heatpump_group_member")
        ]
        nominal = float(self.plant_conf.get("heatpump_nominal_power", 0.0) or 0.0)
        if not members or nominal <= 0:
            self.logger.warning(
                "Self-learning dispatch: no heatpump_group_member loads found and/or "
                "plant_conf['heatpump_nominal_power'] is not set (>0) - aggregate duty "
                "cannot be computed."
            )
            return None
        nominal_powers = self.optim_conf.get("nominal_power_of_deferrable_loads", [])
        for m in members:
            member_nominal = nominal_powers[m] if m < len(nominal_powers) else None
            if isinstance(member_nominal, list):
                member_nominal = max(member_nominal) if member_nominal else None
            if member_nominal is not None and float(member_nominal) > nominal:
                self.logger.warning(
                    "Self-learning dispatch: load %d's own nominal_power_of_deferrable_loads "
                    "(%.0f W) exceeds plant_conf['heatpump_nominal_power'] (%.0f W) - the "
                    "aggregate duty ratio can exceed 1.0 for this load alone, outside the "
                    "[0, 1] range the model was trained on. Consider heatpump_room_shared_group "
                    "or correcting heatpump_nominal_power.",
                    m,
                    float(member_nominal),
                    nominal,
                )
        p_deferrable = self.vars["p_deferrable"]
        total = p_deferrable[members[0]]
        for m in members[1:]:
            total = total + p_deferrable[m]
        return total / nominal

    def _add_self_learning_dispatch_constraints(
        self, constraints, k, hc, data_opt, def_init_temp, duty_expr, sl_neighbor_vars,
        room_blind_positions=None, room_opening_open=None, room_door_open=None,
    ):
        """Dispatch equation for a heatpump_room_self_learning_only room with
        a fitted model (hc["self_learning_dispatch"], see
        utils.py::_append_room_thermal_loads): the room's temperature
        recurrence is the fitted self-learning-physics model's own equation
        (emhass.thermal.self_learning_physics._BASE_FEATURE_NAMES) instead
        of the physics/RC conversion/COP recurrence in
        _add_thermal_battery_constraints - volume/u_value/COP config is not
        read at all here.

        Row alignment (easy to get backwards, see _physics_features's own
        docstring): the fitted model evaluates duty/weather at row t against
        room_last = T[t-1], i.e. T[t] = theta @ features(room=T[t-1],
        duty=duty[t], outdoor[t], ...). Since predicted_temp_thermal[1:]
        represents T[1..n-1], the *exogenous* arrays (duty/outdoor/wind/dni/
        dhi/sun_alt_sin) must be sliced [1:] (aligned with the output), while
        only the self-lag term uses predicted_temp_thermal[:-1] - opposite
        of the RC recurrence immediately above, whose terms are all [:-1].

        DCP-legality, term by term: bias (constant); room_last
        (predicted_temp_thermal[:-1], a Variable slice - affine); duty/
        group_duty (duty_expr[1:], itself an affine combination of
        p_deferrable Variables - affine); cold_below_2c/wind_speed/
        wind_x_outdoor/dni/dhi/sun_alt_sin/blind_x_dni/dni_x_sun_az_sin/
        dni_x_sun_az_cos (plain weather/blind arrays, no decision variable at
        all - affine/constant; blind_x_dni and dni_x_sun_az_sin/cos are each
        a product of two already-plain numpy arrays - room_blind_positions/
        sun_az_sin/sun_az_cos times dni_arr - computed before this method
        builds any CVXPY expression, same legality class as dni/dhi
        themselves); opening_x_outdoor (same legality class again -
        opening_now * delta_env_ref, a product of two plain numpy arrays,
        since delta_env_ref is itself already a fixed reference-trajectory
        -derived array by this point, see below);
        neighbor_diff::* (difference of two Variable slices via
        sl_neighbor_vars - affine); door_x_neighbor_diff::* (door_now, a
        plain 0/1 numpy array, times a real sl_neighbor_vars Variable slice -
        DCP-legal constant-times-affine, routed through cp.multiply since
        the left operand is array-valued, matching
        _add_room_thermal_coupling_constraints's own convention); delta_supply/
        delta_env and their duty-products are the only non-affine features
        (clip() of a Variable, and a Variable-times-Variable product) - both
        are linearized against a REFERENCE trajectory
        (self._sl_reference_trajectories, see
        _perform_two_pass_optimization) evaluated as plain
        numpy *before* this method runs, making them fixed per-timestep
        coefficients multiplying the still-live duty_expr - affine.
        """
        sl = hc["self_learning_dispatch"]
        theta = dict(zip(sl["feature_names"], sl["theta"], strict=True))
        n = self.num_timesteps
        params = self.param_thermal.get(k, {})

        start_temperature = (
            def_init_temp[k]
            if def_init_temp is not None and k < len(def_init_temp) and def_init_temp[k] is not None
            else hc.get("start_temperature", 20.0)
        )
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0
        if "start_temp" in params:
            params["start_temp"].value = start_temperature

        predicted_temp_thermal = cp.Variable(n, name=f"temp_thermal_batt_{k}")
        constraints.append(predicted_temp_thermal[0] == start_temperature)

        if duty_expr is None:
            raise ValueError(
                f"Load {k}: self-learning dispatch requires plant_conf['heatpump_nominal_power'] "
                "> 0 and at least one heatpump_group_member load (see "
                "_build_aggregate_heatpump_duty_expr)."
            )

        outdoor_arr = self._get_clean_outdoor_temp(data_opt, n)
        # Room supply/flow temperature: a flat per-room config constant here
        # (hc["supply_temperature"], the same field the self-learning-only
        # UI toggle hides), not a live time-varying reading - an accepted
        # v1 simplification versus whatever richer supply_temp signal (a
        # real heatpump_flow_temp_sensor, or the model's own room+5 fallback)
        # this room's model may have actually been fit against.
        supply_arr = np.full(n, float(hc.get("supply_temperature", 35.0)))
        wind_arr = self._get_clean_weather_col(data_opt, "wind_speed", n, default=0.0)
        dni_arr = self._get_clean_weather_col(data_opt, "dni", n, default=0.0)
        dhi_arr = self._get_clean_weather_col(data_opt, "dhi", n, default=0.0)
        cold_arr = (outdoor_arr < 2.0).astype(float)
        wind_x_outdoor_arr = wind_arr * outdoor_arr
        # Sun position (see prepare_forecast_and_weather_data, which merges
        # these onto data_opt via Forecast.compute_solar_angles - the same
        # deterministic, timestamp+location-only pvlib computation
        # refit_self_learning_physics_model/compute_self_learning_physics_forecast
        # already use to fit/forecast against, so dispatch stays
        # self-consistent with what was actually fit). dni_x_sun_az_sin/cos
        # let the room's own fitted coefficients express an effective window
        # orientation without ever needing a hand-specified facade azimuth.
        sun_alt_sin_arr = self._get_clean_weather_col(data_opt, "sun_alt_sin", n, default=0.0)
        sun_az_sin_arr = self._get_clean_weather_col(data_opt, "sun_az_sin", n, default=0.0)
        sun_az_cos_arr = self._get_clean_weather_col(data_opt, "sun_az_cos", n, default=0.0)
        dni_x_sun_az_sin_arr = dni_arr * sun_az_sin_arr
        dni_x_sun_az_cos_arr = dni_arr * sun_az_cos_arr
        # Room's own live blind/shading position (0=open, 1=fully closed) -
        # a slowly-changing external signal, held flat across the whole
        # horizon rather than forecast, same simplification as supply_arr
        # above. Falls back to hc["blind_type"]-independent hc.get("blind_position", 0.0)
        # (= fully open = inert) when no live override was resolved for this
        # room this solve (see command_line.py::_build_room_blind_positions).
        blind_position = (
            room_blind_positions[k]
            if room_blind_positions is not None
            and k < len(room_blind_positions)
            and room_blind_positions[k] is not None
            else float(hc.get("blind_position", 0.0))
        )
        blind_x_dni_arr = np.full(n, float(blind_position)) * dni_arr

        room_ref = self._sl_reference_trajectories.get(k)
        if room_ref is None or len(room_ref) != n:
            room_ref = np.full(n, start_temperature)
        delta_supply_ref = np.clip(supply_arr[1:] - room_ref[:-1], a_min=0.0, a_max=None)
        delta_env_ref = np.clip(room_ref[:-1] - outdoor_arr[1:], a_min=0.0, a_max=None)

        # Live "window OR door is open right now" / "door is open right now"
        # signals - unlike blind_position above (held flat across the whole
        # horizon, since blinds change state rarely), these are fast,
        # momentary, live-only signals with no way to forecast future steps,
        # so they only ever affect the FIRST real predicted step (index 0 of
        # these already-[:-1]/[1:]-equivalent length-(n-1) arrays, matching
        # delta_env_ref's own convention) - never held flat like blind_x_dni.
        opening_open_k = (
            room_opening_open is not None
            and k < len(room_opening_open)
            and bool(room_opening_open[k])
        )
        door_open_k = (
            room_door_open is not None and k < len(room_door_open) and bool(room_door_open[k])
        )
        opening_now = np.zeros(n - 1)
        if opening_open_k:
            opening_now[0] = 1.0
        door_now = np.zeros(n - 1)
        if door_open_k:
            door_now[0] = 1.0

        rhs = theta.get("bias", 0.0)
        rhs = rhs + theta.get("room_last", 0.0) * predicted_temp_thermal[:-1]
        rhs = rhs + theta.get("duty", 0.0) * duty_expr[1:]
        rhs = rhs + theta.get("delta_supply", 0.0) * delta_supply_ref
        rhs = rhs + theta.get("duty_x_delta_supply", 0.0) * cp.multiply(delta_supply_ref, duty_expr[1:])
        rhs = rhs + theta.get("delta_env", 0.0) * delta_env_ref
        rhs = rhs + theta.get("duty_x_delta_env", 0.0) * cp.multiply(delta_env_ref, duty_expr[1:])
        rhs = rhs + theta.get("cold_below_2c", 0.0) * cold_arr[1:]
        rhs = rhs + theta.get("wind_speed", 0.0) * wind_arr[1:]
        rhs = rhs + theta.get("wind_x_outdoor", 0.0) * wind_x_outdoor_arr[1:]
        rhs = rhs + theta.get("dni", 0.0) * dni_arr[1:]
        rhs = rhs + theta.get("dhi", 0.0) * dhi_arr[1:]
        rhs = rhs + theta.get("sun_alt_sin", 0.0) * sun_alt_sin_arr[1:]
        rhs = rhs + theta.get("dni_x_sun_az_sin", 0.0) * dni_x_sun_az_sin_arr[1:]
        rhs = rhs + theta.get("dni_x_sun_az_cos", 0.0) * dni_x_sun_az_cos_arr[1:]
        rhs = rhs + theta.get("blind_x_dni", 0.0) * blind_x_dni_arr[1:]
        # opening_now/delta_env_ref are both plain numpy arrays (no decision
        # variable involved) - a constant elementwise product, still affine
        # once scaled by the theta coefficient, so no cp.multiply is needed
        # here (unlike the door_x_neighbor_diff term below, which multiplies
        # a real CVXPY variable slice).
        rhs = rhs + theta.get("opening_x_outdoor", 0.0) * (opening_now * delta_env_ref)
        # group_duty is the SAME underlying signal as duty in every existing
        # fit (see _build_aggregate_heatpump_duty_expr's own docstring) -
        # feed the identical expression, not a second independent one.
        rhs = rhs + theta.get("group_duty", 0.0) * duty_expr[1:]
        for neighbor_name, neighbor_idx in sl.get("neighbor_indices", {}).items():
            feature_name = f"neighbor_diff::{neighbor_name}"
            if feature_name in theta and (k, neighbor_idx) in sl_neighbor_vars:
                rhs = rhs + theta[feature_name] * sl_neighbor_vars[(k, neighbor_idx)][:-1]
            door_feature_name = f"door_x_neighbor_diff::{neighbor_name}"
            if door_feature_name in theta and (k, neighbor_idx) in sl_neighbor_vars:
                rhs = rhs + theta[door_feature_name] * cp.multiply(
                    door_now, sl_neighbor_vars[(k, neighbor_idx)][:-1]
                )

        constraints.append(predicted_temp_thermal[1:] == rhs)

        sense = utils.normalize_heat_cool_mode(
            hc.get("sense") or "heat", field_name="sense", context=f"Load {k} self_learning_dispatch"
        )
        min_temperatures_list = hc.get("min_temperatures", [])
        max_temperatures_list = hc.get("max_temperatures", [])
        min_temps_param = params.get("min_temps")
        max_temps_param = params.get("max_temps")
        if min_temps_param is not None and max_temps_param is not None:
            min_temps_arr = self._pad_temp_array(min_temperatures_list, n, 18.0)
            max_temps_arr = self._pad_temp_array(max_temperatures_list, n, 26.0)
            min_temps_arr, max_temps_arr = self._relax_opening_temp_bounds(
                min_temps_arr, max_temps_arr, opening_open_k
            )
            min_temps_param.value = min_temps_arr
            max_temps_param.value = max_temps_arr
        elif min_temps_param is not None:
            min_temps_param.value = self._pad_temp_array(min_temperatures_list, n, 18.0)
        elif max_temps_param is not None:
            max_temps_param.value = self._pad_temp_array(max_temperatures_list, n, 26.0)

        p_deferrable = self.vars["p_deferrable"][k]
        penalty_term = self._add_thermal_battery_bounds_and_penalty(
            constraints,
            k,
            hc,
            predicted_temp_thermal,
            n,
            min_temps_param,
            max_temps_param,
            min_temperatures_list,
            max_temperatures_list,
            sense,
            p_deferrable,
        )
        # No separate heating-demand quantity exists for a self-learning
        # room (the model predicts temperature directly, no COP/conversion
        # decomposition) - report zeros rather than a physically meaningless
        # value for the heating_demand_heater{k} result column.
        heating_demand_arr = np.zeros(n)
        return predicted_temp_thermal, heating_demand_arr, None, penalty_term

    def _add_rc_physics_dispatch_constraints(
        self, constraints, k, hc, data_opt, def_init_temp, duty_expr,
        room_blind_positions=None, room_opening_open=None, room_door_open=None,
    ):
        """Dispatch equation for a heatpump_room_rc_physics_only room with a
        fitted RC-physics model (hc["rc_physics_dispatch"], see
        utils.py::_append_room_thermal_loads) - the room's temperature
        recurrence is thermal_mass_physics._simulate_open_loop's OWN 4-state
        (T_air/T_mass/T_wall/Q_emit) equation, reformulated as CVXPY
        constraints, instead of the physics/RC conversion/COP recurrence in
        _add_thermal_battery_constraints (a different, simpler model) or the
        self-learning-physics fitted equation immediately above.

        Sibling of _add_self_learning_dispatch_constraints - same "one
        genuine nonlinearity, everything else exogenous" structure, but RC
        has ONE live-state-dependent nonlinearity of its own
        (emit_raw = duty*max(supply-air,0), see module docstring) rather
        than self-learning's two (delta_supply/delta_env), and three chained
        auxiliary states (T_mass/T_wall/Q_emit) rather than a flat feature
        vector.

        Row alignment - re-derived directly from _simulate_open_loop's own
        per-timestep loop (opposite convention from the self-learning method
        immediately above): within iteration i, q_emit/wall/mass are updated
        FIRST using air BEFORE this step's own update (T_air[i-1] in this
        method's [1:]/[:-1] slicing), then d_air_dt uses those JUST-updated
        q_emit[i]/mass[i] plus air[i-1] again for the loss/mass-gain terms,
        before air is finally updated to T_air[i]. So, confirmed term by
        term from _simulate_open_loop: T_wall[1:] = f(T_wall[:-1]) only (no
        air coupling - wall_target depends only on outdoor+solar);
        T_mass[1:] = f(T_mass[:-1], T_air[:-1], T_wall[1:]); Q_emit[1:] =
        f(Q_emit[:-1], duty_expr[1:], and the FROZEN max(supply-air_ref,0)
        reference term - air is genuinely live here, so this is the one term
        that needs _rc_reference_trajectories, the RC sibling of self-
        learning's own delta_supply_ref); T_air[1:] = f(T_air[:-1],
        Q_emit[1:], T_mass[1:]). Every equation is a linear combination of
        Variable slices, fixed numpy arrays, and (for Q_emit's own duty
        term) a live-but-affine duty_expr product - DCP-legal throughout, no
        cp.multiply of two Variables anywhere.

        Deliberately omits _simulate_open_loop's own min(35, max(5, ...))
        physical safety clip on T_air (a piecewise-nonlinear op, not
        representable as an equality constraint) - same simplification
        self-learning's own dispatch equation already makes with no explicit
        floor/ceiling either, relying on comfort bounds (typically 18-26C,
        added below via the shared bounds/penalty tail) to keep the solution
        physically sensible; the 5/35C safety clip essentially never binds
        for a well-posed comfort-constrained dispatch problem.

        facade2/facade3 secondary orientations, wind DIRECTION (ua_wind_sin/
        ua_wind_cos - only wind_speed/ua_wind is modeled), and a Q_emit
        persisted across solves are all left for a later iteration (see the
        design plan's own Scope section) - the missing pieces default to 0
        rather than error, degrading gracefully to the dominant terms.
        """
        from emhass.thermal.thermal_mass_physics import (
            _COP_REFERENCE_OUTDOOR_C,
            DEFAULT_X0,
            PARAM_NAMES,
            _cop_carnot_vectorized,
            _facade_poa_vectorized,
            _facade_trig,
        )

        rc = hc["rc_physics_dispatch"]
        if duty_expr is None:
            raise ValueError(
                f"Load {k}: RC-physics dispatch requires plant_conf['heatpump_nominal_power'] "
                "> 0 and at least one heatpump_group_member load (see "
                "_build_aggregate_heatpump_duty_expr)."
            )

        n = self.num_timesteps
        p = rc.get("params", {}) or {}
        param_values = [
            float(p[name]) if name in p and p[name] is not None else float(DEFAULT_X0[i])
            for i, name in enumerate(PARAM_NAMES)
        ]
        (
            tau_emit, emit_gain, ua_base, ua_wind, _ua_wind_sin, _ua_wind_cos,
            mass_tau, mass_gain, solar_gain, solar_alt_sin_gain, solar_alt_cos_gain,
            solar_az_sin_gain, solar_az_cos_gain, bias, wall_tau, wall_solar_gain,
            wall_to_mass_weight, door_open_extra_loss, window_solar_radiative_fraction,
            facade_azimuth_deg, facade_tilt_deg,
            _facade2_azimuth_deg, _facade2_tilt_deg, _facade3_azimuth_deg, _facade3_tilt_deg,
            carnot_efficiency, _emitter_power_scale_w, cop_sensitivity,
        ) = param_values

        params = self.param_thermal.get(k, {})
        start_temperature = (
            def_init_temp[k]
            if def_init_temp is not None and k < len(def_init_temp) and def_init_temp[k] is not None
            else hc.get("start_temperature", 20.0)
        )
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0
        if "start_temp" in params:
            params["start_temp"].value = start_temperature

        # Exogenous forecast/weather arrays - never depend on live state, so
        # every one of these precomputes as a plain numpy array before any
        # CVXPY expression is built, exactly the classic battery's own
        # heatpump_cops precompute pattern.
        outdoor_arr = self._get_clean_outdoor_temp(data_opt, n)
        wind_arr = self._get_clean_weather_col(data_opt, "wind_speed", n, default=0.0)
        ghi_arr = self._get_clean_weather_col(data_opt, "ghi", n, default=0.0)
        dni_arr = self._get_clean_weather_col(data_opt, "dni", n, default=0.0)
        dhi_arr = self._get_clean_weather_col(data_opt, "dhi", n, default=0.0)
        sun_alt_sin_arr = self._get_clean_weather_col(data_opt, "sun_alt_sin", n, default=0.0)
        sun_alt_cos_arr = self._get_clean_weather_col(data_opt, "sun_alt_cos", n, default=0.0)
        sun_az_sin_arr = self._get_clean_weather_col(data_opt, "sun_az_sin", n, default=0.0)
        sun_az_cos_arr = self._get_clean_weather_col(data_opt, "sun_az_cos", n, default=0.0)

        cos_tilt, sin_tilt, cos_az, sin_az = _facade_trig(facade_azimuth_deg, facade_tilt_deg)
        poa_arr = _facade_poa_vectorized(
            ghi_arr, dni_arr, dhi_arr, sun_alt_sin_arr, sun_alt_cos_arr, sun_az_sin_arr, sun_az_cos_arr,
            cos_tilt, sin_tilt, cos_az, sin_az,
        )
        # horizontal_weight/facade_weight are _simulate_open_loop's own
        # hardcoded defaults (0.35/0.65) - never overridden from config
        # anywhere in this codebase (command_line.py's fit/forecast call
        # sites don't pass them either), so reused literally here for the
        # same q_solar_i the fit itself was evaluated against.
        q_solar_arr = np.maximum(0.0, 0.35 * ghi_arr + 0.65 * poa_arr) / 1000.0

        wall_target_arr = outdoor_arr + wall_solar_gain * q_solar_arr

        direction_loss_arr = np.maximum(0.0, ua_base + ua_wind * wind_arr)

        # Room's own live blind/shading + door/window-open state - same
        # "blind held flat across the horizon, door/window-open only ever
        # affects the FIRST predicted step" convention
        # _add_self_learning_dispatch_constraints already uses (see its own
        # docstring) - neither can be forecast beyond "right now".
        blind_position = (
            room_blind_positions[k]
            if room_blind_positions is not None
            and k < len(room_blind_positions)
            and room_blind_positions[k] is not None
            else float(hc.get("blind_position", 0.0))
        )
        opening_open_k = (
            room_opening_open is not None and k < len(room_opening_open) and bool(room_opening_open[k])
        )
        door_open_k = (
            room_door_open is not None and k < len(room_door_open) and bool(room_door_open[k])
        )
        # ThermalInputs.door_open represents "door OR window open" jointly
        # (see thermal_mass_physics.py's own field docstring), unlike self-
        # learning's separate opening_open/door_open features - either live
        # signal sets it.
        door_now = np.zeros(n - 1)
        if opening_open_k or door_open_k:
            door_now[0] = 1.0
        loss_coeff_sliced = direction_loss_arr[1:] + door_open_extra_loss * door_now

        solar_direction_gain_arr = (
            solar_gain
            + solar_alt_sin_gain * sun_alt_sin_arr
            + solar_alt_cos_gain * sun_alt_cos_arr
            + solar_az_sin_gain * sun_az_sin_arr
            + solar_az_cos_gain * sun_az_cos_arr
        )
        window_solar_total_arr = (
            np.maximum(0.0, solar_direction_gain_arr) * q_solar_arr * (1.0 - blind_position)
        )
        window_solar_convective_arr = (1.0 - window_solar_radiative_fraction) * window_solar_total_arr
        window_solar_radiative_arr = window_solar_radiative_fraction * window_solar_total_arr

        # The ONE live-state-dependent nonlinearity - frozen against the
        # reference trajectory from the two-pass reference solve (see
        # _capture_rc_physics_reference / _perform_two_pass_optimization),
        # the RC sibling of self-learning's own delta_supply_ref.
        supply_arr = np.full(n, float(hc.get("supply_temperature", 35.0)))
        air_ref = self._rc_reference_trajectories.get(k)
        if air_ref is None or len(air_ref) != n:
            air_ref = np.full(n, start_temperature)
        clamped_diff_ref = np.clip(supply_arr[1:] - air_ref[:-1], a_min=0.0, a_max=None)

        # COP-aware duty effectiveness (see thermal_mass_physics.PARAM_NAMES's
        # own cop_sensitivity docstring for why this is a fitted parameter,
        # not a hand-picked constant) - purely exogenous (depends only on
        # carnot_efficiency + the weather forecast, never on live MILP
        # state), so it folds straight into the already-exogenous
        # clamped_diff_ref numpy array below, same DCP-legality argument as
        # every other precompute in this function. cop_sensitivity=0.0
        # (unfit/not-yet-refit models) makes cop_scale_arr all-ones, a no-op.
        cop_arr = _cop_carnot_vectorized(carnot_efficiency, supply_arr, outdoor_arr)
        cop_ref_arr = _cop_carnot_vectorized(
            carnot_efficiency, supply_arr, np.full(n, _COP_REFERENCE_OUTDOOR_C)
        )
        cop_scale_arr = np.clip(1.0 + cop_sensitivity * (cop_arr - cop_ref_arr), 0.1, None)
        clamped_diff_ref = clamped_diff_ref * cop_scale_arr[1:]

        T_air = cp.Variable(n, name=f"rc_air_{k}")
        T_mass = cp.Variable(n, name=f"rc_mass_{k}")
        T_wall = cp.Variable(n, name=f"rc_wall_{k}")
        Q_emit = cp.Variable(n, name=f"rc_qemit_{k}")
        constraints.append(T_air[0] == start_temperature)
        constraints.append(T_mass[0] == start_temperature)
        constraints.append(T_wall[0] == start_temperature)
        # No Q_emit state is persisted across solves (each solve starts
        # fresh) - same simplification compute_heating_forecast's own
        # informational forecast already makes (initial_q_emit=0.0), and
        # self-correcting anyway since T_air[0] is re-pinned to a real
        # sensor reading every solve.
        constraints.append(Q_emit[0] == 0.0)

        emit_alpha = float(np.clip(self.time_step / max(tau_emit, 1e-6), 0.0, 1.0))
        mass_alpha = float(np.clip(self.time_step / max(mass_tau, 1e-6), 0.0, 1.0))
        wall_alpha = float(np.clip(self.time_step / max(wall_tau, 1e-6), 0.0, 1.0))

        emit_raw_expr = cp.multiply(duty_expr[1:], clamped_diff_ref)
        constraints.append(Q_emit[1:] == Q_emit[:-1] + emit_alpha * (emit_raw_expr - Q_emit[:-1]))
        constraints.append(T_wall[1:] == T_wall[:-1] + wall_alpha * (wall_target_arr[1:] - T_wall[:-1]))
        mass_target_expr = T_air[:-1] + wall_to_mass_weight * (T_wall[1:] - T_air[:-1])
        constraints.append(
            T_mass[1:]
            == T_mass[:-1]
            + mass_alpha * (mass_target_expr - T_mass[:-1])
            + self.time_step * window_solar_radiative_arr[1:]
        )
        d_air_dt_expr = (
            emit_gain * Q_emit[1:]
            + window_solar_convective_arr[1:]
            - cp.multiply(loss_coeff_sliced, T_air[:-1] - outdoor_arr[1:])
            + mass_gain * (T_mass[1:] - T_air[:-1])
            + bias
        )
        constraints.append(T_air[1:] == T_air[:-1] + self.time_step * d_air_dt_expr)

        sense = utils.normalize_heat_cool_mode(
            hc.get("sense") or "heat", field_name="sense", context=f"Load {k} rc_physics_dispatch"
        )
        min_temperatures_list = hc.get("min_temperatures", [])
        max_temperatures_list = hc.get("max_temperatures", [])
        min_temps_param = params.get("min_temps")
        max_temps_param = params.get("max_temps")
        if min_temps_param is not None and max_temps_param is not None:
            min_temps_arr = self._pad_temp_array(min_temperatures_list, n, 18.0)
            max_temps_arr = self._pad_temp_array(max_temperatures_list, n, 26.0)
            min_temps_arr, max_temps_arr = self._relax_opening_temp_bounds(
                min_temps_arr, max_temps_arr, opening_open_k
            )
            min_temps_param.value = min_temps_arr
            max_temps_param.value = max_temps_arr
        elif min_temps_param is not None:
            min_temps_param.value = self._pad_temp_array(min_temperatures_list, n, 18.0)
        elif max_temps_param is not None:
            max_temps_param.value = self._pad_temp_array(max_temperatures_list, n, 26.0)

        p_deferrable = self.vars["p_deferrable"][k]
        penalty_term = self._add_thermal_battery_bounds_and_penalty(
            constraints,
            k,
            hc,
            T_air,
            n,
            min_temps_param,
            max_temps_param,
            min_temperatures_list,
            max_temperatures_list,
            sense,
            p_deferrable,
        )
        # No separate heating-demand quantity exists for an RC-physics
        # dispatch room either (same reporting simplification as the self-
        # learning branch immediately above) - report zeros rather than a
        # physically meaningless value for the heating_demand_heater{k}
        # result column.
        heating_demand_arr = np.zeros(n)
        return T_air, heating_demand_arr, None, penalty_term

    def _add_self_learning_dispatch_milp_constraints(
        self, constraints, k, hc, data_opt, def_init_temp, sl_neighbor_vars,
        room_blind_positions=None, room_opening_open=None, room_door_open=None,
    ):
        """Exact-MILP dispatch equation for a heatpump_room_self_learning_only
        room whose resolved heat pump unit (heatpump_room_unit -> Heat Pump
        Units, see utils.py::_load_heatpump_units) has control_mode ==
        "weather_curve", and whose heat-source group has exactly one
        member (see utils.py::
        _append_room_thermal_loads's supply_temp_is_decision_variable
        marker, and the design plan's own Scope section) - promotes
        supply/flow temperature to a genuine per-timestep MILP decision
        variable, bounded by this room's own weather curve, alongside a
        binary on/off for the shared heat source (avoiding compressor
        short-cycling), instead of _add_self_learning_dispatch_constraints's
        frozen-supply-temperature/two-pass-reference-trajectory approach.

        Two regressions from the SAME fitted SelfLearningPhysicsModel are
        used here, both evaluated against LIVE (not reference-trajectory)
        decision variables:

        - hc["self_learning_dispatch"] (theta_temp, room-level): this
          room's own temperature recurrence, exactly like
          _add_self_learning_dispatch_constraints.
        - hc["self_learning_dispatch_elec"] (theta_elec_, WHOLE-HOUSE
          level, see self_learning_physics.py's own module docstring):
          predicts this heat source's total electric power draw. Wired in
          as an EQUALITY constraint on p_deferrable[k] (this load's own
          electric-power decision variable) rather than a separate
          objective term - p_deferrable[k] is already priced by the
          existing objective like any other deferrable load, so pinning it
          to the model's own prediction is sufficient to make the
          optimizer actually pay for whatever supply temperature it picks;
          no separate cost-function wiring needed.

        House-level feature quirk (see self_learning_physics._physics_features
        and command_line.py::refit_self_learning_physics_model's
        df_house_train construction): the whole-house training frame never
        carries a room_temp, blind_position, opening_open or door_open
        column (only each room's OWN per-room training frame does) - so at
        both fit and predict time, the house-level model's "room_last"
        feature is a constant 20.0 and its "blind_x_dni"/"opening_x_outdoor"
        features are always exactly 0 (the flat 0.0 fallback multiplied
        through). This method reproduces that convention exactly (a literal
        20.0, and the two terms omitted) rather than substituting this
        room's own live temperature/blind/opening signals, since
        substituting them would be answering a question the model was never
        fit to answer.

        DCP legality, term by term, room equation: identical to
        _add_self_learning_dispatch_constraints's own docstring for every
        term this method shares with it - bias/weather/blind/opening/
        neighbor terms are unchanged. delta_supply/delta_env are now LIVE
        (room_last = predicted_temp_thermal[:-1], a real decision variable,
        not a frozen reference array) - exact via _linearize_relu (one new
        binary + 3 inequalities per timestep, replacing the old reference-
        trajectory approximation for this room only).

        delta_supply's PLAIN theta term (not just duty_x_delta_supply) is
        gated through heat_source_on too, unlike delta_env's: supply_temp is
        a free decision variable bounded only by the weather curve,
        independent of heat_source_on, so an ungated plain delta_supply term
        would let the solver buy "free" predicted warming (push supply_temp
        toward the curve ceiling) at zero electricity cost while
        heat_source_on=0 - extrapolating the fit far outside the small
        residual delta_supply any real duty=0 training row ever showed (a
        real exploit, caught via a local smoke test against a real fitted
        model - see git history for the fix). delta_env has no equivalent
        exploit (it depends only on room_last, itself pinned by this same
        equation, and the outdoor forecast - neither is a free external
        lever) so it stays ungated, matching
        _add_self_learning_dispatch_constraints's own treatment. Gating
        delta_supply itself makes a separate duty_x_delta_supply McCormick
        step redundant (duty * (duty * delta_supply) == duty * delta_supply
        for a binary) - both theta coefficients are folded onto the single
        gated quantity instead. duty_x_delta_env is still its own separate
        McCormick term, unchanged. All gating uses
        _linearize_binary_times_continuous (McCormick-for-binary, 4
        inequalities per timestep).

        House equation: the same delta_supply/delta_env construction,
        applied against a constant 20.0 in place of room_last (see quirk
        above) - here the delta_supply exploit is structurally impossible
        regardless of gating, since the ENTIRE predicted electric-power
        value is gated through heat_source_on right before being pinned to
        p_deferrable[k] (see below), so an ungated delta_supply term
        upstream still multiplies out to exactly 0 while off. One more
        _linearize_relu call reproduces the model's own predict_recursive's
        `max(0.0, theta_elec_ @ features)` clip on the final predicted
        power before that last gate.
        """
        sl_temp = hc["self_learning_dispatch"]
        theta_temp = dict(zip(sl_temp["feature_names"], sl_temp["theta"], strict=True))
        sl_elec = hc.get("self_learning_dispatch_elec")
        if not sl_elec:
            raise ValueError(
                f"Load {k}: weather_curve exact-MILP dispatch requires "
                "hc['self_learning_dispatch_elec'] (see utils.py::_append_room_thermal_loads) "
                "- this should never be reached without it, since "
                "supply_temp_is_decision_variable is only ever set alongside it."
            )
        theta_elec = dict(zip(sl_elec["feature_names"], sl_elec["theta"], strict=True))
        n = self.num_timesteps
        params = self.param_thermal.get(k, {})

        start_temperature = (
            def_init_temp[k]
            if def_init_temp is not None and k < len(def_init_temp) and def_init_temp[k] is not None
            else hc.get("start_temperature", 20.0)
        )
        start_temperature = float(start_temperature) if start_temperature is not None else 20.0
        if "start_temp" in params:
            params["start_temp"].value = start_temperature

        predicted_temp_thermal = cp.Variable(n, name=f"temp_thermal_batt_{k}")
        constraints.append(predicted_temp_thermal[0] == start_temperature)

        outdoor_arr = self._get_clean_outdoor_temp(data_opt, n)

        # Supply temperature: a genuine per-timestep decision variable,
        # bounded below by the room's own configured floor and above by the
        # weather curve's own recommended value for the current forecast
        # outdoor temperature (apply_heating_curve, already clipped to
        # [min_supply, max_supply]) - lets the optimizer run colder than the
        # naive curve when comfort/price allow, but never hotter than what
        # the curve says is actually needed right now.
        heating_curve = hc["heating_curve"]
        min_supply = float(heating_curve.get("min_supply", 25.0))
        curve_anchor = utils.apply_heating_curve(heating_curve, outdoor_arr)
        supply_temp = cp.Variable(n, name=f"supply_temp_{k}")
        constraints.append(supply_temp >= min_supply)
        constraints.append(supply_temp <= curve_anchor)

        # Shared heat-source binary (single-member group this pass, see
        # Scope in the design plan) - replaces the continuous
        # aggregate_duty_expr this room would otherwise get from
        # _build_aggregate_heatpump_duty_expr; feeds both the "duty" and
        # "group_duty" feature slots below (same underlying signal, same
        # convention _build_aggregate_heatpump_duty_expr's own docstring
        # already documents for the continuous-duty path).
        nominal_power = float(self.plant_conf.get("heatpump_nominal_power", 0.0) or 0.0)
        if nominal_power <= 0:
            raise ValueError(
                f"Load {k}: weather_curve exact-MILP dispatch requires "
                "plant_conf['heatpump_nominal_power'] > 0."
            )
        heat_source_on = cp.Variable(n, boolean=True, name=f"heat_source_on_{k}")
        p_deferrable = self.vars["p_deferrable"][k]
        constraints.append(p_deferrable <= nominal_power * heat_source_on)
        duty_expr = heat_source_on

        wind_arr = self._get_clean_weather_col(data_opt, "wind_speed", n, default=0.0)
        dni_arr = self._get_clean_weather_col(data_opt, "dni", n, default=0.0)
        dhi_arr = self._get_clean_weather_col(data_opt, "dhi", n, default=0.0)
        cold_arr = (outdoor_arr < 2.0).astype(float)
        wind_x_outdoor_arr = wind_arr * outdoor_arr
        sun_alt_sin_arr = self._get_clean_weather_col(data_opt, "sun_alt_sin", n, default=0.0)
        sun_az_sin_arr = self._get_clean_weather_col(data_opt, "sun_az_sin", n, default=0.0)
        sun_az_cos_arr = self._get_clean_weather_col(data_opt, "sun_az_cos", n, default=0.0)
        dni_x_sun_az_sin_arr = dni_arr * sun_az_sin_arr
        dni_x_sun_az_cos_arr = dni_arr * sun_az_cos_arr
        blind_position = (
            room_blind_positions[k]
            if room_blind_positions is not None
            and k < len(room_blind_positions)
            and room_blind_positions[k] is not None
            else float(hc.get("blind_position", 0.0))
        )
        blind_x_dni_arr = np.full(n, float(blind_position)) * dni_arr

        opening_open_k = (
            room_opening_open is not None
            and k < len(room_opening_open)
            and bool(room_opening_open[k])
        )
        door_open_k = (
            room_door_open is not None and k < len(room_door_open) and bool(room_door_open[k])
        )
        opening_now = np.zeros(n - 1)
        if opening_open_k:
            opening_now[0] = 1.0
        door_now = np.zeros(n - 1)
        if door_open_k:
            door_now[0] = 1.0

        # Generous fixed big-M bounds, matching OPENING_RELAX_MIN_TEMP/
        # OPENING_RELAX_MAX_TEMP's own established convention elsewhere in
        # this file of preferring a very loose, unquestionably-valid
        # enclosure over a tightly-computed one - correctness of
        # _linearize_relu/_linearize_binary_times_continuous only needs the
        # bound to actually contain every feasible value, not to be tight.
        temp_delta_bound = 150.0
        elec_power_bound = max(5000.0, 5.0 * nominal_power)

        # Room-level delta_supply/delta_env: LIVE (room_last is
        # predicted_temp_thermal[:-1], a real decision variable this pass),
        # unlike _add_self_learning_dispatch_constraints's reference-
        # trajectory approximation.
        room_last = predicted_temp_thermal[:-1]
        delta_supply_expr = supply_temp[1:] - room_last
        delta_supply, _ = _linearize_relu(
            constraints, delta_supply_expr, -temp_delta_bound, temp_delta_bound, name=f"room{k}_dsup"
        )
        delta_env_expr = room_last - outdoor_arr[1:]
        delta_env, _ = _linearize_relu(
            constraints, delta_env_expr, -temp_delta_bound, temp_delta_bound, name=f"room{k}_denv"
        )
        # delta_supply (unlike delta_env below) MUST be gated through
        # heat_source_on even for its plain (non-duty_x_) theta term, not
        # just the cross term - supply_temp is a free decision variable
        # bounded only by the weather curve, independent of heat_source_on,
        # so an ungated plain "delta_supply" term would let the solver buy
        # "free" predicted warming (raise supply_temp toward the curve
        # ceiling) with zero electricity cost while heat_source_on=0,
        # extrapolating the fit far outside the small residual delta_supply
        # any real "duty=0" training row ever showed (confirmed by a local
        # smoke test against a real fitted model: temperature swung several
        # degrees per step with P_deferrable pinned at 0 the whole time).
        # delta_env has no equivalent exploit - it depends only on room_last
        # (itself pinned by this same equation, not a free external lever)
        # and the outdoor forecast, so it stays ungated, matching
        # _add_self_learning_dispatch_constraints's own treatment. Gating
        # delta_supply first makes a separate duty_x_delta_supply McCormick
        # step redundant (duty * (duty * delta_supply) == duty * delta_supply
        # for a binary) - both theta coefficients are folded onto the one
        # gated quantity instead.
        delta_supply_gated = _linearize_binary_times_continuous(
            constraints, duty_expr[1:], delta_supply, 0.0, temp_delta_bound, name=f"room{k}_dsup_gated"
        )
        duty_x_delta_env = _linearize_binary_times_continuous(
            constraints, duty_expr[1:], delta_env, 0.0, temp_delta_bound, name=f"room{k}_dxe"
        )

        rhs = theta_temp.get("bias", 0.0)
        rhs = rhs + theta_temp.get("room_last", 0.0) * room_last
        rhs = rhs + theta_temp.get("duty", 0.0) * duty_expr[1:]
        rhs = rhs + (
            theta_temp.get("delta_supply", 0.0) + theta_temp.get("duty_x_delta_supply", 0.0)
        ) * delta_supply_gated
        rhs = rhs + theta_temp.get("delta_env", 0.0) * delta_env
        rhs = rhs + theta_temp.get("duty_x_delta_env", 0.0) * duty_x_delta_env
        rhs = rhs + theta_temp.get("cold_below_2c", 0.0) * cold_arr[1:]
        rhs = rhs + theta_temp.get("wind_speed", 0.0) * wind_arr[1:]
        rhs = rhs + theta_temp.get("wind_x_outdoor", 0.0) * wind_x_outdoor_arr[1:]
        rhs = rhs + theta_temp.get("dni", 0.0) * dni_arr[1:]
        rhs = rhs + theta_temp.get("dhi", 0.0) * dhi_arr[1:]
        rhs = rhs + theta_temp.get("sun_alt_sin", 0.0) * sun_alt_sin_arr[1:]
        rhs = rhs + theta_temp.get("dni_x_sun_az_sin", 0.0) * dni_x_sun_az_sin_arr[1:]
        rhs = rhs + theta_temp.get("dni_x_sun_az_cos", 0.0) * dni_x_sun_az_cos_arr[1:]
        rhs = rhs + theta_temp.get("blind_x_dni", 0.0) * blind_x_dni_arr[1:]
        # opening_now is a plain 0/1 numpy array but delta_env is now a live
        # CVXPY expression (unlike _add_self_learning_dispatch_constraints's
        # frozen delta_env_ref) - route through cp.multiply for the same
        # array-times-Variable-slice legality reason door_x_neighbor_diff
        # below already does.
        rhs = rhs + theta_temp.get("opening_x_outdoor", 0.0) * cp.multiply(opening_now, delta_env)
        rhs = rhs + theta_temp.get("group_duty", 0.0) * duty_expr[1:]
        for neighbor_name, neighbor_idx in sl_temp.get("neighbor_indices", {}).items():
            feature_name = f"neighbor_diff::{neighbor_name}"
            if feature_name in theta_temp and (k, neighbor_idx) in sl_neighbor_vars:
                rhs = rhs + theta_temp[feature_name] * sl_neighbor_vars[(k, neighbor_idx)][:-1]
            door_feature_name = f"door_x_neighbor_diff::{neighbor_name}"
            if door_feature_name in theta_temp and (k, neighbor_idx) in sl_neighbor_vars:
                rhs = rhs + theta_temp[door_feature_name] * cp.multiply(
                    door_now, sl_neighbor_vars[(k, neighbor_idx)][:-1]
                )

        constraints.append(predicted_temp_thermal[1:] == rhs)

        # Whole-house electric-draw prediction (theta_elec_), reusing this
        # room's live decision variables but the house-level feature
        # convention described in this method's own docstring (constant
        # 20.0 room_last, zero blind/opening contribution).
        delta_supply_house_expr = supply_temp[1:] - 20.0
        delta_supply_house, _ = _linearize_relu(
            constraints, delta_supply_house_expr, -temp_delta_bound, temp_delta_bound,
            name=f"house{k}_dsup",
        )
        delta_env_house_const = np.clip(20.0 - outdoor_arr[1:], a_min=0.0, a_max=None)
        duty_x_delta_supply_house = _linearize_binary_times_continuous(
            constraints, duty_expr[1:], delta_supply_house, 0.0, temp_delta_bound,
            name=f"house{k}_dxs",
        )
        # delta_env_house_const involves no decision variable at all (it's
        # a function of the outdoor forecast only, at the fixed constant
        # room=20.0) - a plain numpy-times-Variable-slice product, already
        # affine, no McCormick needed (unlike duty_x_delta_supply_house
        # above, whose continuous factor DOES depend on supply_temp).
        duty_x_delta_env_house = cp.multiply(delta_env_house_const, duty_expr[1:])

        elec_rhs = theta_elec.get("bias", 0.0)
        elec_rhs = elec_rhs + theta_elec.get("room_last", 0.0) * 20.0
        elec_rhs = elec_rhs + theta_elec.get("duty", 0.0) * duty_expr[1:]
        elec_rhs = elec_rhs + theta_elec.get("delta_supply", 0.0) * delta_supply_house
        elec_rhs = elec_rhs + theta_elec.get("duty_x_delta_supply", 0.0) * duty_x_delta_supply_house
        elec_rhs = elec_rhs + theta_elec.get("delta_env", 0.0) * delta_env_house_const
        elec_rhs = elec_rhs + theta_elec.get("duty_x_delta_env", 0.0) * duty_x_delta_env_house
        elec_rhs = elec_rhs + theta_elec.get("cold_below_2c", 0.0) * cold_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("wind_speed", 0.0) * wind_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("wind_x_outdoor", 0.0) * wind_x_outdoor_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("dni", 0.0) * dni_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("dhi", 0.0) * dhi_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("sun_alt_sin", 0.0) * sun_alt_sin_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("dni_x_sun_az_sin", 0.0) * dni_x_sun_az_sin_arr[1:]
        elec_rhs = elec_rhs + theta_elec.get("dni_x_sun_az_cos", 0.0) * dni_x_sun_az_cos_arr[1:]
        # blind_x_dni/opening_x_outdoor: always exactly 0 at house level
        # (see docstring) - the corresponding theta_elec_ coefficients, if
        # any, are deliberately never applied.
        elec_rhs = elec_rhs + theta_elec.get("group_duty", 0.0) * duty_expr[1:]

        # Mirrors SelfLearningPhysicsModel.predict_recursive's own
        # max(0.0, theta_elec_ @ features) clip.
        predicted_elec, _ = _linearize_relu(
            constraints, elec_rhs, -elec_power_bound, elec_power_bound, name=f"house{k}_elec"
        )
        # Gate the prediction through heat_source_on before pinning this
        # load's own electric-power decision variable to it (t=1..n-1; t=0
        # is the current/anchor step, left free like every other deferrable
        # load - same convention _add_self_learning_dispatch_constraints
        # already uses for duty[0], which likewise never drives any
        # equation). Without this gate, a fitted bias/weather-term
        # combination that doesn't land at exactly 0 when duty=0 (nothing
        # forces a linear regression to fit that exactly) would conflict
        # with the heat_source_on=0 => p_deferrable<=0 cap above and make
        # the problem infeasible whenever the solver picks "off". Gating
        # guarantees p_deferrable is exactly 0 when off regardless of fit
        # noise - the physically correct answer (a compressor that isn't
        # running draws no power) and a strict superset of what the
        # p_deferrable <= nominal_power * heat_source_on cap above already
        # enforces (kept anyway as an extra safety rail while on).
        gated_predicted_elec = _linearize_binary_times_continuous(
            constraints, duty_expr[1:], predicted_elec, 0.0, elec_power_bound, name=f"house{k}_gated_elec"
        )
        constraints.append(p_deferrable[1:] == gated_predicted_elec)

        sense = utils.normalize_heat_cool_mode(
            hc.get("sense") or "heat", field_name="sense", context=f"Load {k} self_learning_dispatch"
        )
        min_temperatures_list = hc.get("min_temperatures", [])
        max_temperatures_list = hc.get("max_temperatures", [])
        min_temps_param = params.get("min_temps")
        max_temps_param = params.get("max_temps")
        if min_temps_param is not None and max_temps_param is not None:
            min_temps_arr = self._pad_temp_array(min_temperatures_list, n, 18.0)
            max_temps_arr = self._pad_temp_array(max_temperatures_list, n, 26.0)
            min_temps_arr, max_temps_arr = self._relax_opening_temp_bounds(
                min_temps_arr, max_temps_arr, opening_open_k
            )
            min_temps_param.value = min_temps_arr
            max_temps_param.value = max_temps_arr
        elif min_temps_param is not None:
            min_temps_param.value = self._pad_temp_array(min_temperatures_list, n, 18.0)
        elif max_temps_param is not None:
            max_temps_param.value = self._pad_temp_array(max_temperatures_list, n, 26.0)

        penalty_term = self._add_thermal_battery_bounds_and_penalty(
            constraints,
            k,
            hc,
            predicted_temp_thermal,
            n,
            min_temps_param,
            max_temps_param,
            min_temperatures_list,
            max_temperatures_list,
            sense,
            p_deferrable,
        )
        # Tiny tie-breaker nudge toward the lowest legal supply_temp,
        # negligible relative to any real cost/comfort magnitude (confirmed
        # via a local smoke test: with heat_source_on=0 gating everything
        # supply_temp touches down to exactly 0 - see the delta_supply
        # gating fix above - supply_temp itself is otherwise a genuinely
        # free/indifferent variable while off, so an unregularized solve can
        # report it sitting anywhere in [min_supply, curve_anchor], not just
        # the floor). Without this, the published supply_temp_target_heater{k}
        # column would swing to solver-degeneracy-driven, physically
        # meaningless values during "off" timesteps - a confusing signal for
        # a companion automation to read, even though it never affects cost
        # or comfort. penalty_term is a signed (<= 0) term added directly to
        # the Maximize objective (see the objective_expr.args[0] += callers),
        # so subtracting a small positive multiple of supply_temp makes
        # higher values marginally less attractive whenever nothing else
        # already decides between them.
        supply_temp_nudge = -1e-6 * cp.sum(supply_temp)
        penalty_term = supply_temp_nudge if penalty_term is None else penalty_term + supply_temp_nudge
        # Same rationale as _add_self_learning_dispatch_constraints: no
        # separate heating-demand quantity exists for a self-learning room.
        heating_demand_arr = np.zeros(n)
        return predicted_temp_thermal, heating_demand_arr, None, penalty_term, supply_temp, heat_source_on

    def _get_shared_thermal_tanks(self) -> list[dict]:
        """Return the configured shared_thermal_tanks list (or empty)."""
        return list(self.optim_conf.get("shared_thermal_tanks", []) or [])

    def _load_shared_tank_membership(self) -> dict[int, int]:
        """Map load index -> shared_thermal_tanks index (-1 if standalone)."""
        membership: dict[int, int] = {}
        for tank_idx, tank in enumerate(self._get_shared_thermal_tanks()):
            for k in tank.get("load_ids", []) or []:
                membership[int(k)] = tank_idx
        return membership

    def _get_load_source_config(self, k: int) -> dict:
        """Extract source-side fields for load k.

        Backward compat: reads from 'thermal_source' first, then falls back to
        'thermal_battery' (the legacy single-source location for these fields).
        """
        cfg = self.optim_conf["def_load_config"][k]
        return cfg.get("thermal_source") or cfg.get("thermal_battery") or {}

    def _add_shared_thermal_tank_constraints(self, constraints, tank_idx, data_opt, p_load):
        """Build dynamics for ONE shared thermal tank fed by MULTIPLE sources.

        Each source `k` in `tank['load_ids']` contributes
            cop_k[t] * p_deferrable[k][t] / 1000 * dt  (kWh thermal)
        where cop_k is resolved via utils.resolve_thermal_battery_cop (Carnot
        for heat pumps, flat for constant-efficiency sources like gas).

        Returns: (predicted_temp_var, heating_demand_arr, penalty_term) where
        penalty_term is the signed comfort penalty (<= 0, added to the Maximize
        objective) or None when no desired_temperatures are configured.
        """
        tank = self._get_shared_thermal_tanks()[tank_idx]
        tank_id = tank.get("id", f"tank{tank_idx}")
        required_len = self.num_timesteps
        load_ids = [int(k) for k in tank.get("load_ids", [])]
        if not load_ids:
            return None, None, None

        # Tank physics
        volume = tank["volume"]
        density = tank.get("density", 1000)
        heat_capacity = tank.get("heat_capacity", 4.186)
        if density <= 0 or heat_capacity <= 0 or volume <= 0:
            raise ValueError(
                f"Shared tank {tank_id}: positive volume/density/heat_capacity required"
            )
        conversion = 3600 / (density * heat_capacity * volume)

        start_temperature = float(tank.get("start_temperature", 20.0))
        max_temperatures_list = tank.get("max_temperatures", [])
        if not max_temperatures_list:
            raise ValueError(f"Shared tank {tank_id}: requires non-empty max_temperatures")

        base_loss = tank.get("thermal_loss", 0.045)

        # Outdoor temperature - needed for COP, demand, and the optional
        # weather-compensated min_temperature_curve.
        outdoor_temp_arr = self._get_clean_outdoor_temp(data_opt, required_len)

        # Weather-compensated minimum temperature: if `min_temperature_curve` is set,
        # the tank floor follows the heating curve (radiator emission floor). Combined
        # with any static `min_temperatures` via element-wise max so the more
        # conservative floor wins.
        min_temperatures_list = utils.resolve_min_temperatures(tank, outdoor_temp_arr, required_len)
        if not min_temperatures_list:
            raise ValueError(
                f"Shared tank {tank_id}: requires non-empty min_temperatures "
                "or min_temperature_curve"
            )

        # Heating demand resolution: same options as single-source thermal_battery
        # (draw_off_demand for hot-water tanks; physics or HDD for space heating)
        hot_water = self._resolve_draw_off_demand(tank, base_loss, required_len)
        if hot_water is not None:
            heating_demand, thermal_losses = hot_water
        else:
            thermal_losses = np.array(
                utils.calculate_thermal_loss_signed(
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    indoor_temperature=start_temperature,
                    base_loss=base_loss,
                )[:required_len]
            )
            if all(
                key in tank
                for key in ["u_value", "envelope_area", "ventilation_rate", "heated_volume"]
            ):
                indoor_target_temp = tank.get(
                    "indoor_target_temperature",
                    min_temperatures_list[0] if min_temperatures_list else 20.0,
                )
                demand = utils.calculate_heating_demand_physics(
                    u_value=tank["u_value"],
                    envelope_area=tank["envelope_area"],
                    ventilation_rate=tank["ventilation_rate"],
                    heated_volume=tank["heated_volume"],
                    indoor_target_temperature=indoor_target_temp,
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                    sense=tank.get("sense") or "heat",
                )
            elif "specific_heating_demand" in tank and "area" in tank:
                if str(tank.get("sense") or "heat").strip().lower() == "cool":
                    self.logger.warning(
                        "Shared tank %s: the degree-day (specific_heating_demand) "
                        "demand model is heating-only; sense='cool' will be treated "
                        "as heating. Configure the physics model (u_value, "
                        "envelope_area, ventilation_rate, heated_volume) for cooling "
                        "demand.",
                        tank_id,
                    )
                demand = utils.calculate_heating_demand(
                    specific_heating_demand=tank["specific_heating_demand"],
                    floor_area=tank["area"],
                    outdoor_temperature_forecast=outdoor_temp_arr.tolist(),
                    base_temperature=tank.get("base_temperature", 18.0),
                    annual_reference_hdd=tank.get("annual_reference_hdd", 3000.0),
                    optimization_time_step=int(self.freq.total_seconds() / 60),
                )
            else:
                # No heating demand model - idle tank with losses only
                demand = [0.0] * required_len
            heating_demand = np.array(demand[:required_len])

        # Apply surface solar gain if configured at the tank level
        solar_gain = utils.calculate_surface_solar_gain(
            tank,
            data_opt["ghi"].values if "ghi" in data_opt.columns else None,
            optimization_time_step_minutes=int(self.freq.total_seconds() / 60),
            length=required_len,
        )
        if solar_gain is not None:
            heating_demand = heating_demand - solar_gain

        # Per-source COP arrays (HP uses Carnot, gas / oil / district use flat
        # efficiency). Resolve each source's conversion factor from its config.
        cop_arrays: list[np.ndarray] = []
        for k in load_ids:
            src_cfg = self._get_load_source_config(k)
            cops = utils.resolve_thermal_battery_cop(
                src_cfg, outdoor_temp_arr.tolist(), length=required_len
            )
            cop_arrays.append(np.asarray(cops))

        # Comfort sense (heat vs cool). The compiler propagates the destination
        # storage's comfort_sense onto tank["sense"]; default to heat for legacy
        # configs. sense_coeff = +1 for heating (source adds heat), -1 for cooling
        # (source removes heat) — mirrors the per-load thermal paths.
        tank_sense = utils.normalize_heat_cool_mode(
            tank.get("sense") or "heat",
            field_name="sense",
            context=f"shared tank {tank_id}",
        )
        sense_coeff = 1 if tank_sense == "heat" else -1

        # Build CVXPY tank temperature variable
        predicted_temp = cp.Variable(required_len, name=f"temp_shared_{tank_id}")
        constraints.append(predicted_temp[0] == start_temperature)

        # Heat input is the SUM of contributions from all member sources
        # raw_heat[t] = sum_k(cop_k[t] * p_deferrable[k][t] / 1000 * dt)
        raw_heat = 0
        for k, cops in zip(load_ids, cop_arrays):
            p_k = self.vars["p_deferrable"][k]
            raw_heat = raw_heat + cp.multiply(cops[:-1], p_k[:-1]) / 1000 * self.time_step

        # First-order thermal dynamics
        # T[t+1] = T[t] + conversion * (sense_coeff*raw_heat[t] - demand[t] - loss[t])
        # In cool mode (sense_coeff = -1) running a source LOWERS the tank temperature.
        constraints.append(
            predicted_temp[1:]
            == predicted_temp[:-1]
            + conversion * (sense_coeff * raw_heat - heating_demand[:-1] - thermal_losses[:-1])
        )

        # Hard min/max temperature constraints (skipping index 0 - already pinned)
        min_idx = [
            i for i, v in enumerate(min_temperatures_list) if v is not None and 0 < i < required_len
        ]
        if min_idx:
            min_vals = np.array([min_temperatures_list[i] for i in min_idx])
            constraints.append(predicted_temp[min_idx] >= min_vals)
        max_idx = [
            i for i, v in enumerate(max_temperatures_list) if v is not None and 0 < i < required_len
        ]
        if max_idx:
            max_vals = np.array([max_temperatures_list[i] for i in max_idx])
            constraints.append(predicted_temp[max_idx] <= max_vals)

        # Soft comfort constraints (overshoot/desired/penalty) — same pattern as the
        # per-load thermal_battery path. Without this the hard min/max are the ONLY
        # temperature pressure, so in cool mode the zone drifts up to (just under) the
        # hard max and no cooling is ever scheduled. The signed penalty creates the
        # incentive to hold the tank near `desired_temperatures` in the comfort sense.
        penalty_expr = 0
        desired_temps_raw = tank.get("desired_temperatures", [])
        # The compiler may store a scalar desired_temperature; broadcast to horizon.
        if isinstance(desired_temps_raw, int | float):
            desired_temps_list = [float(desired_temps_raw)] * required_len
        else:
            desired_temps_list = list(desired_temps_raw)
        overshoot_temperature = tank.get("overshoot_temperature", None)

        if desired_temps_list and overshoot_temperature is not None:
            is_overshoot = cp.Variable(
                required_len, boolean=True, name=f"is_overshoot_shared_{tank_id}"
            )
            big_m = 100

            if tank_sense == "heat":
                constraints.append(
                    predicted_temp - overshoot_temperature - (big_m * is_overshoot) <= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (big_m * (1 - is_overshoot)) >= 0
                )
            else:
                constraints.append(
                    predicted_temp - overshoot_temperature - (-big_m * is_overshoot) >= 0
                )
                constraints.append(
                    predicted_temp - overshoot_temperature + (-big_m * (1 - is_overshoot)) <= 0
                )

            # Suppress every member source while the tank is in the comfortable region.
            for k in load_ids:
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    nominal_power = max(nominal_power)
                constraints.append(
                    self.vars["p_deferrable"][k] <= nominal_power * (1 - is_overshoot)
                )

        if desired_temps_list:
            penalty_factor = tank.get("penalty_factor", 10)
            valid_indices = [
                i
                for i, val in enumerate(desired_temps_list)
                if val is not None and 0 < i < required_len
            ]
            if valid_indices:
                des_temps = np.array([desired_temps_list[i] for i in valid_indices])
                # deviation in the comfort sense: heat penalises T < desired,
                # cool penalises T > desired (sense_coeff = -1 flips the sign).
                deviation = (predicted_temp[valid_indices] - des_temps) * sense_coeff
                penalty_expr = -cp.pos(-deviation * penalty_factor)

        penalty_term = None if isinstance(penalty_expr, int) else cp.sum(penalty_expr)
        return predicted_temp, heating_demand, penalty_term

    def _add_deferrable_load_constraints(
        self,
        constraints,
        data_opt,
        def_total_hours,
        def_total_timestep,
        def_start_timestep,
        def_end_timestep,
        def_init_temp,
        min_power_of_deferrable_loads,
        p_load,
        room_blind_positions=None,
        room_opening_open=None,
        room_door_open=None,
    ):
        """Master helper for all deferrable load constraints (Vectorized)."""
        p_deferrable = self.vars["p_deferrable"]
        p_def_bin1 = self.vars["p_def_bin1"]
        p_def_start = self.vars["p_def_start"]
        p_def_bin2 = self.vars["p_def_bin2"]
        p_def_stop = self.vars["p_def_stop"]

        predicted_temps = {}
        heating_demands = {}
        q_inputs = {}
        # weather_curve exact-MILP rooms only (see
        # _add_self_learning_dispatch_milp_constraints) - k -> that room's
        # own live supply-temperature decision variable, threaded through
        # to _build_results_dataframe as supply_temp_target_heater{k}.
        supply_temp_targets = {}
        penalty_terms_total = 0
        n = self.num_timesteps

        # Compute shared-tank membership once. Used by the per-load loop to
        # skip loads that belong to a shared tank (handled after the loop)
        # and again by the is_thermal_battery check below.
        shared_tank_membership = self._load_shared_tank_membership()

        # Self-learning-physics dispatch: rooms with a fitted model attached
        # (heatpump_room_self_learning_only + a successful refit, see
        # utils.py::_append_room_thermal_loads) use their own fitted
        # equation instead of the physics/RC recurrence below - see
        # _add_self_learning_dispatch_constraints. {} for every config with
        # no such room (the common case), so this is a no-op cost-wise.
        sl_rooms = self._get_self_learning_room_indices()
        # RC-physics dispatch: rooms with a fitted RC model attached
        # (heatpump_room_rc_physics_only + a successful heating-model-refit,
        # see utils.py::_append_room_thermal_loads) use
        # _add_rc_physics_dispatch_constraints instead of the physics/RC
        # recurrence below - {} for every config with no such room (the
        # common case), so this is a no-op cost-wise.
        rc_rooms = self._get_rc_physics_room_indices()
        aggregate_duty_expr = (
            self._build_aggregate_heatpump_duty_expr() if (sl_rooms or rc_rooms) else None
        )
        sl_neighbor_vars = {
            (k, j): cp.Variable(n, name=f"sl_neighbor_diff_{k}_{j}")
            for k, sl in sl_rooms.items()
            for j in sl.get("neighbor_indices", {}).values()
        }

        # Room-to-room thermal coupling: one free flow cp.Variable per pair,
        # created up front (order-independent) so _add_thermal_battery_constraints
        # can fold each room's net flow into its OWN recurrence equation during
        # the loop below - the flow variable itself only gets pinned to real
        # physics afterward, in _add_room_thermal_coupling_constraints, once
        # every room's predicted_temps[k] exists. Deliberately NOT a second,
        # independent equality bolted onto predicted_temp_thermal after the
        # loop (unlike shared_power_group's p_deferrable cap) - that would
        # force T_i == T_j at every timestep instead of real coupling, since
        # predicted_temp_thermal is already fully pinned by its own recurrence.
        # Self-learning rooms are excluded (sl_rooms passed through): they
        # express coupling natively via their own fitted neighbor_diff
        # coefficient (sl_neighbor_vars above) instead, and a room must never
        # get both mechanisms at once (double-counted coupling physics).
        # RC-physics rooms are excluded too - RC's own model has no per-room
        # coupling concept at all (single-room/whole-house scope, see the
        # design plan's own Scope section), so forcing classic T_i==T_j-
        # style coupling flow into its own, differently-shaped recurrence
        # isn't meaningful.
        room_coupling_pairs = self._get_room_thermal_coupling_pairs(
            shared_tank_membership, sl_room_indices=set(sl_rooms) | set(rc_rooms)
        )
        coupling_flow_vars = {
            (i, j): cp.Variable(n, name=f"q_couple_{i}_{j}") for (i, j, _g) in room_coupling_pairs
        }

        # Initialize max cost vector
        max_cost = self.optim_conf.get(
            "deferrable_load_max_cost", [0.0] * self.optim_conf["number_of_deferrable_loads"]
        )
        self.deferrable_with_max_cost = {}

        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            self.logger.debug(f"Processing deferrable load {k}")

            # Determine Load Type & Dynamic Big-M
            # Calculate a tight Big-M value for this specific load.
            # M must be >= max possible power to allow the binary variable to work.
            # Using a dynamic tight M significantly speeds up the solver (HiGHS/CBC).
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                # Sequence load: M = max peak of the sequence
                M = np.max(self.optim_conf["nominal_power_of_deferrable_loads"][k])
                is_sequence_load = True
            else:
                # Standard load: M = nominal power
                M = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                is_sequence_load = False

            # Safety fallback if M is 0 (e.g., mock load)
            if M <= 0:
                M = 10.0

            # Check if this load has a max cost. Defensive bounds check (matching
            # set_deferrable_max_startups/def_minimum_on_time below): a load index
            # added dynamically after the initial per-load list padding (e.g. by
            # _append_ev_deferrable_loads/_append_room_thermal_loads growing
            # number_of_deferrable_loads) may exceed the configured list length.
            has_max_cost = k < len(max_cost) and max_cost[k] > 0

            # Load Specific Constraints

            # Sequence-based Deferrable Load
            if is_sequence_load:
                power_sequence = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                sequence_length = len(power_sequence)

                # Binary variable y: which sequence to choose?
                # We essentially slice the sequence over the horizon
                y_len = n - sequence_length + 1

                # Handle case where Horizon < Sequence Length
                if y_len < 1:
                    self.logger.warning(
                        f"Deferrable load {k}: Sequence length ({sequence_length}) is longer than "
                        f"optimization horizon ({n}). The sequence will be truncated."
                    )
                    y_len = 1

                y = cp.Variable(y_len, boolean=True, name=f"y_seq_{k}")

                is_manual_load_list = self.optim_conf.get("is_manual_load", [])
                is_manual_auto = k < len(is_manual_load_list) and bool(is_manual_load_list[k])

                if has_max_cost:
                    # Choose *at most* one start time if max cost exists
                    constraints.append(cp.sum(y) <= 1)

                    # Create binary variable that tracks whether load is actually scheduled
                    load_is_scheduled = cp.Variable(boolean=True, name=f"load_is_scheduled_{k}")

                    # Constraint: if any y[k] = 1, then load_is_scheduled must = 1
                    constraints.append(cp.sum(y) == load_is_scheduled)

                    # Store for later use in objective function
                    self.deferrable_with_max_cost[k] = (max_cost[k], load_is_scheduled)

                    self.logger.debug(f"Deferrable sequence load {k}: max cost constraint added")
                elif is_manual_auto and k < len(self.param_sequence_required):
                    # Manually-committed load (see manual_load_enabled): must be able to
                    # go fully idle (param value 0) when not requested, unlike a real
                    # program_based load which always has to run somewhere. See
                    # param_sequence_required's value being set per-solve in
                    # perform_optimization from this cycle's operating_hours override.
                    constraints.append(cp.sum(y) == self.param_sequence_required[k])
                else:
                    # Constraint: Choose exactly one start time
                    constraints.append(cp.sum(y) == 1)

                # Detailed power shape constraint (Convolution-like)
                # We build the matrix explicitly here
                mat_rows = []
                for start_t in range(y_len):
                    row = np.zeros(n)
                    end_t = min(start_t + sequence_length, n)
                    seq_slice = power_sequence[: (end_t - start_t)]
                    row[start_t:end_t] = seq_slice
                    mat_rows.append(row)

                mat_np = np.array(mat_rows)  # Shape (y_len, n)

                constraints.append(p_deferrable[k] == cp.matmul(y, mat_np))

            # Thermal Deferrable Load
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_config" in self.optim_conf["def_load_config"][k]
            ):
                pred_temp, _, penalty_term = self._add_thermal_load_constraints(
                    constraints, k, data_opt, def_init_temp
                )
                predicted_temps[k] = pred_temp
                if penalty_term is not None:
                    penalty_terms_total += penalty_term

            # Thermal Battery Load - skip if this load is a member of a shared
            # thermal tank. Shared tanks are handled once per-tank after the
            # load loop.
            elif (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_battery" in self.optim_conf["def_load_config"][k]
                and k not in shared_tank_membership
            ):
                if k in sl_rooms:
                    hc_k = self.optim_conf["def_load_config"][k]["thermal_battery"]
                    if hc_k.get("supply_temp_is_decision_variable"):
                        (
                            pred_temp, heat_demand, q_input_var, penalty_term,
                            supply_temp_var, _heat_source_on_var,
                        ) = self._add_self_learning_dispatch_milp_constraints(
                            constraints, k, hc_k, data_opt, def_init_temp,
                            sl_neighbor_vars=sl_neighbor_vars,
                            room_blind_positions=room_blind_positions,
                            room_opening_open=room_opening_open,
                            room_door_open=room_door_open,
                        )
                        supply_temp_targets[k] = supply_temp_var
                    else:
                        pred_temp, heat_demand, q_input_var, penalty_term = (
                            self._add_self_learning_dispatch_constraints(
                                constraints, k, hc_k, data_opt, def_init_temp,
                                duty_expr=aggregate_duty_expr, sl_neighbor_vars=sl_neighbor_vars,
                                room_blind_positions=room_blind_positions,
                                room_opening_open=room_opening_open,
                                room_door_open=room_door_open,
                            )
                        )
                elif k in rc_rooms:
                    hc_k = self.optim_conf["def_load_config"][k]["thermal_battery"]
                    pred_temp, heat_demand, q_input_var, penalty_term = (
                        self._add_rc_physics_dispatch_constraints(
                            constraints, k, hc_k, data_opt, def_init_temp,
                            duty_expr=aggregate_duty_expr,
                            room_blind_positions=room_blind_positions,
                            room_opening_open=room_opening_open,
                            room_door_open=room_door_open,
                        )
                    )
                else:
                    pred_temp, heat_demand, q_input_var, penalty_term = (
                        self._add_thermal_battery_constraints(
                            constraints, k, data_opt, p_load, def_init_temp,
                            coupling_flow_vars=coupling_flow_vars,
                            room_blind_positions=room_blind_positions,
                            room_opening_open=room_opening_open,
                        )
                    )
                predicted_temps[k] = pred_temp
                heating_demands[k] = heat_demand
                if q_input_var is not None:
                    q_inputs[k] = q_input_var
                if penalty_term is not None:
                    penalty_terms_total += penalty_term

                # Optional coupling between DHW and a shared heatpump power budget.
                hc = self.optim_conf["def_load_config"][k]["thermal_battery"]
                coupled_idx = int(hc.get("coupled_heatpump_load_index", -1) or -1)
                shared_max = float(hc.get("hp_shared_max_power", 0.0) or 0.0)
                if shared_max > 0 and coupled_idx >= 0 and coupled_idx < self.optim_conf["number_of_deferrable_loads"] and coupled_idx != k:
                    constraints.append(p_deferrable[k] + p_deferrable[coupled_idx] <= shared_max)

            # Detect special load types that have their own energy/operation constraints
            is_thermal_load = (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_config" in self.optim_conf["def_load_config"][k]
            )
            is_thermal_battery = (
                "def_load_config" in self.optim_conf.keys()
                and len(self.optim_conf["def_load_config"]) > k
                and "thermal_battery" in self.optim_conf["def_load_config"][k]
            ) or (k in shared_tank_membership)

            # Standard Deferrable Load - Energy Constraint
            # Now using parameterized Big-M formulation to allow changing operating hours
            # without rebuilding the problem. The constraint is always added but relaxed
            # via Big-M when param_energy_active = 0.
            #
            # When active=1: sum(p) * dt >= target_energy AND sum(p) * dt <= target_energy
            #                (equivalent to equality constraint)
            # When active=0: sum(p) * dt >= target_energy - M AND sum(p) * dt <= target_energy + M
            #                (effectively unconstrained)
            #
            # Skip this constraint for special load types that have their own energy constraints:
            # - Sequence loads (defined by power profile)
            # - Thermal loads (controlled by temperature targets)
            # - Thermal battery loads (controlled by heat demand)

            # Now add the energy constraint (with optional relaxation if max cost exists)
            if (
                k < len(self.param_target_energy)
                and not is_sequence_load
                and not is_thermal_load
                and not is_thermal_battery
            ):
                if has_max_cost:
                    # Create binary variable that tracks whether load is actually scheduled
                    load_is_scheduled = cp.Variable(boolean=True, name=f"load_is_scheduled_{k}")

                    # Constraint: if any p_def_bin2[k] = 1, then load_is_scheduled must = 1
                    # This is enforced by: sum(p_def_bin2[k]) <= n * load_is_scheduled AND sum(p_def_bin2[k]) >= load_is_scheduled
                    constraints.append(cp.sum(p_def_bin2[k]) >= load_is_scheduled)
                    constraints.append(cp.sum(p_def_bin2[k]) <= n * load_is_scheduled)

                    # Store for later use in objective function
                    self.deferrable_with_max_cost[k] = (max_cost[k], load_is_scheduled)

                    self.logger.debug(f"Deferrable load {k}: max cost constraint added")

                # Big-M value: maximum possible energy consumption
                # = max_power * num_timesteps * time_step
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    nominal_power = max(nominal_power)
                M_energy = nominal_power * n * self.time_step * 2  # 2x for safety margin

                # Energy constraint: sum(p) * dt == target_energy (when active)
                # Relaxed to: target_energy - M*(1-active) <= sum(p)*dt <= target_energy + M*(1-active)
                total_energy_expr = cp.sum(p_deferrable[k]) * self.time_step

                if has_max_cost:
                    # Make energy constraint conditional on load being on
                    # When load_is_scheduled = 0: energy constraint is relaxed (Big-M)
                    # When load_is_scheduled = 1: energy constraint is enforced
                    constraints.append(
                        total_energy_expr
                        >= self.param_target_energy[k] * load_is_scheduled
                        - M_energy * (1 - load_is_scheduled * self.param_energy_active[k])
                    )
                    constraints.append(
                        total_energy_expr
                        <= self.param_target_energy[k] * load_is_scheduled
                        + M_energy * (1 - load_is_scheduled * self.param_energy_active[k])
                    )
                else:
                    # No-max-cost energy constraint
                    constraints.append(
                        total_energy_expr
                        >= self.param_target_energy[k]
                        - M_energy * (1 - self.param_energy_active[k])
                    )
                    constraints.append(
                        total_energy_expr
                        <= self.param_target_energy[k]
                        + M_energy * (1 - self.param_energy_active[k])
                    )

            # Generic Constraints (Window)

            # Time Window Logic
            # Calculate Valid Window
            if def_total_timestep and def_total_timestep[k] > 0:
                def_start, def_end, warning = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_timestep[k]),
                    n,
                )
            else:
                def_start, def_end, warning = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_hours[k] / self.time_step),
                    n,
                )
            if warning is not None:
                self.logger.warning(f"Deferrable load {k} : {warning}")

            # Apply Window Constraints using Parameterized Mask
            # This allows changing time windows without rebuilding the problem
            # The mask is set in perform_optimization() before solving
            # mask[t] = 0 forces p_deferrable[k][t] <= 0 (must be off)
            # mask[t] = 1 allows p_deferrable[k][t] <= nominal_power (can operate)
            if k < len(self.param_window_masks):
                nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                if isinstance(nominal_power, list):
                    # For time-series nominal power, use the max value for the constraint
                    nominal_power = max(nominal_power)
                constraints.append(p_deferrable[k] <= nominal_power * self.param_window_masks[k])

            # Optimization: Skip Binary Logic if Possible
            # If a load is:
            # 1. Not Sequence (handled above)
            # 2. Not Semi-Continuous (variable power allowed)
            # 3. No Min Power (min=0)
            # 4. No Startup Penalty
            # 5. Not Single Constant Start
            # Then it is a pure Continuous Variable. We can skip creating/linking binary variables.
            # This dramatically speeds up solving for thermal loads which are often continuous.

            is_semi_cont = self.optim_conf["treat_deferrable_load_as_semi_cont"][k]
            is_single_const = self.optim_conf["set_deferrable_load_single_constant"][k]
            has_min_power = min_power_of_deferrable_loads[k] > 0
            has_startup_penalty = (
                "set_deferrable_startup_penalty" in self.optim_conf
                and self.optim_conf["set_deferrable_startup_penalty"][k] > 0
            )

            # Check if we MUST use binary logic
            use_binary_logic = (
                is_sequence_load
                or is_semi_cont
                or is_single_const
                or has_min_power
                or has_startup_penalty
            )

            if use_binary_logic:
                # Standard Binary/Mixed-Integer Constraints

                # Load deactivation: when param_load_active[k] = 0, force all binary
                # variables to 0. The solver's presolve eliminates these variables
                # instantly, avoiding expensive branching on inactive loads.
                if k < len(self.param_load_active):
                    constraints.append(p_def_bin2[k] <= self.param_load_active[k])
                    constraints.append(p_def_start[k] <= self.param_load_active[k])

                # Minimum Power (if active)
                if has_min_power:
                    constraints.append(
                        p_deferrable[k] >= min_power_of_deferrable_loads[k] * p_def_bin2[k]
                    )

                # Status consistency: P_def <= M * Bin2
                # Use the Dynamic M calculated above (Critical for performance)
                constraints.append(p_deferrable[k] <= M * p_def_bin2[k])

                # Startup Detection: Start[t] >= Bin[t] - Bin[t-1]
                # Uses parameterized current state to allow warm-starting
                constraints.append(
                    p_def_start[k][0] >= p_def_bin2[k][0] - self.param_def_current_state[k]
                )
                constraints.append(p_def_start[k][1:] >= p_def_bin2[k][1:] - p_def_bin2[k][:-1])

                # Startup Limit: Start[t] + Bin[t-1] <= 1
                constraints.append(p_def_start[k][0] + self.param_def_current_state[k] <= 1)
                constraints.append(p_def_start[k][1:] + p_def_bin2[k][:-1] <= 1)

                # Max Startups Limit
                if "set_deferrable_max_startups" in self.optim_conf and k < len(
                    self.optim_conf["set_deferrable_max_startups"]
                ):
                    max_starts = self.optim_conf["set_deferrable_max_startups"][k]
                    # 0 or None means disabled/unlimited. Only apply if > 0.
                    if max_starts and max_starts > 0:
                        # The sum of all start events across the horizon cannot exceed the limit
                        constraints.append(cp.sum(p_def_start[k]) <= max_starts)

                # Minimum ON-time (min-up-time) constraint (issue #952).
                # Primary target: treat_deferrable_load_as_semi_cont loads (heat pump /
                # AC / pump) where bin2=1 forces full nominal power, making min-on
                # fully meaningful. Also works for has_min_power loads (bin2=1 implies
                # power >= min_power). For plain/default loads bin2=1 means power is in
                # [0, nominal]; min-on holds the binary ON but power may be fractional.
                # Does NOT apply to sequence loads (shaped by convolution, not bin2).
                # N == 0 -> no constraint added -> exact byte-identical no-op (default).
                # def_minimum_on_time lives in optim_conf (build-time int), so changing
                # it auto-invalidates the solver cache and triggers a full rebuild.
                # Excluded for single-constant loads: those already run as one
                # continuous block (their own currently-running pin), so a separate
                # min-on-time is redundant and could over-constrain their
                # sum(p_def_bin2) == required_timesteps equality.
                if (
                    not is_sequence_load
                    and not is_single_const
                    and "def_minimum_on_time" in self.optim_conf
                    and k < len(self.optim_conf["def_minimum_on_time"])
                ):
                    min_on_n = self._coerce_nonneg_timesteps(
                        self.optim_conf["def_minimum_on_time"][k], k, "def_minimum_on_time"
                    )
                    if min_on_n > 0:
                        # For every timestep t where p_def_start[k][t] fires (1 = rising
                        # edge), keep bin2 ON for the next min_on_n steps. Clamped to
                        # the horizon end so the constraint is never trivially infeasible.
                        # Self-protecting vs window: if a start can't fit N on-steps
                        # within its operating window, the solver simply won't start the
                        # load -> stays Optimal. (Tested by FEASIBILITY test.)
                        for t in range(n):
                            window_end = min(t + min_on_n, n)
                            constraints.append(
                                cp.sum(p_def_bin2[k][t:window_end])
                                >= (window_end - t) * p_def_start[k][t]
                            )

                # Minimum OFF-time (min-down-time) constraint (#952 follow-on).
                # Symmetric to the min-on constraint above but for the falling edge.
                # Primary target: treat_deferrable_load_as_semi_cont loads (heat pump /
                # AC / compressor) where rapid restart after stopping causes wear.
                # N == 0 -> no constraint added, no new variables -> exact no-op (default).
                # def_minimum_off_time lives in optim_conf (build-time int), so changing
                # it auto-invalidates the solver cache and triggers a full rebuild.
                # Excluded for single-constant and sequence loads (same gating as min-on).
                if (
                    not is_sequence_load
                    and not is_single_const
                    and "def_minimum_off_time" in self.optim_conf
                    and k < len(self.optim_conf["def_minimum_off_time"])
                ):
                    min_off_n = self._coerce_nonneg_timesteps(
                        self.optim_conf["def_minimum_off_time"][k], k, "def_minimum_off_time"
                    )
                    if min_off_n > 0:
                        # Declare p_def_stop[k]: falling-edge binary.
                        # stop[t] = 1 iff the load was ON at t-1 and OFF at t.
                        # Three constraints pin it tightly to the falling edge (no free DOF):
                        #   (a) stop[t] >= bin2[t-1] - bin2[t]   (lower: fires on falling edge)
                        #   (b) stop[t] <= bin2[t-1]              (upper: only fires if was ON)
                        #   (c) stop[t] <= 1 - bin2[t]            (upper: only fires if now OFF)
                        # The two upper bounds are required: without them a price-tie could
                        # force a spurious stop event, creating phantom min-off windows.
                        # At t=0 we use param_def_current_state[k] as bin2[-1].
                        stop_var = cp.Variable(n, boolean=True, name=f"p_def_stop_{k}")
                        p_def_stop[k] = stop_var

                        # t=0: edge from before-horizon state
                        constraints.append(
                            stop_var[0] >= self.param_def_current_state[k] - p_def_bin2[k][0]
                        )
                        constraints.append(stop_var[0] <= self.param_def_current_state[k])
                        constraints.append(stop_var[0] <= 1 - p_def_bin2[k][0])

                        # t=1..n-1: edge from within-horizon state
                        constraints.append(stop_var[1:] >= p_def_bin2[k][:-1] - p_def_bin2[k][1:])
                        constraints.append(stop_var[1:] <= p_def_bin2[k][:-1])
                        constraints.append(stop_var[1:] <= 1 - p_def_bin2[k][1:])

                        # Forward min-off: when load stops at t, it must stay OFF for
                        # the next min_off_n steps. Clamped to horizon end so starts
                        # near the end are self-protecting.
                        for t in range(n):
                            window_end = min(t + min_off_n, n)
                            constraints.append(
                                cp.sum(1 - p_def_bin2[k][t:window_end])
                                >= (window_end - t) * stop_var[t]
                            )

                        # Force-OFF mask: bin2[k] <= param_running_ub[k].
                        # param_running_ub[k] defaults to all-1.0 (no-op); the
                        # remainder block sets forced-off entries to 0.0.
                        # Added ONLY for active min-off loads to avoid bin2<=1 spam.
                        # (This is deliberately gated on min_off_n>0, unlike the
                        # min-on bin2>=param_running_lb mask which is added for all
                        # loads because param_running_lb pre-exists for the
                        # single-const pin; there is no such pre-existing ub.)
                        if k < len(self.param_running_ub):
                            constraints.append(p_def_bin2[k] <= self.param_running_ub[k])

                if not is_sequence_load:
                    # Force-on mask: p_def_bin2[k] >= param_running_lb[k] for all
                    # binary-logic non-sequence loads. The mask is written in the
                    # param-update block by two independent mechanisms:
                    #   - single-constant pin (currently-running single-const load)
                    #   - min-on-time remainder (issue #952; any load with N>0 and elapsed)
                    # Both write to param_running_lb; the update block takes elementwise
                    # MAX so neither overwrites the other. Default mask is all-zeros
                    # (no-op for loads where neither mechanism applies).
                    if k < len(self.param_running_lb):
                        constraints.append(p_def_bin2[k] >= self.param_running_lb[k])

                    # Single Constant Start
                    if is_single_const:
                        # Startup count: normally exactly 1 per active load.
                        # Subtract param_already_running_sc so a currently-running load
                        # requires 0 new starts (it never turned off within the horizon).
                        if k < len(self.param_load_active):
                            already_running = (
                                self.param_already_running_sc[k]
                                if k < len(self.param_already_running_sc)
                                else 0
                            )
                            constraints.append(
                                cp.sum(p_def_start[k])
                                == self.param_load_active[k] - already_running
                            )
                        else:
                            constraints.append(cp.sum(p_def_start[k]) == 1)

                        # Required timesteps constraint using Big-M parameterization
                        # When active=1: sum(bin2) == required_timesteps (tight)
                        # When active=0: sum(bin2) can be anything (relaxed)
                        if k < len(self.param_required_timesteps):
                            M_timesteps = n * 2  # Max possible timesteps * safety
                            sum_bin2 = cp.sum(p_def_bin2[k])
                            constraints.append(
                                sum_bin2
                                >= self.param_required_timesteps[k]
                                - M_timesteps * (1 - self.param_timesteps_active[k])
                            )
                            constraints.append(
                                sum_bin2
                                <= self.param_required_timesteps[k]
                                + M_timesteps * (1 - self.param_timesteps_active[k])
                            )

                    # Semi-continuous
                    if is_semi_cont:
                        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][k]
                        constraints.append(p_deferrable[k] == nominal * p_def_bin1[k])
                        constraints.append(p_def_bin1[k] == p_def_bin2[k])

            else:
                # Pure Continuous Constraints (Faster!)
                # Just bound by nominal power. No binary variables involved.
                constraints.append(p_deferrable[k] >= 0)
                constraints.append(p_deferrable[k] <= M)

                # Load deactivation parity with the binary branch above: a pure
                # continuous load has no binaries for param_load_active to act on,
                # so an inactive load (operating hours = 0, or window outside the
                # horizon) would be left as a free energy sink for surplus PV.
                # Bound it by the same activation parameter instead. Thermal loads
                # are unaffected (param_load_active is pinned to 1 for them).
                if k < len(self.param_load_active):
                    constraints.append(p_deferrable[k] <= M * self.param_load_active[k])

            # Current-power pin at t=0 (issue #605).
            # Applies to pin-eligible loads only: not semi_cont (strict p==nominal*bin),
            # not single_const (fixed-energy block; a below-nominal pin would fight the
            # required-energy target), not sequence (profile-shaped), not thermal
            # (temperature dynamics govern). Uses parametric big-M so the constraint is a
            # structural no-op when param_def_current_power_active[k]=0, enabling cache
            # reuse across calls. The same M already used for this load (computed above)
            # is reused so the bound is consistent with the p<=M*bin2 constraint.
            if (
                not is_semi_cont
                and not is_single_const
                and not is_sequence_load
                and k not in self.param_thermal
                and k < len(self.param_def_current_power)
                and k < len(self.param_def_current_power_active)
            ):
                constraints.append(
                    p_deferrable[k][0]
                    <= self.param_def_current_power[k]
                    + M * (1 - self.param_def_current_power_active[k])
                )
                constraints.append(
                    p_deferrable[k][0]
                    >= self.param_def_current_power[k]
                    - M * (1 - self.param_def_current_power_active[k])
                )

        # Process shared thermal tanks once each, after the per-load loop. Each
        # shared tank is fed by N >= 1 deferrable loads; their per-load
        # thermal_battery dynamics were skipped above.
        for tank_idx, tank in enumerate(self._get_shared_thermal_tanks()):
            shared_pred_temp, shared_demand, shared_penalty = (
                self._add_shared_thermal_tank_constraints(constraints, tank_idx, data_opt, p_load)
            )
            if shared_penalty is not None:
                penalty_terms_total += shared_penalty
            if shared_pred_temp is not None:
                # Surface the tank state on the first member load so downstream
                # publishing has a temperature column to report. Subsequent
                # members reuse the same predicted_temp.
                for k in tank.get("load_ids", []):
                    k = int(k)
                    if k not in predicted_temps:
                        predicted_temps[k] = shared_pred_temp
                    if k not in heating_demands and shared_demand is not None:
                        heating_demands[k] = shared_demand

        self._add_shared_heatpump_group_constraints(constraints)
        self._add_room_thermal_coupling_constraints(
            constraints, predicted_temps, coupling_flow_vars, room_coupling_pairs,
            room_door_open=room_door_open,
        )
        self._add_self_learning_neighbor_diff_constraints(constraints, predicted_temps, sl_neighbor_vars)

        return predicted_temps, heating_demands, penalty_terms_total, q_inputs, supply_temp_targets

    def _add_self_learning_neighbor_diff_constraints(
        self, constraints: list, predicted_temps: dict, sl_neighbor_vars: dict
    ) -> None:
        """Pin each pre-created directed neighbor-diff auxiliary variable
        (sl_neighbor_vars, created up front in _add_deferrable_load_constraints)
        to sl_neighbor_vars[(k, j)] == T_j[t-1] - T_k[t-1], now that every
        room's predicted_temps entry exists. Mirrors
        _add_room_thermal_coupling_constraints's own "free variable pinned
        once every room exists" pattern, but directed (room k's own fitted
        neighbor_diff::j coefficient need not equal room j's toward k,
        unlike the symmetric manual/learned g used by the RC coupling path) -
        so unlike that method, no (i,j) canonicalization/dedup happens here,
        each flagged room's own declared neighbors get their own variable.
        """
        for (k, j), var in sl_neighbor_vars.items():
            if k not in predicted_temps or j not in predicted_temps:
                self.logger.warning(
                    "Self-learning room %d references neighbor %d with no predicted "
                    "temperature available (shared-tank member or invalid index?) - "
                    "that neighbor_diff term is dropped from this solve.",
                    k,
                    j,
                )
                continue
            constraints.append(
                var[:-1] == predicted_temps[j][:-1] - predicted_temps[k][:-1]
            )

    def _add_shared_heatpump_group_constraints(self, constraints: list) -> None:
        """Cap the combined power of thermal_battery loads that share one
        physical heat pump (heatpump_room_shared_group != 0), so N>2 rooms
        can't jointly exceed the unit's real max output.

        This is additive to (not a replacement for) the pairwise
        coupled_heatpump_load_index/hp_shared_max_power mechanism used by
        boilers, which remains correct for exactly 2 coupled loads.
        """
        def_load_config = self.optim_conf.get("def_load_config", [])
        if not isinstance(def_load_config, list):
            return
        p_deferrable = self.vars["p_deferrable"]
        groups: dict[int, list[int]] = {}
        # Each room's own resolved unit nominal power (utils.py::
        # _append_room_thermal_loads stamps thermal_cfg["heatpump_unit_nominal_power"]
        # from heatpump_room_unit -> Heat Pump Units resolution) - a group's
        # cap must use ITS OWN unit's capacity, not the whole house's
        # aggregate (plant_conf["heatpump_nominal_power"] is now a SUM
        # across every configured unit, wrong for this purpose in a
        # multi-unit household).
        group_unit_powers: dict[int, set[float]] = {}
        for k, load_cfg in enumerate(def_load_config):
            hc = load_cfg.get("thermal_battery") if isinstance(load_cfg, dict) else None
            if not hc:
                continue
            group = int(hc.get("shared_power_group", 0) or 0)
            if group != 0:
                groups.setdefault(group, []).append(k)
                unit_power = hc.get("heatpump_unit_nominal_power")
                if unit_power is not None:
                    group_unit_powers.setdefault(group, set()).add(float(unit_power))

        for group, indices in groups.items():
            if len(indices) < 2:
                continue
            unit_powers = group_unit_powers.get(group, set())
            if len(unit_powers) > 1:
                self.logger.warning(
                    "Shared heat pump group %s: member rooms resolve to different "
                    "heat pump units (%s W) - using the smallest for safety.",
                    group,
                    sorted(unit_powers),
                )
            if unit_powers:
                heatpump_max_power = min(unit_powers)
            else:
                # No room in this group carries a resolved unit (e.g. a
                # boiler-only group) - fall back to the aggregate, the same
                # value this constraint always used before per-unit
                # resolution existed.
                heatpump_max_power = float(self.plant_conf.get("heatpump_nominal_power", 0.0) or 0.0)
            if heatpump_max_power <= 0:
                self.logger.warning(
                    "Shared heat pump group %s has %d loads but no positive heat "
                    "pump nominal power could be resolved for it - skipping group "
                    "power cap.",
                    group,
                    len(indices),
                )
                continue
            constraints.append(
                cp.sum([p_deferrable[k] for k in indices]) <= heatpump_max_power
            )

    def _add_phase_balance_constraints(self, constraints: list) -> None:
        """Cap the combined power any single electrical phase (L1/L2/L3)
        draws from - or injects into - the grid, when number_of_phases
        (System) is 2 or 3 (self.phase_labels is only ever non-empty in
        that case - see __init__). Early-returns (zero constraints
        appended) otherwise, leaving every single-phase deployment
        byte-identical.

        A pure additive safety layer: the aggregate power balance and
        objective function (_add_main_power_balance_constraints) are
        completely unchanged - tariffs bill on total energy, not per
        phase, so touching the cost side here would be wrong. Instead
        this reconstructs each phase's own net grid draw from whichever
        deferrable loads/batteries/uncontrolled load/PV are actually
        tagged to that phase (load_phase/battery_phase/
        sensor_power_load_phase/sensor_power_photovoltaics_phase) and
        caps it against maximum_power_from_grid_per_phase/
        maximum_power_to_grid_per_phase - additional to, not instead of,
        the existing whole-house maximum_power_from_grid/
        maximum_power_to_grid limit.

        Sign convention matches _add_main_power_balance_constraints
        exactly: rearranging that constraint into "net grid import" form
        gives G = p_load + p_def_sum - p_pv + p_pv_curtailment -
        p_sto_pos_total - p_sto_neg_total (p_sto_pos = discharge, a
        source; p_sto_neg = charge, a sink, already <= 0 - so
        "- p_sto_pos - p_sto_neg" correctly adds charging draw and
        subtracts discharging supply). This method computes the same
        expression restricted to whatever is tagged to each phase - a
        load/battery left unassigned (phase "") simply never appears in
        any phase's sum, the safe direction of incompleteness.

        A load/battery phase tag is parsed by _resolve_phase_tag - a
        single phase ("L1"), any "+"-joined combination ("L1+L2",
        "L1+L2+L3") for a device wired across more than one phase (power
        assumed evenly split across the named phases), or "" (excluded
        entirely). A tag naming a phase that isn't active (e.g. "L3" while
        number_of_phases=2, or "L1+L3" with the same) is excluded from
        every phase's sum and logged once as a warning, not raised - a
        stale/typo'd tag should degrade coverage visibly, never crash the
        whole optimization run (see validate_num_phases's own philosophy
        in utils.py).
        """
        if not self.phase_labels:
            return

        n = self.num_timesteps
        num_def_loads = int(self.optim_conf.get("number_of_deferrable_loads", 0) or 0)
        load_phase = self.optim_conf.get("load_phase", []) or []
        is_electric = self.optim_conf.get("is_electric_load", [True] * num_def_loads)
        p_deferrable = self.vars.get("p_deferrable", [])

        battery_phase_raw = self.plant_conf.get("battery_phase", "")
        battery_phase = (
            battery_phase_raw if isinstance(battery_phase_raw, list)
            else [battery_phase_raw] * self.n_batt
        )
        p_sto_pos_list = self.vars.get("p_sto_pos", [])
        p_sto_neg_list = self.vars.get("p_sto_neg", [])

        compute_curtailment = bool(self.plant_conf.get("compute_curtailment", False))
        p_pv_curtailment = self.vars.get("p_pv_curtailment") if compute_curtailment else None
        pv_forecast_total = self.param_pv_forecast.value
        if pv_forecast_total is None:
            pv_forecast_total = np.zeros(n)

        def _phase_limit_scalar(raw, i):
            if isinstance(raw, list):
                if len(raw) == self.n_phases:
                    return raw[i]
                return raw[0] if raw else 4000.0
            return raw if raw is not None else 4000.0

        unknown_load_labels: set[str] = set()
        unknown_batt_labels: set[str] = set()

        # Pre-classify every load/battery once (not per phase-loop
        # iteration): each entry is (weight, index) - weight=1.0 for a
        # device pinned to exactly one phase, weight=1/len(labels) for a
        # "+"-combination device that contributes to each named phase's
        # sum (see _resolve_phase_tag).
        load_terms_by_phase: dict[str, list[tuple[float, int]]] = {lbl: [] for lbl in self.phase_labels}
        for k in range(min(num_def_loads, len(p_deferrable))):
            if k < len(is_electric) and not bool(is_electric[k]):
                continue
            raw_tag = str(load_phase[k]).strip() if k < len(load_phase) else ""
            if not raw_tag:
                continue
            resolved = _resolve_phase_tag(raw_tag, self.phase_labels)
            if resolved is None:
                unknown_load_labels.add(raw_tag)
                continue
            weight = 1.0 / len(resolved)
            for lbl in resolved:
                load_terms_by_phase[lbl].append((weight, k))

        batt_terms_by_phase: dict[str, list[tuple[float, int]]] = {lbl: [] for lbl in self.phase_labels}
        for b in range(len(p_sto_pos_list)):
            raw_tag = str(battery_phase[b]).strip() if b < len(battery_phase) else ""
            if not raw_tag:
                continue
            resolved = _resolve_phase_tag(raw_tag, self.phase_labels)
            if resolved is None:
                unknown_batt_labels.add(raw_tag)
                continue
            weight = 1.0 / len(resolved)
            for lbl in resolved:
                batt_terms_by_phase[lbl].append((weight, b))

        for i, label in enumerate(self.phase_labels):
            g_phase = self.param_load_forecast_phase[label] - self.param_pv_forecast_phase[label]
            load_terms = load_terms_by_phase[label]
            if load_terms:
                g_phase = g_phase + cp.sum([w * p_deferrable[k] for w, k in load_terms])
            batt_terms = batt_terms_by_phase[label]
            if batt_terms:
                g_phase = g_phase - cp.sum(
                    [w * (p_sto_pos_list[b] + p_sto_neg_list[b]) for w, b in batt_terms]
                )
            if p_pv_curtailment is not None:
                pv_phase_value = self.param_pv_forecast_phase[label].value
                if pv_phase_value is None:
                    pv_phase_value = np.zeros(n)
                has_pv = np.abs(pv_forecast_total) > 1e-6
                curtailment_share = np.where(
                    has_pv, pv_phase_value / np.where(has_pv, pv_forecast_total, 1.0),
                    1.0 / self.n_phases,
                )
                g_phase = g_phase + cp.multiply(curtailment_share, p_pv_curtailment)

            max_import_arr = self._prepare_power_limit_array(
                _phase_limit_scalar(
                    self.plant_conf.get("maximum_power_from_grid_per_phase", 4000), i
                ),
                f"maximum_power_from_grid_per_phase[{label}]",
                n,
            )
            max_export_arr = self._prepare_power_limit_array(
                _phase_limit_scalar(
                    self.plant_conf.get("maximum_power_to_grid_per_phase", 4000), i
                ),
                f"maximum_power_to_grid_per_phase[{label}]",
                n,
            )
            constraints.append(g_phase <= max_import_arr)
            constraints.append(g_phase >= -max_export_arr)

        if unknown_load_labels:
            self.logger.warning(
                "load_phase contains phase label(s) %s not in the active phase set "
                "%s (number_of_phases=%d) - those loads are excluded from the "
                "per-phase power cap.",
                sorted(unknown_load_labels),
                self.phase_labels,
                self.n_phases,
            )
        if unknown_batt_labels:
            self.logger.warning(
                "battery_phase contains phase label(s) %s not in the active phase "
                "set %s (number_of_phases=%d) - those batteries are excluded from "
                "the per-phase power cap.",
                sorted(unknown_batt_labels),
                self.phase_labels,
                self.n_phases,
            )

    def _get_room_thermal_coupling_pairs(
        self,
        shared_tank_membership: dict[int, int] | None = None,
        sl_room_indices: set[int] | None = None,
    ) -> list[tuple[int, int, float]]:
        """Parse per-room coupled_neighbors/coupling_conductance_kw_per_k
        (heatpump_room_coupled_neighbors/heatpump_room_coupling_conductance,
        already resolved to absolute def_load_config indices by
        utils._append_room_thermal_loads) into a canonicalized, deduplicated
        list of (i, j, conductance_kw_per_k) pairs with i < j.

        A pair only needs to be declared from one side - room i listing room
        j as a neighbor is enough to couple them, room j doesn't also need
        to list room i. If both sides declare the same pair with different
        conductance values, the first one seen wins and a warning is logged
        (not a crash - a live config shouldn't fail to build over this).

        Both sides of a pair must be genuine, standalone thermal_battery
        rooms (not a shared-tank member, which is skipped by
        _add_thermal_battery_constraints entirely and gets the tank's own
        shared variable instead - not something coupled_neighbors can target)
        - otherwise the pre-created flow variable for that pair would never
        get pinned by _add_room_thermal_coupling_constraints, leaving it a
        free, unconstrained variable inside whichever room's recurrence DID
        reference it. Silently dropping the invalid half of the pair here
        avoids ever creating that dangling variable in the first place.

        sl_room_indices (self-learning-dispatch rooms, see
        _add_self_learning_dispatch_constraints) are excluded on either side
        of a pair the same way - those rooms express coupling to their
        declared neighbors natively via their own fitted neighbor_diff
        coefficient instead, and must never get both mechanisms applied to
        the same pair at once.
        """
        def_load_config = self.optim_conf.get("def_load_config", [])
        if not isinstance(def_load_config, list):
            return []
        shared_tank_membership = shared_tank_membership or {}
        sl_room_indices = sl_room_indices or set()
        num_loads = int(self.optim_conf.get("number_of_deferrable_loads", len(def_load_config)) or 0)

        def _is_valid_room(idx: int) -> bool:
            if idx < 0 or idx >= len(def_load_config) or idx in shared_tank_membership:
                return False
            if idx in sl_room_indices:
                return False
            cfg = def_load_config[idx]
            return isinstance(cfg, dict) and bool(cfg.get("thermal_battery"))

        pairs: dict[tuple[int, int], float] = {}
        for k, load_cfg in enumerate(def_load_config):
            hc = load_cfg.get("thermal_battery") if isinstance(load_cfg, dict) else None
            if not hc:
                continue
            if not _is_valid_room(k):
                # k itself is a shared-tank member or otherwise skipped by
                # _add_thermal_battery_constraints - it will never get a
                # predicted_temps entry, so any pair declared from its side
                # would leave the flow variable dangling too.
                continue
            neighbors = hc.get("coupled_neighbors", []) or []
            conductances = hc.get("coupling_conductance_kw_per_k", []) or []
            for j_raw, g_raw in zip(neighbors, conductances):
                try:
                    j = int(j_raw)
                    g = float(g_raw)
                except (TypeError, ValueError):
                    continue
                if g <= 0 or j == k or j < 0 or j >= num_loads:
                    continue
                if not _is_valid_room(j):
                    self.logger.warning(
                        "Room thermal coupling: load %d declares neighbor %d, but %d "
                        "is not a standalone room (shared-tank member, non-thermal_battery "
                        "load, or out of range) - pair skipped.",
                        k,
                        j,
                        j,
                    )
                    continue
                key = (min(k, j), max(k, j))
                if key in pairs and pairs[key] != g:
                    self.logger.warning(
                        "Room thermal coupling pair (%d, %d) declared with conflicting "
                        "conductance (%.4f vs %.4f kW/K) - keeping the first value seen.",
                        key[0],
                        key[1],
                        pairs[key],
                        g,
                    )
                    continue
                pairs[key] = g
        return [(i, j, g) for (i, j), g in pairs.items()]

    def _add_room_thermal_coupling_constraints(
        self,
        constraints: list,
        predicted_temps: dict,
        coupling_flow_vars: dict,
        room_coupling_pairs: list[tuple[int, int, float]],
        room_door_open: list | None = None,
    ) -> None:
        """Pin each pre-created coupling flow variable to the real heat-flow
        physics, now that every room's predicted_temps[k] exists (built by
        the per-load loop above). q_couple_ij = g * dt * (T_i - T_j): a
        positive flow means room i is losing heat to room j. The recurrence
        term this variable feeds into was already folded into each room's
        OWN state equation inside _add_thermal_battery_constraints (see
        coupling_flow_vars in the caller) - this method only fixes the
        variable's value, it does not touch predicted_temp_thermal directly.

        Units: conductance in kW/K, self.time_step in hours, so
        g * time_step * deltaT is kWh - matching heating_demand/thermal_losses'
        existing units and each room's own `conversion` factor.

        room_door_open: optional live per-load "door is open right now" list
        (see command_line.py::_build_room_door_open) - when either room of a
        pair currently has its door open, g is boosted by
        DOOR_OPEN_COUPLING_MULTIPLIER at the near-term timestep only (index 0,
        the same "[:-1]-consumed" family as flow_var/predicted_temps[:-1]
        themselves - see optimization.py module notes on the two index
        families). Naturally a no-op for a room with no declared neighbors,
        since room_coupling_pairs is simply empty for it - never an if/else
        on "has neighbors". This inherits the same cold-build-only cache-hit
        limitation as room_blind_positions: _add_room_thermal_coupling_constraints
        is only ever invoked from a cold/rebuilt _add_deferrable_load_constraints
        call, with no update_* refresh counterpart, since g itself is a bare
        Python float baked into the constraint, not a cp.Parameter.
        """
        n = self.num_timesteps
        for i, j, g in room_coupling_pairs:
            if i not in predicted_temps or j not in predicted_temps:
                self.logger.warning(
                    "Room thermal coupling pair (%d, %d) skipped: missing predicted "
                    "temperature for one or both loads (shared-tank member or invalid "
                    "index?).",
                    i,
                    j,
                )
                continue
            g_arr = np.full(n - 1, g)
            door_open_i = (
                room_door_open is not None and i < len(room_door_open) and room_door_open[i]
            )
            door_open_j = (
                room_door_open is not None and j < len(room_door_open) and room_door_open[j]
            )
            if door_open_i or door_open_j:
                g_arr[0] *= DOOR_OPEN_COUPLING_MULTIPLIER
            flow_var = coupling_flow_vars[(i, j)]
            # g_arr is a per-timestep numpy array now (not a scalar), so the
            # multiplication against the CVXPY predicted_temps difference
            # must go through cp.multiply - bare `*` is ambiguous/unsafe
            # under CVXPY's matmul-vs-elementwise semantics for 1-D
            # expressions once the left operand is array-valued (matches
            # this codebase's own convention elsewhere, e.g.
            # cp.multiply(heatpump_cops[:-1], p_deferrable[:-1])).
            constraints.append(
                flow_var[:-1]
                == cp.multiply(
                    g_arr * self.time_step, predicted_temps[i][:-1] - predicted_temps[j][:-1]
                )
            )

    def _add_deferrable_group_constraints(self, constraints, relaxed=False):
        """Add shared power budget and mutual exclusion constraints for deferrable load groups.

        Args:
            constraints: List of CVXPY constraints to append to.
            relaxed: If True, only add shared power budget constraints (skip mutual
                exclusion, which requires binary variables not available in the relaxed LP).
        """
        groups = self.optim_conf.get("deferrable_load_groups", [])
        if not groups:
            return

        p_deferrable = self.vars["p_deferrable"]

        for gi, group in enumerate(groups):
            indices = [int(name.replace("deferrable", "")) for name in group["names"]]
            max_power = group.get("max_power")
            mutual_exclusion = group.get("mutual_exclusion", False)

            self.logger.debug(f"Adding group {gi} constraints for deferrable loads {indices}")

            # Shared power budget: sum of group members <= max_power at each timestep
            if max_power is not None:
                group_power_sum = sum(p_deferrable[i] for i in indices)
                constraints.append(group_power_sum <= max_power)

            # Mutual exclusion: at most one load active per timestep.
            # Reuses p_def_bin2[i] for semi-continuous members; for non-semi-cont
            # members an anonymous binary plus linking constraint is added on the spot.
            # Skipped in relaxed mode (no binary variables in the LP relaxation).
            if mutual_exclusion and not relaxed:
                semi_cont = self.optim_conf["treat_deferrable_load_as_semi_cont"]
                activity_bins = []
                for i in indices:
                    if semi_cont[i]:
                        activity_bins.append(self.vars["p_def_bin2"][i])
                    else:
                        bin_var = cp.Variable(
                            self.num_timesteps,
                            boolean=True,
                            name=f"group{gi}_active_{i}",
                        )
                        self.vars["group_activity"][(gi, i)] = bin_var
                        nominal = self.optim_conf["nominal_power_of_deferrable_loads"][i]
                        if isinstance(nominal, list):
                            nominal = max(nominal)
                        constraints.append(self.vars["p_deferrable"][i] <= nominal * bin_var)
                        activity_bins.append(bin_var)
                constraints.append(cp.sum(cp.vstack(activity_bins), axis=0) <= 1)

    def _build_results_dataframe(
        self,
        data_opt,
        unit_load_cost,
        unit_prod_price,
        p_load,
        p_pv,
        soc_init,
        predicted_temps,
        heating_demands,
        debug,
        q_inputs=None,
        supply_temp_targets=None,
    ):
        """Build the final results DataFrame (Vectorized extraction)."""
        opt_tp = pd.DataFrame(index=data_opt.index)
        solver_zero_tol = 1e-9

        # Helper to safely get value or zeroes
        def get_val(var):
            if var is None:
                return np.zeros(self.num_timesteps)
            val = var.value
            if val is None:
                return np.zeros(self.num_timesteps)
            arr = np.array(val, copy=True)
            arr[np.isclose(arr, 0.0, atol=solver_zero_tol, rtol=0.0)] = 0.0
            return arr

        # Main Power Variables
        opt_tp["P_PV"] = p_pv
        opt_tp["P_Load"] = p_load

        if self.plant_conf["compute_curtailment"]:
            opt_tp["P_PV_curtailment"] = get_val(self.vars.get("p_pv_curtailment"))

        opt_tp["P_grid_pos"] = get_val(self.vars["p_grid_pos"])
        opt_tp["P_grid_neg"] = get_val(self.vars["p_grid_neg"])
        opt_tp["P_grid"] = opt_tp["P_grid_pos"] + opt_tp["P_grid_neg"]

        # Deferrable Loads
        p_def_sum = np.zeros(self.num_timesteps)
        for k in range(self.optim_conf["number_of_deferrable_loads"]):
            p_def_k = get_val(self.vars["p_deferrable"][k])
            opt_tp[f"P_deferrable{k}"] = p_def_k
            p_def_sum += p_def_k

        # Battery Results (#610). This independently recomputes the SOC/P_batt
        # recursion per battery in numpy space over realized values; it must
        # stay in lockstep with the CVXPY-expression cumsum recursion in
        # _add_battery_constraints (the per-k cap/eff_dis/eff_chg reads below
        # mirror that method exactly). ``soc_init`` is a scalar at n_batt==1
        # (unchanged from today) or a length-n_batt list at n_batt>1 (see
        # perform_optimization).
        if self.optim_conf["set_use_battery"]:
            batt_conf = self._battery_conf_as_lists()
            p_sto_pos_list = [get_val(v) for v in self.vars["p_sto_pos"]]
            p_sto_neg_list = [get_val(v) for v in self.vars["p_sto_neg"]]
            batt_stress_vars = self.vars.get("batt_stress_cost")
            soc_deficit_vars = self.vars.get("soc_deficit_cost")
            soc_surplus_vars = self.vars.get("soc_surplus_cost")

            p_batt_fleet_total = np.zeros(self.num_timesteps)
            soc_opt_list = []
            for k in range(self.n_batt):
                p_batt_k = p_sto_pos_list[k] + p_sto_neg_list[k]
                p_batt_fleet_total = p_batt_fleet_total + p_batt_k

                # Reconstruct SOC for this battery
                eff_dis_k = batt_conf["eff_dis"][k]
                eff_chg_k = batt_conf["eff_chg"][k]
                cap_k = batt_conf["cap"][k]
                power_flow_k = (p_sto_pos_list[k] * (1 / eff_dis_k)) + (
                    p_sto_neg_list[k] * eff_chg_k
                )
                energy_change_k = power_flow_k * self.time_step
                cumulative_change_k = np.cumsum(energy_change_k)
                soc_init_k = soc_init[k] if isinstance(soc_init, list) else soc_init
                soc_opt_k = soc_init_k - (cumulative_change_k / cap_k)
                soc_opt_list.append(soc_opt_k)

                if self.n_batt > 1:
                    # N>1 (#610): per-battery columns; no bare "SOC_opt" -
                    # SOC has no meaningful fleet aggregate.
                    opt_tp[f"P_batt_{k}"] = p_batt_k
                    opt_tp[f"SOC_opt_{k}"] = soc_opt_k
                    if batt_stress_vars is not None:
                        opt_tp[f"batt_stress_cost_{k}"] = get_val(batt_stress_vars[k])
                    if soc_deficit_vars is not None:
                        opt_tp[f"soc_deficit_cost_{k}"] = get_val(soc_deficit_vars[k])
                    if soc_surplus_vars is not None:
                        opt_tp[f"soc_surplus_cost_{k}"] = get_val(soc_surplus_vars[k])

            # Fleet-total P_batt always present (#610); at n_batt==1 this is
            # byte-identical to today's single "P_batt" column.
            opt_tp["P_batt"] = p_batt_fleet_total
            if self.n_batt == 1:
                # N=1: byte-identical to today - bare column names, and
                # SOC_opt is the only SOC column (no per-battery suffix).
                opt_tp["SOC_opt"] = soc_opt_list[0]
                if batt_stress_vars is not None:
                    opt_tp["batt_stress_cost"] = get_val(batt_stress_vars[0])
                if soc_deficit_vars is not None:
                    opt_tp["soc_deficit_cost"] = get_val(soc_deficit_vars[0])
                if soc_surplus_vars is not None:
                    opt_tp["soc_surplus_cost"] = get_val(soc_surplus_vars[0])

        # Hybrid Inverter Results
        if self.plant_conf["inverter_is_hybrid"]:
            opt_tp["P_hybrid_inverter"] = get_val(self.vars["p_hybrid_inverter"])
            if "inv_stress_cost" in self.vars:
                opt_tp["inv_stress_cost"] = get_val(self.vars["inv_stress_cost"])

        # Costs & Prices
        opt_tp["unit_load_cost"] = unit_load_cost
        opt_tp["unit_prod_price"] = unit_prod_price

        # Add Power Limits to Results (Required for Validation/Tests)
        n = self.num_timesteps
        opt_tp["maximum_power_from_grid"] = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_from_grid", 9000), "maximum_power_from_grid", n
        )
        opt_tp["maximum_power_to_grid"] = self._prepare_power_limit_array(
            self.plant_conf.get("maximum_power_to_grid", 9000), "maximum_power_to_grid", n
        )

        # Cost scaling factor (kW conversion and sign flip for minimization -> profit)
        scale = -0.001 * self.time_step

        if self.optim_conf["set_total_pv_sell"]:
            cost_profit = scale * (
                unit_load_cost * (p_load + p_def_sum) + unit_prod_price * opt_tp["P_grid_neg"]
            )
        else:
            cost_profit = scale * (
                unit_load_cost * opt_tp["P_grid_pos"] + unit_prod_price * opt_tp["P_grid_neg"]
            )

        opt_tp["cost_profit"] = cost_profit

        # Specific Cost Function Breakdown
        if self.costfun == "profit":
            opt_tp["cost_fun_profit"] = cost_profit

        elif self.costfun == "cost":
            if self.optim_conf["set_total_pv_sell"]:
                opt_tp["cost_fun_cost"] = scale * unit_load_cost * (p_load + p_def_sum)
            else:
                opt_tp["cost_fun_cost"] = scale * unit_load_cost * opt_tp["P_grid_pos"]

        elif self.costfun == "self-consumption":
            if "SC" in self.vars:
                opt_tp["cost_fun_selfcons"] = scale * unit_load_cost * get_val(self.vars["SC"])
            else:
                opt_tp["cost_fun_selfcons"] = cost_profit

        # Optimization Status
        opt_tp["optim_status"] = self.optim_status

        # Thermal Details
        for k, pred_temp_var in predicted_temps.items():
            temp_values = get_val(pred_temp_var)
            opt_tp[f"predicted_temp_heater{k}"] = np.round(temp_values, 2)

            if "def_load_config" in self.optim_conf:
                # Robustly get config (support both thermal_config and thermal_battery)
                load_conf = self.optim_conf["def_load_config"][k]
                conf = load_conf.get("thermal_config") or load_conf.get("thermal_battery") or {}

                # Store Target/Desired Temperatures (Legacy behavior)
                # Only look for 'desired_temperatures'.
                targets = conf.get("desired_temperatures")

                if targets:
                    tgt_series = pd.Series(targets)
                    if len(tgt_series) > len(opt_tp):
                        tgt_series = tgt_series.iloc[: len(opt_tp)]
                    tgt_series.index = opt_tp.index[: len(tgt_series)]
                    opt_tp[f"target_temp_heater{k}"] = tgt_series

                # Store Explicit Min/Max Constraints (New request)
                for bound in ["min", "max"]:
                    key = f"{bound}_temperatures"
                    if conf.get(key):
                        bound_series = pd.Series(conf[key])
                        # Align length with optimization horizon
                        if len(bound_series) > len(opt_tp):
                            bound_series = bound_series.iloc[: len(opt_tp)]
                        bound_series.index = opt_tp.index[: len(bound_series)]
                        opt_tp[f"{bound}_temp_heater{k}"] = bound_series

        for k, heat_demand in heating_demands.items():
            opt_tp[f"heating_demand_heater{k}"] = heat_demand

        if q_inputs:
            for k, q_input_var in q_inputs.items():
                q_values = get_val(q_input_var)
                opt_tp[f"q_input_heater{k}"] = np.round(q_values, 4)

        # weather_curve exact-MILP rooms only (see
        # _add_self_learning_dispatch_milp_constraints) - the room's own
        # solved supply-temperature target, published alongside (not
        # instead of) predicted_temp_heater{k} so the user can choose which
        # signal to wire into their own automation. Duty needs no separate
        # column - it's already derivable from this load's own solved
        # P_deferrable, same convention _publish_heatpump_dispatch_target
        # already uses.
        if supply_temp_targets:
            for k, supply_temp_var in supply_temp_targets.items():
                opt_tp[f"supply_temp_target_heater{k}"] = np.round(get_val(supply_temp_var), 2)

        # Debug Columns
        if debug:
            for k in range(self.optim_conf["number_of_deferrable_loads"]):
                opt_tp[f"P_def_start_{k}"] = get_val(self.vars["p_def_start"][k])
                opt_tp[f"P_def_bin2_{k}"] = get_val(self.vars["p_def_bin2"][k])

        return opt_tp

    def perform_optimization(
        self,
        data_opt: pd.DataFrame,
        p_pv: np.array,
        p_load: np.array,
        unit_load_cost: np.array,
        unit_prod_price: np.array,
        soc_init: float | list | None = None,
        soc_final: float | list | None = None,
        soc_target: float | None = None,
        soc_target_timestep: int | None = None,
        current_period_peak: float | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
        def_init_temp: list | None = None,
        min_power_of_deferrable_loads: list | None = None,
        debug: bool | None = False,
        stage_times: dict[str, float] | None = None,
        room_blind_positions: list | None = None,
        room_opening_open: list | None = None,
        room_door_open: list | None = None,
    ) -> pd.DataFrame:
        """
        Public entry point. Delegates straight to `_perform_optimization_core`
        UNLESS at least one room is flagged `heatpump_room_self_learning_only`
        (with `hc["self_learning_dispatch"]` attached) and/or
        `heatpump_room_rc_physics_only` (with `hc["rc_physics_dispatch"]`
        attached, set by utils.py::_append_room_thermal_loads) - in that
        case a two-pass solve is required, see `_perform_two_pass_optimization`
        for why (non-convex bilinear terms in either fitted/physics model
        need a reference trajectory, itself only obtainable from a solve).
        """
        sl_rooms = self._get_self_learning_room_indices()
        rc_rooms = self._get_rc_physics_room_indices()
        if not sl_rooms and not rc_rooms:
            return self._perform_optimization_core(
                data_opt,
                p_pv,
                p_load,
                unit_load_cost,
                unit_prod_price,
                soc_init=soc_init,
                soc_final=soc_final,
                soc_target=soc_target,
                soc_target_timestep=soc_target_timestep,
                current_period_peak=current_period_peak,
                def_total_hours=def_total_hours,
                def_total_timestep=def_total_timestep,
                def_start_timestep=def_start_timestep,
                def_end_timestep=def_end_timestep,
                def_init_temp=def_init_temp,
                min_power_of_deferrable_loads=min_power_of_deferrable_loads,
                debug=debug,
                stage_times=stage_times,
                room_blind_positions=room_blind_positions,
                room_opening_open=room_opening_open,
                room_door_open=room_door_open,
            )
        return self._perform_two_pass_optimization(
            sl_rooms,
            rc_rooms,
            data_opt,
            p_pv,
            p_load,
            unit_load_cost,
            unit_prod_price,
            soc_init=soc_init,
            soc_final=soc_final,
            soc_target=soc_target,
            soc_target_timestep=soc_target_timestep,
            current_period_peak=current_period_peak,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            def_init_temp=def_init_temp,
            min_power_of_deferrable_loads=min_power_of_deferrable_loads,
            debug=debug,
            stage_times=stage_times,
            room_blind_positions=room_blind_positions,
            room_opening_open=room_opening_open,
            room_door_open=room_door_open,
        )

    def _get_self_learning_room_indices(self) -> dict[int, dict]:
        """k -> hc["self_learning_dispatch"] for every thermal_battery load
        that has a fitted self-learning dispatch model attached, or {}
        entirely while a reference (RC) pass is being forced (see
        _perform_two_pass_optimization) - during that pass
        every room, flagged or not, must use its ordinary physics/simple
        recurrence so a reference trajectory can be produced for the flagged
        ones. A room stays on the physics/simple path (this returns nothing
        for it) whenever heatpump_room_self_learning_only is set but no
        successful self-learning-physics-refit has produced a model covering
        it yet - utils.py logs a warning in that case, this does not.
        """
        if getattr(self, "_self_learning_force_rc_pass", False):
            return {}
        out: dict[int, dict] = {}
        def_load_config = self.optim_conf.get("def_load_config", []) or []
        for k, cfg in enumerate(def_load_config):
            hc = cfg.get("thermal_battery") if isinstance(cfg, dict) else None
            if isinstance(hc, dict) and hc.get("self_learning_dispatch"):
                out[k] = hc["self_learning_dispatch"]
        return out

    def _get_rc_physics_room_indices(self) -> dict[int, dict]:
        """k -> hc["rc_physics_dispatch"] for every thermal_battery load
        that has a fitted RC-physics model attached
        (heatpump_room_rc_physics_only + a successful heating-model-refit,
        see utils.py::_append_room_thermal_loads), or {} entirely while a
        reference pass is being forced - same
        `_self_learning_force_rc_pass` flag self-learning-physics's own
        indexer checks (see that method's own docstring): RC's own q_emit
        update has the identical "duty times a live-state-dependent clamp"
        nonlinearity, so it needs the same forced-reference-pass treatment,
        not a second independent mechanism. A room stays on the physics/
        simple path (this returns nothing for it) whenever
        heatpump_room_rc_physics_only is set but no successful
        heating-model-refit/tune has produced a model yet - utils.py logs a
        warning in that case, this does not.
        """
        if getattr(self, "_self_learning_force_rc_pass", False):
            return {}
        out: dict[int, dict] = {}
        def_load_config = self.optim_conf.get("def_load_config", []) or []
        for k, cfg in enumerate(def_load_config):
            hc = cfg.get("thermal_battery") if isinstance(cfg, dict) else None
            if isinstance(hc, dict) and hc.get("rc_physics_dispatch"):
                out[k] = hc["rc_physics_dispatch"]
        return out

    def _self_learning_needs_reference_pass(
        self, sl_rooms: dict[int, dict], rc_rooms: dict[int, dict], required_len: int
    ) -> bool:
        """Whether a fresh reference (RC) pass is needed before the real
        self-learning/RC-physics-driven solve, or whether the previous
        solve's own output can be reused as this tick's reference
        trajectory (see _perform_two_pass_optimization's docstring for why
        reusing the model's own prior output is sound, not just an
        optimization: predicted_temp_thermal[0] is re-pinned to a real
        sensor reading every tick regardless via def_init_temp)."""
        max_age = int(
            self.optim_conf.get("self_learning_physics_dispatch_max_cache_age_solves", 6) or 0
        )
        if max_age <= 0:
            return True
        if getattr(self, "_sl_last_solve_status", None) not in ("optimal", "optimal_inaccurate"):
            return True
        cache = getattr(self, "_sl_reference_trajectories", None) or {}
        signatures = getattr(self, "_sl_reference_signature", None) or {}
        for k, sl in sl_rooms.items():
            ref = cache.get(k)
            sig = (
                required_len,
                tuple(sl.get("feature_names", [])),
                tuple(sorted(sl.get("neighbor_indices", {}).items())),
            )
            if ref is None or len(ref) != required_len or signatures.get(k) != sig:
                return True
        rc_cache = getattr(self, "_rc_reference_trajectories", None) or {}
        rc_signatures = getattr(self, "_rc_reference_signature", None) or {}
        for k, rc in rc_rooms.items():
            ref = rc_cache.get(k)
            sig = (required_len, tuple(sorted(rc.get("params", {}).items())))
            if ref is None or len(ref) != required_len or rc_signatures.get(k) != sig:
                return True
        return getattr(self, "_sl_cache_solve_count", 0) >= max_age

    def _capture_self_learning_reference(self, sl_rooms: dict[int, dict], required_len: int) -> None:
        """Stash every flagged room's just-solved predicted_temp_thermal as
        the reference trajectory for the next tick's bilinear-term
        linearization (see _add_self_learning_dispatch_constraints)."""
        self._sl_reference_trajectories = getattr(self, "_sl_reference_trajectories", {}) or {}
        self._sl_reference_signature = getattr(self, "_sl_reference_signature", {}) or {}
        predicted_temps = getattr(self, "predicted_temps", {}) or {}
        for k, sl in sl_rooms.items():
            var = predicted_temps.get(k)
            if var is not None and getattr(var, "value", None) is not None:
                self._sl_reference_trajectories[k] = np.array(var.value, dtype=float)
                self._sl_reference_signature[k] = (
                    required_len,
                    tuple(sl.get("feature_names", [])),
                    tuple(sorted(sl.get("neighbor_indices", {}).items())),
                )
        self._sl_last_solve_status = self.prob.status if self.prob is not None else None

    def _capture_rc_physics_reference(self, rc_rooms: dict[int, dict], required_len: int) -> None:
        """Stash every RC-physics-flagged room's just-solved
        predicted_temp_thermal (T_air) as the reference trajectory for the
        next tick's max(supply-air,0) linearization (see
        _add_rc_physics_dispatch_constraints) - sibling of
        _capture_self_learning_reference, same mechanism, keyed by the
        room's own fitted params instead of feature_names/neighbor_indices."""
        self._rc_reference_trajectories = getattr(self, "_rc_reference_trajectories", {}) or {}
        self._rc_reference_signature = getattr(self, "_rc_reference_signature", {}) or {}
        predicted_temps = getattr(self, "predicted_temps", {}) or {}
        for k, rc in rc_rooms.items():
            var = predicted_temps.get(k)
            if var is not None and getattr(var, "value", None) is not None:
                self._rc_reference_trajectories[k] = np.array(var.value, dtype=float)
                self._rc_reference_signature[k] = (
                    required_len,
                    tuple(sorted(rc.get("params", {}).items())),
                )

    def _perform_two_pass_optimization(
        self, sl_rooms: dict[int, dict], rc_rooms: dict[int, dict], data_opt, p_pv, p_load,
        unit_load_cost, unit_prod_price, **kwargs,
    ) -> pd.DataFrame:
        """Two-pass solve for houses with >=1 heatpump_room_self_learning_only
        and/or heatpump_room_rc_physics_only room with a fitted dispatch
        model. Shared orchestrator for both room types - a room's fitted
        model determines WHICH equation it dispatches through
        (_add_self_learning_dispatch_constraints vs.
        _add_rc_physics_dispatch_constraints), but both need the identical
        reference-pass machinery below, so one orchestrator serves both
        rather than two independent copies.

        Why two passes: self-learning-physics's own duty_x_delta_supply/
        duty_x_delta_env features, AND RC's own q_emit update
        (duty * max(supply - air, 0)), are each a product of the room's OWN
        dispatched-power decision and its OWN temperature decision - a
        genuine bilinear (non-convex) term CVXPY cannot represent in an
        equality constraint. The fix (successive linearization, a standard
        MPC technique): evaluate just the temperature-dependent piece of
        those terms against a REFERENCE temperature trajectory (plain
        numpy, not a CVXPY expression) so they become fixed per-timestep
        coefficients multiplying the (still fully live/decision-variable)
        duty term - affine, solvable. Every other fitted/physics term is
        genuinely affine already and needs no freezing - see
        _add_self_learning_dispatch_constraints/_add_rc_physics_dispatch_constraints
        for the term-by-term proof in each case.

        Pass 1 (reference): forces every room - flagged or not, either
        type - onto its ordinary physics/simple recurrence (today's exact,
        unchanged behavior) by temporarily emptying
        _get_self_learning_room_indices/_get_rc_physics_room_indices
        (both keyed off the SAME `_self_learning_force_rc_pass` flag - one
        forced pass serves both room types at once), purely to obtain a
        temperature trajectory for the flagged rooms to linearize against.
        Skipped when a still-valid cached trajectory exists from a recent
        previous solve (see _self_learning_needs_reference_pass) - an MPC
        loop calling this every few minutes should not double its solve
        time on every single tick just to keep re-deriving a reference
        that barely moves.

        Pass 2 (real): flagged rooms now take their own dispatch branch,
        using the reference (fresh or cached) to linearize their own
        bilinear term(s); every other constraint (battery, PV, other
        loads, comfort bounds) is identical to a normal solve. This pass's
        own output becomes the reference for the NEXT call, which is a
        better reference than pass 1's physics-model guess would be
        (closer to what each room's own fitted/physics model actually
        predicts) - a standard warm-started-linearization-point pattern.

        Cost: forgoes CVXPY warm-starting entirely (both passes force
        `self.prob = None`) for the whole shared problem (not just the
        flagged rooms) on every call that needs a fresh pass 1 - explicit,
        documented, and only ever paid by installs that opt into either
        feature.
        """
        required_len = len(data_opt)
        stage_times = kwargs.get("stage_times")
        if self._self_learning_needs_reference_pass(sl_rooms, rc_rooms, required_len):
            self.logger.info(
                "Two-pass dispatch: running reference (physics-model) pass for self-learning "
                "room(s) %s and/or RC-physics room(s) %s before the real solve.",
                list(sl_rooms), list(rc_rooms),
            )
            self._self_learning_force_rc_pass = True
            self.prob = None
            ref_stage_times = {} if stage_times is not None else None
            ref_kwargs = dict(kwargs)
            ref_kwargs["stage_times"] = ref_stage_times
            self._perform_optimization_core(
                data_opt, p_pv, p_load, unit_load_cost, unit_prod_price, **ref_kwargs
            )
            if stage_times is not None and ref_stage_times:
                stage_times["optim_solve.two_pass_reference_pass"] = sum(ref_stage_times.values())
            self._capture_self_learning_reference(sl_rooms, required_len)
            self._capture_rc_physics_reference(rc_rooms, required_len)
            self._sl_cache_solve_count = 0

        self._self_learning_force_rc_pass = False
        self.prob = None
        final_res = self._perform_optimization_core(
            data_opt, p_pv, p_load, unit_load_cost, unit_prod_price, **kwargs
        )
        self._capture_self_learning_reference(sl_rooms, required_len)
        self._capture_rc_physics_reference(rc_rooms, required_len)
        self._sl_cache_solve_count = getattr(self, "_sl_cache_solve_count", 0) + 1
        return final_res

    def _perform_optimization_core(
        self,
        data_opt: pd.DataFrame,
        p_pv: np.array,
        p_load: np.array,
        unit_load_cost: np.array,
        unit_prod_price: np.array,
        soc_init: float | list | None = None,
        soc_final: float | list | None = None,
        soc_target: float | None = None,
        soc_target_timestep: int | None = None,
        current_period_peak: float | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
        def_init_temp: list | None = None,
        min_power_of_deferrable_loads: list | None = None,
        debug: bool | None = False,
        stage_times: dict[str, float] | None = None,
        room_blind_positions: list | None = None,
        room_opening_open: list | None = None,
        room_door_open: list | None = None,
    ) -> pd.DataFrame:
        r"""
        Perform the actual optimization using Convex Programming (CVXPY).
        Includes automatic fallback to relaxed LP if MILP fails or times out.

        If ``stage_times`` is provided, the wall-clock duration of three
        internal phases is recorded under the keys ``optim_solve.build``,
        ``optim_solve.solve`` and ``optim_solve.extract``. These nest under
        the existing ``optim_solve`` parent timer in ``command_line.py`` and
        sum to it within a few milliseconds.

        ``soc_init``/``soc_final`` accept either a bare float (broadcast to
        every battery - the single-battery calling convention, unchanged at
        ``number_of_batteries == 1``) or a list of exactly
        ``number_of_batteries`` entries (#610); passing per-battery values
        through from command_line.py at runtime is command_line.py's job -
        this signature already accepts the list shape today.
        """
        _build_start_perf = time.perf_counter() if stage_times is not None else 0.0
        # Dynamic Resizing
        # If the input data length differs from the initialized N, we must rebuild the problem.
        current_n = len(data_opt)
        if current_n != self.num_timesteps:
            self.logger.info(
                f"Resizing optimization problem from {self.num_timesteps} to {current_n} timesteps."
            )
            self.num_timesteps = current_n

            # Re-initialize Parameters with new shape
            self.param_pv_forecast = cp.Parameter(current_n, name="pv_forecast")
            self.param_load_forecast = cp.Parameter(current_n, name="load_forecast")
            self.param_load_forecast_phase = {
                lbl: cp.Parameter(current_n, name=f"load_forecast_{lbl}")
                for lbl in self.phase_labels
            }
            self.param_pv_forecast_phase = {
                lbl: cp.Parameter(current_n, nonneg=True, name=f"pv_forecast_{lbl}")
                for lbl in self.phase_labels
            }
            self.param_load_cost = cp.Parameter(current_n, name="load_cost")
            self.param_load_cost_pos = cp.Parameter(current_n, nonneg=True, name="load_cost_pos")
            self.param_export_ceiling = cp.Parameter(current_n, nonneg=True, name="export_ceiling")
            self.param_prod_price = cp.Parameter(current_n, name="prod_price")
            self.param_cost_per_load = [
                cp.Parameter(current_n, name=f"cost_per_load_{k}")
                for k in range(self.optim_conf.get("number_of_deferrable_loads", 0))
            ]

            # Re-initialize SOC recovery parameters with the new horizon
            self._init_soc_recovery_params()

            # Re-initialize the intermediate SOC target mask with the new horizon (#553)
            self._init_soc_target_params()

            # NOTE: param_current_period_peak (issue #623, Phase 2) is a SCALAR
            # cp.Parameter, horizon-independent, so it is intentionally NOT
            # re-created on resize (unlike the soc_target vector floor above).
            # _initialize_decision_variables (re-called below) re-appends its
            # floor constraint against the same persistent scalar parameter.

            # Re-initialize deferrable load parameters (window masks and energy constraints)
            self._init_deferrable_load_params()

            # Re-initialize Variables & Constraints
            self.vars, self.constraints = self._initialize_decision_variables()

            # Force problem rebuild
            self.prob = None

        # Data Validation & Defaults (#610: per-battery lists, k in
        # range(self.n_batt); a bare scalar/None argument broadcasts to every
        # battery via _normalize_soc_arg, so at n_batt==1 this is exactly
        # today's single-battery cross-fallback logic applied to a 1-element
        # list).
        batt_conf = self._battery_conf_as_lists() if self.optim_conf["set_use_battery"] else None
        if self.optim_conf["set_use_battery"]:
            soc_init_list = self._normalize_soc_arg(soc_init)
            soc_final_list = self._normalize_soc_arg(soc_final)
            target_list = batt_conf["soc_target"]
            for k in range(self.n_batt):
                if soc_init_list[k] is None:
                    if soc_final_list[k] is not None:
                        soc_init_list[k] = soc_final_list[k]
                    else:
                        soc_init_list[k] = target_list[k]
                if soc_final_list[k] is None:
                    if soc_init_list[k] is not None:
                        soc_final_list[k] = soc_init_list[k]
                    else:
                        soc_final_list[k] = target_list[k]
            self.logger.debug(
                f"Battery usage enabled. Initial SOC: {soc_init_list}, Final SOC: {soc_final_list}"
            )
        else:
            soc_init_list = None
            soc_final_list = None

        # Optional intermediate SOC target (issue #553).
        # Reset the floor on EVERY call so a target from a previous run is
        # cleared (the constraint is then a no-op). When a target is requested,
        # clamp it to the configured SOC bounds and build the floor vector
        # numerically (target energy at the requested horizon timestep, 0.0
        # elsewhere). The np multiply happens here at set-time, so the problem
        # stays DPP / warm-start safe.
        #
        # #610: soc_target itself is not yet a per-battery runtime input, so
        # the identical target FRACTION is applied to every battery's floor,
        # each against ITS OWN capacity/charge-power (param_soc_target_floor
        # is already a per-battery Parameter list, so a future per-battery
        # target only needs to change the value each entry receives).
        if self.optim_conf["set_use_battery"] and soc_target is not None:
            soc_target_raw = float(soc_target)
            for k in range(self.n_batt):
                soc_min_k = batt_conf["soc_min"][k]
                soc_max_k = batt_conf["soc_max"][k]
                soc_target_clamped = min(max(soc_target_raw, soc_min_k), soc_max_k)
                if soc_target_timestep is None:
                    k_target = self.num_timesteps - 1
                else:
                    k_target = min(max(int(float(soc_target_timestep)), 0), self.num_timesteps - 1)
                # Observability: warn (do not change the constraint) when the request
                # was out of range or appears unreachable in time given charge power.
                if soc_target_raw < soc_min_k or soc_target_raw > soc_max_k:
                    self.logger.warning(
                        f"Battery {k}: passed soc_target={soc_target_raw} is outside "
                        f"[{soc_min_k}, {soc_max_k}], clamping to soc_target={soc_target_clamped}"
                    )
                cap_k = batt_conf["cap"][k]
                # Max stored-energy gain per step is battery_charge_power_max * time_step:
                # the charge constraint caps grid-side power at max_chg / eff so the
                # battery-side energy added (grid * eff) is max_chg * time_step, i.e. the
                # charge efficiency cancels. (Do not multiply by efficiency again here, or
                # the bound under-estimates reach and warns spuriously when eff < 1.)
                reach = (
                    soc_init_list[k]
                    + (batt_conf["charge_power_max"][k] * self.time_step * (k_target + 1)) / cap_k
                )
                if soc_target_clamped > reach + 1e-6:
                    self.logger.warning(
                        f"Battery {k}: intermediate soc_target={soc_target_clamped} may be "
                        f"unreachable by timestep {k_target}: from "
                        f"soc_init={soc_init_list[k]} the maximum reachable SoC is "
                        f"~{reach:.3f} given battery_charge_power_max; the optimization "
                        "may be infeasible."
                    )
                floor = np.zeros(self.num_timesteps)
                floor[k_target] = soc_target_clamped * cap_k
                self.param_soc_target_floor[k].value = floor
                self.logger.debug(
                    f"Battery {k}: intermediate SOC target enabled: SoC >= "
                    f"{soc_target_clamped} by timestep {k_target} (requested "
                    f"soc_target={soc_target}, soc_target_timestep={soc_target_timestep})."
                )
        else:
            for k in range(self.n_batt):
                self.param_soc_target_floor[k].value = np.zeros(self.num_timesteps)

        # Peak grid import already incurred this billing period (issue #623,
        # Phase 2). Reset on EVERY call so a value from a previous MPC tick does
        # not leak. Only meaningful when the capacity charge is active (the
        # peak_import variable and its floor constraint exist only then); when
        # the charge is off this just resets the unused parameter. The value is
        # in WATTS to match p_grid_pos / peak_import, so no scaling is needed. A
        # non-numeric, non-finite (NaN/inf) or negative runtime value falls back to 0.0 (prices
        # the full horizon peak == Phase 1) with a warning rather than crashing.
        if self._get_capacity_cost_per_kw() > 0 and current_period_peak is not None:
            try:
                peak_floor_w = float(current_period_peak)
            except (TypeError, ValueError):
                self.logger.warning(
                    f"Invalid current_period_peak value ({current_period_peak!r}); "
                    "ignoring it (no incurred-peak floor applied)."
                )
                peak_floor_w = 0.0
            # not isfinite(...) catches NaN and +/-inf; the second clause catches
            # negatives. cp.Parameter(nonneg=True) rejects inf, so guard it here.
            if not isfinite(peak_floor_w) or peak_floor_w < 0:
                self.logger.warning(
                    f"current_period_peak must be a finite number >= 0 (Watts), got "
                    f"{current_period_peak!r}; ignoring it (no incurred-peak floor applied)."
                )
                peak_floor_w = 0.0
            self.param_current_period_peak.value = peak_floor_w
            if peak_floor_w > 0:
                self.logger.debug(
                    f"Capacity charge: flooring peak_import at already-incurred "
                    f"current_period_peak = {peak_floor_w} W."
                )
        else:
            self.param_current_period_peak.value = 0.0

        # Pad deferrable load lists
        if def_total_timestep is not None:
            if def_total_hours is None:
                def_total_hours = self.optim_conf["operating_hours_of_each_deferrable_load"]
            def_total_hours = [0 if x != 0 else x for x in def_total_hours]
        elif def_total_hours is None:
            def_total_hours = self.optim_conf["operating_hours_of_each_deferrable_load"]

        if def_start_timestep is None:
            def_start_timestep = self.optim_conf["start_timesteps_of_each_deferrable_load"]
        if def_end_timestep is None:
            def_end_timestep = self.optim_conf["end_timesteps_of_each_deferrable_load"]

        if def_init_temp is None:
            def_init_temp = [None] * self.optim_conf["number_of_deferrable_loads"]

        if room_blind_positions is None:
            room_blind_positions = [None] * self.optim_conf["number_of_deferrable_loads"]

        if room_opening_open is None:
            room_opening_open = [False] * self.optim_conf["number_of_deferrable_loads"]
        if room_door_open is None:
            room_door_open = [False] * self.optim_conf["number_of_deferrable_loads"]

        num_deferrable_loads = self.optim_conf["number_of_deferrable_loads"]

        # Ensure min_power_of_deferrable_loads is available
        if min_power_of_deferrable_loads is None:
            min_power_of_deferrable_loads = self.optim_conf.get(
                "minimum_power_of_deferrable_loads", [0] * num_deferrable_loads
            )

        def pad_list(input_list, target_len, fill=0):
            if input_list is None:
                return [fill] * target_len
            return input_list + [fill] * (target_len - len(input_list))

        min_power_of_deferrable_loads = pad_list(
            min_power_of_deferrable_loads, num_deferrable_loads
        )
        def_total_hours = pad_list(def_total_hours, num_deferrable_loads)
        def_start_timestep = pad_list(def_start_timestep, num_deferrable_loads)
        def_end_timestep = pad_list(def_end_timestep, num_deferrable_loads)
        # Normalize any None elements to 0 (treat as "no time restriction").
        # params.pkl can be corrupted by partial set-config calls that produce
        # [None, 0] instead of [0, 0], causing TypeError in validate_def_timewindow.
        def_start_timestep = [s if s is not None else 0 for s in def_start_timestep]
        def_end_timestep = [e if e is not None else 0 for e in def_end_timestep]

        # Parameter Updates
        self.param_pv_forecast.value = p_pv
        self.param_load_forecast.value = p_load
        # Per-phase load/PV, only when the feature is active (self.phase_labels
        # is only ever non-empty when number_of_phases > 1 - see __init__).
        # Columns are optional (data_opt.get pattern, same as ghi/wind_speed
        # elsewhere) - missing on this particular call (e.g. the
        # perfect-forecast-optim path, which doesn't compute them) degrades to
        # 0 rather than raising, matching _add_phase_balance_constraints'
        # own "safe direction of incompleteness" behavior for that phase's
        # uncontrolled load/PV share.
        for lbl in self.phase_labels:
            load_col = f"p_load_phase_{lbl}"
            pv_col = f"p_pv_phase_{lbl}"
            self.param_load_forecast_phase[lbl].value = (
                data_opt[load_col].to_numpy(dtype=float)
                if load_col in data_opt.columns
                else np.zeros(self.num_timesteps)
            )
            self.param_pv_forecast_phase[lbl].value = (
                data_opt[pv_col].to_numpy(dtype=float)
                if pv_col in data_opt.columns
                else np.zeros(self.num_timesteps)
            )
        self.param_load_cost.value = unit_load_cost
        self.param_load_cost_pos.value = np.maximum(np.asarray(unit_load_cost, dtype=float), 0.0)
        self.param_export_ceiling.value = np.maximum(
            np.asarray(p_pv, dtype=float) - np.asarray(p_load, dtype=float), 0.0
        )
        # Price the terminal-SoC miss well above the dearest import slot so the target is
        # never traded away for energy cost; the 0.001 converts the Wh slacks to kWh. The
        # floor keeps the penalty meaningful when every tariff is zero or negative.
        self.param_soc_final_penalty.value = (
            0.001
            * SOC_FINAL_DEVIATION_PENALTY_FACTOR
            * max(float(np.max(np.maximum(np.asarray(unit_load_cost, dtype=float), 0.0))), 1e-3)
        )
        self.param_prod_price.value = unit_prod_price

        # Per-load cost forecast overrides. Default each load's per-timestep cost
        # to the shared electricity tariff (no-op adjustment in the objective). If
        # the user provides `cost_forecast_per_deferrable_load`, slot the override
        # array into the corresponding parameter.
        cost_per_load_overrides = self.optim_conf.get("cost_forecast_per_deferrable_load", None)
        if cost_per_load_overrides is not None and not isinstance(
            cost_per_load_overrides, (list | tuple)
        ):
            self.logger.warning(
                "cost_forecast_per_deferrable_load is set but is %s, not a list (value: %r). "
                "Treating as 'no override' for all loads. Use JSON null or an array of "
                'per-load arrays (not the string "null").',
                type(cost_per_load_overrides).__name__,
                cost_per_load_overrides,
            )
            cost_per_load_overrides = None
        for k, param in enumerate(self.param_cost_per_load):
            override = (
                cost_per_load_overrides[k]
                if cost_per_load_overrides is not None and k < len(cost_per_load_overrides)
                else None
            )
            if override is None:
                param.value = np.asarray(unit_load_cost, dtype=float)
            elif not isinstance(override, (list | tuple)):
                self.logger.warning(
                    "cost_forecast_per_deferrable_load[%d] is %s (value: %r), expected list. "
                    "Falling back to shared tariff for this load.",
                    k,
                    type(override).__name__,
                    override,
                )
                param.value = np.asarray(unit_load_cost, dtype=float)
            else:
                override_arr = np.asarray(override, dtype=float)
                if len(override_arr) < self.num_timesteps:
                    # Pad with the global cost so missing tail timesteps don't
                    # accidentally apply a zero-cost override.
                    pad_len = self.num_timesteps - len(override_arr)
                    pad_tail = np.asarray(unit_load_cost, dtype=float)[len(override_arr) :]
                    if len(pad_tail) != pad_len:
                        pad_tail = np.full(pad_len, float(unit_load_cost[-1]))
                    override_arr = np.concatenate([override_arr, pad_tail])
                else:
                    override_arr = override_arr[: self.num_timesteps]
                param.value = override_arr

        if self.optim_conf["set_use_battery"]:
            # #610: per battery k, mirroring the pre-#610 single-battery
            # assignments below exactly (at n_batt==1 this loop runs once with
            # k==0, byte-identical values).
            for k in range(self.n_batt):
                self.param_soc_init[k].value = soc_init_list[k]
                self.param_soc_final[k].value = soc_final_list[k]
                self.param_battery_charge_power_max[k].value = float(
                    batt_conf["charge_power_max"][k]
                )
                self.param_battery_discharge_power_max[k].value = float(
                    batt_conf["discharge_power_max"][k]
                )
                low_gap_wh = max(
                    0.0,
                    (batt_conf["soc_min"][k] - soc_init_list[k]) * batt_conf["cap"][k],
                )
                high_gap_wh = max(
                    0.0,
                    (soc_init_list[k] - batt_conf["soc_max"][k]) * batt_conf["cap"][k],
                )
                self.param_soc_low_gap[k].value = low_gap_wh
                self.param_soc_high_gap[k].value = high_gap_wh
                self.param_soc_low_required[k].value = 1.0 if low_gap_wh > 0 else 0.0
                self.param_soc_high_required[k].value = 1.0 if high_gap_wh > 0 else 0.0

        # Update Window Mask Parameters for Deferrable Loads
        # This allows warm-starting even when time windows change
        n = len(p_pv)
        # Track which loads have a configured-but-empty window so we can
        # also deactivate their binary vars and energy constraints below.
        # An empty window means the user's [start, end] is entirely outside
        # [0, n] — emitting binaries / energy constraints for these loads
        # would make the MILP either infeasible (forcing the relaxed-LP
        # fallback) or unnecessarily large.
        window_empty_loads: set[int] = set()
        for k in range(min(num_deferrable_loads, len(self.param_window_masks))):
            # Calculate validated window bounds
            if def_total_timestep and def_total_timestep[k] > 0:
                def_start, def_end, _ = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_timestep[k]),
                    n,
                )
            else:
                def_start, def_end, _ = Optimization.validate_def_timewindow(
                    def_start_timestep[k],
                    def_end_timestep[k],
                    ceil(def_total_hours[k] / self.time_step) if def_total_hours[k] > 0 else 0,
                    n,
                )

            # Detect user-configured-but-empty window. We distinguish three
            # cases:
            #   (a) User explicitly configured [start, end] entirely outside
            #       [0, n] — e.g. start=600, end=800, n=576. validate clamps
            #       both to n, so def_end == def_start == n. Treat as empty.
            #   (b) User left start and end at the defaults (typically both 0)
            #       — treat as "no window restriction", mask = all-1.
            #   (c) Valid window inside the horizon — mask = 1 inside, 0 outside.
            raw_start = def_start_timestep[k] if k < len(def_start_timestep) else 0
            raw_end = def_end_timestep[k] if k < len(def_end_timestep) else 0
            user_configured_window = (raw_start > 0 or raw_end > 0) and raw_start <= raw_end
            effective_window_size = max(0, min(n, raw_end) - max(0, raw_start))

            # Build the window mask
            window_mask = np.zeros(n)
            if def_end > def_start:
                # case (c): valid window inside horizon
                window_mask[def_start:def_end] = 1.0
            elif user_configured_window and effective_window_size <= 0:
                # case (a): structurally empty window — load can never operate.
                # Mask stays zero, and remember k so the load-active and energy
                # constraints get deactivated too.
                window_empty_loads.add(k)
                self.logger.info(
                    "Deferrable load %d: configured window [%d, %d] is entirely "
                    "outside the optimization horizon [0, %d]; deactivating "
                    "binary vars and energy constraint for this tick.",
                    k,
                    raw_start,
                    raw_end,
                    n,
                )
            else:
                # case (b): no window configured — allow operation everywhere
                window_mask[:] = 1.0

            # Live "window OR door is open right now" pause: forces
            # p_deferrable[k][0] <= 0 for the current step only, since a live
            # sensor reading has no meaning for future timesteps. Refreshed
            # every call (cold build and cache hit alike), since this whole
            # per-load loop runs unconditionally.
            if k < len(room_opening_open) and room_opening_open[k]:
                window_mask[0] = 0.0

            self.param_window_masks[k].value = window_mask

            # Manually-committed sequence loads (see manual_load_enabled /
            # param_sequence_required above): gate must-run on this cycle's
            # operating_hours override, the same "ready or committed" signal
            # _apply_manual_load_runtime_overrides already produces for the
            # flat-load path (0 when idle, >0 otherwise). Non-manual loads
            # keep the default 1.0 (always must-run, unchanged behavior).
            if k < len(self.param_sequence_required):
                is_manual_load_list = self.optim_conf.get("is_manual_load", [])
                is_manual_auto_k = k < len(is_manual_load_list) and bool(is_manual_load_list[k])
                if is_manual_auto_k:
                    needs_run = k < len(def_total_hours) and float(def_total_hours[k] or 0.0) > 0
                    self.param_sequence_required[k].value = 1.0 if needs_run else 0.0
                else:
                    self.param_sequence_required[k].value = 1.0

        # Update Thermal Parameters for warm-starting
        # This updates all thermal parameters (outdoor_temp, heating_demand, COPs, etc.)
        # On first call, these will be set during constraint building
        # On subsequent calls (cache hit), this ensures parameters reflect new forecasts
        if self.prob is not None and self.param_thermal:
            self.update_thermal_params(
                self.optim_conf, data_opt, p_load, room_opening_open=room_opening_open
            )
            # Refresh heating_demands for result building (stale numpy refs from first call)
            for k, params in self.param_thermal.items():
                if params["type"] == "thermal_battery":
                    self.heating_demands[k] = params["heating_demand"].value

        # Update def_current_state parameters before the per-load loop so that
        # param_def_current_state[k].value is current when the pinning block reads it.
        self._update_def_current_state_params(num_deferrable_loads)
        # Update def_current_on_timesteps so the min-on remainder block (issue #952)
        # has the correct elapsed on-time when it runs in the per-load loop below.
        self._update_def_current_on_timesteps_params(num_deferrable_loads)
        # Update def_current_off_timesteps so the min-off remainder block (#952 follow-on)
        # has the correct elapsed off-time when it runs in the per-load loop below.
        self._update_def_current_off_timesteps_params(num_deferrable_loads)
        # Update def_current_power (issue #605): runs AFTER _update_def_current_state_params
        # so it can bump param_def_current_state to suppress the phantom t=0 startup.
        self._update_def_current_power_params(num_deferrable_loads)
        # Update def_current_operating_timesteps (issue #983): stores the elapsed completed
        # operating timesteps so the per-load loop below can decrement required_timesteps
        # and target_energy accordingly.
        self._update_def_current_operating_timesteps_params(num_deferrable_loads)

        # Shared-tank members are temperature-driven; used below to exempt them
        # from the operating-timestep deactivation in the param_load_active loop.
        shared_tank_membership = self._load_shared_tank_membership()

        # Loads whose must-run requirement is fully satisfied by the COTS decrement
        # (issue #983): elapsed completed timesteps >= required, so remaining clamps
        # to 0. Such a load MUST be released (treated as a load with no operating
        # requirement) -- otherwise the param_load_active loop keeps it active
        # (has_operating_requirement is True from def_total_hours/timestep) and the
        # single-constant startup constraint forces a phantom extra block. Populated
        # in the decrement branch below; consumed by the param_load_active loop.
        cots_satisfied_loads = set()

        # Update Energy Constraint Parameters for Deferrable Loads
        # These control the Big-M relaxation of energy/timestep constraints
        for k in range(min(num_deferrable_loads, len(self.param_target_energy))):
            # Get nominal power
            nominal_power = self.optim_conf["nominal_power_of_deferrable_loads"][k]
            if isinstance(nominal_power, list):
                nominal_power = max(nominal_power)

            # Dispatch mode per load: hours, program, or energy_kwh
            dispatch_modes = self.optim_conf.get("load_dispatch_mode", ["hours"] * num_deferrable_loads)
            dispatch_mode = (
                dispatch_modes[k] if k < len(dispatch_modes) else "hours"
            )

            # Program-based sequence loads are handled by dedicated sequence constraints.
            if isinstance(self.optim_conf["nominal_power_of_deferrable_loads"][k], list):
                dispatch_mode = "program"

            # Determine operating requirement: def_total_timestep takes priority over def_total_hours
            # def_total_timestep is specified in number of timesteps
            # def_total_hours is specified in hours
            if dispatch_mode == "program":
                required_timesteps = 0
                target_energy = 0.0
                constraint_active = False
            elif dispatch_mode == "energy_kwh":
                required_energy_kwh = self.optim_conf.get(
                    "required_energy_kwh_of_each_deferrable_load", [0.0] * num_deferrable_loads
                )
                required_kwh = (
                    required_energy_kwh[k] if k < len(required_energy_kwh) else 0.0
                )
                if required_kwh > 0:
                    target_energy = float(required_kwh) * 1000.0
                    required_timesteps = (
                        ceil(target_energy / (nominal_power * self.time_step))
                        if nominal_power > 0
                        else 0
                    )
                    constraint_active = True
                else:
                    required_timesteps = 0
                    target_energy = 0.0
                    constraint_active = False
            elif def_total_timestep and k < len(def_total_timestep) and def_total_timestep[k] > 0:
                # Use timestep-based specification
                required_timesteps = ceil(def_total_timestep[k])
                # Convert to energy: power * timesteps * time_step (time_step is in hours)
                target_energy = nominal_power * required_timesteps * self.time_step
                constraint_active = True
            elif def_total_hours and k < len(def_total_hours) and def_total_hours[k] > 0:
                # Use hours-based specification
                operating_hours = def_total_hours[k]
                required_timesteps = ceil(operating_hours / self.time_step)
                target_energy = nominal_power * operating_hours
                constraint_active = True
            else:
                # No constraint specified
                required_timesteps = 0
                target_energy = 0.0
                constraint_active = False

            # Apply completed-operating-timesteps decrement (issue #983).
            # If the caller signals that the load has already run for some timesteps
            # today, reduce the remaining required run and energy proportionally.
            # Clamped at 0 so an elapsed >= required never produces negative values
            # or an infeasible model.  Applies to both standard and single_constant
            # must-run loads (no is_single_const gate -- that is the whole point of
            # this param vs def_current_on_timesteps which is gated not is_single_const).
            if (
                k < len(self.param_current_operating_timesteps)
                and self.param_current_operating_timesteps[k].value > 0
                and constraint_active
            ):
                elapsed_steps = int(self.param_current_operating_timesteps[k].value)
                required_timesteps = max(0, required_timesteps - elapsed_steps)
                # target_energy units: W * h  (nominal_power in W, time_step in h)
                target_energy = max(
                    0.0, target_energy - elapsed_steps * nominal_power * self.time_step
                )
                # If the decrement reduces required to 0, fully relax the constraint
                # so the load is not forced to run.
                if required_timesteps == 0:
                    constraint_active = False
                    # The must-run requirement is now MET. Record the load so the
                    # param_load_active loop deactivates it (param_load_active=0),
                    # the same as a load with no operating requirement -- otherwise
                    # the single-constant startup constraint
                    # (sum(p_def_start) == param_load_active - already_running)
                    # would still force a phantom startup block. This branch only
                    # runs when constraint_active was True, which already excludes
                    # thermal / shared-tank / sequence loads (their energy/timestep
                    # constraints are skipped), but the param_load_active loop guards
                    # those cases again for safety.
                    cots_satisfied_loads.add(k)
                    # CRITICAL INTERACTION with def_current_power (issue #982/#605):
                    # a currently-running load may also have def_current_power[k] > 0,
                    # which (for pin-eligible loads) PINS p_deferrable[k][0] to that
                    # wattage and force-ON's bin2[k][0] via _def_current_power_affected.
                    # With param_load_active=0 the load is bounded to 0 W for the whole
                    # horizon (p_def_bin2 <= 0 and p_deferrable <= M*0), so the t=0
                    # force-on / pin would conflict and make the model INFEASIBLE.
                    # A target-MET load must be allowed to turn off, so release the
                    # current-power pin and force-on for it (the t0 pin is treated as
                    # released for COTS-satisfied loads). Single_const loads are never
                    # affected by def_current_power anyway (excluded in
                    # _update_def_current_power_params), so this is a no-op for them and
                    # only matters for a standard (semi_cont) COTS-satisfied load.
                    if k < len(self._def_current_power_affected):
                        self._def_current_power_affected[k] = False
                    if k < len(self.param_def_current_power_active):
                        self.param_def_current_power_active[k].value = 0.0

            # Set energy constraint parameters. Force-relax the constraint if
            # the load's configured window is entirely outside the horizon
            # (window_empty_loads, populated above) — the load can never run,
            # so emitting a target-energy constraint would force infeasibility
            # and trigger the relaxed-LP fallback.
            if constraint_active and k not in window_empty_loads:
                self.param_target_energy[k].value = target_energy
                self.param_energy_active[k].value = 1.0  # Constraint is active
            else:
                self.param_target_energy[k].value = 0.0
                self.param_energy_active[k].value = 0.0  # Constraint is relaxed (Big-M)

            # For single-constant (binary) loads, set the required timesteps
            is_single_const = self.optim_conf["set_deferrable_load_single_constant"][k]
            if is_single_const and constraint_active and k not in window_empty_loads:
                self.param_required_timesteps[k].value = required_timesteps
                self.param_timesteps_active[k].value = 1.0  # Constraint is active
            else:
                self.param_required_timesteps[k].value = 0.0
                self.param_timesteps_active[k].value = 0.0  # Constraint is relaxed (Big-M)

            # Build param_running_lb mask for this load.
            # Two independent mechanisms can both write to param_running_lb[k]:
            #   A) Single-constant pin: force ON for the remaining required_timesteps
            #      when a single-constant load is currently running.
            #   B) Min-on-time remainder: for any semi-continuous (or min-power)
            #      load that is currently ON, force ON for max(0, N - elapsed) steps
            #      to honour the tail of an in-progress min-on window (issue #952).
            # When both apply to the same k, take the ELEMENTWISE MAX (OR) of the two
            # masks -- the stricter force wins, and neither overwrites the other.
            if k < len(self.param_running_lb):
                current_state = (
                    self.param_def_current_state[k].value > 0.5
                    if k < len(self.param_def_current_state)
                    else False
                )

                # --- A) Single-constant pin ---
                single_const_lb = np.zeros(n)
                if (
                    is_single_const
                    and current_state
                    and constraint_active
                    and required_timesteps > 0
                    and k not in window_empty_loads
                ):
                    # Re-derive the configured window end so we respect def_end_timestep.
                    if def_total_timestep and def_total_timestep[k] > 0:
                        _, cfg_end, _ = Optimization.validate_def_timewindow(
                            def_start_timestep[k],
                            def_end_timestep[k],
                            ceil(def_total_timestep[k]),
                            n,
                        )
                    else:
                        _, cfg_end, _ = Optimization.validate_def_timewindow(
                            def_start_timestep[k],
                            def_end_timestep[k],
                            ceil(def_total_hours[k] / self.time_step)
                            if def_total_hours[k] > 0
                            else 0,
                            n,
                        )
                    # cfg_end == 0 means no window restriction -> treat as full horizon.
                    effective_end = cfg_end if cfg_end > 0 else n
                    pinned_steps = min(required_timesteps, effective_end, n)

                    single_const_lb[:pinned_steps] = 1.0
                    self.param_already_running_sc[k].value = 1.0

                    # Widen the window mask so the forced-on period is never blocked.
                    if k < len(self.param_window_masks):
                        wm = self.param_window_masks[k].value.copy()
                        wm[:pinned_steps] = 1.0
                        self.param_window_masks[k].value = wm

                    self.logger.debug(
                        "Deferrable load %d: single-const running, pinning %d timesteps ON "
                        "(requested %d, window end %d, horizon %d)",
                        k,
                        pinned_steps,
                        required_timesteps,
                        effective_end,
                        n,
                    )
                else:
                    self.param_already_running_sc[k].value = 0.0

                # --- B) Min-on-time remainder (issue #952) ---
                # Applies when: load is currently ON, N > 0, AND elapsed on-time
                # (def_current_on_timesteps[k]) is supplied. Absent elapsed -> no force
                # (NOT assumed-zero; document this clearly).
                min_on_lb = np.zeros(n)
                def_min_on = self.optim_conf.get("def_minimum_on_time", [])
                min_on_n = (
                    self._coerce_nonneg_timesteps(def_min_on[k], k, "def_minimum_on_time")
                    if k < len(def_min_on)
                    else 0
                )
                # Only fire when load is ON, N > 0, NOT single-constant (those use
                # their own currently-running pin), and elapsed is explicitly supplied.
                # COTS-satisfied loads (issue #983) are RELEASED from every force-on
                # mechanism: the param_load_active loop deactivates them
                # (param_load_active=0 => bin2<=0), so a min-on force-on here
                # (param_running_lb => bin2>=1) would conflict and make the MILP
                # INFEASIBLE (rescued only by the global relaxed-LP fallback, which
                # degrades the whole solve). A target-MET load must be free to turn
                # off, so skip Block B for it (mirrors Block A's required_timesteps>0
                # gate and the def_current_power release above).
                if (
                    current_state
                    and min_on_n > 0
                    and not is_single_const
                    and k not in cots_satisfied_loads
                    and "def_current_on_timesteps" in self.optim_conf
                    and k < len(self.optim_conf["def_current_on_timesteps"])
                ):
                    # Use the validated Parameter value (set by
                    # _update_def_current_on_timesteps_params) rather than re-reading
                    # the raw optim_conf entry in the solve loop.
                    elapsed = int(self.param_current_on_timesteps[k].value)
                    remaining = max(0, min_on_n - elapsed)
                    if remaining > 0:
                        # Clamp to horizon and to the load's operating-window end.
                        # Re-derive effective_end using the same logic as the
                        # single-constant pin above (reuses validate_def_timewindow).
                        if (
                            def_total_timestep
                            and k < len(def_total_timestep)
                            and def_total_timestep[k] > 0
                        ):
                            _, cfg_end_mot, _ = Optimization.validate_def_timewindow(
                                def_start_timestep[k]
                                if def_start_timestep and k < len(def_start_timestep)
                                else 0,
                                def_end_timestep[k]
                                if def_end_timestep and k < len(def_end_timestep)
                                else 0,
                                ceil(def_total_timestep[k]),
                                n,
                            )
                        elif (
                            def_total_hours and k < len(def_total_hours) and def_total_hours[k] > 0
                        ):
                            _, cfg_end_mot, _ = Optimization.validate_def_timewindow(
                                def_start_timestep[k]
                                if def_start_timestep and k < len(def_start_timestep)
                                else 0,
                                def_end_timestep[k]
                                if def_end_timestep and k < len(def_end_timestep)
                                else 0,
                                ceil(def_total_hours[k] / self.time_step),
                                n,
                            )
                        else:
                            cfg_end_mot = 0
                        effective_end_mot = cfg_end_mot if cfg_end_mot > 0 else n
                        pinned_mot = min(remaining, effective_end_mot, n)
                        min_on_lb[:pinned_mot] = 1.0

                        # Widen the window mask so forced-on steps are never blocked.
                        if pinned_mot > 0 and k < len(self.param_window_masks):
                            wm_mot = self.param_window_masks[k].value.copy()
                            wm_mot[:pinned_mot] = 1.0
                            self.param_window_masks[k].value = wm_mot

                        self.logger.debug(
                            "Deferrable load %d: min-on remainder, elapsed=%d N=%d "
                            "remaining=%d -> pinning %d timesteps ON (horizon %d, window_end %d)",
                            k,
                            elapsed,
                            min_on_n,
                            remaining,
                            pinned_mot,
                            n,
                            effective_end_mot,
                        )

                # Current-power force-on (issue #605): when load k is affected by
                # def_current_power, force bin2[k][0] = 1 so the load stays ON at
                # t=0. Only index 0 matters here; for pinned loads the power-pin
                # constraint already implies bin2[0]=1, so this is what keeps an affected
                # semi_cont load ON (at nominal). Excludes single_const / sequence /
                # thermal via _def_current_power_affected (set in the update method).
                # Widen window_mask[0] too so a load whose window starts after t=0 is
                # not immediately blocked by the mask (mirrors the single-const / min-on
                # widen pattern above).
                current_power_lb = np.zeros(n)
                if (
                    k < len(self._def_current_power_affected)
                    and self._def_current_power_affected[k]
                ):
                    current_power_lb[0] = 1.0
                    # Widen window mask at t=0 so the forced-on step is never blocked.
                    if k < len(self.param_window_masks):
                        wm_cp = self.param_window_masks[k].value.copy()
                        if wm_cp[0] < 1.0:
                            wm_cp[0] = 1.0
                            self.param_window_masks[k].value = wm_cp

                # ELEMENTWISE MAX: combine single-const pin, min-on remainder, and
                # current-power force-on.  Neither mechanism overwrites the other;
                # the stricter force wins.
                combined_lb = np.maximum(np.maximum(single_const_lb, min_on_lb), current_power_lb)
                self.param_running_lb[k].value = combined_lb

                # --- C) Min-off-time remainder (#952 follow-on) ---
                # Applies when: load is currently OFF, N > 0, NOT single-constant,
                # NOT sequence, and def_current_off_timesteps[k] is supplied.
                # Absent elapsed -> no force (NOT assumed-zero; same pattern as min-on).
                #
                # Force-off is via param_running_ub[k]: set forced-off entries to 0.0.
                # Default is all-1.0 (no-op). Reset to 1.0 each solve so a load that
                # was forced off last tick is free again once the window expires.
                if k < len(self.param_running_ub):
                    self.param_running_ub[k].value = np.ones(n)

                # Determine if this load is a sequence load (list-valued nominal power).
                _nom_pwr = self.optim_conf["nominal_power_of_deferrable_loads"]
                is_sequence_load_rem = k < len(_nom_pwr) and isinstance(_nom_pwr[k], list)

                def_min_off = self.optim_conf.get("def_minimum_off_time", [])
                min_off_n = (
                    self._coerce_nonneg_timesteps(def_min_off[k], k, "def_minimum_off_time")
                    if k < len(def_min_off)
                    else 0
                )
                # Only fire when load is OFF, N > 0, NOT single-constant, NOT sequence,
                # and elapsed is explicitly supplied.
                if (
                    not current_state
                    and min_off_n > 0
                    and not is_single_const
                    and not is_sequence_load_rem
                    and "def_current_off_timesteps" in self.optim_conf
                    and k < len(self.optim_conf["def_current_off_timesteps"])
                ):
                    elapsed_off = int(self.param_current_off_timesteps[k].value)
                    remaining_off = max(0, min_off_n - elapsed_off)
                    if remaining_off > 0:
                        pinned_off = min(remaining_off, n)
                        if k < len(self.param_running_ub):
                            ub_val = self.param_running_ub[k].value.copy()
                            ub_val[:pinned_off] = 0.0
                            self.param_running_ub[k].value = ub_val

                        self.logger.debug(
                            "Deferrable load %d: min-off remainder, elapsed=%d N=%d "
                            "remaining=%d -> forcing %d timesteps OFF (horizon %d)",
                            k,
                            elapsed_off,
                            min_off_n,
                            remaining_off,
                            pinned_off,
                            n,
                        )

        # Update load active parameters: deactivate non-thermal loads with 0 operating timesteps,
        # OR with a configured window that's entirely outside the optimization horizon.
        # Thermal loads (thermal_config, thermal_battery, and shared-tank sources) are
        # always active since they're driven by temperature constraints, not operating
        # timesteps. Shared-tank members already skip the energy/operating constraints
        # above (is_thermal_battery), so they must not be deactivated here either —
        # otherwise a member with operating_hours == 0 (the natural setting for a
        # temperature-driven source) is pinned to 0 W, the tank cannot hold its
        # min_temperatures band, and the problem goes infeasible. Sequence loads
        # (list-valued nominal power) are likewise always active: their runtime is the
        # length of the sequence and operating_hours is meaningless for them, so a value
        # of 0 must not deactivate the load (issue #887). The energy constraint already
        # exempts sequence loads, so this keeps param_load_active consistent with it.
        nominal_powers = self.optim_conf["nominal_power_of_deferrable_loads"]
        for k in range(min(num_deferrable_loads, len(self.param_load_active))):
            is_thermal = k in self.param_thermal or k in shared_tank_membership
            is_sequence = k < len(nominal_powers) and isinstance(nominal_powers[k], list)
            has_operating_requirement = (
                def_total_timestep and k < len(def_total_timestep) and def_total_timestep[k] > 0
            ) or (def_total_hours and k < len(def_total_hours) and def_total_hours[k] > 0)
            window_outside_horizon = k in window_empty_loads
            if is_thermal:
                # Thermal loads are still driven by temperature constraints
                # even if their configured window is outside the horizon.
                self.param_load_active[k].value = 1.0
                # Shared-tank members must also keep an open window mask: a
                # configured window outside the horizon zeroes the mask, which
                # would pin every member to 0 W and make the tank's
                # min_temperatures unreachable (infeasible, then the relaxed
                # fallback fails too and nothing is published).
                if k in shared_tank_membership and window_outside_horizon:
                    if k < len(self.param_window_masks):
                        self.param_window_masks[k].value = np.ones(n)
                    self.logger.warning(
                        "Deferrable load %d is a shared-tank source with a configured "
                        "window outside the horizon; ignoring the window (temperature "
                        "constraints drive this load).",
                        k,
                    )
            elif k in cots_satisfied_loads and not is_sequence:
                # COTS decrement fully satisfied this must-run load (issue #983):
                # remaining required clamped to 0. Deactivate it exactly like a load
                # with no operating requirement so the single-constant startup
                # constraint does not force a phantom block. is_thermal is already
                # handled above; guard is_sequence here too (sequence loads never
                # enter the decrement branch -- their energy constraint is exempt --
                # so cots_satisfied_loads can't contain one, but stay defensive).
                self.param_load_active[k].value = 0.0
                self.logger.debug(
                    f"Deferrable load {k}: deactivated (operating requirement met "
                    "by def_current_operating_timesteps, issue #983)"
                )
            elif (has_operating_requirement or is_sequence) and not window_outside_horizon:
                self.param_load_active[k].value = 1.0
            else:
                self.param_load_active[k].value = 0.0
                if window_outside_horizon:
                    self.logger.debug(
                        f"Deferrable load {k}: deactivated (configured window outside horizon)"
                    )
                else:
                    self.logger.debug(
                        f"Deferrable load {k}: deactivated (no operating timesteps, not thermal)"
                    )

        # Initialize stress config variables (needed by retry path even when
        # self.prob is cached from a previous call, see #770)
        inv_stress_conf = None
        batt_stress_conf = None

        # Build Problem (Lazy Construction)
        if self.prob is None:
            self.logger.info("Building CVXPY problem structure...")

            # Start with bound constraints
            constraints = self.constraints[:]

            if self.optim_conf["set_use_battery"]:
                # #610: one stress config per battery (raw plant_conf read via
                # _battery_conf_as_lists: stress cost is gated per-battery on
                # battery_stress_cost[k] > 0, off by default, and the value is
                # only used to size PWL segments at build time, so runtime
                # parameterisation isn't needed here). batt_stress_conf becomes
                # a list aligned with the battery lists; self.vars["batt_stress_cost"]
                # is only set at all if at least one battery has it active
                # (matches the pre-#610 "key absent when inactive" behavior).
                batt_conf_for_stress = self._battery_conf_as_lists()
                batt_stress_conf = []
                for k in range(self.n_batt):
                    p_batt_max_k = max(
                        batt_conf_for_stress["discharge_power_max"][k],
                        batt_conf_for_stress["charge_power_max"][k],
                    )
                    batt_stress_conf.append(
                        self._setup_battery_stress_cost(
                            k, batt_conf_for_stress["stress_cost"][k], p_batt_max_k
                        )
                    )
                if any(c["active"] for c in batt_stress_conf):
                    self.vars["batt_stress_cost"] = [c["vars"] for c in batt_stress_conf]

            if self.plant_conf["inverter_is_hybrid"]:
                P_nom_inverter_max = max(
                    self.plant_conf.get("inverter_ac_output_max", 0),
                    self.plant_conf.get("inverter_ac_input_max", 0),
                )
                inv_stress_conf = self._setup_stress_cost(
                    "inverter_stress_cost", P_nom_inverter_max, "inv"
                )
                if inv_stress_conf["active"]:
                    self.vars["inv_stress_cost"] = inv_stress_conf["vars"]

            # Add Constraints
            self._add_main_power_balance_constraints(constraints)
            self._add_hybrid_inverter_constraints(constraints, inv_stress_conf)
            self._add_battery_constraints(constraints, batt_stress_conf)
            self._add_phase_balance_constraints(constraints)

            if self.plant_conf["compute_curtailment"]:
                constraints.append(self.vars["p_pv_curtailment"] <= self.param_pv_forecast)

            if self.costfun == "self-consumption" and "SC" in self.vars:
                constraints.append(self.vars["SC"] <= self.param_pv_forecast)
                constraints.append(
                    self.vars["SC"] <= self.param_load_forecast + self.vars["p_def_sum"]
                )

            # Deferrable Loads
            (
                self.predicted_temps, self.heating_demands, penalty_terms_total,
                self.q_inputs, self.supply_temp_targets,
            ) = (
                self._add_deferrable_load_constraints(
                    constraints,
                    data_opt,
                    def_total_hours,
                    def_total_timestep,
                    def_start_timestep,
                    def_end_timestep,
                    def_init_temp,
                    min_power_of_deferrable_loads,
                    p_load,
                    room_blind_positions=room_blind_positions,
                    room_opening_open=room_opening_open,
                    room_door_open=room_door_open,
                )
            )

            # Deferrable Load Group Constraints (shared power budget, mutual exclusion)
            self._add_deferrable_group_constraints(constraints)

            # Build Objective
            objective_expr = self._build_objective_function(
                batt_stress_conf,
                inv_stress_conf,
            )

            # Add penalty term if it exists (not 0)
            if not isinstance(penalty_terms_total, int) or penalty_terms_total != 0:
                objective_expr.args[0] += penalty_terms_total

            self.prob = cp.Problem(objective_expr, constraints)

        # Solver Configuration
        solver_opts = {"verbose": False}
        if debug:
            solver_opts["verbose"] = True

        # Retrieve Constraints (Time & Threads)
        threads = self.optim_conf.get("num_threads", 0)
        timeout = self.optim_conf.get("lp_solver_timeout", 180)

        # Select Solver
        # We strictly default to HiGHS.
        requested_solver = os.environ.get("LP_SOLVER", "HIGHS").upper()
        selected_solver = cp.HIGHS

        if requested_solver == "GUROBI":
            if "GUROBI" in cp.installed_solvers():
                selected_solver = cp.GUROBI
                solver_opts["TimeLimit"] = timeout
                if threads > 0:
                    solver_opts["Threads"] = threads
            else:
                self.logger.warning(
                    "Solver 'GUROBI' requested via Env Var but not found. Falling back to HiGHS."
                )

        elif requested_solver == "CPLEX":
            if "CPLEX" in cp.installed_solvers():
                selected_solver = cp.CPLEX
                cplex_params = {"timelimit": timeout}
                if threads > 0:
                    cplex_params["threads"] = threads
                solver_opts["cplex_params"] = cplex_params
            else:
                self.logger.warning(
                    "Solver 'CPLEX' requested via Env Var but not found. Falling back to HiGHS."
                )

        # Configure HiGHS (The Default)
        if selected_solver == cp.HIGHS:
            solver_opts["time_limit"] = float(timeout)
            if threads > 0:
                solver_opts["threads"] = int(threads)
            # 'run_crossover' ensures a cleaner solution (closer to simplex vertex)
            solver_opts["run_crossover"] = "on"
            # MIP gap tolerance: allows solver to stop when within X% of optimal.
            # The shipped default is 0.01 (1%), set in config_defaults.json /
            # param_definitions.json, which keeps deep-horizon MILPs from timing
            # out before any plan is published (see issue #986). The 0.0 fallback
            # below only applies when the key is absent entirely from a hand-built
            # optim_conf that bypassed the config system; exact optimal is the safe
            # choice there. Set lp_solver_mip_rel_gap: 0 to opt back in to exact
            # optimal. Higher values solve faster still: benchmarks show 5% gap
            # ~1.75x, 10% ~1.86x, 20% ~2.89x speedup.
            mip_gap = self.optim_conf.get("lp_solver_mip_rel_gap", 0.0)
            # Validate MIP gap is within sensible bounds [0, 1]
            if mip_gap < 0:
                self.logger.warning(
                    f"lp_solver_mip_rel_gap={mip_gap} is negative, using 0 (exact optimal)"
                )
                mip_gap = 0.0
            elif mip_gap > 1:
                self.logger.warning(
                    f"lp_solver_mip_rel_gap={mip_gap} exceeds 1.0 (100%), clamping to 1.0"
                )
                mip_gap = 1.0
            if mip_gap > 0:
                solver_opts["mip_rel_gap"] = float(mip_gap)
                self.logger.debug(f"MIP gap tolerance set to {mip_gap:.1%}")
            else:
                self.logger.debug("MIP gap tolerance disabled (exact optimal)")

        # Stage-timer breadcrumb: end of build phase, start of solve phase.
        _solve_start_perf = time.perf_counter() if stage_times is not None else 0.0
        if stage_times is not None:
            stage_times["optim_solve.build"] = _solve_start_perf - _build_start_perf

        # Solve Execution with Fallback
        try:
            self.prob.solve(solver=selected_solver, warm_start=True, **solver_opts)
        except Exception as e:
            self.logger.warning(
                f"Solver {selected_solver} failed: {e}. Checking status for fallback..."
            )

        # Check for failure or "bad" status
        # Note: "user_limit" often means timeout. "infeasible" means configuration conflict.
        fail_statuses = ["infeasible", "unbounded", "user_limit", None]
        if self.prob.status in fail_statuses or self.prob.value is None:
            self.logger.warning(
                f"Optimization failed with status: '{self.prob.status}'. "
                "Retrying with relaxed constraints (Continuous LP)..."
            )

            # Backup Configuration
            original_semi_cont = copy.deepcopy(
                self.optim_conf.get("treat_deferrable_load_as_semi_cont", [])
            )
            original_single_const = copy.deepcopy(
                self.optim_conf.get("set_deferrable_load_single_constant", [])
            )

            # Relax Configuration: Disable Binary Logic
            n_def = self.optim_conf["number_of_deferrable_loads"]
            self.optim_conf["treat_deferrable_load_as_semi_cont"] = [False] * n_def
            self.optim_conf["set_deferrable_load_single_constant"] = [False] * n_def

            # Re-build Constraints (Clean Slate)
            constraints_relaxed = self.constraints[:]  # Start with base bound constraints

            # Re-apply main constraints
            self._add_main_power_balance_constraints(constraints_relaxed)
            # (Note: We reuse previous stress configs as they don't change with relaxation)
            # Guard on feature flags, not on the stress conf objects: stress conf is None
            # on cached-problem retry paths (self.prob is not None), so guarding on the
            # object itself would silently skip these constraints on every retry after the
            # first call, leaving the relaxed problem without battery/inverter constraints.
            if self.plant_conf.get("inverter_is_hybrid", False):
                self._add_hybrid_inverter_constraints(constraints_relaxed, inv_stress_conf)
            if self.optim_conf.get("set_use_battery", False):
                self._add_battery_constraints(constraints_relaxed, batt_stress_conf)
            # Kept hard even under relaxation - silently allowing a fuse
            # overload just to "find a solution" defeats the point of this
            # feature; a genuinely infeasible per-phase configuration should
            # surface as an infeasibility on both the MILP and relaxed-LP
            # attempts, not a silently-dangerous plan.
            self._add_phase_balance_constraints(constraints_relaxed)

            if self.plant_conf["compute_curtailment"]:
                constraints_relaxed.append(self.vars["p_pv_curtailment"] <= self.param_pv_forecast)
            if self.costfun == "self-consumption" and "SC" in self.vars:
                constraints_relaxed.append(self.vars["SC"] <= self.param_pv_forecast)
                constraints_relaxed.append(
                    self.vars["SC"] <= self.param_load_forecast + self.vars["p_def_sum"]
                )

            # Re-call deferrable load constraints (Skipping binary logic due to config change).
            # Also turn every thermal load's own hard min/max comfort bounds
            # elastic for this retry only (see
            # _add_thermal_battery_bounds_and_penalty's own comment) - the
            # same "don't just give up on infeasible, offer a softened
            # fallback" philosophy this whole retry block already applies to
            # every other load's binary logic, extended to comfort bounds so
            # a genuinely-unavoidable violation still yields a usable plan.
            self._soft_comfort_bounds_pass = True
            try:
                (
                    self.predicted_temps, self.heating_demands, penalty_terms_total,
                    self.q_inputs, self.supply_temp_targets,
                ) = (
                    self._add_deferrable_load_constraints(
                        constraints_relaxed,
                        data_opt,
                        def_total_hours,
                        def_total_timestep,
                        def_start_timestep,
                        def_end_timestep,
                        def_init_temp,
                        min_power_of_deferrable_loads,
                        p_load,
                        room_blind_positions=room_blind_positions,
                        room_opening_open=room_opening_open,
                        room_door_open=room_door_open,
                    )
                )
            finally:
                self._soft_comfort_bounds_pass = False

            # Deferrable Load Group Constraints (shared power budget only in relaxed mode)
            self._add_deferrable_group_constraints(constraints_relaxed, relaxed=True)

            # Re-build Objective
            objective_expr = self._build_objective_function(batt_stress_conf, inv_stress_conf)
            if not isinstance(penalty_terms_total, int) or penalty_terms_total != 0:
                objective_expr.args[0] += penalty_terms_total

            # Solve Relaxed Problem
            prob_relaxed = cp.Problem(objective_expr, constraints_relaxed)
            try:
                self.logger.info("Solving relaxed problem (LP)...")
                prob_relaxed.solve(solver=selected_solver, **solver_opts)

                if prob_relaxed.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    self.logger.info("Relaxed optimization successful!")
                    self.prob = prob_relaxed  # Use this result
                    # Mark status so user knows it was relaxed
                    self.prob._status = "Optimal (Relaxed)"
                    # "(Relaxed)" alone only ever meant "some other load's
                    # own binary logic was loosened" - surface it loudly and
                    # specifically when it's actually the comfort_slack_*
                    # elastic bounds above that were needed, since that
                    # means a real comfort target was NOT met this solve,
                    # not just that some scheduling flexibility was used.
                    max_violation_c = 0.0
                    violated_loads: set[str] = set()
                    for var in prob_relaxed.variables():
                        name = var.name()
                        if not (name.startswith("comfort_slack_min_") or name.startswith("comfort_slack_max_")):
                            continue
                        val = var.value
                        if val is None:
                            continue
                        peak = float(np.max(val)) if np.size(val) else 0.0
                        if peak > 1e-3:
                            max_violation_c = max(max_violation_c, peak)
                            violated_loads.add(name.rsplit("_", 1)[-1])
                    if violated_loads:
                        self.logger.warning(
                            "Comfort temperature bound(s) could not be fully met this solve "
                            "for load(s) %s - pushed up to %.2f°C past the configured limit "
                            "(the plan is otherwise as close to comfort as physically possible, "
                            "not silently ignoring it).",
                            sorted(violated_loads),
                            max_violation_c,
                        )
                else:
                    self.logger.error(
                        f"Relaxed optimization also failed with status: {prob_relaxed.status}"
                    )
                    self.prob = prob_relaxed
            except Exception as e:
                self.logger.error(f"Relaxed optimization crashed: {e}")

            # 5. Restore Configuration
            self.optim_conf["treat_deferrable_load_as_semi_cont"] = original_semi_cont
            self.optim_conf["set_deferrable_load_single_constant"] = original_single_const

        # Stage-timer breadcrumb: end of solve phase, start of extract phase.
        _extract_start_perf = time.perf_counter() if stage_times is not None else 0.0
        if stage_times is not None:
            stage_times["optim_solve.solve"] = _extract_start_perf - _solve_start_perf

        # Fix for Status Case: Map "optimal" -> "Optimal"
        status_raw = self.prob.status
        self.optim_status = status_raw.title() if status_raw else "Failure"

        # Helper: Ensure we return "Optimal" for tests if it was "Optimal (Relaxed)" or "Optimal_Inaccurate"
        if self.prob.value is None or self.prob.status not in [
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
            "Optimal (Relaxed)",
        ]:
            self.logger.warning("Cost function cannot be evaluated or Infeasible/Unbounded")

            # Create a DataFrame with the correct index (timestamps)
            opt_tp = pd.DataFrame(index=data_opt.index)

            # explicitely set the status column so downstream functions (like get_injection_dict)
            # don't crash when trying to access or drop it.
            opt_tp["optim_status"] = self.optim_status

            if stage_times is not None:
                stage_times["optim_solve.extract"] = time.perf_counter() - _extract_start_perf
            return opt_tp
        else:
            self.logger.info(
                "Total value of the Cost function = %.02f",
                self.prob.value,
            )

        # Results Extraction
        # #610: pass the per-battery soc_init list built above, except at
        # n_batt==1 where the bare scalar is passed (byte-identical to today's
        # single-battery call, and _build_results_dataframe's own N==1 branch
        # expects a scalar here, not a 1-element list).
        soc_init_for_results = (
            soc_init_list[0] if soc_init_list is not None and self.n_batt == 1 else soc_init_list
        )
        results_df = self._build_results_dataframe(
            data_opt,
            unit_load_cost,
            unit_prod_price,
            p_load,
            p_pv,
            soc_init_for_results,
            self.predicted_temps,
            self.heating_demands,
            debug,
            q_inputs=self.q_inputs,
            supply_temp_targets=self.supply_temp_targets,
        )
        if stage_times is not None:
            stage_times["optim_solve.extract"] = time.perf_counter() - _extract_start_perf
        return results_df

    def perform_perfect_forecast_optim(
        self, df_input_data: pd.DataFrame, days_list: pd.date_range
    ) -> pd.DataFrame:
        r"""
        Perform an optimization on historical data (perfectly known PV production).

        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably photovoltaics and load consumption powers.
        :type df_input_data: pandas.DataFrame
        :param days_list: A list of the days of data that will be retrieved from \
            hass and used for the optimization task. We will retrieve data from \
            now and up to days_to_retrieve days
        :type days_list: list
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform optimization for perfect forecast scenario")
        self.days_list_tz = days_list.tz_convert(self.time_zone).round(self.freq)[
            :-1
        ]  # Converted to tz and without the current day (today)
        self.opt_res = pd.DataFrame()

        # List to collect results for faster one-time concatenation
        results_list = []

        for day in self.days_list_tz:
            self.logger.info(
                "Solving for day: " + str(day.day) + "-" + str(day.month) + "-" + str(day.year)
            )
            # Prepare data
            if day.tzinfo is None:
                day = day.replace(tzinfo=self.time_zone)  # Assign timezone if naive
            else:
                day = day.astimezone(self.time_zone)
            day_start = day
            day_end = day + self.time_delta - self.freq
            if day_start.tzinfo != day_end.tzinfo:
                self.logger.warning(
                    f"Skipping day {day} as days have different timezone, probably because of DST."
                )
                continue  # Skip this day and move to the next iteration
            else:
                day_start = day_start.astimezone(self.time_zone).isoformat()
                day_end = day_end.astimezone(self.time_zone).isoformat()
                # Generate the date range for the current day
                day_range = pd.date_range(start=day_start, end=day_end, freq=self.freq)
            # Check if all timestamps in the range exist in the DataFrame index
            if not day_range.isin(df_input_data.index).all():
                self.logger.warning(
                    f"Skipping day {day} as some timestamps are missing in the data."
                )
                continue  # Skip this day and move to the next iteration

            # If all timestamps exist, proceed with the data preparation
            data_tp = df_input_data.copy().loc[day_range]
            p_pv = data_tp[self.var_pv].values
            p_load = data_tp[self.var_load_new].values
            unit_load_cost = data_tp[self.var_load_cost].values  # currency/kWh
            unit_prod_price = data_tp[self.var_prod_price].values  # currency/kWh

            # Call optimization function
            # The new CVXPY implementation will re-use the problem structure automatically
            opt_tp = self.perform_optimization(
                data_tp, p_pv, p_load, unit_load_cost, unit_prod_price
            )

            results_list.append(opt_tp)

        # Concatenate all results at once (Much faster than appending inside the loop)
        if results_list:
            self.opt_res = pd.concat(results_list, axis=0)
        else:
            self.opt_res = pd.DataFrame()

        return self.opt_res

    def perform_dayahead_forecast_optim(
        self,
        df_input_data: pd.DataFrame,
        p_pv: pd.Series,
        p_load: pd.Series,
        soc_init: float | list | None = None,
        soc_final: float | list | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
        stage_times: dict[str, float] | None = None,
        def_init_temp: list | None = None,
        room_blind_positions: list | None = None,
        room_opening_open: list | None = None,
        room_door_open: list | None = None,
    ) -> pd.DataFrame:
        r"""
        Perform a day-ahead optimization task using real forecast data. \
        This type of optimization is intented to be launched once a day.

        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param p_pv: The forecasted PV power production.
        :type p_pv: pandas.DataFrame
        :param p_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type p_load: pandas.DataFrame
        :param soc_init: Optional initial battery SOC for the optimization, as a \
            single float (broadcast to every battery) or a list of exactly \
            ``number_of_batteries`` entries for a per-battery initial SOC. \
            When ``None`` (the default), falls back to ``soc_final`` if set, \
            otherwise to ``battery_target_state_of_charge`` from the plant config.
        :type soc_init: float | list, optional
        :param soc_final: Optional final battery SOC for the optimization, as a \
            single float (broadcast to every battery) or a list of exactly \
            ``number_of_batteries`` entries for a per-battery final SOC. \
            When ``None`` (the default), falls back to ``soc_init`` if set, \
            otherwise to ``battery_target_state_of_charge``. Passing an explicit \
            ``soc_final`` distinct from ``soc_init`` is required to plan a \
            net battery charge / discharge across the horizon — notably when \
            ``set_battery_first_priority`` is enabled and the horizon starts \
            at a high SOC.
        :type soc_final: float | list, optional
        :param def_total_hours: Optional per-load runtime override for
            ``operating_hours_of_each_deferrable_load`` (e.g. a manual load's
            live ready/committed state, or a resolved WashData profile's
            step count). Falls back to ``self.optim_conf[...]`` when ``None``.
        :type def_total_hours: list, optional
        :param def_total_timestep: Optional per-load runtime override for
            ``operating_timesteps_of_each_deferrable_load``.
        :type def_total_timestep: list, optional
        :param def_start_timestep: Optional per-load runtime override for
            ``start_timesteps_of_each_deferrable_load`` (e.g. a manual load's
            pinned committed-start window).
        :type def_start_timestep: list, optional
        :param def_end_timestep: Optional per-load runtime override for
            ``end_timesteps_of_each_deferrable_load``.
        :type def_end_timestep: list, optional
        :param stage_times: Optional dict to record nested sub-stage timings
            (``optim_solve.build`` / ``optim_solve.solve`` / ``optim_solve.extract``).
        :type stage_times: dict, optional
        :param def_init_temp: Optional per-load live starting temperature override \
            (e.g. from a real HA room/heat-pump sensor), length == number_of_deferrable_loads. \
            Entries that are None fall back to each load's static config start_temperature.
        :type def_init_temp: list, optional
        :param room_blind_positions: Optional per-load live blind/shading position \
            override (0=open, 1=fully closed), length == number_of_deferrable_loads. \
            Entries that are None fall back to each load's static config (open/no shading). \
            Same live-override shape and cache-hit limitation as def_init_temp - see \
            update_thermal_params.
        :type room_blind_positions: list, optional
        :param room_opening_open: Optional per-load live "window OR door is open \
            right now" override, length == number_of_deferrable_loads. Unlike \
            room_blind_positions this is never held flat across the forecast \
            horizon - it only ever affects the near-term/current timestep of this \
            solve (pauses heating and adds an extra ventilation-loss term there), \
            since a live window/door state can't be forecast for future steps. \
            Refreshes on every solve, including cache hits (see param_window_masks).
        :type room_opening_open: list, optional
        :param room_door_open: Optional per-load live "door is open right now" \
            override, length == number_of_deferrable_loads - deliberately door-only \
            (unlike room_opening_open above, which also considers the window \
            sensor), feeding a boosted thermal-coupling conductance to any declared \
            neighbor(s) of that room. Same cold-build-only cache-hit limitation as \
            room_blind_positions - see _add_room_thermal_coupling_constraints.
        :type room_door_open: list, optional
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info(
            f"Perform optimization for the day-ahead with soc_init: {soc_init}, soc_final: {soc_final}"
        )

        # Extract cost arrays (ensure they are flat numpy arrays)
        unit_load_cost = df_input_data[self.var_load_cost].values
        unit_prod_price = df_input_data[self.var_prod_price].values

        # Call optimization function
        # Note: .ravel() ensures 1D arrays, compatible with cvxpy Parameter shapes
        self.opt_res = self.perform_optimization(
            df_input_data,
            p_pv.values.ravel(),
            p_load.values.ravel(),
            unit_load_cost,
            unit_prod_price,
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            stage_times=stage_times,
            def_init_temp=def_init_temp,
            room_blind_positions=room_blind_positions,
            room_opening_open=room_opening_open,
            room_door_open=room_door_open,
        )
        return self.opt_res

    def perform_naive_mpc_optim(
        self,
        df_input_data: pd.DataFrame,
        p_pv: pd.Series,
        p_load: pd.Series,
        prediction_horizon: int,
        soc_init: float | list | None = None,
        soc_final: float | list | None = None,
        soc_target: float | None = None,
        soc_target_timestep: int | None = None,
        current_period_peak: float | None = None,
        def_total_hours: list | None = None,
        def_total_timestep: list | None = None,
        def_start_timestep: list | None = None,
        def_end_timestep: list | None = None,
        stage_times: dict[str, float] | None = None,
        def_init_temp: list | None = None,
        room_blind_positions: list | None = None,
        room_opening_open: list | None = None,
        room_door_open: list | None = None,
    ) -> pd.DataFrame:
        r"""
        Perform a naive approach to a Model Predictive Control (MPC). \
        This implementaion is naive because we are not using the formal formulation \
        of a MPC. Only the sense of a receiding horizon is considered here. \
        This optimization is more suitable for higher optimization frequency, ex: 5min.

        :param df_input_data: A DataFrame containing all the input data used for \
            the optimization, notably the unit load cost for power consumption.
        :type df_input_data: pandas.DataFrame
        :param p_pv: The forecasted PV power production.
        :type p_pv: pandas.DataFrame
        :param p_load: The forecasted Load power consumption. This power should \
            not include the power from the deferrable load that we want to find.
        :type p_load: pandas.DataFrame
        :param prediction_horizon: The prediction horizon of the MPC controller in number \
            of optimization time steps.
        :type prediction_horizon: int
        :param soc_init: The initial battery SOC for the optimization, as a single \
            float (broadcast to every battery) or a list of exactly \
            ``number_of_batteries`` entries for a per-battery initial SOC. This \
            parameter is optional, if not given soc_init = soc_final = soc_target \
            from the configuration file.
        :type soc_init: float | list
        :param soc_final: The final battery SOC for the optimization, as a single \
            float (broadcast to every battery) or a list of exactly \
            ``number_of_batteries`` entries for a per-battery final SOC. This \
            parameter is optional, if not given soc_init = soc_final = soc_target \
            from the configuration file.
        :type soc_final: float | list
        :param soc_target: An optional intermediate minimum battery SOC (fraction in [0, 1]) that \
            must be reached by ``soc_target_timestep``, after which the battery is free to \
            discharge again. When ``None`` (the default) no intermediate target is imposed and \
            behaviour is unchanged. See issue #553.
        :type soc_target: float
        :param soc_target_timestep: The 0-based horizon timestep by which ``soc_target`` must be \
            met. The index refers to the SoC *after* that timestep's flow. Defaults to the last \
            timestep when ``soc_target`` is given but this is ``None``. Ignored when \
            ``soc_target`` is ``None``.
        :type soc_target_timestep: int
        :param current_period_peak: Optional peak grid import (in Watts) already \
            incurred during the current billing period. When the capacity charge \
            (``capacity_cost_per_kw`` > 0) is active, the planned import peak is \
            floored at this value so the optimization does not spend battery or \
            deferrable flexibility shaving below a peak already locked in for the \
            month. ``None`` / 0 (the default) prices the full horizon peak, \
            identical to omitting it. Ignored when ``capacity_cost_per_kw`` is 0. \
            Runtime-only; only used by naive-mpc-optim. See issue #623.
        :type current_period_peak: float
        :param def_total_timestep: The functioning timesteps for this iteration for each deferrable load. \
            (For continuous deferrable loads: functioning timesteps at nominal power)
        :type def_total_timestep: list
        :param def_total_hours: The functioning hours for this iteration for each deferrable load. \
            (For continuous deferrable loads: functioning hours at nominal power)
        :type def_total_hours: list
        :param def_start_timestep: The timestep as from which each deferrable load is allowed to operate.
        :type def_start_timestep: list
        :param def_end_timestep: The timestep before which each deferrable load should operate.
        :type def_end_timestep: list
        :param def_init_temp: Optional per-load live starting temperature override \
            (e.g. from a real HA room/heat-pump sensor), length == number_of_deferrable_loads. \
            Entries that are None fall back to each load's static config start_temperature.
        :type def_init_temp: list, optional
        :param room_blind_positions: Optional per-load live blind/shading position \
            override (0=open, 1=fully closed), length == number_of_deferrable_loads. \
            Entries that are None fall back to each load's static config (open/no shading). \
            Same live-override shape and cache-hit limitation as def_init_temp - see \
            update_thermal_params.
        :type room_blind_positions: list, optional
        :param room_opening_open: Optional per-load live "window OR door is open \
            right now" override, length == number_of_deferrable_loads. Only ever \
            affects the near-term/current timestep of this solve (pauses heating, \
            adds an extra ventilation-loss term) - refreshes on every solve, \
            including cache hits.
        :type room_opening_open: list, optional
        :param room_door_open: Optional per-load live "door is open right now" \
            override, length == number_of_deferrable_loads - door-only, feeds a \
            boosted thermal-coupling conductance to any declared neighbor(s). Same \
            cold-build-only cache-hit limitation as room_blind_positions.
        :type room_door_open: list, optional
        :return: opt_res: A DataFrame containing the optimization results
        :rtype: pandas.DataFrame

        """
        self.logger.info("Perform an iteration of a naive MPC controller")

        if prediction_horizon < 5:
            self.logger.error(
                "Set the MPC prediction horizon to at least 5 times the optimization time step"
            )
            return pd.DataFrame()

        # Verify compatibility with Fixed Problem Size (Define Once Architecture)
        if prediction_horizon != self.num_timesteps:
            self.logger.warning(
                f"MPC Prediction Horizon ({prediction_horizon}) does not match the initialized "
                f"optimization window ({self.num_timesteps}). "
                "This may cause shape mismatch errors in the solver."
            )

        # Slice data to horizon
        subset_data = copy.deepcopy(df_input_data).iloc[:prediction_horizon]

        # Extract inputs as arrays
        # Note: We must ensure p_pv and p_load are sliced exactly like df_input_data
        p_pv_slice = p_pv.iloc[:prediction_horizon].values.ravel()
        p_load_slice = p_load.iloc[:prediction_horizon].values.ravel()
        unit_load_cost = subset_data[self.var_load_cost].values
        unit_prod_price = subset_data[self.var_prod_price].values

        # Call optimization function
        self.opt_res = self.perform_optimization(
            subset_data,
            p_pv_slice,
            p_load_slice,
            unit_load_cost,
            unit_prod_price,
            soc_init=soc_init,
            soc_final=soc_final,
            soc_target=soc_target,
            soc_target_timestep=soc_target_timestep,
            current_period_peak=current_period_peak,
            def_total_hours=def_total_hours,
            def_total_timestep=def_total_timestep,
            def_start_timestep=def_start_timestep,
            def_end_timestep=def_end_timestep,
            stage_times=stage_times,
            def_init_temp=def_init_temp,
            room_blind_positions=room_blind_positions,
            room_opening_open=room_opening_open,
            room_door_open=room_door_open,
        )
        return self.opt_res

    @staticmethod
    def validate_def_timewindow(
        start: int, end: int, min_steps: int, window: int
    ) -> tuple[int, int, str]:
        r"""
        Helper function to validate (and if necessary: correct) the defined optimization window of a deferrable load.

        :param start: Start timestep of the optimization window of the deferrable load
        :type start: int
        :param end: End timestep of the optimization window of the deferrable load
        :type end: int
        :param min_steps: Minimal timesteps during which the load should operate (at nominal power)
        :type min_steps: int
        :param window: Total number of timesteps in the optimization window
        :type window: int
        :return: start_validated: Validated start timestep of the optimization window of the deferrable load
        :rtype: int
        :return: end_validated: Validated end timestep of the optimization window of the deferrable load
        :rtype: int
        :return: warning: Any warning information to be returned from the validation steps
        :rtype: string

        """
        start_validated = 0
        end_validated = 0
        warning = None
        # Verify that start <= end
        if start <= end or start <= 0 or end <= 0:
            # start and end should be within the optimization timewindow [0, window]
            start_validated = max(0, min(window, start))
            end_validated = max(0, min(window, end))
            if end_validated > 0:
                # If the available timeframe is shorter than the number of timesteps needed to meet the hours to operate (def_total_hours), issue a warning.
                if (end_validated - start_validated) < min_steps:
                    warning = "Available timeframe is shorter than the specified number of hours to operate. Optimization will fail."
        else:
            warning = "Invalid timeframe for deferrable load (start timestep is not <= end timestep). Continuing optimization without timewindow constraint."
        return start_validated, end_validated, warning
