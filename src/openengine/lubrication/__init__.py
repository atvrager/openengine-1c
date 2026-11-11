"""Wet-sump splash lubrication, multigrade rheology, FMEP, and oil cooling."""

from .friction import (
    FrictionBreakdownResult,
    compute_valvetrain_mechanical_work_j,
    evaluate_engine_friction,
)
from .rheology import (
    EngineLubricationConfigRoot,
    LubricationConfigSection,
    OilGradeSpec,
    OilState,
    fit_walther_coefficients,
    walther_kinematic_viscosity_cst,
)
from .splash import (
    SplashSystemParams,
    SplashTransportReport,
    evaluate_splash_transport,
)
from .thermal import (
    SumpThermalBalanceResult,
    SumpThermalParams,
    compute_piston_splash_cooling,
    compute_sump_overall_ua_w_k,
    solve_sump_thermal_balance,
)

__all__ = [
    "OilState",
    "OilGradeSpec",
    "LubricationConfigSection",
    "EngineLubricationConfigRoot",
    "fit_walther_coefficients",
    "walther_kinematic_viscosity_cst",
    "SplashSystemParams",
    "SplashTransportReport",
    "evaluate_splash_transport",
    "FrictionBreakdownResult",
    "compute_valvetrain_mechanical_work_j",
    "evaluate_engine_friction",
    "SumpThermalParams",
    "SumpThermalBalanceResult",
    "compute_sump_overall_ua_w_k",
    "compute_piston_splash_cooling",
    "solve_sump_thermal_balance",
]
