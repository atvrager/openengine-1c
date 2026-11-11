"""Thermodynamic and combustion modeling."""

from .cantera_kinetics import (
    CanteraThermochemistry,
    EquilibriumCombustionResult,
    FuelBlendSpec,
    ThermodynamicState,
)
from .cycle import (
    ClosedCycleSimulator,
    CycleResults,
    EngineGeometry,
    wiebe_burned_fraction,
)
from .gas_exchange import (
    FourStrokeResults,
    FourStrokeSimulator,
    ValveGeometry,
    ValveTiming,
    compressible_orifice_flow,
)
from .knock import (
    KnockEvaluationResult,
    KnockSafeOperatingMap,
    KnockSweepPoint,
    compute_adiabatic_end_gas_temperature,
    douaud_eyzat_ignition_delay_s,
    evaluate_cycle_knock,
    select_best_knock_safe_map,
    soc_from_spark_advance,
)
from .woschni import HeatTransferOutputs, WoschniHeatTransfer

__all__ = [
    "EngineGeometry",
    "wiebe_burned_fraction",
    "ClosedCycleSimulator",
    "CycleResults",
    "ValveTiming",
    "ValveGeometry",
    "compressible_orifice_flow",
    "FourStrokeSimulator",
    "FourStrokeResults",
    "WoschniHeatTransfer",
    "HeatTransferOutputs",
    "CanteraThermochemistry",
    "FuelBlendSpec",
    "ThermodynamicState",
    "EquilibriumCombustionResult",
    "KnockEvaluationResult",
    "KnockSweepPoint",
    "KnockSafeOperatingMap",
    "douaud_eyzat_ignition_delay_s",
    "compute_adiabatic_end_gas_temperature",
    "evaluate_cycle_knock",
    "select_best_knock_safe_map",
    "soc_from_spark_advance",
]
