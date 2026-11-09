"""Computational Fluid Dynamics (CFD) simulation suite."""

from openengine.cfd.cht_cooling import (
    ChtCoolingConfig,
    ChtCoolingResult,
    generate_finned_block_mesh,
    run_cht_simulation,
    setup_cht_case,
)
from openengine.cfd.closed_loop import (
    ClosedLoopCalibrationResult,
    ClosedLoopConfig,
    CycleComparisonMetrics,
    run_closed_loop_calibration,
)
from openengine.cfd.combustion import (
    CombustionConfig,
    CombustionResult,
    CombustionStepPoint,
    WiebeFitResult,
    calibrate_wiebe_from_trajectory,
    compute_clearance_height,
    generate_combustion_mesh,
    run_combustion_simulation,
    setup_combustion_case,
)
from openengine.cfd.port_bench import (
    LiftPointResult,
    PortFlowConfig,
    PortFlowSweepResult,
    compute_tumble_and_swirl,
    generate_port_geometry_at_lift,
    run_port_flow_sweep,
    solve_port_flow_at_lift,
)

__all__ = [
    "ChtCoolingConfig",
    "ChtCoolingResult",
    "generate_finned_block_mesh",
    "run_cht_simulation",
    "setup_cht_case",
    "ClosedLoopCalibrationResult",
    "ClosedLoopConfig",
    "CycleComparisonMetrics",
    "run_closed_loop_calibration",
    "CombustionConfig",
    "CombustionResult",
    "CombustionStepPoint",
    "WiebeFitResult",
    "calibrate_wiebe_from_trajectory",
    "compute_clearance_height",
    "generate_combustion_mesh",
    "run_combustion_simulation",
    "setup_combustion_case",
    "LiftPointResult",
    "PortFlowConfig",
    "PortFlowSweepResult",
    "compute_tumble_and_swirl",
    "generate_port_geometry_at_lift",
    "run_port_flow_sweep",
    "solve_port_flow_at_lift",
]
