"""Full-engine CalculiX C3D10 FEA stress atlas across all canonical parts and 5 CAD assemblies."""

import base64
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from openengine.cad.assemblies.crankcase_assembly import build_crankcase_assembly_parts
from openengine.cad.assemblies.dressed_engine_assembly import (
    STRUCTURAL_HOUSING_KEYS,
    build_dressed_engine_parts,
    material_key_for_dressed_part,
)
from openengine.cad.assemblies.head_assembly import build_head_assembly_parts
from openengine.cad.assemblies.rotating_assembly import build_rotating_assembly_parts
from openengine.cad.assembly import EngineAssembly
from openengine.cad.crankcase import CRANK_AXIS_Z_MM
from openengine.fea.calculix import (
    ALUMINUM_4032_T6,
    ALUMINUM_7075_T6,
    BoundaryMeshData,
    FeaNodalResults,
    MaterialProperties,
    get_material_properties,
    parse_calculix_inp_mesh,
    parse_frd_results,
    run_ccx_batch,
)
from openengine.fea.connecting_rod import ConrodFeaConfig, run_conrod_fea
from openengine.fea.piston import PistonFeaConfig, run_piston_fea
from openengine.meshing.gmsh_structural import format_nset_cards, mesh_structural_part_c3d10

AX_X: int = 0
AX_Y: int = 1
AX_Z: int = 2
NUM_AXES: int = 3
ZERO_F: float = 0.0
UNIT_F: float = 1.0
HALF_F: float = 0.5
LARGE_BOUND: float = 1.0e6
RADIAL_EPS_MM: float = 1.0e-6
SPAN_EPS_MM: float = 1.0e-3
MIN_STRESS_EPS_MPA: float = 0.1
BBOX_CONGRUENCE_TOL_MM: float = 0.08

MIN_FIX_NODES: int = 6
MIN_LOAD_NODES: int = 3
MIN_FIX_EXTENT_MM: float = 0.25
FALLBACK_FIX_FRAC: float = 0.20
FALLBACK_LOAD_FRAC: float = 0.80
MIN_YIELD_SF_GATE: float = 1.20

PISTON_PIN_WORLD_Z_OFFSET_MM: float = 3.0
CONROD_LC3A_FORCE_N: float = 28327.0
CONROD_LC3B_FORCE_N: float = 2540.0
PISTON_FIRING_PMAX_MPA: float = 7.8

THIN_PART_MIN_DIM_MM: float = 1.2
SHELL_PART_MIN_DIM_MM: float = 2.5
LARGE_BILLET_VOL_MM3: float = 80000.0
LARGE_BILLET_DIAG_MM: float = 95.0
MEDIUM_PART_DIAG_MM: float = 30.0
THIN_MESH_MIN_MM: float = 0.55
THIN_MESH_MAX_MM: float = 2.0
THIN_DIAG_SCALE: float = 0.03
SHELL_MESH_MIN_MM: float = 0.65
SHELL_MESH_MAX_MM: float = 2.6
SHELL_DIAG_SCALE: float = 0.035
LARGE_MESH_MIN_MM: float = 2.6
LARGE_MESH_MAX_MM: float = 6.2
MED_MESH_MIN_MM: float = 1.0
MED_MESH_MAX_MM: float = 3.0
SMALL_MESH_MIN_MM: float = 0.70
SMALL_MESH_MAX_MM: float = 2.0
CHAIN_KEY: str = "roller_chain_25"
CHAIN_MESH_MIN_MM: float = 1.3
CHAIN_MESH_MAX_MM: float = 3.8

VISUAL_MAX_DISP_MM: float = 0.35
VISUAL_DISP_FLOOR_MM: float = 1.0e-5
DEFAULT_VISUAL_DISP_SCALE: float = 12.0
COORD_ROUND_DECIMALS: int = 3
STRESS_ROUND_DECIMALS: int = 2
DISP_ROUND_DECIMALS: int = 4
SF_ROUND_DECIMALS: int = 2
TIME_ROUND_DECIMALS: int = 2
P95_PERCENTILE: int = 95

TRANSLATION_ALIAS_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("base_bolt_1", ("base_bolt_2", "base_bolt_3", "base_bolt_4")),
    ("base_washer_1", ("base_washer_2", "base_washer_3", "base_washer_4")),
    ("head_bolt_1", ("head_bolt_2", "head_bolt_3", "head_bolt_4")),
    ("head_washer_1", ("head_washer_2", "head_washer_3", "head_washer_4")),
    ("manifold_bolt_1", ("manifold_bolt_2",)),
    ("tower_bolt_1", ("tower_bolt_2", "tower_bolt_3", "tower_bolt_4")),
    ("cover_bolt_1", ("cover_bolt_2", "cover_bolt_3", "cover_bolt_4")),
    (
        "split_bolt_1",
        (
            "split_bolt_2",
            "split_bolt_3",
            "split_bolt_4",
            "split_bolt_5",
            "split_bolt_6",
            "split_bolt_7",
            "split_bolt_8",
        ),
    ),
    ("split_dowel_1", ("split_dowel_2",)),
    (
        "sump_bolt_1",
        (
            "sump_bolt_2",
            "sump_bolt_3",
            "sump_bolt_4",
            "sump_bolt_5",
            "sump_bolt_6",
            "sump_bolt_7",
            "sump_bolt_8",
            "sump_bolt_9",
            "sump_bolt_10",
        ),
    ),
    (
        "timing_cover_bolt_1",
        (
            "timing_cover_bolt_2",
            "timing_cover_bolt_3",
            "timing_cover_bolt_4",
            "timing_cover_bolt_5",
            "timing_cover_bolt_6",
        ),
    ),
    ("conrod_bolt_1", ("conrod_bolt_2",)),
    ("conrod_dowel_1", ("conrod_dowel_2",)),
    ("circlip_left", ("circlip_right",)),
    ("main_bearing_left", ("main_bearing_right",)),
    ("left_shaft_seal", ("right_shaft_seal",)),
    ("rocker_shaft_intake", ("rocker_shaft_exhaust",)),
    ("top_ring", ("second_ring",)),
)


@dataclass(frozen=True, slots=True)
class PartStressSpec:
  """Boundary condition, load application, and UI catalog metadata for one canonical solid."""

  label: str
  group: str
  fix_rule: str
  fix_params: Tuple[float, ...]
  load_rule: str
  load_params: Tuple[float, ...]
  load_mode: str
  force_n: float
  is_housing: bool = False


PART_STRESS_SPECS: Mapping[str, PartStressSpec] = {
    "cylinder_head": PartStressSpec(
        "Cylinder Head (Pent-Roof 6061-T6)",
        "primary",
        "z_band",
        (36.0, 55.0, 35.0, 90.0),
        "z_band",
        (-1.0, 20.0, 0.0, 35.0),
        "force_z",
        28327.0,
        True,
    ),
    "cylinder_block": PartStressSpec(
        "Cylinder Block (Finned 6061-T6)",
        "primary",
        "z_band",
        (-86.0, -70.0, 38.0, 95.0),
        "z_band",
        (-42.0, 0.5, 32.0, 44.0),
        "radial_xy",
        16000.0,
        True,
    ),
    "cylinder_liner": PartStressSpec(
        "Cylinder Liner (DOM Steel A513-T5)",
        "primary",
        "z_band",
        (-10.0, 0.5, 30.0, 40.0),
        "z_band",
        (-48.0, -11.0, 30.0, 36.2),
        "radial_xy",
        22000.0,
        False,
    ),
    "crankshaft": PartStressSpec(
        "Crankshaft (4140 Pre-Hardened)",
        "primary",
        "y_band_r_crank",
        (24.0, 45.0, 0.0, 17.0),
        "crank_pin",
        (13.0, -142.0),
        "force_z",
        -28327.0,
        False,
    ),
    "conrod_beam": PartStressSpec(
        "Connecting Rod Beam (7075-T6)",
        "primary",
        "z_band",
        (-133.0, -116.0, 0.0, 50.0),
        "z_band",
        (-34.0, -10.0, 0.0, 25.0),
        "force_z",
        -28327.0,
        False,
    ),
    "conrod_cap": PartStressSpec(
        "Connecting Rod Cap (7075-T6)",
        "primary",
        "z_band",
        (-135.5, -131.5, 0.0, 50.0),
        "z_band",
        (-163.0, -142.0, 0.0, 26.0),
        "force_z",
        -5800.0,
        False,
    ),
    "crankcase_left": PartStressSpec(
        "Left Crankcase Half (6061-T6)",
        "primary",
        "z_band",
        (-242.0, -230.0, 0.0, 120.0),
        "y_band_r_crank",
        (24.0, 46.0, 0.0, 34.0),
        "force_z",
        -14160.0,
        True,
    ),
    "crankcase_right": PartStressSpec(
        "Right Crankcase Half (6061-T6)",
        "primary",
        "z_band",
        (-242.0, -230.0, 0.0, 120.0),
        "y_band_r_crank",
        (24.0, 46.0, 0.0, 34.0),
        "force_z",
        -14160.0,
        True,
    ),
    "sump_pan": PartStressSpec(
        "Sloped Wet Sump Pan (6061-T6)",
        "primary",
        "z_band",
        (-248.0, -240.0, 0.0, 120.0),
        "z_band",
        (-280.0, -264.0, 0.0, 120.0),
        "force_z",
        -2400.0,
        True,
    ),
    "flywheel": PartStressSpec(
        "Inertia Flywheel (8400 RPM Burst)",
        "primary",
        "y_band_r_crank",
        (55.0, 86.0, 0.0, 20.0),
        "y_band_r_crank",
        (55.0, 86.0, 52.0, 85.0),
        "radial_crank_xz",
        38000.0,
        False,
    ),
    "trigger_wheel": PartStressSpec(
        "36-1 Trigger Wheel (1144 Steel)",
        "cots",
        "y_band_r_crank",
        (80.0, 88.0, 0.0, 17.5),
        "y_band_r_crank",
        (80.0, 88.0, 23.0, 40.0),
        "radial_crank_xz",
        2800.0,
        False,
    ),
    "cam_tower": PartStressSpec(
        "SOHC Cam Tower (6061-T6)",
        "primary",
        "z_band",
        (51.0, 58.0, 0.0, 100.0),
        "z_band",
        (82.0, 107.0, 0.0, 70.0),
        "force_z",
        2600.0,
        True,
    ),
    "cam_cover": PartStressSpec(
        "SOHC Valve & Cam Cover (6061-T6)",
        "primary",
        "z_band",
        (106.0, 111.0, 0.0, 100.0),
        "z_band",
        (122.0, 130.0, 0.0, 100.0),
        "force_z",
        -1800.0,
        True,
    ),
    "right_timing_cover": PartStressSpec(
        "Right Timing Chain Cover (6061-T6)",
        "primary",
        "frac_axis",
        (float(AX_Y), 0.0, 0.28),
        "frac_axis",
        (float(AX_Y), 0.72, 1.0),
        "force_y",
        1200.0,
        True,
    ),
    "fan_shroud": PartStressSpec(
        "Cooling Fan Shroud (6061-T6)",
        "primary",
        "y_band_r_crank",
        (97.0, 108.0, 55.0, 95.0),
        "y_band_r_crank",
        (97.0, 108.0, 12.0, 50.0),
        "force_y",
        450.0,
        True,
    ),
    "intake_manifold_adapter": PartStressSpec(
        "Intake Manifold Adapter (6061-T6)",
        "cots",
        "frac_axis",
        (float(AX_X), 0.0, 0.25),
        "frac_axis",
        (float(AX_X), 0.75, 1.0),
        "force_z",
        -750.0,
        True,
    ),
    "carburetor_vm26": PartStressSpec(
        "VM26 Slide Carburetor (Diecast)",
        "cots",
        "frac_axis",
        (float(AX_X), 0.0, 0.22),
        "frac_axis",
        (float(AX_X), 0.65, 1.0),
        "force_z",
        -350.0,
        True,
    ),
    "camshaft": PartStressSpec(
        "SOHC One-Piece Camshaft (4140)",
        "primary",
        "frac_long_ends",
        (0.24,),
        "frac_long",
        (0.32, 0.68),
        "force_z",
        -1350.0,
        False,
    ),
    "rocker_intake": PartStressSpec(
        "Intake Rocker Arm (7075-T6)",
        "primary",
        "pivot_xz",
        (19.5, 100.0, 0.0, 7.0, 0.0),
        "frac_long",
        (0.75, 1.0),
        "force_z",
        -950.0,
        False,
    ),
    "rocker_exhaust": PartStressSpec(
        "Exhaust Rocker Arm (7075-T6)",
        "primary",
        "pivot_xz",
        (-19.5, 100.0, 0.0, 7.0, 0.0),
        "frac_long",
        (0.0, 0.25),
        "force_z",
        -950.0,
        False,
    ),
    "rocker_shaft_intake": PartStressSpec(
        "Rocker Pivot Shaft (4140 Steel)",
        "cots",
        "frac_long_ends",
        (0.25,),
        "frac_long",
        (0.35, 0.65),
        "force_z",
        -1150.0,
        False,
    ),
    "valve_intake": PartStressSpec(
        "Intake Poppet Valve (34 mm 4140)",
        "primary",
        "frac_long",
        (0.30, 0.58),
        "frac_long",
        (0.0, 0.08),
        "force_z",
        3200.0,
        False,
    ),
    "valve_exhaust": PartStressSpec(
        "Exhaust Poppet Valve (28 mm 4140)",
        "primary",
        "frac_long",
        (0.30, 0.58),
        "frac_long",
        (0.0, 0.08),
        "force_z",
        2600.0,
        False,
    ),
    "valve_seat_intake": PartStressSpec(
        "Intake Valve Seat Ring (4140)",
        "cots",
        "ring_radial",
        (0.80, 1.0),
        "ring_radial",
        (0.0, 0.35),
        "force_z",
        -1800.0,
        False,
    ),
    "valve_seat_exhaust": PartStressSpec(
        "Exhaust Valve Seat Ring (4140)",
        "cots",
        "ring_radial",
        (0.80, 1.0),
        "ring_radial",
        (0.0, 0.35),
        "force_z",
        -1600.0,
        False,
    ),
    "valve_guide_intake": PartStressSpec(
        "Intake Valve Guide (SAE 660 Bronze)",
        "cots",
        "ring_radial",
        (0.82, 1.0),
        "frac_long",
        (0.0, 0.30),
        "force_x",
        360.0,
        False,
    ),
    "valve_guide_exhaust": PartStressSpec(
        "Exhaust Valve Guide (SAE 660 Bronze)",
        "cots",
        "ring_radial",
        (0.82, 1.0),
        "frac_long",
        (0.0, 0.30),
        "force_x",
        -360.0,
        False,
    ),
    "spring_intake": PartStressSpec(
        "Intake Valve Spring (9573K35)",
        "cots",
        "frac_axis",
        (float(AX_Z), 0.0, 0.20),
        "frac_axis",
        (float(AX_Z), 0.80, 1.0),
        "force_z",
        -190.0,
        False,
    ),
    "spring_exhaust": PartStressSpec(
        "Exhaust Valve Spring (9573K35)",
        "cots",
        "frac_axis",
        (float(AX_Z), 0.0, 0.20),
        "frac_axis",
        (float(AX_Z), 0.80, 1.0),
        "force_z",
        -190.0,
        False,
    ),
    "retainer_intake": PartStressSpec(
        "Intake Spring Retainer (7075-T6)",
        "cots",
        "ring_radial",
        (0.0, 0.35),
        "ring_radial",
        (0.75, 1.0),
        "force_z",
        -580.0,
        False,
    ),
    "retainer_exhaust": PartStressSpec(
        "Exhaust Spring Retainer (7075-T6)",
        "cots",
        "ring_radial",
        (0.0, 0.35),
        "ring_radial",
        (0.75, 1.0),
        "force_z",
        -580.0,
        False,
    ),
    "wrist_pin": PartStressSpec(
        "Wrist Pin (16x45 mm 4140)",
        "primary",
        "frac_long_ends",
        (0.26,),
        "frac_long",
        (0.35, 0.65),
        "force_z",
        -18000.0,
        False,
    ),
    "sprocket_cam_36t": PartStressSpec(
        "Cam Sprocket 36T (#25 4140)",
        "cots",
        "ring_radial",
        (0.0, 0.38),
        "ring_radial",
        (0.84, 1.0),
        "force_z",
        -1250.0,
        False,
    ),
    "sprocket_crank_18t": PartStressSpec(
        "Crank Sprocket 18T (#25 4140)",
        "cots",
        "ring_radial",
        (0.0, 0.42),
        "ring_radial",
        (0.84, 1.0),
        "force_z",
        1250.0,
        False,
    ),
    "roller_chain_25": PartStressSpec(
        "#25 Roller Timing Chain (4140)",
        "cots",
        "frac_axis",
        (float(AX_Z), 0.0, 0.16),
        "frac_axis",
        (float(AX_Z), 0.84, 1.0),
        "force_z",
        280.0,
        False,
    ),
    "main_bearing_left": PartStressSpec(
        "6206 Main Ball Bearing (4140)",
        "cots",
        "ring_radial",
        (0.82, 1.0),
        "ring_radial",
        (0.0, 0.30),
        "force_z",
        -9500.0,
        False,
    ),
    "needle_bearing_nks28": PartStressSpec(
        "NKS28 Big-End Needle Bearing",
        "cots",
        "ring_radial",
        (0.82, 1.0),
        "ring_radial",
        (0.0, 0.35),
        "force_z",
        -12000.0,
        False,
    ),
    "top_ring": PartStressSpec(
        "Piston Compression Ring (68 mm)",
        "cots",
        "frac_axis",
        (float(AX_X), 0.0, 0.25),
        "ring_radial",
        (0.0, 0.40),
        "radial_ring",
        65.0,
        False,
    ),
    "oil_ring": PartStressSpec(
        "Piston Oil Control Ring (68 mm)",
        "cots",
        "frac_axis",
        (float(AX_X), 0.0, 0.25),
        "ring_radial",
        (0.0, 0.40),
        "radial_ring",
        55.0,
        False,
    ),
    "circlip_left": PartStressSpec(
        "16 mm Wrist Pin Circlip",
        "cots",
        "ring_radial",
        (0.68, 1.0),
        "ring_radial",
        (0.0, 0.45),
        "force_y",
        -35.0,
        False,
    ),
    "head_bolt_1": PartStressSpec(
        "M8x65 Head SHCS (Grade 10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        12500.0,
        False,
    ),
    "base_bolt_1": PartStressSpec(
        "M8x32 Barrel Base SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        11000.0,
        False,
    ),
    "conrod_bolt_1": PartStressSpec(
        "M6x26 Con-Rod Cap SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        6200.0,
        False,
    ),
    "tower_bolt_1": PartStressSpec(
        "M6x32 Cam Tower SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        5400.0,
        False,
    ),
    "cover_bolt_1": PartStressSpec(
        "M6x26 Cam Cover SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        4800.0,
        False,
    ),
    "split_bolt_1": PartStressSpec(
        "M6x40 Split-Case SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        5800.0,
        False,
    ),
    "sump_bolt_1": PartStressSpec(
        "M6x22 Sump Pan SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        4600.0,
        False,
    ),
    "timing_cover_bolt_1": PartStressSpec(
        "M6x18 Timing Cover SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        4400.0,
        False,
    ),
    "manifold_bolt_1": PartStressSpec(
        "M6x20 Manifold Flange SHCS (10.9)",
        "cots",
        "frac_long",
        (0.0, 0.28),
        "frac_long",
        (0.82, 1.0),
        "force_long",
        4800.0,
        False,
    ),
    "head_washer_1": PartStressSpec(
        "M8 Hardened Head Washer",
        "cots",
        "frac_short",
        (0.0, 0.32),
        "frac_short",
        (0.68, 1.0),
        "force_z",
        -12500.0,
        False,
    ),
    "base_washer_1": PartStressSpec(
        "M8 Hardened Base Washer",
        "cots",
        "frac_short",
        (0.0, 0.32),
        "frac_short",
        (0.68, 1.0),
        "force_z",
        -11000.0,
        False,
    ),
    "conrod_dowel_1": PartStressSpec(
        "5x12 mm Con-Rod Locating Dowel",
        "cots",
        "frac_long_ends",
        (0.28,),
        "frac_long",
        (0.36, 0.64),
        "force_x",
        1200.0,
        False,
    ),
    "split_dowel_1": PartStressSpec(
        "6x16 mm Crankcase Locating Dowel",
        "cots",
        "frac_long_ends",
        (0.28,),
        "frac_long",
        (0.36, 0.64),
        "force_z",
        1600.0,
        False,
    ),
    "spark_plug_cmr7h": PartStressSpec(
        "NGK CMR7H M10 Spark Plug",
        "cots",
        "frac_long",
        (0.22, 0.48),
        "frac_long",
        (0.0, 0.14),
        "force_z",
        1800.0,
        False,
    ),
    "sight_glass_m16": PartStressSpec(
        "M16x1.5 Oil Sight Glass",
        "cots",
        "frac_short",
        (0.72, 1.0),
        "frac_short",
        (0.0, 0.25),
        "force_x",
        320.0,
        False,
    ),
    "oil_fill_plug_m20": PartStressSpec(
        "M20x1.5 Dipstick Fill Plug",
        "cots",
        "frac_long",
        (0.0, 0.30),
        "frac_long",
        (0.75, 1.0),
        "force_long",
        420.0,
        False,
    ),
    "magnetic_drain_plug_m12": PartStressSpec(
        "M12x1.5 Magnetic Drain Plug",
        "cots",
        "frac_long",
        (0.70, 1.0),
        "frac_long",
        (0.0, 0.30),
        "force_long",
        320.0,
        False,
    ),
    "breather_vent_npt": PartStressSpec(
        "1/8-27 NPT Crankcase Breather",
        "cots",
        "frac_long",
        (0.70, 1.0),
        "frac_long",
        (0.0, 0.30),
        "force_long",
        160.0,
        False,
    ),
    "cdi_module": PartStressSpec(
        "CDI Ignition Control Module",
        "cots",
        "frac_long",
        (0.0, 0.25),
        "frac_long",
        (0.70, 1.0),
        "force_x",
        180.0,
        False,
    ),
    "hall_pickup": PartStressSpec(
        "Hall Effect Crank Sensor",
        "cots",
        "frac_long",
        (0.72, 1.0),
        "frac_long",
        (0.0, 0.28),
        "force_y",
        120.0,
        False,
    ),
    "left_shaft_seal": PartStressSpec(
        "30x47x7 Viton Crankshaft Seal",
        "cots",
        "ring_radial",
        (0.80, 1.0),
        "ring_radial",
        (0.0, 0.38),
        "radial_ring",
        65.0,
        False,
    ),
    "cam_cover_oring": PartStressSpec(
        "Cam Cover Viton O-Ring",
        "cots",
        "ring_radial",
        (0.82, 1.0),
        "ring_radial",
        (0.0, 0.38),
        "force_z",
        -12.0,
        False,
    ),
    "sump_perimeter_oring": PartStressSpec(
        "Sump Pan Viton Perimeter O-Ring",
        "cots",
        "frac_short",
        (0.0, 0.35),
        "frac_short",
        (0.65, 1.0),
        "force_z",
        -120.0,
        False,
    ),
    "stem_seal_intake": PartStressSpec(
        "Intake Valve Stem Viton Seal",
        "cots",
        "ring_radial",
        (0.80, 1.0),
        "ring_radial",
        (0.0, 0.38),
        "force_z",
        -4.5,
        False,
    ),
    "stem_seal_exhaust": PartStressSpec(
        "Exhaust Valve Stem Viton Seal",
        "cots",
        "ring_radial",
        (0.80, 1.0),
        "ring_radial",
        (0.0, 0.38),
        "force_z",
        -4.5,
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class AtlasPartRecord:
  """Solved FEA stress and safety summary for one rigid engine part."""

  part_key: str
  canonical_key: str
  label: str
  material: str
  yield_strength_mpa: float
  p95_stress_mpa: float
  max_stress_mpa: float
  max_displacement_mm: float
  yield_safety_factor: float
  num_vertices: int
  num_triangles: int
  offset_mm: Tuple[float, float, float]
  is_housing: bool
  is_pass: bool

  def to_dict(self) -> Dict[str, object]:
    """Serialize solved part summary record to a JSON-safe dictionary."""
    return {
        "part_key": self.part_key,
        "canonical_key": self.canonical_key,
        "label": self.label,
        "material": self.material,
        "yield_strength_mpa": self.yield_strength_mpa,
        "p95_stress_mpa": self.p95_stress_mpa,
        "max_stress_mpa": self.max_stress_mpa,
        "max_displacement_mm": self.max_displacement_mm,
        "yield_safety_factor": self.yield_safety_factor,
        "num_vertices": self.num_vertices,
        "num_triangles": self.num_triangles,
        "offset_mm": list(self.offset_mm),
        "is_housing": self.is_housing,
        "is_pass": self.is_pass,
    }


@dataclass(frozen=True, slots=True)
class AtlasAssemblyRecord:
  """Multi-body assembly summary grouping solved CAD solid stress fields."""

  assembly_id: str
  title: str
  part_count: int
  max_stress_mpa: float
  p95_stress_mpa: float
  min_yield_safety_factor: float
  limiting_part_key: str
  part_keys: Tuple[str, ...]
  is_pass: bool

  def to_dict(self) -> Dict[str, object]:
    """Serialize solved multi-body assembly summary to a JSON-safe dictionary."""
    return {
        "assembly_id": self.assembly_id,
        "title": self.title,
        "part_count": self.part_count,
        "max_stress_mpa": self.max_stress_mpa,
        "p95_stress_mpa": self.p95_stress_mpa,
        "min_yield_safety_factor": self.min_yield_safety_factor,
        "limiting_part_key": self.limiting_part_key,
        "part_keys": list(self.part_keys),
        "is_pass": self.is_pass,
    }


@dataclass(frozen=True, slots=True)
class FeaAtlasSummaryResult:
  """Complete full-engine CalculiX C3D10 stress atlas results."""

  total_solids: int
  unique_meshes: int
  min_yield_safety_factor: float
  limiting_part_key: str
  all_passed: bool
  total_solve_time_s: float
  assemblies: Tuple[AtlasAssemblyRecord, ...]
  parts: Mapping[str, AtlasPartRecord]

  def to_dict(self) -> Dict[str, object]:
    """Serialize full-engine FEA atlas metadata to JSON-safe dictionary."""
    return {
        "total_solids": self.total_solids,
        "unique_meshes": self.unique_meshes,
        "min_yield_safety_factor": self.min_yield_safety_factor,
        "limiting_part_key": self.limiting_part_key,
        "all_passed": self.all_passed,
        "total_solve_time_s": self.total_solve_time_s,
        "assemblies": [item.to_dict() for item in self.assemblies],
        "parts": {k: v.to_dict() for k, v in self.parts.items()},
    }


def _bbox_tuple(part: object) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
  """Return ((dx, dy, dz), (cx, cy, cz)) for any build123d Part."""
  bb = part.bounding_box()
  ctr = bb.center()
  return (float(bb.size.X), float(bb.size.Y), float(bb.size.Z)), (float(ctr.X), float(ctr.Y), float(ctr.Z))


def _resolve_canonical_map(
    dressed_parts: Mapping[str, object],
) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float]]]:
  """Map all 114 dressed parts to canonical template keys and translation offsets."""
  canon_of: Dict[str, str] = {key: key for key in dressed_parts}
  offset_of: Dict[str, Tuple[float, float, float]] = {key: (ZERO_F, ZERO_F, ZERO_F) for key in dressed_parts}
  bboxes = {key: _bbox_tuple(part) for key, part in dressed_parts.items()}

  for template_key, clones in TRANSLATION_ALIAS_GROUPS:
    if template_key not in bboxes:
      continue
    ref_size, ref_ctr = bboxes[template_key]
    for clone_key in clones:
      if clone_key not in bboxes:
        continue
      c_size, c_ctr = bboxes[clone_key]
      if all(abs(c_size[i] - ref_size[i]) <= BBOX_CONGRUENCE_TOL_MM for i in range(NUM_AXES)):
        canon_of[clone_key] = template_key
        offset_of[clone_key] = (
            round(c_ctr[AX_X] - ref_ctr[AX_X], COORD_ROUND_DECIMALS),
            round(c_ctr[AX_Y] - ref_ctr[AX_Y], COORD_ROUND_DECIMALS),
            round(c_ctr[AX_Z] - ref_ctr[AX_Z], COORD_ROUND_DECIMALS),
        )
  return canon_of, offset_of


def _choose_mesh_sizes(part: object, part_key: str) -> Tuple[float, float]:
  """Select adaptive Gmsh C3D10 element bounds balancing accuracy and WebGL payload."""
  if part_key == CHAIN_KEY:
    return CHAIN_MESH_MIN_MM, CHAIN_MESH_MAX_MM
  size_xyz, _ = _bbox_tuple(part)
  min_dim = min(size_xyz)
  diag = math.sqrt(sum(val * val for val in size_xyz))
  vol = float(part.volume)
  if min_dim <= THIN_PART_MIN_DIM_MM:
    return THIN_MESH_MIN_MM, min(THIN_MESH_MAX_MM, max(THIN_MESH_MIN_MM + HALF_F, diag * THIN_DIAG_SCALE))
  if min_dim <= SHELL_PART_MIN_DIM_MM:
    return SHELL_MESH_MIN_MM, min(SHELL_MESH_MAX_MM, max(SHELL_MESH_MIN_MM + HALF_F, diag * SHELL_DIAG_SCALE))
  if vol >= LARGE_BILLET_VOL_MM3 or diag >= LARGE_BILLET_DIAG_MM:
    return LARGE_MESH_MIN_MM, LARGE_MESH_MAX_MM
  if diag >= MEDIUM_PART_DIAG_MM:
    return MED_MESH_MIN_MM, MED_MESH_MAX_MM
  return SMALL_MESH_MIN_MM, SMALL_MESH_MAX_MM


def _normalize_axis(coords: np.ndarray, axis_idx: int) -> np.ndarray:
  """Return [0, 1] normalized node coordinates along one Cartesian axis."""
  c_min = float(np.min(coords[:, axis_idx]))
  c_max = float(np.max(coords[:, axis_idx]))
  span = max(SPAN_EPS_MM, c_max - c_min)
  return (coords[:, axis_idx] - c_min) / span


def _ring_radial_fraction(coords: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
  """Return (short_axis, radial_fraction_0_to_1, in_plane_unit_vectors)."""
  spans = np.max(coords, axis=0) - np.min(coords, axis=0)
  short_ax = int(np.argmin(spans))
  plane_axes = [ax for ax in range(NUM_AXES) if ax != short_ax]
  center = np.mean(coords[:, plane_axes], axis=0)
  delta = coords[:, plane_axes] - center
  radii = np.hypot(delta[:, 0], delta[:, 1])
  r_min = float(np.min(radii))
  r_max = float(np.max(radii))
  r_frac = (radii - r_min) / max(SPAN_EPS_MM, r_max - r_min)
  unit_vec = np.zeros_like(coords)
  safe_r = np.maximum(RADIAL_EPS_MM, radii)
  unit_vec[:, plane_axes[0]] = delta[:, 0] / safe_r
  unit_vec[:, plane_axes[1]] = delta[:, 1] / safe_r
  return short_ax, r_frac, unit_vec


def _mask_z_band(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  z_lo, z_hi, r_lo, r_hi = params
  rxy = np.hypot(coords[:, AX_X], coords[:, AX_Y])
  return (coords[:, AX_Z] >= z_lo) & (coords[:, AX_Z] <= z_hi) & (rxy >= r_lo) & (rxy <= r_hi)


def _mask_y_band_r_crank(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  ay_lo, ay_hi, r_lo, r_hi = params
  ay = np.abs(coords[:, AX_Y])
  rc = np.hypot(coords[:, AX_X], coords[:, AX_Z] - CRANK_AXIS_Z_MM)
  return (ay >= ay_lo) & (ay <= ay_hi) & (rc >= r_lo) & (rc <= r_hi)


def _mask_crank_pin(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  ay_hi, z_lo = params
  return (np.abs(coords[:, AX_Y]) <= ay_hi) & (coords[:, AX_Z] >= z_lo)


def _mask_pivot_xz(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  cx, cz, r_lo, r_hi, ax_lo = params
  rp = np.hypot(coords[:, AX_X] - cx, coords[:, AX_Z] - cz)
  return (rp >= r_lo) & (rp <= r_hi) & (np.abs(coords[:, AX_X]) >= ax_lo)


def _mask_frac_long(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  spans = np.max(coords, axis=0) - np.min(coords, axis=0)
  frac = _normalize_axis(coords, int(np.argmax(spans)))
  return (frac >= params[0]) & (frac <= params[1])


def _mask_frac_long_ends(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  spans = np.max(coords, axis=0) - np.min(coords, axis=0)
  frac = _normalize_axis(coords, int(np.argmax(spans)))
  edge = params[0]
  return (frac <= edge) | (frac >= (UNIT_F - edge))


def _mask_frac_short(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  spans = np.max(coords, axis=0) - np.min(coords, axis=0)
  frac = _normalize_axis(coords, int(np.argmin(spans)))
  return (frac >= params[0]) & (frac <= params[1])


def _mask_frac_axis(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  ax_idx = int(params[0])
  frac = _normalize_axis(coords, ax_idx)
  return (frac >= params[1]) & (frac <= params[2])


def _mask_ring_radial(coords: np.ndarray, params: Tuple[float, ...]) -> np.ndarray:
  _, r_frac, _ = _ring_radial_fraction(coords)
  return (r_frac >= params[0]) & (r_frac <= params[1])


RULE_DISPATCH: Mapping[str, Callable[[np.ndarray, Tuple[float, ...]], np.ndarray]] = {
    "z_band": _mask_z_band,
    "y_band_r_crank": _mask_y_band_r_crank,
    "crank_pin": _mask_crank_pin,
    "pivot_xz": _mask_pivot_xz,
    "frac_long": _mask_frac_long,
    "frac_long_ends": _mask_frac_long_ends,
    "frac_short": _mask_frac_short,
    "frac_axis": _mask_frac_axis,
    "ring_radial": _mask_ring_radial,
}


def _select_boundary_sets(
    node_ids: Sequence[int],
    coords: np.ndarray,
    spec: PartStressSpec,
) -> Tuple[List[int], List[int]]:
  """Resolve disjoint non-degenerate fixed and loaded node sets from surface coordinates."""
  fix_mask = RULE_DISPATCH[spec.fix_rule](coords, spec.fix_params)
  fix_idx = [i for i, keep in enumerate(fix_mask) if keep]

  spans = np.max(coords, axis=0) - np.min(coords, axis=0)
  long_ax = int(np.argmax(spans))
  long_frac = _normalize_axis(coords, long_ax)

  if len(fix_idx) < MIN_FIX_NODES:
    fix_idx = [i for i, val in enumerate(long_frac) if val <= FALLBACK_FIX_FRAC]
  else:
    fix_spans = np.max(coords[fix_idx], axis=0) - np.min(coords[fix_idx], axis=0)
    if sum(1 for val in fix_spans if val >= MIN_FIX_EXTENT_MM) < MIN_LOAD_NODES - 1:
      fix_idx = [i for i, val in enumerate(long_frac) if val <= FALLBACK_FIX_FRAC]

  fix_set = set(fix_idx)
  load_mask = RULE_DISPATCH[spec.load_rule](coords, spec.load_params)
  load_idx = [i for i, keep in enumerate(load_mask) if keep and i not in fix_set]
  if len(load_idx) < MIN_LOAD_NODES:
    load_idx = [i for i, val in enumerate(long_frac) if val >= FALLBACK_LOAD_FRAC and i not in fix_set]
  if len(load_idx) < MIN_LOAD_NODES:
    load_idx = [i for i in range(len(node_ids)) if i not in fix_set]

  return [int(node_ids[i]) for i in fix_idx], [int(node_ids[i]) for i in load_idx]


def _compute_nodal_force_vectors(
    coords: np.ndarray,
    fix_ids: Sequence[int],
    load_ids: Sequence[int],
    nid_to_row: Mapping[int, int],
    spec: PartStressSpec,
) -> List[Tuple[int, float, float, float]]:
  """Compute per-node Cartesian force vectors summing to calibrated physical load."""
  load_rows = np.array([nid_to_row[nid] for nid in load_ids], dtype=int)
  l_coords = coords[load_rows]
  f_per_node = spec.force_n / float(len(load_ids))

  mode_unit: Optional[Tuple[float, float, float]] = {
      "force_x": (UNIT_F, ZERO_F, ZERO_F),
      "force_y": (ZERO_F, UNIT_F, ZERO_F),
      "force_z": (ZERO_F, ZERO_F, UNIT_F),
  }.get(spec.load_mode)

  if mode_unit is not None:
    return [
        (nid, f_per_node * mode_unit[AX_X], f_per_node * mode_unit[AX_Y], f_per_node * mode_unit[AX_Z])
        for nid in load_ids
    ]

  if spec.load_mode == "force_long":
    fix_rows = np.array([nid_to_row[nid] for nid in fix_ids], dtype=int)
    spans = np.max(coords, axis=0) - np.min(coords, axis=0)
    long_ax = int(np.argmax(spans))
    sign = UNIT_F if float(np.mean(l_coords[:, long_ax])) >= float(np.mean(coords[fix_rows, long_ax])) else -UNIT_F
    vec = [ZERO_F, ZERO_F, ZERO_F]
    vec[long_ax] = f_per_node * sign
    return [(nid, vec[AX_X], vec[AX_Y], vec[AX_Z]) for nid in load_ids]

  if spec.load_mode == "radial_xy":
    r = np.maximum(RADIAL_EPS_MM, np.hypot(l_coords[:, AX_X], l_coords[:, AX_Y]))
    return [
        (nid, f_per_node * float(l_coords[idx, AX_X] / r[idx]), f_per_node * float(l_coords[idx, AX_Y] / r[idx]), ZERO_F)
        for idx, nid in enumerate(load_ids)
    ]

  if spec.load_mode == "radial_crank_xz":
    dz = l_coords[:, AX_Z] - CRANK_AXIS_Z_MM
    r = np.maximum(RADIAL_EPS_MM, np.hypot(l_coords[:, AX_X], dz))
    return [
        (nid, f_per_node * float(l_coords[idx, AX_X] / r[idx]), ZERO_F, f_per_node * float(dz[idx] / r[idx]))
        for idx, nid in enumerate(load_ids)
    ]

  _, _, unit_ring = _ring_radial_fraction(coords)
  return [
      (
          nid,
          f_per_node * float(unit_ring[row, AX_X]),
          f_per_node * float(unit_ring[row, AX_Y]),
          f_per_node * float(unit_ring[row, AX_Z]),
      )
      for nid, row in zip(load_ids, load_rows)
  ]


def _write_atlas_ccx_deck(
    job_name: str,
    cases_dir: Path,
    mesh_inp_path: Path,
    mat: MaterialProperties,
    spec: PartStressSpec,
) -> Path:
  """Append CalculiX material, boundary constraints, and nodal loads onto a C3D10 mesh."""
  bmesh = parse_calculix_inp_mesh(mesh_inp_path)
  node_ids = bmesh.unique_nodes
  coords = np.array([bmesh.node_coords[nid] for nid in node_ids], dtype=float)
  nid_to_row = {nid: idx for idx, nid in enumerate(node_ids)}
  fix_ids, load_ids = _select_boundary_sets(node_ids, coords, spec)
  nodal_forces = _compute_nodal_force_vectors(coords, fix_ids, load_ids, nid_to_row, spec)

  deck_path = cases_dir / f"{job_name}.inp"
  cload_lines: List[str] = []
  for nid, fx, fy, fz in nodal_forces:
    if abs(fx) > ZERO_F:
      cload_lines.append(f"{nid}, 1, {fx:.4f}")
    if abs(fy) > ZERO_F:
      cload_lines.append(f"{nid}, 2, {fy:.4f}")
    if abs(fz) > ZERO_F:
      cload_lines.append(f"{nid}, 3, {fz:.4f}")

  sections = [
      "*HEADING",
      f"OpenEngine-1C Atlas C3D10 Stress: {spec.label}",
      mesh_inp_path.read_text(encoding="utf-8").strip(),
      format_nset_cards("NFIX", fix_ids).strip(),
      format_nset_cards("NLOAD", load_ids).strip(),
      f"*MATERIAL, NAME={mat.name}",
      "*ELASTIC",
      f"{mat.youngs_modulus_mpa:.1f}, {mat.poissons_ratio:.2f}",
      "*DENSITY",
      f"{mat.density_tonne_mm3:.3e}",
      f"*SOLID SECTION, ELSET=EALL, MATERIAL={mat.name}",
      "*STEP",
      "*STATIC",
      "*BOUNDARY",
      "NFIX, 1, 3, 0.0",
      "*CLOAD",
      *cload_lines,
      "*NODE FILE",
      "U",
      "*EL FILE",
      "S",
      "*END STEP",
      "",
  ]
  deck_path.write_text("\n".join(line for line in sections if line != ""), encoding="utf-8")
  return deck_path


def _ensure_flagship_solves(base_fea_dir: Path, force_rebuild: bool) -> Dict[str, Tuple[Path, Path, str, MaterialProperties]]:
  """Ensure flagship piston LC-1/LC-2 and connecting-rod LC-3A/LC-3B solves exist."""
  piston_dir = base_fea_dir / "piston"
  if force_rebuild or not (piston_dir / "piston_lc1.frd").exists() or not (piston_dir / "piston_mesh.inp").exists():
    run_piston_fea(config=PistonFeaConfig(peak_pressure_mpa=PISTON_FIRING_PMAX_MPA), output_dir=piston_dir)

  conrod_dir = base_fea_dir / "conrod"
  if force_rebuild or not (conrod_dir / "conrod_lc3a.frd").exists() or not (conrod_dir / "conrod_mesh.inp").exists():
    run_conrod_fea(config=ConrodFeaConfig(), output_dir=conrod_dir)

  return {
      "piston": (piston_dir / "piston_mesh.inp", piston_dir / "piston_lc1.frd", "Piston LC-1 (Firing 78 bar 4032-T6)", ALUMINUM_4032_T6),
      "piston_lc2": (piston_dir / "piston_mesh.inp", piston_dir / "piston_lc2.frd", "Piston LC-2 (Coupled Thermal 4032-T6)", ALUMINUM_4032_T6),
      "connecting_rod": (conrod_dir / "conrod_mesh.inp", conrod_dir / "conrod_lc3a.frd", "One-Piece Con-Rod LC-3A (28.3 kN)", ALUMINUM_7075_T6),
      "connecting_rod_lc3b": (conrod_dir / "conrod_mesh.inp", conrod_dir / "conrod_lc3b.frd", "One-Piece Con-Rod LC-3B (2.54 kN Inertia)", ALUMINUM_7075_T6),
  }


def _transform_mesh_to_world(
    mesh: BoundaryMeshData,
    displacements: Mapping[int, Tuple[float, float, float]],
    canonical_key: str,
    wrist_pin_z0_mm: float,
) -> Tuple[BoundaryMeshData, Dict[int, Tuple[float, float, float]]]:
  """Map local piston and one-piece con-rod coordinates and displacements into master world space."""
  if canonical_key in ("piston", "piston_lc2"):
    z_shift = wrist_pin_z0_mm + PISTON_PIN_WORLD_Z_OFFSET_MM
    coords = {nid: (x, y, z + z_shift) for nid, (x, y, z) in mesh.node_coords.items()}
    return BoundaryMeshData(coords, mesh.boundary_triangles, mesh.unique_nodes), dict(displacements)

  if canonical_key in ("connecting_rod", "connecting_rod_lc3b"):
    coords = {nid: (-z_l, x_l, -y_l + wrist_pin_z0_mm) for nid, (x_l, y_l, z_l) in mesh.node_coords.items()}
    disp = {nid: (-uz_l, ux_l, -uy_l) for nid, (ux_l, uy_l, uz_l) in displacements.items()}
    return BoundaryMeshData(coords, mesh.boundary_triangles, mesh.unique_nodes), disp

  return mesh, dict(displacements)


def _prepare_canonical_atlas_job(
    part_key: str,
    part_geom: object,
    cases_dir: Path,
    force_rebuild: bool,
) -> Tuple[Path, Path, Optional[str]]:
  """Generate cached C3D10 mesh and CalculiX deck for one canonical CAD part."""
  spec = PART_STRESS_SPECS[part_key]
  mat = get_material_properties(material_key_for_dressed_part(part_key))
  mesh_inp = cases_dir / f"{part_key}_mesh.inp"
  job_name = f"{part_key}_atlas"
  deck_inp = cases_dir / f"{job_name}.inp"
  frd_path = cases_dir / f"{job_name}.frd"

  if force_rebuild or not mesh_inp.exists():
    m_min, m_max = _choose_mesh_sizes(part_geom, part_key)
    mesh_structural_part_c3d10(part_geom, mesh_inp, mesh_size_min=m_min, mesh_size_max=m_max)
  if force_rebuild or not deck_inp.exists():
    _write_atlas_ccx_deck(job_name, cases_dir, mesh_inp, mat, spec)
  pending = job_name if (force_rebuild or not frd_path.exists()) else None
  return mesh_inp, frd_path, pending


def _solve_canonical_atlas_jobs(
    canonical_parts: Mapping[str, object],
    cases_dir: Path,
    force_rebuild: bool,
) -> Dict[str, Tuple[Path, Path, FeaNodalResults]]:
  """Mesh and batch-solve every non-flagship canonical part in CalculiX C3D10."""
  pending_jobs: List[str] = []
  job_paths: Dict[str, Tuple[Path, Path]] = {}

  for part_key, part_geom in canonical_parts.items():
    if part_key not in PART_STRESS_SPECS:
      continue
    mesh_inp, frd_path, pending = _prepare_canonical_atlas_job(part_key, part_geom, cases_dir, force_rebuild)
    if pending is not None:
      pending_jobs.append(pending)
    job_paths[part_key] = (mesh_inp, frd_path)

  if pending_jobs:
    run_ccx_batch(pending_jobs, cases_dir)

  return {
      part_key: (mesh_inp, frd_path, parse_frd_results(frd_path))
      for part_key, (mesh_inp, frd_path) in job_paths.items()
  }


def _build_part_record(
    part_key: str,
    canonical_key: str,
    label: str,
    mat: MaterialProperties,
    mesh: BoundaryMeshData,
    fea_res: FeaNodalResults,
    offset_mm: Tuple[float, float, float],
    is_housing: bool,
) -> AtlasPartRecord:
  """Compute stress stats and safety factor across a part's boundary surface."""
  vm_vals = [fea_res.von_mises_stresses.get(nid, ZERO_F) for nid in mesh.unique_nodes]
  max_vm = max(vm_vals) if vm_vals else ZERO_F
  p95_vm = float(np.percentile(vm_vals, P95_PERCENTILE)) if vm_vals else ZERO_F
  sf_yield = mat.yield_strength_mpa / max(MIN_STRESS_EPS_MPA, p95_vm)
  return AtlasPartRecord(
      part_key=part_key,
      canonical_key=canonical_key,
      label=label,
      material=mat.name,
      yield_strength_mpa=round(mat.yield_strength_mpa, STRESS_ROUND_DECIMALS),
      p95_stress_mpa=round(p95_vm, STRESS_ROUND_DECIMALS),
      max_stress_mpa=round(max_vm, STRESS_ROUND_DECIMALS),
      max_displacement_mm=round(fea_res.max_disp_mm, DISP_ROUND_DECIMALS),
      yield_safety_factor=round(sf_yield, SF_ROUND_DECIMALS),
      num_vertices=len(mesh.unique_nodes),
      num_triangles=len(mesh.boundary_triangles),
      offset_mm=offset_mm,
      is_housing=is_housing,
      is_pass=(p95_vm > ZERO_F and sf_yield >= MIN_YIELD_SF_GATE),
  )


def _assembly_definitions(
    dressed_keys: Sequence[str],
    rot_keys: Sequence[str],
    head_keys: Sequence[str],
    case_keys: Sequence[str],
) -> Tuple[Tuple[str, str, Tuple[str, ...]], ...]:
  """Return the 5 multi-body structural assembly definitions and member keys."""
  core_keys = (
      "cylinder_head",
      "cylinder_block",
      "cylinder_liner",
      "crankshaft",
      "connecting_rod",
      "piston",
      "wrist_pin",
      "valve_intake",
      "valve_exhaust",
  )
  barrel_mount_keys = (
      "cylinder_block",
      "cylinder_liner",
      "base_washer_1",
      "base_washer_2",
      "base_washer_3",
      "base_washer_4",
      "base_bolt_1",
      "base_bolt_2",
      "base_bolt_3",
      "base_bolt_4",
  )
  case_barrel_keys = tuple(dict.fromkeys([*case_keys, *barrel_mount_keys]))
  return (
      ("dressed_engine_assembly", "Dressed Engine Assembly", tuple(dressed_keys)),
      ("engine_assembly", "Core Engine Assembly", core_keys),
      ("rotating_assembly", "Rotating & Reciprocating Sub-Assembly", tuple(rot_keys)),
      ("head_assembly", "SOHC Head & Valvetrain Sub-Assembly", tuple(head_keys)),
      ("crankcase_barrel_assembly", "Split Crankcase, Barrel & Sump Sub-Assembly", case_barrel_keys),
  )


def _find_limiting_member(members: Sequence[AtlasPartRecord]) -> AtlasPartRecord:
  """Return the member record with the lowest yield safety factor."""
  limiting = members[0]
  for rec in members[1:]:
    if rec.yield_safety_factor < limiting.yield_safety_factor:
      limiting = rec
  return limiting


def _build_assembly_records(
    specs: Tuple[Tuple[str, str, Tuple[str, ...]], ...],
    part_records: Mapping[str, AtlasPartRecord],
) -> Tuple[AtlasAssemblyRecord, ...]:
  """Aggregate peak stress, P95 stress, and governing safety factor per multi-body assembly."""
  records: List[AtlasAssemblyRecord] = []
  for asm_id, base_title, member_keys in specs:
    members = [part_records[k] for k in member_keys if k in part_records]
    limiting = _find_limiting_member(members)
    max_vm = max(rec.max_stress_mpa for rec in members)
    p95_vm = max(rec.p95_stress_mpa for rec in members)
    records.append(
        AtlasAssemblyRecord(
            assembly_id=asm_id,
            title=f"{base_title} ({len(members)} Solids)",
            part_count=len(members),
            max_stress_mpa=round(max_vm, STRESS_ROUND_DECIMALS),
            p95_stress_mpa=round(p95_vm, STRESS_ROUND_DECIMALS),
            min_yield_safety_factor=round(limiting.yield_safety_factor, SF_ROUND_DECIMALS),
            limiting_part_key=limiting.part_key,
            part_keys=tuple(rec.part_key for rec in members),
            is_pass=all(rec.is_pass for rec in members),
        )
    )
  return tuple(records)


def _gather_solved_atlas_data(
    fea_root: Path,
    force_rebuild: bool,
) -> Tuple[
    Dict[str, str],
    Dict[str, Tuple[float, float, float]],
    Dict[str, Tuple[BoundaryMeshData, FeaNodalResults, MaterialProperties, str, str, bool]],
    Tuple[Tuple[str, str, Tuple[str, ...]], ...],
]:
  """Collect canonical CAD parts, solve/load CalculiX results, and return world-posed meshes."""
  st0 = EngineAssembly().solve_kinematics(ZERO_F)
  wrist_pin_z0 = float(st0.wrist_pin_z_mm)
  dressed_parts = build_dressed_engine_parts(ZERO_F, has_fourth_axis=True, include_fluids=False)
  rot_keys = list(build_rotating_assembly_parts(ZERO_F).keys())
  head_keys = list(build_head_assembly_parts(ZERO_F, has_fourth_axis=True).keys())
  case_keys = list(build_crankcase_assembly_parts(include_oil_pool=False).keys())

  canon_of, offset_of = _resolve_canonical_map(dressed_parts)
  unique_canon_keys = tuple(dict.fromkeys(canon_of.values()))
  canonical_geoms = {k: dressed_parts[k] for k in unique_canon_keys if k in PART_STRESS_SPECS}

  cases_dir = fea_root / "atlas" / "cases"
  cases_dir.mkdir(parents=True, exist_ok=True)
  flagships = _ensure_flagship_solves(fea_root, force_rebuild=force_rebuild)
  batch_solved = _solve_canonical_atlas_jobs(canonical_geoms, cases_dir, force_rebuild=force_rebuild)

  canon_data: Dict[str, Tuple[BoundaryMeshData, FeaNodalResults, MaterialProperties, str, str, bool]] = {}
  for ckey, (mesh_inp, frd_path, label, mat) in flagships.items():
    raw_mesh = parse_calculix_inp_mesh(mesh_inp)
    raw_res = parse_frd_results(frd_path)
    world_mesh, world_disp = _transform_mesh_to_world(raw_mesh, raw_res.displacements, ckey, wrist_pin_z0)
    world_res = FeaNodalResults(
        num_nodes=raw_res.num_nodes,
        displacements=world_disp,
        stresses=raw_res.stresses,
        von_mises_stresses=raw_res.von_mises_stresses,
        temperatures=raw_res.temperatures,
        max_disp_mm=raw_res.max_disp_mm,
        max_radial_growth_mm=raw_res.max_radial_growth_mm,
        max_von_mises_mpa=raw_res.max_von_mises_mpa,
        p95_von_mises_mpa=raw_res.p95_von_mises_mpa,
        mean_von_mises_mpa=raw_res.mean_von_mises_mpa,
        max_principal_stress_mpa=raw_res.max_principal_stress_mpa,
    )
    canon_data[ckey] = (world_mesh, world_res, mat, label, "primary", False)

  for ckey, (mesh_inp, _, res) in batch_solved.items():
    spec = PART_STRESS_SPECS[ckey]
    mat = get_material_properties(material_key_for_dressed_part(ckey))
    bmesh = parse_calculix_inp_mesh(mesh_inp)
    canon_data[ckey] = (bmesh, res, mat, spec.label, spec.group, spec.is_housing)

  asm_specs = _assembly_definitions(list(dressed_parts.keys()), rot_keys, head_keys, case_keys)
  return canon_of, offset_of, canon_data, asm_specs


def _summarize_gathered_atlas(
    canon_of: Mapping[str, str],
    offset_of: Mapping[str, Tuple[float, float, float]],
    canon_data: Mapping[str, Tuple[BoundaryMeshData, FeaNodalResults, MaterialProperties, str, str, bool]],
    asm_specs: Tuple[Tuple[str, str, Tuple[str, ...]], ...],
    elapsed_s: float,
    out_path: Path,
) -> Tuple[FeaAtlasSummaryResult, Dict[str, AtlasPartRecord]]:
  """Build part and assembly records from solved canonical meshes and persist summary JSON."""
  out_path.mkdir(parents=True, exist_ok=True)
  canon_records: Dict[str, AtlasPartRecord] = {
      ckey: _build_part_record(ckey, ckey, label, mat, bmesh, res, (ZERO_F, ZERO_F, ZERO_F), is_housing)
      for ckey, (bmesh, res, mat, label, _, is_housing) in canon_data.items()
  }

  part_records: Dict[str, AtlasPartRecord] = {}
  for pkey, ckey in canon_of.items():
    ref = canon_records[ckey]
    part_records[pkey] = AtlasPartRecord(
        part_key=pkey,
        canonical_key=ckey,
        label=pkey if pkey != ckey else ref.label,
        material=ref.material,
        yield_strength_mpa=ref.yield_strength_mpa,
        p95_stress_mpa=ref.p95_stress_mpa,
        max_stress_mpa=ref.max_stress_mpa,
        max_displacement_mm=ref.max_displacement_mm,
        yield_safety_factor=ref.yield_safety_factor,
        num_vertices=ref.num_vertices,
        num_triangles=ref.num_triangles,
        offset_mm=offset_of.get(pkey, (ZERO_F, ZERO_F, ZERO_F)),
        is_housing=(pkey in STRUCTURAL_HOUSING_KEYS),
        is_pass=ref.is_pass,
    )
  part_records["connecting_rod"] = canon_records["connecting_rod"]

  assemblies = _build_assembly_records(asm_specs, part_records)
  dressed_members = [part_records[k] for k in canon_of]
  limiting_part = _find_limiting_member(dressed_members)
  summary = FeaAtlasSummaryResult(
      total_solids=len(canon_of),
      unique_meshes=len(canon_data),
      min_yield_safety_factor=round(limiting_part.yield_safety_factor, SF_ROUND_DECIMALS),
      limiting_part_key=limiting_part.part_key,
      all_passed=all(r.is_pass for r in part_records.values()) and all(a.is_pass for a in assemblies),
      total_solve_time_s=round(elapsed_s, TIME_ROUND_DECIMALS),
      assemblies=assemblies,
      parts=part_records,
  )
  (out_path / "fea_atlas_results.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
  return summary, canon_records


def run_fea_atlas(
    output_dir: Union[str, Path] = "outputs/fea/atlas",
    force_rebuild: bool = False,
) -> FeaAtlasSummaryResult:
  """Solve C3D10 FEA across all canonical engine solids and aggregate 5 assemblies."""
  t0 = time.perf_counter()
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)
  canon_of, offset_of, canon_data, asm_specs = _gather_solved_atlas_data(out_path.parent, force_rebuild=force_rebuild)
  summary, _ = _summarize_gathered_atlas(
      canon_of, offset_of, canon_data, asm_specs, time.perf_counter() - t0, out_path
  )
  return summary


def _pack_mesh_base64(mesh: BoundaryMeshData, res: FeaNodalResults) -> Dict[str, object]:
  """Encode a canonical surface mesh + von Mises field into compact Float32/Uint32 Base64."""
  v_count = len(mesh.unique_nodes)
  node_to_idx = {nid: idx for idx, nid in enumerate(mesh.unique_nodes)}
  scale = min(DEFAULT_VISUAL_DISP_SCALE, VISUAL_MAX_DISP_MM / max(VISUAL_DISP_FLOOR_MM, res.max_disp_mm))

  coords = np.empty((v_count, NUM_AXES), dtype="<f4")
  vm_arr = np.empty(v_count, dtype="<f4")
  for idx, nid in enumerate(mesh.unique_nodes):
    x0, y0, z0 = mesh.node_coords[nid]
    ux, uy, uz = res.displacements.get(nid, (ZERO_F, ZERO_F, ZERO_F))
    coords[idx, AX_X] = x0 + (scale * ux)
    coords[idx, AX_Y] = y0 + (scale * uy)
    coords[idx, AX_Z] = z0 + (scale * uz)
    vm_arr[idx] = res.von_mises_stresses.get(nid, ZERO_F)

  tri_idx = np.array([[node_to_idx[a], node_to_idx[b], node_to_idx[c]] for a, b, c in mesh.boundary_triangles], dtype="<u4")
  return {
      "v_count": v_count,
      "t_count": len(mesh.boundary_triangles),
      "pos_b64": base64.b64encode(coords.tobytes()).decode("ascii"),
      "vm_b64": base64.b64encode(vm_arr.tobytes()).decode("ascii"),
      "idx_b64": base64.b64encode(tri_idx.tobytes()).decode("ascii"),
  }


def export_fea_atlas_to_webgl(
    fea_root_dir: Union[str, Path] = "outputs/fea",
    output_json_path: Union[str, Path] = "outputs/reports/models/fea_stress_atlas.json",
    force_rebuild: bool = False,
) -> Dict[str, object]:
  """Export packed binary-Base64 C3D10 stress atlas covering all parts and 5 assemblies."""
  t0 = time.perf_counter()
  fea_root = Path(fea_root_dir)
  atlas_dir = fea_root / "atlas"
  atlas_dir.mkdir(parents=True, exist_ok=True)
  canon_of, offset_of, canon_data, asm_specs = _gather_solved_atlas_data(fea_root, force_rebuild=force_rebuild)
  summary, canon_records = _summarize_gathered_atlas(
      canon_of, offset_of, canon_data, asm_specs, time.perf_counter() - t0, atlas_dir
  )

  meshes_payload = {
      ckey: _pack_mesh_base64(bmesh, res)
      for ckey, (bmesh, res, _, _, _, _) in canon_data.items()
  }

  parts_meta: Dict[str, object] = {}
  for ckey, (_, _, _, label, group, is_housing) in canon_data.items():
    rec = canon_records[ckey]
    parts_meta[ckey] = {
        "key": ckey,
        "mesh_key": ckey,
        "label": label,
        "group": group,
        "material": rec.material,
        "yield_strength_mpa": rec.yield_strength_mpa,
        "p95_stress_mpa": rec.p95_stress_mpa,
        "max_stress_mpa": rec.max_stress_mpa,
        "max_displacement_mm": rec.max_displacement_mm,
        "yield_safety_factor": rec.yield_safety_factor,
        "is_housing": is_housing,
    }

  catalog_models: List[Dict[str, object]] = []
  for asm_rec in summary.assemblies:
    instances = [
        {
            "part_key": pkey,
            "mesh_key": (pkey if pkey in ("connecting_rod", "piston") else canon_of[pkey]),
            "offset_mm": list(offset_of.get(pkey, (ZERO_F, ZERO_F, ZERO_F))),
            "is_housing": (pkey in STRUCTURAL_HOUSING_KEYS),
        }
        for pkey in asm_rec.part_keys
    ]
    catalog_models.append({
        "id": asm_rec.assembly_id,
        "title": asm_rec.title,
        "category": "assemblies",
        "part_count": asm_rec.part_count,
        "max_stress_mpa": asm_rec.max_stress_mpa,
        "p95_stress_mpa": asm_rec.p95_stress_mpa,
        "yield_safety_factor": asm_rec.min_yield_safety_factor,
        "limiting_part": asm_rec.limiting_part_key,
        "instances": instances,
    })

  ordered_single_keys = [
      "piston",
      "piston_lc2",
      "connecting_rod",
      "connecting_rod_lc3b",
      *[k for k in canon_data if k not in ("piston", "piston_lc2", "connecting_rod", "connecting_rod_lc3b")],
  ]
  for ckey in ordered_single_keys:
    meta = parts_meta[ckey]
    catalog_models.append({
        "id": f"part_{ckey}",
        "title": meta["label"],
        "category": str(meta["group"]),
        "part_count": 1,
        "max_stress_mpa": meta["max_stress_mpa"],
        "p95_stress_mpa": meta["p95_stress_mpa"],
        "yield_safety_factor": meta["yield_safety_factor"],
        "yield_strength_mpa": meta["yield_strength_mpa"],
        "material": meta["material"],
        "instances": [{"part_key": ckey, "mesh_key": ckey, "offset_mm": [ZERO_F, ZERO_F, ZERO_F], "is_housing": bool(meta["is_housing"])}],
    })

  atlas_payload = {
      "summary": {
          "total_solids": summary.total_solids,
          "unique_meshes": summary.unique_meshes,
          "min_yield_safety_factor": summary.min_yield_safety_factor,
          "limiting_part_key": summary.limiting_part_key,
          "all_passed": summary.all_passed,
      },
      "parts": parts_meta,
      "models": catalog_models,
      "meshes": meshes_payload,
  }
  out_json = Path(output_json_path)
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(atlas_payload), encoding="utf-8")
  return dict(atlas_payload["summary"])
