"""Tests for live B-Rep C500 DFM checks, 2D setup sheets, and CAD twin sync."""

import json
import math
from pathlib import Path
from typing import Set

import pytest
from typer.testing import CliRunner

from openengine.cad.connecting_rod import DEFAULT_CENTER_TO_CENTER_M, DIPPER_WIDTH_X_MM
from openengine.cad.crankshaft import (
    CRANK_SPROCKET_TEETH,
    DEFAULT_COUNTERWEIGHT_RADIUS_M,
    DEFAULT_MAIN_JOURNAL_DIAM_M,
    DEFAULT_STROKE_M,
    DEFAULT_WEB_THICKNESS_M,
)
from openengine.cad.cylinder_block import (
    DEFAULT_BARREL_WALL_M,
    DEFAULT_BLOCK_HEIGHT_M,
    DEFAULT_BORE_M,
    DEFAULT_FIN_DEPTH_Y_M,
    DEFAULT_FIN_PITCH_M,
    DEFAULT_FIN_THICKNESS_M,
    DEFAULT_FIN_WIDTH_X_M,
    DEFAULT_LINER_WALL_M,
    DEFAULT_NUM_FINS,
    HEAD_BOLT_XY_MM,
    THERMAL_BUDGET_W,
)
from openengine.cad.twin_metrics import get_cad_dipper_tip_radius_mm
from openengine.cfd.cht_cooling import ChtCoolingConfig
from openengine.cfd.closed_loop import ClosedLoopConfig
from openengine.cfd.combustion import CombustionConfig
from openengine.cli import app
from openengine.dfm.c500_check import extract_full_twin_parameters, run_c500_dfm_check
from openengine.dfm.drawings import export_c500_setup_sheets
from openengine.dynamics.flywheel import FlywheelDimensions
from openengine.dynamics.slider_crank import SliderCrankDynamics
from openengine.dynamics.sohc_cam import ValvetrainMassSpring
from openengine.fea.calculix import ALUMINUM_6061_T6, STEEL_1144_STRESSPROOF
from openengine.fea.connecting_rod import ConrodFeaConfig
from openengine.fea.head_gasket import HeadGasketFeaConfig
from openengine.lubrication.friction import (
    CRANK_WEB_RADIUS_M,
    CRANK_WEB_TOTAL_WIDTH_M,
    SEAL_SHAFT_DIAM_M,
)
from openengine.lubrication.splash import SplashSystemParams
from openengine.thermo.cycle import EngineGeometry

EXPECTED_AUDIT_COUNT: int = 10
EXPECTED_SHEET_COUNT: int = 10
EXPECTED_FALLBACK_COUNT: int = 5
EXPECTED_TWIN_PARTS: int = 116
MIN_CYL_RADIUS_MM: float = 1.6
CR_MIN: float = 8.9
CR_MAX: float = 9.1
MIN_DIPPER_IMMERSION_MM: float = 4.0
BAL_MIN: float = 0.48
BAL_MAX: float = 0.54
HALF_FACTOR: float = 0.5
PAIR_FACTOR: float = 2.0
MM_PER_M: float = 1000.0
W_MM_K_TO_W_M_K: float = 1000.0
TONNE_MM3_TO_KG_M3: float = 1.0e12
ROUND_FIN_DECIMALS: int = 4
SYNC_REL_TOL: float = 1.0e-6
THREE_AXIS_FALLBACK_PARTS: Set[str] = {
    "poppet_valves",
    "one_piece_camshaft",
    "valve_guides_and_seats",
    "three_ring_piston",
    "trigger_wheel_and_retainers",
}


def test_c500_live_brep_dfm_check_four_axis(tmp_path: Path):
  """Verify live B-Rep audit passes on 4-axis C500 and round-trips JSON twin data."""
  report = run_c500_dfm_check(has_fourth_axis=True, include_twin_params=True)

  assert report.all_passed is True
  assert report.fallback_count == 0
  assert len(report.part_audits) == EXPECTED_AUDIT_COUNT
  assert all(item.is_valid and item.watertight for item in report.part_audits)
  assert all(item.min_cylindrical_radius_mm >= MIN_CYL_RADIUS_MM for item in report.part_audits)

  twin = report.twin_params
  assert twin is not None
  assert twin.total_part_count == EXPECTED_TWIN_PARTS
  assert CR_MIN <= twin.compression_ratio <= CR_MAX
  assert twin.bdc_dipper_immersion_mm >= MIN_DIPPER_IMMERSION_MM
  assert BAL_MIN <= twin.effective_balance_factor <= BAL_MAX

  report_path = report.to_json(tmp_path / "report.json")
  assert report_path.exists()
  loaded = json.loads(report_path.read_text())
  assert loaded["all_passed"] is True
  assert loaded["twin_params"]["total_part_count"] == EXPECTED_TWIN_PARTS


def test_c500_live_brep_dfm_check_three_axis_fallback():
  """Verify 3-axis fallback triggers only on the five rotary-dependent parts."""
  report = run_c500_dfm_check(has_fourth_axis=False, include_twin_params=False)

  assert report.all_passed is True
  assert report.fallback_count == EXPECTED_FALLBACK_COUNT
  assert set(report.fallback_switches) == THREE_AXIS_FALLBACK_PARTS


def test_c500_setup_sheet_svg_dxf_export(tmp_path: Path):
  """Verify 2D orthographic SVG and DXF setup sheet export and workshop callouts."""
  out_dir = tmp_path / "sheets"
  manifest = export_c500_setup_sheets(output_dir=out_dir, has_fourth_axis=True)

  assert len(manifest.sheets) == EXPECTED_SHEET_COUNT
  assert (out_dir / "setup_sheets_manifest.json").exists()

  svg_files = list(out_dir.glob("*.svg"))
  dxf_files = list(out_dir.glob("*.dxf"))
  assert len(svg_files) == EXPECTED_SHEET_COUNT
  assert len(dxf_files) == EXPECTED_SHEET_COUNT
  assert all(f.stat().st_size > 0 for f in svg_files + dxf_files)

  head_svg = (out_dir / "cylinder_head_setup.svg").read_text()
  assert "M8x1.25" in head_svg
  assert "H7" in head_svg
  assert "G54" in head_svg


def test_cli_dfm_check_four_and_three_axis(tmp_path: Path):
  """Verify openengine dfm-check CLI succeeds for both 4-axis and 3-axis fallback."""
  runner = CliRunner()

  res_4ax = runner.invoke(
      app,
      ["dfm-check", "--no-drawings", "--output-dir", str(tmp_path / "dfm_4ax")],
  )
  assert res_4ax.exit_code == 0, res_4ax.output
  assert (tmp_path / "dfm_4ax" / "c500_dfm_report.json").exists()

  res_3ax = runner.invoke(
      app,
      ["dfm-check", "--no-fourth-axis", "--no-drawings", "--output-dir", str(tmp_path / "dfm_3ax")],
  )
  assert res_3ax.exit_code == 0, res_3ax.output
  loaded_3ax = json.loads((tmp_path / "dfm_3ax" / "c500_dfm_report.json").read_text())
  assert loaded_3ax["has_fourth_axis"] is False
  assert loaded_3ax["fallback_count"] == EXPECTED_FALLBACK_COUNT


def test_simulation_defaults_sync_with_live_cad_twin() -> None:
  """Verify simulation and CLI default configs match live B-Rep CAD twin outputs."""
  twin = extract_full_twin_parameters(has_fourth_axis=True)
  geom = EngineGeometry(
      bore=DEFAULT_BORE_M,
      stroke=DEFAULT_STROKE_M,
      con_rod_length=DEFAULT_CENTER_TO_CENTER_M,
      compression_ratio=twin.compression_ratio,
  )

  slider = SliderCrankDynamics(geom=geom)
  assert slider.m_recip == pytest.approx(twin.reciprocating_mass_kg, rel=SYNC_REL_TOL)
  assert slider.m_rot == pytest.approx(twin.rotating_mass_kg, rel=SYNC_REL_TOL)
  assert slider.r_fw == pytest.approx(twin.flywheel_outer_radius_m, rel=SYNC_REL_TOL)

  fw = FlywheelDimensions()
  assert fw.outer_radius_m == pytest.approx(twin.flywheel_outer_radius_m, rel=SYNC_REL_TOL)
  assert fw.inner_bore_radius_m == pytest.approx(twin.flywheel_hub_radius_m, rel=SYNC_REL_TOL)
  assert fw.density_kg_m3 == pytest.approx(
      STEEL_1144_STRESSPROOF.density_tonne_mm3 * TONNE_MM3_TO_KG_M3,
      rel=SYNC_REL_TOL,
  )

  vt = ValvetrainMassSpring()
  assert vt.valve_mass_kg == pytest.approx(twin.valve_intake_mass_kg, rel=SYNC_REL_TOL)
  assert vt.retainer_mass_kg == pytest.approx(twin.retainer_mass_kg, rel=SYNC_REL_TOL)

  conrod = ConrodFeaConfig()
  assert conrod.center_to_center_m == pytest.approx(DEFAULT_CENTER_TO_CENTER_M, rel=SYNC_REL_TOL)
  assert conrod.bore_m == pytest.approx(DEFAULT_BORE_M, rel=SYNC_REL_TOL)
  assert conrod.stroke_m == pytest.approx(DEFAULT_STROKE_M, rel=SYNC_REL_TOL)
  assert conrod.reciprocating_mass_kg == pytest.approx(twin.reciprocating_mass_kg, rel=SYNC_REL_TOL)

  cht = ChtCoolingConfig()
  expected_barrel_r = HALF_FACTOR * DEFAULT_BORE_M + DEFAULT_LINER_WALL_M + DEFAULT_BARREL_WALL_M
  expected_fin_r = round(
      math.sqrt((DEFAULT_FIN_WIDTH_X_M * DEFAULT_FIN_DEPTH_Y_M) / math.pi),
      ROUND_FIN_DECIMALS,
  )
  assert cht.bore_m == pytest.approx(DEFAULT_BORE_M, rel=SYNC_REL_TOL)
  assert cht.barrel_outer_radius_m == pytest.approx(expected_barrel_r, rel=SYNC_REL_TOL)
  assert cht.fin_outer_radius_m == pytest.approx(expected_fin_r, rel=SYNC_REL_TOL)
  assert cht.block_height_m == pytest.approx(DEFAULT_BLOCK_HEIGHT_M, rel=SYNC_REL_TOL)
  assert cht.num_fins == DEFAULT_NUM_FINS
  assert cht.fin_thickness_m == pytest.approx(DEFAULT_FIN_THICKNESS_M, rel=SYNC_REL_TOL)
  assert cht.fin_pitch_m == pytest.approx(DEFAULT_FIN_PITCH_M, rel=SYNC_REL_TOL)
  assert cht.heat_budget_w == pytest.approx(THERMAL_BUDGET_W, rel=SYNC_REL_TOL)
  assert cht.thermal_conductivity == pytest.approx(
      ALUMINUM_6061_T6.thermal_conductivity_w_mm_k * W_MM_K_TO_W_M_K,
      rel=SYNC_REL_TOL,
  )

  for cfg in (CombustionConfig(), ClosedLoopConfig()):
    assert cfg.bore_m == pytest.approx(DEFAULT_BORE_M, rel=SYNC_REL_TOL)
    assert cfg.stroke_m == pytest.approx(DEFAULT_STROKE_M, rel=SYNC_REL_TOL)
    assert cfg.con_rod_length_m == pytest.approx(DEFAULT_CENTER_TO_CENTER_M, rel=SYNC_REL_TOL)
    assert cfg.compression_ratio == pytest.approx(twin.compression_ratio, rel=SYNC_REL_TOL)

  gasket = HeadGasketFeaConfig()
  assert gasket.num_bolts == len(HEAD_BOLT_XY_MM)
  assert gasket.bore_m == pytest.approx(DEFAULT_BORE_M, rel=SYNC_REL_TOL)
  assert gasket.fire_ring_inner_radius_mm == pytest.approx(
      HALF_FACTOR * DEFAULT_BORE_M * MM_PER_M,
      rel=SYNC_REL_TOL,
  )

  splash = SplashSystemParams()
  assert splash.sump_volume_l == pytest.approx(twin.oil_volume_liters, rel=SYNC_REL_TOL)
  assert splash.dipper_immersion_mm == pytest.approx(twin.bdc_dipper_immersion_mm, rel=SYNC_REL_TOL)
  assert splash.dipper_width_mm == pytest.approx(DIPPER_WIDTH_X_MM, rel=SYNC_REL_TOL)
  assert splash.dipper_tip_radius_mm == pytest.approx(get_cad_dipper_tip_radius_mm(), rel=SYNC_REL_TOL)
  assert splash.crank_sprocket_teeth == CRANK_SPROCKET_TEETH
  assert SEAL_SHAFT_DIAM_M == pytest.approx(DEFAULT_MAIN_JOURNAL_DIAM_M, rel=SYNC_REL_TOL)
  assert CRANK_WEB_RADIUS_M == pytest.approx(DEFAULT_COUNTERWEIGHT_RADIUS_M, rel=SYNC_REL_TOL)
  assert CRANK_WEB_TOTAL_WIDTH_M == pytest.approx(PAIR_FACTOR * DEFAULT_WEB_THICKNESS_M, rel=SYNC_REL_TOL)
