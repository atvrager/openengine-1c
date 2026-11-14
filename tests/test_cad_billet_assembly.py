"""Comprehensive verification suite for Phase 4 1:1 C500 billet + offline COTS CAD twin."""

import ast
from pathlib import Path
import socket

from build123d import Box, Part, export_step

from openengine.cad.assemblies import (
    build_dressed_engine_layer_groups,
    build_dressed_engine_parts,
    verify_all_c500_fixtures,
    verify_crankcase_assembly,
    verify_dressed_engine,
    verify_head_assembly,
    verify_rotating_assembly,
)
from openengine.cad.cots_parts import (
    generate_ball_bearing_6206,
    generate_breather_vent_npt,
    generate_carburetor_vm26,
    generate_dowel_pin,
    generate_external_circlip_16mm,
    generate_flat_washer,
    generate_hall_pickup_and_cdi,
    generate_magnetic_drain_plug_m12,
    generate_needle_bearing_nks28,
    generate_oil_fill_plug_m20,
    generate_piston_ring_pack_68mm,
    generate_roller_chain_25_loop,
    generate_shcs,
    generate_sight_glass_m16,
    generate_spark_plug_cmr7h,
    generate_sprocket_25,
    generate_sump_perimeter_oring,
    generate_valve_spring_9573k35,
    generate_viton_oring,
    generate_viton_shaft_seal_30x47x7,
    generate_wrench_clearance_zones,
    generate_wrist_pin_16x45,
    load_or_generate_cots_part,
    resolve_cots_sku_catalog,
)
from openengine.cad.covers_and_aux import (
    generate_cam_cover,
    generate_fan_shroud,
    generate_flywheel,
    generate_intake_manifold_adapter,
    generate_right_timing_cover,
    generate_trigger_wheel,
)
from openengine.cad.crankcase import (
    generate_crankcase_left,
    generate_crankcase_right,
    generate_sump_oil_pool,
    generate_sump_pan,
)
from openengine.cad.valvetrain_hardware import (
    generate_cam_tower,
    generate_rocker_arm,
    generate_rocker_shaft,
    generate_sohc_camshaft,
    generate_sohc_camshaft_parts,
    generate_spring_retainer,
    generate_valve_guide,
    generate_valve_seat,
)
from openengine.reporting.dashboard import generate_interactive_dashboard
from openengine.reporting.gltf_export import export_cad_models_to_gltf

REPO_ROOT = Path(__file__).resolve().parents[1]
CAD_SRC_DIR = REPO_ROOT / "src" / "openengine" / "cad"

FORBIDDEN_NETWORK_MODULES = frozenset({
    "urllib",
    "requests",
    "httpx",
    "http",
    "socket",
    "aiohttp",
    "ftplib",
    "websocket",
    "playwright",
})


def test_cad_package_strictly_offline(monkeypatch):
  """Verify src/openengine/cad imports no network modules and runs cleanly without sockets."""
  py_files = sorted(CAD_SRC_DIR.rglob("*.py"))
  assert len(py_files) >= 18

  for py_file in py_files:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          root_mod = alias.name.split(".")[0]
          assert root_mod not in FORBIDDEN_NETWORK_MODULES, f"{py_file}: imports {alias.name}"
      elif isinstance(node, ast.ImportFrom) and node.module:
        root_mod = node.module.split(".")[0]
        assert root_mod not in FORBIDDEN_NETWORK_MODULES, f"{py_file}: imports from {node.module}"

  def _blocked_socket(*args, **kwargs):
    raise AssertionError(f"Network socket access is forbidden in CAD generation: {args} {kwargs}")

  monkeypatch.setattr(socket, "socket", _blocked_socket)
  catalog = resolve_cots_sku_catalog()
  assert len(catalog) == 41


def test_cots_local_step_override(tmp_path):
  """Verify local vendor STEP files override parametric COTS fallbacks cleanly."""
  step_dir = tmp_path / "cots_steps"
  step_dir.mkdir(parents=True, exist_ok=True)
  sentinel = Box(12.3, 14.5, 16.7)
  export_step(sentinel, str(step_dir / "91290A326.step"))

  loaded = load_or_generate_cots_part(
      sku="91290A326",
      generator_fn=lambda: Box(1.0, 1.0, 1.0),
      step_dir=step_dir,
  )
  assert loaded.is_valid
  expected_vol = 12.3 * 14.5 * 16.7
  assert abs(loaded.volume - expected_vol) < 1.0

  fallback = load_or_generate_cots_part(
      sku="91290A326",
      generator_fn=lambda: Box(5.0, 6.0, 7.0),
      step_dir=tmp_path / "missing_dir",
  )
  assert fallback.is_valid
  assert abs(fallback.volume - (5.0 * 6.0 * 7.0)) < 1.0


def test_billet_and_cots_solid_validity():
  """Verify every Phase 4 billet, cover, valvetrain, fluid, and COTS generator yields valid B-Reps."""
  single_parts: list[Part] = [
      generate_crankcase_left(),
      generate_crankcase_right(),
      generate_sump_pan(),
      generate_sump_oil_pool(),
      generate_cam_tower(),
      generate_sohc_camshaft(has_fourth_axis=True),
      generate_rocker_shaft(),
      generate_rocker_arm(is_intake=True),
      generate_rocker_arm(is_intake=False),
      generate_valve_guide(),
      generate_valve_seat(is_intake=True),
      generate_valve_seat(is_intake=False),
      generate_spring_retainer(),
      generate_cam_cover(),
      generate_right_timing_cover(),
      generate_flywheel(),
      generate_trigger_wheel(),
      generate_intake_manifold_adapter(),
      generate_fan_shroud(),
      generate_shcs(),
      generate_flat_washer(),
      generate_dowel_pin(),
      generate_external_circlip_16mm(),
      generate_viton_oring(),
      generate_sump_perimeter_oring(),
      generate_ball_bearing_6206(),
      generate_viton_shaft_seal_30x47x7(),
      generate_needle_bearing_nks28(),
      generate_sight_glass_m16(),
      generate_oil_fill_plug_m20(),
      generate_magnetic_drain_plug_m12(),
      generate_breather_vent_npt(),
      generate_sprocket_25(),
      generate_roller_chain_25_loop(),
      generate_valve_spring_9573k35(),
      generate_wrist_pin_16x45(),
      generate_spark_plug_cmr7h(),
      generate_carburetor_vm26(),
  ]
  for part in single_parts:
    assert part.is_valid
    assert part.volume > 0.0

  four_axis_cam = generate_sohc_camshaft_parts(has_fourth_axis=True)
  three_axis_cam = generate_sohc_camshaft_parts(has_fourth_axis=False)
  assert len(four_axis_cam) == 1
  assert len(three_axis_cam) == 3

  part_dicts = [
      four_axis_cam,
      three_axis_cam,
      generate_piston_ring_pack_68mm(),
      generate_hall_pickup_and_cdi(),
      generate_wrench_clearance_zones(),
  ]
  for mapping in part_dicts:
    assert len(mapping) >= 1
    for part in mapping.values():
      assert part.is_valid
      assert part.volume > 0.0


def test_subassembly_verifiers():
  """Verify rotating, head, and crankcase sub-assembly verifiers report zero collisions and pass specs."""
  rot = verify_rotating_assembly(crank_angle_deg=0.0)
  assert rot.is_valid, f"Rotating assembly failed: {rot}"
  assert not rot.collision_pairs
  assert rot.flywheel_mass_kg > 1.0
  assert 0.40 <= rot.effective_balance_factor <= 0.60

  head = verify_head_assembly(crank_angle_deg=0.0, has_fourth_axis=True)
  assert head.is_valid, f"Head assembly failed: {head}"
  assert not head.collision_pairs
  assert 8.5 <= head.compression_ratio <= 9.5
  assert head.dome_volume_cm3 > 19.0

  case = verify_crankcase_assembly()
  assert case.is_valid, f"Crankcase assembly failed: {case}"
  assert not case.collision_pairs
  assert 0.60 <= case.oil_volume_liters <= 0.70
  assert case.bdc_dipper_immersion_mm >= 5.0
  assert case.bdc_dipper_submerged_vol_mm3 > 0.0
  assert case.max_solid_overlap_mm3 <= 1.0


def test_dressed_engine_static_kinematic_and_wrench_checks():
  """Verify the complete dressed twin passes part count, layer groups, static/kinematic, and wrench checks."""
  parts = build_dressed_engine_parts(crank_angle_deg=0.0, has_fourth_axis=True, include_fluids=True)
  assert len(parts) >= 50

  layers = build_dressed_engine_layer_groups(crank_angle_deg=0.0, has_fourth_axis=True)
  expected_layers = {
      "crankcase_group",
      "valvetrain_group",
      "covers_and_aux_group",
      "cots_hardware_group",
      "oil_pool_group",
  }
  assert set(layers.keys()) == expected_layers
  for layer_name, compound in layers.items():
    assert compound.is_valid, f"Layer compound {layer_name} invalid"

  report = verify_dressed_engine(sweep_step_deg=60.0, has_fourth_axis=True)
  assert report.is_valid, f"Dressed engine verification failed: {report}"
  assert report.total_part_count >= 50
  assert report.dry_mass_kg > 5.0
  assert report.wet_mass_kg > report.dry_mass_kg
  assert not report.static_solid_collisions
  assert not report.kinematic_sweep_collisions
  assert not report.wrench_zone_collisions


def test_c500_fixtures_four_axis_and_three_axis():
  """Verify NestWorks C500 workpiece envelopes pass in both 4-axis and 3-axis fallback configurations."""
  rep_4ax = verify_all_c500_fixtures(has_fourth_axis=True)
  assert rep_4ax.all_passed, f"4-axis C500 fixture verification failed: {rep_4ax}"
  assert rep_4ax.fallback_count == 0
  assert len(rep_4ax.items) > 0

  rep_3ax = verify_all_c500_fixtures(has_fourth_axis=False)
  assert rep_3ax.all_passed, f"3-axis C500 fixture verification failed: {rep_3ax}"
  assert rep_3ax.fallback_count >= 1
  assert len(rep_3ax.items) >= len(rep_4ax.items)


def test_gltf_layers_and_dashboard_toggles(tmp_path):
  """Verify GLB layer exports and dashboard HTML toggle buttons are generated cleanly."""
  models_dir = tmp_path / "models"
  exported = export_cad_models_to_gltf(
      output_dir=models_dir,
      linear_deflection=0.25,
      angular_deflection=0.5,
  )
  expected_glb_keys = [
      "crankcase_group",
      "valvetrain_group",
      "covers_and_aux_group",
      "cots_hardware_group",
      "oil_pool_group",
      "engine_assembly",
      "dressed_engine_assembly",
  ]
  for key in expected_glb_keys:
    assert key in exported
    glb_path = models_dir / f"{key}.glb"
    assert glb_path.exists()
    assert glb_path.stat().st_size > 0

  html_path = Path(
      generate_interactive_dashboard(
          output_html_path=tmp_path / "engineering_dashboard.html",
          models_dir=models_dir,
      )
  )
  assert html_path.exists()
  html_text = html_path.read_text(encoding="utf-8")
  for btn_id in (
      "layer-case-btn",
      "layer-valvetrain-btn",
      "layer-covers-btn",
      "layer-cots-btn",
      "layer-oil-btn",
  ):
    assert btn_id in html_text
