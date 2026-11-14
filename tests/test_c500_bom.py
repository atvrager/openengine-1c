"""Tests for NestWorks C500 DFM constraints, billet materials, and offline BoM."""

import ast
import copy
import csv
import json
from pathlib import Path
import re
import socket

import pytest
from typer.testing import CliRunner

from openengine.cli import app
from openengine.dfm import (
    C500Spec,
    PartMachiningSpec,
    STATUS_ANCHOR_TABLE,
    STATUS_COTS_SPEC,
    STATUS_VERIFIED_EXACT,
    export_bom_artifacts,
    load_bom_config,
    load_engine_specs,
    resolve_bom,
    resolve_bom_from_files,
)
from openengine.fea import (
    ALUMINUM_4032_T6,
    ALUMINUM_6061_T6,
    ALUMINUM_7075_T6,
    BRONZE_SAE_660,
    MATERIAL_CATALOG,
    STEEL_1144_STRESSPROOF,
    STEEL_4140_PREHARD,
    STEEL_4340,
    STEEL_GRADE_10_9,
    get_material_properties,
)

ORDER_PAD_LINE_RE = re.compile(r"^\d+\t\d{4,5}[A-Z]\d{2,4}$")
BANNED_NETWORK_MODULES = frozenset({
    "urllib",
    "http",
    "requests",
    "httpx",
    "aiohttp",
    "selenium",
    "playwright",
    "socket",
})


def test_fea_billet_materials_and_catalog():
  """Verify 7075-T6 and companion billet alloys exist with calibrated properties."""
  assert ALUMINUM_7075_T6.youngs_modulus_mpa == pytest.approx(71700.0)
  assert ALUMINUM_7075_T6.yield_strength_mpa == pytest.approx(503.0)
  assert ALUMINUM_7075_T6.ultimate_strength_mpa == pytest.approx(572.0)
  assert ALUMINUM_7075_T6.endurance_limit_mpa == pytest.approx(159.0)
  assert ALUMINUM_7075_T6.density_tonne_mm3 == pytest.approx(2.81e-9)

  assert ALUMINUM_6061_T6.yield_strength_mpa == pytest.approx(276.0)
  assert STEEL_1144_STRESSPROOF.yield_strength_mpa == pytest.approx(690.0)
  assert STEEL_4140_PREHARD.yield_strength_mpa == pytest.approx(655.0)
  assert BRONZE_SAE_660.yield_strength_mpa == pytest.approx(140.0)

  expected = {
      ALUMINUM_4032_T6.name,
      ALUMINUM_6061_T6.name,
      ALUMINUM_7075_T6.name,
      BRONZE_SAE_660.name,
      STEEL_1144_STRESSPROOF.name,
      STEEL_4140_PREHARD.name,
      STEEL_4340.name,
      STEEL_GRADE_10_9.name,
  }
  assert expected.issubset(set(MATERIAL_CATALOG.keys()))
  assert get_material_properties(" aluminum_7075_t6 ") == ALUMINUM_7075_T6
  with pytest.raises(KeyError):
    get_material_properties("UNOBTAINIUM_999")


def test_c500_ledger_passes_with_and_without_fourth_axis():
  """Verify all 10 billet parts pass with 4th-axis active and via 3-axis fallback."""
  spec = C500Spec.from_yaml()
  assert spec.has_fourth_axis is True
  assert len(spec.part_ledger) == 10
  assert spec.tooling.max_tool_shank_mm == pytest.approx(6.35)
  assert spec.envelope.max_z_clearance_mm == pytest.approx(160.0)

  active_checks = spec.validate_all_ledger_parts(has_fourth_axis_override=True)
  assert all(chk.is_valid for chk in active_checks)
  assert all(not chk.uses_fallback for chk in active_checks)

  fallback_checks = spec.validate_all_ledger_parts(has_fourth_axis_override=False)
  assert all(chk.is_valid for chk in fallback_checks)
  by_id = {chk.part_id: chk for chk in fallback_checks}
  for rotary_req in ("poppet_valves", "one_piece_camshaft", "valve_guides_and_seats"):
    assert by_id[rotary_req].uses_fallback is True
    assert by_id[rotary_req].active_mode == "fallback_cots_or_3axis"
    assert len(by_id[rotary_req].fallback_reason) > 0
  assert by_id["split_connecting_rod"].uses_fallback is False
  assert by_id["split_crankcase_and_sump_pan"].uses_fallback is False


@pytest.mark.parametrize(
    "mode,envelope,material,pocket,radius,threads,expected_valid",
    [
        ("three_axis_housing", (165.0, 185.0, 55.0), "ALUMINUM_6061_T6", 35.0, 1.6, ("M8x1.25",), True),
        ("three_axis_housing", (166.0, 185.0, 50.0), "ALUMINUM_6061_T6", 20.0, 2.0, (), False),
        ("three_axis_housing", (160.0, 186.0, 50.0), "ALUMINUM_6061_T6", 20.0, 2.0, (), False),
        ("three_axis_housing", (150.0, 150.0, 55.1), "ALUMINUM_6061_T6", 20.0, 2.0, (), False),
        ("rotary_4_axis", (95.0, 95.0, 165.0), "STEEL_4140_PREHARD", 10.0, 1.6, (), True),
        ("rotary_4_axis", (95.1, 95.0, 100.0), "STEEL_4140_PREHARD", 10.0, 1.6, (), False),
        ("rotary_4_axis", (80.0, 80.0, 165.1), "STEEL_4140_PREHARD", 10.0, 1.6, (), False),
        ("three_axis_four_side_flip", (120.0, 130.0, 100.0), "ALUMINUM_6061_T6", 20.0, 1.6, (), True),
        ("three_axis_four_side_flip", (131.0, 120.0, 92.0), "ALUMINUM_6061_T6", 20.0, 1.6, (), False),
        ("three_axis_four_side_flip", (120.0, 120.0, 186.0), "ALUMINUM_6061_T6", 20.0, 1.6, (), False),
        ("three_axis_housing", (120.0, 120.0, 30.0), "ALUMINUM_6061_T6", 35.1, 2.0, (), False),
        ("three_axis_housing", (120.0, 120.0, 30.0), "ALUMINUM_6061_T6", 20.0, 1.59, (), False),
        ("three_axis_housing", (120.0, 120.0, 30.0), "GRAY_CAST_IRON", 20.0, 2.0, (), False),
        ("three_axis_housing", (120.0, 120.0, 30.0), "TITANIUM_6AL4V", 20.0, 2.0, (), False),
        ("three_axis_housing", (120.0, 120.0, 30.0), "ALUMINUM_6061_T6", 20.0, 2.0, ("M14x2.0",), False),
        ("five_axis_trunnion", (100.0, 100.0, 40.0), "ALUMINUM_6061_T6", 10.0, 2.0, (), False),
    ],
)
def test_c500_boundary_checks(
    mode: str,
    envelope: tuple[float, float, float],
    material: str,
    pocket: float,
    radius: float,
    threads: tuple[str, ...],
    expected_valid: bool,
):
  """Verify strict C500 boundary enforcement on envelopes, tools, materials, threads."""
  spec = C500Spec.from_yaml()
  part = PartMachiningSpec(
      part_id="probe_part",
      module="test",
      mode=mode,
      fourth_axis_requirement="none",
      envelope_mm=envelope,
      material=material,
      pocket_depth_mm=pocket,
      min_corner_radius_mm=radius,
      fallback_strategy="Fallback note.",
  )
  chk = spec.check_part(part, threads_used=threads, has_fourth_axis_override=True)
  assert chk.is_valid is expected_valid


def test_c500_missing_and_invalid_yaml_errors(tmp_path: Path):
  """Verify explicit errors on missing config files or malformed part envelopes."""
  with pytest.raises(FileNotFoundError):
    C500Spec.from_yaml(tmp_path / "missing.yaml")
  with pytest.raises(FileNotFoundError):
    load_engine_specs(tmp_path / "missing_engine.yaml")
  with pytest.raises(FileNotFoundError):
    load_bom_config(tmp_path / "missing_bom.yaml")

  spec_path = Path("config/c500_constraints.yaml").read_text()
  bad_yaml = tmp_path / "bad_c500.yaml"
  bad_yaml.write_text(spec_path.replace("[30.0, 30.0, 78.0]", "[30.0, 30.0]"))
  with pytest.raises(ValueError):
    C500Spec.from_yaml(bad_yaml)


def test_baseline_bom_resolution_and_order_pad_export(tmp_path: Path):
  """Verify baseline BoM resolves exact SKUs, URLs, fits, and Order-Pad paste output."""
  report = resolve_bom_from_files()
  assert report.has_fourth_axis is True
  assert report.validation.is_valid is True
  assert report.validation.violations == ()

  by_id = {item.item_id: item for item in report.items}
  assert report.validation.total_items == 40
  assert report.validation.verified_exact_mcmaster_count == 19
  assert report.validation.anchor_table_mcmaster_count == 14
  assert report.validation.powersports_cots_count == 7
  assert by_id["head_tie_bolts"].sku == "91290A452"
  assert by_id["head_tie_bolts"].url == "https://www.mcmaster.com/91290A452/"
  assert by_id["head_tie_bolts"].sku_status == STATUS_VERIFIED_EXACT
  assert by_id["wrist_pin_circlips"].sku == "98541A143"
  assert by_id["main_journal_ball_bearings"].sku == "5972K342"
  assert (
      "Seals > Metric Rotary Shaft Seals > Shaft Dia: 30 mm"
      in by_id["main_shaft_viton_seals"].selection_hint
  )
  assert "cam_and_valve_4140_rotary_bar" in by_id
  assert "cots_fallback_valves_and_guides" not in by_id

  nested_dir = tmp_path / "nested" / "bom_out"
  paths = export_bom_artifacts(report, output_dir=nested_dir)
  # Re-exporting to an existing directory must overwrite cleanly (exist_ok=True)
  paths_overwrite = export_bom_artifacts(report, output_dir=nested_dir)
  assert paths_overwrite == paths
  pad_lines = [ln for ln in paths.order_pad_txt.read_text().splitlines() if ln]
  assert len(pad_lines) == report.validation.verified_exact_mcmaster_count
  assert all(ORDER_PAD_LINE_RE.match(ln) for ln in pad_lines)
  assert all(STATUS_ANCHOR_TABLE not in ln and STATUS_COTS_SPEC not in ln for ln in pad_lines)

  with open(paths.costed_csv, "r", newline="") as handle:
    csv_rows = list(csv.DictReader(handle))
  assert len(csv_rows) == report.validation.total_items

  summary_json = json.loads(paths.summary_json.read_text())
  assert summary_json["validation"]["total_cost_usd"] == pytest.approx(
      report.validation.total_cost_usd
  )


def test_parametric_stroke_and_size_fallback_to_anchor_url():
  """Verify geometry sweeps select longer SKUs and fall back cleanly to anchor links."""
  base_engine = load_engine_specs()
  c500 = C500Spec.from_yaml()
  bom_cfg = load_bom_config()

  cases = (
      (0.080, "91290A216", STATUS_VERIFIED_EXACT, "https://www.mcmaster.com/91290A216/"),
      (0.110, "91290A116", STATUS_VERIFIED_EXACT, "https://www.mcmaster.com/91290A116/"),
      (0.125, "91290A326", STATUS_ANCHOR_TABLE, "https://www.mcmaster.com/91290A326/"),
  )
  for stroke_m, exp_sku, exp_status, exp_url in cases:
    mod_engine = copy.deepcopy(base_engine)
    mod_engine["geometry"]["stroke"] = stroke_m
    rep = resolve_bom(engine_cfg=mod_engine, c500_spec=c500, bom_cfg=bom_cfg)
    bolts = {i.item_id: i for i in rep.items}["head_tie_bolts"]
    assert bolts.sku == exp_sku
    assert bolts.sku_status == exp_status
    assert bolts.url == exp_url
    assert rep.validation.is_valid is True
    if exp_status == STATUS_ANCHOR_TABLE:
      assert "94.5 mm" in bolts.selection_hint

  pin_mod = copy.deepcopy(base_engine)
  pin_mod["geometry"]["wrist_pin_diameter"] = 0.018
  pin_rep = resolve_bom(engine_cfg=pin_mod, c500_spec=c500, bom_cfg=bom_cfg)
  circlip = {i.item_id: i for i in pin_rep.items}["wrist_pin_circlips"]
  assert circlip.sku_status == STATUS_ANCHOR_TABLE
  assert circlip.url == "https://www.mcmaster.com/98541A143/"
  assert "18mm" in circlip.selection_hint


def test_no_fourth_axis_and_duplicate_sku_consolidation(tmp_path: Path):
  """Verify 4th-axis stock swaps to COTS kit and Order Pad merges duplicate SKUs."""
  rep_no4 = resolve_bom_from_files(has_fourth_axis_override=False)
  assert rep_no4.has_fourth_axis is False
  assert rep_no4.validation.is_valid is True
  assert rep_no4.validation.verified_exact_mcmaster_count == 19
  assert rep_no4.validation.anchor_table_mcmaster_count == 12
  assert rep_no4.validation.powersports_cots_count == 8
  assert rep_no4.validation.total_items == 39
  by_id = {item.item_id: item for item in rep_no4.items}
  assert "cam_and_valve_4140_rotary_bar" not in by_id
  assert "guides_932_bronze_rotary_bar" not in by_id
  assert "cots_fallback_valves_and_guides" in by_id

  base_engine = load_engine_specs()
  base_bom = load_bom_config()
  dup_bom = copy.deepcopy(base_bom)
  dup_bom["items"].append({
      "item_id": "spare_m6x20_bolts",
      "channel": "mcmaster_fasteners_seals",
      "description": "Spare M6x20 bolts",
      "vendor": "McMaster-Carr",
      "series": "shcs_alloy_class_12_9",
      "series_key": "M6x1.0",
      "thread": "M6x1.0",
      "target_length_mm": 20.0,
      "required_qty": 100,
      "selection_hint": "M6x20 spare",
  })
  dup_rep = resolve_bom(engine_cfg=base_engine, bom_cfg=dup_bom)
  pad_file = export_bom_artifacts(dup_rep, output_dir=tmp_path).order_pad_txt
  pad_map = {
      line.split("\t")[1]: int(line.split("\t")[0])
      for line in pad_file.read_text().splitlines()
  }
  # 1 pack (qty 18 / 50) + 2 packs (qty 100 / 50) = 3 packs consolidated
  assert pad_map["91290A326"] == 3


def test_interface_mismatch_detection():
  """Verify geometry, bearing, bore, fluid, lift, and ratio mismatches fail."""
  base_engine = load_engine_specs()
  c500 = C500Spec.from_yaml()
  base_bom = load_bom_config()

  bad_crankpin = copy.deepcopy(base_engine)
  bad_crankpin["geometry"]["crankpin_diameter"] = 0.032
  rep_pin = resolve_bom(engine_cfg=bad_crankpin, c500_spec=c500, bom_cfg=base_bom)
  assert rep_pin.validation.is_valid is False
  assert any("Crankpin bearing bore" in v for v in rep_pin.validation.violations)

  bad_bore = copy.deepcopy(base_engine)
  bad_bore["geometry"]["bore"] = 0.075
  rep_bore = resolve_bom(engine_cfg=bad_bore, c500_spec=c500, bom_cfg=base_bom)
  assert rep_bore.validation.is_valid is False
  assert any("Ring bore" in v for v in rep_bore.validation.violations)
  assert any("Liner OD" in v for v in rep_bore.validation.violations)
  assert any("Piston stock" in v and "< required" in v for v in rep_bore.validation.violations)

  bad_rod = copy.deepcopy(base_engine)
  bad_rod["geometry"]["con_rod_length"] = 0.140
  rep_rod = resolve_bom(engine_cfg=bad_rod, c500_spec=c500, bom_cfg=base_bom)
  assert rep_rod.validation.is_valid is False
  assert any("Con-rod billet" in v for v in rep_rod.validation.violations)

  bad_oil = copy.deepcopy(base_engine)
  bad_oil["lubrication"]["oil_grade"] = "SAE 20W-50"
  bad_oil["lubrication"]["sump_volume_l"] = 1.5
  rep_oil = resolve_bom(engine_cfg=bad_oil, c500_spec=c500, bom_cfg=base_bom)
  assert rep_oil.validation.is_valid is False
  assert any("SAE 20W-50" in v for v in rep_oil.validation.violations)
  assert any("sump capacity" in v for v in rep_oil.validation.violations)

  bad_valves = copy.deepcopy(base_engine)
  bad_valves["valves"]["intake_diameter"] = 0.032
  bad_valves["valves"]["max_lift"] = 0.011
  rep_val = resolve_bom(
      engine_cfg=bad_valves,
      c500_spec=c500,
      bom_cfg=base_bom,
      has_fourth_axis_override=False,
  )
  assert rep_val.validation.is_valid is False
  assert any("Spring travel" in v and "< required" in v for v in rep_val.validation.violations)
  assert any("COTS intake valve" in v for v in rep_val.validation.violations)

  bad_bom = copy.deepcopy(base_bom)
  for item in bad_bom["items"]:
    if item["item_id"] == "cam_timing_sprocket_36t":
      item["teeth"] = 34
  rep_ratio = resolve_bom(engine_cfg=base_engine, c500_spec=c500, bom_cfg=bad_bom)
  assert rep_ratio.validation.is_valid is False
  assert any("sprocket ratio" in v and "!= 2.00" in v for v in rep_ratio.validation.violations)


def test_bom_cli_execution(tmp_path: Path):
  """Verify openengine bom CLI command succeeds and fails strictly on errors."""
  runner = CliRunner()
  out_ok = tmp_path / "ok"
  res_ok = runner.invoke(app, ["bom", "--output-dir", str(out_ok)])
  assert res_ok.exit_code == 0
  assert (out_ok / "mcmaster_order_pad.txt").exists()

  out_no4 = tmp_path / "no4"
  res_no4 = runner.invoke(
      app, ["bom", "--no-fourth-axis", "--output-dir", str(out_no4)]
  )
  assert res_no4.exit_code == 0
  assert "3-Axis FALLBACK" in res_no4.stdout

  bad_engine = copy.deepcopy(load_engine_specs())
  bad_engine["geometry"]["bore"] = 0.075
  import yaml

  bad_engine_path = tmp_path / "bad_engine.yaml"
  bad_engine_path.write_text(yaml.safe_dump(bad_engine))
  res_bad = runner.invoke(
      app,
      ["bom", "--config-path", str(bad_engine_path), "--output-dir", str(tmp_path / "bad")],
  )
  assert res_bad.exit_code == 1


def test_zero_network_politeness_guardrail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  """Ensure DFM/BoM resolution performs zero sockets or HTTP imports."""
  def _block_network(*_args, **_kwargs):
    raise AssertionError("Offline BoM must never open network sockets")

  monkeypatch.setattr(socket, "create_connection", _block_network)
  monkeypatch.setattr(socket.socket, "connect", _block_network)

  rep = resolve_bom_from_files()
  assert rep.validation.is_valid is True
  export_bom_artifacts(rep, output_dir=tmp_path)

  dfm_dir = Path("src/openengine/dfm")
  for py_file in dfm_dir.glob("*.py"):
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          root = alias.name.split(".")[0]
          assert root not in BANNED_NETWORK_MODULES, f"{py_file} imports {root}"
      elif isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".")[0]
        assert root not in BANNED_NETWORK_MODULES, f"{py_file} imports {root}"


def perf_c500_ledger_validation(
    n: int = 64,
    *,
    _scales: tuple[int, ...] = (32, 64, 128, 256),
    _max_complexity: str = "O(N)",
):
  """Benchmark C500 envelope validation scaling across N synthetic parts."""
  spec = C500Spec.from_yaml()
  parts = tuple(
      PartMachiningSpec(
          part_id=f"part_{idx}",
          module="bench",
          mode="three_axis_housing",
          fourth_axis_requirement="none",
          envelope_mm=(120.0, 140.0, 40.0),
          material="ALUMINUM_6061_T6",
          pocket_depth_mm=20.0,
          min_corner_radius_mm=2.0,
          fallback_strategy="None",
      )
      for idx in range(n)
  )
  yield
  _ = [spec.check_part(p, threads_used=("M6x1.0",)) for p in parts]


def perf_resolve_bom(
    n: int = 64,
    *,
    _scales: tuple[int, ...] = (32, 64, 128, 256),
    _max_complexity: str = "O(N)",
):
  """Benchmark offline BoM resolution scaling across N replicated hardware items."""
  engine_cfg = load_engine_specs()
  c500 = C500Spec.from_yaml()
  bom_cfg = copy.deepcopy(load_bom_config())
  base_items = list(bom_cfg["items"])
  scaled_items = []
  for idx in range(n):
    template = copy.deepcopy(base_items[idx % len(base_items)])
    if template["item_id"] not in ("crank_timing_sprocket_18t", "cam_timing_sprocket_36t"):
      template["item_id"] = f"{template['item_id']}_{idx}"
    scaled_items.append(template)
  bom_cfg["items"] = base_items + scaled_items
  yield
  _ = resolve_bom(engine_cfg=engine_cfg, c500_spec=c500, bom_cfg=bom_cfg)
