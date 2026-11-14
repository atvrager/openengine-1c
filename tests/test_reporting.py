"""Unit tests for glTF export, WebGL FEA stress mapping, and interactive dashboard."""

import json
from pathlib import Path
import struct
import pytest

from openengine.reporting.dashboard import generate_interactive_dashboard
from openengine.reporting.gltf_export import (
    export_cad_models_to_gltf,
    export_fea_stress_to_webgl,
    turbo_colormap,
)
from openengine.reporting.pages import build_pages_site
from openengine.reporting.summary_md import generate_markdown_summary


def test_turbo_colormap():
  """Verify Turbo/Rainbow scientific colormap bounds and color progression."""
  # Bounds
  r0, g0, b0 = turbo_colormap(0.0)
  assert r0 == 0.0 and b0 == 1.0  # Deep blue

  r_mid, g_mid, b_mid = turbo_colormap(0.5)
  assert g_mid == 1.0  # Green peak

  r1, g1, b1 = turbo_colormap(1.0)
  assert r1 == 1.0 and b1 == 0.0  # Pure red

  # Clamping
  assert turbo_colormap(-0.5) == turbo_colormap(0.0)
  assert turbo_colormap(1.5) == turbo_colormap(1.0)


def test_fea_stress_webgl_export(tmp_path: Path):
  """Verify tetrahedral boundary surface extraction and vertex color mapping."""
  frd_path = Path("outputs/fea/piston/piston_lc1.frd")
  inp_path = Path("outputs/fea/piston/piston_mesh.inp")
  if not frd_path.exists() or not inp_path.exists():
    from openengine.fea.piston import PistonFeaConfig, run_piston_fea

    fea_dir = tmp_path / "piston_fea"
    run_piston_fea(
        config=PistonFeaConfig(mesh_size_min_mm=3.5, mesh_size_max_mm=7.0),
        output_dir=fea_dir,
    )
    frd_path = fea_dir / "piston_lc1.frd"
    inp_path = fea_dir / "piston_mesh.inp"

  out_json = tmp_path / "test_stress.json"
  meta = export_fea_stress_to_webgl(
      frd_path=frd_path,
      inp_path=inp_path,
      output_json_path=out_json,
      title="Test Piston Stress",
      component_name="Piston",
      yield_strength_mpa=315.0,
  )

  assert out_json.exists()
  assert meta["num_vertices"] > 500
  assert meta["num_triangles"] > 1000
  assert meta["yield_safety_factor"] >= 1.50

  with open(out_json, "r") as f:
    data = json.load(f)

  geom = data["geometry"]
  positions = geom["positions"]
  colors = geom["colors"]
  indices = geom["indices"]
  stresses = geom["stresses"]

  assert len(positions) == 3 * meta["num_vertices"]
  assert len(colors) == len(positions)
  assert len(indices) == 3 * meta["num_triangles"]
  assert len(stresses) == meta["num_vertices"]

  # Check RGB values in [0, 1]
  for c in colors:
    assert 0.0 <= c <= 1.0


def test_dashboard_html_generation(tmp_path: Path):
  """Verify generation of self-contained Three.js + Plotly dashboard."""
  html_file = tmp_path / "dashboard.html"
  models_dir = Path("outputs/reports/models")

  res = generate_interactive_dashboard(
      output_html_path=html_file,
      models_dir=models_dir,
  )

  assert html_file.exists()
  assert html_file.stat().st_size > 10000  # Non-trivial HTML file

  content = html_file.read_text()
  assert "three" in content
  assert "Plotly" in content
  assert "canvas-container-3d" in content
  assert "fea-canvas-container" in content
  assert "fea-model-select" in content
  assert "dressed_engine_assembly" in content
  assert "Score: 15 / 15 PASS" in content
  assert "NestWorks C500" in content
  assert "McMaster" in content
  assert "87-AKI" in content
  assert "setup-sheet-select" in content
  assert "dyno-audio-player" in content
  assert "swatch-chip" in content


def test_markdown_summary_generation(tmp_path: Path):
  """Verify generation of markdown verification scorecard."""
  md_file = tmp_path / "summary.md"
  res = generate_markdown_summary(output_md_path=md_file)

  assert md_file.exists()
  text = md_file.read_text()
  assert "Master Acceptance Scorecard" in text
  assert "15 / 15" in text
  assert "Full-Engine C3D10 Stress Atlas" in text
  assert "NestWorks C500" in text
  assert "PASS" in text


def _read_glb_json_chunk(glb_path: Path) -> dict:
  """Parse the JSON chunk from a binary glTF 2.0 (.glb) file."""
  raw = glb_path.read_bytes()
  assert raw[:4] == b"glTF"
  chunk_len, chunk_type = struct.unpack_from("<II", raw, 12)
  assert chunk_type == 0x4E4F534A
  return json.loads(raw[20 : 20 + chunk_len].decode("utf-8"))


def test_gltf_export_preserves_xcaf_materials(tmp_path: Path):
  """Verify XCAF PBR material colors/opacity and multi-body kinematics export cleanly to glTF."""
  models_dir = Path("outputs/reports/models")
  head_glb = models_dir / "cylinder_head.glb"
  conrod_glb = models_dir / "connecting_rod.glb"
  traj_json = models_dir / "kinematic_trajectory.json"
  if not (head_glb.exists() and conrod_glb.exists() and traj_json.exists()):
    export_cad_models_to_gltf(tmp_path)
    models_dir = tmp_path
    head_glb = models_dir / "cylinder_head.glb"
    conrod_glb = models_dir / "connecting_rod.glb"
    traj_json = models_dir / "kinematic_trajectory.json"

  head_gltf = _read_glb_json_chunk(head_glb)
  head_mats = head_gltf.get("materials", [])
  assert head_mats
  assert any(
      mat.get("alphaMode") == "BLEND"
      and mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])[3] < 1.0
      for mat in head_mats
  )

  conrod_gltf = _read_glb_json_chunk(conrod_glb)
  conrod_mats = conrod_gltf.get("materials", [])
  assert conrod_mats
  assert any(
      mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0, 0, 0, 1])[0]
      > mat.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0, 0, 0, 1])[2]
      for mat in conrod_mats
  )

  trajectory = json.loads(traj_json.read_text(encoding="utf-8"))
  assert len(trajectory) > 10
  sample = trajectory[0]
  assert "cam_angle_deg" in sample
  assert "rocker_intake_deg" in sample
  assert "rocker_exhaust_deg" in sample


def test_build_pages_site_bundle_and_links(tmp_path: Path):
  """Verify static GitHub Pages bundle assembly and relative asset link integrity."""
  site_dir = tmp_path / "_site"
  manifest = build_pages_site(outputs_dir="outputs", site_dir=site_dir)

  expected_files = (
      "index.html",
      "engineering_dashboard.html",
      ".nojekyll",
      "summary.md",
      "models/dressed_engine_assembly.glb",
      "cad/dressed_engine_assembly.step",
      "dyno/exhaust_pull_44k1.wav",
      "dfm/drawings/cylinder_head_setup.svg",
      "dfm/drawings/cylinder_head_setup.dxf",
      "bom/mcmaster_order_pad.txt",
  )
  for rel in expected_files:
    target = site_dir / rel
    assert target.is_file(), f"Missing Pages asset: {rel}"
    if rel != ".nojekyll":
      assert target.stat().st_size > 0, f"Empty Pages asset: {rel}"

  assert len(manifest.verified_links) >= 7
