"""Integration tests for full-engine CalculiX C3D10 FEA atlas and WebGL export."""

import base64
import json
from pathlib import Path

from openengine.fea.assembly_atlas import export_fea_atlas_to_webgl, run_fea_atlas

EXPECTED_TOTAL_SOLIDS: int = 114
EXPECTED_UNIQUE_MESHES: int = 69
EXPECTED_ASSEMBLIES: tuple[str, ...] = (
    "dressed_engine_assembly",
    "engine_assembly",
    "rotating_assembly",
    "head_assembly",
    "crankcase_barrel_assembly",
)
MIN_YIELD_SF_EXPECTED: float = 1.20
BYTES_PER_VEC3_F32: int = 12
BYTES_PER_F32: int = 4
BYTES_PER_TRI_U32: int = 12
EXPECTED_MIN_SOLIDS_CORE_ENGINE: int = 9


def test_fea_atlas_covers_all_dressed_solids_and_assemblies() -> None:
  """Verify cached CalculiX C3D10 atlas covers all 114 solids, 69 meshes, and 5 assemblies."""
  res = run_fea_atlas(output_dir="outputs/fea/atlas", force_rebuild=False)
  assert res.total_solids == EXPECTED_TOTAL_SOLIDS
  assert res.unique_meshes == EXPECTED_UNIQUE_MESHES
  assert res.all_passed is True
  assert res.min_yield_safety_factor >= MIN_YIELD_SF_EXPECTED

  asm_by_id = {asm.assembly_id: asm for asm in res.assemblies}
  assert tuple(asm_by_id.keys()) == EXPECTED_ASSEMBLIES
  assert asm_by_id["dressed_engine_assembly"].part_count == EXPECTED_TOTAL_SOLIDS
  assert asm_by_id["engine_assembly"].part_count >= EXPECTED_MIN_SOLIDS_CORE_ENGINE
  for asm in res.assemblies:
    assert asm.is_pass is True
    assert asm.min_yield_safety_factor >= MIN_YIELD_SF_EXPECTED


def test_fea_atlas_webgl_base64_buffers(tmp_path: Path) -> None:
  """Verify WebGL FEA atlas Base64 Float32/Uint32 buffers decode to exact byte counts."""
  out_json = tmp_path / "fea_stress_atlas.json"
  summary = export_fea_atlas_to_webgl(
      fea_root_dir="outputs/fea",
      output_json_path=out_json,
      force_rebuild=False,
  )
  assert out_json.is_file()
  assert summary["all_passed"] is True
  assert summary["total_solids"] == EXPECTED_TOTAL_SOLIDS
  assert summary["unique_meshes"] == EXPECTED_UNIQUE_MESHES

  payload = json.loads(out_json.read_text(encoding="utf-8"))
  meshes = payload["meshes"]
  models = payload["models"]
  assert len(meshes) == EXPECTED_UNIQUE_MESHES
  assert len(models) == len(EXPECTED_ASSEMBLIES) + EXPECTED_UNIQUE_MESHES

  for mesh_entry in meshes.values():
    v_cnt = int(mesh_entry["v_count"])
    t_cnt = int(mesh_entry["t_count"])
    assert v_cnt > 0 and t_cnt > 0
    assert len(base64.b64decode(mesh_entry["pos_b64"])) == v_cnt * BYTES_PER_VEC3_F32
    assert len(base64.b64decode(mesh_entry["vm_b64"])) == v_cnt * BYTES_PER_F32
    assert len(base64.b64decode(mesh_entry["idx_b64"])) == t_cnt * BYTES_PER_TRI_U32
