"""Interactive WebGL 3D CAD, kinematics, FEA stress, DFM, BoM, and dyno dashboard generator."""

from pathlib import Path
from typing import Union

from openengine.reporting._dashboard_html import render_dashboard_html
from openengine.reporting._telemetry import (
    collect_dashboard_telemetry,
    load_json_or_default,
)

__all__ = [
    "generate_interactive_dashboard",
    "load_json_or_default",
]


def generate_interactive_dashboard(
    output_html_path: Union[str, Path] = "outputs/reports/engineering_dashboard.html",
    models_dir: Union[str, Path] = "outputs/reports/models",
    combustion_json: Union[str, Path] = "outputs/cfd/combustion_results.json",
    port_flow_json: Union[str, Path] = "outputs/cfd/port_flow_results.json",
    cht_json: Union[str, Path] = "outputs/cfd/cht_cooling_results.json",
    closed_loop_json: Union[str, Path] = "outputs/calibration/closed_loop_results.json",
    piston_fea_json: Union[str, Path] = "outputs/fea/piston/piston_fea_results.json",
    conrod_fea_json: Union[str, Path] = "outputs/fea/conrod/conrod_fea_results.json",
    gasket_fea_json: Union[str, Path] = "outputs/fea/gasket/head_gasket_fea_results.json",
    fea_atlas_json: Union[str, Path] = "outputs/fea/atlas/fea_atlas_results.json",
    dfm_report_json: Union[str, Path] = "outputs/dfm/c500_dfm_report.json",
    dyno_json: Union[str, Path] = "outputs/dyno/dyno_results.json",
    dyno_wav: Union[str, Path] = "outputs/dyno/exhaust_pull_44k1.wav",
    bom_json: Union[str, Path] = "outputs/bom/bom_summary.json",
    thermo_lub_json: Union[str, Path] = "outputs/lubrication/thermo_lub_results.json",
    setup_sheets_json: Union[str, Path] = "outputs/dfm/drawings/setup_sheets_manifest.json",
    asset_url_prefix: str = "..",
) -> str:
  """Generate the single-file interactive Three.js + Plotly engineering dashboard HTML."""
  del combustion_json, port_flow_json
  out_file = Path(output_html_path)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  m_dir = Path(models_dir)
  m_dir.mkdir(parents=True, exist_ok=True)

  telemetry = collect_dashboard_telemetry(
      models_dir=m_dir,
      cht_json=cht_json,
      closed_loop_json=closed_loop_json,
      piston_fea_json=piston_fea_json,
      conrod_fea_json=conrod_fea_json,
      gasket_fea_json=gasket_fea_json,
      fea_atlas_json=fea_atlas_json,
      dfm_report_json=dfm_report_json,
      dyno_json=dyno_json,
      dyno_wav=dyno_wav,
      bom_json=bom_json,
      thermo_lub_json=thermo_lub_json,
      setup_sheets_json=setup_sheets_json,
  )
  html_content = render_dashboard_html(
      telemetry,
      out_html_path=out_file,
      asset_url_prefix=asset_url_prefix,
  )
  out_file.write_text(html_content, encoding="utf-8")
  return str(out_file)
