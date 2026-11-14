"""Markdown engineering summary and Master Acceptance Scorecard generator."""

from pathlib import Path
from typing import List, Union

from openengine.reporting._telemetry import (
    DashboardTelemetry,
    MM_TO_UM,
    PERCENT_SCALE,
    collect_dashboard_telemetry,
    _fval,
    _ival,
    _submap,
    _sval,
)


def _format_scorecard_section(t: DashboardTelemetry) -> List[str]:
  """Build Markdown table rows for all multi-physics acceptance gates."""
  lines = [
      f"## 1. Master Acceptance Scorecard ({t.passed_count} / {t.total_checks} PASS)",
      "",
      "| Subsystem | Metric | Target | Achieved | Design Margin | Status |",
      "| :--- | :--- | :--- | :--- | :--- | :--- |",
  ]
  for row in t.scorecard:
    status = "**PASS**" if row.passed else "**FAIL**"
    lines.append(
        f"| **{row.subsystem}** | {row.metric} | {row.target} | **{row.achieved}** | {row.margin} | {status} |"
    )
  return lines


def _format_thermo_lub_and_fea_section(t: DashboardTelemetry) -> List[str]:
  """Build Markdown sections covering 87-AKI knock, splash lubrication, and FEA."""
  fuel = _submap(t.thermo_lub, "fuel")
  cycle = _submap(t.thermo_lub, "cycle")
  fric = _submap(t.thermo_lub, "friction_zones")
  splash = _submap(t.thermo_lub, "splash")
  therm = _submap(t.thermo_lub, "thermal")
  piston_lc1 = _submap(t.piston, "lc1_mechanical")

  return [
      "## 2. US 87-Octane E10 Knock, 4-Zone FMEP & Wet-Sump Splash Cooling",
      f"- **Fuel Blend:** {_fval(fuel, 'aki'):.0f} AKI ({_fval(fuel, 'ron'):.1f} RON / {_fval(fuel, 'mon'):.1f} MON), {_fval(fuel, 'ethanol_pct'):.0f}% ethanol, stoichiometric AFR {_fval(fuel, 'stoich_afr'):.1f}",
      f"- **Livengood-Wu Knock Integral:** I = {_fval(cycle, 'knock_integral'):.4f} (margin +{_fval(cycle, 'knock_margin') * PERCENT_SCALE:.1f}%, is_knocking={cycle.get('is_knocking')})",
      f"- **4-Zone Mechanical Friction:** {_fval(fric, 'total_fmep_bar'):.2f} bar FMEP ({_fval(fric, 'total_friction_power_w'):.0f} W total friction)",
      f"- **Splash & Chain Transport:** {_fval(splash, 'static_immersion_mm'):.2f} -> {_fval(splash, 'dynamic_immersion_mm'):.2f} mm dipper immersion, {_fval(splash, 'tip_throw_velocity_m_s'):.2f} m/s tip velocity, {_fval(splash, 'drain_safety_ratio'):.1f}x chain drain safety",
      f"- **Sump Thermal Equilibrium:** {_fval(therm, 'equilibrium_oil_temp_c'):.1f} C equilibrium oil temp (limit <= {_fval(therm, 'max_allowable_oil_temp_c'):.0f} C); splash-cooled piston crown {_fval(therm, 'cooled_piston_crown_temp_c'):.1f} C",
      "",
      "## 3. Structural FEA & Heat Rejection",
      f"- **Piston LC-1 & LC-2 (4032-T6):** Yield SF {_fval(piston_lc1, 'yield_safety_factor'):.2f}, max P95 stress {_fval(piston_lc1, 'p95_von_mises_mpa'):.1f} MPa, hot skirt clearance {_fval(t.piston, 'hot_skirt_clearance_mm') * MM_TO_UM:.1f} um",
      f"- **Connecting Rod LC-3A (7075-T6):** Buckling SF {_fval(t.conrod, 'buckling_safety_factor'):.2f}, Goodman fatigue SF {_fval(t.conrod, 'fatigue_safety_factor'):.2f}",
      f"- **Head Gasket LC-4:** Fire-ring contact {_fval(t.gasket, 'residual_contact_pressure_mpa'):.1f} MPa (sealing SF {_fval(t.gasket, 'sealing_safety_factor'):.2f})",
      f"- **Full-Engine C3D10 Stress Atlas:** {_ival(t.fea_atlas, 'total_solids')} solids across {_ival(t.fea_atlas, 'unique_meshes')} unique C3D10 meshes, min yield SF {_fval(t.fea_atlas, 'min_yield_safety_factor'):.2f} ({_sval(t.fea_atlas, 'limiting_part_key', 'n/a')} gov., all_passed={t.fea_atlas.get('all_passed')})",
      f"- **Air-Fin CHT Cooling:** {_fval(t.cht, 'heat_dissipated_w'):.1f} W dissipated (+{_fval(t.cht, 'thermal_margin_percent'):.1f}% thermal margin)",
  ]


def _format_dfm_bom_dyno_section(t: DashboardTelemetry) -> List[str]:
  """Build Markdown sections for NestWorks C500 DFM, Costed BoM, and Virtual Dyno."""
  bom_val = _submap(t.bom, "validation")
  steady = _submap(t.dyno, "steady_sweep")
  bsfc = _submap(steady, "bsfc_map")
  pull = _submap(t.dyno, "inertia_pull")
  audio = _submap(t.dyno, "exhaust_audio")

  return [
      "## 4. NestWorks C500 CNC DFM, 2D Setup Sheets & Costed BoM",
      f"- **Machine Audit:** {_sval(t.dfm, 'machine_name')} (all_passed={t.dfm.get('all_passed')}, 4th-axis={t.dfm.get('has_fourth_axis')})",
      f"- **1:1 Dressed B-Rep Twin:** {t.total_part_count} watertight solids, dry mass {t.dry_mass_kg:.2f} kg, wet mass {t.wet_mass_kg:.2f} kg, sump fill {t.oil_vol_l:.4f} L",
      f"- **2D Orthographic Setup Sheets:** {len(t.setup_sheets)} SVG + DXF shop drawings in `outputs/dfm/drawings/`",
      f"- **Costed McMaster-Carr + Powersports BoM:** {_ival(bom_val, 'total_items')} items totaling ${_fval(bom_val, 'total_cost_usd'):,.2f} ({_ival(bom_val, 'verified_exact_mcmaster_count')} verified exact SKUs)",
      "",
      "## 5. Virtual Dynamometer & Exhaust Audio",
      f"- **Peak SAE J1349 Brake Power:** {_fval(steady, 'peak_sae_power_hp'):.2f} hp ({_fval(steady, 'peak_sae_power_kw'):.2f} kW) @ {_fval(steady, 'peak_power_rpm'):.0f} RPM",
      f"- **Peak SAE J1349 Torque:** {_fval(steady, 'peak_sae_torque_lb_ft'):.2f} lb-ft ({_fval(steady, 'peak_sae_torque_nm'):.2f} N-m) @ {_fval(steady, 'peak_torque_rpm'):.0f} RPM",
      f"- **Minimum BSFC Island:** {_fval(bsfc, 'min_bsfc_g_kwh'):.1f} g/kWh @ {_fval(bsfc, 'min_bsfc_rpm'):.0f} RPM",
      f"- **Inertia Pull & Crank Shrink Fit:** {_fval(pull, 'wot_pull_duration_s'):.2f} s WOT pull, H6/x6 slip SF {_fval(pull, 'min_crank_press_fit_sf'):.2f}, flywheel burst SF {_fval(pull, 'flywheel_burst_sf_redline'):.1f}",
      f"- **Exhaust Audio:** `outputs/dyno/exhaust_pull_44k1.wav` ({_ival(audio, 'sample_rate_hz')} Hz PCM, {_fval(audio, 'duration_s'):.2f} s)",
  ]


def generate_markdown_summary(
    output_md_path: Union[str, Path] = "outputs/reports/summary.md",
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
    bom_json: Union[str, Path] = "outputs/bom/bom_summary.json",
    thermo_lub_json: Union[str, Path] = "outputs/lubrication/thermo_lub_results.json",
    setup_sheets_json: Union[str, Path] = "outputs/dfm/drawings/setup_sheets_manifest.json",
) -> str:
  """Generate the structured Markdown engineering verification scorecard."""
  del combustion_json, port_flow_json
  out_p = Path(output_md_path)
  out_p.parent.mkdir(parents=True, exist_ok=True)
  t = collect_dashboard_telemetry(
      models_dir=out_p.parent / "models",
      cht_json=cht_json,
      closed_loop_json=closed_loop_json,
      piston_fea_json=piston_fea_json,
      conrod_fea_json=conrod_fea_json,
      gasket_fea_json=gasket_fea_json,
      fea_atlas_json=fea_atlas_json,
      dfm_report_json=dfm_report_json,
      dyno_json=dyno_json,
      bom_json=bom_json,
      thermo_lub_json=thermo_lub_json,
      setup_sheets_json=setup_sheets_json,
  )

  lines: List[str] = [
      "# OpenEngine-1C Multi-Physics Engineering Verification Report",
      "",
      "**Engine Specification:** Single-Cylinder, 4-Stroke SOHC, 2-Valve Pent-Roof Chamber  ",
      f"**Displacement:** {t.bore_mm:.1f} mm bore x {t.stroke_mm:.1f} mm stroke ({t.swept_cm3:.1f} cm3), L = {t.conrod_mm:.1f} mm, CR = {t.compression_ratio:.2f}:1  ",
      f"**Valvetrain Float Margin @ 8400 RPM:** Intake +{t.intake_float_margin_pct:.1f}% ({t.intake_float_rpm:.0f} RPM) / Exhaust +{t.exhaust_float_margin_pct:.1f}% ({t.exhaust_float_rpm:.0f} RPM)  ",
      "",
      "---",
      "",
  ]
  lines.extend(_format_scorecard_section(t))
  lines.extend(["", "---", ""])
  lines.extend(_format_thermo_lub_and_fea_section(t))
  lines.extend(["", "---", ""])
  lines.extend(_format_dfm_bom_dyno_section(t))

  out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return str(out_p)
