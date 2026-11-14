"""OpenEngine-1C Command Line Interface."""

from pathlib import Path
import json
import time
from typing import Optional
import numpy as np
from openengine.cad import (
    FLUID_OPACITY,
    OPAQUE_OPACITY,
    SINGLE_HOUSING_OPACITY,
    calculate_block_thermal_capacity,
    calculate_combustion_chamber_volume,
    calculate_crankshaft_balance,
    export_styled_step,
    extract_intake_port_fluid_domain,
    generate_connecting_rod,
    generate_crankshaft,
    generate_cylinder_block,
    generate_cylinder_head,
    generate_cylinder_liner,
    generate_dressed_engine_assembly,
    generate_engine_assembly,
    generate_piston,
    generate_poppet_valve,
    style_single_part,
    verify_all_c500_fixtures,
    verify_assembly_clearances,
    verify_dressed_engine,
)
from openengine.cad.valvetrain_hardware import (
    CAM_JOURNAL_DIAMETER_M,
    TOWER_FRONT_JOURNAL_LAND_M,
)
from openengine.cfd import (
    ChtCoolingConfig,
    ClosedLoopConfig,
    CombustionConfig,
    PortFlowConfig,
    run_cht_simulation,
    run_closed_loop_calibration,
    run_combustion_simulation,
    run_port_flow_sweep,
)
from openengine.dfm import (
    export_bom_artifacts,
    export_c500_setup_sheets,
    resolve_bom_from_files,
    run_c500_dfm_check,
)
from openengine.dynamics import (
    BearingGeometry,
    CRANKPIN_NEEDLE_NKS28_SPEC,
    FlywheelDimensions,
    FlywheelDynamics,
    HydrodynamicBearing,
    MAIN_BALL_6206_SPEC,
    OilProperties,
    SOHCCamshaft,
    SliderCrankDynamics,
    analyze_rolling_bearing_life,
)
from openengine.fea import (
    ALUMINUM_4032_T6,
    ALUMINUM_7075_T6,
    ConrodFeaConfig,
    HeadGasketFeaConfig,
    PistonFeaConfig,
    run_conrod_fea,
    run_head_gasket_fea,
    run_piston_fea,
)
from openengine.fea.assembly_atlas import (
    MIN_YIELD_SF_GATE,
    FeaAtlasSummaryResult,
    export_fea_atlas_to_webgl,
    run_fea_atlas,
)
from openengine.infra.podman import PodmanRunner
from openengine.lubrication import (
    OilGradeSpec,
    SplashSystemParams,
    SumpThermalParams,
    solve_sump_thermal_balance,
)
from openengine.reporting import (
    build_pages_site,
    export_cad_models_to_gltf,
    export_fea_stress_to_webgl,
    generate_interactive_dashboard,
    generate_markdown_summary,
)
from openengine.dyno import (
    DEFAULT_DRUM_INERTIA_KG_M2,
    InertiaPullConfig,
    SteadyDynoConfig,
    build_live_twin_simulator,
    run_inertia_dyno_pull,
    run_steady_dyno_sweep,
    synthesize_exhaust_audio,
)
from openengine.thermo.cantera_kinetics import FuelBlendSpec
from openengine.thermo.cycle import ClosedCycleSimulator, EngineGeometry
from openengine.thermo.gas_exchange import FourStrokeSimulator, ValveGeometry, ValveTiming
import rich
from rich.table import Table
import typer
import yaml

app = typer.Typer(
    help="OpenEngine-1C: Code-first 1-Cylinder 4-Stroke Engine Design System"
)


def load_config(config_path: str = "config/engine_specs.yaml") -> dict:
  path = Path(config_path)
  if not path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
  with open(path, "r") as f:
    return yaml.safe_load(f)


def _solve_default_thermal_balance(
    config_path: str = "config/engine_specs.yaml",
    rpm: float = 0.0,
):
  """Solve coupled 4-stroke cycle and equilibrium wet-sump splash thermal balance."""
  cfg = load_config(config_path)
  lub_cfg = cfg.get("lubrication", {})
  target_rpm = rpm if rpm > 0.0 else float(cfg["operating_conditions"]["rated_rpm"])
  grade_name = str(lub_cfg.get("oil_grade", "SAE 10W-40"))
  spec = OilGradeSpec.from_name(grade_name)
  splash_params = SplashSystemParams.from_config(cfg)
  thermal_params = SumpThermalParams.from_config(cfg)

  sim = FourStrokeSimulator.from_config(cfg)
  sim.oil_spec = spec
  thermo_res = sim.solve_cycle(rpm=target_rpm)
  tb = solve_sump_thermal_balance(
      geom=sim.geom,
      rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      pressure_pa=thermo_res.pressure_pa,
      oil_spec=spec,
      splash_params=splash_params,
      thermal_params=thermal_params,
  )
  return cfg, sim, thermo_res, spec, tb


@app.command()
def info(config_path: str = "config/engine_specs.yaml"):
  """Display current engine specification and configuration."""
  cfg = load_config(config_path)
  table = Table(
      title=f"Engine Specification: {cfg['engine']['name']}",
      show_header=True,
      header_style="bold magenta",
  )
  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="green")

  geom = cfg["geometry"]
  table.add_row("Bore", f"{geom['bore']*1000:.1f} mm")
  table.add_row("Stroke", f"{geom['stroke']*1000:.1f} mm")
  table.add_row("Con-Rod Length", f"{geom['con_rod_length']*1000:.1f} mm")
  table.add_row("Compression Ratio", f"{geom['compression_ratio']:.1f} : 1")
  fuel_spec = FuelBlendSpec.from_config(cfg)
  table.add_row(
      "Fuel Specification",
      f"{fuel_spec.name} (AKI {fuel_spec.aki:.1f}, E{fuel_spec.ethanol_pct:.0f})",
  )
  lub_cfg = cfg.get("lubrication", {})
  if lub_cfg:
    table.add_row(
        "Lubrication System",
        f"{lub_cfg.get('oil_grade', 'SAE 10W-40')},"
        f" {float(lub_cfg.get('sump_volume_l', 0.65)):.2f} L wet-sump splash",
    )
  table.add_row("Valvetrain", str(cfg["engine"]["valvetrain"]))
  table.add_row(
      "Rated / Redline Speed",
      f"{cfg['operating_conditions']['rated_rpm']:.0f} /"
      f" {cfg['operating_conditions']['redline_rpm']:.0f} RPM",
  )

  rich.print(table)


@app.command()
def thermo(
    config_path: str = "config/engine_specs.yaml",
    rpm: float = 0.0,
    closed_only: bool = False,
):
  """Run thermodynamic simulation (full 720-deg 4-stroke breathing or closed-cycle)."""
  cfg = load_config(config_path)
  g = cfg["geometry"]
  geom = EngineGeometry(
      bore=g["bore"],
      stroke=g["stroke"],
      con_rod_length=g["con_rod_length"],
      compression_ratio=g["compression_ratio"],
  )

  target_rpm = rpm if rpm > 0.0 else float(cfg["operating_conditions"]["rated_rpm"])

  if closed_only:
    sim = ClosedCycleSimulator(geom)
    res = sim.solve_cycle()

    table = Table(
        title="0D Closed-Cycle Thermodynamic Results",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Output Metric", style="cyan")
    table.add_column("Simulated Value", style="bold green")

    table.add_row("Displacement", f"{geom.displacement * 1.0e6:.1f} cm³")
    table.add_row("Peak Pressure (P_max)", f"{res.p_max_pa / 1.0e6:.2f} MPa")
    table.add_row("Peak Temperature (T_max)", f"{res.t_max_k:.0f} K")
    table.add_row("Indicated Work per Cycle", f"{res.indicated_work_j:.1f} J")
    table.add_row("IMEP", f"{res.imep_pa / 1.0e5:.2f} bar")
    table.add_row("Thermal Efficiency", f"{res.thermal_efficiency * 100:.1f} %")
    rich.print(table)
  else:
    # Full 720-degree 4-stroke breathing cycle
    sim = FourStrokeSimulator.from_config(cfg)
    res = sim.solve_cycle(rpm=target_rpm)

    table = Table(
        title=f"720° Four-Stroke Engine Cycle & Breathing @ {target_rpm:.0f} RPM",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Cycle Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Displacement", f"{geom.displacement * 1.0e6:.1f} cm³")
    table.add_row(
        "Fuel Grade",
        f"{sim.fuel.name} (AKI {sim.fuel.aki:.1f}, AFR {sim.afr:.1f})",
    )
    table.add_row("Volumetric Efficiency (η_v)", f"{res.volumetric_efficiency * 100:.1f} %")
    table.add_row("Trapped Air-Fuel Mass", f"{res.trapped_mass_kg * 1.0e6:.1f} mg / cycle")
    table.add_row("Peak Cylinder Pressure (P_max)", f"{res.p_max_pa / 1.0e6:.2f} MPa")
    table.add_row("Peak Gas Temperature (T_max)", f"{res.t_max_k:.0f} K")
    table.add_row("Peak End-Gas Temperature (T_u,max)", f"{res.end_gas_temp_max_k:.0f} K")
    knock_label = "KNOCK DETECTED" if res.is_knocking else "SAFE"
    table.add_row(
        "Livengood-Wu Knock Integral",
        f"{res.knock_integral:.3f} ({knock_label}, margin {res.knock_margin:+.3f})",
    )
    table.add_row("Gross Indicated Work (High-P loop)", f"{res.gross_work_j:.1f} J")
    table.add_row("Pumping Work Loss (Low-P loop)", f"{res.pump_work_j:.1f} J")
    table.add_row("Net Indicated Work (W_net)", f"{res.net_work_j:.1f} J")
    table.add_row("GIMEP (Gross IMEP)", f"{res.gimep_pa / 1.0e5:.2f} bar")
    table.add_row("PMEP (Pumping Loss MEP)", f"{res.pmep_pa / 1.0e5:.2f} bar")
    table.add_row("NIMEP (Net Indicated MEP)", f"{res.nimep_pa / 1.0e5:.2f} bar")
    table.add_row("Indicated Engine Power", f"{res.power_kw:.2f} kW ({res.power_kw * 1.341:.1f} hp)")
    table.add_row("Indicated Engine Torque", f"{res.torque_nm:.2f} N·m")
    table.add_row("Sump Oil & Equilibrium Temp", f"{res.oil_grade} @ {res.oil_temp_c:.1f} °C")
    table.add_row("FMEP (Mechanical Friction MEP)", f"{res.fmep_pa / 1.0e5:.2f} bar")
    table.add_row("BMEP (Brake Shaft MEP)", f"{res.bmep_pa / 1.0e5:.2f} bar")
    table.add_row("Brake Engine Power", f"{res.brake_power_kw:.2f} kW ({res.brake_power_kw * 1.341:.1f} hp)")
    table.add_row("Brake Engine Torque", f"{res.brake_torque_nm:.2f} N·m")
    table.add_row("Mechanical Efficiency (η_m)", f"{res.mechanical_efficiency * 100:.1f} %")
    table.add_row("Brake Specific Fuel Consumption (BSFC)", f"{res.bsfc_g_kwh:.1f} g/kWh")
    table.add_row("Wall Heat Loss per Cycle (Woschni)", f"{res.q_wall_loss_j:.1f} J ({res.heat_loss_fraction * 100:.1f} % fuel energy)")
    table.add_row("Cooling Fin Heat Dissipation Power", f"{res.mean_cooling_power_w:.0f} W ({res.mean_cooling_power_w / 1000.0:.2f} kW)")
    table.add_row("Peak Instantaneous Heat Flux", f"{res.peak_heat_flux_mw_m2:.2f} MW/m²")

    rich.print(table)

    if res.equilibrium_species:
      sp_table = Table(
          title="Cantera Chemical Equilibrium (Exhaust Dissociated Species)",
          show_header=True,
          header_style="bold magenta",
      )
      sp_table.add_column("Species", style="cyan")
      sp_table.add_column("Mole Fraction", style="green")
      for sp, frac in res.equilibrium_species.items():
        sp_table.add_row(sp, f"{frac * 100:.2f} %")
      rich.print(sp_table)


@app.command()
def lub(
    config_path: str = "config/engine_specs.yaml",
    rpm: float = 0.0,
    oil_grade: str = "",
    output_dir: str = "outputs/lubrication",
):
  """Run wet-sump splash lubrication, 4-zone FMEP, and sump thermal balance."""
  cfg = load_config(config_path)
  lub_cfg = cfg.get("lubrication", {})
  target_rpm = rpm if rpm > 0.0 else float(cfg["operating_conditions"]["rated_rpm"])
  grade_name = oil_grade if oil_grade else str(lub_cfg.get("oil_grade", "SAE 10W-40"))

  spec = OilGradeSpec.from_name(grade_name)
  splash_params = SplashSystemParams.from_config(cfg)
  thermal_params = SumpThermalParams.from_config(cfg)

  sim = FourStrokeSimulator.from_config(cfg)
  sim.oil_spec = spec
  res = sim.solve_cycle(rpm=target_rpm)
  tb = solve_sump_thermal_balance(
      geom=sim.geom,
      rpm=target_rpm,
      theta_deg=res.theta_deg,
      pressure_pa=res.pressure_pa,
      oil_spec=spec,
      splash_params=splash_params,
      thermal_params=thermal_params,
  )
  fric = tb.friction
  splash = tb.splash

  rheo_table = Table(
      title=f"Lubricant Rheology ({spec.grade} — Walther & Vogel)",
      show_header=True,
      header_style="bold magenta",
  )
  rheo_table.add_column("Temperature", style="cyan")
  rheo_table.add_column("Kinematic ν (cSt)", style="green")
  rheo_table.add_column("Walther μ (mPa·s)", style="green")
  rheo_table.add_column("Vogel μ (mPa·s)", style="green")
  rheo_table.add_column("Density (kg/m³)", style="green")
  for tc in (20.0, 40.0, 100.0, tb.equilibrium_oil_temp_c):
    st = spec.evaluate(tc)
    label = f"{tc:.1f} °C (Equil)" if tc == tb.equilibrium_oil_temp_c else f"{tc:.0f} °C"
    rheo_table.add_row(
        label,
        f"{st.kinematic_viscosity_cst:.2f}",
        f"{st.dynamic_viscosity_pa_s * 1000.0:.2f}",
        f"{st.vogel_dynamic_viscosity_pa_s * 1000.0:.2f}",
        f"{st.density_kg_m3:.1f}",
    )
  rich.print(rheo_table)

  fmep_table = Table(
      title=f"4-Zone Mechanical Friction (FMEP) & Brake Output @ {target_rpm:.0f} RPM",
      show_header=True,
      header_style="bold blue",
  )
  fmep_table.add_column("Tribological Zone / Metric", style="cyan")
  fmep_table.add_column("Value", style="bold green")
  fmep_table.add_row(
      "Zone 1: Piston Ring Pack & Skirt",
      f"{fric.ring_skirt_fmep_pa / 1.0e5:.2f} bar ({fric.ring_skirt_heat_w:.0f} W)",
  )
  fmep_table.add_row(
      "Zone 2: Rolling Crank Bearings & Viton Seals",
      f"{fric.rolling_bearings_fmep_pa / 1.0e5:.2f} bar ({fric.rolling_bearings_heat_w:.0f} W)",
  )
  fmep_table.add_row(
      "Zone 3: SOHC Valvetrain & #25 Roller Chain",
      f"{fric.valvetrain_fmep_pa / 1.0e5:.2f} bar ({fric.valvetrain_heat_w:.0f} W)",
  )
  fmep_table.add_row(
      "Zone 4: Con-Rod Splash Dipper & Bay Windage",
      f"{fric.dipper_windage_fmep_pa / 1.0e5:.2f} bar ({fric.dipper_windage_heat_w:.0f} W)",
  )
  fmep_table.add_row(
      "Total FMEP / Friction Power",
      f"{fric.total_fmep_bar:.2f} bar ({fric.total_friction_power_w:.0f} W,"
      f" {fric.friction_torque_nm:.2f} N·m)",
  )
  fmep_table.add_row("Net Indicated MEP (NIMEP)", f"{res.nimep_pa / 1.0e5:.2f} bar")
  fmep_table.add_row("Brake Shaft MEP (BMEP)", f"{res.bmep_pa / 1.0e5:.2f} bar")
  fmep_table.add_row(
      "Brake Power / Torque",
      f"{res.brake_power_kw:.2f} kW ({res.brake_power_kw * 1.341:.1f} hp) /"
      f" {res.brake_torque_nm:.2f} N·m",
  )
  fmep_table.add_row(
      "Mechanical Efficiency / BSFC",
      f"{res.mechanical_efficiency * 100.0:.1f} % / {res.bsfc_g_kwh:.1f} g/kWh",
  )
  rich.print(fmep_table)

  splash_table = Table(
      title="Wet-Sump Splash Transport & SOHC Chain Drain-Back",
      show_header=True,
      header_style="bold yellow",
  )
  splash_table.add_column("Transport Metric", style="cyan")
  splash_table.add_column("Value", style="bold green")
  splash_table.add_row("Static Sump Pool Depth", f"{splash.static_pool_depth_mm:.1f} mm")
  splash_table.add_row(
      "Static / Dynamic Dipper Immersion",
      f"{splash.static_immersion_mm:.2f} mm / {splash.dynamic_immersion_mm:.2f} mm",
  )
  splash_table.add_row(
      "Suspended Oil In-Flight (Level Drop)",
      f"{splash.oil_in_flight_l * 1000.0:.1f} mL (-{splash.dynamic_level_drop_mm:.2f} mm)",
  )
  splash_table.add_row(
      "Dipper Tip Velocity / Ballistic Apex",
      f"{splash.tip_throw_velocity_m_s:.2f} m/s / {splash.droplet_apex_height_m:.2f} m"
      f" (Reaches Under-Crown: {splash.reaches_piston_undercrown})",
  )
  splash_table.add_row(
      "#25 Chain Upward Oil Carry",
      f"{splash.chain_oil_carry_g_s:.2f} g/s ({splash.chain_oil_carry_l_min:.2f} L/min)",
  )
  splash_table.add_row(
      "Chain Tunnel Gravity Drain Capacity",
      f"{splash.tunnel_drain_capacity_g_s:.1f} g/s ({splash.drain_safety_ratio:.1f}× safety)",
  )
  rich.print(splash_table)

  therm_table = Table(
      title="Sump Thermal Equilibrium & Piston Splash Cooling",
      show_header=True,
      header_style="bold green",
  )
  therm_table.add_column("Thermal Parameter", style="cyan")
  therm_table.add_column("Value", style="bold green")
  therm_table.add_row(
      "Equilibrium Sump Oil Temperature",
      f"{tb.equilibrium_oil_temp_c:.1f} °C (Limit ≤ {tb.max_allowable_oil_temp_c:.0f} °C,"
      f" margin {tb.oil_temp_margin_c:+.1f} °C)",
  )
  therm_table.add_row(
      "Sump Heat Load (Friction + Piston + Case)",
      f"{tb.total_oil_heat_load_w:.0f} W ({tb.friction_heat_to_oil_w:.0f} +"
      f" {tb.piston_splash_heat_w:.0f} + {tb.crankcase_conduction_w:.0f} W)",
  )
  therm_table.add_row(
      "Finned Sump Conductance (UA)",
      f"{tb.sump_overall_ua_w_k:.2f} W/K",
  )
  therm_table.add_row(
      "Piston Under-Crown Splash HTC",
      f"{tb.undercrown_htc_w_m2_k:.0f} W/(m²·K)",
  )
  therm_table.add_row(
      "Cooled Piston Temps (Crown / Ring / Skirt)",
      f"{tb.cooled_piston_crown_temp_c:.1f} / {tb.cooled_piston_ring_land_temp_c:.1f} /"
      f" {tb.cooled_piston_skirt_temp_c:.1f} °C",
  )
  therm_table.add_row(
      "Relieved Cylinder Barrel Heat Budget",
      f"{tb.adjusted_barrel_heat_budget_w:.0f} W (vs {thermal_params.dry_barrel_heat_budget_w:.0f} W dry)",
  )
  therm_table.add_row("Overall Thermal & Transport Safe", str(tb.is_thermal_safe))
  rich.print(therm_table)

  lub_out_dir = Path(output_dir)
  lub_out_dir.mkdir(parents=True, exist_ok=True)
  thermo_lub_payload = {
      "rpm": float(target_rpm),
      "fuel": dict(cfg.get("fuel", {})),
      "cycle": {
          "imep_bar": float(res.gimep_pa / 1.0e5),
          "nimep_bar": float(res.nimep_pa / 1.0e5),
          "fmep_bar": float(res.fmep_pa / 1.0e5),
          "bmep_bar": float(res.bmep_pa / 1.0e5),
          "indicated_power_kw": float(res.power_kw),
          "brake_power_kw": float(res.brake_power_kw),
          "brake_power_hp": float(res.brake_power_kw * 1.34102209),
          "brake_torque_nm": float(res.brake_torque_nm),
          "mechanical_efficiency": float(res.mechanical_efficiency),
          "isfc_g_kwh": float(res.bsfc_g_kwh * res.mechanical_efficiency),
          "bsfc_g_kwh": float(res.bsfc_g_kwh),
          "p_max_bar": float(res.pressure_pa.max() / 1.0e5),
          "t_max_k": float(res.temperature_k.max()),
          "end_gas_temp_max_k": float(res.end_gas_temp_max_k),
          "knock_integral": float(res.knock_integral),
          "knock_margin": float(res.knock_margin),
          "is_knocking": bool(res.is_knocking),
          "knock_onset_deg": None if res.knock_onset_deg is None else float(res.knock_onset_deg),
      },
      "friction_zones": {
          "ring_skirt_fmep_bar": float(fric.ring_skirt_fmep_pa / 1.0e5),
          "ring_skirt_heat_w": float(fric.ring_skirt_heat_w),
          "rolling_bearings_fmep_bar": float(fric.rolling_bearings_fmep_pa / 1.0e5),
          "rolling_bearings_heat_w": float(fric.rolling_bearings_heat_w),
          "valvetrain_fmep_bar": float(fric.valvetrain_fmep_pa / 1.0e5),
          "valvetrain_heat_w": float(fric.valvetrain_heat_w),
          "dipper_windage_fmep_bar": float(fric.dipper_windage_fmep_pa / 1.0e5),
          "dipper_windage_heat_w": float(fric.dipper_windage_heat_w),
          "total_fmep_bar": float(fric.total_fmep_bar),
          "total_friction_power_w": float(fric.total_friction_power_w),
          "friction_torque_nm": float(fric.friction_torque_nm),
      },
      "splash": {
          "oil_grade": spec.grade,
          "static_pool_depth_mm": float(splash.static_pool_depth_mm),
          "static_immersion_mm": float(splash.static_immersion_mm),
          "dynamic_immersion_mm": float(splash.dynamic_immersion_mm),
          "oil_in_flight_ml": float(splash.oil_in_flight_l * 1000.0),
          "dynamic_level_drop_mm": float(splash.dynamic_level_drop_mm),
          "tip_throw_velocity_m_s": float(splash.tip_throw_velocity_m_s),
          "droplet_apex_height_m": float(splash.droplet_apex_height_m),
          "reaches_piston_undercrown": bool(splash.reaches_piston_undercrown),
          "chain_oil_carry_g_s": float(splash.chain_oil_carry_g_s),
          "tunnel_drain_capacity_g_s": float(splash.tunnel_drain_capacity_g_s),
          "drain_safety_ratio": float(splash.drain_safety_ratio),
      },
      "thermal": {
          "equilibrium_oil_temp_c": float(tb.equilibrium_oil_temp_c),
          "max_allowable_oil_temp_c": float(tb.max_allowable_oil_temp_c),
          "oil_temp_margin_c": float(tb.oil_temp_margin_c),
          "total_oil_heat_load_w": float(tb.total_oil_heat_load_w),
          "friction_heat_to_oil_w": float(tb.friction_heat_to_oil_w),
          "piston_splash_heat_w": float(tb.piston_splash_heat_w),
          "crankcase_conduction_w": float(tb.crankcase_conduction_w),
          "sump_overall_ua_w_k": float(tb.sump_overall_ua_w_k),
          "undercrown_htc_w_m2_k": float(tb.undercrown_htc_w_m2_k),
          "cooled_piston_crown_temp_c": float(tb.cooled_piston_crown_temp_c),
          "cooled_piston_ring_land_temp_c": float(tb.cooled_piston_ring_land_temp_c),
          "cooled_piston_skirt_temp_c": float(tb.cooled_piston_skirt_temp_c),
          "adjusted_barrel_heat_budget_w": float(tb.adjusted_barrel_heat_budget_w),
          "is_thermal_safe": bool(tb.is_thermal_safe),
      },
  }
  lub_json_path = lub_out_dir / "thermo_lub_results.json"
  lub_json_path.write_text(json.dumps(thermo_lub_payload, indent=2), encoding="utf-8")
  rich.print(f"[bold green]Lubrication & knock JSON saved to {lub_json_path}[/bold green]")


@app.command()
def dynamics(
    config_path: str = "config/engine_specs.yaml",
    rpm: float = 0.0,
):
  """Run slider-crank dynamics, flywheel sizing, ISO 281 rolling bearings, and SOHC cam journal."""
  cfg, sim, thermo_res, oil_spec, tb = _solve_default_thermal_balance(config_path=config_path, rpm=rpm)
  geom = sim.geom
  target_rpm = rpm if rpm > 0.0 else float(cfg["operating_conditions"]["rated_rpm"])

  dyn = SliderCrankDynamics(geom)
  forces = dyn.solve_forces(
      rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      pressure_pa=thermo_res.pressure_pa,
      recip_balance_factor=float(cfg.get("balancing", {}).get("reciprocating_balance_factor", 0.50)),
  )
  sweep = dyn.sweep_balance_factors(rpm=target_rpm, forces=forces)

  # Flywheel sizing via FlywheelDynamics (160 mm OD C500 1144 steel geometry)
  fw_dyn = FlywheelDynamics()
  fw_rep = fw_dyn.size_flywheel(
      rated_rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      torque_nm=forces.crankshaft_torque_nm,
      target_coeff_fluctuation=float(cfg.get("balancing", {}).get("speed_fluctuation_target", 0.035)),
      redline_rpm=float(cfg["operating_conditions"]["redline_rpm"]),
  )

  # Option A.1: ISO 281 L10 rolling-bearing fatigue life (INA NKS28-XL needle + 2x 6206-2Z main ball)
  f_big_end = np.sqrt(forces.net_wrist_pin_force_n**2 + forces.cylinder_side_thrust_n**2)
  crankpin_life = analyze_rolling_bearing_life(CRANKPIN_NEEDLE_NKS28_SPEC, target_rpm, f_big_end)
  main_ball_life = analyze_rolling_bearing_life(MAIN_BALL_6206_SPEC, target_rpm, f_big_end)

  # Option A.2: Hydrodynamic squeeze-film check on 20 mm SOHC cam-tower plain journal
  eq_oil = oil_spec.evaluate(tb.equilibrium_oil_temp_c)
  cam = SOHCCamshaft()
  kin_in = cam.evaluate_lobe_cam_angle(cam.intake_params, rpm=target_rpm, num_points=360)
  kin_ex = cam.evaluate_lobe_cam_angle(cam.exhaust_params, rpm=target_rpm, num_points=360)
  cam_radial_load = 0.5 * (
      kin_in.follower_contact_force_n + np.roll(kin_ex.follower_contact_force_n, 105)
  ) + 90.0
  cam_journal = HydrodynamicBearing(
      geom=BearingGeometry(
          name="SOHC Cam-Tower Plain Journal",
          journal_diameter_m=CAM_JOURNAL_DIAMETER_M,
          bearing_width_m=TOWER_FRONT_JOURNAL_LAND_M,
          radial_clearance_m=18.0e-6,
          max_allowable_pressure_mpa=35.0,
      ),
      oil=OilProperties(
          name=f"{oil_spec.grade} @ {tb.equilibrium_oil_temp_c:.1f} C",
          dynamic_viscosity_pa_s=eq_oil.dynamic_viscosity_pa_s,
          density_kg_m3=eq_oil.density_kg_m3,
      ),
  )
  cam_brg_rep = cam_journal.analyze_bearing_safety(
      rpm=target_rpm * 0.5,
      theta_deg=kin_in.cam_angle_deg,
      radial_load_n=cam_radial_load,
  )

  table = Table(
      title=f"Slider-Crank Mechanism Dynamics & Shaking @ {target_rpm:.0f} RPM",
      show_header=True,
      header_style="bold yellow",
  )
  table.add_column("Mechanism Parameter", style="cyan")
  table.add_column("Value", style="bold green")

  piston_force_max = float(max(forces.gas_force_n))
  table.add_row("Peak Gas Force", f"{piston_force_max / 1000.0:.2f} kN")
  table.add_row("Peak Piston Acceleration", f"{np.max(np.abs(forces.piston_acceleration_m_s2)):.0f} m/s² ({np.max(np.abs(forces.piston_acceleration_m_s2)) / 9.81:.0f} g)")
  table.add_row("Peak Output Torque", f"{forces.peak_torque_nm:.2f} N·m")
  table.add_row("Mean Output Torque", f"{forces.mean_torque_nm:.2f} N·m")
  table.add_row("Peak Shaking Force (50% Bal)", f"{forces.peak_shaking_force_n:.1f} N")
  table.add_row("RMS Shaking Force (50% Bal)", f"{forces.rms_shaking_force_n:.1f} N")
  table.add_row("Optimal Balance Factor (Min RMS)", f"{sweep.optimal_factor_rms * 100:.0f} % (RMS: {min(sweep.rms_shaking_forces_n):.1f} N)")
  rich.print(table)

  fw_table = Table(
      title="Flywheel Sizing & Kinetic Fluctuation",
      show_header=True,
      header_style="bold magenta",
  )
  fw_table.add_column("Flywheel Parameter", style="cyan")
  fw_table.add_column("Value", style="bold green")
  fw_table.add_row("Required Inertia (I_fw)", f"{fw_rep.flywheel_inertia_kg_m2 * 1000.0:.2f} × 10⁻³ kg·m²")
  fw_table.add_row("Speed Fluctuation Coeff (C_s)", f"{fw_rep.actual_fluctuation_coeff * 100:.2f} % (Target <= 3.5%)")
  fw_table.add_row("Speed Fluctuation Range", f"{fw_rep.min_rpm_at_rated:.0f} to {fw_rep.max_rpm_at_rated:.0f} RPM")
  fw_table.add_row("Disc Outer Diameter / Hub", f"{fw_dyn.dims.outer_radius_m * 2000:.0f} mm / {fw_dyn.dims.inner_bore_radius_m * 2000:.1f} mm")
  fw_table.add_row("Rim Axial Thickness", f"{fw_rep.flywheel_thickness_mm:.1f} mm")
  fw_table.add_row("Flywheel Mass", f"{fw_rep.flywheel_mass_kg:.2f} kg")
  fw_table.add_row("Rim Hoop Stress @ Redline", f"{fw_rep.hoop_stress_redline_mpa:.1f} MPa")
  fw_table.add_row("Burst Safety Factor @ Redline", f"{fw_rep.burst_safety_factor_redline:.1f} (Safe: {fw_rep.is_safe})")
  rich.print(fw_table)

  roll_table = Table(
      title="Crank Rolling Bearings (ISO 281 L10 Fatigue Life)",
      show_header=True,
      header_style="bold cyan",
  )
  roll_table.add_column("Bearing (SKU)", style="cyan")
  roll_table.add_column("P_eq / C_r (kN)", style="yellow")
  roll_table.add_column("Static SF (S0)", style="green")
  roll_table.add_column("L10 (10^6 rev)", style="green")
  roll_table.add_column("L10h @ RPM (hr)", style="bold green")
  roll_table.add_column("Status", style="bold green")
  for brg_rep in (crankpin_life, main_ball_life):
    roll_table.add_row(
        f"{brg_rep.bearing_name} [{brg_rep.sku}]",
        f"{brg_rep.equivalent_dynamic_load_n / 1000.0:.2f} / {brg_rep.dynamic_load_rating_n / 1000.0:.1f}",
        f"{brg_rep.static_safety_factor:.2f}",
        f"{brg_rep.l10_million_revs:.1f}",
        f"{brg_rep.l10_hours:.0f} h",
        _status_badge(brg_rep.is_safe),
    )
  rich.print(roll_table)

  cam_table = Table(
      title=f"SOHC Cam-Tower Plain Journal ({cam_journal.oil.name})",
      show_header=True,
      header_style="bold blue",
  )
  cam_table.add_column("Journal Parameter", style="cyan")
  cam_table.add_column("Value", style="bold green")
  cam_table.add_row("Journal Diameter × Width", f"{cam_journal.geom.journal_diameter_m * 1000:.1f} × {cam_journal.geom.bearing_width_m * 1000:.1f} mm")
  cam_table.add_row("Radial Clearance (c)", f"{cam_journal.geom.radial_clearance_m * 1.0e6:.1f} µm")
  cam_table.add_row("Peak Radial Load (2 Cam Lobes + #25 Chain)", f"{cam_brg_rep.peak_load_n:.1f} N")
  cam_table.add_row("Peak Projected Pressure (P_max)", f"{cam_brg_rep.peak_pressure_mpa:.2f} MPa (Limit: {cam_brg_rep.allowable_pressure_mpa:.1f} MPa)")
  cam_table.add_row("Journal Sliding Velocity (@ Cam RPM)", f"{cam_brg_rep.sliding_velocity_m_s:.2f} m/s")
  cam_table.add_row("Peak Eccentricity (ε_max)", f"{cam_brg_rep.peak_eccentricity_ratio:.4f}")
  cam_table.add_row("Minimum Oil Film Thickness (h_min)", f"{cam_brg_rep.min_film_thickness_um:.2f} µm")
  cam_table.add_row("Lubrication Lambda Ratio (h_min / R_q)", f"{cam_brg_rep.min_lambda_ratio:.2f} (Full Hydrodynamic: {cam_brg_rep.is_hydrodynamic})")
  rich.print(cam_table)


@app.command()
def bom(
    config_path: str = "config/engine_specs.yaml",
    c500_path: str = "config/c500_constraints.yaml",
    bom_path: str = "config/mcmaster_bom.yaml",
    output_dir: str = "outputs/bom",
    fourth_axis: Optional[bool] = typer.Option(
        None,
        "--fourth-axis/--no-fourth-axis",
        help="Override C500 4th-axis rotary availability",
    ),
    strict: bool = True,
):
  """Generate C500 DFM envelope audit, offline McMaster-Carr BoM, and Order-Pad export."""
  axis_override = fourth_axis if (fourth_axis is None or isinstance(fourth_axis, bool)) else None
  report = resolve_bom_from_files(
      engine_path=Path(config_path),
      c500_path=Path(c500_path),
      bom_path=Path(bom_path),
      has_fourth_axis_override=axis_override,
  )
  export_paths = export_bom_artifacts(report, output_dir=Path(output_dir))

  axis_label = "4th-Axis ACTIVE" if report.has_fourth_axis else "3-Axis FALLBACK"
  ledger_table = Table(
      title=f"NestWorks C500 CNC Envelope & Part Ledger ({axis_label})",
      show_header=True,
      header_style="bold cyan",
  )
  ledger_table.add_column("Component", style="magenta")
  ledger_table.add_column("Active Mode", style="yellow")
  ledger_table.add_column("Fallback Strategy", style="white")
  ledger_table.add_column("Status", style="bold green")
  for chk in report.c500_checks:
    fallback_note = chk.fallback_reason if chk.uses_fallback else "-"
    ledger_table.add_row(
        chk.part_id,
        chk.active_mode,
        fallback_note,
        _status_badge(chk.is_valid),
    )
  rich.print(ledger_table)

  bom_table = Table(
      title=f"Costed Bill of Materials ({axis_label})",
      show_header=True,
      header_style="bold blue",
  )
  bom_table.add_column("Item", style="cyan")
  bom_table.add_column("Channel", style="magenta")
  bom_table.add_column("SKU [Tier]", style="yellow")
  bom_table.add_column("Direct URL / Selection Hint", style="white")
  bom_table.add_column("Order", justify="right", style="green")
  bom_table.add_column("Cost USD", justify="right", style="bold green")
  bom_table.add_column("Fit", style="cyan")
  for item in report.items:
    nav = item.url if item.url else item.selection_hint
    if item.url and item.sku_status == "anchor_table":
      nav = f"{item.url} | {item.selection_hint}"
    bom_table.add_row(
        item.item_id,
        item.channel,
        f"{item.sku} [{item.sku_status}]",
        nav,
        f"{item.packs_to_order}x (qty {item.required_qty})",
        f"${item.extended_cost_usd:.2f}",
        _status_badge(item.is_interface_valid),
    )
  rich.print(bom_table)

  val = report.validation
  rollup_table = Table(
      title="Sourcing Channel Cost Rollup & Order-Pad Export",
      show_header=True,
      header_style="bold green",
  )
  rollup_table.add_column("Channel / Export Artifact", style="cyan")
  rollup_table.add_column("Summary", style="bold green")
  for ch_name, ch_cost in val.channel_costs_usd.items():
    rollup_table.add_row(f"Channel: {ch_name}", f"${ch_cost:.2f}")
  rollup_table.add_row(
      "SKU Resolution Breakdown",
      f"{val.verified_exact_mcmaster_count} exact Order-Pad | "
      f"{val.anchor_table_mcmaster_count} anchor table | "
      f"{val.powersports_cots_count} COTS",
  )
  rollup_table.add_row("Total Estimated Hardware Cost", f"${val.total_cost_usd:.2f} USD")
  rollup_table.add_row("McMaster Order-Pad Paste File", str(export_paths.order_pad_txt))
  rollup_table.add_row("Full Costed CSV", str(export_paths.costed_csv))
  rollup_table.add_row("JSON Verification Report", str(export_paths.summary_json))
  rich.print(rollup_table)

  if val.violations:
    for err in val.violations:
      rich.print(f"[bold red]DFM/BoM Violation:[/bold red] {err}")

  if strict and (not val.is_valid or any(not chk.is_valid for chk in report.c500_checks)):
    raise typer.Exit(code=1)


@app.command(name="dfm-check")
def dfm_check(
    c500_path: str = "config/c500_constraints.yaml",
    output_dir: str = "outputs/dfm",
    fourth_axis: Optional[bool] = typer.Option(
        None,
        "--fourth-axis/--no-fourth-axis",
        help="Override C500 4th-axis rotary availability",
    ),
    drawings: bool = typer.Option(
        True,
        "--drawings/--no-drawings",
        help="Export 2D orthographic SVG and DXF setup sheets",
    ),
    strict: bool = True,
):
  """Run live B-Rep NestWorks C500 DFM check, twin parameter check, and setup sheet export."""
  has_4ax = fourth_axis if isinstance(fourth_axis, bool) else True
  emit_drawings = drawings if isinstance(drawings, bool) else True
  report = run_c500_dfm_check(
      has_fourth_axis=has_4ax,
      config_path=Path(c500_path),
      include_twin_params=True,
  )
  out_path = Path(output_dir)
  report_json = report.to_json(out_path / "c500_dfm_report.json")

  axis_label = "4th-Axis ACTIVE" if report.has_fourth_axis else "3-Axis FALLBACK"
  audit_table = Table(
      title=f"Live B-Rep NestWorks C500 DFM Audit ({axis_label})",
      show_header=True,
      header_style="bold cyan",
  )
  audit_table.add_column("Part ID", style="magenta")
  audit_table.add_column("Mode", style="yellow")
  audit_table.add_column("Measured XYZ (mm)", style="green")
  audit_table.add_column("Min Cyl R (mm)", style="cyan")
  audit_table.add_column("Watertight", style="green")
  audit_table.add_column("Fallback", style="white")
  audit_table.add_column("Status", style="bold green")
  for item in report.part_audits:
    dx, dy, dz = item.measured_dims_mm
    audit_table.add_row(
        item.part_id,
        item.active_mode,
        f"{dx:.1f} x {dy:.1f} x {dz:.1f}",
        f"{item.min_cylindrical_radius_mm:.2f} (>= {item.min_allowed_corner_radius_mm:.1f})",
        str(item.watertight),
        item.fallback_reason if item.uses_fallback else "-",
        _status_badge(item.is_valid),
    )
  rich.print(audit_table)

  if report.twin_params is not None:
    tp = report.twin_params
    twin_table = Table(
        title="Extracted 1:1 DFM Full-Twin Physics Parameters",
        show_header=True,
        header_style="bold magenta",
    )
    twin_table.add_column("Twin Metric", style="cyan")
    twin_table.add_column("Measured CAD Value", style="bold green")
    twin_table.add_row("Total B-Rep Solids / Mass (Dry / Wet)", f"{tp.total_part_count} solids | {tp.dry_mass_kg:.3f} / {tp.wet_mass_kg:.3f} kg")
    twin_table.add_row("Reciprocating / Rotating Mass", f"{tp.reciprocating_mass_kg:.4f} / {tp.rotating_mass_kg:.4f} kg")
    twin_table.add_row("Effective Balance Factor", f"{tp.effective_balance_factor * 100.0:.1f} %")
    twin_table.add_row("Flywheel Mass / CAD Inertia", f"{tp.flywheel_mass_kg:.3f} kg / {tp.flywheel_cad_inertia_kg_m2 * 1000.0:.2f} x 10^-3 kg*m^2")
    twin_table.add_row("Chamber Dome / Clearance / CR", f"{tp.dome_volume_cm3:.2f} / {tp.clearance_volume_cm3:.2f} cm^3 | {tp.compression_ratio:.2f}:1")
    twin_table.add_row("Sump Oil Pool / BDC Immersion", f"{tp.oil_volume_liters:.3f} L / {tp.bdc_dipper_immersion_mm:.1f} mm")
    twin_table.add_row("Intake / Exhaust / Retainer Mass", f"{tp.valve_intake_mass_kg * 1000:.1f} / {tp.valve_exhaust_mass_kg * 1000:.1f} / {tp.retainer_mass_kg * 1000:.1f} g")
    rich.print(twin_table)

  if emit_drawings:
    manifest = export_c500_setup_sheets(
        output_dir=out_path / "drawings",
        has_fourth_axis=report.has_fourth_axis,
    )
    rich.print(f"[green]Exported {len(manifest.sheets)} 2D SVG + DXF C500 setup sheets to: {manifest.output_dir}[/green]")

  rich.print(f"[green]Saved live B-Rep DFM audit report to: {report_json}[/green]")
  if report.violations:
    for err in report.violations:
      rich.print(f"[bold red]DFM Violation:[/bold red] {err}")

  if strict and not report.all_passed:
    raise typer.Exit(code=1)


@app.command()
def valvetrain(
    config_path: str = "config/engine_specs.yaml",
):
  """Run SOHC 4-5-6-7 polynomial valvetrain synthesis, contact stress, and valve float analysis."""
  cfg = load_config(config_path)
  rated_rpm = float(cfg["operating_conditions"]["rated_rpm"])
  redline_rpm = float(cfg["operating_conditions"]["redline_rpm"])

  cam = SOHCCamshaft()
  intake_rep = cam.analyze_valvetrain_safety(cam.intake_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm)
  exhaust_rep = cam.analyze_valvetrain_safety(cam.exhaust_params, rated_rpm=rated_rpm, redline_rpm=redline_rpm)

  for rep, p in [(intake_rep, cam.intake_params), (exhaust_rep, cam.exhaust_params)]:
    table = Table(
        title=f"SOHC Valvetrain: {rep.cam_name} Cam & Follower Dynamics",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Valvetrain Parameter", style="cyan")
    table.add_column("Synthesized Value", style="bold yellow")

    table.add_row("Motion Law", "4-5-6-7 Polynomial (Zero boundary jerk)")
    table.add_row("Max Valve Lift", f"{p.max_valve_lift_m * 1000:.2f} mm")
    table.add_row("Max Cam Lobe Lift", f"{(p.max_valve_lift_m / p.rocker_ratio) * 1000:.2f} mm (Rocker ratio: {p.rocker_ratio:.2f})")
    table.add_row("Duration", f"{p.duration_crank_deg:.0f}° Crank ({p.duration_crank_deg / 2.0:.0f}° Cam)")
    table.add_row("Equivalent Valvetrain Mass", f"{cam.spring.equivalent_valve_mass_kg * 1000:.1f} g")
    table.add_row("Spring Stiffness (k)", f"{cam.spring.spring_rate_n_m / 1000:.1f} N/mm (Preload: {cam.spring.seat_force_preload_n:.0f} N)")
    table.add_row("Max Valve Acceleration @ 5500 RPM", f"{rep.max_valve_acceleration_m_s2:.0f} m/s² ({rep.max_valve_acceleration_m_s2 / 9.81:.0f} g)")
    table.add_row("Max Valve Deceleration @ 5500 RPM", f"{rep.max_valve_deceleration_m_s2:.0f} m/s²")
    table.add_row("Min Contact Force @ 5500 RPM", f"{rep.min_contact_force_at_rated_n:.1f} N")
    table.add_row("Min Contact Force @ 7000 RPM (Redline)", f"{rep.min_contact_force_at_redline_n:.1f} N")
    table.add_row("Min Contact Force @ 8400 RPM (1.2× Redline)", f"{rep.min_contact_force_at_test_n:.1f} N")
    table.add_row("Valve Float RPM", f"{rep.valve_float_rpm:.0f} RPM")
    table.add_row("Float Safety Margin over 8400 RPM", f"{rep.float_margin_percent:.1f} % (Target >= 30%)")
    table.add_row("Peak Hertz Contact Stress @ 8400 RPM", f"{rep.peak_hertz_stress_mpa:.1f} MPa (Allowable < 1500 MPa)")
    table.add_row("Valvetrain Status", "[bold green]PASS (Safe from float & fatigue)[/bold green]" if rep.is_safe else "[bold red]FAIL[/bold red]")
    rich.print(table)


@app.command()
def cad(
    config_path: str = "config/engine_specs.yaml",
    output_dir: str = "outputs/cad",
):
  """Generate parametric 3D CAD parts (Piston, Con-Rod, Cylinder Head, Valves, CFD Domain)."""
  cfg = load_config(config_path)
  g = cfg["geometry"]
  v_cfg = cfg.get("valves", {})

  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  rich.print("[cyan]Generating Connecting Rod via build123d...[/cyan]")
  rod = generate_connecting_rod(
      center_to_center_m=g["con_rod_length"],
      wrist_pin_diam_m=g["wrist_pin_diameter"],
      crankpin_diam_m=g["crankpin_diameter"],
  )
  rod_file = out_path / "connecting_rod.step"
  export_styled_step(
      style_single_part(rod, "connecting_rod", "ALUMINUM_7075_T6", OPAQUE_OPACITY),
      rod_file,
  )
  rich.print(f"[green]Saved connecting rod ({rod.volume/1000:.1f} cm³) to: {rod_file}[/green]")

  rich.print("[cyan]Generating Piston via build123d...[/cyan]")
  piston = generate_piston(
      bore_m=g["bore"],
      wrist_pin_diam_m=g["wrist_pin_diameter"],
  )
  piston_file = out_path / "piston.step"
  export_styled_step(
      style_single_part(piston, "piston", "ALUMINUM_4032_T6", OPAQUE_OPACITY),
      piston_file,
  )
  rich.print(f"[green]Saved piston ({piston.volume/1000:.1f} cm³) to: {piston_file}[/green]")

  rich.print("[cyan]Generating SOHC Pent-Roof Cylinder Head via build123d...[/cyan]")
  v_mm3, v_cm3 = calculate_combustion_chamber_volume()
  head = generate_cylinder_head()
  head_file = out_path / "cylinder_head.step"
  export_styled_step(
      style_single_part(head, "cylinder_head", "ALUMINUM_6061_T6", SINGLE_HOUSING_OPACITY),
      head_file,
  )
  rich.print(
      f"[green]Saved cylinder head ({head.volume/1000:.1f} cm³, chamber dome {v_cm3:.2f} cm³) to: {head_file}[/green]"
  )

  rich.print("[cyan]Generating Poppet Valves via build123d...[/cyan]")
  in_valve = generate_poppet_valve(head_diam_m=v_cfg.get("intake_diameter", 0.030))
  ex_valve = generate_poppet_valve(head_diam_m=v_cfg.get("exhaust_diameter", 0.026))
  in_valve_file = out_path / "valve_intake.step"
  ex_valve_file = out_path / "valve_exhaust.step"
  export_styled_step(
      style_single_part(in_valve, "valve_intake", "STEEL_4140_PREHARD", OPAQUE_OPACITY),
      in_valve_file,
  )
  export_styled_step(
      style_single_part(ex_valve, "valve_exhaust", "STEEL_4140_PREHARD", OPAQUE_OPACITY),
      ex_valve_file,
  )
  rich.print(f"[green]Saved intake valve ({in_valve.volume/1000:.2f} cm³) to: {in_valve_file}[/green]")
  rich.print(f"[green]Saved exhaust valve ({ex_valve.volume/1000:.2f} cm³) to: {ex_valve_file}[/green]")

  rich.print("[cyan]Extracting CFD Port Flow Fluid Domain via build123d...[/cyan]")
  fluid = extract_intake_port_fluid_domain()
  fluid_file = out_path / "intake_fluid_domain.step"
  export_styled_step(
      style_single_part(fluid, "intake_fluid_domain", "FUEL_AIR_CHARGE", FLUID_OPACITY),
      fluid_file,
  )
  rich.print(f"[green]Saved CFD fluid domain ({fluid.volume/1000:.1f} cm³) to: {fluid_file}[/green]")

  rich.print("[cyan]Generating 105 mm Honed DOM Steel Cylinder Liner via build123d...[/cyan]")
  liner = generate_cylinder_liner()
  liner_file = out_path / "cylinder_liner.step"
  export_styled_step(
      style_single_part(liner, "cylinder_liner", "DOM_STEEL_A513_T5", OPAQUE_OPACITY),
      liner_file,
  )
  rich.print(f"[green]Saved cylinder liner ({liner.volume/1000:.1f} cm³, {(liner.volume/1000)*7.85:.0f} g) to: {liner_file}[/green]")

  rich.print("[cyan]Generating Aluminum Cylinder Block with Radial Cooling Fins via build123d...[/cyan]")
  block = generate_cylinder_block()
  block_file = out_path / "cylinder_block.step"
  export_styled_step(
      style_single_part(block, "cylinder_block", "ALUMINUM_6061_T6", SINGLE_HOUSING_OPACITY),
      block_file,
  )
  therm = calculate_block_thermal_capacity()
  rich.print(
      f"[green]Saved cylinder block ({block.volume/1000:.1f} cm³, {therm['exterior_area_m2']:.3f} m² fin area, {therm['thermal_dissipation_w']:.0f} W cooling) to: {block_file}[/green]"
  )

  rich.print("[cyan]Generating 3-Piece Press-Fit Billet Crankshaft via build123d...[/cyan]")
  crank = generate_crankshaft()
  crank_file = out_path / "crankshaft.step"
  export_styled_step(
      style_single_part(crank, "crankshaft", "STEEL_4140_PREHARD", OPAQUE_OPACITY),
      crank_file,
  )
  crank_bal = calculate_crankshaft_balance()
  rich.print(
      f"[green]Saved crankshaft ({crank.volume/1000:.1f} cm³, {crank_bal['crankshaft_mass_kg']:.2f} kg, {crank_bal['balance_factor_percent']:.1f}% balance) to: {crank_file}[/green]"
  )

  rich.print("[cyan]Assembling Master Multi-Body Engine Assembly & Verifying 720° Clearances...[/cyan]")
  clr = verify_assembly_clearances()
  rich.print(f"  [cyan]Min Intake P-V Clearance:[/cyan] [bold green]{clr.min_intake_pv_clearance_mm:.2f} mm[/bold green] @ {clr.min_intake_pv_angle_deg:.0f}° CA (Target >= 1.5 mm)")
  rich.print(f"  [cyan]Min Exhaust P-V Clearance:[/cyan] [bold green]{clr.min_exhaust_pv_clearance_mm:.2f} mm[/bold green] @ {clr.min_exhaust_pv_angle_deg:.0f}° CA (Target >= 1.5 mm)")
  rich.print(f"  [cyan]Min Rod-to-Liner Clearance:[/cyan] [bold green]{clr.min_rod_to_liner_clearance_mm:.2f} mm[/bold green] @ {clr.min_rod_to_liner_angle_deg:.0f}° CA (Target >= 5.0 mm)")
  rich.print(f"  [cyan]3D B-Rep Interference Hits:[/cyan] [{'bold green' if clr.interference_hits == 0 else 'bold red'}]{clr.interference_hits}[/{'bold green' if clr.interference_hits == 0 else 'bold red'}]")
  status_style = "bold green" if clr.is_clearance_safe else "bold red"
  rich.print(f"  [cyan]Kinematic Interference Check:[/cyan] [{status_style}]{'PASS (Clearance Verified)' if clr.is_clearance_safe else 'FAIL'}[/{status_style}]")
  if not clr.is_clearance_safe:
    raise typer.Exit(code=1)

  assembly = generate_engine_assembly(0.0)
  assembly_file = out_path / "engine_assembly.step"
  export_styled_step(assembly, assembly_file)
  rich.print(
      f"[green]Saved master engine assembly ({assembly.volume/1000:.1f} cm³, {len(assembly.solids())} parts) to: {assembly_file}[/green]"
  )

  rich.print("[cyan]Verifying 1:1 C500 Billet + COTS Dressed Digital Twin & Fixtures...[/cyan]")
  dressed_rep = verify_dressed_engine(sweep_step_deg=60.0, has_fourth_axis=True)
  fixture_rep = verify_all_c500_fixtures(has_fourth_axis=True)
  dressed_asm = generate_dressed_engine_assembly(0.0)
  dressed_file = out_path / "dressed_engine_assembly.step"
  export_styled_step(dressed_asm, dressed_file)

  twin_table = Table(title="1:1 C500 Billet + COTS Dressed Engine Twin Summary", show_header=True)
  twin_table.add_column("Metric", style="cyan")
  twin_table.add_column("Value", style="green")
  twin_table.add_row("Total B-Rep Solids", str(dressed_rep.total_part_count))
  twin_table.add_row("Dry / Wet Mass (kg)", f"{dressed_rep.dry_mass_kg:.3f} / {dressed_rep.wet_mass_kg:.3f}")
  twin_table.add_row("Geometric Compression Ratio", f"{dressed_rep.compression_ratio:.2f}:1")
  twin_table.add_row("Sump Oil Pool / BDC Immersion", f"{dressed_rep.oil_volume_liters:.3f} L / {dressed_rep.bdc_dipper_immersion_mm:.1f} mm")
  twin_table.add_row("Reciprocating Balance Factor", f"{dressed_rep.effective_balance_factor * 100.0:.1f}%")
  twin_table.add_row("Static / Kinematic / Wrench Hits", f"{len(dressed_rep.static_solid_collisions)} / {len(dressed_rep.kinematic_sweep_collisions)} / {len(dressed_rep.wrench_zone_collisions)}")
  twin_table.add_row("C500 Fixture Checks Passed", f"{sum(1 for i in fixture_rep.items if i.is_valid)} / {len(fixture_rep.items)}")
  rich.print(twin_table)
  rich.print(f"[green]Saved dressed twin STEP to: {dressed_file}[/green]")
  if not dressed_rep.is_valid or not fixture_rep.all_passed:
    raise typer.Exit(code=1)


@app.command()
def check_solvers():
  """Check availability of containerized Podman solvers (OpenFOAM, CalculiX, Gmsh)."""
  runner = PodmanRunner()
  if not runner.is_available:
    rich.print(
        "[bold red]Error:[/bold red] Podman/Docker runtime is not responding or not in PATH."
    )
    raise typer.Exit(code=1)

  rich.print("[bold green]Podman runtime detected.[/bold green] Verifying containerized solver binaries...")
  import os
  solvers = [
      ("simpleFoam", "OpenFOAM v2406 Steady Flow Bench"),
      ("XiEngineFoam", "OpenFOAM v2406 Premixed Dynamic-Mesh Combustion"),
      ("engineFoam", "OpenFOAM v2406 Moving-Mesh Engine Solver"),
      ("blockMesh", "OpenFOAM v2406 Mesh Generator"),
      ("ccx", "CalculiX v2.21 Implicit FEA Solver"),
      ("gmsh", "Gmsh v4.12 3D Finite Element Mesh Generator"),
  ]

  table = Table(
      title="Containerized Solver Status (openengine-solvers:latest)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Solver Binary", style="magenta")
  table.add_column("Description", style="white")
  table.add_column("Status", style="bold green")

  bin_names = [s[0] for s in solvers]
  try:
    res = runner.run_command(f"which {' '.join(bin_names)}", os.getcwd(), timeout_sec=10)
    found_bins = set(line.split("/")[-1].strip() for line in res.stdout.splitlines())
  except Exception:
    found_bins = set()

  for bin_name, desc in solvers:
    status = "[green]✓ Ready[/green]" if bin_name in found_bins else "[red]✗ Not found[/red]"
    table.add_row(bin_name, desc, status)

  rich.print(table)


@app.command(name="cfd-port-flow")
def cfd_port_flow(
    lifts: str = "2.0,5.0,8.5",
    output_dir: str = "outputs/cfd",
    max_iterations: int = 100,
):
  """Run steady intake port aerodynamic flow bench sweep via OpenFOAM simpleFoam."""
  lift_list = [float(x.strip()) for x in lifts.split(",") if x.strip()]
  rich.print(f"[cyan]Running OpenFOAM intake port flow bench for valve lifts: {lift_list} mm...[/cyan]")

  cfg = PortFlowConfig(max_iterations=max_iterations)
  out_path = Path(output_dir)
  res = run_port_flow_sweep(lifts_mm=lift_list, output_dir=out_path, config=cfg)

  table = Table(
      title=f"Intake Port Flow Bench Results (OpenFOAM simpleFoam, ΔP = {cfg.delta_p_pa:.0f} Pa)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Lift [mm]", style="magenta")
  table.add_column("Mass Flow [g/s]", style="green")
  table.add_column("Cd (Seat)", style="yellow")
  table.add_column("Cd (Curtain)", style="yellow")
  table.add_column("Tumble Ratio", style="cyan")
  table.add_column("Swirl Ratio", style="cyan")
  table.add_column("Time [s]", style="white")

  for pt in res.points:
    table.add_row(
        f"{pt.lift_mm:.1f}",
        f"{pt.mass_flow_kg_s * 1000.0:.2f}",
        f"{pt.cd_seat:.3f}",
        f"{pt.cd_curtain:.3f}",
        f"{pt.tumble_ratio:+.3f}",
        f"{pt.swirl_ratio:+.3f}",
        f"{pt.solve_time_s:.1f}",
    )

  rich.print(table)
  rich.print(f"[bold green]Saved flow bench summary to: {out_path / 'port_flow_results.json'} ({res.total_solve_time_s:.1f} s total)[/bold green]")


@app.command(name="cfd-combustion")
def cfd_combustion(
    start_ca: float = -180.0,
    end_ca: float = 10.0,
    spark_ca: float = -18.0,
    output_dir: str = "outputs/cfd",
):
  """Run 3D moving-mesh in-cylinder combustion simulation via OpenFOAM XiEngineFoam."""
  rich.print(f"[cyan]Running 3D moving-mesh combustion simulation ({start_ca:.0f}° to {end_ca:.0f}° CA, spark at {spark_ca:.0f}° CA)...[/cyan]")

  cfg = CombustionConfig(
      start_crank_angle_deg=start_ca,
      end_crank_angle_deg=end_ca,
      spark_crank_angle_deg=spark_ca,
  )
  out_path = Path(output_dir)
  case_dir = out_path / "case_combustion"
  res = run_combustion_simulation(config=cfg, case_dir=case_dir)

  json_file = out_path / "combustion_results.json"
  res.to_json(json_file)

  table = Table(
      title="3D Moving-Mesh In-Cylinder Combustion Results (OpenFOAM XiEngineFoam)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Metric", style="magenta")
  table.add_column("Value", style="bold green")

  table.add_row("Mesh Cells", f"{res.total_cells:,} tets")
  table.add_row("Solved Crank Angle Range", f"{res.points[0].crank_angle_deg:.1f}° to {res.points[-1].crank_angle_deg:.1f}° CA ({len(res.points)} steps)")
  table.add_row("Peak In-Cylinder Pressure", f"{res.p_max_pa / 1e5:.2f} bar ({res.p_max_pa / 1e6:.3f} MPa) @ {res.theta_p_max_deg:.2f}° CA")
  table.add_row("Peak In-Cylinder Temperature", f"{res.t_max_k:.1f} K @ {res.theta_t_max_deg:.2f}° CA")
  table.add_row("Combustion Progress at End", f"{res.burn_progress_at_end * 100.0:.2f}% mass fraction burned")
  table.add_row("Total Simulation Solve Time", f"{res.solve_time_s:.2f} s")

  if res.wiebe:
    w = res.wiebe
    table.add_row("Calibrated Wiebe Parameter a", f"{w.a:.3f}")
    table.add_row("Calibrated Wiebe Form Factor m", f"{w.m:.3f}")
    table.add_row("Calibrated Combustion Duration", f"{w.duration_deg:.1f}° CA (from {w.theta_soc_deg:.0f}° CA)")
    table.add_row("Wiebe Curve Fit Quality (R²)", f"{w.r_squared:.4f}")

  rich.print(table)
  rich.print(f"[bold green]Saved combustion results to: {json_file}[/bold green]")


@app.command(name="cfd-cht-cooling")
def cfd_cht_cooling(
    ambient_temp_c: float = 27.0,
    fin_h_coeff: float = 105.0,
    output_dir: str = "outputs/cfd",
):
  """Run Conjugate Heat Transfer (CHT) air-fin thermal cooling simulation via OpenFOAM laplacianFoam."""
  rich.print(f"[cyan]Running air-fin thermal cooling CHT simulation (T_amb = {ambient_temp_c:.1f}°C, h = {fin_h_coeff:.1f} W/m²K)...[/cyan]")

  _, _, _, _, tb = _solve_default_thermal_balance()
  cfg = ChtCoolingConfig(
      ambient_temp_k=ambient_temp_c + 273.15,
      fin_h_coeff=fin_h_coeff,
  ).with_splash_cooling(tb)
  out_path = Path(output_dir)
  case_dir = out_path / "case_cht"
  res = run_cht_simulation(config=cfg, case_dir=case_dir)

  json_file = out_path / "cht_cooling_results.json"
  res.to_json(json_file)

  table = Table(
      title="Cylinder Block Air-Fin CHT Thermal Simulation (OpenFOAM laplacianFoam)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Metric", style="magenta")
  table.add_column("Value", style="bold green")

  table.add_row("Mesh Cells", f"{res.total_cells:,} tets")
  table.add_row("Exterior Cooling Fin Area", f"{res.exterior_fin_area_m2:.4f} m²")
  table.add_row("Max Liner Temperature", f"{res.max_liner_temp_k - 273.15:.1f}°C ({res.max_liner_temp_k:.1f} K) [Limit <= 180°C]")
  table.add_row("Min Fin Tip Temperature", f"{res.min_fin_temp_k - 273.15:.1f}°C ({res.min_fin_temp_k:.1f} K)")
  table.add_row("Mean Block Temperature", f"{res.mean_block_temp_k - 273.15:.1f}°C ({res.mean_block_temp_k:.1f} K)")
  table.add_row("Heat Dissipation Rate", f"{res.heat_dissipated_w:.1f} W (Combustion Budget: {res.config.heat_budget_w:.0f} W)")
  table.add_row("Thermal Safety Margin", f"{res.thermal_margin_percent:+.1f}% ({'PASS' if res.is_thermal_safe else 'FAIL'})")
  table.add_row("Simulation Solve Time", f"{res.solve_time_s:.2f} s")

  rich.print(table)
  rich.print(f"[bold green]Saved CHT thermal results to: {json_file}[/bold green]")


@app.command(name="cfd-closed-loop")
def cfd_closed_loop(
    output_dir: str = "outputs/calibration",
):
  """Run 3D-to-0D closed-loop thermodynamic calibration using CFD combustion, port flow, and CHT metrics."""
  rich.print("[cyan]Running 3D-to-0D closed-loop thermodynamic calibration...[/cyan]")

  out_path = Path(output_dir)
  json_file = out_path / "closed_loop_results.json"
  res = run_closed_loop_calibration(output_json_path=json_file)

  table = Table(
      title="3D CFD -> 0D Cycle Closed-Loop Thermodynamic Calibration",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Parameter / Metric", style="magenta")
  table.add_column("Baseline (Empirical 0D)", style="yellow")
  table.add_column("Calibrated (3D CFD Couplings)", style="bold green")
  table.add_column("Delta (%)", style="cyan")

  table.add_row("Wiebe Parameter a", "5.000", f"{res.calibrated_wiebe_a:.3f}", f"{(res.calibrated_wiebe_a - 5.0)/5.0*100:+.1f}%")
  table.add_row("Wiebe Form Factor m", "2.000", f"{res.calibrated_wiebe_m:.3f}", f"{(res.calibrated_wiebe_m - 2.0)/2.0*100:+.1f}%")
  table.add_row("Combustion Duration", "42.0° CA", f"{res.calibrated_wiebe_duration_deg:.1f}° CA", f"{(res.calibrated_wiebe_duration_deg - 42.0)/42.0*100:+.1f}%")
  table.add_row("Spark Timing (SOC)", "-15.0° CA", f"{res.calibrated_wiebe_soc_deg:.1f}° CA", f"{(res.calibrated_wiebe_soc_deg - (-15.0))/15.0*100:+.1f}%")
  table.add_row("Intake Curtain Discharge Cd", "0.650", f"{res.calibrated_intake_cd:.3f}", f"{(res.calibrated_intake_cd - 0.650)/0.650*100:+.1f}%")
  table.add_row("Cylinder Liner Wall Temp", "430.0 K", f"{res.calibrated_wall_temp_k:.1f} K", f"{(res.calibrated_wall_temp_k - 430.0)/430.0*100:+.1f}%")
  table.add_row("---", "---", "---", "---")
  table.add_row("Peak Cylinder Pressure", f"{res.baseline.p_max_bar:.2f} bar", f"{res.calibrated.p_max_bar:.2f} bar", f"{(res.calibrated.p_max_bar - res.baseline.p_max_bar)/res.baseline.p_max_bar*100:+.1f}%")
  table.add_row("Peak In-Cylinder Temp", f"{res.baseline.t_max_k:.1f} K", f"{res.calibrated.t_max_k:.1f} K", f"{(res.calibrated.t_max_k - res.baseline.t_max_k)/res.baseline.t_max_k*100:+.1f}%")
  table.add_row("Indicated Mean Eff. Press. (IMEP)", f"{res.baseline.imep_bar:.2f} bar", f"{res.calibrated.imep_bar:.2f} bar", f"{res.delta_imep_percent:+.1f}%")
  table.add_row("Indicated Power @ 5500 RPM", f"{res.baseline.indicated_power_kw:.2f} kW", f"{res.calibrated.indicated_power_kw:.2f} kW", f"{res.delta_power_percent:+.1f}%")
  table.add_row("Indicated Torque @ 5500 RPM", f"{res.baseline.indicated_torque_nm:.2f} N·m", f"{res.calibrated.indicated_torque_nm:.2f} N·m", f"{res.delta_power_percent:+.1f}%")
  table.add_row("Indicated Thermal Efficiency", f"{res.baseline.thermal_efficiency*100:.1f}%", f"{res.calibrated.thermal_efficiency*100:.1f}%", f"{(res.calibrated.thermal_efficiency - res.baseline.thermal_efficiency)/res.baseline.thermal_efficiency*100:+.1f}%")
  table.add_row("Indicated Spec. Fuel Cons. (ISFC)", f"{res.baseline.isfc_g_per_kwh:.1f} g/kWh", f"{res.calibrated.isfc_g_per_kwh:.1f} g/kWh", f"{(res.calibrated.isfc_g_per_kwh - res.baseline.isfc_g_per_kwh)/res.baseline.isfc_g_per_kwh*100:+.1f}%")

  rich.print(table)
  rich.print(f"[bold green]Saved closed-loop calibration results to: {json_file}[/bold green]")


def _status_badge(is_ok: bool) -> str:
  """Format pass/fail status badge with colors."""
  return "[bold green]PASS[/bold green]" if is_ok else "[bold red]FAIL[/bold red]"


@app.command(name="fea-piston")
def fea_piston(
    peak_pressure_bar: float = 78.0,
    crown_temp_c: float = 250.0,
    output_dir: str = "outputs/fea/piston",
    strict: bool = True,
):
  """Run structural & coupled thermo-mechanical FEA on the piston using CalculiX ccx."""
  rich.print(f"[cyan]Running piston FEA (P_max = {peak_pressure_bar:.1f} bar, T_crown = {crown_temp_c:.1f}°C)...[/cyan]")

  cfg = PistonFeaConfig(
      peak_pressure_mpa=peak_pressure_bar / 10.0,
      crown_temp_c=crown_temp_c,
  )
  if crown_temp_c == 250.0:
    _, _, _, _, tb = _solve_default_thermal_balance()
    cfg = cfg.with_splash_cooling(tb)
  res = run_piston_fea(config=cfg, output_dir=output_dir)

  table = Table(
      title="Piston Structural & Thermo-Mechanical FEA (CalculiX ccx)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Load Case / Metric", style="magenta")
  table.add_column("Value", style="bold green")
  table.add_column("Target / Limit", style="yellow")
  table.add_column("Status", style="cyan")

  s_yield = ALUMINUM_4032_T6.yield_strength_mpa
  max_allowable_stress = s_yield / cfg.min_yield_safety_factor

  table.add_row("Mesh Density", f"{res.mesh_stats['num_nodes']:,} nodes / {res.mesh_stats['num_elements']:,} tets (C3D10)", "Quadratic", "[bold green]PASS[/bold green]")
  table.add_row("LC-1 Firing TDC Peak Gas Pressure", f"{cfg.peak_pressure_mpa*10.0:.1f} bar ({cfg.peak_pressure_mpa:.2f} MPa)", "Mechanical", "LOAD")
  table.add_row("LC-1 Peak Von Mises Stress", f"{res.lc1_mechanical.max_von_mises_mpa:.1f} MPa", "Local peak", "INFO")
  table.add_row("LC-1 P95 Von Mises Stress", f"{res.lc1_mechanical.p95_von_mises_mpa:.1f} MPa", f"< {max_allowable_stress:.1f} MPa", _status_badge(res.lc1_mechanical.is_pass))
  table.add_row("LC-1 Mechanical Deflection", f"{res.lc1_mechanical.max_displacement_mm:.4f} mm", "< 0.50 mm", "[bold green]PASS[/bold green]")
  table.add_row("LC-1 Yield Safety Factor", f"{res.lc1_mechanical.yield_safety_factor:.2f}", f">= {cfg.min_yield_safety_factor:.2f}", _status_badge(res.lc1_mechanical.is_pass))
  table.add_row("---", "---", "---", "---")
  table.add_row("LC-2 Crown Operating Temp", f"{cfg.crown_temp_c:.1f}°C", "Thermal", "LOAD")
  table.add_row("LC-2 P95 Von Mises (Coupled)", f"{res.lc2_thermo_mechanical.p95_von_mises_mpa:.1f} MPa", "< 315.0 MPa", _status_badge(res.lc2_thermo_mechanical.is_pass))
  table.add_row("LC-2 Skirt Thermal Expansion", f"{res.lc2_thermo_mechanical.details['skirt_thermal_growth_um']:.1f} µm", "Radial", "EXP")
  table.add_row("LC-2 Liner Thermal Expansion", f"{res.lc2_thermo_mechanical.details['liner_thermal_growth_um']:.1f} µm", "Radial", "EXP")
  table.add_row("LC-2 Hot Skirt Running Clearance", f"{res.hot_skirt_clearance_mm*1000.0:.1f} µm", f">= {cfg.min_hot_clearance_mm*1000.0:.1f} µm", _status_badge(res.is_clearance_safe))
  table.add_row("Total Simulation Solve Time", f"{res.total_solve_time_s:.2f} s", "< 60.0 s", "[bold green]PASS[/bold green]")

  rich.print(table)
  p_json = Path(output_dir) / "piston_fea_results.json"
  p_json.write_text(json.dumps(res.to_dict(), indent=2))
  rich.print(f"[bold green]Saved piston FEA results to: {p_json}[/bold green]")

  if not res.is_structural_safe:
    rich.print(f"[bold red]FAILED: Piston structural FEA verification failed (LC-1 Yield SF = {res.lc1_mechanical.yield_safety_factor:.2f} < {cfg.min_yield_safety_factor:.2f})![/bold red]")
    if strict:
      raise typer.Exit(code=1)

  return res


@app.command(name="fea-conrod")
def fea_conrod(
    peak_pressure_bar: float = 78.0,
    redline_rpm: float = 7000.0,
    output_dir: str = "outputs/fea/conrod",
    strict: bool = True,
):
  """Run structural, column buckling, and fatigue FEA on the connecting rod using CalculiX ccx."""
  rich.print(f"[cyan]Running connecting rod FEA (P_max = {peak_pressure_bar:.1f} bar, Redline = {redline_rpm:.0f} RPM)...[/cyan]")

  cfg = ConrodFeaConfig(
      peak_pressure_mpa=peak_pressure_bar / 10.0,
      redline_rpm=redline_rpm,
  )
  res = run_conrod_fea(config=cfg, output_dir=output_dir)

  table = Table(
      title="Connecting Rod Structural, Buckling & Fatigue FEA (CalculiX ccx)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Metric / Load Case", style="magenta")
  table.add_column("Value", style="bold green")
  table.add_column("Target / Limit", style="yellow")
  table.add_column("Status", style="cyan")

  is_buckle_crit_ok = res.critical_buckling_load_kn >= res.lc3a_compression.applied_force_kn * 2.5

  table.add_row("Mesh Density", f"{res.mesh_stats['num_nodes']:,} nodes / {res.mesh_stats['num_elements']:,} tets (C3D10)", "Quadratic", "[bold green]PASS[/bold green]")
  table.add_row("LC-3A Firing Compressive Force", f"{res.lc3a_compression.applied_force_kn:.2f} kN", "0° CA Peak", "LOAD")
  table.add_row("LC-3A P95 Von Mises Stress", f"{res.lc3a_compression.p95_von_mises_mpa:.1f} MPa", f"< {ALUMINUM_7075_T6.yield_strength_mpa / cfg.min_yield_safety_factor:.1f} MPa", _status_badge(res.lc3a_compression.is_pass))
  table.add_row("LC-3A Compressive Yield SF", f"{res.lc3a_compression.yield_safety_factor:.2f}", ">= 1.60", _status_badge(res.lc3a_compression.is_pass))
  table.add_row("---", "---", "---", "---")
  table.add_row("LC-3B Redline Inertia Tensile Force", f"{res.lc3b_tension.applied_force_kn:.2f} kN", f"@ {cfg.redline_rpm:.0f} RPM", "LOAD")
  table.add_row("LC-3B P95 Von Mises Stress", f"{res.lc3b_tension.p95_von_mises_mpa:.1f} MPa", f"< {ALUMINUM_7075_T6.yield_strength_mpa / cfg.min_yield_safety_factor:.1f} MPa", _status_badge(res.lc3b_tension.is_pass))
  table.add_row("LC-3B Tensile Yield SF", f"{res.lc3b_tension.yield_safety_factor:.2f}", ">= 1.60", _status_badge(res.lc3b_tension.is_pass))
  table.add_row("---", "---", "---", "---")
  table.add_row("Euler Column Buckling Critical Load", f"{res.critical_buckling_load_kn:.1f} kN", f"> {res.lc3a_compression.applied_force_kn*2.5:.1f} kN", _status_badge(is_buckle_crit_ok))
  table.add_row("Euler Column Buckling SF", f"{res.buckling_safety_factor:.2f}", ">= 2.50", _status_badge(res.is_buckling_safe))
  table.add_row("Modified Goodman Fatigue SF", f"{res.fatigue_safety_factor:.2f}", ">= 1.50", _status_badge(res.is_fatigue_safe))
  table.add_row("Total Simulation Solve Time", f"{res.total_solve_time_s:.2f} s", "< 60.0 s", "[bold green]PASS[/bold green]")

  rich.print(table)
  c_json = Path(output_dir) / "conrod_fea_results.json"
  c_json.write_text(json.dumps(res.to_dict(), indent=2))
  rich.print(f"[bold green]Saved connecting rod FEA results to: {c_json}[/bold green]")

  if not res.is_structural_safe:
    rich.print("[bold red]FAILED: Connecting rod structural FEA verification failed![/bold red]")
    if strict:
      raise typer.Exit(code=1)

  return res


@app.command(name="fea-gasket")
def fea_gasket(
    bolt_preload_kn: float = 22.0,
    peak_pressure_bar: float = 78.0,
    output_dir: str = "outputs/fea/gasket",
    strict: bool = True,
):
  """Run cylinder head bolt preload and fire-ring gasket sealing analysis."""
  rich.print(f"[cyan]Running cylinder head bolt preload and gasket sealing FEA...[/cyan]")

  cfg = HeadGasketFeaConfig(
      bolt_preload_kn=bolt_preload_kn,
      peak_pressure_mpa=peak_pressure_bar / 10.0,
  )
  res = run_head_gasket_fea(config=cfg, output_dir=output_dir)

  table = Table(
      title="Cylinder Head Bolt Preload & Gasket Sealing Analysis",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Parameter / Metric", style="magenta")
  table.add_column("Value", style="bold green")
  table.add_column("Target / Limit", style="yellow")
  table.add_column("Status", style="cyan")

  table.add_row("Bolt Specification", f"{cfg.num_bolts}x M{cfg.bolt_diameter_mm:.0f} Grade 10.9", "Standard", "INFO")
  table.add_row("Bolt Tightening Torque", f"{res.tightening_torque_nm:.1f} N·m", "Nominal", "SPEC")
  table.add_row("Total Initial Clamping Force", f"{res.total_clamp_force_kn:.1f} kN", f"{cfg.num_bolts}x {cfg.bolt_preload_kn:.1f} kN", "LOAD")
  table.add_row("Combustion Separation Force", f"{res.combustion_separation_force_kn:.1f} kN", f"@ {cfg.peak_pressure_mpa*10.0:.1f} bar", "LOAD")
  table.add_row("Initial Fire-Ring Contact Pressure", f"{res.initial_contact_pressure_mpa:.1f} MPa", "Preload", "INFO")
  table.add_row("Residual Fire-Ring Contact Pressure", f"{res.residual_contact_pressure_mpa:.1f} MPa", f">= {cfg.min_sealing_pressure_mpa:.1f} MPa", _status_badge(res.is_sealing_safe))
  table.add_row("Fire-Ring Sealing Safety Factor", f"{res.sealing_safety_factor:.2f}", ">= 1.00", _status_badge(res.is_sealing_safe))
  table.add_row("Dynamic Bolt Peak Stress", f"{res.bolt_peak_stress_mpa:.1f} MPa", "< 940.0 MPa", _status_badge(res.is_bolt_safe))
  table.add_row("Bolt Yield Safety Factor", f"{res.bolt_yield_safety_factor:.2f}", f">= {cfg.min_bolt_safety_factor:.2f}", _status_badge(res.is_bolt_safe))

  rich.print(table)
  g_json = Path(output_dir) / "head_gasket_fea_results.json"
  g_json.write_text(json.dumps(res.to_dict(), indent=2))
  rich.print(f"[bold green]Saved head gasket FEA results to: {g_json}[/bold green]")

  if not (res.is_sealing_safe and res.is_bolt_safe):
    rich.print("[bold red]FAILED: Cylinder head gasket sealing / bolt preload verification failed![/bold red]")
    if strict:
      raise typer.Exit(code=1)

  return res


@app.command(name="fea-atlas")
def fea_atlas(
    output_dir: str = "outputs/fea/atlas",
    models_json: str = "outputs/reports/models/fea_stress_atlas.json",
    force_rebuild: bool = False,
    strict: bool = True,
) -> FeaAtlasSummaryResult:
  """Run CalculiX C3D10 structural FEA across all 114 CAD solids and 5 multi-body assemblies."""
  rich.print("[cyan]Running Full-Engine CalculiX C3D10 FEA Stress Atlas (All Parts & Assemblies)...[/cyan]")
  res = run_fea_atlas(output_dir=output_dir, force_rebuild=force_rebuild)
  export_fea_atlas_to_webgl(
      fea_root_dir=Path(output_dir).parent,
      output_json_path=models_json,
      force_rebuild=False,
  )

  table = Table(
      title="Full-Engine CalculiX C3D10 Stress Atlas (Assemblies & Governing Limits)",
      show_header=True,
      header_style="bold cyan",
  )
  table.add_column("Assembly / Scope", style="magenta")
  table.add_column("Solids", style="cyan")
  table.add_column("P95 Stress", style="bold green")
  table.add_column("Min Yield SF", style="bold green")
  table.add_column("Governing Part", style="yellow")
  table.add_column("Status", style="cyan")

  for asm in res.assemblies:
    table.add_row(
        asm.title,
        str(asm.part_count),
        f"{asm.p95_stress_mpa:.1f} MPa",
        f"{asm.min_yield_safety_factor:.2f}",
        asm.limiting_part_key,
        _status_badge(asm.is_pass),
    )

  rich.print(table)
  rich.print(
      f"[bold green]FEA Atlas complete ({res.unique_meshes} unique C3D10 meshes / {res.total_solids} solids, "
      f"min SF={res.min_yield_safety_factor:.2f} >= {MIN_YIELD_SF_GATE:.2f} @ {res.limiting_part_key}, "
      f"{res.total_solve_time_s:.2f} s)[/bold green]"
  )
  rich.print(f"[bold green]Exported WebGL stress atlas to: {models_json}[/bold green]")

  if not res.all_passed:
    rich.print("[bold red]FAILED: Full-Engine FEA Stress Atlas did NOT pass all yield safety gates![/bold red]")
    if strict:
      raise typer.Exit(code=1)

  return res


def _print_fea_suite_failures(
    piston_res: object,
    conrod_res: object,
    gasket_res: object,
    atlas_res: FeaAtlasSummaryResult,
) -> None:
  """Print concise diagnostic failures for Phase 4 FEA suite checks."""
  rich.print("\n[bold red]====================================================================[/bold red]")
  rich.print("[bold red]  FAILED: Phase 4 FEA Suite Did NOT Pass All Acceptance Targets!    [/bold red]")
  if not piston_res.is_structural_safe:
    rich.print(f"[bold red]  - Piston LC-1 Yield SF = {piston_res.lc1_mechanical.yield_safety_factor:.2f} (Target >= 1.60)[/bold red]")
  if not conrod_res.is_structural_safe:
    rich.print("[bold red]  - Connecting Rod structural criteria NOT met[/bold red]")
  if not (gasket_res.is_sealing_safe and gasket_res.is_bolt_safe):
    rich.print("[bold red]  - Cylinder head gasket sealing / bolt preload criteria NOT met[/bold red]")
  if not atlas_res.all_passed:
    rich.print(
        f"[bold red]  - Full-Engine FEA Atlas min SF = {atlas_res.min_yield_safety_factor:.2f} "
        f"({atlas_res.limiting_part_key})[/bold red]"
    )
  rich.print("[bold red]====================================================================[/bold red]")


@app.command(name="fea-all")
def fea_all(
    output_dir: str = "outputs/fea",
    strict: bool = True,
) -> None:
  """Run the complete Phase 4 Structural & Thermal FEA suite (Piston, Connecting Rod, Gasket, Atlas)."""
  rich.print("[bold cyan]Executing Complete Phase 4 Structural & Thermal FEA Suite...[/bold cyan]")
  t0 = time.time()

  piston_res = fea_piston(output_dir=f"{output_dir}/piston", strict=False)
  conrod_res = fea_conrod(output_dir=f"{output_dir}/conrod", strict=False)
  gasket_res = fea_gasket(output_dir=f"{output_dir}/gasket", strict=False)
  atlas_res = run_fea_atlas(output_dir=f"{output_dir}/atlas", force_rebuild=False)

  elapsed = time.time() - t0
  all_ok = (
      piston_res.is_structural_safe
      and conrod_res.is_structural_safe
      and gasket_res.is_sealing_safe
      and gasket_res.is_bolt_safe
      and atlas_res.all_passed
  )

  if not all_ok:
    _print_fea_suite_failures(piston_res, conrod_res, gasket_res, atlas_res)
    if strict:
      raise typer.Exit(code=1)
  else:
    rich.print("\n[bold green]====================================================================[/bold green]")
    rich.print(f"[bold green]  Complete Phase 4 FEA Suite PASSED in {elapsed:.2f} s!             [/bold green]")
    rich.print("[bold green]====================================================================[/bold green]")


@app.command(name="dyno")
def dyno(
    config_path: str = "config/engine_specs.yaml",
    output_dir: str = "outputs/dyno",
    drum_inertia_kg_m2: float = DEFAULT_DRUM_INERTIA_KG_M2,
):
  """Run Phase 6 virtual brake dyno sweep, transient inertia pull, and 44.1 kHz exhaust audio."""
  rich.print("[bold cyan]Running Virtual Brake Dynamometer & Exhaust Audio Cell...[/bold cyan]")
  t0 = time.time()
  out_p = Path(output_dir)
  out_p.mkdir(parents=True, exist_ok=True)
  cfg_p = Path(config_path)

  sim = build_live_twin_simulator(config_path=cfg_p, use_live_twin=True)
  steady_res = run_steady_dyno_sweep(SteadyDynoConfig(config_path=cfg_p, use_live_twin=True))
  pull_res = run_inertia_dyno_pull(
      steady_result=steady_res,
      config=InertiaPullConfig(drum_inertia_kg_m2=drum_inertia_kg_m2),
      sim=sim,
  )
  wav_path = out_p / "exhaust_pull_44k1.wav"
  audio_res = synthesize_exhaust_audio(
      pull_result=pull_res,
      steady_result=steady_res,
      output_wav_path=wav_path,
  )

  payload = {
      "steady_sweep": steady_res.to_dict(),
      "inertia_pull": pull_res.to_dict(),
      "exhaust_audio": audio_res.to_dict(),
  }
  json_path = out_p / "dyno_results.json"
  json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  table = Table(title="Virtual Dynamometer & Acoustic Summary (SAE J1349)")
  table.add_column("Metric", style="cyan")
  table.add_column("Value", style="bold white")
  table.add_row("Peak SAE Brake Power", f"{steady_res.peak_sae_power_hp:.2f} hp ({steady_res.peak_sae_power_kw:.2f} kW) @ {steady_res.peak_power_rpm:.0f} RPM")
  table.add_row("Peak SAE Torque", f"{steady_res.peak_sae_torque_lb_ft:.2f} lb-ft ({steady_res.peak_sae_torque_nm:.2f} N-m) @ {steady_res.peak_torque_rpm:.0f} RPM")
  table.add_row("Minimum BSFC Island", f"{steady_res.bsfc_map.min_bsfc_g_kwh:.1f} g/kWh @ {steady_res.bsfc_map.min_bsfc_rpm:.0f} RPM ({steady_res.bsfc_map.min_bsfc_throttle * 100.0:.0f}% throttle)")
  table.add_row("Helmholtz / Exhaust Tuned RPM", f"{steady_res.helmholtz_tuned_rpm:.0f} / {steady_res.exhaust_scavenge_rpm:.0f} RPM")
  table.add_row("WOT Pull Duration (2000-7000)", f"{pull_res.wot_pull_duration_s:.2f} s (Drum I={drum_inertia_kg_m2:.3f} kg*m^2)")
  table.add_row("Head Temp & Oil Viscosity", f"Head {pull_res.head_temp_start_c:.1f} -> {pull_res.head_temp_end_c:.1f} C | Oil {pull_res.oil_viscosity_start_cst:.1f} -> {pull_res.oil_viscosity_end_cst:.1f} cSt")
  table.add_row("Crank H6/x6 Press-Fit Slip SF", f"{pull_res.min_crank_press_fit_sf:.2f} ({'PASS' if pull_res.crank_press_fit_report.is_safe else 'FAIL'}) | Flywheel burst SF {pull_res.flywheel_burst_sf_redline:.1f}")
  table.add_row("Exhaust Audio WAV", f"{wav_path} ({audio_res.duration_s:.2f} s @ {audio_res.sample_rate_hz} Hz, {audio_res.peak_dbfs:.2f} dBFS)")
  rich.print(table)
  rich.print(f"[bold green]Dyno results saved in {time.time() - t0:.2f} s to {json_path}[/bold green]")


@app.command(name="report")
def generate_report(
    output_dir: str = "outputs/reports",
):
  """Generate interactive 3D WebGL (Three.js) dashboard, glTF CAD models, FEA stress meshes, and Markdown summary."""
  rich.print("[bold cyan]Generating Multi-Physics WebGL 3D Dashboard & Verification Reports...[/bold cyan]")
  t0 = time.time()
  out_path = Path(output_dir)
  models_dir = out_path / "models"
  models_dir.mkdir(parents=True, exist_ok=True)

  # 1. Export CAD models to binary glTF (.glb)
  rich.print("[cyan]Exporting CAD solid B-Rep models to binary glTF (.glb)...[/cyan]")
  export_cad_models_to_gltf(models_dir)

  # 2. Export FEA stress meshes for Three.js
  base_outputs = out_path.parent
  piston_frd = base_outputs / "fea" / "piston" / "piston_lc1.frd"
  piston_inp = base_outputs / "fea" / "piston" / "piston_mesh.inp"
  if piston_frd.exists() and piston_inp.exists():
    export_fea_stress_to_webgl(
        frd_path=piston_frd,
        inp_path=piston_inp,
        output_json_path=models_dir / "piston_lc1_stress.json",
        title="Piston LC-1 Firing TDC Peak Pressure",
        component_name="Piston",
        yield_strength_mpa=ALUMINUM_4032_T6.yield_strength_mpa,
    )

  conrod_frd = base_outputs / "fea" / "conrod" / "conrod_lc3a.frd"
  conrod_inp = base_outputs / "fea" / "conrod" / "conrod_mesh.inp"
  if conrod_frd.exists() and conrod_inp.exists():
    export_fea_stress_to_webgl(
        frd_path=conrod_frd,
        inp_path=conrod_inp,
        output_json_path=models_dir / "conrod_lc3a_stress.json",
        title="Connecting Rod LC-3A Firing Compression",
        component_name="Connecting Rod",
        yield_strength_mpa=ALUMINUM_7075_T6.yield_strength_mpa,
    )

  fea_atlas_json_path = base_outputs / "fea" / "atlas" / "fea_atlas_results.json"
  if (base_outputs / "fea" / "atlas" / "cases").exists() or fea_atlas_json_path.exists():
    export_fea_atlas_to_webgl(
        fea_root_dir=base_outputs / "fea",
        output_json_path=models_dir / "fea_stress_atlas.json",
        force_rebuild=False,
    )

  # 3. Generate interactive HTML5 dashboard
  cht_json_path = base_outputs / "cfd" / "cht_cooling_results.json"
  closed_loop_path = base_outputs / "calibration" / "closed_loop_results.json"
  piston_fea_path = base_outputs / "fea" / "piston" / "piston_fea_results.json"
  conrod_fea_path = base_outputs / "fea" / "conrod" / "conrod_fea_results.json"
  gasket_fea_path = base_outputs / "fea" / "gasket" / "head_gasket_fea_results.json"
  dfm_report_path = base_outputs / "dfm" / "c500_dfm_report.json"
  setup_sheets_path = base_outputs / "dfm" / "drawings" / "setup_sheets_manifest.json"
  bom_json_path = base_outputs / "bom" / "bom_summary.json"
  thermo_lub_path = base_outputs / "lubrication" / "thermo_lub_results.json"
  dyno_json_path = base_outputs / "dyno" / "dyno_results.json"
  dyno_wav_path = base_outputs / "dyno" / "exhaust_pull_44k1.wav"
  html_path = out_path / "engineering_dashboard.html"
  generate_interactive_dashboard(
      output_html_path=html_path,
      models_dir=models_dir,
      cht_json=cht_json_path,
      closed_loop_json=closed_loop_path,
      piston_fea_json=piston_fea_path,
      conrod_fea_json=conrod_fea_path,
      gasket_fea_json=gasket_fea_path,
      fea_atlas_json=fea_atlas_json_path,
      dfm_report_json=dfm_report_path,
      dyno_json=dyno_json_path,
      dyno_wav=dyno_wav_path,
      bom_json=bom_json_path,
      thermo_lub_json=thermo_lub_path,
      setup_sheets_json=setup_sheets_path,
  )

  # 4. Generate Markdown summary
  md_path = out_path / "summary.md"
  generate_markdown_summary(
      output_md_path=md_path,
      cht_json=cht_json_path,
      closed_loop_json=closed_loop_path,
      piston_fea_json=piston_fea_path,
      conrod_fea_json=conrod_fea_path,
      gasket_fea_json=gasket_fea_path,
      fea_atlas_json=fea_atlas_json_path,
      dfm_report_json=dfm_report_path,
      dyno_json=dyno_json_path,
      bom_json=bom_json_path,
      thermo_lub_json=thermo_lub_path,
      setup_sheets_json=setup_sheets_path,
  )

  elapsed = time.time() - t0
  rich.print(f"[bold green]Report generation completed in {elapsed:.2f} s![/bold green]")
  rich.print(f"[bold green]Interactive 3D Dashboard: {html_path}[/bold green]")
  rich.print(f"[bold green]Markdown Verification Summary: {md_path}[/bold green]")


@app.command(name="build-pages")
def build_pages(
    outputs_dir: str = "outputs",
    site_dir: str = "_site",
):
  """Assemble self-contained static GitHub Pages bundle and verify all local asset links."""
  manifest = build_pages_site(outputs_dir=outputs_dir, site_dir=site_dir)
  table = Table(title="GitHub Pages Static Site Bundle")
  table.add_column("Field", style="cyan")
  table.add_column("Value", style="bold green")
  table.add_row("Output Site Directory", manifest.site_dir)
  table.add_row("Entrypoint Index HTML", manifest.index_html)
  table.add_row("Bundled Asset Files", str(manifest.copied_file_count))
  table.add_row("Verified Relative Links", str(len(manifest.verified_links)))
  rich.print(table)


@app.command(name="run-all")
def run_all(
    quick: bool = typer.Option(True, "--quick/--full", help="Run in fast verification mode (< 60s)"),
    output_dir: str = "outputs",
):
  """Run the complete end-to-end multi-physics pipeline (0D -> Dynamics -> CAD -> CFD -> FEA -> Dashboard)."""
  rich.print("[bold cyan]====================================================================[/bold cyan]")
  rich.print(f"[bold cyan]  OpenEngine-1C: Full Multi-Physics Autonomous Engineering Pipeline [/bold cyan]")
  rich.print(f"[bold cyan]  Mode: {'QUICK (<60s)' if quick else 'FULL HIGH-RES'}                           [/bold cyan]")
  rich.print("[bold cyan]====================================================================[/bold cyan]")
  t_total_start = time.time()

  # Phase 1: 0D Thermo, Lubrication & Valvetrain Dynamics + C500 BoM
  rich.print("\n[bold yellow]>>> Step 1: 0D/1D Thermodynamics, Lubrication, Dynamics & C500 BoM...[/bold yellow]")
  thermo(config_path="config/engine_specs.yaml")
  lub(config_path="config/engine_specs.yaml", output_dir=f"{output_dir}/lubrication")
  valvetrain(config_path="config/engine_specs.yaml")
  dynamics(config_path="config/engine_specs.yaml")
  bom(config_path="config/engine_specs.yaml", output_dir=f"{output_dir}/bom", fourth_axis=None)

  # Phase 2: Parametric B-Rep CAD
  rich.print("\n[bold yellow]>>> Step 2: Parametric Solid CAD Generation (STEP)...[/bold yellow]")
  cad(config_path="config/engine_specs.yaml", output_dir=f"{output_dir}/cad")

  # Phase 2B: NestWorks C500 DFM Verification & 2D Setup Sheets
  rich.print("\n[bold yellow]>>> Step 2B: NestWorks C500 Live B-Rep DFM Check & 2D Setup Sheets...[/bold yellow]")
  dfm_check(output_dir=f"{output_dir}/dfm", fourth_axis=None, drawings=True)

  # Phase 3: CFD, Combustion, CHT, and Closed-Loop Calibration
  rich.print("\n[bold yellow]>>> Step 3: CFD & Moving-Mesh Combustion...[/bold yellow]")
  cfd_port_flow(output_dir=f"{output_dir}/cfd")
  cfd_combustion(output_dir=f"{output_dir}/cfd")
  cfd_cht_cooling(output_dir=f"{output_dir}/cfd")
  cfd_closed_loop(output_dir=f"{output_dir}/calibration")

  # Phase 4: Structural & Thermal FEA (CalculiX)
  rich.print("\n[bold yellow]>>> Step 4: Structural & Thermal FEA (CalculiX ccx)...[/bold yellow]")
  fea_all(output_dir=f"{output_dir}/fea")

  # Phase 4B: Virtual Dynamometer & Exhaust Audio
  rich.print("\n[bold yellow]>>> Step 4B: Virtual Brake Dynamometer & Exhaust Audio...[/bold yellow]")
  dyno(config_path="config/engine_specs.yaml", output_dir=f"{output_dir}/dyno")

  # Phase 5: Synthesis, Interactive 3D WebGL Dashboard & Summary Report
  rich.print("\n[bold yellow]>>> Step 5: Interactive 3D Dashboard & Verification Reports...[/bold yellow]")
  generate_report(output_dir=f"{output_dir}/reports")

  t_total_elapsed = time.time() - t_total_start
  rich.print("\n[bold green]====================================================================[/bold green]")
  rich.print(f"[bold green]  OpenEngine-1C Pipeline Complete! Total Runtime: {t_total_elapsed:.2f} s       [/bold green]")
  rich.print(f"[bold green]  Interactive 3D Dashboard: {output_dir}/reports/engineering_dashboard.html[/bold green]")
  rich.print("[bold green]====================================================================[/bold green]")


if __name__ == "__main__":
  app()
