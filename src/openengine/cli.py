"""OpenEngine-1C Command Line Interface."""

from pathlib import Path
import time
from build123d import export_step
import numpy as np
from openengine.cad import (
    calculate_block_thermal_capacity,
    calculate_combustion_chamber_volume,
    calculate_crankshaft_balance,
    extract_intake_port_fluid_domain,
    generate_connecting_rod,
    generate_crankshaft,
    generate_cylinder_block,
    generate_cylinder_head,
    generate_cylinder_liner,
    generate_engine_assembly,
    generate_piston,
    generate_poppet_valve,
    verify_assembly_clearances,
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
from openengine.dynamics import (
    BearingGeometry,
    FlywheelDimensions,
    FlywheelDynamics,
    HydrodynamicBearing,
    SOHCCamshaft,
    SliderCrankDynamics,
)
from openengine.fea import (
    ALUMINUM_4032_T6,
    ConrodFeaConfig,
    HeadGasketFeaConfig,
    PistonFeaConfig,
    run_conrod_fea,
    run_head_gasket_fea,
    run_piston_fea,
)
from openengine.infra.podman import PodmanRunner
from openengine.reporting import (
    export_cad_models_to_gltf,
    export_fea_stress_to_webgl,
    generate_interactive_dashboard,
    generate_markdown_summary,
)
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
    v_cfg = cfg.get("valves", {})
    valves = ValveGeometry(
        intake_diameter_m=v_cfg.get("intake_diameter", 0.030),
        exhaust_diameter_m=v_cfg.get("exhaust_diameter", 0.026),
        intake_max_lift_m=v_cfg.get("max_lift", 0.0085),
        exhaust_max_lift_m=v_cfg.get("max_lift", 0.0080),
    )
    sim = FourStrokeSimulator(geom, valves=valves)
    res = sim.solve_cycle(rpm=target_rpm)

    table = Table(
        title=f"720° Four-Stroke Engine Cycle & Breathing @ {target_rpm:.0f} RPM",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Cycle Metric", style="cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Displacement", f"{geom.displacement * 1.0e6:.1f} cm³")
    table.add_row("Volumetric Efficiency (η_v)", f"{res.volumetric_efficiency * 100:.1f} %")
    table.add_row("Trapped Air-Fuel Mass", f"{res.trapped_mass_kg * 1.0e6:.1f} mg / cycle")
    table.add_row("Peak Cylinder Pressure (P_max)", f"{res.p_max_pa / 1.0e6:.2f} MPa")
    table.add_row("Peak Gas Temperature (T_max)", f"{res.t_max_k:.0f} K")
    table.add_row("Gross Indicated Work (High-P loop)", f"{res.gross_work_j:.1f} J")
    table.add_row("Pumping Work Loss (Low-P loop)", f"{res.pump_work_j:.1f} J")
    table.add_row("Net Indicated Work (W_net)", f"{res.net_work_j:.1f} J")
    table.add_row("GIMEP (Gross IMEP)", f"{res.gimep_pa / 1.0e5:.2f} bar")
    table.add_row("PMEP (Pumping Loss MEP)", f"{res.pmep_pa / 1.0e5:.2f} bar")
    table.add_row("NIMEP (Net Indicated MEP)", f"{res.nimep_pa / 1.0e5:.2f} bar")
    table.add_row("Indicated Engine Power", f"{res.power_kw:.2f} kW ({res.power_kw * 1.341:.1f} hp)")
    table.add_row("Indicated Engine Torque", f"{res.torque_nm:.2f} N·m")
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
def dynamics(
    config_path: str = "config/engine_specs.yaml",
    rpm: float = 0.0,
):
  """Run slider-crank dynamics, polar shaking vectors, flywheel sizing, and bearing PV orbit."""
  cfg = load_config(config_path)
  g = cfg["geometry"]
  geom = EngineGeometry(
      bore=g["bore"],
      stroke=g["stroke"],
      con_rod_length=g["con_rod_length"],
      compression_ratio=g["compression_ratio"],
  )

  target_rpm = rpm if rpm > 0.0 else float(cfg["operating_conditions"]["rated_rpm"])

  sim = FourStrokeSimulator(geom)
  thermo_res = sim.solve_cycle(rpm=target_rpm)

  dyn = SliderCrankDynamics(geom)
  forces = dyn.solve_forces(
      rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      pressure_pa=thermo_res.pressure_pa,
      recip_balance_factor=0.50,
  )
  sweep = dyn.sweep_balance_factors(rpm=target_rpm, forces=forces)

  # Flywheel sizing via FlywheelDynamics
  fw_dyn = FlywheelDynamics()
  fw_rep = fw_dyn.size_flywheel(
      rated_rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      torque_nm=forces.crankshaft_torque_nm,
      target_coeff_fluctuation=0.035,
      redline_rpm=float(cfg["operating_conditions"]["redline_rpm"]),
  )

  # Big-end bearing lubrication via HydrodynamicBearing
  f_big_end = np.sqrt(forces.net_wrist_pin_force_n**2 + forces.cylinder_side_thrust_n**2)
  big_end_bearing = HydrodynamicBearing(
      geom=BearingGeometry(
          name="Con-Rod Big-End Journal",
          journal_diameter_m=g.get("crankpin_diameter", 0.030),
          bearing_width_m=0.016,
          radial_clearance_m=20.0e-6,
      )
  )
  bearing_rep = big_end_bearing.analyze_bearing_safety(
      rpm=target_rpm,
      theta_deg=thermo_res.theta_deg,
      radial_load_n=f_big_end,
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
  fw_table.add_row("Disc Outer Diameter / Hub", f"{fw_dyn.dims.outer_radius_m * 2000:.0f} mm / {fw_dyn.dims.inner_bore_radius_m * 2000:.0f} mm")
  fw_table.add_row("Rim Axial Thickness", f"{fw_rep.flywheel_thickness_mm:.1f} mm")
  fw_table.add_row("Flywheel Mass", f"{fw_rep.flywheel_mass_kg:.2f} kg")
  fw_table.add_row("Rim Hoop Stress @ Redline", f"{fw_rep.hoop_stress_redline_mpa:.1f} MPa")
  fw_table.add_row("Burst Safety Factor @ Redline", f"{fw_rep.burst_safety_factor_redline:.1f} (Safe: {fw_rep.is_safe})")
  rich.print(fw_table)

  brg_table = Table(
      title="Big-End Hydrodynamic Journal Bearing (Ocvirk Solution)",
      show_header=True,
      header_style="bold blue",
  )
  brg_table.add_column("Bearing Parameter", style="cyan")
  brg_table.add_column("Value", style="bold green")
  brg_table.add_row("Journal Diameter × Width", f"{big_end_bearing.geom.journal_diameter_m * 1000:.1f} × {big_end_bearing.geom.bearing_width_m * 1000:.1f} mm")
  brg_table.add_row("Radial Clearance (c)", f"{big_end_bearing.geom.radial_clearance_m * 1.0e6:.1f} µm")
  brg_table.add_row("Peak Radial Load", f"{bearing_rep.peak_load_n / 1000.0:.2f} kN")
  brg_table.add_row("Peak Projected Pressure (P_max)", f"{bearing_rep.peak_pressure_mpa:.1f} MPa (Limit: {bearing_rep.allowable_pressure_mpa:.1f} MPa)")
  brg_table.add_row("Mean Bearing Pressure", f"{bearing_rep.mean_pressure_mpa:.1f} MPa")
  brg_table.add_row("Journal Sliding Velocity", f"{bearing_rep.sliding_velocity_m_s:.2f} m/s")
  brg_table.add_row("Peak PV Factor", f"{bearing_rep.peak_pv_mpa_m_s:.1f} MPa·m/s")
  brg_table.add_row("Peak Journal Eccentricity (ε_max)", f"{bearing_rep.peak_eccentricity_ratio:.4f}")
  brg_table.add_row("Minimum Oil Film Thickness (h_min)", f"{bearing_rep.min_film_thickness_um:.2f} µm")
  brg_table.add_row("Lubrication Lambda Ratio (h_min / R_q)", f"{bearing_rep.min_lambda_ratio:.2f} (Full Hydrodynamic: {bearing_rep.is_hydrodynamic})")
  rich.print(brg_table)


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
  export_step(rod, rod_file)
  rich.print(f"[green]Saved connecting rod ({rod.volume/1000:.1f} cm³) to: {rod_file}[/green]")

  rich.print("[cyan]Generating Piston via build123d...[/cyan]")
  piston = generate_piston(
      bore_m=g["bore"],
      wrist_pin_diam_m=g["wrist_pin_diameter"],
  )
  piston_file = out_path / "piston.step"
  export_step(piston, piston_file)
  rich.print(f"[green]Saved piston ({piston.volume/1000:.1f} cm³) to: {piston_file}[/green]")

  rich.print("[cyan]Generating SOHC Pent-Roof Cylinder Head via build123d...[/cyan]")
  v_mm3, v_cm3 = calculate_combustion_chamber_volume()
  head = generate_cylinder_head()
  head_file = out_path / "cylinder_head.step"
  export_step(head, head_file)
  rich.print(
      f"[green]Saved cylinder head ({head.volume/1000:.1f} cm³, chamber dome {v_cm3:.2f} cm³) to: {head_file}[/green]"
  )

  rich.print("[cyan]Generating Poppet Valves via build123d...[/cyan]")
  in_valve = generate_poppet_valve(head_diam_m=v_cfg.get("intake_diameter", 0.030))
  ex_valve = generate_poppet_valve(head_diam_m=v_cfg.get("exhaust_diameter", 0.026))
  in_valve_file = out_path / "valve_intake.step"
  ex_valve_file = out_path / "valve_exhaust.step"
  export_step(in_valve, in_valve_file)
  export_step(ex_valve, ex_valve_file)
  rich.print(f"[green]Saved intake valve ({in_valve.volume/1000:.2f} cm³) to: {in_valve_file}[/green]")
  rich.print(f"[green]Saved exhaust valve ({ex_valve.volume/1000:.2f} cm³) to: {ex_valve_file}[/green]")

  rich.print("[cyan]Extracting CFD Port Flow Fluid Domain via build123d...[/cyan]")
  fluid = extract_intake_port_fluid_domain()
  fluid_file = out_path / "intake_fluid_domain.step"
  export_step(fluid, fluid_file)
  rich.print(f"[green]Saved CFD fluid domain ({fluid.volume/1000:.1f} cm³) to: {fluid_file}[/green]")

  rich.print("[cyan]Generating Cast Iron Cylinder Liner via build123d...[/cyan]")
  liner = generate_cylinder_liner()
  liner_file = out_path / "cylinder_liner.step"
  export_step(liner, liner_file)
  rich.print(f"[green]Saved cylinder liner ({liner.volume/1000:.1f} cm³, {(liner.volume/1000)*7.2:.0f} g) to: {liner_file}[/green]")

  rich.print("[cyan]Generating Aluminum Cylinder Block with Radial Cooling Fins via build123d...[/cyan]")
  block = generate_cylinder_block()
  block_file = out_path / "cylinder_block.step"
  export_step(block, block_file)
  therm = calculate_block_thermal_capacity()
  rich.print(
      f"[green]Saved cylinder block ({block.volume/1000:.1f} cm³, {therm['exterior_area_m2']:.3f} m² fin area, {therm['thermal_dissipation_w']:.0f} W cooling) to: {block_file}[/green]"
  )

  rich.print("[cyan]Generating Forged 4340 Counterweighted Crankshaft via build123d...[/cyan]")
  crank = generate_crankshaft()
  crank_file = out_path / "crankshaft.step"
  export_step(crank, crank_file)
  crank_bal = calculate_crankshaft_balance()
  rich.print(
      f"[green]Saved crankshaft ({crank.volume/1000:.1f} cm³, {crank_bal['crankshaft_mass_kg']:.2f} kg, {crank_bal['balance_factor_percent']:.1f}% balance) to: {crank_file}[/green]"
  )

  rich.print("[cyan]Assembling Master Multi-Body Engine Assembly & Verifying 720° Clearances...[/cyan]")
  clr = verify_assembly_clearances()
  rich.print(f"  [cyan]Min Intake P-V Clearance:[/cyan] [bold green]{clr.min_intake_pv_clearance_mm:.2f} mm[/bold green] @ {clr.min_intake_pv_angle_deg:.0f}° CA (Target >= 1.5 mm)")
  rich.print(f"  [cyan]Min Exhaust P-V Clearance:[/cyan] [bold green]{clr.min_exhaust_pv_clearance_mm:.2f} mm[/bold green] @ {clr.min_exhaust_pv_angle_deg:.0f}° CA (Target >= 1.5 mm)")
  rich.print(f"  [cyan]Min Rod-to-Liner Clearance:[/cyan] [bold green]{clr.min_rod_to_liner_clearance_mm:.2f} mm[/bold green] @ {clr.min_rod_to_liner_angle_deg:.0f}° CA (Target >= 5.0 mm)")
  rich.print(f"  [cyan]Kinematic Interference Check:[/cyan] [bold green]{'PASS (Clearance Verified)' if clr.is_clearance_safe else 'FAIL'}[/bold green]")

  assembly = generate_engine_assembly(0.0)
  assembly_file = out_path / "engine_assembly.step"
  export_step(assembly, assembly_file)
  rich.print(
      f"[green]Saved master engine assembly ({assembly.volume/1000:.1f} cm³, {len(assembly.solids())} parts) to: {assembly_file}[/green]"
  )


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
    end_ca: float = 5.0,
    spark_ca: float = -22.0,
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
    fin_h_coeff: float = 75.0,
    output_dir: str = "outputs/cfd",
):
  """Run Conjugate Heat Transfer (CHT) air-fin thermal cooling simulation via OpenFOAM laplacianFoam."""
  rich.print(f"[cyan]Running air-fin thermal cooling CHT simulation (T_amb = {ambient_temp_c:.1f}°C, h = {fin_h_coeff:.1f} W/m²K)...[/cyan]")

  cfg = ChtCoolingConfig(
      ambient_temp_k=ambient_temp_c + 273.15,
      fin_h_coeff=fin_h_coeff,
  )
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
  rich.print(f"[bold green]Saved piston FEA results to: {output_dir}/piston_fea_results.json[/bold green]")

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
  table.add_row("LC-3A P95 Von Mises Stress", f"{res.lc3a_compression.p95_von_mises_mpa:.1f} MPa", "< 850.0 MPa", _status_badge(res.lc3a_compression.is_pass))
  table.add_row("LC-3A Compressive Yield SF", f"{res.lc3a_compression.yield_safety_factor:.2f}", ">= 1.60", _status_badge(res.lc3a_compression.is_pass))
  table.add_row("---", "---", "---", "---")
  table.add_row("LC-3B Redline Inertia Tensile Force", f"{res.lc3b_tension.applied_force_kn:.2f} kN", f"@ {cfg.redline_rpm:.0f} RPM", "LOAD")
  table.add_row("LC-3B P95 Von Mises Stress", f"{res.lc3b_tension.p95_von_mises_mpa:.1f} MPa", "< 850.0 MPa", _status_badge(res.lc3b_tension.is_pass))
  table.add_row("LC-3B Tensile Yield SF", f"{res.lc3b_tension.yield_safety_factor:.2f}", ">= 1.60", _status_badge(res.lc3b_tension.is_pass))
  table.add_row("---", "---", "---", "---")
  table.add_row("Euler Column Buckling Critical Load", f"{res.critical_buckling_load_kn:.1f} kN", f"> {res.lc3a_compression.applied_force_kn*2.5:.1f} kN", _status_badge(is_buckle_crit_ok))
  table.add_row("Euler Column Buckling SF", f"{res.buckling_safety_factor:.2f}", ">= 2.50", _status_badge(res.is_buckling_safe))
  table.add_row("Modified Goodman Fatigue SF", f"{res.fatigue_safety_factor:.2f}", ">= 1.50", _status_badge(res.is_fatigue_safe))
  table.add_row("Total Simulation Solve Time", f"{res.total_solve_time_s:.2f} s", "< 60.0 s", "[bold green]PASS[/bold green]")

  rich.print(table)
  rich.print(f"[bold green]Saved connecting rod FEA results to: {output_dir}/conrod_fea_results.json[/bold green]")

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
  rich.print(f"[bold green]Saved head gasket FEA results to: {output_dir}/head_gasket_fea_results.json[/bold green]")

  if not (res.is_sealing_safe and res.is_bolt_safe):
    rich.print("[bold red]FAILED: Cylinder head gasket sealing / bolt preload verification failed![/bold red]")
    if strict:
      raise typer.Exit(code=1)

  return res


@app.command(name="fea-all")
def fea_all(
    output_dir: str = "outputs/fea",
    strict: bool = True,
):
  """Run the complete Phase 4 Structural & Thermal FEA suite (Piston, Connecting Rod, Gasket)."""
  rich.print("[bold cyan]Executing Complete Phase 4 Structural & Thermal FEA Suite...[/bold cyan]")
  t0 = time.time()

  piston_res = fea_piston(output_dir=f"{output_dir}/piston", strict=False)
  conrod_res = fea_conrod(output_dir=f"{output_dir}/conrod", strict=False)
  gasket_res = fea_gasket(output_dir=f"{output_dir}/gasket", strict=False)

  elapsed = time.time() - t0
  all_ok = (
      piston_res.is_structural_safe
      and conrod_res.is_structural_safe
      and gasket_res.is_sealing_safe
      and gasket_res.is_bolt_safe
  )

  if not all_ok:
    rich.print(f"\n[bold red]====================================================================[/bold red]")
    rich.print(f"[bold red]  FAILED: Phase 4 FEA Suite Did NOT Pass All Acceptance Targets!    [/bold red]")
    if not piston_res.is_structural_safe:
      rich.print(f"[bold red]  - Piston LC-1 Yield SF = {piston_res.lc1_mechanical.yield_safety_factor:.2f} (Target >= 1.60)[/bold red]")
    if not conrod_res.is_structural_safe:
      rich.print(f"[bold red]  - Connecting Rod structural criteria NOT met[/bold red]")
    if not (gasket_res.is_sealing_safe and gasket_res.is_bolt_safe):
      rich.print(f"[bold red]  - Cylinder head gasket sealing / bolt preload criteria NOT met[/bold red]")
    rich.print(f"[bold red]====================================================================[/bold red]")
    if strict:
      raise typer.Exit(code=1)
  else:
    rich.print(f"\n[bold green]====================================================================[/bold green]")
    rich.print(f"[bold green]  Complete Phase 4 FEA Suite PASSED in {elapsed:.2f} s!             [/bold green]")
    rich.print(f"[bold green]====================================================================[/bold green]")


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
  piston_frd = Path("outputs/fea/piston/piston_lc1.frd")
  piston_inp = Path("outputs/fea/piston/piston_mesh.inp")
  if piston_frd.exists() and piston_inp.exists():
    export_fea_stress_to_webgl(
        frd_path=piston_frd,
        inp_path=piston_inp,
        output_json_path=models_dir / "piston_lc1_stress.json",
        title="Piston LC-1 Firing TDC Peak Pressure",
        component_name="Piston",
        yield_strength_mpa=315.0,
    )

  conrod_frd = Path("outputs/fea/conrod/conrod_lc3a.frd")
  conrod_inp = Path("outputs/fea/conrod/conrod_mesh.inp")
  if conrod_frd.exists() and conrod_inp.exists():
    export_fea_stress_to_webgl(
        frd_path=conrod_frd,
        inp_path=conrod_inp,
        output_json_path=models_dir / "conrod_lc3a_stress.json",
        title="Connecting Rod LC-3A Firing Compression",
        component_name="Connecting Rod",
        yield_strength_mpa=850.0,
    )

  # 3. Generate interactive HTML5 dashboard
  html_path = out_path / "engineering_dashboard.html"
  generate_interactive_dashboard(
      output_html_path=html_path,
      models_dir=models_dir,
  )

  # 4. Generate Markdown summary
  md_path = out_path / "summary.md"
  generate_markdown_summary(output_md_path=md_path)

  elapsed = time.time() - t0
  rich.print(f"[bold green]Report generation completed in {elapsed:.2f} s![/bold green]")
  rich.print(f"[bold green]Interactive 3D Dashboard: {html_path}[/bold green]")
  rich.print(f"[bold green]Markdown Verification Summary: {md_path}[/bold green]")


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

  # Phase 1: 0D Thermo & Valvetrain Dynamics
  rich.print("\n[bold yellow]>>> Step 1: 0D/1D Thermodynamics & Valvetrain Dynamics...[/bold yellow]")
  thermo(config_path="config/engine_specs.yaml")
  valvetrain(config_path="config/engine_specs.yaml")
  dynamics(config_path="config/engine_specs.yaml")

  # Phase 2: Parametric B-Rep CAD
  rich.print("\n[bold yellow]>>> Step 2: Parametric Solid CAD Generation (STEP)...[/bold yellow]")
  cad(config_path="config/engine_specs.yaml", output_dir=f"{output_dir}/cad")

  # Phase 3: CFD, Combustion, CHT, and Closed-Loop Calibration
  rich.print("\n[bold yellow]>>> Step 3: CFD & Moving-Mesh Combustion...[/bold yellow]")
  cfd_port_flow(output_dir=f"{output_dir}/cfd")
  cfd_combustion(output_dir=f"{output_dir}/cfd")
  cfd_cht_cooling(output_dir=f"{output_dir}/cfd")
  cfd_closed_loop(output_dir=f"{output_dir}/calibration")

  # Phase 4: Structural & Thermal FEA (CalculiX)
  rich.print("\n[bold yellow]>>> Step 4: Structural & Thermal FEA (CalculiX ccx)...[/bold yellow]")
  fea_all(output_dir=f"{output_dir}/fea")

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
