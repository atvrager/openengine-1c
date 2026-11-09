"""OpenEngine-1C Command Line Interface."""

from pathlib import Path
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
from openengine.dynamics import (
    BearingGeometry,
    FlywheelDimensions,
    FlywheelDynamics,
    HydrodynamicBearing,
    SOHCCamshaft,
    SliderCrankDynamics,
)
from openengine.infra.podman import PodmanRunner
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


if __name__ == "__main__":
  app()
