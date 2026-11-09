"""OpenEngine-1C Command Line Interface."""

from pathlib import Path
from build123d import export_step
from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.cad.piston import generate_piston
from openengine.dynamics.slider_crank import SliderCrankDynamics
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
  """Run slider-crank kinematic force and flywheel balancing calculations across full 720-deg cycle."""
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
  )
  balancer = dyn.size_balancer_and_flywheel(rpm=target_rpm, forces=forces)

  table = Table(
      title=f"Mechanism Dynamics & Balancing @ {target_rpm:.0f} RPM",
      show_header=True,
      header_style="bold yellow",
  )
  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="bold green")

  piston_force_max = float(max(forces.gas_force_n))
  table.add_row("Peak Gas Force", f"{piston_force_max / 1000.0:.2f} kN")
  table.add_row("Peak Output Torque", f"{forces.peak_torque_nm:.2f} N·m")
  table.add_row("Mean Output Torque", f"{forces.mean_torque_nm:.2f} N·m")
  table.add_row(
      "Required Counterweight Mass",
      f"{balancer.counterweight_mass_kg * 1000.0:.1f} g @ r ="
      f" {balancer.counterweight_radius_m * 1000.0:.1f} mm",
  )
  table.add_row(
      "Required Flywheel Inertia",
      f"{balancer.flywheel_inertia_kg_m2 * 1000.0:.2f} × 10⁻³ kg·m²",
  )
  table.add_row(
      "Solid Disc Flywheel Mass", f"{balancer.flywheel_mass_kg:.2f} kg"
  )

  rich.print(table)


@app.command()
def cad(
    config_path: str = "config/engine_specs.yaml",
    output_dir: str = "outputs/cad",
):
  """Generate parametric 3D CAD parts (Piston, Con-Rod) via build123d / Open CASCADE."""
  cfg = load_config(config_path)
  g = cfg["geometry"]

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
  rich.print(f"[green]Saved connecting rod to: {rod_file}[/green]")

  rich.print("[cyan]Generating Piston via build123d...[/cyan]")
  piston = generate_piston(
      bore_m=g["bore"],
      wrist_pin_diam_m=g["wrist_pin_diameter"],
  )
  piston_file = out_path / "piston.step"
  export_step(piston, piston_file)
  rich.print(f"[green]Saved piston to: {piston_file}[/green]")


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
