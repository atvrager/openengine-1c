"""OpenEngine-1C Command Line Interface."""

from pathlib import Path
from build123d import export_step
from openengine.cad.connecting_rod import generate_connecting_rod
from openengine.cad.piston import generate_piston
from openengine.dynamics.slider_crank import SliderCrankDynamics
from openengine.infra.podman import PodmanRunner
from openengine.thermo.cycle import ClosedCycleSimulator, EngineGeometry
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
def thermo(config_path: str = "config/engine_specs.yaml"):
  """Run 0D thermodynamic closed-cycle simulation."""
  cfg = load_config(config_path)
  g = cfg["geometry"]
  geom = EngineGeometry(
      bore=g["bore"],
      stroke=g["stroke"],
      con_rod_length=g["con_rod_length"],
      compression_ratio=g["compression_ratio"],
  )

  sim = ClosedCycleSimulator(geom)
  res = sim.solve_cycle()

  table = Table(
      title="0D Thermodynamic Cycle Results",
      show_header=True,
      header_style="bold blue",
  )
  table.add_column("Output Metric", style="cyan")
  table.add_column("Simulated Value", style="bold green")

  table.add_row("Displacement", f"{geom.displacement * 1.0e6:.1f} cm³")
  table.add_row(
      "Peak Cylinder Pressure (P_max)", f"{res.p_max_pa / 1.0e6:.2f} MPa"
  )
  table.add_row("Peak Temperature (T_max)", f"{res.t_max_k:.0f} K")
  table.add_row("Indicated Work per Cycle", f"{res.indicated_work_j:.1f} J")
  table.add_row("IMEP", f"{res.imep_pa / 1.0e5:.2f} bar")
  table.add_row(
      "Indicated Thermal Efficiency", f"{res.thermal_efficiency * 100:.1f} %"
  )

  rich.print(table)


@app.command()
def dynamics(config_path: str = "config/engine_specs.yaml"):
  """Run slider-crank kinematic force and flywheel balancing calculations."""
  cfg = load_config(config_path)
  g = cfg["geometry"]
  geom = EngineGeometry(
      bore=g["bore"],
      stroke=g["stroke"],
      con_rod_length=g["con_rod_length"],
      compression_ratio=g["compression_ratio"],
  )

  sim = ClosedCycleSimulator(geom)
  thermo_res = sim.solve_cycle()

  dyn = SliderCrankDynamics(geom)
  rpm = float(cfg["operating_conditions"]["rated_rpm"])
  forces = dyn.solve_forces(
      rpm=rpm,
      theta_deg=thermo_res.theta_deg,
      pressure_pa=thermo_res.pressure_pa,
  )
  balancer = dyn.size_balancer_and_flywheel(rpm=rpm, forces=forces)

  table = Table(
      title=f"Mechanism Dynamics & Balancing @ {rpm:.0f} RPM",
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
  """Check availability of containerized Podman solvers."""
  runner = PodmanRunner()
  if runner.is_available:
    rich.print("[green]Podman/Docker runtime is available and responding.[/green]")
  else:
    rich.print(
        "[yellow]Podman/Docker runtime is not responding or not in PATH.[/yellow]"
    )


if __name__ == "__main__":
  app()
