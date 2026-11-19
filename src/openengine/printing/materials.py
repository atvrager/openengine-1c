"""FDM filament material profiles, thermal shrinkage, and snap-fit properties.

Provides calibrated engineering profiles for common functional FDM polymers,
starting with PETG as the primary OpenEngine snap-fit model kit default.
"""

from dataclasses import dataclass

DEFAULT_FILAMENT_NAME: str = "PETG"


@dataclass(frozen=True, slots=True)
class FilamentProfile:
  """Calibrated mechanical, thermal, and FDM tolerance profile for a filament.

  Attributes:
    name: Canonical registry identifier (e.g., ``"PETG"``, ``"PLA_PLUS"``).
    density_g_cm3: Solid polymer density in g/cm^3.
    shrinkage_xy: Fractional post-cooling linear shrinkage in X/Y (e.g. 0.003 = 0.3%).
    shrinkage_z: Fractional post-cooling linear shrinkage in Z (e.g. 0.002 = 0.2%).
    flexural_modulus_mpa: Flexural modulus in MPa (N/mm^2).
    tensile_yield_mpa: Yield tensile strength in MPa.
    allowable_snap_strain: Maximum safe short-term bending strain for cantilever snaps.
    sliding_friction_coeff: Dry polymer-on-polymer sliding coefficient of friction.
    radial_running_clearance_mm: Baseline radial clearance for rotating/sliding motion.
    radial_slip_clearance_mm: Baseline radial clearance for snug hand-assembled slip fits.
    radial_press_interference_mm: Baseline radial interference for pressed dowels/collars.
    snap_clearance_mm: Baseline pocket clearance for cantilever snap hooks.
    max_overhang_deg: Maximum support-free overhang angle measured from vertical (+Z).
    max_bridge_mm: Maximum reliable unsupported horizontal bridge span in mm.
    min_wall_nozzle_mult: Minimum structural wall multiplier relative to nozzle diameter.
    nozzle_temp_c: Recommended extrusion nozzle temperature in deg C.
    bed_temp_c: Recommended build-plate temperature in deg C.
    hdt_c: Heat deflection temperature (0.45 MPa) in deg C.
    adhesion_note: Bed-surface preparation and release guidance.
    print_notes: Process tuning and mechanical trade-off summary.
  """

  name: str
  density_g_cm3: float
  shrinkage_xy: float
  shrinkage_z: float
  flexural_modulus_mpa: float
  tensile_yield_mpa: float
  allowable_snap_strain: float
  sliding_friction_coeff: float
  radial_running_clearance_mm: float
  radial_slip_clearance_mm: float
  radial_press_interference_mm: float
  snap_clearance_mm: float
  max_overhang_deg: float
  max_bridge_mm: float
  min_wall_nozzle_mult: float
  nozzle_temp_c: float
  bed_temp_c: float
  hdt_c: float
  adhesion_note: str
  print_notes: str


# Insertion-ordered registry with PETG first as the primary OpenEngine default.
FILAMENT_PROFILES: dict[str, FilamentProfile] = {
    "PETG": FilamentProfile(
        name="PETG",
        density_g_cm3=1.27,
        shrinkage_xy=0.003,
        shrinkage_z=0.002,
        flexural_modulus_mpa=2100.0,
        tensile_yield_mpa=50.0,
        allowable_snap_strain=0.045,
        sliding_friction_coeff=0.25,
        radial_running_clearance_mm=0.30,
        radial_slip_clearance_mm=0.20,
        radial_press_interference_mm=0.05,
        snap_clearance_mm=0.25,
        max_overhang_deg=45.0,
        max_bridge_mm=20.0,
        min_wall_nozzle_mult=3.0,
        nozzle_temp_c=240.0,
        bed_temp_c=80.0,
        hdt_c=70.0,
        adhesion_note=(
            "Textured PEI sheet or smooth PEI with light glue-stick release agent; "
            "avoid printing directly onto bare glass."
        ),
        print_notes=(
            "Primary OpenEngine default. Dry 4h @ 65C if stringing occurs; use 30-50% "
            "part cooling fan for ductile snap-fit hooks and strong Z-layer fusion."
        ),
    ),
    "PLA": FilamentProfile(
        name="PLA",
        density_g_cm3=1.24,
        shrinkage_xy=0.002,
        shrinkage_z=0.0015,
        flexural_modulus_mpa=3400.0,
        tensile_yield_mpa=60.0,
        allowable_snap_strain=0.018,
        sliding_friction_coeff=0.28,
        radial_running_clearance_mm=0.25,
        radial_slip_clearance_mm=0.15,
        radial_press_interference_mm=0.04,
        snap_clearance_mm=0.20,
        max_overhang_deg=55.0,
        max_bridge_mm=35.0,
        min_wall_nozzle_mult=3.0,
        nozzle_temp_c=210.0,
        bed_temp_c=60.0,
        hdt_c=55.0,
        adhesion_note="Smooth or textured PEI cleaned with IPA; no heated chamber required.",
        print_notes=(
            "High stiffness and crisp bridges/overhangs, but brittle under snap bending "
            "and low HDT (55C); requires slender cantilever snap hooks."
        ),
    ),
    "PLA_PLUS": FilamentProfile(
        name="PLA_PLUS",
        density_g_cm3=1.24,
        shrinkage_xy=0.002,
        shrinkage_z=0.0015,
        flexural_modulus_mpa=2900.0,
        tensile_yield_mpa=55.0,
        allowable_snap_strain=0.030,
        sliding_friction_coeff=0.26,
        radial_running_clearance_mm=0.25,
        radial_slip_clearance_mm=0.15,
        radial_press_interference_mm=0.05,
        snap_clearance_mm=0.20,
        max_overhang_deg=50.0,
        max_bridge_mm=30.0,
        min_wall_nozzle_mult=3.0,
        nozzle_temp_c=215.0,
        bed_temp_c=60.0,
        hdt_c=58.0,
        adhesion_note="Smooth or textured PEI sheet cleaned with IPA.",
        print_notes=(
            "Impact-modified tough PLA+; substantially improved snap resilience over "
            "standard PLA while preserving sharp overhangs."
        ),
    ),
    "PETG_CF": FilamentProfile(
        name="PETG_CF",
        density_g_cm3=1.30,
        shrinkage_xy=0.0015,
        shrinkage_z=0.0015,
        flexural_modulus_mpa=4200.0,
        tensile_yield_mpa=58.0,
        allowable_snap_strain=0.015,
        sliding_friction_coeff=0.32,
        radial_running_clearance_mm=0.28,
        radial_slip_clearance_mm=0.18,
        radial_press_interference_mm=0.04,
        snap_clearance_mm=0.22,
        max_overhang_deg=48.0,
        max_bridge_mm=22.0,
        min_wall_nozzle_mult=3.5,
        nozzle_temp_c=255.0,
        bed_temp_c=85.0,
        hdt_c=78.0,
        adhesion_note=(
            "Textured PEI with light glue-stick release; requires hardened steel nozzle "
            "(>=0.4 mm) to avoid abrasive brass wear."
        ),
        print_notes=(
            "Very stiff, low-warping matte composite finish, but low snap elongation "
            "(1.5% limit) and higher abrasive sliding friction."
        ),
    ),
    "ASA": FilamentProfile(
        name="ASA",
        density_g_cm3=1.07,
        shrinkage_xy=0.006,
        shrinkage_z=0.005,
        flexural_modulus_mpa=2200.0,
        tensile_yield_mpa=45.0,
        allowable_snap_strain=0.040,
        sliding_friction_coeff=0.26,
        radial_running_clearance_mm=0.28,
        radial_slip_clearance_mm=0.18,
        radial_press_interference_mm=0.05,
        snap_clearance_mm=0.24,
        max_overhang_deg=45.0,
        max_bridge_mm=22.0,
        min_wall_nozzle_mult=3.0,
        nozzle_temp_c=260.0,
        bed_temp_c=100.0,
        hdt_c=95.0,
        adhesion_note=(
            "Textured PEI or smooth PEI + glue/slurry inside enclosed chamber (45-55C); "
            "use brim for large flat housings."
        ),
        print_notes=(
            "UV-stable engineering polymer with high HDT (95C) and tough snap ductility; "
            "apply XY shrinkage compensation and print in an enclosure."
        ),
    ),
    "ABS": FilamentProfile(
        name="ABS",
        density_g_cm3=1.04,
        shrinkage_xy=0.007,
        shrinkage_z=0.005,
        flexural_modulus_mpa=2300.0,
        tensile_yield_mpa=42.0,
        allowable_snap_strain=0.042,
        sliding_friction_coeff=0.25,
        radial_running_clearance_mm=0.28,
        radial_slip_clearance_mm=0.18,
        radial_press_interference_mm=0.05,
        snap_clearance_mm=0.24,
        max_overhang_deg=45.0,
        max_bridge_mm=22.0,
        min_wall_nozzle_mult=3.0,
        nozzle_temp_c=255.0,
        bed_temp_c=105.0,
        hdt_c=98.0,
        adhesion_note=(
            "Textured PEI or ABS slurry in heated enclosure (50C+); use wide brim on "
            "tall or wide crankcase parts to prevent corner lift."
        ),
        print_notes=(
            "High HDT (98C) and low sliding friction; exhibits highest XY shrinkage "
            "(0.7%), so bore compensation and draft-free enclosure are essential."
        ),
    ),
    "TPU_95A": FilamentProfile(
        name="TPU_95A",
        density_g_cm3=1.21,
        shrinkage_xy=0.004,
        shrinkage_z=0.003,
        flexural_modulus_mpa=75.0,
        tensile_yield_mpa=32.0,
        allowable_snap_strain=0.150,
        sliding_friction_coeff=0.55,
        radial_running_clearance_mm=0.40,
        radial_slip_clearance_mm=0.25,
        radial_press_interference_mm=0.12,
        snap_clearance_mm=0.25,
        max_overhang_deg=40.0,
        max_bridge_mm=12.0,
        min_wall_nozzle_mult=4.0,
        nozzle_temp_c=225.0,
        bed_temp_c=50.0,
        hdt_c=60.0,
        adhesion_note=(
            "Textured PEI or painter's tape with glue stick; never print directly on "
            "bare smooth PEI without a release barrier."
        ),
        print_notes=(
            "Flexible Shore 95A elastomer for seals, dampers, and compliant flexures; "
            "print slowly (<=3 mm^3/s) on direct-drive, unsuitable for rigid load bearings."
        ),
    ),
}

_FILAMENT_ALIASES: dict[str, str] = {
    "PLA+": "PLA_PLUS",
    "PLAPLUS": "PLA_PLUS",
    "TOUGH_PLA": "PLA_PLUS",
    "PETGCF": "PETG_CF",
    "TPU": "TPU_95A",
    "TPU95A": "TPU_95A",
    "TPU_95": "TPU_95A",
}


def get_filament_profile(name: str) -> FilamentProfile:
  """Look up a ``FilamentProfile`` by canonical key or common alias.

  Matching is case-insensitive and treats spaces/hyphens as underscores while
  accepting common shorthand such as ``"PLA+"``, ``"petg-cf"``, or ``"tpu"``.

  Args:
    name: Filament identifier or alias.

  Returns:
    The matching frozen ``FilamentProfile``.

  Raises:
    KeyError: If ``name`` does not resolve to any registered filament profile.
  """
  cleaned = name.strip().upper()
  if cleaned in _FILAMENT_ALIASES:
    return FILAMENT_PROFILES[_FILAMENT_ALIASES[cleaned]]

  normalized = cleaned.replace("-", "_").replace(" ", "_")
  if normalized.endswith("+"):
    normalized = f"{normalized[:-1]}_PLUS"
  canonical = _FILAMENT_ALIASES.get(normalized, normalized)
  if canonical not in FILAMENT_PROFILES:
    available = ", ".join(FILAMENT_PROFILES.keys())
    raise KeyError(f"Unknown filament profile {name!r}. Supported profiles: {available}")
  return FILAMENT_PROFILES[canonical]
