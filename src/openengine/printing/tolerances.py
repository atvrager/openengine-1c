"""Process-aware FDM printing tolerances, shrinkage compensation, and snap-fit mechanics.

FDM minimum walls and clearances are governed by nozzle extrusion width, layer
stair-stepping, and polymer shrinkage rather than naive geometric scaling.
"""

from dataclasses import dataclass
import math

from build123d import Part

from openengine.printing.materials import (
    DEFAULT_FILAMENT_NAME,
    FilamentProfile,
    get_filament_profile,
)

MIN_KIT_SCALE: float = 0.25
MAX_KIT_SCALE: float = 2.0
DEFAULT_KIT_SCALE: float = 0.5
DEFAULT_NOZZLE_DIAMETER_MM: float = 0.4
DEFAULT_LAYER_HEIGHT_MM: float = 0.2
DEFAULT_PRINT_VOLUME_MM: tuple[float, float, float] = (250.0, 220.0, 270.0)
DEFAULT_INFILL_PCT: float = 20.0
DEFAULT_WALL_LOOPS: int = 3
DEFAULT_TOP_BOTTOM_LAYERS: int = 4
DEFAULT_VOLUMETRIC_FLOW_MM3_S: float = 8.0
DEFAULT_SUPPORT_MASS_OVERHEAD_PCT: float = 15.0
DEFAULT_BRIM_WIDTH_MM: float = 5.0
MESH_QUALITY_PRESETS: dict[str, tuple[float, float]] = {
    "draft": (0.05, 0.20),
    "standard": (0.02, 0.08),
    "fine": (0.01, 0.05),
    "ultra": (0.005, 0.03),
}
VALID_MESH_QUALITIES: tuple[str, ...] = tuple(MESH_QUALITY_PRESETS.keys())
DEFAULT_MESH_QUALITY: str = "ultra"
VALID_PLATE_GROUP_MODES: tuple[str, ...] = ("fewest", "slicer", "category")
DEFAULT_PLATE_GROUP_MODE: str = "fewest"
DEFAULT_PLATE_MARGIN_MM: float = 8.0
DEFAULT_PLATE_GAP_MM: float = 6.0
DEFAULT_BRIM_PLATE_GAP_MM: float = 12.0
MIN_PLATE_MARGIN_MM: float = 0.0
MIN_PLATE_GAP_MM: float = 0.5
MAX_SUPPORT_MASS_OVERHEAD_PCT: float = 200.0
MAX_BRIM_WIDTH_MM: float = 25.0


def resolve_mesh_deflections(mesh_quality: str) -> tuple[float, float]:
  """Return ``(linear_deflection_mm, angular_deflection_rad)`` for a named mesh quality preset."""
  key = str(mesh_quality).strip().lower()
  if key not in MESH_QUALITY_PRESETS:
    raise ValueError(
        f"mesh_quality must be one of {VALID_MESH_QUALITIES}, got {mesh_quality!r}"
    )
  return MESH_QUALITY_PRESETS[key]

MIN_NOZZLE_DIAMETER_MM: float = 0.15
MAX_NOZZLE_DIAMETER_MM: float = 1.2
MIN_LAYER_HEIGHT_MM: float = 0.04
MIN_CLEARANCE_MM: float = 0.05
MIN_PROCESS_DELTA_MM: float = -0.04
NOZZLE_CLEARANCE_SLOPE: float = 0.15
LAYER_CLEARANCE_SLOPE: float = 0.20
PRESS_DELTA_WEIGHT: float = 0.5
MAX_VALID_SHRINKAGE: float = 0.25

NON_STRUCTURAL_WALL_NOZZLE_MULT: float = 2.0
FIN_NOZZLE_MULT: float = 2.5

DEFAULT_SNAP_UNDERCUT_MM: float = 1.0
DEFAULT_SNAP_BEAM_LENGTH_MM: float = 14.0
DEFAULT_SNAP_ROOT_THICKNESS_MM: float = 1.6
DEFAULT_SNAP_BEAM_WIDTH_MM: float = 6.0
DEFAULT_SNAP_TAPER_RATIO: float = 0.6
MIN_SNAP_TAPER_RATIO: float = 0.25
MAX_SNAP_TAPER_RATIO: float = 1.0
DEFAULT_SNAP_LEAD_ANGLE_DEG: float = 30.0
DEFAULT_SNAP_RETENTION_ANGLE_DEG: float = 75.0
DEFAULT_SNAP_STRESS_CONC: float = 1.25
DEFAULT_SNAP_SAFETY_FACTOR: float = 1.5
TARGET_SNAP_FORCE_PER_MM_WIDTH_N: float = 1.25
DEFAULT_SNAP_TIP_FLAT_MM: float = 0.8
AUTO_SNAP_UNDERCUT_LENGTH_RATIO: float = 0.11
MIN_AUTO_SNAP_UNDERCUT_MM: float = 0.45
DESIRED_MIN_AUTO_UNDERCUT_MM: float = 0.88
MAX_AUTO_SNAP_UNDERCUT_MM: float = 1.5
MAX_AUTO_SNAP_SLENDERNESS_RATIO: float = 0.25
ROOT_FILLET_THICKNESS_RATIO: float = 0.35
MIN_ROOT_FILLET_MM: float = 0.3
MAX_ROOT_FILLET_MM: float = 0.8
MAX_ROOT_FILLET_LENGTH_RATIO: float = 0.25
TAPER_HALF_THRESHOLD: float = 0.5
UNIFORM_BEAM_INTEGRAL: float = 1.0 / 3.0
BEAM_FLEXURE_COEFF: float = 12.0
BEAM_ROOT_STRESS_COEFF: float = 6.0
SELF_LOCK_MARGIN: float = 1.0e-3


@dataclass
class PrintKitConfig:
  """Configuration parameters for generating a 3D-printable OpenEngine model kit.

  Attributes:
    scale: Model scale factor relative to full-size engine, validated in [0.25, 2.0].
    filament: Filament profile name or alias (defaults to ``"PETG"``).
    nozzle_diameter_mm: Extruder nozzle orifice diameter in mm.
    layer_height_mm: Slicer layer height in mm.
    print_volume_mm: Printer build volume ``(X, Y, Z)`` in mm (default ``250x220x270``).
    cutaway: Whether to generate sectioned display cutaway housings.
    infill_pct: Target slicer infill percentage in [0.0, 100.0].
    wall_loops: Perimeter wall loop count (>= 1).
    top_bottom_layers: Solid top and bottom shell layer count (>= 1).
    volumetric_flow_mm3_s: Effective average volumetric extrusion flow in mm^3/s (> 0).
    support_mass_overhead_pct: Extra filament mass overhead percentage for tree supports (half for buildplate-only).
    brim_width_mm: Outer brim width in mm applied when a part needs a brim (>= 0).
    mesh_quality: STL tessellation smoothness preset (``"draft"``, ``"standard"``, ``"fine"``, ``"ultra"``).
    compensate_shrinkage: Whether to bake thermal shrinkage compensation into exported STLs.
    clearance_bias_mm: Additive user calibration offset (positive loosens fits).
    include_coupon: Whether to include the filament tolerance calibration coupon.
    pack_plates: Whether to pack kit parts into merged build-plate meta-STLs.
    plate_group_mode: Plate grouping strategy (``"fewest"``, ``"slicer"``, ``"category"``).
    plate_margin_mm: Minimum print-volume edge clearance margin in mm on packed plates.
    plate_gap_mm: Minimum inter-part XY spacing in mm on non-brim packed plates.
  """

  scale: float = DEFAULT_KIT_SCALE
  filament: str = DEFAULT_FILAMENT_NAME
  nozzle_diameter_mm: float = DEFAULT_NOZZLE_DIAMETER_MM
  layer_height_mm: float = DEFAULT_LAYER_HEIGHT_MM
  print_volume_mm: tuple[float, float, float] = DEFAULT_PRINT_VOLUME_MM
  cutaway: bool = False
  infill_pct: float = DEFAULT_INFILL_PCT
  wall_loops: int = DEFAULT_WALL_LOOPS
  top_bottom_layers: int = DEFAULT_TOP_BOTTOM_LAYERS
  volumetric_flow_mm3_s: float = DEFAULT_VOLUMETRIC_FLOW_MM3_S
  support_mass_overhead_pct: float = DEFAULT_SUPPORT_MASS_OVERHEAD_PCT
  brim_width_mm: float = DEFAULT_BRIM_WIDTH_MM
  mesh_quality: str = DEFAULT_MESH_QUALITY
  compensate_shrinkage: bool = True
  clearance_bias_mm: float = 0.0
  include_coupon: bool = True
  pack_plates: bool = True
  plate_group_mode: str = DEFAULT_PLATE_GROUP_MODE
  plate_margin_mm: float = DEFAULT_PLATE_MARGIN_MM
  plate_gap_mm: float = DEFAULT_PLATE_GAP_MM

  def __post_init__(self) -> None:
    self.scale = float(self.scale)
    if not (MIN_KIT_SCALE <= self.scale <= MAX_KIT_SCALE):
      raise ValueError(
          f"PrintKitConfig.scale must be in [{MIN_KIT_SCALE}, {MAX_KIT_SCALE}], got {self.scale}"
      )

    # Validate filament profile resolution immediately so typos fail fast.
    _ = get_filament_profile(self.filament)

    self.nozzle_diameter_mm = float(self.nozzle_diameter_mm)
    if not (MIN_NOZZLE_DIAMETER_MM <= self.nozzle_diameter_mm <= MAX_NOZZLE_DIAMETER_MM):
      raise ValueError(
          f"nozzle_diameter_mm must be in [{MIN_NOZZLE_DIAMETER_MM}, {MAX_NOZZLE_DIAMETER_MM}], "
          f"got {self.nozzle_diameter_mm}"
      )

    self.layer_height_mm = float(self.layer_height_mm)
    if not (MIN_LAYER_HEIGHT_MM <= self.layer_height_mm <= self.nozzle_diameter_mm):
      raise ValueError(
          f"layer_height_mm must be in [{MIN_LAYER_HEIGHT_MM}, nozzle_diameter_mm="
          f"{self.nozzle_diameter_mm}], got {self.layer_height_mm}"
      )

    if len(self.print_volume_mm) != 3 or any(dim <= 0.0 for dim in self.print_volume_mm):
      raise ValueError(f"print_volume_mm must have 3 positive dimensions, got {self.print_volume_mm}")
    self.print_volume_mm = tuple(float(dim) for dim in self.print_volume_mm)  # type: ignore[assignment]

    self.infill_pct = float(self.infill_pct)
    if not (0.0 <= self.infill_pct <= 100.0):
      raise ValueError(f"infill_pct must be in [0.0, 100.0], got {self.infill_pct}")

    if self.wall_loops < 1:
      raise ValueError(f"wall_loops must be >= 1, got {self.wall_loops}")

    if self.top_bottom_layers < 1:
      raise ValueError(f"top_bottom_layers must be >= 1, got {self.top_bottom_layers}")

    self.volumetric_flow_mm3_s = float(self.volumetric_flow_mm3_s)
    if self.volumetric_flow_mm3_s <= 0.0:
      raise ValueError(
          f"volumetric_flow_mm3_s must be > 0, got {self.volumetric_flow_mm3_s}"
      )

    self.support_mass_overhead_pct = float(self.support_mass_overhead_pct)
    if not (0.0 <= self.support_mass_overhead_pct <= MAX_SUPPORT_MASS_OVERHEAD_PCT):
      raise ValueError(
          f"support_mass_overhead_pct must be in [0.0, {MAX_SUPPORT_MASS_OVERHEAD_PCT}], "
          f"got {self.support_mass_overhead_pct}"
      )

    self.brim_width_mm = float(self.brim_width_mm)
    if not (0.0 <= self.brim_width_mm <= MAX_BRIM_WIDTH_MM):
      raise ValueError(
          f"brim_width_mm must be in [0.0, {MAX_BRIM_WIDTH_MM}], got {self.brim_width_mm}"
      )

    self.mesh_quality = str(self.mesh_quality).strip().lower()
    _ = resolve_mesh_deflections(self.mesh_quality)

    self.compensate_shrinkage = bool(self.compensate_shrinkage)
    self.clearance_bias_mm = float(self.clearance_bias_mm)
    self.pack_plates = bool(self.pack_plates)

    self.plate_group_mode = str(self.plate_group_mode).strip().lower()
    if self.plate_group_mode not in VALID_PLATE_GROUP_MODES:
      raise ValueError(
          f"plate_group_mode must be one of {VALID_PLATE_GROUP_MODES}, got {self.plate_group_mode!r}"
      )

    self.plate_margin_mm = float(self.plate_margin_mm)
    if self.plate_margin_mm < MIN_PLATE_MARGIN_MM:
      raise ValueError(
          f"plate_margin_mm must be >= {MIN_PLATE_MARGIN_MM}, got {self.plate_margin_mm}"
      )

    self.plate_gap_mm = float(self.plate_gap_mm)
    if self.plate_gap_mm < MIN_PLATE_GAP_MM:
      raise ValueError(
          f"plate_gap_mm must be >= {MIN_PLATE_GAP_MM}, got {self.plate_gap_mm}"
      )

  @property
  def filament_profile(self) -> FilamentProfile:
    """Resolve the active ``FilamentProfile`` for ``self.filament``."""
    return get_filament_profile(self.filament)

  @property
  def mesh_deflections(self) -> tuple[float, float]:
    """Return ``(linear_deflection_mm, angular_deflection_rad)`` for ``self.mesh_quality``."""
    return resolve_mesh_deflections(self.mesh_quality)


@dataclass(frozen=True, slots=True)
class KitPart:
  """Named 3D-printable model kit part with quantity and orientation metadata."""

  name: str
  part: Part
  quantity: int = 1
  category: str = "general"
  preferred_rotation_xyz_deg: tuple[float, float, float] | None = None
  notes: str = ""

  def __post_init__(self) -> None:
    if not self.name.strip():
      raise ValueError("KitPart.name must be non-empty")
    if self.quantity < 1:
      raise ValueError(f"KitPart.quantity must be >= 1, got {self.quantity}")


@dataclass(frozen=True, slots=True)
class SnapFitDesign:
  """Calculated geometry and structural mechanics for a tapered cantilever snap hook."""

  filament: str
  beam_length_mm: float
  root_thickness_mm: float
  tip_thickness_mm: float
  beam_width_mm: float
  undercut_mm: float
  lead_angle_deg: float
  retention_angle_deg: float
  hook_tip_flat_mm: float
  root_fillet_mm: float
  nominal_strain: float
  max_strain: float
  allowable_strain: float
  stress_concentration_factor: float
  safety_factor: float
  is_safe: bool
  deflection_force_n: float
  mating_force_n: float
  clearance_mm: float


def _process_clearance_delta_mm(cfg: PrintKitConfig) -> float:
  """Compute process-driven clearance shift from nozzle and layer height deltas."""
  delta = (
      NOZZLE_CLEARANCE_SLOPE * (cfg.nozzle_diameter_mm - DEFAULT_NOZZLE_DIAMETER_MM)
      + LAYER_CLEARANCE_SLOPE * (cfg.layer_height_mm - DEFAULT_LAYER_HEIGHT_MM)
  )
  return max(MIN_PROCESS_DELTA_MM, delta)


def _adjusted_clearance_mm(base_mm: float, cfg: PrintKitConfig) -> float:
  """Apply process delta and user bias while enforcing a non-binding floor."""
  return max(MIN_CLEARANCE_MM, base_mm + _process_clearance_delta_mm(cfg) + cfg.clearance_bias_mm)


def min_printable_wall_mm(cfg: PrintKitConfig, structural: bool = True) -> float:
  """Return minimum FDM-printable wall thickness in mm independent of kit scale.

  Structural walls honor both filament wall multiplier and configured ``wall_loops``.
  Non-structural cosmetic walls enforce at least two perimeter extrusion lines.
  """
  profile = cfg.filament_profile
  nozzle = cfg.nozzle_diameter_mm
  if structural:
    mult = max(profile.min_wall_nozzle_mult, float(cfg.wall_loops))
    return mult * nozzle
  non_struct_mult = max(NON_STRUCTURAL_WALL_NOZZLE_MULT, profile.min_wall_nozzle_mult - 1.0)
  return non_struct_mult * nozzle


def min_fin_thickness_mm(cfg: PrintKitConfig) -> float:
  """Return minimum printable cooling-fin thickness in mm independent of kit scale."""
  return max(FIN_NOZZLE_MULT * cfg.nozzle_diameter_mm, min_printable_wall_mm(cfg, structural=False))


def shrinkage_compensation_xyz(profile: FilamentProfile) -> tuple[float, float, float]:
  """Return multiplicative ``(sx, sy, sz)`` scale compensation for thermal shrinkage."""
  for label, shrink in (("shrinkage_xy", profile.shrinkage_xy), ("shrinkage_z", profile.shrinkage_z)):
    if not (0.0 <= shrink < MAX_VALID_SHRINKAGE):
      raise ValueError(f"{label} must be in [0.0, {MAX_VALID_SHRINKAGE}), got {shrink}")
  k_xy = 1.0 / (1.0 - profile.shrinkage_xy)
  k_z = 1.0 / (1.0 - profile.shrinkage_z)
  return (k_xy, k_xy, k_z)


def running_clearance_radial_mm(cfg: PrintKitConfig) -> float:
  """Return radial clearance in mm for rotating or freely sliding FDM joints."""
  return _adjusted_clearance_mm(cfg.filament_profile.radial_running_clearance_mm, cfg)


def slip_clearance_radial_mm(cfg: PrintKitConfig) -> float:
  """Return radial clearance in mm for snug hand-assembled slip/keyed fits."""
  return _adjusted_clearance_mm(cfg.filament_profile.radial_slip_clearance_mm, cfg)


def snap_clearance_mm(cfg: PrintKitConfig) -> float:
  """Return pocket clearance in mm around cantilever snap-fit hooks."""
  return _adjusted_clearance_mm(cfg.filament_profile.snap_clearance_mm, cfg)


def press_interference_radial_mm(cfg: PrintKitConfig) -> float:
  """Return radial press-fit interference in mm adjusted for nozzle and user bias."""
  raw = cfg.filament_profile.radial_press_interference_mm
  adj = PRESS_DELTA_WEIGHT * (_process_clearance_delta_mm(cfg) + cfg.clearance_bias_mm)
  return max(0.0, raw - adj)


def _tapered_beam_integral(taper_ratio: float) -> float:
  """Exact Euler-Bernoulli flexibility integral for a linearly tapered rectangle.

  Evaluates ``int_0^1 (1 - u)^2 / (1 - (1 - r)*u)^3 du`` where ``r = h_tip / h_root``.
  Equals ``1/3`` for uniform thickness (``r = 1``) and ``-5 + 8*ln(2)`` (~0.5452) at ``r = 0.5``.
  """
  if abs(1.0 - taper_ratio) < 1.0e-6:
    return UNIFORM_BEAM_INTEGRAL
  one_minus_r = 1.0 - taper_ratio
  numerator = (taper_ratio * taper_ratio) - (4.0 * taper_ratio) + 3.0 + (2.0 * math.log(taper_ratio))
  return numerator / (-2.0 * (one_minus_r ** 3))


def _peak_to_root_strain_ratio(taper_ratio: float) -> float:
  """Return ratio of peak bending strain along the beam to nominal root strain.

  For ``taper_ratio >= 0.5``, strain peaks strictly at the clamped root. For sharper
  tapers (``taper_ratio < 0.5``), peak strain shifts outboard along the taper.
  """
  if taper_ratio >= TAPER_HALF_THRESHOLD:
    return 1.0
  return 1.0 / (4.0 * taper_ratio * (1.0 - taper_ratio))


def design_cantilever_snap(
    cfg: PrintKitConfig,
    target_undercut_mm: float = DEFAULT_SNAP_UNDERCUT_MM,
    beam_length_mm: float = DEFAULT_SNAP_BEAM_LENGTH_MM,
    root_thickness_mm: float = DEFAULT_SNAP_ROOT_THICKNESS_MM,
    beam_width_mm: float = DEFAULT_SNAP_BEAM_WIDTH_MM,
    *,
    taper_ratio: float = DEFAULT_SNAP_TAPER_RATIO,
    tip_thickness_mm: float | None = None,
    lead_angle_deg: float = DEFAULT_SNAP_LEAD_ANGLE_DEG,
    retention_angle_deg: float = DEFAULT_SNAP_RETENTION_ANGLE_DEG,
    stress_concentration_factor: float = DEFAULT_SNAP_STRESS_CONC,
) -> SnapFitDesign:
  """Compute tapered cantilever snap-fit mechanics for the active filament profile.

  Uses exact linearly tapered Euler-Bernoulli beam bending for deflection force ``P``,
  root stress-concentration strain ``max_strain``, safety factor against
  ``profile.allowable_snap_strain``, and friction-amplified mating insertion force ``W``.
  """
  if target_undercut_mm <= 0.0:
    raise ValueError(f"target_undercut_mm must be > 0, got {target_undercut_mm}")
  if beam_length_mm <= 0.0:
    raise ValueError(f"beam_length_mm must be > 0, got {beam_length_mm}")
  if root_thickness_mm <= 0.0:
    raise ValueError(f"root_thickness_mm must be > 0, got {root_thickness_mm}")
  if beam_width_mm <= 0.0:
    raise ValueError(f"beam_width_mm must be > 0, got {beam_width_mm}")
  if not (0.0 < lead_angle_deg < 89.0):
    raise ValueError(f"lead_angle_deg must be in (0, 89), got {lead_angle_deg}")
  if not (15.0 <= retention_angle_deg <= 90.0):
    raise ValueError(f"retention_angle_deg must be in [15, 90], got {retention_angle_deg}")
  if stress_concentration_factor < 1.0:
    raise ValueError(
        f"stress_concentration_factor must be >= 1.0, got {stress_concentration_factor}"
    )

  if tip_thickness_mm is None:
    if not (MIN_SNAP_TAPER_RATIO <= taper_ratio <= MAX_SNAP_TAPER_RATIO):
      raise ValueError(
          f"taper_ratio must be in [{MIN_SNAP_TAPER_RATIO}, {MAX_SNAP_TAPER_RATIO}], got {taper_ratio}"
      )
    min_printable_tip = min(root_thickness_mm, NON_STRUCTURAL_WALL_NOZZLE_MULT * cfg.nozzle_diameter_mm)
    resolved_tip_mm = max(min_printable_tip, root_thickness_mm * taper_ratio)
  else:
    if not (0.0 < tip_thickness_mm <= root_thickness_mm):
      raise ValueError(
          f"tip_thickness_mm must be in (0, root_thickness_mm={root_thickness_mm}], "
          f"got {tip_thickness_mm}"
      )
    resolved_tip_mm = float(tip_thickness_mm)

  actual_taper = resolved_tip_mm / root_thickness_mm
  profile = cfg.filament_profile
  integ = _tapered_beam_integral(actual_taper)

  deflection_force_n = (
      profile.flexural_modulus_mpa
      * beam_width_mm
      * (root_thickness_mm ** 3)
      * target_undercut_mm
  ) / (BEAM_FLEXURE_COEFF * (beam_length_mm ** 3) * integ)

  root_strain_nom = (
      BEAM_ROOT_STRESS_COEFF * deflection_force_n * beam_length_mm
  ) / (
      profile.flexural_modulus_mpa * beam_width_mm * (root_thickness_mm ** 2)
  )
  nominal_strain = root_strain_nom * _peak_to_root_strain_ratio(actual_taper)
  max_strain = stress_concentration_factor * nominal_strain
  safety_factor = profile.allowable_snap_strain / max(1.0e-12, max_strain)

  mu = profile.sliding_friction_coeff
  tan_lead = math.tan(math.radians(lead_angle_deg))
  denom = 1.0 - (mu * tan_lead)
  if denom <= SELF_LOCK_MARGIN:
    raise ValueError(
        f"Self-locking lead angle {lead_angle_deg} deg for friction coeff {mu}"
    )
  mating_force_n = deflection_force_n * ((mu + tan_lead) / denom)

  tip_flat_mm = max(DEFAULT_SNAP_TIP_FLAT_MM, NON_STRUCTURAL_WALL_NOZZLE_MULT * cfg.nozzle_diameter_mm)
  root_fillet_mm = min(
      max(MIN_ROOT_FILLET_MM, ROOT_FILLET_THICKNESS_RATIO * root_thickness_mm),
      MAX_ROOT_FILLET_MM,
      MAX_ROOT_FILLET_LENGTH_RATIO * beam_length_mm,
  )

  return SnapFitDesign(
      filament=profile.name,
      beam_length_mm=float(beam_length_mm),
      root_thickness_mm=float(root_thickness_mm),
      tip_thickness_mm=float(resolved_tip_mm),
      beam_width_mm=float(beam_width_mm),
      undercut_mm=float(target_undercut_mm),
      lead_angle_deg=float(lead_angle_deg),
      retention_angle_deg=float(retention_angle_deg),
      hook_tip_flat_mm=float(tip_flat_mm),
      root_fillet_mm=float(root_fillet_mm),
      nominal_strain=float(nominal_strain),
      max_strain=float(max_strain),
      allowable_strain=float(profile.allowable_snap_strain),
      stress_concentration_factor=float(stress_concentration_factor),
      safety_factor=float(safety_factor),
      is_safe=safety_factor >= 1.0,
      deflection_force_n=float(deflection_force_n),
      mating_force_n=float(mating_force_n),
      clearance_mm=snap_clearance_mm(cfg),
  )


def auto_size_cantilever_snap(
    cfg: PrintKitConfig,
    available_length_mm: float,
    width_mm: float = DEFAULT_SNAP_BEAM_WIDTH_MM,
    *,
    target_safety_factor: float = DEFAULT_SNAP_SAFETY_FACTOR,
    lead_angle_deg: float = DEFAULT_SNAP_LEAD_ANGLE_DEG,
    retention_angle_deg: float = DEFAULT_SNAP_RETENTION_ANGLE_DEG,
    stress_concentration_factor: float = DEFAULT_SNAP_STRESS_CONC,
) -> SnapFitDesign:
  """Auto-size a tapered cantilever snap hook for the active filament and space envelope.

  Brittle polymers (e.g. PLA, PETG-CF) are automatically thinned and undercut-trimmed
  to preserve ``target_safety_factor`` against ``allowable_snap_strain``, while ductile
  polymers (e.g. PETG, ABS, ASA) target crisp tactile snap engagement force.
  """
  if available_length_mm <= 0.0:
    raise ValueError(f"available_length_mm must be > 0, got {available_length_mm}")
  if width_mm <= 0.0:
    raise ValueError(f"width_mm must be > 0, got {width_mm}")
  if target_safety_factor < 1.0:
    raise ValueError(f"target_safety_factor must be >= 1.0, got {target_safety_factor}")

  profile = cfg.filament_profile
  beam_len = float(available_length_mm)
  min_tip_mm = NON_STRUCTURAL_WALL_NOZZLE_MULT * cfg.nozzle_diameter_mm
  min_root_mm = max(min_tip_mm, min_printable_wall_mm(cfg, structural=False))

  desired_undercut = min(
      MAX_AUTO_SNAP_UNDERCUT_MM,
      max(DESIRED_MIN_AUTO_UNDERCUT_MM, AUTO_SNAP_UNDERCUT_LENGTH_RATIO * beam_len),
  )
  target_force_n = TARGET_SNAP_FORCE_PER_MM_WIDTH_N * width_mm
  nom_taper = DEFAULT_SNAP_TAPER_RATIO
  nom_integ = _tapered_beam_integral(nom_taper)

  h_for_force = (
      (target_force_n * BEAM_FLEXURE_COEFF * (beam_len ** 3) * nom_integ)
      / (profile.flexural_modulus_mpa * width_mm * desired_undercut)
  ) ** (1.0 / 3.0)

  hy_budget = (
      2.0 * (beam_len ** 2) * nom_integ * profile.allowable_snap_strain
  ) / (stress_concentration_factor * target_safety_factor)
  h_for_strain = hy_budget / desired_undercut

  max_root_mm = max(min_root_mm, MAX_AUTO_SNAP_SLENDERNESS_RATIO * beam_len)
  root_t = min(max(min_root_mm, min(h_for_force, h_for_strain)), max_root_mm)

  tip_t = min(root_t, max(min_tip_mm, nom_taper * root_t))
  actual_taper = tip_t / root_t
  actual_integ = _tapered_beam_integral(actual_taper)
  strain_ratio = _peak_to_root_strain_ratio(actual_taper)

  undercut_budget = (
      2.0 * (beam_len ** 2) * actual_integ * profile.allowable_snap_strain
  ) / (stress_concentration_factor * target_safety_factor * root_t * strain_ratio)
  undercut = min(desired_undercut, max(MIN_AUTO_SNAP_UNDERCUT_MM, undercut_budget))

  return design_cantilever_snap(
      cfg=cfg,
      target_undercut_mm=undercut,
      beam_length_mm=beam_len,
      root_thickness_mm=root_t,
      beam_width_mm=width_mm,
      tip_thickness_mm=tip_t,
      lead_angle_deg=lead_angle_deg,
      retention_angle_deg=retention_angle_deg,
      stress_concentration_factor=stress_concentration_factor,
  )
