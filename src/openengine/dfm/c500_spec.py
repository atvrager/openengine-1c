"""NestWorks C500 Desktop CNC + 4th-Axis Rotary constraint specification.

Codifies C500 XY/Z work envelopes, rotary module limits, tool-shank pocket and
corner-radius constraints, MQL cast-iron bans, and 4th-axis fallback rules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, TypedDict
import yaml

DEFAULT_C500_CONFIG_PATH: Path = Path("config/c500_constraints.yaml")

MODE_ROTARY_4_AXIS: str = "rotary_4_axis"
MODE_THREE_AXIS_HOUSING: str = "three_axis_housing"
MODE_THREE_AXIS_FOUR_SIDE_FLIP: str = "three_axis_four_side_flip"
MODE_FALLBACK_COTS_OR_3AXIS: str = "fallback_cots_or_3axis"
ENVELOPE_AXES_COUNT: int = 3


class C500MachineConfig(TypedDict):
  """Typed YAML block for C500 machine hardware specs."""

  name: str
  has_fourth_axis: bool
  spindle_power_w: float
  max_spindle_rpm: float
  coolant_system: str
  exposed_linear_rails: bool


class C500WorkEnvelopeConfig(TypedDict):
  """Typed YAML block for C500 travel and fixture envelopes."""

  atc_xy_mm: List[float]
  max_housing_xy_mm: List[float]
  max_z_clearance_mm: float
  max_split_housing_z_mm: float
  max_four_side_barrel_z_mm: float
  rotary_chuck_od_mm: float
  rotary_max_length_mm: float
  max_rotary_part_od_mm: float
  max_rotary_part_length_mm: float


class C500ToolingConfig(TypedDict):
  """Typed YAML block for C500 spindle, tool, and thread-mill limits."""

  max_tool_shank_mm: float
  max_single_side_pocket_depth_mm: float
  min_internal_corner_radius_mm: float
  thread_milling_only: bool
  min_thread_diameter_mm: float
  max_thread_diameter_mm: float
  supported_threads: Dict[str, float]


class C500MaterialsConfig(TypedDict):
  """Typed YAML block for MQL rail-safe approved and banned materials."""

  banned_materials: List[str]
  approved_materials: List[str]


class C500PartLedgerConfig(TypedDict):
  """Typed YAML block for a single component in the 10-part ledger."""

  module: str
  mode: str
  fourth_axis_requirement: str
  envelope_mm: List[float]
  material: str
  pocket_depth_mm: float
  min_corner_radius_mm: float
  fallback_strategy: str


class C500ConfigRoot(TypedDict):
  """Typed YAML root schema for config/c500_constraints.yaml."""

  machine: C500MachineConfig
  work_envelope: C500WorkEnvelopeConfig
  tooling: C500ToolingConfig
  materials: C500MaterialsConfig
  part_ledger: Dict[str, C500PartLedgerConfig]


@dataclass(frozen=True, slots=True)
class C500WorkEnvelope:
  """Machining envelope bounds for 3-axis housings and 4th-axis rotary stock."""

  atc_xy_mm: Tuple[float, float]
  max_housing_xy_mm: Tuple[float, float]
  max_z_clearance_mm: float
  max_split_housing_z_mm: float
  max_four_side_barrel_z_mm: float
  rotary_chuck_od_mm: float
  rotary_max_length_mm: float
  max_rotary_part_od_mm: float
  max_rotary_part_length_mm: float


@dataclass(frozen=True, slots=True)
class C500ToolingRules:
  """C500 tool geometry and thread-milling capability limits."""

  max_tool_shank_mm: float
  max_single_side_pocket_depth_mm: float
  min_internal_corner_radius_mm: float
  thread_milling_only: bool
  min_thread_diameter_mm: float
  max_thread_diameter_mm: float
  supported_threads: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PartMachiningSpec:
  """Design-for-manufacturability specification for one engine component."""

  part_id: str
  module: str
  mode: str
  fourth_axis_requirement: str
  envelope_mm: Tuple[float, float, float]
  material: str
  pocket_depth_mm: float
  min_corner_radius_mm: float
  fallback_strategy: str


@dataclass(frozen=True, slots=True)
class PartEnvelopeCheck:
  """Result of validating a component against NestWorks C500 limits."""

  part_id: str
  is_valid: bool
  active_mode: str
  uses_fallback: bool
  fallback_reason: str
  violations: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class C500Spec:
  """Complete NestWorks C500 + 4th-Axis machine capability specification."""

  machine_name: str
  has_fourth_axis: bool
  spindle_power_w: float
  max_spindle_rpm: float
  coolant_system: str
  exposed_linear_rails: bool
  envelope: C500WorkEnvelope
  tooling: C500ToolingRules
  banned_materials: frozenset[str]
  approved_materials: frozenset[str]
  part_ledger: Tuple[PartMachiningSpec, ...]

  @classmethod
  def from_config(cls, cfg: C500ConfigRoot) -> "C500Spec":
    """Construct C500Spec from a parsed YAML dictionary matching C500ConfigRoot."""
    mach = cfg["machine"]
    env_cfg = cfg["work_envelope"]
    tool_cfg = cfg["tooling"]
    mat_cfg = cfg["materials"]

    atc_xy = (float(env_cfg["atc_xy_mm"][0]), float(env_cfg["atc_xy_mm"][1]))
    max_xy = (
        float(env_cfg["max_housing_xy_mm"][0]),
        float(env_cfg["max_housing_xy_mm"][1]),
    )
    envelope = C500WorkEnvelope(
        atc_xy_mm=atc_xy,
        max_housing_xy_mm=max_xy,
        max_z_clearance_mm=float(env_cfg["max_z_clearance_mm"]),
        max_split_housing_z_mm=float(env_cfg["max_split_housing_z_mm"]),
        max_four_side_barrel_z_mm=float(env_cfg["max_four_side_barrel_z_mm"]),
        rotary_chuck_od_mm=float(env_cfg["rotary_chuck_od_mm"]),
        rotary_max_length_mm=float(env_cfg["rotary_max_length_mm"]),
        max_rotary_part_od_mm=float(env_cfg["max_rotary_part_od_mm"]),
        max_rotary_part_length_mm=float(env_cfg["max_rotary_part_length_mm"]),
    )
    supported_threads = {
        str(k): float(v) for k, v in tool_cfg["supported_threads"].items()
    }
    tooling = C500ToolingRules(
        max_tool_shank_mm=float(tool_cfg["max_tool_shank_mm"]),
        max_single_side_pocket_depth_mm=float(
            tool_cfg["max_single_side_pocket_depth_mm"]
        ),
        min_internal_corner_radius_mm=float(
            tool_cfg["min_internal_corner_radius_mm"]
        ),
        thread_milling_only=bool(tool_cfg["thread_milling_only"]),
        min_thread_diameter_mm=float(tool_cfg["min_thread_diameter_mm"]),
        max_thread_diameter_mm=float(tool_cfg["max_thread_diameter_mm"]),
        supported_threads=supported_threads,
    )
    ledger = tuple(
        _parse_ledger_part(pid, p_cfg)
        for pid, p_cfg in cfg["part_ledger"].items()
    )
    return cls(
        machine_name=str(mach["name"]),
        has_fourth_axis=bool(mach["has_fourth_axis"]),
        spindle_power_w=float(mach["spindle_power_w"]),
        max_spindle_rpm=float(mach["max_spindle_rpm"]),
        coolant_system=str(mach["coolant_system"]),
        exposed_linear_rails=bool(mach["exposed_linear_rails"]),
        envelope=envelope,
        tooling=tooling,
        banned_materials=frozenset(
            str(m).strip().upper() for m in mat_cfg["banned_materials"]
        ),
        approved_materials=frozenset(
            str(m).strip().upper() for m in mat_cfg["approved_materials"]
        ),
        part_ledger=ledger,
    )

  @classmethod
  def from_yaml(cls, path: Path = DEFAULT_C500_CONFIG_PATH) -> "C500Spec":
    """Load and parse a NestWorks C500 YAML constraint file."""
    resolved = Path(path)
    if not resolved.exists():
      raise FileNotFoundError(f"C500 constraint file not found: {resolved}")
    with open(resolved, "r") as handle:
      raw_cfg: C500ConfigRoot = yaml.safe_load(handle)
    return cls.from_config(raw_cfg)

  def check_material(self, material: str) -> Optional[str]:
    """Validate material against MQL linear-rail cast-iron bans and catalog."""
    normalized = material.strip().upper()
    if normalized in self.banned_materials:
      return f"Banned under MQL linear-rail policy: {material}"
    if normalized not in self.approved_materials:
      return f"Material not in C500 approved alloy list: {material}"
    return None

  def check_thread(self, thread_spec: str) -> Optional[str]:
    """Verify a thread designation is within C500 thread-milling capability."""
    if thread_spec not in self.tooling.supported_threads:
      return f"Unsupported C500 thread-mill specification: {thread_spec}"
    return None

  def check_part(
      self,
      part: PartMachiningSpec,
      threads_used: Tuple[str, ...] = (),
      has_fourth_axis_override: Optional[bool] = None,
  ) -> PartEnvelopeCheck:
    """Validate one billet part against C500 limits and 4th-axis fallbacks."""
    effective_fourth_axis = (
        self.has_fourth_axis
        if has_fourth_axis_override is None
        else has_fourth_axis_override
    )
    violations: List[str] = []

    mat_err = self.check_material(part.material)
    if mat_err is not None:
      violations.append(mat_err)

    if part.pocket_depth_mm > self.tooling.max_single_side_pocket_depth_mm:
      violations.append(
          f"Pocket depth {part.pocket_depth_mm:.1f} mm exceeds"
          f" {self.tooling.max_single_side_pocket_depth_mm:.1f} mm limit"
      )
    if part.min_corner_radius_mm < self.tooling.min_internal_corner_radius_mm:
      violations.append(
          f"Corner radius {part.min_corner_radius_mm:.2f} mm below"
          f" {self.tooling.min_internal_corner_radius_mm:.2f} mm limit"
      )
    for thr in threads_used:
      thr_err = self.check_thread(thr)
      if thr_err is not None:
        violations.append(thr_err)

    uses_fallback = False
    active_mode = part.mode
    fallback_reason = ""
    if part.mode == MODE_ROTARY_4_AXIS and not effective_fourth_axis:
      uses_fallback = True
      active_mode = MODE_FALLBACK_COTS_OR_3AXIS
      fallback_reason = part.fallback_strategy
    else:
      _check_mode_envelope(part, self.envelope, violations)

    return PartEnvelopeCheck(
        part_id=part.part_id,
        is_valid=len(violations) == 0,
        active_mode=active_mode,
        uses_fallback=uses_fallback,
        fallback_reason=fallback_reason,
        violations=tuple(violations),
    )

  def validate_all_ledger_parts(
      self,
      has_fourth_axis_override: Optional[bool] = None,
  ) -> Tuple[PartEnvelopeCheck, ...]:
    """Validate every component in the configured C500 manufacturing ledger."""
    return tuple(
        self.check_part(part, has_fourth_axis_override=has_fourth_axis_override)
        for part in self.part_ledger
    )


def _parse_ledger_part(
    part_id: str,
    cfg: C500PartLedgerConfig,
) -> PartMachiningSpec:
  """Parse one part ledger YAML entry into a frozen PartMachiningSpec."""
  env = cfg["envelope_mm"]
  if len(env) != ENVELOPE_AXES_COUNT:
    raise ValueError(
        f"Part '{part_id}' envelope_mm must have {ENVELOPE_AXES_COUNT} values"
    )
  return PartMachiningSpec(
      part_id=part_id,
      module=str(cfg["module"]),
      mode=str(cfg["mode"]),
      fourth_axis_requirement=str(cfg["fourth_axis_requirement"]),
      envelope_mm=(float(env[0]), float(env[1]), float(env[2])),
      material=str(cfg["material"]),
      pocket_depth_mm=float(cfg["pocket_depth_mm"]),
      min_corner_radius_mm=float(cfg["min_corner_radius_mm"]),
      fallback_strategy=str(cfg["fallback_strategy"]),
  )


def _check_rotary_envelope(
    part: PartMachiningSpec,
    env: C500WorkEnvelope,
    violations: List[str],
) -> None:
  """Validate coaxial part diameter and length against 4th-axis rotary bounds."""
  dx, dy, dz = part.envelope_mm
  od = max(dx, dy)
  if od > env.max_rotary_part_od_mm:
    violations.append(
        f"Rotary OD {od:.1f} mm exceeds {env.max_rotary_part_od_mm:.1f} mm"
    )
  if dz > env.max_rotary_part_length_mm:
    violations.append(
        f"Rotary length {dz:.1f} mm exceeds"
        f" {env.max_rotary_part_length_mm:.1f} mm"
    )


def _check_housing_envelope(
    part: PartMachiningSpec,
    env: C500WorkEnvelope,
    violations: List[str],
) -> None:
  """Validate flat-mounted housing XY/Z bounds against C500 table limits."""
  dx, dy, dz = part.envelope_mm
  lim_x_small = min(env.max_housing_xy_mm)
  lim_y_large = max(env.max_housing_xy_mm)
  if min(dx, dy) > lim_x_small or max(dx, dy) > lim_y_large:
    violations.append(
        f"Housing XY ({dx:.1f} x {dy:.1f} mm) exceeds"
        f" {lim_x_small:.0f} x {lim_y_large:.0f} mm"
    )
  if dz > env.max_split_housing_z_mm:
    violations.append(
        f"Split housing Z {dz:.1f} mm exceeds"
        f" {env.max_split_housing_z_mm:.1f} mm"
    )


def _check_barrel_flip_envelope(
    part: PartMachiningSpec,
    env: C500WorkEnvelope,
    violations: List[str],
) -> None:
  """Validate 4-side indexed barrel block footprint and Z clearance."""
  dx, dy, dz = part.envelope_mm
  lim_x_small = min(env.max_housing_xy_mm)
  lim_y_large = max(env.max_housing_xy_mm)
  if min(dx, dy) > lim_x_small or max(dx, dz) > lim_y_large:
    violations.append(
        f"4-side barrel footprint ({dx:.1f} x {dy:.1f} x {dz:.1f} mm)"
        f" exceeds {lim_x_small:.0f} x {lim_y_large:.0f} mm flip bounds"
    )
  if max(dx, dy) > env.max_four_side_barrel_z_mm:
    violations.append(
        f"4-side flip height {max(dx, dy):.1f} mm exceeds"
        f" {env.max_four_side_barrel_z_mm:.1f} mm Z setup cap"
    )


def _check_mode_envelope(
    part: PartMachiningSpec,
    env: C500WorkEnvelope,
    violations: List[str],
) -> None:
  """Dispatch work-envelope check by active machining mode."""
  checkers = {
      MODE_ROTARY_4_AXIS: _check_rotary_envelope,
      MODE_THREE_AXIS_HOUSING: _check_housing_envelope,
      MODE_THREE_AXIS_FOUR_SIDE_FLIP: _check_barrel_flip_envelope,
  }
  checker = checkers.get(part.mode)
  if checker is None:
    violations.append(f"Unrecognized C500 machining mode: {part.mode}")
    return
  checker(part, env, violations)
