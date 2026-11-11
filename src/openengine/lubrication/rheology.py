"""Temperature-Viscosity Rheology for Wet-Sump Engine Lubricants.

Implements ASTM D341 Walther kinematic viscosity and Vogel dynamic viscosity
correlations alongside temperature-dependent density, specific heat, and thermal
conductivity for multigrade powersports oils (`SAE 10W-40` and `SAE 20W-50`)
from 20 deg C to 140 deg C.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Dict, Tuple, TypedDict
import numpy as np

if TYPE_CHECKING:
  from openengine.dynamics.bearing import OilProperties

KELVIN_OFFSET: float = 273.15
REF_TEMP_15C: float = 15.0
REF_TEMP_20C: float = 20.0
REF_TEMP_40C: float = 40.0
REF_TEMP_80C: float = 80.0
REF_TEMP_100C: float = 100.0
REF_TEMP_130C: float = 130.0
MIN_VALID_TEMP_C: float = -20.0
MAX_VALID_TEMP_C: float = 180.0

WALTHER_SHIFT_CST: float = 0.7
LOG_BASE_10: float = 10.0
CST_TO_M2_S: float = 1.0e-6
PA_S_TO_MPA_S: float = 1.0e3

THERMAL_EXPANSION_KG_M3_K: float = 0.65
OIL_CP_AT_20C_J_KG_K: float = 1880.0
OIL_CP_SLOPE_J_KG_K2: float = 3.85
OIL_CONDUCTIVITY_AT_20C_W_M_K: float = 0.145
OIL_CONDUCTIVITY_SLOPE_W_M_K2: float = 7.5e-5
MIN_CONDUCTIVITY_W_M_K: float = 0.10

# Standard multigrade reference specifications (SAE J300 / JASO MA2)
SAE_10W40_NAME: str = "SAE 10W-40"
SAE_10W40_NU_40_CST: float = 92.0
SAE_10W40_NU_100_CST: float = 14.2
SAE_10W40_RHO_15_KG_M3: float = 862.0

SAE_20W50_NAME: str = "SAE 20W-50"
SAE_20W50_NU_40_CST: float = 158.0
SAE_20W50_NU_100_CST: float = 19.0
SAE_20W50_RHO_15_KG_M3: float = 878.0


class LubricationConfigSection(TypedDict, total=False):
  """Typed YAML schema for the lubrication configuration block."""

  system: str
  oil_grade: str
  sump_volume_l: float
  dipper_immersion_mm: float
  dipper_width_mm: float
  dipper_tip_radius_mm: float
  sump_fin_area_m2: float
  cooling_airflow_m_s: float
  max_oil_temp_c: float


class EngineLubricationConfigRoot(TypedDict, total=False):
  """Root YAML slice containing the lubrication block."""

  lubrication: LubricationConfigSection


@dataclass(frozen=True, slots=True)
class OilState:
  """Thermophysical state of engine oil at a specific temperature."""

  grade: str
  temperature_c: float
  density_kg_m3: float
  kinematic_viscosity_cst: float
  dynamic_viscosity_pa_s: float
  vogel_dynamic_viscosity_pa_s: float
  specific_heat_j_kg_k: float
  thermal_conductivity_w_m_k: float
  prandtl_number: float


@dataclass(frozen=True, slots=True)
class OilGradeSpec:
  """Multigrade engine oil specification with Walther and Vogel parameters."""

  grade: str
  nu_40_cst: float
  nu_100_cst: float
  density_15c_kg_m3: float
  walther_a: float
  walther_b: float
  vogel_a_pa_s: float
  vogel_b_c: float
  vogel_c_c: float

  @classmethod
  def from_viscometric_targets(
      cls,
      grade: str,
      nu_40_cst: float,
      nu_100_cst: float,
      density_15c_kg_m3: float,
  ) -> "OilGradeSpec":
    """Fit exact Walther and 3-point Vogel parameters from 40C/100C targets."""
    walther_a, walther_b = fit_walther_coefficients(nu_40_cst, nu_100_cst)
    vogel_a, vogel_b, vogel_c = _fit_vogel_from_walther(
        walther_a=walther_a,
        walther_b=walther_b,
        density_15c_kg_m3=density_15c_kg_m3,
    )
    return cls(
        grade=grade,
        nu_40_cst=nu_40_cst,
        nu_100_cst=nu_100_cst,
        density_15c_kg_m3=density_15c_kg_m3,
        walther_a=walther_a,
        walther_b=walther_b,
        vogel_a_pa_s=vogel_a,
        vogel_b_c=vogel_b,
        vogel_c_c=vogel_c,
    )

  @classmethod
  def sae_10w40(cls) -> "OilGradeSpec":
    """Return standard JASO MA2 SAE 10W-40 multigrade oil specification."""
    return cls.from_viscometric_targets(
        grade=SAE_10W40_NAME,
        nu_40_cst=SAE_10W40_NU_40_CST,
        nu_100_cst=SAE_10W40_NU_100_CST,
        density_15c_kg_m3=SAE_10W40_RHO_15_KG_M3,
    )

  @classmethod
  def sae_20w50(cls) -> "OilGradeSpec":
    """Return heavy air-cooled SAE 20W-50 multigrade oil specification."""
    return cls.from_viscometric_targets(
        grade=SAE_20W50_NAME,
        nu_40_cst=SAE_20W50_NU_40_CST,
        nu_100_cst=SAE_20W50_NU_100_CST,
        density_15c_kg_m3=SAE_20W50_RHO_15_KG_M3,
    )

  @classmethod
  def from_name(cls, grade_name: str) -> "OilGradeSpec":
    """Lookup supported oil grade by name ('SAE 10W-40' or 'SAE 20W-50')."""
    normalized = grade_name.strip().upper().replace("_", " ")
    catalog: Dict[str, OilGradeSpec] = {
        SAE_10W40_NAME: cls.sae_10w40(),
        "10W-40": cls.sae_10w40(),
        "10W40": cls.sae_10w40(),
        SAE_20W50_NAME: cls.sae_20w50(),
        "20W-50": cls.sae_20w50(),
        "20W50": cls.sae_20w50(),
    }
    if normalized not in catalog:
      raise ValueError(f"Unsupported oil grade: {grade_name}")
    return catalog[normalized]

  def density_kg_m3(self, temp_c: float) -> float:
    """Evaluate liquid oil density [kg/m^3] at temperature temp_c [deg C]."""
    t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
    return float(
        self.density_15c_kg_m3 - THERMAL_EXPANSION_KG_M3_K * (t_clamped - REF_TEMP_15C)
    )

  def kinematic_viscosity_walther_cst(self, temp_c: float) -> float:
    """Evaluate kinematic viscosity [cSt] via the ASTM D341 Walther equation."""
    return walther_kinematic_viscosity_cst(temp_c, self.walther_a, self.walther_b)

  def dynamic_viscosity_walther_pa_s(self, temp_c: float) -> float:
    """Evaluate dynamic viscosity [Pa.s] from Walther nu(T) and rho(T)."""
    nu_m2_s = self.kinematic_viscosity_walther_cst(temp_c) * CST_TO_M2_S
    return float(self.density_kg_m3(temp_c) * nu_m2_s)

  def dynamic_viscosity_vogel_pa_s(self, temp_c: float) -> float:
    """Evaluate dynamic viscosity [Pa.s] via the 3-parameter Vogel equation."""
    t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
    return float(
        self.vogel_a_pa_s * math.exp(self.vogel_b_c / (t_clamped + self.vogel_c_c))
    )

  def specific_heat_j_kg_k(self, temp_c: float) -> float:
    """Evaluate oil specific heat capacity c_p(T) [J/(kg.K)]."""
    t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
    return float(OIL_CP_AT_20C_J_KG_K + OIL_CP_SLOPE_J_KG_K2 * (t_clamped - REF_TEMP_20C))

  def thermal_conductivity_w_m_k(self, temp_c: float) -> float:
    """Evaluate oil thermal conductivity k(T) [W/(m.K)]."""
    t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
    k_val = OIL_CONDUCTIVITY_AT_20C_W_M_K - OIL_CONDUCTIVITY_SLOPE_W_M_K2 * (
        t_clamped - REF_TEMP_20C
    )
    return float(max(MIN_CONDUCTIVITY_W_M_K, k_val))

  def evaluate(self, temp_c: float) -> OilState:
    """Compute complete thermophysical oil state at temperature temp_c [deg C]."""
    t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
    rho = self.density_kg_m3(t_clamped)
    nu_cst = self.kinematic_viscosity_walther_cst(t_clamped)
    mu_walther = rho * nu_cst * CST_TO_M2_S
    mu_vogel = self.dynamic_viscosity_vogel_pa_s(t_clamped)
    cp = self.specific_heat_j_kg_k(t_clamped)
    k_cond = self.thermal_conductivity_w_m_k(t_clamped)
    prandtl = (mu_walther * cp) / k_cond

    return OilState(
        grade=self.grade,
        temperature_c=t_clamped,
        density_kg_m3=rho,
        kinematic_viscosity_cst=nu_cst,
        dynamic_viscosity_pa_s=float(mu_walther),
        vogel_dynamic_viscosity_pa_s=mu_vogel,
        specific_heat_j_kg_k=cp,
        thermal_conductivity_w_m_k=k_cond,
        prandtl_number=float(prandtl),
    )

  def to_bearing_oil_properties(self, temp_c: float) -> OilProperties:
    """Convert rheological state at temp_c into Bearing OilProperties."""
    from openengine.dynamics.bearing import OilProperties

    state = self.evaluate(temp_c)
    return OilProperties(
        name=f"{self.grade} @ {state.temperature_c:.0f}C",
        dynamic_viscosity_pa_s=state.dynamic_viscosity_pa_s,
        density_kg_m3=state.density_kg_m3,
    )


def fit_walther_coefficients(
    nu_40_cst: float,
    nu_100_cst: float,
) -> Tuple[float, float]:
  """Solve ASTM D341 Walther constants (A, B) from 40C and 100C viscosities."""
  t1_k = REF_TEMP_40C + KELVIN_OFFSET
  t2_k = REF_TEMP_100C + KELVIN_OFFSET

  w1 = math.log10(math.log10(nu_40_cst + WALTHER_SHIFT_CST))
  w2 = math.log10(math.log10(nu_100_cst + WALTHER_SHIFT_CST))

  log_t1 = math.log10(t1_k)
  log_t2 = math.log10(t2_k)

  walther_b = (w1 - w2) / (log_t2 - log_t1)
  walther_a = w1 + walther_b * log_t1
  return float(walther_a), float(walther_b)


def walther_kinematic_viscosity_cst(
    temp_c: float,
    walther_a: float,
    walther_b: float,
) -> float:
  """Evaluate ASTM D341 kinematic viscosity [cSt] at temp_c [deg C]."""
  t_clamped = float(np.clip(temp_c, MIN_VALID_TEMP_C, MAX_VALID_TEMP_C))
  t_k = t_clamped + KELVIN_OFFSET
  w_val = walther_a - walther_b * math.log10(t_k)
  z_val = LOG_BASE_10 ** (LOG_BASE_10**w_val)
  return float(z_val - WALTHER_SHIFT_CST)


def _fit_vogel_from_walther(
    walther_a: float,
    walther_b: float,
    density_15c_kg_m3: float,
) -> Tuple[float, float, float]:
  """Solve exact 3-point Vogel parameters (A_v, B_v, C_v) across 20..130 deg C."""
  t1 = REF_TEMP_20C
  t2 = REF_TEMP_80C
  t3 = REF_TEMP_130C

  def _mu_at(tc: float) -> float:
    """Return Walther dynamic viscosity [Pa.s] at reference temperature tc."""
    rho = density_15c_kg_m3 - THERMAL_EXPANSION_KG_M3_K * (tc - REF_TEMP_15C)
    nu = walther_kinematic_viscosity_cst(tc, walther_a, walther_b)
    return rho * nu * CST_TO_M2_S

  mu1 = _mu_at(t1)
  mu2 = _mu_at(t2)
  mu3 = _mu_at(t3)

  log_12 = math.log(mu1 / mu2)
  log_23 = math.log(mu2 / mu3)
  ratio = (log_12 * (t3 - t2)) / (log_23 * (t2 - t1))

  vogel_c = (t3 - ratio * t1) / (ratio - 1.0)
  vogel_b = (log_12 * (t1 + vogel_c) * (t2 + vogel_c)) / (t2 - t1)
  vogel_a = mu1 * math.exp(-vogel_b / (t1 + vogel_c))
  return float(vogel_a), float(vogel_b), float(vogel_c)
