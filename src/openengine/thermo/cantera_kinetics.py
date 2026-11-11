"""Cantera-Based Multi-Species Thermochemistry and Gasoline E10 Kinetics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, TypedDict
import cantera as ct
import numpy as np

DEFAULT_MECHANISM_FILENAME: str = "gasoline_prf_ethanol.yaml"
DEFAULT_OXIDIZER_COMPOSITION: str = "O2:1.0, N2:3.76"

US_87_AKI: float = 87.0
US_87_RON: float = 91.5
US_87_MON: float = 82.5
E10_ETHANOL_PCT: float = 10.0
E10_LHV_J_KG: float = 42.0e6
E10_STOICH_AFR: float = 14.1

PURE_ISOOCTANE_AKI: float = 100.0
PURE_ISOOCTANE_LHV_J_KG: float = 44.3e6
PURE_ISOOCTANE_STOICH_AFR: float = 15.13

ISOOCTANE_SPECIES: str = "C8H18"
NHEPTANE_SPECIES: str = "n-C7H16"
ETHANOL_SPECIES: str = "C2H5OH"

ISOOCTANE_DENSITY_KG_M3: float = 691.9
NHEPTANE_DENSITY_KG_M3: float = 683.8
ETHANOL_DENSITY_KG_M3: float = 789.3

ISOOCTANE_MW_G_MOL: float = 114.2285
NHEPTANE_MW_G_MOL: float = 100.2019
ETHANOL_MW_G_MOL: float = 46.0684

ETHANOL_BLENDING_AKI: float = 113.0
HYDROCARBON_LATENT_HEAT_J_KG: float = 350.0e3
ETHANOL_LATENT_HEAT_J_KG: float = 840.0e3

PERCENT_SCALE: float = 100.0
OCTANE_SCALE: float = 100.0
MIN_FRACTION: float = 0.0
DIATOMIC_STOICH_FACTOR: float = 0.5
MAX_FRACTION: float = 1.0
EPSILON_FRACTION: float = 1.0e-12

MIN_GAS_TEMP_K: float = 200.0
MIN_GAS_PRESSURE_PA: float = 1000.0
STD_REF_TEMP_K: float = 298.15
STD_ATM_PA: float = 101325.0

TRACKED_EQUILIBRIUM_SPECIES: Tuple[str, ...] = (
    "CO2",
    "H2O",
    "CO",
    "H2",
    "O2",
    "N2",
    "NO",
    "OH",
)


class FuelConfigSection(TypedDict, total=False):
  """Typed YAML schema for the fuel configuration block."""

  name: str
  aki: float
  ron: float
  mon: float
  ethanol_pct: float
  lhv_j_kg: float
  stoich_afr: float


class EngineFuelConfigRoot(TypedDict, total=False):
  """Root YAML mapping slice containing the fuel block."""

  fuel: FuelConfigSection


@dataclass(frozen=True, slots=True)
class FuelBlendSpec:
  """Primary Reference Fuel plus Ethanol (PRF+E) gasoline blend specification."""

  name: str = "US 87-Octane E10 Pump Gasoline"
  aki: float = US_87_AKI
  ron: float = US_87_RON
  mon: float = US_87_MON
  ethanol_pct: float = E10_ETHANOL_PCT
  lhv_j_kg: float = E10_LHV_J_KG
  stoich_afr: float = E10_STOICH_AFR

  @classmethod
  def us_87_e10(cls) -> "FuelBlendSpec":
    """Return standard US 87-AKI regular E10 pump gasoline specification."""
    return cls()

  @classmethod
  def pure_isooctane(cls) -> "FuelBlendSpec":
    """Return neat 100-AKI iso-octane reference fuel specification."""
    return cls(
        name="Pure Iso-Octane (PRF 100)",
        aki=PURE_ISOOCTANE_AKI,
        ron=PURE_ISOOCTANE_AKI,
        mon=PURE_ISOOCTANE_AKI,
        ethanol_pct=MIN_FRACTION,
        lhv_j_kg=PURE_ISOOCTANE_LHV_J_KG,
        stoich_afr=PURE_ISOOCTANE_STOICH_AFR,
    )

  @classmethod
  def from_config(cls, cfg: EngineFuelConfigRoot) -> "FuelBlendSpec":
    """Build a FuelBlendSpec from a parsed engine_specs.yaml mapping."""
    fuel_block = cfg.get("fuel")
    if not isinstance(fuel_block, Mapping):
      return cls.us_87_e10()

    aki_val = float(fuel_block.get("aki", US_87_AKI))
    return cls(
        name=str(fuel_block.get("name", "US 87-Octane E10 Pump Gasoline")),
        aki=aki_val,
        ron=float(fuel_block.get("ron", aki_val)),
        mon=float(fuel_block.get("mon", aki_val)),
        ethanol_pct=float(fuel_block.get("ethanol_pct", E10_ETHANOL_PCT)),
        lhv_j_kg=float(fuel_block.get("lhv_j_kg", E10_LHV_J_KG)),
        stoich_afr=float(fuel_block.get("stoich_afr", E10_STOICH_AFR)),
    )

  def liquid_volume_fractions(self) -> Dict[str, float]:
    """Compute liquid volume fractions of C8H18, n-C7H16, and C2H5OH."""
    v_eth = float(np.clip(self.ethanol_pct / PERCENT_SCALE, MIN_FRACTION, MAX_FRACTION))
    v_prf = MAX_FRACTION - v_eth
    if v_prf <= EPSILON_FRACTION:
      return {
          ISOOCTANE_SPECIES: MIN_FRACTION,
          NHEPTANE_SPECIES: MIN_FRACTION,
          ETHANOL_SPECIES: MAX_FRACTION,
      }

    base_aki = (self.aki - v_eth * ETHANOL_BLENDING_AKI) / v_prf
    iso_in_prf = float(np.clip(base_aki / OCTANE_SCALE, MIN_FRACTION, MAX_FRACTION))
    v_iso = v_prf * iso_in_prf
    v_hep = v_prf * (MAX_FRACTION - iso_in_prf)
    return {
        ISOOCTANE_SPECIES: float(v_iso),
        NHEPTANE_SPECIES: float(v_hep),
        ETHANOL_SPECIES: float(v_eth),
    }

  def mass_fractions(self) -> Dict[str, float]:
    """Compute fuel component mass fractions from liquid volume fractions."""
    vols = self.liquid_volume_fractions()
    m_iso = vols[ISOOCTANE_SPECIES] * ISOOCTANE_DENSITY_KG_M3
    m_hep = vols[NHEPTANE_SPECIES] * NHEPTANE_DENSITY_KG_M3
    m_eth = vols[ETHANOL_SPECIES] * ETHANOL_DENSITY_KG_M3
    m_tot = max(EPSILON_FRACTION, m_iso + m_hep + m_eth)
    return {
        ISOOCTANE_SPECIES: float(m_iso / m_tot),
        NHEPTANE_SPECIES: float(m_hep / m_tot),
        ETHANOL_SPECIES: float(m_eth / m_tot),
    }

  def mole_fractions(self) -> Dict[str, float]:
    """Compute vapor mole fractions for Cantera mixture specification."""
    masses = self.mass_fractions()
    n_iso = masses[ISOOCTANE_SPECIES] / ISOOCTANE_MW_G_MOL
    n_hep = masses[NHEPTANE_SPECIES] / NHEPTANE_MW_G_MOL
    n_eth = masses[ETHANOL_SPECIES] / ETHANOL_MW_G_MOL
    n_tot = max(EPSILON_FRACTION, n_iso + n_hep + n_eth)
    raw = {
        ISOOCTANE_SPECIES: float(n_iso / n_tot),
        NHEPTANE_SPECIES: float(n_hep / n_tot),
        ETHANOL_SPECIES: float(n_eth / n_tot),
    }
    return {sp: frac for sp, frac in raw.items() if frac > EPSILON_FRACTION}

  @property
  def liquid_density_kg_m3(self) -> float:
    """Return blended liquid fuel density at 15.56 deg C [kg/m^3]."""
    vols = self.liquid_volume_fractions()
    return float(
        vols[ISOOCTANE_SPECIES] * ISOOCTANE_DENSITY_KG_M3
        + vols[NHEPTANE_SPECIES] * NHEPTANE_DENSITY_KG_M3
        + vols[ETHANOL_SPECIES] * ETHANOL_DENSITY_KG_M3
    )

  @property
  def latent_heat_j_kg(self) -> float:
    """Return mass-weighted fuel latent heat of vaporization [J/kg]."""
    masses = self.mass_fractions()
    hc_frac = masses[ISOOCTANE_SPECIES] + masses[NHEPTANE_SPECIES]
    return float(
        hc_frac * HYDROCARBON_LATENT_HEAT_J_KG
        + masses[ETHANOL_SPECIES] * ETHANOL_LATENT_HEAT_J_KG
    )

  @property
  def mean_fuel_molecular_weight(self) -> float:
    """Return blended fuel vapor molecular weight [g/mol]."""
    moles = self.mole_fractions()
    return float(
        moles.get(ISOOCTANE_SPECIES, MIN_FRACTION) * ISOOCTANE_MW_G_MOL
        + moles.get(NHEPTANE_SPECIES, MIN_FRACTION) * NHEPTANE_MW_G_MOL
        + moles.get(ETHANOL_SPECIES, MIN_FRACTION) * ETHANOL_MW_G_MOL
    )


@dataclass(slots=True)
class ThermodynamicState:
  """Gas mixture state including chemical properties and variable heat capacities."""

  temperature_k: float
  pressure_pa: float
  density_kg_m3: float
  cp_j_kg_k: float  # Specific heat at constant pressure
  cv_j_kg_k: float  # Specific heat at constant volume
  gamma: float  # Ratio of specific heats (cp / cv)
  r_gas_j_kg_k: float  # Specific gas constant (R_universal / M_w)
  enthalpy_j_kg: float
  internal_energy_j_kg: float
  mean_molecular_weight: float


@dataclass(slots=True)
class EquilibriumCombustionResult:
  """Chemical equilibrium state after constant-volume combustion."""

  adiabatic_temperature_k: float
  peak_pressure_pa: float
  gamma_burned: float
  cv_burned_j_kg_k: float
  major_species_mole_fractions: Dict[str, float]


def _resolve_default_mechanism_path() -> Optional[Path]:
  """Locate bundled config/gasoline_prf_ethanol.yaml relative to repository root."""
  repo_candidate = (
      Path(__file__).resolve().parent.parent.parent.parent
      / "config"
      / DEFAULT_MECHANISM_FILENAME
  )
  if repo_candidate.exists():
    return repo_candidate
  cwd_candidate = Path("config") / DEFAULT_MECHANISM_FILENAME
  if cwd_candidate.exists():
    return cwd_candidate
  return None


def _build_in_memory_prf_ethanol_solution() -> ct.Solution:
  """Construct the PRF+ethanol Cantera solution from bundled NASA Glenn data."""
  nasa = {s.name: s for s in ct.Species.list_from_file("nasa_gas.yaml")}
  gri = ct.Solution("gri30.yaml")
  gri_sp = {s.name: s for s in gri.species()}

  iso = ct.Species(ISOOCTANE_SPECIES, nasa["C8H18,isooctane"].composition)
  iso.thermo = nasa["C8H18,isooctane"].thermo
  hep = ct.Species(NHEPTANE_SPECIES, nasa["C7H16,n-heptane"].composition)
  hep.thermo = nasa["C7H16,n-heptane"].thermo
  eth = ct.Species(ETHANOL_SPECIES, nasa["C2H5OH"].composition)
  eth.thermo = nasa["C2H5OH"].thermo

  core_names: List[str] = [
      "O2",
      "N2",
      "AR",
      "CO2",
      "H2O",
      "CO",
      "H2",
      "OH",
      "O",
      "H",
      "HO2",
      "H2O2",
      "NO",
      "NO2",
      "N",
  ]
  species = [iso, hep, eth] + [gri_sp[name] for name in core_names]
  return ct.Solution(thermo="ideal-gas", species=species)


class CanteraThermochemistry:
  """Thermochemical state evaluator and equilibrium solver using Cantera."""

  def __init__(
      self,
      mechanism: Optional[str] = None,
      fuel_species: Optional[str | Mapping[str, float]] = None,
      oxidizer_species: str = DEFAULT_OXIDIZER_COMPOSITION,
      stoichiometric_fuel_to_air_mass: Optional[float] = None,
      fuel_blend: Optional[FuelBlendSpec] = None,
  ) -> None:
    self.fuel_blend: FuelBlendSpec = fuel_blend or FuelBlendSpec.us_87_e10()

    if mechanism is None:
      mech_path = _resolve_default_mechanism_path()
      if mech_path is not None:
        self.mech_name = str(mech_path)
        self.gas = ct.Solution(str(mech_path))
      else:
        self.mech_name = DEFAULT_MECHANISM_FILENAME
        self.gas = _build_in_memory_prf_ethanol_solution()
    else:
      self.mech_name = mechanism
      self.gas = ct.Solution(mechanism)

    if fuel_species is None:
      self.fuel: str | Dict[str, float] = self.fuel_blend.mole_fractions()
    elif isinstance(fuel_species, Mapping):
      self.fuel = {str(k): float(v) for k, v in fuel_species.items()}
    else:
      self.fuel = fuel_species

    self.oxidizer: str = oxidizer_species
    if stoichiometric_fuel_to_air_mass is None:
      self.stoich_fa: float = MAX_FRACTION / self.fuel_blend.stoich_afr
    else:
      self.stoich_fa = stoichiometric_fuel_to_air_mass

  def set_unburned_mixture(
      self,
      t_k: float,
      p_pa: float,
      equivalence_ratio: float = MAX_FRACTION,
  ) -> None:
    """Set the thermodynamic state and stoichiometric unburned mixture."""
    self.gas.TP = max(MIN_GAS_TEMP_K, t_k), max(MIN_GAS_PRESSURE_PA, p_pa)
    # Set equivalence ratio phi: phi = 1.0 is stoichiometric
    self.gas.set_equivalence_ratio(
        phi=equivalence_ratio,
        fuel=self.fuel,
        oxidizer=self.oxidizer,
    )

  def current_state(self) -> ThermodynamicState:
    """Extract thermodynamic state variables from the active Cantera solution."""
    cp = float(self.gas.cp_mass)
    cv = float(self.gas.cv_mass)
    gamma = cp / cv
    r_gas = float(ct.gas_constant / self.gas.mean_molecular_weight)

    return ThermodynamicState(
        temperature_k=float(self.gas.T),
        pressure_pa=float(self.gas.P),
        density_kg_m3=float(self.gas.density),
        cp_j_kg_k=cp,
        cv_j_kg_k=cv,
        gamma=gamma,
        r_gas_j_kg_k=r_gas,
        enthalpy_j_kg=float(self.gas.enthalpy_mass),
        internal_energy_j_kg=float(self.gas.int_energy_mass),
        mean_molecular_weight=float(self.gas.mean_molecular_weight),
    )

  def get_properties_at(
      self,
      t_k: float,
      p_pa: float,
  ) -> Tuple[float, float, float, float]:
    """Fast evaluation of (cp, cv, gamma, r_gas) at specified (T, P)."""
    self.gas.TP = max(MIN_GAS_TEMP_K, t_k), max(MIN_GAS_PRESSURE_PA, p_pa)
    cp = float(self.gas.cp_mass)
    cv = float(self.gas.cv_mass)
    gamma = cp / cv
    r_gas = float(ct.gas_constant / self.gas.mean_molecular_weight)
    return cp, cv, gamma, r_gas

  def stoichiometric_air_fuel_ratio(self) -> float:
    """Calculate exact chemical stoichiometric air-to-fuel mass ratio in Cantera."""
    self.set_unburned_mixture(STD_REF_TEMP_K, STD_ATM_PA, equivalence_ratio=MAX_FRACTION)
    oxidizer_names = ("O2", "N2", "AR")
    y_air = sum(
        float(self.gas[sp].Y[0])
        for sp in oxidizer_names
        if sp in self.gas.species_names
    )
    y_fuel = max(EPSILON_FRACTION, MAX_FRACTION - y_air)
    return float(y_air / y_fuel)

  def lower_heating_value_j_kg(
      self,
      t_ref_k: float = STD_REF_TEMP_K,
      p_ref_pa: float = STD_ATM_PA,
  ) -> float:
    """Calculate vapor-phase lower heating value [J/kg_fuel] from NASA enthalpies."""
    self.set_unburned_mixture(t_ref_k, p_ref_pa, equivalence_ratio=MAX_FRACTION)
    oxidizer_names = ("O2", "N2", "AR")
    y_air = sum(
        float(self.gas[sp].Y[0])
        for sp in oxidizer_names
        if sp in self.gas.species_names
    )
    y_fuel = max(EPSILON_FRACTION, MAX_FRACTION - y_air)
    h_reactants = float(self.gas.enthalpy_mass)

    # Complete stoichiometric oxidation to CO2, H2O(g), and N2 at t_ref_k, p_ref_pa
    elem_c = float(self.gas.elemental_mole_fraction("C"))
    elem_h = float(self.gas.elemental_mole_fraction("H"))
    elem_n = float(self.gas.elemental_mole_fraction("N"))
    products_x = {
        "CO2": elem_c,
        "H2O": DIATOMIC_STOICH_FACTOR * elem_h,
        "N2": DIATOMIC_STOICH_FACTOR * elem_n,
    }
    self.gas.TPX = t_ref_k, p_ref_pa, products_x
    h_products = float(self.gas.enthalpy_mass)
    return float((h_reactants - h_products) / y_fuel)

  def constant_volume_equilibrium(
      self,
      t_unburned_k: float,
      p_unburned_pa: float,
      equivalence_ratio: float = MAX_FRACTION,
  ) -> EquilibriumCombustionResult:
    """Solve constant-volume, constant-internal-energy combustion equilibrium (UV).

    Simulates the theoretical limit of closed-volume flame propagation
    with complete chemical dissociation (formation of CO, H2, NO, OH).
    """
    self.set_unburned_mixture(t_unburned_k, p_unburned_pa, equivalence_ratio)
    # Equilibrate at constant internal energy and specific volume
    self.gas.equilibrate("UV")

    species_fractions: Dict[str, float] = {}
    for sp in TRACKED_EQUILIBRIUM_SPECIES:
      if sp in self.gas.species_names:
        species_fractions[sp] = float(self.gas[sp].X[0])

    cp = float(self.gas.cp_mass)
    cv = float(self.gas.cv_mass)

    return EquilibriumCombustionResult(
        adiabatic_temperature_k=float(self.gas.T),
        peak_pressure_pa=float(self.gas.P),
        gamma_burned=float(cp / cv),
        cv_burned_j_kg_k=cv,
        major_species_mole_fractions=species_fractions,
    )
