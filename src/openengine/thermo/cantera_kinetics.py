"""Cantera-Based Multi-Species Thermochemistry and Combustion Kinetics."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cantera as ct
import numpy as np


@dataclass
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


@dataclass
class EquilibriumCombustionResult:
  """Chemical equilibrium state after constant-volume combustion."""

  adiabatic_temperature_k: float
  peak_pressure_pa: float
  gamma_burned: float
  cv_burned_j_kg_k: float
  major_species_mole_fractions: Dict[str, float]


class CanteraThermochemistry:
  """Thermochemical state evaluator and equilibrium solver using Cantera."""

  def __init__(
      self,
      mechanism: str = "gri30.yaml",
      fuel_species: str = "CH4",
      oxidizer_species: str = "O2:1.0, N2:3.76",
      stoichiometric_fuel_to_air_mass: float = 0.058,  # ~17.2 AFR for CH4 (14.7 for C8H18)
  ):
    self.mech_name = mechanism
    self.gas = ct.Solution(mechanism)
    self.fuel = fuel_species
    self.oxidizer = oxidizer_species
    self.stoich_fa = stoichiometric_fuel_to_air_mass

  def set_unburned_mixture(
      self,
      t_k: float,
      p_pa: float,
      equivalence_ratio: float = 1.0,
  ) -> None:
    """Set the thermodynamic state and stoichiometric unburned mixture."""
    self.gas.TP = max(200.0, t_k), max(1000.0, p_pa)
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
    self.gas.TP = max(200.0, t_k), max(1000.0, p_pa)
    cp = float(self.gas.cp_mass)
    cv = float(self.gas.cv_mass)
    gamma = cp / cv
    r_gas = float(ct.gas_constant / self.gas.mean_molecular_weight)
    return cp, cv, gamma, r_gas

  def constant_volume_equilibrium(
      self,
      t_unburned_k: float,
      p_unburned_pa: float,
      equivalence_ratio: float = 1.0,
  ) -> EquilibriumCombustionResult:
    """Solve constant-volume, constant-internal-energy combustion equilibrium (UV).

    Simulates the theoretical limit of closed-volume flame propagation
    with complete chemical dissociation (formation of CO, H2, NO, OH).
    """
    self.set_unburned_mixture(t_unburned_k, p_unburned_pa, equivalence_ratio)
    # Equilibrate at constant internal energy and specific volume
    self.gas.equilibrate("UV")

    tracked_species = ["CO2", "H2O", "CO", "H2", "O2", "N2", "NO", "OH"]
    species_fractions = {}
    for sp in tracked_species:
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
