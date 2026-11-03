"""Single-fluid MHD quantities shared by every module in ``plasma_codes``.

Everything that enters the Rankine-Hugoniot jump conditions is defined here,
once, on top of plasmapy, so ``Plasma``, ``ShockedPlasma``, the shock utilities
and the constraint scans cannot drift apart.

Conventions
-----------
* Quasi-neutral plasma with n_e = n_i = ``density`` (the convention ``Plasma``
  already uses for every plasmapy call).
* Total thermal pressure  p = n (T_e + T_i).
* Mass density rho is the one plasmapy's ``Alfven_speed`` uses,
  rho = (m_i + Z m_e) n, so v_A, c_s and beta are mutually exact:
  c_s^2 / v_A^2 = gamma * beta / 2.
* ``total_beta`` = 2 mu0 p / B^2 (electrons + ions).  Note plasmapy's ``beta``
  (stored as ``PlasmaParameters.beta``) is the beta of ONE temperature.
* ``mhd_sound_speed`` = sqrt(gamma p / rho) with the same gamma as the jump
  conditions.  This is NOT plasmapy's ``ion_sound_speed`` (gamma_e = 1,
  gamma_i = 3), which ``Plasma`` keeps as ``sound_speed``.
* Temperatures may be given in eV or K.
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.constants import mu0
from plasmapy.particles import Particle

__all__ = [
    "temperature_to_energy",
    "mhd_mass_density",
    "thermal_pressure",
    "total_beta",
    "mhd_sound_speed",
    "magnetosonic_speeds",
]


def temperature_to_energy(T: u.Quantity) -> u.Quantity:
    """k_B T in joules, accepting temperatures in eV or K."""
    return u.Quantity(T).to(u.J, equivalencies=u.temperature_energy())


def mhd_mass_density(density: u.Quantity, ion_species: Particle) -> u.Quantity:
    """Mass density used by plasmapy's ``Alfven_speed``: (m_i + Z m_e) n.
    (Checked against ``Alfven_speed`` in tests/test_units_layer.py.)"""
    ion = Particle(ion_species) if isinstance(ion_species, str) else ion_species
    m_eff = ion.mass + ion.charge_number * Particle("e-").mass
    return (m_eff * u.Quantity(density)).to(u.kg / u.m**3)


def thermal_pressure(density: u.Quantity, electron_temperature: u.Quantity,
                     ion_temperature: u.Quantity) -> u.Quantity:
    """p = n (T_e + T_i)."""
    T_sum = temperature_to_energy(electron_temperature) + temperature_to_energy(ion_temperature)
    return (u.Quantity(density) * T_sum).to(u.Pa)


def total_beta(density: u.Quantity, magnetic_field: u.Quantity,
               electron_temperature: u.Quantity, ion_temperature: u.Quantity) -> u.Quantity:
    """beta = 2 mu0 n (T_e + T_i) / B^2."""
    p = thermal_pressure(density, electron_temperature, ion_temperature)
    return (2 * mu0 * p / u.Quantity(magnetic_field) ** 2).to(u.dimensionless_unscaled)


def mhd_sound_speed(density: u.Quantity, electron_temperature: u.Quantity,
                    ion_temperature: u.Quantity, ion_species: Particle,
                    gamma: float = 5 / 3) -> u.Quantity:
    """Single-fluid sound speed sqrt(gamma p / rho), same gamma as the jump."""
    p = thermal_pressure(density, electron_temperature, ion_temperature)
    rho = mhd_mass_density(density, ion_species)
    return np.sqrt(gamma * p / rho).to(u.m / u.s)


def magnetosonic_speeds(alfven_speed: u.Quantity, sound_speed: u.Quantity,
                        theta: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    """(slow, fast) magnetosonic phase speeds at angle ``theta`` to B.

    c_{f,s}^2 = (v_A^2 + c_s^2 +/- sqrt((v_A^2 + c_s^2)^2 - 4 v_A^2 c_s^2 cos^2 theta)) / 2
    ``sound_speed`` should be ``mhd_sound_speed`` for consistency with the jump.
    """
    vA2 = u.Quantity(alfven_speed).to(u.m / u.s) ** 2
    cs2 = u.Quantity(sound_speed).to(u.m / u.s) ** 2
    cos2 = np.cos(u.Quantity(theta).to(u.rad)) ** 2
    s = vA2 + cs2
    disc = np.sqrt(np.maximum(s**2 - 4 * vA2 * cs2 * cos2, 0 * s**2))
    slow = np.sqrt(np.maximum(0.5 * (s - disc), 0 * s))
    fast = np.sqrt(0.5 * (s + disc))
    return slow, fast
