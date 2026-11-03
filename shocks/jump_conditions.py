"""Units-aware front end to the ideal-MHD Rankine-Hugoniot solver.

Takes the same physical inputs as ``Plasma`` / ``ShockedPlasma`` (astropy
Quantities, plasmapy Particle) and returns a ``JumpConditions`` dataclass of
Quantities.  All the physics lives in ``mhd_shock.solve_shock_array``; this
module only converts units and builds the dimensionless inputs consistently:

    M_A  = u1 / v_A           v_A from plasmapy ``Alfven_speed`` (total |B|)
    beta = 2 mu0 n (T_e + T_i) / B^2      (``formulary.total_beta``)

Inputs may be scalars or arrays; they are broadcast together (except
``ion_species``, which must be a single particle).

Temperature partition
---------------------
Single-fluid MHD only gives the total pressure jump.  Downstream T_e and T_i
are both scaled by T2/T1 = (p2/p1)/r, i.e. the upstream T_e/T_i ratio is
preserved (same convention as the original ``ShockedPlasma``).
"""
from __future__ import annotations

import astropy.units as u
import numpy as np
from plasmapy.formulary import Alfven_speed
from plasmapy.particles import Particle
from plasmapy.utils.decorators import validate_quantities

from plasma_codes.formulary import total_beta
from plasma_codes.schemas import JumpConditions
from plasma_codes.shocks.mhd_shock import solve_shock_array

__all__ = ["mhd_jump_conditions", "jump_conditions_from_plasma"]

_T = {"can_be_negative": False, "equivalencies": u.temperature_energy()}


def _shock_type(s, branch):
    """'switch-on', the branch name, or 'none' where no shock exists."""
    t = np.where(~np.isfinite(s["r"]), "none",
                 np.where(s["switch_on"], "switch-on", branch))
    return str(t) if t.ndim == 0 else t.astype(object)


@validate_quantities(electron_temperature=_T, ion_temperature=_T)
def mhd_jump_conditions(density: u.m**-3,
                        magnetic_field: u.T,
                        electron_temperature: u.K,
                        ion_temperature: u.K,
                        ion_species: Particle,
                        shock_velocity: u.m / u.s,
                        shock_theta: u.rad,
                        gamma: float = 5 / 3,
                        branch: str = "fast") -> JumpConditions:
    """Ideal-MHD jump conditions for a shock moving at ``shock_velocity``
    (normal to itself, relative to the upstream plasma) with the upstream field
    at ``shock_theta`` (= theta_Bn) to the shock normal.

    Returns a ``JumpConditions``; downstream fields are NaN where no shock of
    ``branch`` ('fast' or 'slow') exists at these parameters.
    """
    v_A = Alfven_speed(magnetic_field, density, ion_species)
    beta = total_beta(density, magnetic_field, electron_temperature, ion_temperature)
    MA = (shock_velocity / v_A).to_value(u.dimensionless_unscaled)
    th_deg = shock_theta.to_value(u.deg)

    s = solve_shock_array(MA, beta.value, th_deg, gamma, branch)
    one = u.dimensionless_unscaled
    T_ratio = s["T_ratio"]

    def T_out(T):
        return (T * T_ratio).to(u.eV, equivalencies=u.temperature_energy())

    return JumpConditions(
        branch=branch,
        gamma=gamma,
        shock_theta=shock_theta.to(u.deg),
        shock_velocity=shock_velocity,
        alfven_mach_number=s["MA"] * one,
        sonic_mach_number=s["Ms1"] * one,
        fast_mach_number=s["Mf1"] * one,
        upstream_total_beta=s["beta"] * one,
        exists=np.isfinite(s["r"]),
        switch_on=s["switch_on"] if s["switch_on"].ndim else bool(s["switch_on"]),
        shock_type=_shock_type(s, branch),
        compression_ratio=s["r"] * one,
        magnetic_field_ratio=s["B_ratio"] * one,
        tangential_field_ratio=s["Bt_ratio"] * one,
        pressure_ratio=s["p_ratio"] * one,
        temperature_ratio=T_ratio * one,
        downstream_density=density * s["r"],
        downstream_magnetic_field=magnetic_field * s["B_ratio"],
        downstream_electron_temperature=T_out(electron_temperature),
        downstream_ion_temperature=T_out(ion_temperature),
        downstream_normal_velocity=shock_velocity * s["u2n_over_u1"],
        downstream_tangential_velocity=shock_velocity * s["u2t_over_u1"],
        flow_deflection=s["deflection_deg"] * u.deg,
        downstream_theta=s["theta2_deg"] * u.deg,
        downstream_total_beta=s["beta2"] * one,
        downstream_sonic_mach_number=s["Ms2"] * one,
        downstream_fast_mach_number=s["Mf2"] * one,
    )


def jump_conditions_from_plasma(plasma, shock_velocity, shock_theta,
                                gamma: float = 5 / 3, branch: str = "fast") -> JumpConditions:
    """Same as ``mhd_jump_conditions`` but takes an upstream ``Plasma``
    (or its ``PlasmaParameters``)."""
    pp = getattr(plasma, "plasma_parameters", plasma)
    return mhd_jump_conditions(pp.density, pp.magnetic_field, pp.electron_temperature,
                               pp.ion_temperature, pp.ion_species, shock_velocity,
                               shock_theta, gamma=gamma, branch=branch)
