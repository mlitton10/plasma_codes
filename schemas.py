from dataclasses import dataclass

from astropy.units import Quantity
from plasmapy.particles import Particle


@dataclass
class PlasmaParameters:
    density: Quantity
    magnetic_field: Quantity
    electron_temperature: Quantity
    ion_temperature: Quantity
    ion_species: Particle

    electron_gyrofrequency: Quantity
    ion_gyrofrequency: Quantity
    lower_hybrid_frequency: Quantity
    upper_hybrid_frequency: Quantity
    electron_plasma_frequency: Quantity
    ion_plasma_frequency: Quantity
    electron_electron_collision_frequency: Quantity
    electron_ion_collision_frequency: Quantity
    ion_ion_collision_frequency: Quantity

    debye_length: Quantity
    electron_gyroradius: Quantity
    ion_gyroradius: Quantity
    electron_inertial_length: Quantity
    ion_inertial_length: Quantity
    electron_electron_mfp: Quantity
    electron_ion_mfp: Quantity
    ion_ion_mfp: Quantity

    alfven_speed: Quantity
    sound_speed: Quantity              # plasmapy ion_sound_speed (gamma_e=1, gamma_i=3)
    electron_thermal_velocity: Quantity
    ion_thermal_velocity: Quantity

    spritzer_resistivity: Quantity
    beta: Quantity                     # ELECTRON beta (plasmapy beta(T_e, n, B))
    debye_number: Quantity
    electron_hall_parameter: Quantity
    ion_hall_parameter: Quantity
    bohm_diffusion: Quantity

    # --- single-fluid MHD quantities (plasma_codes.formulary); these are the
    # --- ones the jump conditions use.
    mass_density: Quantity             # same rho as plasmapy Alfven_speed
    thermal_pressure: Quantity         # n (T_e + T_i)
    total_beta: Quantity               # 2 mu0 n (T_e + T_i) / B^2


@dataclass
class JumpConditions:
    """Ideal-MHD Rankine-Hugoniot solution in the normal-incidence frame.

    Every field may be a scalar or an array (broadcast over the inputs).
    Downstream fields are NaN where no shock of ``branch`` exists; check
    ``exists``.  Velocities are in the shock rest frame (NIF).
    """
    branch: str
    gamma: float
    shock_theta: Quantity                  # theta_Bn, upstream
    shock_velocity: Quantity               # upstream normal flow speed in shock frame

    # upstream (always defined)
    alfven_mach_number: Quantity           # u1 / v_A (total B)
    sonic_mach_number: Quantity            # u1 / sqrt(gamma p / rho)
    fast_mach_number: Quantity             # u1 / c_f(theta)
    upstream_total_beta: Quantity

    # ratios
    exists: object                         # bool or bool array
    switch_on: object                      # bool (array): theta = 0 switch-on shock
    shock_type: object                     # str (array): 'fast' | 'switch-on' | 'slow' | 'none'
    compression_ratio: Quantity            # rho2/rho1
    magnetic_field_ratio: Quantity         # |B2|/|B1|
    tangential_field_ratio: Quantity       # B2t/B1t (NaN at theta = 0)
    pressure_ratio: Quantity               # p2/p1 (total thermal)
    temperature_ratio: Quantity            # T2/T1 = (p2/p1)/r, applied to T_e and T_i alike

    # downstream state
    downstream_density: Quantity
    downstream_magnetic_field: Quantity
    downstream_electron_temperature: Quantity
    downstream_ion_temperature: Quantity
    downstream_normal_velocity: Quantity   # u1 / r
    downstream_tangential_velocity: Quantity
    flow_deflection: Quantity
    downstream_theta: Quantity             # theta_Bn, downstream
    downstream_total_beta: Quantity
    downstream_sonic_mach_number: Quantity  # u2n / c_s2 (< 1 means supercritical)
    downstream_fast_mach_number: Quantity
