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
    sound_speed: Quantity
    electron_thermal_velocity: Quantity
    ion_thermal_velocity: Quantity

    spritzer_resistivity: Quantity
    beta: Quantity
    debye_number: Quantity
    electron_hall_parameter: Quantity
    ion_hall_parameter: Quantity
    bohm_diffusion: Quantity
