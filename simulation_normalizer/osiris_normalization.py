from typing import Dict

import numpy as np
from astropy.constants.codata2022 import c
import astropy.units as u
from astropy.units import Quantity
from plasma_equilibrium import Plasma
from plasmapy.particles import Particle
from plasmapy.particles import CustomParticle
from schemas import PlasmaParameters

def gamma_factor(v):
    return np.sqrt(1 / (1-(v/c) ** 2))

class OsirisNormalization(Plasma):
    def __init__(self,
                 density,
                 magnetic_field,
                 electron_temperature,
                 ion_temperature,
                 ion_species):
        super().__init__(density, magnetic_field, electron_temperature, ion_temperature, ion_species)

        self.time_scale = 1 / self.plasma_parameters.electron_plasma_frequency
        self.length_scale = 1 / self.plasma_parameters.electron_inertial_length
        self.velocity_scale = 1 / c

        self.normalized_plasma_parameters = self.rescale_quantities()

    def rescale_quantities(self):
        frequencies = self.normalize_frequencies()
        lengths = self.normalize_lengths()
        velocities = self.normalize_velocities()
        misc = self.normalize_misc()

        normalized_plasma_parameters = PlasmaParameters(density=1,
                                                        magnetic_field=1e3*self.B.to(u.G).value/(3.204 * np.sqrt(self.n.to(u.cm**-3).value)),
                                                        electron_temperature=self.T_e,
                                                        ion_temperature=self.T_i,
                                                        ion_species=self.ion,
                                                        **frequencies, **lengths, **velocities, **misc)
        return normalized_plasma_parameters


    def normalize_frequencies(self) -> Dict[str, Quantity]:
        electron_gyrofrequency = self.plasma_parameters.electron_gyrofrequency * self.time_scale
        ion_gyrofrequency = self.plasma_parameters.ion_gyrofrequency * self.time_scale

        lower_hybrid = self.plasma_parameters.lower_hybrid_frequency * self.time_scale
        upper_hybrid = self.plasma_parameters.upper_hybrid_frequency * self.time_scale

        electron_plasma_frequency = self.plasma_parameters.electron_plasma_frequency * self.time_scale
        ion_plasma_frequency = self.plasma_parameters.ion_plasma_frequency * self.time_scale

        electron_electron_collision_freq = self.plasma_parameters.electron_electron_collision_frequency * self.time_scale
        electron_ion_collision_freq = self.plasma_parameters.electron_ion_collision_frequency * self.time_scale
        ion_ion_collision_freq = self.plasma_parameters.ion_ion_collision_frequency * self.time_scale

        frequencies = {
            'electron_gyrofrequency': electron_gyrofrequency,
            'ion_gyrofrequency': ion_gyrofrequency,
            'electron_plasma_frequency': electron_plasma_frequency,
            'ion_plasma_frequency': ion_plasma_frequency,
            'electron_electron_collision_frequency': electron_electron_collision_freq,
            'electron_ion_collision_frequency': electron_ion_collision_freq,
            'ion_ion_collision_frequency': ion_ion_collision_freq,
            'lower_hybrid_frequency': lower_hybrid,
            'upper_hybrid_frequency': upper_hybrid,
        }
        return frequencies


    def normalize_lengths(self) -> Dict[str, Quantity]:
        debye_length = self.plasma_parameters.debye_length * self.length_scale

        electron_gyroradius = self.plasma_parameters.electron_gyroradius * self.length_scale
        ion_gyroradius = self.plasma_parameters.ion_gyroradius * self.length_scale

        electron_inertial_length = self.plasma_parameters.electron_inertial_length * self.length_scale
        ion_inertial_length  = self.plasma_parameters.ion_inertial_length * self.length_scale


        electron_electron_mfp = self.plasma_parameters.electron_electron_mfp * self.length_scale
        electron_ion_mfp = self.plasma_parameters.electron_ion_mfp * self.length_scale
        ion_ion_mfp = self.plasma_parameters.ion_ion_mfp * self.length_scale

        lengths = {
            'debye_length': debye_length,
            'electron_gyroradius': electron_gyroradius,
            'ion_gyroradius': ion_gyroradius,
            'electron_inertial_length': electron_inertial_length,
            'ion_inertial_length': ion_inertial_length,
            'electron_electron_mfp': electron_electron_mfp,
            'ion_ion_mfp': ion_ion_mfp,
            'electron_ion_mfp': electron_ion_mfp,
        }
        return lengths

    def normalize_velocities(self) -> Dict[str, Quantity]:
        alfven_speed = self.plasma_parameters.alfven_speed * self.velocity_scale
        sound_speed = self.plasma_parameters.sound_speed * self.velocity_scale
        electron_thermal_velocity = self.plasma_parameters.electron_thermal_velocity * self.velocity_scale
        ion_thermal_velocity = self.plasma_parameters.ion_thermal_velocity * self.velocity_scale

        velocities = {
            'alfven_speed': alfven_speed,
            'sound_speed': sound_speed,
            'electron_thermal_velocity': electron_thermal_velocity,
            'ion_thermal_velocity': ion_thermal_velocity,
        }
        return velocities

    def normalized_proper_velocities(self) -> Dict[str, Quantity]:
        alfven_speed = self.plasma_parameters.alfven_speed * self.velocity_scale * gamma_factor(self.plasma_parameters.alfven_speed)
        sound_speed = self.plasma_parameters.sound_speed * self.velocity_scale * gamma_factor(self.plasma_parameters.sound_speed)
        electron_thermal_velocity = self.plasma_parameters.electron_thermal_velocity * self.velocity_scale * gamma_factor(self.plasma_parameters.electron_thermal_velocity)
        ion_thermal_velocity = self.plasma_parameters.ion_thermal_velocity * self.velocity_scale * gamma_factor(self.plasma_parameters.ion_thermal_velocity)

        velocities = {
            'alfven_speed': alfven_speed,
            'sound_speed': sound_speed,
            'electron_thermal_velocity': electron_thermal_velocity,
            'ion_thermal_velocity': ion_thermal_velocity,
        }
        return velocities

    def normalize_misc(self) -> Dict[str, Quantity]:


        misc = {
            'spritzer_resistivity': np.nan,
            'beta': self.plasma_parameters.beta,
            'debye_number': self.plasma_parameters.debye_number,
            'electron_hall_parameter': np.nan,
            'ion_hall_parameter': np.nan,
            'bohm_diffusion': np.nan,
        }

        return misc

if __name__ == "__main__":
    density = 1e14 * u.cm ** (-3)
    magnetic_field = 5000 * u.G
    electron_temperature = 5 * u.eV
    ion_temperature = 1 * u.eV
    # create fake ion with a mass of 100 times the electron mass
    ion_species = CustomParticle(mass=100 * 9.109e-31 * u.kg, charge=1.602e-19 * u.C, symbol="p+")
    shock_velocity = 300e3 * u.m / u.s

    plasma = OsirisNormalization(density,
                    magnetic_field,
                    electron_temperature,
                    ion_temperature,
                    ion_species)

    print(plasma.plasma_parameters.alfven_speed)

    print("Simulation Magnetic Field: ", plasma.normalized_plasma_parameters.magnetic_field)

    print("Simulation Time: ", 100 * 1 / plasma.normalized_plasma_parameters.ion_gyrofrequency)
    print("Simulation Fluid Velocity: ", plasma.normalized_plasma_parameters.alfven_speed * 5)

    print("Simulation electron thermal velocity: ", plasma.normalized_plasma_parameters.electron_thermal_velocity)
    print("Simulation ion thermal velocity: ", plasma.normalized_plasma_parameters.ion_thermal_velocity)

    print("Simulation shock travel length: ", plasma.normalized_plasma_parameters.alfven_speed * 5 * 10 * 1 / plasma.normalized_plasma_parameters.ion_gyrofrequency)

    print("Simulation development length: ", 10 * plasma.normalized_plasma_parameters.ion_inertial_length)

    print("Ion Inertial length scale:", plasma.normalized_plasma_parameters.ion_inertial_length)




