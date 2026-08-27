from astropy.units import Quantity
from plasmapy.formulary import Debye_length, gyroradius, inertial_length, mean_free_path, collision_frequency, \
    Spitzer_resistivity, beta, Debye_number, Hall_parameter, gyrofrequency, lower_hybrid_frequency, \
    upper_hybrid_frequency, plasma_frequency, Bohm_diffusion, Alfven_speed, ion_sound_speed, thermal_speed
from plasmapy.particles import Particle
from schemas import PlasmaParameters


class Plasma:
    def __init__(self, density: Quantity,
                 magnetic_field: Quantity,
                 electron_temperature: Quantity,
                 ion_temperature: Quantity,
                 ion_species: Particle) -> None:

        self.n = density
        self.B = magnetic_field
        self.T_e = electron_temperature
        self.T_i = ion_temperature
        self.ion = ion_species
        self.electron = Particle('e-')

        frequencies = self.frequencies()
        lengths = self.lengths()
        velocities = self.velocities()
        misc = self.misc()
        self.plasma_parameters = PlasmaParameters(density=self.n,
                                                  magnetic_field=self.B,
                                                  electron_temperature=self.T_e,
                                                  ion_temperature=self.T_i,
                                                  ion_species=self.ion,
                                                  **frequencies, **lengths, **velocities, **misc)



    def frequencies(self):
        electron_gyrofrequency = gyrofrequency(self.B, self.electron)
        ion_gyrofrequency = gyrofrequency(self.B, self.ion)

        lower_hybrid = lower_hybrid_frequency(self.B, self.n, self.ion)
        upper_hybrid = upper_hybrid_frequency(self.B, self.n)

        electron_plasma_frequency = plasma_frequency(self.n, self.electron)
        ion_plasma_frequency = plasma_frequency(self.n, self.ion)

        electron_electron_collision_freq = collision_frequency(self.T_e,
                                                               self.n,
                                                                (self.electron, self.electron))
        electron_ion_collision_freq = collision_frequency(self.T_e,
                                                          self.n,
                                                          (self.electron, self.ion))
        ion_ion_collision_freq = collision_frequency(self.T_i,
                                                     self.n,
                                                     (self.ion, self.ion))

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

    def lengths(self):
        debye_length = Debye_length(self.T_e, self.n)
        electron_gyroradius = gyroradius(self.B, particle=self.electron, T=self.T_e)
        ion_gyroradius = gyroradius(self.B, particle=self.ion, T=self.T_e)
        electron_inertial_length = inertial_length(self.n, self.electron)
        ion_inertial_length  = inertial_length(self.n, self.ion)

        electron_electron_mfp = mean_free_path(self.T_e, self.n, (self.electron, self.electron))
        electron_ion_mfp = mean_free_path(self.T_e, self.n, (self.electron, self.ion))
        ion_ion_mfp = mean_free_path(self.T_i, self.n, (self.ion, self.ion))

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

    def velocities(self):
        alfven_speed = Alfven_speed(self.B, self.n, self.ion)
        sound_speed = ion_sound_speed(self.T_e, self.T_i, self.ion)
        electron_thermal_velocity = thermal_speed(self.T_e, self.electron, method='nrl', ndim=3)
        ion_thermal_velocity = thermal_speed(self.T_i, self.ion, method='nrl', ndim=3)

        velocities = {
            'alfven_speed': alfven_speed,
            'sound_speed': sound_speed,
            'electron_thermal_velocity': electron_thermal_velocity,
            'ion_thermal_velocity': ion_thermal_velocity,
        }
        return velocities

    def misc(self):
        spritzer_resistivity = Spitzer_resistivity(self.T_e, self.n, species=(self.electron, self.electron))

        beta_val = beta(self.T_e, self.n, self.B)
        debye_number = Debye_number(self.T_e, self.n)
        electron_hall_parameter = Hall_parameter(self.n, self.T_e, self.B, self.ion, self.electron)
        ion_hall_parameter = Hall_parameter(self.n, self.T_i, self.B, self.ion, self.ion)

        bohm_diffusion = Bohm_diffusion(self.T_e, self.B)

        misc = {
            'spritzer_resistivity': spritzer_resistivity,
            'beta': beta_val,
            'debye_number': debye_number,
            'electron_hall_parameter': electron_hall_parameter,
            'ion_hall_parameter': ion_hall_parameter,
            'bohm_diffusion': bohm_diffusion,
        }

        return misc



