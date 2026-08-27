from astropy.units import Quantity
from plasma_equilibrium import Plasma
from plasmapy.particles import Particle
from shocks.utils.shock_utilities import compression_ratio, pressure_ratio
import astropy.units as u

class ShockedPlasma:
    def __init__(self, density: Quantity,
                 magnetic_field: Quantity,
                 electron_temperature: Quantity,
                 ion_temperature: Quantity,
                 ion_species: Particle,
                 shock_velocity: Quantity,
                 shock_theta: Quantity,
                 gamma):

        self.upstream_plasma = Plasma(density,
                                      magnetic_field,
                                      electron_temperature,
                                      ion_temperature,
                                      ion_species)

        self.shock_velocity = shock_velocity
        self.theta = shock_theta
        self.gamma = gamma

        downstream_n, downstream_flow, downstream_Te, downstream_Ti, downstream_B = self.MHD_jump_condition()
        print(downstream_n, downstream_flow, downstream_Te, downstream_Ti, downstream_B)
        self.downstream_velocity = downstream_flow
        self.downstream_plasma = Plasma(downstream_n,
                                      downstream_B,
                                      downstream_Te,
                                      downstream_Ti,
                                      ion_species)



    def MHD_jump_condition(self):
        r = compression_ratio(self.shock_velocity.to(u.m/u.s),
                              self.upstream_plasma.plasma_parameters.sound_speed.to(u.m/u.s),
                              self.upstream_plasma.plasma_parameters.alfven_speed.to(u.m/u.s),
                              self.theta.to(u.rad).value,
                              self.gamma)

        R = pressure_ratio(self.shock_velocity.to(u.m/u.s),
                           self.upstream_plasma.plasma_parameters.sound_speed.to(u.m/u.s),
                           self.upstream_plasma.plasma_parameters.alfven_speed.to(u.m/u.s),
                           self.theta.to(u.rad).value,
                           self.gamma)

        if self.theta == 0:
            downstream_density = self.upstream_plasma.plasma_parameters.density * r
            downstream_flow = self.shock_velocity / r
            downstream_T_e = self.upstream_plasma.plasma_parameters.electron_temperature * R / r
            downstream_T_i = self.upstream_plasma.plasma_parameters.ion_temperature * R / r
            downstream_B = self.upstream_plasma.plasma_parameters.magnetic_field
            return downstream_density, downstream_flow, downstream_T_e, downstream_T_i, downstream_B
        else:
            raise NotImplementedError

if __name__=='__main__':
    density = 1e13 * u.cm**(-3)
    magnetic_field = 250 * u.G
    electron_temperature = 5 * u.eV
    ion_temperature = 1 * u.eV
    ion_species = Particle('p+')
    shock_velocity = 150e3 * u.m/u.s
    shock_theta = 0.0 * u.deg
    gamma = 5/3

    shocked_plasma = ShockedPlasma(density,
                                   magnetic_field,
                                   electron_temperature,
                                   ion_temperature,
                                   ion_species,
                                   shock_velocity,
                                   shock_theta,
                                   gamma)

    print(shocked_plasma.downstream_velocity/shocked_plasma.downstream_plasma.plasma_parameters.sound_speed)