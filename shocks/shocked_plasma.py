import numpy as np
from astropy.units import Quantity
from matplotlib import pyplot as plt
from plasma_equilibrium import Plasma
from plasmapy.particles import Particle
from shocks.utils.shock_utilities import compression_ratio, pressure_ratio
import astropy.units as u
from astropy.constants import c

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
        print("Alfvenic Mach number = ", self.shock_velocity/self.upstream_plasma.plasma_parameters.alfven_speed)
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

        self.omegas_upstream, self.v_phase_upstream, self.v_group_upstream, self.k_space = self.upstream_plasma.two_fluid_dispersion(self.theta,plot=False)
        self.omegas_downstream, self.v_phase_downstream, self.v_group_downstream, self.k_space = self.downstream_plasma.two_fluid_dispersion(self.theta,plot=False)

        self.k_upstream, self.k_downstream, self.omega_upstream, self.omega_downstream = self.compute_phase_crossing()
        print("upstream wavelength = ", 2*np.pi *u.rad /self.k_upstream)
        print("downstream wavelength = ", 2*np.pi*u.rad/self.k_downstream)

        print("upstream frequency = ", self.omega_upstream / (2*np.pi * u.rad))
        print("downstream frequency = ",self.omega_downstream / (2*np.pi * u.rad))


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

    def compute_phase_crossing(self):

        idx = np.argwhere(np.diff(np.sign(self.v_phase_upstream['fast_mode'].value - self.shock_velocity.value))).flatten()

        k_upstream = self.k_space[idx][0]
        omega_upstream = self.omegas_upstream['fast_mode'][idx][0]

        idx = np.argwhere(np.diff(np.sign(self.v_phase_downstream['alfven_mode'].value - self.downstream_velocity.value))).flatten()
        omega_downstream = self.omegas_downstream['alfven_mode'][idx][0]
        k_downstream = self.k_space[idx][0]
        return k_upstream, k_downstream, omega_upstream, omega_downstream

    def plot_dispersions(self):
        f,a = plt.subplots(1,2)

        for key, value in self.omegas_upstream.items():
            a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                   np.real(value) / self.upstream_plasma.plasma_parameters.electron_plasma_frequency,
                   label=key)

        for key, value in self.omegas_downstream.items():
            a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                   np.real(value) / self.downstream_plasma.plasma_parameters.electron_plasma_frequency,
                   label=key)

        a[0].scatter(self.k_upstream * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                     self.omega_upstream / self.upstream_plasma.plasma_parameters.electron_plasma_frequency)
        a[1].scatter(self.k_downstream * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                     self.omega_downstream / self.downstream_plasma.plasma_parameters.electron_plasma_frequency)

        a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                  self.downstream_velocity*self.k_space / self.downstream_plasma.plasma_parameters.electron_plasma_frequency,)

        a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                  self.shock_velocity * self.k_space / self.upstream_plasma.plasma_parameters.electron_plasma_frequency, )

        a[0].legend()
        a[1].legend()
        a[1].set_xlabel(r"$ck/\omega_{pe}$")
        a[0].set_xlabel(r"$ck/\omega_{pe}$")

        a[0].set_ylabel(r"$\omega/\omega_{pe}$")
        a[1].set_ylabel(r"$\omega/\omega_{pe}$")

        a[0].set_title("Upstream Plasma")
        a[1].set_title("Downstream Plasma")

        a[0].set_yscale("log")
        a[0].set_xscale("log")

        a[1].set_yscale("log")
        a[1].set_xscale("log")

        plt.show()

    def plot_phase_velocities(self):
        f,a = plt.subplots(1,2)

        for key, value in self.v_phase_upstream.items():
            a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                   np.real(value) / c,
                   label=key)

        for key, value in self.v_phase_downstream.items():
            a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                   np.real(value) / c,
                   label=key)

        a[0].scatter(self.k_upstream * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                     self.shock_velocity/c)
        a[1].scatter(self.k_downstream * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                     self.downstream_velocity/c)

        a[1].axhline(self.downstream_velocity/c)

        a[0].axhline(self.shock_velocity/c)

        a[0].legend()
        a[1].legend()
        a[1].set_xlabel(r"$ck/\omega_{pe}$")
        a[0].set_xlabel(r"$ck/\omega_{pe}$")

        a[0].set_ylabel(r"$v_{ph.}/\omega_{pe}$")
        a[1].set_ylabel(r"$v_{ph.}/\omega_{pe}$")

        a[0].set_title("Upstream Plasma")
        a[1].set_title("Downstream Plasma")

        a[0].set_yscale("log")
        a[0].set_xscale("log")

        a[1].set_yscale("log")
        a[1].set_xscale("log")

        plt.show()


if __name__=='__main__':
    density = 1e14* u.cm**(-3)
    magnetic_field = 200 * u.G
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
    shocked_plasma.plot_dispersions()
    shocked_plasma.plot_phase_velocities()
