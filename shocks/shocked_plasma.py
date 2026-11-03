import numpy as np
from astropy.units import Quantity
from matplotlib import pyplot as plt
from plasma_codes.plasma_equilibrium import Plasma
from plasmapy.particles import Particle, CustomParticle
from plasma_codes.formulary import mhd_sound_speed, magnetosonic_speeds
from plasma_codes.shocks.jump_conditions import mhd_jump_conditions
import astropy.units as u
from astropy.constants import c


def _style(styles=('science', 'ieee', 'vibrant')):

    try:
        import scienceplots  # noqa: F401
        return plt.style.context(list(styles))
    except ImportError:
        return plt.style.context('default')


class ShockedPlasma:
    """Upstream plasma + ideal-MHD shock -> downstream plasma, at any theta_Bn.

    All MHD speeds and Mach numbers use the single-fluid definitions in
    ``plasma_codes.formulary`` with the SAME gamma as the jump conditions:
    c_s = sqrt(gamma n (T_e + T_i) / rho), total beta, plasmapy v_A.
    ``ion_acoustic_mach_number`` keeps the old definition (plasmapy
    ``ion_sound_speed``) for reference.
    """

    def __init__(self, density: Quantity,
                 magnetic_field: Quantity,
                 electron_temperature: Quantity,
                 ion_temperature: Quantity,
                 ion_species: Particle,
                 shock_velocity: Quantity,
                 shock_theta: Quantity,
                 gamma,
                 branch: str = "fast",
                 compute_dispersion: bool = False,
                 verbose: bool = False):

        self.upstream_plasma = Plasma(density,
                                      magnetic_field,
                                      electron_temperature,
                                      ion_temperature,
                                      ion_species)
        up = self.upstream_plasma.plasma_parameters

        self.shock_velocity = shock_velocity
        self.theta = shock_theta
        self.gamma = gamma
        self.branch = branch

        self.upstream_mhd_sound_speed = mhd_sound_speed(density, electron_temperature,
                                                        ion_temperature, ion_species, gamma)
        self.upstream_slow_magnetosonic, self.upstream_fast_magnetosonic = magnetosonic_speeds(
            up.alfven_speed, self.upstream_mhd_sound_speed, self.theta)

        v = self.shock_velocity.to(u.m / u.s)
        self.alfven_mach_number = v / up.alfven_speed.to(u.m / u.s)
        self.fast_mach_number = v / self.upstream_fast_magnetosonic
        with np.errstate(divide="ignore"):          # c_slow = 0 at theta = 90
            self.slow_mach_number = v / self.upstream_slow_magnetosonic
        self.sonic_mach_number = v / self.upstream_mhd_sound_speed
        self.ion_acoustic_mach_number = v / up.sound_speed.to(u.m / u.s)

        downstream_n, downstream_flow, downstream_Te, downstream_Ti, downstream_B = self.MHD_jump_condition()
        if not self.jump_conditions.exists:
            raise ValueError(
                f"No {branch} MHD shock exists for M_A = {self.alfven_mach_number:.3g}, "
                f"M_f = {self.fast_mach_number:.3g}, theta = {self.theta}, "
                f"beta = {up.total_beta:.3g}.")
        self.downstream_velocity = downstream_flow
        self.downstream_plasma = Plasma(downstream_n,
                                        downstream_B,
                                        downstream_Te,
                                        downstream_Ti,
                                        ion_species)

        if verbose:
            print("Alfvenic Mach number = ", self.alfven_mach_number)
            print("fast Mach number = ", self.fast_mach_number)
            print(downstream_n, downstream_flow, downstream_Te, downstream_Ti, downstream_B)

        if compute_dispersion:
            self.omegas_upstream, self.v_phase_upstream, self.v_group_upstream, self.k_space = \
                self.upstream_plasma.two_fluid_dispersion(self.theta, plot=False)
            self.omegas_downstream, self.v_phase_downstream, self.v_group_downstream, self.k_space = \
                self.downstream_plasma.two_fluid_dispersion(self.theta, plot=False)
            self.k_upstream, self.k_downstream, self.omega_upstream, self.omega_downstream = \
                self.compute_phase_crossing()
            if verbose:
                print("upstream wavelength = ", 2 * np.pi * u.rad / self.k_upstream)
                print("downstream wavelength = ", 2 * np.pi * u.rad / self.k_downstream)
                print("upstream frequency = ", self.omega_upstream / (2 * np.pi * u.rad))
                print("downstream frequency = ", self.omega_downstream / (2 * np.pi * u.rad))

    def MHD_jump_condition(self):
        """Ideal-MHD Rankine-Hugoniot jump at any theta_Bn.

        Returns (n2, u2n, T_e2, T_i2, |B2|) as before; the full solution
        (tangential flow, downstream angle, beta, criticality, ...) is stored
        on ``self.jump_conditions``.
        """
        up = self.upstream_plasma.plasma_parameters
        self.jump_conditions = mhd_jump_conditions(
            up.density, up.magnetic_field, up.electron_temperature, up.ion_temperature,
            up.ion_species, self.shock_velocity, self.theta, gamma=self.gamma,
            branch=self.branch)
        jc = self.jump_conditions
        return (jc.downstream_density, jc.downstream_normal_velocity,
                jc.downstream_electron_temperature, jc.downstream_ion_temperature,
                jc.downstream_magnetic_field)

    @property
    def is_supercritical(self):
        """True when the downstream normal flow is subsonic (above the first
        critical Mach number, Edmiston & Kennel 1984)."""
        return bool(self.jump_conditions.downstream_sonic_mach_number < 1)

    @property
    def shock_type(self):
        """'fast', 'switch-on' (theta = 0, r = M_A^2, tangential B created) or 'slow'."""
        return self.jump_conditions.shock_type

    def compute_phase_crossing(self):
        """k and omega where the shock speed matches the upstream fast-mode
        phase speed, and the downstream flow matches the downstream Alfven
        (else acoustic) phase speed.  NaN where there is no crossing.
        (Was comparing bare ``.value``s, which silently assumed m/s.)"""
        ms = u.m / u.s

        def crossing(v_phase, omegas, speed):
            diff = np.real(v_phase).to_value(ms) - speed.to_value(ms)
            idx = np.argwhere(np.diff(np.sign(diff))).flatten()
            if len(idx) == 0:
                return None
            return self.k_space[idx][0], omegas[idx][0]

        up = crossing(self.v_phase_upstream['fast_mode'], self.omegas_upstream['fast_mode'],
                      self.shock_velocity)
        k_upstream, omega_upstream = up if up else (np.nan, np.nan)

        down = None
        for mode in ('alfven_mode', 'acoustic_mode'):
            down = crossing(self.v_phase_downstream[mode], self.omegas_downstream[mode],
                            self.downstream_velocity)
            if down:
                break
        k_downstream, omega_downstream = down if down else (np.nan, np.nan)
        return k_upstream, k_downstream, omega_upstream, omega_downstream

    def plot_dispersions(self, return_plot=False):
        with _style():
            f, a = plt.subplots(1, 2, figsize=(6, 5))

            for key, value in self.omegas_upstream.items():
                a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                          np.real(value) / self.upstream_plasma.plasma_parameters.electron_plasma_frequency,
                          label=key)

            for key, value in self.omegas_downstream.items():
                a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                          np.real(value) / self.downstream_plasma.plasma_parameters.electron_plasma_frequency,
                          label=key)

            a[0].scatter(self.k_upstream * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                         self.omega_upstream / self.upstream_plasma.plasma_parameters.electron_plasma_frequency,
                         alpha=1.0, zorder=1000, color='k')

            a[1].scatter(self.k_downstream * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                         self.omega_downstream / self.downstream_plasma.plasma_parameters.electron_plasma_frequency,
                         alpha=1.0, zorder=1000, color='k')

            a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                      self.downstream_velocity * self.k_space / self.downstream_plasma.plasma_parameters.electron_plasma_frequency,
                      color='k', ls='--', label='Shock Velocity')

            a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                      self.shock_velocity * self.k_space / self.upstream_plasma.plasma_parameters.electron_plasma_frequency,
                      color='k', ls='--', label='Shock Velocity')

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
            f.tight_layout()
        if return_plot:
            return f, a

    def plot_phase_velocities(self):
        with _style():
            f, a = plt.subplots(1, 2)

            for key, value in self.v_phase_upstream.items():
                a[0].plot(self.k_space * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                          np.real(value) / c,
                          label=key)

            for key, value in self.v_phase_downstream.items():
                a[1].plot(self.k_space * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                          np.real(value) / c,
                          label=key)

            a[0].scatter(self.k_upstream * self.upstream_plasma.plasma_parameters.electron_inertial_length,
                         self.shock_velocity / c)
            a[1].scatter(self.k_downstream * self.downstream_plasma.plasma_parameters.electron_inertial_length,
                         self.downstream_velocity / c)

            a[1].axhline(self.downstream_velocity / c)

            a[0].axhline(self.shock_velocity / c)

            a[0].legend()
            a[1].legend()
            a[1].set_xlabel(r"$ck/\omega_{pe}$")
            a[0].set_xlabel(r"$ck/\omega_{pe}$")

            a[0].set_ylabel(r"$v_{ph.}/c$")
            a[1].set_ylabel(r"$v_{ph.}/c$")

            a[0].set_title("Upstream Plasma")
            a[1].set_title("Downstream Plasma")

            a[0].set_yscale("log")
            a[0].set_xscale("log")

            a[1].set_yscale("log")
            a[1].set_xscale("log")

            plt.show()


if __name__ == '__main__':
    density = 5e12 * u.cm**(-3)
    magnetic_field = 100 * u.G
    electron_temperature = 5 * u.eV
    ion_temperature = 1 * u.eV
    ion_species = Particle('p+')
    shock_velocity = 300e3 * u.m / u.s
    shock_theta = 30.0 * u.deg      # any angle now works
    gamma = 5 / 3

    shocked_plasma = ShockedPlasma(density,
                                   magnetic_field,
                                   electron_temperature,
                                   ion_temperature,
                                   ion_species,
                                   shock_velocity,
                                   shock_theta,
                                   gamma,
                                   compute_dispersion=True,
                                   verbose=True)

    jc = shocked_plasma.jump_conditions
    print("r =", jc.compression_ratio, " |B2|/|B1| =", jc.magnetic_field_ratio,
          " theta_Bn2 =", jc.downstream_theta, " supercritical:", shocked_plasma.is_supercritical)
    print(shocked_plasma.downstream_velocity / shocked_plasma.downstream_plasma.plasma_parameters.sound_speed)
    shocked_plasma.plot_dispersions()
    shocked_plasma.plot_phase_velocities()
