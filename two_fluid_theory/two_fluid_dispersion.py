import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants.si import c
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plasmapy.dispersion.analytical.two_fluid_ import two_fluid
from plasmapy.formulary import speeds
from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency, wc_, wp_
from plasmapy.formulary.lengths import inertial_length
from plasmapy.particles import Particle


inputs = {
    "k": 10**np.linspace(-2, 8, 1000000) * u.rad / u.m,
    "theta": 0 * u.deg,
    "n_i": 10**13 * u.cm**-3,
    "B": 250 * u.G,
    "T_e": 1 * u.eV,
    "T_i": 1 * u.eV,
    "ion": Particle("p+"),
}

# a few useful plasma parameters
params = {
    "n_e": inputs["n_i"] * abs(inputs["ion"].charge_number),
    "cs": speeds.ion_sound_speed(
        inputs["T_e"],
        inputs["T_i"],
        inputs["ion"],
    ),
    "va": speeds.Alfven_speed(
        inputs["B"],
        inputs["n_i"],
        ion=inputs["ion"],
    ),
    "wci": gyrofrequency(inputs["B"], inputs["ion"]),
}
params["lpe"] = inertial_length(params["n_e"], "e-")
params["wpe"] = plasma_frequency(params["n_e"], "e-")


# compute
omegas = two_fluid(**inputs)

print(list(omegas.keys()), omegas["fast_mode"], omegas["fast_mode"].shape)


fs = 14  # default font size
figwidth, figheight = plt.rcParams["figure.figsize"]
figheight = 1.6 * figheight
fig = plt.figure(figsize=[figwidth, figheight])

# normalize data
k_prime = inputs["k"] * params["lpe"]

# plot
plt.plot(
    k_prime,
    np.real(omegas["fast_mode"] / params["wpe"]),
    "r.",
    ms=1,
    label="Fast",
)
"""
plt.plot(
    k_prime,
    k_prime,
    "k-.",
    ms=1,
    label="Light",
)

plt.plot(
    k_prime,
    0.001 * k_prime,
    "k.",
    ms=1,
    label="Piston",
)
"""
ax = plt.gca()
ax.plot(
    k_prime,
    np.real(omegas["alfven_mode"] / params["wpe"]),
    "b.",
    ms=1,
    label="Alfvén",
)
ax.plot(
    k_prime,
    np.real(omegas["acoustic_mode"] / params["wpe"]),
    "g.",
    ms=1,
    label="Acoustic",
)

# adjust axes
ax.set_xlabel(r"$kc / \omega_{pe}$", fontsize=fs)
ax.set_ylabel(r"$Re(\omega / \omega_{pe})$", fontsize=fs)
ax.set_yscale("log")
ax.set_xscale("log")
#ax.set_ylim(1e-6, 2e-2)

ax.tick_params(
    which="both",
    direction="in",
    width=1,
    labelsize=fs,
    right=True,
    length=5,
)

# annotate
text = (
    rf"$v_A/c_s = {params['va'] / params['cs']:.1f} \qquad "
    rf"c/v_A = 10^{np.log10(c / params['va']):.0f} \qquad "
    f"\\theta = {inputs['theta'].value:.0f}"
    "^{\\circ}$"
)
ax.text(0.25, 0.95, text, transform=ax.transAxes, fontsize=18)
ax.legend(loc="upper left", markerscale=5, fontsize=fs)

plt.show()