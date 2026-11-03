"""Collisionless-shock constraint space (port of
``jump_condition_based_constraints_comparisons.ipynb``).

The notebook's NRL-formulary one-liners (cgs, hard-coded constants) are
replaced by the same plasmapy / ``plasma_codes`` definitions used everywhere
else, with astropy units throughout.  Dimensionless outputs are plain floats
/ ndarrays so the existing plotting cells work unchanged.

Constraint quantities (notebook key -> definition)
--------------------------------------------------
n_period      omega_ci * tau,   tau = L / v_shock          (upstream B)
ion_ratio     L / d_i           d_i = c / omega_pi
lambda_a      L / lambda_A      lambda_A = (v_A / f) sqrt(1 - (f/f_ci)^2),
                                f = wave_frequency_fraction * f_ci (0.7 as before)
mach_a        v_shock / v_A
mach_f        v_shock / c_fast(theta)                       (new)
magnetization D / r_gi,2        downstream ion gyroradius, downstream |B| and T_i
                                (plasmapy gyroradius: thermal speed sqrt(2kT/m),
                                i.e. sqrt(2) x the NRL r_i the notebook used)
beta          downstream TOTAL beta from the MHD jump (the notebook used
              R n T / B1^2 with a gas-dynamic R and upstream B)
compression_ratio, beta_upstream, supercritical               (new)
shock_type    'fast' | 'switch-on' | 'none' (no fast shock)  (new)

The jump uses the full MHD solver at ``shock_theta`` (default 0 deg, matching
the notebook's parallel assumption, but now including the switch-on regime).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import astropy.units as u
import numpy as np
from plasmapy.formulary import Alfven_speed, gyrofrequency, gyroradius, inertial_length
from plasmapy.particles import Particle

from plasma_codes.formulary import magnetosonic_speeds, mhd_sound_speed
from plasma_codes.shocks.jump_conditions import mhd_jump_conditions

__all__ = ["ExperimentRanges", "shock_constraints", "scan_experiment"]

_ANG = u.dimensionless_angles()


def _num(q):
    return np.asarray(u.Quantity(q).to_value(u.dimensionless_unscaled, equivalencies=_ANG), float)


def shock_constraints(density, magnetic_field, electron_temperature, ion_temperature,
                      ion_species, shock_velocity, length, transverse_size=None,
                      shock_theta=0 * u.deg, gamma=5 / 3, wave_frequency_fraction=0.7,
                      solve_jumps=True) -> dict:
    """Constraint-space coordinates for one ion species; all other inputs
    broadcast (scalars or arrays).  Returns a dict of plain floats/arrays."""
    ion = Particle(ion_species) if isinstance(ion_species, str) else ion_species
    n = u.Quantity(density)
    B = u.Quantity(magnetic_field)
    v = u.Quantity(shock_velocity)
    L = u.Quantity(length)

    omega_ci = gyrofrequency(B, ion)                       # rad / s
    tau = L / v
    v_A = Alfven_speed(B, n, ion)
    d_i = inertial_length(n, ion)
    f_ci = omega_ci / (2 * np.pi * u.rad)
    f = wave_frequency_fraction * f_ci
    lambda_A = (v_A / f) * np.sqrt(1 - (f / f_ci) ** 2)

    c_s = mhd_sound_speed(n, electron_temperature, ion_temperature, ion, gamma)
    _, c_f = magnetosonic_speeds(v_A, c_s, shock_theta)

    out = {
        "n_period": _num(omega_ci * tau),
        "ion_ratio": _num(L / d_i),
        "lambda_a": _num(L / lambda_A),
        "mach_a": _num(v / v_A),
        "mach_f": _num(v / c_f),
    }

    shape = np.broadcast(out["n_period"], out["mach_f"]).shape
    nan = np.full(shape, np.nan)
    out.update(magnetization=nan, beta=nan, beta_upstream=nan,
               compression_ratio=nan, n_period=nan, ion_ratio=nan, supercritical=np.zeros(shape, bool),
               shock_type=np.full(shape, "not solved", dtype=object))
    if solve_jumps:
        jc = mhd_jump_conditions(n, B, electron_temperature, ion_temperature, ion, v,
                                 shock_theta, gamma=gamma)
        out["beta"] = _num(jc.downstream_total_beta)
        out["beta_upstream"] = _num(jc.upstream_total_beta)
        out["compression_ratio"] = _num(jc.compression_ratio)

        out["supercritical"] = np.asarray(_num(jc.downstream_sonic_mach_number) < 1)
        out["shock_type"] = np.asarray(jc.shock_type, dtype=object)
        if transverse_size is not None:
            ok = np.asarray(jc.exists)
            B2 = np.where(ok, jc.downstream_magnetic_field.to_value(u.T), 1.0) * u.T
            Ti2 = np.where(ok, jc.downstream_ion_temperature.to_value(u.eV), 1.0) * u.eV
            r_g2 = gyroradius(B2, ion, T=Ti2)
            out["magnetization"] = np.where(ok, _num(u.Quantity(transverse_size) / r_g2), np.nan)
    return {k: np.broadcast_to(val, shape).copy() if np.ndim(val) else val
            for k, val in out.items()}


@dataclass
class ExperimentRanges:
    """Parameter ranges for one device / system (replaces the notebook dicts).

    ``ion_temperature`` defaults to ``ion_to_electron_temperature_ratio * T_e``;
    the notebook used a single T for both species, i.e. ratio = 1.
    ``transverse_size`` is the notebook's 'R' (plasma size used for the
    magnetization D / r_gi); leave None if not defined.
    """
    name: str
    magnetic_field: u.Quantity
    density: u.Quantity
    electron_temperature: u.Quantity
    shock_velocity: u.Quantity
    length: u.Quantity
    ion_species: Sequence = field(default_factory=lambda: [Particle("p+")])
    transverse_size: Optional[u.Quantity] = None
    ion_to_electron_temperature_ratio: float = 1.0
    shock_theta: u.Quantity = 0 * u.deg
    gamma: float = 5 / 3


def scan_experiment(ranges: ExperimentRanges, solve_jumps: bool = True,
                    wave_frequency_fraction: float = 0.7) -> dict:
    """Full-factorial scan (same ordering as the notebook's nested loops:
    ion, T, B, v, n with n fastest).  Returns flat arrays keyed as the old
    ``scanned_parameters`` output plus 'mach_f', 'compression_ratio',
    'beta_upstream', 'supercritical', 'shock_type', 'ion'.  'T', 'B', 'v', 'n' are Quantities.

    theta = 0 or 90 deg is fully vectorised; other angles loop over the
    polynomial solver (~10^4 points/s), so consider ``solve_jumps=False`` for
    large oblique scans that only need the upstream constraints.
    """
    T, B, v, n = np.meshgrid(u.Quantity(ranges.electron_temperature),
                             u.Quantity(ranges.magnetic_field),
                             u.Quantity(ranges.shock_velocity),
                             u.Quantity(ranges.density), indexing="ij")
    T, B, v, n = (q.ravel() for q in (T, B, v, n))
    pieces = []
    for ion in ranges.ion_species:
        ion = Particle(ion) if isinstance(ion, str) else ion
        d = shock_constraints(n, B, T, ranges.ion_to_electron_temperature_ratio * T, ion, v,
                              ranges.length, ranges.transverse_size, ranges.shock_theta,
                              ranges.gamma, wave_frequency_fraction, solve_jumps)
        d.update(T=T, B=B, v=v, n=n, ion=np.full(T.shape, ion.symbol, dtype=object))
        pieces.append(d)
    return {k: (u.Quantity(np.concatenate([p[k].value for p in pieces]), pieces[0][k].unit)
                if isinstance(pieces[0][k], u.Quantity)
                else np.concatenate([p[k] for p in pieces]))
            for k in pieces[0]}
