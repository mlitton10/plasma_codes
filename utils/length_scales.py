import numpy as np
from astropy.units import Quantity
import astropy.units as u
from astropy.constants import m_p, e
print(m_p,e )


def electron_debroigle_length(T: Quantity) -> Quantity:
    return (2.76e-8 / np.sqrt(T.to(u.eV).value)) * u.cm

def classical_minimum_approach(T: Quantity) -> Quantity:
    return (1.44e-7 / T.to(u.eV).value) * u.cm

def electron_gyroradius(T: Quantity, B: Quantity) -> Quantity:
    r = 2.38 * np.sqrt(T.to(u.eV).value) / B.to(u.G).value
    return r * u.cm

def ion_gyroradius(T: Quantity, B: Quantity, m_i: Quantity, q: Quantity) -> Quantity:
    mu = m_i.to(u.kg)/m_p.value
    Z = q / e
    r = 1.02e2 * np.sqrt(mu * T.to(u.eV).value) / (Z * B.to(u.G).value)
    return r * u.cm

def electron_interial_length(n: Quantity) -> Quantity:
    l = 5.31e5 / np.sqrt(n.to(u.cm**3).value)
    return l * u.cm

def ion_interial_length(n: Quantity, m_i: Quantity, q: Quantity) -> Quantity:
    mu = m_i.to(u.kg) / m_p.value
    Z = q / e

    l = 2.28e7 * np.sqrt(mu / n.to(u.cm**3).value) / Z
    return l * u.cm

def debye_length(T: Quantity, n: Quantity) -> Quantity:
    l = 7.43e2 * np.sqrt(T.to(u.eV).value/n.to(u.cm**3).value)
    return l * u.cm
