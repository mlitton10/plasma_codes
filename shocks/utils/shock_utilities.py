import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

def compression_ratio_parallel(v_shock, c_s, gamma):
    M_s = v_shock / c_s
    numerator = (M_s ** 2) * (gamma + 1)
    denominator = (M_s ** 2) * (gamma - 1) + 2
    return numerator / denominator

def compression_ratio_perpendicular(v_shock, c_s, v_a, gamma):
    beta = (2/(gamma)) * ((c_s ** 2) / (v_a ** 2))
    v_f = np.sqrt(v_a ** 2 + c_s ** 2)
    M = v_shock / v_f

    c_0 = -gamma * (gamma + 1) * beta * (M ** 2)
    c_1 = gamma * (2 * (1 + beta) + (gamma - 1) * beta * (M ** 2))
    c_2 = 2 * (2 - gamma)
    p = Polynomial([c_0, c_1, c_2])
    roots = p.roots()
    final_ratio = None
    for root in roots:
        if 0 < root < (gamma + 1) / (gamma - 1):
            final_ratio = root
    print(roots)
    return final_ratio


def compression_ratio(v_shock, c_s, v_a, theta, gamma):
    if theta == 0:
        return compression_ratio_parallel(v_shock, c_s, gamma)
    elif theta == np.pi/2:
        return compression_ratio_perpendicular(v_shock, c_s, v_a, gamma)
    v_shock = v_shock * np.cos(theta)
    c2 = np.cos(theta) ** 2

    a3 = -v_a ** 4 * c2 * (
            2 * v_a ** 2 * c2
            + (gamma - 1) * v_shock ** 2
    )

    a2 = v_a ** 2 * v_shock ** 2 * (
            (gamma + 5) * v_a ** 2 * c2
            + (gamma * (1 + c2) - 2) * v_shock ** 2
    )

    a1 = -v_shock ** 4 * (
            (gamma + 2) * v_a ** 2 * (1 + c2)
            + (gamma - 1) * v_shock ** 2
    )

    a0 = (gamma + 1) * v_shock ** 6

    shock_adiabatic = Polynomial([a0, a1, a2, a3])

    roots = shock_adiabatic.roots()

    final_ratio = None
    for root in roots:
        if root > 1 and np.isreal(root):
            final_ratio = root
    return final_ratio

def pressure_ratio_parallel(v_shock, c_s, gamma):
    r = compression_ratio_parallel(v_shock, c_s, gamma)
    M_s = v_shock / c_s
    return 1 + gamma * (1 - (1/r)) * M_s ** 2

def pressure_ratio_perpendicular(v_shock, c_s, v_a, gamma):
    beta = (2*4*np.pi / gamma) * ((c_s ** 2) / (v_a ** 2))
    v_f = np.sqrt(v_a ** 2 + c_s ** 2)
    M = v_shock / v_f
    r = compression_ratio_perpendicular(v_shock, c_s, v_a, gamma)

    return 1+ gamma * (M ** 2)* (1 - (1 / r)) + (1 - r ** 2) / beta

def pressure_ratio(v_shock, c_s, v_a, theta, gamma):
    if theta == 0:
        return pressure_ratio_parallel(v_shock, c_s, gamma)
    elif theta == np.pi/2:
        return pressure_ratio_perpendicular(v_shock, c_s, v_a, gamma)
    v_shock = v_shock * np.cos(theta)

    r = compression_ratio(v_shock, c_s, v_a, theta, gamma)

    c_0 = 1

    c_1 = (gamma * (v_shock**2) * (r - 1)) / ((c_s ** 2) * r)

    c_2_num_pre = r * (v_a ** 2) * (np.sin(theta) ** 2)
    c_2_num_post = (r+1) * (v_shock ** 2) - 2 * r * (v_a ** 2) *(np.cos(theta) ** 2)
    c_2_num = c_2_num_pre * c_2_num_post

    c_2_den = 2 * ((v_shock ** 2) - r * (v_a ** 2)*(np.cos(theta) ** 2)) ** 2

    c_2 = c_2_num / c_2_den

    return c_0 + c_1 * ( 1 - c_2)

def tangential_field_ratio(v_shock, c_s, v_a, theta, gamma):
    if theta == 0:
        return 1
    else:
        v_shock = v_shock * np.cos(theta)
        r = compression_ratio(v_shock, c_s, v_a, theta, gamma)
        numerator = v_shock**2 - (np.cos(theta)**2) * (v_a**2)
        denominator = v_shock**2 - r * (np.cos(theta)**2) * (v_a**2)
        return r * numerator / denominator

def tangential_velocity_ratio(v_shock, c_s, v_a, theta, gamma):
    if theta == 0:
        return 1
    else:
        v_shock = v_shock * np.cos(theta)
        r = compression_ratio(v_shock, c_s, v_a, theta, gamma)
        numerator = v_shock ** 2 - (np.cos(theta) ** 2) * (v_a ** 2)
        denominator = v_shock ** 2 - r * (np.cos(theta) ** 2) * (v_a ** 2)
        return numerator / denominator

def normal_velocity_ratio(v_shock, c_s, v_a, theta, gamma):
    r = compression_ratio(v_shock, c_s, v_a, theta, gamma)
    return 1 / r


if __name__ == '__main__':
    v_shock = np.linspace(1, 100, 200)
    c_s = 1
    v_a = 10
    gamma = 5/3

    r_parallel = np.array([compression_ratio(vs, c_s, v_a, 0, gamma)  for vs in v_shock])
    r_perp = np.array([compression_ratio(vs, c_s, v_a, np.pi/2, gamma) for vs in v_shock])
    r_oblique = np.array([compression_ratio(vs, c_s, v_a, np.pi/4, gamma) for vs in v_shock])

    v_f_perp = np.sqrt(v_a ** 2 + c_s ** 2)
    v_f_o = np.sqrt(0.5 * (v_a ** 2 + c_s ** 2) + np.sqrt((v_a ** 2 + c_s ** 2) ** 2 - 4*(c_s**2)*(v_a ** 2)))

    f,a = plt.subplots()
    a.plot(v_shock, r_parallel)
    a.plot(v_shock/v_f_perp, r_perp)
    a.plot(v_shock/v_f_o, r_oblique, ls='--')
    a.set_xlabel('v_shock')
    print((2/(gamma)) * ((c_s ** 2) / (v_a ** 2)))
    plt.show()
