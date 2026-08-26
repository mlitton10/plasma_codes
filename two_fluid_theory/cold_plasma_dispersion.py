import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq, linear_sum_assignment


# ============================================================
# Physical constants
# ============================================================

c = 299792458.0
e = 1.602176634e-19
eps0 = 8.8541878128e-12
me = 9.1093837015e-31
mp = 1.67262192369e-27


# ============================================================
# Plasma parameters
# ============================================================

# Background magnetic field [Tesla]
B0 = 8.3e-9

# Number density [m^-3]
n0 = 5e6

# Propagation angle relative to B0
theta = np.deg2rad(45.0)

#theta = np.deg2rad(90.0)


# ============================================================
# Species
#
# IMPORTANT:
# Cyclotron frequency is signed:
#
# Omega_s = q_s B0 / m_s
# ============================================================

species = [
    {
        "name": "electron",
        "q": -e,
        "m": me,
        "n": n0,
    },
    {
        "name": "ion",
        "q": e,
        "m": mp,
        "n": n0,
    },
]

def plasma_frequency(q, m, n):
    return np.sqrt(n * q ** 2 / (eps0 * m))

def cyclotron_frequency(q, m, B):
    return q * B / m

for sp in species:
    sp["wp"] = plasma_frequency(sp['q'], sp['m'], sp['n'])

    sp["Omega"] = cyclotron_frequency(sp['q'], sp['m'], B0)


# Useful reference frequencies
wp_e = species[0]["wp"]
wp_i = species[1]["wp"]

Omega_e = species[0]["Omega"]
Omega_i = species[1]["Omega"]


print("Electron plasma frequency:", wp_e)
print("Ion plasma frequency:", wp_i)

print("Electron cyclotron frequency:", Omega_e)
print("Ion cyclotron frequency:", Omega_i)


# ============================================================
# Cold plasma dielectric tensor
# ============================================================

def dielectric_tensor(omega):
    """
    Return the cold plasma dielectric tensor epsilon(omega).

    omega can be a positive real frequency.
    """

    S = 1.0
    D = 0.0
    P = 1.0

    for sp in species:

        wp = sp["wp"]
        Omega = sp["Omega"]

        S -= wp**2 / (
            omega**2 - Omega**2
        )

        D += (
            (Omega / omega)
            * wp**2
            / (omega**2 - Omega**2)
        )

        P -= wp**2 / omega**2

    epsilon = np.array([
        [S, -1j * D, 0.0],
        [1j * D, S, 0.0],
        [0.0, 0.0, P],
    ], dtype=complex)

    return epsilon


# ============================================================
# Dispersion matrix
#
# M = epsilon - n^2 (I - k_hat k_hat)
# ============================================================

def dispersion_matrix(k, omega, theta):

    khat = np.array([
        np.sin(theta),
        0.0,
        np.cos(theta),
    ])

    I = np.eye(3)

    n2 = (
        c**2 * k**2
        / omega**2
    )

    epsilon = dielectric_tensor(omega)

    M = epsilon - n2 * (
        I - np.outer(khat, khat)
    )

    return M


def dispersion_function(k, omega, theta):

    M = dispersion_matrix(
        k,
        omega,
        theta,
    )

    det = np.linalg.det(M)

    # For real omega and k,
    # the determinant should be real apart
    # from numerical roundoff.
    return det.real


# ============================================================
# Find all roots for a given k
# ============================================================

def find_roots_for_k(
    k,
    theta,
    omega_min,
    omega_max,
    n_scan=10000,
):

    # Search uniformly in log frequency
    omega_grid = np.logspace(
        np.log10(omega_min),
        np.log10(omega_max),
        n_scan,
    )

    f = np.array([
        dispersion_function(
            k,
            omega,
            theta,
        )
        for omega in omega_grid
    ])

    roots = []

    for i in range(len(omega_grid) - 1):

        w1 = omega_grid[i]
        w2 = omega_grid[i + 1]

        f1 = f[i]
        f2 = f[i + 1]

        # Skip poles / NaNs
        if not np.isfinite(f1):
            continue

        if not np.isfinite(f2):
            continue

        # Standard sign-change detection
        if f1 * f2 < 0:

            try:

                root = brentq(
                    lambda w: dispersion_function(
                        k,
                        w,
                        theta,
                    ),
                    w1,
                    w2,
                    xtol=1e-10,
                    rtol=1e-10,
                )

                roots.append(root)

            except ValueError:
                pass

    # Remove duplicate roots
    roots = np.array(
        sorted(roots)
    )

    if len(roots) == 0:
        return roots

    unique_roots = [roots[0]]

    for root in roots[1:]:

        relative_difference = abs(
            root - unique_roots[-1]
        ) / root

        if relative_difference > 1e-6:
            unique_roots.append(root)

    return np.array(unique_roots)


# ============================================================
# Track branches using continuation
# ============================================================

def track_branches(
    k_values,
    theta,
    omega_min,
    omega_max,
    n_scan=100000,
):

    branches = []

    previous_roots = None

    for k_index, k in enumerate(k_values):

        roots = find_roots_for_k(
            k,
            theta,
            omega_min,
            omega_max,
            n_scan=n_scan,
        )

        print(
            f"k index {k_index + 1}/{len(k_values)}: "
            f"{len(roots)} roots"
        )

        # First k value:
        # initialize one branch per root
        if previous_roots is None:

            for root in roots:

                branch = np.full(
                    len(k_values),
                    np.nan,
                )

                branch[k_index] = root

                branches.append(branch)

        else:

            n_old = len(previous_roots)
            n_new = len(roots)

            if n_old > 0 and n_new > 0:

                # Cost matrix in log-frequency.
                # This is better than absolute difference
                # because frequencies may span many decades.
                cost = np.abs(
                    np.log(previous_roots[:, None])
                    - np.log(roots[None, :])
                )

                row_ind, col_ind = (
                    linear_sum_assignment(cost)
                )

                matched_new = set()

                # Update existing branches
                for old_index, new_index in zip(
                    row_ind,
                    col_ind,
                ):

                    branches[old_index][k_index] = (
                        roots[new_index]
                    )

                    matched_new.add(new_index)

                # Create new branches for unmatched roots
                for new_index, root in enumerate(roots):

                    if new_index not in matched_new:

                        branch = np.full(
                            len(k_values),
                            np.nan,
                        )

                        branch[k_index] = root

                        branches.append(branch)

            else:

                # If no previous roots exist,
                # create branches for all current roots.
                if n_old == 0:

                    for root in roots:

                        branch = np.full(
                            len(k_values),
                            np.nan,
                        )

                        branch[k_index] = root

                        branches.append(branch)

        # Store current roots ordered by frequency
        previous_roots = roots

    return np.array(branches)


# ============================================================
# k grid
# ============================================================

# Use dimensionless k c / omega_pe
kc_over_wpe = np.logspace(
    -4,
    2,
    500,
)

k_values = (
    kc_over_wpe
    * wp_e
    / c
)


# ============================================================
# Frequency search range
#
# Include frequencies well below ion cyclotron
# and well above electron cyclotron.
# ============================================================

omega_min = (
    min(abs(Omega_i), wp_i)
    * 1e-4
)

omega_max = max(
    abs(Omega_e),
    wp_e,
) * 100.0


# ============================================================
# Solve and separate branches
# ============================================================

branches = track_branches(
    k_values=k_values,
    theta=theta,
    omega_min=omega_min,
    omega_max=omega_max,
    n_scan=10000,
)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 7))

for i, branch in enumerate(branches):

    valid = np.isfinite(branch)

    if np.any(valid):

        plt.loglog(
            kc_over_wpe[valid],
            branch[valid] / abs(Omega_e),
            linewidth=1.5,
            label=f"Branch {i}",
        )


plt.xlabel(
    r"$kc/\omega_{pe}$"
)

plt.ylabel(
    r"$\omega/|\Omega_e|$"
)

plt.title(
    f"Cold Two-Species Plasma Dispersion Relation\n"
    f"$\\theta = {np.rad2deg(theta):.1f}^\\circ$"
)

plt.grid(
    True,
    which="both",
    alpha=0.3,
)

plt.legend(
    loc="best",
    fontsize=8,
)

plt.show()