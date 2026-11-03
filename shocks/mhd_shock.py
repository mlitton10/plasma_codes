"""
mhd_shock.py -- ideal-MHD (isotropic, polytropic) Rankine-Hugoniot solver for
oblique shocks in the normal-incidence frame (NIF).

This is the DIMENSIONLESS engine.  It knows nothing about astropy or plasmapy;
the units-aware front end is ``plasma_codes.shocks.jump_conditions``.  Every
dimensional jump condition in ``plasma_codes`` goes through
``solve_shock_array`` below, so there is exactly one implementation of the
Rankine-Hugoniot physics.

Geometry / definitions
----------------------
Shock normal = x.  Upstream flow is along +x with speed u1 (the NORMAL
component, so for a flow at angle sigma to the shock surface u1 = u0*sin(sigma)).
Upstream field B1 lies in the x-y plane at angle theta (= theta_Bn) to the
normal:  B1 = B1 (cos theta, sin theta, 0).  Everything is Galilean-invariant,
so a tangential upstream flow can be added back afterwards.

    M_A  = u1 / v_A,   v_A^2 = B1^2/(mu0 rho1)     (TOTAL field, not B_x)
    beta = 2 mu0 p1 / B1^2                         (TOTAL thermal pressure)
    M_s  = M_A * sqrt(2/(gamma*beta))              (sonic Mach, normal flow)
    M_f  = u1 / c_f(theta)                         (fast-magnetosonic, normal flow)
    r    = rho2/rho1                               (compression ratio, r > 1)

theta is folded into [0, 90] deg (the jump is symmetric under theta -> 180 - theta).

Derivation (units rho1 = u1 = mu0 = 1; x = Bx^2, t = B1t^2, p1 = beta (x+t)/2)
------------------------------------------------------------------------------
Conserved across the jump ([f] = f2 - f1):
    mass          [rho u_x]                              = 0   -> u2 = 1/r
    normal mom.   [p + rho u_x^2 + B_t^2/2]              = 0
    tangential    [rho u_x u_t - Bx B_t]                 = 0   -> u2t = Bx (B2t - B1t)
    induction     [u_x B_t - u_t Bx]  (= -[E_z])         = 0
    energy        [u_x(rho u^2/2 + g p/(g-1)) + (u_x B_t - u_t Bx) B_t] = 0
    Bx continuous (div B = 0).

Induction + tangential momentum are linear in (B2t, u2t) and give, exactly,
    B2t/B1t = r (1 - x) / (1 - r x),       u2t = Bx B1t (r - 1)/(1 - r x),
normal momentum gives
    p2 = p1 + 1 - 1/r + (B1t^2 - B2t^2)/2,
and the energy flux balance then becomes a single equation E(r) = 0.
r^2 (1 - r x)^2 E(r) is a polynomial in r; r = 1 is a trivial root (no shock) and
the remainder is the cubic for the fast / intermediate / slow branches.  The
coefficients are obtained by exact polynomial interpolation of that expression.

Special geometries (closed forms, used for speed and exactness)
--------------------------------------------------------------
theta = 90 : quadratic  2(2-g) r^2 + g[2(1+beta) + (g-1) beta M_s^2] r
                         - g(g+1) beta M_s^2 = 0   (Priest; note SONIC Mach)
theta = 0  : gas-dynamic jump (B unchanged) when it is a fast shock
             (M_A^2 > r_hydro); otherwise the SWITCH-ON shock, r = M_A^2, with
             B2t^2 from the energy equation.  The general polynomial cannot
             find the switch-on shock because B1t = 0 makes it degenerate; the
             theta -> 0 limit of the oblique fast shock converges to it.

Branches are identified with the evolutionary conditions:
    fast :  u1 > c_f1,           vAx2 < u2 < c_f2
    slow :  c_s1 < u1 < vAx1,    u2 < c_s2
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
from scipy.optimize import brentq

# angular tolerance (deg) for treating theta as exactly parallel / perpendicular
_THETA_TOL_DEG = 1e-9

# ---------------------------------------------------------------------------
# polynomial machinery (interpolation nodes are fixed, so invert once)
# ---------------------------------------------------------------------------
_DEG = 8
_NODES = np.linspace(0.3, 3.3, _DEG + 1)
_VINV = np.linalg.inv(np.vander(_NODES, _DEG + 1))  # highest power first


def _energy_residual(r, x, t, p1, g):
    """E(r) from the closed form above (vectorised in r)."""
    Bx = np.sqrt(x)
    B1t = np.sqrt(t)
    den = 1.0 - r * x
    B2t = B1t * r * (1.0 - x) / den
    u2t = Bx * B1t * (r - 1.0) / den
    p2 = p1 + 1.0 - 1.0 / r + 0.5 * (t - B2t**2)
    lhs = 0.5 / r**2 + 0.5 * u2t**2 + g / (g - 1.0) * p2 / r + B1t * B2t
    rhs = 0.5 + g / (g - 1.0) * p1 + t
    return lhs - rhs


def _poly_coeffs(x, t, p1, g):
    """Coefficients (highest power first) of r^2 (1 - r x)^2 E(r)."""
    n = _NODES**2 * (1.0 - _NODES * x) ** 2 * _energy_residual(_NODES, x, t, p1, g)
    return _VINV @ n


def _polish(r, x, t, p1, g, iters=4):
    """A few Newton steps on the exact residual E(r) to remove interpolation
    round-off from the polynomial root."""
    for _ in range(iters):
        if abs(1.0 - r * x) < 1e-12:
            break
        h = 1e-6 * max(r - 1.0, 1e-3)
        f0 = _energy_residual(r, x, t, p1, g)
        d = (_energy_residual(r + h, x, t, p1, g) - _energy_residual(r - h, x, t, p1, g)) / (2 * h)
        if d == 0 or not np.isfinite(d):
            break
        step = f0 / d
        if abs(step) > 0.1 * max(r - 1.0, 1e-3):
            break
        r -= step
    return r


# ---------------------------------------------------------------------------
# characteristic speeds (vectorised)
# ---------------------------------------------------------------------------
def _speeds(rho, p, Bx2, Bt2, g):
    """(c_slow, vAx, c_fast) for the normal direction, mu0 = 1.  Array-safe."""
    a2 = g * p / rho
    vA2 = (Bx2 + Bt2) / rho
    vAx2 = Bx2 / rho
    s = a2 + vA2
    disc = np.sqrt(np.maximum(s * s - 4.0 * a2 * vAx2, 0.0))
    return (np.sqrt(np.maximum(0.5 * (s - disc), 0.0)), np.sqrt(vAx2),
            np.sqrt(0.5 * (s + disc)))


@dataclass
class Shock:
    branch: str
    r: float            # rho2/rho1
    Bt_ratio: float     # B2t/B1t   (NaN when B1t = 0)
    B_ratio: float      # |B2|/|B1|
    p_ratio: float      # p2/p1
    T_ratio: float      # T2/T1 (single fluid)
    u2n_over_u1: float  # 1/r
    u2t_over_u1: float  # tangential flow generated in the NIF
    deflection_deg: float   # atan(u2t/u2n) in the NIF
    theta2_deg: float   # downstream B angle to normal
    beta2: float
    Ms2: float          # u2n / downstream sound speed
    Mf2: float          # u2n / downstream fast speed
    Mf1: float
    Ms1: float
    MA: float
    beta: float
    theta_deg: float
    gamma: float
    switch_on: bool = False  # theta = 0 fast shock with r = M_A^2 and B2t != 0


_FLOAT_FIELDS = [f.name for f in fields(Shock) if f.name not in ("branch", "switch_on")]


# ---------------------------------------------------------------------------
# Mach-number conversions (array-safe)
# ---------------------------------------------------------------------------
def MA_from_Mf(Mf, beta, theta_deg, gamma=5 / 3):
    """Alfven Mach number (total B) giving fast Mach number Mf at (beta, theta)."""
    h = 0.5 * gamma * np.asarray(beta, dtype=float)
    c2 = np.cos(np.radians(theta_deg)) ** 2
    K = 0.5 * ((h + 1.0) + np.sqrt((h + 1.0) ** 2 - 4.0 * h * c2))
    return Mf * np.sqrt(K)


def Mf_from_MA(MA, beta, theta_deg, gamma=5 / 3):
    return MA / MA_from_Mf(1.0, beta, theta_deg, gamma)


def Ms_from_MA(MA, beta, gamma=5 / 3):
    """Sonic Mach number from Alfven Mach number: c_s^2 / v_A^2 = gamma beta / 2."""
    return MA * np.sqrt(2.0 / (gamma * np.asarray(beta, dtype=float)))


# ---------------------------------------------------------------------------
# downstream state from (r, B2t): the ONLY place downstream quantities are built
# ---------------------------------------------------------------------------
def _downstream_state(MA, beta, th, g, r, B2t):
    """All Shock fields (except branch) from the compression ratio and the
    downstream tangential field (code units).  Vectorised; th in radians."""
    with np.errstate(divide="ignore", invalid="ignore"):
        B2 = 1.0 / MA**2
        x = B2 * np.cos(th) ** 2
        t = B2 * np.sin(th) ** 2
        Bx, B1t = np.sqrt(x), np.sqrt(t)
        p1 = 0.5 * beta * B2
        p2 = p1 + 1.0 - 1.0 / r + 0.5 * (t - B2t**2)
        u2n = 1.0 / r
        u2t = Bx * (B2t - B1t)                        # tangential momentum
        Bmag2 = np.hypot(Bx, B2t)
        a2 = np.sqrt(g * p2 / r)
        _, _, cf2 = _speeds(r, p2, x, B2t**2, g)
        _, _, cf1 = _speeds(1.0, p1, x, t, g)
        return dict(
            r=r, Bt_ratio=np.where(t > 0, B2t / B1t, np.nan),
            B_ratio=Bmag2 / np.sqrt(B2), p_ratio=p2 / p1, T_ratio=(p2 / p1) / r,
            u2n_over_u1=u2n, u2t_over_u1=u2t,
            deflection_deg=np.degrees(np.arctan2(u2t, u2n)),
            theta2_deg=np.degrees(np.arctan2(np.abs(B2t), Bx)),
            beta2=2.0 * p2 / Bmag2**2, Ms2=u2n / a2, Mf2=u2n / cf2,
            Mf1=1.0 / cf1, Ms1=Ms_from_MA(MA, beta, g), MA=MA, beta=beta,
            theta_deg=np.degrees(th), gamma=g,
        )


# ---------------------------------------------------------------------------
# closed forms for theta = 0 and theta = 90 (vectorised)
# ---------------------------------------------------------------------------
def _parallel(MA, beta, g, branch):
    """theta = 0.  Returns (r, B2t) arrays, NaN where no shock of this branch."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Ms = Ms_from_MA(MA, beta, g)
        r_h = (g + 1.0) * Ms**2 / ((g - 1.0) * Ms**2 + 2.0)
        r = np.full(np.shape(MA), np.nan)
        B2t = np.zeros(np.shape(MA))
        if branch == "slow":
            ok = (Ms > 1.0) & (MA < 1.0)              # c_s1 < u1 < v_Ax1
            r[ok] = r_h[ok]
            return r, B2t
        if branch != "fast":
            raise ValueError("branch must be 'fast' or 'slow'")
        fast_up = (Ms > 1.0) & (MA > 1.0)             # u1 > c_f1 = max(c_s1, v_A1)
        hydro = fast_up & (MA**2 > r_h)               # u2 > v_Ax2: gas-dynamic fast shock
        r[hydro] = r_h[hydro]
        # switch-on: r = M_A^2, u2 = v_Ax2 (induction with B1t = 0)
        so = fast_up & ~hydro
        G = g / (g - 1.0)
        rs = MA**2
        p1 = 0.5 * beta / MA**2
        rhs = 0.5 + G * p1 - 0.5 / rs**2 - G * (p1 + 1.0 - 1.0 / rs) / rs
        Bt2sq = 2.0 * rs * rhs / (1.0 - G)
        p2 = p1 + 1.0 - 1.0 / rs - 0.5 * Bt2sq
        so &= (Bt2sq > 0.0) & (p2 > 0.0) & (rs > 1.0)
        r[so] = rs[so]
        B2t[so] = np.sqrt(Bt2sq[so])
        return r, B2t


def _perpendicular(MA, beta, g, branch):
    """theta = 90.  Returns (r, B2t) arrays, NaN where no fast shock."""
    r = np.full(np.shape(MA), np.nan)
    if branch != "fast":                              # no slow shock at theta = 90
        return r, np.zeros(np.shape(MA))
    with np.errstate(divide="ignore", invalid="ignore"):
        # Priest's quadratic is written with the SONIC Mach number; using
        # beta M_s^2 = 2 M_A^2 / g removes the 1/beta (and works at beta = 0).
        a = 2.0 * (2.0 - g)
        b = 2.0 * g * (1.0 + beta) + 2.0 * (g - 1.0) * MA**2
        c = -2.0 * (g + 1.0) * MA**2
        root = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
        ok = MA**2 > 1.0 + 0.5 * g * beta              # M_f > 1
        r[ok] = root[ok]
        B2t = r / MA                                   # B2t = r B1t, B1t = 1/MA
    return r, np.where(np.isfinite(r), B2t, 0.0)


# ---------------------------------------------------------------------------
# general oblique solver (scalar)
# ---------------------------------------------------------------------------
def _oblique(MA, beta, theta_deg, g, branch):
    """Scalar general-angle solve. Returns (r, B2t) or (nan, nan)."""
    th = np.radians(theta_deg)
    B2 = 1.0 / MA**2
    x = B2 * np.cos(th) ** 2
    t = B2 * np.sin(th) ** 2
    p1 = 0.5 * beta * B2

    cs1, vax1, cf1 = _speeds(1.0, p1, x, t, g)
    if branch == "fast" and not (1.0 > cf1):
        return np.nan, np.nan
    if branch == "slow" and not (cs1 < 1.0 < vax1):
        return np.nan, np.nan

    coef = _poly_coeffs(x, t, p1, g)
    # strip numerical zeros at the top, remove the trivial root r = 1
    scale = np.max(np.abs(coef))
    coef = np.where(np.abs(coef) < 1e-11 * scale, 0.0, coef)
    coef = np.trim_zeros(coef, "f")
    quo, _ = np.polydiv(coef, np.array([1.0, -1.0]))
    roots = np.roots(quo)
    rmax = (g + 1.0) / (g - 1.0)
    best = None
    for rt in roots:
        if abs(rt.imag) > 1e-9 * max(1.0, abs(rt.real)):
            continue
        r = _polish(rt.real, x, t, p1, g)
        if not (1.0 + 1e-9 < r <= rmax * (1 + 1e-9)):
            continue
        if abs(1.0 - r * x) < 1e-12:
            continue
        B2t = np.sqrt(t) * r * (1.0 - x) / (1.0 - r * x)
        p2 = p1 + 1.0 - 1.0 / r + 0.5 * (t - B2t**2)
        if p2 <= 0:
            continue
        cs2, vax2, cf2 = _speeds(r, p2, x, B2t**2, g)
        u2 = 1.0 / r
        ok = (
            (branch == "fast" and vax2 * (1 - 1e-9) < u2 < cf2 * (1 + 1e-9))
            or (branch == "slow" and u2 < cs2 * (1 + 1e-9))
        )
        if not ok:
            continue
        if best is None or r > best[0]:
            best = (r, B2t)
    return best if best is not None else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# public solvers
# ---------------------------------------------------------------------------
def solve_shock_array(MA, beta, theta_deg, gamma=5 / 3, branch="fast"):
    """Vectorised solver.  Inputs broadcast against each other; returns a dict
    with one array per ``Shock`` field.  Downstream fields are NaN where no
    shock of the requested branch exists; upstream fields (MA, beta, Ms1, Mf1,
    theta_deg, gamma) are always filled.

    theta = 0 and theta = 90 use closed forms (fast); other angles loop over
    the polynomial solver.

    The extra boolean key ``switch_on`` is True where the solution is a
    theta = 0 switch-on shock (r = M_A^2, tangential field created).  It is
    False for the gas-dynamic parallel shock and for every theta > 0: there
    the fast shock merely approaches switch-on behaviour continuously as
    theta -> 0, with no sharp boundary.
    """
    if branch not in ("fast", "slow"):
        raise ValueError("branch must be 'fast' or 'slow'")
    MA, beta, th, g = (np.array(a, dtype=float) for a in
                       np.broadcast_arrays(MA, beta, theta_deg, gamma))
    th = np.abs(th) % 180.0
    th = np.where(th > 90.0, 180.0 - th, th)          # fold into [0, 90]

    r = np.full(MA.shape, np.nan)
    B2t = np.full(MA.shape, np.nan)

    par = th < _THETA_TOL_DEG
    perp = np.abs(th - 90.0) < _THETA_TOL_DEG
    obl = ~(par | perp)

    for mask, fn in ((par, _parallel), (perp, _perpendicular)):
        if mask.any():
            rr, bb = fn(MA[mask], beta[mask], g[mask], branch)
            r[mask], B2t[mask] = rr, bb
            th[mask] = 0.0 if fn is _parallel else 90.0
    for idx in np.ndindex(MA.shape):
        if not obl[idx]:
            continue
        r[idx], B2t[idx] = _oblique(MA[idx], beta[idx], th[idx], g[idx], branch)

    out = _downstream_state(MA, beta, np.radians(th), g, r, B2t)
    out = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    with np.errstate(invalid="ignore"):
        out["switch_on"] = par & np.isfinite(r) & (np.nan_to_num(B2t) > 0.0)
    return out


def solve_shock(MA, beta, theta_deg, gamma=5 / 3, branch="fast"):
    """Return a Shock for the requested branch, or None if no such shock exists
    (e.g. flow not super-fast-magnetosonic for branch='fast')."""
    s = solve_shock_array(MA, beta, theta_deg, gamma, branch)
    if not np.isfinite(s["r"]):
        return None
    return Shock(branch=branch, switch_on=bool(s["switch_on"]),
                 **{k: float(s[k]) for k in _FLOAT_FIELDS})


def first_critical_Mf(beta, theta_deg, gamma=5 / 3, Mf_max=12.0, n=240):
    """First critical fast Mach number (Edmiston & Kennel 1984): the smallest Mf
    at which the downstream normal flow speed equals the downstream sound speed
    (gamma p2 r = 1 in code units).  NaN if none exists below Mf_max."""
    def f(Mf):
        s = solve_shock(MA_from_Mf(Mf, beta, theta_deg, gamma), beta, theta_deg, gamma, "fast")
        if s is None:
            return np.nan
        return s.Ms2 ** -2 - 1.0  # >0 when downstream flow is subsonic

    grid = np.linspace(1.0 + 1e-3, Mf_max, n)
    vals = np.array([f(m) for m in grid])
    first = vals[np.isfinite(vals)]
    if first.size and first[0] > 0.0:
        return 1.0   # supercritical from the weak-shock limit: M_f* = 1
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a <= 0.0 < b:
            return brentq(f, grid[i], grid[i + 1], xtol=1e-6)
    return np.nan


def shock_angle_from_deflection(M0, beta, phi_deg, gamma=5 / 3, mhd=True):
    """Weak-branch oblique shock angle sigma (to the upstream flow) for a given
    flow deflection phi, with the upstream B perpendicular to the plane of
    flow and shock normal (out-of-plane field, theta_Bn = 90 deg).
    M0 is the upstream SONIC Mach number of the full flow speed.
    Relation: tan(sigma - phi) = tan(sigma)/r(M0 sin sigma)."""
    phi = np.radians(phi_deg)

    def comp(sig):
        Ms_n = M0 * np.sin(sig)
        if not mhd:
            return ((gamma + 1) * Ms_n**2) / ((gamma - 1) * Ms_n**2 + 2.0)
        MA_n = Ms_n * np.sqrt(0.5 * gamma * beta)
        s = solve_shock(MA_n, beta, 90.0, gamma, "fast")
        return np.nan if s is None else s.r

    def g(sig):
        r = comp(sig)
        return np.nan if not np.isfinite(r) else (sig - np.arctan(np.tan(sig) / r)) - phi

    # weak branch: smallest sigma at which the deflection reaches phi
    sig0 = np.arcsin(1.0 / M0) if not mhd else np.arcsin(
        min(1.0, 1.0 / Mf_perp_unit(M0, beta, gamma)))
    grid = np.linspace(sig0 + 1e-6, np.pi / 2, 400)
    vals = np.array([g(s) for s in grid])
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0 and a <= 0:
            return np.degrees(brentq(g, grid[i], grid[i + 1], xtol=1e-9))
    return np.nan


def Mf_perp_unit(M0, beta, gamma=5 / 3):
    """Upstream Mach number of the full flow w.r.t. the perpendicular fast speed."""
    h = 0.5 * gamma * beta          # a^2 / vA^2
    return M0 * np.sqrt(h / (1.0 + h))   # u / sqrt(a^2 + vA^2)
