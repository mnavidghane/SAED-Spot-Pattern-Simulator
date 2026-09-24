# ==================================================================================================
# SAED SIMULATOR (FULL PROJECT VERSION, FIXED REF SELECTION)
# ==================================================================================================
# Fixes in this version:
#   - Reference selection now strictly enforces: ref2 is NOT collinear with ref1 and ref2 != ±ref1
#   - After any flip/swap for handedness/orientation, we re-check non-collinearity
#   - Center label is exactly "(000)" (no stray characters)
# ==================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.lines import Line2D


# ==================================================================================================
# SECTION A — BRAVAIS LATTICES (14)
# ==================================================================================================

BRAVAIS_14 = [
    ("Cubic P",        "cubic",       "P"),
    ("Cubic I (BCC)",  "cubic",       "I"),
    ("Cubic F (FCC)",  "cubic",       "F"),
    ("Tetragonal P",   "tetragonal",  "P"),
    ("Tetragonal I",   "tetragonal",  "I"),
    ("Orthorhombic P", "orthorhombic","P"),
    ("Orthorhombic C", "orthorhombic","C"),
    ("Orthorhombic I", "orthorhombic","I"),
    ("Orthorhombic F", "orthorhombic","F"),
    ("Hexagonal P",    "hexagonal",   "P"),
    ("Trigonal R",     "trigonal",    "R"),
    ("Monoclinic P",   "monoclinic",  "P"),
    ("Monoclinic C",   "monoclinic",  "C"),
    ("Triclinic P",    "triclinic",   "P"),
]


# ==================================================================================================
# SECTION B — USER-FACING CONSTANTS (GRAPHICS + INTENSITY)
# ==================================================================================================

ZONE_DAMPING = {
    "ZOLZ": 1.00,
    "FOLZ": 0.30,
    "SOLZ": 0.15,
    "HOLZ": 0.05,
}

ZONE_LABEL_COLORS = {
    "ZOLZ": "white",
    "FOLZ": "lime",
    "SOLZ": "yellow",
    "HOLZ": "orange",
}

REF_LABEL_COLOR_1 = "cyan"
REF_LABEL_COLOR_2 = "magenta"

DEFAULT_FIGSIZE = (8.0, 8.0)
DEFAULT_SPOT_SIZE = 70
DEFAULT_REF_LABEL_SIZE = 11
DEFAULT_HKL_LABEL_SIZE = 7
DEFAULT_CENTER_SIZE = 260

DEFAULT_GAMMA = 0.60

MIN_ALPHA = 0.10
MAX_ALPHA = 1.00


# ==================================================================================================
# SECTION C — GENERAL HELPERS / IO
# ==================================================================================================

def deg2rad(x: float) -> float:
    return x * np.pi / 180.0


def norm(v) -> float:
    return float(np.linalg.norm(v))


def unit(v):
    n = norm(v)
    if n < 1e-15:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def hkl_str(hkl) -> str:
    h, k, l = (int(x) for x in hkl)
    return f"({h} {k} {l})"


def prompt_float(msg: str, default=None) -> float:
    while True:
        s = input(msg).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("عدد معتبر وارد کنید.")


def prompt_int(msg: str, default=None) -> int:
    while True:
        s = input(msg).strip()
        if s == "" and default is not None:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("عدد صحیح معتبر وارد کنید.")


def prompt_yesno(msg: str, default_yes: bool = True) -> bool:
    s = input(msg).strip().lower()
    if s == "":
        return default_yes
    if s in ("y", "yes", "1", "true", "t"):
        return True
    if s in ("n", "no", "0", "false", "f"):
        return False
    return default_yes


# ==================================================================================================
# SECTION D — ELECTRON WAVELENGTH (relativistic), lambda in Å
# ==================================================================================================

def electron_wavelength_angstrom(kV: float) -> float:
    """
    Relativistic electron wavelength (approx)
    lambda(Å) = 12.3986 / sqrt(V*(1 + 0.97845e-6 * V)), V in volts
    """
    if kV <= 0:
        raise ValueError(f"Accelerating voltage must be > 0 kV (got {kV}).")
    V = float(kV) * 1000.0
    return float(12.3986 / np.sqrt(V * (1.0 + 0.97845e-6 * V)))


# ==================================================================================================
# SECTION E — CENTERING EXTINCTION RULES
# ==================================================================================================

def allowed_by_centering(h: int, k: int, l: int, centering: str) -> bool:
    centering = centering.upper()
    if centering == "P":
        return True
    if centering == "I":   # BCC
        return ((h + k + l) % 2 == 0)
    if centering == "F":   # FCC
        return ((h % 2 == k % 2) and (k % 2 == l % 2))
    if centering == "C":   # C-centered (base-centered)
        return ((h + k) % 2 == 0)
    if centering == "R":   # Rhombohedral (hex setting)
        return ((-h + k + l) % 3 == 0)
    if centering == "R_RHOMBO":  # Rhombohedral, primitive rhombohedral axes
        return True               # no systematic absences in this basis
    raise ValueError(f"Unknown centering: {centering}")


# ==================================================================================================
# SECTION F — LATTICE CONTAINER
# ==================================================================================================

@dataclass
class Lattice:
    system: str
    centering: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


# ==================================================================================================
# SECTION G — DIRECT & RECIPROCAL BASIS (reciprocal uses 2π)
# ==================================================================================================

def direct_basis_from_params(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = deg2rad(alpha_deg)
    beta  = deg2rad(beta_deg)
    gamma = deg2rad(gamma_deg)

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    if abs(np.sin(gamma)) < 1e-9:
        raise ValueError(f"gamma={gamma_deg:g} deg is too close to 0/180 deg; cell is degenerate.")
    vol_factor_sq = 1.0 - ca*ca - cb*cb - cg*cg + 2.0*ca*cb*cg
    if vol_factor_sq <= 0.0:
        raise ValueError(
            f"alpha={alpha_deg:g}, beta={beta_deg:g}, gamma={gamma_deg:g} deg do not form a valid "
            f"unit cell (1-cos^2(a)-cos^2(b)-cos^2(g)+2cos(a)cos(b)cos(g) = {vol_factor_sq:.4g} <= 0)."
        )

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0], dtype=float)

    cx = c*np.cos(beta)
    cy = c*(np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c*c - cx*cx - cy*cy, 0.0))  # >=0 guaranteed by the check above, up to fp noise
    c_vec = np.array([cx, cy, cz], dtype=float)

    return a_vec, b_vec, c_vec


def reciprocal_basis(a_vec, b_vec, c_vec):
    V = float(np.dot(a_vec, np.cross(b_vec, c_vec)))
    if abs(V) < 1e-15:
        raise ValueError("Cell volume is ~0. Check lattice parameters.")
    a_star = 2*np.pi * np.cross(b_vec, c_vec) / V
    b_star = 2*np.pi * np.cross(c_vec, a_vec) / V
    c_star = 2*np.pi * np.cross(a_vec, b_vec) / V
    return a_star, b_star, c_star


def g_vector_cart(h, k, l, a_star, b_star, c_star):
    return h*a_star + k*b_star + l*c_star


def d_spacing(h, k, l, a_star, b_star, c_star):
    g = g_vector_cart(h, k, l, a_star, b_star, c_star)
    gnorm = norm(g)
    if gnorm < 1e-15:
        return np.inf
    return 2*np.pi / gnorm


# ==================================================================================================
# SECTION H — ANGLES BETWEEN REFLECTIONS
# ==================================================================================================

def angle_between_reflections(h1, k1, l1, h2, k2, l2, a_star, b_star, c_star):
    g1 = g_vector_cart(h1, k1, l1, a_star, b_star, c_star)
    g2 = g_vector_cart(h2, k2, l2, a_star, b_star, c_star)
    n1 = norm(g1)
    n2 = norm(g2)
    if n1 < 1e-15 or n2 < 1e-15:
        return np.nan
    c = float(np.dot(g1, g2) / (n1*n2))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# ==================================================================================================
# SECTION I — ZONE GEOMETRY (plane axes)
# ==================================================================================================

def zone_real_direction(u, v, w, a_vec, b_vec, c_vec):
    return u*a_vec + v*b_vec + w*c_vec


def make_plane_axes(r_hat):
    tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(unit(tmp), r_hat))) > 0.95:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)

    x_hat = tmp - float(np.dot(tmp, r_hat))*r_hat
    x_hat = unit(x_hat)
    y_hat = unit(np.cross(r_hat, x_hat))
    return x_hat, y_hat


def g_perp(g, r_hat):
    return g - float(np.dot(g, r_hat))*r_hat


def to_plane_xy(v, x_hat, y_hat):
    return np.array([float(np.dot(v, x_hat)), float(np.dot(v, y_hat))], dtype=float)


def rotate_2d(coords_xy, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],[s, c]], dtype=float)
    return coords_xy @ R.T


# ==================================================================================================
# SECTION J — EWALD EXCITATION ERROR + THICKNESS INTENSITY
# ==================================================================================================

def excitation_error_s(g, r_hat, k_ewald):
    """
    s ≈ g_parallel + |g|^2/(2k)
    k = 1/lambda
    """
    g_par = float(np.dot(g, r_hat))
    g2 = float(np.dot(g, g))
    return float(g_par + g2/(2.0*k_ewald))


def intensity_sinc2(s, t_angstrom):
    x = float(s) * float(t_angstrom)
    val = float(np.sinc(x))
    return float(val*val)


# ==================================================================================================
# SECTION K — LAUE ZONES (selection)
# ==================================================================================================

def parse_zone_selection(s: str):
    t = s.strip().upper().replace(" ", "")
    if t in ("", "ZOLZ"):
        return {"ZOLZ"}
    if t in ("ALL", "ZOLZ+FOLZ+SOLZ+HOLZ"):
        return {"ZOLZ","FOLZ","SOLZ","HOLZ"}
    parts = [p for p in t.split(",") if p]
    allowed = {"ZOLZ","FOLZ","SOLZ","HOLZ"}
    out = set()
    for p in parts:
        if p not in allowed:
            raise ValueError("Zone selection: ZOLZ | FOLZ | SOLZ | HOLZ | ZOLZ,FOLZ | ALL")
        out.add(p)
    return out


def laue_zone_label(n: int) -> str:
    an = abs(int(n))
    if an == 0: return "ZOLZ"
    if an == 1: return "FOLZ"
    if an == 2: return "SOLZ"
    return "HOLZ"


def inferred_Nmax_from_selection(selected_zones):
    if selected_zones == {"ZOLZ"}: return 0
    if selected_zones == {"FOLZ"}: return 1
    if selected_zones == {"SOLZ"}: return 2
    return None


# ==================================================================================================
# SECTION L — LATTICE INPUT (14 lattices)
# ==================================================================================================

def build_lattice_from_selection():
    print("\n--- Select Bravais lattice (14) ---")
    for i, (name, sys, cent) in enumerate(BRAVAIS_14, 1):
        print(f"{i:2d}) {name:16s}   [system={sys}, centering={cent}]")
    idx = prompt_int("Enter choice number (1-14): ")
    if not (1 <= idx <= 14):
        raise ValueError("Choice out of range.")
    name, system, centering = BRAVAIS_14[idx-1]
    system = system.lower()
    centering = centering.upper()

    print(f"\nChosen: {name}  -> system={system}, centering={centering}")
    print("Units: a,b,c in Å. Angles in degrees.\n")

    if system == "cubic":
        a = prompt_float("Enter a (Å): ")
        b = a; c = a
        alpha = beta = gamma = 90.0

    elif system == "tetragonal":
        a = prompt_float("Enter a (Å): ")
        c = prompt_float("Enter c (Å): ")
        b = a
        alpha = beta = gamma = 90.0

    elif system == "orthorhombic":
        a = prompt_float("Enter a (Å): ")
        b = prompt_float("Enter b (Å): ")
        c = prompt_float("Enter c (Å): ")
        alpha = beta = gamma = 90.0

    elif system == "hexagonal":
        a = prompt_float("Enter a (Å): ")
        c = prompt_float("Enter c (Å): ")
        b = a
        alpha = beta = 90.0
        gamma = 120.0

    elif system == "trigonal":
        mode = input("Trigonal setting? (1=hex a,c ; 2=rhombo a,alpha): ").strip() or "1"
        if mode == "2":
            a = prompt_float("Enter a (Å): ")
            alpha = prompt_float("Enter alpha=beta=gamma (deg): ")
            b = c = a
            beta = gamma = alpha
            centering = "R_RHOMBO"   # primitive rhombohedral axes: no systematic absences
        else:
            a = prompt_float("Enter a (Å): ")
            c = prompt_float("Enter c (Å): ")
            b = a
            alpha = beta = 90.0
            gamma = 120.0
            # centering stays "R" (hexagonal/obverse setting)

    elif system == "monoclinic":
        a = prompt_float("Enter a (Å): ")
        b = prompt_float("Enter b (Å): ")
        c = prompt_float("Enter c (Å): ")
        alpha = 90.0
        gamma = 90.0
        beta = prompt_float("Enter beta (deg): ")

    elif system == "triclinic":
        a = prompt_float("Enter a (Å): ")
        b = prompt_float("Enter b (Å): ")
        c = prompt_float("Enter c (Å): ")
        alpha = prompt_float("Enter alpha (deg): ")
        beta  = prompt_float("Enter beta (deg): ")
        gamma = prompt_float("Enter gamma (deg): ")

    else:
        raise ValueError("Unknown crystal system.")

    return Lattice(system=system, centering=centering, a=a, b=b, c=c,
                   alpha=alpha, beta=beta, gamma=gamma)


# ==================================================================================================
# SECTION M — DETECTOR COORDINATES (mm): R = K / d
# ==================================================================================================

def detector_xy_mm_for_hkl(h, k, l,
                           a_star, b_star, c_star,
                           r_hat, x_hat, y_hat,
                           K_mmA):
    g = g_vector_cart(h, k, l, a_star, b_star, c_star)
    d = d_spacing(h, k, l, a_star, b_star, c_star)
    R = float(K_mmA) / float(d)

    gp = g_perp(g, r_hat)
    xy = to_plane_xy(gp, x_hat, y_hat)
    n = norm(xy)
    if n < 1e-15:
        return 0.0, 0.0, d, R

    uxy = xy / n
    return float(R*uxy[0]), float(R*uxy[1]), float(d), float(R)


# ==================================================================================================
# SECTION N — FIND HKL INDEX IN ARRAY
# ==================================================================================================

def find_index_of_hkl(hkls_arr, target_hkl):
    target = np.array(target_hkl, dtype=int).reshape(3,)
    for i in range(hkls_arr.shape[0]):
        if np.array_equal(hkls_arr[i], target):
            return int(i)
    return None


# ==================================================================================================
# SECTION O — REFERENCE SPOT SELECTION (ROBUST: NEVER ±, NEVER COLLINEAR, TIE-SAFE)
# ==================================================================================================

def _is_same_or_friedel(hkl_a, hkl_b) -> bool:
    a = np.array(hkl_a, dtype=int)
    b = np.array(hkl_b, dtype=int)
    return np.array_equal(a, b) or np.array_equal(a, -b)

def _cross2d(v1, v2) -> float:
    return float(v1[0]*v2[1] - v1[1]*v2[0])

def _noncollinear_2d(v1, v2, eps=1e-10) -> bool:
    return abs(_cross2d(v1, v2)) > eps

def choose_reference_spots_from_clean_zolz(
    u, v, w,
    centering,
    a_star, b_star, c_star,
    r_hat, x_hat, y_hat,
    hkl_cap=12,
    gperp_tol=1e-10
):
    """
    Robust reference selection from CLEAN ZOLZ (n=0), independent of Ewald/intensity.

    Guarantees:
      - ref2 != ±ref1 (always)
      - ref1 and ref2 are non-collinear in diffraction plane (always)
      - enforces (g1 x g2)·zone > 0 by SWAP only (never by flipping to -ref2 blindly)
      - after aligning ref1 to +x, prefers ref2 with y>0; if not possible, picks a different ref2
    """

    # ----------------------------
    # Build clean ZOLZ pool
    # ----------------------------
    hkls = []
    g_list = []
    gp_xy = []
    gnorms = []

    for h in range(-hkl_cap, hkl_cap+1):
        for k in range(-hkl_cap, hkl_cap+1):
            for l in range(-hkl_cap, hkl_cap+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed_by_centering(h, k, l, centering):
                    continue
                n = h*u + k*v + l*w
                if n != 0:
                    continue  # ZOLZ only

                g = g_vector_cart(h, k, l, a_star, b_star, c_star)
                gp = g_perp(g, r_hat)
                xy = to_plane_xy(gp, x_hat, y_hat)
                if norm(xy) <= gperp_tol:
                    continue

                hkls.append((h, k, l))
                g_list.append(g)
                gp_xy.append(xy)
                gnorms.append(norm(g))

    if len(hkls) < 3:
        raise ValueError("Not enough ZOLZ reflections for robust reference selection. Increase hkl_cap.")

    hkls  = np.array(hkls, dtype=int)
    g_list = np.array(g_list, dtype=float)
    gp_xy = np.array(gp_xy, dtype=float)
    gnorms = np.array(gnorms, dtype=float)

    order = np.argsort(gnorms)

    def triple(gA, gB):
        return float(np.dot(np.cross(gA, gB), r_hat))

    # ----------------------------
    # Pick ref1: smallest |g|
    # ----------------------------
    idx1 = int(order[0])
    ref1 = hkls[idx1]
    g1 = g_list[idx1]
    v1 = gp_xy[idx1]

    # ----------------------------
    # Pick ref2 robustly:
    #  - among low-|g| candidates, pick the one that is:
    #      * not ±ref1
    #      * non-collinear
    #      * has the largest |cross2d| with v1 (best spread)
    #  - also prefer y>0 after aligning ref1 to +x
    # ----------------------------
    theta = float(np.arctan2(v1[1], v1[0]))  # rotation to place ref1 on +x

    best = None  # (shell_gnorm, -abs(cross), prefer_y, idx2)
    for j in order[1:]:
        # NEW: force ref2 to come from a different |g|-shell (book-like)
        if gnorms[j] <= 1.05 * gnorms[idx1]:
            continue
        j = int(j)
        cand = hkls[j]
        if _is_same_or_friedel(cand, ref1):
            continue
        v2 = gp_xy[j]
        if not _noncollinear_2d(v1, v2, eps=1e-10):
            continue

        # score: first minimize gnorm, then maximize |cross|, then prefer y>0 after rotation
        v2_rot = rotate_2d(v2.reshape(1,2), -theta)[0]
        prefer_y = 1 if (v2_rot[1] > 0) else 0

        # NEW: penalize axis-type reflections like (±m,0,0) or (0,±m,0) or (0,0,±m)
        h, k, l = (int(x) for x in hkls[j])
        axis_penalty = 1 if ((h != 0) + (k != 0) + (l != 0) == 1) else 0  # 1 if only one nonzero index

        score = (gnorms[j], axis_penalty, -abs(_cross2d(v1, v2)), -prefer_y, j)


        if best is None or score < best:
            best = score

    if best is None:
        raise ValueError("Could not find a valid non-collinear Spot2 (not ±Spot1). Increase hkl_cap.")

    idx2 = int(best[4])
    ref2 = hkls[idx2]
    g2 = g_list[idx2]
    v2 = gp_xy[idx2]

    # ----------------------------
    # Enforce handedness by SWAP only (safe if non-collinear):
    #   Want (g1 x g2)·zone > 0
    # ----------------------------
    if triple(g1, g2) < 0:
        # swap
        idx1, idx2 = idx2, idx1
        ref1, ref2 = ref2, ref1
        g1, g2 = g2, g1
        v1, v2 = v2, v1
        theta = float(np.arctan2(v1[1], v1[0]))

    # ----------------------------
    # Final hard guarantees
    # ----------------------------
    if _is_same_or_friedel(ref2, ref1):
        raise RuntimeError("Internal error: ref2 became ±ref1. This should be impossible now.")
    if not _noncollinear_2d(v1, v2, eps=1e-10):
        raise RuntimeError("Internal error: ref1/ref2 became collinear. This should be impossible now.")
    tp = triple(g1, g2)
    if tp <= 0:
        raise RuntimeError("Internal error: triple product not positive after swap. This should be impossible now.")

    return ref1, ref2, float(tp)


# ==================================================================================================
# SECTION P — REFLECTION GENERATION (NO HARD s_max; intensity-based cutoff after normalization)
# ==================================================================================================

def generate_reflections_all(
    u, v, w,
    centering,
    selected_zones, Nmax,
    a_star, b_star, c_star,
    r_hat,
    apply_ewald: bool,
    k_ewald: float,
    t_angstrom: float,
    hkl_max: int,
):
    out = []
    for h in range(-hkl_max, hkl_max+1):
        for k in range(-hkl_max, hkl_max+1):
            for l in range(-hkl_max, hkl_max+1):
                if h == 0 and k == 0 and l == 0:
                    continue

                if not allowed_by_centering(h, k, l, centering):
                    continue

                n = h*u + k*v + l*w
                if abs(n) > Nmax:
                    continue

                zlab = laue_zone_label(n)
                if zlab not in selected_zones:
                    continue

                g = g_vector_cart(h, k, l, a_star, b_star, c_star)

                if apply_ewald:
                    s = excitation_error_s(g, r_hat, k_ewald)
                else:
                    s = 0.0

                I_geom = intensity_sinc2(s, t_angstrom)
                zone_w = float(ZONE_DAMPING.get(zlab, 1.0))
                I_raw = float(I_geom * zone_w)

                out.append({
                    "h": int(h),
                    "k": int(k),
                    "l": int(l),
                    "n": int(n),
                    "zone": zlab,
                    "s": float(s),
                    "I_raw": float(I_raw),
                    "g": g,
                })
    return out


# ==================================================================================================
# SECTION Q — NORMALIZE INTENSITIES + APPLY CUTOFF
# ==================================================================================================

def normalize_and_filter_by_intensity(reflections, Imin: float, gamma: float):
    if len(reflections) == 0:
        return [], 0.0

    Imax = max(r["I_raw"] for r in reflections)
    if Imax <= 1e-30:
        return [], Imax

    kept = []
    for r in reflections:
        I_norm = float(r["I_raw"] / Imax)
        if I_norm < float(Imin):
            continue
        whiteness = float(np.clip(I_norm, 0.0, 1.0) ** float(gamma))
        alpha = float(np.clip(whiteness, MIN_ALPHA, MAX_ALPHA))

        rr = dict(r)
        rr["I_norm"] = I_norm
        rr["whiteness"] = whiteness
        rr["alpha"] = alpha
        kept.append(rr)

    return kept, Imax


# ==================================================================================================
# SECTION R — BUILD DETECTOR COORDS FOR ALL KEPT REFLECTIONS
# ==================================================================================================

def build_detector_coords(reflections_kept,
                          a_star, b_star, c_star,
                          r_hat, x_hat, y_hat,
                          K_mmA):
    out = []
    for r in reflections_kept:
        h = r["h"]; k = r["k"]; l = r["l"]
        x_mm, y_mm, d, R = detector_xy_mm_for_hkl(h, k, l,
                                                  a_star, b_star, c_star,
                                                  r_hat, x_hat, y_hat,
                                                  K_mmA)
        rr = dict(r)
        rr["x_mm"] = float(x_mm)
        rr["y_mm"] = float(y_mm)
        rr["d"] = float(d)
        rr["R"] = float(R)
        out.append(rr)
    return out


# ==================================================================================================
# SECTION S — LEGEND (EXPLAIN COLORS)
# ==================================================================================================

def add_legend(ax):
    handles = [
        Line2D([0],[0], color="white",  lw=0, marker='o', markersize=7, label="ZOLZ label"),
        Line2D([0],[0], color="lime",   lw=0, marker='o', markersize=7, label="FOLZ label"),
        Line2D([0],[0], color="yellow", lw=0, marker='o', markersize=7, label="SOLZ label"),
        Line2D([0],[0], color="orange", lw=0, marker='o', markersize=7, label="HOLZ label"),
        Line2D([0],[0], color=REF_LABEL_COLOR_1, lw=0, marker='o', markersize=7, label="Reference 1 label"),
        Line2D([0],[0], color=REF_LABEL_COLOR_2, lw=0, marker='o', markersize=7, label="Reference 2 label"),
    ]
    ax.legend(handles=handles,
              loc="upper right",
              frameon=False,
              fontsize=9,
              labelcolor="white")


# ==================================================================================================
# SECTION T — MAIN PLOT (BOOK-LIKE)
# ==================================================================================================

def plot_saed_booklike(reflections_det,
                       ref1, ref2,
                       idx_ref1, idx_ref2,
                       theta_align,
                       title_str: str,
                       show_hkl_labels: bool = True):

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # rotate all points by -theta_align (so ref1 on +x)
    for r in reflections_det:
        xy = np.array([r["x_mm"], r["y_mm"]], dtype=float)
        xy_rot = rotate_2d(xy.reshape(1,2), -theta_align)[0]
        r["x_plot"] = float(xy_rot[0])
        r["y_plot"] = float(xy_rot[1])

    # plot spots
    for r in reflections_det:
        w = float(np.clip(r["whiteness"], 0.0, 1.0))
        col = (w, w, w)
        ax.scatter(r["x_plot"], r["y_plot"],
                   s=DEFAULT_SPOT_SIZE,
                   color=col,
                   alpha=float(r["alpha"]),
                   zorder=3)

        if show_hkl_labels:
            zone = r["zone"]
            tc = ZONE_LABEL_COLORS.get(zone, "white")
            ax.text(r["x_plot"], r["y_plot"],
                    " " + hkl_str((r["h"], r["k"], r["l"])),
                    color=tc,
                    fontsize=DEFAULT_HKL_LABEL_SIZE,
                    ha="left",
                    va="bottom",
                    zorder=4)

    # center 000 (FIXED)
    ax.scatter(0, 0, s=DEFAULT_CENTER_SIZE, color="white", zorder=6)
    ax.text(0, 0, "(000)", color="white", fontsize=10,
            ha="right", va="bottom", zorder=7)

    # reference labels (only label color different)
    if idx_ref1 is not None:
        r1 = reflections_det[idx_ref1]
        ax.text(r1["x_plot"], r1["y_plot"],
                " " + hkl_str(ref1),
                color=REF_LABEL_COLOR_1,
                fontsize=DEFAULT_REF_LABEL_SIZE,
                fontweight="bold",
                ha="left", va="bottom",
                zorder=8)

    if idx_ref2 is not None:
        r2 = reflections_det[idx_ref2]
        ax.text(r2["x_plot"], r2["y_plot"],
                " " + hkl_str(ref2),
                color=REF_LABEL_COLOR_2,
                fontsize=DEFAULT_REF_LABEL_SIZE,
                fontweight="bold",
                ha="left", va="bottom",
                zorder=8)

    # cosmetics
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_str, color="white", fontsize=12)

    add_legend(ax)

    # set limits
    if len(reflections_det) > 0:
        rmax = max(np.hypot(r["x_plot"], r["y_plot"]) for r in reflections_det)
        rmax = max(rmax, 1.0)
    else:
        rmax = 1.0
    ax.set_xlim(-1.1*rmax, 1.1*rmax)
    ax.set_ylim(-1.1*rmax, 1.1*rmax)

    plt.show()


# ==================================================================================================
# SECTION U — MAIN PROGRAM
# ==================================================================================================

def main():
    # (1) lattice input
    lat = build_lattice_from_selection()

    # (2) zone axis
    print("\n--- Zone axis [u v w] (integers) ---")
    u = prompt_int("u: ")
    v = prompt_int("v: ")
    w = prompt_int("w: ")
    if u == 0 and v == 0 and w == 0:
        raise ValueError("Zone axis cannot be [0 0 0].")

    # (3) TEM voltage
    print("\n--- TEM voltage (kV) ---")
    kV = prompt_float("Enter TEM voltage (kV), e.g. 200: ", default=200.0)
    lamA = electron_wavelength_angstrom(kV)
    k_ewald = 2.0 * np.pi / lamA   # must match the 2*pi convention used by reciprocal_basis()

    # (4) Camera constant
    print("\n--- Camera constant ---")
    print("Units: K = L·λ in mm·Å")
    K = prompt_float("Enter K (mm·Å): ")
    if K <= 0:
        raise ValueError(f"Camera constant K must be > 0 mm*Å (got {K}).")

    # (5) Laue zone selection + Nmax
    print("\n--- Laue zone selection ---")
    print("Examples: ZOLZ | FOLZ | SOLZ | HOLZ | ZOLZ,FOLZ | ALL")
    zsel_in = input("Which zones to show? (default ZOLZ): ").strip()
    selected_zones = parse_zone_selection(zsel_in if zsel_in else "ZOLZ")

    Nmax_auto = inferred_Nmax_from_selection(selected_zones)
    if Nmax_auto is None:
        Nmax = prompt_int("Nmax (max |hu+kv+lw|, e.g. 2/3/4): ", default=2)
    else:
        Nmax = Nmax_auto
        print(f"Auto Nmax set to {Nmax} based on selection {sorted(list(selected_zones))}")

    # (6) Ewald toggle
    print("\n--- Ewald sphere ---")
    apply_ewald = prompt_yesno("Apply Ewald sphere (affects s, thus intensity)? (y/n, default y): ", default_yes=True)

    # (7) thickness + intensity cutoff + gamma
    print("\n--- Intensity model (thickness-based sinc^2) ---")
    t_nm = prompt_float("Thickness t (nm, default 50): ", default=50.0)
    if t_nm <= 0:
        raise ValueError(f"Thickness must be > 0 nm (got {t_nm}).")
    t_angstrom = float(t_nm) * 10.0

    Imin = prompt_float("Intensity cutoff I_min (0..1, default 0.02): ", default=0.02)
    gamma = prompt_float(f"Contrast gamma (default {DEFAULT_GAMMA}): ", default=DEFAULT_GAMMA)

    print("\n--- Reflection search bounds ---")
    hkl_max = prompt_int("hkl_max (default 15): ", default=15)

    # (8) Build bases
    a_vec, b_vec, c_vec = direct_basis_from_params(lat.a, lat.b, lat.c,
                                                   lat.alpha, lat.beta, lat.gamma)
    a_star, b_star, c_star = reciprocal_basis(a_vec, b_vec, c_vec)

    r_zone = zone_real_direction(u, v, w, a_vec, b_vec, c_vec)
    r_hat = unit(r_zone)
    x_hat, y_hat = make_plane_axes(r_hat)

    # (9) Choose reference spots from clean ZOLZ (FIXED)
    ref1, ref2, triple = choose_reference_spots_from_clean_zolz(
        u, v, w,
        lat.centering,
        a_star, b_star, c_star,
        r_hat, x_hat, y_hat,
        hkl_cap=max(12, min(30, hkl_max)),
        gperp_tol=1e-10
    )

    # (10) Compute theta alignment from ref1 (detector plane)
    g1 = g_vector_cart(int(ref1[0]), int(ref1[1]), int(ref1[2]), a_star, b_star, c_star)
    gp1 = g_perp(g1, r_hat)
    xy1 = to_plane_xy(gp1, x_hat, y_hat)
    theta_align = float(np.arctan2(xy1[1], xy1[0]))

    # (11) Generate reflections (no hard s_max), normalize + filter by intensity
    reflections = generate_reflections_all(
        u, v, w,
        lat.centering,
        selected_zones, Nmax,
        a_star, b_star, c_star,
        r_hat,
        apply_ewald=apply_ewald,
        k_ewald=k_ewald,
        t_angstrom=t_angstrom,
        hkl_max=hkl_max,
    )

    reflections_kept, Imax = normalize_and_filter_by_intensity(reflections, Imin=Imin, gamma=gamma)

    if len(reflections_kept) == 0:
        raise ValueError("No reflections above intensity cutoff. Decrease I_min or adjust parameters.")

    # (12) Detector coords for kept reflections
    reflections_det = build_detector_coords(reflections_kept,
                                            a_star, b_star, c_star,
                                            r_hat, x_hat, y_hat,
                                            K_mmA=K)

    # (13) Find ref indices among plotted set (may not be present if below cutoff)
    hkls_arr = np.array([(r["h"], r["k"], r["l"]) for r in reflections_det], dtype=int)
    idx_ref1 = find_index_of_hkl(hkls_arr, np.array(ref1, dtype=int))
    idx_ref2 = find_index_of_hkl(hkls_arr, np.array(ref2, dtype=int))

    # (14) Summary output
    h1, k1, l1 = (int(x) for x in ref1)
    h2, k2, l2 = (int(x) for x in ref2)
    ang = angle_between_reflections(h1, k1, l1, h2, k2, l2, a_star, b_star, c_star)

    s_ref1 = excitation_error_s(g1, r_hat, k_ewald) if apply_ewald else 0.0
    g2 = g_vector_cart(h2, k2, l2, a_star, b_star, c_star)
    s_ref2 = excitation_error_s(g2, r_hat, k_ewald) if apply_ewald else 0.0

    print("\n" + "="*100)
    print("SAED SUMMARY (book-like) | Units: Å, mm, Å^-1")
    print("="*100)
    print(f"System={lat.system}   Centering={lat.centering}")
    print(f"Cell: a={lat.a:g} Å  b={lat.b:g} Å  c={lat.c:g} Å  alpha={lat.alpha:g}  beta={lat.beta:g}  gamma={lat.gamma:g}")
    print(f"Zone axis=[{u} {v} {w}]   Zones={sorted(list(selected_zones))}   Nmax={Nmax}   hkl_max={hkl_max}")
    print(f"Voltage={kV:g} kV   lambda={lamA:.6g} Å   k=1/lambda={k_ewald:.6g} Å^-1")
    print(f"Ewald applied: {apply_ewald}")
    print(f"Thickness t={t_nm:g} nm ({t_angstrom:g} Å)")
    print(f"Intensity: I_raw = sinc^2(s*t) * zone_damping")
    print(f"Zone damping: {ZONE_DAMPING}")
    print(f"I_min={Imin:g} (on normalized intensity)   gamma={gamma:g}")
    print(f"Total candidates generated={len(reflections)}   Kept (plotted)={len(reflections_det)}")
    print(f"Imax(raw) among candidates={Imax:.6g}")
    print("-"*100)
    print(f"Reference Spot1 = {hkl_str(ref1)}   (ZOLZ)   s={s_ref1:.6g}")
    print(f"Reference Spot2 = {hkl_str(ref2)}   (ZOLZ)   s={s_ref2:.6g}")
    print(f"Angle(Spot1, Spot2) = {ang:.3f} deg")
    print(f"(g1 x g2)·zone = {triple:.6g}   (should be> 0)")
    if idx_ref1 is None:
        print("NOTE: Spot1 is below intensity cutoff -> not plotted (but still chosen as reference).")
    if idx_ref2 is None:
        print("NOTE: Spot2 is below intensity cutoff -> not plotted (but still chosen as reference).")
    print("="*100 + "\n")

    # ----------------------------------------------------------------------------------------------
    # (15) Plot
    # ----------------------------------------------------------------------------------------------
    show_hkl_labels = prompt_yesno("Show (h k l) labels on spots? (y/n, default y): ", default_yes=True)

    title_str = f"SAED | Zone=[{u} {v} {w}] | {lat.centering} | {kV:.0f}kV | t={t_nm:g}nm"
    plot_saed_booklike(reflections_det,
                       ref1=ref1, ref2=ref2,
                       idx_ref1=idx_ref1, idx_ref2=idx_ref2,
                       theta_align=theta_align,
                       title_str=title_str,
                       show_hkl_labels=show_hkl_labels)


# ==================================================================================================
# ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    main()
