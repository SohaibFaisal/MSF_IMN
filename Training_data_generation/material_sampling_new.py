import math
from dataclasses import dataclass
from typing import Dict, Tuple, List
import random
import numpy as np


# Voigt order: [11, 22, 33, 23, 31, 12]
VOIGT_ORDER = ("11", "22", "33", "23", "31", "12")


@dataclass
class RangeSpec:
    min: float
    max: float
    dist: str = "uniform"   # "uniform" or "loguniform"


# ----------------------------------------------------------------------
# Sampling utilities
# ----------------------------------------------------------------------

def _sample_from_range(rng: np.random.Generator, spec: RangeSpec) -> float:
    if spec.max <= spec.min:
        raise ValueError("Range max must be > min")

    if spec.dist == "uniform":
        return rng.uniform(spec.min, spec.max)

    if spec.dist == "loguniform":
        if spec.min <= 0:
            raise ValueError("loguniform requires min > 0")
        return math.exp(rng.uniform(math.log(spec.min), math.log(spec.max)))

    raise ValueError(f"Unknown distribution '{spec.dist}'")


# ----------------------------------------------------------------------
# Elasticity
# ----------------------------------------------------------------------

def build_compliance(
    E11: float, E22: float, E33: float,
    nu12: float, nu23: float, nu31: float,
    G12: float, G23: float, G31: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build orthotropic compliance matrix (6x6).
    Returns compliance S and derived reciprocal Poisson ratios.
    """

    # Basic physical checks
    if min(E11, E22, E33, G12, G23, G31) <= 0:
        raise ValueError("Elastic moduli must be positive")

    if nu12 >= 0.5 or nu23 >= 0.5 or nu31 >= 0.5:
        raise ValueError("Poisson ratio must be < 0.5")

    # Reciprocity (major symmetry)
    nu21 = nu12 * E22 / E11
    nu32 = nu23 * E33 / E22
    nu13 = nu31 * E11 / E33

    S = np.zeros((6, 6), dtype=float)

    # Normal terms
    S[0, 0] = 1.0 / E11
    S[1, 1] = 1.0 / E22
    S[2, 2] = 1.0 / E33

    S[0, 1] = S[1, 0] = -nu12 / E11
    S[1, 2] = S[2, 1] = -nu23 / E22
    S[0, 2] = S[2, 0] = -nu13 / E11

    # Shear terms
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G31
    S[5, 5] = 1.0 / G12

    derived = {
        "nu21": nu21,
        "nu32": nu32,
        "nu13": nu13
    }

    return S, derived


def is_spd(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Symmetric positive definite check.
    """
    if not np.allclose(A, A.T, atol=1e-10):
        return False
    eigvals = np.linalg.eigvalsh(A)
    return np.min(eigvals) > tol


# ----------------------------------------------------------------------
# Main sampling API (this is what you call)
# ----------------------------------------------------------------------

def sample_orthotropic_material(
    ranges: Dict[str, RangeSpec],
    rng: np.random.Generator,
    max_tries: int = 10_000,
    spd_tol: float = 1e-12
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Sample ONE orthotropic material with SPD stiffness matrix.

    Returns:
        constants : dict of elastic constants (9 independent + 3 derived ν)
        C         : 6x6 stiffness matrix
    """

    required = ["E11", "E22", "E33", "nu12", "nu23", "nu31", "G12", "G23", "G31"]
    for k in required:
        if k not in ranges:
            raise KeyError(f"Missing range for '{k}'")

    for _ in range(max_tries):

        E11 = _sample_from_range(rng, ranges["E11"])
        E22 = _sample_from_range(rng, ranges["E22"])
        E33 = _sample_from_range(rng, ranges["E33"])

        nu12 = _sample_from_range(rng, ranges["nu12"])
        nu23 = _sample_from_range(rng, ranges["nu23"])
        nu31 = _sample_from_range(rng, ranges["nu31"])

        G12 = _sample_from_range(rng, ranges["G12"])
        G23 = _sample_from_range(rng, ranges["G23"])
        G31 = _sample_from_range(rng, ranges["G31"])

        try:
            S, derived = build_compliance(
                E11, E22, E33,
                nu12, nu23, nu31,
                G12, G23, G31
            )

            # Stability checks
            if not is_spd(S, tol=spd_tol):
                continue

            C = np.linalg.inv(S)

            if not is_spd(C, tol=spd_tol):
                continue

            constants = {
                "E11": E11, "E22": E22, "E33": E33,
                "nu12": nu12, "nu23": nu23, "nu31": nu31,
                "G12": G12, "G23": G23, "G31": G31,
                **derived
            }

            return constants, C

        except (ValueError, np.linalg.LinAlgError):
            continue

    raise RuntimeError("Failed to sample a valid orthotropic material")

def sample_interface_material(
    E_min, E_max
) -> Tuple[Dict[str, float], np.ndarray]:


    K_n = random.uniform(E_min, E_max)
    K_t = random.uniform(E_min, E_max)
    C = np.array([[K_n,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0, 0,0],[0,0,0,K_t,0,0],[0,0,0,0,K_t,0],[0,0,0,0,0,0]])
    constants = {
        "K_n": K_n, "K_t": K_t
    }

    return constants, C



    raise RuntimeError("Failed to sample a valid orthotropic material")


def sample_many_orthotropic_materials(
    ranges: Dict[str, RangeSpec],
    n_samples: int,
    seed: int | None = None
) -> List[Tuple[Dict[str, float], np.ndarray]]:
    """
    Convenience wrapper for multiple samples.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        samples.append(
            sample_orthotropic_material(ranges, rng)
        )

    return samples



def sample_many_interface_materials(
    E_min,E_max,
    n_samples: int,
    seed: int | None = None
) -> List[Tuple[Dict[str, float], np.ndarray]]:


    samples = []
    for _ in range(n_samples):
        samples.append(
            sample_interface_material(E_min,E_max)
        )

    return samples



ranges = {
    "E11":  RangeSpec(0.0000001, 0.0001, "loguniform"),
    "E22":  RangeSpec(10, 200, "loguniform"),
    "E33":  RangeSpec(10, 200, "loguniform"),

    "nu12": RangeSpec(0.0, 0.49),
    "nu23": RangeSpec(0.0, 0.49),
    "nu31": RangeSpec(0.0, 0.49),

    "G12":  RangeSpec(0.0000001, 0.001, "loguniform"),
    "G23":  RangeSpec(1, 50, "loguniform"),
    "G31":  RangeSpec(0.0000001, 0.0001, "loguniform"),
}



interface_samples = sample_many_interface_materials(1,200,4,42)
samples = sample_many_orthotropic_materials(ranges, n_samples=4, seed=42)

# a = []
# for consts, C in interface_samples:
#     print(consts)
#     print(C)



