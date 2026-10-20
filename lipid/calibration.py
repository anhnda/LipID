"""
lipid/calibration.py — operationally certified weighted-l1 Lipschitz levels L_cert (CAL-01).

Weighted-l1 metric d_sigma(u, v) = sum_j |u_j - v_j| / sigma_j. Its dual norm gives, for a differentiable f on a
convex domain K (here K = R^d, i.e. GLOBAL certificates):

    |f(u) - f(v)| <= sum_j |d_j f| |u_j - v_j| <= max_j sigma_j sup_z |d_j f(z)| * d_sigma(u, v),

so  L_cert = max_j sigma_j G_j  with  G_j >= sup_z |d_j f(z)|  is a valid Lipschitz level (paper eq. 15, app. H).

Coordinate bounds G_j (generator definitions = phase1.make_model, q(t) = t^2 / (1 + t^2)):
    linear       f = sum_j w_j z_j                        G_j = |w_j|                       (exact)
    unstable_id  f = z_1 + a tanh(4 mean(z))              G_1 = 1 + 4|a|/d,  G_j = 4|a|/d     (exact if a >= 0)
    fa           f = z_1 + a tanh(z_1) sum_{j>=2} q(z_1 - z_j)
                                                          G_1 = 1 + |a|(d-1)(1 + c_q),  G_j = |a| c_q
    stable_ood   f = z_1 + a tanh(sum_{j>=2} (z_1 - z_j)^2)
                                                          G_1 = 1 + 2|a| sqrt(d-1) c_t,  G_j = 2|a| c_t
with c_q = sup|q'| = 9 / (8 sqrt 3) (closed form) and c_t = sup_{u>=0} sqrt(u) sech^2(u) (certified below).

STRICT RULE: nothing in this module evaluates a response function. L_cert depends only on
(generator name, d, a, sigma) — never on anchor labels, hybrid responses or L_V. tests/test_calibration.py checks
this by replacing phase1.make_model with a function that raises.

Note on the legacy code: phase1.lipschitz_true('unstable_id') uses the SIGNED a (1 + 4a/d); for a < 0 that value is
not an upper bound (it can even be negative). All archived runs use a = 4 > 0, where it coincides with L_cert.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

MODELS = ("linear", "fa", "stable_ood", "unstable_id")

# sup_t |q'(t)|, q'(t) = 2t / (1 + t^2)^2, attained at t = 1/sqrt(3): 2/sqrt(3) * 9/16 = 9 / (8 sqrt 3)
CQ = 9.0 / (8.0 * math.sqrt(3.0))
REL_SLACK = 1e-12   # relative slack absorbing floating-point rounding of elementary functions (~1 ulp each)


def _sech2(u: float) -> float:
    c = math.cosh(u)
    return 1.0 / (c * c)


@lru_cache(maxsize=None)
def certify_ct(width: float = 1e-13) -> tuple:
    """Rigorous upper bound on c_t = sup_{u >= 0} g(u),  g(u) = sqrt(u) sech^2(u).

    g'(u) = sech^2(u) (1 - 4 u tanh u) / (2 sqrt u) for u > 0 and H(u) = 4 u tanh u is strictly increasing on u > 0
    (product of two positive increasing functions), with H(0) = 0 and H -> inf. Hence g has a unique stationary
    point u* (H(u*) = 1), g increases on [0, u*] and decreases on [u*, inf): c_t = g(u*).
    Bisection gives lo <= u* <= hi with H(lo) < 1 < H(hi) checked in floating point after widening the bracket by a
    relative 1e-9 (|H'| ~ 3.5 near u*, so the signs are decided far above rounding). Then, because sqrt is increasing
    and sech^2 is decreasing on u >= 0:
        c_t = sqrt(u*) sech^2(u*) <= sqrt(hi) sech^2(lo).
    The returned bound multiplies this by (1 + REL_SLACK). Returns (upper_bound, lo, hi)."""
    H = lambda u: 4.0 * u * math.tanh(u) - 1.0
    lo, hi = 0.25, 1.0
    if not (H(lo) < 0.0 < H(hi)):
        raise AssertionError("initial bracket for u* is wrong")
    while hi - lo > width:
        mid = 0.5 * (lo + hi)
        if H(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    lo, hi = lo * (1.0 - 1e-9), hi * (1.0 + 1e-9)
    if not (H(lo) < 0.0 < H(hi)):
        raise AssertionError("widened bracket does not certify the sign change")
    ub = math.sqrt(hi) * _sech2(lo) * (1.0 + REL_SLACK)
    return ub, lo, hi


def ct_upper() -> float:
    return certify_ct()[0]


def linear_weights_np(d: int) -> np.ndarray:
    """Same weights as phase1.linear_weights (torch.linspace(1.0, 0.2, d), float64). Imported, not re-derived,
    so a change in the generator cannot silently desynchronize the certificate."""
    import phase1 as P   # noqa: local import keeps this module free of model construction
    return P.linear_weights(d).detach().cpu().numpy().astype(np.float64)


def coordinate_bounds(name: str, d: int, a: float) -> np.ndarray:
    """G_j >= sup_z |d f / d z_j| for every coordinate j (unweighted). Global (K = R^d)."""
    if d < 1:
        raise ValueError("d must be >= 1")
    aa = abs(float(a))
    G = np.zeros(d, dtype=np.float64)
    if name == "linear":
        G[:] = np.abs(linear_weights_np(d))
    elif name == "unstable_id":
        G[:] = 4.0 * aa / d
        G[0] = 1.0 + 4.0 * aa / d
    elif name == "fa":
        G[0] = 1.0 + aa * (d - 1) * (1.0 + CQ)
        G[1:] = aa * CQ
    elif name == "stable_ood":
        ct = ct_upper()
        G[0] = 1.0 + 2.0 * aa * math.sqrt(d - 1) * ct
        G[1:] = 2.0 * aa * ct
    else:
        raise ValueError(f"unknown generator {name!r}")
    if name in ("fa", "stable_ood"):
        G *= (1.0 + REL_SLACK)
    return G


@dataclass
class CertifiedL:
    model: str
    d: int
    a: float
    L_cert: float
    argmax_coord: int
    weighted_coord_bounds: list      # sigma_j * G_j
    scope: str = "global"            # certified on K = R^d (contains every anchor and hybrid)
    uses_hybrid_responses: bool = False
    uses_anchor_labels: bool = False
    formula: str = ""

    def as_dict(self):
        return asdict(self)


FORMULA = {
    "linear": "max_j sigma_j |w_j|",
    "unstable_id": "max(sigma_1 (1 + 4|a|/d), (4|a|/d) max_{j>=2} sigma_j)",
    "fa": "max(sigma_1 (1 + |a|(d-1)(1 + c_q)), |a| c_q max_{j>=2} sigma_j),  c_q = 9/(8 sqrt 3)",
    "stable_ood": "max(sigma_1 (1 + 2|a| sqrt(d-1) c_t), 2|a| c_t max_{j>=2} sigma_j),  c_t = sup sqrt(u) sech^2(u)",
}


def certify(name: str, d: int, sigma, a: float) -> CertifiedL:
    """Certified global weighted-l1 Lipschitz level. `sigma` may be a torch tensor or array (only its values are used)."""
    s = np.asarray(sigma.detach().cpu().numpy() if hasattr(sigma, "detach") else sigma, dtype=np.float64).reshape(-1)
    if s.shape[0] != d:
        raise ValueError("sigma has wrong length")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("sigma must be positive and finite")
    wb = s * coordinate_bounds(name, d, a)
    j = int(np.argmax(wb))
    return CertifiedL(model=name, d=d, a=float(a), L_cert=float(wb[j]), argmax_coord=j,
                      weighted_coord_bounds=wb.tolist(), formula=FORMULA[name])


def L_cert(name: str, d: int, sigma, a: float) -> float:
    return certify(name, d, sigma, a).L_cert


def constants() -> dict:
    ub, lo, hi = certify_ct()
    return dict(c_q=CQ, c_t_upper=ub, c_t_root_bracket=[lo, hi], rel_slack=REL_SLACK)
