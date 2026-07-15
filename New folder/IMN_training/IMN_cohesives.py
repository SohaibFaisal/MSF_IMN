"""
IMN 12: Able to use deformation gradient and Green lagrange strain

"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from .Optimization_loop import *
from .data_loader import tangent_from_csv3d
import numpy as np
from pathlib import Path
torch.set_default_dtype(torch.double)


# ---------------------------------------------------------
# Helpers: Voigt conventions (engineering shear)
# ---------------------------------------------------------
def deviatoric_part_voigt(sigma: torch.Tensor) -> torch.Tensor:
    """
    sigma: (6,)
    returns deviatoric part in the same Voigt convention (engineering shear)
    """
    I = torch.tensor([1, 1, 1, 0, 0, 0], dtype=sigma.dtype, device=sigma.device)
    mean = (sigma[:3].sum()) / 3.0
    return sigma - mean * I


def j2_norm_voigt(s: torch.Tensor) -> torch.Tensor:
    """
    s: (6,) deviatoric stress in Voigt (engineering shear)
    J2 norm: sqrt( s:s ) with correct shear weighting for engineering shear
    """
    return torch.sqrt(
        s[0]**2 + s[1]**2 + s[2]**2
        + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
    )


def voigt6_to_sym33(v6: torch.Tensor) -> torch.Tensor:
    """
    v6: (...,6) in Voigt with engineering shear convention used in your script
        [xx, yy, zz, xy, xz, yz]
    Returns: (...,3,3) symmetric
    NOTE: For strains in engineering shear, tensor shear components are v/2.
    For stresses, tensor shear components equal v (engineering shear stress convention).
    We'll provide both below.
    """
    v = v6
    T = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    T[..., 0, 0] = v[..., 0]
    T[..., 1, 1] = v[..., 1]
    T[..., 2, 2] = v[..., 2]
    # treat as "stress-like" by default: off-diagonals = v
    T[..., 0, 1] = v[..., 3]
    T[..., 1, 0] = v[..., 3]
    T[..., 0, 2] = v[..., 4]
    T[..., 2, 0] = v[..., 4]
    T[..., 1, 2] = v[..., 5]
    T[..., 2, 1] = v[..., 5]
    return T

def sym33_to_voigt6(T: torch.Tensor) -> torch.Tensor:
    """
    T: (...,3,3) symmetric
    Returns: (...,6) [xx, yy, zz, xy, xz, yz]
    (stress-like shear)
    """
    v = torch.stack([T[...,0,0], T[...,1,1], T[...,2,2],
                     T[...,0,1], T[...,0,2], T[...,1,2]], dim=-1)
    return v

def strain_voigt6_to_sym33(E6: torch.Tensor) -> torch.Tensor:
    """
    For engineering shear strain: gamma_xy = 2*E_xy.
    If E6 stores engineering shear, then E_xy = E6[3]/2, etc.
    """
    E = E6
    T = torch.zeros(*E.shape[:-1], 3, 3, dtype=E.dtype, device=E.device)
    T[...,0,0] = E[...,0]
    T[...,1,1] = E[...,1]
    T[...,2,2] = E[...,2]
    T[...,0,1] = 0.5 * E[...,3]
    T[...,1,0] = 0.5 * E[...,3]
    T[...,0,2] = 0.5 * E[...,4]
    T[...,2,0] = 0.5 * E[...,4]
    T[...,1,2] = 0.5 * E[...,5]
    T[...,2,1] = 0.5 * E[...,5]
    return T

def sym33_to_strain_voigt6(E: torch.Tensor) -> torch.Tensor:
    """
    Returns engineering shear: gamma_xy = 2*E_xy, etc.
    """
    v = torch.stack([E[...,0,0], E[...,1,1], E[...,2,2],
                     2*E[...,0,1], 2*E[...,0,2], 2*E[...,1,2]], dim=-1)
    return v

def green_lagrange_from_F(F: torch.Tensor) -> torch.Tensor:
    """
    F: (3,3) or (...,3,3)
    Returns E_tensor: (...,3,3) symmetric
    """
    I = torch.eye(3, dtype=F.dtype, device=F.device)
    C = F.transpose(-1, -2) @ F
    E = 0.5 * (C - I)
    return E


def elastic_stiffness_from_Enu(E: float, nu: float, device=None, dtype=torch.double) -> torch.Tensor:
    """
    6x6 isotropic stiffness in Voigt (engineering shear), small strain.
    """
    E = torch.tensor(float(E), dtype=dtype, device=device)
    nu = torch.tensor(float(nu), dtype=dtype, device=device)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    C = torch.zeros(6, 6, dtype=dtype, device=device)
    # normal-normal block
    C[0, 0] = lam + 2 * mu
    C[1, 1] = lam + 2 * mu
    C[2, 2] = lam + 2 * mu
    C[0, 1] = lam
    C[0, 2] = lam
    C[1, 0] = lam
    C[1, 2] = lam
    C[2, 0] = lam
    C[2, 1] = lam
    # shear
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


# ---------------------------------------------------------
# H(n) matrix: maps a_j (3) -> symmetric tensor a⊗_s n in Voigt (6)
# ---------------------------------------------------------
def H_matrix(n: torch.Tensor) -> torch.Tensor:
    """
    n: (3,) normal vector
    returns H(n): (6, 3) such that voigt(a⊗_s n) = H(n) @ a
    """
    n0, n1, n2 = n[0], n[1], n[2]
    zero = torch.zeros((), dtype=n.dtype, device=n.device)

    row1 = torch.stack([n0,   zero, zero])
    row2 = torch.stack([zero, n1,   zero])
    row3 = torch.stack([zero, zero, n2])
    row4 = torch.stack([n1,   n0,   zero])
    row5 = torch.stack([n2,   zero, n0])
    row6 = torch.stack([zero, n2,   n1])

    return torch.stack([row1, row2, row3, row4, row5, row6], dim=0)  # (6,3)

def H_matrix_for_F(n: torch.Tensor) -> torch.Tensor:
    """
    n: (3,) normal vector
    returns H(n): (6, 3) such that voigt(a⊗_s n) = H(n) @ a
    """
    n0, n1, n2 = n[0], n[1], n[2]
    zero = torch.zeros((), dtype=n.dtype, device=n.device)

    row1 = torch.stack([n0,   zero, zero])
    row2 = torch.stack([zero, n0,   zero])
    row3 = torch.stack([zero, zero, n0])
    row4 = torch.stack([n1,   zero,   zero])
    row5 = torch.stack([zero,   n1, zero])
    row6 = torch.stack([zero, zero,   n1])
    row7 = torch.stack([n2, zero, zero])
    row8 = torch.stack([zero, n2, zero])
    row9 = torch.stack([zero, zero, n2])

    return torch.stack([row1, row2, row3, row4, row5, row6, row7, row8, row9], dim=0)  # (9,3)


# ---------------------------------------------------------
# Constrained alpha computation for one interaction V_j:
# enforce sum_{i in Vj} W_i * alpha_{i,j} = 0
# ---------------------------------------------------------
def compute_alphas_for_join(W_children: torch.Tensor,
                            beta_j: torch.Tensor,
                            alpha0_j: torch.Tensor) -> torch.Tensor:
    """
    W_children: (Lj,) positive weights for nodes in V_j
    beta_j:     (Lj-1,) trainable
    alpha0_j:   scalar in (0,1), fixed buffer

    Returns:
        alpha_children: (Lj,)
    """
    Lj = W_children.shape[0]
    assert beta_j.shape[0] == Lj - 1, "beta_j must have length Lj-1."

    W0 = W_children[0]
    denom = torch.sum(beta_j) + 1e-12

    a = torch.zeros_like(W_children)
    a[0] = alpha0_j
    a[1:] = -(W0 * alpha0_j / (W_children[1:] + 1e-12)) * (beta_j / denom)
    return a


# ---------------------------------------------------------
# Node object: holds optimizing + non-optimizing state + local constitutive laws
# ---------------------------------------------------------
class MaterialNode(nn.Module):
    def __init__(self,
                 name: str,
                 node_type: str = "elastic",   # "elastic" or "plastic"
                 amplify: float = 1.0):
        super().__init__()
        self.name = name
        self.node_type = node_type

        self.register_buffer("amplify", torch.tensor(float(amplify), dtype=torch.double))
        self.register_buffer("C", torch.eye(6, dtype=torch.double))

        # plastic internal variables (buffers)
        self.register_buffer("eps_p", torch.zeros(6, dtype=torch.double))
        self.register_buffer("p", torch.zeros((), dtype=torch.double))

        # plastic parameters
        self.register_buffer("E", torch.tensor(210e3, dtype=torch.double))
        self.register_buffer("nu", torch.tensor(0.30, dtype=torch.double))
        self.register_buffer("sigma_y", torch.tensor(250.0, dtype=torch.double))
        self.register_buffer("H", torch.tensor(1000.0, dtype=torch.double))

        # trainable weight param
        self.z = nn.Parameter(0.01 * torch.randn((), dtype=torch.double))

    @torch.no_grad()
    def set_C(self, C_new: torch.Tensor):
        self.C.copy_(C_new)

    @torch.no_grad()
    def set_plastic_props(self, E: float, nu: float, sigma_y: float, H: float):
        self.E.fill_(float(E))
        self.nu.fill_(float(nu))
        self.sigma_y.fill_(float(sigma_y))
        self.H.fill_(float(H))

    @torch.no_grad()
    def reset_state(self):
        self.eps_p.zero_()
        self.p.zero_()

    def weight_raw(self) -> torch.Tensor:
        return F.softplus(self.z) + 1e-12

    def get_state(self):
        # return *detached* copies for trial updates
        return (self.eps_p.detach().clone(), self.p.detach().clone())

    @torch.no_grad()
    def set_state(self, state):
        eps_p_new, p_new = state
        self.eps_p.copy_(eps_p_new)
        self.p.copy_(p_new)

    def eval_local_laws(self, eps: torch.Tensor, state=None, commit: bool = False):
        """
        eps: (6,)
        state: (eps_p, p) for plastic trial; if None uses node buffers.
        commit: if True, writes updated state into node buffers (plastic only)

        Returns:
            sigma: (6,)
            C_tan: (6,6)
            new_state: (eps_p_new, p_new)  (plastic) or None (elastic)
        """
        if self.node_type == "elastic":
            sigma = self.C @ eps
            return sigma, self.C, None

        if self.node_type == "plastic":
            if state is None:
                eps_p, p = self.eps_p, self.p
            else:
                eps_p, p = state

            sigma, Ct, new_state = self._plastic_j2_trial(eps, eps_p, p)

            if commit:
                # commit updated internal vars
                self.set_state(new_state)

            return sigma, Ct, new_state

        raise NotImplementedError(f"Material model '{self.node_type}' not implemented.")

    def _plastic_j2_trial(self, eps: torch.Tensor, eps_p: torch.Tensor, p: torch.Tensor):
        """
        Small-strain J2, isotropic hardening.
        Returns sigma, algorithmic tangent Ct, and updated trial state.
        """
        device = eps.device
        E = self.E.to(device)
        nu = self.nu.to(device)
        sigy = self.sigma_y.to(device)
        Hiso = self.H.to(device)

        mu = E / (2.0 * (1.0 + nu))
        C_e = elastic_stiffness_from_Enu(float(E.item()), float(nu.item()), device=device, dtype=eps.dtype)

        # trial stress
        eps_e_trial = eps - eps_p
        sigma_trial = C_e @ eps_e_trial

        # deviatoric + norm
        s_trial = deviatoric_part_voigt(sigma_trial)
        s_norm = j2_norm_voigt(s_trial)

        f_trial = s_norm - (sigy + Hiso * p)

        if f_trial <= 0:
            # elastic
            return sigma_trial, C_e, (eps_p, p)

        # plastic
        delta_gamma = f_trial / (3.0 * mu + Hiso)
        n_dev = s_trial / (s_norm + 1e-12)

        sigma = sigma_trial - 2.0 * mu * delta_gamma * n_dev

        # updated trial state (DO NOT write to buffers here)
        eps_p_new = eps_p + delta_gamma * n_dev
        p_new = p + delta_gamma

        # simple consistent tangent
        factor = (2.0 * mu) * (2.0 * mu) / (3.0 * mu + Hiso)
        Ct = C_e - factor * torch.outer(n_dev, n_dev)

        return sigma, Ct, (eps_p_new, p_new)



# ---------------------------------------------------------
# Interaction object: angles + beta + alpha0 (fixed)
# beta length depends on Lj = len(Vj)
# ---------------------------------------------------------
class Interaction(nn.Module):
    def __init__(self, Lj: int):
        super().__init__()
        self.Lj = Lj

        # trainable angles (per join)
        self.theta_raw = nn.Parameter(0.1 * torch.randn((), dtype=torch.double))
        self.phi_raw   = nn.Parameter(0.1 * torch.randn((), dtype=torch.double))

        # trainable beta (Lj-1)
        self.beta = nn.Parameter(0.01 * torch.randn(Lj - 1, dtype=torch.double))

        # fixed alpha0 in (0,1)
        self.register_buffer("alpha0", torch.rand((), dtype=torch.double))

    def normal_and_H(self):
        theta = torch.pi * torch.sigmoid(self.theta_raw)      # [0, pi]
        phi   = 2.0 * torch.pi * torch.sigmoid(self.phi_raw)  # [0, 2pi)

        n = torch.stack([
            torch.cos(phi) * torch.sin(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(theta)
        ])
        H = H_matrix(n)  # (6,3)
        return n, H, theta, phi


# ---------------------------------------------------------
# IMN network (constrained params only) with downscale/upscale
# Two modes for C_bar: C16 or SixLoads
# ---------------------------------------------------------
class IMNNet(nn.Module):
    def __init__(self, N_layers: int, mode: str = "C16", kinematics: str = 'small'):
        super().__init__()
        assert mode in ("C16", "SixLoads")
        assert kinematics in ("small", "finite")
        self.mode = mode
        self.kinematics = kinematics

        self.N, self.V_list = self.architecture(N_layers)
        self.M = len(self.V_list)

        # nodes
        self.nodes = nn.ModuleList([MaterialNode(name=f"node_{i}", node_type="elastic") for i in range(self.N)])

        # joins
        self.joins = nn.ModuleList([Interaction(Lj=len(Vj)) for Vj in self.V_list])

    def architecture(self, N_layers: int):

        Vj = []
        if N_layers == 1:
            nodes = 4
            x = [4]

        elif N_layers == 2:
            nodes = 8
            x = [4, 8]

        elif N_layers == 3:
            nodes = 16
            x = [4, 8, 16]

        elif N_layers == 4:
            nodes = 32
            x = [4, 8, 16, 32]

        elif N_layers == 5:
            nodes = 64
            x = [4, 8, 16, 32, 64]

        elif N_layers == 6:
            nodes = 128
            x = [4,8,16,32,64,128]

        elif N_layers == 7:
            nodes = 256
            x = [4,8,16,32,64,128,256]

        elif N_layers == 8:
            nodes = 512
            x = [4,8,16,32,64,128,256,512]

        elif N_layers == 9:
            nodes = 1024
            x = [4,8,16,32,64,128,256,512,1024]

        else:
            raise ValueError("N_layers must be 1..9")

        for gap in x:
            for nn in range(0, nodes, gap):
                Vj.append(list(range(nn, nn + gap)))

        return nodes, Vj

    # ---------- assign C1 to even nodes and C2 to odd nodes ----------
    @torch.no_grad()
    def assign_phases(self, C1: torch.Tensor, C2: torch.Tensor,C3: torch.Tensor):
        for i, node in enumerate(self.nodes):
            if i%3==0:
                node.set_C(C1)
            elif i%3==1:
                node.set_C(C2)
            elif i%3==2:
                node.set_C(C3)


    # ---------- optionally set node types even/odd ----------
    @torch.no_grad()
    def assign_types_even_odd(self, type_even: str = "elastic", type_odd: str = "elastic"):
        for i, node in enumerate(self.nodes):
            node.node_type = type_even if (i % 2 == 0) else type_odd

    # ---------- reset all internal states (important for plastic) ----------
    @torch.no_grad()
    def reset_all_states(self):
        for node in self.nodes:
            node.reset_state()

    # ---------- weights used in computations (positive + even/odd sum constraint) ----------
    def compute_weights(self) -> torch.Tensor:
        w_raw = torch.stack([n.weight_raw() for n in self.nodes], dim=0)  # (N,)

        # Constrained --------------------------------------------
        # even_idx = torch.arange(0, self.N, 2, device=w_raw.device)
        # odd_idx  = torch.arange(1, self.N, 2, device=w_raw.device)
        #
        # sum_even = w_raw[even_idx].sum() + 1e-12
        # sum_odd  = w_raw[odd_idx].sum() + 1e-12
        #
        # W = torch.zeros_like(w_raw)
        # # Set volume fractions
        # # Even matrix
        # # Odd fibers
        # W[even_idx] = 0.83 * w_raw[even_idx] / (sum_even)
        # W[odd_idx]  = 0.17 * w_raw[odd_idx]  / (sum_odd)
        # return W  # (N,)

        # Unconstrained ---------------------------------------------
        w_raw = w_raw/w_raw.sum()
        return w_raw

    # ---------- geometry (H_j) ----------
    def compute_H_list(self):
        H_list, theta_list, phi_list = [], [], []
        for j in range(self.M):
            _, H, theta, phi = self.joins[j].normal_and_H()
            H_list.append(H)
            theta_list.append(theta)
            phi_list.append(phi)
        theta = torch.stack(theta_list)
        phi = torch.stack(phi_list)
        return H_list, theta, phi

    # ---------- alpha matrix (N,M), constrained ----------
    def compute_alpha_matrix(self, W: torch.Tensor) -> torch.Tensor:
        device = W.device
        alpha = torch.zeros(self.N, self.M, dtype=torch.double, device=device)
        for j, Vj in enumerate(self.V_list):
            idx = torch.tensor(Vj, dtype=torch.long, device=device)
            Wc = W[idx]
            beta_j = self.joins[j].beta
            a0 = self.joins[j].alpha0
            alpha_children = compute_alphas_for_join(Wc, beta_j, a0)
            alpha[idx, j] = alpha_children
        return alpha

    # ---------- build D_i matrices (6 x 3M) ----------
    def build_D_list(self, alpha: torch.Tensor, H_list):
        D_list = []
        for i in range(self.N):
            Di = torch.zeros(6, 3 * self.M, dtype=torch.double, device=alpha.device)
            for j in range(self.M):
                Di[:, 3*j:3*(j+1)] = alpha[i, j] * H_list[j]
            D_list.append(Di)
        return D_list

    # ---------- build K, M, Y, C_avg from Appendix C using node tangents ----------
    def build_K_M_Y_Cavg(self, W, D_list, eps_probe: torch.Tensor | None = None):
        """
        For nonlinear materials, tangents depend on state/strain.
        Here we query each node tangent by calling eval_local_laws(eps_probe).
        If eps_probe is None, uses zero strain.
        """
        device = W.device
        dim = 3 * self.M
        sumW = W.sum() + 1e-12

        if eps_probe is None:
            eps_probe = torch.zeros(6, dtype=torch.double, device=device)

        K = torch.zeros(dim, dim, dtype=torch.double, device=device)
        Mmat = torch.zeros(dim, 6, dtype=torch.double, device=device)
        Y = torch.zeros(6, dim, dtype=torch.double, device=device)
        C_avg = torch.zeros(6, 6, dtype=torch.double, device=device)

        for i, node in enumerate(self.nodes):
            Wi = W[i]
            # get tangent at probe strain (elastic constant; plastic depends on state)
            _, Ci = node.eval_local_laws(eps_probe)

            Di = D_list[i]

            Ci_Di = Ci @ Di
            K += Wi * (Di.T @ Ci_Di)
            Mmat += Wi * (Di.T @ Ci)
            Y += Wi * (Ci @ Di)
            C_avg += Wi * Ci

        Y = Y / sumW
        C_avg = C_avg / sumW
        return K, Mmat, Y, C_avg

    # =========================================================
    # downscale: given macro_strain -> compute U and local strains
    # =========================================================
    def downscale(self, macro_strain: torch.Tensor,
                  newton_max_iter: int = 25,
                  newton_tol: float = 1e-10,
                  commit_state: bool = True):
        """
        Newton solve for U:
            R(U) = sum_i W_i D_i^T sigma_i(eps_bar + D_i U) = 0


        """
        if macro_strain.dim() == 2:
            macro_strain = macro_strain.squeeze(0)
        assert macro_strain.shape == (6,)

        W = self.compute_weights()
        H_list, _, _ = self.compute_H_list()
        alpha = self.compute_alpha_matrix(W)
        D_list = self.build_D_list(alpha, H_list)

        sumW = W.sum() + 1e-12
        dim = 3 * self.M
        device = W.device

        # trial internal states per node (for plastic)
        trial_states = []
        for node in self.nodes:
            if node.node_type == "plastic":
                trial_states.append(node.get_state())
            else:
                trial_states.append(None)

        # Newton init
        U = torch.zeros(dim, dtype=torch.double, device=device)

        eps_reg = 1e-10

        # Newton loop
        for it in range(newton_max_iter):
            # assemble residual R, Jacobian K, and also M, Y, C_avg at current trial
            R = torch.zeros(dim, dtype=torch.double, device=device)
            K = torch.zeros(dim, dim, dtype=torch.double, device=device)
            Mmat = torch.zeros(dim, 6, dtype=torch.double, device=device)
            Y = torch.zeros(6, dim, dtype=torch.double, device=device)
            C_avg = torch.zeros(6, 6, dtype=torch.double, device=device)

            new_trial_states = [None] * self.N

            # local strains depend on current U
            for i, node in enumerate(self.nodes):
                Di = D_list[i]  # (6,3M)
                eps_i = macro_strain + Di @ U  # (6,)

                sigma_i, Ci, st_new = node.eval_local_laws(eps_i, state=trial_states[i], commit=False)
                new_trial_states[i] = st_new if node.node_type == "plastic" else None

                Wi = W[i]

                # R += Wi * Di^T * sigma_i
                R += Wi * (Di.T @ sigma_i)


                # K += Wi * Di^T * Ci * Di
                K += Wi * (Di.T @ (Ci @ Di))


                # M += Wi * Di^T * Ci
                Mmat += Wi * (Di.T @ Ci)

                # Y += Wi * Ci * Di
                Y += Wi * (Ci @ Di)

                # C_avg += Wi * Ci
                C_avg += Wi * Ci

            Y = Y / sumW
            C_avg = C_avg / sumW

            # convergence check
            rnorm = torch.norm(R)
            if rnorm.item() < newton_tol:
                trial_states = new_trial_states
                break

            # Newton step: (K + regI) dU = -R
            K_reg = K + eps_reg * torch.eye(dim, dtype=torch.double, device=device)
            dU = torch.linalg.solve(K_reg, -R)
            U = U + dU


            # update trial states for next iteration
            trial_states = new_trial_states

            # optional: step convergence check
            if torch.norm(dU).item() < newton_tol:
                break

        # commit plastic state to nodes (optional)
        if commit_state:
            for i, node in enumerate(self.nodes):
                if node.node_type == "plastic" and trial_states[i] is not None:
                    node.set_state(trial_states[i])

        # final K_reg for outputs (rebuild once using final trial_states)
        # (we already have K/M/Y/C_avg from last iter if you want; simplest is reuse computed K in scope)
        # Here: recompute one last time cleanly for outputs:
        R = torch.zeros(dim, dtype=torch.double, device=device)
        K = torch.zeros(dim, dim, dtype=torch.double, device=device)
        Mmat = torch.zeros(dim, 6, dtype=torch.double, device=device)
        Y = torch.zeros(6, dim, dtype=torch.double, device=device)
        C_avg = torch.zeros(6, 6, dtype=torch.double, device=device)

        for i, node in enumerate(self.nodes):
            Di = D_list[i]
            eps_i = macro_strain + Di @ U
            sigma_i, Ci, _ = node.eval_local_laws(eps_i, state=None, commit=False)  # now buffers already committed if chosen
            Wi = W[i]
            K += Wi * (Di.T @ (Ci @ Di))
            Mmat += Wi * (Di.T @ Ci)
            Y += Wi * (Ci @ Di)
            C_avg += Wi * Ci

        Y = Y / sumW
        C_avg = C_avg / sumW
        K_reg = K + eps_reg * torch.eye(dim, dtype=torch.double, device=device)

        # local strains (final)
        local_strains = []
        for i in range(self.N):
            local_strains.append(macro_strain + D_list[i] @ U)
        local_strains = torch.stack(local_strains, dim=0)  # (N,6)

        return local_strains, U, W, alpha, D_list, K_reg, Mmat, Y, C_avg

    # =========================================================
    # upscale: from local strains -> local stresses -> sigma_bar, C_bar
    # =========================================================
    def upscale(self, local_strains: torch.Tensor,
                W: torch.Tensor,
                D_list,
                K_reg: torch.Tensor,
                Mmat: torch.Tensor,
                Y: torch.Tensor,
                C_avg: torch.Tensor):
        """
        local_strains: (N,6)
        Returns:
            sigma_bar: (6,)
            C_bar: (6,6)
            local_stresses: (N,6)
        """
        sumW = W.sum() + 1e-12

        local_stresses = []
        for i, node in enumerate(self.nodes):
            sigma_i, _, _ = node.eval_local_laws(local_strains[i])
            local_stresses.append(sigma_i)
        local_stresses = torch.stack(local_stresses, dim=0)  # (N,6)

        sigma_bar = (W[:, None] * local_stresses).sum(dim=0) / sumW  # (6,)

        # tangent stiffness
        if self.mode == "C16":
            X = torch.linalg.solve(K_reg, -Mmat)  # (3M,6)
            C_bar = C_avg + Y @ X
        else:
            C_cols = []
            for k in range(6):
                eps_bar = torch.zeros(6, dtype=torch.double, device=W.device)
                eps_bar[k] = 1.0

                U_k = torch.linalg.solve(K_reg, -(Mmat @ eps_bar))  # (3M,)

                sig_k = []
                for i, node in enumerate(self.nodes):
                    eps_i = eps_bar + D_list[i] @ U_k
                    sigma_i, _ = node.eval_local_laws(eps_i)
                    sig_k.append(sigma_i)
                sig_k = torch.stack(sig_k, dim=0)

                sigma_bar_k = (W[:, None] * sig_k).sum(dim=0) / sumW
                C_cols.append(sigma_bar_k)

            C_bar = torch.stack(C_cols, dim=1)

        return sigma_bar, C_bar, local_stresses

    # =========================================================
    # model call: pass macro_strain and get sigma_bar, C_bar
    # =========================================================
    def forward(self, macro_strain: torch.Tensor):
        local_strains, U, W, alpha, D_list, K_reg, Mmat, Y, C_avg = self.downscale(macro_strain)
        sigma_bar, C_bar, local_stresses = self.upscale(local_strains, W, D_list, K_reg, Mmat, Y, C_avg)
        return sigma_bar, C_bar, local_strains, local_stresses, U

    def forward_F(self, F_macro: torch.Tensor):
        """
        Finite kinematics interface:
          Input: F_macro (3,3) or (1,3,3)
          Output:
            P_bar  (3,3)  first Piola
            S_bar  (6,)   second Piola in Voigt
            C_bar  (6,6)  dS/dE in Voigt (reference tangent)
            local_E (N,6) local Green-Lagrange strains (Voigt)
            local_S (N,6) local 2PK stresses (Voigt)
            U       (3M,)
        This uses IMN with macro "strain" = E_bar in Voigt (engineering shear).
        """
        if F_macro.dim() == 3:
            F = F_macro.squeeze(0)
        else:
            F = F_macro
        assert F.shape == (3, 3)

        # 1) compute macro Green-Lagrange E and convert to Voigt (engineering shear)
        E = green_lagrange_from_F(F)  # (3,3)
        E6 = sym33_to_strain_voigt6(E)  # (6,)

        # 2) run the standard IMN machinery but interpret:
        #    "macro_strain" = E6, "stress" returned = S6 (2PK)
        #    This reuses your downscale/upscale structure.
        local_E6, U, W, alpha, D_list, K_reg, Mmat, Y, C_avg = self.downscale(E6)

        # local stresses/tangents from nodes:
        local_S6 = []
        local_Ct = []
        for i, node in enumerate(self.nodes):
            # IMPORTANT: node.eval_local_laws currently assumes sigma = C * strain.
            # In finite mode, interpret that as S = C : E (2PK vs GL).
            S_i, C_i = node.eval_local_laws(local_E6[i])[:2] if isinstance(node.eval_local_laws(local_E6[i]), tuple) else node.eval_local_laws(local_E6[i])
            # If your node.eval_local_laws returns (sigma, C, state) in your newer version, adapt:
            # S_i, C_i, _ = node.eval_local_laws(local_E6[i], state=..., commit=...)
            local_S6.append(S_i)
            local_Ct.append(C_i)
        local_S6 = torch.stack(local_S6, dim=0)  # (N,6)

        # 3) homogenized 2PK stress
        sumW = W.sum() + 1e-12
        S_bar6 = (W[:, None] * local_S6).sum(dim=0) / sumW  # (6,)

        # 4) homogenized tangent (reference): C_bar = dS/dE
        # Use your existing C16 or SixLoads logic but in (E,S) space.
        if self.mode == "C16":
            X = torch.linalg.solve(K_reg, -Mmat)  # (3M,6)
            C_bar = C_avg + Y @ X  # (6,6)
        else:
            # 6 basis GL strains
            cols = []
            for k in range(6):
                E6k = torch.zeros(6, dtype=torch.double, device=F.device)
                E6k[k] = 1.0
                local_E6k, Uk, Wk, _, D_list_k, K_reg_k, Mmat_k, _, _ = self.downscale(E6k)

                local_Sk = []
                for i, node in enumerate(self.nodes):
                    S_i, _ = node.eval_local_laws(local_E6k[i])[:2] if isinstance(node.eval_local_laws(local_E6k[i]), tuple) else node.eval_local_laws(local_E6k[i])
                    local_Sk.append(S_i)
                local_Sk = torch.stack(local_Sk, dim=0)
                S_bar_k = (Wk[:, None] * local_Sk).sum(dim=0) / (Wk.sum() + 1e-12)
                cols.append(S_bar_k)
            C_bar = torch.stack(cols, dim=1)  # (6,6)

        # 5) convert 2PK to 1PK: P = F * S (tensor form)
        S_tensor = voigt6_to_sym33(S_bar6)  # (3,3) stress-like mapping OK
        P_bar = F @ S_tensor  # (3,3)

        return P_bar, S_bar6, C_bar, local_E6, local_S6, U

    # cost uses C_bar compared to target; macro_strain not needed for C_bar evaluation
    def cost_single(self, C_target: torch.Tensor):
        dummy_eps = torch.zeros(6, dtype=torch.double, device=C_target.device)
        _, C_bar, _, _, _ = self.forward(dummy_eps)
        return torch.norm(C_target - C_bar) / (torch.norm(C_target) + 1e-12)

    # inspect alphas (computed)
    @torch.no_grad()
    def get_alpha_matrix(self):
        W = self.compute_weights()
        H_list, _, _ = self.compute_H_list()
        alpha = self.compute_alpha_matrix(W)
        return alpha, W


# ---------------------------------------------------------
# Dataset: returns (C1, C2, C_target) per sample
# Replace this with your real generator
# ---------------------------------------------------------
def get_dataset(num_samples):
    data = []
    # C_1 = data_loader.tangent_from_csv3d('msftrainingdataset/CFibers.csv')
    # C_2 = data_loader.tangent_from_csv3d('msftrainingdataset/CMatrix.csv')
    # C_3 = data_loader.tangent_from_csv3d('msftrainingdataset/CHomoFull.csv')
    C_1 = tangent_from_csv3d('msftrainingdataset2/CFibers.csv')
    C_2 = tangent_from_csv3d('msftrainingdataset2/CMatrix.csv')
    C_3 = tangent_from_csv3d('msftrainingdataset2/CHomoFull.csv')
    for _ in range(num_samples):
        print('@%$^$@#')
        print(C_1[_])
        print(C_2[_])
        print(C_3[_])
        data.append((C_1[_],C_2[_], C_3[_]))
    return data

def get_dataset_npz(num_samples,  F_Training_data_generation, F_Generated_training_data):
    data = []
    C_in = np.load(f'{F_Training_data_generation+ F_Generated_training_data}\\material_stiffness.npz')
    C_out = np.load(f'{F_Training_data_generation+ F_Generated_training_data}\\homo.npz')


    # C_3 = data_loader.tangent_from_csv3d('msftrainingdataset/CHomo.csv')
    # C_1 = data_loader.tangent_from_csv3d('trainingDataset/CFibers.csv')
    # C_2 = data_loader.tangent_from_csv3d('trainingDataset/CMatrix.csv')
    # C_3 = data_loader.tangent_from_csv3d('trainingDataset/ChomoFull.csv')
    discarded = 0
    for _ in range(num_samples):
        if C_out[f'{_}'][0,0] == 0:
            discarded += 1
            print('skipping')
        else:
            data.append((torch.tensor(C_in[f'mat1_{_}']),torch.tensor(C_in[f'mat2_{_}']), torch.tensor(C_in[f'mat3_{_}']), torch.tensor(C_out[f'{_}'])))
    num_samples = num_samples - discarded
    return data,num_samples


# ---------------------------------------------------------
# Print optimized params
# ---------------------------------------------------------
def print_optimized_params(model: IMNNet):
    print("\n================ OPTIMIZED PARAMS ================")
    print(f"Mode: {model.mode} | N={model.N}, M={model.M}")

    # nodes
    print("\n--- Nodes ---")
    for i, node in enumerate(model.nodes):
        print(f"Node {i:02d} name={node.name} type={node.node_type} "
              f"z={node.z.detach().cpu().item():+.6e} "
              f"W_raw={node.weight_raw().detach().cpu().item():+.6e} "
              f"amplify={node.amplify.detach().cpu().item():+.6e} "
              f"p(eq_plast)={node.p.detach().cpu().item():+.6e}")

    print("\nWeights W used (normalized even/odd):")
    print(model.compute_weights().detach().cpu().numpy())

    # joins
    print("\n--- Interactions ---")
    for j, join in enumerate(model.joins):
        theta = (torch.pi * torch.sigmoid(join.theta_raw)).detach().cpu().item()
        phi   = (2.0 * torch.pi * torch.sigmoid(join.phi_raw)).detach().cpu().item()
        print(f"Join {j:02d} Lj={join.Lj} alpha0={join.alpha0.detach().cpu().item():+.6e} "
              f"theta={theta:+.6e} phi={phi:+.6e}")
        print(f"   beta[{j}] = {join.beta.detach().cpu().numpy()}")

    # computed alpha matrix
    alpha, W = model.get_alpha_matrix()
    print("\nComputed alpha matrix (N x M):")
    print(alpha.detach().cpu().numpy())

    print("\nConstraint check: sum_i W_i * alpha_{i,j} (should be ~0)")
    for j in range(model.M):
        lhs = torch.sum(W * alpha[:, j]).item()
        print(f"  j={j:02d}: {lhs:+.3e}")



@torch.no_grad()
def load_model_parameters_from_npz(model, file_name: str, strict: bool = True):
    """
    Load model parameters from the .npz created by write_model_params_to_npz().
    This sets:
      - node.z, node.C, node.amplify, node_type (string), name (string)
      - join.theta_raw, join.phi_raw, join.beta
      - internal plastic state eps_p and p
    It does NOT store alpha (it is recomputed).
    """
    data = np.load(file_name, allow_pickle=True)

    N_file = int(np.array(data["N_nodes"]).reshape(-1)[0])
    M_file = int(np.array(data["M_interactions"]).reshape(-1)[0])

    if strict:
        if model.N != N_file:
            raise ValueError(f"N mismatch: model.N={model.N} vs file N={N_file}")
        if model.M != M_file:
            raise ValueError(f"M mismatch: model.M={model.M} vs file M={M_file}")

    # ---- node stuff ----
    z = data["z"]
    W = data["W"]  # not needed for assignment; just stored
    C_nodes = data["C_nodes"]
    node_amplify = data["node_amplify"]
    node_names = data["node_names"]
    node_types = data["node_types"]
    eps_p = data["eps_p"]
    p_eq = data["p_eq"]

    for i, node in enumerate(model.nodes):
        # trainable param
        node.z.copy_(torch.tensor(float(z[i]), dtype=torch.double))

        # non-optimizing buffers
        node.set_C(torch.tensor(C_nodes[i], dtype=torch.double))
        node.amplify.copy_(torch.tensor(float(node_amplify[i]), dtype=torch.double))

        # internal state
        node.eps_p.copy_(torch.tensor(eps_p[i], dtype=torch.double))
        node.p.copy_(torch.tensor(float(p_eq[i]), dtype=torch.double))

        # python attrs (strings)
        try:
            node.name = str(node_names[i])
        except Exception:
            pass
        try:
            node.node_type = str(node_types[i])
        except Exception:
            pass

    # ---- interaction stuff ----
    theta_raw = data["theta_raw"]
    phi_raw = data["phi_raw"]
    alpha0 = data["alpha0"]

    for j, join in enumerate(model.joins):
        join.theta_raw.copy_(torch.tensor(float(theta_raw[j]), dtype=torch.double))
        join.phi_raw.copy_(torch.tensor(float(phi_raw[j]), dtype=torch.double))

        # beta array per interaction
        beta_j = data[f"beta_{j}"]
        join.beta.copy_(torch.tensor(beta_j, dtype=torch.double))

        # alpha0 is a buffer in join (fixed), but we can restore it too:
        join.alpha0.copy_(torch.tensor(float(alpha0[j]), dtype=torch.double))

    # ---- (optional) restore/verify architecture ----
    # we won't overwrite model.V_list; but we can sanity-check in strict mode
    if strict:
        for j, Vj in enumerate(model.V_list):
            V_file = list(data[f"V_{j}"].astype(int))
            if list(Vj) != V_file:
                raise ValueError(f"V_list mismatch at j={j}: model={Vj}, file={V_file}")

    file_mode = str(np.array(data["mode"]).reshape(-1)[0]) if "mode" in data else "unknown"
    print(f"[IMN] Loaded parameters from {file_name} (file mode={file_mode})")


def write_model_params_to_npz(model: IMNNet, file_name: str):
    """
    Save all IMN model parameters and structure to a NumPy .npz file.
    This is compatible with the current object-based IMNNet.
    """

    data = {}

    # ---------------- basic sizes ----------------
    data["N_nodes"] = np.array([model.N])
    data["M_interactions"] = np.array([model.M])

    # ---------------- architecture ----------------
    for j, Vj in enumerate(model.V_list):
        data[f"V_{j}"] = np.array(Vj, dtype=np.int32)

    # ---------------- node-level parameters ----------------
    # z (trainable)
    z = np.array([node.z.detach().cpu().item() for node in model.nodes])
    data["z"] = z

    # weights (computed)
    W = model.compute_weights().detach().cpu().numpy()
    data["W"] = W

    # node metadata (strings → unicode arrays)
    data["node_names"] = np.array([node.name for node in model.nodes], dtype="<U64")
    data["node_types"] = np.array([node.node_type for node in model.nodes], dtype="<U32")

    # amplify (buffer)
    data["node_amplify"] = np.array(
        [node.amplify.detach().cpu().item() for node in model.nodes]
    )

    # stiffness matrices stored in nodes
    data["C_nodes"] = np.stack(
        [node.C.detach().cpu().numpy() for node in model.nodes], axis=0
    )  # (N,6,6)

    # plastic internal variables (safe even if elastic)
    data["eps_p"] = np.stack(
        [node.eps_p.detach().cpu().numpy() for node in model.nodes], axis=0
    )  # (N,6)
    data["p_eq"] = np.array(
        [node.p.detach().cpu().item() for node in model.nodes]
    )

    # ---------------- interaction-level parameters ----------------
    theta_raw = []
    phi_raw = []
    alpha0 = []

    for j, join in enumerate(model.joins):
        theta_raw.append(join.theta_raw.detach().cpu().item())
        phi_raw.append(join.phi_raw.detach().cpu().item())
        alpha0.append(join.alpha0.detach().cpu().item())

        # beta has variable length Lj-1
        data[f"beta_{j}"] = join.beta.detach().cpu().numpy()

    data["theta_raw"] = np.array(theta_raw)
    data["phi_raw"] = np.array(phi_raw)
    data["alpha0"] = np.array(alpha0)

    # ---------------- computed alpha matrix ----------------
    with torch.no_grad():
        alpha, _ = model.get_alpha_matrix()
        data["alpha"] = alpha.detach().cpu().numpy()  # (N,M)

    # ---------------- mode ----------------
    data["mode"] = np.array([model.mode], dtype="<U16")

    # ---------------- D_list ----------------
    H_list, _, _ = model.compute_H_list()
    D_list = model.build_D_list(alpha, H_list)

    np_D_list = np.array([D.detach().cpu().numpy() for D in D_list])
    data['D_list'] = np_D_list
    # ---------------- save ----------------
    if file_name.exists():
        print(f"[IMN] Overwriting {file_name}...")
    else:
        print(f"[IMN] Writing {file_name}...")
    np.savez(file_name, **data)



def write_model_params_to_txt(model, filename):
    data = {}

    # ---------------- basic sizes ----------------
    data["N_nodes"] = np.array([int(model.N)])
    data["M_interactions"] = np.array([int(model.M)])

    # ---------------- architecture ----------------
    for j, Vj in enumerate(model.V_list):
        Vj_int = [int(x) for x in Vj]
        data[f"V_{j}"] = np.array(Vj_int, dtype=np.int32)

    # ---------------- node-level parameters ----------------
    # z (trainable)
    z = np.array([node.z.detach().cpu().item() for node in model.nodes])
    data["z"] = z

    # weights (computed)
    W = model.compute_weights().detach().cpu().numpy()
    data["W"] = W

    # node metadata (strings → unicode arrays)
    # data["node_names"] = np.array([node.name for node in model.nodes], dtype="<U64")
    # data["node_types"] = np.array([node.node_type for node in model.nodes], dtype="<U32")

    # amplify (buffer)
    # data["node_amplify"] = np.array(
    #     [node.amplify.detach().cpu().item() for node in model.nodes]
    # )

    # stiffness matrices stored in nodes
    # for i in model.nodes:
    #     data[f"C_nodes_{i.name}"] = i.C.detach().cpu().numpy()


    # plastic internal variables (safe even if elastic)
    # data["eps_p"] = np.stack(
    #     [node.eps_p.detach().cpu().numpy() for node in model.nodes], axis=0
    # )  # (N,6)
    # data["p_eq"] = np.array(
    #     [node.p.detach().cpu().item() for node in model.nodes]
    # )

    # ---------------- interaction-level parameters ----------------
    # theta_raw = []
    # phi_raw = []
    # alpha0 = []
    #
    # for j, join in enumerate(model.joins):
    #     theta_raw.append(join.theta_raw.detach().cpu().item())
    #     phi_raw.append(join.phi_raw.detach().cpu().item())
    #     alpha0.append(join.alpha0.detach().cpu().item())
    #
    #     # beta has variable length Lj-1
    #     data[f"beta_{j}"] = join.beta.detach().cpu().numpy()
    #
    # data["theta_raw"] = np.array(theta_raw)
    # data["phi_raw"] = np.array(phi_raw)
    # data["alpha0"] = np.array(alpha0)

    # ---------------- computed alpha matrix ----------------
    with torch.no_grad():
        alpha, _ = model.get_alpha_matrix()
        data["alpha"] = alpha.detach().cpu().numpy()  # (N,M)

        # ---------------- D_list ----------------
    H_list, _, _ = model.compute_H_list()
    D_list = model.build_D_list(alpha, H_list)

    np_D_list = np.array([D.detach().cpu().numpy() for D in D_list])
    data['D_list'] = np_D_list

    # ---------------- mode ----------------
    # data["mode"] = np.array([model.mode], dtype="<U16")
    if filename.exists():
        print(f"[IMN] Overwriting {filename}...")
    else:
        print(f"[IMN] Writing {filename}...")
    with open(filename, "w") as f:
        def write_array(name, A):
            A = np.asarray(A, order="F")  # Fortran order
            shape = A.shape
            rank = len(shape)

            f.write(f"{name} {rank} " + " ".join(map(str, shape)) + "\n")

            if rank == 1:
                if A.dtype == np.float64:
                    f.write(" ".join(f"{x:.16e}" for x in A) + "\n\n")
                else:
                    f.write(" ".join(f"{x}" for x in A) + "\n\n")
            elif rank == 2:
                for i in range(shape[0]):
                    f.write(" ".join(f"{A[i,j]:.16e}" for j in range(shape[1])) + "\n")
                f.write("\n")
            elif rank == 3:

                for i in range(shape[0]):
                    for j in range(shape[1]):
                        f.write(" ".join(f"{A[i, j, k]:.16e}" for k in range(shape[2])) + "\n")

            else:
                raise ValueError("Only rank 1 or 2 supported")

        for k,v in data.items():

            try:
                write_array(k, v)
            except:
                pass

        f.write("\n")
        sum_1 = z[0::3].sum()
        sum_2 = z[1::3].sum()
        sum_3 = z[2::3].sum()
        f.write(f"{sum_1}\n")
        f.write(f"{sum_2}\n")
        f.write(f"{sum_3}\n")

        f.write("\n")
        sum_1 = W[0::3].sum()
        sum_2 = W[1::3].sum()
        sum_3 = W[2::3].sum()
        f.write(f"{sum_1}\n")
        f.write(f"{sum_2}\n")
        f.write(f"{sum_3}\n")




            # write_array("z", z)
            # write_array("nodes", np.array([len(z)]))
            # write_array("theta", theta)
            # write_array("phi", phi)


def npz_to_bin_shape_for_FEAP(npz_file: Path):
    data = np.load(npz_file)
    print(f"[IMN] Exporting FEAP files from {npz_file}")

    FEAP_path = npz_file.parent / "FEAP_files"
    FEAP_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Export D matrix
    # -------------------------------------------------
    if "D_list" not in data:
        raise KeyError("D_list not found in npz file")

    D = data["D_list"]  # expected shape (Nnodes,6,Ndof)
    D = np.asfortranarray(D, dtype=np.float64)
    #D.tofile(FEAP_path / "D.bin")
    D.ravel(order="F").tofile(FEAP_path / "D.bin")
    shape_D = np.array(D.shape, dtype=np.int32)
    shape_D.tofile(FEAP_path / "D.shape")

    print(f"  D.bin written, shape = {D.shape}")

    # -------------------------------------------------
    # Export weights
    # -------------------------------------------------
    if "W" not in data:
        raise KeyError("W not found in npz file")

    W = np.asfortranarray(data["W"], dtype=np.float64)

    W.ravel(order="F").tofile(FEAP_path / "w.bin")

    shape_W = np.array(W.shape, dtype=np.int32)
    shape_W.tofile(FEAP_path / "w.shape")

    print(f"  w.bin written, shape = {W.shape}")


    # -------------------------------------------------
    # Export material properties
    # -------------------------------------------------

    #mat = [1.elastic/2.plastic, E, nu, plastic_var, plastic_var, plastic_var]
    # mat2 = [3, 10, 5, 4, 0.3, 0.25, 0.27, 2, 1.5, 1.5]
    # mat1 = [3, 20, 15, 14, 0.3, 0.25, 0.27, 12, 8.5, 4.5]
    mat2 = [1, 1000, 0.2]
    mat1 = [1, 3000, 0.3]
    mat1 = np.asfortranarray(mat1, dtype=np.float64)
    mat1.ravel(order="F").tofile(FEAP_path / "mat1.bin")
    shape_mat1 = np.array(mat1.shape, dtype=np.int32)
    shape_mat1.tofile(FEAP_path / "mat1.shape")

    mat2 = np.asfortranarray(mat2, dtype=np.float64)
    mat2.ravel(order="F").tofile(FEAP_path / "mat2.bin")
    shape_mat2 = np.array(mat2.shape, dtype=np.int32)
    shape_mat2.tofile(FEAP_path / "mat2.shape")

    print("[IMN] FEAP export complete.")
    print("D_list shape:", data["D_list"].shape)
    print("W shape:", data["W"].shape)
    print("mat1 shape:", mat1.shape)
    print("mat2 shape:", mat2.shape)

def write_model_params_for_FEAP(model, filename):
    data = {}
    W = model.compute_weights().detach().cpu().numpy()
    data["W"] = W
    # ---------------- computed alpha matrix ----------------
    with torch.no_grad():
        alpha, _ = model.get_alpha_matrix()
        data["alpha"] = alpha.detach().cpu().numpy()  # (N,M)
    # ---------------- D_list ----------------
    H_list, _, _ = model.compute_H_list()
    D_list = model.build_D_list(alpha, H_list)
    np_D_list = np.array([D.detach().cpu().numpy() for D in D_list])
    data['D_list'] = np_D_list


    if filename.exists():
        print(f"[IMN] Overwriting {filename}...")
    else:
        print(f"[IMN] Writing {filename}...")
    with open(filename, "w") as f:
        f.write('Weights\n')
        f.write(str(len(data['W']))+'\n')
        for n,w in enumerate(data['W']):
            f.write(f"{n} {w}\n")
        f.write('D_matrix\n')
        f.write(str(np.shape(data['D_list']))[1:-1]+'\n')
        for i in range(data['D_list'].shape[0]):
            for j in data['D_list'][i].reshape(-1):
                f.write(f"{j} ")
            f.write('\n')


# ---------------------------------------------------------
# Training over multiple samples
# Each sample provides (C1, C2, C_target)
# ---------------------------------------------------------
def IMN_cohesives_run(N_layers, MODE,RUN_MODE,num_samples,num_epochs,inner_steps,lr, cost_live_plot, kinematics, training_data_folder
               , F_Training_data_generation, F_Generated_training_data):
    # ---- config ----
    path = Path(training_data_folder)
    path.mkdir(parents=True, exist_ok=True)
    NPZ_FILE_path = Path(training_data_folder + "/data.npz")  # For reading or writing IMN to npz
    TXT_FILE_path = Path(training_data_folder + "/data.txt")  # For reading or writing IMN to npz
    TXT_FILE_path_FEAP = Path(training_data_folder + "/data_FEAP.txt")
    kinematics = 'small'
    # model
    model = IMNNet(N_layers=N_layers, mode=MODE, kinematics=kinematics)
    model.assign_types_even_odd(type_even="elastic", type_odd="elastic")
    # set plastic parameters for odd nodes (optional)
    for i, node in enumerate(model.nodes):
        if i % 2 == 1:
            node.set_plastic_props(E=210e3, nu=0.3, sigma_y=250.0, H=1000.0)


    if RUN_MODE == "TRAIN":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # dataset
        #dataset = get_dataset(num_samples)
        dataset,num_samples = get_dataset_npz(num_samples, F_Training_data_generation, F_Generated_training_data)
        epoch_costs = []
        run_live_optimization(num_epochs, num_samples, dataset, inner_steps, optimizer, model, epoch_costs,cost_live_plot,training_data_folder)

        write_model_params_to_npz(model, NPZ_FILE_path)
        write_model_params_to_txt(model, TXT_FILE_path)
        npz_to_bin_shape_for_FEAP(NPZ_FILE_path)

    elif RUN_MODE == "SOLVE":
        load_model_parameters_from_npz(model, NPZ_FILE_path, strict=True)

        if kinematics == "small":

            E = 1000
            nu = 0.2
            C_1 = elastic_stiffness_from_Enu(float(E), float(nu))
            E = 1000
            nu = 0.2
            C_2 = elastic_stiffness_from_Enu(float(E), float(nu))
            model.assign_phases(C_1,C_2)
            eps_macro = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.double)  # (1,6)
            sigma_bar, C_bar, local_strains, local_stresses, U = model(eps_macro)
            print("\n=== INFERENCE RESULT ===")
            print("eps_macro:", eps_macro.detach().cpu().numpy())
            print("sigma_bar:", sigma_bar.detach().cpu().numpy())
            print("C_bar:", C_bar.detach().cpu().numpy())
            print("U:", U.detach().cpu().numpy())

        elif kinematics == "finite_PK":
            F_macro = torch.tensor([[1, 0, 0],[0, 0, 0],[0,0,0]], dtype=torch.double)  # (1,6)
            P_bar, S_bar6, C_bar_ref, local_E6, local_S6, U = model.forward_F(F_macro)
            print("\n=== INFERENCE RESULT ===")
            print("P_bar:", P_bar.detach().cpu().numpy())
            print("S_bar6:", S_bar6.detach().cpu().numpy())
            print("C_bar_ref:", C_bar_ref.detach().cpu().numpy())
            print("local_E6:", local_E6.detach().cpu().numpy())
            print("local_S6:", local_S6.detach().cpu().numpy())
            print("U:", U.detach().cpu().numpy())


    elif RUN_MODE == "Read_data":
        dataset = get_dataset(num_samples)

    else:
        raise ValueError("RUN_MODE must be 'TRAIN' or 'INFER'")


