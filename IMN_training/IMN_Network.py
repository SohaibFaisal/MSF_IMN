from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= small linear algebra helpers =========
def H_matrix(n: torch.Tensor) -> torch.Tensor:
    """
    Return 6x3 H(n) used in IMN/DMN formulation.
    n: (3,) unit-ish vector
    """
    nx, ny, nz = n[0], n[1], n[2]
    H = torch.zeros((6, 3), dtype=torch.double, device=n.device)

    # Keep this identical to your current implementation unless you
    # have a different exact H-matrix formula in your theory/codebase.
    H[0, 0] = nx
    H[1, 1] = ny
    H[2, 2] = nz
    H[3, 0] = ny
    H[3, 1] = nx
    H[4, 1] = nz
    H[4, 2] = ny
    H[5, 0] = nz
    H[5, 2] = nx
    return H


def compute_alphas_for_join(
    W_children: torch.Tensor,
    beta_j: torch.Tensor,
    alpha0_j: torch.Tensor,
) -> torch.Tensor:
    """
    Build alpha values for one interaction/join.

    Parameters
    ----------
    W_children : (Lj,)
        Weights of the child nodes belonging to join j.
    beta_j : (Lj-1,)
        Positive beta values for children 1..Lj-1.
    alpha0_j : scalar
        Alpha value of child 0 for that join.
    """
    Lj = W_children.shape[0]
    if beta_j.shape[0] != Lj - 1:
        raise ValueError(f"beta_j must have length {Lj - 1}, got {beta_j.shape[0]}")

    W0 = W_children[0]
    denom = torch.sum(beta_j) + 1e-12

    a = torch.zeros_like(W_children)
    a[0] = alpha0_j
    a[1:] = -(W0 * alpha0_j / (W_children[1:] + 1e-12)) * (beta_j / denom)
    return a


def interleave_row_chunks(t: torch.Tensor, g: int) -> torch.Tensor:
    """
    Interleave row chunks of a 2D tensor.

    Example idea:
    [R, C] -> [R, C//g, g] -> [C//g, R, g] -> flatten
    """
    if t.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(t.shape)}")

    R, C = t.shape
    if C % g != 0:
        raise ValueError(f"Number of columns ({C}) must be divisible by chunk size g ({g}).")

    t = t.view(R, C // g, g)
    t = t.permute(1, 0, 2)
    return t.reshape(-1)


def pack_flat_p(
    z_by_phase: torch.Tensor,
    beta_by_phase: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    M: int,
) -> torch.Tensor:
    """
    Keep compatibility with your previous flat-parameter format.

    Parameters
    ----------
    z_by_phase : (P, n_leaf)
    beta_by_phase : (P, M, n_leaf)
    theta : (M,)
    phi : (M,)
    M : int
        Number of joins/interactions
    """
    z_flat = z_by_phase.squeeze(1).T.reshape(-1)
    beta_flat = interleave_row_chunks(beta_by_phase, M)
    theta_phi = torch.stack([theta, phi], dim=1).reshape(-1)
    flat_p = torch.cat([z_flat, beta_flat, theta_phi], dim=0)
    return flat_p


# ========= trainable IMN model =========
class IMNModel(nn.Module):
    """
    Trainable IMN network.

    This version owns the parameters itself:
      - z      : leaf/node weight logits
      - betas  : per-node, per-join beta logits
      - theta  : per-join angle logits
      - phi    : per-join angle logits

    forward() returns homogenized stiffness C_bar.
    """

    def __init__(
        self,
        N_layers: int,
        phases: Sequence[str],
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.double,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.phases = list(phases)
        self.n_phases = len(self.phases)
        self.N_layers = int(N_layers)
        self.dtype = dtype

        self.N, self.V_list = self.architecture(self.N_layers, self.n_phases)
        self.M = len(self.V_list)

        # cyclic node-phase assignment, same as your current code
        self.node_phase = [self.phases[i % self.n_phases] for i in range(self.N)]

        self.Lj_list: List[int] = []
        for Vj in self.V_list:
            self.Lj_list.append(len(Vj))

        # non-trainable stiffness data / working buffers
        self.register_buffer(
            "C_nodes",
            torch.zeros((self.N, 6, 6), dtype=self.dtype, device=device),
        )
        self.register_buffer(
            "H_buffer",
            torch.zeros((self.M, 6, 3), dtype=self.dtype, device=device),
        )

        # ===== trainable parameters =====
        # z -> mapped with softplus + normalization into physical weights W
        self.z = nn.Parameter(init_scale * torch.randn(self.N, dtype=self.dtype, device=device))

        # betas_raw -> mapped with softplus to keep betas positive
        self.betas_raw = nn.Parameter(
            init_scale * torch.randn(self.N, self.M, dtype=self.dtype, device=device)
        )

        # theta_raw, phi_raw are unconstrained logits, mapped to angles in _H_from_theta_phi
        self.theta_raw = nn.Parameter(
            init_scale * torch.randn(self.M, dtype=self.dtype, device=device)
        )
        self.phi_raw = nn.Parameter(
            init_scale * torch.randn(self.M, dtype=self.dtype, device=device)
        )

    # ========= architecture =========
    @staticmethod
    def architecture(N_layers: int, n_phases: int) -> Tuple[int, List[List[int]]]:
        """
        Same logic as your current architecture:
        nodes = phases * 2**N_layers
        V_list: list of child-index groups (joins)
        """
        Vj = []
        gaps = []
        nodes = n_phases * (2 ** N_layers)

        for _ in range(N_layers):
            if not gaps:
                gaps.append(n_phases * 2)
            else:
                gaps.append(gaps[-1] * 2)

        for gap in gaps:
            for start in range(0, nodes, gap):
                Vj.append(list(range(start, start + gap)))

        return nodes, Vj

    # ========= stiffness assignment =========
    def assign_node_stiffness(self, data: Dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        data: dict phase_key -> 6x6 stiffness
        Example keys: "MATRIX", "UD1", "UD2", ...
        """
        with torch.no_grad():
            for i in range(self.N):
                mat = self.node_phase[i]
                Ci = data[mat]
                if isinstance(Ci, np.ndarray):
                    Ci = torch.from_numpy(Ci)
                self.C_nodes[i].copy_(Ci.to(dtype=self.dtype, device=self.C_nodes.device))

    # ========= dimensions =========
    def param_dim_w_beta(self) -> int:
        """
        Flat format used here:
          z(N) + betas(N*M) + theta(M) + phi(M)
        """
        return self.N + self.N * self.M + 2 * self.M

    def param_dim_theta_phi(self) -> int:
        return 2 * self.M

    # ========= constrained parameter mappings =========
    @staticmethod
    def _weights_from_z(z: torch.Tensor) -> torch.Tensor:
        """
        z: (N,) -> W: (N,), positive and normalized
        """
        w_raw = F.softplus(z.to(dtype=torch.double)) + 1e-12
        return w_raw / (w_raw.sum() + 1e-12)

    @staticmethod
    def _betas_from_raw(betas_raw: torch.Tensor) -> torch.Tensor:
        """
        Keep betas positive for stability.
        """
        return F.softplus(betas_raw.to(dtype=torch.double)) + 1e-12

    @staticmethod
    def _H_from_theta_phi(theta_raw: torch.Tensor, phi_raw: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained raw variables to spherical angles:
          theta in [0, pi]
          phi   in [0, 2*pi]
        """
        theta = torch.pi * torch.sigmoid(theta_raw)
        phi = 2.0 * torch.pi * torch.sigmoid(phi_raw)

        n = torch.stack(
            [
                torch.cos(phi) * torch.sin(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(theta),
            ]
        ).to(dtype=torch.double)

        return H_matrix(n)

    def build_H_list(self, theta_raw: torch.Tensor, phi_raw: torch.Tensor) -> torch.Tensor:
        """
        theta_raw, phi_raw: (M,)
        returns: (M, 6, 3)
        """
        H_list = self.H_buffer.clone()
        for j in range(self.M):
            H_list[j, :, :] = self._H_from_theta_phi(theta_raw[j], phi_raw[j])
        return H_list

    def compute_alpha_matrix(
        self,
        W: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        """
        alpha: (N, M)
        """
        device = W.device
        alpha = torch.zeros((self.N, self.M), dtype=self.dtype, device=device)
        W = W.to(dtype=self.dtype)

        for j, Vj in enumerate(self.V_list):
            idx = torch.tensor(Vj, dtype=torch.long, device=device)
            Wc = W[idx]                     # (Lj,)
            beta_full = betas[idx, j]      # (Lj,)
            alpha0_j = beta_full[0]        # scalar
            beta_j = beta_full[1:]         # (Lj-1,)
            alpha_children = compute_alphas_for_join(Wc, beta_j, alpha0_j)
            alpha[idx, j] = alpha_children

        return alpha

    def build_D_list(self, alpha: torch.Tensor, H_list: torch.Tensor) -> torch.Tensor:
        """
        returns: (N, 6, 3M)
        """
        device = alpha.device
        D_list = torch.zeros((self.N, 6, 3 * self.M), dtype=self.dtype, device=device)

        for i in range(self.N):
            Di = torch.zeros((6, 3 * self.M), dtype=self.dtype, device=device)
            for j in range(self.M):
                Di[:, 3 * j:3 * (j + 1)] = alpha[i, j] * H_list[j, :, :]
            D_list[i, :, :] = Di

        return D_list

    # ========= main homogenization =========
    def homogenize(
        self,
        W: torch.Tensor,
        theta_raw: torch.Tensor,
        phi_raw: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main solver. Returns C_bar (6x6).
        """
        device = W.device
        H_list = self.build_H_list(theta_raw, phi_raw)
        alpha = self.compute_alpha_matrix(W, betas)
        D_list = self.build_D_list(alpha, H_list)

        sumW = W.sum() + 1e-12
        dim = 3 * self.M

        K = torch.zeros((dim, dim), dtype=self.dtype, device=device)
        Mmat = torch.zeros((dim, 6), dtype=self.dtype, device=device)
        Y = torch.zeros((6, dim), dtype=self.dtype, device=device)
        C_avg = torch.zeros((6, 6), dtype=self.dtype, device=device)

        for i in range(self.N):
            Di = D_list[i, :, :].to(self.dtype)
            Ci = self.C_nodes[i].to(self.dtype)
            Wi = W[i].to(self.dtype)


            K += Wi * (Di.T @ (Ci @ Di))
            Mmat += Wi * (Di.T @ Ci)
            Y += Wi * (Ci @ Di)
            C_avg += Wi * Ci

        Y = Y / sumW
        C_avg = C_avg / sumW

        eps = 1e-12
        K = K + eps * torch.eye(dim, dtype=K.dtype, device=K.device)
        X = torch.linalg.solve(K, -Mmat)
        C_bar = C_avg + Y @ X
        return C_bar

    # ========= forward =========
    def forward(self) -> torch.Tensor:
        """
        Forward pass of the trainable IMN network.
        """
        W = self._weights_from_z(self.z)
        betas = self._betas_from_raw(self.betas_raw)
        return self.homogenize(W, self.theta_raw, self.phi_raw, betas)

    # ========= flat-parameter compatibility =========
    def get_flat_trainable_parameters(self) -> torch.Tensor:
        """
        Returns current trainable state in flat format:
          z(N) + betas_raw(N*M) + theta_raw(M) + phi_raw(M)
        """
        return torch.cat(
            [
                self.z.reshape(-1),
                self.betas_raw.reshape(-1),
                self.theta_raw.reshape(-1),
                self.phi_raw.reshape(-1),
            ],
            dim=0,
        )

    def set_flat_trainable_parameters_(self, p_hat_1d: torch.Tensor) -> None:
        """
        Load trainable state from a flat tensor in the format:
          z(N) + betas_raw(N*M) + theta_raw(M) + phi_raw(M)
        """
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.reshape(-1)

        expected = self.param_dim_w_beta()
        if p_hat_1d.numel() != expected:
            raise ValueError(f"Expected flat parameter length {expected}, got {p_hat_1d.numel()}")

        idx = 0
        with torch.no_grad():
            self.z.copy_(p_hat_1d[idx:idx + self.N].view_as(self.z).to(self.z))
            idx += self.N

            self.betas_raw.copy_(
                p_hat_1d[idx:idx + self.N * self.M].view_as(self.betas_raw).to(self.betas_raw)
            )
            idx += self.N * self.M

            self.theta_raw.copy_(
                p_hat_1d[idx:idx + self.M].view_as(self.theta_raw).to(self.theta_raw)
            )
            idx += self.M

            self.phi_raw.copy_(
                p_hat_1d[idx:idx + self.M].view_as(self.phi_raw).to(self.phi_raw)
            )

    def homogenize_from_flat_params(self, p_hat_1d: torch.Tensor) -> torch.Tensor:
        """
        Compatibility helper:
        input flat format = z(N) + betas_raw(N*M) + theta_raw(M) + phi_raw(M)
        """
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.reshape(-1)

        expected = self.param_dim_w_beta()
        if p_hat_1d.numel() != expected:
            raise ValueError(f"Expected flat parameter length {expected}, got {p_hat_1d.numel()}")

        idx = 0
        z = p_hat_1d[idx:idx + self.N].to(dtype=self.dtype)
        idx += self.N

        betas_raw = p_hat_1d[idx:idx + self.N * self.M].view(self.N, self.M).to(dtype=self.dtype)
        idx += self.N * self.M

        theta_raw = p_hat_1d[idx:idx + self.M].to(dtype=self.dtype)
        idx += self.M

        phi_raw = p_hat_1d[idx:idx + self.M].to(dtype=self.dtype)

        W = self._weights_from_z(z)
        betas = self._betas_from_raw(betas_raw)
        return self.homogenize(W, theta_raw, phi_raw, betas)

    # ========= diagnostics =========
    @staticmethod
    def normalized_frobenius_mse(
        C_pred: torch.Tensor,
        C_tgt: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        diff = C_pred - C_tgt
        num = (diff * diff).sum()
        den = (C_tgt * C_tgt).sum() + eps
        return num / den

    def current_phase_fractions(self) -> torch.Tensor:
        """
        Returns phase fractions implied by current trainable z.
        """
        W = self._weights_from_z(self.z)
        p = self.n_phases
        return torch.stack([W[x::p].sum() for x in range(p)])

    def phase_fraction_error(self, W_phases: Sequence[float]) -> torch.Tensor:
        """
        Compare current IMN phase fractions against target phase fractions.
        """
        W = self._weights_from_z(self.z)
        W_target = torch.as_tensor(W_phases, dtype=W.dtype, device=W.device)
        p = self.n_phases
        phase_sums = torch.stack([W[x::p].sum() for x in range(p)])

        diff = phase_sums - W_target
        num = (diff * diff).sum()
        den = (W_target * W_target).sum() + 1e-12
        return num / den

    def phase_fraction_error_from_flat_params(
        self,
        p_hat_1d: torch.Tensor,
        W_phases: Sequence[float],
    ) -> torch.Tensor:
        """
        Compatibility helper using a provided flat parameter vector.
        """
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.reshape(-1)

        z = p_hat_1d[:self.N].to(dtype=self.dtype)
        W = self._weights_from_z(z)

        W_target = torch.as_tensor(W_phases, dtype=W.dtype, device=W.device)
        p = self.n_phases
        phase_sums = torch.stack([W[x::p].sum() for x in range(p)])

        diff = phase_sums - W_target
        num = (diff * diff).sum()
        den = (W_target * W_target).sum() + 1e-12
        return num / den

    # ========= export helpers =========
    def output_params(self, FEAP_path: Path | str) -> None:
        """
        Export current model-implied W and D to binary files.
        """
        FEAP_path = Path(FEAP_path)
        FEAP_path.mkdir(parents=True, exist_ok=True)

        W = self._weights_from_z(self.z)
        betas = self._betas_from_raw(self.betas_raw)
        H_list = self.build_H_list(self.theta_raw, self.phi_raw)
        alpha = self.compute_alpha_matrix(W, betas)
        D_list = self.build_D_list(alpha, H_list)

        print("Weights----------------------------------------------------------------------------")
        ppp = int(self.N / (2 ** self.N_layers))
        for pppp in range(ppp):
            print(W[pppp::ppp].sum().item())

        # D = np.asfortranarray(D_list.detach().cpu().numpy(), dtype=np.float64)
        D = np.asfortranarray(D_list.detach().cpu().numpy(), dtype=np.float64)
        D.ravel(order="F").tofile(FEAP_path / "D_0.bin")
        shape_D = np.array(D.shape, dtype=np.int32)
        shape_D.tofile(FEAP_path / f"D_0.shape")
        # np.array(D.shape, dtype=np.int32).tofile(FEAP_path / "D_0.shape")
        print(f"D_0.bin written, shape = {D.shape}")

        W_np = np.asfortranarray(W.detach().cpu().numpy(), dtype=np.float64)
        W_np.ravel(order="F").tofile(FEAP_path / "w_0.bin")
        shape_W = np.array(W.shape, dtype=np.int32)
        shape_W.tofile(FEAP_path / "w_0.shape")
        # np.array(W_np.shape, dtype=np.int32).tofile(FEAP_path / "w_0.shape")

    # def output_params_from_p_flat(self, p_hat_1d: torch.Tensor, FEAP_path: Path | str) -> None:
    #     """
    #     Compatibility export using a provided flat parameter vector.
    #     """
    #     FEAP_path = Path(FEAP_path)
    #     FEAP_path.mkdir(parents=True, exist_ok=True)
    #
    #     if p_hat_1d.ndim != 1:
    #         p_hat_1d = p_hat_1d.reshape(-1)
    #
    #     idx = 0
    #     z = p_hat_1d[idx:idx + self.N].to(dtype=self.dtype)
    #     idx += self.N
    #
    #     betas_raw = p_hat_1d[idx:idx + self.N * self.M].view(self.N, self.M).to(dtype=self.dtype)
    #     idx += self.N * self.M
    #
    #     theta_raw = p_hat_1d[idx:idx + self.M].to(dtype=self.dtype)
    #     idx += self.M
    #
    #     phi_raw = p_hat_1d[idx:idx + self.M].to(dtype=self.dtype)
    #
    #     W = self._weights_from_z(z)
    #     betas = self._betas_from_raw(betas_raw)
    #
    #     print("Weights----------------------------------------------------------------------------")
    #     ppp = int(self.N / (2 ** self.N_layers))
    #     for pppp in range(ppp):
    #         print(W[pppp::ppp].sum().item())
    #
    #     H_list = self.build_H_list(theta_raw, phi_raw)
    #     alpha = self.compute_alpha_matrix(W, betas)
    #     D_list = self.build_D_list(alpha, H_list)
    #
    #     D = np.asfortranarray(D_list.detach().cpu().numpy(), dtype=np.float64)
    #     D.ravel(order="F").tofile(FEAP_path / "D_0.bin")
    #     np.array(D.shape, dtype=np.int32).tofile(FEAP_path / "D_0.shape")
    #     print(f"D_0.bin written, shape = {D.shape}")
    #
    #     W_np = np.asfortranarray(W.detach().cpu().numpy(), dtype=np.float64)
    #     W_np.ravel(order="F").tofile(FEAP_path / "w_0.bin")
    #     np.array(W_np.shape, dtype=np.int32).tofile(FEAP_path / "w_0.shape")

    # ========= optional initializer from old-style data =========
    # def initialize_from_data_dict(self, data: Dict) -> None:
    #     """
    #     Optional helper to initialize the trainable parameters from a dict
    #     shaped like your previous create_flat_p input.
    #
    #     Expected structure:
    #         data[phase]['weights']
    #         data[phase]['betas']
    #         data['theta']
    #         data['phi']
    #
    #     Notes
    #     -----
    #     - weights are converted back to z approximately via inverse-softplus-ish log(exp(w)-1)
    #       after small clamping.
    #     - betas are converted back to betas_raw similarly.
    #     """
    #     flat_p = self.create_flat_p(data)
    #
    #     idx = 0
    #     w_init = flat_p[idx:idx + self.N].to(dtype=self.dtype)
    #     idx += self.N
    #
    #     beta_init = flat_p[idx:idx + self.N * self.M].view(self.N, self.M).to(dtype=self.dtype)
    #     idx += self.N * self.M
    #
    #     theta_init = flat_p[idx:idx + self.M].to(dtype=self.dtype)
    #     idx += self.M
    #
    #     phi_init = flat_p[idx:idx + self.M].to(dtype=self.dtype)
    #
    #     def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    #         y = torch.clamp(y, min=1e-12)
    #         return torch.log(torch.expm1(y))
    #
    #     with torch.no_grad():
    #         # old file used raw z as weights in some places, but for a trainable
    #         # model we want z to be unconstrained logits whose mapped weights are positive/normalized
    #         w_norm = torch.clamp(w_init / (w_init.sum() + 1e-12), min=1e-12)
    #         self.z.copy_(inverse_softplus(w_norm))
    #
    #         self.betas_raw.copy_(inverse_softplus(torch.clamp(beta_init, min=1e-12)))
    #         self.theta_raw.copy_(theta_init)
    #         self.phi_raw.copy_(phi_init)
    #
    # def create_flat_p(self, data: Dict) -> torch.Tensor:
    #     """
    #     Preserve your old helper for compatibility with existing code/data layout.
    #
    #     Layout:
    #       z(N) + betas(N*M) + theta(M) + phi(M)
    #     """
    #     p = [
    #         float(v)
    #         for group in zip(*(data[phase]["weights"] for phase in self.phases))
    #         for v in group
    #     ]
    #
    #     p.extend(
    #         [
    #             float(v)
    #             for group in zip(
    #                 *(np.array(data[phase]["betas"]).flatten() for phase in self.phases)
    #             )
    #             for v in group
    #         ]
    #     )
    #
    #     p.extend([float(v) for v in data["theta"]])
    #     p.extend([float(v) for v in data["phi"]])
    #
    #     return torch.tensor(p, dtype=self.dtype)


# if __name__ == "__main__":
#     # Example
#     phases = ["MATRIX", "UD1"]
#     N_layers = 2
#
#     model = IMNModel(N_layers=N_layers, phases=phases).double()
#
#     # Example stiffness assignment
#     dummy_data = {
#         "MATRIX": np.eye(6) * 3.0,
#         "UD1": np.eye(6) * 10.0,
#     }
#     model.assign_node_stiffness(dummy_data)
#
#     # Forward pass
#     C_pred = model()
#     print("C_pred shape:", C_pred.shape)
#
#     # Example loss
#     C_target = torch.eye(6, dtype=torch.double) * 5.0
#     loss_C = model.normalized_frobenius_mse(C_pred, C_target)
#
#     # Optional phase fraction penalty
#     W_target = [0.6, 0.4]
#     loss_phase = model.phase_fraction_error(W_target)
#
#     loss = loss_C + 0.1 * loss_phase
#     print("loss:", loss.item())
#
#     # Example optimization step
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()