from __future__ import annotations

from typing import Dict, Sequence, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Small linear algebra helpers
# -----------------------------------------------------------------------------
def H_matrix(n: Tensor) -> Tensor:
    """Return one 6x3 H(n) matrix."""
    nx, ny, nz = n[0], n[1], n[2]
    H = torch.zeros((6, 3), dtype=n.dtype, device=n.device)
    H[0, 0] = nx
    H[1, 1] = ny
    H[2, 2] = nz
    H[3, 0] = ny
    H[3, 1] = nx
    H[4, 0] = nz
    H[4, 2] = nx
    H[5, 1] = nz
    H[5, 2] = ny
    return H


def compute_alphas_for_join(W_children: Tensor, beta_j: Tensor, alpha0_j: Tensor) -> Tensor:
    """Compatibility helper for one join. Main IMN path uses vectorized layer-wise version."""
    if beta_j.shape[0] != W_children.shape[0] - 1:
        raise ValueError("beta_j must have length len(W_children) - 1.")

    W0 = W_children[0]
    denom = beta_j.sum() + 1e-12
    alpha = torch.zeros_like(W_children)
    alpha[0] = alpha0_j
    alpha[1:] = -(W0 * alpha0_j / (W_children[1:] + 1e-12)) * (beta_j / denom)
    return alpha


# -----------------------------------------------------------------------------
# IMN Calculator
# -----------------------------------------------------------------------------
class IMNCalculator:
    """
    Vectorized IMN homogenization calculator.

    Public API intentionally matches your previous IMNCalculator:
      - assign_node_stiffness(data)
      - homogenize_from_flat_params(flat_p)
      - normalized_frobenius_mse(...)
      - phase_fraction_error(...)
      - output_params_from_p_flat(...)

    Parameter layout used by homogenize_from_flat_params:
      flat_p = [W/z values (N), betas matrix (N*M), theta_phi values (2*M)]
      total dimension = N + N*M + 2*M

    Notes on speed/safety:
      - H_list is built locally, not stored on self, so cached IMNs do not keep
        old autograd graph history.
      - D_list assembly is vectorized.
      - K, Mmat, Y, and C_avg assembly are vectorized with einsum.
      - Node stiffness assignment is vectorized over repeated phase pattern.
    """

    def __init__(
        self,
        N_layers: int,
        phases: Sequence[str],
        nodes_per_mech_per_phase: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device if device is not None else "cpu")
        self.dtype = dtype
        self.N_layers = int(N_layers)
        self.nodes_per_mech_per_phase = int(nodes_per_mech_per_phase)
        self.configure_phases(phases)

    # ------------------------------------------------------------------
    # Architecture and phase configuration
    # ------------------------------------------------------------------
    def configure_phases(self, phases: Sequence[str]) -> None:
        phase_list = list(phases)
        n_phases = len(phase_list)

        # If only the phase names/order changed but the number of phases is the
        # same, N/M/V_list stay unchanged. Update node_phase metadata only.
        if hasattr(self, "n_phases") and n_phases == self.n_phases:
            self.phases = phase_list
            self.node_phase = [self.phases[i % self.n_phases] for i in range(self.N)]
            return

        self.phases = phase_list
        self.n_phases = n_phases

        self.N, self.V_list = self.architecture(
            self.N_layers,
            self.n_phases,
            self.nodes_per_mech_per_phase,
        )
        self.M = len(self.V_list)
        self.node_phase = [self.phases[i % self.n_phases] for i in range(self.N)]
        self.Lj_list = [len(Vj) for Vj in self.V_list]

        # Layer metadata lets compute_alpha_matrix vectorize over all joins in
        # one layer. Group size differs by layer, so we keep a small loop over
        # layers only, not over all joins/nodes.
        self.layer_specs: List[Tuple[int, int, int]] = []
        # tuple = (first_join_col, number_of_joins_in_layer, group_size)
        base_gap = self.n_phases * self.nodes_per_mech_per_phase
        first_col = 0
        for level in range(self.N_layers):
            group_size = base_gap * (2 ** level)
            num_groups = self.N // group_size
            self.layer_specs.append((first_col, num_groups, group_size))
            first_col += num_groups

        # Node phase indices follow the cyclic pattern used in your original IMN:
        # node i belongs to phases[i % n_phases].
        self.node_phase_indices = (torch.arange(self.N, device=self.device) % self.n_phases).long()

        # Reusable stiffness storage. It is intentionally not part of autograd.
        self.C_nodes = torch.zeros((self.N, 6, 6), dtype=self.dtype, device=self.device)

    @staticmethod
    def architecture(
        N_layers: int,
        n_phases: int,
        nodes_per_mech_per_phase: int,
    ) -> Tuple[int, List[List[int]]]:
        nodes = n_phases * nodes_per_mech_per_phase * 2 ** (N_layers - 1)

        # Must depend on nodes_per_mech_per_phase. Using n_phases * 2 only works
        # for nodes_per_mech_per_phase == 2.
        base_gap = n_phases * nodes_per_mech_per_phase
        gaps = [base_gap * (2 ** level) for level in range(N_layers)]

        V_list: List[List[int]] = []
        for gap in gaps:
            for start in range(0, nodes, gap):
                end = start + gap
                if end > nodes:
                    raise ValueError(
                        f"Invalid IMN architecture: nodes={nodes}, gap={gap}, "
                        f"start={start}, n_phases={n_phases}, "
                        f"nodes_per_mech_per_phase={nodes_per_mech_per_phase}, "
                        f"N_layers={N_layers}."
                    )
                V_list.append(list(range(start, end)))
        return nodes, V_list

    # ------------------------------------------------------------------
    # Data assignment and dimensions
    # ------------------------------------------------------------------
    def assign_node_stiffness(self, data: Dict[str, np.ndarray | Tensor]) -> None:
        """
        Assign phase stiffness matrices from sample data.

        This version stacks only the unique phase matrices, then expands them to
        all N nodes using node_phase_indices. That removes the previous loop over
        all nodes.
        """
        phase_mats = []
        for phase_name in self.phases:
            Ci = data[phase_name]
            if isinstance(Ci, np.ndarray):
                Ci = torch.from_numpy(Ci)
            phase_mats.append(Ci.to(device=self.device, dtype=self.dtype))

        phase_C = torch.stack(phase_mats, dim=0)  # (P, 6, 6)
        with torch.no_grad():
            self.C_nodes.copy_(phase_C[self.node_phase_indices])

    def create_flat_p(self, data):
        """Compatibility helper from your original file."""
        p = [float(v) for group in zip(*(data[phase]["weights"] for phase in self.phases)) for v in group]
        p.extend(
            [
                float(v)
                for group in zip(*(np.array(data[phase]["betas"]).flatten() for phase in self.phases))
                for v in group
            ]
        )
        p.extend([float(v) for group in zip(data["theta"], data["phi"]) for v in group])
        return torch.tensor(p, dtype=self.dtype, device=self.device)

    def param_dim(self) -> int:
        """Actual parameter dimension used by homogenize_from_flat_params."""
        return self.N + self.N * self.M + 2 * self.M

    def param_dim_w_beta(self) -> int:
        return self.N + self.N * self.M

    def param_dim_theta_phi(self) -> int:
        return 2 * self.M

    def param_dim_legacy(self) -> int:
        """Old layout: N + sum_j [2 + (Lj - 1)]. Kept only for debugging."""
        P = self.N
        for Lj in self.Lj_list:
            P += 2 + (Lj - 1)
        return P

    # ------------------------------------------------------------------
    # Core vectorized building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _weights_from_z(z: Tensor) -> Tensor:
        w_raw = F.softplus(z) + 1e-12
        return w_raw / (w_raw.sum() + 1e-12)

    def _build_H_list(self, theta_raw: Tensor, phi_raw: Tensor) -> Tensor:
        """Vectorized build of all M H matrices: returns (M, 6, 3)."""
        theta = torch.pi * torch.sigmoid(theta_raw)
        phi = 2.0 * torch.pi * torch.sigmoid(phi_raw)

        nx = torch.cos(phi) * torch.sin(theta)
        ny = torch.sin(phi) * torch.sin(theta)
        nz = torch.cos(theta)

        H = torch.zeros((self.M, 6, 3), dtype=theta_raw.dtype, device=theta_raw.device)
        H[:, 0, 0] = nx
        H[:, 1, 1] = ny
        H[:, 2, 2] = nz
        H[:, 3, 0] = ny
        H[:, 3, 1] = nx
        H[:, 4, 0] = nz
        H[:, 4, 2] = nx
        H[:, 5, 1] = nz
        H[:, 5, 2] = ny
        return H

    def compute_alpha_matrix(self, W: Tensor, betas: Tensor) -> Tensor:
        """
        Vectorized alpha assembly.

        Previous version looped over every join in V_list. This version loops only
        over N_layers. Within each layer, all joins have the same group size and
        can be processed in one tensor operation.
        """
        eps = 1e-12
        alpha = torch.zeros((self.N, self.M), dtype=W.dtype, device=W.device)

        for first_col, num_groups, group_size in self.layer_specs:
            # rows[g, :] are the child node indices for join g in this layer
            rows = torch.arange(self.N, device=W.device).view(num_groups, group_size)
            cols = torch.arange(first_col, first_col + num_groups, device=W.device).view(num_groups, 1)

            Wc = W[rows]                 # (G, Lj)
            beta_c = betas[rows, cols]   # (G, Lj), one beta column per join

            # alpha0 = beta_c[:, 0]        # (G,)
            alpha0 = torch.full_like(beta_c[:, 0], 0.5)
            denom = beta_c[:, 1:].sum(dim=1, keepdim=True) + eps

            alpha_c = torch.zeros_like(Wc)
            alpha_c[:, 0] = alpha0
            alpha_c[:, 1:] = -(
                Wc[:, :1] * alpha0[:, None] / (Wc[:, 1:] + eps)
            ) * (beta_c[:, 1:] / denom)

            alpha[rows, cols.expand_as(rows)] = alpha_c

        return alpha

    def build_D_list(self, alpha: Tensor, H_list: Tensor) -> Tensor:
        """
        Vectorized D assembly.

        alpha:  (N, M)
        H_list: (M, 6, 3)
        returns D_list: (N, 6, 3M)
        """
        D_blocks = alpha[:, :, None, None] * H_list[None, :, :, :]  # (N, M, 6, 3)
        return D_blocks.permute(0, 2, 1, 3).reshape(self.N, 6, 3 * self.M)

    # ------------------------------------------------------------------
    # Homogenization
    # ------------------------------------------------------------------
    def homogenize(self, W: Tensor, theta_raw: Tensor, phi_raw: Tensor, betas: Tensor) -> Tensor:
        W = W.to(device=self.device, dtype=self.dtype)
        theta_raw = theta_raw.to(device=self.device, dtype=self.dtype)
        phi_raw = phi_raw.to(device=self.device, dtype=self.dtype)
        betas = betas.to(device=self.device, dtype=self.dtype)

        H_list = self._build_H_list(theta_raw, phi_raw)
        alpha = self.compute_alpha_matrix(W, betas)
        D_list = self.build_D_list(alpha, H_list)  # (N, 6, dim)

        C_nodes = self.C_nodes
        sumW = W.sum() + 1e-12
        dim = 3 * self.M

        # C_i @ D_i for all nodes at once: (N, 6, 6) @ (N, 6, dim) -> (N, 6, dim)
        C_D = torch.bmm(C_nodes, D_list)

        # K = sum_i W_i D_i^T C_i D_i
        K = torch.einsum("n,nad,nae->de", W, D_list, C_D)

        # Mmat = sum_i W_i D_i^T C_i
        Mmat = torch.einsum("n,nad,nab->db", W, D_list, C_nodes)

        # Y = sum_i W_i C_i D_i
        Y = torch.einsum("n,nad->ad", W, C_D)

        # C_avg = sum_i W_i C_i
        C_avg = torch.einsum("n,nab->ab", W, C_nodes)

        Y = Y / sumW
        C_avg = C_avg / sumW

        eps = 1e-8 if self.dtype == torch.float32 else 1e-12
        K = K + eps * torch.eye(dim, dtype=K.dtype, device=K.device)
        # Safety for AMP: torch.linalg.solve does not support Half on CUDA
        # K = K.float()
        # Mmat = Mmat.float()
        # Y = Y.float()
        # C_avg = C_avg.float()
        #
        X = torch.linalg.solve(K, -Mmat)
        # return self.fix_predicted_matrix(C_avg + Y @ X)
        return C_avg + Y @ X

    def fix_predicted_matrix(self, C:Tensor) -> Tensor:
        C[0:3, 3:6] = 0
        C[3:6, 0:3] = 0
        C[3, 4] = 0
        C[3, 5] = 0
        C[4, 3] = 0
        C[4, 5] = 0
        C[5, 3] = 0
        C[5, 4] = 0
        return C


    def homogenize_from_flat_params(self, p_hat_1d: Tensor) -> Tensor:
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.view(-1)

        p = p_hat_1d.to(device=self.device, dtype=self.dtype)
        expected = self.param_dim()
        if p.numel() != expected:
            raise ValueError(
                f"flat_p has wrong size for this IMN architecture. "
                f"Got {p.numel()}, expected {expected}. "
                f"N={self.N}, M={self.M}, n_phases={self.n_phases}, "
                f"nodes_per_mech_per_phase={self.nodes_per_mech_per_phase}."
            )

        idx = 0

        W = p[idx : idx + self.N]
        idx += self.N

        betas_mat = p[idx : idx + self.M * self.N].view(self.N, self.M)
        idx += self.M * self.N

        theta_phi = p[idx : idx + 2 * self.M].view(self.M, 2)
        theta_raw = theta_phi[:, 0]
        phi_raw = theta_phi[:, 1]

        return self.homogenize(W, theta_raw, phi_raw, betas_mat)

    # ------------------------------------------------------------------
    # Diagnostics/losses
    # ------------------------------------------------------------------
    @staticmethod
    def normalized_frobenius_mse(C_pred: Tensor, C_tgt: Tensor, eps: float = 1e-12) -> Tensor:
        C_tgt = C_tgt.to(device=C_pred.device, dtype=C_pred.dtype)
        diff = C_pred - C_tgt
        return (diff * diff).sum() / ((C_tgt * C_tgt).sum() + eps)

    def phase_fraction_error(self, p_hat_1d: Tensor, W_phases: Sequence[float] | Tensor) -> Tensor:
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.view(-1)

        W = p_hat_1d[: self.N].to(device=self.device, dtype=self.dtype)
        W_target = torch.as_tensor(W_phases, dtype=W.dtype, device=W.device)

        if W_target.numel() != self.n_phases:
            raise ValueError(
                f"W_phases has {W_target.numel()} values, but this IMN has "
                f"{self.n_phases} phases."
            )

        # Node order is [phase0, phase1, ..., phaseP-1, phase0, phase1, ...]
        phase_sums = W.view(-1, self.n_phases).sum(dim=0)
        diff = phase_sums - W_target
        return (diff * diff).sum() / ((W_target * W_target).sum() + 1e-12)

    def output_params_from_p_flat(self, p_hat_1d: Tensor, FEAP_path) -> None:
        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.view(-1)

        p = p_hat_1d.detach().to(device=self.device, dtype=self.dtype)
        expected = self.param_dim()
        if p.numel() != expected:
            raise ValueError(f"flat_p has size {p.numel()}, expected {expected}.")

        W = p[: self.N]
        idx = self.N
        betas_mat = p[idx : idx + self.M * self.N].view(self.N, self.M)
        idx += self.M * self.N
        theta_phi = p[idx : idx + 2 * self.M].view(self.M, 2)


        H_list = self._build_H_list(theta_phi[:, 0], theta_phi[:, 1])
        alpha = self.compute_alpha_matrix(W, betas_mat)
        D_list = self.build_D_list(alpha, H_list)

        FEAP_path.mkdir(parents=True, exist_ok=True)

        D = np.asfortranarray(D_list.detach().cpu().numpy(), dtype=np.float64)
        D.ravel(order="F").tofile(FEAP_path / "D_0.bin")
        np.array(D.shape, dtype=np.int32).tofile(FEAP_path / "D_0.shape")

        W_np = np.asfortranarray(W.detach().cpu().numpy(), dtype=np.float64)
        W_np.ravel(order="F").tofile(FEAP_path / "w_0.bin")
        np.array(W_np.shape, dtype=np.int32).tofile(FEAP_path / "w_0.shape")


# -----------------------------------------------------------------------------
# Flat parameter packing helpers
# -----------------------------------------------------------------------------
def interleave_row_chunks(t: Tensor, g: int) -> Tensor:
    R, C = t.shape
    if C % g != 0:
        raise ValueError(f"Number of columns ({C}) must be divisible by chunk size g ({g}).")
    return t.view(R, C // g, g).permute(1, 0, 2).reshape(-1)


def pack_flat_p(z_by_phase: Tensor, beta_by_phase: Tensor, theta: Tensor, phi: Tensor, M: int) -> Tensor:
    z_flat = z_by_phase.squeeze(1).T.reshape(-1)
    beta_flat = interleave_row_chunks(beta_by_phase, M)
    theta_phi = torch.stack([theta, phi], dim=1).reshape(-1)
    return torch.cat([z_flat, beta_flat, theta_phi], dim=0)


def normalized_frobenius_mse_graph(C_pred: Tensor, C_tgt: Tensor, eps: float = 1e-12) -> Tensor:
    C_tgt = C_tgt.to(device=C_pred.device, dtype=C_pred.dtype)
    diff = C_pred - C_tgt
    return (diff * diff).sum() / ((C_tgt * C_tgt).sum() + eps)
