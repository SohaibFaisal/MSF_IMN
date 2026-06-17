from __future__ import annotations

from typing import Dict, Sequence, Tuple, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, Batch
except Exception:  # allows importing this file without torch_geometric installed
    Data = Any
    Batch = Any

Tensor = torch.Tensor


# =============================================================================
# 3D Mandel rotation helpers
# Mandel order: [11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12]
# =============================================================================

def _set_block(A: Tensor, rows: Sequence[int], B: Tensor) -> None:
    idx = torch.as_tensor(rows, device=A.device, dtype=torch.long)
    A[idx[:, None], idx[None, :]] = B


def rp_mandel(theta: Tensor) -> Tensor:
    """3x3 in-plane second-order tensor rotation in Mandel notation."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    rt2 = torch.sqrt(torch.as_tensor(2.0, dtype=theta.dtype, device=theta.device))

    R = torch.empty((3, 3), dtype=theta.dtype, device=theta.device)
    R[0, 0] = c * c
    R[0, 1] = s * s
    R[0, 2] = rt2 * s * c
    R[1, 0] = s * s
    R[1, 1] = c * c
    R[1, 2] = -rt2 * s * c
    R[2, 0] = -rt2 * s * c
    R[2, 1] = rt2 * s * c
    R[2, 2] = c * c - s * s
    return R


def rv(theta: Tensor) -> Tensor:
    """2x2 vector rotation matrix."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.empty((2, 2), dtype=theta.dtype, device=theta.device)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def X_mandel(alpha: Tensor) -> Tensor:
    X = torch.zeros((6, 6), dtype=alpha.dtype, device=alpha.device)
    X[0, 0] = 1.0
    _set_block(X, [1, 2, 3], rp_mandel(alpha))
    _set_block(X, [4, 5], rv(alpha))
    return X


def Y_mandel(beta: Tensor) -> Tensor:
    Y = torch.zeros((6, 6), dtype=beta.dtype, device=beta.device)
    Y[1, 1] = 1.0
    _set_block(Y, [0, 2, 4], rp_mandel(-beta))
    _set_block(Y, [3, 5], rv(-beta))
    return Y


def Z_mandel(gamma: Tensor) -> Tensor:
    Z = torch.zeros((6, 6), dtype=gamma.dtype, device=gamma.device)
    Z[2, 2] = 1.0
    _set_block(Z, [0, 1, 5], rp_mandel(gamma))
    _set_block(Z, [3, 4], rv(gamma))
    return Z


def rotate_stiffness_3d(C: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    """
    Rotate a 6x6 small-strain stiffness matrix in Mandel notation.

    C_a  = X(-a) C X(a)
    C_ab = Y(-b) C_a Y(b)
    Cbar = Z(-g) C_ab Z(g)
    """
    C = X_mandel(-alpha) @ C @ X_mandel(alpha)
    C = Y_mandel(-beta) @ C @ Y_mandel(beta)
    C = Z_mandel(-gamma) @ C @ Z_mandel(gamma)
    return 0.5 * (C + C.T)


# =============================================================================
# 3D DMN two-layer linear-elastic building block
# =============================================================================

def homogenize_two_layer_3d(C1: Tensor, C2: Tensor, w1: Tensor, w2: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Analytical 3D two-layer DMN building block.

    C1, C2: child stiffness matrices, shape (6, 6), Mandel notation.
    w1, w2: child weights.

    Local interface normal is direction 3. Therefore:
      constrained strain components: [11, 22, 12] -> indices [0, 1, 5]
      solved strain components:      [33, 23, 13] -> indices [2, 3, 4]

    Returns the homogenized stiffness before the block rotation.
    """
    dtype, device = C1.dtype, C1.device
    tiny = torch.as_tensor(eps, dtype=dtype, device=device)

    f1 = w1 / (w1 + w2 + tiny)
    f2 = 1.0 - f1

    dC = C2 - C1
    Chat = f2 * C1 + f1 * C2

    constrained = torch.tensor([0, 1, 5], device=device, dtype=torch.long)
    unknown = torch.tensor([2, 3, 4], device=device, dtype=torch.long)

    K = Chat[unknown[:, None], unknown[None, :]]
    B = torch.zeros((3, 6), dtype=dtype, device=device)
    B[:, constrained] = f2 * dC[unknown[:, None], constrained[None, :]]
    B[:, unknown] = C2[unknown[:, None], unknown[None, :]]

    reg = 1e-8 if dtype == torch.float32 else 1e-12
    K = K + reg * torch.eye(3, dtype=dtype, device=device)
    S_unknown = torch.linalg.solve(K, B)

    S1 = torch.zeros((6, 6), dtype=dtype, device=device)
    S1[0, 0] = 1.0
    S1[1, 1] = 1.0
    S1[5, 5] = 1.0
    S1[unknown, :] = S_unknown

    C = C2 - f1 * (dC @ S1)
    return 0.5 * (C + C.T)


# =============================================================================
# Trainable 3D DMN calculator
# =============================================================================

class DMNCalculator3D(nn.Module):
    """
    Differentiable, trainable small-strain 3D Deep Material Network.

    Parameter layout:
      flat_p = [z_bottom, alpha_all, beta_all, gamma_all]

    Dimensions:
      bottom_nodes = 2**(N_layers - 1)
      total_nodes  = 2**N_layers - 1
      param_dim    = bottom_nodes + 3*total_nodes = 7*2**(N_layers-1) - 3

    Important usage:
      - get_flat_params() returns the current trainable base DMN vector.
      - homogenize_from_flat_params(p) evaluates the DMN using p without copying
        p into nn.Parameters. This keeps end-to-end gradients intact.
      - assign_node_stiffness(data) sets the current phase stiffnesses for a sample.
    """

    def __init__(
        self,
        N_layers: int,
        phases: Sequence[str],
        device=None,
        dtype: torch.dtype = torch.float32,
        weight_activation: str = "softplus",
        init_angle_scale: float = 0.1,
    ):
        super().__init__()
        if N_layers < 1:
            raise ValueError("N_layers must be >= 1.")

        self.device0 = torch.device(device if device is not None else "cpu")
        self.dtype0 = dtype
        self.N_layers = int(N_layers)
        self.weight_activation = str(weight_activation).lower()
        self.init_angle_scale = float(init_angle_scale)

        self.configure_phases(phases)

        # Trainable base DMN parameters. These are optimized directly if included
        # in the optimizer, and can also be used as the base vector for a hyper-MLP.
        self.z = nn.Parameter(torch.empty(self.bottom_nodes, dtype=dtype, device=self.device0))
        self.alpha = nn.Parameter(torch.empty(self.total_nodes, dtype=dtype, device=self.device0))
        self.beta = nn.Parameter(torch.empty(self.total_nodes, dtype=dtype, device=self.device0))
        self.gamma = nn.Parameter(torch.empty(self.total_nodes, dtype=dtype, device=self.device0))

        # Current sample phase stiffnesses. It is a buffer, not a parameter.
        self.register_buffer("C_leaves", torch.zeros(self.bottom_nodes, 6, 6, dtype=dtype, device=self.device0))

        self.reset_parameters()

    def configure_phases(self, phases: Sequence[str]) -> None:
        self.phases = list(phases)
        if len(self.phases) < 1:
            raise ValueError("At least one phase is required.")
        self.n_phases = len(self.phases)

        self.bottom_nodes = 2 ** (self.N_layers - 1)
        self.total_nodes = 2 ** self.N_layers - 1
        self.n_internal = self.total_nodes - self.bottom_nodes
        self.bottom_start = self.n_internal
        self.bottom_indices = list(range(self.bottom_start, self.total_nodes))

        self.layers: List[List[int]] = []
        for level in range(self.N_layers):
            start = 2**level - 1
            end = 2 ** (level + 1) - 1
            self.layers.append(list(range(start, end)))

        leaf_phase_idx = torch.arange(self.bottom_nodes, device=self.device0) % self.n_phases
        self.register_buffer("leaf_phase_indices", leaf_phase_idx.long())
        self.leaf_phase = [self.phases[i % self.n_phases] for i in range(self.bottom_nodes)]

    def reset_parameters(self) -> None:
        # Positive-ish initial z values help avoid dead ReLU nodes if ReLU is used.
        nn.init.uniform_(self.z, 0.2, 0.8)
        nn.init.uniform_(self.alpha, -self.init_angle_scale, self.init_angle_scale)
        nn.init.uniform_(self.beta, -self.init_angle_scale, self.init_angle_scale)
        nn.init.uniform_(self.gamma, -self.init_angle_scale, self.init_angle_scale)

    # -------------------------------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------------------------------
    def param_dim(self) -> int:
        return self.bottom_nodes + 3 * self.total_nodes

    def param_dim_z(self) -> int:
        return self.bottom_nodes

    def param_dim_angles(self) -> int:
        return 3 * self.total_nodes

    def get_flat_params(self) -> Tensor:
        """Return differentiable flat trainable DMN base parameters."""
        return torch.cat([
            self.z.reshape(-1),
            self.alpha.reshape(-1),
            self.beta.reshape(-1),
            self.gamma.reshape(-1),
        ], dim=0)

    def parse_flat_params(self, p: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if p.ndim != 1:
            p = p.reshape(-1)
        p = p.to(device=self.z.device, dtype=self.z.dtype)
        if p.numel() != self.param_dim():
            raise ValueError(f"flat_p has {p.numel()} values, expected {self.param_dim()}.")

        idx = 0
        z = p[idx: idx + self.bottom_nodes]
        idx += self.bottom_nodes
        alpha = p[idx: idx + self.total_nodes]
        idx += self.total_nodes
        beta = p[idx: idx + self.total_nodes]
        idx += self.total_nodes
        gamma = p[idx: idx + self.total_nodes]
        return z, alpha, beta, gamma

    def create_flat_p(self, data: Dict[str, Any]) -> Tensor:
        z_key = "z" if "z" in data else "weights"
        z = torch.as_tensor(data[z_key], dtype=self.z.dtype, device=self.z.device).reshape(-1)
        alpha = torch.as_tensor(data["alpha"], dtype=self.z.dtype, device=self.z.device).reshape(-1)
        beta = torch.as_tensor(data["beta"], dtype=self.z.dtype, device=self.z.device).reshape(-1)
        gamma = torch.as_tensor(data["gamma"], dtype=self.z.dtype, device=self.z.device).reshape(-1)
        p = torch.cat([z, alpha, beta, gamma], dim=0)
        if p.numel() != self.param_dim():
            raise ValueError(f"Created flat_p has {p.numel()} values, expected {self.param_dim()}.")
        return p

    # -------------------------------------------------------------------------
    # Sample stiffness assignment
    # -------------------------------------------------------------------------
    def assign_node_stiffness(self, data: Dict[str, np.ndarray | Tensor]) -> None:
        """
        Assign phase stiffness matrices to bottom-layer leaves for the current sample.

        data must contain one 6x6 stiffness matrix per phase name.
        Example: data = {"MATRIX": C_matrix, "INCLUSION": C_inclusion}
        """
        phase_mats = []
        for phase_name in self.phases:
            C = data[phase_name]
            if isinstance(C, np.ndarray):
                C = torch.from_numpy(C)
            C = C.to(device=self.C_leaves.device, dtype=self.C_leaves.dtype)
            if C.shape != (6, 6):
                raise ValueError(f"Phase {phase_name!r} stiffness must be (6,6), got {tuple(C.shape)}.")
            phase_mats.append(0.5 * (C + C.T))

        phase_C = torch.stack(phase_mats, dim=0)
        # Copying sample data into a buffer is fine; the DMN parameters remain differentiable.
        with torch.no_grad():
            self.C_leaves.copy_(phase_C[self.leaf_phase_indices])

    # -------------------------------------------------------------------------
    # Core forward DMN operations
    # -------------------------------------------------------------------------
    def weights_from_z(self, z: Tensor) -> Tensor:
        if self.weight_activation == "relu":
            return F.relu(z)
        if self.weight_activation == "softplus":
            return F.softplus(z) + 1e-12
        if self.weight_activation == "exp":
            return torch.exp(z)
        raise ValueError("weight_activation must be 'relu', 'softplus', or 'exp'.")

    def all_weights_from_bottom(self, w_bottom: Tensor) -> Tensor:
        w_all = torch.zeros(self.total_nodes, dtype=w_bottom.dtype, device=w_bottom.device)
        w_all[self.bottom_start:] = w_bottom
        for node in range(self.n_internal - 1, -1, -1):
            w_all[node] = w_all[2 * node + 1] + w_all[2 * node + 2]
        return w_all

    # def homogenize(self, z: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    #     w_bottom = self.weights_from_z(z)
    #     w_all = self.all_weights_from_bottom(w_bottom)
    #
    #     C_all = torch.zeros(self.total_nodes, 6, 6, dtype=self.C_leaves.dtype, device=self.C_leaves.device)
    #
    #     # Bottom layer: phase stiffness + rotation. No two-child homogenization here.
    #     for local_leaf, node in enumerate(self.bottom_indices):
    #         C_all[node] = rotate_stiffness_3d(self.C_leaves[local_leaf], alpha[node], beta[node], gamma[node])
    #
    #     # Internal nodes: child homogenization + rotation.
    #     for node in range(self.n_internal - 1, -1, -1):
    #         left = 2 * node + 1
    #         right = 2 * node + 2
    #         C_h = homogenize_two_layer_3d(C_all[left], C_all[right], w_all[left], w_all[right])
    #         C_all[node] = rotate_stiffness_3d(C_h, alpha[node], beta[node], gamma[node])
    #
    #     return C_all[0]
    #     return self.fix_predicted_matrix(C_all[0])

    def homogenize_from_flat_params(self, p: Tensor) -> Tensor:
        z, alpha, beta, gamma = self.parse_flat_params(p)
        return self.homogenize(z, alpha, beta, gamma)

    def forward(self, p: Optional[Tensor] = None) -> Tensor:
        if p is None:
            p = self.get_flat_params()
        return self.homogenize_from_flat_params(p)

    @staticmethod
    def fix_predicted_matrix(C: Tensor) -> Tensor:
        return 0.5 * (C + C.T)

    # -------------------------------------------------------------------------
    # Losses and diagnostics
    # -------------------------------------------------------------------------
    @staticmethod
    def normalized_frobenius_mse(C_pred: Tensor, C_tgt: Tensor, eps: float = 1e-12) -> Tensor:
        C_tgt = C_tgt.to(device=C_pred.device, dtype=C_pred.dtype)
        diff = C_pred - C_tgt
        return (diff * diff).sum() / ((C_tgt * C_tgt).sum() + eps)

    def regularization_loss(self, p: Optional[Tensor] = None) -> Tensor:
        if p is None:
            p = self.get_flat_params()
        z, _, _, _ = self.parse_flat_params(p)
        w = self.weights_from_z(z)
        target = torch.as_tensor(2.0 ** (self.N_layers - 2), dtype=w.dtype, device=w.device)
        return (w.sum() - target) ** 2

    def phase_fraction_error(self, p: Tensor, W_phases: Sequence[float] | Tensor) -> Tensor:
        z, _, _, _ = self.parse_flat_params(p)
        w = self.weights_from_z(z)
        total = w.sum() + 1e-12
        phase_sums = torch.zeros(self.n_phases, dtype=w.dtype, device=w.device)
        phase_sums.scatter_add_(0, self.leaf_phase_indices.to(w.device), w)
        phase_frac = phase_sums / total

        target = torch.as_tensor(W_phases, dtype=w.dtype, device=w.device)
        if target.numel() != self.n_phases:
            raise ValueError(f"W_phases has {target.numel()} values, expected {self.n_phases}.")
        target = target / (target.sum() + 1e-12)
        diff = phase_frac - target
        return (diff * diff).sum() / ((target * target).sum() + 1e-12)

    def output_params_from_p_flat(self, p_hat_1d: Tensor, FEAP_path) -> None:
        """
        Export trained/predicted 3D DMN parameters for FEAP/Fortran.

        DMN flat parameter layout:
            p = [z_bottom, alpha_all, beta_all, gamma_all]

        Written files:
            z_0.bin / z_0.shape
            w_bottom_0.bin / w_bottom_0.shape
            w_all_0.bin / w_all_0.shape
            alpha_0.bin / alpha_0.shape
            beta_0.bin / beta_0.shape
            gamma_0.bin / gamma_0.shape
            phase_leaf_0.bin / phase_leaf_0.shape

        Notes:
            bottom_nodes = 2**(N_layers - 1)
            total_nodes  = 2**N_layers - 1
        """

        from pathlib import Path

        if p_hat_1d.ndim != 1:
            p_hat_1d = p_hat_1d.reshape(-1)

        p = p_hat_1d.detach().to(device=self.z.device, dtype=self.z.dtype)

        expected = self.param_dim()
        if p.numel() != expected:
            raise ValueError(
                f"flat_p has size {p.numel()}, expected {expected}. "
                f"bottom_nodes={self.bottom_nodes}, total_nodes={self.total_nodes}"
            )

        z, alpha, beta, gamma = self.parse_flat_params(p)

        w_bottom = self.weights_from_z(z)

        w_all_list = self.all_weights_from_bottom(w_bottom)
        w_all = torch.stack(w_all_list, dim=0)

        FEAP_path = Path(FEAP_path)
        FEAP_path.mkdir(parents=True, exist_ok=True)

        def write_real64_1d(arr: Tensor, bin_name: str, shape_name: str):
            arr_np = np.asfortranarray(
                arr.detach().cpu().numpy().reshape(-1),
                dtype=np.float64,
            )
            arr_np.ravel(order="F").tofile(FEAP_path / bin_name)
            np.array(arr_np.shape, dtype=np.int32).tofile(FEAP_path / shape_name)

        def write_int32_1d(arr, bin_name: str, shape_name: str):
            arr_np = np.asfortranarray(
                np.asarray(arr, dtype=np.int32).reshape(-1),
                dtype=np.int32,
            )
            arr_np.ravel(order="F").tofile(FEAP_path / bin_name)
            np.array(arr_np.shape, dtype=np.int32).tofile(FEAP_path / shape_name)

        # write_real64_1d(z, "z_0.bin", "z_0.shape")
        # write_real64_1d(w_bottom, "w_bottom_0.bin", "w_bottom_0.shape")
        # write_real64_1d(w_all, "w_all_0.bin", "w_all_0.shape")
        # write_real64_1d(alpha, "alpha_0.bin", "alpha_0.shape")
        # write_real64_1d(beta, "beta_0.bin", "beta_0.shape")
        # write_real64_1d(gamma, "gamma_0.bin", "gamma_0.shape")
        #
        # # Leaf phase index: 1-based for Fortran.
        # # Example for 2 phases: [1, 2, 1, 2, ...]
        # phase_leaf = (
        #         self.leaf_phase_indices.detach().cpu().numpy().astype(np.int32) + 1
        # )
        # write_int32_1d(phase_leaf, "phase_leaf_0.bin", "phase_leaf_0.shape")
        write_real64_1d(z, "dmn_z.bin", "dmn_z.shape")
        write_real64_1d(alpha, "dmn_alpha.bin", "dmn_alpha.shape")
        write_real64_1d(beta, "dmn_beta.bin", "dmn_beta.shape")
        write_real64_1d(gamma, "dmn_gamma.bin", "dmn_gamma.shape")

        write_real64_1d(w_bottom, "dmn_w_bottom.bin", "dmn_w_bottom.shape")
        write_real64_1d(w_all, "dmn_w_all.bin", "dmn_w_all.shape")



        meta = {
            "N_layers": int(self.N_layers),
            "bottom_nodes": int(self.bottom_nodes),
            "total_nodes": int(self.total_nodes),
            "n_internal": int(self.n_internal),
            "n_phases": int(self.n_phases),
            "param_dim": int(self.param_dim()),
            "weight_activation": str(self.weight_activation),
            "phases": list(self.phases),
        }

        np.savez(FEAP_path / "dmn_meta_0.npz", **meta)

    def all_weights_from_bottom(self, w_bottom: Tensor) -> list[Tensor]:
        w_all = [None for _ in range(self.total_nodes)]

        for local_leaf, node in enumerate(self.bottom_indices):
            w_all[node] = w_bottom[local_leaf]

        for node in range(self.n_internal - 1, -1, -1):
            left = 2 * node + 1
            right = 2 * node + 2
            w_all[node] = w_all[left] + w_all[right]

        return w_all

    def homogenize(self, z: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        w_bottom = self.weights_from_z(z)
        w_all = self.all_weights_from_bottom(w_bottom)

        C_all = [None for _ in range(self.total_nodes)]

        for local_leaf, node in enumerate(self.bottom_indices):
            C_all[node] = rotate_stiffness_3d(
                self.C_leaves[local_leaf],
                alpha[node],
                beta[node],
                gamma[node],
            )

        for node in range(self.n_internal - 1, -1, -1):
            left = 2 * node + 1
            right = 2 * node + 2

            C_h = homogenize_two_layer_3d(
                C_all[left],
                C_all[right],
                w_all[left],
                w_all[right],
            )

            C_all[node] = rotate_stiffness_3d(
                C_h,
                alpha[node],
                beta[node],
                gamma[node],
            )

        return C_all[0]

# =============================================================================
# MLP that maps [GNN feature, base DMN params] -> corrected DMN params
# =============================================================================

class TransformToDMNParams(nn.Module):
    """
    Hyper-network / TNN for DMN parameters.

    Input:  concat([x_feat, base_flat_dmn_params])
    Output: corrected flat DMN parameter vector.

    If residual=True, output = base_params + scale * MLP(input).
    This is normally more stable than predicting all parameters from scratch.
    """

    def __init__(self, x_dim: int, p_dim: int, hidden_dim: int = 256, residual: bool = True, delta_scale: float = 0.1):
        super().__init__()
        self.x_dim = int(x_dim)
        self.p_dim = int(p_dim)
        self.residual = bool(residual)
        self.delta_scale = float(delta_scale)

        self.net = nn.Sequential(
            nn.Linear(self.x_dim + self.p_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.p_dim),
        )

    def forward(self, x_feat: Tensor, base_params: Tensor) -> Tensor:
        if x_feat.ndim == 1:
            x_feat = x_feat.unsqueeze(0)

        if base_params.ndim == 1:
            base_params = base_params.unsqueeze(0).expand(x_feat.shape[0], -1)

        inp = torch.cat([x_feat, base_params], dim=1)
        delta = self.net(inp)
        if self.residual:
            return base_params + self.delta_scale * delta
        return delta





# # =============================================================================
# # Fallback GNN: used only if .GNNs.GraphFeatureExtractor_DMN is unavailable
# # =============================================================================
#
# class GraphFeatureExtractor_DMN_Fallback(nn.Module):
#     def __init__(self, in_dim: int = 8, hidden_dim: int = 64, x_dim: int = 32, heads: int = 4, dropout: float = 0.0):
#         super().__init__()
#         try:
#             from torch_geometric.nn import GATv2Conv, global_mean_pool
#         except Exception as e:
#             raise ImportError("torch_geometric is required for the fallback GNN.") from e
#
#         assert hidden_dim % heads == 0
#         self.dropout = dropout
#         self.GATv2Conv = GATv2Conv
#         self.global_mean_pool = global_mean_pool
#
#         self.mp1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True)
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.mp2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, x_dim)
#
#     def forward(self, graph: Data) -> Tensor:
#         x, edge_index = graph.x, graph.edge_index
#         batch = getattr(graph, "batch", None)
#         if batch is None:
#             batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
#
#         x = self.mp1(x, edge_index)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.mp2(x, edge_index)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         y = self.global_mean_pool(x, batch)
#         y = torch.tanh(self.fc3(y))
#         return self.fc4(y)


# Backward-compatible alias
DMNCalculator = DMNCalculator3D
