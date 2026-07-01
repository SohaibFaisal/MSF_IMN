import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformToIMN_Interaction_Params(nn.Module):
    """
    p_hat = T([X_feats, p_bar])

    Input:  (32 + P)
    Output: (P)
    Softplus to keep IMN weights positive-friendly (your IMN uses softplus on z anyway).
    """
    def __init__(self, p_dim: int, x_dim: int, hidden_dim:int):
        super().__init__()
        # in_dim = x_dim + p_dim chg1
        in_dim = x_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        # z = torch.cat([x_feats, p_bar], dim=-1) chg1
        z = x_feats
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return F.softplus(self.fc3(z))


class TNN_DMN(nn.Module):
    """
    p_hat = T([X_feats, p_bar])

    Input:  (32 + P)
    Output: (P)
    Softplus to keep IMN weights positive-friendly (your IMN uses softplus on z anyway).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim:int):
        super().__init__()
        # in_dim = x_dim + p_dim chg1
        in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # z = torch.cat([x_feats, p_bar], dim=-1) chg1
        z = combined
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.fc3(z)


class graph_checking(nn.Module):
    def __init__(self, p_dim: int, x_dim: int, hidden_dim:int):
        super().__init__()
        # in_dim = x_dim + p_dim chg1
        in_dim = x_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor) -> torch.Tensor:
        # z = torch.cat([x_feats, p_bar], dim=-1) chg1
        z = x_feats
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z= F.softplus(self.fc3(z))
        return z


# Produce W and beta with hard constraint on W
class TransformToIMN_Node_Params(nn.Module):
    def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim: int, weight_index: int):
        super().__init__()

        self.weight_index = weight_index
        assert self.weight_index <= p_dim, "weight_index must be <= p_dim"

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(x_feats))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        # Works for both:
        # z shape (p_dim,)
        # z shape (B, p_dim)
        weights = z[..., :self.weight_index]
        betas   = z[..., self.weight_index:]
        # weights sum to 1 along last dimension
        weights = F.softmax(weights, dim=-1)



        # Put FVC on same device/dtype
        if not torch.is_tensor(FVC):
            FVC = torch.tensor(FVC, dtype=weights.dtype, device=weights.device)
        else:
            FVC = FVC.to(dtype=weights.dtype, device=weights.device)

        # Case 1: single vector weights -> shape (weight_index,)
        if weights.ndim == 1:
            if FVC.ndim > 0:
                FVC = FVC.reshape(-1)[0]
            weights = weights * FVC

        # Case 2: batched weights -> shape (B, weight_index)
        elif weights.ndim == 2:
            if FVC.ndim == 0:
                FVC = FVC.view(1, 1).expand(weights.shape[0], 1)
            elif FVC.ndim == 1:
                FVC = FVC.view(-1, 1)
            elif FVC.ndim == 2 and FVC.shape[1] == 1:
                pass
            else:
                raise ValueError(f"Invalid FVC shape: {FVC.shape}")
            weights = weights * FVC
        else:
            raise ValueError(f"Invalid weights shape: {weights.shape}")

        # betas = F.softplus(betas)
        return torch.cat([weights, betas], dim=-1)

# Produce W and beta with no constraint on W
class TransformToIMN_Node_Params_W_and_Beta(nn.Module):
    def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim: int, weight_index: int):
        super().__init__()
        self.weight_index = weight_index
        assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(x_feats))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        weights = z[..., :self.weight_index]
        betas   = z[..., self.weight_index:]
        weights = F.softmax(weights, dim=-1)
        print('-------------------')
        print(weights.sum())
        print('-------------------')


        # FVC = torch.as_tensor(FVC, dtype=weights.dtype, device=weights.device)


        betas = F.softplus(betas)
        return torch.cat([weights, betas], dim=-1)



# No constraint on W
class TransformToIMN_Node_ParamsW(nn.Module):
    def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(x_feats))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        z = F.softplus(z)
        return z


class TransformToIMN_Node_Paramsbeta(nn.Module):
    def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, p_dim)

    def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.fc1(x_feats))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        z = F.softplus(z)
        return z


# NOT NEEDED
# class TransformToIMN_Node_Params_W_and_Beta_Matrix(nn.Module):
#     def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim:int):
#         super().__init__()
#         self.weight_index = 2**(layers)  # e.g. 8
#         assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, p_dim)
#
#     def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
#         z = F.relu(self.fc1(x_feats))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)  # (B, p_dim)
#
#         # print(z)
#         # Split along feature dimension
#         # weights = z[:, :self.weight_index]   # (B, 8)
#         # betas   = z[:, self.weight_index:]   # (B, p_dim-8)
#         weights = z[:self.weight_index]   # (B, 8)
#         betas   = z[self.weight_index:]   # (B, p_dim-8)
#
#
#         # Constrain first 8 to sum to FVC
#         weights = F.softmax(weights, dim=-1)
#         print(FVC)
#         print('jhfdsjkbfsdscdcdsvdfvdfvdf')
#         # Make sure FVC broadcasts per-sample
#         if not torch.is_tensor(FVC):
#             FVC = torch.tensor(FVC, dtype=weights.dtype)
#         if FVC.ndim == 0:
#             weights = weights * FVC
#         else:
#             weights = weights * FVC.unsqueeze(-1)
#
#         # Optional constraint on betas (keep positive)
#         betas = F.softplus(betas)
#         return torch.cat([weights.squeeze(), betas], dim=-1)
#
#
# # NOT NEEDED
# class TransformToIMN_Node_Params_W_and_Beta_UD(nn.Module):
#     def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim:int):
#         super().__init__()
#         self.weight_index = 2**layers  # e.g. 8
#         assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, p_dim)
#
#     def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
#         z = F.relu(self.fc1(x_feats))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)  # (B, p_dim)
#
#         # print(z)
#         # Split along feature dimension
#         # weights = z[:, :self.weight_index]   # (B, 8)
#         # betas   = z[:, self.weight_index:]   # (B, p_dim-8)
#         weights = z[:self.weight_index]   # (B, 8)
#         betas   = z[self.weight_index:]   # (B, p_dim-8)
#
#
#         # Constrain first 8 to sum to FVC
#         weights = F.softmax(weights, dim=-1)
#
#         # Make sure FVC broadcasts per-sample
#         if not torch.is_tensor(FVC):
#             FVC = torch.tensor(FVC, dtype=weights.dtype)
#         if FVC.ndim == 0:
#             weights = weights * FVC
#         else:
#             weights = weights * FVC.unsqueeze(-1)
#
#         # Optional constraint on betas (keep positive)
#         betas = F.softplus(betas)
#         return torch.cat([weights.squeeze(), betas], dim=-1)
#
#
# # NOT NEEDED
# class TransformToIMN_Node_Params_W_and_Beta_PR(nn.Module):
#     def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim:int):
#         super().__init__()
#         self.weight_index = 2**layers  # e.g. 8
#         assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, p_dim)
#
#     def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
#         z = F.relu(self.fc1(x_feats))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)  # (B, p_dim)
#
#         # print(z)
#         # Split along feature dimension
#         # weights = z[:, :self.weight_index]   # (B, 8)
#         # betas   = z[:, self.weight_index:]   # (B, p_dim-8)
#         weights = z[:self.weight_index]   # (B, 8)
#         betas   = z[self.weight_index:]   # (B, p_dim-8)
#
#
#         # Constrain first 8 to sum to FVC
#         weights = F.softmax(weights, dim=-1)
#
#         # Make sure FVC broadcasts per-sample
#         if not torch.is_tensor(FVC):
#             FVC = torch.tensor(FVC, dtype=weights.dtype)
#         if FVC.ndim == 0:
#             weights = weights * FVC
#         else:
#             weights = weights * FVC.unsqueeze(-1)
#
#         # Optional constraint on betas (keep positive)
#         betas = F.softplus(betas)
#         return torch.cat([weights.squeeze(), betas], dim=-1)
#
#
# # NOT NEEDED
# class TransformToIMN_Node_Params_W_and_Beta_SFR(nn.Module):
#     def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim:int):
#         super().__init__()
#         self.weight_index = 2**layers  # e.g. 8
#         assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, p_dim)
#
#     def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
#         z = F.relu(self.fc1(x_feats))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)  # (B, p_dim)
#
#         # print(z)
#         # Split along feature dimension
#         # weights = z[:, :self.weight_index]   # (B, 8)
#         # betas   = z[:, self.weight_index:]   # (B, p_dim-8)
#         weights = z[:self.weight_index]   # (B, 8)
#         betas   = z[self.weight_index:]   # (B, p_dim-8)
#
#
#         # Constrain first 8 to sum to FVC
#         weights = F.softmax(weights, dim=-1)
#
#         # Make sure FVC broadcasts per-sample
#         if not torch.is_tensor(FVC):
#             FVC = torch.tensor(FVC, dtype=weights.dtype)
#         if FVC.ndim == 0:
#             weights = weights * FVC
#         else:
#             weights = weights * FVC.unsqueeze(-1)
#
#         # Optional constraint on betas (keep positive)
#         betas = F.softplus(betas)
#         return torch.cat([weights.squeeze(), betas], dim=-1)



# # NOT NEEDED
# class TransformToIMN_Node_Params_W_and_Beta(nn.Module):
#     def __init__(self, p_dim: int, in_dim: int, layers: int, hidden_dim:int,weight_index:int):
#         super().__init__()
#         self.weight_index = weight_index
#         assert self.weight_index <= p_dim, "weight_index must be <= p_dim"
#
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, p_dim)
#
#     def forward(self, x_feats: torch.Tensor, FVC: torch.Tensor) -> torch.Tensor:
#         z = F.relu(self.fc1(x_feats))
#         z = F.relu(self.fc2(z))
#         z = self.fc3(z)  # (B, p_dim)
#
#         weights = z[:self.weight_index]   # (B, 8)
#
#         betas   = z[self.weight_index:]   # (B, p_dim-8)
#
#         # Constrain first 8 to sum to FVC
#         weights = F.softmax(weights, dim=-1)
#
#         # Make sure FVC broadcasts per-sample
#         if not torch.is_tensor(FVC):
#             FVC = torch.tensor(FVC, dtype=weights.dtype)
#         if FVC.ndim == 0:
#             weights = weights * FVC
#         else:
#             weights = weights * FVC.unsqueeze(-1)
#
#         # Optional constraint on betas (keep positive)
#         betas = F.softplus(betas)
#
#         return torch.cat([weights.squeeze(), betas], dim=-1)







