
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, GATv2Conv, GlobalAttention, JumpingKnowledge, Set2Set
import torch
import torch.nn as nn
import torch.nn.functional as F





def safe_masked_mean_pool(x, batch, mask):
    """
    Mean pool over nodes selected by mask.
    If a graph has zero selected nodes, output is temporarily zero.
    The caller can replace empty pools with global pools.
    """

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

    batch = batch.to(device=x.device, dtype=torch.long)
    mask = mask.to(device=x.device, dtype=x.dtype).view(-1, 1)

    out = x.new_zeros((num_graphs, x.size(-1)))
    count = x.new_zeros((num_graphs, 1))

    out.index_add_(0, batch, x * mask)
    count.index_add_(0, batch, mask)

    pooled = out / count.clamp(min=1.0)

    return pooled, count


def target_context_global_pool(x, batch, target_mask):
    """
    Returns:
        [target_pool, context_pool, global_pool]

    Works for:
      1. target-aware full graphs
      2. main graph where all is_target = 0
      3. individual phase graphs where all nodes may be target
    """

    global_mask = torch.ones_like(target_mask, dtype=torch.bool)
    context_mask = ~target_mask

    z_global, _ = safe_masked_mean_pool(x, batch, global_mask)
    z_target, target_count = safe_masked_mean_pool(x, batch, target_mask)
    z_context, context_count = safe_masked_mean_pool(x, batch, context_mask)

    # If there are no target nodes, use global pool instead.
    # This happens for the plain main graph where is_target is all zero.
    no_target = target_count.squeeze(-1) == 0
    z_target[no_target] = z_global[no_target]

    # If there are no context nodes, use global pool instead.
    # This can happen for individual phase graphs if all nodes are target.
    no_context = context_count.squeeze(-1) == 0
    z_context[no_context] = z_global[no_context]

    return torch.cat([z_target, z_context, z_global], dim=-1)




class GraphFeatureExtractor_phase_aware(nn.Module):
    """
    Variable-depth GATv2

    Readout:
        target/context/global pooling

    Head:
        MLP -> x_dim
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        x_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        target_col: int = 9,
    ):
        super().__init__()

        assert hidden_dim % heads == 0
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout
        self.target_col = target_col

        self.gats = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_dim = in_dim if layer == 0 else hidden_dim

            self.gats.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
            )

            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))

        # Important change:
        # pooling now returns [target, context, global]
        # so the readout size is 3 * hidden_dim, not hidden_dim.
        self.fc3 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        # Extract target mask before GNN.
        # Your graph stores is_target in x[:, 9].
        if hasattr(graph, "target_mask"):
            target_mask = graph.target_mask.bool()
        else:
            target_mask = x[:, self.target_col] > 0.5

        for gat, fc in zip(self.gats, self.fcs):
            x = gat(x, edge_index)
            x = F.relu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        y = target_context_global_pool(x, batch, target_mask)

        y = torch.tanh(self.fc3(y))
        y = self.fc4(y)

        return y

# =========================================================
# 1) GNN feature extractor Mean
# =========================================================

class GraphFeatureExtractor_original(nn.Module):
    """
    Variable-depth GATv2
    Readout: mean pool
    Head: MLP -> x_dim
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        x_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_dim % heads == 0
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.gats = nn.ModuleList()
        self.fcs = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_dim = in_dim if layer == 0 else hidden_dim

            self.gats.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
            )

            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        for gat, fc in zip(self.gats, self.fcs):
            x = gat(x, edge_index)
            x = F.relu(fc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        y = global_mean_pool(x, batch)
        y = torch.tanh(self.fc3(y))
        y = self.fc4(y)

        return y



# class GraphFeatureExtractor_interaction(nn.Module):
#     """
#     MP (GATv2): (F_in -> 64)
#     FC + ReLU:  (64 -> 64)
#     MP (GATv2): (64 -> 64)
#     FC + ReLU:  (64 -> 64)
#     mean pool:  (64)
#     FC + tanh:  (64 -> 64)
#     FC + softmax:(64 -> 32)
#     """
#     def __init__(self, in_dim: int, hidden_dim: int = 64, x_dim: int = 32, heads: int = 4, p_dim: int = 2):
#         super().__init__()
#         assert hidden_dim % heads == 0
#
#         self.gat1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True)
#         self.fc1  = nn.Linear(hidden_dim, hidden_dim)
#
#         # self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
#         # self.fc2  = nn.Linear(hidden_dim, hidden_dim)
#
#         self.fc3  = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4  = nn.Linear(hidden_dim, x_dim)
#
#         in_dim = x_dim
#         self.fc5 = nn.Linear(in_dim, int(in_dim))
#         self.fc6 = nn.Linear(in_dim, p_dim)
#
#     def forward(self, graph: Data) -> torch.Tensor:
#
#         x, edge_index, batch = graph.x, graph.edge_index, graph.batch
#         x = self.gat1(x, edge_index)
#         x = F.relu(self.fc1(x))
#         # x = self.gat2(x, edge_index)
#         # x = F.relu(self.fc2(x))
#         y = global_mean_pool(x, batch)      # (1, 64) for single graph
#         y = torch.tanh(self.fc3(y))         # (1, 64)
#         y = F.softmax(self.fc4(y), dim=-1)  # (1, 32)
#         z = F.relu(self.fc5(y))
#         return F.softplus(self.fc6(z))
#
#
# class GraphFeatureExtractor_nodes(nn.Module):
#     """
#     MP (GATv2): (F_in -> 64)
#     FC + ReLU:  (64 -> 64)
#     MP (GATv2): (64 -> 64)
#     FC + ReLU:  (64 -> 64)
#     mean pool:  (64)
#     FC + tanh:  (64 -> 64)
#     FC + softmax:(64 -> 32)
#     """
#     def __init__(self, in_dim: int, hidden_dim: int = 64, x_dim: int = 32, heads: int = 4, p_dim: int = 2):
#         super().__init__()
#         assert hidden_dim % heads == 0
#
#         self.gat1 = GATv2Conv(in_dim, hidden_dim // heads, heads=heads, concat=True)
#         self.fc1  = nn.Linear(hidden_dim, hidden_dim)
#
#         # self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
#         # self.fc2  = nn.Linear(hidden_dim, hidden_dim)
#
#         self.fc3  = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4  = nn.Linear(hidden_dim, x_dim)
#
#         in_dim = x_dim
#         self.fc5 = nn.Linear(in_dim, int(in_dim))
#         self.fc6 = nn.Linear(in_dim, p_dim)
#
#     def forward(self, graph: Data) -> torch.Tensor:
#
#         x, edge_index, batch = graph.x, graph.edge_index, graph.batch
#         x = self.gat1(x, edge_index)
#         x = F.relu(self.fc1(x))
#         # x = self.gat2(x, edge_index)
#         # x = F.relu(self.fc2(x))
#         y = global_mean_pool(x, batch)      # (1, 64) for single graph
#         y = torch.tanh(self.fc3(y))         # (1, 64)
#         y = F.softmax(self.fc4(y), dim=-1)  # (1, 32)
#         z = F.relu(self.fc5(y))
#         return F.softplus(self.fc6(z))

class GraphFeatureExtractor_MultiPoolResidual(nn.Module):
    """
    Variable-depth GATv2 blocks + residual connections
    Readout: concat(mean, max, add)
    Head: MLP -> x_dim
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        x_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % heads == 0
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.gats = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_dim = in_dim if layer == 0 else hidden_dim

            self.gats.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
            )

            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.fc3 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        prev = None

        for gat, lin, norm in zip(self.gats, self.lins, self.norms):
            h = gat(x, edge_index)
            h = lin(h)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if prev is not None:
                h = h + prev

            x = h
            prev = h

        y_mean = global_mean_pool(x, batch)
        y_max = global_max_pool(x, batch)
        y_add = global_add_pool(x, batch)

        y = torch.cat([y_mean, y_max, y_add], dim=-1)

        y = F.relu(self.fc3(y))
        y = self.fc4(y)

        return y

class GraphFeatureExtractor_AttentionPool(nn.Module):
    """
    Variable-depth GATv2 layers
    Readout: GlobalAttention
    Head: MLP -> x_dim
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        x_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert hidden_dim % heads == 0
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_dim = in_dim if layer == 0 else hidden_dim

            self.gats.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
            )

            self.norms.append(nn.LayerNorm(hidden_dim))

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        )

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for gat, norm in zip(self.gats, self.norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        y = self.pool(x, batch)

        y = F.relu(self.fc3(y))
        y = self.fc4(y)

        return y

class GraphFeatureExtractor_JK_Set2Set(nn.Module):
    """
    Variable-depth GATv2 layers
    Readout: JumpingKnowledge(cat) + Set2Set
    Head: MLP -> x_dim
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        x_dim: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        set2set_steps: int = 3,
    ):
        super().__init__()

        assert hidden_dim % heads == 0
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout

        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layers):
            layer_in_dim = in_dim if layer == 0 else hidden_dim

            self.gats.append(
                GATv2Conv(
                    layer_in_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                )
            )

            self.norms.append(nn.LayerNorm(hidden_dim))

        self.jk = JumpingKnowledge(mode="cat")

        jk_dim = hidden_dim * num_layers

        self.pool = Set2Set(
            in_channels=jk_dim,
            processing_steps=set2set_steps,
        )

        self.fc3 = nn.Linear(2 * jk_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)

    def forward(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        layer_outputs = []

        for gat, norm in zip(self.gats, self.norms):
            x = gat(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            layer_outputs.append(x)

        x_jk = self.jk(layer_outputs)

        y = self.pool(x_jk, batch)

        y = F.relu(self.fc3(y))
        y = self.fc4(y)

        return y
