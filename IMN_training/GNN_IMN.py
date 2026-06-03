

import torch.optim as optim


from .Optimization_loop_GNN_IMN import *
from .Optimization_loop_graphs import *
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, GATv2Conv, GlobalAttention, JumpingKnowledge, Set2Set
from .IMN_Network import *
from .IMN_calculator import *
from .Optimization_loop import *

# from GNNs import GraphFeatureExtractor_nodes




# =========================================================
# 0) Graph loading (npz -> PyG Data)
# =========================================================

def load_graph_npz(path: str) -> Data:
    d = np.load(path)
    x = torch.tensor(d["x"], dtype=torch.float32)                 # (N, F)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.int32) # (2, E)
    g = Data(x=x, edge_index=edge_index)
    g.batch = torch.zeros(x.size(0), dtype=torch.long)

    # optional metadata
    # if "elem_labels" in d.files:
    #     g.elem_labels = torch.tensor(d["elem_labels"])
    # if "bbox" in d.files:
    #     g.bbox = torch.tensor(d["bbox"], dtype=torch.float32)
    # if "elem_pkey" in d.files:
    #     g.pkey = d["elem_pkey"]

    if "FVC" in d.files:
        g.FVC = torch.tensor(d["FVC"], dtype=torch.float32)

    return g


class HybridGNNIMN(nn.Module):
    def __init__(self, node_feat_dim: int,N_layers:int, gnn_hidden_dim:int,tnn_hidden_dim:int, heads:int, x_dim: int, GNN_structure: int, nodes_per_mech_per_phase:int):

        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.tnn_hidden_dim = tnn_hidden_dim
        self.gnn_structure = GNN_structure
        self.node_feat_dim = node_feat_dim
        self.heads = heads
        self.x_dim = x_dim
        self.N_layers = N_layers
        self.M = 2**N_layers - 1
        self.nodes_per_mech_per_phase = nodes_per_mech_per_phase


        # if GNN_structure == 1:
        #     from .GNNs import GraphFeatureExtractor_original
        #     self.gnn = GraphFeatureExtractor_original(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads) #ok
        if GNN_structure == 1:
            from .GNNs import GraphFeatureExtractor_phase_aware
            self.gnn = GraphFeatureExtractor_phase_aware(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads) #ok
        elif GNN_structure == 2:
            from .GNNs import GraphFeatureExtractor_AttentionPool_phase_aware
            self.gnn = GraphFeatureExtractor_AttentionPool_phase_aware(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)  # Good
        elif GNN_structure == 3:
            from .GNNs import GraphFeatureExtractor_MultiPoolResidual
            self.gnn = GraphFeatureExtractor_MultiPoolResidual(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)  # Not good
        elif GNN_structure == 4:
            from .GNNs import GraphFeatureExtractor_JK_Set2Set_phase_aware
            self.gnn = GraphFeatureExtractor_JK_Set2Set_phase_aware(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)




        from .TNNs import TransformToIMN_Interaction_Params, TransformToIMN_Node_Params_W_and_Beta
        self.T_interaction = TransformToIMN_Interaction_Params(p_dim=int(2*self.M), x_dim=self.x_dim, hidden_dim=tnn_hidden_dim)
        self.T_nodes = TransformToIMN_Node_Params_W_and_Beta(p_dim=self.nodes_per_mech_per_phase * (2 ** (N_layers-1))*(1+self.M), in_dim=self.x_dim, layers=self.N_layers, hidden_dim=tnn_hidden_dim, weight_index=self.nodes_per_mech_per_phase * (2 ** (N_layers-1)))

        self.imn_cache = {}

    def forward(self, phases, main_graph, phase_graphs):
        """
        Returns:
          p_hat_batch: (B, P)
          x_feats: (1, 32)
        """
        n_leaf = 2 ** (self.N_layers - 1) * self.nodes_per_mech_per_phase
        z_list = []
        beta_list = []
        bg = Batch.from_data_list(phase_graphs)
        x_feats_ph = self.gnn(bg)  # (P, x_dim) instead of P calls
        # print(x_feats_ph.shape)


        for k, ph in enumerate(phases):
            x_rep = x_feats_ph[k]
            # print(f'Feature vector of phase: {ph}')
            # print(x_rep.shape)
            # print(phase_graphs[k].FVC[k])
            p_hat = self.T_nodes(x_rep, phase_graphs[k].FVC[k]).reshape(-1)
            z_k, beta_k = p_hat[:n_leaf], p_hat[n_leaf:]
            z_list.append(z_k)
            beta_list.append(beta_k)

        z_by_phase = torch.stack(z_list, dim=0)  # (P, n_leaf)
        beta_by_phase = torch.stack(beta_list, dim=0)  # (P, M x n_leaf)
        # print(z_by_phase.shape)
        # print(beta_by_phase.shape)

        x_feat = self.gnn(main_graph)



        x_rep = x_feat.repeat(1, 1)
        p_hat = self.T_interaction(x_rep).squeeze(0)
        # print('Feature vector of main graph')
        # print(x_feat.shape)
        # print(p_hat.shape)
        theta, phi = p_hat[:self.M], p_hat[self.M:]
        flat_p = pack_flat_p(z_by_phase, beta_by_phase, theta, phi, self.M)


        return flat_p


class Graph_Model(nn.Module):
    def __init__(self, gnn_hidden_dim:int,heads:int, x_dim: int, GNN_structure: int, GNN_layers:int):
        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_structure = GNN_structure
        self.x_dim = x_dim
        self.heads = heads
        self.GNN_layers = GNN_layers
        node_feat_dim = 7



        if GNN_structure == 1:
            from .GNNs import GraphFeatureExtractor_original
            self.gnn = GraphFeatureExtractor_original(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads,num_layers=GNN_layers) #ok
        elif GNN_structure == 2:
            from .GNNs import GraphFeatureExtractor_AttentionPool
            self.gnn = GraphFeatureExtractor_AttentionPool(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads,num_layers=GNN_layers)  # Good
        elif GNN_structure == 3:
            from .GNNs import GraphFeatureExtractor_MultiPoolResidual
            self.gnn = GraphFeatureExtractor_MultiPoolResidual(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads,num_layers=GNN_layers)  # Not good
        elif GNN_structure == 4:
            from .GNNs import GraphFeatureExtractor_JK_Set2Set
            self.gnn = GraphFeatureExtractor_JK_Set2Set(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads,num_layers=GNN_layers)




        from .TNNs import graph_checking
        self.T = graph_checking(p_dim=36, x_dim=self.x_dim, hidden_dim=128)

    def forward(self, phases, phase_graphs, material_in):
        bg = Batch.from_data_list(phase_graphs)
        x_feats_ph = self.gnn(bg)  # (P, x_dim) instead of P calls
        C_pred = torch.zeros((6,6))
        for k, ph in enumerate(phases):
            x_rep = x_feats_ph[k]
            p_hat = self.T(x_rep).squeeze().reshape(6,6)
            p_hat = p_hat @ material_in[ph].float() * phase_graphs[k].FVC
            C_pred += p_hat

        return C_pred




class HybridGNNIMN_with_4_TNNs(nn.Module):
    def __init__(self, node_feat_dim: int,N_layers:int, gnn_hidden_dim:int,tnn_hidden_dim:int, heads:int, x_dim: int, GNN_structure: int):
        super().__init__()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.tnn_hidden_dim = tnn_hidden_dim
        self.gnn_structure = GNN_structure
        self.node_feat_dim = node_feat_dim
        self.heads = heads
        self.x_dim = x_dim
        self.N_layers = N_layers
        self.M = 2**N_layers - 1

        if GNN_structure == 1:
            from .GNNs import GraphFeatureExtractor_original
            self.gnn = GraphFeatureExtractor_original(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)  # ok
        elif GNN_structure == 2:
            from .GNNs import GraphFeatureExtractor_AttentionPool
            self.gnn = GraphFeatureExtractor_AttentionPool(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)  # Good
        elif GNN_structure == 3:
            from .GNNs import GraphFeatureExtractor_MultiPoolResidual
            self.gnn = GraphFeatureExtractor_MultiPoolResidual(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)  # Not good
        elif GNN_structure == 4:
            from .GNNs import GraphFeatureExtractor_JK_Set2Set
            self.gnn = GraphFeatureExtractor_JK_Set2Set(in_dim=node_feat_dim, hidden_dim=gnn_hidden_dim, x_dim=x_dim, heads=heads)




        from .TNNs import TransformToIMN_Interaction_Params
        self.T_interaction = TransformToIMN_Interaction_Params(p_dim=int(2*self.M), x_dim=self.x_dim, hidden_dim=tnn_hidden_dim)
        from .TNNs import TransformToIMN_Node_Params_W_and_Beta_Matrix, TransformToIMN_Node_Params_W_and_Beta_UD, TransformToIMN_Node_Params_W_and_Beta_SFR, TransformToIMN_Node_Params_W_and_Beta_PR
        self.T_nodes_Matrix = TransformToIMN_Node_Params_W_and_Beta_Matrix(p_dim=int((1 + self.M) * (2 ** N_layers)), in_dim=self.x_dim, layers=self.N_layers, hidden_dim=tnn_hidden_dim)
        self.T_nodes_UD = TransformToIMN_Node_Params_W_and_Beta_UD(p_dim=int((1 + self.M) * (2 ** N_layers)), in_dim=self.x_dim, layers=self.N_layers,
                                                                           hidden_dim=tnn_hidden_dim)
        self.T_nodes_SFR = TransformToIMN_Node_Params_W_and_Beta_SFR(p_dim=int((1 + self.M) * (2 ** N_layers)), in_dim=self.x_dim, layers=self.N_layers,
                                                                           hidden_dim=tnn_hidden_dim)
        self.T_nodes_PR = TransformToIMN_Node_Params_W_and_Beta_PR(p_dim=int((1 + self.M) * (2 ** N_layers)), in_dim=self.x_dim, layers=self.N_layers,
                                                                           hidden_dim=tnn_hidden_dim)


        self.imn_cache = {}

    def forward(self, phases, main_graph, phase_graphs):
        """
        Returns:
          p_hat_batch: (B, P)
          x_feats: (1, 32)
        """
        n_leaf = 2 ** self.N_layers
        z_list = []
        beta_list = []



        x_rep = self.gnn(main_graph).squeeze(0)
        for k, ph in enumerate(phases):


            if 'MATRIX' in ph:
                p_hat = self.T_nodes_Matrix(x_rep, phase_graphs[k].FVC).squeeze()
            elif 'UD' in ph:
                p_hat = self.T_nodes_UD(x_rep, phase_graphs[k].FVC).squeeze()
            elif 'SFR' in ph:
                p_hat = self.T_nodes_SFR(x_rep, phase_graphs[k].FVC).squeeze()
            elif 'PR' in ph:
                p_hat = self.T_nodes_PR(x_rep, phase_graphs[k].FVC).squeeze()
            else:
                raise NotImplementedError

            z_k, beta_k = p_hat[:n_leaf], p_hat[n_leaf:]
            z_list.append(z_k)
            beta_list.append(beta_k)


        z_by_phase = torch.stack(z_list, dim=0)  # (P, n_leaf)
        beta_by_phase = torch.stack(beta_list, dim=0)  # (P, M x n_leaf)
        p_hat = self.T_interaction(x_rep).squeeze(0)
        theta, phi = p_hat[:self.M], p_hat[self.M:]
        flat_p = pack_flat_p(z_by_phase, beta_by_phase, theta, phi, self.M)
        return flat_p



def train_hybrid_one_graph(
    N_layers: int,
    num_samples: int,
    num_epochs: int,
    lr_rest: float,
    live_plot,
    training_dataset_folder: Path,
    imn_trained_data_folder: Path,
    optimizing_variables,
    mode:str = 'Train',
    weight_decay:float = 0.0,
    nodes_per_mech_per_phase=2,
    trial=None,
        use_GPU=True

):



    if mode == 'Train':


        device = get_device(use_GPU)
        mesh_folder = training_dataset_folder / 'Meshes'
        training_data_set = get_dataset_main(num_samples, training_dataset_folder)
        cfg = dict(
            node_feat_dim=10,  # derived
            gnn_hidden_dim=optimizing_variables[1],
            gnn_heads=int(optimizing_variables[1]/8),
            tnn_hidden_dim=int(optimizing_variables[0]),
            x_dim=int(optimizing_variables[2]),  # embedding size
        )



        model = HybridGNNIMN(node_feat_dim=cfg['node_feat_dim'],N_layers=N_layers, gnn_hidden_dim=cfg['gnn_hidden_dim'], tnn_hidden_dim=cfg['tnn_hidden_dim'],
                             heads=cfg['gnn_heads'], x_dim=cfg['x_dim'], GNN_structure = optimizing_variables[3], nodes_per_mech_per_phase=nodes_per_mech_per_phase).float().to(device)

        opt = optim.Adam(
            [{"params": [p for n, p in model.named_parameters() if n != "p_bar"], "lr": lr_rest}],
            weight_decay=weight_decay
        )

        best_val = run_live_optimization_GNN_IMN(num_epochs, num_samples, training_data_set,mesh_folder, 1, opt, model, live_plot, imn_trained_data_folder, 1, N_layers,device, nodes_per_mech_per_phase,trial,accumulation_steps=50, samples_per_epoch=2500)
        imn_trained_data_folder.mkdir(parents=True, exist_ok=True)
        GNN_FILE_PATH = imn_trained_data_folder / f"gnn_imn_generator.pt"

        ckpt = {
            "gnn": model.gnn.state_dict(),
            "T_interaction": model.T_interaction.state_dict(),
            "T_nodes": model.T_nodes.state_dict(),
            "N_layers": N_layers,
            "node_feat_dim": model.node_feat_dim,
            "x_dim": model.x_dim,
            "gnn_hidden_dim": model.gnn_hidden_dim, "tnn_hidden_dim": model.tnn_hidden_dim, "heads": model.heads,
            "imn_N_layers": model.N_layers,
            "gnn_structure": model.gnn_structure,

        }
        torch.save(ckpt, GNN_FILE_PATH)


        # return model
        return best_val





def train_imn(
    N_layers: int,
    num_samples: int,
    num_epochs: int,
    lr: float,
    live_plot,
    training_dataset_folder: Path,
    imn_trained_data_folder: Path,
    mode:str = 'Train',
        trial=None

):


    if mode == 'Train':
        mesh_folder = training_dataset_folder / 'Meshes'
        training_data_set = get_dataset_main(num_samples, training_dataset_folder)
        model = IMNModel(5, ['MATRIX', 'UD1']).double()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_val = run_live_optimization_IMN(num_epochs, num_samples, training_data_set, 1, opt, model,[],[], live_plot, imn_trained_data_folder, 1)
        imn_trained_data_folder.mkdir(parents=True, exist_ok=True)


        GNN_FILE_PATH = imn_trained_data_folder / f"trained_imn.pt"

        ckpt = {
            "imn": model.state_dict(),
            "N_layers": model.N_layers,
            "phases": model.phases,}


        torch.save(ckpt, GNN_FILE_PATH)



        return best_val



def fix_homogenized_C(a):

    mult = 0.5
    a[0:3, 3:6] = 0
    a[3:6, 0:3] = 0
    a[3, 4] = 0
    a[3, 5] = 0
    a[4, 3] = 0
    a[4, 5] = 0
    a[5, 3] = 0
    a[5, 4] = 0
    a[3, 3] = mult * a[3, 3]
    a[4, 4] = mult * a[4, 4]
    a[5, 5] = mult * a[5, 5]




def graph_runs(training_dataset_folder,imn_trained_data_folder, num_samples, optimizing_variables, num_epochs,lr_rest,trial=None):
        mesh_folder = training_dataset_folder / 'Meshes'
        training_data_set = get_dataset_main(num_samples, training_dataset_folder)
        gnn_hidden_dim = optimizing_variables["gnn_hidden_dim"]
        gnn_heads = optimizing_variables["gnn_heads"]
        x_feat = optimizing_variables["x_feat"]
        gnn_structure = optimizing_variables["gnn_structure"]
        gnn_layers = optimizing_variables["gnn_layers"]

        cfg = dict(
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_heads=gnn_heads,
            x_dim=x_feat,  # embedding size
            gnn_structure=gnn_structure,
            gnn_layers=gnn_layers,
        )

        graph_model = Graph_Model(gnn_hidden_dim=cfg['gnn_hidden_dim'],heads=cfg['gnn_heads'], x_dim=cfg['x_dim'], GNN_structure = cfg['gnn_structure'], GNN_layers=cfg['gnn_layers'])
        live_plot = False

        graph_model.float()
        opt = optim.Adam(
            [{"params": [p for n, p in graph_model.named_parameters() if n != "p_bar"], "lr": lr_rest}]
        )

        best_val = run_live_optimization_graphs(num_epochs, num_samples, training_data_set,mesh_folder, 1, opt, graph_model, live_plot, imn_trained_data_folder, 1, trial)
        imn_trained_data_folder.mkdir(parents=True, exist_ok=True)
        GNN_FILE_PATH = imn_trained_data_folder / f"gnn_imn_generator.pt"

        ckpt = {
            "gnn": graph_model.gnn.state_dict(),
            "x_dim": graph_model.x_dim,
            "gnn_hidden_dim": graph_model.gnn_hidden_dim, "heads": graph_model.heads,
            "gnn_structure": graph_model.gnn_structure,

        }
        torch.save(ckpt, GNN_FILE_PATH)


        # return model
        return best_val



def get_dataset_main(num_samples,training_dataset_folder):

    main_data_set = {}
    npz = np.load(str(training_dataset_folder / f'material_dictionary.npz'), allow_pickle=True)
    C_in = {
        k: npz[k].item()  # unwrap the dict from numpy object array
        for k in npz.files
    }
    C_out = np.load(str(training_dataset_folder / f'homogenize.npz'), allow_pickle=True)
    npz = np.load(str(training_dataset_folder / f'key_map.npz'), allow_pickle=True)

    key_map = {
        k: npz[k].item()  # unwrap the dict from numpy object array
        for k in npz.files
    }





    for i in range(num_samples):
        main_data_set[str(i)] = {}
        stage, rve, mesh = key_map[str(i)]['ids']
        main_data_set[str(i)]['ids'] = [stage, rve, mesh]
        phases = key_map[str(i)]['phases']
        main_data_set[str(i)]['Phases'] = phases

        for phase in phases:
            main_data_set[str(i)][f'{phase}'] = torch.tensor(C_in[str(i)][phase])

        C_target = torch.tensor(C_out[str(i)])
        fix_homogenized_C(C_target)
        main_data_set[str(i)][f'C_Target'] = C_target


    return main_data_set   #, graphs_data_set



@torch.inference_mode()
def generate_imn_params_for_new_graph_validation(mesh_folder,phases,imn_trained_data_folder,imn_validation_folder, stage, rve, mesh, training_dataset_folder, num_sam, mode):
    GNN_FILE_PATH = imn_trained_data_folder / 'gnn_imn_generator.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(GNN_FILE_PATH, map_location="cpu")
    N_layers = ckpt["N_layers"]
    node_feat_dim = ckpt["node_feat_dim"]
    x_dim = ckpt["x_dim"]
    gnn_hidden_dim = ckpt["gnn_hidden_dim"]
    tnn_hidden_dim = ckpt["tnn_hidden_dim"]
    gnn_structure = ckpt["gnn_structure"]
    heads = ckpt["heads"]
    new_model = HybridGNNIMN(node_feat_dim,N_layers,gnn_hidden_dim=gnn_hidden_dim,tnn_hidden_dim=tnn_hidden_dim, heads=heads, x_dim=x_dim, GNN_structure=gnn_structure, nodes_per_mech_per_phase=nodes_per_mech_per_phase).to(device).eval()

    new_model.gnn.load_state_dict(ckpt["gnn"])
    new_model.T_interaction.load_state_dict(ckpt["T_interaction"])
    new_model.T_nodes.load_state_dict(ckpt["T_nodes"])

    new_model.gnn.eval()
    new_model.T_interaction.eval()
    new_model.T_nodes.eval()

    if mode == 1:
        main_g = load_graph_npz_2(str(mesh_folder / f'graph_stage_{stage}_rve_{rve}_mesh_{mesh}.npz')).to(device)
        phase_g = [load_graph_npz_2(str(mesh_folder / f'graph_stage_{stage}_rve_{rve}_mesh_{mesh}_phase_{ph}.npz')) for ph in phases]
        phase_g = [g.to(device) for g in phase_g]
        p_flat = new_model.forward(phases, main_g, phase_g)
        imn = IMNCalculator(N_layers, phases)
        imn.output_params_from_p_flat(p_flat, imn_validation_folder)

    else:
        training_data_set = get_dataset_main(num_sam, training_dataset_folder)
        target_constants = []
        prediction_constants = []
        for id in range(num_sam):
            sid = str(id)
            ss, rr, mm = training_data_set[sid]['ids']
            main_g = load_graph_npz_2(str(mesh_folder / f'graph_stage_{ss}_rve_{rr}_mesh_{mm}.npz')).to(device)
            phase_g = [load_graph_npz_2(str(mesh_folder / f'graph_stage_{ss}_rve_{rr}_mesh_{mm}_phase_{ph}.npz')) for ph in training_data_set[sid][f"Phases"]]
            phase_g = [g.to(device) for g in phase_g]
            flat_p = new_model.forward(training_data_set[sid]['Phases'], main_g,
                                   phase_g)
            imn = IMNCalculator(N_layers, training_data_set[sid]['Phases'])
            imn.assign_node_stiffness(training_data_set[sid])
            C_pred = imn.homogenize_from_flat_params(flat_p)
            C_tgt = training_data_set[sid]["C_Target"]

            print('Solving for: ' + str(id))
            target_constants.append(extract_engineering_constants(C_tgt))
            prediction_constants.append(extract_engineering_constants(C_pred))

        return target_constants, prediction_constants


@torch.inference_mode()
def generate_imn_params(imn_trained_data_folder,imn_validation_folder):
    GNN_FILE_PATH = imn_trained_data_folder / 'trained_imn.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(GNN_FILE_PATH, map_location="cpu")
    N_layers = ckpt["N_layers"]
    phases = ckpt["phases"]
    new_model = IMNModel(N_layers,phases)
    new_model.load_state_dict(ckpt["imn"])
    new_model.eval()
    new_model.output_params(imn_validation_folder)

def get_device(use_gpu=True):
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cpu")

def GNNIMN(N_layers, num_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder
            , training_dataset_folder, optimizing_variables, weight_decay, nodes_per_mech_per_phase, use_GPU):



    path = Path(training_dataset_folder)
    path.mkdir(parents=True, exist_ok=True)
    train_hybrid_one_graph(
        N_layers=N_layers,
        num_samples=num_samples,
        num_epochs=num_epochs,
        lr_rest=lr,
        live_plot=cost_live_plot,
        training_dataset_folder=training_dataset_folder,
        imn_trained_data_folder=imn_trained_data_folder,
        optimizing_variables=optimizing_variables,
        weight_decay=weight_decay,
        nodes_per_mech_per_phase=nodes_per_mech_per_phase,
        use_GPU=use_GPU

    )


def IMN(N_layers, num_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder
            , training_dataset_folder):

    train_imn(
        N_layers=N_layers,
        num_samples=num_samples,
        num_epochs=num_epochs,
        lr=lr,
        live_plot=cost_live_plot,
        training_dataset_folder=training_dataset_folder,
        imn_trained_data_folder=imn_trained_data_folder
    )


def extract_engineering_constants(C):
    """
    Extract 9 engineering constants from 6x6 stiffness matrix C.

    Parameters:
        C (6x6 numpy array): stiffness matrix

    Returns:
        dict with E1, E2, E3, nu12, nu13, nu23, G12, G23, G13
    """

    # ----------------------------
    # Check input
    # ----------------------------
    C = np.array(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("C must be a 6x6 matrix")

    # ----------------------------
    # Invert stiffness -> compliance
    # ----------------------------
    S = np.linalg.inv(C)

    # ----------------------------
    # Extract Young's moduli
    # ----------------------------
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    E3 = 1.0 / S[2, 2]

    # ----------------------------
    # Extract Poisson ratios
    # ----------------------------
    nu12 = -S[0, 1] / S[0, 0]
    nu13 = -S[0, 2] / S[0, 0]

    nu21 = -S[1, 0] / S[1, 1]
    nu23 = -S[1, 2] / S[1, 1]

    nu31 = -S[2, 0] / S[2, 2]
    nu32 = -S[2, 1] / S[2, 2]

    # ----------------------------
    # Shear moduli
    # ----------------------------
    G12 = 1.0 / S[3, 3]
    G23 = 1.0 / S[4, 4]
    G13 = 1.0 / S[5, 5]

    # ----------------------------
    # Return dictionary
    # ----------------------------
    return {
        "E1": E1, "E2": E2, "E3": E3,
        "nu12": nu12, "nu13": nu13, "nu23": nu23,
        "G12": G12, "G23": G23, "G13": G13,

    }
