import os
import shutil
from IMN_training.GNN_IMN import *
from IMN_validation.Test_IMN import *
from datetime import datetime as dt
from Training_data_generation.data_generation import *
from output_DMN_params_from_pt import export_dmn_params_from_checkpoint
start = dt.now()



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, default=5)
parser.add_argument("--nodes", type=int, default=2)
parser.add_argument("--epochs", type=int, default=3000)
args = parser.parse_args()

print('Running study for : ')
print("Layers:", args.layers)
print("Nodes per mech per phase:", args.nodes)
print("Epochs:", args.epochs)
print('Start at : ', dt.now())
N_layers = args.layers
nodes_per_mech_per_phase = args.nodes
num_epochs = args.epochs
# train_folder_id = f'072{N_layers}'




#N_layers = 4
#nodes_per_mech_per_phase = 1
#num_epochs = 5
#train_folder_id = str(N_layers) + str(nodes_per_mech_per_phase)
# Example:
# train_model(lr=learning_rate, batch_size=batch_size, epochs=epochs)




# -------------------------------------
# DO NOT CHANGE
# -------------------------------------
F_IMN_training = 'IMN_training'
F_IMN_validation = 'IMN_validation'
F_Training_data_generation = 'Training_data_generation'
# -------------------------------------


# -------------------------------------
# HYPER-PARAMETERS
# -------------------------------------

# -------------------------------------


# -------------------------------------
# FOLDER NUMBERS
# -------------------------------------
SIM_NAME = 'OLA'
main_id = 725
data_gen_folder_id = main_id # Change here if needed
# train_folder_id = main_id # Change here if needed
train_folder_id = 825
validation_folder_id = main_id # Change here if needed
# -------------------------------------
training_dataset_folder = Path(F_Training_data_generation + '/Training_data' + f"{int(data_gen_folder_id):04d}") # Remove later
imn_trained_data_folder = Path(F_IMN_training + '/msf' + f"{int(train_folder_id):04d}")
imn_validation_folder = Path(F_IMN_validation + '/Validation' + f"{int(validation_folder_id):04d}")
mesh_folder = training_dataset_folder / 'Meshes'

training_dataset_folder.mkdir(exist_ok=True)
mesh_folder.mkdir(exist_ok=True)
imn_trained_data_folder.mkdir(exist_ok=True)
imn_validation_folder.mkdir(exist_ok=True)
# -------------------------------------


# -------------------------------------
# PROBLEM DEFINITION
# -------------------------------------
training_mode = 'GNN_IMN' # GNN_IMN, IMN, GNN_DMN, DMN

training_data_generation = False
show_mesh = False




imn_training = True
cost_live_plot = True

imn_validation = False
val_solve = True
val_plot = True


imn_validation_2 = False # For elastic constants error
# -------------------------------------


# -------------------------------------
# TRAINING DATA GENERATION
# -------------------------------------
if training_data_generation:
    stage = 1
    mesh_size = 0.3
    fiber_collision_tolerance = mesh_size/2

    smallest_volume_tolerance = mesh_size/2
    strain = 0.02
    materials_per_mesh = 3
    mesh_per_config = 100


    '''
        rve_info_training_data must follow this format.
        
        rve_info_training_data = [RVE1, RVE2, RVE3, ......] (List if Dictionaries)
        
        RVE1 = {Phase1, Phase2, Phase3, ...} (Dictionary of dictionaries)
        
        Phase1 = 'MATRIX' : {'size':[length x, length y, length z]}
        Phase2,3,4... = 'UD1': {'FVC': fvc, 'dia': [avg_dia, dia_normal_dist], 'ori': [avg_theta, theta_norm_dist, avg_phi, phi_norm_dist], 'len': [avg_len, len_norm_dist]}
        For example:
        
        rve_info_training_data = [
            {
        'MATRIX': {'size': [0.2, 3.0, 3.0]},
         'UD1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}
         }, 
         
                 {
        'MATRIX': {'size': [0.2, 3.0, 3.0]},
         'UD1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
         'UD2': {'FVC': 0.15, 'dia': [0.35, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
         'SFR1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
         'PR1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}
         }
         
                 {
        'MATRIX': {'size': [0.2, 3.0, 3.0]},
         'PR1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}
         }

                ]
    '''

    rve_info_training_data = [
        {'MATRIX': {'size': [0.2, 3.0, 3.0]},
         'UD1': {'FVC': 0.22, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [0.2, 3.0, 3.0]},
        #  'UD1': {'FVC': 0.1, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD2': {'FVC': 0.1, 'dia': [0.15, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [0.2, 4.0, 4.0]},
        #  'UD1': {'FVC': 0.08, 'dia': [0.35, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD2': {'FVC': 0.05, 'dia': [0.25, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD3': {'FVC': 0.09, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [4.0, 3.0, 3.0]},
        #  'SFR1': {'FVC': 0.2, 'dia': [0.45, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [4.5, 3.0, 3.0]},
        #  'SFR1': {'FVC': 0.12, 'dia': [0.5, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [2.5, 0]},
        #  'SFR2': {'FVC': 0.1, 'dia': [0.55, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.5, 0]}},
        # {'MATRIX': {'size': [4.0, 3.0, 3.0]},
        #  'SFR1': {'FVC': 0.05, 'dia': [0.55, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'SFR2': {'FVC': 0.08, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [2, 0]},
        #  'SFR3': {'FVC': 0.09, 'dia': [0.45, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [2.5, 0]}},
        # {'MATRIX': {'size': [2.0, 2.0, 2.0]},
        #  'PR1': {'FVC': 0.22, 'dia': [0.5, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [2.0, 2.0, 2.0]},
        #  'PR1': {'FVC': 0.1, 'dia': [0.45, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR2': {'FVC': 0.1, 'dia': [0.4, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [2.0, 2.0, 2.0]},
        #  'PR1': {'FVC': 0.05, 'dia': [0.45, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR2': {'FVC': 0.08, 'dia': [0.42, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR3': {'FVC': 0.09, 'dia': [0.48, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},


                ]

    total_samples_training = int(len(rve_info_training_data)*mesh_per_config*materials_per_mesh)
    total_samples = materials_per_mesh*mesh_per_config*len(rve_info_training_data)

    try:
        for r in range(len(rve_info_training_data)):
            for g_id in range(mesh_per_config):
                for attempts in range(100):
                    if attempts == 99:
                        raise (RuntimeError)
                    try:
                        create_mesh(rve_info_training_data[r], mesh_size, fiber_collision_tolerance, mesh_folder, show_mesh, stage, r,g_id)
                        print(f'RVE {r}... Mesh {g_id} created ')
                        break
                    except Exception as e:
                        last_error = e
                        print(f"Mesh attempt {attempts} failed for RVE {r}, mesh {g_id}: {e}")


        # os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
        # os.system(f'copy Extracting_graph_from_mesh.py ' + str(mesh_folder))

        shutil.copy("temp_abaqus.py", mesh_folder)
        shutil.copy("Extracting_graph_from_mesh_DMN.py", mesh_folder)
        shutil.copy("Extracting_graph_from_mesh.py", mesh_folder)

        for r in range(len(rve_info_training_data)):
            for g_id in range(mesh_per_config):
                print(f' Creating main abaqus file for RVE: {r} and mesh: {g_id}')
                create_abaqus_main_file(mesh_folder, stage, r, g_id)

        # solve_abaqus_main_files(int(len(rve_info_training_data)), training_dataset_folder)

    except Exception as e:
        raise e




    for r in range(len(rve_info_training_data)):
        for g_id in range(mesh_per_config):
            print(f' Creating mesh graph for RVE: {r} and mesh: {g_id}')
            if 'GNN' in training_mode:
                create_mesh_graph(mesh_folder, stage, r, g_id, training_mode)



    create_material_dict(rve_info_training_data, total_samples, training_dataset_folder,stage, mesh_per_config)
    sample_num = 0
    key_map = {}
    for r in range(len(rve_info_training_data)):
        for g_id in range(mesh_per_config):
            print(f' Creating {materials_per_mesh} abaqus input files for RVE: {r} and mesh: {g_id}')
            create_abaqus_input_files(rve_info_training_data[r],sample_num, materials_per_mesh, training_dataset_folder,mesh_folder, stage,r, g_id, key_map)
            sample_num += materials_per_mesh

    key_map_file = training_dataset_folder / 'key_map.npz'
    np.savez_compressed(str(key_map_file), **key_map, allow_pickle=True)

    solve_abaqus_input_files(total_samples, training_dataset_folder)
    os.system(f"abaqus python homogenize_abaqus.py -- {os.getcwd()}/{str(training_dataset_folder)} 0.01")
    # os.system(f"abaqus python homogenize_blocks.py -- {os.getcwd()}/{str(training_dataset_folder)} 0.01")
    cleanup(training_dataset_folder)







# -------------------------------------
# TRAINING
# -------------------------------------
if imn_training:
    # N_layers = 4
    # num_epochs = 11
    lr = 4.5e-3
    weight_decay = 5.05e-8
    # nodes_per_mech_per_phase = 2
    use_GPU = False

    tnn_hidden_dim = 128
    gnn_hidden_dim = 64
    gnn_heads = 8
    x_feat = 128
    gnn_structure = 2
    nodes_per_mech_per_phase = 2
    tnn_layers = 3
    gnn_layers = 1

    optimizing_variables = [
        tnn_hidden_dim,
        gnn_hidden_dim,
        gnn_heads,
        x_feat,
        gnn_structure,
        nodes_per_mech_per_phase,
        tnn_layers,
        gnn_layers,
    ]

    total_samples = 300 # = materials_per_mesh * mesh_per_config * len(rve_info_training_data) HAS TO BE EQUAL TO THE SAMPLES IN THE DATA FOLDER
    Train(N_layers,total_samples,num_epochs,lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder, optimizing_variables, weight_decay, nodes_per_mech_per_phase, use_GPU, training_mode)
    # if training_mode == 'GNN_IMN':
    #     GNNIMN(N_layers,total_samples,num_epochs,lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder, optimizing_variables, weight_decay, nodes_per_mech_per_phase, use_GPU)
    # elif training_mode == 'IMN':
    #     IMN(5, total_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder)
    # elif training_mode == 'GNN_DMN':
    #     GNNIMN(N_layers,total_samples,num_epochs,lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder, optimizing_variables, weight_decay, nodes_per_mech_per_phase, use_GPU)
    # elif training_mode == 'DMN':
    #     DMN(5, total_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder)
# -------------------------------------






# -------------------------------------
# VALIDATION
# -------------------------------------
if imn_validation:

    steps = 100
    create_new_mesh = True # Or use a mesh from the training_data_gen_folder/training_data_id/Meshes
    test_mesh_size = [2,2,2]
    nodes_per_mech_per_phase = 2

    '''
    --- MATERIAL SPECIFICATION ---
    
    Elastic :-                      [1, E, nu]
    Elasto-plastic :-               [2, E, Y0, Yinf, Beta, H_lin]
    Orthotropic :-                  [3, E1, E2, E3, nu12, nu23, nu31, G12, G23, G31] (Not implemented)
    Elastic (Finite strain) :-      [4, E, nu]    
    
    -------------------------------   
    '''
    mat1 = [2, 1500, 0.35, 10.0, 15.0, 20, 15]
    mat2 = [1, 1500*5, 0.2, 0.2, 0.2, 0, 0]
    mat3 = [1, 300, 0.3, 0.2, 0.2, 0, 0]
    mat4 = [1, 300, 0.3, 0.2, 0.2, 0, 0]
    mat = [mat1, mat2, mat3, mat4]
    mesh_size = 0.25
    fiber_collision_tolerance = mesh_size / 2
    smallest_volume_tolerance = mesh_size / 2
    stage = 2 # DO NOT CHANGE
    mesh_per_config = 1
    strain = 0.02

    # DMN



    # Create new mesh or
    if create_new_mesh:

        rve_info_validation_data = [
            {'MATRIX': {'size': [0.5, 4.5, 4.5]},
             'UD1': {'FVC': 0.22, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.3, 0]}},
            # {'MATRIX': {'size': [0.5, 4.5, 4.5]},
            #  'UD1': {'FVC': 0.32, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.3, 0]}},
            # {'MATRIX': {'size': [4.5, 4, 4]},
            #  'UD1': {'FVC': 0.08, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.3, 0]},
            # 'SFR1': {'FVC': 0.08, 'dia': [0.65, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.5, 0]},
            # 'PR1': {'FVC': 0.08, 'dia': [0.8, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.3, 0]}},
        ]


        os.system(f'copy Extracting_graph_from_mesh.py ' + str(mesh_folder))
        shutil.copy("Extracting_graph_from_mesh_DMN.py", mesh_folder)
        os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
        for r in range(len(rve_info_validation_data)):
            for g_id in range(mesh_per_config):
                IMN_material = {}
                phases = list(rve_info_validation_data[r].keys())
                i = 0
                for phase in phases:
                    IMN_material[phase + f' {i+1} '] = mat[i]
                    i += 1
                #

                for attempts in range(200):
                    if attempts == 199:
                        raise (RuntimeError)
                    try:
                        create_mesh(rve_info_validation_data[r], mesh_size, fiber_collision_tolerance, mesh_folder, show_mesh, stage, r, g_id)
                        print(f'RVE {r}... Mesh {g_id} created ')
                        break
                    except:
                        pass
                if 'GNN' in training_mode:
                    create_mesh_graph(mesh_folder, stage, r, g_id, training_mode)
                    print(f' Creating mesh graph for RVE: {r} and mesh: {g_id}')


                new_folder = imn_validation_folder / f'Val_stage_{stage}_rve_{r}_mesh_{g_id}'
                create_FEAP_validation_files(rve_info_validation_data[r], strain, mesh_folder, new_folder,imn_validation_folder, steps, test_mesh_size, IMN_material, stage, r, g_id, training_mode)


                if training_mode == 'GNN_IMN':
                    generate_imn_params_for_new_graph_validation(mesh_folder,phases,imn_trained_data_folder,imn_validation_folder, stage, r, g_id, 0,0,1, nodes_per_mech_per_phase)
                elif training_mode == 'IMN':
                    generate_imn_params(imn_trained_data_folder, imn_validation_folder)
                elif training_mode == 'DMN':
                    generate_dmn_params(imn_trained_data_folder, imn_validation_folder)
                elif training_mode == 'GNN_DMN':
                    generate_dmn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1)

                validation(new_folder, True,val_plot, stage, r, g_id, [1,2,3], )


    else:
        os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
        rve_info_validation_data = [
            {'MATRIX': {'size': [0.2, 3.0, 3.0]},
             'UD1': {'FVC': 0.23, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}}
        ]
        # Has to be same as the one used to create the existing mesh

        r = 0
        g_id = 0
        stage = 1
        new_folder = imn_validation_folder / f'Val_stage_{stage}_rve_{r}_mesh_{g_id}'
        phases = list(rve_info_validation_data[r].keys())
        i = 0
        IMN_material = {}
        for phase in phases:
            IMN_material[phase + f' {i + 1} '] = mat[i]
            i += 1

        create_FEAP_validation_files(rve_info_validation_data[r], strain, mesh_folder,
                                     new_folder, imn_validation_folder, steps, test_mesh_size,
                                     IMN_material, stage, r, g_id, 'DMN')

        # if training_mode == 'GNN_IMN':
        #     generate_imn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1)
        # elif training_mode == 'IMN':
        #     generate_imn_params(imn_trained_data_folder, imn_validation_folder)
        if training_mode == 'GNN_IMN':
            generate_imn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1, nodes_per_mech_per_phase)
        elif training_mode == 'IMN':
            generate_imn_params(imn_trained_data_folder, imn_validation_folder)
        elif training_mode == 'DMN':
            generate_dmn_params(imn_trained_data_folder, imn_validation_folder)
        elif training_mode == 'GNN_DMN':
            generate_dmn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1)

        validation(new_folder, val_solve, val_plot, stage, r, g_id, [1,2,3,4, 5,6])






if imn_validation_2:
    stage = 1
    training_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(220):04d}") # Remove later
    mesh_folder = training_dataset_folder / 'Meshes'
    const_t, const_p = generate_imn_params_for_new_graph_validation(mesh_folder,0,imn_trained_data_folder,imn_validation_folder,0,0,0, training_dataset_folder,90,2)

    errors = dict()
    for k in const_t[0].keys():
        errors[k] = []
    for ct, cp in zip(const_t, const_p):

        for k in ct.keys():
            errors[k].append(100*abs((ct[k]-cp[k])/ct[k]))
    # plot_box(errors)
    # plot_mean_with_scatter(errors)
    plot_just_mean(errors)



# -------------------------------------
# SECONDARY FUNCTIONS
# -------------------------------------
if False:
    # Create a randomized subset from training data
    from Training_data_generation.dataset_subset import dataset_subset
    source_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(724):04d}")  # Define here
    new_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(924):04d}")  # Define here
    dataset_subset(source_dataset_folder, new_dataset_folder,500)

if False:
    # Create combined training graphs
    from IMN_training.compare_trainings import compare_trainings
    folders = {'GNN DMN - 5 Layers': 725,
               'GNN DMN - 6 Layers': 726,
               'GNN IMN - 5 Layers': 825,
               # 'Layers: 3 - Nodes: 3': 33,
               # 'Layers: 4 - Nodes: 1': 41,
               # 'Layers: 4 - Nodes: 2': 42,
               # 'Layers: 4 - Nodes: 3': 43,
               # 'Layers: 5 - Nodes: 1': 51,
               # 'Layers: 5 - Nodes: 2': 52,
               # 'Layers: 5 - Nodes: 3': 53
               }
    compare_trainings(folders, True, 250) # Set True to show the plot, False to save it in the main Directory

if False:
    # Compare validation results
    from IMN_validation.compare_validations import compare_validations

    load_case = [1,2,3]
    folders = {'Layers: 5 - DMN': 726,
               'Layers: 5 - IMN': 725,
               }
    compare_validations(load_case, folders, True)


if False:
    # Join different training datasets
    from Training_data_generation.dataset_combine import combine_datasets_with_meshes
    source_folders = list(Path(F_Training_data_generation + '\\Training_data' + f"{int(idx):04d}") for idx in [721,722,723])
    combined_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(724):04d}")  # Define here
    combine_datasets_with_meshes([source_folders][0], combined_dataset_folder)


print('End at : ', dt.now())




