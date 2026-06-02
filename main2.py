import os

from IMN_training.GNN_IMN import *
from IMN_validation.Test_IMN import *
from datetime import datetime as dt
from Training_data_generation.data_generation import *

start = dt.now()

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
TNN_hidden_dim = 128
GNN_hidden_dim = 128
X_feat = 128
GNN_structure = 4
optimizing_variables = [TNN_hidden_dim, GNN_hidden_dim, X_feat, GNN_structure]
# -------------------------------------


# -------------------------------------
# FOLDER NUMBERS
# -------------------------------------
SIM_NAME = 'OLA'
main_id = 444
data_gen_folder_id = main_id  # Change here if needed
train_folder_id = main_id  # Change here if needed
validation_folder_id = main_id  # Change here if needed
# -------------------------------------
training_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(data_gen_folder_id):04d}")  # Remove later
imn_trained_data_folder = Path(F_IMN_training + '\\msf' + f"{int(train_folder_id):04d}")
imn_validation_folder = Path(F_IMN_validation + '\\Validation' + f"{int(validation_folder_id):04d}")
mesh_folder = training_dataset_folder / 'Meshes'

training_dataset_folder.mkdir(exist_ok=True)
mesh_folder.mkdir(exist_ok=True)
imn_trained_data_folder.mkdir(exist_ok=True)
imn_validation_folder.mkdir(exist_ok=True)
# -------------------------------------


# -------------------------------------
# PROBLEM DEFINITION
# -------------------------------------
training_mode = 'IMN'  # GNN_IMN, IMN,

training_data_generation = True
show_mesh = True

imn_training = False
cost_live_plot = True

imn_validation = False
val_solve = True
val_plot = True

imn_validation_2 = False  # For elastic constants error
# -------------------------------------


# -------------------------------------
# TRAINING DATA GENERATION
# -------------------------------------
if training_data_generation:
    stage = 1
    mesh_size = 0.25
    fiber_collision_tolerance = mesh_size / 2
    smallest_volume_tolerance = mesh_size / 2
    strain = 0.02
    materials_per_mesh = 50
    mesh_per_config = 10

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
        # {'MATRIX': {'size': [0.2, 3.0, 3.0]},
        #  'UD1': {'FVC': 0.22, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [0.2, 3.0, 3.0]},
        #  'UD1': {'FVC': 0.1, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD2': {'FVC': 0.1, 'dia': [0.15, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [0.2, 4.0, 4.0]},
        #  'UD1': {'FVC': 0.08, 'dia': [0.35, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD2': {'FVC': 0.05, 'dia': [0.25, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'UD3': {'FVC': 0.09, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        {'MATRIX': {'size': [5.0, 5.0, 5.0]},
         'SFR1': {'FVC': 0.1, 'dia': [0.65, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [5.0, 3.0, 3.0]},
        #  'SFR1': {'FVC': 0.12, 'dia': [0.5, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [2.5, 0]},
        #  'SFR2': {'FVC': 0.1, 'dia': [0.55, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.5, 0]}},
        # {'MATRIX': {'size': [5.0, 3.0, 3.0]},
        #  'SFR1': {'FVC': 0.05, 'dia': [0.55, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'SFR2': {'FVC': 0.08, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [2, 0]},
        #  'SFR3': {'FVC': 0.09, 'dia': [0.45, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [1.5, 0]}},
        # {'MATRIX': {'size': [3.0, 3.0, 3.0]},
        #  'PR1': {'FVC': 0.22, 'dia': [0.35, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [3.0, 3.0, 3.0]},
        #  'PR1': {'FVC': 0.1, 'dia': [0.25, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR2': {'FVC': 0.1, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},
        # {'MATRIX': {'size': [3.0, 3.0, 3.0]},
        #  'PR1': {'FVC': 0.05, 'dia': [0.15, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR2': {'FVC': 0.08, 'dia': [0.2, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]},
        #  'PR3': {'FVC': 0.09, 'dia': [0.25, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}},

    ]

    total_samples_training = int(len(rve_info_training_data) * mesh_per_config * materials_per_mesh)

    total_samples = materials_per_mesh * mesh_per_config * len(rve_info_training_data)

    for r in range(len(rve_info_training_data)):
        for g_id in range(mesh_per_config):
            for attempts in range(500):
                if attempts == 999:
                    raise (RuntimeError)
                try:
                    create_mesh(rve_info_training_data[r], mesh_size, fiber_collision_tolerance, mesh_folder, show_mesh, stage, r, g_id)
                    print(f'RVE {r}... Mesh {g_id} created ')
                    break
                except:
                    pass

    os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
    os.system(f'copy Extracting_graph_from_mesh.py ' + str(mesh_folder))
    for r in range(len(rve_info_training_data)):
        for g_id in range(mesh_per_config):
            print(f' Creating main abaqus file for RVE: {r} and mesh: {g_id}')
            create_abaqus_main_file(mesh_folder, stage, r, g_id)
            print(f' Creating mesh graph for RVE: {r} and mesh: {g_id}')
            create_mesh_graph(mesh_folder, stage, r, g_id)

    create_material_dict(rve_info_training_data, total_samples, training_dataset_folder, stage, mesh_per_config)
    sample_num = 0
    key_map = {}
    for r in range(len(rve_info_training_data)):
        for g_id in range(mesh_per_config):
            print(f' Creating {materials_per_mesh} abaqus input files for RVE: {r} and mesh: {g_id}')
            create_abaqus_input_files(rve_info_training_data[r], sample_num, materials_per_mesh, training_dataset_folder, mesh_folder, stage, r, g_id, key_map)
            sample_num += materials_per_mesh

    key_map_file = training_dataset_folder / 'key_map.npz'
    np.savez_compressed(str(key_map_file), **key_map, allow_pickle=True)

    solve_abaqus_input_files(total_samples, training_dataset_folder)
    os.system(f"abaqus python homogenize_abaqus.py -- {os.getcwd()}/{str(training_dataset_folder)} 0.01")
    os.system(f"abaqus python homogenize_blocks.py -- {os.getcwd()}/{str(training_dataset_folder)} 0.01")
    cleanup(training_dataset_folder)

# -------------------------------------
# TRAINING
# -------------------------------------
if imn_training:
    N_layers = 5
    num_epochs = 5000
    lr = 1.07e-3
    weight_decay = 5.05e-6

    total_samples = 5  # = materials_per_mesh * mesh_per_config * len(rve_info_training_data) HAS TO BE EQUAL TO THE SAMPLES IN THE DATA FOLDER
    if training_mode == 'GNN_IMN':
        GNNIMN(N_layers, total_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder, optimizing_variables, weight_decay)
    elif training_mode == 'IMN':
        IMN(5, total_samples, num_epochs, lr, cost_live_plot, imn_trained_data_folder, training_dataset_folder)
# -------------------------------------


# -------------------------------------
# VALIDATION
# -------------------------------------
if imn_validation:

    steps = 50
    create_new_mesh = False  # Or use a mesh from the training_data_gen_folder/training_data_id/Meshes
    test_mesh_size = [2, 2, 2]

    '''
    --- MATERIAL SPECIFICATION ---

    Elastic :-                      [1, E, nu]
    Elasto-plastic :-               [2, E, Y0, Yinf, Beta, H_lin]
    Orthotropic :-                  [3, E1, E2, E3, nu12, nu23, nu31, G12, G23, G31] (Not implemented)
    Elastic (Finite strain) :-      [4, E, nu]    

    -------------------------------   
    '''
    mat1 = [2, 50, 0.3, 5, 10, 0, 0]
    mat2 = [1, 200, 0.3, 0.2, 0.2, 0, 0]
    mat3 = [1, 200, 0.3, 0.2, 0.2, 0, 0]
    mat4 = [1, 200, 0.3, 0.2, 0.2, 0, 0]
    mat = [mat1, mat2, mat3, mat4]
    mesh_size = 0.35
    fiber_collision_tolerance = mesh_size / 2
    smallest_volume_tolerance = mesh_size / 2
    stage = 2  # DO NOT CHANGE
    mesh_per_config = 1
    strain = 0.02

    # Create new mesh or
    if create_new_mesh:

        rve_info_validation_data = [
            {'MATRIX': {'size': [0.5, 4.5, 4.5]},
             'UD1': {'FVC': 0.05, 'dia': [0.6, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3.3, 0]}},
        ]

        os.system(f'copy Extracting_graph_from_mesh.py ' + str(mesh_folder))
        os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
        for r in range(len(rve_info_validation_data)):
            for g_id in range(mesh_per_config):
                IMN_material = {}
                phases = list(rve_info_validation_data[r].keys())
                i = 0
                for phase in phases:
                    IMN_material[phase + f' {i + 1} '] = mat[i]
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
                print(f' Creating mesh graph for RVE: {r} and mesh: {g_id}')
                create_mesh_graph(mesh_folder, stage, r, g_id)

                new_folder = imn_validation_folder / f'Val_stage_{stage}_rve_{r}_mesh_{g_id}'
                create_FEAP_validation_files(rve_info_validation_data[r], strain, mesh_folder, new_folder, imn_validation_folder, steps, test_mesh_size, IMN_material, stage, r,
                                             g_id)

                if training_mode == 'GNN_IMN':
                    generate_imn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1)
                elif training_mode == 'IMN':
                    generate_imn_params(imn_trained_data_folder, imn_validation_folder)

                validation(new_folder, val_solve, val_plot, stage, r, g_id, [1, 2, 3, 4, 5, 6])


    else:
        os.system(f'copy temp_abaqus.py ' + str(mesh_folder))
        rve_info_validation_data = [
            {'MATRIX': {'size': [0.2, 3.0, 3.0]},
             'UD1': {'FVC': 0.25, 'dia': [0.3, 0], 'ori': [0, 0, np.pi / 2, 0], 'len': [3, 0]}}
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
                                     IMN_material, stage, r, g_id)

        if training_mode == 'GNN_IMN':
            generate_imn_params_for_new_graph_validation(mesh_folder, phases, imn_trained_data_folder, imn_validation_folder, stage, r, g_id, 0, 0, 1)
        elif training_mode == 'IMN':
            generate_imn_params(imn_trained_data_folder, imn_validation_folder)

        validation(new_folder, val_solve, val_plot, stage, r, g_id, [1, 2, 3, 4, 5, 6])

if imn_validation_2:
    stage = 1
    training_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(220):04d}")  # Remove later
    mesh_folder = training_dataset_folder / 'Meshes'
    const_t, const_p = generate_imn_params_for_new_graph_validation(mesh_folder, 0, imn_trained_data_folder, imn_validation_folder, 0, 0, 0, training_dataset_folder, 90, 2)

    errors = dict()
    for k in const_t[0].keys():
        errors[k] = []
    for ct, cp in zip(const_t, const_p):

        for k in ct.keys():
            errors[k].append(100 * abs((ct[k] - cp[k]) / ct[k]))
    # plot_box(errors)
    # plot_mean_with_scatter(errors)
    plot_just_mean(errors)

# -------------------------------------
# SECONDARY FUNCTIONS
# -------------------------------------
if False:
    # Create a randomized subset from training data
    from Training_data_generation.dataset_subset import dataset_subset

    source_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(115):04d}")  # Define here
    new_dataset_folder = Path(F_Training_data_generation + '\\Training_data' + f"{int(333):04d}")  # Define here
    dataset_subset(source_dataset_folder, new_dataset_folder, 20)

if False:
    from IMN_training.compare_trainings import compare_trainings

    folders = {'name1': 115, 'name2': 640}
    compare_trainings(folders, False)  # Set True to show the plot, False to save it in the main Directory







