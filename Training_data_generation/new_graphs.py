import numpy as np
import pathlib as path
import os
import sys
id = 1

training_folder = path.Path(f'Training_data{id:04d}')

os.chdir(training_folder)
km = np.load('key_map.npz', allow_pickle=True)
md = np.load('material_dictionary.npz', allow_pickle=True)
os.chdir('Meshes')

for t in ['', '_target_UD1', '_target_MATRIX']:
    for k in km.keys():
        rve = km[k].item()['ids'][1]
        mesh = km[k].item()['ids'][2]
        original_graph_name = 'graph_stage_1_rve_' + str(rve) + '_mesh_' +str(mesh) +t+'.npz'
        new_graph_name = 'graph_with_materials_stage_1_rve_' + str(rve) + '_mesh_' +str(mesh) +t+'.npz'
        original_graph = np.load(original_graph_name)
        print(original_graph.keys())
        new_x = original_graph['x']
        very_new_x = np.zeros((new_x.shape[0], new_x.shape[1]+36))

        very_new_x[:, :10] = new_x[:, :10]




        matrix_mat = md[k].item()['MATRIX'].flatten()
        ud_mat = md[k].item()['UD1'].flatten()
        for i, kk in enumerate(very_new_x):
            if kk[4] == 1:
                kk[10:] = matrix_mat
            elif kk[5] == 1:
                kk[10:] = ud_mat
            else:
                print('Problemmm--------------------------------------')
                break

        new_graph = dict()
        new_graph['x'] = very_new_x
        new_graph['edge_index'] = original_graph['edge_index']
        new_graph['FVC'] = original_graph['FVC']
        new_graph['phase_keys'] = original_graph['phase_keys']
        new_graph['feature_names'] = original_graph['feature_names']


        np.savez(new_graph_name, **new_graph)


