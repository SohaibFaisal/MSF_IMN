import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compare_validations(load_case, folders, show):

    for l in load_case:
        for name, id in folders.items():
            F_validation = 'IMN_validation'  # DO NOT CHANGE
            _folder = Path(F_validation + '\\Validation' + f"{int(id):04d}\\Val_stage_2_rve_0_mesh_0\\plots")
            x = np.load(_folder / f'LC{l}_stress_IMN.npz')
            y = np.load(_folder / f'LC{l}_strain_IMN.npz')
            plt.plot(x,y, label=name)
        plt.savefig(f'LC{l}_comparison.png')