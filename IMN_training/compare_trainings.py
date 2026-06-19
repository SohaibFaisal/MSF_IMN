from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compare_trainings(trained_folder_ids, show, epochs):

    for name, id in trained_folder_ids.items():
        F_IMN_training = 'IMN_training' # DO NOT CHANGE
        imn_trained_data_folder = Path(F_IMN_training + '\\msf' + f"{int(id):04d}")
        x = np.load(imn_trained_data_folder / f'epoch_costs_1.npz')
        # print(x['train'])

        plt.plot(x['train'][0:epochs], label=name)

    # plt.xscale('log')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig('Training_comparisons.png')

    plt.close()



