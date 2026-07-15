import os
from pathlib import Path
from .plotting_results import plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

def plot_box_with_mean(error_dict):
    keys = list(error_dict.keys())
    data = [error_dict[k] for k in keys]

    plt.figure()

    # Boxplot
    bp = plt.boxplot(data, labels=keys, showmeans=True)

    # Compute means
    means = [np.mean(d) for d in data]

    # Overlay mean values and annotate
    for i, mean in enumerate(means):
        plt.scatter(i+1, mean)  # boxplot positions start at 1
        # plt.hlines(mean, i + 0.8, i + 1.2, linestyles='solid')
        # plt.text(i + 1, mean, f"{mean:.2f}", ha='center', va='bottom')

    plt.ylabel("Percentage Error (%)")
    plt.title("Error Distribution with Mean Values")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_box(error_dict):
    keys = list(error_dict.keys())
    data = [error_dict[k] for k in keys]

    plt.figure()
    plt.boxplot(data, labels=keys)

    plt.ylabel("Percentage Error (%)")
    plt.title("Error Distribution per Elastic Constant")

    ax = plt.gca()
    # Major grid (existing)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ----------------------------
    # Add minor ticks every 0.01
    # ----------------------------
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    # Enable minor grid
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('easd.png')
    # plt.show()

def plot_mean_with_scatter(error_dict):
    keys = list(error_dict.keys())
    x = np.arange(len(keys))

    plt.figure()

    for i, k in enumerate(keys):
        y = error_dict[k]
        plt.scatter([i]*len(y), y, alpha=0.3, color='black')

    means = [np.mean(error_dict[k]) for k in keys]
    plt.plot(x, means, marker='o')

    plt.xticks(x, keys)
    plt.ylabel("Percentage Error (%)")
    plt.title("Error Distribution + Mean")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("mean_error.png")
    # plt.show()


def plot_just_mean(error_dict):
    keys = list(error_dict.keys())
    x = np.arange(len(keys))

    means = [np.mean(error_dict[k]) for k in keys]

    plt.figure(figsize=(6, 4))

    # Slim, elegant bars
    plt.bar(x, means, width=0.5)

    # Minimal styling
    plt.xticks(x, keys, fontsize=10)
    plt.ylabel("Percentage Error (%)", fontsize=11)
    plt.title("Elastic constants prediction error", fontsize=12)

    # Clean grid (subtle)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Remove top/right spines for publication look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Slight padding
    plt.tight_layout()

    plt.savefig("mean_error.svg", format='svg')
    plt.close()
    # plt.show()


def plot_just_mean_multi(error_dicts, labels=None, save_path="mean_error.svg"):
    """
    error_dicts: list of dicts, e.g. [err_case1, err_case2, err_case3]
    labels: legend labels, e.g. ["GNN", "DMN", "GNN-DMN"]
    """

    if labels is None:
        labels = [f"Case {i+1}" for i in range(len(error_dicts))]

    # Use keys from first dict
    keys = list(error_dicts[0].keys())
    x = np.arange(len(keys))

    n_cases = len(error_dicts)
    bar_width = 0.8 / n_cases

    plt.figure(figsize=(7, 4))

    for i, error_dict in enumerate(error_dicts):
        means = [np.mean(error_dict[k]) for k in keys]

        offset = (i - (n_cases - 1) / 2) * bar_width
        plt.bar(
            x + offset,
            means,
            width=bar_width,
            label=labels[i]
        )

    plt.xticks(x, keys, fontsize=10)
    plt.ylabel("Percentage Error (%)", fontsize=11)
    plt.title("Elastic constants prediction error", fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(save_path, format="svg")
    plt.close()

def plot_error_summary(error_dict):
    """
    error_dict: dict like
    {
        'E1': [...],
        'E2': [...],
        ...
    }
    """

    keys = list(error_dict.keys())

    means = []
    stds  = []

    for k in keys:
        data = np.array(error_dict[k])
        means.append(np.mean(data))
        stds.append(np.std(data))

    x = np.arange(len(keys))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=5)

    plt.xticks(x, keys)
    plt.ylabel("Percentage Error (%)")
    plt.title("Model Performance on Elastic Constants")

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("error_summary.png")
    # plt.show()




def validation(new_folder,val_solve,val_plot, stage,r,g_id, loads_cases):


    if val_solve:
        # new_folder = imn_validation_folder / f'Val_stage_{stage}_rve_{r}_mesh_{g_id}'
        with open(new_folder / f'i_all_tests.bat', 'w') as load_file:
            #for i in ['DNS','IMN']:
            for i in ['IMN']:
                # for x in range(1, 7):
                for x in loads_cases:
                    load_file.write(f'feap86 -iI_val_{i}_stage_{stage}_{x}\n')


        base_dir = Path(f'{new_folder}')
        os.system('copy feap86.exe {}'.format(base_dir))
        os.chdir(base_dir)
        os.system('i_all_tests')
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')

    if val_plot:
        stress = []
        strain = []
        counter = 1
        no_loads = []
        yes_loads = []


        for i in ['DNS']:

            for x in range(1,7):
                load_point = 0
                stress_list = []
                strain_list = []

                try:
                    with open(new_folder / f'O_val_{i}_stage_{stage}_{x}', 'r') as load_file:
                        load_data = load_file.readlines()
                        reading = False
                        another_counter = 0
                        for line in load_data:
                            if 'Material All' in line:
                                load_point += 1
                                reading = True
                                another_counter = 0
                                continue

                            if reading:
                                another_counter += 1
                                if another_counter ==  3:
                                    stress_list.append(line)
                                elif another_counter == 6:
                                    strain_list.append(line)
                                elif another_counter == 7:
                                    reading = False
                                    continue

                        counter += 2
                        stress.append(stress_list)
                        strain.append(strain_list)
                    print(f'for {i} load case {x}, total load points are {load_point}')
                    yes_loads.append(x)
                except:
                    no_loads.append(x)
                    stress.append(stress_list)
                    strain.append(strain_list)
                    print(f'for {i} load case {x}, no data available')




        stress_normal = []
        strain_normal = []
        load_case_number = 1




        for p in stress:

            if load_case_number in no_loads:
                q = stress[int(yes_loads[0]-1)]
                for pp in q:
                    s = [float(x) for x in pp.split()]
                    stress_normal.append(s)
            else:
                for pp in p:
                    s = [float(x) for x in pp.split()]
                    stress_normal.append(s)
            load_case_number += 1


        load_case_number = 1
        for p in strain:
            if load_case_number in no_loads:
                q = strain[int(yes_loads[0]-1)]
                for pp in q:
                    s = [float(x) for x in pp.split()]
                    strain_normal.append(s)
            else:
                for pp in p:
                    s = [float(x) for x in pp.split()]
                    strain_normal.append(s)
            load_case_number += 1



        stress = []
        strain = []
        counter = 2
        for i in ['IMN']:
            for x in range(1,7):
                load_point = 0
                stress_list = []
                strain_list = []
                if x not in no_loads:
                    with open(new_folder / f'O_val_{i}_stage_{stage}_{x}', 'r') as load_file:
                        load_data = load_file.readlines()
                        reading = False
                        another_counter = 0
                        for line in load_data:
                            if 'Material All' in line:
                                load_point += 1
                                reading = True
                                another_counter = 0
                                continue

                            if reading:
                                another_counter += 1
                                if another_counter ==  3:
                                    stress_list.append(line)
                                elif another_counter == 6:
                                    strain_list.append(line)
                                elif another_counter == 7:
                                    reading = False
                                    continue
                        counter += 2
                        stress.append(stress_list)
                        strain.append(strain_list)
                    print(f'for {i} load case {x}, total load points are {load_point}')
                else:
                    stress.append(stress_list)
                    strain.append(strain_list)
                    print(f'for {i} load case {x}, no data available')

        stress_IMN = []
        strain_IMN = []
        load_case_number = 1
        for p in stress:

            if load_case_number in no_loads:

                q = stress[int(yes_loads[0] - 1)]
                for pp in q:
                    s = [float(x) for x in pp.split()]
                    stress_IMN.append(s)

            else:
                for pp in p:
                    s = [float(x) for x in pp.split()]
                    stress_IMN.append(s)
            load_case_number += 1




        load_case_number = 1
        for p in strain:
            if load_case_number in no_loads:
                q = strain[int(yes_loads[0]-1)]
                for pp in q:
                    s = [float(x) for x in pp.split()]
                    strain_IMN.append(s)
            else:
                for pp in p:
                    s = [float(x) for x in pp.split()]
                    strain_IMN.append(s)

            load_case_number += 1

        # print(len(stress_normal))
        # print(len(strain_normal))

        plot(new_folder, stress_normal, strain_normal, stress_IMN, strain_IMN,  )





