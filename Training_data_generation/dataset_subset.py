
from pathlib import Path
import numpy as np
import random


def dataset_subset(source_folder:Path, combined_folder:Path, samples:int):

    combined_folder.mkdir(exist_ok=True)

    files = [source_folder/f for f in ["homogenize.npz", "key_map.npz", "material_dictionary.npz"]]
    data = np.load(files[0], allow_pickle=True)
    keys = sorted(data.files, key=lambda x: int(x))
    selected_keys = random.sample(keys, samples)
    for f in files:
        d = np.load(f, allow_pickle=True)
        new_dict = {}
        for new_idx, old_key in enumerate(selected_keys, start=0):
            new_dict[str(new_idx)] = d[old_key]

        output_path = combined_folder / Path(f).name
        np.savez(output_path, **new_dict)




