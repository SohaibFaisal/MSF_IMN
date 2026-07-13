from pathlib import Path
import random
import numpy as np


def dataset_subset(
    source_folder: Path,
    combined_folder: Path,
    samples: int,
    seed: int | None = None,
):
    combined_folder.mkdir(parents=True, exist_ok=True)

    filenames = [
        "homogenize.npz",
        "key_map.npz",
        "material_dictionary.npz",
    ]

    file_paths = [source_folder / filename for filename in filenames]

    # Load all archives
    archives = {
        path.name: np.load(path, allow_pickle=True)
        for path in file_paths
    }

    try:
        # Find keys available in every archive
        common_keys = set.intersection(
            *(set(archive.files) for archive in archives.values())
        )

        common_keys = sorted(common_keys, key=lambda key: int(key))

        if samples > len(common_keys):
            raise ValueError(
                f"Requested {samples} samples, but only "
                f"{len(common_keys)} common samples exist."
            )

        # Reproducible random selection when seed is provided
        rng = random.Random(seed)
        selected_keys = rng.sample(common_keys, samples)

        print(f"Common samples available: {len(common_keys)}")
        print(f"Selected original keys: {selected_keys}")

        for filename, archive in archives.items():
            new_dict = {
                str(new_idx): archive[old_key]
                for new_idx, old_key in enumerate(selected_keys)
            }

            output_path = combined_folder / filename
            np.savez_compressed(output_path, **new_dict)

            print(f"Saved {len(new_dict)} samples to: {output_path}")

    finally:
        # Properly close all NpzFile objects
        for archive in archives.values():
            archive.close()