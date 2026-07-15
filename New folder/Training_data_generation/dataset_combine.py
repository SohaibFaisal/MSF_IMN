from pathlib import Path
import numpy as np
import shutil
import copy


def _to_python_obj(x):
    """
    Convert npz object-array entries back to normal Python objects when needed.
    """
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return x.item()
    return x


def _update_ids_in_obj(obj, new_ids, old_prefix=None, new_prefix=None):
    """
    Recursively update:
      - obj["ids"] = new_ids
      - string graph-name references, if present
    """
    obj = _to_python_obj(obj)

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "ids":
                out[k] = new_ids
            else:
                out[k] = _update_ids_in_obj(v, new_ids, old_prefix, new_prefix)
        return out

    if isinstance(obj, list):
        return [_update_ids_in_obj(v, new_ids, old_prefix, new_prefix) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_update_ids_in_obj(v, new_ids, old_prefix, new_prefix) for v in obj)

    if isinstance(obj, str) and old_prefix is not None and new_prefix is not None:
        return obj.replace(old_prefix, new_prefix)

    return obj


def _find_ids_and_phases(loaded_files, old_key):
    """
    Looks inside homogenize/key_map/material_dictionary entries
    and tries to find 'ids' and 'Phases'.
    """
    old_ids = None
    phases = None

    for d in loaded_files.values():
        entry = _to_python_obj(d[old_key])

        if isinstance(entry, dict):
            if old_ids is None and "ids" in entry:
                old_ids = entry["ids"]

            if phases is None and "Phases" in entry:
                phases = entry["Phases"]

    return old_ids, phases


def combine_datasets_with_meshes(
    source_folders: list[Path],
    combined_folder: Path,
    meshes_subfolder: str = "Meshes",
    file_names: list[str] | None = None,
    new_stage: str = "0",
    new_rve: str = "0",
    copy_all_matching_graphs: bool = True,
):
    """
    Combines multiple datasets and their graph files.

    Expected structure:

        source_folder/
            homogenize.npz
            key_map.npz
            material_dictionary.npz
            Meshes/
                graph_stage_<stage>_rve_<rve>_mesh_<mesh>.npz
                graph_stage_<stage>_rve_<rve>_mesh_<mesh>_target_MATRIX.npz
                graph_stage_<stage>_rve_<rve>_mesh_<mesh>_target_UD1.npz
                ...

    Output:

        combined_folder/
            homogenize.npz
            key_map.npz
            material_dictionary.npz
            Meshes/
                graph_stage_0_rve_0_mesh_0.npz
                graph_stage_0_rve_0_mesh_0_target_MATRIX.npz
                graph_stage_0_rve_0_mesh_0_target_UD1.npz
                graph_stage_0_rve_0_mesh_1.npz
                ...
    """

    source_folders = [Path(p) for p in source_folders]
    combined_folder = Path(combined_folder)
    combined_mesh_folder = combined_folder / meshes_subfolder

    combined_folder.mkdir(exist_ok=True, parents=True)
    combined_mesh_folder.mkdir(exist_ok=True, parents=True)

    if file_names is None:
        file_names = [
            "homogenize.npz",
            "key_map.npz",
            "material_dictionary.npz",
        ]

    combined_data = {fname: {} for fname in file_names}

    new_idx = 0

    for source_folder in source_folders:
        source_mesh_folder = source_folder / meshes_subfolder

        if not source_mesh_folder.exists():
            raise FileNotFoundError(f"Missing mesh folder: {source_mesh_folder}")

        loaded_files = {}

        for fname in file_names:
            fpath = source_folder / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing file: {fpath}")

            loaded_files[fname] = np.load(fpath, allow_pickle=True)

        ref_keys = sorted(
            loaded_files[file_names[0]].files,
            key=lambda x: int(x)
        )

        # Check all npz files have the same sample keys
        for fname in file_names:
            keys = sorted(loaded_files[fname].files, key=lambda x: int(x))
            if keys != ref_keys:
                raise ValueError(
                    f"Key mismatch in {source_folder / fname}. "
                    f"All dataset files must contain the same sample keys."
                )

        for old_key in ref_keys:
            old_ids, phases = _find_ids_and_phases(loaded_files, old_key)

            if old_ids is None:
                # Fallback assumption if ids are not stored inside the dataset entry
                old_stage = "0"
                old_rve = "0"
                old_mesh = str(old_key)
            else:
                old_stage, old_rve, old_mesh = [str(v) for v in old_ids]

            new_mesh = str(new_idx)
            new_ids = (str(new_stage), str(new_rve), new_mesh)

            old_prefix = f"graph_stage_{old_stage}_rve_{old_rve}_mesh_{old_mesh}"
            new_prefix = f"graph_stage_{new_stage}_rve_{new_rve}_mesh_{new_mesh}"

            # Save reindexed dataset entries
            for fname in file_names:
                old_entry = loaded_files[fname][old_key]

                new_entry = _update_ids_in_obj(
                    old_entry,
                    new_ids=new_ids,
                    old_prefix=old_prefix,
                    new_prefix=new_prefix,
                )

                combined_data[fname][str(new_idx)] = new_entry

            # Copy graph files
            if copy_all_matching_graphs:
                matching_graphs = list(source_mesh_folder.glob(old_prefix + "*.npz"))

                if len(matching_graphs) == 0:
                    raise FileNotFoundError(
                        f"No graph files found for prefix:\n"
                        f"  {source_mesh_folder / (old_prefix + '*.npz')}"
                    )

                for old_graph_path in matching_graphs:
                    new_name = old_graph_path.name.replace(old_prefix, new_prefix, 1)
                    new_graph_path = combined_mesh_folder / new_name
                    shutil.copy2(old_graph_path, new_graph_path)

            else:
                # Copy main graph
                old_main = source_mesh_folder / f"{old_prefix}.npz"
                new_main = combined_mesh_folder / f"{new_prefix}.npz"

                if not old_main.exists():
                    raise FileNotFoundError(f"Missing main graph: {old_main}")

                shutil.copy2(old_main, new_main)

                # Copy target graphs
                if phases is None:
                    raise ValueError(
                        f"Could not find 'Phases' for sample {old_key} in {source_folder}. "
                        f"Set copy_all_matching_graphs=True or ensure Phases exists."
                    )

                for ph in phases:
                    old_target = source_mesh_folder / f"{old_prefix}_target_{ph}.npz"
                    new_target = combined_mesh_folder / f"{new_prefix}_target_{ph}.npz"

                    if not old_target.exists():
                        raise FileNotFoundError(f"Missing target graph: {old_target}")

                    shutil.copy2(old_target, new_target)

            new_idx += 1

        for d in loaded_files.values():
            d.close()

    # Save combined dataset files
    for fname in file_names:
        output_path = combined_folder / fname
        np.savez_compressed(output_path, **combined_data[fname])

    print(f"Combined {new_idx} samples.")
    print(f"Saved dataset to: {combined_folder}")
    print(f"Saved graphs to:  {combined_mesh_folder}")