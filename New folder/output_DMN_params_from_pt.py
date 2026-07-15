from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional
import argparse
import numpy as np
import torch


def _as_numpy_real64(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _write_shape(path: Path, shape) -> None:
    arr = np.asarray(list(shape), dtype=np.int32)
    arr.tofile(path)


def _write_bin(path: Path, array: np.ndarray) -> None:
    # Fortran reads stream/unformatted real64 directly. 1D order is irrelevant.
    np.asfortranarray(array, dtype=np.float64).ravel(order="F").tofile(path)


def _find_state_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
        return obj
    raise TypeError("Checkpoint must be a torch state_dict or a dict containing a state_dict.")


def _get_first_key(sd: Mapping[str, Any], suffixes: list[str]) -> Any:
    # Allows keys such as dmn.z, module.dmn.z, z, etc.
    for suffix in suffixes:
        for k, v in sd.items():
            if k == suffix or k.endswith("." + suffix):
                return v
    raise KeyError(f"Could not find any of these parameter keys: {suffixes}")


def export_dmn_params_from_checkpoint(
    checkpoint_path: str | Path,
    out_dir: str | Path,
    N_layers: Optional[int] = None,
    phase_names: Optional[list[str]] = None,
) -> None:
    """
    Exports trained 3D DMN parameters from a .pt checkpoint into Fortran-readable
    .bin and .shape files.

    Expected DMN parameter layout:
        p = [z_bottom, alpha_all, beta_all, gamma_all]

    Files written:
        dmn_z.bin,     dmn_z.shape
        dmn_alpha.bin, dmn_alpha.shape
        dmn_beta.bin,  dmn_beta.shape
        dmn_gamma.bin, dmn_gamma.shape
        dmn_meta.bin,  dmn_meta.shape

    dmn_meta = [N_layers, bottom_nodes, total_nodes, n_phases]
    """
    checkpoint_path = Path(checkpoint_path) / 'gnn_imn_generator.pt'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = _find_state_dict(ckpt)

    # Option A: checkpoint has separated DMN parameters.
    try:
        z = _as_numpy_real64(_get_first_key(sd, ["z", "dmn.z"]))
        alpha = _as_numpy_real64(_get_first_key(sd, ["alpha", "dmn.alpha"]))
        beta = _as_numpy_real64(_get_first_key(sd, ["beta", "dmn.beta"]))
        gamma = _as_numpy_real64(_get_first_key(sd, ["gamma", "dmn.gamma"]))
    except KeyError:
        # Option B: checkpoint saved a flat p_hat / p_flat.
        p = _as_numpy_real64(_get_first_key(sd, ["p_hat", "p_flat", "flat_p", "dmn_flat_params"]))
        if N_layers is None:
            if isinstance(ckpt, Mapping) and "N_layers" in ckpt:
                N_layers = int(ckpt["N_layers"])
            else:
                raise ValueError("N_layers is required when checkpoint only contains a flat parameter vector.")
        bottom_nodes = 2 ** (N_layers - 1)
        total_nodes = 2 ** N_layers - 1
        expected = bottom_nodes + 3 * total_nodes
        if p.size != expected:
            raise ValueError(f"Flat parameter vector has {p.size} values, expected {expected} for N_layers={N_layers}.")
        i = 0
        z = p[i:i + bottom_nodes]; i += bottom_nodes
        alpha = p[i:i + total_nodes]; i += total_nodes
        beta = p[i:i + total_nodes]; i += total_nodes
        gamma = p[i:i + total_nodes]

    z = z.reshape(-1)
    alpha = alpha.reshape(-1)
    beta = beta.reshape(-1)
    gamma = gamma.reshape(-1)

    bottom_nodes = z.size
    total_nodes = alpha.size
    if beta.size != total_nodes or gamma.size != total_nodes:
        raise ValueError("alpha, beta and gamma must have the same length.")

    if N_layers is None:
        # total_nodes = 2**N - 1
        N_layers_float = np.log2(total_nodes + 1)
        if abs(N_layers_float - round(N_layers_float)) > 1e-12:
            raise ValueError("Could not infer N_layers from total_nodes. Pass N_layers explicitly.")
        N_layers = int(round(N_layers_float))

    n_phases = len(phase_names) if phase_names is not None else int(0)
    meta = np.asarray([N_layers, bottom_nodes, total_nodes, n_phases], dtype=np.float64)

    arrays = {
        "dmn_z": z,
        "dmn_alpha": alpha,
        "dmn_beta": beta,
        "dmn_gamma": gamma,
        "dmn_meta": meta,
    }
    for name, arr in arrays.items():
        _write_bin(out_dir / f"{name}.bin", arr)
        _write_shape(out_dir / f"{name}.shape", [arr.size])

    if phase_names is not None:
        (out_dir / "dmn_phase_names.txt").write_text("\n".join(phase_names) + "\n", encoding="utf-8")

    print(f"Exported DMN parameters to: {out_dir}")
    print(f"N_layers={N_layers}, bottom_nodes={bottom_nodes}, total_nodes={total_nodes}, param_dim={bottom_nodes + 3*total_nodes}")





