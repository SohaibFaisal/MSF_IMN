from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compare_validation_errors(
    load_case,
    folders,
    show=True,
    save_folder=".",
    dpi=600,
    error_method="normalized_absolute",
    dns_mode="separate",
):
    """
    Plot pointwise errors between model stress curves and DNS curves.

    Parameters
    ----------
    load_case : iterable
        Load-case numbers.

    folders : dict
        Dictionary of curve names and validation folder IDs.

        Example:
            folders = {
                "DMN": 101,
                "IMN": 102,
            }

    show : bool, optional
        Display figures when True.

    save_folder : str or pathlib.Path, optional
        Folder in which figures are saved.

    dpi : int, optional
        Figure resolution.

    error_method : str, optional
        Error calculation method:

        "absolute"
            abs(model - DNS)

        "relative"
            abs(model - DNS) / abs(DNS) * 100

        "normalized_absolute"
            abs(model - DNS) / max(abs(DNS)) * 100

        "symmetric_relative"
            2 * abs(model - DNS)
            / (abs(model) + abs(DNS)) * 100

        "range_normalized"
            abs(model - DNS)
            / (max(DNS) - min(DNS)) * 100

    dns_mode : str, optional
        Controls which DNS curve is used:

        "first"
            Load DNS from the first folder and use it for all model curves.

        "separate"
            Load DNS separately from the same folder as each model curve.

    Returns
    -------
    errors : dict
        Nested dictionary containing all calculated error quantities.
    """

    valid_error_methods = {
        "absolute",
        "relative",
        "normalized_absolute",
        "symmetric_relative",
        "range_normalized",
    }

    valid_dns_modes = {
        "first",
        "separate",
    }

    if error_method not in valid_error_methods:
        raise ValueError(
            f"Unknown error_method '{error_method}'. "
            f"Choose from {sorted(valid_error_methods)}."
        )

    if dns_mode not in valid_dns_modes:
        raise ValueError(
            f"Unknown dns_mode '{dns_mode}'. "
            f"Choose from {sorted(valid_dns_modes)}."
        )

    if not folders:
        raise ValueError("'folders' cannot be empty.")

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "savefig.dpi": dpi,
    })

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # colors = {
    #     "1": "#9ECAE1",
    #     "2": "#6BAED6",
    #     "3": "#3182BD",
    #     "4": "#08519C",
    # }
    colors = {
        "1": "#E41A1C",  # red
        "2": "#377EB8",  # blue
        "3": "#4DAF4A",  # green
        "4": "#984EA3",  # purple
    }

    linestyles = ["-", "-", "-", "-"]

    y_labels = {
        "absolute": "Absolute error",
        "relative": r"Relative error [%]",
        "normalized_absolute": r"Normalized absolute error [%]",
        "symmetric_relative": r"Symmetric relative error[%]",
        "range_normalized": r"Range-normalized error [%]",
    }

    errors = {}

    # First folder is used only when dns_mode="first"
    first_folder_id = next(iter(folders.values()))

    for lc in load_case:

        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        errors[lc] = {}

        # -------------------------------------------------
        # Optionally load one shared DNS curve
        # -------------------------------------------------
        shared_dns_strain = None
        shared_dns_stress = None

        if dns_mode == "first":

            first_validation_folder = (
                Path("IMN_validation")
                / f"Validation{int(first_folder_id):04d}"
                / "Val_stage_2_rve_0_mesh_0"
                / "plots"
            )

            shared_dns_stress = np.asarray(
                np.load(
                    first_validation_folder
                    / f"LC{lc}_stress_DNS.npz"
                )["data"],
                dtype=float,
            ).squeeze()

            shared_dns_strain = np.asarray(
                np.load(
                    first_validation_folder
                    / f"LC{lc}_strain_DNS.npz"
                )["data"],
                dtype=float,
            ).squeeze()

            shared_dns_strain, shared_dns_stress = _prepare_curve(
                shared_dns_strain,
                shared_dns_stress,
                curve_name=f"shared DNS, load case {lc}",
            )

        # -------------------------------------------------
        # Process every model folder
        # -------------------------------------------------
        for model_id, (name, folder_id) in enumerate(folders.items()):

            validation_folder = (
                Path("IMN_validation")
                / f"Validation{int(folder_id):04d}"
                / "Val_stage_2_rve_0_mesh_0"
                / "plots"
            )

            # ---------------------------------------------
            # Load model curve
            # ---------------------------------------------
            model_stress = np.asarray(
                np.load(
                    validation_folder
                    / f"LC{lc}_stress_IMN.npz"
                )["data"],
                dtype=float,
            ).squeeze()

            model_strain = np.asarray(
                np.load(
                    validation_folder
                    / f"LC{lc}_strain_IMN.npz"
                )["data"],
                dtype=float,
            ).squeeze()

            model_strain, model_stress = _prepare_curve(
                model_strain,
                model_stress,
                curve_name=f"{name}, load case {lc}",
            )

            # ---------------------------------------------
            # Select the DNS curve
            # ---------------------------------------------
            if dns_mode == "first":

                dns_strain = shared_dns_strain
                dns_stress = shared_dns_stress

            else:
                # DNS is loaded from the same folder
                # as the current model.
                dns_stress = np.asarray(
                    np.load(
                        validation_folder
                        / f"LC{lc}_stress_DNS.npz"
                    )["data"],
                    dtype=float,
                ).squeeze()

                dns_strain = np.asarray(
                    np.load(
                        validation_folder
                        / f"LC{lc}_strain_DNS.npz"
                    )["data"],
                    dtype=float,
                ).squeeze()

                dns_strain, dns_stress = _prepare_curve(
                    dns_strain,
                    dns_stress,
                    curve_name=(
                        f"DNS for {name}, load case {lc}"
                    ),
                )

            # ---------------------------------------------
            # Restrict comparison to common strain range
            # ---------------------------------------------
            common_range = (
                (model_strain >= dns_strain.min())
                & (model_strain <= dns_strain.max())
            )

            comparison_strain = model_strain[common_range]
            comparison_stress = model_stress[common_range]

            if comparison_strain.size == 0:
                print(
                    f"Warning: no common strain range for "
                    f"load case {lc}, model '{name}'."
                )
                continue

            # Interpolate the selected DNS curve onto
            # the model strain values.
            interpolated_dns_stress = np.interp(
                comparison_strain,
                dns_strain,
                dns_stress,
            )

            difference = (
                comparison_stress - interpolated_dns_stress
            )

            absolute_error = np.abs(difference)

            maximum_dns_stress = np.max(
                np.abs(dns_stress)
            )

            dns_stress_range = np.ptp(dns_stress)

            numerical_tolerance = max(
                maximum_dns_stress * 1.0e-10,
                np.finfo(float).eps,
            )

            # ---------------------------------------------
            # Calculate selected pointwise error
            # ---------------------------------------------
            if error_method == "absolute":

                pointwise_error = absolute_error

            elif error_method == "relative":

                pointwise_error = np.full_like(
                    absolute_error,
                    np.nan,
                    dtype=float,
                )

                valid_points = (
                    np.abs(interpolated_dns_stress)
                    > numerical_tolerance
                )

                pointwise_error[valid_points] = (
                    absolute_error[valid_points]
                    / np.abs(
                        interpolated_dns_stress[valid_points]
                    )
                    * 100.0
                )

            elif error_method == "normalized_absolute":

                if maximum_dns_stress > numerical_tolerance:
                    pointwise_error = (
                        absolute_error
                        / maximum_dns_stress
                        * 100.0
                    )
                else:
                    pointwise_error = np.full_like(
                        absolute_error,
                        np.nan,
                        dtype=float,
                    )

            elif error_method == "symmetric_relative":

                denominator = (
                    np.abs(comparison_stress)
                    + np.abs(interpolated_dns_stress)
                )

                pointwise_error = np.full_like(
                    absolute_error,
                    np.nan,
                    dtype=float,
                )

                valid_points = (
                    denominator > numerical_tolerance
                )

                pointwise_error[valid_points] = (
                    2.0
                    * absolute_error[valid_points]
                    / denominator[valid_points]
                    * 100.0
                )

            elif error_method == "range_normalized":

                if dns_stress_range > numerical_tolerance:
                    pointwise_error = (
                        absolute_error
                        / dns_stress_range
                        * 100.0
                    )
                else:
                    pointwise_error = np.full_like(
                        absolute_error,
                        np.nan,
                        dtype=float,
                    )

            # ---------------------------------------------
            # Overall curve metrics
            # ---------------------------------------------
            mean_error = np.nanmean(pointwise_error)
            maximum_error = np.nanmax(pointwise_error)

            mae = np.mean(absolute_error)
            rmse = np.sqrt(np.mean(difference**2))

            if maximum_dns_stress > numerical_tolerance:
                nmae = (
                    mae
                    / maximum_dns_stress
                    * 100.0
                )

                nrmse = (
                    rmse
                    / maximum_dns_stress
                    * 100.0
                )
            else:
                nmae = np.nan
                nrmse = np.nan

            errors[lc][name] = {
                "folder_id": folder_id,
                "dns_mode": dns_mode,
                "dns_folder_id": (
                    first_folder_id
                    if dns_mode == "first"
                    else folder_id
                ),
                "error_method": error_method,
                "strain": comparison_strain,
                "dns_stress": interpolated_dns_stress,
                "model_stress": comparison_stress,
                "difference": difference,
                "absolute_error": absolute_error,
                "pointwise_error": pointwise_error,
                "mean_error": mean_error,
                "maximum_error": maximum_error,
                "mae": mae,
                "rmse": rmse,
                "nmae": nmae,
                "nrmse": nrmse,
            }

            color = colors[
                str((model_id % len(colors)) + 1)
            ]

            linestyle = linestyles[
                model_id % len(linestyles)
            ]

            if error_method == "absolute":
                legend_label = (
                    f"{name} "
                    f" - Mean = {mean_error:.3e}"
                )
            else:
                legend_label = (
                    f"{name}  - "
                    f"Mean = ${mean_error:.2f}\%$"
                    # f"NRMSE = {nrmse:.2f}"
                )

            ax.plot(
                comparison_strain,
                pointwise_error,
                label=legend_label,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )

        # -------------------------------------------------
        # Plot formatting
        # -------------------------------------------------
        ax.set_xlabel("Strain")
        ax.set_ylabel(y_labels[error_method])
        ax.set_title(f"Tension" if lc==2 else f"Shear")

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

        ax.legend(
            frameon=True,
            edgecolor="black",
            fancybox=False,
            loc="best",
        )

        fig.tight_layout()

        fig.savefig(
            save_folder
            / (
                f"LC{lc}_{error_method}_error_"
                f"dns_{dns_mode}.svg"
            ),
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(fig)

    return errors


def _prepare_curve(strain, stress, curve_name="curve"):
    """
    Validate, sort, and remove repeated strain values.
    """

    strain = np.asarray(strain, dtype=float).squeeze()
    stress = np.asarray(stress, dtype=float).squeeze()

    if strain.ndim != 1 or stress.ndim != 1:
        raise ValueError(
            f"{curve_name}: strain and stress must be "
            f"one-dimensional arrays."
        )

    if len(strain) != len(stress):
        raise ValueError(
            f"{curve_name}: strain and stress lengths differ."
        )

    if len(strain) == 0:
        raise ValueError(
            f"{curve_name}: curve is empty."
        )

    finite_points = (
        np.isfinite(strain)
        & np.isfinite(stress)
    )

    strain = strain[finite_points]
    stress = stress[finite_points]

    if len(strain) == 0:
        raise ValueError(
            f"{curve_name}: curve contains no finite values."
        )

    sort_indices = np.argsort(strain)
    strain = strain[sort_indices]
    stress = stress[sort_indices]

    strain, unique_indices = np.unique(
        strain,
        return_index=True,
    )

    stress = stress[unique_indices]

    return strain, stress