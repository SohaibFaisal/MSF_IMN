from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compare_validations_combined(
    load_case,
    folders,
    show=True,
    save_folder=".",
    dpi=600,
    error_method="normalized_absolute",
    dns_error_mode="separate",
    dns_display_mode="first",
):
    """
    Plot stress-strain curves and corresponding pointwise errors
    in one figure using two y-axes.

    Left y-axis
        Stress

    Right y-axis
        Error

    Model stress:
        Solid lines.

    Error:
        Dashed lines with the same color as the corresponding model.

    DNS:
        Open triangle markers.

    Parameters
    ----------
    load_case : iterable
        Load-case numbers.

    folders : dict
        Dictionary containing curve names and validation folder IDs.

        Example:
            folders = {
                "Case 1": 101,
                "Case 2": 102,
                "Case 3": 103,
                "Case 4": 104,
            }

    show : bool, optional
        Show the plots.

    save_folder : str or Path, optional
        Folder where figures are saved.

    dpi : int, optional
        Figure resolution.

    error_method : str, optional
        Method used to calculate the pointwise error.

        Options:
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

    dns_error_mode : str, optional
        Controls which DNS curve is used for ERROR CALCULATION.

        "first"
            DNS from the first folder is used for calculating
            the errors of all model curves.

        "separate"
            Each model is compared against the DNS contained
            in its corresponding folder.

    dns_display_mode : str, optional
        Controls which DNS curves are DISPLAYED.

        "first"
            Only the DNS from the first folder is displayed.

        "separate"
            DNS from every folder is displayed.

    Returns
    -------
    errors : dict
        Nested dictionary containing all calculated error quantities.
    """

    # ============================================================
    # Input checks
    # ============================================================

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

    if dns_error_mode not in valid_dns_modes:
        raise ValueError(
            f"Unknown dns_error_mode '{dns_error_mode}'. "
            f"Choose from {sorted(valid_dns_modes)}."
        )

    if dns_display_mode not in valid_dns_modes:
        raise ValueError(
            f"Unknown dns_display_mode '{dns_display_mode}'. "
            f"Choose from {sorted(valid_dns_modes)}."
        )

    if not folders:
        raise ValueError("'folders' cannot be empty.")

    # ============================================================
    # Matplotlib formatting
    # ============================================================

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

    colors = {
        "1": "#E41A1C",  # red
        "2": "#377EB8",  # blue
        "3": "#4DAF4A",  # green
        "4": "#984EA3",  # purple
    }

    error_ylabels = {
        "absolute": "Absolute error",
        "relative": r"Relative error [%]",
        "normalized_absolute": r"Normalized absolute error [%]",
        "symmetric_relative": r"Symmetric relative error [%]",
        "range_normalized": r"Range-normalized error [%]",
    }

    errors = {}

    # First folder ID
    first_folder_id = next(iter(folders.values()))

    # ============================================================
    # Loop through load cases
    # ============================================================

    for lc in load_case:

        fig, ax_stress = plt.subplots(
            figsize=(5.8*1.5, 4.0*1.5)
        )

        # Right y-axis
        ax_error = ax_stress.twinx()

        errors[lc] = {}

        # ========================================================
        # Load first/shared DNS if required
        # ========================================================

        shared_dns_strain = None
        shared_dns_stress = None

        if (
            dns_error_mode == "first"
            or dns_display_mode == "first"
        ):

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

        # ========================================================
        # Loop through models
        # ========================================================

        for model_id, (name, folder_id) in enumerate(
            folders.items()
        ):

            color = colors[
                str((model_id % len(colors)) + 1)
            ]

            validation_folder = (
                Path("IMN_validation")
                / f"Validation{int(folder_id):04d}"
                / "Val_stage_2_rve_0_mesh_0"
                / "plots"
            )

            # ====================================================
            # Load model stress/strain
            # ====================================================

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

            # ====================================================
            # Load folder-specific DNS if required
            # ====================================================

            folder_dns_strain = None
            folder_dns_stress = None

            if (
                dns_error_mode == "separate"
                or dns_display_mode == "separate"
            ):

                folder_dns_stress = np.asarray(
                    np.load(
                        validation_folder
                        / f"LC{lc}_stress_DNS.npz"
                    )["data"],
                    dtype=float,
                ).squeeze()

                folder_dns_strain = np.asarray(
                    np.load(
                        validation_folder
                        / f"LC{lc}_strain_DNS.npz"
                    )["data"],
                    dtype=float,
                ).squeeze()

                (
                    folder_dns_strain,
                    folder_dns_stress,
                ) = _prepare_curve(
                    folder_dns_strain,
                    folder_dns_stress,
                    curve_name=(
                        f"DNS for {name}, load case {lc}"
                    ),
                )

            # ====================================================
            # Select DNS ONLY for error calculation
            # ====================================================

            if dns_error_mode == "first":

                error_dns_strain = shared_dns_strain
                error_dns_stress = shared_dns_stress

            else:

                error_dns_strain = folder_dns_strain
                error_dns_stress = folder_dns_stress

            # ====================================================
            # Plot model stress
            # ====================================================

            ax_stress.plot(
                model_strain,
                model_stress,
                color=color,
                linestyle="-",
                linewidth=1.5,
                label=name,
            )

            # ====================================================
            # Plot DNS
            # ====================================================

            # ----------------------------------------------------
            # Display only first DNS
            # ----------------------------------------------------

            if (
                dns_display_mode == "first"
                and model_id == 0
            ):

                ax_stress.plot(
                    shared_dns_strain,
                    shared_dns_stress,
                    color="black",
                    linestyle="none",
                    marker="^",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markevery=max(
                        len(shared_dns_strain) // 25,
                        1,
                    ),
                    markersize=4,
                    label="DNS",
                    zorder=10,
                )

            # ----------------------------------------------------
            # Display separate DNS curves
            # ----------------------------------------------------

            elif dns_display_mode == "separate":

                ax_stress.plot(
                    folder_dns_strain,
                    folder_dns_stress,
                    color=color,
                    linestyle="none",
                    marker="^",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markevery=max(
                        len(folder_dns_strain) // 25,
                        1,
                    ),
                    markersize=4,
                    label=f"{name} - DNS",
                    zorder=10,
                )

            # ====================================================
            # Determine common strain range FOR ERROR CALCULATION
            # ====================================================

            common_range = (
                (model_strain >= error_dns_strain.min())
                & (model_strain <= error_dns_strain.max())
            )

            comparison_strain = model_strain[
                common_range
            ]

            comparison_stress = model_stress[
                common_range
            ]

            if comparison_strain.size == 0:

                print(
                    f"Warning: no common strain range for "
                    f"load case {lc}, model '{name}'."
                )

                continue

            # ====================================================
            # Interpolate selected DNS onto model strain points
            # ====================================================

            interpolated_dns_stress = np.interp(
                comparison_strain,
                error_dns_strain,
                error_dns_stress,
            )

            # ====================================================
            # Error quantities
            # ====================================================

            difference = (
                comparison_stress
                - interpolated_dns_stress
            )

            absolute_error = np.abs(
                difference
            )

            maximum_dns_stress = np.max(
                np.abs(error_dns_stress)
            )

            dns_stress_range = np.ptp(
                error_dns_stress
            )

            numerical_tolerance = max(
                maximum_dns_stress * 1.0e-10,
                np.finfo(float).eps,
            )

            # ====================================================
            # Pointwise error
            # ====================================================

            if error_method == "absolute":

                pointwise_error = absolute_error

            # ----------------------------------------------------

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
                        interpolated_dns_stress[
                            valid_points
                        ]
                    )
                    * 100.0
                )

            # ----------------------------------------------------

            elif error_method == "normalized_absolute":

                if (
                    maximum_dns_stress
                    > numerical_tolerance
                ):

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

            # ----------------------------------------------------

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
                    denominator
                    > numerical_tolerance
                )

                pointwise_error[valid_points] = (
                    2.0
                    * absolute_error[valid_points]
                    / denominator[valid_points]
                    * 100.0
                )

            # ----------------------------------------------------

            elif error_method == "range_normalized":

                if (
                    dns_stress_range
                    > numerical_tolerance
                ):

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

            # ====================================================
            # Overall error metrics
            # ====================================================

            mean_error = np.nanmean(
                pointwise_error
            )

            maximum_error = np.nanmax(
                pointwise_error
            )

            mae = np.mean(
                absolute_error
            )

            rmse = np.sqrt(
                np.mean(
                    difference ** 2
                )
            )

            if (
                maximum_dns_stress
                > numerical_tolerance
            ):

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

            # ====================================================
            # Store results
            # ====================================================

            errors[lc][name] = {

                "folder_id": folder_id,

                "dns_error_mode":
                    dns_error_mode,

                "dns_display_mode":
                    dns_display_mode,

                "dns_error_folder_id": (
                    first_folder_id
                    if dns_error_mode == "first"
                    else folder_id
                ),

                "error_method":
                    error_method,

                "strain":
                    comparison_strain,

                "dns_stress":
                    interpolated_dns_stress,

                "model_stress":
                    comparison_stress,

                "difference":
                    difference,

                "absolute_error":
                    absolute_error,

                "pointwise_error":
                    pointwise_error,

                "mean_error":
                    mean_error,

                "maximum_error":
                    maximum_error,

                "mae":
                    mae,

                "rmse":
                    rmse,

                "nmae":
                    nmae,

                "nrmse":
                    nrmse,
            }

            # ====================================================
            # Plot error on right y-axis
            # ====================================================

            if error_method == "absolute":

                error_label = (
                    f"{name} - Error "
                    f"(Mean = {mean_error:.3e})"
                )

            else:

                error_label = (
                    f"{name} - Error "
                    f"(Mean = {mean_error:.2f}%)"
                )

            ax_error.plot(
                comparison_strain,
                pointwise_error,
                color=color,
                linestyle="--",
                linewidth=1.3,
                label=error_label,
            )

        # ========================================================
        # Axis formatting
        # ========================================================

        ax_stress.set_xlabel(
            "Strain"
        )

        ax_stress.set_ylabel(
            "Stress"
        )

        ax_error.set_ylabel(
            error_ylabels[error_method]
        )

        # --------------------------------------------------------
        # Force zero of both y-axes to the same vertical position
        # --------------------------------------------------------

        ax_stress.set_ylim(
            bottom=0
        )

        ax_error.set_ylim(
            bottom=0
        )

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        ax_stress.grid(
            True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

        ax_stress.set_axisbelow(
            True
        )

        # ========================================================
        # Combined legend
        # ========================================================

        stress_lines, stress_labels = (
            ax_stress.get_legend_handles_labels()
        )

        error_lines, error_labels = (
            ax_error.get_legend_handles_labels()
        )

        ax_stress.legend(
            stress_lines + error_lines,
            stress_labels + error_labels,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            loc="best",
        )

        # ========================================================
        # Layout / save
        # ========================================================

        fig.tight_layout()

        fig.savefig(
            save_folder
            / (
                f"LC{lc}_stress_error_"
                f"{error_method}_"
                f"calc_{dns_error_mode}_"
                f"display_{dns_display_mode}.svg"
            ),
            bbox_inches="tight",
        )

        if show:
            plt.show()

        plt.close(fig)

    return errors


def _prepare_curve(
    strain,
    stress,
    curve_name="curve",
):
    """
    Validate, sort, and remove repeated strain values.
    """

    strain = np.asarray(
        strain,
        dtype=float,
    ).squeeze()

    stress = np.asarray(
        stress,
        dtype=float,
    ).squeeze()

    # ------------------------------------------------------------
    # Check dimensions
    # ------------------------------------------------------------

    if (
        strain.ndim != 1
        or stress.ndim != 1
    ):

        raise ValueError(
            f"{curve_name}: strain and stress must be "
            f"one-dimensional arrays."
        )

    # ------------------------------------------------------------

    if len(strain) != len(stress):

        raise ValueError(
            f"{curve_name}: strain and stress lengths differ."
        )

    # ------------------------------------------------------------

    if len(strain) == 0:

        raise ValueError(
            f"{curve_name}: curve is empty."
        )

    # ------------------------------------------------------------
    # Remove non-finite values
    # ------------------------------------------------------------

    finite_points = (
        np.isfinite(strain)
        & np.isfinite(stress)
    )

    strain = strain[
        finite_points
    ]

    stress = stress[
        finite_points
    ]

    if len(strain) == 0:

        raise ValueError(
            f"{curve_name}: curve contains no finite values."
        )

    # ------------------------------------------------------------
    # Sort by strain
    # ------------------------------------------------------------

    sort_indices = np.argsort(
        strain
    )

    strain = strain[
        sort_indices
    ]

    stress = stress[
        sort_indices
    ]

    # ------------------------------------------------------------
    # Remove repeated strain values
    # ------------------------------------------------------------

    strain, unique_indices = np.unique(
        strain,
        return_index=True,
    )

    stress = stress[
        unique_indices
    ]

    return strain, stress