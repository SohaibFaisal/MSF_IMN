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


def plot_just_mean(error_dict, name):
    keys = list(error_dict.keys())
    x = np.arange(len(keys))

    means = [np.mean(error_dict[k]) for k in keys]

    plt.figure(figsize=(24, 16))

    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
    ][:1]
    # Slim, elegant bars
    bars = plt.bar(x, means, width=0.36, color="#4C72B0", edgecolor="black",zorder=3,)
    plt.bar_label(
        bars,
        labels=[f"{value:.0f}" for value in means],
        padding=3,
        fontsize=8,
        rotation=0,
    )


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

    plt.savefig(f"mean_error_{name}.svg", format='svg')
    plt.close()
    # plt.show()


import numpy as np
import matplotlib.pyplot as plt


# def plot_just_mean_multi(
#     error_dicts,
#     labels=None,
#     colors=None,
#     hatches=None,
#     save_path="mean_error.svg",
# ):
#     """
#     Plot grouped mean prediction errors for multiple models.
#
#     Parameters
#     ----------
#     error_dicts : list of dict
#         Each dictionary contains arrays or lists of errors for each
#         elastic constant.
#
#     labels : list of str
#         Labels used in the legend.
#
#     colors : list of str
#         Manually selected colors for each model.
#
#     hatches : list of str, optional
#         Hatch patterns, useful for black-and-white printing.
#
#     save_path : str
#         Output file path.
#     """
#
#     n_cases = len(error_dicts)
#
#     if labels is None:
#         labels = [f"Model {i + 1}" for i in range(n_cases)]
#
#     if colors is None:
#         colors = [
#             "#4C72B0",
#             "#DD8452",
#             "#55A868",
#             "#C44E52",
#         ][:n_cases]
#
#     if hatches is None:
#         hatches = ["", "---", "...", 'xxx'][:n_cases]
#
#     if len(labels) != n_cases:
#         raise ValueError("Number of labels must equal number of datasets.")
#
#     if len(colors) != n_cases:
#         raise ValueError("Number of colors must equal number of datasets.")
#
#     keys = list(error_dicts[0].keys())
#     x = np.arange(len(keys))
#
#     means_per_case = [
#         np.array([np.mean(error_dict[key]) for key in keys])
#         for error_dict in error_dicts
#     ]
#
#     # Slightly narrower bars create more white space between groups
#     total_group_width = 0.72
#     bar_width = total_group_width / n_cases
#
#     fig, ax = plt.subplots(figsize=(12, 7.2))
#
#     for i, means in enumerate(means_per_case):
#         offset = (i - (n_cases - 1) / 2) * bar_width
#         print(i)
#         print(hatches)
#         bars = ax.bar(
#             x + offset,
#             means,
#             width=bar_width,
#             color=colors[0],
#             edgecolor="black",
#             linewidth=0.55,
#             hatch=hatches[i],
#             label=labels[i],
#             zorder=3,
#         )
#
#         # Add compact numerical values above bars
#         ax.bar_label(
#             bars,
#             labels=[f"{value:.0f}" for value in means],
#             padding=3,
#             fontsize=8,
#             rotation=0,
#         )
#
#     ax.set_xticks(x)
#     ax.set_xticklabels(keys, fontsize=10)
#
#     ax.set_ylabel("Mean absolute percentage error (%)", fontsize=11)
#     ax.set_xlabel("Effective elastic constant", fontsize=11)
#
#     # Usually unnecessary in a paper if the caption already explains it
#     # ax.set_title("Elastic constants prediction error", fontsize=12)
#
#     ax.tick_params(
#         axis="both",
#         which="major",
#         labelsize=10,
#         direction="out",
#         length=4,
#         width=0.8,
#     )
#
#     ax.grid(
#         axis="y",
#         linestyle="--",
#         linewidth=0.7,
#         alpha=0.35,
#         zorder=0,
#     )
#
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_linewidth(0.9)
#     ax.spines["bottom"].set_linewidth(0.9)
#
#     ax.set_ylim(
#         0,
#         max(np.max(values) for values in means_per_case) * 1.15,
#     )
#
#     # Horizontal legend above the plot
#     ax.legend(
#         loc="lower center",
#         bbox_to_anchor=(0.5, 1.01),
#         ncol=n_cases,
#         frameon=False,
#         fontsize=10,
#         handlelength=1.8,
#         columnspacing=1.5,
#     )
#
#     fig.tight_layout()
#
#     fig.savefig(
#         save_path,
#         format="svg",
#         bbox_inches="tight",
#     )
#
#     # Optional high-resolution raster output
#     fig.savefig(
#         save_path.replace(".svg", ".png"),
#         dpi=600,
#         bbox_inches="tight",
#     )
#
#     plt.close(fig)
import numpy as np
import matplotlib.pyplot as plt


def plot_just_mean_multi(
    error_dicts,
    labels=None,
    colors=None,
    hatches=None,
    save_path="mean_error.svg",
    show_std=True,
    show_median=True,
    show_outlier_count=False,
    show_sample_size=False,
    outlier_factor=1.5,
):
    """
    Plot grouped mean prediction errors for multiple models.

    In addition to the mean error, the plot can display the standard
    deviation, median, number of statistical outliers, and sample size
    for each elastic constant.

    Outliers are identified using the interquartile-range criterion:

        value < Q1 - outlier_factor * IQR

    or

        value > Q3 + outlier_factor * IQR

    where IQR = Q3 - Q1.

    Parameters
    ----------
    error_dicts : list of dict
        Each dictionary contains arrays or lists of errors for each
        elastic constant. For example:

        {
            "E11": [1.2, 2.5, 3.1, ...],
            "E22": [2.0, 2.3, 4.1, ...],
        }

    labels : list of str, optional
        Labels used in the legend.

    colors : list of str, optional
        Manually selected colors for each model.

    hatches : list of str, optional
        Hatch patterns, useful for black-and-white printing.

    save_path : str, optional
        Output file path for the SVG figure.

    show_std : bool, optional
        If True, display standard-deviation error bars.

    show_median : bool, optional
        If True, display the median as a horizontal marker inside or
        above each bar.

    show_outlier_count : bool, optional
        If True, annotate the number of outliers for distributions
        containing at least one outlier.

    show_sample_size : bool, optional
        If True, annotate the number of samples as n = sample size.

    outlier_factor : float, optional
        Multiplier used in the interquartile-range outlier criterion.
        The conventional value is 1.5.
    """

    n_cases = len(error_dicts)

    if n_cases == 0:
        raise ValueError("At least one error dictionary must be provided.")

    if labels is None:
        labels = [f"Model {i + 1}" for i in range(n_cases)]

    if colors is None:
        default_colors = [
            # "#4C72B0",
            "#DD8452",
            "#55A868",
            "#C44E52",
        ]

        if n_cases > len(default_colors):
            raise ValueError(
                "More datasets were provided than default colors. "
                "Please provide a color for each dataset."
            )

        colors = default_colors[:n_cases]

    if hatches is None:
        default_hatches = ["", "", "", ""]

        if n_cases > len(default_hatches):
            raise ValueError(
                "More datasets were provided than default hatch patterns. "
                "Please provide a hatch pattern for each dataset."
            )

        hatches = default_hatches[:n_cases]

    if len(labels) != n_cases:
        raise ValueError(
            "Number of labels must equal number of datasets."
        )

    if len(colors) != n_cases:
        raise ValueError(
            "Number of colors must equal number of datasets."
        )

    if len(hatches) != n_cases:
        raise ValueError(
            "Number of hatch patterns must equal number of datasets."
        )

    if outlier_factor <= 0:
        raise ValueError("outlier_factor must be greater than zero.")

    # Use the keys of the first dictionary as the plotting order.
    keys = list(error_dicts[0].keys())

    if len(keys) == 0:
        raise ValueError("The error dictionaries must not be empty.")

    # Confirm that every dictionary contains the same elastic constants.
    for i, error_dict in enumerate(error_dicts):
        missing_keys = [key for key in keys if key not in error_dict]
        extra_keys = [key for key in error_dict if key not in keys]

        if missing_keys or extra_keys:
            raise ValueError(
                f"Dataset {i} does not contain the same keys as the "
                f"first dataset. Missing keys: {missing_keys}; "
                f"extra keys: {extra_keys}."
            )

    x = np.arange(len(keys))

    # Store all statistics in arrays with dimensions:
    # number of models x number of elastic constants.
    means_per_case = []
    stds_per_case = []
    medians_per_case = []
    outlier_counts_per_case = []
    sample_sizes_per_case = []

    for case_index, error_dict in enumerate(error_dicts):
        case_means = []
        case_stds = []
        case_medians = []
        case_outlier_counts = []
        case_sample_sizes = []

        for key in keys:
            values = np.asarray(
                error_dict[key],
                dtype=float,
            ).reshape(-1)

            # Remove NaN and infinite values before calculating statistics.
            values = values[np.isfinite(values)]

            if values.size == 0:
                raise ValueError(
                    f"No finite error values were found for key '{key}' "
                    f"in dataset {case_index}."
                )

            mean_value = np.mean(values)
            median_value = np.median(values)

            # ddof=1 gives the sample standard deviation. For a single
            # sample, the standard deviation is set to zero.
            if values.size > 1:
                std_value = np.std(values, ddof=1)
            else:
                std_value = 0.0

            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_limit = q1 - outlier_factor * iqr
            upper_limit = q3 + outlier_factor * iqr

            outlier_mask = (
                (values < lower_limit)
                | (values > upper_limit)
            )

            outlier_count = np.count_nonzero(outlier_mask)

            case_means.append(mean_value)
            case_stds.append(std_value)
            case_medians.append(median_value)
            case_outlier_counts.append(outlier_count)
            case_sample_sizes.append(values.size)

        means_per_case.append(np.asarray(case_means))
        stds_per_case.append(np.asarray(case_stds))
        medians_per_case.append(np.asarray(case_medians))
        outlier_counts_per_case.append(
            np.asarray(case_outlier_counts)
        )
        sample_sizes_per_case.append(
            np.asarray(case_sample_sizes)
        )

    # Slightly narrower bars create more white space between groups.
    total_group_width = 0.72
    bar_width = total_group_width / n_cases

    fig, ax = plt.subplots(figsize=(12, 7.2))

    # Used later to determine a suitable vertical-axis limit.
    maximum_plot_height = 0.0

    for i, means in enumerate(means_per_case):
        stds = stds_per_case[i]
        medians = medians_per_case[i]
        outlier_counts = outlier_counts_per_case[i]
        sample_sizes = sample_sizes_per_case[i]

        offset = (i - (n_cases - 1) / 2) * bar_width
        bar_positions = x + offset

        if show_std:
            yerr = stds
            error_kw = {
                "elinewidth": 0.9,
                "capsize": 3,
                "capthick": 0.9,
            }
        else:
            yerr = None
            error_kw = None

        bars = ax.bar(
            bar_positions,
            means,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.55,
            hatch=hatches[i],
            label=labels[i],
            yerr=yerr,
            error_kw=error_kw,
            zorder=3,
        )

        # Add compact numerical mean values above the bars.
        for j, bar in enumerate(bars):
            mean_value = means[j]
            std_value = stds[j]

            if show_std:
                label_height = mean_value + std_value
            else:
                label_height = mean_value

            ax.annotate(
                f"{mean_value:.1f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    label_height,
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=6,
            )

        # Display the median using a short horizontal line.
        #
        # The median marker is useful because a substantial difference
        # between the mean and median can indicate a skewed distribution
        # or the influence of extreme values.
        if show_median:
            median_half_width = bar_width * 0.28

            for position, median_value in zip(
                bar_positions,
                medians,
            ):
                ax.hlines(
                    y=median_value,
                    xmin=position - median_half_width,
                    xmax=position + median_half_width,
                    color="black",
                    linewidth=1.4,
                    zorder=5,
                )

        # Add outlier and sample-size information.
        for j, position in enumerate(bar_positions):
            annotation_parts = []

            if (
                show_outlier_count
                and outlier_counts[j] > 0
            ):
                annotation_parts.append(
                    f"outliers: {outlier_counts[j]}"
                )

            if show_sample_size:
                annotation_parts.append(
                    f"n = {sample_sizes[j]}"
                )

            if annotation_parts:
                if show_std:
                    annotation_y = means[j] + stds[j]
                else:
                    annotation_y = means[j]

                ax.annotate(
                    "\n".join(annotation_parts),
                    xy=(position, annotation_y),
                    xytext=(0, 18),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    zorder=6,
                )

        if show_std:
            current_maximum = np.max(means + stds)
        else:
            current_maximum = np.max(means)

        maximum_plot_height = max(
            maximum_plot_height,
            current_maximum,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)

    ax.set_ylabel(
        "Mean absolute percentage error (%)",
        fontsize=11,
    )
    ax.set_xlabel(
        "Effective elastic constant",
        fontsize=11,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    # Add additional vertical space when statistical annotations are used.
    ylim_factor = 1.18

    if show_outlier_count or show_sample_size:
        ylim_factor = 1.32
    elif show_std:
        ylim_factor = 1.22

    if maximum_plot_height == 0:
        maximum_plot_height = 1.0

    ax.set_ylim(
        0,
        maximum_plot_height * ylim_factor,
    )

    # Horizontal legend above the plot.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=n_cases,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.5,
    )

    # Explanation of the additional statistical information.
    statistical_notes = []

    if show_std:
        statistical_notes.append(
            "error bars: standard deviation"
        )

    if show_median:
        statistical_notes.append(
            "horizontal markers: median"
        )

    if show_outlier_count:
        statistical_notes.append(
            f"outliers: {outlier_factor:g} × IQR criterion"
        )

    if statistical_notes:
        ax.text(
            0.995,
            0.985,
            "; ".join(statistical_notes),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.30",
        )

    fig.tight_layout()

    fig.savefig(
        save_path,
        format="svg",
        bbox_inches="tight",
    )

    # Optional high-resolution raster output.
    if save_path.lower().endswith(".svg"):
        png_path = save_path[:-4] + ".png"
    else:
        png_path = save_path + ".png"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)



import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def plot_error_boxplots_multi(
    error_dicts,
    labels=None,
    colors=None,
    save_path="error_boxplots.svg",
    show_mean=True,
    show_outliers=False,
    use_log_scale=False,
    max_error=200.0,
):
    """
    Plot grouped boxplots of prediction errors for multiple models.

    For each elastic constant, corresponding samples are removed from all
    datasets when at least one dataset contains:

    - a non-finite value,
    - an error greater than max_error,
    - or, for logarithmic plots, a value less than or equal to zero.

    Parameters
    ----------
    error_dicts : list of dict
        Each dictionary contains arrays or lists of errors for each
        effective elastic constant. For each key, all datasets must contain
        the same number of corresponding entries.

    labels : list of str, optional
        Labels used in the legend.

    colors : list of str, optional
        Colors assigned to the different datasets.

    save_path : str, optional
        Output path for the SVG figure.

    show_mean : bool, optional
        Show the arithmetic mean as a diamond marker.

    show_outliers : bool, optional
        Show observations outside the 1.5 IQR whiskers.

    use_log_scale : bool, optional
        Use a logarithmic y-axis.

    max_error : float or None, optional
        Maximum permitted error percentage. If any dataset has an error
        greater than this value at a particular sample index, that sample
        index is removed from every dataset.

        Set to None to disable maximum-error filtering.
    """

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------
    n_cases = len(error_dicts)

    if n_cases == 0:
        raise ValueError(
            "At least one error dictionary is required."
        )

    if labels is None:
        labels = [
            f"Model {i + 1}"
            for i in range(n_cases)
        ]

    if colors is None:
        # default_colors = [
        #     "#D98C52",
        #     "#55A868",
        #     "#C44E52",
        #     "#4C72B0",
        #     "#8172B3",
        #     "#64B5CD",
        # ]

        # IMN vs DMN
        default_colors = [
            "#4C72B0",
            "#DD8452",
            "#55A868",
            "#C44E52",
        ]

        # IMN vs IMN
        # default_colors = [
        #     "#7FB3E8",  # light blue
        #     "#2F6DB3",  # medium blue
        #     "#0A2F6B",  # very dark navy
        # ]

        if n_cases > len(default_colors):
            raise ValueError(
                f"{n_cases} datasets were provided, but only "
                f"{len(default_colors)} default colors are available. "
                "Please provide a colors list."
            )

        colors = default_colors[:n_cases]

    if len(labels) != n_cases:
        raise ValueError(
            "Number of labels must equal number of datasets."
        )

    if len(colors) != n_cases:
        raise ValueError(
            "Number of colors must equal number of datasets."
        )

    if max_error is not None and max_error <= 0:
        raise ValueError(
            "max_error must be greater than zero or None."
        )

    keys = list(error_dicts[0].keys())

    if len(keys) == 0:
        raise ValueError(
            "The error dictionaries do not contain any data."
        )

    for i, error_dict in enumerate(error_dicts):
        if set(error_dict.keys()) != set(keys):
            raise ValueError(
                f"Dataset {i} does not contain the same keys "
                "as the first dataset."
            )

    # ---------------------------------------------------------
    # Paired filtering
    # ---------------------------------------------------------
    filtered_error_dicts = [
        {} for _ in range(n_cases)
    ]

    skipped_per_key = {}
    original_count_per_key = {}
    remaining_count_per_key = {}

    total_skipped_positions = 0
    total_removed_values = 0

    for key in keys:

        arrays = [
            np.asarray(
                error_dict[key],
                dtype=float,
            ).reshape(-1)
            for error_dict in error_dicts
        ]

        lengths = [
            values.size
            for values in arrays
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"All datasets must contain the same number of "
                f"corresponding values for '{key}'. "
                f"Received lengths: {lengths}"
            )

        if lengths[0] == 0:
            raise ValueError(
                f"No values were provided for '{key}'."
            )

        # Rows represent datasets.
        # Columns represent corresponding sample positions.
        stacked_values = np.vstack(arrays)

        # Only retain positions where all datasets contain finite values.
        valid_mask = np.all(
            np.isfinite(stacked_values),
            axis=0,
        )

        # Remove the position from every dataset when at least one
        # corresponding error is greater than max_error.
        if max_error is not None:
            valid_mask &= np.all(
                stacked_values <= max_error,
                axis=0,
            )

        # A logarithmic axis cannot display zero or negative values.
        if use_log_scale:
            valid_mask &= np.all(
                stacked_values > 0,
                axis=0,
            )

        n_original = stacked_values.shape[1]
        n_remaining = np.count_nonzero(valid_mask)
        n_skipped = n_original - n_remaining

        original_count_per_key[key] = n_original
        remaining_count_per_key[key] = n_remaining
        skipped_per_key[key] = n_skipped

        total_skipped_positions += n_skipped
        total_removed_values += n_skipped * n_cases

        if n_remaining == 0:
            raise ValueError(
                f"No valid paired values remain for '{key}' "
                "after filtering."
            )

        for dataset_index in range(n_cases):
            filtered_error_dicts[dataset_index][key] = (
                stacked_values[dataset_index, valid_mask]
            )

    error_dicts = filtered_error_dicts

    # ---------------------------------------------------------
    # Display filtering summary
    # ---------------------------------------------------------
    print("\nPaired error filtering summary")
    print("-" * 64)

    if max_error is not None:
        print(
            f"Maximum permitted error: {max_error:g}%"
        )
    else:
        print(
            "Maximum-error filtering: disabled"
        )

    for key in keys:
        print(
            f"{key:>10}: "
            f"{skipped_per_key[key]:>5} skipped, "
            f"{remaining_count_per_key[key]:>5} retained "
            f"out of {original_count_per_key[key]}"
        )

    print("-" * 64)

    print(
        "Total corresponding sample positions skipped: "
        f"{total_skipped_positions}"
    )

    print(
        "Total individual values removed: "
        f"{total_removed_values}"
    )

    print()

    # ---------------------------------------------------------
    # Plot configuration
    # ---------------------------------------------------------
    x = np.arange(len(keys))

    # Width occupied by all boxplots in one elastic-constant group.
    total_group_width = 0.70
    box_spacing = total_group_width / n_cases
    box_width = box_spacing * 0.72

    fig, ax = plt.subplots(
        figsize=(12, 7.2)
    )

    legend_handles = []

    # ---------------------------------------------------------
    # Plot each dataset
    # ---------------------------------------------------------
    for i, error_dict in enumerate(error_dicts):

        offset = (
                         i - (n_cases - 1) / 2
                 ) * box_spacing

        positions = x + offset

        data = []

        for key in keys:
            values = np.asarray(
                error_dict[key],
                dtype=float,
            ).reshape(-1)

            data.append(values)

        boxplot = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showmeans=show_mean,
            showfliers=show_outliers,
            whis=1.5,
            medianprops={
                "color": "black",
                "linewidth": 1.4,
            },
            whiskerprops={
                "color": "black",
                "linewidth": 0.8,
            },
            capprops={
                "color": "black",
                "linewidth": 0.8,
            },
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            flierprops={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": colors[i],
                "markersize": 3.5,
                "alpha": 0.55,
            },
        )

        for box in boxplot["boxes"]:
            box.set_facecolor(colors[i])
            box.set_edgecolor("black")
            box.set_linewidth(0.7)
            box.set_alpha(0.85)

        # Calculate mean values.
        means = np.array([
            np.mean(values)
            for values in data
        ])

        # Calculate the actual upper whisker position for each box.
        upper_whiskers = []

        for values in data:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)

            iqr = q3 - q1
            upper_limit = q3 + 1.5 * iqr

            values_within_whisker = values[
                values <= upper_limit
                ]

            upper_whiskers.append(
                np.max(values_within_whisker)
            )

        upper_whiskers = np.asarray(
            upper_whiskers
        )

        # Write each mean above its corresponding box.
        # if show_mean:
        #     for position, mean, upper_whisker in zip(
        #             positions,
        #             means,
        #             upper_whiskers,
        #     ):
        #         ax.annotate(
        #             f"{mean:.1f}%",
        #             xy=(
        #                 position,
        #                 upper_whisker,
        #             ),
        #             xytext=(
        #                 0,
        #                 5,
        #             ),
        #             textcoords="offset points",
        #             ha="center",
        #             va="bottom",
        #             fontsize=8,
        #             color="black",
        #             clip_on=False,
        #             zorder=7,
        #         )

        legend_handles.append(
            boxplot["boxes"][0]
        )
    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        keys,
        fontsize=10,
    )

    ax.set_xlabel(
        "Effective elastic constant",
        fontsize=11,
    )

    ax.set_ylabel(
        "Absolute percentage error (%)",
        fontsize=11,
    )

    if use_log_scale:
        ax.set_yscale("log")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.legend(
        legend_handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=n_cases,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.5,
    )

    # ---------------------------------------------------------
    # Statistical-marker explanation
    # ---------------------------------------------------------
    note = (
        "box: interquartile range; "
        "line: median; "
    )

    if show_mean:
        note += "diamond: mean; "

    note += "whiskers: 1.5 × IQR"

    if show_outliers:
        note += "; circles: outliers"

    if max_error is not None:
        note += (
            f"; paired samples above {max_error:g}% removed"
        )

    # ax.text(
    #     0.995,
    #     0.985,
    #     note,
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=7.5,
    #     color="0.30",
    # )

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    fig.savefig(
        save_path,
        format="svg",
        bbox_inches="tight",
    )

    png_path = (
        save_path[:-4] + ".png"
        if save_path.lower().endswith(".svg")
        else save_path + ".png"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

# def plot_error_boxplots_multi(
#     error_dicts,
#     labels=None,
#     colors=None,
#     save_path="error_boxplots.svg",
#     show_mean=True,
#     show_outliers=False,
#     use_log_scale=False,
# ):
#     """
#     Plot grouped boxplots of prediction errors for multiple models.
#
#     Parameters
#     ----------
#     error_dicts : list of dict
#         Each dictionary contains arrays or lists of errors for each
#         effective elastic constant.
#
#     labels : list of str, optional
#         Labels used in the legend.
#
#     colors : list of str, optional
#         Colors assigned to the different datasets.
#
#     save_path : str, optional
#         Output path for the SVG figure.
#
#     show_mean : bool, optional
#         Show the arithmetic mean as a diamond marker.
#
#     show_outliers : bool, optional
#         Show observations outside the 1.5 IQR whiskers.
#
#     use_log_scale : bool, optional
#         Use a logarithmic y-axis. This is useful when the errors span
#         several orders of magnitude.
#     """
#
#     n_cases = len(error_dicts)
#
#     if n_cases == 0:
#         raise ValueError("At least one error dictionary is required.")
#
#     if labels is None:
#         labels = [f"Model {i + 1}" for i in range(n_cases)]
#
#     if colors is None:
#         colors = [
#             "#D98C52",
#             "#55A868",
#             "#C44E52",
#             "#4C72B0",
#         ][:n_cases]
#
#     if len(labels) != n_cases:
#         raise ValueError(
#             "Number of labels must equal number of datasets."
#         )
#
#     if len(colors) != n_cases:
#         raise ValueError(
#             "Number of colors must equal number of datasets."
#         )
#
#     keys = list(error_dicts[0].keys())
#
#     for i, error_dict in enumerate(error_dicts):
#         if set(error_dict.keys()) != set(keys):
#             raise ValueError(
#                 f"Dataset {i} does not contain the same keys "
#                 "as the first dataset."
#             )
#
#     x = np.arange(len(keys))
#
#     # Width occupied by all boxplots in one elastic-constant group.
#     total_group_width = 0.70
#     box_width = total_group_width / n_cases
#
#     fig, ax = plt.subplots(figsize=(12, 7.2))
#
#     legend_handles = []
#
#     for i, error_dict in enumerate(error_dicts):
#         offset = (
#             i - (n_cases - 1) / 2
#         ) * box_width
#
#         positions = x + offset
#
#         data = []
#
#         for key in keys:
#             values = np.asarray(
#                 error_dict[key],
#                 dtype=float,
#             ).reshape(-1)
#
#             values = values[np.isfinite(values)]
#
#             if values.size == 0:
#                 raise ValueError(
#                     f"No valid values found for '{key}' "
#                     f"in dataset {i}."
#                 )
#
#             # Logarithmic axes cannot display zero.
#             if use_log_scale:
#                 values = values[values > 0]
#
#                 if values.size == 0:
#                     raise ValueError(
#                         f"No positive values found for '{key}' "
#                         f"in dataset {i}."
#                     )
#
#             data.append(values)
#
#         boxplot = ax.boxplot(
#             data,
#             positions=positions,
#             widths=box_width * 0.72,
#             patch_artist=True,
#             showmeans=show_mean,
#             showfliers=show_outliers,
#             whis=1.5,
#             medianprops={
#                 "color": "black",
#                 "linewidth": 1.4,
#             },
#             whiskerprops={
#                 "color": "black",
#                 "linewidth": 0.8,
#             },
#             capprops={
#                 "color": "black",
#                 "linewidth": 0.8,
#             },
#             meanprops={
#                 "marker": "D",
#                 "markerfacecolor": "white",
#                 "markeredgecolor": "black",
#                 "markersize": 4,
#             },
#             flierprops={
#                 "marker": "o",
#                 "markerfacecolor": "none",
#                 "markeredgecolor": colors[i],
#                 "markersize": 3.5,
#                 "alpha": 0.55,
#             },
#         )
#
#         for box in boxplot["boxes"]:
#             box.set_facecolor(colors[i])
#             box.set_edgecolor("black")
#             box.set_linewidth(0.7)
#             box.set_alpha(0.85)
#
#         legend_handles.append(boxplot["boxes"][0])
#
#     ax.set_xticks(x)
#     ax.set_xticklabels(keys, fontsize=10)
#
#     ax.set_xlabel(
#         "Effective elastic constant",
#         fontsize=11,
#     )
#
#     ax.set_ylabel(
#         "Absolute percentage error (%)",
#         fontsize=11,
#     )
#
#     if use_log_scale:
#         ax.set_yscale("log")
#
#     ax.tick_params(
#         axis="both",
#         which="major",
#         labelsize=10,
#         direction="out",
#         length=4,
#         width=0.8,
#     )
#
#     ax.grid(
#         axis="y",
#         linestyle="--",
#         linewidth=0.7,
#         alpha=0.35,
#         zorder=0,
#     )
#
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_linewidth(0.9)
#     ax.spines["bottom"].set_linewidth(0.9)
#
#     ax.legend(
#         legend_handles,
#         labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, 1.01),
#         ncol=n_cases,
#         frameon=False,
#         fontsize=10,
#         handlelength=1.8,
#         columnspacing=1.5,
#     )
#
#     # Small explanation of the statistical markers.
#     note = (
#         "box: interquartile range; line: median; "
#         "diamond: mean; whiskers: 1.5 × IQR"
#     )
#
#     if show_outliers:
#         note += "; circles: outliers"
#
#     ax.text(
#         0.995,
#         0.985,
#         note,
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=7.5,
#         color="0.30",
#     )
#
#     fig.tight_layout()
#
#     fig.savefig(
#         save_path,
#         format="svg",
#         bbox_inches="tight",
#     )
#
#     png_path = (
#         save_path[:-4] + ".png"
#         if save_path.lower().endswith(".svg")
#         else save_path + ".png"
#     )
#
#     fig.savefig(
#         png_path,
#         dpi=600,
#         bbox_inches="tight",
#     )
#
#     plt.close(fig)



#))))))))))))))))))))))))))))___________++++++++++++())_+_)++_)(-09=-0


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_error_violins_multi(
    error_dicts,
    labels=None,
    colors=None,
    save_path="error_violins.svg",
    show_mean=True,
    show_outliers=False,
    use_log_scale=False,
    max_error=100.0,
):
    """
    Plot grouped violin plots of prediction errors for multiple models.

    For each elastic constant, corresponding samples are removed from all
    datasets when at least one dataset has:

        - a non-finite value,
        - a value greater than max_error,
        - or, for logarithmic plots, a value less than or equal to zero.

    Parameters
    ----------
    error_dicts : list of dict
        Each dictionary contains arrays or lists of errors for each
        effective elastic constant. Corresponding arrays for a given key
        must have the same length in every dataset.

    labels : list of str, optional
        Labels used in the legend.

    colors : list of str, optional
        Colors assigned to the different datasets.

    save_path : str, optional
        Output path for the SVG figure.

    show_mean : bool, optional
        Show the arithmetic mean as a diamond marker.

    show_outliers : bool, optional
        Show observations outside the 1.5 IQR limits.

    use_log_scale : bool, optional
        Use a logarithmic y-axis. Values less than or equal to zero are
        removed from all corresponding datasets.

    max_error : float or None, optional
        Maximum permitted error. If any dataset has a value greater than
        this limit at a particular sample index, that index is removed
        from every dataset. Set to None to disable this filtering.
    """

    # ---------------------------------------------------------
    # Initial checks
    # ---------------------------------------------------------
    n_cases = len(error_dicts)

    if n_cases == 0:
        raise ValueError(
            "At least one error dictionary is required."
        )

    if labels is None:
        labels = [
            f"Model {i + 1}"
            for i in range(n_cases)
        ]

    if colors is None:
        default_colors = [
            "#D98C52",
            "#55A868",
            "#C44E52",
            "#4C72B0",
            "#8172B3",
            "#64B5CD",
        ]

        if n_cases > len(default_colors):
            raise ValueError(
                f"{n_cases} datasets were provided, but only "
                f"{len(default_colors)} default colors are available. "
                "Please provide a colors list."
            )

        colors = default_colors[:n_cases]

    if len(labels) != n_cases:
        raise ValueError(
            "Number of labels must equal number of datasets."
        )

    if len(colors) != n_cases:
        raise ValueError(
            "Number of colors must equal number of datasets."
        )

    if max_error is not None and max_error <= 0:
        raise ValueError(
            "max_error must be greater than zero or None."
        )

    keys = list(error_dicts[0].keys())

    if len(keys) == 0:
        raise ValueError(
            "The error dictionaries do not contain any data."
        )

    for i, error_dict in enumerate(error_dicts):
        if set(error_dict.keys()) != set(keys):
            raise ValueError(
                f"Dataset {i} does not contain the same keys "
                "as the first dataset."
            )

    # ---------------------------------------------------------
    # Paired filtering
    # ---------------------------------------------------------
    filtered_error_dicts = [
        {} for _ in range(n_cases)
    ]

    skipped_per_key = {}
    original_count_per_key = {}
    remaining_count_per_key = {}

    total_skipped_positions = 0
    total_removed_values = 0

    for key in keys:

        arrays = [
            np.asarray(
                error_dict[key],
                dtype=float,
            ).reshape(-1)
            for error_dict in error_dicts
        ]

        lengths = [
            values.size
            for values in arrays
        ]

        if len(set(lengths)) != 1:
            raise ValueError(
                f"All datasets must have the same number of "
                f"corresponding values for '{key}'. "
                f"Received lengths: {lengths}"
            )

        if lengths[0] == 0:
            raise ValueError(
                f"No values were provided for '{key}'."
            )

        # Shape:
        # rows    -> datasets
        # columns -> corresponding samples
        stacked_values = np.vstack(arrays)

        # Every dataset must contain a finite value at this index.
        valid_mask = np.all(
            np.isfinite(stacked_values),
            axis=0,
        )

        # If one dataset exceeds max_error, remove the corresponding
        # sample from every dataset.
        if max_error is not None:
            valid_mask &= np.all(
                stacked_values <= max_error,
                axis=0,
            )

        # Logarithmic axes cannot display zero or negative values.
        if use_log_scale:
            valid_mask &= np.all(
                stacked_values > 0,
                axis=0,
            )

        n_original = stacked_values.shape[1]
        n_remaining = np.count_nonzero(valid_mask)
        n_skipped = n_original - n_remaining

        original_count_per_key[key] = n_original
        remaining_count_per_key[key] = n_remaining
        skipped_per_key[key] = n_skipped

        total_skipped_positions += n_skipped
        total_removed_values += n_skipped * n_cases

        if n_remaining == 0:
            raise ValueError(
                f"No valid paired values remain for '{key}' "
                "after filtering."
            )

        for dataset_index in range(n_cases):
            filtered_error_dicts[dataset_index][key] = (
                stacked_values[dataset_index, valid_mask]
            )

    # Use the paired-filtered data from this point onward.
    error_dicts = filtered_error_dicts

    # ---------------------------------------------------------
    # Display filtering summary
    # ---------------------------------------------------------
    print("\nPaired error filtering summary")
    print("-" * 56)

    if max_error is not None:
        print(
            f"Maximum permitted error: {max_error:g}"
        )
    else:
        print(
            "Maximum-error filtering: disabled"
        )

    for key in keys:
        print(
            f"{key:>10}: "
            f"{skipped_per_key[key]:>5} skipped, "
            f"{remaining_count_per_key[key]:>5} retained "
            f"out of {original_count_per_key[key]}"
        )

    print("-" * 56)
    print(
        "Total corresponding sample positions skipped: "
        f"{total_skipped_positions}"
    )
    print(
        "Total individual values removed: "
        f"{total_removed_values}"
    )
    print()

    # ---------------------------------------------------------
    # Plot configuration
    # ---------------------------------------------------------
    x = np.arange(len(keys))

    # Width occupied by all violins in one elastic-constant group.
    total_group_width = 0.70
    violin_spacing = total_group_width / n_cases
    violin_width = violin_spacing * 0.78

    fig, ax = plt.subplots(
        figsize=(12, 7.2)
    )

    legend_handles = []

    # ---------------------------------------------------------
    # Plot every dataset
    # ---------------------------------------------------------
    for i, error_dict in enumerate(error_dicts):

        offset = (
            i - (n_cases - 1) / 2
        ) * violin_spacing

        positions = x + offset

        data = []

        for key in keys:
            values = np.asarray(
                error_dict[key],
                dtype=float,
            ).reshape(-1)

            data.append(values)

        violinplot = ax.violinplot(
            data,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=200,
            bw_method="scott",
        )

        for body in violinplot["bodies"]:
            body.set_facecolor(colors[i])
            body.set_edgecolor("black")
            body.set_linewidth(0.7)
            body.set_alpha(0.85)
            body.set_zorder(2)

        medians = np.array([
            np.median(values)
            for values in data
        ])

        quartile_1 = np.array([
            np.percentile(values, 25)
            for values in data
        ])

        quartile_3 = np.array([
            np.percentile(values, 75)
            for values in data
        ])

        means = np.array([
            np.mean(values)
            for values in data
        ])

        # Interquartile range shown as a thick vertical line.
        ax.vlines(
            positions,
            quartile_1,
            quartile_3,
            color="black",
            linewidth=3.0,
            zorder=4,
        )

        # Median shown as a white circular marker.
        ax.scatter(
            positions,
            medians,
            marker="o",
            s=20,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

        # Mean shown as a white diamond.
        if show_mean:
            ax.scatter(
                positions,
                means,
                marker="D",
                s=24,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )

            # Write the mean below the x-axis.
            for pos, mean in zip(
                positions,
                means,
            ):
                ax.text(
                    pos,
                    +0.023,
                    f"{mean:.1f}\\%",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                    clip_on=False,
                )

        if show_outliers:
            rng = np.random.default_rng(
                seed=42 + i
            )

            for position, values in zip(
                positions,
                data,
            ):
                q1 = np.percentile(
                    values,
                    25,
                )

                q3 = np.percentile(
                    values,
                    75,
                )

                iqr = q3 - q1

                lower_limit = q1 - 1.5 * iqr
                upper_limit = q3 + 1.5 * iqr

                outliers = values[
                    (values < lower_limit)
                    | (values > upper_limit)
                ]

                if outliers.size > 0:
                    # Small horizontal jitter prevents exact overlap.
                    jitter = rng.uniform(
                        -violin_width * 0.08,
                        violin_width * 0.08,
                        size=outliers.size,
                    )

                    ax.scatter(
                        position + jitter,
                        outliers,
                        marker="o",
                        s=10,
                        facecolor="none",
                        edgecolor=colors[i],
                        linewidth=0.7,
                        alpha=0.55,
                        zorder=3,
                    )

        legend_handles.append(
            Patch(
                facecolor=colors[i],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.85,
            )
        )

    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        keys,
        fontsize=10,
    )

    ax.set_xlabel(
        "Effective elastic constant",
        fontsize=11,
        labelpad=25 if show_mean else 8,
    )

    ax.set_ylabel(
        "Absolute percentage error (%)",
        fontsize=11,
    )

    if use_log_scale:
        ax.set_yscale("log")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.legend(
        legend_handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=n_cases,
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.5,
    )

    note = (
        "violin width: probability density; "
        "thick line: interquartile range; "
        "circle: median"
    )

    if show_mean:
        note += "; diamond: mean"

    if show_outliers:
        note += "; open circles: outliers"

    # if max_error is not None:
    #     note += f"; paired samples above {max_error:g}% removed"

    ax.text(
        0.995,
        0.985,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="0.30",
    )

    # Additional bottom space is needed for mean labels.
    if show_mean:
        fig.subplots_adjust(
            bottom=0.17
        )

    fig.tight_layout()

    # ---------------------------------------------------------
    # Save figures
    # ---------------------------------------------------------
    fig.savefig(
        save_path,
        format="svg",
        bbox_inches="tight",
    )

    png_path = (
        save_path[:-4] + ".png"
        if save_path.lower().endswith(".svg")
        else save_path + ".png"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

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
            # for i in ['DNS','IMN']: #-----------------------------------
            for i in ['IMN']:
                # for x in range(1, 7):
                load_file.write('echo Current time: %TIME%\n')
                for x in loads_cases:
                    load_file.write(f'feap86 -iI_val_{i}_stage_{stage}_{x}\n')
                    load_file.write('\necho Current time: %TIME%\n')


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





