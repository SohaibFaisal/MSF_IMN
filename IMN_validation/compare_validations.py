from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compare_validations(
    load_case,
    folders,
    show=True,
    save_folder=".",
    dpi=600,
):
    """
    Compare stress-strain responses of different IMN models against DNS.

    Parameters
    ----------
    load_case : iterable
        List of load case numbers.
    folders : dict
        Dictionary of {"Model Name": folder_id}.
    show : bool
        Display figures.
    save_folder : str or Path
        Output directory.
    dpi : int
        Figure resolution.
    """

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
    save_folder.mkdir(exist_ok=True)

    colors = {
        "DNS": "black",
        "IMN": "#4C72B0",
        "GNN": "#DD8452",
    }

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

    for lc in load_case:

        fig, ax = plt.subplots(figsize=(5.8, 4.0))

        dns_plotted = False
        marker_id = 0

        for name, folder_id in folders.items():

            validation_folder = (
                Path("IMN_validation")
                / f"Validation{int(folder_id):04d}"
                / "Val_stage_2_rve_0_mesh_0"
                / "plots"
            )



            # -------------------------------
            # Model prediction
            # -------------------------------
            stress = np.load(validation_folder / f"LC{lc}_stress_IMN.npz")["data"]
            strain = np.load(validation_folder / f"LC{lc}_strain_IMN.npz")["data"]

            if "IMN" in name:
                color = colors["IMN"]
            else:
                color = colors["GNN"]

            ax.plot(
                strain,
                stress,
                label=name,
                color=color,
                linestyle="-",
                # marker=markers[marker_id % len(markers)],
                # markerfacecolor="white",
                # markevery=max(len(strain) // 20, 1),
                # markersize=4,
                linewidth=2,
            )

            marker_id += 1


        for name, folder_id in folders.items():
            validation_folder = (
                    Path("IMN_validation")
                    / f"Validation{int(folder_id):04d}"
                    / "Val_stage_2_rve_0_mesh_0"
                    / "plots"
            )
            # -------------------------------
            # DNS reference
            # -------------------------------
            if not dns_plotted:
                dns_stress = np.load(validation_folder / f"LC{lc}_stress_DNS.npz")["data"]
                dns_strain = np.load(validation_folder / f"LC{lc}_strain_DNS.npz")["data"]

                ax.plot(
                    dns_strain,
                    dns_stress,
                    # color=colors["DNS"],
                    color='black',
                    linestyle="none",
                    marker='o',
                    markerfacecolor="none",
                    markevery=max(len(dns_strain) // 50, 1),
                    linewidth=0.5,
                    markersize=4,
                    label="DNS",
                )

                dns_plotted = True
                break



        # -------------------------------
        # Formatting
        # -------------------------------
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.set_title(f"Load Case {lc}")

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        ax.legend(
            frameon=True,
            edgecolor="black",
            fancybox=False,
            loc="best",
        )

        fig.tight_layout()

        fig.savefig(
            save_folder / f"LC{lc}_comparison.svg",
            bbox_inches="tight",
        )

        # fig.savefig(
        #     save_folder / f"LC{lc}_comparison.pdf",
        #     bbox_inches="tight",
        # )

        if show:
            plt.show()

        plt.close(fig)