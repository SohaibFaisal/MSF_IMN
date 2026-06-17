import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams["svg.fonttype"] = "path"


# --- Overlap-friendly styles: dashed vs solid + sparse markers ---
MARK_EVERY = 2  # increase if you have many points, decrease if you have few

METHOD_STYLE = {
    "IMN": dict(linestyle="-",  linewidth=1.5, color='black'),
    "DNS": dict(linestyle="None",linewidth=1.0, color='red', marker="o", markersize=4.8, markevery=MARK_EVERY, markerfacecolor='None', markeredgecolor='red'),
}

# ---------------------------
# 2) Professional plot styling
# ---------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",  # clean and always available with matplotlib
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
})



COMPONENTS = [
    ("$\\varepsilon_{11}$", 0),
    ("$\\varepsilon_{22}$", 1),
    ("$\\varepsilon_{33}$", 2),
    ("$\\varepsilon_{12}$", 3),
    ("$\\varepsilon_{23}$", 4),
    ("$\\varepsilon_{31}$", 5),
]
# ---------------------------
# 1) Split flat arrays by load case (LC1..LC6)
# ---------------------------
def split_by_loadcase(stress_flat, strain_flat, n_loadcases=6):
    """
    stress_flat / strain_flat: flat list of rows; each row is [c0..c5] (6 comps)
    Assumes ordering is LC1 block, then LC2 block, ... then LC6 block.

    Returns: stress_lc, strain_lc (dict keyed by lc=1..n_loadcases)
    """
    if len(stress_flat) != len(strain_flat):
        raise ValueError(f"Stress/strain length mismatch: {len(stress_flat)} vs {len(strain_flat)}")

    total = len(stress_flat)
    if total % n_loadcases != 0:
        raise ValueError(f"Total rows ({total}) not divisible by n_loadcases ({n_loadcases}).")

    steps_per_lc = total // n_loadcases

    stress_lc, strain_lc = {}, {}
    for lc in range(1, n_loadcases + 1):
        i0 = (lc - 1) * steps_per_lc
        i1 = lc * steps_per_lc
        stress_lc[lc] = stress_flat[i0:i1]
        strain_lc[lc] = strain_flat[i0:i1]

    return stress_lc, strain_lc




def _plot_component(ax, strain_block, stress_block, comp_idx, label, style):
    xvals = [100*row[comp_idx] for row in strain_block]

    yvals = [row[comp_idx] for row in stress_block]
    ax.plot(xvals, yvals, label=label, **style)







# def _plot_component(ax, strain_block, stress_block, comp_idx, label, style):
#     xvals = [row[comp_idx] for row in strain_block]
#     yvals = [row[comp_idx] for row in stress_block]
#     ax.plot(xvals, yvals, label=label, **style)


def plot_loadcase_grid(lc, outpath, stress_imn_lc, strain_imn_lc, stress_dns_lc, strain_dns_lc):
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    fig.suptitle(f"Load Case {lc}", fontweight="semibold")

    for k, (title, comp_idx) in enumerate(COMPONENTS):
        ax = axs[k // 3, k % 3]

        _plot_component(ax, strain_imn_lc[lc], stress_imn_lc[lc], comp_idx, "IMN", METHOD_STYLE["IMN"])
        _plot_component(ax, strain_dns_lc[lc], stress_dns_lc[lc], comp_idx, "DNS", METHOD_STYLE["DNS"])

        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)

    # Shared labels (cleaner)
    fig.supxlabel("Strain, $\\varepsilon$ [-]")
    fig.supylabel("Stress, $\\sigma$ [MPa]")

    # One legend per figure (consistent placement)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.savefig(outpath, dpi=300,format='svg', bbox_inches="tight")
    plt.close(fig)


def plot_mixed_summary(outpath, stress_imn_lc, strain_imn_lc, stress_dns_lc, strain_dns_lc):
    """
    7th plot:
      eps_11 from LC1,
      eps_22 from LC2,
      eps_33 from LC3,
      eps_12 from LC4,
      eps_23 from LC5,
      eps_31 from LC6
    """
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    fig.suptitle(
        "Mixed Summary: $\\varepsilon_{11}@LC1,\\ \\varepsilon_{22}@LC2,\\ \\varepsilon_{33}@LC3,\\ "
        "\\varepsilon_{12}@LC4,\\ \\varepsilon_{23}@LC5,\\ \\varepsilon_{31}@LC6$",
        fontweight="semibold"
    )

    for k, (title, comp_idx) in enumerate(COMPONENTS):
        lc = k + 1  # 1..6
        ax = axs[k // 3, k % 3]

        _plot_component(ax, strain_imn_lc[lc], stress_imn_lc[lc], comp_idx, "IMN", METHOD_STYLE["IMN"])
        _plot_component(ax, strain_dns_lc[lc], stress_dns_lc[lc], comp_idx, "DNS", METHOD_STYLE["DNS"])

        ax.set_title(f"{title} (LC{lc})")
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.supxlabel("Strain, $\\varepsilon$ [-]")
    fig.supylabel("Stress, $\\sigma$ [units]")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.savefig(outpath, dpi=300,format='svg', bbox_inches="tight")
    plt.close(fig)




def plot_mixed_single_figs(out_dir, stress_imn_lc, strain_imn_lc, stress_dns_lc, strain_dns_lc):
    """
    Save 6 separate figures:
      fig1 -> eps_11 from LC1
      fig2 -> eps_22 from LC2
      fig3 -> eps_33 from LC3
      fig4 -> eps_12 from LC4
      fig5 -> eps_23 from LC5
      fig6 -> eps_31 from LC6
    """
    selected = [(0, 1), (1, 4)]  # (comp_idx, loadcase)
    for k, (title, comp_idx) in enumerate(COMPONENTS):
        lc = k + 1  # LC1..LC6

        fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
        # fig.suptitle(f"{title} from Load Case {lc}", fontweight="semibold")



        _plot_component(ax, strain_imn_lc[lc], stress_imn_lc[lc], comp_idx, "IMN", METHOD_STYLE["IMN"])
        _plot_component(ax, strain_dns_lc[lc], stress_dns_lc[lc], comp_idx, "DNS", METHOD_STYLE["DNS"])



        if lc == 1:
            ax.set_xlabel("Strain, $\\varepsilon_{xx}$ [%]")
            ax.set_ylabel("Stress, $\\sigma_{xx}$ [MPa]")
        elif lc == 4:
            ax.set_xlabel("Strain, $\\varepsilon_{xy}$ [%]")
            ax.set_ylabel("Stress, $\\sigma_{xy}$ [MPa]")
        else:
            ax.set_xlabel("Strain, $\\varepsilon$ [%]")
            ax.set_ylabel("Stress, $\\sigma$ [MPa]")

        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)
        ax.legend(frameon=False)

        fig.savefig(out_dir / f"Mixed_single_LC{lc}_{comp_idx}.svg", dpi=300,format='svg', bbox_inches="tight")
        plt.close(fig)


def plot_loadcase_error_grid(lc, outpath, stress_imn_lc, strain_imn_lc, stress_dns_lc, strain_dns_lc):
    """
    2x3 grid of stress error vs strain:
      x = strain (IMN strain, component-wise)
      y = (stress_IMN - stress_DNS) for that component

    Assumes same number of steps for IMN and DNS within each load case.
    """
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    fig.suptitle(f"Load Case {lc} — Stress Error (IMN − DNS)", fontweight="semibold")

    S_imn = stress_imn_lc[lc]
    E_imn = strain_imn_lc[lc]
    S_dns = stress_dns_lc[lc]

    n = min(len(S_imn), len(S_dns), len(E_imn))  # safety
    for k, (title, comp_idx) in enumerate(COMPONENTS):
        ax = axs[k // 3, k % 3]

        xvals = [E_imn[i][comp_idx] for i in range(n)]
        yvals = [S_imn[i][comp_idx] - S_dns[i][comp_idx] for i in range(n)]

        ax.plot(xvals, yvals, linewidth=1.7)
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)

        ax.set_title(title)
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.supxlabel("Strain, $\\varepsilon$ [-] (IMN)")
    fig.supylabel("Stress error, $\\Delta\\sigma$ = $\\sigma_{IMN}-\\sigma_{DNS}$ [units]")

    fig.savefig(outpath, dpi=300,format='svg', bbox_inches="tight")
    plt.close(fig)


def plot_mixed_error_summary(outpath, stress_imn_lc, strain_imn_lc, stress_dns_lc, strain_dns_lc):
    """
    Mixed error summary:
      eps_11 error from LC1,
      eps_22 error from LC2,
      eps_33 error from LC3,
      eps_12 error from LC4,
      eps_23 error from LC5,
      eps_31 error from LC6
    """
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    fig.suptitle("Mixed Summary — Stress Error (IMN − DNS)", fontweight="semibold")

    for k, (title, comp_idx) in enumerate(COMPONENTS):
        lc = k + 1
        ax = axs[k // 3, k % 3]

        S_imn = stress_imn_lc[lc]
        E_imn = strain_imn_lc[lc]
        S_dns = stress_dns_lc[lc]

        n = min(len(S_imn), len(S_dns), len(E_imn))

        xvals = [E_imn[i][comp_idx] for i in range(n)]
        yvals = [S_imn[i][comp_idx] - S_dns[i][comp_idx] for i in range(n)]

        ax.plot(xvals, yvals, linewidth=1.7)
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)

        ax.set_title(f"{title} (LC{lc})")
        ax.grid(True, alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useMathText=True)

    fig.supxlabel("Strain, $\\varepsilon$ [-] (IMN)")
    fig.supylabel("Stress error, $\\Delta\\sigma$ [units]")

    fig.savefig(outpath, dpi=300,format='svg', bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# 3) Save figures
# ---------------------------
def plot(new_folder, stress_normal, strain_normal, stress_IMN, strain_IMN):

    print(len(stress_normal))
    print(len(strain_normal))
    out_dir = new_folder / 'plots'
    # out_dir = Path(new_folder + '\\plots')
    out_dir.mkdir(exist_ok=True)
    datasets = {
        "stress_DNS": stress_normal,
        "strain_DNS": strain_normal,
        "stress_IMN": stress_IMN,
        "strain_IMN": strain_IMN,
    }

    for lc in range(6):
        for name, data in datasets.items():
            np.savez_compressed(
                out_dir / f"LC{lc + 1}_{name}.npz",
                data=np.asarray(data[lc])
            )





    stress_dns_lc, strain_dns_lc = split_by_loadcase(
            stress_normal, strain_normal, n_loadcases=6
        )

    stress_imn_lc, strain_imn_lc = split_by_loadcase(
        stress_IMN, strain_IMN, n_loadcases=6
    )

    # Optional sanity checks (highly recommended once)
    for lc in range(1,7):
        if len(stress_dns_lc[lc]) != len(stress_imn_lc[lc]):
            raise ValueError(f"Step count mismatch at LC{lc}: DNS={len(stress_dns_lc[lc])} vs IMN={len(stress_imn_lc[lc])}")
        if len(strain_dns_lc[lc]) != len(strain_imn_lc[lc]):
            raise ValueError(f"Step count mismatch at LC{lc}: DNS={len(strain_dns_lc[lc])} vs IMN={len(strain_imn_lc[lc])}")



    for lc in range(1,7):
        pass
        # plot_loadcase_grid(
        #     lc=lc,
        #     outpath=out_dir / f"Load_case{lc}.svg",
        #     stress_imn_lc=stress_imn_lc,
        #     strain_imn_lc=strain_imn_lc,
        #     stress_dns_lc=stress_dns_lc,
        #     strain_dns_lc=strain_dns_lc
        # )

        # NEW: error figure per load case
        # plot_loadcase_error_grid(
        #     lc=lc,
        #     outpath=out_dir / f"Load_case{lc}_error.svg",
        #     stress_imn_lc=stress_imn_lc,
        #     strain_imn_lc=strain_imn_lc,
        #     stress_dns_lc=stress_dns_lc,
        #     strain_dns_lc=strain_dns_lc
        # )

    # plot_mixed_summary(
    #     outpath=out_dir / "Load_case7_mixed.svg",
    #     stress_imn_lc=stress_imn_lc,
    #     strain_imn_lc=strain_imn_lc,
    #     stress_dns_lc=stress_dns_lc,
    #     strain_dns_lc=strain_dns_lc
    # )



    plot_mixed_single_figs(
        out_dir=out_dir,
        stress_imn_lc=stress_imn_lc,
        strain_imn_lc=strain_imn_lc,
        stress_dns_lc=stress_dns_lc,
        strain_dns_lc=strain_dns_lc
    )


    # NEW: mixed error plot
    # plot_mixed_error_summary(
    #     outpath=out_dir / "Load_case7_mixed_error.svg",
    #     stress_imn_lc=stress_imn_lc,
    #     strain_imn_lc=strain_imn_lc,
    #     stress_dns_lc=stress_dns_lc,
    #     strain_dns_lc=strain_dns_lc
    # )




