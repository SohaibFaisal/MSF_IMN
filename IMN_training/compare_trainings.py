from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compare_trainings(
    trained_folder_ids,
    show=True,
    epochs=None,
    save_name="Training_comparisons",
    yscale="linear",
    marker_every=None,
    dpi=600,
):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "savefig.dpi": dpi,
    })
    # plt.rcParams["text.usetex"] = True
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    linestyles = ["-", "-", "-.", ":"]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    base_folder = Path("IMN_training")
    final_points = []

    for i, (name, folder_id) in enumerate(trained_folder_ids.items()):
        folder = base_folder / f"msf{int(folder_id):04d}"
        data = np.load(folder / "epoch_costs_1.npz")

        train = np.asarray(data["train"], dtype=float)

        if epochs is not None:
            train = train[:epochs]

        epoch_axis = np.arange(1, len(train) + 1)

        markevery = marker_every if marker_every is not None else max(1, len(train) // 12)

        line, = ax.plot(
            epoch_axis,
            train,
            label=name,
            marker=markers[i % len(markers)],
            markevery=markevery,
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.9,
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.7,
        )

        final_points.append((epoch_axis[-1], train[-1], name, line.get_color(), markers[i % len(markers)]))

    # Add final solid markers and labels
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max + 0.12 * (x_max - x_min))

    for x_final, y_final, name, color, marker in final_points:
        ax.plot(
            x_final,
            y_final,
            marker=marker,
            markersize=6.0,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="None",
            zorder=5,
        )

        ax.annotate(
            f"{100 * y_final:.2f}%",
            xy=(x_final, y_final),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_yscale(yscale)

    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)

    ax.tick_params(direction="in", length=4, width=0.8, top=True, right=True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.legend(frameon=False, loc="upper right", handlelength=2.8)

    fig.tight_layout()

    if show:
        plt.show()
        fig.savefig(f"{save_name}.png", bbox_inches="tight")
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_name}.svg", bbox_inches="tight")
    else:
        fig.savefig(f"{save_name}.png", bbox_inches="tight")
        fig.savefig(f"{save_name}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_name}.svg", bbox_inches="tight")

    plt.close(fig)