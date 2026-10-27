from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Put this script in the parent directory containing these subdirectories.
base_dir = Path(__file__).resolve().parent
subdirs = ['Training_data_generation\\Training_data' + f for f in ["1004", "1005"]]
filename = "material_dictionary.npz"  # File to read in each subdirectory.


UD_1004 = []
UD_1005 = []
UDs = [UD_1004, UD_1005]

Matrix_1004 = []
Matrix_1005 = []
Matrixs = [Matrix_1004, Matrix_1005]

if __name__ == "__main__":
    for subdir, ud, mat in zip(subdirs, UDs, Matrixs):
        file_path = base_dir / subdir / filename

        if not file_path.is_file():
            print(f"Skipping missing file: {file_path}")
            continue

        with np.load(file_path, allow_pickle=True) as data:
            print(f"\nReading: {file_path}")
            print("Available arrays:", data.files)

            # Access arrays by their saved names, for example:
            # stiffness = data["C"]
            # Add your processing here, inside the 'with' block.
            for key in data.files:
                array = data[key]
                ud.append(array.item()['UD1'])
                mat.append(array.item()['MATRIX'])
                # print(array.item()['UD1'])
                # break
            # break

print(np.shape(UDs))

UD1004_EX = []
UD1005_EX = []

Matrix1004_EX = []
Matrix1005_EX = []

for s in UDs[0]:
    Ss = np.linalg.inv(np.asarray(s, dtype=float))
    UD1004_EX.append(1.0 / Ss[0, 0])

for s in UDs[1]:
    Ss = np.linalg.inv(np.asarray(s, dtype=float))
    UD1005_EX.append(1.0 / Ss[0, 0])

for s in Matrixs[0]:
    Ss = np.linalg.inv(np.asarray(s, dtype=float))
    Matrix1004_EX.append(1.0 / Ss[0, 0])

for s in Matrixs[1]:
    Ss = np.linalg.inv(np.asarray(s, dtype=float))
    Matrix1005_EX.append(1.0 / Ss[0, 0])



plt.scatter(Matrix1004_EX, UD1004_EX)
plt.show()

plt.scatter(Matrix1005_EX, UD1005_EX)
plt.show()

# ---------- Academic formatting and SVG export ----------
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "svg.fonttype": "none",       # Keep text editable in the SVG
})

for label, x_values, y_values in [
    ("1004", Matrix1004_EX, UD1004_EX),
    ("1005", Matrix1005_EX, UD1005_EX),
]:
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")

    ax.scatter(
        x_values, y_values,
        s=18,
        color="#2166AC",
        edgecolors="none",
        alpha=0.75,
        zorder=3,
    )

    ax.set_xlabel(r"Matrix Young's modulus, $E_{\mathrm{x}}$")
    ax.set_ylabel(r"Fiber Young's modulus, $E_{\mathrm{x}}$")

    ax.tick_params(
        which="both", direction="in", top=True, right=True,
        width=0.8,
    )
    ax.tick_params(which="major", length=4)
    ax.tick_params(which="minor", length=2)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(0.05)

    output_path = base_dir / f"Youngs_modulus_{label}.svg"
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)
