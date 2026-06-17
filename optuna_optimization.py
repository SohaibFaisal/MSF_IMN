import optuna
import time
import datetime
from pathlib import Path
import csv
from optuna.trial import TrialState

from IMN_training.GNN_IMN import train_GNN_IMN

viz_dir = Path(r"IMN_training/optuna")
viz_dir.mkdir(parents=True, exist_ok=True)
LOG_FILE = viz_dir / 'log.txt'
CSV_FILE = viz_dir / "trials_summary.csv"
def objective(trial):

    # ----- keep tuning runs cheap -----
    num_samples = 90
    num_epochs = 50

    training_dataset_folder = Path(r"Training_data_generation/Training_data0924")  # your folder
    out_dir = Path(r"IMN_training/optuna") / f"trial_{trial.number:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    status = "COMPLETED"
    best_val = None

    try:

        # ----------------------------
        # Hyperparameter suggestions
        # ----------------------------

        N_layers = trial.suggest_int("N_layers", 3, 5)

        nodes_per_mech_per_phase = trial.suggest_int(
            "nodes_per_mech_per_phase", 1, 3
        )

        tnn_hidden_dim = trial.suggest_categorical(
            "tnn_hidden_dim", [32, 64, 128]
        )

        gnn_hidden_dim = trial.suggest_categorical(
            "gnn_hidden_dim", [32, 64, 128]
        )

        x_feat = trial.suggest_categorical(
            "x_feat", [32, 64, 128, 256]
        )

        tnn_layers = trial.suggest_int(
            "tnn_layers", 1, 3
        )

        gnn_layers = trial.suggest_int(
            "gnn_layers", 1, 3
        )

        gnn_structure = trial.suggest_int(
            "gnn_structure", 1, 3
        )

        lr = trial.suggest_float("lr", 8e-5, 8e-3, log=True)
        gnn_heads = trial.suggest_categorical("gnn_heads", [4, 8, 16])
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

        optimizing_variables = [
            tnn_hidden_dim,
            gnn_hidden_dim,
            gnn_heads,
            x_feat,
            gnn_structure,
            nodes_per_mech_per_phase,
            tnn_layers,
            gnn_layers,
        ]

        # ----------------------------
        # Training
        # ----------------------------

        best_val = train_GNN_IMN(
            N_layers=N_layers,
            num_samples=num_samples,
            num_epochs=num_epochs,
            lr_rest=lr,
            live_plot=False,
            training_dataset_folder=training_dataset_folder,
            imn_trained_data_folder=out_dir,
            optimizing_variables=optimizing_variables,
            weight_decay=weight_decay,
            trial=trial,
            use_GPU=True,
        )

    except optuna.TrialPruned:
        status = "PRUNED"
        raise

    finally:

        runtime = time.perf_counter() - start_time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Save to trial so it is available from the study DB later
        trial.set_user_attr("runtime_sec", runtime)
        # trial.set_user_attr("runtime_ms", int(round(runtime * 1000)))
        trial.set_user_attr("status", status)
        trial.set_user_attr("best_val_loss", best_val)
        # ----------------------------
        # Write log line
        # ----------------------------

        with open(LOG_FILE, "a") as f:

            f.write("====================================\n")
            f.write(f"timestamp : {timestamp}\n")
            f.write(f"trial     : {trial.number}\n")
            f.write(f"status    : {status}\n")

            f.write("parameters:\n")
            for k, v in trial.params.items():
                f.write(f"  {k} : {v}\n")

            f.write(f"best_val_loss : {best_val}\n")
            f.write(f"runtime_sec   : {runtime:.2f}\n")
            f.write(f"epochs_ran     : {trial.user_attrs.get('epochs_ran', None)}\n")
            f.write("====================================\n\n")

    return best_val


def format_for_paper(fig, title=None, width=1400, height=900):
    fig.update_layout(
        template="simple_white",
        width=width,
        height=height,
        title=title,
        font=dict(family="Arial", size=14),
        margin=dict(l=120, r=40, t=80, b=160),  # <-- big bottom margin fixes cut-off
        legend=dict(font=dict(size=18)),
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Objective Value",
            thickness=25,
            outlinewidth=0
        )
    )

    # Make axis labels readable
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=16),
        title_font=dict(size=20),
        automargin=True
    )
    fig.update_yaxes(
        tickfont=dict(size=16),
        title_font=dict(size=20),
        automargin=True
    )

    return fig


def make_left_labels_horizontal(fig, x_shift=-0.06, font_size=14):
    # For each yaxis that has a title, replace it with an annotation
    for k in list(fig.layout):
        if not str(k).startswith("yaxis"):
            continue

        ax = fig.layout[k]
        title = getattr(ax, "title", None)
        if title is None or not getattr(title, "text", None):
            continue

        txt = ax.title.text
        ax.title.text = ""  # remove axis title to avoid duplicates

        # yaxis domain gives the vertical span of that subplot row
        dom = getattr(ax, "domain", None)
        if not dom:
            continue

        y_mid = 0.5 * (dom[0] + dom[1])

        fig.add_annotation(
            x=0 + x_shift,
            y=y_mid,
            xref="paper",
            yref="paper",
            text=txt,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            textangle=0,
            font=dict(size=font_size),
        )
    return fig

def export_trials_csv(study, csv_path: Path):
    # Collect all parameter keys across trials so CSV has consistent columns
    all_param_keys = set()
    for t in study.trials:
        all_param_keys.update(t.params.keys())
    all_param_keys = sorted(all_param_keys)

    fieldnames = (
            ["trial_number", "state", "pruned", "value", "best_val_loss", "runtime_sec", "epochs_ran"]
            + all_param_keys
    )

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()

        for t in study.trials:
            state = t.state.name
            pruned = (t.state == TrialState.PRUNED)

            runtime_sec = t.user_attrs.get("runtime_sec", None)

            row = {
                "trial_number": t.number,
                "state": state,
                "pruned": pruned,
                "value": t.value,
                "best_val_loss": t.user_attrs.get("best_val_loss", None),
                "runtime_sec": int(runtime_sec) if runtime_sec is not None else None,
                "epochs_ran": t.user_attrs.get("epochs_ran", None),
            }

            for k in all_param_keys:
                row[k] = t.params.get(k, None)

            w.writerow(row)

if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=8)
    storage = "sqlite:///optuna_gnn_imn.db"  # creates a local file
    study_name = "gnn_imn_v1"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,  # lets you resume later
        pruner=pruner
    )
    # study.optimize(objective, n_trials=10)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # CSV








    # Visualization
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
        plot_parallel_coordinate,
        plot_contour
    )

    study = optuna.load_study(
        study_name="gnn_imn_v1",
        storage="sqlite:///optuna_gnn_imn.db"
    )
    # Optimization history
    fig = plot_optimization_history(study)
    fig = format_for_paper(fig)
    fig.update_traces(marker=dict(size=5))
    fig.update_traces(line=dict(width=2))
    fig.update_layout(title=None)
    fig.write_image(viz_dir / "opt_history.pdf")
    fig.write_image(viz_dir / "opt_history.svg", scale=3)

    # Hyperparameter importance
    fig = plot_param_importances(study)
    fig = format_for_paper(fig)
    fig.update_layout(title=None)
    fig.write_image(viz_dir / "param_importance.pdf")
    fig.write_image(viz_dir / "param_importance.svg", scale=3)

    # Slice plot
    fig = plot_slice(study)
    fig = format_for_paper(fig)
    fig.update_traces(
        marker=dict(
            colorscale="Viridis"
        )
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Objective Value",
            thickness=20
        )
    )
    fig.update_layout(title=None)
    fig.write_image(viz_dir / "slice_plot.pdf")
    fig.write_image(viz_dir / "slice_plot.svg", scale=3)

    # Parallel coordinate
    fig = plot_parallel_coordinate(study)
    fig = format_for_paper(fig)
    fig.update_traces(
        line=dict(
            colorscale="Viridis",
            showscale=True
        ),
        unselected=dict(
            line=dict(
                opacity=1.0
            )
        )
    )
    fig.update_layout(
        title=None,
        width=1800,
        height=900
    )
    # fig.update_layout(
    #     coloraxis_colorbar=dict(
    #         title="Objective Value",
    #         thickness=20
    #     )
    # )
    fig.write_image(viz_dir / "parallel_coord.pdf")
    fig.write_image(viz_dir / "parallel_coord.svg", scale=3)

    # Contour plot
    fig = plot_contour(study, params=[ "N_layers","x_feat","gnn_structure", "tnn_hidden_dim", "gnn_hidden_dim"])
    # fig = plot_contour(study)
    for t in fig.data:
        if t.type == "contour":
            t.colorscale = "Viridis"
            # t.contours = dict(showlines=False) # Remove black lines


    fig = make_left_labels_horizontal(fig, x_shift=-0.06, font_size=14)
    fig = format_for_paper(fig)
    fig.update_layout(title=None)
    # # Make left-side parameter labels horizontal
    # fig.update_layout(margin=dict(l=150, r=40, t=80, b=160))
    # fig.update_xaxes(
    #     tickfont=dict(size=14),
    #     title_font=dict(size=14),
    #     automargin=True
    # )
    # fig.update_layout(
    #     coloraxis_colorbar=dict(
    #         title="Objective Value",
    #         thickness=20
    #     )
    # )
    fig.write_image(viz_dir / "contour.pdf")
    fig.write_image(viz_dir / "contour.svg", scale=3)

    from optuna.visualization import plot_edf

    fig = plot_edf(study)
    fig = format_for_paper(fig)
    fig.update_layout(title=None)
    fig.write_image(viz_dir / "edf.pdf")
    fig.write_image(viz_dir / "edf.svg", scale=3)


    export_trials_csv(study, CSV_FILE)
    print("Wrote CSV:", CSV_FILE)