from __future__ import annotations

from collections import OrderedDict
from datetime import datetime as dt
from pathlib import Path
import os
import threading

import numpy as np
import optuna
import psutil
import torch
from torch_geometric.data import Data

from .IMN_calculator import IMNCalculator


lock = threading.Lock()
running = True
finish_optim = False
process = psutil.Process(os.getpid())
global_plot_data = {"X": [], "Train": [], "Val": []}


def mem(tag: str) -> None:
    rss_gb = process.memory_info().rss / (1024**3)
    print(f"[{tag}] RSS = {rss_gb:.2f} GB")


def load_graph_npz_2(path: str | Path, target_col: int = 9) -> Data:
    with np.load(str(path), allow_pickle=False) as d:
        x = torch.from_numpy(d["x"]).to(dtype=torch.float32)
        edge_index = torch.from_numpy(d["edge_index"]).to(dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        graph.batch = torch.zeros(x.size(0), dtype=torch.long)

        if "FVC" in d.files:
            graph.FVC = torch.from_numpy(d["FVC"]).to(dtype=torch.float32)

        if "target_mask" in d.files:
            graph.target_mask = torch.from_numpy(d["target_mask"]).to(dtype=torch.bool)
        elif x.size(1) > target_col:
            graph.target_mask = x[:, target_col] > 0.5
        else:
            graph.target_mask = torch.zeros(x.size(0), dtype=torch.bool)

    return graph


class GraphCPUCache:
    def __init__(self, max_graphs: int | None = 2000):
        self.max_graphs = max_graphs
        self._cache: OrderedDict[str, Data] = OrderedDict()

    def get(self, path: str | Path) -> Data:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        graph = load_graph_npz_2(key)
        self._cache[key] = graph
        if self.max_graphs is not None:
            while len(self._cache) > self.max_graphs:
                self._cache.popitem(last=False)
        return graph

    def __len__(self) -> int:
        return len(self._cache)


def _to_device(graph: Data, device: torch.device) -> Data:
    return graph.clone().to(device, non_blocking=True)


class IMNPhaseCountCache:
    def __init__(self, N_layers: int, nodes_per_mech_per_phase: int, device: torch.device, dtype: torch.dtype):
        self.N_layers = int(N_layers)
        self.nodes_per_mech_per_phase = int(nodes_per_mech_per_phase)
        self.device = device
        self.dtype = dtype
        self._cache: dict[int, IMNCalculator] = {}

    def get(self, phases) -> IMNCalculator:
        phase_list = list(phases)
        key = len(phase_list)
        if key not in self._cache:
            self._cache[key] = self._new_imn(phase_list)
        imn = self._cache[key]
        if hasattr(imn, "configure_phases"):
            imn.configure_phases(phase_list)
        else:
            imn.phases = phase_list
            imn.n_phases = len(phase_list)
            imn.node_phase = [phase_list[i % len(phase_list)] for i in range(imn.N)]
        return imn

    def _new_imn(self, phases) -> IMNCalculator:
        try:
            return IMNCalculator(self.N_layers, phases, self.nodes_per_mech_per_phase, self.device, dtype=self.dtype)
        except TypeError:
            return IMNCalculator(self.N_layers, phases, self.nodes_per_mech_per_phase, self.device)

    def __len__(self) -> int:
        return len(self._cache)


def _make_imn(phases, N_layers: int, nodes_per_mech_per_phase: int, device: torch.device, dtype: torch.dtype) -> IMNCalculator:
    try:
        return IMNCalculator(N_layers, list(phases), nodes_per_mech_per_phase, device, dtype=dtype)
    except TypeError:
        return IMNCalculator(N_layers, list(phases), nodes_per_mech_per_phase, device)


def _sample_graphs_to_device(sample: dict, mesh_folder: str | Path, graph_cache: GraphCPUCache, device: torch.device):
    ss, rr, mm = sample["ids"]
    folder = Path(mesh_folder)
    main_graph = _to_device(graph_cache.get(folder / f"graph_stage_{ss}_rve_{rr}_mesh_{mm}.npz"), device)
    phase_graphs = [
        _to_device(graph_cache.get(folder / f"graph_stage_{ss}_rve_{rr}_mesh_{mm}_target_{ph}.npz"), device)
        for ph in sample["Phases"]
    ]
    return main_graph, phase_graphs


def _compute_loss_for_sample(
    sid: str,
    training_data_set: dict,
    mesh_folder: str | Path,
    graph_cache: GraphCPUCache,
    model: torch.nn.Module,
    N_layers: int,
    nodes_per_mech_per_phase: int,
    device: torch.device,
    imn_dtype: torch.dtype,
    imn_cache: IMNPhaseCountCache | None,
):
    sample = training_data_set[sid]
    phases = sample["Phases"]
    main_graph, phase_graphs = _sample_graphs_to_device(sample, mesh_folder, graph_cache, device)

    flat_p = model.forward(phases, main_graph, phase_graphs)

    imn = imn_cache.get(phases) if imn_cache is not None else _make_imn(
        phases,
        N_layers,
        nodes_per_mech_per_phase,
        device,
        imn_dtype,
    )

    imn.assign_node_stiffness(sample)

    # Keep IMN homogenization in float32/float64.
    # torch.linalg.solve does not support Half on CUDA.
    with torch.amp.autocast(device_type=device.type, enabled=False):
        flat_p_imn = flat_p.float()
        C_pred = imn.homogenize_from_flat_params(flat_p_imn)

        C_tgt = sample["C_Target"].to(
            device=device,
            dtype=C_pred.dtype,
            non_blocking=True,
        )

        # Correct paper loss:
        # ||C_pred - C_tgt||_F / ||C_tgt||_F
        denom = torch.linalg.norm(C_tgt, ord="fro").clamp_min(torch.finfo(C_tgt.dtype).eps)
        loss = torch.linalg.norm(C_pred - C_tgt, ord="fro") / denom

    del main_graph, phase_graphs, flat_p, flat_p_imn, C_pred, C_tgt
    return loss

def _average_validation_loss(
    val_idx: torch.Tensor,
    training_data_set: dict,
    mesh_folder: str | Path,
    graph_cache: GraphCPUCache,
    model: torch.nn.Module,
    N_layers: int,
    nodes_per_mech_per_phase: int,
    device: torch.device,
    use_amp: bool,
    imn_dtype: torch.dtype,
    imn_cache: IMNPhaseCountCache | None,
):
    model.eval()
    loss_sum = 0.0
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.inference_mode():
        for idx in val_idx:
            sid = str(idx.item())
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = _compute_loss_for_sample(
                    sid,
                    training_data_set,
                    mesh_folder,
                    graph_cache,
                    model,
                    N_layers,
                    nodes_per_mech_per_phase,
                    device,
                    imn_dtype,
                    imn_cache,
                )
            loss_sum += float(loss.detach().cpu())
            del loss

    n_val = max(1, len(val_idx))
    return loss_sum / n_val

def _append_plot_values(epoch: int, avg_train: float, avg_val: float, plot_data: dict) -> None:
    with lock:
        plot_data["Train"].append(avg_train)
        plot_data["Val"].append(avg_val)
        global_plot_data["X"].append(epoch + 1)
        global_plot_data["Train"].append(avg_train)
        global_plot_data["Val"].append(avg_val)


def run_optimization(
    num_epochs,
    training_data_set,
    mesh_folder,
    inner_steps,
    optimizer,
    model,
    plot_data,
    cost_live_plot,
    imn_trained_data_folder,
    stage,
    N_layers,
    device,
    nodes_per_mech_per_phase=2,
    trial=None,
    accumulation_steps=5,
    val_every=2,
    graph_cache_size=2000,
    use_amp=False,
    force_float32=True,
    cache_imn_by_phase_count=True,
    imn_dtype="float32",
        samples_per_epoch =None
):
    global finish_optim
    finish_optim = False
    mem("start")

    device = torch.device(device)
    mesh_folder = Path(mesh_folder)
    imn_trained_data_folder = Path(imn_trained_data_folder)
    imn_torch_dtype = torch.float32 if str(imn_dtype).lower() in {"float32", "fp32", "torch.float32"} else torch.float64

    if force_float32:
        model.float()

    num_samples = len(training_data_set)
    val_ratio = 1 / 5
    generator = torch.Generator().manual_seed(123)
    all_idx = torch.randperm(num_samples, generator=generator)
    num_val = int(round(val_ratio * num_samples))
    val_idx = all_idx[:num_val]
    train_idx = all_idx[num_val:]
    accumulation_steps = max(1, int(accumulation_steps))
    val_every = max(1, int(val_every))

    for key in plot_data:
        plot_data[key].clear()
    with lock:
        for key in global_plot_data:
            global_plot_data[key].clear()

    print("CUDA available:", torch.cuda.is_available())
    print("Selected device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(device))
        print("AMP enabled:", bool(use_amp))
    print("Graph cache size:", graph_cache_size)
    print("IMN phase-count cache:", bool(cache_imn_by_phase_count))
    print("IMN dtype:", imn_torch_dtype)

    best_val = float("inf")
    best_epoch = -1
    last_val = float("nan")

    graph_cache = GraphCPUCache(max_graphs=graph_cache_size)
    imn_cache = IMNPhaseCountCache(N_layers, nodes_per_mech_per_phase, device, imn_torch_dtype) if cache_imn_by_phase_count else None
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
    amp_enabled = bool(use_amp and device.type == "cuda")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if samples_per_epoch is None:
            epoch_train_idx = train_idx
        else:
            n_epoch_samples = min(int(samples_per_epoch), len(train_idx))
            epoch_train_idx = train_idx[torch.randperm(len(train_idx))[:n_epoch_samples]]

        perm = epoch_train_idx[torch.randperm(len(epoch_train_idx))]
        train_loss_sum = 0.0
        accumulated = 0

        for it, idx in enumerate(perm):
            sid = str(idx.item())
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = _compute_loss_for_sample(
                    sid,
                    training_data_set,
                    mesh_folder,
                    graph_cache,
                    model,
                    N_layers,
                    nodes_per_mech_per_phase,
                    device,
                    imn_torch_dtype,
                    imn_cache,
                )
                scaled_loss = loss / accumulation_steps

            scaler.scale(scaled_loss).backward()
            accumulated += 1
            train_loss_sum += float(loss.detach().cpu())

            is_boundary = accumulated == accumulation_steps
            is_last = it == len(perm) - 1
            if is_boundary or is_last:
                if is_last and accumulated < accumulation_steps:
                    correction = accumulation_steps / accumulated
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0

            del loss, scaled_loss

        n_train = max(1, len(perm))
        avg_train = train_loss_sum / n_train
        run_validation = ((epoch + 1) % val_every == 0) or (epoch == num_epochs - 1)

        if run_validation:
            last_val = _average_validation_loss(
                val_idx,
                training_data_set,
                mesh_folder,
                graph_cache,
                model,
                N_layers,
                nodes_per_mech_per_phase,
                device,
                use_amp,
                imn_torch_dtype,
                imn_cache,
            )
            if last_val < best_val:
                best_val = last_val
                best_epoch = epoch
            if trial is not None:
                trial.report(last_val, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if trial is not None:
            trial.set_user_attr("epochs_ran", epoch + 1)
            trial.set_user_attr("best_epoch", best_epoch + 1 if best_epoch >= 0 else None)

        _append_plot_values(epoch, avg_train, last_val, plot_data)

        val_text = f"{last_val:.6f}" if np.isfinite(last_val) else "skipped"
        imn_cache_count = len(imn_cache) if imn_cache is not None else 0
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} "
            f"train={avg_train:.6f} val={val_text} "
            f"graph_cache={len(graph_cache)} imn_cache={imn_cache_count}"
        )

    imn_trained_data_folder.mkdir(exist_ok=True)
    np.savez(
        str(imn_trained_data_folder / f"epoch_costs_{stage}.npz"),
        train=np.array(plot_data["Train"], dtype=np.float32),
        val=np.array(plot_data["Val"], dtype=np.float32),
    )

    finish_optim = True
    return best_val


def run_live_optimization_GNN_IMN(
    num_epochs,
    num_samples,
    training_data_set,
    mesh_folder,
    inner_steps,
    optimizer,
    model,
    cost_live_plot,
    imn_trained_data_folder,
    stage,
    N_layers,
    device,
    nodes_per_mech_per_phase=2,
    trial=None,
    accumulation_steps=50,
    val_every=5,
    graph_cache_size=2000,
    use_amp=False,
    force_float32=True,
    cache_imn_by_phase_count=True,
    imn_dtype="float32",
        samples_per_epoch=None
):
    plot_data = {"Train": [], "Val": []}

    if not cost_live_plot:
        return run_optimization(
            num_epochs,
            training_data_set,
            mesh_folder,
            inner_steps,
            optimizer,
            model,
            plot_data,
            cost_live_plot,
            imn_trained_data_folder,
            stage,
            N_layers,
            device,
            nodes_per_mech_per_phase,
            trial,
            accumulation_steps,
            val_every,
            graph_cache_size,
            use_amp,
            force_float32,
            cache_imn_by_phase_count,
            imn_dtype,
            samples_per_epoch,
        )

    import pyqtgraph as pg
    import pyqtgraph.exporters as exporters
    from pyqtgraph.Qt import QtCore, QtGui

    global running, finish_optim
    running = True
    finish_optim = False

    app = pg.mkQApp("Live Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="Training Monitor")
    win.setBackground("w")
    hud = pg.LabelItem(justify="center")
    win.addItem(hud, row=0, col=0)
    plot = win.addPlot(row=1, col=0)
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setTitle(f"Training stage {stage}", color="#111111", size="14pt")
    plot.getAxis("left").enableAutoSIPrefix(False)

    title_font = QtGui.QFont("Aptos", 12)
    tick_font = QtGui.QFont("Aptos", 10)
    plot.titleLabel.item.setFont(title_font)
    plot.getAxis("left").setTickFont(tick_font)
    plot.getAxis("bottom").setTickFont(tick_font)

    axis_pen = pg.mkPen(color=(40, 40, 40), width=1)
    plot.getAxis("left").setPen(axis_pen)
    plot.getAxis("bottom").setPen(axis_pen)
    plot.getAxis("left").setTextPen(axis_pen)
    plot.getAxis("bottom").setTextPen(axis_pen)

    legend = plot.addLegend(offset=(10, 10))
    legend.setBrush(pg.mkBrush(255, 255, 255, 210))
    legend.setPen(pg.mkPen(120, 120, 120, 180))
    legend.setLabelTextColor((20, 20, 20))

    train_curve = plot.plot(pen=pg.mkPen(color=(0, 0, 255), width=2), name="Train")
    val_curve = plot.plot(pen=pg.mkPen(color=(0, 160, 0), width=2, style=QtCore.Qt.PenStyle.DashLine), name="Val")
    start = dt.now()

    def export_plot() -> None:
        try:
            folder = Path(imn_trained_data_folder)
            folder.mkdir(exist_ok=True)
            exporter = exporters.ImageExporter(plot)
            exporter.parameters()["width"] = 1600
            exporter.export(str(folder / f"cost_history_{stage}.png"))
        except Exception as exc:
            print("Failed to export plot:", exc)

    def refresh_plot() -> None:
        global finish_optim
        if finish_optim:
            export_plot()
            timer.stop()
            win.close()
            QtCore.QTimer.singleShot(0, app.quit)
            return

        with lock:
            n = min(len(global_plot_data["X"]), len(global_plot_data["Train"]), len(global_plot_data["Val"]))
            if n == 0:
                return
            x_values = np.asarray(global_plot_data["X"][-5000:], dtype=float)
            train_values = np.asarray(global_plot_data["Train"][-5000:], dtype=float)
            val_values = np.asarray(global_plot_data["Val"][-5000:], dtype=float)

        train_curve.setData(x_values, train_values)
        val_curve.setData(x_values, val_values)
        plot.setXRange(1, max(10, n), padding=0.02)

        elapsed = (dt.now() - start).total_seconds()
        sec_per_epoch = elapsed / max(1, n)
        eta_sec = max(0.0, sec_per_epoch * (num_epochs - n))
        last_val = val_values[-1]
        val_text = f"{last_val:.6f}" if np.isfinite(last_val) else "skipped"

        hud.setText(
            '<div style="background-color: rgba(255,255,255,220); color: #111111; '
            'padding: 8px 12px; border-radius: 10px; font-family: Aptos; font-size: 11pt; line-height: 1.3;">'
            f"<b>Progress</b>: {n}/{num_epochs} ({100.0 * n / max(1, num_epochs):.1f}%) | "
            f"<b>Train</b>: {train_values[-1]:.6f} | <b>Val</b>: {val_text} | "
            f"<b>Time/epoch</b>: {sec_per_epoch:.1f}s | "
            f"<b>ETA</b>: {int(eta_sec // 3600):02d}:{int((eta_sec // 60) % 60):02d}:{int(eta_sec % 60):02d}"
            "</div>"
        )

    worker = threading.Thread(
        target=run_optimization,
        args=(
            num_epochs,
            training_data_set,
            mesh_folder,
            inner_steps,
            optimizer,
            model,
            plot_data,
            cost_live_plot,
            imn_trained_data_folder,
            stage,
            N_layers,
            device,
            nodes_per_mech_per_phase,
            trial,
            accumulation_steps,
            val_every,
            graph_cache_size,
            use_amp,
            force_float32,
            cache_imn_by_phase_count,
            imn_dtype,
            samples_per_epoch,
        ),
        daemon=False,
    )

    def on_close() -> None:
        global running, finish_optim
        running = False
        finish_optim = True
        export_plot()
        worker.join(timeout=1)

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh_plot)
    timer.start(100)
    worker.start()
    app.aboutToQuit.connect(on_close)
    app.exec()
