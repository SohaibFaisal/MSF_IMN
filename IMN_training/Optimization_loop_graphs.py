x = []
y_train = []
y_val = []
yw_train = []
yw_val = []
yc_train = []
yc_val = []

global_plot_data = {'X': [],'Train':[], 'Val':[]}

from .IMN_calculator import normalized_frobenius_mse_graph
import optuna
import torch
import numpy as np
import threading
import pyqtgraph.exporters as exporters
lock = threading.Lock()
running = True
from datetime import datetime as dt
import pyqtgraph as pg
import os, psutil, time
from torch_geometric.data import Data, Batch
p = psutil.Process(os.getpid())

def mem(tag):
    rss = p.memory_info().rss / (1024**3)
    print(f"[{tag}] RSS = {rss:.2f} GB")


def load_graph_npz_2(path: str) -> Data:
    d = np.load(path)
    x = torch.tensor(d["x"], dtype=torch.float32)                 # (N, F)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long) # (2, E)
    g = Data(x=x, edge_index=edge_index)
    g.batch = torch.zeros(x.size(0), dtype=torch.long)
    if "FVC" in d.files:
        g.FVC = torch.tensor(d["FVC"], dtype=torch.float32)
    return g

def run_live_optimization_graphs(num_epochs, num_samples, training_data_set,mesh_folder, inner_steps, optimizer, model,  cost_live_plot,imn_trained_data_folder,stage, N_layers, trial=None):


    global running
    plot_data = {'Train':[], 'Val':[], 'C_train':[], 'C_val':[], 'W_train':[], 'W_val':[]}

    if cost_live_plot:
        global finish_optim
        finish_optim = False
        finished_handled = False

        def place_label_top_right():
            # Keep label in the visible top-right corner regardless of autoscale/pan
            vb = plot.getViewBox()
            (x0, x1), (y0, y1) = vb.viewRange()
            xr = x1 - x0
            yr = y1 - y0

        def refresh_plot():
            nonlocal finished_handled
            global finish_optim
            # ALWAYS check finish first
            if finish_optim:
                exporter = exporters.ImageExporter(plot)
                exporter.parameters()['width'] = 1600
                exporter.export(str(imn_trained_data_folder / f"cost_history_{stage}.png"))

                timer.stop()
                win.close()
                QtCore.QTimer.singleShot(0, app.quit)  # safer than direct app.quit()
                return

            try:
                current = dt.now()
                with lock:
                    # n = min(len(x), len(y_train), len(y_val))
                    # n = min(len(x), len(y_train), len(y_val), len(yw_train), len(yw_val))
                    n = min(len(a) for a in global_plot_data.values())

                    if n < 1:
                        return
                    N = 5000
                    pyqt_plot_data = {}
                    for k,v in global_plot_data.items():
                        pyqt_plot_data[k] = np.asarray(global_plot_data[k][max(0, n - N):n], dtype=float)

                    xs = np.asarray(x[max(0, n - N):n], dtype=float)
                    ys_tr = np.asarray(y_train[max(0, n - N):n], dtype=float)
                    ys_va = np.asarray(y_val[max(0, n - N):n], dtype=float)
                    wys_tr = np.asarray(yw_train[max(0, n - N):n], dtype=float)
                    wys_va = np.asarray(yw_val[max(0, n - N):n], dtype=float)


                if len(pyqt_plot_data['X']) == 0:
                    return

                for k, v in pyqt_plot_data.items():
                    if k == 'X':
                        continue
                    curve_plots[k].setData(pyqt_plot_data['X'], pyqt_plot_data[k])


                plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)
                last_tr = pyqt_plot_data['Train'][-1] if len(pyqt_plot_data['Train']) else float("nan")
                last_va = pyqt_plot_data['Val'][-1] if len(pyqt_plot_data['Val']) else float("nan")

                t = (current-start).total_seconds()
                hh = int(t/3600)
                mm = int(t/60)%60
                ss = t%60
                current_time_display = f"Time={hh:02d}:{mm:02d}:{ss:02.0f}"
                t = (current - start).total_seconds() / len(plot_data['Train'])
                hh = int(t / 3600)
                mm = int(t / 60) % 60
                ss = t % 60
                average_time_display = f"Time per epoch={hh:02d}:{mm:02d}:{ss:04.1f}"
                t = t*(num_epochs-1-len(plot_data['Train']))
                hh = int(t / 3600)
                mm = int(t / 60) % 60
                ss = t % 60
                remaining_time_display = f"Remaining time={hh:02d}:{mm:02d}:{int(ss):02.0f}"
                # ---- timing (guard against divide-by-zero) ----
                elapsed = (current - start).total_seconds()
                hh = int(elapsed / 3600)
                mm = int(elapsed / 60) % 60
                ss = elapsed % 60

                epochs_done = max(1, len(plot_data['Train']))
                sec_per_epoch = elapsed / epochs_done
                eta_sec = max(0.0, sec_per_epoch * (num_epochs - epochs_done))

                eh = int(eta_sec / 3600)
                em = int(eta_sec / 60) % 60
                es = int(eta_sec % 60)

                # ---- progress + costs ----
                progress = f"{n}/{num_epochs}  ({100.0 * n / max(1, num_epochs):.1f}%)"

                label_html = (
                    '<div style="background-color: rgba(255,255,255,220);'
                    'color: #111111;'
                    'padding: 8px 12px; border-radius: 10px;'
                    'font-family: Aptos; font-size: 11pt; line-height: 1.3;'
                    'box-shadow: 0 2px 6px rgba(0,0,0,60);">'
                    f'<b>Progress</b>: {progress}  |'
                    f'<b>Train</b>: {last_tr:.6f}  |'
                    f'<b>Val</b>: {last_va:.6f}   |'
                    f'<b>Elapsed</b>: {hh:02d}:{mm:02d}:{ss:02.0f}   |'
                    f'<b>Time/epoch</b>: {sec_per_epoch:6.1f} s   |'
                    f'<b>ETA</b>: {eh:02d}:{em:02d}:{es:02d}'
                    '</div>'
                )

                hud.setText(label_html)






                # plot.setTitle(f"Training")
                # --- X axis: grow smoothly, don't squash early ---
                xmin = 1
                xmax = max(10, n)  # keep at least 10 epochs visible at the beginning
                plot.setXRange(xmin, xmax, padding=0.02)

            except Exception as e:

                pass

        start = dt.now()
        app = pg.mkQApp("Live Plot")
        win = pg.GraphicsLayoutWidget(show=True, title="PyQtGraph Live Update")
        hud = pg.LabelItem(justify="center")
        hud.setText("Starting…")
        win.addItem(hud, row=0, col=0)

        plot = win.addPlot(row=1, col=0)
        hud.setText(
            '<div style="background-color: rgba(0,0,0,170);'
            'padding: 8px 10px; border-radius: 8px;'
            'font-family: Aptos; font-size: 11pt;'
            'line-height: 1.25; color: white;">Starting…</div>'
        )
        ##############

        from pyqtgraph.Qt import QtCore, QtGui

        # --- Professional global style ---
        pg.setConfigOptions(antialias=True)  # smooth lines

        # If you want a light theme:
        win.setBackground("w")


        # --- Axes + grid styling ---
        plot.showGrid(x=True, y=True, alpha=0.25)

        # Darker axes on white background
        axis_pen = pg.mkPen(color=(40, 40, 40), width=1)
        plot.getAxis('left').setPen(axis_pen)
        plot.getAxis('bottom').setPen(axis_pen)

        # Tick/text color
        plot.getAxis('left').setTextPen(axis_pen)
        plot.getAxis('bottom').setTextPen(axis_pen)



        # Title font
        title_font = QtGui.QFont("Aptos", 12)  # uses your preferred Aptos if installed
        plot.setTitle(f"Training stage {stage}", color="#111111", size="14pt")
        plot.titleLabel.item.setFont(title_font)

        # Optional: nicer axis tick font
        tick_font = QtGui.QFont("Aptos", 10)
        plot.getAxis('left').setTickFont(tick_font)
        plot.getAxis('bottom').setTickFont(tick_font)

        # --- Legend (robust across versions) ---
        legend = plot.addLegend(offset=(10, 10))  # keep it visible (top-left-ish)
        # Legend styling
        legend.setBrush(pg.mkBrush(255, 255, 255, 210))  # translucent white
        legend.setPen(pg.mkPen(120, 120, 120, 180))
        legend.setLabelTextColor((20, 20, 20))

        # --- Curves (train solid, val dashed) ---

        pens = {}
        pens['Train'] = pg.mkPen(color=(0, 0, 255), width=2)  # blue
        pens['Val'] = pg.mkPen(color=(0, 255, 0), width=2, style=QtCore.Qt.PenStyle.DashLine)  # blue


        curve_plots = {}
        for k,v in global_plot_data.items():
            if k == 'X':
                continue
            curve_plots[k] = plot.plot(pen=pens[k], name=k)

        plot.getAxis('left').enableAutoSIPrefix(False)



        from pyqtgraph.Qt import QtGui


        timer = QtCore.QTimer()
        timer.timeout.connect(refresh_plot)
        timer.start(30)  # ~33 FPS redraw
        thread = threading.Thread(target=run_optimization, args=(num_epochs, training_data_set, mesh_folder, inner_steps, optimizer, model, plot_data, cost_live_plot, imn_trained_data_folder,stage, N_layers,trial), daemon=False)
        thread.start()

        def on_close():
            global running, finish_optim
            running = False

            try:
                exporter = exporters.ImageExporter(plot)
                exporter.parameters()['width'] = 1600
                exporter.export(str(imn_trained_data_folder / f"cost_history_{stage}.png"))
                print("Plot exported on manual close.")
            except Exception as e:
                print("Failed to export plot on close:", e)

            finish_optim = True  # optional: tell refresh_plot to stop cleanly
            thread.join(timeout=1)

        app.aboutToQuit.connect(on_close)
        app.exec()
        return


    else:
        best_val = run_optimization(num_epochs, training_data_set, mesh_folder, inner_steps, optimizer, model, plot_data, cost_live_plot,imn_trained_data_folder,stage, N_layers,trial)
        return best_val


def run_optimization(num_epochs, training_data_set,mesh_folder,  inner_steps, optimizer, model, plot_data, cost_live_plot,imn_trained_data_folder,stage, N_layers, trial=None):
    if cost_live_plot:
        global running
        global finish_optim

    # ---- 1) split indices once (reproducible) ----
    mem("start")
    num_samples = len(training_data_set)
    val_ratio = 1/5
    seed = 123
    g = torch.Generator().manual_seed(seed)
    all_idx = torch.randperm(num_samples, generator=g)
    num_val = int(round(val_ratio * num_samples))
    val_idx = all_idx[:num_val]
    train_idx = all_idx[num_val:]
    batch_size = 1

    with lock:
        for l in plot_data.keys():
            plot_data[l].clear()

    finish_optim = False
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(num_epochs):
        graph_cache = {}
        # ---- 2) TRAIN ----
        perm = train_idx[torch.randperm(len(train_idx))]  # shuffle train each epoch
        Main_loss_sum = 0.0

        model.train()  # if you use dropout/bn etc.
        Main_loss_batch = torch.zeros(())

        for it, idx in enumerate(perm):

            sid = str(idx.item())
            ss, rr, mm = training_data_set[sid]['ids']
            phase_g = [load_graph_npz_2(str(mesh_folder / f'graph_stage_{ss}_rve_{rr}_mesh_{mm}_phase_{ph}.npz')) for ph in training_data_set[sid][f"Phases"]]
            material_in = {ph:torch.Tensor(training_data_set[sid][ph]) for ph in training_data_set[sid][f"Phases"]}
            C_pred = model.forward(training_data_set[sid]['Phases'], phase_g, material_in).reshape(6,6)
            C_tgt = training_data_set[sid]["C_Target"]
            Main_loss = normalized_frobenius_mse_graph(C_pred, C_tgt)
            Main_loss_batch += Main_loss
            Main_loss_sum += float(Main_loss.detach().cpu())
            if (it+1)%batch_size == 0:
                Main_loss_batch /= batch_size
                optimizer.zero_grad()
                Main_loss_batch.backward()
                optimizer.step()
                Main_loss_batch = torch.zeros(())
                graph_cache.clear()


        avg_train = Main_loss_sum / int(num_samples*(1-val_ratio)) #len(train_idx)
        plot_data['Train'].append(avg_train)

        # ---- 3) VALIDATION (no optimizer, no grads) ----
        model.eval()
        Main_loss_sum = 0.0
        with torch.no_grad():
            for it, idx in enumerate(val_idx):

                sid = str(idx.item())

                ss, rr, mm = training_data_set[sid]['ids']
                phase_g = [load_graph_npz_2(str(mesh_folder / f'graph_stage_{ss}_rve_{rr}_mesh_{mm}_phase_{ph}.npz')) for ph in training_data_set[sid][f"Phases"]]

                material_in = {ph: torch.Tensor(training_data_set[sid][ph]) for ph in training_data_set[sid][f"Phases"]}
                C_pred = model.forward(training_data_set[sid]['Phases'], phase_g, material_in).reshape(6, 6)
                C_tgt = training_data_set[sid]["C_Target"]
                Main_loss = normalized_frobenius_mse_graph(C_pred, C_tgt)
                Main_loss_sum += float(Main_loss.detach().cpu())


        avg_val = Main_loss_sum / int(num_samples*val_ratio)
        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch

        if trial is not None:
            trial.set_user_attr("epochs_ran", epoch + 1)

        if trial is not None:
            trial.report(avg_val, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()


        plot_data['Val'].append(avg_val)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} "
            f"train={avg_train:.6f}  val={avg_val:.6f}"
        )

        with lock:

            for k,v in global_plot_data.items():
                if k =='X':
                    continue
                global_plot_data[k].append(plot_data[k][-1])
            global_plot_data['X'].append(epoch+1)


    imn_trained_data_folder.mkdir(exist_ok=True)
    np.savez(
        str(imn_trained_data_folder / f"epoch_costs_{stage}.npz"),
        train=np.array(plot_data['Train']),
        val=np.array(plot_data['Val']),
    )
    finish_optim = True
    return best_val


