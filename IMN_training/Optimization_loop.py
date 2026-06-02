x = []
y_train = []
y_val = []


import torch
import numpy as np
import threading

import pyqtgraph.exporters as exporters
lock = threading.Lock()
running = True
from datetime import datetime as dt

import pyqtgraph as pg


def run_live_optimization_IMN(num_epochs, num_samples, dataset, inner_steps, optimizer, model, train_costs,val_costs,cost_live_plot,training_data_folder,stage):
    global running

    if cost_live_plot:
        global finish_optim
        finish_optim = False
        finished_handled = False

        def refresh_plot():
            nonlocal finished_handled
            try:
                current = dt.now()
                with lock:
                    n = min(len(x), len(y_train), len(y_val))
                    if n < 1:
                        return
                    N = 5000
                    xs = np.asarray(x[max(0, n - N):n], dtype=float)
                    ys_tr = np.asarray(y_train[max(0, n - N):n], dtype=float)
                    ys_va = np.asarray(y_val[max(0, n - N):n], dtype=float)

                if not np.isfinite(ys_tr).any() and not np.isfinite(ys_va).any():
                    return

                curve_train.setData(xs, ys_tr)
                curve_val.setData(xs, ys_va)

                plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

                # title: show last values safely
                last_tr = ys_tr[-1] if len(ys_tr) else float("nan")
                last_va = ys_va[-1] if len(ys_va) else float("nan")

                t = (current-start).total_seconds()
                hh = int(t/3600)
                mm = int(t/60)%60
                ss = t%60
                current_time_display = f"Time={hh:02d}:{mm:02d}:{ss:02.0f}"
                t = (current - start).total_seconds() / len(train_costs)
                hh = int(t / 3600)
                mm = int(t / 60) % 60
                ss = t % 60
                average_time_display = f"Time per epoch={hh:02d}:{mm:02d}:{ss:04.1f}"
                t = t*(num_epochs-1-len(train_costs))
                hh = int(t / 3600)
                mm = int(t / 60) % 60
                ss = t % 60
                remaining_time_display = f"Remaining time={hh:02d}:{mm:02d}:{int(ss):02.0f}"
                # ---- timing (guard against divide-by-zero) ----
                elapsed = (current - start).total_seconds()
                hh = int(elapsed / 3600)
                mm = int(elapsed / 60) % 60
                ss = elapsed % 60

                epochs_done = max(1, len(train_costs))
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
                xmin = 1
                xmax = max(10, n)  # keep at least 10 epochs visible at the beginning
                plot.setXRange(xmin, xmax, padding=0.02)


                if finish_optim:
                    exporter = exporters.ImageExporter(plot)
                    exporter.parameters()['width'] = 1600  # optional: higher resolution
                    exporter.export(training_data_folder + f"\\cost_history_{stage}.png")
                    timer.stop()  # stop refresh timer
                    win.close()  # close the window
                    app.quit()  # exit app.exec()
                    return

            except Exception as e:
                # Show errors instead of failing silently
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
        train_pen = pg.mkPen(color=(0, 120, 215), width=2)  # blue
        val_pen = pg.mkPen(color=(220, 50, 50), width=2, style=QtCore.Qt.PenStyle.DashLine)  # red dashed

        curve_train = plot.plot(pen=train_pen, name="Train")
        curve_val = plot.plot(pen=val_pen, name="Validation")

        plot.getAxis('left').enableAutoSIPrefix(False)



        from pyqtgraph.Qt import QtGui


        timer = QtCore.QTimer()
        timer.timeout.connect(refresh_plot)
        timer.start(30)  # ~33 FPS redraw
        thread = threading.Thread(target=run_optimization, args=(num_epochs, dataset, inner_steps, optimizer, model, train_costs, val_costs, cost_live_plot, training_data_folder,stage), daemon=False)
        thread.start()

        def on_close():
            global running
            running = False
            thread.join(timeout=1)

        app.aboutToQuit.connect(on_close)
        app.exec()
        return


    else:
        run_optimization(num_epochs, dataset, inner_steps, optimizer, model, train_costs,val_costs, cost_live_plot,training_data_folder,stage)
        return


def run_optimization(num_epochs, dataset, inner_steps, optimizer, model, train_costs,val_costs, cost_live_plot,training_data_folder,stage):
    if cost_live_plot:
        global running
        global finish_optim

    # ---- 1) split indices once (reproducible) ----
    num_samples = len(dataset)
    val_ratio = 0.2
    seed = 123

    g = torch.Generator().manual_seed(seed)
    all_idx = torch.randperm(num_samples, generator=g)

    num_val = int(round(val_ratio * num_samples))
    val_idx = all_idx[:num_val]
    train_idx = all_idx[num_val:]

    train_costs.clear()
    val_costs.clear()
    with lock:
        x.clear()
        y_train.clear()
        y_val.clear()

    finish_optim = False
    for epoch in range(num_epochs):
        if cost_live_plot and not running:
            print("Optimization interrupted by user. Saving partial cost history...")
            break
        # ---- 2) TRAIN ----
        perm = train_idx[torch.randperm(len(train_idx))]  # shuffle train each epoch
        running_train = 0.0

        model.train()  # if you use dropout/bn etc.
        for idx in perm:
            data = dataset[str(int(idx))]
            model.assign_node_stiffness(data)
            C_pred = model()

            for _ in range(inner_steps):
                optimizer.zero_grad()
                loss = model.normalized_frobenius_mse(C_pred, data['C_Target'])
                loss.backward()
                optimizer.step()

            running_train += loss.item()

        avg_train = running_train / len(train_idx)
        train_costs.append(avg_train)

        # ---- 3) VALIDATION (no optimizer, no grads) ----
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for idx in val_idx:
                data = dataset[str(int(idx))]
                model.assign_node_stiffness(data)
                C_pred = model()
                vloss = model.normalized_frobenius_mse(C_pred, data['C_Target'])
                running_val += vloss.item()

        avg_val = running_val / len(val_idx)
        val_costs.append(avg_val)



        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} "
            f"train={avg_train:.6f}  val={avg_val:.6f}"
        )
        with lock:
            x.append(epoch + 1)  # epoch number starting at 1
            y_train.append(avg_train)
            y_val.append(avg_val)

    np.savez(
        training_data_folder / f"epoch_costs_{stage}.npz",
        train=np.array(train_costs),
        val=np.array(val_costs),
    )
    finish_optim = True

